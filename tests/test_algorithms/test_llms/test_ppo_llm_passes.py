# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLM PPO actor and critic passes, fused and split.

Pure CPU on tiny real models in fp32. The reference for each pass pair is one
forward over the doubled ``[actor rows; critic rows]`` batch with one backward
of the summed loss.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.arena.memory import estimate_training
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed.fsdp import FSDPConfig
from agilerl.lora.fused import (
    ROUTING_STATE,
    get_cached_lora_layers,
    set_fused_adapter_routing,
    unset_fused_adapter_routing,
)
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import attention_mask_from_padded_ids
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from tests.test_algorithms.test_llms.llm_helpers import (
    DummyConfig,
    DummyHiddenStatesModel,
    optimizer_state,
    record_calls,
    record_outputs,
    scale_losses,
    trainable_weights,
)
from tests.test_algorithms.test_llms.segment_helpers import (
    lora_weights,
    record_step_gradients,
    seed_lora_weights,
    use_fake_liger_policy_loss,
)
from tests.test_algorithms.test_llms.test_ppo_llm_vision import (
    IMAGE_COUNTS,
    IMAGE_TOKEN_ID,
    make_vision_ppo,
    vision_episodes,
)

PAD_TOKEN_ID = 63
VOCAB = 64
SEQ_LEN = 6
# fp32 on CPU; the doubled forward only regroups the same per-row matmuls.
RTOL = 1e-5
ATOL = 1e-6


class CheckpointedHiddenStatesModel(DummyHiddenStatesModel):
    """Dummy causal LM that recomputes its LoRA-targeted layer in backward."""

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        hidden = checkpoint(
            self.linear_1, self.embed(input_ids.long()), use_reentrant=False
        )
        return CausalLMOutputWithPast(
            logits=self.lm_head(hidden), hidden_states=(hidden,)
        )


def make_ppo(
    model_cls: type[DummyHiddenStatesModel] = DummyHiddenStatesModel,
    **overrides: Any,
) -> LLMPPO:
    """Tiny fp32 PPO on CPU whose actor and critic adapters start different."""
    torch.manual_seed(0)
    # Decoder geometry fields let the memory estimator read the config.
    config = DummyConfig(
        input_size=4,
        max_tokens=4,
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
    )
    config.intermediate_size = 64
    kwargs: dict[str, Any] = {
        "actor_network": AutoModelForCausalLMWithValueHead(
            model_cls(config=config, device="cpu")
        ),
        "pad_token_id": PAD_TOKEN_ID,
        "pad_token": "<pad>",
        "batch_size": 2,
        "micro_batch_size_per_gpu": 2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "update_epochs": 1,
        "wrap": False,
        "gradient_checkpointing": False,
        "calc_position_embeddings": False,
        "device": "cpu",
        "use_liger_loss": False,
        "lora_config": LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
            modules_to_save=["summary"],
        ),
        **overrides,
    }
    agent = LLMPPO(**kwargs)
    with torch.no_grad():
        for name, param in agent.actor.named_parameters():
            if "lora_B" in name:
                param.normal_()
    return agent


def micro_batch(rows: int = 2) -> dict[str, torch.Tensor]:
    """Learn-loop tensors of one micro-batch with two turns per row."""
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, PAD_TOKEN_ID - 2, (rows, SEQ_LEN), generator=generator)
    ids[0, -1] = PAD_TOKEN_ID
    mask = torch.ones(rows, SEQ_LEN - 1)
    mask[0, -1] = 0.0
    turn_ids = torch.zeros(rows, SEQ_LEN - 1, dtype=torch.long)
    turn_ids[:, 2:] = 1
    turn_ids[mask == 0] = -1
    return {
        "ids": ids,
        "mask": mask,
        "old_log_probs": -torch.rand(rows, SEQ_LEN - 1, generator=generator) - 3.0,
        "reference_log_probs": -torch.rand(rows, SEQ_LEN - 1, generator=generator)
        - 3.0,
        "advantages": torch.randn(rows, SEQ_LEN - 1, generator=generator),
        "returns": torch.randn(rows, SEQ_LEN - 1, generator=generator),
        "old_values": torch.randn(rows, SEQ_LEN - 1, generator=generator),
        "turn_ids": turn_ids,
    }


def lora_grads(agent: LLMPPO) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in agent.actor.named_parameters()
        if param.grad is not None
    }


def value_loss(
    agent: LLMPPO, values: torch.Tensor, batch: dict[str, torch.Tensor], mode: str
) -> torch.Tensor:
    return agent._ppo_value_loss(
        values,
        batch["old_values"],
        batch["returns"],
        batch["mask"],
        batch["turn_ids"],
        2,
        mode,
    )


def policy_loss(
    agent: LLMPPO, log_probs: torch.Tensor, batch: dict[str, torch.Tensor], mode: str
) -> torch.Tensor:
    loss, _ = agent._ppo_policy_loss(
        log_probs,
        batch["mask"],
        batch["old_log_probs"],
        batch["reference_log_probs"],
        batch["advantages"],
        batch["turn_ids"],
        2,
        mode,
    )
    return loss


def doubled_forward(
    agent: LLMPPO,
    ids: torch.Tensor,
    pixel_values: torch.Tensor | None,
    image_counts: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Actor log-probs and critic values of one ``[actor rows; critic rows]`` forward."""
    rows = ids.shape[0]
    mask = attention_mask_from_padded_ids(ids, agent.pad_token_id)
    log_probs, values = agent._fused_model_pass(
        ids.repeat(2, 1),
        mask.repeat(2, 1),
        ["actor"] * rows + ["critic"] * rows,
        pixel_values=None if pixel_values is None else pixel_values.repeat(2, 1),
        pixel_image_counts=None if image_counts is None else image_counts * 2,
    )
    return log_probs[:rows], values[rows:]


def vision_micro_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor, list[int]]:
    """Two vision episodes as one micro-batch, plus their vision rows."""
    token_ids, *_, pixel_values = vision_episodes()
    batch = micro_batch()
    batch["ids"] = torch.cat(token_ids[:2])[:, :SEQ_LEN]
    # Each kept placeholder takes the next vision row of its episode.
    counts = [int((row == IMAGE_TOKEN_ID).sum()) for row in batch["ids"]]
    rows = []
    offset = 0
    for count, total in zip(counts, IMAGE_COUNTS[:2], strict=True):
        rows.append(pixel_values[offset : offset + count])
        offset += total
    return batch, torch.cat(rows), counts


class TestPPOCriticValues:
    @pytest.mark.parametrize("use_separate_reference_adapter", [True, False])
    def test_value_loss_trains_only_the_critic_adapter(
        self, use_separate_reference_adapter: bool
    ) -> None:
        # Arrange
        agent = make_ppo(use_separate_reference_adapter=use_separate_reference_adapter)
        batch = micro_batch()

        # Act
        value_loss(agent, agent._critic_values(batch["ids"]), batch, "token").backward()
        unset_fused_adapter_routing(agent.actor)

        # Assert
        grads = lora_grads(agent)
        critic = [
            g for name, g in grads.items() if "lora_" in name and "critic" in name
        ]
        actor = [g for name, g in grads.items() if "lora_" in name and "actor" in name]
        assert critic
        assert any(grad.abs().sum() > 0 for grad in critic)
        assert all(torch.count_nonzero(grad) == 0 for grad in actor)

    def test_gradient_checkpointing_leaves_the_critic_gradient_unchanged(
        self,
    ) -> None:
        # Arrange
        eager = make_ppo()
        recomputed = make_ppo(model_cls=CheckpointedHiddenStatesModel)
        batch = micro_batch()

        # Act
        for agent in (eager, recomputed):
            values = agent._critic_values(batch["ids"])
            value_loss(agent, values, batch, "token").backward()
            unset_fused_adapter_routing(agent.actor)

        # Assert
        eager_grads = lora_grads(eager)
        recomputed_grads = lora_grads(recomputed)
        assert eager_grads.keys() == recomputed_grads.keys()
        assert any(
            grad.abs().sum() > 0
            for name, grad in eager_grads.items()
            if "critic" in name
        )
        for name, grad in eager_grads.items():
            # Same fp32 ops in the same order; recompute is bitwise.
            assert torch.equal(recomputed_grads[name], grad), name


class TestPPOSplitPassesMatchDoubledForward:
    @pytest.mark.parametrize("mode", ["token", "turn"])
    @pytest.mark.parametrize("vision", [False, True], ids=["text", "vision"])
    @pytest.mark.parametrize(
        "model_cls",
        [DummyHiddenStatesModel, CheckpointedHiddenStatesModel],
        ids=["eager", "checkpointed"],
    )
    def test_losses_and_gradients_match(
        self, mode: str, vision: bool, model_cls: type[DummyHiddenStatesModel]
    ) -> None:
        # Arrange
        if vision:
            if model_cls is CheckpointedHiddenStatesModel:
                pytest.skip("vision model has no checkpointed variant")
            reference = make_vision_ppo()
            split = make_vision_ppo()
            batch, pixel_values, counts = vision_micro_batch()
        else:
            reference = make_ppo(model_cls)
            split = make_ppo(model_cls)
            batch, pixel_values, counts = micro_batch(), None, None

        # Act
        log_probs, values = doubled_forward(
            reference, batch["ids"], pixel_values, counts
        )
        reference_loss = policy_loss(reference, log_probs, batch, mode) + value_loss(
            reference, values, batch, mode
        )
        reference_loss.backward()
        unset_fused_adapter_routing(reference.actor)

        actor_loss = policy_loss(
            split,
            split._fused_forward(batch["ids"], pixel_values=pixel_values),
            batch,
            mode,
        )
        actor_loss.backward()
        unset_fused_adapter_routing(split.actor)
        critic_loss = value_loss(
            split, split._critic_values(batch["ids"], pixel_values), batch, mode
        )
        critic_loss.backward()
        unset_fused_adapter_routing(split.actor)

        # Assert
        torch.testing.assert_close(
            actor_loss + critic_loss, reference_loss, rtol=RTOL, atol=ATOL
        )
        reference_grads = lora_grads(reference)
        split_grads = lora_grads(split)
        assert split_grads.keys() == reference_grads.keys()
        assert any("critic" in name for name in split_grads)
        assert any("actor" in name for name in split_grads)
        for name, grad in reference_grads.items():
            torch.testing.assert_close(
                split_grads[name], grad, rtol=RTOL, atol=ATOL, msg=name
            )

    def test_liger_policy_pass_matches_doubled_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        def hidden_policy_loss(
            policy_hidden: torch.Tensor,
            _head_w: torch.Tensor,
            _head_b: torch.Tensor | None,
            _target_ids: torch.Tensor,
            mask: torch.Tensor,
            *_args: Any,
            **_kwargs: Any,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            loss = (policy_hidden.pow(2).sum(-1) * mask).sum() / mask.sum()
            return loss, tuple(loss.detach() for _ in range(5))

        reference = make_ppo()
        split = make_ppo()
        use_fake_liger_policy_loss(split, monkeypatch, "agilerl.algorithms.ppo_llm")
        monkeypatch.setattr(
            "agilerl.algorithms.ppo_llm.apply_fused_policy_loss", hidden_policy_loss
        )
        batch = micro_batch()
        ids = batch["ids"]
        rows = ids.shape[0]

        # Act
        set_fused_adapter_routing(reference.actor, ["actor"] * rows + ["critic"] * rows)
        with reference._patch_lm_head_to_identity():
            hidden, _, values = reference.actor(
                input_ids=ids.repeat(2, 1),
                attention_mask=attention_mask_from_padded_ids(ids, PAD_TOKEN_ID).repeat(
                    2, 1
                ),
            )
        reference_policy, _ = hidden_policy_loss(
            hidden[:rows, :-1], None, None, None, batch["mask"]
        )
        reference_loss = reference_policy + value_loss(
            reference, values[rows:, :-1], batch, "token"
        )
        reference_loss.backward()
        unset_fused_adapter_routing(reference.actor)

        actor_loss, metrics = split._ppo_policy_loss_liger(
            ids,
            batch["mask"],
            batch["old_log_probs"],
            batch["reference_log_probs"],
            batch["advantages"],
            batch["turn_ids"],
            2,
            "token",
        )
        actor_loss.backward()
        unset_fused_adapter_routing(split.actor)
        critic_loss = value_loss(split, split._critic_values(ids), batch, "token")
        critic_loss.backward()
        unset_fused_adapter_routing(split.actor)

        # Assert
        assert set(metrics) == {
            "kl",
            "clipfrac",
            "pg_loss",
            "entropy",
            "kl_clamp_frac",
        }
        torch.testing.assert_close(
            actor_loss + critic_loss, reference_loss, rtol=RTOL, atol=ATOL
        )
        reference_grads = lora_grads(reference)
        split_grads = lora_grads(split)
        assert split_grads.keys() == reference_grads.keys()
        for name, grad in reference_grads.items():
            torch.testing.assert_close(
                split_grads[name], grad, rtol=RTOL, atol=ATOL, msg=name
            )


class TestPPOFusedPassMatchesDoubledForward:
    @pytest.mark.parametrize("mode", ["token", "turn"])
    @pytest.mark.parametrize("vision", [False, True], ids=["text", "vision"])
    @pytest.mark.parametrize(
        "model_cls",
        [DummyHiddenStatesModel, CheckpointedHiddenStatesModel],
        ids=["eager", "checkpointed"],
    )
    def test_losses_and_gradients_match(
        self, mode: str, vision: bool, model_cls: type[DummyHiddenStatesModel]
    ) -> None:
        # Arrange
        if vision:
            if model_cls is CheckpointedHiddenStatesModel:
                pytest.skip("vision model has no checkpointed variant")
            reference = make_vision_ppo()
            fused = make_vision_ppo()
            batch, pixel_values, counts = vision_micro_batch()
        else:
            reference = make_ppo(model_cls)
            fused = make_ppo(model_cls)
            batch, pixel_values, counts = micro_batch(), None, None

        # Act
        log_probs, values = doubled_forward(
            reference, batch["ids"], pixel_values, counts
        )
        reference_loss = policy_loss(reference, log_probs, batch, mode) + value_loss(
            reference, values, batch, mode
        )
        reference_loss.backward()
        unset_fused_adapter_routing(reference.actor)

        hidden, fused_values = fused._actor_critic_hidden_states(
            batch["ids"], pixel_values
        )
        fused_loss = policy_loss(
            fused, fused._actor_log_probs_from_hidden(hidden, batch["ids"]), batch, mode
        ) + value_loss(fused, fused_values, batch, mode)
        fused_loss.backward()
        unset_fused_adapter_routing(fused.actor)

        # Assert
        torch.testing.assert_close(fused_loss, reference_loss, rtol=RTOL, atol=ATOL)
        reference_grads = lora_grads(reference)
        fused_grads = lora_grads(fused)
        assert fused_grads.keys() == reference_grads.keys()
        assert any("critic" in name for name in fused_grads)
        assert any("actor" in name for name in fused_grads)
        for name, grad in reference_grads.items():
            torch.testing.assert_close(
                fused_grads[name], grad, rtol=RTOL, atol=ATOL, msg=name
            )

    def test_blockmask_packing_matches_the_padded_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: the dummy model is per-token, so packing changes no real
        # token's hidden state.
        padded = make_ppo()
        packed = make_ppo()
        monkeypatch.setattr(packed, "_packing_mode", lambda: "blockmask")
        forwards = record_grad_forwards(packed)
        batch = micro_batch()
        real_tokens = int((batch["ids"] != PAD_TOKEN_ID).sum())

        # Act
        losses = []
        for agent in (padded, packed):
            hidden, values = agent._actor_critic_hidden_states(batch["ids"])
            loss = policy_loss(
                agent,
                agent._actor_log_probs_from_hidden(hidden, batch["ids"]),
                batch,
                "turn",
            ) + value_loss(agent, values, batch, "turn")
            loss.backward()
            unset_fused_adapter_routing(agent.actor)
            losses.append(loss)

        # Assert: one packed row per adapter, no pad tokens.
        assert [f["input_ids"].shape for f in forwards] == [(2, real_tokens)]
        assert forwards[0]["adapters"] == {"actor", "critic"}
        torch.testing.assert_close(losses[1], losses[0], rtol=RTOL, atol=ATOL)
        padded_grads = lora_grads(padded)
        packed_grads = lora_grads(packed)
        assert packed_grads.keys() == padded_grads.keys()
        for name, grad in padded_grads.items():
            torch.testing.assert_close(
                packed_grads[name], grad, rtol=RTOL, atol=ATOL, msg=name
            )

    def test_liger_policy_loss_matches_doubled_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        def hidden_policy_loss(
            policy_hidden: torch.Tensor,
            _head_w: torch.Tensor,
            _head_b: torch.Tensor | None,
            _target_ids: torch.Tensor,
            mask: torch.Tensor,
            *_args: Any,
            **_kwargs: Any,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            loss = (policy_hidden.pow(2).sum(-1) * mask).sum() / mask.sum()
            return loss, tuple(loss.detach() for _ in range(5))

        reference = make_ppo()
        fused = make_ppo()
        use_fake_liger_policy_loss(fused, monkeypatch, "agilerl.algorithms.ppo_llm")
        monkeypatch.setattr(
            "agilerl.algorithms.ppo_llm.apply_fused_policy_loss", hidden_policy_loss
        )
        batch = micro_batch()
        ids = batch["ids"]
        rows = ids.shape[0]

        # Act
        set_fused_adapter_routing(reference.actor, ["actor"] * rows + ["critic"] * rows)
        with reference._patch_lm_head_to_identity():
            hidden, _, values = reference.actor(
                input_ids=ids.repeat(2, 1),
                attention_mask=attention_mask_from_padded_ids(ids, PAD_TOKEN_ID).repeat(
                    2, 1
                ),
            )
        reference_policy, _ = hidden_policy_loss(
            hidden[:rows, :-1], None, None, None, batch["mask"]
        )
        reference_loss = reference_policy + value_loss(
            reference, values[rows:, :-1], batch, "token"
        )
        reference_loss.backward()
        unset_fused_adapter_routing(reference.actor)

        fused_hidden, fused_values = fused._actor_critic_hidden_states(ids)
        fused_policy, metrics = fused._ppo_policy_loss_liger_from_hidden(
            fused_hidden,
            ids,
            batch["mask"],
            batch["old_log_probs"],
            batch["reference_log_probs"],
            batch["advantages"],
            batch["turn_ids"],
            2,
            "token",
        )
        fused_loss = fused_policy + value_loss(fused, fused_values, batch, "token")
        fused_loss.backward()
        unset_fused_adapter_routing(fused.actor)

        # Assert
        assert set(metrics) == {
            "kl",
            "clipfrac",
            "pg_loss",
            "entropy",
            "kl_clamp_frac",
        }
        torch.testing.assert_close(fused_loss, reference_loss, rtol=RTOL, atol=ATOL)
        reference_grads = lora_grads(reference)
        fused_grads = lora_grads(fused)
        assert fused_grads.keys() == reference_grads.keys()
        for name, grad in reference_grads.items():
            torch.testing.assert_close(
                fused_grads[name], grad, rtol=RTOL, atol=ATOL, msg=name
            )


def learn_rows(agent: LLMPPO, rows: slice = slice(0, 4)) -> dict[str, float]:
    """One learn over ``rows`` of four single-turn trajectories with equal action counts."""
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(0, PAD_TOKEN_ID, (4, SEQ_LEN + 2), generator=generator)[rows]
    mask = torch.zeros(4, SEQ_LEN + 1, dtype=torch.bool)[rows]
    mask[:, 3:] = True
    rewards = torch.tensor([1.0, 0.0, 0.5, -1.0])[rows]
    return agent.learn((list(ids.split(1)), list(mask.split(1)), rewards))


def adapter_lora_names(agent: LLMPPO) -> set[str]:
    """Names of every actor and critic LoRA parameter."""
    return {
        name
        for name, _ in agent.actor.named_parameters()
        if "lora_" in name and ("actor" in name or "critic" in name)
    }


def record_runtime_backwards(
    agent: LLMPPO, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[frozenset[str], bool]]:
    """Routed adapters of every runtime backward, and whether it stepped the optimizer."""
    calls: list[tuple[frozenset[str], bool]] = []
    lora_layer = get_cached_lora_layers(agent.actor)[0]
    backward = agent.shard_runtime.backward

    def recording_backward(*args: Any, **kwargs: Any) -> Any:
        adapters = frozenset(ROUTING_STATE[lora_layer])
        step = backward(*args, **kwargs)
        calls.append((adapters, step is not None))
        return step

    monkeypatch.setattr(agent.shard_runtime, "backward", recording_backward)
    return calls


def fsdp_learn_worker(
    rank: int,
    world_size: int,
    port: int,
    liger: bool,
    fuse: bool,
    queue: Any,
) -> None:
    """One FSDP2 rank: learn on its half of :func:`learn_rows` and report step grads."""
    try:
        os.environ.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
            }
        )
        # The default FSDP2 mesh follows the host accelerator (MPS on macOS);
        # these ranks train on CPU.
        torch._C._get_accelerator = lambda: torch.device("cpu")
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=2,
            micro_batch_size_per_gpu=1,
            whiten_advantages=False,
            wrap=True,
            fsdp_config=FSDPConfig(optim_cpu_offload=False, param_dtype="float32"),
            fuse_actor_critic_pass=fuse,
        )
        seed_lora_weights(agent)
        monkeypatch = pytest.MonkeyPatch()
        if liger:
            use_fake_liger_policy_loss(agent, monkeypatch, "agilerl.algorithms.ppo_llm")
        steps = record_step_gradients(agent, monkeypatch)
        learn_rows(agent, slice(2 * rank, 2 * rank + 2))
        # NumPy arrays: a torch tensor in the queue shares memory that dies
        # with this process before the parent reads it.
        step_arrays = [
            {name: grad.numpy() for name, grad in step.items()} for step in steps
        ]
        queue.put((rank, "ok", (adapter_lora_names(agent), step_arrays)))
    except Exception:
        queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def spawn_fsdp_learn(liger: bool, fuse: bool, world_size: int = 2) -> list[Any]:
    """Run :func:`fsdp_learn_worker` on ``world_size`` gloo ranks; return each rank's report."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=fsdp_learn_worker,
            args=(rank, world_size, port, liger, fuse, queue),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results = sorted(queue.get(timeout=300) for _ in range(world_size))
    for proc in procs:
        proc.join(timeout=300)
    for rank, status, payload in results:
        assert status == "ok", f"rank {rank}: {payload}"
    return [payload for _, _, payload in results]


def record_grad_forwards(agent: LLMPPO) -> list[dict[str, Any]]:
    """Rows, adapters and vision rows of every gradient-bearing actor forward from now on."""
    forwards: list[dict[str, Any]] = []
    lora_layer = get_cached_lora_layers(agent.actor)[0]

    def record(
        _module: torch.nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if not torch.is_grad_enabled():
            return
        pixel_values = kwargs.get("pixel_values")
        forwards.append(
            {
                "input_ids": kwargs["input_ids"],
                "adapters": set(ROUTING_STATE[lora_layer]),
                "pixel_rows": None if pixel_values is None else pixel_values.shape[0],
            }
        )

    agent.actor.register_forward_pre_hook(record, with_kwargs=True)
    return forwards


FUSE_MODES = pytest.mark.parametrize("fuse", [False, True], ids=["split", "fused"])


class TestPPOLearnPasses:
    @pytest.mark.parametrize("liger", [False, True], ids=["standard", "liger"])
    @FUSE_MODES
    def test_every_gradient_forward_runs_one_micro_batch(
        self, monkeypatch: pytest.MonkeyPatch, liger: bool, fuse: bool
    ) -> None:
        # Arrange
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            fuse_actor_critic_pass=fuse,
        )
        if liger:
            use_fake_liger_policy_loss(agent, monkeypatch, "agilerl.algorithms.ppo_llm")
        forwards = record_grad_forwards(agent)

        # Act
        learn_rows(agent)

        # Assert: split runs an actor and a critic forward of
        # micro_batch_size_per_gpu rows; fused runs both in one forward.
        if fuse:
            assert [f["adapters"] for f in forwards] == [{"actor", "critic"}] * 2
            assert [f["input_ids"].shape[0] for f in forwards] == [4, 4]
        else:
            assert [f["adapters"] for f in forwards] == [{"actor"}, {"critic"}] * 2
            assert [f["input_ids"].shape[0] for f in forwards] == [2, 2, 2, 2]

    @pytest.mark.parametrize("liger", [False, True], ids=["standard", "liger"])
    @FUSE_MODES
    def test_activation_offload_covers_every_gradient_forward(
        self, monkeypatch: pytest.MonkeyPatch, liger: bool, fuse: bool
    ) -> None:
        # Arrange
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            fuse_actor_critic_pass=fuse,
            activation_offload=True,
        )
        if liger:
            use_fake_liger_policy_loss(agent, monkeypatch, "agilerl.algorithms.ppo_llm")
        save_on_cpu = torch.autograd.graph.save_on_cpu
        offload_depth = [0]

        @contextmanager
        def counting_save_on_cpu(**kwargs: Any) -> Iterator[None]:
            offload_depth[0] += 1
            try:
                with save_on_cpu(**kwargs):
                    yield
            finally:
                offload_depth[0] -= 1

        monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", counting_save_on_cpu)
        depths: list[int] = []
        agent.actor.register_forward_pre_hook(
            lambda *_: (
                depths.append(offload_depth[0]) if torch.is_grad_enabled() else None
            )
        )

        # Act
        learn_rows(agent)

        # Assert
        assert depths == [1] * (2 if fuse else 4)

    @pytest.mark.parametrize("liger", [False, True], ids=["standard", "liger"])
    @FUSE_MODES
    def test_accumulated_micro_batches_match_one_micro_batch(
        self, monkeypatch: pytest.MonkeyPatch, liger: bool, fuse: bool
    ) -> None:
        # Arrange: equal action counts per row, so the mean over one 4-row
        # micro-batch equals the mean of four 1-row micro-batch means.
        accumulated = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
            fuse_actor_critic_pass=fuse,
        )
        single = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=4,
            fuse_actor_critic_pass=fuse,
        )
        assert accumulated.gradient_accumulation_steps == 4
        assert single.gradient_accumulation_steps == 1
        if liger:
            for agent in (accumulated, single):
                use_fake_liger_policy_loss(
                    agent, monkeypatch, "agilerl.algorithms.ppo_llm"
                )
        accumulated_steps = record_step_gradients(accumulated, monkeypatch)
        single_steps = record_step_gradients(single, monkeypatch)

        # Act
        accumulated_metrics = learn_rows(accumulated)
        single_metrics = learn_rows(single)

        # Assert
        assert len(accumulated_steps) == 1
        assert len(single_steps) == 1
        assert accumulated_steps[0].keys() == single_steps[0].keys()
        assert any("critic" in name for name in single_steps[0])
        for name, grad in single_steps[0].items():
            torch.testing.assert_close(
                accumulated_steps[0][name], grad, rtol=RTOL, atol=ATOL, msg=name
            )
        # The fake Liger kernel reports the action-token count as ``kl``.
        for key in (
            ("loss", "pg_loss", "vf_loss")
            if liger
            else ("loss", "pg_loss", "vf_loss", "kl")
        ):
            assert accumulated_metrics[key] == pytest.approx(
                single_metrics[key], rel=1e-5, abs=1e-6
            ), key

    @FUSE_MODES
    def test_vision_passes_each_see_only_their_rows_images(self, fuse: bool) -> None:
        # Arrange
        agent = make_vision_ppo(micro_batch_size_per_gpu=2, fuse_actor_critic_pass=fuse)
        forwards = record_grad_forwards(agent)
        token_ids, action_masks, rewards, turn_ids, pixel_values = vision_episodes()

        # Act
        agent.learn(
            (token_ids, action_masks, rewards),
            turn_ids=turn_ids,
            pixel_values=pixel_values,
            pixel_image_counts=IMAGE_COUNTS,
        )

        # Assert: each forward runs a micro-batch of rows per routed adapter,
        # with one vision row per placeholder of those rows.
        if fuse:
            assert [f["adapters"] for f in forwards] == [{"actor", "critic"}] * 2
        else:
            assert [f["adapters"] for f in forwards] == [{"actor"}, {"critic"}] * 2
        for forward in forwards:
            assert forward["input_ids"].shape[0] == 2 * len(forward["adapters"])
            assert forward["pixel_rows"] == int(
                (forward["input_ids"] == IMAGE_TOKEN_ID).sum()
            )

    @pytest.mark.parametrize("liger", [False, True], ids=["standard", "liger"])
    @FUSE_MODES
    def test_optimizer_steps_after_the_last_pass_of_each_window(
        self, monkeypatch: pytest.MonkeyPatch, liger: bool, fuse: bool
    ) -> None:
        # Arrange: two optimizer steps of two micro-batches each.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=2,
            micro_batch_size_per_gpu=1,
            fuse_actor_critic_pass=fuse,
        )
        assert agent.gradient_accumulation_steps == 2
        if liger:
            use_fake_liger_policy_loss(agent, monkeypatch, "agilerl.algorithms.ppo_llm")
        backwards = record_runtime_backwards(agent, monkeypatch)
        steps = record_step_gradients(agent, monkeypatch)

        # Act
        learn_rows(agent)

        # Assert
        actor = frozenset({"actor"})
        critic = frozenset({"critic"})
        both = actor | critic
        window = (
            [(both, False), (both, True)]
            if fuse
            else [(actor, False), (critic, False), (actor, False), (critic, True)]
        )
        assert backwards == window * 2
        assert len(steps) == 2
        for step in steps:
            assert step.keys() == adapter_lora_names(agent)


class TestPPOFusedLearnMatchesSplitLearn:
    @pytest.mark.parametrize("liger", [False, True], ids=["standard", "liger"])
    @pytest.mark.parametrize(
        "model_cls",
        [DummyHiddenStatesModel, CheckpointedHiddenStatesModel],
        ids=["eager", "checkpointed"],
    )
    # Unclipped norms are about 5.3 (actor) and 7.8 (critic).
    @pytest.mark.parametrize(
        "max_grad_norm", [1.0, 6.0], ids=["both_clipped", "critic_clipped"]
    )
    def test_metrics_gradients_and_weights_match(
        self,
        monkeypatch: pytest.MonkeyPatch,
        liger: bool,
        model_cls: type[DummyHiddenStatesModel],
        max_grad_norm: float,
    ) -> None:
        # Arrange: same seed, so both agents start from the same weights.
        agents = {
            fuse: make_ppo(
                model_cls,
                batch_size=4,
                mini_batch_size=4,
                micro_batch_size_per_gpu=2,
                max_grad_norm=max_grad_norm,
                fuse_actor_critic_pass=fuse,
            )
            for fuse in (True, False)
        }
        if liger:
            for agent in agents.values():
                use_fake_liger_policy_loss(
                    agent, monkeypatch, "agilerl.algorithms.ppo_llm"
                )
        steps = {
            fuse: record_step_gradients(agent, monkeypatch)
            for fuse, agent in agents.items()
        }

        # Act
        metrics = {fuse: learn_rows(agent) for fuse, agent in agents.items()}

        # Assert
        assert len(steps[True]) == len(steps[False]) == 1
        assert steps[True][0].keys() == steps[False][0].keys()
        assert any("critic" in name for name in steps[True][0])
        for name, grad in steps[False][0].items():
            torch.testing.assert_close(
                steps[True][0][name], grad, rtol=RTOL, atol=ATOL, msg=name
            )
        for key in (
            "loss",
            "pg_loss",
            "vf_loss",
            "kl",
            "entropy",
            "clipfrac",
            "grad_norm_pre",
            "grad_norm_post",
            "critic_grad_norm_pre",
            "critic_grad_norm_post",
        ):
            assert metrics[True][key] == pytest.approx(
                metrics[False][key], rel=1e-5, abs=1e-6
            ), key
        split_weights = lora_weights(agents[False])
        fused_weights = lora_weights(agents[True])
        assert fused_weights.keys() == split_weights.keys()
        for name, weight in split_weights.items():
            torch.testing.assert_close(
                fused_weights[name], weight, rtol=RTOL, atol=ATOL, msg=name
            )


class TestPPOLearnMetrics:
    @FUSE_MODES
    def test_metrics_are_means_of_the_micro_batch_losses(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange: one optimizer step of four 1-row micro-batches.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
            fuse_actor_critic_pass=fuse,
        )
        policy_loss, policy_outputs = record_outputs(agent._ppo_policy_loss)
        value_loss, value_outputs = record_outputs(agent._ppo_value_loss)
        monkeypatch.setattr(agent, "_ppo_policy_loss", policy_loss)
        monkeypatch.setattr(agent, "_ppo_value_loss", value_loss)

        # Act
        metrics = learn_rows(agent)

        # Assert
        assert len(policy_outputs) == len(value_outputs) == 4
        policy = [loss.item() for loss, _ in policy_outputs]
        value = [loss.item() for loss in value_outputs]
        assert metrics["loss"] == pytest.approx(
            sum(p + v for p, v in zip(policy, value, strict=True)) / 4, rel=1e-6
        )
        assert metrics["vf_loss"] == pytest.approx(sum(value) / 4, rel=1e-6)
        for key in ("pg_loss", "kl", "entropy", "clipfrac"):
            expected = sum(float(out[key]) for _, out in policy_outputs) / 4
            assert metrics[key] == pytest.approx(expected, rel=1e-6), key


def learn_uneven_turns(agent: LLMPPO) -> dict[str, float]:
    """One learn over four trajectories: rows 0-1 take two turns, rows 2-3 one."""
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(0, PAD_TOKEN_ID, (4, SEQ_LEN + 2), generator=generator)
    mask = torch.zeros(4, SEQ_LEN + 1, dtype=torch.bool)
    mask[:, 3:] = True
    turn_ids = torch.full((4, SEQ_LEN + 1), -1)
    turn_ids[:, 3:] = 0
    turn_ids[:2, 5:] = 1
    rewards = torch.tensor([[1.0, 0.5], [0.0, -1.0], [0.5, 0.0], [-1.0, 0.0]])
    return agent.learn(
        (list(ids.split(1)), list(mask.split(1)), rewards), turn_ids=turn_ids
    )


class TestPPOLearnTurnCount:
    @FUSE_MODES
    def test_micro_batch_losses_match_their_own_turn_count(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange: 1-row micro-batches, so rows 2-3 hold fewer turns than the batch.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
            fuse_actor_critic_pass=fuse,
        )
        policy_fn, value_fn = agent._ppo_policy_loss, agent._ppo_value_loss
        policy_loss, policy_calls = record_calls(policy_fn)
        value_loss, value_calls = record_calls(value_fn)
        monkeypatch.setattr(agent, "_ppo_policy_loss", policy_loss)
        monkeypatch.setattr(agent, "_ppo_value_loss", value_loss)

        # Act
        learn_uneven_turns(agent)

        # Assert
        local_counts = []
        for args, kwargs, (loss, _) in policy_calls:
            log_probs, *inputs, turn_ids, num_turns, granularity, sampling = args
            local = int(turn_ids.max()) + 1
            local_counts.append(local)
            assert (num_turns, granularity) == (2, "turn")
            with torch.no_grad():
                expected, _ = policy_fn(
                    log_probs, *inputs, turn_ids, local, granularity, sampling, **kwargs
                )
            assert torch.equal(loss.detach(), expected)
        for args, kwargs, loss in value_calls:
            values, *inputs, turn_ids, num_turns, granularity = args
            assert num_turns == 2
            with torch.no_grad():
                expected = value_fn(
                    values,
                    *inputs,
                    turn_ids,
                    int(turn_ids.max()) + 1,
                    granularity,
                    **kwargs,
                )
            assert torch.equal(loss.detach(), expected)
        assert sorted(local_counts) == [1, 1, 2, 2]
        assert len(value_calls) == 4


class TestPPOLearnNonFiniteLoss:
    @FUSE_MODES
    def test_finite_losses_step(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
            fuse_actor_critic_pass=fuse,
        )
        monkeypatch.setattr(
            agent,
            "_ppo_value_loss",
            scale_losses(agent._ppo_value_loss, iter([1.0] * 4)),
        )
        before = trainable_weights(agent.actor)

        # Act
        learn_rows(agent)

        # Assert
        after = trainable_weights(agent.actor)
        assert any(
            not torch.equal(new, old) for new, old in zip(after, before, strict=True)
        )

    @FUSE_MODES
    @pytest.mark.parametrize(
        "window_scales",
        [[float("nan"), 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, float("nan")]],
        ids=["first_micro_batch", "last_micro_batch"],
    )
    def test_raises_before_the_step_when_a_micro_batch_loss_is_not_finite(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool, window_scales: list[float]
    ) -> None:
        # Arrange: one optimizer step of four 1-row micro-batches per learn; a
        # finite learn first gives the optimizer state to keep.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
            fuse_actor_critic_pass=fuse,
        )
        scales = iter([1.0] * 4 + window_scales)
        monkeypatch.setattr(
            agent, "_ppo_value_loss", scale_losses(agent._ppo_value_loss, scales)
        )
        learn_rows(agent)
        before = trainable_weights(agent.actor)
        state_before = optimizer_state(agent.optimizer)

        # Act
        with pytest.raises(ValueError, match="Loss is not finite"):
            learn_rows(agent)

        # Assert: it raises at the window's last backward, before the step.
        assert list(scales) == []
        for new, old in zip(trainable_weights(agent.actor), before, strict=True):
            assert torch.equal(new, old)
        state_after = optimizer_state(agent.optimizer)
        assert state_before
        assert len(state_after) == len(state_before)
        for new, old in zip(state_after, state_before, strict=True):
            assert torch.equal(new, old)

    @FUSE_MODES
    def test_raises_when_the_micro_batch_left_pending_a_step_is_not_finite(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange: three micro-batches per step over four rows leave the last
        # micro-batch's gradients pending the next learn call.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
            fuse_actor_critic_pass=fuse,
        )
        agent.gradient_accumulation_steps = 3
        scales = iter([1.0, 1.0, 1.0, float("nan")])
        monkeypatch.setattr(
            agent, "_ppo_value_loss", scale_losses(agent._ppo_value_loss, scales)
        )

        # Act / Assert
        with pytest.raises(ValueError, match="Loss is not finite"):
            learn_rows(agent)


def record_step_role_grad_norms(
    agent: LLMPPO, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, float]]:
    """L2 norm of the actor LoRA grads and of the critic LoRA plus value-head grads at each optimizer step."""
    recorded: list[dict[str, float]] = []
    optimizer_step = agent.optimizer.step

    def recording_step(*args: Any, **kwargs: Any) -> None:
        squares = {"actor": 0.0, "critic": 0.0}
        for name, param in agent.actor.named_parameters():
            if param.grad is not None:
                role = "actor" if ".actor." in name else "critic"
                squares[role] += float(param.grad.pow(2).sum())
        recorded.append({role: total**0.5 for role, total in squares.items()})
        optimizer_step(*args, **kwargs)

    monkeypatch.setattr(agent.optimizer, "step", recording_step)
    return recorded


class TestPPOLearnGradNorms:
    @FUSE_MODES
    def test_reports_actor_and_critic_norms_of_their_own_grads(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange: one optimizer step, threshold far above the norms.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=1e6,
            fuse_actor_critic_pass=fuse,
        )
        steps = record_step_role_grad_norms(agent, monkeypatch)

        # Act
        metrics = learn_rows(agent)

        # Assert
        assert len(steps) == 1
        assert steps[0]["actor"] > 0.0
        assert steps[0]["critic"] > 0.0
        # fp32 sums of the same grads in a different order.
        assert metrics["grad_norm_pre"] == pytest.approx(steps[0]["actor"], rel=1e-5)
        assert metrics["grad_norm_post"] == pytest.approx(steps[0]["actor"], rel=1e-5)
        assert metrics["critic_grad_norm_pre"] == pytest.approx(
            steps[0]["critic"], rel=1e-5
        )
        assert metrics["critic_grad_norm_post"] == pytest.approx(
            steps[0]["critic"], rel=1e-5
        )

    @FUSE_MODES
    def test_clipping_scales_post_norms_below_pre_norms(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange: same seed and rows, so both agents see identical pre-clip grads.
        unclipped = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=1e6,
            fuse_actor_critic_pass=fuse,
        )
        clipped = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=1e-3,
            fuse_actor_critic_pass=fuse,
        )
        clipped_steps = record_step_role_grad_norms(clipped, monkeypatch)

        # Act
        unclipped_metrics = learn_rows(unclipped)
        clipped_metrics = learn_rows(clipped)

        # Assert
        for prefix in ("", "critic_"):
            pre = clipped_metrics[f"{prefix}grad_norm_pre"]
            assert math.isfinite(pre)
            assert pre == pytest.approx(
                unclipped_metrics[f"{prefix}grad_norm_pre"], rel=1e-5
            )
            assert clipped_metrics[f"{prefix}grad_norm_post"] < pre
        assert clipped_metrics["grad_norm_post"] == pytest.approx(
            clipped_steps[0]["actor"], rel=1e-5
        )
        assert clipped_metrics["critic_grad_norm_post"] == pytest.approx(
            clipped_steps[0]["critic"], rel=1e-5
        )
        # Actor and critic are each clipped to the threshold.
        assert clipped_metrics["grad_norm_post"] == pytest.approx(1e-3, rel=1e-4)
        assert clipped_metrics["critic_grad_norm_post"] == pytest.approx(1e-3, rel=1e-4)

    @FUSE_MODES
    def test_clipping_the_critic_leaves_a_smaller_actor_norm_unscaled(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange: unclipped norms are about 5.3 (actor) and 7.8 (critic).
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=6.0,
            fuse_actor_critic_pass=fuse,
        )
        steps = record_step_role_grad_norms(agent, monkeypatch)

        # Act
        metrics = learn_rows(agent)

        # Assert
        assert metrics["grad_norm_pre"] < 6.0 < metrics["critic_grad_norm_pre"]
        assert metrics["grad_norm_post"] == metrics["grad_norm_pre"]
        assert metrics["critic_grad_norm_post"] == pytest.approx(6.0, rel=1e-5)
        # fp32 sums of the same grads in a different order.
        assert steps[0]["actor"] == pytest.approx(metrics["grad_norm_pre"], rel=1e-5)
        assert steps[0]["critic"] == pytest.approx(6.0, rel=1e-5)

    @FUSE_MODES
    def test_share_grad_clip_scales_both_by_the_combined_norm(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange: unclipped norms are about 5.3 (actor) and 7.8 (critic).
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=6.0,
            share_grad_clip=True,
            fuse_actor_critic_pass=fuse,
        )
        steps = record_step_role_grad_norms(agent, monkeypatch)

        # Act
        metrics = learn_rows(agent)

        # Assert
        actor_pre = metrics["grad_norm_pre"]
        critic_pre = metrics["critic_grad_norm_pre"]
        coef = 6.0 / math.hypot(actor_pre, critic_pre)
        assert actor_pre < 6.0 < critic_pre
        assert metrics["grad_norm_post"] == pytest.approx(actor_pre * coef, rel=1e-5)
        assert metrics["critic_grad_norm_post"] == pytest.approx(
            critic_pre * coef, rel=1e-5
        )
        # fp32 sums of the same grads in a different order.
        assert math.hypot(steps[0]["actor"], steps[0]["critic"]) == pytest.approx(
            6.0, rel=1e-5
        )


def patch_cuda_device(
    agent: LLMPPO, monkeypatch: pytest.MonkeyPatch, total_memory: int
) -> None:
    """Point ``agent`` at a CUDA device with ``total_memory`` bytes."""
    monkeypatch.setattr(agent, "device", "cuda:0")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            total_memory=total_memory, name="test-gpu", major=8, minor=0
        ),
    )


def save_checkpoint_config(
    agent: LLMPPO, path: Path, config: dict[str, Any] | None = None
) -> None:
    """Write ``config`` (default: the actor's) as the checkpoint's ``config.json``."""
    path.mkdir(parents=True, exist_ok=True)
    if config is None:
        agent.actor.config.save_pretrained(path)
    else:
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    agent.pretrained_model_name_or_path = str(path)


class TestPPOFuseActorCriticPass:
    def test_none_fuses_off_cuda(self) -> None:
        # Arrange
        agent = make_ppo(batch_size=4, mini_batch_size=4, micro_batch_size_per_gpu=2)
        forwards = record_grad_forwards(agent)

        # Act
        learn_rows(agent)

        # Assert
        assert agent.fuse_actor_critic_pass is None
        assert agent._fuses_actor_critic_pass is True
        assert [f["adapters"] for f in forwards] == [{"actor", "critic"}] * 2

    @pytest.mark.parametrize("fuse", [False, True])
    def test_explicit_choice_is_kept(self, fuse: bool) -> None:
        agent = make_ppo(fuse_actor_critic_pass=fuse)

        assert agent.fuse_actor_critic_pass is fuse
        assert agent._fuses_actor_critic_pass is fuse

    @pytest.mark.parametrize(
        ("total_gib", "fuse", "verb", "budget"),
        [(80, True, "fuses", "78.00"), (1, False, "splits", "0.97")],
        ids=["fits", "does-not-fit"],
    )
    def test_none_on_cuda_fuses_only_when_the_estimate_fits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
        total_gib: int,
        fuse: bool,
        verb: str,
        budget: str,
    ) -> None:
        # Arrange: the tiny model's fused estimate is about 1.7 GiB.
        agent = make_ppo()
        save_checkpoint_config(agent, tmp_path)
        patch_cuda_device(agent, monkeypatch, total_gib * 2**30)

        # Act
        with caplog.at_level(logging.INFO, logger="agilerl.algorithms.ppo_llm"):
            resolved = agent._resolve_fuse_actor_critic_pass()

        # Assert
        assert resolved is fuse
        assert f"LLMPPO {verb} the actor and critic passes" in caplog.text
        assert f"budget {budget} GiB" in caplog.text

    @pytest.mark.parametrize(
        ("sleep_mode", "fuse", "budget"),
        [(False, False, "0.30"), (True, True, "3.90")],
        ids=["resident-engine", "sleeping-engine"],
    )
    def test_none_on_cuda_leaves_a_resident_vllm_share_out_of_the_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
        sleep_mode: bool,
        fuse: bool,
        budget: str,
    ) -> None:
        # Arrange: 4 GiB fits the ~1.7 GiB fused estimate unless vLLM keeps
        # 90% of it: 0.975 * 4 - 0.9 * 4 = 0.3 GiB.
        agent = make_ppo()
        save_checkpoint_config(agent, tmp_path)
        patch_cuda_device(agent, monkeypatch, 4 * 2**30)
        monkeypatch.setattr(agent, "colocated", True)
        monkeypatch.setattr(
            agent,
            "vllm_config",
            VLLMConfig(gpu_memory_utilization=0.9, sleep_mode=sleep_mode),
        )

        # Act
        with caplog.at_level(logging.INFO, logger="agilerl.algorithms.ppo_llm"):
            resolved = agent._resolve_fuse_actor_critic_pass()

        # Assert
        assert resolved is fuse
        assert f"budget {budget} GiB" in caplog.text

    def test_none_on_cuda_estimates_the_gradient_micro_batch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange
        agent = make_ppo(batch_size=6, mini_batch_size=6, micro_batch_size_per_gpu=3)
        save_checkpoint_config(agent, tmp_path)
        patch_cuda_device(agent, monkeypatch, 80 * 2**30)
        estimated = []

        def record_estimate(model, device, settings):
            estimated.append(settings)
            return estimate_training(model, device, settings)

        monkeypatch.setattr(
            "agilerl.algorithms.core.base.estimate_training", record_estimate
        )

        # Act
        resolved = agent._resolve_fuse_actor_critic_pass()

        # Assert
        assert resolved is True
        assert [s.micro_batch_size for s in estimated] == [3]
        assert estimated[0].grad_forward_rows == 6

    def test_none_on_cuda_reads_the_checkpoint_config_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange: transformers' NemotronHConfig.to_dict drops num_hidden_layers
        # and renames the layer types; the checkpoint's config.json keeps both.
        agent = make_ppo()
        nemotron = SUPPORTED_MODEL_INFO["nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"]
        save_checkpoint_config(agent, tmp_path, nemotron.config)
        patch_cuda_device(agent, monkeypatch, 80 * 2**30)
        archs = []

        def record_estimate(model, device, settings):
            archs.append(model.arch)
            return estimate_training(model, device, settings)

        monkeypatch.setattr(
            "agilerl.algorithms.core.base.estimate_training", record_estimate
        )

        # Act
        agent._resolve_fuse_actor_critic_pass()

        # Assert
        (arch,) = archs
        assert arch.n_layers == 42
        assert arch.attention_layers == 4
        assert arch.n_mamba_layers == 21
        assert arch.is_moe is False

    def test_clone_keeps_the_requested_mode(self) -> None:
        agent = make_ppo()

        clone = agent.clone()

        assert clone.fuse_actor_critic_pass is None
        assert clone._fuses_actor_critic_pass is True

    def test_checkpoint_load_re_resolves_on_the_loading_device(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange: saved by a CPU agent that fused; loaded on a 1 GiB GPU.
        saved = make_ppo()
        save_checkpoint_config(saved, tmp_path / "model")
        saved.save_checkpoint(str(tmp_path / "checkpoint"))
        loaded = make_ppo()
        save_checkpoint_config(loaded, tmp_path / "model")
        patch_cuda_device(loaded, monkeypatch, 2**30)

        # Act
        loaded.load_checkpoint(str(tmp_path / "checkpoint"))

        # Assert
        assert saved._fuses_actor_critic_pass is True
        assert loaded.fuse_actor_critic_pass is None
        assert loaded._fuses_actor_critic_pass is False


class TestPPOUseAdapter:
    def test_critic_adapter_stays_trainable_after_parameters_are_replaced(
        self,
    ) -> None:
        # Arrange: FSDP2 materialization swaps every parameter object.
        agent = make_ppo()
        agent.use_adapter("actor")
        state = {key: value.clone() for key, value in agent.actor.state_dict().items()}
        agent.actor.load_state_dict(state, assign=True)

        # Act
        agent.use_adapter("actor")

        # Assert
        frozen = [
            name
            for name, param in agent.actor.named_parameters()
            if name in adapter_lora_names(agent) and not param.requires_grad
        ]
        assert frozen == []


@pytest.mark.skipif(
    sys.platform == "win32" or not dist.is_available(), reason="gloo unavailable"
)
class TestPPOLearnFSDP:
    @pytest.mark.parametrize("liger", [False, True], ids=["standard", "liger"])
    @FUSE_MODES
    def test_two_rank_step_matches_one_rank_accumulated_step(
        self, monkeypatch: pytest.MonkeyPatch, liger: bool, fuse: bool
    ) -> None:
        # Arrange: one split rank accumulating all four rows is the reference
        # for two ranks accumulating two rows each.
        single = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
            whiten_advantages=False,
            fuse_actor_critic_pass=False,
        )
        seed_lora_weights(single)
        if liger:
            use_fake_liger_policy_loss(
                single, monkeypatch, "agilerl.algorithms.ppo_llm"
            )
        single_steps = record_step_gradients(single, monkeypatch)
        learn_rows(single)

        # Act
        reports = spawn_fsdp_learn(liger, fuse)

        # Assert
        assert len(single_steps) == 1
        for lora_names, steps in reports:
            assert len(steps) == 1
            assert steps[0].keys() == lora_names
            assert steps[0].keys() == single_steps[0].keys()
            for name, grad in single_steps[0].items():
                torch.testing.assert_close(
                    torch.from_numpy(steps[0][name]),
                    grad,
                    rtol=RTOL,
                    atol=ATOL,
                    msg=name,
                )
