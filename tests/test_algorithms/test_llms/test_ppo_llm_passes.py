# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLM PPO fused actor and critic pass.

Pure CPU on tiny real models in fp32. The reference for each pass pair is one
forward over the doubled ``[actor rows; critic rows]`` batch with one backward
of the summed loss.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.distributed.fsdp import FSDPConfig
from agilerl.lora.fused import (
    ROUTING_STATE,
    get_cached_lora_layers,
    unset_fused_adapter_routing,
)
from agilerl.utils.llm_utils import attention_mask_from_padded_ids
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from tests.rank_launch import run_ranks
from tests.test_algorithms.test_llms.llm_helpers import (
    DummyConfig,
    DummyHiddenStatesModel,
    create_tiny_qwen3,
    optimizer_state,
    record_calls,
    record_outputs,
    scale_losses,
    trainable_weights,
)
from tests.test_algorithms.test_llms.segment_helpers import (
    RESOLVED_FSDP_MEMORY,
    record_step_gradients,
    seed_lora_weights,
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
# fp32 on CPU; masked padded attention and per-document attention reduce the
# same real tokens in a different order.
ATTENTION_ATOL = 1e-5
VARLEN_ATTENTION = "agilerl_test_ppo_varlen"


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
    attn_implementation: str = "eager",
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
    model = model_cls(config=config, device="cpu")
    # The dummy forward ignores attention; set past HF's backend check.
    model.config._attn_implementation = attn_implementation
    kwargs: dict[str, Any] = {
        "actor_network": AutoModelForCausalLMWithValueHead(model),
        "pad_token_id": PAD_TOKEN_ID,
        "pad_token": "<pad>",
        "batch_size": 2,
        "micro_batch_size_per_gpu": 2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "update_epochs": 1,
        "wrap": False,
        "calc_position_embeddings": False,
        "device": "cpu",
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
        _, critic_values = agent._actor_critic_hidden_states(batch["ids"])
        value_loss(agent, critic_values, batch, "token").backward()
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
            _, values = agent._actor_critic_hidden_states(batch["ids"])
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


class TestLLMPPOActorLogProbsFromHidden:
    def test_scored_tokens_match_the_full_head(self) -> None:
        # Arrange
        agent = make_ppo()
        batch = micro_batch()
        hidden, _ = agent._actor_critic_hidden_states(batch["ids"])
        fused_fn, _, _ = agent._fused_logprob_fn_and_head()
        head_w, head_b = agent._get_lm_head().weight, None
        full = agent._fused_chunk_logprobs(
            hidden, batch["ids"], fused_fn, head_w, head_b
        )

        # Act
        scored = agent._actor_log_probs_from_hidden(hidden, batch["ids"], batch["mask"])

        # Assert
        mask = batch["mask"].bool()
        # The head sees the scored rows as one flat row; fp32 GEMM blocking
        # follows the row count.
        torch.testing.assert_close(scored[mask], full[mask], rtol=0, atol=1e-6)
        assert torch.equal(scored[~mask], torch.zeros_like(scored[~mask]))


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
            fused,
            fused._actor_log_probs_from_hidden(hidden, batch["ids"], batch["mask"]),
            batch,
            mode,
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
                agent._actor_log_probs_from_hidden(hidden, batch["ids"], batch["mask"]),
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


@pytest.fixture
def varlen_attention() -> list[torch.Tensor | None]:
    """Register a CPU stand-in for FlashAttention-2 varlen and record each layer's offsets.

    Like the varlen kernel it flattens every row into one sequence and attends
    causally within each ``cu_seq_lens_q`` segment, or within each row when no
    offsets are passed.
    """
    offsets_seen: list[torch.Tensor | None] = []

    def attend(module, query, key, value, attention_mask, scaling=None, **kwargs):
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
        offsets_seen.append(cu_seq_lens_q)
        rows, heads, length, head_dim = query.shape
        query, key, value = (
            tensor.transpose(1, 2)
            .reshape(1, rows * length, -1, head_dim)
            .transpose(1, 2)
            for tensor in (query, key, value)
        )
        offsets = (
            list(range(0, rows * length + 1, length))
            if cu_seq_lens_q is None
            else cu_seq_lens_q.tolist()
        )
        segments = [
            F.scaled_dot_product_attention(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                is_causal=True,
                scale=scaling,
                enable_gqa=True,
            )
            for start, end in itertools.pairwise(offsets)
        ]
        out = torch.cat(segments, dim=2).transpose(1, 2)
        return out.reshape(rows, length, heads, head_dim).contiguous(), None

    AttentionInterface.register(VARLEN_ATTENTION, attend)
    AttentionMaskInterface.register(VARLEN_ATTENTION, flash_attention_mask)
    return offsets_seen


def make_qwen3_ppo(attn_implementation: str) -> LLMPPO:
    """Tiny fp32 Qwen3 PPO on CPU with real attention; actor and critic adapters start different."""
    model = create_tiny_qwen3()
    model.config._attn_implementation = attn_implementation
    # Liger's instance patches are Triton kernels; this model runs on CPU.
    with patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False):
        agent = LLMPPO(
            actor_network=AutoModelForCausalLMWithValueHead(model),
            pad_token_id=PAD_TOKEN_ID,
            pad_token="<pad>",
            batch_size=4,
            micro_batch_size_per_gpu=4,
            max_output_tokens=4,
            max_model_len=16,
            update_epochs=1,
            wrap=False,
            calc_position_embeddings=False,
            device="cpu",
            lora_config=LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj", "down_proj"],
                task_type="CAUSAL_LM",
                lora_dropout=0.0,
                modules_to_save=["summary"],
            ),
        )
    with torch.no_grad():
        for name, param in agent.actor.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.5)
    return agent


def varied_micro_batch(
    lengths: list[int], seq_len: int = 12
) -> dict[str, torch.Tensor]:
    """Learn-loop tensors of right-padded rows of *lengths* real tokens, two turns per row."""
    generator = torch.Generator().manual_seed(3)
    rows = len(lengths)
    ids = torch.randint(0, PAD_TOKEN_ID, (rows, seq_len), generator=generator)
    real = torch.arange(seq_len) < torch.tensor(lengths).unsqueeze(1)
    ids = ids.masked_fill(~real, PAD_TOKEN_ID)
    # Predictions of real tokens after a two-token prompt are actions.
    positions = torch.arange(seq_len - 1)
    last = torch.tensor(lengths).unsqueeze(1) - 1
    mask = (positions >= 2) & (positions < last)
    turn_ids = torch.where(positions < (last + 2) // 2, 0, 1).masked_fill(~mask, -1)
    frame = (rows, seq_len - 1)
    return {
        "ids": ids,
        "mask": mask.float(),
        "old_log_probs": -torch.rand(frame, generator=generator) - 3.0,
        "reference_log_probs": -torch.rand(frame, generator=generator) - 3.0,
        "advantages": torch.randn(frame, generator=generator),
        "returns": torch.randn(frame, generator=generator),
        "old_values": torch.randn(frame, generator=generator),
        "turn_ids": turn_ids,
    }


VARIED_LENGTHS = [12, 9, 6, 11]


class TestPPOFusedPassVarlenPacking:
    """FlashAttention-2 packs the fused pass as an actor copy and a critic copy of one flattened varlen batch."""

    @pytest.mark.parametrize("mode", ["token", "turn"])
    def test_losses_and_gradients_match_the_padded_pass(
        self,
        mode: str,
        monkeypatch: pytest.MonkeyPatch,
        varlen_attention: list[torch.Tensor | None],
    ) -> None:
        # Arrange
        padded = make_qwen3_ppo("sdpa")
        packed = make_qwen3_ppo(VARLEN_ATTENTION)
        monkeypatch.setattr(packed, "_packing_mode", lambda: "varlen")
        forwards = record_grad_forwards(packed)
        batch = varied_micro_batch(VARIED_LENGTHS)

        # Act
        losses = []
        for agent in (padded, packed):
            hidden, values = agent._actor_critic_hidden_states(batch["ids"])
            loss = policy_loss(
                agent,
                agent._actor_log_probs_from_hidden(hidden, batch["ids"], batch["mask"]),
                batch,
                mode,
            ) + value_loss(agent, values, batch, mode)
            loss.backward()
            unset_fused_adapter_routing(agent.actor)
            losses.append(loss)

        # Assert: one padding-free row per adapter, offsets over both copies.
        real_tokens = sum(VARIED_LENGTHS)
        assert [f["input_ids"].shape for f in forwards] == [(2, real_tokens)]
        assert forwards[0]["adapters"] == {"actor", "critic"}
        # Two layers, each recomputed by the checkpointed backward.
        assert len(varlen_attention) == 4
        for offsets in varlen_attention:
            assert offsets is not None
            assert offsets.tolist() == [0, 12, 21, 27, 38, 50, 59, 65, 76]
        torch.testing.assert_close(losses[1], losses[0], rtol=RTOL, atol=ATTENTION_ATOL)
        padded_grads = lora_grads(padded)
        packed_grads = lora_grads(packed)
        assert packed_grads.keys() == padded_grads.keys()
        assert any("critic" in name for name in packed_grads)
        assert any("actor" in name for name in packed_grads)
        for name, grad in padded_grads.items():
            torch.testing.assert_close(
                packed_grads[name], grad, rtol=RTOL, atol=ATTENTION_ATOL, msg=name
            )

    def test_each_copy_runs_under_its_own_adapter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        varlen_attention: list[torch.Tensor | None],
    ) -> None:
        # Arrange: single-adapter packed forwards of the same rows.
        agent = make_qwen3_ppo(VARLEN_ATTENTION)
        monkeypatch.setattr(agent, "_packing_mode", lambda: "varlen")
        batch = varied_micro_batch(VARIED_LENGTHS)
        real = attention_mask_from_padded_ids(batch["ids"], PAD_TOKEN_ID).bool()
        with torch.no_grad():
            actor_hidden = agent._actor_hidden_states(batch["ids"])
            unset_fused_adapter_routing(agent.actor)
            critic_values = agent._critic_values(batch["ids"])
            unset_fused_adapter_routing(agent.actor)

            # Act
            fused_hidden, fused_values = agent._actor_critic_hidden_states(batch["ids"])
            unset_fused_adapter_routing(agent.actor)

        # Assert: the first copy's tokens took the actor delta, the second's
        # the critic delta.
        torch.testing.assert_close(
            fused_hidden[real], actor_hidden[real], rtol=RTOL, atol=ATOL
        )
        torch.testing.assert_close(
            fused_values[real[:, 1:]],
            critic_values[real[:, 1:]],
            rtol=RTOL,
            atol=ATOL,
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


def fsdp_learn_worker(rank: int, world_size: int) -> Any:
    """One FSDP2 rank: learn on its half of :func:`learn_rows` and report step grads."""
    # The default FSDP2 mesh follows the host accelerator (MPS on macOS);
    # these ranks train on CPU.
    torch._C._get_accelerator = lambda: torch.device("cpu")
    agent = make_ppo(
        batch_size=4,
        mini_batch_size=2,
        micro_batch_size_per_gpu=1,
        whiten_advantages=False,
        wrap=True,
        fsdp_config=FSDPConfig(param_dtype="float32", **RESOLVED_FSDP_MEMORY),
    )
    seed_lora_weights(agent)
    monkeypatch = pytest.MonkeyPatch()
    steps = record_step_gradients(agent, monkeypatch)
    learn_rows(agent, slice(2 * rank, 2 * rank + 2))
    # NumPy arrays: a torch tensor sent to the parent shares memory that dies
    # with this process before the parent reads it.
    step_arrays = [
        {name: grad.numpy() for name, grad in step.items()} for step in steps
    ]
    return (adapter_lora_names(agent), step_arrays)


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


class TestPPOLearnPasses:
    def test_every_gradient_forward_runs_one_micro_batch(self) -> None:
        # Arrange
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
        )
        forwards = record_grad_forwards(agent)

        # Act
        learn_rows(agent)

        # Assert: fused runs actor and critic rows in one forward.
        assert [f["adapters"] for f in forwards] == [{"actor", "critic"}] * 2
        assert [f["input_ids"].shape[0] for f in forwards] == [4, 4]

    def test_activation_offload_covers_every_gradient_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
        )
        agent.activation_offload = True
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
        assert depths == [1] * 2

    def test_accumulated_micro_batches_match_one_micro_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: equal action counts per row, so the mean over one 4-row
        # micro-batch equals the mean of four 1-row micro-batch means.
        accumulated = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
        )
        single = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=4,
        )
        assert accumulated.gradient_accumulation_steps == 4
        assert single.gradient_accumulation_steps == 1
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
        for key in ("loss", "pg_loss", "vf_loss", "kl"):
            assert accumulated_metrics[key] == pytest.approx(
                single_metrics[key], rel=1e-5, abs=1e-6
            ), key

    def test_vision_passes_each_see_only_their_rows_images(self) -> None:
        # Arrange
        agent = make_vision_ppo(micro_batch_size_per_gpu=2)
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
        assert [f["adapters"] for f in forwards] == [{"actor", "critic"}] * 2
        for forward in forwards:
            assert forward["input_ids"].shape[0] == 2 * len(forward["adapters"])
            assert forward["pixel_rows"] == int(
                (forward["input_ids"] == IMAGE_TOKEN_ID).sum()
            )

    def test_optimizer_steps_after_the_last_pass_of_each_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: two optimizer steps of two micro-batches each.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=2,
            micro_batch_size_per_gpu=1,
        )
        assert agent.gradient_accumulation_steps == 2
        backwards = record_runtime_backwards(agent, monkeypatch)
        steps = record_step_gradients(agent, monkeypatch)

        # Act
        learn_rows(agent)

        # Assert
        both = frozenset({"actor", "critic"})
        window = [(both, False), (both, True)]
        assert backwards == window * 2
        assert len(steps) == 2
        for step in steps:
            assert step.keys() == adapter_lora_names(agent)

    def test_a_row_without_action_tokens_adds_no_loss_or_metrics(self) -> None:
        """A micro-batch whose every action vLLM forced trains like no row at all."""
        # Arrange
        generator = torch.Generator().manual_seed(2)
        ids = list(
            torch.randint(0, PAD_TOKEN_ID, (2, SEQ_LEN + 2), generator=generator).split(
                1
            )
        )
        masks = [torch.zeros(1, SEQ_LEN + 1, dtype=torch.bool) for _ in range(2)]
        masks[0][0, 3:] = True
        agents = [make_ppo(micro_batch_size_per_gpu=1) for _ in range(2)]

        # Act
        with_forced_row = agents[0].learn((ids, masks, torch.tensor([1.0, 0.0])))
        live_row_only = agents[1].learn((ids[:1], masks[:1], torch.tensor([1.0])))

        # Assert: float32 sums over different batch shapes differ in the last bits
        for name in ("loss", "pg_loss", "vf_loss", "kl", "entropy", "clipfrac"):
            assert with_forced_row[name] == pytest.approx(
                live_row_only[name], rel=1e-6, abs=1e-7
            ), name


def learn_ragged_rows(agent: LLMPPO) -> dict[str, float]:
    """One learn over four trajectories; the last two tokens of row 1 are padding."""
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(0, PAD_TOKEN_ID, (4, SEQ_LEN + 2), generator=generator)
    ids[1, -2:] = PAD_TOKEN_ID
    mask = torch.zeros(4, SEQ_LEN + 1, dtype=torch.bool)
    mask[:, 3:] = True
    mask[1, -2:] = False
    rewards = torch.tensor([1.0, 0.0, 0.5, -1.0])
    return agent.learn((list(ids.split(1)), list(mask.split(1)), rewards))


class TestPPOLearnPaddingStats:
    def test_packed_forward_runs_without_padding(self) -> None:
        # Arrange
        agent = make_ppo(
            attn_implementation="flex_attention",
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
        )
        forwards = record_grad_forwards(agent)

        # Act
        metrics = learn_ragged_rows(agent)

        # Assert: 2 of 32 tokens pad. The 2 x 12-token budget packs rows 0-2
        # (22 tokens), then row 3; each forward stacks its actor and critic rows.
        assert metrics["padding_frac_before_packing"] == pytest.approx(2 / 32)
        assert metrics["padding_frac_after_packing"] == 0.0
        assert [f["input_ids"].shape for f in forwards] == [
            torch.Size([2, 22]),
            torch.Size([2, 8]),
        ]

    def test_padded_forward_keeps_the_batch_padding(self) -> None:
        # Arrange
        agent = make_ppo(batch_size=4, mini_batch_size=4, micro_batch_size_per_gpu=2)

        # Act
        metrics = learn_ragged_rows(agent)

        # Assert
        assert metrics["padding_frac_before_packing"] == pytest.approx(2 / 32)
        assert metrics["padding_frac_after_packing"] == pytest.approx(2 / 32)


def learn_uneven_rows(agent: LLMPPO, turns: bool) -> dict[str, float]:
    """One learn over four rows of 8, 4, 6 and 3 real tokens; with ``turns``, rows 0 and 2 take two."""
    generator = torch.Generator().manual_seed(3)
    lengths = [8, 4, 6, 3]
    ids = torch.randint(0, PAD_TOKEN_ID, (4, 8), generator=generator)
    mask = torch.zeros(4, 7, dtype=torch.bool)
    for row, length in enumerate(lengths):
        ids[row, length:] = PAD_TOKEN_ID
        mask[row, 1 : length - 1] = True
    if not turns:
        return agent.learn(
            (
                list(ids.split(1)),
                list(mask.split(1)),
                torch.tensor([1.0, 0.0, 0.5, -1.0]),
            )
        )
    turn_ids = torch.where(mask, 0, -1)
    turn_ids[0, 4:7] = 1
    turn_ids[2, 3:5] = 1
    rewards = torch.tensor([[1.0, 0.5], [0.0, 0.0], [0.5, -1.0], [-1.0, 0.0]])
    return agent.learn(
        (list(ids.split(1)), list(mask.split(1)), rewards), turn_ids=turn_ids
    )


class TestPPOLearnTokenBudget:
    @pytest.mark.parametrize("turns", [False, True], ids=["token", "turn"])
    def test_step_gradients_match_one_window_micro_batch(
        self, monkeypatch: pytest.MonkeyPatch, turns: bool
    ) -> None:
        # Arrange: a 12-token budget splits the window into micro-batches of
        # unequal tokens, action tokens and turns; the reference runs the
        # whole window as one micro-batch.
        window = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=4,
            whiten_advantages=False,
        )
        budgeted = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=4,
            micro_batch_tokens_per_gpu=12,
            whiten_advantages=False,
        )
        monkeypatch.setattr(budgeted, "_packing_mode", lambda: "varlen")
        forwards = record_grad_forwards(budgeted)
        window_steps = record_step_gradients(window, monkeypatch)
        budgeted_steps = record_step_gradients(budgeted, monkeypatch)

        # Act
        window_metrics = learn_uneven_rows(window, turns)
        budgeted_metrics = learn_uneven_rows(budgeted, turns)

        # Assert
        # Each forward runs one micro-batch's rows under both adapters.
        assert budgeted_metrics["micro_batches_per_step"] == len(forwards)
        assert window_metrics["micro_batches_per_step"] == 1.0
        assert len(forwards) >= 2
        assert all((f["input_ids"] != PAD_TOKEN_ID).sum() <= 2 * 12 for f in forwards)
        assert len(window_steps) == len(budgeted_steps) == 1
        assert budgeted_steps[0].keys() == window_steps[0].keys()
        assert any("critic" in name for name in window_steps[0])
        for name, grad in window_steps[0].items():
            torch.testing.assert_close(
                budgeted_steps[0][name], grad, rtol=RTOL, atol=ATOL, msg=name
            )

    def test_budget_below_the_longest_row_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"micro_batch_tokens_per_gpu \(11\)"):
            make_ppo(micro_batch_tokens_per_gpu=11)


class TestPPOLearnMetrics:
    def test_metrics_are_means_of_the_micro_batch_losses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: one optimizer step of four 1-row micro-batches.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
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


def learn_turns_with_sampling_logps(
    agent: LLMPPO, multi_turn: bool, sampling_logps: bool
) -> dict[str, float]:
    """One learn over four trajectories of one or two turns, with or without vLLM log-probs."""
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(0, PAD_TOKEN_ID, (4, SEQ_LEN + 2), generator=generator)
    mask = torch.zeros(4, SEQ_LEN + 1, dtype=torch.bool)
    mask[:, 3:] = True
    turn_ids = torch.full((4, SEQ_LEN + 1), -1)
    turn_ids[:, 3:] = 0
    if multi_turn:
        turn_ids[:, 5:] = 1
    rewards = torch.tensor([[1.0, 0.5], [0.0, -1.0], [0.5, 0.0], [-1.0, 0.0]])
    if not multi_turn:
        rewards = rewards.sum(dim=-1)
    action_tokens = int(mask[0].sum())
    return agent.learn(
        (list(ids.split(1)), list(mask.split(1)), rewards),
        turn_ids=turn_ids,
        sampling_logps=[torch.full((action_tokens,), -3.0)] * 4
        if sampling_logps
        else None,
    )


class TestPPOLearnTurnCount:
    def test_micro_batch_losses_match_their_own_turn_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: 1-row micro-batches, so rows 2-3 hold fewer turns than the batch.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
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


def learn_three_turns(agent: LLMPPO, forced: bool) -> dict[str, float]:
    """One learn over two three-turn episodes of three tokens per turn.

    With ``forced``, every token of row 0's middle and last turn leaves the
    action mask and keeps its turn id.
    """
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(0, PAD_TOKEN_ID, (2, 11), generator=generator)
    turn_ids = torch.tensor([[-1, 0, 0, 0, 1, 1, 1, 2, 2, 2]]).repeat(2, 1)
    mask = turn_ids >= 0
    if forced:
        mask[0, 4:] = False
    rewards = torch.tensor([[0.5, 1.0, 2.0], [0.0, -1.0, 0.5]])
    return agent.learn(
        (list(ids.split(1)), list(mask.split(1)), rewards), turn_ids=turn_ids
    )


class TestPPOLearnForcedTurns:
    """A turn of forced tokens keeps its reward and its place in the episode."""

    def test_returns_and_advantages_match_the_episode_with_free_turns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        results = {}
        for forced in (False, True):
            agent = make_ppo(advantage_granularity="turn", gamma=0.9)
            gae, gae_calls = record_calls(agent._compute_gae_returns)
            monkeypatch.setattr(agent, "_compute_gae_returns", gae)

            # Act
            metrics = learn_three_turns(agent, forced=forced)
            ((_args, _kwargs, (returns, advantages)),) = gae_calls
            results[forced] = (returns, advantages, metrics["vf_loss"])

        # Assert
        free_returns, free_advantages, free_vf_loss = results[False]
        forced_returns, forced_advantages, forced_vf_loss = results[True]
        assert torch.equal(forced_returns, free_returns)
        assert torch.equal(forced_advantages, free_advantages)
        assert forced_vf_loss == free_vf_loss


class TestPPOLearnNonFiniteLoss:
    def test_finite_losses_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
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

    @pytest.mark.parametrize(
        "window_scales",
        [[float("nan"), 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, float("nan")]],
        ids=["first_micro_batch", "last_micro_batch"],
    )
    def test_raises_before_the_step_when_a_micro_batch_loss_is_not_finite(
        self, monkeypatch: pytest.MonkeyPatch, window_scales: list[float]
    ) -> None:
        # Arrange: one optimizer step of four 1-row micro-batches per learn; a
        # finite learn first gives the optimizer state to keep.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
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

    def test_raises_when_the_micro_batch_left_pending_a_step_is_not_finite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: three micro-batches per step over four rows leave the last
        # micro-batch's gradients pending the next learn call.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
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
    def test_reports_actor_and_critic_norms_of_their_own_grads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: one optimizer step, threshold far above the norms.
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=1e6,
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

    def test_clipping_scales_post_norms_below_pre_norms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: same seed and rows, so both agents see identical pre-clip grads.
        unclipped = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=1e6,
        )
        clipped = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=1e-3,
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

    def test_clipping_the_critic_leaves_a_smaller_actor_norm_unscaled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: unclipped norms are about 5.3 (actor) and 7.8 (critic).
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=6.0,
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

    def test_share_grad_clip_scales_both_by_the_combined_norm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: unclipped norms are about 5.3 (actor) and 7.8 (critic).
        agent = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            max_grad_norm=6.0,
            share_grad_clip=True,
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


def cuda_total_under_peak(peak_bytes: int) -> int:
    """CUDA total whose 95% usable bytes are one byte under ``peak_bytes``."""
    usable = peak_bytes - 1
    total = usable * 20 // 19
    while int(total * 0.95) < usable:
        total += 1
    while int(total * 0.95) > usable:
        total -= 1
    return total


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


class TestPPOLearnFusedActorCriticForward:
    def test_learn_runs_actor_and_critic_rows_in_one_forward(self) -> None:
        # Arrange
        agent = make_ppo(batch_size=4, mini_batch_size=4, micro_batch_size_per_gpu=2)
        forwards = record_grad_forwards(agent)

        # Act
        learn_rows(agent)

        # Assert
        assert [f["adapters"] for f in forwards] == [{"actor", "critic"}] * 2

    def test_clone_learns_fused(self) -> None:
        # Arrange
        agent = make_ppo(batch_size=4, mini_batch_size=4, micro_batch_size_per_gpu=2)
        clone = agent.clone()
        forwards = record_grad_forwards(clone)

        # Act
        learn_rows(clone)

        # Assert
        assert [f["adapters"] for f in forwards] == [{"actor", "critic"}] * 2

    def test_checkpoint_load_keeps_the_fused_learn(self, tmp_path: Path) -> None:
        # Arrange
        saved = make_ppo(batch_size=4, mini_batch_size=4, micro_batch_size_per_gpu=2)
        save_checkpoint_config(saved, tmp_path / "model")
        saved.save_checkpoint(str(tmp_path / "checkpoint"))
        loaded = make_ppo(batch_size=4, mini_batch_size=4, micro_batch_size_per_gpu=2)
        save_checkpoint_config(loaded, tmp_path / "model")

        # Act
        loaded.load_checkpoint(str(tmp_path / "checkpoint"))
        forwards = record_grad_forwards(loaded)
        learn_rows(loaded)

        # Assert
        assert [f["adapters"] for f in forwards] == [{"actor", "critic"}] * 2


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
@pytest.mark.multi_rank
class TestPPOLearnFSDP:
    def test_two_rank_step_matches_one_rank_accumulated_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: one rank accumulating all four rows is the reference
        # for two ranks accumulating two rows each.
        single = make_ppo(
            batch_size=4,
            mini_batch_size=4,
            micro_batch_size_per_gpu=1,
            whiten_advantages=False,
        )
        seed_lora_weights(single)
        single_steps = record_step_gradients(single, monkeypatch)
        learn_rows(single)

        # Act
        reports = run_ranks(fsdp_learn_worker, 2, backend=None)

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


class TestResolveFsdpReduceDtype:
    def test_keeps_configured_reduce_dtype(self) -> None:
        # Arrange
        agent = make_ppo()

        # Act
        resolved = agent._resolve_fsdp_config(FSDPConfig(**RESOLVED_FSDP_MEMORY))

        # Assert
        assert resolved.reduce_dtype == "float32"

    def test_already_equal_dtypes_keep_configured_reduce_dtype(self) -> None:
        # Arrange
        agent = make_ppo()

        # Act
        resolved = agent._resolve_fsdp_config(
            FSDPConfig(reduce_dtype="bfloat16", **RESOLVED_FSDP_MEMORY)
        )

        # Assert
        assert resolved.reduce_dtype == resolved.param_dtype == "bfloat16"
