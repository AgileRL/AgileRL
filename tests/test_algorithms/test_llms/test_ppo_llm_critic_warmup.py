# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM PPO critic warmup: critic-only learn steps before PPO."""

from __future__ import annotations

import math
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.distributed.fsdp import FSDPConfig
from agilerl.utils.algo_utils import CosineLRScheduleConfig
from tests.test_algorithms.test_llms.test_ppo_llm_passes import (
    FUSE_MODES,
    learn_rows,
    make_ppo,
    seed_lora_weights,
)
from tests.test_utils.test_expert_parallel import (
    _init_gloo,
    _spawn_ranks,
    requires_gloo,
)


def make_warmup_ppo(**overrides: Any) -> LLMPPO:
    return make_ppo(
        batch_size=4, mini_batch_size=4, micro_batch_size_per_gpu=2, **overrides
    )


def role_weights(agent: LLMPPO) -> dict[str, dict[str, torch.Tensor]]:
    """Copies of the trainable weights of the actor and of the critic."""
    weights: dict[str, dict[str, torch.Tensor]] = {"actor": {}, "critic": {}}
    for name, param in agent.actor.named_parameters():
        if ".actor." in name:
            weights["actor"][name] = param.detach().clone()
        elif ".critic." in name or "v_head" in name:
            weights["critic"][name] = param.detach().clone()
    return weights


def changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(before[name], after[name]) for name in before)


class TestPPOLearnCriticWarmup:
    @FUSE_MODES
    def test_warmup_step_updates_only_the_critic(self, fuse: bool) -> None:
        # Arrange
        agent = make_warmup_ppo(fuse_actor_critic_pass=fuse, critic_warmup_steps=1)
        before = role_weights(agent)

        # Act
        metrics = learn_rows(agent)

        # Assert
        after = role_weights(agent)
        assert before["actor"]
        assert not changed(before["actor"], after["actor"])
        assert changed(before["critic"], after["critic"])
        assert metrics["critic_warmup"] == 1.0
        assert metrics["pg_loss"] == 0.0
        assert agent.critic_warmup_steps_done == 1

    @FUSE_MODES
    def test_policy_updates_once_warmup_steps_are_done(self, fuse: bool) -> None:
        # Arrange
        agent = make_warmup_ppo(fuse_actor_critic_pass=fuse, critic_warmup_steps=2)
        learn_rows(agent)
        learn_rows(agent)
        before = role_weights(agent)

        # Act
        metrics = learn_rows(agent)

        # Assert
        after = role_weights(agent)
        assert changed(before["actor"], after["actor"])
        assert changed(before["critic"], after["critic"])
        assert metrics["critic_warmup"] == 0.0
        assert agent.critic_warmup_steps_done == 2

    def test_no_warmup_by_default(self) -> None:
        # Arrange
        agent = make_warmup_ppo()
        before = role_weights(agent)

        # Act
        metrics = learn_rows(agent)

        # Assert
        assert agent.critic_warmup_steps == 0
        assert changed(before["actor"], role_weights(agent)["actor"])
        assert metrics["critic_warmup"] == 0.0

    def test_warmup_targets_are_monte_carlo_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = make_warmup_ppo(critic_warmup_steps=1, gae_lambda=0.5)
        lambdas: list[float] = []
        for name in ("_compute_gae_returns", "_compute_gae_returns_token"):
            compute = getattr(agent, name)

            def recording(*args: Any, compute: Any = compute) -> Any:
                lambdas.append(args[-1])
                return compute(*args)

            monkeypatch.setattr(agent, name, recording)

        # Act
        learn_rows(agent)
        learn_rows(agent)

        # Assert
        assert lambdas == [1.0, 0.5]

    @pytest.mark.parametrize("critic_warmup_steps", [0, 1])
    def test_reports_value_fit_every_step(self, critic_warmup_steps: int) -> None:
        # Arrange
        agent = make_warmup_ppo(critic_warmup_steps=critic_warmup_steps)

        # Act
        metrics = learn_rows(agent)

        # Assert
        assert math.isfinite(metrics["explained_variance"])
        assert -1.0 <= metrics["value_return_corr"] <= 1.0


def _data_parallel_warmup_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        # Arrange — four rows per rank
        agent = make_ppo(
            batch_size=4 * world_size,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
            fuse_actor_critic_pass=False,
            critic_warmup_steps=1,
        )
        before = role_weights(agent)

        # Act
        metrics = learn_rows(agent)

        # Assert
        after = role_weights(agent)
        assert not changed(before["actor"], after["actor"])
        assert changed(before["critic"], after["critic"])
        assert metrics["critic_warmup"] == 1.0
        assert metrics["grad_norm_pre"] == 0.0
        assert math.isfinite(metrics["critic_grad_norm_pre"])
        assert metrics["critic_grad_norm_pre"] > 0.0
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestPPOLearnCriticWarmupDataParallel:
    def test_warmup_step_syncs_critic_grads_and_leaves_the_actor(self) -> None:
        _spawn_ranks(_data_parallel_warmup_worker)


class TestPPOCriticWarmupCheckpoint:
    @pytest.mark.parametrize("restore_config", [True, False])
    def test_resumes_mid_warmup(self, tmp_path: Path, restore_config: bool) -> None:
        # Arrange
        saved = make_warmup_ppo(critic_warmup_steps=2)
        learn_rows(saved)
        saved.save_checkpoint(str(tmp_path))
        loaded = make_warmup_ppo(critic_warmup_steps=2)

        # Act
        loaded.load_checkpoint(str(tmp_path), restore_config=restore_config)

        # Assert
        assert loaded.critic_warmup_steps_done == 1
        before = role_weights(loaded)
        critic_lora = {
            name: weight for name, weight in before["critic"].items() if "lora_" in name
        }
        assert learn_rows(loaded)["critic_warmup"] == 1.0
        after = role_weights(loaded)
        assert critic_lora
        assert changed(critic_lora, after["critic"])
        assert not changed(before["actor"], after["actor"])
        assert learn_rows(loaded)["critic_warmup"] == 0.0


def make_fsdp_warmup_ppo() -> LLMPPO:
    return make_ppo(
        batch_size=4,
        mini_batch_size=2,
        micro_batch_size_per_gpu=1,
        wrap=True,
        fsdp_config=FSDPConfig(optim_cpu_offload=False, param_dtype="float32"),
        critic_warmup_steps=2,
    )


def optimizer_states(
    agent: LLMPPO,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Full optimizer state of each trainable param, by role and param name."""
    names = {param: name for name, param in agent.actor.named_parameters()}
    inner = agent.optimizer._single_optimizer()
    states: dict[str, dict[str, dict[str, torch.Tensor]]] = {"actor": {}, "critic": {}}
    for group in inner.param_groups:
        role = "actor" if group["group"].startswith("actor") else "critic"
        for param in group["params"]:
            states[role][names[param]] = {
                key: value.full_tensor() if isinstance(value, DTensor) else value
                for key, value in inner.state.get(param, {}).items()
            }
    return states


def full_role_weights(agent: LLMPPO) -> dict[str, dict[str, torch.Tensor]]:
    """Unsharded :func:`role_weights` of an FSDP2 agent."""
    return {
        role: {
            name: weight.full_tensor() if isinstance(weight, DTensor) else weight
            for name, weight in weights.items()
        }
        for role, weights in role_weights(agent).items()
    }


def _fsdp_warmup_resume_worker(
    path: str,
    lora_only: bool,
    restore_config: bool,
    rank: int,
    world_size: int,
    port: int,
    result_queue: Any,
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        # The default FSDP2 mesh follows the host accelerator (MPS on macOS).
        torch._C._get_accelerator = lambda: torch.device("cpu")
        rows = slice(2 * rank, 2 * rank + 2)
        # Arrange — checkpoint after the first of two critic-only steps
        saved = make_fsdp_warmup_ppo()
        seed_lora_weights(saved)
        learn_rows(saved, rows)
        saved_states = optimizer_states(saved)
        saved_weights = full_role_weights(saved)
        saved.save_checkpoint(path, lora_only=lora_only)
        loaded = make_fsdp_warmup_ppo()
        fresh_weights = full_role_weights(loaded)

        # Act
        loaded.load_checkpoint(path, load_optimizer=True, restore_config=restore_config)

        # Assert
        loaded_weights = full_role_weights(loaded)
        for role in ("actor", "critic"):
            assert changed(fresh_weights[role], saved_weights[role])
            assert loaded_weights[role].keys() == saved_weights[role].keys()
            for name, weight in saved_weights[role].items():
                assert torch.equal(loaded_weights[role][name], weight), name
        loaded_states = optimizer_states(loaded)
        assert loaded.critic_warmup_steps_done == 1
        assert saved_states["actor"]
        assert not any(saved_states["actor"].values())
        assert not any(loaded_states["actor"].values())
        assert loaded_states["critic"].keys() == saved_states["critic"].keys()
        for name, state in saved_states["critic"].items():
            assert state, name
            assert loaded_states["critic"][name].keys() == state.keys(), name
            for key, value in state.items():
                assert torch.equal(loaded_states["critic"][name][key], value), (
                    name,
                    key,
                )
        assert learn_rows(loaded, rows)["critic_warmup"] == 1.0
        assert not any(optimizer_states(loaded)["actor"].values())
        assert learn_rows(loaded, rows)["critic_warmup"] == 0.0
        assert all(optimizer_states(loaded)["actor"].values())
        result_queue.put((rank, "ok", None))
    except Exception:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestPPOCriticWarmupCheckpointFSDP:
    @pytest.mark.parametrize("lora_only", [True, False], ids=["lora", "full"])
    @pytest.mark.parametrize("restore_config", [True, False])
    def test_resumes_mid_warmup_with_optimizer_state(
        self, tmp_path: Path, lora_only: bool, restore_config: bool
    ) -> None:
        _spawn_ranks(
            partial(
                _fsdp_warmup_resume_worker, str(tmp_path), lora_only, restore_config
            )
        )


class TestPPOCriticWarmupLRSchedule:
    def test_actor_schedule_starts_after_warmup_and_critic_from_step_zero(
        self,
    ) -> None:
        # Arrange: 12 steps, 25% warmup. The critic warms up over steps 0-2; the
        # actor holds at 0 for the 2 critic-only steps, then warms up over 2 of
        # its 10 steps.
        agent = make_warmup_ppo(
            critic_warmup_steps=2,
            lr_actor=1e-4,
            lr_critic=1e-3,
            cosine_lr_schedule_config=CosineLRScheduleConfig(
                num_steps=12, warmup_proportion=0.25
            ),
        )
        param_groups = agent.optimizer._single_optimizer().param_groups

        def lrs() -> dict[str, float]:
            return {group["group"]: group["lr"] for group in param_groups}

        # Act
        seen = [lrs()]
        for _ in range(3):
            learn_rows(agent)
            seen.append(lrs())

        # Assert
        assert seen == [
            pytest.approx({"actor": 0.0, "critic": 1e-3 / 3}),
            pytest.approx({"actor": 0.0, "critic": 2e-3 / 3}),
            pytest.approx({"actor": 0.5e-4, "critic": 1e-3}),
            pytest.approx({"actor": 1e-4, "critic": 1e-3}),
        ]
        assert agent.current_lr == pytest.approx(1e-4)

    def test_schedule_without_warmup_steps_starts_both_at_step_zero(self) -> None:
        # Arrange
        agent = make_warmup_ppo(
            lr_actor=1e-4,
            lr_critic=1e-3,
            cosine_lr_schedule_config=CosineLRScheduleConfig(
                num_steps=12, warmup_proportion=0.25
            ),
        )
        param_groups = agent.optimizer._single_optimizer().param_groups

        # Act
        learn_rows(agent)

        # Assert
        assert {group["group"]: group["lr"] for group in param_groups} == (
            pytest.approx({"actor": 2e-4 / 3, "critic": 2e-3 / 3})
        )

    def test_rejects_warmup_steps_past_the_schedule(self) -> None:
        with pytest.raises(ValueError, match="actor_start_step"):
            make_warmup_ppo(
                critic_warmup_steps=12,
                cosine_lr_schedule_config=CosineLRScheduleConfig(num_steps=12),
            )
