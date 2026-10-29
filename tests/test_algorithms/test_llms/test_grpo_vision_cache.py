# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO learn runs a frozen vision tower once per vision row."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
import pytest
import torch
from peft import LoraConfig
from torch.overrides import TorchFunctionMode

from agilerl.algorithms import GRPO
from agilerl.algorithms.core import base
from agilerl.algorithms.core.llm_ops import vision_lockstep as vision_lockstep_module
from agilerl.distributed.process import allreduce_sum_ints, any_rank
from agilerl.utils.vision_rows import VisionRows
from tests.test_algorithms.test_llms.vision_helpers import (
    IMAGE_COUNTS,
    PAD_TOKEN_ID,
    PIXEL_DIM,
    TowerVisionModel,
    VisionRowsLifetime,
    adapter_weights,
    learn_on_vision_episodes,
    tower,
    vision_model,
)


def make_vision_grpo(**overrides: Any) -> GRPO:
    """Tiny fp32 GRPO on CPU with a KL reference pass over 4 episodes."""
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "actor_network": TowerVisionModel(),
        "pad_token_id": PAD_TOKEN_ID,
        "pad_token": "<pad>",
        "batch_size": 2,
        "group_size": 2,
        "beta": 0.01,
        "lr": 1e-2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "micro_batch_size_per_gpu": 2,
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
        ),
        **overrides,
    }
    agent = GRPO(**kwargs)
    agent.liger_loss = False
    return agent


class TestGRPOLearnVisionCache:
    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_learn_runs_the_tower_once_per_vision_row(self, segmented: bool) -> None:
        # Arrange
        agent = make_vision_grpo()

        # Act
        learn_on_vision_episodes(agent, segmented=segmented)

        # Assert
        assert sum(tower(agent).rows) == sum(IMAGE_COUNTS)

    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_learn_matches_the_uncached_tower(
        self, monkeypatch: pytest.MonkeyPatch, segmented: bool
    ) -> None:
        # Arrange
        cached = make_vision_grpo()
        uncached = make_vision_grpo()
        monkeypatch.setattr(uncached._vision_cache, "step", nullcontext)

        # Act
        cached_metrics = learn_on_vision_episodes(cached, segmented=segmented)
        uncached_metrics = learn_on_vision_episodes(uncached, segmented=segmented)

        # Assert: fp32 tower GEMMs over other batches may round differently in
        # the last bits.
        assert sum(tower(uncached).rows) > sum(tower(cached).rows)
        for key in ("loss", "kl"):
            assert math.isfinite(cached_metrics[key]), key
            assert math.isclose(
                cached_metrics[key], uncached_metrics[key], rel_tol=0.0, abs_tol=1e-6
            ), key
        cached_weights = adapter_weights(cached, "actor")
        uncached_weights = adapter_weights(uncached, "actor")
        assert cached_weights.keys() == uncached_weights.keys()
        for name, weight in cached_weights.items():
            assert torch.allclose(
                weight, uncached_weights[name], rtol=0.0, atol=1e-6
            ), name


class PixelMoves(TorchFunctionMode):
    """Vision rows of every vision tensor moved with ``Tensor.to(device)``."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[int] = []

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if (
            func is torch.Tensor.to
            and args[0].dim() == 2
            and args[0].shape[1] == PIXEL_DIM
            and args[0].is_floating_point()
            and (
                "device" in kwargs
                or any(isinstance(arg, (str, torch.device)) for arg in args[1:])
            )
        ):
            self.rows.append(int(args[0].shape[0]))
        return func(*args, **kwargs)


class TestGRPOLearnPixelTransfer:
    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_learn_moves_one_micro_batch_of_vision_rows_at_a_time(
        self, segmented: bool
    ) -> None:
        # Arrange: one training row per micro-batch, and a reference pass.
        agent = make_vision_grpo(micro_batch_size_per_gpu=1)
        moves = PixelMoves()

        # Act
        with moves:
            metrics = learn_on_vision_episodes(agent, segmented=segmented)

        # Assert: no move carries more than one row's vision rows, and the
        # tower still runs once per vision row.
        assert math.isfinite(metrics["loss"])
        assert moves.rows
        assert max(moves.rows) <= max(IMAGE_COUNTS)
        assert sum(tower(agent).rows) == sum(IMAGE_COUNTS)


class PixelCopies(PixelMoves):
    """Vision rows moved from tensors that hold pixels, not :meth:`VisionFeatureCache.stand_in` views."""

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        moves = len(self.rows)
        result = super().__torch_function__(func, types, args, kwargs)
        if (
            len(self.rows) > moves
            and args[0].untyped_storage().nbytes() == args[0].element_size()
        ):
            self.rows.pop()
        return result


class TestGRPOLearnNoGradPixelCopies:
    def test_cached_images_copy_no_vision_rows(self) -> None:
        # Arrange: the reference pass caches every image the actor pass reads.
        agent = make_vision_grpo(micro_batch_size_per_gpu=1)
        copies = PixelCopies()

        # Act
        with copies:
            metrics = learn_on_vision_episodes(agent, segmented=False)

        # Assert
        assert math.isfinite(metrics["loss"])
        assert sum(copies.rows) == sum(IMAGE_COUNTS)
        assert sum(tower(agent).rows) == sum(IMAGE_COUNTS)


class LearnPixelEvents(PixelMoves):
    """Vision row moves and vision forwards of a learn's gradient passes, in order."""

    def __init__(self, agent: GRPO, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__()
        self.events: list[str] = []
        # Whether each move outside the no-grad passes reads pinned memory without blocking.
        self.staged: list[bool] = []
        self.in_no_grad_pass = False
        no_grad_pass = agent._fused_forward_no_grad

        def marked_no_grad_pass(*args: Any, **kwargs: Any) -> Any:
            self.in_no_grad_pass = True
            try:
                return no_grad_pass(*args, **kwargs)
            finally:
                self.in_no_grad_pass = False

        monkeypatch.setattr(agent, "_fused_forward_no_grad", marked_no_grad_pass)
        vision_model(agent).register_forward_pre_hook(lambda *_: self.record("forward"))

    def record(self, event: str) -> None:
        if not self.in_no_grad_pass:
            self.events.append(event)

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        moves = len(self.rows)
        result = super().__torch_function__(func, types, args, kwargs)
        if len(self.rows) > moves:
            self.record("move")
            if not self.in_no_grad_pass:
                self.staged.append(
                    args[0].is_pinned() and bool((kwargs or {}).get("non_blocking"))
                )
        return result


class TestGRPOLearnVisionRowMoves:
    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_warm_cache_moves_no_vision_rows_for_the_gradient_passes(
        self, monkeypatch: pytest.MonkeyPatch, segmented: bool
    ) -> None:
        # Arrange
        agent = make_vision_grpo(micro_batch_size_per_gpu=1)
        events = LearnPixelEvents(agent, monkeypatch)

        # Act
        with events:
            metrics = learn_on_vision_episodes(agent, segmented=segmented)

        # Assert
        assert math.isfinite(metrics["loss"])
        assert events.events.count("forward") >= 2
        assert "move" not in events.events
        assert sum(tower(agent).rows) == sum(IMAGE_COUNTS)

    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_cold_cache_moves_each_micro_batch_right_before_its_forward(
        self, monkeypatch: pytest.MonkeyPatch, segmented: bool
    ) -> None:
        # Arrange
        agent = make_vision_grpo(micro_batch_size_per_gpu=1)
        monkeypatch.setattr(agent._vision_cache, "step", nullcontext)
        events = LearnPixelEvents(agent, monkeypatch)

        # Act
        with events:
            metrics = learn_on_vision_episodes(agent, segmented=segmented)

        # Assert
        forwards = events.events.count("forward")
        assert math.isfinite(metrics["loss"])
        assert forwards >= 2
        assert events.events == ["move", "forward"] * forwards

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cold_cache_copies_from_pinned_memory_without_blocking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = make_vision_grpo(micro_batch_size_per_gpu=1, device="cuda")
        monkeypatch.setattr(agent._vision_cache, "step", nullcontext)
        events = LearnPixelEvents(agent, monkeypatch)

        # Act
        with events:
            metrics = learn_on_vision_episodes(agent, segmented=True)

        # Assert
        assert math.isfinite(metrics["loss"])
        assert len(events.staged) == events.events.count("forward")
        assert all(events.staged)

    @pytest.mark.parametrize("cached", [True, False], ids=["warm", "cold"])
    def test_learn_frees_each_micro_batch_vision_rows_before_backward(
        self, monkeypatch: pytest.MonkeyPatch, cached: bool
    ) -> None:
        # Arrange
        agent = make_vision_grpo(micro_batch_size_per_gpu=1)
        if not cached:
            monkeypatch.setattr(agent._vision_cache, "step", nullcontext)
        lifetime = VisionRowsLifetime(agent, monkeypatch)

        # Act
        learn_on_vision_episodes(agent, segmented=True)

        # Assert
        assert len(lifetime.freed) >= 2
        assert all(lifetime.freed)


class LockstepAgreements:
    """Lockstep all-reduces of a learn whose one rank keeps its tower calls in lockstep."""

    def __init__(self, agent: GRPO, monkeypatch: pytest.MonkeyPatch) -> None:
        self.agent = agent
        self.no_grad: list[str] = []
        self.no_grad_forwards = 0
        self.in_no_grad_pass = False
        no_grad_pass = agent._fused_forward_no_grad

        def marked_no_grad_pass(*args: Any, **kwargs: Any) -> Any:
            self.in_no_grad_pass = True
            try:
                return no_grad_pass(*args, **kwargs)
            finally:
                self.in_no_grad_pass = False

        def counted(kind: str, agree: Callable[[Any], Any]) -> Callable[[Any], Any]:
            def agreement(value: Any) -> Any:
                if self.in_no_grad_pass:
                    self.no_grad.append(kind)
                return agree(value)

            return agreement

        def no_grad_forward(*_: Any) -> None:
            self.no_grad_forwards += self.in_no_grad_pass

        monkeypatch.setattr(agent, "_fused_forward_no_grad", marked_no_grad_pass)
        monkeypatch.setattr(agent, "_vision_step", self.lockstep_step)
        monkeypatch.setattr(
            vision_lockstep_module,
            "allreduce_sum_ints",
            counted("plan", allreduce_sum_ints),
        )
        monkeypatch.setattr(
            vision_lockstep_module, "any_rank", counted("forward", any_rank)
        )
        vision_model(agent).register_forward_pre_hook(no_grad_forward)

    @contextmanager
    def lockstep_step(self, _pixel_values: torch.Tensor | None) -> Iterator[None]:
        cache = self.agent._vision_cache
        with (
            cache.step(),
            vision_lockstep_module.vision_lockstep(
                vision_model(self.agent), cache
            ) as lockstep,
        ):
            self.agent._vision_lockstep = lockstep
            try:
                yield
            finally:
                self.agent._vision_lockstep = None


class TestGRPOLearnVisionLockstep:
    def test_no_grad_passes_agree_on_their_tower_calls_up_front(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = make_vision_grpo(micro_batch_size_per_gpu=1)
        agreements = LockstepAgreements(agent, monkeypatch)

        # Act
        metrics = learn_on_vision_episodes(agent, segmented=False)

        # Assert
        assert math.isfinite(metrics["loss"])
        assert set(agreements.no_grad) == {"plan"}
        assert len(agreements.no_grad) < agreements.no_grad_forwards
        assert sum(tower(agent).rows) == sum(IMAGE_COUNTS)

    def test_images_every_rank_agreed_to_reuse_copy_no_vision_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = make_vision_grpo(micro_batch_size_per_gpu=1)
        LockstepAgreements(agent, monkeypatch)
        copies = PixelCopies()

        # Act
        with copies:
            metrics = learn_on_vision_episodes(agent, segmented=False)

        # Assert
        assert math.isfinite(metrics["loss"])
        assert sum(copies.rows) < 2 * sum(IMAGE_COUNTS)
        assert sum(tower(agent).rows) == sum(IMAGE_COUNTS)


class TestGRPOLearnVisionRows:
    def test_cached_images_get_stand_ins_without_pixels(self) -> None:
        # Arrange: rows 0, 1, 2 hold 2, 1 and 3 vision rows, all cached.
        agent = make_vision_grpo()
        pixel_values = torch.arange(24.0).reshape(6, PIXEL_DIM)
        vision_rows = VisionRows(
            pixel_values, image_counts=[2, 1, 3], image_keys=torch.arange(6)
        )
        tower(agent).train()

        # Act
        with agent._vision_cache.step():
            with agent._vision_cache.images(torch.arange(6)):
                tower(agent)(pixel_values)
            micro_batches = list(
                agent._learn_vision_rows(
                    vision_rows, [np.array([2, 0]), np.array([1])], 3
                )
            )

        # Assert
        [(first, first_keys), (second, second_keys)] = micro_batches
        assert first.shape == (5, PIXEL_DIM)
        assert second.shape == (1, PIXEL_DIM)
        assert first.untyped_storage().nbytes() == pixel_values.element_size()
        assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
        assert first_keys.tolist() == [3, 4, 5, 0, 1]
        assert second_keys.tolist() == [2]

    def test_partly_cached_images_get_their_vision_rows(self) -> None:
        # Arrange: rows 0, 1, 2 hold 2, 1 and 3 vision rows; row 2's are not cached.
        agent = make_vision_grpo()
        pixel_values = torch.arange(24.0).reshape(6, PIXEL_DIM)
        vision_rows = VisionRows(
            pixel_values, image_counts=[2, 1, 3], image_keys=torch.arange(6)
        )
        tower(agent).train()

        # Act
        with agent._vision_cache.step():
            with agent._vision_cache.images(torch.arange(3)):
                tower(agent)(pixel_values[:3])
            micro_batches = list(
                agent._learn_vision_rows(
                    vision_rows, [np.array([2, 0]), np.array([1])], 3
                )
            )

        # Assert
        [(first, first_keys), (second, second_keys)] = micro_batches
        assert torch.equal(first, pixel_values[[3, 4, 5, 0, 1]])
        assert torch.equal(second, pixel_values[[2]])
        assert first_keys.tolist() == [3, 4, 5, 0, 1]
        assert second_keys.tolist() == [2]

    def test_a_miss_on_another_rank_gets_every_rank_its_vision_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: every image is cached on this rank; another rank misses one.
        agent = make_vision_grpo()
        pixel_values = torch.arange(24.0).reshape(6, PIXEL_DIM)
        vision_rows = VisionRows(
            pixel_values, image_counts=[2, 1, 3], image_keys=torch.arange(6)
        )
        tower(agent).train()
        monkeypatch.setattr(agent, "_vision_language_module", vision_model(agent))
        monkeypatch.setattr(base, "get_world_size", lambda: 2)
        monkeypatch.setattr(base, "any_rank", lambda _flag: True)

        # Act
        with agent._vision_cache.step():
            with agent._vision_cache.images(torch.arange(6)):
                tower(agent)(pixel_values)
            micro_batches = list(
                agent._learn_vision_rows(
                    vision_rows, [np.array([2, 0]), np.array([1])], 3
                )
            )

        # Assert
        [(first, first_keys), (second, second_keys)] = micro_batches
        assert torch.equal(first, pixel_values[[3, 4, 5, 0, 1]])
        assert torch.equal(second, pixel_values[[2]])
        assert first_keys.tolist() == [3, 4, 5, 0, 1]
        assert second_keys.tolist() == [2]

    def test_an_engine_without_an_image_tower_skips_the_rank_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: every image is cached; the engine makes no image rollouts.
        agent = make_vision_grpo()
        pixel_values = torch.arange(24.0).reshape(6, PIXEL_DIM)
        vision_rows = VisionRows(
            pixel_values, image_counts=[2, 1, 3], image_keys=torch.arange(6)
        )
        tower(agent).train()
        monkeypatch.setattr(agent, "_vision_language_module", vision_model(agent))
        monkeypatch.setattr(agent, "vllm_config", None)
        monkeypatch.setattr(agent, "rollout_strip_multimodal_towers", True)
        monkeypatch.setattr(base, "get_world_size", lambda: 2)

        def no_rank_check(_flag: bool) -> bool:
            msg = "ranks checked each other"
            raise AssertionError(msg)

        monkeypatch.setattr(base, "any_rank", no_rank_check)

        # Act
        with agent._vision_cache.step():
            with agent._vision_cache.images(torch.arange(6)):
                tower(agent)(pixel_values)
            micro_batches = list(
                agent._learn_vision_rows(
                    vision_rows, [np.array([2, 0]), np.array([1])], 3
                )
            )

        # Assert
        [(first, first_keys), (second, _)] = micro_batches
        assert first.shape == (5, PIXEL_DIM)
        assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
        assert first_keys.tolist() == [3, 4, 5, 0, 1]
