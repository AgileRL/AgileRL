# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Vision lockstep runs the tower on every rank's forward or on none, without changing outputs."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from agilerl.algorithms.core.llm_ops.frozen_vision import (
    VisionFeatureCache,
    install_frozen_vision_no_grad,
)
from agilerl.algorithms.core.llm_ops.vision_lockstep import (
    VisionLockstep,
    vision_language_module,
    vision_lockstep,
)
from agilerl.architectures.gemma4 import install_gemma4_vision_patches
from agilerl.architectures.qwen3_5.vision import install_qwen3_5_vl_patches
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.distributed import fsdp_blocks
from agilerl.distributed.fsdp_blocks import apply_fsdp2
from agilerl.distributed.process import allreduce_sum_ints, any_rank
from tests.rank_launch import run_ranks
from tests.test_architectures import test_gemma4 as gemma4
from tests.test_architectures.qwen3_5.vision_helpers import (
    GRID,
    PATCH_DIM,
    image_prompt,
    restored_qwen_vl_model,
    tiny_qwen_vl,
)
from tests.torch_kernels import use_torch_kernels

TEXT_IDS = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
IMAGE_IDS = torch.tensor([image_prompt([3], images=1)])


def count_tower_runs(model: nn.Module) -> list[int]:
    """Log one entry per real tower forward; cache hits skip the first block."""
    runs: list[int] = []
    model.model.visual.blocks[0].register_forward_hook(
        lambda _module, _args, _output: runs.append(1)
    )
    return runs


class TensorVisionModel(nn.Module):
    """VL stand-in whose forward returns a tensor and counts filler-image tower runs."""

    def __init__(self) -> None:
        super().__init__()
        self.filler_runs = 0

    def forward(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None
    ) -> torch.Tensor:
        return input_ids.float()

    def get_decoder(self) -> nn.Module:
        return self

    def vision_filler_features(self, device: torch.device) -> torch.Tensor:
        self.filler_runs += 1
        return torch.ones(1, device=device)


@pytest.fixture
def qwen_vl(torch_kernels):
    """Tiny Qwen3.5 VL model with vision-row patches and a tower run log."""
    with restored_qwen_vl_model():
        install_qwen3_5_vl_patches(PatchRuntimeConfig())
        model = tiny_qwen_vl()
        yield model, count_tower_runs(model)


@pytest.fixture
def frozen_qwen_vl(torch_kernels):
    """Tiny Qwen3.5 VL model whose frozen tower reuses a feature cache."""
    with restored_qwen_vl_model():
        install_qwen3_5_vl_patches(PatchRuntimeConfig())
        model = tiny_qwen_vl()
        model.model.visual.requires_grad_(False)
        cache = VisionFeatureCache()
        install_frozen_vision_no_grad(model, cache)
        yield model, cache, count_tower_runs(model)


@pytest.fixture
def another_rank_runs_the_tower(monkeypatch):
    """Every forward's rank agreement finds a rank that must run the tower."""
    monkeypatch.setattr(
        "agilerl.algorithms.core.llm_ops.vision_lockstep.any_rank", lambda _flag: True
    )


def mixed_rank_worker(rank: int, world_size: int) -> bool:
    """Shard the tiny VL model, give only rank 0 an image, and train one step."""
    use_torch_kernels(pytest.MonkeyPatch())
    install_qwen3_5_vl_patches(PatchRuntimeConfig())
    model = tiny_qwen_vl().train()
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        apply_fsdp2(
            model,
            FSDPConfig(
                param_dtype="float32",
                reduce_dtype="float32",
            ),
            mesh=init_device_mesh("cpu", (world_size,)),
        )
    if rank == 0:
        input_ids = IMAGE_IDS
        pixel_values = torch.rand(1, GRID, GRID, PATCH_DIM)
    else:
        input_ids, pixel_values = TEXT_IDS, None
    with vision_lockstep(vision_language_module(model), VisionFeatureCache()):
        logits = model(input_ids=input_ids, pixel_values=pixel_values).logits
        logits.float().pow(2).mean().backward()
    return all(param.grad is not None for param in model.model.visual.parameters())


def mixed_rank_gemma4_worker(rank: int, world_size: int) -> bool:
    """Shard a Gemma4 model with per-layer embeddings, give only rank 0 an image, and train one step."""
    use_torch_kernels(pytest.MonkeyPatch())
    install_gemma4_vision_patches(PatchRuntimeConfig())
    model = gemma4.tiny_gemma4().train()
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        apply_fsdp2(
            model,
            FSDPConfig(
                param_dtype="float32",
                reduce_dtype="float32",
            ),
            mesh=init_device_mesh("cpu", (world_size,)),
        )
    if rank == 0:
        image = [gemma4.BOI, *([gemma4.IMAGE] * gemma4.TOKENS_PER_IMAGE), gemma4.EOI]
        input_ids = torch.tensor([[3, *image, 5]])
        pixel_values = torch.rand(1, gemma4.GRID, gemma4.GRID, gemma4.PATCH_DIM)
    else:
        input_ids, pixel_values = TEXT_IDS, None
    with vision_lockstep(vision_language_module(model), VisionFeatureCache()):
        logits = model(input_ids=input_ids, pixel_values=pixel_values).logits
        logits.float().pow(2).mean().backward()
    return all(
        param.grad is not None for param in model.model.vision_tower.parameters()
    )


def cached_mixed_rank_worker(rank: int, world_size: int) -> tuple[int, bool]:
    """Shard a frozen-tower VL model, give only rank 0 an image, and forward it twice in one step."""
    use_torch_kernels(pytest.MonkeyPatch())
    install_qwen3_5_vl_patches(PatchRuntimeConfig())
    model = tiny_qwen_vl().train()
    model.model.visual.requires_grad_(False)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        apply_fsdp2(
            model,
            FSDPConfig(
                param_dtype="float32",
                reduce_dtype="float32",
            ),
            mesh=init_device_mesh("cpu", (world_size,)),
        )
    cache = VisionFeatureCache()
    install_frozen_vision_no_grad(model, cache)
    runs = count_tower_runs(model)
    if rank == 0:
        input_ids, keys = IMAGE_IDS, torch.arange(1)
        pixel_values = torch.rand(1, GRID, GRID, PATCH_DIM)
    else:
        input_ids, pixel_values, keys = TEXT_IDS, None, None
    with (
        cache.step(),
        vision_lockstep(vision_language_module(model), cache),
        cache.images(keys),
    ):
        with torch.no_grad():
            first = model(input_ids=input_ids, pixel_values=pixel_values).logits
        second = model(input_ids=input_ids, pixel_values=pixel_values).logits
        second.float().pow(2).mean().backward()
    return len(runs), torch.equal(first, second.detach())


def count_agreements(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Log each all-reduce of the lockstep: ``"plan"`` or ``"forward"``."""
    agreements: list[str] = []

    def counted_sum(values: list[int]) -> list[int]:
        agreements.append("plan")
        return allreduce_sum_ints(values)

    def counted_any(flag: bool) -> bool:
        agreements.append("forward")
        return any_rank(flag)

    lockstep_module = "agilerl.algorithms.core.llm_ops.vision_lockstep"
    monkeypatch.setattr(f"{lockstep_module}.allreduce_sum_ints", counted_sum)
    monkeypatch.setattr(f"{lockstep_module}.any_rank", counted_any)
    return agreements


def reference_then_actor_pass(
    model: nn.Module,
    cache: VisionFeatureCache,
    lockstep: VisionLockstep,
    images: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Run two one-image forwards, then two more on the same images, as one plan."""
    keys = [[0], [1], [0], [1]] if images is not None else [[]] * 4
    logits = []
    with torch.no_grad(), lockstep.plan(keys):
        for forward_keys in keys:
            with cache.images(torch.tensor(forward_keys) if forward_keys else None):
                if images is None:
                    logits.append(model(input_ids=TEXT_IDS).logits)
                else:
                    logits.append(
                        model(
                            input_ids=IMAGE_IDS,
                            pixel_values=images[forward_keys[0]][None],
                        ).logits
                    )
    return logits


def planned_mixed_rank_worker(
    rank: int, world_size: int
) -> tuple[list[str], int, bool]:
    """Shard a frozen-tower VL model and plan two passes over rank 0's two images."""
    monkeypatch = pytest.MonkeyPatch()
    use_torch_kernels(monkeypatch)
    install_qwen3_5_vl_patches(PatchRuntimeConfig())
    model = tiny_qwen_vl().train()
    model.model.visual.requires_grad_(False)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        apply_fsdp2(
            model,
            FSDPConfig(
                param_dtype="float32",
                reduce_dtype="float32",
            ),
            mesh=init_device_mesh("cpu", (world_size,)),
        )
    cache = VisionFeatureCache()
    install_frozen_vision_no_grad(model, cache)
    runs = count_tower_runs(model)
    agreements = count_agreements(monkeypatch)
    images = torch.rand(2, GRID, GRID, PATCH_DIM) if rank == 0 else None
    with (
        cache.step(),
        vision_lockstep(vision_language_module(model), cache) as lockstep,
    ):
        logits = reference_then_actor_pass(model, cache, lockstep, images)
    reused = all(
        torch.equal(first, again)
        for first, again in zip(logits[:2], logits[2:], strict=True)
    )
    return agreements, len(runs), reused


class TestVisionLanguageModule:
    def test_finds_the_patched_vl_model(self, qwen_vl) -> None:
        model, _ = qwen_vl

        assert vision_language_module(model) is model.model

    def test_text_model_has_none(self) -> None:
        assert vision_language_module(nn.Sequential(nn.Linear(2, 2))) is None


class TestVisionLockstep:
    def test_text_forward_runs_the_tower_when_another_rank_does(
        self, qwen_vl, another_rank_runs_the_tower
    ) -> None:
        # Arrange
        model, tower_runs = qwen_vl
        with torch.no_grad():
            expected = model(input_ids=TEXT_IDS).logits
        tower_runs.clear()

        # Act
        with torch.no_grad(), vision_lockstep(model.model, VisionFeatureCache()):
            actual = model(input_ids=TEXT_IDS).logits

        # Assert
        assert len(tower_runs) == 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_text_forward_skips_the_tower_when_no_rank_needs_it(self, qwen_vl) -> None:
        # Arrange
        model, tower_runs = qwen_vl
        with torch.no_grad():
            expected = model(input_ids=TEXT_IDS).logits

        # Act
        with torch.no_grad(), vision_lockstep(model.model, VisionFeatureCache()):
            actual = model(input_ids=TEXT_IDS).logits

        # Assert
        assert tower_runs == []
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_text_backward_reaches_the_tower(
        self, qwen_vl, another_rank_runs_the_tower
    ) -> None:
        # Arrange
        model, _ = qwen_vl

        # Act
        with vision_lockstep(model.model, VisionFeatureCache()):
            model(input_ids=TEXT_IDS).logits.pow(2).mean().backward()

        # Assert
        assert all(param.grad is not None for param in model.model.visual.parameters())

    def test_image_forward_runs_the_tower_once(self, qwen_vl) -> None:
        # Arrange
        model, tower_runs = qwen_vl
        pixel_values = torch.rand(1, GRID, GRID, PATCH_DIM)
        with torch.no_grad():
            expected = model(input_ids=IMAGE_IDS, pixel_values=pixel_values).logits
        tower_runs.clear()

        # Act
        with torch.no_grad(), vision_lockstep(model.model, VisionFeatureCache()):
            actual = model(input_ids=IMAGE_IDS, pixel_values=pixel_values).logits

        # Assert
        assert len(tower_runs) == 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_pixel_values_run_the_filler(
        self, qwen_vl, another_rank_runs_the_tower
    ) -> None:
        # Arrange
        model, tower_runs = qwen_vl
        with torch.no_grad():
            expected = model(input_ids=TEXT_IDS).logits
        tower_runs.clear()

        # Act
        with torch.no_grad(), vision_lockstep(model.model, VisionFeatureCache()):
            actual = model(
                input_ids=TEXT_IDS,
                pixel_values=torch.empty(0, GRID, GRID, PATCH_DIM),
            ).logits

        # Assert
        assert len(tower_runs) == 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_embeds_forward_runs_the_tower_and_keeps_hidden_states(
        self, qwen_vl, another_rank_runs_the_tower
    ) -> None:
        # Arrange
        model, tower_runs = qwen_vl
        embeds = model.model.get_input_embeddings()(TEXT_IDS)
        with torch.no_grad():
            expected = model.model(inputs_embeds=embeds).last_hidden_state
        tower_runs.clear()

        # Act
        with torch.no_grad(), vision_lockstep(model.model, VisionFeatureCache()):
            actual = model.model(inputs_embeds=embeds).last_hidden_state

        # Assert
        assert len(tower_runs) == 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tuple_output_runs_the_tower_and_keeps_hidden_states(
        self, qwen_vl, another_rank_runs_the_tower
    ) -> None:
        # Arrange
        model, tower_runs = qwen_vl
        with torch.no_grad():
            expected = model.model(input_ids=TEXT_IDS, return_dict=False)
        tower_runs.clear()

        # Act
        with torch.no_grad(), vision_lockstep(model.model, VisionFeatureCache()):
            actual = model.model(input_ids=TEXT_IDS, return_dict=False)

        # Assert
        assert isinstance(actual, tuple)
        assert len(tower_runs) == 1
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)

    def test_tensor_output_runs_the_filler_and_keeps_its_values(
        self, another_rank_runs_the_tower
    ) -> None:
        # Arrange
        module = TensorVisionModel()

        # Act
        with vision_lockstep(module, VisionFeatureCache()):
            actual = module(input_ids=TEXT_IDS)

        # Assert
        assert module.filler_runs == 1
        torch.testing.assert_close(actual, TEXT_IDS.float(), rtol=0, atol=0)

    def test_cached_images_skip_the_tower_on_every_rank(self, frozen_qwen_vl) -> None:
        # Arrange
        model, cache, tower_runs = frozen_qwen_vl
        pixel_values = torch.rand(1, GRID, GRID, PATCH_DIM)

        # Act
        with (
            torch.no_grad(),
            cache.step(),
            vision_lockstep(model.model, cache),
            cache.images(torch.arange(1)),
        ):
            first = model(input_ids=IMAGE_IDS, pixel_values=pixel_values).logits
            second = model(input_ids=IMAGE_IDS, pixel_values=pixel_values).logits

        # Assert
        assert len(tower_runs) == 1
        torch.testing.assert_close(second, first, rtol=0, atol=0)

    def test_cached_images_run_the_tower_when_another_rank_does(
        self, frozen_qwen_vl, monkeypatch
    ) -> None:
        # Arrange
        model, cache, tower_runs = frozen_qwen_vl
        pixel_values = torch.rand(1, GRID, GRID, PATCH_DIM)

        # Act
        with (
            torch.no_grad(),
            cache.step(),
            vision_lockstep(model.model, cache),
            cache.images(torch.arange(1)),
        ):
            first = model(input_ids=IMAGE_IDS, pixel_values=pixel_values).logits
            monkeypatch.setattr(
                "agilerl.algorithms.core.llm_ops.vision_lockstep.any_rank",
                lambda _flag: True,
            )
            second = model(input_ids=IMAGE_IDS, pixel_values=pixel_values).logits

        # Assert
        assert len(tower_runs) == 2
        # A recomputed tower matches the cache to float32 noise across BLAS builds.
        torch.testing.assert_close(second, first, rtol=1e-5, atol=1e-6)

    def test_filler_features_never_enter_the_cache(
        self, frozen_qwen_vl, another_rank_runs_the_tower
    ) -> None:
        # Arrange
        model, cache, tower_runs = frozen_qwen_vl

        # Act
        with (
            torch.no_grad(),
            cache.step(),
            vision_lockstep(model.model, cache),
            cache.images(torch.arange(1)),
        ):
            model(input_ids=TEXT_IDS)
            entries = dict(cache.entries)

        # Assert
        assert len(tower_runs) == 1
        assert entries == {}

    def test_text_forward_skips_the_tower_after_exit(
        self, qwen_vl, another_rank_runs_the_tower
    ) -> None:
        # Arrange
        model, tower_runs = qwen_vl
        with vision_lockstep(model.model, VisionFeatureCache()):
            pass

        # Act
        with torch.no_grad():
            model(input_ids=TEXT_IDS)

        # Assert
        assert tower_runs == []

    @pytest.mark.multi_rank
    def test_mixed_ranks_train_the_sharded_tower_together(self) -> None:
        results = run_ranks(mixed_rank_worker, 2, timeout=180, cpu=True)

        assert results == [True, True]

    @pytest.mark.multi_rank
    def test_mixed_ranks_train_a_tower_after_per_layer_embeddings(self) -> None:
        results = run_ranks(mixed_rank_gemma4_worker, 2, timeout=180, cpu=True)

        assert results == [True, True]

    @pytest.mark.multi_rank
    def test_mixed_ranks_reuse_cached_features_together(self) -> None:
        results = run_ranks(cached_mixed_rank_worker, 2, timeout=180, cpu=True)

        assert results == [(1, True), (1, True)]


class TestVisionLockstepPlan:
    def test_agrees_once_per_run_of_forwards_on_new_images(
        self, frozen_qwen_vl, monkeypatch
    ) -> None:
        # Arrange
        model, cache, tower_runs = frozen_qwen_vl
        agreements = count_agreements(monkeypatch)
        images = torch.rand(2, GRID, GRID, PATCH_DIM)

        # Act
        with cache.step(), vision_lockstep(model.model, cache) as lockstep:
            logits = reference_then_actor_pass(model, cache, lockstep, images)

        # Assert
        assert agreements == ["plan", "plan"]
        assert len(tower_runs) == 2
        torch.testing.assert_close(logits[2:], logits[:2], rtol=0, atol=0)

    def test_unkeyed_images_run_the_tower_on_every_forward(
        self, frozen_qwen_vl, monkeypatch
    ) -> None:
        # Arrange
        model, cache, tower_runs = frozen_qwen_vl
        agreements = count_agreements(monkeypatch)
        pixel_values = torch.rand(1, GRID, GRID, PATCH_DIM)

        # Act
        with (
            torch.no_grad(),
            cache.step(),
            vision_lockstep(model.model, cache) as lockstep,
            lockstep.plan([None, None]),
        ):
            for _ in range(2):
                model(input_ids=IMAGE_IDS, pixel_values=pixel_values)

        # Assert
        assert agreements == ["plan"]
        assert len(tower_runs) == 2

    def test_forwards_past_the_plan_agree_one_at_a_time(
        self, qwen_vl, monkeypatch
    ) -> None:
        # Arrange
        model, tower_runs = qwen_vl
        agreements = count_agreements(monkeypatch)

        # Act
        with (
            torch.no_grad(),
            vision_lockstep(model.model, VisionFeatureCache()) as lockstep,
            lockstep.plan([[]]),
        ):
            for _ in range(2):
                model(input_ids=TEXT_IDS)

        # Assert
        assert agreements == ["plan", "forward"]
        assert tower_runs == []

    def test_empty_plan_skips_the_plan_agreement(self, monkeypatch) -> None:
        # Arrange
        agreements = count_agreements(monkeypatch)
        lockstep = VisionLockstep(VisionFeatureCache())

        # Act
        with lockstep.plan([]):
            reuses_first = lockstep.reuses(0)
            must_run = lockstep.must_run(False)

        # Assert
        assert agreements == ["forward"]
        assert reuses_first is False
        assert must_run is False

    @pytest.mark.multi_rank
    def test_mixed_ranks_agree_on_a_plan_together(self) -> None:
        results = run_ranks(planned_mixed_rank_worker, 2, timeout=180, cpu=True)

        assert results == [(["plan", "plan"], 2, True)] * 2
