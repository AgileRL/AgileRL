# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Every catalog vision family shards its image features, and each CP shard gets its slice of the full forward's decoder inputs (CPU, tiny weights)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

from agilerl.algorithms.core.llm_ops.vision_lockstep import (
    FILLER_METHOD,
    vision_language_module,
)
from agilerl.architectures.catalog import (
    family_vision,
    model_type_vision,
    pretrained_config,
    pretrained_model_type,
)
from agilerl.architectures.gemma4.vision import install_gemma4_vision
from agilerl.architectures.glm5_next.vision import install_glm5_next_vision
from agilerl.architectures.mimo_v2.configuration import MiMoV2Config
from agilerl.architectures.mimo_v2.modeling import (
    MiMoV2ForConditionalGeneration,
    MiMoV2Model,
)
from agilerl.architectures.nemotron_h_omni.vision import (
    install_nemotron_h_omni_vision,
    register_nemotron_h_omni_native_loading,
)
from agilerl.architectures.qwen3_5.vision import patch_qwen_vision_rows
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.architectures.vision_rows import ACCEPTS_VISION_ROWS, SHARDS_IMAGE_FEATURES
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed.context_parallel import (
    IMAGE_TOKEN_SHARD,
    ULYSSES_PARAMS,
    ImageTokenShard,
    clear_ulysses_params,
)
from tests.parallel_helpers import publish_varlen_offsets
from tests.test_architectures.gemma4.test_vision import tiny_gemma4
from tests.test_architectures.glm5_next.test_vision_rows import (
    _tiny_config as tiny_glm_config,
)
from tests.test_architectures.mimo_v2.checkpoint_helpers import (
    IMAGE_TOKEN_ID as MIMO_IMAGE,
)
from tests.test_architectures.mimo_v2.checkpoint_helpers import (
    PATCH_DIM as MIMO_PATCH_DIM,
)
from tests.test_architectures.mimo_v2.checkpoint_helpers import tiny_hub_config
from tests.test_architectures.nemotron_h_omni.test_vision import (
    _tiny_config as tiny_nemotron_config,
)
from tests.test_architectures.qwen3_5.vision_helpers import tiny_qwen_vl

CP = 2
VISION_LANGUAGE_CLASSES: dict[str, type] = {
    "qwen3_5": Qwen3_5Model,
    "qwen3_5_moe": Qwen3_5MoeModel,
    "gemma4": Gemma4Model,
    "glm5_next": Glm5NextModel,
    "nemotron_h_omni": NemotronH_Omni_Reasoning_V3,
    "mimo_v2": MiMoV2Model,
}
PATCHED_CLASSES = (
    Qwen3_5Model,
    Qwen3_5MoeModel,
    Gemma4Model,
    Glm5NextModel,
    NemotronH_Omni_Reasoning_V3,
)
PATCHED_NAMES = ("forward", "get_image_features", ACCEPTS_VISION_ROWS, FILLER_METHOD)


def qwen3_5() -> nn.Module:
    patch_qwen_vision_rows("transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Model")
    return tiny_qwen_vl()


def gemma4() -> nn.Module:
    install_gemma4_vision(PatchRuntimeConfig())
    return tiny_gemma4()


def glm5_next() -> nn.Module:
    install_glm5_next_vision(PatchRuntimeConfig())
    torch.manual_seed(0)
    return Glm5NextForConditionalGeneration(tiny_glm_config()).eval()


def nemotron_h_omni() -> nn.Module:
    register_nemotron_h_omni_native_loading()
    install_nemotron_h_omni_vision(PatchRuntimeConfig())
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(tiny_nemotron_config()).eval()


def mimo_v2() -> nn.Module:
    config = MiMoV2Config.from_dict(tiny_hub_config())
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    return MiMoV2ForConditionalGeneration(config).eval()


@dataclass(frozen=True)
class VisionFamily:
    """Tiny model of one family and the image layout it takes."""

    build: Callable[[], nn.Module]
    image_token_id: int
    grid: tuple[int, int]
    patch_dim: int
    tokens_per_image: int


FAMILIES = {
    "qwen3_5": VisionFamily(qwen3_5, 100, (4, 4), 3 * 2 * 2 * 2, 4),
    "gemma4": VisionFamily(gemma4, 100, (4, 4), 3 * 2 * 2, 4),
    "glm5_next": VisionFamily(glm5_next, 5, (4, 4), 3 * 2 * 2 * 2, 4),
    "nemotron_h_omni": VisionFamily(nemotron_h_omni, 5, (4, 4), 3 * 8 * 8, 4),
    "mimo_v2": VisionFamily(mimo_v2, MIMO_IMAGE, (2, 4), MIMO_PATCH_DIM, 2),
}


def crossing_batch(image: int, tokens: int) -> tuple[torch.Tensor, int]:
    """Two rows: one image across the shard boundary and one after it, then one image in the first shard only."""
    width = 4 * tokens + 4
    half = width // 2
    first = [1] * (half - tokens // 2) + [image] * tokens + [2] + [image] * tokens
    second = [image] * tokens
    rows = [first + [3] * (width - len(first)), second + [4] * (width - len(second))]
    return torch.tensor(rows), 3


def first_shard_batch(image: int, tokens: int) -> tuple[torch.Tensor, int]:
    """One row whose only image sits in the first shard."""
    width = 4 * tokens + 4
    return torch.tensor([[image] * tokens + [1] * (width - tokens)]), 1


BATCHES = {"crossing": crossing_batch, "first_shard": first_shard_batch}


class DecoderReached(Exception):
    """Raised by the decoder pre-hook once it has its inputs."""


def decoder_inputs(model: nn.Module, **inputs: torch.Tensor) -> dict[str, Any]:
    """Keyword inputs the decoder receives, without running it."""
    captured: dict[str, Any] = {}

    def stop(_module: nn.Module, _args: object, kwargs: dict[str, Any]) -> None:
        captured.update(kwargs)
        raise DecoderReached

    handle = model.get_decoder().register_forward_pre_hook(stop, with_kwargs=True)
    try:
        with torch.no_grad(), pytest.raises(DecoderReached):
            model(**inputs)
    finally:
        handle.remove()
    return captured


def tower_ulysses_params(model: nn.Module) -> list[dict[str, object]]:
    """Ulysses params published at each later call of the model's vision tower."""
    seen: list[dict[str, object]] = []
    model.get_encoder(modality="image").register_forward_pre_hook(
        lambda _module, _args: seen.append(dict(ULYSSES_PARAMS))
    )
    return seen


@pytest.fixture
def pristine_vision_classes() -> Iterator[None]:
    """Restore the class attributes the vision installs replace."""
    saved = [
        (cls, name, vars(cls).get(name))
        for cls in PATCHED_CLASSES
        for name in PATCHED_NAMES
    ]
    yield
    for cls, name, value in saved:
        if value is None:
            if name in vars(cls):
                delattr(cls, name)
        else:
            setattr(cls, name, value)


@pytest.mark.usefixtures("pristine_vision_classes")
class TestCatalogVisionModelsShardImageFeatures:
    def test_every_catalog_vision_model_type_has_a_vision_language_class(
        self,
    ) -> None:
        vision_model_types = {
            pretrained_model_type(hub_id)
            for hub_id in SUPPORTED_MODEL_INFO
            if family_vision(hub_id) is not None
        }

        assert vision_model_types <= VISION_LANGUAGE_CLASSES.keys()

    @pytest.mark.parametrize(
        "model_type", VISION_LANGUAGE_CLASSES, ids=VISION_LANGUAGE_CLASSES
    )
    def test_family_vision_install_shards_image_features(self, model_type: str) -> None:
        # Arrange
        vision = model_type_vision(model_type)
        assert vision is not None

        # Act
        if vision.install is not None:
            vision.install(PatchRuntimeConfig())

        # Assert
        get_image_features = VISION_LANGUAGE_CLASSES[model_type].get_image_features
        assert getattr(get_image_features, SHARDS_IMAGE_FEATURES, False) is True

    @pytest.mark.parametrize(
        "hub_id",
        [hub_id for hub_id in SUPPORTED_MODEL_INFO if family_vision(hub_id)],
    )
    def test_catalog_vision_model_attends_causally(self, hub_id: str) -> None:
        config = pretrained_config(hub_id)

        configs = [config, config.get("text_config") or {}]

        assert [c.get("use_bidirectional_attention") for c in configs] == [None, None]


@pytest.mark.usefixtures("pristine_vision_classes")
@pytest.mark.parametrize("family", FAMILIES.values(), ids=FAMILIES.keys())
class TestContextParallelImageFeatures:
    @pytest.mark.parametrize("batch", BATCHES.values(), ids=BATCHES.keys())
    @pytest.mark.parametrize("rank", range(CP))
    def test_each_shard_gets_its_slice_of_the_full_decoder_inputs(
        self,
        family: VisionFamily,
        batch: Callable[[int, int], tuple[torch.Tensor, int]],
        rank: int,
    ) -> None:
        # Arrange
        model = family.build()
        input_ids, images = batch(family.image_token_id, family.tokens_per_image)
        position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        torch.manual_seed(1)
        pixel_values = torch.randn(images, *family.grid, family.patch_dim)
        full = decoder_inputs(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
        )
        shard = ImageTokenShard(input_ids, position_ids, rank, CP)

        # Act
        token = IMAGE_TOKEN_SHARD.set(shard)
        try:
            sharded = decoder_inputs(
                model,
                input_ids=shard.shard(input_ids),
                position_ids=shard.shard(position_ids),
                pixel_values=pixel_values,
            )
        finally:
            IMAGE_TOKEN_SHARD.reset(token)

        # Assert
        assert torch.equal(sharded["inputs_embeds"], shard.shard(full["inputs_embeds"]))
        assert torch.equal(
            sharded["position_ids"], shard.shard(full["position_ids"], seq_dim=-1)
        )

    def test_tower_runs_with_ulysses_off_and_restores_the_params(
        self, family: VisionFamily
    ) -> None:
        # Arrange
        model = family.build()
        input_ids, images = crossing_batch(
            family.image_token_id, family.tokens_per_image
        )
        position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        pixel_values = torch.randn(images, *family.grid, family.patch_dim)
        shard = ImageTokenShard(input_ids, position_ids, 0, CP)
        seen = tower_ulysses_params(model)
        publish_varlen_offsets(torch.tensor([0, 4], dtype=torch.int32))
        published = dict(ULYSSES_PARAMS)

        # Act
        token = IMAGE_TOKEN_SHARD.set(shard)
        try:
            decoder_inputs(
                model,
                input_ids=shard.shard(input_ids),
                position_ids=shard.shard(position_ids),
                pixel_values=pixel_values,
            )
            after_image_call = dict(ULYSSES_PARAMS)
        finally:
            IMAGE_TOKEN_SHARD.reset(token)
        with torch.no_grad():
            vision_language_module(model).vision_filler_features(torch.device("cpu"))
        after_filler = dict(ULYSSES_PARAMS)
        clear_ulysses_params()

        # Assert
        assert seen == [{}, {}]
        assert after_image_call == published
        assert after_filler == published
