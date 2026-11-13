# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``mimo_v2`` Hub config read as a native ``MiMoV2FlashConfig`` language model and its vision tower."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig
from transformers.models.mimo_v2_flash.configuration_mimo_v2_flash import (
    MiMoV2FlashConfig,
)
from typing_extensions import Self

MIMO_V2_MODEL_TYPE = "mimo_v2"
MIMO_V2_TEXT_MODEL_TYPE = "mimo_v2_text"
MIMO_V2_VISION_MODEL_TYPE = "mimo_v2_vision"


def flash_config_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a ``mimo_v2`` ``config.json`` onto text-only ``MiMoV2FlashConfig`` fields, without towers or MTP layers.

    :param raw: Hub ``config.json`` contents.
    :type raw: dict[str, Any]
    :return: ``MiMoV2FlashConfig`` keyword arguments.
    :rtype: dict[str, Any]
    :raises ValueError: If the checkpoint uses a layout the native class cannot build.
    """
    if raw["swa_num_attention_heads"] != raw["num_attention_heads"]:
        msg = "MiMoV2Flash needs SWA and full-attention layers to share num_attention_heads"
        raise ValueError(msg)
    if raw["swa_num_key_value_heads"] != 2 * raw["num_key_value_heads"]:
        msg = "MiMoV2Flash needs twice as many KV heads on SWA layers as on full-attention layers"
        raise ValueError(msg)
    if (
        raw["swa_head_dim"] != raw["head_dim"]
        or raw["swa_v_head_dim"] != raw["v_head_dim"]
    ):
        msg = "MiMoV2Flash needs SWA and full-attention layers to share head dims"
        raise ValueError(msg)
    if raw["add_full_attention_sink_bias"] or not raw["add_swa_attention_sink_bias"]:
        msg = "MiMoV2Flash has attention sinks on SWA layers only"
        raise ValueError(msg)
    if raw["topk_method"] != "noaux_tc" or raw["scoring_func"] != "sigmoid":
        msg = "MiMoV2Flash routes with sigmoid noaux_tc only"
        raise ValueError(msg)
    if raw["n_shared_experts"]:
        msg = "MiMoV2Flash has no shared experts"
        raise ValueError(msg)
    partial_rotary_factor = raw["partial_rotary_factor"]
    return {
        "vocab_size": raw["vocab_size"],
        "hidden_size": raw["hidden_size"],
        "intermediate_size": raw["intermediate_size"],
        "num_hidden_layers": raw["num_hidden_layers"],
        "num_attention_heads": raw["num_attention_heads"],
        "num_key_value_heads": raw["num_key_value_heads"],
        "hidden_act": raw["hidden_act"],
        "max_position_embeddings": raw["max_position_embeddings"],
        "initializer_range": raw["initializer_range"],
        "rms_norm_eps": raw["layernorm_epsilon"],
        "tie_word_embeddings": raw["tie_word_embeddings"],
        "attention_bias": raw["attention_bias"],
        "attention_dropout": raw["attention_dropout"],
        "moe_intermediate_size": raw["moe_intermediate_size"],
        "num_experts_per_tok": raw["num_experts_per_tok"],
        "n_routed_experts": raw["n_routed_experts"],
        "routed_scaling_factor": raw["routed_scaling_factor"],
        "n_group": raw["n_group"],
        "topk_group": raw["topk_group"],
        "norm_topk_prob": raw["norm_topk_prob"],
        "bos_token_id": raw["bos_token_id"],
        "eos_token_id": raw["eos_token_id"],
        "pad_token_id": raw["pad_token_id"],
        "head_dim": raw["head_dim"],
        "v_head_dim": raw["v_head_dim"],
        "sliding_window": raw["sliding_window"],
        "layer_types": [
            "sliding_attention" if pattern == 1 else "full_attention"
            for pattern in raw["hybrid_layer_pattern"]
        ],
        "mlp_layer_types": [
            "sparse" if is_moe else "dense" for is_moe in raw["moe_layer_freq"]
        ],
        "attention_value_scale": raw["attention_value_scale"],
        "rope_parameters": {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": raw["rope_theta"],
                "partial_rotary_factor": partial_rotary_factor,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": raw["swa_rope_theta"],
                "partial_rotary_factor": partial_rotary_factor,
            },
        },
        "dtype": raw["dtype"],
        # Training needs full-precision weights.
        "quantization_config": {**raw["quantization_config"], "dequantize": True},
    }


def vision_config_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Pick the ``vision_config`` fields the tower is built from.

    :param raw: Hub ``vision_config`` contents.
    :type raw: dict[str, Any]
    :return: ``MiMoV2VisionConfig`` keyword arguments.
    :rtype: dict[str, Any]
    """
    names = (
        "depth",
        "hidden_size",
        "intermediate_size",
        "hidden_act",
        "num_heads",
        "num_key_value_heads",
        "in_chans",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "out_hidden_size",
        "fullatt_block_indexes",
        "vit_window_attn_types",
        "visual_token_window_size",
        "use_sink",
    )
    return {name: raw[name] for name in names}


def mimo_v2_config_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a ``mimo_v2`` ``config.json`` onto ``MiMoV2Config`` fields; the audio tower and MTP layers are dropped.

    :param raw: Hub ``config.json`` contents.
    :type raw: dict[str, Any]
    :return: ``MiMoV2Config`` keyword arguments.
    :rtype: dict[str, Any]
    """
    text = flash_config_fields(raw)
    # On the root only: a sub-config dtype overrides the dtype a load asks for.
    dtype = text.pop("dtype")
    return {
        "text_config": text,
        "vision_config": vision_config_fields(raw["vision_config"]),
        "image_token_id": raw["image_token_id"],
        "processor_config": raw["processor_config"],
        "tie_word_embeddings": raw["tie_word_embeddings"],
        "dtype": dtype,
        "quantization_config": text.pop("quantization_config"),
    }


class MiMoV2TextConfig(MiMoV2FlashConfig):
    """Language model of a ``mimo_v2`` checkpoint."""

    model_type = MIMO_V2_TEXT_MODEL_TYPE
    base_config_key = "text_config"


@strict
class MiMoV2VisionConfig(PreTrainedConfig):
    """Vision tower of a ``mimo_v2`` checkpoint, under the Hub's field names.

    ``vit_window_attn_types`` gives each block's window: ``-1`` none, ``0`` the
    patch order, ``1`` the column-major order of the merge windows.
    ``fullatt_block_indexes`` are the blocks that attend over the whole image.
    """

    model_type = MIMO_V2_VISION_MODEL_TYPE
    base_config_key = "vision_config"

    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    hidden_act: str = "silu"
    num_heads: int = 32
    num_key_value_heads: int = 8
    # The checkpoint config names no head dims; vLLM builds 64 for both.
    qk_channels: int = 64
    kv_channels: int = 64
    in_chans: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096
    fullatt_block_indexes: list[int] | None = None
    vit_window_attn_types: list[int] | None = None
    visual_token_window_size: int = 64
    use_sink: bool = True
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs: Any) -> None:
        """Default to the MiMo-V2.6 block layout."""
        if self.fullatt_block_indexes is None:
            self.fullatt_block_indexes = [0, 9, 18, 27]
        if self.vit_window_attn_types is None:
            self.vit_window_attn_types = ([-1] + [0] * 4 + [1] * 4) * 3 + [-1]
        super().__post_init__(**kwargs)


@strict
class MiMoV2Config(PreTrainedConfig):
    """``mimo_v2`` checkpoint: the native ``MiMoV2Flash`` language model and its vision tower."""

    model_type = MIMO_V2_MODEL_TYPE
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {
        "text_config": MiMoV2TextConfig,
        "vision_config": MiMoV2VisionConfig,
    }
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    # Image pixel bounds and patch geometry vLLM's processor and the memory estimator read.
    processor_config: dict[str, Any] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs: Any) -> None:
        """Build the sub-configs from their dicts."""
        if isinstance(self.text_config, dict):
            self.text_config = MiMoV2TextConfig(
                **cast("dict[str, Any]", self.text_config)
            )
        elif self.text_config is None:
            self.text_config = MiMoV2TextConfig()
        if isinstance(self.vision_config, dict):
            self.vision_config = MiMoV2VisionConfig(
                **cast("dict[str, Any]", self.vision_config)
            )
        elif self.vision_config is None:
            self.vision_config = MiMoV2VisionConfig()
        super().__post_init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> Self:
        """Build the native config from Hub ``mimo_v2`` fields.

        :param config_dict: Hub ``config.json`` contents.
        :type config_dict: dict[str, Any]
        :return: The native config, paired with the unused kwargs when
            ``return_unused_kwargs`` is set, as the parent returns.
        :rtype: Self
        """
        return super().from_dict(mimo_v2_config_fields(config_dict), **kwargs)
