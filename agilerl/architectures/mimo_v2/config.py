# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``mimo_v2`` Hub config read as the native text-only ``MiMoV2FlashConfig``."""

from __future__ import annotations

from typing import Any

from transformers.models.mimo_v2_flash.configuration_mimo_v2_flash import (
    MiMoV2FlashConfig,
)
from typing_extensions import Self

MIMO_V2_MODEL_TYPE = "mimo_v2"


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


class MiMoV2Config(MiMoV2FlashConfig):
    """``MiMoV2FlashConfig`` built from a ``mimo_v2`` Hub ``config.json``."""

    model_type = MIMO_V2_MODEL_TYPE

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> Self:
        """Build the native config from Hub ``mimo_v2`` fields.

        :param config_dict: Hub ``config.json`` contents.
        :type config_dict: dict[str, Any]
        :return: The native text-only config, paired with the unused kwargs
            when ``return_unused_kwargs`` is set, as the parent returns.
        :rtype: Self
        """
        return super().from_dict(flash_config_fields(config_dict), **kwargs)
