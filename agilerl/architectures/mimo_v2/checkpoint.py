# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Load-time conversion of the FP8 / MXFP4 ``mimo_v2`` checkpoint into native ``MiMoV2Flash`` weights.

The Hub checkpoint stores:

- each fused ``self_attn.qkv_proj`` as FP8 pre-sharded in
  ``num_key_value_heads`` row chunks ``[Q_0|K_0|V_0|...|Q_n|K_n|V_n]``, each
  chunk with its own 128x128 block-scale grid padded to whole blocks;
- the dense MLP linears as FP8 with 128x128 block scales;
- the routed experts as per-expert MXFP4 (``uint8`` ``[out, in/2]``, low nibble
  first, with E8M0 ``uint8`` ``weight_scale`` ``[out, in/32]``);
- the attention sinks as ``attention_sink_bias``.

:func:`mimo_v2_checkpoint_conversions` turns these into the native separate
``q_proj`` / ``k_proj`` / ``v_proj``, dequantized dense linears, packed 3D
``gate_up_proj`` / ``down_proj`` experts, and ``sinks``. Every converter
source ends in ``$`` so the FP8 quantizer's dequantize pass leaves them as
they are.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers.core_model_loading import (
    Concatenate,
    ConversionOps,
    MergeModulelist,
    WeightConverter,
    WeightRenaming,
)

FP8_BLOCK_SIZE = 128
FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
E8M0_BIAS = 127
DENSE_FP8_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
TOWER_KEY_PATTERNS = (r"^visual\.", r"^audio_encoder\.", r"^speech_embeddings\.")


def dequant_fp8_block(
    weight: torch.Tensor, scale_inv: torch.Tensor, block: int = FP8_BLOCK_SIZE
) -> torch.Tensor:
    """Dequantize a block-scaled FP8 matrix whose scale grid covers whole, possibly padded, blocks.

    :param weight: FP8 ``[rows, cols]`` matrix.
    :type weight: torch.Tensor
    :param scale_inv: ``[ceil(rows / block), ceil(cols / block)]`` scales.
    :type scale_inv: torch.Tensor
    :param block: Square block size.
    :type block: int
    :return: fp32 ``[rows, cols]`` matrix.
    :rtype: torch.Tensor
    """
    rows, cols = weight.shape
    expected_grid = (math.ceil(rows / block), math.ceil(cols / block))
    if tuple(scale_inv.shape) != expected_grid:
        msg = (
            f"FP8 weight {tuple(weight.shape)} needs a {expected_grid} scale grid "
            f"for {block}x{block} blocks, got {tuple(scale_inv.shape)}"
        )
        raise ValueError(msg)
    scale = scale_inv.float().repeat_interleave(block, 0)[:rows]
    return weight.float() * scale.repeat_interleave(block, 1)[:, :cols]


def dequant_fp8_presharded(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    chunks: int,
    block: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize a fused FP8 matrix stored as ``chunks`` row chunks, each with its own padded scale grid.

    :param weight: FP8 ``[rows, cols]`` matrix, ``rows`` divisible by ``chunks``.
    :type weight: torch.Tensor
    :param scale_inv: Per-chunk scale grids stacked on dim 0.
    :type scale_inv: torch.Tensor
    :param chunks: Number of row chunks.
    :type chunks: int
    :param block: Square block size.
    :type block: int
    :return: fp32 ``[rows, cols]`` matrix.
    :rtype: torch.Tensor
    """
    chunk_scale_rows = math.ceil(weight.shape[0] // chunks / block)
    return torch.cat(
        [
            dequant_fp8_block(weight_chunk, scale_chunk, block)
            for weight_chunk, scale_chunk in zip(
                weight.chunk(chunks),
                scale_inv.split(chunk_scale_rows),
                strict=True,
            )
        ]
    )


def split_presharded_qkv(
    qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """De-interleave ``[Q_0|K_0|V_0|...|Q_n|K_n|V_n]`` into contiguous ``Q``, ``K``, ``V``.

    :param qkv: Fused projection rows.
    :type qkv: torch.Tensor
    :param num_heads: Query heads of the layer.
    :type num_heads: int
    :param num_kv_heads: Key / value heads of the layer.
    :type num_kv_heads: int
    :param head_dim: Query / key head dim.
    :type head_dim: int
    :param v_head_dim: Value head dim.
    :type v_head_dim: int
    :param chunks: Number of row chunks.
    :type chunks: int
    :return: ``Q``, ``K`` and ``V`` rows.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    sizes = [
        num_heads // chunks * head_dim,
        num_kv_heads // chunks * head_dim,
        num_kv_heads // chunks * v_head_dim,
    ]
    pieces = [chunk.split(sizes) for chunk in qkv.chunk(chunks)]
    query, key, value = (
        torch.cat([piece[index] for piece in pieces]) for index in range(3)
    )
    return query, key, value


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize OCP MXFP4: ``uint8`` ``[out, in/2]`` (low nibble first) with E8M0 ``uint8`` ``[out, in/block]``.

    :param packed: Two E2M1 values per byte.
    :type packed: torch.Tensor
    :param scale: One E8M0 exponent per block of each row.
    :type scale: torch.Tensor
    :return: fp32 ``[out, in]`` matrix.
    :rtype: torch.Tensor
    """
    values = torch.tensor(FP4_E2M1_VALUES, device=packed.device)
    unpacked = torch.stack(
        [values[(packed & 0xF).long()], values[(packed >> 4).long()]], dim=-1
    ).reshape(packed.shape[0], -1)
    block = unpacked.shape[1] // scale.shape[1]
    exponent = scale.to(torch.int32) - E8M0_BIAS
    return torch.ldexp(unpacked, exponent.repeat_interleave(block, dim=1))


def target_dtype(model: torch.nn.Module | None, full_layer_name: str) -> torch.dtype:
    """Dtype of the parameter a conversion fills, or fp32 when the caller casts on copy.

    :param model: Model being loaded, or None.
    :type model: torch.nn.Module | None
    :param full_layer_name: Parameter name the conversion produces.
    :type full_layer_name: str
    :return: Output dtype.
    :rtype: torch.dtype
    """
    if model is None:
        return torch.float32
    return model.get_parameter(full_layer_name).dtype


class LoadOnly(ConversionOps):
    """Reverse of a MiMo-V2 load conversion: saving back to the Hub format is unsupported."""

    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        """Raise: native MiMo weights cannot be written back as the FP8 / MXFP4 checkpoint.

        :raises NotImplementedError: Always.
        """
        msg = "MiMo-V2 checkpoints convert on load only; saving the Hub format is unsupported"
        raise NotImplementedError(msg)


class ParameterDtypeCast(ConversionOps):
    """Cast a renamed tensor to its parameter's dtype; transformers keeps pre-quantized renames as stored."""

    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        """Cast into the target parameter's dtype.

        :param input_dict: The stored tensor under its source pattern.
        :type input_dict: dict[str, Any]
        :param source_patterns: Converter source patterns.
        :type source_patterns: list[str]
        :param target_patterns: Converter target patterns.
        :type target_patterns: list[str]
        :return: The cast tensor under the target pattern.
        :rtype: dict[str, list[torch.Tensor]]
        """
        (tensor,) = input_dict[source_patterns[0]]
        dtype = target_dtype(kwargs.get("model"), kwargs["full_layer_name"])
        return {target_patterns[0]: [tensor.to(dtype)]}

    @property
    def reverse_op(self) -> ConversionOps:
        """Load-only.

        :return: An op that raises on save.
        :rtype: ConversionOps
        """
        return LoadOnly()


class Fp8BlockDequantize(ConversionOps):
    """Dequantize one FP8 dense linear collected with its ``weight_scale_inv``."""

    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        """Dequantize into the target parameter's dtype.

        :param input_dict: The weight and its scales under one source pattern.
        :type input_dict: dict[str, Any]
        :param source_patterns: Converter source patterns.
        :type source_patterns: list[str]
        :param target_patterns: Converter target patterns.
        :type target_patterns: list[str]
        :return: The dequantized weight under the target pattern.
        :rtype: dict[str, list[torch.Tensor]]
        """
        # Loaders collect keys sorted, so ``weight`` arrives before ``weight_scale_inv``.
        weight, scale_inv = input_dict[source_patterns[0]]
        dtype = target_dtype(kwargs.get("model"), kwargs["full_layer_name"])
        return {target_patterns[0]: [dequant_fp8_block(weight, scale_inv).to(dtype)]}

    @property
    def reverse_op(self) -> ConversionOps:
        """Load-only.

        :return: An op that raises on save.
        :rtype: ConversionOps
        """
        return LoadOnly()


class PreshardedQkvDequantize(ConversionOps):
    """Dequantize a pre-sharded FP8 fused ``qkv_proj`` and split it into ``q_proj`` / ``k_proj`` / ``v_proj``."""

    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        """Dequantize per chunk, de-interleave, and emit the three projections.

        :param input_dict: The fused weight and its scales under one source pattern.
        :type input_dict: dict[str, Any]
        :param source_patterns: Converter source patterns.
        :type source_patterns: list[str]
        :param target_patterns: ``q_proj``, ``k_proj``, ``v_proj`` target patterns.
        :type target_patterns: list[str]
        :return: The three projections under their target patterns.
        :rtype: dict[str, list[torch.Tensor]]
        """
        # Sorted key collection puts ``weight`` before ``weight_scale_inv``.
        weight, scale_inv = input_dict[source_patterns[0]]
        config = kwargs["config"]
        full_layer_name = kwargs["full_layer_name"]
        layer = int(full_layer_name.split(".layers.")[1].split(".")[0])
        # Pre-sharded over ``num_key_value_heads`` chunks; SWA layers carry
        # twice the KV heads in the same chunks.
        chunks = config.num_key_value_heads
        num_kv_heads = chunks
        if config.layer_types[layer] == "sliding_attention":
            num_kv_heads = 2 * chunks
        qkv = dequant_fp8_presharded(weight, scale_inv, chunks)
        projections = split_presharded_qkv(
            qkv,
            config.num_attention_heads,
            num_kv_heads,
            config.head_dim,
            config.v_head_dim,
            chunks,
        )
        dtype = target_dtype(kwargs.get("model"), full_layer_name)
        return {
            pattern: [projection.to(dtype)]
            for pattern, projection in zip(target_patterns, projections, strict=True)
        }

    @property
    def reverse_op(self) -> ConversionOps:
        """Load-only.

        :return: An op that raises on save.
        :rtype: ConversionOps
        """
        return LoadOnly()


class Mxfp4Dequantize(ConversionOps):
    """Dequantize per-expert MXFP4 weights, pairing each ``weight$`` source with its ``weight_scale$`` source."""

    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        """Dequantize every expert of every weight source.

        :param input_dict: Per-expert weights and scales keyed by source pattern.
        :type input_dict: dict[str, Any]
        :param source_patterns: ``...weight$`` and ``...weight_scale$`` patterns.
        :type source_patterns: list[str]
        :param target_patterns: Converter target patterns.
        :type target_patterns: list[str]
        :return: Dequantized per-expert weights keyed by weight source pattern.
        :rtype: dict[str, list[torch.Tensor]]
        """
        dtype = target_dtype(kwargs.get("model"), kwargs["full_layer_name"])
        return {
            pattern: [
                dequant_mxfp4(packed, scale).to(dtype)
                for packed, scale in zip(
                    input_dict[pattern],
                    input_dict[pattern.removesuffix("$") + "_scale$"],
                    strict=True,
                )
            ]
            for pattern in source_patterns
            if pattern.endswith("weight$")
        }

    @property
    def reverse_op(self) -> ConversionOps:
        """Load-only.

        :return: An op that raises on save.
        :rtype: ConversionOps
        """
        return LoadOnly()


def mimo_v2_checkpoint_conversions() -> list[WeightConverter | WeightRenaming]:
    """Conversions from the ``mimo_v2`` Hub checkpoint to native ``MiMoV2Flash`` weights.

    :return: Converters, expert converters before the dense MLP ones so those
        patterns never see expert keys.
    :rtype: list[WeightConverter | WeightRenaming]
    """
    expert_weight = "mlp.experts.*.{}.weight$"
    expert_scale = "mlp.experts.*.{}.weight_scale$"
    return [
        WeightConverter(
            source_patterns="self_attn.attention_sink_bias$",
            target_patterns="self_attn.sinks",
            operations=[ParameterDtypeCast()],
        ),
        WeightConverter(
            source_patterns=[
                expert_weight.format("gate_proj"),
                expert_weight.format("up_proj"),
                expert_scale.format("gate_proj"),
                expert_scale.format("up_proj"),
            ],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[Mxfp4Dequantize(), MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns=[
                expert_weight.format("down_proj"),
                expert_scale.format("down_proj"),
            ],
            target_patterns="mlp.experts.down_proj",
            operations=[Mxfp4Dequantize(), MergeModulelist(dim=0)],
        ),
        WeightConverter(
            source_patterns=r"self_attn.qkv_proj.weight(?:_scale_inv)?$",
            target_patterns=[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ],
            operations=[PreshardedQkvDequantize()],
        ),
        *(
            WeightConverter(
                source_patterns=rf"mlp.{projection}.weight(?:_scale_inv)?$",
                target_patterns=f"mlp.{projection}.weight",
                operations=[Fp8BlockDequantize()],
            )
            for projection in DENSE_FP8_PROJECTIONS
        ),
    ]
