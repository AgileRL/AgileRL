# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel Gated DeltaNet mixers.

The sequence stays sharded. Conv and scan take a flash-linear-attention
context built from the full-sequence ``cu_seqlens``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from itertools import pairwise
from typing import NamedTuple, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from agilerl.architectures.qwen3_5.gated_delta_net import (
    GdnKernel,
    gdn_kernel_import_errors,
)
from agilerl.distributed.context_parallel.sequence import CP_SCAN_CHUNK
from agilerl.distributed.context_parallel.ulysses import ULYSSES_PARAMS

GDN_CP_FLAG = "_agilerl_gdn_cp"
GDN_CP_KERNEL_FLAG = "_agilerl_gdn_cp_kernel"
CONV_KERNEL_NAME = "causal_conv1d_fn"
MODULE_SCAN_KERNEL_NAME = "torch_chunk_gated_delta_rule"


def _fla_build_cp_context(
    cu_seqlens: torch.Tensor,
    group: dist.ProcessGroup,
    conv1d_kernel_size: int,
) -> object:
    """Build a flash-linear-attention context for ``cu_seqlens``."""
    # flash-linear-attention is an optional extra.
    from fla.ops.cp import build_cp_context

    return build_cp_context(
        cu_seqlens,
        group,
        conv1d_kernel_size=conv1d_kernel_size,
    )


def _fla_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    activation: str | None,
    cp_context: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FLA causal conv. ``x`` is ``[B, T, D]``."""
    # flash-linear-attention is an optional extra.
    from fla.modules.convolution import causal_conv1d

    # fla's input_guard types its wrapped functions as returning one Tensor.
    conv = cast(
        "Callable[..., tuple[torch.Tensor, torch.Tensor | None]]", causal_conv1d
    )
    return conv(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        cp_context=cp_context,
    )


class GdnCpContext(NamedTuple):
    """FLA context and scan chunk size of one micro-batch's published offsets."""

    fla: object
    chunk_size: int


def _published_offsets() -> tuple[torch.Tensor, tuple[int, ...]]:
    cu_seqlens = ULYSSES_PARAMS.get("cu_seqlens")
    host = cast("tuple[int, ...]", ULYSSES_PARAMS.get("host_cu_seqlens", ()))
    if isinstance(cu_seqlens, torch.Tensor) and len(host) >= 2:
        return cu_seqlens, host
    msg = "Gated DeltaNet context parallel requires full-sequence cu_seqlens"
    raise RuntimeError(msg)


def _scan_cu_seqlens(batch: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Full-sequence offsets, or one row's when every row has the same length."""
    cu_seqlens, host = _published_offsets()
    if batch == 1:
        return cu_seqlens, host
    n_seq = len(host) - 1
    if n_seq != batch:
        msg = f"Gated DeltaNet context parallel got {batch} rows and {n_seq} sequences"
        raise RuntimeError(msg)
    if len({stop - start for start, stop in pairwise(host)}) != 1:
        msg = "Gated DeltaNet context parallel rows must share one length"
        raise RuntimeError(msg)
    # Ranks hold the same slice of every row, so each row is its own sequence.
    return cu_seqlens[:2], host[:2]


def _gdn_cp_context(
    group: dist.ProcessGroup, conv_kernel_size: int, world: int, batch: int
) -> GdnCpContext:
    """Context of the published offsets, built by the first GDN layer of the micro-batch."""
    contexts = cast("dict[tuple[int, int], GdnCpContext]", ULYSSES_PARAMS["contexts"])
    key = (conv_kernel_size, batch)
    if key not in contexts:
        cu_seqlens, host = _scan_cu_seqlens(batch)
        contexts[key] = GdnCpContext(
            _fla_build_cp_context(
                cu_seqlens, group, conv1d_kernel_size=conv_kernel_size
            ),
            _gdn_chunk_size(host, world),
        )
    return contexts[key]


def _boundary_lengths(bounds: tuple[int, ...], world: int) -> list[int]:
    """Token counts of the pieces whose state crosses a rank boundary."""
    part = bounds[-1] // world
    lengths: list[int] = []
    for rank in range(world):
        start = part * rank
        end = start + part
        overlapping = [
            (bounds[index], bounds[index + 1])
            for index in range(len(bounds) - 1)
            if bounds[index] < end and bounds[index + 1] > start
        ]
        first_start, first_end = overlapping[0]
        last_start, last_end = overlapping[-1]
        if last_end > end:
            lengths.append(end - max(last_start, start))
        if first_start < start:
            lengths.append(min(first_end, end) - start)
    return lengths


def _gdn_chunk_size(bounds: tuple[int, ...], world: int) -> int:
    """Largest of 64, 32, and 16 that divides every rank-boundary piece."""
    # FLA's CP state summary matches a full sequence of whole chunks.
    lengths = _boundary_lengths(bounds, world)
    for chunk in (CP_SCAN_CHUNK, CP_SCAN_CHUNK // 2, CP_SCAN_CHUNK // 4):
        if lengths and all(length % chunk == 0 for length in lengths):
            return chunk
    return 64


def _conv_arguments(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str | None]:
    """HF passes ``x, weight, bias`` positionally or as keywords."""
    call = {**dict(zip(("x", "weight", "bias"), args, strict=False)), **kwargs}
    return (
        cast("torch.Tensor", call["x"]),
        cast("torch.Tensor", call["weight"]),
        cast("torch.Tensor | None", call.get("bias")),
        cast("str | None", call.get("activation")),
    )


def _conv_row(
    hidden: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    activation: str | None,
    cp_context: object,
) -> torch.Tensor:
    # HF calls channels-first [B, D, T]. FLA context-parallel conv is [B, T, D].
    output, _state = _fla_causal_conv1d(
        hidden.transpose(1, 2).contiguous(),
        weight,
        bias,
        activation,
        cp_context,
    )
    return output.transpose(1, 2)


def _batch_row(value: object, index: int, batch: int) -> object:
    if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == batch:
        return value[index : index + 1]
    return value


def _fla_chunk_gated_delta_rule(*args: object, **kwargs: object) -> object:
    """Run flash-linear-attention's scan."""
    # flash-linear-attention is an optional extra.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule(*args, **kwargs)


def _cp_conv_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    activation: str | None,
    cp_context: object,
) -> torch.Tensor:
    if hidden.shape[0] == 1:
        output = _conv_row(hidden, weight, bias, activation, cp_context)
    else:
        output = torch.cat(
            [
                _conv_row(
                    hidden[index : index + 1],
                    weight,
                    bias,
                    activation,
                    cp_context,
                )
                for index in range(int(hidden.shape[0]))
            ],
            dim=0,
        )
    if output.requires_grad:
        return output.clone()
    return output


def _cp_scan_forward(
    scan: Callable[..., object],
    group: dist.ProcessGroup,
    conv_kernel_size: int,
    world: int,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    query = cast("torch.Tensor", args[0] if args else kwargs["q"])
    batch = int(query.shape[0])
    context = _gdn_cp_context(group, conv_kernel_size, world, batch)
    scan_kwargs = {
        **kwargs,
        "cp_context": context.fla,
        "chunk_size": context.chunk_size,
    }
    if batch == 1:
        return scan(*args, **scan_kwargs)
    outputs: list[torch.Tensor] = []
    states: list[torch.Tensor | None] = []
    for index in range(batch):
        row_args = tuple(_batch_row(value, index, batch) for value in args)
        row_kwargs = {
            key: _batch_row(value, index, batch) for key, value in kwargs.items()
        }
        row_kwargs["cp_context"] = context.fla
        row_kwargs["chunk_size"] = context.chunk_size
        output_row, state = cast(
            "tuple[torch.Tensor, torch.Tensor | None]",
            scan(*row_args, **row_kwargs),
        )
        outputs.append(output_row)
        states.append(state)
    output = torch.cat(outputs, dim=0)
    if states[0] is None:
        return output, None
    return output, torch.cat(cast("list[torch.Tensor]", states), dim=0)


def _install_gdn_module_kernels(
    mixer: nn.Module, group: dist.ProcessGroup, world: int
) -> None:
    """Patch modeling-module conv and scan. HF calls those, not mixer attributes."""
    module = sys.modules[type(mixer).__module__]
    conv = getattr(module, CONV_KERNEL_NAME, None)
    scan = getattr(module, MODULE_SCAN_KERNEL_NAME, None)
    if conv is None or scan is None:
        msg = (
            "Gated DeltaNet context parallel requires causal_conv1d_fn and "
            "torch_chunk_gated_delta_rule"
        )
        raise RuntimeError(msg)
    # The torch fallbacks ignore cp_context, so each shard would scan from zero state.
    errors = gdn_kernel_import_errors(
        (
            GdnKernel("fla.ops.cp", "build_cp_context", "flash-linear-attention"),
            GdnKernel(
                "fla.ops.gated_delta_rule",
                "chunk_gated_delta_rule",
                "flash-linear-attention",
            ),
        )
    )
    if errors:
        msg = "Gated DeltaNet context parallel needs " + " and ".join(errors)
        raise RuntimeError(msg)
    kernel = cast("int", mixer.conv_kernel_size)
    if getattr(conv, GDN_CP_KERNEL_FLAG, False):
        return

    def cp_conv(*args: object, **kwargs: object) -> torch.Tensor:
        if "cu_seqlens" not in ULYSSES_PARAMS:
            return conv(*args, **kwargs)
        hidden, weight, bias, activation = _conv_arguments(args, kwargs)
        context = _gdn_cp_context(group, kernel, world, int(hidden.shape[0]))
        return _cp_conv_forward(hidden, weight, bias, activation, context.fla)

    def cp_scan(*args: object, **kwargs: object) -> object:
        if "cu_seqlens" not in ULYSSES_PARAMS:
            return scan(*args, **kwargs)
        return _cp_scan_forward(
            _fla_chunk_gated_delta_rule, group, kernel, world, args, kwargs
        )

    setattr(cp_conv, GDN_CP_KERNEL_FLAG, True)
    setattr(cp_scan, GDN_CP_KERNEL_FLAG, True)
    setattr(module, CONV_KERNEL_NAME, cp_conv)
    setattr(module, MODULE_SCAN_KERNEL_NAME, cp_scan)


def install_gdn_context_parallel(mixer: nn.Module, cp_mesh: DeviceMesh) -> bool:
    """Keep Gated DeltaNet on the sequence shard and pass FLA ``cp_context``.

    :param mixer: GatedDeltaNet module whose modeling module holds the kernels.
    :param cp_mesh: Context-parallel mesh. Size is the CP degree.
    :return: True when this call wrapped ``mixer``.
    """
    if getattr(mixer, GDN_CP_FLAG, False):
        return False
    _install_gdn_module_kernels(mixer, cp_mesh.get_group(), int(cp_mesh.size()))
    object.__setattr__(mixer, GDN_CP_FLAG, True)
    return True


def wraps_gdn_context_parallel(model: nn.Module) -> bool:
    """Whether :func:`install_gdn_context_parallel` wrapped a mixer of ``model``."""
    return any(getattr(module, GDN_CP_FLAG, False) for module in model.modules())
