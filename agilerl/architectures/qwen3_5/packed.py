# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Reset Qwen GatedDeltaNet state at packed-document boundaries.

HF GDN consumes ``seq_idx`` (causal conv) and ``cu_seq_lens_q`` (FLA scan).
The packed trainer forward only sends restarting ``position_ids``, so the
decoder derives those kwargs. When the fused kernels are absent, each
document is run on its own: the torch GDN fallback ignores ``cu_seqlens``.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.llm_packing import RESETS_AT_DOCUMENT_BOUNDARY, packed_seq_idx
from agilerl.utils.patching import class_is_patched, try_import

logger = logging.getLogger(__name__)

__all__ = [
    "install_qwen_gdn_patches",
    "patch_qwen_gdn_packed_sequences",
]

PACKED_TARGETS = (
    (
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeDecoderLayer",
    ),
    (
        "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet",
        "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer",
    ),
)


def _resolve_class(dotted: str) -> type | None:
    """Resolve ``module.Class``, or None when the module cannot be imported."""
    module_path, sep, class_name = dotted.rpartition(".")
    if not sep or not module_path or not class_name:
        message = f"[qwen-packed] invalid class path {dotted!r}"
        raise RuntimeError(message)
    module = try_import(module_path)
    if module is None:
        return None
    resolved = getattr(module, class_name, None)
    if resolved is None:
        message = f"[qwen-packed] {module_path} is present but missing {class_name}"
        raise RuntimeError(message)
    return resolved


def _cu_seqlens(seq_idx: torch.Tensor) -> torch.Tensor:
    """``(D + 1,)`` int32 cumulative lengths of a packed ``(1, N)`` ``seq_idx``."""
    counts = torch.bincount(seq_idx.reshape(-1))
    return F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)


def _per_document(
    run: Callable[[torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,
    seq_idx: torch.Tensor,
) -> torch.Tensor:
    """Run *run* on each packed document of each row and rejoin."""
    rows = []
    for row, row_seq_idx in zip(hidden_states.split(1), seq_idx, strict=True):
        lengths = torch.unique_consecutive(row_seq_idx, return_counts=True)[1]
        documents = row.split(lengths.tolist(), dim=1)
        rows.append(torch.cat([run(document) for document in documents], dim=1))
    return torch.cat(rows)


def _varlen_kernels_ready(mixer: nn.Module) -> bool:
    """Whether conv and scan honor packed document boundaries in one call."""
    conv = getattr(mixer, "causal_conv1d_fn", None)
    scan = getattr(mixer, "chunk_gated_delta_rule", None)
    return (
        conv is not None
        and scan is not None
        and getattr(scan, "__module__", "").startswith("fla")
    )


def _install_packed_mixer(mixer_cls: type) -> None:
    """Honor ``seq_idx`` / ``cu_seq_lens_q``, or run each document on its own."""
    cls: Any = mixer_cls
    forward = cls.__dict__["forward"]

    @functools.wraps(forward)
    def packed_mixer_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        cache_params: object | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        seq_idx = kwargs.get("seq_idx")
        if seq_idx is None:
            return forward(self, hidden_states, cache_params, attention_mask, **kwargs)
        # FLA varlen GDN only accepts a flattened packed row (B=1) with cu_seqlens.
        if _varlen_kernels_ready(self) and hidden_states.shape[0] == 1:
            if kwargs.get("cu_seq_lens_q") is None:
                kwargs = {**kwargs, "cu_seq_lens_q": _cu_seqlens(seq_idx)}
            return forward(self, hidden_states, cache_params, attention_mask, **kwargs)
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"seq_idx", "cu_seq_lens_q"}
        }
        return _per_document(
            lambda document: forward(
                self, document, cache_params, attention_mask, **kwargs
            ),
            hidden_states,
            seq_idx,
        )

    cls.forward = packed_mixer_forward


def _install_packed_decoder(block_cls: type) -> None:
    """Derive ``seq_idx`` from packed ``position_ids`` for GDN layers."""
    cls: Any = block_cls
    forward = cls.__dict__["forward"]

    @functools.wraps(forward)
    def packed_decoder_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if (
            getattr(self, "linear_attn", None) is not None
            and past_key_values is None
            and position_ids is not None
            and kwargs.get("seq_idx") is None
            and hidden_states.shape[0] == 1
        ):
            seq_idx = packed_seq_idx(position_ids)
            if seq_idx is not None:
                kwargs = {**kwargs, "seq_idx": seq_idx}
        return forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    cls.forward = packed_decoder_forward


def patch_qwen_gdn_packed_sequences(mixer: str, block: str) -> None:
    """Reset GDN conv and scan state at every packed-document boundary.

    :param mixer: Dotted path of the GatedDeltaNet class.
    :param block: Dotted path of the decoder layer that calls it.
    """
    mixer_cls = _resolve_class(mixer)
    block_cls = _resolve_class(block)
    if mixer_cls is None or block_cls is None:
        logger.warning(
            "[qwen-packed] %s unavailable; packed rows are not supported",
            mixer.rpartition(".")[2],
        )
        return
    if class_is_patched(mixer_cls, RESETS_AT_DOCUMENT_BOUNDARY):
        return
    _install_packed_mixer(mixer_cls)
    _install_packed_decoder(block_cls)
    setattr(mixer_cls, RESETS_AT_DOCUMENT_BOUNDARY, True)
    logger.info(
        "[qwen-packed] %s resets state at packed-document boundaries",
        mixer_cls.__name__,
    )


def install_qwen_gdn_patches(
    patch: PatchRuntimeConfig,
    model: object | None = None,
) -> None:
    """Install GDN packed-sequence patches for Qwen3.5 / Qwen3.6 families."""
    for mixer, block in PACKED_TARGETS:
        patch_qwen_gdn_packed_sequences(mixer, block)
