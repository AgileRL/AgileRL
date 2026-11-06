# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Reset Qwen GatedDeltaNet state at packed-document boundaries.

HF GDN consumes ``seq_idx`` (causal conv) and ``cu_seq_lens_q`` (FLA scan).
The packed trainer forward only sends restarting ``position_ids``, so the
text model derives ``seq_idx`` and the document offsets once per forward and
its decoder layers pass them to GDN. A padded batch carries a mask instead,
which GDN already applies, so its pad tokens never become documents.
causal-conv1d takes ``seq_idx`` per row; FLA's varlen scan takes one row, so
packed rows reach it flattened, with ``cu_seq_lens_q`` spanning every row.
When the fused kernels are absent, each document is run on its own: the torch
GDN fallback ignores ``cu_seqlens``. FLA scans also take the grouped q/k heads
as they are.
"""

from __future__ import annotations

import functools
import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from agilerl.architectures.qwen3_5.gated_delta_net import (
    GdnKernel,
    apply_gdn_conv_clone,
    gdn_kernel_import_errors,
)
from agilerl.architectures.qwen3_5.gated_norm import install_fused_gated_norm
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.llm_packing import (
    RESETS_AT_DOCUMENT_BOUNDARY,
    packed_layout,
    run_per_document,
)
from agilerl.utils.patching import class_is_patched, library_shim, try_import

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

__all__ = [
    "GROUPED_QK_HEADS",
    "PACKED_CU_SEQLENS",
    "VARLEN_SCAN",
    "install_qwen_gdn_patches",
    "patch_qwen_gdn_grouped_heads",
    "patch_qwen_gdn_packed_sequences",
    "resolve_class",
]

# Module-level GDN chunk scan that transformers backs with FLA when it imports.
VARLEN_SCAN = "torch_chunk_gated_delta_rule"

# Decoder-layer kwarg carrying the flattened document offsets to GDN. Not
# ``cu_seq_lens_q``: FlashAttention layers of the same model would consume it.
PACKED_CU_SEQLENS = "agilerl_packed_cu_seqlens"

# Marks a GDN mixer class whose forward passes grouped q/k heads to the scans.
GROUPED_QK_HEADS = "agilerl_grouped_qk_heads"

# The fused conv and scan transformers swaps in for its GDN torch kernels.
PACKED_GDN_KERNELS = (
    GdnKernel("causal_conv1d", "causal_conv1d_fn", "causal-conv1d"),
    GdnKernel(
        "fla.ops.gated_delta_rule", "chunk_gated_delta_rule", "flash-linear-attention"
    ),
)


def resolve_class(dotted: str) -> type | None:
    """Resolve ``module.Class``, or None when the module cannot be imported.

    :param dotted: Dotted ``module.Class`` path.
    :return: The class, or None when its module is absent.
    """
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


def _flatten_varlen_rows(scan: Callable[..., Any]) -> Callable[..., Any]:
    """Run a varlen scan over packed rows as the one flattened row FLA accepts."""

    @functools.wraps(scan)
    def flattened_scan(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rows = query.shape[0]
        if cu_seqlens is None or rows == 1:
            return scan(
                query, key, value, g=g, beta=beta, cu_seqlens=cu_seqlens, **kwargs
            )
        query, key, value, g, beta = (
            tensor.flatten(0, 1).unsqueeze(0) for tensor in (query, key, value, g, beta)
        )
        out, state = scan(
            query, key, value, g=g, beta=beta, cu_seqlens=cu_seqlens, **kwargs
        )
        return out.reshape(rows, -1, *out.shape[2:]), state

    return flattened_scan


def _install_varlen_scan(mixer_cls: type) -> None:
    """Wrap the GDN scan the mixer's module calls so packed rows share one ``cu_seqlens``."""
    module = vars(sys.modules[mixer_cls.__module__])
    if VARLEN_SCAN not in module:
        message = (
            f"[qwen-packed] {mixer_cls.__module__} is present but missing {VARLEN_SCAN}"
        )
        raise RuntimeError(message)
    module[VARLEN_SCAN] = _flatten_varlen_rows(module[VARLEN_SCAN])


def _varlen_kernels_ready() -> bool:
    """Whether conv and scan honor packed document boundaries in one call.

    transformers resolves the GDN conv and scan to causal-conv1d and FLA when
    those packages import; its PyTorch fallbacks drop ``seq_idx`` / ``cu_seqlens``.
    """
    return not gdn_kernel_import_errors(PACKED_GDN_KERNELS)


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
        cu_seqlens = kwargs.pop(PACKED_CU_SEQLENS, None)
        seq_idx = kwargs.get("seq_idx")
        if seq_idx is None:
            return forward(self, hidden_states, cache_params, attention_mask, **kwargs)
        if _varlen_kernels_ready():
            if cu_seqlens is not None:
                # A fresh tensor per call, also over offsets the caller passed
                # for FlashAttention: FLA caches on its identity, and a cache
                # hit in a checkpoint recompute skips ops the forward ran.
                kwargs["cu_seq_lens_q"] = cu_seqlens.clone()
            return forward(self, hidden_states, cache_params, attention_mask, **kwargs)
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"seq_idx", "cu_seq_lens_q"}
        }
        return run_per_document(
            lambda document: forward(
                self, document, cache_params, attention_mask, **kwargs
            ),
            hidden_states,
            seq_idx,
        )

    cls.forward = packed_mixer_forward


def _text_position_ids(position_ids: torch.Tensor) -> torch.Tensor | None:
    """``(B, L)`` positions the text model gives its decoder layers, or None."""
    if position_ids.ndim == 2:
        return position_ids
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[0]
    return None


def _install_packed_text_model(model_cls: type) -> None:
    """Derive ``seq_idx`` and document offsets once per forward for every decoder layer."""
    cls: Any = model_cls
    forward = cls.__dict__["forward"]

    @functools.wraps(forward)
    def packed_text_model_forward(
        self: PreTrainedModel,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        **kwargs: Any,
    ) -> object:
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = self.config.use_cache
        positions = None
        if (
            position_ids is not None
            and past_key_values is None
            and not use_cache
            and attention_mask is None
        ):
            positions = _text_position_ids(position_ids)
        if positions is not None:
            # Kept out of the checkpointed layers: selective checkpointing
            # replays host copies by call order, and the layout is memoized.
            layout = packed_layout(positions)
            if layout is not None:
                kwargs = {
                    **kwargs,
                    "seq_idx": layout.seq_idx,
                    PACKED_CU_SEQLENS: layout.cu_seqlens,
                }
        return forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    cls.forward = packed_text_model_forward


def patch_qwen_gdn_packed_sequences(mixer: str, text_model: str) -> None:
    """Reset GDN conv and scan state at every packed-document boundary.

    :param mixer: Dotted path of the GatedDeltaNet class.
    :param text_model: Dotted path of the text model whose decoder layers call it.
    """
    mixer_cls = resolve_class(mixer)
    text_model_cls = resolve_class(text_model)
    if mixer_cls is None or text_model_cls is None:
        logger.warning(
            "[qwen-packed] %s unavailable; packed rows are not supported",
            mixer.rpartition(".")[2],
        )
        return
    if class_is_patched(mixer_cls, RESETS_AT_DOCUMENT_BOUNDARY):
        return
    _install_varlen_scan(mixer_cls)
    _install_packed_mixer(mixer_cls)
    _install_packed_text_model(text_model_cls)
    setattr(mixer_cls, RESETS_AT_DOCUMENT_BOUNDARY, True)
    logger.info(
        "[qwen-packed] %s resets state at packed-document boundaries",
        mixer_cls.__name__,
    )


def patch_qwen_gdn_grouped_heads(mixer: str) -> None:
    """Pass FLA's GDN scans the grouped q/k heads instead of q/k repeated to every value head.

    FLA maps value head ``i`` to key head ``i // (num_v_heads // num_k_heads)``,
    the head ``repeat_interleave`` copies into it, and sums the repeated q/k
    gradients inside its backward. The PyTorch fallback scans need the repeat.

    :param mixer: Dotted path of the GatedDeltaNet class.
    """
    mixer_cls = resolve_class(mixer)
    if mixer_cls is None or class_is_patched(mixer_cls, GROUPED_QK_HEADS):
        return
    cls: Any = mixer_cls
    forward = cls.__dict__["forward"]

    @functools.wraps(forward)
    def grouped_heads_forward(self: nn.Module, *args: Any, **kwargs: Any) -> object:
        if not _varlen_kernels_ready():
            return forward(self, *args, **kwargs)
        num_k_heads = self.num_k_heads
        # HF repeats q/k heads before the scan only while num_k_heads < num_v_heads.
        self.num_k_heads = self.num_v_heads
        try:
            return forward(self, *args, **kwargs)
        finally:
            self.num_k_heads = num_k_heads

    cls.forward = grouped_heads_forward
    setattr(mixer_cls, GROUPED_QK_HEADS, True)


def _require_gdn_fast_path() -> None:
    """Fail GPU Qwen GDN setup when fused conv/scan kernels are missing."""
    if not torch.cuda.is_available():
        return
    errors = gdn_kernel_import_errors(PACKED_GDN_KERNELS)
    if errors:
        msg = "Qwen GatedDeltaNet needs " + " and ".join(errors) + " on GPU"
        raise RuntimeError(msg)


@library_shim("transformers", "==5.19.*")
def install_qwen_gdn_patches(
    patch: PatchRuntimeConfig,
    model: object | None = None,
    use_sequence_packing: bool = False,
) -> None:
    """Install the catalog GDN patches and the fused gated norm for Qwen3.5 / Qwen3.6 families.

    :param patch: Catalog patch config.
    :param model: Already-built model, or None; the class-level patches cover it.
    :param use_sequence_packing: Install the packed-sequence wrap.
    :type use_sequence_packing: bool, optional
    """
    _require_gdn_fast_path()
    gdn = patch.gdn
    if gdn is None:
        msg = "Qwen Gated DeltaNet patches need a catalog gdn config"
        raise ValueError(msg)
    for entry in gdn.mixers:
        if use_sequence_packing:
            patch_qwen_gdn_packed_sequences(entry.mixer, entry.text_model)
        patch_qwen_gdn_grouped_heads(entry.mixer)
    install_fused_gated_norm()
    if isinstance(model, nn.Module) and gdn.clone_conv_output:
        apply_gdn_conv_clone(model, tuple(entry.mixer for entry in gdn.mixers))
