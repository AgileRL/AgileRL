# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GraniteMoeHybrid patches: packed documents and Liger registration.

``GraniteMoeHybridModel`` reads ``position_ids`` only for RoPE: neither its
decoder layers nor its mask builder see them, so a packed row would scan and
attend across documents. The model forward derives the document layout once
and passes ``seq_idx`` to the Mamba2 layers, FlashAttention varlen offsets to
the attention layers, and a block-diagonal mask to other attention backends.
On CUDA the fused Mamba2 kernel takes ``seq_idx``; otherwise each document
runs through the mixer on its own.

Liger has no entry for this family; the installer registers the
:mod:`agilerl.architectures.granite_liger` appliers when liger-kernel is
installed.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import torch

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES
from agilerl.architectures.nemotron_h.mamba import install_packed_mixer
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.llm_packing import RESETS_AT_DOCUMENT_BOUNDARY, packed_layout
from agilerl.utils.patching import class_is_patched, library_shim, try_import

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from transformers.masking_utils import create_causal_mask

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

__all__ = [
    "install_granite_hybrid_patches",
    "patch_granite_hybrid_packed_sequences",
]

GRANITE_HYBRID_MODELING = (
    "transformers.models.granitemoehybrid.modeling_granitemoehybrid"
)


def _install_packed_model(model_cls: type) -> None:
    """Route a packed row's document layout from ``position_ids`` to every decoder layer."""
    cls: Any = model_cls
    forward = cls.__dict__["forward"]

    @functools.wraps(forward)
    def packed_model_forward(
        self: PreTrainedModel,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, Any] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> object:
        caching = self.config.use_cache if use_cache is None else use_cache
        layout = None
        # A padding mask or a cache means the row is not packed.
        if (
            attention_mask is None
            and past_key_values is None
            and not caching
            and position_ids is not None
        ):
            layout = packed_layout(position_ids)
        if layout is not None:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
                input_ids = None
            attention_mask = {
                "full_attention": create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=position_ids,
                ),
                "linear_attention": None,
            }
            kwargs = {
                **kwargs,
                "seq_idx": layout.seq_idx,
                "cu_seq_lens_q": layout.cu_seqlens,
                "cu_seq_lens_k": layout.cu_seqlens,
                "max_length_q": layout.max_seqlen,
                "max_length_k": layout.max_seqlen,
            }
        return forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    cls.forward = packed_model_forward


def patch_granite_hybrid_packed_sequences() -> None:
    """Reset Mamba2 state and confine attention to each packed document.

    Unpacked rows take the unpatched path. The Mamba2 layer class is then
    marked with :data:`~agilerl.utils.llm_packing.RESETS_AT_DOCUMENT_BOUNDARY`.

    :return: None
    :rtype: None
    """
    module = try_import(GRANITE_HYBRID_MODELING)
    if module is None:
        logger.warning(
            "[granite-packed] %s unavailable; packed rows are not supported",
            GRANITE_HYBRID_MODELING,
        )
        return
    mixer_cls = module.GraniteMoeHybridMambaLayer
    if class_is_patched(mixer_cls, RESETS_AT_DOCUMENT_BOUNDARY):
        return
    # The Granite mixer runs its kernels on the caller's stream.
    install_packed_mixer(mixer_cls, default_stream=False)
    _install_packed_model(module.GraniteMoeHybridModel)
    setattr(mixer_cls, RESETS_AT_DOCUMENT_BOUNDARY, True)
    logger.debug(
        "[granite-packed] %s resets state at packed-document boundaries",
        mixer_cls.__name__,
    )


@library_shim("transformers", "==5.19.*")
def install_granite_hybrid_patches(
    _patch: PatchRuntimeConfig,
    _model: PreTrainedModel | PeftModel | None = None,
    _use_sequence_packing: bool = False,
) -> None:
    """Install GraniteMoeHybrid packed-sequence patches and register its Liger applier.

    :param _patch: Catalog patch config.
    :type _patch: PatchRuntimeConfig
    :param _model: Already-built model, or None; the class-level patches cover it.
    :type _model: PreTrainedModel | PeftModel | None
    :param _use_sequence_packing: Unused; the hybrid class patches do not branch on it.
    :type _use_sequence_packing: bool, optional
    :return: None
    :rtype: None
    """
    patch_granite_hybrid_packed_sequences()
    if HAS_LIGER_KERNEL:
        # Defers liger-kernel, an optional extra.
        from agilerl.architectures.granite_liger import register_granite_liger

        register_granite_liger()
