# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Liger kernel patches for HuggingFace ``nemotron_h`` (hybrid Mamba/attn/MoE)."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES

if TYPE_CHECKING:
    import torch
    from transformers.cache_utils import Cache
    from transformers.modeling_utils import PreTrainedModel

REGISTERED = {"value": False}

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from transformers.models.nemotron_h import modeling_nemotron_h
else:
    modeling_nemotron_h = None

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.model.llama import lce_maybe_trainable_lm_head
    from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
    from liger_kernel.transformers.model.output_classes import (
        LigerCausalLMOutputWithPast,
    )
    from liger_kernel.transformers.monkey_patch import (
        MODEL_TYPE_TO_APPLY_LIGER_FN,
        _patch_rms_norm_module,
    )
    from liger_kernel.transformers.relu_squared import LigerReLUSquared
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    HAS_LIGER = True
else:
    # Keep names resolvable when liger-kernel isn't installed; call sites
    # guard on HAS_LIGER / modeling_nemotron_h before use.
    HAS_LIGER = False
    MODEL_TYPE_TO_APPLY_LIGER_FN: dict[str, Any] = {}
    _patch_rms_norm_module = None  # type: ignore[assignment]
    LigerRMSNorm = None  # type: ignore[assignment]
    LigerReLUSquared = None  # type: ignore[assignment]
    liger_rotary_pos_emb = None  # type: ignore[assignment]
    LigerCrossEntropyLoss = None  # type: ignore[assignment]
    lce_maybe_trainable_lm_head = None  # type: ignore[assignment]
    unpack_cross_entropy_result = None  # type: ignore[assignment]
    LigerCausalLMOutputWithPast = None  # type: ignore[assignment]


def lce_forward(
    self: Any,  # noqa: ANN401 -- bound as NemotronHForCausalLM.forward via MethodType
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    skip_logits: bool | None = None,
    **kwargs: Any,
) -> tuple[Any, ...] | Any:  # noqa: ANN401 -- mirrors HF CausalLMOutputWithPast / tuple forms
    """Fused linear cross-entropy forward for NemotronHForCausalLM."""
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits and labels is None and shift_labels is None:
        msg = "skip_logits is True, but labels and shift_labels are None"
        raise ValueError(msg)

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        result = lce_maybe_trainable_lm_head(
            self,
            hidden_states=kept_hidden_states,
            hidden_size=self.config.hidden_size,
            labels=labels,
            shift_labels=shift_labels,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output: tuple[Any, ...] = (logits, *outputs[1:])
        if loss is not None:
            output = (loss, *output)
        if token_accuracy is not None:
            output = (*output, token_accuracy)
        if predicted_tokens is not None:
            output = (*output, predicted_tokens)
        return output

    return LigerCausalLMOutputWithPast(
        loss=loss,  # ty: ignore[invalid-argument-type]  # unpack/loss_function widen to Tensor|Any; Liger output expects FloatTensor|None
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        token_accuracy=token_accuracy,  # ty: ignore[invalid-argument-type]  # unpack widens to Tensor|None; Liger output expects FloatTensor|None
        predicted_tokens=predicted_tokens,  # ty: ignore[invalid-argument-type]  # unpack widens to Tensor|None; Liger output expects LongTensor|None
    )


def _patch_relu2_mixer(
    mixer: Any,  # noqa: ANN401 -- HF Nemotron-H MLP/MoE mixer walked via getattr
    block_type: str | None,
) -> None:
    """Set Liger ReLU² on MLP / MoE expert activations; skip Mamba mixers."""
    if block_type == "mlp" and hasattr(mixer, "act_fn"):
        mixer.act_fn = LigerReLUSquared()
        return
    if block_type == "moe":
        shared = getattr(mixer, "shared_experts", None)
        if shared is not None and hasattr(shared, "act_fn"):
            shared.act_fn = LigerReLUSquared()
        experts = getattr(mixer, "experts", None)
        if experts is not None and hasattr(experts, "act_fn"):
            experts.act_fn = LigerReLUSquared()


def apply_liger_kernel_to_nemotron_h(
    rms_norm: bool = True,
    rope: bool = True,
    relu_squared: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    model: PreTrainedModel | None = None,
    **kwargs: Any,
) -> None:
    """Apply Liger kernels to HuggingFace ``nemotron_h`` models.

    :param rms_norm: Patch RMSNorm with LigerRMSNorm.
    :type rms_norm: bool
    :param rope: Patch rotary embeddings with Liger RoPE.
    :type rope: bool
    :param relu_squared: Patch ``relu2`` activations with LigerReLUSquared.
    :type relu_squared: bool
    :param cross_entropy: Use LigerCrossEntropyLoss (mutually exclusive with LCE).
    :type cross_entropy: bool
    :param fused_linear_cross_entropy: Replace CausalLM forward with fused LCE.
    :type fused_linear_cross_entropy: bool
    :param model: Optional loaded model for instance-level patches.
    :type model: PreTrainedModel | None
    """
    if not HAS_LIGER or modeling_nemotron_h is None:
        msg = (
            "liger-kernel and LLM dependencies are required to apply "
            "Nemotron-H Liger patches"
        )
        raise ImportError(msg)

    if cross_entropy and fused_linear_cross_entropy:
        msg = "cross_entropy and fused_linear_cross_entropy cannot both be True."
        raise ValueError(msg)

    if rope:
        modeling_nemotron_h.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_nemotron_h.NemotronHRMSNorm = LigerRMSNorm  # ty: ignore[invalid-assignment]  # intentional Liger drop-in for HF NemotronHRMSNorm
    if relu_squared:
        modeling_nemotron_h.ACT2FN["relu2"] = LigerReLUSquared
    if cross_entropy:
        modeling_nemotron_h.CrossEntropyLoss = LigerCrossEntropyLoss  # ty: ignore[unresolved-attribute]  # monkey-patched onto the HF module at apply time
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(lce_forward, model)
        else:
            modeling_nemotron_h.NemotronHForCausalLM.forward = lce_forward

    if model is not None:
        base_model = getattr(model, model.base_model_prefix, model)
        if rms_norm:
            _patch_rms_norm_module(base_model.norm_f)
        for layer in base_model.layers:  # ty: ignore[not-iterable]  # Nemotron-H layers is a ModuleList at runtime
            if rms_norm:
                _patch_rms_norm_module(layer.norm)
            if relu_squared:
                _patch_relu2_mixer(
                    layer.mixer,  # ty: ignore[unresolved-attribute]  # Nemotron-H block mixer; getattr(model) widens layer type
                    getattr(layer, "block_type", None),
                )


def register_nemotron_h_liger() -> bool:
    """Register ``nemotron_h`` with Liger's AutoLiger apply map.

    Idempotent. Returns False when liger-kernel or LLM deps are unavailable.

    :return: Whether ``nemotron_h`` is registered for AutoLiger.
    :rtype: bool
    """
    if not HAS_LIGER or modeling_nemotron_h is None:
        return False
    if REGISTERED["value"]:
        return True
    MODEL_TYPE_TO_APPLY_LIGER_FN["nemotron_h"] = apply_liger_kernel_to_nemotron_h  # ty: ignore[invalid-assignment]  # extend Liger's closed apply-fn union with nemotron_h
    REGISTERED["value"] = True
    return True
