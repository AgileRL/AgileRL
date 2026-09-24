# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Liger kernel patches for HuggingFace ``nemotron_h`` (hybrid Mamba/attn/MoE)."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any, cast

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
    HAS_LIGER = False
    MODEL_TYPE_TO_APPLY_LIGER_FN: dict[str, Any] = {}
    _patch_rms_norm_module: Any = None
    LigerRMSNorm: Any = None
    LigerReLUSquared: Any = None
    liger_rotary_pos_emb: Any = None
    LigerCrossEntropyLoss: Any = None
    lce_maybe_trainable_lm_head: Any = None
    unpack_cross_entropy_result: Any = None
    LigerCausalLMOutputWithPast: Any = None


def lce_forward(
    self: modeling_nemotron_h.NemotronHForCausalLM,
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
) -> tuple[Any, ...] | LigerCausalLMOutputWithPast:
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
        loss=cast("torch.FloatTensor | None", loss),
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        token_accuracy=cast("torch.FloatTensor | None", token_accuracy),
        predicted_tokens=cast("torch.LongTensor | None", predicted_tokens),
    )


def _patch_relu2_mixer(
    mixer: torch.nn.Module,
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
    fused_linear_cross_entropy: bool = False,
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
    :param fused_linear_cross_entropy: Replace CausalLM ``forward`` with fused LCE.
        Defaults to ``False``: learn identity-patches ``lm_head`` and scores via
        fused logprobs, and LCE's ``self.model(...)`` call makes the first FSDP
        hook a nested unit so prefetch hits an empty comm context.
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

    nemotron_mod: Any = modeling_nemotron_h
    if rope:
        nemotron_mod.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        nemotron_mod.NemotronHRMSNorm = LigerRMSNorm
    if relu_squared:
        nemotron_mod.ACT2FN["relu2"] = LigerReLUSquared
    if cross_entropy:
        nemotron_mod.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(lce_forward, model)
        else:
            nemotron_mod.NemotronHForCausalLM.forward = lce_forward

    if model is not None:
        base_model = getattr(model, model.base_model_prefix, model)
        if rms_norm:
            _patch_rms_norm_module(base_model.norm_f)
        layers = cast("list[Any]", base_model.layers)
        for layer in layers:
            inner = list(getattr(layer, "blocks", ())) or [layer]
            for block in inner:
                if rms_norm:
                    _patch_rms_norm_module(block.norm)
                if relu_squared:
                    _patch_relu2_mixer(
                        block.mixer,
                        getattr(block, "block_type", None),
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
    apply_fns = cast("dict[str, Any]", MODEL_TYPE_TO_APPLY_LIGER_FN)
    apply_fns["nemotron_h"] = apply_liger_kernel_to_nemotron_h
    REGISTERED["value"] = True
    return True
