"""Lean causal LM + value head for PPO (replaces TRL experimental wrapper).

No dtype upcasts/downcasts in the value head or wrapper forward; tensors stay in
the dtype of hidden states and ``nn.Linear`` weights.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

try:
    from peft import PeftModel, get_peft_model
except ImportError:  # pragma: no cover
    PeftModel = None  # type: ignore[misc, assignment]
    get_peft_model = None  # type: ignore[misc, assignment]


VALUE_HEAD_KWARGS = (
    "summary_dropout_prob",
    "v_head_initializer_range",
    "v_head_init_strategy",
)


def _resolve_hidden_size(config: Any) -> int:
    if getattr(config, "word_embed_proj_dim", None) is not None:
        return int(config.word_embed_proj_dim)
    hidden = getattr(config, "hidden_size", None)
    if hidden is None:
        msg = "Cannot infer value-head hidden size from config."
        raise ValueError(msg)
    return int(hidden)


def _pop_value_head_kwargs(kwargs: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in VALUE_HEAD_KWARGS:
        if key in kwargs:
            out[key] = kwargs.pop(key)
    return out


class PPOValueHead(nn.Module):
    """Per-token scalar value: Linear(hidden -> 1). Attribute ``summary`` matches PEFT ``modules_to_save``."""

    def __init__(
        self,
        config: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        kwargs.pop("v_head_initializer_range", None)
        kwargs.pop("v_head_init_strategy", None)
        hidden_size = _resolve_hidden_size(config)
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.summary(hidden_states)


class AutoModelForCausalLMWithValueHead(nn.Module):
    """Wraps ``AutoModelForCausalLM`` and appends a value head; forward returns ``(logits, loss, values)``."""

    def __init__(
        self,
        pretrained_model: PreTrainedModel,
        *,
        summary_dropout_prob: float | None = None,
        v_head_initializer_range: float = 0.2,
        v_head_init_strategy: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _pop_value_head_kwargs(kwargs)
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        self.is_loaded_in_8bit = getattr(pretrained_model, "is_loaded_in_8bit", False)
        self.is_loaded_in_4bit = getattr(pretrained_model, "is_loaded_in_4bit", False)
        self.is_sequential_parallel = False
        self.is_peft_model = PeftModel is not None and isinstance(
            pretrained_model,
            PeftModel,
        )

        if hasattr(pretrained_model, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable = (
                pretrained_model.gradient_checkpointing_disable
            )
        if hasattr(pretrained_model, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable = (
                pretrained_model.gradient_checkpointing_enable
            )
        if hasattr(pretrained_model, "enable_input_require_grads"):
            self.enable_input_require_grads = (
                pretrained_model.enable_input_require_grads
            )

        self.prepare_inputs_for_generation = (
            pretrained_model.prepare_inputs_for_generation
        )

        v_kw: dict[str, Any] = {}
        if summary_dropout_prob is not None:
            v_kw["summary_dropout_prob"] = summary_dropout_prob
        self.v_head = PPOValueHead(self.pretrained_model.config, **v_kw)

        if v_head_init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(
                mean=0.0, std=v_head_initializer_range
            )
            self.v_head.summary.bias.data.zero_()

    @property
    def name_or_path(self) -> str:
        return getattr(self.pretrained_model, "name_or_path", "")

    @name_or_path.setter
    def name_or_path(self, value: str) -> None:
        self.pretrained_model.name_or_path = value

    @property
    def generation_config(self):
        return self.pretrained_model.generation_config

    @generation_config.setter
    def generation_config(self, value) -> None:
        self.pretrained_model.generation_config = value

    @property
    def device(self) -> torch.device:
        return self.pretrained_model.device

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        attention_mask: torch.Tensor | None = None,
        return_past_key_values: bool = False,
        **kwargs: Any,
    ):
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values

        inner_kw = {
            k: v
            for k, v in kwargs.items()
            if k not in {"return_dict", "output_hidden_states"}
        }
        base_out = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            **inner_kw,
        )
        lm_logits = base_out.logits
        loss = getattr(base_out, "loss", None)
        if base_out.hidden_states is None:
            msg = "Base model did not return hidden_states (output_hidden_states must be True)."
            raise RuntimeError(msg)
        last_hidden_state = base_out.hidden_states[-1]
        head_dtype = self.v_head.summary.weight.dtype
        if last_hidden_state.dtype != head_dtype:
            last_hidden_state = last_hidden_state.to(head_dtype)
        value = self.v_head(last_hidden_state).squeeze(-1)

        if return_past_key_values:
            return (lm_logits, loss, value, base_out.past_key_values)
        return (lm_logits, loss, value)

    def generate(self, *args: Any, **kwargs: Any):
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(
                *args, **kwargs
            )
        else:
            pretrained_model_state_dict = {}
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for key, tensor in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{key}"] = tensor
        return pretrained_model_state_dict

    def post_init(self, state_dict: dict[str, Any]) -> None:
        """Load ``v_head.*`` tensors from a checkpoint dict (non-strict)."""
        v_sd = {}
        for key in list(state_dict.keys()):
            if key.startswith("v_head."):
                v_sd[key[len("v_head.") :]] = state_dict[key]
        if v_sd:
            self.v_head.load_state_dict(v_sd, strict=False)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        state_dict = kwargs.get("state_dict")
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict
        if self.is_peft_model:
            save_dir = args[0]
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
            kwargs.pop("state_dict", None)
        return self.pretrained_model.save_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PreTrainedModel,
        *model_args: Any,
        **kwargs: Any,
    ) -> AutoModelForCausalLMWithValueHead:
        v_head_kw = _pop_value_head_kwargs(kwargs)
        peft_config = kwargs.pop("peft_config", None)
        kwargs.pop("is_trainable", None)
        kwargs.pop("reward_adapter", None)
        kwargs.pop("reward_adapter_name", None)

        resume_sd: dict[str, Any] | None = None
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **kwargs,
            )
            if peft_config is not None:
                if get_peft_model is None:
                    msg = "peft is required when peft_config is set."
                    raise ImportError(msg)
                pretrained_model = get_peft_model(pretrained_model, peft_config)
            resume_sd = cls._maybe_load_resume_state_dict(pretrained_model_name_or_path)
        elif isinstance(pretrained_model_name_or_path, PreTrainedModel):
            pretrained_model = pretrained_model_name_or_path
            if peft_config is not None and get_peft_model is not None:
                pretrained_model = get_peft_model(pretrained_model, peft_config)
            resume_sd = pretrained_model.state_dict()
        else:
            msg = (
                "pretrained_model_name_or_path must be str or PreTrainedModel, "
                f"got {type(pretrained_model_name_or_path)}"
            )
            raise TypeError(msg)

        obj = cls(pretrained_model, **v_head_kw)
        if resume_sd is not None and any(k.startswith("v_head.") for k in resume_sd):
            obj.post_init(dict(resume_sd))
        return obj

    @staticmethod
    def _maybe_load_resume_state_dict(path: str) -> dict[str, Any] | None:
        """Load full state dict from local dir if present (for PPO resume with value head)."""
        if not os.path.isdir(path):
            return None
        safe_path = os.path.join(path, "model.safetensors")
        bin_path = os.path.join(path, "pytorch_model.bin")
        try:
            if os.path.isfile(safe_path):
                from safetensors.torch import load_file

                return dict(load_file(safe_path))
            if os.path.isfile(bin_path):
                return torch.load(bin_path, map_location="cpu", weights_only=True)
        except Exception:
            return None
        return None
