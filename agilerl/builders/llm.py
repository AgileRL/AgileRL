# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Build LLM algorithms from the spec, converting LoRA and vLLM sections at construction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import LLMAlgorithm
from agilerl.arena.models.algorithms import (
    AlgorithmSpec,
    LLMAlgorithmSpec,
    RolloutLLMSpec,
)
from agilerl.arena.models.networks import LoraConfigDict
from agilerl.builders.base import (
    AlgorithmBuilder,
    AlgorithmBuildRuntime,
    apply_checkpoint,
    constructor_kwargs,
    spec_kwargs,
)
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import (
    apply_pad_token_id,
    build_bnb_quantization_config,
    load_pad_token_configs,
    resolve_pad_token_id,
)

if TYPE_CHECKING:
    from peft import LoraConfig, PeftModel
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class LLMBuilder(AlgorithmBuilder):
    """LLM fine-tuning."""

    @classmethod
    def algo_class(cls, spec: AlgorithmSpec) -> type[LLMAlgorithm]:
        resolved = super().algo_class(spec)
        if not issubclass(resolved, LLMAlgorithm):
            msg = (
                f"{type(spec).__name__} resolved to {resolved.__name__}, "
                "which is not a subclass of LLMAlgorithm."
            )
            raise TypeError(msg)
        return resolved

    @classmethod
    def build(
        cls,
        spec: AlgorithmSpec,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        runtime: AlgorithmBuildRuntime | None = None,
        actor_network: PreTrainedModel | PeftModel | None = None,
    ) -> LLMAlgorithm:
        """Build an LLM algorithm.

        :param spec: The algorithm spec.
        :type spec: AlgorithmSpec
        :param tokenizer: A HuggingFace ``AutoTokenizer`` instance.
        :type tokenizer: PreTrainedTokenizerBase | None
        :param runtime: Population slot, device, HPO, and optional checkpoint.
            ``index`` defaults to 0 when omitted.
        :type runtime: AlgorithmBuildRuntime | None
        :param actor_network: Pre-built or cloned actor. When provided it is
            handed to the constructor instead of loading the model from
            ``pretrained_model_name_or_path``.
        :type actor_network: PreTrainedModel | PeftModel | None
        :returns: LLM algorithm instance.
        :rtype: LLMAlgorithm
        :raises ValueError: If tokenizer is None.
        """
        if tokenizer is None:
            msg = "LLMBuilder.build requires a tokenizer."
            raise ValueError(msg)
        if not isinstance(spec, LLMAlgorithmSpec):
            msg = f"{type(spec).__name__} is not an LLMAlgorithmSpec."
            raise TypeError(msg)

        runtime = runtime or AlgorithmBuildRuntime(index=0)
        index = 0 if runtime.index is None else runtime.index

        # Only forward explicitly-set fields so the algorithm's own defaults
        # apply to everything a manifest omits, matching direct construction.
        kwargs = spec_kwargs(spec, hp_config=runtime.hp_config)
        kwargs.pop("pretrained_model_name_or_path", None)

        use_vllm = spec.use_vllm if isinstance(spec, RolloutLLMSpec) else False
        if not use_vllm:
            kwargs.pop("max_model_len", None)
            kwargs.pop("vllm_config", None)
        elif kwargs.get("vllm_config") is not None:
            vllm_cfg = kwargs["vllm_config"]
            if isinstance(vllm_cfg, BaseModel):
                vllm_cfg = vllm_cfg.model_dump(exclude_none=True)
            if isinstance(vllm_cfg, dict):
                kwargs["vllm_config"] = VLLMConfig(**vllm_cfg)

        if kwargs.get("lora_config") is not None:
            kwargs["lora_config"] = peft_lora_config(kwargs["lora_config"])

        # Resolve trainer-side bitsandbytes quantization (a preset name or a
        # BitsAndBytesConfig kwargs dict) to the quantization_config the
        # algorithm constructor expects.
        if "quantization" in kwargs:
            kwargs["quantization_config"] = build_bnb_quantization_config(
                kwargs.pop("quantization")
            )

        # A non-"auto" attn_implementation is forwarded through model_config so
        # the model-creation path treats it as authoritative.
        attn_implementation = kwargs.pop("attn_implementation", None)
        if attn_implementation is not None and attn_implementation != "auto":
            model_config = dict(kwargs.get("model_config") or {})
            model_config.setdefault("attn_implementation", attn_implementation)
            kwargs["model_config"] = model_config

        model_config = None
        generation_config = None
        if actor_network is not None:
            model_config = actor_network.config
            generation_config = getattr(actor_network, "generation_config", None)
        if model_config is None:
            model_config, generation_config = load_pad_token_configs(
                spec.pretrained_model_name_or_path
            )

        pad_token_id, pad_source = resolve_pad_token_id(
            tokenizer,
            model_config=model_config,
            generation_config=generation_config,
        )
        apply_pad_token_id(tokenizer, pad_token_id)
        logger.info(
            "Resolved algorithm pad_token_id=%s from %s (eos_token_id=%s)",
            pad_token_id,
            pad_source,
            getattr(tokenizer, "eos_token_id", None),
        )

        algo_cls = cls.algo_class(spec)
        kwargs = constructor_kwargs(algo_cls, kwargs)
        algo = algo_cls(
            model_name=spec.pretrained_model_name_or_path,
            pad_token_id=pad_token_id,
            pad_token=tokenizer.pad_token,
            accelerator=runtime.accelerator,
            index=index,
            device=runtime.device,
            actor_network=actor_network,
            **kwargs,
        )
        apply_checkpoint(
            algo,
            runtime.resume_from_checkpoint,
            runtime.load_weights_from,
            index=index,
        )
        return algo


def peft_lora_config(lora: LoraConfigDict | dict[str, Any] | LoraConfig) -> LoraConfig:
    """Convert the manifest's LoRA section to the peft object the algorithm takes.

    :param lora: The manifest's LoRA section, a mapping, or an already-built peft config.
    :type lora: LoraConfigDict | dict[str, Any] | LoraConfig
    :returns: A peft ``LoraConfig``.
    :raises ImportError: If the LLM extras are not installed.
    """
    if not HAS_LLM_DEPENDENCIES:
        msg = "LLM dependencies are required to resolve LoRA configuration."
        raise ImportError(msg)
    # peft is an optional LLM extra
    from peft import LoraConfig as PeftLoraConfig

    if isinstance(lora, PeftLoraConfig):
        return lora
    peft_lora = LoraConfigDict.model_validate(lora).model_dump()
    peft_lora["r"] = peft_lora.pop("lora_r")
    return PeftLoraConfig(**peft_lora)
