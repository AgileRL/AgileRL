# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Build LLM algorithms from the spec, converting LoRA, vLLM and LR schedule sections at construction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import LLMAlgorithm
from agilerl.arena.memory.resolve.batch import resolve_micro_batch_size_per_gpu
from agilerl.arena.models.algorithms import (
    AlgorithmSpec,
    LLMAlgorithmSpec,
    RolloutLLMSpec,
)
from agilerl.arena.models.env import EnvSpec, LLMEnvSpec
from agilerl.arena.models.manifest import strips_multimodal_towers
from agilerl.arena.models.networks import CosineLRScheduleConfig as CosineLRScheduleSpec
from agilerl.arena.models.networks import LoraConfigDict, default_rollout_vllm_config
from agilerl.builders.base import (
    AlgorithmBuilder,
    AlgorithmBuildRuntime,
    apply_checkpoint,
    constructor_kwargs,
    spec_kwargs,
)
from agilerl.distributed import get_world_size, is_distributed
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig
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

    @staticmethod
    def _resolve_vllm_kwargs(kwargs: dict[str, Any], colocated: bool) -> None:
        """Set a colocated run's ``VLLMConfig``, or drop the vLLM kwargs of any other run.

        :param kwargs: Constructor kwargs, edited in place.
        :type kwargs: dict[str, Any]
        :param colocated: Whether vLLM runs in the trainer process.
        :type colocated: bool
        """
        if not colocated:
            kwargs.pop("max_model_len", None)
            kwargs.pop("vllm_config", None)
            return
        if kwargs.get("vllm_config") is None:
            kwargs["vllm_config"] = default_rollout_vllm_config()
        vllm_cfg = kwargs["vllm_config"]
        if isinstance(vllm_cfg, BaseModel):
            vllm_cfg = vllm_cfg.model_dump(exclude_none=True)
        if isinstance(vllm_cfg, dict):
            # Colocated generation shares the trainer's GPU, so vLLM sleeps during learn.
            vllm_cfg["sleep_mode"] = True
            kwargs["vllm_config"] = VLLMConfig(**vllm_cfg)

    @classmethod
    def build(
        cls,
        spec: AlgorithmSpec,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        runtime: AlgorithmBuildRuntime | None = None,
        actor_network: PreTrainedModel | PeftModel | None = None,
        rollout_mode: str = "",
        environment: EnvSpec | None = None,
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
        :param rollout_mode: ``training.rollout_mode``. Colocated runs vLLM in
            the trainer process.
        :type rollout_mode: str
        :param environment: The run's environment spec; a text-only one lets
            the engine skip a vision-language model's towers.
        :type environment: EnvSpec | None
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

        colocated = isinstance(spec, RolloutLLMSpec) and rollout_mode == "colocated"
        cls._resolve_vllm_kwargs(kwargs, colocated=colocated)
        if isinstance(spec, RolloutLLMSpec):
            kwargs["rollout_strip_multimodal_towers"] = strips_multimodal_towers(
                spec,
                isinstance(environment, LLMEnvSpec) and environment.text_only,
            )

        if kwargs.get("lora_config") is not None:
            kwargs["lora_tower_target_modules"] = kwargs[
                "lora_config"
            ].tower_target_modules
            kwargs["lora_config"] = peft_lora_config(kwargs["lora_config"])
        if kwargs.get("cosine_lr_schedule_config") is not None:
            kwargs["cosine_lr_schedule_config"] = cosine_lr_schedule_config(
                kwargs["cosine_lr_schedule_config"]
            )

        # Resolve trainer-side bitsandbytes quantization (a preset name or a
        # BitsAndBytesConfig kwargs dict) to the quantization_config the
        # algorithm constructor expects.
        if "quantization" in kwargs:
            kwargs["quantization_config"] = build_bnb_quantization_config(
                kwargs.pop("quantization")
            )

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
        kwargs["fsdp_config"] = spec.fsdp
        if "micro_batch_size_per_gpu" not in kwargs and is_distributed():
            kwargs["micro_batch_size_per_gpu"] = resolve_micro_batch_size_per_gpu(
                spec, get_world_size()
            )
        kwargs = constructor_kwargs(algo_cls, kwargs)
        algo = algo_cls(
            model_name=spec.pretrained_model_name_or_path,
            pad_token_id=pad_token_id,
            pad_token=tokenizer.pad_token,
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
    # Tower targets reach the algorithm as lora_tower_target_modules, not through peft.
    peft_lora = LoraConfigDict.model_validate(lora).model_dump(
        exclude={"tower_target_modules"}
    )
    peft_lora["r"] = peft_lora.pop("lora_r")
    return PeftLoraConfig(**peft_lora)


def cosine_lr_schedule_config(schedule: CosineLRScheduleSpec) -> CosineLRScheduleConfig:
    """Convert the manifest's LR schedule section to the config the algorithm takes.

    :param schedule: The manifest's ``cosine_lr_schedule_config`` section.
    :type schedule: CosineLRScheduleSpec
    :returns: The algorithm's warmup-cosine schedule config.
    :rtype: CosineLRScheduleConfig
    """
    return CosineLRScheduleConfig(**schedule.model_dump())
