# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Validate and default an :class:`LLMSetup` before ``LLMAlgorithm`` binds it."""

from __future__ import annotations

import warnings
from dataclasses import replace

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES
from agilerl.algorithms.configs import LLMSetup, LLMTrainSetup, LLMVLLMSetup

if HAS_LLM_DEPENDENCIES:
    from peft import LoraConfig
else:
    LoraConfig = None


def named_llm_setup(llm: LLMSetup, name: str) -> LLMSetup:
    """Set ``runtime.name`` when the caller did not supply one."""
    runtime = llm.runtime
    return replace(llm, runtime=replace(runtime, name=runtime.name or name))


def prepare_llm_setup(llm: LLMSetup) -> LLMSetup:
    """Apply LoRA, Liger, and vLLM defaults before the algorithm binds *llm*."""
    if not HAS_LLM_DEPENDENCIES:
        msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
        raise ImportError(msg)
    model = llm.model
    train = llm.train
    vllm = llm.vllm
    if model.model_name is None and model.actor_network is None:
        msg = "At least one of model_name or actor_network must be provided."
        raise ValueError(msg)
    train = _prepare_liger(train)
    model = replace(model, lora_config=_default_lora_config(model.lora_config, train))
    vllm = _prepare_vllm(vllm)
    return replace(llm, model=model, train=train, vllm=vllm)


def _prepare_liger(train: LLMTrainSetup) -> LLMTrainSetup:
    if train.use_liger_loss and not HAS_LIGER_KERNEL:
        warnings.warn(
            "use_liger_loss=True requested, but `liger-kernel` is not available on this platform/environment. "
            "Falling back to standard loss.",
            stacklevel=3,
        )
        return replace(train, use_liger_loss=False)
    return train


def _default_lora_config(lora_config: object, train: LLMTrainSetup) -> object:
    if lora_config is None:
        warnings.warn(
            "No LoRA config provided. AgileRL can only be used to finetune adapters at present. "
            "Using default LoRA configuration for RL finetuning: "
            "r=16, lora_alpha=32, target_modules='all-linear', task_type='CAUSAL_LM', lora_dropout=0.05."
            "To use a different LoRA configuration, please pass lora_config to the constructor.",
            stacklevel=3,
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
    if train.use_liger_loss:
        warnings.warn(
            "Liger Loss used with LoRA, deactivating LoRA for the lm_head by setting exclude_modules to ['lm_head']",
            stacklevel=3,
        )
        lora_config.exclude_modules = ["lm_head"]
    return lora_config


def _prepare_vllm(vllm: LLMVLLMSetup) -> LLMVLLMSetup:
    use_memory_efficient_params = vllm.use_memory_efficient_params
    vllm_config = vllm.vllm_config
    if not vllm.use_vllm:
        use_memory_efficient_params = False
    if vllm_config is not None and not vllm.use_vllm:
        warnings.warn(
            "vllm_config is provided but use_vllm is False. Setting vllm_config to None.",
            stacklevel=3,
        )
        vllm_config = None
    return replace(
        vllm,
        use_memory_efficient_params=use_memory_efficient_params,
        vllm_config=vllm_config,
    )
