# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Helpers for a colocated vLLM rollout engine (rollout + HF/PEFT trainer in
one process, sharing the GPU via vLLM native sleep/wake).

vLLM and the trainer each hold their own base; only LoRA adapters are synced
per rollout. These helpers are independent of how the base is loaded:

* :func:`patch_vllm_lora_keep_resident` stops vLLM from zeroing the single
  persistent rollout-adapter slot between forwards.
* :func:`get_vllm_internal_model` reaches the live ``nn.Module`` inside an
  in-process (``external_launcher``) engine.

See ``docs/llm_finetuning/quantization.rst`` for the full colocated picture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "get_vllm_internal_model",
    "patch_vllm_3d_moe_lora_flag",
    "patch_vllm_lora_keep_resident",
]


def patch_vllm_3d_moe_lora_flag(model_name_or_path: str) -> bool:
    """Mark the model's vLLM class as taking stacked-3D MoE LoRA adapters.

    vLLM only parses the stacked-experts (PEFT ``target_parameters``) adapter
    format when the model class declares ``is_3d_moe_weight``; classes
    predating the flag reject the adapter in ``add_lora``. Call before engine
    construction — the colocated in-process worker then sees the class patch.
    """
    try:
        from transformers import AutoConfig
        from vllm.model_executor.models.registry import ModelRegistry

        architecture = AutoConfig.from_pretrained(model_name_or_path).architectures[0]
        model_cls = ModelRegistry.models[architecture].load_model_cls()
    except Exception:
        return False
    if getattr(model_cls, "is_3d_moe_weight", False):
        return False
    # setattr: the attribute is undeclared on model classes predating the
    # flag, so plain assignment is a type error.
    setattr(model_cls, "is_3d_moe_weight", True)  # noqa: B010
    return True


def patch_vllm_lora_keep_resident(llm: Any) -> int:  # noqa: ANN401 -- opaque vLLM engine handle walked via getattr
    """Keep vLLM's LoRA slot weights resident by neutralizing ``reset_lora``.

    vLLM (V1) zeroes a LoRA layer's GPU slot via ``reset_lora`` whenever a
    no-LoRA/dummy batch runs, but never re-copies the adapter afterwards
    (activation early-returns once the id is active) — the rollout adapter then
    silently contributes nothing. AgileRL drives a single persistent rollout
    adapter gated per token by vLLM's Punica index mapping, so the slot never
    needs clearing; a genuine adapter switch still overwrites it via
    ``set_lora``. Call once after the engine is constructed; idempotent.

    :param llm: A constructed in-process ``vllm.LLM`` (external_launcher).
    :type llm: Any
    :return: Number of LoRA layers neutralized (0 if LoRA is disabled or the
        model is unreachable).
    :rtype: int
    """
    try:
        model = get_vllm_internal_model(llm)
    except Exception:
        return 0

    count = 0
    for module in model.modules():
        # A LoRA-wrapped layer exposes reset_lora and a stacked GPU slot
        # (lora_b_stacked on linears, w13/w2_lora_b_stacked on fused MoE).
        if (
            hasattr(module, "reset_lora")
            and (
                hasattr(module, "lora_b_stacked")
                or hasattr(module, "w13_lora_b_stacked")
            )
            and not getattr(module, "_agilerl_lora_resident", False)
        ):
            # Deliberate monkeypatch of a live vLLM module.
            object.__setattr__(module, "reset_lora", lambda *args, **kwargs: None)
            object.__setattr__(module, "_agilerl_lora_resident", True)
            count += 1
    return count


def get_vllm_internal_model(llm: Any) -> nn.Module:  # noqa: ANN401 -- opaque vLLM engine handle walked via getattr
    """Return the live ``nn.Module`` inside an in-process vLLM ``LLM``.

    The other colocated patches need to mutate vLLM's running model in place —
    :func:`patch_vllm_lora_keep_resident` neutralizes ``reset_lora`` on its LoRA
    layers. vLLM exposes no public accessor, so this walks the known
    engine-core / executor attribute layouts to reach ``...model_runner.model``.

    :param llm: A constructed ``vllm.LLM`` instance.
    :type llm: Any
    :return: The underlying model module.
    :rtype: nn.Module
    :raises RuntimeError: If the model cannot be located.
    """
    engine = getattr(llm, "llm_engine", getattr(llm, "engine", llm))
    candidates = []
    core = getattr(engine, "engine_core", None)
    if core is not None:
        candidates.append(getattr(core, "engine_core", core))
    candidates.append(engine)

    def _model_from(base: Any) -> nn.Module | None:  # noqa: ANN401 -- opaque vLLM engine-core object walked via getattr
        try:
            return base.model_executor.driver_worker.model_runner.model
        except AttributeError:
            return None

    for base in candidates:
        model = _model_from(base)
        if model is not None:
            return model
    msg = (
        "Could not locate the vLLM internal model. Colocated vLLM requires an "
        "in-process engine (distributed_executor_backend='external_launcher')."
    )
    raise RuntimeError(msg)
