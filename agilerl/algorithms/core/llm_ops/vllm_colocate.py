# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Helpers for a colocated vLLM rollout engine (rollout + HF/PEFT trainer in
one process, sharing the GPU via vLLM native sleep/wake).

vLLM and the trainer each hold their own base; only LoRA adapters are synced
per rollout. These helpers are independent of how the base is loaded:

* :func:`patch_vllm_lora_keep_resident` stops vLLM from zeroing the single
  persistent rollout-adapter slot between forwards.
* :func:`patch_vllm_strip_multimodal_towers` frees the GPU memory held by a
  multimodal base's unused vision/audio towers (text-only RL never runs them).
* :func:`get_vllm_internal_model` reaches the live ``nn.Module`` inside an
  in-process (``external_launcher``) engine.

See ``docs/llm_finetuning/quantization.rst`` for the full colocated picture.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, NoReturn

import torch

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "get_vllm_internal_model",
    "patch_vllm_lora_keep_resident",
    "patch_vllm_strip_multimodal_towers",
]


class _StrippedTower:
    """Falsy placeholder left in place of a tower freed by
    :func:`patch_vllm_strip_multimodal_towers`; any use of it raises.
    """

    def __init__(self, path: str) -> None:
        self._stripped_path = path

    def __bool__(self) -> bool:
        # Falsy so ``if self.vision_tower:``-style guards short-circuit.
        return False

    def _error(self, detail: str) -> str:
        return (
            f"Stripped multimodal tower '{self._stripped_path}' ({detail}). "
            "The tower was freed to save GPU memory for text-only RL training; "
            "this code path should not run. Set "
            "VLLMConfig(strip_multimodal_towers=False) (or omit "
            "--vllm-strip-multimodal-towers) to keep the towers loaded."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(self._error("called as a forward path"))

    def __getattr__(self, name: str) -> NoReturn:
        raise AttributeError(self._error(f"attribute '{name}' accessed"))


def patch_vllm_strip_multimodal_towers(
    llm: Any,  # noqa: ANN401 -- opaque vLLM engine handle walked via getattr
    tower_attrs: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    """Free the GPU memory held by a multimodal base's unused towers.

    Text-only RL rollouts never execute the vision/audio towers, so each tower
    attribute found on the model (or one level down, on ``model.model``) is
    replaced with a :class:`_StrippedTower` and its parameter storage freed.
    Must be called **after** ``LLM(...)`` returns (vLLM's init memory profile
    may touch the towers); idempotent. Checkpoints are unaffected — only the
    LoRA adapter is saved.

    :param llm: A constructed in-process ``vllm.LLM`` (external_launcher).
    :type llm: vllm.LLM
    :param tower_attrs: Attribute names to strip; ``None`` uses the standard
        HF names.
    :type tower_attrs: tuple[str, ...] | list[str] | None
    :return: Mapping ``{tower_path: param_count_freed}``; empty if there is
        nothing to strip or the model could not be reached.
    :rtype: dict[str, int]
    """
    if tower_attrs is None:
        # Standard HF names for *ForConditionalGeneration wrappers, plus
        # Gemma-4-style per-modality embedders.
        tower_attrs = (
            "vision_tower",
            "audio_tower",
            "multi_modal_projector",
            "embed_vision",
            "embed_audio",
        )
    try:
        model = get_vllm_internal_model(llm)
    except Exception:
        return {}

    holders = [(model, "")]
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        holders.append((inner, "model."))

    freed: dict[str, int] = {}
    for holder, prefix in holders:
        for attr in tower_attrs:
            sub = getattr(holder, attr, None)
            if sub is None or isinstance(sub, _StrippedTower):
                continue
            try:
                n_params = sum(int(p.numel()) for p in sub.parameters())
            except Exception:
                n_params = 0
            if isinstance(holder, torch.nn.Module):
                # nn.Module.__setattr__ refuses to replace a registered child
                # with a non-Module; drop the registration first.
                holder._modules.pop(attr, None)
            path = f"{prefix}{attr}"
            setattr(holder, attr, _StrippedTower(path))
            freed[path] = n_params

    if freed:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return freed


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
        # A LoRA-wrapped layer exposes both reset_lora and the stacked GPU slot.
        if (
            hasattr(module, "reset_lora")
            and hasattr(module, "lora_b_stacked")
            and not getattr(module, "_agilerl_lora_resident", False)
        ):
            # Deliberate monkeypatch of a live vLLM module.
            module.reset_lora = lambda *args, **kwargs: None  # ty: ignore[invalid-assignment]
            module._agilerl_lora_resident = True  # ty: ignore[invalid-assignment]
            count += 1
    return count


def get_vllm_internal_model(llm: Any) -> nn.Module:  # noqa: ANN401 -- opaque vLLM engine handle walked via getattr
    """Return the live ``nn.Module`` inside an in-process vLLM ``LLM``.

    The other colocated patches need to mutate vLLM's running model in place —
    :func:`patch_vllm_lora_keep_resident` neutralizes ``reset_lora`` on its LoRA
    layers and :func:`patch_vllm_strip_multimodal_towers` frees its tower
    submodules. vLLM exposes no public accessor, so this walks the known
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
