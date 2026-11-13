# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Architecture families: runtime catalog, patches and parallel plans.

Every subpackage declares one family as ``FAMILY``;
:mod:`agilerl.architectures.catalog` discovers them and maps each Hugging Face
``model_type`` to its family. :func:`family_runtime` reads ``config.json``
then looks up that type. :func:`install_family_patches` registers the family's
Liger applier and installs its text and vision patches.
:func:`family_tensor_parallel_plans` and :func:`family_context_parallel_plans`
resolve every family's
:class:`~agilerl.architectures.runtime.TensorParallelPlan` and
:class:`~agilerl.architectures.runtime.ContextParallelPlan`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from agilerl import HAS_LIGER_KERNEL
from agilerl.architectures.catalog import (
    FAMILIES,
    FAMILY_RUNTIME_CONFIGS,
    family_chat_template,
    family_runtime,
    family_vision,
    model_type_runtime,
    model_type_vision,
    pretrained_model_type,
    sm80_attention_backend,
    trainer_model_class,
)
from agilerl.architectures.runtime import (
    ContextParallelPlan,
    GdnMixerConfig,
    GdnPatchConfig,
    MambaPatchConfig,
    ModelRuntimeConfig,
    ModelTypes,
    PatchRuntimeConfig,
    TensorParallelPlan,
    TrainerRuntimeConfig,
    VisionSupport,
    VllmRuntimeConfig,
    import_attribute,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from peft import PeftModel
    from transformers import PreTrainedModel

__all__ = [
    "FAMILIES",
    "FAMILY_RUNTIME_CONFIGS",
    "ContextParallelPlan",
    "GdnMixerConfig",
    "GdnPatchConfig",
    "MambaPatchConfig",
    "ModelRuntimeConfig",
    "ModelTypes",
    "PatchRuntimeConfig",
    "TensorParallelPlan",
    "TrainerRuntimeConfig",
    "VisionSupport",
    "VllmRuntimeConfig",
    "family_chat_template",
    "family_context_parallel_plans",
    "family_runtime",
    "family_tensor_parallel_plans",
    "family_vision",
    "install_family_patches",
    "model_type_runtime",
    "model_type_vision",
    "pretrained_model_type",
    "sm80_attention_backend",
    "trainer_model_class",
]


def family_tensor_parallel_plans() -> list[TensorParallelPlan]:
    """Resolve each distinct family tensor-parallel plan.

    Plans match their own module classes, so every plan runs on any model,
    including multimodal wrappers whose top-level ``model_type`` differs from
    the language tower's.

    :return: Plans in family order.
    :rtype: list[TensorParallelPlan]
    """
    paths = dict.fromkeys(
        family.tensor_parallel_plan
        for family in FAMILIES
        if family.tensor_parallel_plan is not None
    )
    return [cast("TensorParallelPlan", import_attribute(path)) for path in paths]


def family_context_parallel_plans() -> list[ContextParallelPlan]:
    """Resolve each distinct family context-parallel plan.

    Plans match their own module classes, so every plan runs on any model,
    including multimodal wrappers whose top-level ``model_type`` differs from
    the language tower's.

    :return: Plans in family order.
    """
    paths = dict.fromkeys(
        family.context_parallel_plan
        for family in FAMILIES
        if family.context_parallel_plan is not None
    )
    return [cast("ContextParallelPlan", import_attribute(path)) for path in paths]


def install_family_patches(
    model_type: str | None,
    model: PreTrainedModel | PeftModel | None = None,
    use_sequence_packing: bool = False,
) -> None:
    """Register the family's Liger applier, then install its text and vision patches.

    :param model_type: Hugging Face ``model_type``; None or empty for a custom
        model with no type, which installs nothing.
    :type model_type: str | None
    :param model: Already-built model the patches also apply to, or None.
    :type model: PreTrainedModel | PeftModel | None
    :param use_sequence_packing: Install packed-sequence mixer wraps.
    :type use_sequence_packing: bool, optional
    :raises ValueError: If no family serves ``model_type``.
    """
    if not model_type:
        return
    runtime = model_type_runtime(model_type)
    if HAS_LIGER_KERNEL and runtime.liger is not None:
        cast("Callable[[], object]", import_attribute(runtime.liger))()
    if runtime.patch.install is not None:
        runtime.patch.install(runtime.patch, model, use_sequence_packing)
    vision = model_type_vision(model_type)
    if vision is not None and vision.install is not None:
        vision.install(runtime.patch, model, use_sequence_packing)


for family in FAMILIES:
    if family.register_loading is not None:
        family.register_loading()
