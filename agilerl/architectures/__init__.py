# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Architecture-scoped family runtime catalog and patches.

Per-``model_type`` trainer, vLLM, and patch defaults live in
:mod:`agilerl.architectures.catalog`. :func:`family_runtime` reads
``config.json`` then looks up that type. :func:`install_family_patches` calls
:attr:`~agilerl.architectures.runtime.PatchRuntimeConfig.install` when it is
set. :func:`family_tensor_parallel_plans` resolves every family's
:class:`~agilerl.architectures.runtime.TensorParallelPlan`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from agilerl.architectures.catalog import (
    FAMILY_RUNTIME_CONFIGS,
    family_runtime,
    pretrained_model_type,
)
from agilerl.architectures.runtime import (
    MambaPatchConfig,
    ModelRuntimeConfig,
    PatchRuntimeConfig,
    TensorParallelPlan,
    TrainerRuntimeConfig,
    VllmRuntimeConfig,
)

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

__all__ = [
    "FAMILY_RUNTIME_CONFIGS",
    "MambaPatchConfig",
    "ModelRuntimeConfig",
    "PatchRuntimeConfig",
    "TensorParallelPlan",
    "TrainerRuntimeConfig",
    "VllmRuntimeConfig",
    "family_runtime",
    "family_tensor_parallel_plans",
    "install_family_patches",
    "pretrained_model_type",
]


def family_tensor_parallel_plans() -> list[TensorParallelPlan]:
    """Resolve each distinct catalog tensor-parallel plan.

    Plans match their own module classes, so every plan runs on any model,
    including multimodal wrappers whose top-level ``model_type`` differs from
    the language tower's.

    :return: Plans in catalog order.
    :rtype: list[TensorParallelPlan]
    """
    paths = dict.fromkeys(
        runtime.tensor_parallel_plan
        for runtime in FAMILY_RUNTIME_CONFIGS.values()
        if runtime.tensor_parallel_plan is not None
    )
    plans: list[TensorParallelPlan] = []
    for path in paths:
        module_path, _, attribute = path.partition(":")
        plans.append(getattr(importlib.import_module(module_path), attribute))
    return plans


def install_family_patches(
    model_type: str | None,
    model: PreTrainedModel | PeftModel | None = None,
) -> None:
    """Install the family's catalog patches when patch.install is set.

    :param model_type: Hugging Face ``model_type``, or None.
    :type model_type: str | None
    :param model: Already-built model the patches also apply to, or None.
    :type model: PreTrainedModel | PeftModel | None
    """
    runtime = FAMILY_RUNTIME_CONFIGS.get(model_type, ModelRuntimeConfig())
    install = runtime.patch.install
    if install is None:
        return
    install(runtime.patch, model=model)
