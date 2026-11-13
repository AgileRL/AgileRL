# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Every catalog ``model_type`` resolves to the settings in ``catalog_snapshot.json``."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from agilerl.architectures.catalog import (
    FAMILY_RUNTIME_CONFIGS,
    model_type_runtime,
    model_type_trainer_class,
    model_type_vision,
)

SNAPSHOT = json.loads((Path(__file__).parent / "catalog_snapshot.json").read_text())


def qualified_name(value: object) -> str | None:
    if value is None:
        return None
    return f"{value.__module__}.{value.__qualname__}"


def plan_hook(path: str | None, hook: str) -> str | None:
    if path is None:
        return None
    module_path, _, attribute = path.partition(":")
    plan = getattr(importlib.import_module(module_path), attribute)
    return qualified_name(getattr(plan, hook))


def describe(model_type: str) -> dict[str, Any]:
    """Settings one model_type runs with, as JSON values."""
    runtime = model_type_runtime(model_type)
    vision = model_type_vision(model_type)
    hf_overrides = runtime.vllm.hf_overrides
    vllm = runtime.vllm.model_dump(exclude={"hf_overrides"})
    vllm["hf_overrides"] = (
        hf_overrides
        if isinstance(hf_overrides, dict | None)
        else hf_overrides.__qualname__
    )
    patch = runtime.patch
    return {
        "vllm": vllm,
        "trainer": {
            "model_class": model_type_trainer_class(model_type).__name__,
            "config_class": runtime.trainer.config_class.__name__,
            "trust_remote_code": runtime.trainer.trust_remote_code,
        },
        "patch": {
            "has_install": patch.install is not None
            or (vision is not None and vision.install is not None),
            "mamba": None if patch.mamba is None else patch.mamba.model_dump(),
            "gdn": None if patch.gdn is None else patch.gdn.model_dump(mode="json"),
        },
        "tensor_parallel_shard": plan_hook(runtime.tensor_parallel_plan, "shard"),
        "tensor_parallel_restore": plan_hook(runtime.tensor_parallel_plan, "restore"),
        "context_parallel_wrap": plan_hook(runtime.context_parallel_plan, "wrap"),
        "lora_exclude_modules": sorted(runtime.lora_exclude_modules),
        "chat_format": runtime.chat_format,
        "reasoning_parser": runtime.reasoning_parser,
        "image_token": None if vision is None else vision.image_token,
        "processor_class": None
        if vision is None or vision.processor_class is None
        else vision.processor_class.__name__,
        "tower_connector_lora_targets": []
        if vision is None
        else sorted(vision.tower_connector_lora_targets),
        "tower_modalities": {}
        if vision is None
        else {
            module: sorted(modalities)
            for module, modalities in sorted(vision.tower_modalities.items())
        },
    }


class TestCatalogDerivation:
    def test_catalog_serves_the_snapshot_model_types(self) -> None:
        assert sorted(FAMILY_RUNTIME_CONFIGS) == sorted(SNAPSHOT)

    @pytest.mark.parametrize("model_type", sorted(SNAPSHOT))
    def test_model_type_settings_match_the_snapshot(self, model_type: str) -> None:
        assert describe(model_type) == SNAPSHOT[model_type]
