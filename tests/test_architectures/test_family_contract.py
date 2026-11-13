# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Every discovered family's declaration agrees with the catalog and Arena."""

from __future__ import annotations

import pytest

from agilerl.architectures.catalog import FAMILIES, FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.runtime import ModelRuntimeConfig, import_attribute
from agilerl.arena.memory.config_fields import VISION_RESIZE_BY_MODEL_TYPE
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

FAMILY_PARAMS = [pytest.param(family, id=family.name) for family in FAMILIES]


class TestFamilies:
    def test_names_are_unique(self) -> None:
        names = [family.name for family in FAMILIES]

        assert len(names) == len(set(names))

    @pytest.mark.parametrize("family", FAMILY_PARAMS)
    def test_claims_at_least_one_model_type(self, family: ModelRuntimeConfig) -> None:
        assert family.model_types.text + family.model_types.multimodal

    @pytest.mark.parametrize("family", FAMILY_PARAMS)
    def test_every_claimed_model_type_resolves_to_the_family(
        self, family: ModelRuntimeConfig
    ) -> None:
        claimed = family.model_types.text + family.model_types.multimodal

        assert all(FAMILY_RUNTIME_CONFIGS[t] is family for t in claimed)

    @pytest.mark.parametrize("family", FAMILY_PARAMS)
    def test_vision_is_set_only_for_multimodal_families(
        self, family: ModelRuntimeConfig
    ) -> None:
        assert (family.vision is not None) == bool(family.model_types.multimodal)

    @pytest.mark.parametrize("family", FAMILY_PARAMS)
    def test_parallel_plan_paths_import(self, family: ModelRuntimeConfig) -> None:
        paths = [
            path
            for path in (family.tensor_parallel_plan, family.context_parallel_plan)
            if path is not None
        ]

        assert all(import_attribute(path) is not None for path in paths)


class TestArenaAgreement:
    def test_every_supported_checkpoint_has_a_family(self) -> None:
        model_types = {
            entry.config["model_type"] for entry in SUPPORTED_MODEL_INFO.values()
        }

        assert model_types - set(FAMILY_RUNTIME_CONFIGS) == set()

    def test_every_vision_resize_type_is_a_multimodal_family_type(self) -> None:
        multimodal = {
            model_type
            for family in FAMILIES
            for model_type in family.model_types.multimodal
        }

        assert set(VISION_RESIZE_BY_MODEL_TYPE) - multimodal == set()
