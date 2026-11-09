# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Catalog models the training-memory search sizes."""

import pytest
from pydantic import ValidationError

from agilerl.arena.memory.resolve.batch import catalog_model_spec
from agilerl.arena.memory.specs import ModelArch, ModelSpec
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


def fresh_catalog_model(hub_id: str) -> ModelSpec:
    info = SUPPORTED_MODEL_INFO[hub_id]
    return ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(info.config),
        n_params=info.inspected["num_params"],
    )


class TestCatalogModelSpec:
    @pytest.mark.parametrize(
        "hub_id",
        ["Qwen/Qwen3.5-4B", "openai/gpt-oss-20b", "zai-org/GLM-5.3-Flash"],
    )
    def test_matches_a_model_built_from_the_bundled_config(self, hub_id: str):
        # Act
        first = catalog_model_spec(hub_id)
        again = catalog_model_spec(hub_id)

        # Assert
        assert first == fresh_catalog_model(hub_id)
        assert again == fresh_catalog_model(hub_id)

    def test_shared_model_refuses_writes(self):
        # Arrange
        model = catalog_model_spec("Qwen/Qwen3.5-4B")

        # Act / Assert
        with pytest.raises(ValidationError, match="frozen"):
            model.arch.n_layers = 1
        assert catalog_model_spec("Qwen/Qwen3.5-4B") == fresh_catalog_model(
            "Qwen/Qwen3.5-4B"
        )

    def test_unknown_id_raises(self):
        with pytest.raises(KeyError, match="org/unknown"):
            catalog_model_spec("org/unknown")
