# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.models.networks`` validation helpers and specs."""

from __future__ import annotations

import pytest

from agilerl.models.hpo import RLHyperparameter
from agilerl.models.networks import (
    CosineLRScheduleConfig,
    LoraConfigDict,
    min_max_validator,
)


class TestMinMaxValidator:
    def test_raises_when_min_exceeds_max(self) -> None:
        model = RLHyperparameter(min=0.1, max=1.0)
        model.min = 999
        with pytest.raises(ValueError, match="must be less than or equal to"):
            min_max_validator("min", "max")(model)


class TestLoraConfigDict:
    def test_drops_target_modules(self) -> None:
        # Act
        config = LoraConfigDict.model_validate({"target_modules": ["q_proj"]})

        # Assert
        assert "target_modules" not in config.model_dump(mode="json")


class TestCosineLRScheduleConfig:
    def test_manifest_sets_every_field(self) -> None:
        config = CosineLRScheduleConfig.model_validate(
            {"num_steps": 200, "warmup_proportion": 0.05, "min_lr_ratio": 0.2}
        )

        assert config.num_steps == 200
        assert config.warmup_proportion == 0.05
        assert config.min_lr_ratio == 0.2

    def test_defaults_to_no_warmup_and_a_tenth_floor(self) -> None:
        config = CosineLRScheduleConfig(num_steps=200)

        assert config.warmup_proportion == 0.0
        assert config.min_lr_ratio == 0.1

    @pytest.mark.parametrize(
        ("fields", "match"),
        [
            ({}, "num_steps"),
            ({"num_steps": 0}, "num_steps"),
            ({"num_steps": 10, "warmup_proportion": 1.0}, "warmup_proportion"),
            ({"num_steps": 10, "min_lr_ratio": 1.5}, "min_lr_ratio"),
            ({"num_epochs": 10}, "num_epochs"),
        ],
    )
    def test_rejects_invalid_fields(self, fields, match) -> None:
        with pytest.raises(ValueError, match=match):
            CosineLRScheduleConfig.model_validate(fields)
