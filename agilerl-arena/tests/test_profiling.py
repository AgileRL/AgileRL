# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.arena.models.profiling``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.arena.models.algorithms import CISPOSpec
from agilerl.arena.models.profiling import ProfilingConfig


class TestProfilingConfig:
    def test_defaults_profile_nothing(self) -> None:
        config = ProfilingConfig()

        assert config.memory_snapshot_on_oom is False
        assert config.torch_profile_step is None
        assert config.torch_profile_with_stack is False
        assert config.profile_ranks is None
        assert config.output_dir is None

    @pytest.mark.parametrize(
        "enabled",
        [{"memory_snapshot_on_oom": True}, {"torch_profile_step": 2}],
        ids=["snapshot", "trace"],
    )
    def test_requires_output_dir_when_enabled(self, enabled: dict) -> None:
        with pytest.raises(ValueError, match="output_dir is required"):
            ProfilingConfig(**enabled)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            (
                "memory_history_max_entries",
                0,
                "memory_history_max_entries must be >= 1",
            ),
            ("torch_profile_step", 0, "torch_profile_step must be >= 1"),
            ("profile_ranks", [0, -1], "profile_ranks must be non-negative"),
        ],
    )
    def test_rejects_out_of_range_values(
        self, field: str, value: object, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ProfilingConfig(output_dir="/tmp/profiles", **{field: value})


class TestGRPOSpecProfilingConfig:
    def test_manifest_dict_parses_to_profiling_config(self) -> None:
        spec = CISPOSpec.model_validate(
            {
                "profiling_config": {
                    "output_dir": "/tmp/profiles",
                    "memory_snapshot_on_oom": True,
                    "torch_profile_step": 2,
                },
            },
        )

        assert spec.profiling_config == ProfilingConfig(
            output_dir="/tmp/profiles",
            memory_snapshot_on_oom=True,
            torch_profile_step=2,
        )

    def test_unset_profiling_config_is_none(self) -> None:
        assert CISPOSpec().profiling_config is None

    def test_rejects_unknown_profiling_keys(self) -> None:
        with pytest.raises(ValidationError, match="snapshot_everything"):
            CISPOSpec.model_validate(
                {"profiling_config": {"snapshot_everything": True}},
            )
