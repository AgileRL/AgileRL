"""Tests for arena-related models — ArenaEnvSpec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.models.env import ArenaEnvSpec


# ---------------------------------------------------------------------------
# ArenaEnvSpec
# ---------------------------------------------------------------------------
class TestArenaEnvSpec:
    def test_default_version(self):
        spec = ArenaEnvSpec(name="CartPole-v1")
        assert spec.version == "latest"

    def test_custom_version(self):
        spec = ArenaEnvSpec(name="CartPole-v1", version="v2")
        assert spec.version == "v2"

    def test_inherits_env_spec_fields(self):
        spec = ArenaEnvSpec(name="LunarLander-v3", num_envs=8)
        assert spec.name == "LunarLander-v3"
        assert spec.num_envs == 8

    def test_default_num_envs(self):
        spec = ArenaEnvSpec(name="CartPole-v1")
        assert spec.num_envs == 16

    def test_num_envs_validation(self):
        with pytest.raises(ValidationError):
            ArenaEnvSpec(name="CartPole-v1", num_envs=0)

    def test_pydantic_serialization(self):
        spec = ArenaEnvSpec(name="CartPole-v1", version="v3", num_envs=4)
        data = spec.model_dump()
        assert data["name"] == "CartPole-v1"
        assert data["version"] == "v3"
        assert data["num_envs"] == 4

    def test_pydantic_json_round_trip(self):
        spec = ArenaEnvSpec(name="MountainCar-v0", version="latest", num_envs=32)
        json_str = spec.model_dump_json()
        restored = ArenaEnvSpec.model_validate_json(json_str)
        assert restored.name == spec.name
        assert restored.version == spec.version
        assert restored.num_envs == spec.num_envs
