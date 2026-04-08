"""Tests for arena-related models — ArenaEnvSpec, JobStatus, ArenaVM, ArenaResource, ArenaCluster."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.models import (
    ArenaCluster,
    ArenaResource,
)
from agilerl.models import JobStatus
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


# ---------------------------------------------------------------------------
# JobStatus
# ---------------------------------------------------------------------------
class TestJobStatus:
    def test_all_values(self):
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_string_conversion(self):
        assert str(JobStatus.RUNNING) == "running"
        assert str(JobStatus.COMPLETED) == "completed"

    def test_from_string(self):
        assert JobStatus("pending") == JobStatus.PENDING
        assert JobStatus("failed") == JobStatus.FAILED

    def test_is_str_subclass(self):
        assert isinstance(JobStatus.RUNNING, str)


# ---------------------------------------------------------------------------
# ArenaVM
# ---------------------------------------------------------------------------
class TestArenaVM:
    def test_str_with_gpu(self):
        from agilerl.models import ArenaResource

        vm = ArenaResource.MEDIUM_ACCELERATED.value
        s = str(vm)
        assert "Medium Accelerated" in s
        assert "GPU" in s

    def test_str_without_gpu(self):
        from agilerl.models import ArenaResource

        vm = ArenaResource.XL.value
        s = str(vm)
        assert "XL" in s
        assert vm.gpus == 0

    def test_str_with_gram(self):
        from agilerl.models import ArenaResource

        vm = ArenaResource.A100_4GPU.value
        s = str(vm)
        assert "GRAM" in s

    def test_field_validation(self):
        from agilerl.models import ArenaResource

        vm = ArenaResource.MEDIUM_ACCELERATED.value
        assert vm.cpus >= 1
        assert vm.gpus >= 0
        assert vm.ram_gi >= 1


# ---------------------------------------------------------------------------
# ArenaResource
# ---------------------------------------------------------------------------
class TestArenaResource:
    def test_all_members_exist(self):
        expected = {
            "MEDIUM_ACCELERATED",
            "LARGE_ACCELERATED",
            "XL_ACCELERATED",
            "XL",
            "XXL",
            "A100_2GPU",
            "A100_4GPU",
            "LARGE_4XL4",
        }
        actual = {m.name for m in ArenaResource}
        assert expected == actual

    def test_values_are_arena_vms(self):
        from agilerl.models import ArenaResource
        from agilerl.models import ArenaCluster

        for member in ArenaResource:
            vm = member.value
            assert hasattr(vm, "cpus")
            assert hasattr(vm, "gpus")
            assert hasattr(vm, "ram_gi")

    def test_str_delegates_to_vm(self):
        s = str(ArenaResource.LARGE_ACCELERATED)
        assert "Large Accelerated" in s


# ---------------------------------------------------------------------------
# ArenaCluster
# ---------------------------------------------------------------------------
class TestArenaCluster:
    def test_basic_construction(self):
        cluster = ArenaCluster(resource=ArenaResource.MEDIUM_ACCELERATED, num_nodes=2)
        assert cluster.resource == ArenaResource.MEDIUM_ACCELERATED
        assert cluster.num_nodes == 2

    def test_num_nodes_minimum(self):
        with pytest.raises(ValidationError):
            ArenaCluster(resource=ArenaResource.XL, num_nodes=0)

    def test_serialization(self):
        cluster = ArenaCluster(resource=ArenaResource.A100_2GPU, num_nodes=1)
        data = cluster.model_dump(mode="json")
        assert "resource" in data
        assert data["num_nodes"] == 1

    def test_single_node(self):
        cluster = ArenaCluster(resource=ArenaResource.XXL, num_nodes=1)
        assert cluster.num_nodes == 1
