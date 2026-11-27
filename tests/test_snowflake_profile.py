# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The setup.json profile and the GPU families Arena sizes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agilerl.snowflake.profile import (
    GPU_FAMILIES,
    NO_POOLS,
    profile_from_mapping,
    put_setup_json,
    sole_gpu_pool,
)
from tests.test_snowflake_setup import PROFILE_DOC, FakeSession


class TestProfileFromMapping:
    def test_reads_pools_and_integrations(self) -> None:
        profile = profile_from_mapping(PROFILE_DOC)

        assert profile.pools.broker.pool == "arena_pulsar_pool"
        assert profile.pools.gpu["GPU_NV_S"].gpus_per_node == 1
        assert profile.external_access_integrations == ("arena_hf_eai",)

    @pytest.mark.parametrize(
        "pools",
        [
            None,
            {"broker": {"pool": "b", "family": "CPU_X64_L"}},
            {**PROFILE_DOC["pools"], "gpu": {"GPU_NV_S": "arena_gpu_GPU_NV_S"}},
        ],
    )
    def test_without_pools_says_to_rerun_setup(self, pools: object) -> None:
        section = {**PROFILE_DOC, "pools": pools}

        with pytest.raises(ValueError, match="no compute pools") as raised:
            profile_from_mapping(section)

        assert str(raised.value) == NO_POOLS


class TestGpuFamilies:
    @pytest.mark.parametrize(
        ("family", "gpus", "gpu"),
        [
            ("GPU_NV_S", 1, "A10G"),
            ("GPU_NV_SM", 1, "A10"),
            ("GPU_NV_3M", 2, "A100 80GB"),
            ("GPU_NV_SL", 4, "A100 80GB"),
            ("GPU_GCP_NV_L4_1_24G", 1, "L4"),
            ("GPU_GCP_NV_A100_8_40G", 8, "A100"),
        ],
    )
    def test_covers_aws_azure_and_gcp(self, family: str, gpus: int, gpu: str) -> None:
        spec = GPU_FAMILIES[family]

        assert spec.gpus == gpus
        assert spec.gpu == gpu


def failing_chmod(monkeypatch: pytest.MonkeyPatch, target: Path, code: int) -> None:
    """Make ``os.chmod`` on ``target`` fail with ``code``, as a full or read-only disk does."""
    real_chmod = os.chmod

    def chmod(path: str | os.PathLike[str], mode: int) -> None:
        if Path(path) == target:
            raise OSError(code, os.strerror(code), str(path))
        real_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", chmod)


class TestPutSetupJson:
    def test_uploads_the_profile_document(self) -> None:
        session = FakeSession()

        put_setup_json(
            session, profile_from_mapping(PROFILE_DOC), "@app_state.runs/_arena"
        )

        assert session.file.uploads == [PROFILE_DOC]
        assert session.file.stages == ["@app_state.runs/_arena"]


class TestSoleGpuPool:
    def test_one_gpu_family_names_its_pool(self) -> None:
        assert sole_gpu_pool(profile_from_mapping(PROFILE_DOC)) == "arena_gpu_GPU_NV_S"

    def test_several_gpu_families_have_none(self) -> None:
        doc = json.loads(json.dumps(PROFILE_DOC))
        doc["pools"]["gpu"]["GPU_NV_M"] = {
            "pool": "arena_gpu_GPU_NV_M",
            "max_gpus": 4,
            "gpus_per_node": 4,
            "gpu_memory_gib": 24,
        }

        assert sole_gpu_pool(profile_from_mapping(doc)) is None
