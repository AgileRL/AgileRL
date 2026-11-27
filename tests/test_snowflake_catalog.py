# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Read-only account queries."""

from __future__ import annotations

import json

import pytest

from agilerl.snowflake.catalog import (
    GpuFamily,
    PoolState,
    existing_pools,
    families_from,
    family_sized,
    load_families,
    missing_object,
    remote_profile,
    schema_rows,
    whole_number,
)
from tests.test_snowflake_setup import (
    IMAGE,
    PROFILE_DOC,
    FakeSession,
    Row,
)


def raising(prefix: str, error: Exception) -> FakeSession:
    return FakeSession([(lambda s: s.startswith(prefix), error)])


class TestMissingObject:
    @pytest.mark.parametrize(
        "text",
        [
            "Database 'ARENA' does not exist or not authorized.",
            "Stage not found",
            "No such file",
            (
                "Unknown user-defined function "
                "ARENA.TRAINING.ARENA_GPU_BUDGET!GET_LINKED_RESOURCES."
            ),
        ],
    )
    def test_missing_object_errors(self, text: str) -> None:
        assert missing_object(RuntimeError(text)) is True

    def test_other_errors(self) -> None:
        assert missing_object(RuntimeError("Insufficient privileges")) is False


class TestFamiliesFrom:
    def test_reads_name_gpu_count_and_gpu(self) -> None:
        rows = [Row(name="GPU_NV_M", gpu_count="4", gpu="A10G")]

        assert families_from(rows) == [GpuFamily("GPU_NV_M", 4, "A10G")]

    def test_a_known_family_without_a_gpu_column_uses_arenas_table(self) -> None:
        rows = [Row(instance_family="GPU_NV_S", gpu_count=1)]

        assert families_from(rows) == [GpuFamily("GPU_NV_S", 1, "A10G")]

    def test_an_unknown_family_without_a_gpu_column_has_no_gpu(self) -> None:
        rows = [Row(name="CPU_X64_L", gpu_count=0)]

        assert families_from(rows) == [GpuFamily("CPU_X64_L", 0, "")]


# GCP London rows: Snowflake reuses the AWS R6K names for nodes with half the GPUs.
GCP_REGION = families_from(
    [
        Row(name="GPU_GCP_NV_L4_1_24G", gpu_count=1),
        Row(name="GPU_GCP_NV_L4_4_24G", gpu_count=4),
        Row(name="GPU_R6K_G1_48", gpu_count=1),
        Row(name="GPU_R6K_G1_384", gpu_count=8),
    ]
)


class TestFamilySized:
    @pytest.mark.parametrize("family", ["GPU_GCP_NV_L4_1_24G", "gpu_gcp_nv_l4_4_24g"])
    def test_accepts_an_l4_family_with_the_gpu_count_arena_sizes(
        self, family: str
    ) -> None:
        assert family_sized(family, GCP_REGION) is True

    def test_accepts_an_r6k_family_with_the_aws_gpu_count(self) -> None:
        region = families_from([Row(name="GPU_R6K_G1_48", gpu_count=2)])

        assert family_sized("GPU_R6K_G1_48", region) is True

    def test_refuses_an_r6k_family_with_the_gcp_gpu_count(self) -> None:
        assert family_sized("GPU_R6K_G1_48", GCP_REGION) is False

    def test_refuses_a_family_arena_does_not_size(self) -> None:
        assert family_sized("GPU_R6K_G1_384", GCP_REGION) is False

    def test_refuses_a_row_without_a_gpu_count(self) -> None:
        region = families_from([Row(name="GPU_NV_S")])

        assert family_sized("GPU_NV_S", region) is False

    def test_refuses_a_family_missing_from_the_region(self) -> None:
        assert family_sized("GPU_NV_S", GCP_REGION) is False


class TestWholeNumber:
    @pytest.mark.parametrize(
        ("value", "expected"), [("4.00", 4), (7, 7), (None, 0), ("", 0)]
    )
    def test_reads_a_show_column(self, value: object, expected: int) -> None:
        assert whole_number(value) == expected


class TestRemoteProfile:
    def test_reads_setup_json(self) -> None:
        session = FakeSession(
            [(lambda s: s.startswith("SELECT $1"), [Row(DOC=json.dumps(PROFILE_DOC))])]
        )

        profile = remote_profile(session)

        assert profile is not None
        assert profile.image == IMAGE
        assert profile.pools.gpu["GPU_NV_S"].max_gpus == 2

    def test_a_missing_file(self) -> None:
        session = raising("SELECT $1", RuntimeError("File not found"))

        assert remote_profile(session) is None


class TestLoadFamilies:
    def test_reads_the_region_families(self) -> None:
        session = FakeSession(
            [
                (
                    lambda s: s == "SHOW COMPUTE POOL INSTANCE FAMILIES",
                    [Row(name="GPU_NV_S", gpu_count=1, gpu="A10G")],
                )
            ]
        )

        assert load_families(session) == [GpuFamily("GPU_NV_S", 1, "A10G")]


class TestSchemaRows:
    def test_a_missing_schema_has_no_rows(self) -> None:
        session = raising("SHOW SECRETS", RuntimeError("Schema 'X' does not exist"))

        assert schema_rows(session, "SHOW SECRETS IN SCHEMA x.y") == []

    def test_other_errors_propagate(self) -> None:
        session = raising("SHOW SECRETS", RuntimeError("insufficient privileges"))

        with pytest.raises(RuntimeError, match="insufficient privileges"):
            schema_rows(session, "SHOW SECRETS IN SCHEMA x.y")


class TestExistingPools:
    def test_keys_pools_by_upper_case_name(self) -> None:
        # Arrange
        session = FakeSession(
            [
                (
                    lambda s: s == "SHOW COMPUTE POOLS",
                    [
                        Row(
                            name="app_gpu",
                            instance_family="GPU_NV_S",
                            max_nodes="2",
                            active_nodes=1,
                        ),
                        Row(name="APP_HEAD", max_nodes=4, active_nodes=""),
                    ],
                )
            ]
        )

        # Act
        pools = existing_pools(session)

        # Assert
        assert pools == {
            "APP_GPU": PoolState("app_gpu", "GPU_NV_S", 2, 1),
            "APP_HEAD": PoolState("APP_HEAD", "", 4, None),
        }
