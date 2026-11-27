# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GPU pool sizing messages, and the Snowpark fakes the Snowflake tests share."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from agilerl.snowflake.catalog import GpuFamily
from agilerl.snowflake.setup import (
    check_gpu_cap,
    family_listed,
    missing_family_message,
    no_gpu_message,
    nodes_for_gpus,
    unsupported_gpu_message,
)

ACCOUNT = "ORGNAME-ACCOUNT"
IMAGE = "/arena/training/images/snowflake-job:32.6.0-py313-cu130"
RERUN = "CALL <app>.app_code.set_pools(['<name>'])"
NO_GPU = (
    "This account has no GPU_NV_S pool in this region (1x A10G). "
    "Arena did not create a compute pool.\n"
    "GPU families Arena can size in this region:\n"
    "  (none)\n"
    "Rerun with one of those names:\n"
    f"  {RERUN}\n"
    "A CPU pool cannot run this trainer.\n"
    "Nothing was created."
)
PROFILE_DOC = {
    "image": IMAGE,
    "warehouse": "arena_wh",
    "stage": "arena.training.runs",
    "metrics_table": "arena.training.arena_metrics",
    "external_access_integrations": ["arena_hf_eai"],
    "pools": {
        "broker": {"pool": "arena_pulsar_pool", "family": "CPU_X64_L"},
        "head": {"pool": "arena_head_pool", "family": "CPU_X64_SL"},
        "gpu": {
            "GPU_NV_S": {
                "pool": "arena_gpu_GPU_NV_S",
                "max_gpus": 2,
                "gpus_per_node": 1,
                "gpu_memory_gib": 24,
            }
        },
    },
}


class Row:
    """Snowpark ``Row`` stand-in."""

    def __init__(self, **values: object) -> None:
        self._values = values

    def as_dict(self) -> dict[str, object]:
        return dict(self._values)


class Frame:
    def __init__(self, rows: list[Row]) -> None:
        self._rows = rows

    def collect(self) -> list[Row]:
        return list(self._rows)


class Files:
    def __init__(self) -> None:
        self.uploads: list[dict[str, object]] = []
        self.stages: list[str] = []

    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.uploads.append(json.loads(Path(local).read_text(encoding="utf-8")))
        self.stages.append(stage)
        assert auto_compress is False
        assert overwrite is True


class FakeSession:
    def __init__(self, routes: list[tuple[object, object]] | None = None) -> None:
        self.routes = list(routes or [])
        self.statements: list[str] = []
        self.file = Files()
        self.closed = 0

    def sql(self, statement: str) -> Frame:
        self.statements.append(statement)
        for pred, result in self.routes:
            if pred(statement):
                if isinstance(result, BaseException):
                    raise result
                return Frame(result)
        return Frame([])

    def close(self) -> None:
        self.closed += 1


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point connections.toml at ``tmp_path``. The default key follows."""
    connections = tmp_path / "connections.toml"
    monkeypatch.setattr(
        "agilerl.snowflake.session.connections_file", lambda: connections
    )
    return connections, tmp_path / "arena_job_user.p8"


def admin_routes(
    extra: list[tuple[object, object]] | None = None,
) -> list[tuple[object, object]]:
    """ACCOUNTADMIN identity and a region with CPU and two GPU families. ``extra`` wins."""
    routes: list[tuple[object, object]] = list(extra or [])
    routes.extend(
        [
            (
                lambda statement: statement.startswith("SELECT CURRENT_USER"),
                [
                    Row(
                        **{
                            "CURRENT_USER()": "NICK",
                            "CURRENT_ROLE()": "ACCOUNTADMIN",
                            "CURRENT_ACCOUNT()": "ORG",
                            "CURRENT_REGION()": "AWS",
                        }
                    )
                ],
            ),
            (
                lambda statement: statement == "SELECT CURRENT_ROLE()",
                [Row(**{"CURRENT_ROLE()": "ACCOUNTADMIN"})],
            ),
            (
                lambda statement: statement == "SHOW COMPUTE POOL INSTANCE FAMILIES",
                [
                    Row(name="CPU_X64_L", gpu_count=0, gpu=""),
                    Row(name="CPU_X64_SL", gpu_count=0, gpu=""),
                    Row(name="GPU_NV_S", gpu_count=1, gpu="A10G"),
                    Row(name="GPU_NV_M", gpu_count=4, gpu="A10G"),
                    Row(name="GPU_GCP_NV_L4_1_24G", gpu_count=1, gpu="L4"),
                ],
            ),
        ]
    )
    return routes


class TestNoGpuMessage:
    def test_matches_an_empty_family_list(self) -> None:
        assert no_gpu_message("GPU_NV_S", [], RERUN) == NO_GPU

    def test_a_family_list_replaces_none(self) -> None:
        text = no_gpu_message("GPU_NV_S", [GpuFamily("GPU_NV_M", 4, "A10G")], RERUN)

        assert "  GPU_NV_M  4x A10G" in text
        assert "(none)" not in text


class TestNodesForGpus:
    def test_rounds_up_to_whole_nodes(self) -> None:
        assert nodes_for_gpus(3, 4) == 1
        assert nodes_for_gpus(8, 4) == 2


class TestCheckGpuCap:
    def test_whole_nodes_pass(self) -> None:
        check_gpu_cap("GPU_NV_M", 8, 4)

    def test_a_partial_node_names_the_multiples(self) -> None:
        with pytest.raises(ValueError, match="multiple of 4: 4, 8, 12"):
            check_gpu_cap("GPU_NV_M", 6, 4)


class TestNoGpuMessageListedFamily:
    def test_a_family_with_another_gpu_count_says_so(self) -> None:
        text = no_gpu_message("GPU_NV_S", [GpuFamily("GPU_NV_S", 2, "A10G")], RERUN)

        assert text.startswith(
            "GPU_NV_S is 2x A10G per node in this region, not the 1x A10G Arena sizes."
        )


class TestFamilyListed:
    def test_matches_without_case(self) -> None:
        families = [GpuFamily("GPU_NV_S", 1, "A10G")]

        assert family_listed("gpu_nv_s", families)
        assert not family_listed("GPU_NV_M", families)


class TestRefusalMessages:
    def test_a_missing_cpu_family_names_the_pool(self) -> None:
        assert missing_family_message("CPU_X64_L", "broker") == (
            "This account has no CPU_X64_L instance family in this region.\n"
            "The broker pool was not created.\n"
            "Nothing was created."
        )

    def test_an_unsized_gpu_family_lists_the_known_ones(self) -> None:
        text = unsupported_gpu_message("GPU_XYZ")

        assert text.startswith(
            "GPU_XYZ is not a GPU family Arena sizes.\nKnown families: "
        )
        assert "GPU_NV_S" in text
        assert text.endswith("Nothing was created.")
