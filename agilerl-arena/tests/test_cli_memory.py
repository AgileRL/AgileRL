# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena memory estimate`` against a mocked Arena API."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import yaml
from click.testing import CliRunner

from agilerl.arena.cli import main
from agilerl.arena.client_version import CLIENT_VERSION_HEADER
from agilerl.arena.models import __version__

GiB = 1024**3
ARENA_URL = "https://arena.test"
MODEL = "Qwen/Qwen3.5-0.8B"
MANIFEST = {
    "algorithm": {"name": "GRPO"},
    "network": {"pretrained_model_name_or_path": MODEL},
}


def _phase(name: str, used_gib: int, usable_gib: int) -> dict[str, Any]:
    return {
        "phase": name,
        "components": [
            {
                "key": "base_weights",
                "label": "Base weights (frozen)",
                "bytes": used_gib * GiB,
                "detail": {},
                "note": None,
            }
        ],
        "device_total_bytes": 24 * GiB,
        "device_usable_bytes": usable_gib * GiB,
        "warnings": [],
        "host": [],
        "host_usable_bytes": None,
    }


def _report(training_gib: int) -> dict[str, Any]:
    fits = training_gib <= 20
    return {
        "training": _phase("training", training_gib, 22),
        "generation": _phase("generation", 4, 22),
        "fits": fits,
        "engine": {"tensor_parallel_size": 1, "max_num_seqs": 8},
        "advice": []
        if fits
        else [
            {
                "phase": "training",
                "action": "Halve the LoRA rank (16 -> 8)",
                "saves_bytes": 2 * GiB,
            }
        ],
    }


class Arena:
    """Records what the CLI sent and answers with one response."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response

    def client_factory(self) -> Callable[..., httpx.Client]:
        real_client = httpx.Client

        def build(**kwargs: Any) -> httpx.Client:
            return real_client(transport=httpx.MockTransport(self.handle), **kwargs)

        return build


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(MANIFEST), encoding="utf-8")
    return path


def _estimate(arena: Arena, *args: str) -> Any:
    with patch("agilerl.arena.client.httpx.Client", arena.client_factory()):
        return CliRunner().invoke(
            main,
            [
                "--api-key",
                "arena_key_1",
                "--base-url",
                ARENA_URL,
                "memory",
                "estimate",
                *args,
            ],
        )


class TestMemoryEstimate:
    def test_a_fitting_manifest_prints_the_report_and_exits_0(
        self, manifest_path: Path
    ) -> None:
        # Arrange
        arena = Arena(httpx.Response(200, json={"exit_code": 0, "report": _report(10)}))

        # Act
        result = _estimate(
            arena,
            str(manifest_path),
            "--gpu",
            "NVIDIA L4",
            "--num-gpus",
            "4",
            "--no-color",
        )

        # Assert
        assert result.exit_code == 0, result.output
        assert result.stdout.startswith(f"Memory estimate · {MODEL}\n")
        assert (
            "Rollout engine: tensor_parallel_size=1 · max_num_seqs=8" in result.stdout
        )
        assert result.stdout.count("FITS") == 2
        assert "Fixes by memory saved" not in result.stdout
        request = arena.requests[0]
        assert str(request.url) == f"{ARENA_URL}/api/cli/v1/memory/estimate"
        assert request.headers["Authorization"] == "Bearer arena_key_1"
        assert request.headers[CLIENT_VERSION_HEADER] == __version__
        assert json.loads(request.content) == {
            "manifest": MANIFEST,
            "gpu": "NVIDIA L4",
            "num_gpus": 4,
        }

    def test_the_report_names_a_model_given_under_algorithm(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        path = tmp_path / "manifest.yaml"
        manifest = {
            "algorithm": {"name": "GRPO", "pretrained_model_name_or_path": MODEL}
        }
        path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        arena = Arena(httpx.Response(200, json={"exit_code": 0, "report": _report(10)}))

        # Act
        result = _estimate(arena, str(path), "--device-gb", "24", "--no-color")

        # Assert
        assert result.exit_code == 0, result.output
        assert result.stdout.startswith(f"Memory estimate · {MODEL}\n")

    def test_json_prints_the_report_unchanged(self, manifest_path: Path) -> None:
        report = _report(10)
        arena = Arena(httpx.Response(200, json={"exit_code": 0, "report": report}))

        result = _estimate(arena, str(manifest_path), "--device-gb", "24", "--json")

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == report
        assert json.loads(arena.requests[0].content)["device_gb"] == 24.0

    def test_an_over_budget_manifest_prints_the_fixes_and_exits_3(
        self, manifest_path: Path
    ) -> None:
        arena = Arena(httpx.Response(200, json={"exit_code": 3, "report": _report(21)}))

        result = _estimate(
            arena, str(manifest_path), "--gpu", "NVIDIA L4", "--no-color"
        )

        assert result.exit_code == 3, result.output
        assert "OVER BUDGET" in result.stdout
        assert "SHORTFALL" in result.stdout
        assert "Halve the LoRA rank (16 -> 8) (saves ~2.0 GiB)" in result.stdout

    def test_a_usage_error_prints_the_message_and_exits_2(
        self, manifest_path: Path
    ) -> None:
        message = (
            "Pass --gpu with a GPU type from `arena resources list`, or --device-gb."
        )
        arena = Arena(
            httpx.Response(200, json={"exit_code": 2, "report": {"error": message}})
        )

        result = _estimate(arena, str(manifest_path), "--json")

        assert result.exit_code == 2
        assert result.stdout == ""
        assert result.stderr == f"{message}\n"

    def test_an_unreadable_manifest_exits_2_without_calling_arena(
        self, tmp_path: Path
    ) -> None:
        arena = Arena(httpx.Response(500))

        result = _estimate(arena, str(tmp_path / "missing.yaml"), "--gpu", "NVIDIA L4")

        assert result.exit_code == 2
        assert "could not validate" in result.stderr
        assert arena.requests == []

    @pytest.mark.parametrize("text", ["", "- item\n", "just-a-string\n"])
    def test_a_manifest_that_is_not_a_mapping_exits_2_without_calling_arena(
        self, tmp_path: Path, text: str
    ) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text(text)
        arena = Arena(httpx.Response(500))

        result = _estimate(arena, str(path), "--gpu", "NVIDIA L4")

        assert result.exit_code == 2
        assert "could not validate" in result.stderr
        assert "mapping" in result.stderr
        assert arena.requests == []

    def test_a_rejected_api_key_exits_1(self, manifest_path: Path) -> None:
        arena = Arena(httpx.Response(401, json={"error": "unauthorized"}))

        result = _estimate(arena, str(manifest_path), "--gpu", "NVIDIA L4")

        assert result.exit_code == 1
        assert "Invalid API key" in result.output

    def test_a_non_json_error_exits_1_with_its_text(self, manifest_path: Path) -> None:
        arena = Arena(httpx.Response(502, text="<html>Bad gateway</html>"))

        result = _estimate(arena, str(manifest_path), "--gpu", "NVIDIA L4")

        assert result.exit_code == 1
        assert "Bad gateway" in result.output

    def test_an_unexpected_reply_exits_1(self, manifest_path: Path) -> None:
        arena = Arena(httpx.Response(200, text="<html>login</html>"))

        result = _estimate(arena, str(manifest_path), "--gpu", "NVIDIA L4")

        assert result.exit_code == 1
        assert "unexpected memory estimate" in result.output
