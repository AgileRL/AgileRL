# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena memory estimate`` and ``solve`` against a mocked Arena API."""

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

from agilerl.cli import main
from agilerl.client_version import CLIENT_VERSION_HEADER
from agilerl.models import __version__

GiB = 1024**3
ARENA_URL = "https://arena.test"
MODEL = "Qwen/Qwen3.5-0.8B"
MANIFEST = {
    "algorithm": {"name": "GRPO"},
    "network": {"pretrained_model_name_or_path": MODEL},
}


def _phase(name: str, used_gib: int, usable_gib: int) -> dict[str, Any]:
    budget = usable_gib * GiB * (0.911 if name == "training" else 1)
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
        "fits": used_gib * GiB <= budget,
        "budget_bytes": int(budget),
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


def _memory(arena: Arena, *args: str, request_timeout: int | None = None) -> Any:
    timeout = (
        [] if request_timeout is None else ["--request-timeout", str(request_timeout)]
    )
    with patch("agilerl.client.httpx.Client", arena.client_factory()):
        return CliRunner().invoke(
            main,
            [
                "--api-key",
                "arena_key_1",
                "--base-url",
                ARENA_URL,
                *timeout,
                "memory",
                *args,
            ],
        )


def _estimate(arena: Arena, *args: str) -> Any:
    return _memory(arena, "estimate", *args)


def _solve(arena: Arena, *args: str) -> Any:
    return _memory(arena, "solve", "max_model_len", *args)


def _solve_report(limited_by: str) -> dict[str, Any]:
    return {
        "field": "max_model_len",
        "value": 32768,
        "limited_by": limited_by,
        "bound": 32768 if limited_by == "bound" else 262144,
        "model": MODEL,
        "unchecked_over_budget": [],
        "engine": {
            "tensor_parallel_size": 2,
            "max_num_seqs": 8,
            "gpu_memory_utilization": 0.85,
        },
        "generation": _phase("generation", 4, 22),
        "training": _phase("training", 10, 22),
    }


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

    def test_each_phase_shows_the_verdict_arena_sent(self, manifest_path: Path) -> None:
        # Arrange
        report = _report(10)
        report["generation"]["fits"] = False
        report["fits"] = False
        arena = Arena(httpx.Response(200, json={"exit_code": 3, "report": report}))

        # Act
        result = _estimate(
            arena, str(manifest_path), "--gpu", "NVIDIA L4", "--no-color"
        )

        # Assert
        assert result.exit_code == 3, result.output
        assert result.stdout.count("FITS") == 1
        assert result.stdout.count("OVER BUDGET") == 1
        assert "GiB under the 8.9% buffer" in result.stdout

    def test_a_phase_without_a_verdict_fails(self, manifest_path: Path) -> None:
        report = _report(10)
        del report["training"]["fits"]
        arena = Arena(httpx.Response(200, json={"exit_code": 0, "report": report}))

        result = _estimate(arena, str(manifest_path), "--gpu", "NVIDIA L4")

        assert isinstance(result.exception, KeyError)

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

    @pytest.mark.parametrize(
        ("request_timeout", "expected"), [(None, 90), (30, 90), (300, 300)]
    )
    def test_waits_at_least_90_s_or_the_request_timeout(
        self, manifest_path: Path, request_timeout: int | None, expected: int
    ) -> None:
        arena = Arena(httpx.Response(200, json={"exit_code": 0, "report": _report(10)}))

        result = _memory(
            arena,
            "estimate",
            str(manifest_path),
            "--device-gb",
            "24",
            "--json",
            request_timeout=request_timeout,
        )

        assert result.exit_code == 0, result.output
        assert arena.requests[0].extensions["timeout"]["read"] == expected

    def test_a_manifest_without_a_model_prints_the_report_untitled(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump({"algorithm": {"name": "GRPO"}}))
        arena = Arena(httpx.Response(200, json={"exit_code": 0, "report": _report(10)}))

        # Act
        result = _estimate(arena, str(path), "--device-gb", "24", "--no-color")

        # Assert
        assert result.exit_code == 0, result.output
        assert result.stdout.startswith("Memory estimate\n")

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

        result = _estimate(arena, str(manifest_path))

        assert result.exit_code == 2
        assert result.stdout == ""
        assert result.stderr == f"{message}\n"

    def test_json_prints_a_usage_error_on_stderr_and_exits_2(
        self, manifest_path: Path
    ) -> None:
        report = {"error": "Pass --gpu with a GPU type, or --device-gb."}
        arena = Arena(httpx.Response(200, json={"exit_code": 2, "report": report}))

        result = _estimate(arena, str(manifest_path), "--json")

        assert result.exit_code == 2
        assert result.stdout == ""
        assert json.loads(result.stderr) == {**report, "type": "UsageError"}

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

    def test_config_exits_2_without_calling_arena(self, manifest_path: Path) -> None:
        arena = Arena(httpx.Response(500))

        result = _estimate(
            arena, str(manifest_path), "--gpu", "NVIDIA L4", "--config", "config.json"
        )

        assert result.exit_code == 2
        assert "--config is not supported" in result.stderr
        assert arena.requests == []

    @pytest.mark.parametrize("device_gb", ["nan", "inf", "0", "-8"])
    def test_a_device_gb_that_is_not_positive_and_finite_exits_2_without_calling_arena(
        self, manifest_path: Path, device_gb: str
    ) -> None:
        arena = Arena(httpx.Response(500))

        result = _estimate(arena, str(manifest_path), f"--device-gb={device_gb}")

        assert result.exit_code == 2
        assert result.stderr == (
            "--device-gb: capacity must be a positive number of GiB, "
            f"got {float(device_gb)}.\n"
        )
        assert arena.requests == []

    def test_a_json_usage_error_is_a_json_document_on_stderr(
        self, manifest_path: Path
    ) -> None:
        arena = Arena(httpx.Response(500))

        result = _estimate(arena, str(manifest_path), "--device-gb=-8", "--json")

        assert result.exit_code == 2
        assert result.stdout == ""
        assert json.loads(result.stderr) == {
            "error": "--device-gb: capacity must be a positive number of GiB, got -8.0.",
            "type": "UsageError",
        }
        assert arena.requests == []

    def test_a_rejected_api_key_under_local_json_is_json_on_stderr(
        self, manifest_path: Path
    ) -> None:
        arena = Arena(httpx.Response(401, json={"error": "unauthorized"}))

        result = _estimate(arena, str(manifest_path), "--gpu", "NVIDIA L4", "--json")

        assert result.exit_code == 1
        assert result.stdout == ""
        error = json.loads(result.stderr)
        assert "Invalid API key" in error["error"]
        assert error["type"] == "ArenaAuthError"

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


class TestMemorySolve:
    def test_a_solved_value_prints_the_report_and_exits_0(
        self, manifest_path: Path
    ) -> None:
        # Arrange
        arena = Arena(
            httpx.Response(200, json={"exit_code": 0, "report": _solve_report("bound")})
        )

        # Act
        result = _solve(
            arena,
            str(manifest_path),
            "--gpu",
            "NVIDIA L4",
            "--num-gpus",
            "4",
            "--hi",
            "32768",
            "--no-color",
        )

        # Assert
        assert result.exit_code == 0, result.output
        assert result.stdout.startswith(
            "Solved max_model_len = 32768  (checkpoint / --hi cap)\n"
            f"{MODEL} · TP=2 · 8 seqs · gmu=0.85\n"
        )
        assert result.stdout.count("FITS") == 2
        request = arena.requests[0]
        assert str(request.url) == f"{ARENA_URL}/api/cli/v1/memory/solve"
        assert request.headers["Authorization"] == "Bearer arena_key_1"
        assert request.headers[CLIENT_VERSION_HEADER] == __version__
        assert json.loads(request.content) == {
            "manifest": MANIFEST,
            "field": "max_model_len",
            "gpu": "NVIDIA L4",
            "num_gpus": 4,
            "hi": 32768,
        }

    def test_a_memory_limit_names_gpu_memory(self, manifest_path: Path) -> None:
        arena = Arena(
            httpx.Response(
                200, json={"exit_code": 0, "report": _solve_report("memory")}
            )
        )

        result = _solve(arena, str(manifest_path), "--device-gb", "24", "--no-color")

        assert result.exit_code == 0, result.output
        assert "Solved max_model_len = 32768  (GPU memory)" in result.stdout
        assert json.loads(arena.requests[0].content)["device_gb"] == 24.0

    def test_json_prints_the_report_unchanged(self, manifest_path: Path) -> None:
        report = _solve_report("memory")
        arena = Arena(httpx.Response(200, json={"exit_code": 0, "report": report}))

        result = _solve(arena, str(manifest_path), "--gpu", "NVIDIA L4", "--json")

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == report

    @pytest.mark.parametrize(
        ("exit_code", "message"),
        [
            (3, "no max_model_len value up to 262144 fits on this device"),
            (2, "Pass --gpu with a GPU type from `arena resources list`."),
        ],
    )
    def test_an_error_prints_the_message_and_exits_with_its_code(
        self, manifest_path: Path, exit_code: int, message: str
    ) -> None:
        arena = Arena(
            httpx.Response(
                200, json={"exit_code": exit_code, "report": {"error": message}}
            )
        )

        result = _solve(arena, str(manifest_path))

        assert result.exit_code == exit_code
        assert result.stdout == ""
        assert result.stderr == f"{message}\n"

    def test_json_prints_an_over_budget_report_and_exits_3(
        self, manifest_path: Path
    ) -> None:
        report = {"error": "no max_model_len value up to 262144 fits on this device"}
        arena = Arena(httpx.Response(200, json={"exit_code": 3, "report": report}))

        result = _solve(arena, str(manifest_path), "--json")

        assert result.exit_code == 3
        assert json.loads(result.stdout) == report
        assert result.stderr == ""

    def test_json_usage_error_is_json_on_stderr(self, manifest_path: Path) -> None:
        report = {"error": "Pass --gpu with a GPU type, or --device-gb."}
        arena = Arena(httpx.Response(200, json={"exit_code": 2, "report": report}))

        result = _solve(arena, str(manifest_path), "--json")

        assert result.exit_code == 2
        assert result.stdout == ""
        assert json.loads(result.stderr) == {**report, "type": "UsageError"}

    def test_config_exits_2_without_calling_arena(self, manifest_path: Path) -> None:
        arena = Arena(httpx.Response(500))

        result = _solve(
            arena, str(manifest_path), "--gpu", "NVIDIA L4", "--config", "config.json"
        )

        assert result.exit_code == 2
        assert "--config is not supported" in result.stderr
        assert arena.requests == []

    def test_a_solve_report_without_gmu_omits_it(self, manifest_path: Path) -> None:
        report = _solve_report("memory")
        del report["engine"]["gpu_memory_utilization"]
        arena = Arena(httpx.Response(200, json={"exit_code": 0, "report": report}))

        result = _solve(arena, str(manifest_path), "--gpu", "NVIDIA L4", "--no-color")

        assert result.exit_code == 0, result.output
        assert f"{MODEL} · TP=2 · 8 seqs\n" in result.stdout

    def test_a_negative_hi_exits_2_without_calling_arena(
        self, manifest_path: Path
    ) -> None:
        arena = Arena(httpx.Response(500))

        result = _solve(arena, str(manifest_path), "--gpu", "NVIDIA L4", "--hi", "-1")

        assert result.exit_code == 2
        assert "--hi" in result.stderr
        assert arena.requests == []

    def test_only_max_model_len_is_solvable(self, manifest_path: Path) -> None:
        arena = Arena(httpx.Response(500))

        result = _memory(arena, "solve", "max_num_seqs", str(manifest_path))

        assert result.exit_code == 2
        assert "'max_num_seqs' is not 'max_model_len'" in result.stderr
        assert arena.requests == []

    def test_an_unreadable_manifest_exits_2_without_calling_arena(
        self, tmp_path: Path
    ) -> None:
        arena = Arena(httpx.Response(500))

        result = _solve(arena, str(tmp_path / "missing.yaml"), "--gpu", "NVIDIA L4")

        assert result.exit_code == 2
        assert "could not validate" in result.stderr
        assert arena.requests == []

    def test_an_unexpected_reply_exits_1(self, manifest_path: Path) -> None:
        arena = Arena(httpx.Response(200, text="<html>login</html>"))

        result = _solve(arena, str(manifest_path), "--gpu", "NVIDIA L4")

        assert result.exit_code == 1
