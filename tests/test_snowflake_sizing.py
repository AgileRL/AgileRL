# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Sizing a Snowflake run's manifest with Arena's memory estimate."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from agilerl.client_version import CLIENT_VERSION_HEADER
from agilerl.models import __version__
from agilerl.snowflake.gpu_fit import GpuShortfallError
from agilerl.snowflake.license import SnowflakeUpgradeRequiredError
from agilerl.snowflake.profile import GPU_FAMILIES
from agilerl.snowflake.sizing import manifest_fits
from tests.test_snowflake_job import _gpu, _grpo, _opd_teacher, _sft, _write
from tests.test_snowflake_license import Arena

ARENA_URL = "https://arena.test"
ARENA_AUTH = {"Authorization": "Bearer arena_key_1"}
ESTIMATE_URL = f"{ARENA_URL}/api/cli/v1/memory/estimate"


def _arena_answers(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response | Exception
) -> Arena:
    """Answer the next request with ``response``, and record it."""
    arena = Arena([response])
    monkeypatch.setattr(httpx, "Client", arena)
    return arena


def _estimate(exit_code: int, report: dict) -> httpx.Response:
    return httpx.Response(200, json={"exit_code": exit_code, "report": report})


class TestManifestFits:
    def test_a_fitting_manifest_fits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        arena = _arena_answers(monkeypatch, _estimate(0, {"fits": True}))
        manifest = _write(tmp_path, _sft())

        # Act
        fits = manifest_fits(
            manifest, "GPU_NV_L", _gpu("GPU_NV_L", 8, 40, 16), 16, ARENA_URL, ARENA_AUTH
        )

        # Assert
        assert fits is True
        [call] = arena.calls
        assert call["url"] == ESTIMATE_URL
        assert call["headers"] == {
            "Authorization": "Bearer arena_key_1",
            CLIENT_VERSION_HEADER: __version__,
        }
        assert call["json"] == {
            "manifest": _sft(),
            "gpu": GPU_FAMILIES["GPU_NV_L"].gpu,
            "device_gb": 40.0,
            "num_gpus": 16,
            "gpus_per_node": 8,
        }

    def test_an_over_budget_manifest_does_not_fit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arena_answers(monkeypatch, _estimate(3, {"fits": False}))

        fits = manifest_fits(
            _write(tmp_path, _sft()),
            "GPU_NV_S",
            _gpu("GPU_NV_S", 1, 24, 8),
            1,
            ARENA_URL,
            ARENA_AUTH,
        )

        assert fits is False

    def test_a_usage_error_raises_the_estimator_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arena_answers(monkeypatch, _estimate(2, {"error": "manifest names no model."}))

        with pytest.raises(ValueError, match=r"^manifest names no model\.$"):
            manifest_fits(
                _write(tmp_path, _sft()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 8),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )

    def test_a_usage_error_without_a_message_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arena_answers(monkeypatch, _estimate(2, {}))

        with pytest.raises(KeyError, match="error"):
            manifest_fits(
                _write(tmp_path, _sft()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 8),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )

    def test_rejected_credentials_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arena_answers(monkeypatch, httpx.Response(401, json={"error": "unauthorized"}))

        with pytest.raises(ValueError, match=r"rejected the credentials .*HTTP 401"):
            manifest_fits(
                _write(tmp_path, _sft()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 8),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )

    def test_a_non_json_error_raises_its_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arena_answers(
            monkeypatch, httpx.Response(502, text="<html>Bad gateway</html>")
        )

        with pytest.raises(ValueError, match=r"could not estimate .*Bad gateway"):
            manifest_fits(
                _write(tmp_path, _sft()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 8),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )

    def test_a_non_json_success_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arena_answers(monkeypatch, httpx.Response(200, text="<html>login</html>"))

        with pytest.raises(ValueError, match="not JSON"):
            manifest_fits(
                _write(tmp_path, _sft()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 8),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )

    def test_an_unreachable_arena_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arena_answers(monkeypatch, httpx.ConnectError("connection refused"))

        with pytest.raises(
            ValueError, match=r"Could not reach Arena at https://arena\.test"
        ):
            manifest_fits(
                _write(tmp_path, _sft()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 8),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )

    def test_an_old_client_must_upgrade(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = {"error": "client_upgrade_required", "description": "Upgrade agilerl."}
        _arena_answers(monkeypatch, httpx.Response(426, json=body))

        with pytest.raises(SnowflakeUpgradeRequiredError, match="Upgrade agilerl"):
            manifest_fits(
                _write(tmp_path, _sft()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 8),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )

    def test_an_rl_run_on_too_few_gpus_is_refused_with_the_count_to_ask_for(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        _arena_answers(monkeypatch, _estimate(2, {"error": "1 GPUs, the job needs 2"}))

        # Act
        with pytest.raises(GpuShortfallError) as raised:
            manifest_fits(
                _write(tmp_path, _grpo()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 8),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )

        # Assert
        assert str(raised.value) == (
            "GRPO needs at least 2 GPUs on GPU_NV_S: training and rollout "
            "generation run on separate GPUs.\n"
            "Ask for 2 GPUs or more.\n"
            "Nothing was started."
        )

    def test_a_shortfall_past_the_pool_says_to_raise_its_maximum(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arena_answers(monkeypatch, _estimate(2, {"error": "1 GPUs, the job needs 2"}))

        with pytest.raises(GpuShortfallError, match="holds 1 GPUs, so raise"):
            manifest_fits(
                _write(tmp_path, _grpo()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 1),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )

    def test_a_teacher_run_on_one_gpu_nodes_is_refused_when_it_fits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        _arena_answers(monkeypatch, _estimate(0, {"fits": True}))

        # Act
        with pytest.raises(GpuShortfallError) as raised:
            manifest_fits(
                _write(tmp_path, _opd_teacher()),
                "GPU_NV_S",
                _gpu("GPU_NV_S", 1, 24, 8),
                2,
                ARENA_URL,
                ARENA_AUTH,
            )

        # Assert
        assert str(raised.value) == (
            "OPD with a separate teacher model runs each rollout engine and its "
            "teacher on one node, 2 GPUs, but GPU_NV_S has 1 GPU(s) per node.\n"
            "Use a GPU family with at least 2 GPUs per node.\n"
            "Nothing was started."
        )

    def test_a_teacher_run_fits_a_node_that_holds_the_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arena_answers(monkeypatch, _estimate(0, {"fits": True}))

        fits = manifest_fits(
            _write(tmp_path, _opd_teacher()),
            "GPU_NV_M",
            _gpu("GPU_NV_M", 4, 24, 4),
            4,
            ARENA_URL,
            ARENA_AUTH,
        )

        assert fits is True
