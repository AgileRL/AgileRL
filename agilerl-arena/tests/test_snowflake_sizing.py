# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Sizing a Snowflake run's manifest with Arena's memory estimate."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from agilerl.arena.client_version import CLIENT_VERSION_HEADER
from agilerl.arena.models import __version__
from agilerl.arena.snowflake.license import SnowflakeUpgradeRequiredError
from agilerl.arena.snowflake.sizing import manifest_fits
from tests.test_snowflake_job import _gpu, _sft, _write
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
            manifest, _gpu("GPU_NV_L", 8, 40, 16), 16, ARENA_URL, ARENA_AUTH
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
                _gpu("GPU_NV_S", 1, 24, 8),
                1,
                ARENA_URL,
                ARENA_AUTH,
            )
