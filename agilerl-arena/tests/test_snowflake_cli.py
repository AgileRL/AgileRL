# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""arena snowflake command line."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from agilerl.arena.cli import main
from agilerl.arena.snowflake.cli import ml_job
from agilerl.arena.snowflake.job import SubmitSettings
from agilerl.arena.snowflake.source import TableSource

RUN_ID = "arena-20261109163457-abc123"


def _invoke(args: list[str], manifest: Path | None = None):
    if manifest is not None:
        args = [manifest.as_posix() if part == "MANIFEST" else part for part in args]
    return CliRunner().invoke(main, args)


class TestSnowflakeHelp:
    def test_arena_help_lists_snowflake(self) -> None:
        result = _invoke(["--help"])

        assert result.exit_code == 0
        assert "snowflake" in result.output

    def test_snowflake_help(self) -> None:
        result = _invoke(["snowflake", "--help"])

        assert result.exit_code == 0
        assert "submit" in result.output
        assert "status" in result.output
        assert "logs" in result.output


class TestSubmitCommand:
    def _options(self, extra: list[str] | None = None) -> list[str]:
        args = [
            "snowflake",
            "submit",
            "MANIFEST",
            "--compute-pool",
            "arena_gpu_pool",
            "--runtime-environment",
            "cre@arena_runtime",
            "--payload-stage",
            "arena.training.payload",
            "--runs-stage",
            "arena.training.runs",
            "--metrics-table",
            "arena.training.arena_metrics",
            "--pulsar-url",
            "pulsar://broker.example:6650",
            "--pulsar-topic-prefix",
            "non-persistent://customer/arena",
        ]
        if extra:
            args.extend(extra)
        return args

    def test_prints_run_id_and_job_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("algorithm: {name: SFT}\n", encoding="utf-8")
        seen: dict = {}

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen["session"] = session
            seen["path"] = path
            seen["settings"] = settings
            return (RUN_ID, "mljob-7")

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: "session"
        )

        result = _invoke(
            self._options(
                ["--source-table", "arena.training.prompts", "--eai", "arena_hf_eai"]
            ),
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert f"run_id: {RUN_ID}" in result.output
        assert "job_id: mljob-7" in result.output
        settings = seen["settings"]
        assert settings.source == TableSource("arena.training.prompts")
        assert settings.nodes == 1
        assert settings.external_access_integrations == ("arena_hf_eai",)
        assert settings.runtime_environment == "cre@arena_runtime"
        assert seen["session"] == "session"

    def test_source_table_and_query_together_are_rejected(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")

        result = _invoke(
            self._options(
                [
                    "--source-table",
                    "T",
                    "--source-query",
                    "select 1",
                ]
            ),
            manifest,
        )

        assert result.exit_code != 0
        assert "exactly one" in result.output

    def test_missing_pulsar_url_is_rejected(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        args = [
            part
            for part in self._options(["--source-table", "T"])
            if part not in {"--pulsar-url", "pulsar://broker.example:6650"}
        ]

        result = _invoke(args, manifest)

        assert result.exit_code != 0
        assert "pulsar-url" in result.output

    def test_submit_error_is_shown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            msg = "env_image starts a Kubernetes Pod"
            raise ValueError(msg)

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: "session"
        )

        result = _invoke(self._options(["--source-query", "select 1"]), manifest)

        assert result.exit_code != 0
        assert "Kubernetes Pod" in result.output


class TestStatusAndLogs:
    def test_status_and_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Handle:
            status = "RUNNING"

            def get_logs(self) -> str:
                return "trainer started"

        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.ml_job",
            lambda session, job_id: Handle(),
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: "session"
        )

        status = _invoke(["snowflake", "status", "mljob-7"])
        logs = _invoke(["snowflake", "logs", "mljob-7"])

        assert status.exit_code == 0
        assert status.output.strip() == "RUNNING"
        assert logs.exit_code == 0
        assert "trainer started" in logs.output

    def test_ml_job_uses_the_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict = {}

        class MLJob:
            def __init__(self, job_id: str, session: object = None) -> None:
                seen["job_id"] = job_id
                seen["session"] = session

        module = types.ModuleType("snowflake.ml.jobs")
        module.MLJob = MLJob
        for name in ("snowflake", "snowflake.ml"):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "snowflake.ml.jobs", module)
        session = object()

        handle = ml_job(session, "mljob-7")

        assert isinstance(handle, MLJob)
        assert seen == {"job_id": "mljob-7", "session": session}

    def test_status_and_logs_report_a_missing_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("snowflake", "snowflake.ml"):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "snowflake.ml.jobs", None)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: object()
        )

        status = _invoke(["snowflake", "status", "mljob-7"])
        logs = _invoke(["snowflake", "logs", "mljob-7"])

        assert status.exit_code != 0
        assert "agilerl-arena[snowflake]" in status.output
        assert logs.exit_code != 0
        assert "agilerl-arena[snowflake]" in logs.output

    def test_missing_sdk_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("snowflake", "snowflake.ml"):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "snowflake.ml.jobs", None)

        with pytest.raises(ImportError, match="agilerl-arena\\[snowflake\\]"):
            ml_job(object(), "mljob-7")
