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
from agilerl.arena.snowflake.cli import service_logs, service_status
from agilerl.arena.snowflake.job import SubmitSettings
from agilerl.arena.snowflake.source import QuerySource, TableSource

SERVICE = "ARENA_20261109163457_ABC123"

RUN_ID = "arena-20261109163457-abc123"


class Row:
    """Snowpark ``Row`` stand-in: values only through ``as_dict``."""

    def __init__(self, **values: object) -> None:
        self._values = values

    def as_dict(self) -> dict[str, object]:
        return dict(self._values)


class Closable:
    """Session stand-in that records ``close``."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


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

    def test_submit_help_documents_nodes_and_reward_file(self) -> None:
        result = _invoke(["snowflake", "submit", "--help"])

        assert result.exit_code == 0
        assert "Job instances" in result.output
        assert "Training steps between checkpoints" in result.output
        assert "--reward-file" in result.output


class TestSubmitCommand:
    def _options(self, extra: list[str] | None = None) -> list[str]:
        args = [
            "snowflake",
            "submit",
            "MANIFEST",
            "--compute-pool",
            "arena_gpu_pool",
            "--image",
            "/arena/training/images/snowflake-job:latest",
            "--warehouse",
            "arena_wh",
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

    def test_prints_run_id_and_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("algorithm: {name: SFT}\n", encoding="utf-8")
        seen: dict = {}

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen["session"] = session
            seen["path"] = path
            seen["settings"] = settings
            return (RUN_ID, SERVICE)

        session = Closable()
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: session
        )

        result = _invoke(
            self._options(
                ["--source-table", "arena.training.prompts", "--eai", "arena_hf_eai"]
            ),
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert f"run_id: {RUN_ID}" in result.output
        assert f"service: {SERVICE}" in result.output
        settings = seen["settings"]
        assert settings.source == TableSource("arena.training.prompts")
        assert settings.nodes == 1
        assert settings.external_access_integrations == ("arena_hf_eai",)
        assert settings.image == "/arena/training/images/snowflake-job:latest"
        assert settings.warehouse == "arena_wh"
        assert settings.reward_file is None
        assert seen["session"] is session
        assert session.closed == 1

    def test_reward_file_is_passed_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        reward = tmp_path / "reward.py"
        reward.write_text("", encoding="utf-8")
        seen: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, SERVICE)

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(
            self._options(
                [
                    "--source-table",
                    "T",
                    "--reward-file",
                    reward.as_posix(),
                    "--nodes",
                    "3",
                ]
            ),
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert seen[0].reward_file == reward
        assert seen[0].nodes == 3

    def test_missing_reward_file_is_rejected(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")

        result = _invoke(
            self._options(
                [
                    "--source-table",
                    "T",
                    "--reward-file",
                    (tmp_path / "no.py").as_posix(),
                ]
            ),
            manifest,
        )

        assert result.exit_code != 0
        assert "reward-file" in result.output

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
        assert "at most one" in result.output

    def test_no_source_is_passed_as_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        seen: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, SERVICE)

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(self._options(), manifest)

        assert result.exit_code == 0, result.output
        assert seen[0].source is None

    def test_query_source_is_passed_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        seen: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, SERVICE)

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(self._options(["--source-query", "select 1"]), manifest)

        assert result.exit_code == 0, result.output
        assert seen[0].source == QuerySource("select 1")

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
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(self._options(["--source-query", "select 1"]), manifest)

        assert result.exit_code != 0
        assert "Kubernetes Pod" in result.output


class TestStatusAndLogs:
    def test_status_and_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Handle(Closable):
            def __init__(self) -> None:
                super().__init__()
                self.statements: list[str] = []

            def sql(self, text: str) -> Handle:
                self.statements.append(text)
                return self

            def collect(self) -> list[Row]:
                text = self.statements[-1]
                if text.startswith("SHOW"):
                    return [
                        Row(
                            service_status="DONE",
                            instance_id=index,
                            container_name="main",
                            status="DONE",
                        )
                        for index in (0, 1)
                    ]
                return [Row(**{"SYSTEM$GET_SERVICE_LOGS": "trainer started"})]

        handle = Handle()
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: handle
        )

        status = _invoke(["snowflake", "status", SERVICE])
        logs = _invoke(["snowflake", "logs", SERVICE, "--instance", "1"])

        assert status.exit_code == 0, status.output
        assert status.output.strip() == "0\tmain\tDONE\n1\tmain\tDONE\nstate: done"
        assert "SHOW SERVICE CONTAINERS IN SERVICE" in handle.statements[0]
        assert logs.exit_code == 0, logs.output
        assert "trainer started" in logs.output
        assert "SYSTEM$GET_SERVICE_LOGS" in handle.statements[1]
        assert ", 1, 'main'" in handle.statements[1]
        assert handle.closed == 2

    def test_status_and_logs_use_the_session(self) -> None:
        class Handle:
            def __init__(self) -> None:
                self.text = ""

            def sql(self, text: str) -> Handle:
                self.text = text
                return self

            def collect(self) -> list[Row]:
                if self.text.startswith("SHOW"):
                    return [
                        Row(
                            SERVICE_STATUS="RUNNING",
                            INSTANCE_ID="0",
                            CONTAINER_NAME="main",
                            STATUS="READY",
                        )
                    ]
                return [Row(LOG="line")]

        handle = Handle()

        assert service_status(handle, SERVICE) == "0\tmain\tREADY\nstate: running"
        assert service_logs(handle, SERVICE, 0) == "line"

    def test_status_and_logs_report_a_missing_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
        monkeypatch.setitem(sys.modules, "snowflake.snowpark", None)

        status = _invoke(["snowflake", "status", SERVICE])
        logs = _invoke(["snowflake", "logs", SERVICE])

        assert status.exit_code != 0
        assert "agilerl-arena[snowflake]" in status.output
        assert logs.exit_code != 0
        assert "agilerl-arena[snowflake]" in logs.output

    def test_a_bad_service_name_is_shown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        status = _invoke(["snowflake", "status", "not a service"])

        assert status.exit_code != 0
        assert "Snowflake identifier" in status.output
