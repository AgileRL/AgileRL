# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""arena snowflake command line."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import httpx
import pytest
import yaml
from click.testing import CliRunner, Result
from typing_extensions import Self

from agilerl.arena.cli import main
from agilerl.arena.client_version import CLIENT_VERSION_HEADER
from agilerl.arena.models import __version__
from agilerl.arena.snowflake.catalog import NO_SETUP
from agilerl.arena.snowflake.cli import run_status, service_logs
from agilerl.arena.snowflake.job import RunServices, SubmitSettings
from agilerl.arena.snowflake.profile import (
    AccountPools,
    AccountProfile,
    CpuPool,
    GpuPool,
    admin_connection_fields,
    job_connection_fields,
    write_connection,
)
from agilerl.arena.snowflake.source import QuerySource, TableSource

RUN_ID = "arena-20261109163457-abc123"
STEM = "ARENA_RUN_20261109163457_ABC123"
BROKER = f"{STEM}_BROKER"
HEAD = f"{STEM}_HEAD"
WORKERS = f"{STEM}_WORKERS"
TABLE = "db.sch.t"
PREVIOUS_RUN = "arena-20261108120000-def456"
SETUP_READ = "SELECT $1 AS DOC FROM @arena.training.runs/_arena/setup.json"


class Row:
    """Snowpark ``Row`` stand-in: values only through ``as_dict``."""

    def __init__(self, **values: object) -> None:
        self._values = values

    def as_dict(self) -> dict[str, object]:
        return dict(self._values)


class Closable:
    """Session stand-in that serves ``setup_doc`` as setup.json and records ``close``."""

    setup_doc: dict[str, object] | None = None

    def __init__(self) -> None:
        self.closed = 0
        self.statements: list[str] = []

    def sql(self, text: str) -> _SqlResult:
        self.statements.append(text)
        if text.startswith(SETUP_READ):
            return _SqlResult(self.setup_rows())
        return _SqlResult([])

    def setup_rows(self) -> list[Row]:
        if self.setup_doc is None:
            return []
        return [Row(DOC=json.dumps(self.setup_doc))]

    def close(self) -> None:
        self.closed += 1


def _services() -> RunServices:
    return RunServices(broker=BROKER, head=HEAD, workers=WORKERS)


def _invoke(args: list[str], manifest: Path | None = None):
    if manifest is not None:
        args = [manifest.as_posix() if part == "MANIFEST" else part for part in args]
    return CliRunner().invoke(main, args)


@pytest.fixture(autouse=True)
def _accept_every_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """Submit tests that do not care about fit never call the estimator."""
    monkeypatch.setattr(
        "agilerl.arena.snowflake.cli.manifest_fits",
        lambda manifest, pool, gpus: True,
    )


@pytest.fixture(autouse=True)
def _allow_snowflake_submit(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Existing submit tests cover Snowflake SQL, not the Arena flag."""
    if getattr(request.cls, "__name__", "") == "TestSubmitLicense":
        return
    monkeypatch.setattr(
        "agilerl.arena.snowflake.cli._require_snowflake_training",
        lambda: None,
    )


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
        assert "cancel" in result.output
        assert "setup" in result.output
        assert "check" in result.output
        assert "invite" in result.output
        assert "join" in result.output

    def test_snowflake_help_lists_the_publish_and_account_commands(self) -> None:
        result = _invoke(["snowflake", "--help"])
        commands = result.output.split("Commands:", 1)[1]
        names = {line.split()[0] for line in commands.splitlines() if line.strip()}

        assert result.exit_code == 0
        assert {"publish", "endpoint", "unpublish", "grant", "teardown"} <= names

    def test_submit_help_documents_gpus_and_reward_file(self) -> None:
        result = _invoke(["snowflake", "submit", "--help"])

        assert result.exit_code == 0
        assert "--gpus" in result.output
        assert "--gpu-family" in result.output
        assert "--nodes" not in result.output
        assert "--pulsar-url" not in result.output
        assert "Training steps between checkpoints" in result.output
        help_text = " ".join(result.output.split())
        assert "--reward-file" in help_text
        assert "names no rubric" in help_text
        assert "A named rubric still needs this file" in help_text


class TestSubmitCommand:
    def _options(self, extra: list[str] | None = None) -> list[str]:
        args = [
            "snowflake",
            "submit",
            "MANIFEST",
            "--image",
            "/arena/training/images/snowflake-job:latest",
            "--warehouse",
            "arena_wh",
            "--runs-stage",
            "arena.training.runs",
            "--metrics-table",
            "arena.training.arena_metrics",
            "--gpus",
            "1",
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
            return (RUN_ID, _services())

        session = Closable()
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: session
        )

        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")

        result = _invoke(
            self._options(
                ["--source-table", "arena.training.prompts", "--eai", "arena_hf_eai"]
            ),
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert result.output.splitlines()[-4:] == [
            f"run_id: {RUN_ID}",
            f"broker: {BROKER}",
            f"head: {HEAD}",
            f"workers: {WORKERS}",
        ]
        settings = seen["settings"]
        assert settings.source == TableSource("arena.training.prompts")
        assert settings.gpus == 1
        assert settings.gpu_family == "GPU_NV_S"
        assert settings.gpu_pool == "arena_gpu_GPU_NV_S"
        assert settings.broker_pool == "arena_pulsar_pool"
        assert settings.head_pool == "arena_head_pool"
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
            return (RUN_ID, _services())

        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(
            self._options(
                [
                    "--source-table",
                    TABLE,
                    "--reward-file",
                    reward.as_posix(),
                    "--gpus",
                    "3",
                ]
            ),
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert seen[0].reward_file == reward
        assert seen[0].gpus == 3

    def test_grpo_over_a_table_needs_no_dataset_or_rubric(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(
            "algorithm: {name: GRPO}\n"
            "environment:\n"
            "  env_type: rollout\n"
            "  prompt_template: {user_0: '{QUESTION}'}\n",
            encoding="utf-8",
        )
        reward = tmp_path / "my_reward.py"
        reward.write_text("", encoding="utf-8")
        uploaded: list[dict] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            uploaded.append(yaml.safe_load(path.read_text(encoding="utf-8")))
            return (RUN_ID, _services())

        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )
        args = ["--source-table", TABLE, "--reward-file", reward.as_posix()]

        # Act
        result = _invoke(self._options(args), manifest)

        # Assert
        assert result.exit_code == 0, result.output
        assert uploaded[0]["environment"] == {
            "env_type": "rollout",
            "prompt_template": {"user_0": "{QUESTION}"},
            "dataset": TABLE,
            "rubric_file_path": "reward.py",
        }
        assert "dataset" not in manifest.read_text(encoding="utf-8")

    def test_missing_reward_file_is_rejected(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")

        result = _invoke(
            self._options(
                [
                    "--source-table",
                    TABLE,
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
                    TABLE,
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
            return (RUN_ID, _services())

        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
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
            return (RUN_ID, _services())

        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(self._options(["--source-query", "select 1"]), manifest)

        assert result.exit_code == 0, result.output
        assert seen[0].source == QuerySource("select 1")

    def test_missing_gpus_is_rejected(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        args = [part for part in self._options() if part not in {"--gpus", "1"}]

        result = _invoke(args, manifest)

        assert result.exit_code != 0
        assert "--gpus" in result.output

    def test_removed_flags_are_rejected(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")

        nodes = _invoke(self._options(["--nodes", "2"]), manifest)
        pulsar = _invoke(
            self._options(["--pulsar-url", "pulsar://broker.example:6650"]), manifest
        )

        assert nodes.exit_code != 0
        assert "no such option" in nodes.output.lower()
        assert pulsar.exit_code != 0
        assert "no such option" in pulsar.output.lower()

    def test_submit_error_is_shown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            msg = "env_image starts a Kubernetes Pod"
            raise ValueError(msg)

        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(self._options(["--source-query", "select 1"]), manifest)

        assert result.exit_code != 0
        assert "Kubernetes Pod" in result.output

    def test_an_unqualified_source_table_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")

        def fail_open(name: str) -> Closable:
            raise AssertionError(name)

        monkeypatch.setattr("agilerl.arena.snowflake.cli.open_session", fail_open)

        result = _invoke(self._options(["--source-table", "prompts"]), manifest)

        assert result.exit_code != 0
        assert (
            "Error: --source-table must be database.schema.table, got 'prompts'"
            in result.output
        )

    def test_resume_is_passed_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        seen: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, _services())

        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(
            self._options(["--source-table", TABLE, "--resume", PREVIOUS_RUN]),
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert seen[0].resume_from == PREVIOUS_RUN
        assert seen[0].source == TableSource(TABLE)

    def test_a_sql_error_prints_one_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        errors = _fake_sql_errors(monkeypatch)

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            msg = "002003 (02000): SQL compilation error:\nStage 'RUNS' does not exist."
            raise errors.connector(msg)

        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(self._options(["--source-table", TABLE]), manifest)

        assert result.exit_code == 1
        assert result.output == (
            "Error: 002003 (02000): SQL compilation error: "
            "Stage 'RUNS' does not exist.\n"
        )


class SqlErrors:
    """Fake ``snowflake.connector`` and ``snowflake.snowpark`` error classes."""

    def __init__(self) -> None:
        self.connector = type("ProgrammingError", (Exception,), {})
        self.snowpark = type("SnowparkSQLException", (Exception,), {})


def _fake_sql_errors(monkeypatch: pytest.MonkeyPatch) -> SqlErrors:
    errors = SqlErrors()
    connector_errors = types.ModuleType("snowflake.connector.errors")
    connector_errors.DatabaseError = errors.connector
    snowpark_exceptions = types.ModuleType("snowflake.snowpark.exceptions")
    snowpark_exceptions.SnowparkSQLException = errors.snowpark
    for name, module in (
        ("snowflake", types.ModuleType("snowflake")),
        ("snowflake.connector", types.ModuleType("snowflake.connector")),
        ("snowflake.connector.errors", connector_errors),
        ("snowflake.snowpark", types.ModuleType("snowflake.snowpark")),
        ("snowflake.snowpark.exceptions", snowpark_exceptions),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return errors


class RunHandle(Closable):
    """Session stand-in that answers the status and log statements of one run."""

    def __init__(
        self,
        statuses: dict[str, str],
        containers: list[Row] | None = None,
        live: str = "",
        stored: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.statuses = statuses
        self.containers = [] if containers is None else containers
        self.live = live
        self.stored = [] if stored is None else stored
        self.error = error
        self.statements: list[str] = []

    def sql(self, text: str) -> RunHandle:
        self.statements.append(text)
        return self

    def collect(self) -> list[Row]:
        if self.error is not None:
            raise self.error
        text = self.statements[-1]
        if text.startswith("SHOW JOB SERVICES"):
            return [
                Row(name=name, status=status) for name, status in self.statuses.items()
            ]
        if text.startswith("SHOW SERVICE CONTAINERS"):
            return list(self.containers)
        if "SYSTEM$GET_SERVICE_LOGS" in text:
            return [Row(**{"SYSTEM$GET_SERVICE_LOGS": self.live})]
        if "SPCS_GET_LOGS" in text:
            return [Row(LOG=line) for line in reversed(self.stored)]
        return []


class TestRunStatus:
    def test_a_running_run_lists_every_role_and_container(self) -> None:
        # Arrange
        handle = RunHandle(
            {BROKER: "RUNNING", HEAD: "RUNNING", WORKERS: "PENDING"},
            containers=[Row(instance_id=0, container_name="main", status="READY")],
        )

        # Act
        text = run_status(handle, RUN_ID)

        # Assert
        assert text == (
            "broker\trunning\n  0\tmain\tREADY\n"
            "head\trunning\n  0\tmain\tREADY\n"
            "worker\trunning\n  0\tmain\tREADY\n"
            "state: running"
        )
        assert handle.statements == [
            f"SHOW JOB SERVICES LIKE '{STEM}%'",
            f"SHOW SERVICE CONTAINERS IN SERVICE {BROKER}",
            f"SHOW SERVICE CONTAINERS IN SERVICE {HEAD}",
            f"SHOW SERVICE CONTAINERS IN SERVICE {WORKERS}",
        ]

    def test_a_finished_run_without_its_broker_succeeded(self) -> None:
        handle = RunHandle({HEAD: "DONE", WORKERS: "DONE"})

        text = run_status(handle, RUN_ID)

        assert text == "broker\tgone\nhead\tdone\nworker\tdone\nstate: succeeded"
        assert f"SHOW SERVICE CONTAINERS IN SERVICE {BROKER}" not in handle.statements

    def test_a_failed_worker_fails_the_run(self) -> None:
        handle = RunHandle({BROKER: "RUNNING", HEAD: "RUNNING", WORKERS: "FAILED"})

        assert run_status(handle, RUN_ID).endswith("state: failed")

    def test_a_cancelled_run_is_cancelled(self) -> None:
        handle = RunHandle(
            {BROKER: "CANCELLED", HEAD: "CANCELLING", WORKERS: "CANCELLED"}
        )

        assert run_status(handle, RUN_ID).endswith("state: cancelling")


class TestServiceLogs:
    def test_a_running_service_reads_live_logs(self) -> None:
        handle = RunHandle({HEAD: "RUNNING"}, live="trainer started")

        text = service_logs(handle, RUN_ID, 0)

        assert text == "trainer started"
        assert handle.statements[-1] == (
            f"SELECT SYSTEM$GET_SERVICE_LOGS('{HEAD}', 0, 'main')"
        )

    def test_a_finished_service_reads_the_event_table(self) -> None:
        # Arrange
        handle = RunHandle(
            {HEAD: "DONE", WORKERS: "FAILED"}, stored=["step 1", "step 2", "OOM"]
        )

        # Act
        text = service_logs(handle, RUN_ID, 1, "worker")

        # Assert
        assert text == "step 1\nstep 2\nOOM"
        assert f"TABLE({WORKERS}!SPCS_GET_LOGS(" in handle.statements[-1]
        assert (
            "WHERE INSTANCE_ID = 1 AND CONTAINER_NAME = 'main'"
            in (handle.statements[-1])
        )
        assert not any("SYSTEM$GET_SERVICE_LOGS" in item for item in handle.statements)

    def test_a_dropped_service_is_reported(self) -> None:
        handle = RunHandle({HEAD: "DONE"})

        with pytest.raises(ValueError, match=f"{BROKER} was dropped"):
            service_logs(handle, RUN_ID, 0, "broker")


class TestSnowflakeSqlErrors:
    @pytest.mark.parametrize(
        "args",
        [
            ["snowflake", "status", RUN_ID],
            ["snowflake", "logs", RUN_ID],
            ["snowflake", "cancel", RUN_ID],
        ],
    )
    def test_a_sql_error_prints_one_line(
        self, monkeypatch: pytest.MonkeyPatch, args: list[str]
    ) -> None:
        # Arrange
        errors = _fake_sql_errors(monkeypatch)
        message = "(1304): 01b-abc: 003001 (42501): SQL access control error:\n"
        message += "Insufficient privileges to operate on service."
        handle = RunHandle({}, error=errors.snowpark(message))
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: handle
        )

        # Act
        result = _invoke(args)

        # Assert
        assert result.exit_code == 1
        assert result.output == (
            "Error: (1304): 01b-abc: 003001 (42501): SQL access control error: "
            "Insufficient privileges to operate on service.\n"
        )
        assert handle.closed == 1


class TestStatusAndLogs:
    def test_status_and_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        handle = RunHandle({HEAD: "DONE", WORKERS: "DONE"}, stored=["worker 1 exited"])
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: handle
        )

        # Act
        status = _invoke(["snowflake", "status", RUN_ID])
        logs = _invoke(
            ["snowflake", "logs", RUN_ID, "--role", "worker", "--instance", "1"]
        )
        broker_logs = _invoke(["snowflake", "logs", RUN_ID, "--role", "broker"])

        # Assert
        assert status.exit_code == 0, status.output
        assert status.output == (
            "broker\tgone\nhead\tdone\nworker\tdone\nstate: succeeded\n"
        )
        assert logs.exit_code == 0, logs.output
        assert logs.output == "worker 1 exited\n"
        assert broker_logs.exit_code == 1
        assert f"{BROKER} was dropped" in broker_logs.output
        assert handle.closed == 3

    def test_status_and_logs_report_a_missing_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
        monkeypatch.setitem(sys.modules, "snowflake.snowpark", None)

        status = _invoke(["snowflake", "status", RUN_ID])
        logs = _invoke(["snowflake", "logs", RUN_ID])

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
        assert "run_id must look like" in status.output

    def test_cancel_drops_the_three_services(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Handle(Closable):
            def __init__(self) -> None:
                super().__init__()
                self.statements: list[str] = []

            def sql(self, text: str) -> Handle:
                self.statements.append(text)
                return self

            def collect(self) -> list[Row]:
                return []

        handle = Handle()
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: handle
        )

        result = _invoke(["snowflake", "cancel", RUN_ID])

        assert result.exit_code == 0, result.output
        assert handle.statements == [
            f"DROP SERVICE IF EXISTS {BROKER} FORCE",
            f"DROP SERVICE IF EXISTS {HEAD} FORCE",
            f"DROP SERVICE IF EXISTS {WORKERS} FORCE",
        ]
        assert handle.closed == 1


def _saved_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: str,
    gpu: dict[str, GpuPool] | None = None,
) -> None:
    """Write ``[arena]`` with a key, and serve setup.json from every fake session."""
    connections = tmp_path / "connections.toml"
    key = tmp_path / "key.p8"
    key.write_bytes(b"key")
    monkeypatch.setattr(
        "agilerl.arena.snowflake.session.connections_file", lambda: connections
    )
    write_connection("arena", job_connection_fields("ORGNAME-ACCOUNT", key))
    profile = AccountProfile(
        image=image,
        warehouse="arena_wh",
        stage="arena.training.runs",
        metrics_table="arena.training.arena_metrics",
        external_access_integrations=("arena_hf_eai",),
        pools=AccountPools(
            broker=CpuPool("arena_pulsar_pool", "CPU_X64_L"),
            head=CpuPool("arena_head_pool", "CPU_X64_SL"),
            gpu=gpu or {"GPU_NV_S": GpuPool("arena_gpu_GPU_NV_S", 2, 1, 24)},
        ),
    )
    monkeypatch.setattr(Closable, "setup_doc", profile.as_dict())


class TestSubmitProfile:
    def test_omitted_flags_come_from_the_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("algorithm: {name: SFT}\n", encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, _services())

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(
            [
                "snowflake",
                "submit",
                "MANIFEST",
                "--source-table",
                "db.public.prompts",
                "--gpus",
                "1",
            ],
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert seen[0].gpu_pool == "arena_gpu_GPU_NV_S"
        assert seen[0].gpu_family == "GPU_NV_S"
        assert seen[0].broker_family == "CPU_X64_L"
        assert seen[0].head_family == "CPU_X64_SL"
        assert seen[0].image == "/arena/training/images/snowflake-job:1"
        assert seen[0].warehouse == "arena_wh"
        assert seen[0].external_access_integrations == ("arena_hf_eai",)
        assert not hasattr(seen[0], "pulsar_url")

    def test_a_family_that_does_not_divide_lists_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("algorithm: {name: SFT}\n", encoding="utf-8")
        _saved_account(
            tmp_path,
            monkeypatch,
            "/arena/training/images/snowflake-job:1",
            {"GPU_NV_M": GpuPool("arena_gpu_GPU_NV_M", 16, 4, 24)},
        )
        session = Closable()
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: session
        )

        result = _invoke(
            [
                "snowflake",
                "submit",
                "MANIFEST",
                "--source-table",
                TABLE,
                "--gpus",
                "2",
                "--gpu-family",
                "GPU_NV_M",
            ],
            manifest,
        )

        assert result.exit_code != 0
        assert "GPU_NV_M runs 4, 8, 12, 16 GPUs." in result.output
        assert [s.startswith(SETUP_READ) for s in session.statements] == [True]
        assert session.closed == 1

    def test_a_named_family_that_does_not_fit_stops_before_job_sql(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("algorithm: {name: SFT}\n", encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.manifest_fits",
            lambda manifest_path, pool, gpus: False,
        )
        session = Closable()
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: session
        )

        result = _invoke(
            [
                "snowflake",
                "submit",
                "MANIFEST",
                "--source-table",
                TABLE,
                "--gpus",
                "1",
                "--gpu-family",
                "GPU_NV_S",
            ],
            manifest,
        )

        assert result.exit_code != 0
        assert "GPU_NV_S (24 GiB)" in result.output
        assert "cannot hold this manifest" in result.output
        assert [s.startswith(SETUP_READ) for s in session.statements] == [True]

    def test_omitted_family_skips_a_memory_that_does_not_fit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("algorithm: {name: SFT}\n", encoding="utf-8")
        _saved_account(
            tmp_path,
            monkeypatch,
            "/arena/training/images/snowflake-job:1",
            {
                "GPU_NV_S": GpuPool("arena_gpu_GPU_NV_S", 8, 1, 24),
                "GPU_NV_L": GpuPool("arena_gpu_GPU_NV_L", 16, 8, 40),
            },
        )
        seen: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, _services())

        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.manifest_fits",
            lambda manifest_path, pool, gpus: pool.gpu_memory_gib >= 40,
        )
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(
            ["snowflake", "submit", "MANIFEST", "--source-table", TABLE, "--gpus", "8"],
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert seen[0].gpu_family == "GPU_NV_L"

    def test_fit_checks_the_family_pool_with_every_gpu_of_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("algorithm: {name: SFT}\n", encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen: list[tuple[GpuPool, int]] = []

        def fits(path: Path, pool: GpuPool, gpus: int) -> bool:
            seen.append((pool, gpus))
            return True

        monkeypatch.setattr("agilerl.arena.snowflake.cli.manifest_fits", fits)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.submit_job",
            lambda session, path, settings: (RUN_ID, _services()),
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        # Act
        result = _invoke(
            ["snowflake", "submit", "MANIFEST", "--source-table", TABLE, "--gpus", "8"],
            manifest,
        )

        # Assert
        assert result.exit_code == 0, result.output
        assert seen == [(GpuPool("arena_gpu_GPU_NV_S", 2, 1, 24), 8)]

    def test_a_named_family_the_estimator_cannot_size_still_submits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("algorithm: {name: SFT}\n", encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen: list[SubmitSettings] = []

        def unsized(path: Path, pool: GpuPool, gpus: int) -> bool:
            msg = "no architecture for this model"
            raise ValueError(msg)

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, _services())

        monkeypatch.setattr("agilerl.arena.snowflake.cli.manifest_fits", unsized)
        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )
        args = ["snowflake", "submit", "MANIFEST", "--source-table", TABLE]
        args += ["--gpus", "1", "--gpu-family", "GPU_NV_S"]

        # Act
        with pytest.warns(UserWarning, match="GPU_NV_S is used unchecked"):
            result = _invoke(args, manifest)

        # Assert
        assert result.exit_code == 0, result.output
        assert seen[0].gpu_family == "GPU_NV_S"

    def test_an_explicit_flag_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, _services())

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(
            [
                "snowflake",
                "submit",
                "MANIFEST",
                "--warehouse",
                "other_wh",
                "--eai",
                "custom_eai",
                "--source-table",
                TABLE,
                "--gpus",
                "1",
            ],
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert seen[0].warehouse == "other_wh"
        assert seen[0].gpu_pool == "arena_gpu_GPU_NV_S"
        assert seen[0].image == "/arena/training/images/snowflake-job:1"
        assert seen[0].external_access_integrations == ("arena_hf_eai", "custom_eai")

    def test_eai_adds_to_the_profile_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, _services())

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(
            [
                "snowflake",
                "submit",
                "MANIFEST",
                "--eai",
                "env_eai",
                "--eai",
                "arena_hf_eai",
                "--source-table",
                TABLE,
                "--gpus",
                "1",
            ],
            manifest,
        )

        assert result.exit_code == 0, result.output
        assert seen[0].external_access_integrations == ("arena_hf_eai", "env_eai")

    def test_explicit_flags_print_no_banner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        _saved_account(
            tmp_path, monkeypatch, "/arena/training/images/snowflake-job:saved"
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.submit_job",
            lambda session, path, settings: (RUN_ID, _services()),
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )
        runner = TestSubmitCommand()

        result = _invoke(runner._options(["--eai", "arena_hf_eai"]), manifest)

        assert result.exit_code == 0, result.output
        assert "pool       " not in result.output
        assert result.output.splitlines() == [
            f"run_id: {RUN_ID}",
            f"broker: {BROKER}",
            f"head: {HEAD}",
            f"workers: {WORKERS}",
        ]

    def test_an_empty_image_stops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "")
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )

        result = _invoke(
            ["snowflake", "submit", "MANIFEST", "--source-table", TABLE, "--gpus", "1"],
            manifest,
        )

        assert result.exit_code != 0
        assert result.output == (
            "Error: No trainer image is saved for this account yet.\n"
            "Run arena snowflake setup again. It copies the image.\n"
        )

    def test_the_admin_connection_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections = tmp_path / "connections.toml"
        monkeypatch.setattr(
            "agilerl.arena.snowflake.session.connections_file", lambda: connections
        )
        write_connection(
            "arena-admin", admin_connection_fields("ORGNAME-ACCOUNT", "NICK")
        )

        def fail_open(name: str) -> Closable:
            raise AssertionError(name)

        monkeypatch.setattr("agilerl.arena.snowflake.cli.open_session", fail_open)
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        runner = TestSubmitCommand()

        result = _invoke(runner._options(["--connection", "arena-admin"]), manifest)

        assert result.exit_code != 0
        assert (
            "Connection arena-admin signs in with a browser as ACCOUNTADMIN.\n"
            "Job commands use the arena connection (user arena_job_user)."
        ) in result.output

    def test_a_missing_private_key_is_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections = tmp_path / "connections.toml"
        missing = tmp_path / "missing.p8"
        monkeypatch.setattr(
            "agilerl.arena.snowflake.session.connections_file", lambda: connections
        )
        write_connection("arena", job_connection_fields("ORGNAME-ACCOUNT", missing))
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        runner = TestSubmitCommand()

        result = _invoke(runner._options(["--connection", "arena"]), manifest)

        assert result.exit_code != 0
        assert f"No private key at {missing}." in result.output
        assert "The private key is not in Snowflake." in result.output

    def test_an_account_without_setup_json_names_setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setattr(Closable, "setup_doc", None)
        session = Closable()
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: session
        )
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")

        result = _invoke(
            ["snowflake", "submit", "MANIFEST", "--source-table", TABLE, "--gpus", "1"],
            manifest,
        )

        assert result.exit_code != 0
        assert result.output == f"Error: {NO_SETUP}\n"
        assert session.closed == 1

    def test_reads_setup_json_on_every_submit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("x: 1\n", encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            seen.append(settings)
            return (RUN_ID, _services())

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )
        args = ["snowflake", "submit", "MANIFEST", "--source-table", TABLE]
        args += ["--gpus", "1"]

        # Act
        first = _invoke(args, manifest)
        _saved_account(
            tmp_path,
            monkeypatch,
            "/arena/training/images/snowflake-job:2",
            {"GPU_NV_M": GpuPool("arena_gpu_GPU_NV_M", 8, 4, 24)},
        )
        second = _invoke([*args[:-1], "4"], manifest)

        # Assert
        assert first.exit_code == 0, first.output
        assert second.exit_code == 0, second.output
        assert [s.image for s in seen] == [
            "/arena/training/images/snowflake-job:1",
            "/arena/training/images/snowflake-job:2",
        ]
        assert [s.gpu_pool for s in seen] == [
            "arena_gpu_GPU_NV_S",
            "arena_gpu_GPU_NV_M",
        ]


SFT_MANIFEST = """\
algorithm:
  name: SFT
environment:
  dataset: prompts
  columns:
    prompt: PROMPT
    response: RESPONSE
network:
  pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B
  max_context_length: 128
training:
  max_steps: 2
  checkpoint_export:
    format: merged
"""


class _SqlResult:
    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows

    def collect(self) -> list[Row]:
        return self.rows


class _Files:
    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = False,
        overwrite: bool = False,
    ) -> None:
        return None


class _Recording(Closable):
    def __init__(self) -> None:
        super().__init__()
        self.file = _Files()

    def sql(self, text: str) -> _SqlResult:
        if text.startswith("DESCRIBE"):
            self.statements.append(text)
            return _SqlResult([Row(max_nodes=8)])
        return super().sql(text)


def _capabilities(enabled: object) -> dict[str, object]:
    return {
        "ok": True,
        "data": {
            "schemaVersion": 1,
            "features": {"snowflakeTraining": enabled},
        },
    }


class _HttpResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.is_success = 200 <= status_code < 300
        self.headers: dict[str, str] = {}

    def json(self) -> object:
        return self._payload


def _install_http(
    monkeypatch: pytest.MonkeyPatch,
    get_result: object,
    headers_seen: list[object] | None = None,
) -> list[str]:
    calls: list[str] = []

    class Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def get(
            self, url: str, headers: object = None, timeout: object = None
        ) -> object:
            calls.append(url)
            if headers_seen is not None:
                headers_seen.append(headers)
            if isinstance(get_result, Exception):
                raise get_result
            return get_result

        def post(
            self,
            url: str,
            headers: object = None,
            json: object = None,
            timeout: object = None,
        ) -> object:
            raise AssertionError(url)

    monkeypatch.setattr("agilerl.arena.snowflake.license.httpx.Client", Client)
    return calls


class TestSubmitLicense:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, get_result: object):
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(SFT_MANIFEST, encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setenv("ARENA_ORG_KEY", "arena_org_cli")
        monkeypatch.setenv("ARENA_EXTERNAL_USER_ID", "user-1")
        session = _Recording()
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: session
        )
        calls = _install_http(monkeypatch, get_result)
        result = _invoke(
            TestSubmitCommand()._options(["--source-table", "arena.training.prompts"]),
            manifest,
        )
        return result, session, calls

    @pytest.mark.parametrize(
        ("get_result", "match"),
        [
            (_HttpResponse(200, {"ok": False}), "which is not the Arena API"),
            (_HttpResponse(200, _capabilities(False)), "not enabled"),
            (httpx.ConnectError("refused"), "Could not reach Arena at"),
        ],
    )
    def test_submit_does_not_execute_when_arena_refuses(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        get_result: object,
        match: str,
    ) -> None:
        # Act
        result, session, calls = self._run(tmp_path, monkeypatch, get_result)

        # Assert
        assert result.exit_code != 0
        assert match in result.output
        assert calls
        assert session.statements == []
        assert not any(
            "EXECUTE JOB SERVICE" in statement for statement in session.statements
        )

    def test_submit_executes_when_snowflake_training_is_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Act
        result, session, calls = self._run(
            tmp_path, monkeypatch, _HttpResponse(200, _capabilities(True))
        )

        # Assert
        assert result.exit_code == 0, result.output
        assert calls[0].endswith("/api/cli/v1/capabilities")
        assert any(
            statement.startswith("EXECUTE JOB SERVICE")
            for statement in session.statements
        )
        assert session.closed == 1

    def test_no_local_credentials_skips_the_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(SFT_MANIFEST, encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        for name in ("ARENA_ORG_KEY", "ARENA_EXTERNAL_USER_ID", "ARENA_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.load_credentials", lambda path=None: None
        )
        session = _Recording()
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: session
        )
        calls = _install_http(monkeypatch, AssertionError("no Arena call"))

        # Act
        result = _invoke(
            TestSubmitCommand()._options(["--source-table", "arena.training.prompts"]),
            manifest,
        )

        # Assert
        assert result.exit_code == 0, result.output
        assert calls == []
        assert "arena login" not in result.output
        assert any(
            statement.startswith("EXECUTE JOB SERVICE")
            for statement in session.statements
        )

    def _submit_without_partner_keys(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        stored: dict[str, object] | None,
    ) -> tuple[Result, list[object]]:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(SFT_MANIFEST, encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.delenv("ARENA_ORG_KEY", raising=False)
        monkeypatch.delenv("ARENA_EXTERNAL_USER_ID", raising=False)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.load_credentials", lambda path=None: stored
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: _Recording()
        )
        headers_seen: list[object] = []
        _install_http(
            monkeypatch, _HttpResponse(200, _capabilities(True)), headers_seen
        )
        result = _invoke(
            TestSubmitCommand()._options(["--source-table", "arena.training.prompts"]),
            manifest,
        )
        return result, headers_seen

    def test_an_api_key_is_sent_as_a_bearer_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setenv("ARENA_API_KEY", "arena_key_1")

        # Act
        result, headers_seen = self._submit_without_partner_keys(
            tmp_path, monkeypatch, None
        )

        # Assert
        assert result.exit_code == 0, result.output
        assert headers_seen == [
            {"Authorization": "Bearer arena_key_1", CLIENT_VERSION_HEADER: __version__}
        ]

    def test_a_stored_login_token_is_sent_as_a_bearer_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.delenv("ARENA_API_KEY", raising=False)

        # Act
        result, headers_seen = self._submit_without_partner_keys(
            tmp_path, monkeypatch, {"access_token": "login_tok"}
        )

        # Assert
        assert result.exit_code == 0, result.output
        assert headers_seen == [
            {"Authorization": "Bearer login_tok", CLIENT_VERSION_HEADER: __version__}
        ]

    def test_half_set_partner_credentials_do_not_execute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(SFT_MANIFEST, encoding="utf-8")
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        monkeypatch.setenv("ARENA_ORG_KEY", "arena_org_cli")
        monkeypatch.delenv("ARENA_EXTERNAL_USER_ID", raising=False)
        session = _Recording()
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: session
        )

        # Act
        result = _invoke(
            TestSubmitCommand()._options(["--source-table", "arena.training.prompts"]),
            manifest,
        )

        # Assert
        assert result.exit_code != 0
        assert "must be set together" in result.output
        assert session.statements == []


class TestSubmitImageCheck:
    PROFILE_IMAGE = "/arena/training/images/snowflake-job:32.6.0-py313-cu130"

    def _submit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        cli_bundle: str,
        extra: list[str] | None = None,
    ) -> tuple[Result, list[SubmitSettings]]:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("algorithm: {name: SFT}\n", encoding="utf-8")
        stamp = tmp_path / "bundle_version.txt"
        stamp.write_text(f"{cli_bundle}\n", encoding="utf-8")
        monkeypatch.setattr("agilerl.arena.snowflake.cli.BUNDLE_STAMP", stamp)
        _saved_account(tmp_path, monkeypatch, self.PROFILE_IMAGE)
        submitted: list[SubmitSettings] = []

        def fake_submit(session: object, path: Path, settings: SubmitSettings):
            submitted.append(settings)
            return (RUN_ID, _services())

        monkeypatch.setattr("agilerl.arena.snowflake.cli.submit_job", fake_submit)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.cli.open_session", lambda name: Closable()
        )
        args = ["snowflake", "submit", "MANIFEST", "--source-table", TABLE]
        result = _invoke([*args, "--gpus", "1", *(extra or [])], manifest)
        return result, submitted

    def test_the_matching_image_submits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, submitted = self._submit(tmp_path, monkeypatch, "32.6.0")

        assert result.exit_code == 0, result.output
        assert submitted[0].image == self.PROFILE_IMAGE

    def test_a_newer_cli_stops_before_any_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Act
        result, submitted = self._submit(tmp_path, monkeypatch, "32.7.0")

        # Assert
        assert result.exit_code != 0
        assert (
            "Re-run arena snowflake setup to update the trainer image to 32.7.0."
            in result.output
        )
        assert submitted == []

    def test_a_newer_image_stops_before_any_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Act
        result, submitted = self._submit(tmp_path, monkeypatch, "32.5.0")

        # Assert
        assert result.exit_code != 0
        assert "the account's trainer image is 32.6.0" in result.output
        assert 'pip install -U "agilerl-arena[snowflake]"' in result.output
        assert submitted == []

    def test_an_explicit_image_is_not_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = "/arena/training/images/snowflake-job:32.5.0-py313-cu130"

        result, submitted = self._submit(
            tmp_path, monkeypatch, "32.7.0", ["--image", image]
        )

        assert result.exit_code == 0, result.output
        assert submitted[0].image == image
