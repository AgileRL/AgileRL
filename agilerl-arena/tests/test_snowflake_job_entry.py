# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Head entrypoint: data export, trainer command, and metrics merge."""

from __future__ import annotations

import json
import runpy
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from agilerl.arena.snowflake import job_entry
from agilerl.arena.snowflake.job import job_id_from_run_id
from agilerl.arena.snowflake.job_entry import (
    MetricsPublisher,
    main,
    manifest_columns,
    metrics_file_format,
    metrics_merge_sql,
    run_job,
    source_from_document,
    trainer_command,
    trainer_env,
)
from agilerl.arena.snowflake.source import QuerySource, TableSource

RUN_ID = "arena-20261109163457-abc123"
JOB_ID = str(job_id_from_run_id(RUN_ID))
RETRY_ON = (OSError,)
NETWORK = {
    "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
    "max_context_length": 128,
}


def _job(checkpoint_steps: int | None = 5, reward_file: bool = False) -> dict:
    return {
        "run_id": RUN_ID,
        "nodes": 2,
        "checkpoint_steps": checkpoint_steps,
        "reward_file": reward_file,
        "pulsar_url": "pulsar://broker.example:6650",
        "pulsar_topic_prefix": "non-persistent://customer/arena",
        "runs_stage": "ARENA.TRAINING.RUNS",
        "metrics_table": "ARENA.TRAINING.ARENA_METRICS",
        "source": {"kind": "table", "name": "ARENA.TRAINING.PROMPTS"},
    }


class Frame:
    def __init__(self, rows: list[dict[str, object]], columns: list[str]) -> None:
        self.rows = rows
        self._columns = columns

    @property
    def columns(self) -> list[str]:
        return self._columns

    def select(self, *names: str) -> Frame:
        return Frame(
            [{name: row[name] for name in names} for row in self.rows], list(names)
        )

    def to_pandas_batches(self) -> Iterator[pd.DataFrame]:
        for row in self.rows:
            yield pd.DataFrame([row])


class Result:
    def collect(self) -> list[object]:
        return []


class Session:
    """Records PUTs and SQL; ``failures`` PUTs raise before any succeed."""

    def __init__(
        self, rows: list[dict[str, object]] | None = None, failures: int = 0
    ) -> None:
        self.rows = [{"PROMPT": "one", "RESPONSE": "a"}] if rows is None else rows
        self.failures = failures
        self.puts: list[dict[str, object]] = []
        self.statements: list[str] = []
        self.closed = 0
        self.file = self

    def table(self, name: str) -> Frame:
        columns = list(self.rows[0]) if self.rows else ["PROMPT", "RESPONSE"]
        return Frame(self.rows, columns)

    def sql(self, text: str) -> Result:
        self.statements.append(text)
        return Result()

    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = True,
        overwrite: bool = False,
    ) -> None:
        if self.failures > 0:
            self.failures -= 1
            msg = "stage unavailable"
            raise OSError(msg)
        path = Path(local)
        self.puts.append({"stage": stage, "name": path.name, "data": path.read_bytes()})

    def close(self) -> None:
        self.closed += 1


class TestTrainerCommand:
    def test_command_reads_data_from_the_run_mount(self) -> None:
        run_dir = Path("/arena/run")

        assert trainer_command(_job(), run_dir) == [
            sys.executable,
            "-m",
            "agilerl_ray.train",
            "--experiment-id",
            JOB_ID,
            "--manifest",
            "manifest.yaml",
            "--checkpoint-directory",
            str(job_entry.CKPT_DIR),
            "--metrics-file",
            str(job_entry.METRICS_PATH),
            "--num-nodes",
            "2",
            "--custom-dataset-s3",
            str(run_dir / "data"),
            "--checkpoint-steps",
            "5",
        ]

    def test_reward_file_points_at_the_mount(self) -> None:
        run_dir = Path("/arena/run")

        command = trainer_command(_job(reward_file=True), run_dir)

        assert command[command.index("--custom-reward-s3") + 1] == str(
            run_dir / "reward.py"
        )

    def test_optional_flags_are_omitted_when_unset(self) -> None:
        command = trainer_command(_job(None), Path("/arena/run"))

        assert "--checkpoint-steps" not in command
        assert "--custom-reward-s3" not in command

    def test_a_job_without_a_source_passes_no_dataset(self) -> None:
        job = _job()
        job["source"] = None

        command = trainer_command(job, Path("/arena/run"))

        assert "--custom-dataset-s3" not in command


class TestTrainerEnv:
    def test_sets_ray_address_and_drops_otlp(self) -> None:
        env = trainer_env(
            {"PATH": "/usr/bin", "OTLP_COLLECTOR_ENDPOINT": "http://otel:4317"}
        )

        assert env == {"PATH": "/usr/bin", "RAY_ADDRESS": "auto"}


class TestSourceFromDocument:
    def test_table_and_query(self) -> None:
        assert source_from_document({"kind": "table", "name": "T"}) == TableSource("T")
        assert source_from_document(
            {"kind": "query", "sql": "select 1"}
        ) == QuerySource("select 1")

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown Snowflake source kind"):
            source_from_document({"kind": "stage", "name": "T"})


class TestManifestColumns:
    def test_mapped_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "algorithm": {"name": "SFT"},
                    "environment": {
                        "dataset": "prompts",
                        "columns": {"prompt": "PROMPT"},
                    },
                    "network": NETWORK,
                    "training": {"max_steps": 1},
                }
            ),
            encoding="utf-8",
        )

        assert manifest_columns(path) == {"prompt": "PROMPT"}

    def test_non_llm_manifest_has_no_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "algorithm": {"name": "DQN", "lr": 0.001},
                    "environment": {"name": "LunarLander-v3"},
                    "network": {
                        "latent_dim": 64,
                        "encoder_config": {"hidden_size": [64]},
                        "head_config": {"hidden_size": [64]},
                    },
                    "training": {"max_steps": 1},
                }
            ),
            encoding="utf-8",
        )

        assert manifest_columns(path) is None


class TestMetricsPublisherPublish:
    def _publisher(self, tmp_path: Path, session: Session) -> MetricsPublisher:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        return MetricsPublisher(
            session=session,
            job=_job(),
            metrics_path=tmp_path / "metrics.jsonl",
            scratch=scratch,
            retry_on=RETRY_ON,
            sleep=lambda seconds: None,
        )

    def test_merges_only_complete_lines(self, tmp_path: Path) -> None:
        # Arrange
        session = Session()
        publisher = self._publisher(tmp_path, session)
        publisher.metrics_path.write_bytes(b'{"seq": 1}\n{"seq": 2')

        # Act
        result = publisher.publish(final=False)

        # Assert
        assert result is True
        assert session.puts == [
            {
                "stage": f"@ARENA.TRAINING.RUNS/{RUN_ID}",
                "name": "metrics.jsonl",
                "data": b'{"seq": 1}\n',
            }
        ]
        assert "ON t.RUN_ID = s.RUN_ID AND t.SEQ = s.SEQ" in session.statements[0]
        assert "ARENA.TRAINING.ARENA_JSON" in session.statements[0]

    def test_unchanged_lines_are_not_merged_again(self, tmp_path: Path) -> None:
        session = Session()
        publisher = self._publisher(tmp_path, session)
        publisher.metrics_path.write_bytes(b'{"seq": 1}\n')
        publisher.publish(final=False)
        publisher.metrics_path.write_bytes(b'{"seq": 1}\n{"se')

        publisher.publish(final=False)
        publisher.metrics_path.write_bytes(b'{"seq": 1}\n{"seq": 2}\n')
        publisher.publish(final=True)

        assert [item["data"] for item in session.puts] == [
            b'{"seq": 1}\n',
            b'{"seq": 1}\n{"seq": 2}\n',
        ]
        assert len(session.statements) == 2

    def test_missing_file_is_a_no_op(self, tmp_path: Path) -> None:
        session = Session()

        assert self._publisher(tmp_path, session).publish(final=True) is True
        assert session.puts == []

    def test_transient_failure_is_retried(self, tmp_path: Path) -> None:
        session = Session(failures=2)
        publisher = self._publisher(tmp_path, session)
        publisher.metrics_path.write_text('{"seq": 1}\n', encoding="utf-8")

        assert publisher.publish(final=False) is True
        assert len(session.puts) == 1

    def test_non_final_failure_waits_for_the_next_pass(self, tmp_path: Path) -> None:
        session = Session(failures=3)
        publisher = self._publisher(tmp_path, session)
        publisher.metrics_path.write_text('{"seq": 1}\n', encoding="utf-8")

        assert publisher.publish(final=False) is False
        assert publisher.publish(final=False) is True
        assert len(session.puts) == 1

    def test_final_failure_raises(self, tmp_path: Path) -> None:
        publisher = self._publisher(tmp_path, Session(failures=99))
        publisher.metrics_path.write_text('{"seq": 1}\n', encoding="utf-8")

        with pytest.raises(OSError, match="stage unavailable"):
            publisher.publish(final=True)


class TestMetricsMergeSql:
    def test_bare_table_uses_arena_json_format(self) -> None:
        assert metrics_file_format("ARENA_METRICS") == "ARENA_JSON"
        sql = metrics_merge_sql(RUN_ID, "ARENA_METRICS", "@ARENA.TRAINING.RUNS")
        assert "FILE_FORMAT => 'ARENA_JSON'" in sql
        assert f"@ARENA.TRAINING.RUNS/{RUN_ID}/metrics.jsonl" in sql
        assert f"'{RUN_ID}' AS RUN_ID" in sql

    def test_unsafe_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Snowflake identifier"):
            metrics_merge_sql(RUN_ID, "ARENA_METRICS", "runs; drop table t")
        with pytest.raises(ValueError, match="Snowflake identifier"):
            metrics_merge_sql(RUN_ID, "t'; --", "runs")
        with pytest.raises(ValueError, match="run_id must look like"):
            metrics_merge_sql("x' OR '1'='1", "ARENA_METRICS", "runs")


class Proc:
    """Trainer that exits with ``code`` on its second poll."""

    started: ClassVar[list[Proc]] = []

    def __init__(self, args: list[str], env: dict, cwd: Path) -> None:
        self.args = args
        self.env = env
        self.cwd = cwd
        self.polls = 0
        self.code = 0
        Proc.started.append(self)

    def poll(self) -> int | None:
        self.polls += 1
        return None if self.polls < 2 else self.code

    def kill(self) -> None:
        msg = "trainer already exited"
        raise AssertionError(msg)

    def wait(self) -> int:
        return self.code


class TestRunJob:
    @pytest.fixture
    def run_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(job_entry, "CKPT_DIR", tmp_path / "ckpt")
        monkeypatch.setattr(job_entry, "METRICS_PATH", tmp_path / "metrics.jsonl")
        monkeypatch.setattr(job_entry, "WORK_DIR", tmp_path / "work")
        monkeypatch.setattr(job_entry.subprocess, "Popen", Proc)
        Proc.started = []
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "algorithm": {"name": "SFT"},
                    "environment": {
                        "dataset": "prompts",
                        "columns": {"prompt": "PROMPT", "response": "RESPONSE"},
                    },
                    "network": NETWORK,
                    "training": {
                        "max_steps": 1,
                        "checkpoint_export": {"format": "merged"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "job.json").write_text(json.dumps(_job()), encoding="utf-8")
        return run_dir

    def test_data_is_put_to_the_run_stage_and_the_trainer_reads_it(
        self, run_dir: Path, tmp_path: Path
    ) -> None:
        # Arrange
        session = Session(
            rows=[
                {"PROMPT": "one", "RESPONSE": "a"},
                {"PROMPT": "two", "RESPONSE": "b"},
            ]
        )
        (tmp_path / "metrics.jsonl").write_text('{"seq": 1}\n', encoding="utf-8")

        # Act
        code = run_job(run_dir / "job.json", session, RETRY_ON, 0, lambda s: None)

        # Assert
        assert code == 0
        shards = [put for put in session.puts if put["name"] != "metrics.jsonl"]
        assert [put["name"] for put in shards] == [
            "shard-00000.parquet",
            "shard-00001.parquet",
        ]
        assert {put["stage"] for put in shards} == {
            f"@ARENA.TRAINING.RUNS/{RUN_ID}/data"
        }
        prompts = [
            value
            for put in shards
            for value in pq.read_table(pa.BufferReader(put["data"]))
            .column("PROMPT")
            .to_pylist()
        ]
        assert prompts == ["one", "two"]
        args = Proc.started[0].args
        assert args[args.index("--custom-dataset-s3") + 1] == str(
            run_dir.resolve() / "data"
        )
        assert Proc.started[0].cwd == tmp_path / "work"
        assert Proc.started[0].env["RAY_ADDRESS"] == "auto"
        assert (tmp_path / "work" / "manifest.yaml").is_file()
        assert len(session.statements) == 1

    def test_a_job_without_a_source_exports_nothing(self, run_dir: Path) -> None:
        # Arrange
        job = _job()
        job["source"] = None
        (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
        session = Session(rows=[])

        # Act
        code = run_job(run_dir / "job.json", session, RETRY_ON, 0, lambda s: None)

        # Assert
        assert code == 0
        assert session.puts == []
        assert "--custom-dataset-s3" not in Proc.started[0].args

    def test_a_transient_shard_put_is_retried(self, run_dir: Path) -> None:
        session = Session(failures=1)

        code = run_job(run_dir / "job.json", session, RETRY_ON, 0, lambda s: None)

        assert code == 0
        assert [put["name"] for put in session.puts] == ["shard-00000.parquet"]

    def test_a_shard_put_error_outside_retry_on_raises(self, run_dir: Path) -> None:
        with pytest.raises(OSError, match="stage unavailable"):
            run_job(run_dir / "job.json", Session(failures=1), (), 0, lambda s: None)

        assert Proc.started == []

    def test_trainer_is_killed_when_a_merge_raises(
        self, run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        started: list[Running] = []

        class Running(Proc):
            def __init__(self, args: list[str], env: dict, cwd: Path) -> None:
                super().__init__(args, env, cwd)
                self.killed = False
                started.append(self)

            def poll(self) -> int | None:
                return -9 if self.killed else None

            def kill(self) -> None:
                self.killed = True

        class RejectsMerge(Session):
            def sql(self, text: str) -> Result:
                msg = "metrics table missing"
                raise ValueError(msg)

        monkeypatch.setattr(job_entry.subprocess, "Popen", Running)
        (tmp_path / "metrics.jsonl").write_text('{"seq": 1}\n', encoding="utf-8")

        # Act
        with pytest.raises(ValueError, match="metrics table missing"):
            run_job(run_dir / "job.json", RejectsMerge(), RETRY_ON, 0, lambda s: None)

        # Assert
        assert started[0].killed is True

    def test_trainer_failure_code_is_returned(
        self, run_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(args: list[str], env: dict, cwd: Path) -> Proc:
            process = Proc(args, env, cwd)
            process.code = 3
            return process

        monkeypatch.setattr(job_entry.subprocess, "Popen", failing)

        assert (
            run_job(run_dir / "job.json", Session(), RETRY_ON, 0, lambda s: None) == 3
        )

    def test_empty_source_fails_before_the_trainer_starts(self, run_dir: Path) -> None:
        with pytest.raises(ValueError, match="returned no rows"):
            run_job(run_dir / "job.json", Session(rows=[]), RETRY_ON, 0, lambda s: None)

        assert Proc.started == []

    def test_unknown_mapped_column_fails_before_the_trainer_starts(
        self, run_dir: Path
    ) -> None:
        session = Session(rows=[{"prompt": "one", "response": "a"}])

        with pytest.raises(ValueError, match="upper case"):
            run_job(run_dir / "job.json", session, RETRY_ON, 0, lambda s: None)

        assert Proc.started == []


class TestMain:
    def test_usage(self) -> None:
        with pytest.raises(SystemExit, match="usage:"):
            main([])

    def test_runs_the_job_on_the_in_job_session_and_closes_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        session = Session()
        names: list[str | None] = []
        seen: list[tuple[Path, Session, tuple[type[Exception], ...]]] = []

        def open_session(name: str | None) -> Session:
            names.append(name)
            return session

        def fake_run_job(
            path: Path, current: Session, retry_on: tuple[type[Exception], ...]
        ) -> int:
            seen.append((path, current, retry_on))
            return 4

        monkeypatch.setattr(job_entry, "transient_errors", lambda: RETRY_ON)
        monkeypatch.setattr(job_entry, "open_session", open_session)
        monkeypatch.setattr(job_entry, "run_job", fake_run_job)

        # Act
        code = main([str(tmp_path / "job.json")])

        # Assert
        assert code == 4
        assert names == [None]
        assert seen == [(tmp_path / "job.json", session, RETRY_ON)]
        assert session.closed == 1

    def test_closes_the_session_when_the_job_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = Session()

        def fake_run_job(
            path: Path, current: Session, retry_on: tuple[type[Exception], ...]
        ) -> int:
            msg = "trainer crashed"
            raise RuntimeError(msg)

        monkeypatch.setattr(job_entry, "transient_errors", lambda: RETRY_ON)
        monkeypatch.setattr(job_entry, "open_session", lambda name: session)
        monkeypatch.setattr(job_entry, "run_job", fake_run_job)

        with pytest.raises(RuntimeError, match="trainer crashed"):
            main([str(tmp_path / "job.json")])

        assert session.closed == 1

    def test_module_entrypoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["agilerl.arena.snowflake.job_entry"])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*found in sys.modules.*",
                category=RuntimeWarning,
            )
            with pytest.raises(SystemExit, match="usage:"):
                runpy.run_module(
                    "agilerl.arena.snowflake.job_entry",
                    run_name="__main__",
                )
