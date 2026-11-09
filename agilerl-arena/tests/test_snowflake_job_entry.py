# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""In-job trainer command, checkpoint upload, and metrics merge."""

from __future__ import annotations

import json
import runpy
import sys
import warnings
from pathlib import Path

import pytest
import yaml

from agilerl.arena.snowflake import job_entry
from agilerl.arena.snowflake.job import job_id_from_run_id
from agilerl.arena.snowflake.job_entry import (
    UploadTracker,
    main,
    manifest_columns,
    metrics_file_format,
    metrics_merge_sql,
    publish_outputs,
    ready_checkpoint_files,
    rollout_topic_route,
    run_job,
    source_from_document,
    trainer_command,
    trainer_env,
)
from agilerl.arena.snowflake.source import QuerySource, TableSource

RUN_ID = "arena-20261109163457-abc123"
JOB_ID = str(job_id_from_run_id(RUN_ID))


def _job(checkpoint_steps: int | None = 5) -> dict:
    return {
        "run_id": RUN_ID,
        "nodes": 2,
        "checkpoint_steps": checkpoint_steps,
        "pulsar_url": "pulsar://broker.example:6650",
        "pulsar_topic_prefix": "non-persistent://customer/arena",
        "runs_stage": "ARENA.TRAINING.RUNS",
        "metrics_table": "ARENA.TRAINING.ARENA_METRICS",
        "source": {"kind": "table", "name": "ARENA.TRAINING.PROMPTS"},
    }


class FileApi:
    def __init__(self, puts: list[dict]) -> None:
        self.puts = puts

    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = True,
        overwrite: bool = False,
    ) -> None:
        self.puts.append(
            {
                "local": local,
                "stage": stage,
                "auto_compress": auto_compress,
                "overwrite": overwrite,
            }
        )


class Frame:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def select(self, *names: str) -> Frame:
        return Frame([{name: row[name] for name in names} for row in self.rows])

    def to_pandas_batches(self):
        import pandas as pd

        yield pd.DataFrame(self.rows)


class Session:
    def __init__(self) -> None:
        self.puts: list[dict] = []
        self.statements: list[str] = []
        self.file = FileApi(self.puts)

    def table(self, name: str) -> Frame:
        return Frame([{"PROMPT": "one", "RESPONSE": "a"}])

    def sql(self, text: str) -> Session:
        self.statements.append(text)
        return self

    def collect(self) -> list:
        return []


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestTrainerCommand:
    def test_command_includes_metrics_and_checkpoint_steps(self) -> None:
        assert trainer_command(_job()) == [
            "python",
            "-m",
            "agilerl_ray.train",
            "--experiment-id",
            JOB_ID,
            "--manifest",
            "manifest.yaml",
            "--custom-dataset-s3",
            "/tmp/arena/data",
            "--checkpoint-directory",
            "/tmp/arena/ckpt",
            "--metrics-file",
            "/tmp/arena/metrics.jsonl",
            "--num-nodes",
            "2",
            "--checkpoint-steps",
            "5",
        ]

    def test_checkpoint_steps_are_omitted_when_unset(self) -> None:
        command = trainer_command(_job(None))

        assert "--checkpoint-steps" not in command
        assert command[command.index("--experiment-id") + 1] == JOB_ID


class TestTrainerEnv:
    def test_pulsar_namespace_and_removed_keys(self) -> None:
        env = trainer_env(
            _job(),
            {
                "PATH": "/usr/bin",
                "OTLP_COLLECTOR_ENDPOINT": "http://otel:4317",
                "DISABLE_PULSAR": "1",
            },
        )

        assert env["PATH"] == "/usr/bin"
        assert env["RAY_ADDRESS"] == "auto"
        assert env["APP_PULSAR_URL"] == "pulsar://broker.example:6650"
        assert env["APP_TOPIC_PREFIX"] == "non-persistent://customer/arena"
        assert env["JOB_ID"] == JOB_ID
        assert env["JOB_ID"].isdigit()
        assert "DISABLE_PULSAR" not in env
        assert "OTLP_COLLECTOR_ENDPOINT" not in env
        assert (
            env["AGILERL_ROLLOUT_TOPIC_ROUTE"]
            == "persistent://customer/arena/llm-rollout"
        )

    def test_rollout_route_keeps_a_persistent_suffix(self) -> None:
        assert (
            rollout_topic_route("persistent://customer/arena/llm-rollout")
            == "persistent://customer/arena/llm-rollout"
        )
        assert (
            rollout_topic_route("customer/arena")
            == "persistent://customer/arena/llm-rollout"
        )


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
                    "network": {
                        "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                        "max_context_length": 128,
                    },
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


class TestReadyCheckpointFiles:
    def test_checkpoint_waits_for_a_later_step_or_exit(self, tmp_path: Path) -> None:
        third = tmp_path / "checkpoint_3"
        fourth = tmp_path / "checkpoint_4"
        _touch(third / "adapter.safetensors")
        _touch(third / "merged" / "model.safetensors")
        _touch(fourth / "adapter.safetensors")
        _touch(fourth / "merged" / "model.safetensors")

        waiting = ready_checkpoint_files(tmp_path, trainer_exited=False)

        assert [path.name for path in waiting] == ["adapter.safetensors"]
        assert waiting[0].parent == third

        _touch(third / "merged" / ".complete")
        with_merged = ready_checkpoint_files(tmp_path, trainer_exited=False)
        assert (third / "merged" / "model.safetensors") in with_merged
        assert (fourth / "adapter.safetensors") not in with_merged

        finished = ready_checkpoint_files(tmp_path, trainer_exited=True)
        assert (fourth / "adapter.safetensors") in finished
        assert (fourth / "merged" / "model.safetensors") not in finished

        _touch(fourth / "merged" / ".complete")
        done = ready_checkpoint_files(tmp_path, trainer_exited=True)
        assert (fourth / "merged" / ".complete") in done

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert ready_checkpoint_files(tmp_path, trainer_exited=True) == []


class TestPublishOutputs:
    def test_uploads_once_and_merges_on_run_id_and_seq(self, tmp_path: Path) -> None:
        session = Session()
        ckpt = tmp_path / "ckpt"
        _touch(ckpt / "checkpoint_3" / "adapter.safetensors", "weights")
        _touch(ckpt / "checkpoint_3" / "merged" / ".complete", "")
        _touch(ckpt / "checkpoint_3" / "merged" / "model.safetensors", "merged")
        manifest = tmp_path / "manifest.yaml"
        _touch(manifest, "manifest")
        metrics = tmp_path / "metrics.jsonl"
        _touch(metrics, '{"seq": 1}\n')
        tracker = UploadTracker()
        job = _job()

        publish_outputs(
            session, job, ckpt, manifest, metrics, tracker, trainer_exited=True
        )
        first = list(session.puts)
        publish_outputs(
            session, job, ckpt, manifest, metrics, tracker, trainer_exited=True
        )

        assert session.puts == first
        stages = {item["stage"] for item in first}
        run = f"@ARENA.TRAINING.RUNS/{RUN_ID}"
        assert f"{run}" in stages
        assert f"{run}/checkpoint_3" in stages
        assert f"{run}/checkpoint_3/merged" in stages
        assert all(item["auto_compress"] is False for item in first)
        assert all(item["overwrite"] is True for item in first)
        assert "ON t.RUN_ID = s.RUN_ID AND t.SEQ = s.SEQ" in session.statements[0]
        assert "WHEN NOT MATCHED THEN INSERT" in session.statements[0]
        assert f"'{RUN_ID}'" in session.statements[0]
        assert "ARENA.TRAINING.ARENA_JSON" in session.statements[0]

        metrics.write_text('{"seq": 1}\n{"seq": 2}\n', encoding="utf-8")
        publish_outputs(
            session, job, ckpt, manifest, metrics, tracker, trainer_exited=True
        )

        assert len(session.puts) == len(first) + 1
        assert len(session.statements) == 2
        checkpoint_puts = [
            item for item in session.puts if "checkpoint_3" in item["stage"]
        ]
        assert len(checkpoint_puts) == 3

    def test_bad_stage_name_is_rejected(self, tmp_path: Path) -> None:
        job = _job()
        job["runs_stage"] = "runs; drop table arena_metrics"
        manifest = tmp_path / "manifest.yaml"
        _touch(manifest, "manifest")

        with pytest.raises(ValueError, match="Snowflake identifier"):
            publish_outputs(
                Session(),
                job,
                tmp_path / "ckpt",
                manifest,
                tmp_path / "missing.jsonl",
                UploadTracker(),
                trainer_exited=False,
            )

    def test_bare_table_uses_arena_json_format(self) -> None:
        assert metrics_file_format("ARENA_METRICS") == "ARENA_JSON"
        sql = metrics_merge_sql(RUN_ID, "ARENA_METRICS", "@ARENA.TRAINING.RUNS")
        assert "FILE_FORMAT => 'ARENA_JSON'" in sql
        assert f"@ARENA.TRAINING.RUNS/{RUN_ID}/metrics.jsonl" in sql


class TestRunJob:
    def _patch_paths(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(job_entry, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(job_entry, "CKPT_DIR", tmp_path / "ckpt")
        monkeypatch.setattr(job_entry, "METRICS_PATH", tmp_path / "metrics.jsonl")

    def _payload(self, tmp_path: Path) -> Path:
        payload = tmp_path / "payload"
        payload.mkdir()
        (payload / "manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "algorithm": {"name": "SFT"},
                    "environment": {"dataset": "prompts"},
                    "network": {
                        "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                        "max_context_length": 128,
                    },
                    "training": {
                        "max_steps": 1,
                        "checkpoint_export": {"format": "merged"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (payload / "job.json").write_text(json.dumps(_job()), encoding="utf-8")
        return payload

    def test_failed_trainer_still_uploads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_paths(monkeypatch, tmp_path)
        payload = self._payload(tmp_path)
        _touch(tmp_path / "ckpt" / "checkpoint_4" / "adapter.safetensors")
        _touch(tmp_path / "metrics.jsonl", "{}\n")
        session = Session()

        class Proc:
            def __init__(self, args: list[str], env: dict, cwd: Path) -> None:
                self.args = args
                self.env = env
                self.cwd = cwd
                self.calls = 0
                self.killed = False

            def poll(self) -> int | None:
                self.calls += 1
                if self.calls < 2:
                    return None
                return 3

            def kill(self) -> None:
                self.killed = True

            def wait(self) -> int:
                return 3

        monkeypatch.setattr(job_entry.subprocess, "Popen", Proc)

        code = run_job(payload / "job.json", session, poll_interval_s=0)

        assert code == 3
        assert any("checkpoint_4" in item["stage"] for item in session.puts)
        assert session.statements

    def test_upload_failure_stops_the_trainer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_paths(monkeypatch, tmp_path)
        payload = self._payload(tmp_path)
        killed = {"value": False}

        class Proc:
            def __init__(self, args: list[str], env: dict, cwd: Path) -> None:
                self.killed = False

            def poll(self) -> int | None:
                if self.killed:
                    return 1
                return None

            def kill(self) -> None:
                self.killed = True
                killed["value"] = True

            def wait(self) -> int:
                return 1

        real = job_entry.publish_outputs
        calls = {"n": 0}

        def wrapped(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                msg = "stage down"
                raise RuntimeError(msg)
            return real(*args, **kwargs)

        monkeypatch.setattr(job_entry.subprocess, "Popen", Proc)
        monkeypatch.setattr(job_entry, "publish_outputs", wrapped)

        with pytest.raises(RuntimeError, match="stage down"):
            run_job(payload / "job.json", Session(), poll_interval_s=0)

        assert killed["value"] is True

    def test_opens_the_in_job_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_paths(monkeypatch, tmp_path)
        payload = self._payload(tmp_path)
        session = Session()
        seen: dict[str, str | None] = {}

        def open_session(name: str | None) -> Session:
            seen["name"] = name
            return session

        class Proc:
            def __init__(self, args: list[str], env: dict, cwd: Path) -> None:
                self.returncode = 0

            def poll(self) -> int:
                return 0

            def kill(self) -> None:
                msg = "trainer already exited"
                raise AssertionError(msg)

            def wait(self) -> int:
                return 0

        monkeypatch.setattr(job_entry, "open_session", open_session)
        monkeypatch.setattr(job_entry.subprocess, "Popen", Proc)

        assert run_job(payload / "job.json", poll_interval_s=0) == 0
        assert seen["name"] is None


class TestMain:
    def test_usage(self) -> None:
        with pytest.raises(SystemExit, match="usage:"):
            main([])

    def test_delegates_to_run_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(job_entry, "run_job", lambda path: 0)

        assert main([str(tmp_path / "job.json")]) == 0

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
