# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Client-side Snowflake ML Job submission."""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agilerl.arena.snowflake.job import (
    ENTRYPOINT,
    SubmitSettings,
    job_document,
    job_id_from_run_id,
    new_run_id,
    submit_job,
)
from agilerl.arena.snowflake.source import QuerySource, TableSource

RUN_ID = "arena-20261109163457-abc123"
NETWORK = {
    "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
    "max_context_length": 128,
}


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _sft(export: str = "merged") -> dict:
    return {
        "algorithm": {"name": "SFT"},
        "environment": {
            "dataset": "prompts",
            "columns": {"prompt": "PROMPT", "response": "RESPONSE"},
        },
        "network": NETWORK,
        "training": {
            "max_steps": 2,
            "checkpoint_export": {"format": export},
        },
    }


def _dpo() -> dict:
    return {
        "algorithm": {"name": "DPO"},
        "environment": {"dataset": "prefs"},
        "network": NETWORK,
        "training": {
            "max_steps": 2,
            "checkpoint_export": {"format": "merged"},
        },
    }


def _rl() -> dict:
    return {
        "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
        "environment": {
            "env_type": "rollout",
            "env_url": "https://env.example.com",
            "max_turns": 1,
        },
        "network": NETWORK,
        "training": {
            "max_steps": 2,
            "checkpoint_export": {"format": "merged"},
        },
    }


def _settings(source: TableSource | QuerySource | None = None) -> SubmitSettings:
    return SubmitSettings(
        compute_pool="arena_gpu_pool",
        runtime_environment="cre@arena_runtime",
        payload_stage="arena.training.payload",
        runs_stage="arena.training.runs",
        metrics_table="arena.training.arena_metrics",
        source=source or TableSource("arena.training.prompts"),
        nodes=2,
        external_access_integrations=("arena_hf_eai", "arena_pulsar_eai"),
        checkpoint_steps=10,
        pulsar_url="pulsar://broker.example:6650",
        pulsar_topic_prefix="non-persistent://customer/arena",
    )


def _install_submit(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {}

    def submit_directory(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        payload = Path(args[0])
        calls["manifest"] = (payload / "manifest.yaml").read_text(encoding="utf-8")
        calls["job"] = json.loads((payload / "job.json").read_text(encoding="utf-8"))
        return SimpleNamespace(id="mljob-7")

    jobs = types.ModuleType("snowflake.ml.jobs")
    jobs.submit_directory = submit_directory
    for name in ("snowflake", "snowflake.ml"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "snowflake.ml.jobs", jobs)
    return calls


class TestNewRunId:
    def test_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.snowflake.job.secrets.token_hex", lambda n: "ab" * n
        )
        moment = datetime(2026, 11, 9, 16, 34, 57, tzinfo=timezone.utc)

        assert new_run_id(moment) == "arena-20261109163457-ababab"


class TestJobIdFromRunId:
    def test_hex_suffix_is_an_integer(self) -> None:
        assert job_id_from_run_id(RUN_ID) == int("abc123", 16)

    def test_a_bad_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="run_id must look like"):
            job_id_from_run_id("job-1")


class TestSubmitJob:
    def test_sft_payload_and_submit_arguments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _install_submit(monkeypatch)
        manifest = _write(tmp_path, _sft())
        settings = _settings()
        session = object()

        run_id, job_id = submit_job(session, manifest, settings, run_id=RUN_ID)

        assert run_id == RUN_ID
        assert job_id == "mljob-7"
        payload = Path(calls["args"][0])
        assert calls["args"][1] == "arena_gpu_pool"
        assert calls["kwargs"]["entrypoint"] == ENTRYPOINT
        assert calls["kwargs"]["runtime_environment"] == "cre@arena_runtime"
        assert calls["kwargs"]["target_instances"] == 2
        assert calls["kwargs"]["stage_name"] == "arena.training.payload"
        assert calls["kwargs"]["external_access_integrations"] == [
            "arena_hf_eai",
            "arena_pulsar_eai",
        ]
        assert calls["kwargs"]["session"] is session
        assert yaml.safe_load(calls["manifest"])["algorithm"]["name"] == "SFT"
        document = calls["job"]
        assert document == job_document(settings, RUN_ID)
        assert document["source"] == {"kind": "table", "name": "arena.training.prompts"}
        assert document["pulsar_url"] == "pulsar://broker.example:6650"
        assert document["checkpoint_steps"] == 10
        assert not payload.exists()

    def test_dpo_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _install_submit(monkeypatch)

        run_id, job_id = submit_job(
            object(),
            _write(tmp_path, _dpo()),
            _settings(QuerySource("select 1")),
            RUN_ID,
        )

        assert (run_id, job_id) == (RUN_ID, "mljob-7")
        assert calls["job"]["source"] == {"kind": "query", "sql": "select 1"}

    def test_rl_env_url_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_submit(monkeypatch)

        with pytest.warns(UserWarning, match="External Access Integration"):
            run_id, job_id = submit_job(
                object(), _write(tmp_path, _rl()), _settings(), RUN_ID
            )

        assert (run_id, job_id) == (RUN_ID, "mljob-7")

    def test_dataset_rollout_does_not_need_env_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_submit(monkeypatch)
        document = {
            "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
            "environment": {
                "env_type": "rollout",
                "dataset": "openai/gsm8k",
                "reward_file_path": "reward.py",
                "prompt_template": {"user_0": "{question}"},
            },
            "network": NETWORK,
            "training": {
                "max_steps": 2,
                "checkpoint_export": {"format": "merged"},
            },
        }

        run_id, _job_id = submit_job(
            object(), _write(tmp_path, document), _settings(), RUN_ID
        )

        assert run_id == RUN_ID

    def test_env_image_is_rejected(self, tmp_path: Path) -> None:
        document = _rl()
        document["environment"] = {
            "env_type": "rollout",
            "env_image": "env:latest",
        }

        with pytest.raises(ValueError, match="Kubernetes Pod"):
            submit_job(object(), _write(tmp_path, document), _settings(), RUN_ID)

    def test_rollout_without_env_url_is_rejected(self, tmp_path: Path) -> None:
        document = _rl()
        document["environment"] = {
            "env_type": "rollout",
            "entrypoint": "envs:MyEnv",
        }

        with pytest.raises(ValueError, match="needs env_url"):
            submit_job(object(), _write(tmp_path, document), _settings(), RUN_ID)

    def test_non_llm_manifest_is_rejected(self, tmp_path: Path) -> None:
        document = {
            "algorithm": {"name": "DQN", "lr": 0.001},
            "environment": {"name": "LunarLander-v3", "num_envs": 8},
            "network": {
                "latent_dim": 64,
                "encoder_config": {"hidden_size": [64]},
                "head_config": {"hidden_size": [64]},
            },
            "training": {"max_steps": 10},
        }

        with pytest.raises(ValueError, match="SFT, DPO, and RL"):
            submit_job(object(), _write(tmp_path, document), _settings(), RUN_ID)

    def test_adapter_export_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_submit(monkeypatch)

        with pytest.warns(UserWarning, match="not merged"):
            submit_job(object(), _write(tmp_path, _sft("adapter")), _settings(), RUN_ID)

    def test_empty_integrations_are_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _install_submit(monkeypatch)
        settings = SubmitSettings(
            compute_pool="arena_gpu_pool",
            runtime_environment="cre@arena_runtime",
            payload_stage="arena.training.payload",
            runs_stage="arena.training.runs",
            metrics_table="arena.training.arena_metrics",
            source=TableSource("arena.training.prompts"),
            nodes=1,
            external_access_integrations=(),
            checkpoint_steps=None,
            pulsar_url="pulsar://broker.example:6650",
            pulsar_topic_prefix="non-persistent://customer/arena",
        )

        submit_job(object(), _write(tmp_path, _sft()), settings, RUN_ID)

        assert calls["kwargs"]["external_access_integrations"] is None
        assert calls["job"]["checkpoint_steps"] is None

    def test_a_missing_job_id_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _install_submit(monkeypatch)

        def submit_directory(*args, **kwargs):
            calls["args"] = args
            return SimpleNamespace(id="")

        sys.modules["snowflake.ml.jobs"].submit_directory = submit_directory

        with pytest.raises(RuntimeError, match="did not return an id"):
            submit_job(object(), _write(tmp_path, _sft()), _settings(), RUN_ID)

    def test_missing_sdk_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("snowflake", "snowflake.ml"):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "snowflake.ml.jobs", None)

        with pytest.raises(ImportError, match="agilerl-arena\\[snowflake\\]"):
            submit_job(object(), _write(tmp_path, _sft()), _settings(), RUN_ID)
