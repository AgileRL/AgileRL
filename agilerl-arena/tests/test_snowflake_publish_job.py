# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Publish job inside Snowflake: metrics rows, checkpoint load, Registry and service."""

from __future__ import annotations

import enum
import json
import re
import runpy
import sys
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers

from agilerl.arena.snowflake.publish import PublishDocument
from agilerl.arena.snowflake.publish_job import (
    checkpoint_metrics,
    load_text_pipeline,
    main,
    run_publish,
)

RUN_ID = "arena-20261109163457-abc123"
TARGET = "ARENA.TRAINING.ARENA_POLICY_CHECKPOINT_10"
CHAT_SIGNATURE = {"__call__": "openai-chat"}
C1_ROWS = [
    {
        "seq": 1,
        "time": "2026-11-09T15:00:00Z",
        "metrics": {"agent_0/steps": 9, "agent_0/fitness": 0.5},
    },
    {
        "seq": 2,
        "time": "2026-11-09T15:01:00Z",
        "metrics": {"agent_0/loss": 0.41, "agent_0/grad_norm": None},
    },
    {
        "seq": 3,
        "time": "2026-11-09T15:02:00Z",
        "metrics": {"agent_0/steps": 10, "agent_0/fitness": 0.7, "agent_0/kl": None},
    },
    {
        "seq": 4,
        "time": "2026-11-09T15:03:00Z",
        "metrics": {"agent_0/steps": 11, "agent_0/fitness": 0.9},
    },
]


class InferenceEngine(enum.Enum):
    VLLM = "vllm"


class LoggedVersion:
    """Model version: records the service it was asked to create."""

    def __init__(self, services: list[dict[str, object]]) -> None:
        self.services = services
        self.service: dict[str, object] = {}

    def create_service(self, **options: object) -> str:
        self.service = options
        return "created"

    def list_services(self) -> pd.DataFrame:
        return pd.DataFrame(self.services)


@dataclass
class RegistryCalls:
    """What the stand-in Registry was opened with and asked to log."""

    services: list[dict[str, object]]
    opened: dict[str, object] = field(default_factory=dict)
    logged: dict[str, object] = field(default_factory=dict)
    version: LoggedVersion | None = None


class Session:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_tiny_model(model_dir: Path) -> None:
    """A one-layer bf16 GPT-2 and word-level tokenizer saved like a merged export."""
    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "</s>": 3}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="</s>"
    )
    config = transformers.GPT2Config(
        vocab_size=len(vocab), n_positions=16, n_embd=8, n_layer=1, n_head=1
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config).to(torch.bfloat16)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / ".complete").write_text("", encoding="utf-8")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_dir = tmp_path_factory.mktemp("tiny") / "merged"
    write_tiny_model(model_dir)
    return model_dir


@pytest.fixture
def run_dir(tmp_path: Path, tiny_model: Path) -> Path:
    merged = tmp_path / "checkpoint_10" / "merged"
    merged.parent.mkdir()
    merged.symlink_to(tiny_model, target_is_directory=True)
    write_rows(tmp_path / "metrics.jsonl", C1_ROWS)
    return tmp_path


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> RegistryCalls:
    """Install ``snowflake.ml`` modules whose Registry records its calls."""
    calls = RegistryCalls(
        services=[
            {"name": "ARENA.TRAINING.OTHER", "inference_endpoint": "other.app"},
            {"name": TARGET, "inference_endpoint": "abc.snowflakecomputing.app"},
        ]
    )

    class Registry:
        def __init__(self, session: object, **names: object) -> None:
            calls.opened = {"session": session, **names}

        def log_model(self, model: object, **options: object) -> LoggedVersion:
            calls.logged = {"model": model, **options}
            calls.version = LoggedVersion(calls.services)
            return calls.version

    modules = {
        "snowflake": types.ModuleType("snowflake"),
        "snowflake.ml": types.ModuleType("snowflake.ml"),
        "snowflake.ml.registry": types.ModuleType("snowflake.ml.registry"),
        "snowflake.ml.model": types.ModuleType("snowflake.ml.model"),
        "snowflake.ml.model.openai_signatures": types.ModuleType("signatures"),
        "snowflake.ml.model.inference_engine": types.ModuleType("inference_engine"),
    }
    modules["snowflake.ml.registry"].Registry = Registry
    modules[
        "snowflake.ml.model.openai_signatures"
    ].OPENAI_CHAT_SIGNATURE = CHAT_SIGNATURE
    modules["snowflake.ml.model.inference_engine"].InferenceEngine = InferenceEngine
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def write_request(run_dir: Path, **overrides: object) -> Path:
    request: PublishDocument = {
        "checkpoint": "checkpoint_10",
        "compute_pool": "arena_gpu_pool",
        "database": "ARENA",
        "gpu_count": 1,
        "image_build_compute_pool": "arena_head_pool",
        "max_model_len": None,
        "model_name": "arena_policy",
        "run_id": RUN_ID,
        "schema": "TRAINING",
        "service_name": "arena_policy_checkpoint_10",
        "step": 10,
        "version_name": "checkpoint_10",
    }
    path = run_dir / "publish-ARENA_PUBLISH_X.json"
    path.write_text(json.dumps({**request, **overrides}), encoding="utf-8")
    return path


class TestCheckpointMetrics:
    def test_later_rows_are_not_used(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        write_rows(path, C1_ROWS)

        assert checkpoint_metrics(path, 10) == {
            "agent_0/steps": 10,
            "agent_0/fitness": 0.7,
        }

    def test_step_selects_the_row_with_that_step(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        write_rows(path, C1_ROWS)

        assert checkpoint_metrics(path, 9) == {
            "agent_0/steps": 9,
            "agent_0/fitness": 0.5,
        }

    def test_step_without_a_row_has_no_metrics(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        write_rows(path, C1_ROWS)

        assert checkpoint_metrics(path, 12) is None

    def test_run_without_metrics_has_none(self, tmp_path: Path) -> None:
        assert checkpoint_metrics(tmp_path / "metrics.jsonl", 10) is None


class TestLoadTextPipeline:
    def test_loads_in_the_saved_dtype(self, tiny_model: Path) -> None:
        pipeline = load_text_pipeline(tiny_model)

        assert pipeline.task == "text-generation"
        assert pipeline.model.dtype == torch.bfloat16


class TestRunPublish:
    def test_logs_the_checkpoint_and_serves_it(
        self, run_dir: Path, registry: RegistryCalls
    ) -> None:
        # Arrange
        session = Session()
        request = write_request(run_dir, gpu_count=2, max_model_len=4096)

        # Act
        endpoint = run_publish(request, session)

        # Assert
        assert endpoint == "abc.snowflakecomputing.app"
        assert registry.opened == {
            "session": session,
            "database_name": "ARENA",
            "schema_name": "TRAINING",
        }
        logged = dict(registry.logged)
        assert isinstance(logged.pop("model"), transformers.TextGenerationPipeline)
        assert logged == {
            "model_name": "arena_policy",
            "version_name": "checkpoint_10",
            "comment": f"run_id={RUN_ID} checkpoint=checkpoint_10",
            "metrics": {"agent_0/steps": 10, "agent_0/fitness": 0.7},
            "signatures": CHAT_SIGNATURE,
            "target_platforms": ["SNOWPARK_CONTAINER_SERVICES"],
        }
        assert registry.version is not None
        assert registry.version.service == {
            "service_name": "ARENA.TRAINING.arena_policy_checkpoint_10",
            "service_compute_pool": "arena_gpu_pool",
            "image_build_compute_pool": "arena_head_pool",
            "min_instances": 0,
            "ingress_enabled": True,
            "gpu_requests": "2",
            "inference_engine_options": {
                "engine": InferenceEngine.VLLM,
                "engine_args_override": [
                    "--tensor-parallel-size=2",
                    "--max-model-len=4096",
                ],
            },
        }
        assert session.closed

    def test_step_metrics_come_from_that_steps_row(
        self, run_dir: Path, registry: RegistryCalls
    ) -> None:
        run_publish(write_request(run_dir, step=9, service_name=None), Session())

        assert registry.logged["metrics"] == {
            "agent_0/steps": 9,
            "agent_0/fitness": 0.5,
        }

    def test_no_service_name_only_logs(
        self, run_dir: Path, registry: RegistryCalls
    ) -> None:
        session = Session()

        endpoint = run_publish(write_request(run_dir, service_name=None), session)

        assert endpoint is None
        assert registry.version is not None
        assert registry.version.service == {}
        assert session.closed

    def test_incomplete_checkpoint_is_rejected(
        self, run_dir: Path, registry: RegistryCalls
    ) -> None:
        merged = run_dir / "checkpoint_9" / "merged"
        merged.mkdir(parents=True)

        with pytest.raises(ValueError, match=re.escape(f"{merged} has no .complete")):
            run_publish(write_request(run_dir, checkpoint="checkpoint_9"), Session())

        assert registry.logged == {}

    def test_service_missing_from_the_version_is_rejected(
        self, run_dir: Path, registry: RegistryCalls
    ) -> None:
        registry.services.pop()
        session = Session()

        with pytest.raises(
            ValueError,
            match=r"does not list service ARENA\.TRAINING\.arena_policy_checkpoint_10",
        ):
            run_publish(write_request(run_dir), session)

        assert session.closed


class TestMain:
    def test_prints_the_endpoint(
        self,
        run_dir: Path,
        registry: RegistryCalls,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.snowflake.publish_job.open_session", lambda name: Session()
        )

        code = main([str(write_request(run_dir))])

        assert code == 0
        assert capsys.readouterr().out == "endpoint: abc.snowflakecomputing.app\n"

    def test_log_only_prints_nothing(
        self,
        run_dir: Path,
        registry: RegistryCalls,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.snowflake.publish_job.open_session", lambda name: Session()
        )

        code = main([str(write_request(run_dir, service_name=None))])

        assert code == 0
        assert capsys.readouterr().out == ""

    def test_usage_without_a_request_path(self) -> None:
        with pytest.raises(SystemExit, match="usage: python -m"):
            main([])

    def test_module_entrypoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["agilerl.arena.snowflake.publish_job"])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*found in sys.modules.*",
                category=RuntimeWarning,
            )
            with pytest.raises(SystemExit, match="usage:"):
                runpy.run_module(
                    "agilerl.arena.snowflake.publish_job", run_name="__main__"
                )
