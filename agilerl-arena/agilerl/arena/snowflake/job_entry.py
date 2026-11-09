# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""In-job entrypoint: export the source, train, and write results back."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml

from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.snowflake.job import JobDocument, job_id_from_run_id
from agilerl.arena.snowflake.session import open_session
from agilerl.arena.snowflake.source import (
    QuerySource,
    SnowflakeSession,
    TableSource,
    write_source,
)

DATA_DIR = Path("/tmp/arena/data")
CKPT_DIR = Path("/tmp/arena/ckpt")
METRICS_PATH = Path("/tmp/arena/metrics.jsonl")
POLL_INTERVAL_S = 60.0
METRICS_FORMAT_NAME = "ARENA_JSON"
_CHECKPOINT_DIR = re.compile(r"^checkpoint_(\d+)$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


@dataclass
class UploadTracker:
    """Files already copied to the runs stage, and the last metrics snapshot."""

    uploaded: set[str] = field(default_factory=set)
    metrics_signature: tuple[int, int] | None = None


def main(argv: list[str] | None = None) -> int:
    """Run the job described by ``job.json``.

    :param argv: Arguments after the module name. ``None`` reads ``sys.argv``.
    :return: The trainer's exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        msg = "usage: python -m agilerl.arena.snowflake.job_entry job.json"
        raise SystemExit(msg)
    return run_job(Path(args[0]))


def run_job(
    job_path: Path,
    session: SnowflakeSession | None = None,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> int:
    """Export data, run the trainer, and publish checkpoints and metrics.

    :param job_path: ``job.json`` written by the submitter.
    :param session: Snowpark session. ``None`` opens the in-job session.
    :param poll_interval_s: Seconds between checkpoint and metrics uploads.
    :return: The trainer process exit code.
    """
    job = cast("JobDocument", json.loads(job_path.read_text(encoding="utf-8")))
    if session is None:
        session = open_session(None)
    payload = job_path.resolve().parent
    manifest_path = payload / "manifest.yaml"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    write_source(
        session,
        source_from_document(job["source"]),
        manifest_columns(manifest_path),
        DATA_DIR,
    )
    tracker = UploadTracker()
    publish_outputs(
        session,
        job,
        CKPT_DIR,
        manifest_path,
        METRICS_PATH,
        tracker,
        trainer_exited=False,
    )
    process = subprocess.Popen(
        trainer_command(job),
        env=trainer_env(job, os.environ),
        cwd=payload,
    )
    try:
        while True:
            code = process.poll()
            exited = code is not None
            publish_outputs(
                session,
                job,
                CKPT_DIR,
                manifest_path,
                METRICS_PATH,
                tracker,
                trainer_exited=exited,
            )
            if exited:
                return int(code)
            time.sleep(poll_interval_s)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def trainer_command(job: JobDocument) -> list[str]:
    """Argv for the training subprocess.

    :param job: Parsed ``job.json``.
    :return: The command list.
    """
    job_id = job_id_from_run_id(str(job["run_id"]))
    command = [
        "python",
        "-m",
        "agilerl_ray.train",
        "--experiment-id",
        str(job_id),
        "--manifest",
        "manifest.yaml",
        "--custom-dataset-s3",
        str(DATA_DIR),
        "--checkpoint-directory",
        str(CKPT_DIR),
        "--metrics-file",
        str(METRICS_PATH),
        "--num-nodes",
        str(job["nodes"]),
    ]
    if job["checkpoint_steps"] is not None:
        command.extend(["--checkpoint-steps", str(job["checkpoint_steps"])])
    return command


def trainer_env(job: JobDocument, base: Mapping[str, str]) -> dict[str, str]:
    """Environment for the trainer, with this run's Pulsar namespace.

    :param job: Parsed ``job.json``.
    :param base: Environment to copy. ``DISABLE_PULSAR`` and the OTLP endpoint are removed.
    :return: The trainer environment.
    """
    env = dict(base)
    env["RAY_ADDRESS"] = "auto"
    env["APP_PULSAR_URL"] = str(job["pulsar_url"])
    env["APP_TOPIC_PREFIX"] = str(job["pulsar_topic_prefix"])
    env["AGILERL_ROLLOUT_TOPIC_ROUTE"] = rollout_topic_route(
        str(job["pulsar_topic_prefix"])
    )
    env["JOB_ID"] = str(job_id_from_run_id(str(job["run_id"])))
    env.pop("OTLP_COLLECTOR_ENDPOINT", None)
    env.pop("DISABLE_PULSAR", None)
    return env


def rollout_topic_route(topic_prefix: str) -> str:
    """Persistent rollout route beside the metrics topic prefix.

    :param topic_prefix: ``APP_TOPIC_PREFIX`` value for this run.
    :return: ``persistent://<tenant>/<namespace>/llm-rollout``.
    """
    body = topic_prefix
    for scheme in ("non-persistent://", "persistent://"):
        if body.startswith(scheme):
            body = body[len(scheme) :]
            break
    body = body.strip("/")
    suffix = "/llm-rollout"
    body = body.removesuffix(suffix)
    return f"persistent://{body}/llm-rollout"


def source_from_document(document: Mapping[str, str]) -> TableSource | QuerySource:
    """Rebuild the training source from ``job.json``.

    :param document: The ``source`` object.
    :return: A table or query source.
    :raises ValueError: ``kind`` is not ``table`` or ``query``.
    """
    kind = document["kind"]
    if kind == "table":
        return TableSource(document["name"])
    if kind == "query":
        return QuerySource(document["sql"])
    msg = f"Unknown Snowflake source kind {kind!r}"
    raise ValueError(msg)


def manifest_columns(path: Path) -> dict[str, str] | None:
    """Column mapping from the payload manifest.

    :param path: ``manifest.yaml`` next to ``job.json``.
    :return: The environment ``columns`` map, or ``None``.
    """
    manifest = TrainingManifest.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    environment = manifest.environment
    if not isinstance(environment, LLMEnvSpec) or environment.columns is None:
        return None
    return dict(environment.columns)


def ready_checkpoint_files(ckpt_root: Path, trainer_exited: bool) -> list[Path]:
    """Checkpoint files that are finished and safe to upload.

    A ``merged/`` directory is ready when it contains ``.complete``.
    ``checkpoint_<n>/`` is ready once a higher checkpoint exists, or the trainer
    has exited. Merged files wait for ``.complete`` even after that.

    :param ckpt_root: Trainer checkpoint directory.
    :param trainer_exited: Whether the trainer process has finished.
    :return: Files to upload, in path order.
    """
    numbered: dict[int, Path] = {}
    if ckpt_root.is_dir():
        for child in ckpt_root.iterdir():
            match = _CHECKPOINT_DIR.fullmatch(child.name)
            if match and child.is_dir():
                numbered[int(match.group(1))] = child
    if not numbered:
        return []
    highest = max(numbered)
    ready: list[Path] = []
    for number, directory in sorted(numbered.items()):
        directory_done = trainer_exited or number < highest
        merged_done = (directory / "merged" / ".complete").is_file()
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            in_merged = "merged" in path.relative_to(directory).parts
            if in_merged:
                if merged_done:
                    ready.append(path)
            elif directory_done:
                ready.append(path)
    return ready


def publish_outputs(
    session: SnowflakeSession,
    job: JobDocument,
    ckpt_root: Path,
    manifest_path: Path,
    metrics_path: Path,
    tracker: UploadTracker,
    trainer_exited: bool,
) -> None:
    """Upload new checkpoints and merge new metric rows.

    :param session: Snowpark session.
    :param job: Parsed ``job.json``.
    :param ckpt_root: Local checkpoint directory.
    :param manifest_path: Manifest uploaded once at the start of the run.
    :param metrics_path: JSONL metrics file. Re-uploaded when its size changes.
    :param tracker: Paths already uploaded.
    :param trainer_exited: Whether the trainer process has finished.
    """
    run_id = str(job["run_id"])
    runs_stage = str(job["runs_stage"])
    _put_once(session, tracker, manifest_path, runs_stage, run_id, "manifest.yaml")
    for path in ready_checkpoint_files(ckpt_root, trainer_exited):
        relative = path.relative_to(ckpt_root).as_posix()
        _put_once(session, tracker, path, runs_stage, run_id, relative)
    if not metrics_path.is_file():
        return
    signature = (metrics_path.stat().st_size, metrics_path.stat().st_mtime_ns)
    if signature == tracker.metrics_signature:
        return
    _put_file(
        session, metrics_path, stage_directory(runs_stage, run_id, "metrics.jsonl")
    )
    session.sql(
        metrics_merge_sql(run_id, str(job["metrics_table"]), runs_stage)
    ).collect()
    tracker.metrics_signature = signature


def metrics_merge_sql(run_id: str, metrics_table: str, runs_stage: str) -> str:
    """MERGE new JSONL rows into the metrics table on ``(RUN_ID, SEQ)``.

    :param run_id: Run id, also the stage prefix.
    :param metrics_table: Fully qualified or bare metrics table.
    :param runs_stage: Stage that holds ``metrics.jsonl``.
    :return: The MERGE statement.
    """
    job_id_from_run_id(run_id)
    table = snowflake_identifier(metrics_table, "table")
    stage = stage_directory(runs_stage, run_id, "metrics.jsonl")
    file_format = metrics_file_format(table)
    return (
        f"MERGE INTO {table} AS t "
        "USING ("
        "SELECT "
        f"'{run_id}' AS RUN_ID, "
        "$1:seq::NUMBER AS SEQ, "
        "$1:time::TIMESTAMP_TZ AS TIME, "
        "$1:metrics AS METRICS "
        f"FROM {stage}/metrics.jsonl (FILE_FORMAT => '{file_format}')"
        ") AS s "
        "ON t.RUN_ID = s.RUN_ID AND t.SEQ = s.SEQ "
        "WHEN NOT MATCHED THEN INSERT (RUN_ID, SEQ, TIME, METRICS) "
        "VALUES (s.RUN_ID, s.SEQ, s.TIME, s.METRICS)"
    )


def metrics_file_format(metrics_table: str) -> str:
    """JSON file format that sits next to the metrics table.

    :param metrics_table: Validated table identifier.
    :return: ``<schema>.ARENA_JSON``, or ``ARENA_JSON`` for a bare table name.
    """
    parts = metrics_table.split(".")
    if len(parts) == 1:
        return METRICS_FORMAT_NAME
    return ".".join([*parts[:-1], METRICS_FORMAT_NAME])


def stage_directory(runs_stage: str, run_id: str, relative: str) -> str:
    """Stage URL of the directory that should hold ``relative``.

    :param runs_stage: Stage name, with or without a leading ``@``.
    :param run_id: Run id directory.
    :param relative: Path under the run directory.
    :return: ``@stage/<run_id>/<parent>``.
    """
    stage = snowflake_identifier(runs_stage, "stage")
    parent = Path(relative).parent.as_posix()
    if parent == ".":
        return f"{stage}/{run_id}"
    return f"{stage}/{run_id}/{parent}"


def snowflake_identifier(value: str, kind: str) -> str:
    """Validate an unquoted Snowflake identifier.

    :param value: Table or stage name. A stage may start with ``@``.
    :param kind: ``table`` or ``stage``, used in the error.
    :return: The identifier. Stages are returned with a leading ``@``.
    :raises ValueError: The name is empty or quoted or contains other characters.
    """
    text = value.strip()
    if kind == "stage" and text.startswith("@"):
        text = text[1:]
    parts = text.split(".")
    if not text or any(_IDENTIFIER.fullmatch(part) is None for part in parts):
        msg = f"{kind} must be a Snowflake identifier, got {value!r}"
        raise ValueError(msg)
    if kind == "stage":
        return f"@{text}"
    return text


def _put_once(
    session: SnowflakeSession,
    tracker: UploadTracker,
    local: Path,
    runs_stage: str,
    run_id: str,
    relative: str,
) -> None:
    if relative in tracker.uploaded or not local.is_file():
        return
    _put_file(session, local, stage_directory(runs_stage, run_id, relative))
    tracker.uploaded.add(relative)


def _put_file(session: SnowflakeSession, local: Path, stage: str) -> None:
    session.file.put(
        str(local),
        stage,
        auto_compress=False,
        overwrite=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
