# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Head entrypoint: export the source, train, and merge metrics into Snowflake."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

import yaml

from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.snowflake.export import write_source
from agilerl.arena.snowflake.job import (
    JobDocument,
    job_id_from_run_id,
    put_file,
    runs_prefix,
    snowflake_identifier,
    stage_directory,
)
from agilerl.arena.snowflake.session import open_session, transient_errors
from agilerl.arena.snowflake.source import QuerySource, SnowflakeSession, TableSource
from agilerl.arena.snowflake.upload import CKPT_DIR, upload_with_retries

# Stage mounts reject the rename ``dataset/`` writes under the trainer cwd.
WORK_DIR = Path("/tmp/arena/work")
METRICS_PATH = Path("/tmp/arena/metrics.jsonl")
POLL_INTERVAL_S = 60.0


@dataclass
class MetricsPublisher:
    """Merges complete lines of the trainer's metrics file into the metrics table."""

    session: SnowflakeSession
    job: JobDocument
    metrics_path: Path
    scratch: Path
    retry_on: tuple[type[Exception], ...]
    sleep: Callable[[float], None] = time.sleep
    published_bytes: int = 0

    def publish(self, final: bool) -> bool:
        """PUT and MERGE the metrics file up to its last newline.

        :param final: The trainer has exited. Failure raises instead of waiting a pass.
        :return: Whether the pass succeeded.
        """
        if not self.metrics_path.is_file():
            return True
        data = self.metrics_path.read_bytes()
        complete = data[: data.rfind(b"\n") + 1]
        if len(complete) == self.published_bytes:
            return True
        copy = self.scratch / "metrics.jsonl"
        copy.write_bytes(complete)
        published = upload_with_retries(
            partial(self._merge, copy), final, self.sleep, self.retry_on
        )
        if published:
            self.published_bytes = len(complete)
        return published

    def _merge(self, copy: Path) -> None:
        run_id = str(self.job["run_id"])
        runs_stage = str(self.job["runs_stage"])
        put_file(
            self.session, copy, stage_directory(runs_stage, run_id, "metrics.jsonl")
        )
        self.session.sql(
            metrics_merge_sql(run_id, str(self.job["metrics_table"]), runs_stage)
        ).collect()


def main(argv: list[str] | None = None) -> int:
    """Run the job described by ``job.json``.

    :param argv: Arguments after the module name. ``None`` reads ``sys.argv``.
    :return: The trainer's exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        msg = "usage: python -m agilerl.arena.snowflake.job_entry /arena/run/job.json"
        raise SystemExit(msg)
    retry_on = transient_errors()
    session = open_session(None)
    try:
        return run_job(Path(args[0]), session, retry_on)
    finally:
        session.close()


def run_job(
    job_path: Path,
    session: SnowflakeSession,
    retry_on: tuple[type[Exception], ...],
    poll_interval_s: float = POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """PUT the exported source, if any, to the run stage, train, and merge metrics.

    :param job_path: ``job.json`` in the run directory every instance mounts.
    :param session: Snowpark session.
    :param retry_on: Errors a stage upload retries.
    :param poll_interval_s: Seconds between metrics merges.
    :param sleep: Called between polls and upload retries.
    :return: The trainer process exit code.
    """
    job = cast("JobDocument", json.loads(job_path.read_text(encoding="utf-8")))
    run_dir = job_path.resolve().parent
    manifest_path = run_dir / "manifest.yaml"
    if job["source"] is not None:
        with tempfile.TemporaryDirectory() as tmp:
            shards = write_source(
                session,
                source_from_document(job["source"]),
                manifest_columns(manifest_path),
                Path(tmp),
            )
            upload_with_retries(
                partial(put_shards, session, shards, job), True, sleep, retry_on
            )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, WORK_DIR / "manifest.yaml")
    metrics = MetricsPublisher(
        session=session,
        job=job,
        metrics_path=METRICS_PATH,
        scratch=WORK_DIR,
        retry_on=retry_on,
        sleep=sleep,
    )
    process = subprocess.Popen(
        trainer_command(job, run_dir),
        env=trainer_env(os.environ),
        cwd=WORK_DIR,
    )
    try:
        while True:
            code = process.poll()
            exited = code is not None
            metrics.publish(final=exited)
            if exited:
                return int(code)
            sleep(poll_interval_s)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def put_shards(session: SnowflakeSession, shards: Path, job: JobDocument) -> None:
    """PUT every parquet shard to the run's ``data/`` stage directory.

    Each PUT returns once the file is on the stage, so every instance's mount
    can read it.

    :param session: Snowpark session.
    :param shards: Directory of ``shard-NNNNN.parquet`` files.
    :param job: Parsed ``job.json``.
    """
    stage = f"{runs_prefix(str(job['runs_stage']), str(job['run_id']))}/data"
    for shard in sorted(shards.glob("*.parquet")):
        put_file(session, shard, stage)


def trainer_command(job: JobDocument, run_dir: Path) -> list[str]:
    """Argv for the training subprocess.

    :param job: Parsed ``job.json``.
    :param run_dir: Run directory on the stage mount.
    :return: The command list.
    """
    command = [
        sys.executable,
        "-m",
        "agilerl_ray.train",
        "--experiment-id",
        str(job_id_from_run_id(str(job["run_id"]))),
        "--manifest",
        "manifest.yaml",
        "--checkpoint-directory",
        str(CKPT_DIR),
        "--metrics-file",
        str(METRICS_PATH),
        "--num-nodes",
        str(job["nodes"]),
    ]
    if job["source"] is not None:
        command.extend(["--custom-dataset-s3", str(run_dir / "data")])
    if job["reward_file"]:
        command.extend(["--custom-reward-s3", str(run_dir / "reward.py")])
    if job["checkpoint_steps"] is not None:
        command.extend(["--checkpoint-steps", str(job["checkpoint_steps"])])
    return command


def trainer_env(base: Mapping[str, str]) -> dict[str, str]:
    """Environment for the trainer driver.

    :param base: Environment to copy. The OTLP endpoint is removed.
    :return: The trainer environment.
    """
    env = dict(base)
    env["RAY_ADDRESS"] = "auto"
    env.pop("OTLP_COLLECTOR_ENDPOINT", None)
    return env


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


def metrics_merge_sql(run_id: str, metrics_table: str, runs_stage: str) -> str:
    """MERGE new JSONL rows into the metrics table on ``(RUN_ID, SEQ)``.

    :param run_id: Run id, also the stage prefix.
    :param metrics_table: Fully qualified or bare metrics table.
    :param runs_stage: Stage that holds ``metrics.jsonl``.
    :return: The MERGE statement.
    """
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
        return "ARENA_JSON"
    return ".".join([*parts[:-1], "ARENA_JSON"])


if __name__ == "__main__":
    raise SystemExit(main())
