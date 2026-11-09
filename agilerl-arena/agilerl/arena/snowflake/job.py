# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Submit a training manifest as a Snowflake ML Job."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypedDict, cast

import yaml

from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.snowflake.source import QuerySource, SnowflakeSession, TableSource

RUN_ID_RE = re.compile(r"^arena-(\d{14})-([0-9a-f]{6})$")
ENTRYPOINT = [
    "python",
    "-m",
    "agilerl.arena.snowflake.job_entry",
    "job.json",
]


class JobDocument(TypedDict):
    """JSON payload the in-job entrypoint reads."""

    checkpoint_steps: int | None
    compute_pool: str
    external_access_integrations: list[str]
    metrics_table: str
    nodes: int
    payload_stage: str
    pulsar_topic_prefix: str
    pulsar_url: str
    run_id: str
    runtime_environment: str
    runs_stage: str
    source: dict[str, str]


class SubmittedJob(Protocol):
    """The id field ``submit_directory`` returns."""

    id: str | None


@dataclass(frozen=True)
class SubmitSettings:
    """What one ``arena snowflake submit`` sends with the manifest."""

    compute_pool: str
    runtime_environment: str
    payload_stage: str
    runs_stage: str
    metrics_table: str
    source: TableSource | QuerySource
    nodes: int
    external_access_integrations: tuple[str, ...]
    checkpoint_steps: int | None
    pulsar_url: str
    pulsar_topic_prefix: str


def new_run_id(now: datetime | None = None) -> str:
    """Build ``arena-<UTC yyyymmddHHMMSS>-<6 lowercase hex>``.

    :param now: Clock to stamp. ``None`` uses the current UTC time.
    :return: The run id.
    """
    moment = datetime.now(timezone.utc) if now is None else now
    return f"arena-{moment.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"


def job_id_from_run_id(run_id: str) -> int:
    """Integer job id taken from the run id's hex suffix.

    :param run_id: An id from :func:`new_run_id`.
    :return: The suffix as an integer.
    :raises ValueError: ``run_id`` does not match the run id pattern.
    """
    match = RUN_ID_RE.fullmatch(run_id)
    if match is None:
        msg = f"run_id must look like arena-<yyyymmddHHMMSS>-<6 hex>, got {run_id!r}"
        raise ValueError(msg)
    return int(match.group(2), 16)


def load_manifest(path: Path) -> TrainingManifest:
    """Parse a training manifest.

    :param path: YAML file.
    :return: The validated manifest.
    """
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return TrainingManifest.model_validate(document)


def validate_manifest(manifest: TrainingManifest) -> None:
    """Reject manifests a Snowflake ML Job cannot run, and warn on publish limits.

    :param manifest: Parsed training manifest.
    :raises ValueError: The manifest is not an LLM run, or it starts an env image.
    """
    environment = manifest.environment
    if not isinstance(environment, LLMEnvSpec):
        msg = "Snowflake submit runs SFT, DPO, and RL manifests."
        raise ValueError(msg)
    if environment.env_image is not None:
        msg = (
            "env_image starts a Kubernetes Pod. A Snowflake ML Job cannot "
            "run it. Set env_url to a service the job can reach, and add that "
            "host to an External Access Integration."
        )
        raise ValueError(msg)
    if (
        environment.env_type == "rollout"
        and environment.env_url is None
        and not environment.dataset_backed_rollout
    ):
        msg = (
            "A rollout inside the job needs env_url. Add the host to an "
            "External Access Integration and pass it with --eai."
        )
        raise ValueError(msg)
    if manifest.training.effective_checkpoint_export().format != "merged":
        warnings.warn(
            "training.checkpoint_export is not merged. "
            "Publishing the checkpoint needs merged/.",
            stacklevel=2,
        )
    if environment.env_url is not None:
        warnings.warn(
            "env_url must be reachable from the job. Add its host to an "
            "External Access Integration and pass that integration with --eai.",
            stacklevel=2,
        )


def submit_job(
    session: SnowflakeSession,
    manifest_path: Path,
    settings: SubmitSettings,
    run_id: str | None = None,
) -> tuple[str, str]:
    """Upload the manifest and submit the ML Job.

    :param session: Snowpark session.
    :param manifest_path: Local manifest YAML copied into the payload.
    :param settings: Pool, stages, source, and Pulsar settings.
    :param run_id: Run id to use. ``None`` generates one.
    :return: ``(run_id, job_id)``.
    """
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    chosen = new_run_id() if run_id is None else run_id
    job_id_from_run_id(chosen)
    with tempfile.TemporaryDirectory() as tmp:
        payload = Path(tmp)
        shutil.copyfile(manifest_path, payload / "manifest.yaml")
        document = job_document(settings, chosen)
        (payload / "job.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        submitted = _submit_directory(payload, settings, session)
    job_id = submitted.id
    if job_id is None or job_id == "":
        msg = "Snowflake ML Job did not return an id"
        raise RuntimeError(msg)
    return chosen, str(job_id)


def job_document(settings: SubmitSettings, run_id: str) -> JobDocument:
    """JSON payload the in-job entrypoint reads.

    :param settings: Submit settings.
    :param run_id: Run id for this payload.
    :return: A JSON-ready document.
    """
    return {
        "checkpoint_steps": settings.checkpoint_steps,
        "compute_pool": settings.compute_pool,
        "external_access_integrations": list(settings.external_access_integrations),
        "metrics_table": settings.metrics_table,
        "nodes": settings.nodes,
        "payload_stage": settings.payload_stage,
        "pulsar_topic_prefix": settings.pulsar_topic_prefix,
        "pulsar_url": settings.pulsar_url,
        "run_id": run_id,
        "runtime_environment": settings.runtime_environment,
        "runs_stage": settings.runs_stage,
        "source": _source_document(settings.source),
    }


def _source_document(source: TableSource | QuerySource) -> dict[str, str]:
    if isinstance(source, TableSource):
        return {"kind": "table", "name": source.name}
    return {"kind": "query", "sql": source.sql}


def _submit_directory(
    payload: Path, settings: SubmitSettings, session: SnowflakeSession
) -> SubmittedJob:
    # Deferred optional extra: snowflake.
    try:
        from snowflake.ml.jobs import submit_directory
    except ImportError as exc:
        msg = (
            "Snowflake support needs the snowflake extra: "
            "pip install 'agilerl-arena[snowflake]'"
        )
        raise ImportError(msg) from exc
    integrations = list(settings.external_access_integrations)
    return cast(
        "SubmittedJob",
        submit_directory(
            str(payload),
            settings.compute_pool,
            entrypoint=list(ENTRYPOINT),
            stage_name=settings.payload_stage,
            target_instances=settings.nodes,
            external_access_integrations=integrations or None,
            session=session,
            runtime_environment=settings.runtime_environment,
        ),
    )
