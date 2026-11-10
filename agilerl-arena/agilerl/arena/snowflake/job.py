# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Submit a training manifest as a Snowflake job service."""

from __future__ import annotations

import json
import math
import re
import secrets
import shutil
import tempfile
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import yaml

from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.snowflake.ports import TrainerPorts, service_endpoints
from agilerl.arena.snowflake.source import (
    QuerySource,
    SnowflakeRow,
    SnowflakeSession,
    TableSource,
)

RUN_ID_RE = re.compile(r"^arena-(\d{14})-([0-9a-f]{6})$")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
CONTAINER_NAME = "main"
# The stage volume mounts @<runs stage>/<run_id> here on every instance.
RUN_MOUNT = "/arena/run"
JOB_JSON_PATH = f"{RUN_MOUNT}/job.json"
# The spec command replaces the image entrypoint, so tini stays PID 1 to
# forward SIGTERM and reap Ray's child processes.
NODE_COMMAND = ["tini", "--", "python", "-m", "agilerl.arena.snowflake.node"]
CONTAINER_ARGS = [
    "python",
    "-m",
    "agilerl.arena.snowflake.job_entry",
    JOB_JSON_PATH,
]

# Host memory (GiB) and GPU count from Snowflake's AWS and Azure GPU families.
GPU_NODE_RESOURCES: dict[str, tuple[int, int]] = {
    "GPU_NV_S": (1, 28),
    "GPU_NV_M": (4, 178),
    "GPU_NV_L": (8, 1112),
    "GPU_L40S_G1_8": (1, 58),
    "GPU_L40S_G1_16": (1, 116),
    "GPU_L40S_G1_48": (4, 368),
    "GPU_L40S_G1_192": (8, 1436),
    "GPU_R6K_G1_8": (1, 58),
    "GPU_R6K_G1_16": (1, 116),
    "GPU_R6K_G1_32": (1, 240),
    "GPU_R6K_G1_48": (2, 490),
    "GPU_R6K_G1_96": (4, 984),
    "GPU_R6K_G1_192": (8, 1843),
    "GPU_NV_XS": (1, 26),
    "GPU_NV_SM": (1, 424),
    "GPU_NV_2M": (2, 858),
    "GPU_NV_3M": (2, 424),
    "GPU_NV_SL": (4, 858),
}


class JobDocument(TypedDict):
    """JSON payload the in-job entrypoint reads."""

    checkpoint_steps: int | None
    compute_pool: str
    external_access_integrations: list[str]
    image: str
    metrics_table: str
    nodes: int
    pulsar_topic_prefix: str
    pulsar_url: str
    reward_file: bool
    run_id: str
    runs_stage: str
    source: dict[str, str] | None
    warehouse: str


@dataclass(frozen=True)
class NodeResources:
    """GPUs and host memory of one compute pool node."""

    gpus: int
    memory_gib: int


@dataclass(frozen=True)
class SubmitSettings:
    """What one ``arena snowflake submit`` sends with the manifest."""

    compute_pool: str
    image: str
    warehouse: str
    runs_stage: str
    metrics_table: str
    source: TableSource | QuerySource | None
    nodes: int
    external_access_integrations: tuple[str, ...]
    checkpoint_steps: int | None
    pulsar_url: str
    pulsar_topic_prefix: str
    reward_file: Path | None


def new_run_id(now: datetime | None = None) -> str:
    """Build ``arena-<UTC yyyymmddHHMMSS>-<6 lowercase hex>``.

    :param now: Clock to stamp. ``None`` uses the current UTC time.
    :return: The run id.
    """
    moment = datetime.now(timezone.utc) if now is None else now
    return f"arena-{moment.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"


def service_name(run_id: str) -> str:
    """Service name ``ARENA_<yyyymmddHHMMSS>_<HEX>`` for a run id.

    :param run_id: An id from :func:`new_run_id`.
    :return: An unquoted Snowflake identifier.
    :raises ValueError: ``run_id`` does not match the run id pattern.
    """
    match = _run_id_match(run_id)
    return f"ARENA_{match.group(1)}_{match.group(2).upper()}"


def job_id_from_run_id(run_id: str) -> int:
    """Integer job id taken from the run id's hex suffix.

    :param run_id: An id from :func:`new_run_id`.
    :return: The suffix as an integer.
    :raises ValueError: ``run_id`` does not match the run id pattern.
    """
    match = _run_id_match(run_id)
    return int(match.group(2), 16)


def load_manifest(path: Path) -> TrainingManifest:
    """Parse a training manifest.

    :param path: YAML file.
    :return: The validated manifest.
    """
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return TrainingManifest.model_validate(document)


def validate_manifest(
    manifest: TrainingManifest, has_reward_file: bool, has_source: bool
) -> bool:
    """Reject manifests a job service cannot run, and warn on publish limits.

    :param manifest: Parsed training manifest.
    :param has_reward_file: Whether ``--reward-file`` was passed.
    :param has_source: Whether ``--source-table`` or ``--source-query`` was passed.
    :return: Whether the job exports the source. An env-backed rollout does not.
    :raises ValueError: The manifest is not an LLM run, starts an env image,
        has no env to roll out against, needs a source that was not passed,
        or the reward file does not fit it.
    """
    environment = manifest.environment
    if not isinstance(environment, LLMEnvSpec):
        msg = "Snowflake submit runs SFT, DPO, and RL manifests."
        raise ValueError(msg)
    if environment.env_image is not None:
        msg = (
            "env_image starts a Kubernetes Pod. A Snowflake job service cannot "
            "run it. Set env_url to a service the job can reach."
        )
        raise ValueError(msg)
    rollout = environment.env_type == "rollout"
    if (
        rollout
        and environment.env_url is None
        and not environment.dataset_backed_rollout
    ):
        msg = "A rollout inside the job needs env_url or a dataset."
        raise ValueError(msg)
    if rollout and environment.dataset_backed_rollout and not has_reward_file:
        msg = "A dataset-backed rollout needs --reward-file."
        raise ValueError(msg)
    if not rollout and has_reward_file:
        msg = "--reward-file is only used by RL manifests."
        raise ValueError(msg)
    # The trainer injects no dataset under a rollout that names its own env.
    exports_source = not rollout or (
        environment.entrypoint is None and environment.env_url is None
    )
    if not exports_source and has_source:
        warnings.warn(
            "The rollout env serves the prompts, so the trainer reads no "
            "dataset. The Snowflake source is not exported.",
            stacklevel=2,
        )
    if exports_source and not has_source:
        msg = "This manifest trains on rows. Pass --source-table or --source-query."
        raise ValueError(msg)
    if manifest.training.effective_checkpoint_export().format != "merged":
        warnings.warn(
            "training.checkpoint_export is not merged. "
            "Publishing the checkpoint needs merged/.",
            stacklevel=2,
        )
    if environment.env_url is not None:
        warnings.warn(
            "env_url must be reachable from the job. A host outside the "
            "account needs an External Access Integration passed with --eai. "
            "An SPCS service in the same account needs none.",
            stacklevel=2,
        )
    return exports_source


def trainer_ports(
    manifest: TrainingManifest, resources: NodeResources, nodes: int
) -> TrainerPorts:
    """Population size and the most rollout engines one member can start.

    :param manifest: Parsed training manifest.
    :param resources: One pool node.
    :param nodes: Job instances.
    :return: The bound the spec opens ports for. ``auto`` engines count every GPU.
    """
    training = manifest.training
    engines = training.rollout_engines_per_agent
    if (
        engines == "auto"
        or "rollout_engines_per_agent" not in training.model_fields_set
    ):
        engines = resources.gpus * nodes
    return TrainerPorts(agents=training.pop_size, engines_per_agent=max(1, engines))


def submit_job(
    session: SnowflakeSession,
    manifest_path: Path,
    settings: SubmitSettings,
    run_id: str | None = None,
) -> tuple[str, str]:
    """Upload the run files and start the job service.

    :param session: Snowpark session.
    :param manifest_path: Local manifest YAML copied onto the runs stage.
    :param settings: Pool, image, stage, source, and Pulsar settings.
    :param run_id: Run id to use. ``None`` generates one.
    :return: ``(run_id, service_name)``.
    """
    manifest = load_manifest(manifest_path)
    exports_source = validate_manifest(
        manifest, settings.reward_file is not None, settings.source is not None
    )
    if not exports_source:
        settings = replace(settings, source=None)
    chosen = new_run_id() if run_id is None else run_id
    name = service_name(chosen)
    resources = node_resources(describe_instance_family(session, settings.compute_pool))
    spec = render_spec(
        settings, chosen, resources, trainer_ports(manifest, resources, settings.nodes)
    )
    statement = job_service_sql(settings, name, spec)
    prefix = runs_prefix(settings.runs_stage, chosen)
    with tempfile.TemporaryDirectory() as tmp:
        payload = Path(tmp)
        shutil.copyfile(manifest_path, payload / "manifest.yaml")
        (payload / "job.json").write_text(
            json.dumps(job_document(settings, chosen), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        uploads = ["manifest.yaml", "job.json"]
        if settings.reward_file is not None:
            shutil.copyfile(settings.reward_file, payload / "reward.py")
            uploads.append("reward.py")
        for upload in uploads:
            put_file(session, payload / upload, prefix)
    session.sql(statement).collect()
    return chosen, name


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
        "image": settings.image,
        "metrics_table": settings.metrics_table,
        "nodes": settings.nodes,
        "pulsar_topic_prefix": settings.pulsar_topic_prefix,
        "pulsar_url": settings.pulsar_url,
        "reward_file": settings.reward_file is not None,
        "run_id": run_id,
        "runs_stage": settings.runs_stage,
        "source": None
        if settings.source is None
        else _source_document(settings.source),
        "warehouse": settings.warehouse,
    }


def node_resources(family: str) -> NodeResources:
    """GPUs and host memory for a documented Snowflake GPU family.

    :param family: ``INSTANCE_FAMILY`` from ``DESCRIBE COMPUTE POOL``.
    :return: The resources one node of that family provides.
    :raises ValueError: The family is not a documented GPU family.
    """
    found = GPU_NODE_RESOURCES.get(family)
    if found is None:
        msg = (
            f"Unknown compute pool instance family {family!r}. "
            "Arena jobs run on a Snowflake GPU instance family."
        )
        raise ValueError(msg)
    gpus, memory_gib = found
    return NodeResources(gpus=gpus, memory_gib=memory_gib)


def shm_gib(memory_gib: int) -> int:
    """Size of the ``/dev/shm`` memory volume, 30% of node memory.

    :param memory_gib: Node host memory in GiB.
    :return: GiB to request for the memory volume.
    """
    return min(math.ceil(memory_gib * 0.3), memory_gib)


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
    body = body.strip("/").removesuffix("/llm-rollout")
    return f"persistent://{body}/llm-rollout"


def container_env(settings: SubmitSettings, run_id: str) -> dict[str, str]:
    """Container environment Ray passes from each raylet to its actors.

    :param settings: Pulsar broker and topic prefix.
    :param run_id: Run id that sets ``JOB_ID``.
    :return: Variable name to value.
    """
    return {
        "APP_PULSAR_URL": settings.pulsar_url,
        "APP_TOPIC_PREFIX": settings.pulsar_topic_prefix,
        "AGILERL_ROLLOUT_TOPIC_ROUTE": rollout_topic_route(
            settings.pulsar_topic_prefix
        ),
        "JOB_ID": str(job_id_from_run_id(run_id)),
    }


def render_spec(
    settings: SubmitSettings,
    run_id: str,
    resources: NodeResources,
    trainer: TrainerPorts,
) -> str:
    """Service specification YAML for ``EXECUTE JOB SERVICE``.

    :param settings: Submit settings.
    :param run_id: Run directory on the stage.
    :param resources: GPUs and memory requested from the pool node.
    :param trainer: Population and engine bound the endpoints cover.
    :return: YAML text.
    :raises ValueError: The YAML would break the ``$$`` SQL quote.
    """
    # The Stage Mount V2 sidecar needs 0.5 GiB of the node.
    memory = f"{resources.memory_gib - 0.5:g}Gi"
    spec = {
        "spec": {
            "containers": [
                {
                    "name": CONTAINER_NAME,
                    "image": settings.image,
                    "command": list(NODE_COMMAND),
                    "args": list(CONTAINER_ARGS),
                    "env": container_env(settings, run_id),
                    "resources": {
                        "requests": {
                            "nvidia.com/gpu": resources.gpus,
                            "memory": memory,
                        },
                        "limits": {
                            "nvidia.com/gpu": resources.gpus,
                            "memory": memory,
                        },
                    },
                    "volumeMounts": [
                        {"name": "run", "mountPath": RUN_MOUNT},
                        {"name": "dshm", "mountPath": "/dev/shm"},
                    ],
                }
            ],
            "endpoints": service_endpoints(trainer),
            "volumes": [
                {
                    "name": "run",
                    "source": "stage",
                    "stageConfig": {
                        "name": runs_prefix(settings.runs_stage, run_id),
                    },
                },
                {
                    "name": "dshm",
                    "source": "memory",
                    "size": f"{shm_gib(resources.memory_gib)}Gi",
                },
            ],
        }
    }
    text = yaml.safe_dump(spec, sort_keys=False)
    if "$$" in text:
        msg = "service spec cannot contain $$"
        raise ValueError(msg)
    return text


def job_service_sql(settings: SubmitSettings, name: str, spec: str) -> str:
    """``EXECUTE JOB SERVICE`` statement for one run.

    :param settings: Pool, warehouse, replica count, and integrations.
    :param name: Service identifier.
    :param spec: Specification YAML placed inside ``$$``.
    :return: The SQL statement.
    """
    pool = snowflake_identifier(settings.compute_pool, "compute pool")
    warehouse = snowflake_identifier(settings.warehouse, "warehouse")
    service = snowflake_identifier(name, "service")
    lines = [
        f"EXECUTE JOB SERVICE IN COMPUTE POOL {pool}",
        f"NAME = {service}",
        "ASYNC = TRUE",
        f"REPLICAS = {settings.nodes}",
        f"QUERY_WAREHOUSE = {warehouse}",
    ]
    if settings.external_access_integrations:
        names = ", ".join(
            snowflake_identifier(item, "external access integration")
            for item in settings.external_access_integrations
        )
        lines.append(f"EXTERNAL_ACCESS_INTEGRATIONS = ({names})")
    lines.append(f"FROM SPECIFICATION $$\n{spec}$$")
    return "\n".join(lines)


def service_status_sql(service: str) -> str:
    """``SHOW SERVICE CONTAINERS`` for one job service.

    :param service: Service name.
    :return: The SQL statement.
    """
    name = snowflake_identifier(service, "service")
    return f"SHOW SERVICE CONTAINERS IN SERVICE {name}"


def service_logs_sql(service: str, instance: int) -> str:
    """``SYSTEM$GET_SERVICE_LOGS`` for the ``main`` container.

    :param service: Service name.
    :param instance: Instance id, starting at 0.
    :return: The SQL statement.
    """
    name = snowflake_identifier(service, "service")
    return f"SELECT SYSTEM$GET_SERVICE_LOGS('{name}', {instance}, '{CONTAINER_NAME}')"


def format_status_rows(rows: list[SnowflakeRow]) -> str:
    """One line per container, instance, container name, and status, then the job state.

    :param rows: ``SHOW SERVICE CONTAINERS`` rows.
    :return: The lines, ending with ``state: <running|done|failed>``.
    """
    lines = []
    service_statuses = set()
    for row in rows:
        values = lowered(row)
        service_statuses.add(str(values["service_status"]).upper())
        lines.append(
            f"{values.get('instance_id', '')}\t"
            f"{values.get('container_name', '')}\t"
            f"{values.get('status', '')}"
        )
    lines.append(f"state: {job_state(service_statuses)}")
    return "\n".join(lines)


def job_state(service_statuses: set[str]) -> str:
    """Job state from the job service's own status.

    The service status turns ``DONE`` only after every replica has exited 0,
    so instances that are not listed yet cannot make the job look finished.

    :param service_statuses: Upper-case ``service_status`` values of the rows.
    :return: ``failed`` for ``FAILED`` or ``INTERNAL_ERROR``, ``done`` for
        ``DONE``, else ``running``.
    """
    if service_statuses & {"FAILED", "INTERNAL_ERROR"}:
        return "failed"
    if service_statuses == {"DONE"}:
        return "done"
    return "running"


def log_text(rows: list[SnowflakeRow]) -> str:
    """First column of the first logs result row.

    :param rows: ``SYSTEM$GET_SERVICE_LOGS`` rows.
    :return: The log text, or an empty string.
    """
    if not rows:
        return ""
    values = list(rows[0].as_dict().values())
    if not values:
        return ""
    return str(values[0])


def describe_instance_family(session: SnowflakeSession, compute_pool: str) -> str:
    """Read ``instance_family`` from ``DESCRIBE COMPUTE POOL``.

    :param session: Snowpark session.
    :param compute_pool: Pool name.
    :return: The instance family.
    :raises ValueError: The pool does not return an instance family.
    """
    name = snowflake_identifier(compute_pool, "compute pool")
    rows = session.sql(f"DESCRIBE COMPUTE POOL {name}").collect()
    if not rows:
        msg = f"Compute pool {compute_pool!r} was not found"
        raise ValueError(msg)
    family = lowered(rows[0]).get("instance_family")
    if not isinstance(family, str) or family.strip() == "":
        msg = f"DESCRIBE COMPUTE POOL {compute_pool!r} did not return instance_family"
        raise ValueError(msg)
    return family


def runs_prefix(runs_stage: str, run_id: str) -> str:
    """Stage URL of this run's directory.

    :param runs_stage: Stage name, with or without a leading ``@``.
    :param run_id: Run id directory.
    :return: ``@stage/<run_id>``.
    """
    _run_id_match(run_id)
    text = runs_stage.strip().removeprefix("@")
    _dotted_identifier(text, "stage", runs_stage)
    return f"@{text}/{run_id}"


def stage_directory(runs_stage: str, run_id: str, relative: str) -> str:
    """Stage URL of the directory that should hold ``relative``.

    :param runs_stage: Stage name, with or without a leading ``@``.
    :param run_id: Run id directory.
    :param relative: Path under the run directory.
    :return: ``@stage/<run_id>/<parent>``.
    """
    prefix = runs_prefix(runs_stage, run_id)
    parent = Path(relative).parent.as_posix()
    if parent == ".":
        return prefix
    return f"{prefix}/{parent}"


def snowflake_identifier(value: str, kind: str) -> str:
    """Validate an unquoted Snowflake identifier.

    :param value: Name, optionally qualified with dots.
    :param kind: What the name is, used in the error.
    :return: The identifier with surrounding space removed.
    :raises ValueError: The name is empty, contains ``$$``, or other characters.
    """
    text = value.strip()
    _dotted_identifier(text, kind, value)
    return text


def put_file(session: SnowflakeSession, local: Path, stage: str) -> None:
    """Upload one file uncompressed, replacing any file of the same name.

    :param session: Snowpark session.
    :param local: File to upload. Its name is kept on the stage.
    :param stage: Stage directory URL.
    """
    session.file.put(str(local), stage, auto_compress=False, overwrite=True)


def _source_document(source: TableSource | QuerySource) -> dict[str, str]:
    if isinstance(source, TableSource):
        return {"kind": "table", "name": source.name}
    return {"kind": "query", "sql": source.sql}


def lowered(row: SnowflakeRow) -> dict[str, object]:
    """Row values keyed by lower-case column name.

    :param row: Snowpark row.
    :return: Column name to value.
    """
    return {str(key).lower(): value for key, value in row.as_dict().items()}


def _run_id_match(run_id: str) -> re.Match[str]:
    match = RUN_ID_RE.fullmatch(run_id)
    if match is None:
        msg = f"run_id must look like arena-<yyyymmddHHMMSS>-<6 hex>, got {run_id!r}"
        raise ValueError(msg)
    return match


def _dotted_identifier(text: str, kind: str, original: str) -> None:
    parts = text.split(".")
    if (
        not text
        or "$$" in text
        or any(IDENTIFIER.fullmatch(part) is None for part in parts)
    ):
        msg = f"{kind} must be a Snowflake identifier, got {original!r}"
        raise ValueError(msg)
