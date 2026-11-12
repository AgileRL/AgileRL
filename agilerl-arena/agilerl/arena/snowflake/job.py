# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Submit a training manifest as Snowflake job services."""

from __future__ import annotations

import json
import math
import re
import secrets
import shutil
import tempfile
import warnings
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import TypedDict

import yaml

from agilerl.arena.memory import cli as memory_cli
from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.training import CheckpointExportSpec
from agilerl.arena.snowflake.ports import TrainerPorts, service_endpoints
from agilerl.arena.snowflake.profile import (
    ARENA_API_INTEGRATION,
    ARENA_BASE_URL_SECRET,
    ARENA_EXTERNAL_USER_SECRET,
    ARENA_ORG_KEY_SECRET,
    DATABASE,
    GPU_FAMILIES,
    SCHEMA,
    GpuPool,
)
from agilerl.arena.snowflake.source import (
    QuerySource,
    SnowflakeRow,
    SnowflakeSession,
    TableSource,
)

RUN_ID_RE = re.compile(r"^arena-(\d{14})-([0-9a-f]{6})$")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
TABLE_PART = r'(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:[^"]|"")+")'
QUALIFIED_TABLE = re.compile(rf"^{TABLE_PART}\.{TABLE_PART}\.{TABLE_PART}$")
UPLOADED_MARKER = re.compile(r"^checkpoint_(\d+)/\.uploaded$")
# COPY FILES between internal stages rejects larger files.
COPY_FILES_MAX_BYTES = 5 * 10**9
ROLES = ("broker", "head", "worker")
CONTAINER_NAME = "main"
# The stage volume mounts @<runs stage>/<run_id> here on every instance.
RUN_MOUNT = "/arena/run"
JOB_JSON_PATH = f"{RUN_MOUNT}/job.json"
# The spec command replaces the image entrypoint, so tini stays PID 1 to
# forward SIGTERM and reap child processes.
NODE_COMMAND = ["tini", "--", "python", "-m", "agilerl.arena.snowflake.node"]
BROKER_COMMAND = ["tini", "--", "python", "-m", "agilerl.arena.snowflake.broker"]
CONTAINER_ARGS = [
    "python",
    "-m",
    "agilerl.arena.snowflake.job_entry",
    JOB_JSON_PATH,
]
REWARD_FILE = "reward.py"
QUERY_DATASET = "snowflake-query"
ENV_FIELDS = ("entrypoint", "env_url", "env_image")
DATASET_FIELDS = ("dataset", "dataset_path", "hf_dataset_id")
RUBRIC_FIELDS = ("rubric_file_path", "reward_file_path")


class JobDocument(TypedDict):
    """JSON payload the in-job entrypoint and the broker read."""

    broker_service: str
    checkpoint_steps: int | None
    compute_pool: str
    family: str
    gpus: int
    head_service: str
    image: str
    metrics_table: str
    nodes: int
    resume: bool
    reward_file: bool
    run_id: str
    runs_stage: str
    source: dict[str, str] | None
    warehouse: str
    workers: int
    workers_service: str


@dataclass(frozen=True)
class NodeResources:
    """GPUs and host memory of one compute pool node."""

    gpus: int
    memory_gib: int


@dataclass(frozen=True)
class RunServices:
    """Broker, head, and worker service names for one run."""

    broker: str
    head: str
    workers: str


@dataclass(frozen=True)
class SubmitSettings:
    """What one ``arena snowflake submit`` sends with the manifest."""

    image: str
    warehouse: str
    runs_stage: str
    metrics_table: str
    source: TableSource | QuerySource | None
    external_access_integrations: tuple[str, ...]
    checkpoint_steps: int | None
    reward_file: Path | None
    gpus: int
    gpu_family: str
    gpus_per_node: int
    gpu_pool: str
    broker_pool: str
    broker_family: str
    head_pool: str
    head_family: str
    resume_from: str | None


def new_run_id(now: datetime | None = None) -> str:
    """Build ``arena-<UTC yyyymmddHHMMSS>-<6 lowercase hex>``.

    :param now: Clock to stamp. ``None`` uses the current UTC time.
    :return: The run id.
    """
    moment = datetime.now(timezone.utc) if now is None else now
    return f"arena-{moment.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"


def service_name(run_id: str) -> str:
    """Service-name stem ``ARENA_RUN_<yyyymmddHHMMSS>_<HEX>`` for a run id.

    :param run_id: An id from :func:`new_run_id`.
    :return: An unquoted Snowflake identifier, without a role suffix.
    :raises ValueError: ``run_id`` does not match the run id pattern.
    """
    match = _run_id_match(run_id)
    return f"ARENA_RUN_{match.group(1)}_{match.group(2).upper()}"


def run_services(run_id: str) -> RunServices:
    """Broker, head, and worker service names for a run id.

    :param run_id: An id from :func:`new_run_id`.
    :return: The three service names.
    :raises ValueError: ``run_id`` does not match the run id pattern.
    """
    stem = service_name(run_id)
    return RunServices(
        broker=f"{stem}_BROKER",
        head=f"{stem}_HEAD",
        workers=f"{stem}_WORKERS",
    )


def service_for_role(run_id: str, role: str) -> str:
    """Service name for one role of a run.

    :param run_id: An id from :func:`new_run_id`.
    :param role: ``broker``, ``head``, or ``worker``.
    :return: The service name.
    :raises ValueError: ``run_id`` does not match the run id pattern.
    """
    services = run_services(run_id)
    names = {
        "broker": services.broker,
        "head": services.head,
        "worker": services.workers,
    }
    return names[role]


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


def write_job_manifest(
    manifest_path: Path,
    source: TableSource | QuerySource | None,
    has_reward_file: bool,
    directory: Path,
) -> Path:
    """Copy the manifest with the dataset, rubric, and checkpoint export filled in.

    The trainer replaces the dataset and rubric with the exported rows and
    ``reward.py``. An LLM environment that names no env gets ``dataset`` from
    ``source`` and, for a rollout, ``rubric_file_path`` from ``--reward-file``,
    so the manifest does not need placeholders. A manifest without
    ``training.checkpoint_export`` gets ``format: merged``, which ``publish``
    reads. Fields the manifest sets stay.

    :param manifest_path: The user's manifest.
    :param source: ``--source-table`` or ``--source-query``, or ``None``.
    :param has_reward_file: Whether ``--reward-file`` was passed.
    :param directory: Where the copy is written.
    :return: ``directory / "manifest.yaml"``.
    """
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    # Anything but a mapping is left for TrainingManifest validation to reject.
    environment = document.get("environment") if isinstance(document, dict) else None
    if isinstance(environment, dict) and not any(
        environment.get(field) is not None for field in ENV_FIELDS
    ):
        if source is not None and not any(
            environment.get(field) is not None for field in DATASET_FIELDS
        ):
            environment["dataset"] = (
                source.name if isinstance(source, TableSource) else QUERY_DATASET
            )
        if (
            has_reward_file
            and environment.get("env_type") == "rollout"
            and not any(environment.get(field) is not None for field in RUBRIC_FIELDS)
        ):
            environment["rubric_file_path"] = REWARD_FILE
    training = (
        document.setdefault("training", {}) if isinstance(document, dict) else None
    )
    if isinstance(training, dict) and training.get("checkpoint_export") is None:
        training["checkpoint_export"] = CheckpointExportSpec(
            format="merged"
        ).model_dump()
    path = directory / "manifest.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def validate_manifest(
    manifest: TrainingManifest, has_reward_file: bool, has_source: bool
) -> bool:
    """Reject manifests a job service cannot run or publish.

    :param manifest: Parsed training manifest.
    :param has_reward_file: Whether ``--reward-file`` was passed.
    :param has_source: Whether ``--source-table`` or ``--source-query`` was passed.
    :return: Whether the job exports the source. An env-backed rollout does not.
    :raises ValueError: The manifest is not an LLM run, starts an env image,
        has no env to roll out against, needs a source that was not passed,
        the reward file does not fit it, or it writes no merged checkpoint.
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
    export = manifest.training.effective_checkpoint_export()
    if export.format != "merged":
        msg = (
            "Snowflake runs export merged weights for publish. "
            "Remove training.checkpoint_export or set format: merged."
        )
        raise ValueError(msg)
    if export.trigger == "on_demand":
        msg = (
            "Snowflake runs export merged weights for publish. "
            "Remove training.checkpoint_export or use a trigger other than on_demand."
        )
        raise ValueError(msg)
    if environment.env_url is not None:
        warnings.warn(
            "env_url must be reachable from the job. A host outside the "
            "account needs an External Access Integration passed with --eai. "
            "An SPCS service in the same account needs none.",
            stacklevel=2,
        )
    return exports_source


def trainer_ports(manifest: TrainingManifest, worker_gpus: int) -> TrainerPorts:
    """Population size and the most rollout engines one member can start.

    :param manifest: Parsed training manifest.
    :param worker_gpus: GPUs on the worker service. ``auto`` engines use this count.
    :return: The bound the worker spec opens ports for.
    """
    training = manifest.training
    engines = training.rollout_engines_per_agent
    if (
        engines == "auto"
        or "rollout_engines_per_agent" not in training.model_fields_set
    ):
        engines = worker_gpus
    return TrainerPorts(agents=training.pop_size, engines_per_agent=max(1, engines))


def manifest_fits(manifest_path: Path, pool: GpuPool, gpus: int) -> bool:
    """Whether the memory estimator says the manifest fits the run's GPUs.

    :param manifest_path: Training manifest.
    :param pool: The family's pool, for per-GPU memory and GPUs per node.
    :param gpus: GPUs for the whole run, a multiple of ``pool.gpus_per_node``.
        The trainer lays the run out on ``gpus // pool.gpus_per_node`` nodes.
    :return: Whether both phases fit with the underprediction buffer.
    :raises ValueError: The estimator cannot size the manifest.
    """
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = memory_cli.main(
            str(manifest_path),
            device_gb=float(pool.gpu_memory_gib),
            num_gpus=gpus,
            gpus_per_node=pool.gpus_per_node,
            as_json=True,
        )
    if code == memory_cli.EXIT_OK:
        return True
    if code == memory_cli.EXIT_OVER_BUDGET:
        return False
    detail = stderr.getvalue().strip() or stdout.getvalue().strip()
    raise ValueError(detail or "the memory estimator could not size this manifest")


def choose_gpu_family(
    pools: dict[str, GpuPool],
    gpus: int,
    family: str | None,
    fits: Callable[[GpuPool], bool],
) -> str:
    """Pick a GPU family that divides ``gpus`` and that ``fits`` accepts.

    An omitted family is the one with the smallest per-GPU memory. Ties keep
    the family with more GPUs per node, so the run uses fewer nodes, then the
    name. A named family the estimator cannot size is used with a warning.

    :param pools: Families the account profile enabled.
    :param gpus: Requested GPU count.
    :param family: ``--gpu-family``, or ``None`` to pick one.
    :param fits: Whether the manifest fits that family's GPU memory. Raises
        ``ValueError`` when the estimator cannot size the manifest.
    :return: The family name.
    :raises ValueError: The family is missing, none divide ``gpus``, none
        that divide can hold the manifest, or the estimator cannot size it
        for an omitted family.
    """
    if not pools:
        msg = "This account profile has no GPU pool.\nRun arena snowflake setup again."
        raise ValueError(msg)
    if family is not None:
        pool = pools.get(family)
        if pool is None:
            enabled = ", ".join(sorted(pools))
            msg = (
                f"GPU family {family} is not in this account profile. "
                f"Enabled families: {enabled}."
            )
            raise ValueError(msg)
        if pool.gpus_per_node < 1 or gpus % pool.gpus_per_node != 0:
            raise ValueError(_gpu_counts_message(gpus, {family: pool}, family))
        try:
            fitted = fits(pool)
        except ValueError as exc:
            warnings.warn(
                f"The memory estimator could not size this manifest, so "
                f"{family} is used unchecked: {exc}",
                stacklevel=2,
            )
            return family
        if not fitted:
            raise ValueError(_too_small_message(gpus, [(family, pool)], family))
        return family
    dividing = [
        (name, pool)
        for name, pool in pools.items()
        if pool.gpus_per_node > 0 and gpus % pool.gpus_per_node == 0
    ]
    if not dividing:
        raise ValueError(_gpu_counts_message(gpus, pools, None))
    holding = [(name, pool) for name, pool in dividing if fits(pool)]
    if not holding:
        raise ValueError(_too_small_message(gpus, dividing, None))
    holding.sort(
        key=lambda item: (item[1].gpu_memory_gib, -item[1].gpus_per_node, item[0])
    )
    return holding[0][0]


def submit_job(
    session: SnowflakeSession,
    manifest_path: Path,
    settings: SubmitSettings,
    run_id: str | None = None,
) -> tuple[str, RunServices]:
    """Upload the run files and start the broker, head, and worker services.

    :param session: Snowpark session.
    :param manifest_path: Local manifest YAML copied onto the runs stage.
    :param settings: Pools, image, stage, source, and GPU request. ``gpus``
        is a multiple of ``gpus_per_node``, as :func:`choose_gpu_family` checks.
    :param run_id: Run id to use. ``None`` generates one.
    :return: ``(run_id, services)``.
    :raises ValueError: A pool family is unknown, the worker pool's
        ``MAX_NODES`` is below the worker count, or the run to resume has no
        checkpoint that can be copied.
    """
    manifest = load_manifest(manifest_path)
    exports_source = validate_manifest(
        manifest, settings.reward_file is not None, settings.source is not None
    )
    if not exports_source:
        settings = replace(settings, source=None)
    chosen = new_run_id() if run_id is None else run_id
    services = run_services(chosen)
    workers = settings.gpus // settings.gpus_per_node
    trainer = trainer_ports(manifest, settings.gpus)
    broker_spec = render_broker_spec(settings, chosen)
    head_spec = render_head_spec(settings, chosen)
    workers_spec = render_workers_spec(settings, chosen, trainer)
    _reject_worker_capacity(session, settings, workers)
    checkpoint = None
    if settings.resume_from is not None:
        checkpoint = copy_resume_checkpoint(
            session, settings.runs_stage, settings.resume_from, chosen
        )
    print(run_banner(settings, workers), flush=True)
    if checkpoint is not None:
        print(f"resume   {settings.resume_from}/{checkpoint}", flush=True)
    _upload_run_files(session, manifest_path, settings, chosen, services, workers)
    _execute_services(
        session,
        _service_statements(
            settings, services, (broker_spec, head_spec, workers_spec), workers
        ),
    )
    return chosen, services


def _service_statements(
    settings: SubmitSettings,
    services: RunServices,
    specs: tuple[str, str, str],
    workers: int,
) -> list[tuple[str, str]]:
    broker_spec, head_spec, workers_spec = specs
    integrations = settings.external_access_integrations
    return [
        (
            services.broker,
            job_service_sql(
                settings.broker_pool,
                settings.warehouse,
                services.broker,
                broker_spec,
                1,
                (),
            ),
        ),
        (
            services.head,
            job_service_sql(
                settings.head_pool,
                settings.warehouse,
                services.head,
                head_spec,
                1,
                _head_integrations(integrations),
            ),
        ),
        (
            services.workers,
            job_service_sql(
                settings.gpu_pool,
                settings.warehouse,
                services.workers,
                workers_spec,
                workers,
                integrations,
            ),
        ),
    ]


def resume_checkpoint(listing: list[SnowflakeRow], run_id: str) -> str:
    """Newest ``checkpoint_<n>`` of a run that has its ``.uploaded`` marker.

    The uploader puts ``checkpoint_<n>/.uploaded`` after every file of that
    checkpoint is on the stage.

    :param listing: ``LIST @<runs stage>/<run_id>/`` rows.
    :param run_id: Run the rows are under.
    :return: The directory name.
    :raises ValueError: No checkpoint has ``.uploaded``, or a file of the
        newest one is over the ``COPY FILES`` size limit.
    """
    sizes: dict[str, int] = {}
    for row in listing:
        fields = lowered(row)
        relative = str(fields["name"]).split(f"/{run_id}/", 1)[-1]
        sizes[relative] = int(str(fields["size"]))
    steps = [
        int(match.group(1))
        for match in map(UPLOADED_MARKER.match, sizes)
        if match is not None
    ]
    if not steps:
        msg = (
            f"Run {run_id} has no checkpoint with an .uploaded marker, so none "
            "finished uploading."
        )
        raise ValueError(msg)
    checkpoint = f"checkpoint_{max(steps)}"
    too_large = sorted(
        path
        for path, size in sizes.items()
        if path.startswith(f"{checkpoint}/") and size > COPY_FILES_MAX_BYTES
    )
    if too_large:
        msg = (
            f"Run {run_id} {checkpoint} cannot be copied for resume: COPY FILES "
            f"takes files up to 5 GB, and these are larger: {', '.join(too_large)}"
        )
        raise ValueError(msg)
    return checkpoint


def copy_resume_checkpoint(
    session: SnowflakeSession, runs_stage: str, previous_run_id: str, run_id: str
) -> str:
    """Copy a run's newest uploaded checkpoint to ``<run_id>/resume/checkpoint_<n>/``.

    The trainer reads the resume step from the ``checkpoint_<n>`` directory name.

    :param session: Snowpark session.
    :param runs_stage: Stage that holds both runs.
    :param previous_run_id: Run to resume.
    :param run_id: New run.
    :return: The copied directory name.
    :raises ValueError: A run id is not valid, or no checkpoint of the
        previous run can be copied.
    """
    source = runs_prefix(runs_stage, previous_run_id)
    target = runs_prefix(runs_stage, run_id)
    listing = list(session.sql(f"LIST {source}/").collect())
    checkpoint = resume_checkpoint(listing, previous_run_id)
    session.sql(
        f"COPY FILES INTO {target}/resume/{checkpoint}/ FROM {source}/{checkpoint}/"
    ).collect()
    return checkpoint


def _upload_run_files(
    session: SnowflakeSession,
    manifest_path: Path,
    settings: SubmitSettings,
    run_id: str,
    services: RunServices,
    workers: int,
) -> None:
    """Copy the manifest, job document, and optional reward file to the runs stage."""
    prefix = runs_prefix(settings.runs_stage, run_id)
    with tempfile.TemporaryDirectory() as tmp:
        payload = Path(tmp)
        shutil.copyfile(manifest_path, payload / "manifest.yaml")
        (payload / "job.json").write_text(
            json.dumps(
                job_document(settings, run_id, services, workers),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        uploads = ["manifest.yaml", "job.json"]
        if settings.reward_file is not None:
            shutil.copyfile(settings.reward_file, payload / REWARD_FILE)
            uploads.append(REWARD_FILE)
        for upload in uploads:
            put_file(session, payload / upload, prefix)


def job_document(
    settings: SubmitSettings, run_id: str, services: RunServices, workers: int
) -> JobDocument:
    """JSON payload the in-job entrypoint and the broker read.

    ``nodes`` is the worker count, which the trainer passes as ``--num-nodes``.
    ``compute_pool`` and ``image`` are the worker pool and the shared image.
    ``resume`` means the run directory holds ``resume/checkpoint_<n>/``.

    :param settings: Submit settings.
    :param run_id: Run id for this payload.
    :param services: Service names for this run.
    :param workers: Worker replicas.
    :return: A JSON-ready document.
    """
    return {
        "broker_service": services.broker,
        "checkpoint_steps": settings.checkpoint_steps,
        "compute_pool": settings.gpu_pool,
        "family": settings.gpu_family,
        "gpus": settings.gpus,
        "head_service": services.head,
        "image": settings.image,
        "metrics_table": settings.metrics_table,
        "nodes": workers,
        "resume": settings.resume_from is not None,
        "reward_file": settings.reward_file is not None,
        "run_id": run_id,
        "runs_stage": settings.runs_stage,
        "source": None
        if settings.source is None
        else _source_document(settings.source),
        "warehouse": settings.warehouse,
        "workers": workers,
        "workers_service": services.workers,
    }


def node_resources(family: str) -> NodeResources:
    """GPUs and host memory for a documented Snowflake GPU family.

    :param family: ``INSTANCE_FAMILY`` from ``DESCRIBE COMPUTE POOL``.
    :return: The resources one node of that family provides.
    :raises ValueError: The family is not a documented GPU family.
    """
    spec = GPU_FAMILIES.get(family)
    if spec is None:
        msg = (
            f"Unknown compute pool instance family {family!r}. "
            "Arena jobs run on a Snowflake GPU instance family."
        )
        raise ValueError(msg)
    return NodeResources(gpus=spec.gpus, memory_gib=spec.host_memory_gib)


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


def cpu_memory_gib(family: str) -> int:
    """Host memory in GiB for a Snowflake CPU instance family.

    :param family: ``INSTANCE_FAMILY`` of the broker or head pool.
    :return: Host memory in GiB.
    :raises ValueError: ``family`` is not a known CPU family.
    """
    families = {
        "CPU_X64_XS": 6,
        "CPU_X64_S": 13,
        "CPU_X64_M": 28,
        "CPU_X64_SL": 58,
        "CPU_X64_L": 116,
        "HIGHMEM_X64_S": 58,
        "HIGHMEM_X64_M": 240,
        "HIGHMEM_X64_L": 984,
    }
    found = families.get(family)
    if found is None:
        msg = f"Unknown CPU instance family {family}."
        raise ValueError(msg)
    return found


def container_env(run_id: str) -> dict[str, str]:
    """Container environment Ray passes from each raylet to its actors.

    :param run_id: Run id. It sets ``ARENA_RUN_ID`` and ``JOB_ID``.
    :return: Variable name to value.
    """
    topic_prefix = "non-persistent://arena/run"
    return {
        "APP_TOPIC_PREFIX": topic_prefix,
        "AGILERL_ROLLOUT_TOPIC_ROUTE": rollout_topic_route(topic_prefix),
        "ARENA_RUN_ID": run_id,
        "JOB_ID": str(job_id_from_run_id(run_id)),
    }


def run_banner(settings: SubmitSettings, workers: int) -> str:
    """Lines printed before the three services start.

    :param settings: Pools and the GPU request.
    :param workers: Worker replicas.
    :return: The banner.
    """
    return "\n".join(
        (
            f"broker   {settings.broker_pool}  {settings.broker_family}",
            f"head     {settings.head_pool}  {settings.head_family}",
            (
                f"workers  {settings.gpus} GPUs on {workers} workers "
                f"of {settings.gpu_family}  {settings.gpu_pool}"
            ),
        )
    )


def render_broker_spec(settings: SubmitSettings, run_id: str) -> str:
    """Service specification for the Pulsar broker.

    :param settings: Submit settings.
    :param run_id: Run directory on the stage.
    :return: YAML text.
    :raises ValueError: The YAML would break the ``$$`` SQL quote, or the
        broker family is unknown.
    """
    memory_gib = cpu_memory_gib(settings.broker_family)
    container = _container(
        settings,
        run_id,
        list(BROKER_COMMAND),
        memory_gib,
        None,
        [
            {"name": "run", "mountPath": RUN_MOUNT},
            {"name": "data", "mountPath": "/pulsar-data"},
        ],
    )
    return _dump_spec(
        {
            "spec": {
                "containers": [container],
                "endpoints": [
                    {
                        "name": "pulsar",
                        "port": 6650,
                        "protocol": "TCP",
                        "public": False,
                    },
                    {
                        "name": "pulsar-admin",
                        "port": 8080,
                        "protocol": "TCP",
                        "public": False,
                    },
                ],
                "volumes": [
                    _stage_volume(settings, run_id),
                    _block_volume(),
                ],
            }
        }
    )


def render_head_spec(settings: SubmitSettings, run_id: str) -> str:
    """Service specification for the Ray head.

    :param settings: Submit settings.
    :param run_id: Run directory on the stage.
    :return: YAML text.
    :raises ValueError: The YAML would break the ``$$`` SQL quote, or the
        head family is unknown.
    """
    memory_gib = cpu_memory_gib(settings.head_family)
    container = _ray_container(settings, run_id, memory_gib, None)
    container["secrets"] = arena_container_secrets()
    return _dump_spec(
        {
            "spec": {
                "containers": [
                    container,
                ],
                "endpoints": service_endpoints(None),
                "volumes": [
                    _stage_volume(settings, run_id),
                    _memory_volume(memory_gib),
                ],
            }
        }
    )


def render_workers_spec(
    settings: SubmitSettings, run_id: str, trainer: TrainerPorts
) -> str:
    """Service specification for the GPU workers.

    :param settings: Submit settings.
    :param run_id: Run directory on the stage.
    :param trainer: Population and engine bound the endpoints cover.
    :return: YAML text.
    :raises ValueError: The YAML would break the ``$$`` SQL quote, the GPU
        family is unknown, or a trainer span reaches the Ray ports.
    """
    resources = node_resources(settings.gpu_family)
    return _dump_spec(
        {
            "spec": {
                "containers": [
                    _ray_container(
                        settings, run_id, resources.memory_gib, resources.gpus
                    ),
                ],
                "endpoints": service_endpoints(trainer),
                "volumes": [
                    _stage_volume(settings, run_id),
                    _memory_volume(resources.memory_gib),
                ],
            }
        }
    )


def job_service_sql(
    pool: str,
    warehouse: str,
    name: str,
    spec: str,
    replicas: int,
    external_access_integrations: tuple[str, ...] = (),
) -> str:
    """``EXECUTE JOB SERVICE`` statement for one role.

    :param pool: Compute pool name.
    :param warehouse: Warehouse for queries the service runs.
    :param name: Service identifier.
    :param spec: Specification YAML placed inside ``$$``.
    :param replicas: ``REPLICAS`` for this service.
    :param external_access_integrations: Integrations to attach. Empty omits
        the clause.
    :return: The SQL statement.
    """
    pool_name = snowflake_identifier(pool, "compute pool")
    warehouse_name = snowflake_identifier(warehouse, "warehouse")
    service = snowflake_identifier(name, "service")
    lines = [
        f"EXECUTE JOB SERVICE IN COMPUTE POOL {pool_name}",
        f"NAME = {service}",
        "ASYNC = TRUE",
        f"REPLICAS = {replicas}",
        f"QUERY_WAREHOUSE = {warehouse_name}",
    ]
    if external_access_integrations:
        names = ", ".join(
            snowflake_identifier(item, "external access integration")
            for item in external_access_integrations
        )
        lines.append(f"EXTERNAL_ACCESS_INTEGRATIONS = ({names})")
    lines.append(f"FROM SPECIFICATION $$\n{spec}$$")
    return "\n".join(lines)


def drop_service_sql(name: str, if_exists: bool = False) -> str:
    """Force the drop so a block volume does not have to be snapshotted.

    :param name: Service identifier.
    :param if_exists: When true, the statement is ``DROP SERVICE IF EXISTS ... FORCE``.
    :return: The SQL statement.
    """
    service = snowflake_identifier(name, "service")
    if if_exists:
        return f"DROP SERVICE IF EXISTS {service} FORCE"
    return f"DROP SERVICE {service} FORCE"


def cancel_run(session: SnowflakeSession, run_id: str) -> None:
    """Drop the broker, head, and workers for a run.

    :param session: Snowpark session.
    :param run_id: An id from :func:`new_run_id`.
    :raises ValueError: ``run_id`` does not match the run id pattern.
    """
    services = run_services(run_id)
    for name in (services.broker, services.head, services.workers):
        session.sql(drop_service_sql(name, if_exists=True)).collect()


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


def service_event_logs_sql(service: str, instance: int) -> str:
    """Last 1000 event-table log lines of the ``main`` container, newest first.

    ``<service>!SPCS_GET_LOGS`` reads the account's active event table and
    needs only MONITOR on the service. The service must still exist.

    :param service: Service name.
    :param instance: Instance id, starting at 0.
    :return: The SQL statement.
    """
    name = snowflake_identifier(service, "service")
    return (
        f"SELECT LOG FROM TABLE({name}!SPCS_GET_LOGS("
        "START_TIME => DATEADD('day', -30, CURRENT_TIMESTAMP())))\n"
        f"WHERE INSTANCE_ID = {instance} AND CONTAINER_NAME = '{CONTAINER_NAME}'\n"
        "ORDER BY TIMESTAMP DESC\n"
        "LIMIT 1000"
    )


def event_log_text(rows: list[SnowflakeRow]) -> str:
    """Log lines from :func:`service_event_logs_sql` rows, oldest first.

    :param rows: Rows with a ``LOG`` column, newest first.
    :return: The lines joined by newlines.
    """
    return "\n".join(str(lowered(row)["log"]) for row in reversed(rows))


def job_services_sql(run_id: str) -> str:
    """``SHOW JOB SERVICES`` for the broker, head, and workers of one run.

    :param run_id: An id from :func:`new_run_id`.
    :return: The SQL statement.
    :raises ValueError: ``run_id`` does not match the run id pattern.
    """
    return f"SHOW JOB SERVICES LIKE '{service_name(run_id)}%'"


def role_states(run_id: str, rows: list[SnowflakeRow]) -> dict[str, str]:
    """Job state of each role from ``SHOW JOB SERVICES`` rows.

    :param run_id: An id from :func:`new_run_id`.
    :param rows: :func:`job_services_sql` rows.
    :return: Role to :func:`job_state`. A dropped service is ``gone``.
    """
    found = {
        str(values["name"]).upper(): str(values["status"]).upper()
        for values in map(lowered, rows)
    }
    states = {}
    for role in ROLES:
        status = found.get(service_for_role(run_id, role))
        states[role] = job_state(set() if status is None else {status})
    return states


def run_state(states: Mapping[str, str]) -> str:
    """One state for a run from its role states.

    The head runs the trainer, so its exit decides success. The head drops
    the broker when it ends.

    :param states: Role to :func:`job_state`, for every role in :data:`ROLES`.
    :return: ``failed``, ``cancelling``, ``cancelled``, ``succeeded``,
        ``gone`` when every service is dropped, else ``running``.
    """
    values = set(states.values())
    for state in ("failed", "cancelling", "cancelled"):
        if state in values:
            return state
    if states["head"] == "done":
        return "succeeded"
    if values == {"gone"}:
        return "gone"
    return "running"


def format_run_status(
    states: Mapping[str, str], containers: Mapping[str, list[SnowflakeRow]]
) -> str:
    """One line per role, its containers indented below it, then the run state.

    :param states: Role to :func:`job_state`.
    :param containers: Role to its ``SHOW SERVICE CONTAINERS`` rows. A
        pending service reports one row with no instance, which is skipped.
    :return: The lines, ending with ``state: <run state>``.
    """
    lines = []
    for role, state in states.items():
        lines.append(f"{role}\t{state}")
        for values in map(lowered, containers[role]):
            if values.get("instance_id") is None:
                continue
            lines.append(
                f"  {values['instance_id']}\t"
                f"{values.get('container_name', '')}\t"
                f"{values.get('status', '')}"
            )
    lines.append(f"state: {run_state(states)}")
    return "\n".join(lines)


def job_state(service_statuses: set[str]) -> str:
    """Job state from the job service's own status.

    The service status turns ``DONE`` only after every replica has exited 0,
    so instances that are not listed yet cannot make the job look finished.
    ``CANCELLING`` is a cancel still in progress.

    :param service_statuses: Upper-case service status values. Empty when the
        service was not found.
    :return: ``failed`` for ``FAILED`` or ``INTERNAL_ERROR``, ``cancelled``,
        ``cancelling``, ``gone`` when not found, deleted, or suspended,
        ``done`` for ``DONE``, else ``running``.
    """
    if service_statuses & {"FAILED", "INTERNAL_ERROR"}:
        return "failed"
    if "CANCELLED" in service_statuses:
        return "cancelled"
    if "CANCELLING" in service_statuses:
        return "cancelling"
    if not service_statuses or service_statuses & {
        "DELETING",
        "DELETED",
        "SUSPENDING",
        "SUSPENDED",
    }:
        return "gone"
    if service_statuses == {"DONE"}:
        return "done"
    return "running"


def qualified_table(name: str) -> str:
    """Validate a ``database.schema.table`` name. Parts may be double-quoted.

    :param name: ``--source-table`` value.
    :return: The name with surrounding space removed.
    :raises ValueError: The name is not three dot-separated identifiers.
    """
    text = name.strip()
    if QUALIFIED_TABLE.fullmatch(text) is None:
        msg = f"--source-table must be database.schema.table, got {name!r}"
        raise ValueError(msg)
    return text


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


def describe_max_nodes(session: SnowflakeSession, compute_pool: str) -> int:
    """Read ``max_nodes`` from ``DESCRIBE COMPUTE POOL``.

    :param session: Snowpark session.
    :param compute_pool: Pool name.
    :return: The pool's ``MAX_NODES``.
    :raises ValueError: The pool was not found or the row has no ``max_nodes``.
    """
    name = snowflake_identifier(compute_pool, "compute pool")
    rows = session.sql(f"DESCRIBE COMPUTE POOL {name}").collect()
    if not rows:
        msg = f"Compute pool {compute_pool!r} was not found"
        raise ValueError(msg)
    raw = lowered(rows[0]).get("max_nodes")
    if isinstance(raw, bool) or raw is None or str(raw).strip() == "":
        msg = f"DESCRIBE COMPUTE POOL {compute_pool!r} did not return max_nodes"
        raise ValueError(msg)
    return int(str(raw).strip())


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


def _reject_worker_capacity(
    session: SnowflakeSession, settings: SubmitSettings, workers: int
) -> None:
    max_nodes = describe_max_nodes(session, settings.gpu_pool)
    if workers <= max_nodes:
        return
    msg = (
        f"{workers} workers is above MAX_NODES {max_nodes} on {settings.gpu_pool}.\n"
        f"arena snowflake pools set {settings.gpu_family} --max-gpus {settings.gpus}"
    )
    raise ValueError(msg)


def _execute_services(
    session: SnowflakeSession, statements: list[tuple[str, str]]
) -> None:
    started: list[str] = []
    try:
        for name, statement in statements:
            session.sql(statement).collect()
            started.append(name)
    except Exception:
        for name in reversed(started):
            session.sql(drop_service_sql(name)).collect()
        raise


def _too_small_message(
    gpus: int, pools: list[tuple[str, GpuPool]], family: str | None
) -> str:
    """Families that divide ``gpus`` whose GPU memory cannot hold the manifest."""
    named = sorted(pools)
    listed = "\n".join(f"{name} ({pool.gpu_memory_gib} GiB)" for name, pool in named)
    if family is not None:
        return f"{listed} cannot hold this manifest."
    return (
        f"No GPU family that divides {gpus} GPUs can hold this manifest.\n"
        f"Too small:\n{listed}"
    )


def _gpu_counts_message(
    gpus: int, pools: dict[str, GpuPool], family: str | None
) -> str:
    if family is None:
        lines = [f"No GPU family divides {gpus} GPUs."]
        names = sorted(pools)
    else:
        lines = [f"{family} does not divide {gpus} GPUs."]
        names = [family]
    for name in names:
        pool = pools[name]
        counts: list[int] = []
        count = pool.gpus_per_node
        if count > 0:
            while count <= pool.max_gpus:
                counts.append(count)
                count += pool.gpus_per_node
        if counts:
            listed = ", ".join(str(item) for item in counts)
            lines.append(f"{name} runs {listed} GPUs.")
        else:
            lines.append(f"{name} runs no GPU count within its max.")
    return "\n".join(lines)


def _head_integrations(integrations: tuple[str, ...]) -> tuple[str, ...]:
    if ARENA_API_INTEGRATION in integrations:
        return integrations
    return (*integrations, ARENA_API_INTEGRATION)


def arena_container_secrets() -> list[dict[str, str]]:
    """Container ``secrets`` for the Arena org key, external user id, and base URL.

    :return: Spec entries that set ``ARENA_ORG_KEY``, ``ARENA_EXTERNAL_USER_ID``,
        and ``ARENA_BASE_URL``.
    """
    schema = f"{DATABASE}.{SCHEMA}"
    return [
        {
            "snowflakeSecret": f"{schema}.{ARENA_ORG_KEY_SECRET}",
            "secretKeyRef": "secret_string",
            "envVarName": "ARENA_ORG_KEY",
        },
        {
            "snowflakeSecret": f"{schema}.{ARENA_EXTERNAL_USER_SECRET}",
            "secretKeyRef": "secret_string",
            "envVarName": "ARENA_EXTERNAL_USER_ID",
        },
        {
            "snowflakeSecret": f"{schema}.{ARENA_BASE_URL_SECRET}",
            "secretKeyRef": "secret_string",
            "envVarName": "ARENA_BASE_URL",
        },
    ]


def _ray_container(
    settings: SubmitSettings, run_id: str, memory_gib: int, gpus: int | None
) -> dict[str, object]:
    return _container(
        settings,
        run_id,
        [*NODE_COMMAND, *CONTAINER_ARGS],
        memory_gib,
        gpus,
        [
            {"name": "run", "mountPath": RUN_MOUNT},
            {"name": "dshm", "mountPath": "/dev/shm"},
        ],
    )


def _container(
    settings: SubmitSettings,
    run_id: str,
    command: list[str],
    memory_gib: int,
    gpus: int | None,
    mounts: list[dict[str, str]],
) -> dict[str, object]:
    # The Stage Mount V2 sidecar needs 0.5 GiB of the node.
    memory = f"{memory_gib - 0.5:g}Gi"
    requests: dict[str, object] = {"memory": memory}
    if gpus is not None:
        requests = {"nvidia.com/gpu": gpus, "memory": memory}
    container: dict[str, object] = {
        "name": CONTAINER_NAME,
        "image": settings.image,
        # SPCS rejects ``args`` when ``command`` has more than one entry.
        "command": command,
    }
    container["env"] = container_env(run_id)
    container["resources"] = {"requests": requests, "limits": dict(requests)}
    container["volumeMounts"] = mounts
    return container


def _stage_volume(settings: SubmitSettings, run_id: str) -> dict[str, object]:
    return {
        "name": "run",
        "source": "stage",
        "stageConfig": {"name": runs_prefix(settings.runs_stage, run_id)},
    }


def _memory_volume(memory_gib: int) -> dict[str, object]:
    return {
        "name": "dshm",
        "source": "memory",
        "size": f"{shm_gib(memory_gib)}Gi",
    }


def _block_volume() -> dict[str, object]:
    # size is a volume field. iops and throughput sit under blockConfig.
    return {
        "name": "data",
        "source": "block",
        "size": "1000Gi",
        "blockConfig": {"iops": 16000, "throughput": 1000},
    }


def _dump_spec(spec: dict[str, object]) -> str:
    text = yaml.safe_dump(spec, sort_keys=False)
    if "$$" in text:
        msg = "service spec cannot contain $$"
        raise ValueError(msg)
    return text


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
