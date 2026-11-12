# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Start a Snowflake job that logs a run's merged checkpoint to the Model Registry."""

from __future__ import annotations

import json
import math
import re
import secrets
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Protocol, TypedDict

import yaml

from agilerl.arena.snowflake.job import (
    CONTAINER_NAME,
    RUN_MOUNT,
    describe_instance_family,
    job_state,
    log_text,
    lowered,
    node_resources,
    put_file,
    runs_prefix,
    service_logs_sql,
    service_status_sql,
    snowflake_identifier,
)
from agilerl.arena.snowflake.source import (
    SnowflakeFiles,
    SnowflakeRow,
    SnowflakeSession,
)

MERGED_EXPORT_REQUIRED = (
    "Runs write merged/ when their checkpoint_export trigger fires, by default "
    "at the final save, so a run stopped before then or an earlier step has none."
)


class StageFiles(SnowflakeFiles, Protocol):
    """Stage upload and download."""

    def get_stream(self, stage_location: str) -> IO[bytes]:
        """Open one staged file."""


class PublishSession(SnowflakeSession, Protocol):
    """Snowpark session, including stage downloads."""

    file: StageFiles


class PublishDocument(TypedDict):
    """``publish-<job>.json``, read by the publish job through the run mount."""

    checkpoint: str
    compute_pool: str
    database: str
    gpu_count: int
    image_build_compute_pool: str
    max_model_len: int | None
    model_name: str
    run_id: str
    schema: str
    service_name: str | None
    step: int
    version_name: str


@dataclass(frozen=True)
class PublishSettings:
    """What one ``arena snowflake publish`` asks for.

    :param run_id: Run id from ``arena snowflake submit``.
    :param runs_stage: Stage that holds the run directory.
    :param model_name: Model Registry model name.
    :param version_name: Registry version. ``None`` uses the checkpoint directory name.
    :param step: Checkpoint step. ``None`` takes the newest complete merged export.
    :param serve: Start a vLLM service after logging the model.
    :param compute_pool: Pool for the publish job and the service. ``None`` uses the run's.
    :param image_build_pool: CPU pool that builds the service's inference image.
    :param image: Job image. ``None`` uses the run's.
    :param warehouse: Query warehouse for the job. ``None`` uses the run's.
    :param gpu_count: GPUs the service requests.
    :param max_model_len: vLLM ``--max-model-len``, or ``None``.
    :param service_name: Service name. ``None`` uses ``<model>_<version>``.
    :param database: Registry database. Set with ``schema``, or neither for the session's.
    :param schema: Registry schema. Set with ``database``, or neither for the session's.
    """

    run_id: str
    runs_stage: str
    model_name: str
    version_name: str | None
    step: int | None
    serve: bool
    compute_pool: str | None
    image_build_pool: str
    image: str | None
    warehouse: str | None
    gpu_count: int
    max_model_len: int | None
    service_name: str | None
    database: str | None
    schema: str | None


@dataclass(frozen=True)
class PublishJob:
    """A started publish job service and the request it reads."""

    name: str
    document: PublishDocument


def submit_publish(
    session: PublishSession, settings: PublishSettings, now: datetime | None = None
) -> PublishJob:
    """Check names, write ``publish-<job>.json`` to the run directory, and start the job.

    :param session: Snowpark session.
    :param settings: What to publish and where.
    :param now: Clock for the job name. ``None`` uses the current UTC time.
    :return: The job service name and the request it reads.
    :raises ValueError: A name is unsafe or taken, or the run has no usable checkpoint.
    """
    model_name = single_identifier(settings.model_name, "model name")
    image_build_pool = single_identifier(settings.image_build_pool, "image build pool")
    database, schema = registry_schema(session, settings.database, settings.schema)
    prefix = runs_prefix(settings.runs_stage, settings.run_id)
    run = json.loads(session.file.get_stream(f"{prefix}/job.json").read())
    compute_pool = settings.compute_pool or str(run["compute_pool"])
    listing = session.sql(f"LIST {prefix}/").collect()
    checkpoint, step, checkpoint_bytes = select_checkpoint(
        listing, settings.run_id, settings.step
    )
    version_name = single_identifier(
        settings.version_name or checkpoint, "version name"
    )
    reject_existing_version(session, database, schema, model_name, version_name)
    service_name = None
    if settings.serve:
        service_name = single_identifier(
            settings.service_name or f"{model_name}_{version_name}", "service name"
        )
        reject_existing_service(session, database, schema, service_name)
    resources = node_resources(describe_instance_family(session, compute_pool))
    if settings.serve and settings.gpu_count > resources.gpus:
        msg = (
            f"--gpu-count {settings.gpu_count} is more than the "
            f"{resources.gpus} GPUs on a {compute_pool} node"
        )
        raise ValueError(msg)
    request_gib, limit_gib = job_memory_gib(
        checkpoint, checkpoint_bytes, compute_pool, resources.memory_gib
    )
    name = publish_job_name(now)
    request_file = f"publish-{name}.json"
    spec = render_publish_spec(
        settings.image or str(run["image"]),
        prefix,
        request_file,
        request_gib,
        limit_gib,
    )
    statement = publish_job_sql(
        compute_pool, settings.warehouse or str(run["warehouse"]), name, spec
    )
    document: PublishDocument = {
        "checkpoint": checkpoint,
        "compute_pool": compute_pool,
        "database": database,
        "gpu_count": settings.gpu_count,
        "image_build_compute_pool": image_build_pool,
        "max_model_len": settings.max_model_len,
        "model_name": model_name,
        "run_id": settings.run_id,
        "schema": schema,
        "service_name": service_name,
        "step": step,
        "version_name": version_name,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / request_file
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        put_file(session, path, prefix)
    session.sql(statement).collect()
    return PublishJob(name=name, document=document)


def wait_for_publish(
    session: SnowflakeSession,
    job: PublishJob,
    echo: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    poll_s: float = 10.0,
    timeout_s: float = 3600.0,
) -> None:
    """Poll the publish job until it exits or the deadline passes.

    The job waits for the vLLM service, and a GPU service can take 20 minutes
    or more to start.

    :param session: Snowpark session.
    :param job: Job from :func:`submit_publish`.
    :param echo: Called with ``status: <service status>`` each time it changes.
    :param sleep: Called between polls.
    :param clock: Seconds on a monotonic clock.
    :param poll_s: Seconds between polls.
    :param timeout_s: Seconds to wait before giving up. The job keeps running.
    :raises RuntimeError: The job ended in any state but done, with its log,
        or is still running at the deadline.
    """
    statement = service_status_sql(job.name)
    deadline = clock() + timeout_s
    shown = ""
    while True:
        rows = session.sql(statement).collect()
        statuses = {str(lowered(row)["service_status"]).upper() for row in rows}
        status = ", ".join(sorted(statuses)) or "no containers yet"
        if status != shown:
            echo(f"status: {status}")
            shown = status
        # A job lists no containers until a node takes it.
        state = job_state(statuses) if statuses else "running"
        if state == "done":
            return
        if state != "running":
            logs = log_text(session.sql(service_logs_sql(job.name, 0)).collect())
            msg = f"Publish job {job.name} {state}:\n{logs}"
            raise RuntimeError(msg)
        if clock() >= deadline:
            raise RuntimeError(publish_timeout_message(job, status, timeout_s))
        sleep(poll_s)


def publish_timeout_message(job: PublishJob, status: str, timeout_s: float) -> str:
    """Message when the publish job outlives the wait.

    :param job: The running job.
    :param status: Its last service status.
    :param timeout_s: Seconds waited.
    :return: The message, with the statements that check on the job.
    """
    document = job.document
    lines = [
        (
            f"Publish job {job.name} is still {status} after {timeout_s / 60:g} minutes. "
            "It keeps running in Snowflake."
        ),
        f"Status: SHOW SERVICE CONTAINERS IN SERVICE {job.name}",
        f"Logs: {service_logs_sql(job.name, 0)}",
    ]
    if document["service_name"] is not None:
        lines.append(
            "Endpoint, once the job is done: arena snowflake endpoint "
            f"{document['model_name']} --version {document['version_name']} "
            f"--database {document['database']} --schema {document['schema']}"
        )
    return "\n".join(lines)


def service_endpoint(session: SnowflakeSession, document: PublishDocument) -> str:
    """Ingress URL of the service a publish started.

    :param session: Snowpark session.
    :param document: Request of a finished publish job that started a service.
    :return: The URL, or Snowflake's provisioning message while ingress starts.
    :raises ValueError: The request started no service, or it has no ``inference`` endpoint.
    """
    service = document["service_name"]
    if service is None:
        msg = "This publish started no service"
        raise ValueError(msg)
    return ingress_url(
        session, f"{document['database']}.{document['schema']}.{service}"
    )


def ingress_url(session: SnowflakeSession, service: str) -> str:
    """Ingress URL of a service's ``inference`` endpoint.

    :param session: Snowpark session.
    :param service: Fully qualified service name.
    :return: The URL, or Snowflake's provisioning message while ingress starts.
    :raises ValueError: The service has no ``inference`` endpoint.
    """
    for row in session.sql(f"SHOW ENDPOINTS IN SERVICE {service}").collect():
        values = lowered(row)
        if values.get("name") == "inference":
            return str(values.get("ingress_url"))
    msg = f"Service {service} has no inference endpoint"
    raise ValueError(msg)


@dataclass(frozen=True)
class ServedVersion:
    """One Model Registry version and the inference services on it."""

    model: str
    version_name: str
    services: tuple[str, ...]
    is_default: bool
    only_version: bool


def served_version(
    session: SnowflakeSession,
    database: str,
    schema: str,
    model_name: str,
    version_name: str | None,
) -> ServedVersion:
    """Find a version and its inference services.

    :param session: Snowpark session.
    :param database: Registry database.
    :param schema: Registry schema.
    :param model_name: Model name.
    :param version_name: Version. ``None`` takes the one version that has a service.
    :return: The version as Snowflake stores it, with fully qualified service names.
    :raises ValueError: The model or version does not exist, or ``None`` matches
        no version or several.
    """
    model = f"{database}.{schema}.{single_identifier(model_name, 'model name')}"
    models = session.sql(
        f"SHOW MODELS LIKE '{model_name}' IN SCHEMA {database}.{schema}"
    ).collect()
    if not has_name(models, model_name):
        msg = f"Model {model} does not exist"
        raise ValueError(msg)
    rows = [
        lowered(row) for row in session.sql(f"SHOW VERSIONS IN MODEL {model}").collect()
    ]
    services = {
        str(values["name"]).upper(): tuple(
            str(service)
            for service in json.loads(str(values["inference_services"]))
            # The image build job is listed beside the inference service.
            if "MODEL_BUILD_" not in str(service)
        )
        for values in rows
    }
    if version_name is None:
        served = [name for name, names in services.items() if names]
        if not served:
            msg = f"Model {model} has no inference service"
            raise ValueError(msg)
        if len(served) > 1:
            msg = (
                f"Model {model} has services on versions {', '.join(served)}. "
                "Pass --version."
            )
            raise ValueError(msg)
        chosen = served[0]
    else:
        chosen = single_identifier(version_name, "version name").upper()
        if chosen not in services:
            msg = f"Model {model} has no version {version_name}"
            raise ValueError(msg)
    row = next(values for values in rows if str(values["name"]).upper() == chosen)
    return ServedVersion(
        model=model,
        version_name=str(row["name"]),
        services=services[chosen],
        is_default=bool(row["is_default_version"]),
        only_version=len(rows) == 1,
    )


def version_endpoint(session: SnowflakeSession, served: ServedVersion) -> str:
    """Ingress URL of the version's one inference service.

    :param session: Snowpark session.
    :param served: Version from :func:`served_version`.
    :return: The URL, or Snowflake's provisioning message while ingress starts.
    :raises ValueError: The version has no service or several.
    """
    if len(served.services) != 1:
        listed = "".join(f"\n  {service}" for service in served.services)
        msg = (
            f"Version {served.version_name} of {served.model} has "
            f"{len(served.services)} inference services{listed}"
        )
        raise ValueError(msg)
    return ingress_url(session, served.services[0])


def unpublish(
    session: SnowflakeSession, served: ServedVersion, drop_version: bool
) -> None:
    """Drop the version's inference services and, when asked, the version.

    A model's only version is dropped with the model.

    :param session: Snowpark session.
    :param served: Version from :func:`served_version`.
    :param drop_version: Also drop the version.
    :raises ValueError: Nothing to drop, or the version is the default of
        several. Nothing is dropped then.
    """
    if not served.services and not drop_version:
        msg = (
            f"Version {served.version_name} of {served.model} has no inference "
            "service. Pass --drop-version to drop the version."
        )
        raise ValueError(msg)
    if drop_version and served.is_default and not served.only_version:
        msg = (
            f"{served.version_name} is the default version of {served.model}, and "
            "Snowflake cannot drop it. Set another default first:\n"
            f"ALTER MODEL {served.model} SET DEFAULT_VERSION = <version>"
        )
        raise ValueError(msg)
    for service in served.services:
        session.sql(f"DROP SERVICE {service}").collect()
    if not drop_version:
        return
    if served.only_version:
        session.sql(f"DROP MODEL {served.model}").collect()
    else:
        session.sql(
            f"ALTER MODEL {served.model} DROP VERSION {served.version_name}"
        ).collect()


def select_checkpoint(
    listing: list[SnowflakeRow], run_id: str, step: int | None
) -> tuple[str, int, int]:
    """Pick ``checkpoint_<n>`` whose ``merged/.complete`` is on the stage.

    :param listing: ``LIST @<runs stage>/<run_id>/`` rows.
    :param run_id: Run directory the rows are under.
    :param step: Step to publish. ``None`` takes the highest complete one.
    :return: The directory name as staged, its step, and the bytes under its ``merged/``.
    :raises ValueError: No complete merged export matches.
    """
    complete = re.compile(
        rf"/{re.escape(run_id)}/(checkpoint_(\d+))/merged/\.complete$"
    )
    found: dict[int, str] = {}
    for row in listing:
        match = complete.search(str(lowered(row)["name"]))
        if match is not None:
            found[int(match.group(2))] = match.group(1)
    if not found:
        msg = (
            f"Run {run_id} has no complete merged checkpoint. {MERGED_EXPORT_REQUIRED}"
        )
        raise ValueError(msg)
    chosen = max(found) if step is None else step
    if chosen not in found:
        msg = f"checkpoint {step} has no complete merged/ export. {MERGED_EXPORT_REQUIRED}"
        raise ValueError(msg)
    checkpoint = found[chosen]
    merged = f"/{run_id}/{checkpoint}/merged/"
    size = sum(
        int(values["size"])
        for values in map(lowered, listing)
        if merged in str(values["name"])
    )
    return checkpoint, chosen, size


def job_memory_gib(
    checkpoint: str, checkpoint_bytes: int, compute_pool: str, node_memory_gib: int
) -> tuple[int, float]:
    """Host memory the publish job reserves and may use.

    :param checkpoint: Checkpoint directory, used in the error.
    :param checkpoint_bytes: Size of its ``merged/`` files.
    :param compute_pool: Pool name, used in the error.
    :param node_memory_gib: Host memory of one pool node.
    :return: ``(request, limit)`` in GiB. The request is the weights plus 4 GiB.
    :raises ValueError: The request does not fit one node.
    """
    # The Stage Mount V2 sidecar needs 0.5 GiB of the node.
    limit_gib = node_memory_gib - 0.5
    request_gib = math.ceil(checkpoint_bytes / 2**30) + 4
    if request_gib > limit_gib:
        msg = (
            f"{checkpoint} needs about {request_gib} GiB of host memory to load, "
            f"and a {compute_pool} node has {limit_gib:g} GiB"
        )
        raise ValueError(msg)
    return request_gib, limit_gib


def registry_schema(
    session: SnowflakeSession, database: str | None, schema: str | None
) -> tuple[str, str]:
    """Database and schema that hold the model and the service.

    :param session: Snowpark session.
    :param database: ``--database``.
    :param schema: ``--schema``.
    :return: Both names, from the arguments or the session's current ones.
    :raises ValueError: Only one is set, a name is unsafe, or the session has none.
    """
    if (database is None) != (schema is None):
        msg = "Pass both --database and --schema, or neither."
        raise ValueError(msg)
    if database is None or schema is None:
        values = lowered(
            session.sql(
                "SELECT CURRENT_DATABASE() AS DATABASE_NAME, "
                "CURRENT_SCHEMA() AS SCHEMA_NAME"
            ).collect()[0]
        )
        if values["database_name"] is None or values["schema_name"] is None:
            msg = "The session has no current schema. Pass --database and --schema."
            raise ValueError(msg)
        database = str(values["database_name"])
        schema = str(values["schema_name"])
    return single_identifier(database, "database"), single_identifier(schema, "schema")


def reject_existing_version(
    session: SnowflakeSession,
    database: str,
    schema: str,
    model_name: str,
    version_name: str,
) -> None:
    """Fail when the model already has this version.

    :param session: Snowpark session.
    :param database: Registry database.
    :param schema: Registry schema.
    :param model_name: Model name.
    :param version_name: Version name.
    :raises ValueError: The version exists.
    """
    models = session.sql(
        f"SHOW MODELS LIKE '{model_name}' IN SCHEMA {database}.{schema}"
    ).collect()
    if not has_name(models, model_name):
        return
    versions = session.sql(
        f"SHOW VERSIONS LIKE '{version_name}' IN MODEL {database}.{schema}.{model_name}"
    ).collect()
    if has_name(versions, version_name):
        msg = (
            f"Model {database}.{schema}.{model_name} already has version "
            f"{version_name}. Pass another --version."
        )
        raise ValueError(msg)


def reject_existing_service(
    session: SnowflakeSession, database: str, schema: str, service_name: str
) -> None:
    """Fail when the schema already has a service of this name.

    :param session: Snowpark session.
    :param database: Service database.
    :param schema: Service schema.
    :param service_name: Service name.
    :raises ValueError: The service exists.
    """
    services = session.sql(
        f"SHOW SERVICES LIKE '{service_name}' IN SCHEMA {database}.{schema}"
    ).collect()
    if has_name(services, service_name):
        msg = (
            f"Service {database}.{schema}.{service_name} already exists. "
            "Drop it or pass another --service-name."
        )
        raise ValueError(msg)


def render_publish_spec(
    image: str, run_prefix: str, request_file: str, request_gib: int, limit_gib: float
) -> str:
    """Job service specification that runs the publish job on the run mount.

    :param image: Job image path.
    :param run_prefix: ``@stage/<run_id>``, mounted at ``/arena/run``.
    :param request_file: Request file name in the run directory.
    :param request_gib: Host memory the job reserves on the node.
    :param limit_gib: Host memory the job may use.
    :return: YAML text.
    :raises ValueError: The YAML would break the ``$$`` SQL quote.
    """
    spec = {
        "spec": {
            "containers": [
                {
                    "name": CONTAINER_NAME,
                    "image": image,
                    "command": [
                        "tini",
                        "--",
                        "python",
                        "-m",
                        "agilerl.arena.snowflake.publish_job",
                        f"{RUN_MOUNT}/{request_file}",
                    ],
                    "resources": {
                        "requests": {"memory": f"{request_gib}Gi"},
                        "limits": {"memory": f"{limit_gib:g}Gi"},
                    },
                    "volumeMounts": [{"name": "run", "mountPath": RUN_MOUNT}],
                }
            ],
            "volumes": [
                {"name": "run", "source": "stage", "stageConfig": {"name": run_prefix}}
            ],
        }
    }
    text = yaml.safe_dump(spec, sort_keys=False)
    if "$$" in text:
        msg = "service spec cannot contain $$"
        raise ValueError(msg)
    return text


def publish_job_sql(compute_pool: str, warehouse: str, name: str, spec: str) -> str:
    """``EXECUTE JOB SERVICE`` statement for the publish job.

    :param compute_pool: Pool that runs the job.
    :param warehouse: Warehouse for the job's queries.
    :param name: Job service name.
    :param spec: Specification YAML placed inside ``$$``.
    :return: The SQL statement.
    """
    pool = snowflake_identifier(compute_pool, "compute pool")
    query_warehouse = snowflake_identifier(warehouse, "warehouse")
    service = single_identifier(name, "service")
    return "\n".join(
        [
            f"EXECUTE JOB SERVICE IN COMPUTE POOL {pool}",
            f"NAME = {service}",
            "ASYNC = TRUE",
            f"QUERY_WAREHOUSE = {query_warehouse}",
            f"FROM SPECIFICATION $$\n{spec}$$",
        ]
    )


def publish_job_name(now: datetime | None = None) -> str:
    """Job service name ``ARENA_PUBLISH_<yyyymmddHHMMSS>_<HEX>``.

    :param now: Clock to stamp. ``None`` uses the current UTC time.
    :return: An unquoted Snowflake identifier.
    """
    moment = datetime.now(timezone.utc) if now is None else now
    return (
        f"ARENA_PUBLISH_{moment.strftime('%Y%m%d%H%M%S')}_"
        f"{secrets.token_hex(3).upper()}"
    )


def single_identifier(value: str, kind: str) -> str:
    """Validate one unqualified, unquoted Snowflake identifier.

    :param value: Name to check.
    :param kind: What the name is, used in the error.
    :return: The identifier with surrounding space removed.
    :raises ValueError: The name is not one identifier.
    """
    name = snowflake_identifier(value, kind)
    if "." in name:
        msg = f"{kind} must be one Snowflake identifier, got {value!r}"
        raise ValueError(msg)
    return name


def has_name(rows: list[SnowflakeRow], name: str) -> bool:
    """Whether a ``SHOW`` result lists ``name``, which Snowflake stores upper case.

    :param rows: ``SHOW ... LIKE`` rows. ``LIKE`` treats ``_`` as a wildcard.
    :param name: Unquoted identifier.
    :return: Whether a row's ``name`` is that identifier.
    """
    return any(str(lowered(row)["name"]).upper() == name.upper() for row in rows)
