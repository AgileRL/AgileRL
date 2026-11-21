# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Copy the trainer image from public ECR and check the GPU driver and Arena from the pool."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx
from packaging.version import InvalidVersion, Version

from agilerl.arena.client_version import SNOWFLAKE_UPGRADE_COMMAND
from agilerl.arena.models import __version__
from agilerl.arena.snowflake.job import (
    arena_container_secrets,
    drop_service_sql,
    job_state,
    log_text,
    lowered,
    service_logs_sql,
    service_status_sql,
    snowflake_identifier,
)
from agilerl.arena.snowflake.license_check import RESULT_OK, RESULT_PREFIX
from agilerl.arena.snowflake.profile import ARENA_API_INTEGRATION
from agilerl.arena.snowflake.source import SnowflakeSession

PUBLIC_REGISTRY = "public.ecr.aws"
PUBLIC_REPOSITORY = "x7s7d0o8/containers/snowflake-job"
TRAINER_IMAGE = f"{PUBLIC_REGISTRY}/{PUBLIC_REPOSITORY}"
TRAINER_REPOSITORY = "snowflake-job"
IMAGE_PYTHON = "py313"
IMAGE_VARIANT = "cu130"
# Stable agilerl-arena wheels carry the ML bundle version they were released in.
BUNDLE_STAMP = Path(__file__).with_name("bundle_version.txt")
MANIFEST_TYPES = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.v2+json"
)
REGISTRY_TIMEOUT_S = 30.0
MIN_DRIVER = (580, 65, 6)
MIN_DRIVER_TEXT = "580.65.06"
DOCKER_LOGIN_USER = "0sessiontoken"
PLATFORM = "linux/amd64"
# A Snowflake registry token lasts an hour; a push that outlives it logs in again.
PUSH_ATTEMPTS = 3
DRIVER_CHECK_TIMEOUT_S = 600.0
POOL_START_TIMEOUT_S = 1800.0
# The driver-check container stays up so its log can be read while it runs.
DRIVER_HOLD_S = 120
POOL_READY_STATES = frozenset({"ACTIVE", "IDLE"})

READ_ONLY_REPOSITORY = (
    "This role cannot upload to arena.training.images. It has READ.\n"
    "Run arena snowflake setup as the account admin. That role owns the repository.\n"
    "The job role stays READ, which is what submit uses."
)
REGISTRY_TOKEN_EXPIRED = (
    "The Snowflake registry token expired during the upload, and a fresh token "
    "was rejected too.\n"
    "Run arena snowflake setup again. Docker skips the layers already uploaded."
)
MISSING_IMAGE_TAG = (
    "This agilerl-arena install is not a release build, so it has no matching "
    "trainer image. Pass --image-tag with a released version, such as 32.6.0."
)
DOCKER_MISSING = "Docker is not installed. Install Docker and run setup again."
DOCKER_NOT_RUNNING = (
    "Docker is installed but its daemon is not running. Start Docker and run "
    "setup again."
)
TOKEN_CALL_MISSING = (
    "This snowflake-connector-python has no registry token call "
    "(SnowflakeRestful._token_request). "
    "Install the tested version: pip install 'agilerl-arena[snowflake]'."
)
TOKEN_FAILED = "Failed to connect to Snowflake to retrieve a registry token."


class CommandResult(Protocol):
    """Completed process: status and captured streams."""

    @property
    def returncode(self) -> int:
        """Exit status."""

    @property
    def stdout(self) -> str:
        """Captured standard output."""

    @property
    def stderr(self) -> str:
        """Captured standard error."""


class SqlCursor(Protocol):
    """Connector cursor that can run one statement."""

    def execute(self, statement: str) -> object:
        """Run one statement."""


class TokenRest(Protocol):
    """Connector REST client that issues a registry token."""

    def _token_request(self, request_type: str) -> Mapping[str, object]:
        """Ask the session for a token."""


class RegistryConnection(Protocol):
    """Connector connection the registry token is issued from."""

    _rest: TokenRest | None
    _all_async_queries_finished: Callable[[], bool]

    def cursor(self) -> SqlCursor:
        """Open a cursor."""


class CopySession(SnowflakeSession, Protocol):
    """Snowpark session that can also issue a registry token."""

    connection: RegistryConnection


Runner = Callable[[list[str], str | None], CommandResult]


@dataclass(frozen=True)
class Finished:
    """:class:`CommandResult` from a finished Docker process."""

    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TrainerImage:
    """Released trainer image on public ECR, pinned by digest."""

    tag: str
    digest: str

    @property
    def reference(self) -> str:
        """``public.ecr.aws/.../snowflake-job@<digest>``."""
        return f"{TRAINER_IMAGE}@{self.digest}"


def trainer_image_tag(bundle_version: str) -> str:
    """Release tag ``<bundle.version>-py313-cu130``.

    :param bundle_version: ``bundle.version`` of the ``ml/v*`` release.
    :return: The image tag.
    """
    return f"{bundle_version}-{IMAGE_PYTHON}-{IMAGE_VARIANT}"


def trainer_source(bundle_version: str) -> str:
    """Public ECR image for one release, by tag.

    :param bundle_version: ``bundle.version`` of the ``ml/v*`` release.
    :return: The source image reference.
    """
    return f"{TRAINER_IMAGE}:{trainer_image_tag(bundle_version)}"


def release_bundle_version(
    image_tag: str | None = None, stamp: Path = BUNDLE_STAMP
) -> str:
    """Bundle version of the trainer image.

    :param image_tag: ``32.6.0``, ``ml/v32.6.0``, or ``32.6.0-py313-cu130``.
        ``None`` uses the version stamped into this agilerl-arena release.
    :param stamp: Stamp file a stable wheel carries.
    :return: The bundle version.
    :raises ValueError: No tag was given and this install has no stamp.
    """
    if image_tag is not None:
        return bundle_version_from_image_tag(image_tag)
    if not stamp.is_file():
        raise ValueError(MISSING_IMAGE_TAG)
    return stamp.read_text(encoding="utf-8").strip()


def bundle_version_from_image_tag(image_tag: str) -> str:
    """Bundle version from an explicit trainer tag.

    :param image_tag: ``32.6.0``, ``ml/v32.6.0``, or ``32.6.0-py313-cu130``.
    :return: The bundle version.
    """
    version = image_tag.strip().removeprefix("ml/v")
    suffix = f"-{IMAGE_PYTHON}-{IMAGE_VARIANT}"
    return version.removesuffix(suffix)


def image_bundle_version(image: str) -> Version | None:
    """Bundle version in a trainer image's ``<bundle>-py313-cu130`` tag.

    :param image: Image path or reference, ``.../snowflake-job:<tag>``.
    :return: The version, or ``None`` for a tag with no release bundle version.
    """
    tag = image.rpartition(":")[2]
    suffix = f"-{IMAGE_PYTHON}-{IMAGE_VARIANT}"
    if not tag.endswith(suffix):
        return None
    try:
        return Version(tag.removesuffix(suffix))
    except InvalidVersion:
        return None


def check_image_matches_cli(image: str, stamp: Path) -> None:
    """Stop when the trainer image is from another release than this agilerl-arena.

    Submit writes job files the image's code reads, and their layout carries
    no version. An unstamped install or a tag with no bundle version is not
    checked.

    :param image: Trainer image submit would run.
    :param stamp: Stamp file a stable wheel carries.
    :raises ValueError: The image's bundle version differs from the stamp's.
    """
    if not stamp.is_file():
        return
    image_bundle = image_bundle_version(image)
    if image_bundle is None:
        return
    cli_bundle = Version(release_bundle_version(None, stamp))
    if image_bundle == cli_bundle:
        return
    mismatch = (
        f"This agilerl-arena {__version__} goes with trainer image {cli_bundle}, "
        f"but the account's trainer image is {image_bundle}."
    )
    if cli_bundle > image_bundle:
        msg = (
            f"{mismatch} Re-run arena snowflake setup to update the trainer "
            f"image to {cli_bundle}."
        )
    else:
        msg = f"{mismatch} Run: {SNOWFLAKE_UPGRADE_COMMAND}"
    raise ValueError(msg)


def image_not_found_message(tag: str) -> str:
    """Sentence for a trainer tag public ECR does not have.

    :param tag: Image tag.
    :return: The message.
    """
    return (
        f"The trainer image {TRAINER_IMAGE}:{tag} does not exist. "
        "Pass --image-tag with a released version, such as 32.6.0."
    )


def resolve_trainer_image(
    bundle_version: str, transport: httpx.BaseTransport | None = None
) -> TrainerImage:
    """Digest of the release tag on public ECR, read anonymously.

    :param bundle_version: Release bundle version.
    :param transport: HTTP transport. ``None`` uses the network.
    :return: The tag and its digest.
    :raises ValueError: The registry is unreachable or the tag does not exist.
    """
    tag = trainer_image_tag(bundle_version)
    with httpx.Client(transport=transport, timeout=REGISTRY_TIMEOUT_S) as client:
        try:
            token = _public_pull_token(client)
            response = client.head(
                f"https://{PUBLIC_REGISTRY}/v2/{PUBLIC_REPOSITORY}/manifests/{tag}",
                headers={"Authorization": f"Bearer {token}", "Accept": MANIFEST_TYPES},
            )
        except httpx.TransportError as exc:
            msg = f"Setup could not reach {PUBLIC_REGISTRY}: {exc}"
            raise ValueError(msg) from exc
    if response.status_code == httpx.codes.NOT_FOUND:
        raise ValueError(image_not_found_message(tag))
    digest = response.headers.get("docker-content-digest", "")
    if response.status_code != httpx.codes.OK or not digest.startswith("sha256:"):
        msg = (
            f"{PUBLIC_REGISTRY} returned HTTP {response.status_code} for "
            f"{TRAINER_REPOSITORY}:{tag} without an image digest."
        )
        raise ValueError(msg)
    return TrainerImage(tag, digest)


def _public_pull_token(client: httpx.Client) -> str:
    response = client.get(
        f"https://{PUBLIC_REGISTRY}/token/",
        params={"scope": f"repository:{PUBLIC_REPOSITORY}:pull"},
    )
    if response.status_code != httpx.codes.OK:
        msg = (
            f"{PUBLIC_REGISTRY} refused an anonymous pull token "
            f"(HTTP {response.status_code})."
        )
        raise ValueError(msg)
    return str(response.json()["token"])


def run_docker(args: list[str], stdin: str | None) -> Finished:
    """Run Docker. ``stdin`` is the registry token and is not printed.

    :param args: Docker argv.
    :param stdin: Login payload, or ``None``.
    :return: The captured result.
    :raises ValueError: Docker is not installed.
    """
    try:
        completed = subprocess.run(
            args,
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(DOCKER_MISSING) from exc
    return Finished(completed.returncode, completed.stdout, completed.stderr)


def docker_preflight(run: Runner = run_docker) -> None:
    """Check that Docker is installed and its daemon answers.

    :param run: Process runner. The default is Docker.
    :raises ValueError: Docker is missing or its daemon is not running.
    """
    info = run(["docker", "info", "--format", "{{.ServerVersion}}"], None)
    if info.returncode != 0:
        raise ValueError(DOCKER_NOT_RUNNING)


def image_preflight(
    image_tag: str | None,
    run: Runner = run_docker,
    transport: httpx.BaseTransport | None = None,
) -> TrainerImage:
    """Pick the trainer tag, check Docker, and check the tag exists. Prompts nothing.

    :param image_tag: ``--image-tag``, or ``None`` for this release's tag.
    :param run: Process runner. The default is Docker.
    :param transport: HTTP transport. ``None`` uses the network.
    :return: The trainer image to pull.
    :raises ValueError: No tag, Docker is unusable, or the tag does not exist.
    """
    bundle_version = release_bundle_version(image_tag)
    docker_preflight(run)
    return resolve_trainer_image(bundle_version, transport)


def pull_images(image: TrainerImage, run: Runner = run_docker) -> None:
    """Pull the trainer image by digest. Public ECR allows anonymous pulls.

    :param image: Trainer image from :func:`image_preflight`.
    :param run: Process runner. The default is Docker.
    :raises ValueError: The pull fails.
    """
    # Snowflake runs linux/amd64. A local ARM image cannot start there.
    _require_pull(
        run(["docker", "pull", "--platform", PLATFORM, image.reference], None),
        image_not_found_message(image.tag),
        f"the trainer image {TRAINER_REPOSITORY}:{image.tag}",
    )


def _require_pull(result: CommandResult, not_found: str, label: str) -> None:
    if result.returncode == 0:
        return
    text = f"{result.stderr}\n{result.stdout}"
    _raise_for_storage(text)
    lowered_text = text.lower()
    if "manifest unknown" in lowered_text or "not found" in lowered_text:
        raise ValueError(not_found)
    msg = f"Setup could not pull {label}. {_first_line(text)}"
    raise ValueError(msg)


def registry_token(connection: RegistryConnection) -> dict[str, str | int]:
    """Registry token for ``docker login``, from the connection's session.

    :param connection: Snowflake connector connection on the open session.
    :return: ``token`` and ``expires_in`` for Docker's stdin.
    :raises ValueError: The connector has no token call, or the session cannot
        issue a token.
    """
    connection.cursor().execute(
        "alter session set PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'json'"
    )
    # The issued registry token is tied to this connection.
    connection._all_async_queries_finished = lambda: False
    rest = connection._rest
    if rest is None:
        raise ValueError(TOKEN_FAILED)
    # Same registry token request the Snowflake CLI sends for docker login.
    if not hasattr(rest, "_token_request"):
        raise ValueError(TOKEN_CALL_MISSING)
    data = rest._token_request("ISSUE").get("data")
    if not isinstance(data, dict):
        raise ValueError(TOKEN_FAILED)
    token = data.get("sessionToken")
    expires = data.get("validityInSecondsST")
    if not isinstance(token, str) or not isinstance(expires, int):
        raise ValueError(TOKEN_FAILED)
    return {"token": token, "expires_in": expires}


def repository_url(session: SnowflakeSession) -> str:
    """Registry URL of ``arena.training.images``, host included.

    :param session: Snowpark session.
    :return: ``repository_url`` from ``SHOW IMAGE REPOSITORIES``.
    :raises ValueError: The repository has no URL.
    """
    rows = session.sql(
        "SHOW IMAGE REPOSITORIES LIKE 'images' IN SCHEMA arena.training"
    ).collect()
    for row in rows:
        url = lowered(row).get("repository_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    msg = "Image repository arena.training.images has no repository URL."
    raise ValueError(msg)


def copy_images(
    session: CopySession, image: TrainerImage, run: Runner = run_docker
) -> str:
    """Push the trainer image into the account.

    Each push attempt logs in with a fresh registry token on Docker's stdin. A
    push the registry stops with an authentication error logs in again and resumes.

    :param session: Snowpark session of the role that owns the repository.
    :param image: Trainer image pulled by :func:`pull_images`.
    :param run: Process runner. The default is Docker.
    :return: The tag that was pushed, ``<bundle.version>-py313-cu130``.
    :raises ValueError: The push cannot log in, write, or fit on disk, or the
        registry keeps rejecting fresh tokens.
    """
    url = repository_url(session)
    host = url.split("/", 1)[0]

    def login() -> None:
        payload = json.dumps(registry_token(session.connection))
        _require_login(
            run(
                [
                    "docker",
                    "login",
                    "--username",
                    DOCKER_LOGIN_USER,
                    "--password-stdin",
                    host,
                ],
                payload,
            ),
            host,
        )

    _push(image.reference, f"{url}/{TRAINER_REPOSITORY}:{image.tag}", login, run)
    return image.tag


def _push(
    source: str, destination: str, login: Callable[[], None], run: Runner
) -> None:
    _require_push(run(["docker", "tag", source, destination], None))
    for _attempt in range(PUSH_ATTEMPTS):
        login()
        result = run(["docker", "push", destination], None)
        if result.returncode == 0:
            return
        if not _token_rejected(f"{result.stderr}\n{result.stdout}"):
            _require_push(result)
    raise ValueError(REGISTRY_TOKEN_EXPIRED)


def _require_login(result: CommandResult, host: str) -> None:
    if result.returncode == 0:
        return
    text = f"{result.stderr}\n{result.stdout}"
    _raise_for_storage(text)
    msg = (
        f"Setup could not log in to the Snowflake registry {host}. {_first_line(text)}"
    )
    raise ValueError(msg)


def _require_push(result: CommandResult) -> None:
    if result.returncode == 0:
        return
    text = f"{result.stderr}\n{result.stdout}"
    _raise_for_storage(text)
    if _privilege_denied(text):
        raise ValueError(READ_ONLY_REPOSITORY)
    msg = f"Setup could not upload the image to arena.training.images. {_first_line(text)}"
    raise ValueError(msg)


def _raise_for_storage(text: str) -> None:
    lowered_text = text.lower()
    if (
        "no space left" in lowered_text
        or "out of space" in lowered_text
        or "enospc" in lowered_text
    ):
        msg = "The disk is out of space, so the trainer image was not copied."
        raise ValueError(msg)
    if "read-only file system" in lowered_text:
        msg = "The disk is read-only, so the trainer image was not copied."
        raise ValueError(msg)


def _privilege_denied(text: str) -> bool:
    lowered_text = text.lower()
    return any(
        token in lowered_text
        for token in (
            "insufficient privilege",
            "requested access to the resource is denied",
            "denied:",
            " 403",
        )
    )


def _token_rejected(text: str) -> bool:
    if _privilege_denied(text):
        return False
    lowered_text = text.lower()
    return any(
        token in lowered_text
        for token in (
            "unauthorized",
            "authentication required",
            "token expired",
            "token has expired",
            " 401",
        )
    )


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "Docker failed."


def driver_new_enough(version: str) -> bool:
    """Whether ``version`` is ``580.65.06`` or newer.

    :param version: ``nvidia-smi`` driver version.
    :return: True when the driver can run the trainer image.
    """
    return _version_tuple(version) >= MIN_DRIVER


def parse_driver(log: str) -> str | None:
    """First ``major.minor.patch`` in a driver-check log, outside the licence check line.

    :param log: Container log.
    :return: The version, or ``None`` when the log has none.
    """
    driver_lines = "\n".join(
        line for line in log.splitlines() if not line.startswith(RESULT_PREFIX)
    )
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", driver_lines)
    if match is None:
        return None
    return match.group(1)


def parse_license_check(log: str) -> str | None:
    """Result the licence check printed in a driver-check log.

    :param log: Container log.
    :return: ``ok``, the licence error, or ``None`` before the check prints.
    """
    for line in log.splitlines():
        if line.startswith(RESULT_PREFIX):
            return line.removeprefix(RESULT_PREFIX).strip()
    return None


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def driver_service_name(now: datetime | None = None) -> str:
    """Job service name for one driver check.

    :param now: Clock. ``None`` uses the current UTC time.
    :return: An unquoted identifier.
    """
    moment = datetime.now(timezone.utc) if now is None else now
    return "ARENA_DRIVER_" + moment.strftime("%Y%m%d%H%M%S")


def driver_check_sql(name: str, pool: str, image: str) -> str:
    """``EXECUTE JOB SERVICE`` that prints the GPU driver and the Arena licence check, then waits.

    The trainer image sets ``NVIDIA_DRIVER_CAPABILITIES=compute,utility``, so the
    NVIDIA runtime mounts the host's ``nvidia-smi`` into the container. The
    licence check gets the head's Arena secrets and integration, so it reaches
    Arena the way the head does. The spec ``command`` replaces the image
    entrypoint.

    :param name: Service identifier.
    :param pool: GPU compute pool.
    :param image: In-account trainer image, ``/arena/training/images/snowflake-job:<tag>``.
    :return: The SQL statement.
    """
    script = (
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader; "
        "python -m agilerl.arena.snowflake.license_check; "
        f"sleep {DRIVER_HOLD_S}"
    )
    secrets = [
        line
        for secret in arena_container_secrets()
        for line in (
            f"    - snowflakeSecret: {secret['snowflakeSecret']}",
            f"      secretKeyRef: {secret['secretKeyRef']}",
            f"      envVarName: {secret['envVarName']}",
        )
    ]
    spec = "\n".join(
        (
            "spec:",
            "  containers:",
            "  - name: main",
            f"    image: {image}",
            "    command:",
            "    - sh",
            "    - -c",
            f"    - '{script}'",
            "    secrets:",
            *secrets,
            "    resources:",
            "      requests:",
            "        nvidia.com/gpu: 1",
            "      limits:",
            "        nvidia.com/gpu: 1",
        )
    )
    return "\n".join(
        (
            f"EXECUTE JOB SERVICE IN COMPUTE POOL {pool}",
            f"NAME = {name}",
            "ASYNC = TRUE",
            f"EXTERNAL_ACCESS_INTEGRATIONS = ({ARENA_API_INTEGRATION})",
            f"FROM SPECIFICATION $$\n{spec}\n$$",
        )
    )


def compute_pool_state(session: SnowflakeSession, pool: str) -> str:
    """``state`` of one compute pool, upper case.

    :param session: Snowpark session whose role can see the pool.
    :param pool: Pool name.
    :return: ``STARTING``, ``ACTIVE``, ``IDLE``, ``SUSPENDED``, and so on.
    :raises ValueError: The role cannot see the pool.
    """
    name = snowflake_identifier(pool, "compute pool")
    for row in session.sql(f"SHOW COMPUTE POOLS LIKE '{name}'").collect():
        values = lowered(row)
        if str(values.get("name", "")).upper() == name.upper():
            return str(values.get("state", "")).upper()
    msg = f"Compute pool {name} was not found."
    raise ValueError(msg)


def wait_for_pool(
    session: SnowflakeSession,
    pool: str,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = 10.0,
    clock: Callable[[], float] = time.monotonic,
    deadline_s: float = POOL_START_TIMEOUT_S,
) -> None:
    """Wait until the pool is ``ACTIVE`` or ``IDLE``.

    :param session: Snowpark session whose role can see the pool.
    :param pool: Pool name.
    :param sleep: Called between polls.
    :param poll_s: Seconds between polls.
    :param clock: Monotonic clock.
    :param deadline_s: Seconds to wait.
    :raises ValueError: The pool is not ready by the deadline.
    """
    started = clock()
    while True:
        state = compute_pool_state(session, pool)
        if state in POOL_READY_STATES:
            return
        if clock() - started >= deadline_s:
            minutes = round(deadline_s / 60)
            msg = f"Compute pool {pool} is still {state} after {minutes} minutes."
            raise ValueError(msg)
        sleep(poll_s)


def run_driver_check(
    session: SnowflakeSession,
    pool: str,
    image: str,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = 10.0,
    now: datetime | None = None,
    clock: Callable[[], float] = time.monotonic,
    deadline_s: float = DRIVER_CHECK_TIMEOUT_S,
    pool_deadline_s: float = POOL_START_TIMEOUT_S,
) -> str:
    """Check the GPU driver and Arena from the trainer image on the pool.

    One job runs ``nvidia-smi`` and the head's Arena licence check. The job
    starts the pool. The deadline starts once the pool is ``ACTIVE`` or
    ``IDLE``, so it covers the image pull. The service is force-dropped however
    the check ends.

    :param session: Session as ``arena_trainer``. ACCOUNTADMIN cannot own services.
    :param pool: GPU compute pool.
    :param image: In-account trainer image path.
    :param sleep: Called between polls.
    :param poll_s: Seconds between polls.
    :param now: Clock for the service name.
    :param clock: Monotonic clock.
    :param deadline_s: Seconds to wait for both results once the pool is ready.
    :param pool_deadline_s: Seconds to wait for the pool to start.
    :return: The driver version.
    :raises ValueError: The pool does not start, the job reports no version,
        the driver is too old, Arena refused or could not be reached from
        Snowflake, the job fails, or it does not finish.
    """
    name = driver_service_name(now)
    try:
        session.sql(driver_check_sql(name, pool, image)).collect()
        wait_for_pool(session, pool, sleep, poll_s, clock, pool_deadline_s)
        return _read_check(session, name, sleep, poll_s, clock, deadline_s)
    finally:
        session.sql(drop_service_sql(name, if_exists=True)).collect()


def _read_check(
    session: SnowflakeSession,
    name: str,
    sleep: Callable[[float], None],
    poll_s: float,
    clock: Callable[[], float],
    deadline_s: float,
) -> str:
    started = clock()
    while True:
        containers = [
            lowered(row) for row in session.sql(service_status_sql(name)).collect()
        ]
        statuses = {str(values["service_status"]).upper() for values in containers}
        state = job_state(statuses)
        logs = ""
        if state != "running" or "RUNNING" in statuses:
            logs = log_text(list(session.sql(service_logs_sql(name, 0)).collect()))
        found = parse_driver(logs)
        arena = parse_license_check(logs)
        if found is not None and not driver_new_enough(found):
            msg = f"The GPU driver is {found}. Arena needs {MIN_DRIVER_TEXT} or newer."
            raise ValueError(msg)
        if arena is not None and arena != RESULT_OK:
            msg = f"The Arena check from inside Snowflake failed: {arena}"
            raise ValueError(msg)
        if state == "failed":
            msg = f"The GPU driver check failed. {_status_detail(containers)}"
            raise ValueError(msg)
        if found is not None and arena == RESULT_OK:
            return found
        if state == "done" and found is None:
            msg = (
                "The GPU driver check did not report a driver version. "
                f"{_status_detail(containers)}"
            )
            raise ValueError(msg)
        if state == "done":
            msg = (
                "The Arena check from inside Snowflake did not report a result. "
                f"Last log line: {_last_line(logs)}"
            )
            raise ValueError(msg)
        if clock() - started >= deadline_s:
            minutes = round(deadline_s / 60)
            msg = (
                f"The GPU driver check did not finish in {minutes} minutes. "
                f"{_status_detail(containers)}"
            )
            raise ValueError(msg)
        sleep(poll_s)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "the log is empty."


def _status_detail(containers: list[dict[str, object]]) -> str:
    if not containers:
        return "Snowflake lists no container for the service."
    parts = [
        f"service {values.get('service_status', '')}, "
        f"container {values.get('status', '')}: {values.get('message', '')}"
        for values in containers
    ]
    return "Status: " + "; ".join(parts) + "."
