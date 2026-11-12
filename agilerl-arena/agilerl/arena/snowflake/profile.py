# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Account profile, login files, and the service-user key."""

from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from agilerl.arena.snowflake import session as snowflake_session
from agilerl.arena.snowflake.source import SnowflakeSession

ROLE = "arena_trainer"
SERVICE_USER = "arena_job_user"
WAREHOUSE = "arena_wh"
RESOURCE_MONITOR = "arena_monitor"
BROKER_POOL = "arena_pulsar_pool"
HEAD_POOL = "arena_head_pool"
DATABASE = "arena"
SCHEMA = "training"
BUDGET = "arena_gpu_budget"
IMAGE_REPOSITORY = "images"
RUNS_STAGE = "runs"
FILE_FORMAT = "arena_json"
METRICS_TABLE = "arena_metrics"
HF_RULE = "hf_egress"
HF_INTEGRATION = "arena_hf_eai"
ARENA_ORG_KEY_SECRET = "arena_org_key"
ARENA_EXTERNAL_USER_SECRET = "arena_external_user_id"
ARENA_BASE_URL_SECRET = "arena_base_url"
ARENA_API_RULE = "arena_api_egress"
ARENA_API_INTEGRATION = "arena_api_eai"
ENV_RULE = "arena_env_egress"
ENV_INTEGRATION = "arena_env_eai"
ADMIN_AUTHENTICATOR = "OAUTH_AUTHORIZATION_CODE"
JOB_CONNECTION = "arena"
ADMIN_CONNECTION = "arena-admin"
SETUP_PREFIX = "@arena.training.runs/_arena"
SETUP_JSON = "@arena.training.runs/_arena/setup.json"
QUALIFIED_STAGE = "arena.training.runs"
QUALIFIED_METRICS = "arena.training.arena_metrics"
DEFAULT_BROKER_FAMILY = "CPU_X64_L"
DEFAULT_HEAD_FAMILY = "CPU_X64_SL"
CPU_POOL_MIN_NODES = 1
CPU_POOL_MAX_NODES = 4
POOL_SUSPEND_SECS = 300
DEFAULT_GPU_NODES = 2
DEFAULT_WAREHOUSE_CREDITS = 100
DEFAULT_GPU_CREDITS = 500
PRIVATE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR

NO_IMAGE = (
    "No trainer image is saved for this account yet.\n"
    "Run arena snowflake setup again. It copies the image."
)
NO_POOLS = (
    "This account's setup.json has no compute pools.\nRun arena snowflake setup again."
)


@dataclass(frozen=True)
class GpuFamilySpec:
    """One Snowflake GPU instance family.

    ``memory_gib`` is per GPU. ``host_memory_gib`` is the node's RAM.
    ``node_cap`` is Snowflake's default node limit; ``0`` means on request.
    """

    gpus: int
    memory_gib: int
    host_memory_gib: int
    node_cap: int
    gpu: str


# Snowflake's AWS, Azure, and Google Cloud instance family tables.
GPU_FAMILIES = {
    "GPU_NV_S": GpuFamilySpec(1, 24, 27, 150, "A10G"),
    "GPU_NV_M": GpuFamilySpec(4, 24, 178, 10, "A10G"),
    "GPU_NV_L": GpuFamilySpec(8, 40, 1112, 10, "A100"),
    "GPU_L40S_G1_8": GpuFamilySpec(1, 48, 58, 5, "L40S"),
    "GPU_L40S_G1_16": GpuFamilySpec(1, 48, 116, 5, "L40S"),
    "GPU_L40S_G1_48": GpuFamilySpec(4, 48, 368, 0, "L40S"),
    "GPU_L40S_G1_192": GpuFamilySpec(8, 48, 1436, 0, "L40S"),
    "GPU_R6K_G1_8": GpuFamilySpec(1, 96, 58, 5, "RTX PRO 6000"),
    "GPU_R6K_G1_16": GpuFamilySpec(1, 96, 116, 5, "RTX PRO 6000"),
    "GPU_R6K_G1_32": GpuFamilySpec(1, 96, 240, 5, "RTX PRO 6000"),
    "GPU_R6K_G1_48": GpuFamilySpec(2, 96, 490, 0, "RTX PRO 6000"),
    "GPU_R6K_G1_96": GpuFamilySpec(4, 96, 984, 0, "RTX PRO 6000"),
    "GPU_R6K_G1_192": GpuFamilySpec(8, 96, 1843, 0, "RTX PRO 6000"),
    "GPU_NV_XS": GpuFamilySpec(1, 16, 26, 10, "T4"),
    "GPU_NV_SM": GpuFamilySpec(1, 24, 424, 10, "A10"),
    "GPU_NV_2M": GpuFamilySpec(2, 24, 858, 5, "A10"),
    "GPU_NV_3M": GpuFamilySpec(2, 80, 424, 0, "A100"),
    "GPU_NV_SL": GpuFamilySpec(4, 80, 858, 0, "A100"),
    "GPU_GCP_NV_L4_1_24G": GpuFamilySpec(1, 24, 28, 10, "L4"),
    "GPU_GCP_NV_L4_4_24G": GpuFamilySpec(4, 24, 178, 10, "L4"),
    "GPU_A100_G1_12": GpuFamilySpec(1, 40, 77, 0, "A100"),
    "GPU_A100_G1_48": GpuFamilySpec(4, 40, 324, 0, "A100"),
    "GPU_GCP_NV_A100_8_40G": GpuFamilySpec(8, 40, 654, 0, "A100"),
}


@dataclass(frozen=True)
class CpuPool:
    """Broker or head pool saved in ``setup.json``."""

    pool: str
    family: str


@dataclass(frozen=True)
class GpuPool:
    """One GPU family's pool and the cap recorded by setup."""

    pool: str
    max_gpus: int
    gpus_per_node: int
    gpu_memory_gib: int


@dataclass(frozen=True)
class AccountPools:
    """Compute pools one account setup created."""

    broker: CpuPool
    head: CpuPool
    gpu: dict[str, GpuPool]


@dataclass(frozen=True)
class AccountProfile:
    """Pools and image ``setup.json`` records for the account."""

    image: str
    warehouse: str
    stage: str
    metrics_table: str
    external_access_integrations: tuple[str, ...]
    pools: AccountPools

    def as_dict(self) -> dict[str, object]:
        """Fields written to ``setup.json``. No key material."""
        return {
            "image": self.image,
            "warehouse": self.warehouse,
            "stage": self.stage,
            "metrics_table": self.metrics_table,
            "external_access_integrations": list(self.external_access_integrations),
            "pools": {
                "broker": {
                    "pool": self.pools.broker.pool,
                    "family": self.pools.broker.family,
                },
                "head": {
                    "pool": self.pools.head.pool,
                    "family": self.pools.head.family,
                },
                "gpu": {
                    family: {
                        "pool": item.pool,
                        "max_gpus": item.max_gpus,
                        "gpus_per_node": item.gpus_per_node,
                        "gpu_memory_gib": item.gpu_memory_gib,
                    }
                    for family, item in self.pools.gpu.items()
                },
            },
        }


def default_key_path() -> Path:
    """PKCS#8 key setup creates for ``arena_job_user``, beside ``connections.toml``."""
    return snowflake_session.connections_file().parent / "arena_job_user.p8"


def gpu_pool_name(family: str) -> str:
    """Compute pool that holds one GPU instance family.

    :param family: Instance family, for example ``GPU_NV_S``.
    :return: ``arena_gpu_<family>``.
    """
    return f"arena_gpu_{family}"


def trainer_image_path(tag: str) -> str:
    """In-account image path submit stores after a tag is in the repository.

    :param tag: Image tag, without a host.
    :return: ``/arena/training/images/snowflake-job:<tag>``.
    """
    return f"/{DATABASE}/{SCHEMA}/{IMAGE_REPOSITORY}/snowflake-job:{tag}"


def write_private_text(path: Path, text: str) -> None:
    """Write ``text`` and leave ``path`` readable only by its owner.

    :param path: Destination file. Parent directories are created.
    :param text: File contents.
    :raises ValueError: The disk is read-only or out of space.
    """
    _write_private(path, text.encode("utf-8"))


def write_private_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` and leave ``path`` readable only by its owner.

    :param path: Destination file.
    :param payload: File bytes.
    :raises ValueError: The disk is read-only or out of space.
    """
    _write_private(path, payload)


def _write_private(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.chmod(path, PRIVATE_FILE_MODE)
    except OSError as exc:
        message = _disk_message(exc, path)
        if message is None:
            raise
        raise ValueError(message) from exc


def _disk_message(exc: OSError, path: Path) -> str | None:
    if exc.errno == errno.ENOSPC:
        return f"The disk is out of space, so {path} was not written."
    if exc.errno == errno.EROFS:
        return f"The disk is read-only, so {path} was not written."
    return None


def generate_private_key(path: Path) -> None:
    """Write a 2048-bit PKCS#8 key with no passphrase. Mode ``0600``.

    :param path: Destination. An existing file is not replaced by the caller.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    write_private_bytes(path, pem)


def public_key_body(path: Path) -> str:
    """SubjectPublicKeyInfo body, one line, with the PEM header and footer removed.

    :param path: PKCS#8 private key.
    :return: The public key body Snowflake stores on the user.
    """
    pem = path.read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    lines = [
        line
        for line in public.decode("ascii").splitlines()
        if not line.startswith("-----")
    ]
    return "".join(lines)


def restrict_private_key(path: Path) -> None:
    """Leave an existing key readable only by its owner.

    :param path: Key file that already exists.
    :raises ValueError: The disk is read-only or out of space.
    """
    try:
        os.chmod(path, PRIVATE_FILE_MODE)
    except OSError as exc:
        message = _disk_message(exc, path)
        if message is None:
            raise
        raise ValueError(message) from exc


def write_connection(name: str, fields: dict[str, str]) -> None:
    """Add or replace one connections.toml section. The file mode is ``0600``.

    :param name: Section name.
    :param fields: Connection fields. No password is written.
    """
    # Deferred optional extra: snowflake. tomlkit ships with the connector.
    import tomlkit

    path = snowflake_session.connections_file()
    if path.is_file():
        document = tomlkit.parse(path.read_text(encoding="utf-8"))
    else:
        document = tomlkit.document()
    table = tomlkit.table()
    for key, value in fields.items():
        table.add(key, value)
    document[name] = table
    write_private_text(path, tomlkit.dumps(document))


def remove_connection(name: str) -> bool:
    """Delete one connections.toml section. Other sections stay.

    :param name: Section name.
    :return: True when the section was there.
    """
    # Deferred optional extra: snowflake. tomlkit ships with the connector.
    import tomlkit

    path = snowflake_session.connections_file()
    if not path.is_file():
        return False
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    if name not in document:
        return False
    del document[name]
    write_private_text(path, tomlkit.dumps(document))
    return True


def replaced_connection(name: str, fields: dict[str, str]) -> str | None:
    """Line warning that writing ``fields`` replaces a different ``[name]`` section.

    :param name: Section name.
    :param fields: Fields about to be written.
    :return: The warning, or ``None`` when the section is absent or identical.
    """
    section = snowflake_session.connection_section(name)
    if section is None or {key: str(value) for key, value in section.items()} == fields:
        return None
    user = section.get("user", "?")
    account = section.get("account", "?")
    path = snowflake_session.connections_file()
    return f"[{name}] in {path} signs in as {user} on {account}. It is replaced."


def job_connection_fields(account: str, key_path: Path) -> dict[str, str]:
    """``[arena]`` fields for the service user.

    :param account: Account identifier.
    :param key_path: Private key path.
    :return: Connection fields.
    """
    return {
        "account": account,
        "user": SERVICE_USER,
        "authenticator": "SNOWFLAKE_JWT",
        "private_key_file": str(key_path),
        "role": ROLE,
        "warehouse": WAREHOUSE,
        "database": DATABASE,
        "schema": SCHEMA,
    }


def admin_connection_fields(account: str, login_name: str) -> dict[str, str]:
    """``[arena-admin]`` fields. Snowflake OAuth in a browser, no password.

    The connector uses Snowflake's built-in local OAuth application when no
    client id is set. No role is set, so the token is not scoped to one role;
    setup switches to ACCOUNTADMIN after sign-in.

    :param account: Account identifier.
    :param login_name: ``LOGIN_NAME`` of a user that has ACCOUNTADMIN.
    :return: Connection fields.
    """
    return {
        "account": account,
        "user": login_name,
        "authenticator": ADMIN_AUTHENTICATOR,
    }


def profile_from_mapping(section: Mapping[str, object]) -> AccountProfile:
    """Build a profile from a setup.json object.

    :param section: Saved fields.
    :return: The profile.
    :raises ValueError: The section has no compute pools.
    """
    raw_pools = section.get("pools")
    if not isinstance(raw_pools, Mapping):
        raise ValueError(NO_POOLS)
    raw_integrations = section.get("external_access_integrations")
    names: list[str] = []
    if isinstance(raw_integrations, (list, tuple)):
        names.extend(str(item) for item in raw_integrations)
    return AccountProfile(
        image=str(section.get("image") or ""),
        warehouse=str(section["warehouse"]),
        stage=str(section["stage"]),
        metrics_table=str(section["metrics_table"]),
        external_access_integrations=tuple(names),
        pools=_pools_from(raw_pools),
    )


def _pools_from(raw: Mapping[str, object]) -> AccountPools:
    broker = raw.get("broker")
    head = raw.get("head")
    gpu = raw.get("gpu")
    if (
        not isinstance(broker, Mapping)
        or not isinstance(head, Mapping)
        or not isinstance(gpu, Mapping)
    ):
        raise ValueError(NO_POOLS)
    return AccountPools(
        broker=CpuPool(str(broker["pool"]), str(broker["family"])),
        head=CpuPool(str(head["pool"]), str(head["family"])),
        gpu={str(family): _gpu_pool(item) for family, item in gpu.items()},
    )


def _gpu_pool(raw: object) -> GpuPool:
    if not isinstance(raw, Mapping):
        raise ValueError(NO_POOLS)
    return GpuPool(
        pool=str(raw["pool"]),
        max_gpus=int(str(raw["max_gpus"])),
        gpus_per_node=int(str(raw["gpus_per_node"])),
        gpu_memory_gib=int(str(raw["gpu_memory_gib"])),
    )


def put_setup_json(session: SnowflakeSession, profile: AccountProfile) -> None:
    """Upload ``setup.json`` to the runs stage. The document has no key material.

    :param session: Snowpark session.
    :param profile: Account record.
    """
    text = json.dumps(profile.as_dict(), indent=2) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "setup.json"
        path.write_text(text, encoding="utf-8")
        session.file.put(str(path), SETUP_PREFIX, auto_compress=False, overwrite=True)


def job_connection_name(connection_name: str | None) -> str | None:
    """Connection submit, publish, status, and logs open.

    An ``[arena]`` section in the connections file selects ``arena``. Otherwise
    an omitted name stays unset and the Snowflake default connection is used.

    :param connection_name: ``--connection``, or ``None`` when omitted.
    :return: The connection name to open.
    """
    if connection_name is not None:
        return connection_name
    if snowflake_session.connection_section(JOB_CONNECTION) is not None:
        return JOB_CONNECTION
    return None


def checked_job_connection(connection_name: str | None) -> str | None:
    """:func:`job_connection_name`, refused when it cannot run a job.

    :param connection_name: ``--connection``, or ``None`` when omitted.
    :return: The connection name, or ``None`` for the Snowflake default.
    :raises ValueError: The connection is the browser admin, or its key is missing.
    """
    name = job_connection_name(connection_name)
    if name is None:
        return None
    rejection = browser_admin_rejection(name) or missing_key_rejection(name)
    if rejection is not None:
        raise ValueError(rejection)
    return name


def sole_gpu_pool(profile: AccountProfile) -> str | None:
    """GPU pool name when the profile records one GPU family.

    :param profile: Account record.
    :return: The pool, or ``None`` when there are several GPU families.
    """
    pools = [item.pool for item in profile.pools.gpu.values()]
    if len(pools) == 1:
        return pools[0]
    return None


def profile_pool_names(profile: AccountProfile) -> tuple[str, ...]:
    """Broker, head, and GPU pool names, in that order.

    :param profile: Account record.
    :return: Pool names setup created.
    """
    names = [profile.pools.broker.pool, profile.pools.head.pool]
    names.extend(item.pool for item in profile.pools.gpu.values())
    return tuple(names)


def browser_admin_rejection(connection_name: str) -> str | None:
    """Message when a job command is pointed at the browser admin connection.

    :param connection_name: Section name.
    :return: The message, or ``None`` when the connection can run a job.
    """
    section = snowflake_session.load_connection(connection_name)
    authenticator = str(section.get("authenticator", "")).strip().upper()
    if (
        connection_name == ADMIN_CONNECTION
        or authenticator in snowflake_session.BROWSER_SIGN_IN
    ):
        return (
            f"Connection {connection_name} signs in with a browser as ACCOUNTADMIN.\n"
            f"Job commands use the {JOB_CONNECTION} connection (user {SERVICE_USER})."
        )
    return None


def admin_connection_rejection(connection_name: str) -> str | None:
    """Message when an admin command is pointed at a job connection.

    :param connection_name: Section name.
    :return: The message, or ``None`` when the connection is not a job login.
    """
    section = snowflake_session.load_connection(connection_name)
    user = str(section.get("user", "")).strip() or connection_name
    role = str(section.get("role", "")).strip().upper()
    if user.upper() != SERVICE_USER.upper() and role != ROLE.upper():
        return None
    return (
        f"Connection {connection_name} signs in as user {user}.\n"
        f"Admin commands use the {ADMIN_CONNECTION} connection, or --connection "
        "with a user that has ACCOUNTADMIN."
    )


def missing_key_message(path: Path) -> str:
    """Message when a connection's private key file is not on this machine.

    :param path: Path from the connection.
    :return: The message.
    """
    return (
        f"No private key at {path}.\n"
        "The private key is not in Snowflake.\n"
        "Run arena snowflake join to sign in as your own Snowflake user."
    )


def missing_key_rejection(connection_name: str) -> str | None:
    """Message when ``connection_name`` points at a key file that is not there.

    :param connection_name: Section name.
    :return: The message, or ``None`` when the connection has no key file or the file exists.
    """
    section = snowflake_session.load_connection(connection_name)
    raw = section.get("private_key_file") or section.get("private_key_path")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_file():
        return None
    return missing_key_message(path)
