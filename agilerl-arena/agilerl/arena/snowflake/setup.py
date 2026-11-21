# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Check the account and Arena, confirm once, then create the objects and check the driver and Arena from the pool."""

from __future__ import annotations

import math
import re
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from agilerl.arena.snowflake.catalog import (
    FOREIGN_DATABASE,
    GpuFamily,
    PoolState,
    active_event_table,
    budget_limit,
    collect,
    event_table_warning,
    existing_integrations,
    existing_pools,
    is_foreign_database,
    linked_resources,
    load_families,
    monitor_quota,
    one_line,
    remote_profile,
    require_accountadmin,
    secret_names,
    service_user_key,
)
from agilerl.arena.snowflake.images import (
    copy_images,
    image_preflight,
    pull_images,
    run_driver_check,
)
from agilerl.arena.snowflake.license import (
    DEFAULT_BASE_URL,
    SNOWFLAKE_DENIED,
    SnowflakeLicenseError,
    fetch_capabilities,
    partner_headers,
    snowflake_training_enabled,
)
from agilerl.arena.snowflake.profile import (
    ARENA_API_INTEGRATION,
    ARENA_BASE_URL_SECRET,
    ARENA_EXTERNAL_USER_SECRET,
    ARENA_ORG_KEY_SECRET,
    BROKER_POOL,
    CPU_POOL_MAX_NODES,
    CPU_POOL_MIN_NODES,
    DATABASE,
    DEFAULT_BROKER_FAMILY,
    DEFAULT_GPU_CREDITS,
    DEFAULT_GPU_NODES,
    DEFAULT_HEAD_FAMILY,
    DEFAULT_WAREHOUSE_CREDITS,
    GPU_FAMILIES,
    HEAD_POOL,
    HF_INTEGRATION,
    IMAGE_REPOSITORY,
    JOB_CONNECTION,
    QUALIFIED_METRICS,
    QUALIFIED_STAGE,
    ROLE,
    SCHEMA,
    SERVICE_USER,
    WAREHOUSE,
    AccountPools,
    AccountProfile,
    CpuPool,
    GpuPool,
    default_key_path,
    generate_private_key,
    gpu_pool_name,
    job_connection_fields,
    public_key_body,
    put_setup_json,
    replaced_connection,
    restrict_private_key,
    trainer_image_path,
    write_connection,
)
from agilerl.arena.snowflake.session import connection_section, open_session
from agilerl.arena.snowflake.setup_sql import (
    ArenaCredentials,
    PoolPlan,
    SetupPlan,
    keys_match,
    probe_statements,
    script,
    setup_statements,
)
from agilerl.arena.snowflake.source import SnowflakeSession

TRIAL_ERROR = 509009
TRIAL_MESSAGE = (
    "Snowpark Container Services is not available on trial accounts.\n"
    "Arena needs compute pools and external access integrations "
    "(Snowflake error 509009).\n"
    "Add a payment method to the account, then run arena snowflake setup again.\n"
    "Nothing was created."
)
ORG_KEY_PREFIX = "arena_org_"
ARENA_SECRETS = frozenset(
    name.upper()
    for name in (
        ARENA_ORG_KEY_SECRET,
        ARENA_EXTERNAL_USER_SECRET,
        ARENA_BASE_URL_SECRET,
    )
)
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PRINTED_ORG_KEY = "<arena org key>"
PRINTED_EXTERNAL_USER = "<arena external user id>"
PRINTED_PUBLIC_KEY = "<public key of the arena_job_user private key>"


class SetupIO(Protocol):
    """Prompts and lines for ``arena snowflake setup``."""

    def echo(self, text: str) -> None:
        """Print a line."""

    def ask(self, text: str, default: str | None, hide: bool) -> str:
        """Read one answer. ``default`` is returned when the answer is empty."""


@dataclass(frozen=True)
class SetupRequest:
    """Flags for one setup. ``None`` means a saved value, then a prompt or default."""

    account: str
    broker_family: str | None
    head_family: str | None
    gpu_families: tuple[str, ...]
    max_gpus: int | None
    warehouse_credits: int | None
    gpu_credits: int | None
    key_path: Path | None
    rotate_key: bool
    yes: bool
    notify_email: str | None = None
    org_key: str | None = None
    external_user_id: str | None = None
    base_url: str = DEFAULT_BASE_URL
    image_tag: str | None = None


@dataclass(frozen=True)
class AccountState:
    """What the account already has. Read before any prompt."""

    families: list[GpuFamily]
    pools: dict[str, PoolState]
    integrations: frozenset[str]
    secrets: frozenset[str]
    profile: AccountProfile | None
    stored_key: str | None
    linked: str
    warehouse_credits: int | None
    budget_credits: int | None
    event_table: str


def run_setup(session: SnowflakeSession, request: SetupRequest, io: SetupIO) -> None:
    """Check, confirm, create the account objects, push the image, and check the driver and Arena from the pool.

    Nothing is written, generated, or pulled before the confirmation.

    :param session: Account-admin session.
    :param request: Flags. Missing values come from the account, then ``io``.
    :param io: Prompts and output.
    :raises ValueError: Setup stopped. The message is the text to print.
    :raises SnowflakeLicenseError: Arena rejected the credentials.
    """
    require_accountadmin(session)
    image = image_preflight(request.image_tag)
    state = read_account(session)
    probe_external_access(session, state.integrations)
    arena = resolve_arena(request, state, io)
    plans = resolve_plans(request, state)
    warehouse_credits, budget_credits = resolve_credits(request, state, io)
    notify_email = resolve_notify_email(request, io)
    key_path = resolve_key(request)
    check_existing_key(key_path, state.stored_key, request.rotate_key)
    job_fields = job_connection_fields(request.account, key_path)
    io.echo(
        format_summary(
            account=request.account,
            key_path=key_path,
            plans=plans,
            credits=(warehouse_credits, budget_credits),
            notify_email=notify_email,
            arena=arena,
            replaced=replaced_connection(JOB_CONNECTION, job_fields),
            event_table=state.event_table,
        )
    )
    confirm(request, io)
    if key_path.is_file():
        restrict_private_key(key_path)
    else:
        generate_private_key(key_path)
    pull_images(image)
    plan = SetupPlan(
        pools=tuple(plans),
        existing_max_nodes={name: pool.max_nodes for name, pool in state.pools.items()},
        warehouse_credits=warehouse_credits,
        budget_credits=budget_credits,
        notify_email=notify_email,
        public_key=public_key_body(key_path),
        stored_key=state.stored_key,
        rotate_key=request.rotate_key,
        arena=arena,
        integrations=state.integrations,
        linked=state.linked,
    )
    for statement in setup_statements(plan):
        collect(session, statement)
    write_connection(JOB_CONNECTION, job_fields)
    profile = account_profile(plans, "", state.profile)
    put_setup_json(session, profile)
    trainer = trainer_image_path(copy_images(session, image))
    put_setup_json(session, replace(profile, image=trainer))
    gpu_pool = next(plan.name for plan in plans if plan.role == "gpu")
    with closing(open_session(JOB_CONNECTION)) as job_session:
        run_driver_check(job_session, gpu_pool, trainer)
    io.echo("The account is ready.")
    io.echo("Next: arena snowflake invite USER for each data scientist.")


def read_account(session: SnowflakeSession) -> AccountState:
    """Read families, pools, integrations, secrets, setup.json, quotas, and the event table.

    :param session: ACCOUNTADMIN session.
    :return: The account state.
    :raises ValueError: A database named ``arena`` belongs to something else.
    """
    families = load_families(session)
    if is_foreign_database(session):
        raise ValueError(FOREIGN_DATABASE)
    return AccountState(
        families=families,
        pools=existing_pools(session),
        integrations=frozenset(existing_integrations(session)),
        secrets=frozenset(secret_names(session)),
        profile=remote_profile(session),
        stored_key=service_user_key(session),
        linked=linked_resources(session),
        warehouse_credits=monitor_quota(session),
        budget_credits=budget_limit(session),
        event_table=active_event_table(session),
    )


def probe_external_access(
    session: SnowflakeSession, integrations: frozenset[str]
) -> None:
    """Create and drop a disabled integration to catch trial accounts early.

    :param session: ACCOUNTADMIN session.
    :param integrations: Upper-case names of integrations already created.
    :raises ValueError: The account cannot create external access integrations.
    """
    if HF_INTEGRATION.upper() in integrations:
        return
    create, drop = probe_statements()
    try:
        collect(session, create)
    except Exception as exc:
        raise ValueError(integration_failure_message(exc)) from exc
    collect(session, drop)


def integration_failure_message(exc: Exception) -> str:
    """Trial or external-access message for a failed integration statement.

    :param exc: The server error.
    :return: The message.
    """
    if is_trial(exc):
        return TRIAL_MESSAGE
    return (
        "This account cannot create external access integrations, so Arena "
        "cannot reach Hugging Face.\n"
        f"Snowflake said: {one_line(exc)}\n"
        "An ACCOUNTADMIN has to enable external access for the account. "
        "Then run arena snowflake setup again.\n"
        "Nothing was created."
    )


def is_trial(exc: Exception) -> bool:
    """Whether ``exc`` is Snowflake error 509009.

    :param exc: Server error.
    :return: True for a trial account.
    """
    if getattr(exc, "errno", None) == TRIAL_ERROR:
        return True
    text = str(exc)
    return "509009" in text or "not supported for trial accounts" in text


def resolve_arena(
    request: SetupRequest, state: AccountState, io: SetupIO
) -> ArenaCredentials | None:
    """Arena credentials, checked against Arena before anything is created.

    :param request: Setup flags.
    :param state: Account state.
    :param io: Prompts.
    :return: Values to store, or ``None`` when the stored secrets stay.
    :raises ValueError: A value is missing or is not an org key.
    :raises SnowflakeLicenseError: Arena rejected the credentials or the
        organisation has no Snowflake training.
    """
    org_key, external_user_id = request.org_key, request.external_user_id
    stored = (
        ARENA_SECRETS <= state.secrets
        and ARENA_API_INTEGRATION.upper() in state.integrations
    )
    if org_key is None and external_user_id is None and stored:
        return None
    if request.yes and (org_key is None or external_user_id is None):
        missing = [
            flag
            for flag, value in (
                ("--org-key", org_key),
                ("--external-user-id", external_user_id),
            )
            if value is None
        ]
        msg = f"Pass {' and '.join(missing)}.\nNothing was created."
        raise ValueError(msg)
    org_key = _credential_value(org_key, io, "Arena org key")
    external_user_id = _credential_value(external_user_id, io, "Arena external user id")
    if not org_key.startswith(ORG_KEY_PREFIX):
        msg = (
            "That is not an Arena org key.\n"
            f"Ask AgileRL for a Snowflake org key (it starts with {ORG_KEY_PREFIX}).\n"
            "Nothing was created."
        )
        raise ValueError(msg)
    headers = partner_headers(org_key, external_user_id)
    if not snowflake_training_enabled(fetch_capabilities(request.base_url, headers)):
        raise SnowflakeLicenseError(SNOWFLAKE_DENIED)
    return ArenaCredentials(org_key, external_user_id, request.base_url)


def _credential_value(value: str | None, io: SetupIO, prompt: str) -> str:
    if value is None:
        value = io.ask(prompt, None, True)
    text = value.strip()
    if text == "":
        msg = f"{prompt} is required.\nNothing was created."
        raise ValueError(msg)
    return text


def resolve_plans(request: SetupRequest, state: AccountState) -> list[PoolPlan]:
    """Broker, head, and GPU pools. Flags win, then setup.json, then defaults.

    :param request: Setup flags.
    :param state: Account state.
    :return: Plans in create order.
    :raises ValueError: A family is not in this region, or an existing pool
        uses another family.
    """
    saved = state.profile
    broker = request.broker_family or (
        saved.pools.broker.family if saved else DEFAULT_BROKER_FAMILY
    )
    head = request.head_family or (
        saved.pools.head.family if saved else DEFAULT_HEAD_FAMILY
    )
    saved_gpus = saved.pools.gpu if saved else {}
    if request.gpu_families:
        names = tuple(dict.fromkeys(name.upper() for name in request.gpu_families))
    elif saved_gpus:
        names = tuple(saved_gpus)
    else:
        names = (smallest_gpu_family(state.families),)
    gpu_plans = []
    for name in names:
        saved_cap = saved_gpus[name].max_gpus if name in saved_gpus else None
        cap = request.max_gpus if request.max_gpus is not None else saved_cap
        gpu_plans.append(_gpu_plan(name, cap, state.families))
    plans = [
        _cpu_plan(BROKER_POOL, broker, "broker", state.families),
        _cpu_plan(HEAD_POOL, head, "head", state.families),
        *gpu_plans,
    ]
    for plan in plans:
        pool = state.pools.get(plan.name.upper())
        if pool is not None and pool.family.upper() != plan.family.upper():
            raise ValueError(pool_family_message(plan.name, pool.family))
    return plans


def smallest_gpu_family(families: list[GpuFamily]) -> str:
    """Known GPU family in ``families`` with the fewest GPUs, then the least memory.

    :param families: Families in this region.
    :return: The family name.
    :raises ValueError: This account listed no known GPU family.
    """
    present = sorted(
        (spec.gpus, spec.memory_gib, item.name.upper())
        for item in families
        if (spec := GPU_FAMILIES.get(item.name.upper())) is not None
    )
    if not present:
        raise ValueError(no_default_gpu_message(families))
    return present[0][2]


def nodes_for_gpus(max_gpus: int, gpus_per_node: int) -> int:
    """Nodes needed to hold ``max_gpus`` on a family.

    :param max_gpus: GPU cap.
    :param gpus_per_node: GPUs on one node of the family.
    :return: ``ceil(max_gpus / gpus_per_node)``.
    """
    return math.ceil(max_gpus / gpus_per_node)


def family_listed(family: str, families: list[GpuFamily]) -> bool:
    """Whether ``SHOW COMPUTE POOL INSTANCE FAMILIES`` returned ``family``.

    :param family: Requested instance family.
    :param families: Families in this region.
    :return: True when the name is present.
    """
    return any(item.name.upper() == family.upper() for item in families)


def _cpu_plan(pool: str, family: str, role: str, families: list[GpuFamily]) -> PoolPlan:
    name = family.upper()
    if not family_listed(name, families):
        raise ValueError(missing_family_message(name, role))
    return PoolPlan(pool, name, role, CPU_POOL_MIN_NODES, CPU_POOL_MAX_NODES, None)


def _gpu_plan(family: str, max_gpus: int | None, families: list[GpuFamily]) -> PoolPlan:
    spec = GPU_FAMILIES.get(family)
    if spec is None:
        raise ValueError(unsupported_gpu_message(family))
    if not family_listed(family, families):
        raise ValueError(no_gpu_message(family, families))
    gpus = max_gpus if max_gpus is not None else spec.gpus * DEFAULT_GPU_NODES
    return PoolPlan(
        gpu_pool_name(family), family, "gpu", 1, nodes_for_gpus(gpus, spec.gpus), gpus
    )


def resolve_credits(
    request: SetupRequest, state: AccountState, io: SetupIO
) -> tuple[int, int]:
    """Warehouse quota and pool budget. Flags win, then the account, then a prompt.

    :param request: Setup flags.
    :param state: Account state.
    :param io: Prompts.
    :return: Warehouse credits, then pool budget credits.
    """
    warehouse = _credit(
        request.warehouse_credits,
        state.warehouse_credits,
        DEFAULT_WAREHOUSE_CREDITS,
        "Warehouse credit cap per month",
        request.yes,
        io,
    )
    budget = _credit(
        request.gpu_credits,
        state.budget_credits,
        DEFAULT_GPU_CREDITS,
        "Compute pool credit budget per month",
        request.yes,
        io,
    )
    return warehouse, budget


def _credit(
    flag: int | None,
    saved: int | None,
    default: int,
    prompt: str,
    yes: bool,
    io: SetupIO,
) -> int:
    if flag is not None:
        return flag
    if saved is not None:
        return saved
    if yes:
        return default
    while True:
        answer = io.ask(f"{prompt} [{default}]", str(default), False).strip()
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        io.echo("Enter a whole number greater than 0.")


def resolve_notify_email(request: SetupRequest, io: SetupIO) -> str | None:
    """Address the budget emails at 80% and 100%. Optional.

    :param request: Setup flags.
    :param io: Prompts.
    :return: The address, or ``None`` to leave budget notifications as they are.
    :raises ValueError: The value is not an email address.
    """
    value = request.notify_email
    if value is None and not request.yes:
        value = io.ask("Budget alert email (Enter to skip)", "", False).strip() or None
    if value is not None and EMAIL.fullmatch(value) is None:
        msg = f"{value} is not an email address.\nNothing was created."
        raise ValueError(msg)
    return value


def resolve_key(request: SetupRequest) -> Path:
    """Private key for ``arena_job_user``: ``--key``, then ``[arena]``, then the default.

    :param request: Setup flags.
    :return: The key path. A missing non-explicit path is created after confirmation.
    :raises ValueError: ``--key`` names a file that is not there.
    """
    if request.key_path is not None:
        if not request.key_path.is_file():
            raise ValueError(missing_setup_key_message(request.key_path))
        return request.key_path
    section = connection_section(JOB_CONNECTION) or {}
    saved = section.get("private_key_file")
    if saved and str(section.get("user", "")).lower() == SERVICE_USER:
        return Path(str(saved)).expanduser()
    return default_key_path()


def check_existing_key(path: Path, stored: str | None, rotate_key: bool) -> None:
    """Stop when ``arena_job_user`` has a key the local file does not match.

    :param path: Local private key. May not exist yet.
    :param stored: Key on the user, ``None`` when the user is new.
    :param rotate_key: Replace the user's key.
    :raises ValueError: The keys differ and ``rotate_key`` is off.
    """
    if stored in (None, "") or rotate_key:
        return
    if not path.is_file() or not keys_match(stored, public_key_body(path)):
        raise ValueError(key_mismatch_message(path))


def confirm(request: SetupRequest, io: SetupIO) -> None:
    """Ask once. Enter means yes.

    :param request: Setup flags.
    :param io: Prompts.
    :raises ValueError: The answer was no.
    """
    if request.yes:
        return
    answer = io.ask("Create these objects? [Y/n]", "", False).strip().lower()
    if answer not in {"", "y", "yes"}:
        msg = "Nothing was created."
        raise ValueError(msg)


def format_summary(
    account: str,
    key_path: Path,
    plans: list[PoolPlan],
    credits: tuple[int, int],
    notify_email: str | None,
    arena: ArenaCredentials | None,
    replaced: str | None,
    event_table: str,
) -> str:
    """Block printed before the confirmation.

    :param account: Account identifier.
    :param key_path: Service-user private key.
    :param plans: Pools this setup will create or resize.
    :param credits: Warehouse quota, then pool budget.
    :param notify_email: Budget alert address, or ``None``.
    :param arena: Arena credentials to store, or ``None`` to keep them.
    :param replaced: Warning that ``[arena]`` is replaced, or ``None``.
    :param event_table: The account's active event table, or ``""``.
    :return: The summary, without the question.
    """

    def line(label: str, value: str) -> str:
        return f"  {label:<16}{value}"

    key_state = "existing" if key_path.is_file() else "created"
    lines = [
        f"Account {account} as ACCOUNTADMIN",
        "",
        line("role", ROLE),
        line("user", SERVICE_USER),
        line("key", f"{key_path} ({key_state})"),
        line("warehouse", f"{WAREHOUSE} (X-Small, suspends after 60 seconds)"),
        line(
            "warehouse cap",
            f"{credits[0]} credits/month (notify at 80%, suspend the warehouse at 100%)",
        ),
        *(line(plan.role, pool_summary(plan)) for plan in plans),
        line(
            "pool budget",
            f"{credits[1]} credits/month for all Arena pools "
            "(notification only; the idle suspend stops the pools)",
        ),
        line("budget email", notify_email or "none (pass --notify-email for alerts)"),
        line("database", f"{DATABASE}.{SCHEMA}"),
        line("image repo", f"{DATABASE}.{SCHEMA}.{IMAGE_REPOSITORY} ({ROLE}: READ)"),
        line("stage", QUALIFIED_STAGE),
        line("metrics", QUALIFIED_METRICS),
        line("egress", "Hugging Face and the Arena API on 443"),
        line("arena", arena.base_url if arena else "keep the stored org key"),
    ]
    if any(plan.role == "gpu" and plan.max_nodes == 1 for plan in plans):
        lines.append(
            "  Max nodes is 1. A publish that runs beside training needs a second node."
        )
    if replaced is not None:
        lines.append(f"  {replaced}")
    warning = event_table_warning(event_table)
    if warning is not None:
        lines.append(f"  {warning}")
    lines.extend(["", "Checkpoints and metrics stay in this account."])
    return "\n".join(lines)


def pool_summary(plan: PoolPlan) -> str:
    """One pool line: name, family, min nodes, and max nodes.

    :param plan: Pool setup will create.
    :return: The summary value.
    """
    return f"{plan.name}  {plan.family}  min {plan.min_nodes}  max {plan.max_nodes}"


def print_setup_sql(request: SetupRequest) -> str:
    """Worksheet of the statements setup runs on a fresh account. Creates nothing.

    :param request: Setup flags. No session, Docker, or Arena call is made.
    :return: The worksheet. Secret values and the public key are placeholders.
    :raises ValueError: A GPU family is not one Arena sizes.
    """
    names = tuple(dict.fromkeys(name.upper() for name in request.gpu_families)) or (
        "GPU_NV_S",
    )
    broker = (request.broker_family or DEFAULT_BROKER_FAMILY).upper()
    head = (request.head_family or DEFAULT_HEAD_FAMILY).upper()
    plans = [
        PoolPlan(
            BROKER_POOL, broker, "broker", CPU_POOL_MIN_NODES, CPU_POOL_MAX_NODES, None
        ),
        PoolPlan(HEAD_POOL, head, "head", CPU_POOL_MIN_NODES, CPU_POOL_MAX_NODES, None),
    ]
    for name in names:
        spec = GPU_FAMILIES.get(name)
        if spec is None:
            raise ValueError(unsupported_gpu_message(name))
        gpus = request.max_gpus or spec.gpus * DEFAULT_GPU_NODES
        plans.append(
            PoolPlan(
                gpu_pool_name(name),
                name,
                "gpu",
                1,
                nodes_for_gpus(gpus, spec.gpus),
                gpus,
            )
        )
    plan = SetupPlan(
        pools=tuple(plans),
        existing_max_nodes={},
        warehouse_credits=request.warehouse_credits or DEFAULT_WAREHOUSE_CREDITS,
        budget_credits=request.gpu_credits or DEFAULT_GPU_CREDITS,
        notify_email=request.notify_email,
        public_key=PRINTED_PUBLIC_KEY,
        stored_key=None,
        rotate_key=False,
        arena=ArenaCredentials(
            PRINTED_ORG_KEY, PRINTED_EXTERNAL_USER, request.base_url
        ),
        integrations=frozenset(),
        linked="",
    )
    header = (
        "-- Statements arena snowflake setup runs on a fresh account.\n"
        "-- Replace the <...> values. Setup also pushes the trainer image,\n"
        "-- uploads setup.json, writes the [arena] connection, and runs a job\n"
        "-- on the GPU pool that checks the driver and that Arena answers.\n"
    )
    return header + "\n" + script(["USE ROLE ACCOUNTADMIN", *setup_statements(plan)])


def format_family_lines(families: list[GpuFamily]) -> str:
    """One family per line, or ``(none)``.

    :param families: Families returned for this region.
    :return: The block used in the no-GPU message.
    """
    if not families:
        return "  (none)"
    return "\n".join(
        f"  {item.name}  {item.gpu_count}x {item.gpu or 'GPU'}" for item in families
    )


def no_gpu_message(family: str, families: list[GpuFamily]) -> str:
    """Message when the requested family is not a GPU pool in this region.

    :param family: Requested family. A key of :data:`GPU_FAMILIES`.
    :param families: Families the account returned.
    :return: The message.
    """
    spec = GPU_FAMILIES[family]
    gpu_families = [item for item in families if item.gpu_count > 0]
    return (
        f"This account has no {family} pool in this region "
        f"({spec.gpus}x {spec.gpu}). Arena did not create a compute pool.\n"
        "GPU families SHOW COMPUTE POOL INSTANCE FAMILIES returned:\n"
        f"{format_family_lines(gpu_families)}\n"
        "Rerun with one of those names:\n"
        "  arena snowflake setup --gpu-family <name>\n"
        "A CPU pool cannot run this trainer.\n"
        "Nothing was created."
    )


def no_default_gpu_message(families: list[GpuFamily]) -> str:
    """Message when the account listed no GPU family Arena sizes.

    :param families: Families the account returned.
    :return: The message.
    """
    gpu_families = [item for item in families if item.gpu_count > 0]
    return (
        "This account has no GPU family Arena sizes in this region.\n"
        "GPU families SHOW COMPUTE POOL INSTANCE FAMILIES returned:\n"
        f"{format_family_lines(gpu_families)}\n"
        "Nothing was created."
    )


def missing_family_message(family: str, role: str) -> str:
    """Message when a CPU family is not in this region.

    :param family: Requested instance family.
    :param role: ``broker`` or ``head``.
    :return: The message.
    """
    return (
        f"This account has no {family} instance family in this region.\n"
        f"The {role} pool was not created.\n"
        "Nothing was created."
    )


def unsupported_gpu_message(family: str) -> str:
    """Message when Arena has no GPU count for ``family``.

    :param family: Requested instance family.
    :return: The message.
    """
    return (
        f"{family} is not a GPU family Arena sizes.\n"
        f"Known families: {', '.join(GPU_FAMILIES)}.\n"
        "Nothing was created."
    )


def pool_family_message(pool: str, current: str) -> str:
    """Message when an existing pool uses another family.

    :param pool: Compute pool name.
    :param current: Family the pool already uses.
    :return: The message.
    """
    return (
        f"Pool {pool} is {current}.\n"
        "Setup will not replace a pool.\n"
        "Drop it when nothing is running, then rerun setup."
    )


def missing_setup_key_message(path: Path) -> str:
    """Message when ``--key`` names a file that is not there.

    :param path: The path that was passed.
    :return: The message.
    """
    return (
        f"No private key at {path}.\n"
        f"Leave --key unset and setup will create {default_key_path()}, "
        "or pass the path of an existing PKCS#8 key.\n"
        "Nothing was created."
    )


def key_mismatch_message(path: Path) -> str:
    """Message when the service user key does not match the local file.

    :param path: Local private key.
    :return: The message.
    """
    return (
        f"User {SERVICE_USER} already has a public key, and it does not match {path}.\n"
        "Setup left the user as it is.\n"
        "Pass the matching private key with --key, or run "
        "arena snowflake setup --rotate-key to replace it."
    )


def account_profile(
    plans: list[PoolPlan], image: str, saved: AccountProfile | None
) -> AccountProfile:
    """Profile written by setup.

    :param plans: Pools this setup created.
    :param image: Trainer image path, or ``""`` before the copy finishes.
    :param saved: Profile already in setup.json. Its extra integrations stay.
    :return: The profile.
    """
    broker = next(plan for plan in plans if plan.role == "broker")
    head = next(plan for plan in plans if plan.role == "head")
    gpu: dict[str, GpuPool] = {}
    for plan in plans:
        if plan.role != "gpu" or plan.max_gpus is None:
            continue
        spec = GPU_FAMILIES[plan.family]
        gpu[plan.family] = GpuPool(plan.name, plan.max_gpus, spec.gpus, spec.memory_gib)
    extra = saved.external_access_integrations if saved else ()
    integrations = tuple(dict.fromkeys((HF_INTEGRATION, *extra)))
    return AccountProfile(
        image=image,
        warehouse=WAREHOUSE,
        stage=QUALIFIED_STAGE,
        metrics_table=QUALIFIED_METRICS,
        external_access_integrations=integrations,
        pools=AccountPools(
            broker=CpuPool(broker.name, broker.family),
            head=CpuPool(head.name, head.family),
            gpu=gpu,
        ),
    )
