# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The Arena org key of the Snowflake Native App: connect, rotation, and the saved secrets.

``connect_arena`` gets a key with no person seeing it. Arena writes a one-time code
to ``claims.challenges`` in the application package, and only this app,
installed in that account, can read the row for its own
``CURRENT_REGION()`` and ``CURRENT_ACCOUNT()``. Sending the code back proves
the claim; the admin then opens Arena's signup link, and the app collects the
key into its secrets.

``arena_app`` imports this module, which imports ``app_billing`` and
``app_trial``. The setup script lists all four files in each procedure's
``IMPORTS``, which puts them on the path side by side.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app_billing import (
    saved_partner,
    saved_secret,
    served_models,
    service_live,
    service_statuses,
    sync_billing_task,
)
from app_trial import refuse_ended_trial
from snowflake.snowpark.exceptions import SnowparkSQLException

from agilerl.snowflake.catalog import collect
from agilerl.snowflake.job import lowered
from agilerl.snowflake.license import (
    DEFAULT_BASE_URL,
    SNOWFLAKE_DENIED,
    AppKey,
    AuthRejectedError,
    ClaimAccount,
    ClaimClosedError,
    ClaimLinkedError,
    ClaimWaitingError,
    SnowflakeLicenseError,
    arena_api_base_url,
    arena_api_hostname,
    collect_snowflake_claim_key,
    fetch_capabilities,
    partner_headers,
    prove_snowflake_claim,
    rotate_snowflake_app_key,
    snowflake_priced_families,
    snowflake_training_enabled,
    start_snowflake_claim,
)
from agilerl.snowflake.profile import (
    ARENA_API_RULE,
    ARENA_BASE_URL_SECRET,
    ARENA_CLAIM_SECRET,
    ARENA_EXTERNAL_USER_SECRET,
    ARENA_ORG_KEY_SECRET,
)
from agilerl.snowflake.setup_sql import secret_statements, string_literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from agilerl.snowflake.source import SnowflakeSession

APP_SCHEMA = "app_state"
# Same-region codes arrive within seconds; another region waits for a listing
# refresh Arena triggers, which took 53-110 s when measured.
CODE_WAIT_S = 120.0
CODE_POLL_S = 5.0
ROTATE_AFTER_DAYS = 90
NOT_ACCEPTED = (
    "Arena no longer accepts this app's org key.\n"
    "An app_admin runs CALL app_code.connect_arena() to connect again."
)
RUN_AGAIN = "Run CALL app_code.connect_arena() again in a few minutes."
# Claim id of the row begin_claim holds while Arena starts the claim.
PENDING_PREFIX = "pending-"
ALREADY_STARTING = (
    "Another connect_arena() call is connecting this app.\n"
    "Run CALL app_code.connect_arena() again in a moment."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenClaim:
    """The claim ``connect_arena`` started, as ``app_state.arena_claim`` records it.

    :param claim_id: Arena claim id.
    :param base_url: Arena origin the claim was started on.
    :param proven: Whether Arena accepted the code.
    """

    claim_id: str
    base_url: str
    proven: bool


def connect_arena(session: SnowflakeSession, base_url: str | None) -> str:
    """Connect the app to a new Arena organisation, or report that it is connected.

    Each call carries the claim on: it starts one, proves it with the code
    Arena delivers to this install, returns the signup link, and once the
    admin has finished signing up, saves the key in the app's secrets. The
    key is never returned, logged, or quoted in an error.

    :param session: Procedure session.
    :param base_url: Arena origin, or ``None`` for ``https://arena.agilerl.com``.
    :return: What happened and what the admin does next.
    :raises PermissionError: The trial has ended and the listing is not purchased.
    :raises ValueError: The base URL host is not in the Arena network rule.
    :raises ClaimClosedError: Arena refused the proof; the claim is forgotten.
    :raises SnowflakeLicenseError: Arena could not be reached, or refused the
        claim or the new key.
    """
    refuse_ended_trial(session)
    partner = saved_partner()
    if partner is not None:
        try:
            fetch_capabilities(*partner)
        except AuthRejectedError:
            logger.info(
                "Arena refused the saved org key, so connect_arena starts a claim."
            )
        else:
            rotate_if_due(session, partner)
            return (
                "This app is already connected to Arena.\n"
                "Use app_code.rotate_arena_key() for a new key."
            )
    origin = arena_origin(session, base_url or DEFAULT_BASE_URL)
    claim = open_claim(session)
    if claim is not None and claim.claim_id.startswith(PENDING_PREFIX):
        return ALREADY_STARTING
    if claim is not None and claim.base_url != origin:
        forget_claim(session)
        claim = None
    if claim is not None and claim.proven:
        try:
            issued = collect_snowflake_claim_key(
                claim.claim_id, saved_secret("claim_secret"), origin, time.sleep
            )
        except ClaimWaitingError:
            logger.info("Claim %s is not linked yet.", claim.claim_id)
        except ClaimClosedError as exc:
            logger.info("Arena closed claim %s: %s", claim.claim_id, exc)
            forget_claim(session)
            claim = None
        else:
            store_issued_key(session, origin, issued)
            return connected_message(origin, issued)
    if claim is None:
        try:
            started = begin_claim(session, origin)
        except ClaimWaitingError as exc:
            return f"{exc}\n{RUN_AGAIN}"
        if started is None:
            return ALREADY_STARTING
        claim_id, secret = started
    else:
        claim_id, secret = claim.claim_id, saved_secret("claim_secret")
    return prove_claim(session, claim_id, secret, origin)


def prove_claim(
    session: SnowflakeSession, claim_id: str, secret: str, origin: str
) -> str:
    """Read the claim's code, prove it, and return the signup link.

    :param session: Procedure session.
    :param claim_id: Arena claim id.
    :param secret: The claim's secret.
    :param origin: Arena origin the claim was started on.
    :return: The link, why the admin should run ``connect_arena()`` again,
        or, when the claim is already linked, the connection confirmation.
    :raises ClaimClosedError: Arena refused the proof; the claim is forgotten.
    :raises SnowflakeLicenseError: Arena could not be reached.
    """
    code = claim_code(session, claim_id, time.sleep)
    if code is None:
        return (
            f"Snowflake is still delivering Arena's check to this region.\n{RUN_AGAIN}"
        )
    try:
        signup_url = prove_snowflake_claim(claim_id, secret, code, origin, time.sleep)
    except ClaimLinkedError:
        issued = collect_snowflake_claim_key(claim_id, secret, origin, time.sleep)
        store_issued_key(session, origin, issued)
        return connected_message(origin, issued)
    except ClaimWaitingError as exc:
        return f"{exc}\n{RUN_AGAIN}"
    except ClaimClosedError:
        forget_claim(session)
        raise
    collect(
        session,
        "UPDATE app_state.arena_claim SET proven_at = CURRENT_TIMESTAMP() "
        f"WHERE claim_id = {string_literal(claim_id)}",
    )
    sync_billing_task(session)
    return (
        "Open this link to sign up or sign in to Arena and create the "
        f"organisation for this Snowflake account:\n{signup_url}\n"
        "Then run CALL app_code.connect_arena() again, or wait up to an hour "
        "for the app to connect by itself."
    )


def claim_account(session: SnowflakeSession) -> ClaimAccount:
    """Read the install's region, locator, names, and purchased flag from Snowflake.

    :param session: Procedure session.
    :return: The identity a claim carries.
    """
    [row] = collect(
        session,
        "SELECT CURRENT_REGION() AS REGION, CURRENT_ACCOUNT() AS ACCOUNT, "
        "CURRENT_ACCOUNT_NAME() AS ACCOUNT_NAME, CURRENT_ORGANIZATION_NAME() AS "
        "ORGANIZATION_NAME, CURRENT_DATABASE() AS APPLICATION_NAME, "
        "app_code.listing_purchased() AS PURCHASED",
    )
    values = lowered(row)
    return ClaimAccount(
        region=str(values["region"]),
        account=str(values["account"]),
        account_name=str(values["account_name"]),
        organization_name=str(values["organization_name"]),
        application_name=str(values["application_name"]),
        purchased=values["purchased"] is True,
    )


def claim_code(
    session: SnowflakeSession, claim_id: str, sleep: Callable[[float], None]
) -> str | None:
    """Read the claim's code from the package's ``claims.challenges``, waiting up to :data:`CODE_WAIT_S`.

    Consumers cannot query shared package content, and the row is picked by
    the install's own region and locator, so only this install reads it.

    :param session: Procedure session.
    :param claim_id: Arena claim id.
    :param sleep: Called between reads.
    :return: The code, or ``None`` when it has not arrived.
    """
    query = (
        "SELECT code FROM claims.challenges WHERE region = CURRENT_REGION() "
        f"AND locator = CURRENT_ACCOUNT() AND claim_id = {string_literal(claim_id)}"
    )
    reads = int(CODE_WAIT_S / CODE_POLL_S) + 1
    for read in range(reads):
        rows = collect(session, query)
        if rows:
            return str(lowered(rows[0])["code"])
        if read < reads - 1:
            sleep(CODE_POLL_S)
    return None


def open_claim(session: SnowflakeSession) -> OpenClaim | None:
    """Read the claim ``connect_arena`` started, after dropping one past its expiry.

    :param session: Procedure session.
    :return: The claim, or ``None``.
    """
    collect(
        session,
        "DELETE FROM app_state.arena_claim WHERE expires_at <= CURRENT_TIMESTAMP()",
    )
    rows = collect(
        session,
        "SELECT claim_id, base_url, proven_at IS NOT NULL AS PROVEN "
        "FROM app_state.arena_claim",
    )
    if not rows:
        return None
    values = lowered(rows[0])
    return OpenClaim(
        str(values["claim_id"]), str(values["base_url"]), values["proven"] is True
    )


def begin_claim(session: SnowflakeSession, origin: str) -> tuple[str, str] | None:
    """Insert the one claim row, start it with Arena, and store its secret.

    :param session: Procedure session.
    :param origin: Arena origin.
    :return: The claim id and secret, or ``None`` when another call owns the row.
    :raises SnowflakeLicenseError: Arena did not start the claim.
    """
    reservation = f"{PENDING_PREFIX}{secrets.token_hex(16)}"
    # MERGE locks the table, so of two overlapping calls only one inserts, and
    # it does so before Arena is called.
    [merged] = collect(
        session,
        "MERGE INTO app_state.arena_claim AS t USING (SELECT 1 AS one) AS s ON TRUE "
        "WHEN NOT MATCHED THEN INSERT (claim_id, base_url, expires_at) VALUES "
        f"({string_literal(reservation)}, {string_literal(origin)}, "
        "DATEADD('minute', 3, CURRENT_TIMESTAMP()))",
    )
    if int(str(lowered(merged)["number of rows inserted"])) == 0:
        return None
    try:
        started = start_snowflake_claim(claim_account(session), origin, time.sleep)
        save_secret(session, ARENA_CLAIM_SECRET, started.claim_secret)
        collect(
            session,
            "UPDATE app_state.arena_claim "
            f"SET claim_id = {string_literal(started.claim_id)}, "
            f"base_url = {string_literal(origin)}, "
            f"expires_at = TO_TIMESTAMP_TZ({int(started.expires_at.timestamp() * 1000)}, 3) "
            f"WHERE claim_id = {string_literal(reservation)}",
        )
    except (SnowflakeLicenseError, RuntimeError, SnowparkSQLException):
        collect(
            session,
            "DELETE FROM app_state.arena_claim "
            f"WHERE claim_id = {string_literal(reservation)}",
        )
        raise
    return started.claim_id, started.claim_secret


def forget_claim(session: SnowflakeSession) -> None:
    """Drop the open claim and blank its secret.

    :param session: Procedure session.
    """
    collect(session, "DELETE FROM app_state.arena_claim")
    save_secret(session, ARENA_CLAIM_SECRET, "")


def store_issued_key(session: SnowflakeSession, origin: str, issued: AppKey) -> None:
    """Save a key Arena issued for the claim, then forget the claim.

    Arena issues the key once, so it is saved before anything that can fail.

    :param session: Procedure session.
    :param origin: Arena origin the claim was started on.
    :param issued: The key Arena issued.
    """
    save_arena_key(session, origin, issued, "connect")
    forget_claim(session)
    sync_billing_task(session)


def connected_message(origin: str, issued: AppKey) -> str:
    """Check a saved key with Arena and say which organisation it belongs to.

    :param origin: Arena origin.
    :param issued: The key Arena issued.
    :return: Confirmation, without the key.
    :raises SnowflakeLicenseError: Arena could not be reached or does not
        allow Snowflake training. The key stays saved.
    """
    caps = fetch_capabilities(
        origin, partner_headers(issued.org_key, issued.external_user_id)
    )
    if not snowflake_training_enabled(caps):
        raise SnowflakeLicenseError(SNOWFLAKE_DENIED)
    lines = [
        "Connected to Arena."
        if issued.organization_name is None
        else f"Connected to the Arena organisation {issued.organization_name}."
    ]
    if issued.trial_ends_at is not None:
        lines.append(f"The Arena trial ends {issued.trial_ends_at}.")
    lines.extend(priced_lines(caps))
    return "\n".join(lines)


def priced_lines(caps: dict[str, object]) -> list[str]:
    """List the priced GPU families in one line, when Arena lists them.

    :param caps: Capabilities ``data``.
    :return: Zero or one line.
    """
    priced = snowflake_priced_families(caps)
    if priced is None:
        return []
    return [f"Priced GPU families: {', '.join(sorted(priced)) or 'none'}."]


def collect_linked_key(session: SnowflakeSession) -> str | None:
    """Collect the key of a proven claim once the admin has linked it; for the hourly task.

    Arena failures are logged and the billing run goes on.

    :param session: Procedure session.
    :return: What happened, or ``None`` with no proven claim.
    """
    claim = open_claim(session)
    if claim is None or not claim.proven:
        return None
    try:
        issued = collect_snowflake_claim_key(
            claim.claim_id, saved_secret("claim_secret"), claim.base_url, time.sleep
        )
    except ClaimWaitingError:
        return "claim not linked yet"
    except ClaimClosedError as exc:
        forget_claim(session)
        return f"claim closed: {exc}"
    except SnowflakeLicenseError as exc:
        logger.warning("Could not collect the Arena org key: %s", exc)
        return "could not collect the Arena org key"
    store_issued_key(session, claim.base_url, issued)
    try:
        connected_message(claim.base_url, issued)
    except SnowflakeLicenseError as exc:
        logger.warning("Saved the Arena org key; the Arena check failed: %s", exc)
        return "saved the Arena org key; the Arena check failed"
    logger.info("Collected the Arena org key of claim %s.", claim.claim_id)
    return "collected the Arena org key"


def rotate_arena_key(session: SnowflakeSession) -> str:
    """Swap the app's org key for a new one; the old key stops working 5 minutes after the new one is first used.

    :param session: Procedure session.
    :return: Confirmation, without either key.
    :raises ValueError: The app is not connected, its key came from AgileRL,
        or a run is using the current key.
    :raises SnowflakeLicenseError: Arena refused the rotation or the new key.
    """
    partner = saved_partner()
    if partner is None:
        msg = "Arena is not connected. An app_admin calls app_code.connect_arena first."
        raise ValueError(msg)
    if key_source(session) != "connect":
        msg = (
            "This app's key came from AgileRL, so the app cannot rotate it.\n"
            "Ask AgileRL for a new key and pass it to app_code.configure."
        )
        raise ValueError(msg)
    if runs_live(session):
        msg = (
            "A run is using the current key until it ends.\n"
            "Rotate after it ends, or cancel it first. Nothing was changed."
        )
        raise ValueError(msg)
    origin, headers = partner
    rotated = save_rotated(
        session, origin, rotate_snowflake_app_key(origin, headers, time.sleep)
    )
    fetch_capabilities(*rotated)
    return "Rotated the Arena org key. The old key stops working in 5 minutes."


def rotate_if_due(
    session: SnowflakeSession, partner: tuple[str, dict[str, str]]
) -> tuple[str, dict[str, str]]:
    """Rotate a key ``connect_arena`` collected once it is :data:`ROTATE_AFTER_DAYS` old and no run holds it.

    Called only from calls that already count as use of the app, so rotation
    never adds a month to the listing fee. A refused rotation is logged and
    the current key kept.

    :param session: Procedure session.
    :param partner: Arena origin and headers of the current key.
    :return: The origin and headers to use from now on.
    """
    due = collect(
        session,
        "SELECT issued_at FROM app_state.arena_key WHERE source = 'connect' AND "
        f"issued_at <= DATEADD('day', -{ROTATE_AFTER_DAYS}, CURRENT_TIMESTAMP())",
    )
    if not due or runs_live(session):
        return partner
    origin, headers = partner
    try:
        issued = rotate_snowflake_app_key(origin, headers, time.sleep)
    except SnowflakeLicenseError as exc:
        logger.warning("Could not rotate the Arena org key: %s", exc)
        return partner
    logger.info("Rotated the Arena org key after %d days.", ROTATE_AFTER_DAYS)
    return save_rotated(session, origin, issued)


def save_rotated(
    session: SnowflakeSession, origin: str, issued: AppKey
) -> tuple[str, dict[str, str]]:
    """Save a rotated key and return its origin and headers.

    :param session: Procedure session.
    :param origin: Arena origin.
    :param issued: The new key.
    :return: The origin and the new key's headers.
    """
    save_arena_key(session, origin, issued, "connect")
    return origin, partner_headers(issued.org_key, issued.external_user_id)


def key_source(session: SnowflakeSession) -> str | None:
    """``connect_arena`` or ``configure``: which call saved the current key.

    :param session: Procedure session.
    :return: The source, or ``None`` for a key saved before the app recorded it.
    """
    rows = collect(session, "SELECT source FROM app_state.arena_key")
    return str(lowered(rows[0])["source"]) if rows else None


def runs_live(session: SnowflakeSession) -> bool:
    """Whether a live service in ``app_state`` is not a served model.

    Run services read the org key secret when they start, so they keep the old
    key, which Arena retires 5 minutes after the new one is first used; served
    models are metered by the app with the current key.

    :param session: Procedure session.
    :return: True while such a service runs.
    """
    served = served_models(session)
    return any(
        service_live(status) and name not in served
        for name, status in service_statuses(session).items()
    )


def app_capabilities(base_url: str, headers: dict[str, str]) -> dict[str, object]:
    """Capabilities for the saved key, telling the admin to connect again when Arena refuses it.

    :param base_url: Arena origin.
    :param headers: Partner headers.
    :return: Capabilities ``data``.
    :raises AuthRejectedError: Arena refused the key.
    :raises SnowflakeLicenseError: Arena could not be reached.
    """
    try:
        return fetch_capabilities(base_url, headers)
    except AuthRejectedError:
        raise AuthRejectedError(NOT_ACCEPTED) from None


def save_arena_key(
    session: SnowflakeSession, origin: str, issued: AppKey, source: str
) -> None:
    """Save the three Arena secrets and record where the key came from.

    :param session: Procedure session.
    :param origin: Arena origin.
    :param issued: The key and its external user id.
    :param source: ``connect`` for keys the app collected or rotated,
        ``configure`` for keys AgileRL issued by hand.
    """
    for name, value in (
        (ARENA_ORG_KEY_SECRET, issued.org_key),
        (ARENA_EXTERNAL_USER_SECRET, issued.external_user_id),
        (ARENA_BASE_URL_SECRET, origin),
    ):
        save_secret(session, name, value)
    collect(session, "DELETE FROM app_state.arena_key")
    collect(
        session,
        "INSERT INTO app_state.arena_key (source, issued_at) "
        f"VALUES ({string_literal(source)}, CURRENT_TIMESTAMP())",
    )


def arena_origin(session: SnowflakeSession, base_url: str) -> str:
    """``https://<host>`` for ``base_url``, refused when the Arena rule does not allow the host.

    :param session: Procedure session.
    :param base_url: Arena origin as the admin typed it.
    :return: The origin.
    :raises ValueError: The value does not name one host, or the host is not in the rule.
    """
    try:
        host = arena_api_hostname(base_url or "")
    except ValueError:
        # The library message quotes the value, which may be a pasted secret.
        msg = "The base URL does not name one host.\nNothing was saved."
        raise ValueError(msg) from None
    allowed = rule_hosts(session, f"{APP_SCHEMA}.{ARENA_API_RULE}")
    if f"{host.lower()}:443" not in allowed:
        msg = (
            "The base URL host is not in the app's Arena network rule "
            f"({', '.join(sorted(allowed))}), so the app cannot reach it.\n"
            "Nothing was saved."
        )
        raise ValueError(msg)
    return arena_api_base_url(host)


def rule_hosts(session: SnowflakeSession, rule: str) -> set[str]:
    """Lower-case ``host:port`` values of a network rule.

    :param session: Procedure session.
    :param rule: Qualified network rule name.
    :return: The values.
    """
    hosts: set[str] = set()
    for row in collect(session, f"DESCRIBE NETWORK RULE {rule}"):
        raw = str(lowered(row).get("value_list") or "")
        hosts.update(item.strip().lower() for item in raw.split(",") if item.strip())
    return hosts


def save_secret(session: SnowflakeSession, name: str, value: str) -> None:
    """Create the secret when missing, then set its value.

    :param session: Procedure session.
    :param name: Secret name in ``app_state``.
    :param value: Secret string.
    :raises RuntimeError: Snowflake rejected a statement. The error names the
        secret, not its value.
    """
    try:
        for statement in secret_statements(APP_SCHEMA, name, value):
            collect(session, statement)
    except SnowparkSQLException as exc:
        # A Snowflake error can quote the statement, which holds the value.
        msg = (
            f"Snowflake did not save {APP_SCHEMA}.{name} "
            f"(Snowflake error {exc.sql_error_code})."
        )
        raise RuntimeError(msg) from None
