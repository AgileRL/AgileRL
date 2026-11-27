# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Bill Arena usage from a Snowflake Native App as ``ARENA_USAGE`` billing events."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from agilerl.snowflake.catalog import collect
from agilerl.snowflake.job import lowered
from agilerl.snowflake.license import (
    SnowflakeLicenseError,
    SnowflakeUsage,
    UnknownUsageError,
)
from agilerl.snowflake.session import sql_errors
from agilerl.snowflake.setup_sql import string_literal
from agilerl.snowflake.source import SnowflakeSession

logger = logging.getLogger(__name__)

UNTRAINED_REASON = "The run ended before training started."


def reconcile(
    session: SnowflakeSession,
    fetch: Callable[[str, str], SnowflakeUsage],
    trained: Callable[[str], bool],
    now_ms: int,
) -> str:
    """Bill each unsettled usage item from Arena's hours, then settle finished items.

    Items are read least recently checked first. A closed item settles once
    every charged hour has been attempted. A run with no billed hours is not
    billed until it records a training step: while open it waits, and once
    closed it settles unbilled. An item Arena answers 404 for settles only
    when Arena never answered for it, it has no billed hours, and the 404s
    have lasted 30 days; until then it is read at most daily. The first
    failed read other than a 404 stops reading for this run.

    :param session: Procedure session.
    :param fetch: Arena usage of ``(kind, usage_id)``.
    :param trained: Whether a run has recorded a training step.
    :param now_ms: Current time in epoch milliseconds.
    :return: Counts of items, settled items, emitted events, failed items,
        items Arena answered 404 for, items skipped until their daily read,
        items open over 7 days, hours attempted over an hour ago that
        Snowflake never confirmed, runs waiting for their first training
        step, and runs settled unbilled because they ended before training.
    """
    day_ms = 24 * 3_600_000
    items = collect(
        session,
        "SELECT kind, usage_id, "
        "DATE_PART(EPOCH_MILLISECOND, opened_at) AS OPENED_MS, "
        "DATE_PART(EPOCH_MILLISECOND, first_seen_at) AS SEEN_MS, "
        "DATE_PART(EPOCH_MILLISECOND, last_checked_at) AS CHECKED_MS, "
        "DATE_PART(EPOCH_MILLISECOND, first_404_at) AS MISSING_MS "
        "FROM app_state.usage_items WHERE settled_at IS NULL "
        "ORDER BY last_checked_at NULLS FIRST",
    )
    settled = emitted = failed = missing = skipped = stale = waiting = untrained = 0
    for item in items:
        values = lowered(item)
        kind, usage_id = str(values["kind"]), str(values["usage_id"])
        if checked_today(session, kind, usage_id, values, now_ms):
            skipped += 1
            continue
        try:
            usage = fetch(kind, usage_id)
        except UnknownUsageError:
            missing += 1
            settled += settle_missing(session, kind, usage_id, values, now_ms)
            continue
        except (SnowflakeLicenseError, ValueError) as exc:
            mark_checked(session, kind, usage_id, now_ms, None)
            logger.warning(
                "Could not read %s %s from Arena, so this run reads no more items: %s",
                kind,
                usage_id,
                exc,
            )
            record_error(session, kind, usage_id, now_ms, f"Usage read failed: {exc}")
            failed += 1
            break
        mark_checked(session, kind, usage_id, now_ms, "first_seen_at")
        if kind == "run" and not run_billable(session, usage_id, trained):
            if usage.state == "open":
                waiting += 1
                continue
            settle_untrained(session, usage_id, now_ms)
            settled += 1
            untrained += 1
            continue
        try:
            billed = bill_hours(session, kind, usage_id, usage.billable_hours, now_ms)
        except (TypeError, ValueError, RuntimeError, *sql_errors()) as exc:
            logger.exception("Could not bill %s %s", kind, usage_id)
            record_error(session, kind, usage_id, now_ms, f"Billing failed: {exc}")
            failed += 1
            continue
        emitted += billed.emitted
        if usage.state == "closed" and billed.attempted:
            settle(session, kind, usage_id, now_ms)
            settled += 1
        elif (
            usage.state == "open"
            and now_ms - int(str(values["opened_ms"])) > 7 * day_ms
        ):
            logger.warning("Arena has had %s %s open for over 7 days.", kind, usage_id)
            stale += 1
    unconfirmed = unconfirmed_hours(session, now_ms)
    return (
        f"items: {len(items)}, settled: {settled}, events: {emitted}, "
        f"failed: {failed}, missing: {missing}, skipped: {skipped}, stale: {stale}, "
        f"unconfirmed: {unconfirmed}, waiting: {waiting}, untrained: {untrained}"
    )


def training_started(
    session: SnowflakeSession, metrics_table: str, run_id: str
) -> bool:
    """Whether a run's metrics hold a step count above zero.

    Trainers report ``agent_<n>/steps`` once a learn step has started, including
    one that raises, and not before.

    :param session: Procedure session.
    :param metrics_table: Table the run's metrics merge into.
    :param run_id: Run id.
    :return: True once the run has recorded a training step.
    """
    rows = collect(
        session,
        f"SELECT 1 AS STEPPED FROM {metrics_table}, LATERAL FLATTEN(input => METRICS) "
        f"WHERE RUN_ID = {string_literal(run_id)} AND ENDSWITH(KEY, '/steps') "
        "AND TRY_TO_DOUBLE(TO_VARCHAR(VALUE)) > 0 LIMIT 1",
    )
    return bool(rows)


def run_billable(
    session: SnowflakeSession, run_id: str, trained: Callable[[str], bool]
) -> bool:
    """Whether a run may be billed: it has billed hours already, or it has trained.

    :param session: Procedure session.
    :param run_id: Run id.
    :param trained: Whether a run has recorded a training step.
    :return: True when its hours are billed.
    """
    return has_billed_hours(session, "run", run_id) or trained(run_id)


def settle_untrained(session: SnowflakeSession, run_id: str, now_ms: int) -> None:
    """Settle a run that ended before its first training step, unbilled, with the reason.

    :param session: Procedure session.
    :param run_id: Run id.
    :param now_ms: Current time in epoch milliseconds.
    """
    logger.info("Run %s ended before training started. It is not billed.", run_id)
    collect(
        session,
        f"UPDATE app_state.usage_items SET settled_at = TO_TIMESTAMP_TZ({now_ms}, 3), "
        f"unbilled_reason = {string_literal(UNTRAINED_REASON)} "
        f"WHERE kind = 'run' AND usage_id = {string_literal(run_id)}",
    )


def checked_today(
    session: SnowflakeSession,
    kind: str,
    usage_id: str,
    values: dict[str, object],
    now_ms: int,
) -> bool:
    """Whether a never-seen item Arena answered 404 for was read in the last day.

    :param session: Procedure session.
    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id.
    :param values: Lower-cased ``usage_items`` row with the epoch columns.
    :param now_ms: Current time in epoch milliseconds.
    :return: True when the item waits for tomorrow's read.
    """
    day_ms = 24 * 3_600_000
    checked_ms = epoch_ms(values["checked_ms"])
    return (
        epoch_ms(values["seen_ms"]) is None
        and epoch_ms(values["missing_ms"]) is not None
        and checked_ms is not None
        and now_ms - checked_ms < day_ms
        and not has_billed_hours(session, kind, usage_id)
    )


def unconfirmed_hours(session: SnowflakeSession, now_ms: int) -> int:
    """Count, and log, hours attempted over an hour ago that Snowflake never confirmed.

    :param session: Procedure session.
    :param now_ms: Current time in epoch milliseconds.
    :return: The count.
    """
    [row] = collect(
        session,
        "SELECT COUNT(*) AS N FROM app_state.billed_hours WHERE "
        "attempted_at IS NOT NULL AND emitted_at IS NULL AND "
        f"attempted_at < TO_TIMESTAMP_TZ({now_ms - 3_600_000}, 3)",
    )
    unconfirmed = int(str(lowered(row)["n"]))
    if unconfirmed:
        logger.warning(
            "Snowflake never confirmed %d billed hours attempted over an hour ago. "
            "Check them against "
            "SNOWFLAKE.DATA_SHARING_USAGE.MARKETPLACE_PAID_USAGE_DAILY.",
            unconfirmed,
        )
    return unconfirmed


def settle_missing(
    session: SnowflakeSession,
    kind: str,
    usage_id: str,
    values: dict[str, object],
    now_ms: int,
) -> bool:
    """Stamp a 404 on an item, and settle it unbilled once that is safe.

    :param session: Procedure session.
    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id.
    :param values: Lower-cased ``usage_items`` row with the epoch columns.
    :param now_ms: Current time in epoch milliseconds.
    :return: Whether the item settled.
    """
    mark_checked(session, kind, usage_id, now_ms, "first_404_at")
    seen = epoch_ms(values["seen_ms"]) is not None
    missing_ms = epoch_ms(values["missing_ms"])
    first_404_ms = now_ms if missing_ms is None else missing_ms
    if seen or has_billed_hours(session, kind, usage_id):
        logger.warning(
            "Arena has no %s %s, but it reported usage for it before. Keeping it.",
            kind,
            usage_id,
        )
        return False
    if now_ms - first_404_ms < 30 * 24 * 3_600_000:
        logger.warning(
            "Arena has no %s %s. It is read again daily and settles unbilled "
            "after 30 days of this.",
            kind,
            usage_id,
        )
        return False
    logger.warning(
        "Arena has had no %s %s for 30 days. Settling it unbilled.", kind, usage_id
    )
    settle(session, kind, usage_id, now_ms)
    return True


def epoch_ms(value: object) -> int | None:
    """Epoch milliseconds of a ``DATE_PART`` column, or ``None`` for SQL NULL.

    :param value: Column value.
    :return: The milliseconds.
    """
    return None if value is None else int(str(value))


def has_billed_hours(session: SnowflakeSession, kind: str, usage_id: str) -> bool:
    """Whether ``app_state.billed_hours`` has a row for a usage item.

    :param session: Procedure session.
    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id.
    :return: True when it has one.
    """
    [row] = collect(
        session,
        "SELECT COUNT(*) AS N FROM app_state.billed_hours "
        f"WHERE kind = {string_literal(kind)} "
        f"AND usage_id = {string_literal(usage_id)}",
    )
    return int(str(lowered(row)["n"])) > 0


def mark_checked(
    session: SnowflakeSession,
    kind: str,
    usage_id: str,
    now_ms: int,
    first_column: str | None,
) -> None:
    """Stamp ``last_checked_at``, and ``first_column`` when it is still NULL.

    :param session: Procedure session.
    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id.
    :param now_ms: Current time in epoch milliseconds.
    :param first_column: ``first_seen_at``, ``first_404_at``, or ``None``.
    """
    stamp = f"TO_TIMESTAMP_TZ({now_ms}, 3)"
    first = (
        ""
        if first_column is None
        else f", {first_column} = COALESCE({first_column}, {stamp})"
    )
    collect(
        session,
        f"UPDATE app_state.usage_items SET last_checked_at = {stamp}{first} "
        f"WHERE kind = {string_literal(kind)} "
        f"AND usage_id = {string_literal(usage_id)}",
    )


def record_error(
    session: SnowflakeSession, kind: str, usage_id: str, now_ms: int, error: str
) -> None:
    """Store why reconciling a usage item failed in ``last_error`` and ``last_error_at``.

    :param session: Procedure session.
    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id.
    :param now_ms: Current time in epoch milliseconds.
    :param error: The reason.
    """
    collect(
        session,
        f"UPDATE app_state.usage_items SET last_error = {string_literal(error)}, "
        f"last_error_at = TO_TIMESTAMP_TZ({now_ms}, 3) "
        f"WHERE kind = {string_literal(kind)} "
        f"AND usage_id = {string_literal(usage_id)}",
    )


def settle(session: SnowflakeSession, kind: str, usage_id: str, now_ms: int) -> None:
    """Mark a usage item settled, so reconciling skips it.

    :param session: Procedure session.
    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id.
    :param now_ms: Current time in epoch milliseconds.
    """
    collect(
        session,
        f"UPDATE app_state.usage_items SET settled_at = TO_TIMESTAMP_TZ({now_ms}, 3) "
        f"WHERE kind = {string_literal(kind)} "
        f"AND usage_id = {string_literal(usage_id)}",
    )


@dataclass(frozen=True)
class HoursBilled:
    """Outcome of :func:`bill_hours` for one usage item."""

    emitted: int
    attempted: bool


def bill_hours(
    session: SnowflakeSession,
    kind: str,
    usage_id: str,
    hours: list[object],
    now_ms: int,
) -> HoursBilled:
    """Emit one ``ARENA_USAGE`` billing event per charged hour no call has attempted.

    One MERGE claims the hours with this call's claim id, and ``attempted_at``
    is set just before the billing call. A concurrent call therefore never
    emits the same hour, and an attempted hour is never emitted again.
    Zero-cent hours are not stored.

    :param session: Procedure session.
    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id, at most 64 characters.
    :param hours: Arena ``billableHours``.
    :param now_ms: Current time in epoch milliseconds. A partial hour's event
        ends here.
    :return: Events emitted, and whether every charged hour in ``hours`` has
        been attempted by this or an earlier call.
    :raises TypeError: An hour is not an object. No SQL ran.
    :raises ValueError: The kind, id, or an hour is invalid. No SQL ran.
    :raises RuntimeError: An hour of the item has more than one row, and
        nothing was emitted; Snowflake refused a batch with a documented
        message, and its hours were deleted so a later call bills them; or it
        answered something undocumented, and the hours stay attempted and
        unconfirmed.
    :raises SnowparkSQLException: The billing call failed without a documented
        refusal. The hours stay attempted and unconfirmed.
    """
    charged = charged_hours(kind, usage_id, hours)
    if not charged:
        return HoursBilled(0, True)
    owner = f"kind = {string_literal(kind)} AND usage_id = {string_literal(usage_id)}"
    claim = string_literal(uuid.uuid4().hex)
    claim_hours(session, kind, usage_id, charged, claim)
    mine = f"{owner} AND claim_id = {claim} AND emitted_at IS NULL"
    events = claimed_events(session, kind, usage_id, mine, now_ms)
    emitted = sum(
        emit_batch(session, batch, mine, now_ms) for batch in event_batches(events)
    )
    done_rows = collect(
        session,
        "SELECT hour_start_ms FROM app_state.billed_hours "
        f"WHERE {owner} AND attempted_at IS NOT NULL",
    )
    done = {int(str(lowered(row)["hour_start_ms"])) for row in done_rows}
    return HoursBilled(emitted, set(charged) <= done)


def charged_hours(kind: str, usage_id: str, hours: list[object]) -> dict[int, int]:
    """Cents of each charged hour, keyed by start in epoch milliseconds.

    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id, at most 64 characters.
    :param hours: Arena ``billableHours``.
    :return: Hours with more than zero cents. The first entry of an hour wins.
    :raises TypeError: An hour is not an object.
    :raises ValueError: The kind, id, or an hour is invalid.
    """
    max_id_length = 64
    if kind not in ("run", "deployment"):
        msg = f"Unknown usage kind {kind}. Use run or deployment."
        raise ValueError(msg)
    if not 0 < len(usage_id) <= max_id_length:
        msg = f"A usage id must be 1 to 64 characters, got {len(usage_id)}."
        raise ValueError(msg)
    charged: dict[int, int] = {}
    for hour in hours:
        start_ms, cents = billable_hour(hour)
        if cents > 0:
            charged.setdefault(start_ms, cents)
    return charged


def claim_hours(
    session: SnowflakeSession,
    kind: str,
    usage_id: str,
    charged: dict[int, int],
    claim: str,
) -> None:
    """Claim the unattempted charged hours of a usage item for one call.

    :param session: Procedure session.
    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id.
    :param charged: Cents of each charged hour, keyed by start.
    :param claim: SQL literal of this call's claim id.
    :raises RuntimeError: An hour of the item has more than one row.
    """
    values = ", ".join(f"({start}, {cents})" for start, cents in charged.items())
    collect(
        session,
        "MERGE INTO app_state.billed_hours AS t USING (SELECT column1 AS "
        f"hour_start_ms, column2 AS billable_cents FROM VALUES {values}) AS s "
        f"ON t.kind = {string_literal(kind)} AND t.usage_id = "
        f"{string_literal(usage_id)} AND t.hour_start_ms = s.hour_start_ms "
        "WHEN MATCHED AND t.attempted_at IS NULL AND t.emitted_at IS NULL "
        f"THEN UPDATE SET claim_id = {claim} "
        "WHEN NOT MATCHED THEN INSERT "
        "(kind, usage_id, hour_start_ms, billable_cents, claim_id) VALUES "
        f"({string_literal(kind)}, {string_literal(usage_id)}, s.hour_start_ms, "
        f"s.billable_cents, {claim})",
    )
    # Snowflake does not enforce the primary key, so a duplicate could bill an hour twice.
    duplicates = collect(
        session,
        "SELECT hour_start_ms, COUNT(*) AS N FROM app_state.billed_hours "
        f"WHERE kind = {string_literal(kind)} AND usage_id = {string_literal(usage_id)} "
        "GROUP BY hour_start_ms HAVING COUNT(*) > 1",
    )
    if duplicates:
        starts = sorted(int(str(lowered(row)["hour_start_ms"])) for row in duplicates)
        msg = (
            f"app_state.billed_hours has more than one row for hours {starts} "
            f"of {kind} {usage_id}. Nothing was emitted."
        )
        raise RuntimeError(msg)


def claimed_events(
    session: SnowflakeSession, kind: str, usage_id: str, mine: str, now_ms: int
) -> list[dict[str, object]]:
    """Billing events for the claimed hours no call has attempted, oldest first.

    :param session: Procedure session.
    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id.
    :param mine: SQL condition selecting this call's unemitted hours.
    :param now_ms: Current time in epoch milliseconds. A partial hour's event
        ends here.
    :return: Events for ``SYSTEM$CREATE_BILLING_EVENTS``.
    """
    hour_ms = 3_600_000
    rows = collect(
        session,
        "SELECT hour_start_ms, billable_cents FROM app_state.billed_hours "
        f"WHERE {mine} AND attempted_at IS NULL ORDER BY hour_start_ms",
    )
    return [
        {
            "class": "ARENA_USAGE",
            "subclass": kind,
            "start_timestamp": start,
            "timestamp": min(start + hour_ms, now_ms),
            "base_charge": round(cents * 0.01, 2),
            "objects": json.dumps([usage_id]),
            "additional_info": json.dumps({"billableCents": cents}),
        }
        for start, cents in (
            (int(str(v["hour_start_ms"])), int(str(v["billable_cents"])))
            for v in map(lowered, rows)
        )
    ]


def emit_batch(
    session: SnowflakeSession, batch: list[dict[str, object]], mine: str, now_ms: int
) -> int:
    """Mark a batch's hours attempted, then emit the events of the hours this call holds.

    :param session: Procedure session.
    :param batch: Events Snowflake accepts in one call.
    :param mine: SQL condition selecting this call's unemitted hours.
    :param now_ms: Current time in epoch milliseconds.
    :return: Events emitted.
    :raises RuntimeError: Snowflake refused the batch with a documented
        message, and its hours were deleted; or it answered something
        undocumented, and the hours stay attempted.
    :raises SnowparkSQLException: The billing call failed without a
        documented refusal.
    """
    # SYSTEM$CREATE_BILLING_EVENTS statuses that mean no event was created.
    refusals = (
        "Invalid parameter",
        "Only callable from within an application",
        "Payload length exceeds the limit",
        "Number of events exceeds the limit",
        "Too many calls",
        "not installed from listing",
    )
    starts = in_list([int(str(event["start_timestamp"])) for event in batch])
    collect(
        session,
        "UPDATE app_state.billed_hours SET attempted_at = "
        f"TO_TIMESTAMP_TZ({now_ms}, 3) WHERE {mine} AND attempted_at IS NULL "
        f"AND hour_start_ms IN ({starts})",
    )
    owned_rows = collect(
        session,
        "SELECT hour_start_ms FROM app_state.billed_hours WHERE "
        f"{mine} AND attempted_at IS NOT NULL AND hour_start_ms IN ({starts})",
    )
    owned = {int(str(lowered(row)["hour_start_ms"])) for row in owned_rows}
    sent = [event for event in batch if event["start_timestamp"] in owned]
    if not sent:
        return 0
    sent_starts = in_list(sorted(owned))
    try:
        [row] = collect(
            session,
            "SELECT SYSTEM$CREATE_BILLING_EVENTS("
            f"{string_literal(json.dumps(sent))}) AS STATUS",
        )
        status = str(lowered(row)["status"])
    except sql_errors() as exc:
        if not any(refusal in str(exc) for refusal in refusals):
            raise
        status = str(exc)
    if status != "Success":
        if any(refusal in status for refusal in refusals):
            collect(
                session,
                "DELETE FROM app_state.billed_hours WHERE "
                f"{mine} AND hour_start_ms IN ({sent_starts})",
            )
            msg = f"Snowflake refused {len(sent)} billing events: {status}"
            raise RuntimeError(msg)
        msg = f"Snowflake answered {status!r} to {len(sent)} billing events."
        raise RuntimeError(msg)
    collect(
        session,
        "UPDATE app_state.billed_hours SET emitted_at = "
        f"TO_TIMESTAMP_TZ({now_ms}, 3) WHERE {mine} "
        f"AND hour_start_ms IN ({sent_starts})",
    )
    return len(sent)


def event_batches(events: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """Split billing events into calls Snowflake accepts: 100 events and 9000 characters.

    :param events: Events for ``SYSTEM$CREATE_BILLING_EVENTS``.
    :return: Batches in order.
    """
    max_events, max_length = 100, 9000
    batches: list[list[dict[str, object]]] = []
    for event in events:
        if (
            not batches
            or len(batches[-1]) == max_events
            or len(json.dumps([*batches[-1], event])) >= max_length
        ):
            batches.append([])
        batches[-1].append(event)
    return batches


def billable_hour(hour: object) -> tuple[int, int]:
    """Start in epoch milliseconds and cents of one Arena ``billableHours`` entry.

    :param hour: ``{"hourStart": "<UTC hour, ISO 8601>", "billableCents": <int>}``.
    :return: Both values.
    :raises TypeError: The entry is not an object.
    :raises ValueError: The start is not a whole hour with a time zone, or the
        cents are not a whole number from 0 to 9,999,998. Snowflake refuses a
        base charge of 99,999.99 USD or more.
    """
    if not isinstance(hour, dict):
        msg = f"A billable hour must be an object, got {hour!r}."
        raise TypeError(msg)
    text = str(hour.get("hourStart"))
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    start = datetime.fromisoformat(text)
    if start.tzinfo is None or start.timestamp() % 3600 != 0:
        msg = f"hourStart must be a whole UTC hour, got {hour.get('hourStart')!r}."
        raise ValueError(msg)
    cents = hour.get("billableCents")
    cents_limit = 9_999_999
    if (
        isinstance(cents, bool)
        or not isinstance(cents, int)
        or not 0 <= cents < cents_limit
    ):
        msg = f"billableCents must be a whole number from 0 to 9999998, got {cents!r}."
        raise ValueError(msg)
    return int(start.timestamp()) * 1000, cents


def in_list(starts: list[int]) -> str:
    """Comma-separated hour starts for a SQL ``IN`` list.

    :param starts: Hour starts in epoch milliseconds.
    :return: The list body.
    """
    return ", ".join(str(start) for start in starts)
