# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Usage billing: one Snowflake billing event per charged usage hour."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import pytest

from agilerl.snowflake import billing
from agilerl.snowflake.billing import (
    HoursBilled,
    bill_hours,
    billable_hour,
    event_batches,
    reconcile,
    training_started,
)
from agilerl.snowflake.license import (
    SnowflakeLicenseError,
    SnowflakeUsage,
    UnknownUsageError,
)
from tests.test_snowflake_setup import Row

RUN_ID = "arena-20261116120000-abc123"
# 2026-11-16T13:00:00Z in epoch milliseconds.
HOUR_13_MS = 1_794_834_000_000
HOUR_MS = 3_600_000
# Two hours after 13:00, so 13:00 and 14:00 are closed.
NOW_MS = HOUR_13_MS + 2 * HOUR_MS
DAY_MS = 24 * HOUR_MS
EMIT = "SELECT SYSTEM$CREATE_BILLING_EVENTS("
WHERE = re.compile(r" WHERE (.*?)(?: (?:ORDER|GROUP) BY .*)?$")
EQUALS = re.compile(r"^(\w+) = '([^']*)'$")
IS_NULL = re.compile(r"^(\w+) IS (NOT )?NULL$")
IN_LIST = re.compile(r"^hour_start_ms IN \(([\d, ]+)\)$")
BEFORE = re.compile(r"^attempted_at < TO_TIMESTAMP_TZ\((\d+), 3\)$")
SET_TIME = re.compile(r"(\w+) = (COALESCE\(\w+, )?TO_TIMESTAMP_TZ\((\d+), 3\)")
SET_TEXT = re.compile(r"(\w+) = '((?:[^']|'')*)'")
MERGE_ITEM = re.compile(
    r"^MERGE INTO app_state\.usage_items AS t USING \(SELECT '([^']*)' AS "
    r"service_name\) AS s .* VALUES \('deployment', '([^']*)', "
    r"TO_TIMESTAMP_TZ\((\d+), 3\), s\.service_name\)$"
)
MERGE_VALUES = re.compile(r"FROM VALUES (.*?)\) AS s ")
MERGE_OWNER = re.compile(r"ON t\.kind = '([^']*)' AND t\.usage_id = '([^']*)'")
MERGE_CLAIM = re.compile(r"SET claim_id = '([^']*)'")
STEPPED = re.compile(
    r"^SELECT 1 AS STEPPED FROM app_state\.arena_metrics, LATERAL FLATTEN\(input => "
    r"METRICS\) WHERE RUN_ID = '([^']*)' AND ENDSWITH\(KEY, '/steps'\) AND "
    r"TRY_TO_DOUBLE\(TO_VARCHAR\(VALUE\)\) > 0 LIMIT 1$"
)


def hour(start: str, cents: int) -> dict[str, object]:
    return {"hourStart": start, "billableCents": cents}


def hour_at(index: int, cents: int) -> dict[str, object]:
    return hour(f"2026-11-16T{13 + index:02d}:00:00Z", cents)


def unknown(kind: str, usage_id: str) -> SnowflakeUsage:
    raise UnknownUsageError(usage_id)


def trained(run_id: str) -> bool:
    return True


def untrained(run_id: str) -> bool:
    return False


def matches(row: dict[str, object], where: str) -> bool:
    for term in where.split(" AND "):
        if term.startswith("(") and term.endswith(")"):
            ok = any(matches(row, option) for option in term[1:-1].split(" OR "))
        elif equals := EQUALS.match(term):
            ok = row[equals.group(1)] == equals.group(2)
        elif is_null := IS_NULL.match(term):
            ok = (row[is_null.group(1)] is None) != bool(is_null.group(2))
        elif in_list := IN_LIST.match(term):
            ok = row["hour_start_ms"] in {
                int(start) for start in in_list.group(1).split(", ")
            }
        elif before := BEFORE.match(term):
            attempted = row["attempted_at"]
            ok = attempted is not None and attempted < int(before.group(1))
        else:
            msg = f"Unmodelled predicate {term!r}"
            raise AssertionError(msg)
        if not ok:
            return False
    return True


class Result:
    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows

    def collect(self) -> list[Row]:
        return list(self.rows)


class BillingSession:
    """Procedure session over ``usage_items`` and ``billed_hours``, emitting into ``events``.

    ``hooks`` maps a statement prefix to a callable run once just before the
    first statement with that prefix, standing in for a concurrent call.
    """

    def __init__(self, statuses: list[str | Exception] | None = None) -> None:
        self.items: list[dict[str, object]] = []
        self.billed: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []
        self.payloads: list[str] = []
        self.statuses = list(statuses or [])
        self.statements: list[str] = []
        self.hooks: dict[str, Callable[[], object]] = {}
        self.services: dict[str, str] = {}
        self.served: list[dict[str, object]] = []
        self.grants: list[str] = []
        self.functions: list[str] = []
        self.stepped: set[str] = set()

    def add_item(
        self,
        usage_id: str,
        opened_ms: int,
        kind: str = "run",
        checked_ms: int | None = None,
        first_404_ms: int | None = None,
        service_name: str | None = None,
        trained: bool = True,
    ) -> dict[str, object]:
        if trained:
            self.stepped.add(usage_id)
        item: dict[str, object] = {
            "kind": kind,
            "usage_id": usage_id,
            "opened_at": opened_ms,
            "settled_at": None,
            "first_seen_at": None,
            "last_checked_at": checked_ms,
            "first_404_at": first_404_ms,
            "service_name": service_name,
            "closed_at": None,
            "last_error": None,
            "last_error_at": None,
            "unbilled_reason": None,
        }
        self.items.append(item)
        return item

    def add_billed(self, start_ms: int, usage_id: str = RUN_ID) -> None:
        self.billed.append(
            {
                "kind": "run",
                "usage_id": usage_id,
                "hour_start_ms": start_ms,
                "billable_cents": 10,
                "claim_id": "earlier-call",
                "attempted_at": None,
                "emitted_at": None,
            }
        )

    def state(self, start_ms: int, usage_id: str = RUN_ID) -> dict[str, object]:
        [row] = [
            row
            for row in self.billed
            if row["usage_id"] == usage_id and row["hour_start_ms"] == start_ms
        ]
        return row

    def sql(self, statement: str) -> Result:
        for prefix in list(self.hooks):
            if statement.startswith(prefix):
                self.hooks.pop(prefix)()
        self.statements.append(statement)
        if statement.startswith(EMIT):
            return Result([Row(STATUS=self._emit(statement))])
        if statement.startswith("MERGE INTO app_state.billed_hours "):
            self._merge(statement)
            return Result([])
        if stepped := STEPPED.match(statement):
            run_id = stepped.group(1)
            return Result([Row(STEPPED=1)] if run_id in self.stepped else [])
        if statement == "SHOW SERVICES IN SCHEMA app_state":
            return Result(
                [
                    Row(name=name, status=status)
                    for name, status in self.services.items()
                ]
            )
        if statement.startswith("CREATE FUNCTION IF NOT EXISTS app_state."):
            name = statement.split()[5].split("(", 1)[0]
            if name not in self.functions:
                self.functions.append(name)
            return Result([])
        if statement.startswith(
            ("GRANT USAGE ON FUNCTION app_state.", "GRANT SERVICE ROLE app_state.")
        ):
            self.grants.append(statement)
            return Result([])
        if statement.startswith("DROP FUNCTION IF EXISTS app_state."):
            name = statement.split()[4].split("(", 1)[0]
            if name in self.functions:
                self.functions.remove(name)
            return Result([])
        if statement.startswith("DROP SERVICE IF EXISTS app_state."):
            self.services.pop(statement.rsplit(".", 1)[1], None)
            return Result([])
        if statement.startswith("SELECT service_name, model, version, run_id, "):
            return Result(
                [Row(**{k.upper(): v for k, v in r.items()}) for r in self.served]
            )
        if statement == (
            "SELECT CURRENT_ACCOUNT() AS ACCOUNT, CURRENT_REGION() AS REGION, "
            "app_code.listing_purchased() AS PURCHASED"
        ):
            return Result(
                [Row(ACCOUNT="AB12345", REGION="AWS_EU_WEST_2", PURCHASED=True)]
            )
        if merge := MERGE_ITEM.match(statement):
            service, usage_id, opened_ms = merge.groups()
            open_rows = [
                row
                for row in self.items
                if row["kind"] == "deployment"
                and row["service_name"] == service
                and row["closed_at"] is None
                and row["settled_at"] is None
            ]
            if not open_rows:
                self.add_item(
                    usage_id, int(opened_ms), "deployment", service_name=service
                )
            return Result([Row(**{"number of rows inserted": int(not open_rows)})])
        where_match = WHERE.search(statement)
        assert where_match is not None, statement
        where = where_match.group(1)
        table = self.items if "app_state.usage_items" in statement else self.billed
        selected = [row for row in table if matches(row, where)]
        if statement.startswith("SELECT usage_id, service_name, first_seen_at "):
            assert statement.endswith(" ORDER BY opened_at")
            return Result(
                [
                    Row(
                        USAGE_ID=r["usage_id"],
                        SERVICE_NAME=r["service_name"],
                        FIRST_SEEN_AT=r["first_seen_at"],
                    )
                    for r in sorted(selected, key=lambda r: int(str(r["opened_at"])))
                ]
            )
        if statement.startswith("SELECT usage_id, DATE_PART(EPOCH_MILLISECOND, "):
            assert statement.endswith(" ORDER BY opened_at")
            return Result(
                [
                    Row(USAGE_ID=r["usage_id"], OPENED_MS=r["opened_at"])
                    for r in sorted(selected, key=lambda r: int(str(r["opened_at"])))
                ]
            )
        if statement.startswith("SELECT COUNT(*) AS N "):
            return Result([Row(N=len(selected))])
        if statement.startswith("SELECT kind, usage_id, "):
            assert statement.endswith(" ORDER BY last_checked_at NULLS FIRST")
            ordered = sorted(
                selected,
                key=lambda r: (r["last_checked_at"] is not None, r["last_checked_at"]),
            )
            return Result(
                [
                    Row(
                        KIND=r["kind"],
                        USAGE_ID=r["usage_id"],
                        OPENED_MS=r["opened_at"],
                        SEEN_MS=r["first_seen_at"],
                        CHECKED_MS=r["last_checked_at"],
                        MISSING_MS=r["first_404_at"],
                    )
                    for r in ordered
                ]
            )
        if statement.startswith("SELECT hour_start_ms, COUNT(*) AS N "):
            assert statement.endswith(" GROUP BY hour_start_ms HAVING COUNT(*) > 1")
            counts: dict[object, int] = {}
            for r in selected:
                counts[r["hour_start_ms"]] = counts.get(r["hour_start_ms"], 0) + 1
            return Result(
                [Row(HOUR_START_MS=s, N=n) for s, n in counts.items() if n > 1]
            )
        if statement.startswith("SELECT hour_start_ms, billable_cents "):
            ordered = sorted(selected, key=lambda r: int(str(r["hour_start_ms"])))
            return Result(
                [
                    Row(
                        HOUR_START_MS=r["hour_start_ms"],
                        BILLABLE_CENTS=r["billable_cents"],
                    )
                    for r in ordered
                ]
            )
        if statement.startswith("SELECT hour_start_ms "):
            return Result([Row(HOUR_START_MS=r["hour_start_ms"]) for r in selected])
        if statement.startswith("UPDATE "):
            columns = statement.split(" WHERE ", 1)[0]
            assignments = SET_TIME.findall(columns)
            assert assignments, statement
            texts = SET_TEXT.findall(columns)
            for row in selected:
                for column, coalesce, stamp in assignments:
                    if not coalesce or row[column] is None:
                        row[column] = int(stamp)
                for column, text in texts:
                    row[column] = text.replace("''", "'")
            return Result([])
        if statement.startswith("DELETE FROM app_state.billed_hours "):
            for row in selected:
                self.billed.remove(row)
            return Result([])
        msg = f"Unmodelled statement {statement!r}"
        raise AssertionError(msg)

    def _merge(self, statement: str) -> None:
        values = MERGE_VALUES.search(statement)
        owner = MERGE_OWNER.search(statement)
        claim = MERGE_CLAIM.search(statement)
        assert values, statement
        assert owner, statement
        assert claim, statement
        kind, usage_id = owner.groups()
        for start, cents in re.findall(r"\((\d+), (\d+)\)", values.group(1)):
            existing = [
                row
                for row in self.billed
                if row["kind"] == kind
                and row["usage_id"] == usage_id
                and row["hour_start_ms"] == int(start)
            ]
            if not existing:
                self.billed.append(
                    {
                        "kind": kind,
                        "usage_id": usage_id,
                        "hour_start_ms": int(start),
                        "billable_cents": int(cents),
                        "claim_id": claim.group(1),
                        "attempted_at": None,
                        "emitted_at": None,
                    }
                )
            for row in existing:
                if row["attempted_at"] is None and row["emitted_at"] is None:
                    row["claim_id"] = claim.group(1)

    def _emit(self, statement: str) -> str:
        literal = statement[len(EMIT) :].rsplit(") AS STATUS", 1)[0]
        document = literal[1:-1].replace("''", "'").replace("\\\\", "\\")
        self.payloads.append(document)
        status = self.statuses.pop(0) if self.statuses else "Success"
        if isinstance(status, Exception):
            raise status
        if status == "Success":
            self.events.extend(json.loads(document))
        return status


class SqlError(Exception):
    """A failed SQL statement."""


@pytest.fixture
def sql_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :class:`SqlError` the SQL error billing catches."""
    monkeypatch.setattr(billing, "sql_errors", lambda: (SqlError,))


class TestBillHours:
    def test_emits_one_event_per_charged_hour_with_cents_times_one_hundredth(
        self,
    ) -> None:
        # Arrange
        session = BillingSession()
        hours = [hour_at(0, 123), hour_at(1, 7)]

        # Act
        billed = bill_hours(session, "run", RUN_ID, hours, NOW_MS)

        # Assert
        assert billed == HoursBilled(2, True)
        assert [event["base_charge"] for event in session.events] == [1.23, 0.07]
        assert all(event["class"] == "ARENA_USAGE" for event in session.events)
        assert all(event["subclass"] == "run" for event in session.events)
        assert all(json.loads(event["objects"]) == [RUN_ID] for event in session.events)
        assert session.state(HOUR_13_MS)["emitted_at"] == NOW_MS
        assert session.state(HOUR_13_MS)["attempted_at"] == NOW_MS

    def test_event_spans_the_closed_hour_in_epoch_milliseconds(self) -> None:
        session = BillingSession()

        bill_hours(session, "run", RUN_ID, [hour_at(0, 50)], NOW_MS)

        [event] = session.events
        assert event["start_timestamp"] == HOUR_13_MS
        assert event["timestamp"] == HOUR_13_MS + HOUR_MS
        assert json.loads(event["additional_info"]) == {"billableCents": 50}

    def test_partial_hour_ends_now_not_in_the_future(self) -> None:
        session = BillingSession()
        now_ms = HOUR_13_MS + 20 * 60_000

        bill_hours(session, "run", RUN_ID, [hour_at(0, 50)], now_ms)

        assert session.events[0]["timestamp"] == now_ms

    def test_offset_hour_start_is_the_same_instant(self) -> None:
        session = BillingSession()

        bill_hours(
            session, "run", RUN_ID, [hour("2026-11-16T14:00:00+01:00", 50)], NOW_MS
        )

        assert session.events[0]["start_timestamp"] == HOUR_13_MS

    def test_rerun_with_the_same_hours_emits_nothing(self) -> None:
        # Arrange
        session = BillingSession()
        hours = [hour_at(0, 123)]
        bill_hours(session, "run", RUN_ID, hours, NOW_MS)

        # Act
        billed = bill_hours(session, "run", RUN_ID, hours, NOW_MS)

        # Assert
        assert billed == HoursBilled(0, True)
        assert len(session.events) == 1

    def test_out_of_order_call_emits_only_the_new_hour(self) -> None:
        # Arrange
        session = BillingSession()
        bill_hours(session, "run", RUN_ID, [hour_at(0, 10), hour_at(1, 20)], NOW_MS)

        # Act
        billed = bill_hours(
            session,
            "run",
            RUN_ID,
            [hour_at(2, 30), hour_at(1, 20), hour_at(0, 10)],
            NOW_MS + HOUR_MS,
        )

        # Assert
        assert billed == HoursBilled(1, True)
        assert [event["base_charge"] for event in session.events] == [0.1, 0.2, 0.3]
        assert session.events[-1]["start_timestamp"] == HOUR_13_MS + 2 * HOUR_MS

    def test_hour_repeated_in_one_call_emits_once(self) -> None:
        session = BillingSession()

        bill_hours(session, "run", RUN_ID, [hour_at(0, 10), hour_at(0, 10)], NOW_MS)

        assert len(session.events) == 1

    def test_zero_cent_hours_are_not_stored(self) -> None:
        # Arrange
        session = BillingSession()

        # Act
        billed = bill_hours(
            session, "run", RUN_ID, [hour_at(0, 0), hour_at(1, 5)], NOW_MS
        )

        # Assert
        assert billed == HoursBilled(1, True)
        assert [row["hour_start_ms"] for row in session.billed] == [
            HOUR_13_MS + HOUR_MS
        ]

    @pytest.mark.parametrize("hours", [[], [hour_at(0, 0)]])
    def test_no_charged_hours_runs_no_sql(self, hours: list[object]) -> None:
        session = BillingSession()

        billed = bill_hours(session, "run", RUN_ID, hours, NOW_MS)

        assert billed == HoursBilled(0, True)
        assert session.statements == []

    def test_kinds_are_billed_separately(self) -> None:
        # Arrange
        session = BillingSession()
        hours = [hour_at(0, 10)]
        bill_hours(session, "run", RUN_ID, hours, NOW_MS)

        # Act
        bill_hours(session, "deployment", RUN_ID, hours, NOW_MS)

        # Assert
        assert [event["subclass"] for event in session.events] == ["run", "deployment"]

    def test_many_hours_are_split_into_calls_under_the_payload_limit(self) -> None:
        # Arrange
        session = BillingSession()
        hours = [
            hour(f"2026-11-{16 + n // 24:02d}T{n % 24:02d}:00:00Z", 99_999)
            for n in range(45)
        ]

        # Act
        billed = bill_hours(session, "run", RUN_ID, hours, HOUR_13_MS + 50 * HOUR_MS)

        # Assert
        assert len(session.payloads) == 2
        assert all(len(payload) < 9000 for payload in session.payloads)
        assert billed == HoursBilled(45, True)
        assert len(session.events) == 45

    @pytest.mark.parametrize(
        "point",
        [
            "SELECT hour_start_ms, billable_cents ",
            "UPDATE app_state.billed_hours SET attempted_at",
            EMIT,
        ],
    )
    def test_interleaved_calls_emit_each_hour_once(self, point: str) -> None:
        # Arrange
        session = BillingSession()
        hours = [hour_at(0, 10), hour_at(1, 20)]
        other: list[HoursBilled] = []
        session.hooks[point] = lambda: other.append(
            bill_hours(session, "run", RUN_ID, hours, NOW_MS)
        )

        # Act
        first = bill_hours(session, "run", RUN_ID, hours, NOW_MS)

        # Assert
        assert sorted(event["start_timestamp"] for event in session.events) == [
            HOUR_13_MS,
            HOUR_13_MS + HOUR_MS,
        ]
        assert first.emitted + other[0].emitted == 2
        assert all(row["emitted_at"] == NOW_MS for row in session.billed)

    def test_raising_billing_call_keeps_hours_unconfirmed_and_never_resends(
        self, sql_error: None
    ) -> None:
        # Arrange
        session = BillingSession([SqlError("Connection reset by peer")])
        hours = [hour_at(0, 10)]

        # Act
        with pytest.raises(SqlError, match="Connection reset"):
            bill_hours(session, "run", RUN_ID, hours, NOW_MS)
        retried = bill_hours(session, "run", RUN_ID, hours, NOW_MS + HOUR_MS)

        # Assert
        assert retried == HoursBilled(0, True)
        assert session.events == []
        assert session.state(HOUR_13_MS)["attempted_at"] == NOW_MS
        assert session.state(HOUR_13_MS)["emitted_at"] is None
        assert sum(s.startswith(EMIT) for s in session.statements) == 1

    def test_undocumented_status_keeps_hours_unconfirmed(self) -> None:
        session = BillingSession(["Something new"])

        with pytest.raises(RuntimeError, match="answered 'Something new' to 1"):
            bill_hours(session, "run", RUN_ID, [hour_at(0, 10)], NOW_MS)

        assert session.state(HOUR_13_MS)["attempted_at"] == NOW_MS
        assert session.state(HOUR_13_MS)["emitted_at"] is None

    @pytest.mark.parametrize(
        "refusal",
        [
            "Too many calls. At most 1 call is allowed per 10 millisecond window.",
            SqlError("Only callable from within an application"),
            SqlError(
                "090898 (0A000): Application instance is not installed from listing"
            ),
        ],
    )
    def test_documented_refusal_deletes_hours_so_the_next_call_bills_them(
        self, refusal: str | Exception, sql_error: None
    ) -> None:
        # Arrange
        session = BillingSession([refusal])
        hours = [hour_at(0, 10)]

        # Act
        with pytest.raises(RuntimeError, match="refused 1 billing events"):
            bill_hours(session, "run", RUN_ID, hours, NOW_MS)
        after_refusal = list(session.billed)
        billed = bill_hours(session, "run", RUN_ID, hours, NOW_MS)

        # Assert
        assert after_refusal == []
        assert billed == HoursBilled(1, True)
        assert len(session.events) == 1

    def test_two_rows_for_one_hour_raise_before_anything_is_emitted(self) -> None:
        # Arrange
        session = BillingSession()
        session.add_billed(HOUR_13_MS)
        session.add_billed(HOUR_13_MS)

        # Act
        with pytest.raises(
            RuntimeError, match=r"more than one row for hours \[1794834000000\] of run"
        ):
            bill_hours(session, "run", RUN_ID, [hour_at(0, 10), hour_at(1, 20)], NOW_MS)

        # Assert
        assert session.payloads == []
        assert all(row["attempted_at"] is None for row in session.billed)

    def test_claimed_but_unattempted_hours_are_retried(self) -> None:
        # Arrange
        session = BillingSession()
        session.billed.append(
            {
                "kind": "run",
                "usage_id": RUN_ID,
                "hour_start_ms": HOUR_13_MS,
                "billable_cents": 10,
                "claim_id": "crashed-call",
                "attempted_at": None,
                "emitted_at": None,
            }
        )

        # Act
        billed = bill_hours(session, "run", RUN_ID, [hour_at(0, 10)], NOW_MS)

        # Assert
        assert billed == HoursBilled(1, True)
        assert session.state(HOUR_13_MS)["emitted_at"] == NOW_MS

    @pytest.mark.parametrize(
        ("kind", "usage_id", "hours", "error", "message"),
        [
            ("job", RUN_ID, [], ValueError, "Unknown usage kind job"),
            ("run", "", [], ValueError, "1 to 64 characters, got 0"),
            ("run", "r" * 65, [], ValueError, "1 to 64 characters, got 65"),
            ("run", RUN_ID, [hour_at(0, 9_999_999)], ValueError, "9999998"),
            ("run", RUN_ID, [hour_at(0, 1), "x"], TypeError, "must be an object"),
        ],
    )
    def test_invalid_input_raises_before_any_sql(
        self,
        kind: str,
        usage_id: str,
        hours: list[object],
        error: type[Exception],
        message: str,
    ) -> None:
        session = BillingSession()

        with pytest.raises(error, match=message):
            bill_hours(session, kind, usage_id, hours, NOW_MS)

        assert session.statements == []


class TestReconcile:
    def test_bills_open_items_and_settles_closed_attempted_ones(self) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        session.add_item("arena-open", HOUR_13_MS)
        usage = {
            RUN_ID: SnowflakeUsage("closed", [hour_at(0, 10), hour_at(1, 0)]),
            "arena-open": SnowflakeUsage("open", [hour_at(0, 20)]),
        }

        # Act
        summary = reconcile(session, lambda kind, uid: usage[uid], trained, NOW_MS)

        # Assert
        assert summary == (
            "items: 2, settled: 1, events: 2, failed: 0, missing: 0, skipped: 0, stale: 0, "
            "unconfirmed: 0, waiting: 0, untrained: 0"
        )
        assert [item["settled_at"] for item in session.items] == [NOW_MS, None]
        assert [item["first_seen_at"] for item in session.items] == [NOW_MS] * 2
        assert [item["last_checked_at"] for item in session.items] == [NOW_MS] * 2

    def test_settled_items_are_not_read_again(self) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        closed = SnowflakeUsage("closed", [hour_at(0, 10)])
        reconcile(session, lambda kind, uid: closed, trained, NOW_MS)
        reads: list[str] = []

        # Act
        summary = reconcile(
            session, lambda kind, uid: reads.append(uid) or closed, trained, NOW_MS
        )

        # Assert
        assert summary.startswith("items: 0, settled: 0, events: 0")
        assert reads == []
        assert len(session.events) == 1

    def test_reads_least_recently_checked_items_first(self) -> None:
        # Arrange
        session = BillingSession()
        session.add_item("arena-recent", HOUR_13_MS, checked_ms=NOW_MS - HOUR_MS)
        session.add_item("arena-new", HOUR_13_MS)
        session.add_item("arena-old", HOUR_13_MS, checked_ms=NOW_MS - 2 * HOUR_MS)
        reads: list[str] = []
        open_usage = SnowflakeUsage("open", [])

        # Act
        reconcile(
            session, lambda kind, uid: reads.append(uid) or open_usage, trained, NOW_MS
        )

        # Assert
        assert reads == ["arena-new", "arena-old", "arena-recent"]
        assert all(item["last_checked_at"] == NOW_MS for item in session.items)

    def test_first_failed_read_stops_reading_for_the_run(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item("arena-down", HOUR_13_MS)
        session.add_item(RUN_ID, HOUR_13_MS, checked_ms=NOW_MS - HOUR_MS)
        reads: list[str] = []

        def down_once(kind: str, usage_id: str) -> SnowflakeUsage:
            reads.append(usage_id)
            if len(reads) == 1:
                msg = "Could not read Snowflake usage from Arena."
                raise SnowflakeLicenseError(msg)
            return SnowflakeUsage("open", [])

        # Act
        first = reconcile(session, down_once, trained, NOW_MS)
        reconcile(session, down_once, trained, NOW_MS + HOUR_MS)

        # Assert
        assert first == (
            "items: 2, settled: 0, events: 0, failed: 1, missing: 0, skipped: 0, stale: 0, "
            "unconfirmed: 0, waiting: 0, untrained: 0"
        )
        assert reads == ["arena-down", RUN_ID, "arena-down"]
        assert session.items[0]["settled_at"] is None
        assert session.items[0]["last_error"] == (
            "Usage read failed: Could not read Snowflake usage from Arena."
        )
        assert session.items[0]["last_error_at"] == NOW_MS
        assert session.items[1]["last_error"] is None
        assert (
            "Could not read run arena-down from Arena, so this run reads no more"
            in (caplog.text)
        )

    def test_a_404_does_not_stop_reading(self) -> None:
        # Arrange
        session = BillingSession()
        session.add_item("arena-missing", HOUR_13_MS)
        session.add_item(RUN_ID, HOUR_13_MS, checked_ms=NOW_MS - HOUR_MS)
        reads: list[str] = []

        def first_missing(kind: str, usage_id: str) -> SnowflakeUsage:
            reads.append(usage_id)
            if usage_id == "arena-missing":
                raise UnknownUsageError(usage_id)
            return SnowflakeUsage("open", [])

        # Act
        summary = reconcile(session, first_missing, trained, NOW_MS)

        # Assert
        assert reads == ["arena-missing", RUN_ID]
        assert "failed: 0, missing: 1" in summary

    def test_404_only_item_is_kept_with_a_warning_before_30_days(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, NOW_MS - 2 * DAY_MS)

        # Act
        summary = reconcile(session, unknown, trained, NOW_MS)

        # Assert
        assert summary == (
            "items: 1, settled: 0, events: 0, failed: 0, missing: 1, skipped: 0, stale: 0, "
            "unconfirmed: 0, waiting: 0, untrained: 0"
        )
        [item] = session.items
        assert item["settled_at"] is None
        assert (item["first_404_at"], item["last_checked_at"]) == (NOW_MS, NOW_MS)
        assert f"Arena has no run {RUN_ID}. It is read again daily" in caplog.text

    def test_404_only_item_is_read_at_most_daily(self) -> None:
        # Arrange
        session = BillingSession()
        checked_ms = NOW_MS - DAY_MS + 1
        item = session.add_item(
            RUN_ID, NOW_MS - 5 * DAY_MS, checked_ms=checked_ms, first_404_ms=checked_ms
        )
        reads: list[int] = []

        def counted(kind: str, usage_id: str) -> SnowflakeUsage:
            reads.append(len(reads))
            raise UnknownUsageError(usage_id)

        # Act
        within_a_day = reconcile(session, counted, trained, NOW_MS)
        reads_within_a_day = len(reads)
        reconcile(session, counted, trained, checked_ms + DAY_MS)

        # Assert
        assert reads_within_a_day == 0
        assert "missing: 0, skipped: 1" in within_a_day
        assert len(reads) == 1
        assert item["last_checked_at"] == checked_ms + DAY_MS
        assert item["first_404_at"] == checked_ms

    @pytest.mark.parametrize(
        ("since_first_404_ms", "settled_at"),
        [(30 * DAY_MS - 1, None), (30 * DAY_MS, NOW_MS)],
    )
    def test_404_only_item_settles_unbilled_after_30_days_of_404s(
        self,
        since_first_404_ms: int,
        settled_at: int | None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(
            RUN_ID,
            NOW_MS - 31 * DAY_MS,
            checked_ms=NOW_MS - DAY_MS,
            first_404_ms=NOW_MS - since_first_404_ms,
        )

        # Act
        reconcile(session, unknown, trained, NOW_MS)

        # Assert
        assert session.items[0]["settled_at"] == settled_at
        settled_warning = f"Arena has had no run {RUN_ID} for 30 days" in caplog.text
        assert settled_warning is (settled_at is not None)

    def test_item_arena_answered_before_never_settles_on_404(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        reconcile(
            session, lambda kind, uid: SnowflakeUsage("open", []), trained, NOW_MS
        )
        later_ms = NOW_MS + 31 * DAY_MS
        reconcile(session, unknown, trained, NOW_MS + HOUR_MS)

        # Act
        summary = reconcile(session, unknown, trained, later_ms)

        # Assert
        assert "settled: 0" in summary
        assert "missing: 1" in summary
        assert session.items[0]["settled_at"] is None
        assert session.items[0]["last_checked_at"] == later_ms
        assert f"Arena has no run {RUN_ID}, but it reported usage" in caplog.text

    def test_item_with_billed_hours_never_settles_on_404_and_is_read_hourly(
        self,
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(
            RUN_ID,
            NOW_MS - 40 * DAY_MS,
            checked_ms=NOW_MS - HOUR_MS,
            first_404_ms=NOW_MS - 31 * DAY_MS,
        )
        session.add_billed(HOUR_13_MS)
        reads: list[str] = []

        def counted(kind: str, usage_id: str) -> SnowflakeUsage:
            reads.append(usage_id)
            raise UnknownUsageError(usage_id)

        # Act
        reconcile(session, counted, trained, NOW_MS)

        # Assert
        assert reads == [RUN_ID]
        assert session.items[0]["settled_at"] is None

    def test_open_item_over_7_days_old_is_counted_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, NOW_MS - 7 * DAY_MS - 1)
        session.add_item("arena-young", NOW_MS - 7 * DAY_MS)
        open_usage = SnowflakeUsage("open", [])

        # Act
        summary = reconcile(session, lambda kind, uid: open_usage, trained, NOW_MS)

        # Assert
        assert "stale: 1" in summary
        assert f"Arena has had run {RUN_ID} open for over 7 days" in caplog.text
        assert "arena-young" not in caplog.text

    def test_closed_item_with_an_unconfirmed_hour_settles_without_resending(
        self, caplog: pytest.LogCaptureFixture, sql_error: None
    ) -> None:
        # Arrange
        session = BillingSession([SqlError("Connection reset by peer")])
        session.add_item(RUN_ID, HOUR_13_MS)
        closed = SnowflakeUsage("closed", [hour_at(0, 10)])
        first = reconcile(session, lambda kind, uid: closed, trained, NOW_MS)

        # Act
        later = reconcile(
            session, lambda kind, uid: closed, trained, NOW_MS + HOUR_MS + 1
        )

        # Assert
        assert first == (
            "items: 1, settled: 0, events: 0, failed: 1, missing: 0, skipped: 0, stale: 0, "
            "unconfirmed: 0, waiting: 0, untrained: 0"
        )
        assert later == (
            "items: 1, settled: 1, events: 0, failed: 0, missing: 0, skipped: 0, stale: 0, "
            "unconfirmed: 1, waiting: 0, untrained: 0"
        )
        assert session.events == []
        assert session.items[0]["settled_at"] == NOW_MS + HOUR_MS + 1
        assert (
            session.items[0]["last_error"] == "Billing failed: Connection reset by peer"
        )
        assert session.items[0]["last_error_at"] == NOW_MS
        assert "Snowflake never confirmed 1 billed hours" in caplog.text

    def test_one_failing_bill_does_not_stop_the_others(self) -> None:
        # Arrange
        session = BillingSession(["Something new"])
        session.add_item("arena-bad", HOUR_13_MS)
        session.add_item(RUN_ID, HOUR_13_MS)
        usage = SnowflakeUsage("closed", [hour_at(0, 10)])

        # Act
        summary = reconcile(session, lambda kind, uid: usage, trained, NOW_MS)

        # Assert
        assert summary == (
            "items: 2, settled: 1, events: 1, failed: 1, missing: 0, skipped: 0, stale: 0, "
            "unconfirmed: 0, waiting: 0, untrained: 0"
        )
        assert json.loads(session.events[0]["objects"]) == [RUN_ID]
        bad, good = session.items
        assert bad["last_error"] == (
            "Billing failed: Snowflake answered 'Something new' to 1 billing events."
        )
        assert good["last_error"] is None


class TestRunsThatNeverTrained:
    def test_closed_run_with_no_training_step_settles_unbilled_with_the_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        closed = SnowflakeUsage("closed", [hour_at(0, 252)])

        # Act
        with caplog.at_level("INFO"):
            summary = reconcile(session, lambda kind, uid: closed, untrained, NOW_MS)

        # Assert
        assert summary == (
            "items: 1, settled: 1, events: 0, failed: 0, missing: 0, skipped: 0, stale: 0, "
            "unconfirmed: 0, waiting: 0, untrained: 1"
        )
        assert session.events == []
        assert session.billed == []
        [item] = session.items
        assert item["settled_at"] == NOW_MS
        assert item["unbilled_reason"] == "The run ended before training started."
        assert f"Run {RUN_ID} ended before training started" in caplog.text

    def test_open_run_waits_unbilled_until_its_first_training_step(self) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        usage = SnowflakeUsage("open", [hour_at(0, 100)])
        steps: list[bool] = []

        def trains_on_second_read(run_id: str) -> bool:
            steps.append(run_id == RUN_ID)
            return len(steps) > 1

        # Act
        first = reconcile(
            session, lambda kind, uid: usage, trains_on_second_read, NOW_MS
        )
        events_while_setting_up = list(session.events)
        second = reconcile(
            session, lambda kind, uid: usage, trains_on_second_read, NOW_MS + HOUR_MS
        )

        # Assert
        assert first.endswith("waiting: 1, untrained: 0")
        assert events_while_setting_up == []
        assert session.items[0]["settled_at"] is None
        assert second.endswith("waiting: 0, untrained: 0")
        assert [event["start_timestamp"] for event in session.events] == [HOUR_13_MS]

    def test_run_that_trained_is_billed_up_to_its_end_including_setup(self) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        closed = SnowflakeUsage("closed", [hour_at(0, 100), hour_at(1, 40)])

        # Act
        summary = reconcile(session, lambda kind, uid: closed, trained, NOW_MS)

        # Assert
        assert "settled: 1, events: 2" in summary
        assert [event["base_charge"] for event in session.events] == [1.0, 0.4]
        assert session.items[0]["unbilled_reason"] is None

    def test_run_billed_before_keeps_billing_without_a_training_check(self) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        session.add_billed(HOUR_13_MS)
        checked: list[str] = []
        closed = SnowflakeUsage("closed", [hour_at(0, 10), hour_at(1, 20)])

        # Act
        reconcile(
            session,
            lambda kind, uid: closed,
            lambda run_id: checked.append(run_id) or False,
            NOW_MS,
        )

        # Assert
        assert checked == []
        assert [event["start_timestamp"] for event in session.events] == [
            HOUR_13_MS,
            HOUR_13_MS + HOUR_MS,
        ]
        assert session.items[0]["settled_at"] == NOW_MS

    def test_deployments_are_billed_without_a_training_check(self) -> None:
        # Arrange
        session = BillingSession()
        session.add_item("dep-a", HOUR_13_MS, "deployment", service_name="A")
        closed = SnowflakeUsage("closed", [hour_at(0, 19)])

        # Act
        summary = reconcile(session, lambda kind, uid: closed, untrained, NOW_MS)

        # Assert
        assert "settled: 1, events: 1" in summary
        assert summary.endswith("untrained: 0")


class TestTrainingStarted:
    @pytest.mark.parametrize(("stepped", "started"), [(set(), False), ({RUN_ID}, True)])
    def test_reads_a_step_count_above_zero_from_the_metrics_table(
        self, stepped: set[str], started: bool
    ) -> None:
        session = BillingSession()
        session.stepped = stepped

        assert training_started(session, "app_state.arena_metrics", RUN_ID) is started
        assert session.statements == [
            (
                "SELECT 1 AS STEPPED FROM app_state.arena_metrics, LATERAL FLATTEN(input "
                f"=> METRICS) WHERE RUN_ID = '{RUN_ID}' AND ENDSWITH(KEY, '/steps') AND "
                "TRY_TO_DOUBLE(TO_VARCHAR(VALUE)) > 0 LIMIT 1"
            )
        ]


class TestEventBatches:
    def test_caps_a_batch_at_one_hundred_events(self) -> None:
        events = [{"n": n} for n in range(250)]

        batches = event_batches(events)

        assert [len(batch) for batch in batches] == [100, 100, 50]
        assert [event for batch in batches for event in batch] == events

    def test_keeps_each_batch_under_nine_thousand_characters(self) -> None:
        events = [{"pad": "x" * 1000} for _ in range(20)]

        batches = event_batches(events)

        assert all(len(json.dumps(batch)) < 9000 for batch in batches)
        assert sum(len(batch) for batch in batches) == 20

    def test_no_events_make_no_batches(self) -> None:
        assert event_batches([]) == []


class TestBillableHour:
    def test_returns_epoch_milliseconds_and_cents(self) -> None:
        assert billable_hour(hour("2026-11-16T13:00:00Z", 123)) == (HOUR_13_MS, 123)

    @pytest.mark.parametrize(
        ("entry", "error", "message"),
        [
            (hour("2026-11-16T13:30:00Z", 1), ValueError, "whole UTC hour"),
            (hour("2026-11-16T13:00:00", 1), ValueError, "whole UTC hour"),
            (hour("2026-11-16T13:00:00Z", -1), ValueError, "0 to 9999998"),
            (hour("2026-11-16T13:00:00Z", 1.5), ValueError, "0 to 9999998"),
            (hour("2026-11-16T13:00:00Z", True), ValueError, "0 to 9999998"),
            (hour("2026-11-16T13:00:00Z", 9_999_999), ValueError, "0 to 9999998"),
            ("2026-11-16T13:00:00Z", TypeError, "must be an object"),
        ],
    )
    def test_malformed_hour_raises(
        self, entry: object, error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error, match=message):
            billable_hour(entry)
