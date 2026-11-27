# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native App org key: connect through a package-delivered claim, and key rotation."""

from __future__ import annotations

import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import app_billing
import app_keys
import httpx
import pytest
from app_keys import (
    ALREADY_STARTING,
    app_capabilities,
    collect_linked_key,
    connect_arena,
    rotate_arena_key,
    rotate_if_due,
)
from snowflake.snowpark.exceptions import SnowparkSQLException

from agilerl.snowflake import license as license_mod
from agilerl.snowflake.license import (
    SNOWFLAKE_DENIED,
    AuthRejectedError,
    ClaimClosedError,
    SnowflakeLicenseError,
    partner_headers,
)
from tests.test_snowflake_app_runs import TRIAL_ENDED, Install, listing_trial
from tests.test_snowflake_app_runs import Result as AppResult
from tests.test_snowflake_license import Arena, reply
from tests.test_snowflake_setup import Row

ORIGIN = "https://arena.test"
REGION, LOCATOR = "AWS_EU_WEST_2", "OZ12967"
CLAIM_SECRET = "claim-s3cr3t"
ISSUED_KEY = "arena_org_issued-key"
LINK = "https://arena.test/marketplace/signup#token=link-1"
FUTURE = "2099-01-01T00:00:00+00:00"
ENABLED = {
    "ok": True,
    "data": {
        "schemaVersion": 1,
        "features": {
            "snowflakeTraining": True,
            "snowflakePricedGpuFamilies": ["GPU_NV_S"],
        },
    },
}
SECRET_ALIASES = {
    "org_key": "arena_org_key",
    "external_user_id": "arena_external_user_id",
    "base_url": "arena_base_url",
    "claim_secret": "arena_claim_secret",
}
PROVEN_CLAIMS = (
    "SELECT claim_id FROM app_state.arena_claim WHERE proven_at IS NOT NULL "
    "AND expires_at > CURRENT_TIMESTAMP()"
)
CODE_QUERY = re.compile(
    r"SELECT code FROM claims\.challenges WHERE region = CURRENT_REGION\(\) "
    r"AND locator = CURRENT_ACCOUNT\(\) AND claim_id = '([^']*)'"
)
CLAIM_RESERVE = re.compile(
    r"MERGE INTO app_state\.arena_claim AS t USING \(SELECT 1 AS one\) AS s ON TRUE "
    r"WHEN NOT MATCHED THEN INSERT \(claim_id, base_url, expires_at\) VALUES "
    r"\('([^']*)', '([^']*)', DATEADD\('minute', 3, CURRENT_TIMESTAMP\(\)\)\)"
)
CLAIM_STORE = re.compile(
    r"UPDATE app_state\.arena_claim SET claim_id = '([^']*)', base_url = '([^']*)', "
    r"expires_at = TO_TIMESTAMP_TZ\((\d+), 3\) WHERE claim_id = '([^']*)'"
)
CLAIM_RELEASE = re.compile(
    r"DELETE FROM app_state\.arena_claim WHERE claim_id = '([^']*)'"
)
SECRET_SET = re.compile(r"ALTER SECRET app_state\.(\w+) SET SECRET_STRING = '([^']*)'")
KEY_INSERT = re.compile(
    r"INSERT INTO app_state\.arena_key \(source, issued_at\) "
    r"VALUES \('(\w+)', CURRENT_TIMESTAMP\(\)\)"
)


def now_ms() -> int:
    return int(time.time() * 1000)


class KeysSession:
    """Procedure session over the app's claim, key, secrets, and the package's challenge rows.

    ``challenges`` holds ``(region, locator, claim_id, code)`` rows, readable
    after ``code_after`` reads of the claim's code.
    """

    def __init__(self, install: Install | None = None) -> None:
        self.install = install or Install()
        self.statements: list[str] = []
        self.secrets: dict[str, str] = dict.fromkeys(SECRET_ALIASES.values(), "")
        self.claim: dict[str, object] | None = None
        self.key: dict[str, object] | None = None
        self.challenges: list[tuple[str, str, str, str]] = []
        self.code_after = 0
        self.code_reads = 0
        self.services: dict[str, str] = {}
        self.task = "suspended"
        self.claim_at_reserve: dict[str, object] | None = None
        self.fail_secret: str | None = None

    def sql(self, statement: str) -> AppResult:
        self.statements.append(statement)
        if (ended := self.install.ended(statement)) is not None:
            return ended
        if statement == "DESCRIBE NETWORK RULE app_state.arena_api_egress":
            return AppResult([Row(value_list="arena.agilerl.com:443, arena.test:443")])
        if statement.startswith("SELECT CURRENT_REGION() AS REGION, CURRENT_ACCOUNT()"):
            return AppResult(
                [
                    Row(
                        REGION=REGION,
                        ACCOUNT=LOCATOR,
                        ACCOUNT_NAME="ARENA_CUSTOMER_TEST",
                        ORGANIZATION_NAME="QURHFZE",
                        APPLICATION_NAME="AGILERL_ARENA",
                        PURCHASED=False,
                    )
                ]
            )
        if code := CODE_QUERY.fullmatch(statement):
            self.code_reads += 1
            rows = [
                Row(CODE=row_code)
                for region, locator, claim_id, row_code in self.challenges
                if (region, locator, claim_id) == (REGION, LOCATOR, code.group(1))
            ]
            return AppResult(rows if self.code_reads > self.code_after else [])
        if (
            statement
            == "DELETE FROM app_state.arena_claim WHERE expires_at <= CURRENT_TIMESTAMP()"
        ):
            if (
                self.claim is not None
                and int(str(self.claim["expires_ms"])) <= now_ms()
            ):
                self.claim = None
            return AppResult([])
        if statement == (
            "SELECT claim_id, base_url, proven_at IS NOT NULL AS PROVEN "
            "FROM app_state.arena_claim"
        ):
            if self.claim is None:
                return AppResult([])
            return AppResult(
                [
                    Row(
                        CLAIM_ID=self.claim["claim_id"],
                        BASE_URL=self.claim["base_url"],
                        PROVEN=self.claim["proven"],
                    )
                ]
            )
        if reserved := CLAIM_RESERVE.fullmatch(statement):
            if self.claim_at_reserve is not None:
                self.claim = self.claim_at_reserve
                self.claim_at_reserve = None
            inserted = self.claim is None
            if inserted:
                reservation, base_url = reserved.groups()
                self.claim = {
                    "claim_id": reservation,
                    "base_url": base_url,
                    "expires_ms": now_ms() + 180_000,
                    "proven": False,
                }
            return AppResult([Row(**{"number of rows inserted": int(inserted)})])
        if stored := CLAIM_STORE.fullmatch(statement):
            claim_id, base_url, expires_ms, reservation = stored.groups()
            assert self.claim is not None
            assert self.claim["claim_id"] == reservation
            self.claim = {
                "claim_id": claim_id,
                "base_url": base_url,
                "expires_ms": int(expires_ms),
                "proven": False,
            }
            return AppResult([])
        if released := CLAIM_RELEASE.fullmatch(statement):
            if self.claim is not None and self.claim["claim_id"] == released.group(1):
                self.claim = None
            return AppResult([])
        if statement.startswith("UPDATE app_state.arena_claim SET proven_at = "):
            assert self.claim is not None
            self.claim["proven"] = True
            return AppResult([])
        if statement == "DELETE FROM app_state.arena_claim":
            self.claim = None
            return AppResult([])
        if statement == PROVEN_CLAIMS:
            proven = self.claim is not None and self.claim["proven"]
            return AppResult([Row(CLAIM_ID="c")] if proven else [])
        if statement.startswith("CREATE SECRET IF NOT EXISTS app_state."):
            return AppResult([])
        if secret := SECRET_SET.fullmatch(statement):
            if secret.group(1) == self.fail_secret:
                msg = f"SQL compilation error near '{secret.group(2)}'"
                raise SnowparkSQLException(msg, sql_error_code=1003)
            self.secrets[secret.group(1)] = secret.group(2)
            return AppResult([])
        if statement == "DELETE FROM app_state.arena_key":
            self.key = None
            return AppResult([])
        if key := KEY_INSERT.fullmatch(statement):
            self.key = {"source": key.group(1), "issued_ms": now_ms()}
            return AppResult([])
        if statement == "SELECT source FROM app_state.arena_key":
            return AppResult(
                [] if self.key is None else [Row(SOURCE=self.key["source"])]
            )
        if statement.startswith("SELECT issued_at FROM app_state.arena_key WHERE "):
            days = int(re.search(r"DATEADD\('day', -(\d+),", statement).group(1))
            cutoff = now_ms() - days * 86_400_000
            due = (
                self.key is not None
                and self.key["source"] == "connect"
                and int(str(self.key["issued_ms"])) <= cutoff
            )
            return AppResult([Row(ISSUED_AT=1)] if due else [])
        if statement == "SHOW SERVICES IN SCHEMA app_state":
            return AppResult(
                [
                    Row(name=name, status=status)
                    for name, status in self.services.items()
                ]
            )
        if statement.startswith("SELECT service_name, model, version, run_id, "):
            return AppResult([])
        if statement.startswith("SELECT COUNT(*) AS N FROM app_state.usage_items "):
            return AppResult([Row(N=0)])
        if statement.startswith("ALTER TASK app_state.reconcile_usage_task "):
            self.task = "suspended" if statement.endswith("SUSPEND") else "started"
            return AppResult([])
        msg = f"Unmodelled statement {statement!r}"
        raise AssertionError(msg)

    def connected_with(self, source: str, age_days: float = 0) -> None:
        """Save an org key as ``configure`` or ``connect`` would, ``age_days`` ago."""
        self.secrets.update(
            arena_org_key="arena_org_saved-key",
            arena_external_user_id="snowflake-app",
            arena_base_url=ORIGIN,
        )
        self.key = {
            "source": source,
            "issued_ms": now_ms() - int(age_days * 86_400_000),
        }

    def proven_claim(
        self, claim_id: str = "claim-1", expires_in_s: float = 3600
    ) -> None:
        self.claim = {
            "claim_id": claim_id,
            "base_url": ORIGIN,
            "expires_ms": now_ms() + int(expires_in_s * 1000),
            "proven": True,
        }
        self.secrets["arena_claim_secret"] = CLAIM_SECRET


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> KeysSession:
    """A session whose secrets the procedure reads like Snowflake's ``_snowflake``."""
    fake = KeysSession()
    monkeypatch.setitem(
        sys.modules,
        "_snowflake",
        SimpleNamespace(
            get_generic_secret_string=lambda alias: fake.secrets[SECRET_ALIASES[alias]]
        ),
    )
    monkeypatch.setattr(app_keys.time, "sleep", lambda seconds: None)
    return fake


@pytest.fixture
def arena(monkeypatch: pytest.MonkeyPatch) -> Arena:
    fake = Arena([])
    monkeypatch.setattr(license_mod.httpx, "Client", fake)
    return fake


def claim_started(claim_id: str = "claim-1") -> httpx.Response:
    return reply(
        201,
        {
            "ok": True,
            "data": {
                "claimId": claim_id,
                "claimSecret": CLAIM_SECRET,
                "expiresAt": FUTURE,
            },
        },
    )


def proven(link: str = LINK) -> httpx.Response:
    return reply(200, {"ok": True, "data": {"signupUrl": link}})


def issued(name: str | None = "Acme ML (Snowflake)") -> httpx.Response:
    data: dict[str, object] = {"orgKey": ISSUED_KEY, "externalUserId": "snowflake-app"}
    if name is not None:
        data["organizationName"] = name
        data["trialEndsAt"] = "2026-11-27T18:00:00+00:00"
    return reply(200, {"ok": True, "data": data})


def refused(status: int, error: str) -> httpx.Response:
    return reply(status, {"error": error, "description": f"Arena says {error}."})


def paths(arena: Arena) -> list[str]:
    return [str(call["url"]).removeprefix(ORIGIN) for call in arena.calls]


class TestConnect:
    def test_first_call_proves_the_claim_with_this_installs_code_and_returns_the_link(
        self, session: KeysSession, arena: Arena
    ) -> None:
        # Arrange
        session.challenges = [
            ("GCP_EUROPE_WEST2", LOCATOR, "claim-1", "decoy-region"),
            (REGION, "YZ75740", "claim-1", "decoy-locator"),
            (REGION, LOCATOR, "claim-1", "code-7f3c"),
        ]
        arena.replies = [claim_started(), proven()]

        # Act
        text = connect_arena(session, ORIGIN)

        # Assert
        assert paths(arena) == [
            "/api/cli/v1/snowflake/claims",
            "/api/cli/v1/snowflake/claims/claim-1/prove",
        ]
        assert arena.calls[0]["json"]["snowflakeRegion"] == REGION
        assert arena.calls[0]["json"]["snowflakeAccount"] == LOCATOR
        assert arena.calls[0]["json"]["purchased"] is False
        assert arena.calls[1]["json"] == {
            "claimSecret": CLAIM_SECRET,
            "code": "code-7f3c",
        }
        assert LINK in text
        assert "CALL app_code.connect_arena() again" in text
        assert CLAIM_SECRET not in text
        assert session.secrets["arena_claim_secret"] == CLAIM_SECRET
        assert session.claim is not None
        assert session.claim["proven"] is True
        assert session.task == "started"
        assert session.secrets["arena_org_key"] == ""

    def test_after_signup_saves_the_key_without_returning_it_and_forgets_the_claim(
        self, session: KeysSession, arena: Arena
    ) -> None:
        # Arrange
        session.proven_claim()
        arena.replies = [issued(), reply(200, ENABLED)]

        # Act
        text = connect_arena(session, ORIGIN)

        # Assert
        assert paths(arena) == [
            "/api/cli/v1/snowflake/claims/claim-1/key",
            "/api/cli/v1/capabilities",
        ]
        assert arena.calls[0]["json"] == {"claimSecret": CLAIM_SECRET}
        assert arena.calls[1]["headers"]["Authorization"] == f"Bearer {ISSUED_KEY}"
        assert text.splitlines() == [
            "Connected to the Arena organisation Acme ML (Snowflake).",
            "The Arena trial ends 2026-11-27T18:00:00+00:00.",
            "Priced GPU families: GPU_NV_S.",
        ]
        assert session.secrets == {
            "arena_org_key": ISSUED_KEY,
            "arena_external_user_id": "snowflake-app",
            "arena_base_url": ORIGIN,
            "arena_claim_secret": "",
        }
        assert session.key is not None
        assert session.key["source"] == "connect"
        assert session.claim is None
        assert session.task == "suspended"

    def test_the_key_appears_only_in_secret_statements(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim()
        arena.replies = [issued(), reply(200, ENABLED)]

        connect_arena(session, ORIGIN)

        leaked = [
            s
            for s in session.statements
            if (ISSUED_KEY in s or CLAIM_SECRET in s) and "SECRET" not in s
        ]
        assert leaked == []

    def test_before_signup_proves_again_for_a_fresh_link_and_keeps_the_claim(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim()
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        fresh = "https://arena.test/marketplace/signup#token=link-2"
        arena.replies = [refused(409, "claim_not_linked"), proven(fresh)]

        text = connect_arena(session, ORIGIN)

        assert paths(arena) == [
            "/api/cli/v1/snowflake/claims/claim-1/key",
            "/api/cli/v1/snowflake/claims/claim-1/prove",
        ]
        assert fresh in text
        assert session.claim is not None
        assert session.secrets["arena_org_key"] == ""

    def test_a_code_still_in_transit_after_two_minutes_keeps_the_claim(
        self, session: KeysSession, arena: Arena, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        sleeps: list[float] = []
        monkeypatch.setattr(app_keys.time, "sleep", sleeps.append)
        arena.replies = [claim_started()]

        # Act
        text = connect_arena(session, ORIGIN)

        # Assert
        assert text.startswith("Snowflake is still delivering Arena's check")
        assert sum(sleeps) == 120.0
        assert session.code_reads == 25
        assert session.claim is not None
        assert session.claim["proven"] is False
        assert paths(arena) == ["/api/cli/v1/snowflake/claims"]

    def test_a_code_that_arrives_late_is_still_proven_on_the_same_call(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        session.code_after = 11
        arena.replies = [claim_started(), proven()]

        text = connect_arena(session, ORIGIN)

        assert LINK in text
        assert session.code_reads == 12

    def test_an_unproven_claim_is_carried_on_by_the_next_call(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim()
        assert session.claim is not None
        session.claim["proven"] = False
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        arena.replies = [proven()]

        text = connect_arena(session, ORIGIN)

        assert paths(arena) == ["/api/cli/v1/snowflake/claims/claim-1/prove"]
        assert LINK in text

    def test_an_install_arena_cannot_see_yet_keeps_the_claim(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        arena.replies = [claim_started(), refused(409, "install_not_visible_yet")]

        text = connect_arena(session, ORIGIN)

        assert "install_not_visible_yet" in text
        assert text.endswith(
            "Run CALL app_code.connect_arena() again in a few minutes."
        )
        assert session.claim is not None
        assert session.claim["proven"] is False

    def test_a_rejected_proof_forgets_the_claim(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        arena.replies = [claim_started(), refused(403, "claim_rejected")]

        with pytest.raises(ClaimClosedError, match="claim_rejected") as caught:
            connect_arena(session, ORIGIN)

        assert session.claim is None
        assert session.secrets["arena_claim_secret"] == ""
        assert CLAIM_SECRET not in str(caught.value)

    def test_an_expired_claim_is_replaced_on_the_same_call(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim("claim-old")
        session.challenges = [(REGION, LOCATOR, "claim-2", "code-2")]
        arena.replies = [
            refused(410, "claim_expired"),
            claim_started("claim-2"),
            proven(),
        ]

        text = connect_arena(session, ORIGIN)

        assert paths(arena) == [
            "/api/cli/v1/snowflake/claims/claim-old/key",
            "/api/cli/v1/snowflake/claims",
            "/api/cli/v1/snowflake/claims/claim-2/prove",
        ]
        assert LINK in text
        assert session.claim is not None
        assert session.claim["claim_id"] == "claim-2"

    def test_a_claim_past_its_expiry_is_dropped_before_arena_is_asked(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim("claim-old", expires_in_s=-1)
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        arena.replies = [claim_started(), proven()]

        connect_arena(session, ORIGIN)

        assert paths(arena)[0] == "/api/cli/v1/snowflake/claims"

    def test_a_working_saved_key_is_reported_and_no_claim_starts(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.connected_with("configure")
        arena.replies = [reply(200, ENABLED)]

        text = connect_arena(session, ORIGIN)

        assert text.startswith("This app is already connected to Arena.")
        assert paths(arena) == ["/api/cli/v1/capabilities"]
        assert session.claim is None

    def test_a_saved_key_arena_refuses_starts_a_claim(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.connected_with("connect")
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        arena.replies = [
            httpx.Response(401, text="Unauthorized"),
            claim_started(),
            proven(),
        ]

        text = connect_arena(session, ORIGIN)

        assert paths(arena) == [
            "/api/cli/v1/capabilities",
            "/api/cli/v1/snowflake/claims",
            "/api/cli/v1/snowflake/claims/claim-1/prove",
        ]
        assert LINK in text

    def test_an_ended_trial_refuses_before_arena_is_called(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.install = listing_trial(8)

        with pytest.raises(PermissionError, match=TRIAL_ENDED):
            connect_arena(session, ORIGIN)

        assert arena.calls == []

    def test_a_key_without_snowflake_training_is_saved_and_the_claim_is_forgotten(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim()
        disabled = {"ok": True, "data": {"schemaVersion": 1, "features": {}}}
        arena.replies = [issued(), reply(200, disabled)]

        with pytest.raises(SnowflakeLicenseError, match=SNOWFLAKE_DENIED):
            connect_arena(session, ORIGIN)

        assert session.secrets["arena_org_key"] == ISSUED_KEY
        assert session.claim is None

    @pytest.mark.parametrize("status", [503, 426])
    def test_a_failed_capabilities_check_keeps_the_issued_key(
        self, session: KeysSession, arena: Arena, status: int
    ) -> None:
        session.proven_claim()
        arena.replies = [issued(), reply(status)]

        with pytest.raises(SnowflakeLicenseError):
            connect_arena(session, ORIGIN)

        assert session.secrets["arena_org_key"] == ISSUED_KEY
        assert session.secrets["arena_external_user_id"] == "snowflake-app"
        assert session.secrets["arena_base_url"] == ORIGIN
        assert session.claim is None

    def test_a_claim_another_call_is_starting_asks_for_a_retry(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.claim = {
            "claim_id": "pending-abc",
            "base_url": ORIGIN,
            "expires_ms": now_ms() + 180_000,
            "proven": False,
        }

        text = connect_arena(session, ORIGIN)

        assert text == ALREADY_STARTING
        assert arena.calls == []
        assert session.claim is not None
        assert session.claim["claim_id"] == "pending-abc"

    def test_a_row_that_appears_before_the_insert_does_not_start_another_claim(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.claim_at_reserve = {
            "claim_id": "claim-other",
            "base_url": ORIGIN,
            "expires_ms": now_ms() + 3_600_000,
            "proven": False,
        }

        text = connect_arena(session, ORIGIN)

        assert text == ALREADY_STARTING
        assert arena.calls == []
        assert session.claim is not None
        assert session.claim["claim_id"] == "claim-other"
        assert session.secrets["arena_claim_secret"] == ""

    def test_a_failed_start_drops_the_reservation(
        self, session: KeysSession, arena: Arena
    ) -> None:
        arena.replies = [refused(400, "bad_request")]

        with pytest.raises(SnowflakeLicenseError, match="Could not start connecting"):
            connect_arena(session, ORIGIN)

        assert session.claim is None
        assert session.secrets["arena_claim_secret"] == ""

    def test_a_claim_arena_already_linked_collects_the_key_on_the_same_call(
        self, session: KeysSession, arena: Arena
    ) -> None:
        # Arrange
        session.proven_claim()
        assert session.claim is not None
        session.claim["proven"] = False
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        arena.replies = [refused(409, "claim_linked"), issued(), reply(200, ENABLED)]

        # Act
        text = connect_arena(session, ORIGIN)

        # Assert
        assert paths(arena) == [
            "/api/cli/v1/snowflake/claims/claim-1/prove",
            "/api/cli/v1/snowflake/claims/claim-1/key",
            "/api/cli/v1/capabilities",
        ]
        assert text.startswith("Connected to the Arena organisation Acme ML")
        assert session.secrets["arena_org_key"] == ISSUED_KEY
        assert session.claim is None

    def test_a_start_arena_cannot_see_yet_leaves_no_claim(
        self, session: KeysSession, arena: Arena
    ) -> None:
        arena.replies = [refused(409, "install_not_visible_yet")]

        text = connect_arena(session, ORIGIN)

        assert "install_not_visible_yet" in text
        assert text.endswith(
            "Run CALL app_code.connect_arena() again in a few minutes."
        )
        assert session.claim is None
        assert session.secrets["arena_claim_secret"] == ""

    def test_too_many_proven_claims_keeps_this_claim_for_later(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        arena.replies = [claim_started(), refused(429, "too_many_claims")]

        text = connect_arena(session, ORIGIN)

        assert "too_many_claims" in text
        assert len(arena.calls) == 2
        assert session.claim is not None
        assert session.claim["proven"] is False

    def test_a_host_outside_the_network_rule_is_refused(
        self, session: KeysSession, arena: Arena
    ) -> None:
        with pytest.raises(ValueError, match="not in the app's Arena network rule"):
            connect_arena(session, "https://elsewhere.example")

        assert arena.calls == []

    def test_a_base_url_that_does_not_name_one_host_is_refused(
        self, session: KeysSession, arena: Arena
    ) -> None:
        with pytest.raises(ValueError, match="does not name one host") as caught:
            connect_arena(session, "https://*s3cr3t*")

        assert "s3cr3t" not in str(caught.value)
        assert arena.calls == []

    def test_a_claim_for_another_origin_is_replaced(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim()
        assert session.claim is not None
        session.claim["base_url"] = "https://arena.agilerl.com"
        session.challenges = [(REGION, LOCATOR, "claim-1", "code-7f3c")]
        arena.replies = [claim_started(), proven()]

        text = connect_arena(session, ORIGIN)

        assert paths(arena)[0] == "/api/cli/v1/snowflake/claims"
        assert LINK in text
        assert session.claim is not None
        assert session.claim["base_url"] == ORIGIN

    def test_a_connected_key_without_a_price_list_omits_the_families_line(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim()
        caps = {
            "ok": True,
            "data": {"schemaVersion": 1, "features": {"snowflakeTraining": True}},
        }
        arena.replies = [issued(), reply(200, caps)]

        text = connect_arena(session, ORIGIN)

        assert text.splitlines()[0].startswith("Connected to the Arena organisation")
        assert "Priced GPU families" not in text

    def test_a_secret_snowflake_rejects_names_the_secret_not_the_value(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.fail_secret = "arena_claim_secret"
        arena.replies = [claim_started()]

        with pytest.raises(
            RuntimeError,
            match=r"Snowflake did not save app_state\.arena_claim_secret "
            r"\(Snowflake error 1003\)\.",
        ) as caught:
            connect_arena(session, ORIGIN)

        assert CLAIM_SECRET not in str(caught.value)
        assert session.claim is None


class TestCollectLinkedKey:
    def test_collects_a_linked_claims_key_for_the_hourly_task(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim()
        arena.replies = [issued(name=None), reply(200, ENABLED)]

        summary = collect_linked_key(session)

        assert summary == "collected the Arena org key"
        assert session.secrets["arena_org_key"] == ISSUED_KEY
        assert session.claim is None

    def test_a_closed_claim_is_forgotten(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim()
        arena.replies = [refused(410, "claim_collected")]

        summary = collect_linked_key(session)

        assert summary.startswith("claim closed:")
        assert "claim_collected" in summary
        assert session.claim is None
        assert session.secrets["arena_org_key"] == ""
        assert session.secrets["arena_claim_secret"] == ""

    def test_an_unlinked_claim_is_left_for_the_next_run(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.proven_claim()
        arena.replies = [refused(409, "claim_not_linked")]

        summary = collect_linked_key(session)

        assert summary == "claim not linked yet"
        assert session.claim is not None

    def test_no_proven_claim_calls_nothing(
        self, session: KeysSession, arena: Arena
    ) -> None:
        assert collect_linked_key(session) is None
        assert arena.calls == []

    def test_an_arena_outage_is_logged_and_billing_goes_on(
        self, session: KeysSession, arena: Arena, caplog: pytest.LogCaptureFixture
    ) -> None:
        session.proven_claim()
        arena.replies = [reply(503) for _ in range(5)]

        summary = collect_linked_key(session)

        assert summary == "could not collect the Arena org key"
        assert "Could not collect the Arena org key" in caplog.text
        assert session.claim is not None
        assert session.secrets["arena_org_key"] == ""

    def test_a_failed_capabilities_check_keeps_the_issued_key(
        self, session: KeysSession, arena: Arena, caplog: pytest.LogCaptureFixture
    ) -> None:
        session.proven_claim()
        arena.replies = [issued(), reply(503)]

        summary = collect_linked_key(session)

        assert summary == "saved the Arena org key; the Arena check failed"
        assert "Saved the Arena org key" in caplog.text
        assert session.secrets["arena_org_key"] == ISSUED_KEY
        assert session.claim is None


class TestBillingTaskWithAClaim:
    def test_a_proven_claim_keeps_the_task_resumed_until_it_expires(
        self, session: KeysSession
    ) -> None:
        session.proven_claim()

        assert app_billing.sync_billing_task(session) is True
        assert session.task == "started"

    def test_an_unproven_claim_does_not_resume_the_task(
        self, session: KeysSession
    ) -> None:
        session.proven_claim()
        assert session.claim is not None
        session.claim["proven"] = False

        assert app_billing.sync_billing_task(session) is False
        assert session.task == "suspended"


class TestRotateArenaKey:
    def test_saves_the_new_key_then_uses_it_once(
        self, session: KeysSession, arena: Arena
    ) -> None:
        # Arrange
        session.connected_with("connect", age_days=3)
        arena.replies = [issued(), reply(200, ENABLED)]

        # Act
        text = rotate_arena_key(session)

        # Assert
        assert paths(arena) == [
            "/api/cli/v1/snowflake/app-keys/rotate",
            "/api/cli/v1/capabilities",
        ]
        assert (
            arena.calls[0]["headers"]["Authorization"] == "Bearer arena_org_saved-key"
        )
        assert arena.calls[1]["headers"]["Authorization"] == f"Bearer {ISSUED_KEY}"
        assert session.secrets["arena_org_key"] == ISSUED_KEY
        assert ISSUED_KEY not in text
        assert text.startswith("Rotated the Arena org key.")

    def test_an_app_that_is_not_connected_is_refused(
        self, session: KeysSession, arena: Arena
    ) -> None:
        with pytest.raises(ValueError, match="not connected"):
            rotate_arena_key(session)

        assert arena.calls == []

    def test_a_key_agilerl_issued_is_refused(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.connected_with("configure")

        with pytest.raises(ValueError, match="came from AgileRL"):
            rotate_arena_key(session)

        assert arena.calls == []

    def test_a_live_run_holds_the_key_so_nothing_changes(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.connected_with("connect")
        session.services = {"ARENA_20261120_HEAD": "RUNNING"}

        with pytest.raises(ValueError, match="A run is using the current key"):
            rotate_arena_key(session)

        assert arena.calls == []
        assert session.secrets["arena_org_key"] == "arena_org_saved-key"


class TestRotateIfDue:
    def partner(self) -> tuple[str, dict[str, str]]:
        return ORIGIN, partner_headers("arena_org_saved-key", "snowflake-app")

    def test_a_key_over_ninety_days_old_is_rotated(
        self, session: KeysSession, arena: Arena
    ) -> None:
        session.connected_with("connect", age_days=91)
        arena.replies = [issued()]

        origin, headers = rotate_if_due(session, self.partner())

        assert origin == ORIGIN
        assert headers["Authorization"] == f"Bearer {ISSUED_KEY}"
        assert session.secrets["arena_org_key"] == ISSUED_KEY

    @pytest.mark.parametrize(
        ("source", "age_days", "services"),
        [
            ("connect", 89, {}),
            ("configure", 400, {}),
            ("connect", 91, {"ARENA_20261120_HEAD": "RUNNING"}),
        ],
    )
    def test_a_young_hand_issued_or_held_key_is_kept(
        self,
        session: KeysSession,
        arena: Arena,
        source: str,
        age_days: float,
        services: dict[str, str],
    ) -> None:
        session.connected_with(source, age_days=age_days)
        session.services = services

        partner = rotate_if_due(session, self.partner())

        assert partner == self.partner()
        assert arena.calls == []

    def test_a_refused_rotation_keeps_the_current_key(
        self, session: KeysSession, arena: Arena, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)
        session.connected_with("connect", age_days=91)
        arena.replies = [httpx.Response(403, text="Forbidden")]

        partner = rotate_if_due(session, self.partner())

        assert partner == self.partner()
        assert session.secrets["arena_org_key"] == "arena_org_saved-key"
        assert "Could not rotate the Arena org key" in caplog.text


def test_claim_expiry_reaches_the_claim_row(session: KeysSession, arena: Arena) -> None:
    expires = datetime.now(timezone.utc) + timedelta(hours=2)
    body = {
        "ok": True,
        "data": {
            "claimId": "claim-1",
            "claimSecret": CLAIM_SECRET,
            "expiresAt": expires.isoformat(),
        },
    }
    arena.replies = [reply(201, body)]

    connect_arena(session, ORIGIN)

    assert session.claim is not None
    assert (
        abs(int(str(session.claim["expires_ms"])) - expires.timestamp() * 1000) < 1000
    )


class TestAppCapabilities:
    def test_a_refused_key_tells_the_admin_to_connect_again(self, arena: Arena) -> None:
        arena.replies = [httpx.Response(401, text="Unauthorized")]

        with pytest.raises(
            AuthRejectedError, match=r"CALL app_code\.connect_arena\(\)"
        ):
            app_capabilities(ORIGIN, partner_headers("arena_org_saved-key", "u"))
