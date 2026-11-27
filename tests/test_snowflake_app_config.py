# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native App admin procedures: Arena secrets, compute pools, and source references."""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from snowflake.snowpark.exceptions import SnowparkSQLException

from agilerl.snowflake import license as license_mod
from agilerl.snowflake.license import (
    AUTH_REJECTED,
    SNOWFLAKE_DENIED,
    SnowflakeLicenseError,
)
from snowflake_app.app_code.arena_app import (
    IMAGE,
    configure,
    gpu_families,
    register_reference,
    set_pools,
)
from tests.test_snowflake_setup import FakeSession, Row

APP_DIR = Path(__file__).resolve().parents[1] / "snowflake_app"
CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures" / "snowflake_usage_contract.json").read_text(
        encoding="utf-8"
    )
)
ORG_KEY = "arena_org_s3cr3t-key"
USER_ID = "partner-user-7f3a"
ENABLED = {
    "ok": True,
    "data": {"schemaVersion": 1, "features": {"snowflakeTraining": True}},
}
DISABLED = {
    "ok": True,
    "data": {"schemaVersion": 1, "features": {"snowflakeTraining": False}},
}
RULE = (
    lambda s: s == "DESCRIBE NETWORK RULE app_state.arena_api_egress",
    [Row(name="ARENA_API_EGRESS", value_list="arena.agilerl.com:443")],
)
FAMILIES = (
    lambda s: s == "SHOW COMPUTE POOL INSTANCE FAMILIES",
    [
        Row(name="CPU_X64_L", gpu_count=0),
        Row(name="CPU_X64_SL", gpu_count=0),
        Row(name="GPU_NV_S", gpu_count=1),
        Row(name="GPU_NV_M", gpu_count=4),
    ],
)
GPU_NV_S_POOL = Row(
    name="AGILERL_APP_ARENA_APP_GPU_GPU_NV_S", instance_family="GPU_NV_S", max_nodes=2
)
PREFIX = "SELECT app_code.object_prefix() AS PREFIX"


def arena_replies(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, object], status: int = 200
) -> list[httpx.Request]:
    """Route ``license.py``'s HTTP client to one canned Arena reply.

    :return: Requests Arena received.
    """
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    client = functools.partial(httpx.Client, transport=httpx.MockTransport(answer))
    monkeypatch.setattr(license_mod.httpx, "Client", client)
    return seen


def service_row(status: str, nodes: int) -> Row:
    return Row(
        database_name="APP",
        schema_name="APP_STATE",
        name="ARENA_RUN_1_WORKERS",
        status=status,
        current_instances=nodes,
    )


def secret_sql(session: FakeSession) -> list[str]:
    return [s for s in session.statements if "SECRET" in s]


def pools_session(
    pools: list[Row],
    services: dict[str, list[Row]] | None = None,
    setup_doc: dict[str, object] | None = None,
    prefix: str = "AGILERL_APP_ARENA_APP_",
    families: tuple[object, list[Row]] = FAMILIES,
) -> FakeSession:
    doc_rows = [] if setup_doc is None else [Row(DOC=json.dumps(setup_doc))]
    routes: list[tuple[object, object]] = [
        families,
        (lambda s: s == PREFIX, [Row(PREFIX=prefix)]),
        (lambda s: s.startswith("SELECT $1 AS DOC FROM @app_state.runs"), doc_rows),
        (lambda s: s == "SHOW COMPUTE POOLS", pools),
    ]
    routes.extend(
        (lambda s, pool=pool: s == f"SHOW SERVICES IN COMPUTE POOL {pool}", rows)
        for pool, rows in (services or {}).items()
    )
    return FakeSession(routes)


class TestConfigure:
    def test_saves_three_secrets_after_arena_allows_snowflake_training(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        requests = arena_replies(monkeypatch, ENABLED)
        session = FakeSession([RULE])

        # Act
        text = configure(session, ORG_KEY, USER_ID, "https://arena.agilerl.com/")

        # Assert
        [request] = requests
        assert str(request.url) == "https://arena.agilerl.com/api/cli/v1/capabilities"
        assert all(name in request.headers for name in CONTRACT["headers"])
        assert request.headers["Authorization"] == f"Bearer {ORG_KEY}"
        assert request.headers["X-External-User-Id"] == USER_ID
        assert secret_sql(session) == [
            (
                "CREATE SECRET IF NOT EXISTS app_state.arena_org_key\n"
                "  TYPE = GENERIC_STRING\n"
                f"  SECRET_STRING = '{ORG_KEY}'"
            ),
            f"ALTER SECRET app_state.arena_org_key SET SECRET_STRING = '{ORG_KEY}'",
            (
                "CREATE SECRET IF NOT EXISTS app_state.arena_external_user_id\n"
                "  TYPE = GENERIC_STRING\n"
                f"  SECRET_STRING = '{USER_ID}'"
            ),
            (
                "ALTER SECRET app_state.arena_external_user_id "
                f"SET SECRET_STRING = '{USER_ID}'"
            ),
            (
                "CREATE SECRET IF NOT EXISTS app_state.arena_base_url\n"
                "  TYPE = GENERIC_STRING\n"
                "  SECRET_STRING = 'https://arena.agilerl.com'"
            ),
            (
                "ALTER SECRET app_state.arena_base_url "
                "SET SECRET_STRING = 'https://arena.agilerl.com'"
            ),
        ]
        assert ORG_KEY not in text
        assert USER_ID not in text
        assert session.file.uploads == []

    @pytest.mark.parametrize(
        ("priced", "line"),
        [
            (["GPU_NV_S", "gpu_nv_m"], "Priced GPU families: GPU_NV_M, GPU_NV_S."),
            ([], "Priced GPU families: none."),
        ],
    )
    def test_lists_the_gpu_families_arena_has_a_price_for(
        self, monkeypatch: pytest.MonkeyPatch, priced: list[str], line: str
    ) -> None:
        # Arrange
        features = {"snowflakeTraining": True, "snowflakePricedGpuFamilies": priced}
        arena_replies(
            monkeypatch,
            {"ok": True, "data": {"schemaVersion": 1, "features": features}},
        )

        # Act
        text = configure(
            FakeSession([RULE]), ORG_KEY, USER_ID, "https://arena.agilerl.com"
        )

        # Assert
        assert text.splitlines()[-1] == line

    def test_lists_no_prices_when_arena_sends_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arena_replies(monkeypatch, ENABLED)

        text = configure(
            FakeSession([RULE]), ORG_KEY, USER_ID, "https://arena.agilerl.com"
        )

        assert "Priced" not in text

    def test_secret_values_appear_only_in_the_secret_statements(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arena_replies(monkeypatch, ENABLED)
        session = FakeSession([RULE])

        configure(session, ORG_KEY, USER_ID, "https://arena.agilerl.com")

        others = [s for s in session.statements if s not in secret_sql(session)]
        assert others == [
            "DESCRIBE NETWORK RULE app_state.arena_api_egress",
            "DELETE FROM app_state.arena_key",
            (
                "INSERT INTO app_state.arena_key (source, issued_at) "
                "VALUES ('configure', CURRENT_TIMESTAMP())"
            ),
        ]

    def test_refuses_a_key_without_the_org_prefix_before_any_sql(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = arena_replies(monkeypatch, ENABLED)
        session = FakeSession([RULE])

        with pytest.raises(ValueError, match="starts with arena_org_") as caught:
            configure(session, "sk-live-123", USER_ID, "https://arena.agilerl.com")

        assert "sk-live-123" not in str(caught.value)
        assert session.statements == []
        assert requests == []

    def test_refuses_an_empty_external_user_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arena_replies(monkeypatch, ENABLED)
        session = FakeSession([RULE])

        with pytest.raises(ValueError, match="external user id"):
            configure(session, ORG_KEY, "  ", "https://arena.agilerl.com")

        assert session.statements == []

    def test_refuses_without_snowflake_training_and_saves_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arena_replies(monkeypatch, DISABLED)
        session = FakeSession([RULE])

        with pytest.raises(SnowflakeLicenseError, match=SNOWFLAKE_DENIED) as caught:
            configure(session, ORG_KEY, USER_ID, "https://arena.agilerl.com")

        assert secret_sql(session) == []
        assert ORG_KEY not in str(caught.value)
        assert USER_ID not in str(caught.value)

    def test_rejected_credentials_save_nothing_and_are_not_quoted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arena_replies(monkeypatch, {"error": "unauthorized"}, status=401)
        session = FakeSession([RULE])

        with pytest.raises(SnowflakeLicenseError, match=AUTH_REJECTED) as caught:
            configure(session, ORG_KEY, USER_ID, "https://arena.agilerl.com")

        assert secret_sql(session) == []
        assert ORG_KEY not in str(caught.value)

    def test_refuses_a_host_outside_the_arena_rule_before_calling_arena(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = arena_replies(monkeypatch, ENABLED)
        session = FakeSession([RULE])

        with pytest.raises(
            ValueError,
            match=r"not in the app's Arena network rule \(arena.agilerl.com:443\)",
        ):
            configure(session, ORG_KEY, USER_ID, "https://arena.example.com")

        assert requests == []
        assert secret_sql(session) == []

    def test_a_null_base_url_uses_the_default_arena_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        requests = arena_replies(monkeypatch, ENABLED)
        session = FakeSession([RULE])

        # Act
        configure(session, ORG_KEY, USER_ID, None)

        # Assert
        [request] = requests
        assert str(request.url) == "https://arena.agilerl.com/api/cli/v1/capabilities"
        assert any(
            "arena_base_url" in statement and "https://arena.agilerl.com" in statement
            for statement in session.statements
        )

    def test_a_secret_pasted_as_the_base_url_is_not_quoted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arena_replies(monkeypatch, ENABLED)
        session = FakeSession([RULE])

        with pytest.raises(ValueError, match="does not name one host") as caught:
            configure(session, ORG_KEY, USER_ID, "https://*s3cr3t*")

        assert "s3cr3t" not in str(caught.value)
        assert session.statements == []

    def test_a_failed_secret_statement_names_the_secret_not_the_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arena_replies(monkeypatch, ENABLED)
        failure = SnowparkSQLException(
            f"SQL compilation error near '{ORG_KEY}'", sql_error_code=1003
        )
        session = FakeSession([RULE, (lambda s: "arena_org_key" in s, failure)])

        with pytest.raises(
            RuntimeError,
            match=r"Snowflake did not save app_state\.arena_org_key "
            r"\(Snowflake error 1003\)\.$",
        ) as caught:
            configure(session, ORG_KEY, USER_ID, "https://arena.agilerl.com")

        assert ORG_KEY not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True


class TestRegisterReference:
    @pytest.mark.parametrize(
        ("operation", "statement"),
        [
            ("ADD", "SELECT SYSTEM$ADD_REFERENCE('SOURCE_TABLE', 'ref-1')"),
            ("REMOVE", "SELECT SYSTEM$REMOVE_REFERENCE('SOURCE_TABLE', 'ref-1')"),
            ("CLEAR", "SELECT SYSTEM$REMOVE_ALL_REFERENCES('SOURCE_TABLE')"),
        ],
    )
    def test_runs_the_reference_function_for_each_operation(
        self, operation: str, statement: str
    ) -> None:
        session = FakeSession()

        register_reference(session, "source_table", operation, "ref-1")

        assert session.statements == [statement]

    def test_binds_a_view(self) -> None:
        session = FakeSession()

        register_reference(session, "SOURCE_VIEW", "ADD", "ref-'2")

        assert session.statements == [
            "SELECT SYSTEM$ADD_REFERENCE('SOURCE_VIEW', 'ref-''2')"
        ]

    @pytest.mark.parametrize(
        ("operation", "statement"),
        [
            ("ADD", "SELECT SYSTEM$SET_REFERENCE('EXPORT_STAGE', 'ref-1')"),
            ("REMOVE", "SELECT SYSTEM$REMOVE_REFERENCE('EXPORT_STAGE', 'ref-1')"),
            ("CLEAR", "SELECT SYSTEM$REMOVE_ALL_REFERENCES('EXPORT_STAGE')"),
        ],
    )
    def test_export_stage_holds_one_stage(self, operation: str, statement: str) -> None:
        session = FakeSession()

        register_reference(session, "export_stage", operation, "ref-1")

        assert session.statements == [statement]

    @pytest.mark.parametrize(
        ("operation", "statement"),
        [
            ("ADD", "SELECT SYSTEM$SET_REFERENCE('REWARD_FUNCTION', 'ref-1')"),
            ("REMOVE", "SELECT SYSTEM$REMOVE_REFERENCE('REWARD_FUNCTION', 'ref-1')"),
            ("CLEAR", "SELECT SYSTEM$REMOVE_ALL_REFERENCES('REWARD_FUNCTION')"),
        ],
    )
    def test_reward_function_holds_one_function(
        self, operation: str, statement: str
    ) -> None:
        session = FakeSession()

        register_reference(session, "reward_function", operation, "ref-1")

        assert session.statements == [statement]

    def test_refuses_an_unknown_reference(self) -> None:
        session = FakeSession()

        with pytest.raises(ValueError, match="Unknown reference consumer_secret"):
            register_reference(session, "consumer_secret", "ADD", "ref-1")

        assert session.statements == []

    def test_refuses_an_unknown_operation(self) -> None:
        session = FakeSession()

        with pytest.raises(ValueError, match="Unknown operation REPLACE"):
            register_reference(session, "SOURCE_TABLE", "REPLACE", "ref-1")

        assert session.statements == []


class TestSetPools:
    def test_null_families_use_the_default_broker_and_head_families(self) -> None:
        session = pools_session([])

        text = set_pools(session, ["gpu_nv_m"], 8, None, None)

        assert text.splitlines()[:2] == [
            "AGILERL_APP_ARENA_APP_BROKER_POOL  CPU_X64_L  broker",
            "AGILERL_APP_ARENA_APP_HEAD_POOL  CPU_X64_SL  head",
        ]

    def test_creates_broker_head_and_gpu_pools_with_the_app_names(self) -> None:
        # Arrange
        session = pools_session([])

        # Act
        text = set_pools(session, ["gpu_nv_m"], 8, "CPU_X64_L", "CPU_X64_SL")

        # Assert
        creates = [s for s in session.statements if s.startswith("CREATE COMPUTE POOL")]
        assert creates == [
            (
                "CREATE COMPUTE POOL IF NOT EXISTS AGILERL_APP_ARENA_APP_BROKER_POOL\n"
                "  MIN_NODES = 1\n  MAX_NODES = 4\n  INSTANCE_FAMILY = CPU_X64_L\n"
                "  AUTO_RESUME = TRUE\n  AUTO_SUSPEND_SECS = 300\n"
                "  INITIALLY_SUSPENDED = TRUE"
            ),
            (
                "CREATE COMPUTE POOL IF NOT EXISTS AGILERL_APP_ARENA_APP_HEAD_POOL\n"
                "  MIN_NODES = 1\n  MAX_NODES = 4\n  INSTANCE_FAMILY = CPU_X64_SL\n"
                "  AUTO_RESUME = TRUE\n  AUTO_SUSPEND_SECS = 300\n"
                "  INITIALLY_SUSPENDED = TRUE"
            ),
            (
                "CREATE COMPUTE POOL IF NOT EXISTS AGILERL_APP_ARENA_APP_GPU_GPU_NV_M\n"
                "  MIN_NODES = 1\n  MAX_NODES = 2\n  INSTANCE_FAMILY = GPU_NV_M\n"
                "  AUTO_RESUME = TRUE\n  AUTO_SUSPEND_SECS = 300\n"
                "  INITIALLY_SUSPENDED = TRUE"
            ),
        ]
        assert text == (
            "AGILERL_APP_ARENA_APP_BROKER_POOL  CPU_X64_L  broker\n"
            "AGILERL_APP_ARENA_APP_HEAD_POOL  CPU_X64_SL  head\n"
            "AGILERL_APP_ARENA_APP_GPU_GPU_NV_M  GPU_NV_M  max 8 GPUs  MAX_NODES 2"
        )
        assert not any("BUDGET" in s or "GRANT" in s for s in session.statements)

    def test_writes_setup_json_in_the_cli_shape_to_the_app_stage(self) -> None:
        session = pools_session([])

        set_pools(session, ["GPU_NV_S"], None, "CPU_X64_L", "CPU_X64_SL")

        assert session.file.stages == ["@app_state.runs/_arena"]
        assert session.file.uploads == [
            {
                "image": IMAGE,
                "warehouse": "AGILERL_APP_ARENA_APP_WH",
                "stage": "app_state.runs",
                "metrics_table": "app_state.arena_metrics",
                "external_access_integrations": ["AGILERL_APP_ARENA_APP_HF_EAI"],
                "pools": {
                    "broker": {
                        "pool": "AGILERL_APP_ARENA_APP_BROKER_POOL",
                        "family": "CPU_X64_L",
                    },
                    "head": {
                        "pool": "AGILERL_APP_ARENA_APP_HEAD_POOL",
                        "family": "CPU_X64_SL",
                    },
                    "gpu": {
                        "GPU_NV_S": {
                            "pool": "AGILERL_APP_ARENA_APP_GPU_GPU_NV_S",
                            "max_gpus": 2,
                            "gpus_per_node": 1,
                            "gpu_memory_gib": 24,
                        }
                    },
                },
            }
        ]

    def test_resizes_an_existing_pool_and_keeps_the_saved_cap_when_none_is_given(
        self,
    ) -> None:
        # Arrange
        saved = {
            "image": IMAGE,
            "warehouse": "AGILERL_APP_ARENA_APP_WH",
            "stage": "app_state.runs",
            "metrics_table": "app_state.arena_metrics",
            "external_access_integrations": [
                "AGILERL_APP_ARENA_APP_HF_EAI",
                "arena_env_eai",
            ],
            "pools": {
                "broker": {
                    "pool": "AGILERL_APP_ARENA_APP_BROKER_POOL",
                    "family": "CPU_X64_L",
                },
                "head": {
                    "pool": "AGILERL_APP_ARENA_APP_HEAD_POOL",
                    "family": "CPU_X64_SL",
                },
                "gpu": {
                    "GPU_NV_S": {
                        "pool": "AGILERL_APP_ARENA_APP_GPU_GPU_NV_S",
                        "max_gpus": 5,
                        "gpus_per_node": 1,
                        "gpu_memory_gib": 24,
                    }
                },
            },
        }
        session = pools_session([GPU_NV_S_POOL], setup_doc=saved)

        # Act
        set_pools(session, ["GPU_NV_S"], None, "CPU_X64_L", "CPU_X64_SL")

        # Assert
        assert (
            "ALTER COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_S SET MAX_NODES = 5"
            in session.statements
        )
        assert not any(
            "CREATE COMPUTE POOL IF NOT EXISTS AGILERL_APP_ARENA_APP_GPU_GPU_NV_S" in s
            for s in session.statements
        )
        [doc] = session.file.uploads
        assert doc["external_access_integrations"] == [
            "AGILERL_APP_ARENA_APP_HF_EAI",
            "arena_env_eai",
        ]
        assert doc["pools"]["gpu"]["GPU_NV_S"]["max_gpus"] == 5

    def test_drops_gpu_pools_for_families_no_longer_listed(self) -> None:
        session = pools_session([GPU_NV_S_POOL])

        text = set_pools(session, ["GPU_NV_M"], 4, "CPU_X64_L", "CPU_X64_SL")

        assert session.statements[-2:] == [
            "ALTER COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_S STOP ALL",
            "DROP COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_S",
        ]
        assert text.endswith("AGILERL_APP_ARENA_APP_GPU_GPU_NV_S  dropped")
        assert list(session.file.uploads[0]["pools"]["gpu"]) == ["GPU_NV_M"]

    def test_a_second_app_names_its_pools_warehouse_and_integration_after_itself(
        self,
    ) -> None:
        # Arrange
        session = pools_session([GPU_NV_S_POOL], prefix="AGILERL_APP_ARENA_B_")

        # Act
        text = set_pools(session, ["GPU_NV_S"], 1, "CPU_X64_L", "CPU_X64_SL")

        # Assert
        assert text.splitlines() == [
            "AGILERL_APP_ARENA_B_BROKER_POOL  CPU_X64_L  broker",
            "AGILERL_APP_ARENA_B_HEAD_POOL  CPU_X64_SL  head",
            "AGILERL_APP_ARENA_B_GPU_GPU_NV_S  GPU_NV_S  max 1 GPUs  MAX_NODES 1",
        ]
        assert not any("AGILERL_APP_ARENA_APP_" in s for s in session.statements)
        [doc] = session.file.uploads
        assert doc["warehouse"] == "AGILERL_APP_ARENA_B_WH"
        assert doc["external_access_integrations"] == ["AGILERL_APP_ARENA_B_HF_EAI"]

    def test_keeps_a_gpu_pool_from_before_pools_were_named_per_app(self) -> None:
        old = Row(
            name="AGILERL_APP_GPU_GPU_NV_S", instance_family="GPU_NV_S", max_nodes=2
        )
        session = pools_session([old])

        text = set_pools(session, ["GPU_NV_M"], 4, "CPU_X64_L", "CPU_X64_SL")

        assert not any(s.startswith("DROP") for s in session.statements)
        assert "dropped" not in text

    def test_refuses_to_shrink_under_a_running_service(self) -> None:
        # Arrange
        running = service_row("RUNNING", 2)
        session = pools_session(
            [GPU_NV_S_POOL],
            services={"AGILERL_APP_ARENA_APP_GPU_GPU_NV_S": [running]},
        )

        # Act
        with pytest.raises(
            ValueError,
            match=r"APP\.APP_STATE\.ARENA_RUN_1_WORKERS needs 2 nodes\. "
            r"MAX_NODES 1 is below that\.\nThe pool was not changed\.",
        ):
            set_pools(session, ["GPU_NV_S"], 1, "CPU_X64_L", "CPU_X64_SL")

        # Assert
        assert not any(
            s.startswith(("CREATE", "ALTER", "DROP")) for s in session.statements
        )
        assert session.file.uploads == []

    def test_refuses_to_drop_a_leftover_pool_with_a_running_service(self) -> None:
        running = service_row("RUNNING", 1)
        session = pools_session(
            [GPU_NV_S_POOL], services={"AGILERL_APP_ARENA_APP_GPU_GPU_NV_S": [running]}
        )

        with pytest.raises(
            ValueError,
            match=r"AGILERL_APP_ARENA_APP_GPU_GPU_NV_S is no longer listed, so it would be dropped\.\n"
            r"APP\.APP_STATE\.ARENA_RUN_1_WORKERS needs 1 node\.",
        ):
            set_pools(session, ["GPU_NV_M"], 4, "CPU_X64_L", "CPU_X64_SL")

        assert not any(s.startswith("DROP") for s in session.statements)

    def test_refuses_to_drop_a_pool_with_a_job_waiting_for_a_node(self) -> None:
        # Arrange
        queued = Row(
            database_name="APP",
            schema_name="APP_STATE",
            name="ARENA_RUN_2_WORKERS",
            status="PENDING",
            current_instances=0,
            target_instances=1,
            min_instances=1,
        )
        session = pools_session(
            [GPU_NV_S_POOL], services={"AGILERL_APP_ARENA_APP_GPU_GPU_NV_S": [queued]}
        )

        # Act
        with pytest.raises(ValueError, match="no longer listed") as raised:
            set_pools(session, ["GPU_NV_M"], 4, "CPU_X64_L", "CPU_X64_SL")

        # Assert
        assert str(raised.value) == (
            "AGILERL_APP_ARENA_APP_GPU_GPU_NV_S is no longer listed, "
            "so it would be dropped.\n"
            "APP.APP_STATE.ARENA_RUN_2_WORKERS needs 1 node. MAX_NODES 0 is below that.\n"
            "The pool was not changed."
        )
        assert not any(
            s.startswith(("CREATE", "ALTER", "DROP")) for s in session.statements
        )
        assert session.file.uploads == []

    def test_refuses_to_shrink_below_the_nodes_a_queued_job_asked_for(self) -> None:
        queued = Row(
            database_name="APP",
            schema_name="APP_STATE",
            name="ARENA_RUN_2_WORKERS",
            status="PENDING",
            current_instances=0,
            target_instances=2,
            min_instances=2,
        )
        session = pools_session(
            [GPU_NV_S_POOL], services={"AGILERL_APP_ARENA_APP_GPU_GPU_NV_S": [queued]}
        )

        with pytest.raises(ValueError, match=r"needs 2 nodes\. MAX_NODES 1"):
            set_pools(session, ["GPU_NV_S"], 1, "CPU_X64_L", "CPU_X64_SL")

        assert not any(s.startswith("ALTER") for s in session.statements)

    def test_refuses_a_cap_that_is_not_whole_nodes_before_any_sql(self) -> None:
        session = pools_session([GPU_NV_S_POOL])

        with pytest.raises(ValueError, match="multiple of 4") as raised:
            set_pools(session, ["GPU_NV_M", "GPU_NV_S"], 1, "CPU_X64_L", "CPU_X64_SL")

        assert str(raised.value) == (
            "GPU_NV_M has 4 GPUs per node, so its cap is a multiple of 4: "
            "4, 8, 12 and so on. 1 is not.\n"
            "No pool was changed."
        )
        assert session.statements == []

    @pytest.mark.parametrize("status", ["DONE", "INTERNAL_ERROR", "DELETED"])
    def test_a_finished_service_does_not_block_a_shrink(self, status: str) -> None:
        done = service_row(status, 2)
        session = pools_session(
            [GPU_NV_S_POOL], services={"AGILERL_APP_ARENA_APP_GPU_GPU_NV_S": [done]}
        )

        set_pools(session, ["GPU_NV_S"], 1, "CPU_X64_L", "CPU_X64_SL")

        assert (
            "ALTER COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_S SET MAX_NODES = 1"
            in session.statements
        )

    def test_refuses_an_unknown_gpu_family_before_any_sql(self) -> None:
        session = pools_session([])

        with pytest.raises(
            ValueError, match="GPU_XL_9 is not a GPU family Arena sizes"
        ):
            set_pools(session, ["GPU_XL_9"], 4, "CPU_X64_L", "CPU_X64_SL")

        assert session.statements == []

    def test_refuses_a_gpu_family_missing_from_the_region_listing_the_ones_it_has(
        self,
    ) -> None:
        session = pools_session([])

        with pytest.raises(ValueError, match="no GPU_NV_L pool") as raised:
            set_pools(session, ["GPU_NV_L"], 8, "CPU_X64_L", "CPU_X64_SL")

        assert str(raised.value) == (
            "This account has no GPU_NV_L pool in this region (8x A100). "
            "Arena did not create a compute pool.\n"
            "GPU families Arena can size in this region:\n"
            "  GPU_NV_S  1x A10G\n"
            "  GPU_NV_M  4x A10G\n"
            "Rerun with one of those names:\n"
            "  CALL <app>.app_code.set_pools(['<name>'])\n"
            "A CPU pool cannot run this trainer.\n"
            "Nothing was created."
        )
        assert session.statements == ["SHOW COMPUTE POOL INSTANCE FAMILIES"]

    def test_creates_an_l4_pool_where_the_region_sizes_it_as_the_app_does(
        self,
    ) -> None:
        session = pools_session([], families=GCP_FAMILIES)

        text = set_pools(session, ["GPU_GCP_NV_L4_4_24G"], 8, None, None)

        assert text.splitlines()[2] == (
            "AGILERL_APP_ARENA_APP_GPU_GPU_GCP_NV_L4_4_24G  GPU_GCP_NV_L4_4_24G  "
            "max 8 GPUs  MAX_NODES 2"
        )

    def test_creates_an_r6k_pool_where_the_region_has_the_aws_gpu_count(
        self,
    ) -> None:
        session = pools_session([], families=AWS_R6K_FAMILIES)

        text = set_pools(session, ["GPU_R6K_G1_48"], 4, None, None)

        assert text.splitlines()[2] == (
            "AGILERL_APP_ARENA_APP_GPU_GPU_R6K_G1_48  GPU_R6K_G1_48  "
            "max 4 GPUs  MAX_NODES 2"
        )

    def test_refuses_a_family_the_region_has_with_another_gpu_count(self) -> None:
        # Arrange
        session = pools_session([], families=GCP_FAMILIES)

        # Act
        with pytest.raises(
            ValueError, match="GPU_R6K_G1_48 is 1x RTX PRO 6000"
        ) as raised:
            set_pools(session, ["GPU_R6K_G1_48"], 2, None, None)

        # Assert
        assert str(raised.value) == (
            "GPU_R6K_G1_48 is 1x RTX PRO 6000 per node in this region, not the "
            "2x RTX PRO 6000 Arena sizes. Arena did not create a compute pool.\n"
            "GPU families Arena can size in this region:\n"
            "  GPU_GCP_NV_L4_1_24G  1x L4\n"
            "  GPU_GCP_NV_L4_4_24G  4x L4\n"
            "Rerun with one of those names:\n"
            "  CALL <app>.app_code.set_pools(['<name>'])\n"
            "A CPU pool cannot run this trainer.\n"
            "Nothing was created."
        )
        assert session.statements == ["SHOW COMPUTE POOL INSTANCE FAMILIES"]

    def test_refuses_a_family_the_app_does_not_size_before_any_sql(self) -> None:
        session = pools_session([], families=GCP_FAMILIES)

        with pytest.raises(
            ValueError, match="GPU_R6K_G1_384 is not a GPU family Arena sizes"
        ):
            set_pools(session, ["GPU_R6K_G1_384"], 8, None, None)

        assert session.statements == []

    def test_refuses_a_cpu_family_missing_from_the_region(self) -> None:
        session = pools_session([])

        with pytest.raises(
            ValueError, match="This account has no CPU_X64_M instance family"
        ):
            set_pools(session, ["GPU_NV_S"], 2, "CPU_X64_M", "CPU_X64_SL")

        assert session.statements == ["SHOW COMPUTE POOL INSTANCE FAMILIES"]

    def test_refuses_a_pool_that_uses_another_family(self) -> None:
        other = Row(
            name="AGILERL_APP_ARENA_APP_HEAD_POOL",
            instance_family="CPU_X64_L",
            max_nodes=4,
        )
        session = pools_session([other])

        with pytest.raises(
            ValueError, match="Pool AGILERL_APP_ARENA_APP_HEAD_POOL is CPU_X64_L"
        ):
            set_pools(session, ["GPU_NV_S"], 2, "CPU_X64_L", "CPU_X64_SL")

        assert session.file.uploads == []

    def test_refuses_an_empty_family_list(self) -> None:
        session = pools_session([])

        with pytest.raises(ValueError, match="Name at least one GPU family"):
            set_pools(session, [], 4, "CPU_X64_L", "CPU_X64_SL")

        assert session.statements == []


# GCP London: Snowflake reuses the AWS R6K names for nodes with half the GPUs.
GCP_FAMILIES = (
    lambda s: s == "SHOW COMPUTE POOL INSTANCE FAMILIES",
    [
        Row(name="CPU_X64_L", gpu_count=0),
        Row(name="CPU_X64_SL", gpu_count=0),
        Row(name="GPU_GCP_NV_L4_1_24G", gpu_count=1),
        Row(name="GPU_GCP_NV_L4_4_24G", gpu_count=4),
        Row(name="GPU_R6K_G1_48", gpu_count=1, gpu="RTX PRO 6000"),
        Row(name="GPU_R6K_G1_384", gpu_count=8, gpu="RTX PRO 6000"),
        Row(name="GPU_H999_X8", gpu_count=8, gpu="H999"),
    ],
)
AWS_R6K_FAMILIES = (
    lambda s: s == "SHOW COMPUTE POOL INSTANCE FAMILIES",
    [
        Row(name="CPU_X64_L", gpu_count=0),
        Row(name="CPU_X64_SL", gpu_count=0),
        Row(name="GPU_R6K_G1_48", gpu_count=2, gpu="RTX PRO 6000"),
    ],
)


def priced_replies(
    monkeypatch: pytest.MonkeyPatch, priced: list[str] | None
) -> list[httpx.Request]:
    """Saved Arena secrets, and capabilities pricing ``priced`` (``None``: no price list)."""
    monkeypatch.setitem(
        sys.modules,
        "_snowflake",
        SimpleNamespace(
            get_generic_secret_string={
                "org_key": ORG_KEY,
                "external_user_id": USER_ID,
                "base_url": "https://arena.agilerl.com",
            }.get
        ),
    )
    features: dict[str, object] = {"snowflakeTraining": True}
    if priced is not None:
        features["snowflakePricedGpuFamilies"] = priced
    return arena_replies(
        monkeypatch, {"ok": True, "data": {"schemaVersion": 1, "features": features}}
    )


class TestGpuFamilies:
    def test_lists_the_region_families_the_app_sizes_and_arena_prices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        requests = priced_replies(monkeypatch, ["GPU_GCP_NV_L4_1_24G", "GPU_NV_S"])
        session = FakeSession([GCP_FAMILIES])

        # Act
        text = gpu_families(session)

        # Assert
        assert text == (
            "GPU families set_pools can use here:\n  GPU_GCP_NV_L4_1_24G  1x L4"
        )
        [request] = requests
        assert request.headers["Authorization"] == f"Bearer {ORG_KEY}"
        assert session.statements == ["SHOW COMPUTE POOL INSTANCE FAMILIES"]

    def test_omits_families_the_region_has_with_another_gpu_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        priced_replies(monkeypatch, None)

        text = gpu_families(FakeSession([GCP_FAMILIES]))

        assert text == (
            "GPU families set_pools can use here:\n"
            "  GPU_GCP_NV_L4_1_24G  1x L4\n"
            "  GPU_GCP_NV_L4_4_24G  4x L4"
        )

    def test_lists_an_r6k_family_the_region_has_with_the_aws_gpu_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        priced_replies(monkeypatch, ["GPU_R6K_G1_48"])

        text = gpu_families(FakeSession([AWS_R6K_FAMILIES]))

        assert text == (
            "GPU families set_pools can use here:\n  GPU_R6K_G1_48  2x RTX PRO 6000"
        )

    def test_names_the_region_gpu_families_when_none_is_usable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        priced_replies(monkeypatch, ["GPU_NV_S"])

        text = gpu_families(FakeSession([GCP_FAMILIES]))

        assert text == (
            "No GPU family in this region is both sized by the app and priced by Arena.\n"
            "GPU families SHOW COMPUTE POOL INSTANCE FAMILIES returned:\n"
            "  GPU_GCP_NV_L4_1_24G  1x L4\n"
            "  GPU_GCP_NV_L4_4_24G  4x L4\n"
            "  GPU_R6K_G1_48  1x RTX PRO 6000\n"
            "  GPU_R6K_G1_384  8x RTX PRO 6000\n"
            "  GPU_H999_X8  8x H999"
        )

    def test_refuses_before_any_sql_when_arena_is_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            sys.modules,
            "_snowflake",
            SimpleNamespace(get_generic_secret_string=lambda name: ""),
        )
        session = FakeSession([GCP_FAMILIES])

        with pytest.raises(ValueError, match="Arena is not connected"):
            gpu_families(session)

        assert session.statements == []


class TestManifestImage:
    def test_procedures_record_the_manifest_image(self) -> None:
        manifest = (APP_DIR / "manifest.yml").read_text(encoding="utf-8")

        assert f"- {IMAGE}\n" in manifest
