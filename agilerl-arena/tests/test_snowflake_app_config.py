# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native App admin procedures: Arena secrets, compute pools, and source references."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import httpx
import pytest
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake_app.app_code.arena_app import (
    IMAGE,
    configure,
    register_reference,
    set_pools,
)

from agilerl.arena.snowflake import license as license_mod
from agilerl.arena.snowflake.license import (
    AUTH_REJECTED,
    SNOWFLAKE_DENIED,
    SnowflakeLicenseError,
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
    name="AGILERL_APP_GPU_GPU_NV_S", instance_family="GPU_NV_S", max_nodes=2
)


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
) -> FakeSession:
    doc_rows = [] if setup_doc is None else [Row(DOC=json.dumps(setup_doc))]
    routes: list[tuple[object, object]] = [
        FAMILIES,
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

    def test_secret_values_appear_only_in_the_secret_statements(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arena_replies(monkeypatch, ENABLED)
        session = FakeSession([RULE])

        configure(session, ORG_KEY, USER_ID, "https://arena.agilerl.com")

        others = [s for s in session.statements if s not in secret_sql(session)]
        assert others == ["DESCRIBE NETWORK RULE app_state.arena_api_egress"]

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
            "agilerl_app_broker_pool  CPU_X64_L  broker",
            "agilerl_app_head_pool  CPU_X64_SL  head",
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
                "CREATE COMPUTE POOL IF NOT EXISTS agilerl_app_broker_pool\n"
                "  MIN_NODES = 1\n  MAX_NODES = 4\n  INSTANCE_FAMILY = CPU_X64_L\n"
                "  AUTO_RESUME = TRUE\n  AUTO_SUSPEND_SECS = 300\n"
                "  INITIALLY_SUSPENDED = TRUE"
            ),
            (
                "CREATE COMPUTE POOL IF NOT EXISTS agilerl_app_head_pool\n"
                "  MIN_NODES = 1\n  MAX_NODES = 4\n  INSTANCE_FAMILY = CPU_X64_SL\n"
                "  AUTO_RESUME = TRUE\n  AUTO_SUSPEND_SECS = 300\n"
                "  INITIALLY_SUSPENDED = TRUE"
            ),
            (
                "CREATE COMPUTE POOL IF NOT EXISTS agilerl_app_gpu_GPU_NV_M\n"
                "  MIN_NODES = 1\n  MAX_NODES = 2\n  INSTANCE_FAMILY = GPU_NV_M\n"
                "  AUTO_RESUME = TRUE\n  AUTO_SUSPEND_SECS = 300\n"
                "  INITIALLY_SUSPENDED = TRUE"
            ),
        ]
        assert text == (
            "agilerl_app_broker_pool  CPU_X64_L  broker\n"
            "agilerl_app_head_pool  CPU_X64_SL  head\n"
            "agilerl_app_gpu_GPU_NV_M  GPU_NV_M  max 8 GPUs  MAX_NODES 2"
        )
        assert not any("BUDGET" in s or "GRANT" in s for s in session.statements)

    def test_writes_setup_json_in_the_cli_shape_to_the_app_stage(self) -> None:
        session = pools_session([])

        set_pools(session, ["GPU_NV_S"], None, "CPU_X64_L", "CPU_X64_SL")

        assert session.file.stages == ["@app_state.runs/_arena"]
        assert session.file.uploads == [
            {
                "image": IMAGE,
                "warehouse": "agilerl_app_wh",
                "stage": "app_state.runs",
                "metrics_table": "app_state.arena_metrics",
                "external_access_integrations": ["agilerl_app_hf_eai"],
                "pools": {
                    "broker": {
                        "pool": "agilerl_app_broker_pool",
                        "family": "CPU_X64_L",
                    },
                    "head": {"pool": "agilerl_app_head_pool", "family": "CPU_X64_SL"},
                    "gpu": {
                        "GPU_NV_S": {
                            "pool": "agilerl_app_gpu_GPU_NV_S",
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
            "warehouse": "agilerl_app_wh",
            "stage": "app_state.runs",
            "metrics_table": "app_state.arena_metrics",
            "external_access_integrations": ["agilerl_app_hf_eai", "arena_env_eai"],
            "pools": {
                "broker": {"pool": "agilerl_app_broker_pool", "family": "CPU_X64_L"},
                "head": {"pool": "agilerl_app_head_pool", "family": "CPU_X64_SL"},
                "gpu": {
                    "GPU_NV_S": {
                        "pool": "agilerl_app_gpu_GPU_NV_S",
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
            "ALTER COMPUTE POOL agilerl_app_gpu_GPU_NV_S SET MAX_NODES = 5"
            in session.statements
        )
        assert not any(
            "CREATE COMPUTE POOL IF NOT EXISTS agilerl_app_gpu_GPU_NV_S" in s
            for s in session.statements
        )
        [doc] = session.file.uploads
        assert doc["external_access_integrations"] == [
            "agilerl_app_hf_eai",
            "arena_env_eai",
        ]
        assert doc["pools"]["gpu"]["GPU_NV_S"]["max_gpus"] == 5

    def test_drops_gpu_pools_for_families_no_longer_listed(self) -> None:
        session = pools_session([GPU_NV_S_POOL])

        text = set_pools(session, ["GPU_NV_M"], 4, "CPU_X64_L", "CPU_X64_SL")

        assert session.statements[-2:] == [
            "ALTER COMPUTE POOL AGILERL_APP_GPU_GPU_NV_S STOP ALL",
            "DROP COMPUTE POOL AGILERL_APP_GPU_GPU_NV_S",
        ]
        assert text.endswith("AGILERL_APP_GPU_GPU_NV_S  dropped")
        assert list(session.file.uploads[0]["pools"]["gpu"]) == ["GPU_NV_M"]

    def test_refuses_to_shrink_under_a_running_service(self) -> None:
        # Arrange
        running = service_row("RUNNING", 2)
        session = pools_session(
            [GPU_NV_S_POOL],
            services={"agilerl_app_gpu_GPU_NV_S": [running]},
        )

        # Act
        with pytest.raises(
            ValueError,
            match=r"APP\.APP_STATE\.ARENA_RUN_1_WORKERS is using 2 nodes\. "
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
            [GPU_NV_S_POOL], services={"AGILERL_APP_GPU_GPU_NV_S": [running]}
        )

        with pytest.raises(
            ValueError,
            match=r"AGILERL_APP_GPU_GPU_NV_S is no longer listed, so it would be dropped\.\n"
            r"APP\.APP_STATE\.ARENA_RUN_1_WORKERS is using 1 nodes\.",
        ):
            set_pools(session, ["GPU_NV_M"], 4, "CPU_X64_L", "CPU_X64_SL")

        assert not any(s.startswith("DROP") for s in session.statements)

    def test_a_finished_service_does_not_block_a_shrink(self) -> None:
        done = service_row("DONE", 2)
        session = pools_session(
            [GPU_NV_S_POOL], services={"agilerl_app_gpu_GPU_NV_S": [done]}
        )

        set_pools(session, ["GPU_NV_S"], 1, "CPU_X64_L", "CPU_X64_SL")

        assert (
            "ALTER COMPUTE POOL agilerl_app_gpu_GPU_NV_S SET MAX_NODES = 1"
            in session.statements
        )

    def test_refuses_an_unknown_gpu_family_before_any_sql(self) -> None:
        session = pools_session([])

        with pytest.raises(
            ValueError, match="GPU_XL_9 is not a GPU family Arena sizes"
        ):
            set_pools(session, ["GPU_XL_9"], 4, "CPU_X64_L", "CPU_X64_SL")

        assert session.statements == []

    def test_refuses_a_family_missing_from_the_region(self) -> None:
        session = pools_session([])

        with pytest.raises(
            ValueError, match="This account has no GPU_NV_L instance family"
        ):
            set_pools(session, ["GPU_NV_L"], 8, "CPU_X64_L", "CPU_X64_SL")

        assert session.statements == ["SHOW COMPUTE POOL INSTANCE FAMILIES"]

    def test_refuses_a_pool_that_uses_another_family(self) -> None:
        other = Row(
            name="AGILERL_APP_HEAD_POOL", instance_family="CPU_X64_L", max_nodes=4
        )
        session = pools_session([other])

        with pytest.raises(ValueError, match="Pool agilerl_app_head_pool is CPU_X64_L"):
            set_pools(session, ["GPU_NV_S"], 2, "CPU_X64_L", "CPU_X64_SL")

        assert session.file.uploads == []

    def test_refuses_an_empty_family_list(self) -> None:
        session = pools_session([])

        with pytest.raises(ValueError, match="Name at least one GPU family"):
            set_pools(session, [], 4, "CPU_X64_L", "CPU_X64_SL")

        assert session.statements == []


class TestManifestImage:
    def test_procedures_record_the_manifest_image(self) -> None:
        manifest = (APP_DIR / "manifest.yml").read_text(encoding="utf-8")

        assert f"- {IMAGE}\n" in manifest
