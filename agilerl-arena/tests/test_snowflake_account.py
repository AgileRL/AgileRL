# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""arena snowflake invite, join, check, pools, and grant."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.cli import main
from agilerl.arena.snowflake.account import (
    grant_host,
    grant_table,
    invite,
    join_account,
    join_role_message,
    list_pools,
    missing_user_message,
    run_check,
    set_gpu_cap,
)
from agilerl.arena.snowflake.catalog import NO_SETUP, load_families
from agilerl.arena.snowflake.profile import (
    admin_connection_fields,
    job_connection_fields,
    write_connection,
)
from agilerl.arena.snowflake.setup import no_gpu_message, unsupported_gpu_message
from agilerl.arena.snowflake.setup_cli import grant_command
from tests.test_snowflake_setup import (
    ACCOUNT,
    PROFILE_DOC,
    FakeSession,
    Row,
    admin_routes,
    mode,
    paths,
)

ARENA_TRAINER = [
    Row(name="ARENA_ORG_KEY"),
    Row(name="ARENA_EXTERNAL_USER_ID"),
    Row(name="ARENA_BASE_URL"),
]


def check_routes(
    images: list[Row],
    doc: dict[str, object] | None,
    integrations: list[Row] | None = None,
    secrets: list[Row] | None = None,
    event_table: str = "SNOWFLAKE.TELEMETRY.EVENTS",
) -> list[tuple[object, object]]:
    """What ``arena_trainer`` sees on a set-up account."""
    if integrations is None:
        integrations = [
            Row(name="ARENA_HF_EAI", enabled="true"),
            Row(name="ARENA_API_EAI", enabled="true"),
        ]
    payload = [] if doc is None else [Row(DOC=json.dumps(doc))]
    return [
        (lambda s: s.startswith("USE ROLE"), []),
        (
            lambda s: s.startswith("SHOW COMPUTE"),
            [
                Row(name="arena_pulsar_pool"),
                Row(name="arena_head_pool"),
                Row(name="arena_gpu_GPU_NV_S"),
            ],
        ),
        (lambda s: s.startswith("SHOW WAREHOUSES"), [Row(name="arena_wh")]),
        (lambda s: s.startswith("SHOW IMAGE REPOSITORIES"), [Row(name="images")]),
        (lambda s: s.startswith("SHOW IMAGES"), images),
        (lambda s: s.startswith("SHOW STAGES"), [Row(name="runs")]),
        (lambda s: s.startswith("SHOW TABLES"), [Row(name="ARENA_METRICS")]),
        (lambda s: s.startswith("SHOW FILE FORMATS"), [Row(name="arena_json")]),
        (
            lambda s: s.startswith("SHOW SECRETS"),
            ARENA_TRAINER if secrets is None else secrets,
        ),
        (lambda s: s.startswith("SHOW EXTERNAL"), integrations),
        (
            lambda s: s.startswith("SHOW PARAMETERS LIKE 'EVENT_TABLE'"),
            [Row(key="EVENT_TABLE", value=event_table)],
        ),
        (lambda s: s.startswith("SELECT"), payload),
    ]


def setup_doc_route(doc: dict[str, object]) -> tuple[object, object]:
    return (lambda s: s.startswith("SELECT $1"), [Row(DOC=json.dumps(doc))])


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, str, str]]:
    calls: list[tuple[object, str, str]] = []

    def fake_check(session: object, pool: str, image: str) -> str:
        calls.append((session, pool, image))
        return "580.65.06"

    monkeypatch.setattr("agilerl.arena.snowflake.account.run_driver_check", fake_check)
    return calls


class TestInvite:
    def test_grants_the_role_to_the_stored_name(self) -> None:
        session = FakeSession(
            admin_routes([(lambda s: s.startswith("SHOW USERS"), [Row(name="ALICE")])])
        )

        message = invite(session, "alice", ACCOUNT)

        assert 'GRANT ROLE arena_trainer TO USER "ALICE"' in session.statements
        assert message.startswith("Granted arena_trainer to ALICE.\n")
        assert f"arena snowflake join --account {ACCOUNT}" in message
        assert "No Arena login is needed." in message
        assert message.endswith("Next: arena snowflake submit MANIFEST.yaml")
        assert " She " not in message
        assert " her " not in message

    def test_any_user_name_is_quoted(self) -> None:
        session = FakeSession(
            admin_routes(
                [
                    (
                        lambda s: s.startswith("SHOW USERS"),
                        [Row(name="jane.doe@example.com"), Row(name='o"brien')],
                    )
                ]
            )
        )

        invite(session, "jane.doe@example.com", ACCOUNT)
        invite(session, 'o"brien', ACCOUNT)

        assert "SHOW USERS LIKE 'jane.doe@example.com'" in session.statements
        assert (
            'GRANT ROLE arena_trainer TO USER "jane.doe@example.com"'
            in session.statements
        )
        assert 'GRANT ROLE arena_trainer TO USER "o""brien"' in session.statements

    def test_a_missing_user_is_not_created(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="No user ALICE") as raised:
            invite(session, "ALICE", ACCOUNT)

        assert str(raised.value) == missing_user_message("ALICE")
        assert not any(s.startswith(("GRANT ", "CREATE ")) for s in session.statements)

    def test_a_session_without_accountadmin_grants_nothing(self) -> None:
        session = FakeSession(
            [
                (
                    lambda s: s.startswith("SELECT CURRENT_USER"),
                    [Row(c1="U", c2="PUBLIC", c3="A", c4="R")],
                ),
                (lambda s: s.startswith("USE ROLE"), RuntimeError("no")),
            ]
        )

        with pytest.raises(ValueError, match="cannot use ACCOUNTADMIN"):
            invite(session, "ALICE", ACCOUNT)

        assert not any(s.startswith("GRANT ") for s in session.statements)


class TestJoinAccount:
    def test_reads_setup_and_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = FakeSession(
            [(lambda s: s.startswith("USE ROLE"), []), setup_doc_route(PROFILE_DOC)]
        )

        join_account(session, "Ada")

        assert session.statements[0] == "USE ROLE arena_trainer"
        assert any(s.startswith("SELECT $1") for s in session.statements)
        assert not any(s.startswith("CREATE ") for s in session.statements)
        assert session.file.uploads == []
        assert list(tmp_path.rglob("*.toml")) == []

    def test_without_the_role_stops(self) -> None:
        session = FakeSession(
            [(lambda s: s.startswith("USE ROLE"), RuntimeError("no"))]
        )

        with pytest.raises(ValueError, match="does not have the role") as raised:
            join_account(session, "Ada")

        assert str(raised.value) == join_role_message("Ada")

    def test_without_setup_stops(self) -> None:
        session = FakeSession(
            [
                (lambda s: s.startswith("USE ROLE"), []),
                (
                    lambda s: s.startswith("SELECT"),
                    RuntimeError("Stage does not exist"),
                ),
            ]
        )

        with pytest.raises(ValueError, match="no Arena setup") as raised:
            join_account(session, "Ada")

        assert str(raised.value) == NO_SETUP


class TestRunCheck:
    def test_keeps_the_saved_tag_and_runs_the_driver_check(
        self,
        driver: list[tuple[object, str, str]],
    ) -> None:
        # Arrange
        saved = "/arena/training/images/snowflake-job:1.0"
        images = [
            Row(image_name="other", tags="9.0", created_on="2026-03-01"),
            Row(image_name="snowflake-job", tags="1.0", created_on="2026-01-01"),
            Row(image_name="snowflake-job", tags="2.0", created_on="2026-02-01"),
        ]
        session = FakeSession(check_routes(images, dict(PROFILE_DOC, image=saved)))

        # Act
        report = run_check(session)

        # Assert
        assert report == (
            "snowflake-job tags:\n"
            "  1.0\n"
            "  2.0\n"
            f"image  {saved}\n"
            "driver 580.65.06 on arena_gpu_GPU_NV_S"
        )
        assert driver == [(session, "arena_gpu_GPU_NV_S", saved)]
        assert session.file.uploads[-1]["image"] == saved

    def test_a_missing_saved_tag_uses_the_newest(
        self, driver: list[tuple[object, str, str]]
    ) -> None:
        images = [
            Row(image_name="snowflake-job", tags="1.0", created_on="2026-01-01"),
            Row(image_name="snowflake-job", tags="2.0", created_on="2026-02-01"),
        ]
        session = FakeSession(check_routes(images, dict(PROFILE_DOC, image="")))

        run_check(session)

        newest = "/arena/training/images/snowflake-job:2.0"
        assert session.file.uploads[-1]["image"] == newest
        assert driver[0][2] == newest

    def test_no_event_table_adds_the_sql_to_set_one(
        self, driver: list[tuple[object, str, str]]
    ) -> None:
        images = [Row(image_name="snowflake-job", tags="1.0", created_on="2026-01-01")]
        session = FakeSession(check_routes(images, PROFILE_DOC, event_table=""))

        report = run_check(session)

        assert report.splitlines()[-1] == (
            "Warning: this account has no active event table, so arena snowflake "
            "logs cannot read finished jobs. To keep their logs, run: "
            "ALTER ACCOUNT SET EVENT_TABLE = SNOWFLAKE.TELEMETRY.EVENTS"
        )
        assert not any(s.startswith("ALTER ACCOUNT") for s in session.statements)

    def test_missing_secrets_and_integrations_are_all_named(
        self, driver: list[tuple[object, str, str]]
    ) -> None:
        session = FakeSession(
            check_routes(
                [Row(image_name="snowflake-job", tags="1.0", created_on="2026-01-01")],
                PROFILE_DOC,
                integrations=[Row(name="ARENA_HF_EAI", enabled="false")],
                secrets=[Row(name="ARENA_ORG_KEY")],
            )
        )

        with pytest.raises(ValueError, match="was not found") as raised:
            run_check(session)

        assert str(raised.value) == (
            "Secret arena.training.arena_external_user_id was not found.\n"
            "Secret arena.training.arena_base_url was not found.\n"
            "External access integration arena_hf_eai is missing or not enabled.\n"
            "External access integration arena_api_eai is missing or not enabled.\n"
            "Re-run arena snowflake setup."
        )
        assert driver == []

    def test_a_missing_image_repository_is_named(
        self, driver: list[tuple[object, str, str]]
    ) -> None:
        routes = check_routes([], PROFILE_DOC)
        routes.insert(0, (lambda s: s.startswith("SHOW IMAGE REPOSITORIES"), []))

        with pytest.raises(
            ValueError, match=r"Image repository arena\.training\.images was not"
        ):
            run_check(FakeSession(routes))

    def test_no_image_tells_the_admin_to_rerun_setup(
        self, driver: list[tuple[object, str, str]]
    ) -> None:
        with pytest.raises(ValueError, match="No snowflake-job image") as raised:
            run_check(FakeSession(check_routes([], None)))

        assert str(raised.value) == (
            "No snowflake-job image is in arena.training.images yet.\n"
            "Re-run arena snowflake setup. It copies the image."
        )

    def test_a_role_that_cannot_be_used_names_the_server_error(
        self, driver: list[tuple[object, str, str]]
    ) -> None:
        session = FakeSession(
            [
                (
                    lambda s: s.startswith("USE ROLE"),
                    RuntimeError("Role 'ARENA_TRAINER' does not exist\nmore detail"),
                )
            ]
        )

        with pytest.raises(ValueError, match="cannot be used") as raised:
            run_check(session)

        assert str(raised.value) == (
            "The role arena_trainer cannot be used.\n"
            "Role 'ARENA_TRAINER' does not exist\n"
            "Re-run arena snowflake setup."
        )
        assert driver == []

    def test_an_image_without_setup_json_stops(
        self, driver: list[tuple[object, str, str]]
    ) -> None:
        images = [Row(image_name="snowflake-job", tags="1.0", created_on="2026-01-01")]
        session = FakeSession(check_routes(images, None))

        with pytest.raises(ValueError, match="no Arena setup") as raised:
            run_check(session)

        assert str(raised.value) == NO_SETUP
        assert session.file.uploads == []
        assert driver == []

    def test_a_missing_compute_pool_is_named(
        self, driver: list[tuple[object, str, str]]
    ) -> None:
        images = [Row(image_name="snowflake-job", tags="1.0", created_on="2026-01-01")]
        routes = check_routes(images, PROFILE_DOC)
        routes.insert(
            0,
            (
                lambda s: s.startswith("SHOW COMPUTE"),
                [Row(name="arena_pulsar_pool"), Row(name="arena_head_pool")],
            ),
        )
        session = FakeSession(routes)

        with pytest.raises(ValueError, match="Compute pool") as raised:
            run_check(session)

        assert str(raised.value) == (
            "Compute pool arena_gpu_GPU_NV_S was not found.\n"
            "Re-run arena snowflake setup."
        )
        assert session.file.uploads == []
        assert driver == []


def cap_session(
    pools: list[Row], services: list[Row], describe: list[Row] | None = None
) -> FakeSession:
    extra: list[tuple[object, object]] = [
        setup_doc_route(PROFILE_DOC),
        (lambda s: s == "SHOW COMPUTE POOLS", pools),
        (lambda s: s.startswith("SHOW SERVICES"), services),
    ]
    if describe is not None:
        extra.append((lambda s: s.startswith("DESCRIBE COMPUTE POOL"), describe))
    return FakeSession(admin_routes(extra))


class TestListPools:
    def test_lists_role_cap_and_current_nodes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = cap_session(
            [
                Row(
                    name="arena_pulsar_pool",
                    instance_family="CPU_X64_L",
                    max_nodes=4,
                    active_nodes=1,
                ),
                Row(
                    name="arena_head_pool",
                    instance_family="CPU_X64_SL",
                    max_nodes=4,
                    active_nodes=0,
                ),
                Row(
                    name="arena_gpu_GPU_NV_S",
                    instance_family="GPU_NV_S",
                    max_nodes=2,
                    active_nodes=0,
                ),
            ],
            [],
        )

        text = list_pools(session)

        assert text == (
            "arena_pulsar_pool  CPU_X64_L  broker  1 node\n"
            "arena_head_pool  CPU_X64_SL  head  0 nodes\n"
            "arena_gpu_GPU_NV_S  GPU_NV_S  max 2 GPUs  0 nodes"
        )

    def test_a_missing_node_count_comes_from_describe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = cap_session(
            [Row(name="arena_pulsar_pool", instance_family="CPU_X64_L", max_nodes=4)],
            [],
            [Row(active_nodes=3)],
        )

        text = list_pools(session)

        assert "arena_pulsar_pool  CPU_X64_L  broker  3 nodes" in text
        assert "arena_head_pool  CPU_X64_SL  head  not found" in text

    def test_without_setup_stops(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="no Arena setup") as raised:
            list_pools(session)

        assert str(raised.value) == NO_SETUP
        assert "SHOW COMPUTE POOLS" not in session.statements


class TestSetGpuCap:
    def test_alters_max_nodes_and_the_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = cap_session(
            [
                Row(
                    name="arena_gpu_GPU_NV_S",
                    instance_family="GPU_NV_S",
                    max_nodes=2,
                    active_nodes=0,
                )
            ],
            [],
        )

        text = set_gpu_cap(session, "GPU_NV_S", 4)

        assert (
            "ALTER COMPUTE POOL arena_gpu_GPU_NV_S SET MAX_NODES = 4"
            in session.statements
        )
        assert text == "arena_gpu_GPU_NV_S  GPU_NV_S  max 4 GPUs  MAX_NODES 4"
        assert session.file.uploads[-1]["pools"]["gpu"]["GPU_NV_S"]["max_gpus"] == 4

    def test_creates_a_new_family_past_the_default_node_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = cap_session([], [])

        set_gpu_cap(session, "GPU_NV_M", 44)

        sql = "\n".join(session.statements)
        assert "CREATE COMPUTE POOL IF NOT EXISTS arena_gpu_GPU_NV_M" in sql
        assert "MAX_NODES = 11" in sql
        assert (
            "GRANT MONITOR ON COMPUTE POOL arena_gpu_GPU_NV_M TO ROLE arena_trainer"
            in sql
        )
        assert "'COMPUTE POOL', 'arena_gpu_GPU_NV_M'" in sql
        assert "GRANT MODIFY" not in sql
        saved = session.file.uploads[-1]["pools"]["gpu"]
        assert saved["GPU_NV_M"] == {
            "pool": "arena_gpu_GPU_NV_M",
            "max_gpus": 44,
            "gpus_per_node": 4,
            "gpu_memory_gib": 24,
        }

    def test_refuses_a_shrink_under_a_running_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = cap_session(
            [
                Row(
                    name="arena_gpu_GPU_NV_S",
                    instance_family="GPU_NV_S",
                    max_nodes=4,
                    active_nodes=4,
                )
            ],
            [
                Row(
                    name="arena_run_1_workers",
                    database_name="arena",
                    schema_name="training",
                    status="RUNNING",
                    current_instances=4,
                ),
                Row(name="arena_run_old", status="DONE", current_instances=8),
            ],
        )

        with pytest.raises(
            ValueError, match=r"arena\.training\.arena_run_1_workers"
        ) as raised:
            set_gpu_cap(session, "GPU_NV_S", 1)

        assert "arena_run_old" not in str(raised.value)
        assert not any(s.startswith("ALTER COMPUTE POOL") for s in session.statements)
        assert session.file.uploads == []

    def test_a_service_without_current_or_target_counts_its_minimum(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = cap_session(
            [Row(name="arena_gpu_GPU_NV_S", instance_family="GPU_NV_S", max_nodes=4)],
            [
                Row(
                    name="arena_run_2_workers",
                    database_name="arena",
                    schema_name="training",
                    status="PENDING",
                    current_instances="",
                    min_instances=3,
                )
            ],
        )

        with pytest.raises(ValueError, match="using 3 nodes") as raised:
            set_gpu_cap(session, "GPU_NV_S", 2)

        assert str(raised.value) == (
            "arena.training.arena_run_2_workers is using 3 nodes. "
            "MAX_NODES 2 is below that.\n"
            "The pool was not changed."
        )

    def test_an_unknown_family_runs_no_pool_sql(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="not a GPU family") as raised:
            set_gpu_cap(session, "cpu_x64_l", 4)

        assert str(raised.value) == unsupported_gpu_message("CPU_X64_L")
        assert not any("COMPUTE POOL" in s for s in session.statements)

    def test_a_family_missing_from_the_region_runs_no_pool_sql(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="no GPU_NV_L pool") as raised:
            set_gpu_cap(session, "GPU_NV_L", 8)

        assert str(raised.value) == no_gpu_message("GPU_NV_L", load_families(session))
        assert not any(
            s.startswith(("CREATE COMPUTE", "ALTER COMPUTE"))
            for s in session.statements
        )

    def test_without_setup_stops(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="no Arena setup") as raised:
            set_gpu_cap(session, "GPU_NV_S", 2)

        assert str(raised.value) == NO_SETUP
        assert not any(
            s.startswith(("CREATE COMPUTE", "ALTER COMPUTE"))
            for s in session.statements
        )

    def test_a_pool_on_another_family_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = cap_session(
            [Row(name="arena_gpu_GPU_NV_S", instance_family="GPU_NV_M", max_nodes=2)],
            [],
        )

        with pytest.raises(ValueError, match="Pool arena_gpu_GPU_NV_S is GPU_NV_M"):
            set_gpu_cap(session, "GPU_NV_S", 4)

        assert not any(s.startswith("ALTER COMPUTE POOL") for s in session.statements)
        assert session.file.uploads == []


class TestGrantTable:
    def test_grants_usage_up_the_path_and_select_on_the_table(self) -> None:
        session = FakeSession(admin_routes())

        line = grant_table(session, 'SALES."Raw Data".EVENTS')

        assert session.statements[-3:] == [
            "GRANT USAGE ON DATABASE SALES TO ROLE arena_trainer",
            'GRANT USAGE ON SCHEMA SALES."Raw Data" TO ROLE arena_trainer',
            'GRANT SELECT ON TABLE SALES."Raw Data".EVENTS TO ROLE arena_trainer',
        ]
        assert line == 'Granted SELECT on SALES."Raw Data".EVENTS to arena_trainer.'

    def test_a_two_part_name_runs_nothing(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(
            ValueError, match=r"--table must be database\.schema\.table"
        ):
            grant_table(session, "SALES.EVENTS")

        assert session.statements == []


class TestGrantHost:
    def test_creates_the_rule_and_integration_and_attaches_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        paths(tmp_path, monkeypatch)
        session = FakeSession(admin_routes([setup_doc_route(PROFILE_DOC)]))

        # Act
        line = grant_host(session, "API.Example.com")

        # Assert
        assert (
            "CREATE NETWORK RULE arena.training.arena_env_egress\n"
            "  TYPE = HOST_PORT\n"
            "  MODE = EGRESS\n"
            "  VALUE_LIST = ('api.example.com:443')"
        ) in session.statements
        assert (
            "CREATE EXTERNAL ACCESS INTEGRATION arena_env_eai\n"
            "  ALLOWED_NETWORK_RULES = (arena.training.arena_env_egress)\n"
            "  ENABLED = TRUE"
        ) in session.statements
        assert "GRANT USAGE ON INTEGRATION arena_env_eai TO ROLE arena_trainer" in (
            session.statements
        )
        assert line == "Jobs can reach api.example.com:443 through arena_env_eai."
        assert session.file.uploads[-1]["external_access_integrations"] == [
            "arena_hf_eai",
            "arena_env_eai",
        ]

    def test_a_second_host_is_added_to_the_existing_rule(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        doc = dict(
            PROFILE_DOC, external_access_integrations=["arena_hf_eai", "arena_env_eai"]
        )
        session = FakeSession(
            admin_routes(
                [
                    setup_doc_route(doc),
                    (
                        lambda s: s.startswith("SHOW NETWORK RULES"),
                        [Row(name="ARENA_ENV_EGRESS")],
                    ),
                    (
                        lambda s: s.startswith("SHOW EXTERNAL ACCESS"),
                        [Row(name="ARENA_ENV_EAI", enabled="true")],
                    ),
                ]
            )
        )

        grant_host(session, "db.example.com")

        assert (
            "ALTER NETWORK RULE arena.training.arena_env_egress "
            "ADD VALUE_LIST = ('db.example.com:443')"
        ) in session.statements
        assert any(
            s.startswith("ALTER EXTERNAL ACCESS INTEGRATION arena_env_eai")
            for s in session.statements
        )
        assert session.file.uploads[-1]["external_access_integrations"] == [
            "arena_hf_eai",
            "arena_env_eai",
        ]

    @pytest.mark.parametrize(
        "host", ["https://x.com", "x.com:8443", "*.x.com", "localhost"]
    )
    def test_anything_but_a_host_name_runs_nothing(self, host: str) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="--host must be a host name"):
            grant_host(session, host)

        assert session.statements == []

    def test_without_setup_creates_nothing(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="no Arena setup") as raised:
            grant_host(session, "api.example.com")

        assert str(raised.value) == NO_SETUP
        assert not any(s.startswith(("CREATE ", "ALTER ")) for s in session.statements)


class TestAccountCommands:
    def test_invite_prints_the_next_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))
        session = FakeSession(
            admin_routes([(lambda s: s.startswith("SHOW USERS"), [Row(name="ALICE")])])
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session", lambda name: session
        )

        result = CliRunner().invoke(main, ["snowflake", "invite", "ALICE"])

        assert result.exit_code == 0, result.output
        assert f"arena snowflake join --account {ACCOUNT}" in result.output

    def test_an_admin_command_uses_arena_wh_when_it_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))
        session = FakeSession(
            admin_routes(
                [
                    (lambda s: s.startswith("SHOW USERS"), [Row(name="ALICE")]),
                    (lambda s: s.startswith("SHOW WAREHOUSES"), [Row(name="ARENA_WH")]),
                ]
            )
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session", lambda name: session
        )

        # Act
        result = CliRunner().invoke(main, ["snowflake", "invite", "ALICE"])

        # Assert
        assert result.exit_code == 0, result.output
        assert session.statements[:2] == [
            "SHOW WAREHOUSES LIKE 'arena_wh'",
            "USE WAREHOUSE arena_wh",
        ]
        assert session.closed == 1

    def test_join_writes_the_arena_connection_and_prints_the_next_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, _key = paths(tmp_path, monkeypatch)
        session = FakeSession(
            [(lambda s: s.startswith("USE ROLE"), []), setup_doc_route(PROFILE_DOC)]
        )
        opened: list[str] = []
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_session",
            lambda name: opened.append(name) or session,
        )

        result = CliRunner().invoke(
            main,
            [
                "snowflake",
                "join",
                "--account",
                ACCOUNT,
                "--user",
                "Ada",
                "--token",
                "sekret",
            ],
        )

        assert result.exit_code == 0, result.output
        assert result.output == (
            f"Joined {ACCOUNT} as Ada.\nNext: arena snowflake submit MANIFEST.yaml\n"
        )
        assert opened == ["arena"]
        text = connections.read_text(encoding="utf-8")
        assert "[arena]" in text
        assert "sekret" in text
        assert "PROGRAMMATIC_ACCESS_TOKEN" in text
        if sys.platform != "win32":
            assert mode(connections) == 0o600
        assert list(tmp_path.rglob("*.toml")) == [connections]

    def test_join_asks_before_replacing_a_different_arena_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, key = paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, key))
        before = connections.read_text(encoding="utf-8")
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_session",
            lambda name: pytest.fail("opened a session"),
        )

        result = CliRunner().invoke(
            main,
            [
                "snowflake",
                "join",
                "--account",
                ACCOUNT,
                "--user",
                "Ada",
                "--token",
                "t",
            ],
            input="\n",
        )

        assert result.exit_code == 1
        assert "signs in as arena_job_user" in result.output
        assert "[arena] was left as it is." in result.output
        assert connections.read_text(encoding="utf-8") == before

    def test_join_rejects_a_password(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, _key = paths(tmp_path, monkeypatch)

        result = CliRunner().invoke(
            main,
            ["snowflake", "join", "--account", ACCOUNT, "--user", "Ada"],
            input="password\n",
        )

        assert result.exit_code != 0
        assert "uses password auth" in result.output
        assert not connections.exists()

    def test_join_rejects_both_a_token_and_a_key(self) -> None:
        result = CliRunner().invoke(
            main, ["snowflake", "join", "--token", "sekret", "--key", "key.p8"]
        )

        assert result.exit_code != 0
        assert "one of" in result.output

    def test_check_names_join_when_the_key_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, tmp_path / "gone.p8"))

        result = CliRunner().invoke(main, ["snowflake", "check"])

        assert result.exit_code == 1
        assert "The private key is not in Snowflake." in result.output
        assert "Run arena snowflake join" in result.output

    def test_pools_refuse_the_job_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, key))
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: pytest.fail("opened a session"),
        )

        result = CliRunner().invoke(
            main, ["snowflake", "pools", "--connection", "arena"]
        )

        assert result.exit_code != 0
        assert "user arena_job_user" in result.output
        assert "arena-admin" in result.output

    def test_pools_set_alters_the_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))
        session = cap_session(
            [
                Row(
                    name="arena_gpu_GPU_NV_S",
                    instance_family="GPU_NV_S",
                    max_nodes=2,
                    active_nodes=0,
                )
            ],
            [],
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session", lambda name: session
        )

        result = CliRunner().invoke(
            main, ["snowflake", "pools", "set", "GPU_NV_S", "--max-gpus", "4"]
        )

        assert result.exit_code == 0, result.output
        assert (
            "ALTER COMPUTE POOL arena_gpu_GPU_NV_S SET MAX_NODES = 4"
            in session.statements
        )

    def test_grant_runs_tables_then_hosts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))
        session = FakeSession(admin_routes([setup_doc_route(PROFILE_DOC)]))
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session", lambda name: session
        )

        result = CliRunner().invoke(
            grant_command, ["--table", "DB.SCH.T", "--host", "api.example.com"]
        )

        assert result.exit_code == 0, result.output
        assert result.output == (
            "Granted SELECT on DB.SCH.T to arena_trainer.\n"
            "Jobs can reach api.example.com:443 through arena_env_eai.\n"
        )

    def test_grant_needs_a_table_or_a_host(self) -> None:
        result = CliRunner().invoke(grant_command, [])

        assert result.exit_code == click.UsageError.exit_code
        assert "Pass --table or --host." in result.output
