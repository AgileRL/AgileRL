# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowflake Native App manifest and the hosts its setup script requests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from snowflake_app.app_code import arena_app

from agilerl.arena.snowflake.images import TRAINER_REPOSITORY
from agilerl.arena.snowflake.license import DEFAULT_BASE_URL, arena_api_hostname
from agilerl.arena.snowflake.profile import (
    ARENA_API_INTEGRATION,
    ENV_INTEGRATION,
    HF_INTEGRATION,
    WAREHOUSE,
)
from agilerl.arena.snowflake.setup_sql import (
    HF_HOSTS,
    PROBE_INTEGRATION,
    storage_statements,
    warehouse_statements,
)

APP_DIR = Path(__file__).resolve().parents[1] / "snowflake_app"
MANIFEST = yaml.safe_load((APP_DIR / "manifest.yml").read_text())
SETUP_SQL = (APP_DIR / "setup_script.sql").read_text()


def quoted_lists(keyword: str) -> list[set[str]]:
    """Each ``<keyword> = ('a', 'b')`` list in the setup script, as a set.

    :param keyword: ``VALUE_LIST`` or ``HOST_PORTS``.
    :return: One set per occurrence.
    """
    return [
        set(re.findall(r"'([^']+)'", body))
        for body in re.findall(rf"{keyword} = \(([^)]*)\)", SETUP_SQL)
    ]


class TestManifest:
    def test_uses_automated_privilege_granting(self) -> None:
        assert MANIFEST["manifest_version"] == 2

    def test_setup_script_exists(self) -> None:
        path = APP_DIR / MANIFEST["artifacts"]["setup_script"]

        assert path.is_file()

    def test_one_gpu_image_from_provider_repository(self) -> None:
        services = MANIFEST["artifacts"]["container_services"]

        [image] = services["images"]

        assert services["uses_gpu"] is True
        assert (
            image.rpartition(":")[0]
            == f"/arena_provider/images/repo/{TRAINER_REPOSITORY}"
        )

    def test_requests_only_documented_privileges(self) -> None:
        names = [name for entry in MANIFEST["privileges"] for name in entry]

        assert sorted(names) == [
            "BIND SERVICE ENDPOINT",
            "CREATE COMPUTE POOL",
            "CREATE EXTERNAL ACCESS INTEGRATION",
            "CREATE WAREHOUSE",
            "EXECUTE MANAGED TASK",
            "EXECUTE TASK",
        ]
        assert all(
            entry[name]["description"]
            for entry in MANIFEST["privileges"]
            for name in entry
        )

    def test_references_are_select_on_table_or_view(self) -> None:
        references = {
            name: spec
            for entry in MANIFEST["references"]
            for name, spec in entry.items()
        }

        shapes = {
            name: (spec["object_type"], spec["privileges"])
            for name, spec in references.items()
        }

        assert shapes == {
            "source_table": ("TABLE", ["SELECT"]),
            "source_view": ("VIEW", ["SELECT"]),
        }


class TestSetupScriptExternalAccess:
    def test_rules_and_specifications_name_the_same_hosts(self) -> None:
        # Arrange
        arena = {f"{arena_api_hostname(DEFAULT_BASE_URL)}:443"}
        expected = [set(HF_HOSTS), arena]

        # Act
        rules = quoted_lists("VALUE_LIST")
        specifications = quoted_lists("HOST_PORTS")

        # Assert
        assert rules == expected
        assert specifications == expected

    def test_creates_admin_and_user_application_roles(self) -> None:
        roles = re.findall(r"CREATE APPLICATION ROLE IF NOT EXISTS (\w+);", SETUP_SQL)

        assert roles == ["app_admin", "app_user"]

    def test_integrations_use_names_the_cli_does_not(self) -> None:
        # Arrange
        cli_names = {
            HF_INTEGRATION,
            ARENA_API_INTEGRATION,
            ENV_INTEGRATION,
            PROBE_INTEGRATION,
        }
        readme = (APP_DIR / "README.md").read_text()

        # Act
        created = re.findall(
            r"^\s*CREATE EXTERNAL ACCESS INTEGRATION (\w+)\n",
            SETUP_SQL,
            flags=re.MULTILINE,
        )

        # Assert
        assert created == ["agilerl_app_hf_eai", "agilerl_app_arena_eai"]
        assert set(created).isdisjoint(cli_names)
        assert "CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS" not in SETUP_SQL
        assert all(name in readme for name in created)


class TestReadme:
    def test_omits_account_locator_and_hub_checkout_path(self) -> None:
        readme = (APP_DIR / "README.md").read_text()

        assert "GP07586" not in readme
        assert "oss/agilerl" not in readme
        assert "repository root" in readme


class TestSnowflakeProject:
    def test_build_hint_uses_the_repository_root(self) -> None:
        project = (APP_DIR / "snowflake.yml").read_text()

        assert "oss/agilerl" not in project
        assert "repository root" in project


class TestSetupScriptObjects:
    def test_storage_matches_the_cli_setup(self) -> None:
        cli = [
            f"{statement.replace('arena.training.', 'app_state.')};"
            for statement in storage_statements()
            if "IMAGE REPOSITORY" not in statement
        ]

        assert all(statement in SETUP_SQL for statement in cli)

    def test_warehouse_matches_the_cli_setup_and_no_pool_is_created(self) -> None:
        cli = warehouse_statements(0)[0].replace(WAREHOUSE, arena_app.WAREHOUSE)

        assert f"{cli};" in SETUP_SQL
        assert "CREATE COMPUTE POOL" not in SETUP_SQL

    def test_account_level_names_match_the_procedure_constants(self) -> None:
        warehouses = re.findall(r"CREATE WAREHOUSE IF NOT EXISTS (\w+)", SETUP_SQL)
        integrations = re.findall(
            r"CREATE EXTERNAL ACCESS INTEGRATION (\w+)", SETUP_SQL
        )
        procedure_integrations = re.findall(
            r"EXTERNAL_ACCESS_INTEGRATIONS = \((\w+)\)", SETUP_SQL
        )

        assert warehouses == [arena_app.WAREHOUSE]
        assert integrations == [arena_app.HF_INTEGRATION, arena_app.API_INTEGRATION]
        assert procedure_integrations == [arena_app.API_INTEGRATION] * 4
        assert all(
            name.startswith("agilerl_app_")
            for name in (
                arena_app.BROKER_POOL,
                arena_app.HEAD_POOL,
                arena_app.GPU_POOL_PREFIX,
            )
        )

    def test_every_granted_procedure_is_created_with_a_handler(self) -> None:
        created = {
            name: [arg.split()[1] for arg in args.split(",") if arg.strip()]
            for name, args in re.findall(
                r"CREATE OR REPLACE PROCEDURE app_code\.(\w+)\(([^)]*)\)", SETUP_SQL
            )
        }
        handlers = re.findall(r"HANDLER = 'arena_app\.(\w+)'", SETUP_SQL)
        granted = {
            name: ([arg for arg in args.split(", ") if arg], role)
            for name, args, role in re.findall(
                r"GRANT USAGE ON PROCEDURE app_code\.(\w+)\(([^)]*)\)\s+"
                r"TO APPLICATION ROLE (\w+);",
                SETUP_SQL,
            )
        }

        assert granted == {
            "register_reference": (["STRING", "STRING", "STRING"], "app_admin"),
            "configure": (["STRING", "STRING", "STRING"], "app_admin"),
            "set_pools": (["ARRAY", "NUMBER", "STRING", "STRING"], "app_admin"),
            "submit": (
                [
                    "STRING",
                    "NUMBER",
                    "STRING",
                    "STRING",
                    "STRING",
                    "STRING",
                    "NUMBER",
                    "STRING",
                ],
                "app_admin",
            ),
            "status": (["STRING"], "app_user"),
            "logs": (["STRING", "STRING", "NUMBER"], "app_user"),
            "cancel": (["STRING"], "app_admin"),
            "publish": (
                [
                    "STRING",
                    "STRING",
                    "STRING",
                    "NUMBER",
                    "BOOLEAN",
                    "NUMBER",
                    "STRING",
                    "NUMBER",
                ],
                "app_admin",
            ),
            "endpoint": (["STRING", "STRING"], "app_user"),
            "unpublish": (["STRING", "STRING", "BOOLEAN"], "app_admin"),
        }
        assert {name: args for name, (args, _role) in granted.items()} == {
            name: args
            for name, args in created.items()
            if name not in {"reconcile_usage", "version_init"}
        }
        assert created["reconcile_usage"] == []
        assert handlers == [name for name in created if name != "version_init"]
        assert all(callable(getattr(arena_app, name)) for name in handlers)

    def test_only_the_task_and_snowflake_call_the_billing_procedures(self) -> None:
        grants = re.findall(r"GRANT USAGE ON PROCEDURE app_code\.(\w+)", SETUP_SQL)

        assert "reconcile_usage" not in grants
        assert "version_init" not in grants

    def test_billing_tables_survive_upgrades(self) -> None:
        # Arrange
        pattern = r"CREATE TABLE IF NOT EXISTS app_state\.(\w+) \(([^;]*)\);"

        # Act
        tables = {
            name: [line.strip().rstrip(",") for line in body.strip().splitlines()]
            for name, body in re.findall(pattern, SETUP_SQL)
        }

        # Assert
        assert tables["usage_items"] == [
            "kind STRING",
            "usage_id STRING",
            "opened_at TIMESTAMP_TZ",
            "settled_at TIMESTAMP_TZ",
        ]
        assert tables["billed_hours"] == [
            "kind STRING",
            "usage_id STRING",
            "hour_start_ms NUMBER",
            "billable_cents NUMBER",
            "claim_id STRING",
            "attempted_at TIMESTAMP_TZ",
            "emitted_at TIMESTAMP_TZ",
            "PRIMARY KEY (kind, usage_id, hour_start_ms)",
        ]
        assert tables["served_models"] == [
            "service_name STRING",
            "model STRING",
            "version STRING",
            "run_id STRING",
            "gpu_family STRING",
            "gpu_count NUMBER",
        ]
        assert "CREATE OR REPLACE TABLE" not in SETUP_SQL

    def test_consumers_get_read_only_grants_on_app_state(self) -> None:
        # Arrange
        pattern = r"GRANT (\w+) ON (\w+) (app_state[.\w]*) TO APPLICATION ROLE (\w+);"

        # Act
        grants = re.findall(pattern, SETUP_SQL)

        # Assert
        assert grants == [
            ("USAGE", "SCHEMA", "app_state", "app_user"),
            ("SELECT", "TABLE", "app_state.arena_metrics", "app_user"),
            ("SELECT", "TABLE", "app_state.usage_items", "app_admin"),
            ("SELECT", "TABLE", "app_state.billed_hours", "app_admin"),
            ("SELECT", "TABLE", "app_state.served_models", "app_admin"),
        ]
        assert "GRANT APPLICATION ROLE app_user TO APPLICATION ROLE app_admin;" in (
            SETUP_SQL
        )
        assert SETUP_SQL.index("GRANT SELECT ON TABLE app_state.billed_hours") > (
            SETUP_SQL.index("CREATE TABLE IF NOT EXISTS app_state.billed_hours")
        )

    def test_both_roles_can_use_the_schema_of_the_reference_callback(self) -> None:
        schema_grants = re.findall(
            r"GRANT USAGE ON SCHEMA app_code TO APPLICATION ROLE (\w+);", SETUP_SQL
        )
        callback_grant = SETUP_SQL.index(
            "GRANT USAGE ON PROCEDURE app_code.register_reference("
        )

        assert sorted(schema_grants) == ["app_admin", "app_user"]
        assert SETUP_SQL.index("GRANT USAGE ON SCHEMA app_code") < callback_grant

    def test_usage_check_columns_are_added_on_upgrade(self) -> None:
        create = SETUP_SQL.index("CREATE TABLE IF NOT EXISTS app_state.usage_items")

        added = re.findall(
            r"ALTER TABLE app_state\.usage_items ADD COLUMN IF NOT EXISTS (\w+) (\w+);",
            SETUP_SQL,
        )

        assert added == [
            ("first_seen_at", "TIMESTAMP_TZ"),
            ("last_checked_at", "TIMESTAMP_TZ"),
            ("first_404_at", "TIMESTAMP_TZ"),
            ("service_name", "STRING"),
            ("closed_at", "TIMESTAMP_TZ"),
            ("last_error", "STRING"),
            ("last_error_at", "TIMESTAMP_TZ"),
        ]
        assert SETUP_SQL.index("ALTER TABLE app_state.usage_items") > create

    @pytest.mark.parametrize("name", ["reconcile_usage", "publish", "unpublish"])
    def test_arena_procedures_read_only_secrets_their_integration_allows(
        self, name: str
    ) -> None:
        # Arrange
        procedure = SETUP_SQL[
            SETUP_SQL.index(f"CREATE OR REPLACE PROCEDURE app_code.{name}(") :
        ].split(";", 1)[0]
        allowed = re.search(
            r"ALTER EXTERNAL ACCESS INTEGRATION (\w+) SET\s+"
            r"ALLOWED_AUTHENTICATION_SECRETS = \(([^)]*)\);",
            SETUP_SQL,
        )
        placeholders = re.findall(
            r"CREATE SECRET IF NOT EXISTS (app_state\.\w+)\s+"
            r"TYPE = GENERIC_STRING\s+SECRET_STRING = '';",
            SETUP_SQL,
        )

        # Act
        aliases = dict(re.findall(r"'(\w+)' = (app_state\.\w+)", procedure))

        # Assert
        assert "EXTERNAL_ACCESS_INTEGRATIONS = (agilerl_app_arena_eai)" in procedure
        assert aliases == {
            "org_key": "app_state.arena_org_key",
            "external_user_id": "app_state.arena_external_user_id",
            "base_url": "app_state.arena_base_url",
        }
        assert allowed is not None
        assert allowed.group(1) == arena_app.API_INTEGRATION
        assert {name.strip() for name in allowed.group(2).split(",")} == set(
            aliases.values()
        )
        assert placeholders == list(aliases.values())
        assert SETUP_SQL.index("CREATE SECRET") < SETUP_SQL.index("ALLOWED_AUTH")

    def test_hourly_serverless_task_calls_reconcile_and_is_resumed(self) -> None:
        # Arrange
        pattern = (
            r"CREATE OR ALTER TASK app_state\.reconcile_usage_task\s+"
            r"USER_TASK_MANAGED_INITIAL_WAREHOUSE_SIZE = 'XSMALL'\s+"
            r"SCHEDULE = '60 MINUTES'\s+"
            r"SUSPEND_TASK_AFTER_NUM_FAILURES = 0\s+"
            r"AS CALL app_code\.reconcile_usage\(\);"
        )

        # Act
        task = re.search(pattern, SETUP_SQL)

        # Assert
        assert task is not None
        resume = "ALTER TASK app_state.reconcile_usage_task RESUME;"
        assert SETUP_SQL.index(resume) > task.end()
        assert SETUP_SQL.index(resume) > SETUP_SQL.index(
            "PROCEDURE app_code.reconcile_usage()"
        )
        suspend = "ALTER TASK IF EXISTS app_state.reconcile_usage_task SUSPEND;"
        assert SETUP_SQL.index(suspend) < task.start()
        assert "WAREHOUSE = " not in SETUP_SQL[task.start() : task.end()]
        assert "CREATE OR REPLACE TASK" not in SETUP_SQL

    def test_version_initializer_resumes_the_billing_task(self) -> None:
        # Arrange
        initializer = MANIFEST["lifecycle_callbacks"]["version_initializer"]
        schema, name = initializer.split(".")

        # Act
        procedure = re.search(
            rf"CREATE OR REPLACE PROCEDURE {schema}\.{name}\(\)\s+"
            r"RETURNS STRING\s+LANGUAGE SQL\s+AS\s+\$\$(.*?)\$\$;",
            SETUP_SQL,
            flags=re.DOTALL,
        )

        # Assert
        assert initializer == "app_code.version_init"
        assert "CREATE OR ALTER VERSIONED SCHEMA app_code;" in SETUP_SQL
        assert procedure is not None
        assert (
            "ALTER TASK IF EXISTS app_state.reconcile_usage_task RESUME;"
            in procedure.group(1)
        )

    def test_reference_callback_is_the_registered_procedure(self) -> None:
        callbacks = {
            spec["register_callback"]
            for entry in MANIFEST["references"]
            for spec in entry.values()
        }

        assert callbacks == {"app_code.register_reference"}
        assert "CREATE OR REPLACE PROCEDURE app_code.register_reference(" in SETUP_SQL
