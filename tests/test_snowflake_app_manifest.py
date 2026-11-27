# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowflake Native App manifest and the hosts its setup script requests."""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import get_args

import app_billing
import app_keys
import app_trial
import pytest
import yaml

from agilerl.models.rewards import BuiltinReward
from agilerl.snowflake.license import DEFAULT_BASE_URL, arena_api_hostname
from snowflake_app.app_code import arena_app
from tests.test_snowflake_setup import FakeSession, Row

HF_HOSTS = (
    "huggingface.co:443",
    "cas-server.xethub.hf.co:443",
    "cas-server.xethub-eu.hf.co:443",
    "transfer.xethub.hf.co:443",
    "transfer.xethub-eu.hf.co:443",
    "us.aws.cdn.hf.co:443",
    "us.gcp.cdn.hf.co:443",
    "cdn-lfs.hf.co:443",
    "cdn-lfs-us-1.hf.co:443",
    "cdn-lfs-eu-1.hf.co:443",
)

APP_DIR = Path(__file__).resolve().parents[1] / "snowflake_app"
MANIFEST = yaml.safe_load((APP_DIR / "manifest.yml").read_text())
SETUP_SQL = (APP_DIR / "setup_script.sql").read_text()
APP_README = (APP_DIR / "app_readme.md").read_text()


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
    def test_one_gpu_image_from_provider_repository(self) -> None:
        services = MANIFEST["artifacts"]["container_services"]

        [image] = services["images"]

        assert services["uses_gpu"] is True
        assert image.rpartition(":")[0] == "/arena_provider/images/repo/snowflake-job"

    def test_uses_automated_privilege_granting(self) -> None:
        assert MANIFEST["manifest_version"] == 2

    def test_setup_script_exists(self) -> None:
        path = APP_DIR / MANIFEST["artifacts"]["setup_script"]

        assert path.is_file()

    def test_requests_only_documented_privileges(self) -> None:
        names = [name for entry in MANIFEST["privileges"] for name in entry]

        assert sorted(names) == [
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

    def test_app_creates_no_public_endpoint(self) -> None:
        # Arrange
        files = [
            APP_DIR / "manifest.yml",
            APP_DIR / "setup_script.sql",
            *sorted((APP_DIR / "app_code").glob("*.py")),
        ]

        # Act
        text = "\n".join(path.read_text() for path in files)

        # Assert
        assert "BIND SERVICE ENDPOINT" not in text
        assert "ALL_ENDPOINTS_USAGE" not in text
        assert "ingress=True" not in text
        assert not re.search(r"public[\"']?\s*:\s*true", text, re.IGNORECASE)

    def test_references_read_data_write_one_stage_and_use_one_function(self) -> None:
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
            "export_stage": ("STAGE", ["READ", "WRITE"]),
            "reward_function": ("FUNCTION", ["USAGE"]),
        }
        assert references["export_stage"].get("multi_valued", False) is False
        assert references["reward_function"].get("multi_valued", False) is False


class TestSetupScriptExternalAccess:
    def test_integrations_are_named_per_app(self) -> None:
        # Arrange
        readme = (APP_DIR / "README.md").read_text()

        # Act
        created = re.findall(
            r"^\s*CREATE EXTERNAL ACCESS INTEGRATION IDENTIFIER\(:(\w+)\)\n",
            SETUP_SQL,
            flags=re.MULTILINE,
        )
        prefix = re.search(
            r"FUNCTION app_code\.object_prefix\(\).*?\$\$\s*'(\w+)' \|\| "
            r"REGEXP_REPLACE\(UPPER\(CURRENT_DATABASE\(\)\), '\[\^A-Z0-9_\]', '_'\)",
            SETUP_SQL,
            flags=re.DOTALL,
        )

        # Assert
        assert created == ["hf_integration", "arena_integration"]
        assert prefix is not None
        assert prefix.group(1) == "AGILERL_APP_"
        assert "CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS" not in SETUP_SQL
        assert "AGILERL_APP_<APP>_HF_EAI" in readme
        assert "AGILERL_APP_<APP>_ARENA_EAI" in readme

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


class TestReadme:
    def test_omits_account_locator_and_hub_checkout_path(self) -> None:
        readme = (APP_DIR / "README.md").read_text()

        assert "GP07586" not in readme
        assert "oss/agilerl" not in readme
        assert "repository root" in readme


class TestAppReadme:
    def test_package_ships_it_as_the_app_readme(self) -> None:
        # Arrange
        project = yaml.safe_load((APP_DIR / "snowflake.yml").read_text())
        recipe = (APP_DIR.parent / "justfile").read_text()

        # Act
        artifacts = project["entities"]["arena_pkg"]["artifacts"]

        # Assert
        assert MANIFEST["artifacts"]["readme"] == "README.md"
        assert {"src": "app_readme.md", "dest": "README.md"} in artifacts
        assert "snowflake_app/app_readme.md" in recipe

    def test_documents_every_granted_procedure_with_its_role(self) -> None:
        granted = re.findall(
            r"GRANT USAGE ON PROCEDURE app_code\.(\w+)\([^)]*\)\s+"
            r"TO APPLICATION ROLE (\w+);",
            SETUP_SQL,
        )

        rows = dict(
            re.findall(
                r"^\| `(\w+)\([^`]*\)` \| `(\w+)` \|", APP_README, flags=re.MULTILINE
            )
        )

        assert rows == {name.upper(): role for name, role in granted}

    def test_documents_privileges_access_requests_and_references(self) -> None:
        privileges = [name for entry in MANIFEST["privileges"] for name in entry]
        references = [name for entry in MANIFEST["references"] for name in entry]
        specifications = re.findall(
            r"ALTER APPLICATION SET SPECIFICATION (\w+)", SETUP_SQL
        )

        names = [*privileges, *references, *(s.upper() for s in specifications)]

        assert specifications == ["hf_access", "arena_access"]
        assert all(f"`{name}`" in APP_README for name in names)

    def test_documents_every_built_in_reward_and_the_chat_function(self) -> None:
        types = [
            get_args(reward.model_fields["type"].annotation)[0]
            for reward in get_args(BuiltinReward)
        ]

        assert types
        assert all(f"| `{name}` |" in APP_README for name in types)
        assert "snowflake_function" in APP_README
        assert "<service>_CHAT(messages ARRAY" in APP_README

    def test_is_self_contained_for_consumers(self) -> None:
        text = APP_README.lower()

        assert "gp07586" not in text
        assert "oss/agilerl" not in text
        assert "arena snowflake" not in text
        assert "open source" not in text
        assert "hyperparameter" not in text
        assert "arena@agilerl.com" in text


class TestSnowflakeProject:
    def test_build_hint_uses_the_repository_root(self) -> None:
        project = (APP_DIR / "snowflake.yml").read_text()

        assert "oss/agilerl" not in project
        assert "repository root" in project

    def test_every_app_code_module_ships_and_is_imported_by_every_procedure(
        self,
    ) -> None:
        # Arrange
        modules = sorted(
            f"app_code/{path.name}"
            for path in (APP_DIR / "app_code").glob("*.py")
            if path.name != "__init__.py"
        )
        project = yaml.safe_load((APP_DIR / "snowflake.yml").read_text())
        artifacts = project["entities"]["arena_pkg"]["artifacts"]

        # Act
        shipped = sorted(
            artifact["src"]
            for artifact in artifacts
            if isinstance(artifact, dict) and artifact["src"].startswith("app_code/")
        )
        procedures = re.findall(
            r"CREATE OR REPLACE PROCEDURE [^;]*LANGUAGE PYTHON[^;]*;", SETUP_SQL
        )
        imports = {
            re.search(r"IMPORTS = \(([^)]*)\)", procedure).group(1)
            for procedure in procedures
        }

        # Assert
        assert modules == [
            "app_code/app_billing.py",
            "app_code/app_keys.py",
            "app_code/app_trial.py",
            "app_code/arena_app.py",
        ]
        assert shipped == modules
        assert len(procedures) == 16
        assert imports == {
            (
                "'/app_code/arena_app.py', '/app_code/app_billing.py', "
                "'/app_code/app_keys.py', '/app_code/app_trial.py', "
                "'/agilerl_arena.zip'"
            )
        }


class TestSetupScriptObjects:
    def test_no_pool_is_created(self) -> None:
        assert "CREATE COMPUTE POOL" not in SETUP_SQL

    def test_account_level_names_match_the_procedure_names(self) -> None:
        # Arrange
        session = FakeSession(
            [
                (
                    lambda s: s == "SELECT app_code.object_prefix() AS PREFIX",
                    [Row(PREFIX="AGILERL_APP_ARENA_")],
                )
            ]
        )

        # Act
        names = arena_app.app_names(session)
        suffixes = re.findall(r"app_code\.object_prefix\(\) \|\| '(\w+)", SETUP_SQL)

        # Assert
        assert suffixes == ["HF_EAI", "ARENA_EAI", "WH", "ARENA_EAI"]
        assert names == arena_app.AppNames(
            warehouse="AGILERL_APP_ARENA_WH",
            broker_pool="AGILERL_APP_ARENA_BROKER_POOL",
            head_pool="AGILERL_APP_ARENA_HEAD_POOL",
            gpu_pool_prefix="AGILERL_APP_ARENA_GPU_",
            hf_integration="AGILERL_APP_ARENA_HF_EAI",
            api_integration="AGILERL_APP_ARENA_ARENA_EAI",
        )
        assert not any(
            "EXTERNAL_ACCESS_INTEGRATIONS" in procedure or "SECRETS" in procedure
            for procedure in re.findall(
                r"CREATE OR REPLACE PROCEDURE [^;]*;", SETUP_SQL
            )
        )
        assert re.search(r"agilerl_app_\w", SETUP_SQL, flags=re.IGNORECASE) is None

    def test_every_granted_procedure_is_created_with_a_handler(self) -> None:
        created = {
            name: [arg.split()[1] for arg in args.split(",") if arg.strip()]
            for name, args in re.findall(
                r"CREATE OR REPLACE PROCEDURE app_code\.(\w+)\(([^)]*)\)", SETUP_SQL
            )
        }
        handlers = re.findall(r"HANDLER = '(\w+)\.(\w+)'", SETUP_SQL)
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
            "connect_arena": (["STRING"], "app_admin"),
            "rotate_arena_key": ([], "app_admin"),
            "set_pools": (["ARRAY", "NUMBER", "STRING", "STRING"], "app_admin"),
            "gpu_families": ([], "app_admin"),
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
            "export_model": (["STRING", "NUMBER"], "app_admin"),
            "reconcile_now": ([], "app_admin"),
        }
        assert {name: args for name, (args, _role) in granted.items()} == {
            name: args
            for name, args in created.items()
            if name not in {"reconcile_usage", "version_init"}
        }
        assert created["reconcile_usage"] == []
        assert [name for _module, name in handlers] == [
            name for name in created if name != "reconcile_now"
        ]
        modules = {"arena_app": arena_app, "app_keys": app_keys}
        assert all(
            callable(getattr(modules[module], name)) for module, name in handlers
        )
        assert {module for module, _name in handlers} == set(modules)

    def test_only_the_task_and_snowflake_call_the_billing_procedures(self) -> None:
        grants = re.findall(r"GRANT USAGE ON PROCEDURE app_code\.(\w+)", SETUP_SQL)

        assert "reconcile_usage" not in grants
        assert "version_init" not in grants

    def test_admins_run_billing_by_hand_through_the_task(self) -> None:
        # Arrange
        pattern = (
            r"CREATE OR REPLACE PROCEDURE app_code\.reconcile_now\(\)\s+"
            r"RETURNS STRING\s+LANGUAGE SQL\s+AS\s+\$\$(.*?)\$\$;\s+"
            r"GRANT USAGE ON PROCEDURE app_code\.reconcile_now\(\)\s+"
            r"TO APPLICATION ROLE (\w+);"
        )

        # Act
        procedure = re.search(pattern, SETUP_SQL, flags=re.DOTALL)

        # Assert
        assert procedure is not None
        body, role = procedure.groups()
        assert role == "app_admin"
        assert "EXECUTE TASK app_state.reconcile_usage_task;" in body
        assert "reconcile_usage()" not in body
        assert procedure.start() > SETUP_SQL.index(
            "CREATE OR ALTER TASK app_state.reconcile_usage_task"
        )

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
            ("unbilled_reason", "STRING"),
        ]
        assert SETUP_SQL.index("ALTER TABLE app_state.usage_items") > create

    def test_arena_procedures_get_the_app_integration_after_they_are_created(
        self,
    ) -> None:
        # Arrange
        access = re.search(
            r"SELECT ' SET EXTERNAL_ACCESS_INTEGRATIONS = \(' \|\| "
            r"app_code\.object_prefix\(\) \|\| 'ARENA_EAI\)'\s+INTO :arena_access;",
            SETUP_SQL,
        )

        # Act
        altered = re.findall(
            r"EXECUTE IMMEDIATE 'ALTER PROCEDURE app_code\.(\w+)\([^)]*\)' \|\| ([^;]*);",
            SETUP_SQL,
        )

        # Assert
        assert access is not None
        assert altered == [
            ("configure", "arena_access"),
            ("connect_arena", "arena_access || claim_secrets"),
            ("rotate_arena_key", "arena_access || secrets"),
            ("gpu_families", "arena_access || secrets"),
            ("submit", "arena_access || secrets"),
            ("publish", "arena_access || secrets"),
            ("unpublish", "arena_access || secrets"),
            ("reconcile_usage", "arena_access || claim_secrets"),
        ]
        last_create = SETUP_SQL.index(
            "CREATE OR REPLACE PROCEDURE app_code.reconcile_usage("
        )
        assert last_create < access.start()
        assert access.end() < SETUP_SQL.index("CREATE OR ALTER TASK")

    @pytest.mark.parametrize(
        ("name", "variable"),
        [
            ("reconcile_usage", "claim_secrets"),
            ("connect_arena", "claim_secrets"),
            ("rotate_arena_key", "secrets"),
            ("publish", "secrets"),
            ("unpublish", "secrets"),
            ("submit", "secrets"),
            ("gpu_families", "secrets"),
        ],
    )
    def test_arena_procedures_read_only_secrets_their_integration_allows(
        self, name: str, variable: str
    ) -> None:
        # Arrange
        alter = re.search(
            rf"'ALTER PROCEDURE app_code\.{name}\([^)]*\)' \|\| ([^;]*);", SETUP_SQL
        )
        secrets = re.search(rf"\b{variable} STRING DEFAULT '(.*)';", SETUP_SQL)
        allowed = re.search(
            r"ALTER EXTERNAL ACCESS INTEGRATION IDENTIFIER\(:(\w+)\) SET\s+"
            r"ALLOWED_AUTHENTICATION_SECRETS = \(([^)]*)\);",
            SETUP_SQL,
        )
        placeholders = re.findall(
            r"CREATE SECRET IF NOT EXISTS (app_state\.\w+)\s+"
            r"TYPE = GENERIC_STRING\s+SECRET_STRING = '';",
            SETUP_SQL,
        )

        # Act
        aliases = dict(re.findall(r"''(\w+)'' = (app_state\.\w+)", secrets.group(1)))

        # Assert
        assert alter is not None
        assert alter.group(1) == f"arena_access || {variable}"
        partner = {
            "org_key": "app_state.arena_org_key",
            "external_user_id": "app_state.arena_external_user_id",
            "base_url": "app_state.arena_base_url",
        }
        claim = {"claim_secret": "app_state.arena_claim_secret"}
        assert aliases == (partner if variable == "secrets" else partner | claim)
        assert allowed is not None
        assert allowed.group(1) == "arena_integration"
        allowed_names = [name.strip() for name in allowed.group(2).split(",")]
        assert allowed_names == placeholders == [*partner.values(), *claim.values()]
        assert SETUP_SQL.index("CREATE SECRET") < SETUP_SQL.index("ALLOWED_AUTH")

    def test_hourly_serverless_task_calls_reconcile_and_setup_leaves_it_suspended(
        self,
    ) -> None:
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
        assert "reconcile_usage_task RESUME" not in SETUP_SQL
        assert app_billing.BILLING_TASK == "app_state.reconcile_usage_task"
        suspend = "ALTER TASK IF EXISTS app_state.reconcile_usage_task SUSPEND;"
        assert SETUP_SQL.index(suspend) < task.start()
        assert "WAREHOUSE = " not in SETUP_SQL[task.start() : task.end()]
        assert "CREATE OR REPLACE TASK" not in SETUP_SQL

    def test_version_initializer_syncs_the_billing_task(self) -> None:
        # Arrange
        initializer = MANIFEST["lifecycle_callbacks"]["version_initializer"]
        schema, name = initializer.split(".")

        # Act
        procedure = re.search(
            rf"CREATE OR REPLACE PROCEDURE {schema}\.{name}\(\)\s+"
            r"RETURNS STRING\s+LANGUAGE PYTHON\s+[^;]*"
            r"HANDLER = 'arena_app\.version_init';",
            SETUP_SQL,
        )

        # Assert
        assert initializer == "app_code.version_init"
        assert "CREATE OR ALTER VERSIONED SCHEMA app_code;" in SETUP_SQL
        assert procedure is not None
        assert procedure.start() > SETUP_SQL.index(
            "CREATE OR ALTER TASK app_state.reconcile_usage_task"
        )

    def test_reference_callback_is_the_registered_procedure(self) -> None:
        callbacks = {
            spec["register_callback"]
            for entry in MANIFEST["references"]
            for spec in entry.values()
        }

        assert callbacks == {"app_code.register_reference"}
        assert "CREATE OR REPLACE PROCEDURE app_code.register_reference(" in SETUP_SQL


class TestSetupScriptTrial:
    def test_purchase_is_read_through_an_ungranted_secure_function(self) -> None:
        # Arrange
        pattern = (
            r"CREATE OR REPLACE SECURE FUNCTION app_code\.listing_purchased\(\)\s+"
            r"RETURNS BOOLEAN\s+AS 'SYSTEM\$IS_LISTING_PURCHASED\(\)';"
        )

        # Act
        function = re.search(pattern, SETUP_SQL)

        # Assert
        assert function is not None
        assert function.start() > SETUP_SQL.index(
            "CREATE OR ALTER VERSIONED SCHEMA app_code;"
        )
        assert "listing_purchased" not in "".join(
            re.findall(r"GRANT [^;]*;", SETUP_SQL)
        )
        assert [
            line
            for line in SETUP_SQL.splitlines()
            if "SYSTEM$IS_LISTING_PURCHASED" in line and not line.startswith("--")
        ] == ["  AS 'SYSTEM$IS_LISTING_PURCHASED()';"]

    def test_install_time_is_recorded_once_and_kept_on_upgrade(self) -> None:
        create = SETUP_SQL.index(
            "CREATE TABLE IF NOT EXISTS app_state.trial (\n  started_at TIMESTAMP_TZ\n);"
        )
        insert = SETUP_SQL.index(
            "INSERT INTO app_state.trial (started_at)\n"
            "  SELECT CURRENT_TIMESTAMP() WHERE NOT EXISTS (SELECT 1 FROM app_state.trial);"
        )

        assert create < insert
        assert "app_state.trial TO" not in SETUP_SQL

    def test_cutoff_is_one_task_interval_before_the_listing_trial_ends(self) -> None:
        # Arrange
        [minutes] = re.findall(
            r"CREATE OR ALTER TASK app_state\.reconcile_usage_task\n"
            r"(?:  .*\n)*?  SCHEDULE = '(\d+) MINUTES'",
            SETUP_SQL,
        )

        # Act
        margin = timedelta(days=7) - app_trial.TRIAL

        # Assert
        assert margin == timedelta(minutes=int(minutes)) == timedelta(hours=1)

    def test_trial_module_ships_and_every_python_procedure_imports_it(self) -> None:
        # Arrange
        project = yaml.safe_load((APP_DIR / "snowflake.yml").read_text())
        artifacts = project["entities"]["arena_pkg"]["artifacts"]

        # Act
        procedures = re.findall(
            r"CREATE OR REPLACE PROCEDURE [^;]*LANGUAGE PYTHON[^;]*;", SETUP_SQL
        )

        # Assert
        assert {"src": "app_code/app_trial.py", "dest": "app_code/"} in artifacts
        assert len(procedures) == 16
        assert all("'/app_code/app_trial.py'" in p for p in procedures)
