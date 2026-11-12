# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""arena snowflake setup: preflight, plan, confirmation, statements, and the CLI."""

from __future__ import annotations

import json
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization

from agilerl.arena.auth import EXTERNAL_USER_ID_HEADER
from agilerl.arena.cli import main
from agilerl.arena.snowflake.catalog import FOREIGN_DATABASE, GpuFamily
from agilerl.arena.snowflake.images import (
    MISSING_IMAGE_TAG,
    READ_ONLY_REPOSITORY,
    TrainerImage,
    image_not_found_message,
    image_preflight,
    run_driver_check,
)
from agilerl.arena.snowflake.license import (
    AUTH_REJECTED,
    SNOWFLAKE_DENIED,
    SnowflakeLicenseError,
)
from agilerl.arena.snowflake.profile import (
    admin_connection_fields,
    generate_private_key,
    job_connection_fields,
    missing_key_message,
    profile_from_mapping,
    public_key_body,
    write_connection,
)
from agilerl.arena.snowflake.session import connection_section
from agilerl.arena.snowflake.setup import (
    TRIAL_MESSAGE,
    SetupRequest,
    format_summary,
    key_mismatch_message,
    missing_setup_key_message,
    no_gpu_message,
    nodes_for_gpus,
    pool_family_message,
    print_setup_sql,
    run_setup,
    smallest_gpu_family,
)
from agilerl.arena.snowflake.setup_sql import (
    PROBE_INTEGRATION,
    ArenaCredentials,
    PoolPlan,
)

ACCOUNT = "ORGNAME-ACCOUNT"
TAG = "32.6.0-py313-cu130"
TRAINER = TrainerImage(TAG, "sha256:" + "a" * 64)
IMAGE = f"/arena/training/images/snowflake-job:{TAG}"
ENABLED = {"schemaVersion": 1, "features": {"snowflakeTraining": True}}
NO_GPU = (
    "This account has no GPU_NV_S pool in this region (1x A10G). "
    "Arena did not create a compute pool.\n"
    "GPU families SHOW COMPUTE POOL INSTANCE FAMILIES returned:\n"
    "  (none)\n"
    "Rerun with one of those names:\n"
    "  arena snowflake setup --gpu-family <name>\n"
    "A CPU pool cannot run this trainer.\n"
    "Nothing was created."
)
PROFILE_DOC = {
    "image": IMAGE,
    "warehouse": "arena_wh",
    "stage": "arena.training.runs",
    "metrics_table": "arena.training.arena_metrics",
    "external_access_integrations": ["arena_hf_eai"],
    "pools": {
        "broker": {"pool": "arena_pulsar_pool", "family": "CPU_X64_L"},
        "head": {"pool": "arena_head_pool", "family": "CPU_X64_SL"},
        "gpu": {
            "GPU_NV_S": {
                "pool": "arena_gpu_GPU_NV_S",
                "max_gpus": 2,
                "gpus_per_node": 1,
                "gpu_memory_gib": 24,
            }
        },
    },
}


class Row:
    """Snowpark ``Row`` stand-in."""

    def __init__(self, **values: object) -> None:
        self._values = values

    def as_dict(self) -> dict[str, object]:
        return dict(self._values)


class Frame:
    def __init__(self, rows: list[Row]) -> None:
        self._rows = rows

    def collect(self) -> list[Row]:
        return list(self._rows)


class Files:
    def __init__(self) -> None:
        self.uploads: list[dict[str, object]] = []
        self.stages: list[str] = []

    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.uploads.append(json.loads(Path(local).read_text(encoding="utf-8")))
        self.stages.append(stage)
        assert auto_compress is False
        assert overwrite is True


class Trial(Exception):
    errno = 509009

    def __str__(self) -> str:
        return "External access is not supported for trial accounts"


class FakeSession:
    def __init__(self, routes: list[tuple[object, object]] | None = None) -> None:
        self.routes = list(routes or [])
        self.statements: list[str] = []
        self.file = Files()
        self.closed = 0

    def sql(self, statement: str) -> Frame:
        self.statements.append(statement)
        for pred, result in self.routes:
            if pred(statement):
                if isinstance(result, BaseException):
                    raise result
                return Frame(result)
        return Frame([])

    def close(self) -> None:
        self.closed += 1


class Scripted:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = list(answers or [])
        self.lines: list[str] = []
        self.prompts: list[str] = []
        self.hidden: list[bool] = []

    def echo(self, text: str) -> None:
        self.lines.append(text)

    def ask(self, text: str, default: str | None, hide: bool) -> str:
        self.prompts.append(text)
        self.hidden.append(hide)
        if self.answers:
            return self.answers.pop(0)
        if hide:
            return "arena_org_pasted"
        return default or ""


class JobSession:
    """``[arena]`` session setup opens for the driver check."""

    def close(self) -> None:
        return None


def driver_job_session(log: str) -> FakeSession:
    """``[arena]`` session whose driver-check job is running and printed ``log``."""
    return FakeSession(
        [
            (
                lambda statement: statement.startswith("SHOW COMPUTE POOLS"),
                [Row(name="ARENA_GPU_GPU_NV_S", state="ACTIVE")],
            ),
            (
                lambda statement: statement.startswith("SHOW SERVICE CONTAINERS"),
                [Row(service_status="RUNNING", status="READY", message="")],
            ),
            (
                lambda statement: "SYSTEM$GET_SERVICE_LOGS" in statement,
                [Row(log=log)],
            ),
        ]
    )


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point connections.toml at ``tmp_path``. The default key follows."""
    connections = tmp_path / "connections.toml"
    monkeypatch.setattr(
        "agilerl.arena.snowflake.session.connections_file", lambda: connections
    )
    return connections, tmp_path / "arena_job_user.p8"


def created(session: FakeSession) -> list[str]:
    """CREATE statements other than the trial probe."""
    return [
        statement
        for statement in session.statements
        if statement.startswith("CREATE ") and PROBE_INTEGRATION not in statement
    ]


def admin_routes(
    extra: list[tuple[object, object]] | None = None,
) -> list[tuple[object, object]]:
    """ACCOUNTADMIN identity and a region with CPU and two GPU families. ``extra`` wins."""
    routes: list[tuple[object, object]] = list(extra or [])
    routes.extend(
        [
            (
                lambda statement: statement.startswith("SELECT CURRENT_USER"),
                [
                    Row(
                        **{
                            "CURRENT_USER()": "NICK",
                            "CURRENT_ROLE()": "ACCOUNTADMIN",
                            "CURRENT_ACCOUNT()": "ORG",
                            "CURRENT_REGION()": "AWS",
                        }
                    )
                ],
            ),
            (
                lambda statement: statement == "SELECT CURRENT_ROLE()",
                [Row(**{"CURRENT_ROLE()": "ACCOUNTADMIN"})],
            ),
            (
                lambda statement: statement == "SHOW COMPUTE POOL INSTANCE FAMILIES",
                [
                    Row(name="CPU_X64_L", gpu_count=0, gpu=""),
                    Row(name="CPU_X64_SL", gpu_count=0, gpu=""),
                    Row(name="GPU_NV_S", gpu_count=1, gpu="A10G"),
                    Row(name="GPU_NV_M", gpu_count=4, gpu="A10G"),
                    Row(name="GPU_GCP_NV_L4_1_24G", gpu_count=1, gpu="L4"),
                ],
            ),
        ]
    )
    return routes


def rerun_routes(doc: dict[str, object]) -> list[tuple[object, object]]:
    """An account setup already ran on: secrets, integrations, quotas, and setup.json."""
    return admin_routes(
        [
            (
                lambda statement: statement.startswith("SELECT $1"),
                [Row(DOC=json.dumps(doc))],
            ),
            (
                lambda statement: statement.startswith("SHOW DATABASES"),
                [Row(name="arena")],
            ),
            (
                lambda statement: statement.startswith("SHOW EXTERNAL ACCESS"),
                [
                    Row(name="ARENA_HF_EAI", enabled="true"),
                    Row(name="ARENA_API_EAI", enabled="true"),
                ],
            ),
            (
                lambda statement: statement.startswith("SHOW SECRETS"),
                [
                    Row(name="ARENA_ORG_KEY"),
                    Row(name="ARENA_EXTERNAL_USER_ID"),
                    Row(name="ARENA_BASE_URL"),
                ],
            ),
            (
                lambda statement: statement.startswith("SHOW RESOURCE MONITORS"),
                [Row(name="ARENA_MONITOR", credit_quota="250.00")],
            ),
            (
                lambda statement: "GET_SPENDING_LIMIT" in statement,
                [Row(GET_SPENDING_LIMIT=700)],
            ),
        ]
    )


@pytest.fixture(autouse=True)
def released_image(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, str]]]:
    """Released trainer image, a fake ``[arena]`` session, and Arena that allows training."""
    calls: list[tuple[str, dict[str, str]]] = []

    def capabilities(base_url: str, headers: dict[str, str]) -> dict[str, object]:
        calls.append((base_url, dict(headers)))
        return ENABLED

    monkeypatch.setattr(
        "agilerl.arena.snowflake.setup.image_preflight", lambda image_tag: TRAINER
    )
    monkeypatch.setattr(
        "agilerl.arena.snowflake.setup.open_session",
        lambda connection_name: JobSession(),
    )
    monkeypatch.setattr(
        "agilerl.arena.snowflake.setup.fetch_capabilities", capabilities
    )
    return calls


def patch_copy(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Pull, push, and driver check that record what they saw."""
    seen: dict[str, object] = {}

    def fake_pull(image: TrainerImage) -> None:
        seen["pulled"] = image

    def fake_copy(session: object, image: TrainerImage) -> str:
        seen["version"] = image.tag
        statements = getattr(session, "statements", [])
        seen["repository"] = any(
            "CREATE IMAGE REPOSITORY" in statement for statement in statements
        )
        return image.tag

    def fake_check(session: object, pool: str, image: str) -> str:
        seen["driver_pool"] = pool
        seen["driver_image"] = image
        return "580.65.06"

    monkeypatch.setattr("agilerl.arena.snowflake.setup.pull_images", fake_pull)
    monkeypatch.setattr("agilerl.arena.snowflake.setup.copy_images", fake_copy)
    monkeypatch.setattr("agilerl.arena.snowflake.setup.run_driver_check", fake_check)
    return seen


def request(**changes: object) -> SetupRequest:
    """``--yes`` setup of one two-GPU GPU_NV_S pool with Arena credentials."""
    base = SetupRequest(
        account=ACCOUNT,
        broker_family=None,
        head_family=None,
        gpu_families=("GPU_NV_S",),
        max_gpus=2,
        warehouse_credits=100,
        gpu_credits=500,
        key_path=None,
        rotate_key=False,
        yes=True,
        org_key="arena_org_pasted",
        external_user_id="user-1",
        base_url="https://arena.agilerl.com",
    )
    return replace(base, **changes)


class TestFormatSummary:
    def test_names_credits_pools_and_a_one_node_pool(self) -> None:
        text = format_summary(
            account=ACCOUNT,
            key_path=Path("/keys/arena.p8"),
            plans=[
                PoolPlan("arena_pulsar_pool", "CPU_X64_L", "broker", 1, 4, None),
                PoolPlan("arena_head_pool", "CPU_X64_SL", "head", 1, 4, None),
                PoolPlan("arena_gpu_GPU_NV_S", "GPU_NV_S", "gpu", 1, 1, 1),
            ],
            credits=(100, 500),
            notify_email=None,
            arena=ArenaCredentials("arena_org_x", "u", "https://arena.agilerl.com"),
            replaced=None,
            event_table="SNOWFLAKE.TELEMETRY.EVENTS",
        )

        assert "  warehouse cap   100 credits/month" in text
        assert "  pool budget     500 credits/month for all Arena pools" in text
        assert "  budget email    none (pass --notify-email for alerts)" in text
        assert f"  key             {Path('/keys/arena.p8')} (created)" in text
        assert "arena_pulsar_pool  CPU_X64_L  min 1  max 4" in text
        assert "arena_gpu_GPU_NV_S  GPU_NV_S  min 1  max 1" in text
        assert "Max nodes is 1." in text
        assert "arena_org_x" not in text
        assert "Create these objects?" not in text
        assert "EVENT_TABLE" not in text

    def test_kept_credentials_and_a_replaced_connection_are_shown(self) -> None:
        text = format_summary(
            account=ACCOUNT,
            key_path=Path("/keys/arena.p8"),
            plans=[PoolPlan("arena_gpu_GPU_NV_S", "GPU_NV_S", "gpu", 1, 2, 2)],
            credits=(100, 500),
            notify_email="ops@example.com",
            arena=None,
            replaced="[arena] in /x signs in as ada on OTHER. It is replaced.",
            event_table="SNOWFLAKE.TELEMETRY.EVENTS",
        )

        assert "  arena           keep the stored org key" in text
        assert "  budget email    ops@example.com" in text
        assert "  [arena] in /x signs in as ada on OTHER. It is replaced." in text

    def test_no_event_table_prints_the_sql_to_set_one(self) -> None:
        text = format_summary(
            account=ACCOUNT,
            key_path=Path("/keys/arena.p8"),
            plans=[PoolPlan("arena_gpu_GPU_NV_S", "GPU_NV_S", "gpu", 1, 2, 2)],
            credits=(100, 500),
            notify_email=None,
            arena=None,
            replaced=None,
            event_table="",
        )

        assert text.splitlines()[-3] == (
            "  Warning: this account has no active event table, so arena snowflake "
            "logs cannot read finished jobs. To keep their logs, run: "
            "ALTER ACCOUNT SET EVENT_TABLE = SNOWFLAKE.TELEMETRY.EVENTS"
        )


class TestSmallestGpuFamily:
    def test_fewest_gpus_then_least_memory(self) -> None:
        family = smallest_gpu_family(
            [
                GpuFamily("GPU_NV_M", 4, "A10G"),
                GpuFamily("CPU_X64_L", 0, ""),
                GpuFamily("GPU_NV_XS", 1, "T4"),
                GpuFamily("GPU_NV_S", 1, "A10G"),
            ]
        )

        assert family == "GPU_NV_XS"
        assert nodes_for_gpus(3, 4) == 1
        assert nodes_for_gpus(8, 4) == 2

    def test_no_known_gpu_family_lists_the_gpu_families_seen(self) -> None:
        families = [
            GpuFamily("CPU_X64_L", 0, ""),
            GpuFamily("GPU_H200_X8", 8, ""),
        ]

        with pytest.raises(ValueError, match="no GPU family Arena sizes") as raised:
            smallest_gpu_family(families)

        assert str(raised.value) == (
            "This account has no GPU family Arena sizes in this region.\n"
            "GPU families SHOW COMPUTE POOL INSTANCE FAMILIES returned:\n"
            "  GPU_H200_X8  8x GPU\n"
            "Nothing was created."
        )


class TestNoGpuMessage:
    def test_matches_an_empty_family_list(self) -> None:
        assert no_gpu_message("GPU_NV_S", []) == NO_GPU

    def test_a_family_list_replaces_none(self) -> None:
        text = no_gpu_message("GPU_NV_S", [GpuFamily("GPU_NV_M", 4, "A10G")])

        assert "  GPU_NV_M  4x A10G" in text
        assert "(none)" not in text


class TestRunSetup:
    def test_creates_objects_copies_the_image_and_writes_0600_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        released_image: list[tuple[str, dict[str, str]]],
    ) -> None:
        # Arrange
        connections, key = paths(tmp_path, monkeypatch)
        seen = patch_copy(monkeypatch)
        session = FakeSession(admin_routes())
        io = Scripted()

        # Act
        run_setup(session, request(), io)

        # Assert
        assert seen["pulled"] == TRAINER
        assert seen["repository"] is True
        assert seen["driver_pool"] == "arena_gpu_GPU_NV_S"
        assert seen["driver_image"] == IMAGE
        assert io.prompts == []
        assert io.lines[-2:] == [
            "The account is ready.",
            "Next: arena snowflake invite USER for each data scientist.",
        ]
        assert released_image == [
            (
                "https://arena.agilerl.com",
                {
                    "Authorization": "Bearer arena_org_pasted",
                    EXTERNAL_USER_ID_HEADER: "user-1",
                },
            )
        ]
        sql = session.statements
        assert not any(
            "CREATE OR REPLACE" in s and PROBE_INTEGRATION not in s for s in sql
        )
        assert not any("pulsar_egress" in s or "arena_pulsar_eai" in s for s in sql)
        assert not any("EVENT TABLE" in s for s in sql)
        assert "SHOW PARAMETERS LIKE 'EVENT_TABLE' IN ACCOUNT" in sql
        assert not any(s.startswith("ALTER ACCOUNT") for s in sql)
        assert (
            "CREATE EXTERNAL ACCESS INTEGRATION arena_hf_eai\n" in "\n".join(sql) + "\n"
        )
        pool = next(s for s in created(session) if "arena_gpu_GPU_NV_S" in s)
        assert "MAX_NODES = 2" in pool
        assert "INSTANCE_FAMILY = GPU_NV_S" in pool
        linked = "\n".join(s for s in sql if "ADD_RESOURCE" in s)
        for name in ("arena_pulsar_pool", "arena_head_pool", "arena_gpu_GPU_NV_S"):
            assert f"'COMPUTE POOL', '{name}'" in linked
        assert not any("SET_EMAIL_NOTIFICATIONS" in s for s in sql)
        assert not any("GRANT MODIFY" in s or "GRANT OPERATE" in s for s in sql)
        assert session.file.uploads[0]["image"] == ""
        assert session.file.uploads[1]["image"] == IMAGE
        saved = profile_from_mapping(session.file.uploads[-1])
        assert saved.image == IMAGE
        assert saved.pools.gpu["GPU_NV_S"].max_gpus == 2
        assert saved.external_access_integrations == ("arena_hf_eai",)
        arena = connection_section("arena")
        assert arena["authenticator"] == "SNOWFLAKE_JWT"
        assert arena["private_key_file"] == str(key)
        if sys.platform != "win32":
            assert mode(connections) == 0o600
            assert mode(key) == 0o600
        assert list(tmp_path.rglob("*.toml")) == [connections]
        loaded = serialization.load_pem_private_key(key.read_bytes(), password=None)
        assert loaded.key_size == 2048

    def test_the_probe_runs_before_any_prompt_and_is_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        session = FakeSession(admin_routes())

        run_setup(session, request(), Scripted())

        create = session.statements.index(
            next(s for s in session.statements if PROBE_INTEGRATION in s)
        )
        drop = session.statements.index(
            f"DROP EXTERNAL ACCESS INTEGRATION IF EXISTS {PROBE_INTEGRATION}"
        )
        assert "ENABLED = FALSE" in session.statements[create]
        assert (
            create
            < drop
            < session.statements.index("CREATE ROLE IF NOT EXISTS arena_trainer")
        )

    def test_trial_accounts_stop_before_any_prompt_key_or_pull(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        _connections, key = paths(tmp_path, monkeypatch)
        seen = patch_copy(monkeypatch)
        session = FakeSession(
            admin_routes(
                [(lambda s: PROBE_INTEGRATION in s and "CREATE" in s, Trial())]
            )
        )
        io = Scripted()

        # Act
        with pytest.raises(
            ValueError, match="not available on trial accounts"
        ) as raised:
            run_setup(session, request(yes=False, org_key=None), io)

        # Assert
        assert str(raised.value) == TRIAL_MESSAGE
        assert io.prompts == []
        assert not key.exists()
        assert "pulled" not in seen
        assert created(session) == []

    def test_a_blocked_probe_names_the_server_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = FakeSession(
            admin_routes(
                [
                    (
                        lambda s: PROBE_INTEGRATION in s and "CREATE" in s,
                        RuntimeError("policy"),
                    )
                ]
            )
        )

        with pytest.raises(ValueError, match="Snowflake said: policy"):
            run_setup(session, request(), Scripted())

        assert created(session) == []

    def test_enter_confirms_and_the_key_is_made_after_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        io = Scripted(["", ""])

        run_setup(FakeSession(admin_routes()), request(yes=False), io)

        assert io.prompts == [
            "Budget alert email (Enter to skip)",
            "Create these objects? [Y/n]",
        ]
        assert key.is_file()
        assert io.lines[-2] == "The account is ready."

    def test_without_a_family_the_smallest_listed_gpu_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        seen = patch_copy(monkeypatch)
        session = FakeSession(admin_routes())

        run_setup(session, request(gpu_families=(), max_gpus=None), Scripted())

        pool = next(s for s in created(session) if "arena_gpu_GPU_GCP_NV_L4_1_24G" in s)
        assert "MAX_NODES = 2" in pool
        assert seen["driver_pool"] == "arena_gpu_GPU_GCP_NV_L4_1_24G"
        saved = profile_from_mapping(session.file.uploads[-1])
        assert list(saved.pools.gpu) == ["GPU_GCP_NV_L4_1_24G"]

    def test_yes_without_credit_flags_uses_the_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        session = FakeSession(admin_routes())
        io = Scripted()

        run_setup(session, request(warehouse_credits=None, gpu_credits=None), io)

        assert io.prompts == []
        assert any("CREDIT_QUOTA = 100" in s for s in session.statements)
        assert (
            "CALL arena.training.arena_gpu_budget!SET_SPENDING_LIMIT(500)"
            in session.statements
        )

    def test_credit_prompts_repeat_until_a_whole_number(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        session = FakeSession(admin_routes())
        io = Scripted(["abc", "0", "150", "600", "", ""])

        # Act
        run_setup(
            session,
            request(yes=False, warehouse_credits=None, gpu_credits=None),
            io,
        )

        # Assert
        assert io.prompts == [
            "Warehouse credit cap per month [100]",
            "Warehouse credit cap per month [100]",
            "Warehouse credit cap per month [100]",
            "Compute pool credit budget per month [500]",
            "Budget alert email (Enter to skip)",
            "Create these objects? [Y/n]",
        ]
        assert io.lines[:2] == [
            "Enter a whole number greater than 0.",
            "Enter a whole number greater than 0.",
        ]
        assert any("CREDIT_QUOTA = 150" in s for s in session.statements)
        assert (
            "CALL arena.training.arena_gpu_budget!SET_SPENDING_LIMIT(600)"
            in session.statements
        )

    def test_declining_creates_nothing_and_writes_no_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        connections, key = paths(tmp_path, monkeypatch)
        seen = patch_copy(monkeypatch)
        session = FakeSession(admin_routes())
        io = Scripted(["", "n"])

        # Act
        with pytest.raises(ValueError, match="Nothing was created") as raised:
            run_setup(session, request(yes=False), io)

        # Assert
        assert str(raised.value) == "Nothing was created."
        assert not key.exists()
        assert not connections.exists()
        assert "pulled" not in seen
        assert created(session) == []

    def test_an_existing_key_is_kept_and_restricted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        key = tmp_path / "mine.p8"
        generate_private_key(key)
        key.chmod(0o644)
        original = key.read_bytes()

        run_setup(FakeSession(admin_routes()), request(key_path=key), Scripted())

        assert key.read_bytes() == original
        if sys.platform != "win32":
            assert mode(key) == 0o600

    def test_the_arena_key_is_reused_on_a_rerun(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        key = tmp_path / "elsewhere.p8"
        generate_private_key(key)
        write_connection("arena", job_connection_fields(ACCOUNT, key))
        body = public_key_body(key)
        session = FakeSession(
            admin_routes(
                [
                    (
                        lambda s: s.startswith("SHOW USERS"),
                        [Row(name="ARENA_JOB_USER")],
                    ),
                    (
                        lambda s: s.startswith("DESCRIBE USER"),
                        [Row(property="RSA_PUBLIC_KEY", value=body)],
                    ),
                ]
            )
        )

        run_setup(session, request(), Scripted())

        assert not any("RSA_PUBLIC_KEY" in s for s in session.statements)
        assert not (tmp_path / "arena_job_user.p8").exists()

    def test_a_mismatched_key_stops_before_the_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        key = tmp_path / "mine.p8"
        generate_private_key(key)
        session = FakeSession(
            admin_routes(
                [
                    (
                        lambda s: s.startswith("SHOW USERS"),
                        [Row(name="arena_job_user")],
                    ),
                    (
                        lambda s: s.startswith("DESCRIBE USER"),
                        [Row(property="RSA_PUBLIC_KEY", value="different")],
                    ),
                ]
            )
        )
        io = Scripted()

        with pytest.raises(ValueError, match="does not match") as raised:
            run_setup(session, request(key_path=key), io)

        assert str(raised.value) == key_mismatch_message(key)
        assert io.lines == []
        assert created(session) == []

    def test_a_user_key_with_no_local_file_stops_without_making_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        session = FakeSession(
            admin_routes(
                [
                    (
                        lambda s: s.startswith("SHOW USERS"),
                        [Row(name="arena_job_user")],
                    ),
                    (
                        lambda s: s.startswith("DESCRIBE USER"),
                        [Row(property="RSA_PUBLIC_KEY", value="MIIB")],
                    ),
                ]
            )
        )

        with pytest.raises(ValueError, match="--rotate-key"):
            run_setup(session, request(), Scripted())

        assert not key.exists()

    def test_a_missing_key_flag_creates_nothing(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.p8"
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="Leave --key unset") as raised:
            run_setup(session, request(key_path=missing), Scripted())

        assert str(raised.value) == missing_setup_key_message(missing)
        assert created(session) == []

    def test_no_gpu_family_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = FakeSession(
            admin_routes([(lambda s: s == "SHOW COMPUTE POOL INSTANCE FAMILIES", [])])
        )

        with pytest.raises(ValueError, match="no GPU_NV_S pool") as raised:
            run_setup(session, request(), Scripted())

        assert str(raised.value) == NO_GPU
        assert created(session) == []

    def test_an_unlisted_broker_family_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="no CPU_X64_M instance family") as raised:
            run_setup(session, request(broker_family="cpu_x64_m"), Scripted())

        assert str(raised.value) == (
            "This account has no CPU_X64_M instance family in this region.\n"
            "The broker pool was not created.\n"
            "Nothing was created."
        )
        assert created(session) == []

    def test_an_unknown_gpu_family_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="GPU_NOPE is not a GPU family"):
            run_setup(session, request(gpu_families=("gpu_nope",)), Scripted())

        assert created(session) == []

    def test_a_session_that_is_not_accountadmin_creates_nothing(self) -> None:
        session = FakeSession(
            [
                (
                    lambda s: s.startswith("SELECT CURRENT_USER"),
                    [
                        Row(
                            **{
                                "CURRENT_USER()": "NICK",
                                "CURRENT_ROLE()": "PUBLIC",
                                "CURRENT_ACCOUNT()": "ORG",
                                "CURRENT_REGION()": "AWS",
                            }
                        )
                    ],
                ),
                (lambda s: s.startswith("USE ROLE"), RuntimeError("no")),
            ]
        )

        with pytest.raises(ValueError, match="current role: PUBLIC"):
            run_setup(session, request(), Scripted())

        assert created(session) == []

    def test_a_foreign_database_is_left_alone(self) -> None:
        session = FakeSession(
            admin_routes(
                [
                    (lambda s: s.startswith("SHOW DATABASES"), [Row(name="arena")]),
                    (
                        lambda s: s.startswith("SHOW TABLES"),
                        RuntimeError("Schema 'ARENA.TRAINING' does not exist"),
                    ),
                    (lambda s: s.startswith("LIST"), RuntimeError("does not exist")),
                    (lambda s: s.startswith("SHOW SCHEMAS"), [Row(name="PUBLIC")]),
                ]
            )
        )

        with pytest.raises(ValueError, match="not an Arena account") as raised:
            run_setup(session, request(), Scripted())

        assert str(raised.value) == FOREIGN_DATABASE
        assert not any(PROBE_INTEGRATION in s for s in session.statements)

    def test_a_different_pool_family_is_not_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = FakeSession(
            admin_routes(
                [
                    (
                        lambda s: s == "SHOW COMPUTE POOLS",
                        [
                            Row(
                                name="arena_pulsar_pool",
                                instance_family="CPU_X64_S",
                                max_nodes=4,
                            )
                        ],
                    )
                ]
            )
        )

        with pytest.raises(ValueError, match="will not replace a pool") as raised:
            run_setup(session, request(), Scripted())

        assert str(raised.value) == pool_family_message(
            "arena_pulsar_pool", "CPU_X64_S"
        )
        assert created(session) == []

    def test_many_gpu_nodes_are_left_to_snowflake_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        session = FakeSession(admin_routes())

        run_setup(session, request(gpu_families=("GPU_NV_M",), max_gpus=44), Scripted())

        pool = next(s for s in created(session) if "arena_gpu_GPU_NV_M" in s)
        assert "MAX_NODES = 11" in pool

    def test_a_google_cloud_family_is_sized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        session = FakeSession(admin_routes())

        run_setup(
            session,
            request(gpu_families=("gpu_gcp_nv_l4_1_24g",), max_gpus=3),
            Scripted(),
        )

        uploaded = profile_from_mapping(session.file.uploads[-1])
        gpu = uploaded.pools.gpu["GPU_GCP_NV_L4_1_24G"]
        assert (gpu.max_gpus, gpu.gpus_per_node, gpu.gpu_memory_gib) == (3, 1, 24)
        assert any("MAX_NODES = 3" in s for s in created(session))

    def test_max_nodes_changes_without_a_new_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        session = FakeSession(
            admin_routes(
                [
                    (
                        lambda s: s == "SHOW COMPUTE POOLS",
                        [
                            Row(
                                name="arena_gpu_GPU_NV_S",
                                instance_family="GPU_NV_S",
                                max_nodes=2,
                            )
                        ],
                    )
                ]
            )
        )

        run_setup(session, request(max_gpus=4), Scripted())

        assert (
            "ALTER COMPUTE POOL arena_gpu_GPU_NV_S SET MAX_NODES = 4"
            in session.statements
        )
        assert not any("arena_gpu_GPU_NV_S" in s for s in created(session))

    def test_a_notify_email_is_set_on_the_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        session = FakeSession(admin_routes())

        run_setup(session, request(notify_email="ops@example.com"), Scripted())

        assert (
            "CALL arena.training.arena_gpu_budget!SET_EMAIL_NOTIFICATIONS('ops@example.com')"
            in session.statements
        )

    def test_a_bad_notify_email_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="is not an email address"):
            run_setup(session, request(notify_email="ops"), Scripted())

        assert created(session) == []

    def test_a_failed_copy_leaves_the_login_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, _key = paths(tmp_path, monkeypatch)

        def fail(session: object, image: TrainerImage) -> str:
            raise ValueError(READ_ONLY_REPOSITORY)

        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup.pull_images", lambda image: None
        )
        monkeypatch.setattr("agilerl.arena.snowflake.setup.copy_images", fail)
        session = FakeSession(admin_routes())
        io = Scripted()

        with pytest.raises(ValueError, match="It has READ"):
            run_setup(session, request(), io)

        assert connections.is_file()
        if sys.platform != "win32":
            assert mode(connections) == 0o600
        assert session.file.uploads[-1]["image"] == ""
        assert "The account is ready." not in io.lines

    def test_a_failed_driver_check_keeps_the_image_and_is_not_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)

        def fail(session: object, pool: str, image: str) -> str:
            assert isinstance(session, JobSession)
            msg = "The GPU driver is 535.54.03. Arena needs 580.65.06 or newer."
            raise ValueError(msg)

        monkeypatch.setattr("agilerl.arena.snowflake.setup.run_driver_check", fail)
        session = FakeSession(admin_routes())
        io = Scripted()

        with pytest.raises(ValueError, match="The GPU driver is 535"):
            run_setup(session, request(), io)

        assert "The account is ready." not in io.lines
        assert session.file.uploads[-1]["image"] == IMAGE

    def test_arena_unreachable_from_snowflake_fails_setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup.run_driver_check", run_driver_check
        )
        job = driver_job_session(
            "580.65.06\n"
            "arena-license-check: Could not reach Arena at https://arena.internal: "
            "[Errno -2] Name or service not known.\n"
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup.open_session", lambda connection_name: job
        )
        io = Scripted()

        # Act
        with pytest.raises(ValueError, match="Arena check from inside") as raised:
            run_setup(FakeSession(admin_routes()), request(), io)

        # Assert
        assert str(raised.value) == (
            "The Arena check from inside Snowflake failed: Could not reach Arena "
            "at https://arena.internal: [Errno -2] Name or service not known."
        )
        assert "The account is ready." not in io.lines
        executed = next(s for s in job.statements if s.startswith("EXECUTE JOB"))
        assert "EXTERNAL_ACCESS_INTEGRATIONS = (arena_api_eai)" in executed
        assert "envVarName: ARENA_BASE_URL" in executed

    def test_driver_and_arena_ok_from_snowflake_make_the_account_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup.run_driver_check", run_driver_check
        )
        job = driver_job_session("580.65.06\narena-license-check: ok\n")
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup.open_session", lambda connection_name: job
        )
        io = Scripted()

        run_setup(FakeSession(admin_routes()), request(), io)

        assert "The account is ready." in io.lines
        assert job.statements[-1].startswith("DROP SERVICE IF EXISTS ARENA_DRIVER_")

    def test_a_missing_image_tag_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup.image_preflight", image_preflight
        )
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="Pass --image-tag") as raised:
            run_setup(session, request(), Scripted())

        assert str(raised.value) == MISSING_IMAGE_TAG
        assert session.statements[-1] == "SELECT CURRENT_ROLE()"

    def test_an_image_that_cannot_be_pulled_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)

        def fail(image: TrainerImage) -> None:
            raise ValueError(image_not_found_message(image.tag))

        monkeypatch.setattr("agilerl.arena.snowflake.setup.pull_images", fail)
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="does not exist"):
            run_setup(session, request(), Scripted())

        assert created(session) == []


class TestRunSetupArenaCredentials:
    def test_secrets_are_created_then_altered_and_the_rule_names_one_host(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        session = FakeSession(admin_routes())

        # Act
        run_setup(
            session,
            request(
                org_key="arena_org_ab'c",
                external_user_id="user-1",
                base_url="https://arena.example.com",
            ),
            Scripted(),
        )

        # Assert
        sql = session.statements
        assert (
            "CREATE SECRET IF NOT EXISTS arena.training.arena_org_key\n"
            "  TYPE = GENERIC_STRING\n"
            "  SECRET_STRING = 'arena_org_ab''c'"
        ) in sql
        assert (
            "ALTER SECRET arena.training.arena_org_key SET SECRET_STRING = 'arena_org_ab''c'"
        ) in sql
        assert (
            "ALTER SECRET arena.training.arena_base_url "
            "SET SECRET_STRING = 'https://arena.example.com'"
        ) in sql
        rule = next(
            s for s in sql if "NETWORK RULE arena.training.arena_api_egress" in s
        )
        assert rule.startswith("CREATE OR ALTER NETWORK RULE")
        assert "'arena.example.com:443'" in rule
        assert rule.count(":443") == 1
        assert (
            "CREATE EXTERNAL ACCESS INTEGRATION arena_api_eai\n"
            "  ALLOWED_NETWORK_RULES = (arena.training.arena_api_egress)\n"
            "  ENABLED = TRUE"
        ) in sql
        assert "GRANT USAGE ON INTEGRATION arena_api_eai TO ROLE arena_trainer" in sql

    def test_prompts_are_hidden_and_checked_with_arena(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        released_image: list[tuple[str, dict[str, str]]],
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        io = Scripted(["arena_org_typed", "user-9", "", ""])

        run_setup(
            FakeSession(admin_routes()),
            request(yes=False, org_key=None, external_user_id=None),
            io,
        )

        assert io.prompts[:2] == ["Arena org key", "Arena external user id"]
        assert io.hidden[:2] == [True, True]
        assert released_image[0][1]["Authorization"] == "Bearer arena_org_typed"

    def test_a_key_without_the_org_prefix_creates_nothing(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="starts with arena_org_") as raised:
            run_setup(session, request(org_key="arena_pat_x"), Scripted())

        assert str(raised.value) == (
            "That is not an Arena org key.\n"
            "Ask AgileRL for a Snowflake org key (it starts with arena_org_).\n"
            "Nothing was created."
        )
        assert created(session) == []

    def test_rejected_credentials_stop_before_any_prompt_after_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def reject(base_url: str, headers: dict[str, str]) -> dict[str, object]:
            raise SnowflakeLicenseError(AUTH_REJECTED)

        monkeypatch.setattr("agilerl.arena.snowflake.setup.fetch_capabilities", reject)
        session = FakeSession(admin_routes())
        io = Scripted()

        with pytest.raises(SnowflakeLicenseError, match="rejected the org key"):
            run_setup(session, request(yes=False), io)

        assert io.prompts == []
        assert created(session) == []

    def test_an_organisation_without_snowflake_training_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup.fetch_capabilities",
            lambda base_url, headers: {"schemaVersion": 1, "features": {}},
        )
        session = FakeSession(admin_routes())

        with pytest.raises(SnowflakeLicenseError) as raised:
            run_setup(session, request(), Scripted())

        assert str(raised.value) == SNOWFLAKE_DENIED
        assert created(session) == []

    def test_yes_without_credentials_names_both_flags(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="Nothing was created") as raised:
            run_setup(session, request(org_key=None, external_user_id=None), Scripted())

        assert str(raised.value) == (
            "Pass --org-key and --external-user-id.\nNothing was created."
        )

    def test_yes_without_one_credential_names_that_flag(self) -> None:
        with pytest.raises(ValueError, match="Pass --org-key") as raised:
            run_setup(FakeSession(admin_routes()), request(org_key=None), Scripted())

        assert str(raised.value) == "Pass --org-key.\nNothing was created."

    def test_an_empty_org_key_creates_nothing(self) -> None:
        session = FakeSession(admin_routes())

        with pytest.raises(ValueError, match="is required") as raised:
            run_setup(
                session,
                request(yes=False, org_key=None, external_user_id=None),
                Scripted([""]),
            )

        assert str(raised.value) == "Arena org key is required.\nNothing was created."
        assert created(session) == []


class TestRunSetupRerun:
    def test_saved_families_caps_quotas_and_secrets_are_kept(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        released_image: list[tuple[str, dict[str, str]]],
    ) -> None:
        # Arrange
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        doc = json.loads(json.dumps(PROFILE_DOC))
        doc["pools"]["gpu"] = {
            "GPU_NV_M": {
                "pool": "arena_gpu_GPU_NV_M",
                "max_gpus": 8,
                "gpus_per_node": 4,
                "gpu_memory_gib": 24,
            }
        }
        doc["external_access_integrations"] = ["arena_hf_eai", "arena_env_eai"]
        session = FakeSession(rerun_routes(doc))
        io = Scripted([""])

        # Act
        run_setup(
            session,
            request(
                yes=False,
                gpu_families=(),
                max_gpus=None,
                warehouse_credits=None,
                gpu_credits=None,
                org_key=None,
                external_user_id=None,
                notify_email="ops@example.com",
            ),
            io,
        )

        # Assert
        sql = session.statements
        assert io.prompts == ["Create these objects? [Y/n]"]
        assert released_image == []
        assert not any(s.startswith(("CREATE SECRET", "ALTER SECRET")) for s in sql)
        assert not any("arena_api_egress" in s and "RULE" in s for s in sql)
        assert (
            "ALTER EXTERNAL ACCESS INTEGRATION arena_hf_eai SET "
            "ALLOWED_NETWORK_RULES = (arena.training.hf_egress) ENABLED = TRUE"
        ) in sql
        assert not any(PROBE_INTEGRATION in s for s in sql)
        assert any("CREDIT_QUOTA = 250" in s for s in sql)
        assert "CALL arena.training.arena_gpu_budget!SET_SPENDING_LIMIT(700)" in sql
        assert any("arena_gpu_GPU_NV_M" in s and "MAX_NODES = 2" in s for s in sql)
        saved = profile_from_mapping(session.file.uploads[-1])
        assert list(saved.pools.gpu) == ["GPU_NV_M"]
        assert saved.pools.gpu["GPU_NV_M"].max_gpus == 8
        assert saved.external_access_integrations == ("arena_hf_eai", "arena_env_eai")

    def test_a_replaced_arena_connection_is_warned_about(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        write_connection(
            "arena",
            {
                "account": "OTHER-ACCOUNT",
                "user": "ada",
                "authenticator": "SNOWFLAKE_JWT",
            },
        )
        io = Scripted()

        run_setup(FakeSession(admin_routes()), request(), io)

        assert "[arena] in" in io.lines[0]
        assert "signs in as ada on OTHER-ACCOUNT. It is replaced." in io.lines[0]


class TestPrintSetupSql:
    def test_prints_placeholders_and_creates_nothing(self) -> None:
        text = print_setup_sql(
            request(
                account="",
                org_key="arena_org_real",
                notify_email="ops@example.com",
                base_url="https://arena.example.com",
            )
        )

        assert text.startswith("-- Statements arena snowflake setup runs")
        assert "USE ROLE ACCOUNTADMIN;" in text
        assert "SECRET_STRING = '<arena org key>'" in text
        assert "arena_org_real" not in text
        assert (
            "RSA_PUBLIC_KEY = '<public key of the arena_job_user private key>'" in text
        )
        assert "SECRET_STRING = 'https://arena.example.com'" in text
        assert "SET_EMAIL_NOTIFICATIONS('ops@example.com');" in text
        assert "CREATE OR REPLACE" not in text
        assert text.rstrip().endswith(";")

    def test_an_unknown_family_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a GPU family Arena sizes"):
            print_setup_sql(request(gpu_families=("GPU_NOPE",)))


class TestSetupCommand:
    def test_print_sql_needs_no_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, _key = paths(tmp_path, monkeypatch)

        result = CliRunner().invoke(main, ["snowflake", "setup", "--print-sql"])

        assert result.exit_code == 0, result.output
        assert "CREATE COMPUTE POOL IF NOT EXISTS arena_gpu_GPU_NV_S" in result.output
        assert not connections.exists()

    def test_yes_requires_a_connection(self) -> None:
        result = CliRunner().invoke(main, ["snowflake", "setup", "--yes"])

        assert result.exit_code != 0
        assert "--yes requires --connection." in result.output

    def test_flags_root_options_and_env_reach_the_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        paths(tmp_path, monkeypatch)
        write_connection("admin", admin_connection_fields(ACCOUNT, "NICK"))
        monkeypatch.setenv("ARENA_EXTERNAL_USER_ID", "env-user")
        seen: dict[str, SetupRequest] = {}

        def fake_run(session: object, setup_request: SetupRequest, io: object) -> None:
            seen["request"] = setup_request

        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: FakeSession(),
        )
        monkeypatch.setattr("agilerl.arena.snowflake.setup_cli.run_setup", fake_run)

        # Act
        result = CliRunner().invoke(
            main,
            [
                "--base-url",
                "https://staging.example.com/api",
                "--org-key",
                "arena_org_root",
                "snowflake",
                "setup",
                "--connection",
                "admin",
                "--yes",
                "--notify-email",
                "ops@example.com",
                "--image-tag",
                "ml/v32.6.0",
            ],
        )

        # Assert
        assert result.exit_code == 0, result.output
        sent = seen["request"]
        assert sent.account == ACCOUNT
        assert sent.org_key == "arena_org_root"
        assert sent.external_user_id == "env-user"
        assert sent.base_url == "https://staging.example.com"
        assert sent.notify_email == "ops@example.com"
        assert sent.image_tag == "ml/v32.6.0"

    def test_browser_sign_in_writes_oauth_without_a_role(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, _key = paths(tmp_path, monkeypatch)
        opened: list[str] = []
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: opened.append(name) or FakeSession(),
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.run_setup",
            lambda session, setup_request, io: opened.append(setup_request.account),
        )

        result = CliRunner().invoke(
            main, ["snowflake", "setup"], input="ORGNAME-ACCOUNT\nnick@example.com\n"
        )

        assert result.exit_code == 0, result.output
        assert "Snowflake login name" in result.output
        text = connections.read_text(encoding="utf-8")
        assert 'authenticator = "OAUTH_AUTHORIZATION_CODE"' in text
        assert "role" not in text
        assert "nick@example.com" in text
        if sys.platform != "win32":
            assert mode(connections) == 0o600
        assert opened == ["arena-admin", ACCOUNT]

    def test_an_existing_admin_connection_is_reused_without_prompts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, _key = paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))
        before = connections.read_text(encoding="utf-8")
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: FakeSession(),
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.run_setup", lambda session, req, io: None
        )

        result = CliRunner().invoke(main, ["snowflake", "setup"])

        assert result.exit_code == 0, result.output
        assert f"Signing in with [arena-admin] from {connections}" in result.output
        assert connections.read_text(encoding="utf-8") == before

    def test_a_sql_error_is_one_line_with_the_rerun_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("admin", admin_connection_fields(ACCOUNT, "NICK"))

        class SqlError(Exception):
            pass

        def fail(session: object, setup_request: SetupRequest, io: object) -> None:
            msg = "002003 (02000): SQL compilation error:\nObject does not exist"
            raise SqlError(msg)

        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.sql_errors", lambda: (SqlError,)
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: FakeSession(),
        )
        monkeypatch.setattr("agilerl.arena.snowflake.setup_cli.run_setup", fail)

        result = CliRunner().invoke(
            main, ["snowflake", "setup", "--connection", "admin"]
        )

        assert result.exit_code == 1
        assert result.output == (
            "Error: Snowflake said: 002003 (02000): SQL compilation error:\n"
            "Re-run arena snowflake setup.\n"
        )

    def test_rejected_arena_credentials_are_one_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("admin", admin_connection_fields(ACCOUNT, "NICK"))

        def fail(session: object, setup_request: SetupRequest, io: object) -> None:
            raise SnowflakeLicenseError(AUTH_REJECTED)

        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: FakeSession(),
        )
        monkeypatch.setattr("agilerl.arena.snowflake.setup_cli.run_setup", fail)

        result = CliRunner().invoke(
            main, ["snowflake", "setup", "--connection", "admin"]
        )

        assert result.exit_code == 1
        assert result.output == f"Error: {AUTH_REJECTED}\n"

    def test_prompts_read_hidden_credentials_and_take_defaults_on_enter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        released_image: list[tuple[str, dict[str, str]]],
    ) -> None:
        # Arrange
        paths(tmp_path, monkeypatch)
        patch_copy(monkeypatch)
        write_connection("admin", admin_connection_fields(ACCOUNT, "NICK"))
        session = FakeSession(admin_routes())
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: session,
        )

        # Act
        result = CliRunner().invoke(
            main,
            ["snowflake", "setup", "--connection", "admin"],
            input="arena_org_typed\nuser-9\n\n\n\n\n",
        )

        # Assert
        assert result.exit_code == 0, result.output
        assert "arena_org_typed" not in result.output
        assert "user-9" not in result.output
        assert "The account is ready." in result.output
        assert released_image[0][1]["Authorization"] == "Bearer arena_org_typed"
        assert released_image[0][1][EXTERNAL_USER_ID_HEADER] == "user-9"
        assert any("CREDIT_QUOTA = 100" in s for s in session.statements)
        assert session.closed == 1

    def test_a_connection_without_an_account_opens_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection(
            "admin", {"user": "NICK", "authenticator": "OAUTH_AUTHORIZATION_CODE"}
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: pytest.fail("opened a session"),
        )

        result = CliRunner().invoke(
            main, ["snowflake", "setup", "--connection", "admin"]
        )

        assert result.exit_code == 1
        assert result.output == "Error: Connection admin has no account.\n"


class TestCheckCommand:
    def test_reports_the_check_failure_from_the_arena_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        _connections, key = paths(tmp_path, monkeypatch)
        generate_private_key(key)
        write_connection("arena", job_connection_fields(ACCOUNT, key))
        session = FakeSession(
            [
                (
                    lambda s: s.startswith("USE ROLE"),
                    RuntimeError("Role 'ARENA_TRAINER' does not exist"),
                )
            ]
        )
        opened: list[str] = []
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_session",
            lambda name: opened.append(name) or session,
        )

        # Act
        result = CliRunner().invoke(main, ["snowflake", "check"])

        # Assert
        assert result.exit_code == 1
        assert result.output == (
            "Error: The role arena_trainer cannot be used.\n"
            "Role 'ARENA_TRAINER' does not exist\n"
            "Re-run arena snowflake setup.\n"
        )
        assert opened == ["arena"]
        assert session.statements == ["USE ROLE arena_trainer"]
        assert session.closed == 1


def join_session() -> FakeSession:
    """``[arena]`` session for a user that has arena_trainer on a set-up account."""
    return FakeSession(
        [
            (lambda s: s.startswith("USE ROLE"), []),
            (lambda s: s.startswith("SELECT $1"), [Row(DOC=json.dumps(PROFILE_DOC))]),
        ]
    )


def join_fields(**auth: str) -> dict[str, str]:
    """``[arena]`` section ``join`` writes for Ada."""
    return {
        "account": ACCOUNT,
        "user": "Ada",
        "role": "arena_trainer",
        "warehouse": "arena_wh",
        "database": "arena",
        "schema": "training",
        **auth,
    }


class TestJoinCommand:
    def test_a_key_flag_writes_a_key_pair_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        key = tmp_path / "ada.p8"
        generate_private_key(key)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_session",
            lambda name: join_session(),
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
                "--key",
                str(key),
            ],
        )

        assert result.exit_code == 0, result.output
        assert connection_section("arena") == join_fields(
            authenticator="SNOWFLAKE_JWT", private_key_file=str(key)
        )

    def test_a_prompted_token_is_hidden_and_saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_session",
            lambda name: join_session(),
        )

        result = CliRunner().invoke(
            main,
            ["snowflake", "join", "--account", ACCOUNT, "--user", "Ada"],
            input="Token\nsekret\n",
        )

        assert result.exit_code == 0, result.output
        assert "Programmatic access token" in result.output
        assert "sekret" not in result.output
        assert connection_section("arena") == join_fields(
            authenticator="PROGRAMMATIC_ACCESS_TOKEN", token="sekret"
        )

    def test_a_prompted_key_path_is_saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        key = tmp_path / "ada.p8"
        generate_private_key(key)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_session",
            lambda name: join_session(),
        )

        result = CliRunner().invoke(
            main,
            ["snowflake", "join", "--account", ACCOUNT, "--user", "Ada"],
            input=f"key\n{key}\n",
        )

        assert result.exit_code == 0, result.output
        assert connection_section("arena") == join_fields(
            authenticator="SNOWFLAKE_JWT", private_key_file=str(key)
        )

    def test_a_missing_key_file_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, _key = paths(tmp_path, monkeypatch)
        gone = tmp_path / "gone.p8"

        result = CliRunner().invoke(
            main,
            [
                "snowflake",
                "join",
                "--account",
                ACCOUNT,
                "--user",
                "Ada",
                "--key",
                str(gone),
            ],
        )

        assert result.exit_code == 1
        assert result.output == f"Error: {missing_key_message(gone)}\n"
        assert not connections.exists()

    def test_an_unknown_sign_in_method_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, _key = paths(tmp_path, monkeypatch)

        result = CliRunner().invoke(
            main,
            ["snowflake", "join", "--account", ACCOUNT, "--user", "Ada"],
            input="sso\n",
        )

        assert result.exit_code == 1
        assert result.output.endswith("Error: Sign in with token or key.\n")
        assert not connections.exists()


class TestPoolsCommand:
    def test_lists_each_pool_with_its_nodes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))
        shown = (
            lambda s: s == "SHOW COMPUTE POOLS",
            [
                Row(
                    name="ARENA_PULSAR_POOL",
                    instance_family="CPU_X64_L",
                    max_nodes=4,
                    active_nodes=1,
                ),
                Row(
                    name="ARENA_HEAD_POOL",
                    instance_family="CPU_X64_SL",
                    max_nodes=4,
                    active_nodes=0,
                ),
                Row(
                    name="ARENA_GPU_GPU_NV_S",
                    instance_family="GPU_NV_S",
                    max_nodes=2,
                    active_nodes=2,
                ),
            ],
        )
        session = FakeSession([shown, *rerun_routes(PROFILE_DOC)])
        opened: list[str] = []
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: opened.append(name) or session,
        )

        # Act
        result = CliRunner().invoke(main, ["snowflake", "pools"])

        # Assert
        assert result.exit_code == 0, result.output
        assert result.output == (
            "arena_pulsar_pool  CPU_X64_L  broker  1 node\n"
            "arena_head_pool  CPU_X64_SL  head  0 nodes\n"
            "arena_gpu_GPU_NV_S  GPU_NV_S  max 2 GPUs  2 nodes\n"
        )
        assert opened == ["arena-admin"]
        assert session.closed == 1
