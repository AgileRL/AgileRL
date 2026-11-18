# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native App run procedures: submit, status, logs, cancel, publish, endpoint, unpublish."""

from __future__ import annotations

import io
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake_app.app_code.arena_app import (
    IMAGE,
    cancel,
    endpoint,
    logs,
    publish,
    status,
    submit,
    unpublish,
)

from agilerl.arena.snowflake import license as license_mod
from agilerl.arena.snowflake.license import SnowflakeLicenseError
from tests.test_snowflake_license import Arena, reply
from tests.test_snowflake_setup import Row

RUN_ID = "arena-20261109163457-abc123"
STEM = "ARENA_RUN_20261109163457_ABC123"
RUN_DIR = f"@app_state.runs/{RUN_ID}"
SETUP = {
    "image": IMAGE,
    "warehouse": "AGILERL_APP_ARENA_APP_WH",
    "stage": "app_state.runs",
    "metrics_table": "app_state.arena_metrics",
    "external_access_integrations": ["AGILERL_APP_ARENA_APP_HF_EAI"],
    "pools": {
        "broker": {"pool": "AGILERL_APP_ARENA_APP_BROKER_POOL", "family": "CPU_X64_L"},
        "head": {"pool": "AGILERL_APP_ARENA_APP_HEAD_POOL", "family": "CPU_X64_SL"},
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
SFT = yaml.safe_dump(
    {
        "algorithm": {"name": "SFT"},
        "environment": {"columns": {"prompt": "PROMPT", "response": "RESPONSE"}},
        "network": {
            "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
            "max_context_length": 128,
        },
        "training": {"max_steps": 2},
    }
)
# An owner's-rights procedure runs in its own schema, the versioned app_code.
CURRENT = Row(DATABASE_NAME="ARENA_APP", SCHEMA_NAME="APP_CODE")
Route = tuple[Callable[[str], bool], list[Row]]
SECRETS = {
    "org_key": "arena_org_key-1",
    "external_user_id": "user-1",
    "base_url": "https://arena.test",
}
DEPLOYMENT_ID = "QWEN_CHECKPOINT_2-20261116T130000Z"
DEPLOYMENT_ID_RE = re.compile(r"QWEN_CHECKPOINT_2-\d{8}T\d{6}Z")
INACTIVE = "The Snowflake entitlement is not active."
TRIAL_QUERY = re.compile(
    r"SELECT NOT app_code\.listing_purchased\(\) AND MIN\(started_at\) <= "
    r"DATEADD\('second', -(\d+), CURRENT_TIMESTAMP\(\)\) AS ENDED FROM app_state\.trial"
)
TRIAL_ENDED = "The free trial of AgileRL Arena has ended"


@pytest.fixture
def arena(monkeypatch: pytest.MonkeyPatch) -> Arena:
    """Arena behind the app's saved secrets."""
    monkeypatch.setitem(
        sys.modules,
        "_snowflake",
        SimpleNamespace(get_generic_secret_string=lambda name: SECRETS[name]),
    )
    fake = Arena([])
    monkeypatch.setattr(license_mod.httpx, "Client", fake)
    return fake


class Result:
    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows

    def collect(self) -> list[Row]:
        return list(self.rows)


class Files:
    def __init__(self, staged: dict[str, str]) -> None:
        self.staged = staged
        self.puts: dict[str, str] = {}

    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = False,
        overwrite: bool = False,
    ) -> None:
        path = Path(local)
        self.puts[f"{stage}/{path.name}"] = path.read_text(encoding="utf-8")

    def get_stream(self, stage_location: str) -> io.BytesIO:
        return io.BytesIO(self.staged[stage_location].encode("utf-8"))


@dataclass
class Install:
    """When the app was installed and what ``app_code.listing_purchased()`` returns.

    The default is an install not from a listing, which reads as purchased.
    """

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    purchased: bool = True

    def ended(self, statement: str) -> Result | None:
        """Evaluate the trial query like Snowflake, or ``None`` for any other statement.

        :param statement: SQL to answer.
        :return: One ``ENDED`` row, or ``None``.
        """
        match = TRIAL_QUERY.fullmatch(statement)
        if match is None:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=int(match.group(1)))
        return Result([Row(ENDED=not self.purchased and self.started_at <= cutoff)])


def listing_trial(age_days: float) -> Install:
    """An unpurchased listing install made ``age_days`` ago."""
    started = datetime.now(timezone.utc) - timedelta(days=age_days)
    return Install(started, purchased=False)


class AppSession:
    """Procedure session that records statements and answers from routes."""

    def __init__(
        self,
        routes: list[Route] | None = None,
        staged: dict[str, str] | None = None,
        install: Install | None = None,
    ) -> None:
        self.routes = list(routes or [])
        self.statements: list[str] = []
        self.file = Files(staged or {})
        self.install = install or Install()

    def sql(self, statement: str) -> Result:
        self.statements.append(statement)
        if (ended := self.install.ended(statement)) is not None:
            return ended
        for pred, rows in self.routes:
            if pred(statement):
                return Result(rows)
        return Result([])


class FailingHeadSession(AppSession):
    """Procedure session where starting the head service fails."""

    def sql(self, statement: str) -> Result:
        if statement.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_HEAD_POOL"
        ):
            self.statements.append(statement)
            msg = "head pool is full"
            raise SnowparkSQLException(msg)
        return super().sql(statement)


def exact(statement: str, rows: list[Row]) -> Route:
    return (lambda s: s == statement, rows)


def setup_route(setup: dict | None = None) -> Route:
    return (
        lambda s: s.startswith(
            "SELECT $1 AS DOC FROM @app_state.runs/_arena/setup.json"
        ),
        [Row(DOC=json.dumps(setup or SETUP))],
    )


def known(run_id: str = RUN_ID) -> Route:
    return exact(
        f"LIST @app_state.runs/{run_id}/job.json",
        [Row(name=f"runs/{run_id}/job.json", size=10)],
    )


def executes(session: AppSession) -> list[str]:
    return [s for s in session.statements if s.startswith("EXECUTE JOB SERVICE")]


def spec(statement: str) -> dict:
    return yaml.safe_load(statement.split("$$\n", 1)[1].removesuffix("$$"))


def submit_session(
    setup: dict | None = None,
    region: list[Row] | None = None,
    install: Install | None = None,
) -> AppSession:
    return AppSession(
        [
            setup_route(setup),
            exact(
                "SELECT app_code.object_prefix() AS PREFIX",
                [Row(PREFIX="AGILERL_APP_ARENA_APP_")],
            ),
            exact(
                "SHOW COMPUTE POOL INSTANCE FAMILIES",
                region or [Row(name="GPU_NV_S", gpu_count=1)],
            ),
            (
                lambda s: s.startswith(
                    "DESCRIBE COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_S"
                ),
                [Row(instance_family="GPU_NV_S", max_nodes=2)],
            ),
            (
                lambda s: s.startswith(
                    "DESCRIBE COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_R6K_G1_48"
                ),
                [Row(instance_family="GPU_R6K_G1_48", max_nodes=2)],
            ),
        ],
        install=install,
    )


def r6k_setup() -> dict:
    """``SETUP`` with a GPU_R6K_G1_48 pool recorded at the AWS layout beside GPU_NV_S."""
    r6k = {
        "pool": "AGILERL_APP_ARENA_APP_GPU_GPU_R6K_G1_48",
        "max_gpus": 4,
        "gpus_per_node": 2,
        "gpu_memory_gib": 96,
    }
    pools = {**SETUP["pools"], "gpu": {**SETUP["pools"]["gpu"], "GPU_R6K_G1_48": r6k}}
    return {**SETUP, "pools": pools}


def capabilities(priced: list[str] | None) -> httpx.Response:
    """Arena capabilities with ``priced`` as the priced families, or none listed."""
    features: dict[str, object] = {"snowflakeTraining": True}
    if priced is not None:
        features["snowflakePricedGpuFamilies"] = priced
    return reply(200, {"ok": True, "data": {"schemaVersion": 1, "features": features}})


class TestSubmit:
    @pytest.fixture(autouse=True)
    def priced_arena(self, arena: Arena) -> Arena:
        arena.replies = [capabilities(["GPU_NV_S"])]
        return arena

    def test_starts_the_cli_services_on_the_app_pools_and_secrets(self) -> None:
        # Arrange
        session = submit_session()

        # Act
        text = submit(session, SFT, 1, "source_table", "prompts", None, None, 5, None)

        # Assert
        run_id = text.splitlines()[0].removeprefix("run_id: ")
        broker, head, workers = executes(session)
        assert broker.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_BROKER_POOL\n"
        )
        assert head.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_HEAD_POOL\n"
        )
        assert workers.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_S\n"
        )
        assert all(
            "QUERY_WAREHOUSE = AGILERL_APP_ARENA_APP_WH" in s
            for s in (broker, head, workers)
        )
        assert "EXTERNAL_ACCESS_INTEGRATIONS" not in broker
        assert (
            "EXTERNAL_ACCESS_INTEGRATIONS = (AGILERL_APP_ARENA_APP_HF_EAI, AGILERL_APP_ARENA_APP_ARENA_EAI)"
            in head
        )
        assert (
            "EXTERNAL_ACCESS_INTEGRATIONS = (AGILERL_APP_ARENA_APP_HF_EAI)\n" in workers
        )
        [container] = spec(head)["spec"]["containers"]
        assert container["image"] == IMAGE
        assert container["env"]["ARENA_SNOWFLAKE_CHANNEL"] == "native_app"
        assert all(
            "ARENA_SNOWFLAKE_CHANNEL" not in item["env"]
            for item in spec(workers)["spec"]["containers"]
        )
        assert [item["snowflakeSecret"] for item in container["secrets"]] == [
            "app_state.arena_org_key",
            "app_state.arena_external_user_id",
            "app_state.arena_base_url",
        ]
        job = json.loads(session.file.puts[f"@app_state.runs/{run_id}/job.json"])
        assert [s.splitlines()[1] for s in (broker, head, workers)] == [
            f"NAME = app_state.{job['broker_service']}",
            f"NAME = app_state.{job['head_service']}",
            f"NAME = app_state.{job['workers_service']}",
        ]
        assert job["source"] == {
            "kind": "query",
            "sql": "SELECT * FROM reference('SOURCE_TABLE', 'prompts')",
        }
        assert (job["runs_stage"], job["metrics_table"], job["checkpoint_steps"]) == (
            "app_state.runs",
            "app_state.arena_metrics",
            5,
        )
        staged = yaml.safe_load(
            session.file.puts[f"@app_state.runs/{run_id}/manifest.yaml"]
        )
        assert staged["environment"]["dataset"] == "snowflake-query"
        assert text.splitlines()[1:] == [
            f"broker: {job['broker_service']}",
            f"head: {job['head_service']}",
            f"workers: {job['workers_service']}",
        ]

    def test_records_the_run_for_billing_before_any_service_starts(self) -> None:
        session = submit_session()

        text = submit(
            session, SFT, 1, "source_table", "prompts", None, None, None, None
        )

        run_id = text.splitlines()[0].removeprefix("run_id: ")
        insert = session.statements.index(
            "INSERT INTO app_state.usage_items (kind, usage_id, opened_at) "
            f"VALUES ('run', '{run_id}', CURRENT_TIMESTAMP())"
        )
        assert insert < session.statements.index(executes(session)[0])

    def test_resumes_the_billing_task_after_recording_the_run(self) -> None:
        session = submit_session()

        submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        insert = next(i for i, s in enumerate(session.statements) if "usage_items" in s)
        resume = session.statements.index(
            "ALTER TASK app_state.reconcile_usage_task RESUME"
        )
        assert insert < resume < session.statements.index(executes(session)[0])

    def test_a_failed_service_start_still_leaves_the_run_for_billing(self) -> None:
        # Arrange
        session = FailingHeadSession(submit_session().routes)

        # Act
        with pytest.raises(SnowparkSQLException, match="head pool is full"):
            submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        # Assert
        [insert] = [s for s in session.statements if "usage_items" in s]
        [run_id] = re.findall(
            r"VALUES \('run', '([^']+)', CURRENT_TIMESTAMP\(\)\)", insert
        )
        assert f"@app_state.runs/{run_id}/job.json" in session.file.puts

    def test_stages_the_reward_file_beside_the_manifest(self) -> None:
        manifest = yaml.safe_dump(
            {
                "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
                "environment": {
                    "env_type": "rollout",
                    "prompt_template": {"user_0": "{question}"},
                },
                "network": {
                    "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                    "max_context_length": 128,
                },
                "training": {"max_steps": 2},
            }
        )
        reward = "def reward(completion, **row):\n    return 1.0\n"
        session = submit_session()

        text = submit(
            session, manifest, 1, "SOURCE_VIEW", "gsm8k", "GPU_NV_S", reward, None, None
        )

        run_id = text.splitlines()[0].removeprefix("run_id: ")
        assert session.file.puts[f"@app_state.runs/{run_id}/reward.py"] == reward
        job = json.loads(session.file.puts[f"@app_state.runs/{run_id}/job.json"])
        assert job["reward_file"] is True
        assert job["source"]["sql"] == "SELECT * FROM reference('SOURCE_VIEW', 'gsm8k')"

    def test_returns_the_manifest_warnings(self) -> None:
        manifest = yaml.safe_dump(
            {
                "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
                "environment": {
                    "env_type": "rollout",
                    "env_url": "https://env.example.com",
                    "max_turns": 1,
                },
                "network": {
                    "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                    "max_context_length": 128,
                },
                "training": {"max_steps": 2},
            }
        )
        session = submit_session()

        text = submit(session, manifest, 2, None, None, None, None, None, None)

        [warning] = [line for line in text.splitlines() if line.startswith("warning:")]
        assert warning.startswith("warning: env_url must be reachable from the job.")
        assert len(executes(session)) == 3

    def test_quotes_the_alias_as_a_string(self) -> None:
        session = submit_session()

        text = submit(session, SFT, 1, "source_table", "x') --", None, None, None, None)

        run_id = text.splitlines()[0].removeprefix("run_id: ")
        job = json.loads(session.file.puts[f"@app_state.runs/{run_id}/job.json"])
        assert job["source"]["sql"] == (
            "SELECT * FROM reference('SOURCE_TABLE', 'x'') --')"
        )

    @pytest.mark.parametrize(
        ("reference", "alias"),
        [("source_table", None), (None, "prompts"), ("consumer_secret", "prompts")],
    )
    def test_refuses_a_source_that_is_not_a_bound_reference(
        self, reference: str | None, alias: str | None
    ) -> None:
        session = submit_session()

        with pytest.raises(ValueError, match="Pass source_reference"):
            submit(session, SFT, 1, reference, alias, None, None, None, None)

        assert session.statements == []

    def test_refuses_before_set_pools_has_run(self) -> None:
        session = AppSession()

        with pytest.raises(ValueError, match=r"calls app_code\.set_pools first"):
            submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        assert executes(session) == []
        assert session.file.puts == {}
        assert not any("usage_items" in s for s in session.statements)

    def test_refuses_zero_gpus(self) -> None:
        session = submit_session()

        with pytest.raises(ValueError, match="gpus must be at least 1"):
            submit(session, SFT, 0, "source_table", "prompts", None, None, None, None)

        assert session.statements == []

    def test_refuses_more_workers_than_the_pool_holds(self) -> None:
        session = submit_session()

        with pytest.raises(ValueError, match="above MAX_NODES 2"):
            submit(session, SFT, 4, "source_table", "prompts", None, None, None, None)

        assert executes(session) == []

    def test_refuses_a_named_family_arena_has_no_price_for(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        priced_arena.replies = [capabilities(["GPU_NV_M"])]
        session = submit_session()

        # Act
        with pytest.raises(ValueError, match="no Snowflake price for GPU_NV_S") as exc:
            submit(
                session, SFT, 1, "source_table", "prompts", "GPU_NV_S", None, None, None
            )

        # Assert
        assert "Priced families: GPU_NV_M." in str(exc.value)
        assert not any("usage_items" in s for s in session.statements)
        assert not any("EXECUTE JOB SERVICE" in s for s in session.statements)

    def test_refuses_when_no_app_pool_is_priced(self, priced_arena: Arena) -> None:
        # Arrange
        priced_arena.replies = [capabilities([])]
        session = submit_session()

        # Act
        with pytest.raises(
            ValueError, match=r"any of the app's GPU pools \(GPU_NV_S\)"
        ) as exc:
            submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        # Assert
        assert "Priced families: none." in str(exc.value)
        assert not any("EXECUTE JOB SERVICE" in s for s in session.statements)

    def test_an_arena_that_lists_no_prices_still_starts_the_run(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        priced_arena.replies = [capabilities(None)]
        session = submit_session()

        # Act
        text = submit(
            session, SFT, 1, "source_table", "prompts", None, None, None, None
        )

        # Assert
        assert text.startswith("run_id: arena-")
        assert any("EXECUTE JOB SERVICE" in s for s in session.statements)

    def test_runs_on_an_r6k_pool_where_the_region_has_the_aws_gpu_count(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        priced_arena.replies = [capabilities(["GPU_NV_S", "GPU_R6K_G1_48"])]
        region = [
            Row(name="GPU_NV_S", gpu_count=1),
            Row(name="GPU_R6K_G1_48", gpu_count=2),
        ]
        session = submit_session(r6k_setup(), region)

        # Act
        submit(
            session,
            SFT,
            2,
            "source_table",
            "prompts",
            "GPU_R6K_G1_48",
            None,
            None,
            None,
        )

        # Assert
        workers = executes(session)[2]
        assert workers.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_R6K_G1_48\n"
        )

    def test_refuses_a_named_family_the_region_has_with_another_gpu_count(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        priced_arena.replies = [capabilities(["GPU_NV_S", "GPU_R6K_G1_48"])]
        region = [
            Row(name="GPU_NV_S", gpu_count=1),
            Row(name="GPU_R6K_G1_48", gpu_count=1),
        ]
        session = submit_session(r6k_setup(), region)

        # Act
        with pytest.raises(ValueError, match="does not have GPU_R6K_G1_48") as raised:
            submit(
                session,
                SFT,
                2,
                "source_table",
                "prompts",
                "GPU_R6K_G1_48",
                None,
                None,
                None,
            )

        # Assert
        assert str(raised.value) == (
            "This region does not have GPU_R6K_G1_48 with the GPUs per node the app "
            "sized, so the run would ask for GPUs the nodes lack.\n"
            "An app_admin lists usable families with gpu_families() and calls "
            "set_pools with one of them.\n"
            "Nothing was started."
        )
        assert not any("usage_items" in s for s in session.statements)
        assert executes(session) == []

    def test_refuses_when_no_recorded_family_has_the_sized_gpu_count(self) -> None:
        session = submit_session(region=[Row(name="GPU_NV_S")])

        with pytest.raises(ValueError, match="does not have GPU_NV_S with the GPUs"):
            submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        assert executes(session) == []

    def test_reads_the_prices_with_the_partner_key(self, priced_arena: Arena) -> None:
        # Act
        submit(
            submit_session(), SFT, 1, "source_table", "prompts", None, None, None, None
        )

        # Assert
        [call] = priced_arena.calls
        assert call["url"] == "https://arena.test/api/cli/v1/capabilities"
        assert call["headers"]["Authorization"] == "Bearer arena_org_key-1"

    def test_starts_the_run_during_an_unpurchased_trial(self) -> None:
        session = submit_session(install=listing_trial(6.9))

        text = submit(
            session, SFT, 1, "source_table", "prompts", None, None, None, None
        )

        assert text.startswith("run_id: arena-")
        assert len(executes(session)) == 3

    def test_refuses_once_an_unpurchased_trial_has_ended(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        session = submit_session(install=listing_trial(6.98))

        # Act
        with pytest.raises(PermissionError, match=TRIAL_ENDED) as exc:
            submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        # Assert
        assert "Purchase the AgileRL Arena listing" in str(exc.value)
        assert executes(session) == []
        assert not any("usage_items" in s for s in session.statements)
        assert not any(s.startswith("ALTER TASK") for s in session.statements)
        assert session.file.puts == {}
        assert priced_arena.calls == []

    def test_a_purchased_listing_starts_after_the_trial(self) -> None:
        session = submit_session(
            install=Install(
                datetime.now(timezone.utc) - timedelta(days=30), purchased=True
            )
        )

        text = submit(
            session, SFT, 1, "source_table", "prompts", None, None, None, None
        )

        assert text.startswith("run_id: arena-")
        assert len(executes(session)) == 3


class TestStatus:
    def test_reports_each_role_and_the_run_state(self) -> None:
        # Arrange
        services = [
            Row(name=f"{STEM}_BROKER", status="RUNNING"),
            Row(name=f"{STEM}_HEAD", status="RUNNING"),
            Row(name=f"{STEM}_WORKERS", status="PENDING"),
        ]
        container = [Row(instance_id=0, container_name="main", status="READY")]
        session = AppSession(
            [
                known(),
                exact(
                    f"SHOW JOB SERVICES LIKE '{STEM}%' IN SCHEMA app_state", services
                ),
                (lambda s: s.startswith("SHOW SERVICE CONTAINERS"), container),
            ]
        )

        # Act
        text = status(session, RUN_ID)

        # Assert
        assert session.statements == [
            f"LIST {RUN_DIR}/job.json",
            f"SHOW JOB SERVICES LIKE '{STEM}%' IN SCHEMA app_state",
            f"SHOW SERVICE CONTAINERS IN SERVICE app_state.{STEM}_BROKER",
            f"SHOW SERVICE CONTAINERS IN SERVICE app_state.{STEM}_HEAD",
            f"SHOW SERVICE CONTAINERS IN SERVICE app_state.{STEM}_WORKERS",
        ]
        assert text == (
            "broker\trunning\n  0\tmain\tREADY\n"
            "head\trunning\n  0\tmain\tREADY\n"
            "worker\trunning\n  0\tmain\tREADY\n"
            "state: running"
        )

    def test_refuses_a_run_the_app_does_not_have(self) -> None:
        session = AppSession()

        with pytest.raises(ValueError, match=f"This app has no run {RUN_ID}"):
            status(session, RUN_ID)

        assert not any(s.startswith("SHOW") for s in session.statements)

    def test_refuses_a_malformed_run_id_before_listing(self) -> None:
        session = AppSession()

        with pytest.raises(ValueError, match="run_id must look like"):
            status(session, "../setup")

        assert session.statements == []


class TestLogs:
    def test_reads_the_live_logs_of_a_running_role(self) -> None:
        session = AppSession(
            [
                known(),
                exact(
                    f"SHOW JOB SERVICES LIKE '{STEM}%' IN SCHEMA app_state",
                    [Row(name=f"{STEM}_WORKERS", status="RUNNING")],
                ),
                (
                    lambda s: s.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"),
                    [Row(LOGS="step 1 loss 0.5")],
                ),
            ]
        )

        text = logs(session, RUN_ID, "worker", 1)

        assert text == "step 1 loss 0.5"
        assert session.statements[-1] == (
            f"SELECT SYSTEM$GET_SERVICE_LOGS('app_state.{STEM}_WORKERS', 1, 'main')"
        )

    def test_refuses_an_unknown_role_before_any_sql(self) -> None:
        session = AppSession([known()])

        with pytest.raises(ValueError, match="Unknown role trainer"):
            logs(session, RUN_ID, "trainer", 0)

        assert session.statements == []

    def test_refuses_a_run_the_app_does_not_have(self) -> None:
        session = AppSession()

        with pytest.raises(ValueError, match=f"This app has no run {RUN_ID}"):
            logs(session, RUN_ID, "head", 0)


def prefix_route() -> Route:
    return exact(
        "SELECT app_code.object_prefix() AS PREFIX",
        [Row(PREFIX="AGILERL_APP_ARENA_APP_")],
    )


def gpu_pools_route(*pools: Row) -> Route:
    return exact("SHOW COMPUTE POOLS LIKE 'AGILERL_APP_ARENA_APP_GPU_%'", list(pools))


def suspends(session: AppSession) -> list[str]:
    return [s for s in session.statements if s.startswith("ALTER COMPUTE POOL")]


class TestCancel:
    def test_drops_the_three_services_of_the_run(self) -> None:
        session = AppSession([known(), prefix_route()])

        text = cancel(session, RUN_ID)

        assert text == f"cancelled: {RUN_ID}"
        assert [s for s in session.statements if s.startswith("DROP")] == [
            f"DROP SERVICE IF EXISTS app_state.{STEM}_BROKER FORCE",
            f"DROP SERVICE IF EXISTS app_state.{STEM}_HEAD FORCE",
            f"DROP SERVICE IF EXISTS app_state.{STEM}_WORKERS FORCE",
        ]

    def test_suspends_a_gpu_pool_left_with_nothing_on_it_after_the_drops(
        self,
    ) -> None:
        # Arrange
        starting = Row(
            name="AGILERL_APP_ARENA_APP_GPU_GPU_NV_S",
            state="STARTING",
            num_services=0,
            num_jobs=0,
        )
        session = AppSession([known(), prefix_route(), gpu_pools_route(starting)])

        # Act
        text = cancel(session, RUN_ID)

        # Assert
        assert text == (
            f"cancelled: {RUN_ID}\nsuspended: AGILERL_APP_ARENA_APP_GPU_GPU_NV_S"
        )
        assert session.statements[-1] == (
            "ALTER COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_S SUSPEND"
        )

    def test_leaves_a_gpu_pool_another_run_or_model_still_uses(self) -> None:
        serving = Row(
            name="AGILERL_APP_ARENA_APP_GPU_GPU_NV_M",
            state="ACTIVE",
            num_services=1,
            num_jobs=0,
        )
        training = Row(
            name="AGILERL_APP_ARENA_APP_GPU_GPU_NV_S",
            state="ACTIVE",
            num_services=0,
            num_jobs=1,
        )
        session = AppSession(
            [known(), prefix_route(), gpu_pools_route(serving, training)]
        )

        text = cancel(session, RUN_ID)

        assert text == f"cancelled: {RUN_ID}"
        assert suspends(session) == []

    def test_drops_nothing_for_a_run_the_app_does_not_have(self) -> None:
        session = AppSession()

        with pytest.raises(ValueError, match=f"This app has no run {RUN_ID}"):
            cancel(session, RUN_ID)

        assert not any(s.startswith("DROP") for s in session.statements)


def publish_session(
    job_status: str = "DONE", install: Install | None = None
) -> AppSession:
    run = {
        "compute_pool": "AGILERL_APP_ARENA_APP_GPU_GPU_NV_S",
        "image": IMAGE,
        "warehouse": "AGILERL_APP_ARENA_APP_WH",
    }
    listing = [
        Row(name=f"runs/{RUN_ID}/checkpoint_2/merged/model.safetensors", size=2**30),
        Row(name=f"runs/{RUN_ID}/checkpoint_2/merged/.complete", size=0),
    ]
    return AppSession(
        [
            known(),
            setup_route(),
            exact(f"LIST {RUN_DIR}/", listing),
            (lambda s: s.startswith("SELECT CURRENT_DATABASE()"), [CURRENT]),
            (
                lambda s: s.startswith("DESCRIBE COMPUTE POOL"),
                [Row(instance_family="GPU_NV_S", max_nodes=2)],
            ),
            (
                lambda s: s.startswith(
                    "SHOW SERVICE CONTAINERS IN SERVICE app_state.ARENA_PUBLISH_"
                ),
                [Row(service_status=job_status)],
            ),
            (
                lambda s: s.startswith(
                    "SELECT SYSTEM$GET_SERVICE_LOGS('app_state.ARENA_PUBLISH_"
                ),
                [Row(LOGS="out of memory")],
            ),
            exact(
                "SHOW ENDPOINTS IN SERVICE ARENA_APP.app_state.qwen_checkpoint_2",
                [Row(name="inference", ingress_url="abc-app.snowflakecomputing.app")],
            ),
            exact("SELECT CURRENT_ACCOUNT() AS ACCOUNT", [Row(ACCOUNT="AB12345")]),
            (
                lambda s: s.startswith("MERGE INTO app_state.usage_items "),
                [Row(**{"number of rows inserted": 1})],
            ),
        ],
        staged={f"{RUN_DIR}/job.json": json.dumps(run)},
        install=install,
    )


def deployment_opened() -> httpx.Response:
    return reply(201, {"deploymentId": DEPLOYMENT_ID, "billableHours": []})


class TestPublish:
    def test_logs_and_serves_the_model_in_the_app_schema(self, arena: Arena) -> None:
        # Arrange
        session = publish_session()
        arena.replies = [deployment_opened()]

        # Act
        text = publish(session, RUN_ID, "qwen", None, None, True, 1, None, None)

        # Assert
        [request_path] = [p for p in session.file.puts if "/publish-" in p]
        request = json.loads(session.file.puts[request_path])
        assert (request["database"], request["schema"]) == ("ARENA_APP", "app_state")
        assert request["compute_pool"] == "AGILERL_APP_ARENA_APP_GPU_GPU_NV_S"
        assert request["image_build_compute_pool"] == "AGILERL_APP_ARENA_APP_HEAD_POOL"
        assert (request["checkpoint"], request["service_name"]) == (
            "checkpoint_2",
            "qwen_checkpoint_2",
        )
        [job] = executes(session)
        assert job.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_S\n"
        )
        assert "QUERY_WAREHOUSE = AGILERL_APP_ARENA_APP_WH" in job
        assert spec(job)["spec"]["containers"][0]["image"] == IMAGE
        name = job.splitlines()[1].removeprefix("NAME = ")
        assert name.startswith("app_state.ARENA_PUBLISH_")
        deployment_id = text.rsplit("deployment: ", 1)[1]
        assert DEPLOYMENT_ID_RE.fullmatch(deployment_id)
        assert text == (
            f"job: {name}\n"
            "model: ARENA_APP.app_state.qwen\n"
            "version: checkpoint_2\n"
            "service: qwen_checkpoint_2\n"
            "endpoint: abc-app.snowflakecomputing.app\n"
            f"deployment: {deployment_id}"
        )

    def test_opens_the_deployment_after_the_service_starts(self, arena: Arena) -> None:
        # Arrange
        session = publish_session()
        arena.replies = [deployment_opened()]
        before = datetime.now(timezone.utc)

        # Act
        publish(session, RUN_ID, "qwen", None, None, True, 1, None, None)

        # Assert
        [call] = arena.calls
        assert call["url"] == "https://arena.test/api/cli/v1/snowflake/deployments"
        assert call["headers"]["Authorization"] == "Bearer arena_org_key-1"
        body = dict(call["json"])
        started_at = datetime.fromisoformat(str(body.pop("startedAt")))
        deployment_id = str(body.pop("deploymentId"))
        assert body == {
            "snowflakeAccount": "AB12345",
            "gpuCount": 1,
            "gpuFamily": "GPU_NV_S",
            "channel": "native_app",
            "model": "qwen",
            "version": "checkpoint_2",
            "runId": RUN_ID,
        }
        assert started_at >= before.replace(microsecond=0)
        assert deployment_id == (
            f"QWEN_CHECKPOINT_2-{started_at.strftime('%Y%m%dT%H%M%SZ')}"
        )
        [recorded] = [s for s in session.statements if s.startswith("INSERT")]
        assert recorded == (
            "INSERT INTO app_state.served_models (service_name, model, version, "
            "run_id, gpu_family, gpu_count) VALUES ('QWEN_CHECKPOINT_2', 'qwen', "
            f"'checkpoint_2', '{RUN_ID}', 'GPU_NV_S', 1)"
        )
        [insert] = [s for s in session.statements if s.startswith("MERGE")]
        assert insert == (
            "MERGE INTO app_state.usage_items AS t USING (SELECT "
            "'QWEN_CHECKPOINT_2' AS service_name) AS s ON t.kind = 'deployment' "
            "AND t.service_name = s.service_name AND t.closed_at IS NULL AND "
            "t.settled_at IS NULL WHEN NOT MATCHED THEN INSERT (kind, usage_id, "
            "opened_at, service_name) VALUES ('deployment', "
            f"'{deployment_id}', TO_TIMESTAMP_TZ({int(started_at.timestamp())}000, "
            "3), s.service_name)"
        )
        waits = [
            i
            for i, s in enumerate(session.statements)
            if s.startswith("SHOW SERVICE CONTAINERS")
        ]
        assert session.statements.index(recorded) < min(waits)
        assert session.statements.index(insert) > max(waits)
        grant = session.statements.index(
            "GRANT SERVICE ROLE app_state.QWEN_CHECKPOINT_2!ALL_ENDPOINTS_USAGE "
            "TO APPLICATION ROLE app_user"
        )
        assert max(waits) < grant < session.statements.index(insert)
        assert not any(s.startswith("DROP") for s in session.statements)

    def test_a_failed_job_leaves_the_serve_recorded_for_the_hourly_task(
        self, arena: Arena
    ) -> None:
        session = publish_session("FAILED")

        with pytest.raises(RuntimeError, match="failed"):
            publish(session, RUN_ID, "qwen", None, None, True, 1, None, None)

        inserts = [s for s in session.statements if s.startswith("INSERT")]
        assert [s.split(" (", 1)[0] for s in inserts] == [
            "INSERT INTO app_state.served_models"
        ]
        assert arena.calls == []

    @pytest.mark.parametrize("status", [409, 422])
    def test_a_refused_open_drops_the_service_and_keeps_the_item(
        self, arena: Arena, status: int
    ) -> None:
        # Arrange
        session = publish_session()
        arena.replies = [reply(status, {"error": "conflict", "description": INACTIVE})]

        # Act
        with pytest.raises(SnowflakeLicenseError, match=re.escape(INACTIVE)):
            publish(session, RUN_ID, "qwen", None, None, True, 1, None, None)

        # Assert
        insert = next(
            s
            for s in session.statements
            if s.startswith("MERGE INTO app_state.usage_items")
        )
        deployment_id = re.findall(r"'([^']*)'", insert)[3]
        drop = session.statements.index(
            "DROP SERVICE IF EXISTS app_state.QWEN_CHECKPOINT_2"
        )
        assert drop > session.statements.index(insert)
        refusal = (
            "Could not record the Snowflake deployment with Arena. "
            f"Arena answered {status}: {INACTIVE}"
        )
        assert re.fullmatch(
            r"UPDATE app_state\.usage_items SET closed_at = TO_TIMESTAMP_TZ\(\d+, 3\), "
            f"last_error = '{re.escape(refusal)}', "
            r"last_error_at = TO_TIMESTAMP_TZ\(\d+, 3\) "
            f"WHERE kind = 'deployment' AND usage_id = '{deployment_id}'",
            session.statements[drop + 1],
        )

    def test_serving_without_configured_arena_starts_nothing(
        self, arena: Arena, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(SECRETS, "org_key", "")
        session = publish_session()

        with pytest.raises(ValueError, match="Arena is not configured"):
            publish(session, RUN_ID, "qwen", None, None, True, 1, None, None)

        assert executes(session) == []
        assert arena.calls == []

    def test_publishing_without_serving_opens_no_deployment(self) -> None:
        session = publish_session()

        text = publish(session, RUN_ID, "qwen", None, None, False, 1, None, None)

        assert "deployment" not in text
        assert not any(s.startswith("INSERT") for s in session.statements)
        assert not any(s.startswith("ALTER TASK") for s in session.statements)

    def test_serving_resumes_the_billing_task_after_recording_the_serve(
        self, arena: Arena
    ) -> None:
        session = publish_session("FAILED")

        with pytest.raises(RuntimeError, match="failed"):
            publish(session, RUN_ID, "qwen", None, None, True, 1, None, None)

        recorded = next(
            i
            for i, s in enumerate(session.statements)
            if s.startswith("INSERT INTO app_state.served_models")
        )
        assert session.statements[recorded + 1] == (
            "ALTER TASK app_state.reconcile_usage_task RESUME"
        )

    def test_a_failed_job_raises_with_its_log(self) -> None:
        session = publish_session("FAILED")

        with pytest.raises(RuntimeError, match=r"failed:\nout of memory"):
            publish(session, RUN_ID, "qwen", None, None, False, 1, None, None)

    def test_refuses_a_run_the_app_does_not_have(self, arena: Arena) -> None:
        session = AppSession([setup_route()])

        with pytest.raises(ValueError, match=f"This app has no run {RUN_ID}"):
            publish(session, RUN_ID, "qwen", None, None, True, 1, None, None)

        assert executes(session) == []
        assert session.file.puts == {}
        assert arena.calls == []

    def test_serves_during_an_unpurchased_trial(self, arena: Arena) -> None:
        session = publish_session(install=listing_trial(1))
        arena.replies = [deployment_opened()]

        text = publish(session, RUN_ID, "qwen", None, None, True, 1, None, None)

        assert "service: qwen_checkpoint_2" in text
        assert len(arena.calls) == 1

    def test_refuses_once_an_unpurchased_trial_has_ended(self, arena: Arena) -> None:
        session = publish_session(install=listing_trial(6.98))

        with pytest.raises(PermissionError, match=TRIAL_ENDED):
            publish(session, RUN_ID, "qwen", None, None, True, 1, None, None)

        assert executes(session) == []
        assert session.file.puts == {}
        assert not any(s.startswith("INSERT") for s in session.statements)
        assert not any(s.startswith("ALTER TASK") for s in session.statements)
        assert arena.calls == []


def served_session(services: list[str]) -> AppSession:
    model = "ARENA_APP.app_state.qwen"
    return AppSession(
        [
            (lambda s: s.startswith("SELECT CURRENT_DATABASE()"), [CURRENT]),
            exact(
                "SHOW MODELS LIKE 'qwen' IN SCHEMA ARENA_APP.app_state",
                [Row(name="QWEN")],
            ),
            exact(
                f"SHOW VERSIONS IN MODEL {model}",
                [
                    Row(
                        name="CHECKPOINT_2",
                        inference_services=json.dumps(services),
                        is_default_version=True,
                    )
                ],
            ),
            exact(
                "SHOW ENDPOINTS IN SERVICE ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2",
                [Row(name="inference", ingress_url="abc-app.snowflakecomputing.app")],
            ),
        ]
    )


class TestEndpoint:
    def test_returns_the_ingress_url_of_the_served_version(self) -> None:
        session = served_session(["ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2"])

        text = endpoint(session, "qwen", None)

        assert text == "abc-app.snowflakecomputing.app"

    def test_refuses_a_model_the_app_does_not_have(self) -> None:
        session = served_session([])

        with pytest.raises(
            ValueError, match=r"Model ARENA_APP\.app_state\.llama does not exist"
        ):
            endpoint(session, "llama", None)


class TestUnpublish:
    def test_drops_the_service_and_the_only_version_with_the_model(self) -> None:
        session = served_session(["ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2"])

        text = unpublish(session, "qwen", None, True)

        assert [s for s in session.statements if s.startswith("DROP")] == [
            "DROP SERVICE ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2",
            "DROP MODEL ARENA_APP.app_state.qwen",
        ]
        assert text == (
            "dropped service: ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2\n"
            "dropped model: ARENA_APP.app_state.qwen"
        )

    def test_closes_the_deployment_of_the_dropped_service(self, arena: Arena) -> None:
        # Arrange
        session = served_session(["ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2"])
        session.routes.append(
            exact(
                "SELECT usage_id FROM app_state.usage_items WHERE kind = 'deployment' "
                "AND settled_at IS NULL AND closed_at IS NULL "
                "AND service_name IN ('QWEN_CHECKPOINT_2')",
                [Row(USAGE_ID=DEPLOYMENT_ID)],
            )
        )
        arena.replies = [reply(200, {"billableHours": []})]
        before = datetime.now(timezone.utc)

        # Act
        text = unpublish(session, "qwen", None, False)

        # Assert
        assert text == (
            "dropped service: ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2\n"
            f"closed deployment: {DEPLOYMENT_ID}"
        )
        [call] = arena.calls
        assert call["url"] == (
            f"https://arena.test/api/cli/v1/snowflake/deployments/{DEPLOYMENT_ID}/close"
        )
        assert datetime.fromisoformat(str(call["json"]["endedAt"])) >= before
        drop = session.statements.index(
            "DROP SERVICE ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2"
        )
        assert session.statements[drop + 1] == (
            "DELETE FROM app_state.served_models "
            "WHERE service_name IN ('QWEN_CHECKPOINT_2')"
        )
        assert session.statements[-1].startswith(
            "UPDATE app_state.usage_items SET closed_at = TO_TIMESTAMP_TZ("
        )
        assert session.statements[-1].endswith(f"usage_id = '{DEPLOYMENT_ID}'")

    def test_a_deployment_arena_does_not_know_is_logged_and_left(
        self, arena: Arena, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = served_session(["ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2"])
        session.routes.append(
            (
                lambda s: s.startswith("SELECT usage_id FROM app_state.usage_items"),
                [Row(USAGE_ID=DEPLOYMENT_ID)],
            )
        )
        arena.replies = [reply(404)]

        text = unpublish(session, "qwen", None, False)

        assert text == "dropped service: ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2"
        assert f"Arena has no deployment {DEPLOYMENT_ID} to close" in caplog.text

    def test_refuses_a_version_with_nothing_to_drop(self) -> None:
        session = served_session([])

        with pytest.raises(ValueError, match="has no inference service"):
            unpublish(session, "qwen", "checkpoint_2", False)

        assert not any(s.startswith("DROP") for s in session.statements)


class TestRunProcedureStatements:
    def test_no_run_procedure_runs_use(self, arena: Arena) -> None:
        # Arrange
        arena.replies = [capabilities(["GPU_NV_S"]), deployment_opened()]
        running = Row(name=f"{STEM}_HEAD", status="RUNNING")
        run_session = AppSession(
            [
                known(),
                exact(
                    f"SHOW JOB SERVICES LIKE '{STEM}%' IN SCHEMA app_state", [running]
                ),
                prefix_route(),
            ]
        )
        sessions = [
            submit_session(),
            run_session,
            publish_session(),
            served_session(["ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2"]),
        ]

        # Act
        submit(sessions[0], SFT, 1, "source_table", "prompts", None, None, None, None)
        status(run_session, RUN_ID)
        logs(run_session, RUN_ID, "head", 0)
        cancel(run_session, RUN_ID)
        publish(sessions[2], RUN_ID, "qwen", None, None, True, 1, None, None)
        endpoint(sessions[3], "qwen", None)
        unpublish(sessions[3], "qwen", None, True)

        # Assert
        statements = [s for session in sessions for s in session.statements]
        assert statements
        assert [s for s in statements if s.lstrip().upper().startswith("USE ")] == []
