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

from agilerl.models.manifest import TrainingManifest
from agilerl.snowflake import license as license_mod
from agilerl.snowflake.job import ServiceLeftRunningError
from agilerl.snowflake.license import SnowflakeLicenseError
from snowflake_app.app_code.arena_app import (
    IMAGE,
    cancel,
    endpoint,
    export_model,
    logs,
    publish,
    refuse_remote_code,
    status,
    submit,
    unpublish,
)
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
MIMO = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
REWARD = "def reward(completion, **row):\n    return 1.0\n"
REWARD_REFS = "SELECT SYSTEM$GET_ALL_REFERENCES('REWARD_FUNCTION') AS REFS"


def grpo_manifest(hub_id: str, reward: dict | None = None) -> str:
    """A GRPO rollout over source rows that trains ``hub_id``, scored by ``reward``."""
    return yaml.safe_dump(
        {
            "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
            "environment": {
                "env_type": "rollout",
                "prompt_template": {"user_0": "{question}"},
                "reward": reward or {"type": "exact_match"},
            },
            "network": {
                "pretrained_model_name_or_path": hub_id,
                "max_context_length": 128,
            },
            "training": {"max_steps": 2},
        }
    )


def engine_args_manifest(hub_id: str, trust_remote_code: bool) -> str:
    """``grpo_manifest`` with ``vllm_engine_args.trust_remote_code`` set."""
    document = yaml.safe_load(grpo_manifest(hub_id))
    document["algorithm"]["vllm_engine_args"] = {"trust_remote_code": trust_remote_code}
    return yaml.safe_dump(document)


def opd_manifest(teacher_id: str) -> TrainingManifest:
    """An OPD run of Qwen2.5-0.5B scored by ``teacher_id``."""
    return TrainingManifest.model_validate(
        {
            "algorithm": {
                "name": "OPD",
                "batch_size": 2,
                "vllm_config": {"tensor_parallel_size": 1},
                "teacher": {"model": {"pretrained_model_name_or_path": teacher_id}},
            },
            "environment": {
                "env_type": "rollout",
                "dataset": "openai/gsm8k",
                "reward_file_path": "reward.py",
                "prompt_template": {"user_0": "{question}"},
            },
            "network": {
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_context_length": 512,
            },
            "training": {"max_steps": 100, "rollout_engines_per_agent": 1},
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
INSTALL_ROW = Row(
    REGION="AWS_EU_WEST_2",
    ACCOUNT="AB12345",
    ACCOUNT_NAME="ACME",
    ORGANIZATION_NAME="ACMEORG",
    APPLICATION_NAME="ARENA_APP",
    PURCHASED=True,
)


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
        if statement.startswith("SELECT CURRENT_REGION() AS REGION, CURRENT_ACCOUNT()"):
            return Result([INSTALL_ROW])
        for pred, rows in self.routes:
            if pred(statement):
                return Result(rows)
        return Result([])


class NodeLimitSession(AppSession):
    """Procedure session where the account's node limit refuses the worker service."""

    def sql(self, statement: str) -> Result:
        if statement.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_"
        ):
            self.statements.append(statement)
            msg = "Requested number of nodes 1 exceeds the node limit for the account."
            raise SnowparkSQLException(msg, sql_error_code=395034)
        return super().sql(statement)


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


class FailingHeadAndDropSession(FailingHeadSession):
    """Procedure session where the head fails to start and the broker fails to drop."""

    def sql(self, statement: str) -> Result:
        if statement.startswith("DROP SERVICE"):
            self.statements.append(statement)
            msg = "drop timed out"
            raise SnowparkSQLException(msg)
        return super().sql(statement)


class FailingLogsSession(AppSession):
    """Procedure session where reading live service logs fails."""

    def sql(self, statement: str) -> Result:
        if statement.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"):
            self.statements.append(statement)
            msg = "logs are unavailable"
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


def source_refs(reference: str, aliases: list[str]) -> Route:
    """``SYSTEM$GET_ALL_REFERENCES`` on ``reference`` answering ``aliases``."""
    return exact(
        f"SELECT SYSTEM$GET_ALL_REFERENCES('{reference}') AS REFS",
        [Row(REFS=json.dumps(aliases))],
    )


def submit_session(
    setup: dict | None = None,
    region: list[Row] | None = None,
    install: Install | None = None,
) -> AppSession:
    return AppSession(
        [
            source_refs("SOURCE_TABLE", ["prompts", "x') --"]),
            source_refs("SOURCE_VIEW", ["gsm8k"]),
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
                    "DESCRIBE COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_M"
                ),
                [Row(instance_family="GPU_NV_M", max_nodes=2)],
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


def nv_m_setup(max_gpus: int = 8) -> dict:
    """``SETUP`` with only a GPU_NV_M pool: 4 GPUs per node, at most ``max_gpus``."""
    nv_m = {
        "pool": "AGILERL_APP_ARENA_APP_GPU_GPU_NV_M",
        "max_gpus": max_gpus,
        "gpus_per_node": 4,
        "gpu_memory_gib": 24,
    }
    return {**SETUP, "pools": {**SETUP["pools"], "gpu": {"GPU_NV_M": nv_m}}}


def capabilities(priced: list[str] | None) -> httpx.Response:
    """Arena capabilities with ``priced`` as the priced families, or none listed."""
    features: dict[str, object] = {"snowflakeTraining": True}
    if priced is not None:
        features["snowflakePricedGpuFamilies"] = priced
    return reply(200, {"ok": True, "data": {"schemaVersion": 1, "features": features}})


FITS = reply(200, {"exit_code": 0, "report": {"fits": True}})
# Arena's estimate for an RL run on one GPU: generation needs a second one.
SHORT_OF_GPUS = reply(
    200, {"exit_code": 2, "report": {"error": "1 GPUs, the job needs 2"}}
)


class TestSubmit:
    @pytest.fixture(autouse=True)
    def priced_arena(self, arena: Arena) -> Arena:
        arena.replies = [capabilities(["GPU_NV_S"]), FITS]
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
        assert container["env"]["ARENA_SNOWFLAKE_REGION"] == "AWS_EU_WEST_2"
        assert container["env"]["ARENA_SNOWFLAKE_PURCHASED"] == "true"
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

    def test_no_service_the_run_starts_has_a_public_endpoint(self) -> None:
        # Arrange
        session = submit_session()

        # Act
        submit(session, SFT, 1, "source_table", "prompts", None, None, 5, None)

        # Assert
        endpoints = [
            endpoint
            for statement in executes(session)
            for endpoint in spec(statement)["spec"].get("endpoints", [])
        ]
        assert endpoints
        assert all(endpoint["public"] is False for endpoint in endpoints)

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

    def test_a_failed_service_start_drops_the_started_services_and_the_usage_item(
        self,
    ) -> None:
        # Arrange
        session = FailingHeadSession(submit_session().routes)

        # Act
        with pytest.raises(SnowparkSQLException, match="head pool is full"):
            submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        # Assert
        insert, delete = [s for s in session.statements if "usage_items" in s]
        [run_id] = re.findall(
            r"VALUES \('run', '([^']+)', CURRENT_TIMESTAMP\(\)\)", insert
        )
        assert delete == (
            "DELETE FROM app_state.usage_items WHERE kind = 'run' "
            f"AND usage_id = '{run_id}'"
        )
        [drop] = [s for s in session.statements if s.startswith("DROP SERVICE")]
        assert re.fullmatch(r"DROP SERVICE app_state\.ARENA_RUN_\w+_BROKER FORCE", drop)

    def test_a_service_that_cannot_be_dropped_keeps_its_usage_item(self) -> None:
        # Arrange
        session = FailingHeadAndDropSession(submit_session().routes)

        # Act
        with pytest.raises(
            ServiceLeftRunningError, match=r"_BROKER started but could not be dropped"
        ):
            submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        # Assert
        assert not any(
            s.startswith("DELETE FROM app_state.usage_items")
            for s in session.statements
        )

    def test_the_account_node_limit_is_a_plain_refusal(self) -> None:
        # Arrange
        session = NodeLimitSession(submit_session().routes)

        # Act
        with pytest.raises(ValueError, match="node limit") as exc:
            submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        # Assert
        assert str(exc.value) == (
            "Your Snowflake account's node limit is reached; wait for running "
            "jobs or ask Snowflake to raise it.\nNothing was started."
        )
        assert len([s for s in session.statements if s.startswith("DROP SERVICE")]) == 2
        assert any(
            s.startswith("DELETE FROM app_state.usage_items")
            for s in session.statements
        )

    def test_stages_a_built_in_reward_in_the_manifest(self) -> None:
        manifest = grpo_manifest("Qwen/Qwen3.5-0.8B", {"type": "numeric"})
        session = submit_session()

        text = submit(
            session, manifest, 2, "SOURCE_VIEW", "gsm8k", "GPU_NV_S", None, None, None
        )

        run_id = text.splitlines()[0].removeprefix("run_id: ")
        assert f"@app_state.runs/{run_id}/reward.py" not in session.file.puts
        job = json.loads(session.file.puts[f"@app_state.runs/{run_id}/job.json"])
        assert job["reward_file"] is False
        staged = yaml.safe_load(
            session.file.puts[f"@app_state.runs/{run_id}/manifest.yaml"]
        )
        assert staged["environment"]["reward"] == {"type": "numeric"}
        assert "rubric_file_path" not in staged["environment"]

    def test_refuses_a_reward_file_and_names_both_reward_options(self) -> None:
        session = submit_session()

        with pytest.raises(ValueError, match="reward_file is not available") as exc:
            submit(
                session,
                grpo_manifest("Qwen/Qwen3.5-0.8B"),
                1,
                "SOURCE_VIEW",
                "gsm8k",
                None,
                REWARD,
                None,
                None,
            )

        message = str(exc.value)
        assert message.startswith("reward_file is not available in the Snowflake app.")
        assert "built-in reward" in message
        assert "reward_function reference" in message
        assert message.endswith("Nothing was started.")
        assert session.statements == []
        assert session.file.puts == {}

    def test_refuses_an_invalid_manifest_in_plain_words(self) -> None:
        document = yaml.safe_load(grpo_manifest("Qwen/Qwen3.5-0.8B"))
        document["environment"]["reward"] = {"type": "llm_judge"}
        session = submit_session()

        with pytest.raises(ValueError, match="Nothing was started") as exc:
            submit(
                session,
                yaml.safe_dump(document),
                1,
                "SOURCE_VIEW",
                "gsm8k",
                None,
                None,
                None,
                None,
            )

        message = str(exc.value)
        assert message.splitlines()[1].startswith(
            "environment.reward.type is 'llm_judge', which is not one of the options."
        )
        assert message.endswith("Nothing was started.")
        assert "validation error" not in message
        assert "pydantic" not in message
        assert session.statements == []
        assert session.file.puts == {}

    def test_refuses_a_rubric_file_in_the_manifest(self) -> None:
        document = yaml.safe_load(grpo_manifest("Qwen/Qwen3.5-0.8B"))
        del document["environment"]["reward"]
        document["environment"]["rubric_file_path"] = "/opt/rewards/reward.py"
        session = submit_session()

        with pytest.raises(
            ValueError, match="rubric_file_path is not available in the Snowflake app"
        ):
            submit(
                session,
                yaml.safe_dump(document),
                1,
                "SOURCE_VIEW",
                "gsm8k",
                None,
                None,
                None,
                None,
            )

        assert executes(session) == []
        assert session.file.puts == {}

    def test_refuses_a_function_reward_with_nothing_bound(self) -> None:
        manifest = grpo_manifest("Qwen/Qwen3.5-0.8B", {"type": "snowflake_function"})
        session = submit_session()
        session.routes.append(exact(REWARD_REFS, [Row(REFS="[]")]))

        with pytest.raises(ValueError, match="no function is bound"):
            submit(session, manifest, 1, "SOURCE_VIEW", "gsm8k", None, None, None, None)

        assert session.statements == [REWARD_REFS]
        assert session.file.puts == {}

    def test_starts_a_function_reward_once_a_function_is_bound(self) -> None:
        manifest = grpo_manifest("Qwen/Qwen3.5-0.8B", {"type": "snowflake_function"})
        session = submit_session()
        session.routes.append(exact(REWARD_REFS, [Row(REFS='["ref-1"]')]))

        text = submit(
            session, manifest, 2, "SOURCE_VIEW", "gsm8k", "GPU_NV_S", None, None, None
        )

        run_id = text.splitlines()[0].removeprefix("run_id: ")
        staged = yaml.safe_load(
            session.file.puts[f"@app_state.runs/{run_id}/manifest.yaml"]
        )
        assert staged["environment"]["reward"] == {"type": "snowflake_function"}
        assert len(executes(session)) == 3

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

    def test_refuses_an_alias_not_bound_to_the_reference(self) -> None:
        # Arrange
        session = submit_session()

        # Act
        with pytest.raises(ValueError, match="Nothing is bound to source_table") as exc:
            submit(session, SFT, 1, "source_table", "0000", None, None, None, None)

        # Assert
        assert str(exc.value) == (
            "Nothing is bound to source_table with the alias 0000.\n"
            "Aliases bound to source_table: prompts, x') --.\n"
            "To bind a table, call\n"
            "CALL <app>.app_code.register_reference('source_table', 'ADD', "
            "SYSTEM$REFERENCE('TABLE', '<database>.<schema>.<name>', 'PERSISTENT', "
            "'SELECT'));\n"
            "then pass the alias SHOW REFERENCES IN APPLICATION <app> lists.\n"
            "Nothing was started."
        )
        assert executes(session) == []
        assert session.file.puts == {}
        assert not any("usage_items" in s for s in session.statements)

    def test_refuses_an_alias_bound_to_the_other_source_reference(self) -> None:
        session = submit_session()

        with pytest.raises(
            ValueError, match="Aliases bound to source_view: gsm8k"
        ) as exc:
            submit(session, SFT, 1, "source_view", "prompts", None, None, None, None)

        assert "To bind a view, call" in str(exc.value)
        assert "SYSTEM$REFERENCE('VIEW', " in str(exc.value)
        assert executes(session) == []

    def test_says_none_when_nothing_is_bound(self) -> None:
        session = AppSession([source_refs("SOURCE_TABLE", [])])

        with pytest.raises(ValueError, match=r"Aliases bound to source_table: none\."):
            submit(session, SFT, 1, "source_table", "prompts", None, None, None, None)

        assert executes(session) == []

    def test_refuses_before_set_pools_has_run(self) -> None:
        session = AppSession([source_refs("SOURCE_TABLE", ["prompts"])])

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

    def test_refuses_more_gpus_than_the_pool_holds_before_recording_the_run(
        self,
    ) -> None:
        session = submit_session()

        with pytest.raises(ValueError, match="No GPU pool can run 4 GPUs") as exc:
            submit(session, SFT, 4, "source_table", "prompts", None, None, None, None)

        assert "Valid GPU counts on GPU_NV_S: 1, 2 " in str(exc.value)
        assert str(exc.value).endswith("Nothing was started.")
        assert executes(session) == []
        assert not any("usage_items" in s for s in session.statements)

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
        priced_arena.replies = [capabilities(None), FITS]
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
        priced_arena.replies = [capabilities(["GPU_NV_S", "GPU_R6K_G1_48"]), FITS]
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
        priced_arena.replies = [capabilities(["GPU_NV_S", "GPU_R6K_G1_48"]), FITS]
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

    def test_refuses_a_family_with_no_pool_and_gives_the_set_pools_call(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        session = submit_session()

        # Act
        with pytest.raises(ValueError, match="no GPU pool for GPU_NV_L") as exc:
            submit(
                session, SFT, 1, "source_table", "prompts", "GPU_NV_L", None, None, None
            )

        # Assert
        assert str(exc.value) == (
            "The app has no GPU pool for GPU_NV_L. GPU pools it has: GPU_NV_S.\n"
            "An app_admin adds the pool, keeping the ones it has, with\n"
            "CALL <app>.app_code.set_pools(['GPU_NV_S', 'GPU_NV_L']);\n"
            "Nothing was started."
        )
        assert priced_arena.calls == []
        assert executes(session) == []
        assert not any("usage_items" in s for s in session.statements)

    def test_reads_a_lower_case_family_as_the_pool_family(self) -> None:
        session = submit_session()

        submit(
            session, SFT, 1, "source_table", "prompts", " gpu_nv_s ", None, None, None
        )

        assert executes(session)[2].startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_S\n"
        )

    def test_refuses_a_gpu_count_the_family_does_not_divide_and_lists_valid_ones(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        priced_arena.replies = [capabilities(["GPU_NV_M"])]
        session = submit_session(nv_m_setup(), [Row(name="GPU_NV_M", gpu_count=4)])

        # Act
        with pytest.raises(ValueError, match="cannot run 3 GPUs") as exc:
            submit(
                session, SFT, 3, "source_table", "prompts", "GPU_NV_M", None, None, None
            )

        # Assert
        assert str(exc.value) == (
            "GPU_NV_M has 4 GPUs per node, so it cannot run 3 GPUs.\n"
            "Valid GPU counts on GPU_NV_M: 4, 8 (multiples of 4, up to the "
            "8 GPUs the pool holds).\n"
            "Nothing was started."
        )
        assert executes(session) == []
        assert not any("usage_items" in s for s in session.statements)

    def test_refuses_more_gpus_than_the_named_pool_holds(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        priced_arena.replies = [capabilities(["GPU_NV_M"])]
        session = submit_session(nv_m_setup(), [Row(name="GPU_NV_M", gpu_count=4)])

        # Act
        with pytest.raises(ValueError, match="holds 8 GPUs") as exc:
            submit(
                session,
                SFT,
                12,
                "source_table",
                "prompts",
                "GPU_NV_M",
                None,
                None,
                None,
            )

        # Assert
        assert str(exc.value) == (
            "The GPU_NV_M pool holds 8 GPUs, so it cannot run 12 GPUs.\n"
            "Valid GPU counts on GPU_NV_M: 4, 8 (multiples of 4, up to the "
            "8 GPUs the pool holds).\n"
            "To run more GPUs, raise the pool's maximum GPUs first.\n"
            "Nothing was started."
        )
        assert not any("usage_items" in s for s in session.statements)

    def test_runs_on_every_gpu_of_a_max_between_whole_nodes(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        priced_arena.replies = [capabilities(["GPU_NV_M"]), FITS]
        session = submit_session(nv_m_setup(6), [Row(name="GPU_NV_M", gpu_count=4)])

        # Act
        submit(session, SFT, 8, "source_table", "prompts", "GPU_NV_M", None, None, None)

        # Assert
        assert executes(session)[2].startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_GPU_GPU_NV_M\n"
        )

    def test_refuses_an_rl_run_on_one_gpu_instead_of_starting_it_unchecked(
        self, priced_arena: Arena
    ) -> None:
        # Arrange
        priced_arena.replies = [capabilities(["GPU_NV_S"]), SHORT_OF_GPUS]
        session = submit_session()
        manifest = grpo_manifest("Qwen/Qwen3.5-0.8B")

        # Act
        with pytest.raises(ValueError, match="GRPO needs at least 2 GPUs") as exc:
            submit(
                session,
                manifest,
                1,
                "SOURCE_VIEW",
                "gsm8k",
                "GPU_NV_S",
                None,
                None,
                None,
            )

        # Assert
        assert str(exc.value) == (
            "GRPO needs at least 2 GPUs on GPU_NV_S: training and rollout "
            "generation run on separate GPUs.\n"
            "Ask for 2 GPUs or more.\n"
            "Nothing was started."
        )
        assert executes(session) == []
        assert not any("usage_items" in s for s in session.statements)

    def test_refuses_an_rl_run_on_one_gpu_when_the_family_is_picked(
        self, priced_arena: Arena
    ) -> None:
        priced_arena.replies = [capabilities(["GPU_NV_S"]), SHORT_OF_GPUS]
        session = submit_session()

        with pytest.raises(ValueError, match="GRPO needs at least 2 GPUs on GPU_NV_S"):
            submit(
                session,
                grpo_manifest("Qwen/Qwen3.5-0.8B"),
                1,
                "SOURCE_VIEW",
                "gsm8k",
                None,
                None,
                None,
                None,
            )

        assert executes(session) == []

    def test_reads_prices_and_sizes_the_manifest_with_the_partner_key(
        self, priced_arena: Arena
    ) -> None:
        # Act
        submit(
            submit_session(), SFT, 1, "source_table", "prompts", None, None, None, None
        )

        # Assert
        prices, estimate = priced_arena.calls
        assert prices["url"] == "https://arena.test/api/cli/v1/capabilities"
        assert estimate["url"] == "https://arena.test/api/cli/v1/memory/estimate"
        assert estimate["json"]["num_gpus"] == 1
        assert {call["headers"]["Authorization"] for call in priced_arena.calls} == {
            "Bearer arena_org_key-1"
        }

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

    def test_refuses_a_model_that_runs_remote_code_before_any_sql(self) -> None:
        # Arrange
        session = submit_session()

        # Act
        with pytest.raises(
            ValueError, match="not available in the Snowflake app"
        ) as exc:
            submit(
                session,
                grpo_manifest(MIMO),
                1,
                "SOURCE_VIEW",
                "gsm8k",
                None,
                None,
                None,
                None,
            )

        # Assert
        assert str(exc.value).startswith(f"{MIMO} is not available")
        assert str(exc.value).endswith("Nothing was started.")
        assert session.statements == []
        assert session.file.puts == {}

    def test_refuses_a_model_outside_the_catalog(self) -> None:
        session = submit_session()

        with pytest.raises(
            ValueError, match="acme/custom-model is not in the Arena model catalog"
        ):
            submit(
                session,
                grpo_manifest("acme/custom-model"),
                1,
                "SOURCE_VIEW",
                "gsm8k",
                None,
                None,
                None,
                None,
            )

        assert session.statements == []

    def test_refuses_trust_remote_code_in_the_engine_args_before_any_sql(
        self,
    ) -> None:
        # Arrange
        session = submit_session()
        manifest = engine_args_manifest("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", True)

        # Act
        with pytest.raises(
            ValueError,
            match=r"vllm_engine_args\.trust_remote_code is not available",
        ):
            submit(session, manifest, 1, "SOURCE_VIEW", "gsm8k", None, None, None, None)

        # Assert
        assert session.statements == []
        assert session.file.puts == {}

    def test_engine_args_trust_remote_code_false_still_refuses_mimo(self) -> None:
        session = submit_session()

        with pytest.raises(ValueError, match=f"{MIMO} is not available"):
            submit(
                session,
                engine_args_manifest(MIMO, False),
                1,
                "SOURCE_VIEW",
                "gsm8k",
                None,
                None,
                None,
                None,
            )

        assert session.statements == []
        assert session.file.puts == {}


class TestRefuseRemoteCode:
    def test_accepts_a_catalog_teacher_without_remote_code(self) -> None:
        refuse_remote_code(opd_manifest("Qwen/Qwen3-4B"))

    def test_accepts_trust_remote_code_false_on_a_catalog_model(self) -> None:
        manifest = opd_manifest("Qwen/Qwen3-4B")
        manifest.algorithm.vllm_engine_args["trust_remote_code"] = False

        refuse_remote_code(manifest)

    def test_refuses_a_teacher_whose_family_runs_remote_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(
            "snowflake_app.app_code.arena_app.REMOTE_CODE_MODEL_TYPES",
            frozenset({"qwen3"}),
        )
        manifest = opd_manifest("Qwen/Qwen3-4B")

        # Act / Assert
        with pytest.raises(
            ValueError, match="Qwen/Qwen3-4B is not available in the Snowflake app"
        ):
            refuse_remote_code(manifest)


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


def logs_session(
    status: str, containers: list[Row], live: list[Row] | None = None
) -> AppSession:
    """A run whose workers service has ``status`` and ``containers``, and whose live logs are ``live``."""
    return AppSession(
        [
            known(),
            exact(
                f"SHOW JOB SERVICES LIKE '{STEM}%' IN SCHEMA app_state",
                [Row(name=f"{STEM}_WORKERS", status=status)],
            ),
            exact(
                f"SHOW SERVICE CONTAINERS IN SERVICE app_state.{STEM}_WORKERS",
                containers,
            ),
            (lambda s: s.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"), live or []),
        ]
    )


class TestLogs:
    def test_reads_the_live_logs_of_a_running_role(self) -> None:
        session = logs_session(
            "RUNNING",
            [Row(instance_id=1, container_name="main", status="READY")],
            [Row(LOGS="step 1 loss 0.5")],
        )

        text = logs(session, RUN_ID, "worker", 1)

        assert text == "step 1 loss 0.5"
        assert session.statements[-1] == (
            f"SELECT SYSTEM$GET_SERVICE_LOGS('app_state.{STEM}_WORKERS', 1, 'main')"
        )

    @pytest.mark.parametrize(
        "containers",
        [
            [Row(instance_id=1, container_name="main", status="PENDING")],
            [Row(instance_id=None, container_name="main", status="PENDING")],
            [],
        ],
    )
    def test_says_a_container_that_has_not_started_is_pending(
        self, containers: list[Row]
    ) -> None:
        session = logs_session("PENDING", containers)

        text = logs(session, RUN_ID, "worker", 1)

        assert text == (
            "The worker container has not started yet (state: PENDING). "
            "Its node can take several minutes to start; GPU nodes take longest. "
            f"Check progress with CALL <app>.app_code.status('{RUN_ID}')."
        )
        assert not any("SYSTEM$GET_SERVICE_LOGS" in s for s in session.statements)

    def test_says_a_started_container_has_no_logs_yet(self) -> None:
        session = logs_session(
            "RUNNING",
            [Row(instance_id=0, container_name="main", status="READY")],
            [Row(LOGS="")],
        )

        text = logs(session, RUN_ID, "worker", 0)

        assert text == (
            "The worker container has started but has written no logs yet. "
            "Check again in a minute."
        )

    def test_a_log_read_error_on_a_started_container_is_raised(self) -> None:
        # Arrange
        started = logs_session(
            "RUNNING", [Row(instance_id=0, container_name="main", status="READY")]
        )
        session = FailingLogsSession(started.routes)

        # Act / Assert
        with pytest.raises(SnowparkSQLException, match="logs are unavailable"):
            logs(session, RUN_ID, "worker", 0)

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
            exact(
                "DESCRIBE COMPUTE POOL AGILERL_APP_ARENA_APP_HEAD_POOL",
                [Row(instance_family="CPU_X64_SL", max_nodes=4)],
            ),
            (
                lambda s: s.startswith("DESCRIBE COMPUTE POOL"),
                [Row(instance_family="GPU_NV_S", max_nodes=1)],
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
                (
                    "SELECT CURRENT_ACCOUNT() AS ACCOUNT, CURRENT_REGION() AS REGION, "
                    "app_code.listing_purchased() AS PURCHASED"
                ),
                [Row(ACCOUNT="AB12345", REGION="AWS_EU_WEST_2", PURCHASED=True)],
            ),
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
        assert request["ingress"] is False
        [job] = executes(session)
        assert job.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL AGILERL_APP_ARENA_APP_HEAD_POOL\n"
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
            "function: ARENA_APP.app_state.QWEN_CHECKPOINT_2_CHAT\n"
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
            "snowflakeRegion": "AWS_EU_WEST_2",
            "gpuCount": 1,
            "gpuFamily": "GPU_NV_S",
            "channel": "native_app",
            "model": "qwen",
            "version": "checkpoint_2",
            "purchased": True,
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
        create = session.statements.index(
            "CREATE FUNCTION IF NOT EXISTS app_state.QWEN_CHECKPOINT_2_CHAT"
            "(messages ARRAY, temperature FLOAT, max_completion_tokens INT)\n"
            "RETURNS OBJECT\n"
            "AS $$SELECT app_state.QWEN_CHECKPOINT_2!__CALL__(messages, temperature, "
            "max_completion_tokens, NULL, 1, FALSE, 1.0, 0.0, 0.0, NULL)$$"
        )
        grant = session.statements.index(
            "GRANT USAGE ON FUNCTION app_state.QWEN_CHECKPOINT_2_CHAT"
            "(ARRAY, FLOAT, INT) TO APPLICATION ROLE app_user"
        )
        service_role = session.statements.index(
            "GRANT SERVICE ROLE app_state.QWEN_CHECKPOINT_2!"
            "INFERENCE_SERVICE_FUNCTION_USAGE TO APPLICATION ROLE app_user"
        )
        assert max(waits) < create < grant < service_role
        assert service_role < session.statements.index(insert)
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
        assert session.statements[drop - 1] == (
            "DROP FUNCTION IF EXISTS app_state.QWEN_CHECKPOINT_2_CHAT(ARRAY, FLOAT, INT)"
        )
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

        with pytest.raises(ValueError, match="Arena is not connected"):
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
        ]
    )


class TestEndpoint:
    def test_returns_the_chat_function_and_a_call_to_run(self) -> None:
        session = served_session(["ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2"])

        text = endpoint(session, "qwen", None)

        assert text == (
            "function: ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2_CHAT\n"
            "example:\n"
            "SELECT ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2_CHAT(\n"
            """  PARSE_JSON('[{"role": "user", "content": "Hello"}]'),\n"""
            "  0.7,\n"
            "  256\n"
            "):choices[0]:message:content::STRING AS reply;"
        )

    def test_refuses_a_version_with_no_service(self) -> None:
        session = served_session([])

        with pytest.raises(ValueError, match="has 0 inference services"):
            endpoint(session, "qwen", "checkpoint_2")

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
            (
                "DROP FUNCTION IF EXISTS ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2_CHAT"
                "(ARRAY, FLOAT, INT)"
            ),
        ]
        assert text == (
            "dropped service: ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2\n"
            "dropped function: ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2_CHAT\n"
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
            "dropped function: ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2_CHAT\n"
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
        forget = session.statements.index(
            "DELETE FROM app_state.served_models "
            "WHERE service_name IN ('QWEN_CHECKPOINT_2')"
        )
        assert drop < forget
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

        assert text == (
            "dropped service: ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2\n"
            "dropped function: ARENA_APP.APP_STATE.QWEN_CHECKPOINT_2_CHAT"
        )
        assert f"Arena has no deployment {DEPLOYMENT_ID} to close" in caplog.text

    def test_refuses_a_version_with_nothing_to_drop(self) -> None:
        session = served_session([])

        with pytest.raises(ValueError, match="has no inference service"):
            unpublish(session, "qwen", "checkpoint_2", False)

        assert not any(s.startswith("DROP") for s in session.statements)


REFERENCES_QUERY = "SELECT SYSTEM$GET_ALL_REFERENCES('EXPORT_STAGE') AS REFS"


def export_session(bound: bool = True, install: Install | None = None) -> AppSession:
    """A run with merged exports at steps 2 and 4, and ``export_stage`` bound or not."""
    listing = [
        Row(name=f"runs/{RUN_ID}/job.json", size=10),
        Row(name=f"runs/{RUN_ID}/checkpoint_2/merged/model.safetensors", size=2**30),
        Row(name=f"runs/{RUN_ID}/checkpoint_2/merged/.complete", size=0),
        Row(name=f"runs/{RUN_ID}/checkpoint_4/merged/model.safetensors", size=2**30),
        Row(name=f"runs/{RUN_ID}/checkpoint_4/merged/config.json", size=100),
        Row(name=f"runs/{RUN_ID}/checkpoint_4/merged/.complete", size=0),
        Row(name=f"runs/{RUN_ID}/checkpoint_6/model.safetensors", size=2**30),
    ]
    copied = [
        Row(file="model.safetensors"),
        Row(file="config.json"),
        Row(file=".complete"),
    ]
    return AppSession(
        [
            exact(REFERENCES_QUERY, [Row(REFS='["ref-1"]' if bound else "[]")]),
            known(),
            exact(f"LIST {RUN_DIR}/", listing),
            (lambda s: s.startswith("COPY FILES INTO "), copied),
        ],
        install=install,
    )


def copies(session: AppSession) -> list[str]:
    return [s for s in session.statements if s.startswith("COPY FILES")]


class TestExportModel:
    def test_copies_the_newest_merged_checkpoint_to_the_bound_stage(self) -> None:
        session = export_session()

        text = export_model(session, RUN_ID, None)

        assert copies(session) == [
            (
                "COPY FILES INTO @reference('EXPORT_STAGE')/"
                f"{RUN_ID}/checkpoint_4/ FROM {RUN_DIR}/checkpoint_4/merged/"
            )
        ]
        assert text == (
            f"checkpoint: checkpoint_4\npath: {RUN_ID}/checkpoint_4/\nfiles: 3"
        )

    def test_copies_the_checkpoint_of_the_given_step(self) -> None:
        session = export_session()

        text = export_model(session, RUN_ID, 2)

        assert copies(session) == [
            (
                "COPY FILES INTO @reference('EXPORT_STAGE')/"
                f"{RUN_ID}/checkpoint_2/ FROM {RUN_DIR}/checkpoint_2/merged/"
            )
        ]
        assert text.startswith("checkpoint: checkpoint_2\n")

    def test_starts_nothing_and_records_no_usage(self) -> None:
        session = export_session()

        export_model(session, RUN_ID, None)

        assert executes(session) == []
        assert not any(
            "usage_items" in s or s.startswith("ALTER TASK") for s in session.statements
        )

    def test_exports_after_an_unpurchased_trial_ends(self) -> None:
        session = export_session(install=listing_trial(age_days=8))

        text = export_model(session, RUN_ID, None)

        assert text.startswith("checkpoint: checkpoint_4\n")
        assert len(copies(session)) == 1

    def test_refuses_when_no_stage_is_bound(self) -> None:
        session = export_session(bound=False)

        with pytest.raises(ValueError, match="No stage is bound to export_stage"):
            export_model(session, RUN_ID, None)

        assert copies(session) == []

    def test_refuses_a_run_the_app_does_not_have(self) -> None:
        session = AppSession([exact(REFERENCES_QUERY, [Row(REFS='["ref-1"]')])])

        with pytest.raises(ValueError, match=f"This app has no run {RUN_ID}"):
            export_model(session, RUN_ID, None)

        assert copies(session) == []

    def test_refuses_a_step_without_a_merged_export(self) -> None:
        session = export_session()

        with pytest.raises(ValueError, match="checkpoint 6 has no complete merged/"):
            export_model(session, RUN_ID, 6)

        assert copies(session) == []

    def test_refuses_a_malformed_run_id(self) -> None:
        session = export_session()

        with pytest.raises(ValueError, match="run"):
            export_model(session, "../other", None)

        assert copies(session) == []


class TestRunProcedureStatements:
    def test_no_run_procedure_runs_use(self, arena: Arena) -> None:
        # Arrange
        arena.replies = [capabilities(["GPU_NV_S"]), FITS, deployment_opened()]
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
