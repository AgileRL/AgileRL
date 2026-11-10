# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Client-side Snowflake job service submission."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from agilerl.arena.snowflake.job import (
    CONTAINER_ARGS,
    GPU_NODE_RESOURCES,
    NODE_COMMAND,
    SubmitSettings,
    container_env,
    describe_instance_family,
    format_status_rows,
    job_document,
    job_id_from_run_id,
    job_service_sql,
    job_state,
    load_manifest,
    log_text,
    new_run_id,
    node_resources,
    render_spec,
    rollout_topic_route,
    runs_prefix,
    service_logs_sql,
    service_name,
    service_status_sql,
    shm_gib,
    snowflake_identifier,
    stage_directory,
    submit_job,
    trainer_ports,
)
from agilerl.arena.snowflake.ports import TrainerPorts, service_endpoints
from agilerl.arena.snowflake.source import QuerySource, TableSource

RUN_ID = "arena-20261109163457-abc123"
SERVICE = "ARENA_20261109163457_ABC123"
NETWORK = {
    "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
    "max_context_length": 128,
}
PROMPTS = TableSource("arena.training.prompts")


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _sft(export: str = "merged") -> dict:
    return {
        "algorithm": {"name": "SFT"},
        "environment": {
            "dataset": "prompts",
            "columns": {"prompt": "PROMPT", "response": "RESPONSE"},
        },
        "network": NETWORK,
        "training": {
            "max_steps": 2,
            "checkpoint_export": {"format": export},
        },
    }


def _dpo() -> dict:
    return {
        "algorithm": {"name": "DPO"},
        "environment": {"dataset": "prefs"},
        "network": NETWORK,
        "training": {
            "max_steps": 2,
            "checkpoint_export": {"format": "merged"},
        },
    }


def _rl() -> dict:
    return {
        "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
        "environment": {
            "env_type": "rollout",
            "env_url": "https://env.example.com",
            "max_turns": 1,
        },
        "network": NETWORK,
        "training": {
            "max_steps": 2,
            "checkpoint_export": {"format": "merged"},
        },
    }


def _dataset_rl() -> dict:
    return {
        "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
        "environment": {
            "env_type": "rollout",
            "dataset": "openai/gsm8k",
            "reward_file_path": "reward.py",
            "prompt_template": {"user_0": "{question}"},
        },
        "network": NETWORK,
        "training": {
            "max_steps": 2,
            "checkpoint_export": {"format": "merged"},
        },
    }


def _settings(
    source: TableSource | QuerySource | None = PROMPTS,
    nodes: int = 2,
    reward_file: Path | None = None,
    **overrides: object,
) -> SubmitSettings:
    values: dict[str, object] = {
        "compute_pool": "arena_gpu_pool",
        "image": "/arena/training/images/snowflake-job:latest",
        "warehouse": "arena_wh",
        "runs_stage": "arena.training.runs",
        "metrics_table": "arena.training.arena_metrics",
        "source": source,
        "nodes": nodes,
        "external_access_integrations": ("arena_hf_eai", "arena_pulsar_eai"),
        "checkpoint_steps": 10,
        "pulsar_url": "pulsar://broker.example:6650",
        "pulsar_topic_prefix": "non-persistent://customer/arena",
        "reward_file": reward_file,
    }
    values.update(overrides)
    return SubmitSettings(**values)


class Row:
    """Snowpark ``Row`` stand-in: values only through ``as_dict``."""

    def __init__(self, **values: object) -> None:
        self._values = values

    def as_dict(self) -> dict[str, object]:
        return dict(self._values)


class Result:
    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows

    def collect(self) -> list[Row]:
        return self.rows


class FileApi:
    def __init__(self, session: Session) -> None:
        self.session = session

    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = True,
        overwrite: bool = False,
    ) -> None:
        path = Path(local)
        self.session.events.append(f"put {path.name}")
        self.session.puts.append(
            {
                "name": path.name,
                "text": path.read_text(encoding="utf-8"),
                "stage": stage,
                "auto_compress": auto_compress,
                "overwrite": overwrite,
            }
        )


class Session:
    def __init__(self, family: str = "GPU_NV_S", rows: list[Row] | None = None) -> None:
        self.rows = [Row(instance_family=family)] if rows is None else rows
        self.events: list[str] = []
        self.statements: list[str] = []
        self.puts: list[dict] = []
        self.file = FileApi(self)

    def sql(self, text: str) -> Result:
        self.events.append(text.split("\n")[0])
        self.statements.append(text)
        if text.startswith("DESCRIBE"):
            return Result(self.rows)
        return Result([])


class TestNewRunId:
    def test_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.snowflake.job.secrets.token_hex", lambda n: "ab" * n
        )
        moment = datetime(2026, 11, 9, 16, 34, 57, tzinfo=timezone.utc)

        assert new_run_id(moment) == "arena-20261109163457-ababab"

    def test_uses_the_current_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.snowflake.job.secrets.token_hex", lambda n: "cd" * n
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.job.datetime",
            type(
                "Clock",
                (),
                {
                    "now": staticmethod(
                        lambda tz: datetime(2026, 1, 2, 3, 4, 5, tzinfo=tz)
                    )
                },
            ),
        )

        assert new_run_id() == "arena-20260102030405-cdcdcd"


class TestServiceName:
    def test_drops_dashes_and_uppercases_the_suffix(self) -> None:
        assert service_name(RUN_ID) == SERVICE
        assert job_id_from_run_id(RUN_ID) == int("abc123", 16)

    def test_a_bad_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="run_id must look like"):
            service_name("job-1")
        with pytest.raises(ValueError, match="run_id must look like"):
            job_id_from_run_id("job-1")


class TestNodeResources:
    def test_known_gpu_families(self) -> None:
        small = node_resources("GPU_NV_S")
        large = node_resources("GPU_NV_L")
        blackwell = node_resources("GPU_R6K_G1_192")
        azure = node_resources("GPU_NV_XS")

        assert (small.gpus, small.memory_gib) == GPU_NODE_RESOURCES["GPU_NV_S"]
        assert (large.gpus, large.memory_gib) == (8, 1112)
        assert (blackwell.gpus, blackwell.memory_gib) == (8, 1843)
        assert (azure.gpus, azure.memory_gib) == (1, 26)
        assert shm_gib(28) == 9
        assert shm_gib(1) == 1

    def test_an_unknown_family_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown compute pool instance family"):
            node_resources("CPU_X64_XS")


class TestTrainerPorts:
    def test_unset_engines_count_every_gpu_in_the_job(self, tmp_path: Path) -> None:
        manifest = load_manifest(_write(tmp_path, _rl()))

        ports = trainer_ports(manifest, node_resources("GPU_NV_M"), 2)

        assert ports == TrainerPorts(agents=1, engines_per_agent=8)

    def test_explicit_engines_and_population_are_kept(self, tmp_path: Path) -> None:
        document = _rl()
        document["training"]["rollout_engines_per_agent"] = 3
        manifest = load_manifest(_write(tmp_path, document))

        ports = trainer_ports(manifest, node_resources("GPU_NV_S"), 1)

        assert ports == TrainerPorts(agents=1, engines_per_agent=3)

    def test_auto_engines_count_every_gpu_in_the_job(self, tmp_path: Path) -> None:
        document = _rl()
        document["training"]["rollout_engines_per_agent"] = "auto"
        manifest = load_manifest(_write(tmp_path, document))

        ports = trainer_ports(manifest, node_resources("GPU_NV_S"), 3)

        assert ports == TrainerPorts(agents=1, engines_per_agent=3)


class TestContainerEnv:
    def test_pulsar_and_job_id_values(self) -> None:
        env = container_env(_settings(), RUN_ID)

        assert env == {
            "APP_PULSAR_URL": "pulsar://broker.example:6650",
            "APP_TOPIC_PREFIX": "non-persistent://customer/arena",
            "AGILERL_ROLLOUT_TOPIC_ROUTE": "persistent://customer/arena/llm-rollout",
            "JOB_ID": str(int("abc123", 16)),
        }


class TestRolloutTopicRoute:
    @pytest.mark.parametrize(
        ("prefix", "route"),
        [
            (
                "non-persistent://customer/arena",
                "persistent://customer/arena/llm-rollout",
            ),
            ("persistent://customer/arena/", "persistent://customer/arena/llm-rollout"),
            ("customer/arena/llm-rollout", "persistent://customer/arena/llm-rollout"),
        ],
    )
    def test_route_sits_beside_the_prefix(self, prefix: str, route: str) -> None:
        assert rollout_topic_route(prefix) == route


class TestRenderSpec:
    def test_gpu_nv_s_spec(self) -> None:
        settings = _settings(nodes=1)
        trainer = TrainerPorts(agents=1, engines_per_agent=1)

        spec = yaml.safe_load(
            render_spec(settings, RUN_ID, node_resources("GPU_NV_S"), trainer)
        )
        container = spec["spec"]["containers"][0]

        assert container["name"] == "main"
        assert container["image"] == "/arena/training/images/snowflake-job:latest"
        assert container["command"] == NODE_COMMAND
        assert container["args"] == CONTAINER_ARGS
        assert container["env"] == container_env(settings, RUN_ID)
        assert container["resources"]["requests"] == {
            "nvidia.com/gpu": 1,
            "memory": "27.5Gi",
        }
        assert container["resources"]["limits"] == container["resources"]["requests"]
        assert container["volumeMounts"] == [
            {"name": "run", "mountPath": "/arena/run"},
            {"name": "dshm", "mountPath": "/dev/shm"},
        ]
        assert spec["spec"]["volumes"] == [
            {
                "name": "run",
                "source": "stage",
                "stageConfig": {"name": f"@arena.training.runs/{RUN_ID}"},
            },
            {"name": "dshm", "source": "memory", "size": "9Gi"},
        ]
        assert spec["spec"]["endpoints"] == service_endpoints(trainer)

    def test_multi_node_spec_sizes_ports_for_every_gpu(self, tmp_path: Path) -> None:
        settings = _settings(nodes=4)
        resources = node_resources("GPU_NV_M")
        manifest = load_manifest(_write(tmp_path, _rl()))
        trainer = trainer_ports(manifest, resources, settings.nodes)

        spec = yaml.safe_load(render_spec(settings, RUN_ID, resources, trainer))
        sql = job_service_sql(
            settings, SERVICE, render_spec(settings, RUN_ID, resources, trainer)
        )

        assert trainer.engines_per_agent == 16
        assert spec["spec"]["endpoints"] == service_endpoints(trainer)
        assert spec["spec"]["volumes"][0]["stageConfig"]["name"] == (
            f"@arena.training.runs/{RUN_ID}"
        )
        assert spec["spec"]["containers"][0]["resources"]["requests"] == {
            "nvidia.com/gpu": 4,
            "memory": "177.5Gi",
        }
        assert "REPLICAS = 4" in sql

    def test_a_dollar_quote_in_the_image_is_rejected(self) -> None:
        settings = _settings(image="repo/$$bad")

        with pytest.raises(ValueError, match=r"\$\$"):
            render_spec(
                settings,
                RUN_ID,
                node_resources("GPU_NV_S"),
                TrainerPorts(agents=1, engines_per_agent=1),
            )


class TestJobServiceSql:
    def test_replicas_warehouse_eais_and_async(self) -> None:
        sql = job_service_sql(_settings(), SERVICE, "spec:\n  containers: []\n")

        assert sql.startswith("EXECUTE JOB SERVICE IN COMPUTE POOL arena_gpu_pool\n")
        assert "NAME = ARENA_20261109163457_ABC123" in sql
        assert "ASYNC = TRUE" in sql
        assert "REPLICAS = 2" in sql
        assert "QUERY_WAREHOUSE = arena_wh" in sql
        assert "EXTERNAL_ACCESS_INTEGRATIONS = (arena_hf_eai, arena_pulsar_eai)" in sql
        assert "FROM SPECIFICATION $$\nspec:" in sql

    def test_empty_integrations_are_omitted(self) -> None:
        settings = _settings(nodes=1, external_access_integrations=())

        sql = job_service_sql(settings, SERVICE, "spec: {}\n")

        assert "EXTERNAL_ACCESS_INTEGRATIONS" not in sql

    @pytest.mark.parametrize(
        ("field", "value", "kind"),
        [
            ("compute_pool", "pool-name", "compute pool"),
            ("compute_pool", "pool$$x", "compute pool"),
            ("warehouse", "wh; DROP TABLE t", "warehouse"),
            ("warehouse", '"wh"', "warehouse"),
            ("external_access_integrations", ("ok_eai", "bad eai"), "external access"),
        ],
    )
    def test_unsafe_names_are_rejected(
        self, field: str, value: object, kind: str
    ) -> None:
        settings = _settings(**{field: value})

        with pytest.raises(ValueError, match=kind):
            job_service_sql(settings, SERVICE, "spec: {}\n")


class TestSnowflakeIdentifier:
    def test_dotted_names_pass_through(self) -> None:
        assert snowflake_identifier(" db.schema.t$1 ", "table") == "db.schema.t$1"

    @pytest.mark.parametrize("value", ["", "a..b", "a$$b", "a'b", "1abc"])
    def test_unsafe_names_are_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="table must be a Snowflake identifier"):
            snowflake_identifier(value, "table")


class TestRunsPrefix:
    def test_leading_at_is_optional(self) -> None:
        assert runs_prefix("@arena.training.runs", RUN_ID) == (
            f"@arena.training.runs/{RUN_ID}"
        )
        assert stage_directory("arena.runs", RUN_ID, "checkpoint_2/merged/a.bin") == (
            f"@arena.runs/{RUN_ID}/checkpoint_2/merged"
        )
        assert stage_directory("arena.runs", RUN_ID, "metrics.jsonl") == (
            f"@arena.runs/{RUN_ID}"
        )

    def test_a_bad_stage_or_run_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="stage"):
            runs_prefix("runs stage", RUN_ID)
        with pytest.raises(ValueError, match="run_id must look like"):
            runs_prefix("runs", "x' OR 1=1")


class TestServiceQueries:
    def test_status_and_logs_sql(self) -> None:
        assert service_status_sql(SERVICE) == (
            f"SHOW SERVICE CONTAINERS IN SERVICE {SERVICE}"
        )
        assert service_logs_sql(SERVICE, 0) == (
            f"SELECT SYSTEM$GET_SERVICE_LOGS('{SERVICE}', 0, 'main')"
        )

    def test_status_rows_end_with_the_job_state(self) -> None:
        rows = [
            Row(
                SERVICE_STATUS="RUNNING",
                INSTANCE_ID=0,
                CONTAINER_NAME="main",
                STATUS="DONE",
            ),
            Row(
                service_status="RUNNING",
                instance_id=1,
                container_name="main",
                status="READY",
            ),
        ]

        assert format_status_rows(rows) == (
            "0\tmain\tDONE\n1\tmain\tREADY\nstate: running"
        )

    def test_log_text_reads_the_first_column(self) -> None:
        assert log_text([Row(**{"SYSTEM$GET_SERVICE_LOGS": "trainer started"})]) == (
            "trainer started"
        )
        assert log_text([]) == ""
        assert log_text([Row()]) == ""

    def test_a_bad_service_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="service"):
            service_status_sql("arena-job")


class TestJobState:
    def test_done_when_the_service_is_done(self) -> None:
        assert job_state({"DONE"}) == "done"

    @pytest.mark.parametrize(
        ("statuses", "state"),
        [
            ({"RUNNING"}, "running"),
            ({"PENDING"}, "running"),
            (set(), "running"),
            ({"DONE", "RUNNING"}, "running"),
            ({"FAILED"}, "failed"),
            ({"INTERNAL_ERROR"}, "failed"),
        ],
    )
    def test_other_states(self, statuses: set[str], state: str) -> None:
        assert job_state(statuses) == state


class TestDescribeInstanceFamily:
    def test_reads_the_family(self) -> None:
        session = Session(rows=[Row(INSTANCE_FAMILY="GPU_NV_S")])

        assert describe_instance_family(session, "arena_gpu_pool") == "GPU_NV_S"
        assert session.statements == ["DESCRIBE COMPUTE POOL arena_gpu_pool"]

    def test_a_missing_pool_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="was not found"):
            describe_instance_family(Session(rows=[]), "arena_gpu_pool")

    def test_a_row_without_a_family_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="instance_family"):
            describe_instance_family(Session(rows=[Row(name="pool")]), "pool")
        with pytest.raises(ValueError, match="instance_family"):
            describe_instance_family(Session(rows=[Row(instance_family="  ")]), "pool")


class TestSubmitJob:
    def test_describes_and_renders_then_puts_then_executes(
        self, tmp_path: Path
    ) -> None:
        session = Session()
        manifest = _write(tmp_path, _sft())
        settings = _settings()

        run_id, name = submit_job(session, manifest, settings, run_id=RUN_ID)

        assert (run_id, name) == (RUN_ID, SERVICE)
        assert session.events == [
            "DESCRIBE COMPUTE POOL arena_gpu_pool",
            "put manifest.yaml",
            "put job.json",
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_gpu_pool",
        ]
        assert session.puts[0]["stage"] == f"@arena.training.runs/{RUN_ID}"
        assert session.puts[0]["auto_compress"] is False
        assert session.puts[0]["overwrite"] is True
        assert yaml.safe_load(session.puts[0]["text"])["algorithm"]["name"] == "SFT"
        document = json.loads(session.puts[1]["text"])
        assert document == job_document(settings, RUN_ID)
        assert document["source"] == {"kind": "table", "name": "arena.training.prompts"}
        assert document["reward_file"] is False
        assert "REPLICAS = 2" in session.statements[-1]

    def test_dpo_is_accepted(self, tmp_path: Path) -> None:
        session = Session()

        run_id, name = submit_job(
            session,
            _write(tmp_path, _dpo()),
            _settings(QuerySource("select 1")),
            RUN_ID,
        )

        assert (run_id, name) == (RUN_ID, SERVICE)
        assert json.loads(session.puts[1]["text"])["source"] == {
            "kind": "query",
            "sql": "select 1",
        }

    def test_rl_env_url_warns_about_external_hosts(self, tmp_path: Path) -> None:
        with pytest.warns(UserWarning, match="outside the account"):
            run_id, name = submit_job(
                Session(), _write(tmp_path, _rl()), _settings(None), RUN_ID
            )

        assert (run_id, name) == (RUN_ID, SERVICE)

    def test_env_backed_rollout_runs_without_a_source(self, tmp_path: Path) -> None:
        session = Session()

        with pytest.warns(UserWarning, match="env_url"):
            submit_job(session, _write(tmp_path, _rl()), _settings(None), RUN_ID)

        assert json.loads(session.puts[1]["text"])["source"] is None

    def test_env_backed_rollout_does_not_export_a_passed_source(
        self, tmp_path: Path
    ) -> None:
        session = Session()

        with pytest.warns(UserWarning, match="source is not exported"):
            submit_job(session, _write(tmp_path, _rl()), _settings(), RUN_ID)

        assert json.loads(session.puts[1]["text"])["source"] is None

    @pytest.mark.parametrize("document", [_sft(), _dpo()])
    def test_dataset_manifest_without_a_source_is_rejected(
        self, tmp_path: Path, document: dict
    ) -> None:
        session = Session()

        with pytest.raises(ValueError, match="Pass --source-table or --source-query"):
            submit_job(session, _write(tmp_path, document), _settings(None), RUN_ID)

        assert session.events == []

    def test_dataset_rollout_without_a_source_is_rejected(self, tmp_path: Path) -> None:
        reward = tmp_path / "reward.py"
        reward.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Pass --source-table or --source-query"):
            submit_job(
                Session(),
                _write(tmp_path, _dataset_rl()),
                _settings(None, reward_file=reward),
                RUN_ID,
            )

    def test_dataset_rollout_uploads_the_reward_file(self, tmp_path: Path) -> None:
        reward = tmp_path / "my_reward.py"
        reward.write_text("def reward(x):\n    return 1.0\n", encoding="utf-8")
        session = Session()

        submit_job(
            session,
            _write(tmp_path, _dataset_rl()),
            _settings(reward_file=reward),
            RUN_ID,
        )

        assert [item["name"] for item in session.puts] == [
            "manifest.yaml",
            "job.json",
            "reward.py",
        ]
        assert session.puts[2]["text"] == "def reward(x):\n    return 1.0\n"
        assert session.puts[2]["stage"] == f"@arena.training.runs/{RUN_ID}"
        assert json.loads(session.puts[1]["text"])["reward_file"] is True

    def test_dataset_rollout_without_a_reward_file_is_rejected(
        self, tmp_path: Path
    ) -> None:
        session = Session()

        with pytest.raises(ValueError, match="needs --reward-file"):
            submit_job(session, _write(tmp_path, _dataset_rl()), _settings(), RUN_ID)

        assert session.events == []

    def test_a_reward_file_on_sft_is_rejected(self, tmp_path: Path) -> None:
        reward = tmp_path / "reward.py"
        reward.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="only used by RL"):
            submit_job(
                Session(),
                _write(tmp_path, _sft()),
                _settings(reward_file=reward),
                RUN_ID,
            )

    def test_a_population_over_the_port_bound_puts_nothing(
        self, tmp_path: Path
    ) -> None:
        document = _rl()
        document["training"]["pop_size"] = 40
        session = Session(family="GPU_NV_M")

        with (
            pytest.warns(UserWarning, match="env_url"),
            pytest.raises(ValueError, match="reach the Ray ports"),
        ):
            submit_job(session, _write(tmp_path, document), _settings(None), RUN_ID)

        assert session.puts == []

    def test_env_image_is_rejected(self, tmp_path: Path) -> None:
        document = _rl()
        document["environment"] = {
            "env_type": "rollout",
            "env_image": "env:latest",
        }

        with pytest.raises(ValueError, match="Kubernetes Pod"):
            submit_job(Session(), _write(tmp_path, document), _settings(), RUN_ID)

    def test_rollout_without_env_url_is_rejected(self, tmp_path: Path) -> None:
        document = _rl()
        document["environment"] = {
            "env_type": "rollout",
            "entrypoint": "envs:MyEnv",
        }

        with pytest.raises(ValueError, match="needs env_url"):
            submit_job(Session(), _write(tmp_path, document), _settings(), RUN_ID)

    def test_non_llm_manifest_is_rejected(self, tmp_path: Path) -> None:
        document = {
            "algorithm": {"name": "DQN", "lr": 0.001},
            "environment": {"name": "LunarLander-v3", "num_envs": 8},
            "network": {
                "latent_dim": 64,
                "encoder_config": {"hidden_size": [64]},
                "head_config": {"hidden_size": [64]},
            },
            "training": {"max_steps": 10},
        }

        with pytest.raises(ValueError, match="SFT, DPO, and RL"):
            submit_job(Session(), _write(tmp_path, document), _settings(), RUN_ID)

    def test_adapter_export_warns(self, tmp_path: Path) -> None:
        with pytest.warns(UserWarning, match="not merged"):
            submit_job(
                Session(), _write(tmp_path, _sft("adapter")), _settings(), RUN_ID
            )

    def test_an_unknown_pool_family_is_rejected(self, tmp_path: Path) -> None:
        session = Session(family="CPU_X64_XS")

        with pytest.raises(ValueError, match="Unknown compute pool instance family"):
            submit_job(session, _write(tmp_path, _sft()), _settings(), RUN_ID)

        assert session.puts == []
