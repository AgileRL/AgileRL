# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Client-side Snowflake job service submission."""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from agilerl.arena.snowflake.job import (
    BROKER_COMMAND,
    CONTAINER_ARGS,
    NODE_COMMAND,
    QUERY_DATASET,
    RunServices,
    SubmitSettings,
    cancel_run,
    choose_gpu_family,
    container_env,
    copy_resume_checkpoint,
    cpu_memory_gib,
    describe_instance_family,
    describe_max_nodes,
    event_log_text,
    format_run_status,
    job_document,
    job_id_from_run_id,
    job_service_sql,
    job_services_sql,
    job_state,
    load_manifest,
    log_text,
    manifest_fits,
    new_run_id,
    node_resources,
    qualified_table,
    render_broker_spec,
    render_head_spec,
    render_workers_spec,
    resume_checkpoint,
    role_states,
    rollout_topic_route,
    run_services,
    run_state,
    runs_prefix,
    service_event_logs_sql,
    service_logs_sql,
    service_name,
    service_status_sql,
    shm_gib,
    snowflake_identifier,
    stage_directory,
    submit_job,
    trainer_ports,
    write_job_manifest,
)
from agilerl.arena.snowflake.ports import TrainerPorts, service_endpoints
from agilerl.arena.snowflake.profile import GpuPool
from agilerl.arena.snowflake.source import QuerySource, TableSource

RUN_ID = "arena-20261109163457-abc123"
SERVICE = "ARENA_RUN_20261109163457_ABC123"
SERVICES = RunServices(
    broker=f"{SERVICE}_BROKER",
    head=f"{SERVICE}_HEAD",
    workers=f"{SERVICE}_WORKERS",
)
NETWORK = {
    "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
    "max_context_length": 128,
}
PROMPTS = TableSource("arena.training.prompts")


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _sft(export: str | None = "merged") -> dict:
    training: dict = {"max_steps": 2}
    if export is not None:
        training["checkpoint_export"] = {"format": export}
    return {
        "algorithm": {"name": "SFT"},
        "environment": {
            "dataset": "prompts",
            "columns": {"prompt": "PROMPT", "response": "RESPONSE"},
        },
        "network": NETWORK,
        "training": training,
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
    gpus: int = 2,
    reward_file: Path | None = None,
    **overrides: object,
) -> SubmitSettings:
    values: dict[str, object] = {
        "image": "/arena/training/images/snowflake-job:latest",
        "warehouse": "arena_wh",
        "runs_stage": "arena.training.runs",
        "metrics_table": "arena.training.arena_metrics",
        "source": source,
        "external_access_integrations": ("arena_hf_eai",),
        "checkpoint_steps": 10,
        "reward_file": reward_file,
        "gpus": gpus,
        "gpu_family": "GPU_NV_S",
        "gpus_per_node": 1,
        "gpu_pool": "arena_gpu_GPU_NV_S",
        "broker_pool": "arena_pulsar_pool",
        "broker_family": "CPU_X64_L",
        "head_pool": "arena_head_pool",
        "head_family": "CPU_X64_SL",
        "resume_from": None,
    }
    values.update(overrides)
    return SubmitSettings(**values)


def _spec(sql: str) -> dict:
    body = sql.split("$$\n", 1)[1].removesuffix("$$")
    loaded = yaml.safe_load(body)
    assert isinstance(loaded, dict)
    return loaded


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
    def __init__(
        self,
        family: str = "GPU_NV_S",
        rows: list[Row] | None = None,
        max_nodes: int = 32,
        fail_on_execute: int | None = None,
        listing: list[Row] | None = None,
    ) -> None:
        self.rows = (
            [Row(instance_family=family, max_nodes=max_nodes)] if rows is None else rows
        )
        self.listing = [] if listing is None else listing
        self.fail_on_execute = fail_on_execute
        self.executes = 0
        self.events: list[str] = []
        self.statements: list[str] = []
        self.puts: list[dict] = []
        self.file = FileApi(self)

    def sql(self, text: str) -> Result:
        self.events.append(text.split("\n")[0])
        self.statements.append(text)
        if text.startswith("EXECUTE"):
            self.executes += 1
            if (
                self.fail_on_execute is not None
                and self.executes == self.fail_on_execute
            ):
                msg = "execute failed"
                raise RuntimeError(msg)
        if text.startswith("DESCRIBE"):
            return Result(self.rows)
        if text.startswith("LIST"):
            return Result(self.listing)
        return Result([])


class TestWriteJobManifest:
    def test_a_table_source_fills_the_dataset(self, tmp_path: Path) -> None:
        # Arrange
        document = _sft()
        del document["environment"]["dataset"]
        manifest = _write(tmp_path, document)
        out = tmp_path / "job"
        out.mkdir()

        # Act
        written = write_job_manifest(manifest, PROMPTS, False, out)

        # Assert
        assert written == out / "manifest.yaml"
        environment = yaml.safe_load(written.read_text(encoding="utf-8"))["environment"]
        assert environment["dataset"] == "arena.training.prompts"
        assert "rubric_file_path" not in environment
        load_manifest(written)

    def test_a_query_source_fills_a_named_dataset(self, tmp_path: Path) -> None:
        document = _sft()
        del document["environment"]["dataset"]
        out = tmp_path / "job"
        out.mkdir()

        written = write_job_manifest(
            _write(tmp_path, document), QuerySource("select 1"), False, out
        )

        environment = yaml.safe_load(written.read_text(encoding="utf-8"))["environment"]
        assert environment["dataset"] == QUERY_DATASET

    def test_grpo_over_a_table_takes_the_rubric_from_the_reward_file(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        document = _dataset_rl()
        del document["environment"]["dataset"]
        del document["environment"]["reward_file_path"]
        manifest = _write(tmp_path, document)
        out = tmp_path / "job"
        out.mkdir()

        # Act
        written = write_job_manifest(manifest, PROMPTS, True, out)

        # Assert
        environment = yaml.safe_load(written.read_text(encoding="utf-8"))["environment"]
        assert environment["dataset"] == "arena.training.prompts"
        assert environment["rubric_file_path"] == "reward.py"
        load_manifest(written)

    def test_fields_the_manifest_sets_stay(self, tmp_path: Path) -> None:
        out = tmp_path / "job"
        out.mkdir()

        written = write_job_manifest(
            _write(tmp_path, _dataset_rl()), PROMPTS, True, out
        )

        environment = yaml.safe_load(written.read_text(encoding="utf-8"))["environment"]
        assert environment == _dataset_rl()["environment"]

    def test_a_reward_file_adds_no_rubric_outside_a_rollout(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "job"
        out.mkdir()

        written = write_job_manifest(_write(tmp_path, _sft()), PROMPTS, True, out)

        environment = yaml.safe_load(written.read_text(encoding="utf-8"))["environment"]
        assert environment == _sft()["environment"]

    def test_an_env_backed_rollout_is_unchanged(self, tmp_path: Path) -> None:
        out = tmp_path / "job"
        out.mkdir()

        written = write_job_manifest(_write(tmp_path, _rl()), PROMPTS, True, out)

        assert yaml.safe_load(written.read_text(encoding="utf-8")) == _rl()

    def test_no_source_adds_no_dataset(self, tmp_path: Path) -> None:
        document = _sft()
        del document["environment"]["dataset"]
        out = tmp_path / "job"
        out.mkdir()

        written = write_job_manifest(_write(tmp_path, document), None, False, out)

        environment = yaml.safe_load(written.read_text(encoding="utf-8"))["environment"]
        assert "dataset" not in environment

    def test_an_omitted_checkpoint_export_becomes_merged(self, tmp_path: Path) -> None:
        # Arrange
        out = tmp_path / "job"
        out.mkdir()

        # Act
        written = write_job_manifest(_write(tmp_path, _sft(None)), PROMPTS, False, out)

        # Assert
        training = yaml.safe_load(written.read_text(encoding="utf-8"))["training"]
        assert training["checkpoint_export"] == {"format": "merged", "trigger": "final"}
        assert load_manifest(written).training.effective_checkpoint_export().format == (
            "merged"
        )

    def test_a_manifest_without_training_gets_a_merged_export(
        self, tmp_path: Path
    ) -> None:
        document = _sft()
        del document["training"]
        out = tmp_path / "job"
        out.mkdir()

        written = write_job_manifest(_write(tmp_path, document), PROMPTS, False, out)

        training = yaml.safe_load(written.read_text(encoding="utf-8"))["training"]
        assert training == {
            "checkpoint_export": {"format": "merged", "trigger": "final"}
        }

    def test_an_explicit_adapter_export_stays(self, tmp_path: Path) -> None:
        out = tmp_path / "job"
        out.mkdir()

        written = write_job_manifest(
            _write(tmp_path, _sft("adapter")), PROMPTS, False, out
        )

        training = yaml.safe_load(written.read_text(encoding="utf-8"))["training"]
        assert training["checkpoint_export"] == {"format": "adapter"}


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
        assert run_services(RUN_ID) == SERVICES
        assert job_id_from_run_id(RUN_ID) == int("abc123", 16)

    def test_a_bad_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="run_id must look like"):
            service_name("job-1")
        with pytest.raises(ValueError, match="run_id must look like"):
            run_services("job-1")
        with pytest.raises(ValueError, match="run_id must look like"):
            job_id_from_run_id("job-1")


class TestNodeResources:
    def test_known_gpu_families(self) -> None:
        small = node_resources("GPU_NV_S")
        large = node_resources("GPU_NV_L")
        blackwell = node_resources("GPU_R6K_G1_192")
        azure = node_resources("GPU_NV_XS")
        gcp = node_resources("GPU_GCP_NV_A100_8_40G")

        assert (small.gpus, small.memory_gib) == (1, 27)
        assert (large.gpus, large.memory_gib) == (8, 1112)
        assert (blackwell.gpus, blackwell.memory_gib) == (8, 1843)
        assert (azure.gpus, azure.memory_gib) == (1, 26)
        assert (gcp.gpus, gcp.memory_gib) == (8, 654)
        assert shm_gib(28) == 9
        assert shm_gib(1) == 1

    def test_an_unknown_family_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown compute pool instance family"):
            node_resources("CPU_X64_XS")


class TestTrainerPorts:
    def test_unset_engines_count_worker_gpus(self, tmp_path: Path) -> None:
        manifest = load_manifest(_write(tmp_path, _rl()))

        ports = trainer_ports(manifest, 8)

        assert ports == TrainerPorts(agents=1, engines_per_agent=8)

    def test_explicit_engines_and_population_are_kept(self, tmp_path: Path) -> None:
        document = _rl()
        document["training"]["rollout_engines_per_agent"] = 3
        manifest = load_manifest(_write(tmp_path, document))

        ports = trainer_ports(manifest, 1)

        assert ports == TrainerPorts(agents=1, engines_per_agent=3)

    def test_auto_engines_count_worker_gpus(self, tmp_path: Path) -> None:
        document = _rl()
        document["training"]["rollout_engines_per_agent"] = "auto"
        manifest = load_manifest(_write(tmp_path, document))

        ports = trainer_ports(manifest, 3)

        assert ports == TrainerPorts(agents=1, engines_per_agent=3)


class TestCpuMemory:
    def test_known_families(self) -> None:
        assert cpu_memory_gib("CPU_X64_XS") == 6
        assert cpu_memory_gib("CPU_X64_L") == 116
        assert cpu_memory_gib("HIGHMEM_X64_L") == 984

    def test_an_unknown_family_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="NO_SUCH_CPU"):
            cpu_memory_gib("NO_SUCH_CPU")


class TestContainerEnv:
    def test_topic_run_id_and_job_id_values(self) -> None:
        env = container_env(RUN_ID)

        assert env == {
            "APP_TOPIC_PREFIX": "non-persistent://arena/run",
            "AGILERL_ROLLOUT_TOPIC_ROUTE": "persistent://arena/run/llm-rollout",
            "ARENA_RUN_ID": RUN_ID,
            "JOB_ID": str(int("abc123", 16)),
        }
        assert "APP_PULSAR_URL" not in env


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
    def test_broker_has_no_gpu_and_mounts_a_block_volume(self) -> None:
        spec = yaml.safe_load(render_broker_spec(_settings(), RUN_ID))
        container = spec["spec"]["containers"][0]

        assert container["command"] == BROKER_COMMAND
        assert "args" not in container
        assert container["env"] == container_env(RUN_ID)
        assert "nvidia.com/gpu" not in container["resources"]["requests"]
        assert container["resources"]["requests"]["memory"] == "115.5Gi"
        assert container["resources"]["limits"] == container["resources"]["requests"]
        assert {"name": "data", "mountPath": "/pulsar-data"} in container[
            "volumeMounts"
        ]
        block = next(
            item for item in spec["spec"]["volumes"] if item["source"] == "block"
        )
        assert block["name"] == "data"
        assert block["size"] == "1000Gi"
        assert block["blockConfig"] == {"iops": 16000, "throughput": 1000}
        by_name = {item["name"]: item for item in spec["spec"]["endpoints"]}
        assert by_name["pulsar"] == {
            "name": "pulsar",
            "port": 6650,
            "protocol": "TCP",
            "public": False,
        }
        assert by_name["pulsar-admin"]["port"] == 8080
        assert by_name["pulsar-admin"]["public"] is False

    def test_head_opens_ray_ports_only(self) -> None:
        spec = yaml.safe_load(render_head_spec(_settings(), RUN_ID))
        container = spec["spec"]["containers"][0]

        assert container["command"] == [*NODE_COMMAND, *CONTAINER_ARGS]
        assert "args" not in container
        assert "nvidia.com/gpu" not in container["resources"]["requests"]
        assert container["resources"]["requests"]["memory"] == "57.5Gi"
        assert spec["spec"]["endpoints"] == service_endpoints(None)
        assert spec["spec"]["volumes"][1] == {
            "name": "dshm",
            "source": "memory",
            "size": "18Gi",
        }

    def test_workers_request_the_node_and_count_worker_gpus(
        self, tmp_path: Path
    ) -> None:
        settings = _settings(
            gpus=16,
            gpu_family="GPU_NV_M",
            gpus_per_node=4,
            gpu_pool="arena_gpu_GPU_NV_M",
        )
        manifest = load_manifest(_write(tmp_path, _rl()))
        trainer = trainer_ports(manifest, settings.gpus)

        spec = yaml.safe_load(render_workers_spec(settings, RUN_ID, trainer))
        sql = job_service_sql(
            settings.gpu_pool,
            settings.warehouse,
            SERVICES.workers,
            render_workers_spec(settings, RUN_ID, trainer),
            4,
            settings.external_access_integrations,
        )

        assert trainer.engines_per_agent == 16
        assert spec["spec"]["endpoints"] == service_endpoints(trainer)
        assert spec["spec"]["containers"][0]["resources"]["requests"] == {
            "nvidia.com/gpu": 4,
            "memory": "177.5Gi",
        }
        assert "REPLICAS = 4" in sql
        assert "APP_PULSAR_URL" not in sql

    def test_a_dollar_quote_in_the_image_is_rejected(self) -> None:
        settings = _settings(image="repo/$$bad")

        with pytest.raises(ValueError, match=r"\$\$"):
            render_broker_spec(settings, RUN_ID)


class TestJobServiceSql:
    def test_replicas_warehouse_eais_and_async(self) -> None:
        sql = job_service_sql(
            "arena_gpu_GPU_NV_S",
            "arena_wh",
            SERVICE,
            "spec:\n  containers: []\n",
            2,
            ("arena_hf_eai",),
        )

        assert sql.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_gpu_GPU_NV_S\n"
        )
        assert f"NAME = {SERVICE}" in sql
        assert "ASYNC = TRUE" in sql
        assert "REPLICAS = 2" in sql
        assert "QUERY_WAREHOUSE = arena_wh" in sql
        assert "EXTERNAL_ACCESS_INTEGRATIONS = (arena_hf_eai)" in sql
        assert "FROM SPECIFICATION $$\nspec:" in sql

    def test_empty_integrations_are_omitted(self) -> None:
        sql = job_service_sql("arena_pulsar_pool", "arena_wh", SERVICE, "spec: {}\n", 1)

        assert "EXTERNAL_ACCESS_INTEGRATIONS" not in sql

    @pytest.mark.parametrize(
        ("pool", "warehouse", "integrations", "kind"),
        [
            ("pool-name", "arena_wh", (), "compute pool"),
            ("pool$$x", "arena_wh", (), "compute pool"),
            ("arena_wh", "wh; DROP TABLE t", (), "warehouse"),
            ("arena_wh", '"wh"', (), "warehouse"),
            ("arena_wh", "arena_wh", ("ok_eai", "bad eai"), "external access"),
        ],
    )
    def test_unsafe_names_are_rejected(
        self,
        pool: str,
        warehouse: str,
        integrations: tuple[str, ...],
        kind: str,
    ) -> None:
        with pytest.raises(ValueError, match=kind):
            job_service_sql(pool, warehouse, SERVICE, "spec: {}\n", 1, integrations)


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

    def test_run_status_lists_roles_containers_and_the_run_state(self) -> None:
        # Arrange
        states = {"broker": "gone", "head": "running", "worker": "running"}
        containers = {
            "broker": [],
            "head": [Row(INSTANCE_ID=0, CONTAINER_NAME="main", STATUS="READY")],
            "worker": [
                Row(instance_id=0, container_name="main", status="READY"),
                Row(instance_id=1, container_name="main", status="PENDING"),
            ],
        }

        # Act
        text = format_run_status(states, containers)

        # Assert
        assert text == (
            "broker\tgone\n"
            "head\trunning\n"
            "  0\tmain\tREADY\n"
            "worker\trunning\n"
            "  0\tmain\tREADY\n"
            "  1\tmain\tPENDING\n"
            "state: running"
        )

    def test_a_pending_service_row_without_an_instance_is_skipped(self) -> None:
        states = {"broker": "running", "head": "running", "worker": "running"}
        pending = [Row(instance_id=None, container_name=None, status=None)]

        text = format_run_status(
            states, {"broker": pending, "head": pending, "worker": pending}
        )

        assert text == "broker\trunning\nhead\trunning\nworker\trunning\nstate: running"

    def test_job_services_sql_matches_the_run_stem(self) -> None:
        assert job_services_sql(RUN_ID) == f"SHOW JOB SERVICES LIKE '{SERVICE}%'"

    def test_role_states_read_show_job_services(self) -> None:
        rows = [
            Row(name=SERVICES.head, status="DONE"),
            Row(NAME=SERVICES.workers, STATUS="running"),
        ]

        assert role_states(RUN_ID, rows) == {
            "broker": "gone",
            "head": "done",
            "worker": "running",
        }

    def test_event_logs_sql_filters_instance_and_container(self) -> None:
        assert service_event_logs_sql(SERVICE, 2) == (
            f"SELECT LOG FROM TABLE({SERVICE}!SPCS_GET_LOGS("
            "START_TIME => DATEADD('day', -30, CURRENT_TIMESTAMP())))\n"
            "WHERE INSTANCE_ID = 2 AND CONTAINER_NAME = 'main'\n"
            "ORDER BY TIMESTAMP DESC\n"
            "LIMIT 1000"
        )

    def test_event_log_text_is_oldest_first(self) -> None:
        rows = [Row(LOG="second"), Row(LOG="first")]

        assert event_log_text(rows) == "first\nsecond"
        assert event_log_text([]) == ""

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
            ({"DONE", "RUNNING"}, "running"),
            ({"FAILED"}, "failed"),
            ({"INTERNAL_ERROR"}, "failed"),
            ({"CANCELLED"}, "cancelled"),
            ({"CANCELLING"}, "cancelling"),
            ({"DELETING"}, "gone"),
            ({"DELETED"}, "gone"),
            ({"SUSPENDING"}, "gone"),
            ({"SUSPENDED"}, "gone"),
            (set(), "gone"),
        ],
    )
    def test_other_states(self, statuses: set[str], state: str) -> None:
        assert job_state(statuses) == state


class TestRunState:
    @pytest.mark.parametrize(
        ("broker", "head", "worker", "state"),
        [
            ("running", "running", "running", "running"),
            ("gone", "done", "done", "succeeded"),
            ("gone", "done", "running", "succeeded"),
            ("running", "failed", "running", "failed"),
            ("running", "done", "failed", "failed"),
            ("cancelling", "running", "running", "cancelling"),
            ("cancelled", "cancelled", "cancelled", "cancelled"),
            ("gone", "gone", "gone", "gone"),
            ("running", "gone", "running", "running"),
        ],
    )
    def test_states(self, broker: str, head: str, worker: str, state: str) -> None:
        assert run_state({"broker": broker, "head": head, "worker": worker}) == state


class TestQualifiedTable:
    @pytest.mark.parametrize(
        "name",
        [
            "arena.training.prompts",
            ' db."My Schema".T$1 ',
            '"db"."s.x"."t""q"',
        ],
    )
    def test_three_part_names_pass(self, name: str) -> None:
        assert qualified_table(name) == name.strip()

    @pytest.mark.parametrize(
        "name",
        ["prompts", "training.prompts", "a.b.c.d", "a..b", "db.s.t; drop", 'db.s."t'],
    )
    def test_other_names_are_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match=r"must be database\.schema\.table"):
            qualified_table(name)


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

        run_id, services = submit_job(session, manifest, settings, run_id=RUN_ID)

        assert (run_id, services) == (RUN_ID, SERVICES)
        assert session.events == [
            "DESCRIBE COMPUTE POOL arena_gpu_GPU_NV_S",
            "put manifest.yaml",
            "put job.json",
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_pulsar_pool",
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_head_pool",
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_gpu_GPU_NV_S",
        ]
        assert session.puts[0]["stage"] == f"@arena.training.runs/{RUN_ID}"
        assert session.puts[0]["auto_compress"] is False
        assert session.puts[0]["overwrite"] is True
        assert yaml.safe_load(session.puts[0]["text"])["algorithm"]["name"] == "SFT"
        document = json.loads(session.puts[1]["text"])
        assert document == job_document(settings, RUN_ID, SERVICES, 2)
        assert document["source"] == {"kind": "table", "name": "arena.training.prompts"}
        assert document["reward_file"] is False
        assert document["resume"] is False
        assert document["nodes"] == document["workers"] == 2
        assert document["gpus"] == 2
        assert document["family"] == "GPU_NV_S"
        assert document["broker_service"] == SERVICES.broker
        assert document["head_service"] == SERVICES.head
        assert document["workers_service"] == SERVICES.workers
        assert "pulsar_url" not in document
        assert "APP_PULSAR_URL" not in session.statements[-1]
        assert "REPLICAS = 2" in session.statements[-1]

    def test_dpo_is_accepted(self, tmp_path: Path) -> None:
        session = Session()

        run_id, services = submit_job(
            session,
            _write(tmp_path, _dpo()),
            _settings(QuerySource("select 1")),
            RUN_ID,
        )

        assert (run_id, services) == (RUN_ID, SERVICES)
        assert json.loads(session.puts[1]["text"])["source"] == {
            "kind": "query",
            "sql": "select 1",
        }

    def test_rl_env_url_warns_about_external_hosts(self, tmp_path: Path) -> None:
        with pytest.warns(UserWarning, match="outside the account"):
            run_id, services = submit_job(
                Session(), _write(tmp_path, _rl()), _settings(None), RUN_ID
            )

        assert (run_id, services) == (RUN_ID, SERVICES)

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

    def test_opd_dataset_rollout_without_a_reward_file_submits(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        session = Session()
        manifest = _dataset_rl()
        manifest["algorithm"] = {"name": "OPD", "batch_size": 2}
        del manifest["environment"]["reward_file_path"]

        # Act
        submit_job(session, _write(tmp_path, manifest), _settings(), RUN_ID)

        # Assert
        assert [item["name"] for item in session.puts] == ["manifest.yaml", "job.json"]
        assert json.loads(session.puts[1]["text"])["reward_file"] is False

    def test_opd_dataset_rollout_that_names_a_rubric_needs_the_reward_file(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        session = Session()
        manifest = _dataset_rl()
        manifest["algorithm"] = {"name": "OPD", "batch_size": 2}

        # Act / Assert
        with pytest.raises(ValueError, match="names no rubric"):
            submit_job(session, _write(tmp_path, manifest), _settings(), RUN_ID)

        assert session.events == []

    def test_opd_dataset_rollout_uploads_a_named_rubric(self, tmp_path: Path) -> None:
        # Arrange
        reward = tmp_path / "my_reward.py"
        reward.write_text("def reward(x):\n    return 1.0\n", encoding="utf-8")
        session = Session()
        manifest = _dataset_rl()
        manifest["algorithm"] = {"name": "OPD", "batch_size": 2}

        # Act
        submit_job(
            session,
            _write(tmp_path, manifest),
            _settings(reward_file=reward),
            RUN_ID,
        )

        # Assert
        assert [item["name"] for item in session.puts] == [
            "manifest.yaml",
            "job.json",
            "reward.py",
        ]
        assert session.puts[2]["text"] == "def reward(x):\n    return 1.0\n"
        assert json.loads(session.puts[1]["text"])["reward_file"] is True

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

    def test_an_omitted_export_stages_a_merged_manifest(self, tmp_path: Path) -> None:
        # Arrange
        session = Session()
        out = tmp_path / "job"
        out.mkdir()
        job_manifest = write_job_manifest(
            _write(tmp_path, _sft(None)), PROMPTS, False, out
        )

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            submit_job(session, job_manifest, _settings(), RUN_ID)

        # Assert
        staged = yaml.safe_load(session.puts[0]["text"])
        assert session.puts[0]["name"] == "manifest.yaml"
        assert staged["training"]["checkpoint_export"] == {
            "format": "merged",
            "trigger": "final",
        }

    def test_an_adapter_export_is_rejected_before_anything_starts(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        session = Session()
        out = tmp_path / "job"
        out.mkdir()
        job_manifest = write_job_manifest(
            _write(tmp_path, _sft("adapter")), PROMPTS, False, out
        )

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Snowflake runs export merged weights for publish. "
                "Remove training.checkpoint_export or set format: merged."
            ),
        ):
            submit_job(session, job_manifest, _settings(), RUN_ID)
        assert session.statements == []
        assert session.puts == []

    def test_an_on_demand_trigger_is_rejected_before_anything_starts(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        document = _sft()
        document["training"]["checkpoint_export"]["trigger"] = "on_demand"
        session = Session()
        out = tmp_path / "job"
        out.mkdir()
        job_manifest = write_job_manifest(
            _write(tmp_path, document), PROMPTS, False, out
        )

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Snowflake runs export merged weights for publish. Remove "
                "training.checkpoint_export or use a trigger other than on_demand."
            ),
        ):
            submit_job(session, job_manifest, _settings(), RUN_ID)
        assert session.statements == []
        assert session.puts == []

    def test_an_explicit_merged_export_is_staged_unchanged(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        session = Session()
        out = tmp_path / "job"
        out.mkdir()
        job_manifest = write_job_manifest(
            _write(tmp_path, _sft("merged")), PROMPTS, False, out
        )

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            submit_job(session, job_manifest, _settings(), RUN_ID)

        # Assert
        staged = yaml.safe_load(session.puts[0]["text"])
        assert staged == _sft("merged")

    def test_an_unknown_gpu_family_is_rejected(self, tmp_path: Path) -> None:
        session = Session()

        with pytest.raises(ValueError, match="Unknown compute pool instance family"):
            submit_job(
                session,
                _write(tmp_path, _sft()),
                _settings(gpu_family="NOT_A_GPU"),
                RUN_ID,
            )

        assert session.statements == []
        assert session.puts == []

    def test_an_unknown_cpu_family_is_rejected(self, tmp_path: Path) -> None:
        session = Session()

        with pytest.raises(ValueError, match="NO_SUCH_CPU"):
            submit_job(
                session,
                _write(tmp_path, _sft()),
                _settings(broker_family="NO_SUCH_CPU"),
                RUN_ID,
            )

        assert session.statements == []
        assert session.puts == []


PREVIOUS_RUN = "arena-20261108120000-def456"


def _listed(*relative: str, size: int = 100) -> list[Row]:
    return [Row(name=f"runs/{PREVIOUS_RUN}/{path}", size=size) for path in relative]


class TestResumeCheckpoint:
    def test_the_newest_uploaded_checkpoint_is_chosen(self) -> None:
        listing = _listed(
            "manifest.yaml",
            "checkpoint_2/model.pt",
            "checkpoint_2/.uploaded",
            "checkpoint_10/model.pt",
            "checkpoint_10/.uploaded",
            "checkpoint_12/model.pt",
        )

        chosen = resume_checkpoint(listing, PREVIOUS_RUN)

        assert chosen == "checkpoint_10"

    def test_upper_case_columns_are_read(self) -> None:
        listing = [
            Row(NAME=f"runs/{PREVIOUS_RUN}/checkpoint_3/.uploaded", SIZE=0),
        ]

        assert resume_checkpoint(listing, PREVIOUS_RUN) == "checkpoint_3"

    @pytest.mark.parametrize(
        "listing",
        [
            [],
            _listed("job.json", ".done"),
            _listed("checkpoint_3/model.pt", "checkpoint_4/model.pt", ".done"),
        ],
    )
    def test_no_uploaded_marker_is_rejected(self, listing: list[Row]) -> None:
        with pytest.raises(
            ValueError, match=r"no checkpoint with an \.uploaded marker"
        ):
            resume_checkpoint(listing, PREVIOUS_RUN)

    def test_a_file_over_the_copy_limit_is_rejected(self) -> None:
        listing = [
            *_listed("checkpoint_5/.uploaded", "checkpoint_5/small.pt"),
            *_listed("checkpoint_5/big.safetensors", size=5 * 10**9 + 1),
        ]

        with pytest.raises(
            ValueError, match=r"up to 5 GB, and these are larger: checkpoint_5/big"
        ):
            resume_checkpoint(listing, PREVIOUS_RUN)

    def test_a_large_file_in_another_checkpoint_is_ignored(self) -> None:
        listing = [
            *_listed("checkpoint_5/.uploaded", "checkpoint_6/model.pt"),
            *_listed("checkpoint_4/big.safetensors", size=6 * 10**9),
        ]

        assert resume_checkpoint(listing, PREVIOUS_RUN) == "checkpoint_5"


class TestCopyResumeCheckpoint:
    def test_lists_the_previous_run_then_copies_into_resume(self) -> None:
        session = Session(
            listing=_listed("checkpoint_4/model.pt", "checkpoint_4/.uploaded")
        )

        chosen = copy_resume_checkpoint(
            session, "arena.training.runs", PREVIOUS_RUN, RUN_ID
        )

        assert chosen == "checkpoint_4"
        assert session.statements == [
            f"LIST @arena.training.runs/{PREVIOUS_RUN}/",
            (
                f"COPY FILES INTO @arena.training.runs/{RUN_ID}/resume/checkpoint_4/ "
                f"FROM @arena.training.runs/{PREVIOUS_RUN}/checkpoint_4/"
            ),
        ]

    def test_a_bad_previous_run_id_runs_no_sql(self) -> None:
        session = Session()

        with pytest.raises(ValueError, match="run_id must look like"):
            copy_resume_checkpoint(session, "arena.training.runs", "last-run", RUN_ID)

        assert session.statements == []


class TestSubmitResume:
    def test_copies_the_checkpoint_before_upload_and_marks_job_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange
        session = Session(
            listing=_listed(
                "checkpoint_1/a", "checkpoint_1/.uploaded", "checkpoint_2/a"
            )
        )
        settings = _settings(resume_from=PREVIOUS_RUN)

        # Act
        submit_job(session, _write(tmp_path, _sft()), settings, RUN_ID)

        # Assert
        assert session.events == [
            "DESCRIBE COMPUTE POOL arena_gpu_GPU_NV_S",
            f"LIST @arena.training.runs/{PREVIOUS_RUN}/",
            (
                f"COPY FILES INTO @arena.training.runs/{RUN_ID}/resume/checkpoint_1/ "
                f"FROM @arena.training.runs/{PREVIOUS_RUN}/checkpoint_1/"
            ),
            "put manifest.yaml",
            "put job.json",
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_pulsar_pool",
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_head_pool",
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_gpu_GPU_NV_S",
        ]
        assert json.loads(session.puts[1]["text"])["resume"] is True
        assert f"resume   {PREVIOUS_RUN}/checkpoint_1" in capsys.readouterr().out

    def test_a_run_without_an_uploaded_checkpoint_starts_nothing(
        self, tmp_path: Path
    ) -> None:
        session = Session(listing=_listed("checkpoint_1/a", ".done"))

        with pytest.raises(
            ValueError, match=r"no checkpoint with an \.uploaded marker"
        ):
            submit_job(
                session,
                _write(tmp_path, _sft()),
                _settings(resume_from=PREVIOUS_RUN),
                RUN_ID,
            )

        assert session.puts == []
        assert session.executes == 0
        assert not any(item.startswith("COPY") for item in session.statements)


def _gpu(family: str, per_node: int, memory: int, max_gpus: int) -> GpuPool:
    return GpuPool(f"arena_gpu_{family}", max_gpus, per_node, memory)


def _accepts(pool: GpuPool) -> bool:
    return True


def _rejects(pool: GpuPool) -> bool:
    raise AssertionError(pool.pool)


class TestChooseGpuFamily:
    def test_omitted_family_picks_the_smallest_memory(self) -> None:
        pools = {
            "GPU_NV_L": _gpu("GPU_NV_L", 8, 40, 16),
            "GPU_NV_M": _gpu("GPU_NV_M", 4, 24, 16),
            "GPU_NV_S": _gpu("GPU_NV_S", 1, 24, 8),
        }

        assert choose_gpu_family(pools, 8, None, _accepts) == "GPU_NV_M"

    def test_a_named_family_that_fits_is_used(self) -> None:
        pools = {
            "GPU_NV_M": _gpu("GPU_NV_M", 4, 24, 16),
            "GPU_NV_L": _gpu("GPU_NV_L", 8, 40, 16),
        }

        assert choose_gpu_family(pools, 8, "GPU_NV_L", _accepts) == "GPU_NV_L"

    def test_a_memory_tie_takes_the_family_on_fewer_nodes(self) -> None:
        pools = {
            "GPU_NV_S": _gpu("GPU_NV_S", 1, 24, 8),
            "GPU_NV_M": _gpu("GPU_NV_M", 4, 24, 16),
        }

        assert choose_gpu_family(pools, 4, None, _accepts) == "GPU_NV_M"
        assert choose_gpu_family(pools, 2, None, _accepts) == "GPU_NV_S"

    def test_a_named_family_the_estimator_cannot_size_warns(self) -> None:
        pools = {"GPU_NV_S": _gpu("GPU_NV_S", 1, 24, 8)}

        def unsized(pool: GpuPool) -> bool:
            msg = "unknown model architecture"
            raise ValueError(msg)

        with pytest.warns(UserWarning, match="GPU_NV_S is used unchecked") as caught:
            chosen = choose_gpu_family(pools, 2, "GPU_NV_S", unsized)

        assert chosen == "GPU_NV_S"
        assert "unknown model architecture" in str(caught[0].message)

    def test_an_omitted_family_the_estimator_cannot_size_raises(self) -> None:
        pools = {"GPU_NV_S": _gpu("GPU_NV_S", 1, 24, 8)}

        def unsized(pool: GpuPool) -> bool:
            msg = "unknown model architecture"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="unknown model architecture"):
            choose_gpu_family(pools, 2, None, unsized)

    def test_omitted_family_skips_a_memory_that_does_not_fit(self) -> None:
        pools = {
            "GPU_NV_L": _gpu("GPU_NV_L", 8, 40, 16),
            "GPU_NV_S": _gpu("GPU_NV_S", 1, 24, 8),
        }

        chosen = choose_gpu_family(
            pools, 8, None, lambda pool: pool.gpu_memory_gib >= 40
        )

        assert chosen == "GPU_NV_L"

    def test_a_named_family_that_does_not_fit_names_its_memory(self) -> None:
        pools = {"GPU_NV_S": _gpu("GPU_NV_S", 1, 24, 8)}

        with pytest.raises(ValueError, match=r"GPU_NV_S \(24 GiB\)") as raised:
            choose_gpu_family(pools, 8, "GPU_NV_S", lambda pool: False)

        assert "cannot hold this manifest" in str(raised.value)

    def test_dividing_families_that_do_not_fit_are_named(self) -> None:
        pools = {
            "GPU_NV_M": _gpu("GPU_NV_M", 4, 24, 16),
            "GPU_NV_S": _gpu("GPU_NV_S", 1, 24, 8),
            "GPU_NV_L": _gpu("GPU_NV_L", 8, 40, 16),
        }

        with pytest.raises(ValueError, match="Too small") as raised:
            choose_gpu_family(pools, 4, None, lambda pool: pool.gpu_memory_gib >= 80)

        text = str(raised.value)
        assert "GPU_NV_M (24 GiB)" in text
        assert "GPU_NV_S (24 GiB)" in text
        assert "GPU_NV_L" not in text

    def test_a_family_that_does_not_divide_lists_valid_counts(self) -> None:
        pools = {"GPU_NV_M": _gpu("GPU_NV_M", 4, 24, 16)}

        with pytest.raises(ValueError, match="4, 8, 12, 16") as raised:
            choose_gpu_family(pools, 6, "GPU_NV_M", _rejects)

        assert "GPU_NV_M does not divide 6 GPUs." in str(raised.value)
        assert "GPU_NV_M runs 4, 8, 12, 16 GPUs." in str(raised.value)

    def test_no_enabled_family_dividing_lists_every_family(self) -> None:
        pools = {
            "GPU_NV_M": _gpu("GPU_NV_M", 4, 24, 12),
            "GPU_NV_L": _gpu("GPU_NV_L", 8, 40, 16),
        }

        with pytest.raises(ValueError, match="No GPU family divides 6 GPUs") as raised:
            choose_gpu_family(pools, 6, None, _rejects)

        text = str(raised.value)
        assert "GPU_NV_M runs 4, 8, 12 GPUs." in text
        assert "GPU_NV_L runs 8, 16 GPUs." in text

    def test_a_missing_family_names_the_enabled_ones(self) -> None:
        pools = {"GPU_NV_S": _gpu("GPU_NV_S", 1, 24, 2)}

        with pytest.raises(ValueError, match="GPU_NV_L is not in this account profile"):
            choose_gpu_family(pools, 8, "GPU_NV_L", _rejects)

    def test_a_family_whose_max_is_below_one_node_runs_no_count(self) -> None:
        pools = {
            "GPU_NV_M": _gpu("GPU_NV_M", 4, 24, 2),
            "GPU_NV_L": _gpu("GPU_NV_L", 8, 40, 16),
        }

        with pytest.raises(ValueError, match="No GPU family divides 6 GPUs") as raised:
            choose_gpu_family(pools, 6, None, _rejects)

        assert str(raised.value) == (
            "No GPU family divides 6 GPUs.\n"
            "GPU_NV_L runs 8, 16 GPUs.\n"
            "GPU_NV_M runs no GPU count within its max."
        )

    def test_a_profile_without_gpu_pools_asks_for_setup(self) -> None:
        with pytest.raises(ValueError, match="no GPU pool") as raised:
            choose_gpu_family({}, 2, None, _rejects)

        assert "Run arena snowflake setup again." in str(raised.value)


class TestManifestFits:
    def test_a_manifest_within_the_gpu_memory_fits(self, tmp_path: Path) -> None:
        pool = _gpu("GPU_NV_SL", 1, 80, 8)

        assert manifest_fits(_write(tmp_path, _sft()), pool, 1) is True

    def test_a_manifest_over_the_gpu_memory_does_not_fit(self, tmp_path: Path) -> None:
        pool = _gpu("GPU_NV_S", 1, 1, 8)

        assert manifest_fits(_write(tmp_path, _sft()), pool, 1) is False

    def test_the_estimate_lays_the_run_out_on_the_family_nodes(
        self, tmp_path: Path
    ) -> None:
        # Arrange: eight ranks on one node hold more NCCL buffers per GPU than
        # eight one-GPU nodes, which tips a 4 GiB card over the buffer.
        manifest = _write(tmp_path, _sft())
        one_node = _gpu("GPU_NV_L", 8, 4, 16)
        eight_nodes = _gpu("GPU_NV_S", 1, 4, 8)

        # Act
        on_one_node = manifest_fits(manifest, one_node, 8)
        on_eight_nodes = manifest_fits(manifest, eight_nodes, 8)

        # Assert
        assert (on_one_node, on_eight_nodes) == (False, True)

    def test_an_invalid_manifest_raises_the_estimator_message(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text("algorithm: [\n", encoding="utf-8")

        with pytest.raises(ValueError, match="could not validate"):
            manifest_fits(path, _gpu("GPU_NV_S", 1, 24, 8), 1)


class TestSubmitServices:
    def test_three_specs_and_the_banner(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = Session()
        settings = _settings(
            gpus=8,
            gpu_family="GPU_NV_M",
            gpus_per_node=4,
            gpu_pool="arena_gpu_GPU_NV_M",
        )

        submit_job(session, _write(tmp_path, _sft()), settings, RUN_ID)

        executes = [item for item in session.statements if item.startswith("EXECUTE")]
        assert [item.split("\n", 1)[0] for item in executes] == [
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_pulsar_pool",
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_head_pool",
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_gpu_GPU_NV_M",
        ]
        broker, head, workers = (_spec(item) for item in executes)
        broker_resources = broker["spec"]["containers"][0]["resources"]["requests"]
        head_resources = head["spec"]["containers"][0]["resources"]["requests"]
        worker_resources = workers["spec"]["containers"][0]["resources"]["requests"]
        assert "nvidia.com/gpu" not in broker_resources
        assert "nvidia.com/gpu" not in head_resources
        assert worker_resources["nvidia.com/gpu"] == 4
        assert "REPLICAS = 1" in executes[0]
        assert "REPLICAS = 1" in executes[1]
        assert "REPLICAS = 2" in executes[2]
        assert "EXTERNAL_ACCESS_INTEGRATIONS" not in executes[0]
        assert (
            "EXTERNAL_ACCESS_INTEGRATIONS = (arena_hf_eai, arena_api_eai)"
            in executes[1]
        )
        assert "EXTERNAL_ACCESS_INTEGRATIONS = (arena_hf_eai)" in executes[2]
        block = next(
            item for item in broker["spec"]["volumes"] if item["source"] == "block"
        )
        assert block["blockConfig"] == {"iops": 16000, "throughput": 1000}
        assert {"name": "data", "mountPath": "/pulsar-data"} in broker["spec"][
            "containers"
        ][0]["volumeMounts"]
        assert head["spec"]["endpoints"] == service_endpoints(None)
        assert "APP_PULSAR_URL" not in executes[2]
        assert "non-persistent://arena/run" in executes[2]
        banner = capsys.readouterr().out
        assert "broker   arena_pulsar_pool  CPU_X64_L" in banner
        assert "head     arena_head_pool  CPU_X64_SL" in banner
        assert "workers  8 GPUs on 2 workers of GPU_NV_M  arena_gpu_GPU_NV_M" in banner

    def test_a_passed_arena_api_integration_is_attached_once(
        self, tmp_path: Path
    ) -> None:
        session = Session()
        settings = _settings(
            external_access_integrations=("arena_api_eai", "arena_hf_eai")
        )

        submit_job(session, _write(tmp_path, _sft()), settings, RUN_ID)

        executes = [item for item in session.statements if item.startswith("EXECUTE")]
        assert (
            "EXTERNAL_ACCESS_INTEGRATIONS = (arena_api_eai, arena_hf_eai)"
            in executes[1]
        )
        assert executes[1].count("arena_api_eai") == 1

    def test_a_failed_third_statement_drops_the_first_two(self, tmp_path: Path) -> None:
        session = Session(fail_on_execute=3)

        with pytest.raises(RuntimeError, match="execute failed"):
            submit_job(session, _write(tmp_path, _sft()), _settings(), RUN_ID)

        drops = [item for item in session.statements if item.startswith("DROP")]
        assert drops == [
            f"DROP SERVICE {SERVICES.head} FORCE",
            f"DROP SERVICE {SERVICES.broker} FORCE",
        ]

    def test_workers_above_max_nodes_stops_before_execute(self, tmp_path: Path) -> None:
        session = Session(max_nodes=1)
        settings = _settings(
            gpus=8,
            gpu_family="GPU_NV_M",
            gpus_per_node=4,
            gpu_pool="arena_gpu_GPU_NV_M",
        )

        with pytest.raises(
            ValueError, match="pools set GPU_NV_M --max-gpus 8"
        ) as raised:
            submit_job(session, _write(tmp_path, _sft()), settings, RUN_ID)

        assert "MAX_NODES 1 on arena_gpu_GPU_NV_M" in str(raised.value)
        assert session.puts == []
        assert any(item.startswith("DESCRIBE") for item in session.statements)
        assert not any(item.startswith("EXECUTE") for item in session.statements)

    def test_head_mounts_arena_and_job_json_omits_the_key(self, tmp_path: Path) -> None:
        # Arrange
        session = Session()
        settings = _settings()

        # Act
        submit_job(session, _write(tmp_path, _sft()), settings, RUN_ID)

        # Assert
        executes = [item for item in session.statements if item.startswith("EXECUTE")]
        broker, head, workers = (_spec(item) for item in executes)
        job_text = next(
            item["text"] for item in session.puts if item["name"] == "job.json"
        )
        head_secrets = head["spec"]["containers"][0]["secrets"]
        worker = workers["spec"]["containers"][0]
        broker_container = broker["spec"]["containers"][0]

        assert "arena_org_" not in job_text
        assert head_secrets == [
            {
                "snowflakeSecret": "arena.training.arena_org_key",
                "secretKeyRef": "secret_string",
                "envVarName": "ARENA_ORG_KEY",
            },
            {
                "snowflakeSecret": "arena.training.arena_external_user_id",
                "secretKeyRef": "secret_string",
                "envVarName": "ARENA_EXTERNAL_USER_ID",
            },
            {
                "snowflakeSecret": "arena.training.arena_base_url",
                "secretKeyRef": "secret_string",
                "envVarName": "ARENA_BASE_URL",
            },
        ]
        assert (
            "EXTERNAL_ACCESS_INTEGRATIONS = (arena_hf_eai, arena_api_eai)"
            in (executes[1])
        )
        assert "secrets" not in worker
        assert "arena_org_" not in executes[2]
        assert "arena_base_url" not in executes[2]
        assert "ARENA_BASE_URL" not in executes[2]
        assert "EXTERNAL_ACCESS_INTEGRATIONS = (arena_hf_eai)" in executes[2]
        assert "arena_api_eai" not in executes[2]
        assert "secrets" not in broker_container
        assert "EXTERNAL_ACCESS_INTEGRATIONS" not in executes[0]


class TestDescribeMaxNodes:
    def test_reads_max_nodes(self) -> None:
        session = Session(rows=[Row(max_nodes="4")])

        assert describe_max_nodes(session, "arena_gpu_GPU_NV_S") == 4

    def test_a_row_without_max_nodes_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_nodes"):
            describe_max_nodes(Session(rows=[Row(name="pool")]), "pool")

    def test_a_missing_pool_is_rejected(self) -> None:
        session = Session(rows=[])

        with pytest.raises(ValueError, match="'arena_gpu_pool' was not found"):
            describe_max_nodes(session, "arena_gpu_pool")

        assert session.statements == ["DESCRIBE COMPUTE POOL arena_gpu_pool"]


class TestCancelRun:
    def test_drops_the_three_services(self) -> None:
        session = Session()

        cancel_run(session, RUN_ID)

        assert session.statements == [
            f"DROP SERVICE IF EXISTS {SERVICES.broker} FORCE",
            f"DROP SERVICE IF EXISTS {SERVICES.head} FORCE",
            f"DROP SERVICE IF EXISTS {SERVICES.workers} FORCE",
        ]
