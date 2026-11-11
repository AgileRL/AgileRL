# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""arena snowflake publish: client checks, the request file, and the job service."""

from __future__ import annotations

import dataclasses
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from agilerl.arena.cli import main
from agilerl.arena.snowflake.publish import (
    PublishDocument,
    PublishSettings,
    publish_job_name,
    select_checkpoint,
    service_endpoint,
    submit_publish,
    wait_for_publish,
)

RUN_ID = "arena-20261109163457-abc123"
PREFIX = f"@arena.training.runs/{RUN_ID}"
NOW = datetime(2026, 11, 10, 12, 0, 5, tzinfo=timezone.utc)
GIB = 2**30
RUN_DOCUMENT = {
    "compute_pool": "arena_gpu_pool",
    "image": "/arena/training/images/snowflake-job:1.0",
    "run_id": RUN_ID,
    "runs_stage": "arena.training.runs",
    "warehouse": "arena_wh",
}
SERVICE_DOCUMENT: PublishDocument = {
    "checkpoint": "checkpoint_10",
    "compute_pool": "arena_gpu_pool",
    "database": "ARENA",
    "gpu_count": 1,
    "max_model_len": None,
    "model_name": "arena_policy",
    "run_id": RUN_ID,
    "schema": "TRAINING",
    "service_name": "policy_svc",
    "step": 10,
    "version_name": "checkpoint_10",
}
PUBLISH_ARGS = [
    "snowflake",
    "publish",
    RUN_ID,
    "--runs-stage",
    "arena.training.runs",
    "--model-name",
    "arena_policy",
]


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
        return list(self.rows)


class StageFiles:
    """Stage holding the run's ``job.json``. Uploads keep each file's text."""

    def __init__(self) -> None:
        self.staged = {f"{PREFIX}/job.json": json.dumps(RUN_DOCUMENT).encode()}
        self.uploads: dict[str, str] = {}

    def put(
        self,
        local_file_name: str,
        stage_location: str,
        auto_compress: bool = False,
        overwrite: bool = False,
    ) -> None:
        path = Path(local_file_name)
        self.uploads[f"{stage_location}/{path.name}"] = path.read_text(encoding="utf-8")

    def get_stream(self, stage_location: str) -> io.BytesIO:
        return io.BytesIO(self.staged[stage_location])


class SnowflakeAccount:
    """Session over a fake account.

    ``statuses`` holds one poll's ``service_status`` values. Containers report
    ``DONE`` throughout, so only the service status can end a wait.
    """

    def __init__(
        self,
        listing: list[Row],
        models: tuple[str, ...] = (),
        versions: tuple[str, ...] = (),
        services: tuple[str, ...] = (),
        family: str = "GPU_NV_M",
        current: tuple[str | None, str | None] = ("ARENA", "TRAINING"),
        statuses: list[list[str]] | None = None,
        logs: str = "",
        endpoints: list[Row] | None = None,
    ) -> None:
        self.listing = listing
        self.models = models
        self.versions = versions
        self.services = services
        self.family = family
        self.current = current
        self.statuses = statuses or [["DONE"]]
        self.logs = logs
        self.endpoints = endpoints or []
        self.file = StageFiles()
        self.statements: list[str] = []
        self.closed = False

    def sql(self, query: str) -> Result:
        self.statements.append(query)
        return Result(self._rows(query))

    def table(self, name: str) -> Result:
        raise AssertionError(name)

    def close(self) -> None:
        self.closed = True

    def _rows(self, query: str) -> list[Row]:
        if query.startswith("SELECT CURRENT_DATABASE()"):
            return [Row(DATABASE_NAME=self.current[0], SCHEMA_NAME=self.current[1])]
        if query.startswith("LIST "):
            return self.listing
        if query.startswith("SHOW MODELS"):
            return [Row(name=name) for name in self.models]
        if query.startswith("SHOW VERSIONS"):
            return [Row(name=name) for name in self.versions]
        if query.startswith("SHOW SERVICES"):
            return [Row(name=name) for name in self.services]
        if query.startswith("DESCRIBE COMPUTE POOL"):
            return [Row(instance_family=self.family)]
        if query.startswith("SHOW SERVICE CONTAINERS"):
            return [
                Row(
                    instance_id=0,
                    container_name="main",
                    status="DONE",
                    service_status=status,
                )
                for status in self.statuses.pop(0)
            ]
        if query.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"):
            return [Row(**{"SYSTEM$GET_SERVICE_LOGS": self.logs})]
        if query.startswith("SHOW ENDPOINTS"):
            return self.endpoints
        return []

    def published(self, job_name: str) -> dict:
        return json.loads(self.file.uploads[f"{PREFIX}/publish-{job_name}.json"])

    def job_spec(self) -> dict:
        statement = next(s for s in self.statements if s.startswith("EXECUTE JOB"))
        return yaml.safe_load(statement.split("$$")[1])


def staged(*files: tuple[str, int]) -> list[Row]:
    """``LIST`` rows for files under the run directory."""
    return [
        Row(name=f"runs/{RUN_ID}/{path}", size=size, md5="", last_modified="")
        for path, size in files
    ]


def two_checkpoints() -> list[Row]:
    return staged(
        ("manifest.yaml", 100),
        ("job.json", 100),
        ("metrics.jsonl", 100),
        ("checkpoint_9/merged/model.safetensors", 3 * GIB),
        ("checkpoint_9/merged/.complete", 0),
        ("checkpoint_10/attempt_0/policy.pt", 9 * GIB),
        ("checkpoint_10/merged/model.safetensors", 3 * GIB - 1024),
        ("checkpoint_10/merged/tokenizer.json", 1024),
        ("checkpoint_10/merged/.complete", 0),
    )


def settings(**overrides: object) -> PublishSettings:
    values = PublishSettings(
        run_id=RUN_ID,
        runs_stage="arena.training.runs",
        model_name="arena_policy",
        version_name=None,
        step=None,
        serve=True,
        compute_pool=None,
        image=None,
        warehouse=None,
        gpu_count=1,
        max_model_len=None,
        service_name=None,
        database=None,
        schema=None,
    )
    return dataclasses.replace(values, **overrides)


class TestSubmitPublish:
    def test_writes_the_request_and_starts_the_job(self) -> None:
        # Arrange
        account = SnowflakeAccount(two_checkpoints(), models=("ARENA_POLICY",))

        # Act
        job = submit_publish(account, settings(), now=NOW)

        # Assert
        assert job.name.startswith("ARENA_PUBLISH_20261110120005_")
        assert list(account.file.uploads) == [f"{PREFIX}/publish-{job.name}.json"]
        assert account.published(job.name) == {
            "checkpoint": "checkpoint_10",
            "compute_pool": "arena_gpu_pool",
            "database": "ARENA",
            "gpu_count": 1,
            "max_model_len": None,
            "model_name": "arena_policy",
            "run_id": RUN_ID,
            "schema": "TRAINING",
            "service_name": "arena_policy_checkpoint_10",
            "step": 10,
            "version_name": "checkpoint_10",
        }
        assert job.document == account.published(job.name)
        statement = account.statements[-1]
        assert statement.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_gpu_pool\n"
            f"NAME = {job.name}\nASYNC = TRUE\nQUERY_WAREHOUSE = arena_wh\n"
        )
        container = account.job_spec()["spec"]["containers"][0]
        assert container["image"] == "/arena/training/images/snowflake-job:1.0"
        assert container["command"] == [
            "tini",
            "--",
            "python",
            "-m",
            "agilerl.arena.snowflake.publish_job",
        ]
        assert container["args"] == [f"/arena/run/publish-{job.name}.json"]
        assert container["resources"] == {
            "requests": {"memory": "7Gi"},
            "limits": {"memory": "177.5Gi"},
        }
        assert account.job_spec()["spec"]["volumes"] == [
            {"name": "run", "source": "stage", "stageConfig": {"name": PREFIX}}
        ]

    def test_overrides_replace_the_run_settings(self) -> None:
        account = SnowflakeAccount(two_checkpoints())

        job = submit_publish(
            account,
            settings(
                step=9,
                version_name="v9",
                service_name="policy_svc",
                compute_pool="serve_pool",
                image="/arena/training/images/snowflake-job:2.0",
                warehouse="other_wh",
                gpu_count=4,
                max_model_len=4096,
                database="ml",
                schema="registry",
            ),
        )

        assert {
            key: job.document[key]
            for key in (
                "checkpoint",
                "step",
                "version_name",
                "service_name",
                "compute_pool",
                "gpu_count",
                "max_model_len",
                "database",
                "schema",
            )
        } == {
            "checkpoint": "checkpoint_9",
            "step": 9,
            "version_name": "v9",
            "service_name": "policy_svc",
            "compute_pool": "serve_pool",
            "gpu_count": 4,
            "max_model_len": 4096,
            "database": "ml",
            "schema": "registry",
        }
        assert "IN COMPUTE POOL serve_pool\n" in account.statements[-1]
        assert "QUERY_WAREHOUSE = other_wh\n" in account.statements[-1]
        assert "SHOW SERVICES LIKE 'policy_svc' IN SCHEMA ml.registry" in (
            account.statements
        )
        container = account.job_spec()["spec"]["containers"][0]
        assert container["image"] == "/arena/training/images/snowflake-job:2.0"

    def test_no_serve_names_no_service(self) -> None:
        account = SnowflakeAccount(
            two_checkpoints(), services=("ARENA_POLICY_CHECKPOINT_10",)
        )

        job = submit_publish(account, settings(serve=False, gpu_count=8))

        assert job.document["service_name"] is None
        assert account.published(job.name)["service_name"] is None

    def test_newest_incomplete_checkpoint_is_skipped(self) -> None:
        account = SnowflakeAccount(
            [
                *two_checkpoints(),
                *staged(("checkpoint_11/merged/model.safetensors", 3 * GIB)),
            ]
        )

        job = submit_publish(account, settings())

        assert job.document["checkpoint"] == "checkpoint_10"
        assert job.document["step"] == 10
        assert account.published(job.name)["step"] == 10

    def test_concurrent_publishes_write_separate_requests(self) -> None:
        # Arrange
        account = SnowflakeAccount(two_checkpoints())

        # Act
        first = submit_publish(account, settings(serve=False), now=NOW)
        second = submit_publish(
            account, settings(serve=False, step=9, version_name="v9"), now=NOW
        )

        # Assert
        assert first.name != second.name
        assert account.published(first.name)["checkpoint"] == "checkpoint_10"
        assert account.published(second.name)["checkpoint"] == "checkpoint_9"
        args = [
            yaml.safe_load(statement.split("$$")[1])["spec"]["containers"][0]["args"]
            for statement in account.statements
            if statement.startswith("EXECUTE JOB")
        ]
        assert args == [
            [f"/arena/run/publish-{first.name}.json"],
            [f"/arena/run/publish-{second.name}.json"],
        ]

    def test_existing_version_is_rejected(self) -> None:
        account = SnowflakeAccount(
            two_checkpoints(), models=("ARENA_POLICY",), versions=("CHECKPOINT_10",)
        )

        with pytest.raises(ValueError, match="already has version checkpoint_10"):
            submit_publish(account, settings())

        assert account.file.uploads == {}

    def test_like_wildcard_matches_are_not_existing_names(self) -> None:
        account = SnowflakeAccount(
            two_checkpoints(),
            models=("ARENA_POLICY",),
            versions=("CHECKPOINTX10",),
            services=("ARENA_POLICYXCHECKPOINT_10",),
        )

        job = submit_publish(account, settings())

        assert job.document["version_name"] == "checkpoint_10"

    def test_existing_service_is_rejected(self) -> None:
        account = SnowflakeAccount(
            two_checkpoints(), services=("ARENA_POLICY_CHECKPOINT_10",)
        )

        with pytest.raises(ValueError, match="already exists"):
            submit_publish(account, settings())

        assert account.file.uploads == {}
        assert not any(s.startswith("EXECUTE") for s in account.statements)

    @pytest.mark.parametrize(("database", "schema"), [("ml", None), (None, "registry")])
    def test_database_and_schema_are_set_together(
        self, database: str | None, schema: str | None
    ) -> None:
        account = SnowflakeAccount(two_checkpoints())

        with pytest.raises(ValueError, match="both --database and --schema"):
            submit_publish(account, settings(database=database, schema=schema))

        assert account.statements == []

    def test_session_without_a_schema_is_rejected(self) -> None:
        account = SnowflakeAccount(two_checkpoints(), current=("ARENA", None))

        with pytest.raises(ValueError, match="no current schema"):
            submit_publish(account, settings())

    @pytest.mark.parametrize(
        "unsafe",
        [
            {"model_name": "policy; DROP TABLE t"},
            {"model_name": "ml.policy"},
            {"version_name": "v1' OR '1"},
            {"service_name": "svc$$x"},
            {"database": "ml;x", "schema": "registry"},
            {"compute_pool": "pool name"},
            {"warehouse": "wh'"},
        ],
    )
    def test_unsafe_names_are_rejected_before_anything_is_written(
        self, unsafe: dict[str, str]
    ) -> None:
        account = SnowflakeAccount(two_checkpoints())

        with pytest.raises(ValueError, match="Snowflake identifier"):
            submit_publish(account, settings(**unsafe))

        assert account.file.uploads == {}
        assert not any(s.startswith("EXECUTE") for s in account.statements)

    def test_more_gpus_than_a_node_is_rejected(self) -> None:
        account = SnowflakeAccount(two_checkpoints(), family="GPU_NV_S")

        with pytest.raises(ValueError, match="more than the 1 GPUs"):
            submit_publish(account, settings(gpu_count=2))

    def test_image_that_breaks_the_sql_quote_is_rejected(self) -> None:
        account = SnowflakeAccount(two_checkpoints())

        with pytest.raises(ValueError, match=r"cannot contain \$\$"):
            submit_publish(account, settings(image="/a/b/c/job:$$"))

        assert account.file.uploads == {}

    def test_a_checkpoint_larger_than_a_node_is_rejected(self) -> None:
        account = SnowflakeAccount(
            staged(
                ("checkpoint_3/merged/model.safetensors", 40 * GIB),
                ("checkpoint_3/merged/.complete", 0),
            ),
            family="GPU_NV_S",
        )

        with pytest.raises(ValueError, match="needs about 44 GiB"):
            submit_publish(account, settings())


class TestSelectCheckpoint:
    def test_highest_step_is_compared_as_a_number(self) -> None:
        assert select_checkpoint(two_checkpoints(), RUN_ID, None) == (
            "checkpoint_10",
            10,
            3 * GIB,
        )

    def test_step_selects_that_checkpoint(self) -> None:
        assert select_checkpoint(two_checkpoints(), RUN_ID, 9) == (
            "checkpoint_9",
            9,
            3 * GIB,
        )

    def test_padded_directory_name_is_kept(self) -> None:
        listing = staged(
            ("checkpoint_007/merged/model.safetensors", 10),
            ("checkpoint_007/merged/.complete", 0),
        )

        assert select_checkpoint(listing, RUN_ID, 7) == ("checkpoint_007", 7, 10)

    def test_only_the_exact_merged_directory_counts(self) -> None:
        listing = [
            *two_checkpoints(),
            *staged(
                ("checkpoint_12/merged_lora/.complete", 0),
                ("checkpoint_13/merged-old/.complete", 0),
                ("checkpoint_14/merged/model.safetensors", 10),
                ("other/checkpoint_15/merged/.complete", 0),
            ),
        ]

        assert select_checkpoint(listing, RUN_ID, None)[0] == "checkpoint_10"

    def test_no_complete_merged_export_is_rejected(self) -> None:
        listing = staged(("checkpoint_4/attempt_0/policy.pt", 10))

        with pytest.raises(ValueError, match="no complete merged checkpoint"):
            select_checkpoint(listing, RUN_ID, None)

    def test_step_without_a_complete_export_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="checkpoint 11 has no complete merged"):
            select_checkpoint(two_checkpoints(), RUN_ID, 11)


class TestWaitForPublish:
    def test_returns_once_the_job_is_done(self) -> None:
        account = SnowflakeAccount(
            [], statuses=[[], ["PENDING"], ["RUNNING"], ["DONE"]]
        )

        wait_for_publish(account, "ARENA_PUBLISH_X", sleep=lambda seconds: None)

        assert account.statuses == []
        assert account.statements[-1] == (
            "SHOW SERVICE CONTAINERS IN SERVICE ARENA_PUBLISH_X"
        )

    def test_failure_raises_with_the_job_log(self) -> None:
        account = SnowflakeAccount(
            [], statuses=[["FAILED"]], logs="Traceback: no .complete"
        )

        with pytest.raises(RuntimeError, match="ARENA_PUBLISH_X failed:\nTraceback"):
            wait_for_publish(account, "ARENA_PUBLISH_X", sleep=lambda seconds: None)


class TestServiceEndpoint:
    def test_returns_the_inference_endpoint_by_name(self) -> None:
        account = SnowflakeAccount(
            [],
            endpoints=[
                Row(name="metrics", ingress_url="m.snowflakecomputing.app"),
                Row(name="inference", ingress_url="abc.snowflakecomputing.app"),
            ],
        )

        endpoint = service_endpoint(account, SERVICE_DOCUMENT)

        assert endpoint == "abc.snowflakecomputing.app"
        assert account.statements == [
            "SHOW ENDPOINTS IN SERVICE ARENA.TRAINING.policy_svc"
        ]

    def test_missing_inference_endpoint_is_rejected(self) -> None:
        account = SnowflakeAccount([], endpoints=[Row(name="metrics", ingress_url="")])

        with pytest.raises(ValueError, match="has no inference endpoint"):
            service_endpoint(account, SERVICE_DOCUMENT)

    def test_publish_without_a_service_is_rejected(self) -> None:
        document: PublishDocument = {**SERVICE_DOCUMENT, "service_name": None}

        with pytest.raises(ValueError, match="started no service"):
            service_endpoint(SnowflakeAccount([]), document)


class TestPublishJobName:
    def test_name_is_stamped_and_unquoted(self) -> None:
        name = publish_job_name(NOW)

        assert name.startswith("ARENA_PUBLISH_20261110120005_")
        assert len(name) == len("ARENA_PUBLISH_20261110120005_ABCDEF")
        assert name == name.upper()


class TestPublishCommand:
    def test_help_works_without_the_snowflake_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "snowflake", None)

        result = CliRunner().invoke(main, ["snowflake", "publish", "--help"])

        assert result.exit_code == 0, result.output
        assert "--runs-stage" in result.output
        assert "--gpu-count" in result.output

    def test_snowflake_help_lists_publish(self) -> None:
        result = CliRunner().invoke(main, ["snowflake", "--help"])

        assert result.exit_code == 0
        assert "publish" in result.output

    def test_prints_job_model_version_and_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        account = SnowflakeAccount(
            two_checkpoints(),
            endpoints=[Row(name="inference", ingress_url="abc.snowflakecomputing.app")],
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.publish_cli.open_session", lambda name: account
        )

        result = CliRunner().invoke(main, [*PUBLISH_ARGS, "--serve"])

        assert result.exit_code == 0, result.output
        lines = result.output.splitlines()
        assert lines[0].startswith("job: ARENA_PUBLISH_")
        assert lines[1:] == [
            "model: ARENA.TRAINING.arena_policy",
            "version: checkpoint_10",
            "service: arena_policy_checkpoint_10",
            "endpoint: abc.snowflakecomputing.app",
        ]
        assert account.closed

    def test_no_serve_prints_no_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        account = SnowflakeAccount(two_checkpoints())
        monkeypatch.setattr(
            "agilerl.arena.snowflake.publish_cli.open_session", lambda name: account
        )

        result = CliRunner().invoke(main, PUBLISH_ARGS)

        assert result.exit_code == 0, result.output
        assert result.output.splitlines()[1:] == [
            "model: ARENA.TRAINING.arena_policy",
            "version: checkpoint_10",
        ]

    def test_errors_are_shown_without_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        account = SnowflakeAccount(
            two_checkpoints(), statuses=[["FAILED"]], logs="OSError: disk full"
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.publish_cli.open_session", lambda name: account
        )

        half_set = CliRunner().invoke(main, [*PUBLISH_ARGS, "--schema", "registry"])
        failed = CliRunner().invoke(main, PUBLISH_ARGS)

        assert half_set.exit_code == 1
        assert "Pass both --database and --schema" in half_set.output
        assert failed.exit_code == 1
        assert "OSError: disk full" in failed.output
        assert account.closed

    def test_missing_snowflake_extra_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "snowflake", None)
        monkeypatch.setitem(sys.modules, "snowflake.snowpark", None)

        result = CliRunner().invoke(main, PUBLISH_ARGS)

        assert result.exit_code == 1
        assert "agilerl-arena[snowflake]" in result.output
