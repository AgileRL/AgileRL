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
from agilerl.arena.snowflake.profile import (
    AccountPools,
    AccountProfile,
    CpuPool,
    GpuPool,
    admin_connection_fields,
    job_connection_fields,
    write_connection,
)
from agilerl.arena.snowflake.publish import (
    PublishDocument,
    PublishJob,
    PublishSettings,
    ServedVersion,
    publish_job_name,
    select_checkpoint,
    served_version,
    service_endpoint,
    submit_publish,
    unpublish,
    version_endpoint,
    wait_for_publish,
)
from agilerl.arena.snowflake.publish_cli import endpoint_command, unpublish_command

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
    "image_build_compute_pool": "arena_head_pool",
    "max_model_len": None,
    "model_name": "arena_policy",
    "run_id": RUN_ID,
    "schema": "TRAINING",
    "service_name": "policy_svc",
    "step": 10,
    "version_name": "checkpoint_10",
}
SERVICE_JOB = PublishJob(name="ARENA_PUBLISH_X", document=SERVICE_DOCUMENT)
POLICY_SERVICE = "ARENA.TRAINING.ARENA_POLICY_CHECKPOINT_10"
PUBLISH_ARGS = [
    "snowflake",
    "publish",
    RUN_ID,
    "--runs-stage",
    "arena.training.runs",
    "--model-name",
    "arena_policy",
]


@pytest.fixture(autouse=True)
def no_local_connections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's connections.toml out of command tests."""
    monkeypatch.setattr(
        "agilerl.arena.snowflake.session.connections_file",
        lambda: tmp_path / "connections.toml",
    )


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
    ``DONE`` throughout, so only the service status can end a wait. ``setup_doc``
    is served as setup.json.
    """

    setup_doc: dict[str, object] | None = None

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
        model_versions: list[Row] | None = None,
    ) -> None:
        self.listing = listing
        self.models = models
        self.versions = versions
        self.model_versions = model_versions or []
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
        if query.startswith("SELECT $1 AS DOC FROM @arena.training.runs/_arena/"):
            if self.setup_doc is None:
                return []
            return [Row(DOC=json.dumps(self.setup_doc))]
        if query.startswith("SELECT CURRENT_DATABASE()"):
            return [Row(DATABASE_NAME=self.current[0], SCHEMA_NAME=self.current[1])]
        if query.startswith("LIST "):
            return self.listing
        if query.startswith("SHOW MODELS"):
            return [Row(name=name) for name in self.models]
        if query.startswith("SHOW VERSIONS IN MODEL"):
            return self.model_versions
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


def version_row(name: str, services: list[str], is_default: bool = False) -> Row:
    """``SHOW VERSIONS IN MODEL`` row as Snowflake stores it."""
    return Row(
        name=name.upper(),
        is_default_version=is_default,
        inference_services=json.dumps(services),
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
        image_build_pool="arena_head_pool",
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
            "image_build_compute_pool": "arena_head_pool",
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
            f"/arena/run/publish-{job.name}.json",
        ]
        assert "args" not in container
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
                image_build_pool="build_pool",
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
                "image_build_compute_pool",
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
            "image_build_compute_pool": "build_pool",
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
        request_files = [
            yaml.safe_load(statement.split("$$")[1])["spec"]["containers"][0][
                "command"
            ][-1]
            for statement in account.statements
            if statement.startswith("EXECUTE JOB")
        ]
        assert request_files == [
            f"/arena/run/publish-{first.name}.json",
            f"/arena/run/publish-{second.name}.json",
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
            {"image_build_pool": "pool;x"},
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


class Clock:
    """Monotonic clock that moves only when the wait sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestWaitForPublish:
    def test_prints_each_status_change_until_done(self) -> None:
        # Arrange
        account = SnowflakeAccount(
            [],
            statuses=[[], ["PENDING"], ["PENDING"], ["RUNNING"], ["RUNNING"], ["DONE"]],
        )
        clock = Clock()
        shown: list[str] = []

        # Act
        wait_for_publish(
            account, SERVICE_JOB, echo=shown.append, sleep=clock.sleep, clock=clock
        )

        # Assert
        assert shown == [
            "status: no containers yet",
            "status: PENDING",
            "status: RUNNING",
            "status: DONE",
        ]
        assert account.statuses == []
        assert account.statements[-1] == (
            "SHOW SERVICE CONTAINERS IN SERVICE ARENA_PUBLISH_X"
        )

    @pytest.mark.parametrize("status", ["FAILED", "INTERNAL_ERROR"])
    def test_terminal_state_stops_with_the_job_log(self, status: str) -> None:
        account = SnowflakeAccount(
            [], statuses=[["RUNNING"], [status]], logs="Traceback: no .complete"
        )
        clock = Clock()
        shown: list[str] = []

        with pytest.raises(RuntimeError, match="ARENA_PUBLISH_X failed:\nTraceback"):
            wait_for_publish(
                account, SERVICE_JOB, echo=shown.append, sleep=clock.sleep, clock=clock
            )

        assert shown == ["status: RUNNING", f"status: {status}"]
        assert account.statuses == []

    def test_deadline_stops_the_wait_and_says_how_to_check(self) -> None:
        # Arrange
        account = SnowflakeAccount([], statuses=[["PENDING"]] * 13)
        clock = Clock()

        # Act
        with pytest.raises(RuntimeError) as raised:
            wait_for_publish(
                account,
                SERVICE_JOB,
                echo=lambda line: None,
                sleep=clock.sleep,
                clock=clock,
                poll_s=10.0,
                timeout_s=120.0,
            )

        # Assert
        assert str(raised.value).splitlines() == [
            (
                "Publish job ARENA_PUBLISH_X is still PENDING after 2 minutes. "
                "It keeps running in Snowflake."
            ),
            "Status: SHOW SERVICE CONTAINERS IN SERVICE ARENA_PUBLISH_X",
            "Logs: SELECT SYSTEM$GET_SERVICE_LOGS('ARENA_PUBLISH_X', 0, 'main')",
            (
                "Endpoint, once the job is done: arena snowflake endpoint arena_policy "
                "--version checkpoint_10 --database ARENA --schema TRAINING"
            ),
        ]
        assert clock.now == 120.0
        assert account.statuses == []

    def test_deadline_without_a_service_has_no_endpoint_hint(self) -> None:
        account = SnowflakeAccount([], statuses=[["RUNNING"]])
        job = PublishJob(
            name="ARENA_PUBLISH_X",
            document={**SERVICE_DOCUMENT, "service_name": None},
        )

        with pytest.raises(RuntimeError) as raised:
            wait_for_publish(account, job, echo=lambda line: None, timeout_s=0.0)

        assert "endpoint" not in str(raised.value)


def served(**overrides: object) -> ServedVersion:
    values = ServedVersion(
        model="ARENA.TRAINING.arena_policy",
        version_name="CHECKPOINT_10",
        services=(POLICY_SERVICE,),
        is_default=True,
        only_version=True,
    )
    return dataclasses.replace(values, **overrides)


class TestServedVersion:
    def test_omitted_version_takes_the_one_with_a_service(self) -> None:
        # Arrange
        account = SnowflakeAccount(
            [],
            models=("ARENA_POLICY",),
            model_versions=[
                version_row("checkpoint_9", [], is_default=True),
                version_row(
                    "checkpoint_10",
                    [
                        "ARENA.TRAINING.MODEL_BUILD_1A2B3C",
                        POLICY_SERVICE,
                    ],
                ),
            ],
        )

        # Act
        found = served_version(account, "ARENA", "TRAINING", "arena_policy", None)

        # Assert
        assert found == ServedVersion(
            model="ARENA.TRAINING.arena_policy",
            version_name="CHECKPOINT_10",
            services=(POLICY_SERVICE,),
            is_default=False,
            only_version=False,
        )
        assert "SHOW VERSIONS IN MODEL ARENA.TRAINING.arena_policy" in (
            account.statements
        )

    def test_named_version_is_matched_without_case(self) -> None:
        account = SnowflakeAccount(
            [],
            models=("ARENA_POLICY",),
            model_versions=[version_row("v9", [], is_default=True)],
        )

        found = served_version(account, "ARENA", "TRAINING", "arena_policy", "v9")

        assert found.version_name == "V9"
        assert found.services == ()
        assert found.is_default
        assert found.only_version

    def test_missing_model_is_rejected(self) -> None:
        account = SnowflakeAccount([], models=("ARENA_POLICYX",))

        with pytest.raises(ValueError, match="arena_policy does not exist"):
            served_version(account, "ARENA", "TRAINING", "arena_policy", None)

    def test_missing_version_is_rejected(self) -> None:
        account = SnowflakeAccount(
            [], models=("ARENA_POLICY",), model_versions=[version_row("v1", [])]
        )

        with pytest.raises(ValueError, match="has no version v2"):
            served_version(account, "ARENA", "TRAINING", "arena_policy", "v2")

    def test_model_without_a_service_is_rejected(self) -> None:
        account = SnowflakeAccount(
            [],
            models=("ARENA_POLICY",),
            model_versions=[version_row("v1", ["ARENA.TRAINING.MODEL_BUILD_X"])],
        )

        with pytest.raises(ValueError, match="has no inference service"):
            served_version(account, "ARENA", "TRAINING", "arena_policy", None)

    def test_services_on_several_versions_need_a_version(self) -> None:
        account = SnowflakeAccount(
            [],
            models=("ARENA_POLICY",),
            model_versions=[
                version_row("v1", ["ARENA.TRAINING.SVC_1"]),
                version_row("v2", ["ARENA.TRAINING.SVC_2"]),
            ],
        )

        with pytest.raises(ValueError, match=r"on versions V1, V2\. Pass --version"):
            served_version(account, "ARENA", "TRAINING", "arena_policy", None)


class TestVersionEndpoint:
    def test_returns_the_services_inference_url(self) -> None:
        account = SnowflakeAccount(
            [],
            endpoints=[Row(name="inference", ingress_url="abc.snowflakecomputing.app")],
        )

        url = version_endpoint(account, served())

        assert url == "abc.snowflakecomputing.app"
        assert account.statements == [f"SHOW ENDPOINTS IN SERVICE {POLICY_SERVICE}"]

    @pytest.mark.parametrize(
        "services", [(), ("ARENA.TRAINING.SVC_1", "ARENA.TRAINING.SVC_2")]
    )
    def test_version_needs_exactly_one_service(self, services: tuple[str, ...]) -> None:
        account = SnowflakeAccount([])

        with pytest.raises(ValueError, match=f"has {len(services)} inference services"):
            version_endpoint(account, served(services=services))

        assert account.statements == []


class TestUnpublish:
    def test_drops_only_the_service(self) -> None:
        account = SnowflakeAccount([])

        unpublish(account, served(), drop_version=False)

        assert account.statements == [f"DROP SERVICE {POLICY_SERVICE}"]

    def test_drop_version_drops_the_version_after_its_services(self) -> None:
        account = SnowflakeAccount([])

        unpublish(
            account, served(is_default=False, only_version=False), drop_version=True
        )

        assert account.statements == [
            f"DROP SERVICE {POLICY_SERVICE}",
            "ALTER MODEL ARENA.TRAINING.arena_policy DROP VERSION CHECKPOINT_10",
        ]

    def test_an_only_version_is_dropped_with_the_model(self) -> None:
        account = SnowflakeAccount([])

        unpublish(account, served(services=()), drop_version=True)

        assert account.statements == ["DROP MODEL ARENA.TRAINING.arena_policy"]

    def test_default_of_several_versions_is_rejected_before_any_drop(self) -> None:
        account = SnowflakeAccount([])

        with pytest.raises(ValueError, match="SET DEFAULT_VERSION"):
            unpublish(account, served(only_version=False), drop_version=True)

        assert account.statements == []

    def test_version_without_a_service_needs_drop_version(self) -> None:
        account = SnowflakeAccount([])

        with pytest.raises(ValueError, match="Pass --drop-version"):
            unpublish(account, served(services=()), drop_version=False)

        assert account.statements == []


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
            "status: DONE",
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
            "status: DONE",
            "model: ARENA.TRAINING.arena_policy",
            "version: checkpoint_10",
        ]
        job_name = result.output.splitlines()[0].removeprefix("job: ")
        assert account.published(job_name)["image_build_compute_pool"] == (
            "arena_head_pool"
        )

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


ONE_GPU_FAMILY = {"GPU_NV_S": GpuPool("arena_gpu_GPU_NV_S", 2, 1, 24)}
SEVERAL_GPU_FAMILIES = {
    "GPU_NV_S": GpuPool("arena_gpu_GPU_NV_S", 2, 1, 24),
    "GPU_NV_M": GpuPool("arena_gpu_GPU_NV_M", 8, 4, 24),
    "GPU_NV_L": GpuPool("arena_gpu_GPU_NV_L", 8, 8, 40),
}


class Published:
    """Job object the publish command prints."""

    def __init__(self) -> None:
        self.name = "ARENA_PUBLISH_1"
        self.document = {
            "database": "ARENA",
            "schema": "TRAINING",
            "model_name": "arena_policy",
            "version_name": "v1",
            "service_name": None,
        }


def _saved_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: str,
    gpu: dict[str, GpuPool] = ONE_GPU_FAMILY,
) -> None:
    """Write ``[arena]`` and ``[arena-admin]``, and serve setup.json from the account."""
    key = tmp_path / "key.p8"
    key.write_bytes(b"key")
    write_connection("arena", job_connection_fields("ORGNAME-ACCOUNT", key))
    write_connection(
        "arena-admin", admin_connection_fields("ORGNAME-ACCOUNT", "admin@example.com")
    )
    profile = AccountProfile(
        image=image,
        warehouse="arena_wh",
        stage="arena.training.runs",
        metrics_table="arena.training.arena_metrics",
        external_access_integrations=("arena_hf_eai",),
        pools=AccountPools(
            broker=CpuPool("arena_pulsar_pool", "CPU_X64_L"),
            head=CpuPool("team_head_pool", "CPU_X64_SL"),
            gpu=gpu,
        ),
    )
    monkeypatch.setattr(SnowflakeAccount, "setup_doc", profile.as_dict())


def _capture_settings(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Stand in for the job so only the settings the command builds are kept."""
    seen: dict[str, object] = {}

    def fake_submit(session: object, settings: PublishSettings) -> Published:
        seen["settings"] = settings
        return Published()

    def fake_open(name: str | None) -> SnowflakeAccount:
        seen["connection"] = name
        return SnowflakeAccount([])

    monkeypatch.setattr(
        "agilerl.arena.snowflake.publish_cli.submit_publish", fake_submit
    )
    monkeypatch.setattr(
        "agilerl.arena.snowflake.publish_cli.wait_for_publish",
        lambda session, job, echo: None,
    )
    monkeypatch.setattr("agilerl.arena.snowflake.publish_cli.open_session", fake_open)
    return seen


class TestPublishProfile:
    def test_omitted_flags_come_from_the_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen = _capture_settings(monkeypatch)

        result = CliRunner().invoke(
            main, ["snowflake", "publish", RUN_ID, "--model-name", "arena_policy"]
        )

        assert result.exit_code == 0, result.output
        settings = seen["settings"]
        assert isinstance(settings, PublishSettings)
        assert seen["connection"] == "arena"
        assert settings.compute_pool == "arena_gpu_GPU_NV_S"
        assert settings.image_build_pool == "team_head_pool"
        assert settings.image == "/arena/training/images/snowflake-job:1"
        assert settings.warehouse == "arena_wh"
        assert settings.runs_stage == "arena.training.runs"

    def test_flags_win_over_the_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen = _capture_settings(monkeypatch)

        result = CliRunner().invoke(
            main,
            [
                *PUBLISH_ARGS,
                "--compute-pool",
                "other_pool",
                "--warehouse",
                "other_wh",
                "--image",
                "/arena/training/images/snowflake-job:2",
            ],
        )

        assert result.exit_code == 0, result.output
        settings = seen["settings"]
        assert isinstance(settings, PublishSettings)
        assert settings.compute_pool == "other_pool"
        assert settings.warehouse == "other_wh"
        assert settings.image == "/arena/training/images/snowflake-job:2"
        assert settings.image_build_pool == "team_head_pool"

    def test_several_gpu_families_use_the_runs_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        _saved_account(
            tmp_path,
            monkeypatch,
            "/arena/training/images/snowflake-job:1",
            gpu=SEVERAL_GPU_FAMILIES,
        )
        account = SnowflakeAccount(
            two_checkpoints(),
            endpoints=[Row(name="inference", ingress_url="abc.snowflakecomputing.app")],
        )
        monkeypatch.setattr(
            "agilerl.arena.snowflake.publish_cli.open_session", lambda name: account
        )

        # Act
        result = CliRunner().invoke(main, [*PUBLISH_ARGS, "--serve"])

        # Assert
        assert result.exit_code == 0, result.output
        job_name = result.output.splitlines()[0].removeprefix("job: ")
        assert account.published(job_name)["compute_pool"] == "arena_gpu_pool"
        assert account.published(job_name)["image_build_compute_pool"] == (
            "team_head_pool"
        )
        execute = next(s for s in account.statements if s.startswith("EXECUTE JOB"))
        assert execute.startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_gpu_pool\n"
        )
        container = account.job_spec()["spec"]["containers"][0]
        assert container["image"] == "/arena/training/images/snowflake-job:1"

    def test_an_empty_image_stays_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _saved_account(tmp_path, monkeypatch, "")
        seen = _capture_settings(monkeypatch)

        result = CliRunner().invoke(
            main, ["snowflake", "publish", RUN_ID, "--model-name", "arena_policy"]
        )

        assert result.exit_code == 0, result.output
        settings = seen["settings"]
        assert isinstance(settings, PublishSettings)
        assert settings.image is None
        assert settings.runs_stage == "arena.training.runs"

    def test_omitted_stage_without_setup_json_is_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _capture_settings(monkeypatch)

        result = CliRunner().invoke(
            main, ["snowflake", "publish", RUN_ID, "--model-name", "arena_policy"]
        )

        assert result.exit_code == 2
        assert "This account has no setup.json. Pass --runs-stage." in result.output
        assert "settings" not in seen

    def test_reads_setup_json_on_every_publish(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen = _capture_settings(monkeypatch)
        args = ["snowflake", "publish", RUN_ID, "--model-name", "arena_policy"]

        # Act
        CliRunner().invoke(main, args)
        first = seen["settings"]
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:2")
        result = CliRunner().invoke(main, args)

        # Assert
        assert result.exit_code == 0, result.output
        assert isinstance(first, PublishSettings)
        assert first.image == "/arena/training/images/snowflake-job:1"
        second = seen["settings"]
        assert isinstance(second, PublishSettings)
        assert second.image == "/arena/training/images/snowflake-job:2"

    def test_browser_admin_connection_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        seen = _capture_settings(monkeypatch)

        result = CliRunner().invoke(
            main, [*PUBLISH_ARGS, "--connection", "arena-admin"]
        )

        assert result.exit_code == 1
        assert "signs in with a browser as ACCOUNTADMIN" in result.output
        assert seen == {}


def served_account(
    models: tuple[str, ...] = ("ARENA_POLICY",),
    model_versions: list[Row] | None = None,
) -> SnowflakeAccount:
    """Account whose ``arena_policy`` model has two versions, one served."""
    return SnowflakeAccount(
        [],
        models=models,
        model_versions=model_versions
        or [
            version_row("checkpoint_9", [], is_default=True),
            version_row("checkpoint_10", [POLICY_SERVICE]),
        ],
        endpoints=[Row(name="inference", ingress_url="abc.snowflakecomputing.app")],
    )


def _open_account(
    monkeypatch: pytest.MonkeyPatch, account: SnowflakeAccount
) -> list[str | None]:
    opened: list[str | None] = []

    def fake_open(name: str | None) -> SnowflakeAccount:
        opened.append(name)
        return account

    monkeypatch.setattr("agilerl.arena.snowflake.publish_cli.open_session", fake_open)
    return opened


class TestEndpointCommand:
    def test_prints_the_served_versions_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        account = served_account()
        opened = _open_account(monkeypatch, account)

        # Act
        result = CliRunner().invoke(endpoint_command, ["arena_policy"])

        # Assert
        assert result.exit_code == 0, result.output
        assert result.output == "abc.snowflakecomputing.app\n"
        assert opened == ["arena"]
        assert account.statements[-1] == f"SHOW ENDPOINTS IN SERVICE {POLICY_SERVICE}"
        assert account.closed

    def test_version_and_schema_select_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        account = served_account()
        opened = _open_account(monkeypatch, account)

        result = CliRunner().invoke(
            endpoint_command,
            [
                "arena_policy",
                "--version",
                "checkpoint_10",
                "--database",
                "ml",
                "--schema",
                "registry",
            ],
        )

        assert result.exit_code == 0, result.output
        assert opened == [None]
        assert "SHOW VERSIONS IN MODEL ml.registry.arena_policy" in account.statements

    def test_version_without_a_service_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        account = served_account()
        _open_account(monkeypatch, account)

        result = CliRunner().invoke(
            endpoint_command, ["arena_policy", "--version", "checkpoint_9"]
        )

        assert result.exit_code == 1
        assert "has 0 inference services" in result.output
        assert account.closed

    def test_browser_admin_connection_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        opened = _open_account(monkeypatch, served_account())

        result = CliRunner().invoke(
            endpoint_command, ["arena_policy", "--connection", "arena-admin"]
        )

        assert result.exit_code == 1
        assert "signs in with a browser as ACCOUNTADMIN" in result.output
        assert opened == []


class TestUnpublishCommand:
    def test_drops_the_service_and_keeps_the_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        _saved_account(tmp_path, monkeypatch, "/arena/training/images/snowflake-job:1")
        account = served_account()
        opened = _open_account(monkeypatch, account)

        # Act
        result = CliRunner().invoke(unpublish_command, ["arena_policy"])

        # Assert
        assert result.exit_code == 0, result.output
        assert result.output == f"dropped service: {POLICY_SERVICE}\n"
        assert opened == ["arena"]
        assert account.statements[-1] == f"DROP SERVICE {POLICY_SERVICE}"
        assert not any("DROP VERSION" in s for s in account.statements)
        assert account.closed

    def test_drop_version_drops_the_version_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        account = served_account()
        _open_account(monkeypatch, account)

        result = CliRunner().invoke(
            unpublish_command,
            ["arena_policy", "--version", "checkpoint_10", "--drop-version"],
        )

        assert result.exit_code == 0, result.output
        assert result.output.splitlines() == [
            f"dropped service: {POLICY_SERVICE}",
            "dropped version: CHECKPOINT_10",
        ]
        assert account.statements[-2:] == [
            f"DROP SERVICE {POLICY_SERVICE}",
            "ALTER MODEL ARENA.TRAINING.arena_policy DROP VERSION CHECKPOINT_10",
        ]

    def test_drop_version_of_the_only_version_drops_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        account = served_account(
            model_versions=[
                version_row("checkpoint_10", [POLICY_SERVICE], is_default=True)
            ]
        )
        _open_account(monkeypatch, account)

        result = CliRunner().invoke(
            unpublish_command, ["arena_policy", "--drop-version"]
        )

        assert result.exit_code == 0, result.output
        assert result.output.splitlines() == [
            f"dropped service: {POLICY_SERVICE}",
            "dropped model: ARENA.TRAINING.arena_policy",
        ]
        assert account.statements[-1] == "DROP MODEL ARENA.TRAINING.arena_policy"

    def test_missing_model_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        account = served_account(models=())
        _open_account(monkeypatch, account)

        result = CliRunner().invoke(unpublish_command, ["arena_policy"])

        assert result.exit_code == 1
        assert "Model ARENA.TRAINING.arena_policy does not exist" in result.output
        assert not any(s.startswith("DROP") for s in account.statements)
