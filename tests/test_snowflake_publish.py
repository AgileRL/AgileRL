# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Publish: the request file, the job service, the wait, and served versions."""

from __future__ import annotations

import dataclasses
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from agilerl.snowflake.profile import (
    GpuPool,
)
from agilerl.snowflake.publish import (
    PublishDocument,
    PublishJob,
    PublishSettings,
    ServedVersion,
    chat_example,
    chat_function,
    create_chat_function_sql,
    drop_chat_function_sql,
    grant_chat_function_sql,
    grant_service_function_sql,
    publish_job_name,
    select_checkpoint,
    served_version,
    submit_publish,
    unpublish,
    version_service,
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
    "image_build_compute_pool": "arena_head_pool",
    "ingress": True,
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
        "agilerl.snowflake.session.connections_file",
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
    is served as setup.json. Pools named ``*head_pool`` are ``CPU_X64_SL``; every
    other pool is ``family``.
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
            if query.endswith("head_pool"):
                return [Row(instance_family="CPU_X64_SL")]
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
        ingress=True,
        compute_pool=None,
        head_pool="arena_head_pool",
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
    def test_overrides_replace_the_run_settings(self) -> None:
        account = SnowflakeAccount(two_checkpoints())

        job = submit_publish(
            account,
            settings(
                step=9,
                version_name="v9",
                service_name="policy_svc",
                compute_pool="serve_pool",
                head_pool="build_head_pool",
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
            "image_build_compute_pool": "build_head_pool",
            "gpu_count": 4,
            "max_model_len": 4096,
            "database": "ml",
            "schema": "registry",
        }
        assert "IN COMPUTE POOL build_head_pool\n" in account.statements[-1]
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

        with pytest.raises(ValueError, match="both the database and the schema"):
            submit_publish(account, settings(database=database, schema=schema))

        assert account.statements == []

    def test_session_without_a_schema_is_rejected(self) -> None:
        account = SnowflakeAccount(two_checkpoints(), current=("ARENA", None))

        with pytest.raises(ValueError, match="no current database or schema"):
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
            {"head_pool": "pool;x"},
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

    def test_the_job_leaves_a_one_gpu_node_to_the_service(self) -> None:
        # Arrange
        account = SnowflakeAccount(two_checkpoints(), family="GPU_NV_S")

        # Act
        job = submit_publish(account, settings(compute_pool="gpu_pool"))

        # Assert
        assert job.document["compute_pool"] == "gpu_pool"
        assert account.statements[-1].startswith(
            "EXECUTE JOB SERVICE IN COMPUTE POOL arena_head_pool\n"
        )
        container = account.job_spec()["spec"]["containers"][0]
        assert container["resources"]["limits"] == {"memory": "57.5Gi"}

    def test_image_that_breaks_the_sql_quote_is_rejected(self) -> None:
        account = SnowflakeAccount(two_checkpoints())

        with pytest.raises(ValueError, match=r"cannot contain \$\$"):
            submit_publish(account, settings(image="/a/b/c/job:$$"))

        assert account.file.uploads == {}

    def test_a_checkpoint_larger_than_a_head_node_is_rejected(self) -> None:
        account = SnowflakeAccount(
            staged(
                ("checkpoint_3/merged/model.safetensors", 60 * GIB),
                ("checkpoint_3/merged/.complete", 0),
            )
        )

        with pytest.raises(
            ValueError,
            match=r"needs about 64 GiB .* arena_head_pool node has 57\.5 GiB",
        ):
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
                "Endpoint, once the job is done: "
                "CALL <app>.app_code.endpoint('arena_policy', 'checkpoint_10')"
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

        with pytest.raises(ValueError, match=r"on versions V1, V2\. Pass version_name"):
            served_version(account, "ARENA", "TRAINING", "arena_policy", None)


class TestChatFunction:
    def test_create_calls_the_registry_chat_method_with_openai_defaults(self) -> None:
        statement = create_chat_function_sql("app_state.POLICY")

        assert statement == (
            "CREATE FUNCTION IF NOT EXISTS app_state.POLICY_CHAT"
            "(messages ARRAY, temperature FLOAT, max_completion_tokens INT)\n"
            "RETURNS OBJECT\n"
            "AS $$SELECT app_state.POLICY!__CALL__(messages, temperature, "
            "max_completion_tokens, NULL, 1, FALSE, 1.0, 0.0, 0.0, NULL)$$"
        )

    def test_grant_and_drop_name_the_function_with_its_argument_types(self) -> None:
        grant = grant_chat_function_sql("app_state.POLICY", "APPLICATION ROLE app_user")
        service_role = grant_service_function_sql(
            "app_state.POLICY", "APPLICATION ROLE app_user"
        )
        drop = drop_chat_function_sql("app_state.POLICY")

        assert grant == (
            "GRANT USAGE ON FUNCTION app_state.POLICY_CHAT(ARRAY, FLOAT, INT) "
            "TO APPLICATION ROLE app_user"
        )
        assert service_role == (
            "GRANT SERVICE ROLE app_state.POLICY!INFERENCE_SERVICE_FUNCTION_USAGE "
            "TO APPLICATION ROLE app_user"
        )
        assert drop == (
            "DROP FUNCTION IF EXISTS app_state.POLICY_CHAT(ARRAY, FLOAT, INT)"
        )

    def test_example_reads_the_reply_text(self) -> None:
        example = chat_example(chat_function(POLICY_SERVICE))

        assert example == (
            f"SELECT {POLICY_SERVICE}_CHAT(\n"
            """  PARSE_JSON('[{"role": "user", "content": "Hello"}]'),\n"""
            "  0.7,\n"
            "  256\n"
            "):choices[0]:message:content::STRING AS reply;"
        )


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

        with pytest.raises(ValueError, match="Pass drop_version"):
            unpublish(account, served(services=()), drop_version=False)

        assert account.statements == []


class TestPublishJobName:
    def test_name_is_stamped_and_unquoted(self) -> None:
        name = publish_job_name(NOW)

        assert name.startswith("ARENA_PUBLISH_20261110120005_")
        assert len(name) == len("ARENA_PUBLISH_20261110120005_ABCDEF")
        assert name == name.upper()


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


class TestVersionService:
    def test_one_service_is_returned(self) -> None:
        served = ServedVersion("M", "V1", ("DB.S.SVC",), True, True)

        assert version_service(served) == "DB.S.SVC"

    @pytest.mark.parametrize("services", [(), ("DB.S.A", "DB.S.B")])
    def test_none_or_several_are_refused(self, services: tuple[str, ...]) -> None:
        served = ServedVersion("M", "V1", services, True, True)

        with pytest.raises(ValueError, match=f"has {len(services)} inference services"):
            version_service(served)
