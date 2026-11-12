# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Trainer image copy and the GPU driver check."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import yaml
from packaging.version import Version

from agilerl.arena.models import __version__
from agilerl.arena.snowflake.images import (
    DOCKER_MISSING,
    DOCKER_NOT_RUNNING,
    DRIVER_CHECK_TIMEOUT_S,
    MISSING_IMAGE_TAG,
    POOL_START_TIMEOUT_S,
    PUSH_ATTEMPTS,
    READ_ONLY_REPOSITORY,
    REGISTRY_TOKEN_EXPIRED,
    TOKEN_CALL_MISSING,
    TOKEN_FAILED,
    TRAINER_IMAGE,
    Finished,
    TrainerImage,
    check_image_matches_cli,
    copy_images,
    docker_preflight,
    driver_check_sql,
    driver_new_enough,
    driver_service_name,
    image_bundle_version,
    image_not_found_message,
    image_preflight,
    parse_driver,
    parse_license_check,
    pull_images,
    registry_token,
    release_bundle_version,
    resolve_trainer_image,
    run_docker,
    run_driver_check,
    trainer_source,
)
from agilerl.arena.snowflake.job import arena_container_secrets, drop_service_sql

REGISTRY = "acct.registry.snowflakecomputing.com/arena/training/images"
DIGEST = "sha256:" + "b" * 64
IMAGE = TrainerImage("32.6.0-py313-cu130", DIGEST)
NOW = datetime(2026, 11, 11, 10, 0, 0, tzinfo=timezone.utc)
SERVICE = "ARENA_DRIVER_20261111100000"
POOL = "arena_gpu_pool"
TRAINER_PATH = "/arena/training/images/snowflake-job:32.6.0-py313-cu130"
CHECKED = "580.65.06\narena-license-check: ok\n"


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


class Cursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: str) -> None:
        self.statements.append(statement)


class Rest:
    """Connector REST client that issues numbered registry tokens."""

    def __init__(self) -> None:
        self.count = 0

    def _token_request(self, request_type: str) -> dict[str, object]:
        assert request_type == "ISSUE"
        self.count += 1
        return {
            "data": {
                "sessionToken": f"tok-{self.count}",
                "validityInSecondsST": 3600,
            }
        }


class FixedRest:
    """Connector REST client that returns one scripted token response."""

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def _token_request(self, request_type: str) -> dict[str, object]:
        assert request_type == "ISSUE"
        return self.response


class Connection:
    def __init__(self, rest: object) -> None:
        self.cursor_obj = Cursor()
        self._rest = rest
        self._all_async_queries_finished = lambda: True

    def cursor(self) -> Cursor:
        return self.cursor_obj


class CopySession:
    """Admin session that owns the image repository."""

    def __init__(self, rest: object | None = None) -> None:
        self.connection = Connection(Rest() if rest is None else rest)
        self.statements: list[str] = []

    def sql(self, statement: str) -> Frame:
        self.statements.append(statement)
        return Frame([Row(repository_url=REGISTRY)])

    def close(self) -> None:
        return None


class Docker:
    """Docker stand-in. ``results`` maps a subcommand to scripted outcomes."""

    def __init__(self, results: dict[str, list[Finished]] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, args: list[str], stdin: str | None) -> Finished:
        self.calls.append((args, stdin))
        scripted = self.results.get(args[1], [])
        if scripted:
            return scripted.pop(0)
        return Finished(0, "", "")

    def subcommands(self) -> list[str]:
        return [args[1] for args, _stdin in self.calls]

    def login_tokens(self) -> list[str]:
        return [
            json.loads(stdin or "")["token"]
            for args, stdin in self.calls
            if args[1] == "login"
        ]


def failed(stderr: str) -> Finished:
    return Finished(1, "", stderr)


def ecr(
    status: int = 200,
    digest: str = DIGEST,
    token_status: int = 200,
    seen: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """Public ECR: anonymous token endpoint and manifest HEAD."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path == "/token/":
            return httpx.Response(token_status, json={"token": "anon"})
        headers = {"docker-content-digest": digest} if status == 200 else {}
        return httpx.Response(status, headers=headers)

    return httpx.MockTransport(handler)


class DriverSession:
    """``arena_trainer`` session: scripted pool states, job statuses, and logs."""

    def __init__(
        self,
        pool_states: list[str],
        statuses: list[list[dict[str, str]]],
        logs: list[str],
    ) -> None:
        self.pool_states = pool_states
        self.statuses = statuses
        self.logs = logs
        self.statements: list[str] = []

    def sql(self, statement: str) -> Frame:
        self.statements.append(statement)
        if statement.startswith("SHOW COMPUTE POOLS"):
            state = self._next(self.pool_states)
            return Frame([Row(name=POOL.upper(), state=state)])
        if statement.startswith("SHOW SERVICE CONTAINERS"):
            return Frame([Row(**row) for row in self._next(self.statuses)])
        if statement.startswith("SELECT SYSTEM$GET_SERVICE_LOGS"):
            return Frame([Row(**{"SYSTEM$GET_SERVICE_LOGS": self._next(self.logs)})])
        return Frame([])

    def close(self) -> None:
        return None

    @staticmethod
    def _next(values: list) -> object:
        return values.pop(0) if len(values) > 1 else values[0]


def container(service: str, status: str = "", message: str = "") -> dict[str, str]:
    return {"service_status": service, "status": status, "message": message}


def ticking(step: float) -> Callable[[], float]:
    reads = {"t": -step}

    def clock() -> float:
        reads["t"] += step
        return reads["t"]

    return clock


def no_sleep(_seconds: float) -> None:
    return None


class TestReleaseBundleVersion:
    def test_a_stamped_release_uses_its_bundle_version(self, tmp_path: Path) -> None:
        stamp = tmp_path / "bundle_version.txt"
        stamp.write_text("32.6.0\n", encoding="utf-8")

        version = release_bundle_version(None, stamp)

        assert version == "32.6.0"
        assert trainer_source(version) == (
            "public.ecr.aws/x7s7d0o8/containers/snowflake-job:32.6.0-py313-cu130"
        )

    @pytest.mark.parametrize(
        "image_tag",
        ["32.6.0", "ml/v32.6.0", "32.6.0-py313-cu130", "ml/v32.6.0-py313-cu130"],
    )
    def test_an_explicit_tag_wins_over_the_stamp(
        self, tmp_path: Path, image_tag: str
    ) -> None:
        stamp = tmp_path / "bundle_version.txt"
        stamp.write_text("1.0.0\n", encoding="utf-8")

        assert release_bundle_version(image_tag, stamp) == "32.6.0"

    def test_an_unstamped_install_asks_for_a_tag(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Pass --image-tag") as raised:
            release_bundle_version(None, tmp_path / "bundle_version.txt")

        assert str(raised.value) == MISSING_IMAGE_TAG

    def test_this_checkout_has_no_stamp(self) -> None:
        with pytest.raises(ValueError, match="not a release build"):
            release_bundle_version()


class TestImageBundleVersion:
    def test_reads_the_release_tag(self) -> None:
        assert image_bundle_version(TRAINER_PATH) == Version("32.6.0")

    @pytest.mark.parametrize(
        "image",
        [
            "/arena/training/images/snowflake-job:latest",
            "/arena/training/images/snowflake-job:32.6.0",
            "/arena/training/images/snowflake-job:mr.5-py313-cu130",
        ],
    )
    def test_a_custom_tag_has_no_bundle_version(self, image: str) -> None:
        assert image_bundle_version(image) is None


class TestCheckImageMatchesCli:
    @staticmethod
    def _stamp(tmp_path: Path, bundle: str) -> Path:
        stamp = tmp_path / "bundle_version.txt"
        stamp.write_text(f"{bundle}\n", encoding="utf-8")
        return stamp

    def test_the_matching_image_passes(self, tmp_path: Path) -> None:
        check_image_matches_cli(TRAINER_PATH, self._stamp(tmp_path, "32.6.0"))

    def test_a_newer_cli_asks_for_setup(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Re-run arena snowflake setup") as raised:
            check_image_matches_cli(TRAINER_PATH, self._stamp(tmp_path, "32.7.0"))

        assert str(raised.value) == (
            f"This agilerl-arena {__version__} goes with trainer image 32.7.0, "
            "but the account's trainer image is 32.6.0. Re-run arena snowflake "
            "setup to update the trainer image to 32.7.0."
        )

    def test_a_newer_image_asks_for_a_cli_upgrade(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Run: pip install -U") as raised:
            check_image_matches_cli(TRAINER_PATH, self._stamp(tmp_path, "32.5.1"))

        assert str(raised.value) == (
            f"This agilerl-arena {__version__} goes with trainer image 32.5.1, "
            "but the account's trainer image is 32.6.0. "
            'Run: pip install -U "agilerl-arena[snowflake]"'
        )

    def test_versions_compare_numerically(self, tmp_path: Path) -> None:
        image = "/arena/training/images/snowflake-job:32.10.0-py313-cu130"

        with pytest.raises(ValueError, match="Run: pip install -U"):
            check_image_matches_cli(image, self._stamp(tmp_path, "32.9.0"))

    def test_an_unstamped_install_is_not_checked(self, tmp_path: Path) -> None:
        check_image_matches_cli(TRAINER_PATH, tmp_path / "bundle_version.txt")

    def test_a_custom_tag_is_not_checked(self, tmp_path: Path) -> None:
        check_image_matches_cli(
            "/arena/training/images/snowflake-job:latest",
            self._stamp(tmp_path, "32.7.0"),
        )


class TestResolveTrainerImage:
    def test_the_tag_resolves_to_its_digest_anonymously(self) -> None:
        seen: list[httpx.Request] = []

        image = resolve_trainer_image("32.6.0", ecr(seen=seen))

        assert image == IMAGE
        assert image.reference == f"{TRAINER_IMAGE}@{DIGEST}"
        token, head = seen
        assert token.url.params["scope"] == (
            "repository:x7s7d0o8/containers/snowflake-job:pull"
        )
        assert head.method == "HEAD"
        assert head.url.path == (
            "/v2/x7s7d0o8/containers/snowflake-job/manifests/32.6.0-py313-cu130"
        )
        assert head.headers["authorization"] == "Bearer anon"

    def test_a_missing_tag_is_named(self) -> None:
        with pytest.raises(ValueError, match="does not exist") as raised:
            resolve_trainer_image("99.0.0", ecr(status=404))

        assert str(raised.value) == image_not_found_message("99.0.0-py313-cu130")
        assert "--image-tag" in str(raised.value)

    def test_a_refused_token_stops(self) -> None:
        with pytest.raises(ValueError, match=r"anonymous pull token \(HTTP 503\)"):
            resolve_trainer_image("32.6.0", ecr(token_status=503))

    def test_a_response_without_a_digest_stops(self) -> None:
        with pytest.raises(ValueError, match="without an image digest"):
            resolve_trainer_image("32.6.0", ecr(digest=""))

    def test_an_unreachable_registry_is_one_sentence(self) -> None:
        def offline(request: httpx.Request) -> httpx.Response:
            msg = "no route"
            raise httpx.ConnectError(msg, request=request)

        with pytest.raises(ValueError, match=r"could not reach public\.ecr\.aws"):
            resolve_trainer_image("32.6.0", httpx.MockTransport(offline))


class TestDockerPreflight:
    def test_a_running_daemon_passes(self) -> None:
        docker = Docker()

        docker_preflight(docker)

        assert docker.calls == [
            (["docker", "info", "--format", "{{.ServerVersion}}"], None)
        ]

    def test_a_stopped_daemon_is_named(self) -> None:
        docker = Docker({"info": [failed("Cannot connect to the Docker daemon")]})

        with pytest.raises(ValueError, match="daemon is not running") as raised:
            docker_preflight(docker)

        assert str(raised.value) == DOCKER_NOT_RUNNING

    def test_missing_docker_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(
            *_args: object, **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError

        monkeypatch.setattr("agilerl.arena.snowflake.images.subprocess.run", missing)

        with pytest.raises(ValueError, match="Docker is not installed") as raised:
            docker_preflight(run_docker)

        assert str(raised.value) == DOCKER_MISSING


class TestImagePreflight:
    def test_returns_the_pinned_image_without_pulling(self) -> None:
        docker = Docker()

        image = image_preflight("ml/v32.6.0", docker, ecr())

        assert image == IMAGE
        assert docker.subcommands() == ["info"]

    def test_a_stopped_daemon_stops_before_the_registry(self) -> None:
        seen: list[httpx.Request] = []
        docker = Docker({"info": [failed("Cannot connect")]})

        with pytest.raises(ValueError, match="daemon is not running"):
            image_preflight("32.6.0", docker, ecr(seen=seen))

        assert seen == []

    def test_a_missing_tag_stops(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            image_preflight("32.6.0", Docker(), ecr(status=404))


class TestPullImages:
    def test_pulls_only_the_trainer_by_digest_for_amd64(self) -> None:
        docker = Docker()

        pull_images(IMAGE, docker)

        assert docker.calls == [
            (["docker", "pull", "--platform", "linux/amd64", IMAGE.reference], None),
        ]
        assert "login" not in docker.subcommands()

    def test_a_missing_manifest_is_image_not_found(self) -> None:
        docker = Docker({"pull": [failed("manifest unknown: manifest unknown")]})

        with pytest.raises(ValueError, match="does not exist") as raised:
            pull_images(IMAGE, docker)

        assert str(raised.value) == image_not_found_message(IMAGE.tag)

    def test_a_full_disk_stops_in_one_sentence(self) -> None:
        docker = Docker({"pull": [failed("write /var/lib: no space left on device")]})

        with pytest.raises(
            ValueError,
            match="The disk is out of space, so the trainer image was not copied",
        ):
            pull_images(IMAGE, docker)

    def test_another_pull_error_keeps_dockers_line(self) -> None:
        docker = Docker({"pull": [failed("toomanyrequests")]})

        with pytest.raises(ValueError, match=r"Setup could not pull") as raised:
            pull_images(IMAGE, docker)

        assert str(raised.value) == (
            f"Setup could not pull the trainer image snowflake-job:{IMAGE.tag}. "
            "toomanyrequests"
        )

    def test_a_read_only_disk_stops_in_one_sentence(self) -> None:
        docker = Docker({"pull": [failed("mkdir /var/lib: read-only file system")]})

        with pytest.raises(ValueError, match="disk is read-only") as raised:
            pull_images(IMAGE, docker)

        assert str(raised.value) == (
            "The disk is read-only, so the trainer image was not copied."
        )

    def test_a_silent_failure_says_docker_failed(self) -> None:
        docker = Docker({"pull": [Finished(1, "", "\n  \n")]})

        with pytest.raises(ValueError, match="Docker failed") as raised:
            pull_images(IMAGE, docker)

        assert str(raised.value) == (
            f"Setup could not pull the trainer image snowflake-job:{IMAGE.tag}. "
            "Docker failed."
        )


class TestCopyImages:
    def test_pushes_only_the_trainer_with_a_fresh_token_on_stdin(self) -> None:
        docker = Docker()
        session = CopySession()

        tag = copy_images(session, IMAGE, docker)

        assert tag == "32.6.0-py313-cu130"
        assert docker.subcommands() == ["tag", "login", "push"]
        assert docker.calls[0][0] == [
            "docker",
            "tag",
            IMAGE.reference,
            f"{REGISTRY}/snowflake-job:32.6.0-py313-cu130",
        ]
        login_args, login_stdin = docker.calls[1]
        assert login_args[-1] == "acct.registry.snowflakecomputing.com"
        assert "tok-1" not in login_args
        assert json.loads(login_stdin or "") == {"token": "tok-1", "expires_in": 3600}
        assert docker.login_tokens() == ["tok-1"]
        assert docker.calls[-1][0] == [
            "docker",
            "push",
            f"{REGISTRY}/snowflake-job:32.6.0-py313-cu130",
        ]

    def test_an_expired_token_logs_in_again_and_resumes(self) -> None:
        docker = Docker(
            {"push": [failed("unauthorized: authentication required")]},
        )

        tag = copy_images(CopySession(), IMAGE, docker)

        assert tag == IMAGE.tag
        assert docker.subcommands() == ["tag", "login", "push", "login", "push"]
        assert docker.login_tokens() == ["tok-1", "tok-2"]

    def test_a_token_rejected_on_every_attempt_says_it_expired(self) -> None:
        expired = [failed("unauthorized: token expired")] * PUSH_ATTEMPTS
        docker = Docker({"push": list(expired)})

        with pytest.raises(ValueError, match="token expired") as raised:
            copy_images(CopySession(), IMAGE, docker)

        assert str(raised.value) == REGISTRY_TOKEN_EXPIRED
        assert "It has READ" not in str(raised.value)
        assert docker.subcommands().count("push") == PUSH_ATTEMPTS

    def test_a_privilege_error_says_the_role_is_read_only(self) -> None:
        docker = Docker(
            {"push": [failed("denied: requested access to the resource is denied")]}
        )

        with pytest.raises(ValueError, match="It has READ") as raised:
            copy_images(CopySession(), IMAGE, docker)

        assert str(raised.value) == READ_ONLY_REPOSITORY
        assert docker.subcommands().count("push") == 1

    def test_a_failed_login_names_the_registry_and_hides_the_token(self) -> None:
        docker = Docker({"login": [failed("Error response from daemon: 500")]})

        with pytest.raises(ValueError, match="could not log in") as raised:
            copy_images(CopySession(), IMAGE, docker)

        assert "acct.registry.snowflakecomputing.com" in str(raised.value)
        assert "tok-1" not in str(raised.value)
        assert "push" not in docker.subcommands()

    def test_another_push_error_keeps_dockers_line(self) -> None:
        docker = Docker({"push": [failed("blob upload invalid")]})

        with pytest.raises(
            ValueError, match=r"could not upload the image.*blob upload invalid"
        ):
            copy_images(CopySession(), IMAGE, docker)

    def test_a_repository_without_a_url_runs_no_docker(self) -> None:
        class NoUrl(CopySession):
            def sql(self, statement: str) -> Frame:
                self.statements.append(statement)
                return Frame([Row(repository_url="  ")])

        docker = Docker()

        with pytest.raises(
            ValueError,
            match=r"Image repository arena\.training\.images has no repository URL",
        ):
            copy_images(NoUrl(), IMAGE, docker)

        assert docker.calls == []


class TestRegistryToken:
    def test_issues_a_token_on_the_json_result_format(self) -> None:
        connection = Connection(Rest())

        token = registry_token(connection)

        assert token == {"token": "tok-1", "expires_in": 3600}
        assert connection.cursor_obj.statements == [
            "alter session set PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'json'"
        ]

    def test_a_connector_without_the_token_call_is_named(self) -> None:
        with pytest.raises(ValueError, match="no registry token call") as raised:
            registry_token(Connection(object()))

        assert str(raised.value) == TOKEN_CALL_MISSING

    def test_a_closed_connection_stops(self) -> None:
        with pytest.raises(ValueError, match="retrieve a registry token"):
            registry_token(Connection(None))

    @pytest.mark.parametrize(
        "response",
        [
            {"data": None},
            {"data": {"sessionToken": "tok", "validityInSecondsST": "3600"}},
            {"data": {"validityInSecondsST": 3600}},
        ],
    )
    def test_a_malformed_token_response_stops(
        self, response: dict[str, object]
    ) -> None:
        connection = Connection(FixedRest(response))

        with pytest.raises(ValueError, match="retrieve a registry token") as raised:
            registry_token(connection)

        assert str(raised.value) == TOKEN_FAILED


class TestRunDocker:
    def test_returns_status_and_streams_with_stdin(self) -> None:
        script = "import sys; print(sys.stdin.read().upper()); sys.exit(3)"

        result = run_docker([sys.executable, "-c", script], "tok")

        assert result == Finished(3, "TOK\n", "")


class TestDriverCheckSql:
    def test_580_65_06_passes_and_older_fails(self) -> None:
        assert driver_new_enough("580.65.06")
        assert driver_new_enough("580.65.6")
        assert driver_new_enough("580.66.00")
        assert driver_new_enough("581.0.0")
        assert not driver_new_enough("580.65.05")
        assert not driver_new_enough("535.54.03")

    def test_the_trainer_image_reports_the_driver_and_arena_then_stays_up(
        self,
    ) -> None:
        sql = driver_check_sql(driver_service_name(NOW), POOL, TRAINER_PATH)

        assert f"    image: {TRAINER_PATH}\n" in sql
        assert "docker.io" not in sql
        assert "cuda" not in sql
        assert (
            "    - 'nvidia-smi --query-gpu=driver_version --format=csv,noheader; "
            "python -m agilerl.arena.snowflake.license_check; "
            "sleep 120'"
        ) in sql
        assert "nvidia.com/gpu: 1" in sql
        assert f"EXECUTE JOB SERVICE IN COMPUTE POOL {POOL}" in sql
        assert f"NAME = {SERVICE}" in sql
        assert "ASYNC = TRUE" in sql

    def test_the_job_reaches_arena_like_the_head(self) -> None:
        sql = driver_check_sql(driver_service_name(NOW), POOL, TRAINER_PATH)

        assert "\nEXTERNAL_ACCESS_INTEGRATIONS = (arena_api_eai)\n" in sql
        assert (
            "    secrets:\n"
            "    - snowflakeSecret: arena.training.arena_org_key\n"
            "      secretKeyRef: secret_string\n"
            "      envVarName: ARENA_ORG_KEY\n"
            "    - snowflakeSecret: arena.training.arena_external_user_id\n"
            "      secretKeyRef: secret_string\n"
            "      envVarName: ARENA_EXTERNAL_USER_ID\n"
            "    - snowflakeSecret: arena.training.arena_base_url\n"
            "      secretKeyRef: secret_string\n"
            "      envVarName: ARENA_BASE_URL\n"
            "    resources:\n"
        ) in sql

    def test_the_spec_is_yaml_with_the_head_secrets(self) -> None:
        sql = driver_check_sql(driver_service_name(NOW), POOL, TRAINER_PATH)
        spec = yaml.safe_load(sql.split("$$")[1])

        container = spec["spec"]["containers"][0]
        assert container["secrets"] == arena_container_secrets()
        assert container["command"][:2] == ["sh", "-c"]


class TestParseDriver:
    def test_reads_the_nvidia_smi_line(self) -> None:
        log = "580.65.06\narena-license-check: ok\n"

        assert parse_driver(log) == "580.65.06"

    def test_a_version_in_the_licence_line_is_not_the_driver(self) -> None:
        log = (
            "sh: 1: nvidia-smi: not found\n"
            "arena-license-check: agilerl-arena 0.9.0 is no longer supported\n"
        )

        assert parse_driver(log) is None


class TestParseLicenseCheck:
    def test_ok(self) -> None:
        assert parse_license_check("580.65.06\narena-license-check: ok\n") == "ok"

    def test_returns_the_licence_error(self) -> None:
        log = "580.65.06\narena-license-check: Arena at https://a.test returned HTTP 502.\n"

        assert parse_license_check(log) == "Arena at https://a.test returned HTTP 502."

    def test_no_result_before_the_check_prints(self) -> None:
        assert parse_license_check("580.65.06\n") is None


class TestRunDriverCheck:
    def test_reads_the_driver_and_arena_while_the_container_runs(self) -> None:
        session = DriverSession(
            ["ACTIVE"], [[container("RUNNING", "READY")]], [CHECKED]
        )

        found = run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

        assert found == "580.65.06"
        assert session.statements[0].startswith("EXECUTE JOB SERVICE")
        assert f"image: {TRAINER_PATH}" in session.statements[0]
        assert "EXTERNAL_ACCESS_INTEGRATIONS = (arena_api_eai)" in session.statements[0]
        assert session.statements[-1] == drop_service_sql(SERVICE, True)

    def test_waits_for_the_arena_result_after_the_driver(self) -> None:
        session = DriverSession(
            ["ACTIVE"], [[container("RUNNING", "READY")]], ["580.65.06\n", CHECKED]
        )

        found = run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

        assert found == "580.65.06"
        reads = [s for s in session.statements if "GET_SERVICE_LOGS" in s]
        assert len(reads) == 2

    def test_an_arena_failure_stops_with_the_licence_error(self) -> None:
        # Arrange
        log = (
            "580.65.06\n"
            "arena-license-check: https://arena.internal answered HTTP 200 with "
            "text/html, which is not the Arena API; the Arena URL must be "
            "reachable from the public internet.\n"
        )
        session = DriverSession(["ACTIVE"], [[container("RUNNING", "READY")]], [log])

        # Act
        with pytest.raises(ValueError, match="Arena check from inside") as raised:
            run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

        # Assert
        assert str(raised.value) == (
            "The Arena check from inside Snowflake failed: https://arena.internal "
            "answered HTTP 200 with text/html, which is not the Arena API; the "
            "Arena URL must be reachable from the public internet."
        )
        assert session.statements[-1] == drop_service_sql(SERVICE, True)

    def test_an_old_driver_is_named_before_an_arena_failure(self) -> None:
        log = "535.54.03\narena-license-check: Arena at https://a.test returned HTTP 502.\n"
        session = DriverSession(["ACTIVE"], [[container("RUNNING", "READY")]], [log])

        with pytest.raises(ValueError, match="The GPU driver is 535"):
            run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

    def test_a_finished_job_without_an_arena_result_shows_the_last_log_line(
        self,
    ) -> None:
        log = (
            "580.65.06\n"
            "/home/user/.venv/bin/python: No module named "
            "agilerl.arena.snowflake.license_check\n"
        )
        session = DriverSession(
            ["ACTIVE"], [[container("DONE", "DONE", "Completed")]], [log]
        )

        with pytest.raises(ValueError, match="did not report a result") as raised:
            run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

        assert str(raised.value) == (
            "The Arena check from inside Snowflake did not report a result. "
            "Last log line: /home/user/.venv/bin/python: No module named "
            "agilerl.arena.snowflake.license_check"
        )

    def test_the_deadline_starts_once_the_pool_is_ready(self) -> None:
        # Arrange
        pool_states = ["SUSPENDED", "STARTING", "STARTING", "IDLE"]
        session = DriverSession(
            pool_states,
            [[container("PENDING", "PENDING")], [container("RUNNING", "READY")]],
            [CHECKED],
        )

        # Act
        found = run_driver_check(
            session,
            POOL,
            TRAINER_PATH,
            sleep=no_sleep,
            now=NOW,
            clock=ticking(DRIVER_CHECK_TIMEOUT_S / 2),
        )

        # Assert
        assert found == "580.65.06"
        assert sum(s.startswith("SHOW COMPUTE POOLS") for s in session.statements) == 4

    def test_a_pool_that_never_starts_stops_and_drops_the_service(self) -> None:
        session = DriverSession(["STARTING"], [[]], [""])

        with pytest.raises(ValueError, match="still STARTING after 30 minutes"):
            run_driver_check(
                session,
                POOL,
                TRAINER_PATH,
                sleep=no_sleep,
                now=NOW,
                clock=ticking(POOL_START_TIMEOUT_S),
            )

        assert session.statements[-1] == drop_service_sql(SERVICE, True)
        assert not any(
            s.startswith("SHOW SERVICE CONTAINERS") for s in session.statements
        )

    def test_an_old_driver_is_named(self) -> None:
        session = DriverSession(
            ["ACTIVE"], [[container("RUNNING", "READY")]], ["535.54.03\n"]
        )

        with pytest.raises(ValueError, match="The GPU driver is 535") as raised:
            run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

        assert "580.65.06 or newer" in str(raised.value)
        assert session.statements[-1] == drop_service_sql(SERVICE, True)

    def test_a_failed_job_reports_its_status_and_message(self) -> None:
        session = DriverSession(
            ["ACTIVE"],
            [[container("FAILED", "FAILED", "Failed to pull image")]],
            ["\n"],
        )

        with pytest.raises(ValueError, match="driver check failed") as raised:
            run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

        assert "service FAILED, container FAILED: Failed to pull image" in str(
            raised.value
        )
        assert session.statements[-1] == drop_service_sql(SERVICE, True)

    def test_a_finished_job_without_a_version_stops(self) -> None:
        session = DriverSession(
            ["ACTIVE"], [[container("DONE", "DONE", "Completed")]], ["no gpu\n"]
        )

        with pytest.raises(ValueError, match="did not report a driver version"):
            run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

    def test_a_job_that_never_reports_stops_at_the_deadline(self) -> None:
        session = DriverSession(["ACTIVE"], [[]], [""])

        with pytest.raises(ValueError, match="did not finish in 10 minutes") as raised:
            run_driver_check(
                session,
                POOL,
                TRAINER_PATH,
                sleep=no_sleep,
                now=NOW,
                clock=ticking(DRIVER_CHECK_TIMEOUT_S),
            )

        assert "lists no container" in str(raised.value)
        assert session.statements[-1] == drop_service_sql(SERVICE, True)

    def test_logs_are_not_read_while_the_container_is_pending(self) -> None:
        session = DriverSession(
            ["ACTIVE"],
            [[container("PENDING", "PENDING")], [container("RUNNING", "READY")]],
            [CHECKED],
        )

        run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

        reads = [
            "status" if s.startswith("SHOW SERVICE") else "logs"
            for s in session.statements
            if s.startswith(("SHOW SERVICE", "SELECT SYSTEM$GET_SERVICE_LOGS"))
        ]
        assert reads == ["status", "status", "logs"]

    def test_an_unknown_pool_stops(self) -> None:
        class NoPool(DriverSession):
            def sql(self, statement: str) -> Frame:
                self.statements.append(statement)
                return Frame([])

        session = NoPool(["ACTIVE"], [[]], [""])

        with pytest.raises(ValueError, match="arena_gpu_pool was not found"):
            run_driver_check(session, POOL, TRAINER_PATH, sleep=no_sleep, now=NOW)

        assert session.statements[-1] == drop_service_sql(SERVICE, True)
