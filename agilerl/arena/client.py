from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Self

import httpx

from agilerl.arena.auth import ArenaOAuth2, load_credentials
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaValidationError,
)
from agilerl.arena.logs import EventStream, LogDisplay
from agilerl.models import ArenaCluster, JobStatus, TrainingManifest
from agilerl.utils.arena_utils import (
    prepare_env_upload,
)

logger = logging.getLogger(__name__)

_ARCHIVE_EXCLUDE_DIRS = frozenset({
    ".git",
    "__pycache__",
    "node_modules",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    ".eggs",
    "*.egg-info",
})

_ARCHIVE_EXCLUDE_SUFFIXES = frozenset({
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".egg",
})

_ARCHIVE_EXCLUDE_FILES = frozenset({
    ".env",
    ".DS_Store",
    "Thumbs.db",
})


@dataclass(slots=True)
class _TokenStore:
    """In-memory holder for OAuth tokens with redacted repr."""

    access_token: str | None = None
    refresh_token: str | None = None

    def __repr__(self) -> str:
        has_access = self.access_token is not None
        has_refresh = self.refresh_token is not None
        return f"_TokenStore(access={has_access}, refresh={has_refresh})"

    def clear(self) -> None:
        self.access_token = None
        self.refresh_token = None


class ArenaClient:
    """Client for the Arena RLOps platform.

    Handles authentication, environment management, and training job
    submission.

    Authentication is resolved in priority order:

    1. *api_key* constructor argument
    2. ``ARENA_API_KEY`` environment variable
    3. Stored OAuth credentials from ``~/.arena/credentials.json``
    4. Interactive :meth:`login` (device authorization flow)

    :param api_key: Static API key for bearer-token authentication.
        When provided, OAuth login is not required.
    :param request_timeout: Default timeout in seconds for API requests.
    :param upload_timeout: Timeout in seconds for file-upload requests.
    """

    BASE_URL: ClassVar[str] = "https://arena.agilerl.com"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        request_timeout: int = 30,
        upload_timeout: int = 300,
    ) -> None:

        self._base_url = self.BASE_URL.rstrip("/")
        self._request_timeout = request_timeout
        self._upload_timeout = upload_timeout

        self._api_key = api_key or os.environ.get("ARENA_API_KEY")
        self._auth = ArenaOAuth2()
        self._tokens = _TokenStore()

        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=self._request_timeout,
            follow_redirects=True,
        )

        if self._api_key is None:
            self._try_restore_session()

            if self._tokens.access_token is None:
                logger.info(
                    "No credentials found, please authenticate with Arena by running "
                    "client.login() or provide an API key to the ArenaClient constructor."
                )

    @classmethod
    def configure(
        cls,
        *,
        base_url: str | None = None,
        keycloak_url: str | None = None,
        realm: str | None = None,
        client_id: str | None = None,
    ) -> type[ArenaClient]:
        """Override default URLs for local development or testing.

        Returns the class so calls can be chained with instantiation::

            client = ArenaClient.configure(
                base_url="http://localhost:3001",
                keycloak_url="http://localhost:8023",
            )()
        """
        if base_url is not None:
            cls.BASE_URL = base_url

        # Configure the ArenaOAuth2 instance
        ArenaOAuth2.configure(
            keycloak_url=keycloak_url,
            realm=realm,
            client_id=client_id,
        )
        return cls

    def login(self, *, timeout: int = 300) -> None:
        """Start the device-authorization login flow.

        Opens a browser for verification.  The call blocks until
        the user authorizes or *timeout* seconds elapse.
        On success the tokens are persisted to
        ``~/.arena/credentials.json``.
        """
        tokens = self._auth.device_login(timeout=timeout)
        self._tokens.access_token = tokens["access_token"]
        self._tokens.refresh_token = tokens.get("refresh_token")
        logger.info("Authenticated with Arena.")

    def logout(self) -> None:
        """Clear the current session and remove stored credentials."""
        if self._tokens.refresh_token:
            self._auth.revoke(self._tokens.refresh_token)
        self._tokens.clear()
        logger.info("Logged out of Arena.")

    @property
    def is_authenticated(self) -> bool:
        """``True`` when the client holds a valid API key or access token."""
        return self._api_key is not None or self._tokens.access_token is not None

    # -------------------------------------------------------------------------
    ### Custom Environments ###
    # -------------------------------------------------------------------------

    def _register_environment(
        self,
        name: str,
        source: str | os.PathLike[str] | None = None,
        version: str = "latest",
        entrypoint: str | None = None,
        config: dict[str, Any] | str | os.PathLike[str] | None = None,
        requirements: str | os.PathLike[str] | None = None,
        description: str | None = None,
        multi_agent: bool = False,
        rollouts: bool = False,
        max_steps: int = 200,
    ) -> dict[str, Any]:
        """Register a custom environment on Arena.

        :param name: The name of the environment.
        :type name: str
        :param source: Path to the environment source directory.
        :type source: str | os.PathLike[str] | None
        :param version: The version of the environment. Defaults to "latest".
        :param entrypoint: The entrypoint of the environment.
        :type entrypoint: str | None
        :param config: Environment configuration.
        :type config: dict[str, Any] | str | os.PathLike[str] | None
        :param requirements: Environment requirements.
        :type requirements: str | os.PathLike[str] | None
        :param description: Environment description.
        :type description: str | None
        :param multi_agent: Whether the environment is a multi-agent environment.
        :type multi_agent: bool
        :param rollouts: Whether to include rollout profiling during validation.
        :type rollouts: bool
        :param max_steps: Maximum steps per rollout episode.
        :type max_steps: int
        :returns: Registration report from the Arena API.
        :rtype: dict[str, Any]
        """
        src = Path(os.fspath(source)).resolve()
        if not src.exists():
            msg = f"Environment source not found: {src}"
            raise FileNotFoundError(msg)

        logger.info(
            "Creating .tar.gz archive for environment %s version %s...", name, version
        )

        # Convert the environment source to a tar.gz archive
        payload = prepare_env_upload(
            source=src,
            config=config,
            requirements=requirements,
            description=description,
        )

        logger.info("Uploading environment %s version %s to Arena...", name, version)

        # Send the environment to Arena for registration
        return self._request(
            "POST",
            "api/custom-gym-env-impls/create",
            files={"archive": ("environment.tar.gz", payload, "application/gzip")},
            data={
                "name": name,
                "version": version,
                "entrypoint": entrypoint,
                "multi_agent": str(multi_agent).lower(),
                "rollouts": str(rollouts).lower(),
                "max_steps": str(max_steps),
            },
            timeout=self._upload_timeout,
        )

    def _validate(
        self,
        name: str,
        version: str = "latest",
        entrypoint: str | None = None,
        rollouts: bool = False,
        max_steps: int = 200,
    ) -> dict[str, Any]:
        """Validate a custom environment on Arena.

        :param name: The name of the environment.
        :type name: str
        :param version: The version of the environment. Defaults to "latest".
        :type version: str
        :param entrypoint: The entrypoint of the environment.
        :type entrypoint: str | None
        :param rollouts: Whether to include rollout profiling during validation.
        :type rollouts: bool
        :param max_steps: Maximum steps per rollout episode.
        :type max_steps: int
        :returns: Validation report from the Arena API.
        :rtype: dict[str, Any]
        """
        resp = self._request(
            "GET",
            "api/v1/custom-gym-env-impls/validate",
            params={
                "name": name,
                "version": version,
                "entrypoint": entrypoint,
                "rollouts": str(rollouts).lower(),
                "max_steps": str(max_steps),
            },
        )
        self._check_validation_result(resp)

        if "operation_id" in resp:
            return self.stream_logs(resp["operation_id"])

        return resp

    def get_current_user(self) -> dict[str, Any]:
        """Get the authenticated user's profile details."""
        return self._request("GET", "/api/users/current")

    def get_user_credits(self) -> Any:
        """Get the authenticated user's credit information."""
        return self._request("GET", "/api/users/credits")

    def list_custom_environments(self) -> Any:
        """List custom environments available to the authenticated user."""
        return self._unwrap_cli_data(
            self._request("GET", "/api/cli/v1/environments")
        )

    def custom_environment_exists(self, name: str, version: str = "latest") -> bool:
        """Check whether a custom environment name/version exists."""
        resp = self._unwrap_cli_data(
            self._request(
            "GET",
            "/api/cli/v1/environments/exists",
            params={"name": name, "version": version},
            )
        )
        if isinstance(resp, dict):
            for key in ("exists", "is_registered", "isRegistered"):
                if key in resp:
                    return bool(resp[key])
        return bool(resp)

    def list_custom_environment_entrypoints(
        self, name: str, version: str = "latest"
    ) -> list[str]:
        """List available entrypoints for a custom environment version."""
        resp = self._unwrap_cli_data(
            self._request(
            "GET",
            "/api/cli/v1/environments/entrypoints",
            params={"name": name, "version": version},
            )
        )
        if isinstance(resp, dict):
            entrypoints = resp.get("entrypoints", [])
            if isinstance(entrypoints, list):
                return [str(entrypoint) for entrypoint in entrypoints]
        if isinstance(resp, list):
            return [str(entrypoint) for entrypoint in resp]
        return []

    def validate_custom_environment(
        self,
        *,
        name: str,
        version: str,
        entrypoint: str | None = None,
        do_rollouts: bool = True,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Validate an already created custom environment version."""
        payload: dict[str, Any] = {
            "name": name,
            "version": version,
            "do_rollouts": do_rollouts,
        }
        if entrypoint:
            payload["entrypoint"] = entrypoint

        if not stream:
            return self._request("POST", "/api/cli/v1/environments/validate", json=payload)

        return self._stream_json_request(
            "POST",
            "/api/cli/v1/environments/validate",
            json=payload,
            timeout=self._upload_timeout,
            on_chunk=on_chunk,
        )

    def profile_custom_environment(
        self,
        *,
        name: str,
        version: str,
        multi_agent: bool,
        custom_env_path: str | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Profile a validated custom environment version by name and version."""
        payload: dict[str, Any] = {
            "name": name,
            "version": version,
            "multi_agent": multi_agent,
        }
        if custom_env_path:
            payload["custom_env_path"] = custom_env_path

        if not stream:
            return self._request("POST", "/api/cli/v1/environments/profile", json=payload)

        return self._stream_json_request(
            "POST",
            "/api/cli/v1/environments/profile",
            json=payload,
            timeout=self._upload_timeout,
            on_chunk=on_chunk,
        )

    def delete_custom_environment(self, *, name: str, version: str) -> Any:
        """Delete a custom environment version by name and version."""
        payload = {"name": name, "version": version}
        return self._unwrap_cli_data(
            self._request("DELETE", "/api/cli/v1/environments/delete", json=payload)
        )

    def create_and_validate_custom_environment(
        self,
        *,
        name: str,
        version: str,
        file_path: str | os.PathLike[str],
        env_config_path: str | os.PathLike[str],
        requirements_path: str | os.PathLike[str],
        multi_agent: bool = False,
        do_rollouts: bool = True,
        entrypoint: str | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Create and validate a custom environment via multipart upload."""
        env_archive = Path(os.fspath(file_path)).expanduser().resolve()
        env_config = Path(os.fspath(env_config_path)).expanduser().resolve()
        requirements = Path(os.fspath(requirements_path)).expanduser().resolve()

        for upload_path in (env_archive, env_config, requirements):
            if not upload_path.is_file():
                msg = f"Upload file not found: {upload_path}"
                raise FileNotFoundError(msg)

        with (
            env_archive.open("rb") as archive_handle,
            env_config.open("rb") as config_handle,
            requirements.open("rb") as requirements_handle,
        ):
            payload = {
                "name": name,
                "version": version,
                "multi_agent": str(multi_agent).lower(),
                "do_rollouts": str(do_rollouts).lower(),
            }
            if entrypoint:
                payload["entrypoint"] = entrypoint
            files = {
                "file": (env_archive.name, archive_handle, "application/gzip"),
                "env_config": (env_config.name, config_handle, "application/x-yaml"),
                "requirements": (
                    requirements.name,
                    requirements_handle,
                    "text/plain",
                ),
            }

            if not stream:
                return self._request(
                    "POST",
                    "/api/cli/v1/environments/create-and-validate",
                    data=payload,
                    files=files,
                    timeout=self._upload_timeout,
                )

            auth_retry = False
            while True:
                headers = self._auth_headers()
                try:
                    with self._http.stream(
                        "POST",
                        "/api/cli/v1/environments/create-and-validate",
                        headers=headers,
                        data=payload,
                        files=files,
                        timeout=self._upload_timeout,
                    ) as resp:
                        if (
                            resp.status_code == 401
                            and not auth_retry
                            and self._api_key is None
                            and self._tokens.refresh_token
                        ):
                            tokens = self._auth.refresh_access_token(
                                self._tokens.refresh_token
                            )
                            self._tokens.access_token = tokens["access_token"]
                            self._tokens.refresh_token = tokens.get(
                                "refresh_token", self._tokens.refresh_token
                            )
                            auth_retry = True
                            continue

                        if resp.status_code == 401:
                            msg = (
                                "Session expired and could not be refreshed. "
                                "Please run client.login() again."
                            )
                            raise ArenaAuthError(msg)

                        if not resp.is_success:
                            detail = (
                                resp.read().decode("utf-8", errors="replace")[:500]
                                or "No details"
                            )
                            raise ArenaAPIError(
                                status_code=resp.status_code,
                                detail=detail,
                            )

                        chunks: list[str] = []
                        for chunk in resp.iter_text():
                            if not chunk:
                                continue
                            chunks.append(chunk)
                            if on_chunk is not None:
                                on_chunk(chunk)
                        body = "".join(chunks).strip()
                        if not body:
                            return {}
                        try:
                            return json.loads(body)
                        except json.JSONDecodeError:
                            return {"stream": body}
                except httpx.HTTPError as exc:
                    raise ArenaAPIError(
                        status_code=0,
                        detail=f"Network error communicating with Arena: {exc}",
                    ) from exc

    def submit_experiment_job(
        self,
        *,
        manifest: dict[str, Any] | None = None,
        custom_gym_env_impl_id: int | None = None,
        experiment_id: int | None = None,
        experiment_name: str | None = None,
        gym_env_id: int | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Submit an experiment job to the script-aligned API endpoint."""
        payload: dict[str, Any] = {}
        if manifest is not None:
            payload["manifest"] = manifest
        if custom_gym_env_impl_id is not None:
            payload["custom_gym_env_impl_id"] = custom_gym_env_impl_id
        if experiment_id is not None:
            payload["experiment_id"] = experiment_id
        if experiment_name is not None:
            payload["experiment_name"] = experiment_name
        if gym_env_id is not None:
            payload["gym_env_id"] = gym_env_id

        if not payload:
            msg = (
                "Provide at least one submit field: --manifest, --custom-gym-env-impl-id, "
                "--experiment-id, --experiment-name, or --gym-env-id."
            )
            raise ArenaValidationError([{"message": msg}])

        if not stream:
            return self._request(
                "POST",
                "/api/cli/v1/experiments/jobs/submit",
                json=payload,
                timeout=self._upload_timeout,
            )

        auth_retry = False
        while True:
            headers = self._auth_headers()
            try:
                with self._http.stream(
                    "POST",
                    "/api/cli/v1/experiments/jobs/submit",
                    headers=headers,
                    json=payload,
                    timeout=self._upload_timeout,
                ) as resp:
                    if (
                        resp.status_code == 401
                        and not auth_retry
                        and self._api_key is None
                        and self._tokens.refresh_token
                    ):
                        tokens = self._auth.refresh_access_token(
                            self._tokens.refresh_token
                        )
                        self._tokens.access_token = tokens["access_token"]
                        self._tokens.refresh_token = tokens.get(
                            "refresh_token", self._tokens.refresh_token
                        )
                        auth_retry = True
                        continue

                    if resp.status_code == 401:
                        msg = (
                            "Session expired and could not be refreshed. "
                            "Please run client.login() again."
                        )
                        raise ArenaAuthError(msg)

                    if not resp.is_success:
                        detail = (
                            resp.read().decode("utf-8", errors="replace")[:500]
                            or "No details"
                        )
                        raise ArenaAPIError(
                            status_code=resp.status_code,
                            detail=detail,
                        )

                    chunks: list[str] = []
                    for chunk in resp.iter_text():
                        if not chunk:
                            continue
                        chunks.append(chunk)
                        if on_chunk is not None:
                            on_chunk(chunk)
                    body = "".join(chunks).strip()
                    if not body:
                        return {}
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError:
                        return {"stream": body}
            except httpx.HTTPError as exc:
                raise ArenaAPIError(
                    status_code=0,
                    detail=f"Network error communicating with Arena: {exc}",
                ) from exc

    def get_experiment_status(self, experiment_id: int) -> dict[str, Any]:
        """Get status/details for an experiment."""
        resp = self._request("GET", f"/api/experiments/{experiment_id}")
        return resp if isinstance(resp, dict) else {"experiment_id": experiment_id, "status": resp}

    def validate_job_run_spec(self, run_spec: dict[str, Any]) -> dict[str, Any]:
        """Validate a runspec payload against backend schema/rules."""
        result = self._unwrap_cli_data(
            self._request("POST", "/api/cli/v1/experiments/validate-run-spec", json=run_spec)
        )
        if result in ("", None):
            return {"valid": True}
        return result if isinstance(result, dict) else {"valid": True, "response": result}

    def download_experiment_metrics(
        self, experiment_id: int, metrics: list[str]
    ) -> tuple[bytes, str | None, str | None]:
        """Download experiment metrics payload (CSV or zipped CSV)."""
        return self._request_raw(
            "POST",
            f"/api/experiments/{experiment_id}/metrics",
            json={"metrics": metrics},
        )

    # TODO: Print the environments in a table format using rich
    def list_environments(self) -> None:
        """List all custom environments registered in Arena."""
        self._request("GET", "api/v1/custom-gym-env-impls/list")

    def list_entrypoints(self, name: str, version: str = "latest") -> list[str]:
        """List all entrypoints available for a custom environment version.

        :param name: The name of the environment.
        :param version: The version of the environment. Defaults to "latest".
        :returns: List of entrypoints for the given environment.
        """
        resp = self._request(
            "GET",
            "api/v1/custom-gym-env-impls/list-entrypoints",
            params={"name": name, "version": version},
        )
        return resp["entrypoints"]

    def is_registered_environment(self, name: str, version: str = "latest") -> bool:
        """Check if a custom environment is registered in Arena.

        :param name: The name of the environment.
        :param version: The version of the environment. Defaults to "latest".
        :returns: True if the environment is registered in Arena, False otherwise.
        """
        resp = self._request(
            "GET",
            "api/v1/custom-gym-env-impls/is-registered",
            params={"name": name, "version": version},
        )
        return resp["is_registered"]

    def validate_environment(
        self,
        name: str,
        source: str | os.PathLike[str] | None = None,
        version: str | None = None,
        entrypoint: str | None = None,
        description: str | None = None,
        multi_agent: bool = False,
        config: dict[str, Any] | str | os.PathLike[str] | None = None,
        requirements: str | os.PathLike[str] | None = None,
        rollouts: bool = False,
        max_steps: int = 200,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Validate a custom environment on Arena.

        Users can choose to either upload a custom environment from a directory or a single file, which
        will prompt Arena to create and validate it automatically or provide the name and version of an
        already registered environment to validate.

        :param name: The name of the environment.
        :type name: str
        :param version: The version of the environment. Defaults to "latest".
        :type version: str | None
        :param entrypoint: The entrypoint of the environment.
        :type entrypoint: str | None
        :param description: The description of the environment.
        :type description: str | None
        :param multi_agent: Whether the environment is a multi-agent environment.
        :type multi_agent: bool
        :param config: Optional environment configuration.  Pass a ``dict`` (serialized to JSON automatically) or a path to a ``.json`` file.
        :type config: dict[str, Any] | str | os.PathLike[str] | None
        :param requirements: Optional path to a ``requirements.txt`` file that lists the environment's Python dependencies.
        :type requirements: str | os.PathLike[str] | None
        :param source: Path to a directory or ``.py`` file containing the environment implementation.
        :type source: str | os.PathLike[str] | None
        :param rollouts: Whether to include rollout profiling during validation.
        :type rollouts: bool
        :param max_steps: Maximum steps per rollout episode.
        :type max_steps: int
        :param stream: If ``True``, stream validation logs to the terminal in real time and block until the operation finishes.
        :type stream: bool
        :returns: Validation report from the Arena API.
        :rtype: dict[str, Any]
        """
        common_kwargs = {
            "version": version,
            "entrypoint": entrypoint,
            "rollouts": str(rollouts).lower(),
            "max_steps": str(max_steps),
        }
        if source is not None:
            if self.is_registered_environment(name, version):
                logger.info(
                    "Environment %s version %s is already registered, validating...",
                    name,
                    version,
                )
            else:
                return self._register_environment(
                    name=name,
                    source=source,
                    config=config,
                    requirements=requirements,
                    description=description,
                    multi_agent=multi_agent,
                    **common_kwargs,
                )

        # Validate the registered environment on Arena
        logger.info(
            "Validating registered environment %s version %s on Arena...", name, version
        )
        return self._validate(name=name, **common_kwargs)

    # -------------------------------------------------------------------------
    ### Training Jobs ###
    # -------------------------------------------------------------------------

    def submit_job(
        self, manifest: TrainingManifest, cluster: ArenaCluster | None = None
    ) -> dict[str, Any]:
        """Submit a training job to Arena.

        :param manifest: Fully validated job manifest.
        :type manifest: TrainingManifest
        :param cluster: Optional compute cluster specification.
        :type cluster: ArenaCluster | None
        :returns: Server response including ``job_id`` and initial
            ``status``.
        """
        payload = manifest.model_dump(mode="json")
        if cluster is not None:
            payload["cluster"] = cluster.model_dump(mode="json")

        # Submit job to Arena
        return self._request(
            "POST",
            "api/v1/jobs/submit",
            json=payload,
        )

    def get_job_status(self, job_id: str) -> JobStatus:
        """Retrieve the current status of a training job.

        :param job_id: Identifier returned by :meth:`submit_training_job`.
        :returns: The current status of the job.
        :rtype: JobStatus
        """
        status = self._request("GET", "api/v1/jobs/status", params={"job_id": job_id})
        return JobStatus(status)

    def stop_job(self, job_id: str) -> None:
        """Request stopping of a running job.

        :param job_id: Identifier returned by :meth:`submit_job`.
        """
        return self._request("POST", "api/v1/jobs/stop", params={"job_id": job_id})

    def iter_events(
        self,
        operation_id: str,
        *,
        max_retries: int = 5,
    ) -> EventStream:
        """Return an iterable stream of
        :class:`~agilerl.arena.logs.LogEvent` objects for
        *operation_id*.

        :param operation_id: Identifier returned by an async Arena
            operation (training job, validation, etc.).
        :param max_retries: Maximum reconnection attempts on disconnect.
        :returns: An iterable, context-manager-capable event stream.
        """
        return EventStream(
            http=self._http,
            path=f"api/v1/operations/{operation_id}/events",
            auth_headers=self._auth_headers(),
            max_retries=max_retries,
        )

    def stream_logs(self, operation_id: str) -> dict[str, Any]:
        """Pretty-print live logs for *operation_id* and block until
        the operation completes.

        :param operation_id: Identifier returned by an async Arena
            operation.
        :returns: The final result payload from the ``complete`` event.
        """
        display = LogDisplay()
        try:
            with self.iter_events(operation_id) as events:
                for event in events:
                    display.render(event)
        finally:
            display.stop()
        return display.result

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "authenticated" if self.is_authenticated else "unauthenticated"
        return f"<ArenaClient url={self._base_url!r} {status}>"

    def _try_restore_session(self) -> None:
        creds = load_credentials(ArenaOAuth2.CREDENTIALS_FILE)
        if creds:
            self._tokens.access_token = creds.get("access_token")
            self._tokens.refresh_token = creds.get("refresh_token")

    def _auth_headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}

        if self._tokens.access_token:
            return {"Authorization": f"Bearer {self._tokens.access_token}"}

        msg = (
            "Client has not been authenticated with Arena. Call client.login() or provide an "
            "API key to the ArenaClient constructor."
        )
        raise ArenaAuthError(msg)

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: int | None = None,
        _retried: bool = False,
        **kwargs: dict[str, Any],
    ) -> Any:
        request_headers = dict(kwargs.pop("headers", {}))
        headers = dict(request_headers)
        headers.update(self._auth_headers())

        try:
            resp = self._http.request(
                method,
                path,
                headers=headers,
                timeout=timeout,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise ArenaAPIError(
                status_code=0,
                detail=f"Network error communicating with Arena: {exc}",
            ) from exc

        if (
            resp.status_code == 401
            and not _retried
            and self._api_key is None
            and self._tokens.refresh_token
        ):
            logger.debug("Access token expired, attempting refresh.")
            tokens = self._auth.refresh_access_token(self._tokens.refresh_token)
            self._tokens.access_token = tokens["access_token"]
            self._tokens.refresh_token = tokens.get(
                "refresh_token", self._tokens.refresh_token
            )
            return self._request(
                method,
                path,
                timeout=timeout,
                _retried=True,
                headers=request_headers,
                **kwargs,
            )

        if resp.status_code == 401:
            detail = resp.text[:500] if resp.text else "No details"
            msg = (
                "Session expired and could not be refreshed. "
                f"Please run client.login() again. Server response: {detail}"
            )
            raise ArenaAuthError(msg)

        if not resp.is_success:
            detail = resp.text[:500] if resp.text else "No details"
            raise ArenaAPIError(
                status_code=resp.status_code,
                detail=detail,
            )

        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.text

    def _request_raw(
        self,
        method: str,
        path: str,
        *,
        timeout: int | None = None,
        _retried: bool = False,
        **kwargs: dict[str, Any],
    ) -> tuple[bytes, str | None, str | None]:
        request_headers = dict(kwargs.pop("headers", {}))
        headers = dict(request_headers)
        headers.update(self._auth_headers())

        try:
            resp = self._http.request(
                method,
                path,
                headers=headers,
                timeout=timeout,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise ArenaAPIError(
                status_code=0,
                detail=f"Network error communicating with Arena: {exc}",
            ) from exc

        if (
            resp.status_code == 401
            and not _retried
            and self._api_key is None
            and self._tokens.refresh_token
        ):
            tokens = self._auth.refresh_access_token(self._tokens.refresh_token)
            self._tokens.access_token = tokens["access_token"]
            self._tokens.refresh_token = tokens.get(
                "refresh_token", self._tokens.refresh_token
            )
            return self._request_raw(
                method,
                path,
                timeout=timeout,
                _retried=True,
                headers=request_headers,
                **kwargs,
            )

        if resp.status_code == 401:
            detail = resp.text[:500] if resp.text else "No details"
            msg = (
                "Session expired and could not be refreshed. "
                f"Please run client.login() again. Server response: {detail}"
            )
            raise ArenaAuthError(msg)

        if not resp.is_success:
            detail = resp.text[:500] if resp.text else "No details"
            raise ArenaAPIError(status_code=resp.status_code, detail=detail)

        content_type = resp.headers.get("content-type")
        disposition = resp.headers.get("content-disposition")
        return resp.content, content_type, disposition

    def _stream_json_request(
        self,
        method: str,
        path: str,
        *,
        timeout: int | None = None,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        auth_retry = False
        while True:
            headers = self._auth_headers()
            try:
                with self._http.stream(
                    method,
                    path,
                    headers=headers,
                    timeout=timeout,
                    **kwargs,
                ) as resp:
                    if (
                        resp.status_code == 401
                        and not auth_retry
                        and self._api_key is None
                        and self._tokens.refresh_token
                    ):
                        tokens = self._auth.refresh_access_token(
                            self._tokens.refresh_token
                        )
                        self._tokens.access_token = tokens["access_token"]
                        self._tokens.refresh_token = tokens.get(
                            "refresh_token", self._tokens.refresh_token
                        )
                        auth_retry = True
                        continue

                    if resp.status_code == 401:
                        msg = (
                            "Session expired and could not be refreshed. "
                            "Please run client.login() again."
                        )
                        raise ArenaAuthError(msg)

                    if not resp.is_success:
                        detail = (
                            resp.read().decode("utf-8", errors="replace")[:500]
                            or "No details"
                        )
                        raise ArenaAPIError(
                            status_code=resp.status_code,
                            detail=detail,
                        )

                    chunks: list[str] = []
                    for chunk in resp.iter_text():
                        if not chunk:
                            continue
                        chunks.append(chunk)
                        if on_chunk is not None:
                            on_chunk(chunk)
                    body = "".join(chunks).strip()
                    if not body:
                        return {}
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError:
                        return {"stream": body}
            except httpx.HTTPError as exc:
                raise ArenaAPIError(
                    status_code=0,
                    detail=f"Network error communicating with Arena: {exc}",
                ) from exc

    @staticmethod
    def _check_validation_result(resp: dict[str, Any]) -> None:
        errors = resp.get("errors")
        if errors:
            raise ArenaValidationError(errors)

    @staticmethod
    def _unwrap_cli_data(resp: Any) -> Any:
        """Unwrap standardized CLI envelope responses when present."""
        if isinstance(resp, dict) and resp.get("ok") is True and "data" in resp:
            return resp["data"]
        return resp
