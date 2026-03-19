from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import httpx

from agilerl.arena.auth import ArenaOAuth2, load_credentials
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaValidationError,
)
from agilerl.arena.logs import EventStream, LogDisplay
from agilerl.arena.models import ArenaCluster, ArenaTrainingManifest, JobStatus
from agilerl.utils.arena_utils import (
    prepare_env_upload,
)

logger = logging.getLogger(__name__)

_ARCHIVE_EXCLUDE_DIRS = frozenset(
    {
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
    }
)

_ARCHIVE_EXCLUDE_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".egg",
    }
)

_ARCHIVE_EXCLUDE_FILES = frozenset(
    {
        ".env",
        ".DS_Store",
        "Thumbs.db",
    }
)


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
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MiB

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

    def _create_and_validate(
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
        """Create and validate a custom environment on Arena.

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
        :returns: Validation report from the Arena API.
        :rtype: dict[str, Any]
        """
        src = Path(os.fspath(source)).resolve()
        if not src.exists():
            msg = f"Environment source not found: {src}"
            raise FileNotFoundError(msg)

        # Convert the environment source to a tar.gz archive
        payload = prepare_env_upload(
            source=src,
            config=config,
            requirements=requirements,
            description=description,
        )

        # Send the environment to Arena for validation
        resp = self._request(
            "POST",
            "api/v1/custom-gym-env-impls/create-and-validate",
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
        self._check_validation_result(resp)

        return resp

    def _validate(
        self,
        name: str,
        version: str = "latest",
        entrypoint: str | None = None,
        rollouts: bool = False,
        max_steps: int = 200,
        stream: bool = False,
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
        :param stream: If ``True``, stream validation logs to the terminal in real time and block until the operation finishes.
        :type stream: bool
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

        if stream and "operation_id" in resp:
            return self.stream_logs(resp["operation_id"])

        return resp

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

    def _is_arena_environment(self, name: str, version: str = "latest") -> bool:
        """Check if a custom environment is registered in Arena.

        :param name: The name of the environment.
        :param version: The version of the environment. Defaults to "latest".
        :returns: True if the environment is registered in Arena, False otherwise.
        """
        return self._request(
            "GET",
            "api/v1/custom-gym-env-impls/is-registered",
            params={"name": name, "version": version},
        )

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
            "name": name,
            "version": version,
            "entrypoint": entrypoint,
            "rollouts": str(rollouts).lower(),
            "max_steps": str(max_steps),
            "stream": str(stream).lower(),
        }
        if source is not None:
            return self._create_and_validate(
                source=source,
                config=config,
                requirements=requirements,
                description=description,
                multi_agent=multi_agent,
                **common_kwargs,
            )

        return self._validate(**common_kwargs)

    # -------------------------------------------------------------------------
    ### Training Jobs ###
    # -------------------------------------------------------------------------

    def submit_job(
        self,
        manifest: ArenaTrainingManifest,
        cluster: ArenaCluster | None = None,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Submit a training job to Arena.

        :param manifest: Fully validated job manifest.
        :type manifest: ArenaTrainingManifest
        :param cluster: Optional compute cluster specification.
        :type cluster: ArenaCluster | None
        :param stream: If ``True``, stream training logs to the
            terminal in real time and block until the job finishes.
        :type stream: bool
        :returns: Server response including ``job_id`` and initial
            ``status``.  When *stream* is ``True``, returns the final
            result from the ``complete`` event instead.
        """
        payload = manifest.model_dump(mode="json", exclude_none=True)
        if cluster is not None:
            payload["cluster"] = cluster.model_dump(mode="json")
        resp = self._request(
            "POST",
            "api/v1/jobs/submit",
            json=payload,
        )

        if stream and "operation_id" in resp:
            return self.stream_logs(resp["operation_id"])

        return resp

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
        headers = kwargs.pop("headers", {})
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
            return self._request(method, path, timeout=timeout, _retried=True, **kwargs)

        if resp.status_code == 401:
            msg = (
                "Session expired and could not be refreshed. "
                "Please run client.login() again."
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

    @staticmethod
    def _check_validation_result(resp: dict[str, Any]) -> None:
        errors = resp.get("errors")
        if errors:
            raise ArenaValidationError(errors)
