from __future__ import annotations

import io
import logging
import os
import tarfile
from collections.abc import Callable
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
from agilerl.arena.stream import NDJsonStream, StreamEvent

logger = logging.getLogger(__name__)


def prepare_env_upload(source: str | os.PathLike[str] | bytes) -> tuple[str, bytes]:
    """Resolve an environment source into an upload-ready ``(name, bytes)`` pair.

    *source* may be:

    * A path to a directory — compressed into ``.tar.gz`` automatically.
    * A path to an existing ``.tar.gz`` file — read as-is.
    * Raw ``bytes`` — used directly (assumed to be a valid ``.tar.gz``).

    :returns: ``(archive_name, archive_bytes)``
    :raises FileNotFoundError: If *source* is a path that does not exist.
    """
    if isinstance(source, bytes):
        return ("environment.tar.gz", source)

    path = Path(os.fspath(source)).expanduser().resolve()

    if path.is_dir():
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    tar.add(str(child), arcname=child.relative_to(path).as_posix())
        return (f"{path.name}.tar.gz", buf.getvalue())

    if path.is_file():
        return (path.name, path.read_bytes())

    msg = f"Source path not found: {path}"
    raise FileNotFoundError(msg)


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

    _ERROR_MAP: ClassVar[dict[str, type[ArenaAPIError]]] = {
        "/api/cli/v1/environments/create-and-validate": ArenaValidationError,
        "/api/cli/v1/environments/validate": ArenaValidationError,
        "/api/cli/v1/environments/profile": ArenaValidationError,
    }

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
        self._stream_handler: Callable[[StreamEvent], None] | None = None

        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=self._request_timeout,
            follow_redirects=True,
        )

        if self._api_key is None:
            self._try_restore_session()

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

    def set_stream_handler(self, handler: Callable[[StreamEvent], None] | None) -> None:
        """Register a callback invoked for each :class:`StreamEvent` during streaming.

        Set to ``None`` to clear the handler.
        """
        self._stream_handler = handler

    # -------------------------------------------------------------------------
    ### User ###
    # -------------------------------------------------------------------------

    def get_current_user(self) -> dict[str, Any]:
        """Get the authenticated user's profile details."""
        return self._request("GET", "/api/users/current")

    def get_user_credits(self) -> Any:
        """Get the authenticated user's credit information."""
        return self._request("GET", "/api/users/credits")

    # -------------------------------------------------------------------------
    ### Environments ###
    # -------------------------------------------------------------------------

    def list_environments(self) -> Any:
        """List environments available to the authenticated user."""
        # TODO: Here we want to show a rich table showing all environments available, nested with their
        # versions and whether they are validated and profiled.
        return self._unwrap_cli_data(self._request("GET", "/api/cli/v1/environments"))

    def environment_exists(self, name: str, version: str = "latest") -> bool:
        """Check whether an environment name/version is registered."""
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

    def list_environment_entrypoints(
        self,
        name: str,
        version: str | None = None,
    ) -> list[str]:
        """List available entrypoints for an environment version.

        :param name: Environment name, as specified in Arena.
        :type name: str
        :param version: Environment version. Defaults to None, which resolves to the latest version.
        :type version: str | None
        :returns: A list of entrypoints.
        :rtype: list[str]
        """
        resp = self._unwrap_cli_data(
            self._request(
                "GET",
                "/api/cli/v1/environments/entrypoints",
                params={"name": name, "version": version},
            )
        )
        # TODO: Again, cleaner if we knew exactly what response to expect
        if isinstance(resp, dict):
            entrypoints = resp.get("entrypoints", [])
            if isinstance(entrypoints, list):
                return [str(ep) for ep in entrypoints]
        if isinstance(resp, list):
            return [str(ep) for ep in resp]
        return []

    # TODO: Check with Rob
    # Getting an ambiguous entrypoint error when validating an already registered environment.
    # We should by default use the entrypoint the env was registered with
    def validate_environment(
        self,
        *,
        name: str | None = None,
        version: str = "latest",
        source: str | os.PathLike[str] | bytes | None = None,
        env_config: str | os.PathLike[str] | None = None,
        requirements: str | os.PathLike[str] | None = None,
        entrypoint: str | None = None,
        description: str | None = None,
        multi_agent: bool = False,
        do_rollouts: bool = True,
        stream: bool = False,
    ) -> dict[str, Any] | NDJsonStream:
        """Validate a custom environment on Arena.

        When *source* is provided the environment is uploaded, created, and
        validated in a single step.  When *source* is ``None`` an
        already-registered environment is validated by *name*/*version*.

        By default the stream is consumed internally and the final result
        dict is returned.  Pass ``stream=True`` to get the raw
        :class:`~agilerl.arena.stream.NDJsonStream` instead.

        :param name: Environment name.
        :type name: str | None
        :param version: Environment version.
        :type version: str
        :param source: Environment source — a directory path (compressed
            automatically), a ``.tar.gz`` file path, or raw ``bytes``.
        :type source: str | os.PathLike[str] | bytes | None
        :param env_config: Path to the ``env_config.yaml`` file.
        :type env_config: str | os.PathLike[str] | None
        :param requirements: Path to ``requirements.txt``.
        :type requirements: str | os.PathLike[str] | None
        :param entrypoint: Optional entrypoint override.
        :type entrypoint: str | None
        :param description: Optional human-readable description of the environment.
        :type description: str | None
        :param multi_agent: Whether the environment is multi-agent.
        :type multi_agent: bool
        :param do_rollouts: Whether to run rollout profiling.
        :type do_rollouts: bool
        :param stream: If ``True``, return the raw :class:`NDJsonStream`.
        :type stream: bool
        """
        if name is None and source is None:
            msg = "To validate an environment on Arena, either the name of an already registered environment or the source of a custom environment must be provided."
            raise ValueError(msg)

        if source is not None:
            stream_resp = self._create_and_validate(
                name=name,
                version=version,
                source=source,
                env_config=env_config,
                requirements=requirements,
                entrypoint=entrypoint,
                description=description,
                multi_agent=multi_agent,
                do_rollouts=do_rollouts,
            )
            return stream_resp if stream else stream_resp.collect()

        payload: dict[str, Any] = {
            "name": name,
            "version": version,
            "do_rollouts": do_rollouts,
        }
        if entrypoint:
            payload["entrypoint"] = entrypoint

        stream_resp = self._open_stream(
            "POST",
            "/api/cli/v1/environments/validate",
            json=payload,
            timeout=self._upload_timeout,
        )
        return stream_resp if stream else stream_resp.collect()

    def profile_environment(
        self,
        *,
        name: str,
        version: str | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | NDJsonStream:
        """Profile a validated environment version.

        :param name: Environment name, as specified in Arena.
        :param version: Environment version. Defaults to the latest.
        :param stream: If ``True``, return the raw :class:`NDJsonStream`.
        """
        payload: dict[str, Any] = {
            "name": name,
            "version": version,
        }
        stream_resp = self._open_stream(
            "POST",
            "/api/cli/v1/environments/profile",
            json=payload,
            timeout=self._upload_timeout,
        )
        return stream_resp if stream else stream_resp.collect()

    def delete_environment(self, *, name: str, version: str) -> Any:
        """Delete an environment version.

        :param name: Environment name, as specified in Arena.
        :type name: str
        :param version: Environment version. Defaults to "latest".
        :type version: str
        """
        payload = {"name": name, "version": version}
        return self._unwrap_cli_data(
            self._request("DELETE", "/api/cli/v1/environments/delete", json=payload)
        )

    # -------------------------------------------------------------------------
    ### Training Jobs ###
    # -------------------------------------------------------------------------

    # TODO: Backend needs to return useful logs
    # e.g. Submitting job with ID <job_id> to Arena
    #      Job <job_id> submitted successfully
    #      Job <job_id> is PENDING
    #      View training progress at <url>
    def submit_training_job(
        self,
        *,
        manifest: dict[str, Any],
        resource_id: int | None = None,
        num_nodes: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | NDJsonStream:
        """Submit a training job.

        :param manifest: The fully validated training manifest.
        :param resource_id: The Arena cluster type to submit the experiment to.
        :param stream: If ``True``, return the raw :class:`NDJsonStream`.
        """
        payload: dict[str, Any] = {
            "manifest": manifest,
            "resource_id": resource_id,
        }
        stream_resp = self._open_stream(
            "POST",
            "/api/cli/v1/experiments/jobs/submit",
            json=payload,
            timeout=self._upload_timeout,
        )
        return stream_resp if stream else stream_resp.collect()

    # TODO: Check with Rob
    # Should be a rich table showing [experiment name, job_id, env, algo, last_modified, status]
    def list_experiments(self, project: str) -> list[dict[str, Any]]:
        """List all experiments in a project.

        :param project: The name of the project.
        :type project: str
        :returns: A list of experiments.
        :rtype: list[dict[str, Any]]
        """
        return self._request(
            "GET", "/api/cli/v1/experiments/list", params={"project": project}
        )

    # TODO: Check with Rob
    # Is the only extra arg we should allow 'max_steps' here?
    def resume_training_job(self, job_id: str, max_steps: int) -> dict[str, Any]:
        """Resume a training job.

        :param job_id: The ID of the training job to resume.
        :type job_id: str
        :param max_steps: The maximum number of steps to train for.
        :type max_steps: int
        :returns: A dictionary containing the resume result.
        :rtype: dict[str, Any]
        """
        return self._request(
            "POST",
            "/api/cli/v1/experiments/jobs/resume",
            json={"job_id": job_id, "max_steps": max_steps},
        )

    # TODO: Check with Rob
    # Should be a rich table sorted by evaluation score descending showing
    # [steps, training_score, evaluation_score, size_mb]
    def list_checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        """List all checkpoints for a training job.

        :param job_id: The ID of the training job to list checkpoints for.
        :type job_id: str
        :returns: A list of checkpoints.
        :rtype: list[dict[str, Any]]
        """
        return self._request(
            "GET", "/api/cli/v1/experiments/jobs/checkpoints", params={"job_id": job_id}
        )

    def list_resources(self) -> list[dict[str, Any]]:
        """List all resources available to the authenticated user."""
        return self._request("GET", "/api/cli/v1/resources/list")

    def download_experiment_metrics(
        self, experiment_id: int, metrics: list[str]
    ) -> tuple[bytes, str | None, str | None]:
        """Download experiment metrics payload (CSV or zipped CSV).

        :param experiment_id: The ID of the experiment to download metrics for.
        :type experiment_id: int
        :param metrics: The metrics to download.
        :type metrics: list[str]
        :returns: A tuple of the metrics payload, content type, and disposition.
        :rtype: tuple[bytes, str | None, str | None]
        """
        return self._request_raw(
            "POST",
            f"/api/experiments/{experiment_id}/metrics",
            json={"metrics": metrics},
        )

    # TODO: Check with Rob if tested
    def stop_job(self, job_id: str) -> None:
        """Request stopping of a running job.

        :param job_id: Identifier returned by :meth:`submit_job`.
        """
        return self._request("POST", "api/v1/jobs/stop", params={"job_id": job_id})

    # -------------------------------------------------------------------------
    ### Projects ###
    # -------------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects in Arena."""
        # TODO: Modify endpoint
        return self._request("GET", "/api/projects")

    def create_project(
        self, name: str, description: str | None, llm_based: bool
    ) -> dict[str, Any]:
        """Create a new project in Arena."""
        # TODO: Modify endpoint
        return self._request(
            "POST",
            "/api/projects/create",
            json={"name": name, "description": description, "llm_based": llm_based},
        )

    def delete_project(self, name: str) -> None:
        """Delete a project in Arena."""
        # TODO: Modify endpoint
        return self._request("DELETE", "/api/projects/delete", json={"name": name})

    # -------------------------------------------------------------------------
    ### Inference ###
    # -------------------------------------------------------------------------

    # TODO: Check with Rob
    # My idea is that users train an agent, and when they come back they might want to check
    # the experiments they have trained on Arena
    # 1. list_experiments -> check job_id
    # 2. list_checkpoints <job_id> -> check checkpoint_id
    # 3. deploy_agent <job_id> <checkpoint>
    def deploy_agent(self, job_id: str, checkpoint: str = "best") -> None:
        """Deploy an agent to Arena.

        :param job_id: The ID of the training job to deploy an agent from.
        :type job_id: str
        :param checkpoint: The checkpoint to deploy.
        :type checkpoint: str
        """
        return self._request(
            "POST",
            "/api/cli/v1/inference/deploy",
            json={"job_id": job_id, "checkpoint": checkpoint},
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "authenticated" if self.is_authenticated else "unauthenticated"
        return f"<ArenaClient(url={self._base_url!r}, {status})>"

    def _create_and_validate(
        self,
        *,
        name: str,
        version: str,
        source: str | os.PathLike[str] | bytes,
        env_config: str | os.PathLike[str] | None,
        requirements: str | os.PathLike[str] | None,
        entrypoint: str | None,
        description: str | None,
        multi_agent: bool,
        do_rollouts: bool,
    ) -> NDJsonStream:
        """Upload, create, and validate an environment."""
        # Resolve the environment source into bytes for upload
        archive_name, archive_bytes = prepare_env_upload(source)

        data: dict[str, str] = {
            "name": name,
            "version": version,
            "multi_agent": str(multi_agent).lower(),
            "do_rollouts": str(do_rollouts).lower(),
        }
        if entrypoint:
            data["entrypoint"] = entrypoint
        if description:
            data["description"] = description

        files: dict[str, tuple[str, Any, str]] = {
            "file": (archive_name, archive_bytes, "application/gzip"),
        }

        # Check env_config and resolve to bytes for upload
        if env_config is not None:
            env_cfg = Path(os.fspath(env_config)).expanduser().resolve()
            if not env_cfg.is_file():
                msg = f"Upload file not found: {env_cfg}"
                raise FileNotFoundError(msg)
            files["env_config"] = (
                env_cfg.name,
                env_cfg.read_bytes(),
                "application/x-yaml",
            )
        else:
            files["env_config"] = ("env_config.yaml", b"", "application/x-yaml")

        # Check requirements and resolve to bytes for upload
        if requirements is not None:
            reqs = Path(os.fspath(requirements)).expanduser().resolve()
            if not reqs.is_file():
                msg = f"Upload file not found: {reqs}"
                raise FileNotFoundError(msg)
            files["requirements"] = (reqs.name, reqs.read_bytes(), "text/plain")
        else:
            files["requirements"] = ("requirements.txt", b"", "text/plain")

        return self._open_stream(
            "POST",
            "/api/cli/v1/environments/create-and-validate",
            data=data,
            files=files,
            timeout=self._upload_timeout,
        )

    def _try_restore_session(self) -> None:
        # Try to restore previously saved authentication credentials.
        creds = load_credentials(ArenaOAuth2.CREDENTIALS_FILE)
        if creds:
            self._tokens.access_token = creds.get("access_token")
            self._tokens.refresh_token = creds.get("refresh_token")

    def _auth_headers(self) -> dict[str, str]:
        # If an API key is provided, use it for authentication.
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}

        # If an access token from OAuth2 authentication is available
        if self._tokens.access_token:
            return {"Authorization": f"Bearer {self._tokens.access_token}"}

        msg = (
            "Client has not been authenticated with Arena. Call client.login() or provide an "
            "API key to the ArenaClient constructor."
        )
        raise ArenaAuthError(msg)

    def _send(
        self,
        method: str,
        path: str,
        *,
        stream: bool = False,
        timeout: int | None = None,
        _retried: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send an HTTP request with auth injection, 401-retry, and error handling.

        Returns a validated :class:`httpx.Response`.  When *stream* is
        ``True`` the response body is **not** read — the caller is
        responsible for consuming and closing it.
        """
        # Prepare request headers with authentication credentials.
        request_headers = dict(kwargs.pop("headers", {}))
        headers = dict(request_headers)
        headers.update(self._auth_headers())

        # Send the request.
        try:
            if stream:
                request = self._http.build_request(
                    method, path, headers=headers, timeout=timeout, **kwargs
                )
                resp = self._http.send(request, stream=True)
            else:
                resp = self._http.request(
                    method, path, headers=headers, timeout=timeout, **kwargs
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
            if stream:
                resp.close()

            # Attempt to refresh the access token.
            logger.debug("Access token expired, attempting refresh.")
            tokens = self._auth.refresh_access_token(self._tokens.refresh_token)
            self._tokens.access_token = tokens["access_token"]
            self._tokens.refresh_token = tokens.get(
                "refresh_token", self._tokens.refresh_token
            )
            return self._send(
                method,
                path,
                stream=stream,
                timeout=timeout,
                _retried=True,
                headers=request_headers,
                **kwargs,
            )

        # Handle 401 Unauthorized.
        if resp.status_code == 401:
            raw = self._read_response_body(resp, stream=stream)
            msg = (
                "Session expired and could not be refreshed. "
                f"Please run client.login() again. Server response: {raw[:200]}"
            )
            raise ArenaAuthError(msg)

        # Handle non-success responses.
        if not resp.is_success:
            raw = self._read_response_body(resp, stream=stream)
            error_cls = self._ERROR_MAP.get(path, ArenaAPIError)
            raise error_cls.from_response_body(raw, status_code=resp.status_code)

        return resp

    @staticmethod
    def _read_response_body(resp: httpx.Response, *, stream: bool) -> str:
        """Read the response body as a string and close if streamed."""
        try:
            if stream:
                return resp.read().decode("utf-8", errors="replace")
            return resp.text
        finally:
            if stream:
                resp.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Send a request and return the parsed JSON body (or text)."""
        resp = self._send(method, path, timeout=timeout, **kwargs)
        content_type: str = resp.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            return resp.json()
        return resp.text

    def _request_raw(
        self,
        method: str,
        path: str,
        *,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> tuple[bytes, str | None, str | None]:
        """Send a request and return ``(bytes, content_type, disposition)``."""
        resp = self._send(method, path, timeout=timeout, **kwargs)
        return (
            resp.content,
            resp.headers.get("content-type"),
            resp.headers.get("content-disposition"),
        )

    def _open_stream(
        self,
        method: str,
        path: str,
        *,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> NDJsonStream:
        """Send a streaming request and return an :class:`NDJsonStream`."""
        resp = self._send(method, path, stream=True, timeout=timeout, **kwargs)
        return NDJsonStream(resp, handler=self._stream_handler)

    @staticmethod
    def _unwrap_cli_data(resp: Any) -> Any:
        """Unwrap standardized CLI envelope responses when present.

        :param resp: The response to unwrap.
        :type resp: Any
        :returns: The unwrapped response.
        :rtype: Any
        """
        if isinstance(resp, dict) and resp.get("ok") is True and "data" in resp:
            return resp["data"]
        return resp
