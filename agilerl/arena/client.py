from __future__ import annotations

import io
import logging
import os
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import httpx

from agilerl.arena.auth import ArenaOAuth2, load_credentials
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaTrainingError,
    ArenaValidationError,
)
from agilerl.arena.output import StreamRichRenderer
from agilerl.arena.stream import NDJsonStream, StreamEvent
from agilerl.models.manifest import ArenaManifest

logger = logging.getLogger(__name__)


def prepare_env_upload(source: str | os.PathLike[str] | bytes) -> tuple[str, bytes]:
    """Resolve an environment source into an upload-ready ``(name, bytes)`` pair.

    *source* may be:

    * A path to a directory — compressed into ``.tar.gz`` automatically.
    * A path to an existing ``.tar.gz`` file — read as-is.
    * Raw ``bytes`` — used directly (assumed to be a valid ``.tar.gz``).

    :param source: The source of the environment.
    :type source: str | os.PathLike[str] | bytes
    :returns: The name and bytes of the prepared environment.
    :rtype: tuple[str, bytes]
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


# Deprecated hint: tier ids are dynamic — use :meth:`ArenaClient.list_resources`.
ArenaResource = Literal["arena-small", "arena-medium", "arena-large"]


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
        "/api/cli/v1/experiments/jobs/submit": ArenaTrainingError,
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
        request_timeout: int = 30,
        upload_timeout: int = 300,
        verbose: bool = True,
    ) -> None:

        self._base_url = self.BASE_URL.rstrip("/")
        self._request_timeout = request_timeout
        self._upload_timeout = upload_timeout

        self._api_key = api_key or os.environ.get("ARENA_API_KEY")
        self._auth = ArenaOAuth2()
        self._tokens = _TokenStore()
        self._verbose = verbose
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
        logger.info("Authenticated successfully with Arena.")

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

    def list_environments(
        self,
        name: str | None = None,
        include_arena: str | None = None,
    ) -> list[dict[str, Any]]:
        """List environments available to the authenticated user.

        :param name: Environment name. If None, list all environments.
        :type name: str | None
        :returns: A list of environments.
        :rtype: list[dict[str, Any]]
        """
        return self._request(
            "GET",
            "/api/cli/v1/environments",
            params={"name": name, "include_arena": include_arena},
        )

    def environment_exists(self, name: str, version: str | None = None) -> bool:
        """Check whether an environment name/version is registered.

        :param name: Environment name.
        :type name: str
        :param version: Environment version. Defaults to None, which resolves to the latest version.
        :type version: str
        :returns: True if the environment exists, False otherwise.
        :rtype: bool
        """
        resp = self._request(
            "GET",
            "/api/cli/v1/environments/exists",
            params={"name": name, "version": version},
        )

        if isinstance(resp, dict):
            for key in ("exists", "is_registered", "isRegistered"):
                if key in resp:
                    return bool(resp[key])
        return bool(resp)

    # TODO: In general, for all endpoints that take in a name and version, if the version
    # is None, we should resolve to the latest version in the backend and return an INFO log
    # saying "No version specified, resolving to latest version {latest_version}". The only
    # exception is create-and-validate which should always expect a version to be given.
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
        resp = self._request(
            "GET",
            "/api/cli/v1/environments/entrypoints",
            params={"name": name, "version": version},
        )

        assert type(resp) is list, "List entrypoints response should be a list"

        return resp

    def validate_environment(
        self,
        *,
        name: str | None = None,
        version: str | None = None,
        source: str | os.PathLike[str] | bytes | None = None,
        env_config: str | os.PathLike[str] | None = None,
        requirements: str | os.PathLike[str] | None = None,
        entrypoint: str | None = None,
        description: str | None = None,
        multi_agent: bool = False,
        do_rollouts: bool = True,
    ) -> dict[str, Any]:
        """Validate a custom environment on Arena.

        When *source* is provided the environment is uploaded, created, and
        validated in a single step.  When *source* is ``None`` an
        already-registered environment is validated by *name*/*version*.

        :param name: Environment name.
        :type name: str | None
        :param version: Environment version. If creating an environment from scratch, defaults to "v1",
            if validating an already-registered environment, defaults to None, which resolves to the latest version.
        :type version: str | None
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

        :returns: A dictionary containing the validation result.
        :rtype: dict[str, Any]
        """
        if name is None and source is None:
            msg = (
                "To validate an environment on Arena, either the name of an already "
                "registered environment or the source of a custom environment must be provided."
            )
            raise ValueError(msg)

        if source is not None:
            if version is None:
                logger.info("No version specified, defaulting to v1.")
                version = "v1"

            return self._create_and_validate(
                name=name,
                version=version,
                source=source,
                env_config=env_config,
                requirements=requirements,
                entrypoint=entrypoint,
                description=description,
                multi_agent=multi_agent,
                do_rollouts=do_rollouts,
            ).collect()

        payload: dict[str, Any] = {
            "name": name,
            "version": version,
            "do_rollouts": do_rollouts,
        }
        if entrypoint:
            payload["entrypoint"] = entrypoint

        return self._open_stream(
            "POST",
            "/api/cli/v1/environments/validate",
            json=payload,
            timeout=self._upload_timeout,
        ).collect()

    def profile_environment(
        self,
        *,
        name: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Profile a validated environment version.

        :param name: Environment name, as specified in Arena.
        :type name: str
        :param version: Environment version. Defaults to None, which resolves to the latest version.
        :type version: str | None
        """
        payload: dict[str, Any] = {
            "name": name,
            "version": version,
        }
        return self._open_stream(
            "POST",
            "/api/cli/v1/environments/profile",
            json=payload,
            timeout=self._upload_timeout,
        ).collect()

    def delete_environment(self, *, name: str, version: str | None = None) -> Any:
        """Delete an environment version.

        :param name: Environment name, as specified in Arena.
        :type name: str
        :param version: Environment version. If None, delete all environment versions.
        :type version: str | None
        """
        if version is None:
            # Fetch existing versions (assuming you have a list_environments method)
            versions_data = self.list_environments(name=name)
            if name in versions_data:
                versions_data = versions_data[name]
            version_list = versions_data.keys()

            if not version_list:
                logger.info(
                    "No versions found for environment '%s'. Nothing to delete.", name
                )
                return None

            if version not in version_list:
                logger.info(
                    "Version '%s' not found in environment '%s'. Please specify a version to be deleted.",
                    version,
                    name,
                )
                return {"deleted": False, "name": name, "version": version}

            # Format and Prompt
            logger.info(
                "The following versions for '%s' will be deleted: %s",
                name,
                ", ".join(version_list),
            )
            confirm = input("Do you wish to continue? [y/N]: ").strip().lower()

            if confirm not in ("y", "yes"):
                logger.info("Delete operation cancelled.")
                return None

        # 2. Proceed with the request
        payload = {"name": name, "version": version}
        return self._request("DELETE", "/api/cli/v1/environments/delete", json=payload)

    def duplicate_environment_version(
        self,
        *,
        name: str,
        new_version_name: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Duplicate a custom environment version to a new version name.

        Copies registered artifacts (and validation outputs) and creates a new
        version row, same as the platform **POST**
        ``/api/custom-gym-env-impls/{id}/copy`` flow.

        :param name: Environment name.
        :param new_version_name: New ``version_name`` for the duplicate (e.g. ``v2``).
        :param version: Source version; when omitted, the latest version is used.
        """
        payload: dict[str, Any] = {
            "name": name,
            "new_version_name": new_version_name,
            "version": version,
        }
        return self._request(
            "POST",
            "/api/cli/v1/environments/duplicate",
            json=payload,
        )

    # -------------------------------------------------------------------------
    ### Training Jobs ###
    # -------------------------------------------------------------------------

    def submit_experiment(
        self,
        manifest: str | os.PathLike[str] | dict[str, Any],
        *,
        resource_id: int | None = None,
        num_nodes: int | None = None,
        project: str | None = None,
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """Submit an experiment (a training job).

        :param manifest: Training manifest as a YAML/JSON file path, raw YAML
            string, or a pre-parsed dict.
        :type manifest: str | os.PathLike[str] | dict[str, Any]
        :param resource_id: The Arena resource to submit the experiment to.
        :type resource_id: int | None
        :param num_nodes: The number of nodes to use for training.
        :type num_nodes: int | None
        :param project: The project to submit the experiment to.
        :type project: str | None
        :param experiment_name: The name of the experiment to submit.
        :type experiment_name: str | None
        """
        # Pre-flight Pydantic manifest validation prior to submitting to Arena
        validated = ArenaManifest.get_validated(manifest, mode="json")

        payload: dict[str, Any] = {
            "manifest": validated,
            "resource_id": resource_id,
            "num_nodes": num_nodes,
            "project": project,
            "experiment_name": experiment_name,
        }
        return self._open_stream(
            "POST",
            "/api/cli/v1/experiments/jobs/submit",
            json=payload,
            timeout=self._upload_timeout,
        ).collect()

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
    def resume_experiment(self, experiment_name: str, max_steps: int) -> dict[str, Any]:
        """Resume an experiment (a training job).

        :param experiment_name: The name of the experiment to resume.
        :type experiment_name: str
        :param max_steps: The maximum number of steps to train for.
        :type max_steps: int
        :returns: A dictionary containing the resume result.
        :rtype: dict[str, Any]
        """
        return self._request(
            "POST",
            "/api/cli/v1/experiments/jobs/resume",
            json={"experiment_name": experiment_name, "max_steps": max_steps},
        )

    # TODO: Update HPO params (maybe leave for v2 if too complicated)

    # Should be a rich table sorted by evaluation score descending showing
    # [steps, training_score, evaluation_score, size_mb]
    def list_checkpoints(self, experiment_name: str) -> list[dict[str, Any]]:
        """List all checkpoints for an experiment.

        :param experiment_name: The name of the experiment to list checkpoints for.
        :type experiment_name: str
        :returns: A list of checkpoints.
        :rtype: list[dict[str, Any]]
        """
        return self._request(
            "GET",
            "/api/cli/v1/experiments/jobs/checkpoints",
            params={"experiment_name": experiment_name},
        )

    def preview_experiment_metrics_csv(
        self,
        experiment_name: str,
        *,
        preview_rows: int,
        metrics: Sequence[str] | None = None,
        project: str | None = None,
    ) -> tuple[bytes, str | None, str | None]:
        """Fetch a capped CSV snippet (Arena CLI ``--metric`` / ``--preview-rows``).

        Uses ``GET /api/cli/v1/experiments/metrics`` with ``preview_rows`` set.
        Omit ``metrics`` to include all columns.

        :param experiment_name: Experiment name (latest match in scope).
        :param preview_rows: Maximum number of **data** rows in the CSV (server-capped).
        :param metrics: Metric column names to include (repeat query param ``metric``).
        :param project: Optional exact project name.
        """
        params: list[tuple[str, Any]] = [
            ("experiment_name", experiment_name),
            ("preview_rows", preview_rows),
        ]
        if project is not None:
            params.append(("project", project))
        if metrics:
            for m in metrics:
                params.extend(("metric", m))
        return self._request_raw(
            "GET",
            "/api/cli/v1/experiments/metrics",
            params=params,
        )

    def list_experiment_metric_names(
        self,
        experiment_name: str,
        *,
        project: str | None = None,
        details: bool = False,
    ) -> list[str] | dict[str, Any]:
        r"""List metric column names recorded for an experiment (JSON).

        For a **CSV preview** with ``--metric`` / ``--preview-rows``-style filters,
        use :meth:`preview_experiment_metrics_csv`.

        :param experiment_name: Experiment name (latest updated match in scope).
        :param project: Optional exact project name in the current org.
        :param details: When True, the API returns ``{\"experiment_id\", \"metrics\"}``.
        :returns: Sorted unique metric names, or that object when ``details`` is True.
        """
        params: dict[str, Any] = {"experiment_name": experiment_name}
        if project is not None:
            params["project"] = project
        if details:
            params["details"] = True
        return self._request(
            "GET",
            "/api/cli/v1/experiments/metrics",
            params=params,
        )

    def list_resources(self) -> dict[str, Any]:
        """List compute resource tiers for training (CLI resource_id values).

        Calls **GET** ``/api/cli/v1/resources/list`` (see ``list_resources`` in
        ``agilerl-platform`` ``cli.rs``). The JSON body uses camelCase:
        ``resourceIds`` (public tier ids usable as ``resource_id`` on submit)
        and ``tiers`` (map of id → ``numCpus``, ``numGpus``, ``gpuType``,
        ``ramGb``, ``pricePerNodeHour``).

        :returns: Unwrapped ``data`` object from the API envelope when present.
        """
        return self._request("GET", "/api/cli/v1/resources/list")

    def download_experiment_metrics(
        self,
        experiment_name: str,
        metrics: list[str] | None = None,
    ) -> tuple[bytes, str | None, str | None]:
        """Download experiment metrics payload (CSV or zipped CSV).

        :param experiment_name: The name of the experiment to download metrics for.
        :type experiment_name: str
        :param metrics: The metrics to download. If None, download all metrics.
        :type metrics: list[str] | None
        :returns: A tuple of the metrics payload, content type, and disposition.
        :rtype: tuple[bytes, str | None, str | None]
        """
        return self._request_raw(
            "POST",
            f"/api/cli/v1/experiments/{experiment_name}/metrics",
            json={"metrics": metrics},
        )

    def stop_experiment(self, experiment_name: str) -> Any:
        """Stop a running experiment (training job) via the CLI halt path.

        Calls **POST** ``/api/cli/v1/experiments/jobs/stop`` (same backend as the UI
        halt command). There is no ``POST /api/v1/jobs/stop`` on the Arena platform.

        :param experiment_name: Experiment name (a.k.a. job name in the CLI).
        """
        name = experiment_name.strip()
        if not name:
            msg = "experiment_name must be non-empty."
            raise ValueError(msg)

        return self._request(
            "POST",
            "/api/cli/v1/experiments/jobs/stop",
            json={"experiment_name": name},
        )

    def stop_job(self, job_name: str) -> Any:
        """Deprecated. Use :meth:`stop_experiment` with the experiment (job) name.

        The former URL was not a platform route; this now calls the CLI stop API.
        """
        import warnings

        warnings.warn(
            "ArenaClient.stop_job is deprecated; use stop_experiment(...).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.stop_experiment(job_name)

    # -------------------------------------------------------------------------
    ### Projects ###
    # -------------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects in Arena."""
        return self._request("GET", "/api/cli/v1/projects")

    def create_project(
        self, name: str, description: str | None, llm_based: bool
    ) -> dict[str, Any]:
        """Create a new project in Arena."""
        return self._request(
            "POST",
            "/api/cli/v1/projects/create",
            json={"name": name, "description": description, "llm_based": llm_based},
        )

    def delete_project(self, name: str) -> None:
        """Delete a project in Arena."""
        return self._request(
            "DELETE", "/api/cli/v1/projects/delete", json={"name": name}
        )

    # -------------------------------------------------------------------------
    ### Inference ###
    # -------------------------------------------------------------------------

    # TODO: Make endpoint for deploying an agent
    def deploy_agent(self, experiment_name: str, checkpoint: str | None = None) -> None:
        """Deploy an agent to Arena.

        :param experiment_name: The name of the experiment to deploy an agent from.
        :type experiment_name: str
        :param checkpoint: The checkpoint to deploy. If None, deploy the best checkpoint.
        :type checkpoint: str | None
        """
        return self._request(
            "POST",
            "/api/cli/v1/inference/deploy",
            json={"experiment_name": experiment_name, "checkpoint": checkpoint},
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

        msg = "Client has not been authenticated with Arena."
        raise ArenaAuthError(
            msg,
            sdk_hint="Call client.login() or provide an API key to the ArenaClient constructor.",
            cli_hint="Run 'arena login' to authenticate.",
        )

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
            msg = f"Session expired and could not be refreshed. Server response: {raw[:200]}"
            raise ArenaAuthError(
                msg,
                sdk_hint="Please run client.login() again.",
                cli_hint="Please run 'arena login' to re-authenticate.",
            )

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
            data = resp.json()
            if isinstance(data, dict) and data.get("ok") is True and "data" in data:
                return data["data"]
            return data
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
        handler = self._stream_handler
        renderer: StreamRichRenderer | None = None
        if handler is None and self._verbose:
            error_cls = self._ERROR_MAP.get(path, ArenaAPIError)
            renderer = StreamRichRenderer(error_cls=error_cls)
            handler = renderer.handle_event
        resp = self._send(method, path, stream=True, timeout=timeout, **kwargs)
        return NDJsonStream(resp, handler=handler, renderer=renderer)
