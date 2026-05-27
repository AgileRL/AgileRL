from __future__ import annotations

import io
import json
import logging
import os
import tarfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import httpx

from agilerl.arena.auth import (
    ArenaOAuth2,
    is_oauth_access_token_valid,
    load_credentials,
    oauth_access_token_expires_at,
)
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaConfigError,
    ArenaFileNotFoundError,
    ArenaTrainingError,
    ArenaValidationError,
)
from agilerl.arena.inference import Agent, RLData
from agilerl.arena.inference_cache import (
    load_inference_binding,
    normalized_deployment_name,
    save_inference_binding,
)
from agilerl.arena.output import StreamRichRenderer
from agilerl.arena.stream import NDJsonStream, StreamEvent
from agilerl.models.manifest import ArenaManifest

logger = logging.getLogger(__name__)


def _extract_filename(disposition: str | None) -> str | None:
    """Parse a filename from a Content-Disposition header value."""
    if not disposition:
        return None
    for part in disposition.split(";"):
        part = part.strip()
        if part.startswith("filename="):
            return part.removeprefix("filename=").strip('"')
    return None


def prepare_env_upload(source: str | os.PathLike[str] | bytes) -> tuple[str, bytes]:
    """Resolve an environment source into an upload-ready ``(name, bytes)`` pair.

    *source* may be:

    * A path to a directory — compressed into ``.tar.gz`` automatically.
    * A path to a single file — compressed into ``.tar.gz`` automatically.
    * A path to an existing ``.tar.gz`` file — read as-is.
    * Raw ``bytes`` — used directly (assumed to be a valid ``.tar.gz``).

    :param source: The source of the environment.
    :type source: str | os.PathLike[str] | bytes
    :returns: The name and bytes of the prepared environment.
    :rtype: tuple[str, bytes]
    :raises ArenaFileNotFoundError: If *source* is a path that does not exist.
    """
    if isinstance(source, bytes):
        return ("environment.tar.gz", source)

    path = Path(os.fspath(source)).expanduser().resolve()

    # Directory
    if path.is_dir():
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    tar.add(str(child), arcname=child.relative_to(path).as_posix())
        return (f"{path.name}.tar.gz", buf.getvalue())

    # Single file
    if path.is_file():
        if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
            return (path.name, path.read_bytes())
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(path), arcname=path.name)
        return (f"{path.stem}.tar.gz", buf.getvalue())

    msg = f"Source path not found: {path}"
    raise ArenaFileNotFoundError(msg)


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

    For (1) and (2), the value is sent as ``Authorization: Bearer <value>``.
    Use a **personal access token** from your Arena account profile (``arena_pat_<uuid>_<secret>``)
    to skip OAuth device login. You can also pass a Keycloak access token in the same way
    if you obtain one elsewhere.

    :param api_key: Bearer token material (profile PAT or OAuth access token). When set, device login is not required.
    :param request_timeout: Default timeout in seconds for API requests.
    :param upload_timeout: Timeout in seconds for file-upload requests.
    """

    # TODO: Remove this once we have a production URL
    # BASE_URL: ClassVar[str] = "https://arena.agilerl.com"
    # BASE_URL: ClassVar[str] = "https://arena-dev.agilerl.rlops.ai"
    BASE_URL: ClassVar[str] = "http://localhost:3001"
    CONFIG_DIR: ClassVar[Path] = Path.home() / ".arena"
    CONFIG_FILE: ClassVar[Path] = CONFIG_DIR / "config.json"

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

    @classmethod
    def _read_config(cls) -> dict[str, Any]:
        if not cls.CONFIG_FILE.is_file():
            return {}
        try:
            return json.loads(cls.CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    @classmethod
    def _write_config(cls, data: dict[str, Any]) -> None:
        cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cls.CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")

    def get_default_project(self) -> str | None:
        """Return the default project name from ``~/.arena/config.json``, or None if unset."""
        return self._read_config().get("default_project")

    def set_default_project(self, name: str) -> None:
        """Validate that *name* exists and persist it as the default project.

        :param name: The project name to set as default.
        :type name: str
        :raises ArenaConfigError: If the project does not exist.
        """
        existing = self.list_projects()
        names = [p["name"] for p in existing]
        if name not in names:
            hint = f"Available projects: {', '.join(names) or 'None'}. "
            msg = f"Project {name!r} not found."
            raise ArenaConfigError(msg, sdk_hint=hint, cli_hint=hint)
        config = self._read_config()
        config["default_project"] = name
        self._write_config(config)

    def _resolve_project(self, project: str | None) -> str | None:
        """Return *project* if given, otherwise fall back to the stored default."""
        return project if project is not None else self.get_default_project()

    # -------------------------------------------------------------------------
    ### Authentication ###
    # -------------------------------------------------------------------------

    def login(self, *, timeout: int = 300, force: bool = False) -> None:
        """Start the device-authorization login flow (or reuse a valid stored session).

        When *force* is false (default) and an API key is set, device login is
        skipped. Pass ``force=True`` to run device authorization regardless
        (useful when the API key is invalid and you want to switch to OAuth).

        When *force* is false (default), an unexpired OAuth access token or a
        successful refresh from ``~/.arena/credentials.json`` skips the browser
        flow. Use *force* to always run device authorization.
        On success the tokens are persisted to
        ``~/.arena/credentials.json``.
        """
        if self._api_key is not None and not force:
            logger.info(
                "API key in use; device login not required. Use --force to override."
            )
            return

        if self._api_key is not None and force:
            logger.info(
                "Forcing device login; API key will be ignored for this session."
            )
            self._api_key = None

        if not force:
            if is_oauth_access_token_valid(self._tokens.access_token):
                logger.info("Access token still valid; skipping device login.")
                return

            if self._tokens.refresh_token:
                try:
                    tokens = self._auth.refresh_access_token(self._tokens.refresh_token)
                    self._tokens.access_token = tokens["access_token"]
                    self._tokens.refresh_token = tokens.get(
                        "refresh_token", self._tokens.refresh_token
                    )
                    logger.info(
                        "Session refreshed from stored credentials; skipping device login."
                    )
                    return
                except ArenaAuthError:
                    logger.debug(
                        "Stored refresh token rejected; starting device login.",
                    )

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
        include_arena: bool = False,
    ) -> dict[str, dict[str, dict[str, bool]]]:
        """List environments available to the authenticated user.

        :param name: Environment name. If None, list all environments.
        :type name: str | None
        :param include_arena: Whether to include off-the-shelf Gymnasium/PettingZoo environments
            available in Arena.
        :type include_arena: bool
        :returns: A nested dictionary keyed by environment name, then version:

            .. code-block:: python

                {
                    "<name>": {
                        "<version>": {
                            "validated": True,
                            "profiled": True,
                        }
                    }
                }

        :rtype: dict[str, dict[str, dict[str, bool]]]
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
        logger.info(
            "Found %d entrypoints for environment %s:%s.",
            len(resp["entrypoints"]),
            name,
            resp["version"],
        )
        return resp["entrypoints"]

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
        do_rollouts: bool = False,
    ) -> dict[str, Any]:
        """Validate a custom environment on Arena.

        When source is provided the environment is uploaded, created, and
        validated in a single step.  When source is None an
        already-registered environment is validated by name/version.

        :param name: Environment name.
        :type name: str | None
        :param version: Environment version. If creating an environment from scratch, defaults to "v1",
            if validating an already-registered environment, defaults to None, which resolves to the latest version.
        :type version: str | None
        :param source: Environment source — a directory path (compressed
            automatically), a ``.tar.gz`` file path, or raw ``bytes``.
        :type source: str | os.PathLike[str] | bytes | None
        :param env_config: Path to the environment configuration file containing the environment parameters. Default is None.
        :type env_config: str | os.PathLike[str] | None
        :param requirements: Path to additional dependencies needed for the environment. Default is None.
        :type requirements: str | os.PathLike[str] | None
        :param entrypoint: Optional entrypoint override. Default is None.
        :type entrypoint: str | None
        :param description: Optional description of the environment. Default is None.
        :type description: str | None
        :param multi_agent: Whether the environment is multi-agent. Default is False.
        :type multi_agent: bool
        :param do_rollouts: Whether to perform environment rollouts during validation. Setting this to True will
            run 100 random episodes and collect additional information such as the average random reward and visualize
            the rendered environment. Default is False.
        :type do_rollouts: bool

        :returns: A dictionary containing the validation result.
        :rtype: dict[str, Any]
        """
        if name is None and source is None:
            msg = (
                "To validate an environment on Arena, either the name of an already "
                "registered environment or the source of a custom environment must be provided."
            )
            raise ArenaValidationError(msg)

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

    def delete_environment(
        self, *, name: str, version: str | None = None, confirm: bool = False
    ) -> Any:
        """Delete an environment version (or all versions if version is None).

        :param name: Environment name, as specified in Arena.
        :type name: str
        :param version: Environment version. If None, delete all environment versions.
        :type version: str | None
        :param confirm: Whether to confirm the deletion.
        :type confirm: bool
        """
        # Fetch existing versions
        versions_data = self.list_environments(name=name)
        if name in versions_data:
            versions_data = versions_data[name]

        version_list = versions_data.keys()

        if not version_list:
            logger.info(
                "No versions found for environment '%s'. Nothing to delete.", name
            )
            return None

        if version is None:
            logger.info(
                "The following versions for '%s' will be deleted: %s",
                name,
                ", ".join(version_list),
            )
        else:
            if version not in version_list:
                logger.info(
                    "Version '%s' not found in environment '%s'. Please specify a valid version from the list: %s.",
                    version,
                    name,
                    ", ".join(version_list),
                )
                return None

            logger.info(
                "The following version for '%s' will be deleted: %s",
                name,
                version,
            )

        confirm_prompt = input("Do you wish to continue? [y/N]: ").strip().lower()
        confirm = (confirm_prompt in ("y", "yes")) or confirm
        if not confirm:
            logger.info("No environment was deleted for %s.", name)
            return None
        payload = {"name": name, "version": version}
        return self._request("DELETE", "/api/cli/v1/environments/delete", json=payload)

    def duplicate_environment_version(
        self,
        *,
        name: str,
        new_version: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Duplicate a custom environment version to a new version name.

        :param name: Environment name.
        :type name: str
        :param new_version: New ``version`` for the duplicate (e.g. ``v2``).
        :type new_version: str
        :param version: Source version; when omitted, the latest version is used.
        :type version: str | None
        """
        payload: dict[str, Any] = {
            "name": name,
            "new_version_name": new_version,
            "version": version,
        }
        resp = self._request(
            "POST",
            "/api/cli/v1/environments/duplicate",
            json=payload,
        )
        logger.info(
            "Environment %s:%s duplicated to %s:%s.",
            name,
            resp["source_version"],
            name,
            new_version,
        )
        return resp

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
            "project": self._resolve_project(project),
            "experiment_name": experiment_name,
        }
        return self._open_stream(
            "POST",
            "/api/cli/v1/experiments/jobs/submit",
            json=payload,
            timeout=self._upload_timeout,
        ).collect()

    def submit_training_job(
        self,
        manifest: str | os.PathLike[str] | dict[str, Any],
        *,
        resource_id: int | None = None,
        num_nodes: int | None = None,
        project: str | None = None,
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """Submit a training job to Arena (alias for :meth:`submit_experiment`)."""
        return self.submit_experiment(
            manifest,
            resource_id=resource_id,
            num_nodes=num_nodes,
            project=project,
            experiment_name=experiment_name,
        )

    def list_experiments(self, project: str | None = None) -> list[dict[str, Any]]:
        """List all experiments in a project.

        :param project: The name of the project. Falls back to the default
            project from ``~/.arena/config.json`` if not provided.
        :type project: str | None
        :returns: A list of experiments.
        :rtype: list[dict[str, Any]]
        """
        resolved = self._resolve_project(project)
        if not resolved:
            msg = "No project specified."
            raise ArenaConfigError(
                msg,
                sdk_hint="Pass a project name or set a default with ArenaClient.set_default_project().",
                cli_hint="Use --project or set a default with 'arena projects set-default <name>'.",
            )
        return self._request(
            "GET", "/api/cli/v1/experiments/list", params={"project": resolved}
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

    def resume_training_job(
        self, experiment_name: str, max_steps: int
    ) -> dict[str, Any]:
        """Resume a training job (alias for :meth:`resume_experiment`)."""
        return self.resume_experiment(
            experiment_name=experiment_name, max_steps=max_steps
        )

    # TODO: Update HPO params (maybe leave for v2 if too complicated)

    # TODO: Check this works
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

    # TODO: Check this works
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
        :type experiment_name: str
        :param preview_rows: Maximum number of **data** rows in the CSV (server-capped).
        :type preview_rows: int
        :param metrics: Metric column names to include (repeat query param ``metric``).
        :type metrics: Sequence[str] | None
        :param project: Optional exact project name.
        :type project: str | None
        :returns: A tuple of the metrics payload, content type, and disposition.
        :rtype: tuple[bytes, str | None, str | None]
        """
        resolved_project = self._resolve_project(project)
        params: list[tuple[str, Any]] = [
            ("experiment_name", experiment_name),
            ("preview_rows", preview_rows),
        ]
        if resolved_project is not None:
            params.append(("project", resolved_project))
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
        :type experiment_name: str
        :param project: Optional exact project name in the current org.
        :type project: str | None
        :param details: When True, the API returns ``{\"experiment_id\", \"metrics\"}``.
        :type details: bool
        :returns: Sorted unique metric names, or that object when ``details`` is True.
        :rtype: list[str] | dict[str, Any]
        """
        resolved_project = self._resolve_project(project)
        params: dict[str, Any] = {"experiment_name": experiment_name}
        if resolved_project is not None:
            params["project"] = resolved_project
        if details:
            params["details"] = True
        return self._request(
            "GET",
            "/api/cli/v1/experiments/metrics",
            params=params,
        )

    def list_resources(self) -> dict[str, Any]:
        """List compute resource tiers for Arena training jobs.

        Any of the listed resource IDs can be used as the `resource_id` parameter when submitting a training job
        through `ArenaClient.submit_experiment`.

        :returns: A dictionary of resource tiers.
        :rtype: dict[str, Any]
        """
        return self._request("GET", "/api/cli/v1/resources/list")

    def download_experiment_metrics(
        self,
        experiment_name: str,
        output_path: str | os.PathLike[str] | None = None,
        metrics: list[str] | None = None,
    ) -> Path:
        """Download experiment metrics to a local file.

        :param experiment_name: The name of the experiment to download metrics for.
        :type experiment_name: str
        :param output_path: Destination file path or directory. If a directory,
            the filename is inferred from the server response.
            Defaults to ``{experiment_name}_metrics.csv`` in the current directory.
        :type output_path: str | os.PathLike[str] | None
        :param metrics: The metrics to download. If None, download all metrics.
        :type metrics: list[str] | None
        :returns: The path to the written file.
        :rtype: Path
        :raises FileExistsError: If the resolved output path already exists.
        """
        payload, _, disposition = self._request_raw(
            "POST",
            f"/api/cli/v1/experiments/{experiment_name}/metrics",
            json={"metrics": metrics},
        )

        if output_path is None:
            path = Path(f"{experiment_name}_metrics.csv")
        else:
            path = Path(output_path)
            if path.is_dir():
                filename = (
                    _extract_filename(disposition) or f"{experiment_name}_metrics.csv"
                )
                path = path / filename

        if path.exists():
            msg = f"Output path already exists: {path}. Please remove it or specify a different path."
            raise FileExistsError(msg)

        path.write_bytes(payload)
        logger.info("Metrics saved to %s", path)
        return path

    def stop_experiment(self, experiment_name: str) -> Any:
        """Stop a running experiment in Arena.

        :param experiment_name: Experiment name to halt.
        :type experiment_name: str
        """
        return self._request(
            "POST",
            "/api/cli/v1/experiments/jobs/stop",
            json={"experiment_name": experiment_name.strip()},
        )

    # -------------------------------------------------------------------------
    ### Projects ###
    # -------------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects in Arena for the authenticated user.

        :returns: A list of projects.
        :rtype: list[dict[str, Any]]
        """
        resp = self._request("GET", "/api/cli/v1/projects")
        if resp:
            return [
                {"name": p["name"], "type": p["type"], "description": p["description"]}
                for p in resp
            ]
        return []

    def create_project(
        self, name: str, description: str | None, llm_based: bool
    ) -> dict[str, Any]:
        """Create a new project in Arena.

        :param name: The name of the project to create.
        :type name: str
        :param description: The description of the project to create.
        :type description: str | None
        :param llm_based: Whether the project is based on an LLM.
        :type llm_based: bool
        :returns: A dictionary containing the project creation result.
        :rtype: dict[str, Any]
        """
        resp = self._request(
            "POST",
            "/api/cli/v1/projects/create",
            json={"name": name, "description": description, "llm_based": llm_based},
        )
        logger.info("Project %s created successfully.", name)
        return resp

    def delete_project(self, name: str) -> None:
        """Delete a project in Arena.

        :param name: The name of the project to delete.
        :type name: str
        """
        resp = self._request(
            "DELETE", "/api/cli/v1/projects/delete", json={"name": name}
        )
        logger.info("Project %s deleted successfully.", name)
        return resp

    # -------------------------------------------------------------------------
    ### Inference ###
    # -------------------------------------------------------------------------

    @staticmethod
    def _inference_deployments_list_params(
        *,
        name: str | None = None,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if name is not None and name.strip():
            params["name"] = name.strip()
        en = experiment_name.strip() if experiment_name else ""
        if en:
            params["experimentName"] = en
        pn = project_name.strip() if project_name else ""
        if pn:
            params["projectName"] = pn
        return params

    def deploy_agent(self, experiment_name: str, checkpoint: str | None = None) -> Any:
        """Create an inference deployment from an experiment checkpoint.

        :param experiment_name: The name of the experiment to deploy.
        :type experiment_name: str
        :param checkpoint: The checkpoint to deploy. If None, deploy the best checkpoint.
        :type checkpoint: str | None
        :returns: A dictionary containing the deployment result.
        :rtype: dict[str, Any]
        """
        return self._request(
            "POST",
            "/api/cli/v1/inference/deploy",
            json={"experiment_name": experiment_name, "checkpoint": checkpoint},
        )

    def list_inference_deployments(
        self,
        *,
        name: str | None = None,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List inference deployments available to the user.

        :param name: The name of the deployment to list.
        :type name: str | None
        :param experiment_name: The name of the experiment to list deployments for.
        :type experiment_name: str | None
        :param project_name: The name of the project to list deployments for.
        :type project_name: str | None
        :returns: A list of deployments.
        :rtype: list[dict[str, Any]]
        """
        q = self._inference_deployments_list_params(
            name=name,
            experiment_name=experiment_name,
            project_name=project_name,
        )
        rows = self._request(
            "GET",
            "/api/cli/v1/inference/deployments/list",
            params=q or None,
        )
        if not isinstance(rows, list):
            return []
        return [r for r in rows if isinstance(r, dict)]

    def fetch_deployment_for_inference(
        self,
        deployment_name: str,
        *,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """Load deployments visible to the user; expects exactly one row for *deployment_name*."""
        params = self._inference_deployments_list_params(
            name=normalized_deployment_name(deployment_name),
            experiment_name=experiment_name,
            project_name=project_name,
        )

        rows = self._request(
            "GET",
            "/api/cli/v1/inference/deployments/list",
            params=params,
        )
        if not isinstance(rows, list):
            rows = []

        hint = (
            "Pass --experiment-name and/or --project-name when multiple deployments "
            "share this deployment name."
        )
        if len(rows) == 0:
            msg = f"No deployment found named {deployment_name!r}."
            raise ArenaAPIError(msg, cli_hint=hint)
        if len(rows) > 1:
            msg = (
                f"Multiple deployments named {deployment_name!r} ({len(rows)} matches)."
            )
            raise ArenaAPIError(msg, cli_hint=hint)

        row = rows[0]
        if not isinstance(row, dict):
            msg = "Unexpected deployment list response shape."
            raise ArenaAPIError(msg)
        return row

    @staticmethod
    def deployment_url_and_api_key(row: dict[str, Any]) -> tuple[str, str]:
        """Parse ``spec.url`` and deployment ``api_key`` from an API deployment row."""
        spec = row.get("spec")
        if not isinstance(spec, dict):
            spec = {}
        url = spec.get("url")
        if not isinstance(url, str) or not url.strip():
            msg = "Deployment has no inference URL (spec.url)."
            raise ArenaAPIError(
                msg,
                cli_hint="Wait until provisioning completes, then retry with --refresh.",
            )

        raw_key = row.get("api_key")
        if raw_key is None:
            msg = "Deployment record had no api_key."
            raise ArenaAPIError(
                msg,
                cli_hint="Retry with arena login and --refresh.",
            )
        api_key = str(raw_key).strip()
        if not api_key:
            msg = "Deployment api_key was empty."
            raise ArenaAPIError(msg)

        return url.strip(), api_key

    def ensure_inference_binding(
        self,
        deployment_name: str,
        *,
        refresh: bool = False,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> tuple[str, str]:
        """Return cached ``(url, api_key)`` or fetch from the API, persist, and return."""
        key = normalized_deployment_name(deployment_name)
        if not refresh:
            cached = load_inference_binding(key)
            if cached is not None:
                return cached

        row = self.fetch_deployment_for_inference(
            deployment_name,
            experiment_name=experiment_name,
            project_name=project_name,
        )
        url, api_key = self.deployment_url_and_api_key(row)
        save_inference_binding(key, url, api_key)
        return url, api_key

    @staticmethod
    def parse_inference_observation(raw: str, *, batched: bool = False) -> RLData:
        """Turn ``--obs`` / deployment-style string into arrays for :meth:`Agent.get_action`.

        Delegates to :meth:`~agilerl.arena.inference.Agent.observation_from_string`.
        """
        return Agent.observation_from_string(raw, batched=batched)

    def open_inference_agent(
        self,
        deployment_name: str,
        *,
        refresh: bool = False,
        experiment_name: str | None = None,
        project_name: str | None = None,
        timeout: int | None = None,
    ) -> Agent:
        """Build an :class:`~agilerl.arena.inference.Agent` for a named deployment.

        Deployment observations use ``np.save`` bytes as base64 (nested JSON
        structure mirrors :meth:`~agilerl.arena.inference.Agent.serialize`).
        Use :meth:`parse_inference_observation` on the same string format as
        request body ``obs``, then :meth:`~agilerl.arena.inference.Agent.get_action`.
        CLI: ``arena inference run … --obs '<json-or-base64>'``.
        """
        url, api_key = self.ensure_inference_binding(
            deployment_name,
            refresh=refresh,
            experiment_name=experiment_name,
            project_name=project_name,
        )
        return Agent(
            url,
            api_key=api_key,
            timeout=timeout or self._request_timeout,
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
                raise ArenaFileNotFoundError(msg)
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
                raise ArenaFileNotFoundError(msg)
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
            self._proactively_refresh_oauth()

    def _proactively_refresh_oauth(self) -> None:
        """If stored access token is expired (JWT ``exp``) but refresh exists, refresh once."""
        if self._api_key is not None:
            return
        if not self._tokens.refresh_token or not self._tokens.access_token:
            return
        exp = oauth_access_token_expires_at(self._tokens.access_token)
        if exp is None:
            return
        if exp > time.time() + 60:
            return
        try:
            tokens = self._auth.refresh_access_token(self._tokens.refresh_token)
        except ArenaAuthError:
            logger.debug(
                "Proactive token refresh failed; will retry on 401 if possible.",
            )
            return
        self._tokens.access_token = tokens["access_token"]
        self._tokens.refresh_token = tokens.get(
            "refresh_token", self._tokens.refresh_token
        )

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

            # If the API key failed but we have stored OAuth credentials, retry with those.
            if self._api_key and not _retried and self._tokens.access_token:
                logger.debug(
                    "API key rejected; falling back to stored OAuth credentials."
                )
                self._api_key = None
                return self._send(
                    method,
                    path,
                    stream=stream,
                    timeout=timeout,
                    _retried=True,
                    headers=request_headers,
                    **kwargs,
                )

            if self._api_key:
                msg = (
                    "Invalid API key. Please check that your ARENA_API_KEY is correct."
                )
                raise ArenaAuthError(
                    msg,
                    sdk_hint="Verify the api_key passed to ArenaClient() or the ARENA_API_KEY environment variable.",
                    cli_hint="Verify your --api-key flag or ARENA_API_KEY environment variable.",
                )
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
