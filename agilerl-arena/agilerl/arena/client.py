# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, Literal, TypedDict

import httpx
from typing_extensions import Self

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
    ArenaTrainingError,
    ArenaValidationError,
)
from agilerl.arena.inference import Agent
from agilerl.arena.inference.cache import (
    load_binding,
    normalized_deployment_name,
    save_binding,
)
from agilerl.arena.models import TrainingManifest
from agilerl.arena.output import StreamRichRenderer
from agilerl.arena.stream import NDJsonStream, StreamEvent
from agilerl.arena.typing import JSONValue
from agilerl.arena.utils import (
    discover_env_sidecars,
    extract_filename,
    multipart_text_fields,
    order_dataset_fields,
    prepare_env_upload,
    prepare_file_upload,
)

logger = logging.getLogger(__name__)

DATASET_CATEGORIES = frozenset({"sft", "preference", "reasoning"})

MemoryScope = Literal["user", "organization"]
MEMORY_SCOPES: tuple[MemoryScope, ...] = ("user", "organization")


# Functional syntax because ``in`` (a parameter's location) is a Python keyword
# and can't be a class-statement field name.
ManifestParamSpec = TypedDict(
    "ManifestParamSpec",
    {
        "name": str,
        "in": str,  # "query" | "body" | "client"
        "type": str,  # "string" | "int" | "bool" | "json"
        "required": bool,
        "help": str,
        "click": dict[str, Any],
    },
    total=False,
)


class ManifestInvoke(TypedDict, total=False):
    """The fixed call descriptor for an on-prem command (method, path, params).

    Used both for the hardcoded invokes in ``agilerl.arena.on_prem.endpoints``
    and for command nodes parsed from the server capabilities manifest.
    """

    method: str
    path: str
    responseKind: str  # "json" | "binary"
    params: list[ManifestParamSpec]


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


def _check_writable_target(path: Path) -> None:
    """Ensure *path* can be created as a new file, making parent dirs as needed.

    :param path: The intended destination file.
    :type path: Path
    :raises FileExistsError: If something already exists at *path*.
    :raises NotADirectoryError: If the parent path exists but is not a directory.
    """
    if path.exists():
        kind = "directory" if path.is_dir() else "file"
        msg = (
            f"Cannot write to {path}: a {kind} of that name already exists. "
            f"Remove it, or choose a different output path."
        )
        raise FileExistsError(msg)

    parent = path.parent
    if parent.exists() and not parent.is_dir():
        msg = f"Cannot write to {path}: {parent} is not a directory."
        raise NotADirectoryError(msg)
    parent.mkdir(parents=True, exist_ok=True)


class ArenaClient:
    """Client for the Arena RLOps platform.

    Handles authentication, environment management, and training job submission.

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
    :type api_key: str | None
    :param request_timeout: Default timeout in seconds for API requests.
    :type request_timeout: int
    :param upload_timeout: Timeout in seconds for file-upload requests.
    :type upload_timeout: int
    :param verbose: Whether to enable verbose logging.
    :type verbose: bool
    :returns: None
    :rtype: None
    """

    BASE_URL: ClassVar[str] = "https://arena.agilerl.com"
    CONFIG_DIR: ClassVar[Path] = Path.home() / ".arena"
    CONFIG_FILE: ClassVar[Path] = CONFIG_DIR / "config.json"

    _CAPABILITIES_PATH: ClassVar[str] = "/api/cli/v1/capabilities"
    _CAPABILITIES_TIMEOUT_SECS: ClassVar[float] = 5.0
    _MANIFEST_ALLOWED_PATH_PREFIX: ClassVar[str] = "/api/cli/v1/on-prem"
    _MANIFEST_ALLOWED_METHODS: ClassVar[frozenset[str]] = frozenset(
        {"GET", "POST", "PATCH", "DELETE"}
    )

    _ERROR_MAP: ClassVar[dict[str, type[ArenaAPIError]]] = {
        "/api/cli/v1/environments/create-and-validate": ArenaValidationError,
        "/api/cli/v1/environments/validate": ArenaValidationError,
        "/api/cli/v1/environments/profile": ArenaValidationError,
        "/api/cli/v1/datasets/create": ArenaValidationError,
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

        self._base_url = (os.environ.get("ARENA_BASE_URL") or self.BASE_URL).rstrip("/")
        self._request_timeout = request_timeout
        self._upload_timeout = upload_timeout

        self._api_key = api_key or os.environ.get("ARENA_API_KEY")
        self._auth = ArenaOAuth2()
        self._tokens = _TokenStore()
        self._verbose = verbose
        self._stream_handler: Callable[[StreamEvent], None] | None = None
        self._cli_capabilities_cache: dict[str, Any] | None = None

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
        # Validate that the project exists.
        existing = self.list_projects()
        names = [p["name"] for p in existing]
        if name not in names:
            hint = f"Available projects: {', '.join(names) or 'None'}. "
            msg = f"Project {name!r} not found."
            raise ArenaConfigError(msg, sdk_hint=hint, cli_hint=hint)

        # Update ~/.arena/config.json with the new default project.
        config = self._read_config()
        config["default_project"] = name
        self._write_config(config)
        logger.info("Default project set to %r.", name)

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
        """Get the authenticated user's profile details.

        Includes account fields such as email and name. When the Arena server
        exposes them, the payload may also contain entitlement flags (e.g.
        enterprise / on-prem access) relevant to CLI feature gating.
        """
        return self._request("GET", "/api/users/current")

    def get_user_credits(self) -> JSONValue:
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
        :type version: str | None
        :returns: True if the environment exists, False otherwise.
        :rtype: bool
        """
        resp: dict[str, bool | str] = self._request(
            "GET",
            "/api/cli/v1/environments/exists",
            params={"name": name, "version": version},
        )
        return bool(resp.get("exists", False))

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
        logger.info("Found %d entrypoints for environment %s.", len(resp), name)
        return resp

    def validate_environment(
        self,
        *,
        name: str,
        version: str | None = None,
        source: str | os.PathLike[str] | bytes | None = None,
        env_config: str | os.PathLike[str] | None = None,
        requirements: str | os.PathLike[str] | None = None,
        entrypoint: str | None = None,
        description: str | None = None,
        multi_agent: bool = False,
        language_based: bool = False,
        do_rollouts: bool = False,
    ) -> dict[str, Any]:
        """Validate a custom environment on Arena.

        When source is provided the environment is uploaded, created, and
        validated in a single step.  When source is None an
        already-registered environment is validated by name/version.

        :param name: Environment name.
        :type name: str
        :param version: Environment version. If creating an environment from scratch, defaults to "v1",
            if validating an already-registered environment, defaults to None, which resolves to the latest version.
        :type version: str | None
        :param source: Environment source — a directory path (compressed
            automatically), a ``.tar.gz`` file path, or raw ``bytes``.
        :type source: str | os.PathLike[str] | bytes | None
        :param env_config: Path to the environment configuration file containing the environment parameters.
            When *source* is a directory and this is None, a top-level ``env_config.yaml``, ``env_config.yml``,
            or ``env_config.json`` (in that order) is picked up automatically. Default is None.
        :type env_config: str | os.PathLike[str] | None
        :param requirements: Path to additional dependencies needed for the environment.
            When *source* is a directory and this is None, a top-level ``requirements.txt`` is picked up
            automatically. Default is None.
        :type requirements: str | os.PathLike[str] | None
        :param entrypoint: Optional entrypoint override. Default is None.
        :type entrypoint: str | None
        :param description: Optional description of the environment. Default is None.
        :type description: str | None
        :param multi_agent: Whether the environment is multi-agent. Default is False.
        :type multi_agent: bool
        :param language_based: Whether the environment follows the GEM API (language-based).
            Default is False.
        :type language_based: bool
        :param do_rollouts: Whether to perform environment rollouts during validation. Setting this to True will
            run 100 random episodes and collect additional information such as the average random reward and visualize
            the rendered environment. Default is False.
        :type do_rollouts: bool

        :returns: A dictionary containing the validation result.
        :rtype: dict[str, Any]
        """
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
                language_based=language_based,
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
    ) -> dict[str, Any] | None:
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

        if not confirm:
            confirm_prompt = input("Do you wish to continue? [y/N]: ").strip().lower()
            if confirm_prompt not in ("y", "yes"):
                logger.info("No environment was deleted for %s.", name)
                return None

        payload = {"name": name, "version": version}
        result: dict[str, Any] = self._request(
            "DELETE", "/api/cli/v1/environments/delete", json=payload
        )
        deleted_version = result.get("version", version)
        msg_suffix = f":{deleted_version}" if version else ""
        logger.info("Environment %s%s deleted successfully.", name, msg_suffix)
        return result

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
    ### Datasets ###
    # -------------------------------------------------------------------------

    def list_datasets(
        self,
        *,
        name: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        """List datasets or search HuggingFace datasets.

        :param name: Filter by registered dataset name.
        :type name: str | None
        :param search: HuggingFace dataset search query.
        :type search: str | None
        :returns: List of datasets or search results from Arena.
        :rtype: list[dict[str, Any]]
        """
        params: dict[str, str] | None = None
        if name is not None or search is not None:
            params = {}
            if name is not None:
                params["name"] = name
            if search is not None:
                params["search"] = search
        result = self._request("GET", "/api/cli/v1/datasets", params=params)
        if not isinstance(result, list):
            return result
        return [
            order_dataset_fields(item) if isinstance(item, dict) else item
            for item in result
        ]

    def dataset_exists(self, name: str) -> dict[str, bool | str]:
        """Check whether a dataset name is registered for the active org.

        :param name: Dataset name.
        :type name: str
        :returns: ``exists``, optional ``id``, and ``datasetType`` when present.
        :rtype: dict[str, bool | str]
        """
        return self._request(
            "GET",
            "/api/cli/v1/datasets/exists",
            params={"name": name},
        )

    def create_dataset(
        self,
        *,
        name: str,
        category: str,
        column_mapping: str | dict[str, Any],
        description: str | None = None,
        file: str | os.PathLike[str] | bytes | None = None,
        hf_dataset_name: str | None = None,
        hf_config: str | None = None,
        hf_split: str | None = None,
    ) -> dict[str, Any]:
        """Create an LLM dataset on Arena.

        Upload a local CSV, import from HuggingFace, or create metadata only
        (no file or HF fields). Validation is performed by the Arena API.

        :param name: Dataset name.
        :type name: str
        :param category: Dataset category (e.g. ``reasoning``, ``preference``).
        :type category: str
        :param column_mapping: Column mapping as a JSON string or dict.
        :type column_mapping: str | dict[str, Any]
        :param description: Optional description.
        :type description: str | None
        :param file: Local CSV file path or raw bytes.
        :type file: str | os.PathLike[str] | bytes | None
        :param hf_dataset_name: HuggingFace dataset id for import.
        :type hf_dataset_name: str | None
        :param hf_config: HuggingFace dataset config name.
        :type hf_config: str | None
        :param hf_split: HuggingFace split (required when importing from HF).
        :type hf_split: str | None
        :returns: Created dataset metadata from Arena.
        :rtype: dict[str, Any]
        """
        data, upload_files = self._build_create_dataset_multipart(
            name=name,
            category=category,
            column_mapping=column_mapping,
            description=description,
            file=file,
            hf_dataset_name=hf_dataset_name,
            hf_config=hf_config,
            hf_split=hf_split,
        )
        files: dict[str, tuple[None, str] | tuple[str, Any, str]] = {
            **multipart_text_fields(data),
            **upload_files,
        }
        try:
            resp: dict[str, Any] = self._request(
                "POST",
                "/api/cli/v1/datasets/create",
                files=files,
                timeout=self._upload_timeout,
            )
        finally:
            self._close_upload_files(upload_files)
        if resp and resp.get("is_ready", False) and resp.get("uploaded", False):
            logger.info("Dataset %s created successfully.", name)

        return resp

    def delete_dataset(
        self,
        name: str,
        *,
        confirm: bool = False,
    ) -> dict[str, Any] | None:
        """Archive a dataset by name.

        :param name: Dataset name.
        :type name: str
        :param confirm: When ``True``, skip the interactive confirmation prompt.
        :type confirm: bool
        :returns: Archive result, or ``None`` if the user declined confirmation.
        :rtype: dict[str, Any] | None
        """
        if not confirm:
            confirm_prompt = (
                input(
                    f"Delete dataset {name!r}? [y/N]: ",
                )
                .strip()
                .lower()
            )
            if confirm_prompt not in ("y", "yes"):
                logger.info("Dataset %s was not deleted.", name)
                return None

        resp = self._request(
            "DELETE",
            "/api/cli/v1/datasets/delete",
            json={"name": name},
        )
        logger.info("Dataset %s deleted successfully.", name)
        return resp

    @staticmethod
    def _validate_dataset_category(category: str) -> str:
        normalized = category.strip().lower()
        if normalized not in DATASET_CATEGORIES:
            supported = ", ".join(sorted(DATASET_CATEGORIES))
            msg = (
                f"Invalid dataset category {category!r}. "
                f"Supported categories: {supported}"
            )
            raise ArenaValidationError(msg)
        return normalized

    @staticmethod
    def _build_create_dataset_multipart(
        *,
        name: str,
        category: str,
        column_mapping: str | dict[str, Any],
        description: str | None = None,
        file: str | os.PathLike[str] | bytes | None = None,
        hf_dataset_name: str | None = None,
        hf_config: str | None = None,
        hf_split: str | None = None,
    ) -> tuple[dict[str, str | None], dict[str, tuple[str, Any, str]]]:
        """Build multipart form fields for dataset creation."""
        category = ArenaClient._validate_dataset_category(category)
        column_mapping_str = (
            json.dumps(column_mapping)
            if isinstance(column_mapping, dict)
            else column_mapping
        )
        data = {
            "name": name,
            "category": category,
            "column_mapping": column_mapping_str,
            "description": description,
            "hf_dataset_name": hf_dataset_name,
            "hf_config": hf_config,
            "hf_split": hf_split,
        }

        files: dict[str, tuple[str, Any, str]] = {}
        if file is not None:
            files["file"] = prepare_file_upload(
                file,
                default_name="dataset.csv",
                content_type="text/csv",
            )
        return data, files

    # -------------------------------------------------------------------------
    ### Training Jobs ###
    # -------------------------------------------------------------------------

    def submit_experiment(
        self,
        manifest: str | Path | dict[str, Any],
        *,
        resource_id: str | int | None = None,
        num_nodes: int | None = None,
        project: str | None = None,
        experiment_name: str | None = None,
        reward_file: str | os.PathLike[str] | bytes | None = None,
        completion: str | None = None,
    ) -> dict[str, Any]:
        """Submit an experiment (a training job).

        :param manifest: Training manifest as a YAML/JSON file path, raw YAML
            string, or a pre-parsed dict.
        :type manifest: str | Path | dict[str, Any]
        :param resource_id: Arena cluster type or resource id for the job.
        :type resource_id: str | int | None
        :param num_nodes: The number of nodes to use for training.
        :type num_nodes: int | None
        :param project: The project to submit the experiment to.
        :type project: str | None
        :param experiment_name: The name of the experiment to submit.
        :type experiment_name: str | None
        :param reward_file: Python reward module for reasoning dataset jobs.
            When set, the request is sent as ``multipart/form-data``.
        :type reward_file: str | os.PathLike[str] | bytes | None
        :param completion: Optional model completion for reward validation.
            When omitted, the server uses the reference answer from the first
            dataset row.
        :type completion: str | None
        """
        validated = TrainingManifest.get_validated(manifest, mode="json")
        resolved_project = self._resolve_project(project)

        if reward_file is not None:
            files = self._build_submit_experiment_multipart(
                manifest=validated,
                project=resolved_project,
                resource_id=resource_id,
                num_nodes=num_nodes,
                experiment_name=experiment_name,
                reward_file=reward_file,
                completion=completion,
            )
            try:
                return self._open_stream(
                    "POST",
                    "/api/cli/v1/experiments/jobs/submit",
                    files=files,
                    timeout=self._upload_timeout,
                ).collect()
            finally:
                self._close_upload_files(files)

        payload: dict[str, Any] = {
            "manifest": validated,
            "resource_id": resource_id,
            "num_nodes": num_nodes,
            "project": resolved_project,
            "experiment_name": experiment_name,
        }
        return self._open_stream(
            "POST",
            "/api/cli/v1/experiments/jobs/submit",
            json=payload,
            timeout=self._upload_timeout,
        ).collect()

    @staticmethod
    def _build_submit_experiment_multipart(
        *,
        manifest: dict[str, Any],
        project: str | None,
        resource_id: str | int | None,
        num_nodes: int | None,
        experiment_name: str | None,
        reward_file: str | os.PathLike[str] | bytes,
        completion: str | None,
    ) -> dict[str, tuple[None, str] | tuple[str, BinaryIO | bytes, str]]:
        """Build multipart form parts for reasoning submit with reward validation."""
        text_fields: dict[str, str | None] = {
            "manifest": json.dumps(manifest),
            "project": project,
            "resource_id": str(resource_id) if resource_id is not None else None,
            "num_nodes": str(num_nodes) if num_nodes is not None else None,
            "experiment_name": experiment_name,
            "completion": completion,
        }
        files: dict[str, tuple[None, str] | tuple[str, BinaryIO | bytes, str]] = {
            **multipart_text_fields(text_fields)
        }
        files["reward_file"] = prepare_file_upload(
            reward_file,
            default_name="reward.py",
            content_type="text/x-python",
        )
        return files

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
            params.extend(("metric", m) for m in metrics)
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
        :raises NotADirectoryError: If the parent path exists but is not a directory.
        """
        path = (
            Path(f"{experiment_name}_metrics.csv")
            if output_path is None
            else Path(output_path)
        )
        # A directory target takes its filename from the response's
        # content-disposition, so it can only be checked after the download.
        resolve_after_download = path.is_dir()
        if not resolve_after_download:
            _check_writable_target(path)

        payload, content_type, disposition = self.preview_experiment_metrics_csv(
            experiment_name,
            preview_rows=50_000,
            metrics=metrics,
        )
        if not (content_type or "").startswith("text/csv"):
            body_preview = payload.decode("utf-8", errors="replace")[:500]
            raise ArenaAPIError.from_response_body(body_preview)

        if resolve_after_download:
            filename = extract_filename(disposition) or f"{experiment_name}_metrics.csv"
            path = path / filename
            _check_writable_target(path)

        path.write_bytes(payload)
        logger.info("Metrics saved to %s", path)
        return path

    def stop_experiment(self, experiment_name: str) -> JSONValue:
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
    def _deployment_lookup_params(
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

    @staticmethod
    def _validated_memory_scope(memory_scope: str) -> str:
        """Reject an unknown memory scope here rather than at the API."""
        scope = memory_scope.strip().lower()
        if scope not in MEMORY_SCOPES:
            msg = (
                f"Unknown memory scope {memory_scope!r}. "
                f"Choose one of: {', '.join(MEMORY_SCOPES)}."
            )
            raise ArenaValidationError(msg)
        return scope

    def deploy_agent(
        self,
        experiment_name: str,
        checkpoint: str | None = None,
        memory_scope: MemoryScope | None = None,
    ) -> dict[str, Any]:
        """Create an inference deployment from an experiment checkpoint.

        :param experiment_name: The name of the experiment to deploy.
        :type experiment_name: str
        :param checkpoint: The checkpoint to deploy. If None, deploy the best checkpoint.
        :type checkpoint: str | None
        :param memory_scope: Who an LLM deployment keeps chat sessions for. ``"user"``
            gives every caller their own conversations, ``"organization"`` shares them
            across the organisation. A new deployment defaults to ``"user"``. Redeploying
            with ``None`` keeps the scope already stored, it does not reset it, and the
            scope cannot be changed after the deployment exists.
        :type memory_scope: MemoryScope | None
        :returns: A dictionary containing the deployment result.
        :rtype: dict[str, Any]
        """
        body: dict[str, Any] = {
            "experiment_name": experiment_name,
            "checkpoint": checkpoint,
        }
        if memory_scope is not None:
            body["memoryScope"] = self._validated_memory_scope(memory_scope)
        result = self._request(
            "POST",
            "/api/cli/v1/inference/deploy",
            json=body,
        )
        checkpoint_suffix = (
            f" (checkpoint {checkpoint})" if checkpoint else " (best checkpoint)"
        )
        logger.info(
            "Agent submitted for deployment from experiment %s%s. "
            "Check its status in Arena.",
            experiment_name,
            checkpoint_suffix,
        )
        return result

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
        q = self._deployment_lookup_params(
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

    def _inference_credential(self) -> str | None:
        """Return :meth:`_credential`, refreshing an expiring OAuth token first."""
        self._proactively_refresh_oauth()
        return self._credential()

    def open_inference_agent(
        self,
        deployment_name: str,
        *,
        refresh: bool = False,
        experiment_name: str | None = None,
        project_name: str | None = None,
        timeout: int | None = None,
    ) -> Agent:
        """Build an :class:`~arena.inference.Agent` for a named deployment.

        Attempts to load the deployment from the cache, and if not found, fetches it from the API.

        The agent carries this client's own credential, so run :meth:`login` or set
        ``ARENA_API_KEY`` before using a deployment that keeps memory per user.

        :param deployment_name: The name of the deployment to open.
        :type deployment_name: str
        :param refresh: Whether to refresh the deployment metadata.
        :type refresh: bool
        :param experiment_name: Experiment name to disambiguate when multiple deployments
            share the same deployment name.
        :type experiment_name: str | None
        :param project_name: Project name to disambiguate when multiple deployments
            share the same deployment name.
        :type project_name: str | None
        :param timeout: HTTP timeout in seconds for the returned agent's inference requests.
        :type timeout: int | None
        :returns: An :class:`~arena.inference.Agent` instance.
        :rtype: Agent
        """
        url = self._ensure_inference_binding(
            deployment_name,
            refresh=refresh,
            experiment_name=experiment_name,
            project_name=project_name,
        )
        return Agent(
            url,
            api_key=self._inference_credential(),
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

    # -------------------------------------------------------------------------
    ### Utility Methods ###
    # -------------------------------------------------------------------------

    @staticmethod
    def _deployment_url(row: dict[str, Any]) -> str:
        """Parse ``url`` from an API deployment row."""
        url = row.get("url")
        if not isinstance(url, str) or not url.strip():
            msg = "Deployment has no inference URL."
            raise ArenaAPIError(
                msg,
                cli_hint="Wait until provisioning completes, then retry with --refresh.",
            )
        return url.strip()

    def _fetch_deployment_for_inference(
        self,
        deployment_name: str,
        *,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one deployment detail row for inference binding."""
        params = self._deployment_lookup_params(
            name=normalized_deployment_name(deployment_name),
            experiment_name=experiment_name,
            project_name=project_name,
        )
        if "name" not in params:
            msg = "deployment name is required."
            raise ArenaAPIError(msg)

        try:
            row = self._request(
                "GET",
                "/api/cli/v1/inference/deployments/one",
                params=params,
            )
        except ArenaAPIError as exc:
            hint = (
                "Pass --experiment-name and/or --project-name when multiple deployments "
                "share this deployment name."
            )
            if not exc.cli_hint:
                raise ArenaAPIError(
                    exc.detail, cli_hint=hint, status_code=exc.status_code
                ) from exc
            raise

        if not isinstance(row, dict):
            msg = "Unexpected deployment detail response shape."
            raise ArenaAPIError(msg)
        return row

    def _ensure_inference_binding(
        self,
        deployment_name: str,
        *,
        refresh: bool = False,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> str:
        """Return the cached deployment URL, or fetch it from the API and persist it."""
        key = normalized_deployment_name(deployment_name)

        if not refresh:
            cached = load_binding(key)
            if cached is not None:
                return cached

        row = self._fetch_deployment_for_inference(
            deployment_name,
            experiment_name=experiment_name,
            project_name=project_name,
        )
        url = self._deployment_url(row)
        save_binding(key, url)
        return url

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
        language_based: bool,
        do_rollouts: bool,
    ) -> NDJsonStream:
        """Upload, create, and validate an environment."""
        # When source is a directory, pick up requirements.txt / env_config.*
        # sidecars from it unless the caller passed them explicitly.
        requirements, env_config = discover_env_sidecars(
            source, requirements=requirements, env_config=env_config
        )

        # Resolve the environment source into a streamable upload payload
        archive_name, archive_payload = prepare_env_upload(source)
        data: dict[str, str] = {
            "name": name,
            "version": version,
            "multi_agent": str(multi_agent).lower(),
            "language_based": str(language_based).lower(),
            "do_rollouts": str(do_rollouts).lower(),
        }
        if entrypoint:
            data["entrypoint"] = entrypoint
        if description:
            data["description"] = description

        files: dict[str, tuple[str, Any, str]] = {
            "file": (archive_name, archive_payload, "application/gzip"),
        }

        # Check env_config and resolve for upload
        if env_config is not None:
            is_json = Path(os.fspath(env_config)).suffix.lower() == ".json"
            files["env_config"] = prepare_file_upload(
                env_config,
                default_name="env_config.json" if is_json else "env_config.yaml",
                content_type="application/json" if is_json else "application/x-yaml",
            )
        else:
            files["env_config"] = ("env_config.yaml", b"", "application/x-yaml")

        # Check requirements and resolve for upload
        if requirements is not None:
            files["requirements"] = prepare_file_upload(
                requirements,
                default_name="requirements.txt",
                content_type="text/plain",
            )
        else:
            files["requirements"] = ("requirements.txt", b"", "text/plain")

        try:
            # The request body is fully sent by the time the stream is
            # returned, so the upload handles can be closed afterwards.
            return self._open_stream(
                "POST",
                "/api/cli/v1/environments/create-and-validate",
                data=data,
                files=files,
                timeout=self._upload_timeout,
            )
        finally:
            self._close_upload_files(files)

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

    def _credential(self) -> str | None:
        """Return the bearer material to authenticate with: PAT, else OAuth token."""
        return self._api_key or self._tokens.access_token

    def _auth_headers(self) -> dict[str, str]:
        """Return the authentication headers for the request."""
        credential = self._credential()
        if credential:
            return {"Authorization": f"Bearer {credential}"}

        msg = "Client has not been authenticated with Arena."
        raise ArenaAuthError(
            msg,
            sdk_hint="Call client.login() or provide an API key to the ArenaClient constructor.",
            cli_hint="Run 'arena login' to authenticate.",
        )

    def _stream_timeout(self, timeout: int | None) -> httpx.Timeout:
        """Build the httpx timeout for a streaming (NDJSON progress) request."""
        base = timeout if timeout is not None else self._request_timeout
        return httpx.Timeout(base, read=None)

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

        # An explicit ``timeout=None`` would disable timeouts entirely in
        # httpx; fall back to the client default (``request_timeout``).
        request_timeout = timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT

        try:
            if stream:
                request = self._http.build_request(
                    method,
                    path,
                    headers=headers,
                    timeout=self._stream_timeout(timeout),
                    **kwargs,
                )
                resp = self._http.send(request, stream=True)
            else:
                resp = self._http.request(
                    method, path, headers=headers, timeout=request_timeout, **kwargs
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
            self._rewind_upload_files(kwargs.get("files"))
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
                self._rewind_upload_files(kwargs.get("files"))
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
    def _close_upload_files(files: dict[str, tuple] | None) -> None:
        """Close any open file handles in an httpx multipart ``files`` dict."""
        for value in (files or {}).values():
            payload = value[1] if isinstance(value, tuple) and len(value) > 1 else None
            close = getattr(payload, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _rewind_upload_files(files: dict[str, tuple] | None) -> None:
        """Rewind seekable upload payloads so a retried request resends them."""
        for value in (files or {}).values():
            payload = value[1] if isinstance(value, tuple) and len(value) > 1 else None
            seek = getattr(payload, "seek", None)
            if callable(seek):
                seek(0)

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
    ) -> Any:  # noqa: ANN401 -- callers return this straight into narrower declared types (e.g. dict[str, Any]); JSONValue would force casts at every call site
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

    def _get_cli_capabilities(
        self, *, force_refresh: bool = False
    ) -> dict[str, Any] | None:
        """Fetch CLI capabilities document (internal; used by Arena CLI only)."""
        if self._cli_capabilities_cache is not None and not force_refresh:
            return self._cli_capabilities_cache

        try:
            headers = self._auth_headers()
        except ArenaAuthError:
            self._cli_capabilities_cache = None
            return None

        try:
            resp = self._http.request(
                "GET",
                self._CAPABILITIES_PATH,
                headers=headers,
                timeout=min(self._request_timeout, self._CAPABILITIES_TIMEOUT_SECS),
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "Arena CLI capabilities request failed (%s): %s",
                self._CAPABILITIES_PATH,
                exc,
            )
            self._cli_capabilities_cache = None
            return None

        if resp.status_code == 404:
            self._cli_capabilities_cache = None
            return None

        if not resp.is_success:
            raw = resp.text
            raise ArenaAPIError.from_response_body(raw, status_code=resp.status_code)

        try:
            envelope = resp.json()
        except ValueError:
            # Empty body, HTML fallback pages, or other non-JSON success payloads.
            self._cli_capabilities_cache = None
            return None

        if (
            not isinstance(envelope, dict)
            or envelope.get("ok") is not True
            or "data" not in envelope
        ):
            self._cli_capabilities_cache = None
            return None

        data = envelope["data"]
        if not isinstance(data, dict):
            self._cli_capabilities_cache = None
            return None
        schema_v = data.get("schemaVersion")
        if schema_v != 1 and schema_v != "1":
            logger.warning(
                "Arena CLI capabilities unsupported schemaVersion=%r (need 1)",
                schema_v,
            )
            self._cli_capabilities_cache = None
            return None

        self._cli_capabilities_cache = data
        return data

    def _validate_manifest_invoke(
        self, invoke: ManifestInvoke
    ) -> tuple[str, str, str, list[ManifestParamSpec]]:
        """Check an invoke descriptor and return ``(path, method, responseKind, params)``.

        Guards against unsupported methods/paths so a malformed or untrusted
        server manifest can't drive the client to an unexpected endpoint.
        """
        path = invoke["path"]
        if not isinstance(path, str):
            msg = "The Arena server sent an invalid on-prem command."
            raise ArenaValidationError(
                msg,
                cli_hint="Upgrade agilerl — the server sent an on-prem "
                "configuration this version can't use.",
            )
        if not path.startswith(self._MANIFEST_ALLOWED_PATH_PREFIX):
            msg = "This on-prem command isn't permitted by the CLI."
            raise ArenaValidationError(msg)
        if ".." in path.split("/"):
            msg = "The Arena server sent an invalid on-prem command path."
            raise ArenaValidationError(msg)

        method = str(invoke["method"]).upper()
        if method not in self._MANIFEST_ALLOWED_METHODS:
            msg = f"Unsupported manifest HTTP method {method!r}."
            raise ArenaValidationError(msg)

        response_kind = invoke["responseKind"]
        if response_kind not in {"json", "binary"}:
            msg = f"Unsupported manifest responseKind {response_kind!r}."
            raise ArenaValidationError(msg)

        allowed_param_in = frozenset({"query", "body", "client"})
        allowed_param_types = frozenset({"string", "int", "bool", "json"})

        params_list = list(invoke.get("params") or [])
        for spec in params_list:
            pin = spec.get("in")
            if pin not in allowed_param_in:
                msg = f"Unsupported manifest param location {pin!r}."
                raise ArenaValidationError(msg)
            ptyp = spec.get("type")
            if ptyp not in allowed_param_types:
                msg = f"Unsupported manifest param type {ptyp!r}."
                raise ArenaValidationError(msg)

        return path, method, response_kind, params_list

    def _partition_manifest_args(
        self,
        *,
        method: str,
        params_list: list[ManifestParamSpec],
        parsed_args: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Split parsed CLI args into ``(query, body)`` per each param's ``in``.

        Returns the query dict and the JSON body (``None`` when the command
        takes no body). Raises if a required argument is missing.
        """
        query: dict[str, Any] = {}
        body_obj: dict[str, Any] | None = None

        for spec in params_list:
            where = spec["in"]
            key = spec["name"]
            if where == "client":
                continue

            required = bool(spec["required"])
            if key not in parsed_args:
                if required:
                    msg = f"Missing required argument {key!r}."
                    raise ArenaValidationError(msg)
                continue

            val = parsed_args[key]
            if val is None:
                if required:
                    msg = f"Missing required argument {key!r}."
                    raise ArenaValidationError(msg)
                continue

            if where == "query":
                query[key] = val
            elif where == "body":
                if body_obj is None:
                    body_obj = {}
                body_obj[key] = val

        body_needed = any(spec["in"] == "body" for spec in params_list)
        if body_needed and body_obj is None:
            body_obj = {}

        # Hardcoded invokes (e.g. on-prem install) pass a full payload without
        # manifest param specs — route by HTTP method.
        if not params_list and parsed_args:
            if method == "GET":
                query = {**parsed_args, **query}
            elif method in {"POST", "PATCH", "PUT", "DELETE"} and body_obj is None:
                body_obj = dict(parsed_args)

        return query, body_obj

    def _invoke_manifest_command(
        self,
        invoke: ManifestInvoke,
        parsed_args: Mapping[str, Any],
    ) -> Any:  # noqa: ANN401 -- JSON body (heterogeneous) or a binary (bytes, str|None, str|None) tuple; callers destructure the tuple branch, so a union return would force casts
        """Dispatch an on-prem command using already-parsed CLI kwargs.

        Returns decoded JSON for ``responseKind == "json"`` invokes, or a
        ``(bytes, content_type, content_disposition)`` tuple for ``"binary"``
        ones (e.g. bundle downloads); hence the dynamic ``Any`` return.
        """
        path, method, response_kind, params_list = self._validate_manifest_invoke(
            invoke
        )
        query, body_obj = self._partition_manifest_args(
            method=method,
            params_list=params_list,
            parsed_args=parsed_args,
        )

        req_kw: dict[str, Any] = {}
        if query:
            req_kw["params"] = query
        if body_obj is not None:
            req_kw["json"] = body_obj

        if response_kind == "binary":
            return self._request_raw(method, path, **req_kw)

        return self._request(method, path, **req_kw)

    def _open_stream(
        self,
        method: str,
        path: str,
        *,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> NDJsonStream:
        """Send a streaming request and return an :class:`NDJsonStream`."""
        error_cls = self._ERROR_MAP.get(path, ArenaAPIError)
        handler = self._stream_handler
        renderer: StreamRichRenderer | None = None
        if handler is None and self._verbose:
            renderer = StreamRichRenderer(error_cls=error_cls)
            handler = renderer.handle_event

        resp = self._send(method, path, stream=True, timeout=timeout, **kwargs)
        return NDJsonStream(
            resp, handler=handler, renderer=renderer, error_cls=error_cls
        )
