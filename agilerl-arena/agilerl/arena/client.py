# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import httpx
from typing_extensions import Self

from agilerl.arena.auth import (
    ArenaOAuth2,
    is_oauth_access_token_valid,
    load_credentials,
    oauth_access_token_expires_at,
)
from agilerl.arena.client_datasets import (
    DATASET_CATEGORIES as DATASET_CATEGORIES,
)
from agilerl.arena.client_datasets import (
    DatasetClientMixin,
)
from agilerl.arena.client_environments import (
    EnvironmentClientMixin,
)
from agilerl.arena.client_environments import (
    EnvironmentIdentity as EnvironmentIdentity,
)
from agilerl.arena.client_environments import (
    EnvironmentKind as EnvironmentKind,
)
from agilerl.arena.client_environments import (
    EnvironmentSource as EnvironmentSource,
)
from agilerl.arena.client_experiments import ExperimentClientMixin
from agilerl.arena.client_inference import (
    MEMORY_SCOPES as MEMORY_SCOPES,
)
from agilerl.arena.client_inference import (
    InferenceClientMixin,
)
from agilerl.arena.client_inference import (
    MemoryScope as MemoryScope,
)
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaConfigError,
    ArenaTrainingError,
    ArenaValidationError,
)
from agilerl.arena.inference import Agent as Agent
from agilerl.arena.inference.cache import (
    load_binding as load_binding,
)
from agilerl.arena.inference.cache import (
    save_binding as save_binding,
)
from agilerl.arena.models import TrainingManifest as TrainingManifest
from agilerl.arena.output import StreamRichRenderer
from agilerl.arena.stream import NDJsonStream, StreamEvent
from agilerl.arena.typing import JSONValue

logger = logging.getLogger(__name__)


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


class ArenaClient(
    EnvironmentClientMixin,
    DatasetClientMixin,
    ExperimentClientMixin,
    InferenceClientMixin,
):
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

    def _dispatch_http(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        stream: bool,
        timeout: int | None,
        request_timeout: object,
        **kwargs: object,
    ) -> httpx.Response:
        """Send one HTTP call, mapping transport errors to ``ArenaAPIError``."""
        try:
            if stream:
                request = self._http.build_request(
                    method,
                    path,
                    headers=headers,
                    timeout=self._stream_timeout(timeout),
                    **kwargs,
                )
                return self._http.send(request, stream=True)
            return self._http.request(
                method, path, headers=headers, timeout=request_timeout, **kwargs
            )
        except httpx.HTTPError as exc:
            raise ArenaAPIError(
                status_code=0,
                detail=f"Network error communicating with Arena: {exc}",
            ) from exc

    def _resend_after_auth_change(
        self,
        method: str,
        path: str,
        *,
        stream: bool,
        timeout: int | None,
        request_headers: dict[str, str],
        kwargs: dict[str, object],
    ) -> httpx.Response:
        """Rewind uploads and retry once after refreshing or dropping credentials."""
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

    def _unauthorized_error(
        self, resp: httpx.Response, *, stream: bool
    ) -> ArenaAuthError:
        """Build the 401 error after retries are exhausted."""
        raw = self._read_response_body(resp, stream=stream)
        if self._api_key:
            msg = "Invalid API key. Please check that your ARENA_API_KEY is correct."
            return ArenaAuthError(
                msg,
                sdk_hint="Verify the api_key passed to ArenaClient() or the ARENA_API_KEY environment variable.",
                cli_hint="Verify your --api-key flag or ARENA_API_KEY environment variable.",
            )
        msg = f"Session expired and could not be refreshed. Server response: {raw[:200]}"
        return ArenaAuthError(
            msg,
            sdk_hint="Please run client.login() again.",
            cli_hint="Please run 'arena login' to re-authenticate.",
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
        request_headers = dict(kwargs.pop("headers", {}))
        headers = dict(request_headers)
        headers.update(self._auth_headers())
        # Explicit timeout=None disables httpx timeouts; use the client default.
        request_timeout = timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT
        resp = self._dispatch_http(
            method,
            path,
            headers=headers,
            stream=stream,
            timeout=timeout,
            request_timeout=request_timeout,
            **kwargs,
        )
        if (
            resp.status_code == 401
            and not _retried
            and self._api_key is None
            and self._tokens.refresh_token
        ):
            if stream:
                resp.close()
            logger.debug("Access token expired, attempting refresh.")
            tokens = self._auth.refresh_access_token(self._tokens.refresh_token)
            self._tokens.access_token = tokens["access_token"]
            self._tokens.refresh_token = tokens.get(
                "refresh_token", self._tokens.refresh_token
            )
            return self._resend_after_auth_change(
                method,
                path,
                stream=stream,
                timeout=timeout,
                request_headers=request_headers,
                kwargs=kwargs,
            )
        if resp.status_code == 401:
            if self._api_key and not _retried and self._tokens.access_token:
                logger.debug(
                    "API key rejected; falling back to stored OAuth credentials."
                )
                self._api_key = None
                return self._resend_after_auth_change(
                    method,
                    path,
                    stream=stream,
                    timeout=timeout,
                    request_headers=request_headers,
                    kwargs=kwargs,
                )
            raise self._unauthorized_error(resp, stream=stream)
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
