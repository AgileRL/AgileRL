from __future__ import annotations

import io
import logging
import os
import tarfile
from pathlib import Path
from typing import Any, ClassVar, Self

import httpx

from agilerl.arena import auth
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaValidationError,
)
from agilerl.arena.logs import EventStream, LogDisplay
from agilerl.arena.models import TrainingJobConfig

logger = logging.getLogger(__name__)


class ArenaClient:
    """Client for the Arena RLOps platform.

    Handles authentication, environment management, and training job
    submission.  Designed as a singleton so that a single authenticated
    session can be shared across a notebook or script.

    :param base_url: Root URL of the Arena API.
    :type base_url: str
    :param request_timeout: Default timeout in seconds for API requests.
    :type request_timeout: int
    :param upload_timeout: Timeout in seconds for file upload requests.
    :type upload_timeout: int
    """

    _instance: ClassVar[ArenaClient | None] = None

    _base_url: ClassVar[str] = "https://arena.agilerl.com"
    MAX_UPLOAD_BYTES: ClassVar[int] = 50 * 1024 * 1024

    def __init__(
        self,
    ) -> None:
        if (
            not self._base_url.startswith("https://")
            and "localhost" not in self._base_url
        ):
            msg = (
                "ArenaClient requires HTTPS. Use http:// only for "
                "local development against localhost."
            )
            raise ValueError(msg)

        self.base_url = self._base_url.rstrip("/")
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=self._request_timeout,
            follow_redirects=True,
        )

        self._try_restore_session()

    def login(self, *, port: int = 8400) -> None:
        """Start the browser-based OAuth login flow.

        Opens the default browser to the Arena login page.  On success
        the tokens are persisted to ``~/.arena/credentials.json``.
        """
        token = auth.login(self._base_url, port=port)
        self._access_token = token
        creds = auth.load_credentials() or {}
        self._refresh_token = creds.get("refresh_token")
        logger.info("Authenticated with Arena.")

    def logout(self) -> None:
        """Clear the current session and remove stored credentials."""
        self._access_token = None
        self._refresh_token = None
        auth.logout()
        logger.info("Logged out of Arena.")

    @property
    def is_authenticated(self) -> bool:
        return self._access_token is not None

    def list_environments(self) -> list[dict[str, Any]]:
        """List all environments available in Arena.

        :return: List of environment dicts, each containing at minimum
            ``name`` and ``version`` keys.
        :rtype: list[dict]
        """
        return self._request("GET", "/api/v1/environments")

    def validate_environment(
        self,
        source: str | os.PathLike[str],
        *,
        version: str | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Validate an environment for use with Arena.

        :param source: Either a **local path** (file or directory containing
            the environment implementation) to upload for server-side
            validation, or the **name** of an already-registered Arena
            environment.
        :type source: str or path-like
        :param version: Required when *source* is an environment name.
            Ignored for local-path uploads (the server determines the version).
        :type version: str or None
        :param stream: If ``True``, stream validation logs to the
            terminal in real time and block until the operation finishes.
        :type stream: bool

        :return: Validation report from the Arena API.
        :rtype: dict

        :raises ArenaValidationError: If the environment fails validation.
        :raises FileNotFoundError: If *source* looks like a path but does not exist.
        """
        path = Path(os.fspath(source)) if not isinstance(source, str) else None

        if path is None:
            candidate = Path(source)
            if candidate.exists():
                path = candidate

        if path is not None:
            resp = self._validate_local(path)
        else:
            resp = self._validate_registered(name=source, version=version)

        if stream and "operation_id" in resp:
            return self.stream_logs(resp["operation_id"])

        return resp

    def _validate_local(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        payload = self._prepare_upload(path)

        if len(payload) > self.MAX_UPLOAD_BYTES:
            msg = (
                f"Upload size ({len(payload) / 1024 / 1024:.1f} MiB) "
                f"exceeds the {self.MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MiB limit."
            )
            raise ValueError(msg)

        resp = self._request(
            "POST",
            "/api/v1/environments/validate",
            files={"archive": ("environment.tar.gz", payload, "application/gzip")},
            timeout=self._upload_timeout,
        )
        self._check_validation_result(resp)
        return resp

    def _validate_registered(
        self,
        name: str,
        version: str | None,
    ) -> dict[str, Any]:
        if version is None:
            msg = (
                "A 'version' is required when validating a registered "
                "environment by name."
            )
            raise ValueError(msg)

        resp = self._request(
            "POST",
            "/api/v1/environments/validate",
            json={"name": name, "version": version},
        )
        self._check_validation_result(resp)
        return resp

    @staticmethod
    def _prepare_upload(path: Path) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        arcname = child.relative_to(path).as_posix()
                        tar.add(str(child), arcname=arcname)
            else:
                tar.add(str(path), arcname=path.name)
        return buf.getvalue()

    @staticmethod
    def _check_validation_result(resp: dict[str, Any]) -> None:
        errors = resp.get("errors")
        if errors:
            raise ArenaValidationError(errors)

    def submit_training_job(
        self,
        config: TrainingJobConfig,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Submit an evolutionary training job to Arena.

        :param config: Fully validated job specification.
        :type config: TrainingJobConfig
        :param stream: If ``True``, stream training logs to the
            terminal in real time and block until the job finishes.
        :type stream: bool

        :return: Server response including ``job_id`` and initial ``status``.
            When *stream* is ``True``, returns the final result from
            the ``complete`` event instead.
        :rtype: dict
        """
        resp = self._request(
            "POST",
            "/api/v1/jobs/training",
            json=config.model_dump(mode="json"),
        )

        if stream and "operation_id" in resp:
            return self.stream_logs(resp["operation_id"])

        return resp

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Retrieve the current status of a training job.

        :param job_id: Identifier returned by :meth:`submit_training_job`.
        :type job_id: str
        """
        return self._request("GET", f"/api/v1/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Request cancellation of a running job."""
        return self._request("POST", f"/api/v1/jobs/{job_id}/cancel")

    def iter_events(
        self,
        operation_id: str,
        *,
        max_retries: int = 5,
    ) -> EventStream:
        """Return an iterable stream of :class:`~agilerl.arena.logs.LogEvent`
        objects for *operation_id*.

        :param operation_id: Identifier returned by an async Arena
            operation (training job, validation, etc.).
        :type operation_id: str
        :param max_retries: Maximum reconnection attempts on disconnect.
        :type max_retries: int

        :return: An iterable, context-manager-capable event stream.
        :rtype: EventStream
        """
        return EventStream(
            http=self._http,
            path=f"/api/v1/operations/{operation_id}/events",
            auth_headers=self._auth_headers(),
            max_retries=max_retries,
        )

    def stream_logs(self, operation_id: str) -> dict[str, Any]:
        """Pretty-print live logs for *operation_id* and block until
        the operation completes.

        :param operation_id: Identifier returned by an async Arena
            operation.
        :type operation_id: str

        :return: The final result payload from the ``complete`` event.
        :rtype: dict
        """
        display = LogDisplay()
        try:
            with self.iter_events(operation_id) as stream:
                for event in stream:
                    display.render(event)
        finally:
            display.stop()
        return display.result

    def _try_restore_session(self) -> None:
        creds = auth.load_credentials()
        if creds:
            self._access_token = creds.get("access_token")
            self._refresh_token = creds.get("refresh_token")

    def _auth_headers(self) -> dict[str, str]:
        if not self._access_token:
            msg = "Not authenticated. Call client.login() first."
            raise ArenaAuthError(msg)
        return {"Authorization": f"Bearer {self._access_token}"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: int | None = None,
        _retried: bool = False,
        **kwargs: Any,
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

        if resp.status_code == 401 and not _retried and self._refresh_token:
            logger.debug("Access token expired, attempting refresh.")
            self._access_token = auth.refresh_access_token(
                self._base_url, self._refresh_token
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

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        authed = "authenticated" if self.is_authenticated else "unauthenticated"
        return f"<ArenaClient url={self._base_url!r} {authed}>"
