# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable, Generator, Iterator
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from agilerl.auth import EXTERNAL_USER_ID_HEADER, validate_partner_credentials
from agilerl.exceptions import ArenaAuthError, ArenaInferenceError
from agilerl.typing import JSONValue

logger = logging.getLogger(__name__)

# Re-exported for callers and tests.
__all__ = [
    "Agent",
    "AgentInfo",
    "AgentMetadata",
    "DeploymentAssertion",
    "GenerateCompletion",
    "GenerateParams",
    "GenerateResult",
    "LLMCompletionResult",
    "LLMParams",
    "LLMResults",
    "SessionDetail",
    "SessionInfo",
    "SessionMessage",
    "StatusResponse",
]


class LLMParams(BaseModel):
    """LLM sampling parameters model.

    :param max_new_tokens: The maximum number of new tokens to generate.
    :type max_new_tokens: int
    :param temperature: The temperature of the LLM.
    :type temperature: float
    :param top_p: The top-p value of the LLM.
    :type top_p: float
    :param do_sample: Whether to sample the LLM.
    :type do_sample: bool
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


GenerateParams = LLMParams


class AgentInfo(BaseModel):
    """Agent information model.

    :param algo: The algorithm of the agent.
    :type algo: str
    :param llm: Whether the agent is an LLM.
    :type llm: bool
    """

    # /status also reports fields for non-LLM deployments that the client does not use.
    model_config = ConfigDict(extra="ignore", frozen=True)

    algo: str = ""
    llm: bool = False


class StatusResponse(BaseModel):
    """Status response model.

    ``GET /status`` is public, so it reports little beyond the deployment id.
    Fields a deployment leaves out stay at their defaults, and *agent* is
    ``None`` when it does not describe its shape at all.

    :param success: Whether the status request was successful.
    :type success: bool
    :param deployment_id: The deployment ID.
    :type deployment_id: str
    :param instance_id: The instance ID.
    :type instance_id: str
    :param agent: The agent info, or ``None`` if the deployment did not report it.
    :type agent: AgentInfo | None
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool = True
    deployment_id: str = ""
    instance_id: str = ""
    agent: AgentInfo | None = None


AgentMetadata = StatusResponse


class LLMCompletionResult(BaseModel):
    """Single prompt/completion pair from ``POST /generate``.

    :param prompt: The prompt for the LLM completion.
    :type prompt: str
    :param completion: The completion for the LLM completion.
    :type completion: str
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = ""
    completion: str = ""


GenerateCompletion = LLMCompletionResult


class LLMResults(BaseModel):
    """LLM completion response model.

    :param results: The results of the LLM completions.
    :type results: list[LLMCompletionResult]
    :param batch_size: The batch size of the LLM completions.
    :type batch_size: int
    :param inference_time_ms: The inference time in milliseconds.
    :type inference_time_ms: float
    :param tokens_per_second: The tokens per second of the LLM completions.
    :type tokens_per_second: float
    :param success: Whether the LLM completions were successful.
    :type success: bool
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: list[LLMCompletionResult]
    batch_size: int
    inference_time_ms: float
    tokens_per_second: float
    success: bool = True


GenerateResult = LLMResults


class SessionInfo(BaseModel):
    """One chat session summary from ``GET /sessions``.

    :param session_id: The ID of the session.
    :type session_id: str
    :param created_at: When the session was opened.
    :type created_at: str | None
    :param last_updated: When the session was last written to.
    :type last_updated: str | None
    """

    # Listing JSON also has title and created_by; detail JSON also has success.
    model_config = ConfigDict(extra="ignore", frozen=True)

    session_id: str = ""
    created_at: str | None = None
    last_updated: str | None = None


class SessionMessage(BaseModel):
    """One message in a chat session.

    :param role: The role of the message author.
    :type role: str
    :param content: The message text.
    :type content: str
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = ""
    content: str = ""


class SessionDetail(SessionInfo):
    """A :class:`SessionInfo` plus its messages, from ``GET /sessions/{session_id}``.

    :param messages: The messages in the session.
    :type messages: list[SessionMessage]
    """

    messages: list[SessionMessage] = Field(default_factory=list)


class DeploymentAssertion(BaseModel):
    """Short-lived credential Arena mints for one deployment on a customer's cluster.

    The proxy in front of such a deployment drops ``Authorization``, so the
    caller sends this in *header* instead.

    :param assertion: The credential. Never logged or shown in :func:`repr`.
    :type assertion: str
    :param header: Request header the deployment reads it from.
    :type header: str
    :param expires_in: Seconds until it expires.
    :type expires_in: int
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    assertion: str = Field(repr=False)
    header: str
    expires_in: int


class DeploymentAssertionAuth(httpx.Auth):
    """Adds a deployment assertion once a deployment answers 401 without one.

    Cloud deployments read ``Authorization`` and never answer 401 to a valid
    caller, so they never trigger a mint. After the first 401 every request
    carries the assertion, re-minted shortly before it expires, and a request
    still answered 401 is retried once with a fresh one.

    :param mint: Returns a new assertion for this deployment.
    :type mint: Callable[[], DeploymentAssertion]
    """

    def __init__(self, mint: Callable[[], DeploymentAssertion]) -> None:
        self._mint = mint
        self._assertion: DeploymentAssertion | None = None
        self._expires_at = 0.0

    def _refresh(self) -> DeploymentAssertion:
        assertion = self._mint()
        self._assertion = assertion
        self._expires_at = time.monotonic() + assertion.expires_in
        return assertion

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Send *request*, minting and retrying once if the deployment answers 401."""
        assertion = self._assertion
        if assertion is not None:
            # Leave time for the request to reach the deployment before expiry.
            if time.monotonic() > self._expires_at - 10:
                assertion = self._refresh()
            request.headers[assertion.header] = assertion.assertion
        response = yield request
        if response.status_code != 401:
            return
        assertion = self._refresh()
        request.headers[assertion.header] = assertion.assertion
        yield request


class RefreshingBearerAuth(httpx.Auth):
    """Sends ``Authorization: Bearer`` with the credential *get_credential* returns per request.

    :param get_credential: Returns the current credential, or ``None`` to send none.
    :type get_credential: Callable[[], str | None]
    :param next_auth: Auth flow that sends the request once the header is set,
        or ``None`` to send it as is.
    :type next_auth: httpx.Auth | None
    """

    def __init__(
        self,
        get_credential: Callable[[], str | None],
        next_auth: httpx.Auth | None = None,
    ) -> None:
        self.get_credential = get_credential
        self.next_auth = next_auth

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Set the header from a fresh credential, then send the request."""
        credential = self.get_credential()
        if credential:
            request.headers["Authorization"] = f"Bearer {credential}"
        if self.next_auth is None:
            yield request
        else:
            yield from self.next_auth.auth_flow(request)


class Agent:
    """HTTP client for a deployed Arena inference endpoint.

    When *probe_on_init* is ``True`` (default), :meth:`status` runs once at
    construction and stores :attr:`metadata`. Wrong endpoint for the deployment
    type returns HTTP 400 from the server.

    Every request carries your own Arena credential as ``Authorization: Bearer``.
    A deployment that keeps memory per user works out who is calling from it, so
    there is nothing else to pass. Organisation-key clients also send
    ``X-External-User-Id``. The credential is never logged, shown in
    :func:`repr`, or written to disk.

    :param endpoint: Base URL of the Arena inference deployment.
    :type endpoint: str
    :param api_key: Profile PAT (``arena_pat_<uuid>_<secret>``) or access token, or
        a callable that returns one for each request. A login's access token
        expires within minutes, so
        :meth:`~agilerl.client.ArenaClient.open_inference_agent` passes a callable
        that refreshes it. Falls back to the ``ARENA_API_KEY`` environment variable.
    :type api_key: str | Callable[[], str | None] | None
    :param org_key: Organisation API key (``arena_org_…``). Requires
        *external_user_id*. Falls back to ``ARENA_ORG_KEY``.
    :type org_key: str | None
    :param external_user_id: Partner user id sent as ``X-External-User-Id``.
        Requires *org_key*. Falls back to ``ARENA_EXTERNAL_USER_ID``.
    :type external_user_id: str | None
    :param timeout: Request timeout in seconds.
    :type timeout: int
    :param generate_params: Default LLM sampling parameters; defaults to
        :class:`LLMParams` when ``None``.
    :type generate_params: LLMParams | None
    :param probe_on_init: If ``True``, fetch ``GET /status`` during ``__init__``.
    :type probe_on_init: bool
    :param assertion_source: Mints a :class:`DeploymentAssertion` for the
        deployment id in :attr:`metadata`.
    :type assertion_source: Callable[[str], DeploymentAssertion] | None
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | Callable[[], str | None] | None = None,
        org_key: str | None = None,
        external_user_id: str | None = None,
        timeout: int = 30,
        generate_params: LLMParams | None = None,
        probe_on_init: bool = True,
        assertion_source: Callable[[str], DeploymentAssertion] | None = None,
    ) -> None:
        self._base_url = endpoint.rstrip("/")

        org_key = org_key or os.environ.get("ARENA_ORG_KEY")
        external_user_id = external_user_id or os.environ.get("ARENA_EXTERNAL_USER_ID")
        validate_partner_credentials(org_key, external_user_id)

        headers: dict[str, str] = {}
        auth: httpx.Auth | None = (
            None
            if assertion_source is None
            else DeploymentAssertionAuth(
                lambda: assertion_source(self._loaded_metadata().deployment_id)
            )
        )
        if org_key is not None and external_user_id is not None:
            headers["Authorization"] = f"Bearer {org_key}"
            headers[EXTERNAL_USER_ID_HEADER] = external_user_id
        else:
            credential = api_key or os.environ.get("ARENA_API_KEY")
            if callable(credential):
                auth = RefreshingBearerAuth(credential, auth)
            elif credential:
                headers["Authorization"] = f"Bearer {credential}"

        self._http = httpx.Client(
            headers=headers,
            auth=auth,
            timeout=timeout,
            follow_redirects=True,
        )
        self.generate_params = generate_params or LLMParams()
        self.metadata: StatusResponse | None = (
            self._fetch_metadata() if probe_on_init else None
        )

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return f"{self._base_url}{path}"

    @staticmethod
    def _response_detail(resp: httpx.Response) -> str:
        try:
            return str(resp.json())
        except Exception:
            return resp.text[:500] if resp.text else "No details"

    _AUTH_SDK_HINT = (
        "Call client.login() and build the agent with "
        "client.open_inference_agent(), pass a profile PAT as api_key, "
        "or pass org_key and external_user_id."
    )

    def _raise_for_status(
        self,
        resp: httpx.Response,
        *,
        expected: tuple[int, ...] = (200,),
    ) -> None:
        if resp.status_code in (401, 403):
            msg = (
                "Agent endpoint returned 401 Unauthorized. Your Arena credential "
                "is missing or expired."
                if resp.status_code == 401
                else "Your Arena credential is not allowed to use this deployment."
            )
            raise ArenaAuthError(
                msg,
                sdk_hint=self._AUTH_SDK_HINT,
                cli_hint="Run 'arena login'.",
            )
        if resp.status_code not in expected:
            raise ArenaInferenceError(
                status_code=resp.status_code,
                detail=self._response_detail(resp),
            )

    @staticmethod
    def _network_error(exc: httpx.HTTPError) -> ArenaInferenceError:
        return ArenaInferenceError(
            status_code=0,
            detail=f"Network error reaching agent endpoint: {exc}",
        )

    def _send(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Send a request, turning transport failures into an Arena error."""
        try:
            return self._http.request(method, self._url(path), json=json_body)
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc

    def _request_payload(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> JSONValue:
        """Send a request and return the decoded body, object or array."""
        resp = self._send(method, path, json_body=json_body)
        self._raise_for_status(resp)
        body = resp.json()
        if isinstance(body, dict) and body.get("success") is False:
            raise ArenaInferenceError(
                status_code=resp.status_code,
                detail=self._response_detail(resp),
            )
        return body

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = self._request_payload(method, path, json_body=json_body)
        if not isinstance(body, dict):
            msg = f"Expected JSON object from {path}, got {type(body).__name__}."
            raise ArenaInferenceError(status_code=200, detail=msg)
        return body

    def _request_stream(self, path: str, *, json_body: dict[str, Any]) -> Iterator[str]:
        url = self._url(path)
        try:
            with self._http.stream("POST", url, json=json_body) as resp:
                if resp.status_code != 200:
                    # The error detail is read from the body.
                    resp.read()
                self._raise_for_status(resp)
                for chunk in resp.iter_text():
                    if chunk.startswith("\n[ERROR:"):
                        raise ArenaInferenceError(
                            status_code=200,
                            detail=chunk.strip(),
                        )
                    yield chunk
        except ArenaInferenceError:
            raise
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc

    def _fetch_metadata(self) -> StatusResponse:
        body = self._request_json("GET", "/status")
        return StatusResponse.model_validate(body)

    def _loaded_metadata(self) -> StatusResponse:
        if self.metadata is None:
            msg = "Metadata not loaded. Call status() or construct with probe_on_init=True."
            raise ArenaInferenceError(msg)
        return self.metadata

    def _agent_info(self) -> AgentInfo | None:
        """Agent shape from ``GET /status``, or ``None`` if it did not report one."""
        return self._loaded_metadata().agent

    def _ensure_llm(self) -> None:
        agent = self._agent_info()
        if agent is None:
            # Type unknown, so let the deployment answer 400 for a wrong route
            # rather than guessing here and refusing a call that would work.
            return
        if not agent.llm:
            raise ArenaInferenceError(
                detail=(
                    "This deployment is not an LLM agent; agilerl 3 supports "
                    "only LLM deployments."
                ),
            )

    @staticmethod
    def _validate_prompts(prompts: list[str]) -> None:
        if not prompts:
            raise ArenaInferenceError(detail="At least one prompt is required.")
        for index, prompt in enumerate(prompts):
            if not isinstance(prompt, str) or not prompt.strip():
                raise ArenaInferenceError(
                    detail=f"Prompt at index {index} is empty.",
                )

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ArenaInferenceError(detail="session_id must be a non-empty string.")
        return session_id.strip()

    @classmethod
    def _session_id_for_request(cls, session_id: str | None) -> str:
        """Return *session_id* checked, or a new id: the deployment requires one."""
        if session_id is None:
            return str(uuid.uuid4())
        return cls._validate_session_id(session_id)

    def _merge_llm_params(
        self,
        params: LLMParams | dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = self.generate_params.model_dump()
        if params is None:
            return merged
        overrides = (
            params.model_dump(exclude_unset=True)
            if isinstance(params, LLMParams)
            else params
        )
        merged.update(overrides)
        return merged

    def status(self, *, refresh: bool = False) -> StatusResponse:
        """Return deployment metadata.

        :param refresh: Whether to refresh the metadata.
        :type refresh: bool
        :return: The deployment metadata.
        :rtype: StatusResponse
        """
        if self.metadata is not None and not refresh:
            return self.metadata
        self.metadata = self._fetch_metadata()
        return self.metadata

    def generate(
        self,
        prompts: str | list[str],
        *,
        params: LLMParams | dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> LLMResults:
        """Generate LLM completions.

        :param prompts: The prompts to the agent.
        :type prompts: str | list[str]
        :param params: The parameters to the agent.
        :type params: LLMParams | dict[str, Any] | None
        :param session_id: Chat session to continue, as listed by
            :meth:`list_sessions`. When omitted, a new session id is sent, so the
            call starts a new conversation.
        :type session_id: str | None
        :return: The LLM results.
        :rtype: LLMResults
        """
        self._ensure_llm()
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        self._validate_prompts(prompt_list)
        payload: dict[str, Any] = {
            "prompts": prompt_list,
            "params": self._merge_llm_params(params),
            "session_id": self._session_id_for_request(session_id),
        }
        body = self._request_json("POST", "/generate", json_body=payload)
        return LLMResults.model_validate(body)

    def generate_stream(
        self,
        prompt: str,
        *,
        params: LLMParams | dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Iterator[str]:
        """Stream generated tokens for a single prompt.

        :param prompt: The prompt to the agent.
        :type prompt: str
        :param params: The parameters to the agent.
        :type params: LLMParams | dict[str, Any] | None
        :param session_id: Chat session to continue, as listed by
            :meth:`list_sessions`. When omitted, a new session id is sent, so the
            call starts a new conversation.
        :type session_id: str | None
        :return: An iterator of strings.
        :rtype: Iterator[str]
        """
        self._ensure_llm()
        self._validate_prompts([prompt])
        payload: dict[str, Any] = {
            "prompt": prompt,
            "params": self._merge_llm_params(params),
            "session_id": self._session_id_for_request(session_id),
        }
        yield from self._request_stream("/generate_stream", json_body=payload)

    def list_sessions(self) -> list[SessionInfo]:
        """List the chat sessions the calling user has with this deployment.

        :return: The sessions, most recently updated first as ordered by the
            deployment. Empty when the user has no conversations yet.
        :rtype: list[SessionInfo]
        """
        self._ensure_llm()
        body = self._request_payload("GET", "/sessions")
        rows = body.get("sessions") if isinstance(body, dict) else body
        if not isinstance(rows, list):
            return []
        return [
            SessionInfo.model_validate(row) for row in rows if isinstance(row, dict)
        ]

    def get_session(self, session_id: str) -> SessionDetail:
        """Fetch one chat session and its messages.

        :param session_id: The ID of the session to fetch.
        :type session_id: str
        :return: The session and its messages.
        :rtype: SessionDetail
        """
        self._ensure_llm()
        sid = self._validate_session_id(session_id)
        try:
            body = self._request_json("GET", f"/sessions/{sid}")
        except ArenaInferenceError as exc:
            if exc.status_code == 404:
                raise ArenaInferenceError(
                    status_code=404,
                    detail=f"Session {sid!r} not found.",
                ) from exc
            raise
        return SessionDetail.model_validate(body)

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and every turn the deployment stored for it.

        Deleting is idempotent: a session that is already gone, or that belongs
        to someone else under the deployment's memory scope, answers the same
        way, so this reports whether there was anything to delete rather than
        raising.

        :param session_id: The ID of the session to delete.
        :type session_id: str
        :return: ``True`` if the session was deleted, ``False`` if the deployment
            has no such session visible to you.
        :rtype: bool
        """
        self._ensure_llm()
        sid = self._validate_session_id(session_id)
        resp = self._send("DELETE", f"/sessions/{sid}")
        if resp.status_code == 404:
            return False
        self._raise_for_status(resp, expected=(200, 204))
        return True

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the context manager."""
        self.close()

    def __repr__(self) -> str:
        """Return the representation of the agent.

        :return: The representation of the agent.
        :rtype: str
        """
        return f"<Agent endpoint={self._base_url!r}>"
