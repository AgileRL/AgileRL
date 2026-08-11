# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from agilerl.arena.exceptions import ArenaAuthError, ArenaInferenceError
from agilerl.arena.inference.serde import (
    RLData,
    SerializedRLData,
)
from agilerl.arena.inference.serde import (
    deserialize as _deserialize,
)
from agilerl.arena.inference.serde import (
    get_batch_size as _get_batch_size,
)
from agilerl.arena.inference.serde import (
    serialize as _serialize,
)
from agilerl.arena.typing import JSONValue

logger = logging.getLogger(__name__)

# Module-level aliases for serde (tests and advanced callers).
serialize = _serialize
deserialize = _deserialize
get_batch_size = _get_batch_size

# Re-exported for callers and tests.
__all__ = [
    "Agent",
    "AgentInfo",
    "AgentMetadata",
    "GenerateCompletion",
    "GenerateParams",
    "GenerateResult",
    "LLMCompletionResult",
    "LLMParams",
    "LLMResults",
    "PredictResult",
    "RLData",
    "SerializedRLData",
    "SessionDetail",
    "SessionInfo",
    "SessionMessage",
    "StatusResponse",
    "deserialize",
    "get_batch_size",
    "serialize",
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

    model_config = ConfigDict(frozen=True)

    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


GenerateParams = LLMParams


class AgentInfo(BaseModel):
    """Agent information model.

    :param algo: The algorithm of the agent.
    :type algo: str
    :param multi_agent: Whether the agent is a multi-agent.
    :type multi_agent: bool
    :param llm: Whether the agent is an LLM.
    :type llm: bool
    :param recurrent: Whether the agent is recurrent.
    :type recurrent: bool
    :param supervised: Whether the agent is supervised.
    :type supervised: bool
    """

    model_config = ConfigDict(frozen=True)

    algo: str = ""
    multi_agent: bool = False
    llm: bool = False
    recurrent: bool = False
    supervised: bool = False


class StatusResponse(BaseModel):
    """Status response model.

    :param success: Whether the status request was successful.
    :type success: bool
    :param deployment_id: The deployment ID.
    :type deployment_id: str
    :param instance_id: The instance ID.
    :type instance_id: str
    :param agent: The agent info.
    :type agent: AgentInfo
    """

    model_config = ConfigDict(frozen=True)

    success: bool = True
    deployment_id: str = ""
    instance_id: str = ""
    agent: AgentInfo = Field(default_factory=AgentInfo)


AgentMetadata = StatusResponse


class PredictResult(BaseModel):
    """Metadata from ``POST /predict`` (excluding deserialized tensors).

    :param batch_size: The batch size of the prediction.
    :type batch_size: int
    :param inference_time_ms: The inference time in milliseconds.
    :type inference_time_ms: float
    :param success: Whether the prediction was successful.
    :type success: bool
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    batch_size: int
    inference_time_ms: float
    success: bool = True


class LLMCompletionResult(BaseModel):
    """Single prompt/completion pair from ``POST /generate``.

    :param prompt: The prompt for the LLM completion.
    :type prompt: str
    :param completion: The completion for the LLM completion.
    :type completion: str
    """

    model_config = ConfigDict(frozen=True)

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

    model_config = ConfigDict(frozen=True)

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

    model_config = ConfigDict(extra="ignore", frozen=True)

    role: str = ""
    content: str = ""


class SessionDetail(SessionInfo):
    """A :class:`SessionInfo` plus its messages, from ``GET /sessions/{session_id}``.

    :param messages: The messages in the session.
    :type messages: list[SessionMessage]
    """

    messages: list[SessionMessage] = Field(default_factory=list)


class Agent:
    """HTTP client for a deployed Arena inference endpoint.

    When *probe_on_init* is ``True`` (default), :meth:`status` runs once at
    construction and stores :attr:`metadata`. Wrong endpoint for the deployment
    type returns HTTP 400 from the server.

    Every request carries your own Arena credential as ``Authorization: Bearer``.
    A deployment that keeps memory per user works out who is calling from it, so
    there is nothing else to pass. The credential is never logged, shown in
    :func:`repr`, or written to disk.

    :param endpoint: Base URL of the Arena inference deployment.
    :type endpoint: str
    :param api_key: Profile PAT (``arena_pat_<uuid>_<secret>``) or an access token
        from :meth:`~agilerl.arena.client.ArenaClient.login`. Falls back to the
        ``ARENA_API_KEY`` environment variable.
    :type api_key: str | None
    :param timeout: Request timeout in seconds.
    :type timeout: int
    :param generate_params: Default LLM sampling parameters; defaults to
        :class:`LLMParams` when ``None``.
    :type generate_params: LLMParams | None
    :param probe_on_init: If ``True``, fetch ``GET /status`` during ``__init__``.
    :type probe_on_init: bool
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout: int = 30,
        generate_params: LLMParams | None = None,
        probe_on_init: bool = True,
    ) -> None:
        self._base_url = endpoint.rstrip("/")

        credential = api_key or os.environ.get("ARENA_API_KEY")
        headers: dict[str, str] = {}
        if credential:
            headers["Authorization"] = f"Bearer {credential}"

        self._http = httpx.Client(
            headers=headers,
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
        "client.open_inference_agent(), or pass a profile PAT as api_key."
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
            raise ArenaInferenceError(
                status_code=0,
                detail=f"Network error reaching agent endpoint: {exc}",
            ) from exc

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
            raise ArenaInferenceError(
                status_code=0,
                detail=f"Network error reaching agent endpoint: {exc}",
            ) from exc

    def _fetch_metadata(self) -> StatusResponse:
        body = self._request_json("GET", "/status")
        return StatusResponse.model_validate(body)

    def _agent_info(self) -> AgentInfo:
        if self.metadata is None:
            msg = "Metadata not loaded. Call status() or construct with probe_on_init=True."
            raise ArenaInferenceError(msg)
        return self.metadata.agent

    def _ensure(
        self,
        *,
        rl: bool = False,
        supervised: bool = False,
        llm: bool = False,
    ) -> None:
        agent = self._agent_info()
        if rl and (agent.llm or agent.supervised):
            raise ArenaInferenceError(
                detail="Deployment is not RL. Use predict() or generate().",
            )
        if supervised and (agent.llm or not agent.supervised):
            raise ArenaInferenceError(
                detail="Deployment is not supervised. Use get_action() or generate().",
            )
        if llm and (agent.supervised or not agent.llm):
            raise ArenaInferenceError(
                detail="Deployment is not an LLM. Use get_action() or predict().",
            )

    @staticmethod
    def serialize(data: RLData, batched: bool = False) -> SerializedRLData:
        return _serialize(data, batched)

    @staticmethod
    def deserialize(data: SerializedRLData, batched: bool = False) -> RLData:
        return _deserialize(data, batched)

    @staticmethod
    def get_batch_size(observation: RLData) -> int:
        return _get_batch_size(observation)

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

    @staticmethod
    def _multi_agent_mask(
        info: dict[str, RLData] | None,
    ) -> RLData:
        if info is None or "action_mask" not in info:
            return None
        mask = info["action_mask"]
        if not isinstance(mask, dict):
            msg = (
                "Multi-agent action_mask must be dict[agent_id, array], "
                "not a single array."
            )
            raise ArenaInferenceError(detail=msg)
        return mask

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

    def _build_payload(
        self,
        observation: RLData,
        batched: bool,
        hidden_state: RLData | None,
        info: dict[str, RLData] | None,
        env_defined_actions: RLData,
    ) -> dict[str, Any]:
        """Build the payload for the POST /get_action request."""
        agent = self._agent_info()
        is_multi = agent.multi_agent
        batch_size = get_batch_size(observation) if batched else 1

        # Multi-agent
        if is_multi:
            if not isinstance(observation, dict):
                msg = (
                    "Multi-agent inference requires observation as dict[agent_id, obs]."
                )
                raise ArenaInferenceError(detail=msg)

            payload: dict[str, Any] = {
                "obs": self.serialize(observation, batched),
                "batch_size": batch_size,
            }
            if agent.recurrent:
                payload["hidden_state"] = self.serialize(hidden_state, batched)
            action_mask = self._multi_agent_mask(info)
            if action_mask is not None:
                payload["action_mask"] = self.serialize(action_mask, batched)

            # Environment-defined actions
            eda = env_defined_actions
            if eda is None and info is not None:
                raw_eda = info.get("env_defined_actions")
                if isinstance(raw_eda, dict):
                    eda = raw_eda
            if eda is not None:
                payload["env_defined_actions"] = self.serialize(eda, batched)
            return payload

        # Single-agent
        action_mask = info.get("action_mask") if info is not None else None
        payload = {
            "obs": self.serialize(observation, batched),
            "batch_size": batch_size,
        }

        # Optionally, include hidden state and action mask
        if agent.recurrent:
            payload["hidden_state"] = self.serialize(hidden_state, batched)
        if action_mask is not None:
            payload["action_mask"] = self.serialize(action_mask, batched)

        return payload

    @staticmethod
    def _parse_get_action_response(
        response_json: dict[str, SerializedRLData],
        batched: bool,
        *,
        multi_agent: bool,
        recurrent: bool,
    ) -> tuple[RLData, RLData | None]:
        """Parse the response from the POST /get_action request."""
        action_raw = response_json["action"]
        if multi_agent:
            if not isinstance(action_raw, dict):
                msg = "Expected multi-agent action dict in response."
                raise ArenaInferenceError(status_code=200, detail=msg)
            action: RLData = {
                agent_id: Agent.deserialize(serialized, batched)
                for agent_id, serialized in action_raw.items()
            }
        else:
            deserialized_action = Agent.deserialize(action_raw, batched)
            if deserialized_action is None:
                msg = "Response contained no action."
                raise ArenaInferenceError(status_code=200, detail=msg)
            action = deserialized_action

        hidden_state: RLData | None = None
        if recurrent:
            hidden_raw = response_json.get("hidden_state")
            if hidden_raw is not None:
                hidden_state = Agent.deserialize(hidden_raw, batched)
        return action, hidden_state

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

    def get_action(
        self,
        observation: RLData,
        *,
        batched: bool = False,
        hidden_state: RLData | None = None,
        info: dict[str, RLData] | None = None,
        env_defined_actions: RLData = None,
    ) -> tuple[RLData, RLData | None]:
        """Get actions from a deployed RL agent.

        For multi-agent deployments, pass *observation* as ``dict[agent_id, obs]``.
        Per-agent ``action_mask`` in *info* must be a dict keyed by agent id.

        :param observation: The observation to the agent.
        :type observation: RLData
        :param batched: Whether to batch the observation.
        :type batched: bool
        :param hidden_state: The hidden state of the agent.
        :type hidden_state: RLData | None
        :param info: The info of the agent.
        :type info: dict[str, RLData] | None
        :param env_defined_actions: The env defined actions of the agent.
        :type env_defined_actions: RLData | None
        :return: A tuple containing the action and the hidden state.
        :rtype: tuple[RLData, RLData | None]
        """
        self._ensure(rl=True)
        agent = self._agent_info()
        payload = self._build_payload(
            observation,
            batched,
            hidden_state,
            info,
            env_defined_actions,
        )
        body = self._request_json("POST", "/get_action", json_body=payload)
        return self._parse_get_action_response(
            body,
            batched,
            multi_agent=agent.multi_agent,
            recurrent=agent.recurrent,
        )

    def predict(
        self,
        inputs: RLData,
        *,
        batched: bool = False,
    ) -> tuple[RLData, PredictResult]:
        """Return a prediction from a supervised agent.

        :param inputs: The inputs to the agent.
        :type inputs: RLData
        :param batched: Whether to batch the inputs.
        :type batched: bool
        :return: A tuple containing the prediction and the metadata.
        :rtype: tuple[RLData, PredictResult]
        :raises ArenaInferenceError: If the agent is not supervised.
        """
        self._ensure(supervised=True)
        body = self._request_json(
            "POST",
            "/predict",
            json_body={"inputs": self.serialize(inputs, batched)},
        )
        meta = PredictResult.model_validate(body)
        results = self.deserialize(body["results"], batched)
        if results is None:
            msg = "Response contained no results."
            raise ArenaInferenceError(status_code=200, detail=msg)
        return results, meta

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
            :meth:`list_sessions`. When omitted the field is left out of the
            request and the deployment starts a new conversation; no session id
            is minted here.
        :type session_id: str | None
        :return: The LLM results.
        :rtype: LLMResults
        """
        self._ensure(llm=True)
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        self._validate_prompts(prompt_list)
        payload: dict[str, Any] = {
            "prompts": prompt_list,
            "params": self._merge_llm_params(params),
        }
        if session_id is not None:
            payload["session_id"] = self._validate_session_id(session_id)
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
            :meth:`list_sessions`. When omitted the field is left out of the
            request and the deployment starts a new conversation; no session id
            is minted here.
        :type session_id: str | None
        :return: An iterator of strings.
        :rtype: Iterator[str]
        """
        self._ensure(llm=True)
        self._validate_prompts([prompt])
        payload: dict[str, Any] = {
            "prompt": prompt,
            "params": self._merge_llm_params(params),
        }
        if session_id is not None:
            payload["session_id"] = self._validate_session_id(session_id)
        yield from self._request_stream("/generate_stream", json_body=payload)

    def list_sessions(self) -> list[SessionInfo]:
        """List the chat sessions the calling user has with this deployment.

        :return: The sessions, most recently updated first as ordered by the
            deployment. Empty when the user has no conversations yet.
        :rtype: list[SessionInfo]
        """
        self._ensure(llm=True)
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
        self._ensure(llm=True)
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
        self._ensure(llm=True)
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
