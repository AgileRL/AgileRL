from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import numpy as np

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
    "StatusResponse",
    "deserialize",
    "get_batch_size",
    "serialize",
]


class LLMParams(BaseModel):
    """Sampling parameters for ``POST /generate`` and ``POST /generate_stream``."""

    model_config = ConfigDict(frozen=True)

    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


GenerateParams = LLMParams


class AgentInfo(BaseModel):
    """Agent flags from ``GET /status``."""

    model_config = ConfigDict(frozen=True)

    algo: str = ""
    multi_agent: bool = False
    llm: bool = False
    recurrent: bool = False
    supervised: bool = False


class StatusResponse(BaseModel):
    """Response from ``GET /status``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    deployment_id: str = ""
    instance_id: str = ""
    agent: AgentInfo = Field(default_factory=AgentInfo)


AgentMetadata = StatusResponse


class PredictResult(BaseModel):
    """Metadata from ``POST /predict`` (excluding deserialized tensors)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    batch_size: int
    inference_time_ms: float
    success: bool = True


class LLMCompletionResult(BaseModel):
    """Single prompt/completion pair from ``POST /generate``."""

    model_config = ConfigDict(frozen=True)

    prompt: str = ""
    completion: str = ""


GenerateCompletion = LLMCompletionResult


class LLMResults(BaseModel):
    """Response from ``POST /generate``."""

    model_config = ConfigDict(frozen=True)

    results: list[LLMCompletionResult]
    batch_size: int
    inference_time_ms: float
    tokens_per_second: float
    success: bool = True


GenerateResult = LLMResults


class Agent:
    """HTTP client for a deployed Arena inference endpoint.

    When *probe_on_init* is ``True`` (default), :meth:`status` runs once at
    construction and stores :attr:`metadata`. Wrong endpoint for the deployment
    type returns HTTP 400 from the server.

    :param endpoint: Base URL of the Arena inference deployment.
    :param api_key: API key for ``Authorization: Bearer``.
    :param timeout: Request timeout in seconds.
    :param generate_params: Default LLM sampling parameters; defaults to
        :class:`LLMParams` when ``None``.
    :param probe_on_init: If ``True``, fetch ``GET /status`` during ``__init__``.
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

        headers: dict[str, str] = {}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

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

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 401:
            msg = "Agent endpoint returned 401 Unauthorized. Check your token."
            raise ArenaAuthError(msg)
        if resp.status_code != 200:
            raise ArenaInferenceError(
                status_code=resp.status_code,
                detail=self._response_detail(resp),
            )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._url(path)
        try:
            resp = self._http.request(method, url, json=json_body)
        except httpx.HTTPError as exc:
            raise ArenaInferenceError(
                status_code=0,
                detail=f"Network error reaching agent endpoint: {exc}",
            ) from exc

        self._raise_for_status(resp)
        body = resp.json()
        if not isinstance(body, dict):
            msg = f"Expected JSON object from {path}, got {type(body).__name__}."
            raise ArenaInferenceError(status_code=resp.status_code, detail=msg)
        if body.get("success") is False:
            raise ArenaInferenceError(
                status_code=resp.status_code,
                detail=self._response_detail(resp),
            )
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

    def status(self, *, refresh: bool = False) -> StatusResponse:
        """Return deployment metadata from ``GET /status``.

        Fetched automatically on init when *probe_on_init* was ``True``.
        """
        if self.metadata is not None and not refresh:
            return self.metadata
        self.metadata = self._fetch_metadata()
        return self.metadata

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
    def serialize(data: RLData | None, batched: bool = False) -> SerializedRLData:
        return _serialize(data, batched)

    @staticmethod
    def deserialize(data: SerializedRLData, batched: bool = False) -> RLData | None:
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

    @staticmethod
    def _multi_agent_mask(
        info: dict[str, RLData] | None,
    ) -> dict[str, RLData] | None:
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

    def _build_payload(
        self,
        observation: RLData,
        batched: bool,
        hidden_state: dict[str, np.ndarray] | None,
        info: dict[str, RLData] | None,
        env_defined_actions: dict[str, RLData] | None,
    ) -> dict[str, Any]:
        agent = self._agent_info()
        is_multi = agent.multi_agent
        batch_size = get_batch_size(observation) if batched else 1

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
            action_mask = self._multi_agent_mask(info)
            if action_mask is not None:
                payload["action_mask"] = self.serialize(action_mask, batched)
            eda = env_defined_actions
            if eda is None and info is not None:
                raw_eda = info.get("env_defined_actions")
                if isinstance(raw_eda, dict):
                    eda = raw_eda
            if eda is not None:
                payload["env_defined_actions"] = self.serialize(eda, batched)
            return payload

        action_mask = info.get("action_mask") if info is not None else None
        payload = {
            "obs": self.serialize(observation, batched),
            "batch_size": batch_size,
        }
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
    ) -> tuple[RLData, dict[str, np.ndarray] | None]:
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
            action = Agent.deserialize(action_raw, batched)

        hidden_state = None
        if recurrent:
            hidden_raw = response_json.get("hidden_state")
            if hidden_raw is not None:
                deserialized = Agent.deserialize(hidden_raw, batched)
                if isinstance(deserialized, dict):
                    hidden_state = deserialized
        return action, hidden_state

    def get_action(
        self,
        observation: RLData,
        *,
        batched: bool = False,
        hidden_state: dict[str, np.ndarray] | None = None,
        info: dict[str, RLData] | None = None,
        env_defined_actions: dict[str, RLData] | None = None,
    ) -> tuple[RLData, dict[str, np.ndarray] | None]:
        """Get actions from a deployed RL agent (``POST /get_action``).

        For multi-agent deployments, pass *observation* as ``dict[agent_id, obs]``.
        Per-agent ``action_mask`` in *info* must be a dict keyed by agent id.
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
        """Run supervised inference (``POST /predict``).

        Request body is ``{"inputs": <serialized>}`` only.
        """
        self._ensure(supervised=True)
        body = self._request_json(
            "POST",
            "/predict",
            json_body={"inputs": self.serialize(inputs, batched)},
        )
        meta = PredictResult.model_validate(body)
        results = self.deserialize(body["results"], batched)
        return results, meta

    def generate(
        self,
        prompts: str | list[str],
        *,
        params: LLMParams | dict[str, Any] | None = None,
    ) -> LLMResults:
        """Generate LLM completions (``POST /generate``)."""
        self._ensure(llm=True)
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        self._validate_prompts(prompt_list)
        body = self._request_json(
            "POST",
            "/generate",
            json_body={
                "prompts": prompt_list,
                "params": self._merge_llm_params(params),
            },
        )
        return LLMResults.model_validate(body)

    def generate_stream(
        self,
        prompt: str,
        *,
        params: LLMParams | dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream tokens for a single prompt (``POST /generate_stream``)."""
        self._ensure(llm=True)
        self._validate_prompts([prompt])
        payload = {
            "prompt": prompt,
            "params": self._merge_llm_params(params),
        }
        yield from self._request_stream("/generate_stream", json_body=payload)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Agent endpoint={self._base_url!r}>"
