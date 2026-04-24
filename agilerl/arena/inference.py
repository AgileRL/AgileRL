from __future__ import annotations

import base64
import io
import logging
from typing import Self, TypeAlias

import httpx
import numpy as np

from agilerl.arena.exceptions import ArenaAPIError, ArenaAuthError

logger = logging.getLogger(__name__)

RLData: TypeAlias = np.ndarray | dict[str, "RLData"] | tuple["RLData", ...]
SerializedRLData: TypeAlias = (
    str | dict[str, "SerializedRLData"] | tuple["SerializedRLData", ...] | None
)


class Agent:
    """HTTP client for a deployed Arena inference endpoint.

    :param endpoint: Full URL of the Arena inference endpoint.
    :type endpoint: str
    :param api_key: API key for authentication with the Arena inference endpoint.
    :type api_key: str or None
    :param timeout: Request timeout in seconds.
    :type timeout: int

    Example::

        agent = Agent(
            "https://<id>.inference.agilerl.com",
            api_key="<api_key>",
        )
        hidden_state = None
        status, action, hidden_state = agent.get_action(
            obs, hidden_state=hidden_state,
        )
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:

        self._endpoint = endpoint
        self._get_action_endpoint = f"{endpoint}/get_action"

        headers: dict[str, str] = {}
        if api_key is not None:
            headers["authorization"] = f"Bearer {api_key}"

        self._http = httpx.Client(
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )

    @staticmethod
    def serialize(data: RLData | None, batched: bool = False) -> SerializedRLData:
        """Serialize RL data to a base64-encoded representation.

        NumPy arrays are saved in ``.npy`` format and base64-encoded.
        Dicts and tuples are recursed into.

        When *batched* is ``False`` (the default) a leading batch
        dimension is added before encoding so that the server always
        receives batched data.

        :param data: Observation, action, hidden state, or nested
            structure thereof.
        :param batched: If ``True``, assume *data* already has a batch
            dimension.
        :returns: A JSON-safe mirror of the input with all arrays
            replaced by base64-encoded strings.
        """
        if isinstance(data, dict):
            return {k: Agent.serialize(v, batched) for k, v in data.items()}
        if isinstance(data, (tuple, list)):
            return tuple(Agent.serialize(v, batched) for v in data)
        if data is None:
            return None

        # Add a batch dimension if not already batched
        if not batched:
            data = np.expand_dims(data, axis=0)

        # Save the data to a buffer and encode it to base64
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def deserialize(data: SerializedRLData, batched: bool = False) -> RLData | None:
        """Deserialize a base64-encoded representation back to RL data.

        Inverse of :meth:`serialize`.

        :param data: Base64-encoded string, nested dict/tuple thereof,
            or ``None``.
        :param batched: If ``True``, return arrays as-is; otherwise
            squeeze the leading batch dimension that :meth:`serialize`
            added.
        :returns: Reconstructed NumPy arrays (or nested structure /
            ``None``).
        """
        if isinstance(data, dict):
            return {k: Agent.deserialize(v, batched) for k, v in data.items()}
        if isinstance(data, (tuple, list)):
            return tuple(Agent.deserialize(v, batched) for v in data)
        if data is None:
            return None

        # Decode the base64-encoded data and load it into a NumPy array
        decoded = base64.b64decode(data, validate=True)
        arr: np.ndarray = np.load(io.BytesIO(decoded), allow_pickle=False)

        # Remove the batch dimension if not batched
        return arr if batched else arr.squeeze(axis=0)

    @staticmethod
    def get_batch_size(observation: RLData) -> int:
        """Extract batch size from the first leaf array in an observation."""
        while isinstance(observation, (dict, tuple)):
            if isinstance(observation, dict):
                observation = next(iter(observation.values()))
            else:
                observation = observation[0]
        return observation.shape[0]

    def _build_payload(
        self,
        observation: RLData,
        batched: bool,
        hidden_state: dict[str, np.ndarray] | None,
        info: dict[str, RLData] | None,
    ) -> dict[str, SerializedRLData]:
        """Build the payload for the inference request.

        :param observation: The environment observation.
        :type observation: RLData
        :param batched: Whether the observation already has a batch
            dimension.
        :type batched: bool
        :param hidden_state: The hidden state of the recurrent agent.
        :type hidden_state: dict[str, np.ndarray] | None
        :param info: Info dict; ``action_mask`` is extracted if present.
        :type info: dict[str, RLData] | None
        :returns: The payload for the inference request.
        :rtype: dict[str, SerializedRLData]
        """
        action_mask = info.get("action_mask") if info is not None else None
        batch_size = self.get_batch_size(observation) if batched else 1
        return {
            "obs": self.serialize(observation, batched),
            "hidden_state": self.serialize(hidden_state, batched),
            "action_mask": self.serialize(action_mask, batched),
            "batch_size": batch_size,
        }

    @staticmethod
    def _parse_response(
        response_json: dict[str, SerializedRLData],
        batched: bool,
    ) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Parse the response from the inference request.

        :param response_json: The JSON response from the inference request.
        :type response_json: dict[str, SerializedRLData]
        :param batched: Whether the observation already has a batch
            dimension.
        :type batched: bool
        :returns: The action and hidden state.
        :rtype: tuple[np.ndarray, dict[str, np.ndarray] | None]
        """
        action = Agent.deserialize(response_json["action"], batched)
        hidden_state = Agent.deserialize(response_json.get("hidden_state"), batched)
        return action, hidden_state

    def get_action(
        self,
        observation: RLData,
        *,
        batched: bool = False,
        hidden_state: dict[str, np.ndarray] | None = None,
        info: dict[str, RLData] | None = None,
    ) -> tuple[int, RLData, dict[str, np.ndarray] | None]:
        """Get an action from a deployed recurrent RL agent.

        :param observation: The environment observation.
        :type observation: RLData
        :param batched: Whether the observation already has a batch
            dimension.
        :type batched: bool
        :param hidden_state: The hidden state of the recurrent agent.
            If ``None``, the server uses an initial hidden state of
            zeros.
        :type hidden_state: dict[str, np.ndarray] | None
        :param info: Info dict; ``action_mask`` is extracted if present.
        :type info: dict[str, RLData] | None
        :returns: ``(status_code, action, next_hidden_state)``
        :rtype: tuple[int, RLData, dict[str, np.ndarray] | None]
        """
        payload = self._build_payload(observation, batched, hidden_state, info)

        try:
            resp = self._http.post(self._get_action_endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise ArenaAPIError(
                status_code=0,
                detail=f"Network error reaching agent endpoint: {exc}",
            ) from exc

        if resp.status_code == 401:
            msg = "Agent endpoint returned 401 Unauthorized. Check your token."
            raise ArenaAuthError(msg)

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500] if resp.text else "No details"
            raise ArenaAPIError(
                status_code=resp.status_code,
                detail=str(detail),
            )

        action, next_hidden_state = self._parse_response(resp.json(), batched)
        return (resp.status_code, action, next_hidden_state)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Agent endpoint={self._endpoint!r}>"
