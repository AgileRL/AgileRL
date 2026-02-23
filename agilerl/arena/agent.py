from __future__ import annotations

import logging
from typing import Any, Self

import httpx
import numpy as np

from agilerl.arena.exceptions import ArenaAPIError, ArenaAuthError

logger = logging.getLogger(__name__)


def _serialize_observation(obs: Any) -> Any:
    """Convert numpy arrays (and nested structures) to JSON-safe types."""
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    if isinstance(obs, dict):
        return {k: _serialize_observation(v) for k, v in obs.items()}
    if isinstance(obs, (list, tuple)):
        return [_serialize_observation(item) for item in obs]
    if isinstance(obs, np.integer):
        return int(obs)
    if isinstance(obs, np.floating):
        return float(obs)
    return obs


class Agent:
    """Thin HTTP client for a deployed Arena inference endpoint.

    :param endpoint: Full URL of the deployed agent endpoint
        (e.g. ``https://arena.agilerl.com/agents/<id>/predict``).
    :type endpoint: str
    :param api_key: Bearer token for the endpoint.  When ``None``,
        requests are sent without authentication (suitable for local /
        test endpoints only).
    :type api_key: str or None
    :param timeout: Request timeout in seconds.
    :type timeout: int
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        self._endpoint = endpoint
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.Client(
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )

    def predict(self, observation: Any) -> Any:
        """Send an observation to the deployed agent and return its action.

        :param observation: Environment observation.  NumPy arrays are
            automatically serialized to nested lists.
        :type observation: Any

        :return: The action returned by the deployed agent.
        :rtype: Any
        """
        payload = {"observation": _serialize_observation(observation)}

        try:
            resp = self._http.post(self._endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise ArenaAPIError(
                status_code=0,
                detail=f"Network error reaching agent endpoint: {exc}",
            ) from exc

        if resp.status_code == 401:
            msg = "Agent endpoint returned 401 Unauthorized. Check your api_key."
            raise ArenaAuthError(msg)

        if not resp.is_success:
            detail = resp.text[:500] if resp.text else "No details"
            raise ArenaAPIError(
                status_code=resp.status_code,
                detail=detail,
            )

        data = resp.json()
        return data.get("action", data)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Agent endpoint={self._endpoint!r}>"
