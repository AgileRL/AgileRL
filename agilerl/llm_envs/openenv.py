# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Clients for interacting with an OpenEnv environment.

An environment either runs inside the training process or behind a URL, so there
is one client for each: :class:`InProcessEnvClient` calls the environment object
directly, :class:`RemoteEnvClient` reaches it over a WebSocket session. Both
expose the same surface and hand the observation payload over as received —
rendering it to prompt text is the :class:`~agilerl.llm_envs.rollout.RolloutHarness`'s
job. The server that hosts an environment behind a URL lives in
:mod:`agilerl.llm_envs.openenv_server` (outside the ``llm`` extra).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, TypeVar

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.llm_envs.openenv_server import OpenEnvWrapper, wire_types
from agilerl.protocols import TextEnvProtocol

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    import websockets.exceptions
    from openenv.core.env_server.interfaces import Environment, Observation
    from openenv.core.env_server.mcp_types import CallToolAction

__all__ = [
    "InProcessEnvClient",
    "RemoteEnvClient",
]


logger = logging.getLogger(__name__)

_TransportT = TypeVar("_TransportT")


class InProcessEnvClient:
    """Drives an environment object living in this process — no network involved.

    A plain-text env is wrapped in :class:`OpenEnvWrapper` so it behaves exactly as
    it would when hosted behind a URL; an OpenEnv ``Environment`` (e.g.
    :class:`~agilerl.llm_envs.prompt_dataset.PromptDatasetEnv`) is used as-is, which
    keeps its rubric metadata intact. Built by :meth:`RolloutHarness.local` /
    :meth:`RolloutHarness.from_spec`.

    :param env: The env to drive — plain-text or an OpenEnv ``Environment``.
    :param action_field: The action field the model's text goes into, on the
        action class the env declares (``message`` on our ``TextAction``, but
        e.g. ``code`` for an env whose ``ACTION_CLS`` names it that).
    """

    def __init__(
        self,
        env: TextEnvProtocol | Environment,
        *,
        action_field: str = "message",
    ) -> None:
        """Wrap a local ``env`` as an in-process backend that owns it."""
        if isinstance(env, Environment):
            self._backend: Environment = env
        else:
            self._backend = OpenEnvWrapper(env, owns_inner=True)
        self._action_cls, _ = wire_types(self._backend)
        self._action_field = action_field
        self._evaluation_mode = False

    @contextlib.contextmanager
    def eval_mode(self) -> Iterator[None]:
        """Route resets to the env's held-out split within the block."""
        previous = self._evaluation_mode
        self._evaluation_mode = True
        try:
            yield
        finally:
            self._evaluation_mode = previous

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the local env and return ``(payload, info)`` — the observation's raw fields."""
        obs = self._backend.reset(
            seed=seed,
            row_index=row_index,
            evaluation=True if self._evaluation_mode else None,
        )
        info = dict(obs.metadata) if obs.metadata else {}
        return _observation_payload(obs), info

    def step(
        self, action: object
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step the local env with the action text and return the Gym 5-tuple.

        Env ``info`` — including a rubric's ``rubric_scores`` — travels on the
        observation's metadata.
        """
        text = action if isinstance(action, str) else str(action)
        obs = self._backend.step(
            self._action_cls.model_validate({self._action_field: text})
        )
        truncated = bool(getattr(obs, "truncated", False))
        info = dict(obs.metadata) if obs.metadata else {}
        return (
            _observation_payload(obs),
            float(obs.reward) if obs.reward is not None else 0.0,
            bool(obs.done) and not truncated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Close the wrapped env when it supports it (best-effort)."""
        with contextlib.suppress(Exception):
            self._backend.close()

    @property
    def dataset_size(self) -> int:
        """Dataset rows the env serves (``0`` if not dataset-backed)."""
        return int(getattr(self._backend.state, "dataset_size", 0) or 0)

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the env advertises (empty when none)."""
        return list(getattr(self._backend.state, "tools", None) or [])

    @property
    def rubric_components(self) -> tuple[str, ...]:
        """Leaf rubric names for component metrics (empty when none)."""
        components = getattr(self._backend, "rubric_components", None)
        if components is not None:
            return tuple(components)
        return tuple(getattr(self._backend.state, "rubric_components", None) or ())


class RemoteEnvClient:
    """Drives an environment hosted behind a URL, over one WebSocket session.

    The session *is* the environment instance: the server builds a fresh env when
    the connection opens and destroys it when the connection closes, which is how
    concurrent episodes stay isolated. There is no resume, so a connection lost
    mid-episode fails that episode rather than reconnecting — a fresh connection
    would silently be a fresh environment, while the transcript carried on as if
    nothing had changed. Reconnection therefore happens only at the next
    ``reset``, where a new environment is what the caller wanted anyway.

    :param base_url: Root URL of the env server, or a zero-arg callable returning
        it — called again on every reconnect, so a host that moved is found again.
    :param timeout_s: Per-message timeout; ``None`` (default) is unbounded.
    :param connect_timeout_s: Timeout for establishing the session.
    :param mcp_tool: If set, send text as ``call_tool(mcp_tool, {arg: text})`` for MCP servers.
    :param action_field: Action field (or MCP argument name) carrying the text.
    """

    def __init__(
        self,
        base_url: str | Callable[[], str],
        *,
        timeout_s: float | None = None,
        connect_timeout_s: float = 30.0,
        mcp_tool: str | None = None,
        action_field: str = "message",
    ) -> None:
        """Prepare a session against the OpenEnv server at ``base_url`` (dialled lazily)."""
        if not base_url:
            msg = "RemoteEnvClient requires a base_url"
            raise ValueError(msg)
        if isinstance(base_url, str):
            fixed_url = base_url
            self._url_provider: Callable[[], str] = lambda: fixed_url
        else:
            self._url_provider = base_url
        self._timeout_s = timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._sync: Any = None
        self._mcp_tool = mcp_tool
        self._action_field = action_field
        self._evaluation_mode = False
        self._state: dict[str, Any] | None = None
        self._connected = False
        self._broken = False
        self._redials = 0

    def _build_session(self) -> Any:  # noqa: ANN401 -- OpenEnv's sync client has no public type
        """Build a fresh (unconnected) sync session against the provider's current URL."""
        from openenv.core import GenericEnvClient

        return GenericEnvClient(
            base_url=self._url_provider(),
            connect_timeout_s=self._connect_timeout_s,
            # None -> unbounded; OpenEnv's annotation omits it but forwards
            # straight to asyncio.wait_for, where None means no timeout.
            message_timeout_s=self._timeout_s,  # ty: ignore[invalid-argument-type]
        ).sync()

    def _transport(self, call: Callable[[], _TransportT]) -> _TransportT:
        """Run one session round-trip; mark the session broken on transport failure."""
        if self._broken:
            msg = (
                "RemoteEnvClient session is broken after a transport error; "
                "the episode it carried is lost. The next reset() re-dials a "
                "fresh session."
            )
            raise RuntimeError(msg)
        try:
            return call()
        except Exception as exc:
            if _is_transport_error(exc):
                self._broken = True
            raise

    def _connect(self) -> None:
        """Open the session on first use (idempotent), so construction is cheap."""
        if self._sync is None:
            self._sync = self._build_session()
        if not self._connected:
            self._transport(self._sync.connect)
            self._connected = True

    def _redial(self) -> None:
        """Replace a broken session with a fresh one, at an episode boundary only.

        Jittered backoff so a host restart does not stampede every slot's reconnect
        into the server's capacity limit at once.
        """
        self._redials += 1
        time.sleep(random.uniform(0.05, 0.35))
        with contextlib.suppress(Exception):
            if self._sync is not None:
                self._sync.close()
        self._sync = None
        self._connected = False
        self._broken = False

    def _retry_on_fresh_session(self, call: Callable[[], _TransportT]) -> _TransportT:
        """Re-dial once and re-run ``call`` — boundary-only recovery."""
        self._redial()
        self._connect()
        return self._transport(call)

    @contextlib.contextmanager
    def eval_mode(self) -> Iterator[None]:
        """Route resets to the env's held-out split within the block."""
        previous = self._evaluation_mode
        self._evaluation_mode = True
        try:
            yield
        finally:
            self._evaluation_mode = previous

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[object, dict[str, Any]]:
        """Reset the session's env and return ``(payload, info)`` — the observation as sent.

        ``seed`` / ``row_index`` travel to the server so a group resets to the same
        prompt. A broken or server-reaped session is re-dialled here, the episode
        boundary: one retry, then errors propagate.
        """
        if self._broken:
            self._redial()
        kwargs: dict[str, Any] = {}
        if seed is not None and int(seed) >= 0:
            kwargs["seed"] = int(seed)
        if row_index is not None:
            kwargs["row_index"] = int(row_index)
        if self._evaluation_mode:
            kwargs["evaluation"] = True
        try:
            self._connect()
            result = self._transport(lambda: self._sync.reset(**kwargs))
        except Exception as exc:
            if not _is_transport_error(exc):
                raise
            result = self._retry_on_fresh_session(lambda: self._sync.reset(**kwargs))
        raw_meta = getattr(result, "metadata", None)
        info = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        return result.observation, info

    def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, Any]]:
        """Send one action (model text) over the session and return the Gym 5-tuple."""
        self._connect()
        text = action if isinstance(action, str) else str(action)
        if self._mcp_tool:
            # The session client transports plain dicts, so serialize the MCP action.
            act: dict[str, Any] = CallToolAction(
                tool_name=self._mcp_tool,
                arguments={self._action_field: text},
            ).model_dump()
        else:
            act = {self._action_field: text}
        result = self._transport(lambda: self._sync.step(act))
        obs = result.observation
        reward = result.reward
        # A server that omits ``truncated`` reports every end as a termination.
        truncated = (
            bool(obs.get("truncated", False)) if isinstance(obs, dict) else False
        )
        done = bool(result.done)
        # Env ``info`` (a rubric's ``rubric_scores``, say) rides on the step
        # result's metadata; a server that stamps it on the obs instead is read too.
        info: dict[str, Any] = {}
        raw_meta = getattr(result, "metadata", None)
        if isinstance(raw_meta, dict):
            info.update(raw_meta)
        if isinstance(obs, dict):
            if isinstance(obs.get("metadata"), dict):
                info.update(obs["metadata"])
            if "rubric_scores" in obs and "rubric_scores" not in info:
                info["rubric_scores"] = obs["rubric_scores"]
        return (
            obs,
            float(reward) if reward is not None else 0.0,
            done and not truncated,
            truncated,
            info,
        )

    def close(self) -> None:
        """End the session and stop the client's background event loop."""
        with contextlib.suppress(Exception):
            if self._sync is not None:
                self._sync.close()

    @property
    def dataset_size(self) -> int:
        """Dataset rows the env serves, from its ``state`` (``0`` if not dataset-backed)."""
        return int(self._fetch_state().get("dataset_size", 0) or 0)

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the env advertises over the session (empty when none)."""
        return list(self._fetch_state().get("tools") or [])

    @property
    def rubric_components(self) -> tuple[str, ...]:
        """Leaf rubric names advertised by the remote env (empty when none)."""
        return tuple(self._fetch_state().get("rubric_components") or [])

    def _fetch_state(self) -> dict[str, Any]:
        """Fetch + cache the env's OpenEnv ``state`` (dataset size + tool schemas).

        An application-level ``state`` error means no dataset / no tools; a
        transport failure propagates (rather than resurfacing on the first reset).
        """
        if self._state is None:
            try:
                self._connect()
                state = self._transport(self._sync.state)
            except Exception as exc:
                if not _is_transport_error(exc):
                    logger.warning(
                        "RemoteEnvClient could not read state; assuming no "
                        "dataset and no tools.",
                        exc_info=True,
                    )
                    state = {}
                else:
                    # State is only read between episodes, so this is a boundary.
                    state = self._retry_on_fresh_session(lambda: self._sync.state())
            self._state = state if isinstance(state, dict) else {}
        return self._state


def _is_transport_error(exc: Exception) -> bool:
    """Whether ``exc`` means the session is dead: a drop, timeout, or capacity reject.

    ``CAPACITY_REACHED`` (server at ``max_concurrent_envs``) closes the socket too.
    """
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, OSError)):
        return True
    if isinstance(exc, websockets.exceptions.WebSocketException):
        return True
    return "CAPACITY_REACHED" in str(exc)


def _observation_payload(obs: Observation) -> dict[str, Any]:
    """The observation's raw field values, keyed as they would appear on the wire."""
    return {name: getattr(obs, name) for name in type(obs).model_fields}
