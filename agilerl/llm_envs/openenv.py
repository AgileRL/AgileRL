# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Client half of the OpenEnv glue: drive any text env over a ``/ws`` session or in-process.

The server half lives in :mod:`agilerl.llm_envs.openenv_server` (outside the
``llm`` extra); its names are re-exported here for compatibility.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

try:
    import websockets.exceptions
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.mcp_types import CallToolAction
except ImportError as _exc:  # pragma: no cover - only reachable without the llm extra
    msg = (
        "The OpenEnv backend requires the openenv and websockets packages; "
        "install them with: pip install agilerl[llm]."
    )
    raise ImportError(msg) from _exc

from agilerl.llm_envs.openenv_server import (
    OpenEnvServer,
    OpenEnvWrapper,
    TextAction,
    TextObservation,
    TextState,
    is_url,
    load_env,
    resolve_env,
)
from agilerl.protocols import TextEnvProtocol
from agilerl.utils.algo_utils import is_str_keyed_dict

__all__ = [
    "LocalEnvClient",
    "OpenEnvServer",
    "OpenEnvSessionClient",
    "OpenEnvWrapper",
    "TextAction",
    "TextObservation",
    "TextState",
    "is_url",
    "load_env",
    "resolve_env",
]


logger = logging.getLogger(__name__)

_TransportT = TypeVar("_TransportT")


class LocalEnvClient:
    """In-process ``EnvClientProtocol`` backend — the no-HTTP sibling of :class:`OpenEnvSessionClient`.

    Plain-text envs are driven through the same :class:`OpenEnvWrapper` the server
    hosts, so in-process and URL transports behave identically on the same env.
    OpenEnv ``Environment`` worlds (e.g.
    :class:`~agilerl.llm_envs.prompt_dataset.PromptDatasetEnv`) are driven as-is so
    their rubric metadata is not wrapped away. Used via :meth:`RolloutEnv.local` /
    :meth:`RolloutEnv.from_spec`.

    :param env: The local env — plain-text or an OpenEnv ``Environment``.
    :param instruction: Prompt returned from reset when the env's reset obs is empty.
    """

    def __init__(
        self,
        env: TextEnvProtocol | Environment,
        *,
        instruction: str = "",
    ) -> None:
        """Wrap a local ``env`` as an in-process backend."""
        if isinstance(env, Environment):
            self._backend: Environment = env
        else:
            self._backend = OpenEnvWrapper(env)
        self._env = env
        self._instruction = instruction
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
    ) -> tuple[str, dict[str, Any]]:
        """Reset the local env and return ``(prompt, info)``."""
        obs = self._backend.reset(
            seed=seed,
            row_index=row_index,
            evaluation=True if self._evaluation_mode else None,
        )
        info = dict(obs.metadata) if obs.metadata else {}
        prompt = getattr(obs, "prompt", None) or self._instruction
        return str(prompt), info

    def step(self, action: object) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Step the local env with the action text and return the Gym 5-tuple.

        Env ``info`` — including a rubric's ``rubric_scores`` — travels on the
        observation's metadata.
        """
        text = action if isinstance(action, str) else str(action)
        obs = self._backend.step(TextAction(message=text))
        truncated = bool(getattr(obs, "truncated", False))
        info = dict(obs.metadata) if obs.metadata else {}
        return (
            str(getattr(obs, "prompt", "") or ""),
            float(obs.reward) if obs.reward is not None else 0.0,
            bool(obs.done) and not truncated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Close the wrapped env when it supports it (best-effort)."""
        closer = getattr(self._env, "close", None)
        if callable(closer):
            with contextlib.suppress(Exception):
                closer()

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


class OpenEnvSessionClient:
    """``EnvClientProtocol`` over one OpenEnv ``/ws`` session at a time.

    A session has no resume and *is* the server-side env instance, so a mid-episode
    transport error marks the session broken and fails fast; the client re-dials
    only at the next ``reset`` (an episode boundary), re-resolving ``base_url``
    when it is a callable so a restarted host is found again.

    :param base_url: Root URL of the env server, or a zero-arg callable returning
        it — resolved per dial.
    :param timeout_s: Per-message timeout; ``None`` (default) is unbounded.
    :param connect_timeout_s: Timeout for establishing the session.
    :param mcp_tool: If set, send text as ``call_tool(mcp_tool, {arg: text})`` for MCP servers.
    :param arg: MCP argument name carrying the text (default ``"message"``).
    :param instruction: Prompt returned from reset when the env's reset obs is empty.
    """

    def __init__(
        self,
        base_url: str | Callable[[], str],
        *,
        timeout_s: float | None = None,
        connect_timeout_s: float = 30.0,
        mcp_tool: str | None = None,
        arg: str = "message",
        instruction: str = "",
    ) -> None:
        """Prepare a session against the OpenEnv server at ``base_url`` (dialled lazily)."""
        if not base_url:
            msg = "OpenEnvSessionClient requires a base_url"
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
        self._arg = arg
        self._instruction = instruction
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
            # No keepalive, so ``timeout_s`` is the sole liveness bound: a slow
            # step must not trip the ping deadline before its message timeout.
            websocket_ping_interval_s=None,
        ).sync()

    def _transport(self, call: Callable[[], _TransportT]) -> _TransportT:
        """Run one session round-trip; mark the session broken on transport failure."""
        if self._broken:
            msg = (
                "OpenEnvSessionClient session is broken after a transport error; "
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
    ) -> tuple[str, dict[str, Any]]:
        """Reset the session's env and return ``(prompt, info)``.

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
        return (_observation_text(result.observation) or self._instruction), info

    def step(self, action: object) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Send one action (model text) over the session and return the Gym 5-tuple."""
        self._connect()
        text = action if isinstance(action, str) else str(action)
        if self._mcp_tool:
            # The session client transports plain dicts, so serialize the MCP action.
            act: dict[str, Any] = CallToolAction(
                tool_name=self._mcp_tool,
                arguments={self._arg: text},
            ).model_dump()
        else:
            act = {"message": text}
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
            _observation_text(obs),
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
                        "OpenEnvSessionClient could not read state; assuming no "
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


def _observation_text(obs: object) -> str:
    """Render an OpenEnv observation to prompt text.

    Handles our ``{"prompt": ...}`` / bare-string obs and third-party typed / MCP
    obs (tool-result content blocks, or a ``text`` / ``message`` field).
    """
    if isinstance(obs, str):
        return obs
    if not isinstance(obs, dict):
        return ""
    result = obs.get("result")
    if isinstance(result, dict):
        raw_blocks = result.get("content")
        blocks = raw_blocks if isinstance(raw_blocks, list) else []
        texts: list[str] = []
        for block in blocks:
            if not is_str_keyed_dict(block):
                continue
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
        if texts:
            return "\n".join(texts)
        data = result.get("data")
        if isinstance(data, str):
            return data
    for key in ("prompt", "text", "message", "observation"):
        value = obs.get(key)
        if isinstance(value, str):
            return value
    error = obs.get("error")
    return f"Error: {error}" if error else ""
