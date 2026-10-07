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
import random
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.llm_envs.env_sources import redact_url_userinfo
from agilerl.llm_envs.openenv_server import OpenEnvWrapper, wire_types
from agilerl.llm_envs.tool_parsers import ParsedToolCall
from agilerl.protocols import TextEnvProtocol
from agilerl.utils.algo_utils import is_str_keyed_dict

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    import websockets.exceptions
    from openenv.core import GenericEnvClient
    from openenv.core.env_server.interfaces import Environment, Observation
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

__all__ = [
    "SESSION_LOOP",
    "InProcessEnvClient",
    "RemoteEnvClient",
    "SessionLoop",
]

TransportT = TypeVar("TransportT")


class SessionLoop:
    """One background event loop that every :class:`RemoteEnvClient` session runs on.

    A session then costs one socket. A loop per session would add a thread and the
    loop's own descriptors (epoll, wakeup pipe) for every concurrent episode.
    """

    def __init__(self) -> None:
        """Defer starting the loop thread to the first call."""
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def run(self, call: Callable[[], Awaitable[TransportT]]) -> TransportT:
        """Await ``call()`` on the loop and block the calling thread for its result."""

        # OpenEnv clients pick sync or async dispatch from the running loop of the
        # thread that calls them, so ``call`` itself runs on the loop thread.
        async def invoke() -> TransportT:
            return await call()

        return asyncio.run_coroutine_threadsafe(invoke(), self._started()).result()

    def _started(self) -> asyncio.AbstractEventLoop:
        """The loop, starting its thread on first use."""
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._loop.run_forever,
                    name="openenv-sessions",
                    daemon=True,
                ).start()
            return self._loop


SESSION_LOOP = SessionLoop()


class InProcessEnvClient:
    """Drives an environment object living in this process — no network involved.

    A plain-text env is wrapped in :class:`OpenEnvWrapper` so it behaves exactly as
    it would when hosted behind a URL; an OpenEnv ``Environment`` (e.g.
    :class:`~agilerl.llm_envs.qa_dataset.QADatasetEnv`) is used as-is, which
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
        """Step the local env and return the Gym 5-tuple.

        Env ``info`` — including a rubric's ``rubric_scores`` — travels on the
        observation's metadata. Parsed tool calls step MCP backends; text steps
        text backends. Crossed shapes raise.
        """
        if isinstance(self._backend, MCPEnvironment):
            return self._step_mcp(action)
        if isinstance(action, ParsedToolCall):
            msg = "Parsed tool calls need an MCP backend; this env takes text."
            raise ValueError(msg)
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

    def _step_mcp(
        self, action: object
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step an MCP backend with a parsed tool call."""
        if not isinstance(action, ParsedToolCall):
            msg = "MCP backends take parsed tool calls; the harness parses them."
            raise TypeError(msg)
        if action.name is None or action.arguments is None:
            msg = "MCP backends take valid calls; malformed ones never step."
            raise ValueError(msg)
        obs = self._backend.step(
            self._action_cls.model_validate(
                _call_tool_action(action.name, action.arguments)
            )
        )
        truncated = bool(getattr(obs, "truncated", False))
        info = dict(obs.metadata) if obs.metadata else {}
        info.setdefault("role", "tool")
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
        """Tool schemas the env advertises (empty when none).

        An MCP backend lists its tools; anything else reads the state channel.
        """
        if isinstance(self._backend, MCPEnvironment):
            listed = self._backend.step(ListToolsAction())
            return _openai_tools(getattr(listed, "tools", None) or [])
        return list(getattr(self._backend.state, "tools", None) or [])

    @property
    def takes_tool_calls(self) -> bool:
        """Whether steps take parsed tool calls (an MCP backend) rather than text."""
        return isinstance(self._backend, MCPEnvironment)

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

    Calls block the calling thread while the session's I/O runs on
    :data:`SESSION_LOOP`, shared by every session in the process.

    :param base_url: Root URL of the env server, or a zero-arg callable returning
        it — called again on every reconnect, so a host that moved is found again.
    :param timeout_s: Per-message timeout; ``None`` (default) is unbounded.
    :param connect_timeout_s: Timeout for establishing the session.
    :param mcp_tool: If set, send text as ``call_tool(mcp_tool, {arg: text})`` for MCP servers.
    :param action_field: Action field (or MCP argument name) carrying the text.
    :param tasks: Per-row reset kwargs; ``row_index`` selects the entry merged into ``reset``.
    :param eval_tasks: Held-out per-row reset kwargs used in :meth:`eval_mode` when set.
    """

    def __init__(
        self,
        base_url: str | Callable[[], str],
        *,
        timeout_s: float | None = None,
        connect_timeout_s: float = 30.0,
        mcp_tool: str | None = None,
        action_field: str = "message",
        tasks: Sequence[Mapping[str, Any]] | None = None,
        eval_tasks: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Prepare a session against the OpenEnv server at ``base_url`` (dialled lazily)."""
        if not base_url:
            msg = "RemoteEnvClient requires a base_url"
            raise ValueError(msg)
        if tasks is not None and len(tasks) == 0:
            msg = "tasks must not be empty"
            raise ValueError(msg)
        if eval_tasks is not None and len(eval_tasks) == 0:
            msg = "eval_tasks must not be empty"
            raise ValueError(msg)
        if isinstance(base_url, str):
            self._url_provider: Callable[[], str] = lambda url=base_url: url
        else:
            self._url_provider = base_url
        self._timeout_s = timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._session: GenericEnvClient | None = None
        self._mcp_tool = mcp_tool
        self._action_field = action_field
        self._tasks = _copy_task_rows(tasks) if tasks is not None else None
        self._eval_tasks = (
            _copy_task_rows(eval_tasks) if eval_tasks is not None else None
        )
        self._evaluation_mode = False
        self._state: dict[str, Any] | None = None
        self._tools_cache: list[Any] | None = None
        self._connected = False
        self._broken = False
        self._redials = 0

    def _build_session(self) -> GenericEnvClient:
        """Build a fresh (unconnected) session against the provider's current URL."""
        return GenericEnvClient(
            base_url=self._url_provider(),
            connect_timeout_s=self._connect_timeout_s,
            # None -> unbounded; OpenEnv's annotation omits it but forwards
            # straight to asyncio.wait_for, where None means no timeout.
            message_timeout_s=self._timeout_s,  # ty: ignore[invalid-argument-type]
            # A ping while the observation is written closes the socket.
            websocket_ping_interval_s=None,
            websocket_ping_timeout_s=None,
        )

    def _transport(self, call: Callable[[], Awaitable[TransportT]]) -> TransportT:
        """Run one session round-trip on :data:`SESSION_LOOP`; mark the session broken on transport failure."""
        if self._broken:
            msg = (
                "RemoteEnvClient session is broken after a transport error; "
                "the episode it carried is lost. The next reset() re-dials a "
                "fresh session."
            )
            raise RuntimeError(msg)
        try:
            return SESSION_LOOP.run(call)
        except Exception as exc:
            if _is_transport_error(exc):
                self._broken = True
            _strip_url_userinfo_from_exc(exc)
            raise

    def _connect(self) -> GenericEnvClient:
        """Open the session on first use (idempotent), so construction is cheap."""
        if self._session is None:
            self._session = self._build_session()
        session = self._session
        if not self._connected:
            self._transport(session.connect)
            self._connected = True
        return session

    def _drop_session(self) -> None:
        """Close the live session (if any) and clear flags so the next dial is fresh."""
        with contextlib.suppress(Exception):
            if self._session is not None:
                SESSION_LOOP.run(self._session.close)
        self._session = None
        self._connected = False
        self._broken = False
        self._state = None
        self._tools_cache = None

    def _redial(self, attempt: int) -> None:
        """Replace a broken session with a fresh one, at an episode boundary only.

        Jittered backoff, doubling with ``attempt``, so a host restart does not
        stampede every slot's reconnect into the server's capacity limit at once.
        """
        self._redials += 1
        # Drop before backoff so a CAPACITY_REACHED host is not held during jitter.
        self._drop_session()
        time.sleep(random.uniform(0.05, 0.35) * 2**attempt)

    def _at_boundary(
        self, call: Callable[[GenericEnvClient], Awaitable[TransportT]]
    ) -> TransportT:
        """Run ``call`` on a live session between episodes, re-dialling on transport failure.

        No episode is in flight, so a failed dial or round-trip is retried on a
        fresh session; the fourth transport failure propagates. Application
        errors propagate at once.
        """
        attempts = 4
        attempt = 0
        while True:
            if self._broken:
                self._redial(attempt)
            try:
                session = self._connect()
                return self._transport(partial(call, session))
            except Exception as exc:
                attempt += 1
                if not _is_transport_error(exc) or attempt == attempts:
                    raise

    @contextlib.contextmanager
    def eval_mode(self) -> Iterator[None]:
        """Route resets to the env's held-out split within the block."""
        previous = self._evaluation_mode
        self._evaluation_mode = True
        try:
            yield
        finally:
            self._evaluation_mode = previous

    def _active_tasks(self) -> tuple[dict[str, Any], ...] | None:
        """Per-row reset kwargs for the current mode, or ``None`` when unset."""
        if self._evaluation_mode and self._eval_tasks is not None:
            return self._eval_tasks
        return self._tasks

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[object, dict[str, Any]]:
        """Reset the session's env and return ``(payload, info)`` — the observation as sent.

        ``seed`` is sent when it is >= 0. A ``tasks`` list supplies per-row reset
        kwargs merged when ``row_index`` is set. ``evaluation`` is sent in eval mode.
        A broken, unreachable or server-reaped session is re-dialled here, the
        episode boundary (see :meth:`_at_boundary`).
        """
        kwargs: dict[str, Any] = {}
        if seed is not None and int(seed) >= 0:
            kwargs["seed"] = int(seed)
        active_tasks = self._active_tasks()
        if active_tasks is not None:
            if row_index is not None:
                idx = int(row_index)
                if idx < 0 or idx >= len(active_tasks):
                    msg = f"row_index {idx} out of range for {len(active_tasks)} tasks"
                    raise IndexError(msg)
                kwargs.update(active_tasks[idx])
        elif row_index is not None:
            kwargs["row_index"] = int(row_index)
        if self._evaluation_mode:
            kwargs["evaluation"] = True
        result = self._at_boundary(lambda session: session.reset(**kwargs))
        raw_meta = getattr(result, "metadata", None)
        info = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        return result.observation, info

    def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, Any]]:
        """Send one action over the session and return the Gym 5-tuple.

        Parsed tool calls travel as ``CallToolAction``; anything else travels
        as text. The server validates the shape it speaks.
        """
        if isinstance(action, ParsedToolCall):
            return self._step_tool(action)
        text = action if isinstance(action, str) else str(action)
        session = self._connect()
        if self._mcp_tool:
            # The session client transports plain dicts, so serialize the MCP action.
            act: dict[str, Any] = CallToolAction(
                tool_name=self._mcp_tool,
                arguments={self._action_field: text},
            ).model_dump()
        else:
            act = {self._action_field: text}
        result = self._transport(lambda: session.step(act))
        obs = result.observation
        reward = result.reward
        # A server that omits ``truncated`` reports every end as a termination.
        truncated = (
            bool(obs.get("truncated", False)) if isinstance(obs, dict) else False
        )
        done = bool(result.done)
        return (
            obs,
            float(reward) if reward is not None else 0.0,
            done and not truncated,
            truncated,
            _step_info(obs, result),
        )

    def _step_tool(
        self, action: ParsedToolCall
    ) -> tuple[object, float, bool, bool, dict[str, Any]]:
        """Send one parsed tool call as ``CallToolAction``."""
        name, arguments = action.name, action.arguments
        if name is None or arguments is None:
            msg = "Tool sessions take valid calls; malformed ones never step."
            raise ValueError(msg)
        session = self._connect()
        result = self._transport(
            lambda: session.step(_call_tool_action(name, arguments))
        )
        obs = result.observation
        reward = result.reward
        truncated = (
            bool(obs.get("truncated", False)) if isinstance(obs, dict) else False
        )
        done = bool(result.done)
        info = _step_info(obs, result)
        info.setdefault("role", "tool")
        return (
            obs,
            float(reward) if reward is not None else 0.0,
            done and not truncated,
            truncated,
            info,
        )

    def close(self) -> None:
        """End the session."""
        self._drop_session()

    @property
    def dataset_size(self) -> int:
        """Row count: ``len`` of the active task list when ``tasks`` is set, else env ``state``."""
        active_tasks = self._active_tasks()
        if active_tasks is not None:
            return len(active_tasks)
        return int(self._fetch_state().get("dataset_size", 0) or 0)

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the env advertises (empty when none).

        A state carrying tool descriptors answers directly. A bare state
        means an MCP env, whose tools are listed over the session; a server
        that rejects the listing speaks neither shape and has no tools.
        """
        state = self._fetch_state()
        if "tools" in state:
            return list(state.get("tools") or [])
        return list(self._fetch_tools())

    @property
    def takes_tool_calls(self) -> bool:
        """Whether steps take parsed tool calls rather than text.

        True for an MCP env (tools listed over the session) unless
        ``mcp_tool`` wraps the text as one fixed call.
        """
        if self._mcp_tool is not None:
            return False
        return "tools" not in self._fetch_state() and bool(self._fetch_tools())

    @property
    def rubric_components(self) -> tuple[str, ...]:
        """Leaf rubric names advertised by the remote env (empty when none)."""
        return tuple(self._fetch_state().get("rubric_components") or [])

    def _fetch_state(self) -> dict[str, Any]:
        """Fetch + cache the env's OpenEnv ``state`` (dataset size + tool schemas).

        Only the env's fixed descriptors are read from here — ``dataset_size``,
        ``tools``, ``rubric_components`` — which describe the env the session is
        attached to, not the episode running on it, so they cannot change under a
        live session and are cached for the session's life. Per-turn state travels
        on the ``step`` observation instead. :meth:`_drop_session` clears the cache,
        so a re-dial (including one onto a host that moved) reads them again.

        State is only read between episodes, so transport failures re-dial as
        at a reset (:meth:`_at_boundary`). Application-level ``state`` errors
        propagate without retry.
        """
        if self._state is None:
            state = self._at_boundary(lambda session: session.state())
            self._state = state if isinstance(state, dict) else {}
        return self._state

    def _fetch_tools(self) -> list[Any]:
        """List the session env's tools once, cached for the session's life.

        Only reached for a bare state (no tool descriptors): an MCP env
        answers with its tools. A validation rejection is deterministic —
        the server speaks neither shape — so it caches empty with no redial.
        """
        if self._tools_cache is None:
            self._connect()
            result = self._list_tools_or_none()
            if result is None:
                self._tools_cache = []
            else:
                obs = getattr(result, "observation", None)
                tools = obs.get("tools", []) if isinstance(obs, dict) else []
                self._tools_cache = _openai_tools(tools)
        return self._tools_cache

    def _list_tools_or_none(self) -> object | None:
        """One listing round-trip with a transport retry; ``None`` on rejection."""
        try:
            return self._at_boundary(
                lambda session: session.step({"type": "list_tools", "metadata": {}})
            )
        except Exception as exc:
            if "VALIDATION_ERROR" in str(exc):
                return None
            raise


def _copy_task_rows(
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Store independent dict copies of each row's reset kwargs."""
    return tuple(dict(entry) for entry in tasks)


HTTP_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def _strip_url_userinfo_from_exc(exc: Exception) -> None:
    """Remove credentials from ``exc`` when its message interpolates an HTTP URL."""
    if not exc.args or not isinstance(exc.args[0], str):
        return
    redacted = HTTP_URL_RE.sub(
        lambda match: redact_url_userinfo(match.group(0)),
        exc.args[0],
    )
    if redacted != exc.args[0]:
        exc.args = (redacted, *exc.args[1:])


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


def _step_info(obs: object, result: object) -> dict[str, Any]:
    """Env ``info`` from a step result's metadata and the observation payload."""
    info: dict[str, Any] = {}
    raw_meta = getattr(result, "metadata", None)
    if is_str_keyed_dict(raw_meta):
        info.update(raw_meta)
    if is_str_keyed_dict(obs):
        obs_meta = obs.get("metadata")
        if is_str_keyed_dict(obs_meta):
            info.update(obs_meta)
        if "rubric_scores" in obs and "rubric_scores" not in info:
            info["rubric_scores"] = obs["rubric_scores"]
    return info


def _call_tool_action(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Wire-format ``CallToolAction`` dict for one parsed call."""
    return {
        "type": "call_tool",
        "tool_name": name,
        "arguments": dict(arguments),
        "metadata": {},
    }


def _openai_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """MCP tool specs normalized to OpenAI function schemas."""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("input_schema", tool.get("inputSchema", {}))
        else:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "")
            parameters = getattr(tool, "input_schema", getattr(tool, "inputSchema", {}))
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters if isinstance(parameters, dict) else {},
                },
            }
        )
    return normalized
