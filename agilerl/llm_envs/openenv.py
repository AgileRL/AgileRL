"""AgileRL <-> OpenEnv glue: drive any text env over a URL or in-process.

A :class:`~agilerl.llm_envs.rollout_env.RolloutEnv` reaches its env through an
:class:`OpenEnvSessionClient` (env hosted at a URL, over an OpenEnv ``/ws`` session)
or a :class:`LocalEnvClient` (same process, no HTTP). :class:`OpenEnvWrapper` adapts a
plain-text env to OpenEnv's typed ``Environment``; :class:`OpenEnvServer` hosts one
in-process (with ``make_env`` + ``max_concurrent_envs``, a fresh env per session).
``/ws`` rather than REST because only it carries per-session env state and reaches a
production deployment.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from agilerl import HAS_LLM_DEPENDENCIES

if HAS_LLM_DEPENDENCIES:
    import websockets.exceptions
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata
    from openenv.core.env_server.types import Action, Observation, State
else:  # pragma: no cover - only reachable without the llm extra
    msg = (
        "The OpenEnv backend requires the LLM extra; "
        "install it with: pip install agilerl[llm]."
    )
    raise ImportError(msg)

from agilerl.protocols import TextEnvProtocol
from agilerl.utils.env_utils import resolve_entrypoint_target

if TYPE_CHECKING:
    from typing import Self


logger = logging.getLogger(__name__)


class TextAction(Action):
    """OpenEnv action carrying the policy's generated text."""

    message: str = ""


class TextObservation(Observation):
    """OpenEnv observation carrying the next prompt text.

    ``truncated`` is declared because the wire has only ``done`` (no
    terminated/truncated split), so it can travel and reconstruct the 5-tuple.
    """

    prompt: str = ""
    truncated: bool = False


class OpenEnvWrapper(Environment):
    """Adapt a plain-text env to OpenEnv's typed ``Environment`` ABC.

    Translates between the env's string ``reset``/``step`` and OpenEnv's typed
    ``Action``/``Observation``/``State``, surfacing ``dataset_size``/``tools`` on
    the state. Env ``info`` is not sent on the wire.

    :param inner: The local env to host.
    :param env_name: Name in the OpenEnv metadata; defaults to ``inner``'s class name.
    :param owns_inner: If ``True``, ``close`` closes ``inner`` (the per-session path).
    """

    # Lets OpenEnv allow max_concurrent_envs > 1: each session gets its own
    # fresh inner env, so sessions never share state.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        inner: TextEnvProtocol,
        *,
        env_name: str | None = None,
        owns_inner: bool = False,
    ) -> None:
        """Wrap ``inner`` as an OpenEnv environment."""
        super().__init__()
        self._inner = inner
        self._env_name = env_name
        self._owns_inner = owns_inner
        params = inspect.signature(inner.reset).parameters
        # A **kwargs reset accepts any forwardable name (don't drop seed/row).
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            self._reset_params = {"seed", "row_index", "evaluation"}
        else:
            self._reset_params = set(params)
        self._state = State()

    def get_metadata(self) -> EnvironmentMetadata:
        """Report the wrapped env's name in OpenEnv metadata, not ``OpenEnvWrapper``."""
        name = self._env_name or type(self._inner).__name__
        return EnvironmentMetadata(
            name=name, description=f"{name} environment", version="1.0.0"
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> TextObservation:
        """Reset the inner env, returning the initial prompt as a ``TextObservation``."""
        call: dict[str, Any] = {}
        if seed is not None and "seed" in self._reset_params:
            call["seed"] = seed
        for name in ("row_index", "evaluation"):
            if name in self._reset_params and kwargs.get(name) is not None:
                call[name] = kwargs[name]
        prompt, _info = _normalize_reset(self._inner.reset(**call))
        self._state = State(episode_id=episode_id, step_count=0)
        return TextObservation(prompt=prompt, reward=None, done=False)

    def step(
        self,
        action: TextAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> TextObservation:
        """Step the inner env with the action's text, returning a ``TextObservation``."""
        del timeout_s, kwargs
        prompt, reward, terminated, truncated, info = _normalize_step(
            self._inner.step(action.message)
        )
        self._state.step_count += 1
        # Prefix/suffix are folded into ``prompt``; remaining info keys (e.g.
        # ``reward_components``) are kept on metadata for in-process clients.
        metadata = {
            key: value
            for key, value in (info or {}).items()
            if key not in ("prefix", "suffix")
        }
        return TextObservation(
            prompt=prompt,
            reward=reward,
            done=bool(terminated or truncated),
            truncated=bool(truncated),
            metadata=metadata,
        )

    @property
    def state(self) -> State:
        """OpenEnv state, carrying the inner env's ``dataset_size`` / ``tools``."""
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            dataset_size=int(getattr(self._inner, "dataset_size", 0) or 0),
            tools=list(getattr(self._inner, "tools", None) or []),
        )

    def close(self) -> None:
        """Close ``inner`` when this wrapper owns it (per-session path), else no-op."""
        if not self._owns_inner:
            return
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            with contextlib.suppress(Exception):
                closer()


def _normalize_reset(result: Any) -> tuple[str, dict[str, Any]]:
    """Normalise an env ``reset`` return into ``(prompt, info)``."""
    if isinstance(result, tuple):
        if len(result) >= 2:
            return str(result[0]), (result[1] or {})
        if len(result) == 1:
            return str(result[0]), {}
    return str(result), {}


def _normalize_step(result: Any) -> tuple[str, Any, bool, bool, dict[str, Any]]:
    """Normalise an env ``step`` return into the Gym 5-tuple (accepts the legacy 4)."""
    if not isinstance(result, tuple):
        msg = "env.step must return a tuple"
        raise TypeError(msg)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
        obs, reward, terminated, info = result
        truncated = False
    else:
        msg = f"env.step returned a {len(result)}-tuple; expected 4 or 5"
        raise ValueError(msg)
    return str(obs), reward, bool(terminated), bool(truncated), (info or {})


class OpenEnvServer:
    """Serve OpenEnv's ``create_app`` on uvicorn in a background daemon thread.

    Binds an ephemeral port (read from :attr:`base_url`) so any OpenEnv client can
    reach it by URL — the building block for hosting an env in a Ray actor or container.
    Pass ``env`` to share one env (one session at a time), or ``make_env`` with
    ``max_concurrent_envs`` for a fresh env per ``/ws`` session (concurrent, isolated
    episodes — enough to back a whole :class:`BatchRolloutEnv` group).

    :param env: A single shared local env. Exactly one of ``env``/``make_env``.
    :param make_env: Zero-arg factory building a fresh env per session.
    :param host: Interface to bind (default loopback).
    :param port: TCP port; ``0`` lets the OS pick one.
    :param env_name: Name in the env's OpenEnv metadata; defaults to its class name.
    :param max_concurrent_envs: Max live sessions; set to the group size with ``make_env``.
    """

    def __init__(
        self,
        env: TextEnvProtocol | None = None,
        *,
        make_env: Callable[[], TextEnvProtocol] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        env_name: str | None = None,
        max_concurrent_envs: int | None = None,
    ) -> None:
        """Build (but do not start) a server hosting ``env`` or ``make_env``."""
        if (env is None) == (make_env is None):
            msg = "OpenEnvServer requires exactly one of env or make_env"
            raise ValueError(msg)
        self._env = env
        self._make_env = make_env
        self._host = host
        self._port = port
        self._env_name = env_name
        self._max_concurrent_envs = max_concurrent_envs
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._bound_port: int | None = None
        self._env_closed = False

    @property
    def base_url(self) -> str:
        """The ``http://host:port`` the server is bound to (after :meth:`start`)."""
        if self._bound_port is None:
            msg = "OpenEnvServer is not running; call start() first"
            raise RuntimeError(msg)
        return f"http://{self._host}:{self._bound_port}"

    def start(self) -> Self:
        """Serve in a background daemon thread (waits for bind); returns ``self``."""
        import uvicorn

        env = self._env
        make_env = self._make_env
        env_name = self._env_name

        def app_factory() -> OpenEnvWrapper:
            # ``make_env`` -> a fresh owned env per session; ``env`` -> one shared.
            if make_env is not None:
                return OpenEnvWrapper(make_env(), env_name=env_name, owns_inner=True)
            return OpenEnvWrapper(env, env_name=env_name)

        display_name = env_name or (
            type(env).__name__ if env is not None else "OpenEnvServer"
        )
        app = create_app(
            app_factory,
            TextAction,
            TextObservation,
            env_name=display_name,
            max_concurrent_envs=self._max_concurrent_envs,
        )
        config = uvicorn.Config(
            app, host=self._host, port=self._port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="openenv-server", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + 30.0
        failure: str | None = None
        while not getattr(self._server, "started", False):
            if not self._thread.is_alive():
                failure = (
                    "OpenEnvServer thread exited during startup — the port may "
                    f"be in use or the app failed to start "
                    f"(host={self._host!r}, port={self._port})"
                )
                break
            if time.monotonic() > deadline:
                failure = "OpenEnvServer failed to start within 30s"
                break
            time.sleep(0.02)
        if failure is not None:
            self.stop()  # don't leak the thread or hosted env on a failed start
            raise RuntimeError(failure)
        self._bound_port = self._server.servers[0].sockets[0].getsockname()[1]
        return self

    def stop(self) -> None:
        """Stop serving, release the socket, and close the hosted env once (idempotent)."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._server = None
        self._bound_port = None
        if not self._env_closed:
            self._env_closed = True
            closer = getattr(self._env, "close", None)
            if callable(closer):
                with contextlib.suppress(Exception):
                    closer()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class LocalEnvClient:
    """In-process ``EnvClientProtocol`` backend — the no-HTTP sibling of :class:`OpenEnvSessionClient`.

    Drives the env through the same :class:`OpenEnvWrapper` the server hosts, so
    in-process and URL transports behave identically on the same env. Used via
    :meth:`RolloutEnv.local` / :meth:`RolloutEnv.from_spec`.

    :param env: The local env.
    :param instruction: Prompt returned from reset when the env's reset obs is empty.
    """

    def __init__(self, env: TextEnvProtocol, *, instruction: str = "") -> None:
        """Wrap a local ``env`` as an in-process backend."""
        self._env = env
        self._wrapper = OpenEnvWrapper(env)
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
        obs = self._wrapper.reset(
            seed=seed,
            row_index=row_index,
            evaluation=True if self._evaluation_mode else None,
        )
        return (obs.prompt or self._instruction), {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Step the local env with the action text and return the Gym 5-tuple."""
        text = action if isinstance(action, str) else str(action)
        obs = self._wrapper.step(TextAction(message=text))
        truncated = bool(obs.truncated)
        return (
            obs.prompt,
            float(obs.reward) if obs.reward is not None else 0.0,
            bool(obs.done) and not truncated,
            truncated,
            dict(obs.metadata) if obs.metadata else {},
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
        return int(self._wrapper.state.dataset_size or 0)

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the env advertises (empty when none)."""
        return list(self._wrapper.state.tools or [])


class OpenEnvSessionClient:
    """``EnvClientProtocol`` over one OpenEnv ``/ws`` session per instance.

    Wraps OpenEnv's :class:`~openenv.core.GenericEnvClient`. Each instance opens its
    own session against a fresh server-side env, so one URL backs a whole
    :class:`BatchRolloutEnv` group up to ``max_concurrent_envs``. It is the only
    backend that reaches a production OpenEnv server (``/ws``, no REST routes).

    Single-use after a transport error: ``/ws`` has no resume, so the first failure
    marks the client broken and later calls fail fast — retry by building a new client.

    :param base_url: Root URL (``http(s)://`` or ``ws(s)://``) of the env server.
    :param timeout_s: Per-message timeout; ``None`` (default) is unbounded.
    :param connect_timeout_s: Timeout for establishing the session.
    :param mcp_tool: If set, send text as ``call_tool(mcp_tool, {arg: text})`` for MCP servers.
    :param arg: MCP argument name carrying the text (default ``"message"``).
    :param instruction: Prompt returned from reset when the env's reset obs is empty.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float | None = None,
        connect_timeout_s: float = 30.0,
        mcp_tool: str | None = None,
        arg: str = "message",
        instruction: str = "",
    ) -> None:
        """Open a WebSocket session against the OpenEnv server at ``base_url``."""
        if not base_url:
            msg = "OpenEnvSessionClient requires a base_url"
            raise ValueError(msg)
        from openenv.core import GenericEnvClient

        self._sync = GenericEnvClient(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=timeout_s,  # None -> unbounded (asyncio.wait_for)
            # No keepalive, so ``timeout_s`` is the sole liveness bound: a slow
            # step must not trip the ping deadline before its message timeout.
            websocket_ping_interval_s=None,
        ).sync()
        self._mcp_tool = mcp_tool
        self._arg = arg
        self._instruction = instruction
        self._evaluation_mode = False
        self._state: dict[str, Any] | None = None
        self._connected = False
        self._broken = False

    def _transport(self, call: Callable[[], Any]) -> Any:
        """Run one session round-trip; mark the client broken on transport failure."""
        if self._broken:
            msg = (
                "OpenEnvSessionClient session is broken after a transport "
                "error; build a new client (a fresh session) to retry."
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
        if not self._connected:
            self._transport(self._sync.connect)
            self._connected = True

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

        ``seed`` / ``row_index`` travel to the server so a group resets to the same prompt.
        """
        self._connect()
        kwargs: dict[str, Any] = {}
        if seed is not None and int(seed) >= 0:
            kwargs["seed"] = int(seed)
        if row_index is not None:
            kwargs["row_index"] = int(row_index)
        if self._evaluation_mode:
            kwargs["evaluation"] = True
        result = self._transport(lambda: self._sync.reset(**kwargs))
        return (_observation_text(result.observation) or self._instruction), {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Send one action (model text) over the session and return the Gym 5-tuple."""
        self._connect()
        text = action if isinstance(action, str) else str(action)
        if self._mcp_tool:
            act: dict[str, Any] = {
                "type": "call_tool",
                "tool_name": self._mcp_tool,
                "arguments": {self._arg: text},
            }
        else:
            act = {"message": text}
        result = self._transport(lambda: self._sync.step(act))
        obs = result.observation
        reward = result.reward
        # Our servers carry ``truncated`` in the obs; a server that omits it
        # reports every end as a plain termination.
        truncated = (
            bool(obs.get("truncated", False)) if isinstance(obs, dict) else False
        )
        done = bool(result.done)
        return (
            _observation_text(obs),
            float(reward) if reward is not None else 0.0,
            done and not truncated,
            truncated,
            {},
        )

    def close(self) -> None:
        """End the session and stop the client's background event loop."""
        with contextlib.suppress(Exception):
            self._sync.close()

    @property
    def dataset_size(self) -> int:
        """Dataset rows the env serves, from its ``state`` (``0`` if not dataset-backed)."""
        return int(self._fetch_state().get("dataset_size", 0) or 0)

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the env advertises over the session (empty when none)."""
        return list(self._fetch_state().get("tools") or [])

    def _fetch_state(self) -> dict[str, Any]:
        """Fetch + cache the env's OpenEnv ``state`` (dataset size + tool schemas).

        An application-level ``state`` error means no dataset / no tools; a
        transport failure propagates (rather than resurfacing on the first reset).
        """
        if self._state is None:
            self._connect()
            try:
                state = self._transport(self._sync.state)
            except Exception as exc:
                if _is_transport_error(exc):
                    raise
                logger.warning(
                    "OpenEnvSessionClient could not read state; assuming no "
                    "dataset and no tools.",
                    exc_info=True,
                )
                state = {}
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


def _observation_text(obs: Any) -> str:
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
        blocks = result.get("content") or []
        texts = [
            block["text"]
            for block in blocks
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if texts:
            return "\n".join(texts)
        if isinstance(result.get("data"), str):
            return result["data"]
    for key in ("prompt", "text", "message", "observation"):
        value = obs.get(key)
        if isinstance(value, str):
            return value
    error = obs.get("error")
    return f"Error: {error}" if error else ""


def resolve_env(
    spec: str,
    env_config: dict[str, Any] | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[str, OpenEnvServer | None]:
    """Resolve an env spec to a ``(url, server)`` a ``RolloutEnv`` can hit.

    A **URL** is already hosted -> ``(url, None)``. Otherwise ``spec`` is a
    ``module:Class`` / ``path.py:Class`` entrypoint: built with ``env_config``,
    hosted on a local :class:`OpenEnvServer`, returned as ``(server.base_url, server)``.
    """
    if is_url(spec):
        return spec, None
    if ":" not in spec:
        msg = (
            f"env spec {spec!r} is neither a URL nor a 'module:Class' / "
            "'path.py:Class' entrypoint"
        )
        raise ValueError(msg)
    env = resolve_entrypoint_target(spec)(**(env_config or {}))
    server = OpenEnvServer(
        env, host=host, port=port, env_name=_name_from_spec(spec)
    ).start()
    return server.base_url, server


def load_env(spec: str, env_config: dict[str, Any] | None = None) -> TextEnvProtocol:
    """Build the env from a ``module:Class`` / ``path.py:Class`` entrypoint (no hosting).

    The in-process counterpart to :func:`resolve_env`: returns the env object for
    :meth:`RolloutEnv.local` / :meth:`RolloutEnv.from_spec` to wrap in a :class:`LocalEnvClient`.
    """
    return resolve_entrypoint_target(spec)(**(env_config or {}))


def is_url(spec: str) -> bool:
    """Whether ``spec`` is an HTTP(S) URL (already hosted) rather than an env to load."""
    return isinstance(spec, str) and spec.startswith(("http://", "https://"))


def _name_from_spec(spec: str) -> str:
    """Trailing identifier of an entrypoint / path (``"pkg:Env-v0"`` -> ``"Env-v0"``)."""
    tail = spec.rsplit(":", 1)[-1]
    return tail.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or spec
