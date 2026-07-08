"""The OpenEnv API: reach any text env the same way — over HTTP or in-process.

Every LLM-training env is driven through the same two calls — ``reset()`` returns a
prompt, ``step(text)`` returns the next prompt, a reward, and done flags — and a
``RolloutEnv`` reaches it through a **backend**: an :class:`OpenEnvClient` when the env
is hosted at a URL (a remote Space, or a local :class:`OpenEnvServer`), or a
:class:`LocalEnvClient` when the env runs in the same process (no HTTP). One env is
reached per backend — each concurrent rollout gets its own.

The pieces here fill the gaps OpenEnv's own classes leave:

* :class:`OpenEnvWrapper` — presents a plain-text env as OpenEnv's typed
  ``Environment`` so OpenEnv's server can host it.
* :class:`OpenEnvServer` — runs OpenEnv's app on uvicorn *in-process* (a daemon thread
  on an ephemeral port, ``start`` / ``stop``); OpenEnv's own hosting is a standalone
  blocking process. Closes the hosted env once, on stop.
* :class:`OpenEnvClient` — a small *synchronous* httpx client for an env at a URL; an
  async caller (e.g. a Ray actor) gets concurrency from the actor boundary, not the
  client.
* :class:`LocalEnvClient` — the in-process sibling: same surface, direct calls, no
  socket.
* :class:`ServedEnvClient` — hosts a local env on an :class:`OpenEnvServer`, drives it
  through an :class:`OpenEnvClient`, and owns both, so one ``close`` tears everything
  down.

:func:`resolve_env` turns a URL or a ``module:Class`` entrypoint into a hosted ``(url,
server)``; :func:`load_env` just builds an entrypoint env (no hosting) for the
in-process path.

Two OpenEnv facts shape the design. Servers must run in **simulation mode**: production
mode does not register ``/reset`` / ``/step`` / ``/state``, so the REST client cannot
drive it. And REST holds no session state, so a served env handles one episode at a
time — hence one server per concurrent rollout (see :meth:`RolloutEnv.serving`).
OpenEnv's WebSocket path does support many sessions per server; a WebSocket-backed
backend is the noted evolution if the per-rollout fleet ever becomes the bottleneck —
today rollout concurrency comes from the distributed (Ray) collector.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata
from openenv.core.env_server.types import Action, Observation, State

from agilerl.protocols import TextEnvProtocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

logger = logging.getLogger(__name__)


class TextAction(Action):
    """OpenEnv action carrying the policy's generated text."""

    message: str = ""


class TextObservation(Observation):
    """OpenEnv observation carrying the next prompt text.

    Inherits ``reward`` / ``done`` / ``metadata`` from ``Observation``. (The OpenEnv
    server round-trips declared fields — ``prompt`` / ``truncated`` here, plus
    ``reward`` / ``done`` — but not ``metadata``, so any prefix/suffix is folded into
    ``prompt`` by :class:`OpenEnvWrapper`.) ``truncated`` distinguishes a time-limit
    end from a natural termination inside ``done``, so the HTTP backend reports the
    same Gym 5-tuple split as the in-process one.
    """

    prompt: str = ""
    truncated: bool = False


class OpenEnvWrapper(Environment):
    """Adapt an env to OpenEnv's typed ``Environment`` ABC.

    A typical env takes and returns plain strings —
    ``reset(seed=None[, row_index, evaluation]) -> (prompt, info)`` and
    ``step(action_text) -> (prompt, reward, terminated, truncated, info)`` — while
    OpenEnv works with typed ``Action`` / ``Observation`` / ``State`` objects. This
    class translates between the two, letting OpenEnv's server host any local env
    unchanged. The env's ``dataset_size`` / ``tools`` are surfaced on the OpenEnv
    ``state`` so a client can read them. ``info``'s ``prefix`` / ``suffix`` are folded
    into the prompt (OpenEnv does not round-trip observation metadata).

    :param inner: The local env to host.
    :param env_name: Name reported in the env's OpenEnv metadata; defaults to
        ``inner``'s class name (so a client sees the real env, not ``OpenEnvWrapper``).
    """

    def __init__(self, inner: TextEnvProtocol, *, env_name: str | None = None) -> None:
        """Wrap ``inner`` as an OpenEnv environment."""
        super().__init__()
        self._inner = inner
        self._env_name = env_name
        self._reset_params = set(inspect.signature(inner.reset).parameters)
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
        prompt, info = _normalize_reset(self._inner.reset(**call))
        self._state = State(episode_id=episode_id, step_count=0)
        return TextObservation(prompt=_fold(prompt, info), reward=None, done=False)

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
        return TextObservation(
            prompt=_fold(prompt, info),
            reward=reward,
            done=bool(terminated or truncated),
            truncated=bool(truncated),
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
        """No-op: the wrapper is a per-request adapter and does not own the env."""


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


def _fold(text: str, info: dict[str, Any] | None) -> str:
    """Fold ``info``'s ``prefix`` / ``suffix`` into the prompt text."""
    if not info:
        return text
    prefix = info.get("prefix", "")
    suffix = info.get("suffix", "")
    if prefix:
        text = f"{prefix}{text}"
    if suffix:
        text = f"{text}\n{suffix}"
    return text


class OpenEnvServer:
    """Run OpenEnv's app on uvicorn in-process.

    OpenEnv builds the FastAPI app (``create_app``), but its hosting is meant to be a standalone,
    blocking process e.g. (a HF Space, a container, ``python -m openenv``).
    To enable envs to be served in-process, this class wraps the same ``create_app`` in a
    uvicorn **daemon thread**, binds an ephemeral port (``port=0``) read back from
    :attr:`base_url`, and exposes ``start`` / ``stop`` (and the context-manager protocol).
    Any OpenEnv client can then reach it by URL.
    This class also enables env servers to be hosted by Ray actors in their own process.

    :param env: The local env to serve.
    :param host: Interface to bind (default loopback).
    :param port: TCP port; ``0`` lets the OS pick one.
    :param env_name: Name advertised in the env's OpenEnv metadata / schema; defaults
        to the env's class name.
    """

    def __init__(
        self,
        env: TextEnvProtocol,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        env_name: str | None = None,
    ) -> None:
        """Build (but do not start) a server hosting ``env``."""
        self._env = env
        self._host = host
        self._port = port
        self._env_name = env_name
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

        app = create_app(
            lambda: OpenEnvWrapper(self._env, env_name=self._env_name),
            TextAction,
            TextObservation,
            env_name=self._env_name or type(self._env).__name__,
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
            # A failed start must not leak the thread or the hosted env.
            self.stop()
            raise RuntimeError(failure)
        self._bound_port = self._server.servers[0].sockets[0].getsockname()[1]
        return self

    def stop(self) -> None:
        """Stop serving, release the socket, and close the hosted env once.

        Idempotent. The env is closed exactly once when the server is torn down.
        """
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


class OpenEnvClient:
    """Synchronous httpx client for an OpenEnv env server (text in, text out).

    Implements :class:`~agilerl.protocols.EnvClientProtocol` for
    :class:`~agilerl.llm_envs.rollout_env.RolloutEnv`.

    Our rollout loop drives ``reset`` / ``step`` synchronously, but this client
    can be used asynchronously by sitting inside an async process e.g. Ray actor.

    For an external server whose env is exposed as an MCP tool rather than plain
    ``{"message": text}`` actions, pass ``mcp_tool``: the model's text is sent as
    ``call_tool(mcp_tool, {arg: text})`` and the tool result rendered back to text.
    This is only a transport shim for MCP-only servers; it does not constrain how many
    tools the environment itself supports.

    :param base_url: Root URL of the env server.
    :param headers: Optional HTTP headers (e.g. auth) sent on every request.
    :param timeout_s: Per-request timeout in seconds. ``None`` (the default) leaves
        requests unbounded; the value is supplied from the run manifest.
    :param mcp_tool: Optional MCP transport adapter for external servers that expect
        ``call_tool`` actions. When set, each step is sent as
        ``call_tool(mcp_tool, {arg: text})``; when ``None``, actions are sent as
        ``{"message": text}``. Multi-tool behavior remains environment-driven via the
        env's advertised ``tools`` schemas.
    :param arg: MCP argument name carrying the text (default ``"message"``).
    :param instruction: Prompt returned from reset when the env's reset obs is empty.
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
        mcp_tool: str | None = None,
        arg: str = "message",
        instruction: str = "",
    ) -> None:
        """Build a client for the OpenEnv server at ``base_url``."""
        if not base_url:
            msg = "OpenEnvClient requires a base_url"
            raise ValueError(msg)
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers or {}, timeout=timeout_s
        )
        self._mcp_tool = mcp_tool
        self._arg = arg
        self._instruction = instruction
        self._evaluation_mode = False
        self._state: dict[str, Any] | None = None
        self._mcp_tools: list[Any] | None = None

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Reset the env and return ``(prompt, info)``."""
        payload: dict[str, Any] = {}
        if seed is not None and int(seed) >= 0:
            payload["seed"] = int(seed)
        if row_index is not None:
            payload["row_index"] = int(row_index)
        if self._evaluation_mode:
            payload["evaluation"] = True
        data = self._post("/reset", payload)
        return (_observation_text(data.get("observation")) or self._instruction), {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Forward one action (model text) to the env and return the Gym 5-tuple."""
        text = action if isinstance(action, str) else str(action)
        if self._mcp_tool:
            act: dict[str, Any] = {
                "type": "call_tool",
                "tool_name": self._mcp_tool,
                "arguments": {self._arg: text},
            }
        else:
            act = {"message": text}
        data = self._post("/step", {"action": act})
        obs = data.get("observation")
        reward = data.get("reward")
        done = bool(data.get("done", False))
        # Our servers round-trip ``truncated`` inside the observation
        # (:class:`TextObservation`); servers that don't send it report every end
        # as a termination.
        truncated = (
            bool(obs.get("truncated", False)) if isinstance(obs, dict) else False
        )
        return (
            _observation_text(obs),
            float(reward) if reward is not None else 0.0,
            done and not truncated,
            truncated,
            {},
        )

    def close(self) -> None:
        """Best-effort ``/close``, then release the HTTP connection pool."""
        with contextlib.suppress(Exception):
            self._post("/close", {})
        self._http.close()

    @contextlib.contextmanager
    def eval_mode(self) -> Iterator[None]:
        """Flag resets for the env's held-out split within the block."""
        previous = self._evaluation_mode
        self._evaluation_mode = True
        try:
            yield
        finally:
            self._evaluation_mode = previous

    @property
    def dataset_size(self) -> int:
        """Dataset rows the env serves, from ``/state`` (``0`` if not dataset-backed)."""
        return int(self._fetch_state().get("dataset_size", 0) or 0)

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the env advertises (``/state``, then MCP ``tools/list``).

        ``/state`` is our servers' channel; the MCP fallback covers external
        OpenEnv servers that expose their tools only over ``/mcp``.
        """
        tools = self._fetch_state().get("tools") or []
        if tools:
            return tools
        return self._fetch_mcp_tools()

    def _fetch_state(self) -> dict[str, Any]:
        """Fetch + cache the env's ``/state`` (dataset size + tool schemas).

        This metadata changes training semantics — ``dataset_size`` drives batch
        row pinning and ``tools`` reach the chat template — so transient
        failures are retried and then raised, never silently treated as "no
        dataset, no tools". A 404/405 means the server registers no ``/state``
        route (e.g. a production-mode OpenEnv server) and caches as empty.
        """
        if self._state is not None:
            return self._state
        last_error: Exception | None = None
        for backoff_s in (0.5, 1.0, 2.0, None):
            try:
                response = self._http.get("/state")
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code in (404, 405):
                    self._state = {}
                    return self._state
                try:
                    response.raise_for_status()
                    body = response.json()
                except (httpx.HTTPStatusError, ValueError) as exc:
                    last_error = exc
                else:
                    self._state = body if isinstance(body, dict) else {}
                    return self._state
            if backoff_s is not None:
                time.sleep(backoff_s)
        msg = (
            "could not fetch /state from the OpenEnv server at "
            f"{self._http.base_url} — dataset size and tool schemas drive "
            "training, so this is fatal rather than silently empty. Is the "
            "server up and running in simulation mode?"
        )
        raise RuntimeError(msg) from last_error

    def _fetch_mcp_tools(self) -> list[Any]:
        """Tool schemas from the server's MCP channel (``tools/list`` on ``/mcp``).

        Best-effort: a server without an MCP channel simply advertises no tools,
        so failures here are not errors (unlike ``/state``, which is strict).
        """
        if self._mcp_tools is not None:
            return self._mcp_tools
        self._mcp_tools = []
        with contextlib.suppress(Exception):
            response = self._http.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict):
                tools = (body.get("result") or {}).get("tools")
                if isinstance(tools, list):
                    self._mcp_tools = tools
        return self._mcp_tools

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the server and decode the object response."""
        response = self._http.post(path, json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            msg = f"OpenEnv {path} returned a non-object payload: {type(body)!r}"
            raise TypeError(msg)
        return body


class LocalEnvClient:
    """In-process backend for a local env — the no-HTTP sibling of :class:`OpenEnvClient`.

    Implements :class:`~agilerl.protocols.EnvClientProtocol` for
    :class:`~agilerl.llm_envs.rollout_env.RolloutEnv`.

    Calls a local env's ``reset`` / ``step`` **directly** (no server, no
    socket), exposing the same surface a :class:`RolloutEnv` consumes (``reset`` /
    ``step`` / ``close`` / ``dataset_size`` / ``tools`` / ``eval_mode``). Use it (via
    :meth:`RolloutEnv.local` / :meth:`RolloutEnv.from_spec`) when the env lives in the
    same process — e.g. inside a Ray actor — so there is no pointless loopback HTTP;
    :class:`OpenEnvClient` is the sibling for an env reached over a URL. The prefix /
    suffix folding and reset-arg matching that :class:`OpenEnvWrapper` does server-side
    happen here instead, so both backends hand ``RolloutEnv`` the same shape.

    :param env: The local env: ``reset(seed=None[, row_index, evaluation]) ->
        (prompt, info)`` and ``step(text) -> (obs, reward, terminated, truncated, info)``.
    :param instruction: Prompt returned from reset when the env's reset obs is empty.
    """

    def __init__(self, env: TextEnvProtocol, *, instruction: str = "") -> None:
        """Wrap a local ``env`` as an in-process backend."""
        self._env = env
        self._instruction = instruction
        self._reset_params = set(inspect.signature(env.reset).parameters)
        self._evaluation_mode = False

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Reset the local env and return ``(prompt, info)`` (prefix/suffix folded in)."""
        call: dict[str, Any] = {}
        if seed is not None and "seed" in self._reset_params:
            call["seed"] = seed
        if row_index is not None and "row_index" in self._reset_params:
            call["row_index"] = row_index
        if self._evaluation_mode and "evaluation" in self._reset_params:
            call["evaluation"] = True
        prompt, info = _normalize_reset(self._env.reset(**call))
        return (_fold(prompt, info) or self._instruction), {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Step the local env with the action text and return the Gym 5-tuple."""
        text = action if isinstance(action, str) else str(action)
        prompt, reward, terminated, truncated, info = _normalize_step(
            self._env.step(text)
        )
        return (
            _fold(prompt, info),
            float(reward) if reward is not None else 0.0,
            terminated,
            truncated,
            {},
        )

    def close(self) -> None:
        """Close the wrapped env when it supports it (best-effort)."""
        closer = getattr(self._env, "close", None)
        if callable(closer):
            with contextlib.suppress(Exception):
                closer()

    @contextlib.contextmanager
    def eval_mode(self) -> Iterator[None]:
        """Flag resets for the env's held-out split within the block."""
        previous = self._evaluation_mode
        self._evaluation_mode = True
        try:
            yield
        finally:
            self._evaluation_mode = previous

    @property
    def dataset_size(self) -> int:
        """Dataset rows the env serves (``0`` if not dataset-backed)."""
        return int(getattr(self._env, "dataset_size", 0) or 0)

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the env advertises (empty when none)."""
        return list(getattr(self._env, "tools", None) or [])


class ServedEnvClient:
    """Backend that hosts a local env on its own :class:`OpenEnvServer` and owns both halves.

    It holds both the :class:`OpenEnvServer` and the :class:`OpenEnvClient` so
    ``RolloutEnv`` sees one backend with a single ``close`` that tears both down
    together.

    Any backend that owns transport infrastructure follows this shape: a future
    WebSocket-session backend would likewise hold one session on a shared
    server (letting one server host many concurrent rollouts) and release it in
    ``close`` — with ``RolloutEnv`` unchanged.

    :param env: The local env to host (plain-text ``reset`` / ``step``).
    :param host: Interface the server binds (default loopback).
    :param port: Server TCP port; ``0`` lets the OS pick one.
    :param env_name: Name advertised in the env's OpenEnv metadata; defaults to the
        env's class name.
    :param headers: Optional HTTP headers sent on every client request.
    :param timeout_s: Per-request client timeout in seconds (``None`` = unbounded).
    :param mcp_tool: Optional MCP transport adapter (see :class:`OpenEnvClient`).
    :param arg: MCP argument name carrying the text (default ``"message"``).
    :param instruction: Prompt returned from reset when the env's reset obs is empty.
    """

    def __init__(
        self,
        env: TextEnvProtocol,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        env_name: str | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
        mcp_tool: str | None = None,
        arg: str = "message",
        instruction: str = "",
    ) -> None:
        """Host ``env`` on a fresh server and build the client driving it."""
        self._server = OpenEnvServer(
            env, host=host, port=port, env_name=env_name
        ).start()
        try:
            self._client = OpenEnvClient(
                self._server.base_url,
                headers=headers,
                timeout_s=timeout_s,
                mcp_tool=mcp_tool,
                arg=arg,
                instruction=instruction,
            )
        except Exception:
            self._server.stop()
            raise

    @property
    def base_url(self) -> str:
        """The URL the owned server is bound to."""
        return self._server.base_url

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Reset the served env and return ``(prompt, info)``."""
        return self._client.reset(seed=seed, row_index=row_index)

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Step the served env with the action text and return the Gym 5-tuple."""
        return self._client.step(action)

    def close(self) -> None:
        """Stop the owned server and release the client's connection pool.

        Idempotent. Stopping the server closes the hosted env directly, so the
        client's ``/close`` round-trip to our own server is skipped — it would be
        redundant and can stall teardown on a loaded process.
        """
        self._server.stop()
        self._client._http.close()

    @property
    def dataset_size(self) -> int:
        """Dataset rows the served env exposes (``0`` if not dataset-backed)."""
        return self._client.dataset_size

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the served env advertises (empty when none)."""
        return self._client.tools

    def eval_mode(self) -> Any:
        """Flag resets for the env's held-out split within the block."""
        return self._client.eval_mode()


def _observation_text(obs: Any) -> str:
    """Render an OpenEnv observation to prompt text.

    Handles our own observations (``{"prompt": ...}`` or a bare string) and a third-party
    typed / MCP observation (tool-result content blocks, or a ``text`` / ``message``
    field), so one client drives both our servers and external ones.
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

    * a **URL** (``http://`` / ``https://``) is already hosted -> ``(url, None)``.
    * otherwise ``spec`` is an entrypoint to an env class — ``"package.module:Class"``
      for an installed package (a gym / gem / prime-rl env) or
      ``"/path/to/file.py:Class"`` for one on disk. It is imported, built with
      ``env_config``, hosted locally via :class:`OpenEnvServer`, and returned as
      ``(server.base_url, server)``.
    """
    if is_url(spec):
        return spec, None
    if ":" not in spec:
        msg = (
            f"env spec {spec!r} is neither a URL nor a 'module:Class' / "
            "'path.py:Class' entrypoint"
        )
        raise ValueError(msg)
    env = _load_entrypoint(spec)(**(env_config or {}))
    server = OpenEnvServer(
        env, host=host, port=port, env_name=_name_from_spec(spec)
    ).start()
    return server.base_url, server


def load_env(spec: str, env_config: dict[str, Any] | None = None) -> TextEnvProtocol:
    """Load a ``module:Class`` / ``path.py:Class`` entrypoint and build the env (no hosting).

    The in-process counterpart to :func:`resolve_env` (which hosts the env on an
    :class:`OpenEnvServer`): returns the constructed env object so a caller can drive it
    directly — e.g. :meth:`RolloutEnv.local` / :meth:`RolloutEnv.from_spec` wrapping it in
    a :class:`LocalEnvClient`.
    """
    return _load_entrypoint(spec)(**(env_config or {}))


def is_url(spec: str) -> bool:
    """Whether ``spec`` is an HTTP(S) URL (already hosted) rather than an env to load."""
    return isinstance(spec, str) and spec.startswith(("http://", "https://"))


def _name_from_spec(spec: str) -> str:
    """A label for a hosted env spec: the trailing identifier of the entrypoint / path.

    ``"game:GuessTheNumber-v0"`` -> ``"GuessTheNumber-v0"``,
    ``"/path/to/file.py:Env"`` -> ``"Env"``, a bare name passes through unchanged.
    """
    tail = spec.rsplit(":", 1)[-1]
    return tail.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or spec


def _load_entrypoint(target: str) -> Callable[..., Any]:
    """Import ``module:Class`` (or ``/path/to/file.py:Class``) and return ``Class``."""
    module_part, sep, attr = target.rpartition(":")
    if not sep:
        msg = f"env entrypoint {target!r} must be 'module:Class'"
        raise ValueError(msg)
    if not attr:
        msg = f"env entrypoint {target!r} must be 'module:Class'"
        raise ValueError(msg)
    # Detect filesystem paths across platforms, including Windows-style paths
    # parsed on non-Windows hosts (e.g. CI, cross-platform tests).
    looks_like_path = (
        module_part.endswith(".py")
        or "/" in module_part
        or "\\" in module_part
        or os.sep in module_part
        or bool(re.match(r"^[A-Za-z]:[\\/]", module_part))
    )
    if looks_like_path:
        module = _module_from_path(module_part)
    else:
        module = importlib.import_module(module_part)
    return getattr(module, attr)


def _module_from_path(path: str) -> Any:
    """Import a Python module from a filesystem path (an env definition on disk)."""
    name = "agilerl_env_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load env module from path {path!r}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
