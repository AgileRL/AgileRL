"""The AgileRL <-> OpenEnv seam: host any local env as a server, hit it as a client.

Every LLM-training env is reached the same way — text in, text out, over OpenEnv's
HTTP protocol (https://github.com/meta-pytorch/OpenEnv). Three thin wrappers sit
between AgileRL and the ``openenv`` package; each exists for a concrete reason
OpenEnv's own classes don't cover for our use:

* :class:`OpenEnvWrapper` adapts an AgileRL local env (the ``reset`` / ``step`` text
  contract) to OpenEnv's typed ``Environment`` ABC — the glue that lets OpenEnv host
  our envs at all.
* :class:`OpenEnvServer` runs OpenEnv's app (``create_app``) on uvicorn *inside the
  training process* (a daemon thread on an ephemeral port, with ``start`` / ``stop``),
  because OpenEnv's own host story is a standalone blocking process (a Space, a
  container) — not something a trainer spins up and tears down per rollout.
* :class:`OpenEnvClient` is a small *synchronous* httpx client, because OpenEnv's own
  client is async + WebSocket-first; the rollout loop drives ``reset`` / ``step``
  synchronously, so a plain sync client hitting the (async) server is simpler than
  threading an event loop through the trainer.

A standard text contract — :class:`TextAction` (``message``) / :class:`TextObservation`
(``prompt``) — carries the model's text both ways, so there are no per-env codecs;
:func:`resolve_env` turns a URL or a ``module:Class`` entrypoint into a hosted env.
Because our server speaks the standard OpenEnv wire, OpenEnv's own async client can
drive it too (e.g. the async Ray rollout path).
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import httpx
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata
from openenv.core.env_server.types import Action, Observation, State

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Self

logger = logging.getLogger(__name__)


# --- standard text contract ------------------------------------------------
class TextAction(Action):
    """OpenEnv action carrying the policy's generated text."""

    message: str = ""


class TextObservation(Observation):
    """OpenEnv observation carrying the next prompt text.

    Inherits ``reward`` / ``done`` / ``metadata`` from ``Observation``. (The OpenEnv
    server round-trips ``prompt`` / ``reward`` / ``done`` but not ``metadata``, so any
    prefix/suffix is folded into ``prompt`` by :class:`OpenEnvWrapper`.)
    """

    prompt: str = ""


class OpenEnvWrapper(Environment):
    """Adapt an AgileRL local env to OpenEnv's typed ``Environment`` ABC.

    The required interop glue: an AgileRL env speaks a plain text contract, not
    OpenEnv's typed ``Action`` / ``Observation`` / ``State``, so this bridges the two —
    letting OpenEnv's server (or ours) host any local env unchanged. ``inner`` provides
    ``reset(seed=None[, row_index, evaluation]) -> (prompt, info)`` and
    ``step(action_text) -> (prompt, reward, terminated, truncated, info)``. The env's
    ``dataset_size`` / ``tools`` are surfaced on the OpenEnv ``state`` so a client can
    read them; ``info``'s ``prefix`` / ``suffix`` are folded into the prompt (OpenEnv
    does not round-trip observation metadata).

    :param inner: The local env to host.
    :param env_name: Name reported in the env's OpenEnv metadata; defaults to
        ``inner``'s class name (so a client sees the real env, not ``OpenEnvWrapper``).
    """

    def __init__(self, inner: Any, *, env_name: str | None = None) -> None:
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
        """Close the inner env when it supports it."""
        closer = getattr(self._inner, "close", None)
        if callable(closer):
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
    """Run OpenEnv's app on uvicorn *inside the training process*.

    Why this rather than OpenEnv's host helpers directly: OpenEnv builds the FastAPI
    app (``create_app``), but its hosting is meant to be a standalone, blocking process
    (a HF Space, a container, ``python -m openenv``). A trainer needs to start servers
    and stop them programmatically from its own process — one per concurrent rollout,
    all created once and reused for the whole run (not per episode), and torn down at
    the end — so this wraps the same ``create_app`` in a uvicorn **daemon thread**,
    binds an ephemeral port (``port=0``) read back from :attr:`base_url`, and exposes
    ``start`` / ``stop`` (and the context-manager protocol). The env is wrapped in :class:`OpenEnvWrapper`, so any
    OpenEnv client — :class:`OpenEnvClient`, OpenEnv's own async client, a HF Space
    consumer — reaches it by URL.

    :param env: The local env to serve.
    :param host: Interface to bind (default loopback).
    :param port: TCP port; ``0`` lets the OS pick one.
    :param env_name: Name advertised in the env's OpenEnv metadata / schema; defaults
        to the env's class name.
    """

    def __init__(
        self,
        env: Any,
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
        while not getattr(self._server, "started", False):
            if time.monotonic() > deadline:
                msg = "OpenEnvServer failed to start within 30s"
                raise RuntimeError(msg)
            time.sleep(0.02)
        self._bound_port = self._server.servers[0].sockets[0].getsockname()[1]
        return self

    def stop(self) -> None:
        """Stop serving and release the socket (idempotent)."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._server = None
        self._bound_port = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class OpenEnvClient:
    """Synchronous httpx client for an OpenEnv env server (text in, text out).

    Why this rather than OpenEnv's own client: OpenEnv ships an async, WebSocket-first
    ``EnvClient``. The rollout loop drives ``reset`` / ``step`` synchronously (token by
    token), so a plain **sync** httpx client hitting the (async) FastAPI server is
    simpler than threading an event loop through the trainer — and since our server
    speaks the standard OpenEnv wire, OpenEnv's own async client can still drive it
    (e.g. the async Ray path) when that is wanted.

    Drives any env hosted as an OpenEnv server — our own :class:`OpenEnvServer` or a
    third-party Space — over its REST wire via a single :class:`httpx.Client`.
    ``reset`` / ``step`` speak the standard text contract (the action is the model's
    text, the observation is the next prompt); ``dataset_size`` / ``tools`` come from
    ``/state``; ``row_index`` / ``evaluation`` ride on reset so a ``BatchRolloutEnv``
    can pin a group to one prompt / select the held-out split.

    For an external server whose env exposes an MCP tool rather than the plain text
    contract, pass ``mcp_tool``: the model's text is sent as
    ``call_tool(mcp_tool, {arg: text})`` and the tool result rendered back to text.

    :param base_url: Root URL of the env server.
    :param headers: Optional HTTP headers (e.g. auth) sent on every request.
    :param timeout_s: Per-request timeout in seconds. ``None`` (the default) leaves
        requests unbounded; the value is supplied from the run manifest.
    :param mcp_tool: When set, send the model's text as an MCP ``call_tool`` to this
        tool (for external MCP servers); when ``None``, send ``{"message": text}``.
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
        reward = data.get("reward")
        return (
            _observation_text(data.get("observation")),
            float(reward) if reward is not None else 0.0,
            bool(data.get("done", False)),
            False,
            {},
        )

    def close(self) -> None:
        """Best-effort ``/close``, then release the HTTP connection pool."""
        with contextlib.suppress(Exception):
            self._post("/close", {})
        self._http.close()

    @property
    def dataset_size(self) -> int:
        """Dataset rows the env serves, from ``/state`` (``0`` if not dataset-backed)."""
        return int(self._fetch_state().get("dataset_size", 0) or 0)

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the env advertises via ``/state`` (empty when none)."""
        return self._fetch_state().get("tools") or []

    @property
    def evaluation_mode(self) -> bool:
        """Whether reset requests are currently flagged for the held-out split."""
        return self._evaluation_mode

    @evaluation_mode.setter
    def evaluation_mode(self, value: bool) -> None:
        self._evaluation_mode = bool(value)

    @contextlib.contextmanager
    def eval_mode(self) -> Iterator[None]:
        """Flag reset requests for the env's held-out split within the block."""
        previous = self._evaluation_mode
        self._evaluation_mode = True
        try:
            yield
        finally:
            self._evaluation_mode = previous

    def _fetch_state(self) -> dict[str, Any]:
        """Lazily fetch + cache the env's ``/state`` (dataset size + tool schemas)."""
        if self._state is None:
            self._state = {}
            with contextlib.suppress(Exception):
                response = self._http.get("/state")
                response.raise_for_status()
                body = response.json()
                if isinstance(body, dict):
                    self._state = body
        return self._state

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the server and decode the object response."""
        response = self._http.post(path, json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            msg = f"OpenEnv {path} returned a non-object payload: {type(body)!r}"
            raise TypeError(msg)
        return body


def _observation_text(obs: Any) -> str:
    """Render an OpenEnv observation to prompt text.

    Handles our text contract (``{"prompt": ...}`` or a bare string) and a third-party
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
    if _is_url(spec):
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


def _is_url(spec: str) -> bool:
    """Whether ``spec`` is an HTTP(S) URL (already hosted) rather than an env to load."""
    return isinstance(spec, str) and spec.startswith(("http://", "https://"))


def _name_from_spec(spec: str) -> str:
    """A label for a hosted env spec: the trailing identifier of the entrypoint / path.

    ``"game:GuessTheNumber-v0"`` -> ``"GuessTheNumber-v0"``,
    ``"/path/to/file.py:Env"`` -> ``"Env"``, a bare name passes through unchanged.
    """
    return spec.rsplit(":", 1)[-1].rsplit("/", 1)[-1] or spec


def _load_entrypoint(target: str) -> Callable[..., Any]:
    """Import ``module:Class`` (or ``/path/to/file.py:Class``) and return ``Class``."""
    module_part, _, attr = target.partition(":")
    if not attr:
        msg = f"env entrypoint {target!r} must be 'module:Class'"
        raise ValueError(msg)
    if module_part.endswith(".py") or os.sep in module_part:
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
