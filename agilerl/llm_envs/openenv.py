"""OpenEnv-backed env interface for ``RolloutEnv`` (text in, text out).

Every LLM-training env is hosted + reached through the OpenEnv server and protocol
(https://github.com/meta-pytorch/OpenEnv): an env is wrapped in a
:class:`OpenEnvWrapper` (an OpenEnv ``Environment``), served by OpenEnv's
``HTTPEnvServer`` (:func:`serve`), and ``RolloutEnv`` drives it over the OpenEnv wire
(``POST /reset`` / ``/step``). A standard text contract — :class:`TextAction`
(``message``) and :class:`TextObservation` (``prompt``) — carries the model's text
both ways, so there are no per-env codecs. :func:`local_transport` drives the very
same ``OpenEnvWrapper`` in-process (socket-free) for the common local case, and
:func:`resolve_env` turns a URL or a ``module:Class`` entrypoint into a hosted env.

This module is the AgileRL <-> OpenEnv seam: ``openenv`` (the ``[llm]`` extra) owns
the server/protocol; we own the text contract + the gym/gem wrapper + the sync client.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
import threading
import time
import urllib.request
from typing import TYPE_CHECKING, Any

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


# --- the universal wrapper: any local env -> an OpenEnv Environment ---------
class OpenEnvWrapper(Environment):
    """Wrap any local env (the gym / gem text contract) as an OpenEnv ``Environment``.

    ``inner`` provides ``reset(seed=None[, row_index, evaluation]) -> (prompt, info)``
    and ``step(action_text) -> (prompt, reward, terminated, truncated, info)``. The
    env's ``dataset_size`` / ``tools`` are surfaced on the OpenEnv ``state`` so a
    client can read them; ``info``'s ``prefix`` / ``suffix`` are folded into the
    prompt (OpenEnv does not round-trip observation metadata).

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


def _obs_to_wire(obs: TextObservation) -> dict[str, Any]:
    """Shape a ``TextObservation`` like the OpenEnv server's reset/step response."""
    return {
        "observation": {"prompt": obs.prompt},
        "reward": obs.reward,
        "done": obs.done,
    }


# --- serve: host a local env via OpenEnv's HTTPEnvServer --------------------
class OpenEnvServer:
    """Host a local env over HTTP via OpenEnv's ``HTTPEnvServer`` (uvicorn in a thread).

    Wraps ``env`` in a :class:`OpenEnvWrapper` and serves it, so any OpenEnv client —
    ``RolloutEnv``, OpenEnv's own async client, a HF Space consumer — reaches it by
    URL. Start it in training-loop setup; ``port=0`` (default) binds a free port read
    back from :attr:`base_url`.

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


def serve(
    env: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    env_name: str | None = None,
) -> OpenEnvServer:
    """Wrap ``env`` in an :class:`OpenEnvServer` and start it; returns the running server."""
    return OpenEnvServer(env, host=host, port=port, env_name=env_name).start()


def local_transport(env: Any) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Return an in-process ``(path, payload) -> dict`` transport driving ``env``.

    The socket-free counterpart to :class:`OpenEnvServer`: it runs the same
    :class:`OpenEnvWrapper` the server would, shaping responses like the OpenEnv wire,
    so a local env is driven with no HTTP cost. Pass it as
    ``RolloutEnv(..., transport=local_transport(env))`` (this is what
    :meth:`RolloutEnv.from_dataset` does) or to :class:`OpenEnvClient` directly.
    """
    gym = OpenEnvWrapper(env)

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        route = path.split("?", 1)[0].rstrip("/") or "/"
        payload = payload or {}
        if route == "/reset":
            return _obs_to_wire(
                gym.reset(
                    seed=payload.get("seed"),
                    row_index=payload.get("row_index"),
                    evaluation=payload.get("evaluation"),
                )
            )
        if route == "/step":
            action = TextAction(**(payload.get("action") or {}))
            return _obs_to_wire(gym.step(action))
        if route == "/state":
            return gym.state.model_dump()
        if route == "/close":
            gym.close()
            return {}
        msg = f"unknown OpenEnv route {route!r}"
        raise ValueError(msg)

    return transport


# --- client: drive an OpenEnv env over the wire (REST or in-process) --------
class OpenEnvClient:
    """Sync client that drives an OpenEnv env over its REST wire (or a transport).

    ``reset`` / ``step`` speak the standard text contract: the action is the model's
    text (``{"message": ...}``) and the observation is the next prompt. ``dataset_size``
    and ``tools`` come from the env's ``/state``; ``row_index`` / ``evaluation`` are
    sent on reset so a ``BatchRolloutEnv`` can pin a group to one prompt / select the
    held-out split.

    :param base_url: Root URL of the env server. ``None`` only with a ``transport``.
    :param headers: Optional HTTP headers sent on every request.
    :param timeout_s: Per-request timeout in seconds. ``None`` (the default) leaves
        requests unbounded; the value is supplied from the run manifest.
    :param transport: ``(path, payload) -> dict`` injection seam in place of real HTTP
        (so unit tests — and :func:`local_transport` — need no socket).
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Build a client for the env server at ``base_url`` (or an injected transport)."""
        if not base_url and transport is None:
            msg = "OpenEnvClient requires a base_url (or an injected transport)"
            raise ValueError(msg)
        self._base_url = (base_url or "").rstrip("/")
        self._headers = headers
        self._timeout_s = timeout_s
        self._transport = transport
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
        data = self._request("/reset", payload)
        return _prompt(data), {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Forward one action (model text) to the env and return the Gym 5-tuple."""
        text = action if isinstance(action, str) else str(action)
        data = self._request("/step", {"action": {"message": text}})
        reward = data.get("reward")
        return (
            _prompt(data),
            float(reward) if reward is not None else 0.0,
            bool(data.get("done", False)),
            False,
            {},
        )

    def close(self) -> None:
        """Best-effort ``/close`` (routes to the env's close over a transport)."""
        with contextlib.suppress(Exception):
            self._request("/close", {})

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
                resp = self._request("/state", {}, method="GET")
                if isinstance(resp, dict):
                    self._state = resp
        return self._state

    def _request(
        self, path: str, payload: dict[str, Any], *, method: str = "POST"
    ) -> dict[str, Any]:
        """Dispatch over the injected transport or real HTTP."""
        if self._transport is not None:
            return self._transport(path, payload)
        if method == "GET":
            return _urllib_get(
                f"{self._base_url}{path}",
                headers=self._headers,
                timeout_s=self._timeout_s,
            )
        return _urllib_post(
            f"{self._base_url}{path}",
            payload,
            headers=self._headers,
            timeout_s=self._timeout_s,
        )


def _prompt(data: dict[str, Any]) -> str:
    """Pull the prompt text out of an OpenEnv response."""
    obs = data.get("observation")
    if isinstance(obs, dict):
        prompt = obs.get("prompt")
        if isinstance(prompt, str):
            return prompt
    return obs if isinstance(obs, str) else ""


def _urllib_post(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """POST JSON via the stdlib and decode the object."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("content-type", "application/json")
    request.add_header("accept", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        msg = f"OpenEnv {url} returned a non-object payload: {type(decoded)!r}"
        raise TypeError(msg)
    return decoded


def _urllib_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """GET JSON via the stdlib and decode the object."""
    request = urllib.request.Request(url, method="GET")
    request.add_header("accept", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    return decoded if isinstance(decoded, dict) else {}


# --- adapter: drive a real *external* OpenEnv server as a local env ---------
class OpenEnvHTTPEnv:
    """A local env that proxies a real, *external* OpenEnv REST server.

    Our envs speak the :class:`TextAction` / :class:`TextObservation` text contract,
    but a third-party OpenEnv env (e.g. the echo HF Space) has its own typed schema.
    This bridges one such env to the text contract, so it is driven by ``RolloutEnv``
    via :func:`local_transport` / :func:`serve` like any local env. ``mcp_tool`` handles
    MCP tool envs (the echo Space): the model's text is sent as
    ``call_tool(mcp_tool, {arg: text})`` and the tool result rendered back to text.

    For full OpenEnv compatibility (WebSocket sessions, container Spaces, production
    MCP) use the ``openenv`` package's own client instead.

    :param base_url: Root URL of the external OpenEnv server.
    :param mcp_tool: When set, send the text as an MCP ``call_tool`` to this tool;
        when ``None``, send it as ``{"message": text}``.
    :param arg: MCP argument name carrying the text (default ``"message"``).
    :param instruction: Prompt returned from reset when the env's reset obs is empty.
    :param headers: Optional HTTP headers (e.g. auth) sent on every request.
    :param timeout_s: Per-request timeout in seconds. ``None`` (the default) leaves
        requests unbounded; the value is supplied from the run manifest.
    """

    def __init__(
        self,
        base_url: str,
        *,
        mcp_tool: str | None = None,
        arg: str = "message",
        instruction: str = "",
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Build an adapter for the external OpenEnv server at ``base_url``."""
        self._base_url = base_url.rstrip("/")
        self._mcp_tool = mcp_tool
        self._arg = arg
        self._instruction = instruction
        self._headers = headers
        self._timeout_s = timeout_s

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        """Reset the external env and return ``(prompt, info)``."""
        payload: dict[str, Any] = {}
        if seed is not None and int(seed) >= 0:
            payload["seed"] = int(seed)
        data = _urllib_post(
            f"{self._base_url}/reset",
            payload,
            headers=self._headers,
            timeout_s=self._timeout_s,
        )
        return (_render_external(data.get("observation")) or self._instruction), {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Forward one action (model text) to the external env; return the Gym 5-tuple."""
        text = action if isinstance(action, str) else str(action)
        if self._mcp_tool:
            act: dict[str, Any] = {
                "type": "call_tool",
                "tool_name": self._mcp_tool,
                "arguments": {self._arg: text},
            }
        else:
            act = {"message": text}
        data = _urllib_post(
            f"{self._base_url}/step",
            {"action": act},
            headers=self._headers,
            timeout_s=self._timeout_s,
        )
        reward = data.get("reward")
        return (
            _render_external(data.get("observation")),
            float(reward) if reward is not None else 0.0,
            bool(data.get("done", False)),
            False,
            {},
        )

    def close(self) -> None:
        """No-op: the OpenEnv REST API holds no per-client session to close."""


def _render_external(obs: Any) -> str:
    """Render an external OpenEnv observation (MCP tool result or a text field) to text."""
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


# --- env resolution: a spec -> a URL ---------------------------------------
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
    server = serve(env, host=host, port=port, env_name=_name_from_spec(spec))
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
