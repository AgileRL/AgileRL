"""OpenEnv-style HTTP env interface for ``RolloutEnv``: text in, text out.

One interface for every LLM-training env: whatever backs it — a prompt dataset,
plain Python functions, an imported gem / AxonRL env, a sandboxed VM — it is reached
the same way, over a small JSON-over-HTTP protocol:

* ``POST {url}/reset`` ``{"seed"?, "row_index"?, "evaluation"?}``
  -> ``{"observation": <str>, "info": {...}}``
* ``POST {url}/step``  ``{"action": <str>}``
  -> ``{"observation": <str>, "reward": <float>, "terminated": <bool>,
  "truncated": <bool>, "info": {...}}``
* ``POST {url}/close`` ``{}`` -> ``{}``
* ``POST {url}/info``  ``{}`` -> ``{"dataset_size": <int>, "tools": [<schema>, ...]}``

The action is the model's text and the observation is the next prompt text.
An env that wants tool calls parses them itself in ``step``
and advertises its tool schemas (plain OpenAI function schemas)
via ``/info`` so the client can render them into the prompt.

This module is both halves of the interface:

* :class:`OpenEnvClient` — the **client** ``RolloutEnv`` drives (built from a URL).
* :class:`OpenEnvServer` — the **host**: wraps any local env (the Gym / gem text
  contract) in a tiny ``http.server`` speaking this protocol, so a local env is
  reached by URL exactly like a remote one. :func:`local_transport` is its
  socket-free, in-process counterpart.
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
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Self

logger = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT_S = 30.0


class OpenEnvClient:
    """Client that drives an OpenEnv text env over HTTP (or an injected transport).

    ``reset`` / ``step`` speak plain text — the action is the model's text, the
    observation is the next prompt. ``dataset_size`` and ``tools`` are read from the
    env's ``/info`` route; ``row_index`` and the current :attr:`evaluation_mode` are
    sent on reset so a ``BatchRolloutEnv`` can pin a group to one prompt / select the
    held-out split.

    :param base_url: Root URL of the env server. ``None`` only with a ``transport``.
    :param headers: Optional HTTP headers (e.g. auth) sent on every request.
    :param timeout_s: Per-request timeout.
    :param transport: ``(path, payload) -> dict`` injection seam in place of real
        HTTP (so unit tests — and :func:`local_transport` — need no socket).
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
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
        self._info: dict[str, Any] | None = None  # cached /info (dataset_size + tools)

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Reset the env and return ``(prompt, info)``.

        ``row_index`` (for a dataset-backed env) and the current
        :attr:`evaluation_mode` are sent so the owning ``BatchRolloutEnv`` can pin a
        group to one row / select the held-out split; an env that picks its own task
        ignores them.
        """
        payload: dict[str, Any] = {}
        if seed is not None and int(seed) >= 0:
            payload["seed"] = int(seed)
        if row_index is not None:
            payload["row_index"] = int(row_index)
        if self._evaluation_mode:
            payload["evaluation"] = True
        data = self._post("/reset", payload)
        return _obs_text(data), (data.get("info") or {})

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Forward one action (model text) to the env and return the Gym 5-tuple."""
        text = action if isinstance(action, str) else str(action)
        data = self._post("/step", {"action": text})
        reward = data.get("reward")
        return (
            _obs_text(data),
            float(reward) if reward is not None else 0.0,
            bool(data.get("terminated", data.get("done", False))),
            bool(data.get("truncated", False)),
            data.get("info") or {},
        )

    def close(self) -> None:
        """Best-effort ``/close`` to release any server-side episode resources."""
        with contextlib.suppress(Exception):
            self._post("/close", {})

    @property
    def tools(self) -> list[Any]:
        """Tool schemas the env advertises via ``/info`` (empty when none)."""
        return self._fetch_info().get("tools") or []

    @property
    def dataset_size(self) -> int:
        """Dataset rows the env serves, from ``/info`` (``0`` if not dataset-backed)."""
        return int(self._fetch_info().get("dataset_size", 0) or 0)

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

    def _fetch_info(self) -> dict[str, Any]:
        """Lazily fetch + cache the env's ``/info`` (dataset size + tool schemas)."""
        if self._info is None:
            self._info = {}
            with contextlib.suppress(Exception):
                resp = self._post("/info", {})
                if isinstance(resp, dict):
                    self._info = resp
        return self._info

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` to ``path`` and return the decoded JSON object."""
        if self._transport is not None:
            return self._transport(path, payload)
        return _urllib_post(
            f"{self._base_url}{path}",
            payload,
            headers=self._headers,
            timeout_s=self._timeout_s,
        )


def _obs_text(data: dict[str, Any]) -> str:
    """Pull the observation text out of a response (non-strings render empty)."""
    obs = data.get("observation")
    return obs if isinstance(obs, str) else ""


def _urllib_post(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
) -> dict[str, Any]:
    """POST JSON via the stdlib (no third-party HTTP dependency) and decode the object."""
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


class OpenEnvServer:
    """Serve any local env over the OpenEnv HTTP API so a URL can reach it.

    Wraps an in-process env — anything with the Gym / gem text contract
    ``reset(seed=None) -> (obs_text, info)`` and
    ``step(action) -> (obs_text, reward, terminated, truncated, info)`` (e.g. an
    imported AxonRL / gem ``SudokuEnv``) — and exposes it as a small ``http.server``
    speaking the OpenEnv protocol, so :class:`OpenEnvClient` (and any OpenEnv client)
    can drive it over HTTP. Start it during training-loop setup and hand its
    :attr:`base_url` to ``RolloutEnv``. Stdlib only — no FastAPI/uvicorn.

    A dataset env may also expose ``dataset_size`` and accept ``row_index`` /
    ``evaluation`` on ``reset``; a tool env may expose a ``tools`` list of OpenAI
    function schemas — both surface through ``/info``.

    :param env: The local env to serve.
    :param host: Interface to bind (default loopback, i.e. trainer-local).
    :param port: TCP port; ``0`` (default) lets the OS pick a free one, read back
        from :attr:`base_url` after :meth:`start`.
    """

    def __init__(self, env: Any, *, host: str = "127.0.0.1", port: int = 0) -> None:
        """Build (but do not start) a server wrapping ``env``."""
        self._env = env
        self._host = host
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        """The ``http://host:port`` the server is bound to (after :meth:`start`)."""
        if self._httpd is None:
            msg = "OpenEnvServer is not running; call start() first"
            raise RuntimeError(msg)
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> Self:
        """Bind the socket and serve in a background daemon thread; returns ``self``."""
        if self._httpd is None:
            self._httpd = ThreadingHTTPServer(
                (self._host, self._port), _make_handler(self._env)
            )
            self._thread = threading.Thread(
                target=self._httpd.serve_forever,
                name="openenv-server",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the socket (idempotent)."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def serve(env: Any, *, host: str = "127.0.0.1", port: int = 0) -> OpenEnvServer:
    """Wrap ``env`` in an :class:`OpenEnvServer` and start it; returns the running server.

    Convenience for ``OpenEnvServer(env, ...).start()`` — typically called in
    training-loop setup, handing :attr:`OpenEnvServer.base_url` to ``RolloutEnv``.
    """
    return OpenEnvServer(env, host=host, port=port).start()


# --- env resolution: a spec -> a URL ---------------------------------------
# The env config names either a URL (already hosted, use it raw) or an env to load by
# import path (a ``module:Class`` / ``path.py:Class`` entrypoint — e.g. a gym / gem /
# prime-rl env you have installed or on disk; those ecosystems own their own name
# registries). A loaded env is hosted locally and we hit its URL, so the rest of the
# stack only ever sees a URL.


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
      ``env_config``, hosted locally, and returned as ``(server.base_url, server)``.

    Call this once in training-loop setup and give the URL to every ``RolloutEnv``
    (they share the one server). Keep the returned server to
    :meth:`OpenEnvServer.stop` it on shutdown — it is ``None`` for a URL, which you
    do not own.
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
    server = serve(env, host=host, port=port)
    return server.base_url, server


def _is_url(spec: str) -> bool:
    """Whether ``spec`` is an HTTP(S) URL (already hosted) rather than an env to load."""
    return isinstance(spec, str) and spec.startswith(("http://", "https://"))


def _load_entrypoint(target: str) -> Callable[..., Any]:
    """Import ``module:Attr`` (or ``/path/to/file.py:Attr``) and return ``Attr``."""
    module_part, _, attr = target.partition(":")
    if not attr:
        msg = f"env entrypoint {target!r} must be 'module:Attr'"
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


def local_transport(env: Any) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Return an in-process ``(path, payload) -> dict`` transport driving ``env``.

    The socket-free counterpart to :class:`OpenEnvServer`: it runs the very same
    :func:`_dispatch`, so a local env speaks the OpenEnv protocol with no HTTP cost.
    Pass it as ``RolloutEnv(..., transport=local_transport(env))`` (this is what
    :meth:`RolloutEnv.from_dataset` does) or to :class:`OpenEnvClient` directly.
    """

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        route = path.split("?", 1)[0].rstrip("/") or "/"
        return _dispatch(env, route, payload)

    return transport


def _normalize_reset(result: Any) -> tuple[Any, dict[str, Any]]:
    """Normalise an env ``reset`` return into ``(observation, info)``."""
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[0], (result[1] or {})
        if len(result) == 1:
            return result[0], {}
    return result, {}


def _normalize_step(result: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
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
    return obs, reward, bool(terminated), bool(truncated), (info or {})


def _call_reset(env: Any, payload: dict[str, Any]) -> Any:
    """Call ``env.reset`` passing only the args its signature accepts.

    ``seed`` is positional; ``row_index`` / ``evaluation`` (the dataset selectors)
    are forwarded only to envs whose ``reset`` declares them, so a plain env
    (``reset(seed=None)``, e.g. a gem env) is driven unchanged.
    """
    params = inspect.signature(env.reset).parameters
    kwargs = {
        name: payload[name]
        for name in ("row_index", "evaluation")
        if name in params and name in payload
    }
    seed = payload.get("seed")
    if seed is not None:
        return env.reset(seed, **kwargs)
    return env.reset(**kwargs)


def _dispatch(env: Any, route: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Map one OpenEnv route to the wrapped env's method and shape the response."""
    if route == "/reset":
        obs, info = _normalize_reset(_call_reset(env, payload))
        return {"observation": obs, "info": info}
    if route == "/step":
        obs, reward, terminated, truncated, info = _normalize_step(
            env.step(payload.get("action", ""))
        )
        return {
            "observation": obs,
            "reward": None if reward is None else float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated,
            "info": info,
        }
    if route == "/close":
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()
        return {}
    if route == "/info":
        return {
            "dataset_size": int(getattr(env, "dataset_size", 0) or 0),
            "tools": list(getattr(env, "tools", None) or []),
        }
    msg = f"unknown OpenEnv route {route!r}"
    raise ValueError(msg)


def _make_handler(env: Any) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to ``env`` (one per :class:`OpenEnvServer`)."""

    class _OpenEnvHandler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            """Silence the default per-request stderr access log."""

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except (ValueError, TypeError):
                payload = {}
            route = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                result = _dispatch(env, route, payload)
                status = 200
            except Exception as exc:
                result = {"error": str(exc)}
                status = 500
            body = json.dumps(result).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _OpenEnvHandler
