"""The OpenEnv API: reach any text env the same way — over a URL or in-process.

Every LLM-training env is driven through the same two calls — ``reset()`` returns a
prompt, ``step(text)`` returns the next prompt, a reward, and done flags — and a
``RolloutEnv`` reaches it through a **backend**: an :class:`OpenEnvSessionClient` when
the env is hosted at a URL (a remote Space, or a local :class:`OpenEnvServer`), or a
:class:`LocalEnvClient` when the env runs in the same process (no HTTP).

The pieces here fill the gaps OpenEnv's own classes leave:

* :class:`OpenEnvWrapper` — presents a plain-text env as OpenEnv's typed
  ``Environment`` so OpenEnv's server can host it.
* :class:`OpenEnvServer` — runs OpenEnv's app on uvicorn *in-process* (a daemon thread
  on an ephemeral port, ``start`` / ``stop``); OpenEnv's own hosting is a standalone
  blocking process. Hosts a single shared env, or — with ``make_env`` +
  ``max_concurrent_envs`` — a fresh env per WebSocket session.
* :class:`OpenEnvSessionClient` — drives an env at a URL over an OpenEnv ``/ws``
  session (wrapping OpenEnv's own ``SyncEnvClient``); one session per instance, so a
  single URL serves as many concurrent, isolated episodes as the server allows.
* :class:`LocalEnvClient` — the in-process sibling: same surface, direct calls, no
  socket.
* :class:`ServedEnvClient` — hosts a local env on an :class:`OpenEnvServer`, drives it
  through an :class:`OpenEnvSessionClient`, and owns both, so one ``close`` tears
  everything down.

:func:`resolve_env` turns a URL or a ``module:Class`` entrypoint into a hosted ``(url,
server)``; :func:`load_env` just builds an entrypoint env (no hosting) for the
in-process path.

Why WebSocket sessions: OpenEnv's REST ``/reset`` / ``/step`` / ``/state`` are
registered only in **simulation mode** and carry no session state, so one shared env
serves one episode at a time. The ``/ws`` endpoint is served in both simulation and
production mode and gives each connection its own env instance, so one server (one URL)
hosts a whole rollout group as concurrent, isolated episodes — and the session client
is the only backend that can drive a production OpenEnv deployment.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from agilerl import HAS_LLM_DEPENDENCIES

if HAS_LLM_DEPENDENCIES:
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
    :param owns_inner: When ``True`` (the per-session hosting path), ``close``
        closes ``inner``; when ``False`` (a single shared env driven over REST),
        ``close`` is a no-op so the shared env outlives each per-request adapter.
    """

    # OpenEnv gates ``max_concurrent_envs > 1`` on this flag. In the per-session
    # path the server calls the env factory once per WebSocket session, so each
    # session gets its own wrapper around its own fresh ``inner`` env and distinct
    # sessions never share env state — the isolation concurrent sessions require.
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
        # A ``**kwargs`` reset accepts any keyword, so every forwardable name
        # counts as supported (a delegating proxy must not lose seed/row pinning).
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
        """Close ``inner`` when this wrapper owns it, else no-op.

        The shared-inner REST path keeps this a no-op so one env survives across
        the per-request adapters; the per-session path owns its fresh env and
        releases it when the session ends.
        """
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

    Pass ``env`` to host a single shared env: one session's episode at a time
    (this is what :class:`ServedEnvClient` uses for a per-rollout server). Pass
    ``make_env`` with ``max_concurrent_envs`` to host a *fresh env per WebSocket
    session*: OpenEnv calls the factory once per ``/ws`` connection, so one server
    (one URL) serves that many concurrent, isolated episodes (drive each with its
    own :class:`OpenEnvSessionClient`). This is what lets a whole
    :class:`~agilerl.llm_envs.rollout_env.BatchRolloutEnv` group run against a
    single hosted env service.

    :param env: A single local env, shared across a server that hosts one session.
    :param make_env: A zero-arg factory building a fresh env per session; supply
        with ``max_concurrent_envs`` for the concurrent-session path. Exactly one
        of ``env`` / ``make_env`` must be given.
    :param host: Interface to bind (default loopback).
    :param port: TCP port; ``0`` lets the OS pick one.
    :param env_name: Name advertised in the env's OpenEnv metadata / schema; defaults
        to the env's class name.
    :param max_concurrent_envs: Maximum live WebSocket sessions (env instances).
        ``None`` leaves OpenEnv's default of one; set it to at least the rollout
        group's size when hosting with ``make_env``.
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
            # OpenEnv calls this per REST request and once per WebSocket session.
            # ``make_env`` gives every session its own env (owned, so it is closed
            # with the session); the shared-``env`` path reuses one across requests.
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


class LocalEnvClient:
    """In-process backend for a local env — the no-HTTP sibling of :class:`OpenEnvSessionClient`.

    Implements :class:`~agilerl.protocols.EnvClientProtocol` for
    :class:`~agilerl.llm_envs.rollout_env.RolloutEnv`.

    Drives the env through the same :class:`OpenEnvWrapper` the server path
    hosts, so reset-arg matching, reset/step normalisation and prefix/suffix
    folding live in one adapter and the in-process and HTTP transports behave
    identically on the same env — just without a server or socket. Use it (via
    :meth:`RolloutEnv.local` / :meth:`RolloutEnv.from_spec`) when the env lives
    in the same process — e.g. inside a Ray actor; :class:`OpenEnvSessionClient`
    is the sibling for an env reached over a URL.

    :param env: The local env: ``reset(seed=None[, row_index, evaluation]) ->
        (prompt, info)`` and ``step(text) -> (obs, reward, terminated, truncated, info)``.
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
        """Reset the local env and return ``(prompt, info)`` (prefix/suffix folded in)."""
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
            {},
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
    """Session backend for an OpenEnv server — one WebSocket session per instance.

    Implements :class:`~agilerl.protocols.EnvClientProtocol` on top of OpenEnv's
    own :class:`~openenv.core.GenericEnvClient` (via its synchronous wrapper).
    Each instance opens its own ``/ws`` session, and the server builds a fresh env
    per session, so a single URL hosts as many concurrent, isolated episodes as
    its ``max_concurrent_envs`` allows — the backend
    :class:`~agilerl.llm_envs.rollout_env.BatchRolloutEnv` needs to run a whole
    group against one hosted env service. It is also the only backend that reaches
    a **production-mode** OpenEnv server, which exposes ``/ws`` but not the REST
    ``/reset`` / ``/step`` / ``/state`` routes.

    Actions are plain ``{"message": text}`` (or ``call_tool`` when ``mcp_tool`` is
    set) and observations are decoded by the shared :func:`_observation_text`, the
    same contract the in-process :class:`LocalEnvClient` uses.

    :param base_url: Root URL (``http(s)://`` or ``ws(s)://``) of the env server.
    :param timeout_s: Per-message timeout in seconds. ``None`` uses OpenEnv's
        default; the value is supplied from the run manifest.
    :param connect_timeout_s: Timeout for establishing the session.
    :param mcp_tool: Optional MCP transport adapter: when set, the model's text is
        sent as ``call_tool(mcp_tool, {arg: text})`` for MCP-only servers.
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
            message_timeout_s=timeout_s if timeout_s is not None else 60.0,
        ).sync()
        self._sync.connect()
        self._mcp_tool = mcp_tool
        self._arg = arg
        self._instruction = instruction
        self._evaluation_mode = False
        self._state: dict[str, Any] | None = None

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

        ``seed`` / ``row_index`` travel to the server's env reset so a rollout
        group (one seed per batch row) resets every session to the same prompt.
        """
        kwargs: dict[str, Any] = {}
        if seed is not None and int(seed) >= 0:
            kwargs["seed"] = int(seed)
        if row_index is not None:
            kwargs["row_index"] = int(row_index)
        if self._evaluation_mode:
            kwargs["evaluation"] = True
        result = self._sync.reset(**kwargs)
        return (_observation_text(result.observation) or self._instruction), {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Send one action (model text) over the session and return the Gym 5-tuple."""
        text = action if isinstance(action, str) else str(action)
        if self._mcp_tool:
            act: dict[str, Any] = {
                "type": "call_tool",
                "tool_name": self._mcp_tool,
                "arguments": {self._arg: text},
            }
        else:
            act = {"message": text}
        result = self._sync.step(act)
        obs = result.observation
        reward = result.reward
        # Our servers round-trip ``truncated`` inside the observation
        # (:class:`TextObservation`); a server that doesn't send it reports every
        # end as a plain termination.
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

        Best-effort: a server that does not answer ``state`` over the session
        advertises no dataset and no tools rather than stalling the rollout.
        """
        if self._state is None:
            try:
                state = self._sync.state()
            except Exception:
                logging.getLogger(__name__).warning(
                    "OpenEnvSessionClient could not read state; assuming no "
                    "dataset and no tools.",
                    exc_info=True,
                )
                state = {}
            self._state = state if isinstance(state, dict) else {}
        return self._state


class ServedEnvClient:
    """Backend that hosts a local env on its own :class:`OpenEnvServer` and owns both halves.

    It holds both the :class:`OpenEnvServer` and the :class:`OpenEnvSessionClient`
    driving it, so ``RolloutEnv`` sees one backend with a single ``close`` that
    tears both down together.

    This is the in-process rehearsal of the ``env_url`` transport: the owned
    server hosts the env exactly as a remote deployment would, and the client
    reaches it over the same WebSocket session. Each served client owns one
    server + one session; a :class:`~agilerl.llm_envs.rollout_env.BatchRolloutEnv`
    uses one per concurrent rollout.

    :param env: The local env to host (plain-text ``reset`` / ``step``).
    :param host: Interface the server binds (default loopback).
    :param port: Server TCP port; ``0`` lets the OS pick one.
    :param env_name: Name advertised in the env's OpenEnv metadata; defaults to the
        env's class name.
    :param timeout_s: Per-message client timeout in seconds, defaults to 300 —
        the server is our own loopback process, so a message that outlives this
        means a hung env, and bounding it stops one stuck rollout stalling the
        whole batch forever. Pass ``None`` for unbounded (e.g. an env step that
        legitimately runs a very long tool job).
    :param mcp_tool: Optional MCP transport adapter (see
        :class:`OpenEnvSessionClient`).
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
        timeout_s: float | None = 300.0,
        mcp_tool: str | None = None,
        arg: str = "message",
        instruction: str = "",
    ) -> None:
        """Host ``env`` on a fresh server and open the session driving it."""
        self._server = OpenEnvServer(
            env, host=host, port=port, env_name=env_name
        ).start()
        try:
            self._client = OpenEnvSessionClient(
                self._server.base_url,
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
        """End the session, then stop the owned server (which closes the env).

        Closing the client ends the WebSocket session; stopping the server then
        closes the hosted env directly.
        """
        self._client.close()
        self._server.stop()

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
    env = resolve_entrypoint_target(spec)(**(env_config or {}))
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
    return resolve_entrypoint_target(spec)(**(env_config or {}))


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
