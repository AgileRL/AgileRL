# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Server half of the OpenEnv glue: host any text env behind a ``/ws`` URL.

Deliberately outside the ``llm`` extra: serving needs only openenv, pydantic and
uvicorn — never the trainer's closure. Clients live in :mod:`agilerl.llm_envs.openenv`.
"""

from __future__ import annotations

import contextlib
import inspect
import threading
import time
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, TypeGuard

import uvicorn
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from agilerl.llm_envs.env_specs import is_url, spec_to_factory

if TYPE_CHECKING:
    from typing import Self

    from agilerl.protocols import TextEnvProtocol


def _is_str_keyed_dict(obj: object) -> TypeGuard[dict[str, Any]]:
    """Torch-free twin of ``algo_utils.is_str_keyed_dict`` (this module shuns that closure)."""
    return isinstance(obj, dict)


class TextAction(Action):
    """OpenEnv action carrying the policy's generated text."""

    message: str = ""


class TextObservation(Observation):
    """OpenEnv observation carrying the next prompt text and optional labels.

    ``truncated`` is declared because the wire has only ``done`` (no
    terminated/truncated split), so it can travel and reconstruct the 5-tuple.

    On reset, ``prompt`` is shown to the policy and the labels stay unset. On the
    terminal step, ``question`` / ``answer`` are set so rubrics can score.
    """

    prompt: str = ""
    truncated: bool = False
    question: Any = None
    answer: Any = None


class TextState(State):
    """OpenEnv state carrying dataset size, tool schemas, and rubric component names."""

    dataset_size: int = 0
    tools: list[Any] = Field(default_factory=list)
    rubric_components: list[str] = Field(default_factory=list)


class OpenEnvWrapper(Environment):
    """Adapt a plain-text env to OpenEnv's typed ``Environment`` ABC.

    Translates between the env's string ``reset``/``step`` and OpenEnv's typed
    ``Action``/``Observation``/``State``, surfacing ``dataset_size``/``tools`` on
    the state and forwarding env ``info`` into observation metadata.

    :param inner: The local env to host.
    :param env_name: Name in the OpenEnv metadata; defaults to ``inner``'s class name.
    :param owns_inner: If ``True``, ``close`` closes ``inner`` (the per-session path).
    """

    # Each /ws session gets its own fresh inner env, so sessions never share state.
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
        prompt, info = _normalize_reset(self._inner.reset(**call))
        metadata = dict(info)
        if "system_prompt" not in metadata:
            inner_prompt = getattr(self._inner, "system_prompt", None)
            if isinstance(inner_prompt, str) and inner_prompt:
                metadata["system_prompt"] = inner_prompt
        self._state = State(episode_id=episode_id, step_count=0)
        return TextObservation(
            prompt=prompt, reward=None, done=False, metadata=metadata
        )

    def step(
        self,
        action: TextAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> TextObservation:
        """Step the inner env with the action's text, returning a ``TextObservation``."""
        prompt, reward, terminated, truncated, info = _normalize_step(
            self._inner.step(action.message)
        )
        self._state.step_count += 1
        return TextObservation(
            prompt=prompt,
            reward=reward,
            done=bool(terminated or truncated),
            truncated=bool(truncated),
            metadata=dict(info),
        )

    @property
    def state(self) -> TextState:
        """OpenEnv state, carrying the inner env's ``dataset_size`` / ``tools``."""
        return TextState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            dataset_size=int(getattr(self._inner, "dataset_size", 0) or 0),
            tools=list(getattr(self._inner, "tools", None) or []),
            rubric_components=list(
                getattr(self._inner, "rubric_components", None) or []
            ),
        )

    def close(self) -> None:
        """Close ``inner`` when this wrapper owns it (per-session path), else no-op."""
        if not self._owns_inner:
            return
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            with contextlib.suppress(Exception):
                closer()


def _normalize_reset(result: object) -> tuple[str, dict[str, Any]]:
    """Normalise an env ``reset`` return into ``(prompt, info)``."""
    if isinstance(result, tuple):
        if len(result) >= 2:
            info = result[1]
            return str(result[0]), info if _is_str_keyed_dict(info) else {}
        if len(result) == 1:
            return str(result[0]), {}
    return str(result), {}


def _normalize_step(result: object) -> tuple[str, Any, bool, bool, dict[str, Any]]:
    """Normalise an env ``step`` return into the Gym 5-tuple (also accepts a 4-tuple)."""
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
    return (
        str(obs),
        reward,
        bool(terminated),
        bool(truncated),
        info if _is_str_keyed_dict(info) else {},
    )


def wire_types(env: Environment) -> tuple[type[Action], type[Observation]]:
    """Return the action and observation classes ``env`` speaks over ``/ws``.

    An env may name them itself with ``ACTION_CLS`` / ``OBSERVATION_CLS`` (the
    escape hatch for an env whose interface is neither of the two below). An
    :class:`MCPEnvironment` is driven by tool calls, and anything else -- a plain
    text env, wrapped or not -- by the text interface.

    :param env: The env about to be served.
    :returns: Its ``(action, observation)`` classes.
    :rtype: tuple[type[Action], type[Observation]]
    """
    action_cls = getattr(env, "ACTION_CLS", None)
    observation_cls = getattr(env, "OBSERVATION_CLS", None)
    if (
        isinstance(action_cls, type)
        and issubclass(action_cls, Action)
        and isinstance(observation_cls, type)
        and issubclass(observation_cls, Observation)
    ):
        return action_cls, observation_cls

    if isinstance(env, MCPEnvironment):
        return CallToolAction, CallToolObservation
    return TextAction, TextObservation


class OpenEnvServer:
    """Serve OpenEnv's ``create_app`` on uvicorn in a background daemon thread.

    Pass ``env`` to share one env (one session at a time), or ``make_env`` +
    ``max_concurrent_envs`` for a fresh env per ``/ws`` session.
    ``max_concurrent_envs`` other than ``None`` or ``1`` requires ``make_env``.

    :param env: A single shared local env. Exactly one of ``env``/``make_env``.
    :param make_env: Zero-arg factory building a fresh env per session.
    :param host: Interface to bind (default loopback).
    :param port: TCP port; ``0`` lets the OS pick one.
    :param env_name: Name in the env's OpenEnv metadata; defaults to its class name.
    :param max_concurrent_envs: Max live sessions; set to the group size with ``make_env``.
    :param advertise_host: Host clients should dial, when it differs from the bound
        interface — a server binding ``0.0.0.0`` advertises its routable address
        (a pod IP, say), since ``0.0.0.0`` is not reachable from anywhere.
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
        advertise_host: str | None = None,
    ) -> None:
        """Build (but do not start) a server hosting ``env`` or ``make_env``."""
        if (env is None) == (make_env is None):
            msg = "OpenEnvServer requires exactly one of env or make_env"
            raise ValueError(msg)
        if env is not None and max_concurrent_envs not in (None, 1):
            msg = (
                "OpenEnvServer(env=...) cannot set max_concurrent_envs > 1; "
                "concurrent sessions would share one inner env. Pass make_env."
            )
            raise ValueError(msg)
        self._env = env
        self._make_env = make_env
        self._host = host
        self._port = port
        self._env_name = env_name
        self._max_concurrent_envs = max_concurrent_envs
        self._advertise_host = advertise_host or host
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._bound_port: int | None = None
        self._env_closed = False

    @property
    def port(self) -> int:
        """The TCP port the server bound (after :meth:`start`)."""
        if self._bound_port is None:
            msg = "OpenEnvServer is not running; call start() first"
            raise RuntimeError(msg)
        return self._bound_port

    @property
    def base_url(self) -> str:
        """The ``http://host:port`` clients should dial (after :meth:`start`)."""
        return f"http://{self._advertise_host}:{self.port}"

    def start(self) -> Self:
        """Serve in a background daemon thread (waits for bind); returns ``self``."""
        env = self._env
        make_env = self._make_env
        env_name = self._env_name

        def app_factory() -> Environment:
            # ``make_env`` -> a fresh owned env per session; ``env`` -> one shared.
            # OpenEnv Environments (e.g. QADatasetEnv) are hosted as-is.
            built = make_env() if make_env is not None else env
            if built is None:
                msg = "OpenEnvServer requires env or make_env"
                raise RuntimeError(msg)
            if isinstance(built, Environment):
                return built
            return OpenEnvWrapper(
                built,
                env_name=env_name,
                owns_inner=make_env is not None,
            )

        display_name = env_name or (
            type(env).__name__ if env is not None else "OpenEnvServer"
        )
        # The app declares one action type and validates every message against it,
        # so it has to be the type this env actually speaks -- an MCP env is sent
        # tool calls, not text, and would reject every message otherwise. Reading
        # that off an env means building one up front; under ``make_env`` that is
        # a whole extra env, belonging to no session, so close it again rather
        # than leave whatever it holds (a subprocess, a socket) open for the run.
        probe = app_factory()
        action_cls, observation_cls = wire_types(probe)
        if make_env is not None:
            with contextlib.suppress(Exception):
                probe.close()
        app = create_app(
            app_factory,
            action_cls,
            observation_cls,
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


def resolve_env(
    spec: str,
    env_config: dict[str, Any] | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    max_concurrent_envs: int | None = None,
    advertise_host: str | None = None,
) -> tuple[str, OpenEnvServer | None]:
    """Resolve an env spec to a ``(url, server)`` a ``RolloutHarness`` can hit.

    A URL is already hosted -> ``(url, None)``; anything else is built and hosted
    on a local :class:`OpenEnvServer`. ``max_concurrent_envs`` serves a fresh env
    per ``/ws`` session (one server backs a whole group); unset is one at a time.
    """
    if is_url(spec):
        return spec, None
    factory = spec_to_factory(spec)
    config = dict(env_config or {})
    # Matches ``RolloutHarness.from_spec``: a library factory (``gem.make``)
    # rejects ``system_prompt`` as a kwarg, so it is set on the built env.
    system_prompt = config.pop("system_prompt", None)

    def target(**factory_config: Any) -> TextEnvProtocol:
        env = factory(**factory_config)
        if system_prompt is not None:
            env.system_prompt = system_prompt
        return env

    shared_env = target(**config) if max_concurrent_envs is None else None
    server = OpenEnvServer(
        shared_env,
        make_env=None if shared_env is not None else partial(target, **config),
        host=host,
        port=port,
        env_name=_name_from_spec(spec),
        max_concurrent_envs=max_concurrent_envs,
        advertise_host=advertise_host,
    ).start()
    return server.base_url, server


def _name_from_spec(spec: str) -> str:
    """Trailing identifier of an entrypoint / path (``"pkg:Env-v0"`` -> ``"Env-v0"``)."""
    tail = spec.rsplit(":", 1)[-1]
    return tail.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or spec
