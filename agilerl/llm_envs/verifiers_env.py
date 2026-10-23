# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Serve a verifiers env (the Prime Intellect Environments Hub) as an AgileRL env.

verifiers runs its own episode loop and sends every model call to an
OpenAI-compatible endpoint. A :class:`ModelCallBridge` points that endpoint at a
local server and holds each request open until ``step`` supplies the reply, so
the verifiers loop advances one model call per step. :class:`VerifiersEnv`
serves text envs; :class:`VerifiersToolEnv` serves envs whose tasks bring tools,
as an MCP env the harness steps with parsed tool calls.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import sys
import threading
from typing import TYPE_CHECKING, Any, TypeVar, cast

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
)
from openenv.core.env_server.types import State
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.utils.loaders import load_environment, resolve_env_config

from agilerl.llm_envs.harness import EPISODE_TOOLS_INFO_KEY
from agilerl.llm_envs.model_endpoint import ModelCallEndpoint
from agilerl.llm_envs.openenv import TURN_TOOL_CALL_INDEX_KEY, TURN_TOOL_CALLS_KEY
from agilerl.llm_envs.openenv_server import TextObservation

if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence

    from openenv.core.env_server.types import Action, Observation
    from verifiers.v1 import Env, Episode, Task

__all__ = [
    "PROSE_TOOL",
    "ModelCallBridge",
    "VerifiersEnv",
    "VerifiersToolEnv",
    "make",
]

T = TypeVar("T")

# The manifest's ``answer_tool`` for a tool env: a turn with no tool call
# reaches verifiers as the model's prose reply. Never advertised.
PROSE_TOOL = "verifiers_reply"


def make(
    env_id: str, num_tasks: int | None = None, **env_config: Any
) -> VerifiersEnv | VerifiersToolEnv:
    """Build the env serving the verifiers taskset ``env_id``.

    :param env_id: Taskset id: a built-in one (``textarena``) or an installed
        Hub env (``owner/name``).
    :param num_tasks: Serve only the taskset's first ``num_tasks`` tasks.
        Required for an infinite taskset.
    :param env_config: The verifiers env config (agent roles, taskset options,
        timeouts) as plain data, the shape ``vf-eval`` takes under ``[env]``.
    :return: A :class:`VerifiersToolEnv` when the env's tasks bring tools, else
        a :class:`VerifiersEnv`; either serves tasks by ``row_index``.
    """
    taskset_config = {**env_config.pop("taskset", {}), "id": env_id}
    env = load_environment(
        resolve_env_config({**env_config, "taskset": taskset_config})
    )
    taskset = env.taskset
    if num_tasks is not None:
        taskset = taskset.take(num_tasks)
    elif not taskset.bounded:
        msg = f"verifiers taskset {env_id!r} is infinite; set num_tasks to bound it."
        raise ValueError(msg)
    task_toolsets = type(taskset).task_type().toolsets(taskset.config.task)
    bridge = ModelCallBridge(env, list(taskset))
    if task_toolsets or taskset.toolsets(taskset.config):
        return VerifiersToolEnv(bridge)
    return VerifiersEnv(bridge)


class EventLoopThread:
    """An asyncio event loop on a daemon thread, started on first :meth:`call`."""

    def __init__(self) -> None:
        """Create the loop; the thread starts on first use."""
        self._loop = asyncio.new_event_loop()
        self._exit: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._start_lock = threading.Lock()
        self._started = False

    def call(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the loop and wait for its result."""
        with self._start_lock:
            if not self._started:
                threading.Thread(
                    target=self._run, name="verifiers-env", daemon=True
                ).start()
                self._started = True
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        waiters: list[concurrent.futures.Future[Any]] = [future, self._exit]
        concurrent.futures.wait(waiters, return_when=concurrent.futures.FIRST_COMPLETED)
        if not future.done():
            self._exit.result()
        return future.result()

    def _run(self) -> None:
        """Run the loop, failing every pending :meth:`call` if it dies."""
        try:
            self._loop.run_forever()
        except BaseException as err:
            # verifiers raises SystemExit from its own loop callbacks (missing
            # Prime auth), which stops the loop under every waiter.
            self._exit.set_exception(
                RuntimeError(f"verifiers stopped its event loop: {err}")
            )
            raise


# verifiers keeps loop-bound asyncio state process-wide, so every env in a process
# shares one loop; sys.modules holds it because the env host loads this file once
# per session without registering the module.
SHARED_LOOP_KEY = "agilerl.llm_envs.verifiers_env:SHARED_LOOP"
SHARED_LOOP: EventLoopThread = cast("dict[str, Any]", sys.modules).setdefault(
    SHARED_LOOP_KEY, EventLoopThread()
)


class ModelCallBridge:
    """A verifiers env whose model calls this process answers, one at a time.

    Serves single-agent envs whose transcript only grows: every model call must
    repeat the previous one, then add the reply and what the env said back.
    Anything else raises. Sampling settings in the env's requests are ignored;
    the trainer samples.

    :param env: The verifiers env. This object owns it and closes its serving
        resources in :meth:`close`.
    :param tasks: Tasks served by ``row_index``.
    """

    def __init__(self, env: Env, tasks: Sequence[Task]) -> None:
        """Start the local endpoint and the env's serving resources."""
        self._env = env
        self.tasks = tasks
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self._endpoint = ModelCallEndpoint()
        self._pending_call: int | None = None
        self._episode: asyncio.Task[Episode] | None = None
        self._loop = SHARED_LOOP
        self._loop.call(self._open())

    def begin(self, row_index: int) -> tuple[str, str | None]:
        """Start an episode on task ``row_index``; return its first user message and system prompt."""
        body = self._loop.call(self._begin_episode(self.tasks[row_index]))
        messages = self._take(body)
        system_prompt = None
        turns = messages
        if messages[0]["role"] == "system":
            system_prompt = _text(messages[0])
            turns = messages[1:]
        if [m["role"] for m in turns] != ["user"]:
            msg = (
                "verifiers env opened with roles "
                f"{[m['role'] for m in messages]}; expected an optional system "
                "message then one user message."
            )
            raise ValueError(msg)
        return _text(turns[0]), system_prompt

    def answer(self, message: dict[str, Any]) -> list[dict[str, Any]] | float:
        """Reply ``message`` to the pending call.

        :param message: The assistant message: ``content`` and/or ``tool_calls``.
        :return: What the env added after the reply before its next model call,
            or the episode's reward once it ends.
        """
        if self._pending_call is None:
            msg = "step() requires a pending model call; call reset() first"
            raise RuntimeError(msg)
        previous = self.messages
        body = self._loop.call(self._answer(message))
        if not isinstance(body, dict):
            return body
        messages = self._take(body)
        added = messages[len(previous) :]
        changed = next(
            (
                i
                for i, (old, new) in enumerate(zip(previous, messages, strict=False))
                if _canonical(old) != _canonical(new)
            ),
            None,
        )
        if changed is not None or not added or added[0]["role"] != "assistant":
            detail = (
                f"message {changed} was {previous[changed]!r:.300} and is now "
                f"{messages[changed]!r:.300}"
                if changed is not None
                else f"the reply was followed by roles {[m['role'] for m in added]}"
            )
            msg = (
                "verifiers env rewrote the transcript between model calls "
                f"({detail}); verifiers envs are served when their transcript only grows."
            )
            raise ValueError(msg)
        return added[1:]

    def close(self) -> None:
        """Stop any running episode and release the env's serving resources."""
        self._loop.call(self._shutdown())

    def _take(self, body: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Record and return a model call's messages and tools."""
        if body is None:
            msg = "verifiers episode ended before its first model call"
            raise RuntimeError(msg)
        self.messages = body["messages"]
        self.tools = body.get("tools") or []
        return self.messages

    async def _open(self) -> None:
        """Start :meth:`_serve` and wait until it is serving (or has failed)."""
        ready = asyncio.Event()
        self._lifetime = asyncio.create_task(self._serve(ready))
        ready_wait = asyncio.ensure_future(ready.wait())
        await asyncio.wait(
            {ready_wait, self._lifetime}, return_when=asyncio.FIRST_COMPLETED
        )
        if self._lifetime.done():
            ready_wait.cancel()
            self._lifetime.result()

    async def _serve(self, ready: asyncio.Event) -> None:
        """Hold the endpoint and the env's serving resources until :meth:`close`."""
        self._stop = asyncio.Event()
        base_url = await self._endpoint.open()
        self._context = ModelContext(
            model="agilerl-policy", client=EvalClientConfig(base_url=base_url)
        )
        try:
            async with self._env.serving():
                ready.set()
                await self._stop.wait()
        finally:
            await self._endpoint.close()

    async def _begin_episode(self, task: Task) -> dict[str, Any] | None:
        """Cancel any running episode, start one on ``task``, and await its first call."""
        await self._cancel_episode()
        self._endpoint.reset()
        self._episode = asyncio.create_task(self._env.run_episode(task, self._context))
        return await self._next_call(self._episode)

    async def _answer(self, message: dict[str, Any]) -> dict[str, Any] | float:
        """Reply ``message`` to the pending call; return the next call, or the episode reward."""
        episode = self._episode
        if self._pending_call is None or episode is None:
            msg = "no pending model call"
            raise RuntimeError(msg)
        call_id = self._pending_call
        self._pending_call = None
        self._endpoint.answer(call_id, message)
        body = await self._next_call(episode)
        if body is None:
            return _episode_reward(episode.result())
        return body

    async def _next_call(self, episode: asyncio.Task[Episode]) -> dict[str, Any] | None:
        """Return the next model call's request body, or ``None`` once ``episode`` ends."""
        get = asyncio.ensure_future(self._endpoint.next_call())
        await asyncio.wait({get, episode}, return_when=asyncio.FIRST_COMPLETED)
        if not get.done():
            get.cancel()
            finished = episode.result()
            if not finished.ok:
                raise RuntimeError(_failure_detail(finished))
            return None
        call = get.result()
        self._pending_call = call.call_id
        return call.body

    async def _cancel_episode(self) -> None:
        """Cancel the running episode and its pending model call, and wait for both."""
        if self._pending_call is not None:
            self._endpoint.cancel(self._pending_call)
            self._pending_call = None
        if self._episode is None or self._episode.done():
            return
        self._episode.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._episode

    async def _shutdown(self) -> None:
        """Cancel the running episode and release the serving resources."""
        await self._cancel_episode()
        self._stop.set()
        await self._lifetime


class VerifiersEnv:
    """Text env over a verifiers env: each model call the env makes is one :meth:`step`.

    :param bridge: The bridge to the verifiers env; this env owns it.
    """

    def __init__(self, bridge: ModelCallBridge) -> None:
        """Serve ``bridge``'s env."""
        self._bridge = bridge

    @property
    def dataset_size(self) -> int:
        """Number of tasks served."""
        return len(self._bridge.tasks)

    @property
    def tools(self) -> list[Any]:
        """Tool schemas advertised to the policy (none: text-only envs)."""
        return []

    def reset(self, row_index: int) -> tuple[str, dict[str, Any]]:
        """Start an episode on task ``row_index`` and return its first user message.

        :param row_index: Index into the served tasks.
        :return: The user message, and ``system_prompt`` in the info when the
            env sends one.
        """
        prompt, system_prompt = self._bridge.begin(row_index)
        self._reject_tools()
        info = {} if system_prompt is None else {"system_prompt": system_prompt}
        return prompt, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Send ``action`` as the model's reply and return the env's next user message.

        :param action: The policy's completion text.
        :return: Gym 5-tuple. Reward is the episode's reward on the final
            step and ``0.0`` before it.
        """
        added = self._bridge.answer({"role": "assistant", "content": action})
        if not isinstance(added, list):
            return "", added, True, False, {}
        self._reject_tools()
        return _user_feedback(added), 0.0, False, False, {}

    def close(self) -> None:
        """Stop any running episode and release the env's serving resources."""
        self._bridge.close()

    def _reject_tools(self) -> None:
        """Fail a model call that offers tools: this env steps on text."""
        if self._bridge.tools:
            msg = (
                "verifiers env advertised tools to the model, but its tasks "
                "declare no toolsets, so it is served as a text env."
            )
            raise ValueError(msg)


class VerifiersToolState(State):
    """Episode state plus the task count, which the harness reads as ``dataset_size``."""

    dataset_size: int = 0


class VerifiersToolEnv(MCPEnvironment):
    """MCP env over a verifiers env whose tasks bring tools.

    Each episode's tools are the ones its first model call offers; they are in
    the reset info under :data:`~agilerl.llm_envs.harness.EPISODE_TOOLS_INFO_KEY`.
    A turn's parsed calls reach verifiers as one assistant message on the turn's
    first step; that and the turn's later steps each return one call's result.
    Prose goes in through :data:`PROSE_TOOL` and returns the env's next user
    message.

    :param bridge: The bridge to the verifiers env; this env owns it.
    """

    # The host builds one instance per /ws session.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, bridge: ModelCallBridge) -> None:
        """Serve ``bridge``'s env."""
        super().__init__(FastMCP("verifiers"))
        self._bridge = bridge
        self._state = VerifiersToolState(dataset_size=len(bridge.tasks))
        self._turn_results: list[CallToolObservation] = []
        self._turns = 0

    @property
    def state(self) -> VerifiersToolState:
        """Episode state and task count."""
        return self._state

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        row_index: int | None = None,
        **kwargs: Any,
    ) -> TextObservation:
        """Start an episode on task ``row_index``; its tools are in the metadata."""
        if row_index is None:
            msg = "VerifiersToolEnv serves tasks by row_index; reset needs one."
            raise ValueError(msg)
        prompt, system_prompt = self._bridge.begin(row_index)
        self._state = VerifiersToolState(
            episode_id=episode_id, step_count=0, dataset_size=len(self._bridge.tasks)
        )
        self._turn_results = []
        metadata: dict[str, Any] = {EPISODE_TOOLS_INFO_KEY: self._bridge.tools}
        if system_prompt is not None:
            metadata["system_prompt"] = system_prompt
        return TextObservation(
            prompt=prompt, done=False, reward=None, metadata=metadata
        )

    async def reset_async(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> TextObservation:
        """Reset off the event loop: it waits on the verifiers episode."""
        return await asyncio.to_thread(self.reset, seed, episode_id, **kwargs)

    def step(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        """List this episode's tools, run one call of a turn, or send prose."""
        if isinstance(action, ListToolsAction):
            return ListToolsObservation(
                tools=[_mcp_tool(tool) for tool in self._bridge.tools]
            )
        if not isinstance(action, CallToolAction):
            return self._step_impl(action, timeout_s=timeout_s, **kwargs)
        self._state.step_count += 1
        if action.tool_name == PROSE_TOOL:
            return self._reply(action.arguments)
        index = int(action.metadata.get(TURN_TOOL_CALL_INDEX_KEY, 0))
        if index > 0:
            return self._turn_results[index - 1]
        calls = action.metadata.get(TURN_TOOL_CALLS_KEY) or [
            {"name": action.tool_name, "arguments": action.arguments}
        ]
        return self._run_turn(calls)

    async def step_async(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        """Step off the event loop: it waits on the verifiers episode."""
        return await asyncio.to_thread(self.step, action, timeout_s, **kwargs)

    def close(self) -> None:
        """Stop any running episode and release the env's serving resources."""
        self._bridge.close()
        super().close()

    def _step_impl(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        """Reject actions that are not tool calls."""
        msg = f"VerifiersToolEnv takes tool calls, not {type(action).__name__}."
        raise ValueError(msg)

    def _run_turn(self, calls: list[dict[str, Any]]) -> CallToolObservation:
        """Send a turn's calls as one assistant message; return the first call's result."""
        self._turns += 1
        ids = [f"call_{self._turns}_{index}" for index in range(len(calls))]
        added = self._bridge.answer(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    for call_id, call in zip(ids, calls, strict=True)
                ],
            }
        )
        if not isinstance(added, list):
            self._turn_results = []
            return _observation(calls[0]["name"], "", "tool", reward=added, done=True)
        results = {m.get("tool_call_id"): m for m in added if m["role"] == "tool"}
        if len(added) != len(calls) or set(results) != set(ids):
            msg = (
                "verifiers env answered a turn of "
                f"{len(calls)} tool call(s) with roles {[m['role'] for m in added]}; "
                "expected one tool message per call."
            )
            raise ValueError(msg)
        observations = [
            _observation(call["name"], _text(results[call_id]), "tool")
            for call_id, call in zip(ids, calls, strict=True)
        ]
        self._turn_results = observations[1:]
        return observations[0]

    def _reply(self, arguments: dict[str, Any]) -> CallToolObservation:
        """Send the prose argument as the model's reply; return the env's next user message.

        The prose arrives under the client's action field, which the rollout
        path does not take from the manifest, so it is read by value: the
        call must carry exactly one string argument.
        """
        if len(arguments) != 1:
            msg = (
                f"{PROSE_TOOL} takes the prose as one argument, "
                f"not {sorted(arguments)}."
            )
            raise ValueError(msg)
        ((_, text),) = arguments.items()
        if not isinstance(text, str):
            msg = (
                f"{PROSE_TOOL} takes the prose as a string, not {type(text).__name__}."
            )
            raise TypeError(msg)
        self._turn_results = []
        added = self._bridge.answer({"role": "assistant", "content": text})
        if not isinstance(added, list):
            return _observation(PROSE_TOOL, "", "user", reward=added, done=True)
        return _observation(PROSE_TOOL, _user_feedback(added), "user")


def _observation(
    tool_name: str, text: str, role: str, reward: float = 0.0, done: bool = False
) -> CallToolObservation:
    """A tool step's observation: ``text`` as an MCP result, labelled ``role``."""
    return CallToolObservation(
        tool_name=tool_name,
        result={"content": [{"type": "text", "text": text}]},
        reward=reward,
        done=done,
        metadata={"role": role},
    )


def _mcp_tool(tool: dict[str, Any]) -> Tool:
    """An OpenAI function schema as an MCP tool listing."""
    function = tool["function"]
    return Tool(
        name=function["name"],
        description=function.get("description") or "",
        input_schema=function.get("parameters") or {},
    )


def _canonical(message: dict[str, Any]) -> dict[str, Any]:
    """``message`` without the fields verifiers drops or adds when it re-serializes it.

    An assistant message comes back with ``content: None`` and each tool call's
    stream ``index`` present on one request and absent on the next.
    """
    canonical = {k: v for k, v in message.items() if v is not None}
    if "tool_calls" in canonical:
        canonical["tool_calls"] = [
            {k: v for k, v in call.items() if k != "index"}
            for call in canonical["tool_calls"]
        ]
    return canonical


def _user_feedback(added: list[dict[str, Any]]) -> str:
    """The env's user messages after a reply, joined."""
    if not added or any(m["role"] != "user" for m in added):
        msg = (
            "verifiers env answered with roles "
            f"{[m['role'] for m in added]}; expected user messages."
        )
        raise ValueError(msg)
    return "\n\n".join(_text(m) for m in added)


def _text(message: dict[str, Any]) -> str:
    """Return a chat message's text content: a string, or text parts joined."""
    content = message["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") == "text" for part in content
    ):
        return "".join(part["text"] for part in content)
    msg = (
        f"verifiers env sent non-text {message['role']} content; "
        "verifiers envs are served on text."
    )
    raise TypeError(msg)


def _failure_detail(episode: Episode) -> str:
    """Summarize a failed episode from its episode- and trace-level errors."""
    parts = [f"{e.type}: {e.message}" for e in episode.errors]
    parts.extend(
        f"{err.type}: {err.message}"
        for trace in episode.traces
        if (err := trace.last_error) is not None
    )
    detail = "; ".join(parts) if parts else "no error recorded"
    return f"verifiers episode failed: {detail}"


def _episode_reward(episode: Episode) -> float:
    """Return the reward of an episode that played one agent to completion."""
    if not episode.ok:
        raise RuntimeError(_failure_detail(episode))
    if len(episode.traces) != 1:
        msg = (
            f"verifiers episode ran {len(episode.traces)} agents; verifiers "
            "envs are served when single-agent."
        )
        raise ValueError(msg)
    return float(episode.traces[0].reward)
