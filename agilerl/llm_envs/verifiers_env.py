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
import contextlib
import functools
import json
from typing import TYPE_CHECKING, Any, NamedTuple

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
from agilerl.llm_envs.verifiers_loop import SHARED_LOOP

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from openenv.core.env_server.types import Action, Observation
    from verifiers.v1 import Env, Episode, Task

__all__ = [
    "PROSE_TOOL",
    "ModelCallBridge",
    "VerifiersEnv",
    "VerifiersToolEnv",
    "make",
]

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
        self._context = ModelContext(
            model="agilerl-policy",
            client=EvalClientConfig(base_url=self._endpoint.base_url),
        )
        self._stop = asyncio.Event()
        self._pending_call: int | None = None
        self._episode: asyncio.Task[Episode] | None = None
        self._loop = SHARED_LOOP
        self._lifetime = self._loop.call(self._open())

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
        previous = self.messages
        return self._added(previous, self._loop.call(self._answer_pending(message)))

    async def answer_async(
        self, message: dict[str, Any]
    ) -> list[dict[str, Any]] | float:
        """Reply ``message`` to the pending call without holding a thread.

        :param message: The assistant message: ``content`` and/or ``tool_calls``.
        :return: As :meth:`answer`.
        """
        previous = self.messages
        body = await self._loop.acall(self._answer_pending(message))
        return self._added(previous, body)

    def close(self) -> None:
        """Stop any running episode and release the env's serving resources."""
        self._loop.call(self._shutdown())

    def _answer_pending(
        self, message: dict[str, Any]
    ) -> Coroutine[Any, Any, dict[str, Any] | float]:
        """Return the coroutine replying ``message`` to the pending call."""
        call_id, episode = self._pending_call, self._episode
        if call_id is None or episode is None:
            msg = "step() requires a pending model call; call reset() first"
            raise RuntimeError(msg)
        return self._answer(call_id, episode, message)

    def _added(
        self, previous: list[dict[str, Any]], body: dict[str, Any] | float
    ) -> list[dict[str, Any]] | float:
        """Return what the env added after ``previous`` and the reply, or the reward ``body``."""
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

    def _take(self, body: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Record and return a model call's messages and tools."""
        if body is None:
            msg = "verifiers episode ended before its first model call"
            raise RuntimeError(msg)
        self.messages = body["messages"]
        self.tools = body.get("tools") or []
        return self.messages

    async def _open(self) -> asyncio.Task[None]:
        """Start :meth:`_serve`; return its task once serving, or raise its failure."""
        ready = asyncio.Event()
        lifetime = asyncio.create_task(self._serve(ready))
        ready_wait = asyncio.ensure_future(ready.wait())
        await asyncio.wait({ready_wait, lifetime}, return_when=asyncio.FIRST_COMPLETED)
        if lifetime.done():
            ready_wait.cancel()
            lifetime.result()
        return lifetime

    async def _serve(self, ready: asyncio.Event) -> None:
        """Hold the endpoint and the env's serving resources until :meth:`close`."""
        await self._endpoint.open()
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

    async def _answer(
        self, call_id: int, episode: asyncio.Task[Episode], message: dict[str, Any]
    ) -> dict[str, Any] | float:
        """Reply ``message`` to call ``call_id``; return the next call, or ``episode``'s reward."""
        self._pending_call = None
        self._endpoint.answer(call_id, message)
        body = await self._next_call(episode)
        if body is None:
            return _episode_reward(episode.result())
        return body

    async def _next_call(self, episode: asyncio.Task[Episode]) -> dict[str, Any] | None:
        """Return the next model call's request body, or ``None`` once ``episode`` ends."""
        get = asyncio.ensure_future(self._endpoint.next_call())
        try:
            done, _ = await asyncio.wait(
                {get, episode}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            get.cancel()
        if get not in done:
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


class PendingReply(NamedTuple):
    """An assistant message for verifiers, and what turns its outcome into an observation."""

    message: dict[str, Any]
    finish: Callable[[list[dict[str, Any]] | float], CallToolObservation]


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
        # The last turn's calls and the results of its calls after the first.
        self._turn: tuple[list[dict[str, Any]], list[CallToolObservation]] | None = None
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
        self._turn = None
        prompt, system_prompt = self._bridge.begin(row_index)
        self._state = VerifiersToolState(
            episode_id=episode_id, step_count=0, dataset_size=len(self._bridge.tasks)
        )
        metadata: dict[str, Any] = {EPISODE_TOOLS_INFO_KEY: self._bridge.tools}
        if system_prompt is not None:
            metadata["system_prompt"] = system_prompt
        return TextObservation(
            prompt=prompt, done=False, reward=None, metadata=metadata
        )

    def step(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        """List this episode's tools, run one call of a turn, or send prose."""
        step = self._start_step(action, timeout_s, **kwargs)
        if not isinstance(step, PendingReply):
            return step
        return step.finish(self._bridge.answer(step.message))

    async def step_async(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        """:meth:`step` without holding a thread while the verifiers episode runs."""
        step = self._start_step(action, timeout_s, **kwargs)
        if not isinstance(step, PendingReply):
            return step
        return step.finish(await self._bridge.answer_async(step.message))

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

    def _start_step(
        self, action: Action, timeout_s: float | None, **kwargs: Any
    ) -> Observation | PendingReply:
        """Answer ``action`` here, or return the reply it sends to verifiers."""
        if isinstance(action, ListToolsAction):
            return ListToolsObservation(
                tools=[_mcp_tool(tool) for tool in self._bridge.tools]
            )
        if not isinstance(action, CallToolAction):
            return self._step_impl(action, timeout_s=timeout_s, **kwargs)
        self._state.step_count += 1
        if action.tool_name == PROSE_TOOL:
            return self._prose_reply(action.arguments)
        calls = action.metadata.get(TURN_TOOL_CALLS_KEY) or [
            {"name": action.tool_name, "arguments": action.arguments}
        ]
        index = int(action.metadata.get(TURN_TOOL_CALL_INDEX_KEY, 0))
        if index > 0:
            return self._turn_result(calls, index)
        return self._turn_reply(calls)

    def _turn_result(
        self, calls: list[dict[str, Any]], index: int
    ) -> CallToolObservation:
        """Return the result of call ``index`` of the turn whose first call ran last."""
        if index >= len(calls):
            msg = (
                f"tool call index {index} is outside its turn of {len(calls)} call(s)."
            )
            raise ValueError(msg)
        if self._turn is None or self._turn[0] != calls:
            msg = (
                f"tool call {index} of a turn whose first call did not run last; "
                "send the turn's call 0 first."
            )
            raise ValueError(msg)
        results = self._turn[1]
        if index > len(results):
            msg = f"tool call {index} has no result: the turn's first call ended the episode."
            raise ValueError(msg)
        return results[index - 1]

    def _turn_reply(self, calls: list[dict[str, Any]]) -> PendingReply:
        """Return the reply sending a turn's calls as one assistant message."""
        self._turn = None
        self._turns += 1
        ids = [f"call_{self._turns}_{index}" for index in range(len(calls))]
        message = {
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
        return PendingReply(message, functools.partial(self._finish_turn, calls, ids))

    def _finish_turn(
        self,
        calls: list[dict[str, Any]],
        ids: list[str],
        added: list[dict[str, Any]] | float,
    ) -> CallToolObservation:
        """Return the first call's result and keep the rest for the turn's later steps."""
        if not isinstance(added, list):
            self._turn = (calls, [])
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
        self._turn = (calls, observations[1:])
        return observations[0]

    def _prose_reply(self, arguments: dict[str, Any]) -> PendingReply:
        """Return the reply sending the prose (one string argument, any name) as the model's message."""
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
        self._turn = None
        return PendingReply({"role": "assistant", "content": text}, _finish_prose)


def _finish_prose(added: list[dict[str, Any]] | float) -> CallToolObservation:
    """Return the env's next user message after a prose reply, or the episode's end."""
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
