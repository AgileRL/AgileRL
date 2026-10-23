# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The verifiers adapters drive real verifiers v1 envs, one model call per step."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
import pytest
import torch

pytest.importorskip("verifiers", reason="verifiers extra not installed")
pytest.importorskip("transformers", reason="LLM dependencies not installed")
# verifiers.v1 imports fcntl at package import time, so the top-level import
# succeeds on Windows but anything under v1 does not.
pytest.importorskip("verifiers.v1.episode", reason="verifiers v1 needs fcntl")

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv.core.env_server.types import Action
from transformers import AutoTokenizer
from verifiers.v1.episode import Episode
from verifiers.v1.trace import Error, Trace

from agilerl.llm_envs import verifiers_env
from agilerl.llm_envs.harness import EPISODE_TOOLS_INFO_KEY, RolloutHarness
from agilerl.llm_envs.model_endpoint import (
    AnsweredCall,
    ModelCallEndpoint,
)
from agilerl.llm_envs.openenv import (
    TURN_TOOL_CALL_INDEX_KEY,
    TURN_TOOL_CALLS_KEY,
    InProcessEnvClient,
)
from agilerl.llm_envs.tool_parsers import ParsedToolCall
from agilerl.llm_envs.verifiers_env import (
    PROSE_TOOL,
    EventLoopThread,
    ModelCallBridge,
    VerifiersEnv,
    VerifiersToolEnv,
    _episode_reward,
    _failure_detail,
    _text,
    make,
)
from tests import TINY_LLM_FIXTURE_PATH
from tests.envs import verifiers_calc, verifiers_guess

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterator

TASKSET_ID = "agilerl-verifiers-guess"
CALC_TASKSET_ID = "agilerl-verifiers-calc"
PLAYER = {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}}
# The calc tool server runs as ``python -m tests.envs.verifiers_calc``.
TESTS_ROOT = str(Path(__file__).resolve().parents[2])


@pytest.fixture(autouse=True)
def guess_taskset_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fixture env loadable by the flat id verifiers imports."""
    monkeypatch.setitem(sys.modules, "agilerl_verifiers_guess", verifiers_guess)
    monkeypatch.setitem(sys.modules, "agilerl_verifiers_calc", verifiers_calc)
    path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH", TESTS_ROOT if not path else f"{TESTS_ROOT}{os.pathsep}{path}"
    )


@pytest.fixture
def env() -> Iterator[VerifiersEnv]:
    guess_env = make(TASKSET_ID, player=PLAYER)
    yield guess_env
    guess_env.close()


@pytest.fixture
def tool_env() -> Iterator[VerifiersToolEnv]:
    calc_env = make(CALC_TASKSET_ID, agent=PLAYER)
    yield calc_env
    calc_env.close()


def add_call(a: int, b: int) -> dict[str, object]:
    return {"name": "calc_add", "arguments": {"a": a, "b": b}}


def turn_action(calls: list[dict[str, object]], index: int) -> CallToolAction:
    call = calls[index]
    return CallToolAction(
        tool_name=call["name"],
        arguments=call["arguments"],
        metadata={TURN_TOOL_CALLS_KEY: calls, TURN_TOOL_CALL_INDEX_KEY: index},
    )


def result_text(observation: object) -> str:
    return observation.result["content"][0]["text"]


async def post_completion(
    session: aiohttp.ClientSession, payload: dict[str, object]
) -> dict[str, object]:
    """POST one chat completion; return its JSON body."""
    async with session.post("/v1/chat/completions", json=payload) as response:
        assert response.status == 200
        return await response.json()


async def post_stream(
    session: aiohttp.ClientSession, payload: dict[str, object]
) -> str:
    """POST one streamed chat completion; return its event-stream text."""
    async with session.post("/v1/chat/completions", json=payload) as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == "text/event-stream"
        return await response.text()


class ScriptedLoop:
    """EventLoopThread double answering every call with one body."""

    def __init__(self, body: object) -> None:
        self.body = body

    def call(self, coro: Coroutine[Any, Any, Any]) -> object:
        coro.close()
        return self.body


class ScriptedBridge:
    """ModelCallBridge double answering turns from a script (no verifiers)."""

    def __init__(
        self, answers: list[object], tools: list[dict[str, Any]] | None = None
    ) -> None:
        self.tasks = [object()]
        self.tools = tools or []
        self._answers = list(answers)

    def begin(self, row_index: int) -> tuple[str, None]:
        del row_index
        return "prompt", None

    def answer(self, message: dict[str, Any]) -> object:
        del message
        return self._answers.pop(0)


def bare_bridge(**attrs: object) -> ModelCallBridge:
    """A bridge with only the attributes a validation branch touches."""
    bridge = ModelCallBridge.__new__(ModelCallBridge)
    for name, value in attrs.items():
        setattr(bridge, name, value)
    return bridge


class TestMake:
    def test_serves_every_task_of_a_finite_taskset(self, env: VerifiersEnv) -> None:
        assert env.dataset_size == len(verifiers_guess.SECRETS)
        assert env.tools == []

    def test_num_tasks_bounds_the_taskset(self) -> None:
        env = make(TASKSET_ID, num_tasks=1, player=PLAYER)
        try:
            assert env.dataset_size == 1
        finally:
            env.close()

    def test_infinite_taskset_without_num_tasks_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(verifiers_guess.GuessTaskset, "INFINITE", True)

        with pytest.raises(ValueError, match="infinite; set num_tasks"):
            make(TASKSET_ID, player=PLAYER)


class TestVerifiersEnvReset:
    def test_returns_first_user_message_and_system_prompt(
        self, env: VerifiersEnv
    ) -> None:
        prompt, info = env.reset(row_index=1)

        assert prompt == "Guess fruit number 1."
        assert info == {"system_prompt": verifiers_guess.SYSTEM_PROMPT}

    def test_mid_episode_reset_starts_a_fresh_episode(self, env: VerifiersEnv) -> None:
        # Arrange
        env.reset(row_index=0)
        env.step("banana")

        # Act
        prompt, _ = env.reset(row_index=1)
        result = env.step("pear")

        # Assert
        assert prompt == "Guess fruit number 1."
        assert result == ("", 1.0, True, False, {})


class TestVerifiersEnvStep:
    def test_wrong_reply_returns_the_env_feedback(self, env: VerifiersEnv) -> None:
        env.reset(row_index=0)

        result = env.step("banana")

        assert result == ("Wrong, guess again.", 0.0, False, False, {})

    def test_correct_reply_ends_the_episode_with_its_reward(
        self, env: VerifiersEnv
    ) -> None:
        env.reset(row_index=0)
        env.step("banana")

        result = env.step("apple")

        assert result == ("", 1.0, True, False, {})

    def test_last_wrong_reply_ends_the_episode_with_zero_reward(
        self, env: VerifiersEnv
    ) -> None:
        env.reset(row_index=0)
        env.step("banana")
        env.step("cherry")

        result = env.step("grape")

        assert result == ("", 0.0, True, False, {})

    def test_two_envs_in_one_process_play_independently(
        self, env: VerifiersEnv
    ) -> None:
        # Arrange
        other = make(TASKSET_ID, player=PLAYER)
        try:
            env.reset(row_index=0)
            other.reset(row_index=1)

            # Act
            results = (env.step("apple"), other.step("pear"))
        finally:
            other.close()

        # Assert
        assert results == (("", 1.0, True, False, {}), ("", 1.0, True, False, {}))

    def test_step_before_reset_raises(self, env: VerifiersEnv) -> None:
        with pytest.raises(RuntimeError, match="call reset\\(\\) first"):
            env.step("apple")


class TestRolloutHarnessFromSpecWithVerifiersFactory:
    def test_harness_trains_on_the_verifiers_reward(self) -> None:
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        harness = RolloutHarness.from_spec(
            TASKSET_ID,
            {"player": PLAYER},
            tokenizer,
            max_turns=3,
            factory="agilerl.llm_envs.verifiers_env:make",
        )
        try:
            prompt, _ = harness.reset(row_index=0)
            reply = tokenizer.encode("apple", add_special_tokens=False)
            token_ids = torch.cat([prompt["input_ids"], torch.tensor([reply])], dim=1)

            # Act
            _, reward, terminated, truncated, _ = harness.step(token_ids)

            # Assert
            assert (reward, terminated, truncated) == (1.0, True, False)
            assert harness.turn_rewards == [1.0]
            assert verifiers_guess.SYSTEM_PROMPT in tokenizer.decode(
                prompt["input_ids"][0]
            )
        finally:
            harness.close()


class TestMakeToolTasksets:
    def test_serves_a_taskset_with_toolsets_as_a_tool_env(
        self, tool_env: VerifiersToolEnv
    ) -> None:
        assert isinstance(tool_env, VerifiersToolEnv)
        assert tool_env.state.dataset_size == 1

    def test_serves_a_taskset_without_toolsets_as_a_text_env(
        self, env: VerifiersEnv
    ) -> None:
        assert isinstance(env, VerifiersEnv)


class TestVerifiersToolEnvReset:
    def test_returns_the_prompt_and_the_episode_tools(
        self, tool_env: VerifiersToolEnv
    ) -> None:
        observation = tool_env.reset(row_index=0)

        assert observation.prompt == verifiers_calc.PROMPT
        tools = observation.metadata[EPISODE_TOOLS_INFO_KEY]
        assert [tool["function"]["name"] for tool in tools] == ["calc_add"]
        assert observation.metadata["system_prompt"] == verifiers_calc.SYSTEM_PROMPT

    def test_reset_async_resets_off_the_event_loop(
        self, tool_env: VerifiersToolEnv
    ) -> None:
        observation = asyncio.run(tool_env.reset_async(row_index=0))

        assert observation.prompt == verifiers_calc.PROMPT

    def test_lists_the_episode_tools(self, tool_env: VerifiersToolEnv) -> None:
        tool_env.reset(row_index=0)

        listed = tool_env.step(ListToolsAction())

        assert [tool.name for tool in listed.tools] == ["calc_add"]

    def test_reset_without_row_index_raises(self, tool_env: VerifiersToolEnv) -> None:
        with pytest.raises(ValueError, match="needs one"):
            tool_env.reset()


class TestVerifiersToolEnvStep:
    def test_each_tool_turn_returns_each_call_result(
        self, tool_env: VerifiersToolEnv
    ) -> None:
        # Arrange
        tool_env.reset(row_index=0)
        calls = [add_call(2, 3), add_call(5, 4)]

        # Act
        first = tool_env.step(turn_action(calls, 0))
        second = tool_env.step(turn_action(calls, 1))
        third = tool_env.step(turn_action([add_call(1, 1)], 0))

        # Assert
        assert [result_text(first), result_text(second)] == ["5", "9"]
        assert [result_text(third)] == ["2"]
        assert (first.done, second.done, third.done) == (False, False, False)
        assert first.metadata == {"role": "tool"}

    def test_correct_prose_answer_ends_the_episode_with_its_reward(
        self, tool_env: VerifiersToolEnv
    ) -> None:
        tool_env.reset(row_index=0)
        tool_env.step(turn_action([add_call(2, 3)], 0))

        answer = tool_env.step(
            CallToolAction(tool_name=PROSE_TOOL, arguments={"text": "9"})
        )

        assert (answer.done, answer.reward) == (True, 1.0)

    def test_prose_arrives_under_the_clients_action_field(
        self, tool_env: VerifiersToolEnv
    ) -> None:
        # Remote clients send prose under their own action field ("message" by
        # default), not the manifest's.
        tool_env.reset(row_index=0)

        answer = tool_env.step(
            CallToolAction(tool_name=PROSE_TOOL, arguments={"message": "9"})
        )

        assert (answer.done, answer.reward) == (True, 1.0)

    def test_wrong_prose_answer_scores_zero(self, tool_env: VerifiersToolEnv) -> None:
        tool_env.reset(row_index=0)

        answer = tool_env.step(
            CallToolAction(tool_name=PROSE_TOOL, arguments={"text": "7"})
        )

        assert (answer.done, answer.reward) == (True, 0.0)

    def test_prose_answer_needs_exactly_one_string_argument(
        self, tool_env: VerifiersToolEnv
    ) -> None:
        tool_env.reset(row_index=0)

        with pytest.raises(ValueError, match="one argument"):
            tool_env.step(CallToolAction(tool_name=PROSE_TOOL, arguments={}))
        with pytest.raises(TypeError, match="as a string"):
            tool_env.step(CallToolAction(tool_name=PROSE_TOOL, arguments={"text": 9}))

    def test_actions_that_are_not_tool_calls_raise(
        self, tool_env: VerifiersToolEnv
    ) -> None:
        tool_env.reset(row_index=0)

        with pytest.raises(ValueError, match="takes tool calls"):
            tool_env.step(Action())


class TestInProcessEnvClientOverVerifiersToolEnv:
    def test_steps_parsed_calls_of_one_turn(self, tool_env: VerifiersToolEnv) -> None:
        # Arrange
        client = InProcessEnvClient(tool_env)
        client.reset(row_index=0)
        calls = (
            ParsedToolCall(raw="", name="calc_add", arguments={"a": 2, "b": 3}),
            ParsedToolCall(raw="", name="calc_add", arguments={"a": 5, "b": 4}),
        )
        turn = [
            ParsedToolCall(
                raw="", name=c.name, arguments=c.arguments, turn=calls, turn_index=i
            )
            for i, c in enumerate(calls)
        ]

        # Act
        results = [client.step(call) for call in turn]

        # Assert
        assert client.takes_tool_calls is True
        assert [r[0]["result"]["content"][0]["text"] for r in results] == ["5", "9"]
        assert [r[4]["role"] for r in results] == ["tool", "tool"]


class TestModelCallEndpoint:
    def test_answers_calls_with_sequential_ids_and_records_them(self) -> None:
        async def go() -> None:
            endpoint = ModelCallEndpoint()
            base_url = await endpoint.open()
            requests = [
                {"model": "m", "messages": [{"role": "user", "content": text}]}
                for text in ("first", "second")
            ]
            replies = [
                {"role": "assistant", "content": f"reply {index}"} for index in (0, 1)
            ]
            try:
                async with aiohttp.ClientSession(base_url=base_url + "/") as session:
                    for index, (request, reply) in enumerate(
                        zip(requests, replies, strict=True)
                    ):
                        task = asyncio.create_task(post_completion(session, request))
                        call = await endpoint.next_call()

                        assert call.call_id == index
                        assert call.body == request
                        endpoint.answer(call.call_id, reply)
                        completion = await task

                        assert completion["choices"][0]["message"] == reply
                        assert completion["choices"][0]["finish_reason"] == "stop"
                        endpoint.reset()
            finally:
                await endpoint.close()

            assert endpoint.calls == [
                AnsweredCall(0, requests[0], replies[0]),
                AnsweredCall(1, requests[1], replies[1]),
            ]

        asyncio.run(go())

    def test_streaming_call_answers_as_server_sent_events(self) -> None:
        async def go() -> str:
            endpoint = ModelCallEndpoint()
            base_url = await endpoint.open()
            try:
                async with aiohttp.ClientSession(base_url=base_url + "/") as session:
                    task = asyncio.create_task(
                        post_stream(
                            session,
                            {
                                "model": "m",
                                "stream": True,
                                "messages": [{"role": "user", "content": "add"}],
                            },
                        )
                    )
                    call = await endpoint.next_call()
                    endpoint.answer(
                        call.call_id,
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {
                                        "name": "calc_add",
                                        "arguments": '{"a": 2}',
                                    },
                                }
                            ],
                        },
                    )
                    return await task
            finally:
                await endpoint.close()

        lines = asyncio.run(go()).splitlines()
        frames = [
            json.loads(line[len("data: ") :])
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]

        assert "data: [DONE]" in lines
        delta = frames[0]["choices"][0]["delta"]
        assert delta["tool_calls"][0]["function"]["name"] == "calc_add"
        assert frames[1]["choices"][0]["finish_reason"] == "tool_calls"
        assert frames[2]["usage"]["total_tokens"] == 0

    def test_cancel_fails_the_held_http_call(self) -> None:
        async def go() -> None:
            endpoint = ModelCallEndpoint()
            base_url = await endpoint.open()
            try:
                async with aiohttp.ClientSession(base_url=base_url + "/") as session:
                    task = asyncio.create_task(
                        post_completion(
                            session,
                            {"model": "m", "messages": [{"role": "user"}]},
                        )
                    )
                    call = await endpoint.next_call()
                    endpoint.cancel(call.call_id)

                    with pytest.raises(aiohttp.ClientError):
                        await task
            finally:
                await endpoint.close()

            assert endpoint.calls == []

        asyncio.run(go())

    def test_reset_drops_queued_calls(self) -> None:
        endpoint = ModelCallEndpoint()
        endpoint._pending.put_nowait((0, {"messages": []}))

        endpoint.reset()

        assert endpoint._pending.empty()
        assert endpoint._awaiting == {}


class TestBridgeValidation:
    def test_begin_rejects_an_opening_without_one_user_message(self) -> None:
        bridge = bare_bridge(
            tasks=[object()],
            _loop=ScriptedLoop(
                {
                    "messages": [{"role": "user"}, {"role": "user"}],
                    "tools": [],
                }
            ),
        )

        with pytest.raises(ValueError, match="expected an optional system"):
            bridge.begin(row_index=0)

    def test_answer_rejects_a_rewritten_transcript(self) -> None:
        bridge = bare_bridge(
            messages=[{"role": "user", "content": "hi"}],
            _pending_call=0,
            _loop=ScriptedLoop(
                {
                    "messages": [
                        {"role": "user", "content": "bye"},
                        {"role": "assistant", "content": "x"},
                    ],
                    "tools": [],
                }
            ),
        )

        with pytest.raises(ValueError, match="rewrote the transcript"):
            bridge.answer({"role": "assistant", "content": "hi"})

    def test_take_rejects_an_episode_that_ended_before_its_first_call(self) -> None:
        with pytest.raises(RuntimeError, match="before its first model call"):
            ModelCallBridge._take(bare_bridge(), None)

    def test_answer_without_a_pending_call_raises(self) -> None:
        bridge = bare_bridge(_pending_call=None, _episode=None)

        with pytest.raises(RuntimeError, match="no pending model call"):
            asyncio.run(bridge._answer({"role": "assistant", "content": "x"}))

    def test_next_call_surfaces_a_failed_episode(self) -> None:
        async def go() -> None:
            bridge = bare_bridge(_endpoint=ModelCallEndpoint())

            async def done() -> Episode:
                return Episode.model_construct(
                    ok=False, errors=[Error(type="E", message="m")], traces=[]
                )

            await bridge._next_call(asyncio.create_task(done()))

        with pytest.raises(RuntimeError, match="verifiers episode failed"):
            asyncio.run(go())

    def test_open_fails_when_serving_does(self) -> None:
        class _FailingServe:
            async def __aenter__(self) -> None:
                msg = "serve boom"
                raise RuntimeError(msg)

            async def __aexit__(self, *args: object) -> bool:
                return False

        class _FailingEnv:
            def serving(self) -> _FailingServe:
                return _FailingServe()

        async def go() -> None:
            await bare_bridge(_env=_FailingEnv(), _endpoint=ModelCallEndpoint())._open()

        with pytest.raises(RuntimeError, match="serve boom"):
            asyncio.run(go())


class TestEventLoopThreadDeath:
    def test_call_fails_fast_when_the_loop_died(self) -> None:
        thread = EventLoopThread()
        thread._exit.set_exception(RuntimeError("loop is gone"))

        with pytest.raises(RuntimeError, match="loop is gone"):
            thread.call(asyncio.sleep(60))

    def test_run_fails_waiters_when_the_loop_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread = EventLoopThread()

        def dead_loop() -> None:
            raise SystemExit(1)

        monkeypatch.setattr(thread._loop, "run_forever", dead_loop)
        with pytest.raises(SystemExit):
            thread._run()
        with pytest.raises(RuntimeError, match="stopped its event loop"):
            thread._exit.result()


class TestVerifiersEnvRejectsTools:
    def test_reset_rejects_a_model_call_with_tools(self) -> None:
        env = VerifiersEnv(ScriptedBridge([], tools=[{"function": {"name": "x"}}]))

        with pytest.raises(ValueError, match="served as a text env"):
            env.reset(row_index=0)


class TestVerifiersToolEnvTurnEnd:
    def test_tool_turn_ending_the_episode_returns_its_reward(self) -> None:
        env = VerifiersToolEnv(ScriptedBridge([2.5]))

        result = env.step(turn_action([add_call(1, 2)], 0))

        assert (result.done, result.reward) == (True, 2.5)

    def test_answers_fewer_results_than_calls_raise(self) -> None:
        env = VerifiersToolEnv(ScriptedBridge([[]]))

        with pytest.raises(ValueError, match="one tool message per call"):
            env.step(turn_action([add_call(1, 2)], 0))

    def test_prose_answer_returning_feedback(self) -> None:
        env = VerifiersToolEnv(ScriptedBridge([[{"role": "user", "content": "again"}]]))

        result = env.step(CallToolAction(tool_name=PROSE_TOOL, arguments={"text": "x"}))

        assert result_text(result) == "again"
        assert result.done is False

    def test_user_feedback_rejects_non_user_messages(self) -> None:
        env = VerifiersToolEnv(ScriptedBridge([[{"role": "tool", "content": "x"}]]))

        with pytest.raises(ValueError, match="expected user messages"):
            env.step(CallToolAction(tool_name=PROSE_TOOL, arguments={"text": "x"}))


class TestMessageTextAndRewards:
    def test_text_joins_text_parts(self) -> None:
        message = {
            "role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        }

        assert _text(message) == "ab"

    def test_text_rejects_non_text_content(self) -> None:
        with pytest.raises(TypeError, match="non-text user content"):
            _text({"role": "user", "content": 42})

    def test_episode_reward_rejects_a_failed_episode(self) -> None:
        episode = Episode.model_construct(
            ok=False, errors=[Error(type="E", message="m")], traces=[]
        )

        with pytest.raises(RuntimeError, match="verifiers episode failed"):
            _episode_reward(episode)

    def test_episode_reward_rejects_multi_agent_episodes(self) -> None:
        episode = Episode.model_construct(
            ok=True,
            errors=[],
            traces=[Trace.model_construct(), Trace.model_construct()],
        )

        with pytest.raises(ValueError, match="single-agent"):
            _episode_reward(episode)


class TestFailureDetail:
    def test_surfaces_trace_error(self) -> None:
        trace = Trace.model_construct(
            errors=[Error(type="HarnessError", message="uv sync failed")]
        )
        episode = Episode.model_construct(ok=False, errors=[], traces=[trace])

        assert _failure_detail(episode) == (
            "verifiers episode failed: HarnessError: uv sync failed"
        )

    def test_empty_errors_says_so(self) -> None:
        episode = Episode.model_construct(ok=False, errors=[], traces=[])

        assert _failure_detail(episode) == (
            "verifiers episode failed: no error recorded"
        )


class TestSharedLoopAcrossModuleInstances:
    def test_file_reloads_share_one_loop(self) -> None:
        # The env host loads this file once per session without registering
        # it, so each session gets a fresh module object; all of them must
        # still drive verifiers on one process-wide loop.
        path = Path(verifiers_env.__file__)
        instances = []
        for name in ("agilerl_custom_env_probe", "agilerl_custom_env_session"):
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            instances.append(module)

        first, second = instances
        assert first.SHARED_LOOP is second.SHARED_LOOP is verifiers_env.SHARED_LOOP
