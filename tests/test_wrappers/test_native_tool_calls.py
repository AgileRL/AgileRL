# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tool calling: the harness parses calls with the model family's parser; clients send them."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from agilerl.llm_envs import RolloutHarness
from agilerl.llm_envs.openenv import InProcessEnvClient, RemoteEnvClient
from agilerl.llm_envs.openenv_server import OpenEnvServer, TextObservation
from agilerl.llm_envs.tool_parsers import ParsedToolCall, ToolCallParseStatus

pytest.importorskip("fastmcp")

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Observation, State

TC, TC_END, THINK_END = 9000, 9001, 9002


class TagTokenizer:
    """Char IDs plus tool-call tags, matched by the tag parser."""

    unk_token_id = -1
    pad_token_id = 0

    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        ids = torch.tensor([[ord(c) for c in texts[0]]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], **_: Any) -> str:
        return "".join(chr(i) if i < 9000 else "" for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return {"<tool_call>": TC, "</tool_call>": TC_END}.get(token)


class PlainTokenizer(TagTokenizer):
    """Char IDs with no tool tags: no parser matches."""

    def convert_tokens_to_ids(self, token: str) -> int | None:
        del token
        return None


class GroupingTemplateTokenizer(TagTokenizer):
    """Template that wraps consecutive tool results in one block, like Qwen's."""

    chat_template = "{{ tools }}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> str:
        rendered = ""
        previous = None
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "tool":
                rendered += ("" if previous == "tool" else "<U>") + f"<R>{content}</R>"
            else:
                rendered += ("</U>" if previous == "tool" else "") + (
                    f"<{role}>{content}</{role}>"
                )
            previous = role
        if previous == "tool":
            rendered += "</U>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def call_ids(name: str, arguments: dict[str, Any]) -> list[int]:
    """One JSON tool call as token IDs."""
    payload = json.dumps({"name": name, "arguments": arguments})
    return [TC] + [ord(c) for c in payload] + [TC_END]


def ok_call(name: str, arguments: dict[str, Any]) -> ParsedToolCall:
    """One valid parsed call."""
    return ParsedToolCall(
        raw=json.dumps({"name": name, "arguments": arguments}),
        name=name,
        arguments=arguments,
        status=ToolCallParseStatus.OK,
    )


class CalculatorMCPEnv(MCPEnvironment):
    """Add with ``add``, then ``submit`` the answer for a reward."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, a: int = 2, b: int = 3) -> None:
        mcp = FastMCP("calc")
        self._a = a
        self._b = b
        self._submitted: int | None = None
        self._state = State()

        @mcp.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @mcp.tool
        def submit(answer: int) -> str:
            """Submit the final answer."""
            self._submitted = int(answer)
            return "submitted"

        super().__init__(mcp)
        self._state = State()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> TextObservation:
        del seed, kwargs
        self._submitted = None
        self._state = State(episode_id=episode_id, step_count=0)
        return TextObservation(
            prompt=f"What is {self._a}+{self._b}? Call add, then submit.",
            done=False,
        )

    def step(
        self,
        action: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        return self._with_submit_reward(action, obs)

    async def step_async(
        self,
        action: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self._with_submit_reward(action, obs)

    def _with_submit_reward(self, action: Any, obs: Observation) -> Observation:
        if isinstance(action, CallToolAction) and action.tool_name == "submit":
            correct = self._submitted == self._a + self._b
            obs.reward = 1.0 if correct else 0.0
            obs.done = True
        self._state.step_count += 1
        return obs

    def _step_impl(self, action: Any) -> Observation:
        msg = "CalculatorMCPEnv only accepts CallToolAction."
        raise ValueError(msg)

    @property
    def state(self) -> State:
        return self._state


class TextEnv:
    """Minimal plain-text env."""

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        del seed
        return "prompt", {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        del action
        return "feedback", 0.0, False, False, {}


class RecordingClient:
    """Env client double recording the actions it receives.

    Each step answers ``result <n>`` with ``reward``; a call to ``terminate_on``
    ends the episode.
    """

    def __init__(
        self,
        *,
        takes_tool_calls: bool = True,
        tools: list[Any] | None = None,
        reward: float = 0.0,
        terminate_on: str | None = None,
    ) -> None:
        self.actions: list[Any] = []
        self._takes_tool_calls = takes_tool_calls
        self._tools = list(tools or [])
        self._reward = reward
        self._terminate_on = terminate_on

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        del seed, row_index
        return "prompt", {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        terminated = (
            isinstance(action, ParsedToolCall) and action.name == self._terminate_on
        )
        return (
            f"result {len(self.actions)}",
            self._reward,
            terminated,
            False,
            {"role": "tool"},
        )

    @property
    def takes_tool_calls(self) -> bool:
        return self._takes_tool_calls

    def close(self) -> None:
        return None

    @property
    def dataset_size(self) -> int:
        return 0

    @property
    def tools(self) -> list[Any]:
        return self._tools

    @property
    def rubric_components(self) -> tuple[str, ...]:
        return ()

    @contextmanager
    def eval_mode(self):
        yield


def _step_with_ids(
    harness: RolloutHarness, gen_ids: list[int]
) -> tuple[float, bool, dict[str, Any]]:
    prompt, _ = harness.reset()
    token_ids = torch.cat([prompt["input_ids"], torch.tensor([gen_ids])], dim=1)
    _, reward, terminated, _, info = harness.step(token_ids)
    return reward, terminated, info


def _transcript_text(harness: RolloutHarness) -> str:
    ids = harness.full_ids
    if ids is None:
        return ""
    return harness.tokenizer.decode(ids[0].tolist())


class TestHarnessToolDetection:
    def test_detects_parser_from_tokenizer(self) -> None:
        harness = RolloutHarness(
            RecordingClient(), TagTokenizer(), max_turns=1, apply_chat_template=False
        )
        try:
            assert harness._tool_parser is not None
        finally:
            harness.close()

    def test_leaves_parsing_off_without_tags(self) -> None:
        harness = RolloutHarness(
            RecordingClient(takes_tool_calls=False),
            PlainTokenizer(),
            max_turns=1,
            apply_chat_template=False,
        )
        try:
            harness.reset()

            assert harness._tool_parser is None
        finally:
            harness.close()

    def test_tool_env_without_parser_fails_at_reset(self) -> None:
        harness = RolloutHarness(
            RecordingClient(), PlainTokenizer(), max_turns=1, apply_chat_template=False
        )
        try:
            with pytest.raises(ValueError, match=r"no parser .* matches"):
                harness.reset()
        finally:
            harness.close()


class TestHarnessToolSteps:
    def test_sends_parsed_call_when_env_takes_tool_calls(self) -> None:
        client = RecordingClient()
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            # Act
            _step_with_ids(harness, call_ids("add", {"a": 2}))

            # Assert
            assert len(client.actions) == 1
            sent = client.actions[0]
            assert isinstance(sent, ParsedToolCall)
            assert (sent.name, sent.arguments) == ("add", {"a": 2})
        finally:
            harness.close()

    def test_sends_text_to_text_env_that_lists_tools(self) -> None:
        # A plain-text env renders its tools into the prompt but steps on text.
        client = RecordingClient(takes_tool_calls=False, tools=[{"name": "add"}])
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            # Act
            _step_with_ids(harness, call_ids("add", {"a": 2}))

            # Assert
            assert len(client.actions) == 1
            assert isinstance(client.actions[0], str)
        finally:
            harness.close()

    def test_malformed_call_errors_without_client_hit(self) -> None:
        client = RecordingClient()
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            bad = [TC] + [ord(c) for c in "{bad}"] + [TC_END]

            # Act
            reward, _, info = _step_with_ids(harness, bad)

            # Assert
            assert client.actions == []
            assert reward == 0.0
            assert info["role"] == "tool"
            assert "not valid JSON" in _transcript_text(harness)
        finally:
            harness.close()

    def test_malformed_call_scores_min_reward(self) -> None:
        client = RecordingClient()
        harness = RolloutHarness(
            client,
            TagTokenizer(),
            max_turns=2,
            apply_chat_template=False,
            min_reward=-1.0,
        )
        try:
            bad = [TC] + [ord(c) for c in "{bad}"] + [TC_END]

            # Act
            reward, _, _ = _step_with_ids(harness, bad)

            # Assert
            assert reward == -1.0
            assert harness.turn_rewards == [-1.0]
        finally:
            harness.close()

    def test_malformed_tags_flow_as_text_to_text_envs(self) -> None:
        client = RecordingClient(takes_tool_calls=False)
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            bad = [TC] + [ord(c) for c in "{bad}"] + [TC_END]

            # Act
            _step_with_ids(harness, bad)

            # Assert
            assert len(client.actions) == 1
            assert isinstance(client.actions[0], str)
        finally:
            harness.close()

    def test_prose_without_calls_errors_for_tool_envs(self) -> None:
        client = RecordingClient()
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            # Act
            _step_with_ids(harness, [ord(c) for c in "just an answer"])

            # Assert
            assert client.actions == []
            assert "No tool call found" in _transcript_text(harness)
        finally:
            harness.close()

    def test_runs_several_calls_in_order_within_one_turn(self) -> None:
        client = RecordingClient(reward=0.25)
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            both = call_ids("a", {"x": 1}) + call_ids("b", {"y": 2})

            # Act
            reward, terminated, _ = _step_with_ids(harness, both)

            # Assert
            assert [(call.name, call.arguments) for call in client.actions] == [
                ("a", {"x": 1}),
                ("b", {"y": 2}),
            ]
            assert (reward, terminated) == (0.5, False)
            assert harness.turn_rewards == [0.5]
            assert "result 1\nresult 2" in _transcript_text(harness)
        finally:
            harness.close()

    def test_stops_running_calls_once_the_episode_ends(self) -> None:
        client = RecordingClient(reward=1.0, terminate_on="a")
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            both = call_ids("a", {}) + call_ids("b", {})

            # Act
            reward, terminated, _ = _step_with_ids(harness, both)

            # Assert
            assert [call.name for call in client.actions] == ["a"]
            assert (reward, terminated) == (1.0, True)
        finally:
            harness.close()

    def test_one_malformed_call_runs_none_of_the_turn(self) -> None:
        client = RecordingClient()
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            bad = [TC] + [ord(c) for c in "{bad}"] + [TC_END]

            # Act
            _step_with_ids(harness, call_ids("a", {}) + bad)

            # Assert
            assert client.actions == []
            assert "No calls ran" in _transcript_text(harness)
        finally:
            harness.close()


class TestHarnessFeedbackFrame:
    def test_frames_each_tool_result_as_its_own_message(self) -> None:
        harness = RolloutHarness(RecordingClient(), GroupingTemplateTokenizer())
        try:
            # Act
            ids = harness._tokenize_feedback(
                [
                    {"role": "tool", "content": "a"},
                    {"role": "tool", "content": "b"},
                ]
            )

            # Assert
            assert harness.tokenizer.decode(ids[0].tolist()) == (
                "</assistant><U><R>a</R><R>b</R></U><assistant>"
            )
        finally:
            harness.close()


class TestRemoteToolSteps:
    def test_sends_parsed_call_as_call_tool_action(self) -> None:
        client, session = _stub_remote()

        client.step(ok_call("add", {"a": 2, "b": 3}))

        assert session.step_calls[-1] == {
            "type": "call_tool",
            "tool_name": "add",
            "arguments": {"a": 2, "b": 3},
            "metadata": {},
        }

    def test_labels_tool_results_with_tool_role(self) -> None:
        client, _session = _stub_remote()

        _, _, _, _, info = client.step(ok_call("add", {}))

        assert info["role"] == "tool"

    def test_text_still_sends_as_text(self) -> None:
        client, session = _stub_remote()

        client.step("hello")

        assert session.step_calls[-1] == {"message": "hello"}

    def test_malformed_call_never_steps(self) -> None:
        client, _session = _stub_remote()
        bad = ParsedToolCall(raw="{bad}", status=ToolCallParseStatus.INVALID_JSON)

        with pytest.raises(ValueError, match="never step"):
            client.step(bad)


class TestRemoteToolDiscovery:
    def test_state_tools_answer_without_probe(self) -> None:
        client, session = _stub_remote(state={"tools": [{"name": "t"}]})

        assert client.tools == [{"name": "t"}]
        assert session.step_calls == []

    def test_bare_state_lists_tools_from_server(self) -> None:
        client, _session = _stub_remote(
            state={},
            tools=[{"name": "add", "description": "Add.", "input_schema": {}}],
        )

        assert client.tools == [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add.",
                    "parameters": {},
                },
            }
        ]

    def test_rejected_listing_means_no_tools(self) -> None:
        client, _session = _stub_remote(state={}, reject_list_tools=True)

        assert client.tools == []

    def test_listing_caches_for_the_session(self) -> None:
        client, session = _stub_remote(state={}, tools=[{"name": "add"}])

        _ = client.tools
        _ = client.tools

        probes = [
            call for call in session.step_calls if call.get("type") == "list_tools"
        ]
        assert len(probes) == 1


class TestRemoteEnvClientTakesToolCallsProperty:
    def test_true_for_an_env_that_lists_tools(self) -> None:
        client, _session = _stub_remote(state={}, tools=[{"name": "add"}])

        assert client.takes_tool_calls is True

    def test_false_for_a_text_env_with_state_tools(self) -> None:
        client, _session = _stub_remote(state={"tools": [{"name": "add"}]})

        assert client.takes_tool_calls is False

    def test_false_when_mcp_tool_wraps_the_text(self) -> None:
        client, _session = _stub_remote(
            state={}, tools=[{"name": "echo"}], mcp_tool="echo"
        )

        assert client.takes_tool_calls is False

    def test_false_when_the_server_rejects_the_listing(self) -> None:
        client, _session = _stub_remote(state={}, reject_list_tools=True)

        assert client.takes_tool_calls is False


class TestInProcessToolBackend:
    def test_drives_mcp_env_through_parsed_calls(self) -> None:
        client = InProcessEnvClient(CalculatorMCPEnv())

        payload, _ = client.reset()
        assert "2+3" in payload["prompt"]
        _, reward, terminated, _, info = client.step(ok_call("add", {"a": 2, "b": 3}))

        assert (reward, terminated) == (0.0, False)
        assert info["role"] == "tool"

    def test_submit_ends_the_episode_with_reward(self) -> None:
        client = InProcessEnvClient(CalculatorMCPEnv())
        client.reset()

        _, reward, terminated, _, _ = client.step(ok_call("submit", {"answer": 5}))

        assert (reward, terminated) == (1.0, True)

    def test_text_to_mcp_backend_raises(self) -> None:
        client = InProcessEnvClient(CalculatorMCPEnv())
        client.reset()

        with pytest.raises(TypeError, match="parsed tool calls"):
            client.step("5")

    def test_parsed_call_to_text_env_raises(self) -> None:
        client = InProcessEnvClient(TextEnv())

        with pytest.raises(ValueError, match="takes text"):
            client.step(ok_call("add", {}))

    def test_lists_mcp_tools(self) -> None:
        client = InProcessEnvClient(CalculatorMCPEnv())

        names = [tool["function"]["name"] for tool in client.tools]

        assert sorted(names) == ["add", "submit"]

    def test_mcp_backend_takes_tool_calls(self) -> None:
        assert InProcessEnvClient(CalculatorMCPEnv()).takes_tool_calls is True

    def test_text_backend_takes_text(self) -> None:
        assert InProcessEnvClient(TextEnv()).takes_tool_calls is False


class TestNativeToolsEndToEnd:
    def test_mcp_env_without_parser_fails_at_reset(self) -> None:
        harness = RolloutHarness.local(
            CalculatorMCPEnv(),
            PlainTokenizer(),
            max_turns=2,
            apply_chat_template=False,
        )
        try:
            with pytest.raises(ValueError, match="takes tool calls"):
                harness.reset()
        finally:
            harness.close()

    def test_add_then_submit_in_one_turn_scores_the_episode(self) -> None:
        harness = RolloutHarness.local(
            CalculatorMCPEnv(), TagTokenizer(), max_turns=3, apply_chat_template=False
        )
        try:
            prompt, _ = harness.reset()
            calls = call_ids("add", {"a": 2, "b": 3}) + call_ids(
                "submit", {"answer": 5}
            )
            token_ids = torch.cat([prompt["input_ids"], torch.tensor([calls])], dim=1)

            # Act
            _, reward, terminated, _, _ = harness.step(token_ids)

            # Assert
            assert (reward, terminated) == (1.0, True)
            assert harness.turn_rewards == [1.0]
        finally:
            harness.close()

    def test_harness_trains_through_a_real_tool_loop(self) -> None:
        tokenizer = TagTokenizer()
        harness = RolloutHarness.local(
            CalculatorMCPEnv(), tokenizer, max_turns=3, apply_chat_template=False
        )
        try:
            prompt, _ = harness.reset()
            token_ids = torch.cat(
                [
                    prompt["input_ids"],
                    torch.tensor([call_ids("add", {"a": 2, "b": 3})]),
                ],
                dim=1,
            )

            # Act
            _, reward, terminated, _, info = harness.step(token_ids)

            # Assert
            assert (reward, terminated) == (0.0, False)
            assert info["role"] == "tool"
            assert harness.turn_rewards == [0.0]
        finally:
            harness.close()

    def test_remote_client_drives_hosted_mcp_env(self) -> None:
        with OpenEnvServer(CalculatorMCPEnv()) as server:
            client = RemoteEnvClient(server.base_url)
            try:
                names = [tool["function"]["name"] for tool in client.tools]
                assert sorted(names) == ["add", "submit"]

                _, reward, terminated, _, info = client.step(
                    ok_call("submit", {"answer": 5})
                )

                assert (reward, terminated) == (1.0, True)
                assert info["role"] == "tool"
            finally:
                client.close()

    def test_harness_drives_hosted_mcp_env(self) -> None:
        with OpenEnvServer(CalculatorMCPEnv()) as server:
            harness = RolloutHarness(
                server.base_url, TagTokenizer(), max_turns=3, apply_chat_template=False
            )
            try:
                prompt, _ = harness.reset()
                token_ids = torch.cat(
                    [
                        prompt["input_ids"],
                        torch.tensor([call_ids("submit", {"answer": 5})]),
                    ],
                    dim=1,
                )

                # Act
                _, reward, terminated, _, info = harness.step(token_ids)

                # Assert
                assert (reward, terminated) == (1.0, True)
                assert info["role"] == "tool"
            finally:
                harness.close()


class _StubSession:
    """Async session double recording step payloads."""

    def __init__(
        self,
        state: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        reject_list_tools: bool = False,
    ) -> None:
        self.step_calls: list[dict[str, Any]] = []
        self._state = {"tools": []} if state is None else state
        self._tools = tools or []
        self._reject_list_tools = reject_list_tools

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def reset(self, **kwargs: Any) -> Any:
        del kwargs
        return SimpleNamespace(observation={"prompt": "hi"}, metadata={})

    async def step(self, action: dict[str, Any]) -> Any:
        self.step_calls.append(action)
        if action.get("type") == "list_tools":
            if self._reject_list_tools:
                msg = "Server error: bad action (code: VALIDATION_ERROR)"
                raise RuntimeError(msg)
            return SimpleNamespace(
                observation={"tools": self._tools},
                reward=0.0,
                done=False,
                metadata={},
            )
        return SimpleNamespace(
            observation={"result": {"data": "5"}},
            reward=0.0,
            done=False,
            metadata={},
        )

    async def state(self) -> dict[str, Any]:
        return self._state


def _stub_remote(
    *,
    state: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    reject_list_tools: bool = False,
    mcp_tool: str | None = None,
) -> tuple[RemoteEnvClient, _StubSession]:
    """Remote client whose transport is a :class:`_StubSession` (no network)."""
    session = _StubSession(
        state=state, tools=tools, reject_list_tools=reject_list_tools
    )
    client = RemoteEnvClient("http://stub.invalid", mcp_tool=mcp_tool)
    client._session = session
    return client, session
