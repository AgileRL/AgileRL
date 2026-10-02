# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native tool calling: harness auto-detects the grammar, clients send structured calls."""

from __future__ import annotations

import json
from contextlib import contextmanager
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
    """Char IDs plus tool-call tags, detected as the tag grammar."""

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
    """Char IDs with no tool tags: detection finds no grammar."""

    def convert_tokens_to_ids(self, token: str) -> int | None:
        del token
        return None


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
    """Env client double recording the actions it receives."""

    def __init__(
        self, tools: list[Any] | None = None, mcp_tool: str | None = None
    ) -> None:
        self.actions: list[Any] = []
        self._tools = list(tools or [])
        self.mcp_tool = mcp_tool

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        del seed, row_index
        return "prompt", {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        return "feedback", 0.0, False, False, {"role": "tool"}

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


class TestHarnessToolDetection:
    def test_detects_grammar_from_tokenizer(self) -> None:
        harness = RolloutHarness(
            RecordingClient(), TagTokenizer(), max_turns=1, apply_chat_template=False
        )
        try:
            assert harness._tool_parser is not None
        finally:
            harness.close()

    def test_leaves_parsing_off_without_tags(self) -> None:
        harness = RolloutHarness(
            RecordingClient(), PlainTokenizer(), max_turns=1, apply_chat_template=False
        )
        try:
            assert harness._tool_parser is None
        finally:
            harness.close()


class TestHarnessToolSteps:
    def test_sends_parsed_call_when_tools_advertised(self) -> None:
        client = RecordingClient(tools=[{"name": "add"}])
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

    def test_sends_text_when_no_tools_advertised(self) -> None:
        # Guard: the env expects text, so parsed tags flow through as prose.
        client = RecordingClient()
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
        client = RecordingClient(tools=[{"name": "add"}])
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            bad = [TC] + [ord(c) for c in "{bad}"] + [TC_END]

            # Act
            _, _, info = _step_with_ids(harness, bad)

            # Assert
            assert client.actions == []
            assert info["role"] == "tool"
            assert "not valid JSON" in harness._feedback_texts[-1]
        finally:
            harness.close()

    def test_malformed_tags_flow_as_text_without_tools(self) -> None:
        client = RecordingClient()
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
        client = RecordingClient(tools=[{"name": "add"}])
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            # Act
            _step_with_ids(harness, [ord(c) for c in "just an answer"])

            # Assert
            assert client.actions == []
            assert "No tool call found" in harness._feedback_texts[-1]
        finally:
            harness.close()

    def test_several_calls_error_without_client_hit(self) -> None:
        client = RecordingClient(tools=[{"name": "add"}])
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            both = call_ids("a", {}) + call_ids("b", {})

            # Act
            _step_with_ids(harness, both)

            # Assert
            assert client.actions == []
            assert "one tool call per turn" in harness._feedback_texts[-1]
        finally:
            harness.close()

    def test_mcp_tool_sends_text_even_when_tools_are_listed(self) -> None:
        client = RecordingClient(
            tools=[{"name": "echo_message"}], mcp_tool="echo_message"
        )
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            # Act
            _step_with_ids(harness, [ord(c) for c in "hello"])

            # Assert
            assert client.actions == ["hello"]
        finally:
            harness.close()

    def test_mcp_tool_does_not_parse_native_calls(self) -> None:
        client = RecordingClient(tools=[{"name": "add"}], mcp_tool="echo_message")
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


class TestRemoteToolSteps:
    def test_sends_parsed_call_as_call_tool_action(self) -> None:
        client = RemoteEnvClient("http://stub.invalid")
        client._sync = _StubSync()

        client.step(ok_call("add", {"a": 2, "b": 3}))

        assert client._sync.step_calls[-1] == {
            "type": "call_tool",
            "tool_name": "add",
            "arguments": {"a": 2, "b": 3},
            "metadata": {},
        }

    def test_labels_tool_results_with_tool_role(self) -> None:
        client = RemoteEnvClient("http://stub.invalid")
        client._sync = _StubSync()

        _, _, _, _, info = client.step(ok_call("add", {}))

        assert info["role"] == "tool"

    def test_text_still_sends_as_text(self) -> None:
        client = RemoteEnvClient("http://stub.invalid")
        client._sync = _StubSync()

        client.step("hello")

        assert client._sync.step_calls[-1] == {"message": "hello"}

    def test_malformed_call_never_steps(self) -> None:
        client = RemoteEnvClient("http://stub.invalid")
        client._sync = _StubSync()
        bad = ParsedToolCall(raw="{bad}", status=ToolCallParseStatus.INVALID_JSON)

        with pytest.raises(ValueError, match="never step"):
            client.step(bad)


class TestRemoteToolDiscovery:
    def test_state_tools_answer_without_probe(self) -> None:
        client = RemoteEnvClient("http://stub.invalid")
        client._sync = _StubSync(state={"tools": [{"name": "t"}]})

        assert client.tools == [{"name": "t"}]
        assert client._sync.step_calls == []

    def test_bare_state_lists_tools_from_server(self) -> None:
        client = RemoteEnvClient("http://stub.invalid")
        client._sync = _StubSync(
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
        client = RemoteEnvClient("http://stub.invalid")
        client._sync = _StubSync(state={}, reject_list_tools=True)

        assert client.tools == []

    def test_listing_caches_for_the_session(self) -> None:
        client = RemoteEnvClient("http://stub.invalid")
        client._sync = _StubSync(state={}, tools=[{"name": "add"}])

        _ = client.tools
        _ = client.tools

        probes = [
            call for call in client._sync.step_calls if call.get("type") == "list_tools"
        ]
        assert len(probes) == 1


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


class TestNativeToolsEndToEnd:
    def test_mcp_env_without_grammar_errors_instead_of_text(self) -> None:
        harness = RolloutHarness.local(
            CalculatorMCPEnv(),
            PlainTokenizer(),
            max_turns=2,
            apply_chat_template=False,
        )
        try:
            prompt, _ = harness.reset()
            token_ids = torch.cat(
                [prompt["input_ids"], torch.tensor([[ord(c) for c in "5"]])],
                dim=1,
            )

            # Act
            _, _, _, _, info = harness.step(token_ids)

            # Assert
            assert info["role"] == "tool"
            assert "No tool-call grammar" in harness._feedback_texts[-1]
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


class _StubSync:
    """Minimal sync session double recording step payloads."""

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

    def connect(self) -> None:
        return None

    def reset(self, **kwargs: Any) -> Any:
        del kwargs
        from types import SimpleNamespace

        return SimpleNamespace(observation={"prompt": "hi"}, metadata={})

    def step(self, action: dict[str, Any]) -> Any:
        from types import SimpleNamespace

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

    def state(self) -> dict[str, Any]:
        return self._state

    def close(self) -> None:
        return None
