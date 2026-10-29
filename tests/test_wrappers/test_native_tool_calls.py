# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tool calling: the harness parses calls with the model family's parser; clients send them."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from agilerl.llm_envs import RolloutHarness
from agilerl.llm_envs.chat_formats import (
    ParsedToolCall,
    ToolCallParseStatus,
    ToolCallTagFormat,
)
from agilerl.llm_envs.harness import TOOL_CALL_ERROR_INFO_KEY, TOOL_CALLS_INFO_KEY
from agilerl.llm_envs.openenv import InProcessEnvClient, RemoteEnvClient
from agilerl.llm_envs.openenv_server import OpenEnvServer, TextObservation

pytest.importorskip("fastmcp")

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Observation, State

TC, TC_END, THINK_END = 9000, 9001, 9002
MSTART, MEND, MRESP_START, MRESP_END = 9200, 9201, 9202, 9203


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
        tokenize: bool = False,
        **_: Any,
    ) -> str | list[int]:
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
        if tokenize:
            return self.encode(rendered)
        return rendered


def call_ids(name: str, arguments: dict[str, Any]) -> list[int]:
    """One JSON tool call as token IDs."""
    payload = json.dumps({"name": name, "arguments": arguments})
    return [TC] + [ord(c) for c in payload] + [TC_END]


class GemmaTemplateTokenizer(TagTokenizer):
    """Gemma tag IDs plus a template that skips tool turns, like the real one."""

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return {
            "<|tool_call>": MSTART,
            "<tool_call|>": MEND,
            "<|tool_response>": MRESP_START,
            "<tool_response|>": MRESP_END,
        }.get(token)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **_: Any,
    ) -> str | list[int]:
        if any(message["role"] == "tool" for message in messages):
            msg = "Gemma template skips standalone tool messages"
            raise ValueError(msg)
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<model>"
        if tokenize:
            return self.encode(rendered)
        return rendered


HARMONY_TAGS = {
    "<|start|>": 9300,
    "<|message|>": 9301,
    "<|call|>": 9302,
    "<|end|>": 9303,
    "<|return|>": 9304,
    "<|channel|>": 9305,
}
HARMONY_TAG_RE = re.compile("|".join(re.escape(tag) for tag in HARMONY_TAGS))


class HarmonyTemplateTokenizer(TagTokenizer):
    """gpt-oss tags as special IDs plus a template that rejects a bare tool message, like the real one."""

    @property
    def all_special_ids(self) -> list[int]:
        return list(HARMONY_TAGS.values())

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return HARMONY_TAGS.get(token)

    def encode(self, text: str, **_: Any) -> list[int]:
        ids: list[int] = []
        position = 0
        for match in HARMONY_TAG_RE.finditer(text):
            ids += [ord(c) for c in text[position : match.start()]]
            ids.append(HARMONY_TAGS[match.group()])
            position = match.end()
        return ids + [ord(c) for c in text[position:]]

    def decode(self, ids: list[int], **_: Any) -> str:
        reverse = {value: key for key, value in HARMONY_TAGS.items()}
        return "".join(reverse.get(i, chr(i)) for i in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **_: Any,
    ) -> str | list[int]:
        if any(message["role"] == "tool" for message in messages):
            msg = "Message has tool role, but there was no previous assistant message with a tool call!"
            raise ValueError(msg)
        rendered = "".join(
            f"<|start|>{message['role']}<|message|>{message['content']}<|end|>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|start|>assistant"
        if tokenize:
            return self.encode(rendered)
        return rendered


def gemma_call_ids(name: str, args: str) -> list[int]:
    """One Gemma call as token IDs."""
    return [MSTART, *[ord(c) for c in f"call:{name}{args}"], MEND]


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


class AsyncScoredCalculatorEnv(CalculatorMCPEnv):
    """Scores only in ``step_async``, the method the OpenEnv server awaits."""

    def step(
        self,
        action: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        return MCPEnvironment.step(self, action, timeout_s=timeout_s, **kwargs)


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
        answer_tool: str | None = None,
    ) -> None:
        self.actions: list[Any] = []
        self._takes_tool_calls = takes_tool_calls
        self._tools = list(tools or [])
        self._reward = reward
        self._terminate_on = terminate_on
        self._answer_tool = answer_tool

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

    @property
    def answer_tool(self) -> str | None:
        return self._answer_tool

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
    def test_detects_the_chat_format_from_the_tokenizer(self) -> None:
        harness = RolloutHarness(
            RecordingClient(), TagTokenizer(), max_turns=1, apply_chat_template=False
        )
        try:
            assert harness._chat_format is not None
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

            assert harness._chat_format is None
        finally:
            harness.close()

    def test_tool_env_without_a_chat_format_fails_at_reset(self) -> None:
        harness = RolloutHarness(
            RecordingClient(), PlainTokenizer(), max_turns=1, apply_chat_template=False
        )
        try:
            with pytest.raises(ValueError, match=r"no chat format .* matches"):
                harness.reset()
        finally:
            harness.close()

    def test_named_chat_format_is_used(self) -> None:
        harness = RolloutHarness(
            RecordingClient(),
            TagTokenizer(),
            max_turns=1,
            apply_chat_template=False,
            chat_format="tool_call",
        )
        try:
            assert isinstance(harness._chat_format, ToolCallTagFormat)
        finally:
            harness.close()

    def test_unknown_chat_format_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chat_format 'llama'"):
            RolloutHarness(
                RecordingClient(),
                TagTokenizer(),
                apply_chat_template=False,
                chat_format="llama",
            )

    def test_named_chat_format_needs_its_tokens_in_the_vocab(self) -> None:
        with pytest.raises(ValueError, match="tokenizer vocab"):
            RolloutHarness(
                RecordingClient(),
                PlainTokenizer(),
                apply_chat_template=False,
                chat_format="tool_call",
            )


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

    def test_prose_goes_to_answer_tool_when_set(self) -> None:
        client = RecordingClient(answer_tool="answer")
        harness = RolloutHarness(
            client, TagTokenizer(), max_turns=2, apply_chat_template=False
        )
        try:
            # Act
            _step_with_ids(harness, [ord(c) for c in "just an answer"])

            # Assert
            assert client.actions == ["just an answer"]
            assert "No tool call found" not in _transcript_text(harness)
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

    def test_kept_turn_replay_keeps_each_tool_result_as_its_own_message(self) -> None:
        class LongPromptClient(RecordingClient):
            def reset(
                self, seed: int | None = None, *, row_index: int | None = None
            ) -> tuple[str, dict[str, Any]]:
                del seed, row_index
                return "P" * 400, {}

        harness = RolloutHarness(
            LongPromptClient(),
            GroupingTemplateTokenizer(),
            max_turns=3,
            segment_prompt_tokens=500,
            restart_keep_turns=1,
        )
        try:
            prompt, _ = harness.reset()
            both = call_ids("a", {"x": 1}) + call_ids("b", {"y": 2})

            next_prompt, *_ = harness.step(
                torch.cat([prompt["input_ids"], torch.tensor([both])], dim=1)
            )

            decoded = harness.tokenizer.decode(next_prompt["input_ids"][0].tolist())
            assert "Previous actions:" in decoded
            assert "<R>result 1</R><R>result 2</R>" in decoded
            assert "<R>result 1\n\nresult 2</R>" not in decoded
        finally:
            harness.close()


class TestGemmaToolResponses:
    def test_tool_result_renders_as_response_block_without_template(self) -> None:
        client = RecordingClient()
        harness = RolloutHarness(client, GemmaTemplateTokenizer(), max_turns=2)
        try:
            prompt, _ = harness.reset()
            gen = gemma_call_ids("guess", '{message:<|"|>7<|"|>}')
            token_ids = torch.cat([prompt["input_ids"], torch.tensor([gen])], dim=1)

            # Act
            harness.step(token_ids)

            # Assert
            assert client.actions[0].name == "guess"
            assert (
                '<|tool_response>response:guess{value:<|"|>result 1<|"|>}'
                "<tool_response|>" in _transcript_text(harness)
            )
        finally:
            harness.close()

    def test_prose_error_renders_as_unknown_response(self) -> None:
        client = RecordingClient()
        harness = RolloutHarness(client, GemmaTemplateTokenizer(), max_turns=2)
        try:
            prompt, _ = harness.reset()
            prose = [ord(c) for c in "just an answer"]
            token_ids = torch.cat([prompt["input_ids"], torch.tensor([prose])], dim=1)

            # Act
            harness.step(token_ids)

            # Assert
            assert client.actions == []
            assert "response:unknown{" in _transcript_text(harness)
        finally:
            harness.close()


class TestHarmonyToolResponses:
    def test_tool_result_follows_the_call_without_a_template_frame(self) -> None:
        # Arrange
        client = RecordingClient()
        tokenizer = HarmonyTemplateTokenizer()
        harness = RolloutHarness(client, tokenizer, max_turns=2)
        try:
            prompt, _ = harness.reset()
            gen = tokenizer.encode(
                "<|channel|>analysis<|message|>Add them.<|end|>"
                "<|start|>assistant to=functions.add<|channel|>commentary json"
                '<|message|>{"a": 1}<|call|>'
            )

            # Act
            harness.step(torch.cat([prompt["input_ids"], torch.tensor([gen])], dim=1))

            # Assert
            assert [action.name for action in client.actions] == ["add"]
            assert _transcript_text(harness).endswith(
                '{"a": 1}<|call|>'
                "<|start|>functions.add to=assistant<|channel|>commentary"
                '<|message|>"result 1"<|end|><|start|>assistant'
            )
        finally:
            harness.close()


class TestHarmonyFeedbackTerminator:
    def test_user_feedback_after_a_return_drops_the_frame_end(self) -> None:
        tokenizer = HarmonyTemplateTokenizer()
        harness = RolloutHarness(RecordingClient(takes_tool_calls=False), tokenizer)
        try:
            # Act
            ids = harness._text_feedback_ids(
                [{"role": "user", "content": "again"}], HARMONY_TAGS["<|return|>"]
            )

            # Assert
            assert tokenizer.decode(ids[0].tolist()) == (
                "<|start|>user<|message|>again<|end|><|start|>assistant"
            )
        finally:
            harness.close()

    def test_truncated_turn_keeps_the_frame_end(self) -> None:
        tokenizer = HarmonyTemplateTokenizer()
        harness = RolloutHarness(RecordingClient(takes_tool_calls=False), tokenizer)
        try:
            # Act
            ids = harness._text_feedback_ids(
                [{"role": "user", "content": "again"}], ord("x")
            )

            # Assert
            assert tokenizer.decode(ids[0].tolist()).startswith("<|end|><|start|>user")
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

    def test_non_validation_listing_error_propagates(self) -> None:
        client = RemoteEnvClient("http://stub.invalid")
        client._session = _ExplodingSession(state={})

        with pytest.raises(RuntimeError, match="boom"):
            _ = client.tools

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

    def test_answer_tool_keeps_tool_calls_on(self) -> None:
        client, _session = _stub_remote(
            state={}, tools=[{"name": "answer"}], answer_tool="answer"
        )

        assert client.takes_tool_calls is True
        assert client.answer_tool == "answer"

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

    def test_env_scored_in_step_async_is_scored(self) -> None:
        client = InProcessEnvClient(AsyncScoredCalculatorEnv())
        client.reset()

        _, reward, terminated, _, _ = client.step(ok_call("submit", {"answer": 5}))

        assert (reward, terminated) == (1.0, True)

    def test_text_to_mcp_backend_raises(self) -> None:
        client = InProcessEnvClient(CalculatorMCPEnv())
        client.reset()

        with pytest.raises(TypeError, match="parsed tool calls"):
            client.step("5")

    def test_text_goes_to_answer_tool(self) -> None:
        client = InProcessEnvClient(
            CalculatorMCPEnv(), answer_tool="submit", action_field="answer"
        )
        client.reset()

        _, reward, terminated, _, _ = client.step("5")

        assert (reward, terminated) == (1.0, True)

    def test_parsed_call_to_text_env_raises(self) -> None:
        client = InProcessEnvClient(TextEnv())

        with pytest.raises(ValueError, match="takes text"):
            client.step(ok_call("add", {}))

    def test_malformed_call_never_steps(self) -> None:
        client = InProcessEnvClient(CalculatorMCPEnv())
        client.reset()

        with pytest.raises(ValueError, match="malformed"):
            client.step(ParsedToolCall(raw="add(?)", name=None, arguments=None))

    def test_lists_mcp_tools(self) -> None:
        client = InProcessEnvClient(CalculatorMCPEnv())

        names = [tool["function"]["name"] for tool in client.tools]

        assert sorted(names) == ["add", "submit"]

    def test_mcp_backend_takes_tool_calls(self) -> None:
        assert InProcessEnvClient(CalculatorMCPEnv()).takes_tool_calls is True

    def test_text_backend_takes_text(self) -> None:
        assert InProcessEnvClient(TextEnv()).takes_tool_calls is False


class TestNativeToolsEndToEnd:
    def test_mcp_env_without_a_chat_format_fails_at_reset(self) -> None:
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

    def test_step_info_counts_valid_calls(self) -> None:
        harness = RolloutHarness.local(
            CalculatorMCPEnv(), TagTokenizer(), max_turns=3, apply_chat_template=False
        )
        try:
            prompt, _ = harness.reset()
            calls = call_ids("add", {"a": 1, "b": 1}) + call_ids(
                "add", {"a": 2, "b": 3}
            )
            token_ids = torch.cat([prompt["input_ids"], torch.tensor([calls])], dim=1)

            # Act
            *_, info = harness.step(token_ids)

            # Assert
            assert info[TOOL_CALLS_INFO_KEY] == 2
            assert TOOL_CALL_ERROR_INFO_KEY not in info
        finally:
            harness.close()

    def test_step_info_flags_a_rejected_turn(self) -> None:
        harness = RolloutHarness.local(
            CalculatorMCPEnv(), TagTokenizer(), max_turns=3, apply_chat_template=False
        )
        try:
            prompt, _ = harness.reset()
            malformed = [TC, *(ord(c) for c in "{oops"), TC_END]
            token_ids = torch.cat(
                [prompt["input_ids"], torch.tensor([malformed])], dim=1
            )

            # Act
            *_, info = harness.step(token_ids)

            # Assert
            assert info[TOOL_CALLS_INFO_KEY] == 0
            assert info[TOOL_CALL_ERROR_INFO_KEY] is True
        finally:
            harness.close()

    def test_text_env_step_info_has_no_tool_counts(self) -> None:
        harness = RolloutHarness.local(
            TextEnv(), TagTokenizer(), max_turns=3, apply_chat_template=False
        )
        try:
            prompt, _ = harness.reset()
            token_ids = torch.cat(
                [prompt["input_ids"], torch.tensor([[ord("x")]])], dim=1
            )

            # Act
            *_, info = harness.step(token_ids)

            # Assert
            assert TOOL_CALLS_INFO_KEY not in info
            assert TOOL_CALL_ERROR_INFO_KEY not in info
        finally:
            harness.close()

    def test_prose_answer_scores_through_answer_tool(self) -> None:
        harness = RolloutHarness.local(
            CalculatorMCPEnv(),
            TagTokenizer(),
            max_turns=3,
            apply_chat_template=False,
            answer_tool="submit",
            action_field="answer",
        )
        try:
            prompt, _ = harness.reset()
            prose = [ord(c) for c in "5"]
            token_ids = torch.cat([prompt["input_ids"], torch.tensor([prose])], dim=1)

            # Act
            _, reward, terminated, _, _ = harness.step(token_ids)

            # Assert
            assert (reward, terminated) == (1.0, True)
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


class _ExplodingSession(_StubSession):
    """Session double whose steps fail outside validation."""

    async def step(self, action: dict[str, Any]) -> Any:
        del action
        msg = "boom"
        raise RuntimeError(msg)


def _stub_remote(
    *,
    state: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    reject_list_tools: bool = False,
    answer_tool: str | None = None,
) -> tuple[RemoteEnvClient, _StubSession]:
    """Remote client whose transport is a :class:`_StubSession` (no network)."""
    session = _StubSession(
        state=state, tools=tools, reject_list_tools=reject_list_tools
    )
    client = RemoteEnvClient("http://stub.invalid", answer_tool=answer_tool)
    client._session = session
    return client, session
