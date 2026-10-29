# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Feedback frames the harness splices match each model family's real chat template."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import huggingface_hub.constants as hub_constants
import pytest
from huggingface_hub import get_token, snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agilerl.architectures import family_runtime
from agilerl.llm_envs.harness import RolloutHarness
from tests.helpers.rollout_doubles import FakeEnvClient

# Tokenizer files come from the Hub, and Gemma 4 is gated. CI runs these on the
# Linux CPU shard, which has HF_TOKEN and a Hub cache.
if get_token() is None:
    pytest.skip("needs a Hugging Face Hub token", allow_module_level=True)

# One checkpoint per chat template family, pinned so an upstream template edit
# shows up as a revision bump here, not as a red run.
MODEL_REVISIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
    "ibm-granite/granite-4.0-micro": "56111ae135df9c53a78c99028e7bc24035a9e979",
    "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89",
    "google/gemma-4-E4B-it": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
}
TOKENIZER_FILES = [
    "tokenizer*",
    "special_tokens_map.json",
    "chat_template*",
    "*.jinja",
    "generation_config.json",
]
FEEDBACK_ROLES = (("user",), ("tool",), ("system",), ("tool", "tool"))
UNSUPPORTED_FEEDBACK_ROLES = {
    (
        "Qwen/Qwen3.5-2B",
        "system",
    ): "template rejects a system message after the first turn",
    (
        "openai/gpt-oss-20b",
        "system",
    ): "template renders no mid-conversation system message",
}
GPT_OSS = "openai/gpt-oss-20b"
ADD_CALL = {"type": "function", "function": {"name": "add", "arguments": {"a": 1}}}


def load_tokenizer(model_id: str) -> tuple[PreTrainedTokenizerBase, Path]:
    """Tokenizer files at the pinned revision."""
    # tests/conftest.py forces the Hub offline; this module downloads on purpose.
    with patch.object(hub_constants, "HF_HUB_OFFLINE", False):
        path = Path(
            snapshot_download(
                model_id,
                revision=MODEL_REVISIONS[model_id],
                allow_patterns=TOKENIZER_FILES,
            )
        )
    return AutoTokenizer.from_pretrained(path), path


def rendered_ids(
    tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, Any]]
) -> list[int]:
    """Template token IDs for ``messages`` with the generation prompt."""
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return list(rendered["input_ids"] if isinstance(rendered, Mapping) else rendered)


def turn_end_ids(tokenizer: PreTrainedTokenizerBase, path: Path) -> set[int]:
    """IDs that stop generation: the generation config's EOS list plus the tokenizer's."""
    config_path = path / "generation_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    eos = config.get("eos_token_id", tokenizer.eos_token_id)
    return {*(eos if isinstance(eos, list) else [eos]), tokenizer.eos_token_id}


def conversation(model_id: str, roles: tuple[str, ...]) -> list[dict[str, Any]]:
    """A user question, the assistant's reply, then one feedback message per role.

    Tool feedback follows an assistant turn holding one tool call per result.
    """
    assistant: dict[str, Any] = {"role": "assistant", "content": "4"}
    tool_results = roles.count("tool")
    if tool_results:
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [ADD_CALL] * tool_results,
        }
        if model_id == GPT_OSS:
            assistant["thinking"] = "Add."
    feedback = [
        {"role": role, "content": f"Try again {i}."}
        | ({"name": "add"} if role == "tool" else {})
        for i, role in enumerate(roles)
    ]
    return [{"role": "user", "content": "What is 2+2?"}, assistant, *feedback]


def harness_for(tokenizer: PreTrainedTokenizerBase, model_id: str) -> RolloutHarness:
    """A two-turn harness over ``tokenizer`` with the family's catalog tool parser."""
    return RolloutHarness(
        FakeEnvClient(),
        tokenizer,
        max_turns=2,
        chat_format=family_runtime(model_id).chat_format,
    )


def assistant_turn_end(
    model_id: str, roles: tuple[str, ...]
) -> tuple[PreTrainedTokenizerBase, list[int], int, int]:
    """The template render of ``conversation`` and where its assistant turn ends.

    :return: The tokenizer, the rendered ids, the index of the assistant turn's
        last token, and the token the model samples there.
    """
    tokenizer, path = load_tokenizer(model_id)
    expected = rendered_ids(tokenizer, conversation(model_id, roles))
    prompt = rendered_ids(tokenizer, conversation(model_id, roles)[:1])
    # A re-render may drop the prompt's reasoning prefill, so the assistant
    # turn starts where the two renders part.
    turn_start = next(
        (i for i, (a, b) in enumerate(zip(prompt, expected, strict=False)) if a != b),
        len(prompt),
    )
    ends = turn_end_ids(tokenizer, path)
    gpt_oss_answer = model_id == GPT_OSS and "tool" not in roles
    if gpt_oss_answer:
        # History closes a final answer with <|end|>; the model samples <|return|>.
        ends = {tokenizer.convert_tokens_to_ids("<|end|>")}
    turn_end = next(i for i in range(turn_start, len(expected)) if expected[i] in ends)
    sampled_end = (
        tokenizer.convert_tokens_to_ids("<|return|>")
        if gpt_oss_answer
        else expected[turn_end]
    )
    return tokenizer, expected, turn_end, sampled_end


def case_id(model_id: str, roles: tuple[str, ...]) -> str:
    return f"{model_id}-{'+'.join(roles)}"


def supported_cases() -> list[Any]:
    return [
        pytest.param(model_id, roles, id=case_id(model_id, roles))
        for model_id in MODEL_REVISIONS
        for roles in FEEDBACK_ROLES
        if not any((model_id, role) in UNSUPPORTED_FEEDBACK_ROLES for role in roles)
    ]


class TestRolloutHarnessTextFeedbackIds:
    @pytest.mark.parametrize(("model_id", "roles"), supported_cases())
    def test_frame_matches_the_template_after_the_turn_end(
        self, model_id: str, roles: tuple[str, ...]
    ) -> None:
        # Arrange
        tokenizer, expected, turn_end, sampled_end = assistant_turn_end(model_id, roles)
        harness = harness_for(tokenizer, model_id)

        # Act
        frame = harness._text_feedback_ids(
            conversation(model_id, roles)[2:], sampled_end
        )

        # Assert
        assert tokenizer.decode(frame[0].tolist()) == tokenizer.decode(
            expected[turn_end + 1 :]
        )
        assert frame[0].tolist() == expected[turn_end + 1 :]

    def test_gpt_oss_answer_keeps_the_sampled_return_where_history_has_end(
        self,
    ) -> None:
        """The one token the transcript and a template re-render disagree on."""
        # Arrange
        tokenizer, expected, turn_end, sampled_end = assistant_turn_end(
            GPT_OSS, ("user",)
        )
        harness = harness_for(tokenizer, GPT_OSS)

        # Act
        frame = harness._text_feedback_ids(
            conversation(GPT_OSS, ("user",))[2:], sampled_end
        )
        transcript = [*expected[:turn_end], sampled_end, *frame[0].tolist()]

        # Assert
        assert len(transcript) == len(expected)
        diffs = [
            i
            for i, (a, b) in enumerate(zip(transcript, expected, strict=True))
            if a != b
        ]
        assert diffs == [turn_end]
        assert tokenizer.convert_ids_to_tokens(transcript[turn_end]) == "<|return|>"
        assert tokenizer.convert_ids_to_tokens(expected[turn_end]) == "<|end|>"

    @pytest.mark.parametrize(
        ("model_id", "role"),
        [
            pytest.param(model_id, role, id=f"{model_id}-{role}")
            for model_id, role in UNSUPPORTED_FEEDBACK_ROLES
        ],
    )
    def test_role_the_template_cannot_frame_raises(
        self, model_id: str, role: str
    ) -> None:
        # Arrange
        tokenizer, _path = load_tokenizer(model_id)
        harness = harness_for(tokenizer, model_id)
        feedback = {"role": role, "content": "Try again."}

        # Act / Assert
        with pytest.raises(
            RuntimeError,
            match=f"could not render a '{role}' feedback turn.*Label the observation",
        ):
            harness._text_feedback_ids([feedback], tokenizer.eos_token_id)
