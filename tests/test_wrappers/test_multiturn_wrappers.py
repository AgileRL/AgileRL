# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for TokenObservationWrapper sliding-window prompt fields."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import cloudpickle
import pytest
import torch
from transformers import AutoTokenizer

from agilerl.llm_envs import (
    FormatRewardWrapper,
    SearchTool,
    SyncMultiTurnVecEnv,
    TokenObservationWrapper,
    Trajectory,
    TrajectoryBuffer,
)
from agilerl.llm_envs.sync_vec_env import _as_prompt
from agilerl.utils.chat_template import inject_chat_template_kwargs
from tests import TINY_LLM_FIXTURE_PATH


class TestAsPrompt:
    def test_returns_tokenized_prompt_unchanged(self):
        prompt = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }
        assert _as_prompt(prompt) is prompt

    def test_rejects_str_observation(self):
        with pytest.raises(TypeError, match="got str"):
            _as_prompt("a plain text observation")

    def test_rejects_non_prompt_mapping(self):
        with pytest.raises(TypeError, match="expects ReasoningPrompts"):
            _as_prompt({"not_a_prompt": 1})


class _StubTokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(ids)


def _bare_wrapper() -> TokenObservationWrapper:
    w = TokenObservationWrapper.__new__(TokenObservationWrapper)
    w.strict_chat_template_boundary = True
    return w


class TestTokenObservationWrapperBuildModelPromptFields:
    def test_build_model_prompt_fields_no_truncation(self) -> None:
        w = _bare_wrapper()
        w.tokenizer = _StubTokenizer()
        w._initial_prompt_len = 3
        w.turn_boundaries = []
        w.full_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

        out = w.build_model_prompt_fields(max_prompt_tokens=100)
        assert torch.equal(out["trajectory_input_ids"], w.full_ids)
        assert out["stitch_prefix_ids"].shape[1] == 0
        assert out["initial_prompt_len"] == 3

    def test_sliding_window_reconstructs_full_sequence(self) -> None:
        """After dropping oldest post-initial turns, stitch + trunc[:I] + trunc[I:] equals full."""
        w = _bare_wrapper()
        w.tokenizer = _StubTokenizer()
        w._initial_prompt_len = 2
        # [init0,init1 | gen0a,gen0b | fb0a,fb0b | gen1a,gen1b | tail0,tail1]
        w.full_ids = torch.tensor(
            [[0, 1, 10, 11, 20, 21, 40, 41, 50, 51]], dtype=torch.long
        )
        w.turn_boundaries = [
            (2, 4, 0),
            (6, 8, 1),
        ]
        out = w.build_model_prompt_fields(max_prompt_tokens=6)
        trunc = out["trajectory_input_ids"]
        stitch = out["stitch_prefix_ids"]
        il = out["initial_prompt_len"]
        assert il == 2
        merged = torch.cat([trunc[:, :il], stitch, trunc[:, il:]], dim=1)
        assert torch.equal(merged, w.full_ids)

    def test_build_model_prompt_fields_errors(self) -> None:
        w = _bare_wrapper()
        w.tokenizer = _StubTokenizer()
        w.full_ids = None
        with pytest.raises(RuntimeError, match="No prompt"):
            w.build_model_prompt_fields(max_prompt_tokens=8)

        w.full_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        w._initial_prompt_len = 3
        w.turn_boundaries = []
        with pytest.raises(RuntimeError, match="Initial prompt"):
            w.build_model_prompt_fields(max_prompt_tokens=2)

        w.full_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        w._initial_prompt_len = 2
        w.turn_boundaries = []
        with pytest.raises(RuntimeError, match="Could not fit prompt"):
            w.build_model_prompt_fields(max_prompt_tokens=5)

    def test_build_model_prompt_fields_drop_from_ge_seq_len_branch(self) -> None:
        w = _bare_wrapper()
        w.tokenizer = _StubTokenizer()
        w._initial_prompt_len = 2
        w.full_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
        w.turn_boundaries = [(5, 5, 0)]
        out = w.build_model_prompt_fields(max_prompt_tokens=4)
        assert torch.equal(
            out["trajectory_input_ids"],
            torch.tensor([[10, 11]], dtype=torch.long),
        )


def test_max_prompt_tokens_for_sliding_window() -> None:
    from agilerl.utils.llm_utils import max_prompt_tokens_for_sliding_window

    assert max_prompt_tokens_for_sliding_window(8192, 1024) == 8192 - 1024
    assert max_prompt_tokens_for_sliding_window(100, 50) == 50
    assert max_prompt_tokens_for_sliding_window(100, None) == 99


def test_middle_stitch_tensor_layout_matches_vllm_colocate() -> None:
    """Same cat layout as LLMAlgorithm._generate_with_vllm_colocate post-process."""
    il = 2
    stitch = torch.tensor([[30, 31]], dtype=torch.long)
    block = torch.tensor([[0, 1, 10, 11, 100, 101]], dtype=torch.long)
    merged = torch.cat([block[:, :il], stitch, block[:, il:]], dim=1)
    expected = torch.tensor([[0, 1, 30, 31, 10, 11, 100, 101]], dtype=torch.long)
    assert torch.equal(merged, expected)


class _ChrTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"

    def __call__(self, texts, **kwargs):
        ids = [[ord(c) for c in texts[0]]]
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in s]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(x)) for x in ids)


class _RecordingGemEnv:
    def __init__(self) -> None:
        self.last_gen: str | None = None

    def reset(self):
        return "hello", {}

    def step(self, gen_text: str):
        self.last_gen = gen_text
        return "", 1.0, True, False, {}


class TestTokenObservationWrapperReset:
    def test_reset_returns_tuple_with_text_and_sets_prompt_len(self) -> None:
        inner = _RecordingGemEnv()
        w = TokenObservationWrapper(
            inner,
            _ChrTokenizer(),
            max_turns=3,
            pad_id=None,
            apply_chat_template=False,
        )
        obs, info = w.reset()
        assert isinstance(info, dict)
        assert set(obs.keys()) >= {"input_ids", "attention_mask", "text"}
        assert obs["text"] == "hello"
        assert w._last_full_prompt_token_len == obs["input_ids"].shape[1]

    def test_reset_seed_fallback_for_seedless_env(self) -> None:
        w = TokenObservationWrapper(
            _SeedlessResetEnv(),
            _ChrTokenizer(),
            max_turns=1,
            pad_id=None,
            apply_chat_template=False,
        )
        obs, _ = w.reset(seed=123)
        assert "input_ids" in obs


class TestTokenObservationWrapperStep:
    def test_step_from_full_completion_slices_generation(self) -> None:
        inner = _RecordingGemEnv()
        w = TokenObservationWrapper(
            inner,
            _ChrTokenizer(),
            max_turns=3,
            pad_id=None,
            apply_chat_template=False,
        )
        obs, _ = w.reset()
        obs["input_ids"].shape[1]
        gen_ids = torch.tensor([[ord("x"), ord("y")]], dtype=torch.long)
        full = torch.cat([obs["input_ids"], gen_ids], dim=1)
        _pd, _r, term, _trunc, _i = w.step(full)
        assert inner.last_gen == "xy"
        assert term is True
        assert _pd == {}

    def test_step_raises_without_prior_policy_observation(self) -> None:
        w = _bare_wrapper()
        w._last_full_prompt_token_len = None
        with pytest.raises(RuntimeError, match="requires a prior reset"):
            w.step(torch.ones(1, 2, dtype=torch.long))

    def test_chat_template_paths_and_nonterminal_step_feedback_append(self) -> None:
        env = _NonTerminalEnv()
        w = TokenObservationWrapper(
            env,
            _ChatTokenizer(),
            max_turns=2,
            pad_id=None,
            apply_chat_template=True,
            max_model_len=32,
            max_output_tokens=4,
            enable_sliding_window=True,
            strict_chat_template_boundary=False,
        )
        obs, _ = w.reset()
        assert obs["input_ids"].dtype == torch.long
        assert "trajectory_input_ids" in obs

        completion = torch.cat(
            [obs["input_ids"], torch.tensor([[7, 8]], dtype=torch.long)],
            dim=1,
        )
        next_obs, reward, terminated, truncated, _ = w.step(completion)
        assert reward == 0.5
        assert not terminated
        assert not truncated
        assert env.last_gen == "7|8"
        assert next_obs["input_ids"].shape[1] > completion.shape[1]
        assert w._feedback_texts[-1] == "F:feedback\nT"

    def test_non_chat_feedback_tokenization_path(self) -> None:
        env = _NonTerminalEnv()
        w = TokenObservationWrapper(
            env,
            _ChrTokenizer(),
            max_turns=2,
            pad_id=None,
            apply_chat_template=False,
        )
        obs, _ = w.reset()
        completion = torch.cat(
            [obs["input_ids"], torch.tensor([[120, 121]], dtype=torch.long)],
            dim=1,
        )
        next_obs, _, terminated, truncated, _ = w.step(completion)
        assert not terminated
        assert not truncated
        assert next_obs["input_ids"].shape[1] > completion.shape[1]

    def test_strict_mode_terminates_on_context_overflow(self) -> None:
        """When sliding window is disabled and the cumulative prompt would
        exceed ``max_model_len - max_output_tokens``, the trajectory ends
        with ``truncated=True`` and an ``agilerl_context_overflow``
        breadcrumb in ``info``.
        """
        env = _NonTerminalEnv()
        # Tiny budget: 20 - 4 = 16 prompt tokens. Initial prompt "P:hello\nS"
        # is 9 char-tokens; +2 gen +12 feedback ("F:feedback\nT") => 23 > 16.
        w = TokenObservationWrapper(
            env,
            _ChrTokenizer(),
            max_turns=4,
            pad_id=None,
            apply_chat_template=False,
            max_model_len=20,
            max_output_tokens=4,
            # enable_sliding_window defaults to False (strict mode).
        )
        obs, _ = w.reset()
        completion = torch.cat(
            [obs["input_ids"], torch.tensor([[120, 121]], dtype=torch.long)],
            dim=1,
        )
        next_obs, _, terminated, truncated, info = w.step(completion)
        assert truncated is True
        assert terminated is False
        assert next_obs == {}
        assert "agilerl_context_overflow" in info
        overflow = info["agilerl_context_overflow"]
        assert overflow["max_prompt_tokens"] == 16
        assert overflow["max_model_len"] == 20
        assert overflow["max_output_tokens"] == 4
        assert overflow["full_prompt_len"] > overflow["max_prompt_tokens"]

    def test_sliding_window_silently_truncates_instead_of_terminating(
        self,
    ) -> None:
        """Counterpart to strict-mode test: with sliding window enabled,
        overflow is masked by dropping older turns from the prompt.
        """
        env = _NonTerminalEnv()
        w = TokenObservationWrapper(
            env,
            _ChrTokenizer(),
            max_turns=4,
            pad_id=None,
            apply_chat_template=False,
            max_model_len=20,
            max_output_tokens=4,
            enable_sliding_window=True,
        )
        obs, _ = w.reset()
        completion = torch.cat(
            [obs["input_ids"], torch.tensor([[120, 121]], dtype=torch.long)],
            dim=1,
        )
        next_obs, _, terminated, truncated, info = w.step(completion)
        assert not terminated
        assert not truncated
        assert "agilerl_context_overflow" not in info
        assert "trajectory_input_ids" in next_obs
        assert next_obs["trajectory_input_ids"].shape[1] <= 16


class TestTokenObservationWrapperChatTemplateBoundary:
    """Verify the assistant→user→assistant boundary is computed via the
    tokenizer's chat template rather than hard-coded ChatML markers.
    """

    def test_gemma_style_template_emits_full_boundary(self) -> None:
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_gemma_chat)

        out = w._chat_template_boundary_ids("FEEDBACK")

        assert out is not None
        decoded = "".join(chr(int(x)) for x in out[0].tolist())
        # Must close the assistant turn, open a user turn with the feedback,
        # close it, then open a fresh model turn. Placeholder must not leak.
        assert TokenObservationWrapper._BOUNDARY_PLACEHOLDER not in decoded
        assert decoded.startswith("<end_of_turn>\n<start_of_turn>user\n")
        assert "FEEDBACK" in decoded
        assert decoded.endswith("<start_of_turn>model\n")

    def test_qwen_style_template_emits_full_boundary(self) -> None:
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_chatml)

        out = w._chat_template_boundary_ids("FEEDBACK")

        assert out is not None
        decoded = "".join(chr(int(x)) for x in out[0].tolist())
        assert TokenObservationWrapper._BOUNDARY_PLACEHOLDER not in decoded
        assert decoded.startswith("<|im_end|>\n<|im_start|>user\n")
        assert "FEEDBACK" in decoded
        assert decoded.endswith("<|im_start|>assistant\n")

    def test_llama_style_template_emits_full_boundary(self) -> None:
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_llama)

        out = w._chat_template_boundary_ids("FEEDBACK")

        assert out is not None
        decoded = "".join(chr(int(x)) for x in out[0].tolist())
        assert TokenObservationWrapper._BOUNDARY_PLACEHOLDER not in decoded
        # Should close assistant via <|eot_id|> then open a user header.
        assert decoded.startswith("<|eot_id|><|start_header_id|>user")
        assert "FEEDBACK" in decoded
        assert decoded.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_returns_none_when_template_renderer_raises(self) -> None:
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_raises)
        assert w._chat_template_boundary_ids("F") is None

    def test_returns_none_when_placeholder_is_stripped(self) -> None:
        # Pathological template whose render drops content entirely; the
        # placeholder doesn't survive, so we can't slice. Caller falls back.
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_drops_content)
        assert w._chat_template_boundary_ids("F") is None

    def test_returns_none_when_render_is_not_a_string(self) -> None:
        # Some tokenizers tokenize regardless of ``tokenize=False`` and hand
        # back ids; we can only slice a string render, so fall back.
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_returns_ids)
        assert w._chat_template_boundary_ids("F") is None

    def test_returns_none_when_boundary_text_is_empty(self) -> None:
        # Render ends exactly at the placeholder -> nothing after it to
        # tokenize as the boundary, so fall back.
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_ends_at_placeholder)
        assert w._chat_template_boundary_ids("F") is None

    def test_returns_none_when_boundary_encodes_to_no_tokens(self) -> None:
        # A tokenizer that maps the boundary text to zero ids gives us
        # nothing to append, so fall back.
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _EmptyEncodeTokenizer(_render_chatml)
        assert w._chat_template_boundary_ids("F") is None

    def test_tokenize_feedback_prefers_chat_template_boundary(self) -> None:
        # With a working (Gemma-style) template, _tokenize_feedback must
        # return the template-derived boundary — not the hard-coded ChatML
        # fallback markers.
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_gemma_chat)
        out = w._tokenize_feedback("FEEDBACK")
        decoded = "".join(chr(int(x)) for x in out[0].tolist())
        assert decoded.startswith("<end_of_turn>\n<start_of_turn>user\n")
        assert decoded.endswith("<start_of_turn>model\n")
        assert "<|im_start|>" not in decoded  # ChatML fallback not used

    def test_full_tokenize_feedback_falls_back_to_chatml(self) -> None:
        # If the chat-template path returns None (no apply_chat_template at
        # all on the tokenizer), a non-strict wrapper falls back to ChatML so
        # we never crash.
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.strict_chat_template_boundary = False
        w.tokenizer = _ChrTokenizerWithChatTemplateBroken()
        out = w._tokenize_feedback("F")
        assert out.shape[0] == 1
        assert out.shape[1] > 0

    def test_strict_tokenize_feedback_raises_instead_of_guessing(self) -> None:
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChrTokenizerWithChatTemplateBroken()
        with pytest.raises(RuntimeError, match="Could not render the chat template"):
            w._tokenize_feedback("F")

    def test_strict_is_the_default(self) -> None:
        w = TokenObservationWrapper(_FeedbackEnv(), _ChrTokenizer(), max_turns=1)
        assert w.strict_chat_template_boundary is True


def _render_gemma_chat(messages, add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else m["role"]
        parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n")
    if add_generation_prompt:
        parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _render_chatml(messages, add_generation_prompt: bool) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _render_llama(messages, add_generation_prompt: bool) -> str:
    parts = ["<|begin_of_text|>"]
    parts.extend(
        f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
        for m in messages
    )
    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _render_raises(messages, add_generation_prompt: bool) -> str:
    msg = "template error"
    raise RuntimeError(msg)


def _render_drops_content(messages, add_generation_prompt: bool) -> str:
    """Template that drops content entirely — placeholder cannot be located."""
    body = "|".join(m["role"] for m in messages)
    if add_generation_prompt:
        body += "|gen"
    return body


def _render_returns_ids(messages, add_generation_prompt: bool):
    """Renderer that tokenizes despite ``tokenize=False`` — non-str render."""
    del add_generation_prompt
    return [ord(c) for m in messages for c in m["content"]]


def _render_ends_at_placeholder(messages, add_generation_prompt: bool) -> str:
    """Render whose last bytes are the placeholder — empty boundary text."""
    del add_generation_prompt
    return "prefix" + messages[1]["content"]


class _ChatTemplateRecordingTokenizer:
    """Tokenizer that delegates rendering to a passed-in callable."""

    pad_token_id = 0
    pad_token = "<pad>"

    def __init__(self, renderer):
        self._renderer = renderer

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ):
        if tokenize:
            text = self._renderer(messages, add_generation_prompt)
            return {"input_ids": [[ord(c) for c in text]]}
        return self._renderer(messages, add_generation_prompt)

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in s]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(x)) for x in ids)


class _EmptyEncodeTokenizer(_ChatTemplateRecordingTokenizer):
    """Renders fine but encodes every string to zero token ids."""

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        del s, add_special_tokens
        return []


class _ChrTokenizerWithChatTemplateBroken(_ChrTokenizer):
    """Has no apply_chat_template at all, so the boundary diff path errors."""


class TestTokenObservationWrapperPolicyObservationFromState:
    def test_policy_observation_merges_sliding_window_when_max_model_len_set(
        self,
    ) -> None:
        w = _bare_wrapper()
        w.tokenizer = _StubTokenizer()
        w._sw_max_model_len = 100
        w._sw_max_output_tokens = 10
        w._sw_enabled = True
        w._initial_prompt_len = 2
        w.full_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        w.turn_boundaries = []
        obs = w._policy_observation_from_state()
        assert "trajectory_input_ids" in obs
        assert "trajectory_text" in obs
        assert obs["input_ids"].shape[1] == 4

    def test_policy_observation_skips_sliding_window_by_default(self) -> None:
        """Strict mode (sliding window disabled) does NOT emit trunc/stitch fields."""
        w = _bare_wrapper()
        w.tokenizer = _StubTokenizer()
        w._sw_max_model_len = 100
        w._sw_max_output_tokens = 10
        w._sw_enabled = False
        w._initial_prompt_len = 2
        w.full_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        w.turn_boundaries = []
        obs = w._policy_observation_from_state()
        assert "trajectory_input_ids" not in obs
        assert "stitch_prefix_ids" not in obs
        assert obs["input_ids"].shape[1] == 4

    def test_policy_observation_raises_without_full_ids(self) -> None:
        w = _bare_wrapper()
        w.full_ids = None
        with pytest.raises(RuntimeError, match="No prompt"):
            w._policy_observation_from_state()


class _SyncStubEnv:
    def __init__(self, sw_max_model_len: int | None = None) -> None:
        self._sw_max_model_len = sw_max_model_len
        self.turn_boundaries: list[int] = []
        self.reset_calls: list[int | None] = []
        self.close_calls = 0

    def reset(self, seed: int | None = None):
        self.reset_calls.append(seed)
        return (
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
            {},
        )

    def step(self, full_completion: torch.Tensor):
        del full_completion
        self.turn_boundaries.append(1)
        return ({}, 1.0, True, False, {})

    def close(self) -> None:
        self.close_calls += 1

    def get_episode_data(self):
        return (
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
        )


class TestSyncMultiTurnVecEnvReset:
    def test_sync_gem_vec_env_reset_seeds_per_batch_group(self) -> None:
        vec_env = SyncMultiTurnVecEnv(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=10)
        seen = [traj.env.reset_calls[-1] for traj in vec_env.trajectories]
        assert seen == [10, 10, 11, 11]

    def test_sync_gem_vec_env_reset_with_none_seed(self) -> None:
        vec_env = SyncMultiTurnVecEnv(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=None)
        seen = [traj.env.reset_calls[-1] for traj in vec_env.trajectories]
        assert seen == [None, None, None, None]

    def test_sync_gem_vec_env_reset_reuses_existing_trajectories(self) -> None:
        vec_env = SyncMultiTurnVecEnv(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=10)
        _ = vec_env.reset(seed=20)
        seen = [traj.env.reset_calls[-1] for traj in vec_env.trajectories]
        assert seen == [20, 20, 21, 21]


class TestSyncMultiTurnVecEnvStep:
    def test_sync_gem_vec_env_step_raises_when_completion_count_mismatches_active(
        self,
    ) -> None:
        vec_env = SyncMultiTurnVecEnv(
            env_factory=_SyncStubEnv,
            batch_size=1,
            group_size=2,
        )
        _ = vec_env.reset(seed=0)
        with pytest.raises(
            RuntimeError,
            match="Number of completions does not match number of active trajectories",
        ):
            vec_env.step([torch.ones(1, 5, dtype=torch.long)])

    def test_sync_vec_env_step_happy_path_1d_and_2d_and_active_filtering(self) -> None:
        created = [
            _StepVariantEnv(done_after_step=False),
            _StepVariantEnv(done_after_step=True),
        ]
        idx = {"i": 0}

        def _factory():
            env = created[idx["i"]]
            idx["i"] += 1
            return env

        vec = SyncMultiTurnVecEnv(env_factory=_factory, batch_size=1, group_size=2)
        _ = vec.reset(seed=0)
        prompts = vec.step(
            [
                torch.tensor([1, 2, 3], dtype=torch.long),
                torch.tensor([[1, 2, 3]], dtype=torch.long),
            ]
        )
        assert prompts is not None
        assert isinstance(prompts, list)
        assert len(prompts) == 1
        assert prompts[0]["input_ids"].shape == (1, 4)
        assert created[0].step_shapes == [(1, 3)]
        assert created[1].step_shapes == [(1, 3)]

    def test_sync_vec_env_step_raises_on_sampling_logps_count_mismatch(self) -> None:
        vec_env = SyncMultiTurnVecEnv(
            env_factory=_SyncStubEnv,
            batch_size=1,
            group_size=2,
        )
        _ = vec_env.reset(seed=0)
        with pytest.raises(
            RuntimeError,
            match="Number of sampling logprobs does not match number of active",
        ):
            vec_env.step(
                [
                    torch.ones(1, 5, dtype=torch.long),
                    torch.ones(1, 5, dtype=torch.long),
                ],
                sampling_logps=[torch.tensor([-0.1, -0.2])],  # 1 != 2 active
            )

    def test_sync_vec_env_step_accumulates_sampling_logps_per_trajectory(self) -> None:
        """Each turn's vLLM sampling logprobs append onto that trajectory's
        ``Trajectory.sampling_logps``; ``None`` rows (nothing captured) are
        skipped. ``get_trajectories`` concatenates across turns and keeps a
        per-trajectory ``None`` for rows that never captured any.
        """
        vec = SyncMultiTurnVecEnv(
            env_factory=lambda: _StepVariantEnv(done_after_step=False),
            batch_size=1,
            group_size=2,
        )
        _ = vec.reset(seed=0)
        completions = [
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
        ]
        _ = vec.step(completions, sampling_logps=[torch.tensor([-0.1, -0.2]), None])
        _ = vec.step(completions, sampling_logps=[torch.tensor([-0.3]), None])
        assert len(vec.trajectories[0].sampling_logps) == 2
        assert vec.trajectories[1].sampling_logps == []

        *_parts, sampling = vec.get_trajectories()
        assert sampling is not None
        assert torch.equal(sampling[0], torch.tensor([-0.1, -0.2, -0.3]))
        assert sampling[1] is None

    def test_sync_vec_env_sampling_logps_collapse_to_none_when_uncaptured(
        self,
    ) -> None:
        """Without captured logprobs the rollout-wide entry is a single
        ``None`` (not a list of ``None``s), and a reset clears any logprobs
        accumulated in a previous rollout.
        """
        vec = SyncMultiTurnVecEnv(
            env_factory=lambda: _StepVariantEnv(done_after_step=False),
            batch_size=1,
            group_size=2,
        )
        _ = vec.reset(seed=0)
        completions = [
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
        ]
        # sampling_logps omitted entirely -> nothing accumulates.
        _ = vec.step(completions)
        *_parts, sampling = vec.get_trajectories()
        assert sampling is None

        # Captured logprobs from one rollout must not leak past a reset.
        _ = vec.step(completions, sampling_logps=[torch.tensor([-0.5]), None])
        assert len(vec.trajectories[0].sampling_logps) == 1
        _ = vec.reset(seed=1)
        assert vec.trajectories[0].sampling_logps == []
        *_parts, sampling = vec.get_trajectories()
        assert sampling is None


class TestSyncMultiTurnVecEnvClose:
    def test_sync_vec_env_close_calls_underlying_env_close_once(self) -> None:
        vec_env = SyncMultiTurnVecEnv(
            env_factory=lambda: _SyncStubEnv(sw_max_model_len=1024),
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=0)
        vec_env.close()
        close_counts = [traj.env.close_calls for traj in vec_env.trajectories]
        assert close_counts == [1, 1, 1, 1]

    def test_sync_vec_env_close_dedupes_same_env_instance(self) -> None:
        shared = _SyncStubEnv()
        vec = SyncMultiTurnVecEnv(
            env_factory=lambda: shared, batch_size=2, group_size=2
        )
        _ = vec.reset(seed=0)
        vec.close()
        assert shared.close_calls == 1


class TestSyncMultiTurnVecEnvInit:
    def test_sync_vec_env_constructor_rejects_non_positive_sizes(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            _ = SyncMultiTurnVecEnv(
                env_factory=_SyncStubEnv,
                batch_size=0,
                group_size=1,
            )
        with pytest.raises(ValueError, match="group_size must be > 0"):
            _ = SyncMultiTurnVecEnv(
                env_factory=_SyncStubEnv,
                batch_size=1,
                group_size=0,
            )


class TestTrajectoryBufferResetTrajectory:
    def test_trajectory_buffer_reset_trajectory_out_of_bounds(self) -> None:
        buf = TrajectoryBuffer(batch_size=1, group_size=1)
        with pytest.raises(IndexError, match="env_idx out of bounds"):
            buf.reset_trajectory(seed=0, env_idx=0)

    def test_trajectory_buffer_reset_trajectory_success_path(self) -> None:
        env = _SyncStubEnv()
        traj = Trajectory(
            env=env,
            batch_idx=0,
            group_idx=0,
            prompt={},
            done=True,
        )
        buf = TrajectoryBuffer(batch_size=1, group_size=1)
        buf.add_trajectory(traj)
        buf.reset_trajectory(seed=5, env_idx=0)
        assert buf[0].done is False
        assert buf[0].prompt["input_ids"].shape == (1, 3)
        assert env.reset_calls[-1] == 5


class _ChatTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        del add_generation_prompt
        content = messages[0]["content"]
        ids = [100] + [ord(c) for c in content]
        if tokenize:
            # Transformers v5: dict with input_ids/attention_mask.
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return content

    def __call__(self, texts, **kwargs):
        del kwargs
        ids = [[ord(c) for c in texts[0]]]
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in s]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "|".join(str(int(x)) for x in ids)


class _NestedChatTokenizer(_ChatTokenizer):
    """Chat tokenizer whose ``apply_chat_template`` returns batched (nested)
    token-id lists — ``{"input_ids": [[...]]}`` — as some tokenizers do.
    """

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        out = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if tokenize:
            return {
                "input_ids": [out["input_ids"]],
                "attention_mask": [out["attention_mask"]],
            }
        return out


class TestTokenObservationWrapperTokenizeInitialPrompt:
    def test_initial_prompt_unwraps_batched_token_id_lists(self) -> None:
        """Tokenizers returning ``[[ids]]`` (batch dim) and ``[ids]`` (flat)
        from ``apply_chat_template`` must produce identical ``(1, T)``
        tensors.
        """
        flat_ids = _ChatTokenizer().apply_chat_template(
            [{"role": "user", "content": "hi"}]
        )["input_ids"]

        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _NestedChatTokenizer()
        out = w._tokenize_initial_prompt("hi")
        assert out["input_ids"].shape == (1, len(flat_ids))
        assert out["input_ids"][0].tolist() == flat_ids
        assert torch.equal(out["attention_mask"], torch.ones_like(out["input_ids"]))


class _SeedlessResetEnv:
    def reset(self):
        return "obs", {}

    def step(self, gen_text: str):
        del gen_text
        return "done", 0.0, True, False, {}


class _NonTerminalEnv:
    def __init__(self) -> None:
        self.calls = 0
        self.last_gen: str | None = None

    def reset(self, seed: int | None = None):
        del seed
        return "hello", {"prefix": "P:", "suffix": "S"}

    def step(self, gen_text: str):
        self.calls += 1
        self.last_gen = gen_text
        if self.calls == 1:
            return "feedback", 0.5, False, False, {"prefix": "F:", "suffix": "T"}
        return "done", 1.0, True, False, {}


class TestTokenObservationWrapperFormatObs:
    def test_format_obs_prefix_suffix_and_empty_info(self) -> None:
        assert TokenObservationWrapper._format_obs("x", None) == "x"
        assert (
            TokenObservationWrapper._format_obs("x", {"prefix": "A:", "suffix": "B"})
            == "A:x\nB"
        )


class TestTokenObservationWrapperGetEpisodeData:
    def test_get_episode_data_keeps_pad_inside_generation_span(self) -> None:
        """Pad tokens inside a turn generation stay masked as actions.

        When ``pad_token_id == eos_token_id`` (typical after assigning
        ``tokenizer.pad_token = tokenizer.eos_token``), the EOS at the end of
        a generation must remain an action token so vLLM sampling-logprob
        counts stay aligned for importance-sampling correction.
        """
        w = _bare_wrapper()
        w.pad_id = 0
        w.max_turns = 3
        # Token 0 at index 2 sits inside turn-0 generation [1, 3).
        w.full_ids = torch.tensor([[9, 5, 0, 7, 8]], dtype=torch.long)
        w.turn_boundaries = [(1, 3, 0), (3, 5, 1)]
        w.turn_rewards = [1.5]
        full_ids, action_mask, turn_ids, rewards = w.get_episode_data()
        assert torch.equal(full_ids, w.full_ids)
        assert action_mask.dtype == torch.bool
        assert turn_ids.dtype == torch.long
        assert rewards.tolist() == [1.5, 0.0, 0.0]
        # Shifted index 1 ↔ full_ids[2] == pad, but still inside gen span.
        assert action_mask[0, 1].item() is True
        assert turn_ids[0, 1].item() == 0
        assert int(action_mask.sum().item()) == 4

    def test_get_episode_data_clears_pad_outside_generation_span(self) -> None:
        w = _bare_wrapper()
        w.pad_id = 0
        w.max_turns = 2
        # Prompt token at index 1 is pad and outside any generation span.
        w.full_ids = torch.tensor([[9, 0, 5, 7]], dtype=torch.long)
        w.turn_boundaries = [(2, 4, 0)]
        w.turn_rewards = [0.25]
        _full_ids, action_mask, turn_ids, rewards = w.get_episode_data()
        assert rewards.tolist() == [0.25, 0.0]
        # Shifted index 0 ↔ full_ids[1] == pad, outside gen → cleared.
        assert action_mask[0, 0].item() is False
        assert turn_ids[0, 0].item() == -1
        # Generation tokens remain actions.
        assert action_mask[0, 1].item() is True
        assert action_mask[0, 2].item() is True
        assert turn_ids[0, 1].item() == 0
        assert turn_ids[0, 2].item() == 0

    def test_get_episode_data_raises_without_reset(self) -> None:
        w = _bare_wrapper()
        w.full_ids = None
        with pytest.raises(RuntimeError, match="No episode data"):
            w.get_episode_data()


class TestTokenObservationWrapperGetDebugInfo:
    def test_get_debug_info_paths(self) -> None:
        w = _bare_wrapper()
        w.full_ids = None
        assert w.get_debug_info() == {"error": "No episode data"}

        w = _bare_wrapper()
        w.tokenizer = _ChrTokenizer()
        w.pad_id = None
        w.max_turns = 2
        w.full_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        w.turn_boundaries = [(2, 4, 0)]
        w.turn_rewards = [0.5]
        w._gen_texts = ["AB"]
        w._feedback_texts = ["fb"]
        w._prompt_text = "prompt"
        info = w.get_debug_info()
        assert info["n_turns"] == 1
        assert info["turn_rewards_padded"] == [0.5, 0.0]
        assert info["feedback_texts"] == ["fb"]
        assert info["turn_details"][0]["gen_len"] == 2


class TestSearchToolParseAction:
    def test_search_tool_parse_action_and_instruction(self) -> None:
        tool = SearchTool(search_url="http://x")
        query, parsed_action, valid = tool._parse_action("a<search> q </search>z")
        assert (query, parsed_action, valid) == ("q", "a<search> q </search>", True)
        assert tool._parse_action("no tags") == ("", "", False)
        assert "<answer>" in tool.instruction_string()


class TestSearchToolSearch:
    def test_search_tool_search_success_and_failure_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tool = SearchTool(search_url="http://x", topk=1, timeout=1)

        class _Resp:
            def json(self):
                return {"results": [{"content": "first"}, {"content": "second"}]}

        def _ok_get(url, params, timeout):
            del url, params, timeout
            return _Resp()

        monkeypatch.setattr("agilerl.llm_envs.requests.get", _ok_get)
        out = tool._search("hello")
        assert "first" in out
        assert "second" not in out

        def _fail_get(url, params, timeout):
            del url, params, timeout
            msg = "search request failed"
            raise RuntimeError(msg)

        monkeypatch.setattr("agilerl.llm_envs.requests.get", _fail_get)
        assert "[SearchTool Error:" in tool._search("hello")

        no_url = SearchTool(search_url=None)
        monkeypatch.delenv("SEARCH_URL", raising=False)
        with pytest.raises(ValueError, match="search_url must be provided"):
            no_url._search("x")


class TestSearchToolExecuteAction:
    def test_search_tool_passages_to_string_and_execute_action(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tool = SearchTool(search_url="http://x")
        passages = [
            {"document": {"contents": "Title A\nBody A"}},
            {"document": {"contents": "Title B\nBody B"}},
        ]
        formatted = tool._passages2string(passages)
        assert "Doc 1(Title: Title A) Body A" in formatted

        valid, has_error, observation, parsed_action = tool.execute_action("nope")
        assert (valid, has_error, observation, parsed_action) == (False, True, "", "")

        monkeypatch.setattr(tool, "_search", lambda q: f"res:{q}")
        valid, has_error, observation, parsed_action = tool.execute_action(
            "<search>cats</search> trailing",
        )
        assert valid is True
        assert has_error is False
        assert "<information>res:cats</information>" in observation
        assert parsed_action == "<search>cats</search>"


class _FormatEnv:
    def __init__(self):
        self.state = "ok"

    def reset(self, **kwargs):
        return ("obs", kwargs)

    def step(self, action: str, **kwargs):
        del kwargs
        return ("next", 1.0, True, False, {"correct": False, "action": action})


class TestFormatRewardWrapperStep:
    def test_format_reward_wrapper_branches_and_passthrough(self) -> None:
        env = _FormatEnv()
        wrapped = FormatRewardWrapper(env, format_bonus=0.3)
        assert wrapped.format_bonus == 0.3
        assert wrapped.state == "ok"
        obs, rew, term, trunc, info = wrapped.step("<answer>bad</answer>")
        assert (obs, term, trunc) == ("next", True, False)
        assert info["correct"] is False
        assert rew == 1.3
        assert wrapped.reset(seed=7) == ("obs", {"seed": 7})

        class _NoBonusEnv(_FormatEnv):
            def step(self, action: str, **kwargs):
                del action, kwargs
                return ("next", 2.0, True, False, {"correct": True})

        no_bonus = FormatRewardWrapper(_NoBonusEnv(), format_bonus=0.5)
        assert no_bonus.step("<answer>good</answer>")[1] == 2.0


class TestTrajectoryBufferInvariantsAndHelpers:
    def test_trajectory_buffer_invariants_and_helpers(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            _ = TrajectoryBuffer(batch_size=0, group_size=1)
        with pytest.raises(ValueError, match="group_size must be > 0"):
            _ = TrajectoryBuffer(batch_size=1, group_size=0)

        env = _SyncStubEnv()
        t1 = Trajectory(
            env=env,
            batch_idx=1,
            group_idx=0,
            prompt={
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
            },
            done=False,
        )
        t2 = Trajectory(
            env=env,
            batch_idx=0,
            group_idx=1,
            prompt={
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
            },
            done=True,
        )
        buf = TrajectoryBuffer(batch_size=1, group_size=2)
        buf.add_trajectory(t1)
        buf.add_trajectory(t2)
        assert buf.is_initialized is True
        assert buf.has_active() is True
        assert len(buf) == 2
        assert next(iter(buf)) is t1
        assert buf[1] is t2
        buf.sort(key=lambda t: (t.batch_idx, t.group_idx))
        assert [t.batch_idx for t in buf] == [0, 1]
        buf.clear()
        assert len(buf) == 0
        assert buf.has_active() is False


class TestTrajectoryBufferGetActiveTrajectories:
    def test_trajectory_buffer_get_active_trajectories_sorting(self) -> None:
        env = _SyncStubEnv()
        t0 = Trajectory(
            env=env,
            batch_idx=1,
            group_idx=0,
            prompt={
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
            },
            done=False,
        )
        t1 = Trajectory(
            env=env,
            batch_idx=0,
            group_idx=1,
            prompt={
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
            },
            done=False,
        )
        t2 = Trajectory(
            env=env,
            batch_idx=0,
            group_idx=0,
            prompt={
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
            },
            done=True,
        )
        buf = TrajectoryBuffer(batch_size=2, group_size=2)
        buf.add_trajectory(t0)
        buf.add_trajectory(t1)
        buf.add_trajectory(t2)

        unsorted_active = buf.get_active_trajectories(sorted_by_index=False)
        assert unsorted_active == [t0, t1]
        sorted_active = buf.get_active_trajectories(sorted_by_index=True)
        assert sorted_active == [t1, t0]


class TestTrajectoryBufferGetPrompts:
    def test_trajectory_buffer_get_prompts_returns_none_when_no_active(self) -> None:
        env = _SyncStubEnv()
        done_traj = Trajectory(
            env=env,
            batch_idx=0,
            group_idx=0,
            prompt={
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            },
            done=True,
        )
        active_traj = Trajectory(
            env=env,
            batch_idx=0,
            group_idx=1,
            prompt={
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
            done=False,
        )
        buf = TrajectoryBuffer(batch_size=1, group_size=2)
        buf.add_trajectory(done_traj)
        buf.add_trajectory(active_traj)

        prompts = buf.get_prompts()
        assert prompts is not None
        assert isinstance(prompts, list)
        assert len(prompts) == 1
        assert prompts[0]["input_ids"].shape == (1, 3)
        assert prompts[0]["attention_mask"].shape == (1, 3)

        active_traj.done = True
        assert buf.get_prompts() is None

    def test_trajectory_buffer_stack_prompt_validation(self) -> None:
        env = _SyncStubEnv()
        a = Trajectory(
            env=env,
            batch_idx=0,
            group_idx=0,
            prompt={
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "trajectory_input_ids": torch.ones(1, 2, dtype=torch.long),
                "trajectory_attention_mask": torch.ones(1, 2, dtype=torch.long),
                "stitch_prefix_ids": torch.ones(1, 1, dtype=torch.long),
                "initial_prompt_len": 2,
            },
            done=False,
        )
        b = Trajectory(
            env=env,
            batch_idx=0,
            group_idx=1,
            prompt={
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "trajectory_input_ids": torch.ones(1, 3, dtype=torch.long),
                "trajectory_attention_mask": torch.ones(1, 3, dtype=torch.long),
                "stitch_prefix_ids": torch.ones(1, 2, dtype=torch.long),
                "initial_prompt_len": 3,
            },
            done=False,
        )
        buf = TrajectoryBuffer(batch_size=1, group_size=2)
        buf.add_trajectory(a)
        buf.add_trajectory(b)
        prompts = buf.get_prompts()
        assert prompts is not None
        assert len(prompts) == 2
        assert [int(p["initial_prompt_len"]) for p in prompts] == [2, 3]
        assert prompts[0]["input_ids"].shape == (1, 2)
        assert prompts[1]["input_ids"].shape == (1, 3)


class _StepVariantEnv:
    def __init__(self, done_after_step: bool, include_turn_boundaries: bool = True):
        self.done_after_step = done_after_step
        self.include_turn_boundaries = include_turn_boundaries
        self.step_shapes: list[tuple[int, ...]] = []
        if include_turn_boundaries:
            self.turn_boundaries: list[int] = []

    def reset(self, seed: int | None = None):
        del seed
        return (
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
            {},
        )

    def step(self, full_completion: torch.Tensor):
        self.step_shapes.append(tuple(full_completion.shape))
        if self.include_turn_boundaries:
            self.turn_boundaries.append(1)
        if self.done_after_step:
            return ({}, 1.0, True, False, {})
        return (
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            0.5,
            False,
            False,
            {},
        )

    def close(self) -> None:
        pass

    def get_episode_data(self):
        return (
            torch.ones(1, 5, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
        )


class TestSyncMultiTurnVecEnvGetTrajectories:
    def test_sync_vec_env_get_trajectories_counts_steps_with_and_without_turn_boundaries(
        self,
    ) -> None:
        created = [
            _StepVariantEnv(done_after_step=True, include_turn_boundaries=True),
            _StepVariantEnv(done_after_step=True, include_turn_boundaries=False),
        ]
        idx = {"i": 0}

        def _factory():
            env = created[idx["i"]]
            idx["i"] += 1
            return env

        vec = SyncMultiTurnVecEnv(env_factory=_factory, batch_size=1, group_size=2)
        _ = vec.reset(seed=0)
        _ = vec.step(
            [
                torch.tensor([[1, 2, 3]], dtype=torch.long),
                torch.tensor([[1, 2, 3]], dtype=torch.long),
            ]
        )
        # get_trajectories now returns (..., batch_steps, all_sampling_logps);
        # batch_steps is second-to-last.
        *_parts, batch_steps, _sampling_logps = vec.get_trajectories()
        assert batch_steps == 1


class _TerminatorTokenizer:
    """Char tokenizer whose id 7 is a declared special (a turn terminator)."""

    all_special_ids: ClassVar[list[int]] = [7]

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(x)) for x in ids)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        rendered = ""
        for m in messages:
            rendered += m["content"]
            rendered += "\x07" if m["role"] == "assistant" else "|"
        if add_generation_prompt:
            rendered += "A:"
        return rendered


class _FeedbackEnv:
    def reset(self, seed=None):
        return "obs", {}

    def step(self, action):
        return "fb", 0.5, False, False, {}


class TestFeedbackTerminatorDedupe:
    def _wrapper(self, tokenizer) -> TokenObservationWrapper:
        w = TokenObservationWrapper(
            env=_FeedbackEnv(),
            tokenizer=tokenizer,
            max_turns=3,
        )
        w.full_ids = torch.tensor([[65, 66]], dtype=torch.long)
        return w

    def test_sampled_terminator_is_not_doubled(self) -> None:
        w = self._wrapper(_TerminatorTokenizer())
        w._step(torch.tensor([[65, 66, 67, 7]], dtype=torch.long), "C")
        assert w.full_ids[0].tolist().count(7) == 1

    def test_truncated_turn_still_gets_the_frame_terminator(self) -> None:
        w = self._wrapper(_TerminatorTokenizer())
        w._step(torch.tensor([[65, 66, 67]], dtype=torch.long), "C")
        assert w.full_ids[0].tolist().count(7) == 1

    def test_non_special_equal_token_is_kept(self) -> None:
        tokenizer = _TerminatorTokenizer()
        tokenizer.all_special_ids = []
        w = self._wrapper(tokenizer)
        w._step(torch.tensor([[65, 66, 67, 7]], dtype=torch.long), "C")
        assert w.full_ids[0].tolist().count(7) == 2


class TestTokenObservationSpecialTokenGuards:
    def test_raw_encode_adds_no_specials(self) -> None:
        class _BosAddingTokenizer:
            def __call__(self, texts, add_special_tokens=True, **kwargs):
                ids = [ord(c) for c in texts[0]]
                if add_special_tokens:
                    ids = [1, *ids]
                t = torch.tensor([ids], dtype=torch.long)
                return {"input_ids": t, "attention_mask": torch.ones_like(t)}

        w = TokenObservationWrapper(
            env=_FeedbackEnv(),
            tokenizer=_BosAddingTokenizer(),
            max_turns=1,
            apply_chat_template=False,
        )
        out = w._tokenize_initial_prompt("hi")
        assert out["input_ids"][0].tolist() == [ord("h"), ord("i")]

    def test_raw_feedback_encode_adds_no_specials(self) -> None:
        class _BosAddingEncoder:
            def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
                ids = [ord(c) for c in s]
                return [1, *ids] if add_special_tokens else ids

        w = TokenObservationWrapper(
            env=_FeedbackEnv(),
            tokenizer=_BosAddingEncoder(),
            max_turns=1,
            apply_chat_template=False,
        )
        assert w._tokenize_feedback("fb")[0].tolist() == [ord("f"), ord("b")]

    def test_chatml_fallback_warns(self) -> None:
        class _TemplatelessTokenizer:
            def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
                del add_special_tokens
                return [ord(c) for c in s]

        w = TokenObservationWrapper(
            env=_FeedbackEnv(),
            tokenizer=_TemplatelessTokenizer(),
            max_turns=1,
            strict_chat_template_boundary=False,
        )
        with pytest.warns(UserWarning, match="ChatML markers"):
            w._tokenize_feedback("fb")


class _ScriptedEnv:
    def __init__(self, results):
        self._results = list(results)

    def reset(self, seed=None):
        return "obs", {}

    def step(self, action):
        return self._results.pop(0)


class TestSamplingLogprobAlignmentWithEosPad:
    """With pad == eos, captured vLLM logprob counts match the action mask."""

    def _wrapper(self, results) -> TokenObservationWrapper:
        w = TokenObservationWrapper(
            env=_ScriptedEnv(results),
            tokenizer=_TerminatorTokenizer(),
            max_turns=3,
            pad_id=7,
        )
        w.full_ids = torch.tensor([[65, 66]], dtype=torch.long)
        return w

    def test_eos_terminated_turn_aligns_with_captured_logprobs(self) -> None:
        from types import SimpleNamespace

        from agilerl.algorithms.core.base import LLMAlgorithm

        w = self._wrapper([("done", 1.0, True, False, {})])
        w._step(torch.tensor([[65, 66, 67, 7]], dtype=torch.long), "C")
        _ids, action_mask, _turn_ids, _rewards = w.get_episode_data()
        assert int(action_mask.sum()) == 2

        aligned, n_skipped = LLMAlgorithm._align_sampling_logprobs(
            SimpleNamespace(),
            [torch.tensor([-0.1, -0.2])],
            action_mask,
            torch.zeros(1, action_mask.shape[1]),
        )
        assert n_skipped == 0
        assert aligned is not None

    def test_multi_turn_mid_sequence_eos_stays_trainable(self) -> None:
        w = self._wrapper(
            [("fb", 0.5, False, False, {}), ("done", 1.0, True, False, {})]
        )
        w._step(torch.tensor([[65, 66, 67, 7]], dtype=torch.long), "C")
        completion = torch.cat(
            [w.full_ids, torch.tensor([[68, 7]], dtype=torch.long)], dim=1
        )
        w._step(completion, "D")
        _ids, action_mask, _turn_ids, _rewards = w.get_episode_data()
        assert int(action_mask.sum()) == 4


# A ChatML-style template whose generation prompt and latest user turn are
# controlled by render kwargs, as the Nemotron and Qwen families do.
_THINKING_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] }}"
    "{%- if mark_effort | default(false) and loop.last "
    "and message['role'] == 'user' %}"
    "{{- '\n\n{reasoning effort: efficient}' }}"
    "{%- endif %}"
    "{{- '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- if enable_thinking | default(false) %}"
    "{{- '<think>\n' }}"
    "{%- else %}"
    "{{- '<think></think>' }}"
    "{%- endif %}"
    "{%- endif %}"
)

_LLAMA_STYLE_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)

_LAST_MESSAGE_ONLY_TEMPLATE = "{{- messages[-1]['content'] }}"

_INITIAL_OBS = "Solve this sudoku."
_FEEDBACK = "Wrong, try again."
_SECOND_FEEDBACK = "Still wrong, try again."
_EFFORT_MARKER = "{reasoning effort: efficient}"
_SYSTEM_PROMPT = "Classify the intent, then reply. Valid intents: cancel_order."
_THINK_OPEN_PROMPT = "<|im_start|>assistant\n<think>\n"
_THINK_CLOSED_PROMPT = "<|im_start|>assistant\n<think></think>"


def _templated_tokenizer(template: str, chat_template_kwargs: dict | None = None):
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    tokenizer.chat_template = template
    if chat_template_kwargs is not None:
        inject_chat_template_kwargs(tokenizer, chat_template_kwargs)
    return tokenizer


def _attribute_only_tokenizer(template: str, chat_template_kwargs: dict):
    """Tokenizer carrying render kwargs as a plain attribute, with no bound wrap."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    tokenizer.chat_template = template
    tokenizer.chat_template_default_kwargs = dict(chat_template_kwargs)
    return tokenizer


def _templated_wrapper(tokenizer, env=None, **kwargs) -> TokenObservationWrapper:
    return TokenObservationWrapper(
        env if env is not None else _FeedbackEnv(),
        tokenizer,
        max_turns=3,
        **kwargs,
    )


def _turn_generation_prompts(wrapper, feedbacks: list[str]) -> list[str]:
    """Decode the text opening each turn of a rollout driven by *feedbacks*."""
    tokenizer = wrapper.tokenizer
    opening = tokenizer.decode(
        wrapper._tokenize_initial_prompt(_INITIAL_OBS)["input_ids"][0].tolist(),
    )
    return [opening] + [
        tokenizer.decode(wrapper._tokenize_feedback(feedback)[0].tolist())
        for feedback in feedbacks
    ]


class TestChatTemplateDerivedTurnPrompts:
    """Every turn's generation prompt comes from the tokenizer's own template."""

    @pytest.mark.parametrize(
        ("enable_thinking", "expected_generation_prompt"),
        [(False, _THINK_CLOSED_PROMPT), (True, _THINK_OPEN_PROMPT)],
        ids=["think-closed", "think-open"],
    )
    def test_boundary_carries_thinking_control(
        self,
        enable_thinking,
        expected_generation_prompt,
    ) -> None:
        w = _templated_wrapper(
            _templated_tokenizer(
                _THINKING_TEMPLATE,
                {"enable_thinking": enable_thinking},
            ),
        )

        boundary = w._chat_template_boundary_text(_FEEDBACK)

        assert boundary == (
            f"<|im_end|>\n<|im_start|>user\n{_FEEDBACK}<|im_end|>\n"
            + expected_generation_prompt
        )

    @pytest.mark.parametrize("enable_thinking", [False, True])
    def test_append_only_trajectory_matches_a_full_render(
        self,
        enable_thinking,
    ) -> None:
        tokenizer = _templated_tokenizer(
            _THINKING_TEMPLATE,
            {"enable_thinking": enable_thinking},
        )
        w = _templated_wrapper(tokenizer)

        opening = tokenizer.decode(
            w._tokenize_initial_prompt(_INITIAL_OBS)["input_ids"][0].tolist(),
        )
        generated = "REASONING</think>ANSWER" if enable_thinking else "ANSWER"
        boundary = tokenizer.decode(w._tokenize_feedback(_FEEDBACK)[0].tolist())
        thinking_prefix = "<think>\n" if enable_thinking else "<think></think>"

        assert opening + generated + boundary == tokenizer.apply_chat_template(
            [
                {"role": "user", "content": _INITIAL_OBS},
                {"role": "assistant", "content": thinking_prefix + generated},
                {"role": "user", "content": _FEEDBACK},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @pytest.mark.parametrize(
        ("enable_thinking", "generation_prompt"),
        [(False, _THINK_CLOSED_PROMPT), (True, _THINK_OPEN_PROMPT)],
    )
    def test_every_turn_opens_with_the_same_generation_prompt(
        self,
        enable_thinking,
        generation_prompt,
    ) -> None:
        w = _templated_wrapper(
            _templated_tokenizer(
                _THINKING_TEMPLATE,
                {"enable_thinking": enable_thinking},
            ),
        )

        prompts = _turn_generation_prompts(w, [_FEEDBACK, _SECOND_FEEDBACK])

        assert len(prompts) == 3
        for prompt in prompts:
            assert prompt.endswith(generation_prompt)

    @pytest.mark.parametrize(
        "build_tokenizer",
        [_templated_tokenizer, _attribute_only_tokenizer],
        ids=["bound-wrap", "defaults-attribute-only"],
    )
    def test_configured_kwargs_reach_every_turn(self, build_tokenizer) -> None:
        w = _templated_wrapper(
            build_tokenizer(
                _THINKING_TEMPLATE,
                {"enable_thinking": True, "mark_effort": True},
            ),
        )

        turn_one, turn_two, turn_three = _turn_generation_prompts(
            w,
            [_FEEDBACK, _SECOND_FEEDBACK],
        )

        assert f"{_INITIAL_OBS}\n\n{_EFFORT_MARKER}" in turn_one
        assert f"{_FEEDBACK}\n\n{_EFFORT_MARKER}" in turn_two
        assert f"{_SECOND_FEEDBACK}\n\n{_EFFORT_MARKER}" in turn_three

    def test_configured_kwargs_survive_a_serialization_round_trip(self) -> None:
        tokenizer = _templated_tokenizer(
            _THINKING_TEMPLATE,
            {"enable_thinking": True, "mark_effort": True},
        )
        w = _templated_wrapper(cloudpickle.loads(cloudpickle.dumps(tokenizer)))

        prompts = _turn_generation_prompts(w, [_FEEDBACK, _SECOND_FEEDBACK])

        assert [_EFFORT_MARKER in prompt for prompt in prompts] == [True] * 3

    @pytest.mark.parametrize(
        "chat_template_kwargs",
        [{"enable_thinking": True}, {"enable_thinking": True, "mark_effort": False}],
    )
    def test_unconfigured_kwargs_leave_no_marker(self, chat_template_kwargs) -> None:
        w = _templated_wrapper(
            _templated_tokenizer(_THINKING_TEMPLATE, chat_template_kwargs),
        )

        prompts = _turn_generation_prompts(w, [_FEEDBACK, _SECOND_FEEDBACK])

        assert len(prompts) == 3
        for prompt in prompts:
            assert _EFFORT_MARKER not in prompt
            assert prompt.endswith(_THINK_OPEN_PROMPT)

    def test_non_chatml_template_derives_its_own_markers(self) -> None:
        w = _templated_wrapper(_templated_tokenizer(_LLAMA_STYLE_TEMPLATE))

        boundary = w._chat_template_boundary_text(_FEEDBACK)

        assert boundary == (
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{_FEEDBACK}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        assert "<|im_start|>" not in boundary
        assert "<|im_end|>" not in boundary

    def test_unsliceable_template_raises_instead_of_guessing(self) -> None:
        w = _templated_wrapper(_templated_tokenizer(_LAST_MESSAGE_ONLY_TEMPLATE))

        with pytest.raises(RuntimeError, match="assistant probe"):
            w._tokenize_feedback(_FEEDBACK)

    def test_unrenderable_template_raises(self) -> None:
        w = _templated_wrapper(_templated_tokenizer("{%- for %}"))

        with pytest.raises(RuntimeError, match="Could not render the chat template"):
            w._tokenize_feedback(_FEEDBACK)

    def test_a_non_strict_wrapper_warns_and_falls_back(self) -> None:
        w = _templated_wrapper(
            _templated_tokenizer(_LAST_MESSAGE_ONLY_TEMPLATE),
            strict_chat_template_boundary=False,
        )

        with pytest.warns(UserWarning, match="ChatML markers"):
            ids = w._tokenize_feedback(_FEEDBACK)

        assert w.tokenizer.decode(ids[0].tolist()).startswith("<|im_end|>")


class TestEnvSystemPrompt:
    """An env-declared system prompt is rendered in the system role."""

    def _wrapper(self, system_prompt) -> TokenObservationWrapper:
        return _templated_wrapper(
            _templated_tokenizer(_THINKING_TEMPLATE, {"enable_thinking": False}),
            env=SimpleNamespace(system_prompt=system_prompt),
        )

    def test_initial_prompt_matches_an_offline_rendering(self) -> None:
        w = self._wrapper(_SYSTEM_PROMPT)

        rendered = w._tokenize_initial_prompt(_INITIAL_OBS)
        offline = w.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _INITIAL_OBS},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        assert rendered["input_ids"][0].tolist() == list(offline["input_ids"])

    def test_initial_prompt_opens_a_system_turn_carrying_the_prompt(self) -> None:
        w = self._wrapper(_SYSTEM_PROMPT)

        opening = w.tokenizer.decode(
            w._tokenize_initial_prompt(_INITIAL_OBS)["input_ids"][0].tolist(),
        )

        assert opening.startswith(f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n")
        assert f"<|im_start|>user\n{_INITIAL_OBS}<|im_end|>\n" in opening
        assert opening.endswith(_THINK_CLOSED_PROMPT)
        assert opening.count(_SYSTEM_PROMPT) == 1

    def test_an_env_without_a_system_prompt_renders_the_user_turn_alone(self) -> None:
        assert _templated_wrapper(
            _templated_tokenizer(_THINKING_TEMPLATE),
        )._initial_messages(_INITIAL_OBS) == [
            {"role": "user", "content": _INITIAL_OBS},
        ]

    @pytest.mark.parametrize("system_prompt", ["", "   \n ", 17, None])
    def test_a_non_prompt_attribute_is_ignored(self, system_prompt) -> None:
        w = self._wrapper(system_prompt)

        assert w._env_system_prompt() is None
        assert w._initial_messages(_INITIAL_OBS) == [
            {"role": "user", "content": _INITIAL_OBS},
        ]
