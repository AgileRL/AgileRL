# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for RolloutHarness prompt fields and context-overflow handling."""

from __future__ import annotations

import itertools
import re
import threading
from typing import ClassVar

import pytest
import torch

from agilerl.llm_envs import (
    RolloutCollector,
    RolloutHarness,
)
from agilerl.llm_envs.rollout import _mix_seed
from agilerl.llm_envs.rubrics import reward_fn_to_rubric
from tests.helpers.rollout_doubles import (
    FakeEnvClient,
    RolloutEnvDoubleMixin,
    bare_rollout_env,
)

# The boundary marker is a per-render ``uuid4().hex`` (32 lowercase hex chars);
# a correctly sliced boundary contains no such run.
_UUID4_HEX = re.compile(r"[0-9a-f]{32}")


class _StubTokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(ids)


def test_max_prompt_tokens_for_model_len() -> None:
    from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len

    assert max_prompt_tokens_for_model_len(8192, 1024) == 8192 - 1024
    assert max_prompt_tokens_for_model_len(100, 50) == 50
    assert max_prompt_tokens_for_model_len(100, None) == 99


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

    def reset(self, seed: int | None = None, *, row_index: int | None = None):
        del seed, row_index
        return "hello", {}

    def step(self, gen_text: str):
        self.last_gen = gen_text
        return "", 1.0, True, False, {}


class TestRolloutEnvReset:
    def test_reset_returns_tuple_with_text_and_sets_prompt_len(self, serve_env) -> None:
        inner = _RecordingGemEnv()
        w = RolloutHarness(
            serve_env(inner),
            _ChrTokenizer(),
            max_turns=3,
            apply_chat_template=False,
        )
        obs, info = w.reset()
        assert isinstance(info, dict)
        assert set(obs.keys()) >= {"input_ids"}
        assert w._prompt_text == "hello"
        assert w._last_full_prompt_token_len == obs["input_ids"].shape[1]


class TestRolloutEnvStep:
    def test_step_from_full_completion_slices_generation(self, serve_env) -> None:
        inner = _RecordingGemEnv()
        w = RolloutHarness(
            serve_env(inner),
            _ChrTokenizer(),
            max_turns=3,
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
        w = bare_rollout_env()
        w._last_full_prompt_token_len = None
        with pytest.raises(RuntimeError, match="requires a prior reset"):
            w.step(torch.ones(1, 2, dtype=torch.long))

    def test_chat_template_paths_and_nonterminal_step_feedback_append(
        self, serve_env
    ) -> None:
        env = _NonTerminalEnv()
        w = RolloutHarness(
            serve_env(env),
            _ChatTokenizer(),
            max_turns=2,
            apply_chat_template=True,
            strict_chat_template_boundary=False,
        )
        obs, _ = w.reset()
        assert obs["input_ids"].dtype == torch.long

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

    def test_non_chat_feedback_tokenization_path(self, serve_env) -> None:
        env = _NonTerminalEnv()
        w = RolloutHarness(
            serve_env(env),
            _ChrTokenizer(),
            max_turns=2,
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

    def test_strict_mode_terminates_on_context_overflow(self, serve_env) -> None:
        """When the cumulative prompt would exceed
        ``max_model_len - max_output_tokens``, the trajectory ends with
        ``truncated=True`` and no next prompt.
        """
        env = _NonTerminalEnv()
        # Tiny budget: 20 - 4 = 16 prompt tokens. Initial prompt "P:hello\nS"
        # is 9 char-tokens; +2 gen +12 feedback ("F:feedback\nT") => 23 > 16.
        w = RolloutHarness(
            serve_env(env),
            _ChrTokenizer(),
            max_turns=4,
            apply_chat_template=False,
            max_model_len=20,
            max_output_tokens=4,
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
        assert info == {}


class TestRolloutEnvChatTemplateBoundary:
    """Verify the assistant→user→assistant boundary is computed via the
    tokenizer's chat template rather than hard-coded ChatML markers.
    """

    def test_gemma_style_template_emits_full_boundary(self) -> None:
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_gemma_chat)

        out = w._chat_template_boundary_ids("FEEDBACK")

        assert out is not None
        decoded = "".join(chr(int(x)) for x in out[0].tolist())
        # Must close the assistant turn, open a user turn with the feedback,
        # close it, then open a fresh model turn. Placeholder must not leak.
        assert not _UUID4_HEX.search(decoded)
        assert decoded.startswith("<end_of_turn>\n<start_of_turn>user\n")
        assert "FEEDBACK" in decoded
        assert decoded.endswith("<start_of_turn>model\n")

    def test_qwen_style_template_emits_full_boundary(self) -> None:
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_chatml)

        out = w._chat_template_boundary_ids("FEEDBACK")

        assert out is not None
        decoded = "".join(chr(int(x)) for x in out[0].tolist())
        assert not _UUID4_HEX.search(decoded)
        assert decoded.startswith("<|im_end|>\n<|im_start|>user\n")
        assert "FEEDBACK" in decoded
        assert decoded.endswith("<|im_start|>assistant\n")

    def test_llama_style_template_emits_full_boundary(self) -> None:
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_llama)

        out = w._chat_template_boundary_ids("FEEDBACK")

        assert out is not None
        decoded = "".join(chr(int(x)) for x in out[0].tolist())
        assert not _UUID4_HEX.search(decoded)
        # Should close assistant via <|eot_id|> then open a user header.
        assert decoded.startswith("<|eot_id|><|start_header_id|>user")
        assert "FEEDBACK" in decoded
        assert decoded.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_returns_none_when_template_renderer_raises(self) -> None:
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_raises)
        assert w._chat_template_boundary_ids("F") is None

    def test_returns_none_when_placeholder_is_stripped(self) -> None:
        # Pathological template whose render drops content entirely; the
        # placeholder doesn't survive, so we can't slice. Caller falls back.
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_drops_content)
        assert w._chat_template_boundary_ids("F") is None

    def test_returns_none_when_render_is_not_a_string(self) -> None:
        # Some tokenizers tokenize regardless of ``tokenize=False`` and hand
        # back ids; we can only slice a string render, so fall back.
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_returns_ids)
        assert w._chat_template_boundary_ids("F") is None

    def test_returns_none_when_boundary_text_is_empty(self) -> None:
        # Render ends exactly at the placeholder -> nothing after it to
        # tokenize as the boundary, so fall back.
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_ends_at_placeholder)
        assert w._chat_template_boundary_ids("F") is None

    def test_returns_none_when_boundary_encodes_to_no_tokens(self) -> None:
        # A tokenizer that maps the boundary text to zero ids gives us
        # nothing to append, so fall back.
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _EmptyEncodeTokenizer(_render_chatml)
        assert w._chat_template_boundary_ids("F") is None

    def test_tokenize_feedback_prefers_chat_template_boundary(self) -> None:
        # With a working (Gemma-style) template, _tokenize_feedback must
        # return the template-derived boundary — not the hard-coded ChatML
        # fallback markers.
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_gemma_chat)
        out = w._tokenize_feedback("FEEDBACK")
        decoded = "".join(chr(int(x)) for x in out[0].tolist())
        assert decoded.startswith("<end_of_turn>\n<start_of_turn>user\n")
        assert decoded.endswith("<start_of_turn>model\n")
        assert "<|im_start|>" not in decoded  # ChatML fallback not used

    def test_strict_boundary_raises_instead_of_falling_back(self) -> None:
        # Under the default strict setting a template that cannot render the
        # boundary is an error: emitting ChatML markers for a non-ChatML
        # tokenizer would malform the transcript silently.
        w = bare_rollout_env()
        w.apply_chat_template = True
        w._strict_chat_template_boundary = True
        w.tokenizer = _ChrTokenizerWithChatTemplateBroken()
        with pytest.raises(RuntimeError, match="could not render a feedback turn"):
            w._tokenize_feedback("F")

    def test_full_tokenize_feedback_falls_back_to_chatml(self) -> None:
        # If the chat-template path returns None (no apply_chat_template at
        # all on the tokenizer), _tokenize_feedback falls back to ChatML so
        # we never crash.
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChrTokenizerWithChatTemplateBroken()
        out = w._tokenize_feedback("F")
        assert out.shape[0] == 1
        assert out.shape[1] > 0


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


class _KwargsRecordingChatTokenizer(_ChatTemplateRecordingTokenizer):
    """Records the extra kwargs each ``apply_chat_template`` call receives."""

    def __init__(self, renderer):
        super().__init__(renderer)
        self.calls: list[dict] = []

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs,
    ):
        self.calls.append(dict(kwargs))
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class TestRolloutEnvChatTemplateKwargs:
    """Configured ``chat_template_kwargs`` reach every chat-template render."""

    def test_kwargs_forwarded_on_initial_prompt(self) -> None:
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.chat_template_kwargs = {"enable_thinking": False}
        w.tokenizer = _KwargsRecordingChatTokenizer(_render_chatml)
        w._tokenize_initial_prompt("hi")
        assert w.tokenizer.calls == [{"enable_thinking": False}]

    def test_kwargs_forwarded_on_boundary_frame_render(self) -> None:
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.chat_template_kwargs = {"enable_thinking": False}
        w.tokenizer = _KwargsRecordingChatTokenizer(_render_chatml)
        out = w._chat_template_boundary_ids("FEEDBACK")
        assert out is not None
        assert w.tokenizer.calls == [{"enable_thinking": False}]

    def test_env_tools_win_over_configured_tools_entry(self) -> None:
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.chat_template_kwargs = {"enable_thinking": False, "tools": ["stale"]}
        w.tools = [{"type": "function"}]
        w.tokenizer = _KwargsRecordingChatTokenizer(_render_chatml)
        w._tokenize_initial_prompt("hi")
        assert w.tokenizer.calls == [
            {"enable_thinking": False, "tools": [{"type": "function"}]}
        ]

    def test_constructor_kwargs_reach_reset_render(self) -> None:
        tokenizer = _KwargsRecordingChatTokenizer(_render_chatml)
        w = RolloutHarness(
            FakeEnvClient(),
            tokenizer,
            max_turns=1,
            chat_template_kwargs={"enable_thinking": False},
        )
        w.reset()
        assert tokenizer.calls == [{"enable_thinking": False}]


class TestRolloutEnvPolicyObservationFromState:
    def test_policy_prompt_returns_current_prompt_fields(self) -> None:
        """The policy prompt carries the current ``input_ids`` directly."""
        w = bare_rollout_env()
        w.tokenizer = _StubTokenizer()
        w.full_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        w.turn_boundaries = []
        obs = w._policy_prompt_from_state()
        assert set(obs.keys()) == {"input_ids"}
        assert obs["input_ids"].shape[1] == 4

    def test_policy_observation_raises_without_full_ids(self) -> None:
        w = bare_rollout_env()
        w.full_ids = None
        with pytest.raises(RuntimeError, match="No prompt"):
            w._policy_prompt_from_state()


class TestRolloutEnvFromDataset:
    """from_dataset bundles (question, answer) rows + a rubric into a single-turn env."""

    _ROWS: ClassVar[list[dict]] = [
        {"question": "q0", "answer": "a0"},
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]
    _TEST_ROWS: ClassVar[list[dict]] = [{"question": "tq0", "answer": "ta0"}]

    def _make(self):
        calls: list[tuple] = []

        def reward_fn(completion, answer, question):
            calls.append((completion, answer, question))
            return 1.0 if answer == "a1" else 0.25

        env = RolloutHarness.from_dataset(
            self._ROWS,
            reward_fn_to_rubric(reward_fn),
            _ChrTokenizer(),
            test_dataset=self._TEST_ROWS,
            prompt_builder=lambda row: f"P:{row['question']}",
            apply_chat_template=False,
        )
        return env, calls

    def _reset_prompt(self, env, **kwargs) -> str:
        obs, _info = env.reset(**kwargs)
        return _ChrTokenizer().decode(obs["input_ids"][0].tolist())

    def test_reset_serves_pinned_row_and_step_scores_single_turn(self) -> None:
        env, calls = self._make()
        assert env.dataset_size == 3
        obs, _info = env.reset(row_index=1)
        assert _ChrTokenizer().decode(obs["input_ids"][0].tolist()) == "P:q1"
        gen = torch.tensor([[ord("z")]], dtype=torch.long)
        _obs, reward, terminated, truncated, _i = env.step(
            torch.cat([obs["input_ids"], gen], dim=1)
        )
        assert calls == [("z", "a1", "q1")]
        assert reward == 1.0
        assert terminated is True
        assert truncated is False

    def test_bare_reset_walks_rows_and_wraps(self) -> None:
        env, _calls = self._make()
        prompts = [self._reset_prompt(env) for _ in range(4)]
        assert prompts == ["P:q0", "P:q1", "P:q2", "P:q0"]

    def test_eval_mode_serves_test_rows(self) -> None:
        env, _calls = self._make()
        with env.eval_mode():
            assert self._reset_prompt(env, row_index=0) == "P:tq0"
        assert self._reset_prompt(env, row_index=0) == "P:q0"

    def test_default_prompt_is_str_question_without_builder(self) -> None:
        """With no ``prompt_builder`` the prompt is ``str(question)`` verbatim."""
        env = RolloutHarness.from_dataset(
            self._ROWS,
            reward_fn_to_rubric(lambda c, a, q: 0.0),
            _ChrTokenizer(),
            apply_chat_template=False,
        )
        assert self._reset_prompt(env, row_index=2) == "q2"

    def test_max_turns_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="single-turn"):
            RolloutHarness.from_dataset(
                self._ROWS,
                reward_fn_to_rubric(lambda c, a, q: 0.0),
                _ChrTokenizer(),
                max_turns=2,
            )


class _SyncStubEnv(RolloutEnvDoubleMixin):
    dataset_size = 0

    def __init__(self) -> None:
        self.turn_boundaries: list[int] = []
        self.reset_calls: list[int | None] = []
        self.close_calls = 0
        self.done = False
        self.current_prompt: dict = {}
        self.sampling_logps: list[torch.Tensor] = []

    def reset(self, seed: int | None = None):
        self.reset_calls.append(seed)
        self.done = False
        self.sampling_logps = []
        self.current_prompt = {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        return (self.current_prompt, {})

    def step(self, full_completion: torch.Tensor, sampling_logps=None):
        del full_completion
        self.turn_boundaries.append(1)
        if sampling_logps is not None:
            self.sampling_logps.append(sampling_logps)
        self.done = True
        self.current_prompt = {}
        return ({}, 1.0, True, False, {})

    def close(self) -> None:
        self.close_calls += 1

    def get_episode_data(self):
        return (
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
            torch.cat(self.sampling_logps) if self.sampling_logps else None,
        )


class TestBatchRolloutEnvReset:
    def test_sync_gem_vec_env_reset_seeds_per_batch_group(self) -> None:
        vec_env = RolloutCollector(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=10)
        seen = [env.reset_calls[-1] for env in vec_env.envs]
        assert seen == [_mix_seed(10)] * 2 + [_mix_seed(11)] * 2

    def test_sync_gem_vec_env_reset_with_none_seed(self) -> None:
        vec_env = RolloutCollector(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=None)
        seen = [env.reset_calls[-1] for env in vec_env.envs]
        assert seen == [None, None, None, None]

    def test_sync_gem_vec_env_reset_reuses_existing_trajectories(self) -> None:
        vec_env = RolloutCollector(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=10)
        _ = vec_env.reset(seed=20)
        seen = [env.reset_calls[-1] for env in vec_env.envs]
        assert seen == [_mix_seed(20)] * 2 + [_mix_seed(21)] * 2


class TestBatchRolloutEnvIoTimeout:
    def test_map_env_io_raises_on_a_hung_round_trip(self) -> None:
        """A round-trip that outlives ``io_timeout_s`` fails fast, not forever."""
        import threading
        import time

        release = threading.Event()
        vec = RolloutCollector(
            env_factory=_SyncStubEnv, batch_size=2, group_size=1, io_timeout_s=0.3
        )
        try:
            started = time.monotonic()
            with pytest.raises(
                TimeoutError, match="did not finish within io_timeout_s"
            ):
                vec._map_env_io([lambda: release.wait(30), lambda: 1])
            # Bounded by the deadline, not the 30s wait.
            assert time.monotonic() - started < 5.0
        finally:
            release.set()  # let the abandoned worker unblock and exit cleanly
            vec.close()

    def test_map_env_io_returns_results_within_the_deadline(self) -> None:
        """Quick round-trips complete and return in submission order."""
        vec = RolloutCollector(
            env_factory=_SyncStubEnv, batch_size=2, group_size=1, io_timeout_s=5.0
        )
        try:
            assert vec._map_env_io([lambda: 1, lambda: 2]) == [1, 2]
        finally:
            vec.close()

    def test_map_env_io_none_timeout_keeps_the_direct_single_thunk_path(self) -> None:
        """``io_timeout_s=None`` restores the unbounded direct call for one thunk."""
        vec = RolloutCollector(
            env_factory=_SyncStubEnv, batch_size=1, group_size=1, io_timeout_s=None
        )
        try:
            assert vec._map_env_io([]) == []  # no active envs -> nothing to run
            assert vec._map_env_io([lambda: 7]) == [7]
        finally:
            vec.close()


class TestBatchRolloutEnvStep:
    def test_sync_gem_vec_env_step_raises_when_sequence_count_mismatches_active(
        self,
    ) -> None:
        vec_env = RolloutCollector(
            env_factory=_SyncStubEnv,
            batch_size=1,
            group_size=2,
        )
        _ = vec_env.reset(seed=0)
        with pytest.raises(
            RuntimeError,
            match="Number of token sequences does not match number of active envs",
        ):
            vec_env.step([torch.ones(1, 5, dtype=torch.long)])

    def test_batch_rollout_env_step_happy_path_1d_and_2d_and_active_filtering(
        self,
    ) -> None:
        created = [
            _StepVariantEnv(done_after_step=False),
            _StepVariantEnv(done_after_step=True),
        ]
        idx = {"i": 0}

        def _factory():
            env = created[idx["i"]]
            idx["i"] += 1
            return env

        vec = RolloutCollector(env_factory=_factory, batch_size=1, group_size=2)
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

    def test_batch_rollout_env_step_raises_on_sampling_logps_count_mismatch(
        self,
    ) -> None:
        vec_env = RolloutCollector(
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

    def test_batch_rollout_env_step_accumulates_sampling_logps_per_trajectory(
        self,
    ) -> None:
        """Each turn's vLLM sampling logprobs append onto that env's
        ``sampling_logps``; ``None`` rows (nothing captured) are skipped.
        ``get_trajectories`` concatenates across turns and keeps a per-env
        ``None`` for rows that never captured any.
        """
        vec = RolloutCollector(
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
        assert len(vec.envs[0].sampling_logps) == 2
        assert vec.envs[1].sampling_logps == []

        *_parts, sampling = vec.get_trajectories()
        assert sampling is not None
        assert torch.equal(sampling[0], torch.tensor([-0.1, -0.2, -0.3]))
        assert sampling[1] is None

    def test_batch_rollout_env_sampling_logps_collapse_to_none_when_uncaptured(
        self,
    ) -> None:
        """Without captured logprobs the rollout-wide entry is a single
        ``None`` (not a list of ``None``s), and a reset clears any logprobs
        accumulated in a previous rollout.
        """
        vec = RolloutCollector(
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
        assert len(vec.envs[0].sampling_logps) == 1
        _ = vec.reset(seed=1)
        assert vec.envs[0].sampling_logps == []
        *_parts, sampling = vec.get_trajectories()
        assert sampling is None


class TestBatchRolloutEnvClose:
    def test_batch_rollout_env_close_calls_underlying_env_close_once(self) -> None:
        vec_env = RolloutCollector(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=0)
        envs = list(vec_env.envs)
        vec_env.close()
        assert [env.close_calls for env in envs] == [1, 1, 1, 1]
        assert vec_env.envs == []


class TestBatchRolloutEnvInit:
    def test_batch_rollout_env_constructor_rejects_non_positive_sizes(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            _ = RolloutCollector(
                env_factory=_SyncStubEnv,
                batch_size=0,
                group_size=1,
            )
        with pytest.raises(ValueError, match="group_size must be > 0"):
            _ = RolloutCollector(
                env_factory=_SyncStubEnv,
                batch_size=1,
                group_size=0,
            )


def _stub_env(*, done: bool = False, prompt: dict | None = None) -> _SyncStubEnv:
    """A stub env preset to a given pool state (``done`` / ``current_prompt``)."""
    env = _SyncStubEnv()
    env.done = done
    env.current_prompt = {} if prompt is None else prompt
    return env


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


class TestRolloutEnvTokenizeInitialPrompt:
    def test_initial_prompt_unwraps_batched_token_id_lists(self) -> None:
        """Tokenizers returning ``[[ids]]`` (batch dim) and ``[ids]`` (flat)
        from ``apply_chat_template`` must produce identical ``(1, T)``
        tensors.
        """
        flat_ids = _ChatTokenizer().apply_chat_template(
            [{"role": "user", "content": "hi"}]
        )["input_ids"]

        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _NestedChatTokenizer()
        out = w._tokenize_initial_prompt("hi")
        assert out.shape == (1, len(flat_ids))
        assert out[0].tolist() == flat_ids


class _NonTerminalEnv:
    def __init__(self) -> None:
        self.calls = 0
        self.last_gen: str | None = None

    def reset(self, seed: int | None = None, *, row_index: int | None = None):
        del seed, row_index
        return "P:hello\nS", {}

    def step(self, gen_text: str):
        self.calls += 1
        self.last_gen = gen_text
        if self.calls == 1:
            return "F:feedback\nT", 0.5, False, False, {}
        return "done", 1.0, True, False, {}


class TestRolloutEnvGetEpisodeData:
    def test_get_episode_data_keeps_every_generated_token_trainable(self) -> None:
        # A generated token equal to the pad id (pad == eos on pad-less model
        # families) is real content — the sampled stop token — and must stay in
        # the action mask or the model never gets gradient toward stopping.
        w = bare_rollout_env()
        w.max_turns = 3
        w.full_ids = torch.tensor([[9, 5, 0, 7, 8]], dtype=torch.long)
        w.turn_boundaries = [(1, 3, 0), (3, 5, 1)]
        w.turn_rewards = [1.5]
        full_ids, action_mask, turn_ids, rewards, _logps = w.get_episode_data()
        assert torch.equal(full_ids, w.full_ids)
        assert action_mask.dtype == torch.bool
        assert turn_ids.dtype == torch.long
        assert rewards.tolist() == [1.5, 0.0, 0.0]
        assert action_mask[0].tolist() == [True, True, True, True]
        assert turn_ids[0].tolist() == [0, 0, 1, 1]

    def test_get_episode_data_raises_without_reset(self) -> None:
        w = bare_rollout_env()
        w.full_ids = None
        with pytest.raises(RuntimeError, match="No episode data"):
            w.get_episode_data()


class _TerminatorTokenizer:
    """Char tokenizer whose id 7 is a declared special (a turn terminator)."""

    all_special_ids: ClassVar[list[int]] = [7]

    def __call__(self, texts, **kwargs):
        ids = [[ord(c) for c in texts[0]]]
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(x)) for x in ids)


class TestFeedbackTerminatorDedupe:
    def _env(self, tokenizer) -> RolloutHarness:
        w = bare_rollout_env()
        w.tokenizer = tokenizer
        w.apply_chat_template = True
        w.max_turns = 3
        w._max_model_len = None
        w._max_output_tokens = None
        w._turn_idx = 0
        w.turn_rewards = []
        w._feedback_texts = []
        w._gen_texts = []
        w.turn_boundaries = []
        # Frame whose prefix begins with the terminator (chr(7) encodes to id 7).
        w._boundary_parts = ("\x07U:", ":A")
        w._boundary_parts_known = True
        return w

    def test_sampled_terminator_is_not_doubled(self) -> None:
        w = self._env(_TerminatorTokenizer())
        w.full_ids = torch.tensor([[65, 66, 7]], dtype=torch.long)  # ends with EOS
        w._step_apply(("fb", 0.5, False, False, {}))
        # One terminator total: the sampled one; the frame's duplicate is dropped.
        assert w.full_ids[0].tolist().count(7) == 1

    def test_truncated_turn_still_gets_the_frame_terminator(self) -> None:
        w = self._env(_TerminatorTokenizer())
        w.full_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)  # no EOS sampled
        w._step_apply(("fb", 0.5, False, False, {}))
        assert w.full_ids[0].tolist().count(7) == 1

    def test_non_special_equal_token_is_kept(self) -> None:
        tokenizer = _TerminatorTokenizer()
        tokenizer.all_special_ids = []
        w = self._env(tokenizer)
        w.full_ids = torch.tensor([[65, 66, 7]], dtype=torch.long)
        w._step_apply(("fb", 0.5, False, False, {}))
        # id 7 is ordinary content here; nothing may be silently dropped.
        assert w.full_ids[0].tolist().count(7) == 2


class TestRolloutEnvSpecialTokenGuards:
    def test_raw_encode_adds_no_specials(self) -> None:
        # apply_chat_template=False means the text is already fully rendered
        # (e.g. a template-baked BOS); re-adding specials would double the BOS.
        class _BosAddingTokenizer:
            def __call__(self, texts, add_special_tokens=True, **kwargs):
                ids = [ord(c) for c in texts[0]]
                if add_special_tokens:
                    ids = [1, *ids]
                t = torch.tensor([ids], dtype=torch.long)
                return {"input_ids": t, "attention_mask": torch.ones_like(t)}

        w = bare_rollout_env()
        w.tokenizer = _BosAddingTokenizer()
        w.apply_chat_template = False
        out = w._tokenize_initial_prompt("hi")
        assert out[0].tolist() == [ord("h"), ord("i")]

    def test_chatml_fallback_warns_once_per_site(self) -> None:
        w = bare_rollout_env()
        w.tokenizer = _ChrTokenizer()  # no apply_chat_template -> frame render fails
        w.apply_chat_template = True
        with pytest.warns(UserWarning, match="ChatML markers"):
            w._tokenize_feedback("fb")

    def test_warns_when_template_cannot_render_tools(self) -> None:
        from tests.helpers.rollout_doubles import FakeEnvClient

        tokenizer = _ChrTokenizer()
        tokenizer.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
        env = RolloutHarness(
            FakeEnvClient(tools=[{"type": "function", "function": {"name": "f"}}]),
            tokenizer,
            max_turns=1,
        )
        with pytest.warns(UserWarning, match="never references 'tools'"):
            _ = env.tools


class TestRolloutEnvClientKwargGuard:
    def test_transport_kwargs_rejected_with_a_prebuilt_client(self) -> None:
        with pytest.raises(ValueError, match="already-built env client"):
            RolloutHarness(FakeEnvClient(), _ChrTokenizer(), max_turns=1, timeout_s=5.0)


class TestPerEpisodeAssignmentBranches:
    def _collector(self) -> RolloutCollector:
        return RolloutCollector(
            env_factory=_SyncStubEnv,
            batch_size=1,
            group_size=2,
            base_seed=7,
        )

    def test_unpinned_episode_and_id_snapshot(self) -> None:
        vec = self._collector()
        try:
            vec.reset_episode("free", logical_slot=None)  # unpinned: (None, None)
            assert vec.active_episode_ids() == ["free"]
        finally:
            vec.close()

    def test_out_of_window_slot_raises(self) -> None:
        vec = self._collector()
        try:
            with pytest.raises(IndexError, match="outside the current rollout"):
                vec.reset_episode("oob", logical_slot=99)
        finally:
            vec.close()


class TestBatchRolloutEnvHelpers:
    def test_is_initialized_flips_once_all_slots_built(self) -> None:
        vec = RolloutCollector(env_factory=_SyncStubEnv, batch_size=1, group_size=2)
        assert vec._is_initialized is False
        for _ in range(2):
            vec.envs.append(_SyncStubEnv())
        assert vec._is_initialized is True

    def test_get_rubric_score_means_averages_and_nans(self) -> None:
        vec = RolloutCollector(env_factory=_SyncStubEnv, batch_size=1, group_size=2)
        e0 = _SyncStubEnv()
        e1 = _SyncStubEnv()
        e0.rubric_score_sums = {"fmt": 1.0, "correct": 2.0}
        e1.rubric_score_sums = {"fmt": 3.0}
        vec.envs = [e0, e1]
        vec.rubric_component_names = ("fmt", "correct", "missing")
        means = vec.get_rubric_score_means()
        assert means["fmt"] == 2.0
        assert means["correct"] == 2.0
        assert means["missing"] != means["missing"]  # NaN


class TestBatchRolloutEnvActiveEnvs:
    def test_active_envs_excludes_done_and_preserves_list_order(self) -> None:
        vec = RolloutCollector(env_factory=_SyncStubEnv, batch_size=2, group_size=2)
        e0 = _stub_env(done=False)
        e1 = _stub_env(done=True)
        e2 = _stub_env(done=False)
        vec.envs = [e0, e1, e2]
        # done e1 excluded; the rest keep their list (batch/group) order.
        assert vec._active_envs() == [e0, e2]


class TestBatchRolloutEnvGetPrompts:
    def test_get_prompts_returns_none_when_no_active(self) -> None:
        vec = RolloutCollector(env_factory=_SyncStubEnv, batch_size=1, group_size=2)
        done_env = _stub_env(
            done=True,
            prompt={
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            },
        )
        active_env = _stub_env(
            done=False,
            prompt={
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
        )
        vec.envs = [done_env, active_env]

        prompts = vec._get_prompts()
        assert prompts is not None
        assert isinstance(prompts, list)
        assert len(prompts) == 1
        assert prompts[0]["input_ids"].shape == (1, 3)
        assert prompts[0]["attention_mask"].shape == (1, 3)

        active_env.done = True
        assert vec._get_prompts() is None

    def test_get_prompts_returns_active_in_index_order(self) -> None:
        vec = RolloutCollector(env_factory=_SyncStubEnv, batch_size=1, group_size=2)
        a = _stub_env(
            done=False,
            prompt={
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            },
        )
        b = _stub_env(
            done=False,
            prompt={
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
        )
        vec.envs = [a, b]
        prompts = vec._get_prompts()
        assert prompts is not None
        assert len(prompts) == 2
        assert [int(p["input_ids"].shape[1]) for p in prompts] == [2, 3]
        assert prompts[0]["input_ids"].shape == (1, 2)
        assert prompts[1]["input_ids"].shape == (1, 3)


class _StepVariantEnv(RolloutEnvDoubleMixin):
    dataset_size = 0

    def __init__(self, done_after_step: bool, include_turn_boundaries: bool = True):
        self.done_after_step = done_after_step
        self.include_turn_boundaries = include_turn_boundaries
        self.step_shapes: list[tuple[int, ...]] = []
        self.done = False
        self.current_prompt: dict = {}
        self.sampling_logps: list[torch.Tensor] = []
        self.turn_boundaries: list[int] = []

    def reset(self, seed: int | None = None):
        del seed
        self.done = False
        self.sampling_logps = []
        self.current_prompt = {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        return (self.current_prompt, {})

    def step(self, full_completion: torch.Tensor, sampling_logps=None):
        self.step_shapes.append(tuple(full_completion.shape))
        if self.include_turn_boundaries:
            self.turn_boundaries.append(1)
        if sampling_logps is not None:
            self.sampling_logps.append(sampling_logps)
        if self.done_after_step:
            self.done = True
            self.current_prompt = {}
            return ({}, 1.0, True, False, {})
        self.current_prompt = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        return (self.current_prompt, 0.5, False, False, {})

    def close(self) -> None:
        pass

    def get_episode_data(self):
        return (
            torch.ones(1, 5, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
            torch.cat(self.sampling_logps) if self.sampling_logps else None,
        )


class TestBatchRolloutEnvGetTrajectories:
    def test_batch_rollout_env_get_trajectories_counts_steps_with_and_without_turn_boundaries(
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

        vec = RolloutCollector(env_factory=_factory, batch_size=1, group_size=2)
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


def _render_ends_at_feedback(messages, add_generation_prompt: bool) -> str:
    """Render whose last bytes are the feedback placeholder — empty suffix."""
    del add_generation_prompt
    # ...assistant_ph...<prefix>...feedback_ph  (nothing after feedback -> suffix "")
    return f"A{messages[1]['content']}B{messages[2]['content']}"


class TestRolloutEnvChatTemplateBoundaryExtra:
    """Remaining boundary branches: the cached frame and an empty-suffix render."""

    def test_boundary_frame_is_cached_after_first_render(self) -> None:
        # A second call reuses the cached (prefix, suffix) frame instead of
        # re-rendering the template.
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_gemma_chat)
        first = w._chat_template_boundary_ids("FEEDBACK")
        assert first is not None
        assert w._boundary_parts_known
        second = w._chat_template_boundary_ids("FEEDBACK")
        assert second is not None
        assert torch.equal(first, second)

    def test_returns_none_when_boundary_suffix_is_empty(self) -> None:
        # Placeholders are found and ordered, but nothing follows the feedback
        # slot, so the suffix is empty and we fall back.
        w = bare_rollout_env()
        w.apply_chat_template = True
        w.tokenizer = _ChatTemplateRecordingTokenizer(_render_ends_at_feedback)
        assert w._chat_template_boundary_ids("F") is None


class TestRolloutEnvEvalMode:
    def test_eval_mode_delegates_to_the_env_client(self) -> None:
        """``eval_mode`` runs the block inside the client's own eval context."""
        w = bare_rollout_env()
        client = FakeEnvClient()
        w._env_client = client
        with w.eval_mode():
            assert client.eval_mode_depth == 1
        assert client.eval_mode_depth == 0
        assert client.eval_mode_entries == 1


class TestRolloutEnvStepSamplingLogps:
    def test_step_records_sampling_logps(self, serve_env) -> None:
        """A turn's vLLM sampling logprobs are appended to the episode's row."""
        inner = _RecordingGemEnv()
        w = RolloutHarness(
            serve_env(inner),
            _ChrTokenizer(),
            max_turns=3,
            apply_chat_template=False,
        )
        obs, _ = w.reset()
        full = torch.cat(
            [obs["input_ids"], torch.tensor([[ord("x")]], dtype=torch.long)], dim=1
        )
        logps = torch.tensor([-0.5])
        w.step(full, sampling_logps=logps)
        assert len(w.sampling_logps) == 1
        assert torch.equal(w.sampling_logps[0], logps)


class _RowRecordingEnv(_SyncStubEnv):
    dataset_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.row_calls: list[int | None] = []

    def reset(self, seed: int | None = None, *, row_index: int | None = None):
        self.row_calls.append(row_index)
        return super().reset(seed=seed)


class _FailingFirstResetEnv(_SyncStubEnv):
    fail_next = True

    def reset(self, seed: int | None = None):
        if _FailingFirstResetEnv.fail_next:
            _FailingFirstResetEnv.fail_next = False
            msg = "transient reset failure"
            raise RuntimeError(msg)
        return super().reset(seed=seed)


class _BlockingStepEnv(_SyncStubEnv):
    step_entered: ClassVar[threading.Event]
    release_step: ClassVar[threading.Event]

    def _step_env(self, gen_text: str):
        type(self).step_entered.set()
        assert type(self).release_step.wait(timeout=10.0), "step was never released"
        return super()._step_env(gen_text)


class TestBatchRolloutEnvPerEpisode:
    def _collector(self, **kwargs) -> RolloutCollector:
        defaults: dict = {
            "env_factory": _SyncStubEnv,
            "batch_size": 2,
            "group_size": 2,
            "base_seed": 10,
        }
        defaults.update(kwargs)
        return RolloutCollector(**defaults)

    def test_interleaved_episodes_share_group_tasks_and_release_slots(self) -> None:
        vec_env = self._collector()
        vec_env.set_group_seed(0)
        for i in range(4):
            prompts, _info = vec_env.reset_episode(f"ep{i}", logical_slot=i)
            assert prompts
        assert vec_env.active_episode_count() == 4
        seen = [env.reset_calls[-1] for env in vec_env.envs]
        assert seen == [_mix_seed(10)] * 2 + [_mix_seed(11)] * 2
        completion = torch.ones(1, 5, dtype=torch.long)
        for i in range(4):
            _prompt, reward, terminated, _trunc, _info = vec_env.step_episode(
                f"ep{i}", completion
            )
            assert reward == 1.0
            assert terminated
        for i in range(4):
            data = vec_env.get_episode_data(f"ep{i}")
            assert data[0].shape == (1, 4)
        assert vec_env.active_episode_count() == 0
        # All slots released: a second window fills without blocking.
        vec_env.set_group_seed(2)
        for i in range(4):
            vec_env.reset_episode(f"w2-ep{i}", logical_slot=i)
        seen = [env.reset_calls[-1] for env in vec_env.envs]
        assert seen == [_mix_seed(12)] * 2 + [_mix_seed(13)] * 2

    def test_dataset_rows_are_group_contiguous(self) -> None:
        vec_env = self._collector(env_factory=_RowRecordingEnv)
        for i in range(4):
            vec_env.reset_episode(f"ep{i}", logical_slot=i)
        rows = [env.row_calls[-1] for env in vec_env.envs]
        assert rows[0] == rows[1]
        assert rows[2] == rows[3]
        assert rows[0] != rows[2]
        assert all(r is not None and 0 <= r < 4 for r in rows)

    def test_duplicate_episode_id_raises(self) -> None:
        vec_env = self._collector()
        vec_env.reset_episode("ep0", logical_slot=0)
        with pytest.raises(RuntimeError, match="already active"):
            vec_env.reset_episode("ep0", logical_slot=1)

    def test_step_unknown_episode_raises(self) -> None:
        vec_env = self._collector()
        vec_env.reset_episode("ep0", logical_slot=0)
        with pytest.raises(KeyError, match="not active"):
            vec_env.step_episode("ghost", torch.ones(1, 5, dtype=torch.long))

    def test_finalize_is_idempotent_and_get_episode_data_is_strict(self) -> None:
        vec_env = self._collector()
        vec_env.reset_episode("ep0", logical_slot=0)
        vec_env.step_episode("ep0", torch.ones(1, 5, dtype=torch.long))
        assert vec_env.finalize_episode("ep0") is not None
        assert vec_env.finalize_episode("ep0") is None
        with pytest.raises(KeyError, match="not active"):
            vec_env.get_episode_data("ep0")

    def test_slot_exhaustion_times_out(self) -> None:
        vec_env = self._collector(
            batch_size=1, group_size=1, slot_acquire_timeout_s=0.05
        )
        vec_env.reset_episode("ep0", logical_slot=0)
        with pytest.raises(TimeoutError, match="acquiring a free env slot"):
            vec_env.reset_episode("ep1", logical_slot=0)

    def test_failed_reset_releases_its_slot(self) -> None:
        _FailingFirstResetEnv.fail_next = True
        vec_env = self._collector(
            env_factory=_FailingFirstResetEnv,
            batch_size=1,
            group_size=1,
            slot_acquire_timeout_s=0.05,
        )
        with pytest.raises(RuntimeError, match="transient reset failure"):
            vec_env.reset_episode("ep0", logical_slot=0)
        assert vec_env.active_episode_count() == 0
        prompts, _info = vec_env.reset_episode("ep1", logical_slot=0)
        assert prompts

    def test_geometry_update_guards_and_reshapes(self) -> None:
        vec_env = self._collector()
        vec_env.reset_episode("ep0", logical_slot=0)
        with pytest.raises(RuntimeError, match="still active"):
            vec_env.update_rollout_geometry(rollout_batch_size=1, group_size=2)
        vec_env.finalize_episode("ep0")
        vec_env.update_rollout_geometry(rollout_batch_size=1, group_size=2)
        assert vec_env.batch_size * vec_env.group_size == 2
        with pytest.raises(ValueError, match="exceeds the slot pool"):
            vec_env.update_rollout_geometry(rollout_batch_size=5, group_size=1)

    def test_concurrent_drivers_stay_consistent(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        vec_env = self._collector()

        def run_episode(i: int) -> tuple:
            eid = f"ep{i}"
            vec_env.reset_episode(eid, logical_slot=i)
            vec_env.step_episode(eid, torch.ones(1, 5, dtype=torch.long))
            return vec_env.get_episode_data(eid)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run_episode, range(4)))
        assert len(results) == 4
        assert vec_env.active_episode_count() == 0
        assert all(r[0].shape == (1, 4) for r in results)

    def _blocked_step_thread(
        self, vec_env: RolloutCollector, episode_id: str
    ) -> tuple[threading.Thread, list[Exception]]:
        caught: list[Exception] = []

        def drive() -> None:
            try:
                vec_env.step_episode(episode_id, torch.ones(1, 5, dtype=torch.long))
            except Exception as exc:
                caught.append(exc)

        thread = threading.Thread(target=drive, daemon=True)
        thread.start()
        assert _BlockingStepEnv.step_entered.wait(timeout=10.0)
        return thread, caught

    def test_stale_step_never_lands_on_a_reused_slot(self) -> None:
        _BlockingStepEnv.step_entered = threading.Event()
        _BlockingStepEnv.release_step = threading.Event()
        vec_env = self._collector(
            env_factory=_BlockingStepEnv, batch_size=1, group_size=1
        )
        vec_env.reset_episode("ep-old", logical_slot=0)
        thread, caught = self._blocked_step_thread(vec_env, "ep-old")

        assert vec_env.finalize_episode("ep-old") is not None
        vec_env.reset_episode("ep-new", logical_slot=0)
        _BlockingStepEnv.release_step.set()
        thread.join(timeout=10.0)
        assert not thread.is_alive()

        assert len(caught) == 1
        assert isinstance(caught[0], RuntimeError)
        assert "no longer owns its env slot" in str(caught[0])
        # The successor episode saw no phantom turn and still steps normally.
        env = vec_env.envs[0]
        assert env.turn_boundaries == []
        _prompt, reward, terminated, _trunc, _info = vec_env.step_episode(
            "ep-new", torch.ones(1, 5, dtype=torch.long)
        )
        assert reward == 1.0
        assert terminated
        assert env.turn_boundaries == [1]

    def test_stale_step_is_rejected_when_the_episode_id_is_reused(self) -> None:
        _BlockingStepEnv.step_entered = threading.Event()
        _BlockingStepEnv.release_step = threading.Event()
        vec_env = self._collector(
            env_factory=_BlockingStepEnv, batch_size=1, group_size=1
        )
        vec_env.reset_episode("ep", logical_slot=0)
        thread, caught = self._blocked_step_thread(vec_env, "ep")

        vec_env.finalize_episode("ep")
        vec_env.reset_episode("ep", logical_slot=0)
        _BlockingStepEnv.release_step.set()
        thread.join(timeout=10.0)
        assert not thread.is_alive()

        # Identity alone cannot catch this: the reused id maps to the same slot,
        # so only the bumped activation token distinguishes the new episode.
        assert len(caught) == 1
        assert isinstance(caught[0], RuntimeError)
        assert "no longer owns its env slot" in str(caught[0])
        assert vec_env.envs[0].turn_boundaries == []


class TestMixSeed:
    def test_seeds_stay_within_the_numpy_seed_range(self) -> None:
        """An env seeding through numpy rejects anything at or above 2**32."""
        seeds = [_mix_seed(base) for base in range(0, 100_000, 7)]

        assert all(0 <= seed < 2**31 for seed in seeds)

    def test_consecutive_windows_land_far_apart(self) -> None:
        """The point of mixing: neighbouring inputs must not give neighbouring seeds."""
        consecutive = [_mix_seed(base) for base in range(64)]

        assert len(set(consecutive)) == 64
        gaps = [abs(b - a) for a, b in itertools.pairwise(consecutive)]
        assert min(gaps) > 1_000_000

    def test_is_deterministic(self) -> None:
        assert [_mix_seed(i) for i in range(16)] == [_mix_seed(i) for i in range(16)]
