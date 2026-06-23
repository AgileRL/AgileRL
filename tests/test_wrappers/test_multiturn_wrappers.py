"""Tests for RolloutHarness prompt fields and context-overflow handling."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from agilerl.llm_envs import (
    BatchRolloutEnv,
    RolloutBuffer,
    RolloutHarness,
    Trajectory,
)

# The boundary marker is a per-render ``uuid4().hex`` (32 lowercase hex chars);
# a correctly sliced boundary contains no such run.
_UUID4_HEX = re.compile(r"[0-9a-f]{32}")


class _StubTokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(ids)


def _bare_wrapper() -> RolloutHarness:
    w = RolloutHarness.__new__(RolloutHarness)
    w.tools = None  # optional config; __init__ default, read by the tokenize paths
    return w


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

    def encode(self, s: str) -> list[int]:
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


class TestRolloutHarnessReset:
    def test_reset_returns_tuple_with_text_and_sets_prompt_len(self) -> None:
        inner = _RecordingGemEnv()
        w = RolloutHarness(
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


class TestRolloutHarnessStep:
    def test_step_from_full_completion_slices_generation(self) -> None:
        inner = _RecordingGemEnv()
        w = RolloutHarness(
            inner,
            _ChrTokenizer(),
            max_turns=3,
            pad_id=None,
            apply_chat_template=False,
        )
        obs, _ = w.reset()
        pl = obs["input_ids"].shape[1]
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
        w = RolloutHarness(
            env,
            _ChatTokenizer(),
            max_turns=2,
            pad_id=None,
            apply_chat_template=True,
        )
        obs, _ = w.reset()
        assert obs["input_ids"].dtype == torch.long

        completion = torch.cat(
            [obs["input_ids"], torch.tensor([[7, 8]], dtype=torch.long)],
            dim=1,
        )
        next_obs, reward, terminated, truncated, _ = w.step(completion)
        assert reward == 0.5
        assert not terminated and not truncated
        assert env.last_gen == "7|8"
        assert next_obs["input_ids"].shape[1] > completion.shape[1]
        assert w._feedback_texts[-1] == "F:feedback\nT"

    def test_non_chat_feedback_tokenization_path(self) -> None:
        env = _NonTerminalEnv()
        w = RolloutHarness(
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
        assert not terminated and not truncated
        assert next_obs["input_ids"].shape[1] > completion.shape[1]

    def test_strict_mode_terminates_on_context_overflow(self) -> None:
        """When the cumulative prompt would exceed
        ``max_model_len - max_output_tokens``, the trajectory ends with
        ``truncated=True`` and an ``agilerl_context_overflow`` breadcrumb in
        ``info``."""
        env = _NonTerminalEnv()
        # Tiny budget: 20 - 4 = 16 prompt tokens. Initial prompt "P:hello\nS"
        # is 9 char-tokens; +2 gen +12 feedback ("F:feedback\nT") => 23 > 16.
        w = RolloutHarness(
            env,
            _ChrTokenizer(),
            max_turns=4,
            pad_id=None,
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
        assert "agilerl_context_overflow" in info
        overflow = info["agilerl_context_overflow"]
        assert overflow["max_prompt_tokens"] == 16
        assert overflow["max_model_len"] == 20
        assert overflow["max_output_tokens"] == 4
        assert overflow["full_prompt_len"] > overflow["max_prompt_tokens"]


class TestRolloutHarnessChatTemplateBoundary:
    """Verify the assistant→user→assistant boundary is computed via the
    tokenizer's chat template rather than hard-coded ChatML markers."""

    def test_gemma_style_template_emits_full_boundary(self) -> None:
        w = _bare_wrapper()
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
        w = _bare_wrapper()
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
        w = _bare_wrapper()
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
        # all on the tokenizer), _tokenize_feedback falls back to ChatML so
        # we never crash.
        w = _bare_wrapper()
        w.apply_chat_template = True
        w.tokenizer = _ChrTokenizerWithChatTemplateBroken()
        out = w._tokenize_feedback("F")
        assert out.shape[0] == 1 and out.shape[1] > 0


def _render_gemma_chat(messages, add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else m["role"]
        parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n")
    if add_generation_prompt:
        parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _render_chatml(messages, add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _render_llama(messages, add_generation_prompt: bool) -> str:
    parts = ["<|begin_of_text|>"]
    for m in messages:
        parts.append(
            f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
            f"{m['content']}<|eot_id|>"
        )
    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _render_raises(messages, add_generation_prompt: bool) -> str:
    raise RuntimeError("template error")


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


class TestRolloutHarnessPolicyObservationFromState:
    def test_policy_observation_returns_current_prompt_fields(self) -> None:
        """The policy observation carries the current ``input_ids`` directly."""
        w = _bare_wrapper()
        w.tokenizer = _StubTokenizer()
        w.full_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        w.turn_boundaries = []
        obs = w._policy_observation_from_state()
        assert set(obs.keys()) >= {"input_ids", "attention_mask", "text"}
        assert obs["input_ids"].shape[1] == 4

    def test_policy_observation_raises_without_full_ids(self) -> None:
        w = _bare_wrapper()
        w.full_ids = None
        with pytest.raises(RuntimeError, match="No prompt"):
            w._policy_observation_from_state()


class _SyncStubEnv:
    def __init__(self) -> None:
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


class TestBatchRolloutEnvReset:
    def test_sync_gem_vec_env_reset_seeds_per_batch_group(self) -> None:
        vec_env = BatchRolloutEnv(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=10)
        seen = [traj.env.reset_calls[-1] for traj in vec_env.trajectories]
        assert seen == [10, 10, 11, 11]

    def test_sync_gem_vec_env_reset_with_none_seed(self) -> None:
        vec_env = BatchRolloutEnv(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=None)
        seen = [traj.env.reset_calls[-1] for traj in vec_env.trajectories]
        assert seen == [None, None, None, None]

    def test_sync_gem_vec_env_reset_reuses_existing_trajectories(self) -> None:
        vec_env = BatchRolloutEnv(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=10)
        _ = vec_env.reset(seed=20)
        seen = [traj.env.reset_calls[-1] for traj in vec_env.trajectories]
        assert seen == [20, 20, 21, 21]


class TestBatchRolloutEnvStep:
    def test_sync_gem_vec_env_step_raises_when_completion_count_mismatches_active(
        self,
    ) -> None:
        vec_env = BatchRolloutEnv(
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

        vec = BatchRolloutEnv(env_factory=_factory, batch_size=1, group_size=2)
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
        vec_env = BatchRolloutEnv(
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
        """Each turn's vLLM sampling logprobs append onto that trajectory's
        ``Trajectory.sampling_logps``; ``None`` rows (nothing captured) are
        skipped. ``get_trajectories`` concatenates across turns and keeps a
        per-trajectory ``None`` for rows that never captured any."""
        vec = BatchRolloutEnv(
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

    def test_batch_rollout_env_sampling_logps_collapse_to_none_when_uncaptured(
        self,
    ) -> None:
        """Without captured logprobs the rollout-wide entry is a single
        ``None`` (not a list of ``None``s), and a reset clears any logprobs
        accumulated in a previous rollout."""
        vec = BatchRolloutEnv(
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


class TestBatchRolloutEnvClose:
    def test_batch_rollout_env_close_calls_underlying_env_close_once(self) -> None:
        vec_env = BatchRolloutEnv(
            env_factory=_SyncStubEnv,
            batch_size=2,
            group_size=2,
        )
        _ = vec_env.reset(seed=0)
        vec_env.close()
        close_counts = [traj.env.close_calls for traj in vec_env.trajectories]
        assert close_counts == [1, 1, 1, 1]

    def test_batch_rollout_env_close_dedupes_same_env_instance(self) -> None:
        shared = _SyncStubEnv()
        vec = BatchRolloutEnv(env_factory=lambda: shared, batch_size=2, group_size=2)
        _ = vec.reset(seed=0)
        vec.close()
        assert shared.close_calls == 1


class TestBatchRolloutEnvInit:
    def test_batch_rollout_env_constructor_rejects_non_positive_sizes(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            _ = BatchRolloutEnv(
                env_factory=_SyncStubEnv,
                batch_size=0,
                group_size=1,
            )
        with pytest.raises(ValueError, match="group_size must be > 0"):
            _ = BatchRolloutEnv(
                env_factory=_SyncStubEnv,
                batch_size=1,
                group_size=0,
            )


class TestRolloutBufferResetTrajectory:
    def test_trajectory_buffer_reset_trajectory_out_of_bounds(self) -> None:
        buf = RolloutBuffer(batch_size=1, group_size=1)
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
        buf = RolloutBuffer(batch_size=1, group_size=1)
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
    token-id lists — ``{"input_ids": [[...]]}`` — as some tokenizers do."""

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


class TestRolloutHarnessTokenizeInitialPrompt:
    def test_initial_prompt_unwraps_batched_token_id_lists(self) -> None:
        """Tokenizers returning ``[[ids]]`` (batch dim) and ``[ids]`` (flat)
        from ``apply_chat_template`` must produce identical ``(1, T)``
        tensors."""
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


class _NonTerminalEnv:
    def __init__(self) -> None:
        self.calls = 0
        self.last_gen: str | None = None

    def reset(self, seed: int | None = None, *, row_index: int | None = None):
        del seed, row_index
        return "hello", {"prefix": "P:", "suffix": "S"}

    def step(self, gen_text: str):
        self.calls += 1
        self.last_gen = gen_text
        if self.calls == 1:
            return "feedback", 0.5, False, False, {"prefix": "F:", "suffix": "T"}
        return "done", 1.0, True, False, {}


class TestRolloutHarnessFormatObs:
    def test_format_obs_prefix_suffix_and_empty_info(self) -> None:
        assert RolloutHarness._format_obs("x", None) == "x"
        assert (
            RolloutHarness._format_obs("x", {"prefix": "A:", "suffix": "B"}) == "A:x\nB"
        )


class TestRolloutHarnessGetEpisodeData:
    def test_get_episode_data_padding_and_pad_mask(self) -> None:
        w = _bare_wrapper()
        w.pad_id = 0
        w.max_turns = 3
        w.full_ids = torch.tensor([[9, 5, 0, 7, 8]], dtype=torch.long)
        w.turn_boundaries = [(1, 3, 0), (3, 5, 1)]
        w.turn_rewards = [1.5]
        full_ids, action_mask, turn_ids, rewards = w.get_episode_data()
        assert torch.equal(full_ids, w.full_ids)
        assert action_mask.dtype == torch.bool
        assert turn_ids.dtype == torch.long
        assert rewards.tolist() == [1.5, 0.0, 0.0]
        assert action_mask[0, 1].item() is False
        assert turn_ids[0, 1].item() == -1

    def test_get_episode_data_raises_without_reset(self) -> None:
        w = _bare_wrapper()
        w.full_ids = None
        with pytest.raises(RuntimeError, match="No episode data"):
            w.get_episode_data()


class TestRolloutHarnessGetDebugInfo:
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


class TestRolloutBufferInvariantsAndHelpers:
    def test_trajectory_buffer_invariants_and_helpers(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            _ = RolloutBuffer(batch_size=0, group_size=1)
        with pytest.raises(ValueError, match="group_size must be > 0"):
            _ = RolloutBuffer(batch_size=1, group_size=0)

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
        buf = RolloutBuffer(batch_size=1, group_size=2)
        buf.add_trajectory(t1)
        buf.add_trajectory(t2)
        assert buf.is_initialized is True
        assert buf.has_active() is True
        assert len(buf) == 2
        assert list(iter(buf))[0] is t1
        assert buf[1] is t2
        buf.sort(key=lambda t: (t.batch_idx, t.group_idx))
        assert [t.batch_idx for t in buf] == [0, 1]
        buf.clear()
        assert len(buf) == 0 and buf.has_active() is False


class TestRolloutBufferGetActiveTrajectories:
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
        buf = RolloutBuffer(batch_size=2, group_size=2)
        buf.add_trajectory(t0)
        buf.add_trajectory(t1)
        buf.add_trajectory(t2)

        unsorted_active = buf.get_active_trajectories(sorted_by_index=False)
        assert unsorted_active == [t0, t1]
        sorted_active = buf.get_active_trajectories(sorted_by_index=True)
        assert sorted_active == [t1, t0]


class TestRolloutBufferGetPrompts:
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
        buf = RolloutBuffer(batch_size=1, group_size=2)
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
            },
            done=False,
        )
        buf = RolloutBuffer(batch_size=1, group_size=2)
        buf.add_trajectory(a)
        buf.add_trajectory(b)
        prompts = buf.get_prompts()
        assert prompts is not None
        assert len(prompts) == 2
        assert [int(p["input_ids"].shape[1]) for p in prompts] == [2, 3]
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

        vec = BatchRolloutEnv(env_factory=_factory, batch_size=1, group_size=2)
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
