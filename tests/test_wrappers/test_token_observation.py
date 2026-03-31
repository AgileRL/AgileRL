"""Tests for TokenObservationWrapper sliding-window prompt fields."""

from __future__ import annotations

import torch

from agilerl.wrappers.token_observation import TokenObservationWrapper


class _StubTokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(ids)


def _bare_wrapper() -> TokenObservationWrapper:
    return TokenObservationWrapper.__new__(TokenObservationWrapper)


def test_build_model_prompt_fields_no_truncation() -> None:
    w = _bare_wrapper()
    w.tokenizer = _StubTokenizer()
    w._initial_prompt_len = 3
    w.turn_boundaries = []
    w.full_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    out = w.build_model_prompt_fields(max_prompt_tokens=100)
    assert torch.equal(out["model_input_ids"], w.full_ids)
    assert out["stitch_prefix_ids"].shape[1] == 0
    assert out["model_window_initial_len"] == 3


def test_sliding_window_reconstructs_full_sequence() -> None:
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
    trunc = out["model_input_ids"]
    stitch = out["stitch_prefix_ids"]
    il = out["model_window_initial_len"]
    assert il == 2
    merged = torch.cat([trunc[:, :il], stitch, trunc[:, il:]], dim=1)
    assert torch.equal(merged, w.full_ids)


def test_max_prompt_tokens_for_sliding_window() -> None:
    from agilerl.utils.llm_utils import max_prompt_tokens_for_sliding_window

    assert max_prompt_tokens_for_sliding_window(8192, 1024) == 8192 - 1024
    assert max_prompt_tokens_for_sliding_window(100, 50) == 50


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

    def encode(self, s: str) -> list[int]:
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


def test_reset_returns_tuple_with_text_and_sets_prompt_len() -> None:
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


def test_step_from_full_completion_slices_generation() -> None:
    inner = _RecordingGemEnv()
    w = TokenObservationWrapper(
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


def test_policy_observation_merges_sliding_window_when_max_model_len_set() -> None:
    w = _bare_wrapper()
    w.tokenizer = _StubTokenizer()
    w._sw_max_model_len = 100
    w._sw_max_output_tokens = 10
    w._initial_prompt_len = 2
    w.full_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    w.turn_boundaries = []
    obs = w._policy_observation_from_state()
    assert "model_input_ids" in obs
    assert "model_text" in obs
    assert obs["input_ids"].shape[1] == 4
