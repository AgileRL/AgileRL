# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tool-aware tokenization & masking in ``RolloutHarness``.

Masking is by *generation provenance*: a token contributes to the policy loss
iff the policy sampled it (``turn_boundaries``), so env-observation / tool-result
tokens appended via the feedback path are already excluded. The genuinely-new
piece is ``tools=`` schema injection into the chat template.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from agilerl.llm_envs import RolloutHarness
from tests.helpers.rollout_doubles import (
    MiniTokenizer,
    TinyDatasetEnv,
    bare_rollout_env,
)

_WIP = "pending tool-path wiring (engine / _align_sampling_logprobs)"


def _reasoning_env() -> TinyDatasetEnv:
    """A tiny dataset-backed reasoning env with a held-out split."""
    return TinyDatasetEnv(
        questions=["train-q"],
        answers=["train-a"],
        reward_fn=lambda c, a, q: 0.0,
        prompt_builder=lambda q: q,
        test_questions=["eval-q"],
        test_answers=["eval-a"],
    )


def _wrap(inner: object) -> RolloutHarness:
    """Drive ``inner`` at the token level in-process (no HTTP).

    The returned env owns the local client, so callers must ``close()`` it.
    """
    return RolloutHarness.local(
        inner,
        MiniTokenizer(),
        max_turns=1,
        apply_chat_template=False,
    )


def _mask_wrapper() -> RolloutHarness:
    """Wrapper carrying just the fields ``get_episode_data`` reads.

    ``full_ids`` layout: ``[p0 p1 | g0 g1 | f0 f1 | g2 g3]`` — initial prompt,
    generated turn 0, appended feedback (stands in for a tool result), generated
    turn 1.
    """
    w = bare_rollout_env()
    w.full_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    w.turn_boundaries = [(2, 4, 0), (6, 8, 1)]
    w.turn_rewards = [0.5, 1.0]
    w.max_turns = 2
    w.sampling_logps = []
    return w


def test_action_mask_excludes_appended_feedback() -> None:
    """Tool-result / feedback tokens (appended after each generated span) are
    masked 0 — locks the provenance guarantee that tool results never train.
    """
    _full, action_mask, _turn_ids, _rewards, _logps = _mask_wrapper().get_episode_data()
    # Mask is over positions [1 .. seq_len-1]; True only on generated spans.
    assert action_mask[0].tolist() == [False, True, True, False, False, True, True]


def test_turn_ids_track_generation_spans_only() -> None:
    """turn_ids hold the turn index on each generated span and -1 elsewhere
    (prompt, feedback / tool-result, pad).
    """
    _full, _action_mask, turn_ids, _rewards, _logps = _mask_wrapper().get_episode_data()
    assert turn_ids[0].tolist() == [-1, 0, 0, -1, -1, 1, 1]


class _RecordingTokenizer:
    """Captures the ``tools=`` argument passed to ``apply_chat_template``."""

    def __init__(self) -> None:
        self.last_tools: Any = "UNSET"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        tools: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.last_tools = tools
        return {"input_ids": [[1, 2, 3]]}


_TOOLS = [
    {
        "type": "function",
        "function": {"name": "calc", "description": "add two ints", "parameters": {}},
    }
]


def test_tool_schema_injected_into_prompt() -> None:
    """When ``tools`` is set, they are forwarded to ``apply_chat_template`` so the
    template renders the schemas into the initial prompt.
    """
    w = bare_rollout_env()
    w.apply_chat_template = True
    w.tools = _TOOLS
    w.tokenizer = _RecordingTokenizer()
    w._tokenize_initial_prompt("hi")
    assert w.tokenizer.last_tools == _TOOLS


def test_tools_none_is_backward_compatible() -> None:
    """With ``tools=None`` (default) no ``tools=`` kwarg is forwarded, preserving
    the exact pre-tool behaviour.
    """
    w = bare_rollout_env()
    w.apply_chat_template = True
    w.tools = None
    w.tokenizer = _RecordingTokenizer()
    w._tokenize_initial_prompt("hi")
    assert w.tokenizer.last_tools is None  # template default; tools= not passed


def test_tool_schema_injected_into_feedback_boundary() -> None:
    """``tools`` are forwarded to ``apply_chat_template`` on the multi-turn
    feedback-boundary render (``_chat_template_boundary_ids``), not only the
    initial prompt.
    """
    w = bare_rollout_env()
    w.tools = _TOOLS
    w.tokenizer = _RecordingTokenizer()
    w._chat_template_boundary_ids("tool result")
    assert w.tokenizer.last_tools == _TOOLS


def test_dataset_size_reflects_wrapped_env() -> None:
    """``dataset_size`` reports the wrapped env's training-row count."""
    w = _wrap(_reasoning_env())
    try:
        assert w.dataset_size == 1
    finally:
        w.close()


def test_eval_mode_routes_to_the_eval_split() -> None:
    """``eval_mode()`` routes resets to the held-out split, restoring after."""
    w = _wrap(_reasoning_env())
    try:
        w.reset(row_index=0)
        assert w._prompt_text == "train-q"
        with w.eval_mode():
            w.reset(row_index=0)
            assert w._prompt_text == "eval-q"
        w.reset(row_index=0)
        assert w._prompt_text == "train-q"
    finally:
        w.close()


def test_reset_forwards_row_index_to_wrapped_env() -> None:
    """``reset`` passes ``row_index`` through to the wrapped env, selecting that row."""
    inner = TinyDatasetEnv(
        questions=["q0", "q1"],
        answers=["a0", "a1"],
        reward_fn=lambda c, a, q: 0.0,
        prompt_builder=lambda q: f"P:{q}",
    )
    w = _wrap(inner)
    try:
        w.reset(row_index=1)
        assert inner._question == "q1"
        assert inner._answer == "a1"
    finally:
        w.close()


def test_dataset_size_falls_back_when_env_lacks_it() -> None:
    """An env exposing no ``dataset_size`` degrades to ``0``; eval_mode stays usable."""

    class _NoDataset:
        def reset(self, seed=None):
            return "hi", {}

        def step(self, action):
            return "", 0.0, True, False, {}

    w = _wrap(_NoDataset())
    try:
        assert w.dataset_size == 0
        # eval_mode stays usable even when the env exposes no held-out split.
        with w.eval_mode():
            w.reset()
        w.reset()
    finally:
        w.close()


@pytest.mark.skip(reason=_WIP)
def test_sampling_logps_align_across_tool_turns() -> None:
    """Per-row sampling_logps line up 1:1 with action_mask==1 positions across a
    2-tool-call episode (incl. pad==eos tokenizers); guards the silent row-skip in
    ``_align_sampling_logprobs``.
    """
    ...


@pytest.mark.skip(reason=_WIP)
def test_prompt_prefix_stable_across_tool_turn() -> None:
    """Appending a tool-result turn keeps the prior prompt ids as a literal prefix."""
    ...


@pytest.mark.skip(reason=_WIP)
def test_multiple_tool_calls_in_one_turn() -> None:
    """Parallel tool calls in one turn -> one contiguous trained span; the single
    appended result span is masked.
    """
    ...
