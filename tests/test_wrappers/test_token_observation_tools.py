"""Tool-aware tokenization & masking in ``TokenObservationWrapper``.

Masking is by *generation provenance*: a token contributes to the policy loss
iff the policy sampled it (``turn_boundaries``), so env-observation / tool-result
tokens appended via the feedback path are already excluded. The genuinely-new
piece is ``tools=`` schema injection into the chat template.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from agilerl.llm_envs import TokenObservationWrapper

_WIP = "pending tool-path wiring (engine / _align_sampling_logprobs)"


def _bare_wrapper() -> TokenObservationWrapper:
    return TokenObservationWrapper.__new__(TokenObservationWrapper)


def _mask_wrapper() -> TokenObservationWrapper:
    """Wrapper carrying just the fields ``get_episode_data`` reads.

    ``full_ids`` layout: ``[p0 p1 | g0 g1 | f0 f1 | g2 g3]`` — initial prompt,
    generated turn 0, appended feedback (stands in for a tool result), generated
    turn 1.
    """
    w = _bare_wrapper()
    w.full_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    w.turn_boundaries = [(2, 4, 0), (6, 8, 1)]
    w.turn_rewards = [0.5, 1.0]
    w.pad_id = None
    w.max_turns = 2
    return w


def test_action_mask_excludes_appended_feedback() -> None:
    """Tool-result / feedback tokens (appended after each generated span) are
    masked 0 — locks the provenance guarantee that tool results never train."""
    _full, action_mask, _turn_ids, _rewards = _mask_wrapper().get_episode_data()
    # Mask is over positions [1 .. seq_len-1]; True only on generated spans.
    assert action_mask[0].tolist() == [False, True, True, False, False, True, True]


def test_turn_ids_track_generation_spans_only() -> None:
    """turn_ids hold the turn index on each generated span and -1 elsewhere
    (prompt, feedback / tool-result, pad)."""
    _full, _action_mask, turn_ids, _rewards = _mask_wrapper().get_episode_data()
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
    template renders the schemas into the initial prompt."""
    w = _bare_wrapper()
    w.apply_chat_template = True
    w.tools = _TOOLS
    w.tokenizer = _RecordingTokenizer()
    w._tokenize_initial_prompt("hi")
    assert w.tokenizer.last_tools == _TOOLS


def test_tools_none_is_backward_compatible() -> None:
    """With ``tools=None`` (default) no ``tools=`` kwarg is forwarded, preserving
    the exact pre-tool behaviour."""
    w = _bare_wrapper()
    w.apply_chat_template = True
    w.tools = None
    w.tokenizer = _RecordingTokenizer()
    w._tokenize_initial_prompt("hi")
    assert w.tokenizer.last_tools is None  # template default; tools= not passed


def test_tool_schema_injected_into_feedback_boundary() -> None:
    """``tools`` are forwarded to ``apply_chat_template`` on the multi-turn
    feedback-boundary render (``_chat_template_boundary_ids``), not only the
    initial prompt."""
    w = _bare_wrapper()
    w.tools = _TOOLS
    w.tokenizer = _RecordingTokenizer()
    w._chat_template_boundary_ids("tool result")
    assert w.tokenizer.last_tools == _TOOLS


@pytest.mark.skip(reason=_WIP)
def test_sampling_logps_align_across_tool_turns() -> None:
    """Per-row sampling_logps line up 1:1 with action_mask==1 positions across a
    2-tool-call episode (incl. pad_id==eos_id); guards the silent row-skip in
    ``_align_sampling_logprobs``."""
    ...


@pytest.mark.skip(reason=_WIP)
def test_prompt_prefix_stable_across_tool_turn() -> None:
    """Appending a tool-result turn keeps the prior prompt ids as a literal prefix
    (covered structurally by the build_model_prompt_fields stitch tests)."""
    ...


@pytest.mark.skip(reason=_WIP)
def test_multiple_tool_calls_in_one_turn() -> None:
    """Parallel tool calls in one turn -> one contiguous trained span; the single
    appended result span is masked."""
    ...
