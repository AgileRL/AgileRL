"""Phase-0 scaffold: tool-aware tokenization & masking in ``TokenObservationWrapper``.

TDD scaffolding for the P0-4 contract of tool-calling env support (driven from
agilerl-integration). Each test is skipped until the tool-aware path lands, then
the skip is removed.

Contract (sampled-token provenance): a token contributes to the policy loss iff
the policy sampled it. Prompt / replayed context / env-observation / tool-result
tokens are masked out; the assistant's text AND its emitted tool-call tokens are
trained. ``sampling_logps`` must stay aligned 1:1 with the ``action_mask == 1``
positions across tool turns. Full design + research live in agilerl-integration:
``docs/design/tool-calling-envs.md`` (Token-masking rule / P0-4).
"""

from __future__ import annotations

import pytest

from agilerl.llm_envs import TokenObservationWrapper  # noqa: F401

_WIP = "Phase-0 P0-4: tool-aware tokenization/masking not yet implemented"


@pytest.mark.skip(reason=_WIP)
def test_action_mask_excludes_appended_feedback() -> None:
    """Env-observation tokens appended after a turn's generated span are masked.

    Locks the existing ``turn_boundaries`` guarantee that the tokens appended via
    the feedback path (which tool RESULTS will reuse) are already excluded from
    the loss.
    """
    ...


@pytest.mark.skip(reason=_WIP)
def test_tool_result_tokens_excluded_from_mask() -> None:
    """A full tool turn (model emits a tool call -> tool result appended) masks
    only the result span, keeping the model's tool-call tokens trained."""
    ...


@pytest.mark.skip(reason=_WIP)
def test_tool_schema_injected_into_prompt() -> None:
    """``reset()`` renders the tool schemas into the chat-template prompt
    (``tools=``) so the policy can emit structured tool calls."""
    ...


@pytest.mark.skip(reason=_WIP)
def test_sampling_logps_align_across_tool_turns() -> None:
    """Per-row ``sampling_logps`` line up 1:1 with ``action_mask == 1`` positions
    across a 2-tool-call episode, so IS-correction stays enabled."""
    ...


@pytest.mark.skip(reason=_WIP)
def test_prompt_prefix_stable_across_tool_turn() -> None:
    """Appending a tool-result turn preserves the prior prompt token prefix
    (prefix-cache safe; the ``build_model_prompt_fields`` stitch invariant holds)."""
    ...


@pytest.mark.skip(reason=_WIP)
def test_multiple_tool_calls_in_one_turn() -> None:
    """Parallel tool calls in one assistant turn -> one contiguous trained span;
    all tool results masked."""
    ...


@pytest.mark.skip(reason=_WIP)
def test_turn_ids_correct_across_tool_turns() -> None:
    """``turn_ids`` increment per generated turn; tool results do not create
    spurious turns."""
    ...
