# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Expected row lengths a rollout run is priced at: measured, or generic defaults."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import (
    SUPPORTED_MODEL_INFO,
    resolve_reasoning_parser,
)

# Generic planning defaults for a single-turn run without expected lengths:
# tokens the turn generates with and without reasoning, and the prompt.
DEFAULT_REASONING_TURN_TOKENS = 4096
DEFAULT_TURN_TOKENS = 128
DEFAULT_PROMPT_TOKENS = 256
# Share of the row budget a multi-turn row fills: 0.50-0.54 on Sudoku-v0-hard
# (Qwen3-4B), 0.38-0.43 on AWM (Qwen3.6-35B-A3B).
MULTI_TURN_ROW_FILL = 0.46
# Share of a multi-turn row the model decodes; the rest is prompt and env text.
MULTI_TURN_DECODE_SHARE = 0.5
# Mean turns of an episode over the env's max_turns: 11-12 of 50 on
# Sudoku-v0-hard, 2-5 of 20 on AWM, rising as the policy trains.
MULTI_TURN_TURN_SHARE = 0.2
# A turn's mean decoded tokens: 380-430 on Sudoku-v0-hard, 220-450 on AWM.
MULTI_TURN_TURN_DECODE_TOKENS = 352


class ExpectedLengths(BaseModel):
    """Expected mean lengths of a run's rows, in place of the generic defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: float = Field(
        ge=0,
        description=(
            "Mean tokens of a row the model did not generate: the prompt, and "
            "for multi-turn runs every environment observation too."
        ),
    )
    generated_tokens_per_turn: float = Field(
        gt=0, description="Mean tokens the model generates in one turn."
    )
    turns: float = Field(ge=1, description="Mean turns of an episode.")
    generated_tokens_per_turn_low: float | None = Field(
        default=None,
        gt=0,
        description="Low end of the tokens generated in one turn, for a step range.",
    )
    generated_tokens_per_turn_high: float | None = Field(
        default=None,
        gt=0,
        description="High end of the tokens generated in one turn, for a step range.",
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        """Both range ends or neither, around ``generated_tokens_per_turn``."""
        low = self.generated_tokens_per_turn_low
        high = self.generated_tokens_per_turn_high
        if low is None and high is None:
            return self
        if low is None or high is None:
            msg = "set both generated_tokens_per_turn_low and _high, or neither"
            raise ValueError(msg)
        if not low <= self.generated_tokens_per_turn <= high:
            msg = (
                "generated_tokens_per_turn_low <= generated_tokens_per_turn "
                "<= generated_tokens_per_turn_high does not hold"
            )
            raise ValueError(msg)
        return self


def _turn_cap(spec: RolloutLLMSpec) -> int:
    """A turn's output-token cap: ``max_output_tokens``, else the row budget."""
    row_budget = spec.max_row_tokens or spec.max_model_len
    return min(spec.max_output_tokens or row_budget, row_budget)


def default_row_lengths(
    manifest: TrainingManifest, spec: RolloutLLMSpec
) -> tuple[int, int, int]:
    """Mean trained-row tokens, decoded tokens per row and turns from the budgets.

    A single turn decodes :data:`DEFAULT_REASONING_TURN_TOKENS` when the model
    reasons, else :data:`DEFAULT_TURN_TOKENS`, at most its cap
    (:func:`_turn_cap`), after a :data:`DEFAULT_PROMPT_TOKENS` prompt within
    what the decode leaves of the row. A multi-turn row fills
    :data:`MULTI_TURN_ROW_FILL` of its budget over :data:`MULTI_TURN_TURN_SHARE`
    of ``max_turns``, or as many full turns as the row holds when turns are
    unbounded.
    """
    row_budget = spec.max_row_tokens or spec.max_model_len
    turn_tokens = _turn_cap(spec)
    env = manifest.environment
    max_turns = env.max_turns if isinstance(env, LLMEnvSpec) else 1
    if max_turns == 1:
        generated = (
            DEFAULT_REASONING_TURN_TOKENS
            if _generates_reasoning(manifest, spec)
            else DEFAULT_TURN_TOKENS
        )
        decode_tokens = min(generated, turn_tokens)
        prompt_tokens = min(DEFAULT_PROMPT_TOKENS, row_budget - decode_tokens)
        return decode_tokens + prompt_tokens, decode_tokens, 1
    row_tokens = math.ceil(MULTI_TURN_ROW_FILL * row_budget)
    turns = max(
        2,
        math.ceil(row_tokens / turn_tokens)
        if max_turns is None
        else math.ceil(MULTI_TURN_TURN_SHARE * max_turns),
    )
    decode_tokens = min(
        math.ceil(MULTI_TURN_DECODE_SHARE * row_tokens),
        turns * MULTI_TURN_TURN_DECODE_TOKENS,
    )
    return row_tokens, decode_tokens, turns


def measured_row_lengths(
    manifest: TrainingManifest,
    spec: RolloutLLMSpec,
    lengths: ExpectedLengths,
) -> tuple[int, int, int]:
    """Mean trained-row tokens, decoded tokens per row and turns from expected lengths.

    A turn decodes ``generated_tokens_per_turn`` tokens, at most its cap
    (:func:`_turn_cap`). Turns are at most the env's ``max_turns``, one for a
    single-turn env; the decode is priced at those turns and the reported
    turns are rounded to a whole number of at least one, halves to even as
    :func:`round` does. The row is the prompt plus every turn's decode, within
    the row budget.
    """
    row_budget = spec.max_row_tokens or spec.max_model_len
    turn_decode = min(lengths.generated_tokens_per_turn, _turn_cap(spec))
    env = manifest.environment
    max_turns = env.max_turns if isinstance(env, LLMEnvSpec) else 1
    mean_turns = lengths.turns if max_turns is None else min(lengths.turns, max_turns)
    turns = max(1, round(mean_turns))
    decode_tokens = min(math.ceil(turn_decode * mean_turns), row_budget)
    row_tokens = min(math.ceil(lengths.prompt_tokens) + decode_tokens, row_budget)
    return row_tokens, decode_tokens, turns


def _generates_reasoning(manifest: TrainingManifest, spec: RolloutLLMSpec) -> bool:
    """Whether rollouts reason: the model has a reasoning parser and thinking is not turned off.

    :param manifest: Validated rollout manifest.
    :param spec: ``manifest``'s rollout algorithm.
    :return: ``True`` when the engine parses reasoning and the env's chat
        template does not set ``enable_thinking`` to ``False``.
    """
    info = SUPPORTED_MODEL_INFO.get(spec.pretrained_model_name_or_path or "")
    model_type = None if info is None else info.model_type
    env = manifest.environment
    template = env.chat_template_kwargs if isinstance(env, LLMEnvSpec) else {}
    return (
        resolve_reasoning_parser(spec.vllm_engine_args, model_type) is not None
        and template.get("enable_thinking") is not False
    )


def expected_lengths_for_run(
    manifest: TrainingManifest,
    on: ExpectedLengths | None,
    off: ExpectedLengths | None,
) -> ExpectedLengths | None:
    """The lengths a single-turn rollout run prices at, picked by whether it reasons.

    A multi-turn run keeps the generic defaults, so the lengths' ``turns`` is
    never read for one; a single-turn run caps it at one turn.

    :param manifest: Validated run manifest.
    :param on: Lengths measured with thinking on.
    :param off: Lengths measured with thinking off.
    :return: ``on`` when the run generates reasoning (:func:`_generates_reasoning`),
        else ``off``; ``None`` for a multi-turn run or a run without rollouts.
    """
    spec = manifest.algorithm
    env = manifest.environment
    max_turns = env.max_turns if isinstance(env, LLMEnvSpec) else 1
    if max_turns != 1 or not isinstance(spec, RolloutLLMSpec):
        return None
    return on if _generates_reasoning(manifest, spec) else off
