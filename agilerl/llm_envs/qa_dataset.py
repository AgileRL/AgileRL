# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Dataset-backed single-turn text environment scored by an OpenEnv rubric."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.rubrics.base import Rubric

from agilerl.llm_envs.openenv_server import TextObservation, TextState
from agilerl.llm_envs.rubrics import _require_rubric, register_component_hooks


def _json_safe(value: object) -> object:
    """Coerce numpy / nested containers toward JSON-friendly Python types.

    Already-safe values pass through untouched. Unrecognised types pass through
    with a one-time warning (hosting requires pydantic-serializable labels).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()  # ty: ignore[no-matching-overload]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    warnings.warn(
        f"label value of type {type(value).__name__} is not coerced; "
        "hosting over /ws requires a pydantic-serializable label",
        UserWarning,
        stacklevel=2,
    )
    return value


class QADatasetEnv(Environment):
    """Serve (question, answer) rows as single-turn prompts, scored by a rubric.

    ``reset`` returns the prompt with labels unset. The terminal ``step``
    observation keeps that same ``prompt`` and adds ``question`` / ``answer``
    so rubrics can score (including when ``prompt_builder`` differs from the
    raw question); the policy never sees labels at reset.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        dataset: Sequence[Mapping[str, Any]],
        *,
        rubric: Rubric,
        test_dataset: Sequence[Mapping[str, Any]] | None = None,
        prompt_builder: Callable[[Mapping[str, Any]], str] | None = None,
        question_column: str = "question",
        answer_column: str = "answer",
    ) -> None:
        super().__init__(rubric=_require_rubric(rubric))
        self._components = register_component_hooks(self.rubric)
        self._train = dataset
        self._test = test_dataset
        self._prompt_builder = prompt_builder
        self._question_column = question_column
        self._answer_column = answer_column
        self._cursor = 0
        self._split = ""
        self._prompt: str = ""
        self._question: Any = None
        self._answer: Any = None
        self._episode_id: str | None = None
        self._step_count = 0

    @property
    def rubric_components(self) -> tuple[str, ...]:
        """Leaf rubric names reported in ``metadata["rubric_scores"]``."""
        return self._components

    @property
    def state(self) -> TextState:
        return TextState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            dataset_size=len(self._train),
            rubric_components=list(self._components),
        )

    def _next_index(
        self,
        row_index: int | None,
        rows: Sequence[Mapping[str, Any]],
        seed: int | None = None,
    ) -> int:
        if row_index is not None:
            idx = int(row_index)
            n = len(rows)
            if idx < 0 or idx >= n:
                msg = f"row_index {idx} is out of range for {n} rows"
                raise IndexError(msg)
            return idx
        n = max(len(rows), 1)
        if seed is not None:
            return int(seed) % n
        idx = self._cursor
        self._cursor = (self._cursor + 1) % n
        return idx

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        row_index: int | None = None,
        evaluation: bool | None = None,
        **kwargs: Any,
    ) -> TextObservation:
        rows = self._test if (evaluation and self._test is not None) else self._train
        if not rows:
            msg = "QADatasetEnv has no rows to serve"
            raise RuntimeError(msg)
        row = rows[self._next_index(row_index, rows, seed)]
        self._question = _json_safe(row[self._question_column])
        self._answer = _json_safe(row[self._answer_column])
        self._episode_id = episode_id
        self._step_count = 0
        self._split = "test" if (evaluation and self._test is not None) else "train"
        self._prompt = (
            self._prompt_builder(row)
            if self._prompt_builder is not None
            else str(self._question)
        )
        return TextObservation(prompt=self._prompt, done=False)

    def step(
        self,
        action: object,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> TextObservation:
        obs = TextObservation(
            prompt=self._prompt,
            done=True,
            question=self._question,
            answer=self._answer,
        )
        score = self._apply_rubric(action, obs)
        if inspect.iscoroutine(score):
            score.close()
            msg = "async rubrics require step_async; see OpenEnv docs"
            raise TypeError(msg)
        obs.reward = float(score)
        self._step_count += 1
        return obs
