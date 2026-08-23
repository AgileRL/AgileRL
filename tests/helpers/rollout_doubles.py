# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared mixin for ``RolloutHarness`` test doubles driven by ``RolloutCollector``.

``RolloutCollector`` drives its envs through the phased interface
(``_reset_fetch`` / ``_reset_apply`` and ``_step_prepare`` / ``_step_env`` /
``_step_apply``) so it can overlap the backend round-trips across envs. A double
only needs to implement plain ``reset()`` / ``step()``; this mixin maps the
phased interface onto them so the double drives the real (concurrent) collector
path without reimplementing the split.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any

import torch

from agilerl.llm_envs.harness import RolloutHarness


class FakeEnvClient:
    """Minimal in-memory ``EnvClientProtocol`` implementation for doubles.

    A text backend with a scripted episode length: ``step`` terminates after
    ``terminate_after`` steps (never, when ``None``). Call counters and the
    eval-mode depth let tests assert how the client was driven.
    """

    def __init__(
        self,
        *,
        reward: float = 1.0,
        terminate_after: int | None = None,
        dataset_size: int = 0,
        tools: list[Any] | None = None,
    ) -> None:
        self.reward = reward
        self.terminate_after = terminate_after
        self._dataset_size = dataset_size
        self._tools = list(tools or [])
        self._episode_steps = 0
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.eval_mode_entries = 0
        self.eval_mode_depth = 0

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        del seed, row_index
        self.reset_calls += 1
        self._episode_steps = 0
        return "prompt", {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        del action
        self.step_calls += 1
        self._episode_steps += 1
        terminated = (
            self.terminate_after is not None
            and self._episode_steps >= self.terminate_after
        )
        return "feedback", self.reward, terminated, False, {}

    def close(self) -> None:
        self.close_calls += 1

    @property
    def dataset_size(self) -> int:
        return self._dataset_size

    @property
    def tools(self) -> list[Any]:
        return self._tools

    @property
    def rubric_components(self) -> tuple[str, ...]:
        return ()

    @contextmanager
    def eval_mode(self):
        self.eval_mode_entries += 1
        self.eval_mode_depth += 1
        try:
            yield
        finally:
            self.eval_mode_depth -= 1


class RolloutEnvDoubleMixin:
    """Map ``RolloutCollector``'s phased reset/step onto a double's reset()/step()."""

    @property
    def rubric_components(self) -> tuple[str, ...]:
        return ()

    def _reset_fetch(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> Any:
        params = inspect.signature(self.reset).parameters
        kwargs = {"row_index": row_index} if "row_index" in params else {}
        self._pending_reset = self.reset(seed=seed, **kwargs)
        return self._pending_reset

    def _reset_apply(self, obs_text: Any, info: Any) -> Any:
        del obs_text, info
        return self._pending_reset

    def _step_prepare(self, full_completion: Any, sampling_logps: Any = None) -> str:
        # Mirror the real phase contract: completions are normalized to 2-D here.
        if full_completion.dim() == 1:
            full_completion = full_completion.unsqueeze(0)
        self._pending_step = (full_completion, sampling_logps)
        return ""

    def _step_env(self, gen_text: str) -> Any:
        del gen_text
        return None

    def _step_apply(self, env_result: Any) -> Any:
        del env_result
        full_completion, sampling_logps = self._pending_step
        return self.step(full_completion, sampling_logps=sampling_logps)


class MiniTokenizer:
    """Minimal tokenizer for the non-chat-template token path."""

    pad_token_id = 0

    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        del texts
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: Any, **_: Any) -> str:
        del ids
        return "go"

    def encode(self, text: str, **_: Any) -> list[int]:
        del text
        return [7, 7]


class TinyDatasetEnv:
    """A tiny dataset-backed env (questions/answers + reward_fn).

    Drives ``RolloutHarness``'s dataset_size / eval-split / row-index plumbing —
    in-process or hosted, or loaded by its ``module:Class`` entrypoint.
    ``reset`` pins a row (routing to the held-out split under ``evaluation``);
    ``step`` scores the completion and ends the turn.
    """

    def __init__(
        self,
        questions: list[str],
        answers: list[str],
        reward_fn: Any,
        *,
        prompt_builder: Any = None,
        test_questions: list[str] | None = None,
        test_answers: list[str] | None = None,
    ) -> None:
        self.questions = questions
        self.answers = answers
        self.reward_fn = reward_fn
        self.prompt_builder = prompt_builder or (lambda q: q)
        self.test_questions = test_questions
        self.test_answers = test_answers
        self._question = ""
        self._answer = ""

    @property
    def dataset_size(self) -> int:
        return len(self.questions)

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int = 0,
        evaluation: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del seed
        if evaluation and self.test_questions is not None:
            questions, answers = self.test_questions, self.test_answers
        else:
            questions, answers = self.questions, self.answers
        row = (row_index or 0) % len(questions)
        self._question, self._answer = questions[row], answers[row]
        return self.prompt_builder(self._question), {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        reward = float(self.reward_fn(action, self._answer, self._question))
        return "", reward, True, False, {}


def bare_rollout_env() -> RolloutHarness:
    """A ``RolloutHarness`` shell carrying only the fields the tokenize paths read."""
    w = RolloutHarness.__new__(RolloutHarness)
    w.tools = None  # optional config; __init__ default, read by the tokenize paths
    w.chat_template_kwargs = {}  # __init__ default, read by the tokenize paths
    w.sampling_logps = []  # read by get_episode_data
    # per-role boundary-frame cache; __init__ default, read by the feedback tokenize path
    w._boundary_parts = {}
    w._system_prompt = None  # __init__ default, read by the initial-prompt path
    w._special_ids_cache = None  # __init__ default, read by the feedback dedupe
    # These tests drive the ChatML fallback deliberately, so they opt out of strict.
    w._strict_chat_template_boundary = False
    return w
