"""Reasoning env: a dataset of prompts scored by a reward function.

The canonical RL-for-LLM task — a row of the dataset seeds the prompt, the model
produces a completion, and ``reward_fn(completion, answer, question)`` scores it.
With ``max_turns=1`` (the default) this is single-turn reasoning; larger values
re-score the same row over several turns.

It is a plain local env (the Gym / gem text contract: ``reset -> (prompt, info)``,
``step -> (obs, reward, terminated, truncated, info)``) so it is driven exactly like
any other env: wrap it in an :class:`~agilerl.llm_envs.openenv.OpenEnvServer` (or the
socket-free :func:`~agilerl.llm_envs.openenv.local_transport`) and point a
``RolloutEnv`` at it. :meth:`RolloutEnv.from_dataset` does that wiring for you.

``reset`` accepts a ``row_index`` (so the owning ``BatchRolloutEnv`` can pin a whole
group to one prompt — the variance-reduction GRPO relies on) and an ``evaluation``
flag (to serve the held-out split); both arrive over the OpenEnv API.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class ReasoningEnv:
    """Dataset-seeded prompt in, reward-scored text out.

    :param questions: Per-row question strings (the training split).
    :type questions: list[str]
    :param answers: Per-row training answer strings, aligned with ``questions``.
    :type answers: list[str]
    :param reward_fn: ``(completion, answer, question) -> float`` scorer.
    :type reward_fn: Callable[[str, str, str], float] | None
    :param prompt_builder: Maps a question to the prompt text shown to the model,
        defaults to the identity.
    :type prompt_builder: Callable[[str], str] | None
    :param test_questions: Held-out question strings used under ``evaluation``.
    :type test_questions: list[str] | None
    :param test_answers: Held-out answer strings used under ``evaluation``.
    :type test_answers: list[str] | None
    :param max_turns: Number of generation turns before the episode terminates.
    :type max_turns: int
    :param tools: Optional tool schemas advertised to the policy.
    :type tools: list | None
    """

    def __init__(
        self,
        questions: list[str] | None = None,
        answers: list[str] | None = None,
        reward_fn: Callable[[str, str, str], float] | None = None,
        *,
        prompt_builder: Callable[[str], str] | None = None,
        test_questions: list[str] | None = None,
        test_answers: list[str] | None = None,
        max_turns: int = 1,
        tools: list[Any] | None = None,
    ) -> None:
        """Build a reasoning env over ``(questions, answers)`` scored by ``reward_fn``."""
        self.questions = questions
        self.answers = answers
        self.reward_fn = reward_fn if reward_fn is not None else (lambda *_: 0.0)
        self.prompt_builder = (
            prompt_builder if prompt_builder is not None else (lambda q: q)
        )
        self.test_questions = test_questions
        self.test_answers = test_answers
        self.max_turns = max_turns
        self.tools = list(tools) if tools else []
        self.evaluation_mode = False
        self._turn = 0
        self._question: str = ""
        self._answer: str = ""
        # Cursor for the standalone path (the batch path always passes a row_index):
        self._cursor = 0
        self._cursor_split = ""

    @property
    def dataset_size(self) -> int:
        """Number of training rows backing this env (``0`` when not dataset-backed)."""
        return len(self.questions) if self.questions else 0

    def _active_rows(self, evaluation: bool) -> tuple[list[str], list[str]]:
        """Return the ``(questions, answers)`` for the requested split."""
        if evaluation and self.test_questions is not None:
            return self.test_questions, self.test_answers
        return self.questions, self.answers

    @contextmanager
    def eval_mode(self) -> Iterator[None]:
        """Serve the held-out split for the duration of the block (standalone use).

        Over the OpenEnv interface a client sends ``evaluation`` per reset instead;
        this is the convenience for driving the env directly.
        """
        previous = self.evaluation_mode
        self.evaluation_mode = True
        try:
            yield
        finally:
            self.evaluation_mode = previous

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
        evaluation: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Select a dataset row and return its prompt text plus info.

        :param seed: Unused (the row, not a seed, determines the prompt).
        :type seed: int | None
        :param row_index: Row to serve; when ``None`` the env walks its active
            split sequentially from a per-split cursor (the standalone path).
        :type row_index: int | None
        :param evaluation: Serve the held-out split; ``None`` falls back to
            :attr:`evaluation_mode`.
        :type evaluation: bool | None
        """
        del seed
        self._turn = 0
        use_eval = self.evaluation_mode if evaluation is None else bool(evaluation)
        questions, answers = self._active_rows(use_eval)
        if row_index is None:
            split = "eval" if use_eval and self.test_questions is not None else "train"
            if split != self._cursor_split:
                self._cursor = 0
                self._cursor_split = split
            row_index = self._cursor
            self._cursor += 1
        row = row_index % len(questions)
        self._question = questions[row]
        self._answer = answers[row]
        return self.prompt_builder(self._question), {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Score the completion against the current row; terminate at ``max_turns``."""
        self._turn += 1
        reward = float(self.reward_fn(action, self._answer, self._question))
        terminated = self._turn >= self.max_turns
        return "", reward, terminated, False, {}
