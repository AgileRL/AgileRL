"""Generation rollout envs, with reasoning as the one-turn case.

A :class:`RolloutEnv` is the generation half of the env taxonomy: the model
generates a completion to a dataset-seeded prompt and the env scores it with a
``reward_fn``. Reasoning is the degenerate ``max_turns=1`` configuration — a
plain :class:`RolloutEnv` instance, no subclass. Callers wrap it in
:class:`~agilerl.llm_envs.token_observation.TokenObservationWrapper` so it plugs
into ``BatchRolloutEnv`` like any other rollout env, deriving a prompt builder
from a conversation template via :func:`_default_prompt_builder` and pulling
question/answer columns via :func:`_extract_question_answer_columns`.

Dataset order is deterministic: a ``BatchRolloutEnv`` owns the shared dataset
cursor across its trajectories, re-drawing a seeded per-epoch shuffle
(:func:`dataloader_shuffle_order`) at each epoch boundary so a per-row seed
selects one reproducible dataset row for every trajectory in its group. Batch/row
order need only be deterministic and group-consistent, which is what
grouped-advantage training relies on. A standalone (eval) env owns no batch
cursor and walks its active split sequentially.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from agilerl.llm_envs.base import LLMEnv


def dataloader_shuffle_order(
    dataset_size: int,
    seed: int,
) -> list[int]:
    """Deterministic shuffle of dataset row indices for one epoch.

    A single ``torch.randperm`` drawn from a seeded generator. The result is
    reproducible for a given ``seed`` and is a full permutation covering every row
    exactly once. The exact permutation need not match any particular
    ``DataLoader``: grouped-advantage training only needs the row order to be
    deterministic and group-consistent, not to mirror a specific batch order.
    ``BatchRolloutEnv`` calls this once per epoch (varying the seed per epoch).

    :param dataset_size: Number of rows in the dataset.
    :type dataset_size: int
    :param seed: Generator seed (matches the env seed).
    :type seed: int
    :return: Permutation of dataset row indices, ``dataset_size`` long.
    :rtype: list[int]
    """
    if dataset_size <= 0:
        msg = f"dataset_size must be > 0, got {dataset_size}."
        raise ValueError(msg)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator).tolist()


class RolloutEnv(LLMEnv):
    """Generation rollout env: dataset-seeded prompt in, scored text out.

    Text observation / text action. With ``max_turns=1`` (the default) this is the
    reasoning env: the model produces one completion to a dataset-seeded prompt and
    the env scores it via ``reward_fn(completion, answer, question)`` on the decoded
    generation. Multi-turn / tool-using rollouts subclass this and override
    :meth:`step`. Wrapped by
    :class:`~agilerl.llm_envs.token_observation.TokenObservationWrapper` to
    participate in the rollout taxonomy.

    :param max_turns: Number of generation turns before the episode terminates.
    :type max_turns: int
    :param tools: Optional tool schemas available to the policy.
    :type tools: list | None
    :param questions: Per-row question strings.
    :type questions: list[str] | None
    :param answers: Per-row training answer strings.
    :type answers: list[str] | None
    :param reward_fn: ``(completion, answer, question) -> float`` scorer.
    :type reward_fn: Callable[[str, str, str], float] | None
    :param prompt_builder: Maps a question to the prompt text shown to the model.
    :type prompt_builder: Callable[[str], str] | None
    :param test_questions: Held-out question strings used under ``eval_mode``.
    :type test_questions: list[str] | None
    :param test_answers: Held-out answer strings used under ``eval_mode``.
    :type test_answers: list[str] | None
    """

    def __init__(
        self,
        *,
        max_turns: int = 1,
        tools: list | None = None,
        questions: list[str] | None = None,
        answers: list[str] | None = None,
        reward_fn: Callable[[str, str, str], float] | None = None,
        prompt_builder: Callable[[str], str] | None = None,
        test_questions: list[str] | None = None,
        test_answers: list[str] | None = None,
    ) -> None:
        self.max_turns = max_turns
        self.tools = list(tools) if tools else []
        self.questions = questions
        self.answers = answers
        self.reward_fn = reward_fn
        self.prompt_builder = prompt_builder
        self.test_questions = test_questions
        self.test_answers = test_answers
        self.evaluation_mode = False
        self._turn = 0
        self._question: str = ""
        self._answer: str = ""
        # Dataset cursor owned by ``BatchRolloutEnv`` when batched; on the
        # standalone / eval path the env walks its active split sequentially.
        self._cursor = 0
        self._cursor_split = ""

    @property
    def dataset_size(self) -> int:
        """Number of training rows backing this env (0 when not dataset-backed)."""
        return len(self.questions) if self.questions else 0

    def _active_rows(self) -> tuple[list[str], list[str]]:
        """Return the (questions, answers) for the current train/eval split."""
        if self.evaluation_mode and self.test_questions is not None:
            return self.test_questions, self.test_answers
        return self.questions, self.answers

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Select a dataset row and return its prompt text plus info.

        :param seed: Optional reset seed (unused on the standalone path, which
            walks its split sequentially).
        :type seed: int | None
        :param row_index: Dataset row chosen by the owning ``BatchRolloutEnv``;
            when ``None`` the env resolves a row from its own per-split cursor.
        :type row_index: int | None
        """
        self._turn = 0
        questions, answers = self._active_rows()
        if row_index is None:
            split = (
                "eval"
                if self.evaluation_mode and self.test_questions is not None
                else "train"
            )
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


def _extract_question_answer_columns(
    dataset: Any,
) -> tuple[list[str], list[str]]:
    """Pull ``question`` / ``answer`` columns from an HF or torch-style dataset.

    HuggingFace datasets support column access (``dataset["question"]``); plain
    ``torch.utils.data.Dataset`` instances are indexed per row and return a dict.
    """
    try:
        return list(dataset["question"]), list(dataset["answer"])
    except (KeyError, TypeError):
        questions = [dataset[i]["question"] for i in range(len(dataset))]
        answers = [dataset[i]["answer"] for i in range(len(dataset))]
        return questions, answers


def _default_prompt_builder(
    conversation_template: list[dict[str, str]],
) -> Callable[[str], str]:
    """Build a question->prompt-text function from a conversation template.

    Formats each template message's ``content`` with the question (answer left
    blank, mirroring generation time) and joins them. The wrapped
    ``TokenObservationWrapper`` applies the tokenizer's chat template to this
    text, so the builder only assembles the user-visible prompt string.
    """

    def build(question: str) -> str:
        rendered = [
            msg["content"].format(question=question, answer="")
            for msg in conversation_template
        ]
        return "\n".join(part for part in rendered if part)

    return build
