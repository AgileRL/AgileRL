"""Reasoning as a single-turn ``RolloutEnv``.

Reasoning is the degenerate one-turn case of the rollout taxonomy: the model
generates one completion to a dataset-seeded prompt and the env scores it with
a ``reward_fn``. This module provides the raw single-turn env
(:class:`SingleTurnReasoningEnv`, text obs / text action) and a factory
(:func:`make_reasoning_rollout_env`) that wraps it in
:class:`~agilerl.llm_envs.token_observation.TokenObservationWrapper` with
``max_turns=1`` so it plugs into ``BatchRolloutEnv`` like any other rollout env.

Dataset order is deterministic: the seeded shuffle is precomputed as an explicit
index list (:func:`dataloader_shuffle_order`) and shared (with a cursor) across
the trajectories of a ``BatchRolloutEnv`` so a per-row seed selects one
reproducible dataset row for every trajectory in its group. Batch/row order need
not match the old reasoning dataloader — only be deterministic and
group-consistent, which is what grouped-advantage training relies on.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from agilerl.llm_envs.token_observation import TokenObservationWrapper

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import AutoTokenizer


def dataloader_shuffle_order(
    dataset_size: int,
    seed: int,
    num_epochs: int,
) -> list[int]:
    """Deterministic per-epoch shuffle of dataset row indices.

    Each epoch is a fresh ``torch.randperm`` drawn from a single seeded generator,
    and the epochs are concatenated. The result is reproducible for a given
    ``seed`` and covers every row exactly once per epoch. The exact permutation
    need not match any particular ``DataLoader``: grouped-advantage training only
    needs the row order to be deterministic and group-consistent, not to mirror a
    specific batch order.

    :param dataset_size: Number of rows in the dataset.
    :type dataset_size: int
    :param seed: Generator seed (matches the env seed).
    :type seed: int
    :param num_epochs: How many epochs of ordering to materialize (>= 1).
    :type num_epochs: int
    :return: Flat list of dataset row indices, ``dataset_size * num_epochs`` long.
    :rtype: list[int]
    """
    if dataset_size <= 0:
        msg = f"dataset_size must be > 0, got {dataset_size}."
        raise ValueError(msg)
    epochs = max(1, num_epochs)
    generator = torch.Generator().manual_seed(seed)
    order: list[int] = []
    for _ in range(epochs):
        order.extend(torch.randperm(dataset_size, generator=generator).tolist())
    return order


@dataclass
class ReasoningRolloutState:
    """Shared dataset-iteration state for a group of single-turn reasoning envs.

    A ``BatchRolloutEnv`` builds one env per trajectory but seeds per batch row
    (shared across the group). Sharing this state lets every trajectory draw from
    one precomputed shuffle order and one monotonic cursor, so the row a given
    ``reset(seed)`` selects is deterministic and group-consistent.

    :param shuffle_order: Precomputed dataset row indices (see
        :func:`dataloader_shuffle_order`).
    :type shuffle_order: list[int]
    :param seed: Generator seed used to build ``shuffle_order``.
    :type seed: int
    :param dataset_size: Rows in the dataset, used to extend the order lazily.
    :type dataset_size: int
    :param cursor: Next position to consume in ``shuffle_order``.
    :type cursor: int
    :param epochs_built: Epochs of ordering already materialized.
    :type epochs_built: int
    """

    shuffle_order: list[int]
    seed: int
    dataset_size: int
    cursor: int = 0
    epochs_built: int = 1
    _seed_to_position: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        seed: int = 42,
        column: str = "question",
    ) -> ReasoningRolloutState:
        """Build state with a first-epoch shuffle order over ``dataset``'s rows.

        :param dataset: Dataset whose length sets the shuffle range.
        :type dataset: Dataset
        :param seed: Shuffle seed (shared across a trajectory group).
        :type seed: int
        :param column: Column used only to size the dataset when ``len`` is
            unavailable; ignored otherwise.
        :type column: str
        :return: A fresh shared state seeded for one epoch.
        :rtype: ReasoningRolloutState
        """
        try:
            dataset_size = len(dataset)
        except TypeError:
            dataset_size = len(dataset[column])
        return cls(
            shuffle_order=dataloader_shuffle_order(dataset_size, seed, 1),
            seed=seed,
            dataset_size=dataset_size,
        )

    def position_for_seed(self, seed: int | None) -> int:
        """Map a reset seed to a dataset-order position, monotonic per unique seed.

        ``BatchRolloutEnv`` reuses the same per-row seed across a group, so all
        trajectories in a group must land on the same row. The first time a seed
        is seen it is bound to the current cursor (which then advances); repeats of
        that seed return the bound position. ``None`` always advances the cursor.

        :param seed: The reset seed, or ``None`` for unconditioned advancement.
        :type seed: int | None
        :return: Index into ``shuffle_order`` for the selected row.
        :rtype: int
        """
        if seed is not None and seed in self._seed_to_position:
            return self._seed_to_position[seed]
        position = self.cursor
        self.cursor += 1
        if seed is not None:
            self._seed_to_position[seed] = position
        return position

    def row_index(self, position: int) -> int:
        """Return the dataset row at ``position``, extending the order if needed.

        Extension appends whole epochs computed from ``seed`` so the originally
        supplied first-epoch order is never overwritten (it may have been seeded
        explicitly to mirror an existing dataloader).
        """
        while position >= len(self.shuffle_order):
            self.epochs_built += 1
            full = dataloader_shuffle_order(
                self.dataset_size,
                self.seed,
                self.epochs_built,
            )
            self.shuffle_order.extend(
                full[(self.epochs_built - 1) * self.dataset_size :]
            )
        return self.shuffle_order[position]


class SingleTurnReasoningEnv:
    """Raw single-turn reasoning env: dataset-seeded prompt in, scored text out.

    Text observation / text action, terminating after one turn. Wrapped by
    :class:`~agilerl.llm_envs.token_observation.TokenObservationWrapper` (via
    :func:`make_reasoning_rollout_env`) to participate in the rollout taxonomy.
    The reward is ``reward_fn(completion, answer, question)`` on the decoded
    generation — identical to the scoring the batched reasoning env applied.

    :param questions: Per-row question strings.
    :type questions: list[str]
    :param answers: Per-row training answer strings.
    :type answers: list[str]
    :param reward_fn: ``(completion, answer, question) -> float`` scorer.
    :type reward_fn: Callable[[str, str, str], float]
    :param prompt_builder: Maps a question to the prompt text shown to the model.
    :type prompt_builder: Callable[[str], str]
    :param state: Shared dataset-iteration state across a trajectory group.
    :type state: ReasoningRolloutState
    :param test_questions: Held-out question strings used under ``eval_mode``.
    :type test_questions: list[str] | None
    :param test_answers: Held-out answer strings used under ``eval_mode``.
    :type test_answers: list[str] | None
    """

    max_turns = 1

    def __init__(
        self,
        questions: list[str],
        answers: list[str],
        reward_fn: Callable[[str, str, str], float],
        prompt_builder: Callable[[str], str],
        state: ReasoningRolloutState,
        test_questions: list[str] | None = None,
        test_answers: list[str] | None = None,
    ) -> None:
        self.questions = questions
        self.answers = answers
        self.reward_fn = reward_fn
        self.prompt_builder = prompt_builder
        self.state = state
        self.test_questions = test_questions
        self.test_answers = test_answers
        self.evaluation_mode = False
        self._question: str = ""
        self._answer: str = ""

    def _active_rows(self) -> tuple[list[str], list[str]]:
        """Return the (questions, answers) for the current train/eval split."""
        if self.evaluation_mode and self.test_questions is not None:
            return self.test_questions, self.test_answers
        return self.questions, self.answers

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        """Select the next dataset row and return its prompt text plus info."""
        questions, answers = self._active_rows()
        position = self.state.position_for_seed(seed)
        row = self.state.row_index(position) % len(questions)
        self._question = questions[row]
        self._answer = answers[row]
        return self.prompt_builder(self._question), {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Score the completion against the current row; terminate (one turn)."""
        reward = float(self.reward_fn(action, self._answer, self._question))
        return "", reward, True, False, {}

    @contextmanager
    def eval_mode(self) -> Generator[None, None, None]:
        """Draw from the held-out test split for the duration of the block."""
        previous = self.evaluation_mode
        self.evaluation_mode = True
        try:
            yield
        finally:
            self.evaluation_mode = previous

    def close(self) -> None:
        """No resources to release."""


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


def make_reasoning_rollout_env(
    train_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: AutoTokenizer,
    reward_fn: Callable[[str, str, str], float],
    *,
    prompt_builder: Callable[[str], str] | None = None,
    conversation_template: list[dict[str, str]] | None = None,
    evaluation_mode: bool = False,
    seed: int = 42,
    max_model_len: int | None = None,
    max_output_tokens: int | None = None,
    state: ReasoningRolloutState | None = None,
) -> TokenObservationWrapper:
    """Build a single-turn reasoning ``RolloutEnv`` wrapped for token rollouts.

    The returned wrapper exposes the rollout-env surface (``reset`` ->
    ``(obs_dict, info)``, ``step(full_completion_ids)`` -> 5-tuple,
    ``get_episode_data``) that ``BatchRolloutEnv`` drives. Pass a shared
    ``state`` (and identical ``seed``) to every env in a trajectory group so the
    dataset order is consistent and reproducible.

    :param train_dataset: Training dataset with ``question`` / ``answer`` columns.
    :type train_dataset: Dataset
    :param test_dataset: Held-out dataset (used when ``evaluation_mode``).
    :type test_dataset: Dataset
    :param tokenizer: Tokenizer used for chat-templating and decoding.
    :type tokenizer: AutoTokenizer
    :param reward_fn: ``(completion, answer, question) -> float`` scorer.
    :type reward_fn: Callable[[str, str, str], float]
    :param prompt_builder: Optional explicit question->prompt-text function;
        defaults to one derived from ``conversation_template``.
    :type prompt_builder: Callable[[str], str] | None
    :param conversation_template: Template used to build ``prompt_builder`` when
        the latter is not given.
    :type conversation_template: list[dict[str, str]] | None
    :param evaluation_mode: Draw from ``test_dataset`` instead of ``train_dataset``.
    :type evaluation_mode: bool
    :param seed: Shuffle seed (shared across a trajectory group).
    :type seed: int
    :param max_model_len: Optional context window forwarded to the wrapper.
    :type max_model_len: int | None
    :param max_output_tokens: Optional generation cap forwarded to the wrapper.
    :type max_output_tokens: int | None
    :param state: Optional shared dataset-iteration state; created if omitted.
    :type state: ReasoningRolloutState | None
    :return: A token-observation-wrapped single-turn reasoning env.
    :rtype: TokenObservationWrapper
    """
    train_questions, train_answers = _extract_question_answer_columns(train_dataset)
    test_questions, test_answers = _extract_question_answer_columns(test_dataset)

    if prompt_builder is None:
        if conversation_template is None:
            msg = "Provide either prompt_builder or conversation_template."
            raise ValueError(msg)
        prompt_builder = _default_prompt_builder(conversation_template)

    if state is None:
        state = ReasoningRolloutState.from_dataset(train_dataset, seed=seed)

    raw_env = SingleTurnReasoningEnv(
        questions=train_questions,
        answers=train_answers,
        reward_fn=reward_fn,
        prompt_builder=prompt_builder,
        state=state,
        test_questions=test_questions,
        test_answers=test_answers,
    )
    raw_env.evaluation_mode = evaluation_mode
    pad_id = getattr(tokenizer, "pad_token_id", None)
    wrapper = TokenObservationWrapper(
        raw_env,
        tokenizer=tokenizer,
        max_turns=1,
        pad_id=pad_id,
        apply_chat_template=True,
        max_model_len=max_model_len,
        max_output_tokens=max_output_tokens,
    )
    # Surface ``eval_mode`` so ``agent.test`` evaluates on the held-out split
    # (the trainer reuses the same factory/config for train and test envs).
    wrapper.eval_mode = raw_env.eval_mode
    return wrapper
