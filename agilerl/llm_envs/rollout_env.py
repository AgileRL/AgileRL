"""Generation rollout envs, with reasoning as the one-turn case.

A :class:`RolloutEnv` is the generation half of the env taxonomy: the model
generates a completion to a dataset-seeded prompt and the env scores it with a
``reward_fn``. Reasoning is the degenerate ``max_turns=1`` configuration — a
plain :class:`RolloutEnv` instance, no subclass. :func:`make_reasoning_rollout_env`
builds one and wraps it in
:class:`~agilerl.llm_envs.token_observation.TokenObservationWrapper` so it plugs
into ``BatchRolloutEnv`` like any other rollout env.

Dataset order is deterministic: the seeded shuffle is precomputed as an explicit
index list (:func:`dataloader_shuffle_order`). A ``BatchRolloutEnv`` owns the
shared cursor (:class:`BatchIterationState`) across its trajectories so a per-row
seed selects one reproducible dataset row for every trajectory in its group.
Batch/row order need only be deterministic and group-consistent, which is what
grouped-advantage training relies on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from agilerl.llm_envs.base import LLMEnv

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import AutoTokenizer

    from agilerl.llm_envs.token_observation import TokenObservationWrapper


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
class BatchIterationState:
    """Shared dataset-iteration state for a group of dataset-backed rollout trajectories.

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
    def from_dataset_size(
        cls,
        dataset_size: int,
        seed: int = 42,
    ) -> BatchIterationState:
        """Build state with a first-epoch shuffle order over ``dataset_size`` rows.

        :param dataset_size: Number of rows the shuffle order ranges over.
        :type dataset_size: int
        :param seed: Shuffle seed (shared across a trajectory group).
        :type seed: int
        :return: A fresh shared state seeded for one epoch.
        :rtype: BatchIterationState
        """
        return cls(
            shuffle_order=dataloader_shuffle_order(dataset_size, seed, 1),
            seed=seed,
            dataset_size=dataset_size,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        seed: int = 42,
        column: str = "question",
    ) -> BatchIterationState:
        """Build state with a first-epoch shuffle order over ``dataset``'s rows.

        :param dataset: Dataset whose length sets the shuffle range.
        :type dataset: Dataset
        :param seed: Shuffle seed (shared across a trajectory group).
        :type seed: int
        :param column: Column used only to size the dataset when ``len`` is
            unavailable; ignored otherwise.
        :type column: str
        :return: A fresh shared state seeded for one epoch.
        :rtype: BatchIterationState
        """
        try:
            dataset_size = len(dataset)
        except TypeError:
            dataset_size = len(dataset[column])
        return cls.from_dataset_size(dataset_size, seed=seed)

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

    def row_for_seed(self, seed: int | None) -> int:
        """Return the dataset row a reset ``seed`` selects.

        Convenience over :meth:`position_for_seed` then :meth:`row_index`: it
        binds (or reuses) the seed's position and resolves it to a row index.

        :param seed: The reset seed, or ``None`` for unconditioned advancement.
        :type seed: int | None
        :return: Dataset row index for the selected position.
        :rtype: int
        """
        return self.row_index(self.position_for_seed(seed))

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


class RolloutEnv(LLMEnv):
    """Generation rollout env: dataset-seeded prompt in, scored text out.

    Text observation / text action. With ``max_turns=1`` (the default) this is the
    reasoning env: the model produces one completion to a dataset-seeded prompt and
    the env scores it via ``reward_fn(completion, answer, question)`` on the decoded
    generation. Multi-turn / tool-using rollouts subclass this and override
    :meth:`step`. Wrapped by
    :class:`~agilerl.llm_envs.token_observation.TokenObservationWrapper` (via
    :func:`make_reasoning_rollout_env`) to participate in the rollout taxonomy.

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
        # Dataset cursor owned by ``BatchRolloutEnv`` when batched; built lazily
        # here only for the standalone / eval path, per active split.
        self._standalone_state: BatchIterationState | None = None
        self._standalone_split: str = ""

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

        :param seed: Optional reset seed; only consulted on the standalone path.
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
            if self._standalone_state is None or self._standalone_split != split:
                self._standalone_state = BatchIterationState.from_dataset_size(
                    len(questions),
                    seed=42,
                )
                self._standalone_split = split
            row_index = self._standalone_state.row_for_seed(seed)
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
) -> TokenObservationWrapper:
    """Build a single-turn reasoning ``RolloutEnv`` wrapped for token rollouts.

    The returned wrapper exposes the rollout-env surface (``reset`` ->
    ``(obs_dict, info)``, ``step(full_completion_ids)`` -> 5-tuple,
    ``get_episode_data``) that ``BatchRolloutEnv`` drives. When driven by a
    ``BatchRolloutEnv`` the dataset cursor is owned there, so the same per-row
    seed selects one reproducible row for every env in a trajectory group.

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
    :return: A token-observation-wrapped single-turn reasoning env.
    :rtype: TokenObservationWrapper
    """
    from agilerl.llm_envs.token_observation import TokenObservationWrapper

    train_questions, train_answers = _extract_question_answer_columns(train_dataset)
    test_questions, test_answers = _extract_question_answer_columns(test_dataset)

    if prompt_builder is None:
        if conversation_template is None:
            msg = "Provide either prompt_builder or conversation_template."
            raise ValueError(msg)
        prompt_builder = _default_prompt_builder(conversation_template)

    raw_env = RolloutEnv(
        max_turns=1,
        questions=train_questions,
        answers=train_answers,
        reward_fn=reward_fn,
        prompt_builder=prompt_builder,
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
