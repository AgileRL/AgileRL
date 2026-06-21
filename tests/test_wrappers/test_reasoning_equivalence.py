"""Equivalence oracle for the reasoning -> single-turn ``RolloutEnv`` fold.

Reasoning used to be a dedicated ``ReasoningGym`` with a batched, shuffled
``DataLoader``. The fold makes it a one-turn ``RolloutEnv``
(:mod:`agilerl.llm_envs.reasoning_rollout`). The behavioural contract that has to
survive the fold is:

1. **Dataset row order** — must be deterministic per seed and a full permutation
   each epoch. The exact ordering need NOT match the old dataloader's batch order
   (a different valid shuffle is fine); grouped-advantage training only relies on
   the order being deterministic and group-consistent.
2. **Group consistency** — a ``BatchRolloutEnv`` reuses one per-row seed across a
   trajectory group, so every trajectory in a group must land on the same row.
3. **Prompt + reward parity** — the prompt is the templated question and the
   reward is ``reward_fn(completion, answer, question)`` on the decoded
   generation, scored once (single turn).
4. **eval_mode** — draws from the held-out split, then restores the train split.
"""

from __future__ import annotations

from agilerl.llm_envs.reasoning_rollout import (
    ReasoningRolloutState,
    SingleTurnReasoningEnv,
    dataloader_shuffle_order,
)


def test_shuffle_order_is_deterministic_and_a_full_permutation() -> None:
    """Reproducible for a seed; every epoch is a full permutation of all rows."""
    dataset_size, seed, epochs = 37, 42, 3
    order = dataloader_shuffle_order(dataset_size, seed, epochs)

    # Reproducible: same seed -> identical order.
    assert order == dataloader_shuffle_order(dataset_size, seed, epochs)
    # A different seed shuffles differently (the shuffle actually happens).
    assert order != dataloader_shuffle_order(dataset_size, seed + 1, epochs)
    # Each epoch covers every row exactly once.
    assert len(order) == dataset_size * epochs
    for epoch in range(epochs):
        chunk = order[epoch * dataset_size : (epoch + 1) * dataset_size]
        assert sorted(chunk) == list(range(dataset_size))


def test_shuffle_order_extends_without_rewriting_first_epoch() -> None:
    """Row lookups past the first epoch append fresh epochs, never overwrite epoch 0."""
    dataset_size, seed = 11, 7
    state = ReasoningRolloutState(
        shuffle_order=dataloader_shuffle_order(dataset_size, seed, 1),
        seed=seed,
        dataset_size=dataset_size,
    )
    first_epoch = list(state.shuffle_order)
    # Force the order to grow across the epoch boundary.
    _ = state.row_index(dataset_size + 3)
    assert state.shuffle_order[:dataset_size] == first_epoch
    assert (
        dataloader_shuffle_order(dataset_size, seed, state.epochs_built)
        == (state.shuffle_order[: dataset_size * state.epochs_built])
    )


def test_group_shares_one_row_per_seed() -> None:
    """A reused per-row seed pins every group trajectory to the same dataset row."""
    state = ReasoningRolloutState(
        shuffle_order=dataloader_shuffle_order(10, 7, 1),
        seed=7,
        dataset_size=10,
    )
    group_positions = [state.position_for_seed(5) for _ in range(4)]
    assert group_positions == [group_positions[0]] * 4

    next_row_position = state.position_for_seed(6)
    assert next_row_position != group_positions[0]
    # Re-presenting the original seed still returns its bound position.
    assert state.position_for_seed(5) == group_positions[0]


def test_prompt_is_templated_and_reward_scores_once() -> None:
    """reset() returns the templated prompt; step() scores via reward_fn and ends."""
    questions, answers = ["2+2", "3+5"], ["4", "8"]
    state = ReasoningRolloutState(shuffle_order=[0, 1], seed=0, dataset_size=2)

    def reward_fn(completion: str, answer: str, _question: str) -> float:
        return 1.0 if answer in completion else 0.0

    def prompt_builder(question: str) -> str:
        return f"Q: {question}\nA:"

    env = SingleTurnReasoningEnv(
        questions=questions,
        answers=answers,
        reward_fn=reward_fn,
        prompt_builder=prompt_builder,
        state=state,
    )

    prompt, info = env.reset(seed=0)
    assert prompt == "Q: 2+2\nA:"
    assert info == {}

    _, reward, terminated, truncated, _ = env.step("the answer is 4")
    assert reward == 1.0
    assert terminated is True
    assert truncated is False

    # A wrong completion on the next row scores zero, still one turn.
    next_prompt, _ = env.reset(seed=1)
    assert next_prompt == "Q: 3+5\nA:"
    _, wrong_reward, terminated, _, _ = env.step("definitely 99")
    assert wrong_reward == 0.0
    assert terminated is True


def test_eval_mode_draws_from_held_out_split() -> None:
    """Under eval_mode the env serves the test split, restoring the train split after."""
    state = ReasoningRolloutState(shuffle_order=[0], seed=0, dataset_size=1)
    env = SingleTurnReasoningEnv(
        questions=["train-q"],
        answers=["train-a"],
        reward_fn=lambda c, a, q: 0.0,
        prompt_builder=lambda q: q,
        state=state,
        test_questions=["eval-q"],
        test_answers=["eval-a"],
    )
    with env.eval_mode():
        eval_prompt, _ = env.reset(seed=0)
    assert eval_prompt == "eval-q"

    state.cursor = 0
    train_prompt, _ = env.reset(seed=1)
    assert train_prompt == "train-q"
