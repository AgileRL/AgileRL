"""Generation rollout envs, with reasoning as the one-turn case.

A :class:`RolloutEnv` is the generation half of the env taxonomy: the model
generates a completion to a dataset-seeded prompt and the env scores it with a
``reward_fn``. Reasoning is the degenerate ``max_turns=1`` configuration — a
plain :class:`RolloutEnv` instance, no subclass. Callers wrap it in
:class:`~agilerl.llm_envs.rollout_harness.RolloutHarness` so it plugs
into ``BatchRolloutEnv`` like any other rollout env.

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
from typing import TYPE_CHECKING, Any

import torch

from agilerl.llm_envs.base import LLMEnv
from agilerl.llm_envs.rollout_buffer import RolloutBuffer, Trajectory

if TYPE_CHECKING:
    from agilerl.typing import RolloutPrompts


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
    :class:`~agilerl.llm_envs.token_observation.RolloutHarness` to
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


class BatchRolloutEnv:
    """Batched in-process collector of LLM rollout episodes.

    Maintains ``batch_size * group_size`` independent rollout environments and steps all
    active trajectories in lock-step using policy completions.
    """

    def __init__(
        self,
        env_factory: Callable[..., RolloutEnv],
        batch_size: int,
        group_size: int,
        env_config: dict[str, Any] | None = None,
    ):
        """Create ``batch_size * group_size`` independent environments.

        :param env_factory: Factory that builds one multi-turn environment.
        :type env_factory: Callable[..., RolloutEnv]
        :param batch_size: Number of logical batch items.
        :type batch_size: int
        :param group_size: Number of grouped trajectories per batch item.
        :type group_size: int
        :param env_config: Optional kwargs passed to ``env_factory``.
        :type env_config: dict[str, Any] | None
        """
        if batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}."
            raise ValueError(msg)
        if group_size <= 0:
            msg = f"group_size must be > 0, got {group_size}."
            raise ValueError(msg)
        if env_config is None:
            env_config = {}
        self.env_factory = env_factory
        self.env_config = env_config
        self.num_envs = batch_size * group_size
        self.batch_size = batch_size
        self.group_size = group_size
        self.trajectories = RolloutBuffer(batch_size, group_size)
        self._cursor = 0
        self._epoch_order: list[int] | None = None
        self._dataset_size = 0
        self._shuffle_seed = 0

    def _next_row(self) -> int:
        """Advance the shared dataset cursor and return the next (shuffled) row.

        Re-shuffles deterministically at each epoch boundary so every row is seen
        once per epoch; the seed makes the order reproducible.
        """
        epoch, pos = divmod(self._cursor, self._dataset_size)
        if pos == 0:
            self._epoch_order = dataloader_shuffle_order(
                self._dataset_size, self._shuffle_seed + epoch
            )
        row = self._epoch_order[pos]
        self._cursor += 1
        return row

    def reset(
        self,
        seed: int | None = None,
    ) -> list[RolloutPrompts] | None:
        """Reset all environments and initialize trajectories.

        Seeds are assigned per batch row (same seed across groups). The shared
        dataset cursor resolves a single ``row_index`` per batch row (advancing
        once per row, re-shuffling each epoch), and every group env of that row is
        reset with both the row seed and that one row, so the group is
        row-consistent. Prompts are returned in stable ``(batch_idx, group_idx)``
        order. Envs that are not dataset-backed (``dataset_size == 0``) skip the
        cursor entirely.

        :param seed: Optional base seed for deterministic rollouts.
        :type seed: int | None
        :return: Active prompt dictionaries after reset.
        :rtype: list[RolloutPrompts] | None
        """
        seed_base = seed
        for batch_idx in range(self.batch_size):
            batch_seed = None if seed_base is None else seed_base + batch_idx
            row_index: int | None = None
            for group_idx in range(self.group_size):
                env_idx = batch_idx * self.group_size + group_idx
                if not self.trajectories.is_initialized:
                    env_i = self.env_factory(**self.env_config)
                    # Size the shared cursor from the first env's dataset, before
                    # resolving any row, so all rows draw from one shuffle order.
                    # Envs that aren't dataset-backed (no ``dataset_size``, or 0)
                    # get no cursor and a reset without a ``row_index``.
                    ds_size = getattr(env_i, "dataset_size", 0)
                    if self._dataset_size == 0 and ds_size > 0:
                        self._dataset_size = ds_size
                        self._shuffle_seed = seed if seed is not None else 42
                    if group_idx == 0 and self._dataset_size > 0:
                        row_index = self._next_row()
                    prompt_dict, _ = (
                        env_i.reset(seed=batch_seed, row_index=row_index)
                        if row_index is not None
                        else env_i.reset(seed=batch_seed)
                    )
                    self.trajectories.add_trajectory(
                        Trajectory(
                            env=env_i,
                            batch_idx=batch_idx,
                            group_idx=group_idx,
                            prompt=prompt_dict,
                            done=False,
                        )
                    )
                else:
                    if group_idx == 0 and self._dataset_size > 0:
                        row_index = self._next_row()
                    self.trajectories.reset_trajectory(
                        env_idx=env_idx,
                        seed=batch_seed,
                        row_index=row_index,
                    )
        return self.trajectories.get_prompts()

    def step(
        self,
        completion_ids: list[torch.Tensor],
        sampling_logps: list[torch.Tensor | None] | None = None,
    ) -> list[RolloutPrompts] | None:
        """Step each active trajectory with its corresponding completion.

        :param completion_ids: One completion tensor per active trajectory.
        :type completion_ids: list[torch.Tensor]
        :param sampling_logps: Sampling logprobs from vLLM rollout for this
            turn, parallel to ``completion_ids``; entries (or the whole list)
            may be ``None`` when nothing was captured.
        :type sampling_logps: list[torch.Tensor | None] | None
        :return: Next active prompt dictionaries after stepping.
        :rtype: list[RolloutPrompts] | None
        """
        active = self.trajectories.get_active_trajectories(sorted_by_index=True)
        if len(completion_ids) != len(active):
            msg = (
                "Number of completions does not match number of active trajectories: "
                f"{len(completion_ids)} != {len(active)}"
            )
            raise RuntimeError(msg)
        if sampling_logps is not None:
            if len(sampling_logps) != len(active):
                msg = (
                    "Number of sampling logprobs does not match number of active "
                    f"trajectories: {len(sampling_logps)} != {len(active)}"
                )
                raise RuntimeError(msg)
            for traj, slp in zip(active, sampling_logps, strict=True):
                if slp is not None:
                    traj.sampling_logps.append(slp)
        for traj, completion in zip(active, completion_ids, strict=False):
            full_completion = completion
            if full_completion.dim() == 1:
                full_completion = full_completion.unsqueeze(0)
            next_prompt, _reward, terminated, truncated, _info = traj.env.step(
                full_completion,
            )
            traj.done = bool(terminated or truncated)
            if not traj.done:
                traj.prompt = next_prompt
        return self.trajectories.get_prompts()

    def close(self) -> None:
        """Close all underlying environments."""
        seen: set[int] = set()
        for traj in self.trajectories:
            env = traj.env
            env_id = id(env)
            if env_id in seen:
                continue
            seen.add(env_id)
            if hasattr(env, "close"):
                env.close()

    def get_trajectories(
        self,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        int,
        list[torch.Tensor | None] | None,
    ]:
        """Collect complete episode tensors from all trajectories.

        :return: ``(completion_ids_list, action_masks_list, all_turn_ids,
            all_rewards, batch_steps, all_sampling_logps)`` where ``batch_steps``
            is the summed number of recorded turn boundaries across trajectories.
            ``all_sampling_logps`` is ``None`` when no vLLM logprobs were captured
            this rollout; otherwise it holds one 1-D tensor of generated-token
            logprobs per trajectory (concatenated across turns), with ``None`` for
            any trajectory that captured none.
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int, list[torch.Tensor | None] | None]
        """
        completion_ids_list: list[torch.Tensor] = []
        action_masks_list: list[torch.Tensor] = []
        all_turn_ids: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        all_sampling_logps: list[torch.Tensor | None] = []
        batch_steps = 0
        self.trajectories.sort(key=lambda t: (t.batch_idx, t.group_idx))
        for traj in self.trajectories:
            ep_ids, action_mask, turn_ids, turn_rewards_t = traj.env.get_episode_data()
            completion_ids_list.append(ep_ids)
            action_masks_list.append(action_mask)
            all_turn_ids.append(turn_ids)
            all_rewards.append(turn_rewards_t)
            batch_steps += len(getattr(traj.env, "turn_boundaries", []))
            turns = traj.sampling_logps
            all_sampling_logps.append(torch.cat(turns) if turns else None)

        return (
            completion_ids_list,
            action_masks_list,
            all_turn_ids,
            all_rewards,
            batch_steps,
            # Collapse to a single ``None`` when nothing was captured, so the
            # caller needs only an ``is not None`` check (no per-row re-scan).
            (
                all_sampling_logps
                if any(logps is not None for logps in all_sampling_logps)
                else None
            ),
        )
