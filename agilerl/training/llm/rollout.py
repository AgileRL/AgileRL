# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Generative rollout training for LLM finetuning.

:func:`train_llm_rollout` drives on-policy RL (GRPO / PPO / REINFORCE) over a
:class:`~agilerl.llm_envs.RolloutHarness`; single-turn reasoning
(``max_turns=1``) and multi-turn share this one loop.
"""

import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from accelerate import Accelerator

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import GRPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.protocols import SelectionStrategyProtocol
from agilerl.training.llm.common import _validate_finetune_args
from agilerl.utils.llm_utils import (
    align_completion_batch_shapes_across_ranks,
    allreduce_minmax_int,
    needs_cross_rank_seq_padding,
    safe_aggregate_metrics,
)
from agilerl.utils.utils import (
    data_parallel_topology,
    default_progress_bar,
    init_loggers,
    resolve_selection_strategy,
    run_selection_and_mutation,
    save_llm_checkpoint,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.algorithms import LLMPPO, LLMREINFORCE
    from agilerl.llm_envs import RolloutCollector, RolloutHarness
    from agilerl.rollouts.on_policy import collate_llm_rollouts, collect_rollouts_llm

if TYPE_CHECKING:
    SupportedRollout = GRPO | LLMPPO | LLMREINFORCE


def _any_rank_flag(
    local: bool,
    accelerator: Accelerator | None,
) -> bool:
    """Whether any data-parallel rank has this flag set."""
    if accelerator is None:
        return local
    _, max_flag = allreduce_minmax_int(int(local), accelerator)
    return max_flag == 1


def _any_rank_empty_batch(
    local_empty: bool,
    accelerator: Accelerator | None,
) -> bool:
    """Whether any data-parallel rank has an empty rollout this step."""
    # Mixed empty/non-empty ranks would split learn() collectives.
    return _any_rank_flag(local_empty, accelerator)


def train_llm_rollout(
    pop: "list[SupportedRollout]",
    max_turns: int,
    env_factory: "Callable[[], RolloutHarness]",
    init_hp: dict[str, Any] | None = None,
    max_steps: int = 32768,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    csv: bool = False,
    csv_log_dir: str | None = None,
    evo_steps: int | None = None,
    checkpoint_steps: int | None = None,
    checkpoint_path: str | None = None,
    selection_strategy: SelectionStrategyProtocol | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    evaluation_interval: int = 50,
    eval_loop: int = 1,
    max_reward: float | None = None,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    max_wall_seconds: float | None = None,
    io_timeout_s: float | None = 600.0,
) -> "tuple[list[SupportedRollout], Any]":
    """Train a population of LLM agents over rollout (generate-and-score) environments.

    Collects token-level episodes (``reset`` returns ``(obs, info)``, repeated
    ``get_action`` / ``step`` (full completion tensor), then ``get_episode_data``),
    then runs turn-level updates. For a ``RolloutHarness`` with ``max_model_len`` set,
    a trajectory whose cumulative prompt would overflow the context is stopped
    with ``truncated=True``.

    :param pop: Population of LLMPPO, LLMREINFORCE or GRPO agents to finetune.
    :type pop: list[SupportedRollout]
    :param max_turns: Maximum interaction turns per episode.
    :type max_turns: int
    :param env_factory: Zero-arg factory that returns a fresh env for each
        trajectory rollout. Required to ensure trajectory state isolation.
    :type env_factory: Callable[[], RolloutHarness]
    :param init_hp: Initial hyperparameters.
    :type init_hp: dict, optional
    :param max_steps: Progress-bar budget in sample steps, defaults to 32768.
    :type max_steps: int
    :param save_elite: Whether to save the elite checkpoint, defaults to None.
    :type save_elite: bool, optional
    :param elite_path: Directory for checkpoints, defaults to None.
    :type elite_path: str, optional
    :param wb: Whether to log to Weights and Biases, defaults to False.
    :type wb: bool, optional
    :param tensorboard: Whether to log to TensorBoard, defaults to False.
    :type tensorboard: bool, optional
    :param tensorboard_log_dir: Directory for TensorBoard event files, defaults to None.
    :type tensorboard_log_dir: str, optional
    :param csv: Whether to log aggregate metrics to CSV, defaults to False.
    :type csv: bool, optional
    :param csv_log_dir: Path for the CSV file, defaults to None.
    :type csv_log_dir: str, optional
    :param evo_steps: Steps between evolution (requires a selection strategy and mutation).
    :type evo_steps: int, optional
    :param checkpoint_steps: Save checkpoint every N outer iterations when no evolution.
    :type checkpoint_steps: int, optional
    :param checkpoint_path: Directory for periodic checkpoints; falls back to elite_path.
    :type checkpoint_path: str, optional
    :param selection_strategy: Selection strategy driving evolution, defaults to None.
    :type selection_strategy: SelectionStrategyProtocol | None, optional
    :param tournament: Deprecated alias for selection_strategy, defaults to None.
    :type tournament: TournamentSelection, optional
    :param mutation: Mutation operator for evolution, defaults to None.
    :type mutation: Mutations, optional
    :param wandb_api_key: W&B API key, defaults to None.
    :type wandb_api_key: str, optional
    :param wandb_kwargs: Additional kwargs forwarded to ``wandb.init()``.
    :type wandb_kwargs: dict, optional
    :param evaluation_interval: How often to call ``agent.test`` on a fresh
        environment from ``env_factory``.
    :type evaluation_interval: int, optional
    :param eval_loop: Episodes averaged per evaluation for the HPO fitness score;
        defaults to 1 (matching the other trainers). Raise it for less noisy
        tournament selection at higher eval cost.
    :type eval_loop: int, optional
    :param max_reward: If set, adds accuracy metric vs this threshold.
    :type max_reward: float, optional
    :param verbose: Progress bar and periodic train summaries, defaults to True.
    :type verbose: bool
    :param accelerator: Hugging Face Accelerate instance, defaults to None.
    :type accelerator: Accelerator, optional
    :param max_wall_seconds: Stop after this wall-clock duration (seconds); ``None`` disables.
    :type max_wall_seconds: float | None
    :param io_timeout_s: Backstop deadline for one concurrent round of env
        round-trips; a hung env or stalled transport raises ``TimeoutError``
        rather than blocking the batch forever. Defaults to 600 s; ``None``
        disables it. Forwarded to :class:`RolloutCollector`.
    :type io_timeout_s: float | None
    :return: The finetuned population and its last recorded fitnesses.
    :rtype: tuple[list[SupportedRollout], Any]
    """
    selection_strategy = resolve_selection_strategy(selection_strategy, tournament)

    _validate_finetune_args(
        evo_steps,
        selection_strategy,
        mutation,
        None,
        max_steps,
        pop,
        # GRPO covers its variants too -- CISPO and GSPO subclass it, so they
        # pass this check without being named.
        (LLMPPO, LLMREINFORCE, GRPO),
        (
            "The algorithm must be LLMPPO, LLMREINFORCE, or GRPO (including the "
            "GRPO variants CISPO and GSPO) for rollout finetuning. Got "
            f"{type(pop[0])} instead."
        ),
        checkpoint_steps=checkpoint_steps,
    )

    if init_hp is None:
        init_hp = {
            "BATCH_SIZE_PER_GPU": pop[0].batch_size_per_process,
            "ALGO": pop[0].algo,
        }

    batch_size = init_hp.get("BATCH_SIZE", pop[0].batch_size)
    env_name = init_hp.get("env_name", "rollout")
    # Tensor-parallel ranks of one replica generate the same sequences, so the
    # data-parallel topology -- not the process group -- is what splits work.
    dp_rank, data_increment = data_parallel_topology(
        accelerator,
        getattr(getattr(pop[0], "vllm_config", None), "tensor_parallel_size", 1) or 1,
    )
    effective_data_batch_size = data_increment * batch_size

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = batch_size
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path
        init_hp["max_turns"] = max_turns

    pbar = default_progress_bar(max_steps, accelerator)

    loggers = init_loggers(
        algo=init_hp.get("ALGO", pop[0].algo),
        env_name=env_name,
        pbar=pbar,
        verbose=verbose,
        wb=wb,
        tensorboard=tensorboard,
        csv=csv,
        tensorboard_log_dir=tensorboard_log_dir,
        csv_log_dir=csv_log_dir,
        accelerator=accelerator,
        wandb_api_key=wandb_api_key,
        wandb_kwargs=wandb_kwargs,
        init_hyperparams=init_hp,
    )

    population = Population(agents=pop, accelerator=accelerator, loggers=loggers)

    total_steps = 0
    i = 0
    next_checkpoint_step = checkpoint_steps
    max_steps_checkpoint_saved = False
    # Count consecutive iterations that advanced ``total_steps`` by 0 so a rollout
    # that can never make progress aborts instead of looping forever (see the
    # guard after the per-agent loop).
    consecutive_stalls = 0
    group_size = getattr(pop[0], "group_size", 1)
    # Rank-offset seed for procedural env reset(seed=...) uniqueness only. Dataset
    # rows are split by the TaskAssigner's contiguous per-rank shard instead.
    group_seed = int(pop[0].seed) + dp_rank * (1 << 31)
    rollout_collector = RolloutCollector(
        env_factory,
        batch_size,
        group_size,
        io_timeout_s=io_timeout_s,
        rank=dp_rank,
        world_size=data_increment,
    )
    test_env: RolloutHarness | None = None
    try:
        wall_deadline = (
            time.monotonic() + max_wall_seconds
            if max_wall_seconds is not None and max_wall_seconds > 0
            else None
        )
        while total_steps < max_steps:
            # Every DP rank takes the same stop/continue branch; a rank-local
            # wall-clock check can leave a peer blocked in collect's barrier.
            if wall_deadline is not None and _any_rank_flag(
                time.monotonic() >= wall_deadline,
                accelerator,
            ):
                if accelerator is None or accelerator.is_main_process:
                    print(
                        f"\nStopping rollout training: wall time limit ({max_wall_seconds}s) reached.",
                    )
                break

            iteration_steps = 0
            for agent in population.agents:
                # Refresh the KL reference once per completed pass over the dataset
                # rows; a non-dataset env keeps ``num_epochs == 0``, leaving the
                # anchor at the initial policy.
                agent.set_reference_policy(rollout_collector.num_epochs)
                agent.init_training_step()
                (
                    token_ids_list,
                    action_masks_list,
                    all_turn_ids,
                    all_rewards,
                    batch_steps,
                    group_seed,
                    all_sampling_logps,
                ) = collect_rollouts_llm(
                    agent=agent,
                    env=rollout_collector,
                    n_steps=max_turns,
                    batch_size=batch_size,
                    group_seed=group_seed,
                )

                # Collated a prompt group at a time, so the group-divisibility the
                # algorithms rely on holds by construction, and a misaligned
                # rollout fails here rather than inside a loss.
                batch = collate_llm_rollouts(
                    token_ids_list,
                    action_masks_list,
                    all_turn_ids,
                    all_rewards,
                    all_sampling_logps,
                    group_size=group_size,
                )
                # Empty collated batches have no trajectories; learn() raises on them.
                # iteration_steps stays 0, so the stall guard after this loop fires.
                if _any_rank_empty_batch(
                    batch_steps == 0 or batch.is_empty,
                    accelerator,
                ):
                    agent.finalize_training_step(0)
                else:
                    episode_scores = batch.rewards.sum(dim=1)
                    mean_score = episode_scores.mean().to(agent.device)

                    experiences = batch.experiences()
                    turn_ids = batch.turn_ids
                    if accelerator is not None and needs_cross_rank_seq_padding(
                        agent, world_size=data_increment
                    ):
                        # Multi-rank Liger token-level losses allreduce per chunk, so
                        # every rank must pad to one global sequence length.
                        token_ids, action_masks, rewards = (
                            align_completion_batch_shapes_across_ranks(
                                batch.token_ids,
                                batch.action_masks,
                                batch.rewards,
                                pad_token_id=agent.pad_token_id,
                                accelerator=accelerator,
                            )
                        )
                        experiences = (token_ids, action_masks, rewards)
                        if turn_ids is None:
                            msg = "aligned batches are non-empty"
                            raise RuntimeError(msg)
                        target_mask_len = int(action_masks.shape[1])
                        if int(turn_ids.shape[1]) < target_mask_len:
                            turn_ids = torch.nn.functional.pad(
                                turn_ids,
                                (0, target_mask_len - int(turn_ids.shape[1])),
                                value=-1,
                            )

                    agent.learn(
                        experiences,
                        turn_ids=turn_ids,
                        sampling_logps=batch.sampling_logps,
                    )

                    agg_score = safe_aggregate_metrics(accelerator, mean_score)

                    if max_reward is not None:
                        if "accuracy" not in agent.metrics.additional_metrics:
                            agent.metrics.register("accuracy")
                        accuracy = (
                            (episode_scores >= max_reward)
                            .float()
                            .mean()
                            .to(agent.device)
                        )
                        agg_accuracy = safe_aggregate_metrics(accelerator, accuracy)
                        if accelerator is None or accelerator.is_main_process:
                            agent.metrics.log("accuracy", agg_accuracy)

                    for (
                        name,
                        mean_value,
                    ) in rollout_collector.get_rubric_score_means().items():
                        metric = f"reward_{name}"
                        if metric not in agent.metrics.additional_metrics:
                            agent.metrics.register(metric)
                        agg = safe_aggregate_metrics(
                            accelerator, mean_score.new_tensor(mean_value)
                        )
                        if accelerator is None or accelerator.is_main_process:
                            agent.metrics.log(metric, agg)

                    effective_batch_steps = batch_steps * data_increment
                    agent.finalize_training_step(batch_steps)
                    total_steps += effective_batch_steps
                    iteration_steps += effective_batch_steps

                    if accelerator is None or accelerator.is_main_process:
                        agent.add_scores([float(agg_score)])

                if (i + 1) % evaluation_interval == 0:
                    if test_env is None:
                        test_env = env_factory()
                    agent.test(test_env, loop=eval_loop)
                    if accelerator is not None:
                        accelerator.wait_for_everyone()

            # ``total_steps`` only advances by ``batch_steps``; an iteration where
            # every agent's rollout yielded no turns leaves it unchanged, so a
            # rollout that always yields nothing would loop forever. Tolerate a few
            # such iterations (e.g. an occasional over-budget dataset row) but abort
            # once it is clearly systematic, rather than spinning silently.
            if iteration_steps == 0:
                consecutive_stalls += 1
                if accelerator is None or accelerator.is_main_process:
                    warnings.warn(
                        "Rollout produced no usable turns this iteration "
                        "(batch_steps == 0 for every agent), so training did not "
                        "advance. Every prompt likely exceeds the context budget — "
                        "check that max_output_tokens leaves room under "
                        "max_context_length.",
                        stacklevel=2,
                    )
                if consecutive_stalls >= 8:
                    msg = (
                        f"Rollout training made no progress for {consecutive_stalls} "
                        "consecutive iterations (batch_steps == 0 every time), so "
                        "total_steps can never reach max_steps. Aborting instead of "
                        "looping forever. Likely cause: the prompt budget is "
                        "exhausted (max_output_tokens too close to "
                        "max_context_length), or every sampled prompt exceeds the "
                        "context length."
                    )
                    raise RuntimeError(msg)
            else:
                consecutive_stalls = 0

            if accelerator is None or accelerator.is_main_process:
                pbar.update(iteration_steps // len(population.agents))
                population.report_metrics(clear=True)
            else:
                # Metrics accumulate on every rank; only main reports, so the
                # others must still clear or their stores grow for the whole run.
                population.clear_agent_metrics()

            if accelerator is not None:
                accelerator.wait_for_everyone()

            if (
                selection_strategy is not None
                and mutation is not None
                and evo_steps is not None
            ):
                if (i + 1) % evo_steps == 0:
                    if accelerator is not None:
                        accelerator.wait_for_everyone()
                    population.update(
                        run_selection_and_mutation(
                            selection_strategy,
                            population=population.agents,
                            mutation=mutation,
                            env_name=env_name,
                            accelerator=accelerator,
                            language_model=True,
                            elite_path=elite_path,
                            save_elite=bool(save_elite),
                        ),
                    )
                    group_size = getattr(population.agents[0], "group_size", 1)
                    rollout_collector.update_rollout_geometry(
                        rollout_batch_size=batch_size,
                        group_size=group_size,
                    )
                    if accelerator is not None:
                        accelerator.wait_for_everyone()

                    population.increment_evo_step()
            else:
                checkpoint_due = False
                if checkpoint_steps is not None:
                    while (
                        next_checkpoint_step is not None
                        and total_steps >= next_checkpoint_step
                    ):
                        checkpoint_due = True
                        next_checkpoint_step += checkpoint_steps
                if (
                    total_steps >= max_steps
                    and not max_steps_checkpoint_saved
                    and (
                        checkpoint_steps is not None
                        or checkpoint_path is not None
                        or elite_path is not None
                    )
                ):
                    checkpoint_due = True
                    max_steps_checkpoint_saved = True
                if checkpoint_due:
                    save_llm_checkpoint(
                        population.agents[-1],
                        checkpoint_path if checkpoint_path is not None else elite_path,
                    )

            i += 1

        if save_elite and elite_path is not None:
            elite = max(
                population.agents,
                key=lambda agent: agent.fitness[-1] if agent.fitness else float("-inf"),
            )
            save_llm_checkpoint(elite, elite_path)
    finally:
        # Release the rollout envs (and any per-rollout OpenEnv servers they own)
        # plus the test env, including on error.
        try:
            rollout_collector.close()
        finally:
            if test_env is not None:
                test_env.close()

    population.finish()
    pbar.close()
    return population.agents, population.last_fitnesses
