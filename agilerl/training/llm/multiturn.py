# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
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
    needs_cross_rank_seq_padding,
    safe_aggregate_metrics,
)
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    resolve_selection_strategy,
    run_selection_and_mutation,
    save_llm_checkpoint,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.algorithms import LLMPPO, LLMREINFORCE
    from agilerl.llm_envs import SyncMultiTurnVecEnv
    from agilerl.protocols import TokenizedMultiTurnEnv
    from agilerl.rollouts.on_policy import collect_rollouts_llm
    from agilerl.utils.algo_utils import stack_and_pad_experiences

if TYPE_CHECKING:
    SupportedMultiturn = LLMPPO | LLMREINFORCE | GRPO


def finetune_llm_multiturn(
    pop: "list[SupportedMultiturn]",
    max_turns: int,
    env_factory: "Callable[[], TokenizedMultiTurnEnv]",
    env_config: dict[str, Any] | None = None,
    init_hp: dict[str, Any] | None = None,
    max_steps: int = 32768,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    evo_steps: int | None = None,
    checkpoint_steps: int | None = None,
    checkpoint_path: str | None = None,
    selection_strategy: SelectionStrategyProtocol | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    evaluation_interval: int = 50,
    max_wall_seconds: float | None = None,
    max_reward: float | None = None,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
) -> "tuple[list[SupportedMultiturn], list[float]]":
    """Finetune a population of agents on a multi-turn environment.

    Collects token-level episodes via ``SyncMultiTurnVecEnv`` and
    ``collect_rollouts_llm``, then runs turn-level updates (PPO/REINFORCE with
    ``turn_ids``, or GRPO without).

    :param pop: Population of LLMPPO, LLMREINFORCE, or GRPO agents.
    :type pop: PopulationType
    :param max_turns: Maximum interaction turns per episode.
    :type max_turns: int
    :param env_factory: Factory returning a fresh tokenized multi-turn env for
        each rollout.
    :type env_factory: Callable[[], TokenizedMultiTurnEnv]
    :param env_config: Configuration for the environment factory.
    :type env_config: dict[str, Any], optional
    :param init_hp: Initial hyperparameters.
    :type init_hp: dict, optional
    :param max_steps: Progress-bar budget in sample steps, defaults to 32768.
    :type max_steps: int
    :param save_elite: Whether to save the elite checkpoint, defaults to None.
    :type save_elite: bool, optional
    :param elite_path: Directory for checkpoints, defaults to None.
    :type elite_path: str, optional
    :param wb: Whether to log to Weights and Biases, defaults to False.
    :type wb: bool
    :param tensorboard: TensorBoard tracking, defaults to False.
    :type tensorboard: bool
    :param tensorboard_log_dir: Directory for TensorBoard logs, defaults to None.
    :type tensorboard_log_dir: str, optional
    :param evo_steps: Steps between evolution (requires tournament and mutation).
    :type evo_steps: int, optional
    :param checkpoint_steps: Save checkpoint every N outer iterations when no evolution.
    :type checkpoint_steps: int, optional
    :param checkpoint_path: Directory for periodic checkpoints; falls back to elite_path, defaults to None
    :type checkpoint_path: str | None, optional
    :param selection_strategy: Selection strategy driving evolution, defaults to None.
    :type selection_strategy: SelectionStrategyProtocol | None, optional
    :param tournament: Deprecated alias for selection_strategy, defaults to None.
    :type tournament: TournamentSelection, optional
    :param mutation: Mutation operator for evolution, defaults to None.
    :type mutation: Mutations, optional
    :param wandb_api_key: W&B API key, defaults to None.
    :type wandb_api_key: str, optional
    :param wandb_kwargs: Additional kwargs to pass to wandb.init()
    :type wandb_kwargs: dict, optional
    :param evaluation_interval: Evaluate every N outer iterations.
    :type evaluation_interval: int
    :param max_wall_seconds: Stop after this wall-clock duration (seconds); ``None`` disables.
    :type max_wall_seconds: float | None
    :param max_reward: If set, adds accuracy metric vs this threshold.
    :type max_reward: float, optional
    :param verbose: Progress bar and periodic train summaries, defaults to True.
    :type verbose: bool
    :param accelerator: Hugging Face Accelerate instance, defaults to None.
    :type accelerator: Accelerator, optional
    :return: The finetuned population (same list object, possibly mutated in place).
    :rtype: PopulationType
    """
    selection_strategy = resolve_selection_strategy(selection_strategy, tournament)

    _validate_finetune_args(
        evo_steps,
        selection_strategy,
        mutation,
        None,
        max_steps,
        pop,
        (LLMPPO, LLMREINFORCE, GRPO),
        (
            "The algorithm must be LLMPPO, LLMREINFORCE, or GRPO for multi-turn "
            f"finetuning. Got {type(pop[0])} instead."
        ),
        checkpoint_steps=checkpoint_steps,
        algo="multiturn",
    )

    if init_hp is None:
        init_hp = {
            "BATCH_SIZE_PER_GPU": pop[0].batch_size_per_process,
            "ALGO": pop[0].algo,
        }

    batch_size = init_hp.get("BATCH_SIZE", pop[0].batch_size)

    data_increment = accelerator.num_processes if accelerator is not None else 1
    effective_data_batch_size = data_increment * batch_size
    env_name = init_hp.get("env_name", "multiturn")

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
        tensorboard_log_dir=tensorboard_log_dir,
        accelerator=accelerator,
        wandb_api_key=wandb_api_key,
        wandb_kwargs=wandb_kwargs,
        init_hyperparams=init_hp,
    )

    population = Population(
        agents=pop,
        accelerator=accelerator,
        loggers=loggers,
    )

    total_steps = 0
    i = 0
    next_checkpoint_step = checkpoint_steps
    max_steps_checkpoint_saved = False
    group_size = getattr(pop[0], "group_size", 1)
    rollout_env = SyncMultiTurnVecEnv(env_factory, batch_size, group_size, env_config)
    # ``agent.test`` expects a single ``TokenizedMultiTurnEnv``; ``rollout_env``
    # is a ``SyncMultiTurnVecEnv`` wrapping N inner envs whose state is
    # mid-rollout during training. Build a separate test env so evaluation is
    # isolated.
    # NOTE: this means one extra env is held for the run's lifetime. Future
    # refactor could share a subset of the rollout envs (e.g. lease one of the
    # vec env's inner envs when no trajectory is active) to avoid the
    # duplication for heavy env setups.
    test_env = env_factory(**(env_config or {}))
    group_seed = np.random.randint(0, 1_000_000)
    wall_deadline = (
        time.monotonic() + max_wall_seconds
        if max_wall_seconds is not None and max_wall_seconds > 0
        else None
    )
    while total_steps < max_steps:
        if wall_deadline is not None and time.monotonic() >= wall_deadline:
            if accelerator is None or accelerator.is_main_process:
                print(
                    f"\nStopping multiturn training: wall time limit ({max_wall_seconds}s) reached.",
                )
            break

        # Collect rollouts and learn
        iteration_steps = 0
        for agent in population.agents:
            agent.init_training_step()
            (
                completion_ids_list,
                action_masks_list,
                all_turn_ids,
                all_rewards,
                batch_steps,
                group_seed,
                all_sampling_logps,
            ) = collect_rollouts_llm(
                agent=agent,
                env=rollout_env,
                n_steps=max_turns,
                batch_size=batch_size,
                group_size=group_size,
                group_seed=group_seed,
            )

            normalized_rewards = [
                reward.unsqueeze(0) if reward.dim() == 1 else reward
                for reward in all_rewards
            ]
            (turn_ids_padded,) = stack_and_pad_experiences(
                all_turn_ids,
                padding_values=[-1],
            )
            (rewards_stacked,) = stack_and_pad_experiences(
                normalized_rewards,
                padding_values=[0.0],
            )
            rewards_2d = rewards_stacked.float()

            episode_scores = (
                rewards_2d.sum(dim=1) if rewards_2d.dim() > 1 else rewards_2d
            )
            mean_score = episode_scores.mean().to(agent.device)

            experiences = (
                completion_ids_list,
                action_masks_list,
                rewards_2d,
            )

            # Pass turn_ids to every multi-turn RL agent that accepts it
            # (LLMPPO/LLMREINFORCE, and the GRPO family — GRPO/CISPO/GSPO —
            # which use it for turn-level importance sampling + per-turn
            # group-relative advantages). Agents that don't need it (e.g.
            # token/sequence levels, GSPO) simply ignore it.
            learn_kwargs = (
                {"turn_ids": turn_ids_padded}
                if isinstance(agent, (LLMREINFORCE, LLMPPO, GRPO))
                else {}
            )
            if all_sampling_logps is not None and isinstance(
                agent, (GRPO, LLMPPO, LLMREINFORCE)
            ):
                learn_kwargs["sampling_logps"] = all_sampling_logps
            if accelerator is not None and needs_cross_rank_seq_padding(
                agent,
                world_size=accelerator.num_processes,
            ):
                completion_ids, action_masks, rewards_2d = (
                    align_completion_batch_shapes_across_ranks(
                        completion_ids_list,
                        action_masks_list,
                        rewards_2d,
                        pad_token_id=agent.pad_token_id,
                        accelerator=accelerator,
                    )
                )
                experiences = (completion_ids, action_masks, rewards_2d)
                # Keep turn_ids time dim aligned with padded action masks.
                target_mask_len = int(action_masks.shape[1])
                if (
                    "turn_ids" in learn_kwargs
                    and turn_ids_padded is not None
                    and int(turn_ids_padded.shape[1]) < target_mask_len
                ):
                    pad_t = target_mask_len - int(turn_ids_padded.shape[1])
                    learn_kwargs["turn_ids"] = torch.nn.functional.pad(
                        turn_ids_padded, (0, pad_t), value=-1
                    )
            agent.learn(experiences, **learn_kwargs)

            agg_score = safe_aggregate_metrics(accelerator, mean_score)

            if max_reward is not None:
                if "accuracy" not in agent.metrics.additional_metrics:
                    agent.metrics.register("accuracy")

                accuracy = (
                    (episode_scores >= max_reward).float().mean().to(agent.device)
                )
                agg_accuracy = safe_aggregate_metrics(accelerator, accuracy)

                if accelerator is None or accelerator.is_main_process:
                    agent.metrics.log("accuracy", agg_accuracy)

            effective_batch_steps = batch_steps * data_increment
            agent.finalize_training_step(batch_steps)
            total_steps += effective_batch_steps
            iteration_steps += effective_batch_steps

            if accelerator is None or accelerator.is_main_process:
                agent.add_scores([float(agg_score)])

            # Evaluate performance
            if (i + 1) % evaluation_interval == 0:
                agent.test(test_env)

        # Report training metrics
        if accelerator is None or accelerator.is_main_process:
            pbar.update(iteration_steps // len(population.agents))
        population.report_metrics(clear=True)

        if accelerator is not None:
            accelerator.wait_for_everyone()

        # Selection and mutation
        if selection_strategy is not None and mutation is not None:
            # `_validate_finetune_args` rejects an unset `evo_steps` here.
            assert evo_steps is not None
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
                    agent,
                    checkpoint_path if checkpoint_path is not None else elite_path,
                )

        i += 1

    if save_elite and elite_path is not None:
        elite = max(
            population.agents,
            key=lambda a: a.fitness[-1] if a.fitness else float("-inf"),
        )
        save_llm_checkpoint(elite, elite_path)

    population.finish()
    pbar.close()
    # LLM fitnesses are scalar mean rewards; `Population` types them as the wider
    # scalar-or-per-agent-dict row shared with multi-agent training.
    return population.agents, population.last_scalar_fitnesses
