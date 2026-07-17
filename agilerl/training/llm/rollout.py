"""Generative rollout training for LLM finetuning.

:func:`train_llm_rollout` drives on-policy RL (GRPO / PPO / REINFORCE) over a
:class:`~agilerl.llm_envs.RolloutEnv`; single-turn reasoning
(``max_turns=1``) and multi-turn share this one loop.
"""

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from accelerate import Accelerator

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import GRPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.training.llm.common import _validate_finetune_args
from agilerl.utils.llm_utils import safe_aggregate_metrics
from agilerl.utils.utils import (
    _distributed_rank,
    _distributed_world_size,
    default_progress_bar,
    init_loggers,
    save_llm_checkpoint,
    tournament_selection_and_mutation,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.algorithms import LLMPPO, LLMREINFORCE
    from agilerl.llm_envs import BatchRolloutEnv, RolloutEnv
    from agilerl.rollouts.on_policy import collect_rollouts_llm
    from agilerl.utils.algo_utils import stack_and_pad_experiences

if TYPE_CHECKING:
    SupportedRollout = GRPO | LLMPPO | LLMREINFORCE


def train_llm_rollout(
    pop: "list[SupportedRollout]",
    max_turns: int,
    env_factory: "Callable[[], RolloutEnv]",
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
) -> "tuple[list[SupportedRollout], Any]":
    """Train a population of LLM agents over rollout (generate-and-score) environments.

    Collects token-level episodes (``reset`` returns ``(obs, info)``, repeated
    ``get_action`` / ``step`` (full completion tensor), then ``get_episode_data``),
    then runs turn-level updates. For a ``RolloutEnv`` with ``max_model_len`` set,
    a trajectory whose cumulative prompt would overflow the context is stopped
    with ``truncated=True``.

    :param pop: Population of LLMPPO, LLMREINFORCE or GRPO agents to finetune.
    :type pop: list[SupportedRollout]
    :param max_turns: Maximum interaction turns per episode.
    :type max_turns: int
    :param env_factory: Zero-arg factory that returns a fresh env for each
        trajectory rollout. Required to ensure trajectory state isolation.
    :type env_factory: Callable[[], RolloutEnv]
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
    :param evo_steps: Steps between evolution (requires tournament and mutation).
    :type evo_steps: int, optional
    :param checkpoint_steps: Save checkpoint every N outer iterations when no evolution.
    :type checkpoint_steps: int, optional
    :param checkpoint_path: Directory for periodic checkpoints; falls back to elite_path.
    :type checkpoint_path: str, optional
    :param tournament: Tournament selection for evolution, defaults to None.
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
    :return: The finetuned population and its last recorded fitnesses.
    :rtype: tuple[list[SupportedRollout], Any]
    """
    _validate_finetune_args(
        evo_steps,
        tournament,
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
        algo="rollout",
    )

    if init_hp is None:
        init_hp = {
            "BATCH_SIZE_PER_GPU": pop[0].batch_size_per_process,
            "ALGO": pop[0].algo,
        }

    batch_size = init_hp.get("BATCH_SIZE", pop[0].batch_size)
    env_name = init_hp.get("env_name", "rollout")
    data_increment = _distributed_world_size(accelerator)
    effective_data_batch_size = data_increment * batch_size

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = batch_size
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path
        init_hp["max_turns"] = max_turns

    pbar = default_progress_bar(max_steps, accelerator)

    loggers = init_loggers(
        algo=init_hp.get("ALGO", "LLMPPO"),
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
    group_size = getattr(pop[0], "group_size", 1)
    # Derive rollout seeding from the agent's configured seed so dataset-row
    # order and env tasks are reproducible from the manifest, folding the rank
    # in so data-parallel ranks draw decorrelated rows and tasks. The
    # ``1 << 31`` offset is arbitrary but large enough that the rank's
    # contribution decorrelates the RNG streams even for a small base seed.
    group_seed = int(pop[0].seed) + _distributed_rank(accelerator) * (1 << 31)
    rollout_env = BatchRolloutEnv(env_factory, batch_size, group_size)
    # ``agent.test`` expects a single ``RolloutEnv``; ``rollout_env`` is a
    # ``BatchRolloutEnv`` wrapping N inner envs whose state is mid-rollout during
    # training. A separate test env keeps evaluation isolated; it is built lazily
    # on the first evaluation, so runs that never reach ``evaluation_interval``
    # don't pay for an extra env (or its server).
    test_env: RolloutEnv | None = None
    try:
        wall_deadline = (
            time.monotonic() + max_wall_seconds
            if max_wall_seconds is not None and max_wall_seconds > 0
            else None
        )
        while total_steps < max_steps:
            if wall_deadline is not None and time.monotonic() >= wall_deadline:
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
                agent.set_reference_policy(rollout_env.num_epochs)
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

                # Normalize rewards to 2D [1, n_turns] per trajectory so padding
                # stacks into [batch, max_turns] rather than flattening to 1D.
                normalized_rewards = [
                    reward.unsqueeze(0) if reward.dim() == 1 else reward
                    for reward in all_rewards
                ]
                (turn_ids_padded,) = stack_and_pad_experiences(
                    all_turn_ids,
                    padding_values=[-1],
                )
                (rewards_2d,) = stack_and_pad_experiences(
                    normalized_rewards,
                    padding_values=[0.0],
                )
                rewards_2d = rewards_2d.float()
                episode_scores = (
                    rewards_2d.sum(dim=1) if rewards_2d.dim() > 1 else rewards_2d
                )
                mean_score = episode_scores.mean().to(agent.device)

                experiences = (
                    completion_ids_list,
                    action_masks_list,
                    rewards_2d,
                )

                agent.learn(
                    experiences,
                    turn_ids=turn_ids_padded,
                    sampling_logps=all_sampling_logps,
                )

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

                if (i + 1) % evaluation_interval == 0:
                    if test_env is None:
                        test_env = env_factory()
                    agent.test(test_env, loop=eval_loop)
                    if accelerator is not None:
                        accelerator.wait_for_everyone()

            if accelerator is None or accelerator.is_main_process:
                pbar.update(iteration_steps // len(population.agents))
                population.report_metrics(clear=True)

            if accelerator is not None:
                accelerator.wait_for_everyone()

            if tournament is not None and mutation is not None:
                if (i + 1) % evo_steps == 0:
                    if accelerator is not None:
                        accelerator.wait_for_everyone()
                    population.update(
                        tournament_selection_and_mutation(
                            population=population.agents,
                            tournament=tournament,
                            mutation=mutation,
                            env_name=env_name,
                            accelerator=accelerator,
                            language_model=True,
                            elite_path=elite_path,
                            save_elite=save_elite,
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
                if total_steps >= max_steps and not max_steps_checkpoint_saved:
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
        rollout_env.close()
        if test_env is not None:
            test_env.close()

    population.finish()
    pbar.close()
    return population.agents, population.last_fitnesses
