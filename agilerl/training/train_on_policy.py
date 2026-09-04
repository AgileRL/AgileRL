# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any

import gymnasium as gym
from accelerate import Accelerator

from agilerl.algorithms import PPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.protocols import SelectionStrategyProtocol
from agilerl.rollouts import collect_rollouts, collect_rollouts_recurrent
from agilerl.typing import InitHyperparams, RolloutReturn
from agilerl.utils.algo_utils import get_num_envs
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    resolve_selection_strategy,
    run_selection_and_mutation,
    save_population_checkpoint,
)
from agilerl.vector import DummyVecEnv

OnPolicyAlgorithms = PPO
PopulationType = list[OnPolicyAlgorithms]

# Rollout collectors take the agent and environment positionally and the rest of
# the rollout state by keyword.
CollectRolloutsFn = Callable[..., RolloutReturn]

logger = logging.getLogger(__name__)


def train_on_policy(
    env: gym.Env | gym.vector.VectorEnv,
    env_name: str,
    algo: str,
    pop: PopulationType,
    init_hp: InitHyperparams = None,
    mut_p: InitHyperparams = None,
    max_steps: int = 1000000,
    evo_steps: int = 10000,
    eval_steps: int | None = None,
    eval_loop: int = 1,
    target: float | None = None,
    selection_strategy: SelectionStrategyProtocol | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    checkpoint: int | None = None,
    checkpoint_path: str | None = None,
    overwrite_checkpoints: bool = False,
    save_elite: bool = False,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    collect_rollouts_fn: CollectRolloutsFn | None = None,
) -> tuple[PopulationType, list[float]]:
    """Run the general on-policy RL training; returns trained population of agents
    and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[SingleAgentAlgorithm]
    :param init_hp: Dictionary containing initial hyperparameters, defaults to None
    :type init_hp: dict, optional
    :param mut_p: Dictionary containing mutation parameters, defaults to None
    :type mut_p: dict, optional
    :param max_steps: Maximum number of steps in environment, defaults to 1000000
    :type max_steps: int, optional
    :param evo_steps: Evolution frequency (steps), defaults to 10000
    :type evo_steps: int, optional
    :param eval_steps: Number of evaluation steps per episode. If None, will evaluate until
        environment terminates or truncates. Defaults to None
    :type eval_steps: int, optional
    :param eval_loop: Number of evaluation episodes, defaults to 1
    :type eval_loop: int, optional
    :param target: Target score for early stopping, defaults to None
    :type target: float, optional
    :param selection_strategy: selection strategy driving population evolution. A
        :class:`~agilerl.hpo.tournament.TournamentSelection` or
        :class:`~agilerl.hpo.multi_frequency.MultiFrequencySelection` (MF-PBT) object,
        defaults to None
    :type selection_strategy: object, optional
    :param tournament: Deprecated alias for selection_strategy (a
        :class:`~agilerl.hpo.tournament.TournamentSelection` object), defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (steps), defaults to None
    :type checkpoint: int, optional
    :param checkpoint_path: Location to save checkpoint, defaults to None
    :type checkpoint_path: str, optional
    :param overwrite_checkpoints: Overwrite previous checkpoints during training, defaults to False
    :type overwrite_checkpoints: bool, optional
    :param save_elite: Boolean flag indicating whether to save elite member at the end
        of training, defaults to False
    :type save_elite: bool, optional
    :param elite_path: Location to save elite agent, defaults to None
    :type elite_path: str, optional
    :param wb: Weights & Biases tracking, defaults to False
    :type wb: bool, optional
    :param tensorboard: TensorBoard tracking, defaults to False
    :type tensorboard: bool, optional
    :param tensorboard_log_dir: Directory for TensorBoard logs, defaults to None
    :type tensorboard_log_dir: str, optional
    :param verbose: Display training stats, defaults to True
    :type verbose: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wandb_api_key: API key for Weights & Biases, defaults to None
    :type wandb_api_key: str, optional
    :param wandb_kwargs: Additional kwargs to pass to wandb.init()
    :type wand_kwargs: dict, optional
    :param collect_rollouts_fn: Optional function used to collect rollouts. If
        ``None`` and agents use a rollout buffer, a default function will be
        selected based on whether the agent is recurrent.
    :type collect_rollouts_fn: Callable or None, optional

    :return: Trained population of agents and their fitnesses
    :rtype: list[SingleAgentAlgorithm], list[float]
    """
    selection_strategy = resolve_selection_strategy(selection_strategy, tournament)
    assert isinstance(
        algo,
        str,
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    if target is not None:
        assert isinstance(
            target,
            (float, int),
        ), "Target score must be a float or an integer."
    if checkpoint is not None:
        assert isinstance(checkpoint, int), "Checkpoint must be an integer."
    assert isinstance(
        wb,
        bool,
    ), "'wb' must be a boolean flag, indicating whether to record run with W&B"
    assert isinstance(verbose, bool), "Verbose must be a boolean."
    if save_elite is False and elite_path is not None:
        warnings.warn(
            "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True.",
            stacklevel=2,
        )
    if checkpoint is None and checkpoint_path is not None:
        warnings.warn(
            "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined.",
            stacklevel=2,
        )

    # Ensure environment has vectorized interface. `DummyVecEnv` duck-types the
    # `VectorEnv` API rather than subclassing it, so the cast matches the annotations
    # of the rollout and evaluation helpers it is handed to.
    vec_env: gym.vector.VectorEnv = (
        env if isinstance(env, gym.vector.VectorEnv) else DummyVecEnv(env)
    )

    num_envs = get_num_envs(vec_env)
    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name,
            algo,
            datetime.now().strftime("%m%d%Y%H%M%S"),
        )
    )

    pbar = default_progress_bar(max_steps, accelerator)

    loggers = init_loggers(
        algo=algo,
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
        mutation_hyperparams=mut_p,
    )

    # Initialize population wrapper
    population = Population(
        agents=pop,
        accelerator=accelerator,
        loggers=loggers,
    )

    # Enable the per-neuron gradient capture that ReGraMa parameter mutations read.
    capture_grama = mutation is not None and mutation.parameters_mut > 0

    checkpoint_count = 0

    # Pre-training mutation
    if accelerator is None and mutation is not None:
        population.update(mutation.mutation(population.agents, pre_training_mut=True))

    # RL training loop
    active_collect = collect_rollouts_fn
    while population.all_below(max_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

        for agent in population.agents:
            agent.set_training_mode(True)
            agent.init_training_step(capture_grama)

            steps = 0
            completed_episode_scores: list[float] = []
            n_steps = -(agent.learn_step // -num_envs)
            if active_collect is None:
                if getattr(agent, "recurrent", False):
                    active_collect = collect_rollouts_recurrent
                else:
                    active_collect = collect_rollouts

            # Collect rollouts and learn until evo_steps is reached
            last_obs, last_done, last_scores, last_info = None, None, None, None
            for _ in range(-(evo_steps // -agent.learn_step)):
                # Collect rollouts and save in buffer
                episode_scores, last_obs, last_done, last_scores, last_info = (
                    active_collect(
                        agent,
                        vec_env,
                        n_steps=n_steps,
                        last_obs=last_obs,
                        last_done=last_done,
                        last_scores=last_scores,
                        last_info=last_info,
                    )
                )

                agent.learn()  # learn from the rollouts collected in the buffer

                # Update step counter and scores
                steps += n_steps * num_envs
                completed_episode_scores += episode_scores

            agent.add_scores(completed_episode_scores)
            agent.finalize_training_step(steps)
            pbar.update(steps // population.size)

        # Evaluate population
        for agent in population.agents:
            agent.test(
                vec_env,
                max_steps=eval_steps,
                loop=eval_loop,
            )

        # Report progress
        population.report_metrics(clear=True)
        population.increment_evo_step()

        # Check if we have met the target score
        if population.should_stop(target):
            logger.info("Target score has been reached. Stopping training.")
            population.finish()
            pbar.close()
            # Single-agent fitnesses are scalars; `Population` types them as the
            # wider scalar-or-per-agent-dict row shared with multi-agent training.
            return population.agents, population.last_scalar_fitnesses

        # Perform HPO
        if selection_strategy is not None and mutation is not None:
            population.update(
                run_selection_and_mutation(
                    selection_strategy,
                    population=population.agents,
                    mutation=mutation,
                    env_name=env_name,
                    algo=algo,
                    elite_path=elite_path,
                    save_elite=save_elite,
                    accelerator=accelerator,
                ),
            )

        # Save model checkpoint
        if checkpoint is not None:
            if population.local_step // checkpoint > checkpoint_count:
                save_population_checkpoint(
                    population=population.agents,
                    save_path=save_path,
                    overwrite_checkpoints=overwrite_checkpoints,
                    accelerator=accelerator,
                )
                checkpoint_count += 1

    population.finish()
    pbar.close()
    return population.agents, population.last_scalar_fitnesses
