import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
from accelerate import Accelerator
from gymnasium import spaces

from agilerl.algorithms import PPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.logger import StdOutLogger, TensorboardLogger, WandbLogger
from agilerl.networks import StochasticActor
from agilerl.population import Population
from agilerl.typing import GymEnvType
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import (
    default_progress_bar,
    init_wandb,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)
from agilerl.vector import DummyVecEnv

InitDictType = dict[str, Any] | None
OnPolicyAlgorithms = PPO
PopulationType = list[OnPolicyAlgorithms]


def train_on_policy(
    env: GymEnvType,
    env_name: str,
    algo: str,
    pop: PopulationType,
    INIT_HP: InitDictType = None,
    MUT_P: InitDictType = None,
    swap_channels: bool = False,
    max_steps: int = 1000000,
    evo_steps: int = 10000,
    eval_steps: int | None = None,
    eval_loop: int = 1,
    target: float | None = None,
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
    collect_rollouts_fn: (
        Callable[[OnPolicyAlgorithms, GymEnvType, int], None] | None
    ) = None,
) -> tuple[PopulationType, list[list[float]]]:
    """Run the general on-policy RL training; returns trained population of agents
    and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[RLAlgorithm]
    :param INIT_HP: Dictionary containing initial hyperparameters, defaults to None
    :type INIT_HP: dict, optional
    :param MUT_P: Dictionary containing mutation parameters, defaults to None
    :type MUT_P: dict, optional
    :param swap_channels: Swap image channels dimension from last to first
        [H, W, C] -> [C, H, W], defaults to False
    :type swap_channels: bool, optional
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
    :param tournament: Tournament selection object, defaults to None
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
    :rtype: list[RLAlgorithm], list[list[float]]
    """
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

    if wb:
        init_wandb_kwargs = {
            "algo": algo,
            "env_name": env_name,
            "init_hyperparams": INIT_HP,
            "mutation_hyperparams": MUT_P,
            "wandb_api_key": wandb_api_key,
            "accelerator": accelerator,
        }
        if wandb_kwargs is not None:
            init_wandb_kwargs.update(wandb_kwargs)

        init_wandb(**init_wandb_kwargs)

    if not hasattr(env, "num_envs"):
        env = DummyVecEnv(env)

    num_envs = env.num_envs

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name,
            algo,
            datetime.now().strftime("%m%d%Y%H%M%S"),
        )
    )

    if accelerator is not None:
        print(f"\nDistributed training on {accelerator.device}...")
    else:
        print("\nTraining...")

    pbar = default_progress_bar(max_steps, accelerator)

    # Define logger configuration
    loggers = [StdOutLogger(pbar)]
    if wb:
        loggers.append(WandbLogger(accelerator))
    if tensorboard:
        loggers.append(
            TensorboardLogger(log_dir=tensorboard_log_dir, accelerator=accelerator)
        )

    # Initialize population wrapper
    population = Population(
        agents=pop,
        accelerator=accelerator,
        loggers=loggers,
    )

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
            agent.init_evo_step()

            steps = 0
            completed_episode_scores: list[float] = []
            n_steps = -(agent.learn_step // -num_envs)
            if active_collect is None and getattr(agent, "use_rollout_buffer", False):
                if getattr(agent, "recurrent", False):
                    from agilerl.rollouts import (
                        collect_rollouts_recurrent as active_collect,
                    )
                else:
                    from agilerl.rollouts import collect_rollouts as active_collect

            if (
                getattr(agent, "use_rollout_buffer", False)
                and active_collect is not None
            ):
                last_obs, last_done, last_scores, last_info = None, None, None, None
                for _ in range(-(evo_steps // -agent.learn_step)):
                    # Collect rollouts and save in buffer
                    episode_scores, last_obs, last_done, last_scores, last_info = (
                        active_collect(
                            agent,
                            env,
                            n_steps=n_steps,
                            last_obs=last_obs,
                            last_done=last_done,
                            last_scores=last_scores,
                            last_info=last_info,
                        )
                    )

                    agent.learn()

                    # Update step counter and scores
                    steps += n_steps * num_envs
                    completed_episode_scores += episode_scores

            # Collect rollouts explicitly without saving to rollout buffer
            else:
                obs, info = env.reset()
                scores = np.zeros(num_envs)
                for _ in range(-(evo_steps // -agent.learn_step)):
                    observations = []
                    actions = []
                    log_probs = []
                    rewards = []
                    dones = []
                    values = []

                    done = np.zeros(num_envs)
                    for _ in range(-(agent.learn_step // -num_envs)):
                        if swap_channels:
                            obs = obs_channels_to_first(obs)

                        action_mask = info.get("action_mask", None)
                        action, log_prob, _entropy, value = agent.get_action(
                            obs,
                            action_mask=action_mask,
                        )

                        # Clip action to action space
                        policy = getattr(agent, agent.registry.policy())
                        if isinstance(policy, StochasticActor) and isinstance(
                            agent.action_space,
                            spaces.Box,
                        ):
                            if policy.squash_output:
                                clipped_action = policy.scale_action(action)
                            else:
                                clipped_action = np.clip(
                                    action,
                                    agent.action_space.low,
                                    agent.action_space.high,
                                )
                        else:
                            clipped_action = action

                        next_obs, reward, term, trunc, info = env.step(clipped_action)

                        # Check if termination condition is met
                        if isinstance(term, (list, np.ndarray)):
                            next_done = (
                                np.logical_or(term, trunc).astype(np.int8)
                                if isinstance(trunc, (list, np.ndarray))
                                else term
                            )
                        else:
                            next_done = term or trunc

                        reward_np = np.atleast_1d(reward)
                        next_done_np = np.atleast_1d(next_done)
                        value_np = np.atleast_1d(value)
                        log_prob_np = np.atleast_1d(log_prob)

                        observations.append(obs)
                        actions.append(action)
                        log_probs.append(log_prob_np)
                        rewards.append(reward_np)
                        dones.append(done)
                        values.append(value_np)

                        obs = next_obs
                        done = next_done

                        scores += reward_np
                        steps += num_envs

                        for idx, env_done in enumerate(next_done_np):
                            if env_done:
                                completed_episode_scores.append(scores[idx])
                                scores[idx] = 0

                    if swap_channels:
                        next_obs = obs_channels_to_first(next_obs)

                    experiences = (
                        observations,
                        actions,
                        log_probs,
                        rewards,
                        dones,
                        values,
                        next_obs,
                        next_done,
                    )
                    agent.learn(experiences)

            agent.add_scores(completed_episode_scores)
            agent.finalize_evo_step(steps)
            pbar.update(steps // population.size)

        population.increment_evo_step()

        # Evaluate population
        for agent in population.agents:
            agent.test(
                env,
                swap_channels=swap_channels,
                max_steps=eval_steps,
                loop=eval_loop,
            )

        # Aggregate metrics from all agents and log
        population.report_metrics()

        if population.should_stop(target):
            population.finish()
            pbar.close()
            return population.agents, population.last_fitnesses

        population.clear_agent_metrics()

        # Tournament selection and population mutation
        if tournament and mutation is not None:
            population.update(
                tournament_selection_and_mutation(
                    population=population.agents,
                    tournament=tournament,
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
            if population.agents[0].metrics.steps // checkpoint > checkpoint_count:
                save_population_checkpoint(
                    population=population.agents,
                    save_path=save_path,
                    overwrite_checkpoints=overwrite_checkpoints,
                    accelerator=accelerator,
                )
                checkpoint_count += 1

    population.finish()
    pbar.close()
    return population.agents, population.last_fitnesses
