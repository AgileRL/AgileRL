import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from accelerate import Accelerator
from pettingzoo import ParallelEnv
from torch.utils.data import DataLoader

from agilerl.algorithms import MADDPG, MATD3
from agilerl.components.data import MultiAgentTransition, ReplayDataset
from agilerl.components.replay_buffer import MultiAgentReplayBuffer
from agilerl.components.sampler import Sampler
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)
from agilerl.vector import PzDummyVecEnv
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

if TYPE_CHECKING:
    from tensordict import TensorDictBase

InitDictType = dict[str, Any] | None
PopulationType = list[MADDPG | MATD3]


def train_multi_agent_off_policy(
    env: ParallelEnv | AsyncPettingZooVecEnv,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: MultiAgentReplayBuffer,
    sum_scores: bool = True,
    init_hp: InitDictType = None,
    mut_p: InitDictType = None,
    max_steps: int = 50000,
    evo_steps: int = 25,
    eval_steps: int | None = None,
    eval_loop: int = 1,
    learning_delay: int = 0,
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
) -> tuple[PopulationType, list[float]]:
    """Run the general off-policy multi-agent RL training; returns trained
    population of agents and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[MADDPG | MATD3]
    :param memory: Experience Replay Buffer
    :type memory: MultiAgentReplayBuffer
    :param sum_scores: Boolean flag indicating whether to sum sub-agents scores,
        typically True for co-operative environments, defaults to True
    :type sum_scores: bool, optional
    :param init_hp: Dictionary containing initial hyperparameters.
    :type init_hp: dict
    :param mut_p: Dictionary containing mutation parameters, defaults to None
    :type mut_p: dict, optional
    :param max_steps: Maximum number of steps in environment, defaults to 50000
    :type max_steps: int, optional
    :param evo_steps: Evolution frequency (steps), defaults to 25
    :type evo_steps: int, optional
    :param eval_steps: Number of evaluation steps per episode. If None, will evaluate until
        environment terminates or truncates. Defaults to None
    :type eval_steps: int, optional
    :param eval_loop: Number of evaluation episodes, defaults to 1
    :type eval_loop: int, optional
    :param learning_delay: Steps in environment before starting learning, defaults to 0
    :type learning_delay: int, optional
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
    :param overwrite_checkpoints: Overwrite previous checkpoints during training,
        defaults to False
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
    :type wandb_kwargs: dict, optional

    :return: Trained population of agents and their fitnesses
    :rtype: tuple[list[MADDPG | MATD3], list[float]]
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

    # Ensure environment has vectorized interface
    if not hasattr(env, "num_envs"):
        env = PzDummyVecEnv(env)

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
        # Create dataloader from replay buffer
        replay_dataset = ReplayDataset(memory, pop[0].batch_size)
        replay_dataloader = DataLoader(replay_dataset, batch_size=None)
        replay_dataloader = accelerator.prepare(replay_dataloader)
        sampler = Sampler(dataset=replay_dataset, dataloader=replay_dataloader)
    else:
        sampler = Sampler(memory=memory)

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
        wandb_kwargs={"project": "AgileRLMultiAgent", **(wandb_kwargs or {})},
        init_hyperparams=init_hp,
        mutation_hyperparams=mut_p,
    )

    # Initialize population wrapper for metrics reporting
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
    while population.all_below(max_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

        for agent in population.agents:
            agent.set_training_mode(True)
            agent.init_evo_step()

            obs, info = env.reset()
            scores = (
                np.zeros((num_envs, 1))
                if sum_scores
                else np.zeros((num_envs, len(agent.agent_ids)))
            )
            completed_episode_scores: list[float] | list[list[float]] = []
            steps = 0

            for idx_step in range(evo_steps // num_envs):
                # Get next action from agent
                action, raw_action = agent.get_action(obs=obs, infos=info)

                # Act in environment
                next_obs, reward, termination, truncation, info = env.step(action)

                # Compute score increment (replace NaNs representing inactive agents with 0)
                agent_rewards = np.column_stack(
                    [np.asarray(v).ravel() for v in reward.values()]
                )
                agent_rewards = np.where(np.isnan(agent_rewards), 0, agent_rewards)
                score_increment = (
                    np.sum(agent_rewards, axis=-1)[:, np.newaxis]
                    if sum_scores
                    else agent_rewards
                )

                scores += score_increment
                steps += num_envs

                # Make a tensorclass out of the transition for easy adding to the replay buffer
                transition: TensorDictBase = MultiAgentTransition(
                    obs=obs,
                    action=raw_action,
                    reward=reward,
                    next_obs=next_obs,
                    done=termination,
                )
                transition = transition.to_tensordict()
                transition.batch_size = [num_envs]
                memory.add(transition)

                # Learn according to learning frequency
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        experiences = sampler.sample(agent.batch_size)
                        agent.learn(experiences)

                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        experiences = sampler.sample(agent.batch_size)
                        agent.learn(experiences)

                obs = next_obs

                reset_noise_indices = []

                # Find which agents are "done" - i.e. terminated or truncated
                dones = {}
                for agent_id in agent.agent_ids:
                    terminated = termination.get(agent_id, True)
                    truncated = truncation.get(agent_id, False)

                    # Replace NaNs with True (indicate killed agent)
                    terminated = np.where(
                        np.isnan(terminated),
                        True,
                        terminated,
                    ).astype(bool)
                    truncated = np.where(np.isnan(truncated), False, truncated).astype(
                        bool,
                    )

                    dones[agent_id] = terminated | truncated

                for idx, agent_dones in enumerate(zip(*dones.values(), strict=False)):
                    if all(agent_dones):
                        completed_score = (
                            np.asarray(scores[idx]).item()
                            if sum_scores
                            else list(scores[idx])
                        )
                        completed_episode_scores.append(completed_score)
                        scores[idx].fill(0)
                        reset_noise_indices.append(idx)

                agent.reset_action_noise(reset_noise_indices)

            agent.add_scores(completed_episode_scores)
            agent.finalize_evo_step(steps)
            pbar.update(evo_steps // population.size)

        population.increment_evo_step()

        # Evaluate population
        for agent in population.agents:
            agent.test(
                env,
                max_steps=eval_steps,
                loop=eval_loop,
                sum_scores=sum_scores,
            )

        population.report_metrics(clear=True)

        # Check if we have met the target score
        if population.should_stop(target):
            population.finish()
            pbar.close()
            return population.agents, population.last_fitnesses

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
    return population.agents, population.last_fitnesses
