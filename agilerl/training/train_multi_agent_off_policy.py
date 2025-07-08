import time
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import wandb
from accelerate import Accelerator
from pettingzoo import ParallelEnv
from torch.utils.data import DataLoader
from tqdm import trange

from agilerl.algorithms import MADDPG, MATD3
from agilerl.components.data import ReplayDataset
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.components.sampler import Sampler
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import (
    init_wandb,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

InitDictType = Optional[Dict[str, Any]]
PopulationType = List[MADDPG | MATD3]


def train_multi_agent_off_policy(
    env: ParallelEnv | AsyncPettingZooVecEnv,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: MultiAgentReplayBuffer,
    sum_scores: bool = True,
    INIT_HP: InitDictType = None,
    MUT_P: InitDictType = None,
    swap_channels: bool = False,
    max_steps: int = 50000,
    evo_steps: int = 25,
    eval_steps: Optional[int] = None,
    eval_loop: int = 1,
    learning_delay: int = 0,
    target: Optional[float] = None,
    tournament: Optional[TournamentSelection] = None,
    mutation: Optional[Mutations] = None,
    checkpoint: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    overwrite_checkpoints: bool = False,
    save_elite: bool = False,
    elite_path: Optional[str] = None,
    wb: bool = False,
    verbose: bool = True,
    accelerator: Optional[Accelerator] = None,
    wandb_api_key: Optional[str] = None,
) -> Tuple[PopulationType, List[List[float]]]:
    """The general off-policy multi-agent RL training function. Returns trained population of agents
    and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[object]
    :param memory: Experience Replay Buffer
    :type memory: object
    :param sum_scores: Boolean flag indicating whether to sum sub-agents scores, typically True for co-operative environments, defaults to True
    :type sum_scores: bool, optional
    :param INIT_HP: Dictionary containing initial hyperparameters.
    :type INIT_HP: dict
    :param MUT_P: Dictionary containing mutation parameters, defaults to None
    :type MUT_P: dict, optional
    :param swap_channels: Swap image channels dimension from last to first
        [H, W, C] -> [C, H, W], defaults to False
    :type swap_channels: bool, optional
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
    :param overwrite_checkpoints: Overwrite previous checkpoints during training, defaults to False
    :type overwrite_checkpoints: bool, optional
    :param save_elite: Boolean flag indicating whether to save elite member at the end
        of training, defaults to False
    :type save_elite: bool, optional
    :param elite_path: Location to save elite agent, defaults to None
    :type elite_path: str, optional
    :param wb: Weights & Biases tracking, defaults to False
    :type wb: bool, optional
    :param verbose: Display training stats, defaults to True
    :type verbose: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wandb_api_key: API key for Weights & Biases, defaults to None
    :type wandb_api_key: str, optional
    """
    assert isinstance(
        algo, str
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    if target is not None:
        assert isinstance(
            target, (float, int)
        ), "Target score must be a float or an integer."
    if checkpoint is not None:
        assert isinstance(checkpoint, int), "Checkpoint must be an integer."
    assert isinstance(
        wb, bool
    ), "'wb' must be a boolean flag, indicating whether to record run with W&B"
    assert isinstance(verbose, bool), "Verbose must be a boolean."
    if save_elite is False and elite_path is not None:
        warnings.warn(
            "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
        )
    if checkpoint is None and checkpoint_path is not None:
        warnings.warn(
            "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
        )

    start_time = time.time()
    if wb:
        init_wandb(
            algo=algo,
            env_name=env_name,
            init_hyperparams=INIT_HP,
            mutation_hyperparams=MUT_P,
            wandb_api_key=wandb_api_key,
            project="AgileRLMultiAgent",
            accelerator=accelerator,
        )

    if hasattr(env, "num_envs"):
        is_vectorised = True
        num_envs = env.num_envs
    else:
        is_vectorised = False
        num_envs = 1

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
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

    if accelerator is not None:
        print(f"\nDistributed training on {accelerator.device}...")
    else:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if accelerator is not None:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process,
        )
    else:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    agent_ids = deepcopy(env.agents)
    pop_actor_loss = [{agent_id: [] for agent_id in agent_ids} for _ in pop]
    pop_critic_loss = [{agent_id: [] for agent_id in agent_ids} for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None
    checkpoint_count = 0

    # Pre-training mutation
    if accelerator is None:
        if mutation is not None:
            pop = mutation.mutation(pop, pre_training_mut=True)

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        if accelerator is not None:
            accelerator.wait_for_everyone()

        pop_episode_scores = []
        pop_fps = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            agent.set_training_mode(True)

            # Reset environment at start of episode
            obs, info = env.reset()
            scores = (
                np.zeros((num_envs, 1))
                if sum_scores
                else np.zeros((num_envs, len(agent_ids)))
            )
            losses = {agent_id: [] for agent_id in agent_ids}
            completed_episode_scores = []
            steps = 0

            if swap_channels:
                expand_dims = not is_vectorised
                obs = {
                    agent_id: obs_channels_to_first(s, expand_dims=expand_dims)
                    for agent_id, s in obs.items()
                }

            start_time = time.time()
            for idx_step in range(evo_steps // num_envs):
                # Get next action from agent
                action, raw_action = agent.get_action(obs=obs, infos=info)

                if not is_vectorised:
                    action = {agent: act[0] for agent, act in action.items()}

                # Act in environment
                next_obs, reward, termination, truncation, info = env.step(action)

                # Compute score increment (replace NaNs representing inactive agents with 0)
                agent_rewards = np.array(list(reward.values())).transpose()
                agent_rewards = np.where(np.isnan(agent_rewards), 0, agent_rewards)
                score_increment = (
                    (
                        np.sum(agent_rewards, axis=-1)[:, np.newaxis]
                        if is_vectorised
                        else np.sum(agent_rewards, axis=-1)
                    )
                    if sum_scores
                    else agent_rewards
                )

                scores += score_increment
                total_steps += num_envs
                steps += num_envs

                # Save experience to replay buffer
                if swap_channels:
                    if not is_vectorised:
                        obs = {agent_id: np.squeeze(s) for agent_id, s in obs.items()}

                    next_obs = {
                        agent_id: obs_channels_to_first(ns)
                        for agent_id, ns in next_obs.items()
                    }

                memory.save_to_memory(
                    obs,
                    raw_action,
                    reward,
                    next_obs,
                    termination,
                    is_vectorised=is_vectorised,
                )

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        # Sample replay buffer
                        experiences = sampler.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        loss = agent.learn(experiences)
                        for agent_id in agent_ids:
                            losses[agent_id].append(loss[agent_id])

                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        experiences = sampler.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        loss = agent.learn(experiences)
                        for agent_id in agent_ids:
                            losses[agent_id].append(loss[agent_id])

                # Update the state
                if swap_channels and not is_vectorised:
                    next_obs = {
                        agent_id: np.expand_dims(ns, 0)
                        for agent_id, ns in next_obs.items()
                    }

                obs = next_obs

                reset_noise_indices = []

                # Find which agents are "done" - i.e. terminated or truncated
                dones = {}
                for agent_id in agent.agent_ids:
                    terminated = termination.get(agent_id, True)
                    truncated = truncation.get(agent_id, False)

                    # Replace NaNs with True (indicate killed agent)
                    terminated = np.where(
                        np.isnan(terminated), True, terminated
                    ).astype(bool)
                    truncated = np.where(np.isnan(truncated), False, truncated).astype(
                        bool
                    )

                    dones[agent_id] = terminated | truncated

                if not is_vectorised:
                    dones = {
                        agent: np.array([dones[agent_id]]) for agent in agent.agent_ids
                    }

                for idx, agent_dones in enumerate(zip(*dones.values())):
                    if all(agent_dones):
                        completed_score = (
                            float(scores[idx]) if sum_scores else list(scores[idx])
                        )
                        completed_episode_scores.append(completed_score)
                        agent.scores.append(completed_score)
                        scores[idx].fill(0)
                        reset_noise_indices.append(idx)
                        if not is_vectorised:
                            obs, info = env.reset()

                            if swap_channels:
                                expand_dims = not is_vectorised
                                obs = {
                                    agent_id: obs_channels_to_first(
                                        s, expand_dims=expand_dims
                                    )
                                    for agent_id, s in obs.items()
                                }

                agent.reset_action_noise(reset_noise_indices)

            pbar.update(evo_steps // len(pop))

            agent.steps[-1] += steps
            fps = steps / (time.time() - start_time)
            pop_fps.append(fps)
            pop_episode_scores.append(completed_episode_scores)
            if len(losses[agent_ids[0]]) > 0:
                if all([losses[a_id] for a_id in agent_ids]):
                    for agent_id in agent_ids:
                        actor_losses, critic_losses = list(zip(*losses[agent_id]))
                        actor_losses = [
                            loss for loss in actor_losses if loss is not None
                        ]
                        if actor_losses:
                            pop_actor_loss[agent_idx][agent_id].append(
                                np.mean(actor_losses)
                            )
                        pop_critic_loss[agent_idx][agent_id].append(
                            np.mean(critic_losses)
                        )

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=swap_channels,
                max_steps=eval_steps,
                loop=eval_loop,
                sum_scores=sum_scores,
            )
            for agent in pop
        ]
        pop_fitnesses.append(fitnesses)
        if sum_scores:
            mean_scores = [
                (
                    np.mean(episode_scores)
                    if len(episode_scores) > 0
                    else "0 completed episodes"
                )
                for episode_scores in pop_episode_scores
            ]
            mean_score_dict = {
                "train/mean_score": np.mean(
                    [
                        mean_score
                        for mean_score in mean_scores
                        if not isinstance(mean_score, str)
                    ]
                )
            }
            fitness_dict = {
                "eval/mean_fitness": np.mean(fitnesses),
                "eval/best_fitness": np.max(fitnesses),
            }
        else:
            pop_mean_scores = [
                np.mean(np.array(score), axis=0)
                for score in pop_episode_scores
                if score
            ]
            if pop_episode_scores:
                mean_scores = np.stack(pop_mean_scores, axis=0)
                mean_score_dict = {
                    "train/mean_score/" + agent: np.mean(mean_scores[:, idx], axis=-1)
                    for idx, agent in enumerate(agent_ids)
                }
            else:
                mean_score_dict = {
                    "train/mean_score/" + agent: np.nan
                    for idx, agent in enumerate(agent_ids)
                }

            mean_fitnesses = np.mean(fitnesses, axis=0)
            max_fitnesses = np.max(fitnesses, axis=0)
            fitness_dict = {
                "eval/mean_fitness/" + agent: mean_fitnesses[idx]
                for idx, agent in enumerate(agent_ids)
            }
            best_fitness_dict = {
                "eval/best_fitness/" + agent: max_fitnesses[idx]
                for idx, agent in enumerate(agent_ids)
            }
            fitness_dict.update(best_fitness_dict)

        if wb:
            wandb_dict = {
                "global_step": (
                    total_steps * accelerator.state.num_processes
                    if accelerator is not None and accelerator.is_main_process
                    else total_steps
                ),
                "fps": np.mean(pop_fps),
            }
            wandb_dict.update(fitness_dict)
            wandb_dict.update(mean_score_dict)

            actor_loss_dict = {}
            critic_loss_dict = {}

            for agent_idx, agent in enumerate(pop):
                for agent_id, actor_loss, critic_loss in zip(
                    pop_actor_loss[agent_idx].keys(),
                    pop_actor_loss[agent_idx].values(),
                    pop_critic_loss[agent_idx].values(),
                ):
                    if actor_loss:
                        actor_loss_dict[
                            f"train/agent_{agent_idx}_{agent_id}_actor_loss"
                        ] = np.mean(actor_loss[-10:])
                        wandb_dict.update(actor_loss_dict)
                    if critic_loss:
                        critic_loss_dict[
                            f"train/agent_{agent_idx}_{agent_id}_critic_loss"
                        ] = np.mean(critic_loss[-10:])
                        wandb_dict.update(critic_loss_dict)

            if accelerator is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    wandb.log(wandb_dict)
                accelerator.wait_for_everyone()
            else:
                wandb.log(wandb_dict)

            for idx, agent in enumerate(pop):
                wandb.log(
                    {
                        f"learn_step_agent_{idx}": agent.learn_step,
                        f"learning_rate_actor_agent_{idx}": agent.lr_actor,
                        f"learning_rate_critic_agent_{idx}": agent.lr_critic,
                        f"batch_size_agent_{idx}": agent.batch_size,
                        f"indi_fitness_agent_{idx}": agent.fitness[-1],
                    }
                )
        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

        # Early stop if consistently reaches target
        if target is not None:
            if (
                np.all(
                    np.greater([np.mean(agent.fitness[-10:]) for agent in pop], target)
                )
                and len(pop[0].steps) >= 100
            ):
                if wb:
                    wandb.finish()
                return pop, pop_fitnesses

        # Tournament selection and population mutation
        if tournament and mutation is not None:
            pop = tournament_selection_and_mutation(
                population=pop,
                tournament=tournament,
                mutation=mutation,
                env_name=env_name,
                algo=algo,
                elite_path=elite_path,
                save_elite=save_elite,
                accelerator=accelerator,
            )

        if verbose:
            if sum_scores:
                fitness = ["%.2f" % fitness for fitness in fitnesses]
                avg_fitness = ["%.2f" % np.mean(agent.fitness[-5:]) for agent in pop]
                avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
            else:
                fitness_arr = np.array([fitness for fitness in fitnesses])
                avg_fitness_arr = np.array(
                    [np.mean(agent.fitness[-5:], axis=0) for agent in pop]
                )
                avg_score_arr = np.array(
                    [np.mean(agent.scores[-10:], axis=0) for agent in pop]
                )
                fitness = {
                    agent: fitness_arr[:, idx] for idx, agent in enumerate(agent_ids)
                }
                avg_fitness = {
                    agent: avg_fitness_arr[:, idx]
                    for idx, agent in enumerate(agent_ids)
                }
                avg_score = {
                    agent: avg_score_arr[:, idx] for idx, agent in enumerate(agent_ids)
                }

            agents = [agent.index for agent in pop]
            num_steps = [agent.steps[-1] for agent in pop]
            muts = [agent.mut for agent in pop]
            pbar.update(0)

            print()
            print(
                "DateTime, now, H:m:s-u",
                datetime.now().hour,
                ":",
                datetime.now().minute,
                ":",
                datetime.now().second,
                "-",
                datetime.now().microsecond,
            )
            total_time = time.time() - start_time
            print(
                "Steps",
                total_steps / total_time,
                "per sec,",
                total_steps / (total_time / 60),
                "per min.",
            )
            print(
                f"""
                --- Global Steps {total_steps} ---
                Fitness:\t{fitness}
                Score:\t\t{mean_scores}
                5 fitness avgs:\t{avg_fitness}
                10 score avgs:\t{avg_score}
                Agents:\t\t{agents}
                Steps:\t\t{num_steps}
                Mutations:\t{muts}
                """,
                end="\r",
            )

        # Save model checkpoint
        if checkpoint is not None:
            if pop[0].steps[-1] // checkpoint > checkpoint_count:
                save_population_checkpoint(
                    population=pop,
                    save_path=save_path,
                    overwrite_checkpoints=overwrite_checkpoints,
                    accelerator=accelerator,
                )
                checkpoint_count += 1

    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.wait_for_everyone()
        else:
            wandb.finish()

    pbar.close()
    return pop, pop_fitnesses
