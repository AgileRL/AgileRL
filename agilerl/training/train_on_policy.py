import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import wandb
from accelerate import Accelerator
from gymnasium import spaces
from tqdm import trange

from agilerl.algorithms import PPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import (
    init_wandb,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)

InitDictType = Optional[Dict[str, Any]]
OnPolicyAlgorithms = PPO
PopulationType = List[OnPolicyAlgorithms]


def train_on_policy(
    env: gym.Env,
    env_name: str,
    algo: str,
    pop: PopulationType,
    INIT_HP: InitDictType = None,
    MUT_P: InitDictType = None,
    swap_channels: bool = False,
    max_steps: int = 1000000,
    evo_steps: int = 10000,
    eval_steps: Optional[int] = None,
    eval_loop: int = 1,
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
    wandb_kwargs: Optional[Dict[str, Any]] = None,
    collect_rollouts_fn: Optional[
        Callable[[OnPolicyAlgorithms, gym.Env, int], None]
    ] = None,
) -> Tuple[PopulationType, List[List[float]]]:
    """The general on-policy RL training function. Returns trained population of agents
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

    # Detect if environment is vectorised
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

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None
    checkpoint_count = 0

    # Pre-training mutation
    if accelerator is None and mutation is not None:
        pop = mutation.mutation(pop, pre_training_mut=True)

    # Initialize list to store entropy values for each agent
    pop_entropy = [[] for _ in pop]

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        if accelerator is not None:
            accelerator.wait_for_everyone()
        pop_episode_scores = []
        pop_fps = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            agent.set_training_mode(True)

            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            start_time = time.time()

            active_collect = collect_rollouts_fn
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
                for _ in range(-(evo_steps // -agent.learn_step)):
                    active_collect(agent, env, n_steps=agent.learn_step)

                    buffer_size = agent.rollout_buffer.pos
                    rewards_np = (
                        agent.rollout_buffer.buffer["rewards"][:buffer_size]
                        .cpu()
                        .numpy()
                    )
                    dones_np = (
                        agent.rollout_buffer.buffer["dones"][:buffer_size].cpu().numpy()
                    )
                else:
                    from agilerl.rollouts import collect_rollouts as active_collect

                    for r_step, d_step in zip(rewards_np, dones_np):
                        scores += np.array(r_step)
                        finished = np.array(d_step, dtype=bool)
                        for idx, fin in enumerate(finished):
                            if fin:
                                completed_episode_scores.append(scores[idx])
                                agent.scores.append(scores[idx])
                                scores[idx] = 0

                    steps += buffer_size * num_envs
                    total_steps += buffer_size * num_envs
                    loss = agent.learn()
                    pop_loss[agent_idx].append(loss)

            else:
                obs, info = env.reset()
                for _ in range(-(evo_steps // -agent.learn_step)):
                    observations = []
                    actions = []
                    log_probs = []
                    rewards = []
                    dones = []
                    values = []

                    done = np.zeros(num_envs)
                    for idx_step in range(-(agent.learn_step // -num_envs)):
                        if swap_channels:
                            obs = obs_channels_to_first(obs)

                        action_mask = info.get("action_mask", None)
                        action, log_prob, entropy, value = agent.get_action(
                            obs, action_mask=action_mask
                        )

                        if not is_vectorised:
                            action = action[0]
                            log_prob = log_prob[0]
                            value = value[0]
                            entropy = (
                                entropy[0] if hasattr(entropy, "__len__") else entropy
                            )

                        pop_entropy[agent_idx].append(entropy)

                        if isinstance(agent.action_space, spaces.Box):
                            if agent.actor.squash_output:
                                clipped_action = agent.actor.scale_action(action)
                            else:
                                clipped_action = np.clip(
                                    action,
                                    agent.action_space.low,
                                    agent.action_space.high,
                                )
                        else:
                            clipped_action = action

                        next_obs, reward, term, trunc, info = env.step(clipped_action)
                        next_done = np.logical_or(term, trunc).astype(np.int8)

                        total_steps += num_envs
                        steps += num_envs

                        observations.append(obs)
                        actions.append(action)
                        log_probs.append(log_prob)
                        rewards.append(reward)
                        dones.append(done)
                        values.append(value)

                        obs = next_obs
                        done = next_done
                        scores += np.array(reward)

                        if not is_vectorised:
                            term = [term]
                            trunc = [trunc]

                        for idx, (d, t) in enumerate(zip(term, trunc)):
                            if d or t:
                                completed_episode_scores.append(scores[idx])
                                agent.scores.append(scores[idx])
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
                    loss = agent.learn(experiences)
                    pop_loss[agent_idx].append(loss)

            agent.steps[-1] += steps
            fps = steps / (time.time() - start_time)
            pop_fps.append(fps)
            pbar.update(evo_steps // len(pop))
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env, swap_channels=swap_channels, max_steps=eval_steps, loop=eval_loop
            )
            for agent in pop
        ]
        pop_fitnesses.append(fitnesses)
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        if wb:
            wandb_dict = {
                "global_step": (
                    total_steps * accelerator.state.num_processes
                    if accelerator is not None and accelerator.is_main_process
                    else total_steps
                ),
                "fps": np.mean(pop_fps),
                "train/mean_score": np.mean(
                    [
                        mean_score
                        for mean_score in mean_scores
                        if not isinstance(mean_score, str)
                    ]
                ),
                "eval/mean_fitness": np.mean(fitnesses),
                "eval/best_fitness": np.max(fitnesses),
            }

            agent_loss_dict = {
                f"train/agent_{index}_loss": np.mean(loss_[-10:])
                for index, loss_ in enumerate(pop_loss)
            }
            wandb_dict.update(agent_loss_dict)

            # Add entropy metrics to wandb
            agent_entropy_dict = {
                f"train/agent_{index}_entropy": (
                    np.mean(entropy_[-100:]) if len(entropy_) > 0 else 0
                )
                for index, entropy_ in enumerate(pop_entropy)
            }
            wandb_dict.update(agent_entropy_dict)

            # Add mean entropy across all agents
            all_entropies = [
                np.mean(entropy_[-100:]) if len(entropy_) > 0 else 0
                for entropy_ in pop_entropy
            ]
            if all_entropies:
                wandb_dict["train/mean_entropy"] = np.mean(all_entropies)

            if accelerator is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    wandb.log(wandb_dict)
                accelerator.wait_for_everyone()
            else:
                wandb.log(wandb_dict)

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
            fitness = ["%.2f" % fitness for fitness in fitnesses]
            avg_fitness = ["%.2f" % np.mean(agent.fitness[-5:]) for agent in pop]
            avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
            agents = [agent.index for agent in pop]
            num_steps = [agent.steps[-1] for agent in pop]
            muts = [agent.mut for agent in pop]
            pbar.update(0)

            print(
                f"""
                --- Global Steps {total_steps} ---
                Fitness:\t\t{fitness}
                Score:\t\t{mean_scores}
                5 fitness avgs:\t{avg_fitness}
                10 score avgs:\t{avg_score}
                Agents:\t\t{agents}
                Steps:\t\t{num_steps}
                Mutations:\t\t{muts}
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
