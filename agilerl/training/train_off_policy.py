import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import wandb
from accelerate import Accelerator
from tensordict import TensorDictBase
from torch.utils.data import DataLoader
from tqdm import trange

from agilerl.algorithms import DDPG, DQN, TD3, RainbowDQN
from agilerl.algorithms.core.base import RLAlgorithm
from agilerl.components import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.data import ReplayDataset, Transition
from agilerl.components.sampler import Sampler
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import (
    init_wandb,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)

InitDictType = Optional[Dict[str, Any]]
PopulationType = List[RLAlgorithm]
BufferType = Union[ReplayBuffer, PrioritizedReplayBuffer, MultiStepReplayBuffer]


def train_off_policy(
    env: gym.Env,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: BufferType,
    INIT_HP: InitDictType = None,
    MUT_P: InitDictType = None,
    swap_channels: bool = False,
    max_steps: int = 1000000,
    evo_steps: int = 10000,
    eval_steps: Optional[int] = None,
    eval_loop: int = 1,
    learning_delay: int = 0,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_decay: float = 0.995,
    target: Optional[float] = None,
    n_step: bool = False,
    per: bool = False,
    n_step_memory: Optional[MultiStepReplayBuffer] = None,
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
) -> Tuple[PopulationType, List[List[float]]]:
    """The general online RL training function. Returns trained population of agents
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
    :param learning_delay: Steps in environment before starting learning, defaults to 0
    :type learning_delay: int, optional
    :param eps_start: Maximum exploration - initial epsilon value, defaults to 1.0
    :type eps_start: float, optional
    :param eps_end: Minimum exploration - final epsilon value, defaults to 0.1
    :type eps_end: float, optional
    :param eps_decay: Epsilon decay per episode, defaults to 0.995
    :type eps_decay: float, optional
    :param target: Target score for early stopping, defaults to None
    :type target: float, optional
    :param n_step: Use multi-step experience replay buffer, defaults to False
    :type n_step: bool, optional
    :param per: Using prioritized experience replay buffer, defaults to False
    :type per: bool, optional
    :param memory: Multi-step Experience Replay Buffer to be used alongside Prioritized
        ERB, defaults to None
    :type memory: object, optional
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
    :param wandb_kwargs: Additional kwargs to pass to wandb.init()
    :type wand_kwargs: dict, optional
    """
    assert isinstance(
        algo, str
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    assert isinstance(eps_start, float), "Starting epsilon must be a float."
    assert isinstance(eps_end, float), "Final value of epsilone must be a float."
    assert isinstance(eps_decay, float), "Epsilon decay rate must be a float."
    if target is not None:
        assert isinstance(
            target, (float, int)
        ), "Target score must be a float or an integer."
    assert isinstance(n_step, bool), "'n_step' must be a boolean."
    assert isinstance(per, bool), "'per' must be a boolean."
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
        num_envs = env.num_envs
        is_vectorised = True
    else:
        num_envs = 1
        is_vectorised = False

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
        if n_step_memory is not None:
            n_step_sampler = Sampler(memory=n_step_memory)

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
            disable=not accelerator.is_local_main_process,
        )
    else:
        pbar = trange(max_steps, unit="step", bar_format=bar_format, ascii=True)

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None
    checkpoint_count = 0

    # Pre-training mutation
    if accelerator is None and mutation is not None:
        pop = mutation.mutation(pop, pre_training_mut=True)

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        if accelerator is not None:
            accelerator.wait_for_everyone()

        pop_episode_scores = []
        pop_fps = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores, losses = [], []
            steps = 0

            if isinstance(agent, (DQN, RainbowDQN)):
                train_actions_hist = [0] * agent.action_dim

            if isinstance(agent, DQN):
                epsilon = eps_start

            start_time = time.time()
            for idx_step in range(evo_steps // num_envs):
                if swap_channels:
                    state = obs_channels_to_first(state)

                # Get next action from agent
                if isinstance(agent, DQN):
                    action_mask = info.get("action_mask", None)
                    action = agent.get_action(state, epsilon, action_mask=action_mask)
                    # Decay epsilon for exploration
                    epsilon = max(eps_end, epsilon * eps_decay)
                elif isinstance(agent, RainbowDQN):
                    action_mask = info.get("action_mask", None)
                    action = agent.get_action(state, action_mask=action_mask)
                else:
                    action = agent.get_action(state)

                if isinstance(agent, (DQN, RainbowDQN)):
                    for a in action:
                        if not isinstance(a, int):
                            a = int(a)
                        train_actions_hist[a] += 1

                if not is_vectorised:
                    action = action[0]

                # Act in environment
                next_state, reward, done, trunc, info = env.step(action)
                scores += np.array(reward)

                if not is_vectorised:
                    done = np.array([done])
                    trunc = np.array([trunc])

                reset_noise_indices = []
                for idx, (d, t) in enumerate(zip(done, trunc)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                if isinstance(agent, (DDPG, TD3)):
                    agent.reset_action_noise(reset_noise_indices)

                total_steps += num_envs
                steps += num_envs

                # Save experience to replay buffer
                next_state = (
                    obs_channels_to_first(next_state) if swap_channels else next_state
                )

                transition: TensorDictBase = Transition(
                    obs=state,
                    action=action,
                    reward=reward,
                    next_obs=next_state,
                    done=done,
                )
                if not is_vectorised:
                    transition = transition.unsqueeze(0)

                # Add transition to replay buffer
                transition = transition.to_tensordict()
                transition.batch_size = [num_envs]
                if n_step_memory is not None:
                    one_step_transition = n_step_memory.add(transition)
                    if one_step_transition is not None:
                        memory.add(one_step_transition)
                else:
                    memory.add(transition)
                if per:
                    fraction = min(
                        ((agent.steps[-1] + idx_step + 1) * num_envs / max_steps), 1.0
                    )
                    agent.beta += fraction * (1.0 - agent.beta)

                # Learn according to learning frequency
                # Handle learn_step > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.size > learning_delay
                    ):
                        if per:
                            experiences = sampler.sample(agent.batch_size, agent.beta)
                            if n_step_memory is not None:
                                n_step_experiences = n_step_sampler.sample(
                                    experiences["idxs"]
                                )
                            else:
                                n_step_experiences = None

                            loss, idxs, priorities = agent.learn(
                                experiences, n_experiences=n_step_experiences, per=per
                            )
                            memory.update_priorities(idxs, priorities)
                        else:
                            experiences = sampler.sample(
                                agent.batch_size,
                                return_idx=True if n_step_memory is not None else False,
                            )
                            if n_step_memory is not None:
                                n_step_experiences = n_step_sampler.sample(
                                    experiences["idxs"]
                                )
                                loss, *_ = agent.learn(
                                    experiences, n_experiences=n_step_experiences
                                )
                            else:
                                loss = agent.learn(experiences)
                                if isinstance(agent, RainbowDQN):
                                    loss, *_ = loss

                elif len(memory) >= agent.batch_size and memory.size > learning_delay:
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        # Learn according to agent's RL algorithm
                        if per:
                            experiences = sampler.sample(agent.batch_size, agent.beta)
                            if n_step_memory is not None:
                                n_step_experiences = n_step_sampler.sample(
                                    experiences["idxs"]
                                )
                            else:
                                n_step_experiences = None
                            loss, idxs, priorities = agent.learn(
                                experiences, n_experiences=n_step_experiences, per=per
                            )
                            memory.update_priorities(idxs, priorities)
                        else:
                            experiences = sampler.sample(
                                agent.batch_size,
                                return_idx=True if n_step_memory is not None else False,
                            )
                            if n_step_memory is not None:
                                n_step_experiences = n_step_sampler.sample(
                                    experiences["idxs"]
                                )
                                loss, *_ = agent.learn(
                                    experiences, n_experiences=n_step_experiences
                                )
                            else:
                                loss = agent.learn(experiences)
                                if isinstance(agent, RainbowDQN):
                                    loss, *_ = loss

                if loss is not None:
                    losses.append(loss)

                state = next_state

            pbar.update(evo_steps // len(pop))

            agent.steps[-1] += steps
            fps = steps / (time.time() - start_time)
            pop_fps.append(fps)
            pop_episode_scores.append(completed_episode_scores)

            if len(losses) > 0:
                if isinstance(losses[-1], tuple):
                    actor_losses, critic_losses = list(zip(*losses))
                    mean_loss = np.mean(
                        [loss for loss in actor_losses if loss is not None]
                    ), np.mean(critic_losses)
                else:
                    mean_loss = np.mean(losses)

                pop_loss[agent_idx].append(mean_loss)

        if isinstance(agent, DQN):
            # Reset epsilon start to final epsilon value of this epoch
            eps_start = epsilon

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

            # Create the loss dictionaries
            if isinstance(agent, (DQN, RainbowDQN)):
                actor_loss_dict = {
                    f"train/agent_{index}_actor_loss": np.mean(loss[-10:])
                    for index, loss in enumerate(pop_loss)
                }
                wandb_dict.update(actor_loss_dict)
            elif isinstance(agent, (DDPG, TD3)):
                actor_loss_dict = {
                    f"train/agent_{index}_actor_loss": np.mean(
                        list(zip(*loss_list))[0][-10:]
                    )
                    for index, loss_list in enumerate(pop_loss)
                }
                critic_loss_dict = {
                    f"train/agent_{index}_critic_loss": np.mean(
                        list(zip(*loss_list))[-1][-10:]
                    )
                    for index, loss_list in enumerate(pop_loss)
                }
                wandb_dict.update(actor_loss_dict)
                wandb_dict.update(critic_loss_dict)

            if isinstance(agent, (DQN, RainbowDQN)):
                train_actions_hist = [
                    freq / sum(train_actions_hist) for freq in train_actions_hist
                ]
                train_actions_dict = {
                    f"train/action_{index}": action
                    for index, action in enumerate(train_actions_hist)
                }
                wandb_dict.update(train_actions_dict)

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
