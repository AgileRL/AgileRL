import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from agilerl.algorithms import DDPG, DQN, TD3, RainbowDQN
from agilerl.components import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.data import ReplayDataset, Transition
from agilerl.components.sampler import Sampler
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.networks.actors import DeterministicActor
from agilerl.population import Population
from agilerl.typing import GymEnvType
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)
from agilerl.vector import DummyVecEnv

if TYPE_CHECKING:
    from tensordict import TensorDictBase

InitDictType = dict[str, Any] | None
SupportedOffPolicy = DQN | RainbowDQN | DDPG | TD3
PopulationType = list[SupportedOffPolicy]
BufferType = ReplayBuffer | PrioritizedReplayBuffer | MultiStepReplayBuffer


def _learn_step(
    agent: SupportedOffPolicy,
    sampler: Sampler,
    memory: BufferType,
    n_step_memory: MultiStepReplayBuffer | None,
    n_step_sampler: Sampler | None,
    per: bool,
) -> None:
    """Execute a single learning step for the agent."""
    if per:
        experiences = sampler.sample(agent.batch_size, agent.beta)
        n_step_experiences = (
            n_step_sampler.sample(experiences["idxs"])
            if n_step_sampler is not None
            else None
        )
        _loss, idxs, priorities = agent.learn(
            experiences,
            n_experiences=n_step_experiences,
            per=per,
        )
        memory.update_priorities(idxs, priorities)
    else:
        experiences = sampler.sample(
            agent.batch_size,
            return_idx=n_step_memory is not None,
        )
        if n_step_sampler is not None:
            n_step_experiences = n_step_sampler.sample(experiences["idxs"])
            agent.learn(experiences, n_experiences=n_step_experiences)
        else:
            agent.learn(experiences)


def train_off_policy(
    env: GymEnvType,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: BufferType,
    INIT_HP: InitDictType = None,
    MUT_P: InitDictType = None,
    swap_channels: bool = False,
    max_steps: int = 1000000,
    evo_steps: int = 10000,
    eval_steps: int | None = None,
    eval_loop: int = 1,
    learning_delay: int = 0,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_decay: float = 0.995,
    target: float | None = None,
    n_step: bool = False,
    per: bool = False,
    n_step_memory: MultiStepReplayBuffer | None = None,
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
    """Run the general online off-policy RL training; returns trained population
    of agents and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[RLAlgorithm]
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
    :param n_step_memory: Multi-step Experience Replay Buffer to be used alongside Prioritized
        ERB, defaults to None
    :type n_step_memory: object, optional
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
    :type wandb_kwargs: dict, optional

    :return: Trained population of agents and their fitnesses
    :rtype: tuple[list[RLAlgorithm], list[float]]
    """
    assert isinstance(
        algo,
        str,
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    assert isinstance(eps_start, float), "Starting epsilon must be a float."
    assert isinstance(eps_end, float), "Final value of epsilon must be a float."
    assert isinstance(eps_decay, float), "Epsilon decay rate must be a float."
    if target is not None:
        assert isinstance(
            target,
            (float, int),
        ), "Target score must be a float or an integer."
    assert isinstance(n_step, bool), "'n_step' must be a boolean."
    assert isinstance(per, bool), "'per' must be a boolean."
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
        # Create dataloader from replay buffer
        replay_dataset = ReplayDataset(memory, pop[0].batch_size)
        replay_dataloader = DataLoader(replay_dataset, batch_size=None)
        replay_dataloader = accelerator.prepare(replay_dataloader)
        sampler = Sampler(dataset=replay_dataset, dataloader=replay_dataloader)
    else:
        sampler = Sampler(memory=memory)
        if n_step_memory is not None:
            n_step_sampler = Sampler(memory=n_step_memory)

    # Format progress bar
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
        init_hyperparams=INIT_HP,
        mutation_hyperparams=MUT_P,
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
            scores = np.zeros(num_envs)
            completed_episode_scores: list[float] = []
            steps = 0

            if isinstance(agent, DQN):
                epsilon = eps_start

            for idx_step in range(evo_steps // num_envs):
                if swap_channels:
                    obs = obs_channels_to_first(obs)

                # Get next action from agent
                if isinstance(agent, DQN):
                    action_mask = info.get("action_mask", None)
                    action = agent.get_action(obs, epsilon, action_mask=action_mask)
                    epsilon = max(eps_end, epsilon * eps_decay)
                elif isinstance(agent, RainbowDQN):
                    action_mask = info.get("action_mask", None)
                    action = agent.get_action(obs, action_mask=action_mask)
                else:
                    raw_action = agent.get_action(obs)

                    # Rescale action to action space bounds
                    action = DeterministicActor.rescale_action(
                        action=torch.from_numpy(raw_action),
                        low=agent.action_low,
                        high=agent.action_high,
                        output_activation=agent.actor.output_activation,
                    )
                    action = action.cpu().numpy()

                # Act in environment
                next_obs, reward, done, trunc, info = env.step(action)
                scores += np.array(reward)

                reset_noise_indices = []
                for idx, (d, t) in enumerate(zip(done, trunc, strict=False)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                if isinstance(agent, (DDPG, TD3)):
                    agent.reset_action_noise(reset_noise_indices)

                steps += num_envs

                # Save experience to replay buffer
                next_obs = (
                    obs_channels_to_first(next_obs) if swap_channels else next_obs
                )

                # Save network output in buffer
                if isinstance(agent, (DDPG, TD3)):
                    action = raw_action

                transition: TensorDictBase = Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                )

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
                        ((agent.metrics.steps + idx_step + 1) * num_envs / max_steps),
                        1.0,
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
                        _learn_step(
                            agent,
                            sampler,
                            memory,
                            n_step_memory,
                            n_step_sampler if n_step_memory is not None else None,
                            per,
                        )

                elif len(memory) >= agent.batch_size and memory.size > learning_delay:
                    for _ in range(num_envs // agent.learn_step):
                        _learn_step(
                            agent,
                            sampler,
                            memory,
                            n_step_memory,
                            n_step_sampler if n_step_memory is not None else None,
                            per,
                        )

                obs = next_obs

            agent.add_scores(completed_episode_scores)
            agent.finalize_evo_step(steps)
            pbar.update(evo_steps // population.size)

        if isinstance(agent, DQN):
            eps_start = epsilon

        population.increment_evo_step()

        # Evaluate population
        for agent in population.agents:
            agent.test(
                env,
                swap_channels=swap_channels,
                max_steps=eval_steps,
                loop=eval_loop,
            )

        population.report_metrics(clear=True)

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
