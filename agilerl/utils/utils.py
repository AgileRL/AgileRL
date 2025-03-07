import os
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import wandb
from accelerate import Accelerator
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    MADDPG,
    MATD3,
    PPO,
    TD3,
    NeuralTS,
    NeuralUCB,
    RainbowDQN,
)
from agilerl.algorithms.core import EvolvableAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.modules.base import EvolvableModule
from agilerl.typing import GymSpaceType, PopulationType
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

SupportedObservationSpace = Union[
    spaces.Box, spaces.Discrete, spaces.Dict, spaces.Tuple
]


def make_vect_envs(
    env_name: Optional[str] = None,
    num_envs=1,
    *,
    make_env: Optional[Callable] = None,
    should_async_vector: bool = True,
    **env_kwargs,
):
    """Returns async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param make_env: Function that creates a gym environment, defaults use gym.make(env_name)
    :type make_env: Callable, optional
    :param should_async_vector: Whether to asynchronous vectorized environments, defaults to True
    :type should_async_vector: bool, optional
    """
    if env_name is None and make_env is None:
        raise ValueError("Either env_name or make_env must be provided")

    vectorize = (
        gym.vector.AsyncVectorEnv if should_async_vector else gym.vector.SyncVectorEnv
    )

    def default_make_env():
        return gym.make(env_name, **env_kwargs)

    make_env = make_env or default_make_env

    return vectorize([make_env for _ in range(num_envs)])


def make_multi_agent_vect_envs(
    env: Callable[[], ParallelEnv], num_envs: int = 1, **env_kwargs: Any
) -> AsyncPettingZooVecEnv:
    """Returns async-vectorized PettingZoo parallel environments.

    :param env: PettingZoo parallel environment object
    :type env: pettingzoo.utils.env.ParallelEnv
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    env_fns = [lambda: env(**env_kwargs) for _ in range(num_envs)]
    return AsyncPettingZooVecEnv(env_fns=env_fns)


def make_skill_vect_envs(
    env_name: str, skill: Any, num_envs: int = 1
) -> gym.vector.AsyncVectorEnv:
    """Returns async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param skill: Skill wrapper to apply to environment
    :type skill: agilerl.wrappers.learning.Skill
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: skill(gym.make(env_name)) for i in range(num_envs)]
    )


def observation_space_channels_to_first(
    observation_space: SupportedObservationSpace,
) -> SupportedObservationSpace:
    """Swaps the channel order of an observation space from [H, W, C] -> [C, H, W].

    :param observation_space: Observation space
    :type observation_space: spaces.Box, spaces.Dict, spaces.Tuple, spaces.Discrete
    :return: Observation space with swapped channels
    :rtype: spaces.Box, spaces.Dict, spaces.Tuple, spaces.Discrete
    """
    if isinstance(observation_space, spaces.Dict):
        for key in observation_space.spaces.keys():
            if (
                isinstance(observation_space[key], spaces.Box)
                and len(observation_space[key].shape) == 3
            ):
                observation_space[key] = observation_space_channels_to_first(
                    observation_space[key]
                )
    elif isinstance(observation_space, spaces.Tuple):
        observation_space = spaces.Tuple(
            [
                (
                    observation_space_channels_to_first(space)
                    if isinstance(space, spaces.Box) and len(space.shape) == 3
                    else space
                )
                for space in observation_space.spaces
            ]
        )
    elif isinstance(observation_space, spaces.Box):
        low = observation_space.low.transpose(2, 0, 1)
        high = observation_space.high.transpose(2, 0, 1)
        observation_space = spaces.Box(
            low=low, high=high, dtype=observation_space.dtype
        )

    return observation_space


def create_population(
    algo: str,
    observation_space: GymSpaceType,
    action_space: GymSpaceType,
    net_config: Optional[Dict[str, Any]],
    INIT_HP: Dict[str, Any],
    hp_config: Optional[HyperparameterConfig] = None,
    actor_network: Optional[EvolvableModule] = None,
    critic_network: Optional[EvolvableModule] = None,
    agent_wrapper: Optional[Callable] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    population_size: int = 1,
    num_envs: int = 1,
    device: str = "cpu",
    accelerator: Optional[Any] = None,
    torch_compiler: Optional[Any] = None,
) -> PopulationType:
    """Returns population of identical agents.

    :param algo: RL algorithm
    :type algo: str
    :param observation_space: Observation space
    :type observation_space: spaces.Space
    :param action_space: Action space
    :type action_space: spaces.Space
    :param net_config: Network configuration
    :type net_config: dict or None
    :param INIT_HP: Initial hyperparameters
    :type INIT_HP: dict
    :param hp_config: Choice of algorithm hyperparameters to mutate during training, defaults to None
    :type hp_config: HyperparameterConfig, optional
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
    :param critic_network: Custom critic network, defaults to None
    :type critic_network: nn.Module, optional
    :param population_size: Number of agents in population, defaults to 1
    :type population_size: int, optional
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param torch_compiler: Torch compiler, defaults to None
    :type torch_compiler: Any, optional
    :return: Population of agents
    :rtype: list[EvolvableAlgorithm]
    """
    population = []
    if algo == "DQN":
        for idx in range(population_size):
            agent = DQN(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                double=INIT_HP["DOUBLE"],
                cudagraphs=INIT_HP["CUDAGRAPHS"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "Rainbow DQN":
        for idx in range(population_size):
            agent = RainbowDQN(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                beta=INIT_HP["BETA"],
                prior_eps=INIT_HP["PRIOR_EPS"],
                num_atoms=INIT_HP["NUM_ATOMS"],
                v_min=INIT_HP["V_MIN"],
                v_max=INIT_HP["V_MAX"],
                n_step=INIT_HP["N_STEP"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "DDPG":
        for idx in range(population_size):
            agent = DDPG(
                observation_space=observation_space,
                action_space=action_space,
                O_U_noise=INIT_HP["O_U_NOISE"],
                expl_noise=INIT_HP["EXPL_NOISE"],
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP["MEAN_NOISE"],
                theta=INIT_HP["THETA"],
                dt=INIT_HP["DT"],
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr_actor=INIT_HP["LR_ACTOR"],
                lr_critic=INIT_HP["LR_CRITIC"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                policy_freq=INIT_HP["POLICY_FREQ"],
                actor_network=actor_network,
                critic_network=critic_network,
                device=device,
                accelerator=accelerator,
            )

            agent = (
                agent_wrapper(agent, **wrapper_kwargs)
                if agent_wrapper is not None
                else agent
            )
            population.append(agent)

    elif algo == "PPO":
        for idx in range(population_size):
            agent = PPO(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                gae_lambda=INIT_HP["GAE_LAMBDA"],
                action_std_init=INIT_HP["ACTION_STD_INIT"],
                clip_coef=INIT_HP["CLIP_COEF"],
                ent_coef=INIT_HP["ENT_COEF"],
                vf_coef=INIT_HP["VF_COEF"],
                max_grad_norm=INIT_HP["MAX_GRAD_NORM"],
                target_kl=INIT_HP["TARGET_KL"],
                update_epochs=INIT_HP["UPDATE_EPOCHS"],
                actor_network=actor_network,
                critic_network=critic_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "CQN":
        for idx in range(population_size):
            agent = CQN(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                double=INIT_HP["DOUBLE"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "TD3":
        for idx in range(population_size):
            agent = TD3(
                observation_space=observation_space,
                action_space=action_space,
                O_U_noise=INIT_HP["O_U_NOISE"],
                expl_noise=INIT_HP["EXPL_NOISE"],
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP["MEAN_NOISE"],
                theta=INIT_HP["THETA"],
                dt=INIT_HP["DT"],
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr_actor=INIT_HP["LR_ACTOR"],
                lr_critic=INIT_HP["LR_CRITIC"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                policy_freq=INIT_HP["POLICY_FREQ"],
                actor_network=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "MADDPG":
        for idx in range(population_size):
            agent = MADDPG(
                observation_spaces=observation_space,
                action_spaces=action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
                O_U_noise=INIT_HP["O_U_NOISE"],
                expl_noise=INIT_HP["EXPL_NOISE"],
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP["MEAN_NOISE"],
                theta=INIT_HP["THETA"],
                dt=INIT_HP["DT"],
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr_actor=INIT_HP["LR_ACTOR"],
                lr_critic=INIT_HP["LR_CRITIC"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                actor_networks=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
                torch_compiler=torch_compiler,
            )
            population.append(agent)

    elif algo == "MATD3":
        for idx in range(population_size):
            agent = MATD3(
                observation_spaces=observation_space,
                action_spaces=action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
                O_U_noise=INIT_HP["O_U_NOISE"],
                expl_noise=INIT_HP["EXPL_NOISE"],
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP["MEAN_NOISE"],
                theta=INIT_HP["THETA"],
                dt=INIT_HP["DT"],
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr_actor=INIT_HP["LR_ACTOR"],
                lr_critic=INIT_HP["LR_CRITIC"],
                policy_freq=INIT_HP["POLICY_FREQ"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                actor_networks=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
                torch_compiler=torch_compiler,
            )
            population.append(agent)

    elif algo == "NeuralUCB":
        for idx in range(population_size):
            agent = NeuralUCB(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                gamma=INIT_HP["GAMMA"],
                lamb=INIT_HP["LAMBDA"],
                reg=INIT_HP["REG"],
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "NeuralTS":
        for idx in range(population_size):
            agent = NeuralTS(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                gamma=INIT_HP["GAMMA"],
                lamb=INIT_HP["LAMBDA"],
                reg=INIT_HP["REG"],
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    return population


def save_population_checkpoint(
    population: PopulationType,
    save_path: str,
    overwrite_checkpoints: bool,
    accelerator: Optional[Accelerator] = None,
) -> None:
    """Saves checkpoint of population of agents.

    :param population: Population of agents
    :type population: list[PopulationType]
    :param save_path: Path to save checkpoint
    :type save_path: str
    :param overwrite_checkpoints: Flag to overwrite checkpoints
    :type overwrite_checkpoints: bool
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    """
    if accelerator is not None:
        # Need to unwrap models from acccelerator before saving
        accelerator.wait_for_everyone()
        for model in population:
            model.unwrap_models()
        accelerator.wait_for_everyone()

        # Save checkpoint on main process
        if accelerator.is_main_process:
            for i, agent in enumerate(population):
                current_checkpoint_path = (
                    f"{save_path}_{i}.pt"
                    if overwrite_checkpoints
                    else f"{save_path}_{i}_{agent.steps[-1]}.pt"
                )
                agent.save_checkpoint(current_checkpoint_path)
            print("Saved checkpoint.")
        accelerator.wait_for_everyone()

        # Load models back to accelerator processes
        for model in population:
            model.wrap_models()
        accelerator.wait_for_everyone()
    else:
        # Save checkpoint
        for i, agent in enumerate(population):
            current_checkpoint_path = (
                f"{save_path}_{i}.pt"
                if overwrite_checkpoints
                else f"{save_path}_{i}_{agent.steps[-1]}.pt"
            )
            agent.save_checkpoint(current_checkpoint_path)
        print("Saved checkpoint.")


def tournament_selection_and_mutation(
    population: PopulationType,
    tournament: TournamentSelection,
    mutation: Mutations,
    env_name: str,
    algo: Optional[str] = None,
    elite_path: Optional[str] = None,
    save_elite: bool = False,
    accelerator: Optional[Accelerator] = None,
) -> PopulationType:
    """Performs tournament selection and mutation on a population of agents.

    :param population: Population of agents
    :type population: list[PopulationType]
    :param tournament: Tournament selection object
    :type tournament: TournamentSelection
    :param mutation: Mutation object
    :type mutation: Mutations
    :param env_name: Environment name
    :type env_name: str
    :param elite_path: Path to save elite agent, defaults to None
    :type elite_path: str, optional
    :param save_elite: Flag to save elite agent, defaults to False
    :type save_elite: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional

    :return: Population of agents after tournament selection and mutation
    :rtype: list[PopulationType]
    """
    if algo is None:
        algo = population[0].__class__.__name__

    # Save temporary models for accelerator processes
    if accelerator is not None:
        accel_temp_models_path = f"models/{env_name}"
        if accelerator.is_main_process:
            if not os.path.exists(accel_temp_models_path):
                os.makedirs(accel_temp_models_path)

    if accelerator is not None:
        # Need to unwrap models from acccelerator before selecting and mutating
        accelerator.wait_for_everyone()
        for model in population:
            model.unwrap_models()

        accelerator.wait_for_everyone()

        # Perform tournament selection and mutation on main process
        if accelerator.is_main_process:
            elite, population = tournament.select(population)
            population = mutation.mutation(population)
            for pop_i, model in enumerate(population):
                model.save_checkpoint(f"{accel_temp_models_path}/{algo}_{pop_i}.pt")
        accelerator.wait_for_everyone()

        # Load models back to accelerator processes
        if not accelerator.is_main_process:
            for pop_i, model in enumerate(population):
                model.load_checkpoint(f"{accel_temp_models_path}/{algo}_{pop_i}.pt")
        accelerator.wait_for_everyone()

        # Wrap models back to accelerator
        for model in population:
            model.wrap_models()
    else:
        # Perform tournament selection and mutation
        elite, population = tournament.select(population)
        population = mutation.mutation(population)

    if save_elite:
        elite_save_path = (
            elite_path.split(".pt")[0]
            if elite_path is not None
            else f"{env_name}-elite_{algo}"
        )
        elite.save_checkpoint(f"{elite_save_path}.pt")

    return population


def init_wandb(
    algo: str,
    env_name: str,
    init_hyperparams: Optional[Dict[str, Any]] = None,
    mutation_hyperparams: Optional[Dict[str, Any]] = None,
    wandb_api_key: Optional[str] = None,
    accelerator: Optional[Accelerator] = None,
    project: str = "AgileRL",
) -> None:
    """Initializes wandb for logging hyperparameters and run metadata.

    :param algo: RL algorithm
    :type algo: str
    :param env_name: Environment name
    :type env_name: str
    :param init_hyperparams: Initial hyperparameters, defaults to None
    :type init_hyperparams: dict, optional
    :param mutation_hyperparams: Mutation hyperparameters, defaults to None
    :type mutation_hyperparams: dict, optional
    :param wandb_api_key: Wandb API key, defaults to None
    :type wandb_api_key: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    """
    if not hasattr(wandb, "api"):
        if wandb_api_key is not None:
            wandb.login(key=wandb_api_key)
        else:
            warnings.warn("Must login to wandb with API key.")

    config_dict = {}
    if init_hyperparams is not None:
        config_dict.update(init_hyperparams)
    if mutation_hyperparams is not None:
        config_dict.update(mutation_hyperparams)

    if accelerator is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            wandb.init(
                # set the wandb project where this run will be logged
                project=project,
                name="{}-EvoHPO-{}-{}".format(
                    env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
                ),
                # track hyperparameters and run metadata
                config=config_dict,
            )
        accelerator.wait_for_everyone()
    else:
        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            name="{}-EvoHPO-{}-{}".format(
                env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
            ),
            # track hyperparameters and run metadata
            config=config_dict,
        )


def calculate_vectorized_scores(
    rewards: np.ndarray,
    terminations: np.ndarray,
    include_unterminated: bool = False,
    only_first_episode: bool = True,
) -> List[float]:
    """
    Calculate the vectorized scores for episodes based on rewards and terminations.

    :param rewards: Array of rewards for each environment.
    :type rewards: np.ndarray
    :param terminations: Array indicating termination points for each environment.
    :type terminations: np.ndarray
    :param include_unterminated: Whether to include rewards from unterminated episodes, defaults to False.
    :type include_unterminated: bool, optional
    :param only_first_episode: Whether to consider only the first episode, defaults to True.
    :type only_first_episode: bool, optional
    :return: List of episode rewards.
    :rtype: list[float]
    """
    episode_rewards = []
    num_envs, _ = rewards.shape

    for env_index in range(num_envs):
        # Find the indices where episodes terminate for the current environment
        termination_indices = np.where(terminations[env_index] == 1)[0]

        # If no terminations, sum the entire reward array for this environment
        if len(termination_indices) == 0:
            episode_reward = np.sum(rewards[env_index])
            episode_rewards.append(episode_reward)
            continue  # Skip to the next environment

        # Initialize the starting index for segmenting
        start_index = 0

        for termination_index in termination_indices:
            # Sum the rewards for the current episode
            episode_reward = np.sum(
                rewards[env_index, start_index : termination_index + 1]
            )

            # Store the episode reward
            episode_rewards.append(episode_reward)

            # If only the first episode is required, break after processing it
            if only_first_episode:
                break

            # Update the starting index for segmenting
            start_index = termination_index + 1

        # If include_unterminated is True, sum the rewards from the last termination index to the end
        if (
            not only_first_episode
            and include_unterminated
            and start_index < len(rewards[env_index])
        ):
            episode_reward = np.sum(rewards[env_index, start_index:])
            episode_rewards.append(episode_reward)

    return episode_rewards


def print_hyperparams(pop: PopulationType) -> None:
    """Prints current hyperparameters of agents in a population and their fitnesses.

    :param pop: Population of agents
    :type pop: list[EvolvableAlgorithm]
    """
    for agent in pop:
        print(
            "Agent ID: {}    Mean 5 Fitness: {:.2f}    Attributes: {}".format(
                agent.index,
                np.mean(agent.fitness[-5:]),
                EvolvableAlgorithm.inspect_attributes(agent),
            )
        )


def plot_population_score(pop: PopulationType) -> None:
    """Plots the fitness scores of agents in a population.

    :param pop: Population of agents
    :type pop: list[EvolvableAlgorithm]
    """
    plt.figure()
    for agent in pop:
        scores = agent.fitness
        steps = agent.steps[:-1]
        plt.plot(steps, scores)
    plt.title("Score History - Mutations")
    plt.xlabel("Steps")
    plt.ylim(bottom=-400)
    plt.show()


def get_env_defined_actions(info, agents):
    env_defined_actions = {
        agent: info[agent].get("env_defined_action", None) for agent in agents
    }

    if all(eda is None for eda in env_defined_actions.values()):
        return

    return env_defined_actions
