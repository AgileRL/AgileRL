import os
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    GRPO,
    IPPO,
    MADDPG,
    MATD3,
    PPO,
    TD3,
    NeuralTS,
    NeuralUCB,
    RainbowDQN,
)
from agilerl.algorithms.core import EvolvableAlgorithm, LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.modules.base import EvolvableModule
from agilerl.typing import GymSpaceType, PopulationType
from agilerl.utils.algo_utils import CosineLRScheduleConfig, clone_llm
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
    env: Callable[[], ParallelEnv],
    num_envs: int = 1,
    **env_kwargs: Any,
) -> AsyncPettingZooVecEnv:
    """Returns async-vectorized PettingZoo parallel environments.

    :param env: PettingZoo parallel environment object
    :type env: pettingzoo.utils.env.ParallelEnv
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional

    :return: Async-vectorized PettingZoo parallel environments
    :rtype: agilerl.vector.pz_async_vec_env.AsyncPettingZooVecEnv
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
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                double=INIT_HP.get("DOUBLE", False),
                cudagraphs=INIT_HP.get("CUDAGRAPHS", False),
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
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                beta=INIT_HP.get("BETA", 0.4),
                prior_eps=INIT_HP.get("PRIOR_EPS", 0.00001),
                num_atoms=INIT_HP.get("NUM_ATOMS", 51),
                v_min=INIT_HP.get("V_MIN", -100),
                v_max=INIT_HP.get("V_MAX", 100),
                n_step=INIT_HP.get("N_STEP", 3),
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
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                policy_freq=INIT_HP.get("POLICY_FREQ", 2),
                actor_network=actor_network,
                critic_network=critic_network,
                share_encoders=INIT_HP.get("SHARE_ENCODERS", True),
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
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 2048),
                gamma=INIT_HP.get("GAMMA", 0.99),
                gae_lambda=INIT_HP.get("GAE_LAMBDA", 0.95),
                action_std_init=INIT_HP.get("ACTION_STD_INIT", 0.6),
                clip_coef=INIT_HP.get("CLIP_COEF", 0.2),
                ent_coef=INIT_HP.get("ENT_COEF", 0.01),
                vf_coef=INIT_HP.get("VF_COEF", 0.5),
                max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 0.5),
                target_kl=INIT_HP.get("TARGET_KL"),
                update_epochs=INIT_HP.get("UPDATE_EPOCHS", 4),
                share_encoders=INIT_HP.get("SHARE_ENCODERS", True),
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
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                double=INIT_HP.get("DOUBLE", False),
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
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.005),
                policy_freq=INIT_HP.get("POLICY_FREQ", 2),
                actor_network=actor_network,
                critic_networks=critic_network,
                share_encoders=INIT_HP.get("SHARE_ENCODERS", True),
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
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.95),
                tau=INIT_HP.get("TAU", 0.01),
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
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                policy_freq=INIT_HP.get("POLICY_FREQ", 2),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.95),
                tau=INIT_HP.get("TAU", 0.01),
                actor_networks=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
                torch_compiler=torch_compiler,
            )
            population.append(agent)

    elif algo == "IPPO":
        for idx in range(population_size):
            agent = IPPO(
                observation_spaces=observation_space,
                action_spaces=action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
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
                actor_networks=actor_network,
                critic_networks=critic_network,
                action_batch_size=INIT_HP.get("ACTION_BATCH_SIZE", None),
                device=device,
                accelerator=accelerator,
                torch_compiler=torch_compiler,
            )
            population.append(agent)

    elif algo == "IPPO":
        for idx in range(population_size):
            agent = IPPO(
                observation_spaces=observation_space,
                action_spaces=action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
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
                gamma=INIT_HP.get("GAMMA", 1),
                lamb=INIT_HP.get("LAMBDA", 1),
                reg=INIT_HP.get("REG", 0.000625),
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 2),
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
                gamma=INIT_HP.get("GAMMA", 1),
                lamb=INIT_HP.get("LAMBDA", 1),
                reg=INIT_HP.get("REG", 0.000625),
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.003),
                learn_step=INIT_HP.get("LEARN_STEP", 2),
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "GRPO":
        for idx in range(population_size):
            agent = GRPO(
                observation_space=observation_space,
                action_space=action_space,
                actor_network=clone_llm(actor_network),
                pad_token_id=INIT_HP.get("PAD_TOKEN_ID"),
                hp_config=hp_config,
                index=idx,
                batch_size=INIT_HP.get("BATCH_SIZE_PER_GPU", 1),
                beta=INIT_HP.get("BETA", 0.001),
                lr=INIT_HP.get("LR", 5e-7),
                clip_coef=INIT_HP.get("CLIP_COEF", 0.2),
                max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 0.1),
                update_epochs=INIT_HP.get("UPDATE_EPOCHS", 1),
                group_size=INIT_HP.get("GROUP_SIZE", 8),
                temperature=INIT_HP.get("TEMPERATURE", 0.9),
                calc_position_embeddings=INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
                reduce_memory_peak=INIT_HP.get("REDUCE_MEMORY_PEAK", False),
                max_output_tokens=INIT_HP.get("MAX_OUTPUT_TOKENS", 1024),
                min_output_tokens=INIT_HP.get("MIN_OUTPUT_TOKENS", None),
                cosine_lr_schedule_config=(
                    CosineLRScheduleConfig(**INIT_HP.get("COSINE_lR_SCHEDULER", None))
                    if INIT_HP.get("COSINE_lR_SCHEDULER", None) is not None
                    else None
                ),
                accelerator=Accelerator() if accelerator else None,
                device=device,
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


def tournament_selection_and_mutation(
    population: PopulationType,
    tournament: TournamentSelection,
    mutation: Mutations,
    env_name: str,
    algo: Optional[str] = None,
    elite_path: Optional[str] = None,
    save_elite: bool = False,
    accelerator: Optional[Accelerator] = None,
    language_model: Optional[bool] = False,
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
    :param language_model: Flag to indicate if the environment is a language model, defaults to False
    :type language_model: bool, optional
    :return: Population of agents after tournament selection and mutation
    :rtype: list[PopulationType]
    """
    if algo is None:
        algo = population[0].__class__.__name__

    if language_model:
        elite, population = tournament.select(population)
        if accelerator is None or (
            accelerator is not None and accelerator.is_main_process
        ):
            population = mutation.mutation(population)
        if accelerator is not None:
            accelerator.wait_for_everyone()
            consolidate_mutations(population)
            accelerator.wait_for_everyone()
        if save_elite:
            save_llm_checkpoint(elite, elite_path)
        return population

    if accelerator is not None:
        # Save temporary models for accelerator processes
        accel_temp_models_path = f"models/{env_name}"
        if accelerator.is_main_process:
            if not os.path.exists(accel_temp_models_path):
                os.makedirs(accel_temp_models_path)
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


def gather_tensor(
    tensor: Union[torch.Tensor, float], accelerator: Accelerator
) -> torch.Tensor:
    """Gather tensors from gpus

    :param tensor: Tensor to gather
    :type tensor: torch.Tensor
    :param accelerator: Accelerator object
    :type accelerator: accelerate.Accelerator
    :return: Stacked tensors
    :rtype: torch.Tensor
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device=accelerator.device)
    tensor = tensor.to(accelerator.device)
    gathered_tensors = accelerator.gather(tensor)
    return gathered_tensors


def aggregate_metrics_across_gpus(
    accelerator: Accelerator, metric_tensor: Union[torch.Tensor, float]
) -> float:
    """Aggregate gathered tensors

    :param accelerator: Accelerator object
    :type accelerator: accelerate.Accelerator
    :param metric_tensor: Metrics
    :type metric_tensor: torch.Tensor
    :return: Mean metric
    :rtype: float
    """
    all_metrics = gather_tensor(metric_tensor, accelerator)
    avg_metrics = all_metrics.mean().item()
    return avg_metrics


def save_llm_checkpoint(agent: EvolvableAlgorithm, checkpoint_path: str | None) -> None:
    """Checkpoint the LLM

    :param agent: Agent
    :type agent: EvolvableAlgorithm
    :param checkpoint_path: Checkpoint path
    :type checkpoint_path: str
    """
    base_path = "./saved_checkpoints" if checkpoint_path is None else checkpoint_path
    path = base_path + f"/{agent.algo}"
    os.makedirs(path, exist_ok=True)
    if agent.accelerator is not None:
        agent.accelerator.wait_for_everyone()
        agent.actor.save_pretrained(path)
        agent.accelerator.wait_for_everyone()
    else:
        agent.actor.save_pretrained(path)


def consolidate_mutations(population: PopulationType) -> None:
    """Consolidate mutations across processes during LLM fintuning

    :param population: Population of agents
    :type population: list[EvolvableAlgorithm]
    """
    if not isinstance(population[0], LLMAlgorithm):
        warnings.warn("Consolidate mutations is only supported for LLMAlgorithm.")
        return
    for agent in population:
        index, mut, mut_value = broadcast_object_list(
            [
                agent.index,
                agent.mut,
                getattr(agent, agent.mut if agent.mut is not None else "None", "None"),
            ],
            from_process=0,
        )
        assert index == agent.index
        agent.mut = mut
        setattr(agent, mut, mut_value)
        if mut == "lr":
            LLMAlgorithm.update_lr(agent.optimizer, getattr(agent, mut))
