# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from pettingzoo.utils.env import ParallelEnv

from agilerl.algorithms.core import EvolvableAlgorithm, LLMAlgorithm
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.logger import CSVLogger, StdOutLogger, TensorboardLogger, WandbLogger
from agilerl.protocols import EvolvableAlgorithmProtocol, SelectionStrategyProtocol
from agilerl.training.configs import LoggerExperiment, TrainLoggingConfig
from agilerl.typing import InfosDict, PopulationType
from agilerl.utils.algo_utils import DummyOptimizer
from agilerl.utils.constructor_kwargs import accept_flat_kwargs
from agilerl.utils.population import (
    _lora_config_from_init_hp as _lora_config_from_init_hp,
)
from agilerl.utils.population import (
    _normalize_algo_name as _normalize_algo_name,
)
from agilerl.utils.population import (
    _prepare_llm_algo_kwargs as _prepare_llm_algo_kwargs,
)
from agilerl.utils.population import (
    _validate_llm_kwargs as _validate_llm_kwargs,
)
from agilerl.utils.population import (
    create_population as create_population,
)
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

AgentT = TypeVar("AgentT", bound=EvolvableAlgorithmProtocol)

_BOX2D_ENV_PREFIXES = (
    "LunarLander",
    "LunarLanderContinuous",
    "BipedalWalker",
    "BipedalWalkerHardcore",
    "CarRacing",
)


def _check_box2d_available(env_name: str) -> None:
    """Raise a helpful error if a Box2D environment is requested but box2d-py is missing."""
    if not any(env_name.startswith(prefix) for prefix in _BOX2D_ENV_PREFIXES):
        return
    try:
        import Box2D  # noqa: F401
    except ImportError:
        msg = (
            f"Environment '{env_name}' requires the Box2D physics engine.\n"
            "Install it with:  pip install agilerl[box2d]  (or  uv sync --extra box2d)\n"
            "Note: building box2d-py requires the system package 'swig'.\n"
            "  Ubuntu/Debian: sudo apt-get install swig\n"
            "  macOS:         brew install swig\n"
            "Alternatively, use a non-Box2D environment such as 'CartPole-v1'."
        )
        raise ImportError(msg) from None


def make_vect_envs(
    env_name: str | None = None,
    num_envs: int = 1,
    *,
    make_env: Callable[..., Any] | None = None,
    should_async_vector: bool = True,
    extra_wrappers: list[type] | None = None,
    **env_kwargs: Any,
) -> gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv:
    """Return async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param make_env: Function that creates a gym environment, defaults use gym.make(env_name)
    :type make_env: Callable, optional
    :param should_async_vector: Whether to asynchronous vectorized environments, defaults to True
    :type should_async_vector: bool, optional
    :param extra_wrappers: Optional list of wrapper classes to apply to each individual
        environment before vectorization.
    :type extra_wrappers: list[type] or None, optional
    :return: Vectorized gym environments
    :rtype: gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv
    """
    if env_name is not None:
        _check_box2d_available(env_name)

    vectorize = (
        gym.vector.AsyncVectorEnv if should_async_vector else gym.vector.SyncVectorEnv
    )

    if make_env is None:
        if env_name is None:
            msg = "Either env_name or make_env must be provided"
            raise ValueError(msg)
        env_id: str = env_name

        def make_env() -> gym.Env:
            return gym.make(env_id, **env_kwargs)

    if extra_wrappers is not None:
        _inner_make_env = make_env

        def make_env() -> gym.Env:
            env = _inner_make_env()
            for wrapper_cls in extra_wrappers:
                env = wrapper_cls(env)
            return env

    return vectorize([make_env for _ in range(num_envs)])


def make_multi_agent_vect_envs(
    env: Callable[..., ParallelEnv],
    num_envs: int = 1,
    *,
    extra_wrappers: list[type] | None = None,
    **env_kwargs: Any,
) -> AsyncPettingZooVecEnv:
    """Return async-vectorized PettingZoo parallel environments.

    :param env: PettingZoo parallel environment object
    :type env: pettingzoo.utils.env.ParallelEnv
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param extra_wrappers: Optional list of wrapper classes to apply to each individual
        environment before vectorization.
    :type extra_wrappers: list[type] or None, optional

    :return: Async-vectorized PettingZoo parallel environments
    :rtype: agilerl.vector.pz_async_vec_env.AsyncPettingZooVecEnv
    """
    if extra_wrappers is not None:
        _original_env = env

        def env(**kwargs: Any) -> ParallelEnv:
            e = _original_env(**kwargs)
            for wrapper_cls in extra_wrappers:
                e = wrapper_cls(e)
            return e

    env_fns = [lambda: env(**env_kwargs) for _ in range(num_envs)]
    return AsyncPettingZooVecEnv(env_fns=env_fns)


def make_skill_vect_envs(
    env_name: str,
    skill: Callable[..., gym.Env],
    num_envs: int = 1,
) -> gym.vector.AsyncVectorEnv:
    """Return async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param skill: Skill wrapper to apply to environment
    :type skill: agilerl.wrappers.learning.Skill
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: skill(gym.make(env_name)) for i in range(num_envs)],
    )


def suppress_verbose_logging() -> None:
    """Suppress verbose logging from DeepSpeed, Accelerate, and related libraries."""
    # Suppress DeepSpeed logging
    logging.getLogger("deepspeed").setLevel(logging.WARNING)

    # Suppress Accelerate logging
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    # Suppress specific DeepSpeed components
    logging.getLogger("deepspeed.runtime.engine").setLevel(logging.WARNING)
    logging.getLogger("deepspeed.runtime.zero").setLevel(logging.WARNING)
    logging.getLogger("deepspeed.checkpoint").setLevel(logging.WARNING)

    # Suppress JAX logging (if used)
    logging.getLogger("jax").setLevel(logging.WARNING)

    # Set root logger to INFO to avoid suppressing important messages
    logging.getLogger().setLevel(logging.INFO)


def default_progress_bar(
    max_steps: int,
    accelerator: Accelerator | None = None,
) -> tqdm.tqdm:
    """Return a default progress bar.

    :param max_steps: Maximum number of steps
    :type max_steps: int
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :return: Progress bar
    :rtype: tqdm.tqdm
    """
    bar_format = (
        "Training Progress │ "
        "{percentage:3.0f}% │ "
        "{bar:20} │ "
        "{n_fmt}/{total_fmt} steps │ "
        "⏱️ {elapsed} │ "
        "⏳ {remaining} │ "
        "{rate_fmt}"
        "{postfix}"
    )
    disable = (
        not accelerator.is_local_main_process if accelerator is not None else False
    )
    return tqdm.trange(
        max_steps,
        unit="step",
        bar_format=bar_format,
        ascii=False,
        dynamic_ncols=True,
        disable=disable,
    )


def save_population_checkpoint(
    population: list[AgentT],
    save_path: str,
    overwrite_checkpoints: bool,
    accelerator: Accelerator | None = None,
) -> None:
    """Save checkpoint of population of agents.

    :param population: Population of agents
    :type population: list[AgentT]
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
                    else f"{save_path}_{i}_{agent.steps}.pt"
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
                else f"{save_path}_{i}_{agent.steps}.pt"
            )
            agent.save_checkpoint(current_checkpoint_path)


def _save_standard_elite(
    elite: EvolvableAlgorithmProtocol,
    *,
    env_name: str,
    algo: str,
    elite_path: str | None,
) -> None:
    """Checkpoint a non-LLM elite agent to disk.

    :param elite: The elite agent to save.
    :type elite: EvolvableAlgorithmProtocol
    :param env_name: Environment name, used to build the default filename.
    :type env_name: str
    :param algo: Algorithm name, used to build the default filename.
    :type algo: str
    :param elite_path: Explicit .pt path to save to; when None a
        {env_name}-elite_{algo}.pt name is used.
    :type elite_path: str | None
    """
    elite_save_path = (
        elite_path.split(".pt")[0]
        if elite_path is not None
        else f"{env_name}-elite_{algo}"
    )
    elite.save_checkpoint(f"{elite_save_path}.pt")


def run_selection_and_mutation(
    selection_strategy: SelectionStrategyProtocol | None,
    population: list[AgentT],
    mutation: Mutations,
    env_name: str,
    algo: str | None = None,
    elite_path: str | None = None,
    save_elite: bool = False,
    accelerator: Accelerator | None = None,
    language_model: bool | None = False,
) -> list[AgentT]:
    """Perform a hyperparameter optimisation step on a population of agents.

    :param selection_strategy: The selection strategy driving evolution; None
        returns the population unchanged.
    :type selection_strategy: SelectionStrategyProtocol | None
    :param population: Population of agents.
    :type population: list[AgentT]
    :param mutation: Mutation object.
    :type mutation: Mutations
    :param env_name: Environment name.
    :type env_name: str
    :param algo: Algorithm name; inferred from the population when None. Defaults to None.
    :type algo: str, optional
    :param elite_path: Path to save the elite agent, defaults to None.
    :type elite_path: str, optional
    :param save_elite: Flag to save the elite agent, defaults to False.
    :type save_elite: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None.
    :type accelerator: accelerate.Accelerator(), optional
    :param language_model: Flag indicating an LLM environment, defaults to False.
    :type language_model: bool, optional
    :return: Population of agents after evolution.
    :rtype: list[AgentT]
    """
    if selection_strategy is None:
        return population

    algo = algo or population[0].__class__.__name__

    if language_model:
        elite, population, indices = selection_strategy.select(population)
        if accelerator is None or accelerator.is_main_process:
            population = mutation.mutation(population, indices=indices)
        if accelerator is not None:
            accelerator.wait_for_everyone()
            # This branch only runs for LLM populations.
            consolidate_mutations(
                [agent for agent in population if isinstance(agent, LLMAlgorithm)]
            )
            accelerator.wait_for_everyone()
        if save_elite:
            assert isinstance(elite, LLMAlgorithm), (
                "LLM checkpoints require an LLMAlgorithm elite."
            )
            save_llm_checkpoint(elite, elite_path)
        return population

    elite = None
    if accelerator is not None:
        # Unwrap every model from the accelerator before selecting and mutating.
        accel_temp_models_path = f"models/{env_name}"
        if accelerator.is_main_process:
            Path(accel_temp_models_path).mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
        for model in population:
            model.unwrap_models()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            elite, population, indices = selection_strategy.select(population)
            population = mutation.mutation(population, indices=indices)
            for pop_i, model in enumerate(population):
                model.save_checkpoint(f"{accel_temp_models_path}/{algo}_{pop_i}.pt")
        accelerator.wait_for_everyone()

        # Load the evolved models back onto the worker processes.
        if not accelerator.is_main_process:
            for pop_i, model in enumerate(population):
                model.load_checkpoint(f"{accel_temp_models_path}/{algo}_{pop_i}.pt")
        accelerator.wait_for_everyone()

        # Wrap the models back onto the accelerator.
        for model in population:
            model.wrap_models()
    else:
        elite, population, indices = selection_strategy.select(population)
        population = mutation.mutation(population, indices=indices)

    if save_elite and elite is not None:
        _save_standard_elite(elite, env_name=env_name, algo=algo, elite_path=elite_path)

    return population


def tournament_selection_and_mutation(
    population: PopulationType,
    tournament: TournamentSelection,
    mutation: Mutations,
    env_name: str,
    algo: str | None = None,
    elite_path: str | None = None,
    save_elite: bool = False,
    accelerator: Accelerator | None = None,
    language_model: bool | None = False,
) -> PopulationType:
    """Deprecated. Use :func:`run_selection_and_mutation` instead.


    :param population: Population of agents.
    :type population: list[PopulationType]
    :param tournament: Tournament selection object.
    :type tournament: TournamentSelection
    :param mutation: Mutation object.
    :type mutation: Mutations
    :param env_name: Environment name.
    :type env_name: str
    :param algo: Algorithm name, defaults to None.
    :type algo: str, optional
    :param elite_path: Path to save the elite agent, defaults to None.
    :type elite_path: str, optional
    :param save_elite: Flag to save the elite agent, defaults to False.
    :type save_elite: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None.
    :type accelerator: accelerate.Accelerator(), optional
    :param language_model: Flag to indicate a language model environment, defaults to False.
    :type language_model: bool, optional
    :return: Population of agents after tournament selection and mutation.
    :rtype: list[PopulationType]
    """
    warnings.warn(
        "tournament_selection_and_mutation is deprecated; use "
        "run_selection_and_mutation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_selection_and_mutation(
        tournament,
        population=population,
        mutation=mutation,
        env_name=env_name,
        algo=algo,
        elite_path=elite_path,
        save_elite=save_elite,
        accelerator=accelerator,
        language_model=bool(language_model),
    )


def resolve_selection_strategy(
    selection_strategy: SelectionStrategyProtocol | None,
    tournament: TournamentSelection | None,
) -> SelectionStrategyProtocol | None:
    """Fold the deprecated tournament trainer argument into selection_strategy.

    :param selection_strategy: The selection strategy passed via the new argument.
    :type selection_strategy: SelectionStrategyProtocol | None
    :param tournament: The strategy passed via the deprecated tournament argument.
    :type tournament: TournamentSelection | None
    :return: The resolved selection strategy.
    :rtype: SelectionStrategyProtocol | None
    """
    if tournament is None:
        return selection_strategy
    warnings.warn(
        "The 'tournament' argument to the AgileRL trainers is deprecated; pass the "
        "selection strategy via 'selection_strategy' instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    if selection_strategy is None:
        return tournament
    if selection_strategy is not tournament:
        warnings.warn(
            "Both 'selection_strategy' and the deprecated 'tournament' argument were "
            "provided; ignoring 'tournament'.",
            stacklevel=3,
        )
    return selection_strategy


def init_wandb(
    algo: str,
    env_name: str,
    init_hyperparams: dict[str, Any] | None = None,
    mutation_hyperparams: dict[str, Any] | None = None,
    wandb_api_key: str | None = None,
    accelerator: Accelerator | None = None,
    project: str = "AgileRL",
    addl_args: dict[str, Any] | None = None,
) -> None:
    """Initialize wandb for logging hyperparameters and run metadata.

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
    :type accelerator: accelerate.Accelerator(), optional
    :param addl_args: Additional kwargs to pass to wandb.init()
    :type addl_args: dict, optional
    """
    api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        warnings.warn(
            "No Weights & Biases API key found (pass wandb_api_key or set the "
            "WANDB_API_KEY environment variable); wandb may prompt interactively "
            "or fail to log.",
            UserWarning,
            stacklevel=2,
        )

    config_dict = {}
    if init_hyperparams is not None:
        config_dict.update(init_hyperparams)
    if mutation_hyperparams is not None:
        config_dict.update(mutation_hyperparams)

    # track hyperparameters and run metadata
    kwargs: dict[str, Any] = {
        "config": config_dict,
        "project": project,  # wandb project where this run will be logged
    }
    # A WANDB_NAME environment override wins; only generate a default run name
    # when it is unset, since passing name explicitly would silently ignore it.
    if not os.environ.get("WANDB_NAME"):
        kwargs["name"] = "{}-EvoHPO-{}-{}".format(
            env_name,
            algo,
            datetime.now().strftime("%m%d%Y%H%M%S"),
        )

    if addl_args is not None:
        kwargs.update(addl_args)

    if accelerator is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            wandb.init(**kwargs)
        accelerator.wait_for_everyone()
    else:
        wandb.init(**kwargs)


@accept_flat_kwargs
def init_loggers(
    experiment: LoggerExperiment,
    pbar: tqdm.tqdm,
    logging: TrainLoggingConfig | None = None,
    accelerator: Accelerator | None = None,
) -> list:
    """Build the list of loggers for a training run.

    :param experiment: Algorithm, environment, and hyperparameter identity.
    :type experiment: LoggerExperiment
    :param pbar: ``tqdm`` progress bar for :class:`~agilerl.logger.StdOutLogger`.
    :type pbar: tqdm.tqdm
    :param logging: W&B, TensorBoard, CSV, and stdout destinations.
    :type logging: TrainLoggingConfig, optional
    :param accelerator: HuggingFace Accelerator for distributed training, defaults to None
    :type accelerator: Accelerator | None
    :returns: List of configured :class:`~agilerl.logger.Logger` instances.
    :rtype: list[Logger]
    """
    logging = logging or TrainLoggingConfig()
    algo = experiment.algo
    env_name = experiment.env_name
    init_hyperparams = experiment.init_hyperparams
    mutation_hyperparams = experiment.mutation_hyperparams
    verbose = logging.verbose
    wb = logging.wb
    tensorboard = logging.tensorboard
    csv = logging.csv
    tensorboard_log_dir = logging.tensorboard_log_dir
    csv_log_dir = logging.csv_log_dir
    wandb_api_key = logging.wandb_api_key
    wandb_kwargs = logging.wandb_kwargs

    loggers = []
    if verbose:
        loggers.append(StdOutLogger(pbar, accelerator))

    if wb:
        init_wandb_kwargs: dict[str, Any] = {
            "algo": algo,
            "env_name": env_name,
            "init_hyperparams": init_hyperparams,
            "mutation_hyperparams": mutation_hyperparams,
            "wandb_api_key": wandb_api_key,
            "accelerator": accelerator,
        }
        if wandb_kwargs is not None:
            init_wandb_kwargs.update(wandb_kwargs)
        init_wandb(**init_wandb_kwargs)
        loggers.append(WandbLogger(accelerator))

    if tensorboard:
        experiment_name = f"{algo}-{env_name}"
        loggers.append(
            TensorboardLogger(
                log_dir=tensorboard_log_dir,
                experiment_name=experiment_name,
                accelerator=accelerator,
            )
        )
    if csv:
        if csv_log_dir is None:
            msg = "csv_log_dir must be provided when csv=True"
            raise ValueError(msg)
        loggers.append(CSVLogger(csv_log_dir))

    return loggers


def calculate_vectorized_scores(
    rewards: npt.NDArray,
    terminations: npt.NDArray,
    include_unterminated: bool = False,
    only_first_episode: bool = True,
) -> list[float]:
    """Calculate the vectorized scores for episodes based on rewards and terminations.

    :param rewards: Array of rewards for each environment.
    :type rewards: npt.NDArray
    :param terminations: Array indicating termination points for each environment.
    :type terminations: npt.NDArray
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
                rewards[env_index, start_index : termination_index + 1],
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
    """Print current hyperparameters of agents in a population and their fitnesses.

    :param pop: Population of agents
    :type pop: list[EvolvableAlgorithm]
    """
    for agent in pop:
        mean_fitness = (
            np.mean([np.mean(f) for f in agent.fitness[-5:]]).item()
            if len(agent.fitness) > 0
            else float("nan")
        )
        # GraMa scores are operator-internal state, not a hyperparameter.
        # Exclude them from this display only.
        attrs = EvolvableAlgorithm.inspect_attributes(agent, exclude=("grama_scores",))
        lines = [
            f"Agent ID: {agent.index}  |  Mean 5 Fitness: {mean_fitness:.2f}",
            "Attributes:",
            *[f"  {k}: {v}" for k, v in sorted(attrs.items())],
        ]
        print("\n".join(lines) + "\n")


def get_env_defined_actions(
    info: InfosDict,
    agents: list[str],
) -> dict[str, Any] | None:
    """Get the environment-defined actions for a list of agents.

    :param info: Info dictionary
    :type info: InfosDict
    :param agents: List of agents
    :type agents: list[str]
    :return: Environment-defined actions
    :rtype: dict[str, Any]
    """
    env_defined_actions = {
        agent: info[agent].get("env_defined_action", None) for agent in agents
    }

    if all(eda is None for eda in env_defined_actions.values()):
        return None

    return env_defined_actions


def save_llm_checkpoint(
    agent: LLMAlgorithm,
    checkpoint_path: str | None,
) -> None:
    """Checkpoint the LLM, saving LoRA adapter weights via HuggingFace ``save_pretrained``.

    The saved directory can be reloaded with::

        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained("<base-model-name>")
        model = PeftModel.from_pretrained(base_model, "<checkpoint_path>/actor/")

    :param agent: Agent
    :type agent: LLMAlgorithm
    :param checkpoint_path: Checkpoint path — used as-is (no algo sub-directory is appended).
        Defaults to ``"./saved_checkpoints"`` when ``None``.
    :type checkpoint_path: str
    """
    assert agent.actor is not None, "Actor is not initialized"
    path = "./saved_checkpoints" if checkpoint_path is None else checkpoint_path
    Path(path).mkdir(parents=True, exist_ok=True)
    if agent.accelerator is not None:
        agent.accelerator.wait_for_everyone()
        agent.save_checkpoint(path)
        agent.accelerator.wait_for_everyone()
    else:
        agent.save_checkpoint(path)


def consolidate_mutations(population: list[LLMAlgorithm]) -> None:
    """Consolidate mutations across processes during LLM fintuning.

    :param population: Population of agents
    :type population: list[EvolvableAlgorithm]
    """
    if not isinstance(population[0], LLMAlgorithm):
        warnings.warn(
            "Consolidate mutations is only supported for LLMAlgorithm.", stacklevel=2
        )
        return
    for agent in population:
        assert agent.actor is not None, "Actor is not initialized"
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

        if mut in ("lr", "critic_lr"):
            assert agent.optimizer is not None, "Optimizer is not initialized"
            opt = (
                agent.optimizer
                if not isinstance(agent.optimizer.optimizer, DummyOptimizer)
                # DeepSpeed engines expose the wrapped optimizer on the actor.
                else agent.actor.optimizer
            )
            lr = (
                (agent.lr, agent.lr_critic)
                if getattr(agent, "lr_critic", None) is not None
                else agent.lr
            )
            update_lr_kw: dict[str, Any] = {
                "optimizer": opt,
                "lr": lr,
                "accelerator": agent.accelerator,
                "scheduler_config": agent.cosine_lr_schedule_config,
            }
            agent.accelerator, agent.lr_scheduler = LLMAlgorithm.update_lr(
                **update_lr_kw
            )


def _distributed_world_size(accelerator: Accelerator | None) -> int:
    """World size for batch accounting: prefer Accelerate, else torch.distributed."""
    if accelerator is not None:
        return accelerator.num_processes
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _distributed_rank(accelerator: Accelerator | None) -> int:
    """Process rank (e.g. for seed decorrelation): prefer Accelerate, else torch.distributed."""
    if accelerator is not None:
        return accelerator.process_index
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def data_parallel_topology(
    accelerator: Accelerator | None,
    processes_per_replica: int = 1,
) -> tuple[int, int]:
    """Return this process's ``(rank, world_size)`` among data-parallel replicas.

    Training here is data-parallel only; a replica spans more than one process
    solely because the *generation* engine shards a model across contiguous ranks
    (vLLM's ``tensor_parallel_size``). Those ranks must be handed the same data,
    so anything that splits work across replicas — dataset shards, reset seeds,
    effective batch size — counts replicas rather than processes, and only agrees
    with the process rank when a replica is one process.

    :param accelerator: Accelerator whose process group defines the topology.
    :type accelerator: Accelerator | None
    :param processes_per_replica: Contiguous processes making up one model
        replica (the generation engine's shard count; ``1`` when unsharded).
    :type processes_per_replica: int
    :returns: This replica's index, and the number of replicas.
    :rtype: tuple[int, int]
    """
    world_size = _distributed_world_size(accelerator)
    rank = _distributed_rank(accelerator)
    shards = max(1, int(processes_per_replica))
    if shards == 1:
        return rank, world_size
    if world_size % shards:
        msg = (
            f"processes_per_replica={shards} does not divide the {world_size}-process "
            "group, so replicas would straddle it; every replica must be a whole "
            "number of processes."
        )
        raise ValueError(msg)
    return rank // shards, world_size // shards
