"""Helper functions for :mod:`agilerl.training.trainer`."""

from __future__ import annotations

import warnings
from functools import singledispatch
from typing import TYPE_CHECKING, Any

from agilerl.algorithms.core.base import (
    LLMAlgorithm,
    MultiAgentRLAlgorithm,
    RLAlgorithm,
)
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.replay_buffer import (
    MultiAgentReplayBuffer,
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.models.algo import (
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.typing import GymEnvType, PzEnvType
from agilerl.utils.llm_utils import PreferenceGym, ReasoningGym
from agilerl.wrappers.learning import BanditEnv

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator
    from gymnasium import spaces

LLMEnvType = ReasoningGym | PreferenceGym
EnvironmentT = GymEnvType | PzEnvType | BanditEnv | LLMEnvType
AlgoSpecT = RLAlgorithmSpec | LLMAlgorithmSpec | MultiAgentRLAlgorithmSpec
PopulationT = list[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]
BufferT = (
    ReplayBuffer
    | MultiStepReplayBuffer
    | MultiAgentReplayBuffer
    | PrioritizedReplayBuffer
)


def hp_config_from_mutation_spec(spec: MutationSpec) -> HyperparameterConfig | None:
    """Convert :class:`MutationSpec.rl_hp_selection` to a :class:`HyperparameterConfig`.

    :param spec: Mutation specification containing RL HP ranges.
    :returns: A :class:`HyperparameterConfig`, or ``None`` if no HP ranges.
    """
    if not spec.rl_hp_selection:
        return None

    return HyperparameterConfig(
        **{
            name: RLParameter(
                min=hp.min,
                max=hp.max,
                grow_factor=hp.grow_factor,
                shrink_factor=hp.shrink_factor,
            )
            for name, hp in spec.rl_hp_selection.items()
        }
    )


@singledispatch
def get_spaces_from_env(
    algo_spec: AlgoSpecT, env: GymEnvType | PzEnvType
) -> tuple[dict[str, spaces.Space], dict[str, spaces.Space]]:
    """Get the observation and action spaces from the environment.

    :param algo_spec: Algorithm spec.
    :type algo_spec: AlgoSpecT
    :param env: Environment.
    :type env: GymEnvType | PzEnvType
    """
    msg = f"Algorithm spec type {type(algo_spec)} not supported."
    raise NotImplementedError(msg)


@get_spaces_from_env.register(MultiAgentRLAlgorithmSpec)
def get_spaces_from_env_multi_agent(
    algo_spec: MultiAgentRLAlgorithmSpec,
    env: GymEnvType | PzEnvType,
) -> tuple[dict[str, spaces.Space], dict[str, spaces.Space]]:
    """Get the observation and action spaces from the environment for a multi-agent algorithm.

    :param algo_spec: Algorithm spec.
    :type algo_spec: MultiAgentRLAlgorithmSpec
    :param env: Environment.
    :type env: GymEnvType | PzEnvType
    :returns: A tuple of observation and action spaces.
    :rtype: tuple[dict[str, spaces.Space], dict[str, spaces.Space]]
    """
    return {agent: env.single_observation_space(agent) for agent in env.agents}, {
        agent: env.single_action_space(agent) for agent in env.agents
    }


@get_spaces_from_env.register(RLAlgorithmSpec)
def get_spaces_from_env_single_agent(
    algo_spec: RLAlgorithmSpec,
    env: GymEnvType | PzEnvType,
) -> tuple[spaces.Space, spaces.Space]:
    """Get the observation and action spaces from the environment for a single-agent algorithm.

    :param algo_spec: Algorithm spec.
    :type algo_spec: RLAlgorithmSpec
    :param env: Environment.
    :type env: GymEnvType | PzEnvType
    :returns: A tuple of observation and action spaces.
    :rtype: tuple[spaces.Space, spaces.Space]
    """
    return env.single_observation_space, env.single_action_space


def create_population_from_spec(
    population_size: int,
    algo_spec: AlgoSpecT,
    env: EnvironmentT,
    mutation_spec: MutationSpec | None,
    replay_buffer_spec: ReplayBufferSpec | None,
    device: str | torch.device = "cpu",
    resume_from_checkpoint: str | None = None,
    accelerator: Accelerator | None = None,
    tokenizer: Any | None = None,
) -> PopulationT:
    """Instantiate a population of agents from an algorithm spec.

    :param population_size: Number of agents to create.
    :type population_size: int
    :param algo_spec: Algorithm spec.
    :type algo_spec: AlgoSpecT
    :param mutation_spec: Optional mutation spec for HP range fallback.
    :type mutation_spec: MutationSpec | None
    :param env: RL environment following Gymnasium or PettingZoo API.
    :type env: EnvironmentT
    :param replay_buffer_spec: Replay buffer specification.
    :type replay_buffer_spec: ReplayBufferSpec | None
    :param device: Torch device string.
    :type device: str | torch.device
    :param resume_from_checkpoint: Path to resume from checkpoint.
    :type resume_from_checkpoint: str | None
    :param accelerator: Accelerator instance.
    :type accelerator: Accelerator | None
    :param tokenizer: Pre-loaded HuggingFace tokenizer for LLM algorithms.
    :type tokenizer: Any | None
    :returns: A list of algorithm instances.
    :rtype: PopulationT
    """
    from agilerl.models.algorithms import RainbowDQNSpec

    # Override the hp_config with the one defined in MutationSpec if not already set
    hp_config = algo_spec.hp_config
    if hp_config is None and mutation_spec is not None:
        hp_config = hp_config_from_mutation_spec(mutation_spec)
        algo_spec.hp_config = hp_config

    # Some algorithms require num_envs as argument -> add to algo_spec
    # NOTE: We should identify these lazily during training...
    for num_envs_arg in ["num_envs", "vect_noise_dim"]:
        if hasattr(algo_spec, num_envs_arg):
            setattr(algo_spec, num_envs_arg, env.num_envs)

    # Classic RL algorithms
    if isinstance(algo_spec, (RLAlgorithmSpec, MultiAgentRLAlgorithmSpec)):
        observation_space, action_space = get_spaces_from_env(algo_spec, env)

        if (
            isinstance(algo_spec, RainbowDQNSpec)
            and replay_buffer_spec is not None
            and replay_buffer_spec.n_step_buffer
        ):
            algo_spec.n_step = replay_buffer_spec.n_step_buffer_args.n_step

        return [
            algo_spec.build_algorithm(
                observation_space,
                action_space,
                index=i,
                device=device,
                accelerator=accelerator,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            for i in range(population_size)
        ]

    # LLM algorithms
    return [
        algo_spec.build_algorithm(
            tokenizer=tokenizer,
            index=i,
            accelerator=accelerator,
            device=device,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        for i in range(population_size)
    ]


def build_mutations_from_spec(
    mutation_spec: MutationSpec | None, device: str | torch.device = "cpu"
) -> Mutations | None:
    """Convert a :class:`MutationSpec` into a :class:`Mutations` instance.

    :param mutation_spec: Mutation specification.
    :type mutation_spec: MutationSpec | None
    :param device: Torch device string.
    :type device: str | torch.device
    :returns: A :class:`Mutations` instance, or ``None`` if *mutation_spec* is ``None``.
    :rtype: Mutations | None
    """
    if mutation_spec is None:
        return None

    p = mutation_spec.probabilities
    return Mutations(
        no_mutation=p.no_mut,
        architecture=p.arch_mut,
        new_layer_prob=p.new_layer,
        parameters=p.params_mut,
        activation=p.act_mut,
        rl_hp=p.rl_hp_mut,
        mutation_sd=mutation_spec.mutation_sd,
        rand_seed=mutation_spec.rand_seed,
        device=device,
    )


def build_tournament_from_spec(
    tournament_spec: TournamentSelectionSpec | None,
    training_spec: TrainingSpec,
) -> TournamentSelection | None:
    """Convert a :class:`TournamentSelectionSpec` into a :class:`TournamentSelection`.

    :param tournament_spec: Tournament selection specification.
    :type tournament_spec: TournamentSelectionSpec | None
    :param training_spec: Training specification.
    :type training_spec: TrainingSpec
    :returns: A :class:`TournamentSelection` instance, or ``None`` if *tournament_spec* is ``None``.
    :rtype: TournamentSelection | None
    """
    if tournament_spec is None:
        return None

    return TournamentSelection(
        tournament_size=tournament_spec.tournament_size,
        elitism=tournament_spec.elitism,
        population_size=training_spec.population_size,
    )


def build_replay_buffer_from_spec(
    algo_spec: RLAlgorithmSpec | MultiAgentRLAlgorithmSpec,
    buffer_spec: ReplayBufferSpec | None,
    device: str | torch.device = "cpu",
) -> BufferT | None:
    """Convert a :class:`ReplayBufferSpec` into a :class:`ReplayBuffer`,
    :class:`MultiStepReplayBuffer`, :class:`MultiAgentReplayBuffer`, or
    :class:`PrioritizedReplayBuffer` instance, given an algorithm spec.

    A buffer is created for off-policy **and** offline algorithms.
    On-policy algorithms return ``None``.

    :param algo_spec: Algorithm spec.
    :type algo_spec: RLAlgorithmSpec | MultiAgentRLAlgorithmSpec
    :param buffer_spec: Replay buffer specification.
    :type buffer_spec: ReplayBufferSpec | None
    :param device: Torch device string.
    :type device: str | torch.device
    :returns: A replay buffer instance, or ``None`` for on-policy algorithms.
    :rtype: BufferT | None
    """
    needs_buffer = algo_spec.off_policy or algo_spec.offline or algo_spec.bandit
    if not needs_buffer:
        return None

    if buffer_spec is None:
        warnings.warn(
            "No replay buffer specified for off-policy/offline algorithm. "
            "Using default replay buffer with size 100,000.",
            stacklevel=2,
        )
        buffer_spec = ReplayBufferSpec(max_size=100_000)

    return buffer_spec.init_buffer(algo_spec, device)
