# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for :mod:`agilerl.training.trainer`."""

from __future__ import annotations

import warnings
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Protocol

from accelerate import Accelerator
from gymnasium import spaces

from agilerl.algorithms.core.base import (
    LLMAlgorithm,
    MultiAgentRLAlgorithm,
    RLAlgorithm,
)
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.replay_buffer import BufferType
from agilerl.hpo.multi_frequency import MultiFrequencySelection
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.llm_envs import DatasetEnv, RolloutHarness
from agilerl.models.algo import (
    AlgoSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationSpec,
    SelectionStrategySpec,
    TournamentSelectionSpec,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.protocols import BanditEnvProtocol, SelectionStrategyProtocol
from agilerl.utils.env_utils import GymEnvType, PzEnvType

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


LLMEnvType = RolloutHarness | DatasetEnv
# Union of every env type an ``EnvSpec.make_env`` builds: vectorized gym/pettingzoo
# envs, a bandit env satisfying ``BanditEnvProtocol``, or an LLM env.
EnvironmentType = GymEnvType | PzEnvType | BanditEnvProtocol | LLMEnvType
PopulationType = list[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]


class _SingleAgentVectorEnv(Protocol):
    """Vectorized single-agent env exposing batch-less spaces as attributes."""

    single_observation_space: spaces.Space
    single_action_space: spaces.Space


class _MultiAgentVectorEnv(Protocol):
    """Vectorized multi-agent env exposing per-agent batch-less spaces."""

    agents: list[str]

    def single_observation_space(self, agent: str) -> spaces.Space: ...

    def single_action_space(self, agent: str) -> spaces.Space: ...


def hp_config_from_mutation_spec(spec: MutationSpec) -> HyperparameterConfig | None:
    """Convert :class:`MutationSpec.rl_hp_selection` to a :class:`HyperparameterConfig`.

    :param spec: Mutation specification containing RL HP ranges.
    :returns: A :class:`HyperparameterConfig`, or ``None`` if no HP ranges.
    """
    if not spec.rl_hp_selection:
        return None

    rl_params: dict[str, Any] = {
        name: RLParameter(
            min=hp.min,
            max=hp.max,
            grow_factor=hp.grow_factor,
            shrink_factor=hp.shrink_factor,
        )
        for name, hp in spec.rl_hp_selection.items()
    }
    return HyperparameterConfig(**rl_params)


@singledispatch
def get_spaces_from_env(
    algo_spec: AlgoSpec, env: GymEnvType | PzEnvType
) -> tuple[
    spaces.Space | dict[str, spaces.Space],
    spaces.Space | dict[str, spaces.Space],
]:
    """Get the observation and action spaces from the environment.

    :param algo_spec: Algorithm spec.
    :type algo_spec: AlgoSpec
    :param env: Environment.
    :type env: GymEnvType | PzEnvType
    """
    msg = f"Algorithm spec type {type(algo_spec)} not supported."
    raise NotImplementedError(msg)


@get_spaces_from_env.register(MultiAgentRLAlgorithmSpec)
def get_spaces_from_env_multi_agent(
    algo_spec: MultiAgentRLAlgorithmSpec,
    env: _MultiAgentVectorEnv,
) -> tuple[dict[str, spaces.Space], dict[str, spaces.Space]]:
    """Get the observation and action spaces from the environment for a multi-agent algorithm.

    :param algo_spec: Algorithm spec.
    :type algo_spec: MultiAgentRLAlgorithmSpec
    :param env: Vectorized multi-agent environment.
    :type env: _MultiAgentVectorEnv
    :returns: A tuple of observation and action spaces.
    :rtype: tuple[dict[str, spaces.Space], dict[str, spaces.Space]]
    """
    return {agent: env.single_observation_space(agent) for agent in env.agents}, {
        agent: env.single_action_space(agent) for agent in env.agents
    }


@get_spaces_from_env.register(RLAlgorithmSpec)
def get_spaces_from_env_single_agent(
    algo_spec: RLAlgorithmSpec,
    env: _SingleAgentVectorEnv,
) -> tuple[spaces.Space, spaces.Space]:
    """Get the observation and action spaces from the environment for a single-agent algorithm.

    :param algo_spec: Algorithm spec.
    :type algo_spec: RLAlgorithmSpec
    :param env: Vectorized single-agent environment.
    :type env: _SingleAgentVectorEnv
    :returns: A tuple of observation and action spaces.
    :rtype: tuple[spaces.Space, spaces.Space]
    """
    return env.single_observation_space, env.single_action_space


def create_population_from_spec(
    population_size: int,
    algo_spec: AlgoSpec,
    env: EnvironmentType | None,
    mutation_spec: MutationSpec | None,
    replay_buffer_spec: ReplayBufferSpec | None,
    device: str | torch.device = "cpu",
    resume_from_checkpoint: str | None = None,
    load_weights_from: str | None = None,
    accelerator: Accelerator | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    selection_strategy_spec: SelectionStrategySpec | None = None,
) -> PopulationType:
    """Instantiate a population of agents from an algorithm spec.

    :param population_size: Number of agents to create.
    :type population_size: int
    :param algo_spec: Algorithm spec.
    :type algo_spec: AlgoSpec
    :param mutation_spec: Optional mutation spec for HP range fallback.
    :type mutation_spec: MutationSpec | None
    :param env: RL environment following Gymnasium or PettingZoo API, or
        ``None`` for multi-turn LLM training where no environment is
        instantiated up front.
    :type env: EnvironmentType | None
    :param replay_buffer_spec: Replay buffer specification.
    :type replay_buffer_spec: ReplayBufferSpec | None
    :param device: Torch device string.
    :type device: str | torch.device
    :param resume_from_checkpoint: Checkpoint to continue an interrupted run from,
        restoring optimizer state and the hyperparameters it belongs to. Mutually
        exclusive with ``load_weights_from``.
    :type resume_from_checkpoint: str | None
    :param load_weights_from: Checkpoint to warm-start a new run from, taking only
        the weights. Mutually exclusive with ``resume_from_checkpoint``.
    :type load_weights_from: str | None
    :param accelerator: Accelerator instance.
    :type accelerator: Accelerator | None
    :param tokenizer: Pre-loaded HuggingFace tokenizer for LLM algorithms.
    :type tokenizer: PreTrainedTokenizerBase | None
    :param selection_strategy_spec: Selection-strategy spec. Only MF-PBT
        consumes it here, to tag each agent with its subpopulation.
    :type selection_strategy_spec: SelectionStrategySpec | None
    :returns: A list of algorithm instances.
    :rtype: PopulationType
    """
    from agilerl.models.algorithms import RainbowDQNSpec
    from agilerl.utils.algo_utils import get_num_envs

    # Override the hp_config with the one defined in MutationSpec if not already set
    hp_config = algo_spec.hp_config
    if hp_config is None and mutation_spec is not None:
        hp_config = hp_config_from_mutation_spec(mutation_spec)
        algo_spec.hp_config = hp_config

    # Some algorithms require num_envs as argument -> add to algo_spec
    # NOTE: We should identify these lazily during training...
    for num_envs_arg in ["num_envs", "vect_noise_dim"]:
        if hasattr(algo_spec, num_envs_arg):
            if env is None:
                msg = (
                    f"Algorithm spec {type(algo_spec).__name__} requires an "
                    "environment to resolve num_envs."
                )
                raise ValueError(msg)
            # Not every member of the env union exposes ``num_envs``; the algos
            # that set it always run on a vectorized env that does.
            setattr(algo_spec, num_envs_arg, get_num_envs(env))

    # Classic RL algorithms
    if isinstance(algo_spec, (RLAlgorithmSpec, MultiAgentRLAlgorithmSpec)):
        if env is None:
            msg = "Classic RL algorithms require an instantiated environment."
            raise ValueError(msg)
        observation_space, action_space = get_spaces_from_env(algo_spec, env)

        if (
            isinstance(algo_spec, RainbowDQNSpec)
            and replay_buffer_spec is not None
            and replay_buffer_spec.n_step_buffer
        ):
            algo_spec.n_step = replay_buffer_spec.n_step_buffer_args.n_step

        # ``get_spaces_from_env`` returns a per-agent mapping for multi-agent specs
        # and a plain space for single-agent specs; narrow the shared return here.
        if isinstance(algo_spec, MultiAgentRLAlgorithmSpec):
            ma_error = "Multi-agent specs require per-agent space mappings."
            assert isinstance(observation_space, dict), ma_error
            assert isinstance(action_space, dict), ma_error
            # ``spaces.Dict`` is itself dict-like, so the ``dict`` narrow above leaves
            # an ambiguous value type; rebuild explicit per-agent space mappings.
            obs_by_agent: dict[str, spaces.Space] = {}
            action_by_agent: dict[str, spaces.Space] = {}
            for agent_id, obs_space in observation_space.items():
                assert isinstance(obs_space, spaces.Space)
                obs_by_agent[str(agent_id)] = obs_space
            for agent_id, act_space in action_space.items():
                assert isinstance(act_space, spaces.Space)
                action_by_agent[str(agent_id)] = act_space
            multi_agent_population: PopulationType = [
                algo_spec.build_algorithm(
                    obs_by_agent,
                    action_by_agent,
                    index=i,
                    resume_from_checkpoint=resume_from_checkpoint,
                    device=device,
                    accelerator=accelerator,
                )
                for i in range(population_size)
            ]
            _assign_subpopulations(multi_agent_population, selection_strategy_spec)
            return multi_agent_population

        sa_error = "Single-agent specs require plain observation/action spaces."
        assert not isinstance(observation_space, dict), sa_error
        assert not isinstance(action_space, dict), sa_error
        single_agent_population: PopulationType = [
            algo_spec.build_algorithm(
                observation_space,
                action_space,
                index=i,
                resume_from_checkpoint=resume_from_checkpoint,
                load_weights_from=load_weights_from,
                device=device,
                accelerator=accelerator,
            )
            for i in range(population_size)
        ]
        _assign_subpopulations(single_agent_population, selection_strategy_spec)
        return single_agent_population

    # LLM algorithms — build agent 0 fully, then clone the actor for agents 1..N.
    # Each agent beyond the first gets a fresh Accelerator to avoid sharing the
    # same DeepSpeed distributed context.
    from agilerl.utils.algo_utils import clone_llm
    from agilerl.utils.llm_utils import get_state_dict

    agent_0 = algo_spec.build_algorithm(
        tokenizer=tokenizer,
        index=0,
        resume_from_checkpoint=resume_from_checkpoint,
        load_weights_from=load_weights_from,
        accelerator=accelerator,
        device=device,
    )
    population = [agent_0]

    for i in range(1, population_size):
        agent_accelerator = Accelerator() if accelerator is not None else None
        assert agent_0.actor is not None, "Agent 0 actor is not initialized"
        cloned_actor = clone_llm(
            agent_0.actor,
            zero_stage=getattr(algo_spec, "zero_stage", 0),
            state_dict=(
                agent_0.actor.state_dict()
                if accelerator is None
                else get_state_dict(agent_0.actor)
            ),
        )
        population.append(
            algo_spec.build_algorithm(
                tokenizer=tokenizer,
                index=i,
                resume_from_checkpoint=resume_from_checkpoint,
                load_weights_from=load_weights_from,
                accelerator=agent_accelerator,
                device=device,
                actor_network=cloned_actor,
            )
        )
    _assign_subpopulations(population, selection_strategy_spec)
    return population


def _assign_subpopulations(
    population: PopulationType,
    selection_strategy_spec: SelectionStrategySpec | None,
) -> None:
    """Tag each agent with its MF-PBT subpopulation from its population slot.

    :param population: The freshly-built or resumed population, in slot order.
    :type population: PopulationType
    :param selection_strategy_spec: The resolved selection-strategy spec, or None.
    :type selection_strategy_spec: SelectionStrategySpec | None
    """
    if not isinstance(selection_strategy_spec, MultiFrequencySelectionSpec):
        return
    subpopulation_size = len(population) // selection_strategy_spec.n_subpopulations
    for slot, agent in enumerate(population):
        agent.subpopulation_id = MultiFrequencySelection._subpopulation_for_position(
            slot, subpopulation_size
        )


def build_mutations_from_spec(
    mutation_spec: MutationSpec | None,
    device: str | torch.device = "cpu",
    accelerator: Accelerator | None = None,
) -> Mutations | None:
    """Convert a :class:`MutationSpec` into a :class:`Mutations` instance.

    :param mutation_spec: Mutation specification.
    :type mutation_spec: MutationSpec | None
    :param device: Torch device string.
    :type device: str | torch.device
    :param accelerator: Optional accelerator for distributed mutation operations.
    :type accelerator: Accelerator | None
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
        device=str(device),
        accelerator=accelerator,
    )


def build_tournament_from_spec(
    selection_spec: SelectionStrategySpec | None,
    training_spec: TrainingSpec,
) -> TournamentSelection | None:
    """Convert a :class:`TournamentSelectionSpec` into a :class:`TournamentSelection`.

    :param selection_spec: The resolved selection-strategy spec, or ``None`` when unset.
    :type selection_spec: SelectionStrategySpec | None
    :param training_spec: Training specification.
    :type training_spec: TrainingSpec
    :returns: A :class:`TournamentSelection` instance, or None when tournament
        selection is not the configured strategy.
    :rtype: TournamentSelection | None
    """
    if not isinstance(selection_spec, TournamentSelectionSpec):
        return None

    return TournamentSelection(
        tournament_size=selection_spec.tournament_size,
        elitism=selection_spec.elitism,
        population_size=training_spec.pop_size,
    )


def build_multi_frequency_selection_from_spec(
    selection_spec: SelectionStrategySpec | None,
    training_spec: TrainingSpec,
    seed: int | None = None,
) -> MultiFrequencySelection | None:
    """Convert a spec into a :class:`MultiFrequencySelection` instance.

    :param selection_spec: The resolved selection-strategy spec, or ``None`` when unset.
    :type selection_spec: SelectionStrategySpec | None
    :param training_spec: Training specification supplying the population size.
    :type training_spec: TrainingSpec
    :param seed: The run's global seed, forwarded to the operator's RNG so the
        winner-clone selection varies (reproducibly) with the run seed, defaults to
        None.
    :type seed: int | None
    :returns: A :class:`MultiFrequencySelection` instance, or None when MF-PBT
        is not the configured strategy.
    :rtype: MultiFrequencySelection | None
    """
    if not isinstance(selection_spec, MultiFrequencySelectionSpec):
        return None

    return MultiFrequencySelection(
        population_size=training_spec.pop_size,
        n_subpopulations=selection_spec.n_subpopulations,
        evolution_frequency_ratios=selection_spec.evolution_frequency_ratios,
        n_winners=selection_spec.n_winners,
        n_survivors=selection_spec.n_survivors,
        n_open_for_migration=selection_spec.n_open_for_migration,
        n_losers=selection_spec.n_losers,
        seed=seed,
    )


def build_selection_from_spec(
    selection_spec: SelectionStrategySpec | None,
    training_spec: TrainingSpec,
    seed: int | None = None,
) -> SelectionStrategyProtocol | None:
    """Build the selection operator named by the manifest's selection_strategy block.

    :param selection_spec: The resolved selection-strategy spec, or ``None`` when no
        selection strategy is configured.
    :type selection_spec: SelectionStrategySpec | None
    :param training_spec: Training specification supplying the population size.
    :type training_spec: TrainingSpec
    :param seed: The run's global seed, forwarded to MF-PBT's RNG. Defaults to None.
    :type seed: int | None
    :returns: The selection operator, or ``None`` when *selection_spec* is ``None``.
    :rtype: SelectionStrategyProtocol | None
    :raises TypeError: If *selection_spec* is neither ``None`` nor a known
        selection-strategy spec.
    """
    if selection_spec is None:
        return None

    if isinstance(selection_spec, MultiFrequencySelectionSpec):
        return build_multi_frequency_selection_from_spec(
            selection_spec, training_spec, seed=seed
        )

    if isinstance(selection_spec, TournamentSelectionSpec):
        return build_tournament_from_spec(selection_spec, training_spec)

    msg = (
        "selection_spec must be a TournamentSelectionSpec or a "
        f"MultiFrequencySelectionSpec, got {type(selection_spec).__name__}."
    )
    raise TypeError(msg)


def resolve_deprecated_selection_kwargs(
    selection_strategy: SelectionStrategySpec | None,
    kwargs: dict[str, Any],
    *,
    deprecated_key: str = "tournament",
    caller: str,
) -> SelectionStrategySpec | None:
    """Fold a deprecated selection keyword argument into selection_strategy.

    :param selection_strategy: The spec passed via the current argument, or None
        when unset.
    :type selection_strategy: SelectionStrategySpec | None
    :param kwargs: The catch-all keyword arguments of the calling API.
    :type kwargs: dict[str, Any]
    :param deprecated_key: The superseded keyword the caller still accepts,
        defaults to "tournament".
    :type deprecated_key: str
    :param caller: Name of the calling API, used in the warning and error messages.
    :type caller: str
    :returns: The resolved selection-strategy spec, or None when neither spelling
        supplied one.
    :rtype: SelectionStrategySpec | None
    :raises TypeError: If kwargs holds any key other than deprecated_key.
    :raises ValueError: If both spellings were used and carry different specs.
    """
    unexpected = sorted(key for key in kwargs if key != deprecated_key)
    if unexpected:
        msg = (
            f"{caller} got unexpected keyword argument(s): "
            f"{', '.join(repr(key) for key in unexpected)}."
        )
        raise TypeError(msg)

    if deprecated_key not in kwargs:
        return selection_strategy

    warnings.warn(
        f"The {deprecated_key!r} argument to {caller} is deprecated and will be "
        "removed in a future release; pass the selection strategy via "
        "selection_strategy instead.",
        DeprecationWarning,
        stacklevel=3,
    )

    deprecated_spec = kwargs[deprecated_key]
    if deprecated_spec is None:
        return selection_strategy

    if selection_strategy is None:
        return deprecated_spec

    # Pydantic equality compares class and field values, so equal-but-distinct
    # specs route cleanly while a tournament spec and an MF-PBT spec never match
    if selection_strategy != deprecated_spec:
        msg = (
            f"{caller} received conflicting selection strategies: "
            f"'selection_strategy'={selection_strategy!r} and the deprecated "
            f"{deprecated_key!r}={deprecated_spec!r}. Pass only 'selection_strategy'."
        )
        raise ValueError(msg)

    return selection_strategy


def build_replay_buffer_from_spec(
    algo_spec: AlgoSpec,
    buffer_spec: ReplayBufferSpec | None,
    device: str | torch.device = "cpu",
) -> BufferType | None:
    """Convert a :class:`ReplayBufferSpec` into a :class:`ReplayBuffer`,
    :class:`MultiStepReplayBuffer`, or :class:`PrioritizedReplayBuffer`
    instance, given an algorithm spec.

    A buffer is created for off-policy **and** offline algorithms.
    On-policy algorithms return ``None``.

    :param algo_spec: Algorithm spec.
    :type algo_spec: AlgoSpec
    :param buffer_spec: Replay buffer specification.
    :type buffer_spec: ReplayBufferSpec | None
    :param device: Torch device string.
    :type device: str | torch.device
    :returns: A replay buffer instance, or ``None`` for on-policy algorithms.
    :rtype: BufferType | None
    """
    if not (algo_spec.off_policy or algo_spec.offline or algo_spec.bandit):
        return None

    if buffer_spec is None:
        warnings.warn(
            "No replay buffer specified for off-policy/offline algorithm. "
            "Using default replay buffer with size 100,000.",
            stacklevel=2,
        )
        buffer_spec = ReplayBufferSpec(max_size=100_000)

    return buffer_spec.init_buffer(algo_spec, device)
