"""Helper functions for :mod:`agilerl.trainer`.

Each function converts a Pydantic spec into the corresponding runtime
object, keeping the Trainer classes lean and testable.
"""

from __future__ import annotations

import warnings
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

if TYPE_CHECKING:
    import torch

EnvironmentT = GymEnvType | PzEnvType
AlgoSpecT = RLAlgorithmSpec | LLMAlgorithmSpec | MultiAgentRLAlgorithmSpec
PopulationT = list[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]


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


def create_population_from_spec(
    pop_size: int,
    algo_spec: AlgoSpecT,
    mutation_spec: MutationSpec | None,
    env: GymEnvType | PzEnvType,
    device: str | torch.device = "cpu",
) -> PopulationT:
    """Instantiate a population of agents from an algorithm spec.

    :param pop_size: Number of agents to create.
    :type pop_size: int
    :param algo_spec: Algorithm spec.
    :type algo_spec: AlgoSpecT
    :param mutation_spec: Optional mutation spec for HP range fallback.
    :type mutation_spec: MutationSpec | None
    :param env: RL environment following Gymnasium or PettingZoo API.
    :type env: GymEnvType | PzEnvType
    :param device: Torch device string.
    :type device: str | torch.device
    :returns: A list of algorithm instances.
    :rtype: PopulationT
    """
    # Override the hp_config with the one defined in MutationSpec if not already set
    hp_config = algo_spec.hp_config
    if hp_config is None and mutation_spec is not None:
        hp_config = hp_config_from_mutation_spec(mutation_spec)
        algo_spec.hp_config = hp_config

    # Some algorithms require num_envs as argument -> add to algo_spec
    for num_envs_arg in ["num_envs", "vect_noise_dim"]:
        if hasattr(algo_spec, num_envs_arg):
            setattr(algo_spec, num_envs_arg, env.num_envs)

    # Classic RL algorithms
    if isinstance(algo_spec, (RLAlgorithmSpec, MultiAgentRLAlgorithmSpec)):
        return [
            algo_spec.build_algorithm(
                env.single_observation_space, env.single_action_space, index, device
            )
            for index in range(pop_size)
        ]

    # TODO: Add support for LLMAlgorithmSpec
    msg = f"Algorithm spec type {type(algo_spec)} not supported."
    raise NotImplementedError(msg)


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
        population_size=training_spec.pop_size,
    )


def build_replay_buffer_from_spec(
    algo_spec: RLAlgorithmSpec | MultiAgentRLAlgorithmSpec,
    buffer_spec: ReplayBufferSpec | None,
    device: str | torch.device = "cpu",
) -> ReplayBuffer | MultiStepReplayBuffer | MultiAgentReplayBuffer | None:
    """Convert a :class:`ReplayBufferSpec` into a :class:`ReplayBuffer`.

    :param algo_spec: Algorithm spec.
    :type algo_spec: RLAlgorithmSpec | MultiAgentRLAlgorithmSpec
    :param buffer_spec: Replay buffer specification.
    :type buffer_spec: ReplayBufferSpec | None
    :param device: Torch device string.
    :type device: str | torch.device
    :returns: A :class:`ReplayBuffer` or :class:`MultiStepReplayBuffer` instance, or ``None`` if *buffer_spec* is ``None``.
    :rtype: ReplayBuffer | MultiStepReplayBuffer | None
    """
    if buffer_spec is None:
        if algo_spec.off_policy:
            warnings.warn(
                "No replay buffer specified for off-policy algorithm. Using default replay buffer with size 100,000.",
                stacklevel=2,
            )
            return ReplayBuffer(max_size=100_000, device=device)

        return None

    # TODO: Add support for Prioritized / MultiAgentReplayBuffer

    if buffer_spec.n_step_buffer:
        return MultiStepReplayBuffer(
            max_size=buffer_spec.max_size,
            n_step=buffer_spec.n_step_buffer_args.n_step,
            gamma=0.99,
            device=device,
        )
    return ReplayBuffer(max_size=buffer_spec.max_size, device=device)


def build_train_kwargs(
    *,
    algo_spec: AlgoSpecT,
    env: GymEnvType | PzEnvType,
    env_name: str | None = None,
    pop: PopulationT,
    training: TrainingSpec,
    tournament: TournamentSelection | None,
    mutations: Mutations | None,
    memory: ReplayBuffer | None,
    swap_channels: bool = False,
    checkpoint: int | None = None,
    checkpoint_path: str | None = None,
    save_elite: bool = False,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    verbose: bool = True,
    accelerator: Any | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the keyword-argument dict for a training function call.

    Conditionally includes ``memory``, ``learning_delay``, and
    ``wandb_kwargs`` based on the algorithm's training loop requirements.
    """
    kwargs: dict[str, Any] = {
        "env": env,
        "env_name": env_name,
        "algo": algo_spec.name,
        "pop": pop,
        "swap_channels": swap_channels,
        "max_steps": training.max_steps,
        "evo_steps": training.evo_steps,
        "eval_loop": training.eval_loop,
        "target": training.target_score,
        "tournament": tournament,
        "mutation": mutations,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "save_elite": save_elite,
        "elite_path": elite_path,
        "wb": wb,
        "tensorboard": tensorboard,
        "tensorboard_log_dir": tensorboard_log_dir,
        "verbose": verbose,
        "accelerator": accelerator,
        "wandb_api_key": wandb_api_key,
    }

    if algo_spec.off_policy:
        kwargs["memory"] = memory
        kwargs["learning_delay"] = training.learning_delay

    return kwargs
