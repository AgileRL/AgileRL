"""Trainer abstraction for AgileRL evolutionary training."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agilerl import HAS_ARENA_DEPENDENCIES
from agilerl.algorithms.core.base import (
    LLMAlgorithm,
    MultiAgentRLAlgorithm,
    RLAlgorithm,
)
from agilerl.models.algo import (
    ALGO_REGISTRY,
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.env import EnvironmentSpec
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.networks import NetworkSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.typing import GymEnvType, PzEnvType
from agilerl.utils.trainer_utils import (
    build_mutations_from_spec,
    build_replay_buffer_from_spec,
    build_tournament_from_spec,
    build_train_kwargs,
    create_population_from_spec,
)
from agilerl.vector import AsyncPettingZooVecEnv

logger = logging.getLogger(__name__)

AlgoSpecT = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec
AlgorithmT = AlgoSpecT | str
EnvironmentT = GymEnvType | PzEnvType | AsyncPettingZooVecEnv
ArenaEnvT = EnvironmentSpec | dict[str, str]
ReplayBufferT = ReplayBufferSpec | None
PopulationT = list[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]

if HAS_ARENA_DEPENDENCIES:
    from agilerl.arena.client import ArenaClient
    from agilerl.models import ArenaTrainingManifest
else:
    ArenaClient = None

if TYPE_CHECKING:
    import torch


class Trainer(ABC):
    """Abstract base trainer for AgileRL evolutionary training.

    :param algorithm: An algorithm spec or a string algorithm name.
    :type algorithm: AlgorithmT
    :param environment: A ``gymnasium.Env`` instance, a PettingZoo ``ParallelEnv`` instance, or an env-name string.
    :type environment: EnvironmentT
    :param training: Training loop parameters (max steps, population size,
        etc.).  Defaults to :class:`TrainingSpec` with sensible values.
    :type training: TrainingSpec | None
    :param mutation: Mutation probabilities and RL-HP ranges.
    :type mutation: MutationSpec | None
    :param tournament: Tournament selection configuration.
    :type tournament: TournamentSelectionSpec | None
    :param replay_buffer: Replay buffer configuration.  Off-policy algorithms
        auto-create a default buffer when this is ``None``.
    :type replay_buffer: ReplayBufferT | None
    :param device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    :type device: str
    """

    def __init__(
        self,
        algorithm: AlgorithmT,
        environment: EnvironmentT,
        training: TrainingSpec | None = None,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferT | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if isinstance(algorithm, str):
            algorithm = ALGO_REGISTRY.get(algorithm).spec_cls()

        self.algorithm: AlgoSpecT = algorithm
        self.environment = environment
        self.training = training or TrainingSpec()
        self.mutation = mutation
        self.tournament = tournament
        self.replay_buffer = replay_buffer
        self.device = device

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> tuple[PopulationT, list[list[float]]]:
        """Run the training loop.

        :returns: A tuple of ``(population, fitness_history)`` where
            *population* is the final evolved population and
            *fitness_history* is a list of per-generation fitness scores.
        :rtype: tuple[PopulationT, list[list[float]]]
        """
        msg = "Trainer subclass must implement train method."
        raise NotImplementedError(msg)


class LocalTrainer(Trainer):
    """Local trainer that streamlines the AgileRL evolutionary training process.

    :param algorithm: An `:class:`AlgorithmSpec` instance or a string algorithm name.
    :type algorithm: AlgorithmSpec | str
    :param environment: An RL environment following Gymnasium or PettingZoo API.
    :type environment: gym.Env | ParallelEnv
    :param training: Training parameters.  Defaults to
        :class:`TrainingSpec` with sensible values.
    :type training: TrainingSpec | None
    :param mutation: Mutation probabilities and RL-HP ranges.  When an
        :class:`RLAlgorithmSpec` is used and ``hp_config`` is not set on it,
        HP ranges are derived from ``mutation.rl_hp_selection``.
    :type mutation: MutationSpec | Mutations | None
    :param tournament: Tournament selection configuration.
    :type tournament: TournamentSelectionSpec | TournamentSelection | None
    :param replay_buffer: Replay buffer configuration.
    :type replay_buffer: ReplayBufferSpec | ReplayBuffer | None
    :param device: Torch device string.
    :type device: str
    """

    def __init__(
        self,
        algorithm: AlgorithmT,
        environment: EnvironmentT,
        training: TrainingSpec | None = None,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferT | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            algorithm,
            environment,
            training=training,
            mutation=mutation,
            tournament=tournament,
            replay_buffer=replay_buffer,
            device=device,
        )

    def train(
        self,
        *,
        env_name: str | None = None,
        verbose: bool = True,
        accelerator: Any | None = None,
        checkpoint: int | None = None,
        checkpoint_path: str | None = None,
        save_elite: bool = False,
        elite_path: str | None = None,
        wb: bool = False,
        tensorboard: bool = False,
        tensorboard_log_dir: str | None = None,
        wandb_api_key: str | None = None,
        wandb_kwargs: dict[str, Any] | None = None,
        swap_channels: bool = False,
    ) -> tuple[PopulationT, list[list[float]]]:
        """Run local training.

        :returns: A tuple of ``(population, fitness_history)`` where
            *population* is the final evolved population and
            *fitness_history* is a list of per-generation fitness scores.
        :rtype: tuple[PopulationT, list[list[float]]]
        """
        population = create_population_from_spec(
            pop_size=self.training.pop_size,
            algo_spec=self.algorithm,
            mutation_spec=self.mutation,
            env=self.environment,
            device=self.device,
        )
        mutations = build_mutations_from_spec(self.mutation, self.device)
        tourn = build_tournament_from_spec(self.tournament, self.training)
        memory = build_replay_buffer_from_spec(
            self.algorithm, self.replay_buffer, self.device
        )
        train_fn = self.algorithm.resolve_training_fn()

        kwargs = build_train_kwargs(
            algo_spec=self.algorithm,
            env=self.environment,
            env_name=env_name,
            pop=population,
            training=self.training,
            tournament=tourn,
            mutations=mutations,
            memory=memory,
            swap_channels=swap_channels,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            save_elite=save_elite,
            elite_path=elite_path,
            wb=wb,
            tensorboard=tensorboard,
            tensorboard_log_dir=tensorboard_log_dir,
            verbose=verbose,
            accelerator=accelerator,
            wandb_api_key=wandb_api_key,
            wandb_kwargs=wandb_kwargs,
        )

        return train_fn(**kwargs)


class ArenaTrainer(Trainer):
    """Submits evolutionary training jobs to the Arena RLOps platform.

    Builds an :class:`~agilerl.models.ArenaTrainingManifest` from the
    provided specs and submits it via
    :class:`~agilerl.arena.client.ArenaClient`.

    :param algorithm: An :class:`RLAlgorithmSpec` or a string algorithm
        name (uses the default spec from the registry).
    :type algorithm: AlgorithmT
    :param environment: An :class:`EnvironmentSpec` or a string env name.
    :type environment: EnvironmentT
    :param training: Training loop parameters.  Defaults to
        :class:`TrainingSpec` with sensible values.
    :type training: TrainingT | None
    :param mutation: Mutation probabilities and RL-HP ranges.
    :type mutation: MutationT | None
    :param tournament: Tournament selection configuration.
    :type tournament: TournamentSelectionT | None
    :param replay_buffer: Replay buffer configuration.
    :type replay_buffer: ReplayBufferT | None
    :param client: An authenticated :class:`ArenaClient`.  One is created
        automatically if not provided.
    :type client: ArenaClient | None
    :param device: Torch device string.
    :type device: str | torch.device
    """

    def __init__(
        self,
        algorithm: AlgorithmT,
        environment: ArenaEnvT,
        *,
        client: ArenaClient | None = None,
        api_key: str | None = None,
        training: TrainingSpec | None = None,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferT | None = None,
    ) -> None:

        super().__init__(
            algorithm,
            environment,
            training=training,
            mutation=mutation,
            tournament=tournament,
            replay_buffer=replay_buffer,
        )

        if client is not None:
            self._client = client
        else:
            if ArenaClient is None:
                msg = "Arena dependencies are not installed. Please install them using: pip install agilerl[arena]"
                raise ImportError(msg)

            self._client = ArenaClient(api_key=api_key)

    def train(self, *, stream: bool = False) -> dict[str, Any]:
        """Build the manifest and submit the training job to Arena.

        :param stream: If ``True``, stream logs to the terminal and block
            until the job finishes.
        :type stream: bool
        :returns: Arena API response including ``job_id`` and ``status``.
            When *stream* is ``True``, returns the final result payload.
        :rtype: dict[str, Any]
        """
        manifest = self.to_manifest()
        return self._client.submit_job(manifest, stream=stream)

    def to_manifest(self) -> Any:
        """Build an :class:`~agilerl.models.ArenaTrainingManifest`.

        Assembles the manifest directly from the spec objects stored on
        this trainer, without any intermediate dict parsing.

        :returns: A fully validated manifest ready for submission.
        :rtype: ArenaTrainingManifest
        """
        env_spec = self._resolve_env_spec()
        net_config: NetworkSpec | None = getattr(self.algorithm, "net_config", None)

        # Validate section-level Arena serialization early.
        self.algorithm.to_manifest()
        env_spec.to_manifest()
        if net_config is not None:
            net_config.to_manifest()
        self.training.to_manifest(name=self.algorithm.resolve_training_fn().__name__)
        if self.mutation is not None:
            self.mutation.model_dump(exclude_none=True)
        if self.replay_buffer is not None:
            self.replay_buffer.model_dump(exclude_none=True)
        if self.tournament is not None:
            self.tournament.model_dump(exclude_none=True)

        return ArenaTrainingManifest(
            algorithm=self.algorithm,
            environment=env_spec,
            mutation=self.mutation,
            network=net_config,
            replay_buffer=self.replay_buffer,
            tournament_selection=self.tournament,
            training=self.training,
        )

    def _resolve_env_spec(self) -> EnvironmentSpec:
        """Return the environment as an :class:`EnvironmentSpec`.

        :returns: Environment spec instance.
        :rtype: EnvironmentSpec
        :raises TypeError: If the environment is not a string or
            :class:`EnvironmentSpec`.
        """
        if isinstance(self.environment, EnvironmentSpec):
            return self.environment
        if isinstance(self.environment, dict):
            return EnvironmentSpec.from_dict(self.environment)
        msg = (
            f"ArenaTrainer requires an EnvironmentSpec or a dictionary with 'name', 'version', and 'num_envs' keys, "
            f"got {type(self.environment).__name__}"
        )
        raise TypeError(msg)
