"""Trainer abstraction for AgileRL evolutionary training.

Provides :class:`LocalTrainer` for local training and :class:`ArenaTrainer`
for submitting jobs to the Arena RLOps platform.  Both accept Pydantic spec
objects as their primary configuration interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agilerl.models.algo import AlgorithmMeta, RLAlgorithmSpec
from agilerl.models.env import EnvironmentSpec
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.utils.trainer_utils import (
    build_mutations,
    build_replay_buffer,
    build_tournament,
    build_train_kwargs,
    create_population_from_spec,
    get_algo_meta,
    get_training_fn,
    resolve_algo_name,
)
from agilerl.vector import DummyVecEnv

if TYPE_CHECKING:
    import gymnasium as gym

    from agilerl.algorithms.core import EvolvableAlgorithm
    from agilerl.arena.client import ArenaClient
    from agilerl.typing import PopulationType

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """Abstract base trainer for AgileRL evolutionary training.

    :param algorithm: An :class:`~agilerl.algorithms.core.EvolvableAlgorithm`
        instance, an :class:`RLAlgorithmSpec`, or a string algorithm name
        (e.g. ``"PPO"``).
    :type algorithm: EvolvableAlgorithm | RLAlgorithmSpec | str
    :param environment: A ``gymnasium.Env`` instance, an
        :class:`EnvironmentSpec`, or an env-name string (for Arena).
    :type environment: gym.Env | EnvironmentSpec | str
    :param training: Training loop parameters (max steps, population size,
        etc.).  Defaults to :class:`TrainingSpec` with sensible values.
    :type training: TrainingSpec | None
    :param mutation: Mutation probabilities and RL-HP ranges.
    :type mutation: MutationSpec | None
    :param tournament: Tournament selection configuration.
    :type tournament: TournamentSelectionSpec | None
    :param replay_buffer: Replay buffer configuration.  Off-policy algorithms
        auto-create a default buffer when this is ``None``.
    :type replay_buffer: ReplayBufferSpec | None
    :param device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    :type device: str
    """

    def __init__(
        self,
        algorithm: EvolvableAlgorithm | RLAlgorithmSpec | str,
        environment: gym.Env | EnvironmentSpec | str,
        *,
        training: TrainingSpec | None = None,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferSpec | None = None,
        device: str = "cpu",
    ) -> None:
        self.algorithm = algorithm
        self.environment = environment
        self.training = training or TrainingSpec()
        self.mutation = mutation
        self.tournament = tournament
        self.replay_buffer = replay_buffer
        self.device = device

        self._algo_name = resolve_algo_name(algorithm)
        self._algo_meta: AlgorithmMeta = get_algo_meta(self._algo_name)

    @abstractmethod
    def train(self) -> Any:
        """Run the training loop.  Return type varies by subclass."""
        ...


class LocalTrainer(Trainer):
    """Local trainer that streamlines the AgileRL evolutionary training process.

    :param algorithm: An :class:`~agilerl.algorithms.core.EvolvableAlgorithm`
        instance (cloned to build the population), an
        :class:`RLAlgorithmSpec` (used to construct agents from scratch),
        or a string name (uses the default spec for that algorithm).
    :type algorithm: EvolvableAlgorithm | RLAlgorithmSpec | str
    :param environment: A ``gymnasium.Env`` instance.
    :type environment: gym.Env
    :param population: Pre-built population list.  When provided,
        *algorithm* is used only for name resolution.
    :type population: PopulationType | None
    :param training: Training loop parameters.  Defaults to
        :class:`TrainingSpec` with sensible values.
    :type training: TrainingSpec | None
    :param mutation: Mutation probabilities and RL-HP ranges.
    :type mutation: MutationSpec | None
    :param tournament: Tournament selection configuration.
    :type tournament: TournamentSelectionSpec | None
    :param replay_buffer: Replay buffer configuration.
    :type replay_buffer: ReplayBufferSpec | None
    :param verbose: Print progress during training.
    :type verbose: bool
    :param accelerator: Optional HuggingFace ``Accelerator``.
    :type accelerator: Any | None
    :param checkpoint: Save a checkpoint every *n* steps (``None`` to
        disable).
    :type checkpoint: int | None
    :param checkpoint_path: Directory for checkpoint files.
    :type checkpoint_path: str | None
    :param save_elite: Save the elite agent after training.
    :type save_elite: bool
    :param elite_path: Directory for elite agent files.
    :type elite_path: str | None
    :param wb: Enable Weights & Biases logging.
    :type wb: bool
    :param tensorboard: Enable TensorBoard logging.
    :type tensorboard: bool
    :param tensorboard_log_dir: Directory for TensorBoard logs.
    :type tensorboard_log_dir: str | None
    :param wandb_api_key: W&B API key.
    :type wandb_api_key: str | None
    :param wandb_kwargs: Extra kwargs forwarded to ``wandb.init``.
    :type wandb_kwargs: dict[str, Any] | None
    :param swap_channels: Whether to swap observation channels (e.g. for
        Atari image observations).
    :type swap_channels: bool
    :param env_name: Human-readable environment name forwarded to the
        training loop for logging.
    :type env_name: str | None
    :param device: Torch device string.
    :type device: str
    """

    def __init__(
        self,
        algorithm: EvolvableAlgorithm | RLAlgorithmSpec | str,
        environment: gym.Env,
        *,
        population: PopulationType | None = None,
        training: TrainingSpec | None = None,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferSpec | None = None,
    ) -> None:
        if not hasattr(environment, "num_envs"):
            environment = DummyVecEnv(environment)

        super().__init__(
            algorithm,
            environment,
            training=training,
            mutation=mutation,
            tournament=tournament,
            replay_buffer=replay_buffer,
        )

    def train(
        self,
        *,
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
        env_name: str | None = None,
        device: str = "cpu",
    ) -> tuple[PopulationType, list[list[float]]]:
        """Run local training.

        :returns: A tuple of ``(population, fitness_history)`` where
            *population* is the final evolved population and
            *fitness_history* is a list of per-generation fitness scores.
        :rtype: tuple[PopulationType, list[list[float]]]
        """
        pop = self._resolve_population()
        mutations = build_mutations(self.mutation, self.device)
        tournament = build_tournament(self.tournament, self.training)
        memory = build_replay_buffer(self.replay_buffer, self._algo_meta, self.device)
        train_fn = get_training_fn(self._algo_meta)

        env_name = self.env_name or str(self.environment)

        kwargs = build_train_kwargs(
            algo_meta=self._algo_meta,
            algo_name=self._algo_name,
            env=self.environment,
            env_name=env_name,
            pop=pop,
            training=self.training,
            tournament=tournament,
            mutations=mutations,
            memory=memory,
            swap_channels=self.swap_channels,
            checkpoint=self.checkpoint,
            checkpoint_path=self.checkpoint_path,
            save_elite=self.save_elite,
            elite_path=self.elite_path,
            wb=self.wb,
            tensorboard=self.tensorboard,
            tensorboard_log_dir=self.tensorboard_log_dir,
            verbose=self.verbose,
            accelerator=self.accelerator,
            wandb_api_key=self.wandb_api_key,
            wandb_kwargs=self.wandb_kwargs,
        )

        return train_fn(**kwargs)

    def _resolve_population(self) -> PopulationType:
        """Build or return the agent population.

        Resolution order:

        1. Pre-built *population* (returned as-is).
        2. :class:`EvolvableAlgorithm` instance (cloned *pop_size* times).
        3. :class:`RLAlgorithmSpec` (constructs agents from the spec).
        4. String name (uses the default spec from the registry).

        :returns: A list of algorithm instances.
        :rtype: PopulationType
        :raises TypeError: If *algorithm* is not a supported type and no
            *population* was provided.
        """
        if self.population is not None:
            return self.population

        from agilerl.algorithms.core import EvolvableAlgorithm

        if isinstance(self.algorithm, EvolvableAlgorithm):
            return [
                self.algorithm.clone(index=i) for i in range(self.training.pop_size)
            ]

        if isinstance(self.algorithm, RLAlgorithmSpec):
            return create_population_from_spec(
                self.algorithm,
                self._algo_meta,
                self.environment,
                self.training.pop_size,
                self.device,
            )

        if isinstance(self.algorithm, str):
            spec = self._algo_meta.spec_cls()
            return create_population_from_spec(
                spec,
                self._algo_meta,
                self.environment,
                self.training.pop_size,
                self.device,
            )

        msg = (
            f"Cannot create population from algorithm of type "
            f"{type(self.algorithm).__name__}. Pass an EvolvableAlgorithm "
            f"instance, an RLAlgorithmSpec, a string name, or a pre-built "
            f"population via the 'population' parameter."
        )
        raise TypeError(msg)


class ArenaTrainer(Trainer):
    """Submits evolutionary training jobs to the Arena RLOps platform.

    Builds an :class:`~agilerl.models.ArenaTrainingManifest` from the
    provided specs and submits it via
    :class:`~agilerl.arena.client.ArenaClient`.

    :param algorithm: An :class:`RLAlgorithmSpec` or a string algorithm
        name (uses the default spec from the registry).
    :type algorithm: RLAlgorithmSpec | str
    :param environment: An :class:`EnvironmentSpec` or a string env name.
    :type environment: EnvironmentSpec | str
    :param training: Training loop parameters.  Defaults to
        :class:`TrainingSpec` with sensible values.
    :type training: TrainingSpec | None
    :param mutation: Mutation probabilities and RL-HP ranges.
    :type mutation: MutationSpec | None
    :param tournament: Tournament selection configuration.
    :type tournament: TournamentSelectionSpec | None
    :param replay_buffer: Replay buffer configuration.
    :type replay_buffer: ReplayBufferSpec | None
    :param client: An authenticated :class:`ArenaClient`.  One is created
        automatically if not provided.
    :type client: ArenaClient | None
    :param device: Torch device string.
    :type device: str
    """

    def __init__(
        self,
        algorithm: RLAlgorithmSpec | str,
        environment: EnvironmentSpec | str,
        *,
        training: TrainingSpec | None = None,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferSpec | None = None,
        client: ArenaClient | None = None,
        device: str = "cpu",
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

        if client is not None:
            self._client = client
        else:
            from agilerl.arena.client import ArenaClient as _ArenaClient

            self._client = _ArenaClient()

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
        from agilerl.models import ArenaTrainingManifest

        algo_spec = self._resolve_algo_spec()
        env_spec = self._resolve_env_spec()

        return ArenaTrainingManifest(
            algorithm=algo_spec,
            environment=env_spec,
            mutation=self.mutation,
            network=algo_spec.net_config,
            replay_buffer=self.replay_buffer,
            tournament_selection=self.tournament,
            training=self.training,
        )

    def _resolve_algo_spec(self) -> RLAlgorithmSpec:
        """Return the algorithm as an :class:`RLAlgorithmSpec`.

        If the algorithm was provided as a string, the default spec for
        that algorithm is created from the registry.

        :returns: Algorithm spec instance.
        :rtype: RLAlgorithmSpec
        """
        if isinstance(self.algorithm, RLAlgorithmSpec):
            return self.algorithm
        return self._algo_meta.spec_cls()

    def _resolve_env_spec(self) -> EnvironmentSpec:
        """Return the environment as an :class:`EnvironmentSpec`.

        :returns: Environment spec instance.
        :rtype: EnvironmentSpec
        :raises TypeError: If the environment is not a string or
            :class:`EnvironmentSpec`.
        """
        if isinstance(self.environment, EnvironmentSpec):
            return self.environment
        if isinstance(self.environment, str):
            return EnvironmentSpec(name=self.environment)
        msg = (
            f"ArenaTrainer requires an EnvironmentSpec or a string env name, "
            f"got {type(self.environment).__name__}"
        )
        raise TypeError(msg)
