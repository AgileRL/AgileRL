"""Trainer abstraction for AgileRL evolutionary training.

Provides :class:`LocalTrainer` for local training and :class:`ArenaTrainer`
for submitting jobs to the Arena RLOps platform.  Both accept Pydantic spec
objects as their primary configuration interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agilerl.algorithms.core import (
    EvolvableAlgorithm,
)
from agilerl.arena.client import ArenaClient
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.models.algo import AlgorithmMeta, LLMAlgorithmSpec, RLAlgorithmSpec
from agilerl.models.env import EnvironmentSpec
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.typing import GymEnvType, PopulationType, PzEnvType
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
from agilerl.vector import DummyVecEnv, PzDummyVecEnv

logger = logging.getLogger(__name__)

AlgorithmT = EvolvableAlgorithm | RLAlgorithmSpec | LLMAlgorithmSpec | str
EnvironmentT = GymEnvType | PzEnvType | EnvironmentSpec
ReplayBufferT = ReplayBuffer | MultiAgentReplayBuffer | ReplayBufferSpec
TournamentSelectionT = TournamentSelectionSpec | TournamentSelection
MutationT = MutationSpec | Mutations

if TYPE_CHECKING:
    import torch


class Trainer(ABC):
    """Abstract base trainer for AgileRL evolutionary training.

    :param algorithm: An :class:`~agilerl.algorithms.core.EvolvableAlgorithm`
        instance, an :class:`RLAlgorithmSpec`, or a string algorithm name
        (e.g. ``"PPO"``).
    :type algorithm: AlgorithmType
    :param environment: A ``gymnasium.Env`` instance, an
        :class:`EnvironmentSpec`, or an env-name string (for Arena).
    :type environment: EnvironmentType
    :param training: Training loop parameters (max steps, population size,
        etc.).  Defaults to :class:`TrainingSpec` with sensible values.
    :type training: TrainingT | None
    :param mutation: Mutation probabilities and RL-HP ranges.
    :type mutation: MutationT | None
    :param tournament: Tournament selection configuration.
    :type tournament: TournamentSelectionT | None
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
        mutation: MutationT | None = None,
        tournament: TournamentSelectionT | None = None,
        replay_buffer: ReplayBufferT | None = None,
        device: str | torch.device = "cpu",
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

    @property
    def algo_meta(self) -> AlgorithmMeta:
        """Return the metadata of the algorithm."""
        return self._algo_meta

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
    :param environment: A vectorized ``gymnasium.Env``, a bare ``gym.Env``
        (auto-wrapped with :class:`DummyVecEnv`), a string env name (auto-
        created via :func:`make_vect_envs`), or an :class:`EnvironmentSpec`.
    :type environment: EnvironmentT
    :param population: Pre-built population list.  When provided,
        *algorithm* is used only for name resolution.
    :type population: PopulationType | None
    :param training: Training loop parameters.  Defaults to
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
        population: PopulationType | None = None,
        training: TrainingSpec | None = None,
        mutation: MutationT | None = None,
        tournament: TournamentSelectionT | None = None,
        replay_buffer: ReplayBufferT | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        env = self._resolve_environment(environment)

        super().__init__(
            algorithm,
            env,
            training=training,
            mutation=mutation,
            tournament=tournament,
            replay_buffer=replay_buffer,
            device=device,
        )
        self.population = population
        self._env_name = self._infer_env_name(environment)

    @staticmethod
    def _resolve_environment(environment: EnvironmentT) -> GymEnvType | PzDummyVecEnv:
        """Ensure *environment* is a vectorized environment.

        Resolution order:

        1. String name — created via :func:`make_vect_envs`.
        2. :class:`EnvironmentSpec` — created via :func:`make_vect_envs`
           using the spec's ``name`` and ``num_envs``.
        3. Already-vectorized env (has ``num_envs``) — returned as-is.
        4. PettingZoo ``ParallelEnv`` — wrapped with :class:`PzDummyVecEnv`.
        5. Bare ``gym.Env`` — wrapped with :class:`DummyVecEnv`.

        :param environment: Environment input.
        :type environment: EnvironmentT
        :returns: A vectorized environment.
        :rtype: GymEnvType | PzDummyVecEnv
        """
        from pettingzoo import ParallelEnv

        from agilerl.utils.utils import make_vect_envs

        if isinstance(environment, str):
            return make_vect_envs(env_name=environment)

        if isinstance(environment, EnvironmentSpec):
            return make_vect_envs(
                env_name=environment.name,
                num_envs=environment.num_envs,
            )

        if hasattr(environment, "num_envs"):
            return environment

        if isinstance(environment, ParallelEnv):
            return PzDummyVecEnv(environment)

        return DummyVecEnv(environment)

    @staticmethod
    def _infer_env_name(environment: EnvironmentT) -> str:
        """Extract a human-readable name from the environment input.

        :param environment: Original environment input (before resolution).
        :type environment: EnvironmentT
        :returns: Environment name string.
        :rtype: str
        """
        if isinstance(environment, str):
            return environment
        if isinstance(environment, EnvironmentSpec):
            return environment.name
        spec = getattr(environment, "spec", None)
        if spec is not None and hasattr(spec, "id"):
            return spec.id
        return type(environment).__name__

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
        device: str | torch.device = "cpu",
    ) -> tuple[PopulationType, list[list[float]]]:
        """Run local training.

        :returns: A tuple of ``(population, fitness_history)`` where
            *population* is the final evolved population and
            *fitness_history* is a list of per-generation fitness scores.
        :rtype: tuple[PopulationType, list[list[float]]]
        """
        pop = self._resolve_population()
        mutations = build_mutations(self.mutation, self.device)
        tourn = build_tournament(self.tournament, self.training)
        memory = build_replay_buffer(self.replay_buffer, self._algo_meta, self.device)
        train_fn = get_training_fn(self._algo_meta)

        kwargs = build_train_kwargs(
            algo_meta=self._algo_meta,
            algo_name=self._algo_name,
            env=self.environment,
            env_name=self._env_name,
            pop=pop,
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

        mutation_spec = (
            self.mutation if isinstance(self.mutation, MutationSpec) else None
        )

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
                mutation_spec=mutation_spec,
            )

        if isinstance(self.algorithm, str):
            spec = self._algo_meta.spec_cls()
            return create_population_from_spec(
                spec,
                self._algo_meta,
                self.environment,
                self.training.pop_size,
                self.device,
                mutation_spec=mutation_spec,
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
        environment: EnvironmentT,
        training: TrainingSpec | None = None,
        mutation: MutationT | None = None,
        tournament: TournamentSelectionT | None = None,
        replay_buffer: ReplayBufferT | None = None,
        client: ArenaClient | None = None,
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
