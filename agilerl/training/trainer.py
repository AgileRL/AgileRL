"""Trainer abstraction for AgileRL evolutionary training."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from typing_extensions import Self

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
from agilerl.models.env import (
    ArenaEnvSpec,
    GymEnvSpec,
    LLMEnvSpec,
    OfflineEnvSpec,
    PzEnvSpec,
)
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.manifest import TrainingManifest
from agilerl.models.networks import NetworkSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.protocols import AgentType
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
EnvironmentT = (
    GymEnvType
    | PzEnvType
    | AsyncPettingZooVecEnv
    | GymEnvSpec
    | PzEnvSpec
    | OfflineEnvSpec
)
ArenaEnvT = ArenaEnvSpec | dict[str, str]
ReplayBufferT = ReplayBufferSpec | None
PopulationT = list[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]

if HAS_ARENA_DEPENDENCIES:
    from agilerl.arena.client import ArenaClient
    from agilerl.models import TrainingManifest
else:
    ArenaClient = None

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator


SelfTrainerT = TypeVar("SelfTrainerT", bound="Trainer")


class Trainer(ABC):
    """Abstract base trainer for AgileRL evolutionary training.

    :param algorithm: An algorithm spec or a string algorithm name.
    :type algorithm: AlgorithmT
    :param environment: A ``gymnasium.Env`` instance, a PettingZoo ``ParallelEnv`` instance, or an env-name string.
    :type environment: EnvironmentT
    :param training: Training loop parameters (max steps, population size,
        etc.).
    :type training: TrainingSpec
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
        training: TrainingSpec,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferT | None = None,
        device: str | torch.device = "cpu",
    ) -> None:

        # Convert string algorithm name to spec if provided.
        if isinstance(algorithm, str):
            algorithm: AlgoSpecT = ALGO_REGISTRY.get(algorithm).spec_cls()

        self.algorithm = algorithm
        self.environment = environment
        self.training = training
        self.mutation = mutation
        self.tournament = tournament
        self.replay_buffer = replay_buffer
        self.device = device

    @classmethod
    def from_manifest(
        cls, manifest: str | Path | dict[str, Any], **kwargs: Any
    ) -> Self:
        """Instantiate a :class:`Trainer` from a YAML, JSON, or dict manifest.

        The ``algorithm.name`` field is used to dispatch to the correct
        :class:`~agilerl.models.algo.AlgorithmSpec` subclass via the
        algorithm registry.  The environment section is resolved into
        the appropriate env spec based on the concrete trainer subclass
        (``cls``).

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any]
        :param kwargs: Extra keyword arguments forwarded to the trainer constructor.
        :type kwargs: Any
        :returns: A fully configured trainer instance.
        :rtype: SelfTrainerT
        """
        # Load manifest from file or dict.
        if isinstance(manifest, (str, Path)):
            with open(manifest) as fh:
                data = yaml.safe_load(fh)
        else:
            data = manifest

        # Validate manifest and resolve environment spec.
        validated_manifest = TrainingManifest.model_validate(data)
        env_spec = cls._resolve_env_spec(validated_manifest)

        # 'network' component of manifest corresponds to algorithm's net_config
        if validated_manifest.network is not None and hasattr(
            validated_manifest.algorithm, "net_config"
        ):
            validated_manifest.algorithm.net_config = validated_manifest.network

        return cls(
            algorithm=validated_manifest.algorithm,
            environment=env_spec,
            training=validated_manifest.training,
            mutation=validated_manifest.mutation,
            tournament=validated_manifest.tournament_selection,
            replay_buffer=validated_manifest.replay_buffer,
            **kwargs,
        )

    @classmethod
    def _resolve_env_spec(cls, manifest: TrainingManifest) -> Any:
        """Build an environment spec from the parsed manifest.

        Subclasses override this to produce the appropriate spec type.

        :param manifest: The validated training manifest.
        :type manifest: TrainingManifest
        :returns: An environment spec.
        :raises NotImplementedError: If the subclass has not overridden
            this method.
        """
        msg = "Trainer subclasses must implement _resolve_env_spec"
        raise NotImplementedError(msg)

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
    :param training: Training parameters.
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
        accelerator: Accelerator | None = None,
    ) -> None:

        super().__init__(
            algorithm,
            environment,
            training=training,
            mutation=mutation,
            tournament=tournament,
            replay_buffer=replay_buffer,
        )

        self.device = device
        self.accelerator = accelerator

        # Instantiate the training components from their specs.
        self.env = self.environment.make_env()
        self.population = create_population_from_spec(
            population_size=self.training.population_size,
            algo_spec=self.algorithm,
            mutation_spec=self.mutation,
            env=self.env,
            device=self.device,
            accelerator=self.accelerator,
        )
        self.mutations = build_mutations_from_spec(self.mutation, self.device)
        self.tourn = build_tournament_from_spec(self.tournament, self.training)
        self.memory = build_replay_buffer_from_spec(
            self.algorithm, self.replay_buffer, self.device
        )
        self.train_fn = self.algorithm.resolve_training_fn()

    @classmethod
    def _resolve_env_spec(
        cls, manifest: TrainingManifest
    ) -> GymEnvSpec | PzEnvSpec | LLMEnvSpec | OfflineEnvSpec:
        """Build the appropriate environment spec from the manifest.

        Uses the algorithm's ``agent_type`` and ``offline`` flag to
        choose the spec class.  For LLM algorithms, ``env_type`` is
        injected from the algorithm spec so the manifest environment
        section doesn't need to duplicate it.
        """
        env_data = dict(manifest.environment)
        env_data = {k: v for k, v in env_data.items() if v is not None}
        agent_type = manifest.algorithm.agent_type

        if agent_type == AgentType.LLMAgent:
            env_data.setdefault("env_type", manifest.algorithm.env_type)
            return LLMEnvSpec(**env_data)

        if agent_type == AgentType.MultiAgent:
            return PzEnvSpec(**env_data)

        if manifest.algorithm.offline:
            return OfflineEnvSpec(**env_data)

        return GymEnvSpec(**env_data)

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
    ) -> tuple[PopulationT, list[list[float]]]:
        """Run local training.

        :param verbose: If ``True``, print verbose output.
        :type verbose: bool
        :param accelerator: An :class:`accelerate.Accelerator` instance.
        :type accelerator: Any | None
        :param checkpoint: The number of episodes to save a checkpoint.
        :type checkpoint: int | None
        :param checkpoint_path: The path to save the checkpoint.
        :type checkpoint_path: str | None
        :param save_elite: If ``True``, save the elite agent.
        :type save_elite: bool
        :param elite_path: The path to save the elite agent.
        :type elite_path: str | None
        :param wb: If ``True``, enable Weights & Biases logging.
        :type wb: bool
        :param tensorboard: If ``True``, enable TensorBoard logging.
        :type tensorboard: bool
        :param tensorboard_log_dir: The path to save the TensorBoard logs.
        :type tensorboard_log_dir: str | None
        :param wandb_api_key: The Weights & Biases API key.
        :type wandb_api_key: str | None
        :param wandb_kwargs: The Weights & Biases keyword arguments.
        :type wandb_kwargs: dict[str, Any] | None
        :param swap_channels: If ``True``, swap the image channels from HWC to CHW.
        :type swap_channels: bool

        :returns: A tuple of ``(population, fitness_history)`` where
            *population* is the final evolved population and
            *fitness_history* is a list of per-generation fitness scores.
        :rtype: tuple[PopulationT, list[list[float]]]
        """
        kwargs = build_train_kwargs(
            algo_spec=self.algorithm,
            env=self.env,
            env_name=self.environment.name,
            env_spec=self.environment,
            population=self.population,
            training=self.training,
            tournament=self.tourn,
            mutations=self.mutations,
            memory=self.memory,
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

        return self.train_fn(**kwargs)


class ArenaTrainer(Trainer):
    """Submits evolutionary training jobs to the Arena RLOps platform.

    Builds an :class:`~agilerl.models.TrainingManifest` from the
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

    @classmethod
    def _resolve_env_spec(cls, manifest: TrainingManifest) -> ArenaEnvSpec:
        """Build an :class:`ArenaEnvSpec` from the manifest.

        :param manifest: The validated training manifest.
        :type manifest: TrainingManifest
        :returns: An environment spec for training on a validated Arena environment.
        :rtype: ArenaEnvSpec
        """
        env_data = manifest.environment

        if env_data.get("name") is None:
            msg = "Environment name is required for Arena training."
            raise ValueError(msg)

        return ArenaEnvSpec(
            name=env_data.get("name", ""),
            num_envs=env_data.get("num_envs", 1),
            version=str(env_data.get("version", "latest")),
        )

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
        """Build an :class:`~agilerl.models.TrainingManifest`.

        Assembles the manifest directly from the spec objects stored on
        this trainer, without any intermediate dict parsing.

        :returns: A fully validated manifest ready for submission.
        :rtype: TrainingManifest
        """
        env_spec = self.get_environment_spec()
        net_config: NetworkSpec | None = getattr(self.algorithm, "net_config", None)

        # Validate section-level Arena serialization early.
        self.algorithm.to_manifest()
        if net_config is not None:
            net_config.to_manifest()
        self.training.to_manifest(name=self.algorithm.resolve_training_fn().__name__)
        if self.mutation is not None:
            self.mutation.model_dump(exclude_none=True)
        if self.replay_buffer is not None:
            self.replay_buffer.model_dump(exclude_none=True)
        if self.tournament is not None:
            self.tournament.model_dump(exclude_none=True)

        return TrainingManifest(
            algorithm=self.algorithm,
            environment=env_spec,
            mutation=self.mutation,
            network=net_config,
            replay_buffer=self.replay_buffer,
            tournament_selection=self.tournament,
            training=self.training,
        )

    def get_environment_spec(self) -> ArenaEnvSpec:
        """Return the environment as an :class:`ArenaEnvSpec`.

        :returns: Environment spec instance.
        :rtype: ArenaEnvSpec
        :raises TypeError: If the environment is not a string or
            :class:`ArenaEnvSpec`.
        """
        if isinstance(self.environment, ArenaEnvSpec):
            return self.environment
        if isinstance(self.environment, dict):
            return ArenaEnvSpec(**self.environment)
        msg = (
            f"ArenaTrainer requires an ArenaEnvSpec or a dictionary with 'name', 'version', and 'num_envs' keys, "
            f"got {type(self.environment).__name__}"
        )
        raise TypeError(msg)
