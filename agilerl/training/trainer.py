"""Trainer abstraction for AgileRL evolutionary training."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from typing_extensions import Self

from agilerl import HAS_ARENA_DEPENDENCIES, HAS_LLM_DEPENDENCIES
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
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.protocols import AgentType
from agilerl.typing import GymEnvType, PzEnvType
from agilerl.utils.trainer_utils import (
    build_mutations_from_spec,
    build_replay_buffer_from_spec,
    build_tournament_from_spec,
    create_population_from_spec,
)
from agilerl.vector import AsyncPettingZooVecEnv
from agilerl.wrappers.image_transpose import (
    ImageTranspose,
    PettingZooImageTranspose,
    needs_image_transpose,
)

logger = logging.getLogger(__name__)

AlgoSpecT = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec
AlgorithmT = AlgoSpecT | str
EnvSpecT = str | GymEnvSpec | PzEnvSpec | OfflineEnvSpec | LLMEnvSpec
EnvT = GymEnvType | PzEnvType | AsyncPettingZooVecEnv
ArenaEnvT = ArenaEnvSpec | dict[str, str]
ReplayBufferT = ReplayBufferSpec | None
PopulationT = list[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]

if HAS_ARENA_DEPENDENCIES:
    from agilerl.arena.client import ArenaClient
else:
    ArenaClient = None

if HAS_LLM_DEPENDENCIES:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None

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
    :param resume_from_checkpoint: Path to resume from checkpoint.
    :type resume_from_checkpoint: str | None
    :param device: Torch device (e.g. ``"cpu"``, ``"cuda"``).
    :type device: str | torch.device
    :param accelerator: Accelerator instance.
    :type accelerator: Accelerator | None
    """

    def __init__(
        self,
        algorithm: AlgorithmT,
        environment: EnvSpecT,
        training: TrainingSpec,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferT | None = None,
        *,
        resume_from_checkpoint: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
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
        self.resume_from_checkpoint = resume_from_checkpoint
        self.accelerator = accelerator

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path | dict[str, Any],
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        resume_from_checkpoint: str | None = None,
    ) -> Self:
        """Instantiate a :class:`Trainer` from a YAML, JSON, or dict manifest.

        The ``algorithm.name`` field is used to dispatch to the correct
        :class:`~agilerl.models.algo.AlgorithmSpec` subclass via the
        algorithm registry.  The environment section is resolved into
        the appropriate env spec based on the concrete trainer subclass
        (``cls``).

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any]
        :param device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        :type device: str | torch.device
        :param accelerator: Accelerator instance.
        :type accelerator: Accelerator | None
        :param resume_from_checkpoint: Path to resume from checkpoint.
        :type resume_from_checkpoint: str | None
        :returns: A fully configured :class:`Trainer` instance.
        :rtype: SelfTrainerT
        """
        # Load manifest from file or dictionary.
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
            resume_from_checkpoint=resume_from_checkpoint,
            device=device,
            accelerator=accelerator,
        )

    def to_manifest(self) -> dict[str, Any]:
        """Build a manifest from the :class:`Trainer` instance.

        :returns: A fully validated manifest ready for submission to Arena.
        :rtype: dict[str, Any]
        """
        network = getattr(self.algorithm, "net_config", None)
        manifest = TrainingManifest(
            algorithm=self.algorithm,
            environment=self.environment,
            training=self.training,
            network=network,
            mutation=self.mutation,
            replay_buffer=self.replay_buffer,
            tournament_selection=self.tournament,
        )
        return manifest.model_dump(mode="json", exclude_none=True)

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
    :type training: TrainingSpec
    :param mutation: Mutation probabilities and RL-HP ranges.  When an
        :class:`RLAlgorithmSpec` is used and ``hp_config`` is not set on it,
        HP ranges are derived from ``mutation.rl_hp_selection``.
    :type mutation: MutationSpec | Mutations | None
    :param tournament: Tournament selection configuration.
    :type tournament: TournamentSelectionSpec | TournamentSelection | None
    :param replay_buffer: Replay buffer configuration.
    :type replay_buffer: ReplayBufferSpec | ReplayBuffer | None
    :param resume_from_checkpoint: Path to resume from checkpoint.
    :type resume_from_checkpoint: str | None
    :param device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    :type device: str
    :param accelerator: Accelerator instance.
    :type accelerator: Accelerator | None
    """

    def __init__(
        self,
        algorithm: AlgorithmT,
        environment: EnvSpecT,
        training: TrainingSpec,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferT | None = None,
        *,
        resume_from_checkpoint: str | None = None,
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
            resume_from_checkpoint=resume_from_checkpoint,
            device=device,
            accelerator=accelerator,
        )

        # For LLM algorithms, load the tokenizer once and share it.
        self.tokenizer = None
        if isinstance(self.algorithm, LLMAlgorithmSpec):
            if AutoTokenizer is None:
                msg = "LLM dependencies are not installed. Please install them using: pip install agilerl[llm]"
                raise ImportError(msg)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.algorithm.pretrained_model_name_or_path
            )

        # Instantiate the training components from their specs.
        self.env = self._make_env()
        self.population = create_population_from_spec(
            population_size=self.training.population_size,
            algo_spec=self.algorithm,
            mutation_spec=self.mutation,
            env=self.env,
            device=self.device,
            accelerator=self.accelerator,
            tokenizer=self.tokenizer,
            resume_from_checkpoint=self.resume_from_checkpoint,
        )
        self.mutations = build_mutations_from_spec(self.mutation, self.device)
        self.tournament_selection = build_tournament_from_spec(
            self.tournament, self.training
        )
        self.memory = build_replay_buffer_from_spec(
            self.algorithm, self.replay_buffer, self.device
        )
        self.train_fn = self.algorithm.get_training_fn()

    def _make_env(self) -> EnvT:
        """Create the environment to train on.

        :returns: The environment to train on.
        :rtype: GymEnvType | PzEnvType | LLMEnvType
        """
        if isinstance(self.environment, LLMEnvSpec):
            return self.environment.make_env(
                tokenizer=self.tokenizer, accelerator=self.accelerator
            )

        # Check if the environment contains an image-last observation space,
        # in which case we transpose through a wrapper -> (H,W,C) -> (C,H,W)
        extra_wrappers = None
        probe = self.environment.make_single_env()
        if isinstance(self.environment, PzEnvSpec):
            sample_agent = probe.possible_agents[0]
            if needs_image_transpose(probe.observation_space(sample_agent)):
                extra_wrappers = [PettingZooImageTranspose]

        elif isinstance(self.environment, GymEnvSpec):
            if needs_image_transpose(probe.observation_space):
                extra_wrappers = [ImageTranspose]

        probe.close()
        return self.environment.make_env(extra_wrappers=extra_wrappers)

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
        save_elite: bool = False,
        elite_path: str | None = None,
        wb: bool = False,
        tensorboard: bool = False,
        tensorboard_log_dir: str | None = None,
        wandb_api_key: str | None = None,
        wandb_kwargs: dict[str, Any] | None = None,
    ) -> tuple[PopulationT, list[list[float]]]:
        """Run a local training job given the passed configuration.

        :param verbose: If ``True``, print verbose output.
        :type verbose: bool
        :param accelerator: An :class:`accelerate.Accelerator` instance.
        :type accelerator: Any | None
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

        :returns: A tuple of ``(population, fitness_history)`` where
            *population* is the final evolved population and
            *fitness_history* is a list of per-generation fitness scores.
        :rtype: tuple[PopulationT, list[list[float]]]
        """
        kwargs: dict[str, Any] = {
            "pop": self.population,
            "env": self.env,
            "max_steps": self.training.max_steps,
            "evo_steps": self.training.evo_steps,
            "tournament": self.tournament_selection,
            "mutation": self.mutations,
            "save_elite": save_elite,
            "elite_path": elite_path,
            "wb": wb,
            "tensorboard": tensorboard,
            "tensorboard_log_dir": tensorboard_log_dir,
            "verbose": verbose,
            "accelerator": accelerator or self.accelerator,
            "wandb_api_key": wandb_api_key,
            "wandb_kwargs": wandb_kwargs,
        }

        # Extract algo-specific kwargs from the algorithm spec.
        kwargs.update(
            self.algorithm.get_training_kwargs(
                training=self.training,
                env_spec=self.environment,
                memory=self.memory,
            )
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
    :type training: TrainingT
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
        training: TrainingSpec,
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
