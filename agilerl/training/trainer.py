"""Trainer abstraction for AgileRL evolutionary training."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_args

import yaml
from typing_extensions import Self

from agilerl import HAS_ARENA_DEPENDENCIES, HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core.base import (
    LLMAlgorithm,
    MultiAgentRLAlgorithm,
    RLAlgorithm,
)
from agilerl.models import ArenaCluster
from agilerl.models.algo import (
    ALGO_REGISTRY,
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.env import (
    ArenaEnvSpec,
    BanditEnvSpec,
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
from agilerl.utils.trainer_utils import (
    EnvironmentT,
    build_mutations_from_spec,
    build_replay_buffer_from_spec,
    build_tournament_from_spec,
    create_population_from_spec,
)
from agilerl.wrappers.image_transpose import (
    ImageTranspose,
    PettingZooImageTranspose,
    needs_image_transpose,
)

logger = logging.getLogger(__name__)

AlgoSpecT = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec
EnvSpecT = GymEnvSpec | PzEnvSpec | OfflineEnvSpec | LLMEnvSpec | BanditEnvSpec
ArenaEnvT = ArenaEnvSpec | dict[str, str] | str
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
    :type algorithm: AlgoSpecT | str
    :param environment: A ``gymnasium.Env`` instance, a PettingZoo ``ParallelEnv`` instance, or an env-name string.
    :type environment: EnvSpecT | str
    :param training: Training loop parameters (max steps, population size, etc.).
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
        algorithm: AlgoSpecT | str,
        environment: EnvSpecT | str,
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

        # Convert a plain environment name string to the appropriate spec.
        if isinstance(environment, str):
            environment = self._env_spec_from_string(algorithm, environment)

        self.algorithm_spec = algorithm
        self.env_spec = environment
        self.training_spec = training
        self.mutation_spec = mutation
        self.tournament_selection_spec = tournament
        self.replay_buffer_spec = replay_buffer
        self.device = device
        self.accelerator = accelerator
        self.resume_from_checkpoint = resume_from_checkpoint

    @staticmethod
    def _env_spec_from_string(
        algorithm: AlgoSpecT,
        name: str,
    ) -> EnvSpecT:
        """Build an environment spec from a plain environment name string.

        Only standard Gymnasium and PettingZoo environments can be
        resolved from a name alone.  Offline, bandit, and LLM algorithms
        need richer configuration and must be given a full spec object.

        :param algorithm: The resolved algorithm spec.
        :type algorithm: AlgoSpecT
        :param name: The environment name (e.g. ``"CartPole-v1"``).
        :type name: str
        :returns: The appropriate environment spec.
        :rtype: EnvSpecT
        :raises ValueError: When the algorithm's agent type is not
            single-agent or multi-agent.
        """
        agent_type = algorithm.agent_type

        if agent_type == AgentType.SingleAgent:
            return GymEnvSpec(name=name)

        if agent_type == AgentType.MultiAgent:
            return PzEnvSpec(name=name)

        msg = (
            "Only Gym and PettingZoo-based environments support passing "
            "a string for the environment."
        )
        raise ValueError(msg)

    @staticmethod
    def get_validated_manifest(
        manifest: str | Path | dict[str, Any],
    ) -> TrainingManifest:
        """Get a validated manifest from a YAML, JSON, or dict.

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any]
        :returns: A validated manifest.
        :rtype: TrainingManifest
        """
        if isinstance(manifest, (str, Path)):
            with open(manifest) as fh:
                data = yaml.safe_load(fh)
        else:
            data = manifest

        validated_manifest = TrainingManifest.model_validate(data)

        # 'network' component of manifest corresponds to algorithm's net_config.
        # Resolve the raw dict into the algorithm's concrete NetworkSpec
        # subclass (e.g. QNetworkSpec for DQN, StochasticActorSpec for PPO).
        algo_spec_cls = type(validated_manifest.algorithm)
        net_config_field = algo_spec_cls.model_fields.get("net_config")
        if net_config_field is not None and validated_manifest.network is not None:
            # get the NetworkSpec class from the type annotation and validate
            spec_cls: NetworkSpec = next(
                (
                    t
                    for t in get_args(net_config_field.annotation)
                    if t is not type(None)
                ),
                None,
            )
            if spec_cls is not None:
                validated_manifest.algorithm.net_config = spec_cls.model_validate(
                    validated_manifest.network
                )

        return validated_manifest

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path | dict[str, Any],
        *,
        resume_from_checkpoint: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
    ) -> Self:
        """Instantiate a :class:`Trainer` from a YAML, JSON, or dict manifest.

        The ``algorithm.name`` field is used to dispatch to the correct
        :class:`~agilerl.models.algo.AlgorithmSpec` subclass via the
        algorithm registry.  The environment section is resolved into
        the appropriate env spec based on the concrete trainer subclass
        (``cls``).

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any]
        :param resume_from_checkpoint: Path to resume from checkpoint.
        :type resume_from_checkpoint: str | None
        :param device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        :type device: str | torch.device
        :param accelerator: Accelerator instance.
        :type accelerator: Accelerator | None
        :returns: A fully configured :class:`Trainer` instance.
        :rtype: SelfTrainerT
        """
        # Validate manifest and resolve environment spec.
        validated_manifest = Trainer.get_validated_manifest(manifest)
        env_spec = cls._resolve_env_spec(validated_manifest)
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
        network = getattr(self.algorithm_spec, "net_config", None)
        manifest = TrainingManifest(
            algorithm=self.algorithm_spec,
            environment=self.env_spec,
            training=self.training_spec,
            network=network,
            mutation=self.mutation_spec,
            replay_buffer=self.replay_buffer_spec,
            tournament_selection=self.tournament_selection_spec,
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

    Automatically builds the components necessary for RL training with evolutionary HPO
    from a series of Pydantic models that validate the specified training configuration,
    and dispatches to the algorithm-specific training loop through `LocalTrainer.train()`.
    Handles all of the RL training paradigms available in AgileRL.

    :param algorithm: An `:class:`AlgorithmSpec` instance or a string algorithm name.
    :type algorithm: AlgorithmSpec | str
    :param environment: An RL environment following Gymnasium or PettingZoo API.
    :type environment: gym.Env | ParallelEnv
    :param training: Training parameters.
    :type training: TrainingSpec
    :param mutation: Mutation probabilities and RL hyperparameter ranges.  When an
        :class:`RLAlgorithmSpec` is used and ``hp_config`` is not set on it,
        hyperparameter ranges are derived from ``mutation.rl_hp_selection``.
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
        algorithm: AlgoSpecT | str,
        environment: EnvSpecT | str,
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
        self.tokenizer = (
            self._make_tokenizer()
            if isinstance(self.algorithm_spec, LLMAlgorithmSpec)
            else None
        )

        # Instantiate the training components from their specs.
        self.env = self._make_env()
        self.population = create_population_from_spec(
            population_size=self.training_spec.population_size,
            algo_spec=self.algorithm_spec,
            env=self.env,
            mutation_spec=self.mutation_spec,
            replay_buffer_spec=self.replay_buffer_spec,
            device=self.device,
            accelerator=self.accelerator,
            tokenizer=self.tokenizer,
            resume_from_checkpoint=self.resume_from_checkpoint,
        )
        self.mutations = build_mutations_from_spec(
            self.mutation_spec, self.device, accelerator=self.accelerator
        )
        self.tournament_selection = build_tournament_from_spec(
            self.tournament_selection_spec, self.training_spec
        )
        self.memory = build_replay_buffer_from_spec(
            self.algorithm_spec, self.replay_buffer_spec, self.device
        )
        self.train_fn = self.algorithm_spec.get_training_fn()

    def _make_tokenizer(self) -> AutoTokenizer:
        """Create the tokenizer for the LLM algorithm.

        :returns: The tokenizer.
        :rtype: AutoTokenizer
        :raises ImportError: If the LLM dependencies are not installed.
        """
        if AutoTokenizer is None:
            msg = "LLM dependencies are not installed. Please install them using: pip install agilerl[llm]"
            raise ImportError(msg)

        tokenizer = AutoTokenizer.from_pretrained(
            self.algorithm_spec.pretrained_model_name_or_path
        )

        # NOTE: For now we provide a simple chat template but could always give options to the user in
        # in the future.
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message['role'].capitalize() + ': ' + message['content'] + '\\n\\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ 'Assistant: ' }}"
                "{% endif %}"
            )
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _make_env(self) -> EnvironmentT:
        """Create the environment to train on.

        :returns: The environment to train on.
        :rtype: GymEnvType | PzEnvType | LLMEnvType | BanditEnv
        """
        if isinstance(self.env_spec, LLMEnvSpec):
            # Some LLMEnvSpec fields are dependent on the algo configuration
            self.env_spec.return_raw_completions = getattr(
                self.algorithm_spec, "use_vllm", False
            )
            self.env_spec.max_context_length = self.algorithm_spec.max_model_len
            self.env_spec.seed = self.algorithm_spec.seed

            return self.env_spec.make_env(
                tokenizer=self.tokenizer, accelerator=self.accelerator
            )

        if isinstance(self.env_spec, BanditEnvSpec):
            return self.env_spec.make_env()

        # Check if the environment contains an image-last observation space,
        # in which case we transpose through a wrapper -> (H,W,C) -> (C,H,W)
        extra_wrappers = None
        probe = self.env_spec.make_single_env()
        if isinstance(self.env_spec, PzEnvSpec):
            sample_agent = probe.possible_agents[0]
            if needs_image_transpose(probe.observation_space(sample_agent)):
                extra_wrappers = [PettingZooImageTranspose]

        elif isinstance(self.env_spec, GymEnvSpec):
            if needs_image_transpose(probe.observation_space):
                extra_wrappers = [ImageTranspose]

        probe.close()

        if extra_wrappers is not None:
            msg = "Found environment with channels-last observation space. Transposing to channels-first."
            logger.warning(msg)

        return self.env_spec.make_env(extra_wrappers=extra_wrappers)

    @classmethod
    def _resolve_env_spec(cls, manifest: TrainingManifest) -> EnvSpecT:
        """Build the appropriate environment spec from the manifest.

        Uses the algorithm's ``agent_type`` to choose the spec class.
        For LLM algorithms, ``env_type`` is injected from the algorithm
        spec so the manifest environment section doesn't need to
        duplicate it.
        """
        env_data = dict(manifest.environment)
        env_data = {k: v for k, v in env_data.items() if v is not None}
        agent_type = manifest.algorithm.agent_type

        if agent_type == AgentType.LLMAgent:
            env_data.setdefault("env_type", manifest.algorithm.env_type)
            return LLMEnvSpec(**env_data)

        if agent_type == AgentType.MultiAgent:
            return PzEnvSpec(**env_data)

        if agent_type == AgentType.OfflineAgent:
            return OfflineEnvSpec(**env_data)

        if agent_type == AgentType.BanditAgent:
            return BanditEnvSpec(**env_data)

        return GymEnvSpec(**env_data)

    def train(
        self,
        *,
        verbose: bool = True,
        save_elite: bool = False,
        elite_path: str | None = None,
        wb: bool = False,
        tensorboard: bool = False,
        tensorboard_log_dir: str | None = None,
        checkpoint_steps: int | None = None,
        checkpoint_path: str | None = None,
        overwrite_checkpoints: bool = False,
        wandb_api_key: str | None = None,
        wandb_kwargs: dict[str, Any] | None = None,
    ) -> tuple[PopulationT, list[list[float]]]:
        """Run a local training job given the passed configuration.

        :param verbose: If ``True``, print verbose output. Defaults to ``True``.
        :type verbose: bool
        :param save_elite: If ``True``, save the elite agent. Defaults to ``False``.
        :type save_elite: bool
        :param elite_path: The path to save the elite agent. Defaults to ``None``.
        :type elite_path: str | None
        :param wb: If ``True``, enable Weights & Biases logging. Defaults to ``False``.
        :type wb: bool
        :param tensorboard: If ``True``, enable TensorBoard logging. Defaults to ``False``.
        :type tensorboard: bool
        :param tensorboard_log_dir: The path to save the TensorBoard logs. Defaults to ``None``,
            which will use the default TensorBoard log directory ``tensorboard_logs``.
        :type tensorboard_log_dir: str | None
        :param checkpoint_steps: The number of steps between checkpoints. Defaults to ``None``.
        :type checkpoint_steps: int | None
        :param checkpoint_path: The path to save the checkpoints. Defaults to ``None``.
        :type checkpoint_path: str | None
        :param overwrite_checkpoints: If ``True``, overwrite the checkpoint. Defaults to ``False``.
        :type overwrite_checkpoints: bool
        :param wandb_api_key: The Weights & Biases API key. Defaults to ``None``.
        :type wandb_api_key: str | None
        :param wandb_kwargs: The Weights & Biases keyword arguments. Defaults to ``None``.
        :type wandb_kwargs: dict[str, Any] | None

        :returns: A tuple of ``(population, fitness_history)`` where
            *population* is the final evolved population and
            *fitness_history* is a list of per-generation fitness scores.
        :rtype: tuple[PopulationT, list[list[float]]]
        """
        manifest = self.to_manifest()
        kwargs: dict[str, Any] = {
            "pop": self.population,
            "env": self.env,
            "init_hp": manifest,
            "max_steps": self.training_spec.max_steps,
            "evo_steps": self.training_spec.evo_steps,
            "tournament": self.tournament_selection,
            "mutation": self.mutations,
            "save_elite": save_elite,
            "elite_path": elite_path,
            "wb": wb,
            "tensorboard": tensorboard,
            "tensorboard_log_dir": tensorboard_log_dir,
            "verbose": verbose,
            "accelerator": self.accelerator,
            "wandb_api_key": wandb_api_key,
            "wandb_kwargs": wandb_kwargs,
        }

        # Add checkpointing arguments to the training spec
        self.training_spec.checkpoint_steps = checkpoint_steps
        self.training_spec.checkpoint_path = checkpoint_path
        self.training_spec.overwrite_checkpoints = overwrite_checkpoints

        # Extract algo-specific kwargs from the algorithm spec.
        kwargs.update(
            self.algorithm_spec.get_training_kwargs(
                training=self.training_spec,
                env_spec=self.env_spec,
                memory=self.memory,
            )
        )
        return self.train_fn(**kwargs)


class ArenaTrainer(Trainer):
    """Submits evolutionary training jobs to the Arena RLOps platform.

    Builds an :class:`~agilerl.models.TrainingManifest` from the
    provided specs and submits it via
    :class:`~agilerl.arena.client.ArenaClient`.

    :param algorithm: An `:class:`AlgorithmSpec` instance or a string algorithm name.
    :type algorithm: AlgoSpecT | str
    :param environment: An `:class:`ArenaEnvSpec` instance or a string env name.
    :type environment: ArenaEnvSpec | str
    :param training: Training loop parameters.
    :type training: TrainingSpec
    :param client: An authenticated :class:`ArenaClient`.  One is created
        automatically using the provided API key. Defaults to ``None``.
    :type client: ArenaClient | None
    :param api_key: The Arena API key. Defaults to ``None``.
    :type api_key: str | None
    :param mutation: Mutation probabilities and RL-HP ranges. Defaults to ``None``.
    :type mutation: MutationSpec | None
    :param tournament: Tournament selection configuration. Defaults to ``None``.
    :type tournament: TournamentSelectionSpec | None
    :param replay_buffer: Replay buffer configuration. Defaults to ``None``.
    :type replay_buffer: ReplayBufferT | None
    """

    def __init__(
        self,
        algorithm: AlgoSpecT | str,
        environment: ArenaEnvT,
        training: TrainingSpec,
        *,
        client: ArenaClient | None = None,
        api_key: str | None = None,
        mutation: MutationSpec | None = None,
        tournament: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferT | None = None,
    ) -> None:

        if isinstance(environment, str):
            environment = ArenaEnvSpec(name=environment)

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
    def from_manifest(
        cls,
        manifest: str | Path | dict[str, Any],
        *,
        client: ArenaClient | None = None,
        api_key: str | None = None,
    ) -> Self:
        """Instantiate a :class:`ArenaTrainer` from a YAML, JSON, or dict manifest.

        Automatically dispatches to the correct Pydantic models based on the manifest
        fields.

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any]
        :param client: An authenticated :class:`ArenaClient`.  One is created
            automatically if not provided.
        :type client: ArenaClient | None
        :param api_key: The Arena API key.
        :type api_key: str | None
        :returns: A fully configured :class:`ArenaTrainer` instance.
        :rtype: ArenaTrainer
        """
        # Validate manifest and resolve environment spec.
        validated_manifest = Trainer.get_validated_manifest(manifest)
        env_spec = cls._resolve_env_spec(validated_manifest)

        return cls(
            algorithm=validated_manifest.algorithm,
            environment=env_spec,
            client=client,
            api_key=api_key,
            training=validated_manifest.training,
            mutation=validated_manifest.mutation,
            tournament=validated_manifest.tournament_selection,
            replay_buffer=validated_manifest.replay_buffer,
        )

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
            num_envs=env_data.get("num_envs", 16),
            version=str(env_data.get("version", "latest")),
        )

    def train(
        self, resources: ArenaCluster | None = None, stream: bool = False
    ) -> dict[str, Any]:
        """Build the manifest and submit the training job to Arena.

        :param resources: The resources to use for the training job.
        :type resources: ArenaCluster | None
        :param stream: If ``True``, stream logs to the terminal and block
            until the job finishes.
        :type stream: bool
        :returns: Arena API response including ``job_id`` and ``status``.
            When *stream* is ``True``, returns the final result payload.
        :rtype: dict[str, Any]
        """
        manifest = self.to_manifest()
        return self._client.submit_job(manifest, stream=stream)
