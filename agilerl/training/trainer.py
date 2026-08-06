# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Trainer abstraction for AgileRL evolutionary training."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_origin

from typing_extensions import Self

from agilerl import HAS_ARENA_DEPENDENCIES, HAS_LLM_DEPENDENCIES, AgentType
from agilerl.algorithms.core.base import (
    LLMAlgorithm,
    MultiAgentRLAlgorithm,
    RLAlgorithm,
)
from agilerl.hpo.multi_frequency import MultiFrequencySelection
from agilerl.hpo.tournament import TournamentSelection
from agilerl.models import (
    ALGO_REGISTRY,
    AlgoSpec,
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    MultiFrequencySelectionSpec,
    MutationSpec,
    ReplayBufferSpec,
    RLAlgorithmSpec,
    TournamentSelectionSpec,
    TrainingManifest,
    TrainingSpec,
)
from agilerl.models.env import (
    BanditEnvSpec,
    GymEnvSpec,
    LLMEnvSpec,
    LLMEnvType,
    OfflineEnvSpec,
    PzEnvSpec,
)
from agilerl.models.hpo import SelectionStrategySpec
from agilerl.models.networks import (
    NetworkSpec,
    encoder_spec_for_arch,
    infer_encoder_arch,
    network_arch_is_resolvable,
)
from agilerl.utils.llm_utils import (
    apply_pad_token_id,
    load_pad_token_configs,
    resolve_pad_token_id,
)
from agilerl.utils.trainer_utils import (
    EnvironmentType,
    build_mutations_from_spec,
    build_replay_buffer_from_spec,
    build_selection_from_spec,
    create_population_from_spec,
    get_spaces_from_env,
    resolve_deprecated_selection_kwargs,
)

logger = logging.getLogger(__name__)

EnvSpecType = GymEnvSpec | PzEnvSpec | OfflineEnvSpec | LLMEnvSpec | BanditEnvSpec
ReplayBufferType = ReplayBufferSpec | None
PopulationType = list[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]


if HAS_ARENA_DEPENDENCIES:
    from agilerl.arena import ArenaClient
    from agilerl.arena.models import TrainingManifest as ArenaManifest
    from agilerl.arena.models.env import EnvSpec as ArenaEnvSpec
else:
    ArenaClient = None
    ArenaManifest = None
    ArenaEnvSpec = None

if HAS_LLM_DEPENDENCIES:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator
    from gymnasium import spaces
    from transformers import PreTrainedTokenizerBase


SelfTrainerT = TypeVar("SelfTrainerT", bound="Trainer")


class Trainer(ABC):
    """Abstract base trainer for AgileRL evolutionary training.

    :param algorithm: An algorithm spec or a string algorithm name.
    :type algorithm: AlgoSpec | str
    :param environment: A ``gymnasium.Env`` instance, a PettingZoo ``ParallelEnv`` instance, or an env-name string.
    :type environment: EnvSpecType | str
    :param training: Training loop parameters (max steps, population size, etc.).
    :type training: TrainingSpec
    :param mutation: Mutation probabilities and RL-HP ranges.
    :type mutation: MutationSpec | None
    :param selection_strategy: Selection strategy driving evolutionary HPO — a
        :class:`~agilerl.models.hpo.TournamentSelectionSpec` or a
        :class:`~agilerl.models.hpo.MultiFrequencySelectionSpec` (MF-PBT).
    :type selection_strategy: SelectionStrategySpec | None
    :param replay_buffer: Replay buffer configuration.  Off-policy algorithms
        auto-create a default buffer when this is ``None``.
    :type replay_buffer: ReplayBufferType | None
    :param resume_from_checkpoint: Path to resume from checkpoint.
    :type resume_from_checkpoint: str | None
    :param device: Torch device (e.g. ``"cpu"``, ``"cuda"``).
    :type device: str | torch.device
    :param accelerator: Accelerator instance.
    :type accelerator: Accelerator | None
    :param kwargs: Accepts the deprecated tournament alias for
        selection_strategy.
    """

    def __init__(
        self,
        algorithm: AlgoSpec | str,
        environment: EnvSpecType | str,
        training: TrainingSpec | None = None,
        mutation: MutationSpec | None = None,
        selection_strategy: SelectionStrategySpec | None = None,
        replay_buffer: ReplayBufferType | None = None,
        *,
        resume_from_checkpoint: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        **kwargs: Any,
    ) -> None:

        selection_strategy = resolve_deprecated_selection_kwargs(
            selection_strategy, kwargs, caller=type(self).__name__
        )

        # Convert string algorithm name to spec if provided.
        if isinstance(algorithm, str):
            algorithm: AlgoSpec = ALGO_REGISTRY.get(algorithm).spec_cls()

        # Convert a plain environment name string to the appropriate spec.
        if isinstance(environment, str):
            environment = self._env_spec_from_string(algorithm, environment)

        self.algorithm_spec = algorithm
        self.env_spec = environment
        self.training_spec = training or TrainingSpec()
        self.mutation_spec = mutation
        self.selection_strategy_spec = selection_strategy
        self.replay_buffer_spec = replay_buffer
        self.device = device
        self.accelerator = accelerator
        self._resume_checkpoint = resume_from_checkpoint

        # MF-PBT's bracket sizes derive from pop_size, which the spec cannot see on its own.
        if isinstance(selection_strategy, MultiFrequencySelectionSpec):
            if "pop_size" not in self.training_spec.model_fields_set:
                msg = "pop_size is required in the training block."
                raise ValueError(msg)

            spec = selection_strategy
            # Only the bracket sizes are written back: the spec's own validator
            # already resolved the rest
            (
                _,
                _,
                _,
                spec.n_winners,
                spec.n_survivors,
                spec.n_open_for_migration,
                spec.n_losers,
            ) = MultiFrequencySelection._resolve_and_validate(
                self.training_spec.pop_size,
                spec.n_subpopulations,
                spec.evolution_frequency_ratios,
                spec.n_winners,
                spec.n_survivors,
                spec.n_open_for_migration,
                spec.n_losers,
            )

    @property
    def tournament_selection_spec(self) -> TournamentSelectionSpec | None:
        """The configured tournament-selection spec.

        .. deprecated::
            Superseded by :attr:`selection_strategy_spec`, which holds whichever
            selection strategy is configured.

        :returns: The tournament-selection spec, or None when MF-PBT or no
            strategy is configured.
        :rtype: TournamentSelectionSpec | None
        """
        warnings.warn(
            "Trainer.tournament_selection_spec is deprecated and will be removed in "
            "a future release; use Trainer.selection_strategy_spec instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        spec = self.selection_strategy_spec
        return spec if isinstance(spec, TournamentSelectionSpec) else None

    @staticmethod
    def _env_spec_from_string(
        algorithm: AlgoSpec,
        name: str,
    ) -> EnvSpecType:
        """Build an environment spec from a plain environment name string.

        Only standard Gymnasium and PettingZoo environments can be
        resolved from a name alone.  Offline, bandit, and LLM algorithms
        need richer configuration and must be given a full spec object.

        :param algorithm: The resolved algorithm spec.
        :type algorithm: AlgoSpec
        :param name: The environment name (e.g. ``"CartPole-v1"``).
        :type name: str
        :returns: The appropriate environment spec.
        :rtype: EnvSpecType
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

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path | dict[str, Any] | TrainingManifest,
        **kwargs: Any,
    ) -> Self:
        """Instantiate a :class:`Trainer` from a JSON-style manifest or a TrainingManifest instance.

        The manifest supplies the algorithm, environment, and training
        configuration; any trainer-specific construction arguments are passed
        through as keyword arguments (e.g. ``device``, ``accelerator``, and
        ``resume_from_checkpoint`` for :class:`LocalTrainer`, or ``client`` and
        ``api_key`` for :class:`ArenaTrainer`).

        :param manifest: Path to a YAML/JSON file, or a raw dict, or a TrainingManifest instance.
        :type manifest: str | Path | dict[str, Any] | TrainingManifest
        :param kwargs: Trainer-specific construction arguments forwarded to the
            subclass constructor.
        :returns: A fully configured :class:`Trainer` instance.
        :rtype: SelfTrainerT
        """
        validated_manifest = (
            TrainingManifest.get_validated(manifest, mode="python")
            if not isinstance(manifest, TrainingManifest)
            else manifest
        )
        # `mode="python"` always yields a model; only `mode="json"` returns a dict.
        assert isinstance(validated_manifest, TrainingManifest)
        env_spec = cls._resolve_env_spec(validated_manifest)
        return cls(
            algorithm=validated_manifest.algorithm,
            environment=env_spec,
            training=validated_manifest.training,
            mutation=validated_manifest.mutation,
            selection_strategy=validated_manifest.selection_strategy,
            replay_buffer=validated_manifest.replay_buffer,
        )

    @staticmethod
    def _resolve_env_spec(manifest: TrainingManifest) -> Any:  # noqa: ANN401 -- env spec type is subclass-specific (EnvSpecType locally vs ArenaEnvSpec)
        """Build an environment spec from the parsed manifest.

        :param manifest: The validated training manifest.
        :type manifest: TrainingManifest
        :returns: An environment spec.
        :raises NotImplementedError: If the subclass has not overridden
            this method.
        """
        msg = "Trainer subclasses must implement _resolve_env_spec"
        raise NotImplementedError(msg)

    @abstractmethod
    def train(self) -> tuple[PopulationType, list[float]] | dict[str, Any]:
        """Run the training loop.

        - :class:`LocalTrainer` runs training locally and returns a tuple of
          ``(population, fitnesses)`` where *population* is the final evolved
          population and *fitnesses* contains each agent's fitness from the
          final evaluation round.
        - :class:`ArenaTrainer` submits a job to Arena and returns the API
          response as a ``dict``.

        :returns: The training result, whose type depends on the trainer.
        :rtype: tuple[PopulationType, list[float]] | dict[str, Any]
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
    :param training: Training parameters. Defaults to ``TrainingSpec()`` (1M steps,
        single agent, no HPO).
    :type training: TrainingSpec | None
    :param mutation: Mutation probabilities and RL hyperparameter ranges.  When an
        :class:`RLAlgorithmSpec` is used and ``hp_config`` is not set on it,
        hyperparameter ranges are derived from ``mutation.rl_hp_selection``.
    :type mutation: MutationSpec | Mutations | None
    :param selection_strategy: Selection strategy driving evolutionary HPO: a
        :class:`~agilerl.models.hpo.TournamentSelectionSpec` or a
        :class:`~agilerl.models.hpo.MultiFrequencySelectionSpec` (MF-PBT).
    :type selection_strategy: SelectionStrategySpec | None
    :param replay_buffer: Replay buffer configuration.
    :type replay_buffer: ReplayBufferSpec | ReplayBuffer | None
    :param hpo: Whether to enable evolutionary HPO using default mutation probabilities, tournament selection,
        and RL hyperparameters to mutate. Defaults to ``False``.
    :type hpo: bool
    :param resume_from_checkpoint: Path to resume from checkpoint.
    :type resume_from_checkpoint: str | None
    :param device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    :type device: str
    :param accelerator: Accelerator instance.
    :type accelerator: Accelerator | None
    :param kwargs: Accepts the deprecated tournament alias for
        selection_strategy.
    """

    def __init__(
        self,
        algorithm: AlgoSpec | str,
        environment: EnvSpecType | str,
        training: TrainingSpec | None = None,
        mutation: MutationSpec | None = None,
        selection_strategy: SelectionStrategySpec | None = None,
        replay_buffer: ReplayBufferType | None = None,
        *,
        hpo: bool = False,
        resume_from_checkpoint: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            algorithm,
            environment,
            training=training,
            mutation=mutation,
            selection_strategy=resolve_deprecated_selection_kwargs(
                selection_strategy, kwargs, caller=type(self).__name__
            ),
            replay_buffer=replay_buffer,
            resume_from_checkpoint=resume_from_checkpoint,
            device=device,
            accelerator=accelerator,
        )

        # If HPO is enabled, use default mutation probabilities, RL hyperparameters
        # to mutate, and, unless a strategy was configured, tournament selection
        if hpo:
            self.mutation_spec = self.mutation_spec or MutationSpec()
            if self.selection_strategy_spec is None:
                self.selection_strategy_spec = TournamentSelectionSpec()

        # For LLM algorithms, load the tokenizer once and share it.
        self.tokenizer = (
            self._make_tokenizer()
            if isinstance(self.algorithm_spec, LLMAlgorithmSpec)
            else None
        )

        # Instantiate the training components from their specs.
        self.env = self._make_env()
        self._resolve_deferred_net_config()
        self.population = create_population_from_spec(
            population_size=self.training_spec.pop_size,
            algo_spec=self.algorithm_spec,
            env=self.env,
            mutation_spec=self.mutation_spec,
            replay_buffer_spec=self.replay_buffer_spec,
            device=self.device,
            accelerator=(
                self.accelerator
                if not isinstance(self.algorithm_spec, LLMAlgorithmSpec)
                else None
            ),
            tokenizer=self.tokenizer,
            resume_from_checkpoint=self._resume_checkpoint,
            selection_strategy_spec=self.selection_strategy_spec,
        )
        # Classic RL algorithms keep Accelerate; LLM training uses torch.distributed.
        if not isinstance(self.algorithm_spec, LLMAlgorithmSpec):
            self.mutations = build_mutations_from_spec(
                self.mutation_spec, self.device, accelerator=self.accelerator
            )
        else:
            self.mutations = build_mutations_from_spec(
                self.mutation_spec, self.device, accelerator=None
            )

        self.selection_strategy = build_selection_from_spec(
            self.selection_strategy_spec,
            self.training_spec,
            seed=(
                self.mutation_spec.rand_seed if self.mutation_spec is not None else None
            ),
        )
        self.memory = build_replay_buffer_from_spec(
            self.algorithm_spec,
            self.replay_buffer_spec,
            self.device,
        )
        self.n_step_memory = (
            self.replay_buffer_spec.init_n_step_buffer(self.algorithm_spec, self.device)
            if self.replay_buffer_spec is not None
            else None
        )
        self._multiturn = (
            isinstance(self.env_spec, LLMEnvSpec)
            and self.env_spec.env_type == LLMEnvType.MULTITURN
        )

        # Multi-turn LLM training requires an env factory rather than an instantiated environment.
        if self._multiturn:
            assert isinstance(self.env_spec, LLMEnvSpec)
            assert isinstance(self.algorithm_spec, LLMAlgorithmSpec)
            max_model_len = getattr(self.algorithm_spec, "max_model_len", None)
            max_output_tokens = getattr(self.algorithm_spec, "max_output_tokens", None)
            assert self.tokenizer is not None
            self.env_factory = self.env_spec.make_multiturn_env_factory(
                self.tokenizer,
                max_model_len=max_model_len,
                max_output_tokens=max_output_tokens,
            )
            self.train_fn = self.algorithm_spec.get_training_fn(multiturn=True)
        else:
            self.env_factory = None
            self.train_fn = self.algorithm_spec.get_training_fn()

    def _resolve_deferred_net_config(self) -> None:
        """Resolve a manifest network section whose ``arch`` was omitted.

        When the manifest did not declare ``arch``, ``net_config`` is left as a
        raw dict by manifest validation. Now that the environment (hence the
        observation space) exists, infer the arch and validate the network into
        the algorithm's concrete ``NetworkSpec``. No-ops when ``net_config`` is
        already a validated spec (programmatic construction) or None.
        """
        net_config = getattr(self.algorithm_spec, "net_config", None)
        if not isinstance(net_config, dict):
            return
        if network_arch_is_resolvable(net_config):
            return

        observation_space, _ = get_spaces_from_env(self.algorithm_spec, self.env)
        simba = bool(net_config.get("simba", False))
        recurrent = bool(getattr(self.algorithm_spec, "recurrent", False))

        if isinstance(observation_space, dict):
            # ``isinstance`` narrowing leaves a ``Space & dict`` intersection, so
            # rebuild the per-agent mapping for the multi-agent resolver.
            obs_by_agent: dict[str, Any] = {
                str(agent_id): space for agent_id, space in observation_space.items()
            }
            self._resolve_deferred_net_config_multi_agent(
                net_config,
                obs_by_agent,
                simba,
                recurrent,
            )
            return

        encoder_config = self._resolve_encoder_config(
            observation_space,
            net_config.get("encoder_config"),
            simba=simba,
            recurrent=recurrent,
        )
        resolved = {**net_config, "encoder_config": encoder_config}
        if isinstance(self.algorithm_spec, RLAlgorithmSpec):
            self.algorithm_spec.net_config = self._algo_net_spec_cls().model_validate(
                resolved
            )

    def _resolve_encoder_config(
        self,
        observation_space: spaces.Space,
        user_encoder_config: dict[str, Any] | None,
        *,
        simba: bool,
        recurrent: bool,
    ) -> dict[str, Any]:
        """Build a resolved ``encoder_config`` for a single observation space.

        ``arch`` alone isn't enough to validate: variant-specific fields (e.g.
        ``MlpSpec.hidden_size``, ``CnnSpec.channel_size``) are REQUIRED with no
        default. Seed ONLY those required fields from the default-config helper
        the algorithms use; every OPTIONAL field must fall through to its
        ``*Spec`` pydantic default so the deferred path resolves to the same HP
        bounds as the eager (arch-declared) path. User-provided ``encoder_config``
        keys overlay the seed, and ``arch`` is set last.

        :param observation_space: The (per-agent) observation space.
        :type observation_space: gymnasium.spaces.Space
        :param user_encoder_config: The manifest-provided ``encoder_config``, if any.
        :type user_encoder_config: dict[str, Any] | None
        :param simba: Whether the network requests a SimBa encoder.
        :type simba: bool
        :param recurrent: Whether the algorithm requests a recurrent encoder.
        :type recurrent: bool
        :returns: A resolved ``encoder_config`` dict including its ``arch``.
        :rtype: dict[str, Any]
        """
        from agilerl.utils.evolvable_networks import get_default_encoder_config

        arch = infer_encoder_arch(observation_space, recurrent=recurrent, simba=simba)
        encoder_spec_cls = encoder_spec_for_arch(arch)
        required = {
            name
            for name, field in encoder_spec_cls.model_fields.items()
            if field.is_required()
        }
        default_encoder_config = get_default_encoder_config(
            observation_space, simba=simba, recurrent=recurrent
        )
        seed = {k: v for k, v in default_encoder_config.items() if k in required}
        return {**seed, **(user_encoder_config or {}), "arch": arch}

    def _algo_net_spec_cls(self) -> type[NetworkSpec]:
        """Resolve the algorithm's concrete ``NetworkSpec`` subclass.

        Picks the non-``None`` member of the ``net_config`` field's type
        annotation (e.g. ``StochasticActorSpec`` for IPPO), falling back to the
        base :class:`~agilerl.models.networks.NetworkSpec`.

        :returns: The ``NetworkSpec`` subclass used to validate the network.
        :rtype: type[NetworkSpec]
        """
        net_config_field = type(self.algorithm_spec).model_fields.get("net_config")
        if net_config_field is None:
            return NetworkSpec
        return next(
            (
                t
                for t in get_args(net_config_field.annotation)
                if t is not type(None) and get_origin(t) is not dict
            ),
            NetworkSpec,
        )

    def _resolve_deferred_net_config_multi_agent(
        self,
        net_config: dict[str, Any],
        observation_spaces: dict[str, Any],
        simba: bool,
        recurrent: bool,
    ) -> None:
        """Resolve a deferred multi-agent network section, per agent group.

        Infers the encoder arch from each agent's own observation space and
        builds a per-group nested ``net_config`` so heterogeneous agents each
        get the right encoder while homogeneous agents share one. The manifest's
        shared network settings (``latent_dim``, ``head_config``, ...) are
        applied to every group; only ``encoder_config`` is resolved per group.

        Agents are keyed by their shared-policy group id (the same
        ``rsplit("_", 1)`` prefix the algorithm uses to group homogeneous
        agents), because ``build_net_config`` rejects individual-agent keys in a
        grouped setting yet resolves group-id keys for both grouped and
        ungrouped environments.

        :param net_config: The raw (deferred) network section from the manifest.
        :type net_config: dict[str, Any]
        :param observation_spaces: Per-agent observation spaces.
        :type observation_spaces: dict[str, gymnasium.spaces.Space]
        :param simba: Whether the network requests a SimBa encoder.
        :type simba: bool
        :param recurrent: Whether the algorithm requests a recurrent encoder.
        :type recurrent: bool
        """
        spec_cls = self._algo_net_spec_cls()
        shared_fields = {k: v for k, v in net_config.items() if k != "encoder_config"}
        user_encoder_config = net_config.get("encoder_config")

        resolved: dict[str, NetworkSpec] = {}
        for agent_id, observation_space in observation_spaces.items():
            group_id = (
                agent_id.rsplit("_", 1)[0] if isinstance(agent_id, str) else agent_id
            )
            if group_id in resolved:
                continue
            encoder_config = self._resolve_encoder_config(
                observation_space,
                user_encoder_config,
                simba=simba,
                recurrent=recurrent,
            )
            agent_net_config = {**shared_fields, "encoder_config": encoder_config}
            resolved[group_id] = spec_cls.model_validate(agent_net_config)

        if isinstance(self.algorithm_spec, MultiAgentRLAlgorithmSpec):
            self.algorithm_spec.net_config = resolved

    def _make_tokenizer(self) -> PreTrainedTokenizerBase:
        """Create the tokenizer for the LLM algorithm.

        :returns: The tokenizer.
        :rtype: PreTrainedTokenizerBase
        :raises ImportError: If the LLM dependencies are not installed.
        """
        if AutoTokenizer is None:
            msg = "LLM dependencies are not installed. Please install them using: pip install agilerl[llm]"
            raise ImportError(msg)

        assert isinstance(self.algorithm_spec, LLMAlgorithmSpec)
        tokenizer = AutoTokenizer.from_pretrained(
            self.algorithm_spec.pretrained_model_name_or_path
        )
        assert tokenizer is not None

        # NOTE: For now we provide a simple chat template but could always
        # give options to the user in the future.
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message['role'].capitalize() + ': ' + message['content'] + '\\n\\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ 'Assistant: ' }}"
                "{% endif %}"
            )

        model_config, generation_config = load_pad_token_configs(
            self.algorithm_spec.pretrained_model_name_or_path
        )
        pad_token_id, pad_source = resolve_pad_token_id(
            tokenizer,
            model_config=model_config,
            generation_config=generation_config,
        )
        apply_pad_token_id(tokenizer, pad_token_id)
        logger.info(
            "Resolved tokenizer pad_token_id=%s from %s (eos_token_id=%s)",
            pad_token_id,
            pad_source,
            tokenizer.eos_token_id,
        )

        return tokenizer

    def _make_env(self) -> EnvironmentType | None:
        """Create the environment to train on.

        :rtype: GymEnvType | PzEnvType | LLMEnvType | BanditEnv | None
        """
        if isinstance(self.env_spec, LLMEnvSpec):
            if self.env_spec.env_type == LLMEnvType.MULTITURN:
                return None

            assert isinstance(self.algorithm_spec, LLMAlgorithmSpec)

            # Some LLMEnvSpec fields are dependent on the algo configuration
            self.env_spec.return_raw_completions = getattr(
                self.algorithm_spec, "use_vllm", False
            )
            self.env_spec.max_context_length = self.algorithm_spec.max_model_len
            self.env_spec.seed = self.algorithm_spec.seed
            self.env_spec.data_batch_size_per_gpu = self.algorithm_spec.batch_size

            assert self.tokenizer is not None
            return self.env_spec.make_env(tokenizer=self.tokenizer)

        return self.env_spec.make_env()

    @staticmethod
    def _resolve_env_spec(manifest: TrainingManifest) -> EnvSpecType:
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
            # `agent_type == LLMAgent` selects an LLM spec, on which `env_type`
            # is a class variable; narrow so it resolves without an ignore.
            assert isinstance(manifest.algorithm, LLMAlgorithmSpec)
            env_data.setdefault("env_type", manifest.algorithm.env_type)
            return LLMEnvSpec(**env_data)

        if agent_type == AgentType.MultiAgent:
            return PzEnvSpec(**env_data)

        if agent_type == AgentType.OfflineAgent:
            return OfflineEnvSpec(**env_data)

        if agent_type == AgentType.BanditAgent:
            return BanditEnvSpec(**env_data)

        return GymEnvSpec(**env_data)

    @property
    def tournament_selection(self) -> TournamentSelection | None:
        """The built tournament-selection operator.

        .. deprecated::
            Superseded by :attr:`selection_strategy`, which holds whichever selection
            operator was built.

        :returns: The :class:`~agilerl.hpo.tournament.TournamentSelection` operator,
            or None when MF-PBT or no strategy is configured.
        :rtype: TournamentSelection | None
        """
        warnings.warn(
            "LocalTrainer.tournament_selection is deprecated and will be removed in "
            "a future release; use LocalTrainer.selection_strategy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        strategy = self.selection_strategy
        return strategy if isinstance(strategy, TournamentSelection) else None

    def to_manifest(self) -> dict[str, Any]:
        """Build a local training manifest from the :class:`LocalTrainer` instance.

        :returns: A JSON-serializable manifest using core :mod:`agilerl.models`.
        :rtype: dict[str, Any]
        """
        manifest = TrainingManifest.from_trainer_specs(
            algorithm=self.algorithm_spec,
            environment=self.env_spec,
            training=self.training_spec,
            mutation=self.mutation_spec,
            replay_buffer=self.replay_buffer_spec,
            selection_strategy=self.selection_strategy_spec,
        )
        return manifest.model_dump(mode="json", exclude_none=True)

    def train(
        self,
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
    ) -> tuple[PopulationType, list[float]]:
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

        :returns: A tuple of ``(population, fitnesses)`` where
            *population* is the final evolved population and
            *fitnesses* contains each agent's fitness from the final
            evaluation round.
        :rtype: tuple[PopulationType, list[float]]
        """
        manifest = self.to_manifest()
        evo_steps = (
            self.training_spec.evo_steps
            if self.training_spec.evo_steps is not None
            else self.algorithm_spec.default_evo_steps
        )
        kwargs: dict[str, Any] = {
            "pop": self.population,
            "init_hp": manifest,
            "max_steps": self.training_spec.max_steps,
            "evo_steps": evo_steps,
            "selection_strategy": self.selection_strategy,
            "mutation": self.mutations,
            "save_elite": save_elite,
            "elite_path": elite_path,
            "wb": wb,
            "tensorboard": tensorboard,
            "tensorboard_log_dir": tensorboard_log_dir,
            "verbose": verbose,
            "wandb_api_key": wandb_api_key,
            "wandb_kwargs": wandb_kwargs,
        }

        if self._multiturn:
            assert isinstance(self.env_spec, LLMEnvSpec)
            kwargs["env_factory"] = self.env_factory
            kwargs["max_turns"] = self.env_spec.max_turns
            manifest["env_name"] = self.env_spec.name
            if self.training_spec.max_wall_seconds is not None:
                kwargs["max_wall_seconds"] = self.training_spec.max_wall_seconds
        else:
            kwargs["env"] = self.env
            if self.training_spec.max_wall_seconds is not None:
                warnings.warn(
                    "max_wall_seconds is only supported by multi-turn LLM "
                    "fine-tuning and will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

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
                n_step_memory=self.n_step_memory,
            )
        )
        return self.train_fn(**kwargs)


class ArenaTrainer(Trainer):
    """Submits AgileRL training jobs to the Arena RLOps platform.

    :param algorithm: An `:class:`AlgorithmSpec` instance or a string algorithm name.
    :type algorithm: AlgoSpec | str
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
    :param selection_strategy: Tournament selection configuration. Arena runs
        tournament selection only, so MF-PBT is rejected. Defaults to ``None``.
    :type selection_strategy: TournamentSelectionSpec | None
    :param replay_buffer: Replay buffer configuration. Defaults to ``None``.
    :type replay_buffer: ReplayBufferType | None
    :param kwargs: Accepts the deprecated tournament alias for
        selection_strategy.
    """

    def __init__(
        self,
        algorithm: AlgoSpec | str,
        environment: ArenaEnvSpec | str,
        training: TrainingSpec | None = None,
        *,
        client: ArenaClient | None = None,
        api_key: str | None = None,
        mutation: MutationSpec | None = None,
        selection_strategy: TournamentSelectionSpec | None = None,
        replay_buffer: ReplayBufferType | None = None,
        **kwargs: Any,
    ) -> None:

        resolved_selection = resolve_deprecated_selection_kwargs(
            selection_strategy, kwargs, caller=type(self).__name__
        )
        if isinstance(resolved_selection, MultiFrequencySelectionSpec):
            msg = (
                "ArenaTrainer only supports tournament selection: multi-frequency "
                "selection (MF-PBT) is not available on the Arena platform. Use "
                "LocalTrainer to run MF-PBT."
            )
            raise ValueError(msg)
        selection_strategy = resolved_selection

        if isinstance(environment, str):
            if ArenaEnvSpec is None:
                msg = (
                    "Arena dependencies are not installed. "
                    "Please install them using: pip install agilerl-arena"
                )
                raise ImportError(msg)

            environment = ArenaEnvSpec(name=environment)

        # Arena specs mirror the core ones on the server side; they are a separate
        # model hierarchy, so they are not members of `EnvSpecType`/`AlgoSpec` and are
        # bridged through Any for the base constructor.
        arena_environment: Any = environment
        super().__init__(
            algorithm,
            arena_environment,
            training=training,
            mutation=mutation,
            selection_strategy=selection_strategy,
            replay_buffer=replay_buffer,
        )

        if client is not None:
            self._client = client
        else:
            if not HAS_ARENA_DEPENDENCIES or ArenaClient is None:
                msg = (
                    "Arena dependencies are not installed. "
                    "Please install them using: pip install agilerl-arena"
                )
                raise ImportError(msg)

            self._client = ArenaClient(api_key=api_key)

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path | dict[str, Any] | TrainingManifest,
        **kwargs: Any,
    ) -> Self:
        """Instantiate a :class:`ArenaTrainer` from a YAML, JSON, or dict manifest.

        Automatically dispatches to the correct Pydantic models based on the manifest
        fields.

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any] | TrainingManifest
        :param kwargs: Arena-specific construction arguments.  Recognises ``client``
            (an authenticated :class:`ArenaClient`, created automatically when
            omitted) and ``api_key`` (the Arena API key).
        :returns: A fully configured :class:`ArenaTrainer` instance.
        :rtype: ArenaTrainer
        """
        if ArenaManifest is None:
            msg = (
                "Arena dependencies are not installed. "
                "Please install them using: pip install agilerl-arena"
            )
            raise ImportError(msg)

        client: ArenaClient | None = kwargs.get("client")
        api_key: str | None = kwargs.get("api_key")

        # Arena training is driven by the Arena manifest schema, so a pre-validated
        # core :class:`TrainingManifest` instance cannot be submitted directly.
        if isinstance(manifest, TrainingManifest):
            msg = (
                "ArenaTrainer.from_manifest expects a serialized manifest "
                "(a path, JSON string, or dict), not a core TrainingManifest instance."
            )
            raise TypeError(msg)

        validated_manifest = ArenaManifest.get_validated(manifest, mode="python")
        env_spec = cls._resolve_env_spec(validated_manifest)

        # Arena specs mirror the core ones on the server side but form a separate
        # model hierarchy, so they are bridged through `Any` for the base
        # constructor, as `__init__` does for the environment spec.
        algorithm: Any = validated_manifest.algorithm
        training: Any = validated_manifest.training
        mutation: Any = validated_manifest.mutation
        selection_strategy: Any = validated_manifest.selection_strategy
        replay_buffer: Any = validated_manifest.replay_buffer

        # Deferred network (``arch`` omitted): the arena manifest leaves
        # ``net_config`` unset and keeps the raw network section. Carry it on the
        # spec so it is submitted for the server to resolve, not dropped.
        if (
            "net_config" in type(algorithm).model_fields
            and algorithm.net_config is None
            and validated_manifest.network
        ):
            algorithm.net_config = validated_manifest.network

        return cls(
            algorithm=algorithm,
            environment=env_spec,
            client=client,
            api_key=api_key,
            training=training,
            mutation=mutation,
            selection_strategy=selection_strategy,
            replay_buffer=replay_buffer,
        )

    @staticmethod
    def _resolve_env_spec(manifest: Any) -> ArenaEnvSpec:  # noqa: ANN401 -- Arena manifest is a distinct schema; a concrete type would break the base-method override
        """Build an :class:`ArenaEnvSpec` from the manifest.

        :param manifest: The validated training manifest.
        :type manifest: TrainingManifest
        :returns: An environment spec for training on a validated Arena environment.
        :rtype: ArenaEnvSpec
        """
        if ArenaEnvSpec is None:
            msg = (
                "Arena dependencies are not installed. "
                "Please install them using: pip install agilerl-arena"
            )
            raise ImportError(msg)

        env_data = manifest.environment

        if env_data.get("name") is None:
            msg = "Environment name is required for Arena training."
            raise ValueError(msg)

        return ArenaEnvSpec(
            name=env_data.get("name", ""),
            num_envs=env_data.get("num_envs", 16),
            version=str(env_data.get("version")),
        )

    def to_manifest(self) -> dict[str, Any]:
        """Build an Arena submission manifest from the :class:`ArenaTrainer` instance.

        :returns: A JSON-serializable manifest validated with
            :class:`~agilerl.arena.models.TrainingManifest`.
        :rtype: dict[str, Any]
        """
        manifest = TrainingManifest.from_trainer_specs(
            algorithm=self.algorithm_spec,
            environment=self.env_spec,
            training=self.training_spec,
            mutation=self.mutation_spec,
            replay_buffer=self.replay_buffer_spec,
            selection_strategy=self.selection_strategy_spec,
        )
        return TrainingManifest.to_arena_manifest(manifest)

    def train(
        self,
        resource_id: str | int | None = None,
        num_nodes: int | None = None,
        project: str | None = None,
        experiment_name: str | None = None,
        reward_file: str | Path | bytes | None = None,
        completion: str | None = None,
    ) -> dict[str, Any]:
        """Build the manifest and submit the training job to Arena.

        :param resource_id: Arena cluster type or resource id for the job.
        :type resource_id: str | int | None
        :param num_nodes: The number of nodes to use for training.
        :type num_nodes: int | None
        :param project: The project to submit the experiment to.
        :type project: str | None
        :param experiment_name: The name of the experiment to submit.
        :type experiment_name: str | None
        :param reward_file: Python reward module for reasoning dataset jobs.
        :type reward_file: str | Path | bytes | None
        :param completion: Optional model completion for reward validation.
        :type completion: str | None
        :returns: Arena API response.
        :rtype: dict[str, Any]
        """
        return self._client.submit_experiment(
            self.to_manifest(),
            resource_id=resource_id,
            num_nodes=num_nodes,
            project=project,
            experiment_name=experiment_name,
            reward_file=reward_file,
            completion=completion,
        )

    def resume_from_checkpoint(self, job_id: str, max_steps: int) -> dict[str, Any]:
        """Resume a training job from a checkpoint.

        :param job_id: The ID of the training job to resume from.
        :type job_id: str
        :param max_steps: The maximum number of steps to train for.
        :type max_steps: int
        :returns: Arena API response.
        :rtype: dict[str, Any]
        """
        return self._client.resume_experiment(job_id, max_steps)

    def list_experiments(self, project: str) -> list[dict[str, Any]]:
        """List all experiments in the project.

        :param project: The name of the project to list experiments for.
        :type project: str
        :returns: A list of experiments.
        :rtype: list[dict[str, Any]]
        """
        return self._client.list_experiments(project)

    def list_checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        """List all checkpoints for a training job.

        :param job_id: The ID of the training job to list checkpoints for.
        :type job_id: str
        :returns: A list of checkpoints.
        :rtype: list[dict[str, Any]]
        """
        return self._client.list_checkpoints(job_id)
