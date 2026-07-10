from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agilerl import HAS_LLM_DEPENDENCIES, AgentType

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator

    from agilerl.algorithms.core import (
        LLMAlgorithm,
        MultiAgentRLAlgorithm,
        RLAlgorithm,
    )
    from agilerl.algorithms.core.registry import HyperparameterConfig
    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.models.env import (
        BanditEnvSpec,
        GymEnvSpec,
        LLMEnvSpec,
        LLMEnvType,
        OfflineEnvSpec,
        PzEnvSpec,
    )
    from agilerl.models.training import TrainingSpec
    from agilerl.typing import SupportedActionSpace, SupportedObservationSpace

    if HAS_LLM_DEPENDENCIES:
        from peft import LoraConfig

    AlgoT = TypeVar("AlgoT", bound="RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm")
    EnvSpecT = GymEnvSpec | PzEnvSpec | OfflineEnvSpec | LLMEnvSpec | BanditEnvSpec
    ReplayBufferT = ReplayBuffer
    PopulationT = list[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]
else:
    HyperparameterConfig = Any
    LoraConfig = Any
    AlgoT = TypeVar("AlgoT")


logger = logging.getLogger(__name__)

AlgoSpecTV = TypeVar("AlgoSpecTV", bound="AlgorithmSpec")


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """A single entry in the algorithm registry.

    :param spec_cls: The algorithm spec class.
    """

    spec_cls: type[AlgorithmSpec]


class AlgorithmRegistry:
    """Central registry mapping algorithm names to their spec classes.

    Populated at import time by the :func:`register` decorator applied to
    each concrete :class:`AlgorithmSpec` subclass.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def add(self, name: str, spec_cls: type[AlgorithmSpec]) -> None:
        """Register a spec class under *name*.

        :param name: Algorithm name (e.g. ``"DQN"``).
        :type name: str
        :param spec_cls: The spec class to register.
        :type spec_cls: type[AlgorithmSpec]
        """
        if name in self._entries:
            logger.warning("Overriding existing registration for algorithm %r", name)

        self._entries[name] = RegistryEntry(spec_cls=spec_cls)

    def get(self, name: str) -> RegistryEntry:
        """Look up an entry by algorithm name.

        :param name: Algorithm name.
        :type name: str
        :returns: The registry entry.
        :rtype: RegistryEntry
        :raises KeyError: If *name* is not registered.
        """
        try:
            return self._entries[name]
        except KeyError as err:
            supported = ", ".join(sorted(self._entries))
            msg = f"No registry entry for algorithm {name!r}. Registered: {supported}"
            raise KeyError(msg) from err


ALGO_REGISTRY = AlgorithmRegistry()


def register() -> Callable[[type[AlgorithmSpec]], type[AlgorithmSpec]]:
    """Class decorator that registers an algorithm spec.

    The registry key is derived from the spec class name by stripping
    the ``"Spec"`` suffix (e.g. ``DQNSpec`` -> ``"DQN"``).

    :returns: The decorator function.
    :rtype: Callable[[type[AlgorithmSpec]], type[AlgorithmSpec]]

    Example::

        @register()
        class DQNSpec(RLAlgorithmSpec):
            ...
    """

    def decorator(spec_cls: type[AlgorithmSpec]) -> type[AlgorithmSpec]:
        name = spec_cls.__name__.removesuffix("Spec")
        ALGO_REGISTRY.add(name, spec_cls)
        return spec_cls

    return decorator


def off_policy() -> Callable[[type[AlgoSpecTV]], type[AlgoSpecTV]]:
    """Decorate an algorithm to mark it as off-policy.

    By doing this we automatically signal the use
    of a replay buffer and, optionally, epsilon decay during training.

    :return: Decorated algorithm spec class
    :rtype: Callable[[type[AlgoSpecTV]], type[AlgoSpecTV]]
    """

    def decorator(algo_spec_class: type[AlgoSpecTV]) -> type[AlgoSpecTV]:
        algo_spec_class.off_policy = True
        return algo_spec_class

    return decorator


def offline() -> Callable[[type[AlgoSpecTV]], type[AlgoSpecTV]]:
    """Decorate an algorithm to mark it as offline.

    Offline algorithms learn from a fixed dataset rather than
    interacting with the environment.  This flag signals that the
    trainer should create a replay buffer and pre-fill it with
    data from the dataset source declared in :class:`OfflineEnvSpec`.

    :return: Decorated algorithm spec class
    :rtype: Callable[[type[AlgoSpecTV]], type[AlgoSpecTV]]
    """

    def decorator(algo_spec_class: type[AlgoSpecTV]) -> type[AlgoSpecTV]:
        algo_spec_class.offline = True
        algo_spec_class.agent_type = AgentType.OfflineAgent
        return algo_spec_class

    return decorator


def bandit() -> Callable[[type[AlgoSpecTV]], type[AlgoSpecTV]]:
    """Decorate an algorithm to mark it as a contextual bandit.

    Bandit algorithms learn from tabular datasets wrapped as
    :class:`~agilerl.wrappers.learning.BanditEnv`.  They use a
    replay buffer and the :func:`~agilerl.training.train_bandits.train_bandits`
    training loop.

    :return: Decorated algorithm spec class
    :rtype: Callable[[type[AlgoSpecTV]], type[AlgoSpecTV]]
    """

    def decorator(algo_spec_class: type[AlgoSpecTV]) -> type[AlgoSpecTV]:
        algo_spec_class.bandit = True
        algo_spec_class.agent_type = AgentType.BanditAgent
        return algo_spec_class

    return decorator


# TrainingSpec fields with no equivalent in the LLM fine-tuning loops
_LLM_UNSUPPORTED_TRAINING_FIELDS = (
    "target_score",
    "eval_steps",
    "eval_loop",
    "learning_delay",
    "eps_start",
    "eps_end",
    "eps_decay",
    "overwrite_checkpoints",
)


def _warn_ignored_llm_training_fields(training: TrainingSpec) -> None:
    """Warn when explicitly-set TrainingSpec fields are ignored by LLM loops."""
    ignored = [
        name
        for name in _LLM_UNSUPPORTED_TRAINING_FIELDS
        if name in training.model_fields_set
        and getattr(training, name) != type(training).model_fields[name].default
    ]
    if ignored:
        warnings.warn(
            "TrainingSpec fields not supported by LLM fine-tuning are ignored: "
            + ", ".join(ignored),
            UserWarning,
            stacklevel=3,
        )


# Values whose type makes a meaningful ``!=`` comparison; anything else (an
# Accelerator, a LoraConfig, a registry) is skipped when diffing hyperparameters.
_COMPARABLE_HYPERPARAMETERS = (bool, int, float, str, type(None))


def _apply_checkpoint(
    algo: AlgoT,
    resume_from_checkpoint: str | None,
    load_weights_from: str | None,
) -> None:
    """Seed a freshly-built agent from a checkpoint, if asked to.

    The two options are different operations and must not be combined:

    * ``resume_from_checkpoint`` continues an interrupted run. The checkpoint is
      authoritative -- optimizer moments, the learning-rate schedule position and
      the hyperparameters they were computed under all come back, so the run picks
      up exactly where it stopped. A hyperparameter that disagrees with the spec is
      warned about, because the spec no longer describes what will be trained.
    * ``load_weights_from`` starts a *new* run from a previous one's parameters (a
      warm start). Only the weights are taken; the optimizer, schedule and progress
      start fresh, and the spec keeps every hyperparameter.

    :param algo: A freshly-built algorithm, configured from its spec.
    :type algo: AlgoT
    :param resume_from_checkpoint: Checkpoint to resume the run from.
    :type resume_from_checkpoint: str | None
    :param load_weights_from: Checkpoint to take weights from.
    :type load_weights_from: str | None
    """
    if resume_from_checkpoint is not None and load_weights_from is not None:
        msg = (
            "Provide exactly one of 'resume_from_checkpoint' (continue a run, "
            "restoring optimizer state and its hyperparameters) or "
            "'load_weights_from' (warm-start a new run from prior weights)."
        )
        raise ValueError(msg)

    if load_weights_from is not None:
        algo.load_weights(load_weights_from)
    elif resume_from_checkpoint is not None:
        _resume_and_warn_on_drift(algo, resume_from_checkpoint)


def _resume_and_warn_on_drift(algo: AlgoT, path: str) -> None:
    """Restore a checkpoint, warning about hyperparameters it overrode.

    Resuming keeps the checkpoint's hyperparameters, because they are the ones the
    restored optimizer state and lr schedule belong to. That silently diverges from
    the spec if the two disagree, so say so.

    :param algo: A freshly-built algorithm, configured from its spec.
    :type algo: AlgoT
    :param path: Checkpoint to resume from.
    :type path: str
    """
    from agilerl.algorithms.core.base import EvolvableAlgorithm

    configured = EvolvableAlgorithm.inspect_attributes(algo, input_args_only=True)

    algo.load_checkpoint(path)

    drifted = {
        name: (configured[name], getattr(algo, name))
        for name in configured
        if isinstance(configured[name], _COMPARABLE_HYPERPARAMETERS)
        and hasattr(algo, name)
        and configured[name] != getattr(algo, name)
    }
    if drifted:
        changes = ", ".join(
            f"{name}: {new!r} (checkpoint) overrides {old!r} (spec)"
            for name, (old, new) in sorted(drifted.items())
        )
        warnings.warn(
            f"Resuming from {path} restored hyperparameters that differ from the "
            f"spec, and the checkpoint's values win because the optimizer state "
            f"belongs to them -- {changes}. Update the spec to match, or use "
            f"'load_weights_from' to warm-start with the spec's values instead.",
            UserWarning,
            stacklevel=3,
        )


class AlgorithmSpec(BaseModel):
    """Base specification for all algorithms.

    Defines common fields and behavior for algorithm specifications, including
    batch size and hyperparameter configuration.  Concrete subclasses must set
    the ``agent_type`` class variable and override :meth:`get_training_fn`.

    The algorithm class is resolved lazily from ``agilerl.algorithms`` using
    the naming convention ``<Name>Spec`` -> ``<Name>`` (e.g. ``PPOSpec`` ->
    ``PPO``).  This avoids importing heavy dependencies at spec-import time.
    """

    batch_size: int = Field(default=128, ge=1)
    hp_config: HyperparameterConfig | None = None

    off_policy: ClassVar[bool] = False
    offline: ClassVar[bool] = False
    bandit: ClassVar[bool] = False
    default_evo_steps: ClassVar[int] = 10_000

    _algo_class_cache: ClassVar[type | None] = None

    agent_type: ClassVar[AgentType]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def algo_class(cls) -> type[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]:
        """Lazily resolve the algorithm class from ``agilerl.algorithms``."""
        if cls._algo_class_cache is None:
            from agilerl import algorithms

            cls._algo_class_cache = getattr(
                algorithms, cls.__name__.removesuffix("Spec")
            )
        return cls._algo_class_cache

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.__class__.__name__.removesuffix("Spec")

    def build_algorithm(self) -> AlgoT:
        """Build the algorithm instance using spec fields + runtime args."""
        msg = "Algorithm specs must implement a build_algorithm method."
        raise NotImplementedError(msg)

    @staticmethod
    def get_training_fn() -> Callable[..., tuple[PopulationT, list[float]]]:
        """Return the training function for this algorithm.

        Concrete specs **must** override this to return their training
        function (e.g. ``train_off_policy``).

        :return: Training function
        :rtype: Callable[..., tuple[PopulationT, list[float]]]
        :raises NotImplementedError: If the training function is not implemented.
        """
        msg = "Algorithm specs must implement get_training_fn."
        raise NotImplementedError(msg) from None

    def get_training_kwargs(
        self,
        *,
        training: TrainingSpec,
        env_spec: EnvSpecT,
        memory: ReplayBufferT = None,
        n_step_memory: ReplayBufferT = None,
    ) -> dict[str, Any]:
        """Return additional kwargs for the training loop.

        :param training: Training specification.
        :type training: TrainingSpec
        :param env_spec: Environment specification.
        :type env_spec: EnvSpecT
        :param memory: Replay buffer instance.
        :type memory: ReplayBufferT | None
        :param n_step_memory: N-step replay buffer for combined PER + n-step setups.
        :type n_step_memory: ReplayBufferT | None
        :returns: Extra keyword arguments for the training function.
        :rtype: dict[str, Any]
        """
        kwargs = {}
        if isinstance(self, LLMAlgorithmSpec):
            if env_spec.max_reward is not None:
                kwargs["max_reward"] = env_spec.max_reward

            if training.checkpoint_steps is not None:
                kwargs["checkpoint_steps"] = training.checkpoint_steps

            if training.checkpoint_path is not None:
                kwargs["checkpoint_path"] = training.checkpoint_path

            kwargs["evaluation_interval"] = training.evaluation_interval
            if training.num_epochs is not None:
                kwargs["num_epochs"] = training.num_epochs

            _warn_ignored_llm_training_fields(training)
            return kwargs

        # Core RL algorithm kwargs
        kwargs.update(
            {
                "env_name": env_spec.name,
                "algo": self.name,
                "eval_steps": training.eval_steps,
                "eval_loop": training.eval_loop,
                "target": training.target_score,
                "checkpoint": training.checkpoint_steps,
                "checkpoint_path": training.checkpoint_path,
                "overwrite_checkpoints": training.overwrite_checkpoints,
            }
        )

        if self.off_policy or self.offline or self.bandit:
            kwargs["memory"] = memory

        if self.off_policy:
            kwargs["learning_delay"] = training.learning_delay
            if training.eps_start is not None:
                kwargs["eps_start"] = training.eps_start
            if training.eps_end is not None:
                kwargs["eps_end"] = training.eps_end
            if training.eps_decay is not None:
                kwargs["eps_decay"] = training.eps_decay
            if n_step_memory is not None:
                kwargs["n_step_memory"] = n_step_memory
        elif self.offline:
            if env_spec.minari_dataset_id is not None:
                kwargs["minari_dataset_id"] = env_spec.minari_dataset_id
                kwargs["remote"] = env_spec.remote
            elif env_spec.dataset_path is not None:
                import h5py

                kwargs["dataset"] = h5py.File(env_spec.dataset_path, "r")
        if self.bandit:
            kwargs["episode_steps"] = training.episode_steps
        if self.agent_type == AgentType.MultiAgent:
            kwargs["sum_scores"] = training.sum_scores

        return kwargs


class RLAlgorithmSpec(AlgorithmSpec):
    """Specification for single-agent reinforcement learning algorithms.

    Extends :class:`AlgorithmSpec` with single-agent specific fields like
    network configuration, learning step frequency, and discount factor.
    """

    learn_step: int = Field(default=5, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)

    agent_type: ClassVar[AgentType] = AgentType.SingleAgent

    def build_algorithm(
        self,
        observation_space: SupportedObservationSpace,
        action_space: SupportedActionSpace,
        index: int,
        resume_from_checkpoint: str | None = None,
        load_weights_from: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
    ) -> RLAlgorithm:
        """Build a single-agent algorithm instance from spec fields.

        :param observation_space: Observation space.
        :type observation_space: SupportedObservationSpace
        :param action_space: Action space.
        :type action_space: SupportedActionSpace
        :param index: Index of the algorithm in the population.
        :type index: int
        :param resume_from_checkpoint: Checkpoint to continue an interrupted run
            from, restoring optimizer state and the hyperparameters it belongs to.
            Mutually exclusive with ``load_weights_from``.
        :type resume_from_checkpoint: str | None
        :param load_weights_from: Checkpoint to warm-start a new run from, taking
            only the weights. Mutually exclusive with ``resume_from_checkpoint``.
        :type load_weights_from: str | None
        :param device: Torch device. Defaults to "cpu".
        :type device: str | torch.device
        :param accelerator: Accelerator object for distributed computing.
        :type accelerator: Accelerator | None
        :returns: Single-agent algorithm instance.
        :rtype: RLAlgorithm
        """
        algo_cls = self.algo_class()
        algo = algo_cls(
            observation_space=observation_space,
            action_space=action_space,
            index=index,
            device=device,
            accelerator=accelerator,
            **self.model_dump(mode="python", exclude_unset=True),
        )

        _apply_checkpoint(algo, resume_from_checkpoint, load_weights_from)

        return algo


class MultiAgentRLAlgorithmSpec(AlgorithmSpec):
    """Specification for multi-agent reinforcement learning algorithms.

    Extends :class:`AlgorithmSpec` with multi-agent specific fields and
    support for multiple observation/action spaces and agent IDs.
    """

    learn_step: int = Field(default=2048, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    torch_compiler: str | None = Field(default=None)

    agent_type: ClassVar[AgentType] = AgentType.MultiAgent

    def build_algorithm(
        self,
        observation_spaces: dict[str, SupportedObservationSpace],
        action_spaces: dict[str, SupportedActionSpace],
        index: int,
        resume_from_checkpoint: str | None = None,
        load_weights_from: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
    ) -> MultiAgentRLAlgorithm:
        """Build a multi-agent algorithm from spec fields.

        :param observation_spaces: Per-agent observation spaces.
        :type observation_spaces: dict[str, SupportedObservationSpace]
        :param action_spaces: Per-agent action spaces.
        :type action_spaces: dict[str, SupportedActionSpace]
        :param index: Index of the algorithm in the population.
        :type index: int
        :param resume_from_checkpoint: Checkpoint to continue an interrupted run
            from, restoring optimizer state and the hyperparameters it belongs to.
            Mutually exclusive with ``load_weights_from``.
        :type resume_from_checkpoint: str | None
        :param load_weights_from: Checkpoint to warm-start a new run from, taking
            only the weights. Mutually exclusive with ``resume_from_checkpoint``.
        :type load_weights_from: str | None
        :param device: Torch device. Defaults to "cpu".
        :type device: str | torch.device
        :param accelerator: Accelerator object for distributed computing.
        :type accelerator: Accelerator | None
        :returns: Multi-agent algorithm instance.
        :rtype: MultiAgentRLAlgorithm
        """
        algo_cls = self.algo_class()
        algo = algo_cls(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            index=index,
            device=device,
            accelerator=accelerator,
            **self.model_dump(mode="python", exclude_unset=True),
        )

        _apply_checkpoint(algo, resume_from_checkpoint, load_weights_from)

        return algo


class LLMAlgorithmSpec(AlgorithmSpec):
    """Specification for LLM fine-tuning algorithms.

    Extends :class:`AlgorithmSpec` with LLM-specific fields including LoRA
    configuration, model parameters, and training hyperparameters.

    Subclasses must set the :attr:`env_type` class variable to indicate which
    LLM env the algorithm requires: ``"rollout"`` for a
    :class:`~agilerl.llm_envs.RolloutEnv`, or ``"dataset"`` for a
    :class:`~agilerl.llm_envs.DatasetEnv` (which additionally sets
    :attr:`objective` to ``"preference"`` or ``"sft"``).
    """

    beta: float = Field(default=0.001, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.1, ge=0.0)
    update_epochs: int = Field(default=1, ge=1)
    reduce_memory_peak: bool = Field(default=False)
    use_separate_reference_adapter: bool = Field(default=False)
    calc_position_embeddings: bool = Field(default=True)
    gradient_checkpointing: bool = Field(default=True)
    use_liger_loss: bool = Field(default=False)
    seed: int = Field(default=42)
    quantization: str | dict[str, Any] | None = Field(default=None)
    activation_offload: bool = Field(default=False)
    use_sequence_packing: bool = Field(default=False)
    lora_target_scope: str | None = Field(default=None)
    fused_logprobs_chunk_rows: int | None = Field(default=None)
    fused_loss_chunk_rows: int | None = Field(default=None)
    vllm_importance_sampling_correction: bool = Field(default=True)
    vllm_importance_sampling_cap: float = Field(default=2.0, ge=0.0)
    attn_implementation: str | None = Field(default=None)

    # These fields come from the "network" section of the manifest
    pretrained_model_name_or_path: str | None = Field(default=None, min_length=1)
    max_model_len: int = Field(default=1024, ge=1)
    lora_config: LoraConfig | None = Field(default=None)

    agent_type: ClassVar[AgentType] = AgentType.LLMAgent
    default_evo_steps: ClassVar[int] = 5
    env_type: ClassVar[LLMEnvType]
    objective: ClassVar[str | None] = None

    def build_algorithm(
        self,
        tokenizer: Any,
        index: int = 0,
        resume_from_checkpoint: str | None = None,
        load_weights_from: str | None = None,
        accelerator: Accelerator | None = None,
        device: str | torch.device = "cpu",
        actor_network: Any | None = None,
    ) -> LLMAlgorithm:
        """Build an LLM algorithm instance from spec fields.

        :param tokenizer: A HuggingFace ``AutoTokenizer`` instance.
        :type tokenizer: Any
        :param index: Index of the algorithm in the population.
        :type index: int
        :param resume_from_checkpoint: Checkpoint to continue an interrupted run
            from, restoring optimizer state and the hyperparameters it belongs to.
            Mutually exclusive with ``load_weights_from``.
        :type resume_from_checkpoint: str | None
        :param load_weights_from: Checkpoint to warm-start a new run from, taking
            only the weights. Mutually exclusive with ``resume_from_checkpoint``.
        :type load_weights_from: str | None
        :param accelerator: HuggingFace ``Accelerator`` instance.
        :type accelerator: Accelerator | None
        :param device: Torch device. Defaults to "cpu".
        :type device: str | torch.device
        :param actor_network: Pre-built or cloned actor network. When provided,
            this is passed directly to the algorithm constructor instead of loading
            the model from ``pretrained_model_name_or_path``.
        :type actor_network: Any | None
        :returns: LLM algorithm instance.
        :rtype: LLMAlgorithm
        """
        micro_batch_size_per_gpu = None
        if accelerator is not None:
            micro_batch_size_per_gpu = max(
                self.batch_size // accelerator.num_processes, 1
            )

        use_vllm = getattr(self, "use_vllm", False)
        if not use_vllm and hasattr(self, "vllm_config"):
            self.vllm_config = None

        vllm_cfg = getattr(self, "vllm_config", None)
        if isinstance(vllm_cfg, dict):
            from agilerl.utils.algo_utils import VLLMConfig

            self.vllm_config = VLLMConfig(**vllm_cfg)

        # Only forward explicitly-set fields so the algorithm's own defaults
        # apply to everything a manifest omits, matching direct construction.
        kwargs = {k: v for k, v in vars(self).items() if k in self.model_fields_set}
        kwargs.pop("pretrained_model_name_or_path", None)
        if not use_vllm:
            kwargs.pop("max_model_len", None)

        # Resolve trainer-side bitsandbytes quantization (a preset name or a
        # BitsAndBytesConfig kwargs dict) to the quantization_config the
        # algorithm constructor expects.
        if "quantization" in kwargs:
            from agilerl.utils.llm_utils import build_bnb_quantization_config

            kwargs["quantization_config"] = build_bnb_quantization_config(
                kwargs.pop("quantization")
            )

        # A non-"auto" attn_implementation is forwarded through model_config so
        # the model-creation path treats it as authoritative.
        attn_implementation = kwargs.pop("attn_implementation", None)
        if attn_implementation is not None and attn_implementation != "auto":
            model_config = dict(kwargs.get("model_config") or {})
            model_config.setdefault("attn_implementation", attn_implementation)
            kwargs["model_config"] = model_config

        algo_cls = self.algo_class()
        algo = algo_cls(
            model_name=self.pretrained_model_name_or_path,
            pad_token_id=tokenizer.eos_token_id,
            pad_token=tokenizer.eos_token,
            accelerator=accelerator,
            index=index,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            device=device,
            actor_network=actor_network,
            **kwargs,
        )

        _apply_checkpoint(algo, resume_from_checkpoint, load_weights_from)

        return algo


AlgoSpecT = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec
