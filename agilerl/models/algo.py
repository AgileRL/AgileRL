# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from agilerl import HAS_LLM_DEPENDENCIES, algorithms
from agilerl.algorithms.core import (
    LLMAlgorithm,
    MultiAgentRLAlgorithm,
    RLAlgorithm,
)
from agilerl.algorithms.core.registry import HyperparameterConfig
from agilerl.arena import AgentType
from agilerl.arena.models.algo import AlgorithmSpec as ArenaAlgorithmSpec
from agilerl.builders import LLMBuilder, MultiAgentBuilder, SingleAgentBuilder
from agilerl.builders.base import AlgorithmBuildRuntime

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator
    from gymnasium import spaces
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from agilerl.components.replay_buffer import BufferType
    from agilerl.models.env import (
        BanditEnvSpec,
        GymEnvSpec,
        LLMEnvSpec,
        OfflineEnvSpec,
        PzEnvSpec,
    )
    from agilerl.models.env_types import LLMEnvType
    from agilerl.models.training import TrainingSpec
    from agilerl.strategies.base import TrainingLoop

    if HAS_LLM_DEPENDENCIES:
        from peft import LoraConfig

    AnyAlgorithm = RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm[Any]
    AlgoT = TypeVar("AlgoT", bound="AnyAlgorithm")
    EnvSpecType = GymEnvSpec | PzEnvSpec | OfflineEnvSpec | LLMEnvSpec | BanditEnvSpec
else:
    LoraConfig = Any
    AnyAlgorithm = Any
    AlgoT = TypeVar("AlgoT")


logger = logging.getLogger(__name__)

# TypeVar over the AlgoSpec union so registration decorators return the concrete spec subclass.
AlgoSpecT = TypeVar("AlgoSpecT", bound="AlgoSpec")


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """A single entry in the algorithm registry.

    :param spec_cls: The algorithm spec class.
    """

    spec_cls: type[AlgoSpec]


class AlgorithmRegistry:
    """Central registry mapping algorithm names to their spec classes.

    Populated at import time by the :func:`register` decorator applied to
    each concrete :class:`AlgorithmSpec` subclass.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def add(self, name: str, spec_cls: type[AlgoSpec]) -> None:
        """Register a spec class under *name*.

        :param name: Algorithm name (e.g. ``"DQN"``).
        :type name: str
        :param spec_cls: The spec class to register.
        :type spec_cls: type[AlgoSpec]
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


def register() -> Callable[[type[AlgoSpecT]], type[AlgoSpecT]]:
    """Class decorator that registers an algorithm spec.

    The registry key is derived from the spec class name by stripping
    the ``"Spec"`` suffix (e.g. ``DQNSpec`` -> ``"DQN"``).

    :returns: The decorator function.
    :rtype: Callable[[type[AlgoSpecT]], type[AlgoSpecT]]

    Example::

        @register()
        class DQNSpec(RLAlgorithmSpec):
            ...
    """

    def decorator(spec_cls: type[AlgoSpecT]) -> type[AlgoSpecT]:
        name = spec_cls.__name__.removesuffix("Spec")
        ALGO_REGISTRY.add(name, spec_cls)
        return spec_cls

    return decorator


def off_policy() -> Callable[[type[AlgoSpecT]], type[AlgoSpecT]]:
    """Decorate an algorithm to mark it as off-policy.

    By doing this we automatically signal the use
    of a replay buffer and, optionally, epsilon decay during training.

    :return: Decorated algorithm spec class
    :rtype: Callable[[type[AlgoSpecT]], type[AlgoSpecT]]
    """

    def decorator(algo_spec_class: type[AlgoSpecT]) -> type[AlgoSpecT]:
        algo_spec_class.off_policy = True
        return algo_spec_class

    return decorator


def offline() -> Callable[[type[AlgoSpecT]], type[AlgoSpecT]]:
    """Decorate an algorithm to mark it as offline.

    Offline algorithms learn from a fixed dataset rather than
    interacting with the environment.  This flag signals that the
    trainer should create a replay buffer and pre-fill it with
    data from the dataset source declared in :class:`OfflineEnvSpec`.

    :return: Decorated algorithm spec class
    :rtype: Callable[[type[AlgoSpecT]], type[AlgoSpecT]]
    """

    def decorator(algo_spec_class: type[AlgoSpecT]) -> type[AlgoSpecT]:
        algo_spec_class.offline = True
        algo_spec_class.agent_type = AgentType.OfflineAgent
        return algo_spec_class

    return decorator


def bandit() -> Callable[[type[AlgoSpecT]], type[AlgoSpecT]]:
    """Decorate an algorithm to mark it as a contextual bandit.

    Bandit algorithms learn from tabular datasets wrapped as
    :class:`~agilerl.wrappers.learning.BanditEnv`.  They use a
    replay buffer and the :func:`~agilerl.training.train_bandits.train_bandits`
    training loop.

    :return: Decorated algorithm spec class
    :rtype: Callable[[type[AlgoSpecT]], type[AlgoSpecT]]
    """

    def decorator(algo_spec_class: type[AlgoSpecT]) -> type[AlgoSpecT]:
        algo_spec_class.bandit = True
        algo_spec_class.agent_type = AgentType.BanditAgent
        return algo_spec_class

    return decorator


class AlgorithmSpec(ArenaAlgorithmSpec):
    """Framework algorithm spec: arena fields plus construction.

    The algorithm class is resolved from ``agilerl.algorithms`` as
    ``<Name>Spec`` -> ``<Name>``. Training dispatch lives in
    :mod:`agilerl.strategies`.
    """

    hp_config: HyperparameterConfig | None = None

    off_policy: ClassVar[bool] = False
    offline: ClassVar[bool] = False
    bandit: ClassVar[bool] = False

    _algo_class_cache: ClassVar[
        type[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm] | None
    ] = None

    @classmethod
    def algo_class(cls) -> type[RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm]:
        """Resolve the algorithm class from ``agilerl.algorithms``."""
        if cls._algo_class_cache is None:
            cls._algo_class_cache = getattr(
                algorithms, cls.__name__.removesuffix("Spec")
            )
        return cls._algo_class_cache

    def build_algorithm(self) -> AlgoT:
        """Build the algorithm instance using spec fields + runtime args."""
        msg = "Algorithm specs must implement a build_algorithm method."
        raise NotImplementedError(msg)

    @classmethod
    def get_training_fn(cls) -> TrainingLoop:
        """Return the training loop for this spec.

        :return: Training function
        :rtype: TrainingLoop
        """
        from agilerl.strategies import select_strategy

        spec = cls.model_construct()
        if not isinstance(
            spec, (RLAlgorithmSpec, MultiAgentRLAlgorithmSpec, LLMAlgorithmSpec)
        ):
            msg = f"{cls.__name__} is not an algorithm spec."
            raise TypeError(msg)
        return select_strategy(spec).get_training_loop(spec)

    def get_training_kwargs(
        self,
        *,
        training: TrainingSpec,
        env_spec: EnvSpecType,
        memory: BufferType | None = None,
        n_step_memory: BufferType | None = None,
    ) -> dict[str, Any]:
        """Return additional kwargs for the training loop.

        :param training: Training specification.
        :type training: TrainingSpec
        :param env_spec: Environment specification.
        :type env_spec: EnvSpecType
        :param memory: Replay buffer instance.
        :type memory: BufferType | None
        :param n_step_memory: N-step replay buffer for combined PER + n-step setups.
        :type n_step_memory: BufferType | None
        :returns: Extra keyword arguments for the training function.
        :rtype: dict[str, Any]
        """
        from agilerl.strategies import select_strategy

        if not isinstance(
            self, (RLAlgorithmSpec, MultiAgentRLAlgorithmSpec, LLMAlgorithmSpec)
        ):
            msg = f"{type(self).__name__} is not an algorithm spec."
            raise TypeError(msg)
        return select_strategy(self).get_trainer_kwargs(
            self,
            training=training,
            env_spec=env_spec,
            memory=memory,
            n_step_memory=n_step_memory,
        )


class RLAlgorithmSpec(AlgorithmSpec):
    """Single-agent RL spec: construction methods.

    Concrete specs inherit the arena algorithm class for fields. This base
    does not inherit arena ``RLAlgorithmSpec``; that would put base-class
    field defaults ahead of the concrete arena spec in the MRO.
    """

    # Concrete specs declare ``net_config``; the trainer writes it through this base.
    if TYPE_CHECKING:
        net_config: Any = None

    @classmethod
    def algo_class(cls) -> type[RLAlgorithm]:
        """Resolve the concrete single-agent algorithm class for this spec."""
        resolved = super().algo_class()
        assert issubclass(resolved, RLAlgorithm)
        return resolved

    def build_algorithm(
        self,
        observation_space: spaces.Space | None = None,
        action_space: spaces.Space | None = None,
        index: int | None = None,
        resume_from_checkpoint: str | None = None,
        load_weights_from: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
    ) -> RLAlgorithm:
        """Build a single-agent algorithm instance from spec fields.

        :param observation_space: Observation space.
        :type observation_space: SupportedObservationSpace | None
        :param action_space: Action space.
        :type action_space: SupportedActionSpace | None
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
        :raises ValueError: If observation_space, action_space, or index is None.
        """
        return SingleAgentBuilder.build(
            self,
            observation_space=observation_space,
            action_space=action_space,
            runtime=AlgorithmBuildRuntime(
                index=index,
                device=device,
                accelerator=accelerator,
                resume_from_checkpoint=resume_from_checkpoint,
                load_weights_from=load_weights_from,
            ),
        )


class MultiAgentRLAlgorithmSpec(AlgorithmSpec):
    """Multi-agent RL spec: construction methods.

    Concrete specs inherit the arena algorithm class for fields. This base
    does not inherit arena ``MultiAgentRLAlgorithmSpec``; that would put
    base-class field defaults ahead of the concrete arena spec in the MRO.
    """

    # Concrete specs declare ``net_config``; the trainer writes it through this base.
    if TYPE_CHECKING:
        net_config: Any = None

    @classmethod
    def algo_class(cls) -> type[MultiAgentRLAlgorithm]:
        """Resolve the concrete multi-agent algorithm class for this spec."""
        resolved = super().algo_class()
        assert issubclass(resolved, MultiAgentRLAlgorithm)
        return resolved

    def build_algorithm(
        self,
        observation_spaces: dict[str, spaces.Space] | None = None,
        action_spaces: dict[str, spaces.Space] | None = None,
        index: int | None = None,
        resume_from_checkpoint: str | None = None,
        load_weights_from: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
    ) -> MultiAgentRLAlgorithm:
        """Build a multi-agent algorithm from spec fields.

        :param observation_spaces: Per-agent observation spaces.
        :type observation_spaces: dict[str, SupportedObservationSpace] | None
        :param action_spaces: Per-agent action spaces.
        :type action_spaces: dict[str, SupportedActionSpace] | None
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
        :raises ValueError: If observation_spaces, action_spaces, or index is None.
        """
        return MultiAgentBuilder.build(
            self,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            runtime=AlgorithmBuildRuntime(
                index=index,
                device=device,
                accelerator=accelerator,
                resume_from_checkpoint=resume_from_checkpoint,
                load_weights_from=load_weights_from,
            ),
        )


class LLMAlgorithmSpec(AlgorithmSpec):
    """LLM spec: construction methods plus PEFT ``lora_config``.

    Concrete specs inherit the arena algorithm class for fields. This base
    does not inherit arena ``LLMAlgorithmSpec``; that would put base-class
    field defaults ahead of the concrete arena spec in the MRO.
    """

    lora_config: LoraConfig | None = None

    # Arena LLM specs declare these; construction and the trainer read them here.
    if TYPE_CHECKING:
        env_type: ClassVar[LLMEnvType]
        objective: ClassVar[str | None]
        pretrained_model_name_or_path: str | None = None
        max_model_len: int = 1024
        seed: int = 42

    @classmethod
    def algo_class(cls) -> type[LLMAlgorithm]:
        """Resolve the concrete LLM algorithm class for this spec."""
        resolved = super().algo_class()
        assert issubclass(resolved, LLMAlgorithm)
        return resolved

    def build_algorithm(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        index: int = 0,
        resume_from_checkpoint: str | None = None,
        load_weights_from: str | None = None,
        accelerator: Accelerator | None = None,
        device: str | torch.device = "cpu",
        actor_network: Any | None = None,  # noqa: ANN401 -- concrete HF/PEFT models (PreTrainedModelType) forwarded here do not structurally satisfy PreTrainedModelProtocol under ty (device attr variance)
    ) -> LLMAlgorithm:
        """Build an LLM algorithm instance from spec fields.

        :param tokenizer: A HuggingFace ``AutoTokenizer`` instance.
        :type tokenizer: PreTrainedTokenizerBase | None
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
        :raises ValueError: If tokenizer is None.
        """
        return LLMBuilder.build(
            self,
            tokenizer=tokenizer,
            actor_network=actor_network,
            runtime=AlgorithmBuildRuntime(
                index=index,
                device=device,
                accelerator=accelerator,
                resume_from_checkpoint=resume_from_checkpoint,
                load_weights_from=load_weights_from,
            ),
        )


AlgoSpec = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec
