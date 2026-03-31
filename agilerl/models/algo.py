from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import LLMAlgorithm, MultiAgentRLAlgorithm, RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig
from agilerl.models.networks import NetworkSpec
from agilerl.protocols import AgentType
from agilerl.typing import SupportedActionSpace, SupportedObservationSpace

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import LoraConfig  # noqa: TC002

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

AlgoT = TypeVar("AlgoT", bound=RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm)
AlgoSpecT = TypeVar("AlgoSpecT", bound="AlgorithmSpec")


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """A single entry in the algorithm registry.

    :param spec_cls: The algorithm spec class.
    :param arena: Whether the algorithm is available for training on Arena.
    """

    spec_cls: type[AlgorithmSpec]
    arena: bool


class AlgorithmRegistry:
    """Central registry mapping algorithm names to their spec classes.

    Populated at import time by the :func:`register` decorator applied to
    each concrete :class:`AlgorithmSpec` subclass.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def add(self, name: str, spec_cls: type[AlgorithmSpec], *, arena: bool) -> None:
        """Register a spec class under *name*.

        :param name: Algorithm name (e.g. ``"DQN"``).
        :type name: str
        :param spec_cls: The spec class to register.
        :type spec_cls: type[AlgorithmSpec]
        :param arena: Whether the algorithm is available on Arena.
        :type arena: bool
        """
        if name in self._entries:
            logger.warning("Overriding existing registration for algorithm %r", name)
        self._entries[name] = RegistryEntry(spec_cls=spec_cls, arena=arena)

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

    def arena_algorithms(self) -> dict[str, RegistryEntry]:
        """Return the Pydantic models for algorithms available on Arena.

        :returns: The registry entries for Arena-eligible algorithms.
        :rtype: dict[str, RegistryEntry]
        """
        return {k: v for k, v in self._entries.items() if v.arena}

    def local_algorithms(self) -> dict[str, RegistryEntry]:
        """Return the Pydantic models for algorithms available locally.

        :returns: The registry entries for local-only algorithms.
        :rtype: dict[str, RegistryEntry]
        """
        return {k: v for k, v in self._entries.items() if not v.arena}


ALGO_REGISTRY = AlgorithmRegistry()


def register(
    arena: bool = False,
) -> Callable[[type[AlgorithmSpec]], type[AlgorithmSpec]]:
    """Class decorator that registers an algorithm spec.

    The registry key is derived from ``spec_cls.algo_class.__name__``.

    :param arena: Whether the algorithm is available for training on Arena.
    :type arena: bool

    :returns: The decorator function.
    :rtype: Callable[[type[AlgorithmSpec]], type[AlgorithmSpec]]

    Usage::

        @register(arena=True)
        class DQNSpec(RLAlgorithmSpec):
            algo_class: ClassVar[type[DQN]] = DQN
            ...
    """

    def decorator(spec_cls: type[AlgorithmSpec]) -> type[AlgorithmSpec]:
        name = spec_cls.algo_class.__name__
        ALGO_REGISTRY.add(name, spec_cls, arena=arena)
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


class AlgorithmSpec(BaseModel):
    """Base specification for all algorithms.

    Defines common fields and behavior for algorithm specifications, including
    batch size and hyperparameter configuration.  Concrete subclasses must set
    the ``algo_class`` and ``agent_type`` class variables, and override
    :meth:`get_training_loop`.
    """

    batch_size: int = Field(default=128, ge=1)
    hp_config: HyperparameterConfig | None = None
    off_policy: ClassVar[bool] = False

    algo_class: ClassVar[type[AlgoT]]

    agent_type: ClassVar[AgentType]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.__class__.__name__.rstrip("Spec")

    def build_algorithm(self) -> AlgoT:
        """Build the algorithm instance using spec fields + runtime args."""
        msg = "Algorithm specs must implement a build_algorithm method."
        raise NotImplementedError(msg)

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this spec for Arena manifest payloads."""
        return {
            "name": self.name,
            **self.model_dump(mode="json", exclude_none=True, exclude={"hp_config"}),
        }

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Return the training function for this algorithm.

        Concrete specs **must** override this to return their training
        function (e.g. ``train_off_policy``).

        :return: Training function
        :rtype: Callable[..., Any]
        :raises NotImplementedError: If the training function is not implemented.
        """
        msg = "Algorithm specs must implement get_training_fn."
        raise NotImplementedError(msg) from None

    def resolve_training_fn(self) -> Callable[..., Any]:
        """Return the training function, supporting legacy spec APIs."""
        try:
            return self.get_training_fn()
        except NotImplementedError:
            legacy = getattr(self, "get_training_loop", None)
            if callable(legacy):
                return legacy()
            raise


class RLAlgorithmSpec(AlgorithmSpec):
    """Specification for single-agent reinforcement learning algorithms.

    Extends :class:`AlgorithmSpec` with single-agent specific fields like
    network configuration, learning step frequency, and discount factor.
    """

    net_config: NetworkSpec | None = Field(default=None)
    learn_step: int = Field(default=5, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)

    agent_type: ClassVar[AgentType] = AgentType.SingleAgent

    def build_algorithm(
        self,
        observation_space: SupportedObservationSpace,
        action_space: SupportedActionSpace,
        index: int,
        device: torch.device,
    ) -> RLAlgorithm:
        """Build a single-agent algorithm instance from spec fields.

        :param observation_space: Observation space.
        :type observation_space: SupportedObservationSpace
        :param action_space: Action space.
        :type action_space: SupportedActionSpace
        :param index: Index of the algorithm in the population.
        :type index: int
        :param device: Torch device.
        :type device: torch.device
        :returns: Single-agent algorithm instance.
        :rtype: RLAlgorithm
        """
        return self.algo_class(
            observation_space=observation_space,
            action_space=action_space,
            index=index,
            device=device,
            **self.model_dump(),
        )

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this RL spec for Arena manifest payloads."""
        return {
            "name": self.name,
            **self.model_dump(
                mode="json",
                exclude_none=True,
                exclude={"hp_config", "net_config"},
            ),
        }


class MultiAgentRLAlgorithmSpec(AlgorithmSpec):
    """Specification for multi-agent reinforcement learning algorithms.

    Extends :class:`AlgorithmSpec` with multi-agent specific fields and
    support for multiple observation/action spaces and agent IDs.
    """

    net_config: NetworkSpec | None = Field(default=None)
    learn_step: int = Field(default=2048, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)

    agent_type: ClassVar[AgentType] = AgentType.MultiAgent

    def build_algorithm(
        self,
        observation_spaces: dict[str, SupportedObservationSpace],
        action_spaces: dict[str, SupportedActionSpace],
        index: int,
        device: torch.device,
    ) -> MultiAgentRLAlgorithm:
        """Build a multi-agent algorithm from spec fields.

        :param observation_spaces: Per-agent observation spaces.
        :type observation_spaces: dict[str, SupportedObservationSpace]
        :param action_spaces: Per-agent action spaces.
        :type action_spaces: dict[str, SupportedActionSpace]
        :param index: Index of the algorithm in the population.
        :type index: int
        :param device: Torch device.
        :type device: torch.device
        :returns: Multi-agent algorithm instance.
        :rtype: MultiAgentRLAlgorithm
        """
        return self.algo_class(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            index=index,
            device=device,
            **self.model_dump(),
        )

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this multi-agent spec for Arena manifest payloads."""
        return {
            "name": self.name,
            **self.model_dump(
                mode="json",
                exclude_none=True,
                exclude={"hp_config", "net_config"},
            ),
        }


class LoraConfigDict(TypedDict):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""

    lora_r: int
    lora_alpha: int
    lora_dropout: float
    task_type: str


class LLMAlgorithmSpec(AlgorithmSpec):
    """Specification for LLM fine-tuning algorithms.

    Extends :class:`AlgorithmSpec` with LLM-specific fields including LoRA
    configuration, model parameters, and training hyperparameters.
    """

    beta: float = Field(default=0.001, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.1, ge=0.0)
    update_epochs: int = Field(..., ge=1)
    reduce_memory_peak: bool = Field(default=False)
    lora_config: LoraConfig
    max_model_len: int
    use_separate_reference_adapter: bool
    pretrained_model_name_or_path: str
    calc_position_embeddings: bool
