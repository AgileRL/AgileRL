"""Protocol definitions for AgileRL's evolvable algorithms and neural networks.

This module contains Protocol classes and type definitions that define the interfaces
for evolvable components in the AgileRL framework. These protocols ensure type safety
and provide clear contracts for implementing evolvable algorithms, neural networks,
and optimization components.

The key protocols include:
- EvolvableAlgorithm: Interface for algorithms that can evolve through mutations
- EvolvableModule: Interface for neural network modules that support mutations
- EvolvableNetwork: Interface for neural networks with encoder-decoder structure
- MutationMethod: Interface for mutation operations on networks
- OptimizerWrapper: Interface for optimizer management

Type aliases are provided for common types used throughout the framework.
"""

from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import torch
from accelerate import Accelerator
from numpy.typing import ArrayLike
from torch.optim.optimizer import Optimizer

NumpyObsType = Union[ArrayLike, Dict[str, ArrayLike], Tuple[ArrayLike, ...]]
TorchObsType = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]
ObservationType = Union[NumpyObsType, TorchObsType]
DeviceType = Union[str, torch.device]


class MutationType(Enum):
    """Enumeration of mutation types for evolvable neural networks.

    :param LAYER: Mutation that affects network layers (add/remove layers)
    :param NODE: Mutation that affects nodes within layers (add/remove nodes)
    :param ACTIVATION: Mutation that changes activation functions
    """

    LAYER = "layer"
    NODE = "node"
    ACTIVATION = "activation"


@runtime_checkable
class MutationMethod(Protocol):
    """Protocol for mutation methods that can be applied to evolvable modules.

    Mutation methods must have a mutation type and optional recreation kwargs
    to specify how the network should be rebuilt after mutation.
    """

    _mutation_type: MutationType
    _recreate_kwargs: Dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class OptimizerWrapper(Protocol):
    """Protocol for optimizer wrapper classes that manage optimization.

    Provides a consistent interface for optimizer management across different
    network configurations and training setups.
    """

    optimizer: Union[Optimizer, Dict[str, Optimizer]]
    optimizer_cls: Union[Type[Optimizer], Dict[str, Type[Optimizer]]]
    lr: Callable[[], float]
    optimizer_kwargs: Dict[str, Any]


@runtime_checkable
class EvolvableModule(Protocol):
    """Protocol for neural network modules that support evolutionary mutations.

    Evolvable modules can undergo mutations to their architecture (layers, nodes,
    activations) and maintain state information about recent mutations for
    reconstruction and optimization purposes.
    """

    init_dict: Dict[str, Any]
    device: DeviceType
    layer_mutation_methods: List[str]
    node_mutation_methods: List[str]
    mutation_methods: List[str]
    last_mutation_attr: str
    last_mutation: Callable[[Any], Any]
    rng: Optional[Generator]

    @property
    def activation(self) -> Optional[str]: ...
    def change_activation(self, activation: str, output: bool) -> None: ...
    def forward(self, x: Any) -> Any: ...
    def parameters(self) -> Generator: ...
    def to(self, device: DeviceType) -> None: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def disable_mutations(self) -> None: ...
    def get_mutation_methods(self) -> Dict[str, MutationMethod]: ...
    def get_mutation_probs(self, new_layer_prob: float) -> List[float]: ...
    def sample_mutation_method(
        self, new_layer_prob: float, rng: Optional[Generator]
    ) -> str: ...
    def clone(self) -> "EvolvableModule": ...
    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True
    ) -> None: ...


@runtime_checkable
class EvolvableNetwork(Protocol):
    """Protocol for neural networks with encoder-decoder architecture.

    Evolvable networks consist of an encoder for feature extraction and
    a head network for task-specific outputs. Both components can evolve
    independently through mutations.
    """

    encoder: EvolvableModule
    head_net: EvolvableModule

    def forward_head(self, latent: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...
    def extract_features(self, x: TorchObsType) -> torch.Tensor: ...
    def build_network_head(self, *args, **kwargs) -> None: ...
    def _build_encoder(self, *args, **kwargs) -> None: ...


T = TypeVar("T", bound=Union[EvolvableModule, EvolvableNetwork])


@runtime_checkable
class ModuleDict(Protocol, Generic[T]):
    """Protocol for dictionary-like containers of evolvable modules.

    Provides access to multiple evolvable modules through a dictionary interface
    and aggregates mutation methods across all contained modules.
    """

    def __getitem__(self, key: str) -> T: ...
    def keys(self) -> Iterable[str]: ...
    def values(self) -> Iterable[T]: ...
    def items(self) -> Iterable[Tuple[str, T]]: ...
    def modules(self) -> Dict[str, T]: ...
    def get_mutation_methods(self) -> Dict[str, MutationMethod]: ...
    def filter_mutation_methods(self, method: str) -> None: ...

    @property
    def mutation_methods(self) -> List[str]: ...
    @property
    def layer_mutation_methods(self) -> List[str]: ...
    @property
    def node_mutation_methods(self) -> List[str]: ...


EvolvableNetworkType = Union[EvolvableModule, ModuleDict]
OptimizerType = Union[Optimizer, Dict[str, Optimizer], OptimizerWrapper]
EvolvableAttributeType = Union[EvolvableNetworkType, OptimizerType]
EvolvableNetworkDict = Dict[str, EvolvableNetworkType]
EvolvableAttributeDict = Dict[str, EvolvableAttributeType]


@runtime_checkable
class NetworkConfig(Protocol):
    """Protocol for network configuration information.

    Stores metadata about networks including their name, evaluation status,
    and associated optimizer.
    """

    name: str
    eval: bool
    optimizer: Optional[str]


@runtime_checkable
class NetworkGroup(Protocol):
    """Protocol for grouping related networks in an algorithm.

    Groups evaluation and shared networks together, indicating whether
    they represent policy networks and if they're used in multi-agent setups.
    """

    eval: EvolvableNetworkType
    shared: Optional[Union[EvolvableNetworkType, List[EvolvableNetworkType]]]
    policy: bool
    multiagent: bool


@runtime_checkable
class OptimizerConfig(Protocol):
    """Protocol for optimizer configuration and management.

    Defines the configuration for optimizers including which networks they
    optimize, learning rate, optimizer class, and additional parameters.
    """

    name: str
    networks: Union[str, List[str]]
    lr: str
    optimizer_cls: Union[Type[Optimizer], List[Type[Optimizer]]]
    optimizer_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]]
    multiagent: bool

    def get_optimizer_cls(self) -> Union[Type[Optimizer], List[Type[Optimizer]]]: ...


@runtime_checkable
class MutationRegistry(Protocol):
    """Protocol for registering and managing mutation-related components.

    Maintains collections of network groups, optimizers, and hooks that
    are used during the mutation and evolution process.
    """

    groups: List[NetworkGroup]
    optimizers: List[OptimizerConfig]
    hooks: List[Callable]

    def networks(self) -> List[NetworkConfig]: ...


SelfEvolvableAlgorithm = TypeVar("SelfEvolvableAlgorithm", bound="EvolvableAlgorithm")


@runtime_checkable
class EvolvableAlgorithm(Protocol):
    """Protocol for reinforcement learning algorithms that support evolution.

    Evolvable algorithms can undergo mutations to their network architectures
    and hyperparameters. They maintain state about fitness, scores, and steps
    for selection and mutation processes.
    """

    device: Union[str, torch.device]
    accelerator: Accelerator
    registry: MutationRegistry
    mut: Optional[str]
    index: int
    scores: List[float]
    fitness: List[float]
    steps: List[int]
    torch_compiler: Optional[str]

    def unwrap_models(self) -> None: ...
    def wrap_models(self) -> None: ...
    def load(
        cls: Type[SelfEvolvableAlgorithm], path: str
    ) -> SelfEvolvableAlgorithm: ...
    def load_checkpoint(
        self, path: str, device: str, accelerator: Optional[Accelerator]
    ) -> None: ...
    def save_checkpoint(self, path: str) -> None: ...
    def learn(
        self, experiences: Tuple[Iterable[ObservationType], ...], **kwargs
    ) -> None: ...
    def get_action(self, obs: ObservationType, **kwargs) -> Any: ...
    def test(self, *args, **kwargs) -> ArrayLike: ...
    def evolvable_attributes(
        self, networks_only: bool = False
    ) -> EvolvableAttributeDict: ...
    def inspect_attributes(
        agent: SelfEvolvableAlgorithm, input_args_only: bool = False
    ) -> Dict[str, Any]: ...
    def clone(
        self: SelfEvolvableAlgorithm, index: Optional[int], wrap: bool
    ) -> SelfEvolvableAlgorithm: ...
    def recompile(self) -> None: ...
    def mutation_hook(self) -> None: ...


# Define a TypeVar for EvolvableAlgorithm that can be used for generic typing
T_EvolvableAlgorithm = TypeVar("T_EvolvableAlgorithm", bound=EvolvableAlgorithm)


@runtime_checkable
class AgentWrapper(Protocol, Generic[T_EvolvableAlgorithm]):
    """Protocol for wrapper classes that encapsulate evolvable algorithms.

    Agent wrappers provide additional functionality around evolvable algorithms
    while maintaining the core interface for action selection and learning.
    """

    agent: T_EvolvableAlgorithm

    def get_action(self, obs: ObservationType, **kwargs) -> Any: ...
    def learn(
        self, experiences: Tuple[Iterable[ObservationType], ...], **kwargs
    ) -> None: ...
