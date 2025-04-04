from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
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
    LAYER = "layer"
    NODE = "node"
    ACTIVATION = "activation"


@runtime_checkable
class MutationMethod(Protocol):
    _mutation_type: MutationType
    _recreate_kwargs: Dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class OptimizerWrapper(Protocol):
    optimizer: Union[Optimizer, Iterable[Optimizer]]
    optimizer_cls: Union[Type[Optimizer], Iterable[Type[Optimizer]]]
    lr: Callable[[], float]
    optimizer_kwargs: Dict[str, Any]
    multiagent: bool


@runtime_checkable
class EvolvableModule(Protocol):
    init_dict: Dict[str, Any]
    device: DeviceType
    layer_mutation_methods: List[str]
    node_mutation_methods: List[str]
    last_mutation_attr: str
    last_mutation: Callable[[Any], Any]

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
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


@runtime_checkable
class EvolvableNetwork(EvolvableModule, Protocol):
    encoder: EvolvableModule
    head_net: EvolvableModule

    def forward_head(self, latent: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...
    def extract_features(self, x: TorchObsType) -> torch.Tensor: ...
    def build_network_head(self, *args, **kwargs) -> None: ...
    def _build_encoder(self, *args, **kwargs) -> None: ...


EvolvableNetworkType = Union[EvolvableModule, List[EvolvableModule]]
OptimizerType = Union[Optimizer, Iterable[Optimizer], OptimizerWrapper]
EvolvableAttributeType = Union[EvolvableNetworkType, OptimizerType]
EvolvableNetworkDict = Dict[str, EvolvableNetworkType]
EvolvableAttributeDict = Dict[str, EvolvableAttributeType]


@runtime_checkable
class NetworkConfig(Protocol):
    name: str
    eval: bool
    optimizer: Optional[str]


@runtime_checkable
class NetworkGroup(Protocol):
    eval: EvolvableModule
    shared: Optional[Union[EvolvableModule, List[EvolvableModule]]]
    policy: bool
    multiagent: bool


@runtime_checkable
class OptimizerConfig(Protocol):
    name: str
    networks: Union[str, List[str]]
    lr: str
    optimizer_cls: Union[Type[Optimizer], List[Type[Optimizer]]]
    optimizer_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]]
    multiagent: bool

    def get_optimizer_cls(self) -> Union[Type[Optimizer], List[Type[Optimizer]]]: ...


@runtime_checkable
class MutationRegistry(Protocol):
    groups: List[NetworkGroup]
    optimizers: List[OptimizerConfig]
    hooks: List[Callable]

    def networks(self) -> List[NetworkConfig]: ...


SelfEvolvableAlgorithm = TypeVar("SelfEvolvableAlgorithm", bound="EvolvableAlgorithm")


@runtime_checkable
class EvolvableAlgorithm(Protocol):
    device: Union[str, torch.device]
    accelerator: Accelerator
    registry: MutationRegistry
    mut: Optional[str]
    index: int
    scores: List[float]
    fitness: List[float]
    steps: List[int]

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


@runtime_checkable
class AgentWrapper(Protocol):
    agent: EvolvableAlgorithm

    def get_action(self, obs: ObservationType, **kwargs) -> Any: ...
    def learn(
        self, experiences: Tuple[Iterable[ObservationType], ...], **kwargs
    ) -> None: ...
