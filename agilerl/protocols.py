from typing import Any, Dict, List, Optional, Tuple, Protocol, Union, Iterable, runtime_checkable
from enum import Enum
from numpy.typing import ArrayLike
from accelerate import Accelerator
import torch
from torch.optim.optimizer import Optimizer

NumpyObsType = Union[ArrayLike, Dict[str, ArrayLike], Tuple[ArrayLike, ...]]

class MutationType(Enum):
    LAYER = "layer"
    NODE = "node"

@runtime_checkable
class MutationMethod(Protocol):
    _mutation_type: MutationType
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

@runtime_checkable
class EvolvableModule(Protocol):
    init_dict: Dict[str, Any]

    def state_dict(self) -> Dict[str, Any]:
        ...
    def get_mutation_methods(self) -> Dict[str, MutationMethod]:
        ...
    def get_mutation_probs(self, new_layer_prob: float) -> List[float]:
        ...
    def clone(self) -> "EvolvableModule":
        ...

EvolvableNetworkType = Union[EvolvableModule, Iterable[EvolvableModule]]
OptimizerType = Union[Optimizer, Iterable[Optimizer]]
EvolvableAttributeType = Union[EvolvableNetworkType, OptimizerType]
EvolvableNetworkDict = Dict[str, EvolvableNetworkType]
EvolvableAttributeDict = Dict[str, EvolvableAttributeType]

@runtime_checkable
class EvolvableAlgorithm(Protocol):
    device: Union[str, torch.device]
    accelerator: Accelerator
    learn_step: int
    algo: str
    mut: Optional[str]
    index: int
    scores: List[float]
    fitness: List[float]
    steps: List[int]

    def unwrap_models(self) -> None:
        ...
    def load_checkpoint(self, path: str, device: str, accelerator: Optional[Accelerator]) -> None:
        ...
    def save_checkpoint(self, path: str) -> None:
        ...
    def learn(self, experiences: Tuple[Iterable[ArrayLike], ...], **kwargs) -> None:
        ...
    def get_action(self, state: NumpyObsType, **kwargs) -> Any:
        ...
    def test(self, *args, **kwargs) -> ArrayLike:
        ...
    def load(cls, path: str) -> "EvolvableAlgorithm":
        ...
    def evolvable_attributes(self, networks_only: bool = False) -> EvolvableAttributeDict:
        ...
    def inspect_attributes(self, input_args_only: bool = False) -> Dict[str, Any]:
        ...