from typing import Union, Dict, List, Tuple, Any
import torch
from numpy.typing import ArrayLike
from torch.optim import Optimizer
from torch.nn import Module
from accelerate.optimizer import AcceleratedOptimizer

ArrayOrTensor = Union[ArrayLike, torch.Tensor]
TensorDict = Dict[str, torch.Tensor]
ArrayDict = Dict[str, ArrayOrTensor]
NetConfigType = Dict[str, Any]

NumpyObsType = Union[ArrayLike, ArrayDict]
TorchObsType = Union[torch.Tensor, TensorDict]

OptimizerType = Union[Optimizer, AcceleratedOptimizer]
NetworkType = Union[Module, List[Module], Tuple[Module, ...]]