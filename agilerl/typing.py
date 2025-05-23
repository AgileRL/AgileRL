from enum import Enum
from numbers import Number
from typing import Any, ClassVar, Dict, List, Protocol, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from numpy.typing import ArrayLike
from pettingzoo import ParallelEnv
from tensordict import TensorDict
from torch._dynamo import OptimizedModule
from torch.nn import Module
from torch.optim import Optimizer

from agilerl.protocols import EvolvableAlgorithm, EvolvableModule, EvolvableNetwork

# Type variable for module types - bound to Module to ensure all types inherit from it
T = TypeVar("T", bound=Union[Module, OptimizedModule])


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


class MultiAgentSetup(Enum):
    """Enum to specify the type of multi-agent setup."""

    HOMOGENEOUS = "homogeneous"  # all agents have the same network architecture
    MIXED = "mixed"  # contains a mix of different network architectures
    HETEROGENEOUS = "heterogeneous"  # all agents have different network architectures


class ModuleType(Enum):
    """Enum to specify the type of module."""

    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    MULTI_INPUT = "multi_input"


SupportedObsSpaces = Union[
    spaces.Box,
    spaces.Discrete,
    spaces.MultiDiscrete,
    spaces.Dict,
    spaces.Tuple,
    spaces.MultiBinary,
]

SupportedActionSpaces = Union[
    spaces.Discrete,
    spaces.MultiDiscrete,
    spaces.MultiBinary,
    spaces.Box,
]

ArrayOrTensor = Union[ArrayLike, torch.Tensor]
StandardTensorDict = Dict[str, torch.Tensor]
TensorTuple = Tuple[torch.Tensor, ...]
ArrayDict = Dict[str, np.ndarray]
ArrayTuple = Tuple[ArrayLike, ...]
NetConfigType = Dict[str, Union[Dict[str, Any], Any]]
KernelSizeType = Union[int, Tuple[int, ...]]
GymSpaceType = Union[SupportedObsSpaces, List[SupportedObsSpaces]]
GymEnvType = Union[str, gym.Env, gym.vector.VectorEnv, gym.vector.AsyncVectorEnv]
PzEnvType = Union[str, ParallelEnv]

NumpyObsType = Union[np.ndarray, ArrayDict, ArrayTuple]
TorchObsType = Union[torch.Tensor, TensorDict, TensorTuple, StandardTensorDict]
ObservationType = Union[NumpyObsType, TorchObsType, Number]
MultiAgentObservationType = Dict[str, ObservationType]
ActionType = Union[int, float, np.ndarray, torch.Tensor]
InfosDict = Dict[str, Dict[str, Any]]
MaybeObsList = Union[List[ObservationType], ObservationType]
ExperiencesType = Union[Dict[str, ObservationType], Tuple[ObservationType, ...]]
ActionReturnType = Union[Tuple[Union[ActionType, Any], ...], ActionType, Any]
GymStepReturn = Tuple[NumpyObsType, ActionType, float, MaybeObsList, InfosDict]
PzStepReturn = Tuple[
    Dict[str, NumpyObsType], ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]
]

SingleAgentModule = Union[T, EvolvableModule, OptimizedModule, EvolvableNetwork]
MultiAgentModule = Dict[str, SingleAgentModule[T]]
NetworkType = Union[SingleAgentModule[T], MultiAgentModule[T]]
PopulationType = List[EvolvableAlgorithm]
DeviceType = Union[str, torch.device]
OptimizerType = Union[Optimizer, AcceleratedOptimizer]
ConfigType = Union[IsDataclass, NetConfigType]
StateDict = Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
