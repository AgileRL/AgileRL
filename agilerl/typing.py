from enum import Enum
from numbers import Number
from typing import Any, Callable, ClassVar, Dict, List, Protocol, Tuple, TypeVar, Union

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

from agilerl.protocols import (
    EvolvableAlgorithm,
    EvolvableModule,
    EvolvableNetwork,
    ModuleDict,
)

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
MultiAgentModule = ModuleDict[SingleAgentModule[T]]
NetworkType = Union[SingleAgentModule[T], MultiAgentModule[T]]
EvolvableNetworkType = Union[EvolvableModule, ModuleDict[EvolvableModule]]
DeviceType = Union[str, torch.device]
OptimizerType = Union[Optimizer, AcceleratedOptimizer]
ConfigType = Union[IsDataclass, NetConfigType]
StateDict = Union[Dict[str, Any], Dict[str, Dict[str, Any]]]

SingleAgentMutReturnType = Dict[str, Any]
MultiAgentMutReturnType = Dict[str, Dict[str, Any]]
MutationReturnType = Union[SingleAgentMutReturnType, MultiAgentMutReturnType]
PopulationType = List[EvolvableAlgorithm]
MutationMethod = Callable[[EvolvableAlgorithm], EvolvableAlgorithm]
ConfigType = Union[IsDataclass, Dict[str, Any]]
StateDict = Union[Dict[str, Any], List[Dict[str, Any]]]


class BatchDimension:
    def __repr__(self):
        return "BatchDimension"


class BPTTSequenceType(Enum):
    """Enum for BPTT sequence generation methods. It specifies the strategy used when generating sequences for BPTT training.

    CHUNKED is the default method which uses the least amount of memory while keeping all sampled trajectories available in the buffer for sequencing.
        The number of sequences generated is then:  (num_steps / max_seq_len) * num_envs
    MAXIMUM generates all possible overlapping sequences, which is the most memory-intensive option.
        The number of sequences generated is then:  (num_steps - max_seq_len + 1) * num_envs
    FIFTY_PERCENT_OVERLAP generates sequences with 50% overlap, which is a compromise between the two.
        The number of sequences generated is then:  (num_steps / max_seq_len * 2) * num_envs
    """

    CHUNKED = "chunked"  # Generate sequences by non-overlapping chunks
    MAXIMUM = "maximum"  # Generate all possible overlapping sequences
    FIFTY_PERCENT_OVERLAP = (
        "fifty_percent_overlap"  # Generate sequences with 50% overlap
    )
