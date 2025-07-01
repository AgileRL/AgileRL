from enum import Enum
from numbers import Number
from typing import Any, ClassVar, Dict, List, Protocol, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from numpy.typing import ArrayLike
from pettingzoo import ParallelEnv
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from agilerl.protocols import EvolvableAlgorithm


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


ArrayOrTensor = Union[ArrayLike, torch.Tensor]
StandardTensorDict = Dict[str, torch.Tensor]
TensorTuple = Tuple[torch.Tensor, ...]
ArrayDict = Dict[str, np.ndarray]
ArrayTuple = Tuple[ArrayLike, ...]
NetConfigType = Dict[str, Any]
KernelSizeType = Union[int, Tuple[int, ...]]
SupportedGymSpaces = Union[
    spaces.Box,
    spaces.Discrete,
    spaces.MultiDiscrete,
    spaces.Dict,
    spaces.Tuple,
    spaces.MultiBinary,
]
GymSpaceType = Union[SupportedGymSpaces, List[SupportedGymSpaces]]
GymEnvType = Union[str, gym.Env, gym.vector.VectorEnv]
PzEnvType = Union[str, ParallelEnv]

NumpyObsType = Union[np.ndarray, ArrayDict, ArrayTuple]
TorchObsType = Union[torch.Tensor, TensorDict, TensorTuple, StandardTensorDict]
ObservationType = Union[NumpyObsType, TorchObsType, Number]
ActionType = Union[int, float, np.ndarray, torch.Tensor]
InfosDict = Dict[str, Dict[str, Any]]
MaybeObsList = Union[List[ObservationType], ObservationType]
ExperiencesType = Union[Dict[str, ObservationType], Tuple[ObservationType, ...]]
StepType = Tuple[NumpyObsType, ActionType, float, MaybeObsList, InfosDict]
MultiAgentStepType = Tuple[
    Dict[str, NumpyObsType], ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]
]

DeviceType = Union[str, torch.device]
OptimizerType = Union[Optimizer, AcceleratedOptimizer]
NetworkType = Union[Module, List[Module], Tuple[Module, ...]]
PopulationType = List[EvolvableAlgorithm]
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
