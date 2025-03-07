from typing import Any, ClassVar, Dict, List, Protocol, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from numpy.typing import ArrayLike
from torch.nn import Module
from torch.optim import Optimizer

from agilerl.protocols import EvolvableAlgorithm


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


ArrayOrTensor = Union[ArrayLike, torch.Tensor]
TensorDict = Dict[str, torch.Tensor]
TensorTuple = Tuple[torch.Tensor, ...]
ArrayDict = Dict[str, np.ndarray]
ArrayTuple = Tuple[ArrayLike, ...]
NetConfigType = Dict[str, Any]
KernelSizeType = Union[int, Tuple[int, ...]]
SupportedGymSpaces = Union[
    spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.Dict, spaces.Tuple
]
GymSpaceType = Union[SupportedGymSpaces, List[SupportedGymSpaces]]
GymEnvType = Union[str, gym.Env, gym.vector.VectorEnv]

NumpyObsType = Union[np.ndarray, ArrayDict, ArrayTuple]
TorchObsType = Union[torch.Tensor, TensorDict, TensorTuple]
ObservationType = Union[NumpyObsType, TorchObsType]
ActionType = Union[int, float, ArrayLike, torch.Tensor]
InfosDict = Dict[str, Dict[str, Any]]
MaybeObsList = Union[List[ObservationType], ObservationType]
ExperiencesType = Tuple[ObservationType, ...]
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
