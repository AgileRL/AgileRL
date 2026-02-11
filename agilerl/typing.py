from enum import Enum
from numbers import Number
from typing import (
    Any,
    Callable,
    ClassVar,
    Protocol,
    TypedDict,
    TypeVar,
)

import gymnasium as gym
import numpy as np
import torch
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
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
T = TypeVar("T", bound=Module | OptimizedModule)


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


# TODO ideally adjust the ReasoningGym to match the PreferencePrompts type (e.g the lists/btaching happens for the values of the
# keys, not just returning a list of 'ReasoningPrompts')
class ReasoningPrompts(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    text: str | None


class PreferencePrompts(TypedDict):
    prompt: list[str]
    prompt_lengths: list[int]
    chosen: list[str]
    rejected: list[str]
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor


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


SupportedObsSpaces = (
    spaces.Box
    | spaces.Discrete
    | spaces.MultiDiscrete
    | spaces.Dict
    | spaces.Tuple
    | spaces.MultiBinary
)
SupportedActionSpaces = (
    spaces.Discrete | spaces.MultiDiscrete | spaces.MultiBinary | spaces.Box
)

ArrayOrTensor = np.ndarray | torch.Tensor
StandardTensorDict = dict[str, torch.Tensor]
TensorTuple = tuple[torch.Tensor, ...]
ArrayDict = dict[str, np.ndarray]
ArrayTuple = tuple[np.ndarray, ...]
NetConfigType = dict[str, dict[str, Any] | Any]
KernelSizeType = int | tuple[int, ...]
GymSpaceType = SupportedObsSpaces | list[SupportedObsSpaces]
GymEnvType = str | gym.Env | gym.vector.VectorEnv | gym.vector.AsyncVectorEnv
PzEnvType = str | ParallelEnv
LLMObsType = list[ReasoningPrompts]

NumpyObsType = np.ndarray | ArrayDict | ArrayTuple
TorchObsType = torch.Tensor | TensorDict | TensorTuple | StandardTensorDict
ObservationType = NumpyObsType | TorchObsType | Number | LLMObsType
MultiAgentObservationType = dict[str, ObservationType]
ActionType = int | float | np.ndarray | torch.Tensor
InfosDict = dict[str, dict[str, Any]]
MaybeObsList = list[ObservationType] | ObservationType
ExperiencesType = dict[str, ObservationType] | tuple[ObservationType, ...]
ActionReturnType = tuple[ActionType | Any, ...] | ActionType | Any
GymStepReturn = tuple[NumpyObsType, ActionType, float, MaybeObsList, InfosDict]
PzStepReturn = tuple[
    dict[str, NumpyObsType], ArrayDict, ArrayDict, ArrayDict, dict[str, Any]
]

SingleAgentModule = T | EvolvableModule | OptimizedModule | EvolvableNetwork
MultiAgentModule = ModuleDict[SingleAgentModule[T]]
NetworkType = SingleAgentModule[T] | MultiAgentModule[T]
EvolvableNetworkType = EvolvableModule | ModuleDict[EvolvableModule]
DeviceType = str | torch.device
OptimizerType = Optimizer | AcceleratedOptimizer
ConfigType = IsDataclass | NetConfigType
StateDict = dict[str, Any] | dict[str, dict[str, Any]]

SingleAgentMutReturnType = dict[str, Any]
MultiAgentMutReturnType = dict[str, dict[str, Any]]
MutationReturnType = SingleAgentMutReturnType | MultiAgentMutReturnType
PopulationType = list[EvolvableAlgorithm]
MutationMethod = Callable[[EvolvableAlgorithm], EvolvableAlgorithm]
ConfigType = IsDataclass | dict[str, Any]
StateDict = dict[str, Any] | list[dict[str, Any]]


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
