from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from numbers import Number
from typing import (
    Any,
    ClassVar,
    NamedTuple,
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
from tensordict import TensorClass, TensorDict
from torch._dynamo import OptimizedModule
from torch.nn import Module
from torch.optim import Optimizer

from agilerl.protocols import (
    EvolvableAlgorithmProtocol,
    EvolvableModuleProtocol,
    EvolvableNetworkProtocol,
    ModuleDictProtocol,
)

# Type variable for module types - bound to Module to ensure all types inherit from it
T = TypeVar("T", bound=Module | OptimizedModule)


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


class ReasoningPrompts(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    question: str | list[str] | None
    answer: str | list[str] | None
    trajectory_input_ids: torch.Tensor | None
    trajectory_attention_mask: torch.Tensor | None
    initial_prompt_len: int | list[int] | torch.Tensor | None
    stitch_prefix_ids: torch.Tensor | None


class PreferencePrompts(TypedDict):
    prompt: list[str]
    prompt_lengths: list[int]
    chosen: list[str]
    rejected: list[str]
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor


class SFTPrompts(TypedDict):
    prompt: list[str]
    prompt_lengths: list[int]
    response: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class CheckpointInfo(TypedDict):
    modules: dict[str, Module]
    optimizers: dict[str, Optimizer]
    network_names: list[str]
    optimizer_names: list[str]


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


SupportedObservationSpace = (
    spaces.Box
    | spaces.Discrete
    | spaces.MultiDiscrete
    | spaces.Dict
    | spaces.Tuple
    | spaces.MultiBinary
)
SupportedActionSpace = (
    spaces.Discrete | spaces.MultiDiscrete | spaces.MultiBinary | spaces.Box
)
# Per-agent spaces for multi-agent algorithms: an ordered iterable of spaces,
# a mapping keyed by agent id, or a ``spaces.Dict``.
MultiAgentSpacesType = Iterable[spaces.Space] | Mapping[str, spaces.Space] | spaces.Dict

ArrayOrTensor = np.ndarray | torch.Tensor
StandardTensorDict = dict[str, torch.Tensor]
TensorTuple = tuple[torch.Tensor, ...]
ArrayDict = dict[str, np.ndarray]
ArrayTuple = tuple[np.ndarray, ...]
NetConfigType = dict[str, Any]
KernelSizeType = int | tuple[int, ...]
GymSpaceType = SupportedObservationSpace | list[SupportedObservationSpace]
GymEnvType = str | gym.Env | gym.vector.VectorEnv | gym.vector.AsyncVectorEnv
PzEnvType = str | ParallelEnv
LLMObsType = list[ReasoningPrompts] | ReasoningPrompts

NumpyObsType = np.ndarray | ArrayDict | ArrayTuple
TorchObsType = torch.Tensor | TensorDict | TensorTuple | StandardTensorDict
ObservationType = NumpyObsType | TorchObsType | Number | LLMObsType
MultiAgentObservationType = dict[str, ObservationType]
ActionType = int | float | np.ndarray | torch.Tensor
# A recorded fitness: a scalar, or a per-sub-agent row (multi-agent, sum_scores=False).
FitnessValue = float | np.ndarray
InfosDict = dict[str, dict[str, Any]]
MaybeObsList = list[ObservationType] | ObservationType
ExperiencesType = dict[str, ObservationType] | tuple[ObservationType, ...]

# The batch type an algorithm's ``learn`` consumes. Each concrete algorithm binds
# this to the exact shape its buffer/rollout produces (e.g. ``ReplayBatch`` for
# off-policy value methods, ``PreferencePrompts`` for DPO), so ``learn`` reads its
# batch with precise key/element typing instead of narrowing a broad union by hand.
ExperiencesT = TypeVar("ExperiencesT")


class ReplayBatch(TensorClass):
    """One off-policy sample from a :class:`~agilerl.components.replay_buffer.ReplayBuffer`.

    A :class:`~tensordict.TensorClass`: attribute access (``batch.reward``) resolves
    statically to its declared type (:class:`torch.Tensor`), while the object wraps a
    real ``TensorDict`` at runtime. Build one with ``ReplayBatch.from_tensordict`` and
    bind the result to a ``: ReplayBatch`` annotation so the field types flow through.
    """

    obs: TorchObsType
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: TorchObsType
    done: torch.Tensor


class PrioritizedReplayBatch(ReplayBatch):
    """A :class:`ReplayBatch` plus the priority weights and indices returned by
    prioritized (or ``return_idx=True``) sampling.
    """

    weights: torch.Tensor
    idxs: torch.Tensor


class BanditBatch(TensorClass):
    """One sample from a bandit replay buffer (context and reward only)."""

    obs: TorchObsType
    reward: torch.Tensor


class RolloutMinibatch(TensorClass):
    """One flattened (non-BPTT) PPO minibatch drawn from the rollout buffer.

    ``action_masks`` is ``None`` when the policy does not use action masking. The
    buffer stores value predictions under the key ``"values"``, which collides with
    ``TensorDict.values()``; PPO renames it to ``value_preds`` when it wraps the
    buffer so it reads back as a plain attribute here.
    """

    observations: TorchObsType
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    value_preds: torch.Tensor
    action_masks: torch.Tensor | None = None


class RolloutSequenceMinibatch(TensorClass):
    """The padded per-sequence half of a truncated-BPTT PPO minibatch.

    Sequences are padded to a common length; ``pad_mask`` marks the real steps.
    Initial recurrent hidden states ride along as a non-tensor entry and are read
    with ``get_non_tensor`` at the call site.
    """

    observations: TorchObsType
    actions: torch.Tensor
    pad_mask: torch.Tensor
    action_masks: torch.Tensor | None = None


class RolloutSequenceTargets(TensorClass):
    """The unpadded training-target half of a truncated-BPTT PPO minibatch.

    As with :class:`RolloutMinibatch`, the buffer's ``"values"`` key is renamed to
    ``value_preds`` at wrap time (it collides with ``TensorDict.values()``).
    """

    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    value_preds: torch.Tensor


class MultiAgentReplayBatch(TensorClass):
    """One multi-agent off-policy sample from the shared replay buffer.

    Each field is a nested per-agent :class:`~tensordict.TensorDict` (agent id ->
    tensor). Build one from a sampled batch with
    ``MultiAgentReplayBatch.from_tensordict`` and bind the result to a
    ``: MultiAgentReplayBatch`` annotation so the fields resolve.
    """

    obs: TensorDict
    action: TensorDict
    reward: TensorDict
    next_obs: TensorDict
    done: TensorDict


# One LLM RL rollout consumed by GRPO/LLMPPO/LLMREINFORCE's ``learn``:
# ``(completion_ids, action_masks, rewards)``. ``completion_ids`` and
# ``action_masks`` are per-trajectory tensor lists, or already-stacked tensors
# after cross-rank sequence-padding alignment; ``rewards`` is a ``(batch,)`` (or
# ``(batch, max_turns)`` for per-turn) tensor.
LLMRolloutExperiences = tuple[
    list[torch.Tensor] | torch.Tensor,
    list[torch.Tensor] | torch.Tensor,
    torch.Tensor,
]


ActionReturnType = tuple[ActionType | Any, ...] | ActionType | Any
GymStepReturn = tuple[NumpyObsType, ActionType, float, MaybeObsList, InfosDict]
PzStepReturn = tuple[
    dict[str, NumpyObsType],
    ArrayDict,
    ArrayDict,
    ArrayDict,
    dict[str, Any],
]

SingleAgentModule = (
    T | EvolvableModuleProtocol | OptimizedModule | EvolvableNetworkProtocol
)
MultiAgentModule = ModuleDictProtocol[SingleAgentModule[T]]
NetworkType = SingleAgentModule[T] | MultiAgentModule[T]
EvolvableNetworkType = (
    EvolvableModuleProtocol | ModuleDictProtocol[EvolvableModuleProtocol]
)
DeviceType = str | torch.device
OptimizerType = Optimizer | AcceleratedOptimizer

SingleAgentMutReturnType = dict[str, Any]
MultiAgentMutReturnType = dict[str, dict[str, Any]]
MutationReturnType = SingleAgentMutReturnType | MultiAgentMutReturnType
PopulationType = list[EvolvableAlgorithmProtocol]
MutationMethod = Callable[[EvolvableAlgorithmProtocol], EvolvableAlgorithmProtocol]
ConfigType = IsDataclass | NetConfigType
StateDict = dict[str, Any] | dict[str, dict[str, Any]] | list[dict[str, Any]]

# A decoded JSON value: any JSON primitive, array, or object (recursive).
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

# A nested container of tensors/arrays for the LLM pytree map helpers (recursive).
# Leaves are tensors or arrays; interior nodes are dicts, lists, or tuples.
PyTree = (
    torch.Tensor
    | np.ndarray
    | dict[str, "PyTree"]
    | list["PyTree"]
    | tuple["PyTree", ...]
)
LrNameType = str | tuple[str, str]


class BatchDimension:
    def __repr__(self) -> str:
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


class ActionResult(NamedTuple):
    """Structured return of an LLM algorithm's :meth:`get_action`.

    A tuple subclass, so callers may unpack positionally *or* (preferred, and
    forward-compatible if fields are added) read by attribute. ``sampling_logps``
    holds the per-completion vLLM sampling logprobs captured for the
    sampling-mismatch correction, or ``None`` when not captured (HF generation,
    evaluation, or correction disabled).
    """

    completion_ids: list[torch.Tensor]
    action_masks: list[torch.Tensor]
    sampling_logps: list[torch.Tensor | None] | None = None
