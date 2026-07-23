"""Shared type aliases, TypedDicts, and batch dataclasses used across AgileRL.

Naming conventions (keep new aliases consistent with these):

* Suffix by kind: a ``*Type`` suffix marks a **type alias** (a name for a union
  or concrete type set, e.g. ``DeviceType``, ``ObservationType``, ``BufferType``);
  a bare ``*T`` suffix is reserved for ``TypeVar`` generic parameters (e.g.
  ``T``, ``ExperiencesT``, ``AgentT``) and is never used for a plain alias.
* Plain structural aliases may drop the suffix (``ArrayDict``, ``TensorTuple``,
  ``TensorMapping``) where the shape already reads as a type.
* Multi-agent aliases use the ``MultiAgent*`` prefix (not ``MARL*``).
* Observation aliases use the ``*ObsType`` suffix; the two hubs
  ``ObservationType`` / ``MultiAgentObservationType`` keep the fuller word.
* Function / tuple return aliases use the ``*Return`` suffix (not ``*ReturnType``).
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
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
from tensordict import TensorClass, TensorDict
from torch._dynamo import OptimizedModule
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import Never, NotRequired

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


# ── TypedDicts: LLM prompts, checkpoint & mutation payloads ──────────────────
class ReasoningPrompts(TypedDict):
    """Tokenized reasoning / multi-turn observation prompts.

    ``input_ids`` and ``attention_mask`` are always present. Remaining keys are
    filled by collate, multi-turn wrappers, or sliding-window truncation as
    needed.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    question: NotRequired[str | list[str] | None]
    answer: NotRequired[str | list[str] | None]
    trajectory_input_ids: NotRequired[torch.Tensor | None]
    trajectory_attention_mask: NotRequired[torch.Tensor | None]
    initial_prompt_len: NotRequired[int | list[int] | torch.Tensor | None]
    stitch_prefix_ids: NotRequired[torch.Tensor | None]
    text: NotRequired[str | None]
    trajectory_text: NotRequired[str | None]


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
    # Serialized bags keyed by ``{attr}_{cls|init_dict|state_dict|...}`` (modules)
    # and ``{attr}_{cls|state_dict|networks|lr|kwargs|...}`` (optimizers).
    # Values are heterogeneous pickled classes, init dicts, and state dicts.
    modules: dict[str, Any]
    optimizers: dict[str, Any]
    network_names: list[str]
    optimizer_names: list[str]


class MutationApplyDict(TypedDict, total=False):
    """Closed key set returned by single-module architecture mutations."""

    numb_new_nodes: int
    hidden_layer: int
    numb_new_channels: int
    kernel_size: int | tuple[int, ...] | list[int]


# ── Enums ────────────────────────────────────────────────────────────────────
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


# ── Gymnasium / PettingZoo space & env aliases ───────────────────────────────
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

# ── Array / tensor container aliases ─────────────────────────────────────────
ArrayOrTensor = np.ndarray | torch.Tensor
# Plain dict of tensors (the tensor twin of ``ArrayDict``); distinct from the
# tensordict-library ``TensorDict`` class also used in ``TorchObsType``.
TensorMapping = dict[str, torch.Tensor]
TensorTuple = tuple[torch.Tensor, ...]
ArrayDict = dict[str, np.ndarray]
ArrayTuple = tuple[np.ndarray, ...]
# Imported mid-file so ``modules.configs`` can load without a typing↔configs cycle
# (``modules.__init__`` is lazy). Canonical definition lives in ``modules.configs``.
from agilerl.modules.configs import NetConfigType as NetConfigType  # noqa: E402

KernelSizeType = int | tuple[int, ...]
GymSpaceType = SupportedObservationSpace | list[SupportedObservationSpace]
LLMObsType = list[ReasoningPrompts] | ReasoningPrompts

# ── Observation & action aliases ─────────────────────────────────────────────
NumpyObsType = np.ndarray | ArrayDict | ArrayTuple
TorchObsType = torch.Tensor | TensorDict | TensorTuple | TensorMapping
ObservationType = NumpyObsType | TorchObsType | Number | LLMObsType
MultiAgentObservationType = dict[str, ObservationType]
# Per-agent tensor observations keyed by agent id (used by the multi-agent wrappers).
MultiAgentTensorObsType = dict[str, TorchObsType]
ActionType = int | float | np.ndarray | torch.Tensor
# A recorded fitness: a scalar, or a per-sub-agent row (multi-agent, sum_scores=False).
FitnessValue = float | np.ndarray

# Raw Gym/PettingZoo ``info["action_mask"]`` before stacking into a tensor
# (1 = legal, 0 = illegal). ``None`` means the agent provided no mask.
ActionMask = np.ndarray | Sequence[int | float | bool]
MaybeActionMask = ActionMask | None


def coerce_action_mask(value: object) -> MaybeActionMask:
    """Narrow an env ``info["action_mask"]`` value to :data:`MaybeActionMask`."""
    if value is None or isinstance(value, np.ndarray):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        coerced: list[int | float | bool] = []
        for item in value:
            if not isinstance(item, (int, float, bool)):
                return None
            coerced.append(item)
        return coerced
    return None


def numpy_action_mask(
    value: np.ndarray | Sequence[np.ndarray] | torch.Tensor,
) -> np.ndarray:
    """Stack or convert a non-None :data:`ActionMaskInput` to ``np.ndarray``."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return np.stack(list(value))
        return value
    return np.stack(list(value))


# Per-agent masks from ``process_infos`` / ``extract_action_masks`` (raw or stacked).
MultiAgentActionMasks = Mapping[str, MaybeActionMask | torch.Tensor]

# Masks accepted by single-agent ``get_action`` / distribution heads: stacked
# ndarray, vectorized per-env masks, or already tensorised. Raw
# ``Sequence[int | float | bool]`` from env info belongs in ``MaybeActionMask``.
ActionMaskInput = np.ndarray | Sequence[np.ndarray] | torch.Tensor | None

# MADDPG / MATD3 / base-like ``process_infos`` return.
ProcessInfosReturn = tuple[
    MultiAgentActionMasks,
    ArrayDict | None,
    ArrayDict | None,
]

# IPPO stacks per-group masks into tensors (``None`` when absent).
IPPOActionMasks = Mapping[str, torch.Tensor | None]
IPPOProcessInfosReturn = tuple[
    IPPOActionMasks,
    ArrayDict | None,
    ArrayDict | None,
]


class AgentInfo(TypedDict, total=False):
    """Per-agent (or single-env) step ``info`` keys used by AgileRL."""

    action_mask: MaybeActionMask
    env_defined_actions: ActionType | None


# Flat single-agent Gymnasium info when only AgileRL keys are present.
GymInfo = AgentInfo
# Multi-agent env infos keyed by agent id. PettingZoo types the inner mapping
# as a bare ``dict``; ``AgentInfo`` documents keys we read.
InfosDict = Mapping[str, Mapping[str, object]]
MaybeObsList = list[ObservationType] | ObservationType
ExperiencesType = dict[str, ObservationType] | tuple[ObservationType, ...]

# Observation as a dict or tuple of arrays/tensors (Dict / Tuple spaces).
TupleOrDictObsType = dict[str, ArrayOrTensor] | tuple[ArrayOrTensor, ...]

# One transition bag for replay-buffer storage (plain dict or TensorDict).
DataType = dict[str, ArrayOrTensor] | TensorDict

# Layer metadata detected from an arbitrary user network (heterogeneous values).
LayerInfo = dict[str, Any]

# Gymnasium / PettingZoo wrapper factory: ``(cls, kwargs)``, import path, or callable.
WrapperSpec = tuple[Any, dict[str, Any]] | str | Callable[..., Any]

# Zero-arg factory that builds a single (non-vectorized) Gymnasium env.
EnvFactory = Callable[[], gym.Env]

# Network input size inferred from an observation space (leaf / Dict / Tuple).
InputSizeFromSpace = (
    tuple[int, ...]
    | dict[str, tuple[int, ...]]
    | tuple[tuple[int, ...] | dict[str, tuple[int, ...]], ...]
)
# Network output size inferred from an action space (leaf / Dict / Tuple).
OutputSizeFromSpace = int | dict[str, int] | tuple[int | dict[str, int], ...]
# Observation shape for a Gymnasium space (leaf / Dict / Tuple).
ObsShape = tuple[int, ...] | dict[str, tuple[int, ...]] | tuple[tuple[int, ...], ...]

# Scores, episode lengths, and terminal info from an on-policy rollout collection.
# Info stays ``dict[str, Any]`` to match Gymnasium's open info bags.
RolloutReturn = tuple[
    list[float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]


class HFGeneratePrompts(TypedDict):
    """Prompt tensors prepared for HuggingFace ``generate``."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    stitch_prefix_ids: torch.Tensor | None
    initial_prompt_len: int | None


# The batch type an algorithm's ``learn`` consumes. Each concrete algorithm binds
# this to the exact shape its buffer/rollout produces (e.g. ``ReplayBatch`` for
# off-policy value methods, ``PreferencePrompts`` for DPO), so ``learn`` reads its
# batch with precise key/element typing instead of narrowing a broad union by hand.
ExperiencesT = TypeVar("ExperiencesT")


# ── Replay / rollout batch dataclasses (TensorClass) ─────────────────────────
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


# A wrapped agent's ``get_action`` return: a bare action, a per-agent action
# dict, or a tuple of either (e.g. PPO's ``(action, log_prob, entropy, value)``
# or MADDPG's ``(env_actions, raw_actions)``). Kept intentionally gradual: an
# ``AgentWrapper`` re-binds ``self.agent.get_action`` to its own across
# heterogeneous algorithms, so a concrete union — wider than any single
# algorithm's declared return — would not be assignable to that attribute.
ActionReturn = tuple[ActionType | Any, ...] | ActionType | Any
GymStepReturn = tuple[NumpyObsType, ActionType, float, MaybeObsList, GymInfo]
PzStepReturn = tuple[
    dict[str, NumpyObsType],
    ArrayDict,
    ArrayDict,
    ArrayDict,
    InfosDict,
]
# TokenObservationWrapper obs: ReasoningPrompts mid-episode, empty mapping at done.
TokenObsType = ReasoningPrompts | dict[str, Never]
TokenObsStepReturn = tuple[TokenObsType, float, bool, bool, dict[str, Any]]

# ── Network / module / optimizer aliases ─────────────────────────────────────
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

MultiAgentMutReturn = dict[str, MutationApplyDict]
MutationReturn = MutationApplyDict | MultiAgentMutReturn
PopulationType = list[EvolvableAlgorithmProtocol]
MutationMethod = Callable[[EvolvableAlgorithmProtocol], EvolvableAlgorithmProtocol]
ConfigType = IsDataclass | NetConfigType
StateDict = dict[str, Any] | dict[str, dict[str, Any]] | list[dict[str, Any]]

# Training / mutation hyperparameter bags passed to train_* and logging helpers.
InitHyperparams = dict[str, Any] | None

# Per-layer transformer KV cache: exact (key, value) tensor pair.
KVCacheType = tuple[torch.Tensor, torch.Tensor]
# Full stack of per-layer KV caches (inner tuple length may vary by backend).
PastKeyValues = tuple[tuple[torch.Tensor, ...], ...]

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
