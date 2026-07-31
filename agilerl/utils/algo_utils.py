# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import shutil
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import singledispatch
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    Protocol,
    TypeGuard,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from gymnasium import spaces
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
from torch import nn
from torch._dynamo import OptimizedModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing_extensions import TypeVarTuple, Unpack

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.modules.custom_components import NoisyLinear
from agilerl.modules.dummy import DummyEvolvable
from agilerl.protocols import (
    EvolvableAttributeType,
    EvolvableModuleProtocol,
    EvolvableNetworkProtocol,
)
from agilerl.typing import (
    ArrayOrTensor,
    BatchDimension,
    BPTTSequenceType,
    DeviceType,
    InputSizeFromSpace,
    LeafSpace,
    MaybeObsList,
    NetConfigType,
    NumpyObsType,
    ObservationType,
    ObsShape,
    OutputSizeFromSpace,
    SpaceLike,
    TensorMapping,
    TensorTuple,
    TorchObsType,
)

if TYPE_CHECKING:
    from agilerl.algorithms.core.base import EvolvableAlgorithm
    from agilerl.modules.base import ModuleDict

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import PeftConfig, PeftModel, get_peft_model
    from transformers import PreTrainedModel

    PreTrainedModelType = PeftModel | PreTrainedModel
else:
    # Annotations referencing PreTrainedModelType are evaluated at function
    # definition time, so provide a runtime placeholder when the LLM
    # dependencies are not installed.
    PreTrainedModelType = Any


# Layers whose forward output differs between train() and eval() mode.
MODE_SENSITIVE_MODULES = (
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
    nn.modules.dropout._DropoutNd,
    NoisyLinear,
)


def is_train_eval_invariant(module: nn.Module) -> bool:
    """Whether ``module`` produces identical outputs in train and eval mode.

    :param module: Network to inspect.
    :type module: torch.nn.Module
    :return: True if train/eval modes are equivalent.
    :rtype: bool
    """
    return not any(isinstance(m, MODE_SENSITIVE_MODULES) for m in module.modules())


@contextmanager
def eval_mode(*modules: nn.Module, mode_invariant: bool) -> Iterator[None]:
    """Put ``modules`` in eval mode for the block, restoring train mode afterwards.

    Does nothing when ``mode_invariant``, since the toggle would be a no-op and
    each call otherwise walks the whole module tree twice.

    :param modules: Networks to switch into eval mode.
    :type modules: torch.nn.Module
    :param mode_invariant: Whether train and eval modes are equivalent.
    :type mode_invariant: bool
    """
    if mode_invariant:
        yield
        return

    for module in modules:
        module.eval()
    try:
        yield
    finally:
        for module in modules:
            module.train()


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Update ``target <- tau * source + (1 - tau) * target``, fused across parameters.

    :param source: Network whose parameters are copied from.
    :type source: torch.nn.Module
    :param target: Target network updated in place.
    :type target: torch.nn.Module
    :param tau: Interpolation weight toward ``source``.
    :type tau: float
    """
    torch._foreach_lerp_(list(target.parameters()), list(source.parameters()), tau)


def adam_kwargs(
    device: str | torch.device,
    accelerator: Accelerator | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Add ``fused=True`` to Adam kwargs when the fused CUDA kernel is usable.

    :param device: Device the optimizer's parameters live on.
    :type device: str | torch.device
    :param accelerator: Accelerator owning the step, if any.
    :type accelerator: accelerate.Accelerator | None
    :param kwargs: Base optimizer keyword arguments to augment.
    :return: ``kwargs``, with ``fused=True`` added when applicable.
    :rtype: dict[str, Any]
    """
    if (
        "fused" not in kwargs
        and not kwargs.get("capturable")
        and accelerator is None
        and "cuda" in str(device)
    ):
        kwargs["fused"] = True
    return kwargs


def configure_tf32_precision() -> None:
    """Configure TF32 using a single, legacy-compatible API path.

    Some runtimes import third-party code that still calls the legacy TF32 API.
    To avoid mixed "new + legacy" TF32 state in one process (which can break
    torch.compile/inductor), keep AgileRL on the legacy setter only.
    """
    torch.set_float32_matmul_precision("high")


def check_supported_space(observation_space: spaces.Space) -> None:
    """Check if the observation space is supported by AgileRL.

    :param observation_space: The observation space to check.
    :type observation_space: spaces.Space
    """
    assert isinstance(
        observation_space,
        spaces.Space,
    ), "Observation space must be an instance of gymnasium.spaces.Space."

    assert not isinstance(
        observation_space,
        (spaces.Graph, spaces.Sequence, spaces.OneOf),
    ), "AgileRL does not support Graph, Sequence, and OneOf spaces."

    if isinstance(observation_space, spaces.Dict):
        for subspace in observation_space.spaces.values():
            assert not isinstance(
                subspace,
                (spaces.Tuple, spaces.Dict),
            ), "AgileRL does not support nested Tuple and Dict spaces in Dict spaces."
            check_supported_space(subspace)
    elif isinstance(observation_space, spaces.Tuple):
        for subspace in observation_space.spaces:
            assert not isinstance(
                subspace,
                (spaces.Tuple, spaces.Dict),
            ), "AgileRL does not support nested Tuple and Dict spaces in Tuple spaces."
            check_supported_space(subspace)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        assert len(observation_space.nvec.shape) == 1, (
            "AgileRL does not support multi-dimensional MultiDiscrete spaces. Got shape "
            f"{observation_space.nvec.shape}."
        )


_LEAF_SPACE_TYPES = (
    spaces.Box,
    spaces.Discrete,
    spaces.MultiDiscrete,
    spaces.MultiBinary,
)


def is_str_keyed_dict(obj: object) -> TypeGuard[dict[str, object]]:
    """Narrow a value to a ``str``-keyed dict."""
    return isinstance(obj, dict)


def narrow_tensor(value: object) -> torch.Tensor:
    """Narrow a :class:`~tensordict.TensorDict` entry to the tensor it holds."""
    assert isinstance(value, torch.Tensor), (
        f"Expected a tensor entry, got {type(value).__name__}."
    )
    return value


def to_agent_tensors(per_agent: TensorDict, device: DeviceType) -> TensorMapping:
    """Move each per-agent leaf of ``per_agent`` onto ``device``."""
    return {
        agent_id: narrow_tensor(agent_tensor).to(device)
        for agent_id, agent_tensor in per_agent.items()
    }


@overload
def get_input_size_from_space(observation_space: LeafSpace) -> tuple[int, ...]: ...


@overload
def get_input_size_from_space(
    observation_space: spaces.Dict | dict[str, spaces.Space],
) -> dict[str, tuple[int, ...]]: ...


@overload
def get_input_size_from_space(
    observation_space: spaces.Tuple | list[spaces.Space] | tuple[spaces.Space, ...],
) -> tuple[tuple[int, ...] | dict[str, tuple[int, ...]], ...]: ...


@overload
def get_input_size_from_space(
    observation_space: SpaceLike,
) -> InputSizeFromSpace: ...


def get_input_size_from_space(observation_space: SpaceLike) -> InputSizeFromSpace:
    """Return the dimension of the state space as it pertains to the underlying
    networks (i.e. the input size of the networks).

    :param observation_space: The observation space of the environment.
    :type observation_space: spaces.Space or list[spaces.Space] or dict[str, spaces.Space].

    :return: The dimension of the state space.
    :rtype: tuple[int, ...] | dict[str, tuple[int, ...]] | tuple[tuple[int, ...] | dict[str, tuple[int, ...]], ...]
    """
    if isinstance(observation_space, spaces.Space):
        return _input_size(observation_space)
    if isinstance(observation_space, (list, tuple)):
        return tuple(_input_size(_require_space(space)) for space in observation_space)
    if isinstance(observation_space, dict):
        sizes_by_key: dict[str, InputSizeFromSpace] = {}
        for key, subspace in observation_space.items():
            sizes_by_key[key] = _input_size(_require_space(subspace))
        return sizes_by_key
    msg = f"Can't access state dimensions for {type(observation_space)} spaces."
    raise AttributeError(
        msg,
    )


def _require_space(space: object) -> spaces.Space:
    assert isinstance(space, spaces.Space)
    return space


def _input_size_leaf(space: spaces.Space) -> tuple[int, ...]:
    """Input size for a leaf observation space."""
    if isinstance(space, spaces.Discrete):
        return (int(space.n),)
    if isinstance(space, spaces.MultiDiscrete):
        return (int(sum(space.nvec)),)
    if isinstance(space, spaces.Box):
        return tuple(int(dim) for dim in space.shape)
    if isinstance(space, spaces.MultiBinary):
        n = space.n
        if isinstance(n, tuple):
            return tuple(int(dim) for dim in n)
        return (int(n),)
    msg = f"Can't access state dimensions for {type(space)} spaces."
    raise AttributeError(msg)


def _input_size(space: spaces.Space) -> InputSizeFromSpace:
    """Input size for a leaf, or a (possibly nested) Dict / Tuple space."""
    if isinstance(space, spaces.Dict):
        return {key: _input_size(sub) for key, sub in space.spaces.items()}
    if isinstance(space, spaces.Tuple):
        return tuple(_input_size(sub) for sub in space.spaces)
    return _input_size_leaf(space)


@overload
def get_output_size_from_space(action_space: LeafSpace) -> int: ...


@overload
def get_output_size_from_space(
    action_space: spaces.Dict | dict[str, spaces.Space],
) -> dict[str, int]: ...


@overload
def get_output_size_from_space(
    action_space: list[spaces.Space] | tuple[spaces.Space, ...],
) -> tuple[int | dict[str, int], ...]: ...


@overload
def get_output_size_from_space(
    action_space: SpaceLike,
) -> OutputSizeFromSpace: ...


def get_output_size_from_space(action_space: SpaceLike) -> OutputSizeFromSpace:
    """Return the dimension of the action space as it pertains to the underlying
    networks (i.e. the output size of the networks).

    :param action_space: The action space of the environment.
    :type action_space: spaces.Space or list[spaces.Space] or dict[str, spaces.Space].

    :return: The dimension of the action space.
    :rtype: int | dict[str, int] | tuple[int | dict[str, int], ...]
    """
    if isinstance(action_space, spaces.Space):
        return _output_size(action_space)
    if isinstance(action_space, (list, tuple)):
        return tuple(_output_size(_require_space(space)) for space in action_space)
    if isinstance(action_space, dict):
        sizes_by_key: dict[str, OutputSizeFromSpace] = {}
        for key, subspace in action_space.items():
            sizes_by_key[key] = _output_size(_require_space(subspace))
        return sizes_by_key
    msg = f"Can't access action dimensions for {type(action_space)} spaces."
    raise AttributeError(
        msg,
    )


def _output_size_leaf(space: spaces.Space) -> int:
    """Output size for a leaf action space."""
    if isinstance(space, spaces.Discrete):
        return int(space.n)
    if isinstance(space, spaces.MultiBinary):
        return int(np.prod(space.shape))
    if isinstance(space, spaces.MultiDiscrete):
        return int(sum(space.nvec))
    if isinstance(space, spaces.Box):
        # Continuous actions are one-dimensional
        return int(space.shape[0])
    msg = f"Can't access action dimensions for {type(space)} spaces."
    raise AttributeError(msg)


def _output_size(space: spaces.Space) -> OutputSizeFromSpace:
    """Output size for a leaf, or a (possibly nested) Dict / Tuple space."""
    if isinstance(space, spaces.Dict):
        return {key: _output_size(sub) for key, sub in space.spaces.items()}
    if isinstance(space, spaces.Tuple):
        return tuple(_output_size(sub) for sub in space.spaces)
    return _output_size_leaf(space)


def share_encoder_parameters(
    policy: EvolvableNetworkProtocol,
    *others: EvolvableNetworkProtocol,
) -> None:
    """Shares the encoder parameters between the policy and any number of other networks.

    :param policy: The policy network whose encoder parameters will be used.
    :type policy: EvolvableNetworkProtocol
    :param others: The other networks whose encoder parameters will be pinned to the policy.
    :type others: EvolvableNetworkProtocol
    """
    from agilerl.networks.base import EvolvableNetwork

    assert isinstance(policy, EvolvableNetwork), "Policy must be an EvolvableNetwork"
    networks: list[EvolvableNetwork] = []
    for other in others:
        assert isinstance(other, EvolvableNetwork), (
            "All others must be EvolvableNetwork"
        )
        networks.append(other)

    # detaching encoder parameters from computation graph reduces
    # memory overhead and speeds up training
    param_vals: TensorDict = from_module(policy.encoder).detach()
    for network in networks:
        target_params: TensorDict = param_vals.clone().lock_()
        target_params.to_module(network.encoder)


def get_hidden_states_shape_from_model(
    model: nn.Module,
) -> dict[str, tuple[int | type[BatchDimension], ...]]:
    """Loops through all of the modules in the model and checks if they have a
    `hidden_state_architecture` attribute. If they do, it adds the items to a
    dictionary and returns it. This should make it easier to initialize the
    hidden states of the model.

    :param model: The model to get the hidden states from.
    :type model: nn.Module
    :return: The hidden states shape from the model.
    :rtype: dict[str, tuple[int | type[BatchDimension], ...]]
    """
    hidden_state_architecture: dict[str, tuple[int | type[BatchDimension], ...]] = {}
    for _, module in model.named_modules():
        if hasattr(module, "hidden_state_architecture"):
            hidden_state_architecture.update(
                {
                    f"{module.name}_{k}": v
                    for k, v in module.hidden_state_architecture.items()
                },
            )

    return hidden_state_architecture


def extract_sequences_from_episode(
    episode: torch.Tensor,
    max_seq_len: int,
    sequence_type: BPTTSequenceType = BPTTSequenceType.CHUNKED,
) -> list[torch.Tensor]:
    """Extract sequences from an episode.

    - `BPTTSequenceType.CHUNKED`: Extracts sequences by chunking the episode into unique
        chunks of size `max_seq_len`. This is the most memory efficient and default option.
    - `BPTTSequenceType.MAXIMUM`: Extracts all possible sequences in an episode by taking a
        maximum of `max_seq_len` steps at a time. This is the most memory-intensive option.
    - `BPTTSequenceType.FIFTY_PERCENT_OVERLAP`: Extracts sequences by taking a maximum of
        `max_seq_len` steps at a time, with 50% overlap between sequences.

    :param episode: The episode to extract sequences from.
    :type episode: torch.Tensor
    :param max_seq_len: The maximum sequence length.
    :type max_seq_len: int
    :param sequence_type: The type of sequence to extract.
    :type sequence_type: BPTTSequenceType
    :return: The sequences extracted from the episode.
    :rtype: list[torch.Tensor]
    """
    assert max_seq_len > 0, "max_seq_len must be greater than 0"
    assert len(episode) > 0, "episode must be non-empty"
    assert max_seq_len <= len(
        episode,
    ), "max_seq_len must be less than or equal to the length of the episode"

    if sequence_type == BPTTSequenceType.CHUNKED:
        num_chunks = max(1, len(episode) // max_seq_len)
        sequences = [
            episode[chunk_i * max_seq_len : (chunk_i + 1) * max_seq_len]
            for chunk_i in range(num_chunks)
        ]
    elif sequence_type == BPTTSequenceType.MAXIMUM:
        sequences = [
            episode[start : start + max_seq_len]
            for start in range(len(episode) - max_seq_len + 1)
        ]
    elif sequence_type == BPTTSequenceType.FIFTY_PERCENT_OVERLAP:
        step_size = max_seq_len // 2
        sequences = [
            episode[start : start + max_seq_len]
            for start in range(0, len(episode) - max_seq_len + 1, step_size)
        ]
    else:
        msg = f"Received unrecognized sequence type: {sequence_type}"
        raise NotImplementedError(
            msg,
        )
    return sequences


def multi_dim_clamp(
    min_val: float | torch.Tensor,
    max_val: float | torch.Tensor,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Multi-dimensional clamp function.

    :param min_val: Minimum value or array of minimum values
    :type min_val: float | torch.Tensor
    :param max_val: Maximum value or array of maximum values
    :type max_val: float | torch.Tensor
    :param input_tensor: Input tensor to be clamped
    :type input_tensor: torch.Tensor
    :return: Clamped tensor
    :rtype: torch.Tensor
    """
    if not isinstance(min_val, torch.Tensor) and not isinstance(max_val, torch.Tensor):
        return torch.clamp(input_tensor, min_val, max_val)

    # torch.min/torch.max require tensor bounds on the input's device
    min_t = torch.as_tensor(min_val, device=input_tensor.device)
    max_t = torch.as_tensor(max_val, device=input_tensor.device)
    clamped: torch.Tensor = torch.max(torch.min(input_tensor, max_t), min_t)
    return clamped.to(input_tensor.dtype)


def is_image_space(space: spaces.Space) -> bool:
    """Check if the space is an image space. We ignore dtype and number of channels
    checks.

    :param space: Input space
    :type space: spaces.Space

    :return: True if the space is an image space, False otherwise
    :rtype: bool
    """
    return isinstance(space, spaces.Box) and len(space.shape) == 3


def is_channels_last(space: spaces.Box) -> bool:
    """Detect whether a 3-D Box space is in channels-last (HWC) format.

    Uses a simple heuristic: if the smallest dimension is last, the
    observation is assumed to be ``(H, W, C)`` — i.e. channels-last.

    :param space: A gymnasium Box space.
    :type space: spaces.Box
    :returns: ``True`` when the space looks like a channels-last image.
    :rtype: bool
    """
    if not is_image_space(space):
        return False
    return int(np.argmin(space.shape)) == 2


def needs_image_transpose(observation_space: spaces.Space) -> bool:
    """Check whether *any* image subspace requires a channels-last → first transpose.

    Recursively inspects :class:`~gymnasium.spaces.Dict` and
    :class:`~gymnasium.spaces.Tuple` spaces.

    :param observation_space: The observation space to inspect.
    :type observation_space: spaces.Space
    :returns: ``True`` if at least one Box subspace is channels-last.
    :rtype: bool
    """
    if isinstance(observation_space, spaces.Box):
        return is_channels_last(observation_space)
    if isinstance(observation_space, spaces.Dict):
        return any(needs_image_transpose(s) for s in observation_space.spaces.values())
    if isinstance(observation_space, spaces.Tuple):
        return any(needs_image_transpose(s) for s in observation_space.spaces)
    return False


def transpose_image_space(space: spaces.Space) -> spaces.Space:
    """Return a copy of *space* with channels-last Box subspaces transposed to CHW.

    Subspaces that are already channels-first (e.g. stacked frames with shape
    ``(C, H, W)``) are left untouched, so mixed Dict/Tuple spaces only have
    their channels-last leaves transposed.

    :param space: Space to transpose
    :type space: spaces.Space
    :return: Transposed space
    :rtype: spaces.Space
    """
    if isinstance(space, spaces.Box) and len(space.shape) == 3:
        if not is_channels_last(space):
            return space
        low = space.low.transpose(2, 0, 1)
        high = space.high.transpose(2, 0, 1)
        dtype = space.dtype
        assert dtype is not None, "Box spaces always carry a dtype"
        return spaces.Box(low=low, high=high, dtype=dtype.type)

    if isinstance(space, spaces.Dict):
        return spaces.Dict(
            {key: transpose_image_space(s) for key, s in space.spaces.items()}
        )

    if isinstance(space, spaces.Tuple):
        return spaces.Tuple(tuple(transpose_image_space(s) for s in space.spaces))

    return space


@overload
def transpose_image_observation(
    observation: torch.Tensor, original_space: spaces.Space
) -> torch.Tensor: ...


@overload
def transpose_image_observation(
    observation: np.ndarray, original_space: spaces.Space
) -> np.ndarray: ...


@overload
def transpose_image_observation(
    observation: dict[str, np.ndarray], original_space: spaces.Space
) -> dict[str, np.ndarray]: ...


@overload
def transpose_image_observation(
    observation: tuple[np.ndarray, ...], original_space: spaces.Space
) -> tuple[np.ndarray, ...]: ...


def transpose_image_observation(
    observation: NumpyObsType | torch.Tensor, original_space: spaces.Space
) -> NumpyObsType | torch.Tensor:
    """Transpose 3-D observations from HWC to CHW.

    Supports both NumPy arrays and PyTorch tensors. Observations that already
    match the channels-first layout the space leaf maps to (e.g. an
    always-channels-first stacked-frames leaf inside a mixed Dict space) are
    returned unchanged.

    :param observation: Observation
    :type observation: npt.NDArray | torch.Tensor
    :param original_space: Original observation space
    :type original_space: spaces.Space
    :return: Transposed observation
    :rtype: npt.NDArray | torch.Tensor
    """
    if isinstance(original_space, spaces.Box) and len(original_space.shape) == 3:
        # The channels-first layout this leaf should end up in
        shape = tuple(original_space.shape)
        target = (
            (shape[2], shape[0], shape[1])
            if is_channels_last(original_space)
            else shape
        )
        if isinstance(observation, torch.Tensor):
            ndim = observation.ndim
            if ndim == 3:
                if tuple(observation.shape) == target:
                    return observation
                return observation.permute(2, 0, 1)
            if ndim == 4:
                if tuple(observation.shape[1:]) == target:
                    return observation
                return observation.permute(0, 3, 1, 2)
        arr = np.asarray(observation)
        if arr.ndim == 3:
            if tuple(arr.shape) == target:
                return arr
            return arr.transpose(2, 0, 1)
        if arr.ndim == 4:
            if tuple(arr.shape[1:]) == target:
                return arr
            return arr.transpose(0, 3, 1, 2)

    if isinstance(original_space, spaces.Dict):
        assert is_str_keyed_dict(observation), (
            f"Expected dict observation for Dict space, got {type(observation)}"
        )
        transposed: dict[str, Any] = {}
        for key, value in observation.items():
            assert isinstance(value, (np.ndarray, torch.Tensor))
            transposed[key] = transpose_image_observation(value, original_space[key])
        return transposed

    if isinstance(original_space, spaces.Tuple):
        assert isinstance(observation, tuple), (
            f"Expected tuple observation for Tuple space, got {type(observation)}"
        )
        transposed_tuple: list[Any] = []
        for o, s in zip(observation, original_space.spaces, strict=True):
            assert isinstance(o, (np.ndarray, torch.Tensor))
            transposed_tuple.append(transpose_image_observation(o, s))
        return tuple(transposed_tuple)

    return observation


@overload
def get_obs_shape(space: LeafSpace) -> tuple[int, ...]: ...


@overload
def get_obs_shape(space: spaces.Dict) -> dict[str, tuple[int, ...]]: ...


@overload
def get_obs_shape(space: spaces.Tuple) -> tuple[tuple[int, ...], ...]: ...


@overload
def get_obs_shape(space: spaces.Space) -> ObsShape: ...


def get_obs_shape(space: spaces.Space) -> ObsShape:
    """Return the shape of the observation space.

    :param space: Observation space
    :type space: spaces.Space
    :return: Shape of the observation space
    :rtype: tuple[int, ...] | dict[str, tuple[int, ...]] | tuple[tuple[int, ...], ...]
    """
    if isinstance(space, _LEAF_SPACE_TYPES):
        return _obs_shape_leaf(space)
    if isinstance(space, spaces.Dict):
        return {
            key: _obs_shape_leaf(subspace) for (key, subspace) in space.spaces.items()
        }
    if isinstance(space, spaces.Tuple):
        return tuple(_obs_shape_leaf(subspace) for subspace in space.spaces)
    msg = f"{space} observation space is not supported"
    raise NotImplementedError(msg)


def _obs_shape_leaf(space: spaces.Space) -> tuple[int, ...]:
    """Observation shape for a leaf space."""
    if isinstance(space, spaces.Box):
        return tuple(int(dim) for dim in space.shape)
    if isinstance(space, spaces.Discrete):
        return (1,)
    if isinstance(space, spaces.MultiDiscrete):
        return (len(space.nvec),)
    if isinstance(space, spaces.MultiBinary):
        return tuple(int(dim) for dim in space.shape)
    msg = f"{space} observation space is not supported"
    raise NotImplementedError(msg)


def get_num_actions(space: spaces.Space) -> int:
    """Return the number of actions.

    :param space: Action space
    :type space: spaces.Space
    :return: Number of actions
    :rtype: int
    """
    if isinstance(space, spaces.Box):
        return spaces.flatdim(space)
    if isinstance(space, spaces.Discrete):
        return 1
    if isinstance(space, spaces.MultiDiscrete):
        return len(space.nvec)
    if isinstance(space, spaces.MultiBinary):
        return int(np.prod(space.shape))
    msg = f"{space} action space is not supported by AgileRL."
    raise NotImplementedError(msg)


def get_action_mask_size(space: spaces.Space) -> int:
    """Return the size of the action mask for a given action space.

    Action masks are only applicable to discrete action spaces. For continuous
    (Box) spaces, returns 0.

    :param space: Action space
    :type space: spaces.Space
    :return: Size of the action mask, or 0 if masking is not applicable
    :rtype: int
    """
    if isinstance(space, spaces.Discrete):
        return int(space.n)
    if isinstance(space, spaces.MultiDiscrete):
        return int(sum(space.nvec))
    if isinstance(space, spaces.MultiBinary):
        return int(np.prod(space.shape))
    return 0


@runtime_checkable
class _SupportsNumEnvs(Protocol):
    num_envs: int


def get_num_envs(env: object) -> int:
    """Return a vectorized env's sub-environment count, or 1 if it exposes none."""
    return env.num_envs if isinstance(env, _SupportsNumEnvs) else 1


# Copies preserve the caller's concrete module type. Each positional argument
# keeps its own type through an independent TypeVar, so a heterogeneous call
# (actor, critic, ...) returns a precise tuple rather than a widened list.
ModuleT = TypeVar("ModuleT", bound=EvolvableModuleProtocol)
CopyT1 = TypeVar("CopyT1")
CopyT2 = TypeVar("CopyT2")
CopyT3 = TypeVar("CopyT3")


@overload
def make_safe_deepcopies(args: list[ModuleT], /) -> list[ModuleT]: ...


@overload
def make_safe_deepcopies(args: ModuleT, /) -> ModuleT: ...


@overload
def make_safe_deepcopies(a: CopyT1, b: CopyT2, /) -> tuple[CopyT1, CopyT2]: ...


@overload
def make_safe_deepcopies(
    a: CopyT1, b: CopyT2, c: CopyT3, /
) -> tuple[CopyT1, CopyT2, CopyT3]: ...


def make_safe_deepcopies(
    *args: EvolvableModuleProtocol | list[EvolvableModuleProtocol],
) -> (
    EvolvableModuleProtocol
    | list[EvolvableModuleProtocol]
    | tuple[EvolvableModuleProtocol | list[EvolvableModuleProtocol], ...]
):
    """Make deep copies of EvolvableModule objects and their attributes.

    :param args: EvolvableModuleProtocol or lists of EvolvableModuleProtocol objects to copy.
    :type args: EvolvableModuleProtocol | list[EvolvableModuleProtocol].

    :return: Deep copies of the EvolvableModule objects and their attributes.
    :rtype: EvolvableModuleProtocol | list[EvolvableModuleProtocol]
    """
    copies: list[EvolvableModuleProtocol | list[EvolvableModuleProtocol]] = []
    for arg in args:
        if isinstance(arg, list):
            inner_copies: list[EvolvableModuleProtocol] = []
            for inner_arg in arg:
                assert isinstance(inner_arg, EvolvableModuleProtocol)
                inner_copies.append(inner_arg.clone())
            copies.append(inner_copies)
        else:
            copies.append(arg.clone())

    return copies[0] if len(copies) == 1 else tuple(copies)


def isroutine(obj: object) -> bool:
    """Check if an attribute is a routine, considering also methods wrapped by
    CudaGraphModule.

    :param attr: The attribute to check.
    :type attr: str

    :return: True if the attribute is a routine, False otherwise.
    :rtype: bool
    """
    if isinstance(obj, CudaGraphModule):
        return True

    return inspect.isroutine(obj)


def recursive_check_module_attrs(obj: object, networks_only: bool = False) -> bool:
    """Recursively check if the object has any attributes that are EvolvableModuleProtocol objects or Optimizer's,
    excluding metaclasses.

    :param obj: The object to check for EvolvableModuleProtocol objects or Optimizer's.
    :type obj: Any
    :param networks_only: If True, only check for EvolvableModule objects, defaults to False
    :type networks_only: bool, optional

    :return: True if the object has any attributes that are EvolvableModuleProtocol objects or Optimizer's, False otherwise.
    :rtype: bool
    """
    from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
    from agilerl.modules.base import EvolvableModule

    check_types = (OptimizedModule, EvolvableModule)
    if not networks_only:
        check_types += (OptimizerWrapper,)

    # Exclude metaclasses
    if isinstance(obj, type):
        return False

    if isinstance(obj, check_types):
        return True
    if isinstance(obj, Optimizer):
        msg = "Optimizer objects should be wrapped by OptimizerWrapper."
        raise TypeError(msg)
    if isinstance(obj, dict):
        return any(
            recursive_check_module_attrs(v, networks_only=networks_only)
            for v in obj.values()
        )
    if isinstance(obj, list):
        return any(
            recursive_check_module_attrs(v, networks_only=networks_only) for v in obj
        )
    return False


@overload
def chkpt_attribute_to_device(
    chkpt_dict: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Any]: ...


@overload
def chkpt_attribute_to_device(
    chkpt_dict: list[dict[str, Any]],
    device: str | torch.device,
) -> list[dict[str, Any]]: ...


def chkpt_attribute_to_device(
    chkpt_dict: dict[str, Any] | list[dict[str, Any]],
    device: str | torch.device,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Place checkpoint attributes on device. Used when loading saved agents.

    :param chkpt_dict: Checkpoint dictionary or list of checkpoint dictionaries
    :type chkpt_dict: dict[str, Any] | list[dict[str, Any]]
    :param device: Device for accelerated computing, 'cpu' or 'cuda'
    :type device: str

    :return: Checkpoint dictionary (or list thereof) with attributes on device
    :rtype: dict[str, Any] | list[dict[str, Any]]
    """
    if isinstance(chkpt_dict, list):
        return [chkpt_attribute_to_device(chkpt, device) for chkpt in chkpt_dict]

    assert isinstance(chkpt_dict, dict), f"Expected dict, got {type(chkpt_dict)}"

    for key, value in chkpt_dict.items():
        if isinstance(value, torch.Tensor):
            chkpt_dict[key] = value.to(device)

    return chkpt_dict


def filter_init_dict(init_dict: Mapping[str, object], cls: type) -> dict[str, Any]:
    """Filter the init dict to only include parameters that are valid for the given class.

    :param init_dict: Initialization dictionary
    :type init_dict: Mapping[str, object]
    :param cls: Class to filter the init dict for
    :type cls: type

    :return: Filtered initialization dictionary
    :rtype: dict[str, Any]
    """
    init_params = inspect.signature(cls.__init__).parameters.keys()
    return {k: v for k, v in init_dict.items() if k in init_params}


def key_in_nested_dict(nested_dict: Mapping[str, Any], target: str) -> bool:
    """Determine if key is in nested dictionary.

    :param nested_dict: Nested dictionary
    :type nested_dict: Mapping[str, dict[str, ...]]
    :param target: Target string
    :type target: str

    :return: True if key is in nested dictionary, False otherwise
    :rtype: bool
    """
    for k, v in nested_dict.items():
        if k == target:
            return True
        if isinstance(v, dict):
            return key_in_nested_dict(v, target)
    return False


def remove_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove _orig_mod prefix on state dict created by torch compile.

    :param state_dict: model state dict
    :type state_dict: dict
    :return: state dict with prefix removed
    :rtype: dict[str, Any]
    """
    return OrderedDict(
        [
            (k.split(".", 1)[1], v) if k.startswith("_orig_mod") else (k, v)
            for k, v in state_dict.items()
        ],
    )


def module_checkpoint_dict(
    module: EvolvableAttributeType, name: str
) -> dict[str, object]:
    """Return a dictionary containing the module's class, init dict, and state dict.

    :param module: The module to checkpoint.
    :type module: EvolvableAttributeType
    :param name: The name of the attribute to checkpoint.
    :type name: str

    :return: A dictionary containing the module's class, init dict, and state dict.
    :rtype: dict[str, object]
    """
    from agilerl.modules.base import EvolvableModule, ModuleDict

    # Checkpointing is only invoked on network attributes; the wider
    # EvolvableAttributeType arms (optimizers) are handled by OptimizerWrapper.
    if isinstance(module, ModuleDict):
        return module_checkpoint_multiagent(module, name)

    assert isinstance(module, EvolvableModule)
    return module_checkpoint_single(module, name)


def module_checkpoint_single(
    module: EvolvableModuleProtocol,
    name: str,
) -> dict[str, object]:
    """Return a dictionary containing the module's class, init dict, and state dict.

    :param module: The module to checkpoint.
    :type module: EvolvableModuleProtocol
    :param name: The name of the attribute to checkpoint.
    :type name: str
    :return: A dictionary containing the module's class, init dict, and state dict.
    :rtype: dict[str, object]
    """
    module_cls = (
        module._orig_mod.__class__
        if isinstance(module, OptimizedModule)
        else module.__class__
    )
    init_dict = module.init_dict
    state_dict = remove_compile_prefix(module.state_dict())
    return {
        f"{name}_cls": module_cls,
        f"{name}_init_dict": init_dict,
        f"{name}_state_dict": state_dict,
        f"{name}_module_dict_cls": None,
    }


def module_checkpoint_multiagent(
    module: "ModuleDict[Any]", name: str
) -> dict[str, object]:
    """Return a dictionary containing the module's class, init dict, and state dict.

    :param module: The module to checkpoint.
    :type module: ModuleDictProtocol
    :param name: The name of the attribute to checkpoint.
    :type name: str
    :return: A dictionary containing the module's class, init dict, and state dict.
    :rtype: dict[str, object]
    """
    agent_module_cls = OrderedDict()
    agent_init_dicts = OrderedDict()
    agent_state_dicts = OrderedDict()
    for agent_id, agent_mod in module.items():
        agent_mod_cls = (
            agent_mod._orig_mod.__class__
            if isinstance(agent_mod, OptimizedModule)
            else agent_mod.__class__
        )
        agent_module_cls[agent_id] = agent_mod_cls
        agent_init_dicts[agent_id] = agent_mod.init_dict
        agent_state_dicts[agent_id] = remove_compile_prefix(agent_mod.state_dict())

    return {
        f"{name}_cls": agent_module_cls,
        f"{name}_init_dict": agent_init_dicts,
        f"{name}_state_dict": agent_state_dicts,
        f"{name}_module_dict_cls": module.__class__,
    }


def format_shared_critic_encoder(
    encoder_configs: Mapping[str, NetConfigType],
) -> NetConfigType:
    """Format the shared critic  (i.e. `EvolvableMultiInput`) config from the available
    encoder configs from all of the sub-agents. This dictionary is built when extracting the net
    config passed by the user in `MultiAgentAlgorithm.extract_net_config`.

    .. note::
        If the user specified multiple different MLP configurations for different sub-agents /
        groups, the deepest MLP config will be used for the shared critics `EvolvableMLP`.

    :param encoder_configs: Network configuration
    :type encoder_configs: Mapping[str, NetConfigType]
    :return: Formatted shared critic encoder config
    :rtype: NetConfigType
    """
    encoder_config: NetConfigType = {}
    init_dicts: dict[str, NetConfigType] = {}
    for encoder_key, config in encoder_configs.items():
        if encoder_key == "mlp_config":
            encoder_config["mlp_config"] = config
            hidden_size = config.get("hidden_size", [32])
            encoder_config["latent_dim"] = (
                hidden_size[-1] if isinstance(hidden_size, list) else 32
            )
            min_mlp_nodes = config.get("min_mlp_nodes", 8)
            encoder_config["min_latent_dim"] = (
                min_mlp_nodes if isinstance(min_mlp_nodes, int) else 8
            )
            max_mlp_nodes = config.get("max_mlp_nodes", 1024)
            encoder_config["max_latent_dim"] = (
                max_mlp_nodes if isinstance(max_mlp_nodes, int) else 1024
            )
        else:
            init_dicts[encoder_key] = config

    if init_dicts:
        encoder_config["init_dicts"] = init_dicts

    return encoder_config


def get_deepest_head_config(
    net_config: NetConfigType,
    agent_ids: list[str],
) -> NetConfigType:
    """Return the deepest head config from the nested net config.

    :param net_config: Network configuration
    :type net_config: NetConfigType
    :param agent_ids: List of agent IDs
    :type agent_ids: list[str]
    :return: Largest head config
    """
    assert all(agent_id in net_config for agent_id in agent_ids), (
        "All passed agent IDs must be present in the net config."
    )

    deepest = None
    for agent_id in agent_ids:
        agent_config = net_config[agent_id]
        agent_head_config = agent_config.get("head_config", None)
        if agent_head_config is not None:
            if deepest is None or len(agent_head_config["hidden_size"]) > len(
                deepest["hidden_size"],
            ):
                deepest = agent_head_config

    if deepest is None:
        msg = "No head config found in the passed net config."
        raise ValueError(msg)

    return deepest


def concatenate_spaces(space_list: list[spaces.Space]) -> spaces.Space:
    """Concatenates a list of spaces into a single space. If spaces correspond to images,
    we check that their shapes are the same and use the first space's shape as the shape of the
    concatenated space.

    :param space_list: List of spaces to concatenate
    :type space_list: list[spaces.Space]
    :return: Concatenated space
    :rtype: spaces.Space
    """
    dict_spaces = [space for space in space_list if isinstance(space, spaces.Dict)]
    if len(dict_spaces) == len(space_list):
        return spaces.Dict(
            {
                key: concatenate_spaces([space[key] for space in dict_spaces])
                for key in dict_spaces[0].spaces
            },
        )

    tuple_spaces = [space for space in space_list if isinstance(space, spaces.Tuple)]
    if len(tuple_spaces) == len(space_list):
        return spaces.Tuple(
            [
                concatenate_spaces([space[i] for space in tuple_spaces])
                for i in range(len(tuple_spaces[0]))
            ],
        )

    box_spaces = [space for space in space_list if isinstance(space, spaces.Box)]
    if len(box_spaces) == len(space_list):
        # Require image spaces to have the same shape in order to concatenate
        if all(is_image_space(space) for space in box_spaces):
            assert all(space.shape == box_spaces[0].shape for space in box_spaces), (
                "Cannot concatenate image spaces with different CxHxW dimensions."
            )

            return box_spaces[0]

        low = np.concatenate([space.low for space in box_spaces], axis=0)
        high = np.concatenate([space.high for space in box_spaces], axis=0)
        dtype = box_spaces[0].dtype
        assert dtype is not None, "Box spaces always carry a dtype"
        return spaces.Box(low=low, high=high, dtype=dtype.type)

    discrete_spaces = [
        space for space in space_list if isinstance(space, spaces.Discrete)
    ]
    if len(discrete_spaces) == len(space_list):
        n = sum(int(space.n) for space in discrete_spaces)
        return spaces.Discrete(n)

    multidiscrete_spaces = [
        space for space in space_list if isinstance(space, spaces.MultiDiscrete)
    ]
    if len(multidiscrete_spaces) == len(space_list):
        nvec = np.concatenate([space.nvec for space in multidiscrete_spaces], axis=0)
        return spaces.MultiDiscrete(nvec)

    msg = f"Unsupported space types: { {type(space) for space in space_list} }"
    raise TypeError(
        msg,
    )


@overload
def obs_to_tensor(obs: TensorDict, device: str | torch.device) -> TensorDict: ...


@overload
def obs_to_tensor(
    obs: ArrayOrTensor | Number | list[Any], device: str | torch.device
) -> torch.Tensor: ...


@overload
def obs_to_tensor(
    obs: Mapping[str, ArrayOrTensor], device: str | torch.device
) -> dict[str, torch.Tensor]: ...


@overload
def obs_to_tensor(
    obs: tuple[ArrayOrTensor, ...], device: str | torch.device
) -> tuple[torch.Tensor, ...]: ...


@overload
def obs_to_tensor(obs: ObservationType, device: str | torch.device) -> TorchObsType: ...


@overload
def obs_to_tensor(
    obs: dict[str, ObservationType], device: str | torch.device
) -> dict[str, TorchObsType]: ...


def obs_to_tensor(
    obs: object, device: str | torch.device
) -> TorchObsType | dict[str, TorchObsType]:
    """Move the observation to the given device as a PyTorch tensor.

    :param obs: Observation to convert
    :type obs: ObservationType
    :param device: PyTorch device
    :type device: str | torch.device
    :return: PyTorch tensor of the observation on a desired device.
    :rtype: TorchObsType
    """
    if isinstance(obs, TensorDict):
        return obs if obs.device == device else obs.to(device)
    if isinstance(obs, torch.Tensor):
        return obs.float().to(device)
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device).float()
    if is_str_keyed_dict(obs):
        converted: dict[str, torch.Tensor] = {}
        for key, _obs in obs.items():
            converted[key] = torch.as_tensor(_obs, device=device).float()
        return converted
    if isinstance(obs, tuple):
        return tuple(torch.as_tensor(_obs, device=device).float() for _obs in obs)
    if isinstance(obs, (list, Number)):
        return torch.tensor(obs, device=device).float()

    msg = f"Unrecognized type of observation {type(obs)}"
    raise TypeError(msg)


def get_vect_dim(
    observation: ObservationType | Mapping[str, ObservationType],
    observation_space: spaces.Space,
) -> int:
    """Return the number of vectorized environments given an observation and
    its corresponding space.

    :param observation: Observation
    :type observation: ObservationType
    :param observation_space: Observation space
    :type observation_space: spaces.Space
    :return: Number of vectorized environments
    """
    obs: object = observation
    space: spaces.Space = observation_space

    while isinstance(space, (spaces.Dict, spaces.Tuple)):
        if isinstance(space, spaces.Dict):
            assert is_str_keyed_dict(obs), (
                f"Expected dict observation for Dict space, got {type(obs)}"
            )
            first_key, obs = next(iter(obs.items()))
            space = space[first_key]
        else:
            assert isinstance(obs, tuple), (
                f"Expected tuple observation for Tuple space, got {type(obs)}"
            )
            obs = obs[0]
            space = space[0]

    space_shape = space.shape
    assert space_shape is not None, (
        f"{type(space)} spaces have no shape to infer the vectorization dimension from."
    )
    array = obs if isinstance(obs, np.ndarray) else np.array(obs)
    return array.shape[0] if len(array.shape) > len(space_shape) else 1


def add_placeholder_value(obs: torch.Tensor, placeholder_value: float) -> torch.Tensor:
    """Add placeholder value to observation.

    :param obs: Observation
    :type obs: torch.Tensor
    :param placeholder_value: Placeholder value
    :type placeholder_value: float
    :return: Observation with placeholder value
    :rtype: torch.Tensor
    """
    return torch.where(
        torch.isnan(obs),
        torch.full_like(obs, placeholder_value),
        obs,
    ).to(torch.float32)


ArrT = TypeVar("ArrT", npt.NDArray, torch.Tensor)


@singledispatch
def maybe_add_batch_dim(
    array_like: ArrT,
    space: LeafSpace,
    actions: bool = False,
) -> ArrT:
    """Add batch dimension if necessary.

    :param array_like: Array or tensor
    :type array_like: npt.NDArray | torch.Tensor
    :param space: Observation space
    :type space: spaces.Space
    :param actions: Whether the array is an action, defaults to False
    :type actions: bool, optional
    :return: Observation tensor with batch dimension
    :rtype: npt.NDArray | torch.Tensor
    """
    msg = f"Cannot add batch dimension for {type(array_like)}."
    raise TypeError(msg)


@maybe_add_batch_dim.register(np.ndarray)
def maybe_add_batch_dim_np(
    array_like: npt.NDArray,
    space: LeafSpace,
    actions: bool = False,
) -> npt.NDArray:
    space_shape = (
        get_input_size_from_space(space) if not actions else (get_num_actions(space),)
    )
    if len(array_like.shape) == len(space_shape):
        array_like = np.expand_dims(array_like, 0)
    elif len(array_like.shape) == len(space_shape) + 2:
        array_like = array_like.reshape(-1, *space_shape)
    elif len(array_like.shape) != len(space_shape) + 1:
        msg = f"Expected observation to have {len(space_shape) + 1} dimensions, got {len(array_like.shape)}."
        raise ValueError(
            msg,
        )

    return array_like


@maybe_add_batch_dim.register(torch.Tensor)
def maybe_add_batch_dim_torch(
    array_like: torch.Tensor,
    space: LeafSpace,
    actions: bool = False,
) -> torch.Tensor:
    space_shape = (
        get_input_size_from_space(space) if not actions else (get_num_actions(space),)
    )
    if array_like.ndim == len(space_shape):
        array_like = array_like.unsqueeze(0)
    elif array_like.ndim == len(space_shape) + 2:
        array_like = array_like.view(-1, *space_shape)
    elif array_like.ndim != len(space_shape) + 1:
        msg = f"Expected observation to have {len(space_shape) + 1} dimensions, got {len(array_like.shape)}."
        raise ValueError(
            msg,
        )

    return array_like


@singledispatch
def preprocess_observation(
    observation_space: spaces.Space,
    observation: ObservationType,
    device: str | torch.device = "cpu",
    normalize_images: bool = True,
    placeholder_value: float | None = None,
    swap_channels: bool = False,
) -> TorchObsType:
    """Preprocesses observations for forward pass through neural network.

    :param observation_space: The observation space of the environment, defaults to the agent's observation space
    :type observation_space: spaces.Space
    :param observation: Observations of environment
    :type observation: ObservationType
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to "cpu"
    :type device: str | torch.device, optional
    :param normalize_images: Normalize images from [0. 255] to [0, 1], defaults to True
    :type normalize_images: bool, optional
    :param placeholder_value: The value to use as placeholder for missing observations, defaults to None.
    :type placeholder_value: Any | None, optional
    :param swap_channels: Whether to swap channels, defaults to False
    :type swap_channels: bool, optional
    :return: Preprocessed observations
    :rtype: TorchObsType
    """
    msg = f"AgileRL currently doesn't support {type(observation_space)} spaces."
    raise TypeError(
        msg,
    )


@preprocess_observation.register(spaces.Dict)
def preprocess_dict_observation(
    observation_space: spaces.Dict,
    observation: dict[str, npt.NDArray | torch.Tensor],
    device: str | torch.device = "cpu",
    normalize_images: bool = True,
    placeholder_value: float | None = None,
    swap_channels: bool = False,
) -> TensorMapping:
    """Preprocess dictionary observations.

    :param observation: Dictionary observation
    :type observation: dict[str, npt.NDArray | torch.Tensor]
    :param observation_space: Dictionary observation space
    :type observation_space: spaces.Dict
    :param device: Computing device
    :type device: str | torch.device, optional
    :param normalize_images: Whether to normalize images
    :type normalize_images: bool, optional
    :param placeholder_value: Value to replace NaNs with
    :type placeholder_value: Any | None, optional
    :param swap_channels: Whether to swap channels, defaults to False
    :type swap_channels: bool, optional
    :return: Preprocessed dictionary observation
    :rtype: dict[str, torch.Tensor]
    """
    assert isinstance(
        observation,
        (dict, TensorDict),
    ), f"Expected dict, got {type(observation)}"

    preprocessed_obs: dict[str, Any] = OrderedDict()
    for key, _obs in observation.items():
        preprocessed_obs[key] = preprocess_observation(
            observation_space[key],
            observation=_obs,
            device=device,
            normalize_images=normalize_images,
            placeholder_value=placeholder_value,
            swap_channels=swap_channels,
        )

    return preprocessed_obs


@preprocess_observation.register(spaces.Tuple)
def preprocess_tuple_observation(
    observation_space: spaces.Tuple,
    observation: tuple[npt.NDArray | torch.Tensor, ...],
    device: str | torch.device = "cpu",
    normalize_images: bool = True,
    placeholder_value: float | None = None,
    swap_channels: bool = False,
) -> TensorTuple:
    """Preprocess tuple observations.

    :param observation: Tuple observation
    :type observation: tuple[npt.NDArray | torch.Tensor, ...]
    :param observation_space: Tuple observation space
    :type observation_space: spaces.Tuple
    :param device: Computing device
    :type device: str | torch.device, optional
    :param normalize_images: Whether to normalize images
    :type normalize_images: bool, optional
    :param placeholder_value: Value to replace NaNs with
    :type placeholder_value: Any | None, optional
    :param swap_channels: Whether to swap channels, defaults to False
    :type swap_channels: bool, optional
    :return: Preprocessed tuple observation
    :rtype: tuple[torch.Tensor, ...]
    """
    obs_tuple: tuple[Any, ...]
    if isinstance(observation, TensorDict):
        # Convert to tuple with values ordered by index at the end of key
        dict_keys = [key for key in observation.keys() if isinstance(key, str)]
        dict_keys.sort(key=lambda x: int(x.split("_")[-1]))
        obs_tuple = tuple(observation[key] for key in dict_keys)
    else:
        assert isinstance(
            observation,
            tuple,
        ), f"Expected tuple observation, got {type(observation)}"
        obs_tuple = observation

    preprocessed: tuple[Any, ...] = tuple(
        preprocess_observation(
            _space,
            observation=_obs,
            device=device,
            normalize_images=normalize_images,
            placeholder_value=placeholder_value,
            swap_channels=swap_channels,
        )
        for _obs, _space in zip(obs_tuple, observation_space.spaces, strict=False)
    )
    return preprocessed


@preprocess_observation.register(spaces.Box)
def preprocess_box_observation(
    observation_space: spaces.Box,
    observation: npt.NDArray | torch.Tensor,
    device: str | torch.device = "cpu",
    normalize_images: bool = True,
    placeholder_value: float | None = None,
    swap_channels: bool = False,
) -> torch.Tensor:
    """Preprocess box observations (continuous spaces).

    :param observation: Box observation
    :type observation: npt.NDArray | torch.Tensor
    :param observation_space: Box observation space
    :type observation_space: spaces.Box
    :param device: Computing device
    :type device: str | torch.device, optional
    :param normalize_images: Whether to normalize images
    :type normalize_images: bool, optional
    :param placeholder_value: Value to replace NaNs with
    :type placeholder_value: Any | None, optional
    :param swap_channels: Whether to swap channels, defaults to False
    :type swap_channels: bool, optional
    :return: Preprocessed box observation
    :rtype: torch.Tensor
    """
    # Convert to tensor
    observation = obs_to_tensor(observation, device)

    # Replace NaNs with placeholder value if specified
    if placeholder_value is not None:
        observation = add_placeholder_value(observation, placeholder_value)

    if swap_channels:
        observation = transpose_image_observation(observation, observation_space)

    # Normalize images if applicable and specified
    if len(observation_space.shape) == 3 and normalize_images:
        observation = apply_image_normalization(observation, observation_space)

    # Check add batch dimension if necessary
    return maybe_add_batch_dim(observation, observation_space)


@preprocess_observation.register(spaces.Discrete)
def preprocess_discrete_observation(
    observation_space: spaces.Discrete,
    observation: npt.NDArray | torch.Tensor,
    device: str | torch.device = "cpu",
    normalize_images: bool = True,
    placeholder_value: float | None = None,
    swap_channels: bool = False,
) -> torch.Tensor:
    """Preprocess discrete observations.

    :param observation: Discrete observation
    :type observation: npt.NDArray | torch.Tensor
    :param observation_space: Discrete observation space
    :type observation_space: spaces.Discrete
    :param device: Computing device
    :type device: str | torch.device, optional
    :param normalize_images: Whether to normalize images
    :type normalize_images: bool, optional
    :param placeholder_value: Value to replace NaNs with
    :type placeholder_value: Any | None, optional
    :param swap_channels: Whether to swap channels, defaults to False
    :type swap_channels: bool, optional
    :return: Preprocessed discrete observation (one-hot encoded)
    :rtype: torch.Tensor
    """
    # Convert to tensor
    observation = obs_to_tensor(observation, device)

    # Replace NaNs with placeholder value if specified
    if placeholder_value is not None:
        observation = add_placeholder_value(observation, placeholder_value)

    # One hot encoding of discrete observation
    observation = F.one_hot(
        observation.long(),
        num_classes=int(observation_space.n),
    ).float()

    if observation_space.n > 1:
        observation = observation.squeeze()  # If n == 1 then squeeze removes obs dim

    # Check add batch dimension if necessary
    return maybe_add_batch_dim(observation, observation_space)


@preprocess_observation.register(spaces.MultiDiscrete)
def preprocess_multidiscrete_observation(
    observation_space: spaces.MultiDiscrete,
    observation: npt.NDArray | torch.Tensor,
    device: str | torch.device = "cpu",
    normalize_images: bool = True,
    placeholder_value: float | None = None,
    swap_channels: bool = False,
) -> torch.Tensor:
    """Preprocess multi-discrete observations.

    :param observation: Multi-discrete observation
    :type observation: npt.NDArray | torch.Tensor
    :param observation_space: Multi-discrete observation space
    :type observation_space: spaces.MultiDiscrete
    :param device: Computing device
    :type device: str | torch.device, optional
    :param normalize_images: Whether to normalize images
    :type normalize_images: bool, optional
    :param placeholder_value: Value to replace NaNs with
    :type placeholder_value: Any | None, optional
    :param swap_channels: Whether to swap channels, defaults to False
    :type swap_channels: bool, optional
    :return: Preprocessed multi-discrete observation (one-hot encoded)
    :rtype: torch.Tensor
    """
    # Convert to tensor
    observation = obs_to_tensor(observation, device)

    # Replace NaNs with placeholder value if specified
    if placeholder_value is not None:
        observation = add_placeholder_value(observation, placeholder_value)

    # Need to add batch dimension prior to splitting
    observation = maybe_add_batch_dim(observation, observation_space)

    # Tensor concatenation of one hot encodings of each Categorical sub-space
    observation = torch.cat(
        [
            F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
            for idx, obs_ in enumerate(torch.split(observation.long(), 1, dim=1))
        ],
        dim=-1,
    )

    return observation.squeeze(1)  # Remove leftover dimension from torch.cat


@preprocess_observation.register(spaces.MultiBinary)
def preprocess_multibinary_observation(
    observation_space: spaces.MultiBinary,
    observation: npt.NDArray | torch.Tensor,
    device: str | torch.device = "cpu",
    normalize_images: bool = True,
    placeholder_value: float | None = None,
    swap_channels: bool = False,
) -> torch.Tensor:
    """Preprocess multi-binary observations.

    :param observation: Multi-binary observation
    :type observation: npt.NDArray | torch.Tensor
    :param observation_space: Multi-binary observation space
    :type observation_space: spaces.MultiBinary
    :param device: Computing device
    :type device: str | torch.device, optional
    :param normalize_images: Whether to normalize images
    :type normalize_images: bool, optional
    :param placeholder_value: Value to replace NaNs with
    :type placeholder_value: Any | None, optional
    :param swap_channels: Whether to swap channels, defaults to False
    :type swap_channels: bool, optional
    :return: Preprocessed multi-binary observation
    :rtype: torch.Tensor
    """
    # Convert to tensor
    observation = obs_to_tensor(observation, device)

    # Replace NaNs with placeholder value if specified
    if placeholder_value is not None:
        observation = add_placeholder_value(observation, placeholder_value)

    observation = observation.float()

    # Check add batch dimension if necessary
    return maybe_add_batch_dim(observation, observation_space)


@overload
def apply_image_normalization(
    observation: torch.Tensor,
    observation_space: spaces.Box,
) -> torch.Tensor: ...


@overload
def apply_image_normalization(
    observation: npt.NDArray,
    observation_space: spaces.Box,
) -> np.ndarray: ...


def apply_image_normalization(
    observation: ArrayOrTensor,
    observation_space: spaces.Box,
) -> ArrayOrTensor:
    """Normalize images using minmax scaling.

    :param observation: Observation
    :type observation: npt.NDArray | torch.Tensor
    :param observation_space: Observation space
    :type observation_space: spaces.Box
    :return: Observation
    :rtype: npt.NDArray | torch.Tensor
    """
    if not isinstance(observation_space, spaces.Box):
        msg = f"Expected spaces.Box, got {type(observation_space)}"
        raise TypeError(msg)

    if np.inf in observation_space.high:
        warnings.warn(
            "np.inf detected in observation_space.high, bypassing normalization.",
            stacklevel=2,
        )
        return observation

    if -np.inf in observation_space.low:
        warnings.warn(
            "-np.inf detected in observation_space.low, bypassing normalization.",
            stacklevel=2,
        )
        return observation

    if np.all(observation_space.high == 1) and np.all(observation_space.low == 0):
        return observation

    if isinstance(observation, torch.Tensor):
        low = torch.tensor(
            observation_space.low,
            device=observation.device,
            dtype=observation.dtype,
        )
        high = torch.tensor(
            observation_space.high,
            device=observation.device,
            dtype=observation.dtype,
        )
    else:
        low = observation_space.low
        high = observation_space.high

    return (observation - low) / (high - low)


# TODO: The following functions are currently used in PPO (on-policy) as a means of handling
# experiences in the absence of a rollout buffer -> This will not be needed in the future.
_ExperienceTs = TypeVarTuple("_ExperienceTs")


def get_experiences_samples(
    minibatch_indices: npt.NDArray,
    *experiences: Unpack[_ExperienceTs],
) -> tuple[Unpack[_ExperienceTs]]:
    """Sample experiences given minibatch indices.

    :param minibatch_indices: Minibatch indices
    :type minibatch_indices: npt.NDArray
    :param experiences: Experiences to sample from
    :type experiences: tuple[torch.Tensor[float], ...]

    :return: Sampled experiences
    :rtype: tuple[torch.Tensor[float], ...]
    """

    def _take(value: object) -> torch.Tensor:
        assert isinstance(value, torch.Tensor)
        return value[minibatch_indices]

    sampled_experiences: list[Any] = []
    for exp in experiences:
        sampled_exp: Any
        if isinstance(exp, torch.Tensor):
            sampled_exp = exp[minibatch_indices]
        elif isinstance(exp, dict):
            sampled_exp = {key: _take(value) for key, value in exp.items()}
        elif isinstance(exp, tuple):
            sampled_exp = tuple(_take(value) for value in exp)
        elif exp is None:
            sampled_exp = None
        else:
            msg = f"Unsupported experience type: {type(exp)}"
            raise TypeError(msg)

        sampled_experiences.append(sampled_exp)

    # The loop indexes each element in place, preserving its type, but that is
    # not statically expressible against the TypeVarTuple return.
    result: Any = tuple(sampled_experiences)
    return result


def stack_experiences(
    *experiences: MaybeObsList,
    to_torch: bool = True,
) -> tuple[ObservationType, ...]:
    """Stacks experiences into a single array or tensor.

    :param experiences: Experiences to stack
    :type experiences: list[npt.NDArray] or list[dict[str, npt.NDArray]]
    :param to_torch: If True, convert the stacked experiences to a torch tensor, defaults to True
    :type to_torch: bool, optional

    :return: Stacked experiences
    :rtype: tuple[ArrayOrTensor, ...]
    """
    stacked_experiences: list[Any] = []
    for exp in experiences:
        # Some cases where an experience just involves e.g. a single "next_state"
        stacked_exp: Any
        if not isinstance(exp, list):
            stacked_exp = exp
            if to_torch and isinstance(exp, np.ndarray):
                stacked_exp = torch.from_numpy(exp)

            stacked_experiences.append(stacked_exp)
            continue

        # The list is homogeneous, so `first`'s type applies to every element;
        # each branch re-asserts that element type since narrowing `first` does
        # not narrow `exp`.
        first = exp[0]
        if isinstance(first, dict):
            grouped: defaultdict[str, list[Any]] = defaultdict(list)
            for it in exp:
                assert is_str_keyed_dict(it)
                for key, value in it.items():
                    grouped[key].append(value)

            stacked_exp = {key: np.array(value) for key, value in grouped.items()}
            if to_torch:
                stacked_exp = {
                    key: torch.from_numpy(value) for key, value in stacked_exp.items()
                }
        elif isinstance(first, tuple):
            transposed: list[list[Any]] = [[] for _ in first]
            for it in exp:
                assert isinstance(it, tuple)
                for i, value in enumerate(it):
                    transposed[i].append(value)

            arrays = [np.array(value) for value in transposed]
            if to_torch:
                stacked_exp = tuple(torch.from_numpy(value) for value in arrays)
            else:
                stacked_exp = tuple(arrays)

        elif isinstance(first, (np.ndarray, Number)):
            stacked_array = np.stack([np.asarray(e) for e in exp])
            stacked_exp = torch.from_numpy(stacked_array) if to_torch else stacked_array

        elif isinstance(first, torch.Tensor):
            stacked_exp = torch.stack([e for e in exp if isinstance(e, torch.Tensor)])

        else:
            msg = f"Unsupported experience type: {type(first)}"
            raise TypeError(msg)

        stacked_experiences.append(stacked_exp)

    return tuple(stacked_experiences)


def stack_and_pad_experiences(
    *experiences: Sequence[ObservationType] | ObservationType,
    padding_values: list[int | float | bool | None],
    padding_side: str = "right",
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, ...]:
    """Stacks experiences into a single tensor, padding them to the maximum length.

    :param experiences: Experiences to stack: per-position tensor lists or
        already-stacked tensors.
    :type experiences: Sequence[ObservationType] | ObservationType
    :param padding_side: Side to pad on, defaults to "right"
    :type padding_side: str, optional

    :return: Stacked experiences
    :rtype: tuple[torch.Tensor, ...]
    """
    stacked_experiences: list[Any] = []
    for exp, padding in zip(experiences, padding_values, strict=False):
        stacked_exp: Any
        # Each list is homogeneous, so `exp[0]`'s type applies to every element;
        # each branch re-asserts that element type since narrowing the first
        # element does not narrow `exp`.
        if not isinstance(exp, list):
            # Pass-through experiences (e.g. an already-stacked tensor)
            stacked_exp = exp
        elif isinstance(exp[0], torch.Tensor):
            tensors = [e for e in exp if isinstance(e, torch.Tensor)]
            stacked_exp = _stack_and_pad_tensor_list(tensors, padding, padding_side)
        elif isinstance(exp[0], (list, tuple)):
            tensors = [torch.tensor(e).unsqueeze(0) for e in exp]
            stacked_exp = _stack_and_pad_tensor_list(tensors, padding, padding_side)
        else:
            msg = f"Unsupported experience type: {type(exp[0])}"
            raise TypeError(msg)
        if device is not None:
            assert isinstance(stacked_exp, torch.Tensor)
            stacked_exp = stacked_exp.to(device)
        stacked_experiences.append(stacked_exp)
    return tuple(stacked_experiences)


def _stack_and_pad_tensor_list(
    exp: list[torch.Tensor],
    padding: float | bool | None,
    padding_side: str = "right",
) -> torch.Tensor:
    """Stack and pad a list of tensors.

    :param exp: List of tensors to stack and pad
    :type exp: list[torch.Tensor]
    :param padding: Value to pad with
    :type padding: int | float | bool
    :param padding_side: Side to pad on, defaults to "right"
    :type padding_side: str, optional
    """
    max_size = max(e.shape[-1] for e in exp)
    padding_sizes = [(max_size - e.shape[-1]) for e in exp]
    if sum(padding_sizes) != 0:
        exp = [
            F.pad(
                e,
                ((0, padding_size) if padding_side == "right" else (padding_size, 0)),
                value=padding,
            )
            for e, padding_size in zip(exp, padding_sizes, strict=False)
        ]
    return torch.cat(exp, dim=0)


def flatten_experiences(*experiences: ObservationType) -> tuple[ArrayOrTensor, ...]:
    """Flattens experiences into a single array or tensor.

    :param experiences: Experiences to flatten
    :type experiences: tuple[npt.NDArray, ...] or tuple[torch.Tensor[float], ...]

    :return: Flattened experiences
    :rtype: tuple[npt.NDArray, ...] or tuple[torch.Tensor[float], ...]
    """

    def flatten(arr: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        # Need to flatten batch and n_env dimensions
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)

        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    flattened_experiences: list[Any] = []
    for exp in experiences:
        flattened_exp: Any
        if isinstance(exp, (torch.Tensor, np.ndarray)):
            flattened_exp = flatten(exp)
        elif is_str_keyed_dict(exp):
            flattened_exp = {
                key: flatten(value)
                for key, value in exp.items()
                if isinstance(value, (np.ndarray, torch.Tensor))
            }
        elif isinstance(exp, tuple):
            flattened_exp = tuple(
                flatten(value)
                for value in exp
                if isinstance(value, (np.ndarray, torch.Tensor))
            )
        else:
            msg = f"Unsupported experience type: {type(exp)}"
            raise TypeError(msg)

        flattened_experiences.append(flattened_exp)

    return tuple(flattened_experiences)


def is_vectorized_experiences(*experiences: NumpyObsType | TorchObsType) -> bool:
    """Check if experiences are vectorised.

    :param experiences: Experiences to check
    :type experiences: tuple[npt.NDArray, ...] or tuple[torch.Tensor[float], ...]

    :return: True if experiences are vectorised, False otherwise
    :rtype: bool
    """
    is_vec_ls = []
    for exp in experiences:
        if isinstance(exp, (torch.Tensor, np.ndarray)):
            is_vec = exp.ndim > 1
        elif isinstance(exp, dict):
            is_vec = all(value.ndim > 1 for value in exp.values())
        elif isinstance(exp, tuple):
            is_vec = all(value.ndim > 1 for value in exp)
        else:
            is_vec = exp.ndim > 1

        is_vec_ls.append(is_vec)

    return all(is_vec_ls)


@dataclass
class CosineLRScheduleConfig:
    """Data class to configure a cosine LR scheduler."""

    num_epochs: int
    warmup_proportion: float


@dataclass
class VLLMConfig:
    """Data class to configure a colocated vLLM instance.

    :param tensor_parallel_size: Number of GPUs for tensor parallelism, defaults to 1.
    :type tensor_parallel_size: int, optional
    :param gpu_memory_utilization: Fraction of GPU memory to reserve for vLLM KV cache,
        defaults to 0.3.
    :type gpu_memory_utilization: float, optional
    :param max_num_seqs: Maximum number of sequences processed concurrently.  For GRPO,
        set this to at least ``group_size`` to avoid request queuing, defaults to 8.
    :type max_num_seqs: int, optional
    :param max_num_batched_tokens: Cap on tokens vLLM may process in one scheduler
        iteration (prefill batching / compile profiling).  ``None`` uses
        :func:`~agilerl.utils.llm_utils.resolve_vllm_max_num_batched_tokens`
        (not ``max_num_seqs * max_model_len``, which OOMs long-context colocated
        init).  Set explicitly when you need full parallel max-length prefills.
    :type max_num_batched_tokens: int | None, optional
    :param sleep_mode: Put vLLM to sleep between ``get_action`` calls to free GPU memory
        for training.  Cannot be used with agent populations on a single device,
        defaults to False.
    :type sleep_mode: bool, optional
    :param sleep_mode_level: Sleep level passed to ``llm.sleep(level=...)`` when
        ``sleep_mode`` is enabled. ``1`` backs the base weights up to CPU and
        drops the KV cache; ``2`` discards the weights entirely, so it is only
        safe when new base weights are pushed into vLLM after every wake. The
        colocated LoRA-only sync never re-pushes the base, so it requires
        level 1. Defaults to 1.
    :type sleep_mode_level: int, optional
    :param dtype: Model weight dtype passed to the vLLM ``LLM`` constructor
        (e.g. ``"bfloat16"``, ``"float16"``).  ``None`` lets vLLM choose,
        defaults to None.
    :type dtype: str | None, optional
    :param quantization: Quantization method passed to the vLLM ``LLM`` constructor
        (e.g. ``"awq"``, ``"gptq"``).  ``None`` disables quantization, defaults to None.
    :type quantization: str | None, optional
    :param vllm_model_name_or_path: Optional HF id or path for the vLLM engine only.
        When set, the trainer may use a different ``model_name`` (e.g. bnb NF4 base)
        while rollout loads this checkpoint (e.g. an AWQ export).  ``None`` uses the
        trainer model path, defaults to None.
    :type vllm_model_name_or_path: str | None, optional
    :param kv_cache_dtype: Bare passthrough to vLLM's ``kv_cache_dtype`` kwarg
        (e.g. ``"fp8"`` on Hopper+ / Ada / Blackwell, ``"auto"``).  AgileRL does
        not validate any value — the string is forwarded verbatim and vLLM
        emits its own hardware errors / warnings.  ``None`` (the default) omits
        the kwarg so vLLM keeps its own default.  FP8 KV requires compute
        capability 8.9+; on A100 leave this unset.
    :type kv_cache_dtype: str | None, optional
    :param stop_sequences: List of strings that terminate generation early (e.g.
        ``["</answer>"]``).  Passed as ``stop`` to ``SamplingParams``, defaults to None.
    :type stop_sequences: list[str] | None, optional
    :param presence_penalty: Penalise tokens that have already appeared in the output;
        positive values discourage repetition.  Passed to ``SamplingParams``,
        defaults to 0.0 (disabled).
    :type presence_penalty: float, optional
    :param frequency_penalty: Penalise tokens proportionally to how often they have
        appeared so far.  Passed to ``SamplingParams``, defaults to 0.0 (disabled).
    :type frequency_penalty: float, optional
    :param max_lora_rank: Maximum LoRA rank passed to the vLLM ``LLM`` constructor.
        Should be at least the trainer's ``lora_config.r``.  Defaults to 16.
    :type max_lora_rank: int, optional
    :param max_loras: Maximum number of LoRA adapters vLLM can hold concurrently.
        Defaults to 1 (actor rollout only).
    :type max_loras: int, optional
    :param kv_cache_memory_bytes: Manually pin KV cache size in bytes instead of
        letting vLLM auto-size from ``gpu_memory_utilization``.  When set, vLLM
        uses this exact value for the KV cache and skips the auto-sizing path
        in ``determine_available_memory`` — but ``gpu_memory_utilization`` is
        **still honoured** by the upfront ``free_memory >= total_memory *
        gpu_memory_utilization`` startup check in
        ``vllm/v1/worker/gpu_worker.py:init_device``.  When running multiple
        vLLM processes concurrently you must keep ``gpu_memory_utilization``
        small enough that every worker's startup check passes.

        **Required for safe parallel/colocated vLLM**: vLLM's startup
        ``determine_available_memory`` profile run asserts that GPU free-memory
        does not increase between the pre- and post-profile snapshots.  When
        peer processes on the same GPU release memory mid-profile (concurrent
        xdist workers, sibling CI containers sharing one GPU), the assertion
        fires with ``Error in memory profiling. Initial free memory ... current
        free memory ...``.  Setting ``kv_cache_memory_bytes`` triggers vLLM's
        early-return path in ``determine_available_memory`` and skips that
        assertion entirely.  CI tests set this to a small value (e.g. 32 MiB)
        on the tiny test fixture; production deployments running a single
        vLLM should leave it unset.  Defaults to None.
    :type kv_cache_memory_bytes: int | None, optional
    :param strip_multimodal_towers: Free the GPU memory held by a multimodal
        base's unused towers after engine init (text-only RL). ``True`` strips
        the standard HF attribute names (``vision_tower``, ``audio_tower``,
        ``multi_modal_projector``, ``embed_vision``, ``embed_audio``); a list
        of attribute names strips those instead, for models that mount
        unwanted modalities elsewhere. Defaults to ``False``.
    :type strip_multimodal_towers: bool | list[str], optional
    :param lora_staging_dir: Root directory where the trained LoRA adapter is
        exported for vLLM to (re)load each sync. Staging is always
        process-private: in distributed runs each rank stages under a
        ``rank_<process_index>`` subdirectory of this root. Set explicitly
        when the adapter must live at a known path (e.g. orchestrated/arena
        deployments); user-supplied directories are created if missing and
        never deleted by AgileRL. ``None`` (default) uses a fresh
        process-private temporary directory, removed on ``clean_up``.
    :type lora_staging_dir: str | None, optional
    """

    # Colocate mode parameters
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.3
    max_num_seqs: int = 8
    max_num_batched_tokens: int | None = None
    enforce_eager: bool | None = None
    sleep_mode: bool = False
    sleep_mode_level: int = 1
    dtype: str | None = None
    quantization: str | None = None
    vllm_model_name_or_path: str | None = None
    kv_cache_dtype: str | None = None
    max_lora_rank: int = 16
    max_loras: int = 1
    strip_multimodal_towers: bool | list[str] = False
    stop_sequences: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # See class docstring above. Required to avoid vLLM's memory-profiling
    # assertion when running multiple vLLM processes on a shared GPU.
    kv_cache_memory_bytes: int | None = None
    lora_staging_dir: str | None = None

    def __post_init__(self) -> None:
        if self.sleep_mode_level not in (1, 2):
            msg = (
                "vllm sleep_mode_level must be either 1 or 2, got "
                f"{self.sleep_mode_level}."
            )
            raise ValueError(msg)

        # sleep_mode toggles the native vLLM sleep/wake cycle (base backed up to
        # host RAM, KV freed) between rollout and training for a single colocated
        # agent; it is not usable with a population on one device.
        if self.sleep_mode:
            warnings.warn(
                "VLLM sleep mode cannot be used with populations of agents on a "
                "single device. To use sleep mode, ensure you are training a "
                "single agent or, alternatively, use a different device for "
                "each agent.",
                stacklevel=2,
            )
            if self.gpu_memory_utilization <= 0.5:
                warnings.warn(
                    f"vLLM sleep_mode=True with gpu_memory_utilization="
                    f"{self.gpu_memory_utilization} — conservative for rollout "
                    f"after sleep, but vLLM still allocates its KV pool during "
                    f"``LLM()`` init before ``sleep()`` frees GPU memory. On "
                    f"smaller GPUs or long context, cap init with "
                    f"kv_cache_memory_bytes or a lower gpu_memory_utilization.",
                    stacklevel=2,
                )


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: CosineLRScheduleConfig,
    min_lr: float,
    max_lr: float,
) -> SequentialLR:
    """Create cosine annealing lr scheduler with warm-up.

    :param optimizer: Optimizer
    :type optimizer: torch.optim.Optimizer
    :param config: LR scheduler config
    :type config: CosineLRScheduleConfig
    :param min_lr: Minimum learning rate
    :type min_lr: float
    :param max_lr: Maximum learning rate
    :type max_lr: float
    :return: Return sequential learning rate scheduler
    :rtype: SequentialLR
    """
    num_epochs = config.num_epochs
    warmup_proportion = config.warmup_proportion
    warmup_epochs = int(num_epochs * warmup_proportion)
    remaining_epochs = num_epochs - warmup_epochs
    for param_group in optimizer.param_groups:
        param_group["lr"] = max_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=min_lr / max_lr,  # Start factor to get from min_lr to max_lr
        end_factor=1.0,  # End with the full max_lr
        total_iters=warmup_epochs,
    )
    # Decay scheduler: Cosine decay from max_lr to min_lr
    # Double T_max to ensure we only use the first half of the cosine curve (strictly decreasing)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=remaining_epochs * 2,  # Doubled to ensure strictly decreasing LR
        eta_min=min_lr,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def remove_nested_files(files: list[str]) -> None:
    """Remove nested files from a list of files.

    :param files: List of files to remove nested files from
    :type files: list[str]
    :param depth: Depth of the nested files, defaults to 0
    :type depth: int, optional
    """
    for f in files:
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)


def vectorize_experiences_by_agent(
    experiences: dict[str, Any],
    dim: int = 1,
) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...]:
    """Reorganizes experiences into a tensor, vectorized by time step.

    Example input:
    {'agent_0': [[1, 2, 3, 4]], 'agent_1': [[5, 6, 7, 8]]}
    Example output:
    torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    :param experiences: Dictionaries containing experiences indexed by agent_id that share a policy agent.
    :type experiences: dict[str, ObservationType]
    :param dim: New dimension to stack along
    :type dim: int
    :return: Tensor, dict of tensors, or tuple of tensors of experiences, stacked along provided dimension
    :rtype: torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...]
    """
    if not experiences:
        return torch.tensor([])

    # Get a sample value to determine the type
    sample_value = next(iter(experiences.values()))

    if isinstance(sample_value, dict):
        # Handle dictionary observations
        keys = sample_value.keys()
        vectorized_dict: dict[str, Any] = {
            k: vectorize_experiences_by_agent(
                {agent_id: experiences[agent_id][k] for agent_id in experiences},
                dim=dim,
            )
            for k in keys
        }
        return vectorized_dict
    if isinstance(sample_value, tuple):
        # Handle tuple observations
        tuple_length = len(sample_value)
        vectorized_tuple: tuple[Any, ...] = tuple(
            vectorize_experiences_by_agent(
                {agent_id: experiences[agent_id][i] for agent_id in experiences},
                dim=dim,
            )
            for i in range(tuple_length)
        )
        return vectorized_tuple
    # Original implementation for array/tensor observations
    tensors: list[torch.Tensor] = []
    for experience in experiences.values():
        if experience is None:
            continue
        tensors.append(torch.Tensor(np.array(experience)))

    # Check if all tensors have the same shape
    if all(t.shape == tensors[0].shape for t in tensors):
        stacked_tensor = torch.stack(tensors, dim=dim)
    else:
        # Concatenate along the specified dimension
        stacked_tensor = torch.cat(tensors)

    return stacked_tensor


def vectorize_agent_experiences_flat(
    experiences: dict[str, Any],
    dim: int = 1,
) -> torch.Tensor:
    """Vectorize flat per-agent experiences (log-probs, rewards, dones, values).

    :param experiences: Per-agent experiences indexed by agent id.
    :type experiences: dict[str, Any]
    :param dim: Dimension to stack along.
    :type dim: int
    :return: The stacked experiences as a single tensor.
    :rtype: torch.Tensor
    :raises TypeError: If the experiences vectorize to a structured container
        rather than a flat tensor.
    """
    vectorized = vectorize_experiences_by_agent(experiences, dim=dim)
    if not isinstance(vectorized, torch.Tensor):
        msg = (
            "vectorize_agent_experiences_flat expects flat per-agent scalars, "
            "but the experiences vectorized to a structured observation container."
        )
        raise TypeError(msg)
    return vectorized


def experience_to_tensors(
    experience: Any,  # noqa: ANN401 -- nested heterogeneous experience (dict/tuple/array-like) forwarded to np.array
    space: spaces.Space,
    actions: bool = False,
) -> TorchObsType:
    """Convert experience to tensors matching the structure of the given space.

    :param experience: Experience to convert (dict, tuple, or array-like)
    :type experience: dict[str, Any] | tuple[Any, ...] | npt.NDArray | Any
    :param space: Space to convert experience to
    :type space: spaces.Space
    :param actions: Whether the experience is an action, defaults to False
    :type actions: bool, optional
    :return: Tensor(s) of the experience
    :rtype: TorchObsType
    """
    if isinstance(experience, dict):
        assert isinstance(space, spaces.Dict), (
            f"Expected Dict space for dict experience, got {type(space)}"
        )
        tensor_dict: dict[str, Any] = {
            key: experience_to_tensors(value, space[key])
            for key, value in experience.items()
        }
        return tensor_dict
    if isinstance(experience, tuple):
        assert isinstance(space, spaces.Tuple), (
            f"Expected Tuple space for tuple experience, got {type(space)}"
        )
        tensor_tuple: tuple[Any, ...] = tuple(
            experience_to_tensors(exp, space[i]) for i, exp in enumerate(experience)
        )
        return tensor_tuple
    array = np.array(experience)

    # Ensure experience has a batch dimension
    array = maybe_add_batch_dim(array, space, actions)
    return torch.from_numpy(array)


def concatenate_tensors(tensors: list[TorchObsType]) -> TorchObsType:
    """Concatenate tensors along first dimension.

    :param tensors: List of tensors to concatenate
    :type tensors: list[TorchObsType]
    :return: Concatenated tensor
    :rtype: TorchObsType
    """
    first = tensors[0]
    if isinstance(first, dict):
        # Homogeneous by construction: all entries share the first entry's structure
        concat_dict: dict[str, Any] = {}
        for key in first:
            column: list[TorchObsType] = []
            for t in tensors:
                assert isinstance(t, dict)
                assert not isinstance(t, torch.Tensor)
                value = t[key]
                assert isinstance(value, torch.Tensor)
                column.append(value)
            concat_dict[key] = concatenate_tensors(column)
        return concat_dict
    if isinstance(first, tuple):
        concat_tuple: tuple[Any, ...] = tuple(
            _concatenate_tuple_column(tensors, i) for i in range(len(first))
        )
        return concat_tuple
    tensor_list: list[torch.Tensor] = []
    for t in tensors:
        assert isinstance(t, torch.Tensor)
        tensor_list.append(t)
    return torch.cat(tensor_list, dim=0)


def _concatenate_tuple_column(tensors: list[TorchObsType], i: int) -> TorchObsType:
    """Concatenate the i-th positional entry across a list of tuple observations."""
    column: list[TorchObsType] = []
    for t in tensors:
        assert isinstance(t, tuple)
        value = t[i]
        assert isinstance(value, torch.Tensor)
        column.append(value)
    return concatenate_tensors(column)


def reshape_from_space(tensor: TorchObsType, space: spaces.Space) -> TorchObsType:
    """Reshape tensor from space.

    :param tensor: Tensor to reshape
    :type tensor: TorchObsType
    :param space: Space to reshape tensor to
    :type space: spaces.Space
    :return: Reshaped tensor
    :rtype: TorchObsType
    """
    if isinstance(tensor, (torch.Tensor, TensorDict)):
        space_shape = space.shape
        assert space_shape is not None, (
            f"{type(space)} spaces have no shape to reshape to."
        )
        reshaped = tensor.reshape(-1, *space_shape)
        for squeeze_dim in [0, -1]:
            if reshaped.size(squeeze_dim) == 1:
                reshaped = reshaped.squeeze(squeeze_dim)

        return reshaped
    if isinstance(tensor, dict):
        assert isinstance(space, spaces.Dict), (
            f"Expected Dict space for dict tensor, got {type(space)}"
        )
        reshaped_dict: dict[str, Any] = {
            key: reshape_from_space(value, space[key]) for key, value in tensor.items()
        }
        return reshaped_dict
    if isinstance(tensor, tuple):
        assert isinstance(space, spaces.Tuple), (
            f"Expected Tuple space for tuple tensor, got {type(space)}"
        )
        reshaped_tuple: tuple[Any, ...] = tuple(
            reshape_from_space(value, space[i]) for i, value in enumerate(tensor)
        )
        return reshaped_tuple

    msg = f"Unsupported tensor type: {type(tensor)}"
    raise TypeError(msg)


def concatenate_experiences_into_batches(
    experiences: dict[str, Any],
    space: spaces.Space,
    actions: bool = False,
) -> TorchObsType:
    """Reorganizes experiences into a batched tensor.

    Example input:
    {'agent_0': [[[...1], [...2]], [[...5], [...6]]],
        'agent_1': [[[...3], [...4]], [[...7], [...8]]]}

    Example output:
    torch.Tensor([...1], [...2], [...3], [...4], [...5], [...6], [...7], [...8])

    :param experiences: Dictionaries containing experiences indexed by agent_id that share a policy agent.
    :type experiences: dict[str, ObservationType]
    :param space: Observation/action/etc space to maintain
    :type space: spaces.Space
    :param actions: Whether the experiences are actions, defaults to False
    :type actions: bool, optional
    :return: Tensor, dict of tensors, or tuple of tensors of experiences, stacked along first dimension, with shape (num_experiences, *shape)
    :rtype: torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...]
    """
    tensors: list[TorchObsType] = []
    for agent_id in experiences:
        exp = experience_to_tensors(experiences[agent_id], space, actions)
        tensors.append(exp)

    stacked_tensor = concatenate_tensors(tensors)
    return reshape_from_space(stacked_tensor, space)


def is_peft_model(model: nn.Module) -> bool:
    """Check if a model is a PEFT model.

    :param model: Model to check
    :type model: nn.Module
    :return: True if the model is a PEFT model, False otherwise
    :rtype: bool
    """
    return isinstance(model, PeftModel)


def _rename_peft_primary_adapter_keys_in_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    old_adapter: str,
    new_adapter: str,
) -> dict[str, torch.Tensor]:
    """Rewrite state-dict keys when the primary PEFT adapter is renamed (e.g. to ``actor``)."""
    if old_adapter == new_adapter:
        return state_dict
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k.replace(f".{old_adapter}.", f".{new_adapter}.")
        nk = nk.replace(f"lora_{old_adapter}", f"lora_{new_adapter}")
        out[nk] = v
    return out


def clone_llm(
    original_model: PreTrainedModelType | DummyEvolvable,
    zero_stage: int | None,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> PreTrainedModelType:
    """Clone the actor.

    :param original_model: Model to clone
    :type original_model: PreTrainedModelType
    :param zero_stage: Zero stage to use, defaults to 0
    :type zero_stage: int | None, optional
    :param state_dict: State dict to load, defaults to None
    :type state_dict: dict[str, torch.Tensor] | None, optional
    :return: Cloned model
    """
    match original_model:
        case PeftModel() | PreTrainedModel():
            source_model = original_model
        case DummyEvolvable():
            # DummyEvolvable wraps an arbitrary module; the RL-clone path is only
            # reached with a pretrained model inside it.
            inner_model = original_model.module
            assert isinstance(inner_model, (PeftModel, PreTrainedModel))
            source_model = inner_model
        case _:
            msg = f"Invalid 'original_model' type: {type(original_model)}"
            raise ValueError(msg)
    model_config = source_model.config
    base_model = source_model.model
    assert isinstance(base_model, nn.Module)
    model: nn.Module = type(base_model)(model_config)
    adapter_names: list[str] = []

    # Any model carrying peft_config has adapters to copy, including
    # wrappers that are not PeftModel subclasses. The attribute is dynamic,
    # so pin the adapter-name/config pairs to their concrete peft types.
    if hasattr(source_model, "peft_config"):
        raw_peft_config = source_model.peft_config
        assert is_str_keyed_dict(raw_peft_config)
        peft_configs: dict[str, PeftConfig] = {
            name: config
            for name, config in raw_peft_config.items()
            if isinstance(config, PeftConfig)
        }
        adapter_names = list(peft_configs.keys())

        if len(adapter_names) > 1:
            warnings.warn(
                "Multiple adapters detected. Only the first adapter will be used for RL finetuning.",
                stacklevel=2,
            )
        # AgileRL standardizes on adapter name "actor" for the primary adapter.
        first_adapter = adapter_names[0]
        keep_adapter_base_dtype = zero_stage == 3
        model = get_peft_model(
            model,
            peft_configs[first_adapter],
            adapter_name="actor",
            autocast_adapter_dtype=not keep_adapter_base_dtype,
        )

        # Add remaining adapters using add_adapter
        for adapter_name in adapter_names[1:]:
            model.add_adapter(
                peft_config=peft_configs[adapter_name],
                adapter_name=adapter_name,
                autocast_adapter_dtype=not keep_adapter_base_dtype,
            )
        if keep_adapter_base_dtype:
            for name, param in model.named_parameters():
                if "lora" in name and param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)
        model.disable_adapter()

    if state_dict is not None:
        sd = state_dict
        if adapter_names and adapter_names[0] != "actor":
            sd = _rename_peft_primary_adapter_keys_in_state_dict(
                sd,
                old_adapter=adapter_names[0],
                new_adapter="actor",
            )
        model.load_state_dict(sd, strict=False)
    return model


class DummyOptimizer:
    """Placeholder optimizer class to pass to the OptimizerWrapper when the optimizer is defined in the deepspeed config."""

    def __init__(self, params: list[torch.Tensor], **kwargs: Any) -> None:
        """Sentinel class to use for the optimizer when the optimizer is defined in the deepspeed config.

        :param params: Parameters to optimize.
        :type params: list[torch.Tensor]
        """

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> NoReturn:
        msg = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        raise RuntimeError(
            msg,
        )

    def zero_grad(self) -> NoReturn:
        msg = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        raise RuntimeError(
            msg,
        )

    def state_dict(self) -> NoReturn:
        msg = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        raise RuntimeError(
            msg,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> NoReturn:
        msg = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        raise RuntimeError(
            msg,
        )


def _match_action_ndims(
    reference: npt.NDArray, other: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Prepend singleton axes until continuous action arrays share the same ndim."""
    while other.ndim < reference.ndim:
        other = np.expand_dims(other, 0)
    while reference.ndim < other.ndim:
        reference = np.expand_dims(reference, 0)
    return reference, other


def _reconcile_shapes(
    reference: npt.NDArray, other: npt.NDArray, discrete_actions: bool
) -> tuple[npt.NDArray, npt.NDArray]:
    """Squeeze and broadcast `other` to match `reference` shape where possible.

    :param reference: Reference array to match shape to.
    :type reference: npt.NDArray
    :param other: Array to reconcile shape of.
    :type other: npt.NDArray
    :param discrete_actions: Whether the actions are discrete, defaults to False
    :type discrete_actions: bool, optional
    :return: Tuple of reconciled arrays.
    :rtype: tuple[npt.NDArray, npt.NDArray]
    """
    if reference.shape == other.shape:
        return reference, other

    if np.prod(other.shape) == np.prod(reference.shape):
        if discrete_actions:
            if other.ndim < reference.ndim:
                reference = reference.squeeze()
            else:
                other = other.squeeze()
        else:
            reference, other = _match_action_ndims(reference, other)

    return reference, np.broadcast_to(other, reference.shape)


def apply_env_defined_actions(
    agent_ids: list[str],
    action_dict: dict[str, npt.NDArray],
    env_defined_actions: dict[str, npt.NDArray],
    agent_masks: dict[str, npt.NDArray],
    discrete_actions: bool,
) -> dict[str, npt.NDArray]:
    """Apply env-defined actions to agent actions where the agent mask is True.

    :param agent_ids: Agent identifiers to process.
    :type agent_ids: list[str]
    :param action_dict: Mutable mapping of agent id → action array.
    :type action_dict: dict[str, npt.NDArray]
    :param env_defined_actions: Mapping of agent id → override action array.
    :type env_defined_actions: dict[str, npt.NDArray]
    :param agent_masks: Mapping of agent id → boolean mask array.
    :type agent_masks: dict[str, npt.NDArray]
    :param discrete_actions: Whether the actions are discrete, defaults to False
    :type discrete_actions: bool, optional
    :return: `action_dict` with overrides applied in-place.
    :rtype: dict[str, npt.NDArray]
    """
    for agent_id in agent_ids:
        action = action_dict[agent_id]
        override = env_defined_actions[agent_id]
        mask = agent_masks[agent_id]
        action, override = _reconcile_shapes(action, override, discrete_actions)
        action, mask = _reconcile_shapes(action, mask, discrete_actions)
        action[mask] = override[mask]
        action_dict[agent_id] = action
    return action_dict


def _resolve_lr(
    agent: "EvolvableAlgorithm", lr: str | tuple[str, str]
) -> tuple[Any, Any | None]:
    """Resolve the learning-rate value(s) from the agent attribute name(s).

    :param lr: Learning-rate attribute name, or a (actor, critic) pair of names
    :type lr: str | tuple[str, str]
    :return: Learning-rate value(s); the second element is None for a single name
    :rtype: tuple[Any, Any | None]
    """
    if isinstance(lr, tuple):
        return getattr(agent, lr[0]), getattr(agent, lr[1])
    return getattr(agent, lr), None


def inherit_init_signature(
    parent: type, fixed: set[str] | None = None
) -> Callable[[type], type]:
    """Class decorator giving a subclass its ``parent``'s ``__init__`` signature.

    A subclass that pins some of its parent's constructor arguments via
    ``*args``/``**kwargs`` loses its introspectable signature
    (``inspect.signature`` would just report ``(self, *args, **kwargs)``).
    AgileRL reads ``inspect.signature(agent.__init__).parameters`` to build the
    clone/checkpoint ``init_dict`` (see :class:`EvolvableAlgorithm`), so this
    restores the parent's real parameters — minus the ones the subclass fixes —
    on both the class and its ``__init__``.

    :param parent: Parent class whose ``__init__`` signature to inherit.
    :type parent: type
    :param fixed: Parameter names the subclass pins internally and therefore must
        not accept (excluded from the inherited signature), defaults to ``None``.
    :type fixed: set[str] | None, optional
    :return: A class decorator.
    :rtype: Callable[[type], type]
    """
    fixed = fixed or set()
    parent_sig = inspect.signature(parent.__init__)
    kept = [p for p in parent_sig.parameters.values() if p.name not in fixed]

    def decorate(cls: type) -> type:
        if "__init__" not in cls.__dict__:
            msg = (
                f"{cls.__name__} must define its own __init__ before "
                "@inherit_init_signature (otherwise the parent's signature is "
                "mutated)."
            )
            raise TypeError(msg)
        # inspect.signature(cls) is the constructor *call* — drop ``self``.
        cls.__signature__ = parent_sig.replace(  # ty: ignore[unresolved-attribute]  # metaprogramming: __signature__ is set dynamically on the class; typeshed does not model it
            parameters=[p for p in kept if p.name != "self"]
        )
        # inspect.signature(cls.__init__) is the *method* — keep ``self``. This
        # is the one EvolvableAlgorithm reads to build the clone/checkpoint dict.
        cls.__init__.__signature__ = parent_sig.replace(parameters=kept)  # ty: ignore[unresolved-attribute]  # metaprogramming: __signature__ is set dynamically on the function object; typeshed does not model it
        return cls

    return decorate
