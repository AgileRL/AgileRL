# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeGuard,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
import torch
from accelerate import Accelerator
from gymnasium import spaces
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
from torch import nn
from torch._dynamo import OptimizedModule
from torch.optim import Optimizer

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.modules.base import EvolvableModule, ModuleDict
from agilerl.modules.custom_components import NoisyLinear
from agilerl.protocols import (
    EvolvableAttributeType,
    EvolvableModuleProtocol,
    EvolvableNetworkProtocol,
)
from agilerl.typing import (
    BatchDimension,
    BPTTSequenceType,
    DeviceType,
    InputSizeFromSpace,
    LeafSpace,
    NetConfigType,
    NumpyObsType,
    ObsShape,
    OutputSizeFromSpace,
    SpaceLike,
    TensorMapping,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import PeftModel
    from transformers import PreTrainedModel

    PreTrainedModelType = PeftModel | PreTrainedModel
else:
    # Annotations referencing PreTrainedModelType are evaluated at function
    # definition time, so provide a runtime placeholder when the LLM
    # dependencies are not installed.
    PeftModel = None
    PreTrainedModel = None
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
    # circular import with agilerl.networks
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
    # circular import with agilerl.algorithms.core.registry
    from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper

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
        # Require image spaces to have the same shape for concatenation
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

