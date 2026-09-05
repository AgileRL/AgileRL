# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from functools import singledispatch
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from gymnasium import spaces
from tensordict import TensorDict
from typing_extensions import TypeVarTuple, Unpack

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.typing import (
    ArrayOrTensor,
    LeafSpace,
    MaybeObsList,
    NumpyObsType,
    ObservationType,
    TensorMapping,
    TensorTuple,
    TorchObsType,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import PeftModel
    from transformers import PreTrainedModel

    PreTrainedModelType = PeftModel | PreTrainedModel
else:
    # Annotations referencing PreTrainedModelType are evaluated at function
    # definition time, so provide a runtime placeholder when the LLM
    # dependencies are not installed.
    PreTrainedModelType = Any


from agilerl.utils.algo_spaces import (
    get_input_size_from_space,
    get_num_actions,
    is_str_keyed_dict,
    transpose_image_observation,
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

