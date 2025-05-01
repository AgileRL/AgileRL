import inspect
import os
import shutil
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import singledispatch
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeGuard, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from gymnasium import spaces
from peft import PeftModel, get_peft_model
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
from torch._dynamo import OptimizedModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import PreTrainedModel

from agilerl.protocols import (
    EvolvableAttributeType,
    EvolvableModule,
    EvolvableNetwork,
    OptimizerWrapper,
)
from agilerl.typing import (
    ArrayOrTensor,
    ExperiencesType,
    MaybeObsList,
    NetworkType,
    NumpyObsType,
    ObservationType,
    OptimizerType,
    TorchObsType,
)

PreTrainedModelType = Union[PeftModel, PreTrainedModel]


def share_encoder_parameters(
    policy: EvolvableNetwork, *others: EvolvableNetwork
) -> None:
    """Shares the encoder parameters between the policy and any number of other networks.

    :param policy: The policy network whose encoder parameters will be used.
    :type policy: EvolvableNetwork
    :param others: The other networks whose encoder parameters will be pinned to the policy.
    :type others: EvolvableNetwork
    """
    assert isinstance(policy, EvolvableNetwork), "Policy must be an EvolvableNetwork"
    assert all(
        isinstance(other, EvolvableNetwork) for other in others
    ), "All others must be EvolvableNetwork"

    # detaching encoder parameters from computation graph reduces
    # memory overhead and speeds up training
    param_vals: TensorDict = from_module(policy.encoder).detach()
    for other in others:
        target_params: TensorDict = param_vals.clone().lock_()
        target_params.to_module(other.encoder)

        # Disable architecture mutations since we will be
        # reinitializing directly through a mutation hook
        other.encoder.disable_mutations()


def is_image_space(space: spaces.Space) -> bool:
    """Check if the space is an image space. We ignore dtype and number of channels
    checks.

    :param space: Input space
    :type space: spaces.Space

    :return: True if the space is an image space, False otherwise
    :rtype: bool
    """
    return isinstance(space, spaces.Box) and len(space.shape) == 3


def contains_image_space(space: spaces.Space) -> bool:
    """Checks if the space contains an image space.

    :param space: Observation space
    :type space: spaces.Space
    :return: True if the space contains an image space, False otherwise
    :rtype: bool
    """
    if isinstance(space, spaces.Dict):
        return any(contains_image_space(subspace) for subspace in space.spaces.values())
    elif isinstance(space, spaces.Tuple):
        return any(contains_image_space(subspace) for subspace in space.spaces)
    elif isinstance(space, spaces.Box):
        return is_image_space(space)
    return False


def multi_agent_sample_tensor_from_space(
    space: spaces.Space,
    n_agents: int,
    critic: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Gets the sample tensor from an observation space for multi-agent settings.

    :param space: Observation space
    :type space: spaces.Space
    :param n_agents: Number of agents
    :type n_agents: int
    :param critic: If True, the tensor is for the critic, defaults to False
    :type critic: bool, optional

    :return: Sample tensor
    :rtype: torch.Tensor or dict[str, torch.Tensor]
    """

    def sample(image_shape: Tuple[int, ...], critic: bool) -> torch.Tensor:
        tensor = torch.zeros(
            (1, *image_shape), dtype=torch.float32, device=device
        ).unsqueeze(2)

        if critic:
            tensor = tensor.repeat(1, 1, n_agents, 1, 1)

        return tensor

    if isinstance(space, spaces.Dict):
        sample_tensor = {
            key: multi_agent_sample_tensor_from_space(
                subspace, n_agents, critic, device
            )
            for key, subspace in space.spaces.items()
            if is_image_space(subspace)
        }
    elif isinstance(space, spaces.Tuple):
        sample_tensor = tuple(
            (
                multi_agent_sample_tensor_from_space(subspace, n_agents, critic, device)
                if is_image_space(subspace)
                else None
            )
            for subspace in space.spaces
        )
    elif is_image_space(space):
        sample_tensor = sample(space.shape, critic)
    else:
        sample_tensor = None

    return sample_tensor


def make_safe_deepcopies(
    *args: Union[EvolvableModule, List[EvolvableModule]]
) -> List[EvolvableModule]:
    """Makes deep copies of EvolvableModule objects and their attributes.

    :param args: EvolvableModule or lists of EvolvableModule objects to copy.
    :type args: Union[EvolvableModule, List[EvolvableModule]].

    :return: Deep copies of the EvolvableModule objects and their attributes.
    :rtype: List[EvolvableModule].
    """
    copies = []
    for arg in args:
        if isinstance(arg, list):
            arg_copy = [inner_arg.clone() for inner_arg in arg]
        else:
            arg_copy = arg.clone()

        copies.append(arg_copy)

    return copies[0] if len(copies) == 1 else copies


def is_module_list(obj: EvolvableAttributeType) -> TypeGuard[Iterable[EvolvableModule]]:
    """Type guard to check if an object is a list of EvolvableModule objects.

    :param obj: The object to check.
    :type obj: EvolvableAttributeType.

    :return: True if the object is a list of EvolvableModule objects, False otherwise.
    :rtype: bool.
    """
    if not isinstance(obj, list):
        return False

    return all(
        isinstance(inner_obj, (OptimizedModule, EvolvableModule)) for inner_obj in obj
    )


def is_optimizer_list(obj: EvolvableAttributeType) -> TypeGuard[Iterable[Optimizer]]:
    """Type guard to check if an object is a list of Optimizer's.

    :param obj: The object to check.
    :type obj: EvolvableAttributeType.

    :return: True if the object is a list of Optimizer's, False otherwise.
    :rtype: bool.
    """
    return all(isinstance(inner_obj, Optimizer) for inner_obj in obj)


def assert_supported_space(space: spaces.Space) -> bool:
    """Checks if the space is supported by the AgileRL framework.

    :param space: The space to check.
    :type space: spaces.Space.

    :return: True if the space is supported, False otherwise.
    :rtype: bool.
    """
    # Nested Dict or Tuple spaces are not supported
    if isinstance(space, spaces.Dict) and any(
        isinstance(subspace, (spaces.Dict, spaces.Tuple))
        for subspace in space.spaces.values()
    ):
        raise TypeError(f"Nested {type(space)} spaces are not supported.")
    elif isinstance(space, spaces.Tuple) and any(
        isinstance(subspace, (spaces.Dict, spaces.Tuple)) for subspace in space.spaces
    ):
        raise TypeError(f"Nested {type(space)} spaces are not supported.")


def isroutine(obj: object) -> bool:
    """Checks if an attribute is a routine, considering also methods wrapped by
    CudaGraphModule.

    :param attr: The attribute to check.
    :type attr: str

    :return: True if the attribute is a routine, False otherwise.
    :rtype: bool
    """
    if isinstance(obj, CudaGraphModule):
        return True

    return inspect.isroutine(obj)


def unwrap_optimizer(
    optimizer: OptimizerType, network: NetworkType, lr: float
) -> Optimizer:
    """Unwraps AcceleratedOptimizer to get the underlying optimizer.

    :param optimizer: AcceleratedOptimizer
    :type optimizer: AcceleratedOptimizer
    :param network: Network or list of networks
    :type network: Union[Module, List[Module], Tuple[Module, ...]]
    :param lr: Learning rate
    :type lr: float
    :return: Unwrapped optimizer
    rtype: Optimizer
    """
    if isinstance(optimizer, AcceleratedOptimizer):
        if isinstance(network, (list, tuple)):
            optim_arg = [{"params": net.parameters(), "lr": lr} for net in network]
            unwrapped_optimizer: Optimizer = type(optimizer.optimizer)(optim_arg)
        else:
            unwrapped_optimizer: Optimizer = type(optimizer.optimizer)(
                network.parameters(), lr=lr
            )
        unwrapped_optimizer.load_state_dict(optimizer.state_dict())
        return unwrapped_optimizer
    else:
        return optimizer


def recursive_check_module_attrs(obj: Any, networks_only: bool = False) -> bool:
    """Recursively check if the object has any attributes that are EvolvableModule objects or Optimizer's.

    :param obj: The object to check for EvolvableModule objects or Optimizer's.
    :type obj: Any
    :return: True if the object has any attributes that are EvolvableModule objects or Optimizer's, False otherwise.
    :rtype: bool
    :param networks_only: If True, only check for EvolvableModule objects, defaults to False
    :type networks_only: bool, optional
    """
    check_types = (OptimizedModule, EvolvableModule)
    if not networks_only:
        check_types += (OptimizerWrapper, DeepSpeedOptimizerWrapper)

    if isinstance(obj, check_types):
        return True
    elif isinstance(obj, Optimizer):
        raise TypeError("Optimizer objects should be wrapped by OptimizerWrapper.")
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


def chkpt_attribute_to_device(
    chkpt_dict: Dict[str, torch.Tensor], device: str
) -> Dict[str, Any]:
    """Place checkpoint attributes on device. Used when loading saved agents.

    :param chkpt_dict: Checkpoint dictionary
    :type chkpt_dict: dict
    :param device: Device for accelerated computing, 'cpu' or 'cuda'
    :type device: str
    """
    if isinstance(chkpt_dict, list):
        return [chkpt_attribute_to_device(chkpt, device) for chkpt in chkpt_dict]

    assert isinstance(chkpt_dict, dict), f"Expected dict, got {type(chkpt_dict)}"

    for key, value in chkpt_dict.items():
        if isinstance(value, torch.Tensor):
            chkpt_dict[key] = value.to(device)

    return chkpt_dict


def key_in_nested_dict(nested_dict: Dict[str, Any], target: str) -> bool:
    """Helper function to determine if key is in nested dictionary

    :param nested_dict: Nested dictionary
    :type nested_dict: Dict[str, Dict[str, ...]]
    :param target: Target string
    :type target: str
    """
    for k, v in nested_dict.items():
        if k == target:
            return True
        if isinstance(v, dict):
            return key_in_nested_dict(v, target)
    return False


def compile_model(model: Module, mode: Optional[str] = "default") -> Module:
    """Compiles torch model if not already compiled

    :param model: torch model
    :type model: nn.Module
    :param mode: torch compile mode, defaults to "default"
    :type mode: str, optional
    :return: compiled model
    :rtype: OptimizedModule
    """
    return (
        torch.compile(model, mode=mode)
        if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
        and mode is not None
        else model
    )


def remove_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Removes _orig_mod prefix on state dict created by torch compile

    :param state_dict: model state dict
    :type state_dict: dict
    :return: state dict with prefix removed
    :rtype: dict
    """
    return OrderedDict(
        [
            (k.split(".", 1)[1], v) if k.startswith("_orig_mod") else (k, v)
            for k, v in state_dict.items()
        ]
    )


SupportedSpace = Union[
    spaces.Box, spaces.Dict, spaces.Tuple, spaces.Discrete, spaces.MultiDiscrete
]


def concatenate_spaces(space_list: List[SupportedSpace]) -> spaces.Space:
    """Concatenates a list of spaces into a single space. If spaces correspond to images,
    we check that their shapes are the same and use the first space's shape as the shape of the
    concatenated space.

    :param spaces: List of spaces to concatenate
    :type spaces: List[spaces.Space]
    :return: Concatenated space
    :rtype: spaces.Space
    """
    if all(isinstance(space, spaces.Dict) for space in space_list):
        return spaces.Dict(
            {
                key: concatenate_spaces([space[key] for space in space_list])
                for key in space_list[0].spaces.keys()
            }
        )

    elif all(isinstance(space, spaces.Tuple) for space in space_list):
        return spaces.Tuple(
            [
                concatenate_spaces([space[i] for space in space_list])
                for i in range(len(space_list[0]))
            ]
        )

    elif all(isinstance(space, spaces.Box) for space in space_list):
        # NOTE: For image spaces the concatenation is handled under-the-hood through the
        # specification of `n_agents` in EvolvableNetwork objects, whereby 3d convolutions
        # are used. This is why we enforce all image spaces to have the same shape.
        if all(is_image_space(space) for space in space_list):
            assert all(space.shape == space_list[0].shape for space in space_list), (
                "AgileRL only supports multi-agent settings with the same shape for the image "
                "spaces of different agents."
            )

            return space_list[0]

        low = np.concatenate([space.low for space in space_list], axis=0)
        high = np.concatenate([space.high for space in space_list], axis=0)
        return spaces.Box(low=low, high=high, dtype=space_list[0].dtype)

    elif all(isinstance(space, spaces.Discrete) for space in space_list):
        n = sum(space.n for space in space_list)
        return spaces.Discrete(n)

    elif all(isinstance(space, spaces.MultiDiscrete) for space in space_list):
        nvec = np.concatenate([space.nvec for space in space_list], axis=0)
        return spaces.MultiDiscrete(nvec)

    else:
        raise TypeError(
            f"Unsupported space types: {set(type(space) for space in spaces)}"
        )


def obs_channels_to_first(
    observation: NumpyObsType, expand_dims: bool = False
) -> NumpyObsType:
    """Converts observation space from channels last to channels first format.

    :param observation_space: Observation space
    :type observation_space: Union[spaces.Box, spaces.Dict]
    :param expand_dims: If True, expand the dimensions of the observation, defaults to False
    :type expand_dims: bool, optional
    :return: Observation space with channels first format
    :rtype: Union[spaces.Box, spaces.Dict]
    """
    if isinstance(observation, np.ndarray):
        if expand_dims:
            observation = np.expand_dims(observation, axis=0)

        if observation.ndim == 3 or observation.ndim == 4:
            return np.moveaxis(observation, -1, -3)
        else:
            return observation

    elif isinstance(observation, dict):
        return {key: obs_channels_to_first(obs) for key, obs in observation.items()}
    else:
        raise TypeError(f"Expected np.ndarray or dict, got {type(observation)}")


def obs_to_tensor(
    obs: ObservationType, device: Union[str, torch.device]
) -> TorchObsType:
    """
    Moves the observation to the given device as a PyTorch tensor.

    :param obs:
    :type obs: NumpyObsType
    :param device: PyTorch device
    :type device: Union[str, torch.device]
    :return: PyTorch tensor of the observation on a desired device.
    :rtype: TorchObsType
    """
    if isinstance(obs, TensorDict):
        return obs if obs.device == device else obs.to(device)
    elif isinstance(obs, torch.Tensor):
        return obs.float().to(device)
    elif isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device).float()
    elif isinstance(obs, dict):
        return {
            key: torch.as_tensor(_obs, device=device).float()
            for (key, _obs) in obs.items()
        }
    elif isinstance(obs, tuple):
        return tuple(torch.as_tensor(_obs, device=device).float() for _obs in obs)
    elif isinstance(obs, (list, Number)):
        return torch.tensor(obs, device=device).float()
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


def maybe_add_batch_dim(
    obs: ObservationType, space_shape: Tuple[int, ...]
) -> ObservationType:
    """Adds batch dimension if necessary.

    :param obs: Observation tensor
    :type obs: ObservationType
    :param space_shape: Observation space shape
    :type space_shape: Tuple[int, ...]
    :return: Observation tensor with batch dimension
    :rtype: ObservationType
    """
    if len(obs.shape) == len(space_shape):
        if isinstance(obs, np.ndarray):
            obs = np.expand_dims(obs, 0)
        else:
            obs = obs.unsqueeze(0)
    elif len(obs.shape) == len(space_shape) + 2:
        if isinstance(obs, np.ndarray):
            obs = obs.reshape(-1, *space_shape)
        else:
            obs = obs.view(-1, *space_shape)
    elif len(obs.shape) != len(space_shape) + 1:
        raise ValueError(
            f"Expected observation to have {len(space_shape) + 1} dimensions, got {len(obs.shape)}."
        )

    return obs


def get_vect_dim(observation: NumpyObsType, observation_space: spaces.Space) -> int:
    """Returns the number of vectorized environments given an observation and
    its corresponding space.

    :param observation: Observation
    :type observation: NumpyObsType
    :param observation_space: Observation space
    :type observation_space: spaces.Space
    :return: Number of vectorized environments
    """
    if isinstance(observation_space, spaces.Dict):
        first_key, first_obs = next(iter(observation.items()))
        return get_vect_dim(first_obs, observation_space[first_key])
    elif isinstance(observation_space, spaces.Tuple):
        return get_vect_dim(observation[0], observation_space[0])
    elif isinstance(observation_space, spaces.MultiBinary):
        observation = (
            observation
            if isinstance(observation, np.ndarray)
            else np.array(observation)
        )
        return (
            observation.shape[0]
            if len(observation.shape) > observation_space.shape
            else 1
        )
    else:
        observation = (
            observation
            if isinstance(observation, np.ndarray)
            else np.array(observation)
        )
        array_shape = observation.shape
        return array_shape[0] if len(array_shape) > len(observation_space.shape) else 1


@singledispatch
def preprocess_observation(
    observation_space: spaces.Space,
    observation: ObservationType,
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
    placeholder_value: Optional[Any] = None,
) -> TorchObsType:
    """Preprocesses observations for forward pass through neural network.

    :param observation_space: The observation space of the environment, defaults to the agent's observation space
    :type observation_space: spaces.Space
    :param observation: Observations of environment
    :type observation: ObservationType
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to "cpu"
    :type device: Union[str, torch.device], optional
    :param normalize_images: Normalize images from [0. 255] to [0, 1], defaults to True
    :type normalize_images: bool, optional
    :param placeholder_value: The value to use as placeholder for missing observations, defaults to None.
    :type placeholder_value: Optional[Any], optional

    :return: Preprocessed observations
    :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or Tuple[torch.Tensor[float], ...]
    """
    raise TypeError(
        f"AgileRL currently doesn't support {type(observation_space)} spaces."
    )


@preprocess_observation.register(spaces.Dict)
def preprocess_dict_observation(
    observation_space: spaces.Dict,
    observation: Dict[str, np.ndarray | torch.Tensor],
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
    placeholder_value: Optional[Any] = None,
) -> Dict[str, TorchObsType]:
    """Preprocess dictionary observations.

    :param observation: Dictionary observation
    :param observation_space: Dictionary observation space
    :param device: Computing device
    :param normalize_images: Whether to normalize images
    :param placeholder_value: Value to replace NaNs with
    :return: Preprocessed dictionary observation
    """
    assert isinstance(
        observation, (dict, TensorDict)
    ), f"Expected dict, got {type(observation)}"

    preprocessed_obs = {}
    for key, _obs in observation.items():
        preprocessed_obs[key] = preprocess_observation(
            observation_space[key],
            observation=_obs,
            device=device,
            normalize_images=normalize_images,
            placeholder_value=placeholder_value,
        )

    return preprocessed_obs


@preprocess_observation.register(spaces.Tuple)
def preprocess_tuple_observation(
    observation_space: spaces.Tuple,
    observation: Tuple[np.ndarray | torch.Tensor, ...],
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
    placeholder_value: Optional[Any] = None,
) -> Tuple[TorchObsType, ...]:
    """Preprocess tuple observations.

    :param observation: Tuple observation
    :param observation_space: Tuple observation space
    :param device: Computing device
    :param normalize_images: Whether to normalize images
    :param placeholder_value: Value to replace NaNs with
    :return: Preprocessed tuple observation
    """
    if isinstance(observation, TensorDict):
        # Convert to tuple with values ordered by index at the end of key
        dict_keys = list(observation.keys())
        dict_keys.sort(key=lambda x: int(x.split("_")[-1]))
        observation = tuple(observation[key] for key in dict_keys)

    assert isinstance(observation, tuple), f"Expected tuple, got {type(observation)}"

    return tuple(
        preprocess_observation(
            _space,
            observation=_obs,
            device=device,
            normalize_images=normalize_images,
            placeholder_value=placeholder_value,
        )
        for _obs, _space in zip(observation, observation_space.spaces)
    )


@preprocess_observation.register(spaces.Box)
def preprocess_box_observation(
    observation_space: spaces.Box,
    observation: NumpyObsType,
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
    placeholder_value: Optional[Any] = None,
) -> torch.Tensor:
    """Preprocess box observations (continuous spaces).

    :param observation: Box observation
    :param observation_space: Box observation space
    :param device: Computing device
    :param normalize_images: Whether to normalize images
    :param placeholder_value: Value to replace NaNs with
    :return: Preprocessed box observation
    """
    # Convert to tensor
    observation = obs_to_tensor(observation, device)

    # Replace NaNs with placeholder value if specified
    if placeholder_value is not None:
        observation = torch.where(
            torch.isnan(observation),
            torch.full_like(observation, placeholder_value),
            observation,
        ).to(torch.float32)

    # Normalize images if applicable and specified
    if len(observation_space.shape) == 3 and normalize_images:
        observation = apply_image_normalization(observation, observation_space)

    space_shape = observation_space.shape

    # Check add batch dimension if necessary
    observation = maybe_add_batch_dim(observation, space_shape)

    return observation


@preprocess_observation.register(spaces.Discrete)
def preprocess_discrete_observation(
    observation_space: spaces.Discrete,
    observation: NumpyObsType,
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
    placeholder_value: Optional[Any] = None,
) -> torch.Tensor:
    """Preprocess discrete observations.

    :param observation: Discrete observation
    :param observation_space: Discrete observation space
    :param device: Computing device
    :param normalize_images: Whether to normalize images
    :param placeholder_value: Value to replace NaNs with
    :return: Preprocessed discrete observation (one-hot encoded)
    """
    # Convert to tensor
    observation = obs_to_tensor(observation, device)

    # Replace NaNs with placeholder value if specified
    if placeholder_value is not None:
        observation = torch.where(
            torch.isnan(observation),
            torch.full_like(observation, placeholder_value),
            observation,
        ).to(torch.float32)

    # One hot encoding of discrete observation
    observation = F.one_hot(
        observation.long(), num_classes=int(observation_space.n)
    ).float()

    if observation_space.n > 1:
        observation = observation.squeeze()  # If n == 1 then squeeze removes obs dim

    space_shape = (observation_space.n,)

    # Check add batch dimension if necessary
    observation = maybe_add_batch_dim(observation, space_shape)

    return observation


@preprocess_observation.register(spaces.MultiDiscrete)
def preprocess_multidiscrete_observation(
    observation_space: spaces.MultiDiscrete,
    observation: NumpyObsType,
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
    placeholder_value: Optional[Any] = None,
) -> torch.Tensor:
    """Preprocess multi-discrete observations.

    :param observation: Multi-discrete observation
    :param observation_space: Multi-discrete observation space
    :param device: Computing device
    :param normalize_images: Whether to normalize images
    :param placeholder_value: Value to replace NaNs with
    :return: Preprocessed multi-discrete observation (one-hot encoded)
    """
    # Convert to tensor
    observation = obs_to_tensor(observation, device)

    # Replace NaNs with placeholder value if specified
    if placeholder_value is not None:
        observation = torch.where(
            torch.isnan(observation),
            torch.full_like(observation, placeholder_value),
            observation,
        ).to(torch.float32)

    # Need to add batch dimension prior to splitting
    space_shape = (sum(observation_space.nvec),)
    observation: torch.Tensor = maybe_add_batch_dim(observation, space_shape)

    # Tensor concatenation of one hot encodings of each Categorical sub-space
    observation = torch.cat(
        [
            F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
            for idx, obs_ in enumerate(torch.split(observation.long(), 1, dim=1))
        ],
        dim=-1,
    )

    # Check add batch dimension if necessary
    observation = maybe_add_batch_dim(observation, space_shape)

    return observation


@preprocess_observation.register(spaces.MultiBinary)
def preprocess_multibinary_observation(
    observation_space: spaces.MultiBinary,
    observation: NumpyObsType,
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
    placeholder_value: Optional[Any] = None,
) -> torch.Tensor:
    """Preprocess multi-binary observations.

    :param observation: Multi-binary observation
    :param observation_space: Multi-binary observation space
    :param device: Computing device
    :param normalize_images: Whether to normalize images
    :param placeholder_value: Value to replace NaNs with
    :return: Preprocessed multi-binary observation
    """
    # Convert to tensor
    observation = obs_to_tensor(observation, device)

    # Replace NaNs with placeholder value if specified
    if placeholder_value is not None:
        observation = torch.where(
            torch.isnan(observation),
            torch.full_like(observation, placeholder_value),
            observation,
        ).to(torch.float32)

    observation = observation.float()
    space_shape = (observation_space.n,)

    # Check add batch dimension if necessary
    observation = maybe_add_batch_dim(observation, space_shape)

    return observation


def apply_image_normalization(
    observation: ArrayOrTensor, observation_space: spaces.Box
) -> ArrayOrTensor:
    """Normalize images using minmax scaling

    :param observation: Observation
    :type observation: ArrayOrTensor
    :param observation_space: Observation space
    :type observation_space: spaces.Box
    :return: Observation
    :rtype: ArrayOrTensor
    """
    if not isinstance(observation_space, spaces.Box):
        raise TypeError(f"Expected spaces.Box, got {type(observation_space)}")

    if np.inf in observation_space.high:
        warnings.warn(
            "np.inf detected in observation_space.high, bypassing normalization."
        )
        return observation

    if -np.inf in observation_space.low:
        warnings.warn(
            "-np.inf detected in observation_space.low, bypassing normalization."
        )
        return observation

    if np.all(observation_space.high == 1) and np.all(observation_space.low == 0):
        return observation

    if isinstance(observation, torch.Tensor):
        low = torch.tensor(
            observation_space.low, device=observation.device, dtype=observation.dtype
        )
        high = torch.tensor(
            observation_space.high, device=observation.device, dtype=observation.dtype
        )
    else:
        low = observation_space.low
        high = observation_space.high

    return (observation - low) / (high - low)


# TODO: The following functions are currently used in PPO (on-policy) as a means of handling
# experiences in the absence of a rollout buffer -> This will not be needed in the future.
def get_experiences_samples(
    minibatch_indices: np.ndarray, *experiences: TorchObsType
) -> Tuple[TorchObsType, ...]:
    """Samples experiences given minibatch indices.

    :param minibatch_indices: Minibatch indices
    :type minibatch_indices: numpy.ndarray[int]
    :param experiences: Experiences to sample from
    :type experiences: Tuple[torch.Tensor[float], ...]

    :return: Sampled experiences
    :rtype: Tuple[torch.Tensor[float], ...]
    """
    sampled_experiences = []
    for exp in experiences:
        if isinstance(exp, dict):
            sampled_exp = {key: value[minibatch_indices] for key, value in exp.items()}
        elif isinstance(exp, tuple):
            sampled_exp = tuple(value[minibatch_indices] for value in exp)
        elif isinstance(exp, torch.Tensor):
            sampled_exp = exp[minibatch_indices]
        else:
            raise TypeError(f"Unsupported experience type: {type(exp)}")

        sampled_experiences.append(sampled_exp)

    return tuple(sampled_experiences)


def stack_experiences(
    *experiences: MaybeObsList, to_torch: bool = True
) -> Tuple[ObservationType, ...]:
    """Stacks experiences into a single array or tensor.

    :param experiences: Experiences to stack
    :type experiences: list[numpy.ndarray[float]] or list[dict[str, numpy.ndarray[float]]]
    :param to_torch: If True, convert the stacked experiences to a torch tensor, defaults to True
    :type to_torch: bool, optional

    :return: Stacked experiences
    :rtype: Tuple[ArrayOrTensor, ...]
    """
    stacked_experiences = []
    for exp in experiences:
        # Some cases where an experience just involves e.g. a single "next_state"
        if not isinstance(exp, list):
            stacked_exp = exp
            if to_torch and isinstance(exp, np.ndarray):
                stacked_exp = torch.from_numpy(stacked_exp)

            stacked_experiences.append(stacked_exp)
            continue

        if isinstance(exp[0], dict):
            stacked_exp = defaultdict(list)
            for it in exp:
                for key, value in it.items():
                    stacked_exp[key].append(value)

            stacked_exp = {key: np.array(value) for key, value in stacked_exp.items()}
            if to_torch:
                stacked_exp = {
                    key: torch.from_numpy(value) for key, value in stacked_exp.items()
                }
        elif isinstance(exp[0], tuple):
            stacked_exp = [[] for _ in exp[0]]
            for it in exp:
                for i, value in enumerate(it):
                    stacked_exp[i].append(value)

            stacked_exp = [np.array(value) for value in stacked_exp]
            if to_torch:
                stacked_exp = [torch.from_numpy(value) for value in stacked_exp]

            stacked_exp = tuple(stacked_exp)

        elif isinstance(exp[0], (np.ndarray, Number)):
            stacked_exp = np.stack(exp)
            if to_torch:
                stacked_exp = torch.from_numpy(stacked_exp)

        elif isinstance(exp[0], torch.Tensor):
            stacked_exp = torch.stack(exp)

        else:
            raise TypeError(f"Unsupported experience type: {type(exp[0])}")

        stacked_experiences.append(stacked_exp)

    return tuple(stacked_experiences)


def stack_and_pad_experiences(
    *experiences: MaybeObsList,
    padding_values: List[Union[int, float, bool]],
    padding_side: str = "right",
) -> Tuple[ArrayOrTensor, ...]:
    """Stacks experiences into a single tensor, padding them to the maximum length.

    :param experiences: Experiences to stack
    :type experiences: list[numpy.ndarray[float]] or list[dict[str, numpy.ndarray[float]]]
    :param to_torch: If True, convert the stacked experiences to a torch tensor, defaults to True
    :type to_torch: bool, optional
    :param padding_side: Side to pad on, defaults to "right"
    :type padding_side: str, optional

    :return: Stacked experiences
    :rtype: Tuple[ArrayOrTensor, ...]
    """
    stacked_experiences = []
    for exp, padding in zip(experiences, padding_values):
        if not isinstance(exp, list):
            stacked_exp = exp
        elif isinstance(exp[0], torch.Tensor):
            max_size = max(e.shape[-1] for e in exp)
            padding_sizes = [(max_size - e.shape[-1]) for e in exp]
            if sum(padding_sizes) != 0:
                exp = [
                    F.pad(
                        e,
                        (
                            (0, padding_size)
                            if padding_side == "right"
                            else (padding_size, 0)
                        ),
                        value=padding,
                    )
                    for e, padding_size in zip(exp, padding_sizes)
                ]
            stacked_exp = torch.cat(exp, dim=0)
        else:
            raise TypeError(f"Unsupported experience type: {type(exp[0])}")
        stacked_experiences.append(stacked_exp)
    return tuple(stacked_experiences)


def flatten_experiences(*experiences: ObservationType) -> Tuple[ArrayOrTensor, ...]:
    """Flattens experiences into a single array or tensor.

    :param experiences: Experiences to flatten
    :type experiences: Tuple[numpy.ndarray[float], ...] or Tuple[torch.Tensor[float], ...]

    :return: Flattened experiences
    :rtype: Tuple[numpy.ndarray[float], ...] or Tuple[torch.Tensor[float], ...]
    """

    def flatten(arr: ArrayOrTensor) -> ArrayOrTensor:
        # Need to flatten batch and n_env dimensions
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)

        arr = arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
        return arr

    flattened_experiences = []
    for exp in experiences:
        if isinstance(exp, dict):
            flattened_exp = {key: flatten(value) for key, value in exp.items()}
        elif isinstance(exp, tuple):
            flattened_exp = tuple(flatten(value) for value in exp)
        elif isinstance(exp, (torch.Tensor, np.ndarray)):
            flattened_exp = flatten(exp)
        else:
            raise TypeError(f"Unsupported experience type: {type(exp)}")

        flattened_experiences.append(flattened_exp)

    return tuple(flattened_experiences)


def is_vectorized_experiences(*experiences: ObservationType) -> bool:
    """Checks if experiences are vectorised.

    :param experiences: Experiences to check
    :type experiences: Tuple[numpy.ndarray[float], ...] or Tuple[torch.Tensor[float], ...]

    :return: True if experiences are vectorised, False otherwise
    :rtype: bool
    """
    is_vec_ls = []
    for exp in experiences:
        if isinstance(exp, dict):
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


def create_warmup_cosine_scheduler(
    optimizer: Union[DeepSpeedOptimizerWrapper, OptimizerWrapper],
    config: CosineLRScheduleConfig,
    min_lr: float,
    max_lr: float,
) -> SequentialLR:
    """Helper function to create cosine annealing lr scheduler with warm-up

    :param optimizer: Optimizer
    :type optimizer: Union[DeepSpeedOptimizerWrapper, OptimizerWrapper]
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
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return scheduler


def remove_nested_files(files: List[str]) -> None:
    """Remove nested files from a list of files.

    :param files: List of files to remove nested files from
    :type files: List[str]
    :param depth: Depth of the nested files, defaults to 0
    :type depth: int, optional
    """
    for f in files:
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)


def vectorize_experiences_by_agent(
    experiences: ExperiencesType, dim: int = 1
) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
    """Reorganizes experiences into a tensor, vectorized by time step

    Example input:
    {'agent_0': [[1, 2, 3, 4]], 'agent_1': [[5, 6, 7, 8]]}
    Example output:
    torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    :param experiences: Dictionaries containing experiences indexed by agent_id that share a policy agent.
    :type experiences: ExperiencesType
    :param dim: New dimension to stack along
    :type dim: int
    :return: Tensor, dict of tensors, or tuple of tensors of experiences, stacked along provided dimension
    :rtype: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]
    """
    if not experiences:
        return torch.tensor([])

    # Get a sample value to determine the type
    sample_value = next(iter(experiences.values()))

    if isinstance(sample_value, dict):
        # Handle dictionary observations
        keys = sample_value.keys()
        return {
            k: vectorize_experiences_by_agent(
                {agent_id: experiences[agent_id][k] for agent_id in experiences},
                dim=dim,
            )
            for k in keys
        }
    elif isinstance(sample_value, tuple):
        # Handle tuple observations
        tuple_length = len(sample_value)
        return tuple(
            vectorize_experiences_by_agent(
                {agent_id: experiences[agent_id][i] for agent_id in experiences},
                dim=dim,
            )
            for i in range(tuple_length)
        )
    else:
        # Original implementation for array/tensor observations
        tensors: List[torch.Tensor] = []
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


def get_space_shape(space: spaces.Space) -> Tuple[int, ...]:
    """Get the shape of a space

    :param space: Space to get shape of
    :type space: spaces.Space
    :return: Shape of space
    :rtype: Tuple[int, ...]
    """
    if isinstance(space, spaces.Discrete):
        return (1,)
    elif isinstance(space, spaces.Box):
        return space.shape
    elif isinstance(space, spaces.MultiDiscrete):
        return (len(space.nvec),)
    elif isinstance(space, spaces.MultiBinary):
        return (space.n,)
    else:
        raise TypeError(f"Unsupported space type: {type(space)}")


def experience_to_tensors(
    experience: dict | tuple | np.ndarray, space: spaces.Space
) -> TorchObsType:
    """Convert experience to numpy array

    :param experience: Experience to convert
    :type experience: dict | tuple | np.ndarray
    :param space: Space to convert experience to
    :type space: spaces.Space
    :return: Numpy array of experience
    :rtype: np.ndarray
    """
    if isinstance(experience, dict):
        return {
            key: experience_to_tensors(value, space[key])
            for key, value in experience.items()
        }
    elif isinstance(experience, tuple):
        return tuple(
            experience_to_tensors(exp, space[i]) for i, exp in enumerate(experience)
        )
    else:
        array = np.array(experience)

        # Ensure experience has a batch dimension
        space_shape = get_space_shape(space)
        array = maybe_add_batch_dim(array, space_shape)
        return torch.from_numpy(array)


def concatenate_tensors(tensors: List[TorchObsType]) -> TorchObsType:
    """Concatenate tensors along first dimension

    :param tensors: List of tensors to concatenate
    :type tensors: List[TorchObsType]
    :return: Concatenated tensor
    :rtype: TorchObsType
    """
    if isinstance(tensors[0], dict):
        return {
            key: concatenate_tensors([t[key] for t in tensors])
            for key in tensors[0].keys()
        }
    elif isinstance(tensors[0], tuple):
        return tuple(
            concatenate_tensors([t[i] for t in tensors]) for i in range(len(tensors[0]))
        )
    else:
        return torch.cat(tensors, dim=0)


def reshape_from_space(tensor: TorchObsType, space: spaces.Space) -> TorchObsType:
    """Reshape tensor from space

    :param tensor: Tensor to reshape
    :type tensor: TorchObsType
    :param space: Space to reshape tensor to
    :type space: spaces.Space
    :return: Reshaped tensor
    :rtype: TorchObsType
    """
    if isinstance(tensor, dict):
        return {
            key: reshape_from_space(value, space[key]) for key, value in tensor.items()
        }
    elif isinstance(tensor, tuple):
        return tuple(
            reshape_from_space(value, space[i]) for i, value in enumerate(tensor)
        )
    else:
        #
        reshaped: torch.Tensor = tensor.reshape(-1, *space.shape)
        for squeeze_dim in [0, -1]:
            if reshaped.size(squeeze_dim) == 1:
                reshaped = reshaped.squeeze(squeeze_dim)
        return reshaped


def concatenate_experiences_into_batches(
    experiences: ExperiencesType, space: spaces.Space
) -> TorchObsType:
    """Reorganizes experiences into a batched tensor

    Example input:
    {'agent_0': [[[...1], [...2]], [[...5], [...6]]],
        'agent_1': [[[...3], [...4]], [[...7], [...8]]]}

    Example output:
    torch.Tensor([...1], [...2], [...3], [...4], [...5], [...6], [...7], [...8])

    :param experiences: Dictionaries containing experiences indexed by agent_id that share a policy agent.
    :type experiences: ExperiencesType
    :param space: Observation/action/etc space to maintain
    :type space: spaces.Space
    :return: Tensor, dict of tensors, or tuple of tensors of experiences, stacked along first dimension, with shape (num_experiences, *shape)
    :rtype: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]
    """
    tensors = []
    for agent_id in experiences.keys():
        exp = experience_to_tensors(experiences[agent_id], space)
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


def clone_llm(
    original_model: PreTrainedModelType, load_state_dict: bool = True
) -> PreTrainedModelType:
    """Clone the actor.

    :param model: Model to clone
    :type model: PreTrainedModelType
    :return: Cloned model
    """
    model_config = original_model.config
    base_model = original_model.model
    model = type(base_model)(model_config)
    if is_peft_model(original_model):
        peft_config = original_model.peft_config[original_model.active_adapter]
        model = get_peft_model(model, peft_config)
    if load_state_dict:
        model.load_state_dict(clone_tensors_for_torch_save(original_model.state_dict()))
    return model
