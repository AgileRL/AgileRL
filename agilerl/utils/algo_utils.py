import copy
import inspect
import warnings
from collections import OrderedDict, defaultdict
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeGuard, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from tensordict.nn import CudaGraphModule
from torch._dynamo import OptimizedModule
from torch.nn import Module
from torch.optim import Optimizer

from agilerl.protocols import EvolvableAttributeType, EvolvableModule, OptimizerWrapper
from agilerl.typing import (
    ArrayOrTensor,
    MaybeObsList,
    NetworkType,
    NumpyObsType,
    ObservationType,
    OptimizerType,
    TorchObsType,
)


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


def make_safe_deepcopies(*args: Union[Module, List[Module]]) -> List[Module]:
    """Makes deep copies of EvolvableModule objects and their attributes.

    :param args: EvolvableModule or lists of EvolvableModule objects to copy.
    :type args: Union[EvolvableModule, List[EvolvableModule]].

    :return: Deep copies of the EvolvableModule objects and their attributes.
    :rtype: List[EvolvableModule].
    """
    copies = []
    for arg in args:
        if isinstance(arg, list):
            arg_copy = [
                copy.deepcopy(inner_arg.cpu()).to(inner_arg.device) for inner_arg in arg
            ]
        else:
            arg_copy = copy.deepcopy(arg.cpu()).to(arg.device)

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
        check_types += (OptimizerWrapper,)

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


def obs_channels_to_first(observation: NumpyObsType) -> NumpyObsType:
    """Converts observation space from channels last to channels first format.

    :param observation_space: Observation space
    :type observation_space: Union[spaces.Box, spaces.Dict]
    :return: Observation space with channels first format
    :rtype: Union[spaces.Box, spaces.Dict]
    """
    if isinstance(observation, np.ndarray):
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
    if isinstance(obs, torch.Tensor):
        return obs.float().to(device)
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device).float()
    elif isinstance(obs, dict):
        return {
            key: torch.as_tensor(_obs, device=device).float()
            for (key, _obs) in obs.items()
        }
    elif isinstance(obs, tuple):
        return tuple(torch.as_tensor(_obs, device=device).float() for _obs in obs)
    elif isinstance(obs, Number):
        return torch.tensor(obs, device=device).float()
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


def maybe_add_batch_dim(
    obs: TorchObsType, space_shape: Tuple[int, ...]
) -> TorchObsType:
    """Adds batch dimension if necessary.

    :param obs: Observation tensor
    :type obs: torch.Tensor[float]
    :param space_shape: Observation space shape
    :type space_shape: Tuple[int, ...]
    :return: Observation tensor with batch dimension
    :rtype: torch.Tensor[float]
    """
    if obs.dim() == len(space_shape):
        obs = obs.unsqueeze(0)
    elif obs.dim() == len(space_shape) + 2:
        obs = obs.view(-1, *space_shape)
    elif obs.dim() != len(space_shape) + 1:
        raise ValueError(
            f"Expected observation to have {len(space_shape) + 1} dimensions, got {obs.dim()}."
        )

    return obs


def preprocess_observation(
    observation: NumpyObsType,
    observation_space: spaces.Space,
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
) -> TorchObsType:
    """Preprocesses observations for forward pass through neural network.

    :param observations: Observations of environment
    :type observations: ObservationType
    :param observation_space: The observation space of the environment, defaults to the agent's observation space
    :type observation_space: spaces.Space
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to "cpu"
    :type device: Union[str, torch.device], optional
    :param normalize_images: Normalize images from [0. 255] to [0, 1], defaults to True
    :type normalize_images: bool, optional

    :return: Preprocessed observations
    :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or Tuple[torch.Tensor[float], ...]
    """
    observation = obs_to_tensor(observation, device)

    # Preprocess different spaces accordingly
    if isinstance(observation_space, spaces.Dict):
        assert isinstance(observation, dict), f"Expected dict, got {type(observation)}"
        preprocessed_obs = {}
        for key, _obs in observation.items():
            preprocessed_obs[key] = preprocess_observation(
                observation=_obs,
                observation_space=observation_space[key],
                device=device,
                normalize_images=normalize_images,
            )

        return preprocessed_obs

    elif isinstance(observation_space, spaces.Tuple):
        assert isinstance(
            observation, tuple
        ), f"Expected tuple, got {type(observation)}"
        return tuple(
            preprocess_observation(_obs, _space, device, normalize_images)
            for _obs, _space in zip(observation, observation_space.spaces)
        )

    assert isinstance(
        observation, torch.Tensor
    ), f"Expected torch.Tensor, got {type(observation)}"

    if isinstance(observation_space, spaces.Box):
        # Normalize images if applicable and specified
        if len(observation_space.shape) == 3 and normalize_images:
            observation = observation / 255.0

        space_shape = observation_space.shape

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding of discrete observation
        observation = F.one_hot(
            observation.long(), num_classes=int(observation_space.n)
        ).float()
        if observation_space.n > 1:
            observation = (
                observation.squeeze()
            )  # If n == 1 then squeeze removes obs dim

        space_shape = (observation_space.n,)

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        observation = torch.cat(
            [
                F.one_hot(
                    obs_.long(), num_classes=int(observation_space.nvec[idx])
                ).float()
                for idx, obs_ in enumerate(torch.split(observation.long(), 1, dim=1))
            ],
            dim=-1,
        )
        space_shape = (sum(observation_space.nvec),)
    else:
        raise TypeError(
            f"AgileRL currently doesn't support {type(observation_space)} spaces."
        )

    # Check add batch dimension if necessary
    observation = maybe_add_batch_dim(observation, space_shape)

    return observation


def apply_image_normalization(
    observation: NumpyObsType, observation_space: spaces.Space
) -> NumpyObsType:
    """Normalize images using minmax scaling

    :param observation: Observation
    :type observation: NumpyObsType
    :param observation_space: Observation space
    :type observation_space: spaces.Space
    :return: Observation
    :rtype: NumpyObsType
    """
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
        # minmax scaling
        return observation

    observation = (observation - observation_space.low) / (
        observation_space.high - observation_space.low
    )
    return observation


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
        elif isinstance(exp, torch.Tensor):
            sampled_exp = exp[minibatch_indices]
        else:
            raise TypeError(f"Unsupported experience type: {type(exp)}")

        sampled_experiences.append(sampled_exp)

    return tuple(sampled_experiences)


def stack_experiences(
    *experiences: MaybeObsList, to_torch: bool = True
) -> Tuple[ArrayOrTensor, ...]:
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

        elif isinstance(exp[0], dict):
            stacked_exp = defaultdict(list)
            for it in exp:
                for key, value in it.items():
                    stacked_exp[key].append(value)

            stacked_exp = {key: np.array(value) for key, value in stacked_exp.items()}
            if to_torch:
                stacked_exp = {
                    key: torch.from_numpy(value) for key, value in stacked_exp.items()
                }

        elif isinstance(exp[0], (np.ndarray, Number)):
            stacked_exp = np.array(exp)
            if to_torch:
                stacked_exp = torch.from_numpy(stacked_exp)

        elif isinstance(exp[0], torch.Tensor):
            stacked_exp = torch.stack(exp)

        else:
            raise TypeError(f"Unsupported experience type: {type(exp[0])}")

        stacked_experiences.append(stacked_exp)

    return tuple(stacked_experiences)


def flatten_experiences(*experiences: ArrayOrTensor) -> Tuple[ArrayOrTensor, ...]:
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
        elif isinstance(exp, (torch.Tensor, np.ndarray)):
            flattened_exp = flatten(exp)
        else:
            raise TypeError(f"Unsupported experience type: {type(exp)}")

        flattened_experiences.append(flattened_exp)

    return tuple(flattened_experiences)


def is_vectorized_experiences(*experiences: ArrayOrTensor) -> bool:
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
        else:
            is_vec = exp.ndim > 1

        is_vec_ls.append(is_vec)

    return all(is_vec_ls)
