from collections import OrderedDict, defaultdict
from numbers import Number
from typing import Union, Dict, Any, Tuple
import torch
import numpy as np
from gymnasium import spaces
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from torch.optim import Optimizer
from torch.nn import Module
import torch.nn.functional as F
from torch._dynamo import OptimizedModule

from agilerl.modules.base import EvolvableModule
from agilerl.typing import (
    NumpyObsType,
    TorchObsType,
    NetworkType,
    OptimizerType,
    MaybeObsList,
    ArrayOrTensor
)

def unwrap_optimizer(
        optimizer: OptimizerType,
        network: NetworkType,
        lr: float
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
            unwrapped_optimizer: Optimizer = type(optimizer.optimizer)(network.parameters(), lr=lr)
        unwrapped_optimizer.load_state_dict(optimizer.state_dict())
        return unwrapped_optimizer
    else:
        return optimizer
    
def recursive_check_module_attrs(obj: Any, networks_only: bool = False) -> bool:
    """Recursively check if the object has any attributes that are EvolvableModule's or Optimizer's.

    :param obj: The object to check for EvolvableModule's or Optimizer's.
    :type obj: Any
    :return: True if the object has any attributes that are EvolvableModule's or Optimizer's, False otherwise.
    :rtype: bool
    """
    check_types = (OptimizedModule, EvolvableModule)
    if not networks_only:
        check_types += (Optimizer,)

    if isinstance(obj, check_types):
        return True
    if isinstance(obj, dict):
        return any(recursive_check_module_attrs(v, networks_only=networks_only) for v in obj.values())
    if isinstance(obj, list):
        return any(recursive_check_module_attrs(v, networks_only=networks_only) for v in obj)
    return False

def chkpt_attribute_to_device(chkpt_dict: Dict[str, torch.Tensor], device: str) -> Dict[str, Any]:
    """Place checkpoint attributes on device. Used when loading saved agents.

    :param chkpt_dict: Checkpoint dictionary
    :type chkpt_dict: dict
    :param device: Device for accelerated computing, 'cpu' or 'cuda'
    :type device: str
    """
    for key, value in chkpt_dict.items():
        if hasattr(value, "device") and not isinstance(value, Accelerator):
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

def compile_model(model: Module, mode: Union[str, None] = "default") -> Module:
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

def observation_space_channels_to_first(observation_space: Union[spaces.Box, spaces.Dict]) -> spaces.Box:
    """Swaps the channel order of an image observation space from [H, W, C] -> [C, H, W].

    :param observation_space: Observation space
    :type observation_space: spaces.Box
    :return: Observation space with swapped channels
    :rtype: spaces.Box
    """
    if isinstance(observation_space, spaces.Dict):
        for key in observation_space.spaces.keys():
            if isinstance(observation_space[key], spaces.Box) and len(observation_space[key].shape) == 3:
                observation_space[key] = observation_space_channels_to_first(observation_space[key])
        return observation_space
    
    low = observation_space.low.transpose(2, 0, 1)
    high = observation_space.high.transpose(2, 0, 1)
    return spaces.Box(low=low, high=high, dtype=observation_space.dtype)

def obs_channels_to_first(observation: NumpyObsType) ->  NumpyObsType:
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

def obs_to_tensor(obs: NumpyObsType, device: Union[str, torch.device]) -> TorchObsType:
    """
    Moves the observation to the given device.

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
        return {key: torch.as_tensor(_obs, device=device).float() for (key, _obs) in obs.items()}
    elif isinstance(obs, tuple):
        return tuple(torch.as_tensor(_obs, device=device).float() for _obs in obs)
    elif isinstance(obs, Number):
        return torch.tensor(obs, device=device).float()
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")
    
def maybe_add_batch_dim(obs: TorchObsType, space_shape: Tuple[int, ...]) -> TorchObsType:
    """Adds batch dimension if necessary

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
        normalize_images: bool = True
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
                normalize_images=normalize_images
                )

        return preprocessed_obs
    
    elif isinstance(observation_space, spaces.Tuple):
        assert isinstance(observation, tuple), f"Expected tuple, got {type(observation)}"
        return tuple(
            preprocess_observation(_obs, _space, device, normalize_images) for _obs, _space in zip(observation, observation_space.spaces)
            )
    
    assert isinstance(observation, torch.Tensor), f"Expected torch.Tensor, got {type(observation)}"
    
    if isinstance(observation_space, spaces.Box):
        # Normalize images if applicable and specified
        if len(observation_space.shape) == 3 and normalize_images:
            observation = observation.float() / 255.0

        space_shape = observation_space.shape
    
    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding of discrete observation
        observation = F.one_hot(observation.long(), num_classes=int(observation_space.n)).float()
        if observation_space.n > 1:
            observation = observation.squeeze() # If n == 1 then squeeze removes obs dim

        space_shape = (observation_space.n,)

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        observation = torch.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(torch.split(observation.long(), 1, dim=1))
            ],
            dim=-1,
        )
        space_shape = (sum(observation_space.nvec),)
    else:
        raise TypeError(f"AgileRL currently doesn't support {type(observation_space)} spaces.")
    
    # Check add batch dimension if necessary
    observation = maybe_add_batch_dim(observation, space_shape)

    return observation

def get_experiences_samples(minibatch_indices: np.ndarray, *experiences: TorchObsType) -> Tuple[TorchObsType, ...]:
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
            print(exp.size())
            sampled_exp = exp[minibatch_indices]
        else:
            raise TypeError(f"Unsupported experience type: {type(exp)}")
        
        sampled_experiences.append(sampled_exp)
    
    return tuple(sampled_experiences)

def stack_experiences(*experiences: MaybeObsList, to_torch: bool = True) -> Tuple[ArrayOrTensor, ...]:
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
                stacked_exp = {key: torch.from_numpy(value) for key, value in stacked_exp.items()}
            
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
            print(exp.size())
            flattened_exp = flatten(exp)
        else:
            raise TypeError(f"Unsupported experience type: {type(exp)}")
        
        flattened_experiences.append(flattened_exp)
    
    return tuple(flattened_experiences)