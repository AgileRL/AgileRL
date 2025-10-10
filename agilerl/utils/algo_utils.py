import inspect
import os
import shutil
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import singledispatch
from numbers import Number
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from peft import PeftModel, get_peft_model
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
from torch._dynamo import OptimizedModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import PreTrainedModel

from agilerl.modules.dummy import DummyEvolvable
from agilerl.protocols import (
    EvolvableAttributeType,
    EvolvableModule,
    EvolvableNetwork,
    ModuleDict,
    OptimizerWrapper,
)
from agilerl.typing import (
    ArrayOrTensor,
    BPTTSequenceType,
    ExperiencesType,
    GymSpaceType,
    MaybeObsList,
    MultiAgentModule,
    NetConfigType,
    NumpyObsType,
    ObservationType,
    SupportedObsSpaces,
    TorchObsType,
)

PreTrainedModelType = Union[PeftModel, PreTrainedModel]


def check_supported_space(observation_space: GymSpaceType) -> None:
    """Checks if the observation space is supported by AgileRL.

    :param observation_space: The observation space to check.
    :type observation_space: GymSpaceType
    """
    assert isinstance(
        observation_space, spaces.Space
    ), "Observation space must be an instance of gymnasium.spaces.Space."

    assert not isinstance(
        observation_space, (spaces.Graph, spaces.Sequence, spaces.OneOf)
    ), "AgileRL does not support Graph, Sequence, and OneOf spaces."

    if isinstance(observation_space, spaces.Dict):
        for subspace in observation_space.spaces.values():
            assert not isinstance(
                subspace, (spaces.Tuple, spaces.Dict)
            ), "AgileRL does not support nested Tuple and Dict spaces in Dict spaces."
            check_supported_space(subspace)
    elif isinstance(observation_space, spaces.Tuple):
        for subspace in observation_space.spaces:
            assert not isinstance(
                subspace, (spaces.Tuple, spaces.Dict)
            ), "AgileRL does not support nested Tuple and Dict spaces in Tuple spaces."
            check_supported_space(subspace)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        assert len(observation_space.nvec.shape) == 1, (
            "AgileRL does not support multi-dimensional MultiDiscrete spaces. Got shape "
            f"{observation_space.nvec.shape}."
        )


def get_input_size_from_space(
    observation_space: GymSpaceType,
) -> Union[int, dict[str, int], tuple[int, ...]]:
    """Returns the dimension of the state space as it pertains to the underlying
    networks (i.e. the input size of the networks).

    :param observation_space: The observation space of the environment.
    :type observation_space: spaces.Space or list[spaces.Space] or dict[str, spaces.Space].

    :return: The dimension of the state space.
    :rtype: Union[int, dict[str, int], tuple[int, ...]]
    """
    if isinstance(observation_space, (list, tuple, spaces.Tuple)):
        return tuple(get_input_size_from_space(space) for space in observation_space)
    elif isinstance(observation_space, (spaces.Dict, dict)):
        return {
            key: get_input_size_from_space(subspace)
            for key, subspace in observation_space.items()
        }
    elif isinstance(observation_space, spaces.Discrete):
        return (observation_space.n,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        return (sum(observation_space.nvec),)
    elif isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.MultiBinary):
        return (observation_space.n,)
    else:
        raise AttributeError(
            f"Can't access state dimensions for {type(observation_space)} spaces."
        )


def get_output_size_from_space(
    action_space: GymSpaceType,
) -> Union[int, dict[str, int], tuple[int, ...]]:
    """Returns the dimension of the action space as it pertains to the underlying
    networks (i.e. the output size of the networks).

    :param action_space: The action space of the environment.
    :type action_space: spaces.Space or list[spaces.Space] or dict[str, spaces.Space].

    :return: The dimension of the action space.
    :rtype: Union[int, dict[str, int], tuple[int, ...]]
    """
    if isinstance(action_space, (list, tuple)):
        return tuple(get_output_size_from_space(space) for space in action_space)
    elif isinstance(action_space, (spaces.Dict, dict)):
        return {
            key: get_output_size_from_space(subspace)
            for key, subspace in action_space.items()
        }
    elif isinstance(action_space, spaces.MultiBinary):
        return action_space.n
    elif isinstance(action_space, spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        return sum(action_space.nvec)
    elif isinstance(action_space, spaces.Box):
        # NOTE: Assume continuous actions are always one-dimensional
        return action_space.shape[0]
    else:
        raise AttributeError(
            f"Can't access action dimensions for {type(action_space)} spaces."
        )


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


def get_hidden_states_shape_from_model(model: nn.Module) -> dict[str, int]:
    """Loops through all of the modules in the model and checks if they have a
    `hidden_state_architecture` attribute. If they do, it adds the items to a
    dictionary and returns it. This should make it easier to initialize the
    hidden states of the model.

    :param model: The model to get the hidden states from.
    :type model: nn.Module
    :return: The hidden states shape from the model.
    :rtype: dict[str, int]
    """
    hidden_state_architecture = {}
    for name, module in model.named_modules():
        if hasattr(module, "hidden_state_architecture"):
            hidden_state_architecture.update(
                {
                    f"{module.name}_{k}": v
                    for k, v in module.hidden_state_architecture.items()
                }
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
        episode
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
            for start in range(0, len(episode) - max_seq_len + 1)
        ]
    elif sequence_type == BPTTSequenceType.FIFTY_PERCENT_OVERLAP:
        step_size = max_seq_len // 2
        sequences = [
            episode[start : start + max_seq_len]
            for start in range(0, len(episode) - max_seq_len + 1, step_size)
        ]
    else:
        raise NotImplementedError(
            f"Received unrecognized sequence type: {sequence_type}"
        )
    return sequences


def is_image_space(space: spaces.Space) -> bool:
    """Check if the space is an image space. We ignore dtype and number of channels
    checks.

    :param space: Input space
    :type space: spaces.Space

    :return: True if the space is an image space, False otherwise
    :rtype: bool
    """
    return isinstance(space, spaces.Box) and len(space.shape) == 3


def get_obs_shape(space: spaces.Space) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """Returns the shape of the observation space.

    :param space: Observation space
    :type space: spaces.Space
    :return: Shape of the observation space
    :rtype: tuple[int, ...] | dict[str, tuple[int, ...]]
    """
    if isinstance(space, spaces.Box):
        return space.shape
    elif isinstance(space, spaces.Discrete):
        return (1,)
    elif isinstance(space, spaces.MultiDiscrete):
        return (len(space.nvec),)
    elif isinstance(space, spaces.MultiBinary):
        return space.shape
    elif isinstance(space, spaces.Dict):
        return {
            key: get_obs_shape(subspace) for (key, subspace) in space.spaces.items()
        }
    elif isinstance(space, spaces.Tuple):
        return tuple(get_obs_shape(subspace) for subspace in space)
    else:
        raise NotImplementedError(f"{space} observation space is not supported")


def get_num_actions(space: spaces.Space) -> int:
    """Returns the number of actions.

    :param space: Action space
    :type space: spaces.Space
    :return: Number of actions
    :rtype: int
    """
    if isinstance(space, spaces.Box):
        return spaces.flatdim(space)
    elif isinstance(space, spaces.Discrete):
        return 1
    elif isinstance(space, spaces.MultiDiscrete):
        return len(space.nvec)
    elif isinstance(space, spaces.MultiBinary):
        return space.n
    else:
        raise NotImplementedError(f"{space} action space is not supported by AgileRL.")


def make_safe_deepcopies(
    *args: Union[EvolvableModule, list[EvolvableModule]],
) -> list[EvolvableModule]:
    """Makes deep copies of EvolvableModule objects and their attributes.

    :param args: EvolvableModule or lists of EvolvableModule objects to copy.
    :type args: Union[EvolvableModule, list[EvolvableModule]].

    :return: Deep copies of the EvolvableModule objects and their attributes.
    :rtype: list[EvolvableModule].
    """
    copies = []
    for arg in args:
        if isinstance(arg, list):
            arg_copy = [inner_arg.clone() for inner_arg in arg]
        else:
            arg_copy = arg.clone()

        copies.append(arg_copy)

    return copies[0] if len(copies) == 1 else copies


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


def recursive_check_module_attrs(obj: Any, networks_only: bool = False) -> bool:
    """Recursively check if the object has any attributes that are EvolvableModule objects or Optimizer's,
    excluding metaclasses.

    :param obj: The object to check for EvolvableModule objects or Optimizer's.
    :type obj: Any
    :param networks_only: If True, only check for EvolvableModule objects, defaults to False
    :type networks_only: bool, optional

    :return: True if the object has any attributes that are EvolvableModule objects or Optimizer's, False otherwise.
    :rtype: bool
    """
    check_types = (OptimizedModule, EvolvableModule)
    if not networks_only:
        check_types += (OptimizerWrapper,)

    # Exclude metaclasses
    if isinstance(obj, type):
        return False

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
    chkpt_dict: dict[str, torch.Tensor], device: str
) -> dict[str, Any]:
    """Place checkpoint attributes on device. Used when loading saved agents.

    :param chkpt_dict: Checkpoint dictionary
    :type chkpt_dict: dict
    :param device: Device for accelerated computing, 'cpu' or 'cuda'
    :type device: str

    :return: Checkpoint dictionary with attributes on device
    :rtype: dict[str, Any]
    """
    if isinstance(chkpt_dict, list):
        return [chkpt_attribute_to_device(chkpt, device) for chkpt in chkpt_dict]

    assert isinstance(chkpt_dict, dict), f"Expected dict, got {type(chkpt_dict)}"

    for key, value in chkpt_dict.items():
        if isinstance(value, torch.Tensor):
            chkpt_dict[key] = value.to(device)

    return chkpt_dict


def key_in_nested_dict(nested_dict: dict[str, Any], target: str) -> bool:
    """Helper function to determine if key is in nested dictionary

    :param nested_dict: Nested dictionary
    :type nested_dict: dict[str, dict[str, ...]]
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
    """Removes _orig_mod prefix on state dict created by torch compile

    :param state_dict: model state dict
    :type state_dict: dict
    :return: state dict with prefix removed
    :rtype: dict[str, Any]
    """
    return OrderedDict(
        [
            (k.split(".", 1)[1], v) if k.startswith("_orig_mod") else (k, v)
            for k, v in state_dict.items()
        ]
    )


def module_checkpoint_dict(module: EvolvableAttributeType, name: str) -> dict[str, Any]:
    """Returns a dictionary containing the module's class, init dict, and state dict.

    :param module: The module to checkpoint.
    :type module: EvolvableAttributeType
    :param name: The name of the attribute to checkpoint.
    :type name: str

    :return: A dictionary containing the module's class, init dict, and state dict.
    :rtype: dict[str, Any]
    """
    if isinstance(module, ModuleDict):
        return module_checkpoint_multiagent(module, name)

    return module_checkpoint_single(module, name)


def module_checkpoint_single(module: EvolvableModule, name: str) -> dict[str, Any]:
    """Returns a dictionary containing the module's class, init dict, and state dict.

    :param module: The module to checkpoint.
    :type module: EvolvableModule
    :param name: The name of the attribute to checkpoint.
    :type name: str
    :return: A dictionary containing the module's class, init dict, and state dict.
    :rtype: dict[str, Any]
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


def module_checkpoint_multiagent(module: MultiAgentModule, name: str) -> dict[str, Any]:
    """Returns a dictionary containing the module's class, init dict, and state dict.

    :param module: The module to checkpoint.
    :type module: ModuleDict
    :param name: The name of the attribute to checkpoint.
    :type name: str
    :return: A dictionary containing the module's class, init dict, and state dict.
    :rtype: dict[str, Any]
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


def format_shared_critic_encoder(encoder_configs: NetConfigType) -> dict[str, Any]:
    """Formats the shared critic  (i.e. `EvolvableMultiInput`) config from the available
    encoder configs from all of the sub-agents. This dictionary is built when extracting the net
    config passed by the user in `MultiAgentAlgorithm.extract_net_config`.

    .. note::
        If the user specified multiple different MLP configurations for different sub-agents /
        groups, the deepest MLP config will be used for the shared critics `EvolvableMLP`.

    :param encoder_configs: Network configuration
    :type encoder_configs: dict[str, Any]
    :return: Formatted shared critic encoder config
    :rtype: dict[str, Any]
    """
    encoder_config = defaultdict(dict)
    for encoder_key, config in encoder_configs.items():
        if encoder_key == "mlp_config":
            encoder_config[encoder_key] = config

            # If we have homogeneous agents, we can process the raw observations with an EvolvableMLP
            encoder_config["vector_space_mlp"] = len(encoder_configs) == 1
            encoder_config["latent_dim"] = config.get("hidden_size", [32])[-1]
            encoder_config["output_layernorm"] = config.get("layer_norm", False)
        else:
            encoder_config["init_dicts"][encoder_key] = config

    return encoder_config


def get_deepest_head_config(
    net_config: NetConfigType, agent_ids: list[str]
) -> NetConfigType:
    """Returns the deepest head config from the nested net config.

    :param net_config: Network configuration
    :type net_config: NetConfigType
    :param agent_ids: List of agent IDs
    :type agent_ids: list[str]
    :return: Largest head config
    """
    assert all(
        agent_id in net_config.keys() for agent_id in agent_ids
    ), "All passed agent IDs must be present in the net config."

    deepest = None
    for agent_id in agent_ids:
        agent_config = net_config[agent_id]
        agent_head_config = agent_config.get("head_config", None)
        if agent_head_config is not None:
            if deepest is None:
                deepest = agent_head_config
            elif len(agent_head_config["hidden_size"]) > len(deepest["hidden_size"]):
                deepest = agent_head_config

    if deepest is None:
        raise ValueError("No head config found in the passed net config.")

    return deepest


def concatenate_spaces(space_list: list[SupportedObsSpaces]) -> spaces.Space:
    """Concatenates a list of spaces into a single space. If spaces correspond to images,
    we check that their shapes are the same and use the first space's shape as the shape of the
    concatenated space.

    :param space_list: List of spaces to concatenate
    :type space_list: list[SupportedObsSpaces]
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
        # Require image spaces to have the same shape in order to concatenate
        if all(is_image_space(space) for space in space_list):
            assert all(
                space.shape == space_list[0].shape for space in space_list
            ), "Cannot concatenate image spaces with different CxHxW dimensions."

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

    raise TypeError(f"Unrecognized type of observation {type(obs)}")


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


def add_placeholder_value(obs: torch.Tensor, placeholder_value: float) -> torch.Tensor:
    """Adds placeholder value to observation.

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


@singledispatch
def maybe_add_batch_dim(
    array_like: ObservationType, space: spaces.Space, actions: bool = False
) -> ObservationType:
    """Adds batch dimension if necessary.

    :param array_like: Array or tensor
    :type array_like: ObservationType
    :param space: Observation space
    :type space: spaces.Space
    :param actions: Whether the array is an action, defaults to False
    :type actions: bool, optional
    :return: Observation tensor with batch dimension
    :rtype: ObservationType
    """
    raise TypeError(f"Cannot add batch dimension for {type(array_like)}.")


@maybe_add_batch_dim.register(np.ndarray)
def maybe_add_batch_dim_np(
    array_like: np.ndarray, space: spaces.Space, actions: bool = False
) -> np.ndarray:
    space_shape = (
        get_input_size_from_space(space) if not actions else (get_num_actions(space),)
    )
    if len(array_like.shape) == len(space_shape):
        array_like = np.expand_dims(array_like, 0)
    elif len(array_like.shape) == len(space_shape) + 2:
        array_like = array_like.reshape(-1, *space_shape)
    elif len(array_like.shape) != len(space_shape) + 1:
        raise ValueError(
            f"Expected observation to have {len(space_shape) + 1} dimensions, got {len(array_like.shape)}."
        )

    return array_like


@maybe_add_batch_dim.register(torch.Tensor)
def maybe_add_batch_dim_torch(
    array_like: torch.Tensor, space: spaces.Space, actions: bool = False
) -> torch.Tensor:
    space_shape = (
        get_input_size_from_space(space) if not actions else (get_num_actions(space),)
    )
    if array_like.ndim == len(space_shape):
        array_like = array_like.unsqueeze(0)
    elif array_like.ndim == len(space_shape) + 2:
        array_like = array_like.view(-1, *space_shape)
    elif array_like.ndim != len(space_shape) + 1:
        raise ValueError(
            f"Expected observation to have {len(space_shape) + 1} dimensions, got {len(array_like.shape)}."
        )

    return array_like


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
    :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or tuple[torch.Tensor[float], ...]
    """
    raise TypeError(
        f"AgileRL currently doesn't support {type(observation_space)} spaces."
    )


@preprocess_observation.register(spaces.Dict)
def preprocess_dict_observation(
    observation_space: spaces.Dict,
    observation: dict[str, np.ndarray | torch.Tensor],
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
    placeholder_value: Optional[Any] = None,
) -> dict[str, TorchObsType]:
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

    preprocessed_obs = OrderedDict()
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
    observation: tuple[np.ndarray | torch.Tensor, ...],
    device: Union[str, torch.device] = "cpu",
    normalize_images: bool = True,
    placeholder_value: Optional[Any] = None,
) -> tuple[TorchObsType, ...]:
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
        dict_keys: list[str] = list(observation.keys())
        dict_keys.sort(key=lambda x: int(x.split("_")[-1]))
        observation = tuple(observation[key] for key in dict_keys)

    assert isinstance(
        observation, tuple
    ), f"Expected tuple observation, got {type(observation)}"

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
    observation: np.ndarray | torch.Tensor,
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
        observation = add_placeholder_value(observation, placeholder_value)

    # Normalize images if applicable and specified
    if len(observation_space.shape) == 3 and normalize_images:
        observation = apply_image_normalization(observation, observation_space)

    # Check add batch dimension if necessary
    observation = maybe_add_batch_dim(observation, observation_space)

    return observation


@preprocess_observation.register(spaces.Discrete)
def preprocess_discrete_observation(
    observation_space: spaces.Discrete,
    observation: np.ndarray | torch.Tensor,
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
        observation = add_placeholder_value(observation, placeholder_value)

    # One hot encoding of discrete observation
    observation = F.one_hot(
        observation.long(), num_classes=int(observation_space.n)
    ).float()

    if observation_space.n > 1:
        observation = observation.squeeze()  # If n == 1 then squeeze removes obs dim

    # Check add batch dimension if necessary
    observation = maybe_add_batch_dim(observation, observation_space)

    return observation


@preprocess_observation.register(spaces.MultiDiscrete)
def preprocess_multidiscrete_observation(
    observation_space: spaces.MultiDiscrete,
    observation: np.ndarray | torch.Tensor,
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
    observation: np.ndarray | torch.Tensor,
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
        observation = add_placeholder_value(observation, placeholder_value)

    observation = observation.float()

    # Check add batch dimension if necessary
    observation = maybe_add_batch_dim(observation, observation_space)

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
) -> tuple[TorchObsType, ...]:
    """Samples experiences given minibatch indices.

    :param minibatch_indices: Minibatch indices
    :type minibatch_indices: numpy.ndarray[int]
    :param experiences: Experiences to sample from
    :type experiences: tuple[torch.Tensor[float], ...]

    :return: Sampled experiences
    :rtype: tuple[torch.Tensor[float], ...]
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
) -> tuple[ObservationType, ...]:
    """Stacks experiences into a single array or tensor.

    :param experiences: Experiences to stack
    :type experiences: list[numpy.ndarray[float]] or list[dict[str, numpy.ndarray[float]]]
    :param to_torch: If True, convert the stacked experiences to a torch tensor, defaults to True
    :type to_torch: bool, optional

    :return: Stacked experiences
    :rtype: tuple[ArrayOrTensor, ...]
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
    padding_values: list[Union[int, float, bool]],
    padding_side: str = "right",
    device: Optional[str] = None,
) -> tuple[ArrayOrTensor, ...]:
    """Stacks experiences into a single tensor, padding them to the maximum length.

    :param experiences: Experiences to stack
    :type experiences: list[numpy.ndarray[float]] or list[dict[str, numpy.ndarray[float]]]
    :param to_torch: If True, convert the stacked experiences to a torch tensor, defaults to True
    :type to_torch: bool, optional
    :param padding_side: Side to pad on, defaults to "right"
    :type padding_side: str, optional

    :return: Stacked experiences
    :rtype: tuple[ArrayOrTensor, ...]
    """
    stacked_experiences = []
    for exp, padding in zip(experiences, padding_values):
        if not isinstance(exp, list):
            stacked_exp = exp
        elif isinstance(exp[0], torch.Tensor):
            stacked_exp = _stack_and_pad_tensor_list(exp, padding, padding_side)
        elif isinstance(exp[0], (list, tuple)):
            exp = [torch.tensor(e).unsqueeze(0) for e in exp]
            stacked_exp = _stack_and_pad_tensor_list(exp, padding, padding_side)
        else:
            raise TypeError(f"Unsupported experience type: {type(exp[0])}")
        if device is not None:
            stacked_exp = stacked_exp.to(device)
        stacked_experiences.append(stacked_exp)
    return tuple(stacked_experiences)


def _stack_and_pad_tensor_list(
    exp: list[torch.Tensor], padding: int, padding_side: str = "right"
) -> torch.Tensor:
    """
    Stack and pad a list of tensors.

    :param exp: List of tensors to stack and pad
    :type exp: list[torch.Tensor]
    :param padding_value: Value to pad with
    :type padding_value: int
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
            for e, padding_size in zip(exp, padding_sizes)
        ]
    return torch.cat(exp, dim=0)


def flatten_experiences(*experiences: ObservationType) -> tuple[ArrayOrTensor, ...]:
    """Flattens experiences into a single array or tensor.

    :param experiences: Experiences to flatten
    :type experiences: tuple[numpy.ndarray[float], ...] or tuple[torch.Tensor[float], ...]

    :return: Flattened experiences
    :rtype: tuple[numpy.ndarray[float], ...] or tuple[torch.Tensor[float], ...]
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
    :type experiences: tuple[numpy.ndarray[float], ...] or tuple[torch.Tensor[float], ...]

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


@dataclass
class VLLMConfig:
    """Data class to configure a VLLM client.

    Note: has the same defaults as the VLLMClient class from trl library.

    :param base_url: Base URL of the VLLM server, defaults to None
    :type base_url: Optional[str], optional
    :param host: Host of the VLLM server, defaults to "0.0.0.0"
    :type host: str, optional
    :param server_port: Server port of the VLLM server, defaults to 8000
    :type server_port: int, optional
    :param group_port: Group port of the VLLM server, defaults to 51216
    :type group_port: int, optional
    """

    # Colocate mode parameters
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.3
    max_num_seqs: int = 8


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: CosineLRScheduleConfig,
    min_lr: float,
    max_lr: float,
) -> SequentialLR:
    """Helper function to create cosine annealing lr scheduler with warm-up

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
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return scheduler


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
    experiences: ExperiencesType, dim: int = 1
) -> Union[torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
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
    :rtype: Union[torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor, ...]]
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


def experience_to_tensors(
    experience: dict | tuple | np.ndarray, space: spaces.Space, actions: bool = False
) -> TorchObsType:
    """Convert experience to numpy array

    :param experience: Experience to convert
    :type experience: dict | tuple | np.ndarray
    :param space: Space to convert experience to
    :type space: spaces.Space
    :param actions: Whether the experience is an action, defaults to False
    :type actions: bool, optional
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
        array = maybe_add_batch_dim(array, space, actions)
        return torch.from_numpy(array)


def concatenate_tensors(tensors: list[TorchObsType]) -> TorchObsType:
    """Concatenate tensors along first dimension

    :param tensors: List of tensors to concatenate
    :type tensors: list[TorchObsType]
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

    reshaped: torch.Tensor = tensor.reshape(-1, *space.shape)
    for squeeze_dim in [0, -1]:
        if reshaped.size(squeeze_dim) == 1:
            reshaped = reshaped.squeeze(squeeze_dim)

    return reshaped


def concatenate_experiences_into_batches(
    experiences: ExperiencesType, space: spaces.Space, actions: bool = False
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
    :param actions: Whether the experiences are actions, defaults to False
    :type actions: bool, optional
    :return: Tensor, dict of tensors, or tuple of tensors of experiences, stacked along first dimension, with shape (num_experiences, *shape)
    :rtype: Union[torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor, ...]]
    """
    tensors = []
    for agent_id in experiences.keys():
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


def clone_llm(
    original_model: PreTrainedModelType | DummyEvolvable,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> PreTrainedModelType:
    """Clone the actor.

    :param original_model: Model to clone
    :type original_model: PreTrainedModelType
    :param state_dict: State dict to load, defaults to None
    :type state_dict: Optional[dict[str, torch.Tensor]], optional
    :return: Cloned model
    """
    match original_model:
        case PeftModel():
            pass
        case PreTrainedModel():
            pass
        case DummyEvolvable():
            original_model = original_model.module
        case _:
            raise ValueError(f"Invalid 'original_model' type: {type(original_model)}")

    model_config = original_model.config
    base_model = original_model.model
    model = type(base_model)(model_config)
    # Get all adapter names

    if hasattr(original_model, "peft_config"):
        adapter_names = list(original_model.peft_config.keys())

        if len(adapter_names) > 1:
            warnings.warn(
                "Multiple adapters detected. Only the first adapter will be used for RL finetuning."
            )
        # Add first adapter using get_peft_model
        first_adapter = adapter_names[0]
        first_config = original_model.peft_config[first_adapter]
        model = get_peft_model(model, first_config, adapter_name=first_adapter)

        # Add remaining adapters using add_adapter
        for adapter_name in adapter_names[1:]:
            peft_config = original_model.peft_config[adapter_name]
            model.add_adapter(peft_config=peft_config, adapter_name=adapter_name)
        model.disable_adapter()

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    return model
