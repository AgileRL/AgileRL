import random
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from tensordict import TensorDict

from agilerl.components.data import Transition
from agilerl.modules import EvolvableModule
from agilerl.typing import NumpyObsType, TorchObsType


def assert_state_dicts_equal(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Compare two PyTorch state dictionaries using torch.allclose for efficient comparison.

    :param state_dict1: First state dictionary
    :type state_dict1: Dict[str, torch.Tensor]
    :param state_dict2: Second state dictionary
    :type state_dict2: Dict[str, torch.Tensor]
    :param rtol: Relative tolerance for torch.allclose
    :type rtol: float
    :param atol: Absolute tolerance for torch.allclose
    :type atol: float
    """
    # First check that they have the same keys
    assert set(state_dict1.keys()) == set(
        state_dict2.keys()
    ), f"State dict keys don't match: {set(state_dict1.keys())} vs {set(state_dict2.keys())}"

    # Then check each tensor
    for key in state_dict1:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
            if tensor1.device != tensor2.device:
                tensor1 = tensor1.cpu()
                tensor2 = tensor2.cpu()

            assert (
                tensor1.shape == tensor2.shape
            ), f"Tensors for key '{key}' have different shapes: {tensor1.shape} != {tensor2.shape}"
            assert torch.allclose(
                tensor1, tensor2, rtol=rtol, atol=atol
            ), f"Tensors for key '{key}' are not close enough"


def assert_not_equal_state_dict(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Compare two PyTorch state dictionaries using torch.allclose for efficient comparison.

    :param state_dict1: First state dictionary
    :type state_dict1: Dict[str, torch.Tensor]
    :param state_dict2: Second state dictionary
    :type state_dict2: Dict[str, torch.Tensor]
    :param rtol: Relative tolerance for torch.allclose
    :type rtol: float
    :param atol: Absolute tolerance for torch.allclose
    :type atol: float
    """
    try:
        assert_state_dicts_equal(state_dict1, state_dict2, rtol, atol)
    except AssertionError:
        return

    raise AssertionError(f"State dicts are equal: {state_dict1} == {state_dict2}")


def check_equal_params_ind(
    before_ind: Union[nn.Module, EvolvableModule],
    mutated_ind: Union[nn.Module, EvolvableModule],
) -> None:
    before_dict = dict(before_ind.named_parameters())
    after_dict = mutated_ind.named_parameters()
    for key, param in after_dict:
        if key in before_dict:
            old_param = before_dict[key]
            old_size = old_param.data.size()
            new_size = param.data.size()
            if old_size == new_size:
                # If the sizes are the same, just copy the parameter
                param.data = old_param.data
            elif "norm" not in key:
                # Create a slicing index to handle tensors with varying sizes
                slice_index = tuple(
                    slice(0, min(o, n)) for o, n in zip(old_size, new_size)
                )
                assert torch.all(
                    torch.eq(param.data[slice_index], old_param.data[slice_index])
                ), f"Parameter {key} not equal after mutation {mutated_ind.last_mutation_attr}:\n{param.data[slice_index]}\n{old_param.data[slice_index]}"


def unpack_network(model: nn.Sequential) -> List[nn.Module]:
    """Unpacks an nn.Sequential type model"""
    layer_list = []
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            # If it's an nn.Sequential, recursively unpack its children
            layer_list.extend(unpack_network(layer))
        else:
            if isinstance(layer, nn.Flatten):
                pass
            else:
                layer_list.append(layer)

    return layer_list


def check_models_same(model1: nn.Module, model2: nn.Module) -> bool:
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def generate_random_box_space(
    shape: Tuple[int, ...], low: Optional[Number] = None, high: Optional[Number] = None
) -> spaces.Box:
    return spaces.Box(
        low=random.randint(0, 5) if low is None else low,
        high=random.randint(6, 10) if high is None else high,
        shape=shape,
        dtype=np.float32,
    )


def generate_discrete_space(n: int) -> spaces.Discrete:
    return spaces.Discrete(n)


def generate_multidiscrete_space(n: int, m: int) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([n] * m)


def generate_dict_or_tuple_space(
    n_image: int,
    n_vector: int,
    image_shape: Tuple[int, ...] = (3, 32, 32),
    vector_shape: Tuple[int] = (4,),
    dict_space: Optional[bool] = True,
) -> Union[spaces.Dict, spaces.Tuple]:

    if dict_space is None:
        dict_space = True if random.random() < 0.5 else False

    image_spaces = [
        generate_random_box_space(image_shape, low=0, high=255) for _ in range(n_image)
    ]
    vector_spaces = [
        generate_random_box_space(vector_shape, low=-1, high=1) for _ in range(n_vector)
    ]

    if dict_space:
        image_spaces = {f"image_{i}": space for i, space in enumerate(image_spaces)}
        vector_spaces = {f"vector_{i}": space for i, space in enumerate(vector_spaces)}
        return spaces.Dict(image_spaces | vector_spaces)

    return spaces.Tuple(image_spaces + vector_spaces)


def generate_multi_agent_box_spaces(
    n_agents: int,
    shape: Tuple[int, ...],
    low: Optional[Union[Number, List[Number]]] = -1,
    high: Optional[Union[Number, List[Number]]] = 1,
) -> List[spaces.Box]:
    if isinstance(low, list):
        assert len(low) == n_agents
    if isinstance(high, list):
        assert len(high) == n_agents

    spaces = []
    for i in range(n_agents):
        if isinstance(low, list):
            _low = low[i]
        else:
            _low = low
        if isinstance(high, list):
            _high = high[i]
        else:
            _high = high

        spaces.append(generate_random_box_space(shape, _low, _high))

    return spaces


def generate_multi_agent_discrete_spaces(
    n_agents: int, m: int
) -> List[spaces.Discrete]:
    return [generate_discrete_space(m) for _ in range(n_agents)]


def generate_multi_agent_multidiscrete_spaces(
    n_agents: int, m: int
) -> List[spaces.MultiDiscrete]:
    return [generate_multidiscrete_space(m, m) for _ in range(n_agents)]


def gen_multi_agent_dict_or_tuple_spaces(
    n_agents: int,
    n_image: int,
    n_vector: int,
    image_shape: Tuple[int, ...] = (3, 16, 16),
    vector_shape: Tuple[int] = (4,),
    dict_space: Optional[bool] = False,
) -> List[Union[spaces.Dict, spaces.Tuple]]:
    return [
        generate_dict_or_tuple_space(
            n_image, n_vector, image_shape, vector_shape, dict_space
        )
        for _ in range(n_agents)
    ]


def get_sample_from_space(
    space: spaces.Space,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> NumpyObsType:
    """
    Generate a sample from the given space.

    :param space: The space to generate a sample from.
    :type space: spaces.Space
    :param batch_size: The batch size.
    :type batch_size: int
    :return: A sample from the space.
    :rtype: NumpyObsType
    """
    if isinstance(space, spaces.Box):
        if batch_size is None:
            return np.random.uniform(low=space.low, high=space.high, size=space.shape)
        else:
            return np.random.uniform(
                low=space.low, high=space.high, size=(batch_size, *space.shape)
            )
    elif isinstance(space, spaces.Discrete):
        if batch_size is None:
            return np.random.randint(space.n, size=(1,))
        else:
            return np.random.randint(space.n, size=(batch_size, 1))
    elif isinstance(space, spaces.MultiDiscrete):
        if batch_size is None:
            return np.random.randint(space.nvec, size=(len(space.nvec),))
        else:
            return np.random.randint(space.nvec, size=(batch_size, len(space.nvec)))
    elif isinstance(space, spaces.Dict):
        return {
            key: get_sample_from_space(value, batch_size)
            for key, value in space.items()
        }
    elif isinstance(space, spaces.Tuple):
        return tuple(get_sample_from_space(value, batch_size) for value in space)
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


def is_processed_observation(observation: TorchObsType, space: spaces.Space) -> bool:
    if isinstance(space, spaces.Box):
        print(observation.shape, space.shape)
        return (
            isinstance(observation, torch.Tensor)
            and observation.shape[1:] == space.shape
        )
    elif isinstance(space, spaces.Discrete):
        return isinstance(observation, torch.Tensor) and observation.shape[1:] == (1,)
    elif isinstance(space, spaces.MultiDiscrete):
        return isinstance(observation, torch.Tensor) and observation.shape[1:] == (
            len(space.nvec),
        )
    elif isinstance(space, spaces.Dict):
        return isinstance(observation, dict) and all(
            is_processed_observation(observation[key], space[key])
            for key in space.keys()
        )
    elif isinstance(space, spaces.Tuple):
        return isinstance(observation, tuple) and all(
            is_processed_observation(value, space[i])
            for i, value in enumerate(observation)
        )
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


def get_experiences_batch(
    observation_space: spaces.Space,
    action_space: spaces.Space,
    batch_size: int,
    device: Optional[torch.device] = None,
) -> TensorDict:
    """
    Generate a batch of experiences from the observation and action spaces.

    :param observation_space: The observation space.
    :type observation_space: spaces.Space
    :param action_space: The action space.
    :type action_space: spaces.Space
    :param batch_size: The batch size.
    :type batch_size: int
    :param device: The device to run the experiences on.
    :type device: torch.device
    :return: A batch of experiences.
    :rtype: TensorDict
    """
    device = device if device is not None else "cpu"
    states = get_sample_from_space(observation_space, batch_size)
    actions = get_sample_from_space(action_space, batch_size)
    rewards = torch.randn((batch_size, 1))
    next_states = get_sample_from_space(observation_space, batch_size)
    dones = torch.randint(0, 2, (batch_size, 1))
    return Transition(
        obs=states,
        action=actions,
        reward=rewards,
        next_obs=next_states,
        done=dones,
        batch_size=[batch_size],
        device=device,
    ).to_tensordict()


def assert_close_dict(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    for key, value in before.items():
        if isinstance(value, dict):
            assert_close_dict(value, after[key])
        elif isinstance(value, torch.Tensor):
            assert torch.allclose(
                value, after[key]
            ), f"Value not close: {value} != {after[key]}"
        else:
            assert value == after[key], f"Value not equal: {value} != {after[key]}"
