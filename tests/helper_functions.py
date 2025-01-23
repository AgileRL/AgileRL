from typing import List, Tuple, Optional, Union
from numbers import Number
import random
import torch.nn as nn
import torch
import numpy as np
from gymnasium import spaces

from agilerl.protocols import EvolvableAlgorithm, EvolvableModule

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
        shape: Tuple[int, ...],
        low: Optional[Number] = None,
        high: Optional[Number] = None
        ) -> spaces.Box:
    return spaces.Box(
        low=random.randint(0, 5) if low is None else low,
        high=random.randint(6, 10) if high is None else high,
        shape=shape,
        dtype=np.float32
    )

def generate_discrete_space(n: int) -> spaces.Discrete:
    return spaces.Discrete(n)

def generate_multidiscrete_space(n: int, m: int) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([n] * m)

def generate_dict_or_tuple_space(
        n_image: int,
        n_vector: int,
        image_shape: Tuple[int, ...] = (3, 128, 128),
        vector_shape: Tuple[int] = (5,),
        dict_space: Optional[bool] = False
        ) -> Union[spaces.Dict, spaces.Tuple]:

    if dict_space is None:
        dict_space = True if random.random() < 0.5 else False
    
    image_spaces = [generate_random_box_space(image_shape) for _ in range(n_image)]
    vector_spaces = [generate_random_box_space(vector_shape) for _ in range(n_vector)]

    if dict_space:
        image_spaces = {"image_{}".format(i): space for i, space in enumerate(image_spaces)}
        vector_spaces = {"vector_{}".format(i): space for i, space in enumerate(vector_spaces)}
        return spaces.Dict(image_spaces | vector_spaces)

    return spaces.Tuple(image_spaces + vector_spaces)

def generate_multi_agent_box_spaces(
        n_agents: int,
        shape: Tuple[int, ...],
        low: Optional[Union[Number, List[Number]]] = -1,
        high: Optional[Union[Number, List[Number]]] = 1
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

def generate_multi_agent_discrete_spaces(n_agents: int, m: int) -> List[spaces.Discrete]:
    return [generate_discrete_space(m) for _ in range(n_agents)]

def gen_multi_agent_dict_or_tuple_spaces(
        n_agents: int,
        n_image: int,
        n_vector: int,
        image_shape: Tuple[int, ...] = (3, 128, 128),
        vector_shape: Tuple[int] = (5,),
        dict_space: Optional[bool] = False
        ) -> List[Union[spaces.Dict, spaces.Tuple]]:
    return [generate_dict_or_tuple_space(n_image, n_vector, image_shape, vector_shape, dict_space) for _ in range(n_agents)]

def check_equal_params_ind(before_ind:  Union[nn.Module, EvolvableModule], mutated_ind: Union[nn.Module, EvolvableModule]) -> None:
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
                slice_index = tuple(slice(0, min(o, n)) for o, n in zip(old_size, new_size))
                print(mutated_ind.last_mutation_attr)
                assert (
                    torch.all(torch.eq(param.data[slice_index], old_param.data[slice_index]))), \
                    f"Parameter {key} not equal after mutation {mutated_ind.last_mutation_attr}:\n{param.data[slice_index]}\n{old_param.data[slice_index]}"

def assert_equal_state_dict(before_pop: List[EvolvableAlgorithm], mutated_pop: List[EvolvableAlgorithm]) -> None:
    not_eq = []
    for before_ind, mutated in zip(before_pop, mutated_pop):
        before_modules = before_ind.evolvable_attributes(networks_only=True).values()
        mutated_modules = mutated.evolvable_attributes(networks_only=True).values()
        for before_mod, mutated_mod in zip(before_modules, mutated_modules):
            if isinstance(before_mod, list):
                for before, mutated in zip(before_mod, mutated_mod):
                    check_equal_params_ind(before, mutated)
            else:
                check_equal_params_ind(before_mod, mutated_mod)
    
    assert not not_eq, f"Parameters not equal: {not_eq}"

def assert_close_dict(before: dict, after: dict) -> None:
    for key, value in before.items():
        if isinstance(value, dict):
            assert_close_dict(value, after[key])
        elif isinstance(value, torch.Tensor):
            assert torch.allclose(value, after[key]), f"Value not close: {value} != {after[key]}"
        else:
            assert value == after[key], f"Value not equal: {value} != {after[key]}"