from typing import List, Tuple, Optional, Union
from numbers import Number
import random
import torch.nn as nn
import numpy as np
from gymnasium import spaces

def unpack_network(model):
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


def check_models_same(model1, model2):
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