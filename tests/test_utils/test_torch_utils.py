import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from agilerl.utils.torch_utils import (
    get_transformer_logs,
    map_pytree,
    parameter_norm,
    to_decorator,
)


# Should return the same input if it's not a dict, list, set, tuple, np.ndarray or torch.Tensor
def test_return_same_input():
    input_data = 5
    result = map_pytree(lambda x: x, input_data)
    assert result == input_data


# Should apply the function f to each np.ndarray or torch.Tensor in the input
def test_apply_function_to_tensors():
    input_data = [np.array([1, 2, 3]), torch.tensor([4, 5, 6])]
    result = map_pytree(lambda x: x + 1, input_data)
    expected_result = [np.array([2, 3, 4]), torch.tensor([5, 6, 7])]
    assert str(result) == str(expected_result)


# Should apply the function f to each np.ndarray or torch.Tensor in nested lists, sets, tuples, and dicts
def test_apply_function_to_nested_structures():
    input_data = {
        "a": [np.array([1, 2]), torch.tensor([3, 4])],
        "b": (np.array([5, 6]), torch.tensor([7, 8])),
    }
    result = map_pytree(lambda x: x * 2, input_data)
    expected_result = {
        "a": [np.array([2, 4]), torch.tensor([6, 8])],
        "b": [np.array([10, 12]), torch.tensor([14, 16])],
    }
    assert str(result) == str(expected_result)


# Should raise a TypeError if f is not a callable
def test_raise_type_error():
    input_data = np.array([1, 2, 3])
    with pytest.raises(TypeError):
        map_pytree(123, input_data)


# Should return None if the input is None
def test_return_none_for_none_input():
    input_data = None
    result = map_pytree(lambda x: x + 1, input_data)
    assert result is None


# Should return an empty dict if the input is an empty dict
def test_return_empty_dict_for_empty_dict():
    input_data = {}
    result = map_pytree(lambda x: x + 1, input_data)
    assert result == {}


# The function should take in a function f and a device as input.
def test_function_and_device_input():
    device = torch.device("cpu")

    def f(x):
        return x + 1

    decorated_f = to_decorator(f, device)
    result = decorated_f(5)
    assert result == 6


# Should return the correct norm for a model with multiple parameters
def test_multiple_parameters():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    expected_norm = 0.0
    for param in model.parameters():
        expected_norm += (param.norm() ** 2).item()
    expected_norm = math.sqrt(expected_norm)
    assert parameter_norm(model) == expected_norm


# The value of "attention_entropy" is a tuple containing the model's attention entropy and the total number of non-masked tokens.
def test_attention_entropy_value_is_tuple_with_entropy_and_total_tokens():
    attentions = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5, 0.6])]
    model = nn.Module()
    attn_mask = torch.tensor([1, 1, 1])

    logs = get_transformer_logs(attentions, model, attn_mask)

    attention_entropy = logs["attention_entropy"]
    assert isinstance(attention_entropy, tuple)
    assert len(attention_entropy) == 2
    assert isinstance(attention_entropy[0].item(), float)
    assert isinstance(attention_entropy[1].item(), int)

    parameter_norm = logs["parameter_norm"]
    assert isinstance(parameter_norm, tuple)
    assert len(parameter_norm) == 2
    assert isinstance(parameter_norm[0], float)
    assert isinstance(parameter_norm[1], int)
