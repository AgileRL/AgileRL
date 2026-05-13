import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from agilerl.utils.torch_utils import (
    entropy_from_space,
    get_transformer_logs,
    log_prob_discrete,
    log_prob_from_space,
    map_pytree,
    parameter_norm,
    sample_from_space,
    to_decorator,
)


class TestMapPytree:
    # Should return the same input if it's not a dict, list, set, tuple, np.ndarray or torch.Tensor
    def test_return_same_input(self):
        input_data = 5
        result = map_pytree(lambda x: x, input_data)
        assert result == input_data

    # Should apply the function f to each np.ndarray or torch.Tensor in the input
    def test_apply_function_to_tensors(self):
        input_data = [np.array([1, 2, 3]), torch.tensor([4, 5, 6])]
        result = map_pytree(lambda x: x + 1, input_data)
        expected_result = [np.array([2, 3, 4]), torch.tensor([5, 6, 7])]
        assert str(result) == str(expected_result)

    # Should apply the function f to each np.ndarray or torch.Tensor in nested lists, sets, tuples, and dicts
    def test_apply_function_to_nested_structures(self):
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
    def test_raise_type_error(self):
        input_data = np.array([1, 2, 3])
        with pytest.raises(TypeError):
            map_pytree(123, input_data)

    # Should return None if the input is None
    def test_return_none_for_none_input(self):
        input_data = None
        result = map_pytree(lambda x: x + 1, input_data)
        assert result is None

    # Should return an empty dict if the input is an empty dict
    def test_return_empty_dict_for_empty_dict(self):
        input_data = {}
        result = map_pytree(lambda x: x + 1, input_data)
        assert result == {}


# The function should take in a function f and a device as input.
def test_to_decorator_function_and_device_input():
    device = torch.device("cpu")

    def f(x):
        return x + 1

    decorated_f = to_decorator(f, device)
    result = decorated_f(5)
    assert result == 6


# Should return the correct norm for a model with multiple parameters
def test_parameter_norm_multiple_parameters():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    expected_norm = 0.0
    for param in model.parameters():
        expected_norm += (param.norm() ** 2).item()
    expected_norm = math.sqrt(expected_norm)
    assert parameter_norm(model) == expected_norm


# The value of "attention_entropy" is a tuple containing the model's attention entropy and the total number of non-masked tokens.
def test_get_transformer_logs_attention_entropy_value_is_tuple_with_entropy_and_total_tokens():
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


# --------------------------------------------------------------------------- #
# Distribution dispatch (sample_from_space, log_prob_from_space, entropy_from_space)
# --------------------------------------------------------------------------- #


class TestSampleFromSpace:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_sample_from_space_discrete(self, batch_size):
        """sample_from_space with Discrete action space returns valid actions."""
        action_space = spaces.Discrete(5)
        logits = torch.randn(batch_size, 5)
        action = sample_from_space(action_space, logits=logits)
        assert action.shape == (batch_size,)
        assert action.dtype in (torch.long, torch.int64)
        assert torch.all(action >= 0)
        assert torch.all(action < 5)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_sample_from_space_box(self, batch_size):
        """sample_from_space with Box action space returns correct shape; squash clips to [-1,1]."""
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mu = torch.zeros(batch_size, 2)
        log_std = torch.zeros(batch_size, 2)
        action = sample_from_space(
            action_space, mu=mu, log_std=log_std, squash_output=True
        )
        assert action.shape == (batch_size, 2)
        assert torch.all(action >= -1.0)
        assert torch.all(action <= 1.0)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_sample_from_space_multi_discrete(self, batch_size):
        """sample_from_space with MultiDiscrete returns valid actions per dimension."""
        action_space = spaces.MultiDiscrete([3, 2, 4])
        # 3 + 2 + 4 = 9 logits
        logits = torch.randn(batch_size, 9)
        action = sample_from_space(action_space, logits=logits)
        assert action.shape == (batch_size, 3)
        assert torch.all(action[:, 0] >= 0) and torch.all(action[:, 0] < 3)
        assert torch.all(action[:, 1] >= 0) and torch.all(action[:, 1] < 2)
        assert torch.all(action[:, 2] >= 0) and torch.all(action[:, 2] < 4)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_sample_from_space_multi_binary(self, batch_size):
        """sample_from_space with MultiBinary returns 0/1 per dimension."""
        action_space = spaces.MultiBinary(4)
        logits = torch.randn(batch_size, 4)
        action = sample_from_space(action_space, logits=logits)
        assert action.shape == (batch_size, 4)
        assert torch.all((action == 0) | (action == 1))

    def test_sample_from_space_unsupported_raises(self):
        """Unsupported space type raises NotImplementedError."""
        # Tuple is not registered for the distribution dispatch
        action_space = spaces.Tuple((spaces.Discrete(2),))
        with pytest.raises(NotImplementedError, match="Unsupported action space"):
            sample_from_space(action_space, logits=torch.randn(1, 2))


class TestLogProbFromSpace:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_log_prob_from_space_discrete(self, batch_size):
        """log_prob_from_space with Discrete: shape and finite values."""
        action_space = spaces.Discrete(5)
        logits = torch.randn(batch_size, 5)
        action = torch.randint(0, 5, (batch_size,))
        lp = log_prob_from_space(action_space, action, logits=logits)
        assert lp.shape == (batch_size,)
        assert torch.all(torch.isfinite(lp))
        assert torch.all(lp <= 0)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_log_prob_from_space_box(self, batch_size):
        """log_prob_from_space with Box: shape and finite values."""
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mu = torch.zeros(batch_size, 2)
        log_std = torch.zeros(batch_size, 2)
        action = torch.randn(batch_size, 2)
        lp = log_prob_from_space(action_space, action, mu=mu, log_std=log_std)
        assert lp.shape == (batch_size,)
        assert torch.all(torch.isfinite(lp))

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_log_prob_from_space_multi_discrete(self, batch_size):
        """log_prob_from_space with MultiDiscrete: shape and finite values."""
        action_space = spaces.MultiDiscrete([3, 2, 4])
        logits = torch.randn(batch_size, 9)
        action = torch.stack(
            [
                torch.randint(0, 3, (batch_size,)),
                torch.randint(0, 2, (batch_size,)),
                torch.randint(0, 4, (batch_size,)),
            ],
            dim=-1,
        )
        lp = log_prob_from_space(action_space, action, logits=logits)
        assert lp.shape == (batch_size,)
        assert torch.all(torch.isfinite(lp))

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_log_prob_from_space_multi_binary(self, batch_size):
        """log_prob_from_space with MultiBinary: shape and finite values."""
        action_space = spaces.MultiBinary(4)
        logits = torch.randn(batch_size, 4)
        action = torch.randint(0, 2, (batch_size, 4)).float()
        lp = log_prob_from_space(action_space, action, logits=logits)
        assert lp.shape == (batch_size,)
        assert torch.all(torch.isfinite(lp))

    def test_distribution_roundtrip_discrete(self):
        """Sample then log_prob returns finite log prob for Discrete."""
        action_space = spaces.Discrete(5)
        logits = torch.randn(4, 5)
        action = sample_from_space(action_space, logits=logits)
        lp = log_prob_from_space(action_space, action, logits=logits)
        assert lp.shape == (4,)
        assert torch.all(torch.isfinite(lp))

    def test_log_prob_from_space_unsupported_raises(self):
        """Unsupported space type raises NotImplementedError for log_prob_from_space."""
        action_space = spaces.Tuple((spaces.Discrete(2),))
        with pytest.raises(NotImplementedError, match="Unsupported action space"):
            log_prob_from_space(
                action_space, torch.randn(1, 2), logits=torch.randn(1, 2)
            )


class TestEntropyFromSpace:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_entropy_from_space_discrete(self, batch_size):
        """entropy_from_space with Discrete: positive and bounded by log(n)."""
        action_space = spaces.Discrete(5)
        logits = torch.randn(batch_size, 5)
        ent = entropy_from_space(action_space, logits=logits)
        assert ent.shape == (batch_size,)
        assert torch.all(ent > 0)
        assert torch.all(ent <= math.log(5) + 0.01)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_entropy_from_space_box(self, batch_size):
        """entropy_from_space with Box: positive and finite."""
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mu = torch.zeros(batch_size, 2)
        log_std = torch.zeros(batch_size, 2)
        ent = entropy_from_space(action_space, mu=mu, log_std=log_std)
        assert ent.shape == (batch_size,)
        assert torch.all(torch.isfinite(ent))
        assert torch.all(ent > 0)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_entropy_from_space_multi_discrete(self, batch_size):
        """entropy_from_space with MultiDiscrete: positive."""
        action_space = spaces.MultiDiscrete([3, 2, 4])
        logits = torch.randn(batch_size, 9)
        ent = entropy_from_space(action_space, logits=logits)
        assert ent.shape == (batch_size,)
        assert torch.all(ent > 0)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_entropy_from_space_multi_binary(self, batch_size):
        """entropy_from_space with MultiBinary: positive."""
        action_space = spaces.MultiBinary(4)
        logits = torch.randn(batch_size, 4)
        ent = entropy_from_space(action_space, logits=logits)
        assert ent.shape == (batch_size,)
        assert torch.all(ent > 0)

    def test_entropy_from_space_unsupported_raises(self):
        """Unsupported space type raises NotImplementedError for entropy_from_space."""
        action_space = spaces.Tuple((spaces.Discrete(2),))
        with pytest.raises(NotImplementedError, match="Unsupported action space"):
            entropy_from_space(action_space, logits=torch.randn(1, 2))


class TestLogProbDiscrete:
    def test_log_prob_discrete_action_same_ndim_trailing_one(self):
        """Action with same ndim as logits and shape[-1]==1 is treated as indices."""
        logits = torch.randn(4, 5)
        action = torch.randint(0, 5, (4, 1))
        lp = log_prob_discrete(logits, action)
        assert lp.shape == (4,)
        assert torch.all(torch.isfinite(lp))

    def test_log_prob_discrete_one_hot_action(self):
        """One-hot encoded action is decoded via argmax when n_actions is provided."""
        logits = torch.randn(4, 5)
        action_idx = torch.randint(0, 5, (4,))
        one_hot = torch.zeros(4, 5)
        one_hot.scatter_(1, action_idx.unsqueeze(-1), 1)
        lp = log_prob_discrete(logits, one_hot, n_actions=5)
        assert lp.shape == (4,)
        assert torch.all(torch.isfinite(lp))
        lp_ref = log_prob_discrete(logits, action_idx)
        torch.testing.assert_close(lp, lp_ref)

    def test_log_prob_discrete_same_ndim_invalid_shape_raises(self):
        """Same ndim but incompatible last dim raises ValueError."""
        logits = torch.randn(4, 5)
        action = torch.randint(0, 5, (4, 3))
        with pytest.raises(ValueError, match="not compatible with Discrete space"):
            log_prob_discrete(logits, action)

    def test_log_prob_discrete_wrong_ndim_raises(self):
        """Completely wrong ndim raises ValueError."""
        logits = torch.randn(4, 5)
        action = torch.randint(0, 5, (2, 4, 1))
        with pytest.raises(ValueError, match="not compatible with logits ndim"):
            log_prob_discrete(logits, action)
