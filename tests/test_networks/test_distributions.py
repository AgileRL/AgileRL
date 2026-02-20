"""Tests for TorchDistribution and distribution dispatch (torch-primitive based)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule
from agilerl.networks.distributions import EvolvableDistribution, TorchDistribution


@pytest.mark.parametrize("batch_size", [1, 4])
def test_torch_distribution_discrete_sample_log_prob_entropy(batch_size):
    """TorchDistribution with Discrete: sample, log_prob, entropy are consistent."""
    action_space = spaces.Discrete(5)
    logits = torch.randn(batch_size, 5)
    dist = TorchDistribution(action_space=action_space, logits=logits)

    action = dist.sample()
    assert action.shape == (batch_size,)
    assert torch.all(action >= 0) and torch.all(action < 5)

    lp = dist.log_prob(action)
    assert lp.shape == (batch_size,)
    assert torch.all(torch.isfinite(lp))

    ent = dist.entropy()
    assert ent.shape == (batch_size,)
    assert torch.all(ent > 0)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_torch_distribution_box_sample_log_prob_entropy(batch_size):
    """TorchDistribution with Box: sample, log_prob, entropy; squash_output clips action."""
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
    mu = torch.zeros(batch_size, 2)
    log_std = torch.zeros(batch_size, 2)
    dist = TorchDistribution(
        action_space=action_space,
        mu=mu,
        log_std=log_std,
        squash_output=True,
    )

    action = dist.sample()
    assert action.shape == (batch_size, 2)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    lp = dist.log_prob(action)
    assert lp.shape == (batch_size,)
    assert torch.all(torch.isfinite(lp))

    ent = dist.entropy()
    assert ent.shape == (batch_size,)
    assert torch.all(torch.isfinite(ent))


@pytest.mark.parametrize("batch_size", [1, 4])
def test_torch_distribution_multi_discrete_sample_log_prob_entropy(batch_size):
    """TorchDistribution with MultiDiscrete: sample, log_prob, entropy."""
    action_space = spaces.MultiDiscrete([3, 2, 4])
    logits = torch.randn(batch_size, 9)
    dist = TorchDistribution(action_space=action_space, logits=logits)

    action = dist.sample()
    assert action.shape == (batch_size, 3)
    lp = dist.log_prob(action)
    assert lp.shape == (batch_size,)
    ent = dist.entropy()
    assert ent.shape == (batch_size,)
    assert torch.all(ent > 0)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_torch_distribution_multi_binary_sample_log_prob_entropy(batch_size):
    """TorchDistribution with MultiBinary: sample, log_prob, entropy."""
    action_space = spaces.MultiBinary(4)
    logits = torch.randn(batch_size, 4)
    dist = TorchDistribution(action_space=action_space, logits=logits)

    action = dist.sample()
    assert action.shape == (batch_size, 4)
    assert torch.all((action == 0) | (action == 1))
    lp = dist.log_prob(action)
    assert lp.shape == (batch_size,)
    ent = dist.entropy()
    assert ent.shape == (batch_size,)
    assert torch.all(ent > 0)


def test_torch_distribution_sampled_action_stored():
    """TorchDistribution stores last sampled action in _sampled_action."""
    action_space = spaces.Discrete(3)
    logits = torch.randn(2, 3)
    dist = TorchDistribution(action_space=action_space, logits=logits)
    assert dist._sampled_action is None
    action = dist.sample()
    assert dist._sampled_action is not None
    torch.testing.assert_close(dist._sampled_action, action)


# --------------------------------------------------------------------------- #
# Helper: minimal EvolvableModule for testing EvolvableDistribution
# --------------------------------------------------------------------------- #


class _DummyModule(EvolvableModule):
    """Simple linear module that returns logits of the right size."""

    def __init__(self, in_features: int, out_features: int, device: str = "cpu"):
        super().__init__(device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def recreate_network(self):
        self.linear = nn.Linear(self.in_features, self.out_features)


# --------------------------------------------------------------------------- #
# EvolvableDistribution: get_distribution unsupported space (line 212-213)
# --------------------------------------------------------------------------- #


def test_evolvable_distribution_get_distribution_unsupported_raises():
    """get_distribution raises NotImplementedError for an unsupported action space."""
    action_space = spaces.Tuple((spaces.Discrete(2),))
    net = _DummyModule(4, 2)
    ed = EvolvableDistribution(action_space=action_space, network=net)
    with pytest.raises(NotImplementedError, match="not supported"):
        ed.get_distribution(torch.randn(1, 2))


# --------------------------------------------------------------------------- #
# EvolvableDistribution: apply_mask unsupported space (lines 285-286)
# --------------------------------------------------------------------------- #


def test_evolvable_distribution_apply_mask_unsupported_raises():
    """apply_mask raises NotImplementedError for a non-discrete action space."""
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
    net = _DummyModule(4, 2)
    ed = EvolvableDistribution(action_space=action_space, network=net)
    with pytest.raises(NotImplementedError, match="not supported for masking"):
        ed.apply_mask(torch.randn(1, 2), torch.ones(1, 2, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# EvolvableDistribution: forward with list-of-arrays action_mask (lines 318-324)
# --------------------------------------------------------------------------- #


def test_evolvable_distribution_forward_list_action_mask():
    """forward handles action_mask given as a list of numpy arrays."""
    action_space = spaces.Discrete(3)
    net = _DummyModule(4, 3)
    ed = EvolvableDistribution(action_space=action_space, network=net)
    latent = torch.randn(2, 4)
    mask = [np.array([True, True, False]), np.array([False, True, True])]
    action, log_prob, entropy = ed(latent, action_mask=mask)
    assert action is not None
    assert action.shape == (2,)
    assert torch.all(torch.isfinite(log_prob))
    assert torch.all(torch.isfinite(entropy))


def test_evolvable_distribution_forward_object_array_action_mask():
    """forward handles action_mask given as a numpy object array."""
    action_space = spaces.Discrete(3)
    net = _DummyModule(4, 3)
    ed = EvolvableDistribution(action_space=action_space, network=net)
    latent = torch.randn(2, 4)
    mask = np.empty(2, dtype=object)
    mask[0] = np.array([True, True, False])
    mask[1] = np.array([False, True, True])
    action, log_prob, entropy = ed(latent, action_mask=mask)
    assert action is not None
    assert action.shape == (2,)
    assert torch.all(torch.isfinite(log_prob))
    assert torch.all(torch.isfinite(entropy))
