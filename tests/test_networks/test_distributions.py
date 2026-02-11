"""Tests for TorchDistribution and distribution dispatch (torch-primitive based)."""

import pytest
import torch
from gymnasium import spaces

from agilerl.networks.distributions import TorchDistribution


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
