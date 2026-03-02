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


def test_evolvable_distribution_get_distribution_for_all_supported_spaces():
    box_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
    box_net = _DummyModule(4, 2)
    ed_box = EvolvableDistribution(action_space=box_space, network=box_net)
    dist_box = ed_box.get_distribution(torch.randn(3, 2))
    assert isinstance(dist_box, TorchDistribution)
    assert dist_box.mu.shape == (3, 2)
    assert dist_box.log_std.shape == (3, 2)
    assert ed_box.net_config == box_net.net_config

    md_space = spaces.MultiDiscrete([2, 3])
    md_net = _DummyModule(4, 5)
    ed_md = EvolvableDistribution(action_space=md_space, network=md_net)
    dist_md = ed_md.get_distribution(torch.randn(3, 5))
    assert isinstance(dist_md, TorchDistribution)

    mb_space = spaces.MultiBinary(4)
    mb_net = _DummyModule(4, 4)
    ed_mb = EvolvableDistribution(action_space=mb_space, network=mb_net)
    dist_mb = ed_mb.get_distribution(torch.randn(3, 4))
    assert isinstance(dist_mb, TorchDistribution)


def test_evolvable_distribution_log_prob_entropy_require_forward():
    action_space = spaces.Discrete(3)
    net = _DummyModule(4, 3)
    ed = EvolvableDistribution(action_space=action_space, network=net)

    with pytest.raises(ValueError, match="Distribution not initialized"):
        ed.log_prob(torch.tensor([0]))
    with pytest.raises(ValueError, match="Distribution not initialized"):
        ed.entropy()


def test_evolvable_distribution_log_prob_entropy_after_forward():
    action_space = spaces.Discrete(3)
    net = _DummyModule(4, 3)
    ed = EvolvableDistribution(action_space=action_space, network=net)
    latent = torch.randn(2, 4)
    action, _, _ = ed(latent, sample=True)
    log_prob = ed.log_prob(action)
    entropy = ed.entropy()
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)


def test_evolvable_distribution_apply_mask_multidiscrete_and_multibinary():
    md_space = spaces.MultiDiscrete([2, 3])
    md_net = _DummyModule(4, 5)
    ed_md = EvolvableDistribution(action_space=md_space, network=md_net)
    md_logits = torch.randn(2, 5)
    md_mask = torch.tensor([[1, 0, 1, 1, 0], [1, 1, 0, 1, 1]], dtype=torch.bool)
    md_masked = ed_md.apply_mask(md_logits, md_mask)
    assert md_masked.shape == md_logits.shape
    assert torch.all(md_masked[~md_mask] < -1e7)

    mb_space = spaces.MultiBinary(4)
    mb_net = _DummyModule(4, 4)
    ed_mb = EvolvableDistribution(action_space=mb_space, network=mb_net)
    mb_logits = torch.randn(2, 4)
    mb_mask = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 0]], dtype=torch.bool)
    mb_masked = ed_mb.apply_mask(mb_logits, mb_mask)
    assert mb_masked.shape == mb_logits.shape
    assert torch.all(mb_masked[~mb_mask] < -1e7)


def test_evolvable_distribution_forward_sample_false_returns_none_action():
    action_space = spaces.Discrete(3)
    net = _DummyModule(4, 3)
    ed = EvolvableDistribution(action_space=action_space, network=net)
    latent = torch.randn(2, 4)
    action, log_prob, entropy = ed(latent, sample=False)
    assert action is None
    assert log_prob is None
    assert entropy.shape == (2,)


def test_evolvable_distribution_forward_mask_stack_failure_raises():
    action_space = spaces.Discrete(3)
    net = _DummyModule(4, 3)
    ed = EvolvableDistribution(action_space=action_space, network=net)
    latent = torch.randn(2, 4)
    bad_mask = [np.array([1, 0, 1]), np.array([1, 0])]  # non-uniform lengths
    with pytest.raises(Exception):
        ed(latent, action_mask=bad_mask)


def test_evolvable_distribution_clone():
    action_space = spaces.Discrete(3)
    net = _DummyModule(4, 3)
    ed = EvolvableDistribution(action_space=action_space, network=net)
    clone = ed.clone()
    assert isinstance(clone, EvolvableDistribution)
    assert clone.action_space == ed.action_space
    assert clone.action_std_init == ed.action_std_init
