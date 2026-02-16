"""Tests for agilerl.networks.distributions."""

import pytest
import torch
from gymnasium import spaces
from torch.distributions import Beta, Bernoulli, Categorical, Normal

from agilerl.modules import EvolvableMLP
from agilerl.networks.distributions import (
    BernoulliHandler,
    CategoricalHandler,
    DistributionHandler,
    EvolvableDistribution,
    MultiCategoricalHandler,
    NormalHandler,
    TorchDistribution,
)


# ---------------------------------------------------------------------------
# ValueErrors when self.dist is None (EvolvableDistribution)
# ---------------------------------------------------------------------------


def test_evolvable_distribution_log_prob_raises_when_dist_is_none(
    vector_space, discrete_space
):
    """EvolvableDistribution.log_prob raises ValueError when self.dist is None."""
    network = EvolvableMLP(4, discrete_space.n, hidden_size=[8])
    evo = EvolvableDistribution(
        action_space=discrete_space,
        network=network,
        device="cpu",
    )
    assert evo.dist is None
    with pytest.raises(
        ValueError, match="Distribution not initialized. Call forward first."
    ):
        evo.log_prob(torch.zeros(1, 1, dtype=torch.long))


def test_evolvable_distribution_entropy_raises_when_dist_is_none(
    vector_space, discrete_space
):
    """EvolvableDistribution.entropy raises ValueError when self.dist is None."""
    network = EvolvableMLP(4, discrete_space.n, hidden_size=[8])
    evo = EvolvableDistribution(
        action_space=discrete_space,
        network=network,
        device="cpu",
    )
    assert evo.dist is None
    with pytest.raises(
        ValueError, match="Distribution not initialized. Call forward first."
    ):
        evo.entropy()


# ---------------------------------------------------------------------------
# NotImplementedError when action space or Distribution is not supported
# ---------------------------------------------------------------------------


def test_get_distribution_raises_for_unsupported_action_space(vector_space):
    """EvolvableDistribution.get_distribution raises NotImplementedError for unsupported action space."""
    # spaces.Dict is not supported (only Box, Discrete, MultiDiscrete, MultiBinary are)
    action_space = spaces.Dict({"a": spaces.Discrete(2)})
    network = EvolvableMLP(4, 2, hidden_size=[8])
    evo = EvolvableDistribution(
        action_space=action_space,
        network=network,
        device="cpu",
    )
    logits = torch.randn(1, 2)
    with pytest.raises(NotImplementedError, match="Action space .* not supported."):
        evo.get_distribution(logits)


def test_apply_mask_raises_for_unsupported_action_space(vector_space):
    """EvolvableDistribution.apply_mask raises NotImplementedError for unsupported action space (e.g. Box)."""
    # apply_mask supports Discrete, MultiDiscrete, MultiBinary; Box falls through to else
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
    network = EvolvableMLP(4, 2, hidden_size=[8])
    evo = EvolvableDistribution(
        action_space=action_space,
        network=network,
        device="cpu",
    )
    logits = torch.randn(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="Action space .* not supported."):
        evo.apply_mask(logits, mask)


def test_torch_distribution_raises_for_unsupported_distribution_type():
    """TorchDistribution raises NotImplementedError when Distribution type is not supported."""
    # Beta is not in _handlers (only Normal, Bernoulli, Categorical, list are)
    dist = Beta(
        concentration1=torch.ones(1, 2),
        concentration0=torch.ones(1, 2),
    )
    with pytest.raises(NotImplementedError, match="Distribution .* not supported."):
        TorchDistribution(dist)


# ---------------------------------------------------------------------------
# DistributionHandler protocol – handlers implement the protocol
# ---------------------------------------------------------------------------


def test_normal_handler_satisfies_distribution_handler_protocol():
    """NormalHandler implements DistributionHandler protocol (sample, log_prob, entropy)."""
    handler = NormalHandler()
    dist = Normal(loc=torch.zeros(2, 3), scale=torch.ones(2, 3))
    # Protocol: sample(distribution) -> Tensor
    sample = handler.sample(dist)
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (2, 3)
    # Protocol: log_prob(distribution, action) -> Tensor
    log_prob = handler.log_prob(dist, sample)
    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.shape == (2,)
    # Protocol: entropy(distribution) -> Tensor | None
    ent = handler.entropy(dist)
    assert isinstance(ent, torch.Tensor)
    assert ent.shape == (2,)


def test_bernoulli_handler_satisfies_distribution_handler_protocol():
    """BernoulliHandler implements DistributionHandler protocol."""
    handler = BernoulliHandler()
    dist = Bernoulli(logits=torch.zeros(2, 4))
    sample = handler.sample(dist)
    assert isinstance(sample, torch.Tensor)
    log_prob = handler.log_prob(dist, sample)
    assert isinstance(log_prob, torch.Tensor)
    ent = handler.entropy(dist)
    assert isinstance(ent, torch.Tensor)


def test_categorical_handler_satisfies_distribution_handler_protocol():
    """CategoricalHandler implements DistributionHandler protocol."""
    handler = CategoricalHandler()
    dist = Categorical(logits=torch.randn(2, 3))
    sample = handler.sample(dist)
    assert isinstance(sample, torch.Tensor)
    log_prob = handler.log_prob(dist, sample)
    assert isinstance(log_prob, torch.Tensor)
    ent = handler.entropy(dist)
    assert isinstance(ent, torch.Tensor)


def test_multi_categorical_handler_satisfies_distribution_handler_protocol():
    """MultiCategoricalHandler implements DistributionHandler protocol (list of Categorical)."""
    handler = MultiCategoricalHandler()
    dist_list = [
        Categorical(logits=torch.randn(2, 2)),
        Categorical(logits=torch.randn(2, 3)),
    ]
    sample = handler.sample(dist_list)
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (2, 2)
    log_prob = handler.log_prob(dist_list, sample)
    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.shape == (2,)
    ent = handler.entropy(dist_list)
    assert isinstance(ent, torch.Tensor)


def test_distribution_handler_protocol_accepts_all_handlers():
    """DistributionHandler protocol: function accepting DistributionHandler works with all handler types."""
    from torch.distributions import Distribution

    def use_handler(
        handler: DistributionHandler,
        distribution: Distribution | list,
        action: torch.Tensor | None = None,
    ):
        sample = handler.sample(distribution)
        if action is None:
            action = sample
        log_prob = handler.log_prob(distribution, action)
        entropy = handler.entropy(distribution)
        return sample, log_prob, entropy

    # NormalHandler
    dist_n = Normal(loc=torch.zeros(1, 2), scale=torch.ones(1, 2))
    s, lp, e = use_handler(NormalHandler(), dist_n)
    assert s.shape == (1, 2)
    assert lp.shape == (1,)
    assert e.shape == (1,)

    # BernoulliHandler
    dist_b = Bernoulli(logits=torch.zeros(1, 2))
    s, lp, e = use_handler(BernoulliHandler(), dist_b)
    assert s.shape == (1, 2)

    # CategoricalHandler
    dist_c = Categorical(logits=torch.randn(1, 3))
    s, lp, e = use_handler(CategoricalHandler(), dist_c)
    assert s.shape == (1,)

    # MultiCategoricalHandler
    dist_m = [
        Categorical(logits=torch.randn(1, 2)),
        Categorical(logits=torch.randn(1, 2)),
    ]
    s, lp, e = use_handler(MultiCategoricalHandler(), dist_m)
    assert s.shape == (1, 2)
