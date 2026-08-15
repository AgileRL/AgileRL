"""Tests for the τ-dormant neuron diagnostic.

The measurement is shared by two consumers -- the logged
``eval/best_dormant_fraction`` diagnostic and the function-preserving architecture
mutation's removals -- so these tests pin the semantics both rely on: which layers
are measured, that the scoring forward pass leaves no hooks or mode changes behind,
and how non-finite activations are treated.
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn

from agilerl.utils.dormant_neurons import (
    _target_activations,
    capture_per_neuron_scores,
    collect_observation_batch,
    dormant_neuron_fraction,
    normalised_scores,
)

LATENT_DIM = 6
HIDDEN = 8


class _TinyEncoder(nn.Module):
    """An MLP encoder shaped like ``create_mlp``'s output (naming included)."""

    def __init__(self, in_dim: int = 4) -> None:
        super().__init__()
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("feature_linear_layer_0", nn.Linear(in_dim, HIDDEN)),
                    ("feature_activation_0", nn.ReLU()),
                    ("feature_linear_layer_output", nn.Linear(HIDDEN, LATENT_DIM)),
                    ("feature_activation_output", nn.Identity()),
                ]
            )
        )

    def forward(self, x):
        return self.model(x)


class _TinyHead(nn.Module):
    def __init__(self, out_dim: int = 2) -> None:
        super().__init__()
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("head_linear_layer_0", nn.Linear(LATENT_DIM, HIDDEN)),
                    ("head_activation_0", nn.ReLU()),
                    ("head_linear_layer_output", nn.Linear(HIDDEN, out_dim)),
                    ("head_activation_output", nn.Identity()),
                ]
            )
        )

    def forward(self, x):
        return self.model(x)


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TinyEncoder()
        self.head_net = _TinyHead()

    def forward(self, x):
        return self.head_net(self.encoder(x))


class _Group:
    def __init__(self, name: str) -> None:
        self.eval_network = name


class _Registry:
    def __init__(self, names: list[str]) -> None:
        self.groups = [_Group(name) for name in names]


class _FakeAgent:
    """The minimal surface the diagnostic needs: registry groups + eval networks."""

    def __init__(self, **networks: nn.Module) -> None:
        for name, network in networks.items():
            setattr(self, name, network)
        self.registry = _Registry(list(networks))

    def preprocess_observation(self, obs, network_ids=None):
        return obs


def _hook_count(net: nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in net.modules())


class TestTargetActivations:
    """Which layers the diagnostic measures."""

    def test_measures_hidden_layers_and_the_encoder_output(self):
        net = _TinyNet()
        # encoder: hidden + latent output; head: hidden only (its output units have
        # fixed semantics -- logits / state value).
        assert _target_activations(net) == [
            net.encoder.model.feature_activation_0,
            net.encoder.model.feature_activation_output,
            net.head_net.model.head_activation_0,
        ]

    def test_no_measured_layers_yields_nothing(self):
        assert capture_per_neuron_scores(nn.Identity(), torch.randn(4, 4)) == []


class TestCapturePerNeuronScores:
    """The scoring forward pass."""

    def test_pairs_layers_with_their_scores_in_forward_order(self):
        net = _TinyNet()
        scored = capture_per_neuron_scores(net, torch.randn(5, 4))
        assert [module for module, _ in scored] == _target_activations(net)
        assert [tuple(t.shape) for _, t in scored] == [
            (HIDDEN,),
            (LATENT_DIM,),
            (HIDDEN,),
        ]

    def test_scores_are_mean_absolute_activations(self):
        net = _TinyNet()
        obs = torch.randn(5, 4)
        expected = net.encoder.model.feature_activation_0(
            net.encoder.model.feature_linear_layer_0(obs)
        )
        (_module, scored), *_ = capture_per_neuron_scores(net, obs)
        assert torch.allclose(scored, expected.abs().mean(dim=0))

    def test_leaves_no_hooks_and_restores_training_mode(self):
        net = _TinyNet()
        net.train()
        capture_per_neuron_scores(net, torch.randn(5, 4))
        assert _hook_count(net) == 0
        assert net.training  # the pass runs under eval() but must restore the mode

    def test_conv_feature_maps_count_as_one_neuron_each(self):
        class _ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.model = nn.Sequential(
                    OrderedDict(
                        [
                            ("feature_conv_layer_0", nn.Conv2d(3, 5, 3)),
                            ("feature_activation_0", nn.ReLU()),
                        ]
                    )
                )

            def forward(self, x):
                return self.encoder.model(x)

        (_module, scored), *_ = capture_per_neuron_scores(
            _ConvNet(), torch.rand(4, 3, 8, 8)
        )
        assert scored.shape == torch.Size([5])  # one entry per channel, not per pixel


class TestNormalisedScores:
    """Definition 3.1's normalisation, shared with the removal operator."""

    def test_divides_by_the_layer_mean(self):
        scores = normalised_scores(torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(scores, torch.tensor([0.5, 1.0, 1.5]))

    def test_all_zero_layer_is_fully_dormant(self):
        assert torch.equal(normalised_scores(torch.zeros(4)), torch.zeros(4))

    def test_non_finite_entries_are_dropped(self):
        # A single inf would otherwise drive the layer mean to infinity and read
        # every other unit as dormant.
        scores = normalised_scores(torch.tensor([float("inf"), 1.0, 1.0]))
        assert scores.numel() == 2
        assert torch.allclose(scores, torch.ones(2))

    def test_no_finite_entries_yields_none(self):
        assert normalised_scores(torch.tensor([float("nan")])) is None


class TestDormantNeuronFraction:
    """Definition 3.1 aggregated over an agent's evaluation networks."""

    def test_counts_units_at_or_below_tau(self):
        net = _TinyNet()
        with torch.no_grad():  # half of each measured ReLU layer pinned off
            for layer in (
                net.encoder.model.feature_linear_layer_0,
                net.head_net.model.head_linear_layer_0,
            ):
                layer.weight.zero_()
                layer.bias[: HIDDEN // 2] = 0.0
                layer.bias[HIDDEN // 2 :] = 1.0
            net.encoder.model.feature_linear_layer_output.weight.zero_()
            net.encoder.model.feature_linear_layer_output.bias.fill_(1.0)

        agent = _FakeAgent(actor=net)
        # 4/8 dormant in each ReLU layer, 0/6 in the (constant) latent.
        assert dormant_neuron_fraction(agent, torch.randn(5, 4), tau=0.1) == (
            pytest.approx(8 / 22)
        )

    def test_measurement_failure_never_breaks_training(self):
        agent = _FakeAgent(actor=_TinyNet())
        result = dormant_neuron_fraction(agent, "not an observation batch")
        assert result != result  # nan

    def test_no_measurable_layers_reads_as_nan(self):
        agent = _FakeAgent(actor=nn.Identity())
        result = dormant_neuron_fraction(agent, torch.randn(5, 4))
        assert result != result  # nan


class TestCollectObservationBatch:
    """The fresh batch the scores are measured on."""

    class _FakeVecEnv:
        def __init__(self, rows: int = 4) -> None:
            self.obs = np.zeros((rows, 4), dtype=np.float32)
            self.steps = 0

        def reset(self):
            return self.obs, {}

        def step(self, _action):
            self.steps += 1
            z = np.zeros(len(self.obs), dtype=np.float32)
            return self.obs, z, z.astype(bool), z.astype(bool), {}

    class _Agent:
        def get_action(self, _obs, action_mask=None):
            return np.zeros(4, dtype=np.float32)

    def test_collects_up_to_the_requested_batch_size(self):
        env = self._FakeVecEnv(rows=4)
        batch = collect_observation_batch(env, self._Agent(), batch_size=10)
        assert batch.shape == (10, 4)
        assert env.steps > 0  # it had to step to reach 10 rows

    def test_stepping_failure_returns_what_was_gathered(self):
        # The diagnostic must never crash training, so a broken agent/env still
        # yields at least the reset observations.
        class _BrokenAgent:
            def get_action(self, _obs, action_mask=None):
                msg = "no action for you"
                raise RuntimeError(msg)

        batch = collect_observation_batch(
            self._FakeVecEnv(rows=4), _BrokenAgent(), batch_size=10
        )
        assert batch.shape == (4, 4)
