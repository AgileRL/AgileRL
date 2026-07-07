"""Unit tests for the gradient-based (GraMa) dormant-neuron diagnostics."""

import math
from collections import OrderedDict

import pytest
import torch
from torch import nn

from agilerl.utils.dormant_neurons import (
    GraMaCapture,
    _count_dormant,
    _per_neuron_grad,
    _target_activations,
    capture_per_neuron_scores,
    dormant_neuron_fraction,
)


class _TinyNet(nn.Module):
    """Minimal encoder + head with the ``*_activation_*`` naming convention."""

    def __init__(self, in_dim=4, hidden=3, out=2):
        super().__init__()
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ("linear_0", nn.Linear(in_dim, hidden)),
                    ("_activation_0", nn.ReLU()),
                ]
            )
        )
        self.head_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear_0", nn.Linear(hidden, out)),
                    ("_activation_0", nn.ReLU()),
                    ("output_linear", nn.Linear(out, 1)),
                    ("_activation_output", nn.Identity()),
                ]
            )
        )

    def forward(self, x):
        return self.head_net(self.encoder(x))


class _Group:
    eval_network = "net"


class _Registry:
    groups = [_Group()]


class _FakeAgent:
    """Just enough surface for ``_eval_networks`` / ``GraMaCapture``."""

    def __init__(self, net):
        self.net = net
        self.registry = _Registry()


# --------------------------------------------------------------------------- #
# _per_neuron_grad
# --------------------------------------------------------------------------- #
def test_per_neuron_grad_dense_reduces_over_batch():
    go = torch.tensor([[-2.0, 1.0], [0.0, -3.0]])
    out = _per_neuron_grad((go,))
    assert torch.allclose(out, torch.tensor([1.0, 2.0]))


def test_per_neuron_grad_conv_reduces_over_batch_and_spatial():
    go = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2, 1)  # (B, C, H, W)
    out = _per_neuron_grad((go,))
    assert out.shape == (2,)
    assert torch.allclose(out, go.abs().mean(dim=(0, 2, 3)))


def test_per_neuron_grad_handles_missing_gradient():
    assert _per_neuron_grad((None,)) is None
    assert _per_neuron_grad(None) is None
    assert _per_neuron_grad(()) is None


# --------------------------------------------------------------------------- #
# _count_dormant
# --------------------------------------------------------------------------- #
def test_count_dormant_all_zero_is_fully_dormant():
    assert _count_dormant(torch.zeros(4), tau=0.1) == (4, 4)


def test_count_dormant_thresholds_on_normalised_score():
    # mean = 0.5 -> norm = [0, 2, 0, 2]; two entries at 0 <= 0.1 are dormant.
    per_neuron = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert _count_dormant(per_neuron, tau=0.1) == (2, 4)


# --------------------------------------------------------------------------- #
# GraMaCapture
# --------------------------------------------------------------------------- #
def test_grama_capture_stores_running_mean_of_abs_gradient():
    torch.manual_seed(0)
    net = _TinyNet()
    agent = _FakeAgent(net)

    # Reference: independently capture the encoder activation's grad_output.
    ref = []

    def _refhook(_m, _gi, go):
        ref.append(go[0].detach().abs().mean(dim=0))

    handle = net.encoder._activation_0.register_full_backward_hook(_refhook)
    with GraMaCapture(agent):
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            x = torch.randn(5, 4)
            net.zero_grad(set_to_none=True)
            net(x).sum().backward()
    handle.remove()

    # One list per eval network; each aligned to _target_activations (2 entries:
    # encoder ReLU + head ReLU; the Identity output activation is excluded).
    scores = agent._grama_scores
    assert len(scores) == 1
    assert len(scores[0]) == len(_target_activations(net)) == 2

    expected = torch.stack(ref).mean(dim=0)
    assert torch.allclose(scores[0][0], expected, atol=1e-6)


def test_grama_capture_removes_hooks_on_exit():
    net = _TinyNet()
    agent = _FakeAgent(net)
    with GraMaCapture(agent):
        net(torch.randn(2, 4)).sum().backward()
    # No hooks should remain registered on the measured activations.
    for module in _target_activations(net):
        assert not module._backward_hooks


def test_grama_capture_removes_hooks_even_on_exception():
    net = _TinyNet()
    agent = _FakeAgent(net)
    with pytest.raises(RuntimeError):
        with GraMaCapture(agent):
            raise RuntimeError("boom")
    for module in _target_activations(net):
        assert not module._backward_hooks


# --------------------------------------------------------------------------- #
# capture_per_neuron_scores
# --------------------------------------------------------------------------- #
def test_capture_per_neuron_scores_length_guard():
    net = _TinyNet()  # _target_activations has 2 entries
    assert capture_per_neuron_scores(net, None) == []
    assert capture_per_neuron_scores(net, [torch.ones(3)]) == []  # wrong length


def test_capture_per_neuron_scores_skips_none_entries():
    net = _TinyNet()
    pairs = capture_per_neuron_scores(net, [torch.ones(3), None])
    assert len(pairs) == 1
    module, per_neuron = pairs[0]
    assert module is _target_activations(net)[0]
    assert torch.equal(per_neuron, torch.ones(3))


# --------------------------------------------------------------------------- #
# dormant_neuron_fraction
# --------------------------------------------------------------------------- #
def test_dormant_neuron_fraction_from_snapshot():
    net = _TinyNet()
    agent = _FakeAgent(net)
    # encoder layer: 1 dormant of 3; head layer: 0 dormant of 2 -> 1/5.
    agent._grama_scores = [[torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, 1.0])]]
    assert dormant_neuron_fraction(agent, tau=0.1) == pytest.approx(1 / 5)


def test_dormant_neuron_fraction_nan_without_snapshot():
    agent = _FakeAgent(_TinyNet())
    assert math.isnan(dormant_neuron_fraction(agent, tau=0.1))


def test_dormant_neuron_fraction_nan_when_all_layers_skipped():
    agent = _FakeAgent(_TinyNet())
    agent._grama_scores = [[None, None]]
    assert math.isnan(dormant_neuron_fraction(agent, tau=0.1))
