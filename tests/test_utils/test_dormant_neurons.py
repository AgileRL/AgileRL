"""Unit tests for the gradient-based (GraMa) dormant-neuron diagnostics."""

import math
from collections import OrderedDict

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from agilerl.algorithms import DQN, RainbowDQN
from agilerl.utils.dormant_neurons import (
    GraMaCapture,
    _count_dormant,
    _eval_networks,
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


class TestPerNeuronGrad:
    """Reduction of a captured gradient to one magnitude per neuron."""

    def test_dense_reduces_over_batch(self):
        go = torch.tensor([[-2.0, 1.0], [0.0, -3.0]])

        out = _per_neuron_grad((go,))

        assert torch.allclose(out, torch.tensor([1.0, 2.0]))

    def test_conv_reduces_over_batch_and_spatial(self):
        go = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2, 1)  # (B, C, H, W)

        out = _per_neuron_grad((go,))

        assert out.shape == (2,)
        assert torch.allclose(out, go.abs().mean(dim=(0, 2, 3)))

    @pytest.mark.parametrize(
        "grad",
        [
            pytest.param((None,), id="none-entry"),
            pytest.param(None, id="no-grad-tuple"),
            pytest.param((), id="empty-tuple"),
        ],
    )
    def test_missing_gradient_is_not_measurable(self, grad):
        # A measured layer outside the training loss graph receives no gradient;
        # it must read as unmeasured rather than as a magnitude of zero.
        assert _per_neuron_grad(grad) is None


class TestCountDormant:
    """Thresholding neurons on their *normalised* score."""

    def test_all_zero_is_fully_dormant(self):
        assert _count_dormant(torch.zeros(4), tau=0.1) == (4, 4)

    def test_thresholds_on_normalised_score(self):
        # mean = 0.5 -> norm = [0, 2, 0, 2]; two entries at 0 <= 0.1 are dormant.
        per_neuron = torch.tensor([0.0, 1.0, 0.0, 1.0])

        assert _count_dormant(per_neuron, tau=0.1) == (2, 4)


class TestCountDormantNonFinite:
    """A diverged agent can reach the diagnostic with NaN/inf gradients.

    Its fitness stays finite, so the tournament's fitness guard does not stop it
    and it can be the agent selected for measurement. Non-finite scores are
    excluded from the count rather than coerced to a magnitude: ``NaN`` never
    satisfies ``<= tau`` and would silently mask real dormancy in the same layer,
    while a single ``inf`` drives the layer mean to infinity and crushes every
    other neuron's normalised score to zero.
    """

    @pytest.mark.parametrize(
        ("per_neuron", "expected"),
        [
            pytest.param(torch.tensor([1.0, float("nan"), 0.0, 1.0]), (1, 3), id="nan"),
            pytest.param(torch.tensor([float("inf"), 1.0, 0.0, 1.0]), (1, 3), id="inf"),
        ],
    )
    def test_excludes_non_finite_and_still_finds_dormant(self, per_neuron, expected):
        # The genuinely dormant neuron (score 0.0) must still be counted, and the
        # non-finite one must leave both the count and the total.
        assert _count_dormant(per_neuron, tau=0.1) == expected

    def test_all_non_finite_layer_is_not_measurable(self):
        per_neuron = torch.tensor([float("nan")] * 4)
        assert _count_dormant(per_neuron, tau=0.1) == (0, 0)

    def test_diverged_agent_is_distinguishable_from_dead_one(self):
        """A collapsed network and an exploded one must not read the same.

        Coercing non-finite scores to zero would report both as fully dormant,
        collapsing two opposite pathologies into one number.
        """
        dead = _FakeAgent(_TinyNet())
        dead._grama_scores = [[torch.zeros(3), torch.zeros(2)]]

        diverged = _FakeAgent(_TinyNet())
        diverged._grama_scores = [
            [torch.full((3,), float("nan")), torch.full((2,), float("nan"))]
        ]

        assert dormant_neuron_fraction(dead, tau=0.1) == pytest.approx(1.0)
        assert math.isnan(dormant_neuron_fraction(diverged, tau=0.1))


# --------------------------------------------------------------------------- #
# GraMaCapture
# --------------------------------------------------------------------------- #
class TestGraMaCaptureKeepsLastMinibatch:
    """The snapshot is the most recent minibatch, not a mean over the cycle.

    Eq. 2's ``E_{x∈D}`` is an expectation at *fixed* parameters, whereas a cycle
    mean spans every optimizer step of that cycle and so mixes gradients taken
    w.r.t. parameter vectors that no longer exist. It also matters for ReBorn,
    which rewrites the network as it stands at the end of the cycle and must
    therefore score it as it stands at the end of the cycle.
    """

    @staticmethod
    def _capture_three_backward_passes():
        """Run three differing minibatches; return the agent and a per-pass reference."""
        torch.manual_seed(0)
        net = _TinyNet()
        agent = _FakeAgent(net)

        # Reference: independently capture the encoder activation's grad_input,
        # i.e. the pre-activation gradient the capture is defined on.
        ref = []

        def _refhook(_m, gi, _go):
            ref.append(gi[0].detach().abs().mean(dim=0))

        handle = net.encoder._activation_0.register_full_backward_hook(_refhook)
        with GraMaCapture(agent):
            for seed in (1, 2, 3):
                torch.manual_seed(seed)
                x = torch.randn(5, 4)
                net.zero_grad(set_to_none=True)
                net(x).sum().backward()
        handle.remove()
        return agent, net, ref

    def test_snapshot_shape_follows_measured_layers(self):
        # One list per eval network; each aligned to _target_activations (2 entries:
        # encoder ReLU + head ReLU; the Identity output activation is excluded).
        agent, net, _ref = self._capture_three_backward_passes()
        scores = agent._grama_scores
        assert len(scores) == 1
        assert len(scores[0]) == len(_target_activations(net)) == 2

    def test_stores_the_final_minibatch_gradient(self):
        agent, _net, ref = self._capture_three_backward_passes()
        assert torch.allclose(agent._grama_scores[0][0], ref[-1], atol=1e-6)

    def test_earlier_minibatches_are_discarded(self):
        """Guards the truncation itself -- a running mean would pass the test above
        only if every minibatch happened to be identical."""
        agent, _net, ref = self._capture_three_backward_passes()
        running_mean = torch.stack(ref).mean(dim=0)
        assert not torch.allclose(agent._grama_scores[0][0], running_mean, atol=1e-6)


def _dead_unit_net():
    """A ``_TinyNet`` whose encoder unit 1 can never fire, with an active head.

    Unit 1's incoming weights are zero and its bias strongly negative, so its
    pre-activation is negative for *every* input and its ReLU output is always 0.
    All remaining weights are set so each other measured unit fires on every
    sample, making that one unit the network's only dormancy.
    """
    net = _TinyNet(in_dim=4, hidden=3, out=2)
    with torch.no_grad():
        net.encoder.linear_0.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0] * 4, [0.0, 1.0, 0.0, 0.0]])
        )
        net.encoder.linear_0.bias.copy_(torch.tensor([1.0, -10.0, 1.0]))
        net.head_net.linear_0.weight.fill_(0.1)
        net.head_net.linear_0.bias.fill_(1.0)
        net.head_net.output_linear.weight.fill_(1.0)
        net.head_net.output_linear.bias.zero_()
    return net


# Strictly positive features, so every non-dead unit's pre-activation stays > 0.
_DEAD_UNIT_INPUT = torch.tensor(
    [[1.0, 2.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0]]
)


class TestInactiveUnitDetection:
    """A permanently inactive ReLU unit must read as dormant.

    This is the case that separates the two candidate gradients. The gradient
    w.r.t. a dead unit's *post*-activation output does not vanish -- it is the
    downstream weight projection, which stays healthy (here an identical 0.2 for
    all three encoder units), so the unit is indistinguishable from its live
    neighbours. It is the gradient w.r.t. the *pre*-activation that carries the
    σ'(z) factor and collapses to exactly zero, which is what makes GraMa
    activation-function aware at all.
    """

    @staticmethod
    def _capture():
        net = _dead_unit_net()
        agent = _FakeAgent(net)
        with GraMaCapture(agent):
            net.zero_grad(set_to_none=True)
            net(_DEAD_UNIT_INPUT).sum().backward()
        return agent

    def test_dead_unit_counts_towards_dormant_fraction(self):
        # Measured units: 3 encoder + 2 head; only encoder unit 1 is inactive.
        assert dormant_neuron_fraction(self._capture(), tau=0.1) == pytest.approx(1 / 5)

    def test_dead_unit_scores_zero_and_live_units_do_not(self):
        """The score itself must be zero -- ReBorn recycles by this value."""
        encoder_scores = self._capture()._grama_scores[0][0]
        assert encoder_scores[1] == 0.0
        assert (encoder_scores[[0, 2]] > 0).all()


class TestGraMaCaptureHookLifecycle:
    """Hooks must not outlive the ``with`` block.

    Asserted on ``module._backward_hooks`` -- a torch internal, but registration
    *is* the contract here and nothing else observes it: the hooks write to the
    capture's own buffer and only ``__exit__`` publishes ``_grama_scores``, so a
    leaked hook leaves the snapshot looking perfectly correct while it
    accumulates one more registration per training cycle for the rest of the run.
    """

    @staticmethod
    def _registered_hooks(net):
        return [
            len(module._backward_hooks or ()) for module in _target_activations(net)
        ]

    def test_hooks_are_released_on_exit(self):
        # Arrange
        net = _TinyNet()
        agent = _FakeAgent(net)

        # Act
        with GraMaCapture(agent):
            assert all(count > 0 for count in self._registered_hooks(net))
            net(torch.randn(2, 4)).sum().backward()

        # Assert
        assert self._registered_hooks(net) == [0, 0]

    def test_hooks_are_released_even_when_the_block_raises(self):
        # A training block that raises must not leave the network instrumented.
        net = _TinyNet()
        agent = _FakeAgent(net)

        with pytest.raises(RuntimeError, match="boom"):
            with GraMaCapture(agent):
                raise RuntimeError("boom")

        assert self._registered_hooks(net) == [0, 0]

    def test_repeated_captures_do_not_accumulate_hooks(self):
        """Each cycle re-instruments the same modules; the count must not grow."""
        net = _TinyNet()
        agent = _FakeAgent(net)

        for _cycle in range(3):
            with GraMaCapture(agent):
                net.zero_grad(set_to_none=True)
                net(torch.randn(2, 4)).sum().backward()

        assert self._registered_hooks(net) == [0, 0]


class TestCapturePerNeuronScores:
    """Pairing a captured snapshot back up with the modules it describes."""

    def test_pairs_each_score_with_its_activation(self):
        net = _TinyNet()

        pairs = capture_per_neuron_scores(net, [torch.ones(3), torch.ones(2)])

        assert [module for module, _scores in pairs] == _target_activations(net)

    @pytest.mark.parametrize(
        "per_neuron_list",
        [
            pytest.param(None, id="no-snapshot"),
            pytest.param([torch.ones(3)], id="too-short"),
            pytest.param([torch.ones(3)] * 3, id="too-long"),
        ],
    )
    def test_length_mismatch_yields_nothing(self, per_neuron_list):
        # A snapshot that does not line up with the measured layers cannot be
        # routed safely, so none of it is used -- ``_TinyNet`` measures 2 layers.
        net = _TinyNet()
        assert len(_target_activations(net)) == 2

        assert capture_per_neuron_scores(net, per_neuron_list) == []

    def test_skips_layers_with_no_captured_gradient(self):
        net = _TinyNet()

        pairs = capture_per_neuron_scores(net, [torch.ones(3), None])

        assert len(pairs) == 1
        module, per_neuron = pairs[0]
        assert module is _target_activations(net)[0]
        assert torch.equal(per_neuron, torch.ones(3))


class TestDormantNeuronFraction:
    """The fraction reported to the caller, pooled over every measured layer."""

    def test_pools_dormant_counts_across_layers(self):
        agent = _FakeAgent(_TinyNet())
        # encoder layer: 1 dormant of 3; head layer: 0 dormant of 2 -> 1/5.
        agent._grama_scores = [
            [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, 1.0])]
        ]

        assert dormant_neuron_fraction(agent, tau=0.1) == pytest.approx(1 / 5)

    def test_nan_without_snapshot(self):
        # An untrained agent has nothing to report; 0.0 would read as "healthy".
        agent = _FakeAgent(_TinyNet())

        assert math.isnan(dormant_neuron_fraction(agent, tau=0.1))

    def test_nan_when_every_layer_was_skipped(self):
        agent = _FakeAgent(_TinyNet())
        agent._grama_scores = [[None, None]]

        assert math.isnan(dormant_neuron_fraction(agent, tau=0.1))


# --------------------------------------------------------------------------- #
# _target_activations -- encoder coverage across encoder types
# --------------------------------------------------------------------------- #
# Each evolvable encoder names its activation sub-modules differently:
# ``EvolvableMLP`` emits ``*_activation_output``, ``EvolvableCNN``/``EvolvableLSTM``/
# ``EvolvableResNet`` emit ``*_output_activation``, and ``EvolvableMultiInput``
# emits a bare ``output``. Layer discovery must not depend on the naming, so the
# expectations below are keyed on the real activation modules each encoder builds.
_ENCODER_CASES = [
    pytest.param(
        spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
        ["encoder_activation_1", "encoder_activation_2", "encoder_activation_output"],
        id="mlp",
    ),
    pytest.param(
        spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8),
        ["encoder_activation_1", "encoder_activation_2", "encoder_output_activation"],
        id="cnn",
    ),
    pytest.param(
        spaces.Dict(
            {
                "a": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                "b": spaces.Box(-1, 1, shape=(6,), dtype=np.float32),
            }
        ),
        ["output"],
        id="multi_input",
    ),
]


class TestTargetActivations:
    """Which activation layers a network exposes to the GraMa capture."""

    @staticmethod
    def _encoder_module(net, suffix):
        """Return the encoder sub-module whose qualified name ends with *suffix*."""
        matches = [m for n, m in net.encoder.named_modules() if n.endswith(suffix)]
        assert len(matches) == 1, f"expected exactly one {suffix!r}, got {len(matches)}"
        return matches[0]

    @pytest.mark.parametrize(("observation_space", "expected"), _ENCODER_CASES)
    def test_measures_every_encoder_activation(self, observation_space, expected):
        # Arrange
        agent = DQN(observation_space, spaces.Discrete(4), device="cpu")
        (_network_id, net), *_ = _eval_networks(agent)

        # Act
        measured = {id(m) for m in _target_activations(net)}

        # Assert
        for suffix in expected:
            module = self._encoder_module(net, suffix)
            assert id(module) in measured, f"encoder activation {suffix!r} not measured"

    def test_excludes_head_output_activation(self):
        # Arrange
        agent = DQN(
            spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            spaces.Discrete(4),
            device="cpu",
        )
        (_network_id, net), *_ = _eval_networks(agent)

        # Act
        measured = {id(m) for m in _target_activations(net)}

        # Assert
        head = dict(net.head_net.named_modules())
        assert id(head["model.value_activation_1"]) in measured
        assert id(head["model.value_activation_output"]) not in measured

    def test_excludes_output_activation_of_every_duelling_stream(self):
        """A duelling head has two parallel streams, each with its own output.

        Dropping only the *last* activation of ``head_net`` would leave the value
        stream's output activation measured, so both streams must be handled.
        """
        # Arrange
        agent = RainbowDQN(
            spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            spaces.Discrete(4),
            device="cpu",
        )
        (_network_id, net), *_ = _eval_networks(agent)

        # Act
        measured = {id(m) for m in _target_activations(net)}

        # Assert
        head = dict(net.head_net.named_modules())
        assert id(head["model.value_activation_1"]) in measured
        assert id(head["advantage_net.advantage_activation_1"]) in measured
        assert id(head["model.value_activation_output"]) not in measured
        assert id(head["advantage_net.advantage_activation_output"]) not in measured


class TestSimBaEncoderActivations:
    """SimBa's residual blocks must expose their non-linearity to the capture.

    Each block holds ``hidden_size * scale_factor`` hidden units between its two
    linear layers -- nearly every parameter in the encoder. A bare functional
    non-linearity registers no sub-module, so no backward hook can be attached and
    the whole trunk goes unscored while the network still reports a plausible
    dormant fraction from its latent alone.
    """

    # 20 * 4 = 80 hidden units per block, distinct from the network's other
    # measured widths (the 64-wide latent, the 16-wide head layer) so a block
    # layer can be identified by width alone.
    HIDDEN = 20
    BLOCKS = 2
    SCALE = 4  # EvolvableSimBa's default scale_factor

    def test_scores_each_residual_block_hidden_layer(self):
        # Arrange
        agent = DQN(
            spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            spaces.Discrete(4),
            net_config={
                "simba": True,
                "encoder_config": {
                    "hidden_size": self.HIDDEN,
                    "num_blocks": self.BLOCKS,
                },
                "head_config": {"hidden_size": [16]},
            },
            device="cpu",
        )
        (_network_id, net), *_ = _eval_networks(agent)
        obs = torch.randn(4, 8)

        # Act
        with GraMaCapture(agent):
            net(obs).sum().backward()

        # Assert
        widths = [s.numel() for s in agent._grama_scores[0] if s is not None]
        assert widths.count(self.HIDDEN * self.SCALE) == self.BLOCKS
