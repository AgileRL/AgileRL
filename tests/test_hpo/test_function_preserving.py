# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from torch import nn

from agilerl.hpo import function_preserving as fp
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.networks.custom_modules import DuelingDistributionalMLP


def make_rng(seed: int = 0) -> np.random.Generator:
    """Return a seeded generator, mirroring the one Mutations owns."""
    return np.random.default_rng(seed)


def make_mlp(hidden_size: list[int] | None = None, **kwargs) -> EvolvableMLP:
    """Return a small unnormalised ReLU MLP."""
    options = {
        "num_inputs": 4,
        "num_outputs": 2,
        "hidden_size": hidden_size or [8, 8],
        "layer_norm": False,
        "min_mlp_nodes": 1,
        "device": "cpu",
        "name": "mlp",
    }
    options.update(kwargs)
    return EvolvableMLP(**options)


def make_cnn(**kwargs) -> EvolvableCNN:
    """Return a small unnormalised ReLU CNN."""
    options = {
        "input_shape": [3, 16, 16],
        "num_outputs": 4,
        "channel_size": [4, 4],
        "kernel_size": [3, 3],
        "stride_size": [1, 1],
        "layer_norm": False,
        "min_channel_size": 1,
        "device": "cpu",
        "name": "cnn",
    }
    options.update(kwargs)
    return EvolvableCNN(**options)


def make_norm(norm_type: type[nn.Module], units: int = 8) -> nn.Module:
    """Instantiate a normalisation of the given type over ``units`` units."""
    if norm_type is nn.GroupNorm:
        return nn.GroupNorm(2, units)
    return norm_type(units)


def splice_norm(mlp: EvolvableMLP, norm: nn.Module) -> EvolvableMLP:
    """Insert a normalisation between the first hidden layer and its consumer.

    The blocker reads registration order out of the container, so a spliced
    module lands in the forward path exactly where a configured one would.
    """
    rebuilt: dict[str, nn.Module] = {}
    for name, layer in mlp.model.named_children():
        rebuilt[name] = layer
        if name.endswith("linear_layer_1"):
            rebuilt["spliced_norm"] = norm
    mlp.model = nn.Sequential(*rebuilt.values())
    return mlp


def make_duelling(
    hidden_size: list[int] | None = None, **kwargs
) -> DuelingDistributionalMLP:
    """Return a duelling distributional head with two parallel streams."""
    options = {
        "num_inputs": 8,
        "num_outputs": 2,
        "hidden_size": hidden_size or [8, 8],
        "num_atoms": 3,
        "support": torch.linspace(-1.0, 1.0, 3),
        "layer_norm": False,
        "min_mlp_nodes": 1,
        "device": "cpu",
    }
    options.update(kwargs)
    return DuelingDistributionalMLP(**options)


class TestHiddenWidths:
    """Snapshot the output width of every non-output weight layer."""

    def test_mlp_reports_each_hidden_layer_width(self):
        widths = fp.hidden_widths(make_mlp(hidden_size=[8, 16]))

        assert widths == [8, 16]

    def test_cnn_reports_each_conv_channel_count(self):
        widths = fp.hidden_widths(make_cnn(channel_size=[4, 6]))

        assert widths == [4, 6]

    def test_output_layer_is_excluded(self):
        widths = fp.hidden_widths(make_mlp(hidden_size=[8], num_outputs=5))

        assert 5 not in widths

    def test_module_without_weight_layers_reports_nothing(self):
        assert fp.hidden_widths(nn.Sequential(nn.ReLU())) == []


class TestWeightStacks:
    """Discover the parallel weight-layer streams of a sub-module."""

    def test_flat_mlp_has_a_single_stream(self):
        stacks = fp.weight_stacks(make_mlp())

        assert len(stacks) == 1
        assert len(stacks[0]) == 3

    def test_duelling_head_has_two_parallel_streams(self):
        stacks = fp.weight_stacks(make_duelling())

        assert len(stacks) == 2

    def test_duelling_streams_share_their_hidden_widths(self):
        stacks = fp.weight_stacks(make_duelling(hidden_size=[8, 8]))

        first, second = ([layer.weight_mu.shape[0] for layer in s[:-1]] for s in stacks)
        assert first == second

    def test_a_sibling_of_a_different_shape_is_not_a_stream(self):
        head = make_duelling()
        head.unrelated = nn.Sequential(nn.Linear(99, 7), nn.Linear(7, 3))

        assert len(fp.weight_stacks(head)) == 2

    def test_module_without_weight_layers_has_no_streams(self):
        assert fp.weight_stacks(nn.Sequential(nn.ReLU())) == []


class TestNodeAdditionBlocker:
    """Decide whether widening a layer can preserve the network's function."""

    def test_unnormalised_relu_mlp_is_supported(self):
        assert fp.node_addition_blocker(make_mlp(), 0) is None

    def test_unnormalised_relu_cnn_is_supported(self):
        assert fp.node_addition_blocker(make_cnn(), 0) is None

    def test_a_norm_between_producer_and_activation_blocks(self):
        assert fp.node_addition_blocker(make_mlp(layer_norm=True), 0) == "norm"

    def test_a_convolution_batch_norm_blocks(self):
        assert fp.node_addition_blocker(make_cnn(layer_norm=True), 0) == "norm"

    @pytest.mark.parametrize(
        "norm_type",
        fp.NORM_LAYER_TYPES,
        ids=[norm_type.__name__ for norm_type in fp.NORM_LAYER_TYPES],
    )
    def test_every_recognised_normalisation_blocks(self, norm_type):
        """Parametrized over the tuple itself, so a type added to it is covered.

        Only ``LayerNorm``, ``RMSNorm`` and ``GroupNorm`` pool across units and
        genuinely defeat the fade; the batch and instance norms are per channel
        and are declined only to keep one uniform list.
        """
        mlp = splice_norm(make_mlp(), make_norm(norm_type))

        assert fp.node_addition_blocker(mlp, 0) == "norm"

    @pytest.mark.parametrize(
        "norm_type",
        [nn.LayerNorm, nn.RMSNorm, nn.GroupNorm],
        ids=["LayerNorm", "RMSNorm", "GroupNorm"],
    )
    def test_a_pooling_normalisation_must_block(self, norm_type):
        """Named rather than read off the tuple, so narrowing it fails here.

        These three pool their statistics across units, so a widening under one
        moves every existing unit's output however small the new fan-out is.
        Dropping any of them from the tuple would silently ship a mutation that
        claims to be function-preserving and is not.
        """
        assert norm_type in fp.NORM_LAYER_TYPES

        mlp = splice_norm(make_mlp(), make_norm(norm_type))

        assert fp.node_addition_blocker(mlp, 0) == "norm"

    def test_a_cross_unit_activation_blocks(self):
        assert fp.node_addition_blocker(make_mlp(activation="Softmax"), 0) == (
            "cross_unit_activation"
        )

    def test_a_recurrent_core_blocks(self):
        from agilerl.modules.lstm import EvolvableLSTM

        lstm = EvolvableLSTM(
            input_size=4,
            hidden_state_size=8,
            num_outputs=2,
            min_hidden_state_size=1,
            device="cpu",
        )

        assert fp.node_addition_blocker(lstm, 0) == "recurrent"

    def test_a_multi_input_encoder_blocks(self, dict_space):
        from agilerl.modules.multi_input import EvolvableMultiInput

        encoder = EvolvableMultiInput(
            observation_space=dict_space,
            num_outputs=4,
            latent_dim=8,
            min_latent_dim=1,
            device="cpu",
        )

        assert fp.node_addition_blocker(encoder, 0) == "multi_input"

    def test_a_simba_block_blocks(self):
        from agilerl.modules.simba import EvolvableSimBa

        simba = EvolvableSimBa(
            num_inputs=4,
            num_outputs=2,
            hidden_size=8,
            num_blocks=1,
            min_mlp_nodes=1,
            device="cpu",
        )

        assert fp.node_addition_blocker(simba, 0) == "simba"

    def test_a_residual_block_blocks(self):
        from agilerl.modules.resnet import EvolvableResNet

        resnet = EvolvableResNet(
            input_shape=[3, 16, 16],
            num_outputs=2,
            channel_size=4,
            kernel_size=3,
            stride_size=1,
            num_blocks=1,
            min_channel_size=1,
            device="cpu",
        )

        assert fp.node_addition_blocker(resnet, 0) == "residual"

    def test_an_out_of_range_layer_index_blocks(self):
        assert fp.node_addition_blocker(make_mlp(hidden_size=[8]), 5) == "no_consumer"


class TestLayerAdditionBlocker:
    """Decide whether an inserted layer can be an identity."""

    def test_unnormalised_relu_mlp_is_supported(self):
        blocker = fp.layer_addition_blocker(make_mlp(hidden_size=[8, 8]))

        assert blocker is None

    @pytest.mark.parametrize("activation", ["Tanh", "GELU", "LeakyReLU"])
    def test_a_non_relu_activation_blocks(self, activation):
        blocker = fp.layer_addition_blocker(
            make_mlp(hidden_size=[8, 8], activation=activation),
        )

        assert blocker == "non_relu"

    def test_a_norm_blocks(self):
        blocker = fp.layer_addition_blocker(
            make_mlp(hidden_size=[8, 8], layer_norm=True),
        )

        assert blocker == "norm"

    def test_a_convolutional_stack_blocks(self):
        blocker = fp.layer_addition_blocker(make_cnn())

        assert blocker == "not_mlp"

    def test_a_non_square_new_layer_blocks(self):
        blocker = fp.layer_addition_blocker(make_mlp(hidden_size=[8, 16]))

        assert blocker == "non_square"


class TestLatentAdditionBlocker:
    """Decide whether widening the latent can preserve the function."""

    def test_unnormalised_mlp_encoder_is_supported(self, vector_space, discrete_space):
        from agilerl.networks.q_networks import QNetwork

        network = QNetwork(
            vector_space,
            discrete_space,
            latent_dim=8,
            min_latent_dim=1,
            encoder_config={
                "hidden_size": [8],
                "min_mlp_nodes": 1,
                "layer_norm": False,
            },
            device="cpu",
        )

        assert fp.latent_addition_blocker(network) is None

    def test_a_normalised_mlp_encoder_blocks(self, vector_space, discrete_space):
        from agilerl.networks.q_networks import QNetwork

        network = QNetwork(
            vector_space,
            discrete_space,
            latent_dim=8,
            min_latent_dim=1,
            encoder_config={"hidden_size": [8], "min_mlp_nodes": 1, "layer_norm": True},
            device="cpu",
        )

        assert fp.latent_addition_blocker(network) == "norm"

    def test_a_continuous_q_network_is_supported(self, vector_space):
        from agilerl.networks.q_networks import ContinuousQNetwork

        network = ContinuousQNetwork(
            vector_space,
            vector_space,
            latent_dim=8,
            min_latent_dim=1,
            encoder_config={"hidden_size": [8], "min_mlp_nodes": 1},
            device="cpu",
        )

        assert fp.latent_addition_blocker(network) is None

    def test_a_recurrent_encoder_is_supported(self, vector_space, discrete_space):
        """A latent widening is head-side surgery, so the core is irrelevant.

        The recurrent core owns no per-unit weight rows, which rules out
        widening a layer *inside* it -- but ``add_latent_node`` only fades the
        head's new input columns, and the encoder's own output rows are
        appended at the tail like any other module's.
        """
        from agilerl.networks.q_networks import QNetwork

        network = QNetwork(
            vector_space,
            discrete_space,
            latent_dim=8,
            min_latent_dim=1,
            recurrent=True,
            encoder_config={"hidden_state_size": 8},
            device="cpu",
        )

        assert fp.latent_addition_blocker(network) is None

    def test_a_multi_input_encoder_is_supported(self, dict_space, discrete_space):
        """The interleaving is at the encoder's input, not at its output."""
        from agilerl.networks.q_networks import QNetwork

        network = QNetwork(
            dict_space, discrete_space, latent_dim=8, min_latent_dim=1, device="cpu"
        )

        assert fp.latent_addition_blocker(network) is None

    def test_a_simba_encoder_is_supported(self, vector_space, discrete_space):
        from agilerl.networks.q_networks import QNetwork

        network = QNetwork(
            vector_space,
            discrete_space,
            latent_dim=8,
            min_latent_dim=1,
            simba=True,
            encoder_config={"hidden_size": 8, "num_blocks": 1},
            device="cpu",
        )

        assert fp.latent_addition_blocker(network) is None

    def test_a_multi_input_module_blocks_its_own_latent(self, dict_space):
        """A multi-input encoder's *own* latent feeds the interleaved fusion.

        Widening it resizes every sub-encoder's output, so the fusion layer's
        new columns are spread through its input rather than appended at the
        tail -- the one latent mutation this machinery cannot preserve.
        """
        from agilerl.modules.multi_input import EvolvableMultiInput

        encoder = EvolvableMultiInput(
            observation_space=dict_space, num_outputs=8, device="cpu"
        )

        assert fp.latent_addition_blocker(encoder) == "multi_input"

    def test_a_network_without_a_latent_blocks(self):
        assert fp.latent_addition_blocker(nn.Sequential(nn.ReLU())) == "no_latent"


class TestPreserveAddedNodes:
    """Fade a widened layer's new units so the network's output is unchanged."""

    @staticmethod
    def widen(
        module, hidden_layer=0, added=4, noise_scale=0.0, seed=0, method="add_node"
    ):
        """Widen a layer and apply the fixup, returning whether it landed."""
        old_width = fp.hidden_widths(module)[hidden_layer]
        count = "numb_new_channels" if method == "add_channel" else "numb_new_nodes"
        getattr(module, method)(hidden_layer=hidden_layer, **{count: added})
        return fp.preserve_added_nodes(
            module, hidden_layer, old_width, make_rng(seed), noise_scale=noise_scale
        )

    def test_mlp_output_is_unchanged(self):
        module = make_mlp().eval()
        observation = torch.randn(4, 4)
        before = module(observation)

        self.widen(module)

        torch.testing.assert_close(module(observation), before, rtol=0, atol=0)

    def test_the_layer_actually_grew(self):
        module = make_mlp()

        self.widen(module)

        assert module.hidden_size[0] == 12

    def test_cnn_output_is_unchanged_between_convolutions(self):
        module = make_cnn().eval()
        observation = torch.randn(2, 3, 16, 16)
        before = module(observation)

        self.widen(module, hidden_layer=0, method="add_channel")

        torch.testing.assert_close(module.eval()(observation), before, rtol=0, atol=0)

    def test_cnn_output_is_unchanged_across_the_flatten_boundary(self):
        module = make_cnn().eval()
        observation = torch.randn(2, 3, 16, 16)
        before = module(observation)

        self.widen(module, hidden_layer=1, method="add_channel")

        # Zero columns still lengthen the projection's reduction, so the result
        # differs in the last bits of float32.
        torch.testing.assert_close(
            module.eval()(observation), before, rtol=0, atol=1e-6
        )

    def test_duelling_head_output_is_unchanged(self):
        module = make_duelling().eval()
        observation = torch.randn(4, 8)
        before = module(observation)

        self.widen(module)

        # A rebuilt noisy module returns in train mode, where its resampled
        # epsilon buffers move the output whatever the surgery did.
        torch.testing.assert_close(module.eval()(observation), before, rtol=0, atol=0)

    def test_both_duelling_streams_are_faded(self):
        module = make_duelling()

        self.widen(module)

        for stack in fp.weight_stacks(module):
            assert torch.count_nonzero(stack[1].weight_mu[:, 8:]) == 0

    def test_a_noisy_consumer_keeps_no_noise_on_the_new_columns(self):
        module = make_duelling()

        self.widen(module, noise_scale=0.5)

        for stack in fp.weight_stacks(module):
            assert torch.count_nonzero(stack[1].weight_sigma[:, 8:]) == 0

    def test_it_reports_that_the_fixup_landed(self):
        assert self.widen(make_mlp(), added=4) is True

    def test_a_blocked_growth_reports_nothing_to_preserve(self):
        module = make_mlp(hidden_size=[8], max_mlp_nodes=8)

        assert self.widen(module, added=4) is True
        assert module.hidden_size == [8]

    def test_noise_leaves_the_new_columns_non_zero(self):
        module = make_mlp()

        self.widen(module, noise_scale=0.5)

        assert torch.count_nonzero(fp.weight_stacks(module)[0][1].weight[:, 8:]) > 0

    def test_noise_stays_below_the_existing_column_scale(self):
        module = make_mlp()

        self.widen(module, noise_scale=fp.FP_NOISE_SCALE)

        consumer = fp.weight_stacks(module)[0][1].weight
        assert consumer[:, 8:].std() < consumer[:, :8].std()

    def test_the_same_seed_gives_the_same_weights(self):
        first, second = make_mlp(), make_mlp()
        second.load_state_dict(first.state_dict())

        # Forked so seeding the stock operator's own initialisation does not
        # leak a fixed global RNG state into whatever test runs next.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.widen(first, noise_scale=0.5, seed=7)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.widen(second, noise_scale=0.5, seed=7)

        for left, right in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        ):
            assert torch.equal(left, right)

    @pytest.mark.parametrize("noise_scale", [0.0, 0.5])
    def test_it_never_consumes_the_global_generator(self, noise_scale):
        module = make_mlp()
        old_width = fp.hidden_widths(module)[0]
        module.add_node(hidden_layer=0, numb_new_nodes=4)
        state = torch.random.get_rng_state()

        fp.preserve_added_nodes(
            module, 0, old_width, make_rng(), noise_scale=noise_scale
        )

        assert torch.equal(torch.random.get_rng_state(), state)


class TestPreserveAddedLayer:
    """Initialise an inserted layer to the identity."""

    def test_mlp_output_is_unchanged(self):
        module = make_mlp(hidden_size=[8]).eval()
        observation = torch.randn(4, 4)
        before = module(observation)

        module.add_layer()
        fp.preserve_added_layer(module)

        torch.testing.assert_close(
            module.eval()(observation), before, rtol=0, atol=1e-7
        )

    def test_the_network_actually_deepened(self):
        module = make_mlp(hidden_size=[8])

        module.add_layer()
        fp.preserve_added_layer(module)

        assert len(module.hidden_size) == 2

    def test_the_new_layer_is_the_identity(self):
        module = make_mlp(hidden_size=[8])

        module.add_layer()
        fp.preserve_added_layer(module)

        new_layer = fp.weight_stacks(module)[0][-2]
        assert torch.equal(new_layer.weight, torch.eye(8))
        assert torch.count_nonzero(new_layer.bias) == 0

    def test_duelling_head_output_is_unchanged(self):
        module = make_duelling(hidden_size=[8]).eval()
        observation = torch.randn(4, 8)
        before = module(observation)

        module.add_layer()
        fp.preserve_added_layer(module)

        torch.testing.assert_close(
            module.eval()(observation), before, rtol=0, atol=1e-6
        )

    def test_both_duelling_streams_get_an_identity(self):
        module = make_duelling(hidden_size=[8])

        module.add_layer()
        fp.preserve_added_layer(module)

        for stack in fp.weight_stacks(module):
            assert torch.equal(stack[-2].weight_mu, torch.eye(8))
            assert torch.count_nonzero(stack[-2].weight_sigma) == 0

    def test_it_reports_whether_a_layer_was_written(self):
        module = make_mlp(hidden_size=[8])
        module.add_layer()

        assert fp.preserve_added_layer(module) is True

    def test_a_non_square_layer_is_left_alone(self):
        assert fp.preserve_added_layer(make_mlp(hidden_size=[8, 16])) is False


class TestPreserveAddedLatent:
    """Fade the head's new latent columns so the network's output is unchanged."""

    @staticmethod
    def q_network(vector_space, discrete_space, **kwargs):
        from agilerl.networks.q_networks import QNetwork

        return QNetwork(
            vector_space,
            discrete_space,
            latent_dim=8,
            min_latent_dim=1,
            max_latent_dim=128,
            encoder_config={
                "hidden_size": [8],
                "min_mlp_nodes": 1,
                "layer_norm": False,
            },
            head_config={"hidden_size": [8], "min_mlp_nodes": 1, "layer_norm": False},
            device="cpu",
            **kwargs,
        )

    @staticmethod
    def widen(network, added=4, noise_scale=0.0, seed=0):
        """Widen the latent and apply the fixup, returning whether it landed."""
        old_latent = network.latent_dim
        network.add_latent_node(numb_new_nodes=added)
        return fp.preserve_added_latent(
            network, old_latent, make_rng(seed), noise_scale=noise_scale
        )

    def test_output_is_unchanged(self, vector_space, discrete_space):
        network = self.q_network(vector_space, discrete_space).eval()
        observation = torch.randn(4, 4)
        before = network(observation)

        self.widen(network)

        torch.testing.assert_close(network(observation), before, rtol=0, atol=0)

    def test_the_latent_actually_grew(self, vector_space, discrete_space):
        network = self.q_network(vector_space, discrete_space)

        self.widen(network)

        assert network.latent_dim == 12

    def test_the_new_head_columns_are_faded(self, vector_space, discrete_space):
        network = self.q_network(vector_space, discrete_space)

        self.widen(network)

        entry = fp.weight_stacks(network.head_net)[0][0]
        assert torch.count_nonzero(entry.weight[:, 8:12]) == 0

    def test_it_reports_that_the_fixup_landed(self, vector_space, discrete_space):
        network = self.q_network(vector_space, discrete_space)

        assert self.widen(network, added=4) is True

    def test_a_blocked_growth_reports_nothing_to_preserve(
        self, vector_space, discrete_space
    ):
        network = self.q_network(vector_space, discrete_space)
        network.max_latent_dim = 8

        assert self.widen(network, added=4) is True
        assert network.latent_dim == 8

    def test_a_network_without_a_head_reports_a_miss(self):
        network = nn.Module()
        network.latent_dim = 12

        assert fp.preserve_added_latent(network, 8, make_rng()) is False

    def test_a_multi_input_encoder_output_is_unchanged(
        self, dict_space, discrete_space
    ):
        """The fade is exact even though the encoder fuses several inputs.

        Growing the latent only appends rows to the fusion layer's output, so
        every pre-existing latent coordinate survives and the head's faded new
        columns carry the rest.
        """
        from agilerl.networks.q_networks import QNetwork

        network = QNetwork(
            dict_space,
            discrete_space,
            latent_dim=8,
            min_latent_dim=1,
            max_latent_dim=128,
            head_config={"hidden_size": [8], "min_mlp_nodes": 1, "layer_norm": False},
            device="cpu",
        ).eval()
        observation = {
            key: torch.as_tensor(subspace.sample()).unsqueeze(0).float()
            for key, subspace in dict_space.spaces.items()
        }
        before = network(observation)

        assert self.widen(network) is True

        # Not bit-exact: the faded columns contribute exactly zero, but widening
        # the fusion layer's input changes the matmul's tiling, so the surviving
        # terms are summed in a different order. The residual is float32 epsilon.
        torch.testing.assert_close(network(observation), before, rtol=0, atol=1e-6)

    def test_a_continuous_critic_keeps_its_action_sensitivity(self, vector_space):
        from agilerl.networks.q_networks import ContinuousQNetwork

        network = ContinuousQNetwork(
            vector_space,
            vector_space,
            latent_dim=8,
            min_latent_dim=1,
            encoder_config={"hidden_size": [8], "min_mlp_nodes": 1},
            head_config={"hidden_size": [8], "min_mlp_nodes": 1, "layer_norm": False},
            device="cpu",
        ).eval()
        observation = torch.randn(4, 4)
        actions = torch.randn(4, 4, requires_grad=True)
        before = network(observation, actions)

        self.widen(network)

        after = network(observation, actions)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        gradient = torch.autograd.grad(after.sum(), actions)[0]
        assert torch.count_nonzero(gradient) > 0

    def test_a_continuous_critic_slides_its_action_block(self, vector_space):
        from agilerl.networks.q_networks import ContinuousQNetwork

        network = ContinuousQNetwork(
            vector_space,
            vector_space,
            latent_dim=8,
            min_latent_dim=1,
            encoder_config={"hidden_size": [8], "min_mlp_nodes": 1},
            head_config={"hidden_size": [8], "min_mlp_nodes": 1, "layer_norm": False},
            device="cpu",
        )
        entry = fp.weight_stacks(network.head_net)[0][0]
        action_block = entry.weight[:, 8:12].detach().clone()

        self.widen(network)

        moved = fp.weight_stacks(network.head_net)[0][0]
        assert torch.equal(moved.weight[:, 12:16], action_block)


class TestBaseMutation:
    """Reduce a possibly prefixed mutation name to its trailing method."""

    @pytest.mark.parametrize(
        ("mut_method", "expected"),
        [
            ("add_node", "add_node"),
            ("head_net.add_node", "add_node"),
            ("agent_0.head_net.add_node", "add_node"),
            ("agent_0.add_latent_node", "add_latent_node"),
            (None, ""),
        ],
    )
    def test_it_returns_the_trailing_method(self, mut_method, expected):
        assert fp.base_mutation(mut_method) == expected


class TestIsLatentMutation:
    """Recognise the mutations that resize a network's latent."""

    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            ("add_latent_node", True),
            ("remove_latent_node", True),
            ("add_node", False),
            ("add_layer", False),
        ],
    )
    def test_it_recognises_latent_methods(self, base, expected):
        assert fp.is_latent_mutation(base) is expected


class TestResolveTarget:
    """Walk a dotted mutation name to the module it acts on."""

    @staticmethod
    def q_network(vector_space, discrete_space):
        from agilerl.networks.q_networks import QNetwork

        return QNetwork(
            vector_space,
            discrete_space,
            latent_dim=8,
            min_latent_dim=1,
            encoder_config={
                "hidden_size": [8],
                "min_mlp_nodes": 1,
                "layer_norm": False,
            },
            head_config={"hidden_size": [8], "min_mlp_nodes": 1, "layer_norm": False},
            device="cpu",
        )

    def test_a_bare_method_resolves_to_the_module_itself(self):
        module = make_mlp()

        assert fp.resolve_target(module, "add_node") is module

    def test_a_submodule_segment_is_followed(self, vector_space, discrete_space):
        network = self.q_network(vector_space, discrete_space)

        assert fp.resolve_target(network, "head_net.add_node") is network.head_net

    def test_a_latent_method_resolves_to_the_network(
        self, vector_space, discrete_space
    ):
        network = self.q_network(vector_space, discrete_space)

        assert fp.resolve_target(network, "add_latent_node") is network

    def test_an_agent_segment_indexes_a_module_dict(self, vector_space, discrete_space):
        from agilerl.modules import ModuleDict

        policies = ModuleDict({"agent_0": self.q_network(vector_space, discrete_space)})

        target = fp.resolve_target(policies, "agent_0.head_net.add_node")

        assert target is policies["agent_0"].head_net

    def test_an_unresolvable_name_returns_nothing(self):
        assert fp.resolve_target(make_mlp(), "missing.add_node") is None


class TestNodeAdditionBlockerWithoutAnIndex:
    """A mutation that reports no layer index cannot be located."""

    def test_a_missing_layer_index_blocks(self):
        assert fp.node_addition_blocker(make_mlp(), None) == "no_consumer"

    def test_a_structural_exclusion_outranks_a_missing_index(self):
        from agilerl.modules.lstm import EvolvableLSTM

        lstm = EvolvableLSTM(
            input_size=4,
            hidden_state_size=8,
            num_outputs=2,
            min_hidden_state_size=1,
            device="cpu",
        )

        assert fp.node_addition_blocker(lstm, None) == "recurrent"


class TestModuleScanGuards:
    """Layer lookups degrade to an empty scan rather than raising."""

    def test_modules_between_an_unregistered_layer_is_empty(self):
        container = make_mlp()
        stray = nn.Linear(4, 4)

        assert fp._modules_between(container, stray, stray) == []

    def test_modules_after_an_unregistered_layer_is_empty(self):
        assert fp._modules_after(make_mlp(), nn.Linear(4, 4)) == []

    def test_a_module_without_a_feature_map_spans_one_column(self):
        assert fp._spatial_block(make_mlp()) == 1

    def test_a_layer_owning_no_weight_tensor_is_rejected(self):
        # Unreachable via _is_weight_layer; raising says a scan escaped it
        # rather than letting an AttributeError surface somewhere further on.
        with pytest.raises(TypeError, match="owns no weight tensor"):
            fp._primary_weight(nn.ReLU())


class TestLayerAdditionBlockerEdges:
    """Structural and degenerate stacks decline before anything is written."""

    def test_a_simba_block_blocks(self):
        from agilerl.modules.simba import EvolvableSimBa

        simba = EvolvableSimBa(
            num_inputs=4,
            num_outputs=2,
            hidden_size=8,
            num_blocks=1,
            min_mlp_nodes=1,
            device="cpu",
        )

        blocker = fp.layer_addition_blocker(simba)

        assert blocker == "simba"

    def test_a_stack_without_two_layers_blocks(self):
        module = nn.Sequential(nn.Linear(4, 2))

        blocker = fp.layer_addition_blocker(module)

        assert blocker == "not_mlp"


class TestLatentAdditionBlockerEdges:
    """An encoder with no weight layers exposes no latent to widen."""

    def test_an_encoder_without_weight_layers_blocks(self):
        network = nn.Module()
        network.latent_dim = 8
        network.encoder = nn.Sequential(nn.ReLU())

        assert fp.latent_addition_blocker(network) == "no_latent"


class TestPreserveSkipsWhatItCannotWrite:
    """A fixup that cannot write leaves the module untouched and says so."""

    def test_an_out_of_range_layer_index_preserves_nothing(self):
        module = make_mlp()

        assert fp.preserve_added_nodes(module, 9, 8, make_rng()) is False

    def test_a_single_layer_stack_gets_no_identity(self):
        assert fp.preserve_added_layer(nn.Sequential(nn.Linear(4, 4))) is False

    def test_a_head_narrower_than_the_latent_is_left_alone(self):
        network = nn.Module()
        network.latent_dim = 12
        network.head_net = nn.Sequential(nn.Linear(4, 2))
        entry = network.head_net[0]
        before = entry.weight.detach().clone()

        preserved = fp.preserve_added_latent(network, 8, make_rng())

        assert preserved is False
        assert torch.equal(entry.weight, before)

    def test_a_missing_mutation_name_resolves_to_nothing(self):
        assert fp.resolve_target(make_mlp(), None) is None
