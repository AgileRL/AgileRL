# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from torch import nn

from agilerl.algorithms import DQN, PPO, TD3, RainbowDQN
from agilerl.modules.custom_components import (
    NoisyLinear,
    ResidualBlock,
    SimbaResidualBlock,
)
from agilerl.utils import mutation_utils
from agilerl.utils.algo_utils import share_encoder_parameters
from tests.helper_functions import capture_grama_snapshot


def make_rng(seed: int = 0) -> np.random.Generator:
    """Return a seeded generator, mirroring the one Mutations owns."""
    return np.random.default_rng(seed)


def reset_producer(
    producer: nn.Module,
    next_layers: list[nn.Module],
    per_neuron: torch.Tensor,
    dormant_threshold: float = 0.01,
    seed: int = 0,
    norm: nn.Module | None = None,
    cnn_channels: int | None = None,
    cnn_spatial: int | None = None,
) -> list[int]:
    """Reset one hand-built layer's dormant neurons and return their indices."""
    consumers = mutation_utils._resolve_consumers(
        producer,
        next_layers,
        cnn_channels,
        cnn_spatial,
    )
    indices = mutation_utils._dormant_indices(per_neuron, dormant_threshold)
    mutation_utils._reset_layer_neurons(
        producer, consumers, norm, indices, make_rng(seed)
    )
    return indices


def mlp_net_config() -> dict:
    """Return a fresh MLP net config."""
    return {
        "latent_dim": 8,
        "min_latent_dim": 1,
        "encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
        "head_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
    }


def latent_marked_dormant(network, index: int) -> list[torch.Tensor | None]:
    """Snapshot for network in which only latent unit index is dormant."""
    encoder, head = network.encoder, network.head_net
    terminal = mutation_utils._activation_modules(encoder, include_output=True)[-1]
    scores: list[torch.Tensor | None] = []
    for activation in mutation_utils.target_activations(network):
        producer = mutation_utils._resolve_producer_and_next(
            activation, encoder, head
        ).producer
        if producer is None:
            scores.append(None)
            continue
        per_neuron = torch.ones(mutation_utils._weight_param(producer).shape[0])
        if activation is terminal:
            per_neuron[index] = 0.0
        scores.append(per_neuron)
    return scores


@pytest.fixture
def dqn_agent(vector_space, discrete_space):
    return DQN(
        vector_space,
        discrete_space,
        net_config=mlp_net_config(),
        device="cpu",
    )


class TestDormantIndices:
    """Select the neurons whose normalised GraMa score is at or below tau."""

    def test_threshold_applies_to_the_layer_normalised_score(self):
        # Mean is 1.0, so the scores are the values themselves.
        per_neuron = torch.tensor([2.0, 0.05, 1.0, 0.95])

        result = mutation_utils._dormant_indices(per_neuron, 0.1)

        assert result == [1]

    def test_threshold_is_scale_invariant(self):
        per_neuron = torch.tensor([2.0, 0.05, 1.0, 0.95])

        scaled = mutation_utils._dormant_indices(per_neuron * 1000.0, 0.1)

        # Normalising by the layer mean removes the overall scale.
        assert scaled == mutation_utils._dormant_indices(per_neuron, 0.1)

    def test_layer_with_no_gradient_anywhere_is_entirely_dormant(self):
        result = mutation_utils._dormant_indices(torch.zeros(4), 0.01)

        assert result == [0, 1, 2, 3]

    def test_zero_threshold_selects_only_exactly_dead_neurons(self):
        per_neuron = torch.tensor([1.0, 0.0, 1e-9, 2.0])

        result = mutation_utils._dormant_indices(per_neuron, 0.0)

        assert result == [1]

    def test_non_finite_scores_are_treated_as_dormant(self):
        # A diverged unit is exactly one worth re-initialising.
        per_neuron = torch.tensor([1.0, float("nan"), float("inf"), 2.0])

        result = mutation_utils._dormant_indices(per_neuron, 0.01)

        assert result == [1, 2]

    def test_empty_layer_selects_nothing(self):
        assert mutation_utils._dormant_indices(torch.empty(0), 0.01) == []


class TestResetLayerNeurons:
    """Re-initialise dormant neurons and re-seed their outgoing weights."""

    def build(self) -> tuple[nn.Linear, nn.Linear, torch.Tensor]:
        producer = nn.Linear(3, 5)
        consumer = nn.Linear(5, 2)
        with torch.no_grad():
            producer.weight.fill_(1.0)
            producer.bias.fill_(7.0)
            consumer.weight.fill_(2.0)
        # Neurons 1 and 2 are dormant, 0/3/4 are live.
        return producer, consumer, torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])

    def test_dormant_rows_are_reinitialised_and_their_biases_zeroed(self):
        producer, consumer, per_neuron = self.build()

        indices = reset_producer(producer, [consumer], per_neuron)

        assert indices == [1, 2]
        for index in indices:
            assert not torch.allclose(producer.weight[index], torch.ones(3))
            assert producer.bias[index].item() == 0.0

    def test_live_neurons_are_left_exactly_as_they_were(self):
        producer, consumer, per_neuron = self.build()

        reset_producer(producer, [consumer], per_neuron)

        for index in (0, 3, 4):
            assert torch.equal(producer.weight[index], torch.ones(3))
            assert producer.bias[index].item() == pytest.approx(7.0)
            assert torch.equal(consumer.weight[:, index], torch.full((2,), 2.0))

    def test_incoming_weights_stay_within_the_xavier_bound(self):
        producer, consumer, per_neuron = self.build()
        bound = np.sqrt(6.0 / (3 + 5))

        indices = reset_producer(producer, [consumer], per_neuron)

        for index in indices:
            assert producer.weight[index].abs().max().item() <= bound

    def test_outgoing_block_is_scaled_to_the_live_column_reference(self):
        producer, consumer, per_neuron = self.build()
        columns = consumer.weight.detach()[:, [0, 3, 4]]
        live = float(columns.pow(2).sum(0).sqrt().median())

        indices = reset_producer(producer, [consumer], per_neuron)

        expected = mutation_utils.REGRAMA_OUT_SCALE * live
        for index in indices:
            assert consumer.weight[:, index].norm().item() == pytest.approx(
                expected,
                rel=1e-5,
            )

    def test_revived_neuron_receives_an_incoming_gradient(self):
        producer, consumer, per_neuron = self.build()
        indices = reset_producer(producer, [consumer], per_neuron)

        producer.zero_grad()
        consumer(torch.relu(producer(torch.rand(8, 3)))).square().mean().backward()

        assert producer.weight.grad[indices].abs().sum().item() > 0.0

    def test_reset_is_reproducible_for_a_given_generator_seed(self):
        first_producer, first_consumer, per_neuron = self.build()
        second_producer, second_consumer, _ = self.build()

        reset_producer(first_producer, [first_consumer], per_neuron, seed=7)
        reset_producer(second_producer, [second_consumer], per_neuron, seed=7)

        assert torch.equal(first_producer.weight, second_producer.weight)
        assert torch.equal(first_consumer.weight, second_consumer.weight)

    def test_reset_does_not_consume_the_global_torch_generator(self):
        producer, consumer, per_neuron = self.build()
        torch.manual_seed(0)
        before = torch.rand(1).item()

        torch.manual_seed(0)
        reset_producer(producer, [consumer], per_neuron)

        assert torch.rand(1).item() == pytest.approx(before)

    def test_non_finite_incoming_weights_are_scrubbed(self):
        producer, consumer, per_neuron = self.build()
        with torch.no_grad():
            producer.weight[0].fill_(float("nan"))

        reset_producer(producer, [consumer], per_neuron)

        assert torch.isfinite(producer.weight).all()
        assert torch.isfinite(consumer.weight).all()

    def test_a_producer_without_a_bias_is_reset_without_error(self):
        # A bias is genuinely optional, so its absence must be a normal outcome.
        producer = nn.Linear(3, 5, bias=False)
        consumer = nn.Linear(5, 2)
        per_neuron = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])

        indices = reset_producer(producer, [consumer], per_neuron)

        assert indices == [1, 2]
        assert producer.bias is None


class TestResetLayerNeuronsAcrossTheFlattenBoundary:
    """Rewrite the right strided block when a conv encoder feeds a dense head."""

    CHANNELS = 4
    SPATIAL = 9  # a 3x3 feature map, flattened channel-major

    def build(self) -> tuple[nn.Conv2d, nn.Linear, torch.Tensor]:
        producer = nn.Conv2d(2, self.CHANNELS, kernel_size=3)
        consumer = nn.Linear(self.CHANNELS * self.SPATIAL, 2)
        with torch.no_grad():
            producer.weight.fill_(1.0)
            producer.bias.fill_(7.0)
            for channel in range(self.CHANNELS):
                consumer.weight[:, self.columns(channel)] = float(channel + 1)
        # Only feature map 1 is dormant.
        return producer, consumer, torch.tensor([1.0, 0.0, 1.0, 1.0])

    def columns(self, channel: int) -> slice:
        return slice(channel * self.SPATIAL, (channel + 1) * self.SPATIAL)

    def block(self, consumer: nn.Linear, channel: int) -> torch.Tensor:
        return consumer.weight.detach()[:, self.columns(channel)]

    def reset(self, producer, consumer, per_neuron) -> list[int]:
        return reset_producer(
            producer,
            [consumer],
            per_neuron,
            cnn_channels=self.CHANNELS,
            cnn_spatial=self.SPATIAL,
        )

    def test_the_neighbouring_feature_maps_blocks_are_untouched(self):
        # An off-by-one in the stride arithmetic would corrupt the feature maps
        # either side of the one being recycled.
        producer, consumer, per_neuron = self.build()

        indices = self.reset(producer, consumer, per_neuron)

        assert indices == [1]
        for channel in (0, 2, 3):
            assert torch.equal(
                self.block(consumer, channel),
                torch.full((2, self.SPATIAL), float(channel + 1)),
            )

    def test_the_dormant_maps_block_is_scaled_to_the_live_block_reference(self):
        producer, consumer, per_neuron = self.build()
        live = float(
            torch.tensor(
                [self.block(consumer, channel).norm() for channel in (0, 2, 3)],
            ).median(),
        )

        self.reset(producer, consumer, per_neuron)

        assert self.block(consumer, 1).norm().item() == pytest.approx(
            mutation_utils.REGRAMA_OUT_SCALE * live,
            rel=1e-5,
        )

    def test_the_dormant_maps_filter_is_reinitialised_and_its_bias_zeroed(self):
        producer, consumer, per_neuron = self.build()

        self.reset(producer, consumer, per_neuron)

        assert not torch.allclose(producer.weight[1], torch.ones(2, 3, 3))
        assert producer.bias[1].item() == 0.0
        for channel in (0, 2, 3):
            assert torch.equal(producer.weight[channel], torch.ones(2, 3, 3))


class TestLiveColumnScale:
    """Median outgoing-column norm, measured over the neurons left alone."""

    def test_dense_consumer_matches_the_true_median_column_norm(self):
        weight = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        expected = float(weight.pow(2).sum(0).sqrt().median())

        result = mutation_utils._live_column_scale(weight, 1, list(range(6)))

        assert result == pytest.approx(expected, rel=1e-6)

    def test_conv_to_dense_boundary_measures_whole_strided_blocks(self):
        # 3 producer feature maps, each spending 4 flattened columns.
        weight = torch.rand(2, 12)
        blocks = weight.reshape(2, 3, 4)
        expected = float(blocks.pow(2).sum(dim=(0, 2)).sqrt().median())

        result = mutation_utils._live_column_scale(weight, 4, [0, 1, 2])

        assert result == pytest.approx(expected, rel=1e-6)

    def test_conv_consumer_measures_whole_filters_not_kernel_positions(self):
        # The neuron axis is dimension 1.
        weight = torch.rand(2, 3, 3, 3)
        blocks = weight.reshape(2, 3, -1)
        expected = float(blocks.pow(2).sum(dim=(0, 2)).sqrt().median())

        result = mutation_utils._live_column_scale(weight, 1, [0, 1, 2])

        assert result == pytest.approx(expected, rel=1e-6)

    def test_only_the_kept_neurons_are_measured(self):
        # Measuring the whole layer would let a repeatedly-reset layer size each
        # revival against the previous one's.
        weight = torch.ones(2, 5)
        weight[:, :3] = 100.0

        kept = mutation_utils._live_column_scale(weight, 1, [3, 4])
        everything = mutation_utils._live_column_scale(weight, 1, [0, 1, 2, 3, 4])

        assert kept < everything

    def test_no_kept_neuron_falls_back_to_the_whole_layers_median(self):
        weight = torch.ones(2, 3) * 4.0
        expected = float(weight.pow(2).sum(0).sqrt().median())

        result = mutation_utils._live_column_scale(weight, 1, [])

        assert result == pytest.approx(expected, rel=1e-6)

    def test_an_all_zero_layer_falls_back_to_the_xavier_bound(self):
        weight = torch.zeros(2, 3)

        result = mutation_utils._live_column_scale(weight, 1, [])

        assert result > 0.0

    def test_a_layer_with_no_columns_at_all_falls_back_to_the_xavier_bound(self):
        weight = torch.zeros(2, 0)

        result = mutation_utils._live_column_scale(weight, 1, [])

        assert result > 0.0


class TestRevivedBlock:
    """Draw the outgoing block a reset neuron is revived with."""

    @pytest.mark.parametrize("is_noise_scale", [False, True], ids=["weight", "noise"])
    def test_block_is_rescaled_to_the_requested_norm(self, is_noise_scale):
        result = mutation_utils._revived_block(
            torch.zeros(4),
            0.5,
            make_rng(),
            is_noise_scale=is_noise_scale,
        )

        assert result.norm().item() == pytest.approx(0.5, rel=1e-6)

    def test_a_noise_block_is_non_negative_and_uniform(self):
        result = mutation_utils._revived_block(
            torch.zeros(3, 2),
            1.0,
            make_rng(),
            is_noise_scale=True,
        )

        assert (result >= 0).all()
        assert result.unique().numel() == 1

    def test_a_weight_block_is_a_signed_direction(self):
        result = mutation_utils._revived_block(torch.zeros(64), 1.0, make_rng())

        assert (result < 0).any()
        assert (result > 0).any()

    @pytest.mark.parametrize("is_noise_scale", [False, True], ids=["weight", "noise"])
    def test_non_positive_scale_restores_the_zero_column(self, is_noise_scale):
        result = mutation_utils._revived_block(
            torch.ones(4),
            0.0,
            make_rng(),
            is_noise_scale=is_noise_scale,
        )

        assert torch.equal(result, torch.zeros(4))

    @pytest.mark.parametrize("is_noise_scale", [False, True], ids=["weight", "noise"])
    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)],
    )
    def test_shape_dtype_and_device_follow_the_template(self, device, is_noise_scale):
        template = torch.zeros(2, 3, dtype=torch.float64, device=device)

        result = mutation_utils._revived_block(
            template,
            1.0,
            make_rng(),
            is_noise_scale=is_noise_scale,
        )

        assert result.shape == template.shape
        assert result.dtype == template.dtype
        assert result.device == template.device

    def test_an_all_zero_draw_restores_the_zero_column(self):
        # A direction of zero norm cannot be rescaled onto the requested one.
        class ZeroDraw:
            def standard_normal(self, size):
                return np.zeros(size)

        result = mutation_utils._revived_block(torch.ones(4), 1.0, ZeroDraw())

        assert torch.equal(result, torch.zeros(4))


class TestModuleTraversalHelpers:
    """Locate weight layers and strip wrappers across the module tree."""

    def test_first_weight_layer_of_an_absent_module_is_none(self):
        assert mutation_utils._first_weight_layer(None) is None

    def test_first_weight_layer_of_a_weightless_module_is_none(self):
        assert mutation_utils._first_weight_layer(nn.Sequential(nn.ReLU())) is None

    def test_unwrap_of_an_absent_module_is_none(self):
        assert mutation_utils._unwrap_module(None) is None

    def test_unwrap_stops_on_a_self_referential_wrapper(self):
        # A cycle must terminate rather than hang.
        module = nn.ReLU()
        module.wrapped = module

        assert mutation_utils._unwrap_module(module) is module

    def test_head_entry_layers_of_an_absent_head_is_empty(self):
        assert mutation_utils._head_entry_layers(None) == []

    def test_flat_head_reports_a_single_entry_layer(self):
        head = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

        result = mutation_utils._head_entry_layers(head)

        assert result == [head[0]]

    def test_branched_head_reports_one_entry_layer_per_stream(self):
        head = nn.ModuleDict(
            {
                "value": nn.Sequential(nn.Linear(4, 3)),
                "advantage": nn.Sequential(nn.Linear(4, 2)),
            },
        )

        result = mutation_utils._head_entry_layers(head)

        assert len(result) == 2

    def test_weight_accessor_rejects_a_layer_with_no_weight_tensor(self):
        with pytest.raises(TypeError, match="exposes no weight tensor"):
            mutation_utils._weight_param(nn.ReLU())


class TestResolveConsumers:
    """Pair a producer with the consumers whose columns are safe to rewrite."""

    def test_dense_consumer_is_matched_with_a_unit_stride(self):
        producer = nn.Linear(3, 5)
        consumer = nn.Linear(5, 2)

        result = mutation_utils._resolve_consumers(producer, [consumer], None, None)

        assert [target.stride for target in result] == [1]

    def test_conv_to_dense_consumer_takes_the_flattened_spatial_stride(self):
        producer = nn.Conv2d(3, 4, kernel_size=3)
        consumer = nn.Linear(16, 2)

        result = mutation_utils._resolve_consumers(producer, [consumer], 4, 4)

        assert [target.stride for target in result] == [4]

    def test_consumer_spending_the_wrong_number_of_columns_is_skipped(self):
        # A nested sub-encoder's features are only a slice of a fusion layer's
        # input, so rewriting its columns would corrupt other neurons' weights.
        producer = nn.Linear(3, 5)
        fusion = nn.Linear(9, 2)

        result = mutation_utils._resolve_consumers(producer, [fusion], None, None)

        assert result == []

    def test_conv_to_dense_without_a_known_layout_is_skipped(self):
        producer = nn.Conv2d(3, 4, kernel_size=3)
        consumer = nn.Linear(16, 2)

        result = mutation_utils._resolve_consumers(producer, [consumer], None, None)

        assert result == []

    def test_noisy_consumer_contributes_its_noise_columns_too(self):
        producer = nn.Linear(3, 5)
        consumer = NoisyLinear(5, 2)

        result = mutation_utils._resolve_consumers(producer, [consumer], None, None)

        # The mean weight and its parallel noise scale ride together.
        assert len(result) == 2

    def test_dense_producer_feeding_a_convolution_is_skipped(self):
        # A boundary the surgery has no column layout for.
        producer = nn.Linear(3, 5)
        consumer = nn.Conv2d(5, 2, kernel_size=3)

        result = mutation_utils._resolve_consumers(producer, [consumer], None, None)

        assert result == []


class TestConv1dBlocks:
    """EvolvableCNN builds Conv1d stacks too, so they must be reachable."""

    @pytest.mark.parametrize(
        "layer",
        [nn.Conv1d(3, 4, 3), nn.Conv2d(3, 4, 3), nn.Conv3d(3, 4, 3)],
        ids=["Conv1d", "Conv2d", "Conv3d"],
    )
    def test_every_evolvable_cnn_block_type_is_a_weight_layer(self, layer):
        assert mutation_utils._is_weight_layer(layer)

    def test_conv1d_pair_is_a_conv_to_conv_boundary(self):
        targets = mutation_utils._resolve_consumers(
            nn.Conv1d(3, 4, 3),
            [nn.Conv1d(4, 2, 3)],
            None,
            None,
        )

        assert [target.stride for target in targets] == [1]

    def test_conv1d_to_dense_consumer_takes_the_flattened_spatial_stride(self):
        producer = nn.Conv1d(3, 4, kernel_size=3)
        consumer = nn.Linear(16, 2)

        result = mutation_utils._resolve_consumers(producer, [consumer], 4, 4)

        assert [target.stride for target in result] == [4]

    def test_conv1d_producer_is_found_rather_than_silently_skipped(self):
        encoder = nn.Sequential(
            nn.Conv1d(3, 4, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(4, 2, kernel_size=3),
            nn.ReLU(),
        )

        result = mutation_utils._resolve_producer_and_next(encoder[1], encoder, None)

        assert result.producer is encoder[0]
        assert result.consumers == [encoder[2]]

    def test_dormant_conv1d_filters_are_reset_and_live_ones_untouched(self):
        # 4 feature maps, of which 1 and 2 are dormant.
        producer = nn.Conv1d(3, 4, kernel_size=3)
        consumer = nn.Conv1d(4, 2, kernel_size=3)
        with torch.no_grad():
            producer.weight.fill_(1.0)
            producer.bias.fill_(7.0)
            consumer.weight.fill_(2.0)
        per_neuron = torch.tensor([1.0, 0.0, 0.0, 1.0])

        indices = reset_producer(producer, [consumer], per_neuron)

        # Assert whole filters move, and the live ones are bit-identical.
        assert indices == [1, 2]
        for index in indices:
            assert not torch.allclose(producer.weight[index], torch.ones(3, 3))
            assert producer.bias[index].item() == 0.0
        for index in (0, 3):
            assert torch.equal(producer.weight[index], torch.ones(3, 3))
            assert producer.bias[index].item() == pytest.approx(7.0)
            assert torch.equal(consumer.weight[:, index], torch.full((2, 3), 2.0))

    def test_revived_conv1d_filter_columns_are_scaled_to_the_live_reference(self):
        producer = nn.Conv1d(3, 4, kernel_size=3)
        consumer = nn.Conv1d(4, 2, kernel_size=3)
        with torch.no_grad():
            consumer.weight.fill_(2.0)
        live = consumer.weight[:, 0].norm().item()

        reset_producer(producer, [consumer], torch.tensor([1.0, 0.0, 1.0, 1.0]))

        assert consumer.weight[:, 1].norm().item() == pytest.approx(
            mutation_utils.REGRAMA_OUT_SCALE * live,
            rel=1e-5,
        )


class TestResetNoisyLayers:
    """A revived noisy neuron starts from the layer's initial noise scale."""

    def test_revived_rows_are_reseeded_at_the_initial_sigma(self):
        producer = NoisyLinear(3, 5, std_init=0.5)
        consumer = nn.Linear(5, 2)
        with torch.no_grad():
            producer.weight_sigma.fill_(1e-8)
            producer.bias_sigma.fill_(1e-8)
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        indices = reset_producer(producer, [consumer], per_neuron)

        assert producer.weight_sigma[indices[0]].abs().max().item() == pytest.approx(
            0.5 / np.sqrt(3),
            rel=1e-5,
        )
        assert producer.bias_sigma[indices[0]].item() == pytest.approx(
            0.5 / np.sqrt(5),
            rel=1e-5,
        )

    def test_live_neurons_keep_their_learned_noise_scale(self):
        producer = NoisyLinear(3, 5, std_init=0.5)
        consumer = nn.Linear(5, 2)
        with torch.no_grad():
            producer.weight_sigma.fill_(1e-8)
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        reset_producer(producer, [consumer], per_neuron)

        assert producer.weight_sigma[0].abs().max().item() == pytest.approx(1e-8)

    def test_consumer_noise_columns_stay_non_negative(self):
        producer = nn.Linear(3, 5)
        consumer = NoisyLinear(5, 32)
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        indices = reset_producer(producer, [consumer], per_neuron)

        assert (consumer.weight_sigma[:, indices[0]] >= 0).all()

    def test_consumer_noise_column_stays_uniform(self):
        producer = nn.Linear(3, 5)
        consumer = NoisyLinear(5, 4)
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        indices = reset_producer(producer, [consumer], per_neuron)

        assert consumer.weight_sigma[:, indices[0]].unique().numel() == 1

    def test_consumer_noise_column_is_damped_against_its_own_scale(self):
        # Leaving the noise columns behind would let a revived neuron inject a
        # full-magnitude perturbation on a path whose mean weight is tiny.
        producer = nn.Linear(3, 5)
        consumer = NoisyLinear(5, 2)
        initial = consumer.weight_sigma[0, 0].item()
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        indices = reset_producer(producer, [consumer], per_neuron)

        assert consumer.weight_sigma[:, indices[0]].max().item() == pytest.approx(
            mutation_utils.REGRAMA_OUT_SCALE * initial,
            rel=1e-5,
        )

    def test_untouched_neurons_keep_their_consumer_noise_columns(self):
        producer = nn.Linear(3, 5)
        consumer = NoisyLinear(5, 2)
        before = consumer.weight_sigma.data.clone()
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        indices = reset_producer(producer, [consumer], per_neuron)

        keep = [n for n in range(5) if n not in indices]
        assert torch.equal(consumer.weight_sigma.data[:, keep], before[:, keep])


class TestResetNormalisedLayers:
    """A revived neuron's normalisation entry returns to the identity."""

    @pytest.mark.parametrize(
        "norm",
        [nn.LayerNorm(5), nn.BatchNorm1d(5)],
        ids=["layer_norm", "batch_norm"],
    )
    def test_revived_neuron_gets_a_neutral_affine(self, norm):
        # A decayed gain would immediately re-suppress the new unit.
        producer = nn.Linear(3, 5)
        consumer = nn.Linear(5, 2)
        with torch.no_grad():
            norm.weight.fill_(0.01)
            norm.bias.fill_(4.0)
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        indices = reset_producer(producer, [consumer], per_neuron, norm=norm)

        assert norm.weight[indices[0]].item() == pytest.approx(1.0)
        assert norm.bias[indices[0]].item() == pytest.approx(0.0)

    def test_live_neurons_keep_their_normalisation_state(self):
        producer = nn.Linear(3, 5)
        consumer = nn.Linear(5, 2)
        norm = nn.LayerNorm(5)
        with torch.no_grad():
            norm.weight.fill_(0.01)
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        reset_producer(producer, [consumer], per_neuron, norm=norm)

        assert norm.weight[0].item() == pytest.approx(0.01)

    def test_revived_neuron_gets_fresh_running_statistics(self):
        producer = nn.Linear(3, 5)
        consumer = nn.Linear(5, 2)
        norm = nn.BatchNorm1d(5)
        with torch.no_grad():
            norm.running_mean.fill_(9.0)
            norm.running_var.fill_(9.0)
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        indices = reset_producer(producer, [consumer], per_neuron, norm=norm)

        assert norm.running_mean[indices[0]].item() == pytest.approx(0.0)
        assert norm.running_var[indices[0]].item() == pytest.approx(1.0)

    def test_normalisation_over_a_different_axis_is_left_alone(self):
        # A norm whose length does not match the producer does not index
        # by neuron, so writing into it would corrupt unrelated state.
        producer = nn.Linear(3, 5)
        consumer = nn.Linear(5, 2)
        norm = nn.LayerNorm(3)
        original = norm.weight.detach().clone()
        per_neuron = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        reset_producer(producer, [consumer], per_neuron, norm=norm)

        assert torch.equal(norm.weight.detach(), original)


class TestResolveProducerAndNext:
    """Locate a measured activation's producing layer, norm and consumers."""

    def test_mlp_encoder_activation_resolves_to_its_own_linear_pair(
        self,
        dqn_agent,
    ):
        network = dqn_agent.actor
        activation = mutation_utils.target_activations(network)[0]

        context = mutation_utils._resolve_producer_and_next(
            activation,
            network.encoder,
            network.head_net,
        )

        assert isinstance(context.producer, nn.Linear)
        assert len(context.consumers) == 1
        assert isinstance(context.consumers[0], nn.Linear)
        assert context.producer.out_features == context.consumers[0].in_features

    def test_norm_between_producer_and_activation_is_reported(self, dqn_agent):
        # The evolvable MLP emits linear -> layer_norm -> activation by default.
        network = dqn_agent.actor
        activation = mutation_utils.target_activations(network)[0]

        context = mutation_utils._resolve_producer_and_next(
            activation,
            network.encoder,
            network.head_net,
        )

        assert isinstance(context.norm, mutation_utils.NORM_LAYER_TYPES)

    def test_norm_preceding_the_producer_is_not_reported(self):
        # A SimBa block is layer_norm -> linear -> activation: the norm applies to
        # the block's input and leaves these neurons alone.
        block = SimbaResidualBlock(hidden_size=8, scale_factor=2)
        encoder = nn.Sequential(block)

        context = mutation_utils._resolve_producer_and_next(block.act, encoder, None)

        assert context.producer is block.linear1
        assert context.norm is None

    def test_residual_block_activation_resolves_between_its_two_convs(self):
        # ResidualBlock is conv1 -> bn1 -> act -> conv2.
        block = ResidualBlock(in_channels=3, kernel_size=3, scale_factor=2)
        encoder = nn.Sequential(block)

        context = mutation_utils._resolve_producer_and_next(block.act, encoder, None)

        assert context.producer is block.conv1
        assert context.norm is block.bn1
        assert context.consumers == [block.conv2]

    def test_residual_block_hidden_channels_are_measured(self):
        block = ResidualBlock(in_channels=3, kernel_size=3, scale_factor=2)
        encoder = nn.Sequential(block)

        measured = mutation_utils._activation_modules(encoder, include_output=False)

        assert block.act in measured

    def test_encoder_latent_resolves_to_every_head_stream(
        self,
        vector_space,
        discrete_space,
    ):
        # A duelling Rainbow head has two parallel streams that both consume the
        # whole latent.
        agent = RainbowDQN(vector_space, discrete_space, device="cpu")
        network = agent.actor
        latent = mutation_utils._activation_modules(
            network.encoder, include_output=True
        )[-1]

        context = mutation_utils._resolve_producer_and_next(
            latent,
            network.encoder,
            network.head_net,
        )

        assert len(context.consumers) == 2

    def test_nested_sub_encoder_tail_resolves_to_the_fusion_layer(
        self,
        dict_space,
        discrete_space,
        encoder_multi_input_config,
    ):
        # EvolvableMultiInput ends each sub-encoder well before the
        # encoder's own output, these neurons feed final_dense.
        agent = DQN(
            dict_space,
            discrete_space,
            net_config=encoder_multi_input_config,
            device="cpu",
        )
        encoder = agent.actor.encoder
        sub_encoder = next(iter(encoder.feature_net.values()))
        tail = mutation_utils._activation_modules(sub_encoder, include_output=True)[-1]

        context = mutation_utils._resolve_producer_and_next(
            tail,
            encoder,
            agent.actor.head_net,
        )

        # The fusion layer, never the head's entry layer.
        assert context.consumers == [encoder.final_dense]

    def test_unknown_activation_resolves_to_nothing(self, dqn_agent):
        network = dqn_agent.actor

        context = mutation_utils._resolve_producer_and_next(
            nn.ReLU(),
            network.encoder,
            network.head_net,
        )

        # The caller skips the layer rather than guessing.
        assert context == mutation_utils.ProducerContext(None, None, [])

    def test_activation_is_found_in_the_head_when_the_encoder_is_absent(self):
        head = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

        context = mutation_utils._resolve_producer_and_next(head[1], None, head)

        assert context.producer is head[0]
        assert context.consumers == [head[2]]


class TestTargetActivations:
    """Choose which activations are measured, and therefore which are reset."""

    @pytest.mark.parametrize(
        ("observation_fixture", "config_fixture"),
        [
            ("vector_space", "encoder_mlp_config"),
            ("image_space", "encoder_cnn_config"),
            ("dict_space", "encoder_multi_input_config"),
            ("vector_space", "encoder_simba_config"),
        ],
        ids=["mlp", "cnn", "multi_input", "simba"],
    )
    def test_every_encoder_activation_is_measured(
        self,
        observation_fixture,
        config_fixture,
        discrete_space,
        request,
    ):
        agent = DQN(
            request.getfixturevalue(observation_fixture),
            discrete_space,
            net_config=request.getfixturevalue(config_fixture),
            device="cpu",
        )
        encoder = agent.actor.encoder
        encoder_activations = {
            id(module)
            for module in encoder.modules()
            if isinstance(module, mutation_utils.ACTIVATION_TYPES)
        }

        measured = {
            id(module) for module in mutation_utils.target_activations(agent.actor)
        }

        assert encoder_activations <= measured

    def test_head_output_activation_is_never_measured(self, dqn_agent):
        # This single exclusion is what stops ReGraMa resetting the units with
        # fixed semantics: action logits or a state value.
        head = dict(dqn_agent.actor.head_net.named_modules())

        measured = {
            id(module) for module in mutation_utils.target_activations(dqn_agent.actor)
        }

        assert id(head["model.value_activation_1"]) in measured
        assert id(head["model.value_activation_output"]) not in measured

    def test_both_duelling_streams_exclude_their_output_activation(
        self,
        vector_space,
        discrete_space,
    ):
        # Dropping only the last activation of head_net would leave the value
        # stream's output measured, because the two streams are siblings.
        agent = RainbowDQN(vector_space, discrete_space, device="cpu")
        head = dict(agent.actor.head_net.named_modules())

        measured = {
            id(module) for module in mutation_utils.target_activations(agent.actor)
        }

        assert id(head["model.value_activation_output"]) not in measured
        assert id(head["advantage_net.advantage_activation_output"]) not in measured

    def test_simba_residual_trunk_is_measured(
        self,
        vector_space,
        discrete_space,
        encoder_simba_config,
    ):
        agent = DQN(
            vector_space,
            discrete_space,
            net_config=encoder_simba_config,
            device="cpu",
        )
        blocks = [
            module
            for module in agent.actor.encoder.modules()
            if isinstance(module, SimbaResidualBlock)
        ]

        measured = {
            id(module) for module in mutation_utils.target_activations(agent.actor)
        }

        assert blocks
        assert all(id(block.act) in measured for block in blocks)

    def test_activations_are_recognised_by_type_not_by_name(self):
        root = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))

        result = mutation_utils._activation_modules(root, include_output=True)

        assert [type(module) for module in result] == [nn.Tanh]


class TestSnapshotPairing:
    """Pair captured scores with the layers they were captured from."""

    @pytest.mark.parametrize("dormant_index", [0, 1, 2])
    def test_layers_are_paired_by_position(self, dqn_agent, dormant_index):
        # The snapshot carries no layer identity, so a score is attributed by its
        # index alone. A reset rewrites its own producer's rows and its consumers'
        # columns, both at or downstream of that layer, so every producer ahead of
        # the dormant one must come through untouched.
        network = dqn_agent.actor
        producers = [
            mutation_utils._resolve_producer_and_next(
                activation,
                network.encoder,
                network.head_net,
            ).producer
            for activation in mutation_utils.target_activations(network)
        ]
        scores = [
            torch.ones(mutation_utils._weight_param(producer).shape[0])
            for producer in producers
        ]
        scores[dormant_index] = torch.zeros_like(scores[dormant_index])
        before = [producer.weight.detach().clone() for producer in producers]

        report = mutation_utils.reset_dormant_neurons(network, scores, 0.01, make_rng())

        assert report == scores[dormant_index].numel()
        assert not torch.equal(
            producers[dormant_index].weight.detach(),
            before[dormant_index],
        )
        assert all(
            torch.equal(producer.weight.detach(), original)
            for producer, original in zip(
                producers[:dormant_index],
                before[:dormant_index],
                strict=True,
            )
        )

    @pytest.mark.parametrize(
        "snapshot",
        [None, [], [torch.ones(3)]],
        ids=["absent", "empty", "wrong-length"],
    )
    def test_a_snapshot_that_does_not_fit_skips_the_whole_network(
        self,
        dqn_agent,
        snapshot,
    ):
        # Graceful degradation after an architecture mutation rebuilt the network:
        # mis-pairing scores with layers would corrupt unrelated neurons.
        before = [p.detach().clone() for p in dqn_agent.actor.parameters()]

        report = mutation_utils.reset_dormant_neurons(
            dqn_agent.actor,
            snapshot,
            0.01,
            make_rng(),
        )

        assert report == 0
        assert all(
            torch.equal(current.detach(), original)
            for current, original in zip(
                dqn_agent.actor.parameters(),
                before,
                strict=True,
            )
        )

    def test_layers_with_no_captured_gradient_are_dropped(self, dqn_agent):
        # A layer outside the training loss is stored as None and dropped from the
        # pairing rather than scored against the wrong entry.
        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))
        dormant = [
            None if entry is None else torch.zeros_like(entry)
            for entry in dqn_agent.grama_scores[0]
        ]

        measured = mutation_utils.reset_dormant_neurons(
            dqn_agent.actor,
            dormant,
            0.01,
            make_rng(),
        )
        unmeasured = mutation_utils.reset_dormant_neurons(
            dqn_agent.actor,
            [None] * len(dormant),
            0.01,
            make_rng(),
        )

        assert measured > 0
        assert unmeasured == 0


class TestResetDormantNeurons:
    """Reset a whole evaluation network from its captured snapshot."""

    def test_a_fully_dormant_network_is_reset(self, dqn_agent):
        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [
            None if entry is None else torch.zeros_like(entry)
            for entry in dqn_agent.grama_scores[0]
        ]
        measured = sum(entry.numel() for entry in scores if entry is not None)
        before = {k: v.clone() for k, v in dqn_agent.actor.state_dict().items()}

        report = mutation_utils.reset_dormant_neurons(
            dqn_agent.actor,
            scores,
            0.01,
            make_rng(),
        )

        assert 0 < report <= measured
        after = dqn_agent.actor.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)

    def test_a_healthy_network_is_left_untouched(self, dqn_agent):
        # Arrange: a uniform snapshot normalises to a score of 1.0 everywhere, so
        # no neuron is dormant at any threshold below one.
        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [
            None if entry is None else torch.ones_like(entry)
            for entry in dqn_agent.grama_scores[0]
        ]
        before = {k: v.clone() for k, v in dqn_agent.actor.state_dict().items()}

        report = mutation_utils.reset_dormant_neurons(
            dqn_agent.actor,
            scores,
            0.01,
            make_rng(),
        )

        assert report == 0
        after = dqn_agent.actor.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_missing_snapshot_is_a_no_op(self, dqn_agent):
        report = mutation_utils.reset_dormant_neurons(
            dqn_agent.actor, None, 0.01, make_rng()
        )

        assert report == 0

    def test_snapshot_of_the_wrong_width_skips_that_layer(self, dqn_agent):
        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [torch.zeros(entry.numel() + 1) for entry in dqn_agent.grama_scores[0]]
        before = {k: v.clone() for k, v in dqn_agent.actor.state_dict().items()}

        report = mutation_utils.reset_dormant_neurons(
            dqn_agent.actor,
            scores,
            0.01,
            make_rng(),
        )

        assert report == 0
        after = dqn_agent.actor.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_head_output_layer_is_never_treated_as_a_producer(self, dqn_agent):
        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [torch.zeros_like(entry) for entry in dqn_agent.grama_scores[0]]
        output_layer = dqn_agent.actor.head_net.model.value_linear_layer_output
        with torch.no_grad():
            output_layer.bias.fill_(3.0)

        mutation_utils.reset_dormant_neurons(dqn_agent.actor, scores, 0.01, make_rng())

        # A producer's dormant neurons have their bias zeroed, so a surviving bias
        # is the observable proof the layer was never treated as one.
        assert torch.equal(
            output_layer.bias.detach(),
            torch.full_like(output_layer.bias, 3.0),
        )

    def test_head_output_layer_columns_are_still_rewritten(self, dqn_agent):
        # It consumes the last hidden activation, so when those neurons are
        # reset its columns must follow. That half is not excluded.
        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [torch.zeros_like(entry) for entry in dqn_agent.grama_scores[0]]
        output_layer = dqn_agent.actor.head_net.model.value_linear_layer_output
        before = output_layer.weight.detach().clone()

        mutation_utils.reset_dormant_neurons(dqn_agent.actor, scores, 0.01, make_rng())

        assert not torch.equal(output_layer.weight.detach(), before)

    def test_conv_encoder_is_reset_across_the_flatten_boundary(
        self,
        image_space,
        discrete_space,
        encoder_cnn_config,
    ):
        agent = DQN(
            image_space,
            discrete_space,
            net_config=encoder_cnn_config,
            device="cpu",
        )
        capture_grama_snapshot(agent, torch.rand(2, 3, 32, 32))
        scores = [
            None if entry is None else torch.zeros_like(entry)
            for entry in agent.grama_scores[0]
        ]

        report = mutation_utils.reset_dormant_neurons(
            agent.actor, scores, 0.01, make_rng()
        )

        assert report > 0
        assert all(
            torch.isfinite(value).all() for value in agent.actor.state_dict().values()
        )

    def test_multi_input_encoder_is_reset(
        self,
        dict_space,
        discrete_space,
        encoder_multi_input_config,
    ):
        agent = DQN(
            dict_space,
            discrete_space,
            net_config=encoder_multi_input_config,
            device="cpu",
        )
        agent.capture_grama = True
        agent.init_training_step()
        observation = {
            key: torch.rand(2, *space.shape) for key, space in dict_space.items()
        }
        agent.actor(observation).square().mean().backward()
        agent.finalize_training_step(1)
        scores = [
            None if entry is None else torch.zeros_like(entry)
            for entry in agent.grama_scores[0]
        ]

        report = mutation_utils.reset_dormant_neurons(
            agent.actor, scores, 0.01, make_rng()
        )

        # 5 conv channels per sub-encoder, the 8-unit latent, and the head's two
        # 8-unit layers. The two 32-unit sub-encoder tails are excluded: their
        # features are one offset slice of the fusion layer's input.
        assert report == 5 + 5 + 8 + 8 + 8
        assert all(
            torch.isfinite(value).all() for value in agent.actor.state_dict().values()
        )

    def test_multi_input_sub_encoder_tails_are_left_unreset(
        self,
        dict_space,
        discrete_space,
        encoder_multi_input_config,
    ):
        # A sub-encoder's features are one offset slice of the fusion layer's
        # input, so its tail has no consumer the surgery can index and is skipped
        # rather than rewritten over columns belonging to other sub-encoders.
        agent = DQN(
            dict_space,
            discrete_space,
            net_config=encoder_multi_input_config,
            device="cpu",
        )
        agent.capture_grama = True
        agent.init_training_step()
        observation = {
            key: torch.rand(2, *space.shape) for key, space in dict_space.items()
        }
        agent.actor(observation).square().mean().backward()
        agent.finalize_training_step(1)

        tails = {
            id(mutation_utils._activation_modules(sub_encoder, include_output=True)[-1])
            for sub_encoder in agent.actor.encoder.feature_net.values()
        }
        scores = []
        for activation, entry in zip(
            mutation_utils.target_activations(agent.actor),
            agent.grama_scores[0],
            strict=True,
        ):
            if entry is None:
                scores.append(None)
            else:
                # Scores are normalised by their layer mean, so a uniform fill of
                # zero reads as dormant and any positive fill reads as healthy.
                dormant = id(activation) in tails
                scores.append(
                    torch.zeros_like(entry) if dormant else torch.ones_like(entry)
                )
        before = {
            name: value.clone() for name, value in agent.actor.state_dict().items()
        }

        report = mutation_utils.reset_dormant_neurons(
            agent.actor, scores, 0.01, make_rng()
        )

        assert report == 0
        assert all(
            torch.equal(value, before[name])
            for name, value in agent.actor.state_dict().items()
        )

    def test_recurrent_core_is_left_unreset(
        self,
        vector_space,
        discrete_space,
    ):
        # An LSTM's gate non-linearities are fused, so no per-neuron gradient is
        # captured for them and its hidden units own no contiguous weight rows:
        # only the layers from its output projection onward are ever reset.
        agent = PPO(
            vector_space,
            discrete_space,
            recurrent=True,
            net_config={"encoder_config": {"hidden_state_size": 8}},
            device="cpu",
        )
        # Every measured layer is marked fully dormant, so anything the surgery
        # could reach would be.
        scores = [
            torch.zeros(
                mutation_utils._weight_param(
                    mutation_utils._resolve_producer_and_next(
                        activation,
                        agent.actor.encoder,
                        agent.actor.head_net,
                    ).producer,
                ).shape[0],
            )
            for activation in mutation_utils.target_activations(agent.actor)
        ]
        lstm_before = {
            name: value.clone()
            for name, value in agent.actor.state_dict().items()
            if "lstm." in name
        }

        mutation_utils.reset_dormant_neurons(agent.actor, scores, 0.01, make_rng())

        assert lstm_before
        after = agent.actor.state_dict()
        assert all(
            torch.equal(value, after[name]) for name, value in lstm_before.items()
        )

    def test_borrowed_encoder_parameters_are_not_reset(
        self,
        vector_space,
        discrete_space,
    ):
        # share_encoder_parameters pins the critic's encoder to detached clones of
        # the actor's, which the mutation hook re-pins moments later, so writing
        # there is discarded while the matching head rewrite survives, leaving the
        # head compensating a reset that no longer exists.
        agent = PPO(vector_space, discrete_space, device="cpu")
        share_encoder_parameters(agent.actor, agent.critic)
        # Every measured layer of the critic is marked fully dormant.
        scores = [
            torch.zeros(
                mutation_utils._weight_param(
                    mutation_utils._resolve_producer_and_next(
                        activation,
                        agent.critic.encoder,
                        agent.critic.head_net,
                    ).producer,
                ).shape[0],
            )
            for activation in mutation_utils.target_activations(agent.critic)
        ]
        before = {
            name: value.clone()
            for name, value in agent.critic.encoder.state_dict().items()
        }

        mutation_utils.reset_dormant_neurons(agent.critic, scores, 0.01, make_rng())

        after = agent.critic.encoder.state_dict()
        assert all(torch.equal(before[name], after[name]) for name in before)


class TestSharedLatentBlocks:
    """Expose a shared latent's columns inside every borrowing head."""

    def test_dense_entry_exposes_its_leading_latent_columns(self):
        producer = nn.Linear(4, 8)

        result = mutation_utils._shared_latent_blocks(producer, [nn.Linear(10, 5)])

        assert [tuple(target.weight.shape) for target in result] == [(5, 8)]

    def test_noisy_entry_also_exposes_its_noise_columns(self):
        producer = nn.Linear(4, 8)

        result = mutation_utils._shared_latent_blocks(producer, [NoisyLinear(8, 5)])

        assert [target.is_noise_scale for target in result] == [False, True]

    @pytest.mark.parametrize(
        "make_entry",
        [lambda: nn.Linear(4, 5), lambda: nn.Conv2d(8, 4, 3)],
        ids=["narrower-than-the-latent", "not-a-dense-layer"],
    )
    def test_an_entry_that_cannot_hold_the_latent_is_skipped(self, make_entry):
        producer = nn.Linear(4, 8)

        result = mutation_utils._shared_latent_blocks(producer, [make_entry()])

        assert result == []

    def test_a_convolutional_producer_has_no_latent_columns_to_share(self):
        # A shared latent crosses the encoder-head boundary as a flat vector,
        # so a conv producer is never the layer feeding a sharing head.
        producer = nn.Conv2d(3, 8, 3)

        result = mutation_utils._shared_latent_blocks(producer, [nn.Linear(8, 5)])

        assert result == []


class TestSharedEncoderCompensation:
    """A latent shared by several networks is faded in every one of them."""

    def ppo(self, vector_space, discrete_space, *, share):
        return PPO(vector_space, discrete_space, share_encoders=share, device="cpu")

    def critic_head(self, network):
        return mutation_utils._head_entry_layers(network.head_net)[0]

    def reset_actor_latent(self, agent, index=0):
        """Reset one latent unit of the policy, compensating shared consumers."""
        return mutation_utils.reset_dormant_neurons(
            agent.actor,
            latent_marked_dormant(agent.actor, index),
            0.01,
            make_rng(),
            shared_latent_heads=mutation_utils.shared_encoder_heads(
                agent.eval_networks(), None, agent.actor
            ),
        )

    def test_only_the_network_that_owns_the_encoder_has_shared_consumers(
        self,
        vector_space,
        discrete_space,
    ):
        agent = self.ppo(vector_space, discrete_space, share=True)

        assert mutation_utils.shared_encoder_heads(
            agent.eval_networks(), None, agent.actor
        )
        assert (
            mutation_utils.shared_encoder_heads(
                agent.eval_networks(), None, agent.critic
            )
            == []
        )

    def test_a_network_without_an_encoder_is_not_a_shared_consumer(
        self,
        vector_space,
        discrete_space,
    ):
        # Nothing borrowed, so nothing to compensate either.
        agent = self.ppo(vector_space, discrete_space, share=True)
        networks = [
            (None, agent.actor),
            (None, nn.Sequential(nn.Linear(4, 2))),
        ]

        assert mutation_utils.shared_encoder_heads(networks, None, agent.actor) == []

    def test_the_critic_head_is_reported_as_a_shared_consumer(
        self,
        vector_space,
        discrete_space,
    ):
        agent = self.ppo(vector_space, discrete_space, share=True)

        result = mutation_utils.shared_encoder_heads(
            agent.eval_networks(), None, agent.actor
        )

        assert result == mutation_utils._head_entry_layers(agent.critic.head_net)

    def test_unshared_encoders_report_no_shared_consumer(
        self,
        vector_space,
        discrete_space,
    ):
        # A network that owns its encoder compensates its own head, so pulling it
        # in here would fade the same column twice.
        agent = self.ppo(vector_space, discrete_space, share=False)

        assert (
            mutation_utils.shared_encoder_heads(
                agent.eval_networks(), None, agent.actor
            )
            == []
        )

    def test_reset_latent_is_faded_in_the_critic_head_too(
        self,
        vector_space,
        discrete_space,
    ):
        agent = self.ppo(vector_space, discrete_space, share=True)
        head = self.critic_head(agent.critic)
        live = head.weight.data[:, 1:].norm(dim=0).median().item()

        self.reset_actor_latent(agent)

        assert head.weight.data[:, 0].norm().item() == pytest.approx(
            mutation_utils.REGRAMA_OUT_SCALE * live,
            rel=1e-5,
        )

    def test_latents_left_alone_keep_their_critic_columns(
        self,
        vector_space,
        discrete_space,
    ):
        agent = self.ppo(vector_space, discrete_space, share=True)
        head = self.critic_head(agent.critic)
        before = head.weight.data.clone()

        self.reset_actor_latent(agent)

        assert torch.equal(head.weight.data[:, 1:], before[:, 1:])

    def test_widening_is_opt_in(self, vector_space, discrete_space):
        agent = self.ppo(vector_space, discrete_space, share=True)
        head = self.critic_head(agent.critic)
        before = head.weight.data.clone()

        mutation_utils.reset_dormant_neurons(
            agent.actor,
            latent_marked_dormant(agent.actor, 0),
            0.01,
            make_rng(),
        )

        assert torch.equal(head.weight.data, before)

    def test_td3_critics_are_faded_without_touching_their_action_columns(
        self,
        vector_space,
    ):
        agent = TD3(vector_space, vector_space, share_encoders=True, device="cpu")
        span = agent.critic_1.latent_dim
        heads = [self.critic_head(net) for net in (agent.critic_1, agent.critic_2)]
        before = [head.weight.data.clone() for head in heads]

        self.reset_actor_latent(agent)

        for head, prior in zip(heads, before, strict=True):
            assert head.weight.data[:, 0].norm() < prior[:, 0].norm()
            assert torch.equal(head.weight.data[:, 1:span], prior[:, 1:span])
            assert torch.equal(head.weight.data[:, span:], prior[:, span:])

    def test_continuous_q_network_consumes_the_latent_first(self, vector_space):
        agent = TD3(vector_space, vector_space, share_encoders=False, device="cpu")
        head = self.critic_head(agent.critic_1)
        span = agent.critic_1.latent_dim
        with torch.no_grad():
            head.weight[:, :span] = 0.0

        action = torch.rand(1, 4)
        with torch.no_grad():
            same_action = [agent.critic_1(torch.rand(1, 4), action) for _ in range(2)]
            other_action = agent.critic_1(torch.rand(1, 4), torch.rand(1, 4))

        assert torch.equal(*same_action)
        assert not torch.equal(same_action[0], other_action)
