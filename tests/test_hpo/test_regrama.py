# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest
import torch
from torch import nn

from agilerl.algorithms import DQN, PPO, TD3, RainbowDQN
from agilerl.hpo import regrama
from agilerl.modules.custom_components import (
    NoisyLinear,
    ResidualBlock,
    SimbaResidualBlock,
)
from agilerl.utils.algo_utils import share_encoder_parameters


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
    consumers = regrama.resolve_consumers(
        producer,
        next_layers,
        cnn_channels,
        cnn_spatial,
    )
    indices = regrama.dormant_indices(per_neuron, dormant_threshold)
    regrama.reset_layer_neurons(producer, consumers, norm, indices, make_rng(seed))
    return indices


def capture_snapshot(agent, observation) -> None:
    """Run one real backward pass through agent's policy under capture."""
    agent.capture_grama = True
    agent.init_training_step()
    policy = getattr(agent, agent.registry.policy())
    head = regrama.unwrap_module(policy.head_net)
    head(policy.encoder(observation)).square().mean().backward()
    agent.finalize_training_step(1)


def mlp_net_config() -> dict:
    """Return a fresh MLP net config.

    Deliberately not the shared encoder_mlp_config fixture.
    """
    return {
        "latent_dim": 8,
        "min_latent_dim": 1,
        "encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
        "head_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
    }


def latent_marked_dormant(network, index: int) -> list[torch.Tensor | None]:
    """Snapshot for network in which only latent unit index is dormant."""
    encoder, head = network.encoder, network.head_net
    terminal = regrama.activation_modules(encoder, include_output=True)[-1]
    scores: list[torch.Tensor | None] = []
    for activation in regrama.target_activations(network):
        producer = regrama.resolve_producer_and_next(activation, encoder, head).producer
        if producer is None:
            scores.append(None)
            continue
        per_neuron = torch.ones(regrama.weight_param(producer).shape[0])
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


class TestPerNeuronGrad:
    """Reduce a backward hook's grad_input to one magnitude per neuron."""

    def test_dense_gradient_averages_over_the_batch(self):
        gradient = torch.tensor([[1.0, -2.0, 3.0], [1.0, -2.0, 3.0]])

        result = regrama.per_neuron_grad((gradient,))

        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_conv_gradient_averages_over_batch_and_spatial(self):
        # (batch=2, channels=3, 4, 4), one constant per channel.
        gradient = torch.ones(2, 3, 4, 4)
        gradient[:, 1] = -5.0
        gradient[:, 2] = 2.0

        result = regrama.per_neuron_grad((gradient,))

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([1.0, 5.0, 2.0]))

    def test_absolute_value_is_taken_before_the_reduction(self):
        # A neuron whose gradient cancels across the batch.
        gradient = torch.tensor([[4.0], [-4.0]])

        result = regrama.per_neuron_grad((gradient,))

        # Assert 4.0, not 0.0.
        assert torch.allclose(result, torch.tensor([4.0]))

    def test_already_per_neuron_gradient_is_returned_unreduced(self):
        gradient = torch.tensor([1.0, -2.0])

        result = regrama.per_neuron_grad((gradient,))

        assert torch.allclose(result, torch.tensor([1.0, 2.0]))

    @pytest.mark.parametrize("grad_input", [(None,), None, ()])
    def test_missing_gradient_is_unmeasured_rather_than_zero(self, grad_input):
        result = regrama.per_neuron_grad(grad_input)

        assert result is None


class TestDormantIndices:
    """Select the neurons whose normalised GraMa score is at or below tau."""

    def test_threshold_applies_to_the_layer_normalised_score(self):
        # Mean is 1.0, so the scores are the values themselves.
        per_neuron = torch.tensor([2.0, 0.05, 1.0, 0.95])

        result = regrama.dormant_indices(per_neuron, 0.1)

        assert result == [1]

    def test_threshold_is_scale_invariant(self):
        per_neuron = torch.tensor([2.0, 0.05, 1.0, 0.95])

        scaled = regrama.dormant_indices(per_neuron * 1000.0, 0.1)

        # Normalising by the layer mean removes the overall scale.
        assert scaled == regrama.dormant_indices(per_neuron, 0.1)

    def test_layer_with_no_gradient_anywhere_is_entirely_dormant(self):
        result = regrama.dormant_indices(torch.zeros(4), 0.01)

        assert result == [0, 1, 2, 3]

    def test_zero_threshold_selects_only_exactly_dead_neurons(self):
        per_neuron = torch.tensor([1.0, 0.0, 1e-9, 2.0])

        result = regrama.dormant_indices(per_neuron, 0.0)

        assert result == [1]

    def test_non_finite_scores_are_treated_as_dormant(self):
        # S diverged unit is exactly one worth re-initialising.
        per_neuron = torch.tensor([1.0, float("nan"), float("inf"), 2.0])

        result = regrama.dormant_indices(per_neuron, 0.01)

        assert result == [1, 2]

    def test_empty_layer_selects_nothing(self):
        assert regrama.dormant_indices(torch.empty(0), 0.01) == []


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

        expected = regrama.REGRAMA_OUT_SCALE * live
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
            regrama.REGRAMA_OUT_SCALE * live,
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

        result = regrama.live_column_scale(weight, 1, list(range(6)))

        assert result == pytest.approx(expected, rel=1e-6)

    def test_conv_to_dense_boundary_measures_whole_strided_blocks(self):
        # 3 producer feature maps, each spending 4 flattened columns.
        weight = torch.rand(2, 12)
        blocks = weight.reshape(2, 3, 4)
        expected = float(blocks.pow(2).sum(dim=(0, 2)).sqrt().median())

        result = regrama.live_column_scale(weight, 4, [0, 1, 2])

        assert result == pytest.approx(expected, rel=1e-6)

    def test_conv_consumer_measures_whole_filters_not_kernel_positions(self):
        # The neuron axis is dimension 1.
        weight = torch.rand(2, 3, 3, 3)
        blocks = weight.reshape(2, 3, -1)
        expected = float(blocks.pow(2).sum(dim=(0, 2)).sqrt().median())

        result = regrama.live_column_scale(weight, 1, [0, 1, 2])

        assert result == pytest.approx(expected, rel=1e-6)

    def test_only_the_kept_neurons_are_measured(self):
        # Measuring the whole layer would let a repeatedly-reset layer size each
        # revival against the previous one's.
        weight = torch.ones(2, 5)
        weight[:, :3] = 100.0

        kept = regrama.live_column_scale(weight, 1, [3, 4])
        everything = regrama.live_column_scale(weight, 1, [0, 1, 2, 3, 4])

        assert kept < everything

    def test_collapsed_layer_falls_back_to_a_positive_reference(self):
        # A layer with nothing live left must still produce a usable scale, or a
        # revival would be silently zero-scaled and unable to learn.
        weight = torch.zeros(2, 3)

        result = regrama.live_column_scale(weight, 1, [])

        assert result > 0.0

    def test_every_neuron_reset_falls_back_to_the_whole_layer(self):
        # No neuron is kept, so there is no live reference to measure.
        weight = torch.ones(2, 3) * 4.0
        expected = float(weight.pow(2).sum(0).sqrt().median())

        result = regrama.live_column_scale(weight, 1, [])

        # The layer's own median, rather than a fabricated constant.
        assert result == pytest.approx(expected, rel=1e-6)

    def test_a_consumer_with_no_columns_still_yields_a_positive_reference(self):
        # Neither the kept neurons nor the whole layer can supply a reference
        # when there is nothing to measure, so the Xavier bound stands in.
        weight = torch.zeros(2, 0)

        result = regrama.live_column_scale(weight, 1, [])

        assert result > 0.0


class TestRevivedOutBlock:
    """Draw the outgoing weights a reset neuron is revived with."""

    def test_block_is_rescaled_to_the_requested_norm(self):
        result = regrama.revived_out_block(torch.zeros(4), 0.5, make_rng())

        assert result.norm().item() == pytest.approx(0.5, rel=1e-6)

    def test_non_positive_scale_restores_the_zero_column(self):
        result = regrama.revived_out_block(torch.ones(4), 0.0, make_rng())

        assert torch.equal(result, torch.zeros(4))

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)],
    )
    def test_shape_dtype_and_device_follow_the_template(self, device):
        template = torch.zeros(2, 3, dtype=torch.float64, device=device)

        result = regrama.revived_out_block(template, 1.0, make_rng())

        assert result.shape == template.shape
        assert result.dtype == template.dtype
        assert result.device == template.device

    def test_an_all_zero_draw_restores_the_zero_column(self):
        # A direction of zero norm cannot be rescaled onto the requested one.
        class ZeroDraw:
            def standard_normal(self, size):
                return np.zeros(size)

        result = regrama.revived_out_block(torch.ones(4), 1.0, ZeroDraw())

        assert torch.equal(result, torch.zeros(4))


class TestRevivedNoiseBlock:
    """Draw the outgoing noise scales a reset neuron is revived with."""

    def test_block_is_rescaled_to_the_requested_norm(self):
        result = regrama.revived_noise_block(torch.zeros(4), 0.5)

        assert result.norm().item() == pytest.approx(0.5, rel=1e-6)

    def test_block_is_non_negative_and_uniform(self):
        result = regrama.revived_noise_block(torch.zeros(3, 2), 1.0)

        assert (result >= 0).all()
        assert result.unique().numel() == 1

    def test_non_positive_scale_restores_the_zero_column(self):
        result = regrama.revived_noise_block(torch.ones(4), 0.0)

        assert torch.equal(result, torch.zeros(4))

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)],
    )
    def test_shape_dtype_and_device_follow_the_template(self, device):
        template = torch.zeros(2, 3, dtype=torch.float64, device=device)

        result = regrama.revived_noise_block(template, 1.0)

        assert result.shape == template.shape
        assert result.dtype == template.dtype
        assert result.device == template.device


class TestModuleTraversalHelpers:
    """Locate weight layers and strip wrappers across the module tree."""

    def test_first_weight_layer_of_an_absent_module_is_none(self):
        assert regrama.first_weight_layer(None) is None

    def test_first_weight_layer_of_a_weightless_module_is_none(self):
        assert regrama.first_weight_layer(nn.Sequential(nn.ReLU())) is None

    def test_unwrap_of_an_absent_module_is_none(self):
        assert regrama.unwrap_module(None) is None

    def test_unwrap_stops_on_a_self_referential_wrapper(self):
        # A cycle must terminate rather than hang.
        module = nn.ReLU()
        module.wrapped = module

        assert regrama.unwrap_module(module) is module

    def test_head_entry_layers_of_an_absent_head_is_empty(self):
        assert regrama.head_entry_layers(None) == []

    def test_flat_head_reports_a_single_entry_layer(self):
        head = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

        result = regrama.head_entry_layers(head)

        assert result == [head[0]]

    def test_branched_head_reports_one_entry_layer_per_stream(self):
        head = nn.ModuleDict(
            {
                "value": nn.Sequential(nn.Linear(4, 3)),
                "advantage": nn.Sequential(nn.Linear(4, 2)),
            },
        )

        result = regrama.head_entry_layers(head)

        assert len(result) == 2

    def test_cnn_dims_of_an_absent_encoder_is_empty(self):
        assert regrama.cnn_dims_by_module(None) == {}

    def test_weight_accessor_rejects_a_layer_with_no_weight_tensor(self):
        with pytest.raises(TypeError, match="exposes no weight tensor"):
            regrama.weight_param(nn.ReLU())

    def test_bias_accessor_reports_no_bias_rather_than_raising(self):
        # A bias is genuinely optional, so its absence is a normal outcome.
        assert regrama.bias_param(nn.Linear(3, 2, bias=False)) is None


class TestResolveConsumers:
    """Pair a producer with the consumers whose columns are safe to rewrite."""

    def test_dense_consumer_is_matched_with_a_unit_stride(self):
        producer = nn.Linear(3, 5)
        consumer = nn.Linear(5, 2)

        result = regrama.resolve_consumers(producer, [consumer], None, None)

        assert [target.stride for target in result] == [1]

    def test_conv_to_dense_consumer_takes_the_flattened_spatial_stride(self):
        producer = nn.Conv2d(3, 4, kernel_size=3)
        consumer = nn.Linear(16, 2)

        result = regrama.resolve_consumers(producer, [consumer], 4, 4)

        assert [target.stride for target in result] == [4]

    def test_consumer_spending_the_wrong_number_of_columns_is_skipped(self):
        # A nested sub-encoder's features are only a slice of a fusion layer's
        # input, so rewriting its columns would corrupt other neurons' weights.
        producer = nn.Linear(3, 5)
        fusion = nn.Linear(9, 2)

        result = regrama.resolve_consumers(producer, [fusion], None, None)

        assert result == []

    def test_conv_to_dense_without_a_known_layout_is_skipped(self):
        producer = nn.Conv2d(3, 4, kernel_size=3)
        consumer = nn.Linear(16, 2)

        result = regrama.resolve_consumers(producer, [consumer], None, None)

        assert result == []

    def test_noisy_consumer_contributes_its_noise_columns_too(self):
        producer = nn.Linear(3, 5)
        consumer = NoisyLinear(5, 2)

        result = regrama.resolve_consumers(producer, [consumer], None, None)

        # The mean weight and its parallel noise scale ride together.
        assert len(result) == 2

    def test_dense_producer_feeding_a_convolution_is_skipped(self):
        # A boundary the surgery has no column layout for.
        producer = nn.Linear(3, 5)
        consumer = nn.Conv2d(5, 2, kernel_size=3)

        result = regrama.resolve_consumers(producer, [consumer], None, None)

        assert result == []


class TestConv1dBlocks:
    """EvolvableCNN builds Conv1d stacks too, so they must be reachable."""

    @pytest.mark.parametrize(
        "layer",
        [nn.Conv1d(3, 4, 3), nn.Conv2d(3, 4, 3), nn.Conv3d(3, 4, 3)],
        ids=["Conv1d", "Conv2d", "Conv3d"],
    )
    def test_every_evolvable_cnn_block_type_is_a_weight_layer(self, layer):
        assert regrama.is_weight_layer(layer)

    def test_conv1d_pair_is_a_conv_to_conv_boundary(self):
        result = regrama.boundary_kind(nn.Conv1d(3, 4, 3), nn.Conv1d(4, 2, 3))

        assert result == "conv_conv"

    def test_conv1d_to_dense_consumer_takes_the_flattened_spatial_stride(self):
        producer = nn.Conv1d(3, 4, kernel_size=3)
        consumer = nn.Linear(16, 2)

        result = regrama.resolve_consumers(producer, [consumer], 4, 4)

        assert [target.stride for target in result] == [4]

    def test_conv1d_producer_is_found_rather_than_silently_skipped(self):
        encoder = nn.Sequential(
            nn.Conv1d(3, 4, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(4, 2, kernel_size=3),
            nn.ReLU(),
        )

        result = regrama.resolve_producer_and_next(encoder[1], encoder, None)

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
            regrama.REGRAMA_OUT_SCALE * live,
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
            regrama.REGRAMA_OUT_SCALE * initial,
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
        # S norm whose length does not match the producer does not index
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
        activation = regrama.target_activations(network)[0]

        context = regrama.resolve_producer_and_next(
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
        activation = regrama.target_activations(network)[0]

        context = regrama.resolve_producer_and_next(
            activation,
            network.encoder,
            network.head_net,
        )

        assert regrama.is_norm_layer(context.norm)

    def test_norm_preceding_the_producer_is_not_reported(self):
        # A SimBa block is layer_norm -> linear -> activation: the norm applies to
        # the block's input and leaves these neurons alone.
        block = SimbaResidualBlock(hidden_size=8, scale_factor=2)
        encoder = nn.Sequential(block)

        context = regrama.resolve_producer_and_next(block.act, encoder, None)

        assert context.producer is block.linear1
        assert context.norm is None

    def test_residual_block_activation_resolves_between_its_two_convs(self):
        # ResidualBlock is conv1 -> bn1 -> act -> conv2.
        block = ResidualBlock(in_channels=3, kernel_size=3, scale_factor=2)
        encoder = nn.Sequential(block)

        context = regrama.resolve_producer_and_next(block.act, encoder, None)

        assert context.producer is block.conv1
        assert context.norm is block.bn1
        assert context.consumers == [block.conv2]

    def test_residual_block_hidden_channels_are_measured(self):
        block = ResidualBlock(in_channels=3, kernel_size=3, scale_factor=2)
        encoder = nn.Sequential(block)

        measured = regrama.activation_modules(encoder, include_output=False)

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
        latent = regrama.activation_modules(network.encoder, include_output=True)[-1]

        context = regrama.resolve_producer_and_next(
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
        tail = regrama.activation_modules(sub_encoder, include_output=True)[-1]

        context = regrama.resolve_producer_and_next(
            tail,
            encoder,
            agent.actor.head_net,
        )

        # The fusion layer, never the head's entry layer.
        assert context.consumers == [encoder.final_dense]

    def test_unknown_activation_resolves_to_nothing(self, dqn_agent):
        network = dqn_agent.actor

        context = regrama.resolve_producer_and_next(
            nn.ReLU(),
            network.encoder,
            network.head_net,
        )

        # The caller skips the layer rather than guessing.
        assert context == regrama.ProducerContext(None, None, [])

    def test_activation_is_found_in_the_head_when_the_encoder_is_absent(self):
        head = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

        context = regrama.resolve_producer_and_next(head[1], None, head)

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
            if isinstance(module, regrama.ACTIVATION_TYPES)
        }

        measured = {id(module) for module in regrama.target_activations(agent.actor)}

        assert encoder_activations <= measured

    def test_head_output_activation_is_never_measured(self, dqn_agent):
        # This single exclusion is what stops ReGraMa resetting the units with
        # fixed semantics: action logits or a state value.
        head = dict(dqn_agent.actor.head_net.named_modules())

        measured = {
            id(module) for module in regrama.target_activations(dqn_agent.actor)
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

        measured = {id(module) for module in regrama.target_activations(agent.actor)}

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

        measured = {id(module) for module in regrama.target_activations(agent.actor)}

        assert blocks
        assert all(id(block.act) in measured for block in blocks)

    def test_activations_are_recognised_by_type_not_by_name(self):
        root = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))

        result = regrama.activation_modules(root, include_output=True)

        assert [type(module) for module in result] == [nn.Tanh]


class TestGraMaCapture:
    """Capture per-neuron pre-activation gradients during a training block."""

    def test_snapshot_layout_matches_the_measured_layers(self, dqn_agent):
        expected = [
            len(regrama.target_activations(network))
            for _network_id, network in regrama.eval_networks(dqn_agent)
        ]

        capture_snapshot(dqn_agent, torch.rand(4, 4))

        assert [len(entry) for entry in dqn_agent.grama_scores] == expected

    def test_captured_widths_match_the_producing_layers(self, dqn_agent):
        capture_snapshot(dqn_agent, torch.rand(4, 4))

        for entry in dqn_agent.grama_scores[0]:
            assert entry is None or entry.dim() == 1

    def test_only_the_last_minibatch_survives(self, dqn_agent):
        # The metric's expectation is taken at fixed parameters, and the reset acts
        # on the network as it stands at the end of the cycle.
        dqn_agent.capture_grama = True
        dqn_agent.init_training_step()
        dqn_agent.actor(torch.ones(4, 4) * 100.0).square().mean().backward()
        small = torch.rand(4, 4) * 1e-3
        dqn_agent.actor(small).square().mean().backward()
        dqn_agent.finalize_training_step(1)
        captured = dqn_agent.grama_scores[0][0]

        # Reproduce the second minibatch alone.
        dqn_agent.capture_grama = True
        dqn_agent.init_training_step()
        dqn_agent.actor(small).square().mean().backward()
        dqn_agent.finalize_training_step(1)

        assert torch.allclose(captured, dqn_agent.grama_scores[0][0], atol=1e-8)

    def test_hooks_are_released_when_the_block_completes(self, dqn_agent):
        activation = regrama.target_activations(dqn_agent.actor)[0]

        capture_snapshot(dqn_agent, torch.rand(4, 4))

        assert not activation._backward_hooks

    def test_hooks_are_released_when_the_block_raises(self, dqn_agent):
        activation = regrama.target_activations(dqn_agent.actor)[0]

        def blow_up() -> None:
            msg = "training blew up"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError), regrama.GraMaCapture(dqn_agent):
            blow_up()

        assert not activation._backward_hooks

    def test_repeated_captures_do_not_accumulate_hooks(self, dqn_agent):
        activation = regrama.target_activations(dqn_agent.actor)[0]

        for _ in range(3):
            capture_snapshot(dqn_agent, torch.rand(4, 4))

        assert not activation._backward_hooks

    def test_layer_outside_the_loss_graph_is_stored_as_unmeasured(
        self,
        vector_space,
        discrete_space,
    ):
        # Only PPO's actor is exercised, so the critic never fires.
        agent = PPO(vector_space, discrete_space, device="cpu")
        capture_snapshot(agent, torch.rand(4, 4))
        networks = [network for _network_id, network in regrama.eval_networks(agent)]
        critic_index = networks.index(agent.critic)

        # Recorded as None, so it is skipped downstream rather than read
        # as a fully dormant layer and needlessly reset.
        assert all(entry is None for entry in agent.grama_scores[critic_index])
        assert any(entry is not None for entry in agent.grama_scores[0])

    def test_a_failing_hook_never_breaks_the_backward_pass(
        self,
        dqn_agent,
        monkeypatch,
    ):
        # A raising backward hook would abort real training.
        def explode(_grad_input):
            msg = "hook blew up"
            raise RuntimeError(msg)

        monkeypatch.setattr(regrama, "per_neuron_grad", explode)

        capture_snapshot(dqn_agent, torch.rand(4, 4))

        # Training completed, and the layers are simply unmeasured.
        assert all(entry is None for entry in dqn_agent.grama_scores[0])

    def test_a_layer_whose_gradient_reduces_to_nothing_stays_unmeasured(
        self,
        dqn_agent,
        monkeypatch,
    ):
        monkeypatch.setattr(regrama, "per_neuron_grad", lambda _grad_input: None)

        capture_snapshot(dqn_agent, torch.rand(4, 4))

        assert all(entry is None for entry in dqn_agent.grama_scores[0])

    def test_an_uncollectable_snapshot_is_reported_as_missing(self, dqn_agent):
        class Unreadable:
            def __iter__(self):
                msg = "snapshot went away"
                raise RuntimeError(msg)

        activation = regrama.target_activations(dqn_agent.actor)[0]
        capture = regrama.GraMaCapture(dqn_agent).register()
        capture._latest = [Unreadable()]

        capture.release()

        assert dqn_agent.grama_scores is None
        assert not activation._backward_hooks

    def test_an_agent_that_rejects_the_snapshot_still_releases_cleanly(
        self,
        dqn_agent,
    ):
        # Storing the snapshot must never break training either.
        class Rejecting:
            registry = dqn_agent.registry
            actor = dqn_agent.actor
            actor_target = dqn_agent.actor_target
            accelerator = None

            def __setattr__(self, name, value):
                msg = "read-only agent"
                raise AttributeError(msg)

        activation = regrama.target_activations(dqn_agent.actor)[0]

        with regrama.GraMaCapture(Rejecting()):
            pass

        assert not activation._backward_hooks

    def test_capture_never_breaks_on_an_agent_without_networks(self):
        # Registration is best-effort so a training run cannot be broken
        # by an agent that does not expose the expected surface.
        class Bare:
            grama_scores = None

        agent = Bare()

        with regrama.GraMaCapture(agent):
            pass

        assert agent.grama_scores == []


class TestScoredActivations:
    """Pair captured scores with the layers they were captured from."""

    def test_layers_are_paired_in_forward_order(self, dqn_agent):
        capture_snapshot(dqn_agent, torch.rand(4, 4))
        targets = regrama.target_activations(dqn_agent.actor)

        result = regrama.scored_activations(
            dqn_agent.actor,
            dqn_agent.grama_scores[0],
        )

        assert [module for module, _score in result] == targets

    def test_length_mismatch_skips_the_whole_network(self, dqn_agent):
        # Graceful degradation after an architecture mutation rebuilt the network:
        # mis-pairing scores with layers would corrupt unrelated neurons.
        result = regrama.scored_activations(dqn_agent.actor, [torch.ones(3)])

        assert result == []

    def test_unmeasured_layers_are_dropped(self, dqn_agent):
        capture_snapshot(dqn_agent, torch.rand(4, 4))
        scores = dqn_agent.grama_scores[0]
        scores[0] = None

        result = regrama.scored_activations(dqn_agent.actor, scores)

        assert len(result) == len(scores) - 1


class TestResetDormantNeurons:
    """Reset a whole evaluation network from its captured snapshot."""

    def test_a_fully_dormant_network_is_reset(self, dqn_agent):
        capture_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [
            None if entry is None else torch.zeros_like(entry)
            for entry in dqn_agent.grama_scores[0]
        ]
        measured = sum(entry.numel() for entry in scores if entry is not None)
        before = {k: v.clone() for k, v in dqn_agent.actor.state_dict().items()}

        report = regrama.reset_dormant_neurons(
            dqn_agent.actor,
            scores,
            0.01,
            make_rng(),
        )

        assert 0 < report.neurons_reset <= measured
        after = dqn_agent.actor.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)

    def test_a_healthy_network_is_left_untouched(self, dqn_agent):
        # Arrange: a uniform snapshot normalises to a score of 1.0 everywhere, so
        # no neuron is dormant at any threshold below one.
        capture_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [
            None if entry is None else torch.ones_like(entry)
            for entry in dqn_agent.grama_scores[0]
        ]
        before = {k: v.clone() for k, v in dqn_agent.actor.state_dict().items()}

        report = regrama.reset_dormant_neurons(
            dqn_agent.actor,
            scores,
            0.01,
            make_rng(),
        )

        assert report.neurons_reset == 0
        after = dqn_agent.actor.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_missing_snapshot_is_a_no_op(self, dqn_agent):
        report = regrama.reset_dormant_neurons(dqn_agent.actor, None, 0.01, make_rng())

        assert report.neurons_reset == 0

    def test_snapshot_of_the_wrong_width_skips_that_layer(self, dqn_agent):
        capture_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [torch.zeros(entry.numel() + 1) for entry in dqn_agent.grama_scores[0]]
        before = {k: v.clone() for k, v in dqn_agent.actor.state_dict().items()}

        report = regrama.reset_dormant_neurons(
            dqn_agent.actor,
            scores,
            0.01,
            make_rng(),
        )

        assert report.neurons_reset == 0
        after = dqn_agent.actor.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_head_output_layer_is_never_treated_as_a_producer(self, dqn_agent):
        capture_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [torch.zeros_like(entry) for entry in dqn_agent.grama_scores[0]]
        output_layer = dqn_agent.actor.head_net.model.value_linear_layer_output
        with torch.no_grad():
            output_layer.bias.fill_(3.0)

        regrama.reset_dormant_neurons(dqn_agent.actor, scores, 0.01, make_rng())

        # A producer's dormant neurons have their bias zeroed, so a surviving bias
        # is the observable proof the layer was never treated as one.
        assert torch.equal(
            output_layer.bias.detach(),
            torch.full_like(output_layer.bias, 3.0),
        )

    def test_head_output_layer_columns_are_still_rewritten(self, dqn_agent):
        # It consumes the last hidden activation, so when those neurons are
        # reset its columns must follow. That half is not excluded.
        capture_snapshot(dqn_agent, torch.rand(4, 4))
        scores = [torch.zeros_like(entry) for entry in dqn_agent.grama_scores[0]]
        output_layer = dqn_agent.actor.head_net.model.value_linear_layer_output
        before = output_layer.weight.detach().clone()

        regrama.reset_dormant_neurons(dqn_agent.actor, scores, 0.01, make_rng())

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
        capture_snapshot(agent, torch.rand(2, 3, 32, 32))
        scores = [
            None if entry is None else torch.zeros_like(entry)
            for entry in agent.grama_scores[0]
        ]

        report = regrama.reset_dormant_neurons(agent.actor, scores, 0.01, make_rng())

        assert report.neurons_reset > 0
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

        report = regrama.reset_dormant_neurons(agent.actor, scores, 0.01, make_rng())

        assert report.neurons_reset > 0
        assert all(
            torch.isfinite(value).all() for value in agent.actor.state_dict().values()
        )

    def test_recurrent_core_is_reported_as_out_of_scope(
        self,
        vector_space,
        discrete_space,
    ):
        # An LSTM's gate non-linearities are fused, so no per-neuron gradient is
        # captured for them and its hidden units own no contiguous weight rows.
        agent = PPO(
            vector_space,
            discrete_space,
            recurrent=True,
            net_config={"encoder_config": {"hidden_state_size": 8}},
            device="cpu",
        )

        report = regrama.reset_dormant_neurons(agent.actor, None, 0.01, make_rng())

        assert report.recurrent_seen is True

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
                regrama.weight_param(
                    regrama.resolve_producer_and_next(
                        activation,
                        agent.critic.encoder,
                        agent.critic.head_net,
                    ).producer,
                ).shape[0],
            )
            for activation in regrama.target_activations(agent.critic)
        ]
        before = {
            name: value.clone()
            for name, value in agent.critic.encoder.state_dict().items()
        }

        regrama.reset_dormant_neurons(agent.critic, scores, 0.01, make_rng())

        after = agent.critic.encoder.state_dict()
        assert all(torch.equal(before[name], after[name]) for name in before)


class TestSharedLatentBlocks:
    """Expose a shared latent's columns inside every borrowing head."""

    def test_dense_entry_exposes_its_leading_latent_columns(self):
        producer = nn.Linear(4, 8)

        result = regrama.shared_latent_blocks(producer, [nn.Linear(10, 5)])

        assert [tuple(target.weight.shape) for target in result] == [(5, 8)]

    def test_noisy_entry_also_exposes_its_noise_columns(self):
        producer = nn.Linear(4, 8)

        result = regrama.shared_latent_blocks(producer, [NoisyLinear(8, 5)])

        assert [target.is_noise_scale for target in result] == [False, True]

    @pytest.mark.parametrize(
        "make_entry",
        [lambda: nn.Linear(4, 5), lambda: nn.Conv2d(8, 4, 3)],
        ids=["narrower-than-the-latent", "not-a-dense-layer"],
    )
    def test_an_entry_that_cannot_hold_the_latent_is_skipped(self, make_entry):
        producer = nn.Linear(4, 8)

        result = regrama.shared_latent_blocks(producer, [make_entry()])

        assert result == []

    def test_a_convolutional_producer_has_no_latent_columns_to_share(self):
        # A shared latent crosses the encoder-head boundary as a flat vector,
        # so a conv producer is never the layer feeding a sharing head.
        producer = nn.Conv2d(3, 8, 3)

        result = regrama.shared_latent_blocks(producer, [nn.Linear(8, 5)])

        assert result == []


class TestSharedEncoderCompensation:
    """A latent shared by several networks is faded in every one of them."""

    def ppo(self, vector_space, discrete_space, *, share):
        return PPO(vector_space, discrete_space, share_encoders=share, device="cpu")

    def critic_head(self, network):
        return regrama.head_entry_layers(network.head_net)[0]

    def reset_actor_latent(self, agent, index=0):
        """Reset one latent unit of the policy, compensating shared consumers."""
        return regrama.reset_dormant_neurons(
            agent.actor,
            latent_marked_dormant(agent.actor, index),
            0.01,
            make_rng(),
            shared_latent_heads=regrama.shared_encoder_heads(agent, None, agent.actor),
        )

    def test_a_borrowed_encoder_is_recognised_as_pinned(
        self,
        vector_space,
        discrete_space,
    ):
        agent = self.ppo(vector_space, discrete_space, share=True)

        assert not regrama.encoder_is_pinned(agent.actor)
        assert regrama.encoder_is_pinned(agent.critic)

    def test_an_owned_encoder_is_not_pinned(self, vector_space, discrete_space):
        agent = self.ppo(vector_space, discrete_space, share=False)

        assert not regrama.encoder_is_pinned(agent.critic)

    def test_a_network_without_an_encoder_is_not_pinned(self):
        # Nothing borrowed, so nothing to compensate either.
        assert not regrama.encoder_is_pinned(nn.Sequential(nn.Linear(4, 2)))

    def test_the_critic_head_is_reported_as_a_shared_consumer(
        self,
        vector_space,
        discrete_space,
    ):
        agent = self.ppo(vector_space, discrete_space, share=True)

        result = regrama.shared_encoder_heads(agent, None, agent.actor)

        assert result == regrama.head_entry_layers(agent.critic.head_net)

    def test_unshared_encoders_report_no_shared_consumer(
        self,
        vector_space,
        discrete_space,
    ):
        # A network that owns its encoder compensates its own head, so pulling it
        # in here would fade the same column twice.
        agent = self.ppo(vector_space, discrete_space, share=False)

        assert regrama.shared_encoder_heads(agent, None, agent.actor) == []

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
            regrama.REGRAMA_OUT_SCALE * live,
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

        regrama.reset_dormant_neurons(
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
        # A ContinuousQNetwork head consumes cat([latent, actions]), so
        # the latent is only the leading block of a wider input.
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
        # The leading-block rewrite is correct only while ContinuousQNetwork feeds
        # its head torch.cat([latent, actions]).
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


class TestSetGraMaCapture:
    """Enable capture only when the mutation operator will actually use it."""

    class FakeMutation:
        def __init__(self, regrama_param_mut: bool) -> None:
            self.regrama_param_mut = regrama_param_mut

    def test_capture_is_enabled_for_a_regrama_mutation(self, dqn_agent):
        regrama.set_grama_capture([dqn_agent], self.FakeMutation(True))

        assert dqn_agent.capture_grama is True

    def test_capture_stays_off_for_a_plain_gaussian_mutation(self, dqn_agent):
        regrama.set_grama_capture([dqn_agent], self.FakeMutation(False))

        assert dqn_agent.capture_grama is False

    def test_capture_stays_off_without_a_mutation_operator(self, dqn_agent):
        regrama.set_grama_capture([dqn_agent], None)

        assert dqn_agent.capture_grama is False

    def test_compiled_agents_still_capture_but_are_warned(self, dqn_agent):
        # Capture works through torch.compile; what it costs is the compiled
        # graph, which fragments around the hooks. That is the user's call to
        # make, so the operator reports it and carries on.
        dqn_agent.torch_compiler = "default"

        with pytest.warns(UserWarning, match=r"torch\.compile"):
            regrama.set_grama_capture([dqn_agent], self.FakeMutation(True))

        assert dqn_agent.capture_grama is True

    def test_compiled_agents_are_warned_about_once(
        self,
        dqn_agent,
        vector_space,
        discrete_space,
    ):
        # One notice per population, not one per agent.
        second = DQN(
            vector_space,
            discrete_space,
            net_config=mlp_net_config(),
            device="cpu",
        )
        dqn_agent.torch_compiler = second.torch_compiler = "default"

        with pytest.warns(UserWarning, match=r"torch\.compile") as caught:
            regrama.set_grama_capture([dqn_agent, second], self.FakeMutation(True))

        assert len(caught) == 1
        assert all(agent.capture_grama for agent in (dqn_agent, second))

    def test_compiled_agents_are_silent_without_regrama(self, dqn_agent):
        # Nothing is hooked when ReGraMa is off, so there is no cost to report.
        dqn_agent.torch_compiler = "default"

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            regrama.set_grama_capture([dqn_agent], self.FakeMutation(False))

        assert dqn_agent.capture_grama is False


class TestEvalNetworks:
    """Enumerate the networks ReGraMa measures and rewrites."""

    def test_target_networks_are_excluded(self, dqn_agent):
        measured = [
            network for _network_id, network in regrama.eval_networks(dqn_agent)
        ]

        # The frozen copy must never be scored or reset.
        assert dqn_agent.actor in measured
        assert dqn_agent.actor_target not in measured

    def test_actor_and_critic_are_both_measured(self, vector_space, discrete_space):
        agent = PPO(vector_space, discrete_space, device="cpu")

        measured = [network for _network_id, network in regrama.eval_networks(agent)]

        assert agent.actor in measured
        assert agent.critic in measured

    def test_multi_agent_module_dicts_are_unrolled_per_sub_policy(
        self,
        ma_vector_space,
        ma_discrete_space,
    ):
        from agilerl.algorithms import IPPO

        agent = IPPO(
            ma_vector_space,
            ma_discrete_space,
            agent_ids=["agent_0", "agent_1", "agent_2"],
            device="cpu",
        )
        policy = getattr(agent, agent.registry.policy())

        result = regrama.eval_networks(agent)

        # One entry per sub-policy, each tagged with its own key, so a
        # captured snapshot is never routed to another sub-agent's network.
        for key, sub_network in policy.items():
            assert any(
                network_id == key and network is sub_network
                for network_id, network in result
            )


class TestPolicyNetworkIds:
    """Identify the policy networks whose latent other networks may borrow."""

    def test_the_policy_evaluation_network_is_reported(self, dqn_agent):
        assert regrama.policy_network_ids(dqn_agent) == {id(dqn_agent.actor)}

    def test_multi_agent_policies_report_every_sub_policy(
        self,
        ma_vector_space,
        ma_discrete_space,
    ):
        # Each sub-policy owns an encoder its own critic may borrow, so all of
        # them count as policy networks.
        from agilerl.algorithms import IPPO

        agent = IPPO(
            ma_vector_space,
            ma_discrete_space,
            agent_ids=["agent_0", "agent_1", "agent_2"],
            device="cpu",
        )
        policy = getattr(agent, agent.registry.policy())

        result = regrama.policy_network_ids(agent)

        assert result == {id(sub_network) for _key, sub_network in policy.items()}

    def test_an_agent_without_a_policy_group_reports_no_policy(self, dqn_agent):
        dqn_agent.registry.groups = []

        assert regrama.policy_network_ids(dqn_agent) == set()

    def test_a_policy_the_agent_does_not_carry_reports_no_policy(self, dqn_agent):
        dqn_agent.actor = None

        assert regrama.policy_network_ids(dqn_agent) == set()


class TestGraMaCaptureUnderAccelerator:
    """Capture on wrapped networks still lines up with the unwrapped ones."""

    def test_snapshot_survives_the_unwrap_before_selection(
        self,
        vector_space,
        discrete_space,
        encoder_mlp_config,
    ):
        from accelerate import Accelerator

        agent = DQN(
            vector_space,
            discrete_space,
            net_config=encoder_mlp_config,
            accelerator=Accelerator(cpu=True, device_placement=False),
            device="cpu",
        )
        agent.wrap_models()
        capture_snapshot(agent, torch.rand(4, 4))

        agent.unwrap_models()
        result = regrama.scored_activations(agent.actor, agent.grama_scores[0])

        assert len(result) == len(regrama.target_activations(agent.actor))

    def test_capture_measures_a_distributed_wrapped_network(
        self,
        gloo_process_group,
        vector_space,
        discrete_space,
        encoder_mlp_config,
    ):
        # On a multi-process launch accelerator.prepare returns a
        # DistributedDataParallel, which keeps the real network at .module.
        from accelerate import Accelerator

        agent = DQN(
            vector_space,
            discrete_space,
            net_config=encoder_mlp_config,
            accelerator=Accelerator(cpu=True, device_placement=False),
            device="cpu",
        )
        agent.actor = nn.parallel.DistributedDataParallel(agent.actor)
        inner = agent.accelerator.unwrap_model(agent.actor)

        agent.capture_grama = True
        agent.init_training_step()
        agent.actor(torch.rand(4, 4)).square().mean().backward()
        agent.finalize_training_step(1)

        # Every measured layer of the wrapped network scored a gradient,
        # so the reset acts instead of silently degrading to Gaussian noise.
        measured = agent.grama_scores[0]
        assert len(measured) == len(regrama.target_activations(inner))
        assert all(entry is not None for entry in measured)

    def test_distributed_wrapped_module_dicts_are_still_unrolled(
        self,
        gloo_process_group,
        ma_vector_space,
        ma_discrete_space,
    ):
        # A multi-agent policy is a ModuleDict, so wrap_models hands the whole
        # container to accelerator.prepare and the wrapper hides every sub-policy
        # behind .module.
        from accelerate import Accelerator

        from agilerl.algorithms import IPPO

        agent = IPPO(
            ma_vector_space,
            ma_discrete_space,
            agent_ids=["agent_0", "agent_1", "agent_2"],
            accelerator=Accelerator(cpu=True, device_placement=False),
            device="cpu",
        )
        policy_name = agent.registry.policy()
        policy = getattr(agent, policy_name)
        setattr(agent, policy_name, nn.parallel.DistributedDataParallel(policy))

        result = regrama.eval_networks(agent)

        # Still one entry per sub-policy, each tagged with its own key.
        for key, sub_network in policy.items():
            assert any(
                network_id == key and network is sub_network
                for network_id, network in result
            )
