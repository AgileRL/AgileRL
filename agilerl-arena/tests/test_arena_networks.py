"""Tests for ``agilerl.arena.models.networks`` validation helpers and specs."""

from __future__ import annotations

import pytest
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.algorithms.llmppo import LLMPPOSpec
from agilerl.arena.models.algorithms.llmreinforce import LLMREINFORCESpec
from agilerl.arena.models.algorithms.rainbow_dqn import RainbowDQNSpec
from agilerl.arena.models.networks import (
    CnnSpec,
    LoraConfigDict,
    LstmSpec,
    MlpSpec,
    MultiInputSpec,
    QNetworkSpec,
    SimbaSpec,
    min_max_validator,
)


class TestMinMaxValidator:
    def test_raises_when_min_exceeds_max(self) -> None:
        model = MlpSpec(hidden_size=[64], min_hidden_layers=1, max_hidden_layers=6)
        model.min_mlp_nodes = 999
        model.max_mlp_nodes = 8
        with pytest.raises(ValueError, match="must be less than or equal to"):
            min_max_validator("min_mlp_nodes", "max_mlp_nodes")(model)


class TestMlpSpec:
    def test_model_dump_removes_none_output_activation(self) -> None:
        spec = MlpSpec(hidden_size=[64])
        dumped = spec.model_dump()
        assert "output_activation" not in dumped

    def test_model_dump_keeps_output_activation(self) -> None:
        spec = MlpSpec(hidden_size=[64], output_activation="Tanh")
        assert spec.model_dump()["output_activation"] == "Tanh"

    def test_hidden_size_below_min_hidden_layers(self) -> None:
        with pytest.raises(
            ValueError,
            match="hidden_size must be greater than or equal to min_hidden_layers",
        ):
            MlpSpec(hidden_size=[64], min_hidden_layers=3)

    def test_hidden_size_above_max_hidden_layers(self) -> None:
        with pytest.raises(
            ValueError,
            match="hidden_size must be less than or equal to max_hidden_layers",
        ):
            MlpSpec(hidden_size=[32] * 8, max_hidden_layers=3)

    def test_node_below_min_mlp_nodes(self) -> None:
        with pytest.raises(
            ValueError,
            match="hidden_size elements must be greater than or equal to min_mlp_nodes",
        ):
            MlpSpec(hidden_size=[2], min_mlp_nodes=8)

    def test_node_above_max_mlp_nodes(self) -> None:
        with pytest.raises(
            ValueError,
            match="hidden_size elements must be less than or equal to max_mlp_nodes",
        ):
            MlpSpec(hidden_size=[512], max_mlp_nodes=256)


class TestSimbaSpec:
    def test_valid_construction(self) -> None:
        spec = SimbaSpec(hidden_size=64, num_blocks=2)
        assert spec.hidden_size == 64

    def test_hidden_size_below_min_mlp_nodes(self) -> None:
        with pytest.raises(
            ValueError,
            match="hidden_size must be greater than or equal to min_mlp_nodes",
        ):
            SimbaSpec(hidden_size=4, num_blocks=1, min_mlp_nodes=8)

    def test_hidden_size_above_max_mlp_nodes(self) -> None:
        with pytest.raises(
            ValueError, match="hidden_size must be less than or equal to max_mlp_nodes"
        ):
            SimbaSpec(hidden_size=512, num_blocks=1, max_mlp_nodes=256)


class TestCnnSpec:
    def test_valid_construction(self) -> None:
        spec = CnnSpec(channel_size=[16], kernel_size=[3], stride_size=[1])
        assert spec.channel_size == [16]

    def test_channel_size_outside_layer_range(self) -> None:
        with pytest.raises(ValueError, match="hidden_layers must be between"):
            CnnSpec(
                channel_size=[16] * 8,
                kernel_size=[3] * 8,
                stride_size=[1] * 8,
                max_hidden_layers=3,
            )

    def test_channel_below_min(self) -> None:
        with pytest.raises(
            ValueError,
            match="channel_size must be greater than or equal to min_channel_size",
        ):
            CnnSpec(
                channel_size=[2],
                kernel_size=[3],
                stride_size=[1],
                min_channel_size=8,
            )

    def test_channel_above_max(self) -> None:
        with pytest.raises(
            ValueError,
            match="channel_size must be less than or equal to max_channel_size",
        ):
            CnnSpec(
                channel_size=[512],
                kernel_size=[3],
                stride_size=[1],
                max_channel_size=256,
            )

    def test_mismatched_sizes(self) -> None:
        with pytest.raises(ValueError, match="must have the same length"):
            CnnSpec(channel_size=[16, 32], kernel_size=[3], stride_size=[1, 1])


class TestMultiInputSpec:
    def test_valid_construction(self) -> None:
        spec = MultiInputSpec(latent_dim=32)
        assert spec.latent_dim == 32

    def test_latent_dim_exceeds_max(self) -> None:
        with pytest.raises(ValueError, match="must be less than or equal to"):
            MultiInputSpec(latent_dim=200, max_latent_dim=128)


class TestLstmSpec:
    def test_valid_construction(self) -> None:
        spec = LstmSpec(hidden_state_size=64)
        assert spec.hidden_state_size == 64

    def test_hidden_state_size_exceeds_max(self) -> None:
        with pytest.raises(ValueError, match="must be less than or equal to"):
            LstmSpec(hidden_state_size=512, max_hidden_state_size=256)

    def test_num_layers_exceeds_max(self) -> None:
        with pytest.raises(ValueError, match="must be less than or equal to"):
            LstmSpec(hidden_state_size=64, num_layers=10, max_layers=6)


class TestNetworkSpec:
    def test_detects_simba_encoder(self) -> None:
        spec = QNetworkSpec(
            encoder_config=SimbaSpec(hidden_size=64, num_blocks=1),
            head_config=MlpSpec(hidden_size=[64]),
        )
        assert spec.simba is True


class TestLoraConfigDict:
    def test_serializes_set_target_modules(self) -> None:
        config = LoraConfigDict(target_modules={"q_proj", "v_proj"})
        dumped = config.model_dump(mode="json")
        assert dumped["target_modules"] == ["q_proj", "v_proj"]


class TestLLMAlgorithmValidators:
    @pytest.mark.parametrize(
        "spec_cls, kwargs",
        [
            (GRPOSpec, {"group_size": 2, "pretrained_model_name_or_path": "gpt2"}),
            (LLMPPOSpec, {"pretrained_model_name_or_path": "gpt2"}),
            (LLMREINFORCESpec, {"pretrained_model_name_or_path": "gpt2"}),
        ],
    )
    def test_use_vllm_requires_config(self, spec_cls, kwargs) -> None:
        with pytest.raises(ValueError, match="VLLM config is not set"):
            spec_cls(use_vllm=True, **kwargs)


class TestRainbowDQNSpec:
    def test_rejects_invalid_value_range(self) -> None:
        with pytest.raises(ValueError, match="v_min must be less than v_max"):
            RainbowDQNSpec(v_min=10.0, v_max=10.0)
