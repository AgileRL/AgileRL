# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
import torch
from gymnasium import spaces

from agilerl.modules.cnn_lstm import EvolvableCnnLstm
from agilerl.modules.configs import CnnLstmNetConfig
from agilerl.networks.base import (
    EvolvableNetwork,
    assert_correct_cnn_lstm_net_config,
)
from agilerl.typing import BatchDimension
from agilerl.utils.evolvable_networks import (
    config_from_dict,
    get_default_encoder_config,
)
from tests.helper_functions import assert_state_dicts_equal


def _tiny_cnn_lstm(**overrides) -> EvolvableCnnLstm:
    identity = {
        "input_shape",
        "num_outputs",
        "device",
        "name",
        "random_seed",
        "net_config",
    }
    net_config = {
        "channel_size": [4],
        "kernel_size": [3],
        "stride_size": [2],
        "hidden_state_size": 8,
        "num_layers": 1,
    }
    kwargs = {
        "input_shape": [3, 8, 8],
        "num_outputs": 4,
        "device": "cpu",
        "name": "encoder",
    }
    for key, value in overrides.items():
        if key in identity:
            kwargs[key] = value
        else:
            net_config[key] = value
    kwargs.setdefault("net_config", net_config)
    return EvolvableCnnLstm(**kwargs)


class TestEvolvableCnnLstmForward:
    def test_forward_updates_hidden_state(self):
        encoder = _tiny_cnn_lstm()
        batch = 2
        x = torch.randn(batch, 3, 8, 8)
        hidden = {
            "encoder_h": torch.zeros(1, batch, 8),
            "encoder_c": torch.zeros(1, batch, 8),
        }

        out, next_hidden = encoder(x, hidden_state=hidden)

        assert out.shape == (batch, 4)
        assert next_hidden["encoder_h"].shape == (1, batch, 8)
        assert next_hidden["encoder_c"].shape == (1, batch, 8)
        assert not torch.equal(next_hidden["encoder_h"], hidden["encoder_h"])

    def test_forward_bptt_flatten_matches_hidden_batch(self):
        encoder = _tiny_cnn_lstm()
        batch_seq = 2
        seq_len = 3
        x = torch.randn(batch_seq * seq_len, 3, 8, 8)
        hidden = {
            "encoder_h": torch.zeros(1, batch_seq, 8),
            "encoder_c": torch.zeros(1, batch_seq, 8),
        }

        out, next_hidden = encoder(x, hidden_state=hidden)

        assert out.shape == (batch_seq * seq_len, 4)
        assert next_hidden["encoder_h"].shape == (1, batch_seq, 8)
        assert next_hidden["encoder_c"].shape == (1, batch_seq, 8)

    def test_forward_5d_matching_batch_keeps_time_axis(self):
        encoder = _tiny_cnn_lstm()
        batch = 2
        seq_len = 3
        x = torch.randn(batch, seq_len, 3, 8, 8)
        hidden = {
            "encoder_h": torch.zeros(1, batch, 8),
            "encoder_c": torch.zeros(1, batch, 8),
        }

        out, next_hidden = encoder(x, hidden_state=hidden)

        assert out.shape == (batch, seq_len, 4)
        assert next_hidden["encoder_h"].shape == (1, batch, 8)
        assert next_hidden["encoder_c"].shape == (1, batch, 8)

    def test_forward_requires_hidden_state(self):
        encoder = _tiny_cnn_lstm(input_shape=[3, 4, 4], num_outputs=2)
        with pytest.raises(ValueError, match="Hidden state is required"):
            encoder(torch.randn(1, 3, 4, 4), hidden_state=None)

    def test_forward_numpy_input_converts_to_tensor(self):
        encoder = _tiny_cnn_lstm()
        x = torch.randn(2, 3, 8, 8).numpy()
        hidden = {
            "encoder_h": torch.zeros(1, 2, 8),
            "encoder_c": torch.zeros(1, 2, 8),
        }

        out, _ = encoder(x, hidden_state=hidden)

        assert out.shape == (2, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_forward_uses_parameter_device_after_cuda_to(self):
        encoder = _tiny_cnn_lstm(device="cpu")
        encoder = encoder.cuda()
        x = torch.randn(2, 3, 8, 8, device="cuda")
        hidden = {
            "encoder_h": torch.zeros(1, 2, 8, device="cpu"),
            "encoder_c": torch.zeros(1, 2, 8, device="cpu"),
        }

        out, next_hidden = encoder(x, hidden_state=hidden)

        assert out.device.type == "cuda"
        assert next_hidden["encoder_h"].device.type == "cuda"

    def test_forward_rejects_invalid_rank(self):
        encoder = _tiny_cnn_lstm()
        hidden = {
            "encoder_h": torch.zeros(1, 1, 8),
            "encoder_c": torch.zeros(1, 1, 8),
        }
        with pytest.raises(ValueError, match="expected 4D or 5D image tensor"):
            encoder(torch.randn(3, 8), hidden_state=hidden)

    def test_forward_rejects_channel_mismatch(self):
        encoder = _tiny_cnn_lstm()
        hidden = {
            "encoder_h": torch.zeros(1, 2, 8),
            "encoder_c": torch.zeros(1, 2, 8),
        }
        with pytest.raises(RuntimeError, match="channel"):
            encoder(torch.randn(2, 5, 8, 8), hidden_state=hidden)


class TestEvolvableCnnLstmInit:
    def test_rejects_invalid_input_shape_rank(self):
        with pytest.raises(AssertionError, match="input_shape should be"):
            _tiny_cnn_lstm(input_shape=[8])

    def test_rejects_invalid_block_type(self):
        with pytest.raises(ValueError, match="Invalid block type"):
            _tiny_cnn_lstm(block_type="InvalidBlock")

    def test_rejects_kernel_size_length_mismatch(self):
        with pytest.raises(AssertionError, match="kernel size list must match"):
            _tiny_cnn_lstm(kernel_size=[3, 3])

    def test_rejects_stride_size_length_mismatch(self):
        with pytest.raises(AssertionError, match="stride size list must match"):
            _tiny_cnn_lstm(stride_size=[2, 1])

    def test_conv3d_requires_sample_input(self):
        with pytest.raises(AssertionError, match="Sample input"):
            _tiny_cnn_lstm(block_type="Conv3d")

    def test_conv3d_uses_sample_input(self):
        sample = torch.zeros(1, 3, 4, 8, 8)

        encoder = _tiny_cnn_lstm(
            block_type="Conv3d",
            sample_input=sample,
            stride_size=[1],
        )

        assert encoder.block_type == "Conv3d"
        assert torch.equal(encoder.sample_input, sample)

    def test_uses_provided_sample_input(self):
        sample = torch.zeros(1, 3, 8, 8)
        encoder = _tiny_cnn_lstm(sample_input=sample)
        assert torch.equal(encoder.sample_input, sample)

    def test_init_weights_gaussian_keeps_output_shape(self):
        encoder = _tiny_cnn_lstm()
        encoder.init_weights_gaussian(std_coeff=4, output_coeff=2)
        hidden = {
            "encoder_h": torch.zeros(1, 1, 8),
            "encoder_c": torch.zeros(1, 1, 8),
        }

        out, _ = encoder(torch.randn(1, 3, 8, 8), hidden_state=hidden)

        assert out.shape == (1, 4)

    def test_change_activation_rebuilds_network(self):
        encoder = _tiny_cnn_lstm()
        encoder.change_activation("Tanh")
        assert encoder.activation == "Tanh"
        encoder.change_activation("ReLU", output=True)
        assert encoder.output_activation == "ReLU"

    def test_accepts_cnn_lstm_net_config_object(self):
        cfg = CnnLstmNetConfig(
            channel_size=[4],
            kernel_size=[3],
            stride_size=[2],
            hidden_state_size=8,
        )
        encoder = EvolvableCnnLstm(
            input_shape=[3, 8, 8],
            num_outputs=4,
            net_config=cfg,
        )
        assert encoder.hidden_state_size == 8
        assert encoder.channel_size == [4]


class TestEvolvableCnnLstmClone:
    def test_clone_instance(self):
        encoder = _tiny_cnn_lstm(name="enc")
        clone = encoder.clone()

        assert isinstance(clone, EvolvableCnnLstm)
        assert clone.init_dict["name"] == "enc"
        clone_cfg = {
            key: value
            for key, value in clone.init_dict["net_config"].items()
            if key != "sample_input"
        }
        encoder_cfg = {
            key: value
            for key, value in encoder.init_dict["net_config"].items()
            if key != "sample_input"
        }
        assert clone_cfg == encoder_cfg
        assert torch.equal(
            clone.init_dict["net_config"]["sample_input"],
            encoder.init_dict["net_config"]["sample_input"],
        )
        assert_state_dicts_equal(clone.state_dict(), encoder.state_dict())


class TestEvolvableCnnLstmMutations:
    def test_architecture_mutations_disabled(self):
        encoder = _tiny_cnn_lstm(input_shape=[3, 4, 4], num_outputs=2)
        assert encoder.mutation_methods == []


class TestEvolvableCnnLstmNetConfig:
    def test_net_config_strips_constructor_fields(self):
        encoder = _tiny_cnn_lstm(
            input_shape=[3, 4, 4],
            num_outputs=2,
            name="enc",
        )

        config = encoder.net_config

        assert "input_shape" not in config
        assert "num_outputs" not in config
        assert "device" not in config
        assert "name" not in config
        assert config["hidden_state_size"] == 8


class TestBuildEncoderImageRecurrent:
    class RecurrentImageNet(EvolvableNetwork):
        def __init__(self, observation_space: spaces.Space, encoder_config=None):
            super().__init__(
                observation_space=observation_space,
                encoder_config=encoder_config,
                recurrent=True,
                latent_dim=16,
                device="cpu",
            )
            self.build_network_head(net_config={"hidden_size": [8]})

        def build_network_head(self, net_config=None):
            self.head_net = self.create_mlp(
                num_inputs=self.latent_dim,
                num_outputs=1,
                name="head",
                net_config=net_config,
            )

        def recreate_network(self) -> None:
            pass

        def forward(self, x, hidden_state=None):
            features, hidden_state = self.encoder(x, hidden_state)
            return self.head_net(features), hidden_state

    def test_build_encoder_selects_cnn_lstm(self):
        space = spaces.Box(0, 1, shape=(3, 10, 10), dtype="float32")
        config = {
            "channel_size": [4],
            "kernel_size": [3],
            "stride_size": [2],
            "hidden_state_size": 8,
            "num_layers": 1,
            "output_activation": "ReLU",
        }
        assert_correct_cnn_lstm_net_config(config)
        net = self.RecurrentImageNet(space, encoder_config=config)
        assert isinstance(net.encoder, EvolvableCnnLstm)
        assert net.encoder.hidden_state_architecture["h"] == (
            1,
            BatchDimension,
            8,
        )

    def test_default_encoder_config_builds_cnn_lstm(self):
        space = spaces.Box(0, 1, shape=(3, 10, 10), dtype="float32")
        net = self.RecurrentImageNet(space, encoder_config=None)
        assert isinstance(net.encoder, EvolvableCnnLstm)

    def test_recreate_encoder_uses_trimmed_net_config(self):
        space = spaces.Box(0, 1, shape=(3, 8, 8), dtype="float32")
        net = self.RecurrentImageNet(
            space,
            encoder_config=get_default_encoder_config(space, recurrent=True),
        )
        net.latent_dim = 24
        net.recreate_encoder()
        assert isinstance(net.encoder, EvolvableCnnLstm)
        assert net.encoder.num_outputs == 24


class TestDefaultEncoderConfigCnnLstm:
    def test_image_recurrent_merges_cnn_and_lstm_keys(self):
        space = spaces.Box(0, 1, shape=(3, 8, 8), dtype="float32")
        config = get_default_encoder_config(space, recurrent=True)
        assert "channel_size" in config
        assert "hidden_state_size" in config
        assert_correct_cnn_lstm_net_config(config)


class TestConfigFromDictCnnLstm:
    def test_channel_and_hidden_state_selects_cnn_lstm_config(self):
        cfg = config_from_dict(
            {
                "channel_size": [8],
                "kernel_size": [3],
                "stride_size": [1],
                "hidden_state_size": 16,
            }
        )
        assert isinstance(cfg, CnnLstmNetConfig)


class TestCnnLstmDefaultAlignment:
    def test_omitted_net_config_fields_match_cnn_lstm_net_config_defaults(self):
        default_fields = (
            "activation",
            "output_activation",
            "block_type",
            "num_layers",
            "min_hidden_layers",
            "max_hidden_layers",
            "min_channel_size",
            "max_channel_size",
            "layer_norm",
            "init_layers",
            "min_hidden_state_size",
            "max_hidden_state_size",
            "min_layers",
            "max_layers",
            "dropout",
        )
        config_defaults = {
            field.name: field.default
            for field in dataclasses.fields(CnnLstmNetConfig)
            if field.name in default_fields
        }
        encoder = EvolvableCnnLstm(
            input_shape=[3, 8, 8],
            num_outputs=4,
            net_config={
                "channel_size": [4],
                "kernel_size": [3],
                "stride_size": [2],
                "hidden_state_size": 8,
            },
        )
        for name, expected in config_defaults.items():
            assert getattr(encoder, name) == expected, name
