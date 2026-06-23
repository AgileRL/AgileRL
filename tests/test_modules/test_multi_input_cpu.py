"""CPU-only EvolvableMultiInput helper tests (no GPU mark)."""

from dataclasses import asdict

import numpy as np
import pytest

from agilerl.modules.configs import CnnNetConfig, MlpNetConfig
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.typing import ModuleType
from tests.helper_functions import generate_dict_or_tuple_space


@pytest.fixture(scope="module")
def default_cnn_config():
    return asdict(
        CnnNetConfig(
            channel_size=[32],
            kernel_size=[3],
            stride_size=[1],
            output_activation="ReLU",
        ),
    )


@pytest.fixture(scope="module")
def default_mlp_config():
    return asdict(
        MlpNetConfig(
            hidden_size=[64],
            output_activation="ReLU",
        ),
    )


class TestEvolvableMultiInputHelpers:
    def test_reformat_mlp_config(self, device, default_mlp_config):
        evolvable = EvolvableMultiInput(
            observation_space=generate_dict_or_tuple_space(2, 3),
            num_outputs=10,
            mlp_config=default_mlp_config,
            device=device,
        )
        config = {"layer_norm": False, "activation": "Tanh"}
        reformatted = evolvable._reformat_mlp_config(config)
        assert reformatted["output_vanish"] is False
        assert reformatted["output_layernorm"] is False
        assert reformatted["output_activation"] == "Tanh"

    def test_get_inner_init_dict_invalid_default(self, device, default_mlp_config):
        evolvable = EvolvableMultiInput(
            observation_space=generate_dict_or_tuple_space(2, 3),
            num_outputs=10,
            mlp_config=default_mlp_config,
            device=device,
        )
        with pytest.raises(ValueError, match="Invalid default value provided"):
            evolvable.get_inner_init_dict("missing_key", ModuleType.RNN)

    def test_forward_tuple_numpy_observations(
        self,
        device,
        default_cnn_config,
        default_mlp_config,
    ):
        observation_space = generate_dict_or_tuple_space(2, 3, dict_space=False)
        evolvable = EvolvableMultiInput(
            observation_space=observation_space,
            num_outputs=10,
            cnn_config=default_cnn_config,
            mlp_config=default_mlp_config,
            device=device,
        )
        sample = observation_space.sample()
        input_tuple = tuple(np.expand_dims(comp, 0) for comp in sample)
        output = evolvable.forward(input_tuple)
        assert output.shape == (1, 10)
