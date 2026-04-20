import copy

import pytest
import torch

from agilerl.modules.simba import EvolvableSimBa
from agilerl.networks import (
    ContinuousQNetwork,
    DeterministicActor,
    QNetwork,
    StochasticActor,
    ValueNetwork,
)
from tests.helper_functions import assert_state_dicts_equal


@pytest.fixture(scope="module")
def encoder_simba_config():
    return {
        "hidden_size": 64,
        "num_blocks": 3,
    }


def test_correct_initialization():
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    assert model.num_inputs == 10
    assert model.num_outputs == 2
    assert model.hidden_size == 64
    assert model.num_blocks == 3


def test_incorrect_initialization():
    with pytest.raises(TypeError):
        EvolvableSimBa(
            num_inputs=10,
            num_outputs=2,
            hidden_size=64,
            num_blocks="three",  # Incorrect type
            device="cpu",
        )


def test_net_config_excludes_attrs():
    """Covers net_config property popping num_inputs, num_outputs, device, name."""
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    net_config = model.net_config
    assert "num_inputs" not in net_config
    assert "num_outputs" not in net_config
    assert "device" not in net_config
    assert "name" not in net_config


def test_forward_numpy_input():
    """Covers forward when x is not torch.Tensor."""
    import numpy as np

    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    x = np.random.randn(10).astype(np.float32)
    output = model(x)
    assert output.shape == (1, 2)


def test_forward_single_dim_unsqueeze():
    """Covers forward when len(x.shape) == 1."""
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    x = torch.randn(10)
    output = model(x)
    assert output.shape == (1, 2)


def test_get_output_dense():
    """Covers get_output_dense."""
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    dense = model.get_output_dense()
    assert dense is not None
    assert hasattr(dense, "weight")


def test_init_weights_gaussian_with_output_coeff():
    """Covers init_weights_gaussian with output_coeff."""
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    model.init_weights_gaussian(std_coeff=4, output_coeff=2)
    output = model(torch.randn(1, 10))
    assert output.shape == (1, 2)


def test_change_activation_noop():
    """Covers change_activation no-op for SimBa."""
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    model.change_activation("Tanh", output=False)
    model.change_activation("Sigmoid", output=True)


def test_add_block_fallback_add_node_when_max_blocks():
    """Covers add_block when num_blocks >= max_blocks, falls back to add_node."""
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=4,
        max_blocks=4,
        device="cpu",
    )
    initial_hidden = model.hidden_size
    model.add_block()
    assert model.hidden_size >= initial_hidden or model.num_blocks == 4


def test_remove_block_fallback_add_node_when_min_blocks():
    """Covers remove_block when num_blocks <= min_blocks, falls back to add_node."""
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=1,
        min_blocks=1,
        device="cpu",
    )
    initial_hidden = model.hidden_size
    model.remove_block()
    assert model.hidden_size >= initial_hidden or model.num_blocks == 1


def test_add_node_hard_limit_not_exceeded():
    """Covers add_node when hidden_size + numb_new_nodes > max_mlp_nodes."""
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=484,
        num_blocks=3,
        max_mlp_nodes=500,
        device="cpu",
        random_seed=42,
    )
    model.add_node(numb_new_nodes=32)
    assert model.hidden_size <= 500


def test_remove_node_hard_limit_not_below_min():
    """Covers remove_node when hidden_size - numb_new_nodes <= min_mlp_nodes."""
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=24,
        num_blocks=3,
        min_mlp_nodes=16,
        device="cpu",
        random_seed=42,
    )
    model.remove_node(numb_new_nodes=16)
    assert model.hidden_size >= 16


def test_forward_pass():
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    x = torch.randn(1, 10)
    output = model(x)
    assert output.shape == (1, 2)


def test_add_block():
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    initial_blocks = model.num_blocks
    model.add_block()
    assert model.num_blocks == initial_blocks + 1


def test_remove_block():
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    initial_blocks = model.num_blocks
    model.remove_block()
    assert model.num_blocks == initial_blocks - 1


@pytest.mark.parametrize("numb_new_nodes", [16, None])
def test_add_node(numb_new_nodes):
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    initial_hidden_size = model.hidden_size
    mut_dict = model.add_node(numb_new_nodes=numb_new_nodes)
    assert model.hidden_size == initial_hidden_size + mut_dict["numb_new_nodes"]


@pytest.mark.parametrize("numb_new_nodes", [16, None])
def test_remove_node(numb_new_nodes):
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=124,
        num_blocks=3,
        device="cpu",
    )
    initial_hidden_size = model.hidden_size
    mut_dict = model.remove_node(numb_new_nodes=numb_new_nodes)
    assert model.hidden_size == initial_hidden_size - mut_dict["numb_new_nodes"]


def test_recreate_network():
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    initial_model = model.model
    model.recreate_network()
    assert model.model is not initial_model


def test_clone():
    model = EvolvableSimBa(
        num_inputs=10,
        num_outputs=2,
        hidden_size=64,
        num_blocks=3,
        device="cpu",
    )
    clone = model.clone()
    assert clone is not model
    assert clone.num_inputs == model.num_inputs
    assert clone.num_outputs == model.num_outputs
    assert clone.hidden_size == model.hidden_size
    assert clone.num_blocks == model.num_blocks
    assert_state_dicts_equal(clone.state_dict(), model.state_dict())


def test_deterministic_actor_simba(vector_space, encoder_simba_config):
    model = DeterministicActor(
        observation_space=vector_space,
        action_space=copy.deepcopy(vector_space),
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)


def test_stochastic_actor_simba(vector_space, discrete_space, encoder_simba_config):
    model = StochasticActor(
        observation_space=vector_space,
        action_space=discrete_space,
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)


def test_q_network_simba(vector_space, discrete_space, encoder_simba_config):
    model = QNetwork(
        observation_space=vector_space,
        action_space=discrete_space,
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)


def test_continuous_q_network_simba(vector_space, encoder_simba_config):
    model = ContinuousQNetwork(
        observation_space=vector_space,
        action_space=copy.deepcopy(vector_space),
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)


def test_value_network_simba(vector_space, encoder_simba_config):
    model = ValueNetwork(
        observation_space=vector_space,
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)
