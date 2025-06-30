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
        num_inputs=10, num_outputs=2, hidden_size=64, num_blocks=3, device="cpu"
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


def test_forward_pass():
    model = EvolvableSimBa(
        num_inputs=10, num_outputs=2, hidden_size=64, num_blocks=3, device="cpu"
    )
    x = torch.randn(1, 10)
    output = model(x)
    assert output.shape == (1, 2)


def test_add_block():
    model = EvolvableSimBa(
        num_inputs=10, num_outputs=2, hidden_size=64, num_blocks=3, device="cpu"
    )
    initial_blocks = model.num_blocks
    model.add_block()
    assert model.num_blocks == initial_blocks + 1


def test_remove_block():
    model = EvolvableSimBa(
        num_inputs=10, num_outputs=2, hidden_size=64, num_blocks=3, device="cpu"
    )
    initial_blocks = model.num_blocks
    model.remove_block()
    assert model.num_blocks == initial_blocks - 1


@pytest.mark.parametrize("numb_new_nodes", [16, None])
def test_add_node(numb_new_nodes):
    model = EvolvableSimBa(
        num_inputs=10, num_outputs=2, hidden_size=64, num_blocks=3, device="cpu"
    )
    initial_hidden_size = model.hidden_size
    mut_dict = model.add_node(numb_new_nodes=numb_new_nodes)
    assert model.hidden_size == initial_hidden_size + mut_dict["numb_new_nodes"]


@pytest.mark.parametrize("numb_new_nodes", [16, None])
def test_remove_node(numb_new_nodes):
    model = EvolvableSimBa(
        num_inputs=10, num_outputs=2, hidden_size=124, num_blocks=3, device="cpu"
    )
    initial_hidden_size = model.hidden_size
    mut_dict = model.remove_node(numb_new_nodes=numb_new_nodes)
    assert model.hidden_size == initial_hidden_size - mut_dict["numb_new_nodes"]


def test_recreate_network():
    model = EvolvableSimBa(
        num_inputs=10, num_outputs=2, hidden_size=64, num_blocks=3, device="cpu"
    )
    initial_model = model.model
    model.recreate_network()
    assert model.model is not initial_model


def test_clone():
    model = EvolvableSimBa(
        num_inputs=10, num_outputs=2, hidden_size=64, num_blocks=3, device="cpu"
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
