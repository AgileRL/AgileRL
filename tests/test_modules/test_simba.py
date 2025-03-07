import pytest
import torch
from gymnasium import spaces

from agilerl.modules.simba import EvolvableSimBa
from agilerl.networks import (
    ContinuousQNetwork,
    DeterministicActor,
    QNetwork,
    StochasticActor,
    ValueNetwork,
)


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


def test_add_node():
    model = EvolvableSimBa(
        num_inputs=10, num_outputs=2, hidden_size=64, num_blocks=3, device="cpu"
    )
    initial_hidden_size = model.hidden_size
    model.add_node(numb_new_nodes=16)
    assert model.hidden_size == initial_hidden_size + 16


def test_remove_node():
    model = EvolvableSimBa(
        num_inputs=10, num_outputs=2, hidden_size=64, num_blocks=3, device="cpu"
    )
    initial_hidden_size = model.hidden_size
    model.remove_node(numb_new_nodes=16)
    assert model.hidden_size == initial_hidden_size - 16


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
    assert str(clone.state_dict()) == str(model.state_dict())


@pytest.fixture
def observation_space():
    return spaces.Box(low=-1.0, high=1.0, shape=(10,))


@pytest.fixture
def action_space_discrete():
    return spaces.Discrete(4)


@pytest.fixture
def action_space_continuous():
    return spaces.Box(low=-1.0, high=1.0, shape=(2,))


@pytest.fixture
def encoder_simba_config():
    return {
        "hidden_size": 64,
        "num_blocks": 3,
    }


def test_deterministic_actor_simba(
    observation_space, action_space_continuous, encoder_simba_config
):
    model = DeterministicActor(
        observation_space=observation_space,
        action_space=action_space_continuous,
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)


def test_stochastic_actor_simba(
    observation_space, action_space_discrete, encoder_simba_config
):
    model = StochasticActor(
        observation_space=observation_space,
        action_space=action_space_discrete,
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)


def test_q_network_simba(
    observation_space, action_space_discrete, encoder_simba_config
):
    model = QNetwork(
        observation_space=observation_space,
        action_space=action_space_discrete,
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)


def test_continuous_q_network_simba(
    observation_space, action_space_continuous, encoder_simba_config
):
    model = ContinuousQNetwork(
        observation_space=observation_space,
        action_space=action_space_continuous,
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)


def test_value_network_simba(observation_space, encoder_simba_config):
    model = ValueNetwork(
        observation_space=observation_space,
        encoder_config=encoder_simba_config,
        simba=True,
        device="cpu",
    )
    assert isinstance(model.encoder, EvolvableSimBa)
