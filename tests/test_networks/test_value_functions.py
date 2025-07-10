from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces

from agilerl.modules import (
    EvolvableCNN,
    EvolvableLSTM,
    EvolvableMLP,
    EvolvableModule,
    EvolvableMultiInput,
    EvolvableSimBa,
)
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.networks.value_networks import ValueNetwork
from tests.helper_functions import (
    assert_close_dict,
    assert_not_equal_state_dict,
    assert_state_dicts_equal,
    check_equal_params_ind,
)


@pytest.fixture(scope="module")
def head_config():
    return asdict(MlpNetConfig(hidden_size=[64, 64]))


@pytest.mark.parametrize(
    "observation_space, encoder_type",
    [
        ("dict_space", "multi_input"),
        ("discrete_space", "mlp"),
        ("vector_space", "mlp"),
        ("image_space", "cnn"),
    ],
)
def test_value_function_initialization(observation_space, encoder_type, request):
    observation_space = request.getfixturevalue(observation_space)

    network = ValueNetwork(observation_space)

    assert network.observation_space == observation_space

    if encoder_type == "multi_input":
        assert isinstance(network.encoder, EvolvableMultiInput)
    elif encoder_type == "mlp":
        assert isinstance(network.encoder, EvolvableMLP)
    elif encoder_type == "cnn":
        assert isinstance(network.encoder, EvolvableCNN)

    evolvable_modules = network.modules()
    assert "encoder" in evolvable_modules
    assert "head_net" in evolvable_modules


def test_value_function_initialization_recurrent(vector_space):
    observation_space = vector_space
    network = ValueNetwork(observation_space, recurrent=True)

    assert network.observation_space == observation_space
    assert isinstance(network.encoder, EvolvableLSTM)

    evolvable_modules = network.modules()
    assert "encoder" in evolvable_modules
    assert "head_net" in evolvable_modules


def test_value_function_initialization_simba(vector_space):
    network = ValueNetwork(vector_space, simba=True)

    assert network.observation_space == vector_space
    assert isinstance(network.encoder, EvolvableSimBa)

    evolvable_modules = network.modules()
    assert "encoder" in evolvable_modules
    assert "head_net" in evolvable_modules


@pytest.mark.parametrize(
    "observation_space", ["dict_space", "discrete_space", "vector_space", "image_space"]
)
def test_value_function_mutation_methods(
    observation_space, head_config, dummy_rng, request
):
    observation_space = request.getfixturevalue(observation_space)
    network = ValueNetwork(observation_space, head_config=head_config)
    network.rng = dummy_rng

    for method in network.mutation_methods:
        new_network = network.clone()
        getattr(new_network, method)()

        if "." in method:
            net_name = method.split(".")[0]
            mutated_module: EvolvableModule = getattr(new_network, net_name)
            exec_method = new_network.last_mutation_attr.split(".")[-1]

            if isinstance(observation_space, (spaces.Tuple, spaces.Dict)):
                mutated_attr = mutated_module.last_mutation_attr.split(".")[-1]
            else:
                mutated_attr = mutated_module.last_mutation_attr

            assert mutated_attr == exec_method

        if new_network.last_mutation_attr is not None:
            # Check that architecture has changed
            assert_not_equal_state_dict(network.state_dict(), new_network.state_dict())

            # Checks that parameters that are not mutated are the same
            check_equal_params_ind(network, new_network)
        else:
            raise ValueError(
                f"Last mutation attribute is None. Expected {method} to be applied."
            )


@pytest.mark.parametrize(
    "observation_space", ["dict_space", "discrete_space", "vector_space", "image_space"]
)
def test_value_function_forward(observation_space: spaces.Space, request):
    observation_space = request.getfixturevalue(observation_space)

    network = ValueNetwork(observation_space)

    x_np = observation_space.sample()

    if isinstance(observation_space, spaces.Discrete):
        x_np = (
            F.one_hot(torch.tensor(x_np), num_classes=observation_space.n)
            .float()
            .numpy()
        )

    with torch.no_grad():
        out = network(x_np)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 1])

    if isinstance(observation_space, spaces.Dict):
        x = {key: torch.tensor(value) for key, value in x_np.items()}
    elif isinstance(observation_space, spaces.Tuple):
        x = tuple(torch.tensor(value) for value in x_np)
    else:
        x = torch.tensor(x_np)

    with torch.no_grad():
        out = network(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 1])


@pytest.mark.parametrize(
    "observation_space", ["dict_space", "discrete_space", "vector_space", "image_space"]
)
def test_value_function_clone(observation_space: spaces.Space, request):
    observation_space = request.getfixturevalue(observation_space)

    network = ValueNetwork(observation_space)

    original_net_dict = dict(network.named_parameters())
    clone = network.clone()
    assert isinstance(clone, EvolvableNetwork)

    assert_close_dict(network.init_dict, clone.init_dict)

    assert_state_dicts_equal(clone.state_dict(), network.state_dict())
    for key, param in clone.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key])
