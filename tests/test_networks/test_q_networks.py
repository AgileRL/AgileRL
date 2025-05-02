import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.networks.base import EvolvableNetwork
from agilerl.networks.q_networks import ContinuousQNetwork, QNetwork, RainbowQNetwork
from tests.helper_functions import (
    assert_close_dict,
    check_equal_params_ind,
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_random_box_space,
)


@pytest.mark.parametrize(
    "observation_space, encoder_type",
    [
        (generate_dict_or_tuple_space(2, 3), "multi_input"),
        (generate_discrete_space(4), "mlp"),
        (generate_random_box_space((8,)), "mlp"),
        (generate_random_box_space((3, 32, 32)), "cnn"),
    ],
)
def test_q_network_initialization(observation_space, encoder_type):
    action_space = spaces.Discrete(4)
    network = QNetwork(observation_space, action_space)

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


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_q_network_mutation_methods(observation_space):
    action_space = spaces.Discrete(4)
    network = QNetwork(observation_space, action_space)

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

        check_equal_params_ind(network, new_network)


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_q_network_forward(observation_space: spaces.Space):
    action_space = spaces.Discrete(4)
    network = QNetwork(observation_space, action_space)

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
    assert out.shape == torch.Size([1, 4])

    if isinstance(observation_space, spaces.Dict):
        x = {key: torch.tensor(value) for key, value in x_np.items()}
    elif isinstance(observation_space, spaces.Tuple):
        x = tuple(torch.tensor(value) for value in x_np)
    else:
        x = torch.tensor(x_np)

    with torch.no_grad():
        out = network(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 4])


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_q_network_clone(observation_space: spaces.Space):
    action_space = spaces.Discrete(4)
    network = QNetwork(observation_space, action_space)

    original_net_dict = dict(network.named_parameters())
    clone = network.clone()
    assert isinstance(clone, EvolvableNetwork)

    assert_close_dict(network.init_dict, clone.init_dict)

    assert str(clone.state_dict()) == str(network.state_dict())
    for key, param in clone.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key])


@pytest.mark.parametrize(
    "observation_space, encoder_type",
    [
        (generate_dict_or_tuple_space(2, 3), "multi_input"),
        (generate_discrete_space(4), "mlp"),
        (generate_random_box_space((8,)), "mlp"),
        (generate_random_box_space((3, 32, 32)), "cnn"),
    ],
)
def test_rainbow_q_network_initialization(observation_space, encoder_type):
    action_space = spaces.Discrete(4)
    support = torch.linspace(-10, 10, 51)
    network = RainbowQNetwork(observation_space, action_space, support)

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


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_rainbow_q_network_mutation_methods(observation_space):
    action_space = spaces.Discrete(4)
    support = torch.linspace(-10, 10, 51)
    network = RainbowQNetwork(observation_space, action_space, support)

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

        check_equal_params_ind(network, new_network)


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_rainbow_q_network_forward(observation_space: spaces.Space):
    action_space = spaces.Discrete(4)
    support = torch.linspace(-10, 10, 51)
    network = RainbowQNetwork(observation_space, action_space, support)

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
    assert out.shape == torch.Size([1, 4])

    if isinstance(observation_space, spaces.Dict):
        x = {key: torch.tensor(value) for key, value in x_np.items()}
    elif isinstance(observation_space, spaces.Tuple):
        x = tuple(torch.tensor(value) for value in x_np)
    else:
        x = torch.tensor(x_np)

    with torch.no_grad():
        out = network(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 4])


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_rainbow_q_network_clone(observation_space: spaces.Space):
    action_space = spaces.Discrete(4)
    support = torch.linspace(-10, 10, 51)
    network = RainbowQNetwork(observation_space, action_space, support)

    original_net_dict = dict(network.named_parameters())
    clone = network.clone()
    assert isinstance(clone, EvolvableNetwork)

    assert_close_dict(network.init_dict, clone.init_dict)

    assert str(clone.state_dict()) == str(network.state_dict())
    for key, param in clone.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key])


@pytest.mark.parametrize(
    "observation_space, encoder_type",
    [
        (generate_dict_or_tuple_space(2, 3), "multi_input"),
        (generate_discrete_space(4), "mlp"),
        (generate_random_box_space((8,)), "mlp"),
        (generate_random_box_space((3, 32, 32)), "cnn"),
    ],
)
def test_continuous_q_network_initialization(observation_space, encoder_type):
    action_space = spaces.Box(low=-1, high=1, shape=(4,))
    network = ContinuousQNetwork(observation_space, action_space)

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


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_continuous_q_network_mutation_methods(observation_space):
    action_space = spaces.Box(low=-1, high=1, shape=(4,))
    network = ContinuousQNetwork(observation_space, action_space)

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

        check_equal_params_ind(network, new_network)


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_continuous_q_network_forward(observation_space: spaces.Space):
    action_space = spaces.Box(low=-1, high=1, shape=(4,))
    network = ContinuousQNetwork(observation_space, action_space)

    x_np = observation_space.sample()
    actions_np = action_space.sample()

    if isinstance(observation_space, spaces.Discrete):
        x_np = (
            F.one_hot(torch.tensor(x_np), num_classes=observation_space.n)
            .float()
            .numpy()
        )

    with torch.no_grad():
        out = network(x_np, actions_np)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 1])

    if isinstance(observation_space, spaces.Dict):
        x = {key: torch.tensor(value) for key, value in x_np.items()}
    elif isinstance(observation_space, spaces.Tuple):
        x = tuple(torch.tensor(value) for value in x_np)
    else:
        x = torch.tensor(x_np)

    actions = torch.tensor(actions_np)
    with torch.no_grad():
        out = network(x, actions)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 1])


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_continuous_q_network_clone(observation_space: spaces.Space):
    action_space = spaces.Box(low=-1, high=1, shape=(4,))
    network = ContinuousQNetwork(observation_space, action_space)

    original_net_dict = dict(network.named_parameters())
    clone = network.clone()
    assert isinstance(clone, EvolvableNetwork)

    assert_close_dict(network.init_dict, clone.init_dict)

    assert str(clone.state_dict()) == str(network.state_dict())
    for key, param in clone.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key])
