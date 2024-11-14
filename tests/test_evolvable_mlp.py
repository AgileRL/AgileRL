import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.networks.custom_components import NoisyLinear

######### Define fixtures #########
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_noisy_linear(device):
    noisy_linear = NoisyLinear(2, 10).to(device)
    noisy_linear.training = False
    with torch.no_grad():
        output = noisy_linear.forward(torch.randn(1, 2).to(device))
    assert output.shape == (1, 10)


######### Test instantiation #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [(10, 5, [32, 64, 128]), (2, 1, [32]), (100, 3, [8, 8, 8, 8, 8, 8, 8])],
)
def test_instantiation(num_inputs, num_outputs, hidden_size, device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        device=device,
    )
    assert isinstance(evolvable_mlp, EvolvableMLP)


@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [(0, 20, [16]), (20, 0, [16]), (10, 2, []), (10, 2, [0])],
)
def test_incorrect_instantiation(num_inputs, num_outputs, hidden_size, device):
    with pytest.raises(Exception):
        EvolvableMLP(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            hidden_size=hidden_size,
            device=device,
        )


@pytest.mark.parametrize(
    "activation, output_activation",
    [
        ("ELU", "Softmax"),
        ("Tanh", "PReLU"),
        ("LeakyReLU", "GELU"),
        ("ReLU", "Sigmoid"),
        ("Tanh", "Softplus"),
        ("Tanh", "Softsign"),
    ],
)
def test_instantiation_with_different_activations(
    activation, output_activation, device
):
    evolvable_mlp = EvolvableMLP(
        num_inputs=6,
        num_outputs=4,
        hidden_size=[32],
        mlp_activation=activation,
        mlp_output_activation=output_activation,
        output_vanish=True,
        device=device,
    )
    assert isinstance(evolvable_mlp, EvolvableMLP)


def test_reset_noise(device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=10,
        num_outputs=4,
        hidden_size=[32, 32],
        rainbow=True,
        output_vanish=True,
        support=torch.linspace(0.0, 200.0, 50).to(device),
        device=device,
    )
    evolvable_mlp.reset_noise()
    assert isinstance(evolvable_mlp.value_net[0], NoisyLinear)
    assert isinstance(evolvable_mlp.advantage_net[0], NoisyLinear)

def test_rainbow_instantiation(device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=10,
        num_outputs=4,
        hidden_size=[32, 32],
        rainbow=True,
        support=torch.linspace(0.0, 200.0, 50).to(device),
        output_vanish=True,
        device=device,
    )
    feature_net, value_net, advantage_net = evolvable_mlp.build_networks()
    assert isinstance(evolvable_mlp, EvolvableMLP)
    assert isinstance(feature_net, nn.Module)
    assert isinstance(advantage_net, nn.Module)
    assert isinstance(value_net, nn.Module)

######### Test forward #########
@pytest.mark.parametrize(
    "input_tensor, num_inputs, num_outputs, hidden_size, output_size",
    [
        (torch.randn(1, 10), 10, 5, [32, 64, 128], (1, 5)),
        (torch.randn(1, 2), 2, 1, [32], (1, 1)),
        (torch.randn(1, 100), 100, 3, [8, 8, 8, 8, 8, 8, 8], (1, 3)),
        (np.random.randn(1, 100), 100, 3, [8, 8, 8, 8, 8, 8, 8], (1, 3)),
    ],
)
def test_forward(
    input_tensor, num_inputs, num_outputs, hidden_size, output_size, device
):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        device=device,
    )
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output_tensor = evolvable_mlp.forward(input_tensor)
    assert output_tensor.shape == output_size


def test_forward_rainbow(device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=10,
        num_outputs=4,
        hidden_size=[32, 32],
        rainbow=True,
        support=torch.linspace(0.0, 200.0, 51).to(device),
        device=device,
    )
    input_tensor = torch.randn(1, 10).to(device)
    with torch.no_grad():
        output = evolvable_mlp.forward(input_tensor)
    assert output.shape == (1, 4)


######### Test add_mlp_layer #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [
        (10, 5, [32, 64, 128]),
        (2, 1, [32]),
        (100, 3, [8, 8, 8, 8, 8, 8, 8]),
        (10, 4, [16] * 10),
    ],
)
def test_add_mlp_layer(num_inputs, num_outputs, hidden_size, device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        max_hidden_layers=10,
        device=device,
    )

    initial_hidden_size = len(evolvable_mlp.hidden_size)
    initial_net = evolvable_mlp.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_mlp.add_mlp_layer()
    new_net = evolvable_mlp.feature_net
    if initial_hidden_size < 10:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size + 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size


def test_add_mlp_layer_rainbow(device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=10,
        num_outputs=4,
        hidden_size=[32, 32],
        rainbow=True,
        support=torch.linspace(0.0, 200.0, 50).to(device),
        device=device,
    )
    initial_hidden_size = len(evolvable_mlp.hidden_size)
    initial_net = evolvable_mlp.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_mlp.add_mlp_layer()
    new_net = evolvable_mlp.feature_net
    if initial_hidden_size < 10:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size + 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size


######### Test remove_mlp_layer #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [(10, 5, [32, 64, 128]), (2, 1, [32]), (100, 3, [8, 8, 8, 8, 8, 8, 8])],
)
def test_remove_mlp_layer(num_inputs, num_outputs, hidden_size, device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        min_hidden_layers=1,
        max_hidden_layers=10,
        device=device,
    )

    initial_hidden_size = len(evolvable_mlp.hidden_size)
    initial_net = evolvable_mlp.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_mlp.remove_mlp_layer()
    new_net = evolvable_mlp.feature_net
    if initial_hidden_size > 1:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size - 1
        for key, param in new_net.named_parameters():
            print(key, param.shape)
            if (
                key in initial_net_dict.keys()
                and param.shape == initial_net_dict[key].shape
            ):
                torch.testing.assert_close(param, initial_net_dict[key]), evolvable_mlp
    else:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size


def test_remove_mlp_layer_rainbow(device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=10,
        num_outputs=4,
        hidden_size=[32, 32, 32],
        rainbow=True,
        output_vanish=True,
        support=torch.linspace(0.0, 200.0, 50).to(device),
        device=device,
    )

    initial_hidden_size = len(evolvable_mlp.hidden_size)
    initial_net = evolvable_mlp.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_mlp.remove_mlp_layer()
    new_net = evolvable_mlp.feature_net
    if initial_hidden_size > 1:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size - 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key]), evolvable_mlp
    else:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size


######### Test add_mlp_node #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size, hidden_layer, numb_new_nodes",
    [
        (10, 5, [32, 64, 128], None, 4),
        (2, 1, [32], None, None),
        (100, 3, [8, 8, 8, 8, 8, 8, 8], 1, None),
    ],
)
def test_add_nodes(
    num_inputs, num_outputs, hidden_size, hidden_layer, numb_new_nodes, device
):
    mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        device=device,
    )
    original_hidden_size = copy.deepcopy(mlp.hidden_size)
    result = mlp.add_mlp_node(hidden_layer=hidden_layer, numb_new_nodes=numb_new_nodes)
    hidden_layer = result["hidden_layer"]
    numb_new_nodes = result["numb_new_nodes"]
    assert (
        mlp.hidden_size[hidden_layer]
        == original_hidden_size[hidden_layer] + numb_new_nodes
    )


######### Test remove_mlp_node #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size, hidden_layer, numb_new_nodes",
    [
        (10, 5, [256, 256, 256], 1, None),
        (2, 1, [32], None, 4),
        (100, 3, [8, 8, 8, 8, 8, 8, 8], None, 4),
    ],
)
def test_remove_nodes(
    num_inputs, num_outputs, hidden_size, hidden_layer, numb_new_nodes, device
):
    mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        min_mlp_nodes=2,
        device=device,
    )
    original_hidden_size = copy.deepcopy(mlp.hidden_size)
    result = mlp.remove_mlp_node(
        numb_new_nodes=numb_new_nodes, hidden_layer=hidden_layer
    )
    hidden_layer = result["hidden_layer"]
    if numb_new_nodes is None:
        numb_new_nodes = result["numb_new_nodes"]
    assert (
        mlp.hidden_size[hidden_layer]
        == original_hidden_size[hidden_layer] - numb_new_nodes
    )


######### Test clone #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [(10, 5, [32, 64, 128]), (2, 1, [32]), (100, 3, [8, 8, 8, 8, 8, 8, 8])],
)
def test_clone_instance(num_inputs, num_outputs, hidden_size, device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        device=device,
    )
    original_net_dict = dict(evolvable_mlp.feature_net.named_parameters())
    clone = evolvable_mlp.clone()
    clone_net = clone.feature_net
    assert isinstance(clone, EvolvableMLP)
    assert clone.init_dict == evolvable_mlp.init_dict
    assert str(clone.state_dict()) == str(evolvable_mlp.state_dict())
    for key, param in clone_net.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key]), evolvable_mlp


def test_clone_rainbow(device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=10,
        num_outputs=4,
        hidden_size=[32, 32],
        rainbow=True,
        support=torch.linspace(0.0, 200.0, 50).to(device),
        device=device,
    )
    original_feature_net_dict = dict(evolvable_mlp.feature_net.named_parameters())
    original_value_net_dict = dict(evolvable_mlp.value_net.named_parameters())
    original_advantage_net_dict = dict(evolvable_mlp.advantage_net.named_parameters())
    clone = evolvable_mlp.clone()
    clone_feature_net = clone.feature_net
    clone_value_net = clone.value_net
    clone_advantage_net = clone.advantage_net
    assert isinstance(clone, EvolvableMLP)
    print(clone.init_dict)
    print(evolvable_mlp.init_dict)
    for key, value in clone.init_dict.items():
        assert key in evolvable_mlp.init_dict
        if key == "support":
            torch.testing.assert_close(evolvable_mlp.init_dict[key], value)
        else:
            assert evolvable_mlp.init_dict[key] == value
    assert str(clone.state_dict()) == str(evolvable_mlp.state_dict())
    for key, param in clone_feature_net.named_parameters():
        torch.testing.assert_close(param, original_feature_net_dict[key])
    for key, param in clone_value_net.named_parameters():
        torch.testing.assert_close(param, original_value_net_dict[key])
    for key, param in clone_advantage_net.named_parameters():
        torch.testing.assert_close(param, original_advantage_net_dict[key])
