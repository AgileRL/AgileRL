import copy

import pytest
import torch
import torch.nn as nn

from agilerl.modules.custom_components import NoisyLinear
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import unpack_network


class TwoArgCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv3d(
            in_channels=4, out_channels=16, kernel_size=(2, 3, 3), stride=(4, 4, 4)
        )  # W: 160, H: 210
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=(2, 2, 2)
        )  # W:

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(15202, 256)
        self.fc2 = nn.Linear(256, 2)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x, xc):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.cat([x, xc], dim=1)
        # Forward pass through fully connected layers
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax for classification
        x = self.softmax(x)

        return x


@pytest.fixture
def simple_mlp():
    network = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Tanh(),
    )
    return network


@pytest.fixture
def simple_mlp_2():
    network = nn.Sequential(
        nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
    )
    return network


@pytest.fixture
def mlp_norm_layers():
    network = nn.Sequential(
        nn.Linear(10, 20),
        nn.LayerNorm(20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.LayerNorm(10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid(),
    )
    return network


@pytest.fixture
def simple_cnn():
    network = nn.Sequential(
        nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 3 (for RGB images), Output channels: 16
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 16, Output channels: 32
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),  # Flatten the 2D feature map to a 1D vector
        nn.Linear(32 * 16 * 16, 128),  # Fully connected layer with 128 output features
        nn.ReLU(),
        nn.Linear(128, 1),  # Output layer with num_classes output features
    )
    return network


@pytest.fixture
def cnn_norm_layers():
    network = nn.Sequential(
        nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 3 (for RGB images), Output channels: 16
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 16, Output channels: 32
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),  # Flatten the 2D feature map to a 1D vector
        nn.Linear(32 * 16 * 16, 128),  # Fully connected layer with 128 output features
        nn.ReLU(),
        nn.Linear(128, 1),  # Output layer with num_classes output features
    )
    return network


@pytest.fixture
def two_arg_cnn():
    return TwoArgCNN()


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


######### Test instantiation #########
# The class can be instantiated with all the required parameters and no errors occur.
@pytest.mark.parametrize(
    "network, input_tensor",
    [("simple_mlp", torch.randn(1, 10)), ("simple_cnn", torch.randn(1, 3, 64, 64))],
)
def test_instantiation_with_required_parameters(network, input_tensor, request):
    network = request.getfixturevalue(network)
    evolvable_network = MakeEvolvable(network, input_tensor)
    assert isinstance(evolvable_network, MakeEvolvable)
    assert str(unpack_network(evolvable_network)) == str(unpack_network(network))


# The class can be instantiated with minimal parameters and default values are assigned correctly.
def test_instantiation_with_minimal_parameters():
    network = nn.Sequential(
        nn.Linear(10, 20), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 1), nn.ReLU()
    )
    input_tensor = torch.randn(1, 10)
    evolvable_network = MakeEvolvable(network, input_tensor)
    assert isinstance(evolvable_network, MakeEvolvable)


def test_instantiation_with_rainbow():
    network = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
    input_tensor = torch.randn(1, 3)
    support = torch.linspace(-200, 200, 51)
    evolvable_network = MakeEvolvable(
        network, input_tensor, support=support, rainbow=True
    )
    assert isinstance(evolvable_network, MakeEvolvable)
    assert (
        str(evolvable_network)
        == """MakeEvolvable(
  (feature_net): Sequential(
    (feature_linear_layer_0): Linear(in_features=3, out_features=128, bias=True)
    (feature_activation_0): ReLU()
  )
  (value_net): Sequential(
    (value_linear_layer_0): NoisyLinear()
    (value_activation_0): ReLU()
    (value_linear_layer_output): NoisyLinear()
  )
  (advantage_net): Sequential(
    (advantage_linear_layer_0): NoisyLinear()
    (advantage_activation_0): ReLU()
    (advantage_linear_layer_output): NoisyLinear()
  )
)"""
    )

######### Test forward #########
@pytest.mark.parametrize(
    "network, input_tensor, secondary_input_tensor, expected_result",
    [
        ("simple_mlp", torch.randn(1, 10), None, (1, 1)),
        ("simple_cnn", torch.randn(1, 3, 64, 64), None, (1, 1)),
        ("two_arg_cnn", torch.randn(1, 4, 2, 210, 160), torch.randn(1, 2), (1, 2)),
    ],
)
def test_forward_method(
    network, input_tensor, secondary_input_tensor, expected_result, request, device
):
    network = request.getfixturevalue(network)
    if secondary_input_tensor is None:
        evolvable_network = MakeEvolvable(network, input_tensor, device=device)
        with torch.no_grad():
            actual_output = evolvable_network.forward(input_tensor)
    else:
        evolvable_network = MakeEvolvable(
            network,
            input_tensor,
            secondary_input_tensor,
            device=device,
        )
        with torch.no_grad():
            input_tensor = input_tensor.to(dtype=torch.float16)
            actual_output = evolvable_network.forward(
                input_tensor, secondary_input_tensor
            )
    output_shape = actual_output.shape
    assert output_shape == expected_result


@pytest.mark.parametrize(
    "network, input_tensor, secondary_input_tensor, expected_result",
    [
        ("simple_mlp", torch.randn(1, 10), None, (1, 1)),
        ("simple_cnn", torch.randn(1, 3, 64, 64), None, (1, 1)),
        ("two_arg_cnn", torch.randn(1, 4, 2, 210, 160), torch.randn(1, 2), (1, 2)),
    ],
)
def test_forward_method_rainbow(
    network, input_tensor, secondary_input_tensor, expected_result, request, device
):
    network = request.getfixturevalue(network)
    support = torch.linspace(-200, 200, 51).to(device)
    if secondary_input_tensor is None:
        evolvable_network = MakeEvolvable(
            network, input_tensor, support=support, rainbow=True, device=device
        )
        with torch.no_grad():
            actual_output = evolvable_network.forward(input_tensor)
    else:
        evolvable_network = MakeEvolvable(
            network,
            input_tensor,
            secondary_input_tensor,
            support=support,
            rainbow=True,
            device=device,
        )
        with torch.no_grad():
            input_tensor = input_tensor.to(dtype=torch.float16)
            actual_output = evolvable_network.forward(
                input_tensor, secondary_input_tensor
            )
    output_shape = actual_output.shape
    assert output_shape == expected_result


# The forward() method can handle different types of input tensors (e.g., numpy array, torch tensor).
def test_forward_method_with_different_input_types(simple_mlp):
    input_tensor = torch.randn(1, 10)
    numpy_array = input_tensor.numpy()
    evolvable_network = MakeEvolvable(simple_mlp, input_tensor)
    with torch.no_grad():
        output1 = evolvable_network.forward(input_tensor)
        output2 = evolvable_network.forward(numpy_array)
    assert isinstance(output1, torch.Tensor)
    assert isinstance(output2, torch.Tensor)


# The forward() method can handle different types of normalization layers (e.g., BatchNorm2d, InstanceNorm3d).
def test_forward_with_different_normalization_layers():
    network = nn.Sequential(
        nn.Linear(10, 20),
        nn.LayerNorm(20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
    )
    input_tensor = torch.randn(1, 10)
    evolvable_network = MakeEvolvable(network, input_tensor)
    with torch.no_grad():
        output = evolvable_network.forward(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert str(unpack_network(evolvable_network)) == str(unpack_network(network))


def test_reset_noise():
    network = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
    input_tensor = torch.randn(1, 3)
    support = torch.linspace(-200, 200, 51)
    evolvable_mlp = MakeEvolvable(network, input_tensor, support=support, rainbow=True)
    evolvable_mlp.reset_noise()
    assert isinstance(evolvable_mlp.value_net[0], NoisyLinear)
    assert isinstance(evolvable_mlp.advantage_net[0], NoisyLinear)


######### Test detect architecture function #########


# Detects architecture of a neural network with convolutional layers and without normalization layers
def test_detect_architecture_mlp_simple(device):
    net = nn.Sequential(
        nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)
    )
    evolvable_net = MakeEvolvable(
        net, torch.randn(1, 4), device=device, output_vanish=True
    )
    assert evolvable_net.mlp_layer_info == {"activation_layers": {0: "ReLU", 1: "ReLU"}}
    assert str(unpack_network(net)) == str(unpack_network(evolvable_net))


def test_detect_architecture_medium(device):
    net = nn.Sequential(
        nn.Linear(4, 16), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()
    )
    evolvable_net = MakeEvolvable(net, torch.randn(1, 4), device=device)
    assert evolvable_net.mlp_layer_info == {"activation_layers": {1: "ReLU", 2: "Tanh"}}
    assert str(unpack_network(net)) == str(unpack_network(evolvable_net))


def test_detect_architecture_complex(device):
    net = nn.Sequential(
        nn.Linear(4, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Tanh(),
    )
    evolvable_net = MakeEvolvable(net, torch.randn(1, 4), device=device)
    assert evolvable_net.mlp_layer_info == {
        "activation_layers": {1: "ReLU", 2: "Tanh"},
        "norm_layers": {0: "LayerNorm"},
    }, evolvable_net.mlp_layer_info
    assert str(unpack_network(net)) == str(unpack_network(evolvable_net))


# Test if network after detect arch has the same arch as original network
@pytest.mark.parametrize(
    "network, input_tensor",
    [
        ("simple_mlp", torch.randn(1, 10)),
        ("simple_cnn", torch.randn(1, 3, 64, 64)),
        ("mlp_norm_layers", torch.randn(1, 10)),
        ("cnn_norm_layers", torch.randn(1, 3, 64, 64)),
    ],
)
def test_detect_architecture_networks_the_same(network, input_tensor, device, request):
    network = request.getfixturevalue(network)
    evolvable_network = MakeEvolvable(network, input_tensor, device=device)
    assert str(unpack_network(network)) == str(unpack_network(evolvable_network))


def test_detect_architecure_exception(device):
    net = nn.Sequential(
        nn.Linear(4, 16),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.LogSoftmax(dim=-1),
    )
    with pytest.raises(Exception):
        MakeEvolvable(net, torch.randn(1, 4), device=device)


def test_detect_architecture_empty_hidden_size(device):
    net = nn.Sequential(nn.Linear(2, 10))
    with pytest.raises(TypeError):
        MakeEvolvable(net, torch.randn(1, 2), device=device)


def test_detect_architecure_different_acitvations(device):
    net = nn.Sequential(
        nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 4)
    )
    with pytest.raises(TypeError):
        MakeEvolvable(net, torch.randn(1, 2), device=device)


######### Test add_mlp_layer #########


def test_add_mlp_layer_simple(simple_mlp, device):
    input_tensor = torch.randn(1, 10)
    evolvable_network = MakeEvolvable(
        simple_mlp, input_tensor, init_layers=True, device=device
    )
    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())
    initial_num_layers = len(evolvable_network.hidden_size)
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {0: "ReLU", 1: "ReLU", 2: "Tanh"}
    }, evolvable_network.mlp_layer_info
    evolvable_network.add_mlp_layer()
    new_feature_net = evolvable_network.feature_net
    assert len(evolvable_network.hidden_size) == initial_num_layers + 1
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {0: "ReLU", 1: "ReLU", 2: "ReLU", 3: "Tanh"}
    }, evolvable_network.mlp_layer_info
    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            assert torch.equal(param, feature_net_dict[key])


def test_add_mlp_layer_medium(device):
    network = nn.Sequential(
        nn.Linear(4, 16), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()
    )
    evolvable_network = MakeEvolvable(network, torch.randn(1, 4), device=device)
    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())
    initial_num_layers = len(evolvable_network.hidden_size)
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {1: "ReLU", 2: "Tanh"}
    }, evolvable_network.mlp_layer_info
    evolvable_network.add_mlp_layer()
    new_feature_net = evolvable_network.feature_net
    assert len(evolvable_network.hidden_size) == initial_num_layers + 1
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {1: "ReLU", 2: "ReLU", 3: "Tanh"}
    }, evolvable_network.mlp_layer_info
    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            assert torch.equal(param, feature_net_dict[key])


def test_add_mlp_layer_complex(device):
    net = nn.Sequential(
        nn.Linear(4, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Tanh(),
    )
    evolvable_network = MakeEvolvable(net, torch.randn(1, 4), device=device)
    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())
    initial_num_layers = len(evolvable_network.hidden_size)
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {1: "ReLU", 2: "Tanh"},
        "norm_layers": {0: "LayerNorm"},
    }, evolvable_network.mlp_layer_info
    evolvable_network.add_mlp_layer()
    new_feature_net = evolvable_network.feature_net
    assert len(evolvable_network.hidden_size) == initial_num_layers + 1
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {1: "ReLU", 2: "ReLU", 3: "Tanh"},
        "norm_layers": {0: "LayerNorm"},
    }, evolvable_network.mlp_layer_info
    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            assert torch.equal(param, feature_net_dict[key])


def test_add_mlp_layer_else_statement(device):
    network = nn.Sequential(
        nn.Linear(4, 16), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()
    )
    evolvable_network = MakeEvolvable(
        network, torch.randn(1, 4), device=device, max_hidden_layers=2
    )
    initial_num_layers = len(evolvable_network.hidden_size)
    evolvable_network.add_mlp_layer()
    assert initial_num_layers == len(evolvable_network.hidden_size)


######### Test remove_mlp_layer #########
def test_remove_mlp_layer_simple(simple_mlp_2, device):
    input_tensor = torch.randn(1, 10)
    evolvable_network = MakeEvolvable(simple_mlp_2, input_tensor, device=device)
    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())
    initial_num_layers = len(evolvable_network.hidden_size)
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {0: "ReLU", 1: "ReLU"}
    }
    evolvable_network.remove_mlp_layer()
    new_feature_net = evolvable_network.feature_net
    assert len(evolvable_network.hidden_size) == initial_num_layers - 1
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {0: "ReLU"}
    }, evolvable_network.mlp_layer_info
    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            torch.testing.assert_close(param, feature_net_dict[key])


def test_remove_mlp_layer_medium(device):
    network = nn.Sequential(
        nn.Linear(4, 16), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()
    )
    evolvable_network = MakeEvolvable(network, torch.randn(1, 4), device=device)
    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())
    initial_num_layers = len(evolvable_network.hidden_size)
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {1: "ReLU", 2: "Tanh"}
    }, evolvable_network.mlp_layer_info
    evolvable_network.remove_mlp_layer()
    new_feature_net = evolvable_network.feature_net
    assert len(evolvable_network.hidden_size) == initial_num_layers - 1
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {1: "Tanh"}
    }, evolvable_network.mlp_layer_info
    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            assert torch.equal(param, feature_net_dict[key])


def test_remove_mlp_layer_complex(device):
    net = nn.Sequential(
        nn.Linear(4, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 16),
        nn.LayerNorm(16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Tanh(),
    )
    evolvable_network = MakeEvolvable(net, torch.randn(1, 4), device=device)
    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())
    initial_num_layers = len(evolvable_network.hidden_size)
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {1: "ReLU", 2: "Tanh"},
        "norm_layers": {0: "LayerNorm", 1: "LayerNorm"},
    }, evolvable_network.mlp_layer_info
    evolvable_network.remove_mlp_layer()
    new_feature_net = evolvable_network.feature_net
    assert len(evolvable_network.hidden_size) == initial_num_layers - 1
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {1: "Tanh"},
        "norm_layers": {0: "LayerNorm"},
    }, evolvable_network.mlp_layer_info
    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            assert torch.equal(param, feature_net_dict[key])


def test_remove_mlp_layer_else_statement(device):
    network = nn.Sequential(
        nn.Linear(4, 16), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()
    )
    evolvable_network = MakeEvolvable(
        network, torch.randn(1, 4), device=device, min_hidden_layers=2
    )
    initial_num_layers = len(evolvable_network.hidden_size)
    evolvable_network.remove_mlp_layer()
    assert initial_num_layers == len(evolvable_network.hidden_size)


def test_remove_mlp_layer_no_output_activation(device):
    net = nn.Sequential(
        nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16), nn.Linear(16, 1), nn.Tanh()
    )
    evolvable_network = MakeEvolvable(net, torch.randn(1, 4), device=device)
    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())
    initial_num_layers = len(evolvable_network.hidden_size)
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {0: "ReLU", 2: "Tanh"}
    }, evolvable_network.mlp_layer_info
    evolvable_network.remove_mlp_layer()
    new_feature_net = evolvable_network.feature_net
    assert len(evolvable_network.hidden_size) == initial_num_layers - 1
    assert evolvable_network.mlp_layer_info == {
        "activation_layers": {0: "ReLU", 1: "Tanh"}
    }, evolvable_network.mlp_layer_info
    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            assert torch.equal(param, feature_net_dict[key])


######### Test add_mlp_node #########
def test_add_mlp_node_fixed(simple_mlp, device):
    input_tensor = torch.randn(1, 10)
    evolvable_network = MakeEvolvable(simple_mlp, input_tensor, device=device)

    # Test adding a new node to a specific layer
    hidden_layer = 1
    numb_new_nodes = 8
    result = evolvable_network.add_mlp_node(hidden_layer, numb_new_nodes)

    # Check if the hidden layer and number of new nodes are updated correctly
    assert evolvable_network.hidden_size[hidden_layer] == 18
    assert result["hidden_layer"] == hidden_layer
    assert result["numb_new_nodes"] == numb_new_nodes


######### Test remove_mlp_node #########
def test_remove_mlp_node(simple_mlp_2, device):
    input_tensor = torch.randn(1, 10)
    evolvable_network = MakeEvolvable(simple_mlp_2, input_tensor, device=device)

    # Check the initial number of nodes in the hidden layers
    assert len(evolvable_network.hidden_size) == 2

    # Remove a node from the second hidden layer
    evolvable_network.remove_mlp_node(hidden_layer=1, numb_new_nodes=10)

    # Check that the number of nodes in the second hidden layer has decreased by 10
    assert evolvable_network.hidden_size[1] == 118

    # Remove a node from the first hidden layer
    evolvable_network.remove_mlp_node(hidden_layer=0, numb_new_nodes=5)

    # Check that the number of nodes in the first hidden layer has decreased by 5
    assert evolvable_network.hidden_size[0] == 123


def test_remove_mlp_node_no_arg(device):
    mlp = nn.Sequential(
        nn.Linear(10, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    input_tensor = torch.randn(1, 10)
    evolvable_network = MakeEvolvable(mlp, input_tensor, device=device, min_mlp_nodes=2)
    original_hidden_size = copy.deepcopy(evolvable_network.hidden_size)
    # Test adding a new node to a specific layer
    hidden_layer = None
    numb_new_nodes = None
    result = evolvable_network.remove_mlp_node(hidden_layer, numb_new_nodes)
    hidden_layer = result["hidden_layer"]
    numb_new_nodes = result["numb_new_nodes"]

    # Check if the hidden layer and number of new nodes are updated correctly
    assert (
        evolvable_network.hidden_size[hidden_layer]
        == original_hidden_size[hidden_layer] - numb_new_nodes
    )


######### Test add_cnn_layer #########
def test_make_evo_add_cnn_layer(simple_cnn, device):
    input_tensor = torch.randn(1, 3, 64, 64)
    evolvable_network = MakeEvolvable(simple_cnn, input_tensor, device=device)

    # Check the initial number of layers
    assert len(evolvable_network.channel_size) == 2

    # Add a new CNN layer
    evolvable_network.add_cnn_layer()

    # Check if a new layer has been added
    assert len(evolvable_network.channel_size) == 3
    assert evolvable_network.cnn_layer_info == {
        "activation_layers": {0: "ReLU", 1: "ReLU", 2: "ReLU"},
        "conv_layer_type": "Conv2d",
        "pooling_layers": {
            0: {"name": "MaxPool2d", "kernel": 2, "stride": 2, "padding": 0},
            1: {"name": "MaxPool2d", "kernel": 2, "stride": 2, "padding": 0},
        },
    }, evolvable_network.cnn_layer_info


def test_make_evo_add_cnn_layer_multi(two_arg_cnn, device):
    evolvable_cnn = MakeEvolvable(
        two_arg_cnn,
        torch.randn(1, 4, 2, 210, 160),
        torch.randn(1, 2),
        device=device,
    )
    original_channels = copy.deepcopy(evolvable_cnn.channel_size)
    evolvable_cnn.add_cnn_layer()
    assert len(original_channels) + 1 == len(evolvable_cnn.channel_size)


def test_make_evo_add_cnn_layer_no_activation(device):
    cnn = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(32768, 128),  # Fully connected layer with 128 output features
        nn.ReLU(),
        nn.Linear(128, 4),
    )
    evolvable_cnn = MakeEvolvable(cnn, torch.randn(1, 3, 32, 32), device=device)
    original_channels = copy.deepcopy(evolvable_cnn.channel_size)
    evolvable_cnn.add_cnn_layer()
    assert len(original_channels) + 1 == len(evolvable_cnn.channel_size)


def test_make_evo_add_cnn_layer_else_statement(simple_cnn, device):
    evolvable_cnn = MakeEvolvable(
        simple_cnn, torch.randn(1, 3, 64, 64), device=device, max_cnn_hidden_layers=2
    )
    original_channels = copy.deepcopy(evolvable_cnn.channel_size)
    evolvable_cnn.add_cnn_layer()
    assert len(original_channels) == len(evolvable_cnn.channel_size)


######### Test remove_cnn_layer #########
def test_remove_cnn_layer(simple_cnn, device):
    input_tensor = torch.randn(1, 3, 64, 64)
    evolvable_network = MakeEvolvable(simple_cnn, input_tensor, device=device)

    # Check the initial number of layers
    assert len(evolvable_network.channel_size) == 2

    # Remove a CNN layer
    evolvable_network.remove_cnn_layer()

    # Check if a layer has been removed
    assert len(evolvable_network.channel_size) == 1
    assert evolvable_network.cnn_layer_info == {
        "activation_layers": {0: "ReLU"},
        "conv_layer_type": "Conv2d",
        "pooling_layers": {
            0: {"name": "MaxPool2d", "kernel": 2, "stride": 2, "padding": 0}
        },
    }, evolvable_network.cnn_layer_info


def test_remove_cnn_layer_no_activation(device):
    cnn = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.Flatten(),
        nn.Linear(32768, 128),  # Fully connected layer with 128 output features
        nn.ReLU(),
        nn.Linear(128, 4),
    )
    evolvable_cnn = MakeEvolvable(cnn, torch.randn(1, 3, 32, 32), device=device)
    original_channels = copy.deepcopy(evolvable_cnn.channel_size)
    evolvable_cnn.remove_cnn_layer()
    assert len(original_channels) - 1 == len(evolvable_cnn.channel_size)


def test_remove_cnn_layer_else_statement(simple_cnn, device):
    evolvable_cnn = MakeEvolvable(
        simple_cnn, torch.randn(1, 3, 64, 64), device=device, min_cnn_hidden_layers=2
    )
    original_channels = copy.deepcopy(evolvable_cnn.channel_size)
    evolvable_cnn.remove_cnn_layer()
    assert len(original_channels) == len(evolvable_cnn.channel_size)


######### Test add_cnn_channel #########
def test_make_evo_add_cnn_channel(simple_cnn, device):
    input_tensor = torch.randn(1, 3, 64, 64)
    evolvable_network = MakeEvolvable(simple_cnn, input_tensor, device=device)
    numb_new_channels = 16
    layer = 1
    original_channel_size = evolvable_network.channel_size[layer]
    evolvable_network.add_cnn_channel(
        hidden_layer=layer, numb_new_channels=numb_new_channels
    )
    assert evolvable_network.channel_size[layer] == original_channel_size + 16


######### Test remove_cnn_channel #########
def test_remove_cnn_channel(device):
    input_tensor = torch.randn(1, 3, 64, 64)
    cnn = nn.Sequential(
        nn.Conv2d(3, 256, 3, 2),
        nn.ReLU(),
        nn.Conv2d(256, 128, 3, 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(28800, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    evolvable_network = MakeEvolvable(cnn, input_tensor, device=device)
    numb_new_channels = None
    layer = None
    original_channel_size = copy.deepcopy(evolvable_network.channel_size)
    evolvable_network.min_channel_size = 16
    result = evolvable_network.remove_cnn_channel(
        hidden_layer=layer, numb_new_channels=numb_new_channels
    )
    numb_new_channels = result["numb_new_channels"]
    hidden_layer = result["hidden_layer"]
    assert (
        evolvable_network.channel_size[hidden_layer]
        == original_channel_size[hidden_layer] - numb_new_channels
    )


def test_remove_cnn_channel_specified_hidden_layer(device):
    input_tensor = torch.randn(1, 3, 64, 64)
    cnn = nn.Sequential(
        nn.Conv2d(3, 256, 3, 2),
        nn.ReLU(),
        nn.Conv2d(256, 128, 3, 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(28800, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    evolvable_network = MakeEvolvable(cnn, input_tensor, device=device)
    numb_new_channels = None
    layer = 0
    original_channel_size = copy.deepcopy(evolvable_network.channel_size)
    evolvable_network.min_channel_size = 16
    result = evolvable_network.remove_cnn_channel(
        hidden_layer=layer, numb_new_channels=numb_new_channels
    )
    numb_new_channels = result["numb_new_channels"]
    assert (
        evolvable_network.channel_size[layer]
        == original_channel_size[layer] - numb_new_channels
    )


######### Test change_cnn_kernel #########
def test_change_cnn_kernel(simple_cnn, device):
    input_tensor = torch.randn(1, 3, 64, 64)
    evolvable_network = MakeEvolvable(simple_cnn, input_tensor, device=device)

    # Check initial kernel sizes
    assert evolvable_network.kernel_size == [(3, 3), (3, 3)]

    # Change kernel size
    evolvable_network.change_cnn_kernel()

    while evolvable_network.kernel_size == [(3, 3), (3, 3)]:
        evolvable_network.change_cnn_kernel()

    # Check if kernel size has changed
    assert evolvable_network.kernel_size != [
        (3, 3),
        (3, 3),
    ], evolvable_network.kernel_size


def test_change_cnn_kernel_else_statement(device):
    network = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=(3, 3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16384, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    input_tensor = torch.randn(1, 3, 64, 64)
    evolvable_network = MakeEvolvable(network, input_tensor, device=device)

    # Change kernel size
    evolvable_network.change_cnn_kernel()
    # Check if kernel size has changed
    assert len(evolvable_network.kernel_size) == 2
    assert len(evolvable_network.kernel_size[-1]) == 2


def test_change_kernel_multi_single_arg(device):
    network = nn.Sequential(
        nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=1),
        nn.ReLU(),
        nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(69120, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    evolvable_cnn = MakeEvolvable(
        network, torch.randn(1, 3, 1, 210, 160), device=device
    )
    while evolvable_cnn.kernel_size == [(1, 3, 3), (1, 3, 3)]:
        evolvable_cnn.change_cnn_kernel()

    assert evolvable_cnn.kernel_size != [
        (1, 3, 3),
        (1, 3, 3),
    ], evolvable_cnn.kernel_size


def test_change_kernel_multi_single_arg_else_statement(device):
    network = nn.Sequential(
        nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(181440, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    evolvable_cnn = MakeEvolvable(
        network, torch.randn(1, 3, 1, 210, 160), device=device
    )
    evolvable_cnn.change_cnn_kernel()
    assert len(evolvable_cnn.kernel_size) == 2
    assert len(evolvable_cnn.kernel_size[-1]) == 3


def test_change_kernel_multi_two_arg(two_arg_cnn, device):
    evolvable_cnn = MakeEvolvable(
        two_arg_cnn,
        torch.randn(1, 4, 2, 210, 160),
        torch.randn(1, 2),
        device=device,
    )
    while evolvable_cnn.kernel_size == [(2, 3, 3), (1, 3, 3)]:
        evolvable_cnn.change_cnn_kernel()

    assert evolvable_cnn.kernel_size != [
        (2, 3, 3),
        (1, 3, 3),
    ], evolvable_cnn.kernel_size


######### Test recreate_nets #########
def test_recreate_nets_parameters_preserved(simple_mlp, device):
    input_tensor = torch.randn(1, 10)
    evolvable_network = MakeEvolvable(simple_mlp, input_tensor, device=device)

    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())

    # Modify the architecture
    evolvable_network.hidden_size += [evolvable_network.hidden_size[-1]]

    evolvable_network.recreate_nets()
    new_feature_net = evolvable_network.feature_net

    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            assert torch.equal(param, feature_net_dict[key])


def test_recreate_nets_parameters_preserved_rainbow(simple_mlp, device):
    input_tensor = torch.randn(1, 10)
    support = torch.linspace(-200, 200, 51).to(device)
    evolvable_network = MakeEvolvable(
        simple_mlp, input_tensor, support=support, rainbow=True, device=device
    )

    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())

    value_net = evolvable_network.value_net
    value_net_dict = dict(value_net.named_parameters())

    advantage_net = evolvable_network.advantage_net
    advantage_net_dict = dict(advantage_net.named_parameters())

    # Modify the architecture
    evolvable_network.hidden_size += [evolvable_network.hidden_size[-1]]

    evolvable_network.recreate_nets()
    new_feature_net = evolvable_network.feature_net
    new_value_net = evolvable_network.value_net
    new_advantage_net = evolvable_network.advantage_net

    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            assert torch.equal(param, feature_net_dict[key])

    for key, param in new_value_net.named_parameters():
        if key in value_net_dict.keys():
            assert torch.equal(param, value_net_dict[key])

    for key, param in new_advantage_net.named_parameters():
        if key in advantage_net_dict.keys():
            assert torch.equal(param, advantage_net_dict[key])


def test_recreate_nets_parameters_shrink_preserved(device):
    network = nn.Sequential(
        nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)
    )

    input_tensor = torch.randn(1, 4)
    evolvable_network = MakeEvolvable(network, input_tensor, device=device)

    feature_net = evolvable_network.feature_net
    feature_net_dict = dict(feature_net.named_parameters())

    # Modify the architecture
    evolvable_network.hidden_size = evolvable_network.hidden_size[:-1]
    evolvable_network.recreate_nets(shrink_params=True)
    new_feature_net = evolvable_network.feature_net

    for key, param in new_feature_net.named_parameters():
        if key in feature_net_dict.keys():
            torch.testing.assert_close(param, feature_net_dict[key])


######### Test clone #########


# The clone() method successfully creates a deep copy of the model.
@pytest.mark.parametrize(
    "network, input_tensor, secondary_input_tensor",
    [
        ("simple_mlp", torch.randn(1, 10), None),
        ("simple_cnn", torch.randn(1, 3, 64, 64), None),
        ("two_arg_cnn", torch.randn(1, 4, 2, 210, 160), torch.randn(1, 2)),
    ],
)
def test_clone_method_with_equal_state_dicts(
    network,
    input_tensor,
    secondary_input_tensor,
    request,
    device,
):
    network = request.getfixturevalue(network)
    if secondary_input_tensor is None:
        evolvable_network = MakeEvolvable(network, input_tensor, device=device)
    else:
        evolvable_network = MakeEvolvable(
            network,
            input_tensor,
            secondary_input_tensor,
            device=device,
        )
    clone_network = evolvable_network.clone()
    assert isinstance(clone_network, MakeEvolvable)
    assert str(evolvable_network.state_dict()) == str(clone_network.state_dict())
