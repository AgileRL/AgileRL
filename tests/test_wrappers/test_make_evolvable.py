import copy
import gc

import pytest
import torch
from torch import nn

from agilerl.modules.custom_components import NoisyLinear
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import assert_state_dicts_equal, unpack_network


# Tiny shapes used for the multi-input Conv3d fixture. Kept large enough that
# ``calc_max_kernel_sizes`` (height_out * 0.2) yields >=3 on the final conv,
# so ``change_cnn_kernel`` can actually pick a different kernel and the test's
# while-loop terminates quickly.
_TWO_ARG_INPUT_SHAPE = (1, 4, 2, 24, 24)
_TWO_ARG_SECONDARY_SHAPE = (1, 2)
# After conv1 (k=(2,3,3), s=(1,1,1)) -> (1, 4, 1, 22, 22)
# After conv2 (k=(1,3,3), s=(1,1,1)) -> (1, 8, 1, 20, 20) -> 3200 features
_TWO_ARG_FLAT_FEATURES = 8 * 1 * 20 * 20 + _TWO_ARG_SECONDARY_SHAPE[-1]


class TwoArgCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Tiny Conv3d stack — enough to exercise the 5D / multi-input code paths
        # without blowing up the fc layer or the per-recreate forward pass.
        self.conv1 = nn.Conv3d(
            in_channels=4,
            out_channels=4,
            kernel_size=(2, 3, 3),
            stride=(1, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            in_channels=4,
            out_channels=8,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
        )

        self.fc1 = nn.Linear(_TWO_ARG_FLAT_FEATURES, 32)
        self.fc2 = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x, xc):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, xc], dim=1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)

        return self.softmax(x)


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
    yield network
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def simple_mlp_2():
    network = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    yield network
    gc.collect()
    torch.cuda.empty_cache()


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
    yield network
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def simple_cnn():
    # 32x32 input -> 16x16 -> 8x8 after two MaxPool(2)s, so flattened size = 32*8*8.
    network = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    yield network
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def cnn_norm_layers():
    network = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    yield network
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def two_arg_cnn():
    network = TwoArgCNN()
    yield network
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


######### Test instantiation #########
class TestMakeEvolvableInit:
    # The class can be instantiated with all the required parameters and no errors occur.
    @pytest.mark.parametrize(
        "network, input_tensor",
        [
            ("simple_mlp", torch.randn(1, 10)),
            ("simple_cnn", torch.randn(1, 3, 32, 32)),
        ],
    )
    def test_instantiation_with_required_parameters(
        self, network, input_tensor, request
    ):
        network = request.getfixturevalue(network)
        evolvable_network = MakeEvolvable(network, input_tensor)
        assert isinstance(evolvable_network, MakeEvolvable)
        assert str(unpack_network(evolvable_network)) == str(unpack_network(network))

    # The class can be instantiated with minimal parameters and default values are assigned correctly.
    def test_instantiation_with_minimal_parameters(self):
        network = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.ReLU(),
        )
        input_tensor = torch.randn(1, 10)
        evolvable_network = MakeEvolvable(network, input_tensor)
        assert isinstance(evolvable_network, MakeEvolvable)

    def test_instantiation_with_rainbow(self):
        network = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
        input_tensor = torch.randn(1, 3)
        support = torch.linspace(-200, 200, 51)
        evolvable_network = MakeEvolvable(
            network,
            input_tensor,
            support=support,
            rainbow=True,
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
    (value_linear_layer_0): NoisyLinear(in_features=128, out_features=8)
    (value_activation_0): ReLU()
    (value_linear_layer_output): NoisyLinear(in_features=8, out_features=51)
  )
  (advantage_net): Sequential(
    (advantage_linear_layer_0): NoisyLinear(in_features=128, out_features=8)
    (advantage_activation_0): ReLU()
    (advantage_linear_layer_output): NoisyLinear(in_features=8, out_features=102)
  )
)"""
        )

    @pytest.mark.gpu
    def test_make_evolvable_init_layers_output_vanish(self, device):
        net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        evolvable = MakeEvolvable(
            net,
            torch.randn(1, 10),
            device=device,
            init_layers=True,
            output_vanish=True,
        )
        with torch.no_grad():
            out = evolvable(torch.randn(2, 10))
        assert out.shape == (2, 1)


######### Test forward #########
class TestMakeEvolvableForward:
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "network, input_tensor, secondary_input_tensor, expected_result",
        [
            ("simple_mlp", torch.randn(1, 10), None, (1, 1)),
            ("simple_cnn", torch.randn(1, 3, 32, 32), None, (1, 1)),
            (
                "two_arg_cnn",
                torch.randn(*_TWO_ARG_INPUT_SHAPE),
                torch.randn(*_TWO_ARG_SECONDARY_SHAPE),
                (1, 2),
            ),
        ],
    )
    def test_forward_method(
        self,
        network,
        input_tensor,
        secondary_input_tensor,
        expected_result,
        request,
        device,
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
                    input_tensor,
                    secondary_input_tensor,
                )
        output_shape = actual_output.shape
        assert output_shape == expected_result

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "network, input_tensor, secondary_input_tensor, expected_result",
        [
            ("simple_mlp", torch.randn(1, 10), None, (1, 1)),
            ("simple_cnn", torch.randn(1, 3, 32, 32), None, (1, 1)),
            (
                "two_arg_cnn",
                torch.randn(*_TWO_ARG_INPUT_SHAPE),
                torch.randn(*_TWO_ARG_SECONDARY_SHAPE),
                (1, 2),
            ),
        ],
    )
    def test_forward_method_rainbow(
        self,
        network,
        input_tensor,
        secondary_input_tensor,
        expected_result,
        request,
        device,
    ):
        network = request.getfixturevalue(network)
        support = torch.linspace(-200, 200, 51).to(device)
        if secondary_input_tensor is None:
            evolvable_network = MakeEvolvable(
                network,
                input_tensor,
                support=support,
                rainbow=True,
                device=device,
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
                    input_tensor,
                    secondary_input_tensor,
                )
        output_shape = actual_output.shape
        assert output_shape == expected_result

    # The forward() method can handle different types of input tensors (e.g., numpy array, torch tensor).
    def test_forward_method_with_different_input_types(self, simple_mlp):
        input_tensor = torch.randn(1, 10)
        numpy_array = input_tensor.numpy()
        evolvable_network = MakeEvolvable(simple_mlp, input_tensor)
        with torch.no_grad():
            output1 = evolvable_network.forward(input_tensor)
            output2 = evolvable_network.forward(numpy_array)
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)

    # The forward() method can handle different types of normalization layers (e.g., BatchNorm2d, InstanceNorm3d).
    def test_forward_with_different_normalization_layers(self):
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


class TestMakeEvolvableResetNoise:
    def test_reset_noise(self):
        network = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
        input_tensor = torch.randn(1, 3)
        support = torch.linspace(-200, 200, 51)
        evolvable_mlp = MakeEvolvable(
            network, input_tensor, support=support, rainbow=True
        )
        evolvable_mlp.reset_noise()
        assert isinstance(evolvable_mlp.value_net[0], NoisyLinear)
        assert isinstance(evolvable_mlp.advantage_net[0], NoisyLinear)


######### Test detect architecture function #########
class TestMakeEvolvableDetectArchitecture:
    # Detects architecture of a neural network with convolutional layers and without normalization layers
    @pytest.mark.gpu
    def test_detect_architecture_mlp_simple(self, device):
        net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        evolvable_net = MakeEvolvable(
            net,
            torch.randn(1, 4),
            device=device,
            output_vanish=True,
        )
        assert evolvable_net.mlp_layer_info == {
            "activation_layers": {0: "ReLU", 1: "ReLU"}
        }
        assert str(unpack_network(net)) == str(unpack_network(evolvable_net))

    @pytest.mark.gpu
    def test_detect_architecture_medium(self, device):
        net = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        evolvable_net = MakeEvolvable(net, torch.randn(1, 4), device=device)
        assert evolvable_net.mlp_layer_info == {
            "activation_layers": {1: "ReLU", 2: "Tanh"}
        }
        assert str(unpack_network(net)) == str(unpack_network(evolvable_net))

    @pytest.mark.gpu
    def test_detect_architecture_complex(self, device):
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
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "network, input_tensor",
        [
            ("simple_mlp", torch.randn(1, 10)),
            ("simple_cnn", torch.randn(1, 3, 32, 32)),
            ("mlp_norm_layers", torch.randn(1, 10)),
            ("cnn_norm_layers", torch.randn(1, 3, 32, 32)),
        ],
    )
    def test_detect_architecture_networks_the_same(
        self, network, input_tensor, device, request
    ):
        network = request.getfixturevalue(network)
        evolvable_network = MakeEvolvable(network, input_tensor, device=device)
        assert str(unpack_network(network)) == str(unpack_network(evolvable_network))

    @pytest.mark.gpu
    def test_detect_architecure_exception(self, device):
        net = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.LogSoftmax(dim=-1),
        )
        with pytest.raises(Exception):  # noqa: B017
            MakeEvolvable(net, torch.randn(1, 4), device=device)

    @pytest.mark.gpu
    def test_detect_architecture_empty_hidden_size(self, device):
        net = nn.Sequential(nn.Linear(2, 10))
        with pytest.raises(TypeError):
            MakeEvolvable(net, torch.randn(1, 2), device=device)

    @pytest.mark.gpu
    def test_detect_architecure_different_acitvations(self, device):
        net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
        )
        with pytest.raises(TypeError):
            MakeEvolvable(net, torch.randn(1, 2), device=device)


######### Test add_mlp_layer #########
class TestMakeEvolvableAddMlpLayer:
    @pytest.mark.gpu
    def test_add_mlp_layer_simple(self, simple_mlp, device):
        input_tensor = torch.randn(1, 10)
        evolvable_network = MakeEvolvable(
            simple_mlp,
            input_tensor,
            init_layers=True,
            device=device,
        )
        feature_net = evolvable_network.feature_net
        feature_net_dict = dict(feature_net.named_parameters())
        initial_num_layers = len(evolvable_network.hidden_size)
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {0: "ReLU", 1: "ReLU", 2: "Tanh"},
        }, evolvable_network.mlp_layer_info
        evolvable_network.add_mlp_layer()
        new_feature_net = evolvable_network.feature_net
        assert len(evolvable_network.hidden_size) == initial_num_layers + 1
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {0: "ReLU", 1: "ReLU", 2: "ReLU", 3: "Tanh"},
        }, evolvable_network.mlp_layer_info
        for key, param in new_feature_net.named_parameters():
            if key in feature_net_dict:
                assert torch.equal(param, feature_net_dict[key])

    @pytest.mark.gpu
    def test_add_mlp_layer_medium(self, device):
        network = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        evolvable_network = MakeEvolvable(network, torch.randn(1, 4), device=device)
        feature_net = evolvable_network.feature_net
        feature_net_dict = dict(feature_net.named_parameters())
        initial_num_layers = len(evolvable_network.hidden_size)
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {1: "ReLU", 2: "Tanh"},
        }, evolvable_network.mlp_layer_info
        evolvable_network.add_mlp_layer()
        new_feature_net = evolvable_network.feature_net
        assert len(evolvable_network.hidden_size) == initial_num_layers + 1
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {1: "ReLU", 2: "ReLU", 3: "Tanh"},
        }, evolvable_network.mlp_layer_info
        for key, param in new_feature_net.named_parameters():
            if key in feature_net_dict:
                assert torch.equal(param, feature_net_dict[key])

    @pytest.mark.gpu
    def test_add_mlp_layer_complex(self, device):
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
            if key in feature_net_dict:
                assert torch.equal(param, feature_net_dict[key])

    @pytest.mark.gpu
    def test_add_mlp_layer_else_statement(self, device):
        network = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        evolvable_network = MakeEvolvable(
            network,
            torch.randn(1, 4),
            device=device,
            max_hidden_layers=2,
        )
        initial_num_layers = len(evolvable_network.hidden_size)
        evolvable_network.add_mlp_layer()
        assert initial_num_layers == len(evolvable_network.hidden_size)


######### Test remove_mlp_layer #########
class TestMakeEvolvableRemoveMlpLayer:
    @pytest.mark.gpu
    def test_remove_mlp_layer_simple(self, simple_mlp_2, device):
        input_tensor = torch.randn(1, 10)
        evolvable_network = MakeEvolvable(simple_mlp_2, input_tensor, device=device)
        feature_net = evolvable_network.feature_net
        feature_net_dict = dict(feature_net.named_parameters())
        initial_num_layers = len(evolvable_network.hidden_size)
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {0: "ReLU", 1: "ReLU"},
        }
        evolvable_network.remove_mlp_layer()
        new_feature_net = evolvable_network.feature_net
        assert len(evolvable_network.hidden_size) == initial_num_layers - 1
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {0: "ReLU"},
        }, evolvable_network.mlp_layer_info
        for key, param in new_feature_net.named_parameters():
            if key in feature_net_dict:
                torch.testing.assert_close(param, feature_net_dict[key])

    @pytest.mark.gpu
    def test_remove_mlp_layer_medium(self, device):
        network = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        evolvable_network = MakeEvolvable(network, torch.randn(1, 4), device=device)
        feature_net = evolvable_network.feature_net
        feature_net_dict = dict(feature_net.named_parameters())
        initial_num_layers = len(evolvable_network.hidden_size)
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {1: "ReLU", 2: "Tanh"},
        }, evolvable_network.mlp_layer_info
        evolvable_network.remove_mlp_layer()
        new_feature_net = evolvable_network.feature_net
        assert len(evolvable_network.hidden_size) == initial_num_layers - 1
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {1: "Tanh"},
        }, evolvable_network.mlp_layer_info
        for key, param in new_feature_net.named_parameters():
            if key in feature_net_dict:
                assert torch.equal(param, feature_net_dict[key])

    @pytest.mark.gpu
    def test_remove_mlp_layer_complex(self, device):
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
            if key in feature_net_dict:
                assert torch.equal(param, feature_net_dict[key])

    @pytest.mark.gpu
    def test_remove_mlp_layer_else_statement(self, device):
        network = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        evolvable_network = MakeEvolvable(
            network,
            torch.randn(1, 4),
            device=device,
            min_hidden_layers=2,
        )
        initial_num_layers = len(evolvable_network.hidden_size)
        evolvable_network.remove_mlp_layer()
        assert initial_num_layers == len(evolvable_network.hidden_size)

    @pytest.mark.gpu
    def test_remove_mlp_layer_no_output_activation(self, device):
        net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        evolvable_network = MakeEvolvable(net, torch.randn(1, 4), device=device)
        feature_net = evolvable_network.feature_net
        feature_net_dict = dict(feature_net.named_parameters())
        initial_num_layers = len(evolvable_network.hidden_size)
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {0: "ReLU", 2: "Tanh"},
        }, evolvable_network.mlp_layer_info
        evolvable_network.remove_mlp_layer()
        new_feature_net = evolvable_network.feature_net
        assert len(evolvable_network.hidden_size) == initial_num_layers - 1
        assert evolvable_network.mlp_layer_info == {
            "activation_layers": {0: "ReLU", 1: "Tanh"},
        }, evolvable_network.mlp_layer_info
        for key, param in new_feature_net.named_parameters():
            if key in feature_net_dict:
                assert torch.equal(param, feature_net_dict[key])


######### Test add_mlp_node #########
class TestMakeEvolvableAddMlpNode:
    @pytest.mark.gpu
    def test_add_mlp_node_fixed(self, simple_mlp, device):
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
class TestMakeEvolvableRemoveMlpNode:
    @pytest.mark.gpu
    def test_remove_mlp_node(self, simple_mlp_2, device):
        input_tensor = torch.randn(1, 10)
        # ``simple_mlp_2`` has 32-node hidden layers so we need a low ``min_mlp_nodes``.
        evolvable_network = MakeEvolvable(
            simple_mlp_2, input_tensor, device=device, min_mlp_nodes=2
        )

        assert len(evolvable_network.hidden_size) == 2

        evolvable_network.remove_mlp_node(hidden_layer=1, numb_new_nodes=10)
        assert evolvable_network.hidden_size[1] == 22

        evolvable_network.remove_mlp_node(hidden_layer=0, numb_new_nodes=5)
        assert evolvable_network.hidden_size[0] == 27

    @pytest.mark.gpu
    def test_remove_mlp_node_no_arg(self, device):
        # ``remove_mlp_node`` (with no args) randomly picks ``numb_new_nodes`` from
        # [16, 32, 64], so both hidden layers need >= 64 + ``min_mlp_nodes`` nodes
        # for the removal to always succeed.
        mlp = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        input_tensor = torch.randn(1, 10)
        evolvable_network = MakeEvolvable(
            mlp, input_tensor, device=device, min_mlp_nodes=2
        )
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
class TestMakeEvolvableAddCnnLayer:
    @pytest.mark.gpu
    def test_make_evo_add_cnn_layer(self, simple_cnn, device):
        input_tensor = torch.randn(1, 3, 32, 32)
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

    @pytest.mark.gpu
    def test_make_evo_add_cnn_layer_multi(self, two_arg_cnn, device):
        evolvable_cnn = MakeEvolvable(
            two_arg_cnn,
            torch.randn(*_TWO_ARG_INPUT_SHAPE),
            torch.randn(*_TWO_ARG_SECONDARY_SHAPE),
            device=device,
        )
        original_channels = copy.deepcopy(evolvable_cnn.channel_size)
        evolvable_cnn.add_cnn_layer()
        assert len(original_channels) + 1 == len(evolvable_cnn.channel_size)

    @pytest.mark.gpu
    def test_make_evo_add_cnn_layer_no_activation(self, device):
        cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        evolvable_cnn = MakeEvolvable(cnn, torch.randn(1, 3, 16, 16), device=device)
        original_channels = copy.deepcopy(evolvable_cnn.channel_size)
        evolvable_cnn.add_cnn_layer()
        assert len(original_channels) + 1 == len(evolvable_cnn.channel_size)

    @pytest.mark.gpu
    def test_make_evo_add_cnn_layer_else_statement(self, simple_cnn, device):
        evolvable_cnn = MakeEvolvable(
            simple_cnn,
            torch.randn(1, 3, 32, 32),
            device=device,
            max_cnn_hidden_layers=2,
        )
        original_channels = copy.deepcopy(evolvable_cnn.channel_size)
        evolvable_cnn.add_cnn_layer()
        assert len(original_channels) == len(evolvable_cnn.channel_size)


######### Test remove_cnn_layer #########
class TestMakeEvolvableRemoveCnnLayer:
    @pytest.mark.gpu
    def test_remove_cnn_layer(self, simple_cnn, device):
        input_tensor = torch.randn(1, 3, 32, 32)
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
                0: {"name": "MaxPool2d", "kernel": 2, "stride": 2, "padding": 0},
            },
        }, evolvable_network.cnn_layer_info

    @pytest.mark.gpu
    def test_remove_cnn_layer_no_activation(self, device):
        cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        evolvable_cnn = MakeEvolvable(cnn, torch.randn(1, 3, 16, 16), device=device)
        original_channels = copy.deepcopy(evolvable_cnn.channel_size)
        evolvable_cnn.remove_cnn_layer()
        assert len(original_channels) - 1 == len(evolvable_cnn.channel_size)

    @pytest.mark.gpu
    def test_remove_cnn_layer_else_statement(self, simple_cnn, device):
        evolvable_cnn = MakeEvolvable(
            simple_cnn,
            torch.randn(1, 3, 32, 32),
            device=device,
            min_cnn_hidden_layers=2,
        )
        original_channels = copy.deepcopy(evolvable_cnn.channel_size)
        evolvable_cnn.remove_cnn_layer()
        assert len(original_channels) == len(evolvable_cnn.channel_size)


######### Test add_cnn_channel #########
class TestMakeEvolvableAddCnnChannel:
    @pytest.mark.gpu
    def test_make_evo_add_cnn_channel(self, simple_cnn, device):
        input_tensor = torch.randn(1, 3, 32, 32)
        evolvable_network = MakeEvolvable(simple_cnn, input_tensor, device=device)
        numb_new_channels = 16
        layer = 1
        original_channel_size = evolvable_network.channel_size[layer]
        evolvable_network.add_cnn_channel(
            hidden_layer=layer,
            numb_new_channels=numb_new_channels,
        )
        assert evolvable_network.channel_size[layer] == original_channel_size + 16


######### Test remove_cnn_channel #########
class TestMakeEvolvableRemoveCnnChannel:
    @pytest.mark.gpu
    def test_remove_cnn_channel(self, device):
        # Channels sized so that any value drawn from [8, 16, 32] keeps both layers
        # strictly above the configured ``min_channel_size`` of 16, so the removal
        # always succeeds.
        input_tensor = torch.randn(1, 3, 16, 16)
        cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        evolvable_network = MakeEvolvable(cnn, input_tensor, device=device)
        numb_new_channels = None
        layer = None
        original_channel_size = copy.deepcopy(evolvable_network.channel_size)
        evolvable_network.min_channel_size = 16
        result = evolvable_network.remove_cnn_channel(
            hidden_layer=layer,
            numb_new_channels=numb_new_channels,
        )
        numb_new_channels = result["numb_new_channels"]
        hidden_layer = result["hidden_layer"]
        assert (
            evolvable_network.channel_size[hidden_layer]
            == original_channel_size[hidden_layer] - numb_new_channels
        )

    @pytest.mark.gpu
    def test_remove_cnn_channel_specified_hidden_layer(self, device):
        input_tensor = torch.randn(1, 3, 16, 16)
        cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        evolvable_network = MakeEvolvable(cnn, input_tensor, device=device)
        numb_new_channels = None
        layer = 0
        original_channel_size = copy.deepcopy(evolvable_network.channel_size)
        evolvable_network.min_channel_size = 16
        result = evolvable_network.remove_cnn_channel(
            hidden_layer=layer,
            numb_new_channels=numb_new_channels,
        )
        numb_new_channels = result["numb_new_channels"]
        assert (
            evolvable_network.channel_size[layer]
            == original_channel_size[layer] - numb_new_channels
        )


######### Test change_cnn_kernel #########
class TestMakeEvolvableChangeCnnKernel:
    @pytest.mark.gpu
    def test_change_cnn_kernel(self, simple_cnn, device):
        input_tensor = torch.randn(1, 3, 32, 32)
        evolvable_network = MakeEvolvable(simple_cnn, input_tensor, device=device)

        assert evolvable_network.kernel_size == [(3, 3), (3, 3)]

        evolvable_network.change_cnn_kernel()

        while evolvable_network.kernel_size == [(3, 3), (3, 3)]:
            evolvable_network.change_cnn_kernel()

        assert evolvable_network.kernel_size != [
            (3, 3),
            (3, 3),
        ], evolvable_network.kernel_size

    @pytest.mark.gpu
    def test_change_cnn_kernel_else_statement(self, device):
        # 16x16 input -> conv keeps 16x16 -> MaxPool(2) -> 8x8 -> 16*8*8 = 1024.
        network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        input_tensor = torch.randn(1, 3, 16, 16)
        evolvable_network = MakeEvolvable(network, input_tensor, device=device)

        evolvable_network.change_cnn_kernel()
        assert len(evolvable_network.kernel_size) == 2
        assert len(evolvable_network.kernel_size[-1]) == 2

    @pytest.mark.gpu
    def test_change_kernel_multi_single_arg(self, device):
        # Tiny Conv3d stack with stride 1, padding 0:
        # input (1,3,1,24,24) -> conv1 (1,22,22) -> conv2 (1,20,20) -> 32*1*20*20 = 12800
        # ``calc_max_kernel_sizes`` gives (1, 4, 4) on the last layer, so picking from
        # [3,4,5,7] yields a kernel different from the original (1,3,3) ~15/16 of the
        # time and the while-loop terminates in O(1) iterations.
        network = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 1 * 20 * 20, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        evolvable_cnn = MakeEvolvable(
            network,
            torch.randn(1, 3, 1, 24, 24),
            device=device,
        )
        while evolvable_cnn.kernel_size == [(1, 3, 3), (1, 3, 3)]:
            evolvable_cnn.change_cnn_kernel()

        assert evolvable_cnn.kernel_size != [
            (1, 3, 3),
            (1, 3, 3),
        ], evolvable_cnn.kernel_size

    @pytest.mark.gpu
    def test_change_kernel_multi_single_arg_else_statement(self, device):
        # Single Conv3d -> ``change_cnn_kernel`` falls through to ``add_cnn_layer``.
        # input (1,3,1,24,24) -> conv (1,22,22) -> 16*1*22*22 = 7744
        network = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 1 * 22 * 22, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        evolvable_cnn = MakeEvolvable(
            network,
            torch.randn(1, 3, 1, 24, 24),
            device=device,
        )
        evolvable_cnn.change_cnn_kernel()
        assert len(evolvable_cnn.kernel_size) == 2
        assert len(evolvable_cnn.kernel_size[-1]) == 3

    @pytest.mark.gpu
    def test_change_kernel_multi_two_arg(self, two_arg_cnn, device):
        evolvable_cnn = MakeEvolvable(
            two_arg_cnn,
            torch.randn(*_TWO_ARG_INPUT_SHAPE),
            torch.randn(*_TWO_ARG_SECONDARY_SHAPE),
            device=device,
        )
        while evolvable_cnn.kernel_size == [(2, 3, 3), (1, 3, 3)]:
            evolvable_cnn.change_cnn_kernel()

        assert evolvable_cnn.kernel_size != [
            (2, 3, 3),
            (1, 3, 3),
        ], evolvable_cnn.kernel_size


######### Test recreate_nets #########
class TestMakeEvolvableRecreateNetwork:
    @pytest.mark.gpu
    def test_recreate_nets_parameters_preserved(self, simple_mlp, device):
        input_tensor = torch.randn(1, 10)
        evolvable_network = MakeEvolvable(simple_mlp, input_tensor, device=device)

        feature_net = evolvable_network.feature_net
        feature_net_dict = dict(feature_net.named_parameters())

        # Modify the architecture
        evolvable_network.hidden_size += [evolvable_network.hidden_size[-1]]

        evolvable_network.recreate_network()
        new_feature_net = evolvable_network.feature_net

        for key, param in new_feature_net.named_parameters():
            if key in feature_net_dict:
                assert torch.equal(param, feature_net_dict[key])

    @pytest.mark.gpu
    def test_recreate_nets_parameters_preserved_rainbow(self, simple_mlp, device):
        input_tensor = torch.randn(1, 10)
        support = torch.linspace(-200, 200, 51).to(device)
        evolvable_network = MakeEvolvable(
            simple_mlp,
            input_tensor,
            support=support,
            rainbow=True,
            device=device,
        )

        feature_net = evolvable_network.feature_net
        feature_net_dict = dict(feature_net.named_parameters())

        value_net = evolvable_network.value_net
        value_net_dict = dict(value_net.named_parameters())

        advantage_net = evolvable_network.advantage_net
        advantage_net_dict = dict(advantage_net.named_parameters())

        # Modify the architecture
        evolvable_network.hidden_size += [evolvable_network.hidden_size[-1]]

        evolvable_network.recreate_network()
        new_feature_net = evolvable_network.feature_net
        new_value_net = evolvable_network.value_net
        new_advantage_net = evolvable_network.advantage_net

        for key, param in new_feature_net.named_parameters():
            if key in feature_net_dict:
                assert torch.equal(param, feature_net_dict[key])

        for key, param in new_value_net.named_parameters():
            if key in value_net_dict:
                assert torch.equal(param, value_net_dict[key])

        for key, param in new_advantage_net.named_parameters():
            if key in advantage_net_dict:
                assert torch.equal(param, advantage_net_dict[key])

    @pytest.mark.gpu
    def test_recreate_nets_parameters_shrink_preserved(self, device):
        network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        input_tensor = torch.randn(1, 4)
        evolvable_network = MakeEvolvable(network, input_tensor, device=device)

        feature_net = evolvable_network.feature_net
        feature_net_dict = dict(feature_net.named_parameters())

        # Modify the architecture
        evolvable_network.hidden_size = evolvable_network.hidden_size[:-1]
        evolvable_network.recreate_network(shrink_params=True)
        new_feature_net = evolvable_network.feature_net

        for key, param in new_feature_net.named_parameters():
            if key in feature_net_dict:
                torch.testing.assert_close(param, feature_net_dict[key])


######### Test clone #########
class TestMakeEvolvableClone:
    # The clone() method successfully creates a deep copy of the model.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "network, input_tensor, secondary_input_tensor",
        [
            ("simple_mlp", torch.randn(1, 10), None),
            ("simple_cnn", torch.randn(1, 3, 32, 32), None),
            (
                "two_arg_cnn",
                torch.randn(*_TWO_ARG_INPUT_SHAPE),
                torch.randn(*_TWO_ARG_SECONDARY_SHAPE),
            ),
        ],
    )
    def test_clone_method_with_equal_state_dicts(
        self,
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
        assert_state_dicts_equal(
            evolvable_network.state_dict(), clone_network.state_dict()
        )


@pytest.fixture
def mlp_net():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Tanh(),
    )


@pytest.fixture
def cnn_net():
    # 32x32 input -> two MaxPool(2) -> 8x8 -> 32*8*8 = 2048.
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


class TestMakeEvolvableChangeActivation:
    @pytest.mark.gpu
    def test_change_activation_output_false(self, mlp_net, device):
        evolvable = MakeEvolvable(mlp_net, torch.randn(1, 10), device=device)
        evolvable.change_activation("LeakyReLU", output=False)
        assert evolvable.mlp_activation == "LeakyReLU"

    @pytest.mark.gpu
    def test_change_activation_output_true(self, mlp_net, device):
        evolvable = MakeEvolvable(mlp_net, torch.randn(1, 10), device=device)
        evolvable.change_activation("LeakyReLU", output=True)
        assert evolvable.mlp_activation == "LeakyReLU"
        assert evolvable.mlp_output_activation == "LeakyReLU"


class TestMakeEvolvableInitWeightsGaussian:
    @pytest.mark.gpu
    def test_init_weights_gaussian_cnn(self, cnn_net, device):
        evolvable = MakeEvolvable(cnn_net, torch.randn(1, 3, 32, 32), device=device)
        evolvable.init_weights_gaussian(std_coeff=2.0, output_coeff=1.0)
        assert evolvable.feature_net is not None
        assert evolvable.value_net is not None


class TestMakeEvolvableCalcStrideSizeRanges:
    @pytest.mark.gpu
    def test_calc_stride_size_ranges_conv3d(self, device):
        # input (1,3,1,24,24) -> conv (1,22,22) -> conv (1,20,20) -> 32*1*20*20 = 12800.
        net = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 1 * 20 * 20, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        input_tensor = torch.randn(1, 3, 1, 24, 24)
        evolvable = MakeEvolvable(
            net,
            input_tensor,
            device=device,
            min_cnn_hidden_layers=1,
            max_cnn_hidden_layers=4,
        )
        strides = evolvable.calc_stride_size_ranges()
        assert len(strides) == len(evolvable.channel_size)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in strides)


class TestMakeEvolvableMutations:
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "method",
        ["add_mlp_layer", "remove_mlp_layer", "add_mlp_node", "remove_mlp_node"],
    )
    def test_make_evolvable_mlp_mutations(self, mlp_net, device, method):
        evolvable = MakeEvolvable(mlp_net, torch.randn(1, 10), device=device)
        getattr(evolvable, method)()


class TestMakeEvolvableGetOutputDense:
    @pytest.mark.gpu
    def test_make_evolvable_get_output_dense(self, mlp_net, device):
        evolvable = MakeEvolvable(mlp_net, torch.randn(1, 10), device=device)
        layer = evolvable.get_output_dense()
        assert isinstance(layer, nn.Linear)
