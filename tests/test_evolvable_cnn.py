import torch
import torch.nn as nn
import pytest 
import copy

from agilerl.networks.evolvable_cnn import EvolvableCNN

######### Define fixtures #########
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

######### Test instantiation #########

@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([1, 16, 16], [32], [(3, 3)], [(1, 1)], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_instantiation_without_errors(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       device=device)
    assert isinstance(evolvable_cnn, EvolvableCNN)


@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3, 3], [1], [128], 10),
            ([1, 16, 16], [32], [3], [1, 1], [128], 10),
            ([1, 16, 16], [32], [(3, 3)], [(1, 1)], [], 10),
            ([1, 16, 16], [32], [(3, 3)], [(1, 1)], [128], 0),
            ([3, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_incorrect_instantiation(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    with pytest.raises(AssertionError):
        evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                                     channel_size=channel_size, 
                                     kernel_size=kernel_size, 
                                     stride_size=stride_size, 
                                     hidden_size=hidden_size, 
                                     num_actions=num_actions,
                                     device=device)

@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, n_agents",
        [
        ([1, 16, 16], [3, 32], [(1, 3, 3), (1, 3, 3)], [2, 2], [32, 32], 10, 2)
        ]
)
def test_instantiation_for_multi_agents(input_shape, channel_size, kernel_size, stride_size, 
                                        hidden_size, num_actions, n_agents, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                                     channel_size=channel_size, 
                                     kernel_size=kernel_size, 
                                     stride_size=stride_size, 
                                     hidden_size=hidden_size, 
                                     num_actions=num_actions,
                                     n_agents=n_agents,
                                     multi=True,
                                     critic=True,
                                     device=device)
    assert isinstance(evolvable_cnn, EvolvableCNN)

@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, multi, n_agents",
        [
        ([1, 16, 16], [3, 32], [(1, 3, 3), (1, 3, 3)], [2, 2], [32, 32], 10, True, None),
        ([1, 16, 16], [3, 32], [(1, 3, 3), (1, 3, 3)], [2, 2], [32, 32], 10, False, 2)
        ]
)
def test_incorrect_instantiation_for_multi_agents(input_shape, channel_size, kernel_size, stride_size, 
                                        hidden_size, num_actions, multi, n_agents, device):
    with pytest.raises(AssertionError):
        evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                                    channel_size=channel_size, 
                                    kernel_size=kernel_size, 
                                    stride_size=stride_size, 
                                    hidden_size=hidden_size, 
                                    num_actions=num_actions,
                                    n_agents=n_agents,
                                    multi=multi,
                                    critic=True,
                                    device=device)

######### Test get_activation #########
def test_returns_correct_activation_function_for_all_supported_names(device):
    activation_names = [
        "Tanh",
        "Identity",
        "ReLU",
        "ELU",
        "Softsign",
        "Sigmoid",
        "GumbelSoftmax",
        "Softplus",
        "Softmax",
        "LeakyReLU",
        "PReLU",
        "GELU",
    ]
    for name in activation_names:
        activation = EvolvableCNN([1, 16, 16], 
                                  [32], 
                                  [3], 
                                  [1], 
                                  [128], 
                                  10,
                                  device=device).get_activation(name)
        assert isinstance(activation, nn.Module)

######### Test forward #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, output_shape",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10, (1, 10)),
            ([1, 16, 16], [32], [(3, 3)], [(1, 1)], [128], 10 , (1, 10)),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1, (1, 1)),
        ]
)
def test_forward(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, 
                 output_shape, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       device=device)
    input_tensor = torch.randn(input_shape).unsqueeze(0) # To add in a batch size dimension
    input_tensor = input_tensor.to(device)
    print(input_tensor.shape)
    output = evolvable_cnn.forward(input_tensor)
    print(output)
    assert output.shape == output_shape

@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, \
        hidden_size, num_actions, n_agents, output_shape",
        [
            ([1, 16, 16], [3, 32], [(1, 3, 3), (1, 3, 3)], [2, 2], [32, 32], 10, 2, (1, 10))
        ]
)
def test_forward_multi(input_shape, channel_size, kernel_size, stride_size, hidden_size, 
                       num_actions, n_agents, output_shape, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                                     channel_size=channel_size, 
                                     kernel_size=kernel_size, 
                                     stride_size=stride_size, 
                                     hidden_size=hidden_size, 
                                     num_actions=num_actions,
                                     n_agents=n_agents,
                                     multi=True,
                                     critic=False,
                                     device=device)
    input_tensor = torch.randn(1, *input_shape).unsqueeze(2).to(device)
    with torch.no_grad():
        output = evolvable_cnn.forward(input_tensor)
    assert output.shape == output_shape

@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, \
        hidden_size, num_actions, n_agents, output_shape, secondary_tensor",
        [
            ([1, 16, 16], [3, 32], [(1, 3, 3), (1, 3, 3)], [2, 2], [32, 32], 2, 2, (1, 1), (1, 2))
        ]
)
def test_forward_multi_critic(input_shape, channel_size, kernel_size, stride_size, hidden_size, 
                       num_actions, n_agents, output_shape, secondary_tensor, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                                     channel_size=channel_size, 
                                     kernel_size=kernel_size, 
                                     stride_size=stride_size, 
                                     hidden_size=hidden_size, 
                                     num_actions=num_actions,
                                     n_agents=n_agents,
                                     multi=True,
                                     critic=True,
                                     device=device)
    input_tensor = torch.randn(1, *input_shape).unsqueeze(2).to(device)
    secondary_tensor = torch.randn(secondary_tensor).to(device)
    with torch.no_grad():
        output = evolvable_cnn.forward(input_tensor, secondary_tensor)
    assert output.shape == output_shape

######### Test add_mlp_layer #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([1, 16, 16], [32], [(3, 3)], [(1, 1)], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_add_mlp_layer(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       max_hidden_layers=5,
                       device=device)

    initial_hidden_size = len(evolvable_cnn.hidden_size)
    initial_net = evolvable_cnn.value_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_cnn.add_mlp_layer()
    new_net = evolvable_cnn.value_net
    if initial_hidden_size < 10:
        assert len(evolvable_cnn.hidden_size) == initial_hidden_size + 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_cnn.hidden_size) == initial_hidden_size

######### Test remove_mlp_layer #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_remove_mlp_layer(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       max_hidden_layers=5,
                       min_hidden_layers=1,
                       device=device)

    initial_hidden_size = len(evolvable_cnn.hidden_size)
    initial_net = evolvable_cnn.value_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_cnn.remove_mlp_layer()
    new_net = evolvable_cnn.value_net
    if initial_hidden_size > 1:
        assert len(evolvable_cnn.hidden_size) == initial_hidden_size - 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys() and param.shape == initial_net_dict[key].shape:
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_cnn.hidden_size) == initial_hidden_size

######### Test add_mlp_node #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_add_nodes(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       device=device)
    original_hidden_size = copy.deepcopy(evolvable_cnn.hidden_size)
    result = evolvable_cnn.add_mlp_node()
    hidden_layer = result["hidden_layer"]
    numb_new_nodes = result["numb_new_nodes"]
    assert evolvable_cnn.hidden_size[hidden_layer] == original_hidden_size[hidden_layer] + numb_new_nodes

######### Test remove_mlp_node #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_remove_nodes(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       min_mlp_nodes=4,
                       min_hidden_layers=1,
                       device=device)
    original_hidden_size = copy.deepcopy(evolvable_cnn.hidden_size)
    numb_new_nodes = 4
    result = evolvable_cnn.remove_mlp_node(numb_new_nodes=numb_new_nodes)
    hidden_layer = result["hidden_layer"]
    assert evolvable_cnn.hidden_size[hidden_layer] == original_hidden_size[hidden_layer] - numb_new_nodes

######### Test add_cnn_layer #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_add_cnn_layer(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       device=device)
    initial_channel_num = len(evolvable_cnn.channel_size)
    initial_net = evolvable_cnn.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_cnn.add_cnn_layer()
    new_net = evolvable_cnn.feature_net
    if initial_channel_num < 6:
        assert len(evolvable_cnn.channel_size) == initial_channel_num + 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_cnn.channel_size) == initial_channel_num
    
######### Test remove_cnn_layer #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_remove_cnn_layer(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       device=device)
    initial_channel_num = len(evolvable_cnn.channel_size)
    initial_net = evolvable_cnn.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_cnn.remove_cnn_layer()
    new_net = evolvable_cnn.feature_net
    if initial_channel_num > 1:
        assert len(evolvable_cnn.channel_size) == initial_channel_num - 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys() and param.shape == initial_net_dict[key].shape:
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_cnn.channel_size) == initial_channel_num

######### Test add_cnn_channel #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_add_channels(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       device=device)
    original_channel_size = copy.deepcopy(evolvable_cnn.channel_size)
    result = evolvable_cnn.add_cnn_channel()
    hidden_layer = result["hidden_layer"]
    numb_new_channels = result["numb_new_channels"]
    assert evolvable_cnn.channel_size[hidden_layer] == original_channel_size[hidden_layer] + numb_new_channels

######### Test remove_cnn_channel #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_remove_channels(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       min_channel_size=4,
                       device=device)
    original_channel_size = copy.deepcopy(evolvable_cnn.channel_size)
    numb_new_channels = 2
    result = evolvable_cnn.remove_cnn_channel(numb_new_channels=numb_new_channels)
    hidden_layer = result["hidden_layer"]
    assert evolvable_cnn.channel_size[hidden_layer] == original_channel_size[hidden_layer] - numb_new_channels


######### Test clone #########
@pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions",
        [
            ([1, 16, 16], [32], [3], [1], [128], 10),
            ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
        ]
)
def test_clone_instance(input_shape, channel_size, kernel_size, stride_size, hidden_size, num_actions, device):
    evolvable_cnn = EvolvableCNN(input_shape=input_shape, 
                       channel_size=channel_size, 
                       kernel_size=kernel_size, 
                       stride_size=stride_size, 
                       hidden_size=hidden_size, 
                       num_actions=num_actions,
                       device=device)
    original_feature_net_dict = dict(evolvable_cnn.feature_net.named_parameters())
    original_value_net_dict = dict(evolvable_cnn.value_net.named_parameters())
    clone = evolvable_cnn.clone()
    clone_feature_net = clone.feature_net
    clone_value_net = clone.value_net
    assert isinstance(clone, EvolvableCNN)
    assert clone.init_dict == evolvable_cnn.init_dict
    assert str(clone.state_dict()) == str(evolvable_cnn.state_dict())
    for key, param in clone_feature_net.named_parameters():
            torch.testing.assert_close(param, original_feature_net_dict[key])
    for key, param in clone_value_net.named_parameters():
            torch.testing.assert_close(param, original_value_net_dict[key])