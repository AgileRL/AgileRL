from typing import Union
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium.spaces import Dict, Tuple

from agilerl.modules.multi_input import EvolvableMultiInput
from tests.helper_functions import generate_dict_or_tuple_space

DictOrTupleSpace = Union[Dict, Tuple]

######### Define fixtures #########
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

######### Test instantiation #########

@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [32], [(3, 3)], [(1, 1)], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
    ],
)
def test_instantiation_without_errors(
    observation_space: DictOrTupleSpace,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        device=device,
    )
    assert isinstance(evolvable_composed, EvolvableMultiInput)


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3, 3], [1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1, 1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [32], [(3, 3)], [(1, 1)], [], 10),
        (generate_dict_or_tuple_space(2, 3), [32], [(3, 3)], [(1, 1)], [128], 0),
        (generate_dict_or_tuple_space(2, 3, image_shape=(3, 128)), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
    ],
)
def test_incorrect_instantiation(
    observation_space: DictOrTupleSpace,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    with pytest.raises(AssertionError):
        EvolvableMultiInput(
            observation_space=observation_space,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            hidden_size=hidden_size,
            num_outputs=num_outputs,
            device=device,
        )


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, n_agents",
    [(generate_dict_or_tuple_space(2, 3), [3, 32], [3, 3], [2, 2], [32, 32], 10, 2)],
)
def test_instantiation_for_multi_agents(
    observation_space: DictOrTupleSpace,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    n_agents,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        n_agents=n_agents,
        critic=True,
        device=device,
    )
    assert isinstance(evolvable_composed, EvolvableMultiInput)


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, n_agents",
    [
        (
            generate_dict_or_tuple_space(2, 3),
            [3, 32],
            [3, 3],
            [2, 2],
            [32, 32],
            10,
            2.5,
        ),
        (generate_dict_or_tuple_space(2, 3), [3, 32], [3, 3], [2, 2], [32, 32], 10, 0),
    ],
)
def test_incorrect_instantiation_for_multi_agents(
    observation_space: DictOrTupleSpace,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    n_agents,
    device,
):
    with pytest.raises(AssertionError):
        EvolvableMultiInput(
            observation_space=observation_space,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            hidden_size=hidden_size,
            num_outputs=num_outputs,
            n_agents=n_agents,
            critic=True,
            device=device,
        )


def test_rainbow_instantiation(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[64, 64],
        num_outputs=10,
        layer_norm=True,
        rainbow=True,
        device=device,
    )
    assert isinstance(evolvable_composed, EvolvableMultiInput)

######### Test forward #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, output_shape",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10, (1, 10)),
        (generate_dict_or_tuple_space(2, 3), [32], [(3, 3)], [(1, 1)], [128], 10, (1, 10)),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1, (1, 1)),
    ],
)
def test_forward(
    observation_space: DictOrTupleSpace,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    output_shape,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        device=device,
    )

    sample = observation_space.sample()
    if isinstance(sample, dict):
        input_array = {k: np.expand_dims(sample[k], 0) for k in sample}
        input_tensor = {k: torch.tensor(sample[k]).unsqueeze(0).to(device) for k in sample}
    else:
        input_tensor = tuple([torch.tensor(comp).unsqueeze(0).to(device) for comp in sample])

    output = evolvable_composed.forward(input_tensor)
    output_array = evolvable_composed.forward(input_array)
    assert output.shape == output_shape
    assert output_array.shape == output_shape


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, \
        hidden_size, num_outputs, n_agents, output_shape",
    [(generate_dict_or_tuple_space(2, 3), [3, 32], [3, 3], [2, 2], [32, 32], 10, 2, (1, 10))],
)
def test_forward_multi(
    observation_space: DictOrTupleSpace,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    n_agents,
    output_shape,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        n_agents=n_agents,
        critic=False,
        device=device,
    )

    sample = observation_space.sample()
    if isinstance(sample, dict):
        input_tensor = {k: torch.tensor(sample[k]).unsqueeze(0).to(device) for k in sample}
    else:
        input_tensor = tuple([torch.tensor(comp).unsqueeze(0).to(device) for comp in sample])

    with torch.no_grad():
        output = evolvable_composed.forward(input_tensor)
    assert output.shape == output_shape


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, \
        hidden_size, num_outputs, n_agents, output_shape, secondary_tensor",
    [
        (
            generate_dict_or_tuple_space(2, 3),
            [3, 32],
            [3, 3],
            [2, 2],
            [32, 32],
            2,
            2,
            (1, 1),
            (1, 2),
        )
    ],
)
def test_forward_multi_critic(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    n_agents,
    output_shape,
    secondary_tensor,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        n_agents=n_agents,
        critic=True,
        device=device,
    )
    input_tensor = (
        torch.randn(1, *observation_space)
        .unsqueeze(2)
        .to(device)
        .repeat(1, 1, n_agents, 1, 1)
    )
    secondary_tensor = torch.randn(secondary_tensor).to(device)
    with torch.no_grad():
        output = evolvable_composed.forward(input_tensor, secondary_tensor)
    assert output.shape == output_shape


def test_forward_rainbow(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[64, 64],
        num_outputs=10,
        rainbow=True,
        layer_norm=True,
        support=torch.linspace(0.0, 200.0, 51).to(device),
        device=device,
    )
    input_tensor = torch.randn(1, 1, 16, 16).to(device)
    with torch.no_grad():
        output = evolvable_composed.forward(input_tensor)
    assert output.shape == (1, 10)

######### Test add_mlp_layer #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [32], [(3, 3)], [(1, 1)], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
    ],
)
def test_add_mlp_layer(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        max_hidden_layers=5,
        device=device,
    )

    initial_hidden_size = len(evolvable_composed.hidden_size)
    initial_net = evolvable_composed.value_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_composed.add_mlp_layer()
    new_net = evolvable_composed.value_net
    if initial_hidden_size < 10:
        assert len(evolvable_composed.hidden_size) == initial_hidden_size + 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_composed.hidden_size) == initial_hidden_size


def test_add_mlp_layer_else_statement(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[32, 32],
        num_outputs=4,
        max_hidden_layers=2,
        device=device,
    )
    initial_hidden_size = len(evolvable_composed.hidden_size)
    evolvable_composed.add_mlp_layer()
    assert initial_hidden_size == len(evolvable_composed.hidden_size)


######### Test remove_mlp_layer #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
    ],
)
def test_remove_mlp_layer(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        max_hidden_layers=5,
        min_hidden_layers=1,
        device=device,
    )

    initial_hidden_size = len(evolvable_composed.hidden_size)
    initial_net = evolvable_composed.value_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_composed.remove_mlp_layer()
    new_net = evolvable_composed.value_net
    if initial_hidden_size > 1:
        assert len(evolvable_composed.hidden_size) == initial_hidden_size - 1
        for key, param in new_net.named_parameters():
            if (
                key in initial_net_dict.keys()
                and param.shape == initial_net_dict[key].shape
            ):
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_composed.hidden_size) == initial_hidden_size


######### Test add_mlp_node #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, layer_index",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10, None),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1, 1),
    ],
)
def test_add_nodes(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    layer_index,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        device=device,
    )
    original_hidden_size = copy.deepcopy(evolvable_composed.hidden_size)
    layer = layer_index
    result = evolvable_composed.add_mlp_node(hidden_layer=layer)
    hidden_layer = result["hidden_layer"]
    numb_new_nodes = result["numb_new_nodes"]
    assert (
        evolvable_composed.hidden_size[hidden_layer]
        == original_hidden_size[hidden_layer] + numb_new_nodes
    )


######### Test remove_mlp_node #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, layer_index, numb_new_nodes",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10, 1, None),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1, None, 4),
    ],
)
def test_remove_nodes(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    layer_index,
    numb_new_nodes,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        min_mlp_nodes=4,
        min_hidden_layers=1,
        device=device,
    )
    layer = layer_index
    original_hidden_size = copy.deepcopy(evolvable_composed.hidden_size)
    result = evolvable_composed.remove_mlp_node(
        numb_new_nodes=numb_new_nodes, hidden_layer=layer
    )
    hidden_layer = result["hidden_layer"]
    numb_new_nodes = result["numb_new_nodes"]
    assert (
        evolvable_composed.hidden_size[hidden_layer]
        == original_hidden_size[hidden_layer] - numb_new_nodes
    )


######### Test add_cnn_layer #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
    ],
)
def test_add_cnn_layer_simple(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        device=device,
    )
    initial_channel_num = len(evolvable_composed.channel_size)
    initial_net = evolvable_composed.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_composed.add_cnn_layer()
    new_net = evolvable_composed.feature_net
    if initial_channel_num < 6:
        assert len(evolvable_composed.channel_size) == initial_channel_num + 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_composed.channel_size) == initial_channel_num


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32, 32], [5, 5], [2, 2], [128], 10),  # invalid output size
        (
            [1, 164, 164],
            [8, 8, 8, 8, 8, 8],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 1, 1, 1, 1],
            [128],
            10,
        ),  # exceeds max layer limit
    ],
)
def test_add_cnn_layer_no_layer_added(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        device=device,
    )
    evolvable_composed.add_cnn_layer()
    assert len(channel_size) == len(evolvable_composed.channel_size)


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        ([3, 84, 84], [8, 8], [2, 2], [2, 2], [128], 10),  # exceeds max-layer limit
    ],
)
def test_add_and_remove_multiple_cnn_layers(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        device=device,
    )
    # Keep adding layers until we reach max or it is infeasible
    for _ in range(evolvable_composed.max_cnn_hidden_layers):
        evolvable_composed.add_cnn_layer()

    # Do a forward pass to ensure network parameter validity
    output = evolvable_composed(torch.ones(1, 3, 84, 84).to(device))
    assert output.squeeze().shape[0] == num_outputs
    assert len(evolvable_composed.stride_size) == len(evolvable_composed.channel_size)
    assert len(evolvable_composed.kernel_size) == len(evolvable_composed.channel_size)

    # Remove as many layers as possible
    for _ in evolvable_composed.channel_size:
        evolvable_composed.remove_cnn_layer()

    assert len(evolvable_composed.channel_size) == evolvable_composed.min_cnn_hidden_layers

    # Do a forward pass to ensure network parameter validity
    output = evolvable_composed(torch.ones(1, 3, 84, 84).to(device))
    assert output.squeeze().shape[0] == num_outputs
    assert len(evolvable_composed.stride_size) == len(evolvable_composed.channel_size)
    assert len(evolvable_composed.kernel_size) == len(evolvable_composed.channel_size)


def test_add_cnn_layer_else_statement(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[32, 32],
        num_outputs=4,
        max_cnn_hidden_layers=2,
        device=device,
    )
    original_num_hidden_layers = copy.deepcopy(evolvable_composed.channel_size)
    evolvable_composed.add_cnn_layer()
    assert len(original_num_hidden_layers) == len(evolvable_composed.channel_size)


def test_add_cnn_layer_rainbow(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[64, 64],
        num_outputs=10,
        rainbow=True,
        layer_norm=True,
        support=torch.linspace(0.0, 200.0, 51).to(device),
        device=device,
    )
    initial_channel_num = len(evolvable_composed.channel_size)
    initial_net = evolvable_composed.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_composed.add_cnn_layer()
    new_net = evolvable_composed.feature_net
    if initial_channel_num < 6:
        assert len(evolvable_composed.channel_size) == initial_channel_num + 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_composed.channel_size) == initial_channel_num


######### Test remove_cnn_layer #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
    ],
)
def test_remove_cnn_layer(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        device=device,
    )
    initial_channel_num = len(evolvable_composed.channel_size)
    initial_net = evolvable_composed.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_composed.remove_cnn_layer()
    new_net = evolvable_composed.feature_net
    if initial_channel_num > 1:
        assert len(evolvable_composed.channel_size) == initial_channel_num - 1
        for key, param in new_net.named_parameters():
            if (
                key in initial_net_dict.keys()
                and param.shape == initial_net_dict[key].shape
            ):
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_composed.channel_size) == initial_channel_num


def test_remove_cnn_layer_rainbow(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[64, 64],
        num_outputs=10,
        rainbow=True,
        layer_norm=True,
        support=torch.linspace(0.0, 200.0, 51).to(device),
        device=device,
    )
    initial_channel_num = len(evolvable_composed.channel_size)
    initial_net = evolvable_composed.feature_net
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_composed.remove_cnn_layer()
    new_net = evolvable_composed.feature_net
    if initial_channel_num < 6:
        assert len(evolvable_composed.channel_size) == initial_channel_num - 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_composed.channel_size) == initial_channel_num


######### Test add_cnn_channel #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, layer_index",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10, 0),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1, None),
    ],
)
def test_add_channels(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    layer_index,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        device=device,
    )
    original_channel_size = copy.deepcopy(evolvable_composed.channel_size)
    result = evolvable_composed.add_cnn_channel(hidden_layer=layer_index)
    hidden_layer = result["hidden_layer"]
    numb_new_channels = result["numb_new_channels"]
    assert (
        evolvable_composed.channel_size[hidden_layer]
        == original_channel_size[hidden_layer] + numb_new_channels
    )


######### Test remove_cnn_channel #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, layer_index, numb_new_channels",
    [
        (generate_dict_or_tuple_space(2, 3), [256], [3], [1], [128], 10, None, None),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1, 0, 2),
    ],
)
def test_remove_channels(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    layer_index,
    numb_new_channels,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        min_channel_size=4,
        device=device,
    )
    original_channel_size = copy.deepcopy(evolvable_composed.channel_size)
    result = evolvable_composed.remove_cnn_channel(
        numb_new_channels=numb_new_channels, hidden_layer=layer_index
    )
    hidden_layer = result["hidden_layer"]
    numb_new_channels = result["numb_new_channels"]
    assert (
        evolvable_composed.channel_size[hidden_layer]
        == original_channel_size[hidden_layer] - numb_new_channels
    )


######### Test change_cnn_kernel #########
def test_change_cnn_kernel(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[32, 32],
        num_outputs=4,
        device=device,
    )
    # Change kernel size
    evolvable_composed.change_cnn_kernel()

    while evolvable_composed.kernel_size == [(3, 3), (3, 3)]:
        evolvable_composed.change_cnn_kernel()

    # Check if kernel size has changed
    assert evolvable_composed.kernel_size != [
        (3, 3),
        (3, 3),
    ], evolvable_composed.kernel_size


def test_change_kernel_size(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[32, 32],
        num_outputs=4,
        device=device,
    )

    for _ in range(100):
        # Change kernel size and ensure we can make a valid forward pass
        evolvable_composed.change_cnn_kernel()
        output = evolvable_composed(torch.ones(1, 16, 16).to(device))
        assert output.squeeze().shape[0] == 4  # (num actions)


def test_change_cnn_kernel_else_statement(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[32, 32],
        num_outputs=4,
        device=device,
    )

    # Change kernel size
    evolvable_composed.change_cnn_kernel()

    while evolvable_composed.kernel_size == [3, 3]:
        evolvable_composed.change_cnn_kernel()

    # Check if kernel size has changed
    assert evolvable_composed.kernel_size != [3, 3]


@pytest.mark.parametrize("critic", [(True), (False)])
def test_change_cnn_kernel_multi(critic, device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[32, 32],
        num_outputs=4,
        critic=critic,
        device=device,
    )

    # Change kernel size
    evolvable_composed.change_cnn_kernel()

    while evolvable_composed.kernel_size == [3, 3]:
        evolvable_composed.change_cnn_kernel()

    # Check if kernel size has changed
    assert evolvable_composed.kernel_size != [
        3,
        3,
    ], evolvable_composed.kernel_size


def test_change_cnn_kernel_multi_else_statement(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32],
        kernel_size=[3],
        stride_size=[1],
        hidden_size=[32, 32],
        num_outputs=4,
        device=device,
    )

    # Change kernel size
    while evolvable_composed.kernel_size == [3]:
        evolvable_composed.change_cnn_kernel()

    # Check if kernel size has changed
    assert len(evolvable_composed.kernel_size) == 2


######### Test clone #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [8, 8, 8], [2, 2, 2], [2, 2, 2], [32, 32, 32], 1),
    ],
)
def test_clone_instance(
    observation_space,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        device=device,
    )
    original_feature_net_dict = dict(evolvable_composed.feature_net.named_parameters())
    original_value_net_dict = dict(evolvable_composed.value_net.named_parameters())
    clone = evolvable_composed.clone()
    clone_feature_net = clone.feature_net
    clone_value_net = clone.value_net
    assert isinstance(clone, EvolvableMultiInput)
    assert clone.init_dict == evolvable_composed.init_dict
    assert str(clone.state_dict()) == str(evolvable_composed.state_dict())
    for key, param in clone_feature_net.named_parameters():
        torch.testing.assert_close(param, original_feature_net_dict[key])
    for key, param in clone_value_net.named_parameters():
        torch.testing.assert_close(param, original_value_net_dict[key])
