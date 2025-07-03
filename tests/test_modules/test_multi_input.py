import copy
from dataclasses import asdict
from typing import Union

import numpy as np
import pytest
import torch
from gymnasium.spaces import Dict, Tuple

from agilerl.modules.configs import CnnNetConfig, MlpNetConfig
from agilerl.modules.multi_input import EvolvableMultiInput
from tests.helper_functions import (
    assert_state_dicts_equal,
    generate_dict_or_tuple_space,
)

DictOrTupleSpace = Union[Dict, Tuple]


@pytest.fixture(scope="module")
def default_cnn_config():
    return asdict(
        CnnNetConfig(
            channel_size=[32],
            kernel_size=[3],
            stride_size=[1],
            output_activation="ReLU",
        )
    )


@pytest.fixture(scope="module")
def multiagent_cnn_config(
    device, image_shape: tuple = (3, 32, 32), sample_input: str = "default"
):
    if sample_input == "default":
        sample_input = torch.randn(1, *image_shape).unsqueeze(2).to(device)
    else:
        sample_input = None

    return asdict(
        CnnNetConfig(
            channel_size=[32],
            kernel_size=[3],
            stride_size=[1],
            output_activation="ReLU",
            block_type="Conv3d",
            sample_input=sample_input,
        )
    )


@pytest.fixture(scope="module")
def default_mlp_config():
    return asdict(
        MlpNetConfig(
            hidden_size=[64],
            output_activation="ReLU",
        )
    )


######### Test instantiation #########
@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_instantiation_without_errors(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    device,
    default_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        mlp_config=default_mlp_config,
        device=device,
    )
    assert isinstance(evolvable_composed, EvolvableMultiInput)


@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 0),  # Invalid num_outputs
        (
            generate_dict_or_tuple_space(2, 3, image_shape=(3, 3, 128)),
            0,
        ),  # Invalid latent dim
    ],
)
def test_incorrect_instantiation(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    device,
    default_cnn_config,
    default_mlp_config,
):
    with pytest.raises((AssertionError, ValueError)):
        EvolvableMultiInput(
            observation_space=observation_space,
            num_outputs=num_outputs,
            cnn_config=default_cnn_config,
            mlp_config=default_mlp_config,
            device=device,
        )


@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3, dict_space=True), 10),
    ],
)
def test_instantiation_for_multi_agents(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    device,
    multiagent_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=multiagent_cnn_config,
        mlp_config=default_mlp_config,
        device=device,
    )
    assert isinstance(evolvable_composed, EvolvableMultiInput)


@pytest.mark.parametrize(
    "observation_space, num_outputs, sample_input",
    [
        (generate_dict_or_tuple_space(2, 3, dict_space=False), 10, None),
        (generate_dict_or_tuple_space(2, 3, dict_space=True), 10, None),
    ],
)
def test_incorrect_instantiation_for_multi_agents(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    sample_input,
    device,
    multiagent_cnn_config,
    default_mlp_config,
):
    with pytest.raises(TypeError):
        EvolvableMultiInput(
            observation_space=observation_space,
            num_outputs=num_outputs,
            cnn_config=multiagent_cnn_config(sample_input=sample_input),
            mlp_config=default_mlp_config,
            device=device,
        )


######### Test forward #########
@pytest.mark.parametrize(
    "observation_space, num_outputs, output_shape",
    [
        (generate_dict_or_tuple_space(2, 3), 10, (1, 10)),
        (generate_dict_or_tuple_space(2, 3), 1, (1, 1)),
    ],
)
def test_forward(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    output_shape: tuple,
    device,
    default_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        mlp_config=default_mlp_config,
        device=device,
    )

    sample = observation_space.sample()
    if isinstance(sample, dict):
        input_array = {k: np.expand_dims(sample[k], 0) for k in sample}
        input_tensor = {
            k: torch.tensor(sample[k]).unsqueeze(0).to(device) for k in sample
        }
    else:
        input_array = tuple([np.expand_dims(comp, 0) for comp in sample])
        input_tensor = tuple(
            [torch.tensor(comp).unsqueeze(0).to(device) for comp in sample]
        )

    output = evolvable_composed.forward(input_tensor)
    output_array = evolvable_composed.forward(input_array)
    assert output.shape == output_shape
    assert output_array.shape == output_shape


@pytest.mark.parametrize(
    "observation_space, num_outputs, output_shape",
    [
        (generate_dict_or_tuple_space(2, 3, dict_space=True), 10, (1, 10)),
    ],
)
def test_forward_multi(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    output_shape: tuple,
    device,
    multiagent_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=multiagent_cnn_config,
        mlp_config=default_mlp_config,
        device=device,
    )

    sample = observation_space.sample()
    if isinstance(sample, dict):
        input_tensor = {
            k: torch.tensor(sample[k]).unsqueeze(0).to(device) for k in sample
        }
    else:
        input_tensor = tuple(
            [torch.tensor(comp).unsqueeze(0).to(device) for comp in sample]
        )

    with torch.no_grad():
        output = evolvable_composed.forward(input_tensor)
    assert output.shape == output_shape


######### Test add_mlp_layer #########
@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_add_mlp_layer(
    observation_space,
    num_outputs,
    device,
    default_mlp_config,
):
    config = default_mlp_config.copy()
    config["hidden_size"] = [64]
    config["max_hidden_layers"] = 5

    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        mlp_config=config,
        vector_space_mlp=True,
        device=device,
    )

    initial_hidden_size = len(evolvable_composed.mlp_config["hidden_size"])
    initial_net = evolvable_composed.feature_net["vector_mlp"]
    initial_net_dict = dict(initial_net.named_parameters())
    getattr(evolvable_composed, "feature_net.vector_mlp.add_layer")()
    new_net = evolvable_composed.feature_net["vector_mlp"]
    if initial_hidden_size < 10:
        assert (
            len(evolvable_composed.init_dicts["vector_mlp"]["hidden_size"])
            == initial_hidden_size + 1
        )
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert (
            len(evolvable_composed.init_dicts["vector_mlp"]["hidden_size"])
            == initial_hidden_size
        )


def test_add_mlp_layer_else_statement(device, default_mlp_config):
    config = default_mlp_config.copy()
    config["hidden_size"] = [64, 64]
    config["max_hidden_layers"] = 2

    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        num_outputs=4,
        mlp_config=config,
        vector_space_mlp=True,
        device=device,
    )
    initial_hidden_size = len(evolvable_composed.mlp_config["hidden_size"])
    getattr(evolvable_composed, "feature_net.vector_mlp.add_layer")()
    assert initial_hidden_size == len(
        evolvable_composed.init_dicts["vector_mlp"]["hidden_size"]
    )


######### Test remove_mlp_layer #########
@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_remove_mlp_layer(
    observation_space,
    num_outputs,
    device,
    default_mlp_config,
):
    config = default_mlp_config.copy()
    config["hidden_size"] = [64, 64]
    config["max_hidden_layers"] = 5
    config["min_hidden_layers"] = 1

    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        mlp_config=config,
        vector_space_mlp=True,
        device=device,
    )

    initial_hidden_size = len(evolvable_composed.mlp_config["hidden_size"])
    initial_net = evolvable_composed.feature_net["vector_mlp"]
    initial_net_dict = dict(initial_net.named_parameters())
    getattr(evolvable_composed, "feature_net.vector_mlp.remove_layer")()
    new_net = evolvable_composed.feature_net["vector_mlp"]
    if initial_hidden_size > 1:
        assert (
            len(evolvable_composed.init_dicts["vector_mlp"]["hidden_size"])
            == initial_hidden_size - 1
        )
        for key, param in new_net.named_parameters():
            if (
                key in initial_net_dict.keys()
                and param.shape == initial_net_dict[key].shape
            ):
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert (
            len(evolvable_composed.init_dicts["vector_mlp"]["hidden_size"])
            == initial_hidden_size
        )


######### Test add_mlp_node #########
@pytest.mark.parametrize(
    "observation_space, num_outputs, layer_index",
    [
        (generate_dict_or_tuple_space(2, 3), 10, None),
        (generate_dict_or_tuple_space(2, 3), 1, 1),
    ],
)
def test_add_nodes(
    observation_space,
    num_outputs,
    layer_index,
    device,
    default_mlp_config,
):
    config = default_mlp_config.copy()
    config["hidden_size"] = [64, 64]

    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        mlp_config=config,
        vector_space_mlp=True,
        device=device,
    )
    original_hidden_size = copy.deepcopy(evolvable_composed.mlp_config["hidden_size"])
    layer = layer_index
    result = getattr(evolvable_composed, "feature_net.vector_mlp.add_node")(
        hidden_layer=layer
    )
    hidden_layer = result["hidden_layer"]
    numb_new_nodes = result["numb_new_nodes"]
    assert (
        evolvable_composed.init_dicts["vector_mlp"]["hidden_size"][hidden_layer]
        == original_hidden_size[hidden_layer] + numb_new_nodes
    )


######### Test remove_mlp_node #########
@pytest.mark.parametrize(
    "observation_space, num_outputs, layer_index, numb_new_nodes",
    [
        (generate_dict_or_tuple_space(2, 3), 10, 1, None),
        (generate_dict_or_tuple_space(2, 3), 1, None, 4),
    ],
)
def test_remove_nodes(
    observation_space,
    num_outputs,
    layer_index,
    numb_new_nodes,
    device,
    default_mlp_config,
):
    config = default_mlp_config.copy()
    config["hidden_size"] = [70, 70]
    config["min_mlp_nodes"] = 4
    config["min_hidden_layers"] = 1

    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        mlp_config=config,
        vector_space_mlp=True,
        device=device,
    )
    layer = layer_index
    original_hidden_size = copy.deepcopy(evolvable_composed.mlp_config["hidden_size"])
    result = getattr(evolvable_composed, "feature_net.vector_mlp.remove_node")(
        numb_new_nodes=numb_new_nodes, hidden_layer=layer
    )
    hidden_layer = result["hidden_layer"]
    numb_new_nodes = result["numb_new_nodes"]
    assert (
        evolvable_composed.init_dicts["vector_mlp"]["hidden_size"][hidden_layer]
        == original_hidden_size[hidden_layer] - numb_new_nodes
    )


######### Test add_cnn_layer #########
@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3, image_shape=(3, 128, 128)), 10),
        (generate_dict_or_tuple_space(2, 3, image_shape=(3, 128, 128)), 1),
    ],
)
def test_add_cnn_layer_simple(
    observation_space,
    num_outputs,
    device,
    default_cnn_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        device=device,
    )
    initial_channel_num = len(evolvable_composed.cnn_config["channel_size"])
    initial_net = evolvable_composed.feature_net["image_1"]
    initial_net_dict = dict(initial_net.named_parameters())
    getattr(evolvable_composed, "feature_net.image_1.add_layer")()
    new_net = evolvable_composed.feature_net["image_1"]
    if initial_channel_num < 6:
        assert (
            len(evolvable_composed.init_dicts["image_1"]["channel_size"])
            == initial_channel_num + 1
        )
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys() and "linear_output" not in key:
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert (
            len(evolvable_composed.init_dicts["image_1"]["channel_size"])
            == initial_channel_num
        )


@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),  # exceeds max layer limit
    ],
)
def test_add_cnn_layer_no_layer_added(
    observation_space,
    num_outputs,
    device,
    default_cnn_config,
):
    # Modify config to have max layers
    config = default_cnn_config.copy()
    config["channel_size"] = [8, 8, 8, 8, 8, 8]
    config["kernel_size"] = [2, 2, 2, 2, 2, 2]
    config["stride_size"] = [2, 2, 1, 1, 1, 1]

    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=config,
        device=device,
    )
    getattr(evolvable_composed, "feature_net.image_1.add_layer")()
    assert len(config["channel_size"]) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )


@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),  # exceeds max-layer limit
    ],
)
def test_add_and_remove_multiple_cnn_layers(
    observation_space,
    num_outputs,
    device,
    default_cnn_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        device=device,
    )
    # Keep adding layers until we reach max or it is infeasible
    for _ in range(evolvable_composed.cnn_config["max_hidden_layers"]):
        getattr(evolvable_composed, "feature_net.image_1.add_layer")()

    # Do a forward pass to ensure network parameter validity
    sample_input = observation_space.sample()
    output = evolvable_composed(sample_input)
    assert output.squeeze().shape[0] == num_outputs
    assert len(evolvable_composed.init_dicts["image_1"]["stride_size"]) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )
    assert len(evolvable_composed.init_dicts["image_1"]["kernel_size"]) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )

    # Remove as many layers as possible
    for _ in evolvable_composed.init_dicts["image_1"]["channel_size"]:
        getattr(evolvable_composed, "feature_net.image_1.remove_layer")()

    assert (
        len(evolvable_composed.init_dicts["image_1"]["channel_size"])
        == evolvable_composed.init_dicts["image_1"]["min_hidden_layers"]
    )

    # Do a forward pass to ensure network parameter validity
    output = evolvable_composed(sample_input)
    assert output.squeeze().shape[0] == num_outputs
    assert len(evolvable_composed.init_dicts["image_1"]["stride_size"]) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )
    assert len(evolvable_composed.init_dicts["image_1"]["kernel_size"]) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )


def test_add_cnn_layer_else_statement(device, default_cnn_config):
    config = default_cnn_config.copy()
    config["channel_size"] = [32, 32]
    config["kernel_size"] = [3, 3]
    config["stride_size"] = [1, 1]
    config["max_hidden_layers"] = 2

    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        num_outputs=4,
        cnn_config=config,
        device=device,
    )
    original_num_hidden_layers = copy.deepcopy(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )
    getattr(evolvable_composed, "feature_net.image_1.add_layer")()
    assert len(original_num_hidden_layers) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )


######### Test remove_cnn_layer #########
@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_remove_cnn_layer(
    observation_space,
    num_outputs,
    device,
    default_cnn_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        device=device,
    )
    initial_channel_num = len(evolvable_composed.cnn_config["channel_size"])
    initial_net = evolvable_composed.feature_net["image_1"]
    initial_net_dict = dict(initial_net.named_parameters())
    getattr(evolvable_composed, "feature_net.image_1.remove_layer")()
    new_net = evolvable_composed.feature_net["image_1"]

    if initial_channel_num > 1:
        assert (
            len(evolvable_composed.init_dicts["image_1"]["channel_size"])
            == initial_channel_num - 1
        )
        for key, param in new_net.named_parameters():
            if (
                key in initial_net_dict.keys()
                and param.shape == initial_net_dict[key].shape
            ):
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert (
            len(evolvable_composed.init_dicts["image_1"]["channel_size"])
            == initial_channel_num
        )


######### Test add_cnn_channel #########
@pytest.mark.parametrize(
    "observation_space, num_outputs, layer_index",
    [
        (generate_dict_or_tuple_space(2, 3), 10, 0),
        (generate_dict_or_tuple_space(2, 3), 1, None),
    ],
)
def test_add_channels(
    observation_space,
    num_outputs,
    layer_index,
    device,
    default_cnn_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        device=device,
    )
    original_channel_size = copy.deepcopy(evolvable_composed.cnn_config["channel_size"])
    result = getattr(evolvable_composed, "feature_net.image_1.add_channel")(
        hidden_layer=layer_index
    )
    hidden_layer = result["hidden_layer"]
    numb_new_channels = result["numb_new_channels"]
    assert (
        evolvable_composed.init_dicts["image_1"]["channel_size"][hidden_layer]
        == original_channel_size[hidden_layer] + numb_new_channels
    )


######### Test remove_cnn_channel #########
@pytest.mark.parametrize(
    "observation_space, num_outputs, layer_index, numb_new_channels",
    [
        (generate_dict_or_tuple_space(2, 3), 10, None, None),
        (generate_dict_or_tuple_space(2, 3), 1, 0, 2),
    ],
)
def test_remove_channels(
    observation_space,
    num_outputs,
    layer_index,
    numb_new_channels,
    device,
    default_cnn_config,
):
    config = default_cnn_config.copy()
    config["min_channel_size"] = 4

    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=config,
        device=device,
    )
    original_channel_size = copy.deepcopy(evolvable_composed.cnn_config["channel_size"])
    result = getattr(evolvable_composed, "feature_net.image_1.remove_channel")(
        numb_new_channels=numb_new_channels, hidden_layer=layer_index
    )
    hidden_layer = result["hidden_layer"]
    numb_new_channels = result["numb_new_channels"]
    assert (
        evolvable_composed.init_dicts["image_1"]["channel_size"][hidden_layer]
        == original_channel_size[hidden_layer] - numb_new_channels
    )


######### Test change_cnn_kernel #########
def test_change_cnn_kernel(device, default_cnn_config, dict_space):
    config = default_cnn_config.copy()
    config["channel_size"] = [32, 32]
    config["kernel_size"] = [3, 3]
    config["stride_size"] = [1, 1]

    evolvable_composed = EvolvableMultiInput(
        observation_space=dict_space,
        num_outputs=4,
        cnn_config=config,
        device=device,
    )
    # Change kernel size
    getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    while evolvable_composed.init_dicts["image_1"]["kernel_size"] == [(3, 3), (3, 3)]:
        getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    # Check if kernel size has changed
    assert evolvable_composed.init_dicts["image_1"]["kernel_size"] != [
        (3, 3),
        (3, 3),
    ], evolvable_composed.init_dicts["image_1"]["kernel_size"]


def test_change_kernel_size(device, default_cnn_config, dict_space):
    config = default_cnn_config.copy()
    config["channel_size"] = [32, 32]
    config["kernel_size"] = [3, 3]
    config["stride_size"] = [1, 1]

    evolvable_composed = EvolvableMultiInput(
        observation_space=dict_space,
        num_outputs=4,
        cnn_config=config,
        device=device,
    )

    for _ in range(10):
        # Change kernel size and ensure we can make a valid forward pass
        getattr(evolvable_composed, "feature_net.image_1.change_kernel")()
        sample_input = dict_space.sample()
        output = evolvable_composed(sample_input)
        assert output.squeeze().shape[0] == 4  # (num actions)


def test_change_cnn_kernel_else_statement(device, default_cnn_config, dict_space):
    config = default_cnn_config.copy()
    config["channel_size"] = [32, 32]
    config["kernel_size"] = [3, 3]
    config["stride_size"] = [1, 1]

    evolvable_composed = EvolvableMultiInput(
        observation_space=dict_space,
        num_outputs=4,
        cnn_config=config,
        device=device,
    )

    # Change kernel size
    getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    while evolvable_composed.init_dicts["image_1"]["kernel_size"] == [3, 3]:
        getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    # Check if kernel size has changed
    assert evolvable_composed.init_dicts["image_1"]["kernel_size"] != [3, 3]


def test_change_cnn_kernel_multi(device, multiagent_cnn_config, dict_space):
    config = multiagent_cnn_config.copy()
    config["channel_size"] = [32, 32]
    config["kernel_size"] = [3, 3]
    config["stride_size"] = [1, 1]

    evolvable_composed = EvolvableMultiInput(
        observation_space=dict_space,
        num_outputs=4,
        cnn_config=config,
        device=device,
    )

    # Change kernel size
    getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    while evolvable_composed.init_dicts["image_1"]["kernel_size"] == [3, 3]:
        getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    # Check if kernel size has changed
    assert evolvable_composed.init_dicts["image_1"]["kernel_size"] != [
        3,
        3,
    ], evolvable_composed.init_dicts["image_1"]["kernel_size"]


def test_change_cnn_kernel_multi_else_statement(
    device, multiagent_cnn_config, dict_space
):
    config = multiagent_cnn_config.copy()
    config["channel_size"] = [32]
    config["kernel_size"] = [3]
    config["stride_size"] = [1]

    evolvable_composed = EvolvableMultiInput(
        observation_space=dict_space,
        num_outputs=4,
        cnn_config=config,
        device=device,
    )

    # Change kernel size
    while evolvable_composed.init_dicts["image_1"]["kernel_size"] == [3]:
        getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    # Check if kernel size has changed
    assert len(evolvable_composed.init_dicts["image_1"]["kernel_size"]) == 2


######### Test clone #########
@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_clone_instance(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    device,
    default_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        mlp_config=default_mlp_config,
        device=device,
    )

    original_nets = {
        k: dict(net.named_parameters())
        for k, net in evolvable_composed.feature_net.items()
    }
    clone = evolvable_composed.clone()
    assert isinstance(clone, EvolvableMultiInput)
    assert_state_dicts_equal(clone.state_dict(), evolvable_composed.state_dict())

    for key, cloned_net in clone.feature_net.items():
        original_net = original_nets[key]
        for key, param in cloned_net.named_parameters():
            torch.testing.assert_close(param, original_net[key])


@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_add_latent_node(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    device,
    default_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        latent_dim=20,
        min_latent_dim=8,
        max_latent_dim=128,
        cnn_config=default_cnn_config,
        mlp_config=default_mlp_config,
        device=device,
    )

    initial_latent_dim = evolvable_composed.latent_dim
    evolvable_composed.add_latent_node()
    assert evolvable_composed.latent_dim > initial_latent_dim


@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_remove_latent_node(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    device,
    default_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        latent_dim=100,
        min_latent_dim=8,
        max_latent_dim=128,
        cnn_config=default_cnn_config,
        mlp_config=default_mlp_config,
        device=device,
    )

    initial_latent_dim = evolvable_composed.latent_dim
    evolvable_composed.remove_latent_node()
    assert evolvable_composed.latent_dim < initial_latent_dim


@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_change_activation(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    device,
    default_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        mlp_config=default_mlp_config,
        device=device,
    )

    new_activation = "Tanh"
    evolvable_composed.change_activation(new_activation)
    for key, net in evolvable_composed.feature_net.modules().items():
        assert net.activation == new_activation


@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_vector_space_mlp(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    device,
    default_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        mlp_config=default_mlp_config,
        vector_space_mlp=True,
        device=device,
    )

    sample = observation_space.sample()
    if isinstance(sample, dict):
        input_tensor = {
            k: torch.tensor(sample[k]).unsqueeze(0).to(device) for k in sample
        }
    else:
        input_tensor = tuple(
            [torch.tensor(comp).unsqueeze(0).to(device) for comp in sample]
        )

    output = evolvable_composed.forward(input_tensor)
    assert output.shape[1] == num_outputs


@pytest.mark.parametrize(
    "observation_space, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), 10),
        (generate_dict_or_tuple_space(2, 3), 1),
    ],
)
def test_latent_dim_bounds(
    observation_space: DictOrTupleSpace,
    num_outputs: int,
    device,
    default_cnn_config,
    default_mlp_config,
):
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        mlp_config=default_mlp_config,
        latent_dim=128,
        min_latent_dim=8,
        max_latent_dim=128,
        device=device,
    )
    # Test maximum bound
    evolvable_composed.add_latent_node()
    assert evolvable_composed.latent_dim <= 128

    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        num_outputs=num_outputs,
        cnn_config=default_cnn_config,
        mlp_config=default_mlp_config,
        latent_dim=8,
        min_latent_dim=8,
        max_latent_dim=128,
        device=device,
    )
    # Test minimum bound
    evolvable_composed.remove_latent_node()
    assert evolvable_composed.latent_dim >= 8
