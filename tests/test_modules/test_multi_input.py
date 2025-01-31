import copy
from typing import Union

import numpy as np
import pytest
import torch
from gymnasium.spaces import Dict, Tuple

from agilerl.modules.multi_input import EvolvableMultiInput
from tests.helper_functions import generate_dict_or_tuple_space

DictOrTupleSpace = Union[Dict, Tuple]


######### Define fixtures #########
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def cleanup():
    yield  # Run the test first
    torch.cuda.empty_cache()  # Free up GPU memory


######### Test instantiation #########


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [32], [(3, 3)], [(1, 1)], [128], 10),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
        ),
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
        (
            generate_dict_or_tuple_space(2, 3, image_shape=(3, 128)),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
        ),
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
    with pytest.raises((AssertionError, ValueError)):
        EvolvableMultiInput(
            observation_space=observation_space,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            hidden_size=hidden_size,
            vector_space_mlp=True if not hidden_size else False,
            num_outputs=num_outputs,
            device=device,
        )


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (
            generate_dict_or_tuple_space(2, 3, dict_space=True),
            [3, 32],
            [3, 3],
            [2, 2],
            [32, 32],
            10,
        )
    ],
)
def test_instantiation_for_multi_agents(
    observation_space: DictOrTupleSpace,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    sample_input = {
        k: torch.randn(1, *space.shape).unsqueeze(2).to(device)
        for k, space in observation_space.spaces.items()
        if "image" in k
    }

    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        cnn_block_type="Conv3d",
        sample_input=sample_input,
        device=device,
    )
    assert isinstance(evolvable_composed, EvolvableMultiInput)


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (
            generate_dict_or_tuple_space(2, 3, dict_space=False),
            [3, 32],
            [3, 3],
            [2, 2],
            [32, 32],
            10,
        ),
        (
            generate_dict_or_tuple_space(2, 3, dict_space=True),
            [3, 32],
            [3, 3],
            [2, 2],
            [32, 32],
            10,
        ),
    ],
)
def test_incorrect_instantiation_for_multi_agents(
    observation_space: DictOrTupleSpace,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    device,
):
    with pytest.raises(TypeError):
        EvolvableMultiInput(
            observation_space=observation_space,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            hidden_size=hidden_size,
            num_outputs=num_outputs,
            cnn_block_type="Conv3d",
            sample_input=None,
            device=device,
        )


######### Test forward #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, output_shape",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10, (1, 10)),
        (
            generate_dict_or_tuple_space(2, 3),
            [32],
            [(3, 3)],
            [(1, 1)],
            [128],
            10,
            (1, 10),
        ),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
            (1, 1),
        ),
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
    "observation_space, channel_size, kernel_size, stride_size, \
        hidden_size, num_outputs, output_shape",
    [
        (
            generate_dict_or_tuple_space(2, 3, dict_space=True),
            [3, 32],
            [3, 3],
            [2, 2],
            [32, 32],
            10,
            (1, 10),
        )
    ],
)
def test_forward_multi(
    observation_space: DictOrTupleSpace,
    channel_size,
    kernel_size,
    stride_size,
    hidden_size,
    num_outputs,
    output_shape,
    device,
):
    sample_input = {
        k: torch.randn(1, *space.shape).unsqueeze(2).to(device)
        for k, space in observation_space.spaces.items()
        if "image" in k
    }
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        hidden_size=hidden_size,
        num_outputs=num_outputs,
        cnn_block_type="Conv3d",
        sample_input=sample_input,
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
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (generate_dict_or_tuple_space(2, 3), [32], [(3, 3)], [(1, 1)], [128], 10),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
        ),
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
        vector_space_mlp=True,
        device=device,
    )

    initial_hidden_size = len(
        evolvable_composed.init_dicts["vector_mlp"]["hidden_size"]
    )
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


def test_add_mlp_layer_else_statement(device):
    evolvable_composed = EvolvableMultiInput(
        observation_space=generate_dict_or_tuple_space(2, 3),
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[32, 32],
        num_outputs=4,
        max_hidden_layers=2,
        vector_space_mlp=True,
        device=device,
    )
    initial_hidden_size = len(
        evolvable_composed.init_dicts["vector_mlp"]["hidden_size"]
    )
    getattr(evolvable_composed, "feature_net.vector_mlp.add_layer")()
    assert initial_hidden_size == len(
        evolvable_composed.init_dicts["vector_mlp"]["hidden_size"]
    )


######### Test remove_mlp_layer #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
        ),
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
        vector_space_mlp=True,
        device=device,
    )

    initial_hidden_size = len(
        evolvable_composed.init_dicts["vector_mlp"]["hidden_size"]
    )
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
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, layer_index",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10, None),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
            1,
        ),
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
        vector_space_mlp=True,
        device=device,
    )
    original_hidden_size = copy.deepcopy(
        evolvable_composed.init_dicts["vector_mlp"]["hidden_size"]
    )
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
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, layer_index, numb_new_nodes",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10, 1, None),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
            None,
            4,
        ),
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
        vector_space_mlp=True,
        device=device,
    )
    layer = layer_index
    original_hidden_size = copy.deepcopy(
        evolvable_composed.init_dicts["vector_mlp"]["hidden_size"]
    )
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
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
        ),
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
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (
            generate_dict_or_tuple_space(2, 3),
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
    getattr(evolvable_composed, "feature_net.image_1.add_layer")()
    assert len(channel_size) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )


@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8],
            [2, 2],
            [2, 2],
            [128],
            10,
        ),  # exceeds max-layer limit
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
        == evolvable_composed.min_cnn_hidden_layers
    )

    # Do a forward pass to ensure network parameter validity
    print(evolvable_composed.feature_net["image_1"])
    print(evolvable_composed.feature_net["image_1"].init_dict["kernel_size"])
    print(evolvable_composed.init_dicts["image_1"]["kernel_size"])
    output = evolvable_composed(sample_input)
    assert output.squeeze().shape[0] == num_outputs
    assert len(evolvable_composed.init_dicts["image_1"]["stride_size"]) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )
    assert len(evolvable_composed.init_dicts["image_1"]["kernel_size"]) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )


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
    original_num_hidden_layers = copy.deepcopy(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )
    getattr(evolvable_composed, "feature_net.image_1.add_layer")()
    assert len(original_num_hidden_layers) == len(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )


######### Test remove_cnn_layer #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
        ),
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
    initial_channel_num = len(evolvable_composed.init_dicts["image_1"]["channel_size"])
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
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, layer_index",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10, 0),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
            None,
        ),
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
    original_channel_size = copy.deepcopy(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )
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
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs, layer_index, numb_new_channels",
    [
        (generate_dict_or_tuple_space(2, 3), [256], [3], [1], [128], 10, None, None),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
            0,
            2,
        ),
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
    original_channel_size = copy.deepcopy(
        evolvable_composed.init_dicts["image_1"]["channel_size"]
    )
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
    getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    while evolvable_composed.init_dicts["image_1"]["kernel_size"] == [(3, 3), (3, 3)]:
        getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    # Check if kernel size has changed
    assert evolvable_composed.init_dicts["image_1"]["kernel_size"] != [
        (3, 3),
        (3, 3),
    ], evolvable_composed.init_dicts["image_1"]["kernel_size"]


def test_change_kernel_size(device):
    observation_space = generate_dict_or_tuple_space(2, 3)
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[32, 32],
        num_outputs=4,
        device=device,
    )

    for _ in range(100):
        # Change kernel size and ensure we can make a valid forward pass
        getattr(evolvable_composed, "feature_net.image_1.change_kernel")()
        sample_input = observation_space.sample()
        output = evolvable_composed(sample_input)
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
    getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    while evolvable_composed.init_dicts["image_1"]["kernel_size"] == [3, 3]:
        getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    # Check if kernel size has changed
    assert evolvable_composed.init_dicts["image_1"]["kernel_size"] != [3, 3]


def test_change_cnn_kernel_multi(device):
    observation_space = generate_dict_or_tuple_space(2, 3, dict_space=True)
    sample_input = {
        k: torch.randn(1, *space.shape).unsqueeze(2).to(device)
        for k, space in observation_space.spaces.items()
        if "image" in k
    }
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        hidden_size=[32, 32],
        num_outputs=4,
        cnn_block_type="Conv3d",
        sample_input=sample_input,
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


def test_change_cnn_kernel_multi_else_statement(device):
    observation_space = generate_dict_or_tuple_space(2, 3, dict_space=True)
    sample_input = {
        k: torch.randn(1, *space.shape).unsqueeze(2).to(device)
        for k, space in observation_space.spaces.items()
        if "image" in k
    }
    evolvable_composed = EvolvableMultiInput(
        observation_space=observation_space,
        channel_size=[32],
        kernel_size=[3],
        stride_size=[1],
        hidden_size=[32, 32],
        num_outputs=4,
        cnn_block_type="Conv3d",
        sample_input=sample_input,
        device=device,
    )

    # Change kernel size
    while evolvable_composed.init_dicts["image_1"]["kernel_size"] == [3]:
        getattr(evolvable_composed, "feature_net.image_1.change_kernel")()

    # Check if kernel size has changed
    assert len(evolvable_composed.init_dicts["image_1"]["kernel_size"]) == 2


######### Test clone #########
@pytest.mark.parametrize(
    "observation_space, channel_size, kernel_size, stride_size, hidden_size, num_outputs",
    [
        (generate_dict_or_tuple_space(2, 3), [32], [3], [1], [128], 10),
        (
            generate_dict_or_tuple_space(2, 3),
            [8, 8, 8],
            [2, 2, 2],
            [2, 2, 2],
            [32, 32, 32],
            1,
        ),
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

    original_nets = {
        k: dict(net.named_parameters())
        for k, net in evolvable_composed.feature_net.items()
    }
    clone = evolvable_composed.clone()
    assert isinstance(clone, EvolvableMultiInput)
    assert str(clone.state_dict()) == str(evolvable_composed.state_dict())

    for key, cloned_net in clone.feature_net.items():
        original_net = original_nets[key]

        for key, param in cloned_net.named_parameters():
            torch.testing.assert_close(param, original_net[key])
