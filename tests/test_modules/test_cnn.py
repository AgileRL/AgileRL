import copy

import numpy as np
import pytest
import torch

from agilerl.modules.cnn import EvolvableCNN


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
    "input_shape, channel_size, kernel_size, stride_size, num_outputs",
    [
        ([1, 16, 16], [32], [3], [1], 10),
        ([1, 16, 16], [32], [(3, 3)], [(1, 1)], 10),
        ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], 1),
    ],
)
def test_instantiation_without_errors(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        device=device,
    )
    assert isinstance(evolvable_cnn, EvolvableCNN)


@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs",
    [
        ([1, 16, 16], [32], [3, 3], [1], 10),
        ([1, 16, 16], [32], [3], [1, 1], 10),
        ([1, 16, 16], [32], [(3, 3)], [(1, 1)], 0),
        ([3, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], 1),
    ],
)
def test_incorrect_instantiation(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    device,
):
    with pytest.raises(AssertionError):
        EvolvableCNN(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            device=device,
        )


@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs",
    [([1, 16, 16], [3, 32], [3, 3], [2, 2], 10)],
)
def test_instantiation_for_multi_agents(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        block_type="Conv3d",
        sample_input=torch.randn(1, *input_shape).unsqueeze(2).to(device),
        device=device,
    )
    assert isinstance(evolvable_cnn, EvolvableCNN)


@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, sample_input, num_outputs",
    [
        ([1, 16, 16], [3, 32], [3, 3], [2, 2], "tensor", 10),
        ([1, 16, 16], [3, 32], [3, 3], [2, 2], None, 10),
    ],
)
def test_incorrect_instantiation_for_multi_agents(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    sample_input,
    num_outputs,
    device,
):
    if sample_input == "tensor":
        sample_input = torch.randn(1, *input_shape).to(device)

    with pytest.raises(AssertionError):
        EvolvableCNN(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            block_type="Conv3d",
            sample_input=sample_input,
            device=device,
        )


######### Test forward #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, output_shape",
    [
        ([1, 16, 16], [32], [3], [1], 10, (1, 10)),
        ([1, 16, 16], [32], [(3, 3)], [(1, 1)], 10, (1, 10)),
        ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], 1, (1, 1)),
    ],
)
def test_forward(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    output_shape,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        device=device,
    )
    input_tensor = (
        torch.randn(input_shape).unsqueeze(0).to(dtype=torch.float32)
    )  # To add in a batch size dimension
    input_array = np.expand_dims(np.random.randn(*input_shape), 0)
    input_tensor = input_tensor.to(device)
    output = evolvable_cnn.forward(input_tensor)
    output_array = evolvable_cnn.forward(input_array)
    assert output.shape == output_shape
    assert output_array.shape == output_shape


@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, \
        num_outputs, output_shape",
    [([1, 16, 16], [3, 32], [3, 3], [2, 2], 10, (1, 10))],
)
def test_forward_multi(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    output_shape,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        block_type="Conv3d",
        sample_input=torch.randn(1, *input_shape).unsqueeze(2).to(device),
        device=device,
    )
    input_tensor = torch.randn(1, *input_shape).unsqueeze(2).to(device)
    with torch.no_grad():
        output = evolvable_cnn.forward(input_tensor)
    assert output.shape == output_shape


######### Test add_cnn_layer #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs",
    [
        ([1, 16, 16], [32], [3], [1], 10),
        ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], 1),
    ],
)
def test_add_cnn_layer_simple(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        device=device,
    )
    initial_channel_num = len(evolvable_cnn.channel_size)
    initial_net = evolvable_cnn.model
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_cnn.add_layer()
    new_net = evolvable_cnn.model
    if initial_channel_num < 6:
        assert len(evolvable_cnn.channel_size) == initial_channel_num + 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys() and "linear_output" not in key:
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_cnn.channel_size) == initial_channel_num


@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs",
    [
        ([1, 16, 16], [32, 32], [5, 5], [2, 2], 10),  # invalid output size
        (
            [1, 164, 164],
            [8, 8, 8, 8, 8, 8],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 1, 1, 1, 1],
            10,
        ),  # exceeds max layer limit
    ],
)
def test_add_cnn_layer_no_layer_added(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        device=device,
    )
    evolvable_cnn.add_layer()
    assert len(channel_size) == len(evolvable_cnn.channel_size)


@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs",
    [
        ([3, 84, 84], [8, 8], [2, 2], [2, 2], 10),  # exceeds max-layer limit
    ],
)
def test_add_and_remove_multiple_cnn_layers(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        device=device,
    )
    # Keep adding layers until we reach max or it is infeasible
    for _ in range(evolvable_cnn.max_hidden_layers):
        evolvable_cnn.add_layer()

    # Do a forward pass to ensure network parameter validity
    output = evolvable_cnn(torch.ones(1, 3, 84, 84).to(device))
    assert output.squeeze().shape[0] == num_outputs
    assert len(evolvable_cnn.stride_size) == len(evolvable_cnn.channel_size)
    assert len(evolvable_cnn.mut_kernel_size) == len(evolvable_cnn.channel_size)

    # Remove as many layers as possible
    for _ in evolvable_cnn.channel_size:
        evolvable_cnn.remove_layer()

    assert len(evolvable_cnn.channel_size) == evolvable_cnn.min_hidden_layers

    # Do a forward pass to ensure network parameter validity
    output = evolvable_cnn(torch.ones(1, 3, 84, 84).to(device))
    assert output.squeeze().shape[0] == num_outputs
    assert len(evolvable_cnn.stride_size) == len(evolvable_cnn.channel_size)
    assert len(evolvable_cnn.mut_kernel_size) == len(evolvable_cnn.channel_size)


def test_add_cnn_layer_else_statement(device):
    evolvable_cnn = EvolvableCNN(
        input_shape=[1, 16, 16],
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        num_outputs=4,
        max_hidden_layers=2,
        device=device,
    )
    original_num_hidden_layers = copy.deepcopy(evolvable_cnn.channel_size)
    evolvable_cnn.add_layer()
    assert len(original_num_hidden_layers) == len(evolvable_cnn.channel_size)


######### Test remove_cnn_layer #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs",
    [
        ([1, 16, 16], [32], [3], [1], 10),
        ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], 1),
    ],
)
def test_remove_cnn_layer(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        device=device,
    )
    initial_channel_num = len(evolvable_cnn.channel_size)
    initial_net = evolvable_cnn.model
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_cnn.remove_layer()
    new_net = evolvable_cnn.model
    if initial_channel_num > 1:
        assert len(evolvable_cnn.channel_size) == initial_channel_num - 1
        for key, param in new_net.named_parameters():
            if (
                key in initial_net_dict.keys()
                and param.shape == initial_net_dict[key].shape
            ):
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_cnn.channel_size) == initial_channel_num


######### Test add_cnn_channel #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, layer_index",
    [
        ([1, 16, 16], [32], [3], [1], 10, 0),
        ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], 1, None),
    ],
)
def test_add_channels(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    layer_index,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        device=device,
    )
    original_channel_size = copy.deepcopy(evolvable_cnn.channel_size)
    result = evolvable_cnn.add_channel(hidden_layer=layer_index)
    hidden_layer = result["hidden_layer"]
    numb_new_channels = result["numb_new_channels"]
    assert (
        evolvable_cnn.channel_size[hidden_layer]
        == original_channel_size[hidden_layer] + numb_new_channels
    )


######### Test remove_cnn_channel #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, layer_index, numb_new_channels",
    [
        ([1, 16, 16], [256], [3], [1], 10, None, None),
        ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], 1, 0, 2),
    ],
)
def test_remove_channels(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    layer_index,
    numb_new_channels,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        min_channel_size=4,
        device=device,
    )
    original_channel_size = copy.deepcopy(evolvable_cnn.channel_size)
    result = evolvable_cnn.remove_channel(
        numb_new_channels=numb_new_channels, hidden_layer=layer_index
    )
    hidden_layer = result["hidden_layer"]
    numb_new_channels = result["numb_new_channels"]
    assert (
        evolvable_cnn.channel_size[hidden_layer]
        == original_channel_size[hidden_layer] - numb_new_channels
    )


######### Test change_cnn_kernel #########
def test_change_cnn_kernel(device):
    evolvable_cnn = EvolvableCNN(
        input_shape=[1, 16, 16],
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        num_outputs=4,
        device=device,
    )
    # Change kernel size
    evolvable_cnn.change_kernel()

    while evolvable_cnn.mut_kernel_size == [(3, 3), (3, 3)]:
        evolvable_cnn.change_kernel()

    # Check if kernel size has changed
    assert evolvable_cnn.mut_kernel_size != [
        (3, 3),
        (3, 3),
    ], evolvable_cnn.mut_kernel_size


def test_change_kernel_size(device):
    evolvable_cnn = EvolvableCNN(
        input_shape=[1, 16, 16],
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        num_outputs=4,
        device=device,
    )

    for _ in range(100):
        # Change kernel size and ensure we can make a valid forward pass
        evolvable_cnn.change_kernel()
        output = evolvable_cnn(torch.ones(1, 1, 16, 16).to(device))
        assert output.squeeze().shape[0] == 4  # (num actions)


def test_change_cnn_kernel_else_statement(device):
    evolvable_cnn = EvolvableCNN(
        input_shape=[1, 16, 16],
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        num_outputs=4,
        device=device,
    )

    # Change kernel size
    evolvable_cnn.change_kernel()

    while evolvable_cnn.mut_kernel_size == [3, 3]:
        evolvable_cnn.change_kernel()

    # Check if kernel size has changed
    assert evolvable_cnn.mut_kernel_size != [3, 3]


def test_change_cnn_kernel_multi(device):
    evolvable_cnn = EvolvableCNN(
        input_shape=[1, 16, 16],
        channel_size=[32, 32],
        kernel_size=[3, 3],
        stride_size=[1, 1],
        num_outputs=4,
        block_type="Conv3d",
        sample_input=torch.randn(1, 1, 16, 16).unsqueeze(2).to(device),
        device=device,
    )

    # Change kernel size
    evolvable_cnn.change_kernel()

    while evolvable_cnn.mut_kernel_size.int_sizes == [3, 3]:
        evolvable_cnn.change_kernel()

    # Check if kernel size has changed
    assert evolvable_cnn.mut_kernel_size != [
        3,
        3,
    ], evolvable_cnn.mut_kernel_size


def test_change_cnn_kernel_multi_else_statement(device):
    evolvable_cnn = EvolvableCNN(
        input_shape=[1, 16, 16],
        channel_size=[32],
        kernel_size=[3],
        stride_size=[1],
        block_type="Conv3d",
        sample_input=torch.randn(1, 1, 16, 16).unsqueeze(2).to(device),
        num_outputs=4,
        device=device,
    )

    # Change kernel size
    while evolvable_cnn.mut_kernel_size.int_sizes == [3]:
        evolvable_cnn.change_kernel()

    # Check if kernel size has changed
    assert len(evolvable_cnn.mut_kernel_size) == 2


######### Test clone #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs",
    [
        ([1, 16, 16], [32], [3], [1], 10),
        ([3, 128, 128], [8, 8, 8], [2, 2, 2], [2, 2, 2], 1),
    ],
)
def test_clone_instance(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    device,
):
    evolvable_cnn = EvolvableCNN(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        device=device,
    )
    original_feature_net_dict = dict(evolvable_cnn.model.named_parameters())
    clone = evolvable_cnn.clone()
    clone_net = clone.model
    assert isinstance(clone, EvolvableCNN)
    assert str(clone.state_dict()) == str(evolvable_cnn.state_dict())
    for key, param in clone_net.named_parameters():
        torch.testing.assert_close(param, original_feature_net_dict[key])
