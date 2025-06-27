import pytest
import torch

from agilerl.modules.resnet import EvolvableResNet
from tests.helper_functions import assert_state_dicts_equal


######### Test instantiation #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
    [
        ([3, 32, 32], 64, 3, 1, 10, 2),
        ([3, 64, 64], 128, 3, 2, 10, 3),
    ],
)
def test_instantiation_without_errors(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    num_blocks,
    device,
):
    evolvable_resnet = EvolvableResNet(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        num_blocks=num_blocks,
        device=device,
    )
    assert isinstance(evolvable_resnet, EvolvableResNet)


@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
    [
        ([3, 32, 32], 64, 3, 1, 10, "two"),  # Incorrect type for num_blocks
        ([3, 32, 32], 64, 3, 1, 0, 2),  # num_outputs cannot be zero
    ],
)
def test_incorrect_instantiation(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    num_blocks,
    device,
):
    with pytest.raises(AssertionError):
        EvolvableResNet(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            num_blocks=num_blocks,
            device=device,
        )


######### Test forward #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks, output_shape",
    [
        ([3, 32, 32], 64, 3, 1, 10, 2, (1, 10)),
        ([3, 64, 64], 128, 3, 2, 10, 3, (1, 10)),
    ],
)
def test_forward(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    num_blocks,
    output_shape,
    device,
):
    evolvable_resnet = EvolvableResNet(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        num_blocks=num_blocks,
        device=device,
    )
    input_tensor = torch.randn(1, *input_shape).to(device)
    output = evolvable_resnet.forward(input_tensor)
    assert output.shape == output_shape


######### Test add_block #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
    [
        ([3, 32, 32], 64, 3, 1, 10, 2),
        ([3, 64, 64], 128, 3, 2, 10, 3),
    ],
)
def test_add_block(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    num_blocks,
    device,
):
    evolvable_resnet = EvolvableResNet(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        num_blocks=num_blocks,
        device=device,
    )
    initial_blocks = evolvable_resnet.num_blocks
    evolvable_resnet.add_block()
    assert evolvable_resnet.num_blocks == initial_blocks + 1


######### Test remove_block #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
    [
        ([3, 32, 32], 64, 3, 1, 10, 2),
        ([3, 64, 64], 128, 3, 2, 10, 3),
    ],
)
def test_remove_block(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    num_blocks,
    device,
):
    evolvable_resnet = EvolvableResNet(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        num_blocks=num_blocks,
        device=device,
    )
    initial_blocks = evolvable_resnet.num_blocks
    evolvable_resnet.remove_block()
    assert evolvable_resnet.num_blocks == initial_blocks - 1


######### Test add_channel #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
    [
        ([3, 32, 32], 64, 3, 1, 10, 2),
        ([3, 64, 64], 128, 3, 2, 10, 3),
    ],
)
def test_add_channel(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    num_blocks,
    device,
):
    evolvable_resnet = EvolvableResNet(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        num_blocks=num_blocks,
        device=device,
    )
    initial_channel_size = evolvable_resnet.channel_size
    result = evolvable_resnet.add_channel()
    numb_new_channels = result["numb_new_channels"]
    assert evolvable_resnet.channel_size == initial_channel_size + numb_new_channels


######### Test remove_channel #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
    [
        ([3, 32, 32], 128, 3, 1, 10, 2),
        ([3, 64, 64], 128, 3, 2, 10, 3),
    ],
)
def test_remove_channel(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    num_blocks,
    device,
):
    evolvable_resnet = EvolvableResNet(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        num_blocks=num_blocks,
        device=device,
    )
    initial_channel_size = evolvable_resnet.channel_size
    result = evolvable_resnet.remove_channel()
    numb_new_channels = result["numb_new_channels"]
    assert evolvable_resnet.channel_size == initial_channel_size - numb_new_channels


######### Test clone #########
@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
    [
        ([3, 32, 32], 64, 3, 1, 10, 2),
        ([3, 64, 64], 128, 3, 2, 10, 3),
    ],
)
def test_clone_instance(
    input_shape,
    channel_size,
    kernel_size,
    stride_size,
    num_outputs,
    num_blocks,
    device,
):
    evolvable_resnet = EvolvableResNet(
        input_shape=input_shape,
        channel_size=channel_size,
        kernel_size=kernel_size,
        stride_size=stride_size,
        num_outputs=num_outputs,
        num_blocks=num_blocks,
        device=device,
    )
    original_net_dict = dict(evolvable_resnet.model.named_parameters())
    clone = evolvable_resnet.clone()
    clone_net = clone.model
    assert isinstance(clone, EvolvableResNet)
    assert_state_dicts_equal(clone.state_dict(), evolvable_resnet.state_dict())
    for key, param in clone_net.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key])
