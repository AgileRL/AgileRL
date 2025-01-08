
import pytest
import torch.nn as nn

from agilerl.utils.evolvable_networks import (
    get_activation,
    calc_max_kernel_sizes,
    create_cnn,
    create_mlp
)

######### Test get_activation #########
def test_returns_correct_activation_function_for_all_supported_names():
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
        activation = get_activation(name)
        assert isinstance(activation, nn.Module)

@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size",
    [
        ([1, 16, 16], [32, 16], [3, 2], [1, 1]),
    ],
)
def test_calc_max_kernel_sizes(
    input_shape,
    channel_size,
    kernel_size,
    stride_size
):
    max_kernel_sizes = calc_max_kernel_sizes(
        channel_size, kernel_size, stride_size, input_shape
    )
    assert max_kernel_sizes == [3, 3]


@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size",
    [
        ([1, 3, 3], [32, 16], [1, 1], [1, 1]),
    ],
)
def test_max_kernel_size_negative(
    input_shape,
    channel_size,
    kernel_size,
    stride_size
):
    max_kernel_sizes = calc_max_kernel_sizes(
        channel_size, kernel_size, stride_size, input_shape
    )
    assert max_kernel_sizes == [1, 1]
    

@pytest.mark.parametrize("output_vanish, noisy", [(True, True), (False, True)])
def test_create_mlp(output_vanish, noisy):
    net = create_mlp(
        10,
        4,
        [32, 32],
        output_vanish=output_vanish,
        output_activation=None,
        noisy=noisy,
        )
    assert isinstance(net, nn.Module)

######### Test create_mlp and create_cnn########
@pytest.mark.parametrize("noisy, output_vanish", [(False, True), (True, False)])
def test_create_cnn(noisy, output_vanish):
    feature_net = create_cnn(
        "Conv2d",
        1,
        [32, 32], 
        [3, 3],
        [1, 1],
        "feature",
        layer_norm=True,
        )

    head = create_mlp(
        10,
        4,
        [64, 64],
        output_activation=None,
        noisy=noisy,
        name="value",
        output_vanish=output_vanish,
    )

    feature_net = nn.ModuleDict(feature_net)

    assert isinstance(head, nn.Module)
    assert isinstance(feature_net, nn.Module)


@pytest.mark.parametrize("noisy, output_vanish", [(False, True), (True, False)])
def test_create_cnn_multi(noisy, output_vanish):
    feature_net = create_cnn(
        "Conv3d",
        1,
        [32, 32],
        [3, 3],
        [1, 1],
        "feature",
        layer_norm=True,
        )
    head = create_mlp(
        10,
        4,
        [64, 64],
        output_activation=None,
        noisy=noisy,
        name="value",
        output_vanish=output_vanish,
    )
    
    feature_net = nn.ModuleDict(feature_net)

    assert isinstance(head, nn.Module)
    assert isinstance(feature_net, nn.Module)
