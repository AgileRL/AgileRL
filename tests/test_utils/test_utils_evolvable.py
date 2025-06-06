from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch._dynamo.eval_frame import OptimizedModule

from agilerl.modules.cnn import MutableKernelSizes
from agilerl.utils.evolvable_networks import (
    compile_model,
    create_cnn,
    create_mlp,
    get_activation,
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
def test_calc_max_kernel_sizes(input_shape, channel_size, kernel_size, stride_size):
    mutable_kernel_sizes = MutableKernelSizes(
        sizes=kernel_size,
        cnn_block_type="Conv2d",
        sample_input=torch.randn(1, 3, 32, 32),
        rng=np.random.default_rng(42),
    )
    max_kernel_sizes = mutable_kernel_sizes.calc_max_kernel_sizes(
        channel_size, stride_size, input_shape
    )
    assert max_kernel_sizes == [3, 3]


@pytest.mark.parametrize(
    "input_shape, channel_size, kernel_size, stride_size",
    [
        ([1, 3, 3], [32, 16], [1, 1], [1, 1]),
    ],
)
def test_max_kernel_size_negative(input_shape, channel_size, kernel_size, stride_size):
    mutable_kernel_sizes = MutableKernelSizes(
        sizes=kernel_size,
        cnn_block_type="Conv2d",
        sample_input=torch.randn(1, 3, 32, 32),
        rng=np.random.default_rng(42),
    )
    max_kernel_sizes = mutable_kernel_sizes.calc_max_kernel_sizes(
        channel_size, stride_size, input_shape
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


def test_compile_model():
    model = nn.Linear(10, 10)

    # Test with mocked torch.compile
    with patch("torch.compile", return_value=model):
        compiled_model = compile_model(model, mode="default")
        # Should have called torch.compile
        assert compiled_model is model

    # Test with already compiled model (mock OptimizedModule)
    with patch("torch._dynamo.eval_frame.OptimizedModule", type(OptimizedModule)):
        optimized_model = Mock(spec=OptimizedModule)

        # Should return the model without recompiling
        result = compile_model(optimized_model, mode="default")
        assert result is optimized_model

    # Test with mode=None
    result = compile_model(model, mode=None)
    assert result is model


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
