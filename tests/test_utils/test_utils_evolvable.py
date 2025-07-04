from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from torch._dynamo.eval_frame import OptimizedModule

from agilerl.modules.cnn import MutableKernelSizes
from agilerl.utils.evolvable_networks import (
    compile_model,
    create_cnn,
    create_mlp,
    get_activation,
    get_input_size_from_space,
    get_output_size_from_space,
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


######### Test get_input_size_from_space #########
class TestGetInputSizeFromSpace:
    """Comprehensive tests for get_input_size_from_space function."""

    def test_discrete_space(self):
        """Test Discrete space returns correct tuple."""
        space = spaces.Discrete(5)
        result = get_input_size_from_space(space)
        assert result == (5,)

    def test_multi_discrete_space(self):
        """Test MultiDiscrete space returns sum of nvec."""
        space = spaces.MultiDiscrete([2, 3, 4])
        result = get_input_size_from_space(space)
        assert result == (9,)  # 2 + 3 + 4

    def test_box_space_1d(self):
        """Test 1D Box space returns shape."""
        space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        result = get_input_size_from_space(space)
        assert result == (4,)

    def test_box_space_2d(self):
        """Test 2D Box space returns shape."""
        space = spaces.Box(low=-1, high=1, shape=(3, 4), dtype=np.float32)
        result = get_input_size_from_space(space)
        assert result == (3, 4)

    def test_box_space_3d_image(self):
        """Test 3D Box space (image) returns shape."""
        space = spaces.Box(low=0, high=255, shape=(3, 32, 32), dtype=np.uint8)
        result = get_input_size_from_space(space)
        assert result == (3, 32, 32)

    def test_multi_binary_space(self):
        """Test MultiBinary space returns tuple of n."""
        space = spaces.MultiBinary(8)
        result = get_input_size_from_space(space)
        assert result == (8,)

    def test_dict_space(self):
        """Test Dict space returns dict of sizes."""
        space = spaces.Dict(
            {
                "position": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
                "velocity": spaces.Discrete(4),
                "sensor": spaces.MultiBinary(6),
            }
        )
        result = get_input_size_from_space(space)
        expected = {"position": (2,), "velocity": (4,), "sensor": (6,)}
        assert result == expected

    def test_tuple_space(self):
        """Test Tuple space returns tuple of sizes."""
        space = spaces.Tuple(
            (
                spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                spaces.Discrete(5),
                spaces.MultiDiscrete([2, 3]),
            )
        )
        result = get_input_size_from_space(space)
        expected = ((3,), (5,), (5,))  # (3,), (5,), (2+3,)
        assert result == expected

    def test_list_of_spaces(self):
        """Test list of spaces returns tuple of sizes."""
        spaces_list = [
            spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            spaces.Discrete(3),
        ]
        result = get_input_size_from_space(spaces_list)
        expected = ((2,), (3,))
        assert result == expected

    def test_nested_dict_space(self):
        """Test nested Dict space."""
        space = spaces.Dict(
            {
                "observations": spaces.Dict(
                    {
                        "position": spaces.Box(
                            low=-10, high=10, shape=(3,), dtype=np.float32
                        ),
                        "velocity": spaces.Box(
                            low=-1, high=1, shape=(3,), dtype=np.float32
                        ),
                    }
                ),
                "action_mask": spaces.MultiBinary(5),
            }
        )
        result = get_input_size_from_space(space)
        expected = {
            "observations": {"position": (3,), "velocity": (3,)},
            "action_mask": (5,),
        }
        assert result == expected

    def test_nested_tuple_space(self):
        """Test nested Tuple space."""
        space = spaces.Tuple(
            (
                spaces.Tuple(
                    (
                        spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                        spaces.Discrete(3),
                    )
                ),
                spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            )
        )
        result = get_input_size_from_space(space)
        expected = (((2,), (3,)), (4,))
        assert result == expected

    def test_unsupported_space_raises_error(self):
        """Test that unsupported space types raise AttributeError."""

        # Create a mock space that's not supported
        class UnsupportedSpace:
            pass

        unsupported_space = UnsupportedSpace()
        with pytest.raises(AttributeError, match="Can't access state dimensions"):
            get_input_size_from_space(unsupported_space)

    def test_empty_dict_space(self):
        """Test empty Dict space."""
        space = spaces.Dict({})
        result = get_input_size_from_space(space)
        assert result == {}

    def test_empty_tuple_space(self):
        """Test empty Tuple space."""
        space = spaces.Tuple(())
        result = get_input_size_from_space(space)
        assert result == ()


######### Test get_output_size_from_space #########
class TestGetOutputSizeFromSpace:
    """Comprehensive tests for get_output_size_from_space function."""

    def test_discrete_space(self):
        """Test Discrete space returns n."""
        space = spaces.Discrete(5)
        result = get_output_size_from_space(space)
        assert result == 5

    def test_multi_discrete_space(self):
        """Test MultiDiscrete space returns sum of nvec."""
        space = spaces.MultiDiscrete([2, 3, 4])
        result = get_output_size_from_space(space)
        assert result == 9  # 2 + 3 + 4

    def test_box_space_1d(self):
        """Test 1D Box space returns first dimension."""
        space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        result = get_output_size_from_space(space)
        assert result == 4

    def test_box_space_multi_dim(self):
        """Test multi-dimensional Box space returns first dimension."""
        space = spaces.Box(low=-1, high=1, shape=(3, 4, 5), dtype=np.float32)
        result = get_output_size_from_space(space)
        assert result == 3

    def test_multi_binary_space(self):
        """Test MultiBinary space returns n."""
        space = spaces.MultiBinary(8)
        result = get_output_size_from_space(space)
        assert result == 8

    def test_dict_space(self):
        """Test Dict space returns dict of sizes."""
        space = spaces.Dict(
            {
                "discrete_action": spaces.Discrete(4),
                "continuous_action": spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                ),
                "binary_action": spaces.MultiBinary(3),
            }
        )
        result = get_output_size_from_space(space)
        expected = {"discrete_action": 4, "continuous_action": 2, "binary_action": 3}
        assert result == expected

    def test_list_of_spaces(self):
        """Test list of spaces returns tuple of sizes."""
        spaces_list = [
            spaces.Discrete(3),
            spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        ]
        result = get_output_size_from_space(spaces_list)
        expected = (3, 2)
        assert result == expected

    def test_nested_dict_space(self):
        """Test nested Dict space."""
        space = spaces.Dict(
            {
                "movement": spaces.Dict(
                    {
                        "direction": spaces.Discrete(4),
                        "speed": spaces.Box(
                            low=0, high=1, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
                "special_action": spaces.MultiBinary(2),
            }
        )
        result = get_output_size_from_space(space)
        expected = {"movement": {"direction": 4, "speed": 1}, "special_action": 2}
        assert result == expected

    def test_unsupported_space_raises_error(self):
        """Test that unsupported space types raise AttributeError."""

        # Create a mock space that's not supported
        class UnsupportedSpace:
            pass

        unsupported_space = UnsupportedSpace()
        with pytest.raises(AttributeError, match="Can't access action dimensions"):
            get_output_size_from_space(unsupported_space)

    def test_empty_dict_space(self):
        """Test empty Dict space."""
        space = spaces.Dict({})
        result = get_output_size_from_space(space)
        assert result == {}

    def test_box_space_scalar(self):
        """Test scalar Box space (shape=())."""
        space = spaces.Box(low=-1, high=1, shape=(), dtype=np.float32)
        # This should raise IndexError since shape[0] doesn't exist
        with pytest.raises(IndexError):
            get_output_size_from_space(space)

    def test_python_tuple(self):
        """Test that Python tuples work."""
        # Python tuple should work
        python_tuple = (
            spaces.Discrete(3),
            spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        )
        result = get_output_size_from_space(python_tuple)
        assert result == (3, 2)


######### Test Edge Cases and Complex Scenarios #########
class TestSpaceSizeEdgeCases:
    """Test edge cases and complex scenarios for both functions."""

    def test_complex_mixed_nested_structure(self):
        """Test complex nested structure with mixed space types."""
        # Complex observation space
        obs_space = spaces.Dict(
            {
                "visual": spaces.Tuple(
                    (
                        spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
                        spaces.Box(low=0, high=255, shape=(1, 32, 32), dtype=np.uint8),
                    )
                ),
                "proprioception": spaces.Dict(
                    {
                        "position": spaces.Box(
                            low=-10, high=10, shape=(6,), dtype=np.float32
                        ),
                        "velocity": spaces.Box(
                            low=-5, high=5, shape=(6,), dtype=np.float32
                        ),
                        "discrete_state": spaces.Discrete(10),
                    }
                ),
                "sensors": spaces.Tuple(
                    (spaces.MultiBinary(16), spaces.MultiDiscrete([4, 8, 2]))
                ),
            }
        )

        # Complex action space - Using Python tuple instead of spaces.Tuple
        # since get_output_size_from_space doesn't support spaces.Tuple
        action_space = spaces.Dict(
            {
                "motor_commands": spaces.Box(
                    low=-1, high=1, shape=(12,), dtype=np.float32
                ),
                "binary_switches": spaces.MultiBinary(8),
            }
        )

        # Test input size
        input_result = get_input_size_from_space(obs_space)
        expected_input = {
            "visual": ((3, 64, 64), (1, 32, 32)),
            "proprioception": {
                "position": (6,),
                "velocity": (6,),
                "discrete_state": (10,),
            },
            "sensors": ((16,), (14,)),
        }
        assert input_result == expected_input

        # Test output size
        output_result = get_output_size_from_space(action_space)
        expected_output = {
            "motor_commands": 12,
            "binary_switches": 8,
        }
        assert output_result == expected_output

    def test_consistency_with_gymnasium_spaces(self):
        """Test that results are consistent with actual gymnasium space sampling."""
        spaces_to_test = [
            spaces.Discrete(7),
            spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
            spaces.MultiDiscrete([3, 4, 5]),
            spaces.MultiBinary(10),
            spaces.Box(low=0, high=255, shape=(3, 28, 28), dtype=np.uint8),
        ]

        for space in spaces_to_test:
            # Get the size from our function
            input_size = get_input_size_from_space(space)

            if isinstance(space, spaces.Discrete):
                # For discrete spaces, we expect tuple with n values
                assert input_size == (space.n,)
            elif isinstance(space, spaces.Box):
                # For box spaces, should match the shape
                assert input_size == space.shape
            elif isinstance(space, spaces.MultiDiscrete):
                # Should be sum of nvec
                assert input_size == (sum(space.nvec),)
            elif isinstance(space, spaces.MultiBinary):
                # Should be n
                assert input_size == (space.n,)

    def test_python_dict_vs_gymnasium_dict(self):
        """Test that function works with both Python dict and gymnasium Dict."""
        # Create equivalent spaces
        gym_dict_space = spaces.Dict(
            {
                "a": spaces.Discrete(3),
                "b": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            }
        )

        python_dict_space = {
            "a": spaces.Discrete(3),
            "b": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        }

        # Both should give same result
        gym_result = get_input_size_from_space(gym_dict_space)
        python_result = get_input_size_from_space(python_dict_space)

        expected = {"a": (3,), "b": (2,)}
        assert gym_result == expected
        assert python_result == expected

    def test_tuple_vs_list_vs_gymnasium_tuple(self):
        """Test that function works with tuple, list, and gymnasium Tuple."""
        discrete_space = spaces.Discrete(3)
        box_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Create equivalent structures
        python_tuple = (discrete_space, box_space)
        python_list = [discrete_space, box_space]
        gym_tuple = spaces.Tuple((discrete_space, box_space))

        # All should give same result
        tuple_result = get_input_size_from_space(python_tuple)
        list_result = get_input_size_from_space(python_list)
        gym_result = get_input_size_from_space(gym_tuple)

        expected = ((3,), (2,))
        assert tuple_result == expected
        assert list_result == expected
        assert gym_result == expected
