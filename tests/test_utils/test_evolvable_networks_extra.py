import pytest
import torch.nn as nn

from agilerl.utils.evolvable_networks import get_pooling


class TestGetPooling:
    def test_maxpool3d(self):
        assert isinstance(get_pooling("MaxPool3d", 2, 1, 0), nn.MaxPool3d)

    def test_avgpool3d(self):
        assert isinstance(get_pooling("AvgPool3d", 2, 1, 0), nn.AvgPool3d)

    def test_maxpool2d(self):
        assert isinstance(get_pooling("MaxPool2d", 2, 1, 0), nn.MaxPool2d)

    def test_invalid_pooling_name_raises(self):
        with pytest.raises(ValueError, match="Invalid pooling layer"):
            get_pooling("NotARealPool", 2, 1, 0)

    def test_scalar_sizes_stay_scalar(self):
        # Scalar args keep an int kernel_size so the repr matches a reference
        # network built with ints.
        assert get_pooling("MaxPool2d", 2, 1, 0).kernel_size == 2

    def test_tuple_size_is_normalized_to_constructor_arity(self):
        pool = get_pooling("MaxPool2d", (2, 2), 1, 0)
        assert pool.kernel_size == (2, 2)
        assert pool.stride == (1, 1)
