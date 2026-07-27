from gymnasium import spaces

from agilerl.utils.algo_utils import _input_size, _output_size


class TestInputSize:
    def test_multibinary_tuple_shape(self):
        # MultiBinary with a multi-dimensional shape returns a tuple of dims.
        assert _input_size(spaces.MultiBinary([2, 3])) == (2, 3)

    def test_tuple_space(self):
        out = _input_size(spaces.Tuple((spaces.Discrete(3), spaces.Discrete(4))))
        assert out == ((3,), (4,))

    def test_dict_space(self):
        out = _input_size(spaces.Dict({"a": spaces.Discrete(3)}))
        assert out == {"a": (3,)}


class TestOutputSize:
    def test_tuple_space(self):
        out = _output_size(spaces.Tuple((spaces.Discrete(3), spaces.Discrete(4))))
        assert out == (3, 4)

    def test_dict_space(self):
        out = _output_size(spaces.Dict({"a": spaces.Discrete(3)}))
        assert out == {"a": 3}

    def test_leaf_spaces(self):
        assert _output_size(spaces.Box(low=-1, high=1, shape=(5,))) == 5
        assert _output_size(spaces.MultiBinary(4)) == 4
        assert _output_size(spaces.MultiDiscrete([2, 3])) == 5
