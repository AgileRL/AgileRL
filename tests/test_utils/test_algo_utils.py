import numpy as np
import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator
from gymnasium import spaces

from agilerl.utils.algo_utils import apply_image_normalization, unwrap_optimizer


@pytest.mark.parametrize("distributed", [(True), (False)])
def test_algo_utils_single_net(distributed):
    simple_net = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
    lr = 0.01
    optimizer = torch.optim.Adam(simple_net.parameters(), lr=lr)
    if distributed:
        accelerator = Accelerator(device_placement=False)
        optimizer = accelerator.prepare(optimizer)
    else:
        accelerator = None

    unwrapped_optimizer = unwrap_optimizer(optimizer, simple_net, lr)
    assert isinstance(unwrapped_optimizer, torch.optim.Adam)


def test_algo_utils_multi_nets():
    simple_net = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
    simple_net_two = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    lr = 0.01
    optimizer = torch.optim.Adam(
        [
            {"params": simple_net.parameters(), "lr": lr},
            {"params": simple_net_two.parameters(), "lr": lr},
        ]
    )
    accelerator = Accelerator(device_placement=False)
    optimizer = accelerator.prepare(optimizer)
    unwrapped_optimizer = unwrap_optimizer(optimizer, [simple_net, simple_net_two], lr)
    assert isinstance(unwrapped_optimizer, torch.optim.Adam)


def test_inf_in_high():
    # Create observation space with inf in high
    obs_space = spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, 1]))
    obs = np.array([0.5, 0.5])

    with pytest.warns(UserWarning, match="np.inf detected in observation_space.high"):
        result = apply_image_normalization(obs, obs_space)

    np.testing.assert_array_equal(result, obs)


def test_neg_inf_in_low():
    # Create observation space with -inf in low
    obs_space = spaces.Box(low=np.array([-np.inf, 0]), high=np.array([1, 1]))
    obs = np.array([0.5, 0.5])

    with pytest.warns(UserWarning, match="-np.inf detected in observation_space.low"):
        result = apply_image_normalization(obs, obs_space)

    np.testing.assert_array_equal(result, obs)


def test_already_normalized():
    # Create observation space that's already normalized (high=1, low=0)
    obs_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]))
    obs = np.array([0.5, 0.5])

    result = apply_image_normalization(obs, obs_space)
    np.testing.assert_array_equal(result, obs)


def test_normalization_needed():
    # Create observation space that needs normalization
    obs_space = spaces.Box(low=np.array([0, 0]), high=np.array([255, 255]))
    obs = np.array([127.5, 127.5])

    result = apply_image_normalization(obs, obs_space)
    expected = obs / 255.0  # Expected normalized values
    np.testing.assert_array_almost_equal(result, expected)


def test_multi_dimensional():
    # Test with multi-dimensional array
    obs_space = spaces.Box(low=np.zeros((2, 2)), high=np.ones((2, 2)) * 255)
    obs = np.ones((2, 2)) * 127.5

    result = apply_image_normalization(obs, obs_space)
    expected = obs / 255.0
    np.testing.assert_array_almost_equal(result, expected)


def test_different_ranges():
    # Test with different ranges for different dimensions
    obs_space = spaces.Box(low=np.array([0, -1]), high=np.array([255, 1]))
    obs = np.array([127.5, 0])

    result = apply_image_normalization(obs, obs_space)
    expected = np.array([127.5 / 255, 0.5])  # Each dimension normalized to its range
    np.testing.assert_array_almost_equal(result, expected)


# Helper function to check warning was raised
def assert_warning_raised(warning_list, expected_message):
    assert any(expected_message in str(w.message) for w in warning_list)
