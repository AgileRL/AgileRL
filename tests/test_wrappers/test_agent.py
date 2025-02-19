from unittest.mock import MagicMock

import pytest
import torch
from gymnasium import spaces

from agilerl.algorithms.core import MultiAgentRLAlgorithm, RLAlgorithm
from agilerl.wrappers.agent import RSNorm


@pytest.fixture
def setup_rs_norm():
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
    mock_agent = MagicMock(spec=RLAlgorithm)
    mock_agent.observation_space = observation_space
    mock_agent.action_space = spaces.Discrete(2)
    mock_agent.device = "cpu"
    mock_agent.training = True

    wrapper = RSNorm(mock_agent)
    return wrapper, mock_agent


@pytest.fixture
def setup_rs_norm_dict():
    observation_space = spaces.Dict(
        {
            "sensor1": spaces.Box(low=-1.0, high=1.0, shape=(3,)),
            "sensor2": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
        }
    )
    mock_agent = MagicMock(spec=RLAlgorithm)
    mock_agent.observation_space = observation_space
    mock_agent.action_space = spaces.Discrete(2)
    mock_agent.device = "cpu"
    mock_agent.training = True

    wrapper = RSNorm(mock_agent)
    return wrapper, mock_agent


@pytest.fixture
def setup_rs_norm_tuple():
    observation_space = spaces.Tuple(
        (
            spaces.Box(low=-1.0, high=1.0, shape=(3,)),
            spaces.Box(low=-1.0, high=1.0, shape=(2,)),
        )
    )
    mock_agent = MagicMock(spec=RLAlgorithm)
    mock_agent.observation_space = observation_space
    mock_agent.action_space = spaces.Discrete(2)
    mock_agent.device = "cpu"
    mock_agent.training = True

    wrapper = RSNorm(mock_agent)
    return wrapper, mock_agent


@pytest.fixture
def setup_rs_norm_multi_agent():
    observation_space = {
        "agent_1": spaces.Box(low=-1.0, high=1.0, shape=(3,)),
        "agent_2": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
    }
    mock_agent = MagicMock(spec=MultiAgentRLAlgorithm)
    mock_agent.observation_space = observation_space
    mock_agent.action_space = {
        "agent_1": spaces.Discrete(2),
        "agent_2": spaces.Discrete(2),
    }
    mock_agent.device = "cpu"
    mock_agent.training = True

    wrapper = RSNorm(mock_agent)
    return wrapper, mock_agent


def test_normalize_observation(setup_rs_norm):
    wrapper, _ = setup_rs_norm
    obs = torch.tensor([1.0, 2.0, 3.0])
    wrapper.obs_rms.mean = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms.var = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms.epsilon = 1e-4

    normalized_obs = wrapper._normalize_observation(obs)
    expected_obs = (obs - wrapper.obs_rms.mean) / torch.sqrt(
        wrapper.obs_rms.var + wrapper.obs_rms.epsilon
    )
    assert torch.allclose(normalized_obs, expected_obs)


def test_update_statistics(setup_rs_norm):
    wrapper, _ = setup_rs_norm
    obs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    wrapper.update_statistics(obs)

    assert torch.allclose(
        wrapper.obs_rms.mean, torch.tensor([2.5, 3.5, 4.5]), atol=1e-2
    )
    assert torch.allclose(
        wrapper.obs_rms.var, torch.tensor([2.25, 2.25, 2.25]), atol=1e-2
    )


def test_get_action(setup_rs_norm):
    wrapper, mock_agent = setup_rs_norm
    obs = torch.tensor([1.0, 2.0, 3.0])
    wrapper.get_action(obs)


def test_learn(setup_rs_norm):
    wrapper, mock_agent = setup_rs_norm
    experiences = (
        torch.tensor([1.0, 2.0, 3.0]),  # State
        torch.tensor([0]),  # Action
        torch.tensor([1.0]),  # Reward
        torch.tensor([4.0, 5.0, 6.0]),  # Next state
        torch.tensor([0]),  # Done
    )
    wrapper.learn(experiences)


def test_normalize_observation_dict(setup_rs_norm_dict):
    wrapper, _ = setup_rs_norm_dict
    obs = {
        "sensor1": torch.tensor([1.0, 2.0, 3.0]),
        "sensor2": torch.tensor([1.0, 2.0]),
    }
    wrapper.obs_rms["sensor1"].mean = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms["sensor1"].var = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms["sensor1"].epsilon = 1e-4
    wrapper.obs_rms["sensor2"].mean = torch.tensor([1.0, 1.0])
    wrapper.obs_rms["sensor2"].var = torch.tensor([1.0, 1.0])
    wrapper.obs_rms["sensor2"].epsilon = 1e-4

    normalized_obs = wrapper._normalize_observation(obs)
    expected_obs = {
        "sensor1": (obs["sensor1"] - wrapper.obs_rms["sensor1"].mean)
        / torch.sqrt(
            wrapper.obs_rms["sensor1"].var + wrapper.obs_rms["sensor1"].epsilon
        ),
        "sensor2": (obs["sensor2"] - wrapper.obs_rms["sensor2"].mean)
        / torch.sqrt(
            wrapper.obs_rms["sensor2"].var + wrapper.obs_rms["sensor2"].epsilon
        ),
    }
    assert torch.allclose(normalized_obs["sensor1"], expected_obs["sensor1"])
    assert torch.allclose(normalized_obs["sensor2"], expected_obs["sensor2"])


def test_update_statistics_dict(setup_rs_norm_dict):
    wrapper, _ = setup_rs_norm_dict
    obs = {
        "sensor1": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "sensor2": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }
    wrapper.update_statistics(obs)

    assert torch.allclose(
        wrapper.obs_rms["sensor1"].mean, torch.tensor([2.5, 3.5, 4.5]), atol=1e-2
    )
    assert torch.allclose(
        wrapper.obs_rms["sensor1"].var, torch.tensor([2.25, 2.25, 2.25]), atol=1e-2
    )
    assert torch.allclose(
        wrapper.obs_rms["sensor2"].mean, torch.tensor([2.0, 3.0]), atol=1e-2
    )
    assert torch.allclose(
        wrapper.obs_rms["sensor2"].var, torch.tensor([1.0, 1.0]), atol=1e-2
    )


def test_normalize_observation_tuple(setup_rs_norm_tuple):
    wrapper, _ = setup_rs_norm_tuple
    obs = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0]))
    wrapper.obs_rms[0].mean = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms[0].var = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms[0].epsilon = 1e-4
    wrapper.obs_rms[1].mean = torch.tensor([1.0, 1.0])
    wrapper.obs_rms[1].var = torch.tensor([1.0, 1.0])
    wrapper.obs_rms[1].epsilon = 1e-4

    normalized_obs = wrapper._normalize_observation(obs)
    expected_obs = (
        (obs[0] - wrapper.obs_rms[0].mean)
        / torch.sqrt(wrapper.obs_rms[0].var + wrapper.obs_rms[0].epsilon),
        (obs[1] - wrapper.obs_rms[1].mean)
        / torch.sqrt(wrapper.obs_rms[1].var + wrapper.obs_rms[1].epsilon),
    )
    assert torch.allclose(normalized_obs[0], expected_obs[0])
    assert torch.allclose(normalized_obs[1], expected_obs[1])


def test_update_statistics_tuple(setup_rs_norm_tuple):
    wrapper, _ = setup_rs_norm_tuple
    obs = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    wrapper.update_statistics(obs)

    assert torch.allclose(
        wrapper.obs_rms[0].mean, torch.tensor([2.5, 3.5, 4.5]), atol=1e-2
    )
    assert torch.allclose(
        wrapper.obs_rms[0].var, torch.tensor([2.25, 2.25, 2.25]), atol=1e-2
    )
    assert torch.allclose(wrapper.obs_rms[1].mean, torch.tensor([2.0, 3.0]), atol=1e-2)
    assert torch.allclose(wrapper.obs_rms[1].var, torch.tensor([1.0, 1.0]), atol=1e-2)


def test_normalize_observation_multi_agent(setup_rs_norm_multi_agent):
    wrapper, _ = setup_rs_norm_multi_agent
    obs = {
        "agent_1": torch.tensor([1.0, 2.0, 3.0]),
        "agent_2": torch.tensor([1.0, 2.0]),
    }
    wrapper.obs_rms["agent_1"].mean = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms["agent_1"].var = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms["agent_1"].epsilon = 1e-4
    wrapper.obs_rms["agent_2"].mean = torch.tensor([1.0, 1.0])
    wrapper.obs_rms["agent_2"].var = torch.tensor([1.0, 1.0])
    wrapper.obs_rms["agent_2"].epsilon = 1e-4

    normalized_obs = wrapper._normalize_observation(obs)
    expected_obs = {
        "agent_1": (obs["agent_1"] - wrapper.obs_rms["agent_1"].mean)
        / torch.sqrt(
            wrapper.obs_rms["agent_1"].var + wrapper.obs_rms["agent_1"].epsilon
        ),
        "agent_2": (obs["agent_2"] - wrapper.obs_rms["agent_2"].mean)
        / torch.sqrt(
            wrapper.obs_rms["agent_2"].var + wrapper.obs_rms["agent_2"].epsilon
        ),
    }
    assert torch.allclose(normalized_obs["agent_1"], expected_obs["agent_1"])
    assert torch.allclose(normalized_obs["agent_2"], expected_obs["agent_2"])


def test_update_statistics_multi_agent(setup_rs_norm_multi_agent):
    wrapper, _ = setup_rs_norm_multi_agent
    obs = {
        "agent_1": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "agent_2": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }
    wrapper.update_statistics(obs)

    assert torch.allclose(
        wrapper.obs_rms["agent_1"].mean, torch.tensor([2.5, 3.5, 4.5]), atol=1e-2
    )
    assert torch.allclose(
        wrapper.obs_rms["agent_1"].var, torch.tensor([2.25, 2.25, 2.25]), atol=1e-2
    )
    assert torch.allclose(
        wrapper.obs_rms["agent_2"].mean, torch.tensor([2.0, 3.0]), atol=1e-2
    )
    assert torch.allclose(
        wrapper.obs_rms["agent_2"].var, torch.tensor([1.0, 1.0]), atol=1e-2
    )
