import os
from unittest.mock import MagicMock

import dill
import pytest
import torch
from accelerate import Accelerator
from gymnasium import spaces

from agilerl.algorithms import DDPG
from agilerl.algorithms.core import MultiAgentRLAlgorithm, RLAlgorithm
from agilerl.modules import EvolvableMLP
from agilerl.wrappers.agent import RSNorm
from tests.helper_functions import generate_random_box_space


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


@pytest.fixture
def discrete_space():
    return spaces.Discrete(2)


@pytest.fixture
def continuous_space():
    return spaces.Box(low=-1.0, high=1.0, shape=(3,))


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
    assert torch.allclose(normalized_obs["sensor1"], expected_obs["sensor1"], atol=1e-2)
    assert torch.allclose(normalized_obs["sensor2"], expected_obs["sensor2"], atol=1e-2)


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
    assert torch.allclose(normalized_obs[0], expected_obs[0], atol=1e-2)
    assert torch.allclose(normalized_obs[1], expected_obs[1], atol=1e-2)


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
    assert torch.allclose(normalized_obs["agent_1"], expected_obs["agent_1"], atol=1e-2)
    assert torch.allclose(normalized_obs["agent_2"], expected_obs["agent_2"], atol=1e-2)


# Clones the agent and returns an identical agent.
def test_clone_returns_identical_agent():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,))

    ddpg_norm = RSNorm(DDPG(observation_space, action_space))
    ddpg = ddpg_norm.agent
    ddpg.fitness = [200, 200, 200]
    ddpg.scores = [94, 94, 94]
    ddpg.steps = [2500]
    ddpg.tensor_attribute = torch.randn(1)
    clone = ddpg_norm.clone()
    clone_agent = clone.agent

    assert clone_agent.observation_space == ddpg.observation_space
    assert clone_agent.action_space == ddpg.action_space
    assert clone_agent.batch_size == ddpg.batch_size
    assert clone_agent.lr_actor == ddpg.lr_actor
    assert clone_agent.lr_critic == ddpg.lr_critic
    assert clone_agent.learn_step == ddpg.learn_step
    assert clone_agent.gamma == ddpg.gamma
    assert clone_agent.tau == ddpg.tau
    assert clone_agent.mut == ddpg.mut
    assert clone_agent.device == ddpg.device
    assert clone_agent.accelerator == ddpg.accelerator
    assert str(clone_agent.actor.state_dict()) == str(ddpg.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(clone_agent.critic.state_dict()) == str(ddpg.critic.state_dict())
    assert str(clone_agent.critic_target.state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        ddpg.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_optimizer.state_dict()) == str(
        ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores
    assert clone_agent.tensor_attribute == ddpg.tensor_attribute

    accelerator = Accelerator()
    ddpg_norm = RSNorm(DDPG(observation_space, action_space, accelerator=accelerator))
    ddpg = ddpg_norm.agent
    clone = ddpg_norm.clone()
    clone_agent = clone.agent

    assert clone_agent.observation_space == ddpg.observation_space
    assert clone_agent.action_space == ddpg.action_space
    assert clone_agent.batch_size == ddpg.batch_size
    assert clone_agent.lr_actor == ddpg.lr_actor
    assert clone_agent.lr_critic == ddpg.lr_critic
    assert clone_agent.learn_step == ddpg.learn_step
    assert clone_agent.gamma == ddpg.gamma
    assert clone_agent.tau == ddpg.tau
    assert clone_agent.mut == ddpg.mut
    assert clone_agent.device == ddpg.device
    assert clone_agent.accelerator == ddpg.accelerator
    assert str(clone_agent.actor.state_dict()) == str(ddpg.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(clone_agent.critic.state_dict()) == str(ddpg.critic.state_dict())
    assert str(clone_agent.critic_target.state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        ddpg.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_optimizer.state_dict()) == str(
        ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores

    accelerator = Accelerator()
    ddpg_norm = RSNorm(
        DDPG(observation_space, action_space, accelerator=accelerator, wrap=False)
    )
    ddpg = ddpg_norm.agent
    clone = ddpg_norm.clone(wrap=False)
    clone_agent = clone.agent

    assert clone_agent.observation_space == ddpg.observation_space
    assert clone_agent.action_space == ddpg.action_space
    assert clone_agent.batch_size == ddpg.batch_size
    assert clone_agent.lr_actor == ddpg.lr_actor
    assert clone_agent.lr_critic == ddpg.lr_critic
    assert clone_agent.learn_step == ddpg.learn_step
    assert clone_agent.gamma == ddpg.gamma
    assert clone_agent.tau == ddpg.tau
    assert clone_agent.mut == ddpg.mut
    assert clone_agent.device == ddpg.device
    assert clone_agent.accelerator == ddpg.accelerator
    print(clone_agent.wrap, ddpg.wrap)
    print("1 = ", clone_agent.actor.state_dict())
    print("\n\n2 = ", ddpg.actor.state_dict())
    assert str(clone_agent.actor.state_dict()) == str(ddpg.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(clone_agent.critic.state_dict()) == str(ddpg.critic.state_dict())
    assert str(clone_agent.critic_target.state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        ddpg.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_optimizer.state_dict()) == str(
        ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores


def test_save_load_checkpoint(tmp_path):
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,))

    ddpg_norm = RSNorm(DDPG(observation_space, action_space))
    checkpoint_path = os.path.join(tmp_path, "checkpoint.pth")
    ddpg_norm.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "wrapper_cls" in checkpoint
    assert "wrapper_init_dict" in checkpoint
    assert "wrapper_attrs" in checkpoint
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr_actor" in checkpoint
    assert "lr_critic" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # load_checkpoint
    loaded_agent = RSNorm(DDPG(observation_space, action_space))
    print(checkpoint_path)
    loaded_agent.load_checkpoint(checkpoint_path)
    ddpg = ddpg_norm.agent

    assert isinstance(loaded_agent, RSNorm)
    # Check if properties and weights are loaded correctly
    assert isinstance(ddpg.actor.encoder, EvolvableMLP)
    assert isinstance(ddpg.actor_target.encoder, EvolvableMLP)
    assert isinstance(ddpg.critic.encoder, EvolvableMLP)
    assert isinstance(ddpg.critic_target.encoder, EvolvableMLP)
    assert ddpg.lr_actor == 1e-4
    assert ddpg.lr_critic == 1e-3
    assert str(ddpg.actor.state_dict()) == str(ddpg.actor_target.state_dict())
    assert str(ddpg.critic.state_dict()) == str(ddpg.critic_target.state_dict())
    assert ddpg.batch_size == 64
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 1e-3
    assert ddpg.mut is None
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]

    loaded_agent = DDPG.load(checkpoint_path)
    ddpg = loaded_agent.agent

    assert isinstance(loaded_agent, RSNorm)
    # Check if properties and weights are loaded correctly
    assert isinstance(ddpg.actor.encoder, EvolvableMLP)
    assert isinstance(ddpg.actor_target.encoder, EvolvableMLP)
    assert isinstance(ddpg.critic.encoder, EvolvableMLP)
    assert isinstance(ddpg.critic_target.encoder, EvolvableMLP)
    assert ddpg.lr_actor == 1e-4
    assert ddpg.lr_critic == 1e-3
    assert str(ddpg.actor.state_dict()) == str(ddpg.actor_target.state_dict())
    assert str(ddpg.critic.state_dict()) == str(ddpg.critic_target.state_dict())
    assert ddpg.batch_size == 64
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 1e-3
    assert ddpg.mut is None
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]
