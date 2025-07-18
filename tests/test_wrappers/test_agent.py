import copy
import os
from unittest.mock import MagicMock

import dill
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from gymnasium import spaces
from pettingzoo import ParallelEnv

from agilerl.algorithms import DDPG, IPPO
from agilerl.algorithms.core import MultiAgentRLAlgorithm, RLAlgorithm
from agilerl.modules import EvolvableMLP
from agilerl.utils.utils import make_multi_agent_vect_envs
from agilerl.wrappers.agent import AsyncAgentsWrapper, RSNorm
from tests.helper_functions import assert_state_dicts_equal, get_experiences_batch


class DummyMultiEnvAsync(ParallelEnv):
    def __init__(self, observation_spaces, action_spaces):
        super().__init__()
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.agents = ["agent_0", "agent_1", "other_agent_0"]
        self.possible_agents = ["agent_0", "agent_1", "other_agent_0"]
        self.metadata = None
        self.render_mode = None

        # Define observation frequencies (every N steps)
        self.observation_frequencies = {
            "agent_0": 1,  # observes every step
            "agent_1": 1,  # observes every 2 steps
            "other_agent_0": 4,  # observes every 4 steps
        }

        # Initialize step counters for each agent
        self.agent_step_counters = {agent: 0 for agent in self.agents}

        self.active_agents = self.agents.copy()  # Initially all agents are active
        self.current_step = 0

    def action_space(self, agent):
        idx = self.possible_agents.index(agent)
        return self.action_spaces[idx]

    def observation_space(self, agent):
        idx = self.possible_agents.index(agent)
        return self.observation_spaces[idx]

    def reset(self, seed=None, options=None):
        # Reset step counters
        self.current_step = 0
        self.agent_step_counters = {agent: 0 for agent in self.agents}

        # All agents observe at reset (step 0)
        self.active_agents = self.agents.copy()

        observations = {
            agent: np.random.rand(
                *self.observation_spaces[self.possible_agents.index(agent)].shape
            )
            for agent in self.active_agents
        }

        infos = {agent: {} for agent in self.active_agents}
        for agent in self.active_agents:
            infos[agent]["env_defined_actions"] = None

        # Always provide env_defined_actions for agent_0 if active
        if "agent_0" in self.active_agents:
            infos["agent_0"]["env_defined_actions"] = np.array([1])

        return observations, infos

    def step(self, action):
        # Increment the global step counter
        self.current_step += 1

        # Increment step counters for each agent
        for agent in self.agents:
            self.agent_step_counters[agent] += 1

        # Determine which agents should observe based on their frequency
        self.active_agents = [
            agent
            for agent in self.agents
            if self.agent_step_counters[agent] % self.observation_frequencies[agent]
            == 0
        ]

        observations = {
            agent: np.random.rand(
                *self.observation_spaces[self.possible_agents.index(agent)].shape
            )
            for agent in self.active_agents
        }

        rewards = {agent: np.random.randint(0, 5) for agent in action.keys()}

        # Different grouped agents done at different times
        dones = {}
        for agent in self.active_agents:
            if agent in ["agent_0", "agent_1"]:
                dones[agent] = self.current_step >= 30
            else:
                dones[agent] = self.current_step >= 40

        truncated = {agent: False for agent in self.active_agents}
        infos = {agent: {} for agent in self.active_agents}

        return observations, rewards, dones, truncated, infos


@pytest.fixture(scope="function")
def setup_rs_norm():
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
    mock_agent = MagicMock(spec=RLAlgorithm)
    mock_agent.observation_space = observation_space
    mock_agent.action_space = spaces.Discrete(2)
    mock_agent.device = "cpu"
    mock_agent.training = True

    wrapper = RSNorm(mock_agent)
    return wrapper, mock_agent


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
def setup_rs_norm_multi_agent():
    observation_space = {
        "agent_1": spaces.Box(low=-1.0, high=1.0, shape=(3,)),
        "other_agent_1": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
    }
    mock_agent = MagicMock(spec=MultiAgentRLAlgorithm)
    mock_agent.observation_space = observation_space
    mock_agent.action_space = {
        "agent_1": spaces.Discrete(2),
        "other_agent_1": spaces.Discrete(2),
    }
    mock_agent.device = "cpu"
    mock_agent.training = True

    wrapper = RSNorm(mock_agent)
    return wrapper, mock_agent


def test_set_get_state(setup_rs_norm):
    wrapper, mock_agent = setup_rs_norm
    state = wrapper.__getstate__()
    wrapper_2 = RSNorm(mock_agent)
    wrapper_2.__setstate__(state)
    assert torch.allclose(wrapper.obs_rms.mean, wrapper_2.obs_rms.mean, atol=1e-2)
    assert torch.allclose(wrapper.obs_rms.var, wrapper_2.obs_rms.var, atol=1e-2)
    assert wrapper.obs_rms.epsilon == wrapper_2.obs_rms.epsilon


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
        "other_agent_1": torch.tensor([1.0, 2.0]),
    }
    wrapper.obs_rms["agent_1"].mean = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms["agent_1"].var = torch.tensor([1.0, 1.0, 1.0])
    wrapper.obs_rms["agent_1"].epsilon = 1e-4
    wrapper.obs_rms["other_agent_1"].mean = torch.tensor([1.0, 1.0])
    wrapper.obs_rms["other_agent_1"].var = torch.tensor([1.0, 1.0])
    wrapper.obs_rms["other_agent_1"].epsilon = 1e-4

    normalized_obs = wrapper._normalize_observation(obs)
    expected_obs = {
        "agent_1": (obs["agent_1"] - wrapper.obs_rms["agent_1"].mean)
        / torch.sqrt(
            wrapper.obs_rms["agent_1"].var + wrapper.obs_rms["agent_1"].epsilon
        ),
        "other_agent_1": (obs["other_agent_1"] - wrapper.obs_rms["other_agent_1"].mean)
        / torch.sqrt(
            wrapper.obs_rms["other_agent_1"].var
            + wrapper.obs_rms["other_agent_1"].epsilon
        ),
    }
    assert torch.allclose(normalized_obs["agent_1"], expected_obs["agent_1"], atol=1e-2)
    assert torch.allclose(
        normalized_obs["other_agent_1"], expected_obs["other_agent_1"], atol=1e-2
    )


@pytest.mark.parametrize("observation_space", ["vector_space", "dict_space"])
def test_rsnorm_get_action(observation_space, request):
    observation_space = request.getfixturevalue(observation_space)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
    agent = DDPG(observation_space, action_space)
    wrapper = RSNorm(agent)
    obs = wrapper.observation_space.sample()
    action = wrapper.get_action(obs)
    assert action.shape == (1,) + agent.action_space.shape


@pytest.mark.parametrize(
    "observation_space",
    ["vector_space", "discrete_space", "multidiscrete_space", "dict_space"],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_rsnorm_learn(observation_space, vector_space, request, accelerator):
    observation_space = request.getfixturevalue(observation_space)
    action_space = vector_space
    batch_size = 4
    policy_freq = 4

    # Create an instance of the ddpg class
    ddpg = DDPG(
        observation_space,
        action_space,
        batch_size=batch_size,
        policy_freq=policy_freq,
        accelerator=accelerator,
    )
    ddpg = RSNorm(ddpg)

    # Copy state dict before learning - should be different to after updating weights
    actor = ddpg.actor
    actor_target = ddpg.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(ddpg.actor.state_dict()))
    critic = ddpg.critic
    critic_target = ddpg.critic_target
    critic_pre_learn_sd = str(copy.deepcopy(ddpg.critic.state_dict()))

    for i in range(policy_freq * 2):
        # Create a batch of experiences & learn
        device = accelerator.device if accelerator else "cpu"
        experiences = get_experiences_batch(
            observation_space, action_space, batch_size, device
        )
        ddpg.scores.append(0)
        actor_loss, critic_loss = ddpg.learn(experiences)

    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
    assert critic_loss >= 0.0
    assert actor == ddpg.actor
    assert actor_target == ddpg.actor_target
    assert actor_pre_learn_sd != str(ddpg.actor.state_dict())
    assert critic == ddpg.critic
    assert critic_target == ddpg.critic_target
    assert critic_pre_learn_sd != str(ddpg.critic.state_dict())


# Clones the agent and returns an identical agent.
def test_rsnorm_clone_returns_identical_agent(vector_space):
    observation_space = vector_space
    action_space = copy.deepcopy(vector_space)
    ddpg_norm = RSNorm(
        DDPG(observation_space=observation_space, action_space=action_space)
    )
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ddpg.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), ddpg.actor_target.state_dict()
    )
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ddpg.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.critic_target.state_dict(), ddpg.critic_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), ddpg.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_optimizer.state_dict(), ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores
    assert clone_agent.tensor_attribute == ddpg.tensor_attribute

    accelerator = Accelerator()
    ddpg_norm = RSNorm(
        DDPG(
            observation_space=observation_space,
            action_space=action_space,
            accelerator=accelerator,
        )
    )
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ddpg.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), ddpg.actor_target.state_dict()
    )
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ddpg.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.critic_target.state_dict(), ddpg.critic_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), ddpg.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_optimizer.state_dict(), ddpg.critic_optimizer.state_dict()
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ddpg.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), ddpg.actor_target.state_dict()
    )
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ddpg.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.critic_target.state_dict(), ddpg.critic_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), ddpg.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_optimizer.state_dict(), ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores


def test_rsnorm_save_load_checkpoint(tmp_path, vector_space):
    observation_space = vector_space
    action_space = copy.deepcopy(vector_space)

    ddpg_norm = RSNorm(
        DDPG(observation_space=observation_space, action_space=action_space)
    )
    checkpoint_path = os.path.join(tmp_path, "checkpoint.pth")
    ddpg_norm.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill, weights_only=False)

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
    assert_state_dicts_equal(ddpg.actor.state_dict(), ddpg.actor_target.state_dict())
    assert_state_dicts_equal(ddpg.critic.state_dict(), ddpg.critic_target.state_dict())
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
    assert_state_dicts_equal(ddpg.actor.state_dict(), ddpg.actor_target.state_dict())
    assert_state_dicts_equal(ddpg.critic.state_dict(), ddpg.critic_target.state_dict())
    assert ddpg.batch_size == 64
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 1e-3
    assert ddpg.mut is None
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]


@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_ippo_custom_training_with_async_env(
    device, ma_vector_space, ma_discrete_space, compile_mode, num_envs
):
    # Create async environment with agents that return observations asynchronously
    vectorized = num_envs > 1
    observation_spaces = ma_vector_space
    action_spaces = ma_discrete_space
    if vectorized:
        env = make_multi_agent_vect_envs(
            DummyMultiEnvAsync,
            num_envs=num_envs,
            **dict(observation_spaces=observation_spaces, action_spaces=action_spaces),
        )
    else:
        env = DummyMultiEnvAsync(observation_spaces, action_spaces)

    agent_ids = ["agent_0", "agent_1", "other_agent_0"]

    # Initialize IPPO agent
    agent = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
        batch_size=64,
        lr=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        torch_compiler=compile_mode,
    )

    async_agent = AsyncAgentsWrapper(agent)

    # Custom training loop for multiple iterations
    for _ in range(5):
        # Reset environment
        observations, infos = env.reset()

        states = {agent_id: [] for agent_id in agent_ids}
        actions = {agent_id: [] for agent_id in agent_ids}
        log_probs = {agent_id: [] for agent_id in agent_ids}
        rewards = {agent_id: [] for agent_id in agent_ids}
        dones = {agent_id: [] for agent_id in agent_ids}
        values = {agent_id: [] for agent_id in agent_ids}

        done = {
            agent_id: np.zeros((num_envs,), dtype=np.int8) for agent_id in agent_ids
        }

        # Collect experiences for multiple steps
        max_steps = 105
        for _ in range(max_steps):
            # Get actions for current active agents
            action_dict, logprob_dict, _, value_dict = async_agent.get_action(
                observations, infos
            )

            # Verify actions are only for active agents
            assert all(agent_id in observations for agent_id in action_dict)

            # Step the environment
            next_observations, reward_dict, terminated, truncated, next_infos = (
                env.step(action_dict)
            )

            # Store experiences for active agents
            for agent_id in observations:
                states[agent_id].append(observations[agent_id])
                actions[agent_id].append(action_dict[agent_id])
                log_probs[agent_id].append(logprob_dict[agent_id])
                values[agent_id].append(value_dict[agent_id])
                dones[agent_id].append(done[agent_id])
                rewards[agent_id].append(reward_dict[agent_id])

            next_dones = {}
            for agent_id in terminated:
                term = terminated[agent_id]
                trunc = truncated[agent_id]

                # Process asynchronous dones
                if vectorized:
                    mask = ~(np.isnan(term) | np.isnan(trunc))
                    result = np.full_like(mask, np.nan, dtype=float)
                    result[mask] = np.logical_or(term[mask], trunc[mask])

                    next_dones[agent_id] = result
                else:
                    next_dones[agent_id] = np.array(
                        [np.logical_or(term, trunc)]
                    ).astype(np.int8)

            # Update for next step
            observations = next_observations
            done = next_dones
            infos = next_infos

            # Break if all agents report done
            for idx, agent_dones in enumerate(zip(*next_dones.values())):
                if all(agent_dones):
                    if not vectorized:
                        observations, info = env.reset()

                    done = {
                        agent_id: np.zeros(num_envs) for agent_id in agent.agent_ids
                    }

        # Skip learning if no experiences collected
        if not any(states.values()):
            continue

        # Create experience tuple for learning
        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_observations,  # next_states
            next_dones,
        )

        # Train on collected experiences if we have any
        if any(len(states[agent_id]) > 0 for agent_id in states):
            loss_info = async_agent.learn(experiences)

            # Verify that learning worked for at least one agent
            assert any(agent_id in loss_info for agent_id in agent.shared_agent_ids)

    # Final test: verify agent can handle completely different set of active agents
    test_observations, test_infos = env.reset()
    test_actions, _, _, _ = async_agent.get_action(test_observations, test_infos)
    assert all(agent_id in test_observations for agent_id in test_actions)
