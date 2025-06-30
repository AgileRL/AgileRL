import copy
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from torch._dynamo import OptimizedModule

from agilerl.algorithms.ippo import IPPO
from agilerl.modules import EvolvableMLP, ModuleDict
from agilerl.modules.custom_components import GumbelSoftmax
from agilerl.networks.actors import StochasticActor
from agilerl.networks.value_networks import ValueNetwork
from agilerl.utils.evolvable_networks import get_default_encoder_config
from agilerl.utils.utils import make_multi_agent_vect_envs
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    assert_not_equal_state_dict,
    assert_state_dicts_equal,
    get_sample_from_space,
)


class DummyMultiEnv(ParallelEnv):
    def __init__(self, observation_spaces, action_spaces):
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.agents = ["agent_0", "agent_1", "other_agent_0"]
        self.possible_agents = ["agent_0", "agent_1", "other_agent_0"]
        self.metadata = None
        self.render_mode = None

    def action_space(self, agent):
        return Discrete(self.action_spaces[0].n)

    def observation_space(self, agent):
        return Box(0, 1, self.observation_spaces[0].shape)

    def reset(self, seed=None, options=None):
        return {
            agent: np.random.rand(*self.observation_spaces[i].shape)
            for i, agent in enumerate(self.agents)
        }, {
            "agent_0": {"env_defined_actions": np.array([1])},
            "agent_1": {"env_defined_actions": None},
            "other_agent_0": {"env_defined_actions": None},
        }

    def step(self, action):
        return (
            {
                agent: np.random.rand(*self.observation_spaces[i].shape)
                for i, agent in enumerate(self.agents)
            },
            {agent: np.random.randint(0, 5) for agent in self.agents},
            {agent: 1 for agent in self.agents},
            {agent: np.random.randint(0, 2) for agent in self.agents},
            self.reset()[1],
        )


class MultiAgentCNNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(3, 3), stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(288, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.output_activation = GumbelSoftmax()

    def forward(self, state_tensor):
        x = self.relu(self.conv1(state_tensor))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        action_dist = self.output_activation(self.fc2(x))
        return action_dist


class MultiAgentCNNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(3, 3), stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(288, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, state_tensor):
        x = self.relu(self.conv1(state_tensor))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class DummyStochasticActor(StochasticActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def no_sync(self):
        class DummyNoSync:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass  # Add cleanup or handling if needed

        return DummyNoSync()


class DummyValueNetwork(ValueNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def no_sync(self):
        class DummyNoSync:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass  # Add cleanup or handling if needed

        return DummyNoSync()


@pytest.fixture(scope="function")
def mlp_actor(observation_spaces, action_spaces, request):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    return nn.Sequential(
        nn.Linear(observation_spaces[0].shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, action_spaces[0].n),
        nn.Softmax(dim=-1),
    )


@pytest.fixture(scope="function")
def mlp_critic(observation_spaces, request):
    observation_spaces = request.getfixturevalue(observation_spaces)
    return nn.Sequential(
        nn.Linear(observation_spaces[0].shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


@pytest.fixture(scope="function")
def cnn_actor():
    return MultiAgentCNNActor()


@pytest.fixture(scope="function")
def cnn_critic():
    return MultiAgentCNNCritic()


@pytest.fixture(scope="module")
def mocked_accelerator():
    MagicMock(spec=Accelerator)


@pytest.fixture(scope="function")
def accelerated_experiences(
    batch_size, observation_spaces, action_spaces, agent_ids, device, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    one_hot = all(isinstance(space, Discrete) for space in observation_spaces)
    discrete_actions = all(isinstance(space, Discrete) for space in action_spaces)
    state_size = (
        observation_spaces[0].shape if not one_hot else (observation_spaces[0].n,)
    )
    action_size = 1 if discrete_actions else action_spaces[0].shape[0]
    if one_hot:
        states = {
            agent: np.random.randint(0, state_size[0], (batch_size, 1))
            for agent in agent_ids
        }
    else:
        states = {
            agent: np.random.randn(batch_size, *state_size) for agent in agent_ids
        }

    if discrete_actions:
        actions = {
            agent: np.random.randint(0, action_size, (batch_size,))
            for agent in agent_ids
        }
    else:
        actions = {
            agent: np.random.randn(batch_size, action_size) for agent in agent_ids
        }
    log_probs = {agent: np.random.randn(batch_size, 1) for agent in agent_ids}
    rewards = {agent: np.random.randn(batch_size, 1) for agent in agent_ids}
    dones = {agent: np.random.randint(0, 2, (batch_size, 1)) for agent in agent_ids}
    values = {agent: np.random.randn(batch_size, 1) for agent in agent_ids}
    if one_hot:
        next_state = {
            agent: np.random.randint(0, state_size[0], (1,)) for agent in agent_ids
        }
    else:
        next_state = {agent: np.random.randn(*state_size) for agent in agent_ids}

    next_done = {agent: np.random.randint(0, 2, (1,)) for agent in agent_ids}

    return states, actions, log_probs, rewards, dones, values, next_state, next_done


@pytest.fixture(scope="function")
def experiences(
    batch_size, observation_spaces, action_spaces, agent_ids, device, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    one_hot = all(isinstance(space, Discrete) for space in observation_spaces)
    discrete_actions = all(isinstance(space, Discrete) for space in action_spaces)
    state_size = (
        observation_spaces[0].shape if not one_hot else (observation_spaces[0].n,)
    )
    action_size = 1 if discrete_actions else action_spaces[0].shape[0]
    if one_hot:
        states = {
            agent: np.random.randint(0, state_size[0], (batch_size, 1))
            for agent in agent_ids
        }
    else:
        states = {
            agent: np.random.randn(batch_size, *state_size) for agent in agent_ids
        }

    if discrete_actions:
        actions = {
            agent: np.random.randint(0, action_size, (batch_size, 1))
            for agent in agent_ids
        }
    else:
        actions = {
            agent: np.random.randn(batch_size, action_size) for agent in agent_ids
        }
    log_probs = {agent: np.random.randn(batch_size, 1) for agent in agent_ids}
    rewards = {agent: np.random.randn(batch_size) for agent in agent_ids}
    dones = {agent: np.random.randint(0, 2, (batch_size,)) for agent in agent_ids}
    values = {agent: np.random.randn(batch_size, 1) for agent in agent_ids}
    if one_hot:
        next_state = {
            agent: np.random.randint(0, state_size[0], (1,)) for agent in agent_ids
        }
    else:
        next_state = {agent: np.random.randn(*state_size) for agent in agent_ids}

    next_done = {agent: np.random.randint(0, 2, (1,)) for agent in agent_ids}

    return states, actions, log_probs, rewards, dones, values, next_state, next_done


@pytest.fixture(scope="function")
def vectorized_experiences(
    batch_size, vect_dim, observation_spaces, action_spaces, agent_ids, device, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    one_hot = all(isinstance(space, Discrete) for space in observation_spaces)
    discrete_actions = all(isinstance(space, Discrete) for space in action_spaces)
    state_size = (
        observation_spaces[0].shape if not one_hot else (observation_spaces[0].n,)
    )
    action_size = 1 if discrete_actions else action_spaces[0].shape[0]
    if one_hot:
        states = {
            agent: np.random.randint(0, state_size[0], (batch_size, vect_dim, 1))
            for agent in agent_ids
        }
    else:
        states = {
            agent: np.random.randn(batch_size, vect_dim, *state_size)
            for agent in agent_ids
        }

    if discrete_actions:
        actions = {
            agent: np.random.randint(0, action_size, (batch_size, vect_dim, 1))
            for agent in agent_ids
        }
    else:
        actions = {
            agent: np.random.randn(batch_size, vect_dim, action_size)
            for agent in agent_ids
        }
    log_probs = {agent: np.random.randn(batch_size, vect_dim, 1) for agent in agent_ids}
    rewards = {agent: np.random.randn(batch_size, vect_dim) for agent in agent_ids}
    dones = {
        agent: np.random.randint(0, 2, (batch_size, vect_dim)) for agent in agent_ids
    }
    values = {agent: np.random.randn(batch_size, vect_dim, 1) for agent in agent_ids}
    if one_hot:
        next_state = {
            agent: np.random.randint(
                0,
                state_size[0],
                (
                    vect_dim,
                    1,
                ),
            )
            for agent in agent_ids
        }
    else:
        next_state = {
            agent: np.random.randn(vect_dim, *state_size) for agent in agent_ids
        }

    next_done = {agent: np.random.randint(0, 2, (1, vect_dim)) for agent in agent_ids}

    return states, actions, log_probs, rewards, dones, values, next_state, next_done


@pytest.mark.parametrize("sum_score", [True, False])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("observation_spaces", ["ma_vector_space", "ma_image_space"])
@pytest.mark.parametrize("vectorized", [False, True])
def test_loop(
    device,
    sum_score,
    compile_mode,
    observation_spaces,
    vectorized,
    ma_discrete_space,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    if vectorized:
        env = make_multi_agent_vect_envs(
            DummyMultiEnv,
            2,
            **dict(
                observation_spaces=observation_spaces, action_spaces=ma_discrete_space
            ),
        )
    else:
        env = DummyMultiEnv(observation_spaces, ma_discrete_space)

    ippo = IPPO(
        observation_spaces,
        ma_discrete_space,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        device=device,
        torch_compiler=compile_mode,
    )
    mean_score = ippo.test(env, max_steps=10, sum_scores=sum_score)
    if sum_score:
        assert isinstance(mean_score, float)
    else:
        assert isinstance(mean_score, np.ndarray)
        assert len(mean_score) == 2


@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("wrap", [True, False])
def test_ippo_clone_returns_identical_agent(
    accelerator_flag, wrap, compile_mode, observation_spaces, ma_discrete_space, request
):
    # Clones the agent and returns an identical copy.
    observation_spaces = request.getfixturevalue(observation_spaces)
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    index = 0
    batch_size = 64
    lr = 1e-4
    learn_step = 2048
    gamma = 0.99
    gae_lambda = 0.95
    mut = None
    action_std_init = 0.0
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    update_epochs = 4
    actor_networks = None
    critic_networks = None
    device = "cpu"
    accelerator = Accelerator(device_placement=False) if accelerator_flag else None

    ippo = IPPO(
        observation_spaces,
        ma_discrete_space,
        agent_ids,
        index=index,
        batch_size=batch_size,
        lr=lr,
        learn_step=learn_step,
        gamma=gamma,
        gae_lambda=gae_lambda,
        mut=mut,
        action_std_init=action_std_init,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        update_epochs=update_epochs,
        actor_networks=actor_networks,
        critic_networks=critic_networks,
        device=device,
        accelerator=accelerator,
        wrap=wrap,
        torch_compiler=compile_mode,
    )

    clone_agent = ippo.clone(wrap=wrap)

    assert isinstance(clone_agent, IPPO)
    assert clone_agent.observation_spaces == ippo.observation_spaces
    assert clone_agent.action_spaces == ippo.action_spaces
    assert clone_agent.n_agents == ippo.n_agents
    assert clone_agent.agent_ids == ippo.agent_ids
    assert clone_agent.shared_agent_ids == ippo.shared_agent_ids
    assert clone_agent.grouped_agents == ippo.grouped_agents
    assert clone_agent.unique_observation_spaces == ippo.unique_observation_spaces
    assert clone_agent.unique_action_spaces == ippo.unique_action_spaces
    assert clone_agent.grouped_spaces == ippo.grouped_spaces
    assert clone_agent.n_unique_agents == ippo.n_unique_agents
    assert clone_agent.index == ippo.index
    assert clone_agent.batch_size == ippo.batch_size
    assert clone_agent.lr == ippo.lr
    assert clone_agent.learn_step == ippo.learn_step
    assert clone_agent.gamma == ippo.gamma
    assert clone_agent.gae_lambda == ippo.gae_lambda
    assert clone_agent.action_std_init == ippo.action_std_init
    assert clone_agent.clip_coef == ippo.clip_coef
    assert clone_agent.ent_coef == ippo.ent_coef
    assert clone_agent.vf_coef == ippo.vf_coef
    assert clone_agent.max_grad_norm == ippo.max_grad_norm
    assert clone_agent.target_kl == ippo.target_kl
    assert clone_agent.update_epochs == ippo.update_epochs
    assert clone_agent.device == ippo.device
    assert clone_agent.accelerator == ippo.accelerator

    for shared_id in ippo.shared_agent_ids:
        actor = ippo.actors[shared_id]
        clone_actor = clone_agent.actors[shared_id]
        assert_state_dicts_equal(clone_actor.state_dict(), actor.state_dict())

        critic = ippo.critics[shared_id]
        clone_critic = clone_agent.critics[shared_id]
        assert_state_dicts_equal(clone_critic.state_dict(), critic.state_dict())


@pytest.mark.parametrize("compile_mode", [None])
def test_clone_new_index(compile_mode, ma_vector_space, ma_discrete_space):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]

    ippo = IPPO(
        ma_vector_space,
        ma_discrete_space,
        agent_ids,
        torch_compiler=compile_mode,
    )
    clone_agent = ippo.clone(index=100)

    assert clone_agent.index == 100


@pytest.mark.parametrize("compile_mode", [None])
def test_clone_after_learning(compile_mode, ma_vector_space, ma_discrete_space):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    batch_size = 8

    ippo = IPPO(
        ma_vector_space,
        ma_discrete_space,
        agent_ids,
        batch_size=batch_size,
        torch_compiler=compile_mode,
        target_kl=1e-10,
    )

    states = {
        agent_id: np.random.randn(batch_size, ma_vector_space[idx].shape[0])
        for idx, agent_id in enumerate(agent_ids)
    }
    actions = {
        agent_id: np.random.randint(0, ma_discrete_space[idx].n, (batch_size,))
        for idx, agent_id in enumerate(agent_ids)
    }
    log_probs = {agent_id: np.random.randn(batch_size, 1) for agent_id in agent_ids}
    rewards = {agent_id: np.random.randn(batch_size, 1) for agent_id in agent_ids}
    dones = {agent_id: torch.zeros(batch_size, 1) for agent_id in agent_ids}
    values = {agent_id: np.random.randn(batch_size, 1) for agent_id in agent_ids}
    next_state = {
        agent_id: np.random.randn(ma_vector_space[idx].shape[0])
        for idx, agent_id in enumerate(agent_ids)
    }
    next_done = {agent: np.random.randint(0, 2, (1,)) for agent in agent_ids}

    experiences = (
        states,
        actions,
        log_probs,
        rewards,
        dones,
        values,
        next_state,
        next_done,
    )
    ippo.learn(experiences)
    clone_agent = ippo.clone()

    assert isinstance(clone_agent, IPPO)
    assert clone_agent.observation_spaces == ippo.observation_spaces
    assert clone_agent.action_spaces == ippo.action_spaces
    assert clone_agent.n_agents == ippo.n_agents
    assert clone_agent.agent_ids == ippo.agent_ids
    assert clone_agent.index == ippo.index
    assert clone_agent.batch_size == ippo.batch_size
    assert clone_agent.lr == ippo.lr
    assert clone_agent.learn_step == ippo.learn_step
    assert clone_agent.gamma == ippo.gamma
    assert clone_agent.gae_lambda == ippo.gae_lambda
    assert clone_agent.device == ippo.device
    assert clone_agent.accelerator == ippo.accelerator

    for shared_id in ippo.shared_agent_ids:
        actor = ippo.actors[shared_id]
        clone_actor = clone_agent.actors[shared_id]
        assert_state_dicts_equal(clone_actor.state_dict(), actor.state_dict())

        critic = ippo.critics[shared_id]
        clone_critic = clone_agent.critics[shared_id]
        assert_state_dicts_equal(clone_critic.state_dict(), critic.state_dict())

        actor_opt = ippo.actor_optimizers[shared_id]
        clone_actor_opt = clone_agent.actor_optimizers[shared_id]
        assert str(clone_actor_opt) == str(actor_opt)

        critic_opt = ippo.critic_optimizers[shared_id]
        clone_critic_opt = clone_agent.critic_optimizers[shared_id]
        assert str(clone_critic_opt) == str(critic_opt)


@pytest.mark.parametrize("compile_mode", [None])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("agent_ids", [["agent_0", "agent_1", "other_agent_0"]])
@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space"])
def test_ippo_learns_from_experiences_distributed(
    observation_spaces,
    action_spaces,
    agent_ids,
    accelerated_experiences,
    compile_mode,
    batch_size,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    accelerator = Accelerator(device_placement=False)
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )

    for shared_id in ippo.shared_agent_ids:
        actor = ippo.actors[shared_id]
        critic = ippo.critics[shared_id]
        actor.no_sync = no_sync.__get__(actor)
        critic.no_sync = no_sync.__get__(critic)

    actors = ippo.actors
    actors_pre_learn_sd = {
        shared_id: copy.deepcopy(actor.state_dict())
        for shared_id, actor in ippo.actors.items()
    }
    critics = ippo.critics
    critics_pre_learn_sd = {
        shared_id: copy.deepcopy(critic.state_dict())
        for shared_id, critic in ippo.critics.items()
    }

    for _ in range(2):
        ippo.scores.append(0)
        loss = ippo.learn(accelerated_experiences)

    assert isinstance(loss, dict)
    for agent_id in ippo.shared_agent_ids:
        assert agent_id in loss

    for shared_id in ippo.shared_agent_ids:
        old_actor = actors[shared_id]
        updated_actor = ippo.actors[shared_id]
        assert old_actor == updated_actor

        old_critic = critics[shared_id]
        updated_critic = ippo.critics[shared_id]
        assert old_critic == updated_critic

        old_critic_state_dict = critics_pre_learn_sd[shared_id]
        assert_not_equal_state_dict(old_critic_state_dict, updated_critic.state_dict())

        old_actor_state_dict = actors_pre_learn_sd[shared_id]
        assert_not_equal_state_dict(old_actor_state_dict, updated_actor.state_dict())


@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("observation_spaces", ["ma_image_space", "ma_vector_space"])
@pytest.mark.parametrize("agent_ids", [["agent_0", "agent_1", "other_agent_0"]])
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space"])
def test_ippo_learns_from_experiences(
    observation_spaces,
    agent_ids,
    action_spaces,
    experiences,
    batch_size,
    device,
    compile_mode,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )

    actors = ippo.actors
    actors_pre_learn_sd = {
        shared_id: copy.deepcopy(actor.state_dict())
        for shared_id, actor in ippo.actors.items()
    }
    critics = ippo.critics
    critics_pre_learn_sd = {
        shared_id: copy.deepcopy(critic.state_dict())
        for shared_id, critic in ippo.critics.items()
    }

    for _ in range(2):
        ippo.scores.append(0)
        loss = ippo.learn(experiences)

    assert isinstance(loss, dict)
    for shared_id in ippo.shared_agent_ids:
        assert shared_id in loss

        old_actor = actors[shared_id]
        updated_actor = ippo.actors[shared_id]
        assert old_actor == updated_actor

        old_actor_state_dict = actors_pre_learn_sd[shared_id]
        assert_not_equal_state_dict(old_actor_state_dict, updated_actor.state_dict())

        old_critic = critics[shared_id]
        updated_critic = ippo.critics[shared_id]
        assert old_critic == updated_critic

        old_critic_state_dict = critics_pre_learn_sd[shared_id]
        assert_not_equal_state_dict(old_critic_state_dict, updated_critic.state_dict())


@pytest.mark.parametrize("compile_mode", [None])
@pytest.mark.parametrize("vect_dim", [1, 8])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("agent_ids", [["agent_0", "agent_1", "other_agent_0"]])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space", "ma_vector_space"])
@pytest.mark.parametrize("observation_spaces", ["ma_image_space", "ma_vector_space"])
def test_ippo_learns_from_vectorized_experiences(
    agent_ids,
    observation_spaces,
    action_spaces,
    vectorized_experiences,
    batch_size,
    device,
    compile_mode,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
        batch_size=batch_size,
        lr=0.1,
    )

    actors = ippo.actors
    actors_pre_learn_sd = {
        shared_id: copy.deepcopy(actor.state_dict())
        for shared_id, actor in ippo.actors.items()
    }
    critics = ippo.critics
    critics_pre_learn_sd = {
        shared_id: copy.deepcopy(critic.state_dict())
        for shared_id, critic in ippo.critics.items()
    }

    for _ in range(2):
        ippo.scores.append(0)
        loss = ippo.learn(vectorized_experiences)

    assert isinstance(loss, dict)
    for agent_id in ippo.shared_agent_ids:
        assert agent_id in loss

    for shared_id in ippo.shared_agent_ids:
        old_actor = actors[shared_id]
        updated_actor = ippo.actors[shared_id]
        assert old_actor == updated_actor

        old_actor_state_dict = actors_pre_learn_sd[shared_id]
        assert_not_equal_state_dict(old_actor_state_dict, updated_actor.state_dict())

        old_critic = critics[shared_id]
        updated_critic = ippo.critics[shared_id]
        assert old_critic == updated_critic

        old_critic_state_dict = critics_pre_learn_sd[shared_id]
        assert_not_equal_state_dict(old_critic_state_dict, updated_critic.state_dict())


def test_ippo_learns_from_hardcoded_vectorized_experiences_mlp(
    ma_vector_space,
    device,
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    states = {
        agent: np.array(
            [
                [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]],
            ]
        )
        * i
        for i, agent in enumerate(agent_ids)
    }
    actions = {
        agent: np.array(
            [
                [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]],
            ]
        )
        * i
        for i, agent in enumerate(agent_ids)
    }
    log_probs = {
        agent: np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]) * i
        for i, agent in enumerate(agent_ids)
    }
    rewards = {
        agent: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * i
        for i, agent in enumerate(agent_ids)
    }
    dones = {
        agent: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for i, agent in enumerate(agent_ids)
    }
    values = {
        agent: np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]) * i
        for i, agent in enumerate(agent_ids)
    }
    next_state = {
        agent: np.array(
            [[4, 4, 4, 4, 4, 4], [7, 7, 7, 7, 7, 7], [10, 10, 10, 10, 10, 10]]
        )
        * i
        for i, agent in enumerate(agent_ids)
    }
    next_done = {agent: np.array([[0, 1, 0]]) for i, agent in enumerate(agent_ids)}

    experiences = (
        states,
        actions,
        log_probs,
        rewards,
        dones,
        values,
        next_state,
        next_done,
    )
    ippo = IPPO(
        observation_spaces=ma_vector_space,
        action_spaces=ma_vector_space,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=None,
        batch_size=32,
    )

    actors = ippo.actors
    actors_pre_learn_sd = {
        shared_id: copy.deepcopy(actor.state_dict())
        for shared_id, actor in ippo.actors.items()
    }
    critics = ippo.critics
    critics_pre_learn_sd = {
        shared_id: copy.deepcopy(critic.state_dict())
        for shared_id, critic in ippo.critics.items()
    }

    for _ in range(4):
        ippo.scores.append(0)
        loss = ippo.learn(experiences)

    assert isinstance(loss, dict)
    for agent_id in ippo.shared_agent_ids:
        assert agent_id in loss

    for shared_id in ippo.shared_agent_ids:
        old_actor = actors[shared_id]
        updated_actor = ippo.actors[shared_id]
        assert old_actor == updated_actor

        old_actor_state_dict = actors_pre_learn_sd[shared_id]
        assert_not_equal_state_dict(old_actor_state_dict, updated_actor.state_dict())

        old_critic = critics[shared_id]
        updated_critic = ippo.critics[shared_id]
        assert old_critic == updated_critic

        old_critic_state_dict = critics_pre_learn_sd[shared_id]
        assert_not_equal_state_dict(old_critic_state_dict, updated_critic.state_dict())


def no_sync(self):
    class DummyNoSync:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass  # Add cleanup or handling if needed

    return DummyNoSync()


@pytest.mark.parametrize(
    "action_spaces",
    [
        "ma_discrete_space",
        "ma_vector_space",
    ],
)
@pytest.mark.parametrize("compile_mode", [None])
def test_ippo_get_action_agent_masking(
    ma_vector_space, action_spaces, device, compile_mode, request
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {agent: np.random.randn(*ma_vector_space[0].shape) for agent in agent_ids}
    action_spaces = request.getfixturevalue(action_spaces)
    discrete_actions = all(
        isinstance(space, spaces.Discrete) for space in action_spaces
    )
    if discrete_actions:
        info = {
            "agent_0": {"env_defined_actions": 1},
            "agent_1": {"env_defined_actions": None},
            "other_agent_0": {"env_defined_actions": None},
        }
    else:
        info = {
            "agent_0": {"env_defined_actions": np.array([0, 1, 0, 1, 0, 1])},
            "agent_1": {"env_defined_actions": None},
            "other_agent_0": {"env_defined_actions": None},
        }

    # Define the ippo agent
    ippo = IPPO(
        observation_spaces=ma_vector_space,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )

    # Get the action
    actions, _, _, _ = ippo.get_action(obs=state, infos=info)

    if discrete_actions:
        assert np.array_equal(actions["agent_0"], np.array([[1]])), actions["agent_0"]
    else:
        assert np.array_equal(
            actions["agent_0"], np.array([[0, 1, 0, 1, 0, 1]])
        ), actions["agent_0"]


@pytest.mark.parametrize(
    "observation_spaces",
    [
        "ma_discrete_space",
        "ma_vector_space",
        "ma_image_space",
        "ma_dict_space",
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        "ma_discrete_space",
        "ma_vector_space",
        "ma_multidiscrete_space",
        "ma_multibinary_space",
    ],
)
@pytest.mark.parametrize("action_batch_size", [None, 16])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_ippo_get_action(
    observation_spaces,
    action_spaces,
    device,
    compile_mode,
    accelerator,
    action_batch_size,
    request,
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)

    state = {}
    for agent_id, obs_space in zip(agent_ids, observation_spaces):
        sample_state = get_sample_from_space(obs_space)

        if (
            not isinstance(obs_space, (spaces.Dict, spaces.Tuple))
            and action_batch_size is not None
        ):
            sample_state = np.stack([sample_state] * action_batch_size)
        else:
            action_batch_size = None

        state[agent_id] = sample_state

    info = None
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
        action_batch_size=action_batch_size,
    )
    actions, log_probs, dist_entropy, state_values = ippo.get_action(
        obs=state, infos=info
    )

    # Check action shapes
    assert isinstance(actions, dict)
    assert isinstance(log_probs, dict)

    for agent_id in agent_ids:
        assert agent_id in actions
        assert agent_id in log_probs
        assert agent_id in dist_entropy
        assert agent_id in state_values


@pytest.mark.parametrize(
    "observation_spaces",
    [
        "ma_vector_space",
        "ma_image_space",
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        "ma_vector_space",
        "ma_discrete_space",
    ],
)
@pytest.mark.parametrize("compile_mode", [None])
def test_ippo_get_action_vectorized(
    observation_spaces, action_spaces, device, compile_mode, request
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    vect_dim = 2
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    if all(isinstance(space, spaces.Discrete) for space in observation_spaces):
        state = {
            agent: np.random.randint(0, observation_spaces[idx].n, (vect_dim, 1))
            for idx, agent in enumerate(agent_ids)
        }
        info = None
    else:
        state = {
            agent: np.random.randn(vect_dim, *observation_spaces[idx].shape).astype(
                np.float32
            )
            for idx, agent in enumerate(agent_ids)
        }
        info = {agent: {} for agent in agent_ids}

    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )
    actions, log_probs, dist_entropy, state_values = ippo.get_action(
        obs=state, infos=info
    )

    # Check action shapes
    assert isinstance(actions, dict)
    assert isinstance(log_probs, dict)

    for agent_id in agent_ids:
        assert agent_id in actions
        assert agent_id in log_probs
        assert agent_id in dist_entropy
        assert agent_id in state_values


def test_ippo_get_action_action_masking_exception(
    ma_vector_space, ma_discrete_space, device
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: {
            "observation": np.random.randn(*ma_vector_space[idx].shape),
            "action_mask": [0, 1, 0, 1],
        }
        for idx, agent in enumerate(agent_ids)
    }
    ippo = IPPO(
        observation_spaces=ma_vector_space,
        action_spaces=ma_discrete_space,
        agent_ids=agent_ids,
        device=device,
    )
    with pytest.raises(AssertionError):
        actions, log_probs, dist_entropy, state_values = ippo.get_action(obs=state)


def test_ippo_get_action_action_masking(ma_vector_space, ma_discrete_space, device):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: np.random.randn(*ma_vector_space[idx].shape).astype(np.float32)
        for idx, agent in enumerate(agent_ids)
    }
    info = {
        agent: {
            "action_mask": [0, 1],
        }
        for agent in agent_ids
    }
    ippo = IPPO(
        observation_spaces=ma_vector_space,
        action_spaces=ma_discrete_space,
        agent_ids=agent_ids,
        device=device,
    )
    actions, _, _, _ = ippo.get_action(obs=state, infos=info)

    # With action mask [0, 1, 0, 1], actions should be index 1 or 3
    for agent_id, action_array in actions.items():
        for action in action_array:
            assert action in [1, 3]


@pytest.mark.parametrize(
    "mode", (None, 0, False, "default", "reduce-overhead", "max-autotune")
)
def test_ippo_init_torch_compiler_no_error(ma_vector_space, ma_discrete_space, mode):
    ippo = IPPO(
        observation_spaces=ma_vector_space,
        action_spaces=ma_discrete_space,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_compiler=mode,
    )
    if isinstance(mode, str):
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule)
            for a in ippo.actors.values()
        )
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule)
            for a in ippo.critics.values()
        )
        assert ippo.torch_compiler == mode
    else:
        assert isinstance(ippo, IPPO)


@pytest.mark.parametrize("mode", (1, True, "max-autotune-no-cudagraphs"))
def test_ippo_init_torch_compiler_error(
    mode, ma_vector_space, ma_discrete_space, device
):
    err_string = (
        "Choose between torch compiler modes: "
        "default, reduce-overhead, max-autotune or None"
    )
    with pytest.raises(AssertionError, match=err_string):
        IPPO(
            observation_spaces=ma_vector_space,
            action_spaces=ma_discrete_space,
            agent_ids=["agent_0", "agent_1", "other_agent_0"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_compiler=mode,
        )


@pytest.mark.parametrize(
    "observation_spaces",
    [
        "ma_vector_space",
        "ma_image_space",
    ],
)
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None])
def test_initialize_ippo_with_net_config(
    accelerator_flag,
    observation_spaces,
    ma_discrete_space,
    device,
    compile_mode,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    batch_size = 64
    accelerator = Accelerator() if accelerator_flag else None

    net_config = {
        "encoder_config": get_default_encoder_config(observation_spaces[0]),
        "head_config": {"hidden_size": [32]},
    }
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=ma_discrete_space,
        agent_ids=agent_ids,
        net_config=net_config,
        accelerator=accelerator,
        device=device,
        torch_compiler=compile_mode,
        target_kl=0.5,
    )

    assert ippo.observation_spaces == observation_spaces
    assert ippo.action_spaces == ma_discrete_space
    assert ippo.n_agents == len(agent_ids)
    assert ippo.agent_ids == agent_ids
    assert ippo.batch_size == batch_size
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]
    assert ippo.target_kl == 0.5

    expected_actor_cls = (
        OptimizedModule
        if compile_mode is not None and accelerator is None
        else StochasticActor
    )
    expected_critic_cls = (
        OptimizedModule
        if compile_mode is not None and accelerator is None
        else ValueNetwork
    )
    assert all(isinstance(actor, expected_actor_cls) for actor in ippo.actors.values())
    assert all(
        isinstance(critic, expected_critic_cls) for critic in ippo.critics.values()
    )

    expected_opt_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    assert all(
        isinstance(actor_optimizer, expected_opt_cls)
        for actor_optimizer in ippo.actor_optimizers.values()
    )
    assert all(
        isinstance(critic_optimizer, expected_opt_cls)
        for critic_optimizer in ippo.critic_optimizers.values()
    )
    assert isinstance(ippo.criterion, nn.MSELoss)


# TODO: This will be deprecated in the future
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None])
@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space"])
def test_initialize_ippo_with_mlp_networks(
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    accelerator_flag,
    device,
    compile_mode,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    accelerator = Accelerator() if accelerator_flag else None
    evo_actors = ModuleDict(
        {
            "agent": MakeEvolvable(
                network=mlp_actor, input_tensor=torch.randn(1, 6), device=device
            ),
            "other_agent": MakeEvolvable(
                network=mlp_actor, input_tensor=torch.randn(1, 6), device=device
            ),
        }
    )
    evo_critics = ModuleDict(
        {
            "agent": MakeEvolvable(
                network=mlp_critic, input_tensor=torch.randn(1, 6), device=device
            ),
            "other_agent": MakeEvolvable(
                network=mlp_critic, input_tensor=torch.randn(1, 6), device=device
            ),
        }
    )
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in ippo.actors.values())
        assert all(
            isinstance(critic, OptimizedModule) for critic in ippo.critics.values()
        )
    else:
        assert all(isinstance(actor, MakeEvolvable) for actor in ippo.actors.values())
        assert all(
            isinstance(critic, MakeEvolvable) for critic in ippo.critics.values()
        )

    assert ippo.observation_spaces == observation_spaces
    assert ippo.action_spaces == action_spaces
    assert ippo.n_agents == 3
    assert ippo.agent_ids == agent_ids
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in ippo.actor_optimizers.values()
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in ippo.critic_optimizers.values()
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in ippo.actor_optimizers.values()
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in ippo.critic_optimizers.values()
        )
    assert isinstance(ippo.criterion, nn.MSELoss)


@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space"])
def test_initialize_ippo_with_mlp_networks_gumbel_softmax(
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    device,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    net_config = {
        "encoder_config": {
            "hidden_size": [64, 64],
            "min_hidden_layers": 1,
            "max_hidden_layers": 3,
            "min_mlp_nodes": 64,
            "max_mlp_nodes": 500,
            "activation": "ReLU",
            "init_layers": False,
        },
        "head_config": {
            "output_activation": "GumbelSoftmax",
            "activation": "ReLU",
            "hidden_size": [64, 64],
            "init_layers": False,
        },
    }
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        net_config=net_config,
        device=device,
        torch_compiler="reduce-overhead",
    )
    assert ippo.torch_compiler == "reduce-overhead"


# TODO: This will be deprecated in the future
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_ippo_with_cnn_networks(
    cnn_actor,
    cnn_critic,
    ma_image_space,
    ma_discrete_space,
    accelerator_flag,
    device,
    compile_mode,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    evo_actors = ModuleDict(
        {
            "agent": MakeEvolvable(
                network=cnn_actor,
                input_tensor=torch.randn(1, *ma_image_space[0].shape),
                device=device,
            ),
            "other_agent": MakeEvolvable(
                network=cnn_actor,
                input_tensor=torch.randn(1, *ma_image_space[0].shape),
                device=device,
            ),
        }
    )
    evo_critics = ModuleDict(
        {
            "agent": MakeEvolvable(
                network=cnn_critic,
                input_tensor=torch.randn(1, *ma_image_space[0].shape),
                device=device,
            ),
            "other_agent": MakeEvolvable(
                network=cnn_critic,
                input_tensor=torch.randn(1, *ma_image_space[0].shape),
                device=device,
            ),
        }
    )
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    ippo = IPPO(
        observation_spaces=ma_image_space,
        action_spaces=ma_discrete_space,
        agent_ids=agent_ids,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in ippo.actors.values())
        assert all(
            isinstance(critic, OptimizedModule) for critic in ippo.critics.values()
        )
    else:
        assert all(isinstance(actor, MakeEvolvable) for actor in ippo.actors.values())
        assert all(
            isinstance(critic, MakeEvolvable) for critic in ippo.critics.values()
        )

    assert ippo.observation_spaces == ma_image_space
    assert ippo.action_spaces == ma_discrete_space
    assert ippo.n_agents == 3
    assert ippo.agent_ids == agent_ids
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in ippo.actor_optimizers.values()
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in ippo.critic_optimizers.values()
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in ippo.actor_optimizers.values()
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in ippo.critic_optimizers.values()
        )
    assert isinstance(ippo.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_spaces, net",
    [
        ("ma_image_space", "cnn"),
        ("ma_vector_space", "mlp"),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_ippo_with_evo_networks(
    observation_spaces,
    ma_discrete_space,
    net,
    device,
    compile_mode,
    accelerator,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    net_config = get_default_encoder_config(observation_spaces[0])

    head_config = {
        "output_activation": "Softmax" if net == "mlp" else "GumbelSoftmax",
        "activation": "ReLU",
        "hidden_size": [64, 64],
        "init_layers": False,
    }

    critic_head_config = copy.deepcopy(head_config)
    critic_head_config.update({"output_activation": None})

    net_config = {"encoder_config": net_config, "head_config": head_config}
    critic_net_config = {
        "encoder_config": copy.deepcopy(net_config["encoder_config"]),
        "head_config": critic_head_config,
    }

    evo_actors = ModuleDict(
        {
            "agent": StochasticActor(
                observation_spaces[0], ma_discrete_space[0], device=device, **net_config
            ),
            "other_agent": StochasticActor(
                observation_spaces[2], ma_discrete_space[2], device=device, **net_config
            ),
        }
    )
    evo_critics = ModuleDict(
        {
            "agent": ValueNetwork(
                observation_space=observation_spaces[0],
                device=device,
                **critic_net_config,
            ),
            "other_agent": ValueNetwork(
                observation_space=observation_spaces[1],
                device=device,
                **critic_net_config,
            ),
        }
    )
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=ma_discrete_space,
        agent_ids=agent_ids,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
    )

    expected_actor_cls = (
        OptimizedModule
        if compile_mode is not None and accelerator is None
        else StochasticActor
    )
    expected_critic_cls = (
        OptimizedModule
        if compile_mode is not None and accelerator is None
        else ValueNetwork
    )
    assert all(isinstance(actor, expected_actor_cls) for actor in ippo.actors.values())
    assert all(
        isinstance(critic, expected_critic_cls) for critic in ippo.critics.values()
    )

    expected_opt_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    assert all(
        isinstance(actor_optimizer, expected_opt_cls)
        for actor_optimizer in ippo.actor_optimizers.values()
    )
    assert all(
        isinstance(critic_optimizer, expected_opt_cls)
        for critic_optimizer in ippo.critic_optimizers.values()
    )

    assert ippo.observation_spaces == observation_spaces
    assert ippo.action_spaces == ma_discrete_space
    assert ippo.n_agents == 3
    assert ippo.agent_ids == agent_ids
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]
    assert isinstance(ippo.criterion, nn.MSELoss)


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_ippo_with_incorrect_evo_networks(
    compile_mode, ma_vector_space, ma_discrete_space
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    evo_actors = []
    evo_critics = []

    with pytest.raises(AssertionError):
        ippo = IPPO(
            observation_spaces=ma_vector_space,
            action_spaces=ma_discrete_space,
            agent_ids=agent_ids,
            actor_networks=evo_actors,
            critic_networks=evo_critics,
            torch_compiler=compile_mode,
        )
        assert ippo


@pytest.mark.parametrize(
    "observation_spaces, action_spaces, actors, critics",
    [
        (
            "ma_vector_space",
            "ma_discrete_space",
            [EvolvableMLP(4, 2, [32]) for _ in range(2)],
            [1 for _ in range(2)],
        ),
        (
            "ma_vector_space",
            "ma_discrete_space",
            [1 for _ in range(2)],
            [EvolvableMLP(4, 2, [32]) for _ in range(2)],
        ),
    ],
)
def test_initialize_ippo_with_incorrect_networks(
    observation_spaces, action_spaces, actors, critics, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    with pytest.raises(TypeError):
        ippo = IPPO(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=["agent_0", "agent_1", "other_agent_0"],
            actor_networks=actors,
            critic_networks=critics,
        )
        assert ippo


@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space"])
def test_ippo_init_warning(
    mlp_actor, observation_spaces, action_spaces, device, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    warning_string = "Actor and critic network lists must both be supplied to use custom networks. Defaulting to net config."
    evo_actors = [
        MakeEvolvable(network=mlp_actor, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(2)
    ]
    with pytest.warns(UserWarning, match=warning_string):
        IPPO(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=["agent_0", "agent_1", "other_agent_0"],
            actor_networks=evo_actors,
            device=device,
        )


def test_grouped_outputs_functions(ma_vector_space, ma_discrete_space):
    """Test that the assemble_grouped_outputs and disassemble_grouped_outputs
    functions work as expected and are inverses of each other."""

    # Initialize agent with grouped agents
    compile_mode = None
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    agent = IPPO(
        observation_spaces=ma_vector_space,
        action_spaces=ma_discrete_space,
        agent_ids=agent_ids,
        device="cpu",
        torch_compiler=compile_mode,
    )

    # Setting up grouped agent configuration
    agent.shared_agent_ids = ["agent", "other_agent"]
    agent.grouped_agents = {
        "agent": ["agent_0", "agent_1"],
        "other_agent": ["other_agent_0"],
    }

    # Create sample individual agent outputs
    vect_dim = 4
    output_dim = 5
    agent_outputs = {
        "agent_0": np.random.rand(vect_dim, output_dim),
        "agent_1": np.random.rand(vect_dim, output_dim),
        "other_agent_0": np.random.rand(vect_dim, output_dim),
    }

    # Test assemble_grouped_outputs
    grouped_outputs = agent.assemble_grouped_outputs(agent_outputs, vect_dim)

    # Check that the grouped outputs have the correct keys
    assert set(grouped_outputs.keys()) == {"agent", "other_agent"}

    # Check that the agent outputs are assembled correctly
    assert grouped_outputs["agent"].shape == (2 * vect_dim, output_dim)
    assert grouped_outputs["other_agent"].shape == (1 * vect_dim, output_dim)

    # Test disassemble_grouped_outputs
    disassembled_outputs = agent.disassemble_grouped_outputs(
        grouped_outputs, vect_dim, agent.grouped_agents
    )

    # Check that the disassembled outputs have the correct keys
    assert set(disassembled_outputs.keys()) == {"agent_0", "agent_1", "other_agent_0"}

    # Check that the outputs are correctly disassembled
    for agent_id in agent_ids:
        assert disassembled_outputs[agent_id].shape == (vect_dim, output_dim)

    # Check the round trip consistency by comparing the original and disassembled outputs
    for agent_id in agent_ids:
        np.testing.assert_allclose(
            agent_outputs[agent_id],
            disassembled_outputs[agent_id],
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_get_action_distributed(compile_mode, ma_vector_space, ma_discrete_space):
    accelerator = Accelerator()
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: np.random.randn(*ma_vector_space[idx].shape).astype(np.float32)
        for idx, agent in enumerate(agent_ids)
    }
    ippo = IPPO(
        observation_spaces=ma_vector_space,
        action_spaces=ma_discrete_space,
        agent_ids=agent_ids,
        accelerator=accelerator,
        torch_compiler=compile_mode,
        net_config={
            "encoder_config": {"hidden_size": [16, 16], "init_layers": False},
            "head_config": {"hidden_size": [16], "init_layers": False},
        },
    )
    new_actors = ModuleDict(
        {
            shared_id: DummyStochasticActor(
                observation_space=actor.observation_space,
                action_space=actor.action_space,
                device=actor.device,
                action_std_init=ippo.action_std_init,
                encoder_config={"hidden_size": [16, 16], "init_layers": False},
                head_config={"hidden_size": [16], "init_layers": False},
            )
            for shared_id, actor in ippo.actors.items()
        }
    )
    ippo.actors = new_actors
    actions, log_probs, dist_entropy, state_values = ippo.get_action(obs=state)

    # Check returns are the proper format
    assert isinstance(actions, dict)
    assert isinstance(log_probs, dict)
    assert isinstance(dist_entropy, dict)
    assert isinstance(state_values, dict)

    for agent_id in agent_ids:
        assert agent_id in actions
        assert agent_id in log_probs
        assert agent_id in dist_entropy
        assert agent_id in state_values
