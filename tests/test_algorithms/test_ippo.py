import copy
import gc
from pathlib import Path
from unittest.mock import MagicMock

import dill
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
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.modules.custom_components import GumbelSoftmax
from agilerl.networks.actors import StochasticActor
from agilerl.networks.value_networks import ValueNetwork
from agilerl.utils.evolvable_networks import get_default_encoder_config
from agilerl.utils.utils import make_multi_agent_vect_envs
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    gen_multi_agent_dict_or_tuple_spaces,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_multi_agent_multidiscrete_spaces,
    get_sample_from_space,
)


@pytest.fixture(autouse=True)
def cleanup():
    yield  # Run the test first
    torch.cuda.empty_cache()  # Free up GPU memory


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
            "agent_1": 2,  # observes every 2 steps
            "other_agent_0": 4,  # observes every 4 steps
        }

        # Probability-based method (keeping for backward compatibility)
        self.observation_probabilities = {
            "agent_0": 0.8,  # 80% chance to return observation
            "agent_1": 0.6,  # 60% chance to return observation
            "other_agent_0": 0.4,  # 40% chance to return observation
        }

        # Initialize step counters for each agent
        self.agent_step_counters = {agent: 0 for agent in self.agents}

        # Observation skipping mode (frequency or probability)
        self.observation_mode = "frequency"  # can be "frequency" or "probability"

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

        if self.observation_mode == "frequency":
            # All agents observe at reset (step 0)
            self.active_agents = self.agents.copy()
        else:
            # Probability-based method
            self.active_agents = [
                agent
                for agent in self.agents
                if np.random.random() < self.observation_probabilities[agent]
            ]

            # Ensure at least one agent is active
            if not self.active_agents:
                self.active_agents = [np.random.choice(self.agents)]

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

        if self.observation_mode == "frequency":
            # Determine which agents should observe based on their frequency
            self.active_agents = [
                agent
                for agent in self.agents
                if self.agent_step_counters[agent] % self.observation_frequencies[agent]
                == 0
            ]
        else:
            # Probability-based method
            self.active_agents = [
                agent
                for agent in self.agents
                if np.random.random() < self.observation_probabilities[agent]
            ]

            # Ensure at least one agent is active
            if not self.active_agents:
                self.active_agents = [np.random.choice(self.agents)]

        observations = {
            agent: np.random.rand(
                *self.observation_spaces[self.possible_agents.index(agent)].shape
            )
            for agent in self.active_agents
        }

        rewards = {agent: np.random.randint(0, 5) for agent in action.keys()}
        dones = {agent: False for agent in self.active_agents}
        truncated = {agent: False for agent in self.active_agents}

        infos = {agent: {} for agent in self.active_agents}
        for agent in self.active_agents:
            infos[agent]["env_defined_actions"] = None

        # Always provide env_defined_actions for agent_0 if active
        if "agent_0" in self.active_agents:
            infos["agent_0"]["env_defined_actions"] = np.array([1])

        return observations, rewards, dones, truncated, infos


class MultiAgentCNNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=(3, 3), stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(15200, 256)
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
            in_channels=4, out_channels=16, kernel_size=(3, 3), stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(15200, 256)
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


@pytest.fixture
def mlp_actor(observation_spaces, action_spaces):
    net = nn.Sequential(
        nn.Linear(observation_spaces[0].shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, action_spaces[0].n),
        nn.Softmax(dim=-1),
    )
    yield net
    del net
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def mlp_critic(observation_spaces):
    net = nn.Sequential(
        nn.Linear(observation_spaces[0].shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    yield net
    del net
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def cnn_actor():
    net = MultiAgentCNNActor()
    yield net
    del net
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def cnn_critic():
    net = MultiAgentCNNCritic()
    yield net
    del net
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def mocked_accelerator():
    MagicMock(spec=Accelerator)


@pytest.fixture
def accelerated_experiences(
    batch_size, observation_spaces, action_spaces, agent_ids, device
):
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


@pytest.fixture
def experiences(batch_size, observation_spaces, action_spaces, agent_ids, device):
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


@pytest.fixture
def vectorized_experiences(
    batch_size, vect_dim, observation_spaces, action_spaces, agent_ids, device
):
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
@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(3, (6,)),
        generate_multi_agent_box_spaces(3, (3, 32, 32)),
    ],
)
@pytest.mark.parametrize("vectorized", [False, True])
def test_loop(device, sum_score, compile_mode, observation_spaces, vectorized):
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)

    if vectorized:
        env = make_multi_agent_vect_envs(
            DummyMultiEnv,
            2,
            **dict(observation_spaces=observation_spaces, action_spaces=action_spaces)
        )
    else:
        env = DummyMultiEnv(observation_spaces, action_spaces)

    ippo = IPPO(
        observation_spaces,
        action_spaces,
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


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(3, (6,)),
        generate_multi_agent_box_spaces(3, (3, 32, 32)),
        gen_multi_agent_dict_or_tuple_spaces(3, 2, 2, dict_space=True),
        gen_multi_agent_dict_or_tuple_spaces(3, 2, 2, dict_space=False),
    ],
)
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("wrap", [True, False])
def test_ippo_clone_returns_identical_agent(
    accelerator_flag, wrap, compile_mode, observation_spaces
):
    # Clones the agent and returns an identical copy.
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)
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
    if accelerator_flag:
        accelerator = Accelerator(device_placement=False)
    else:
        accelerator = None

    ippo = IPPO(
        observation_spaces,
        action_spaces,
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
    assert clone_agent.one_hot == ippo.one_hot
    assert clone_agent.n_agents == ippo.n_agents
    assert clone_agent.agent_ids == ippo.agent_ids
    assert clone_agent.discrete_actions == ippo.discrete_actions
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
    for clone_actor, actor in zip(clone_agent.actors, ippo.actors):
        assert str(clone_actor.state_dict()) == str(actor.state_dict())
    for clone_critic, critic in zip(clone_agent.critics, ippo.critics):
        assert str(clone_critic.state_dict()) == str(critic.state_dict())


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_clone_new_index(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(3, (4,))
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]

    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids,
        torch_compiler=compile_mode,
    )
    clone_agent = ippo.clone(index=100)

    assert clone_agent.index == 100


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_clone_after_learning(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(3, (4,))
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    batch_size = 8

    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids,
        batch_size=batch_size,
        torch_compiler=compile_mode,
        target_kl=1e-10,
    )

    states = {
        agent_id: np.random.randn(batch_size, observation_spaces[idx].shape[0])
        for idx, agent_id in enumerate(agent_ids)
    }
    actions = {
        agent_id: np.random.randint(0, action_spaces[idx].n, (batch_size,))
        for idx, agent_id in enumerate(agent_ids)
    }
    log_probs = {agent_id: np.random.randn(batch_size, 1) for agent_id in agent_ids}
    rewards = {agent_id: np.random.randn(batch_size, 1) for agent_id in agent_ids}
    dones = {agent_id: torch.zeros(batch_size, 1) for agent_id in agent_ids}
    values = {agent_id: np.random.randn(batch_size, 1) for agent_id in agent_ids}
    next_state = {
        agent_id: np.random.randn(observation_spaces[idx].shape[0])
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
    assert clone_agent.one_hot == ippo.one_hot
    assert clone_agent.n_agents == ippo.n_agents
    assert clone_agent.agent_ids == ippo.agent_ids
    assert clone_agent.discrete_actions == ippo.discrete_actions
    assert clone_agent.index == ippo.index
    assert clone_agent.batch_size == ippo.batch_size
    assert clone_agent.lr == ippo.lr
    assert clone_agent.learn_step == ippo.learn_step
    assert clone_agent.gamma == ippo.gamma
    assert clone_agent.gae_lambda == ippo.gae_lambda
    assert clone_agent.device == ippo.device
    assert clone_agent.accelerator == ippo.accelerator
    for clone_actor, actor in zip(clone_agent.actors, ippo.actors):
        assert str(clone_actor.state_dict()) == str(actor.state_dict())
    for clone_critic, critic in zip(clone_agent.critics, ippo.critics):
        assert str(clone_critic.state_dict()) == str(critic.state_dict())
    for clone_actor_opt, actor_opt in zip(
        clone_agent.actor_optimizers.optimizer, ippo.actor_optimizers.optimizer
    ):
        assert str(clone_actor_opt) == str(actor_opt)
    for clone_critic_opt, critic_opt in zip(
        clone_agent.critic_optimizers.optimizer, ippo.critic_optimizers.optimizer
    ):
        assert str(clone_critic_opt) == str(critic_opt)


@pytest.mark.parametrize(
    "observation_spaces, encoder_cls",
    [
        (generate_multi_agent_box_spaces(1, (6,)), EvolvableMLP),
        (generate_multi_agent_box_spaces(1, (3, 32, 32)), EvolvableCNN),
        (
            gen_multi_agent_dict_or_tuple_spaces(1, 2, 2, dict_space=True),
            EvolvableMultiInput,
        ),
        (
            gen_multi_agent_dict_or_tuple_spaces(1, 2, 2, dict_space=False),
            EvolvableMultiInput,
        ),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_save_load_checkpoint_correct_data_and_format(
    tmpdir, device, accelerator, compile_mode, observation_spaces, encoder_cls
):
    # Initialize the ippo agent
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        agent_ids=["agent_0"],
        torch_compiler=compile_mode,
        device=device,
        accelerator=accelerator,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ippo.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actors_init_dict" in checkpoint["network_info"]["modules"]
    assert "actors_state_dict" in checkpoint["network_info"]["modules"]
    assert "critics_init_dict" in checkpoint["network_info"]["modules"]
    assert "critics_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "gae_lambda" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # Load checkpoint
    loaded_ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        agent_ids=["agent_0"],
        torch_compiler=compile_mode,
        device=device,
        accelerator=accelerator,
    )
    loaded_ippo.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in loaded_ippo.actors)
        assert all(
            isinstance(critic, OptimizedModule) for critic in loaded_ippo.critics
        )
    else:
        assert all(
            isinstance(actor.encoder, encoder_cls) for actor in loaded_ippo.actors
        )
        assert all(
            isinstance(critic.encoder, encoder_cls) for critic in loaded_ippo.critics
        )
    assert ippo.lr == 1e-4

    for actor, loaded_actor in zip(ippo.actors, loaded_ippo.actors):
        assert str(actor.state_dict()) == str(loaded_actor.state_dict())

    for critic, loaded_critic in zip(ippo.critics, loaded_ippo.critics):
        assert str(critic.state_dict()) == str(loaded_critic.state_dict())

    assert ippo.batch_size == 64
    assert ippo.learn_step == 2048
    assert ippo.gamma == 0.99
    assert ippo.gae_lambda == 0.95
    assert ippo.mut is None
    assert ippo.index == 0
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]


# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize(
    "observation_spaces, action_spaces",
    [
        (
            generate_multi_agent_box_spaces(1, (6,)),
            generate_multi_agent_discrete_spaces(1, 2),
        )
    ],
)
def test_ippo_save_load_checkpoint_correct_data_and_format_make_evo(
    tmpdir,
    observation_spaces,
    action_spaces,
    mlp_actor,
    mlp_critic,
    device,
    compile_mode,
    accelerator,
):
    evo_actors = [
        MakeEvolvable(network=mlp_actor, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(1)
    ]
    evo_critics = [
        MakeEvolvable(network=mlp_critic, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(1)
    ]
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0"],
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
    )
    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ippo.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actors_init_dict" in checkpoint["network_info"]["modules"]
    assert "actors_state_dict" in checkpoint["network_info"]["modules"]
    assert "critics_init_dict" in checkpoint["network_info"]["modules"]
    assert "critics_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "gae_lambda" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # Load checkpoint
    loaded_ippo = IPPO(
        observation_spaces=generate_multi_agent_box_spaces(
            1, (3, 32, 32), low=0, high=255
        ),
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        agent_ids=["agent_0"],
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
        net_config={
            "encoder_config": {
                "channel_size": [16],
                "kernel_size": [3],
                "stride_size": [1],
                "init_layers": False,
            },
            "head_config": {"hidden_size": [32], "init_layers": False},
        },
    )
    loaded_ippo.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in loaded_ippo.actors)
        assert all(
            isinstance(critic, OptimizedModule) for critic in loaded_ippo.critics
        )
    else:
        assert all(isinstance(actor, MakeEvolvable) for actor in loaded_ippo.actors)
        assert all(isinstance(critic, MakeEvolvable) for critic in loaded_ippo.critics)
    assert ippo.lr == 1e-4

    for actor, loaded_actor in zip(ippo.actors, loaded_ippo.actors):
        assert str(actor.state_dict()) == str(loaded_actor.state_dict())

    for critic, loaded_critic in zip(ippo.critics, loaded_ippo.critics):
        assert str(critic.state_dict()) == str(loaded_critic.state_dict())

    assert ippo.batch_size == 64
    assert ippo.learn_step == 2048
    assert ippo.gamma == 0.99
    assert ippo.gae_lambda == 0.95
    assert ippo.mut is None
    assert ippo.index == 0
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_ippo_unwrap_models(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(3, (6,))
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)
    accelerator = Accelerator()
    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        accelerator=accelerator,
        torch_compiler=compile_mode,
        net_config={
            "encoder_config": {"hidden_size": [16, 16], "init_layers": False},
            "head_config": {"hidden_size": [16], "init_layers": False},
        },
    )
    ippo.unwrap_models()
    for actor, critic in zip(ippo.actors, ippo.critics):
        assert isinstance(actor, nn.Module)
        assert isinstance(critic, nn.Module)


# The saved checkpoint file contains the correct data and format.
@pytest.mark.parametrize(
    "observation_spaces, encoder_cls",
    [
        (generate_multi_agent_box_spaces(3, (4,)), EvolvableMLP),
        (generate_multi_agent_box_spaces(3, (3, 32, 32)), EvolvableCNN),
        (
            gen_multi_agent_dict_or_tuple_spaces(3, 2, 2, dict_space=True),
            EvolvableMultiInput,
        ),
        (
            gen_multi_agent_dict_or_tuple_spaces(3, 2, 2, dict_space=False),
            EvolvableMultiInput,
        ),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_load_from_pretrained(
    device, accelerator, tmpdir, compile_mode, observation_spaces, encoder_cls
):
    # Initialize the ippo agent
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=generate_multi_agent_discrete_spaces(3, 2),
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        torch_compiler=compile_mode,
        accelerator=accelerator,
        device=device,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ippo.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_ippo = IPPO.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_ippo.observation_spaces == ippo.observation_spaces
    assert new_ippo.action_spaces == ippo.action_spaces
    assert new_ippo.one_hot == ippo.one_hot
    assert new_ippo.n_agents == ippo.n_agents
    assert new_ippo.agent_ids == ippo.agent_ids
    assert new_ippo.lr == ippo.lr
    for (
        new_actor,
        new_critic,
        actor,
        critic,
    ) in zip(
        new_ippo.actors,
        new_ippo.critics,
        ippo.actors,
        ippo.critics,
    ):

        if compile_mode is not None and accelerator is None:
            assert isinstance(new_actor, OptimizedModule)
            assert isinstance(new_critic, OptimizedModule)
        else:
            assert isinstance(new_actor.encoder, encoder_cls)
            assert isinstance(new_critic.encoder, encoder_cls)

        new_actor_sd = str(new_actor.state_dict())
        new_critic_sd = str(new_critic.state_dict())

        assert new_actor_sd == str(actor.state_dict())
        assert new_critic_sd == str(critic.state_dict())

    assert new_ippo.batch_size == ippo.batch_size
    assert new_ippo.learn_step == ippo.learn_step
    assert new_ippo.gamma == ippo.gamma
    assert new_ippo.gae_lambda == ippo.gae_lambda
    assert new_ippo.mut == ippo.mut
    assert new_ippo.index == ippo.index
    assert new_ippo.scores == ippo.scores
    assert new_ippo.fitness == ippo.fitness
    assert new_ippo.steps == ippo.steps


# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize(
    "observation_spaces, action_spaces, arch, input_tensor, critic_input_tensor, compile_mode",
    [
        (
            generate_multi_agent_box_spaces(3, (4,)),
            generate_multi_agent_discrete_spaces(3, 2),
            "mlp",
            torch.randn(1, 4),
            torch.randn(1, 4),
            None,
        ),
        (
            generate_multi_agent_box_spaces(3, (4, 210, 160), low=0, high=255),
            generate_multi_agent_discrete_spaces(3, 2),
            "cnn",
            torch.randn(1, 4, 210, 160),
            torch.randn(1, 4, 210, 160),
            None,
        ),
        (
            generate_multi_agent_box_spaces(3, (4,)),
            generate_multi_agent_discrete_spaces(3, 2),
            "mlp",
            torch.randn(1, 4),
            torch.randn(1, 4),
            "default",
        ),
        (
            generate_multi_agent_box_spaces(3, (4, 210, 160), low=0, high=255),
            generate_multi_agent_discrete_spaces(3, 2),
            "cnn",
            torch.randn(1, 4, 210, 160),
            torch.randn(1, 4, 210, 160),
            "default",
        ),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_networks(
    mlp_actor,
    mlp_critic,
    cnn_actor,
    cnn_critic,
    observation_spaces,
    action_spaces,
    arch,
    input_tensor,
    critic_input_tensor,
    tmpdir,
    compile_mode,
    device,
):
    if arch == "mlp":
        actor_network = mlp_actor
        critic_network = mlp_critic
    elif arch == "cnn":
        actor_network = cnn_actor
        critic_network = cnn_critic

    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_network = MakeEvolvable(
        critic_network,
        critic_input_tensor,
    )

    # Initialize the ippo agent
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        actor_networks=[actor_network, copy.deepcopy(actor_network)],
        critic_networks=[critic_network, copy.deepcopy(critic_network)],
        torch_compiler=compile_mode,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ippo.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_ippo = IPPO.load(checkpoint_path, device=device)

    # Check if properties and weights are loaded correctly
    assert new_ippo.observation_spaces == ippo.observation_spaces
    assert new_ippo.action_spaces == ippo.action_spaces
    assert new_ippo.one_hot == ippo.one_hot
    assert new_ippo.n_agents == ippo.n_agents
    assert new_ippo.agent_ids == ippo.agent_ids
    assert new_ippo.lr == ippo.lr
    for (
        new_actor,
        new_critic,
        actor,
        critic,
    ) in zip(
        new_ippo.actors,
        new_ippo.critics,
        ippo.actors,
        ippo.critics,
    ):
        assert isinstance(new_actor, nn.Module)
        assert isinstance(new_critic, nn.Module)
        assert str(new_actor.to("cpu").state_dict()) == str(actor.state_dict())
        assert str(new_critic.to("cpu").state_dict()) == str(critic.state_dict())

    assert new_ippo.batch_size == ippo.batch_size
    assert new_ippo.learn_step == ippo.learn_step
    assert new_ippo.gamma == ippo.gamma
    assert new_ippo.gae_lambda == ippo.gae_lambda
    assert new_ippo.mut == ippo.mut
    assert new_ippo.index == ippo.index
    assert new_ippo.scores == ippo.scores
    assert new_ippo.fitness == ippo.fitness
    assert new_ippo.steps == ippo.steps


@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("agent_ids", [["agent_0", "agent_1", "other_agent_0"]])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(3, 2)])
@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(3, (6,)),
    ],
)
def test_ippo_learns_from_experiences_distributed(
    observation_spaces,
    accelerated_experiences,
    batch_size,
    action_spaces,
    agent_ids,
    compile_mode,
):
    accelerator = Accelerator(device_placement=False)
    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )

    for actor, critic in zip(ippo.actors, ippo.critics):
        actor.no_sync = no_sync.__get__(actor)
        critic.no_sync = no_sync.__get__(critic)

    actors = ippo.actors
    actors_pre_learn_sd = [
        str(copy.deepcopy(actor.state_dict())) for actor in ippo.actors
    ]
    critics = ippo.critics
    critics_pre_learn_sd = [
        str(copy.deepcopy(critic.state_dict())) for critic in ippo.critics
    ]

    for _ in range(3):
        ippo.scores.append(0)
        loss = ippo.learn(accelerated_experiences)

    assert isinstance(loss, dict)
    for agent_id in ippo.shared_agent_ids:
        assert agent_id in loss

    for old_actor, updated_actor in zip(actors, ippo.actors):
        assert old_actor == updated_actor

    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, ippo.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())

    for old_critic, updated_critic in zip(critics, ippo.critics):
        assert old_critic == updated_critic

    for old_critic_state_dict, updated_critic in zip(
        critics_pre_learn_sd, ippo.critics
    ):
        assert old_critic_state_dict != str(updated_critic.state_dict())


@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("agent_ids", [["agent_0", "agent_1", "other_agent_0"]])
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(3, 2)])
@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(3, (3, 32, 32)),
        generate_multi_agent_box_spaces(3, (6,)),
    ],
)
def test_ippo_learns_from_experiences(
    observation_spaces,
    experiences,
    batch_size,
    action_spaces,
    agent_ids,
    device,
    compile_mode,
):
    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )

    actors = ippo.actors
    actors_pre_learn_sd = [copy.deepcopy(actor.state_dict()) for actor in ippo.actors]
    critics = ippo.critics
    critics_pre_learn_sd = [
        str(copy.deepcopy(critic.state_dict())) for critic in ippo.critics
    ]

    for _ in range(4):
        ippo.scores.append(0)
        loss = ippo.learn(experiences)

    assert isinstance(loss, dict)
    for agent_id in ippo.shared_agent_ids:
        assert agent_id in loss

    for old_actor, updated_actor in zip(actors, ippo.actors):
        assert old_actor == updated_actor

    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, ippo.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())

    for old_critic, updated_critic in zip(critics, ippo.critics):
        assert old_critic == updated_critic

    for old_critic_state_dict, updated_critic in zip(
        critics_pre_learn_sd, ippo.critics
    ):
        assert old_critic_state_dict != str(updated_critic.state_dict())


@pytest.mark.parametrize("agent_ids", [["agent_0", "agent_1", "other_agent_0"]])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("vect_dim", [1, 8])
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_discrete_spaces(3, 2),
        generate_multi_agent_box_spaces(3, (6,)),
    ],
)
@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(3, (3, 32, 32)),  # cnn
        generate_multi_agent_box_spaces(3, (6,)),  # mlp
    ],
)
def test_ippo_learns_from_vectorized_experiences(
    observation_spaces,
    vectorized_experiences,
    batch_size,
    action_spaces,
    agent_ids,
    device,
    compile_mode,
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
        batch_size=batch_size,
        lr=0.1,
    )

    actors = ippo.actors
    actors_pre_learn_sd = [
        str(copy.deepcopy(actor.state_dict())) for actor in ippo.actors
    ]
    critics = ippo.critics
    critics_pre_learn_sd = [
        copy.deepcopy(critic.state_dict()) for critic in ippo.critics
    ]

    for _ in range(4):
        ippo.scores.append(0)
        loss = ippo.learn(vectorized_experiences)

    assert isinstance(loss, dict)
    for agent_id in ippo.shared_agent_ids:
        assert agent_id in loss

    for old_actor, updated_actor in zip(actors, ippo.actors):
        assert old_actor == updated_actor

    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, ippo.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())

    for old_critic, updated_critic in zip(critics, ippo.critics):
        assert old_critic == updated_critic

    for old_critic_state_dict, updated_critic in zip(
        critics_pre_learn_sd, ippo.critics
    ):
        assert old_critic_state_dict != str(updated_critic.state_dict())


@pytest.mark.parametrize(
    "observation_spaces, batch_size, vect_dim, action_spaces, agent_ids, compile_mode",
    [
        (
            generate_multi_agent_box_spaces(3, (3,)),
            4,
            3,
            generate_multi_agent_box_spaces(3, (3,)),
            ["agent_0", "agent_1", "other_agent_0"],
            None,
        ),
    ],
)
def test_ippo_learns_from_hardcoded_vectorized_experiences_mlp(
    observation_spaces,
    batch_size,
    vect_dim,
    action_spaces,
    agent_ids,
    device,
    compile_mode,
):
    states = {
        agent: np.array(
            [
                [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                [[7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ]
        )
        * i
        for i, agent in enumerate(agent_ids)
    }

    actions = {
        agent: np.array(
            [
                [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                [[7, 7, 7], [8, 8, 8], [9, 9, 9]],
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
        agent: np.array([[4, 4, 4], [7, 7, 7], [10, 10, 10]]) * i
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
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
        batch_size=batch_size,
    )

    actors = ippo.actors
    actors_pre_learn_sd = [copy.deepcopy(actor.state_dict()) for actor in ippo.actors]
    critics = ippo.critics
    critics_pre_learn_sd = [
        str(copy.deepcopy(critic.state_dict())) for critic in ippo.critics
    ]

    for _ in range(4):
        ippo.scores.append(0)
        loss = ippo.learn(experiences)

    assert isinstance(loss, dict)
    for agent_id in ippo.shared_agent_ids:
        assert agent_id in loss

    for old_actor, updated_actor in zip(actors, ippo.actors):
        assert old_actor == updated_actor

    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, ippo.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())

    for old_critic, updated_critic in zip(critics, ippo.critics):
        assert old_critic == updated_critic

    for old_critic_state_dict, updated_critic in zip(
        critics_pre_learn_sd, ippo.critics
    ):
        assert old_critic_state_dict != str(updated_critic.state_dict())


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
        generate_multi_agent_discrete_spaces(3, 2),
        generate_multi_agent_box_spaces(3, (2,)),
    ],
)
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize(
    "observation_spaces", [generate_multi_agent_box_spaces(3, (6,))]
)
def test_ippo_get_action_agent_masking(
    observation_spaces, action_spaces, device, compile_mode
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: np.random.randn(*observation_spaces[0].shape) for agent in agent_ids
    }
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
            "agent_0": {"env_defined_actions": np.array([0, 1])},
            "agent_1": {"env_defined_actions": None},
            "other_agent_0": {"env_defined_actions": None},
        }

    # Define the ippo agent
    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )

    # Get the action
    actions, _, _, _ = ippo.get_action(obs=state, infos=info)

    if discrete_actions:
        assert np.array_equal(actions["agent_0"], np.array([[1]])), actions["agent_0"]
    else:
        assert np.array_equal(actions["agent_0"], np.array([[0, 1]])), actions[
            "agent_0"
        ]


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_discrete_spaces(3, 4),
        generate_multi_agent_box_spaces(3, (4,)),
        generate_multi_agent_box_spaces(3, (4, 32, 32), low=0, high=255),
        generate_multi_agent_multidiscrete_spaces(3, 2),
        gen_multi_agent_dict_or_tuple_spaces(3, 1, 1, dict_space=True),
        gen_multi_agent_dict_or_tuple_spaces(3, 1, 1, dict_space=False),
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_discrete_spaces(3, 2),
        generate_multi_agent_box_spaces(3, (2,), low=-1, high=1),
        [spaces.MultiBinary(2) for _ in range(3)],
        generate_multi_agent_multidiscrete_spaces(3, 2),
    ],
)
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_ippo_get_action(
    observation_spaces, action_spaces, device, compile_mode, accelerator
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent_id: get_sample_from_space(observation_space)
        for agent_id, observation_space in zip(agent_ids, observation_spaces)
    }
    info = (
        {agent: {"env_defined_actions": None} for agent in agent_ids}
        if all(isinstance(space, spaces.Box) for space in action_spaces)
        else None
    )

    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
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
        generate_multi_agent_box_spaces(3, (6,)),
        generate_multi_agent_box_spaces(3, (3, 32, 32), low=0, high=255),
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_box_spaces(3, (2,), low=-1, high=1),
        generate_multi_agent_discrete_spaces(3, 2),
    ],
)
@pytest.mark.parametrize("training", [1, 0])
@pytest.mark.parametrize("compile_mode", [None])
def test_ippo_get_action_vectorized(
    training, observation_spaces, action_spaces, device, compile_mode
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    vect_dim = 2
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
        observation_spaces,
        action_spaces,
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


def test_ippo_get_action_action_masking_exception(device):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    observation_spaces = generate_multi_agent_box_spaces(3, (6,))
    action_spaces = generate_multi_agent_discrete_spaces(3, 4)
    state = {
        agent: {
            "observation": np.random.randn(*observation_spaces[idx].shape),
            "action_mask": [0, 1, 0, 1],
        }
        for idx, agent in enumerate(agent_ids)
    }
    ippo = IPPO(
        observation_spaces,
        action_spaces,
        net_config={
            "encoder_config": {"hidden_size": [64, 64], "init_layers": False},
            "head_config": {"hidden_size": [32], "init_layers": False},
        },
        agent_ids=agent_ids,
        device=device,
    )
    with pytest.raises(AssertionError):
        actions, log_probs, dist_entropy, state_values = ippo.get_action(obs=state)


def test_ippo_get_action_action_masking(device):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    observation_spaces = generate_multi_agent_box_spaces(3, (6,))
    action_spaces = generate_multi_agent_discrete_spaces(3, 4)
    state = {
        agent: np.random.randn(*observation_spaces[idx].shape).astype(np.float32)
        for idx, agent in enumerate(agent_ids)
    }
    info = {
        agent: {
            "action_mask": [0, 1, 0, 1],
        }
        for agent in agent_ids
    }
    ippo = IPPO(
        observation_spaces,
        action_spaces,
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
def test_ippo_init_torch_compiler_no_error(mode):
    ippo = IPPO(
        observation_spaces=generate_multi_agent_box_spaces(3, (1,)),
        action_spaces=generate_multi_agent_discrete_spaces(3, 1),
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_compiler=mode,
    )
    if isinstance(mode, str):
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule) for a in ippo.actors
        )
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule)
            for a in ippo.critics
        )
        assert ippo.torch_compiler == mode
    else:
        assert isinstance(ippo, IPPO)


@pytest.mark.parametrize("mode", (1, True, "max-autotune-no-cudagraphs"))
def test_ippo_init_torch_compiler_error(mode):
    err_string = (
        "Choose between torch compiler modes: "
        "default, reduce-overhead, max-autotune or None"
    )
    with pytest.raises(AssertionError, match=err_string):
        IPPO(
            observation_spaces=generate_multi_agent_box_spaces(3, (1,)),
            action_spaces=generate_multi_agent_discrete_spaces(3, 1),
            agent_ids=["agent_0", "agent_1", "other_agent_0"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_compiler=mode,
        )


@pytest.mark.parametrize(
    "observation_spaces, net_config",
    [
        (
            generate_multi_agent_box_spaces(3, (4,)),
            {
                "encoder_config": {"hidden_size": [64, 64], "init_layers": False},
                "head_config": {"hidden_size": [32], "init_layers": False},
            },
        ),
        (
            generate_multi_agent_box_spaces(3, (3, 32, 32), low=0, high=255),
            {
                "encoder_config": {
                    "channel_size": [3],
                    "kernel_size": [3],
                    "stride_size": [1],
                    "init_layers": False,
                },
                "head_config": {"hidden_size": [32], "init_layers": False},
            },
        ),
    ],
)
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_ippo_with_net_config(
    net_config, accelerator_flag, observation_spaces, device, compile_mode
):
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    batch_size = 64
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None

    ippo = IPPO(
        observation_spaces=observation_spaces,
        net_config=net_config,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        accelerator=accelerator,
        device=device,
        torch_compiler=compile_mode,
        target_kl=0.5,
    )

    assert ippo.observation_spaces == observation_spaces
    assert ippo.action_spaces == action_spaces
    assert ippo.n_agents == len(agent_ids)
    assert ippo.agent_ids == agent_ids
    assert ippo.batch_size == batch_size
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]
    assert ippo.target_kl == 0.5

    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in ippo.actors)
        assert all(isinstance(critic, OptimizedModule) for critic in ippo.critics)
    else:
        assert all(isinstance(actor, StochasticActor) for actor in ippo.actors)
        assert all(isinstance(critic, ValueNetwork) for critic in ippo.critics)

    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in ippo.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in ippo.critic_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in ippo.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in ippo.critic_optimizers
        )
    assert isinstance(ippo.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_spaces", [generate_multi_agent_box_spaces(3, (6,))]
)
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(3, 2)])
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_ippo_with_mlp_networks(
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    accelerator_flag,
    device,
    compile_mode,
):
    accelerator = Accelerator() if accelerator_flag else None
    evo_actors = [
        MakeEvolvable(network=mlp_actor, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(2)
    ]
    evo_critics = [
        MakeEvolvable(network=mlp_critic, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(2)
    ]
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in ippo.actors)
        assert all(isinstance(critic, OptimizedModule) for critic in ippo.critics)
    else:
        assert all(isinstance(actor, MakeEvolvable) for actor in ippo.actors)
        assert all(isinstance(critic, MakeEvolvable) for critic in ippo.critics)

    assert ippo.observation_spaces == observation_spaces
    assert ippo.action_spaces == action_spaces
    assert ippo.one_hot is False
    assert ippo.n_agents == 3
    assert ippo.agent_ids == ["agent_0", "agent_1", "other_agent_0"]
    assert ippo.discrete_actions is True
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in ippo.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in ippo.critic_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in ippo.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in ippo.critic_optimizers
        )
    assert isinstance(ippo.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_spaces", [generate_multi_agent_box_spaces(3, (6,))]
)
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(3, 2)])
def test_initialize_ippo_with_mlp_networks_gumbel_softmax(
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    device,
):
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
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        net_config=net_config,
        device=device,
        torch_compiler="reduce-overhead",
    )
    assert ippo.torch_compiler == "reduce-overhead"


@pytest.mark.parametrize(
    "observation_spaces",
    [generate_multi_agent_box_spaces(3, (4, 210, 160), low=0, high=255)],
)
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(3, 2)])
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_ippo_with_cnn_networks(
    cnn_actor,
    cnn_critic,
    observation_spaces,
    action_spaces,
    accelerator_flag,
    device,
    compile_mode,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    evo_actors = [
        MakeEvolvable(
            network=cnn_actor,
            input_tensor=torch.randn(1, 4, 210, 160),
            device=device,
        )
        for _ in range(2)
    ]
    evo_critics = [
        MakeEvolvable(
            network=cnn_critic,
            input_tensor=torch.randn(1, 4, 210, 160),
            device=device,
        )
        for _ in range(2)
    ]
    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in ippo.actors)
        assert all(isinstance(critic, OptimizedModule) for critic in ippo.critics)
    else:
        assert all(isinstance(actor, MakeEvolvable) for actor in ippo.actors)
        assert all(isinstance(critic, MakeEvolvable) for critic in ippo.critics)

    assert ippo.observation_spaces == observation_spaces
    assert ippo.action_spaces == action_spaces
    assert ippo.one_hot is False
    assert ippo.n_agents == 3
    assert ippo.agent_ids == ["agent_0", "agent_1", "other_agent_0"]
    assert ippo.discrete_actions is True
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in ippo.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in ippo.critic_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in ippo.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in ippo.critic_optimizers
        )
    assert isinstance(ippo.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_spaces, net",
    [
        (generate_multi_agent_box_spaces(3, (4, 210, 160), low=0, high=255), "cnn"),
        (generate_multi_agent_box_spaces(3, (4,)), "mlp"),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_ippo_with_evo_networks(
    observation_spaces, net, device, compile_mode, accelerator
):
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)
    net_config = get_default_encoder_config(observation_spaces[0])

    # For image spaces we need to give a sample input tensor to build networks
    if len(observation_spaces[0].shape) == 3:
        net_config["sample_input"] = torch.zeros(
            (1, *observation_spaces[0].shape), dtype=torch.float32, device=device
        )

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

    evo_actors = [
        StochasticActor(
            observation_spaces[x], action_spaces[x], device=device, **net_config
        )
        for x in range(2)
    ]
    evo_critics = [
        ValueNetwork(
            observation_space=observation_spaces[x], device=device, **critic_net_config
        )
        for x in range(2)
    ]

    ippo = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
    )

    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in ippo.actors)
        assert all(isinstance(critic, OptimizedModule) for critic in ippo.critics)
    else:
        assert all(
            isinstance(actor.encoder, (EvolvableMLP, EvolvableCNN))
            for actor in ippo.actors
        )
        assert all(
            isinstance(critic.encoder, (EvolvableMLP, EvolvableCNN))
            for critic in ippo.critics
        )

    assert ippo.observation_spaces == observation_spaces
    assert ippo.action_spaces == action_spaces
    assert ippo.one_hot is False
    assert ippo.n_agents == 3
    assert ippo.agent_ids == ["agent_0", "agent_1", "other_agent_0"]
    assert ippo.discrete_actions is True
    assert ippo.scores == []
    assert ippo.fitness == []
    assert ippo.steps == [0]

    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in ippo.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in ippo.critic_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in ippo.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in ippo.critic_optimizers
        )

    assert isinstance(ippo.criterion, nn.MSELoss)


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_ippo_with_incorrect_evo_networks(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(3, (4,))
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)
    evo_actors = []
    evo_critics = []

    with pytest.raises(AssertionError):
        ippo = IPPO(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=["agent_0", "agent_1", "other_agent_0"],
            actor_networks=evo_actors,
            critic_networks=evo_critics,
            torch_compiler=compile_mode,
        )
        assert ippo


@pytest.mark.parametrize(
    "observation_spaces, action_spaces, actors, critics",
    [
        (
            generate_multi_agent_box_spaces(3, (4,)),
            generate_multi_agent_discrete_spaces(3, 2),
            [EvolvableMLP(4, 2, [32]) for _ in range(3)],
            [1 for _ in range(3)],
        ),
        (
            generate_multi_agent_box_spaces(3, (4,)),
            generate_multi_agent_discrete_spaces(3, 2),
            [1 for _ in range(3)],
            [EvolvableMLP(4, 2, [32]) for _ in range(3)],
        ),
    ],
)
def test_initialize_ippo_with_incorrect_networks(
    observation_spaces, action_spaces, actors, critics
):

    with pytest.raises(TypeError):
        ippo = IPPO(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=["agent_0", "agent_1", "other_agent_0"],
            actor_networks=actors,
            critic_networks=critics,
        )
        assert ippo


@pytest.mark.parametrize(
    "observation_spaces", [generate_multi_agent_box_spaces(3, (6,))]
)
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(3, 2)])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_ippo_init_warning(
    mlp_actor, observation_spaces, action_spaces, device, compile_mode
):
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
            torch_compiler=compile_mode,
        )


def test_homogeneous_outputs_functions():
    """Test that the assemble_homogeneous_outputs and disassemble_homogeneous_outputs
    functions work as expected and are inverses of each other."""

    # Initialize agent with homogeneous agents
    observation_spaces = generate_multi_agent_box_spaces(3, (6,))
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)
    compile_mode = None
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    agent = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device="cpu",
        torch_compiler=compile_mode,
    )

    # Setting up homogeneous agent configuration
    agent.shared_agent_ids = ["agent", "other_agent"]
    agent.homogeneous_agents = {
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

    # Test assemble_homogeneous_outputs
    homo_outputs = agent.assemble_homogeneous_outputs(agent_outputs, vect_dim)

    # Check that the homogeneous outputs have the correct keys
    assert set(homo_outputs.keys()) == {"agent", "other_agent"}

    # Check that the agent outputs are assembled correctly
    assert homo_outputs["agent"].shape == (2 * vect_dim, output_dim)
    assert homo_outputs["other_agent"].shape == (1 * vect_dim, output_dim)

    # Test disassemble_homogeneous_outputs
    disassembled_outputs = agent.disassemble_homogeneous_outputs(
        homo_outputs, vect_dim, agent.homogeneous_agents
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
def test_get_action_distributed(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(3, (6,))
    action_spaces = generate_multi_agent_discrete_spaces(3, 2)
    accelerator = Accelerator()
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: np.random.randn(*observation_spaces[idx].shape).astype(np.float32)
        for idx, agent in enumerate(agent_ids)
    }
    ippo = IPPO(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        accelerator=accelerator,
        torch_compiler=compile_mode,
        net_config={
            "encoder_config": {"hidden_size": [16, 16], "init_layers": False},
            "head_config": {"hidden_size": [16], "init_layers": False},
        },
    )
    new_actors = [
        DummyStochasticActor(
            observation_space=actor.observation_space,
            action_space=actor.action_space,
            device=actor.device,
            action_std_init=ippo.action_std_init,
            encoder_config={"hidden_size": [16, 16], "init_layers": False},
            head_config={"hidden_size": [16], "init_layers": False},
        )
        for actor in ippo.actors
    ]
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


@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize(
    "observation_spaces, action_spaces",
    [
        (
            generate_multi_agent_box_spaces(3, (6,)),
            generate_multi_agent_discrete_spaces(3, 2),
        ),
        (
            generate_multi_agent_box_spaces(3, (6,)),
            generate_multi_agent_discrete_spaces(3, 2),
        ),
        (
            generate_multi_agent_box_spaces(3, (6,)),
            generate_multi_agent_discrete_spaces(3, 2),
        ),
        (
            generate_multi_agent_box_spaces(3, (6,)),
            generate_multi_agent_discrete_spaces(3, 2),
        ),
    ],
)
@pytest.mark.parametrize("observation_mode", ["frequency"])
def test_ippo_custom_training_with_async_env(
    observation_spaces, action_spaces, device, compile_mode, observation_mode
):
    """Test IPPO with a custom training loop on asynchronous environment for multiple iterations."""

    # Create async environment with agents that return observations asynchronously
    env = DummyMultiEnvAsync(observation_spaces, action_spaces)

    # Set observation mode (frequency-based or probability-based)
    env.observation_mode = observation_mode

    if observation_mode == "frequency":
        # Configure agents with different observation frequencies
        # NOTE: Assume homogeneous agents have the same frequency
        env.observation_frequencies = {
            "agent_0": 1,  # Observes every step
            "agent_1": 1,  # Observes every 2 steps
            "other_agent_0": 4,  # Observes every 4 steps
        }
    else:
        # Configure agents with different observation probabilities
        env.observation_probabilities = {
            "agent_0": 0.9,  # 90% chance to return observation
            "agent_1": 0.5,  # 50% chance to return observation
            "other_agent_0": 0.25,  # 25% chance to return observation
        }

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

    # Custom training loop for multiple iterations
    n_iterations = 5

    for iteration in range(n_iterations):
        # Reset environment
        observations, infos = env.reset()

        # Track agent participation for analysis
        agent_participation = {agent_id: 0 for agent_id in agent_ids}
        steps_with_all_agents = 0

        states = {agent_id: [] for agent_id in agent_ids}
        actions = {agent_id: [] for agent_id in agent_ids}
        log_probs = {agent_id: [] for agent_id in agent_ids}
        rewards = {agent_id: [] for agent_id in agent_ids}
        dones = {agent_id: [] for agent_id in agent_ids}
        values = {agent_id: [] for agent_id in agent_ids}

        done = {agent_id: np.zeros((1,), dtype=np.int8) for agent_id in agent_ids}

        # Collect experiences for multiple steps
        max_steps = 105
        for step in range(max_steps):
            # Get actions for current active agents
            action_dict, logprob_dict, _, value_dict = agent.get_action(
                observations, infos
            )

            # Verify actions are only for active agents
            assert all(agent_id in observations for agent_id in action_dict)

            # Update participation stats
            for agent_id in observations:
                agent_participation[agent_id] += 1

            # Count steps with all agents or no agents
            if len(observations) == len(agent_ids):
                steps_with_all_agents += 1

            # Step the environment
            next_observations, reward_dict, terminated, truncated, next_infos = (
                env.step(action_dict)
            )

            # Store experiences for active agents
            next_dones = {}
            for agent_id in observations:
                states[agent_id].append(observations[agent_id])
                actions[agent_id].append(action_dict[agent_id])
                log_probs[agent_id].append(logprob_dict[agent_id])
                values[agent_id].append(value_dict[agent_id])
                dones[agent_id].append(done[agent_id])
                rewards[agent_id].append(reward_dict[agent_id])

            for agent_id in terminated:
                next_dones[agent_id] = np.expand_dims(
                    np.logical_or(terminated[agent_id], truncated[agent_id]).astype(
                        np.int8
                    ),
                    axis=-1,
                )

            # Update for next step
            observations = next_observations
            done = next_dones
            infos = next_infos

            # Break if all agents report done
            if all(done.values()):
                break

        # Verify asymmetric observation patterns based on mode
        if observation_mode == "frequency":
            # In frequency mode, agent_0 should observe most, followed by agent_1, then other_agent_0
            assert agent_participation["agent_0"] >= agent_participation["agent_1"]
            assert (
                agent_participation["agent_1"] >= agent_participation["other_agent_0"]
            )

            # At least 20% of steps should have all agents observing
            assert steps_with_all_agents > 0, "No steps with all agents observing"
        else:
            # In probability mode, check if participation roughly matches probabilities
            total_steps = sum(agent_participation.values())
            if total_steps > 0:
                # Just verify agent_0 has highest participation
                assert agent_participation["agent_0"] >= agent_participation["agent_1"]

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
            observations,  # next_states
            next_dones,
        )

        # Train on collected experiences if we have any
        if any(len(states[agent_id]) > 0 for agent_id in states):
            loss_info = agent.learn(experiences)

            # Verify that learning worked for at least one agent
            assert any(agent_id in loss_info for agent_id in agent.shared_agent_ids)

        # Verify that agent can still get actions after learning
        if observations:
            final_actions, _, _, _ = agent.get_action(observations, infos)
            assert all(agent_id in observations for agent_id in final_actions)

    # Final test: verify agent can handle completely different set of active agents
    test_observations, test_infos = env.reset()
    test_actions, _, _, _ = agent.get_action(test_observations, test_infos)
    assert all(agent_id in test_observations for agent_id in test_actions)
