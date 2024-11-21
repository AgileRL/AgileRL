import copy
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
from torch._dynamo import OptimizedModule
from gymnasium import spaces
from gymnasium.spaces import Discrete

from agilerl.algorithms.matd3 import MATD3
from agilerl.networks.custom_components import GumbelSoftmax
from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.utils.utils import make_multi_agent_vect_envs
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.test_algorithms.test_maddpg import DummyMultiEnv
from tests.helper_functions import generate_multi_agent_box_spaces, generate_multi_agent_discrete_spaces

class MultiAgentCNNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=4, out_channels=16, kernel_size=(1, 3, 3), stride=4
        )
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=2
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
        x = self.output_activation(self.fc2(x))

        return x


class MultiAgentCNNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=4, out_channels=16, kernel_size=(2, 3, 3), stride=4
        )
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(15202, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, state_tensor, action_tensor):
        x = self.relu(self.conv1(state_tensor))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action_tensor], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DummyEvolvableMLP(EvolvableMLP):
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


class DummyEvolvableCNN(EvolvableCNN):
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
        GumbelSoftmax(),
    )
    return net


@pytest.fixture
def mlp_critic(action_spaces, observation_spaces):
    net = nn.Sequential(
        nn.Linear(observation_spaces[0].shape[0] + action_spaces[0].n, 64), nn.ReLU(), nn.Linear(64, 1)
    )
    return net


@pytest.fixture
def cnn_actor():
    net = MultiAgentCNNActor()
    return net


@pytest.fixture
def cnn_critic():
    net = MultiAgentCNNCritic()
    return net


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def mocked_accelerator():
    MagicMock(spec=Accelerator)


@pytest.fixture
def accelerated_experiences(batch_size, observation_spaces, action_spaces, agent_ids):
    one_hot = all(isinstance(space, Discrete) for space in observation_spaces)
    discrete_actions = all(isinstance(space, Discrete) for space in action_spaces)
    state_size = observation_spaces[0].shape if not one_hot else (observation_spaces[0].n,)
    action_size = action_spaces[0].n if discrete_actions else action_spaces[0].shape[0]
    if one_hot:
        states = {
            agent: torch.randint(0, state_size[0], (batch_size, 1)).float()
            for agent in agent_ids
        }
    else:
        states = {agent: torch.randn(batch_size, *state_size) for agent in agent_ids}

    actions = {agent: torch.randn(batch_size, action_size) for agent in agent_ids}
    rewards = {agent: torch.randn(batch_size, 1) for agent in agent_ids}
    dones = {agent: torch.randint(0, 2, (batch_size, 1)) for agent in agent_ids}
    if one_hot:
        next_states = {
            agent: torch.randint(0, state_size[0], (batch_size, 1)).float()
            for agent in agent_ids
        }
    else:
        next_states = {
            agent: torch.randn(batch_size, *state_size) for agent in agent_ids
        }

    return states, actions, rewards, next_states, dones


@pytest.fixture
def experiences(batch_size, observation_spaces, action_spaces, agent_ids, device):
    one_hot = all(isinstance(space, Discrete) for space in observation_spaces)
    discrete_actions = all(isinstance(space, Discrete) for space in action_spaces)
    state_size = observation_spaces[0].shape if not one_hot else (observation_spaces[0].n,)
    action_size = action_spaces[0].n if discrete_actions else action_spaces[0].shape[0]
    if one_hot:
        states = {
            agent: torch.randint(0, state_size[0], (batch_size, 1)).float().to(device)
            for agent in agent_ids
        }
    else:
        states = {
            agent: torch.randn(batch_size, *state_size).to(device)
            for agent in agent_ids
        }

    actions = {
        agent: torch.randn(batch_size, action_size).to(device) for agent in agent_ids
    }
    rewards = {agent: torch.randn(batch_size, 1).to(device) for agent in agent_ids}
    dones = {
        agent: torch.randint(0, 2, (batch_size, 1)).to(device) for agent in agent_ids
    }
    if one_hot:
        next_states = {
            agent: torch.randint(0, state_size[0], (batch_size, 1)).float().to(device)
            for agent in agent_ids
        }
    else:
        next_states = {
            agent: torch.randn(batch_size, *state_size).to(device)
            for agent in agent_ids
        }

    return states, actions, rewards, next_states, dones

@pytest.mark.parametrize(
    "net_config, accelerator_flag, observation_spaces, compile_mode",
    [
        ({"arch": "mlp", "hidden_size": [64, 64]}, False, generate_multi_agent_box_spaces(2, (4,)), None),
        (
            {
                "arch": "cnn",
                "hidden_size": [8],
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
                "normalize": False,
            },
            False,
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            None,
        ),
        (
            {
                "arch": "cnn",
                "hidden_size": [8],
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
                "normalize": False,
            },
            True,
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            None,
        ),
        ({"arch": "mlp", "hidden_size": [64, 64]}, False, generate_multi_agent_box_spaces(2, (4,)), "default"),
        (
            {
                "arch": "cnn",
                "hidden_size": [8],
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
                "normalize": False,
            },
            False,
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            "default",
        ),
        (
            {
                "arch": "cnn",
                "hidden_size": [8],
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
                "normalize": False,
            },
            True,
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            "default",
        ),
    ],
)
def test_initialize_matd3_with_net_config(
    net_config, accelerator_flag, observation_spaces, device, compile_mode
):
    action_spaces = generate_multi_agent_box_spaces(2, (2,))
    n_agents = 2
    agent_ids = ["agent_0", "agent_1"]
    expl_noise = 0.1
    batch_size = 64
    policy_freq = 2
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        net_config=net_config,
        action_spaces=action_spaces,
        n_agents=n_agents,
        agent_ids=agent_ids,
        accelerator=accelerator,
        device=device,
        policy_freq=policy_freq,
        torch_compiler=compile_mode,
    )
    net_config.update({"output_activation": "Softmax"})
    assert matd3.observation_spaces == observation_spaces
    assert matd3.action_spaces == action_spaces
    assert matd3.policy_freq == policy_freq
    assert matd3.n_agents == n_agents
    assert matd3.agent_ids == agent_ids
    for noise_vec in matd3.expl_noise:
        assert torch.all(noise_vec == expl_noise)
    assert matd3.net_config == net_config, matd3.net_config
    assert matd3.batch_size == batch_size
    assert matd3.multi
    assert matd3.total_state_dims == sum(state.shape[0] for state in observation_spaces)
    assert matd3.total_actions == sum(space.shape[0] for space in action_spaces)
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]
    assert matd3.actor_networks is None
    assert matd3.critic_networks is None
    if net_config["arch"] == "mlp":
        evo_type = EvolvableMLP
        assert matd3.arch == "mlp"
    else:
        evo_type = EvolvableCNN
        assert matd3.arch == "cnn"
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in matd3.actors)
        assert all(isinstance(critic, OptimizedModule) for critic in matd3.critics_1)
        assert all(isinstance(critic, OptimizedModule) for critic in matd3.critics_2)
    else:
        assert all(isinstance(actor, evo_type) for actor in matd3.actors)
        assert all(isinstance(critic, evo_type) for critic in matd3.critics_1)
        assert all(isinstance(critic, evo_type) for critic in matd3.critics_2)
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in matd3.actor_optimizers
        )
        assert all(
            isinstance(critic_1_optimizer, optim.Adam)
            for critic_1_optimizer in matd3.critic_1_optimizers
        )
        assert all(
            isinstance(critic_2_optimizer, optim.Adam)
            for critic_2_optimizer in matd3.critic_2_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in matd3.actor_optimizers
        )
        assert all(
            isinstance(critic_1_optimizer, AcceleratedOptimizer)
            for critic_1_optimizer in matd3.critic_1_optimizers
        )
        assert all(
            isinstance(critic_2_optimizer, AcceleratedOptimizer)
            for critic_2_optimizer in matd3.critic_2_optimizers
        )
    assert isinstance(matd3.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_spaces, action_spaces, accelerator_flag, compile_mode",
    [
        (generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), False, "reduce-overhead"),
    ],
)
def test_initialize_matd3_with_mlp_networks_gumbel_softmax(
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    accelerator_flag,
    device,
    compile_mode,
):
    net_config = {
        "arch": "mlp",
        "hidden_size": [64, 64],
        "min_hidden_layers": 1,
        "max_hidden_layers": 3,
        "min_mlp_nodes": 64,
        "max_mlp_nodes": 500,
        "mlp_output_activation": "GumbelSoftmax",
        "mlp_activation": "ReLU",
    }
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1"],
        n_agents=len(observation_spaces),
        net_config=net_config,
        device=device,
        torch_compiler=compile_mode,
    )
    assert matd3.torch_compiler == "default"


@pytest.mark.parametrize(
    "observation_spaces, action_spaces, accelerator_flag, compile_mode",
    [
        (generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), False, None),
        (generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), True, None),
        (generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), False, "default"),
        (generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), True, "default"),
    ],
)
def test_initialize_matd3_with_mlp_networks(
    mlp_actor,
    mlp_critic,
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
        MakeEvolvable(network=mlp_actor, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(2)
    ]
    evo_critics_1 = [
        MakeEvolvable(network=mlp_critic, input_tensor=torch.randn(1, 8), device=device)
        for _ in range(2)
    ]
    evo_critics_2 = [
        MakeEvolvable(network=mlp_critic, input_tensor=torch.randn(1, 8), device=device)
        for _ in range(2)
    ]
    evo_critics = [evo_critics_1, evo_critics_2]
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1"],
        n_agents=len(observation_spaces),
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        policy_freq=2,
        torch_compiler=compile_mode,
    )
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in matd3.actors)
        assert all(isinstance(critic, OptimizedModule) for critic in matd3.critics_1)
        assert all(isinstance(critic, OptimizedModule) for critic in matd3.critics_2)
    else:
        assert all(isinstance(actor, MakeEvolvable) for actor in matd3.actors)
        assert all(isinstance(critic, MakeEvolvable) for critic in matd3.critics_1)
        assert all(isinstance(critic, MakeEvolvable) for critic in matd3.critics_2)
    assert matd3.net_config is None
    assert matd3.arch == "mlp"
    assert matd3.observation_spaces == observation_spaces
    assert matd3.action_spaces == action_spaces
    assert matd3.one_hot is False
    assert matd3.n_agents == 2
    assert matd3.policy_freq == 2
    assert matd3.agent_ids == ["agent_0", "agent_1"]
    assert matd3.discrete_actions is True
    assert matd3.multi
    assert matd3.total_state_dims == sum(state.shape[0] for state in observation_spaces)
    assert matd3.total_actions == sum(space.n for space in action_spaces)
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in matd3.actor_optimizers
        )
        assert all(
            isinstance(critic_1_optimizer, optim.Adam)
            for critic_1_optimizer in matd3.critic_1_optimizers
        )
        assert all(
            isinstance(critic_2_optimizer, optim.Adam)
            for critic_2_optimizer in matd3.critic_2_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in matd3.actor_optimizers
        )
        assert all(
            isinstance(critic_1_optimizer, AcceleratedOptimizer)
            for critic_1_optimizer in matd3.critic_1_optimizers
        )
        assert all(
            isinstance(critic_2_optimizer, AcceleratedOptimizer)
            for critic_2_optimizer in matd3.critic_2_optimizers
        )
    assert isinstance(matd3.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_spaces, action_spaces, accelerator_flag, compile_mode",
    [
        (generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), False, None),
        (generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), True, None),
        (generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), False, "default"),
        (generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), True, "default"),
    ],
)
def test_initialize_matd3_with_cnn_networks(
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
            input_tensor=torch.randn(1, 4, 2, 210, 160),
            device=device,
        )
        for _ in range(2)
    ]
    evo_critics_1 = [
        MakeEvolvable(
            network=cnn_critic,
            input_tensor=torch.randn(1, 4, 2, 210, 160),
            secondary_input_tensor=torch.randn(1, 2),
            device=device,
        )
        for _ in range(2)
    ]
    evo_critics_2 = [
        MakeEvolvable(
            network=cnn_critic,
            input_tensor=torch.randn(1, 4, 2, 210, 160),
            secondary_input_tensor=torch.randn(1, 2),
            device=device,
        )
        for _ in range(2)
    ]
    evo_critics = [evo_critics_1, evo_critics_2]
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1"],
        n_agents=len(observation_spaces),
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        policy_freq=2,
        torch_compiler=compile_mode,
    )
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in matd3.actors)
        assert all(isinstance(critic, OptimizedModule) for critic in matd3.critics_1)
        assert all(isinstance(critic, OptimizedModule) for critic in matd3.critics_2)
    else:
        assert all(isinstance(actor, MakeEvolvable) for actor in matd3.actors)
        assert all(isinstance(critic, MakeEvolvable) for critic in matd3.critics_1)
        assert all(isinstance(critic, MakeEvolvable) for critic in matd3.critics_2)
    assert matd3.net_config is None
    assert matd3.arch == "cnn"
    assert matd3.observation_spaces == observation_spaces
    assert matd3.policy_freq == 2
    assert matd3.action_spaces == action_spaces
    assert matd3.one_hot is False
    assert matd3.n_agents == 2
    assert matd3.agent_ids == ["agent_0", "agent_1"]
    assert matd3.discrete_actions is True
    assert matd3.multi
    assert matd3.total_state_dims == sum(state.shape[0] for state in observation_spaces)
    assert matd3.total_actions == sum(space.n for space in action_spaces)
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in matd3.actor_optimizers
        )
        assert all(
            isinstance(critic_1_optimizer, optim.Adam)
            for critic_1_optimizer in matd3.critic_1_optimizers
        )
        assert all(
            isinstance(critic_2_optimizer, optim.Adam)
            for critic_2_optimizer in matd3.critic_2_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in matd3.actor_optimizers
        )
        assert all(
            isinstance(critic_1_optimizer, AcceleratedOptimizer)
            for critic_1_optimizer in matd3.critic_1_optimizers
        )
        assert all(
            isinstance(critic_2_optimizer, AcceleratedOptimizer)
            for critic_2_optimizer in matd3.critic_2_optimizers
        )
    assert isinstance(matd3.criterion, nn.MSELoss)


@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize(
    "observation_spaces, action_spaces, net, compile_mode",
    [
        (generate_multi_agent_box_spaces(2, (4,)), generate_multi_agent_discrete_spaces(2, 2), "mlp", None),
        (generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), "cnn", None),
        (generate_multi_agent_box_spaces(2, (4,)), generate_multi_agent_discrete_spaces(2, 2), "mlp", "default"),
        (generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), "cnn", "default"),
    ],
)
def test_initialize_matd3_with_evo_networks(
    observation_spaces, action_spaces, net, device, compile_mode, accelerator
):
    if net == "mlp":
        evo_actors = [
            EvolvableMLP(
                num_inputs=observation_spaces[x].shape[0],
                num_outputs=action_spaces[x].n,
                hidden_size=[64, 64],
                mlp_activation="ReLU",
                mlp_output_activation="Tanh",
            )
            for x in range(2)
        ]
        evo_critics = [
            [
                EvolvableMLP(
                    num_inputs=sum(observation_space.shape[0] for observation_space in observation_spaces)
                    + sum(space.n for space in action_spaces),
                    num_outputs=1,
                    hidden_size=[64, 64],
                    mlp_activation="ReLU",
                )
                for x in range(2)
            ]
            for _ in range(2)
        ]
    else:
        evo_actors = [
            EvolvableCNN(
                input_shape=observation_spaces[0].shape,
                num_outputs=action_spaces[0].n,
                channel_size=[8, 8],
                kernel_size=[2, 2],
                stride_size=[1, 1],
                hidden_size=[64, 64],
                mlp_activation="ReLU",
                n_agents=2,
                mlp_output_activation="Tanh",
            )
            for _ in range(2)
        ]
        evo_critics = [
            [
                EvolvableCNN(
                    input_shape=observation_spaces[0].shape,
                    num_outputs=sum(space.n for space in action_spaces),
                    channel_size=[8, 8],
                    kernel_size=[2, 2],
                    stride_size=[1, 1],
                    hidden_size=[64, 64],
                    n_agents=2,
                    critic=True,
                    mlp_activation="ReLU",
                )
                for _ in range(2)
            ]
            for _ in range(2)
        ]
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1"],
        n_agents=len(observation_spaces),
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
    )
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in matd3.actors)
        assert all(isinstance(critic, OptimizedModule) for critic in matd3.critics_1)
        assert all(isinstance(critic, OptimizedModule) for critic in matd3.critics_2)
    else:
        assert all(
            isinstance(actor, (EvolvableMLP, EvolvableCNN)) for actor in matd3.actors
        )
        assert all(
            isinstance(critic, (EvolvableMLP, EvolvableCNN))
            for critic in matd3.critics_1
        )
        assert all(
            isinstance(critic, (EvolvableMLP, EvolvableCNN))
            for critic in matd3.critics_2
        )
    if net == "mlp":
        assert matd3.arch == "mlp"
    else:
        assert matd3.arch == "cnn"
    assert matd3.observation_spaces == observation_spaces
    assert matd3.policy_freq == 2
    assert matd3.action_spaces == action_spaces
    assert matd3.one_hot is False
    assert matd3.n_agents == 2
    assert matd3.agent_ids == ["agent_0", "agent_1"]
    assert matd3.discrete_actions is True
    assert matd3.multi
    assert matd3.total_state_dims == sum(state.shape[0] for state in observation_spaces)
    assert matd3.total_actions == sum(space.n for space in action_spaces)
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in matd3.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in matd3.critic_1_optimizers
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in matd3.critic_2_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in matd3.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in matd3.critic_1_optimizers
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in matd3.critic_2_optimizers
        )

    assert isinstance(matd3.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_spaces, action_spaces, compile_mode",
    [
        (generate_multi_agent_box_spaces(2, (4,)), generate_multi_agent_discrete_spaces(2, 2), None),
        (generate_multi_agent_box_spaces(2, (4,)), generate_multi_agent_discrete_spaces(2, 2), "default"),
    ],
)
def test_initialize_matd3_with_incorrect_evo_networks(
    observation_spaces, action_spaces, compile_mode
):
    evo_actors = []
    evo_critics = []

    with pytest.raises(AssertionError):
        matd3 = MATD3(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=["agent_0", "agent_1"],
            n_agents=len(observation_spaces),
            actor_networks=evo_actors,
            critic_networks=evo_critics,
            torch_compiler=compile_mode,
        )
        assert matd3


@pytest.mark.parametrize(
    "observation_spaces, action_spaces, compile_mode",
    [
        (generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), None),
        (generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), "default"),
    ],
)
def test_matd3_init_warning(mlp_actor, observation_spaces, action_spaces, device, compile_mode):
    warning_string = "Actor and critic network lists must both be supplied to use custom networks. Defaulting to net config."
    evo_actors = [
        MakeEvolvable(network=mlp_actor, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(2)
    ]
    with pytest.warns(UserWarning, match=warning_string):
        MATD3(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=["agent_0", "agent_1"],
            n_agents=len(observation_spaces),
            actor_networks=evo_actors,
            device=device,
            torch_compiler=compile_mode,
        )


@pytest.mark.parametrize(
    "mode", (None, 0, False, "default", "reduce-overhead", "max-autotune")
)
def test_matd3_init_with_compile_no_error(mode):
    matd3 = MATD3(
        observation_spaces=generate_multi_agent_box_spaces(2, (1,)),
        action_spaces=generate_multi_agent_discrete_spaces(2, 1),
        agent_ids=["agent_0", "agent_1"],
        n_agents=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_compiler=mode,
    )
    if isinstance(mode, str):
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule)
            for a in matd3.actors
        )
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule)
            for a in matd3.actor_targets
        )
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule)
            for a in matd3.critics_1
        )
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule)
            for a in matd3.critics_2
        )
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule)
            for a in matd3.critic_targets_1
        )
        assert all(
            isinstance(a, torch._dynamo.eval_frame.OptimizedModule)
            for a in matd3.critic_targets_2
        )
        assert matd3.torch_compiler == "default"
    else:
        assert isinstance(matd3, MATD3)


@pytest.mark.parametrize("mode", (1, True, "max-autotune-no-cudagraphs"))
def test_matd3_init_with_compile_error(mode):
    err_string = (
        "Choose between torch compiler modes: "
        "default, reduce-overhead, max-autotune or None"
    )
    with pytest.raises(AssertionError, match=err_string):
        MATD3(
            observation_spaces=generate_multi_agent_box_spaces(2, (1,)),
            action_spaces=generate_multi_agent_discrete_spaces(2, 1),
            agent_ids=["agent_0", "agent_1"],
            n_agents=2,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_compiler=mode,
        )


@pytest.mark.parametrize(
    "training, observation_spaces, action_spaces, compile_mode",
    [
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,), low=-1, high=1), None),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,), low=-1, high=1), None),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), None),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), None),
        (1, generate_multi_agent_discrete_spaces(2, 6), generate_multi_agent_box_spaces(2, (2,), low=-1, high=1), None),
        (0, generate_multi_agent_discrete_spaces(2, 6), generate_multi_agent_box_spaces(2, (2,), low=-1, high=1), None),
        (1, generate_multi_agent_discrete_spaces(2, 6), generate_multi_agent_discrete_spaces(2, 2), None),
        (0, generate_multi_agent_discrete_spaces(2, 6), generate_multi_agent_discrete_spaces(2, 2), None),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,), low=-1, high=1), "default"),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,), low=-1, high=1), "default"),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), "default"),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), "default"),
        (1, generate_multi_agent_discrete_spaces(2, 6), generate_multi_agent_box_spaces(2, (2,), low=-1, high=1), "default"),
        (0, generate_multi_agent_discrete_spaces(2, 6), generate_multi_agent_box_spaces(2, (2,), low=-1, high=1), "default"),
        (1, generate_multi_agent_discrete_spaces(2, 6), generate_multi_agent_discrete_spaces(2, 2), "default"),
        (0, generate_multi_agent_discrete_spaces(2, 6), generate_multi_agent_discrete_spaces(2, 2), "default"),
    ],
)
def test_matd3_get_action_mlp(
    training, observation_spaces, action_spaces, device, compile_mode
):
    agent_ids = ["agent_0", "agent_1"]
    if all(isinstance(space, spaces.Discrete) for space in observation_spaces):
        state = {
            agent: np.random.randint(0, observation_spaces[idx].shape, 1)
            for idx, agent in enumerate(agent_ids)
        }
    else:
        state = {
            agent: np.random.randn(*observation_spaces[idx].shape)
            for idx, agent in enumerate(agent_ids)
        }

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        net_config={"arch": "mlp", "hidden_size": [8, 8]},
        n_agents=2,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )
    cont_actions, discrete_action = matd3.get_action(state, training)
    discrete_actions = all(isinstance(space, spaces.Discrete) for space in action_spaces)
    for idx, env_actions in enumerate(list(cont_actions.values())):
        action_dim = action_spaces[idx].shape[0] if isinstance(action_spaces[idx], spaces.Box) else action_spaces[idx].n
        for action in env_actions:
            assert len(action) == action_dim
            if discrete_actions:
                torch.testing.assert_close(
                    sum(action),
                    1.0,
                    atol=0.1,
                    rtol=1e-3,
                )
            act = action[idx]
            assert act.dtype == np.float32
            assert -1 <= act.all() <= 1

    if discrete_actions:
        for idx, env_action in enumerate(list(discrete_action.values())):
            for action in env_action:
                assert action <= action_spaces[idx].n - 1
    matd3 = None


@pytest.mark.parametrize(
    "training, observation_spaces, action_spaces, compile_mode",
    [
        (1, generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255), generate_multi_agent_box_spaces(2, (2,)), None),
        (0, generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255), generate_multi_agent_box_spaces(2, (2,)), None),
        (1, generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), None),
        (0, generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), None),
        (1, generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255), generate_multi_agent_box_spaces(2, (2,)), "default"),
        (0, generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255), generate_multi_agent_box_spaces(2, (2,)), "default"),
        (1, generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), "default"),
        (0, generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255), generate_multi_agent_discrete_spaces(2, 2), "default"),
    ],
)
def test_matd3_get_action_cnn(
    training, observation_spaces, action_spaces, device, compile_mode
):
    agent_ids = ["agent_0", "agent_1"]
    net_config = {
        "arch": "cnn",
        "hidden_size": [64, 64],
        "channel_size": [16],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }
    state = {
        agent: np.random.randn(*observation_spaces[idx].shape) for idx, agent in enumerate(agent_ids)
    }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=agent_ids,
        net_config=net_config,
        device=device,
        torch_compiler=compile_mode,
    )
    cont_actions, discrete_action = matd3.get_action(state, training)
    discrete_actions = all(isinstance(space, spaces.Discrete) for space in action_spaces)
    for idx, env_actions in enumerate(list(cont_actions.values())):
        action_dim = action_spaces[idx].shape[0] if isinstance(action_spaces[idx], spaces.Box) else action_spaces[idx].n
        for action in env_actions:
            assert len(action) == action_dim
            if discrete_actions:
                torch.testing.assert_close(
                    sum(action),
                    1.0,
                    atol=0.1,
                    rtol=1e-3,
                )
            act = action[idx]
            assert act.dtype == np.float32
            assert -1 <= act.all() <= 1

    if discrete_actions:
        for idx, env_action in enumerate(list(discrete_action.values())):
            action_dim = action_spaces[idx].shape[0] if isinstance(action_spaces[idx], spaces.Box) else action_spaces[idx].n
            for action in env_action:
                assert action <= action_dim - 1



@pytest.mark.parametrize(
    "training, observation_spaces, action_spaces, compile_mode",
    [
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), None),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), None),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,)), None),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,)), None),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), "default"),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), "default"),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,)), "default"),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,)), "default"),
    ],
)
def test_matd3_get_action_distributed(
    training, observation_spaces, action_spaces, compile_mode
):
    accelerator = Accelerator()
    agent_ids = ["agent_0", "agent_1"]
    state = {
        agent: np.random.randn(*observation_spaces[idx].shape) for idx, agent in enumerate(agent_ids)
    }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=agent_ids,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    new_actors = [
        DummyEvolvableMLP(
            num_inputs=actor.num_inputs,
            num_outputs=actor.num_outputs,
            hidden_size=actor.hidden_size,
            device=actor.device,
            mlp_output_activation=actor.mlp_output_activation,
        )
        for actor in matd3.actors
    ]
    matd3.actors = new_actors
    cont_actions, discrete_action = matd3.get_action(state, training)
    discrete_actions = all(isinstance(space, spaces.Discrete) for space in action_spaces)
    for idx, env_actions in enumerate(list(cont_actions.values())):
        action_dim = action_spaces[idx].shape[0] if isinstance(action_spaces[idx], spaces.Box) else action_spaces[idx].n
        for action in env_actions:
            assert len(action) == action_dim
            if discrete_actions:
                torch.testing.assert_close(
                    sum(action),
                    1.0,
                    atol=0.1,
                    rtol=1e-3,
                )
            act = action[idx]
            assert act.dtype == np.float32
            assert -1 <= act.all() <= 1

    if discrete_actions:
        for idx, env_action in enumerate(list(discrete_action.values())):
            action_dim = action_spaces[idx].shape[0] if isinstance(action_spaces[idx], spaces.Box) else action_spaces[idx].n
            for action in env_action:
                assert action <= action_dim - 1


@pytest.mark.parametrize(
    "training, observation_spaces, action_spaces, compile_mode",
    [
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,)), None),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,)), None),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), None),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), None),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,)), "default"),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (2,)), "default"),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), "default"),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 2), "default"),
    ],
)
def test_matd3_get_action_agent_masking(
    training, observation_spaces, action_spaces, compile_mode, device
):
    agent_ids = ["agent_0", "agent_1"]
    state = {agent: np.random.randn(*observation_spaces[0].shape) for agent in agent_ids}
    discrete_actions = all(isinstance(space, spaces.Discrete) for space in action_spaces)
    if discrete_actions:
        info = {
            "agent_0": {"env_defined_actions": 1},
            "agent_1": {"env_defined_actions": None},
        }
    else:
        info = {
            "agent_0": {"env_defined_actions": np.array([0, 1])},
            "agent_1": {"env_defined_actions": None},
        }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )
    cont_actions, discrete_action = matd3.get_action(state, training, infos=info)
    if discrete_actions:
        assert np.array_equal(
            discrete_action["agent_0"], np.array([[1]])
        ), discrete_action["agent_0"]
    else:
        assert np.array_equal(
            cont_actions["agent_0"], np.array([[0, 1]])
        ), cont_actions["agent_0"]


@pytest.mark.parametrize(
    "training, observation_spaces, action_spaces, compile_mode",
    [
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (6,)), None),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (6,)), None),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 6), None),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 6), None),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (6,)), "default"),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_box_spaces(2, (6,)), "default"),
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 6), "default"),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 6), "default"),
    ],
)
def test_matd3_get_action_vectorized_agent_masking(
    training, observation_spaces, action_spaces, device, compile_mode
):
    num_envs = 6
    agent_ids = ["agent_0", "agent_1"]
    state = {
        agent: np.array([np.random.randn(*observation_spaces[0].shape) for _ in range(num_envs)])
        for agent in agent_ids
    }
    discrete_actions = all(isinstance(space, spaces.Discrete) for space in action_spaces)
    if discrete_actions:
        env_defined_action = np.array(
            [np.random.randint(0, observation_spaces[0].shape[0] + 1) for _ in range(num_envs)]
        )
    else:
        env_defined_action = np.array(
            [np.random.randn(*observation_spaces[0].shape) for _ in range(num_envs)]
        )
    nan_array = np.zeros(env_defined_action.shape)
    nan_array[:] = np.nan
    info = {
        "agent_0": {"env_defined_actions": env_defined_action},
        "agent_1": {"env_defined_actions": nan_array},
    }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )
    cont_actions, discrete_action = matd3.get_action(state, training, infos=info)
    if discrete_actions:
        assert np.array_equal(
            discrete_action["agent_0"].squeeze(), info["agent_0"]["env_defined_actions"]
        ), discrete_action["agent_0"]
    else:
        assert np.isclose(
            cont_actions["agent_0"], info["agent_0"]["env_defined_actions"]
        ).all(), cont_actions["agent_0"]


@pytest.mark.parametrize(
    "training, observation_spaces, action_spaces",
    [
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 4)),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 4)),
    ],
)
def test_matd3_get_action_action_masking_exception(
    training, observation_spaces, action_spaces, device
):
    agent_ids = ["agent_0", "agent_1"]
    state = {
        agent: {
            "observation": np.random.randn(*observation_spaces[idx].shape),
            "action_mask": [0, 1, 0, 1],
        }
        for idx, agent in enumerate(agent_ids)
    }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        net_config={"arch": "mlp", "hidden_size": [64, 64]},
        n_agents=2,
        agent_ids=agent_ids,
        device=device,
    )
    with pytest.raises(AssertionError):
        _, discrete_action = matd3.get_action(state, training)


@pytest.mark.parametrize(
    "training, observation_spaces, action_spaces",
    [
        (1, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 4)),
        (0, generate_multi_agent_box_spaces(2, (6,)), generate_multi_agent_discrete_spaces(2, 4)),
    ],
)
def test_matd3_get_action_action_masking(
    training, observation_spaces, action_spaces, device
):
    agent_ids = ["agent_0", "agent_1"]
    state = {
        agent: np.random.randn(*observation_spaces[idx].shape) for idx, agent in enumerate(agent_ids)
    }
    info = {
        agent: {
            "action_mask": [0, 1, 0, 1],
        }
        for idx, agent in enumerate(agent_ids)
    }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        net_config={"arch": "mlp", "hidden_size": [64, 64]},
        n_agents=2,
        agent_ids=agent_ids,
        device=device,
    )
    _, discrete_action = matd3.get_action(state, training, info)
    assert all(i in [1, 3] for i in discrete_action.values())


@pytest.mark.parametrize(
    "observation_spaces, batch_size, action_spaces, agent_ids, compile_mode",
    [
        (generate_multi_agent_box_spaces(2, (6,)), 64, generate_multi_agent_box_spaces(2, (2,)), ["agent_0", "agent_1"], None),
        (generate_multi_agent_discrete_spaces(2, 6), 64, generate_multi_agent_box_spaces(2, (2,)), ["agent_0", "agent_1"], None),
        (generate_multi_agent_box_spaces(2, (6,)), 64, generate_multi_agent_discrete_spaces(2, 2), ["agent_0", "agent_1"], None),
        (generate_multi_agent_discrete_spaces(2, 6), 64, generate_multi_agent_discrete_spaces(2, 2), ["agent_0", "agent_1"], None),
        (generate_multi_agent_box_spaces(2, (6,)), 64, generate_multi_agent_box_spaces(2, (2,)), ["agent_0", "agent_1"], "default"),
        (generate_multi_agent_discrete_spaces(2, 6), 64, generate_multi_agent_box_spaces(2, (2,)), ["agent_0", "agent_1"], "default"),
        (generate_multi_agent_box_spaces(2, (6,)), 64, generate_multi_agent_discrete_spaces(2, 2), ["agent_0", "agent_1"], "default"),
        (generate_multi_agent_discrete_spaces(2, 6), 64, generate_multi_agent_discrete_spaces(2, 2), ["agent_0", "agent_1"], "default"),
    ],
)
def test_matd3_learns_from_experiences_mlp(
    observation_spaces,
    experiences,
    batch_size,
    action_spaces,
    agent_ids,
    device,
    compile_mode,
):
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    agent_ids = ["agent_0", "agent_1"]
    policy_freq = 2
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=agent_ids,
        device=device,
        policy_freq=policy_freq,
        torch_compiler=compile_mode,
    )
    actors = matd3.actors
    actor_targets = matd3.actor_targets
    actors_pre_learn_sd = [copy.deepcopy(actor.state_dict()) for actor in matd3.actors]
    critics_1 = matd3.critics_1
    critic_targets_1 = matd3.critic_targets_1
    critics_2 = matd3.critics_2
    critic_targets_2 = matd3.critic_targets_2
    critics_1_pre_learn_sd = [
        str(copy.deepcopy(critic_1.state_dict())) for critic_1 in matd3.critics_1
    ]
    critics_2_pre_learn_sd = [
        str(copy.deepcopy(critic_2.state_dict())) for critic_2 in matd3.critics_2
    ]

    for _ in range(4 * policy_freq):
        matd3.scores.append(0)
        loss = matd3.learn(experiences)

    assert isinstance(loss, dict)
    for agent_id in matd3.agent_ids:
        assert loss[agent_id][-1] >= 0.0
    for old_actor, updated_actor in zip(actors, matd3.actors):
        assert old_actor == updated_actor
    for old_actor_target, updated_actor_target in zip(
        actor_targets, matd3.actor_targets
    ):
        assert old_actor_target == updated_actor_target
    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, matd3.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())
    for old_critic_1, updated_critic_1 in zip(critics_1, matd3.critics_1):
        assert old_critic_1 == updated_critic_1
    for old_critic_target_1, updated_critic_target_1 in zip(
        critic_targets_1, matd3.critic_targets_1
    ):
        assert old_critic_target_1 == updated_critic_target_1
    for old_critic_1_state_dict, updated_critic_1 in zip(
        critics_1_pre_learn_sd, matd3.critics_1
    ):
        assert old_critic_1_state_dict != str(updated_critic_1.state_dict())
    for old_critic_2, updated_critic_2 in zip(critics_2, matd3.critics_2):
        assert old_critic_2 == updated_critic_2
    for old_critic_target_2, updated_critic_target_2 in zip(
        critic_targets_2, matd3.critic_targets_2
    ):
        assert old_critic_target_2 == updated_critic_target_2
    for old_critic_2_state_dict, updated_critic_2 in zip(
        critics_2_pre_learn_sd, matd3.critics_2
    ):
        assert old_critic_2_state_dict != str(updated_critic_2.state_dict())


def no_sync(self):
    class DummyNoSync:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass  # Add cleanup or handling if needed

    return DummyNoSync()


@pytest.mark.parametrize(
    "observation_spaces, batch_size, action_spaces, agent_ids, compile_mode",
    [
        (generate_multi_agent_box_spaces(2, (6,)), 64, generate_multi_agent_box_spaces(2, (2,)), ["agent_0", "agent_1"], None),
        (generate_multi_agent_discrete_spaces(2, 6), 64, generate_multi_agent_box_spaces(2, (2,)), ["agent_0", "agent_1"], None),
        (generate_multi_agent_box_spaces(2, (6,)), 64, generate_multi_agent_discrete_spaces(2, 2), ["agent_0", "agent_1"], None),
        (generate_multi_agent_discrete_spaces(2, 6), 64, generate_multi_agent_discrete_spaces(2, 2), ["agent_0", "agent_1"], None),
        (generate_multi_agent_box_spaces(2, (6,)), 64, generate_multi_agent_box_spaces(2, (2,)), ["agent_0", "agent_1"], "default"),
        (generate_multi_agent_discrete_spaces(2, 6), 64, generate_multi_agent_box_spaces(2, (2,)), ["agent_0", "agent_1"], "default"),
        (generate_multi_agent_box_spaces(2, (6,)), 64, generate_multi_agent_discrete_spaces(2, 2), ["agent_0", "agent_1"], "default"),
        (generate_multi_agent_discrete_spaces(2, 6), 64, generate_multi_agent_discrete_spaces(2, 2), ["agent_0", "agent_1"], "default"),
    ],
)
def test_matd3_learns_from_experiences_mlp_distributed(
    observation_spaces,
    accelerated_experiences,
    batch_size,
    action_spaces,
    agent_ids,
    compile_mode,
):
    accelerator = Accelerator(device_placement=False)
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    agent_ids = ["agent_0", "agent_1"]
    policy_freq = 2
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=agent_ids,
        accelerator=accelerator,
        policy_freq=policy_freq,
        torch_compiler=compile_mode,
    )

    for (
        actor,
        critic_1,
        critic_2,
        actor_target,
        critic_target_1,
        critic_target_2,
    ) in zip(
        matd3.actors,
        matd3.critics_1,
        matd3.critics_2,
        matd3.actor_targets,
        matd3.critic_targets_1,
        matd3.critic_targets_2,
    ):
        actor.no_sync = no_sync.__get__(actor)
        critic_1.no_sync = no_sync.__get__(critic_1)
        critic_2.no_sync = no_sync.__get__(critic_2)
        actor_target.no_sync = no_sync.__get__(actor_target)
        critic_target_1.no_sync = no_sync.__get__(critic_target_1)
        critic_target_2.no_sync = no_sync.__get__(critic_target_2)

    actors = matd3.actors
    actor_targets = matd3.actor_targets
    actors_pre_learn_sd = [copy.deepcopy(actor.state_dict()) for actor in matd3.actors]
    critics_1 = matd3.critics_1
    critic_targets_1 = matd3.critic_targets_1
    critics_2 = matd3.critics_2
    critic_targets_2 = matd3.critic_targets_2
    critics_1_pre_learn_sd = [
        str(copy.deepcopy(critic_1.state_dict())) for critic_1 in matd3.critics_1
    ]
    critics_2_pre_learn_sd = [
        str(copy.deepcopy(critic_2.state_dict())) for critic_2 in matd3.critics_2
    ]

    for _ in range(4 * policy_freq):
        matd3.scores.append(0)
        loss = matd3.learn(accelerated_experiences)

    assert isinstance(loss, dict)
    for agent_id in matd3.agent_ids:
        assert loss[agent_id][-1] >= 0.0
    for old_actor, updated_actor in zip(actors, matd3.actors):
        assert old_actor == updated_actor
    for old_actor_target, updated_actor_target in zip(
        actor_targets, matd3.actor_targets
    ):
        assert old_actor_target == updated_actor_target
    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, matd3.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())
    for old_critic_1, updated_critic_1 in zip(critics_1, matd3.critics_1):
        assert old_critic_1 == updated_critic_1
    for old_critic_target_1, updated_critic_target_1 in zip(
        critic_targets_1, matd3.critic_targets_1
    ):
        assert old_critic_target_1 == updated_critic_target_1
    for old_critic_1_state_dict, updated_critic_1 in zip(
        critics_1_pre_learn_sd, matd3.critics_1
    ):
        assert old_critic_1_state_dict != str(updated_critic_1.state_dict())
    for old_critic_2, updated_critic_2 in zip(critics_2, matd3.critics_2):
        assert old_critic_2 == updated_critic_2
    for old_critic_target_2, updated_critic_target_2 in zip(
        critic_targets_2, matd3.critic_targets_2
    ):
        assert old_critic_target_2 == updated_critic_target_2
    for old_critic_2_state_dict, updated_critic_2 in zip(
        critics_2_pre_learn_sd, matd3.critics_2
    ):
        assert old_critic_2_state_dict != str(updated_critic_2.state_dict())


@pytest.mark.parametrize(
    "observation_spaces, batch_size, action_spaces, agent_ids, compile_mode",
    [
        (
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            64,
            generate_multi_agent_box_spaces(2, (2,)),
            ["agent_0", "agent_1"],
            None,
        ),
        (
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            64,
            generate_multi_agent_discrete_spaces(2, 2),
            ["agent_0", "agent_1"],
            "default",
        ),
    ],
)
def test_matd3_learns_from_experiences_cnn(
    observation_spaces,
    experiences,
    batch_size,
    action_spaces,
    agent_ids,
    device,
    compile_mode,
):
    agent_ids = ["agent_0", "agent_1"]
    policy_freq = 2
    net_config = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [16],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        net_config=net_config,
        agent_ids=agent_ids,
        device=device,
        policy_freq=policy_freq,
        torch_compiler=compile_mode,
    )
    actors = matd3.actors
    actor_targets = matd3.actor_targets
    actors_pre_learn_sd = [copy.deepcopy(actor.state_dict()) for actor in matd3.actors]
    critics_1 = matd3.critics_1
    critic_targets_1 = matd3.critic_targets_1
    critics_2 = matd3.critics_2
    critic_targets_2 = matd3.critic_targets_2
    critics_1_pre_learn_sd = [
        str(copy.deepcopy(critic_1.state_dict())) for critic_1 in matd3.critics_1
    ]
    critics_2_pre_learn_sd = [
        str(copy.deepcopy(critic_2.state_dict())) for critic_2 in matd3.critics_2
    ]

    for _ in range(100 * policy_freq):
        matd3.scores.append(0)
        loss = matd3.learn(experiences)

    assert isinstance(loss, dict)
    for agent_id in matd3.agent_ids:
        assert loss[agent_id][-1] >= 0.0
    for old_actor, updated_actor in zip(actors, matd3.actors):
        assert old_actor == updated_actor
    for old_actor_target, updated_actor_target in zip(
        actor_targets, matd3.actor_targets
    ):
        assert old_actor_target == updated_actor_target
    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, matd3.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())
    for old_critic_1, updated_critic_1 in zip(critics_1, matd3.critics_1):
        assert old_critic_1 == updated_critic_1
    for old_critic_target_1, updated_critic_target_1 in zip(
        critic_targets_1, matd3.critic_targets_1
    ):
        assert old_critic_target_1 == updated_critic_target_1
    for old_critic_1_state_dict, updated_critic_1 in zip(
        critics_1_pre_learn_sd, matd3.critics_1
    ):
        assert old_critic_1_state_dict != str(updated_critic_1.state_dict())
    for old_critic_2, updated_critic_2 in zip(critics_2, matd3.critics_2):
        assert old_critic_2 == updated_critic_2
    for old_critic_target_2, updated_critic_target_2 in zip(
        critic_targets_2, matd3.critic_targets_2
    ):
        assert old_critic_target_2 == updated_critic_target_2
    for old_critic_2_state_dict, updated_critic_2 in zip(
        critics_2_pre_learn_sd, matd3.critics_2
    ):
        assert old_critic_2_state_dict != str(updated_critic_2.state_dict())


@pytest.mark.parametrize(
    "observation_spaces, batch_size, action_spaces, agent_ids, compile_mode",
    [
        (
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            64,
            generate_multi_agent_box_spaces(2, (2,)),
            ["agent_0", "agent_1"],
            None,
        ),
        (
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            64,
            generate_multi_agent_discrete_spaces(2, 2),
            ["agent_0", "agent_1"],
            None,
        ),
        (
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            64,
            generate_multi_agent_box_spaces(2, (2,)),
            ["agent_0", "agent_1"],
            "default",
        ),
        (
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            64,
            generate_multi_agent_discrete_spaces(2, 2),
            ["agent_0", "agent_1"],
            "default",
        ),
    ],
)
def test_matd3_learns_from_experiences_cnn_distributed(
    observation_spaces,
    accelerated_experiences,
    batch_size,
    action_spaces,
    agent_ids,
    device,
    compile_mode,
):
    accelerator = Accelerator(device_placement=False)
    agent_ids = ["agent_0", "agent_1"]
    net_config = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [16],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }
    policy_freq = 2
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        net_config=net_config,
        agent_ids=agent_ids,
        accelerator=accelerator,
        policy_freq=policy_freq,
        torch_compiler=compile_mode,
    )

    for (
        actor,
        critic_1,
        critic_2,
        actor_target,
        critic_target_1,
        critic_target_2,
    ) in zip(
        matd3.actors,
        matd3.critics_1,
        matd3.critics_2,
        matd3.actor_targets,
        matd3.critic_targets_1,
        matd3.critic_targets_2,
    ):
        actor.no_sync = no_sync.__get__(actor)
        critic_1.no_sync = no_sync.__get__(critic_1)
        critic_2.no_sync = no_sync.__get__(critic_2)
        actor_target.no_sync = no_sync.__get__(actor_target)
        critic_target_1.no_sync = no_sync.__get__(critic_target_1)
        critic_target_2.no_sync = no_sync.__get__(critic_target_2)

    actors = matd3.actors
    actor_targets = matd3.actor_targets
    actors_pre_learn_sd = [copy.deepcopy(actor.state_dict()) for actor in matd3.actors]
    critics_1 = matd3.critics_1
    critic_targets_1 = matd3.critic_targets_1
    critics_1_pre_learn_sd = [
        str(copy.deepcopy(critic_1.state_dict())) for critic_1 in matd3.critics_1
    ]
    critics_2 = matd3.critics_2
    critic_targets_2 = matd3.critic_targets_2
    critics_2_pre_learn_sd = [
        str(copy.deepcopy(critic_2.state_dict())) for critic_2 in matd3.critics_2
    ]

    for _ in range(4):
        matd3.scores.append(0)
        loss = matd3.learn(accelerated_experiences)

    assert isinstance(loss, dict)
    for agent_id in matd3.agent_ids:
        assert loss[agent_id][-1] >= 0.0
    for old_actor, updated_actor in zip(actors, matd3.actors):
        assert old_actor == updated_actor
    for old_actor_target, updated_actor_target in zip(
        actor_targets, matd3.actor_targets
    ):
        assert old_actor_target == updated_actor_target
    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, matd3.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())

    for old_critic_1, updated_critic_1 in zip(critics_1, matd3.critics_1):
        assert old_critic_1 == updated_critic_1
    for old_critic_target_1, updated_critic_target_1 in zip(
        critic_targets_1, matd3.critic_targets_1
    ):
        assert old_critic_target_1 == updated_critic_target_1
    for old_critic_1_state_dict, updated_critic_1 in zip(
        critics_1_pre_learn_sd, matd3.critics_1
    ):
        assert old_critic_1_state_dict != str(updated_critic_1.state_dict())

    for old_critic_2, updated_critic_2 in zip(critics_2, matd3.critics_2):
        assert old_critic_2 == updated_critic_2
    for old_critic_target_2, updated_critic_target_2 in zip(
        critic_targets_2, matd3.critic_targets_2
    ):
        assert old_critic_target_2 == updated_critic_target_2
    for old_critic_2_state_dict, updated_critic_2 in zip(
        critics_2_pre_learn_sd, matd3.critics_2
    ):
        assert old_critic_2_state_dict != str(updated_critic_2.state_dict())


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_soft_update(device, compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(2, (6,))
    action_spaces = generate_multi_agent_box_spaces(2, (2,))
    accelerator = None

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        accelerator=accelerator,
        device=device,
        torch_compiler=compile_mode,
    )

    for (
        actor,
        actor_target,
        critic_1,
        critic_target_1,
        critic_2,
        critic_target_2,
    ) in zip(
        matd3.actors,
        matd3.actor_targets,
        matd3.critics_1,
        matd3.critic_targets_1,
        matd3.critics_2,
        matd3.critic_targets_2,
    ):
        # Check actors
        matd3.soft_update(actor, actor_target)
        eval_params = list(actor.parameters())
        target_params = list(actor_target.parameters())
        expected_params = [
            matd3.tau * eval_param + (1.0 - matd3.tau) * target_param
            for eval_param, target_param in zip(eval_params, target_params)
        ]
        assert all(
            torch.allclose(expected_param, target_param)
            for expected_param, target_param in zip(expected_params, target_params)
        )
        matd3.soft_update(critic_1, critic_target_1)
        eval_params = list(critic_1.parameters())
        target_params = list(critic_target_1.parameters())
        expected_params = [
            matd3.tau * eval_param + (1.0 - matd3.tau) * target_param
            for eval_param, target_param in zip(eval_params, target_params)
        ]

        assert all(
            torch.allclose(expected_param, target_param)
            for expected_param, target_param in zip(expected_params, target_params)
        )
        matd3.soft_update(critic_2, critic_target_2)
        eval_params = list(critic_2.parameters())
        target_params = list(critic_target_2.parameters())
        expected_params = [
            matd3.tau * eval_param + (1.0 - matd3.tau) * target_param
            for eval_param, target_param in zip(eval_params, target_params)
        ]

        assert all(
            torch.allclose(expected_param, target_param)
            for expected_param, target_param in zip(expected_params, target_params)
        )


@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("sum_score", [True, False])
def test_matd3_algorithm_test_loop(device, compile_mode, sum_score):
    observation_spaces = generate_multi_agent_box_spaces(2, (6,))
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    accelerator = None

    env = DummyMultiEnv(observation_spaces[0], action_spaces)

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        accelerator=accelerator,
        device=device,
        torch_compiler=compile_mode,
    )
    mean_score = matd3.test(env, max_steps=10, sum_scores=sum_score)
    if sum_score:
        assert isinstance(mean_score, float)
    else:
        assert isinstance(mean_score, np.ndarray)
        assert len(mean_score) == 2


@pytest.mark.parametrize("sum_score", [True, False])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_algorithm_test_loop_cnn(device, sum_score, compile_mode):
    env_observation_spaces = generate_multi_agent_box_spaces(2, (32, 32, 3), low=0, high=255)
    agent_observation_spaces = generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255)
    net_config = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [16],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    accelerator = None
    env = DummyMultiEnv(env_observation_spaces[0], action_spaces)
    matd3 = MATD3(
        agent_observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        net_config=net_config,
        accelerator=accelerator,
        device=device,
        torch_compiler=compile_mode,
    )
    mean_score = matd3.test(env, max_steps=10, swap_channels=True)
    mean_score = matd3.test(env, max_steps=10, swap_channels=True, sum_scores=sum_score)
    if sum_score:
        assert isinstance(mean_score, float)
    else:
        assert isinstance(mean_score, np.ndarray)
        assert len(mean_score) == 2


@pytest.mark.parametrize("sum_score", [True, False])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_algorithm_test_loop_cnn_vectorized(device, sum_score, compile_mode):
    env_observation_spaces = generate_multi_agent_box_spaces(2, (32, 32, 3), low=0, high=255)
    agent_observation_spaces = generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255)
    net_config = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [16],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    accelerator = None
    env = make_multi_agent_vect_envs(
        DummyMultiEnv, 2, **dict(observation_spaces=env_observation_spaces[0], action_spaces=action_spaces)
    )
    matd3 = MATD3(
        agent_observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        net_config=net_config,
        accelerator=accelerator,
        device=device,
        torch_compiler=compile_mode,
    )
    mean_score = matd3.test(env, max_steps=10, swap_channels=True, sum_scores=sum_score)
    if sum_score:
        assert isinstance(mean_score, float)
    else:
        assert isinstance(mean_score, np.ndarray)
        assert len(mean_score) == 2
    env.close()


@pytest.mark.parametrize(
    "accelerator_flag, wrap, compile_mode",
    [
        (False, True, None),
        (True, True, None),
        (True, False, None),
        (False, True, "default"),
        (True, True, "default"),
        (True, False, "default"),
    ],
)
def test_matd3_clone_returns_identical_agent(accelerator_flag, wrap, compile_mode):
    # Clones the agent and returns an identical copy.
    observation_spaces = generate_multi_agent_box_spaces(2, (4,))
    action_spaces = generate_multi_agent_box_spaces(2, (2,), low=-1, high=1)
    n_agents = 2
    agent_ids = ["agent_0", "agent_1"]
    expl_noise = 0.1
    index = 0
    net_config = {"arch": "mlp", "hidden_size": [64, 64]}
    batch_size = 64
    lr_actor = 0.001
    lr_critic = 0.01
    learn_step = 5
    gamma = 0.95
    tau = 0.01
    mut = None
    actor_networks = None
    critic_networks = None
    policy_freq = 2
    device = "cpu"
    if accelerator_flag:
        accelerator = Accelerator(device_placement=False)
    else:
        accelerator = None

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents,
        agent_ids,
        expl_noise=expl_noise,
        index=index,
        policy_freq=policy_freq,
        net_config=net_config,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mut=mut,
        actor_networks=actor_networks,
        critic_networks=critic_networks,
        device=device,
        accelerator=accelerator,
        torch_compiler=compile_mode,
        wrap=wrap,
    )

    clone_agent = matd3.clone(wrap=wrap)

    assert isinstance(clone_agent, MATD3)
    assert clone_agent.observation_spaces == matd3.observation_spaces
    assert clone_agent.action_spaces == matd3.action_spaces
    assert clone_agent.one_hot == matd3.one_hot
    assert clone_agent.n_agents == matd3.n_agents
    assert clone_agent.agent_ids == matd3.agent_ids
    assert np.all(np.stack(clone_agent.max_action) == np.stack(matd3.max_action))
    assert np.all(np.stack(clone_agent.min_action) == np.stack(matd3.min_action))
    assert np.array_equal(clone_agent.expl_noise, matd3.expl_noise)
    assert clone_agent.discrete_actions == matd3.discrete_actions
    assert clone_agent.index == matd3.index
    assert clone_agent.net_config == matd3.net_config
    assert clone_agent.batch_size == matd3.batch_size
    assert clone_agent.lr_actor == matd3.lr_actor
    assert clone_agent.lr_critic == matd3.lr_critic
    assert clone_agent.learn_step == matd3.learn_step
    assert clone_agent.gamma == matd3.gamma
    assert clone_agent.tau == matd3.tau
    assert clone_agent.device == matd3.device
    assert clone_agent.accelerator == matd3.accelerator

    assert clone_agent.torch_compiler == matd3.torch_compiler

    for clone_actor, actor in zip(clone_agent.actors, matd3.actors):
        assert str(clone_actor.state_dict()) == str(actor.state_dict())
    for clone_critic_1, critic_1 in zip(clone_agent.critics_1, matd3.critics_1):
        assert str(clone_critic_1.state_dict()) == str(critic_1.state_dict())
    for clone_actor_target, actor_target in zip(
        clone_agent.actor_targets, matd3.actor_targets
    ):
        assert str(clone_actor_target.state_dict()) == str(actor_target.state_dict())
    for clone_critic_target_1, critic_target_1 in zip(
        clone_agent.critic_targets_1, matd3.critic_targets_1
    ):
        assert str(clone_critic_target_1.state_dict()) == str(
            critic_target_1.state_dict()
        )

    for clone_critic_2, critic_2 in zip(clone_agent.critics_2, matd3.critics_2):
        assert str(clone_critic_2.state_dict()) == str(critic_2.state_dict())

    for clone_critic_target_2, critic_target_2 in zip(
        clone_agent.critic_targets_2, matd3.critic_targets_2
    ):
        assert str(clone_critic_target_2.state_dict()) == str(
            critic_target_2.state_dict()
        )

    assert clone_agent.actor_networks == matd3.actor_networks
    assert clone_agent.critic_networks == matd3.critic_networks


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_clone_new_index(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(2, (4,))
    action_spaces = generate_multi_agent_box_spaces(2, (2,))
    n_agents = 2
    agent_ids = ["agent_0", "agent_1"]

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents,
        agent_ids,
        torch_compiler=compile_mode,
    )
    clone_agent = matd3.clone(index=100)

    assert clone_agent.index == 100

@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_clone_after_learning(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(2, (4,))
    action_spaces = generate_multi_agent_box_spaces(2, (2,))
    n_agents = 2
    agent_ids = ["agent_0", "agent_1"]
    batch_size = 8

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents,
        agent_ids,
        batch_size=batch_size,
        torch_compiler=compile_mode,
    )

    states = {
        agent_id: torch.randn(batch_size, observation_spaces[idx].shape[0])
        for idx, agent_id in enumerate(agent_ids)
    }
    actions = {
        agent_id: torch.randn(batch_size, action_spaces[idx].n)
        for idx, agent_id in enumerate(agent_ids)
    }
    rewards = {agent_id: torch.randn(batch_size, 1) for agent_id in agent_ids}
    next_states = {
        agent_id: torch.randn(batch_size, observation_spaces[idx].shape[0])
        for idx, agent_id in enumerate(agent_ids)
    }
    dones = {agent_id: torch.zeros(batch_size, 1) for agent_id in agent_ids}

    experiences = states, actions, rewards, next_states, dones
    matd3.learn(experiences)
    clone_agent = matd3.clone()
    assert isinstance(clone_agent, MATD3)
    assert clone_agent.observation_spaces == matd3.observation_spaces
    assert clone_agent.action_spaces == matd3.action_spaces
    assert clone_agent.one_hot == matd3.one_hot
    assert clone_agent.n_agents == matd3.n_agents
    assert clone_agent.agent_ids == matd3.agent_ids
    assert np.all(np.stack(clone_agent.max_action) == np.stack(matd3.max_action))
    assert np.all(np.stack(clone_agent.min_action) == np.stack(matd3.min_action))
    assert np.array_equal(clone_agent.expl_noise, matd3.expl_noise)
    assert clone_agent.discrete_actions == matd3.discrete_actions
    assert clone_agent.index == matd3.index
    assert clone_agent.net_config == matd3.net_config
    assert clone_agent.batch_size == matd3.batch_size
    assert clone_agent.lr_actor == matd3.lr_actor
    assert clone_agent.lr_critic == matd3.lr_critic
    assert clone_agent.learn_step == matd3.learn_step
    assert clone_agent.gamma == matd3.gamma
    assert clone_agent.tau == matd3.tau
    assert clone_agent.device == matd3.device
    assert clone_agent.accelerator == matd3.accelerator

    assert clone_agent.torch_compiler == compile_mode
    assert matd3.torch_compiler == compile_mode

    for clone_actor, actor in zip(clone_agent.actors, matd3.actors):
        assert str(clone_actor.state_dict()) == str(actor.state_dict())
    for clone_critic_1, critic_1 in zip(clone_agent.critics_1, matd3.critics_1):
        assert str(clone_critic_1.state_dict()) == str(critic_1.state_dict())
    for clone_actor_target, actor_target in zip(
        clone_agent.actor_targets, matd3.actor_targets
    ):
        assert str(clone_actor_target.state_dict()) == str(actor_target.state_dict())
    for clone_critic_target_1, critic_target_1 in zip(
        clone_agent.critic_targets_1, matd3.critic_targets_1
    ):
        assert str(clone_critic_target_1.state_dict()) == str(
            critic_target_1.state_dict()
        )

    for clone_critic_2, critic_2 in zip(clone_agent.critics_2, matd3.critics_2):
        assert str(clone_critic_2.state_dict()) == str(critic_2.state_dict())

    for clone_critic_target_2, critic_target_2 in zip(
        clone_agent.critic_targets_2, matd3.critic_targets_2
    ):
        assert str(clone_critic_target_2.state_dict()) == str(
            critic_target_2.state_dict()
        )

    assert clone_agent.actor_networks == matd3.actor_networks
    assert clone_agent.critic_networks == matd3.critic_networks

    for clone_actor_opt, actor_opt in zip(
        clone_agent.actor_optimizers, matd3.actor_optimizers
    ):
        assert str(clone_actor_opt) == str(actor_opt)
    for clone_critic_opt_1, critic_opt_1 in zip(
        clone_agent.critic_1_optimizers, matd3.critic_1_optimizers
    ):
        assert str(clone_critic_opt_1) == str(critic_opt_1)
    for clone_critic_opt_2, critic_opt_2 in zip(
        clone_agent.critic_2_optimizers, matd3.critic_2_optimizers
    ):
        assert str(clone_critic_opt_2) == str(critic_opt_2)


@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize(
    "accelerator, compile_mode",
    [
        (None, None),
        (Accelerator(), None),
        (None, "default"),
        (Accelerator(), "default"),
    ],
)
def test_matd3_save_load_checkpoint_correct_data_and_format(
    tmpdir, device, accelerator, compile_mode
):
    net_config = {"arch": "mlp", "hidden_size": [32, 32]}
    # Initialize the ddpg agent
    matd3 = MATD3(
        observation_spaces=generate_multi_agent_box_spaces(1, (6,)),
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        n_agents=1,
        agent_ids=["agent_0"],
        torch_compiler=compile_mode,
        device=device,
        accelerator=accelerator,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    matd3.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actors_init_dict" in checkpoint
    assert "actors_state_dict" in checkpoint
    assert "actor_targets_init_dict" in checkpoint
    assert "actor_targets_state_dict" in checkpoint
    assert "actor_optimizers_state_dict" in checkpoint
    assert "critics_1_init_dict" in checkpoint
    assert "critics_1_state_dict" in checkpoint
    assert "critic_targets_1_init_dict" in checkpoint
    assert "critic_targets_1_state_dict" in checkpoint
    assert "critic_2_optimizers_state_dict" in checkpoint
    assert "critics_2_init_dict" in checkpoint
    assert "critics_2_state_dict" in checkpoint
    assert "critic_targets_2_init_dict" in checkpoint
    assert "critic_targets_2_state_dict" in checkpoint
    assert "critic_2_optimizers_state_dict" in checkpoint
    assert "policy_freq" in checkpoint
    assert "net_config" in checkpoint
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

    # Load checkpoint
    loaded_matd3 = MATD3(
        observation_spaces=generate_multi_agent_box_spaces(1, (6,)),
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        n_agents=1,
        agent_ids=["agent_0"],
        torch_compiler=compile_mode,
        device=device,
        accelerator=accelerator,
    )
    loaded_matd3.load_checkpoint(checkpoint_path)
    # Check if properties and weights are loaded correctly
    assert loaded_matd3.net_config == net_config
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in loaded_matd3.actors)
        assert all(
            isinstance(actor_target, OptimizedModule)
            for actor_target in loaded_matd3.actor_targets
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in loaded_matd3.critics_1
        )
        assert all(
            isinstance(critic_target, OptimizedModule)
            for critic_target in loaded_matd3.critic_targets_1
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in loaded_matd3.critics_2
        )
        assert all(
            isinstance(critic_target, OptimizedModule)
            for critic_target in loaded_matd3.critic_targets_2
        )
    else:
        assert all(isinstance(actor, EvolvableMLP) for actor in loaded_matd3.actors)
        assert all(
            isinstance(actor_target, EvolvableMLP)
            for actor_target in loaded_matd3.actor_targets
        )
        assert all(
            isinstance(critic, EvolvableMLP) for critic in loaded_matd3.critics_1
        )
        assert all(
            isinstance(critic_target, EvolvableMLP)
            for critic_target in loaded_matd3.critic_targets_1
        )
        assert all(
            isinstance(critic, EvolvableMLP) for critic in loaded_matd3.critics_2
        )
        assert all(
            isinstance(critic_target, EvolvableMLP)
            for critic_target in loaded_matd3.critic_targets_2
        )
    assert matd3.lr_actor == 0.001
    assert matd3.lr_critic == 0.01

    for actor, actor_target in zip(loaded_matd3.actors, loaded_matd3.actor_targets):
        assert str(actor.state_dict()) == str(actor_target.state_dict())

    for critic_1, critic_target_1 in zip(
        loaded_matd3.critics_1, loaded_matd3.critic_targets_1
    ):
        assert str(critic_1.state_dict()) == str(critic_target_1.state_dict())

    for critic_2, critic_target_2 in zip(
        loaded_matd3.critics_2, loaded_matd3.critic_targets_2
    ):
        assert str(critic_2.state_dict()) == str(critic_target_2.state_dict())

    assert matd3.batch_size == 64
    assert matd3.learn_step == 5
    assert matd3.gamma == 0.95
    assert matd3.tau == 0.01
    assert matd3.mut is None
    assert matd3.index == 0
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]
    assert matd3.policy_freq == 2


@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize(
    "accelerator, compile_mode",
    [
        (None, None),
        (Accelerator(), None),
        (None, "default"),
        (Accelerator(), "default"),
    ],
)
def test_matd3_save_load_checkpoint_correct_data_and_format_cnn(
    tmpdir, device, accelerator, compile_mode
):
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [16],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }
    policy_freq = 2
    # Initialize the ddpg agent
    matd3 = MATD3(
        observation_spaces=generate_multi_agent_box_spaces(1, (3, 32, 32), low=0, high=255),
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        n_agents=1,
        agent_ids=["agent_0"],
        net_config=net_config_cnn,
        policy_freq=policy_freq,
        torch_compiler=compile_mode,
        device=device,
        accelerator=accelerator,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    matd3.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actors_init_dict" in checkpoint
    assert "actors_state_dict" in checkpoint
    assert "actor_targets_init_dict" in checkpoint
    assert "actor_targets_state_dict" in checkpoint
    assert "actor_optimizers_state_dict" in checkpoint
    assert "critics_1_init_dict" in checkpoint
    assert "critics_1_state_dict" in checkpoint
    assert "critic_targets_1_init_dict" in checkpoint
    assert "critic_targets_1_state_dict" in checkpoint
    assert "critic_1_optimizers_state_dict" in checkpoint
    assert "critics_2_init_dict" in checkpoint
    assert "critics_2_state_dict" in checkpoint
    assert "critic_targets_2_init_dict" in checkpoint
    assert "critic_targets_2_state_dict" in checkpoint
    assert "critic_2_optimizers_state_dict" in checkpoint
    assert "net_config" in checkpoint
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
    assert "policy_freq" in checkpoint

    # Load checkpoint
    loaded_matd3 = MATD3(
        observation_spaces=generate_multi_agent_box_spaces(1, (3, 32, 32), low=0, high=255),
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        n_agents=1,
        agent_ids=["agent_0"],
        torch_compiler=compile_mode,
        device=device,
        accelerator=accelerator,
    )
    loaded_matd3.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert loaded_matd3.net_config == net_config_cnn
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in loaded_matd3.actors)
        assert all(
            isinstance(actor_target, OptimizedModule)
            for actor_target in loaded_matd3.actor_targets
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in loaded_matd3.critics_1
        )
        assert all(
            isinstance(critic_target, OptimizedModule)
            for critic_target in loaded_matd3.critic_targets_1
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in loaded_matd3.critics_2
        )
        assert all(
            isinstance(critic_target, OptimizedModule)
            for critic_target in loaded_matd3.critic_targets_2
        )
    else:
        assert all(isinstance(actor, EvolvableCNN) for actor in loaded_matd3.actors)
        assert all(
            isinstance(actor_target, EvolvableCNN)
            for actor_target in loaded_matd3.actor_targets
        )
        assert all(
            isinstance(critic, EvolvableCNN) for critic in loaded_matd3.critics_1
        )
        assert all(
            isinstance(critic_target, EvolvableCNN)
            for critic_target in loaded_matd3.critic_targets_1
        )
        assert all(
            isinstance(critic, EvolvableCNN) for critic in loaded_matd3.critics_2
        )
        assert all(
            isinstance(critic_target, EvolvableCNN)
            for critic_target in loaded_matd3.critic_targets_2
        )
    assert matd3.lr_actor == 0.001
    assert matd3.lr_critic == 0.01

    for actor, actor_target in zip(loaded_matd3.actors, loaded_matd3.actor_targets):
        assert str(actor.state_dict()) == str(actor_target.state_dict())

    for critic_1, critic_target_1 in zip(
        loaded_matd3.critics_1, loaded_matd3.critic_targets_1
    ):
        assert str(critic_1.state_dict()) == str(critic_target_1.state_dict())

    for critic_2, critic_target_2 in zip(
        loaded_matd3.critics_2, loaded_matd3.critic_targets_2
    ):
        assert str(critic_2.state_dict()) == str(critic_target_2.state_dict())

    assert matd3.batch_size == 64
    assert matd3.learn_step == 5
    assert matd3.gamma == 0.95
    assert matd3.tau == 0.01
    assert matd3.mut is None
    assert matd3.index == 0
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]
    assert matd3.policy_freq == policy_freq


@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize(
    "accelerator, compile_mode",
    [
        (None, None),
        (Accelerator(), None),
        (None, "default"),
        (Accelerator(), "default"),
    ],
)
@pytest.mark.parametrize(
    "observation_spaces, action_spaces",
    [
        (
            generate_multi_agent_box_spaces(1, (6,)),
            generate_multi_agent_discrete_spaces(1, 2),
        )
    ],
)
def test_matd3_save_load_checkpoint_correct_data_and_format_make_evo(
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
    evo_critics_1 = [
        MakeEvolvable(network=mlp_critic, input_tensor=torch.randn(1, 8), device=device)
        for _ in range(1)
    ]
    evo_critics_2 = [
        MakeEvolvable(network=mlp_critic, input_tensor=torch.randn(1, 8), device=device)
        for _ in range(1)
    ]
    evo_critics = [evo_critics_1, evo_critics_2]
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        n_agents=1,
        agent_ids=["agent_0"],
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
    )
    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    matd3.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actors_init_dict" in checkpoint
    assert "actors_state_dict" in checkpoint
    assert "actor_targets_init_dict" in checkpoint
    assert "actor_targets_state_dict" in checkpoint
    assert "actor_optimizers_state_dict" in checkpoint
    assert "critics_1_init_dict" in checkpoint
    assert "critics_1_state_dict" in checkpoint
    assert "critic_targets_1_init_dict" in checkpoint
    assert "critic_targets_1_state_dict" in checkpoint
    assert "critic_1_optimizers_state_dict" in checkpoint
    assert "critics_2_init_dict" in checkpoint
    assert "critics_2_state_dict" in checkpoint
    assert "critic_targets_2_init_dict" in checkpoint
    assert "critic_targets_2_state_dict" in checkpoint
    assert "critic_2_optimizers_state_dict" in checkpoint
    assert "net_config" in checkpoint
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
    assert "policy_freq" in checkpoint

    # Load checkpoint
    loaded_matd3 = MATD3(
        observation_spaces=generate_multi_agent_box_spaces(1, (3, 32, 32), low=0, high=255),
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        n_agents=1,
        agent_ids=["agent_0"],
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
    )
    loaded_matd3.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    if compile_mode is not None and accelerator is None:
        assert all(isinstance(actor, OptimizedModule) for actor in loaded_matd3.actors)
        assert all(
            isinstance(actor_target, OptimizedModule)
            for actor_target in loaded_matd3.actor_targets
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in loaded_matd3.critics_1
        )
        assert all(
            isinstance(critic_target, OptimizedModule)
            for critic_target in loaded_matd3.critic_targets_1
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in loaded_matd3.critics_2
        )
        assert all(
            isinstance(critic_target, OptimizedModule)
            for critic_target in loaded_matd3.critic_targets_2
        )
    else:
        assert all(isinstance(actor, MakeEvolvable) for actor in loaded_matd3.actors)
        assert all(
            isinstance(actor_target, MakeEvolvable)
            for actor_target in loaded_matd3.actor_targets
        )
        assert all(
            isinstance(critic, MakeEvolvable) for critic in loaded_matd3.critics_1
        )
        assert all(
            isinstance(critic_target, MakeEvolvable)
            for critic_target in loaded_matd3.critic_targets_1
        )
        assert all(
            isinstance(critic, MakeEvolvable) for critic in loaded_matd3.critics_2
        )
        assert all(
            isinstance(critic_target, MakeEvolvable)
            for critic_target in loaded_matd3.critic_targets_2
        )
    assert matd3.lr_actor == 0.001
    assert matd3.lr_critic == 0.01

    for actor, actor_target in zip(loaded_matd3.actors, loaded_matd3.actor_targets):
        assert str(actor.state_dict()) == str(actor_target.state_dict())

    for critic_1, critic_target_1 in zip(
        loaded_matd3.critics_1, loaded_matd3.critic_targets_1
    ):
        assert str(critic_1.state_dict()) == str(critic_target_1.state_dict())

    for critic_2, critic_target_2 in zip(
        loaded_matd3.critics_2, loaded_matd3.critic_targets_2
    ):
        assert str(critic_2.state_dict()) == str(critic_target_2.state_dict())

    assert matd3.batch_size == 64
    assert matd3.learn_step == 5
    assert matd3.gamma == 0.95
    assert matd3.tau == 0.01
    assert matd3.mut is None
    assert matd3.index == 0
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]
    assert matd3.policy_freq == 2


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_unwrap_models(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(2, (6,))
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    accelerator = Accelerator()
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    matd3.unwrap_models()
    for (
        actor,
        critic_1,
        critic_2,
        actor_target,
        critic_target_1,
        critic_target_2,
    ) in zip(
        matd3.actors,
        matd3.critics_1,
        matd3.critics_2,
        matd3.actor_targets,
        matd3.critic_targets_1,
        matd3.critic_targets_2,
    ):
        assert isinstance(actor, nn.Module)
        assert isinstance(actor_target, nn.Module)
        assert isinstance(critic_1, nn.Module)
        assert isinstance(critic_target_1, nn.Module)
        assert isinstance(critic_2, nn.Module)
        assert isinstance(critic_target_2, nn.Module)


# Returns the input action scaled to the action space defined by self.min_action and self.max_action.
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_action_scaling(compile_mode):
    action = torch.Tensor([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    lows = [-1, -2, 0, 0, -1]
    highs = [1, 2, 1, 2, 2]

    matd3 = MATD3(
        observation_spaces=generate_multi_agent_box_spaces(5, (4,)),
        action_spaces=generate_multi_agent_box_spaces(5, (1,), low=lows, high=highs),
        n_agents=5,
        agent_ids=["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"],
        torch_compiler=compile_mode,
    )
    matd3.actors[0].mlp_output_activation = "Tanh"
    scaled_action = matd3.scale_to_action_space(action, idx=0)
    assert torch.equal(scaled_action, torch.Tensor([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))

    matd3.actors[1].mlp_output_activation = "Tanh"
    action = torch.Tensor([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    scaled_action = matd3.scale_to_action_space(action, idx=1)
    torch.testing.assert_close(
        scaled_action, torch.Tensor([0.2, 0.4, 0.6, -0.2, -0.4, -0.6])
    )

    matd3.actors[2].mlp_output_activation = "Sigmoid"
    action = torch.Tensor([0.1, 0.2, 0.3, 0])
    scaled_action = matd3.scale_to_action_space(action, idx=2)
    assert torch.equal(scaled_action, torch.Tensor([0.1, 0.2, 0.3, 0]))

    matd3.actors[3].mlp_output_activation = "GumbelSoftmax"
    action = torch.Tensor([0.1, 0.2, 0.3, 0])
    scaled_action = matd3.scale_to_action_space(action, idx=3)
    assert torch.equal(scaled_action, torch.Tensor([0.2, 0.4, 0.6, 0]))

    matd3.actors[4].mlp_output_activation = "Tanh"
    action = torch.Tensor([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    scaled_action = matd3.scale_to_action_space(action, idx=4)
    torch.testing.assert_close(
        scaled_action,
        torch.Tensor(
            [
                0.55 * 3 - 1,
                0.6 * 3 - 1,
                0.65 * 3 - 1,
                0.45 * 3 - 1,
                0.4 * 3 - 1,
                0.35 * 3 - 1,
            ]
        ),
    )


@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize(
    "accelerator, compile_mode",
    [
        (None, None),
        (Accelerator(), None),
        (None, "default"),
        (Accelerator(), "default"),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained(device, accelerator, compile_mode, tmpdir):
    # Initialize the matd3 agent
    matd3 = MATD3(
        observation_spaces=generate_multi_agent_box_spaces(2, (4,)),
        action_spaces=generate_multi_agent_discrete_spaces(2, 2),
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        torch_compiler=compile_mode,
        accelerator=accelerator,
        device=device,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    matd3.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_matd3 = MATD3.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_matd3.observation_spaces == matd3.observation_spaces
    assert new_matd3.action_spaces == matd3.action_spaces
    assert new_matd3.one_hot == matd3.one_hot
    assert new_matd3.n_agents == matd3.n_agents
    assert new_matd3.agent_ids == matd3.agent_ids
    assert np.all(np.stack(new_matd3.max_action) == np.stack(matd3.max_action))
    assert np.all(np.stack(new_matd3.min_action) == np.stack(matd3.min_action))
    assert new_matd3.net_config == matd3.net_config
    assert new_matd3.lr_actor == matd3.lr_actor
    assert new_matd3.lr_critic == matd3.lr_critic
    for (
        new_actor,
        new_actor_target,
        new_critic_1,
        new_critic_target_1,
        new_critic_2,
        new_critic_target_2,
        actor,
        actor_target,
        critic_1,
        critic_target_1,
        critic_2,
        critic_target_2,
    ) in zip(
        new_matd3.actors,
        new_matd3.actor_targets,
        new_matd3.critics_1,
        new_matd3.critic_targets_1,
        new_matd3.critics_2,
        new_matd3.critic_targets_2,
        matd3.actors,
        matd3.actor_targets,
        matd3.critics_1,
        matd3.critic_targets_1,
        matd3.critics_2,
        matd3.critic_targets_2,
    ):
        if compile_mode is not None and accelerator is None:
            assert isinstance(new_actor, OptimizedModule)
            assert isinstance(new_actor_target, OptimizedModule)
            assert isinstance(new_critic_1, OptimizedModule)
            assert isinstance(new_critic_target_1, OptimizedModule)
            assert isinstance(new_critic_2, OptimizedModule)
            assert isinstance(new_critic_target_2, OptimizedModule)
        else:
            assert isinstance(new_actor, EvolvableMLP)
            assert isinstance(new_actor_target, EvolvableMLP)
            assert isinstance(new_critic_1, EvolvableMLP)
            assert isinstance(new_critic_target_1, EvolvableMLP)
            assert isinstance(new_critic_2, EvolvableMLP)
            assert isinstance(new_critic_target_2, EvolvableMLP)

        new_actor_sd = str(new_actor.state_dict())
        new_actor_target_sd = str(new_actor_target.state_dict())
        new_critic_1_sd = str(new_critic_1.state_dict())
        new_critic_target_1_sd = str(new_critic_target_1.state_dict())
        new_critic_2_sd = str(new_critic_2.state_dict())
        new_critic_target_2_sd = str(new_critic_target_2.state_dict())

        assert new_actor_sd == str(actor.state_dict())
        assert new_actor_target_sd == str(actor_target.state_dict())
        assert new_critic_1_sd == str(critic_1.state_dict())
        assert new_critic_target_1_sd == str(critic_target_1.state_dict())
        assert new_critic_2_sd == str(critic_2.state_dict())
        assert new_critic_target_2_sd == str(critic_target_2.state_dict())

    assert new_matd3.batch_size == matd3.batch_size
    assert new_matd3.learn_step == matd3.learn_step
    assert new_matd3.gamma == matd3.gamma
    assert new_matd3.tau == matd3.tau
    assert new_matd3.mut == matd3.mut
    assert new_matd3.index == matd3.index
    assert new_matd3.scores == matd3.scores
    assert new_matd3.fitness == matd3.fitness
    assert new_matd3.steps == matd3.steps


@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize(
    "accelerator, compile_mode",
    [
        (None, None),
        (Accelerator(), None),
        (None, "default"),
        (Accelerator(), "default"),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_cnn(device, accelerator, compile_mode, tmpdir):
    # Initialize the matd3 agent
    matd3 = MATD3(
        observation_spaces=generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
        action_spaces=generate_multi_agent_box_spaces(2, (1,)),
        n_agents=2,
        agent_ids=["agent_a", "agent_b"],
        net_config={
            "arch": "cnn",
            "hidden_size": [8],
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
            "normalize": False,
        },
        torch_compiler=compile_mode,
        accelerator=accelerator,
        device=device,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    matd3.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_matd3 = MATD3.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_matd3.observation_spaces == matd3.observation_spaces
    assert new_matd3.action_spaces == matd3.action_spaces
    assert new_matd3.one_hot == matd3.one_hot
    assert new_matd3.n_agents == matd3.n_agents
    assert new_matd3.agent_ids == matd3.agent_ids
    assert np.all(np.stack(new_matd3.max_action) == np.stack(matd3.max_action))
    assert np.all(np.stack(new_matd3.min_action) == np.stack(matd3.min_action))
    assert new_matd3.net_config == matd3.net_config
    assert new_matd3.lr_actor == matd3.lr_actor
    assert new_matd3.lr_critic == matd3.lr_critic
    for (
        new_actor,
        new_actor_target,
        new_critic_1,
        new_critic_target_1,
        new_critic_2,
        new_critic_target_2,
        actor,
        actor_target,
        critic_1,
        critic_target_1,
        critic_2,
        critic_target_2,
    ) in zip(
        new_matd3.actors,
        new_matd3.actor_targets,
        new_matd3.critics_1,
        new_matd3.critic_targets_1,
        new_matd3.critics_2,
        new_matd3.critic_targets_2,
        matd3.actors,
        matd3.actor_targets,
        matd3.critics_1,
        matd3.critic_targets_1,
        matd3.critics_2,
        matd3.critic_targets_2,
    ):
        if compile_mode is not None and accelerator is None:
            assert isinstance(new_actor, OptimizedModule)
            assert isinstance(new_actor_target, OptimizedModule)
            assert isinstance(new_critic_1, OptimizedModule)
            assert isinstance(new_critic_target_1, OptimizedModule)
            assert isinstance(new_critic_2, OptimizedModule)
            assert isinstance(new_critic_target_2, OptimizedModule)
        else:
            assert isinstance(new_actor, EvolvableCNN)
            assert isinstance(new_actor_target, EvolvableCNN)
            assert isinstance(new_critic_1, EvolvableCNN)
            assert isinstance(new_critic_target_1, EvolvableCNN)
            assert isinstance(new_critic_2, EvolvableCNN)
            assert isinstance(new_critic_target_2, EvolvableCNN)

        new_actor_sd = str(new_actor.state_dict())
        new_actor_target_sd = str(new_actor_target.state_dict())
        new_critic_1_sd = str(new_critic_1.state_dict())
        new_critic_target_1_sd = str(new_critic_target_1.state_dict())
        new_critic_2_sd = str(new_critic_2.state_dict())
        new_critic_target_2_sd = str(new_critic_target_2.state_dict())

        assert new_actor_sd == str(actor.state_dict())
        assert new_actor_target_sd == str(actor_target.state_dict())
        assert new_critic_1_sd == str(critic_1.state_dict())
        assert new_critic_target_1_sd == str(critic_target_1.state_dict())
        assert new_critic_2_sd == str(critic_2.state_dict())
        assert new_critic_target_2_sd == str(critic_target_2.state_dict())

    assert new_matd3.batch_size == matd3.batch_size
    assert new_matd3.learn_step == matd3.learn_step
    assert new_matd3.gamma == matd3.gamma
    assert new_matd3.tau == matd3.tau
    assert new_matd3.mut == matd3.mut
    assert new_matd3.index == matd3.index
    assert new_matd3.scores == matd3.scores
    assert new_matd3.fitness == matd3.fitness
    assert new_matd3.steps == matd3.steps


@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize(
    "observation_spaces, action_spaces, arch, input_tensor, critic_input_tensor, secondary_input_tensor, compile_mode",
    [
        (generate_multi_agent_box_spaces(2, (4,)), generate_multi_agent_discrete_spaces(2, 2), "mlp", torch.randn(1, 4), torch.randn(1, 6), None, None),
        (
            generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255),
            generate_multi_agent_discrete_spaces(2, 2),
            "cnn",
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 2),
            None,
        ),
        (
            generate_multi_agent_box_spaces(2, (4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            "mlp",
            torch.randn(1, 4),
            torch.randn(1, 6),
            None,
            "default",
        ),
        (
            generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255),
            generate_multi_agent_discrete_spaces(2, 2),
            "cnn",
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 2),
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
    secondary_input_tensor,
    tmpdir,
    compile_mode,
    device,
):
    one_hot = False
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
        secondary_input_tensor=secondary_input_tensor,
    )

    # Initialize the matd3 agent
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        one_hot=one_hot,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        actor_networks=[actor_network, copy.deepcopy(actor_network)],
        critic_networks=[
            [critic_network, copy.deepcopy(critic_network)],
            [copy.deepcopy(critic_network), copy.deepcopy(critic_network)],
        ],
        torch_compiler=compile_mode,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    matd3.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_matd3 = MATD3.load(checkpoint_path, device=device)

    # Check if properties and weights are loaded correctly
    assert new_matd3.observation_spaces == matd3.observation_spaces
    assert new_matd3.action_spaces == matd3.action_spaces
    assert new_matd3.one_hot == matd3.one_hot
    assert new_matd3.n_agents == matd3.n_agents
    assert new_matd3.agent_ids == matd3.agent_ids
    assert np.all(np.stack(new_matd3.max_action) == np.stack(matd3.max_action))
    assert np.all(np.stack(new_matd3.min_action) == np.stack(matd3.min_action))
    assert new_matd3.net_config == matd3.net_config
    assert new_matd3.lr_actor == matd3.lr_actor
    assert new_matd3.lr_critic == matd3.lr_critic
    for (
        new_actor,
        new_actor_target,
        new_critic_1,
        new_critic_target_1,
        new_critic_2,
        new_critic_target_2,
        actor,
        actor_target,
        critic_1,
        critic_target_1,
        critic_2,
        critic_target_2,
    ) in zip(
        new_matd3.actors,
        new_matd3.actor_targets,
        new_matd3.critics_1,
        new_matd3.critic_targets_1,
        new_matd3.critics_2,
        new_matd3.critic_targets_2,
        matd3.actors,
        matd3.actor_targets,
        matd3.critics_1,
        matd3.critic_targets_1,
        matd3.critics_2,
        matd3.critic_targets_2,
    ):
        assert isinstance(new_actor, nn.Module)
        assert isinstance(new_actor_target, nn.Module)
        assert isinstance(new_critic_1, nn.Module)
        assert isinstance(new_critic_target_1, nn.Module)
        assert isinstance(new_critic_2, nn.Module)
        assert isinstance(new_critic_target_2, nn.Module)
        assert str(new_actor.to("cpu").state_dict()) == str(actor.state_dict())
        assert str(new_actor_target.to("cpu").state_dict()) == str(
            actor_target.state_dict()
        )
        assert str(new_critic_1.to("cpu").state_dict()) == str(critic_1.state_dict())
        assert str(new_critic_target_1.to("cpu").state_dict()) == str(
            critic_target_1.state_dict()
        )
        assert str(new_critic_2.to("cpu").state_dict()) == str(critic_2.state_dict())
        assert str(new_critic_target_2.to("cpu").state_dict()) == str(
            critic_target_2.state_dict()
        )
    assert new_matd3.batch_size == matd3.batch_size
    assert new_matd3.learn_step == matd3.learn_step
    assert new_matd3.gamma == matd3.gamma
    assert new_matd3.tau == matd3.tau
    assert new_matd3.mut == matd3.mut
    assert new_matd3.index == matd3.index
    assert new_matd3.scores == matd3.scores
    assert new_matd3.fitness == matd3.fitness
    assert new_matd3.steps == matd3.steps
