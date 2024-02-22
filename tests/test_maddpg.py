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

from agilerl.algorithms.maddpg import MADDPG
from agilerl.networks.custom_components import GumbelSoftmax
from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.utils.utils import makeMultiAgentVectEnvs
from agilerl.wrappers.make_evolvable import MakeEvolvable


class DummyMultiEnv:
    def __init__(self, state_dims, action_dims):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = ["agent_0", "agent_1"]
        self.metadata = None
        self.observation_space = None
        self.action_space = None

    def reset(self, seed=None, options=None):
        return {agent: np.random.rand(*self.state_dims) for agent in self.agents}, {
            "info_string": None,
            "agent_mask": {"agent_0": False, "agent_1": True},
            "env_defined_actions": {"agent_0": np.array([0, 1]), "agent_1": None},
        }

    def step(self, action):
        return (
            {agent: np.random.rand(*self.state_dims) for agent in self.agents},
            {agent: np.random.randint(0, 5) for agent in self.agents},
            {agent: np.random.randint(0, 2) for agent in self.agents},
            {agent: np.random.randint(0, 2) for agent in self.agents},
            {agent: "info_string" for agent in self.agents},
        )


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
def mlp_actor(state_dims, action_dims):
    net = nn.Sequential(
        nn.Linear(state_dims[0][0], 64),
        nn.ReLU(),
        nn.Linear(64, action_dims[0]),
        GumbelSoftmax(),
    )
    return net


@pytest.fixture
def mlp_critic(action_dims, state_dims):
    net = nn.Sequential(
        nn.Linear(state_dims[0][0] + action_dims[0], 64), nn.ReLU(), nn.Linear(64, 1)
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
def accelerated_experiences(batch_size, state_dims, action_dims, agent_ids, one_hot):
    state_size = state_dims[0]
    action_size = action_dims[0]
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
def experiences(batch_size, state_dims, action_dims, agent_ids, one_hot, device):
    state_size = state_dims[0]
    action_size = action_dims[0]
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
    "net_config, accelerator_flag, state_dims",
    [
        ({"arch": "mlp", "h_size": [64, 64]}, False, [(4,), (4,)]),
        (
            {
                "arch": "cnn",
                "h_size": [8],
                "c_size": [3],
                "k_size": [3],
                "s_size": [1],
                "normalize": False,
            },
            False,
            [(3, 32, 32), (3, 32, 32)],
        ),
        (
            {
                "arch": "cnn",
                "h_size": [8],
                "c_size": [3],
                "k_size": [3],
                "s_size": [1],
                "normalize": False,
            },
            True,
            [(3, 32, 32), (3, 32, 32)],
        ),
    ],
)
def test_initialize_maddpg_with_net_config(
    net_config, accelerator_flag, state_dims, device
):
    action_dims = [2, 2]
    one_hot = False
    n_agents = 2
    agent_ids = ["agent_0", "agent_1"]
    max_action = [(1,), (1,)]
    min_action = [(-1,), (-1,)]
    discrete_actions = False
    expl_noise = 0.1
    batch_size = 64
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    maddpg = MADDPG(
        state_dims=state_dims,
        net_config=net_config,
        action_dims=action_dims,
        one_hot=one_hot,
        n_agents=n_agents,
        agent_ids=agent_ids,
        max_action=max_action,
        min_action=min_action,
        discrete_actions=discrete_actions,
        accelerator=accelerator,
        device=device,
    )
    net_config.update({"output_activation": "Softmax"})
    assert maddpg.state_dims == state_dims
    assert maddpg.action_dims == action_dims
    assert maddpg.one_hot == one_hot
    assert maddpg.n_agents == n_agents
    assert maddpg.agent_ids == agent_ids
    assert maddpg.max_action == max_action
    assert maddpg.min_action == min_action
    assert maddpg.discrete_actions == discrete_actions
    assert maddpg.expl_noise == expl_noise
    assert maddpg.net_config == net_config, maddpg.net_config
    assert maddpg.batch_size == batch_size
    assert maddpg.multi
    assert maddpg.total_state_dims == sum(state[0] for state in state_dims)
    assert maddpg.total_actions == sum(action_dims)
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]
    assert maddpg.actor_networks is None
    assert maddpg.critic_networks is None
    if net_config["arch"] == "mlp":
        evo_type = EvolvableMLP
        assert maddpg.arch == "mlp"
    else:
        evo_type = EvolvableCNN
        assert maddpg.arch == "cnn"
    assert all(isinstance(actor, evo_type) for actor in maddpg.actors)
    assert all(isinstance(critic, evo_type) for critic in maddpg.critics)
    assert all(
        isinstance(actor_target, evo_type) for actor_target in maddpg.actor_targets
    )
    assert all(
        isinstance(critic_target, evo_type) for critic_target in maddpg.critic_targets
    )
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in maddpg.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in maddpg.critic_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in maddpg.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in maddpg.critic_optimizers
        )
    assert isinstance(maddpg.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "state_dims, action_dims, accelerator_flag",
    [
        ([(6,) for _ in range(2)], [2 for _ in range(2)], False),
        ([(6,) for _ in range(2)], [2 for _ in range(2)], True),
    ],
)
def test_initialize_maddpg_with_mlp_networks(
    mlp_actor, mlp_critic, state_dims, action_dims, accelerator_flag, device
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    evo_actors = [
        MakeEvolvable(network=mlp_actor, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(2)
    ]
    evo_critics = [
        MakeEvolvable(network=mlp_critic, input_tensor=torch.randn(1, 8), device=device)
        for _ in range(2)
    ]
    maddpg = MADDPG(
        state_dims=state_dims,
        action_dims=action_dims,
        one_hot=False,
        agent_ids=["agent_0", "agent_1"],
        n_agents=len(state_dims),
        max_action=[(1,), (1,)],
        min_action=[(-1,), (-1,)],
        discrete_actions=True,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
    )
    assert all(isinstance(actor, MakeEvolvable) for actor in maddpg.actors)
    assert all(isinstance(critic, MakeEvolvable) for critic in maddpg.critics)
    assert maddpg.net_config is None
    assert maddpg.arch == "mlp"
    assert maddpg.state_dims == state_dims
    assert maddpg.action_dims == action_dims
    assert maddpg.one_hot is False
    assert maddpg.n_agents == 2
    assert maddpg.agent_ids == ["agent_0", "agent_1"]
    assert maddpg.max_action == [(1,), (1,)]
    assert maddpg.min_action == [(-1,), (-1,)]
    assert maddpg.discrete_actions is True
    assert maddpg.multi
    assert maddpg.total_state_dims == sum(state[0] for state in state_dims)
    assert maddpg.total_actions == sum(action_dims)
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in maddpg.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in maddpg.critic_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in maddpg.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in maddpg.critic_optimizers
        )
    assert isinstance(maddpg.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "state_dims, action_dims, accelerator_flag",
    [
        ([(4, 210, 160) for _ in range(2)], [2 for _ in range(2)], False),
        ([(4, 210, 160) for _ in range(2)], [2 for _ in range(2)], True),
    ],
)
def test_initialize_maddpg_with_cnn_networks(
    cnn_actor, cnn_critic, state_dims, action_dims, accelerator_flag, device
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
    evo_critics = [
        MakeEvolvable(
            network=cnn_critic,
            input_tensor=torch.randn(1, 4, 2, 210, 160),
            secondary_input_tensor=torch.randn(1, 2),
            extra_critic_dims=2,
            device=device,
        )
        for _ in range(2)
    ]
    maddpg = MADDPG(
        state_dims=state_dims,
        action_dims=action_dims,
        one_hot=False,
        agent_ids=["agent_0", "agent_1"],
        n_agents=len(state_dims),
        max_action=[(1,), (1,)],
        min_action=[(-1,), (-1,)],
        discrete_actions=True,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
    )
    assert all(isinstance(actor, MakeEvolvable) for actor in maddpg.actors)
    assert all(isinstance(critic, MakeEvolvable) for critic in maddpg.critics)
    assert maddpg.net_config is None
    assert maddpg.arch == "cnn"
    assert maddpg.state_dims == state_dims
    assert maddpg.action_dims == action_dims
    assert maddpg.one_hot is False
    assert maddpg.n_agents == 2
    assert maddpg.agent_ids == ["agent_0", "agent_1"]
    assert maddpg.max_action == [(1,), (1,)]
    assert maddpg.min_action == [(-1,), (-1,)]
    assert maddpg.discrete_actions is True
    assert maddpg.multi
    assert maddpg.total_state_dims == sum(state[0] for state in state_dims)
    assert maddpg.total_actions == sum(action_dims)
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]
    if accelerator is None:
        assert all(
            isinstance(actor_optimizer, optim.Adam)
            for actor_optimizer in maddpg.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, optim.Adam)
            for critic_optimizer in maddpg.critic_optimizers
        )
    else:
        assert all(
            isinstance(actor_optimizer, AcceleratedOptimizer)
            for actor_optimizer in maddpg.actor_optimizers
        )
        assert all(
            isinstance(critic_optimizer, AcceleratedOptimizer)
            for critic_optimizer in maddpg.critic_optimizers
        )
    assert isinstance(maddpg.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "state_dims, action_dims",
    [
        ([(6,) for _ in range(2)], [2 for _ in range(2)]),
    ],
)
def test_maddpg_init_warning(mlp_actor, state_dims, action_dims, device):
    warning_string = "Actor and critic network lists must both be supplied to use custom networks. Defaulting to net config."
    evo_actors = [
        MakeEvolvable(network=mlp_actor, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(2)
    ]
    with pytest.warns(UserWarning, match=warning_string):
        MADDPG(
            state_dims=state_dims,
            action_dims=action_dims,
            one_hot=False,
            agent_ids=["agent_0", "agent_1"],
            n_agents=len(state_dims),
            max_action=[(1,), (1,)],
            min_action=[(-1,), (-1,)],
            discrete_actions=True,
            actor_networks=evo_actors,
            device=device,
        )


@pytest.mark.parametrize(
    "epsilon, state_dims, action_dims, discrete_actions, one_hot",
    [
        (1, [(6,) for _ in range(2)], [2 for _ in range(2)], False, False),
        (0, [(6,) for _ in range(2)], [2 for _ in range(2)], False, False),
        (1, [(6,) for _ in range(2)], [2 for _ in range(2)], True, False),  #
        (0, [(6,) for _ in range(2)], [2 for _ in range(2)], True, False),
        (1, [(6,) for _ in range(2)], [2 for _ in range(2)], False, True),
        (0, [(6,) for _ in range(2)], [2 for _ in range(2)], False, True),
        (1, [(6,) for _ in range(2)], [2 for _ in range(2)], True, True),  #
        (0, [(6,) for _ in range(2)], [2 for _ in range(2)], True, True),
    ],
)
def test_maddpg_getAction_epsilon_greedy_mlp(
    epsilon, state_dims, action_dims, discrete_actions, one_hot, device
):
    agent_ids = ["agent_0", "agent_1"]
    if one_hot:
        state = {
            agent: np.random.randint(0, state_dims[0], *state_dims[0])
            for agent in agent_ids
        }
    else:
        state = {agent: np.random.randn(*state_dims[0]) for agent in agent_ids}
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=one_hot,
        net_config={"arch": "mlp", "h_size": [64, 64]},
        n_agents=2,
        agent_ids=agent_ids,
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=discrete_actions,
        device=device,
    )
    cont_actions, discrete_action = maddpg.getAction(state, epsilon)
    for idx, action in enumerate(list(cont_actions.values())):
        if one_hot:
            assert len(action[0]) == action_dims[idx]
            assert len(action) == state_dims[idx][0]
            if discrete_actions:
                torch.testing.assert_close(
                    sum(np.sum(action, 0)), float(state_dims[idx][0])
                )
            act = action[idx]
            assert act.dtype == np.float32
            assert -1 <= act.all() <= 1
        else:
            assert len(action) == action_dims[idx]
            if discrete_actions:
                torch.testing.assert_close(sum(action), 1.0)
            act = action[idx]
            assert isinstance(act, np.float32)
            assert -1 <= act <= 1
    if discrete_actions:
        for idx, action in enumerate(list(discrete_action.values())):
            if one_hot:
                assert action.all() <= action_dims[idx] - 1
            else:
                assert action <= action_dims[idx] - 1
    maddpg = None


@pytest.mark.parametrize(
    "epsilon, state_dims, action_dims, discrete_actions",
    [
        (1, [(3, 32, 32) for _ in range(2)], [2 for _ in range(2)], False),
        (0, [(3, 32, 32) for _ in range(2)], [2 for _ in range(2)], False),
        (1, [(3, 32, 32) for _ in range(2)], [2 for _ in range(2)], True),  #
        (0, [(3, 32, 32) for _ in range(2)], [2 for _ in range(2)], True),
    ],
)
def test_maddpg_getAction_epsilon_greedy_cnn(
    epsilon, state_dims, action_dims, discrete_actions, device
):
    agent_ids = ["agent_0", "agent_1"]
    net_config = {
        "arch": "cnn",
        "h_size": [64, 64],
        "c_size": [16],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    state = {agent: np.random.randn(1, *state_dims[0]) for agent in agent_ids}
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=False,
        n_agents=2,
        agent_ids=agent_ids,
        net_config=net_config,
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=discrete_actions,
        device=device,
    )
    cont_actions, discrete_action = maddpg.getAction(state, epsilon)
    if discrete_actions:
        for idx, action in enumerate(list(cont_actions.values())):
            assert len(action) == action_dims[idx]
            torch.testing.assert_close(sum(action), 1.00)
            act = action[idx]
            assert isinstance(act, np.float32)
            assert -1 <= act <= 1
        for idx, action in enumerate(list(discrete_action.values())):
            assert action <= action_dims[idx] - 1
    else:
        for idx, action in enumerate(list(cont_actions.values())):
            assert len(action) == action_dims[idx]
            act = action[idx]
            assert isinstance(act, np.float32)
            assert -1 <= act <= 1


@pytest.mark.parametrize(
    "epsilon, state_dims, action_dims, discrete_actions",
    [
        (1, [(6,) for _ in range(2)], [2 for _ in range(2)], True),  #
        (0, [(6,) for _ in range(2)], [2 for _ in range(2)], True),
        (1, [(6,) for _ in range(2)], [2 for _ in range(2)], False),
        (0, [(6,) for _ in range(2)], [2 for _ in range(2)], False),
    ],
)
def test_getAction_epsilon_greedy_distributed(
    epsilon, state_dims, action_dims, discrete_actions
):
    accelerator = Accelerator()
    agent_ids = ["agent_0", "agent_1"]
    state = {agent: np.random.randn(*state_dims[0]) for agent in agent_ids}
    from agilerl.algorithms.maddpg import MADDPG

    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=False,
        n_agents=2,
        agent_ids=agent_ids,
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=discrete_actions,
        accelerator=accelerator,
    )
    new_actors = [
        DummyEvolvableMLP(
            num_inputs=actor.num_inputs,
            num_outputs=actor.num_outputs,
            hidden_size=actor.hidden_size,
            device=actor.device,
            mlp_output_activation=actor.mlp_output_activation,
        )
        for actor in maddpg.actors
    ]
    maddpg.actors = new_actors
    cont_actions, discrete_action = maddpg.getAction(state, epsilon)
    if discrete_actions:
        for idx, action in enumerate(list(cont_actions.values())):
            assert len(action) == action_dims[idx]
            torch.testing.assert_close(sum(action), 1.00)
            act = action[idx]
            assert isinstance(act, np.float32)
            assert -1 <= act <= 1
        for idx, action in enumerate(list(discrete_action.values())):
            assert action <= action_dims[idx] - 1
    else:
        for idx, action in enumerate(list(cont_actions.values())):
            assert len(action) == action_dims[idx]
            act = action[idx]
            assert isinstance(act, np.float32)
            assert -1 <= act <= 1


@pytest.mark.parametrize(
    "epsilon, state_dims, action_dims, discrete_actions",
    [
        (1, [(3, 32, 32) for _ in range(2)], [2 for _ in range(2)], True),
        (0, [(3, 32, 32) for _ in range(2)], [2 for _ in range(2)], True),
        (1, [(3, 32, 32) for _ in range(2)], [2 for _ in range(2)], False),
        (0, [(3, 32, 32) for _ in range(2)], [2 for _ in range(2)], False),
    ],
)
def test_maddpg_getAction_epsilon_greedy_distributed_cnn(
    epsilon, state_dims, action_dims, discrete_actions
):
    accelerator = Accelerator()
    agent_ids = ["agent_0", "agent_1"]
    net_config = {
        "arch": "cnn",
        "h_size": [64, 64],
        "c_size": [16],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    state = {agent: np.random.randn(1, *state_dims[0]) for agent in agent_ids}
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=False,
        n_agents=2,
        agent_ids=agent_ids,
        max_action=[[1], [1]],
        min_action=[[-1], [0]],
        net_config=net_config,
        discrete_actions=discrete_actions,
        accelerator=accelerator,
    )
    new_actors = [
        DummyEvolvableCNN(
            input_shape=actor.input_shape,
            num_actions=actor.num_actions,
            channel_size=net_config["c_size"],
            kernel_size=net_config["k_size"],
            stride_size=net_config["s_size"],
            hidden_size=net_config["h_size"],
            normalize=net_config["normalize"],
            mlp_output_activation=net_config["output_activation"],
            multi=actor.multi,
            n_agents=actor.n_agents,
            accelerator=accelerator,
        )
        for actor in maddpg.actors
    ]
    maddpg.actors = new_actors
    cont_actions, discrete_action = maddpg.getAction(state, epsilon)
    if discrete_actions:
        for idx, action in enumerate(list(cont_actions.values())):
            assert len(action) == action_dims[idx]
            torch.testing.assert_close(sum(action), 1.00)
            act = action[idx]
            assert isinstance(act, np.float32)
            assert -1 <= act <= 1
        for idx, action in enumerate(list(discrete_action.values())):
            assert action <= action_dims[idx] - 1
    else:
        for idx, action in enumerate(list(cont_actions.values())):
            assert len(action) == action_dims[idx]
            act = action[idx]
            assert isinstance(act, np.float32)
            assert -1 <= act <= 1


@pytest.mark.parametrize(
    "epsilon, state_dims, action_dims, discrete_actions",
    [
        (1, [(6,) for _ in range(2)], [2 for _ in range(2)], False),
        (0, [(6,) for _ in range(2)], [2 for _ in range(2)], False),
        (1, [(6,) for _ in range(2)], [2 for _ in range(2)], True),
        (0, [(6,) for _ in range(2)], [2 for _ in range(2)], True),
    ],
)
def test_maddpg_getAction_agent_masking(
    epsilon, state_dims, action_dims, discrete_actions, device
):
    agent_ids = ["agent_0", "agent_1"]
    state = {agent: np.random.randn(*state_dims[0]) for agent in agent_ids}
    agent_mask = {"agent_0": False, "agent_1": True}
    if discrete_actions:
        env_defined_actions = {"agent_0": 1, "agent_1": None}
    else:
        env_defined_actions = {"agent_0": np.array([0, 1]), "agent_1": None}
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=False,
        n_agents=2,
        agent_ids=agent_ids,
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=discrete_actions,
        device=device,
    )
    cont_actions, discrete_action = maddpg.getAction(
        state, epsilon, agent_mask=agent_mask, env_defined_actions=env_defined_actions
    )
    if discrete_actions:
        assert np.array_equal(discrete_action["agent_0"], 1), discrete_action["agent_0"]
    assert np.array_equal(cont_actions["agent_0"], np.array([0, 1])), cont_actions[
        "agent_0"
    ]


@pytest.mark.parametrize(
    "state_dims, discrete_actions, batch_size, action_dims, agent_ids, one_hot",
    [
        ([(6,), (6,)], False, 64, [2, 2], ["agent_0", "agent_1"], False),
        ([(6,), (6,)], False, 64, [2, 2], ["agent_0", "agent_1"], True),
        ([(6,), (6,)], True, 64, [2, 2], ["agent_0", "agent_1"], False),
        ([(6,), (6,)], True, 64, [2, 2], ["agent_0", "agent_1"], True),
    ],
)
def test_maddpg_learns_from_experiences_mlp(
    state_dims,
    experiences,
    discrete_actions,
    batch_size,
    action_dims,
    agent_ids,
    one_hot,
    device,
):
    action_dims = [2, 2]
    agent_ids = ["agent_0", "agent_1"]
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=one_hot,
        n_agents=2,
        agent_ids=agent_ids,
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=discrete_actions,
        device=device,
    )
    actors = maddpg.actors
    actor_targets = maddpg.actor_targets
    actors_pre_learn_sd = [copy.deepcopy(actor.state_dict()) for actor in maddpg.actors]
    critics = maddpg.critics
    critic_targets = maddpg.critic_targets
    critics_pre_learn_sd = [
        str(copy.deepcopy(critic.state_dict())) for critic in maddpg.critics
    ]

    for _ in range(4):
        maddpg.scores.append(0)
        loss = maddpg.learn(experiences)

    assert isinstance(loss, dict)
    for agent_id in maddpg.agent_ids:
        assert loss[agent_id][-1] >= 0.0
    for old_actor, updated_actor in zip(actors, maddpg.actors):
        assert old_actor == updated_actor
    for old_actor_target, updated_actor_target in zip(
        actor_targets, maddpg.actor_targets
    ):
        assert old_actor_target == updated_actor_target
    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, maddpg.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())
    for old_critic, updated_critic in zip(critics, maddpg.critics):
        assert old_critic == updated_critic
    for old_critic_target, updated_critic_target in zip(
        critic_targets, maddpg.critic_targets
    ):
        assert old_critic_target == updated_critic_target
    for old_critic_state_dict, updated_critic in zip(
        critics_pre_learn_sd, maddpg.critics
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
    "state_dims, discrete_actions, batch_size, action_dims, agent_ids, one_hot",
    [
        ([(6,), (6,)], False, 64, [2, 2], ["agent_0", "agent_1"], False),
        ([(6,), (6,)], False, 64, [2, 2], ["agent_0", "agent_1"], True),
        ([(6,), (6,)], True, 64, [2, 2], ["agent_0", "agent_1"], False),
        ([(6,), (6,)], True, 64, [2, 2], ["agent_0", "agent_1"], True),
    ],
)
def test_maddpg_learns_from_experiences_mlp_distributed(
    state_dims,
    accelerated_experiences,
    discrete_actions,
    batch_size,
    action_dims,
    agent_ids,
    one_hot,
):
    accelerator = Accelerator(device_placement=False)
    action_dims = [2, 2]
    agent_ids = ["agent_0", "agent_1"]
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=one_hot,
        n_agents=2,
        agent_ids=agent_ids,
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=discrete_actions,
        accelerator=accelerator,
    )

    for actor, critic, actor_target, critic_target in zip(
        maddpg.actors, maddpg.critics, maddpg.actor_targets, maddpg.critic_targets
    ):
        actor.no_sync = no_sync.__get__(actor)
        critic.no_sync = no_sync.__get__(critic)
        actor_target.no_sync = no_sync.__get__(actor_target)
        critic_target.no_sync = no_sync.__get__(critic_target)

    actors = maddpg.actors
    actor_targets = maddpg.actor_targets
    actors_pre_learn_sd = [
        str(copy.deepcopy(actor.state_dict())) for actor in maddpg.actors
    ]
    critics = maddpg.critics
    critic_targets = maddpg.critic_targets
    critics_pre_learn_sd = [
        str(copy.deepcopy(critic.state_dict())) for critic in maddpg.critics
    ]

    for _ in range(3):
        maddpg.scores.append(0)
        loss = maddpg.learn(accelerated_experiences)

    assert isinstance(loss, dict)
    for agent_id in maddpg.agent_ids:
        assert loss[agent_id][-1] >= 0.0
    for old_actor, updated_actor in zip(actors, maddpg.actors):
        assert old_actor == updated_actor
    for old_actor_target, updated_actor_target in zip(
        actor_targets, maddpg.actor_targets
    ):
        assert old_actor_target == updated_actor_target
    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, maddpg.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())
    for old_critic, updated_critic in zip(critics, maddpg.critics):
        assert old_critic == updated_critic
    for old_critic_target, updated_critic_target in zip(
        critic_targets, maddpg.critic_targets
    ):
        assert old_critic_target == updated_critic_target
    for old_critic_state_dict, updated_critic in zip(
        critics_pre_learn_sd, maddpg.critics
    ):
        assert old_critic_state_dict != str(updated_critic.state_dict())


@pytest.mark.parametrize(
    "state_dims, discrete_actions, batch_size, action_dims, agent_ids, one_hot",
    [
        ([(3, 32, 32), (3, 32, 32)], False, 64, [2, 2], ["agent_0", "agent_1"], False),
        ([(3, 32, 32), (3, 32, 32)], True, 64, [2, 2], ["agent_0", "agent_1"], False),
    ],
)
def test_maddpg_learns_from_experiences_cnn(
    state_dims,
    experiences,
    discrete_actions,
    batch_size,
    action_dims,
    agent_ids,
    one_hot,
    device,
):
    action_dims = [2, 2]
    agent_ids = ["agent_0", "agent_1"]
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [16],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=one_hot,
        n_agents=2,
        net_config=net_config,
        agent_ids=agent_ids,
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=discrete_actions,
        device=device,
    )

    actors = maddpg.actors
    actor_targets = maddpg.actor_targets
    actors_pre_learn_sd = [copy.deepcopy(actor.state_dict()) for actor in maddpg.actors]
    critics = maddpg.critics
    critic_targets = maddpg.critic_targets
    critics_pre_learn_sd = [
        str(copy.deepcopy(critic.state_dict())) for critic in maddpg.critics
    ]

    for _ in range(4):
        maddpg.scores.append(0)
        loss = maddpg.learn(experiences)

    assert isinstance(loss, dict)
    for agent_id in maddpg.agent_ids:
        assert loss[agent_id][-1] >= 0.0
    for old_actor, updated_actor in zip(actors, maddpg.actors):
        assert old_actor == updated_actor
    for old_actor_target, updated_actor_target in zip(
        actor_targets, maddpg.actor_targets
    ):
        assert old_actor_target == updated_actor_target
    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, maddpg.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())
    for old_critic, updated_critic in zip(critics, maddpg.critics):
        assert old_critic == updated_critic
    for old_critic_target, updated_critic_target in zip(
        critic_targets, maddpg.critic_targets
    ):
        assert old_critic_target == updated_critic_target
    for old_critic_state_dict, updated_critic in zip(
        critics_pre_learn_sd, maddpg.critics
    ):
        assert old_critic_state_dict != str(updated_critic.state_dict())


@pytest.mark.parametrize(
    "state_dims, discrete_actions, batch_size, action_dims, agent_ids, one_hot",
    [
        ([(3, 32, 32), (3, 32, 32)], False, 64, [2, 2], ["agent_0", "agent_1"], False),
        ([(3, 32, 32), (3, 32, 32)], True, 64, [2, 2], ["agent_0", "agent_1"], False),
    ],
)
def test_maddpg_learns_from_experiences_cnn_distributed(
    state_dims,
    accelerated_experiences,
    discrete_actions,
    batch_size,
    action_dims,
    agent_ids,
    one_hot,
    device,
):
    accelerator = Accelerator(device_placement=False)
    action_dims = [2, 2]
    agent_ids = ["agent_0", "agent_1"]
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [16],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=one_hot,
        n_agents=2,
        net_config=net_config,
        agent_ids=agent_ids,
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=discrete_actions,
        accelerator=accelerator,
    )

    for actor, critic, actor_target, critic_target in zip(
        maddpg.actors, maddpg.critics, maddpg.actor_targets, maddpg.critic_targets
    ):
        actor.no_sync = no_sync.__get__(actor)
        critic.no_sync = no_sync.__get__(critic)
        actor_target.no_sync = no_sync.__get__(actor_target)
        critic_target.no_sync = no_sync.__get__(critic_target)

    actors = maddpg.actors
    actor_targets = maddpg.actor_targets
    actors_pre_learn_sd = [copy.deepcopy(actor.state_dict()) for actor in maddpg.actors]
    critics = maddpg.critics
    critic_targets = maddpg.critic_targets
    critics_pre_learn_sd = [
        str(copy.deepcopy(critic.state_dict())) for critic in maddpg.critics
    ]

    for _ in range(4):
        maddpg.scores.append(0)
        loss = maddpg.learn(accelerated_experiences)

    assert isinstance(loss, dict)
    for agent_id in maddpg.agent_ids:
        assert loss[agent_id][-1] >= 0.0
    for old_actor, updated_actor in zip(actors, maddpg.actors):
        assert old_actor == updated_actor
    for old_actor_target, updated_actor_target in zip(
        actor_targets, maddpg.actor_targets
    ):
        assert old_actor_target == updated_actor_target
    for old_actor_state_dict, updated_actor in zip(actors_pre_learn_sd, maddpg.actors):
        assert old_actor_state_dict != str(updated_actor.state_dict())
    for old_critic, updated_critic in zip(critics, maddpg.critics):
        assert old_critic == updated_critic
    for old_critic_target, updated_critic_target in zip(
        critic_targets, maddpg.critic_targets
    ):
        assert old_critic_target == updated_critic_target
    for old_critic_state_dict, updated_critic in zip(
        critics_pre_learn_sd, maddpg.critics
    ):
        assert old_critic_state_dict != str(updated_critic.state_dict())


def test_maddpg_soft_update(device):
    state_dims = [(6,), (6,)]
    action_dims = [2, 2]
    accelerator = None

    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=False,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=False,
        accelerator=accelerator,
        device=device,
    )

    for actor, actor_target, critic, critic_target in zip(
        maddpg.actors, maddpg.actor_targets, maddpg.critics, maddpg.critic_targets
    ):
        # Check actors
        maddpg.softUpdate(actor, actor_target)
        eval_params = list(actor.parameters())
        target_params = list(actor_target.parameters())
        expected_params = [
            maddpg.tau * eval_param + (1.0 - maddpg.tau) * target_param
            for eval_param, target_param in zip(eval_params, target_params)
        ]
        assert all(
            torch.allclose(expected_param, target_param)
            for expected_param, target_param in zip(expected_params, target_params)
        )
        maddpg.softUpdate(critic, critic_target)
        eval_params = list(critic.parameters())
        target_params = list(critic_target.parameters())
        expected_params = [
            maddpg.tau * eval_param + (1.0 - maddpg.tau) * target_param
            for eval_param, target_param in zip(eval_params, target_params)
        ]

        assert all(
            torch.allclose(expected_param, target_param)
            for expected_param, target_param in zip(expected_params, target_params)
        )


def test_maddpg_algorithm_test_loop(device):
    state_dims = [(6,), (6,)]
    action_dims = [2, 2]
    accelerator = None

    env = DummyMultiEnv(state_dims[0], action_dims)

    # env = makeVectEnvs("CartPole-v1", num_envs=num_envs)
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=False,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=True,
        accelerator=accelerator,
        device=device,
    )
    mean_score = maddpg.test(env, max_steps=10)
    assert isinstance(mean_score, float)


def test_maddpg_algorithm_test_loop_cnn(device):
    env_state_dims = [(32, 32, 3), (32, 32, 3)]
    agent_state_dims = [(3, 32, 32), (3, 32, 32)]
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [16],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    action_dims = [2, 2]
    accelerator = None
    env = DummyMultiEnv(env_state_dims[0], action_dims)
    maddpg = MADDPG(
        agent_state_dims,
        action_dims,
        one_hot=False,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        net_config=net_config,
        discrete_actions=False,
        accelerator=accelerator,
        device=device,
    )
    mean_score = maddpg.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


def test_maddpg_algorithm_test_loop_cnn_vectorized(device):
    env_state_dims = [(32, 32, 3), (32, 32, 3)]
    agent_state_dims = [(3, 32, 32), (3, 32, 32)]
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [16],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    action_dims = [2, 2]
    accelerator = None
    env = makeMultiAgentVectEnvs(DummyMultiEnv(env_state_dims[0], action_dims), 2)
    maddpg = MADDPG(
        agent_state_dims,
        action_dims,
        one_hot=False,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        net_config=net_config,
        discrete_actions=False,
        accelerator=accelerator,
        device=device,
    )
    mean_score = maddpg.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


@pytest.mark.parametrize(
    "accelerator_flag, wrap", [(False, True), (True, True), (True, False)]
)
def test_maddpg_clone_returns_identical_agent(accelerator_flag, wrap):
    # Clones the agent and returns an identical copy.
    state_dims = [(4,), (4,)]
    action_dims = [2, 2]
    one_hot = False
    n_agents = 2
    agent_ids = ["agent_0", "agent_1"]
    max_action = [(1,), (1,)]
    min_action = [(-1,), (-1,)]
    expl_noise = 0.1
    discrete_actions = False
    index = 0
    net_config = {"arch": "mlp", "h_size": [64, 64]}
    batch_size = 64
    lr_actor = 0.001
    lr_critic = 0.01
    learn_step = 5
    gamma = 0.95
    tau = 0.01
    mutation = None
    actor_networks = None
    critic_networks = None
    device = "cpu"
    if accelerator_flag:
        accelerator = Accelerator(device_placement=False)
    else:
        accelerator = None

    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot,
        n_agents,
        agent_ids,
        max_action,
        min_action,
        discrete_actions,
        expl_noise,
        index,
        net_config,
        batch_size,
        lr_actor,
        lr_critic,
        learn_step,
        gamma,
        tau,
        mutation,
        actor_networks,
        critic_networks,
        device,
        accelerator,
        wrap,
    )

    clone_agent = maddpg.clone(wrap=wrap)

    assert isinstance(clone_agent, MADDPG)
    assert clone_agent.state_dims == maddpg.state_dims
    assert clone_agent.action_dims == maddpg.action_dims
    assert clone_agent.one_hot == maddpg.one_hot
    assert clone_agent.n_agents == maddpg.n_agents
    assert clone_agent.agent_ids == maddpg.agent_ids
    assert clone_agent.max_action == maddpg.max_action
    assert clone_agent.min_action == maddpg.min_action
    assert clone_agent.expl_noise == maddpg.expl_noise
    assert clone_agent.discrete_actions == maddpg.discrete_actions
    assert clone_agent.index == maddpg.index
    assert clone_agent.net_config == maddpg.net_config
    assert clone_agent.batch_size == maddpg.batch_size
    assert clone_agent.lr_actor == maddpg.lr_actor
    assert clone_agent.lr_critic == maddpg.lr_critic
    assert clone_agent.learn_step == maddpg.learn_step
    assert clone_agent.gamma == maddpg.gamma
    assert clone_agent.tau == maddpg.tau
    assert clone_agent.device == maddpg.device
    assert clone_agent.accelerator == maddpg.accelerator
    for clone_actor, actor in zip(clone_agent.actors, maddpg.actors):
        assert str(clone_actor.state_dict()) == str(actor.state_dict())
    for clone_critic, critic in zip(clone_agent.critics, maddpg.critics):
        assert str(clone_critic.state_dict()) == str(critic.state_dict())
    for clone_actor_target, actor_target in zip(
        clone_agent.actor_targets, maddpg.actor_targets
    ):
        assert str(clone_actor_target.state_dict()) == str(actor_target.state_dict())
    for clone_critic_target, critic_target in zip(
        clone_agent.critic_targets, maddpg.critic_targets
    ):
        assert str(clone_critic_target.state_dict()) == str(critic_target.state_dict())
    assert clone_agent.actor_networks == maddpg.actor_networks
    assert clone_agent.critic_networks == maddpg.critic_networks


def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    net_config = {"arch": "mlp", "h_size": [32, 32]}
    # Initialize the maddpg agent
    maddpg = MADDPG(
        state_dims=[
            [
                6,
            ]
        ],
        action_dims=[2],
        one_hot=False,
        n_agents=1,
        agent_ids=["agent_0"],
        max_action=[[1]],
        min_action=[[-1]],
        net_config=net_config,
        discrete_actions=True,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    maddpg.saveCheckpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actors_init_dict" in checkpoint
    assert "actors_state_dict" in checkpoint
    assert "actor_targets_init_dict" in checkpoint
    assert "actor_targets_state_dict" in checkpoint
    assert "actor_optimizers_state_dict" in checkpoint
    assert "critics_init_dict" in checkpoint
    assert "critics_state_dict" in checkpoint
    assert "critic_targets_init_dict" in checkpoint
    assert "critic_targets_state_dict" in checkpoint
    assert "critic_optimizers_state_dict" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr_actor" in checkpoint
    assert "lr_critic" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mutation" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # Load checkpoint
    loaded_maddpg = MADDPG(
        state_dims=[
            [
                6,
            ]
        ],
        action_dims=[2],
        one_hot=False,
        n_agents=1,
        agent_ids=["agent_0"],
        max_action=[(1,)],
        min_action=[(-1,)],
        discrete_actions=True,
    )
    loaded_maddpg.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert loaded_maddpg.net_config == net_config
    assert all(isinstance(actor, EvolvableMLP) for actor in loaded_maddpg.actors)
    assert all(
        isinstance(actor_target, EvolvableMLP)
        for actor_target in loaded_maddpg.actor_targets
    )
    assert all(isinstance(critic, EvolvableMLP) for critic in loaded_maddpg.critics)
    assert all(
        isinstance(critic_target, EvolvableMLP)
        for critic_target in loaded_maddpg.critic_targets
    )
    assert maddpg.lr_actor == 0.001
    assert maddpg.lr_critic == 0.01

    for actor, actor_target in zip(loaded_maddpg.actors, loaded_maddpg.actor_targets):
        assert str(actor.state_dict()) == str(actor_target.state_dict())

    for critic, critic_target in zip(
        loaded_maddpg.critics, loaded_maddpg.critic_targets
    ):
        assert str(critic.state_dict()) == str(critic_target.state_dict())

    assert maddpg.batch_size == 64
    assert maddpg.learn_step == 5
    assert maddpg.gamma == 0.95
    assert maddpg.tau == 0.01
    assert maddpg.mut is None
    assert maddpg.index == 0
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]


def test_maddpg_save_load_checkpoint_correct_data_and_format_cnn(tmpdir):
    net_config_cnn = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [16],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }

    # Initialize the maddpg agent
    maddpg = MADDPG(
        state_dims=[[3, 32, 32]],
        action_dims=[2],
        one_hot=False,
        n_agents=1,
        agent_ids=["agent_0"],
        net_config=net_config_cnn,
        max_action=[[1]],
        min_action=[[-1]],
        discrete_actions=True,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    maddpg.saveCheckpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actors_init_dict" in checkpoint
    assert "actors_state_dict" in checkpoint
    assert "actor_targets_init_dict" in checkpoint
    assert "actor_targets_state_dict" in checkpoint
    assert "actor_optimizers_state_dict" in checkpoint
    assert "critics_init_dict" in checkpoint
    assert "critics_state_dict" in checkpoint
    assert "critic_targets_init_dict" in checkpoint
    assert "critic_targets_state_dict" in checkpoint
    assert "critic_optimizers_state_dict" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr_actor" in checkpoint
    assert "lr_critic" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mutation" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # Load checkpoint
    loaded_maddpg = MADDPG(
        state_dims=[[3, 32, 32]],
        action_dims=[2],
        one_hot=False,
        n_agents=1,
        agent_ids=["agent_0"],
        max_action=[(1,)],
        min_action=[(-1,)],
        discrete_actions=True,
    )
    loaded_maddpg.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert loaded_maddpg.net_config == net_config_cnn
    assert all(isinstance(actor, EvolvableCNN) for actor in loaded_maddpg.actors)
    assert all(
        isinstance(actor_target, EvolvableCNN)
        for actor_target in loaded_maddpg.actor_targets
    )
    assert all(isinstance(critic, EvolvableCNN) for critic in loaded_maddpg.critics)
    assert all(
        isinstance(critic_target, EvolvableCNN)
        for critic_target in loaded_maddpg.critic_targets
    )
    assert maddpg.lr_actor == 0.001
    assert maddpg.lr_critic == 0.01

    for actor, actor_target in zip(loaded_maddpg.actors, loaded_maddpg.actor_targets):
        assert str(actor.state_dict()) == str(actor_target.state_dict())

    for critic, critic_target in zip(
        loaded_maddpg.critics, loaded_maddpg.critic_targets
    ):
        assert str(critic.state_dict()) == str(critic_target.state_dict())

    assert maddpg.batch_size == 64
    assert maddpg.learn_step == 5
    assert maddpg.gamma == 0.95
    assert maddpg.tau == 0.01
    assert maddpg.mut is None
    assert maddpg.index == 0
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]


@pytest.mark.parametrize(
    "state_dims, action_dims",
    [
        (
            [[6]],
            [2],
        )
    ],
)
def test_maddpg_save_load_checkpoint_correct_data_and_format_make_evo(
    tmpdir, state_dims, action_dims, mlp_actor, mlp_critic, device
):
    evo_actors = [
        MakeEvolvable(network=mlp_actor, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(1)
    ]
    evo_critics = [
        MakeEvolvable(network=mlp_critic, input_tensor=torch.randn(1, 8), device=device)
        for _ in range(1)
    ]
    maddpg = MADDPG(
        state_dims=state_dims,
        action_dims=action_dims,
        one_hot=False,
        n_agents=1,
        agent_ids=["agent_0"],
        max_action=[[1]],
        min_action=[[-1]],
        discrete_actions=True,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
    )
    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    maddpg.saveCheckpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actors_init_dict" in checkpoint
    assert "actors_state_dict" in checkpoint
    assert "actor_targets_init_dict" in checkpoint
    assert "actor_targets_state_dict" in checkpoint
    assert "actor_optimizers_state_dict" in checkpoint
    assert "critics_init_dict" in checkpoint
    assert "critics_state_dict" in checkpoint
    assert "critic_targets_init_dict" in checkpoint
    assert "critic_targets_state_dict" in checkpoint
    assert "critic_optimizers_state_dict" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr_actor" in checkpoint
    assert "lr_critic" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mutation" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # Load checkpoint
    loaded_maddpg = MADDPG(
        state_dims=[[3, 32, 32]],
        action_dims=[2],
        one_hot=False,
        n_agents=1,
        agent_ids=["agent_0"],
        max_action=[(1,)],
        min_action=[(-1,)],
        discrete_actions=True,
    )
    loaded_maddpg.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert all(isinstance(actor, MakeEvolvable) for actor in loaded_maddpg.actors)
    assert all(
        isinstance(actor_target, MakeEvolvable)
        for actor_target in loaded_maddpg.actor_targets
    )
    assert all(isinstance(critic, MakeEvolvable) for critic in loaded_maddpg.critics)
    assert all(
        isinstance(critic_target, MakeEvolvable)
        for critic_target in loaded_maddpg.critic_targets
    )
    assert maddpg.lr_actor == 0.001
    assert maddpg.lr_critic == 0.01

    for actor, actor_target in zip(loaded_maddpg.actors, loaded_maddpg.actor_targets):
        assert str(actor.state_dict()) == str(actor_target.state_dict())

    for critic, critic_target in zip(
        loaded_maddpg.critics, loaded_maddpg.critic_targets
    ):
        assert str(critic.state_dict()) == str(critic_target.state_dict())

    assert maddpg.batch_size == 64
    assert maddpg.learn_step == 5
    assert maddpg.gamma == 0.95
    assert maddpg.tau == 0.01
    assert maddpg.mut is None
    assert maddpg.index == 0
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]


def test_maddpg_unwrap_models():
    state_dims = [(6,), (6,)]
    action_dims = [2, 2]
    accelerator = Accelerator()
    maddpg = MADDPG(
        state_dims,
        action_dims,
        one_hot=False,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=True,
        accelerator=accelerator,
    )
    maddpg.unwrap_models()
    for actor, critic, actor_target, critic_target in zip(
        maddpg.actors, maddpg.critics, maddpg.actor_targets, maddpg.critic_targets
    ):
        assert isinstance(actor, nn.Module)
        assert isinstance(actor_target, nn.Module)
        assert isinstance(critic, nn.Module)
        assert isinstance(critic_target, nn.Module)


# Returns the input action scaled to the action space defined by self.min_action and self.max_action.
def test_action_scaling():
    action = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    max_actions = [(1,), (2,), (1,), (2,), (2,)]
    min_actions = [(-1,), (-2,), (0,), (0,), (-1,)]

    maddpg = MADDPG(
        state_dims=[[4], [4], [4], [4], [4]],
        action_dims=[1, 1, 1, 1, 1],
        n_agents=5,
        agent_ids=["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"],
        discrete_actions=False,
        one_hot=False,
        max_action=max_actions,
        min_action=min_actions,
    )

    scaled_action = maddpg.scale_to_action_space(action, idx=0)
    assert np.array_equal(scaled_action, np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))

    action = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    scaled_action = maddpg.scale_to_action_space(action, idx=1)
    assert np.array_equal(scaled_action, np.array([0.2, 0.4, 0.6, -0.2, -0.4, -0.6]))

    action = np.array([0.1, 0.2, 0.3, 0])
    scaled_action = maddpg.scale_to_action_space(action, idx=2)
    assert np.array_equal(scaled_action, np.array([0.1, 0.2, 0.3, 0]))

    action = np.array([0.1, 0.2, 0.3, 0])
    scaled_action = maddpg.scale_to_action_space(action, idx=3)
    assert np.array_equal(scaled_action, np.array([0.2, 0.4, 0.6, 0]))

    action = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    scaled_action = maddpg.scale_to_action_space(action, idx=4)
    assert np.array_equal(scaled_action, np.array([0.2, 0.4, 0.6, -0.1, -0.2, -0.3]))


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained(device, accelerator, tmpdir):
    # Initialize the maddpg agent
    maddpg = MADDPG(
        state_dims=[[4], [4]],
        action_dims=[2, 2],
        one_hot=False,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=True,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    maddpg.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_maddpg = MADDPG.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_maddpg.state_dims == maddpg.state_dims
    assert new_maddpg.action_dims == maddpg.action_dims
    assert new_maddpg.one_hot == maddpg.one_hot
    assert new_maddpg.n_agents == maddpg.n_agents
    assert new_maddpg.agent_ids == maddpg.agent_ids
    assert new_maddpg.min_action == maddpg.min_action
    assert new_maddpg.max_action == maddpg.max_action
    assert new_maddpg.net_config == maddpg.net_config
    assert new_maddpg.lr_actor == maddpg.lr_actor
    assert new_maddpg.lr_critic == maddpg.lr_critic
    for (
        new_actor,
        new_actor_target,
        new_critic,
        new_critic_target,
        actor,
        actor_target,
        critic,
        critic_target,
    ) in zip(
        new_maddpg.actors,
        new_maddpg.actor_targets,
        new_maddpg.critics,
        new_maddpg.critic_targets,
        maddpg.actors,
        maddpg.actor_targets,
        maddpg.critics,
        maddpg.critic_targets,
    ):
        assert isinstance(new_actor, EvolvableMLP)
        assert isinstance(new_actor_target, EvolvableMLP)
        assert isinstance(new_critic, EvolvableMLP)
        assert isinstance(new_critic_target, EvolvableMLP)
        assert str(new_actor.to("cpu").state_dict()) == str(actor.state_dict())
        assert str(new_actor_target.to("cpu").state_dict()) == str(
            actor_target.state_dict()
        )
        assert str(new_critic.to("cpu").state_dict()) == str(critic.state_dict())
        assert str(new_critic_target.to("cpu").state_dict()) == str(
            critic_target.state_dict()
        )
    assert new_maddpg.batch_size == maddpg.batch_size
    assert new_maddpg.learn_step == maddpg.learn_step
    assert new_maddpg.gamma == maddpg.gamma
    assert new_maddpg.tau == maddpg.tau
    assert new_maddpg.mut == maddpg.mut
    assert new_maddpg.index == maddpg.index
    assert new_maddpg.scores == maddpg.scores
    assert new_maddpg.fitness == maddpg.fitness
    assert new_maddpg.steps == maddpg.steps


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_cnn(device, accelerator, tmpdir):
    # Initialize the maddpg agent
    maddpg = MADDPG(
        state_dims=[[3, 32, 32], [3, 32, 32]],
        action_dims=[2, 2],
        one_hot=False,
        n_agents=2,
        agent_ids=["agent_a", "agent_b"],
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=False,
        net_config={
            "arch": "cnn",
            "h_size": [8],
            "c_size": [3],
            "k_size": [3],
            "s_size": [1],
            "normalize": False,
        },
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    maddpg.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_maddpg = MADDPG.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_maddpg.state_dims == maddpg.state_dims
    assert new_maddpg.action_dims == maddpg.action_dims
    assert new_maddpg.one_hot == maddpg.one_hot
    assert new_maddpg.n_agents == maddpg.n_agents
    assert new_maddpg.agent_ids == maddpg.agent_ids
    assert new_maddpg.min_action == maddpg.min_action
    assert new_maddpg.max_action == maddpg.max_action
    assert new_maddpg.net_config == maddpg.net_config
    assert new_maddpg.lr_actor == maddpg.lr_actor
    assert new_maddpg.lr_critic == maddpg.lr_critic
    for (
        new_actor,
        new_actor_target,
        new_critic,
        new_critic_target,
        actor,
        actor_target,
        critic,
        critic_target,
    ) in zip(
        new_maddpg.actors,
        new_maddpg.actor_targets,
        new_maddpg.critics,
        new_maddpg.critic_targets,
        maddpg.actors,
        maddpg.actor_targets,
        maddpg.critics,
        maddpg.critic_targets,
    ):
        assert isinstance(new_actor, EvolvableCNN)
        assert isinstance(new_actor_target, EvolvableCNN)
        assert isinstance(new_critic, EvolvableCNN)
        assert isinstance(new_critic_target, EvolvableCNN)
        assert str(new_actor.to("cpu").state_dict()) == str(actor.state_dict())
        assert str(new_actor_target.to("cpu").state_dict()) == str(
            actor_target.state_dict()
        )
        assert str(new_critic.to("cpu").state_dict()) == str(critic.state_dict())
        assert str(new_critic_target.to("cpu").state_dict()) == str(
            critic_target.state_dict()
        )
    assert new_maddpg.batch_size == maddpg.batch_size
    assert new_maddpg.learn_step == maddpg.learn_step
    assert new_maddpg.gamma == maddpg.gamma
    assert new_maddpg.tau == maddpg.tau
    assert new_maddpg.mut == maddpg.mut
    assert new_maddpg.index == maddpg.index
    assert new_maddpg.scores == maddpg.scores
    assert new_maddpg.fitness == maddpg.fitness
    assert new_maddpg.steps == maddpg.steps


@pytest.mark.parametrize(
    "state_dims, action_dims, arch, input_tensor, critic_input_tensor, secondary_input_tensor, extra_critic_dims",
    [
        ([[4], [4]], [2, 2], "mlp", torch.randn(1, 4), torch.randn(1, 6), None, None),
        (
            [[4, 210, 160], [4, 210, 160]],
            [2, 2],
            "cnn",
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 2),
            2,
        ),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_networks(
    mlp_actor,
    mlp_critic,
    cnn_actor,
    cnn_critic,
    state_dims,
    action_dims,
    arch,
    input_tensor,
    critic_input_tensor,
    secondary_input_tensor,
    extra_critic_dims,
    tmpdir,
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
        extra_critic_dims=extra_critic_dims,
    )

    # Initialize the maddpg agent
    maddpg = MADDPG(
        state_dims=state_dims,
        action_dims=action_dims,
        one_hot=one_hot,
        n_agents=2,
        agent_ids=["agent_0", "agent_1"],
        max_action=[[1], [1]],
        min_action=[[-1], [-1]],
        discrete_actions=True,
        actor_networks=[actor_network, copy.deepcopy(actor_network)],
        critic_networks=[critic_network, copy.deepcopy(critic_network)],
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    maddpg.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_maddpg = MADDPG.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_maddpg.state_dims == maddpg.state_dims
    assert new_maddpg.action_dims == maddpg.action_dims
    assert new_maddpg.one_hot == maddpg.one_hot
    assert new_maddpg.n_agents == maddpg.n_agents
    assert new_maddpg.agent_ids == maddpg.agent_ids
    assert new_maddpg.min_action == maddpg.min_action
    assert new_maddpg.max_action == maddpg.max_action
    assert new_maddpg.net_config == maddpg.net_config
    assert new_maddpg.lr_actor == maddpg.lr_actor
    assert new_maddpg.lr_critic == maddpg.lr_critic
    for (
        new_actor,
        new_actor_target,
        new_critic,
        new_critic_target,
        actor,
        actor_target,
        critic,
        critic_target,
    ) in zip(
        new_maddpg.actors,
        new_maddpg.actor_targets,
        new_maddpg.critics,
        new_maddpg.critic_targets,
        maddpg.actors,
        maddpg.actor_targets,
        maddpg.critics,
        maddpg.critic_targets,
    ):
        assert isinstance(new_actor, nn.Module)
        assert isinstance(new_actor_target, nn.Module)
        assert isinstance(new_critic, nn.Module)
        assert isinstance(new_critic_target, nn.Module)
        assert str(new_actor.to("cpu").state_dict()) == str(actor.state_dict())
        assert str(new_actor_target.to("cpu").state_dict()) == str(
            actor_target.state_dict()
        )
        assert str(new_critic.to("cpu").state_dict()) == str(critic.state_dict())
        assert str(new_critic_target.to("cpu").state_dict()) == str(
            critic_target.state_dict()
        )
    assert new_maddpg.batch_size == maddpg.batch_size
    assert new_maddpg.learn_step == maddpg.learn_step
    assert new_maddpg.gamma == maddpg.gamma
    assert new_maddpg.tau == maddpg.tau
    assert new_maddpg.mut == maddpg.mut
    assert new_maddpg.index == maddpg.index
    assert new_maddpg.scores == maddpg.scores
    assert new_maddpg.fitness == maddpg.fitness
    assert new_maddpg.steps == maddpg.steps
