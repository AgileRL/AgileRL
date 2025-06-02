import copy
import gc
from pathlib import Path

import dill
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from gymnasium.spaces import Discrete
from torch._dynamo import OptimizedModule

from agilerl.algorithms.matd3 import MATD3
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput, ModuleDict
from agilerl.modules.custom_components import GumbelSoftmax
from agilerl.networks.actors import DeterministicActor
from agilerl.networks.q_networks import ContinuousQNetwork
from agilerl.utils.algo_utils import concatenate_spaces
from agilerl.utils.evolvable_networks import get_default_encoder_config
from agilerl.utils.utils import make_multi_agent_vect_envs
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    gen_multi_agent_dict_or_tuple_spaces,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_multi_agent_multidiscrete_spaces,
)
from tests.test_algorithms.test_maddpg import DummyMultiEnv


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


class DummyContinuousQNetwork(ContinuousQNetwork):
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


class DummyDeterministicActor(DeterministicActor):
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
    yield net
    del net
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def mlp_critic(action_spaces, observation_spaces):
    net = nn.Sequential(
        nn.Linear(observation_spaces[0].shape[0] + action_spaces[0].n, 64),
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
def accelerated_experiences(batch_size, observation_spaces, action_spaces, agent_ids):
    one_hot = all(isinstance(space, Discrete) for space in observation_spaces)
    discrete_actions = all(isinstance(space, Discrete) for space in action_spaces)
    state_size = (
        observation_spaces[0].shape if not one_hot else (observation_spaces[0].n,)
    )
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

    yield states, actions, rewards, next_states, dones


@pytest.fixture
def experiences(batch_size, observation_spaces, action_spaces, agent_ids, device):
    one_hot = all(isinstance(space, Discrete) for space in observation_spaces)
    discrete_actions = all(isinstance(space, Discrete) for space in action_spaces)
    state_size = (
        observation_spaces[0].shape if not one_hot else (observation_spaces[0].n,)
    )
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

    yield states, actions, rewards, next_states, dones


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (4,)),
        generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
        gen_multi_agent_dict_or_tuple_spaces(2, 2, 2, dict_space=True),
        gen_multi_agent_dict_or_tuple_spaces(2, 2, 2, dict_space=False),
        generate_multi_agent_multidiscrete_spaces(2, 2),
    ],
)
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_matd3_with_net_config(
    observation_spaces, accelerator_flag, device, compile_mode
):
    net_config = {
        "encoder_config": get_default_encoder_config(observation_spaces[0]),
        "head_config": {"hidden_size": [16]},
    }
    action_spaces = generate_multi_agent_box_spaces(2, (2,))
    agent_ids = ["agent_0", "other_agent_0"]
    expl_noise = 0.1
    batch_size = 64
    policy_freq = 2
    accelerator = Accelerator() if accelerator_flag else None
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        net_config=net_config,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        accelerator=accelerator,
        device=device,
        policy_freq=policy_freq,
        torch_compiler=compile_mode,
    )
    assert matd3.observation_spaces == observation_spaces
    assert matd3.action_spaces == action_spaces
    assert matd3.policy_freq == policy_freq
    assert matd3.n_agents == len(agent_ids)
    assert matd3.agent_ids == agent_ids
    for noise_vec in matd3.expl_noise.values():
        assert torch.all(noise_vec == expl_noise)

    assert matd3.batch_size == batch_size
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]

    if compile_mode is not None and accelerator is None:
        assert all(
            isinstance(actor, OptimizedModule) for actor in matd3.actors.values()
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in matd3.critics_1.values()
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in matd3.critics_2.values()
        )
    else:
        assert all(
            isinstance(actor, DeterministicActor) for actor in matd3.actors.values()
        )
        assert all(
            isinstance(critic, ContinuousQNetwork)
            for critic in matd3.critics_1.values()
        )
        assert all(
            isinstance(critic, ContinuousQNetwork)
            for critic in matd3.critics_2.values()
        )

    expected_optimizer_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    for agent_id in matd3.agent_ids:
        actor_optimizer = matd3.actor_optimizers[agent_id]
        critic_1_optimizer = matd3.critic_1_optimizers[agent_id]
        critic_2_optimizer = matd3.critic_2_optimizers[agent_id]
        assert isinstance(actor_optimizer, expected_optimizer_cls)
        assert isinstance(critic_1_optimizer, expected_optimizer_cls)
        assert isinstance(critic_2_optimizer, expected_optimizer_cls)

    assert isinstance(matd3.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_spaces", [generate_multi_agent_box_spaces(2, (6,))]
)
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(2, 2)])
def test_initialize_matd3_with_mlp_networks_gumbel_softmax(
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    device,
):
    compile_mode = "reduce-overhead"
    net_config = {
        "encoder_config": {
            "hidden_size": [64, 64],
            "min_hidden_layers": 1,
            "max_hidden_layers": 3,
            "min_mlp_nodes": 64,
            "max_mlp_nodes": 500,
            "output_activation": "GumbelSoftmax",
            "activation": "ReLU",
        }
    }
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "other_agent_0"],
        net_config=net_config,
        device=device,
        torch_compiler=compile_mode,
    )
    assert matd3.torch_compiler == compile_mode


# TODO: This will be deprecated in the future
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize(
    "observation_spaces", [generate_multi_agent_box_spaces(2, (6,))]
)
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(2, 2)])
def test_initialize_matd3_with_mlp_networks(
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    accelerator_flag,
    device,
    compile_mode,
):
    accelerator = Accelerator() if accelerator_flag else None
    agent_ids = ["agent_0", "other_agent_0"]
    evo_actors = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=mlp_actor, input_tensor=torch.randn(1, 6), device=device
            )
            for agent_id in agent_ids
        }
    )
    evo_critics_1 = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=mlp_critic, input_tensor=torch.randn(1, 8), device=device
            )
            for agent_id in agent_ids
        }
    )
    evo_critics_2 = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=mlp_critic, input_tensor=torch.randn(1, 8), device=device
            )
            for agent_id in agent_ids
        }
    )
    evo_critics = [evo_critics_1, evo_critics_2]
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        policy_freq=2,
        torch_compiler=compile_mode,
    )
    expected_module_cls = (
        OptimizedModule
        if compile_mode is not None and accelerator is None
        else MakeEvolvable
    )
    assert all(
        isinstance(actor, expected_module_cls) for actor in matd3.actors.values()
    )
    assert all(
        isinstance(critic, expected_module_cls) for critic in matd3.critics_1.values()
    )
    assert all(
        isinstance(critic, expected_module_cls) for critic in matd3.critics_2.values()
    )

    assert matd3.observation_spaces == observation_spaces
    assert matd3.action_spaces == action_spaces
    assert matd3.n_agents == 2
    assert matd3.policy_freq == 2
    assert matd3.agent_ids == agent_ids
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]

    expected_optimizer_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    for agent_id in matd3.agent_ids:
        actor_optimizer = matd3.actor_optimizers[agent_id]
        critic_1_optimizer = matd3.critic_1_optimizers[agent_id]
        critic_2_optimizer = matd3.critic_2_optimizers[agent_id]
        assert isinstance(actor_optimizer, expected_optimizer_cls)
        assert isinstance(critic_1_optimizer, expected_optimizer_cls)
        assert isinstance(critic_2_optimizer, expected_optimizer_cls)

    assert isinstance(matd3.criterion, nn.MSELoss)


# TODO: This will be deprecated in the future
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_matd3_with_cnn_networks(
    cnn_actor,
    cnn_critic,
    accelerator_flag,
    device,
    compile_mode,
):
    observation_spaces = generate_multi_agent_box_spaces(
        2, (4, 210, 160), low=0, high=255
    )
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    accelerator = Accelerator() if accelerator_flag else None
    agent_ids = ["agent_0", "other_agent_0"]
    evo_actors = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=cnn_actor,
                input_tensor=torch.randn(1, 4, 2, 210, 160),
                device=device,
            )
            for agent_id in agent_ids
        }
    )
    evo_critics_1 = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=cnn_critic,
                input_tensor=torch.randn(1, 4, 2, 210, 160),
                secondary_input_tensor=torch.randn(1, 2),
                device=device,
            )
            for agent_id in agent_ids
        }
    )
    evo_critics_2 = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=cnn_critic,
                input_tensor=torch.randn(1, 4, 2, 210, 160),
                secondary_input_tensor=torch.randn(1, 2),
                device=device,
            )
            for agent_id in agent_ids
        }
    )
    evo_critics = [evo_critics_1, evo_critics_2]
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        policy_freq=2,
        torch_compiler=compile_mode,
    )
    expected_module_cls = (
        OptimizedModule
        if compile_mode is not None and accelerator is None
        else MakeEvolvable
    )
    assert all(
        isinstance(actor, expected_module_cls) for actor in matd3.actors.values()
    )
    assert all(
        isinstance(critic, expected_module_cls) for critic in matd3.critics_1.values()
    )
    assert all(
        isinstance(critic, expected_module_cls) for critic in matd3.critics_2.values()
    )
    assert matd3.observation_spaces == observation_spaces
    assert matd3.policy_freq == 2
    assert matd3.action_spaces == action_spaces
    assert matd3.n_agents == 2
    assert matd3.agent_ids == agent_ids
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]

    expected_optimizer_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    for agent_id in matd3.agent_ids:
        actor_optimizer = matd3.actor_optimizers[agent_id]
        critic_1_optimizer = matd3.critic_1_optimizers[agent_id]
        critic_2_optimizer = matd3.critic_2_optimizers[agent_id]
        assert isinstance(actor_optimizer, expected_optimizer_cls)
        assert isinstance(critic_1_optimizer, expected_optimizer_cls)
        assert isinstance(critic_2_optimizer, expected_optimizer_cls)

    assert isinstance(matd3.criterion, nn.MSELoss)


@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize(
    "observation_spaces, encoder_cls",
    [
        (generate_multi_agent_box_spaces(2, (4,)), EvolvableMLP),
        (
            generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255),
            EvolvableCNN,
        ),
    ],
)
def test_initialize_matd3_with_evo_networks(
    observation_spaces, encoder_cls, device, compile_mode, accelerator
):
    agent_ids = ["agent_0", "other_agent_0"]
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    observation_space = spaces.Dict(
        {agent_id: observation_spaces[idx] for idx, agent_id in enumerate(agent_ids)}
    )

    evo_actors = ModuleDict(
        {
            agent_id: DeterministicActor(
                observation_spaces[idx], action_spaces[idx], device=device
            )
            for idx, agent_id in enumerate(agent_ids)
        }
    )
    evo_critics_1 = ModuleDict(
        {
            agent_id: ContinuousQNetwork(
                observation_space=observation_space,
                action_space=concatenate_spaces(action_spaces),
                device=device,
            )
            for agent_id in agent_ids
        }
    )
    evo_critics_2 = ModuleDict(
        {
            agent_id: ContinuousQNetwork(
                observation_space=observation_space,
                action_space=concatenate_spaces(action_spaces),
                device=device,
            )
            for agent_id in agent_ids
        }
    )
    evo_critics = [evo_critics_1, evo_critics_2]
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
    )
    if compile_mode is not None and accelerator is None:
        assert all(
            isinstance(actor, OptimizedModule) for actor in matd3.actors.values()
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in matd3.critics_1.values()
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in matd3.critics_2.values()
        )
    else:
        assert all(
            isinstance(actor.encoder, encoder_cls) for actor in matd3.actors.values()
        )
        assert all(
            isinstance(critic.encoder, EvolvableMultiInput)
            for critic in matd3.critics_1.values()
        )
        assert all(
            isinstance(critic.encoder, EvolvableMultiInput)
            for critic in matd3.critics_2.values()
        )
    assert matd3.observation_spaces == observation_spaces
    assert matd3.policy_freq == 2
    assert matd3.action_spaces == action_spaces
    assert matd3.n_agents == 2
    assert matd3.agent_ids == agent_ids
    assert matd3.scores == []
    assert matd3.fitness == []
    assert matd3.steps == [0]

    expected_optimizer_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    for agent_id in matd3.agent_ids:
        actor_optimizer = matd3.actor_optimizers[agent_id]
        critic_1_optimizer = matd3.critic_1_optimizers[agent_id]
        critic_2_optimizer = matd3.critic_2_optimizers[agent_id]
        assert isinstance(actor_optimizer, expected_optimizer_cls)
        assert isinstance(critic_1_optimizer, expected_optimizer_cls)
        assert isinstance(critic_2_optimizer, expected_optimizer_cls)

    assert isinstance(matd3.criterion, nn.MSELoss)


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_matd3_with_incorrect_evo_networks(compile_mode):
    evo_actors = []
    evo_critics = []
    observation_spaces = generate_multi_agent_box_spaces(2, (4,))
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    with pytest.raises(AssertionError):
        _ = MATD3(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=["agent_0", "other_agent_0"],
            actor_networks=evo_actors,
            critic_networks=evo_critics,
            torch_compiler=compile_mode,
        )


@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize(
    "observation_spaces", [generate_multi_agent_box_spaces(2, (6,))]
)
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(2, 2)])
def test_matd3_init_warning(
    mlp_actor, device, compile_mode, observation_spaces, action_spaces
):
    warning_string = "Actor and critic network must both be supplied to use custom networks. Defaulting to net config."
    agent_ids = ["agent_0", "other_agent_0"]
    evo_actors = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=mlp_actor, input_tensor=torch.randn(1, 6), device=device
            )
            for agent_id in agent_ids
        }
    )
    with pytest.warns(UserWarning, match=warning_string):
        MATD3(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=agent_ids,
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
        agent_ids=["agent_0", "other_agent_0"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_compiler=mode,
    )
    if isinstance(mode, str):
        assert all(isinstance(a, OptimizedModule) for a in matd3.actors.values())
        assert all(isinstance(a, OptimizedModule) for a in matd3.actor_targets.values())
        assert all(isinstance(a, OptimizedModule) for a in matd3.critics_1.values())
        assert all(isinstance(a, OptimizedModule) for a in matd3.critics_2.values())
        assert all(
            isinstance(a, OptimizedModule) for a in matd3.critic_targets_1.values()
        )
        assert all(
            isinstance(a, OptimizedModule) for a in matd3.critic_targets_2.values()
        )
        assert matd3.torch_compiler == mode
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
            agent_ids=["agent_0", "other_agent_0"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_compiler=mode,
        )


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (6,)),
        generate_multi_agent_discrete_spaces(2, 6),
        generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255),
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_box_spaces(2, (2,), low=-1, high=1),
        generate_multi_agent_discrete_spaces(2, 2),
    ],
)
@pytest.mark.parametrize("training", [0, 1])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_get_action(
    training, observation_spaces, action_spaces, device, compile_mode
):
    agent_ids = ["agent_0", "other_agent_0"]
    if all(isinstance(space, spaces.Discrete) for space in observation_spaces):
        state = {
            agent: np.random.randint(0, observation_spaces[idx].n, 1)
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
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )
    matd3.set_training_mode(bool(training))
    processed_action, raw_action = matd3.get_action(state)
    discrete_actions = all(
        isinstance(space, spaces.Discrete) for space in action_spaces
    )
    for idx, env_actions in enumerate(list(raw_action.values())):
        action_dim = (
            action_spaces[idx].shape[0]
            if isinstance(action_spaces[idx], spaces.Box)
            else action_spaces[idx].n
        )
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
        for idx, env_action in enumerate(list(processed_action.values())):
            for action in env_action:
                assert action <= action_spaces[idx].n - 1
    matd3 = None


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (6,)),
        generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255),
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_discrete_spaces(2, 2),
        generate_multi_agent_box_spaces(2, (2,)),
    ],
)
@pytest.mark.parametrize("training", [0, 1])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_get_action_distributed(
    training, observation_spaces, action_spaces, compile_mode
):
    accelerator = Accelerator()
    agent_ids = ["agent_0", "other_agent_0"]
    state = {
        agent: np.random.randn(*observation_spaces[idx].shape)
        for idx, agent in enumerate(agent_ids)
    }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    new_actors = ModuleDict(
        {
            agent_id: DummyDeterministicActor(
                observation_space=actor.observation_space,
                action_space=actor.action_space,
                encoder_config=actor.encoder.net_config,
                head_config=actor.head_net.net_config,
                device=actor.device,
            )
            for agent_id, actor in matd3.actors.items()
        }
    )
    matd3.actors = new_actors
    matd3.set_training_mode(bool(training))
    processed_action, raw_action = matd3.get_action(state)
    discrete_actions = all(
        isinstance(space, spaces.Discrete) for space in action_spaces
    )
    for idx, env_actions in enumerate(list(raw_action.values())):
        action_dim = (
            action_spaces[idx].shape[0]
            if isinstance(action_spaces[idx], spaces.Box)
            else action_spaces[idx].n
        )
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
        for idx, env_action in enumerate(list(processed_action.values())):
            action_dim = (
                action_spaces[idx].shape[0]
                if isinstance(action_spaces[idx], spaces.Box)
                else action_spaces[idx].n
            )
            for action in env_action:
                assert action <= action_dim - 1


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (6,)),
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_box_spaces(2, (2,)),
        generate_multi_agent_discrete_spaces(2, 2),
    ],
)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_get_action_agent_masking(
    training, observation_spaces, action_spaces, device, compile_mode
):
    agent_ids = ["agent_0", "other_agent_0"]
    state = {
        agent: np.random.randn(*observation_spaces[0].shape) for agent in agent_ids
    }
    discrete_actions = all(
        isinstance(space, spaces.Discrete) for space in action_spaces
    )
    if discrete_actions:
        info = {
            "agent_0": {"env_defined_actions": 1},
            "other_agent_0": {"env_defined_actions": None},
        }
    else:
        info = {
            "agent_0": {"env_defined_actions": np.array([0, 1])},
            "other_agent_0": {"env_defined_actions": None},
        }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )
    matd3.set_training_mode(training)
    action, _ = matd3.get_action(state, infos=info)
    if discrete_actions:
        assert np.array_equal(action["agent_0"], np.array([1])), action["agent_0"]
    else:
        assert np.array_equal(action["agent_0"], np.array([[0, 1]])), action["agent_0"]


@pytest.mark.parametrize(
    "observation_spaces", [generate_multi_agent_box_spaces(2, (6,))]
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_box_spaces(2, (6,)),
        generate_multi_agent_discrete_spaces(2, 6),
    ],
)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_get_action_vectorized_agent_masking(
    training, observation_spaces, action_spaces, device, compile_mode
):
    num_envs = 6
    agent_ids = ["agent_0", "other_agent_0"]
    state = {
        agent: np.array(
            [np.random.randn(*observation_spaces[0].shape) for _ in range(num_envs)]
        )
        for agent in agent_ids
    }
    discrete_actions = all(
        isinstance(space, spaces.Discrete) for space in action_spaces
    )
    if discrete_actions:
        env_defined_action = np.array(
            [
                np.random.randint(0, observation_spaces[0].shape[0] + 1)
                for _ in range(num_envs)
            ]
        )
    else:
        env_defined_action = np.array(
            [np.random.randn(*observation_spaces[0].shape) for _ in range(num_envs)]
        )
    nan_array = np.zeros(env_defined_action.shape)
    nan_array[:] = np.nan
    info = {
        "agent_0": {"env_defined_actions": env_defined_action},
        "other_agent_0": {"env_defined_actions": nan_array},
    }
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )
    matd3.set_training_mode(training)
    action, _ = matd3.get_action(state, infos=info)
    if discrete_actions:
        assert np.array_equal(
            action["agent_0"].squeeze(), info["agent_0"]["env_defined_actions"]
        ), action["agent_0"]
    else:
        assert np.isclose(
            action["agent_0"], info["agent_0"]["env_defined_actions"]
        ).all(), action["agent_0"]


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize(
    "observation_spaces, action_spaces",
    [
        generate_multi_agent_box_spaces(2, (6,)),
        generate_multi_agent_discrete_spaces(2, 4),
    ],
)
def test_matd3_get_action_action_masking_exception(
    training, observation_spaces, action_spaces, device
):
    agent_ids = ["agent_0", "other_agent_0"]
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
        agent_ids=agent_ids,
        device=device,
    )
    with pytest.raises(AssertionError):
        matd3.set_training_mode(training)
        _, raw_action = matd3.get_action(state)


@pytest.mark.parametrize("training", [False, True])
def test_matd3_get_action_action_masking(training, device):
    observation_spaces = generate_multi_agent_box_spaces(2, (6,))
    action_spaces = generate_multi_agent_discrete_spaces(2, 4)
    agent_ids = ["agent_0", "other_agent_0"]
    state = {
        agent: np.random.randn(*observation_spaces[idx].shape)
        for idx, agent in enumerate(agent_ids)
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
        agent_ids=agent_ids,
        device=device,
    )
    matd3.set_training_mode(training)
    action, _ = matd3.get_action(state, info)
    assert all(i in [1, 3] for i in action.values())


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (6,)),
        generate_multi_agent_discrete_spaces(2, 6),
        generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_box_spaces(2, (2,)),
        generate_multi_agent_discrete_spaces(2, 2),
    ],
)
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("agent_ids", [["agent_0", "other_agent_0"]])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_learns_from_experiences(
    observation_spaces,
    experiences,
    batch_size,
    action_spaces,
    agent_ids,
    device,
    compile_mode,
):
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    agent_ids = ["agent_0", "other_agent_0"]
    policy_freq = 2
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        policy_freq=policy_freq,
        torch_compiler=compile_mode,
    )
    actors_pre_learn_sd = {
        agent_id: copy.deepcopy(actor.state_dict())
        for agent_id, actor in matd3.actors.items()
    }
    critics_1_pre_learn_sd = {
        agent_id: str(copy.deepcopy(critic_1.state_dict()))
        for agent_id, critic_1 in matd3.critics_1.items()
    }
    critics_2_pre_learn_sd = {
        agent_id: str(copy.deepcopy(critic_2.state_dict()))
        for agent_id, critic_2 in matd3.critics_2.items()
    }

    for _ in range(4 * policy_freq):
        matd3.scores.append(0)
        loss = matd3.learn(experiences)

    assert isinstance(loss, dict)

    for agent_id in matd3.agent_ids:
        assert loss[agent_id][-1] >= 0.0

    for agent_id, old_actor_target in matd3.actor_targets.items():
        updated_actor_target = matd3.actor_targets[agent_id]
        assert old_actor_target == updated_actor_target

    for agent_id, old_actor_state_dict in actors_pre_learn_sd.items():
        updated_actor = matd3.actors[agent_id]
        assert old_actor_state_dict != str(updated_actor.state_dict())

    for agent_id, old_critic_target in matd3.critic_targets_1.items():
        updated_critic_target = matd3.critic_targets_1[agent_id]
        assert old_critic_target == updated_critic_target

    for agent_id, old_critic_state_dict in critics_1_pre_learn_sd.items():
        updated_critic = matd3.critics_1[agent_id]
        assert old_critic_state_dict != str(updated_critic.state_dict())

    for agent_id, old_critic_target in matd3.critic_targets_2.items():
        updated_critic_target = matd3.critic_targets_2[agent_id]
        assert old_critic_target == updated_critic_target

    for agent_id, old_critic_state_dict in critics_2_pre_learn_sd.items():
        updated_critic = matd3.critics_2[agent_id]
        assert old_critic_state_dict != str(updated_critic.state_dict())


def no_sync(self):
    class DummyNoSync:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass  # Add cleanup or handling if needed

    return DummyNoSync()


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (6,)),
        generate_multi_agent_discrete_spaces(2, 6),
        generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_box_spaces(2, (2,)),
        generate_multi_agent_discrete_spaces(2, 2),
    ],
)
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("agent_ids", [["agent_0", "other_agent_0"]])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_learns_from_experiences_distributed(
    observation_spaces,
    accelerated_experiences,
    batch_size,
    action_spaces,
    agent_ids,
    compile_mode,
):
    accelerator = Accelerator(device_placement=False)
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    agent_ids = ["agent_0", "other_agent_0"]
    policy_freq = 2
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        accelerator=accelerator,
        policy_freq=policy_freq,
        torch_compiler=compile_mode,
    )

    for agent_id in matd3.agent_ids:
        actor = matd3.actors[agent_id]
        critic_1 = matd3.critics_1[agent_id]
        critic_2 = matd3.critics_2[agent_id]
        actor_target = matd3.actor_targets[agent_id]
        critic_target_1 = matd3.critic_targets_1[agent_id]
        critic_target_2 = matd3.critic_targets_2[agent_id]
        actor.no_sync = no_sync.__get__(actor)
        critic_1.no_sync = no_sync.__get__(critic_1)
        critic_2.no_sync = no_sync.__get__(critic_2)
        actor_target.no_sync = no_sync.__get__(actor_target)
        critic_target_1.no_sync = no_sync.__get__(critic_target_1)
        critic_target_2.no_sync = no_sync.__get__(critic_target_2)

    actors_pre_learn_sd = {
        agent_id: copy.deepcopy(actor.state_dict())
        for agent_id, actor in matd3.actors.items()
    }
    critics_1_pre_learn_sd = {
        agent_id: str(copy.deepcopy(critic_1.state_dict()))
        for agent_id, critic_1 in matd3.critics_1.items()
    }
    critics_2_pre_learn_sd = {
        agent_id: str(copy.deepcopy(critic_2.state_dict()))
        for agent_id, critic_2 in matd3.critics_2.items()
    }

    for _ in range(4 * policy_freq):
        matd3.scores.append(0)
        loss = matd3.learn(accelerated_experiences)

    assert isinstance(loss, dict)
    for agent_id in matd3.agent_ids:
        assert loss[agent_id][-1] >= 0.0

    for agent_id, old_actor_sd in actors_pre_learn_sd.items():
        updated_actor = matd3.actors[agent_id]
        assert old_actor_sd != str(updated_actor.state_dict())

    for agent_id, old_critic_1_sd in critics_1_pre_learn_sd.items():
        updated_critic_1 = matd3.critics_1[agent_id]
        assert old_critic_1_sd != str(updated_critic_1.state_dict())

    for agent_id, old_critic_2_sd in critics_2_pre_learn_sd.items():
        updated_critic_2 = matd3.critics_2[agent_id]
        assert old_critic_2_sd != str(updated_critic_2.state_dict())

    for agent_id, old_actor_target in matd3.actor_targets.items():
        updated_actor_target = matd3.actor_targets[agent_id]
        assert old_actor_target == updated_actor_target

    for agent_id, old_critic_target_1 in matd3.critic_targets_1.items():
        updated_critic_target_1 = matd3.critic_targets_1[agent_id]
        assert old_critic_target_1 == updated_critic_target_1

    for agent_id, old_critic_target_2 in matd3.critic_targets_2.items():
        updated_critic_target_2 = matd3.critic_targets_2[agent_id]
        assert old_critic_target_2 == updated_critic_target_2


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_matd3_soft_update(device, compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(2, (6,))
    action_spaces = generate_multi_agent_box_spaces(2, (2,))
    accelerator = None

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids=["agent_0", "other_agent_0"],
        accelerator=accelerator,
        device=device,
        torch_compiler=compile_mode,
    )

    for agent_id in matd3.agent_ids:
        actor = matd3.actors[agent_id]
        actor_target = matd3.actor_targets[agent_id]
        critic_1 = matd3.critics_1[agent_id]
        critic_target_1 = matd3.critic_targets_1[agent_id]
        critic_2 = matd3.critics_2[agent_id]
        critic_target_2 = matd3.critic_targets_2[agent_id]

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


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (6,)),
        generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
    ],
)
@pytest.mark.parametrize("sum_score", [True, False])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("vectorized", [True, False])
def test_matd3_algorithm_test_loop(
    observation_spaces, device, compile_mode, sum_score, vectorized
):
    action_spaces = generate_multi_agent_discrete_spaces(2, 2)
    accelerator = None

    # Define environment and algorithm
    if vectorized:
        env = make_multi_agent_vect_envs(
            DummyMultiEnv,
            2,
            **dict(
                observation_spaces=observation_spaces[0], action_spaces=action_spaces
            )
        )
    else:
        env = DummyMultiEnv(observation_spaces[0], action_spaces)

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids=["agent_0", "other_agent_0"],
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


@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (4,)),
        generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
        generate_multi_agent_discrete_spaces(2, 2),
        gen_multi_agent_dict_or_tuple_spaces(2, 2, 2, dict_space=True),
        gen_multi_agent_dict_or_tuple_spaces(2, 2, 2, dict_space=False),
    ],
)
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("wrap", [True, False])
def test_matd3_clone_returns_identical_agent(
    accelerator_flag, wrap, compile_mode, observation_spaces
):
    # Clones the agent and returns an identical copy.
    action_spaces = generate_multi_agent_box_spaces(2, (2,), low=-1, high=1)
    agent_ids = ["agent_0", "other_agent_0"]
    expl_noise = 0.1
    index = 0
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
    accelerator = Accelerator(device_placement=False) if accelerator_flag else None
    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids,
        expl_noise=expl_noise,
        index=index,
        policy_freq=policy_freq,
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
    assert clone_agent.n_agents == matd3.n_agents
    assert clone_agent.agent_ids == matd3.agent_ids
    assert all(
        torch.equal(clone_expl_noise, expl_noise)
        for clone_expl_noise, expl_noise in zip(
            clone_agent.expl_noise, matd3.expl_noise
        )
    )
    assert clone_agent.index == matd3.index
    assert clone_agent.batch_size == matd3.batch_size
    assert clone_agent.lr_actor == matd3.lr_actor
    assert clone_agent.lr_critic == matd3.lr_critic
    assert clone_agent.learn_step == matd3.learn_step
    assert clone_agent.gamma == matd3.gamma
    assert clone_agent.tau == matd3.tau
    assert clone_agent.device == matd3.device
    assert clone_agent.accelerator == matd3.accelerator
    assert clone_agent.torch_compiler == matd3.torch_compiler

    for agent_id in clone_agent.agent_ids:
        clone_actor = clone_agent.actors[agent_id]
        actor = matd3.actors[agent_id]
        assert str(clone_actor.state_dict()) == str(actor.state_dict())

        clone_actor_target = clone_agent.actor_targets[agent_id]
        actor_target = matd3.actor_targets[agent_id]
        assert str(clone_actor_target.state_dict()) == str(actor_target.state_dict())

        clone_critic_1 = clone_agent.critics_1[agent_id]
        critic_1 = matd3.critics_1[agent_id]
        assert str(clone_critic_1.state_dict()) == str(critic_1.state_dict())

        clone_critic_target_1 = clone_agent.critic_targets_1[agent_id]
        critic_target_1 = matd3.critic_targets_1[agent_id]
        assert str(clone_critic_target_1.state_dict()) == str(
            critic_target_1.state_dict()
        )

        clone_critic_2 = clone_agent.critics_2[agent_id]
        critic_2 = matd3.critics_2[agent_id]
        assert str(clone_critic_2.state_dict()) == str(critic_2.state_dict())

        clone_critic_target_2 = clone_agent.critic_targets_2[agent_id]
        critic_target_2 = matd3.critic_targets_2[agent_id]
        assert str(clone_critic_target_2.state_dict()) == str(
            critic_target_2.state_dict()
        )


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_clone_new_index(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(2, (4,))
    action_spaces = generate_multi_agent_box_spaces(2, (2,))
    agent_ids = ["agent_0", "other_agent_0"]

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids,
        torch_compiler=compile_mode,
    )
    clone_agent = matd3.clone(index=100)

    assert clone_agent.index == 100


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_clone_after_learning(compile_mode):
    observation_spaces = generate_multi_agent_box_spaces(2, (4,))
    action_spaces = generate_multi_agent_box_spaces(2, (2,))
    agent_ids = ["agent_0", "other_agent_0"]
    batch_size = 8

    matd3 = MATD3(
        observation_spaces,
        action_spaces,
        agent_ids,
        batch_size=batch_size,
        torch_compiler=compile_mode,
    )

    states = {
        agent_id: torch.randn(batch_size, observation_spaces[idx].shape[0])
        for idx, agent_id in enumerate(agent_ids)
    }
    actions = {
        agent_id: torch.randn(batch_size, action_spaces[idx].shape[0])
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
    assert clone_agent.n_agents == matd3.n_agents
    assert clone_agent.agent_ids == matd3.agent_ids
    assert all(
        torch.equal(clone_expl_noise, expl_noise)
        for clone_expl_noise, expl_noise in zip(
            clone_agent.expl_noise, matd3.expl_noise
        )
    )
    assert clone_agent.index == matd3.index
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

    for agent_id in clone_agent.agent_ids:
        clone_actor = clone_agent.actors[agent_id]
        actor = matd3.actors[agent_id]
        assert str(clone_actor.state_dict()) == str(actor.state_dict())

        clone_actor_target = clone_agent.actor_targets[agent_id]
        actor_target = matd3.actor_targets[agent_id]
        assert str(clone_actor_target.state_dict()) == str(actor_target.state_dict())

        clone_critic_1 = clone_agent.critics_1[agent_id]
        critic_1 = matd3.critics_1[agent_id]
        assert str(clone_critic_1.state_dict()) == str(critic_1.state_dict())

        clone_critic_target_1 = clone_agent.critic_targets_1[agent_id]
        critic_target_1 = matd3.critic_targets_1[agent_id]
        assert str(clone_critic_target_1.state_dict()) == str(
            critic_target_1.state_dict()
        )

        clone_critic_2 = clone_agent.critics_2[agent_id]
        critic_2 = matd3.critics_2[agent_id]
        assert str(clone_critic_2.state_dict()) == str(critic_2.state_dict())

        clone_critic_target_2 = clone_agent.critic_targets_2[agent_id]
        critic_target_2 = matd3.critic_targets_2[agent_id]
        assert str(clone_critic_target_2.state_dict()) == str(
            critic_target_2.state_dict()
        )


@pytest.mark.parametrize(
    "observation_spaces, encoder_cls",
    [
        (generate_multi_agent_box_spaces(1, (6,)), EvolvableMLP),
        (
            generate_multi_agent_box_spaces(1, (3, 32, 32), low=0, high=255),
            EvolvableCNN,
        ),
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
def test_matd3_save_load_checkpoint_correct_data_and_format(
    tmpdir, device, accelerator, compile_mode, observation_spaces, encoder_cls
):
    # Initialize the matd3 agent
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
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
    assert "actors_init_dict" in checkpoint["network_info"]["modules"]
    assert "actors_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_targets_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_targets_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critics_1_init_dict" in checkpoint["network_info"]["modules"]
    assert "critics_1_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_targets_1_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_targets_1_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_2_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critics_2_init_dict" in checkpoint["network_info"]["modules"]
    assert "critics_2_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_targets_2_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_targets_2_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_2_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "policy_freq" in checkpoint
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
        observation_spaces=observation_spaces,
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        agent_ids=["agent_0"],
        torch_compiler=compile_mode,
        device=device,
        accelerator=accelerator,
    )
    loaded_matd3.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    if compile_mode is not None and accelerator is None:
        for agent_id in loaded_matd3.agent_ids:
            actor = loaded_matd3.actors[agent_id]
            actor_target = loaded_matd3.actor_targets[agent_id]
            assert isinstance(actor, OptimizedModule)
            assert isinstance(actor_target, OptimizedModule)

            critic = loaded_matd3.critics_1[agent_id]
            critic_target = loaded_matd3.critic_targets_1[agent_id]
            assert isinstance(critic, OptimizedModule)
            assert isinstance(critic_target, OptimizedModule)

            critic = loaded_matd3.critics_2[agent_id]
            critic_target = loaded_matd3.critic_targets_2[agent_id]
            assert isinstance(critic, OptimizedModule)
            assert isinstance(critic_target, OptimizedModule)

    else:
        for agent_id in loaded_matd3.agent_ids:
            actor = loaded_matd3.actors[agent_id]
            actor_target = loaded_matd3.actor_targets[agent_id]
            assert isinstance(actor.encoder, encoder_cls)
            assert isinstance(actor_target.encoder, encoder_cls)

            critic = loaded_matd3.critics_1[agent_id]
            critic_target = loaded_matd3.critic_targets_1[agent_id]
            assert isinstance(critic.encoder, EvolvableMultiInput)
            assert isinstance(critic_target.encoder, EvolvableMultiInput)

            critic = loaded_matd3.critics_2[agent_id]
            critic_target = loaded_matd3.critic_targets_2[agent_id]
            assert isinstance(critic.encoder, EvolvableMultiInput)
            assert isinstance(critic_target.encoder, EvolvableMultiInput)

    assert matd3.lr_actor == 0.001
    assert matd3.lr_critic == 0.01

    for agent_id in loaded_matd3.agent_ids:
        actor = loaded_matd3.actors[agent_id]
        actor_target = loaded_matd3.actor_targets[agent_id]
        assert str(actor.state_dict()) == str(actor_target.state_dict())

        critic_1 = loaded_matd3.critics_1[agent_id]
        critic_target_1 = loaded_matd3.critic_targets_1[agent_id]
        assert str(critic_1.state_dict()) == str(critic_target_1.state_dict())

        critic_2 = loaded_matd3.critics_2[agent_id]
        critic_target_2 = loaded_matd3.critic_targets_2[agent_id]
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


# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize(
    "observation_spaces", [generate_multi_agent_box_spaces(1, (6,))]
)
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(1, 2)])
def test_matd3_save_load_checkpoint_correct_data_and_format_make_evo(
    tmpdir,
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    device,
    compile_mode,
    accelerator,
):
    evo_actors = ModuleDict(
        {
            "agent_0": MakeEvolvable(
                network=mlp_actor, input_tensor=torch.randn(1, 6), device=device
            )
        }
    )
    evo_critics_1 = ModuleDict(
        {
            "agent_0": MakeEvolvable(
                network=mlp_critic, input_tensor=torch.randn(1, 8), device=device
            )
        }
    )
    evo_critics_2 = ModuleDict(
        {
            "agent_0": MakeEvolvable(
                network=mlp_critic, input_tensor=torch.randn(1, 8), device=device
            )
        }
    )
    evo_critics = [evo_critics_1, evo_critics_2]
    matd3 = MATD3(
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
    matd3.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actors_init_dict" in checkpoint["network_info"]["modules"]
    assert "actors_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_targets_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_targets_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critics_1_init_dict" in checkpoint["network_info"]["modules"]
    assert "critics_1_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_targets_1_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_targets_1_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_2_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critics_2_init_dict" in checkpoint["network_info"]["modules"]
    assert "critics_2_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_targets_2_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_targets_2_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_2_optimizers_state_dict" in checkpoint["network_info"]["optimizers"]
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
        observation_spaces=generate_multi_agent_box_spaces(
            1, (3, 32, 32), low=0, high=255
        ),
        action_spaces=generate_multi_agent_discrete_spaces(1, 2),
        agent_ids=["agent_0"],
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
    )
    loaded_matd3.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    expected_module_class = (
        OptimizedModule
        if compile_mode is not None and accelerator is None
        else MakeEvolvable
    )
    assert all(
        isinstance(actor, expected_module_class) for actor in loaded_matd3.actors
    )
    assert all(
        isinstance(actor_target, expected_module_class)
        for actor_target in loaded_matd3.actor_targets
    )
    assert all(
        isinstance(critic, expected_module_class) for critic in loaded_matd3.critics_1
    )
    assert all(
        isinstance(critic_target, expected_module_class)
        for critic_target in loaded_matd3.critic_targets_1
    )
    assert all(
        isinstance(critic, expected_module_class) for critic in loaded_matd3.critics_2
    )
    assert all(
        isinstance(critic_target, expected_module_class)
        for critic_target in loaded_matd3.critic_targets_2
    )
    assert matd3.lr_actor == 0.001
    assert matd3.lr_critic == 0.01

    for agent_id in loaded_matd3.agent_ids:
        actor = loaded_matd3.actors[agent_id]
        actor_target = loaded_matd3.actor_targets[agent_id]
        assert str(actor.state_dict()) == str(actor_target.state_dict())

        critic_1 = loaded_matd3.critics_1[agent_id]
        critic_target_1 = loaded_matd3.critic_targets_1[agent_id]
        assert str(critic_1.state_dict()) == str(critic_target_1.state_dict())

        critic_2 = loaded_matd3.critics_2[agent_id]
        critic_target_2 = loaded_matd3.critic_targets_2[agent_id]
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
        agent_ids=["agent_0", "other_agent_0"],
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    matd3.unwrap_models()
    for agent_id in matd3.agent_ids:
        actor = matd3.actors[agent_id]
        actor_target = matd3.actor_targets[agent_id]
        critic_1 = matd3.critics_1[agent_id]
        critic_target_1 = matd3.critic_targets_1[agent_id]
        critic_2 = matd3.critics_2[agent_id]
        critic_target_2 = matd3.critic_targets_2[agent_id]
        assert isinstance(actor, nn.Module)
        assert isinstance(actor_target, nn.Module)
        assert isinstance(critic_1, nn.Module)
        assert isinstance(critic_target_1, nn.Module)
        assert isinstance(critic_2, nn.Module)
        assert isinstance(critic_target_2, nn.Module)


# The saved checkpoint file contains the correct data and format.
@pytest.mark.parametrize(
    "observation_spaces, encoder_cls",
    [
        (generate_multi_agent_box_spaces(2, (6,)), EvolvableMLP),
        (
            generate_multi_agent_box_spaces(2, (3, 32, 32), low=0, high=255),
            EvolvableCNN,
        ),
        (
            gen_multi_agent_dict_or_tuple_spaces(2, 2, 2, dict_space=True),
            EvolvableMultiInput,
        ),
        # (
        #     gen_multi_agent_dict_or_tuple_spaces(2, 2, 2, dict_space=False),
        #     EvolvableMultiInput,
        # ),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_load_from_pretrained(
    device, accelerator, compile_mode, tmpdir, observation_spaces, encoder_cls
):
    # Initialize the matd3 agent
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=generate_multi_agent_discrete_spaces(2, 2),
        agent_ids=["agent_0", "other_agent_0"],
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
    assert new_matd3.n_agents == matd3.n_agents
    assert new_matd3.agent_ids == matd3.agent_ids
    assert new_matd3.lr_actor == matd3.lr_actor
    assert new_matd3.lr_critic == matd3.lr_critic

    for agent_id in new_matd3.agent_ids:
        new_actor = new_matd3.actors[agent_id]
        new_actor_target = new_matd3.actor_targets[agent_id]
        new_critic_1 = new_matd3.critics_1[agent_id]
        new_critic_target_1 = new_matd3.critic_targets_1[agent_id]
        new_critic_2 = new_matd3.critics_2[agent_id]
        new_critic_target_2 = new_matd3.critic_targets_2[agent_id]
        if compile_mode is not None and accelerator is None:
            assert isinstance(new_actor, OptimizedModule)
            assert isinstance(new_actor_target, OptimizedModule)
            assert isinstance(new_critic_1, OptimizedModule)
            assert isinstance(new_critic_target_1, OptimizedModule)
            assert isinstance(new_critic_2, OptimizedModule)
            assert isinstance(new_critic_target_2, OptimizedModule)
        else:
            assert isinstance(new_actor.encoder, encoder_cls)
            assert isinstance(new_actor_target.encoder, encoder_cls)
            assert isinstance(new_critic_1.encoder, EvolvableMultiInput)
            assert isinstance(new_critic_target_1.encoder, EvolvableMultiInput)
            assert isinstance(new_critic_2.encoder, EvolvableMultiInput)
            assert isinstance(new_critic_target_2.encoder, EvolvableMultiInput)

        new_actor_sd = str(new_actor.state_dict())
        new_actor_target_sd = str(new_actor_target.state_dict())
        new_critic_1_sd = str(new_critic_1.state_dict())
        new_critic_target_1_sd = str(new_critic_target_1.state_dict())
        new_critic_2_sd = str(new_critic_2.state_dict())
        new_critic_target_2_sd = str(new_critic_target_2.state_dict())

        assert new_actor_sd == str(matd3.actors[agent_id].state_dict())
        assert new_actor_target_sd == str(matd3.actor_targets[agent_id].state_dict())
        assert new_critic_1_sd == str(matd3.critics_1[agent_id].state_dict())
        assert new_critic_target_1_sd == str(
            matd3.critic_targets_1[agent_id].state_dict()
        )
        assert new_critic_2_sd == str(matd3.critics_2[agent_id].state_dict())
        assert new_critic_target_2_sd == str(
            matd3.critic_targets_2[agent_id].state_dict()
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


# The saved checkpoint file contains the correct data and format.
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize("action_spaces", [generate_multi_agent_discrete_spaces(2, 2)])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize(
    "observation_spaces, arch, input_tensor, critic_input_tensor, secondary_input_tensor",
    [
        (
            generate_multi_agent_box_spaces(2, (4,)),
            "mlp",
            torch.randn(1, 4),
            torch.randn(1, 6),
            None,
        ),
        (
            generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255),
            "cnn",
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 2),
        ),
        (
            generate_multi_agent_box_spaces(2, (4,)),
            "mlp",
            torch.randn(1, 4),
            torch.randn(1, 6),
            None,
        ),
        (
            generate_multi_agent_box_spaces(2, (4, 210, 160), low=0, high=255),
            "cnn",
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 4, 2, 210, 160),
            torch.randn(1, 2),
        ),
    ],
)
def test_load_from_pretrained_make_evo(
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
    if arch == "mlp":
        actor_network = mlp_actor
        critic_network = mlp_critic
    elif arch == "cnn":
        actor_network = cnn_actor
        critic_network = cnn_critic

    agent_ids = ["agent_0", "other_agent_0"]
    actor_network = ModuleDict(
        {agent_id: MakeEvolvable(actor_network, input_tensor) for agent_id in agent_ids}
    )
    critic_network = ModuleDict(
        {
            agent_id: MakeEvolvable(
                critic_network,
                critic_input_tensor,
                secondary_input_tensor=secondary_input_tensor,
            )
            for agent_id in agent_ids
        }
    )

    # Initialize the matd3 agent
    matd3 = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        actor_networks=actor_network,
        critic_networks=[critic_network, copy.deepcopy(critic_network)],
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
    assert new_matd3.n_agents == matd3.n_agents
    assert new_matd3.agent_ids == matd3.agent_ids
    assert new_matd3.lr_actor == matd3.lr_actor
    assert new_matd3.lr_critic == matd3.lr_critic

    for agent_id in new_matd3.agent_ids:
        new_actor = new_matd3.actors[agent_id]
        new_actor_target = new_matd3.actor_targets[agent_id]
        new_critic_1 = new_matd3.critics_1[agent_id]
        new_critic_target_1 = new_matd3.critic_targets_1[agent_id]
        new_critic_2 = new_matd3.critics_2[agent_id]
        new_critic_target_2 = new_matd3.critic_targets_2[agent_id]
        actor = matd3.actors[agent_id]
        actor_target = matd3.actor_targets[agent_id]
        critic_1 = matd3.critics_1[agent_id]
        critic_target_1 = matd3.critic_targets_1[agent_id]
        critic_2 = matd3.critics_2[agent_id]
        critic_target_2 = matd3.critic_targets_2[agent_id]
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
