import pytest
import torch
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from accelerate import Accelerator
from agilerl.algorithms.core import RLAlgorithm, MultiAgentAlgorithm
from agilerl.algorithms.core.registry import NetworkGroup
from agilerl.modules.base import EvolvableModule
from agilerl.modules.mlp import EvolvableMLP
from agilerl.algorithms.core.wrappers import OptimizerWrapper

import torch.nn as nn

# Helper classes
class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, shape=(4,))
        self.action_space = spaces.Discrete(2)
        
    def reset(self):
        return np.zeros(4), {}
        
    def step(self, action):
        return np.zeros(4), 0, False, False, {}

class MultiAgentDummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, shape=(4,))
        self.action_space = spaces.Discrete(2) 
        self.n_agents = 2
        
    def reset(self):
        return [np.zeros(4) for _ in range(self.n_agents)], {}
        
    def step(self, actions):
        return [np.zeros(4) for _ in range(self.n_agents)], [0]*self.n_agents, False, False, {}

class DummyNetwork(EvolvableMLP):
    def __init__(self, obs_dim, act_dim, device="cpu"):
        super().__init__(
            num_inputs=obs_dim,
            num_outputs=act_dim,
            hidden_size=[64, 64],
            device=device
        )

class DummyAlgorithm(RLAlgorithm):
    def __init__(self, env, device="cpu"): 
        super().__init__(
            observation_space=env.observation_space,
            action_space=env.action_space,
            index=0,
            device=device
        )
        
        # Create networks and optimizer
        self.network = DummyNetwork(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.n,
            device=device
        )
        
        # Register network group
        self.register_network_group(
            NetworkGroup(
                eval=self.network,
                shared=None,
                policy=True
            )
        )
        
        self.optimizer = OptimizerWrapper(
            optimizer_cls=Adam,
            networks=self.network, 
            optimizer_kwargs={"lr": 0.001}
        )

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.network(state).argmax().item()
        return action

    def learn(self, experiences):
        pass
        
    def test(self, env):
        return 0.0

class MultiAgentDummyAlgorithm(MultiAgentAlgorithm):
    def __init__(self, env, device="cpu"):
        super().__init__(
            observation_spaces=[env.observation_space]*env.n_agents,
            action_spaces=[env.action_space]*env.n_agents, 
            agent_ids=list(range(env.n_agents)),
            index=0,
            device=device
        )

        # Create networks for each agent
        self.networks = []
        self.optimizers = []
        for _ in range(env.n_agents):
            network = DummyNetwork(
                obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.n,
                device=device
            )
            self.networks.append(network)
            
            # Register network group for each agent
            self.register_network_group(
                NetworkGroup(
                    eval=network,
                    shared=None, 
                    policy=True,
                    multiagent=True
                )
            )
            
            self.optimizers.append(
                OptimizerWrapper(
                    optimizer_cls=Adam,
                    networks=network,
                    optimizer_kwargs={"lr": 0.001}
                )
            )

    def get_action(self, states):
        actions = []
        for i, state in enumerate(states):
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action = self.networks[i](state).argmax().item()
            actions.append(action)
        return actions

    def learn(self, experiences):
        pass
        
    def test(self, env):
        return [0.0]*self.n_agents

# Tests
def test_rl_algorithm_init():
    env = DummyEnv()
    algo = DummyAlgorithm(env)
    
    assert isinstance(algo.network, DummyNetwork)
    assert isinstance(algo.optimizer.optimizer, Adam)
    assert algo.device == "cpu"
    assert algo.index == 0

def test_rl_algorithm_wrap_models():
    env = DummyEnv()
    algo = DummyAlgorithm(env)
    accelerator = Accelerator()
    algo.accelerator = accelerator
    
    algo.wrap_models()
    assert isinstance(algo.network, nn.parallel.DistributedDataParallel)

def test_rl_algorithm_clone():
    env = DummyEnv()
    algo = DummyAlgorithm(env)
    
    clone = algo.clone(index=1)
    assert clone.index == 1
    assert id(clone.network) != id(algo.network)
    assert clone.device == algo.device

def test_multi_agent_algorithm_init():
    env = MultiAgentDummyEnv() 
    algo = MultiAgentDummyAlgorithm(env)
    
    assert len(algo.networks) == env.n_agents
    assert len(algo.optimizers) == env.n_agents
    for i in range(env.n_agents):
        assert isinstance(algo.networks[i], DummyNetwork)
        assert isinstance(algo.optimizers[i].optimizer, Adam)
    assert algo.device == "cpu"
    assert algo.index == 0

def test_multi_agent_algorithm_wrap_models():
    env = MultiAgentDummyEnv()
    algo = MultiAgentDummyAlgorithm(env)
    accelerator = Accelerator()
    algo.accelerator = accelerator
    
    algo.wrap_models()
    for network in algo.networks:
        assert isinstance(network, nn.parallel.DistributedDataParallel)

def test_multi_agent_algorithm_clone():
    env = MultiAgentDummyEnv()
    algo = MultiAgentDummyAlgorithm(env)
    
    clone = algo.clone(index=1)
    assert clone.index == 1
    assert len(clone.networks) == len(algo.networks)
    for i in range(len(algo.networks)):
        assert id(clone.networks[i]) != id(algo.networks[i])
    assert clone.device == algo.device