from typing import Union, Optional, Dict, Any
from dataclasses import asdict
import torch
from gymnasium import spaces

from agilerl.typing import TorchObsType
from agilerl.configs import MlpNetConfig, CnnNetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.modules.base import EvolvableModule
from agilerl.modules.mlp import EvolvableMLP

DiscreteSpace = Union[spaces.Discrete, spaces.MultiDiscrete]

class QNetwork(EvolvableNetwork):
    """Q Networks correspond to state-action value functions in deep reinforcement learning. From any given 
    state, they predict the value of each action that can be taken from that state.
    
    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: DiscreteSpace
    :param net_config: Configuration of the Q network.
    :type net_config: Dict[str, Any]
    :param n_agents: Number of agents in the environment. Defaults to None, which corresponds to 
        single-agent environments.
    :type n_agents: Optional[int]
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: DiscreteSpace,
            encoder_config: Dict[str, Any],
            head_config: Optional[Dict[str, Any]] = None,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: str = "cpu",
            ):
        # For multi-agent settings, we use a depth corresponding to the number of agents
        # for the first layer of CNN-based networks
        if n_agents is not None and "kernel_size" in encoder_config.keys():
            encoder_config = EvolvableNetwork.modify_multi_agent_config(
                encoder_config, n_agents=n_agents, n_agents_depth=True
                )
        
        super().__init__(
            observation_space, action_space, encoder_config, n_agents, latent_dim, device
            )

        assert (
            isinstance(action_space, spaces.Discrete) 
            or isinstance(action_space, spaces.MultiDiscrete),
            "Action space must be either Discrete or MultiDiscrete"
        )

        if head_config is None:
            head_config = asdict(MlpNetConfig(hidden_size=[64, 64], device=self.device))

        self.num_actions = spaces.flatdim(action_space)
        self.net_head = self.build_network_head(head_config)
    
    def build_network_head(self, net_config: Dict[str, Any]) -> EvolvableModule:
        """Builds the head of the network based on the passed configuration.
        
        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]
        
        :return: Network head.
        :rtype: EvolvableModule
        """
        model = EvolvableMLP(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions,
            name="head"
        )

    def forward(self, x: TorchObsType) -> torch.Tensor:
        """Forward pass of the Q network.

        :param x: Input to the network.
        :type x: TorchObsType

        :return: Output of the network.
        :rtype: torch.Tensor
        """
        # If using rainbow we need to calculate the Q value through (maybe log) softmax
        # if self.rainbow:
        #     value: torch.Tensor = self.value_net(x)
        #     advantage: torch.Tensor = self.advantage_net(x)
        #     value = value.view(-1, 1, self.num_atoms)
        #     advantage = advantage.view(-1, self.num_outputs, self.num_atoms)
        #     x = value + advantage - advantage.mean(1, keepdim=True)
        #     if log:
        #         x = F.log_softmax(x, dim=2)
        #         return x
        #     else:
        #         x = F.softmax(x, dim=2)

        #     # Output at this point has shape -> (batch_size, actions, num_support)
        #     if q:
        #         x = torch.sum(x * self.support.expand_as(x), dim=2)