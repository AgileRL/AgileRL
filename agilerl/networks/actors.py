from typing import Union, Optional, Tuple, Type, Dict, Any
from dataclasses import asdict
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Distribution, Categorical, Normal, Independent

from agilerl.typing import TorchObsType, ConfigType
from agilerl.configs import MlpNetConfig, NetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.modules.mlp import EvolvableMLP
from agilerl.modules.base import EvolvableModule

def get_distribution_cls(action_space: spaces.Space) -> Type[Distribution]:
    if isinstance(action_space, spaces.Discrete):
        return Categorical
    elif isinstance(action_space, spaces.Box):
        return Normal
    else:
        raise NotImplementedError(f"Action space {action_space} not supported.")
    
class DistributionWrapper(EvolvableModule):
    """Wrapper for a distribution over an action space. It provides methods to sample
    actions and compute log probabilities.

    :param action_space: Action space of the environment.
    :type action_space: spaces.Space
    :param network: Network that outputs the logits of the distribution.
    :type network: EvolvableModule
    """
    def __init__(self, action_space: spaces.Space, network: EvolvableModule):
        self.action_space = action_space
        self.action_net = network
    

class Actor(EvolvableNetwork):
    """Actor network for policy-gradient algorithms. Given an observation space, it outputs 
    a distribution over the action space.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: DiscreteSpace
    :param encoder_config: Configuration of the encoder network.
    :type encoder_config: ConfigType
    :param head_config: Configuration of the network MLP head.
    :type head_config: Optional[ConfigType]
    :param n_agents: Number of agents in the environment. Defaults to None, which corresponds to 
        single-agent environments.
    :type n_agents: Optional[int]
    :param latent_dim: Dimension of the latent space representation.
    :type latent_dim: int
    :param device: Device to use for the network.
    :type device: str
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            encoder_config: Optional[ConfigType] = None,
            head_config: Optional[ConfigType] = None,
            min_latent_dim: int = 8,
            max_latent_dim: int = 128,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: str = "cpu"
            ):
        
        super().__init__(
            observation_space, 
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim, 
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
            device=device
            )
        

        