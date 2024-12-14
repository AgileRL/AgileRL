from typing import Union, Optional, Dict, Any
import copy
from gymnasium import spaces

from agilerl.typing import ConfigType
from agilerl.networks.base import EvolvableNetwork

class ValueFunction(EvolvableNetwork):
    """Value functions are used in reinforcement learning to estimate the expected value of a state. 
    Therefore, for any given observation, we predict a single scalar value that represents 
    the discounted return from that state.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            encoder_config: ConfigType,
            head_config: Optional[ConfigType] = None,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: str = "cpu"
            ):

        super().__init__(
            observation_space, encoder_config, n_agents=n_agents,
            latent_dim=latent_dim, device=device
            )
        
        

