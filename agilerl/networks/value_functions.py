from typing import Union, Optional, Dict, Any
import copy
from gymnasium import spaces

from agilerl.networks.base import EvolvableNetwork

DiscreteSpace = Union[spaces.Discrete, spaces.MultiDiscrete]

class ValueFunction(EvolvableNetwork):
    """Value functions are used in reinforcement learning to estimate the value of a state. 
    Therefore, for any given observation, we predict a single scalar value that represents 
    the discounted return that can be expected from that state.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: DiscreteSpace,
            n_agents: Optional[int] = None,
            net_config: Dict[str, Any] = {},
            ):
        super().__init__(observation_space, action_space)

        assert isinstance(action_space, spaces.Discrete) or isinstance(action_space, spaces.MultiDiscrete), \
            "Action space must be either Discrete or MultiDiscrete"