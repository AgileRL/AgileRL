from gymnasium import spaces
from agilerl.modules.base import EvolvableModule

class EvolvableNetwork(EvolvableModule):
    """Base class for evolvable networks i.e. modules that are configured in 
    a specific way for a reinforcement learning algorithm.
    
    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: spaces.Space
    """
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super(EvolvableNetwork, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
    