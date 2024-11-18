from typing import Optional, Any, Union, Tuple, Dict
from abc import ABC, abstractmethod
from gymnasium import spaces
from accelerate import Accelerator

from agilerl.networks.base import EvolvableModule
from agilerl.utils.algo_utils import obs_to_tensor
from agilerl.typing import NumpyObsType, TorchObsType

class EvolvableAlgorithm(ABC):
    """Base object for all algorithms in the AgileRL framework. Provides an abstraction that allows for 
    seamless mutations to the underlying hyperparameters and network architectures.
    
    :param observation_space: The observation space of the environment.
    :type observation_space: spaces.Space.
    :param action_space: The action space of the environment.
    :type action_space: spaces.Space.
    :param index: The index of the individual.
    :type index: int.
    :param learn_step: Learning frequency, defaults to 2048
    :type learn_step: int, optional
    :param device: Device to run the algorithm on, defaults to "cpu"
    :type device: str, optional
    :param accelerator: Accelerator object for distributed computing, defaults to None
    :type accelerator: Optional[Accelerator], optional
    :param name: Name of the algorithm, defaults to the class name
    :type name: Optional[str], optional
    """
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            index: int,
            net_config: Dict[str, Any],
            learn_step: int = 2048,
            device: str = "cpu",
            accelerator: Optional[Accelerator] = None,
            name: Optional[str] = None,
            ) -> None:

        assert isinstance(
            observation_space, spaces.Space
        ), "Observation space must be an instance of gym.spaces.Space."
        assert isinstance(
            action_space, spaces.Space
        ), "Action space must be an instance of gym.spaces.Space."

        self._net_config = net_config
        self._index = index
        self._device = device
        self._accelerator = accelerator
        self._learn_step = learn_step
        self._name = name if name is not None else self.__class__.__name__
        self._mut = None
        self._observation_space = observation_space
        self._action_space = action_space

        self.scores = []
        self.fitness = []
        self.steps = [0]

    @property
    def observation_space(self) -> spaces.Space:
        """Returns the observation space of the environment."""
        return self._observation_space
    
    @property
    def action_space(self) -> spaces.Space:
        """Returns the action space of the environment."""
        return self._action_space
    
    @property
    def index(self) -> int:
        """Returns the index of the algorithm."""
        return self._index
    
    @property
    def device(self) -> str:
        """Returns the device of the algorithm."""
        return self._device
    
    @property
    def accelerator(self) -> Optional[Accelerator]:
        """Returns the accelerator of the algorithm."""
        return self._accelerator
    
    @property
    def learn_step(self) -> int:
        """Returns the learn step of the algorithm."""
        return self._learn_step
    
    @property
    def algo(self) -> str:
        """Returns the name of the algorithm."""
        return self._name
    
    @property
    def net_config(self) -> Dict[str, Any]:
        """Returns the network configuration of the algorithm."""
        return self._net_config
    
    @property
    def mut(self) -> Any:
        """Returns the mutation object of the algorithm."""
        return self._mut
    
    @mut.setter
    def mut(self, value: Optional[str]) -> None:
        """Sets the mutation object of the algorithm."""
        self._mut = value

    # TODO: This is a bit of a temporary hack until we fully refactor the framework
    @property
    def state_dim(self) -> Union[int, Tuple[int, ...]]:
        """Returns the dimension of the state space."""
        if isinstance(self.observation_space, spaces.Dict):
            raise AttributeError("Can't access state_dim for Dict observation spaces.")
        elif isinstance(self.observation_space, spaces.Discrete):
            return self.observation_space.n
        elif isinstance(self.observation_space, spaces.Box):
            return self.observation_space.shape
        else:
            raise AttributeError(f"What do we do with {type(self.observation_space)} spaces?")

    @property
    def action_dim(self) -> int:
        """Returns the dimension of the action space."""
        if isinstance(self.action_space, spaces.Dict):
            raise AttributeError("Can't access action_dim for Dict action spaces.")
        elif isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, spaces.Box):
            return self.action_space.shape[0]
        else:
            raise AttributeError(f"What do we do with {type(self.action_space)} spaces?")
    
    @abstractmethod
    def learn(self, *args, **kwargs) -> None:
        """Abstract method for learning the algorithm."""
        ...

    @abstractmethod
    def get_action(self, *args, **kwargs) -> Any:
        """Abstract method for getting an action from the algorithm."""
        ...

    def obs_to_tensor(self, observation: NumpyObsType) -> TorchObsType:
        """Prepares state for forward pass through neural network.

        :param state: Observation of environment
        :type state: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed state
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]]
        """
        device = self.device if self.accelerator is None else self.accelerator.device
        return obs_to_tensor(observation, device)

    def networks(self) :
        """Returns the evolvable networks in the algorithm."""
        # Inspect the class to find all attributes that are EvolvableModule's
        return [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), EvolvableModule)]