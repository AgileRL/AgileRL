from typing import Optional, Union, Tuple, Iterable, TypeGuard, Any, Dict, List, Set
import inspect
from abc import ABC, abstractmethod
from gymnasium import spaces
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
import numpy as np
from numpy.typing import ArrayLike
from torch.optim import Optimizer
import torch
import torch.nn as nn
from torch._dynamo import OptimizedModule
import dill

from agilerl.typing import NumpyObsType, TorchObsType
from agilerl.networks.base import EvolvableModule
from agilerl.utils.algo_utils import (
    obs_to_tensor,
    recursive_check_module_attrs,
    remove_compile_prefix
)

EvolvableNetworkType = Union[EvolvableModule, Iterable[EvolvableModule]]
OptimizerType = Union[Optimizer, Iterable[Optimizer]]
EvolvableAttributeType = Union[EvolvableNetworkType, OptimizerType]
EvolvableNetworkDict = Dict[str, EvolvableNetworkType]
EvolvableAttributeDict = Dict[str, EvolvableAttributeType]
    
def is_module_list(obj: EvolvableAttributeType) -> TypeGuard[Iterable[EvolvableModule]]:
    """Type guard to check if an object is a list of EvolvableModule's.
    
    :param obj: The object to check.
    :type obj: EvolvableAttributeType.
    
    :return: True if the object is a list of EvolvableModule's, False otherwise.
    :rtype: bool.
    """
    return all(isinstance(inner_obj, (OptimizedModule, EvolvableModule)) for inner_obj in obj)

def is_optimizer_list(obj: EvolvableAttributeType) -> TypeGuard[Iterable[Optimizer]]:
    """Type guard to check if an object is a list of Optimizer's.
    
    :param obj: The object to check.
    :type obj: EvolvableAttributeType.
    
    :return: True if the object is a list of Optimizer's, False otherwise.
    :rtype: bool.
    """
    return all(isinstance(inner_obj, Optimizer) for inner_obj in obj)

class EvolvableAlgorithm(ABC):
    """Base object for all algorithms in the AgileRL framework. 
    
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
            index: int,
            learn_step: int = 2048,
            device: Union[str, torch.device] = "cpu",
            accelerator: Optional[Accelerator] = None,
            name: Optional[str] = None,
            ) -> None:

        assert isinstance(index, int), "Agent index must be an integer."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(device, (str, torch.device)), "Device must be a string."
        assert isinstance(accelerator, (type(None), Accelerator)), "Accelerator must be an instance of Accelerator."
        assert isinstance(name, (type(None), str)), "Name must be a string."

        self.accelerator = accelerator
        self.device = device
        self.learn_step = learn_step
        self.algo = name if name is not None else self.__class__.__name__

        self._mut = None
        self._index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        self.optim_to_modules: Dict[str, List[str]] = {}
        self.module_to_optim: Dict[str, str] = {}
        self.param_to_modules: Dict[nn.Parameter, Set[str]] = {}

    @property
    def index(self) -> int:
        """Returns the index of the algorithm."""
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        """Sets the index of the algorithm."""
        self._index = value
    
    @property
    def mut(self) -> Any:
        """Returns the mutation object of the algorithm."""
        return self._mut
    
    @mut.setter
    def mut(self, value: Optional[str]) -> None:
        """Sets the mutation object of the algorithm."""
        self._mut = value
    
    # TODO: Is it too ambitious to generalise this too?
    @abstractmethod
    def clone() -> "EvolvableAlgorithm":
        """Abstract method for cloning the algorithm."""
        raise NotImplementedError

    # TODO: Can this be generalised to all algorithms?
    @abstractmethod
    def unwrap_models(self):
        """Unwraps the models in the algorithm from the accelerator."""
        raise NotImplementedError
    
    # TODO: Can probably implement this as well but will leave for now
    @abstractmethod
    def load_checkpoint(self, path: str, device: str, accelerator: Optional[Accelerator]) -> None:
        """Loads a checkpoint of agent properties and network weights from path.

        :param path: Location to load checkpoint from
        :type path: string
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, experiences: Tuple[Iterable[ArrayLike], ...], **kwargs) -> None:
        """Abstract method for learning the algorithm."""
        raise NotImplementedError

    @abstractmethod
    def get_action(self, *args, **kwargs) -> Any:
        """Abstract method for getting an action from the algorithm."""
        raise NotImplementedError
    
    @abstractmethod
    def test(self, *args, **kwargs) -> np.ndarray:
        """Abstract method for testing the algorithm."""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "EvolvableAlgorithm":
        """Loads an algorithm from a checkpoint.

        :param path: Location to load checkpoint from
        :type path: string

        :return: An instance of the algorithm
        :rtype: EvolvableAlgorithm
        """
        raise NotImplementedError

    # def __init_subclass__(cls, **kwargs):
    #     """Ensure the mapping method runs after initialization in subclasses."""
    #     original_init = cls.__init__

    #     def wrapped_init(self: EvolvableAlgorithm, *args, **kwargs):
    #         original_init(self, *args, **kwargs)  # Call the original __init__ method
    #         self._init_evolvable_mappings()  # Automatically run the mapping method

    #     cls.__init__ = wrapped_init  # Replace the subclass's __init__ with the wrapped version
    #     super().__init_subclass__(**kwargs)

    def _init_evolvable_mappings(self):
        """Automatically map optimizers to the modules they optimize."""
        self._identify_param_to_modules()
        for attr_name, attr in self.evolvable_attributes().items():
            if isinstance(attr, (AcceleratedOptimizer, Optimizer)):
                # Extract modules from the optimizer's parameter groups
                modules = self._extract_modules_from_optimizer(attr)
                self.optim_to_modules[attr] = modules
                for module in modules:
                    self.module_to_optim[module] = attr_name

    def _extract_modules_from_optimizer(self, optimizer: Optimizer) -> List[str]:
        """Extract modules from the optimizer's parameter groups."""
        modules = set()
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param in self.param_to_modules:
                    # Already linked to one or more modules
                    linked_modules = self.param_to_modules[param]
                else:
                    linked_modules = set()

                # Look for modules in the current class attributes
                for attr_name, attr in self.evolvable_attributes().items():
                    if isinstance(attr, nn.Module) and attr_name in param._evol_module:
                        linked_modules.add(attr_name)

                self.param_to_modules[param] = linked_modules
                modules.update(linked_modules)

        return list(modules)

    def _identify_param_to_modules(self) -> None:
        """Add an attribute identifying the parameter to the module/s it belongs to."""
        for attr_name, attr in self.evolvable_attributes().items():
            if isinstance(attr, nn.Module):
                for param in attr.parameters():
                    if not hasattr(param, "module"):
                        param._evol_module = set([attr_name])
                    else:
                        param._evol_module.add(attr_name)

    def obs_to_tensor(self, observation: NumpyObsType) -> TorchObsType:
        """Prepares state for forward pass through neural network.

        :param state: Observation of environment
        :type state: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed state
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]]
        """
        device = self.device if self.accelerator is None else self.accelerator.device
        return obs_to_tensor(observation, device)

    def evolvable_attributes(self) -> EvolvableAttributeDict:
        """Returns the attributes related to the evolvable networks in the algorithm. Includes 
        attributes that are either evolvable networks or a list of evolvable networks, as well 
        as the optimizers associated with the networks.
        
        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """
        return {
            attr: getattr(self, attr) for attr in dir(self)
            if recursive_check_module_attrs(getattr(self, attr))
            and not attr.startswith("_") and not attr.endswith("_")
            and not "network" in attr # NOTE: We shouldn't need to do this...
        }
    
    def inspect_attributes(self, input_args_only: bool = False) -> Dict[str, Any]:
        """
        Inspect and retrieve the attributes of the current object, excluding attributes related to the 
        underlying evolvable networks (i.e. `EvolvableModule`'s, `torch.optim.Optimizer`'s) and with 
        an option to include only the attributes that are input arguments to the constructor.

        :param input_args_only: If True, only include attributes that are input arguments to the constructor. 
                                Defaults to False.
        :type input_args_only: bool
        :return: A dictionary of attribute names and their values.
        :rtype: dict[str, Any]
        """
        # Get all attributes of the current object
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))

        # Exclude attributes that are EvolvableModule's or Optimizer's (also check for nested 
        # module-related attributes for multi-agent algorithms)
        exclude = list(self.evolvable_attributes().keys())

        # Exclude private and built-in attributes
        attributes = [
            a for a in attributes if not (a[0].startswith("_") or a[0].endswith("_"))
        ]

        # If input_args_only is True, only include attributes that are 
        # input arguments to the constructor
        if input_args_only:
            constructor_params = inspect.signature(self.__init__).parameters.keys()
            attributes = {
                k: v
                for k, v in attributes
                if k not in exclude and k in constructor_params
            }
        else:
            # Remove the algo specific guarded variables (if specified)
            attributes = {k: v for k, v in attributes if k not in exclude}

        return attributes

    def wrap_models(self) -> None:
        """Wraps the models in the algorithm with the accelerator."""
        if self.accelerator is not None:
            for attr in self.evolvable_attributes():
                obj = getattr(self, attr)
                if isinstance(obj, list):
                    setattr(self, attr, [self.accelerator.prepare(m) for m in obj])
                else:
                    setattr(self, attr, self.accelerator.prepare(obj))


class RLAlgorithm(EvolvableAlgorithm, ABC):
    """Base object for all single-agent algorithms in the AgileRL framework. 
    
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
    multi: bool = True # NOTE: This is to maintain compatibility

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            index: int,
            net_config: Dict[str, Any],
            learn_step: int = 2048,
            device: Union[str, torch.device] = "cpu",
            accelerator: Optional[Accelerator] = None,
            name: Optional[str] = None,
            ) -> None:

        super().__init__(index, learn_step, device, accelerator, name)

        assert isinstance(observation_space, spaces.Space), "Observation space must be an instance of gym.spaces.Space."
        assert isinstance(action_space, spaces.Space), "Action space must be an instance of gym.spaces.Space."

        self.net_config = net_config
        self.observation_space = observation_space
        self.action_space = action_space

        # TODO: This is a bit of a temporary hack until we fully refactor the framework
        self.state_dim = self.get_state_dim(observation_space)
        self.action_dim = self.get_action_dim(action_space)
        self.one_hot = isinstance(observation_space, spaces.Discrete) and observation_space.n > 1
        self.discrete_actions = isinstance(action_space, spaces.Discrete)
        self.min_action = np.array(action_space.low) if hasattr(action_space, "low") else None
        self.max_action = np.array(action_space.high) if hasattr(action_space, "high") else None

    @staticmethod
    def get_state_dim(observation_space: spaces.Space) -> Tuple[int, ...]:
        """Returns the dimension of the state space.
        
        :param observation_space: The observation space of the environment.
        :type observation_space: spaces.Space.
        
        :return: The dimension of the state space.
        :rtype: Tuple[int, ...]."""
        if isinstance(observation_space, spaces.Discrete):
            return (observation_space.n,)
        elif isinstance(observation_space, spaces.Box):
            return observation_space.shape
        else:
            raise AttributeError(f"Can't access state dimensions for {type(observation_space)} spaces.")

    @staticmethod
    def get_action_dim(action_space: spaces.Space) -> int:
        """Returns the dimension of the action space.
        
        :param action_space: The action space of the environment.
        :type action_space: spaces.Space.
        
        :return: The dimension of the action space.
        :rtype: int.
        """
        if isinstance(action_space, spaces.Discrete):
            return action_space.n
        elif isinstance(action_space, spaces.Box):
            # NOTE: Here we assume the action space only has one dimension
            #       (i.e. the actions correspond to a one-dimensional vector)
            return action_space.shape[0]
        else:
            raise AttributeError(f"Can't access state dimensions for {type(action_space)} spaces.")
        

    def save_checkpoint(self, path: str) -> None:
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        :param exclude_accelerator: If True, exclude the accelerator from the checkpoint. Defaults to False.
        :type exclude_accelerator: bool
        """
        attribute_dict = self.inspect_attributes()

        # Extract Evolvable module state dicts and architectures for current checkpoint, as 
        # well as optimizer state dicts
        network_info = {}
        for attr in self.evolvable_attributes():
            obj: EvolvableAttributeType = getattr(self, attr)
            if isinstance(obj, (OptimizedModule, EvolvableModule)):
                    network_info[f"{attr}_init_dict"] = obj.init_dict
                    network_info[f"{attr}_state_dict"] = remove_compile_prefix(obj.state_dict())
            elif isinstance(obj, Optimizer):
                network_info[f"{attr}_state_dict"] = obj.state_dict()
            else:
                raise TypeError(
                    f"Attribute {attr} should be an instance of either EvolvableModule or Optimizer."
                    )

        attribute_dict.update(network_info)

        torch.save(
            attribute_dict,
            path,
            pickle_module=dill,
        )
    
class MultiAgentAlgorithm(EvolvableAlgorithm, ABC):
    """Base object for all multi-agent algorithms in the AgileRL framework. 
    
    :param observation_spaces: The observation spaces of the agent environments.
    :type observation_spaces: List[spaces.Space]
    :param action_space: The action spaces of the agent environments.
    :type action_space: List[spaces.Space]
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
    multi: bool = True # NOTE: This is to maintain compatibility

    def __init__(
            self,
            observation_spaces: Iterable[spaces.Space],
            action_spaces: Iterable[spaces.Space],
            index: int,
            net_config: Dict[str, Any],
            learn_step: int = 2048,
            device: Union[str, torch.device] = "cpu",
            accelerator: Optional[Accelerator] = None,
            torch_compiler: Optional[Any] = None,
            name: Optional[str] = None,
            ) -> None:

        super().__init__(index, learn_step, device, accelerator, name)

        assert isinstance(observation_spaces, (list, tuple)), "Observation spaces must be a list or tuple."
        assert (
            all(isinstance(observation_space, spaces.Space) for observation_space in observation_spaces),
            "Observation spaces must be instances of gym.spaces.Space."
        )
        assert isinstance(action_spaces, (list, tuple)), "Action spaces must be a list or tuple."
        assert (
            all(isinstance(action_space, spaces.Space) for action_space in action_spaces),
            "Action spaces must be instances of gym.spaces.Space."
        )

        # TODO: This is a bit of a temporary hack until we fully refactor the framework
        self.state_dims = self.get_state_dims(observation_spaces)
        self.action_dims = self.get_action_dims(action_spaces)
        self.one_hot = all(isinstance(space, spaces.Discrete) and space.n > 1 for space in observation_spaces)
        self.discrete_actions = all(isinstance(space, spaces.Discrete) for space in action_spaces)

        # For continuous action spaces, store the min and max action values
        if not self.discrete_actions:
            self.min_action = [space.low for space in action_spaces]
            self.max_action = [space.high for space in action_spaces]
        else:
            self.min_action, self.max_action = None, None

        self.torch_compiler = torch_compiler
        self.net_config = net_config
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.total_actions = sum(self.action_dims)
        self.total_state_dims = sum(state_dim[0] for state_dim in self.state_dims)

    @staticmethod
    def get_state_dims(observation_spaces: Iterable[spaces.Space]) -> List[Tuple[int, ...]]:
        """Returns the dimension of the state space.
        
        :param observation_space: The observation space of the environment.
        :type observation_space: spaces.Space.
        
        :return: The dimension of the state space.
        :rtype: Tuple[int, ...]."""
        return [RLAlgorithm.get_state_dim(space) for space in observation_spaces]
    
    @staticmethod
    def get_action_dims(action_spaces: Iterable[spaces.Space]) -> List[int]:
        """Returns the dimension of the action space.
        
        :param action_space: The action space of the environment.
        :type action_space: spaces.Space.
        
        :return: The dimension of the action space.
        :rtype: int.
        """
        return [RLAlgorithm.get_action_dim(space) for space in action_spaces]

    def save_checkpoint(self, path: str) -> None:
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        :param exclude_accelerator: If True, exclude the accelerator from the checkpoint. Defaults to False.
        :type exclude_accelerator: bool
        """
        attribute_dict = self.inspect_attributes()

        # Extract Evolvable module state dicts and architectures for current checkpoint, 
        # as well as optimizer state dicts
        network_info = {}
        for attr in self.evolvable_attributes():
            obj: EvolvableAttributeType = getattr(self, attr)
            if is_module_list(obj):
                network_info[f"{attr}_init_dict"] = [net.init_dict for net in obj]
                network_info[f"{attr}_state_dict"] = [
                    remove_compile_prefix(net.state_dict()) for net in obj
                ]
            elif is_optimizer_list(obj):
                network_info[f"{attr}_state_dict"] = [opt.state_dict() for opt in obj]
            else:
                raise TypeError(f"Attribute {attr} should be a list of either EvolvableModule's or Optimizer's.")

        attribute_dict.update(network_info)
        attribute_dict.pop("accelerator", None)

        torch.save(
            attribute_dict,
            path,
            pickle_module=dill,
        )
