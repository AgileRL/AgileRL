from typing import Optional, Union, Tuple, Iterable, Callable, Any, Dict, TypeVar
import inspect
from abc import ABC, ABCMeta, abstractmethod
from gymnasium import spaces
from accelerate import Accelerator
import numpy as np
from tensordict import TensorDict
from numpy.typing import ArrayLike
import torch
from torch._dynamo import OptimizedModule
import dill

from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.algorithms.core.registry import MutationRegistry, NetworkGroup, OptimizerConfig
from agilerl.protocols import EvolvableModule, EvolvableAttributeType, EvolvableAttributeDict
from agilerl.typing import NumpyObsType, TorchObsType, ObservationType, GymSpaceType
from agilerl.utils.algo_utils import (
    compile_model,
    recursive_check_module_attrs,
    remove_compile_prefix,
    assert_supported_space,
    is_module_list,
    isroutine,
    is_optimizer_list,
    preprocess_observation
)

__all__ = ["EvolvableAlgorithm", "RLAlgorithm", "MultiAgentAlgorithm"]

SelfEvolvableAlgorithm = TypeVar("SelfEvolvableAlgorithm", bound="EvolvableAlgorithm")

class _RegistryMeta(type):
    """Metaclass to wrap registry information after algorithm is done 
    intiializing with specified network groups and optimizers."""
    def __call__(cls, *args, **kwargs):
        # Create the instance
        instance: SelfEvolvableAlgorithm = super().__call__(*args, **kwargs)

        # Call the base class post_init_hook after all initialization
        if isinstance(instance, cls) and hasattr(instance, "_register_networks"):
            instance._register_networks()

        return instance

class RegistryMeta(_RegistryMeta, ABCMeta):
    ...

class EvolvableAlgorithm(ABC, metaclass=RegistryMeta):
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

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(index, int), "Agent index must be an integer."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert isinstance(device, (str, torch.device)), "Device must be a string."
        assert isinstance(name, (type(None), str)), "Name must be a string."
        assert (
            isinstance(accelerator, (type(None), Accelerator)), 
            "Accelerator must be an instance of Accelerator."
        )

        self.accelerator = accelerator
        self.device = device if self.accelerator is None else self.accelerator.device
        self.learn_step = learn_step
        self.algo = name if name is not None else self.__class__.__name__

        self.eval_network = None
        self._mut = None
        self._index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]
        self.registry = MutationRegistry()

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
    def preprocess_observation(self, observation: ObservationType) -> TorchObsType:
        """Preprocesses observations for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]]
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

    @staticmethod
    def get_state_dim(observation_space: GymSpaceType) -> Tuple[int, ...]:
        """Returns the dimension of the state space.
        
        :param observation_space: The observation space of the environment.
        :type observation_space: spaces.Space or List[spaces.Space].
        
        :return: The dimension of the state space.
        :rtype: Tuple[int, ...]."""
        if isinstance(observation_space, (list, tuple, spaces.Tuple)):
            assert_supported_space(observation_space)
            return tuple(EvolvableAlgorithm.get_state_dim(space) for space in observation_space)
        elif isinstance(observation_space, spaces.Discrete):
            return (observation_space.n,)
        elif isinstance(observation_space, spaces.Box):
            return observation_space.shape
        elif isinstance(observation_space, spaces.Dict):
            assert_supported_space(observation_space)
            return {key: EvolvableAlgorithm.get_state_dim(subspace) for key, subspace in observation_space.spaces.items()}
        else:
            raise AttributeError(f"Can't access state dimensions for {type(observation_space)} spaces.")

    @staticmethod
    def get_action_dim(action_space: GymSpaceType) -> int:
        """Returns the dimension of the action space.
        
        :param action_space: The action space of the environment.
        :type action_space: spaces.Space or List[spaces.Space].
        
        :return: The dimension of the action space.
        :rtype: int.
        """
        if isinstance(action_space, (list, tuple)):
            return tuple(EvolvableAlgorithm.get_action_dim(space) for space in action_space)
        if isinstance(action_space, spaces.Discrete):
            return action_space.n
        elif isinstance(action_space, spaces.Box):
            # NOTE: Here we assume the action space only has one dimension
            #       (i.e. the actions correspond to a one-dimensional vector)
            return action_space.shape[0]
        else:
            raise AttributeError(f"Can't access action dimensions for {type(action_space)} spaces.")
    
    def _register_networks(self) -> None:
        """Registers the networks in the algorithm with the module registry."""
        if not self.registry.groups:
            raise ValueError(
                "No network groups have been registered with the algorithm's __init__ method. "
                "Please register NetworkGroup objects specifying all of the evaluation and "
                "shared/target networks through the `register_network_group()` method."
            )

        # Extract mapping from optimizer names to network attribute names
        for attr, obj in self.evolvable_attributes().items():
            if isinstance(obj, OptimizerWrapper):
                # Set up config from OptimizerWrapper
                config = OptimizerConfig(
                    name=attr,
                    networks=obj.network_names,
                    optimizer_cls=obj.optimizer_cls,
                    optimizer_kwargs=obj.optimizer_kwargs
                    )
                self.registry.register_optimizer(config)

        # Check that all the inspected evolvable attributes can be found in the registry
        all_registered = self.registry.all_registered()
        not_found = [attr for attr in self.evolvable_attributes() if attr not in all_registered]
        if not_found:
            raise ValueError(
                f"The following evolvable attributes could not be found in the registry: {not_found}. "
                "Please check that the defined NetworkGroup objects contain all of the EvolvableModule's "
                "in the algorithm."
            )

        # Check that one of the network groups relates to a policy
        if not any(group.policy for group in self.registry.groups):
            raise ValueError(
                "No network group has been registered as a policy (e.g. the network used to "
                "select actions) in the registry. Please register a NetworkGroup object "
                "specifying the policy network."
            )


    def register_network_group(self, group: NetworkGroup) -> None:
        """Sets the evaluation network for the algorithm.

        :param name: The name of the evaluation network.
        :type name: str
        """
        self.registry.register_group(group)
    
    def register_init_hook(self, hook: Callable) -> None:
        """Registers a hook to be executed after a mutation is performed on 
        the algorithm.
        
        :param hook: The hook to be executed after mutation.
        :type hook: Callable
        """
        self.registry.register_hook(hook)
    
    def init_hook(self) -> None:
        """Executes the hooks registered with the algorithm."""
        for hook in self.registry.hooks:
            getattr(self, hook)()

    def evolvable_attributes(self, networks_only: bool = False) -> EvolvableAttributeDict:
        """Returns the attributes related to the evolvable networks in the algorithm. Includes 
        attributes that are either evolvable networks or a list of evolvable networks, as well 
        as the optimizers associated with the networks.

        :param networks_only: If True, only include evolvable networks, defaults to False
        :type networks_only: bool, optionals
        
        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """
        def is_evolvable(attr: str, obj: Any):
            return (
                recursive_check_module_attrs(obj, networks_only)
                and not attr.startswith("_") and not attr.endswith("_")
                and not "network" in attr # NOTE: We shouldn't need to do this...
            )
        
        # Inspect evolvable given specs
        evolvable_attrs = {}
        for attr in dir(self):
            obj = getattr(self, attr)
            if is_evolvable(attr, obj):
                evolvable_attrs[attr] = obj

        return evolvable_attrs

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
        attributes = inspect.getmembers(self, lambda a: not isroutine(a))

        # Exclude attributes that are EvolvableModule's or Optimizer's (also check for nested 
        # module-related attributes for multi-agent algorithms)
        exclude = list(self.evolvable_attributes().keys())
        exclude += [attr for attr, val in attributes if isinstance(val, TensorDict)]

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
    
    def _wrap_attr(self, attr: EvolvableAttributeType) -> EvolvableModule:
        """Wraps the model with the accelerator."""
        if isinstance(attr, OptimizerWrapper):
            if isinstance(attr.optimizer, list):
                wrapped_opt = [self.accelerator.prepare(opt) for opt in attr.optimizer]
            else:
                wrapped_opt = self.accelerator.prepare(attr.optimizer)

            attr.optimizer = wrapped_opt
            return attr
        
        # Only wrap the model if its part of the computation graph
        return self.accelerator.prepare(attr) if attr.state_dict() else attr

    def wrap_models(self) -> None:
        """Wraps the models in the algorithm with the accelerator."""
        if self.accelerator is None:
            raise AttributeError("No accelerator has been set for the algorithm.")

        for attr in self.evolvable_attributes():
            obj = getattr(self, attr)
            if isinstance(obj, list):
                setattr(self, attr, [self._wrap_attr(m) for m in obj])
            else:
                setattr(self, attr, self._wrap_attr(obj))
    
    def models_to_device(self) -> None:
        """Moves the models in the algorithm to the device."""
        for name, obj in self.evolvable_attributes(networks_only=True).items():
            if isinstance(obj, list):
                setattr(self, name, [m.to(self.device) for m in obj])
            else:
                setattr(self, name, obj.to(self.device))

    # @classmethod
    # def load(cls, path: str, device: str = "cpu", accelerator: Optional[Accelerator] = None):
    #     """Creates agent with properties and network weights loaded from path.

    #     :param path: Location to load checkpoint from
    #     :type path: string
    #     :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    #     :type device: str, optional
    #     :param accelerator: Accelerator for distributed computing, defaults to None
    #     :type accelerator: accelerate.Accelerator(), optional
    #     """
    #     checkpoint: Dict[str, Any] = torch.load(path, map_location=device, pickle_module=dill)

    #     # Assign specified device to underlying model init dicts
    #     init_dict_keys = [key for key in checkpoint.keys() if key.endswith("_init_dict")]
    #     n_agents = len(checkpoint["agent_ids"])
    #     for agent_idx in range(n_agents):
    #         for key in init_dict_keys:
    #             checkpoint[key][agent_idx]["device"] = device
        
    #     # Remove state and init dicts of models and optimizers from checkpoint
    #     state_dict_keys = [key for key in checkpoint.keys() if key.endswith("_state_dict")]
    #     model_keys = init_dict_keys + state_dict_keys
    #     model_cache = {key: checkpoint.pop(key) for key in model_keys}

    #     # Change device and accelerator for the agent
    #     checkpoint["device"] = device
    #     checkpoint["accelerator"] = accelerator

    #     # Create agent instance adn initilize networks
    #     constructor_params = inspect.signature(cls.__init__).parameters.keys()
    #     class_init_dict = {
    #         k: v for k, v in checkpoint.items() if k in constructor_params
    #     }

    #     raise NotImplementedError("Need to implement the rest of the load method.")

    # def load_checkpoint(self, path: str) -> None:
    #     """Loads saved agent properties and network weights from checkpoint.

    #     :param path: Location to load checkpoint from
    #     :type path: string
    #     """

    #     if self.accelerator is not None:
    #         self.wrap_models()
    #     else:
    #         self.models_to_device()
    #         if self.torch_compiler:
    #             torch.set_float32_matmul_precision("high")
    #             self.recompile()

    #     raise NotImplementedError("Need to implement the rest of the load_checkpoint method.")
    
    # def clone(self, index: Optional[int] = None, wrap: bool = False) -> "EvolvableAlgorithm":
    #     """Creates a clone of the algorithm.

    #     :param index: The index of the clone, defaults to None
    #     :type index: Optional[int], optional
    #     :param wrap: If True, wrap the models in the clone with the accelerator, defaults to False
    #     :type wrap: bool, optional

    #     :return: A clone of the algorithm
    #     :rtype: EvolvableAlgorithm
    #     """
    #     # Make copy using input arguments
    #     input_args = self.inspect_attributes(input_args_only=True)
    #     input_args["wrap"] = wrap
    #     clone = type(self)(**input_args)

    #     # TODO: Copy over evolvable attributes
    #     # Here we need to know which evolvable modules belong to which optimizers,
    #     # which is hard to determine systematically without user input -> Need to 
    #     # think harder about this...

    #     # Copy non-evolvable attributes back to clone
    #     for attribute in self.inspect_attributes().keys():
    #         if hasattr(self, attribute) and hasattr(clone, attribute):
    #             attr, clone_attr = getattr(self, attribute), getattr(clone, attribute)
    #             if isinstance(attr, torch.Tensor) or isinstance(
    #                 clone_attr, torch.Tensor
    #             ):
    #                 if not torch.equal(attr, clone_attr):
    #                     setattr(
    #                         clone, attribute, copy.deepcopy(getattr(self, attribute))
    #                     )
    #             else:
    #                 if attr != clone_attr:
    #                     setattr(
    #                         clone, attribute, copy.deepcopy(getattr(self, attribute))
    #                     )
    #         else:
    #             setattr(clone, attribute, copy.deepcopy(getattr(self, attribute)))

    #     if index is not None:
    #         clone.index = index
        
    #     return clone

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
    :param normalize_images: If True, normalize images, defaults to True
    :type normalize_images: bool, optional
    :param name: Name of the algorithm, defaults to the class name
    :type name: Optional[str], optional
    """
    multi: bool = False # NOTE: This is for backwards compatibility

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            index: int,
            net_config: Dict[str, Any],
            learn_step: int = 2048,
            device: Union[str, torch.device] = "cpu",
            accelerator: Optional[Accelerator] = None,
            normalize_images: bool = True,
            name: Optional[str] = None,
            ) -> None:

        super().__init__(index, learn_step, device, accelerator, name)

        assert isinstance(observation_space, spaces.Space), "Observation space must be an instance of gym.spaces.Space."
        assert isinstance(action_space, spaces.Space), "Action space must be an instance of gym.spaces.Space."

        self.net_config = net_config
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images

        # TODO: This is a bit of a temporary hack to support legacy code
        self.state_dim = self.get_state_dim(observation_space)
        self.action_dim = self.get_action_dim(action_space)
        self.discrete_actions = isinstance(action_space, spaces.Discrete)
        self.min_action = np.array(action_space.low) if hasattr(action_space, "low") else None
        self.max_action = np.array(action_space.high) if hasattr(action_space, "high") else None
    
    def preprocess_observation(self, observation: NumpyObsType) -> TorchObsType:
        """Preprocesses observations for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: ObservationType

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or Tuple[torch.Tensor[float], ...]
        """
        return preprocess_observation(
            observation=observation,
            observation_space=self.observation_space,
            device=self.device,
            normalize_images=self.normalize_images
        )
    
    def to_device(self, *experiences: TorchObsType) -> Tuple[TorchObsType, ...]:
        """Moves experiences to the device.

        :param experiences: Experiences to move to device
        :type experiences: Tuple[torch.Tensor[float], ...]

        :return: Experiences on the device
        :rtype: Tuple[torch.Tensor[float], ...]
        """
        device = self.device if self.accelerator is None else self.accelerator.device
        on_device = []
        for exp in experiences:
            if isinstance(exp, dict):
                exp = {key: val.to(device) for key, val in exp.items()}
            elif isinstance(exp, (list, tuple)):
                exp = [val.to(device) for val in exp]
            elif isinstance(exp, torch.Tensor):
                exp = exp.to(device)
            else:
                raise TypeError(f"Unsupported experience type: {type(exp)}")
            
            on_device.append(exp)
        
        return on_device

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
            elif isinstance(obj, OptimizerWrapper):
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
    :param agent_ids: The agent IDs of the agents in the environment.
    :type agent_ids: List[int]
    :param index: The index of the individual in the population.
    :type index: int.
    :param learn_step: Learning frequency, defaults to 2048
    :type learn_step: int, optional
    :param device: Device to run the algorithm on, defaults to "cpu"
    :type device: str, optional
    :param accelerator: Accelerator object for distributed computing, defaults to None
    :type accelerator: Optional[Accelerator], optional
    :param normalize_images: If True, normalize images, defaults to True
    :type normalize_images: bool, optional
    :param torch_compiler: The torch compiler mode to use, defaults to None
    :type torch_compiler: Optional[Any], optional
    :param name: Name of the algorithm, defaults to the class name
    :type name: Optional[str], optional
    """
    multi: bool = True # NOTE: This is for backwards compatibility

    def __init__(
            self,
            observation_spaces: Iterable[spaces.Space],
            action_spaces: Iterable[spaces.Space],
            agent_ids: Iterable[int],
            index: int,
            net_config: Dict[str, Any],
            learn_step: int = 2048,
            device: Union[str, torch.device] = "cpu",
            accelerator: Optional[Accelerator] = None,
            normalize_images: bool = True,
            torch_compiler: Optional[Any] = None,
            name: Optional[str] = None,
            ) -> None:

        super().__init__(index, learn_step, device, accelerator, name)

        assert isinstance(
            agent_ids, (tuple, list)
        ), "Agent IDs must be stores in a tuple or list."
        assert len(agent_ids) == len(observation_spaces), "Number of agent IDs must match number of observation spaces."
        assert isinstance(observation_spaces, (list, tuple)), "Observation spaces must be a list or tuple."
        assert (
            all(isinstance(_space, spaces.Space) for _space in observation_spaces),
            "Observation spaces must be instances of gym.spaces.Space."
        )
        assert isinstance(action_spaces, (list, tuple)), "Action spaces must be a list or tuple."
        assert (
            all(isinstance(_space, spaces.Space) for _space in action_spaces),
            "Action spaces must be instances of gym.spaces.Space."
        )

        if torch_compiler:
            assert torch_compiler in [
                "default",
                "reduce-overhead",
                "max-autotune",
            ], "Choose between torch compiler modes: default, reduce-overhead, max-autotune or None"

         # TODO: This is a bit of a temporary hack to support legacy code
        self.state_dims = self.get_state_dim(observation_spaces)
        self.action_dims = self.get_action_dim(action_spaces)
        self.one_hot = all(isinstance(space, spaces.Discrete) for space in observation_spaces)
        self.discrete_actions = all(isinstance(space, spaces.Discrete) for space in action_spaces)

        # For continuous action spaces, store the min and max action values
        if not self.discrete_actions:
            self.min_action = [space.low for space in action_spaces]
            self.max_action = [space.high for space in action_spaces]
        else:
            self.min_action, self.max_action = None, None

        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.torch_compiler = torch_compiler
        self.net_config = net_config
        self.normalize_images = normalize_images
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.total_actions = sum(self.action_dims)
        self.total_state_dims = sum(state_dim[0] for state_dim in self.state_dims)

        # Build observation and action space dictionaries using agent IDs
        self.observation_space = spaces.Dict({
            agent_id: space for agent_id, space in zip(agent_ids, observation_spaces)
        })
        self.action_space = spaces.Dict({
            agent_id: space for agent_id, space in zip(agent_ids, action_spaces)
        })

    def recompile(self) -> None:
        """Recompiles the evolvable modules in the algorithm with the specified torch compiler."""
        for name, obj in self.evolvable_attributes(networks_only=True).items():
            compiled_modules = [compile_model(module, self.torch_compiler) for module in obj]
            setattr(self, name, compiled_modules)

    def preprocess_observation(self, observation: ObservationType) -> Dict[str, TorchObsType]:
        """Preprocesses observations for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or Tuple[torch.Tensor[float], ...]
        """
        preprocessed = {}
        for agent_id, obs in observation.items():
            preprocessed[agent_id] = preprocess_observation(
                observation=obs,
                observation_space=self.observation_space.get(agent_id),
                device=self.device,
                normalize_images=self.normalize_images
            )

        return preprocessed

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
            if isinstance(obj, OptimizerWrapper):
                network_info[f"{attr}_state_dict"] = [opt.state_dict() for opt in obj.optimizer]
            elif is_module_list(obj):
                network_info[f"{attr}_init_dict"] = [net.init_dict for net in obj]
                network_info[f"{attr}_state_dict"] = [
                    remove_compile_prefix(net.state_dict()) for net in obj
                ]
            else:
                raise TypeError(
                    f"Attribute {attr} should be a list of either EvolvableModule's or Optimizer's."
                    )

        attribute_dict.update(network_info)
        attribute_dict.pop("accelerator", None)

        torch.save(
            attribute_dict,
            path,
            pickle_module=dill,
        )