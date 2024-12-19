from typing import Optional, Union, Tuple, Iterable, Callable, Any, Dict, TypeVar, Type
import inspect
import copy
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
from agilerl.typing import NumpyObsType, TorchObsType, ObservationType, GymSpaceType, DeviceType
from agilerl.utils.algo_utils import (
    chkpt_attribute_to_device,
    compile_model,
    recursive_check_module_attrs,
    remove_compile_prefix,
    assert_supported_space,
    is_module_list,
    isroutine,
    preprocess_observation
)

__all__ = ["EvolvableAlgorithm", "RLAlgorithm", "MultiAgentAlgorithm"]

SelfEvolvableAlgorithm = TypeVar("SelfEvolvableAlgorithm", bound="EvolvableAlgorithm")
SelfRLAlgorithm = TypeVar("T", bound="RLAlgorithm")

class _RegistryMeta(type):
    """Metaclass to wrap registry information after algorithm is done 
    intiializing with specified network groups and optimizers."""
    def __call__(cls: Type[SelfEvolvableAlgorithm], *args, **kwargs) -> SelfEvolvableAlgorithm:
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
    
    def unwrap_models(self):
        if self.accelerator is None:
            raise AttributeError("No accelerator has been set for the algorithm.")
        
        for attr in self.evolvable_attributes(networks_only=True):
            obj = getattr(self, attr)
            if isinstance(obj, list):
                setattr(self, attr, [self.accelerator.unwrap_model(m) for m in obj])
            else:
                setattr(self, attr, self.accelerator.unwrap_model(obj))

    def models_to_device(self) -> None:
        """Moves the models in the algorithm to the device."""
        for name, obj in self.evolvable_attributes(networks_only=True).items():
            if isinstance(obj, list):
                setattr(self, name, [m.to(self.device) for m in obj])
            else:
                setattr(self, name, obj.to(self.device))

    def clone(
            self: SelfEvolvableAlgorithm,
            index: Optional[int] = None,
            wrap: bool = True
            ) -> SelfEvolvableAlgorithm:
        """Creates a clone of the algorithm.

        :param index: The index of the clone, defaults to None
        :type index: Optional[int], optional
        :param wrap: If True, wrap the models in the clone with the accelerator, defaults to False
        :type wrap: bool, optional

        :return: A clone of the algorithm
        :rtype: EvolvableAlgorithm
        """
        # Make copy using input arguments
        input_args = self.inspect_attributes(input_args_only=True)
        input_args["wrap"] = wrap
        clone = type(self)(**input_args)

        # Clone evolvable modules
        cloned_modules = {}
        for attr, obj in self.evolvable_attributes(networks_only=True).items():
            cloned_modules[attr] = obj.clone()
            setattr(clone, attr, cloned_modules[attr])
        
        # Reinitialize optimizers
        for opt_config in self.registry.optimizers:
            opt = OptimizerWrapper(
                getattr(torch.optim, opt_config.optimizer_cls),
                networks=[cloned_modules[net] for net in opt_config.networks],
                network_names=opt_config.networks,
                optimizer_kwargs=opt_config.optimizer_kwargs
            )
            orig_optimizer: OptimizerWrapper = getattr(self, opt_config.name)
            opt.load_state_dict(orig_optimizer.state_dict())
            setattr(clone, opt_config.name, opt)
        
        # Prepare with accelerator if necessary
        if self.accelerator is not None and wrap:
            clone.wrap_models()

        # Copy non-evolvable attributes back to clone
        for attribute in self.inspect_attributes().keys():
            if hasattr(self, attribute) and hasattr(clone, attribute):
                attr, clone_attr = getattr(self, attribute), getattr(clone, attribute)
                if isinstance(attr, torch.Tensor) or isinstance(
                    clone_attr, torch.Tensor
                ):
                    if not torch.equal(attr, clone_attr):
                        setattr(
                            clone, attribute, copy.deepcopy(getattr(self, attribute))
                        )
                elif isinstance(attr, np.ndarray) or isinstance(clone_attr, np.ndarray):
                    if not np.array_equal(attr, clone_attr):
                        setattr(
                            clone, attribute, copy.deepcopy(getattr(self, attribute))
                        )
                elif attr != clone_attr:
                        setattr(
                            clone, attribute, copy.deepcopy(getattr(self, attribute))
                        )
            else:
                setattr(clone, attribute, copy.deepcopy(getattr(self, attribute)))

        if index is not None:
            clone.index = index
        
        return clone

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
            learn_step: int = 2048,
            device: Union[str, torch.device] = "cpu",
            accelerator: Optional[Accelerator] = None,
            normalize_images: bool = True,
            name: Optional[str] = None,
            ) -> None:

        super().__init__(index, learn_step, device, accelerator, name)

        assert isinstance(observation_space, spaces.Space), "Observation space must be an instance of gym.spaces.Space."
        assert isinstance(action_space, spaces.Space), "Action space must be an instance of gym.spaces.Space."

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
    
    def load_checkpoint(self, path: str) -> None:
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint: Dict[str, Any] = torch.load(path, map_location=self.device, pickle_module=dill)

        # Recreate evolvable modules
        network_info: Dict[str, Dict[str, Any]] = checkpoint['network_info']
        network_names = network_info['network_names']
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info['modules'].items() if k.startswith(name)
                }
            
            module_cls = net_dict[f"{name}_cls"]
            state_dict = net_dict[f"{name}_state_dict"]
            loaded_module: EvolvableModule = module_cls(**net_dict[f"{name}_init_dict"])
            
            # NOTE: Sometimes we may have empty state dicts (e.g. detached modules)
            if state_dict:
                loaded_module.load_state_dict(net_dict[f"{name}_state_dict"])

            setattr(self, name, loaded_module)
        
        optimizer_names = network_info['optimizer_names']
        for name in optimizer_names:
            opt_dict = {
                k: v for k, v in network_info['optimizers'].items() if k.startswith(name)
                }
            
            # Add device to optimizer kwargs
            opt_kwargs = opt_dict[f"{name}_kwargs"]
            optimizer_cls = opt_dict[f"{name}_cls"]
            opt_networks = opt_dict[f"{name}_networks"]
            optimizer = OptimizerWrapper(
                getattr(torch.optim, optimizer_cls),
                networks=[getattr(self, net) for net in opt_networks],
                network_names=opt_networks,
                optimizer_kwargs=opt_kwargs
                )

            optimizer.load_state_dict(opt_dict[f"{name}_state_dict"])
            setattr(self, name, optimizer)

        # Init hooks
        for hook in self.registry.hooks:
            getattr(self, hook)()

        # Load other attributes
        checkpoint.pop('network_info')
        for attribute in checkpoint.keys():
            setattr(self, attribute, checkpoint[attribute])


    def save_checkpoint(self, path: str) -> None:
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        :param exclude_accelerator: If True, exclude the accelerator from the checkpoint. Defaults to False.
        :type exclude_accelerator: bool
        """
        attribute_dict = self.inspect_attributes()

        # Extract info on evolvable modules and optimizers in the algorithm
        network_info: Dict[str, Dict[str, Any]] = {"modules": {}, "optimizers": {}}
        for attr in self.evolvable_attributes():
            obj: EvolvableAttributeType = getattr(self, attr)
            if isinstance(obj, (OptimizedModule, EvolvableModule)):
                obj_cls = obj._orig_mod.__class__ if isinstance(obj, OptimizedModule) else obj.__class__
                network_info["modules"].update({
                    f"{attr}_cls": obj_cls,
                    f"{attr}_init_dict": obj.init_dict,
                    f"{attr}_state_dict": remove_compile_prefix(obj.state_dict())
                })
            elif isinstance(obj, OptimizerWrapper):
                network_info['optimizers'].update({
                    f"{attr}_cls": obj.optimizer_cls.__name__,
                    f"{attr}_state_dict": obj.state_dict(),
                    f"{attr}_networks": obj.network_names,
                    f"{attr}_kwargs": obj.optimizer_kwargs
                })
            else:
                raise TypeError(
                    f"Something went wrong. Identified '{attr}' as an evolvable module "
                    f"when it is of type {type(obj)}."
                )

        network_attr_names = [name for name in self.evolvable_attributes() if isinstance(getattr(self, name), (OptimizedModule, EvolvableModule))]
        optimizer_attr_names = [name for name in self.evolvable_attributes() if isinstance(getattr(self, name), OptimizerWrapper)]
        network_info["network_names"] = network_attr_names
        network_info["optimizer_names"] = optimizer_attr_names
        attribute_dict['network_info'] = network_info

        torch.save(
            attribute_dict,
            path,
            pickle_module=dill,
        )

    @classmethod
    def load(
        cls: Type[SelfRLAlgorithm],
        path: str,
        device: DeviceType = 'cpu',
        accelerator: Optional[Accelerator] = None
        ) -> SelfRLAlgorithm:
        """Loads an algorithm from a checkpoint.

        :param path: Location to load checkpoint from.
        :type path: string
        :param device: Device to load the algorithm on, defaults to 'cpu'
        :type device: str, optional
        :param accelerator: Accelerator object for distributed computing, defaults to None
        :type accelerator: Optional[Accelerator], optional

        :return: An instance of the algorithm
        :rtype: RLAlgorithm
        """
        checkpoint: Dict[str, Any] = torch.load(
            path,
            map_location=device,
            pickle_module=dill
            )

        # Reconstruct evolvable modules in algorithm
        network_info: Dict[str, Dict[str, Any]] = checkpoint['network_info']
        network_names = network_info['network_names']
        loaded_modules = {}
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info['modules'].items() if k.startswith(name)
                }

            # Add device to init dict
            init_dict = net_dict.get(f"{name}_init_dict", None)
            if init_dict is None:
                raise ValueError(f"Init dict for {name} not found in checkpoint.")

            init_dict['device'] = device
            init_dict = chkpt_attribute_to_device(init_dict, device)

            state_dict = net_dict.get(f"{name}_state_dict", None)
            if state_dict is None:
                raise ValueError(f"State dict for {name} not found in checkpoint.")

            state_dict = chkpt_attribute_to_device(state_dict, device)

            # Reconstruct the modules
            module_cls: Type[EvolvableModule] = net_dict[f"{name}_cls"]
            module = module_cls(**init_dict)

            if state_dict: # Sometimes we may have empty state dicts (e.g. detached modules)
                module.load_state_dict(state_dict)

            loaded_modules[name] = module

        # Reconstruct optimizers in algorithm
        optimizer_names = network_info['optimizer_names']
        loaded_optimizers = {}
        for name in optimizer_names:
            opt_dict = {
                k: v for k, v in network_info['optimizers'].items() if k.startswith(name)
                }

            # Add device to optimizer kwargs
            opt_kwargs = chkpt_attribute_to_device(opt_dict[f"{name}_kwargs"], device)
            optimizer_cls = opt_dict[f"{name}_cls"]
            opt_networks = opt_dict[f"{name}_networks"]
            optimizer = OptimizerWrapper(
                getattr(torch.optim, optimizer_cls),
                networks=[loaded_modules[net] for net in opt_networks],
                network_names=opt_networks,
                optimizer_kwargs=opt_kwargs
                )

            state_dict = chkpt_attribute_to_device(opt_dict[f"{name}_state_dict"], device)
            optimizer.load_state_dict(state_dict)
            loaded_optimizers[name] = optimizer

        # Reconstruct the algorithm
        constructor_params = inspect.signature(cls.__init__).parameters.keys()
        class_init_dict = {
            k: v for k, v in checkpoint.items() if k in constructor_params
        }

        class_init_dict['accelerator'] = accelerator
        self = cls(**class_init_dict)

        # Assign loaded modules and optimizers to the algorithm
        for name, module in loaded_modules.items():
            setattr(self, name, module)

        for name, optimizer in loaded_optimizers.items():
            setattr(self, name, optimizer)

        if accelerator is not None:
            self.wrap_models()

        for attribute in self.inspect_attributes().keys():
            setattr(self, attribute, checkpoint[attribute])

        # Run init hooks
        for hook in self.registry.hooks:
            getattr(self, hook)()

        return self

    
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
