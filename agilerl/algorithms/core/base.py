import copy
import gc
import glob
import inspect
import os
import tempfile
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from importlib.metadata import version
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import dill
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from gymnasium import spaces
from numpy.typing import ArrayLike
from tensordict import TensorDict
from torch._dynamo import OptimizedModule
from torch.optim.lr_scheduler import SequentialLR

from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    MutationRegistry,
    NetworkGroup,
    OptimizerConfig,
)
from agilerl.modules.configs import (
    MlpNetConfig,
)
from agilerl.protocols import (
    AgentWrapper,
    EvolvableAttributeDict,
    EvolvableAttributeType,
    EvolvableModule,
    ModuleDict,
)
from agilerl.typing import (
    ActionType,
    ArrayDict,
    DeviceType,
    ExperiencesType,
    GymSpaceType,
    InfosDict,
    ModuleType,
    MultiAgentObservationType,
    MultiAgentSetup,
    NetConfigType,
    ObservationType,
    TorchObsType,
)
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    chkpt_attribute_to_device,
    clone_llm,
    create_warmup_cosine_scheduler,
    isroutine,
    key_in_nested_dict,
    module_checkpoint_dict,
    preprocess_observation,
    recursive_check_module_attrs,
    stack_experiences,
)
from agilerl.utils.evolvable_networks import (
    compile_model,
    config_from_dict,
    get_default_encoder_config,
    get_input_size_from_space,
    get_output_size_from_space,
    is_image_space,
    is_vector_space,
)

__all__ = ["EvolvableAlgorithm", "RLAlgorithm", "MultiAgentRLAlgorithm"]

SelfEvolvableAlgorithm = TypeVar("SelfEvolvableAlgorithm", bound="EvolvableAlgorithm")
SelfRLAlgorithm = TypeVar("SelfRLAlgorithm", bound="RLAlgorithm")
SelfAgentWrapper = TypeVar("SelfAgentWrapper", bound="AgentWrapper")


class _RegistryMeta(type):
    """Metaclass to wrap registry information after algorithm is done
    initializing with specified network groups and optimizers."""

    def __call__(
        cls: Type[SelfEvolvableAlgorithm], *args, **kwargs
    ) -> SelfEvolvableAlgorithm:
        # Create the instance
        instance: SelfEvolvableAlgorithm = super().__call__(*args, **kwargs)

        # Call the base class post_init_hook after all initialization
        if isinstance(instance, cls) and hasattr(instance, "_registry_init"):
            instance._registry_init()

        return instance


class RegistryMeta(_RegistryMeta, ABCMeta): ...


def get_checkpoint_dict(agent: SelfEvolvableAlgorithm) -> Dict[str, Any]:
    """Returns a dictionary of the agent's attributes to save in a checkpoint.

    :param agent: The agent to save.
    :type agent: EvolvableAlgorithm

    :return: A dictionary of the agent's attributes.
    :rtype: dict[str, Any]
    """
    attribute_dict = EvolvableAlgorithm.inspect_attributes(agent)

    # Get checkpoint dictionaries for evolvable modules and optimizers
    network_info: Dict[str, Dict[str, Any]] = {"modules": {}, "optimizers": {}}
    for attr in agent.evolvable_attributes():
        evolvable_obj: EvolvableAttributeType = getattr(agent, attr)
        if isinstance(evolvable_obj, OptimizerWrapper):
            optimizer_chkpt = evolvable_obj.checkpoint_dict(attr)
            network_info["optimizers"].update(optimizer_chkpt)

        elif isinstance(evolvable_obj, (OptimizedModule, EvolvableModule)):
            module_chkpt = module_checkpoint_dict(evolvable_obj, attr)
            network_info["modules"].update(module_chkpt)

        else:
            raise TypeError(
                f"Something went wrong. Identified '{attr}' as an evolvable module or "
                f"optimizer when it is of type {type(evolvable_obj)}."
            )

    network_attr_names = [
        name for name in agent.evolvable_attributes(networks_only=True)
    ]
    optimizer_attr_names = [
        name
        for name in agent.evolvable_attributes()
        if isinstance(getattr(agent, name), OptimizerWrapper)
    ]

    network_info["network_names"] = network_attr_names
    network_info["optimizer_names"] = optimizer_attr_names
    attribute_dict["network_info"] = network_info
    attribute_dict["agilerl_version"] = version("agilerl")
    attribute_dict.pop("accelerator", None)

    if attribute_dict.pop("lr_scheduler", None) is not None:
        attribute_dict["lr_scheduler"] = agent.lr_scheduler.state_dict()

    return attribute_dict


def get_optimizer_cls(
    optimizer_cls: Union[str, Dict[str, str]],
) -> Union[Type[torch.optim.Optimizer], Dict[str, Type[torch.optim.Optimizer]]]:
    """Returns the optimizer class from the string or dictionary of optimizer classes.

    :param optimizer_cls: The optimizer class or dictionary of optimizer classes.
    :type optimizer_cls: Union[str, Dict[str, str]]
    :return: The optimizer class or dictionary of optimizer classes.
    :rtype: Union[Type[torch.optim.Optimizer], Dict[str, Type[torch.optim.Optimizer]]]
    """
    if isinstance(optimizer_cls, dict):
        optimizer_cls = {
            agent_id: getattr(torch.optim, optimizer_cls[agent_id])
            for agent_id in optimizer_cls.keys()
        }
    else:
        optimizer_cls = getattr(torch.optim, optimizer_cls)

    return optimizer_cls


class EvolvableAlgorithm(ABC, metaclass=RegistryMeta):
    """Base object for all algorithms in the AgileRL framework.

    :param index: The index of the individual.
    :type index: int
    :param hp_config: Hyperparameter configuration for the algorithm, defaults to None.
    :type hp_config: Optional[HyperparameterConfig], optional
    :param device: Device to run the algorithm on, defaults to "cpu".
    :type device: Union[str, torch.device], optional
    :param accelerator: Accelerator object for distributed computing, defaults to None.
    :type accelerator: Optional[Accelerator], optional
    :param torch_compiler: The torch compiler mode to use, defaults to None.
    :type torch_compiler: Optional[Any], optional
    :param name: Name of the algorithm, defaults to the class name.
    :type name: Optional[str], optional
    """

    def __init__(
        self,
        index: int,
        hp_config: Optional[HyperparameterConfig] = None,
        device: Union[str, torch.device] = "cpu",
        accelerator: Optional[Accelerator] = None,
        torch_compiler: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> None:

        assert isinstance(index, int), "Agent index must be an integer."
        assert isinstance(device, (str, torch.device)), "Device must be a string."
        assert isinstance(name, (type(None), str)), "Name must be a string."
        assert isinstance(
            accelerator, (type(None), Accelerator)
        ), "Accelerator must be an instance of Accelerator."
        if torch_compiler:
            assert torch_compiler in [
                "default",
                "reduce-overhead",
                "max-autotune",
            ], "Choose between torch compiler modes: default, reduce-overhead, max-autotune or None"

        self.accelerator = accelerator
        self.device = device if self.accelerator is None else self.accelerator.device
        self.torch_compiler = torch_compiler
        self.algo = name if name is not None else self.__class__.__name__

        self._mut = None
        self._index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]
        self.registry = MutationRegistry(hp_config)
        self.training = True

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
    def learn(self, experiences: ExperiencesType, **kwargs) -> Any:
        """Abstract method for learning the algorithm."""
        raise NotImplementedError

    @abstractmethod
    def get_action(
        self, obs: Union[ObservationType, MultiAgentObservationType], *args, **kwargs
    ) -> ActionType:
        """Abstract method for getting an action from the algorithm.

        :param obs: The observation to get an action for.
        :type obs: Union[ObservationType, MultiAgentObservationType]
        :param args: Additional arguments to pass to the action function.
        :type args: Any
        :param kwargs: Additional keyword arguments to pass to the action function.
        :type kwargs: Any
        :return: The action to take.
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, *args, **kwargs) -> ArrayLike:
        """Abstract method for testing the algorithm."""
        raise NotImplementedError

    @staticmethod
    def get_state_dim(observation_space: GymSpaceType) -> Tuple[int, ...]:
        """Returns the dimension of the state space as it pertains to the underlying
        networks (i.e. the input size of the networks).

        :param observation_space: The observation space of the environment.
        :type observation_space: spaces.Space or List[spaces.Space].

        :return: The dimension of the state space.
        :rtype: Tuple[int, ...].
        """
        warnings.warn(
            "This method is deprecated. Use get_input_size_from_space instead.",
            category=DeprecationWarning,
        )
        return get_input_size_from_space(observation_space)

    @staticmethod
    def get_action_dim(action_space: GymSpaceType) -> Tuple[int, ...]:
        """Returns the dimension of the action space as it pertains to the underlying
        networks (i.e. the output size of the networks).

        :param action_space: The action space of the environment.
        :type action_space: spaces.Space or List[spaces.Space].

        :return: The dimension of the action space.
        :rtype: int.
        """
        warnings.warn(
            "This method is deprecated. Use get_output_size_from_space instead.",
            category=DeprecationWarning,
        )
        return get_output_size_from_space(action_space)

    @staticmethod
    def inspect_attributes(
        agent: SelfEvolvableAlgorithm, input_args_only: bool = False
    ) -> Dict[str, Any]:
        """
        Inspect and retrieve the attributes of the current object, excluding attributes related to the
        underlying evolvable networks (i.e. `EvolvableModule`, `torch.optim.Optimizer`) and with
        an option to include only the attributes that are input arguments to the constructor.

        :param input_args_only: If True, only include attributes that are input arguments to the constructor.
                                Defaults to False.
        :type input_args_only: bool
        :return: A dictionary of attribute names and their values.
        :rtype: dict[str, Any]
        """
        # Get all attributes of the current object
        attributes = inspect.getmembers(agent, lambda a: not isroutine(a))

        # Exclude attributes that are EvolvableModule or Optimizer objects (also check for nested
        # module-related attributes for multi-agent algorithms)
        exclude = list(agent.evolvable_attributes().keys())
        exclude += [attr for attr, val in attributes if isinstance(val, TensorDict)]

        # Exclude private and built-in attributes
        attributes = [
            a for a in attributes if not (a[0].startswith("_") or a[0].endswith("_"))
        ]

        # If input_args_only is True, only include attributes that are
        # input arguments to the constructor
        if input_args_only:
            constructor_params = inspect.signature(agent.__init__).parameters.keys()
            attributes = {
                k: v
                for k, v in attributes
                if k not in exclude and k in constructor_params
            }
        else:
            # Remove the algo specific guarded variables (if specified)
            attributes = {k: v for k, v in attributes if k not in exclude}
        return attributes

    @staticmethod
    def copy_attributes(
        agent: SelfEvolvableAlgorithm, clone: SelfEvolvableAlgorithm
    ) -> SelfEvolvableAlgorithm:
        """Copies the non-evolvable attributes of the algorithm to a clone.

        :param clone: The clone of the algorithm.
        :type clone: SelfEvolvableAlgorithm

        :return: The clone of the algorithm.
        :rtype: SelfEvolvableAlgorithm
        """
        for attribute in EvolvableAlgorithm.inspect_attributes(agent).keys():
            if hasattr(agent, attribute) and hasattr(clone, attribute):
                attr, clone_attr = getattr(agent, attribute), getattr(clone, attribute)

                # NOTE: Here we handle the case where the individual is wrapped by an
                # AgentWrapper object, which includes the agent itself and functools.partial
                # objects as attributes that shouldn't be copied
                if callable(attr) or isinstance(attr, EvolvableAlgorithm):
                    continue
                elif isinstance(attr, torch.Tensor) or isinstance(
                    clone_attr, torch.Tensor
                ):
                    if not torch.equal(attr, clone_attr):
                        try:
                            setattr(
                                clone,
                                attribute,
                                copy.deepcopy(getattr(agent, attribute)),
                            )
                        except RuntimeError:
                            # If the tensor is not a leaf tensor, we need to clone it using torch.clone
                            setattr(
                                clone, attribute, torch.clone(getattr(agent, attribute))
                            )

                elif isinstance(attr, np.ndarray) or isinstance(clone_attr, np.ndarray):
                    if not np.array_equal(attr, clone_attr):
                        setattr(
                            clone, attribute, copy.deepcopy(getattr(agent, attribute))
                        )
                elif isinstance(attr, list) or isinstance(clone_attr, list):
                    setattr(clone, attribute, [copy.deepcopy(el) for el in attr])
                elif isinstance(attr, dict) or isinstance(clone_attr, dict):
                    setattr(
                        clone,
                        attribute,
                        {key: copy.deepcopy(value) for key, value in attr.items()},
                    )
                elif attr != clone_attr or isinstance(attr, MutationRegistry):
                    setattr(clone, attribute, copy.deepcopy(getattr(agent, attribute)))
            else:
                setattr(clone, attribute, copy.deepcopy(getattr(agent, attribute)))
        return clone

    @classmethod
    def population(
        cls: Type[SelfEvolvableAlgorithm],
        size: int,
        observation_space: GymSpaceType,
        action_space: GymSpaceType,
        wrapper_cls: Optional[Type[SelfAgentWrapper]] = None,
        wrapper_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> List[Union[SelfEvolvableAlgorithm, SelfAgentWrapper]]:
        """Creates a population of algorithms.

        :param size: The size of the population.
        :type size: int.

        :return: A list of algorithms.
        :rtype: List[SelfEvolvableAlgorithm].
        """
        if wrapper_cls is not None:
            return [
                wrapper_cls(
                    cls(observation_space, action_space, index=i, **kwargs),
                    **wrapper_kwargs,
                )
                for i in range(size)
            ]

        return [
            cls(observation_space, action_space, index=i, **kwargs) for i in range(size)
        ]

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets the attribute of the algorithm. If the attribute is an OptimizerWrapper,
        we register the optimizer with the algorithms registry.

        :param name: The name of the attribute.
        :type name: str
        :param value: The value of the attribute.
        :type value: Any
        """
        if isinstance(value, OptimizerWrapper) and name not in [
            config.name for config in self.registry.optimizers
        ]:
            config = OptimizerConfig(
                name=name,
                networks=value.network_names,
                lr=value.lr_name,
                optimizer_cls=value.optimizer_cls,
                optimizer_kwargs=value.optimizer_kwargs,
            )
            self.registry.register_optimizer(config)

        super().__setattr__(name, value)

    def _registry_init(self) -> None:
        """Registers the networks, optimizers, and algorithm hyperparameters in the algorithm with
        the mutations registry. We also check that all of the evolvable networks and their respective
        optimizers have been registered with the algorithm, and that the user-specified hyperparameters
        to mutate have been set as attributes in the algorithm."""

        if not self.registry.groups:
            raise AttributeError(
                "No network groups have been registered in the algorithms __init__ method. "
                "Please register NetworkGroup objects specifying all of the evaluation and "
                "shared/target networks through the `register_network_group()` method."
            )

        # Check that all the inspected evolvable attributes can be found in the registry
        all_registered = self.registry.all_registered()
        not_found = [
            attr for attr in self.evolvable_attributes() if attr not in all_registered
        ]
        if not_found:
            raise AttributeError(
                f"The following evolvable attributes could not be found in the registry: {not_found}. "
                "Please check that the defined NetworkGroup objects contain all of the EvolvableModule objects "
                "in the algorithm."
            )

        # Check that one of the network groups relates to a policy
        if not any(group.policy for group in self.registry.groups):
            raise AttributeError(
                "No network group has been registered as a policy (i.e. the network used to "
                "select actions) in the registry. Please register a NetworkGroup object "
                "specifying the policy network."
            )

        # Check that all the hyperparameters to mutate have been set as attributes in the algorithm
        if self.registry.hp_config is not None:
            for hp in self.registry.hp_config:
                if not hasattr(self, hp):
                    raise AttributeError(
                        f"Hyperparameter {hp} was found in the mutations configuration but has "
                        "not been set as an attribute in the algorithm."
                    )

    def _wrap_attr(self, attr: EvolvableAttributeType) -> EvolvableAttributeType:
        """Wraps the model with the accelerator.

        :param attr: The attribute to wrap.
        :type attr: EvolvableAttributeType

        :return: The wrapped attribute.
        :rtype: EvolvableAttributeType
        """
        if isinstance(attr, OptimizerWrapper):
            if isinstance(attr.optimizer, dict):
                wrapped_opt = {
                    agent_id: self.accelerator.prepare(opt)
                    for agent_id, opt in attr.optimizer.items()
                }
            else:
                wrapped_opt = self.accelerator.prepare(attr.optimizer)

            attr.optimizer = wrapped_opt
            return attr

        # Only wrap the model if its part of the computation graph
        return self.accelerator.prepare(attr) if attr.state_dict() else attr

    def set_training_mode(self, training: bool) -> None:
        """Sets the training mode of the algorithm.

        :param training: If True, set the algorithm to training mode.
        :type training: bool
        """
        self.training = training

    def get_lr_names(self) -> List[str]:
        """Returns the learning rates of the algorithm."""
        return [opt.lr for opt in self.registry.optimizers]

    def register_network_group(self, group: NetworkGroup) -> None:
        """Sets the evaluation network for the algorithm.

        :param name: The name of the evaluation network.
        :type name: str
        """
        self.registry.register_group(group)

    def register_mutation_hook(self, hook: Callable) -> None:
        """Registers a hook to be executed after a mutation is performed on
        the algorithm.

        :param hook: The hook to be executed after mutation.
        :type hook: Callable
        """
        self.registry.register_hook(hook)

    def mutation_hook(self) -> None:
        """Executes the hooks registered with the algorithm."""
        for hook in self.registry.hooks:
            getattr(self, hook)()

    def get_policy(self) -> EvolvableModule:
        """Returns the policy network of the algorithm."""
        for group in self.registry.groups:
            if group.policy:
                return getattr(self, group.eval_network)

        raise AttributeError(
            "No policy network has been registered with the algorithm."
        )

    def recompile(self) -> None:
        """Recompiles the evolvable modules in the algorithm with the specified torch compiler."""
        for name, obj in self.evolvable_attributes(networks_only=True).items():
            setattr(self, name, compile_model(obj, self.torch_compiler))

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
            elif isinstance(exp, (list, tuple)) and isinstance(exp[0], torch.Tensor):
                exp = tuple(val.to(device) for val in exp)
            elif isinstance(exp, torch.Tensor):
                exp = exp.to(device)

            on_device.append(exp)

        return on_device

    def evolvable_attributes(
        self, networks_only: bool = False
    ) -> EvolvableAttributeDict:
        """Returns the attributes related to the evolvable networks in the algorithm. Includes
        attributes that are either EvolvableModule or ModuleDict objects, as well as the optimizers
        associated with the networks.

        :param networks_only: If True, only include evolvable networks, defaults to False
        :type networks_only: bool, optional

        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """

        def is_evolvable(attr: str, obj: Any):
            return (
                recursive_check_module_attrs(obj, networks_only)
                and not attr.startswith("_")
                and not attr.endswith("_")
            )

        # Inspect evolvable given specs
        evolvable_attrs = {}
        for attr in dir(self):
            obj = getattr(self, attr)
            if is_evolvable(attr, obj):
                evolvable_attrs[attr] = obj

        return evolvable_attrs

    def wrap_models(self) -> None:
        """Wraps the models in the algorithm with the accelerator."""
        if self.accelerator is None:
            return

        for attr in self.evolvable_attributes():
            obj = getattr(self, attr)
            if isinstance(obj, dict):
                wrapped_obj = {
                    agent_id: self._wrap_attr(opt) for agent_id, opt in obj.items()
                }
            else:
                wrapped_obj = self._wrap_attr(obj)

            setattr(self, attr, wrapped_obj)

    def unwrap_models(self) -> None:
        """Unwraps the models in the algorithm from the accelerator."""
        if self.accelerator is None:
            raise AttributeError("No accelerator has been set for the algorithm.")

        for attr in self.evolvable_attributes(networks_only=True):
            obj = getattr(self, attr)
            if isinstance(obj, dict):
                unwrapped_obj = {
                    agent_id: self.accelerator.unwrap_model(opt)
                    for agent_id, opt in obj.items()
                }
            else:
                unwrapped_obj = self.accelerator.unwrap_model(obj)

            setattr(self, attr, unwrapped_obj)

    def clone(
        self: SelfEvolvableAlgorithm, index: Optional[int] = None, wrap: bool = True
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
        input_args = EvolvableAlgorithm.inspect_attributes(self, input_args_only=True)
        input_args["wrap"] = wrap

        clone = type(self)(**input_args)

        if self.accelerator is not None:
            self.unwrap_models()

        # Clone evolvable modules
        cloned_modules = {}
        for attr, obj in self.evolvable_attributes(networks_only=True).items():
            cloned_modules[attr] = obj.clone()
            setattr(clone, attr, cloned_modules[attr])

        # Run mutation hook at this step given possibility of sharing
        # encoder parameters between networks
        clone.mutation_hook()

        # Reinitialize optimizers
        for opt_config in self.registry.optimizers:
            orig_optimizer: OptimizerWrapper = getattr(self, opt_config.name)

            networks = [cloned_modules[net] for net in opt_config.networks]
            opt = OptimizerWrapper(
                getattr(torch.optim, opt_config.optimizer_cls),
                networks=networks,
                lr=orig_optimizer.lr,
                network_names=opt_config.networks,
                lr_name=opt_config.lr,
                optimizer_kwargs=opt_config.optimizer_kwargs,
            )
            opt.load_state_dict(orig_optimizer.state_dict())
            setattr(clone, opt_config.name, opt)

        # Prepare with accelerator / compiler if necessary
        if self.accelerator is not None and wrap:
            clone.wrap_models()
        elif self.torch_compiler:
            torch.set_float32_matmul_precision("high")
            clone.recompile()

        # Copy non-evolvable attributes back to clone
        clone = EvolvableAlgorithm.copy_attributes(self, clone)
        if index is not None:
            clone.index = index

        return clone

    def save_checkpoint(self, path: str) -> None:
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        torch.save(
            get_checkpoint_dict(self),
            path,
            pickle_module=dill,
        )

    def load_checkpoint(self, path: str) -> None:
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint: Dict[str, Any] = torch.load(
            path, map_location=self.device, pickle_module=dill, weights_only=False
        )

        # Recreate evolvable modules
        network_info: Dict[str, Dict[str, Any]] = checkpoint["network_info"]
        network_names = network_info["network_names"]
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info["modules"].items() if k.startswith(name)
            }

            module_cls = net_dict[f"{name}_cls"]
            init_dict = net_dict[f"{name}_init_dict"]

            module_dict_cls = net_dict.get(f"{name}_module_dict_cls", None)
            if isinstance(module_cls, dict):
                loaded_modules = {}
                for agent_id, mod in module_cls.items():
                    init_dict[agent_id]["device"] = self.device
                    loaded_modules[agent_id] = mod(**init_dict[agent_id])

                setattr(self, name, module_dict_cls(loaded_modules))
            else:
                init_dict["device"] = self.device
                loaded_module: EvolvableModule = module_cls(**init_dict)
                setattr(self, name, loaded_module)

        # Apply mutation hooks
        # NOTE: We do this before loading the state dicts because there may be
        # hooks that pertain to the network parameters such as e.g. encoder parameter
        # sharing
        self.mutation_hook()

        # Load state dicts after applying mutation hook
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info["modules"].items() if k.startswith(name)
            }
            loaded_module = getattr(self, name)
            state_dict = net_dict[f"{name}_state_dict"]
            if isinstance(loaded_module, ModuleDict):
                for agent_id, mod in loaded_module.items():
                    if state_dict[agent_id]:
                        mod.load_state_dict(state_dict[agent_id])

            elif state_dict:
                loaded_module.load_state_dict(state_dict)

        optimizer_names = network_info["optimizer_names"]
        for name in optimizer_names:
            opt_dict = {
                k: v
                for k, v in network_info["optimizers"].items()
                if k.startswith(name)
            }

            # Initialize optimizer
            opt_kwargs = opt_dict[f"{name}_kwargs"]
            optimizer_cls = get_optimizer_cls(opt_dict[f"{name}_cls"])
            opt_networks = opt_dict[f"{name}_networks"]
            opt_lr = opt_dict[f"{name}_lr"]
            networks = [getattr(self, net) for net in opt_networks]

            optimizer = OptimizerWrapper(
                optimizer_cls=optimizer_cls,
                networks=networks,
                lr=getattr(self, opt_lr),
                optimizer_kwargs=opt_kwargs,
                network_names=opt_networks,
                lr_name=opt_lr,
            )

            # Load optimizer state
            optimizer.load_state_dict(opt_dict[f"{name}_state_dict"])
            setattr(self, name, optimizer)

        # Check loaded registry is consistent with the algorithm
        if checkpoint["registry"] != self.registry:
            raise ValueError(
                "Loaded registry does not match the algorithm's registry. Please make "
                "sure you are loading the checkpoint with the correct algorithm."
            )

        # Load other attributes
        checkpoint.pop("network_info")
        for attribute in checkpoint.keys():
            setattr(self, attribute, checkpoint[attribute])

        # Wrap models / compile if necessary
        if self.accelerator is not None:
            self.wrap_models()
        elif self.torch_compiler:
            torch.set_float32_matmul_precision("high")
            self.recompile()

    @classmethod
    def load(
        cls: Type[SelfEvolvableAlgorithm],
        path: str,
        device: DeviceType = "cpu",
        accelerator: Optional[Accelerator] = None,
    ) -> SelfEvolvableAlgorithm:
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
            path, map_location=device, pickle_module=dill, weights_only=False
        )

        # Reconstruct evolvable modules in algorithm
        network_info: Optional[Dict[str, Dict[str, Any]]] = checkpoint.get(
            "network_info"
        )
        if network_info is None:
            raise ValueError(
                "Network info not found in checkpoint. You may be loading a checkpoint from "
                "an older version of AgileRL. Since v2.0, we require AgileRL algorithms to "
                "have a specific structure to simplify evolutionary hyperparameter optimization. "
                "Please downgrade to v1.0.30 to load checkpoints from before this change."
            )

        network_names = network_info["network_names"]
        loaded_modules: Dict[str, EvolvableAttributeType] = {}
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info["modules"].items() if k.startswith(name)
            }

            # Add device to init dict
            init_dict = net_dict.get(f"{name}_init_dict", None)
            if init_dict is None:
                raise ValueError(f"Init dict for {name} not found in checkpoint.")

            init_dict = chkpt_attribute_to_device(init_dict, device)

            # Reconstruct the module dict class if necessary
            ModuleDictCls = net_dict.get(f"{name}_module_dict_cls", None)
            if ModuleDictCls is not None:
                loaded_modules[name] = ModuleDictCls()

            # Reconstruct the modules
            module_cls: Union[
                Type[EvolvableModule], Dict[str, Type[EvolvableModule]]
            ] = net_dict[f"{name}_cls"]
            if isinstance(module_cls, dict):
                for agent_id, mod_cls in module_cls.items():
                    d = init_dict[agent_id]
                    d["device"] = device
                    mod: EvolvableModule = mod_cls(**d)
                    loaded_modules[name][agent_id] = mod
            else:
                init_dict["device"] = device
                module = module_cls(**init_dict)
                loaded_modules[name] = module

        # Reconstruct the algorithm
        constructor_params = inspect.signature(cls.__init__).parameters.keys()
        checkpoint["accelerator"] = accelerator
        checkpoint["device"] = device
        class_init_dict = {
            k: v for k, v in checkpoint.items() if k in constructor_params
        }
        self = cls(**class_init_dict)
        registry: MutationRegistry = checkpoint["registry"]
        self.registry = registry

        # Set loaded modules
        for name, module in loaded_modules.items():
            setattr(self, name, module)

        # Apply mutation hooks
        self.mutation_hook()

        # Load state dictionaries
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info["modules"].items() if k.startswith(name)
            }
            loaded_module: Union[EvolvableModule, ModuleDict] = getattr(self, name)
            state_dict = net_dict[f"{name}_state_dict"]
            if isinstance(loaded_module, ModuleDict):
                for agent_id, agent_module in loaded_module.items():
                    agent_state_dict = state_dict[agent_id]
                    if agent_state_dict:
                        agent_module.load_state_dict(agent_state_dict)

            elif state_dict:
                loaded_module.load_state_dict(state_dict)

        # Reconstruct optimizers in algorithm
        optimizer_names = network_info["optimizer_names"]
        loaded_optimizers = {}
        for name in optimizer_names:
            opt_dict = {
                k: v
                for k, v in network_info["optimizers"].items()
                if k.startswith(name)
            }

            # Add device to optimizer kwargs
            opt_kwargs = chkpt_attribute_to_device(opt_dict[f"{name}_kwargs"], device)
            lr = opt_dict[f"{name}_lr"]
            optimizer_cls = get_optimizer_cls(opt_dict[f"{name}_cls"])
            opt_networks = opt_dict[f"{name}_networks"]
            networks = [loaded_modules[net] for net in opt_networks]

            optimizer = OptimizerWrapper(
                optimizer_cls=optimizer_cls,
                networks=networks,
                lr=getattr(self, lr),
                network_names=opt_networks,
                lr_name=lr,
                optimizer_kwargs=opt_kwargs,
            )

            state_dict = chkpt_attribute_to_device(
                opt_dict[f"{name}_state_dict"], device
            )
            optimizer.load_state_dict(state_dict)
            loaded_optimizers[name] = optimizer

        # Assign loaded modules and optimizers to the algorithm
        for name, module in loaded_modules.items():
            setattr(self, name, module)

        for name, optimizer in loaded_optimizers.items():
            setattr(self, name, optimizer)

        # Assign other attributes to the algorithm
        for attribute in EvolvableAlgorithm.inspect_attributes(self).keys():
            if attribute not in checkpoint:
                warnings.warn(
                    f"Attribute {attribute} not found in checkpoint. Skipping."
                )
                continue

            setattr(self, attribute, checkpoint.get(attribute))

        # Wrap models / compile if necessary
        if accelerator is not None:
            self.wrap_models()
        elif self.torch_compiler:
            torch.set_float32_matmul_precision("high")
            self.recompile()

        # Check for agent wrapper
        wrapper_cls = checkpoint.get("wrapper_cls")
        if wrapper_cls is not None:
            init_dict = checkpoint.get("wrapper_init_dict")
            wrapper_attributes = checkpoint.get("wrapper_attrs")
            self = wrapper_cls(self, **init_dict)
            for attr in wrapper_attributes:
                setattr(self, attr, wrapper_attributes[attr])

        return self


class RLAlgorithm(EvolvableAlgorithm, ABC):
    """Base object for all single-agent algorithms in the AgileRL framework.

    :param observation_space: The observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: The action space of the environment.
    :type action_space: spaces.Space
    :param index: The index of the individual.
    :type index: int
    :param learn_step: Learning frequency, defaults to 2048.
    :type learn_step: int, optional
    :param device: Device to run the algorithm on, defaults to "cpu".
    :type device: Union[str, torch.device], optional
    :param accelerator: Accelerator object for distributed computing, defaults to None.
    :type accelerator: Optional[Accelerator], optional
    :param normalize_images: If True, normalize images, defaults to True.
    :type normalize_images: bool, optional
    :param name: Name of the algorithm, defaults to the class name.
    :type name: Optional[str], optional
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        index: int,
        hp_config: Optional[HyperparameterConfig] = None,
        device: Union[str, torch.device] = "cpu",
        accelerator: Optional[Accelerator] = None,
        torch_compiler: Optional[Any] = None,
        normalize_images: bool = True,
        name: Optional[str] = None,
    ) -> None:

        super().__init__(index, hp_config, device, accelerator, torch_compiler, name)

        assert isinstance(
            observation_space, spaces.Space
        ), "Observation space must be an instance of gymnasium.spaces.Space."
        assert isinstance(
            action_space, spaces.Space
        ), "Action space must be an instance of gymnasium.spaces.Space."

        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images
        self.action_dim = get_output_size_from_space(self.action_space)

    def preprocess_observation(self, observation: ObservationType) -> TorchObsType:
        """Preprocesses observations for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: ObservationType

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or Tuple[torch.Tensor[float], ...]
        """
        return preprocess_observation(
            self.observation_space,
            observation=observation,
            device=self.device,
            normalize_images=self.normalize_images,
        )


class MultiAgentRLAlgorithm(EvolvableAlgorithm, ABC):
    """Base object for all multi-agent algorithms in the AgileRL framework.

    :param observation_spaces: The observation spaces of the agent environments.
    :type observation_spaces: Union[List[spaces.Space], spaces.Dict]
    :param action_spaces: The action spaces of the agent environments.
    :type action_spaces: Union[List[spaces.Space], spaces.Dict]
    :param index: The index of the individual in the population.
    :type index: int.
    :param agent_ids: The agent IDs of the agents in the environment.
    :type agent_ids: Optional[List[int]], optional
    :param learn_step: Learning frequency, defaults to 2048
    :type learn_step: int, optional
    :param device: Device to run the algorithm on, defaults to "cpu"
    :type device: str, optional
    :param accelerator: Accelerator object for distributed computing, defaults to None
    :type accelerator: Optional[Accelerator], optional
    :param torch_compiler: The torch compiler mode to use, defaults to None
    :type torch_compiler: Optional[Any], optional
    :param normalize_images: If True, normalize images, defaults to True
    :type normalize_images: bool, optional
    :param placeholder_value: The value to use as placeholder for missing observations, defaults to -1.
    :type placeholder_value: Optional[Any], optional
    :param name: Name of the algorithm, defaults to the class name
    :type name: Optional[str], optional
    """

    possible_observation_spaces: Dict[str, spaces.Space]
    possible_action_spaces: Dict[str, spaces.Space]

    shared_agent_ids: List[str]
    grouped_agents: Dict[str, List[str]]
    unique_observation_spaces: Dict[str, spaces.Space]
    unique_action_spaces: Dict[str, spaces.Space]

    def __init__(
        self,
        observation_spaces: Union[Iterable[spaces.Space], spaces.Dict],
        action_spaces: Union[Iterable[spaces.Space], spaces.Dict],
        index: int,
        agent_ids: Optional[Iterable[int]] = None,
        hp_config: Optional[HyperparameterConfig] = None,
        device: Union[str, torch.device] = "cpu",
        accelerator: Optional[Accelerator] = None,
        torch_compiler: Optional[Any] = None,
        normalize_images: bool = True,
        placeholder_value: Optional[Any] = -1,
        name: Optional[str] = None,
    ) -> None:

        super().__init__(index, hp_config, device, accelerator, torch_compiler, name)

        assert type(observation_spaces) is type(action_spaces), (
            "Observation spaces and action spaces must be the same type. "
            f"Got {type(observation_spaces)} and {type(action_spaces)}."
        )

        if isinstance(observation_spaces, (list, tuple)):
            assert isinstance(
                agent_ids, (tuple, list)
            ), "Agent IDs must be specified if observation spaces are passed as a list."
            assert len(agent_ids) == len(
                observation_spaces
            ), "Number of agent IDs must match number of observation spaces."
            assert all(
                isinstance(_space, spaces.Space) for _space in observation_spaces
            ), "Observation spaces must be instances of gymnasium.spaces.Space."
            assert all(
                isinstance(_space, spaces.Space) for _space in action_spaces
            ), "Action spaces must be instances of gymnasium.spaces.Space."
            self.possible_observation_spaces = spaces.Dict(
                {
                    agent_id: space
                    for agent_id, space in zip(agent_ids, observation_spaces)
                }
            )
            self.possible_action_spaces = spaces.Dict(
                {agent_id: space for agent_id, space in zip(agent_ids, action_spaces)}
            )
        elif isinstance(observation_spaces, (spaces.Dict, dict)):
            if isinstance(observation_spaces, dict):
                observation_spaces = spaces.Dict(observation_spaces)
                action_spaces = spaces.Dict(action_spaces)

            self.possible_observation_spaces = observation_spaces
            self.possible_action_spaces = action_spaces
        else:
            raise ValueError(
                f"Observation spaces must be a list or dictionary of spaces.Space objects. Got {type(observation_spaces)}."
            )

        self.agent_ids = agent_ids or list(self.possible_observation_spaces.keys())
        self.n_agents = len(self.agent_ids)
        self.placeholder_value = placeholder_value
        self.normalize_images = normalize_images

        # These attributes are deprecated and will be removed in the future
        self.observation_spaces = list(self.possible_observation_spaces.values())
        self.action_spaces = list(self.possible_action_spaces.values())

        self.action_dims = get_output_size_from_space(self.possible_action_spaces)

        # Determine groups of agents from their IDs
        self.shared_agent_ids = []
        self.grouped_agents = defaultdict(list)
        self.unique_observation_spaces = OrderedDict()
        self.unique_action_spaces = OrderedDict()
        for agent_id in self.agent_ids:
            obs_space = self.possible_observation_spaces[agent_id]
            action_space = self.possible_action_spaces[agent_id]
            # Split agent names on expected pattern of e.g. speaker_0, speaker_1,
            # listener_0, listener_1, to determine which agents are homogeneous
            group_id = self.get_group_id(agent_id)
            if group_id not in self.grouped_agents:
                self.shared_agent_ids.append(group_id)
                self.unique_observation_spaces[group_id] = obs_space
                self.unique_action_spaces[group_id] = action_space

            assert obs_space == self.unique_observation_spaces[group_id], (
                f"Homogeneous agents, i.e. agents that share the prefix {group_id}, "
                f"must have the same observation space. Found {self.unique_observation_spaces[group_id]} and {obs_space}."
            )
            assert action_space == self.unique_action_spaces[group_id], (
                f"Homogeneous agents, i.e. agents that share the prefix {group_id}, "
                f"must have the same action space. Found {self.unique_action_spaces[group_id]} and {action_space}."
            )

            self.grouped_agents[group_id].append(agent_id)

        self.n_unique_agents = len(self.shared_agent_ids)

        # Dictionary containing groups of agents for each space type
        self.grouped_spaces = defaultdict(list)
        for agent_id in self.agent_ids:
            obs_space = self.possible_observation_spaces[agent_id]
            if is_vector_space(obs_space):
                self.grouped_spaces[ModuleType.MLP].append(agent_id)
            elif is_image_space(obs_space):
                self.grouped_spaces[ModuleType.CNN].append(agent_id)
            elif isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
                self.grouped_spaces[ModuleType.MULTI_INPUT].append(agent_id)
            else:
                raise ValueError(f"Unknown observation space type: {type(obs_space)}")

        self.setup = self.get_setup()

        # Build observation space based on setup
        if self.has_grouped_agents():
            self.observation_space = self.unique_observation_spaces
            self.action_space = self.unique_action_spaces
        else:
            self.observation_space = self.possible_observation_spaces
            self.action_space = self.possible_action_spaces

    def _registry_init(self) -> None:
        super()._registry_init()

        # Additional check to ensure multi-agent networks are initialized with valid keys
        for name, network in self.evolvable_attributes(networks_only=True).items():
            if isinstance(network, ModuleDict):
                for key in network.keys():
                    if (key not in self.agent_ids) and (
                        key not in self.shared_agent_ids
                    ):
                        raise ValueError(
                            f"Network '{name}' contains key '{key}' which is not present in `self.agent_ids` "
                            f"or `self.shared_agent_ids`. Please initialize multi-agent networks through agilerl.modules.ModuleDict "
                            "objects with the agent or group/shared IDs as keys."
                        )

    def has_grouped_agents(self) -> bool:
        """Whether the algorithm contains groups of agents assigned to the same
        policy for centralized execution.

        :rtype: bool
        """
        return len(self.shared_agent_ids) < len(self.agent_ids)

    def get_setup(self) -> MultiAgentSetup:
        """Get the type of multi-agent setup, as determined by the observation spaces of the agents.
        By having the 'same' observation space, we mean that the spaces are analogous, i.e. we can use
        the same `EvolvableModule` to process their observations.

        1. HOMOGENEOUS: All agents have the same observation space.
        2. MIXED: Agents can be grouped by their observation spaces.
        3. HETEROGENEOUS: All agents have different observation spaces.

        :return: The type of multi-agent setup.
        :rtype: MultiAgentSetup
        """
        return (
            MultiAgentSetup.HOMOGENEOUS
            if len(self.grouped_spaces) == 1
            else (
                MultiAgentSetup.MIXED
                if len(self.grouped_spaces) < len(self.agent_ids)
                else MultiAgentSetup.HETEROGENEOUS
            )
        )

    def preprocess_observation(
        self, observation: ObservationType
    ) -> Dict[str, TorchObsType]:
        """Preprocesses observations for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or Tuple[torch.Tensor[float], ...]
        """
        preprocessed = {}
        for agent_id, agent_obs in observation.items():
            preprocessed[agent_id] = preprocess_observation(
                self.possible_observation_spaces.get(agent_id),
                observation=agent_obs,
                device=self.device,
                normalize_images=self.normalize_images,
                placeholder_value=self.placeholder_value,
            )

        return preprocessed

    def extract_action_masks(self, infos: InfosDict) -> ArrayDict:
        """Extract action masks from info dictionary

        :param infos: Info dict
        :type infos: Dict[str, Dict[...]]

        :return: Action masks
        :rtype: Dict[str, np.ndarray]
        """
        # Get dict of form {"agent_id" : [1, 0, 0, 0]...} etc
        action_masks = {
            agent: info.get("action_mask", None) if isinstance(info, dict) else None
            for agent, info in infos.items()
            if agent in self.agent_ids
        }

        return action_masks

    def extract_agent_masks(
        self, infos: Optional[InfosDict] = None
    ) -> Tuple[ArrayDict, ArrayDict]:
        """Extract env_defined_actions from info dictionary and determine agent masks

        :param infos: Info dict
        :type infos: Dict[str, Dict[...]]

        :return: Env defined actions and agent masks
        :rtype: Tuple[ArrayDict, ArrayDict]
        """
        # Deal with case of no env_defined_actions defined in the info dict
        # Deal with empty info dicts for each sub agent
        if (
            infos is None
            or not key_in_nested_dict(infos, "env_defined_actions")
            or all(not info for agent, info in infos.items() if agent in self.agent_ids)
        ):
            return None, None

        env_defined_actions = {
            agent: (
                info.get("env_defined_actions", None)
                if isinstance(info, dict)
                else None
            )
            for agent, info in infos.items()
            if agent in self.agent_ids
        }
        agent_masks = None
        if env_defined_actions is not None:
            agent_masks = {}
            for agent_id in env_defined_actions.keys():
                # Handle None if environment isn't vectorized
                if env_defined_actions[agent_id] is None:
                    if not isinstance(
                        self.possible_action_spaces[agent_id], spaces.Discrete
                    ):
                        nan_arr = np.empty(self.action_dims[agent_id])
                        nan_arr[:] = np.nan
                    else:
                        nan_arr = np.array([[np.nan]])

                    env_defined_actions[agent_id] = nan_arr

                # Handle discrete actions + env not vectorized
                if isinstance(env_defined_actions[agent_id], (int, float)):
                    env_defined_actions[agent_id] = np.array(
                        [[env_defined_actions[agent_id]]]
                    )

                # Ensure additional dimension is added in so shapes align for masking
                if len(env_defined_actions[agent_id].shape) == 1:
                    env_defined_actions[agent_id] = (
                        env_defined_actions[agent_id][:, np.newaxis]
                        if isinstance(
                            self.possible_action_spaces[agent_id], spaces.Discrete
                        )
                        else env_defined_actions[agent_id][np.newaxis, :]
                    )
                agent_masks[agent_id] = np.where(
                    np.isnan(env_defined_actions[agent_id]), 0, 1
                ).astype(bool)

        return env_defined_actions, agent_masks

    def build_net_config(
        self,
        net_config: Optional[NetConfigType] = None,
        flatten: bool = True,
        return_encoders: bool = False,
    ) -> Union[NetConfigType, Tuple[NetConfigType, Dict[str, NetConfigType]]]:
        """Extract an appropriate net config for each sub-agent from the passed net config dictionary. If
        grouped_agents is True, the net config will be built for the grouped agents i.e. through their
        common prefix in their agent_id, whenever the passed net config is None.

        .. note::
            If return_encoders is True, we return the encoder configs for each sub-agent. The only exception is
            for MLPs, where we only return the deepest architecture found. This is useful for algorithms
            with shared critics that process the observations of all agents, and therefore use an `EvolvableMultiInput`
            module to process the observations of all agents (assigning an encoder to each sub-agent and, optionally, a
            single `EvolvableMLP` to process the concatenated vector observations).

        :param net_config: Net config dictionary
        :type net_config: Optional[NetConfigType]
        :param flatten: Whether to return a net config for each possible sub-agent, even in grouped settings.
        :type flatten: bool, optional
        :param return_encoders: Whether to return the encoder configs for each sub-agent. Defaults to False.
        :type return_encoders: bool, optional
        :return: Net config dictionary for each sub-agent
        :rtype: NetConfigType
        """
        grouped_config = self.has_grouped_agents() and not flatten
        agent_ids = self.shared_agent_ids if grouped_config else self.agent_ids
        observation_spaces = (
            self.unique_observation_spaces
            if grouped_config
            else self.possible_observation_spaces
        )
        encoder_configs = OrderedDict()

        # Helper function to append unique configs to the unique_configs dictionary
        # -> Access to unique configs is relevant for algorithms with networks that process
        # multiple agents' observations (e.g. shared critic in MADDPG)
        def _add_to_encoder_configs(config: Dict[str, Any], agent_id: str = "") -> None:
            config = config_from_dict(config)
            config_key = "mlp_config" if isinstance(config, MlpNetConfig) else agent_id

            if config_key not in encoder_configs:
                encoder_configs[config_key] = asdict(config)

            # Update MLP config if new one has deeper architecture
            elif isinstance(config, MlpNetConfig) and len(config["hidden_size"]) > len(
                encoder_configs["mlp_config"]["hidden_size"]
            ):
                encoder_configs[config_key] = asdict(config)

        # Helper function to check if any agent ID exists in the net_config
        def _has_agent_ids(config: NetConfigType) -> bool:
            return any(
                (agent_id in self.agent_ids) or (agent_id in self.shared_agent_ids)
                for agent_id in config.keys()
            )

        # Helper function to get or create encoder config for an agent
        def _get_encoder_config(config: NetConfigType, agent_id: str) -> NetConfigType:
            encoder_config = config.get("encoder_config")
            simba = config.get("simba", False)
            if encoder_config is None:
                encoder_config = get_default_encoder_config(
                    observation_spaces[agent_id], simba
                )
                config["encoder_config"] = encoder_config

            return encoder_config

        # 1. net_config is None -> Automatically define an encoder for each sub-agent or group
        if net_config is None:
            net_config = defaultdict(OrderedDict)
            for agent_id in agent_ids:
                encoder_config = get_default_encoder_config(
                    observation_spaces[agent_id]
                )
                net_config[agent_id]["encoder_config"] = encoder_config
                _add_to_encoder_configs(encoder_config, agent_id)

            if return_encoders:
                return net_config, encoder_configs

            return net_config

        # 2a. (Legacy) -> Passed a single-level config in a multi-agent setting - can only
        # do this in homogeneous settings where all agents have the same observation space as
        # it pertains to the network (i.e. allow as long as the observation spaces result in the
        # same encoder)
        if not _has_agent_ids(net_config):
            assert self.setup == MultiAgentSetup.HOMOGENEOUS, (
                "Single-level net config can only be passed when the multi-agent environment is homogeneous "
                "(i.e. all agents can use the same encoder to process their observations). Please specify "
                "a net config for some combination of agents (or groups of agents) in the multi-agent environment."
            )

            encoder_config = _get_encoder_config(net_config, agent_ids[0])

            full_config = OrderedDict()
            for agent_id in agent_ids:
                # Create a copy of the config for each agent
                full_config[agent_id] = net_config.copy()

                if return_encoders:
                    _add_to_encoder_configs(encoder_config, agent_id)

            if return_encoders:
                return full_config, encoder_configs

            return full_config

        if any(
            agent_id in self.agent_ids and grouped_config
            for agent_id in net_config.keys()
        ):
            raise KeyError(
                "Found key in net_config corresponding to an individual sub-agent in a grouped setting. "
                "Please specify the configuration for groups instead (e.g. {'agent': {...}, ...} rather than {'agent_0': {...}, ...})"
            )

        # 2b. Handle nested config with agent/group IDs
        result_config = {}
        config_keys = net_config.keys()
        for agent_id in agent_ids:
            group_id = self.get_group_id(agent_id) if not grouped_config else agent_id

            # 2bi. Check if agent_id is present in net_config
            if agent_id in config_keys:
                agent_config = net_config[agent_id]
                encoder_config = _get_encoder_config(agent_config, agent_id)
                result_config[agent_id] = agent_config

            # 2bii. Check if group_id is present in net_config
            elif group_id in config_keys:
                group_config = net_config[group_id]
                encoder_config = _get_encoder_config(group_config, agent_id)
                result_config[agent_id] = group_config

            # 2biii. agent_id or group_id not in net_config -> Add default encoder config
            else:
                default_config = {}
                encoder_config = get_default_encoder_config(
                    observation_spaces[agent_id]
                )
                default_config["encoder_config"] = encoder_config
                result_config[agent_id] = default_config

            if return_encoders:
                _add_to_encoder_configs(encoder_config, agent_id)

        if return_encoders:
            return result_config, encoder_configs

        return result_config

    ####---------------------------------------####
    #### Grouped Multi-Agent Utility Functions ####
    ####---------------------------------------####
    def get_group_id(self, agent_id: str) -> str:
        """Get the group ID for an agent.

        :param agent_id: The agent ID
        :type agent_id: str
        :return: The group ID
        """
        return agent_id.rsplit("_", 1)[0] if isinstance(agent_id, str) else agent_id

    def assemble_shared_inputs(self, experience: ExperiencesType) -> ExperiencesType:
        """Preprocesses inputs by constructing dictionaries by shared agents.

        :param experience: experience to reshape from environment
        :type experience: ExperiencesType

        :return: Preprocessed inputs
        :rtype: ExperiencesType
        """
        stacked_experience = {group_id: {} for group_id in self.observation_space}
        for agent_id, inp in experience.items():
            group_id = (
                self.get_group_id(agent_id) if self.has_grouped_agents() else agent_id
            )
            if isinstance(inp, list):
                stacked_exp = (
                    stack_experiences(inp, to_torch=False)[0] if len(inp) > 0 else None
                )
            else:
                stacked_exp = inp

            stacked_experience[group_id][agent_id] = stacked_exp

        return stacked_experience

    def disassemble_grouped_outputs(
        self,
        group_outputs: ArrayDict,
        vect_dim: int,
        grouped_agents: Dict[str, List[str]],
    ) -> ArrayDict:
        """Disassembles batched output by shared policies into their grouped agents' outputs.

        .. note:: This assumes that for any given sub-agent the termination condition is deterministic,
            i.e. any given agent will always terminate at the same timestep in different vectorized environments.

        :param group_outputs: Dictionary to be disassembled, has the form {'agent': [4, 7, 8]}
        :type group_outputs: Dict[str, np.ndarray]
        :param vect_dim: Vectorization dimension size, i.e. number of vect envs
        :type vect_dim: int
        :param grouped_agents: Dictionary of grouped agent IDs
        :type grouped_agents: Dict[str, List[str]]
        :return: Assembled dictionary, e.g. {'agent_0': 4, 'agent_1': 7, 'agent_2': 8}
        :rtype: Dict[str, np.ndarray]
        """
        output_dict = {}
        for group_id, agent_ids in grouped_agents.items():
            group_outputs[group_id] = np.reshape(
                group_outputs[group_id],
                (len(agent_ids), vect_dim, -1),
            )
            for i, agent_id in enumerate(agent_ids):
                output_dict[agent_id] = group_outputs[group_id][i]

        return output_dict

    def sum_shared_rewards(self, rewards: ArrayDict) -> ArrayDict:
        """Sums the rewards for grouped agents

        :param rewards: Reward dictionary from environment
        :type rewards: Dict[str, np.ndarray]
        :return: Summed rewards dictionary
        :rtype: Dict[str, np.ndarray]
        """
        reward_shape = list(rewards.values())[0]
        reward_shape = (
            reward_shape.shape if isinstance(reward_shape, np.ndarray) else (1,)
        )
        summed_rewards = {
            agent_id: np.zeros(reward_shape) for agent_id in self.shared_agent_ids
        }
        for agent_id, reward in rewards.items():
            group_id = self.get_group_id(agent_id)
            summed_rewards[group_id] += reward

        return summed_rewards

    def assemble_grouped_outputs(
        self, agent_outputs: ArrayDict, vect_dim: int
    ) -> ArrayDict:
        """Assembles individual agent outputs into batched outputs for shared policies.

        :param agent_outputs: Dictionary with individual agent outputs, e.g. {'agent_0': 4, 'agent_1': 7, 'agent_2': 8}
        :type agent_outputs: Dict[str, np.ndarray]
        :param vect_dim: Vectorization dimension size, i.e. number of vect envs
        :type vect_dim: int
        :return: Assembled dictionary with the form {'agent': [4, 7, 8]}
        :rtype: Dict[str, np.ndarray]
        """
        group_outputs = {}
        for group_id in self.shared_agent_ids:
            # Get all outputs for agents that share this ID
            group_agent_outputs = []
            for group in self.grouped_agents[group_id]:
                if group in agent_outputs:
                    group_agent_outputs.append(agent_outputs[group])

            if group_agent_outputs:
                # Stack outputs along first dimension
                stacked_outputs = np.stack(group_agent_outputs, axis=0)
                # Reshape into a form suitable for batch processing
                group_outputs[group_id] = np.reshape(
                    stacked_outputs, (len(group_agent_outputs) * vect_dim, -1)
                )

        return group_outputs


class LLMAlgorithm(EvolvableAlgorithm, ABC):
    """Base object for all LLM algorithms in the AgileRL framework.

    :param observation_space: The observation space of the environment.
    :type observation_space: gymnasium.spaces.Space
    :param action_space: The action space of the environment.
    :type action_space: gymnasium.spaces.Space
    :param index: The index of the algorithm.
    :type index: int
    :param hp_config: The hyperparameter configuration.
    :type hp_config: Optional[HyperparameterConfig]
    :param device: The device to run the algorithm on.
    :type device: Union[str, torch.device]
    :param accelerator: The accelerator to use.
    :type accelerator: Optional[Accelerator]
    :param name: The name of the algorithm.
    :type name: Optional[str]
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        index: int,
        hp_config: Optional[HyperparameterConfig] = None,
        device: Union[str, torch.device] = "cpu",
        accelerator: Optional[Accelerator] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(index, hp_config, device, accelerator, None, name)
        assert isinstance(
            observation_space, spaces.Space
        ), "Observation space must be an instance of gymnasium.spaces.Space."
        assert isinstance(
            action_space, spaces.Space
        ), "Action space must be an instance of gymnasium.spaces.Space."

        self.observation_space = observation_space
        self.action_space = action_space
        self.zero_stage = None
        self.reference_update_tracker = 0  # Updated every time the reference policy is updated which is updated each time we pass through the train dataset

        if self.accelerator is not None:
            self.zero_stage = self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "zero_optimization"
            ]["stage"]
        if (
            self.zero_stage is not None
            and self.zero_stage > 2
            and self.accelerator.is_main_process
        ):
            warnings.warn(
                "Zero stage 3 support is nascent and has not been thoroughly tested. It may be unstable or subject to change. We recommend caution in production environments."
            )

        seed = 42
        if self.accelerator is not None:
            if accelerator.is_main_process:
                seed = np.random.randint(0, 2**32 - 1)
            if accelerator.num_processes > 1:
                seed = broadcast_object_list([seed], from_process=0)[0]
        self.rng = np.random.RandomState(seed)

    def preprocess_observation(self, observation: ObservationType) -> TorchObsType:
        """Dummy preprocesses observations for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or Tuple[torch.Tensor[float], ...]
        """
        return observation

    def save_checkpoint(self, path: str) -> None:
        """
        Override the save_checkpoint method to provide guidance on the correct method to use.
        :param path: Location to save checkpoint at
        :type path: string
        """
        raise NotImplementedError(
            "The save_checkpoint method is not supported for this algorithm class. "
            "Please use agent.actor.save_pretrained(checkpoint_path) instead."
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Override the load_checkpoint method to provide guidance on the correct method to use.

        :param path: Location to load checkpoint from
        :type path: string
        """
        raise NotImplementedError(
            "The load_checkpoint method is not supported for this algorithm class."
            """
            To load a saved LLM, please load the model as follows, and then re-instantiate the GRPO
            class.

            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
            model = PeftModel.from_pretrained(base_model, "/path/to/adapter/folder")
            """
        )

    def _save_distributed_actor(self, path: str) -> None:
        """
        Override the save_checkpoint method to provide guidance on the correct method to use.

        :param path: Output directory to save the checkpoint at
        :type path: str
        """
        if self.accelerator is not None:
            os.makedirs(path, exist_ok=True)
            self.actor.save_checkpoint(path, tag="checkpoint")
            self.actor.set_adapter("actor")
        else:
            warnings.warn(
                "Distributed actor save not supported for non-distributed training."
            )

    def _load_distributed_actor(self, path: str) -> None:
        """
        Override the load_checkpoint method to provide guidance on the correct method to use.

        :param path: Output directory to load the checkpoint from
        :type path: str
        """
        if self.accelerator is not None:
            deepspeed_dirs = sorted(glob.glob(f"{path}/checkpoint"))
            try:
                assert len(deepspeed_dirs) > 0
                load_path, _ = self.actor.load_checkpoint(
                    path,
                    tag="checkpoint",
                    load_module_strict=False,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                )
                if load_path is None:
                    raise ValueError(
                        f"Deepspeed failed to resume from checkpoint {path}"
                    )
                self.actor.set_adapter("actor")

            except Exception as e:
                raise ValueError(
                    f"Deepspeed failed to resume from checkpoint {path}"
                ) from e
        else:
            warnings.warn(
                "Distributed actor load not supported for non-distributed training."
            )

    @classmethod
    def load(
        cls,
        path: str,
        device: DeviceType = "cpu",
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        raise NotImplementedError(
            "The load class method is not supported for this algorithm class."
            """
            To load a saved LLM, please load the model as follows, and then re-instantiate the GRPO
            class, using the pre-trained model.

            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
            model = PeftModel.from_pretrained(base_model, "/path/to/adapter/folder")
            """
        )

    def wrap_models(self) -> None:
        """Wrap the models in the accelerator"""
        if self.accelerator is not None:
            opt = (
                self.optimizer.optimizer
                if self.optimizer.optimizer_cls == torch.optim.AdamW
                else self.optimizer
            )
            self.actor, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.actor, opt, self.lr_scheduler
            )
        else:
            self.actor = self.actor.to(self.device)
            self.actor.gradient_checkpointing_enable()

    def clean_up(self) -> None:
        """Clean up the algorithm."""

        if self.accelerator is not None:
            (
                self.actor,
                self.optimizer,
                self.lr_scheduler,
            ) = self.accelerator.free_memory(
                self.actor,
                self.optimizer,
                self.lr_scheduler,
            )
            self.accelerator.wait_for_everyone()
        else:
            (
                self.actor,
                self.optimizer,
                self.lr_scheduler,
            ) = (
                None,
                None,
                None,
            )
        gc.collect()
        torch.cuda.empty_cache()

    def clone(self, index: Optional[int] = None, wrap: bool = True):
        """Creates a clone of the algorithm.

        :param index: The index of the clone, defaults to None
        :type index: Optional[int], optional
        :param wrap: If True, wrap the models in the clone with the accelerator, defaults to False
        :type wrap: bool, optional

        :return: A clone of the algorithm
        :rtype: EvolvableAlgorithm
        """
        with tempfile.TemporaryDirectory() as temp_dir:

            # We need to use the same temp_dir for all processes, so we broadcast the temp_dir from the main process
            if self.accelerator is not None and self.accelerator.num_processes > 1:
                temp_dir = broadcast_object_list([temp_dir], from_process=0)[0]

            if self.zero_stage is not None and self.zero_stage >= 2:
                self.accelerator.wait_for_everyone()
                self._save_distributed_actor(f"{temp_dir}/agent_{self.index}")
                self.accelerator.wait_for_everyone()

            input_args = EvolvableAlgorithm.inspect_attributes(
                self, input_args_only=True
            )
            input_args["wrap"] = False
            input_args["clone"] = True

            actor = (
                self.accelerator.unwrap_model(self.actor)
                if self.accelerator is not None
                else self.actor
            )

            actor_state_dict = None
            if self.zero_stage is None or self.zero_stage < 2:
                actor_state_dict = clone_tensors_for_torch_save(actor.state_dict())

            cloned_model = clone_llm(actor, state_dict=actor_state_dict)

            actor = None  # De-reference the actor
            input_args["actor_network"] = cloned_model
            input_args["accelerator"] = (
                Accelerator() if self.accelerator is not None else None
            )

            clone = type(self)(**input_args)
            clone.mutation_hook()

            # Clone attributes
            accelerator = clone.accelerator
            lr_scheduler = clone.lr_scheduler
            cloned_lr_scheduler = clone.lr_scheduler
            original_lr_scheduler = self.lr_scheduler
            clone.lr_scheduler = None
            self.lr_scheduler = None
            clone = EvolvableAlgorithm.copy_attributes(self, clone)
            clone.accelerator = accelerator
            clone.lr_scheduler = lr_scheduler
            clone.lr_scheduler = cloned_lr_scheduler
            self.lr_scheduler = original_lr_scheduler

            if self.accelerator is None:
                clone.optimizer.optimizer.load_state_dict(
                    self.optimizer.optimizer.state_dict()
                )
                if self.lr_scheduler is not None:
                    clone.lr_scheduler.load_state_dict(self.lr_scheduler.state_dict())

            # Set the index
            if index is not None:
                clone.index = index

            clone.wrap_models()

            if self.zero_stage is not None and self.zero_stage >= 2:
                clone.accelerator.wait_for_everyone()
                clone._load_distributed_actor(f"{temp_dir}/agent_{self.index}")
                clone.accelerator.wait_for_everyone()
            else:
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

        return clone

    @staticmethod
    def update_lr(
        optimizer: DeepSpeedOptimizerWrapper,
        lr: float,
        accelerator: Optional[Accelerator] = None,
        scheduler_config: Optional[CosineLRScheduleConfig] = None,
    ) -> Tuple[Optional[Accelerator], Optional[SequentialLR]]:
        """Update the learning rate of the optimizer

        :param optimizer: Optimizer
        :type optimizer: Optimizer
        :param lr: Learning rate
        :type lr: float
        :param accelerator: Accelerator
        :type accelerator: Optional[Accelerator]
        :param scheduler_config: Scheduler configuration
        :type scheduler_config: Optional[CosineLRScheduleConfig]

        :return: Tuple of accelerator and scheduler
        :return: Accelerator
        """

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if accelerator is None:
            scheduler = (
                create_warmup_cosine_scheduler(optimizer, scheduler_config, 1e-8, lr)
                if scheduler_config is not None
                else None
            )
            return accelerator, scheduler

        if (
            accelerator.state.deepspeed_plugin.deepspeed_config.get("scheduler", None)
            is not None
        ):
            accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "warmup_max_lr"
            ] = lr

        if (
            accelerator.state.deepspeed_plugin.deepspeed_config.get("optimizer", None)
            is not None
        ):
            accelerator.deepspeed_plugin.deepspeed_config["optimizer"]["params"][
                "lr"
            ] = lr

        return accelerator, None
