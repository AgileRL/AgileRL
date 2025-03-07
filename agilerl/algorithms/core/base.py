import copy
import inspect
from abc import ABC, ABCMeta, abstractmethod
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
from gymnasium import spaces
from numpy.typing import ArrayLike
from tensordict import TensorDict
from torch._dynamo import OptimizedModule

from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    MutationRegistry,
    NetworkGroup,
    OptimizerConfig,
)
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.protocols import (
    AgentWrapper,
    EvolvableAttributeDict,
    EvolvableAttributeType,
    EvolvableModule,
)
from agilerl.typing import (
    ActionType,
    DeviceType,
    GymSpaceType,
    NumpyObsType,
    ObservationType,
    TorchObsType,
)
from agilerl.utils.algo_utils import (
    assert_supported_space,
    chkpt_attribute_to_device,
    compile_model,
    is_module_list,
    isroutine,
    preprocess_observation,
    recursive_check_module_attrs,
    remove_compile_prefix,
)
from agilerl.utils.evolvable_networks import is_image_space

__all__ = ["EvolvableAlgorithm", "RLAlgorithm", "MultiAgentRLAlgorithm"]

SelfEvolvableAlgorithm = TypeVar("SelfEvolvableAlgorithm", bound="EvolvableAlgorithm")
SelfRLAlgorithm = TypeVar("SelfRLAlgorithm", bound="RLAlgorithm")
SelfAgentWrapper = TypeVar("SelfAgentWrapper", bound="AgentWrapper")
MARLObservationType = Dict[str, ObservationType]


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

    # Extract info on evolvable modules and optimizers in the algorithm
    network_info: Dict[str, Dict[str, Any]] = {"modules": {}, "optimizers": {}}
    for attr in agent.evolvable_attributes():
        obj: EvolvableAttributeType = getattr(agent, attr)
        if isinstance(obj, OptimizerWrapper):
            network_info["optimizers"].update(
                {
                    f"{attr}_cls": obj.optimizer_cls.__name__,
                    f"{attr}_state_dict": obj.state_dict(),
                    f"{attr}_networks": obj.network_names,
                    f"{attr}_lr": obj.lr_name,
                    f"{attr}_kwargs": obj.optimizer_kwargs,
                    f"{attr}_multiagent": obj.multiagent,
                }
            )
        elif isinstance(obj, (OptimizedModule, EvolvableModule)) or is_module_list(obj):
            if is_module_list(obj):
                obj_list = obj
                obj_cls = [
                    (
                        m._orig_mod.__class__
                        if isinstance(m, OptimizedModule)
                        else m.__class__
                    )
                    for m in obj_list
                ]
                init_dict = [m.init_dict for m in obj_list]
                state_dict = [remove_compile_prefix(m.state_dict()) for m in obj_list]
            else:
                obj_list = [obj]
                obj_cls = (
                    obj._orig_mod.__class__
                    if isinstance(obj, OptimizedModule)
                    else obj.__class__
                )
                init_dict = obj.init_dict
                state_dict = remove_compile_prefix(obj.state_dict())

            network_info["modules"].update(
                {
                    f"{attr}_cls": obj_cls,
                    f"{attr}_init_dict": init_dict,
                    f"{attr}_state_dict": state_dict,
                }
            )
        else:
            raise TypeError(
                f"Something went wrong. Identified '{attr}' as an evolvable module "
                f"when it is of type {type(obj)}."
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
    return attribute_dict


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
    def learn(
        self, experiences: Tuple[Iterable[ObservationType], ...], **kwargs
    ) -> Any:
        """Abstract method for learning the algorithm."""
        raise NotImplementedError

    @abstractmethod
    def get_action(
        self, obs: Union[ObservationType, MARLObservationType], *args, **kwargs
    ) -> ActionType:
        """Abstract method for getting an action from the algorithm."""
        raise NotImplementedError

    @abstractmethod
    def test(self, *args, **kwargs) -> ArrayLike:
        """Abstract method for testing the algorithm."""
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
            return tuple(
                EvolvableAlgorithm.get_state_dim(space) for space in observation_space
            )
        elif isinstance(observation_space, spaces.Discrete):
            return (observation_space.n,)
        elif isinstance(observation_space, spaces.MultiDiscrete):
            return (sum(observation_space.nvec),)
        elif isinstance(observation_space, spaces.Box):
            return observation_space.shape
        elif isinstance(observation_space, spaces.Dict):
            assert_supported_space(observation_space)
            return {
                key: EvolvableAlgorithm.get_state_dim(subspace)
                for key, subspace in observation_space.spaces.items()
            }
        else:
            raise AttributeError(
                f"Can't access state dimensions for {type(observation_space)} spaces."
            )

    @staticmethod
    def get_action_dim(action_space: GymSpaceType) -> int:
        """Returns the dimension of the action space.

        :param action_space: The action space of the environment.
        :type action_space: spaces.Space or List[spaces.Space].

        :return: The dimension of the action space.
        :rtype: int.
        """
        if isinstance(action_space, (list, tuple)):
            return tuple(
                EvolvableAlgorithm.get_action_dim(space) for space in action_space
            )
        if isinstance(action_space, spaces.Discrete):
            return action_space.n
        elif isinstance(action_space, spaces.Box):
            # NOTE: Here we assume the action space only has one dimension
            #       (i.e. the actions correspond to a one-dimensional vector)
            return action_space.shape[0]
        else:
            raise AttributeError(
                f"Can't access action dimensions for {type(action_space)} spaces."
            )

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
                elif attr != clone_attr:
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
        wrapper_cls: Type[SelfAgentWrapper] = None,
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
                multiagent=value.multiagent,
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

    def _wrap_attr(self, attr: EvolvableAttributeType) -> EvolvableModule:
        """Wraps the model with the accelerator.

        :param attr: The attribute to wrap.
        :type attr: EvolvableAttributeType

        :return: The wrapped attribute.
        :rtype: EvolvableModule
        """
        if isinstance(attr, OptimizerWrapper):
            if isinstance(attr.optimizer, list):
                wrapped_opt = [self.accelerator.prepare(opt) for opt in attr.optimizer]
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

    def get_policy(self) -> EvolvableModule:
        """Returns the policy network of the algorithm."""
        for group in self.registry.groups:
            if group.policy:
                return getattr(self, group.eval)

        raise AttributeError(
            "No policy network has been registered with the algorithm."
        )

    def recompile(self) -> None:
        """Recompiles the evolvable modules in the algorithm with the specified torch compiler."""
        for name, obj in self.evolvable_attributes(networks_only=True).items():
            if isinstance(obj, list):
                compiled = [
                    compile_model(module, self.torch_compiler) for module in obj
                ]
            else:
                compiled = compile_model(obj, self.torch_compiler)

            setattr(self, name, compiled)

    def evolvable_attributes(
        self, networks_only: bool = False
    ) -> EvolvableAttributeDict:
        """Returns the attributes related to the evolvable networks in the algorithm. Includes
        attributes that are either evolvable networks or a list of evolvable networks, as well
        as the optimizers associated with the networks.

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
            if isinstance(obj, list):
                setattr(self, attr, [self._wrap_attr(m) for m in obj])
            else:
                setattr(self, attr, self._wrap_attr(obj))

    def unwrap_models(self):
        """Unwraps the models in the algorithm from the accelerator."""
        if self.accelerator is None:
            raise AttributeError("No accelerator has been set for the algorithm.")

        for attr in self.evolvable_attributes(networks_only=True):
            obj = getattr(self, attr)
            if isinstance(obj, list):
                setattr(self, attr, [self.accelerator.unwrap_model(m) for m in obj])
            else:
                setattr(self, attr, self.accelerator.unwrap_model(obj))

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
            if isinstance(obj, list):
                cloned_modules[attr] = [m.clone() for m in obj]
            else:
                cloned_modules[attr] = obj.clone()

            setattr(clone, attr, cloned_modules[attr])

        # Reinitialize optimizers
        for opt_config in self.registry.optimizers:
            orig_optimizer: OptimizerWrapper = getattr(self, opt_config.name)

            networks = (
                cloned_modules[opt_config.networks[0]]
                if opt_config.multiagent
                else [cloned_modules[net] for net in opt_config.networks]
            )

            opt = OptimizerWrapper(
                getattr(torch.optim, opt_config.optimizer_cls),
                networks=networks,
                lr=orig_optimizer.lr,
                network_names=opt_config.networks,
                lr_name=opt_config.lr,
                optimizer_kwargs=opt_config.optimizer_kwargs,
                multiagent=opt_config.multiagent,
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

        # Run init hooks
        for hook in clone.registry.hooks:
            getattr(clone, hook)()

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
            path, map_location=self.device, pickle_module=dill
        )

        # Recreate evolvable modules
        network_info: Dict[str, Dict[str, Any]] = checkpoint["network_info"]
        network_names = network_info["network_names"]
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info["modules"].items() if k.startswith(name)
            }

            module_cls = net_dict[f"{name}_cls"]
            state_dict = net_dict[f"{name}_state_dict"]
            init_dict = net_dict[f"{name}_init_dict"]
            if isinstance(module_cls, list):
                loaded_modules = []
                for mod, d, state in zip(module_cls, init_dict, state_dict):
                    loaded_mod: EvolvableModule = mod(**d)

                    if state:
                        loaded_mod.load_state_dict(state)

                    loaded_modules.append(loaded_mod)

                setattr(self, name, loaded_modules)
            else:
                loaded_module: EvolvableModule = module_cls(**init_dict)

                # NOTE: Sometimes we may have empty state dicts (e.g. detached modules)
                if state_dict:
                    loaded_module.load_state_dict(state_dict)

                setattr(self, name, loaded_module)

        optimizer_names = network_info["optimizer_names"]
        for name in optimizer_names:
            opt_dict = {
                k: v
                for k, v in network_info["optimizers"].items()
                if k.startswith(name)
            }

            # Initialize optimizer
            opt_kwargs = opt_dict[f"{name}_kwargs"]
            optimizer_cls = opt_dict[f"{name}_cls"]
            opt_networks = opt_dict[f"{name}_networks"]
            opt_lr = opt_dict[f"{name}_lr"]
            is_multiagent = opt_dict[f"{name}_multiagent"]
            networks = (
                getattr(self, opt_networks[0])
                if is_multiagent
                else [getattr(self, net) for net in opt_networks]
            )
            optimizer = OptimizerWrapper(
                getattr(torch.optim, optimizer_cls),
                networks=networks,
                lr=getattr(self, opt_lr),
                network_names=opt_networks,
                lr_name=opt_lr,
                optimizer_kwargs=opt_kwargs,
                multiagent=is_multiagent,
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

        # Init hooks
        for hook in self.registry.hooks:
            getattr(self, hook)()

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
            path, map_location=device, pickle_module=dill
        )

        # Reconstruct evolvable modules in algorithm
        print("Checkpoint: ", checkpoint)
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

            state_dict = net_dict.get(f"{name}_state_dict", None)
            if state_dict is None:
                raise ValueError(f"State dict for {name} not found in checkpoint.")

            state_dict = chkpt_attribute_to_device(state_dict, device)

            # Reconstruct the modules
            module_cls: Union[Type[EvolvableModule], List[Type[EvolvableModule]]] = (
                net_dict[f"{name}_cls"]
            )
            if isinstance(module_cls, list):
                loaded_modules[name] = []
                for mod_cls, d, state in zip(module_cls, init_dict, state_dict):
                    d["device"] = device
                    mod: EvolvableModule = mod_cls(**d)

                    if state:
                        mod.load_state_dict(state)

                    loaded_modules[name].append(mod)
            else:
                init_dict["device"] = device
                module = module_cls(**init_dict)

                if (
                    state_dict
                ):  # Sometimes we may have empty state dicts (e.g. detached modules)
                    module.load_state_dict(state_dict)

                loaded_modules[name] = module

        # Reconstruct the algorithm
        constructor_params = inspect.signature(cls.__init__).parameters.keys()
        class_init_dict = {
            k: v for k, v in checkpoint.items() if k in constructor_params
        }

        checkpoint["accelerator"] = accelerator
        checkpoint["device"] = device
        self = cls(**class_init_dict)
        registry: MutationRegistry = checkpoint["registry"]
        self.registry = registry

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
            optimizer_cls = opt_dict[f"{name}_cls"]
            opt_networks = opt_dict[f"{name}_networks"]

            networks = (
                loaded_modules[opt_networks[0]]
                if opt_dict[f"{name}_multiagent"]
                else [loaded_modules[net] for net in opt_networks]
            )

            optimizer = OptimizerWrapper(
                getattr(torch.optim, optimizer_cls),
                networks=networks,
                lr=getattr(self, lr),
                network_names=opt_networks,
                lr_name=lr,
                optimizer_kwargs=opt_kwargs,
                multiagent=opt_dict[f"{name}_multiagent"],
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
            setattr(self, attribute, checkpoint[attribute])

        # Wrap models / compile if necessary
        if accelerator is not None:
            self.wrap_models()
        elif self.torch_compiler:
            torch.set_float32_matmul_precision("high")
            self.recompile()

        # Run init hooks
        for hook in self.registry.hooks:
            getattr(self, hook)()

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

        # TODO: Temporary hack to support legacy code
        self.state_dim = self.get_state_dim(observation_space)
        self.action_dim = self.get_action_dim(action_space)
        self.discrete_actions = isinstance(
            action_space, (spaces.Discrete, spaces.MultiDiscrete)
        )
        self.min_action = (
            np.array(action_space.low) if hasattr(action_space, "low") else None
        )
        self.max_action = (
            np.array(action_space.high) if hasattr(action_space, "high") else None
        )

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
            normalize_images=self.normalize_images,
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
            elif isinstance(exp, (list, tuple)) and isinstance(exp[0], torch.Tensor):
                exp = [val.to(device) for val in exp]
            elif isinstance(exp, torch.Tensor):
                exp = exp.to(device)

            on_device.append(exp)

        return on_device


class MultiAgentRLAlgorithm(EvolvableAlgorithm, ABC):
    """Base object for all multi-agent algorithms in the AgileRL framework.

    :param observation_spaces: The observation spaces of the agent environments.
    :type observation_spaces: List[spaces.Space]
    :param action_spaces: The action spaces of the agent environments.
    :type action_spaces: List[spaces.Space]
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

    def __init__(
        self,
        observation_spaces: Iterable[spaces.Space],
        action_spaces: Iterable[spaces.Space],
        agent_ids: Iterable[int],
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
            agent_ids, (tuple, list)
        ), "Agent IDs must be stores in a tuple or list."
        assert len(agent_ids) == len(
            observation_spaces
        ), "Number of agent IDs must match number of observation spaces."
        assert isinstance(
            observation_spaces, (list, tuple)
        ), "Observation spaces must be a list or tuple."
        assert all(
            isinstance(_space, spaces.Space) for _space in observation_spaces
        ), "Observation spaces must be instances of gymnasium.spaces.Space."
        assert isinstance(
            action_spaces, (list, tuple)
        ), "Action spaces must be a list or tuple."
        assert all(
            isinstance(_space, spaces.Space) for _space in action_spaces
        ), "Action spaces must be instances of gymnasium.spaces.Space."

        if not all(
            isinstance(space, observation_spaces[0].__class__)
            for space in observation_spaces
        ):
            raise ValueError(
                "AgileRL 2.0 only supports homogeneous multi-agent environments "
                "(i.e. all agents have the same type of observation space)"
            )

        # TODO: This can be removed once we have a more robust implementation
        self.state_dims = self.get_state_dim(observation_spaces)
        self.action_dims = self.get_action_dim(action_spaces)
        self.one_hot = all(
            isinstance(space, spaces.Discrete) for space in observation_spaces
        )
        self.discrete_actions = all(
            isinstance(space, spaces.Discrete) for space in action_spaces
        )

        # For continuous action spaces, store the min and max action values
        if not self.discrete_actions:
            self.min_action = [space.low for space in action_spaces]
            self.max_action = [space.high for space in action_spaces]
        else:
            self.min_action, self.max_action = None, None

        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.normalize_images = normalize_images
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.total_actions = sum(self.action_dims)
        self.total_state_dims = None

        if not any(
            isinstance(space, (spaces.Dict, spaces.Tuple))
            for space in observation_spaces
        ):
            self.total_state_dims = sum(state_dim[0] for state_dim in self.state_dims)

        # Build observation and action space dictionaries using agent IDs
        self.single_space = observation_spaces[0]
        self.observation_space = spaces.Dict(
            {agent_id: space for agent_id, space in zip(agent_ids, observation_spaces)}
        )
        self.action_space = spaces.Dict(
            {agent_id: space for agent_id, space in zip(agent_ids, action_spaces)}
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
        for agent_id, obs in observation.items():
            preprocessed[agent_id] = preprocess_observation(
                observation=obs,
                observation_space=self.observation_space.get(agent_id),
                device=self.device,
                normalize_images=self.normalize_images,
            )

        return preprocessed

    def stack_critic_observations(self, obs: Dict[str, TorchObsType]) -> TorchObsType:
        """Process observations for critic network input

        :param obs: Observation dict
        :type obs: Dict[str, torch.Tensor]

        :return: Stacked observations
        :rtype: torch.Tensor
        """
        obs = list(obs.values())
        if isinstance(self.single_space, spaces.Dict):
            processed_obs = {}
            for key, space in self.single_space.spaces.items():
                if is_image_space(space):
                    processed_obs[key] = torch.stack(
                        [obs[i][key] for i in range(self.n_agents)], dim=2
                    )
                else:
                    processed_obs[key] = torch.cat(
                        [obs[i][key] for i in range(self.n_agents)], dim=1
                    )

        elif isinstance(self.single_space, spaces.Tuple):
            processed_obs = []
            for i, space in enumerate(self.single_space):
                if is_image_space(space):
                    processed_obs.append(
                        torch.stack([obs[j][i] for j in range(self.n_agents)], dim=2)
                    )
                else:
                    processed_obs.append(
                        torch.cat([obs[j][i] for j in range(self.n_agents)], dim=1)
                    )
            processed_obs = tuple(processed_obs)

        elif is_image_space(self.single_space):
            processed_obs = torch.stack(obs, dim=2)
        else:
            processed_obs = torch.cat(obs, dim=1)

        return processed_obs
