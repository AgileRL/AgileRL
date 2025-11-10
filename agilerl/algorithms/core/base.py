import copy
import gc
import glob
import inspect
import os
import re
import tempfile
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    TypeVar,
    Union,
    cast,
)

import dill
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from gymnasium import spaces
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from tensordict import TensorDict
from torch._dynamo import OptimizedModule
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from vllm import LLM, SamplingParams

from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    MutationRegistry,
    NetworkGroup,
    OptimizerConfig,
)
from agilerl.modules.configs import MlpNetConfig
from agilerl.modules.dummy import DummyEvolvable
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
    OptimizerType,
    TorchObsType,
)
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    VLLMConfig,
    check_supported_space,
    chkpt_attribute_to_device,
    clone_llm,
    create_warmup_cosine_scheduler,
    get_input_size_from_space,
    get_output_size_from_space,
    isroutine,
    key_in_nested_dict,
    module_checkpoint_dict,
    preprocess_observation,
    recursive_check_module_attrs,
    stack_and_pad_experiences,
    stack_experiences,
)
from agilerl.utils.evolvable_networks import (
    compile_model,
    config_from_dict,
    get_default_encoder_config,
    is_image_space,
    is_vector_space,
)
from agilerl.utils.llm_utils import (
    DummyOptimizer,
    create_model_from_name_or_path,
    gather_if_zero3,
)

__all__ = ["EvolvableAlgorithm", "RLAlgorithm", "MultiAgentRLAlgorithm"]

SelfEvolvableAlgorithm = TypeVar("SelfEvolvableAlgorithm", bound="EvolvableAlgorithm")
SelfRLAlgorithm = TypeVar("SelfRLAlgorithm", bound="RLAlgorithm")
SelfAgentWrapper = TypeVar("SelfAgentWrapper", bound="AgentWrapper")


class _RegistryMeta(type):
    """Metaclass to wrap registry information after algorithm is done
    initializing with specified network groups and optimizers."""

    def __call__(
        cls: type[SelfEvolvableAlgorithm], *args, **kwargs
    ) -> SelfEvolvableAlgorithm:
        # Create the instance
        instance: SelfEvolvableAlgorithm = super().__call__(*args, **kwargs)

        # Call the base class post_init_hook after all initialization
        if isinstance(instance, cls) and hasattr(instance, "_registry_init"):
            instance._registry_init()

        return instance


class RegistryMeta(_RegistryMeta, ABCMeta): ...


def get_checkpoint_dict(
    agent: SelfEvolvableAlgorithm, using_deepspeed: bool = False
) -> dict[str, Any]:
    """Returns a dictionary of the agent's attributes to save in a checkpoint.

    Note: Accelerator is always excluded from the checkpoint as it cannot be serialized.

    :param agent: The agent to save.
    :type agent: EvolvableAlgorithm
    :param using_deepspeed: Whether the agent is using deepspeed.
    :type using_deepspeed: bool, optional

    :return: A dictionary of the agent's attributes.
    :rtype: dict[str, Any]
    """
    attribute_dict = EvolvableAlgorithm.inspect_attributes(agent)
    attribute_dict["agilerl_version"] = version("agilerl")
    attribute_dict.pop("accelerator", None)

    if attribute_dict.pop("lr_scheduler", None) is not None:
        attribute_dict["lr_scheduler"] = agent.lr_scheduler.state_dict()

    if using_deepspeed:
        attribute_dict.pop("actor", None)
        return attribute_dict

    if "rollout_buffer" in attribute_dict:
        attribute_dict.pop("rollout_buffer")

    # Get checkpoint dictionaries for evolvable modules and optimizers
    network_info: dict[str, dict[str, Any]] = {"modules": {}, "optimizers": {}}
    for attr in agent.evolvable_attributes():
        evolvable_obj: EvolvableAttributeType = getattr(agent, attr)
        if isinstance(evolvable_obj, OptimizerWrapper):
            if not using_deepspeed:
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

    return attribute_dict


def get_optimizer_cls(
    optimizer_cls: Union[str, dict[str, str]],
) -> Union[type[torch.optim.Optimizer], dict[str, type[torch.optim.Optimizer]]]:
    """Returns the optimizer class from the string or dictionary of optimizer classes.

    :param optimizer_cls: The optimizer class or dictionary of optimizer classes.
    :type optimizer_cls: Union[str, dict[str, str]]
    :return: The optimizer class or dictionary of optimizer classes.
    :rtype: Union[type[torch.optim.Optimizer], dict[str, type[torch.optim.Optimizer]]]
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
    def test(self, *args, **kwargs) -> np.ndarray:
        """Abstract method for testing the algorithm."""
        raise NotImplementedError

    @staticmethod
    def get_state_dim(observation_space: GymSpaceType) -> tuple[int, ...]:
        """Returns the dimension of the state space as it pertains to the underlying
        networks (i.e. the input size of the networks).

        :param observation_space: The observation space of the environment.
        :type observation_space: spaces.Space or list[spaces.Space].

        :return: The dimension of the state space.
        :rtype: tuple[int, ...].
        """
        warnings.warn(
            "This method is deprecated. Use get_input_size_from_space instead.",
            category=DeprecationWarning,
        )
        return get_input_size_from_space(observation_space)

    @staticmethod
    def get_action_dim(action_space: GymSpaceType) -> tuple[int, ...]:
        """Returns the dimension of the action space as it pertains to the underlying
        networks (i.e. the output size of the networks).

        :param action_space: The action space of the environment.
        :type action_space: spaces.Space or list[spaces.Space].

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
    ) -> dict[str, Any]:
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
        cls: type[SelfEvolvableAlgorithm],
        size: int,
        observation_space: GymSpaceType,
        action_space: GymSpaceType,
        wrapper_cls: Optional[type[SelfAgentWrapper]] = None,
        wrapper_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> list[Union[SelfEvolvableAlgorithm, SelfAgentWrapper]]:
        """Creates a population of algorithms.

        :param size: The size of the population.
        :type size: int.

        :return: A list of algorithms.
        :rtype: list[SelfEvolvableAlgorithm].
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

                # Assign dtype to hyperparameter spec
                hp_value = getattr(self, hp)
                hp_spec = self.registry.hp_config[hp]
                dtype = type(hp_value)
                if dtype not in [int, float, np.ndarray]:
                    raise TypeError(
                        f"Can't mutate hyperparameter {hp} of type {dtype}. AgileRL only supports "
                        "mutating integer, float, and numpy ndarray hyperparameters."
                    )

                hp_spec.dtype = dtype

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

    def _reinit_opt_from_config(
        self: SelfEvolvableAlgorithm, config: OptimizerConfig
    ) -> None:
        """Reinitializes an optimizer from its configuration.

        :param config: The optimizer configuration.
        :type config: OptimizerConfig
        """
        opt: Optional[Union[OptimizerWrapper, DeepSpeedOptimizerWrapper]] = getattr(
            self, config.name
        )
        optimizer = opt.optimizer if hasattr(opt, "optimizer") else None

        if isinstance(opt, DeepSpeedOptimizerWrapper):
            if isinstance(opt.optimizer, DummyOptimizer):
                opt = getattr(
                    getattr(self, "actor"), "optimizer"
                )  # If the optimizer is defined in the deepspeed config, we do this

            self.accelerator, self.lr_scheduler = LLMAlgorithm.update_lr(
                opt,
                lr=getattr(self, config.lr),
                accelerator=self.accelerator,
                scheduler_config=self.cosine_lr_schedule_config,
            )
        else:
            # Multiple optimizers in a single attribute (i.e. multi-agent)
            # or one module optimized by a single optimizer
            if isinstance(optimizer, dict) or len(opt.network_names) == 1:
                opt_nets = getattr(self, opt.network_names[0])

            # Multiple modules optimized by a single optimizer (e.g. PPO)
            else:
                opt_nets = [getattr(self, net) for net in opt.network_names]

            # Reinitialize optimizer with mutated nets
            # NOTE: We need to do this since there is a chance the network parameters have changed
            # due to architecture mutations
            offspring_opt = OptimizerWrapper(
                optimizer_cls=config.get_optimizer_cls(),
                networks=opt_nets,
                lr=getattr(self, opt.lr_name),
                optimizer_kwargs=opt.optimizer_kwargs,
                network_names=opt.network_names,
                lr_name=opt.lr_name,
            )

            setattr(self, config.name, offspring_opt)

    def set_training_mode(self, training: bool) -> None:
        """Sets the training mode of the algorithm.

        :param training: If True, set the algorithm to training mode.
        :type training: bool
        """
        self.training = training

    def get_lr_names(self) -> list[str]:
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

    def reinit_optimizers(
        self,
        optimizer: Optional[OptimizerConfig] = None,
    ) -> None:
        """Reinitialize the optimizers of an algorithm. If no optimizer is passed, all optimizers are reinitialized.

        :param optimizer: The optimizer to reinitialize, defaults to None, in which case
            all optimizers are reinitialized.
        :type optimizer: Optional[OptimizerConfig], optional
        """
        if optimizer is not None:
            self._reinit_opt_from_config(optimizer)
        else:
            optimizer_configs = self.registry.optimizers
            for opt_config in optimizer_configs:
                self._reinit_opt_from_config(opt_config)

    def recompile(self) -> None:
        """Recompiles the evolvable modules in the algorithm with the specified torch compiler."""
        for name, obj in self.evolvable_attributes(networks_only=True).items():
            setattr(self, name, compile_model(obj, self.torch_compiler))

    def to_device(self, *experiences: TorchObsType) -> tuple[TorchObsType, ...]:
        """Moves experiences to the device.

        :param experiences: Experiences to move to device
        :type experiences: tuple[torch.Tensor[float], ...]

        :return: Experiences on the device
        :rtype: tuple[torch.Tensor[float], ...]
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
        checkpoint: dict[str, Any] = torch.load(
            path, map_location=self.device, pickle_module=dill, weights_only=False
        )

        # Recreate evolvable modules
        network_info: dict[str, dict[str, Any]] = checkpoint["network_info"]
        network_names = network_info["network_names"]
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info["modules"].items() if k.startswith(name)
            }

            module_cls = net_dict.get(f"{name}_cls", None)
            if module_cls is None:
                # This allows us to super this method in the LLMAlgorithm class
                # as we don't want to reinstantiate the network in this class
                break
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

        if "lr_scheduler" in checkpoint.keys():
            self.lr_scheduler.load_state_dict(state_dict=checkpoint["lr_scheduler"])
            checkpoint.pop("lr_scheduler")

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
        cls: type[SelfEvolvableAlgorithm],
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
        checkpoint: dict[str, Any] = torch.load(
            path, map_location=device, pickle_module=dill, weights_only=False
        )

        # Reconstruct evolvable modules in algorithm
        network_info: Optional[dict[str, dict[str, Any]]] = checkpoint.get(
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
        loaded_modules: dict[str, EvolvableAttributeType] = {}
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
                type[EvolvableModule], dict[str, type[EvolvableModule]]
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

        check_supported_space(observation_space)
        check_supported_space(action_space)

        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images
        self.action_dim = get_output_size_from_space(self.action_space)

    def preprocess_observation(self, observation: ObservationType) -> TorchObsType:
        """Preprocesses observations for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: ObservationType

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or tuple[torch.Tensor[float], ...]
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
    :type observation_spaces: Union[list[spaces.Space], spaces.Dict]
    :param action_spaces: The action spaces of the agent environments.
    :type action_spaces: Union[list[spaces.Space], spaces.Dict]
    :param index: The index of the individual in the population.
    :type index: int.
    :param agent_ids: The agent IDs of the agents in the environment.
    :type agent_ids: Optional[list[int]], optional
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

    possible_observation_spaces: dict[str, spaces.Space]
    possible_action_spaces: dict[str, spaces.Space]

    shared_agent_ids: list[str]
    grouped_agents: dict[str, list[str]]
    unique_observation_spaces: dict[str, spaces.Space]
    unique_action_spaces: dict[str, spaces.Space]

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

        for obs_space in self.possible_observation_spaces.values():
            check_supported_space(obs_space)
        for action_space in self.possible_action_spaces.values():
            check_supported_space(action_space)

        self.agent_ids = list(self.possible_observation_spaces.keys())
        self.n_agents = len(self.agent_ids)
        self.placeholder_value = placeholder_value
        self.normalize_images = normalize_images
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
    ) -> dict[str, TorchObsType]:
        """Preprocesses observations for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or tuple[torch.Tensor[float], ...]
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
        :type infos: dict[str, dict[...]]

        :return: Action masks
        :rtype: dict[str, np.ndarray]
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
    ) -> tuple[ArrayDict, ArrayDict]:
        """Extract env_defined_actions from info dictionary and determine agent masks

        :param infos: Info dict
        :type infos: dict[str, dict[...]]

        :return: Env defined actions and agent masks
        :rtype: tuple[ArrayDict, ArrayDict]
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
    ) -> Union[NetConfigType, tuple[NetConfigType, dict[str, NetConfigType]]]:
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
        def _add_to_encoder_configs(config: dict[str, Any], agent_id: str = "") -> None:
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
        grouped_agents: dict[str, list[str]],
    ) -> ArrayDict:
        """Disassembles batched output by shared policies into their grouped agents' outputs.

        .. note:: This assumes that for any given sub-agent the termination condition is deterministic,
            i.e. any given agent will always terminate at the same timestep in different vectorized environments.

        :param group_outputs: Dictionary to be disassembled, has the form {'agent': [4, 7, 8]}
        :type group_outputs: dict[str, np.ndarray]
        :param vect_dim: Vectorization dimension size, i.e. number of vect envs
        :type vect_dim: int
        :param grouped_agents: Dictionary of grouped agent IDs
        :type grouped_agents: dict[str, list[str]]
        :return: Assembled dictionary, e.g. {'agent_0': 4, 'agent_1': 7, 'agent_2': 8}
        :rtype: dict[str, np.ndarray]
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
        :type rewards: dict[str, np.ndarray]
        :return: Summed rewards dictionary
        :rtype: dict[str, np.ndarray]
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
        :type agent_outputs: dict[str, np.ndarray]
        :param vect_dim: Vectorization dimension size, i.e. number of vect envs
        :type vect_dim: int
        :return: Assembled dictionary with the form {'agent': [4, 7, 8]}
        :rtype: dict[str, np.ndarray]
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
        index: int,
        batch_size: int,
        lr: float,
        max_grad_norm: float,
        clone: bool,
        reduce_memory_peak: bool,
        calc_position_embeddings: bool,
        seed: int,
        pad_token_id: int,
        pad_token: str,
        lora_config: LoraConfig | None,
        use_separate_reference_adapter: bool,
        model_name: str | None = None,
        actor_network: PreTrainedModel | None = None,
        micro_batch_size_per_gpu: int | None = None,
        cosine_lr_schedule_config: Optional[CosineLRScheduleConfig] = None,
        hp_config: Optional[HyperparameterConfig] = None,
        wrap: bool = True,
        device: Union[str, torch.device] = "cpu",
        accelerator: Optional[Accelerator] = None,
        name: Optional[str] = None,
        model_config: dict[str, Any] | PretrainedConfig | None = None,
        gradient_checkpointing: bool = True,
    ):
        if model_name is None and actor_network is None:
            raise ValueError(
                "At least one of model_name or actor_network must be provided."
            )
        if (
            accelerator is not None
            and cosine_lr_schedule_config is not None
            and accelerator.is_main_process
        ):
            warnings.warn(
                "Cannot specify the optimizer in the deepspeed config and use AgileRL's LR scheduler. If you want to use LR scheduling, \
            please specify in the deepspeed config. Setting LR scheduler to None."
            )
            cosine_lr_schedule_config = None

        super().__init__(index, hp_config, device, accelerator, None, name)
        self.gradient_checkpointing = gradient_checkpointing
        self.zero_stage = None
        self.reference_update_tracker = 0  # Updated every time the reference policy is updated which is updated each time we pass through the train dataset
        self.calc_position_embeddings = calc_position_embeddings
        self.pad_token_id = pad_token_id
        self.pad_token = pad_token
        self.pretrained_model_name_or_path = (
            model_name if model_name is not None else actor_network.name_or_path
        )
        self.model_config = model_config

        if not clone and reduce_memory_peak and micro_batch_size_per_gpu is not None:
            raise ValueError(
                "Cannot specify micro_batch_size_per_gpu when reduce_memory_peak is True."
            )

        self._configure_batch_size(
            batch_size, clone, reduce_memory_peak, micro_batch_size_per_gpu
        )
        self.batch_size = self.batch_size_per_process * (
            self.accelerator.num_processes if self.accelerator is not None else 1
        )
        if self.accelerator is not None:
            if (
                self.accelerator.state.deepspeed_plugin.deepspeed_config.get(
                    "optimizer", None
                )
                is not None
            ):
                optim_lr = self.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "optimizer"
                ]["params"]["lr"]
                if optim_lr is not None and optim_lr != lr:
                    warnings.warn(
                        "Argument 'lr' will be overwritten by the 'lr' value set in the deepspeed config."
                    )
                    lr = optim_lr

        if lora_config is None and not isinstance(actor_network, PeftModel):
            warnings.warn(
                "No LoRA config provided. AgileRL can only be used to finetune adapters at present. Using default LoRA configuration for RL finetuning."
            )
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            )
        self.lr = lr
        self.lora_config = lora_config
        self.wrap = wrap
        self.use_separate_reference_adapter = use_separate_reference_adapter
        self.cosine_lr_schedule_config = cosine_lr_schedule_config

        if max_grad_norm and (accelerator is not None) and accelerator.is_main_process:
            warnings.warn(
                "Argument 'max_grad_norm' will be overwritten by the 'gradient_clipping' value set in the deepspeed config."
            )
            self.max_grad_norm = None
        else:
            self.max_grad_norm = max_grad_norm
        self.reduce_memory_peak = reduce_memory_peak

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
                    "DeepSpeed ZeRO Stage 3 is nascent and may not work as expected, proceed with caution when using this feature."
                )
        if self.accelerator is not None:
            if self.accelerator.is_main_process:
                seed = np.random.randint(0, 2**32 - 1)
            if self.accelerator.num_processes > 1:
                seed = broadcast_object_list([seed], from_process=0)[0]
        self.rng = np.random.RandomState(seed)
        if self.accelerator is not None:
            set_seed(seed, device_specific=True)

    def preprocess_observation(self, observation: ObservationType) -> TorchObsType:
        """Dummy preprocesses observations for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or tuple[torch.Tensor[float], ...]
        """
        return cast(TorchObsType, observation)

    def save_checkpoint(self, path: str, weights_only: bool = True) -> None:
        """
        Override the save_checkpoint method to provide guidance on the correct method to use.
        :param path: Location to save checkpoint at
        :type path: string
        :param weights_only: If True, only save the weights of the model, defaults to False
        :type weights_only: bool, optional
        """

        if self.accelerator is not None:
            if not weights_only:
                self._save_distributed_actor(path, tag="save_checkpoint")
            else:
                selected_adapters = (
                    ["actor", "reference"]
                    if self.use_separate_reference_adapter
                    else ["actor"]
                )
                model_ref = self.accelerator.unwrap_model(self.actor)
                with gather_if_zero3(self.zero_stage, list(model_ref.parameters())):
                    model_ref.save_pretrained(
                        save_directory=path,
                        selected_adapters=selected_adapters,
                        is_main_process=self.accelerator.is_main_process,
                    )

        checkpoint_dict = get_checkpoint_dict(
            self, using_deepspeed=self.accelerator is not None
        )
        checkpoint_dict["_weights_only"] = weights_only
        checkpoint_dict.pop("llm", None)
        checkpoint_dict.pop("tp_group", None)

        if self.accelerator is None or self.accelerator.is_main_process:
            torch.save(
                checkpoint_dict,
                path + "/attributes.pt",
                pickle_module=dill,
            )
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path: str) -> None:
        """
        Override the load_checkpoint method to provide guidance on the correct method to use.

        :param path: Location to load checkpoint from
        :type path: string
        """
        if self.accelerator is not None:
            checkpoint = torch.load(path + "/attributes.pt", weights_only=False)
            weights_only = checkpoint.get("_weights_only", False)

            if weights_only:
                if self.use_separate_reference_adapter:
                    self._update_existing_adapter(
                        path,
                        "reference",
                    )

                self._update_existing_adapter(
                    path,
                    "actor",
                )
            else:
                self._load_distributed_actor(path, tag="save_checkpoint")

            for attr, value in checkpoint.items():
                setattr(self, attr, value)

            self.device = self.accelerator.device

            self.optimizer = None
            self.optimizer = OptimizerWrapper(
                optimizer_cls=self._select_optim_class(),
                networks=[self.actor],
                network_names=["actor"],
                lr=self.lr,
                lr_name="lr",
            )
        else:
            super().load_checkpoint(path + "/attributes.pt")

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
            model = PeftModel.from_pretrained(base_model, path)
            """
        )

    def _select_optim_class(self) -> Union[type[OptimizerType], type[DummyOptimizer]]:
        """Select the optimizer class based on the accelerator and deepspeed config.

        :return: Optimizer class
        :rtype: Union[type[torch.optim.Optimizer], type[DummyOptimizer]]
        """
        if self.accelerator is None:
            return AdamW
        if (
            self.accelerator.state.deepspeed_plugin is not None
            and self.accelerator.state.deepspeed_plugin.deepspeed_config.get(
                "optimizer", None
            )
            is not None
        ):
            return DummyOptimizer
        return AdamW

    def _save_distributed_actor(
        self, path: str, tag: str = "intermediate_checkpoint"
    ) -> None:
        """
        Override the save_checkpoint method to provide guidance on the correct method to use.

        :param path: Output directory to save the checkpoint at
        :type path: str
        """
        if self.accelerator is not None:
            os.makedirs(path, exist_ok=True)
            assert (
                self.actor is not None
            ), "Actor is not defined, please check that the actor is defined."
            self.actor.save_checkpoint(path, tag=tag)
            self.actor.set_adapter("actor")
        else:
            warnings.warn(
                "Distributed actor save not supported for non-distributed training."
            )

    def _load_distributed_actor(
        self, path: str, tag: str = "intermediate_checkpoint"
    ) -> None:
        """
        Override the load_checkpoint method to provide guidance on the correct method to use.

        :param path: Output directory to load the checkpoint from
        :type path: str
        """
        if self.accelerator is not None:
            deepspeed_dirs = sorted(glob.glob(f"{path}/{tag}"))
            try:
                assert len(deepspeed_dirs) > 0
                load_path, _ = self.actor.load_checkpoint(
                    path,
                    tag=tag,
                    load_module_strict=False,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                )
                if load_path is None:
                    raise ValueError(
                        "Load path is returned as None from deepspeed load_checkpoint."
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

    def wrap_models(self) -> None:
        """Wrap the models in the accelerator, DeepSpeed objects must be wrapped at the same time,
        not individually."""
        if self.accelerator is not None:
            assert (
                self.optimizer is not None
            ), "Optimizer is set to None, please check that the optimizer is correctly defined."
            is_dummy_optimizer = isinstance(self.optimizer.optimizer, DummyOptimizer)
            self.actor, optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.actor, self.optimizer.optimizer, self.lr_scheduler
            )
            self.optimizer.optimizer = (
                optimizer if not is_dummy_optimizer else self.actor.optimizer
            )
            self.optimizer.optimizer_cls = (
                type(optimizer)
                if not is_dummy_optimizer
                else type(self.actor.optimizer)
            )
            if self.gradient_checkpointing:
                self.actor.module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        else:
            assert (
                self.actor is not None
            ), "Actor is set to None, please check that the actor is defined."
            self.actor = self.actor.to(self.device)
            if self.gradient_checkpointing:
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
        if hasattr(self, "llm"):
            del self.llm.llm_engine.model_executor
            del self.llm

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

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

            if (
                self.accelerator is not None
                and self.zero_stage is not None
                and self.zero_stage >= 2
            ):
                self.accelerator.wait_for_everyone()
                self._save_distributed_actor(f"{temp_dir}/agent_{self.index}")
                self.accelerator.wait_for_everyone()

            input_args = EvolvableAlgorithm.inspect_attributes(
                self, input_args_only=True
            )
            input_args["wrap"] = False
            input_args["clone"] = True

            actor: PeftModel = cast(
                PeftModel,
                (
                    self.accelerator.unwrap_model(self.actor)
                    if self.accelerator is not None
                    else self.actor
                ),
            )

            actor_state_dict = None
            if self.zero_stage is None or self.zero_stage < 2:
                actor_state_dict = clone_tensors_for_torch_save(actor.state_dict())

            cloned_model = clone_llm(
                actor, self.zero_stage, state_dict=actor_state_dict
            )
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
            if self.use_vllm:
                original_llm = self.llm
                cloned_llm = clone.llm
                clone.llm = None
                self.llm = None
            clone = EvolvableAlgorithm.copy_attributes(self, clone)
            clone.accelerator = accelerator
            clone.lr_scheduler = lr_scheduler
            clone.lr_scheduler = cloned_lr_scheduler
            self.lr_scheduler = original_lr_scheduler
            if self.use_vllm:
                clone.llm = cloned_llm
                self.llm = original_llm

            if self.accelerator is None:
                clone.optimizer.optimizer.load_state_dict(
                    state_dict=self.optimizer.optimizer.state_dict()
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
        optimizer: torch.optim.Optimizer,  # Deepspeed optimizers are subclasses of torch.optim.Optimizer
        lr: float,
        accelerator: Optional[Accelerator] = None,
        scheduler_config: Optional[CosineLRScheduleConfig] = None,
    ) -> tuple[Optional[Accelerator], Optional[SequentialLR]]:
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
            not hasattr(accelerator.state, "deepspeed_plugin")
            or accelerator.state.deepspeed_plugin is None
        ):
            raise ValueError(
                "Accelerator must be instantiated with a deepspeed plugin."
            )

        if not hasattr(accelerator.state.deepspeed_plugin, "deepspeed_config"):
            raise ValueError(
                "Deepspeed config not found in accelerator state, make sure DeepSpeed is configured in your accelerator config."
            )

        if (
            accelerator.state.deepspeed_plugin.deepspeed_config.get("scheduler", None)
            is not None
        ):
            accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "warmup_max_lr"
            ] = lr

        if (
            accelerator.state.deepspeed_plugin.deepspeed_config is not None
            and accelerator.state.deepspeed_plugin.deepspeed_config.get(
                "optimizer", None
            )
            is not None
        ):
            accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"][
                "lr"
            ] = lr

        return accelerator, None

    def set_reference_policy(self, reference_update_tracker: int) -> None:
        """Update the reference policy when the reference policy update tracker is greater than the current reference policy update tracker.

        :param reference_update_tracker: The reference policy update tracker
        :type reference_update_tracker: int
        """
        assert (
            reference_update_tracker >= self.reference_update_tracker
        ), "Reference policy update tracker should be greater than or equal to the current reference policy update tracker."
        if reference_update_tracker > self.reference_update_tracker:

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            # Merge adapter into base model
            # Update the reference update tracker
            if self.use_separate_reference_adapter:
                # Activate both adapters
                # Iterate over the params
                ref_param = None
                actor_param = None
                for name, param in self.actor.named_parameters():
                    if "lora" in name:
                        if "reference" in name:
                            ref_param = param
                        elif "actor" in name:
                            actor_param = param
                        else:
                            raise ValueError(
                                f"Only adapter names 'actor' and 'reference' are allowed, nether was found in {name}"
                            )
                    if ref_param is not None and actor_param is not None:
                        ref_param.data.copy_(actor_param.data)
                        ref_param = None
                        actor_param = None
            else:
                if self.accelerator is not None:
                    merged_base_model = self.accelerator.unwrap_model(
                        self.actor
                    ).merge_and_unload()
                else:
                    merged_base_model = self.actor.merge_and_unload()
                self.actor = None  # De-reference the old actor base model
                self.actor = get_peft_model(
                    merged_base_model, self.lora_config, adapter_name="actor"
                )
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                self.actor.set_adapter("actor")

                # Reinit optimizer
                optim_class = self._select_optim_class()
                self.optimizer = OptimizerWrapper(
                    optim_class, networks=[self.actor], lr=self.lr
                )
                self.wrap_models()
            self.reference_update_tracker += 1

    def _initialize_actors(
        self, base_model: PreTrainedModel | None, add_adapters: bool = True
    ):
        """Initialize the actor network.

        :param base_model: Base model
        :type base_model: PreTrainedModel
        :param add_adapters: Flag to indicate if adapters should be added to the model, defaults to True
        :type add_adapters: bool, optional
        """

        if base_model is None:
            base_model = create_model_from_name_or_path(
                self.pretrained_model_name_or_path
            )

        if isinstance(base_model, PeftModel) and add_adapters:
            # Handles backwards compatibility with user providing a peft model as the actor network
            if self.lora_config is None:
                adapter_name = list(base_model.peft_config.keys())
                self.lora_config = base_model.peft_config[adapter_name[0]]
            with gather_if_zero3(self.zero_stage, list(base_model.parameters())):
                base_model = base_model.merge_and_unload()
            if "default" in list(base_model.peft_config.keys()):
                base_model.peft_config.pop("default")

        self.actor: PeftModel = (
            get_peft_model(base_model, self.lora_config, adapter_name="actor")
            if add_adapters
            else base_model
        )

        if self.use_separate_reference_adapter and add_adapters:
            self.actor.add_adapter(
                adapter_name="reference", peft_config=self.lora_config  # type: ignore
            )

        self.actor.set_adapter("actor")

        if self.accelerator is None:
            self.actor = DummyEvolvable(module=self.actor, device=self.device)

        optim_class = self._select_optim_class()
        self.optimizer = OptimizerWrapper(
            optim_class, networks=[self.actor], lr=self.lr
        )
        self.lr_scheduler = (
            create_warmup_cosine_scheduler(
                (
                    self.optimizer.optimizer
                    if self.optimizer.optimizer_cls != DummyOptimizer
                    else self.actor.optimizer
                ),
                self.cosine_lr_schedule_config,
                1e-8,
                self.lr,
            )
            if self.cosine_lr_schedule_config is not None
            else None
        )

    def _get_logprobs(
        self,
        ids: torch.Tensor,
        batch_size: int,
        use_reference: bool = False,
        eval_mode: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Find the log probabilities for a set of previously generated ids.

        :param ids: Completion IDs.
        :type ids: torch.Tensor
        :param batch_size: Batch size.
        :type batch_size: int
        :param use_reference: Flag to indicate to use reference policy, defaults to False
        :type use_reference: bool, optional
        :param eval_mode: Flag to indicate setting policy network to evaluation mode, defaults to False
        :type eval_mode: bool, optional
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor, optional
        :return: Log probabilities of the completion IDs.
        :rtype: torch.Tensor
        """

        with self.select_policy(use_reference):
            self.actor.train(mode=not eval_mode)
            num_samples = ids.shape[0]
            if attention_mask is None:
                # TODO this calc is avoided when using PreferenceGym, need to make ReasoningGym do the same
                attention_mask = ids != self.pad_token_id
            if self.calc_position_embeddings:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

            # Split the sample into batches
            log_probs = []
            for batch in range(0, num_samples, batch_size):
                end_idx = min((batch + batch_size), num_samples)
                batch_ids = ids[batch:end_idx, :]
                batch_attention_mask = attention_mask[batch:end_idx, :]
                batch_model_kwargs = {
                    "input_ids": batch_ids,
                    "attention_mask": batch_attention_mask,
                    "use_cache": False,
                }
                if self.calc_position_embeddings:
                    batch_position_ids = position_ids[batch:end_idx, :]
                    batch_model_kwargs |= {"position_ids": batch_position_ids}
                logits = self.actor.forward(**batch_model_kwargs).logits
                logits = logits / self.temperature
                log_prob = LLMAlgorithm._memory_efficient_logits(
                    logits[:, :-1], batch_ids[:, 1:]
                )
                batch_model_kwargs = None
                logits = None
                log_probs.append(log_prob)
        return torch.cat(log_probs, dim=0)

    def _backward_pass(self, loss: float) -> None:
        """Perform a backward pass

        :param loss: Loss
        :type loss: float
        """
        if self.accelerator is not None:
            self.accelerator.backward(loss)
            if (
                self.accelerator.state.deepspeed_plugin.deepspeed_config.get(
                    "optimizer", None
                )
                is None
            ):
                # Accelerate handles optimizer step and zero grad if optimizer is defined in deepspeed config
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr = self.lr_scheduler.get_last_lr()[0]

    @contextmanager
    def select_policy(self, use_reference: bool = False) -> None:
        """Select the policy."""
        if use_reference:
            self._use_reference_policy()
        else:
            self._use_policy()
        yield
        self._use_policy()

    def _use_reference_policy(self) -> None:
        """Use the reference policy."""
        if self.use_separate_reference_adapter:
            self.actor.set_adapter("reference")
            for name, param in self.actor.named_parameters():
                if param is not None and "reference" in name:
                    param.requires_grad = False
        else:
            self.actor.base_model.disable_adapter_layers()

    def _use_policy(self) -> None:
        """Use the policy."""
        if self.use_separate_reference_adapter:
            self.actor.set_adapter("actor")
        else:
            self.actor.base_model.enable_adapter_layers()

    def _move_model_to_vllm(self) -> None:
        """Move the deepspeed model to vllm."""

        # TODO: Add support for ZeRO Stage 3
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        model_ref = self.accelerator.unwrap_model(self.actor)
        model_ref.set_adapter("actor")
        with gather_if_zero3(self.zero_stage, list(model_ref.parameters())):
            model_ref.merge_adapter()
            for name, param in model_ref.named_parameters():
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if model_ref.prefix in name:
                    continue

                if "original_module" in name:
                    continue

                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                llm_model.load_weights([(name, param.data)])
            model_ref.unmerge_adapter()

        self.llm.reset_prefix_cache()

    def _generate_with_vllm_colocate(
        self, prompts: list[dict[str, int]], group_size: int
    ) -> list[torch.Tensor]:

        # I need to make the following happen
        # prompts = [prompt1, prompt1, ..., prompt1 (group_size times), prompt2, prompt2, ..., prompt2 (group_size times), ...]

        # The below line returns a list: [prompt1 * group_size, ..., promptN * group_size],
        # where N is the data batch size per gpu, list length is group_size * N
        group_prompts = [prompt for prompt in prompts for _ in range(group_size)]
        prompts_ids = [prompt["input_ids"] for prompt in group_prompts]
        prompts_text = [prompt["text"] for prompt in group_prompts]
        prompts_text = [
            re.sub(rf"^({re.escape(str(self.pad_token))})+", "", text)
            for text in prompts_text
        ]

        # max_output_tokens now acts as a global
        max_token_cap = (
            self.max_output_tokens
            if self.max_output_tokens is not None
            else self.max_model_len
        )

        max_output_tokens = [
            min(max_token_cap, self.max_model_len - len(prompt_id))
            for prompt_id in prompts_ids
        ]

        generation_kwargs = {
            "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": -1 if self.top_k is None else self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "min_tokens": (
                0 if self.min_output_tokens is None else self.min_output_tokens
            ),
        }
        sampling_params = [
            SamplingParams(**generation_kwargs, max_tokens=max_output_token)
            for max_output_token in max_output_tokens
        ]

        if self.vllm_config.tensor_parallel_size > 1:

            orig_size = len(prompts_text)

            gathered_prompts_ids = [
                None for _ in range(self.vllm_config.tensor_parallel_size)
            ]
            gathered_prompts_text = [
                None for _ in range(self.vllm_config.tensor_parallel_size)
            ]

            torch.distributed.all_gather_object(
                gathered_prompts_ids, prompts_ids, group=self.tp_group
            )
            torch.distributed.all_gather_object(
                gathered_prompts_text, prompts_text, group=self.tp_group
            )

            all_prompts_ids = [
                prompt_id for sublist in gathered_prompts_ids for prompt_id in sublist
            ]
            all_prompts_text = [
                prompt_text
                for sublist in gathered_prompts_text
                for prompt_text in sublist
            ]
        else:
            all_prompts_text = prompts_text
            all_prompts_ids = prompts_ids

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        all_outputs = self.llm.generate(
            all_prompts_text,
            sampling_params=sampling_params,
            use_tqdm=True,
        )  # Change this to False

        completion_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        if self.vllm_config.tensor_parallel_size > 1:
            # Slice completions for this rank within its TP group.
            # Each rank generates all outputs — we keep only our share.
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            tp_slice = slice(
                local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size
            )
            completion_ids = completion_ids[tp_slice]
            prompts_ids = all_prompts_ids[tp_slice]

        completion_ids = [
            torch.cat(
                [
                    torch.cat(
                        prompts_ids[group_size * i : group_size * (i + 1)], dim=0
                    ),
                    stack_and_pad_experiences(
                        completion_ids[group_size * i : group_size * (i + 1)],
                        padding_values=[self.pad_token_id],
                        device=self.device,
                    )[0],
                ],
                dim=1,
            )
            for i, _ in enumerate(prompts)
        ]

        num_input_tokens = [prompt_ids.shape[1] for prompt_ids in prompts_ids][
            ::group_size
        ]
        action_masks = []

        for i, completion_id in enumerate(completion_ids):
            action_mask = torch.zeros_like(completion_id, device=self.device)
            action_mask[:, num_input_tokens[i] :] = True
            action_mask[completion_id == self.pad_token_id] = False
            action_mask = action_mask[:, 1:]
            action_masks.append(action_mask)

        return completion_ids, action_masks

    @staticmethod
    def _memory_efficient_logits(
        logits: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the log probabilities for a set of previously generated ids, looping to reduce peak memory consumption.

        :param logits: Logits.
        :type logits: torch.Tensor
        :param index: Index.
        :type index: torch.Tensor
        :return: Log probabilities of the completion IDs.
        :rtype: torch.Tensor
        """
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def _configure_batch_size(
        self,
        batch_size: int,
        clone: bool,
        reduce_memory_peak: bool,
        micro_batch_size_per_gpu: int | None,
    ) -> None:
        if self.accelerator is None or clone:
            self.batch_size_per_process = batch_size
            return

        if batch_size % self.accelerator.num_processes != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by the number of processes ({self.accelerator.num_processes})."
            )

        ds_config = self.accelerator.state.deepspeed_plugin.deepspeed_config

        if reduce_memory_peak:
            self.batch_size_per_process = 1
            self.micro_batch_size_per_gpu = 1
            ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size_per_gpu
            gradient_accumulation_steps = batch_size / self.accelerator.num_processes
            ds_config["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
            return

        self.batch_size_per_process = int(batch_size / self.accelerator.num_processes)

        if micro_batch_size_per_gpu is None:
            if (
                self.batch_size_per_process
                % ds_config.get("gradient_accumulation_steps", 1)
                != 0
            ):
                raise ValueError(
                    f"Batch size ({batch_size}) must be divisible by the product of the number of processes ({self.accelerator.num_processes}) and gradient accumulation steps ({self.accelerator.state.deepspeed_plugin.deepspeed_config.get('gradient_accumulation_steps', 1)})."
                    "Gradient accumulation steps can be updated in the deepspeed config by changing the 'gradient_accumulation_steps' parameter."
                )
            self.micro_batch_size_per_gpu = (
                self.batch_size_per_process
                // ds_config.get("gradient_accumulation_steps", 1)
            )
            if self.micro_batch_size_per_gpu == 0:
                raise ValueError("Calculated micro_batch_size_per_gpu is 0...")

            if ds_config.get("train_micro_batch_size_per_gpu", "auto") == "auto":
                ds_config["train_micro_batch_size_per_gpu"] = (
                    self.micro_batch_size_per_gpu
                )
            return

        self.micro_batch_size_per_gpu = int(micro_batch_size_per_gpu)
        if (
            batch_size
            % (self.micro_batch_size_per_gpu * self.accelerator.num_processes)
            != 0
        ):
            raise ValueError(
                f"When specifying micro_batch_size_per_gpu, batch_size ({batch_size}) must be divisible by the product of the number of processes ({self.accelerator.num_processes}) and micro_batch_size_per_gpu ({self.micro_batch_size_per_gpu})."
            )
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size_per_gpu
        gradient_accumulation_steps = (
            batch_size / self.accelerator.num_processes / self.micro_batch_size_per_gpu
        )
        warnings.warn(
            f"Overwriting deepspeed config gradient accumulation steps from {self.accelerator.state.deepspeed_plugin.deepspeed_config.get('gradient_accumulation_steps', 'auto')} to {gradient_accumulation_steps}"
        )
        ds_config["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
        return

    def recompile(self) -> None:
        """Recompiles the algorithm."""
        raise NotImplementedError(
            "Recompile method is not available for LLM finetuning algorithms."
        )

    def _update_existing_adapter(
        self,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        """
        Overwrite weights of an existing adapter in-place without creating new parameters.
        xw
        :param checkpoint_dir: Checkpoint directory
        :type checkpoint_dir: str
        :param adapter_name: Adapter name
        :type adapter_name: str

        :return: None
        :rtype: None
        """
        base_model = self.accelerator.unwrap_model(self.actor)
        if hasattr(base_model, "module"):
            base_model = base_model.module

        adapter_path = f"{checkpoint_dir}/{adapter_name}/adapter_model.safetensors"
        adapter_state = load_file(adapter_path, device="cpu")

        with gather_if_zero3(
            self.zero_stage, list(base_model.parameters()), modifier_rank=0
        ):
            with torch.no_grad():
                set_peft_model_state_dict(
                    base_model, adapter_state, adapter_name=adapter_name
                )
            base_model.set_adapter(adapter_name)

            # Make reference weights not trainable
            for name, param in base_model.named_parameters():
                if "reference" in name:
                    param.requires_grad = False
        self.accelerator.wait_for_everyone()

    @staticmethod
    def create_prompt_masks(prompt_lengths: list[int], max_length: int) -> torch.Tensor:
        """
        Creates a mask for the prompts based on the prompt lengths (vectorized).

        :param prompt_lengths: List of prompt lengths
        :type prompt_lengths: list[int]
        :param max_length: Maximum length of the prompts
        :type max_length: int
        :return: Mask tensor [batch_size, max_length]
        :rtype: torch.Tensor
        """
        prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.long)
        positions = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        mask = positions > prompt_lengths_tensor.unsqueeze(1)
        return mask

    def _configure_vllm(self) -> None:
        """
        Configure vLLM for efficient inference during generation in 'get_action'.

        """
        if self.vllm_config is None:
            warnings.warn(
                "No VLLM config provided. Using default VLLM configuration for generation."
            )
            self.vllm_config = VLLMConfig()
        if self.accelerator is not None:
            if (
                self.accelerator.num_processes % self.vllm_config.tensor_parallel_size
                != 0
            ):
                raise ValueError(
                    f"Tensor parallel size {self.vllm_config.tensor_parallel_size} must be a multiple of the number of processes {self.accelerator.num_processes}."
                )

            if self.vllm_config.tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(
                            range(
                                i * self.vllm_config.tensor_parallel_size,
                                (i + 1) * self.vllm_config.tensor_parallel_size,
                            )
                        )
                        for i in range(
                            self.accelerator.num_processes
                            // self.vllm_config.tensor_parallel_size
                        )
                    ]
                )

            # vLLM requires the environment variables to be set for distributed training.
            os.environ["RANK"] = str(self.accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

            self.llm = LLM(
                model=self.pretrained_model_name_or_path,
                tensor_parallel_size=self.vllm_config.tensor_parallel_size,
                gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
                max_num_seqs=self.vllm_config.max_num_seqs,
                max_model_len=self.max_model_len,
                distributed_executor_backend="external_launcher",
                seed=self.accelerator.process_index
                // self.vllm_config.tensor_parallel_size,
                max_num_batched_tokens=self.vllm_config.max_num_seqs
                * self.max_model_len,
                model_impl="vllm",
                enable_sleep_mode=self.vllm_config.sleep_mode,
            )
            if self.vllm_config.sleep_mode:
                self.llm.sleep(level=2)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
