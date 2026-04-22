from __future__ import annotations

import copy
import gc
import inspect
import os
import pickle
import tempfile
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
)

import dill
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from gymnasium import spaces
from tensordict import TensorDict
from torch._dynamo import OptimizedModule
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from typing_extensions import Self

from agilerl import HAS_DEEPSPEED, HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES, HAS_VLLM
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
    AgentWrapperProtocol,
    EvolvableAttributeDict,
    EvolvableAttributeType,
    EvolvableModuleProtocol,
    ModuleDictProtocol,
    PeftModelProtocol,
    PretrainedConfigProtocol,
    PreTrainedModelProtocol,
)
from agilerl.typing import (
    ActionType,
    ArrayDict,
    CheckpointInfo,
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
    DummyOptimizer,
    VLLMConfig,
    check_supported_space,
    chkpt_attribute_to_device,
    clone_llm,
    create_warmup_cosine_scheduler,
    filter_init_dict,
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

if TYPE_CHECKING:
    from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
    from torch.optim.lr_scheduler import SequentialLR

# Make imports visible to typechecker and import when required
if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file

    from agilerl.utils.llm_utils import create_model_from_name_or_path, gather_if_zero3

if TYPE_CHECKING or HAS_DEEPSPEED:
    from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

if TYPE_CHECKING or HAS_VLLM:
    from vllm import LLM, SamplingParams

    from agilerl.algorithms.core.fused_lora import (
        clear_fused_adapter_routing,
        patch_lora_for_fused_forward,
        set_fused_adapter_routing,
    )
    from agilerl.utils.llm_utils import (
        align_deepspeed_lr,
        create_model_from_name_or_path,
        gather_if_zero3,
        get_model_name_or_path,
        move_params_to_cpu,
        move_params_to_gpu,
        stitch_completion_after_windowed_vllm_generate,
    )

__all__ = ["EvolvableAlgorithm", "MultiAgentRLAlgorithm", "RLAlgorithm"]

SelfAgentWrapper = TypeVar("SelfAgentWrapper", bound=AgentWrapperProtocol)


class _RegistryMeta(type):
    """Metaclass to wrap registry information after algorithm is done
    initializing with specified network groups and optimizers.
    """

    def __call__(
        cls: type[EvolvableAlgorithm],  # type: ignore[misc]
        *args: Any,
        **kwargs: Any,
    ) -> EvolvableAlgorithm:
        # Create the instance
        instance: EvolvableAlgorithm = super().__call__(*args, **kwargs)  # type: ignore[misc]

        # Call the base class post_init_hook after all initialization
        if isinstance(instance, cls) and hasattr(instance, "_registry_init"):
            instance._registry_init()

        return instance


class RegistryMeta(_RegistryMeta, ABCMeta):
    """Metaclass combining registry initialization with ABC support."""


def get_checkpoint_dict(
    agent: EvolvableAlgorithm,
    omit_actor_info: bool = False,
    omit_optimizer_info: bool = False,
) -> dict[str, Any]:
    """Return a dictionary of the agent's attributes to save in a checkpoint.

    Note: Accelerator is always excluded from the checkpoint as it cannot be serialized.

    :param agent: The agent to save.
    :type agent: EvolvableAlgorithm
    :param omit_actor_info: Whether to remove the 'actor' attribute prior to saving.
        To be used when saving LoRA weights only or when using Deepspeed.
    :type omit_actor_info: bool, optional
    :param omit_optimizer_info: Whether to remove the 'optimizer' attribute prior to saving.
        To be used when saving LoRA weights only or when using Deepspeed.
    :type omit_optimizer_info: bool, optional
    :return: A dictionary of the agent's attributes.
    :rtype: dict[str, Any]
    """
    from agilerl.modules import EvolvableModule

    attribute_dict = EvolvableAlgorithm.inspect_attributes(agent)
    attribute_dict["agilerl_version"] = version("agilerl")
    attribute_dict.pop("accelerator", None)
    attribute_dict.pop("rollout_buffer", None)
    if attribute_dict.pop("lr_scheduler", None) is not None:
        attribute_dict["lr_scheduler"] = agent.lr_scheduler.state_dict()

    # Get checkpoint dictionaries for evolvable modules and optimizers
    # Use type CheckpointInfo so load code can rely on the key existing.
    checkpoint_info = CheckpointInfo(
        modules={},
        optimizers={},
        network_names=[],
        optimizer_names=[],
    )

    for name in agent.evolvable_attributes():
        obj = getattr(agent, name)
        if isinstance(obj, (OptimizedModule, EvolvableModule)):
            if not omit_actor_info:
                checkpoint_info["modules"].update(module_checkpoint_dict(obj, name))
                checkpoint_info["network_names"].append(name)
        elif isinstance(obj, OptimizerWrapper):
            if not omit_optimizer_info:
                checkpoint_info["optimizers"].update(obj.checkpoint_dict(name))
                checkpoint_info["optimizer_names"].append(name)

    attribute_dict["network_info"] = checkpoint_info

    return attribute_dict


def get_optimizer_cls(
    optimizer_cls: str | dict[str, str],
) -> type[torch.optim.Optimizer] | dict[str, type[torch.optim.Optimizer]]:
    """Return the optimizer class from the string or dictionary of optimizer classes.

    :param optimizer_cls: The optimizer class or dictionary of optimizer classes.
    :type optimizer_cls: str | dict[str, str]
    :return: The optimizer class or dictionary of optimizer classes.
    :rtype: type[torch.optim.Optimizer] | dict[str, type[torch.optim.Optimizer]]
    """
    if isinstance(optimizer_cls, dict):
        optimizer_cls = {
            agent_id: getattr(torch.optim, optimizer_cls[agent_id])
            for agent_id in optimizer_cls
        }
    else:
        optimizer_cls = getattr(torch.optim, optimizer_cls)

    return optimizer_cls


class EvolvableAlgorithm(ABC, metaclass=RegistryMeta):
    """Base object for all algorithms in the AgileRL framework.

    :param index: The index of the individual.
    :type index: int
    :param hp_config: Hyperparameter configuration for the algorithm, defaults to None.
    :type hp_config: HyperparameterConfig | None, optional
    :param device: Device to run the algorithm on, defaults to "cpu".
    :type device: str | torch.device, optional
    :param accelerator: Accelerator object for distributed computing, defaults to None.
    :type accelerator: Accelerator | None, optional
    :param torch_compiler: The torch compiler mode to use, defaults to None.
    :type torch_compiler: Any | None, optional
    :param name: Name of the algorithm, defaults to the class name.
    :type name: str | None, optional
    """

    def __init__(
        self,
        index: int,
        hp_config: HyperparameterConfig | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        torch_compiler: Any | None = None,
        name: str | None = None,
    ) -> None:

        assert isinstance(index, int), "Agent index must be an integer."
        assert isinstance(device, (str, torch.device)), "Device must be a string."
        assert isinstance(name, (type(None), str)), "Name must be a string."
        assert isinstance(
            accelerator,
            (type(None), Accelerator),
        ), "Accelerator must be an instance of Accelerator."
        if torch_compiler:
            assert torch_compiler in [
                "default",
                "reduce-overhead",
                "max-autotune",
            ], (
                "Choose between torch compiler modes: default, reduce-overhead, max-autotune or None"
            )

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
        """Return the index of the algorithm."""
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        """Set the index of the algorithm."""
        self._index = value

    @property
    def mut(self) -> Any:
        """Return the mutation object of the algorithm."""
        return self._mut

    @mut.setter
    def mut(self, value: str | None) -> None:
        """Set the mutation object of the algorithm."""
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
        self,
        obs: ObservationType | MultiAgentObservationType,
        *args: Any,
        **kwargs: Any,
    ) -> ActionType:
        """Abstract method for getting an action from the algorithm.

        :param obs: The observation to get an action for.
        :type obs: ObservationType | MultiAgentObservationType
        :param args: Additional arguments to pass to the action function.
        :type args: Any
        :param kwargs: Additional keyword arguments to pass to the action function.
        :type kwargs: Any
        :return: The action to take.
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Abstract method for testing the algorithm."""
        raise NotImplementedError

    @staticmethod
    def get_state_dim(observation_space: GymSpaceType) -> tuple[int, ...]:
        """Return the dimension of the state space as it pertains to the underlying
        networks (i.e. the input size of the networks).

        :param observation_space: The observation space of the environment.
        :type observation_space: spaces.Space or list[spaces.Space].

        :return: The dimension of the state space.
        :rtype: tuple[int, ...].
        """
        warnings.warn(
            "This method is deprecated. Use get_input_size_from_space instead.",
            stacklevel=2,
            category=DeprecationWarning,
        )
        return get_input_size_from_space(observation_space)

    @staticmethod
    def get_action_dim(action_space: GymSpaceType) -> tuple[int, ...]:
        """Return the dimension of the action space as it pertains to the underlying
        networks (i.e. the output size of the networks).

        :param action_space: The action space of the environment.
        :type action_space: spaces.Space or list[spaces.Space].

        :return: The dimension of the action space.
        :rtype: int.
        """
        warnings.warn(
            "This method is deprecated. Use get_output_size_from_space instead.",
            stacklevel=2,
            category=DeprecationWarning,
        )
        return get_output_size_from_space(action_space)

    @staticmethod
    def inspect_attributes(
        agent: EvolvableAlgorithm,
        input_args_only: bool = False,
    ) -> dict[str, Any]:
        """Inspect and retrieve the attributes of the current object, excluding attributes related to the
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
        agent: EvolvableAlgorithm,
        clone: EvolvableAlgorithm,
    ) -> EvolvableAlgorithm:
        """Copy the non-evolvable attributes of the algorithm to a clone.

        :param clone: The clone of the algorithm.
        :type clone: EvolvableAlgorithm

        :return: The clone of the algorithm.
        :rtype: EvolvableAlgorithm
        """
        for attribute in EvolvableAlgorithm.inspect_attributes(agent):
            if hasattr(agent, attribute) and hasattr(clone, attribute):
                attr, clone_attr = getattr(agent, attribute), getattr(clone, attribute)

                # NOTE: Here we handle the case where the individual is wrapped by an
                # AgentWrapper object, which includes the agent itself and functools.partial
                # objects as attributes that shouldn't be copied
                if callable(attr) or isinstance(attr, EvolvableAlgorithm):
                    continue
                if isinstance(attr, torch.Tensor) or isinstance(
                    clone_attr,
                    torch.Tensor,
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
                                clone,
                                attribute,
                                torch.clone(getattr(agent, attribute)),
                            )

                elif isinstance(attr, np.ndarray) or isinstance(clone_attr, np.ndarray):
                    if not np.array_equal(attr, clone_attr):
                        setattr(
                            clone,
                            attribute,
                            copy.deepcopy(getattr(agent, attribute)),
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
        cls,
        size: int,
        observation_space: GymSpaceType,
        action_space: GymSpaceType,
        wrapper_cls: type[SelfAgentWrapper] | None = None,
        wrapper_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[Self | SelfAgentWrapper]:
        """Create a population of algorithms.

        :param size: The size of the population.
        :type size: int.

        :return: A list of algorithms.
        :rtype: list[EvolvableAlgorithm].
        """
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
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
        """Set the attribute of the algorithm. If the attribute is an OptimizerWrapper,
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
        """Register the networks, optimizers, and algorithm hyperparameters in the algorithm with
        the mutations registry. We also check that all of the evolvable networks and their respective
        optimizers have been registered with the algorithm, and that the user-specified hyperparameters
        to mutate have been set as attributes in the algorithm.
        """
        if not self.registry.groups:
            msg = (
                "No network groups have been registered in the algorithms __init__ method. "
                "Please register NetworkGroup objects specifying all of the evaluation and "
                "shared/target networks through the `register_network_group()` method."
            )
            raise AttributeError(
                msg,
            )

        # Check that all the inspected evolvable attributes can be found in the registry
        all_registered = self.registry.all_registered()
        not_found = [
            attr for attr in self.evolvable_attributes() if attr not in all_registered
        ]
        if not_found:
            msg = (
                f"The following evolvable attributes could not be found in the registry: {not_found}. "
                "Please check that the defined NetworkGroup objects contain all of the EvolvableModule objects "
                "in the algorithm."
            )
            raise AttributeError(
                msg,
            )

        # Check that one of the network groups relates to a policy
        if not any(group.policy for group in self.registry.groups):
            msg = (
                "No network group has been registered as a policy (i.e. the network used to "
                "select actions) in the registry. Please register a NetworkGroup object "
                "specifying the policy network."
            )
            raise AttributeError(
                msg,
            )

        # Check that all the hyperparameters to mutate have been set as attributes in the algorithm
        if self.registry.hp_config is not None:
            for hp in self.registry.hp_config:
                if not hasattr(self, hp):
                    msg = (
                        f"Hyperparameter {hp} was found in the mutations configuration but has "
                        "not been set as an attribute in the algorithm."
                    )
                    raise AttributeError(
                        msg,
                    )

                # Assign dtype to hyperparameter spec
                hp_value = getattr(self, hp)
                hp_spec = self.registry.hp_config[hp]
                dtype = type(hp_value)
                if dtype not in [int, float, np.ndarray]:
                    msg = (
                        f"Can't mutate hyperparameter {hp} of type {dtype}. AgileRL only supports "
                        "mutating integer, float, and numpy ndarray hyperparameters."
                    )
                    raise TypeError(
                        msg,
                    )

                hp_spec.dtype = dtype

    def _wrap_attr(self, attr: EvolvableAttributeType) -> EvolvableAttributeType:
        """Wrap the model with the accelerator.

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
        self,
        config: OptimizerConfig,
    ) -> None:
        """Reinitializes an optimizer from its configuration.

        :param config: The optimizer configuration.
        :type config: OptimizerConfig
        """
        opt: OptimizerWrapper | DeepSpeedOptimizerWrapper | None = getattr(
            self,
            config.name,
        )
        optimizer = getattr(opt, "optimizer", None)

        if isinstance(self, LLMAlgorithm):
            if hasattr(self.actor, "optimizer"):
                optimizer = (
                    self.actor.optimizer
                )  # If the optimizer is defined in the deepspeed config, we do this
            else:
                optimizer = opt.optimizer

            if isinstance(config.lr, tuple):
                lr_actor = getattr(self, config.lr[0])
                lr_critic = getattr(self, config.lr[1])
                self.accelerator, self.lr_scheduler = LLMAlgorithm.update_lr(
                    optimizer,
                    lr=lr_actor,
                    lr_critic=lr_critic,
                    accelerator=self.accelerator,
                    scheduler_config=self.cosine_lr_schedule_config,
                )
            else:
                self.accelerator, self.lr_scheduler = LLMAlgorithm.update_lr(
                    optimizer,
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
        """Set the training mode of the algorithm.

        :param training: If True, set the algorithm to training mode.
        :type training: bool
        """
        self.training = training
        for name, network in self.evolvable_attributes(networks_only=True).items():
            if "actor" in name:
                network.train(mode=training)

    def get_lr_names(self) -> list[str]:
        """Return the learning rates of the algorithm."""
        return [opt.lr for opt in self.registry.optimizers]

    def register_network_group(self, group: NetworkGroup) -> None:
        """Set the evaluation network for the algorithm.

        :param name: The name of the evaluation network.
        :type name: str
        """
        self.registry.register_group(group)

    def register_mutation_hook(self, hook: Callable) -> None:
        """Register a hook to be executed after a mutation is performed on
        the algorithm.

        :param hook: The hook to be executed after mutation.
        :type hook: Callable
        """
        self.registry.register_hook(hook)

    def mutation_hook(self) -> None:
        """Execute the hooks registered with the algorithm."""
        for hook in self.registry.hooks:
            getattr(self, hook)()

    def get_policy(self) -> EvolvableModuleProtocol:
        """Return the policy network of the algorithm."""
        for group in self.registry.groups:
            if group.policy:
                return getattr(self, group.eval_network)

        msg = "No policy network has been registered with the algorithm."
        raise AttributeError(
            msg,
        )

    def reinit_optimizers(
        self,
        optimizer: OptimizerConfig | None = None,
    ) -> None:
        """Reinitialize the optimizers of an algorithm. If no optimizer is passed, all optimizers are reinitialized.

        :param optimizer: The optimizer to reinitialize, defaults to None, in which case
            all optimizers are reinitialized.
        :type optimizer: OptimizerConfig | None, optional
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
        """Move experiences to the device.

        :param experiences: Experiences to move to device
        :type experiences: tuple[torch.Tensor[float], ...]

        :return: Experiences on the device
        :rtype: tuple[torch.Tensor[float], ...]
        """
        device = self.device if self.accelerator is None else self.accelerator.device
        on_device = []
        for exp in experiences:
            if isinstance(exp, dict):
                moved = {key: val.to(device) for key, val in exp.items()}
            elif isinstance(exp, (list, tuple)) and isinstance(exp[0], torch.Tensor):
                moved = tuple(val.to(device) for val in exp)
            elif isinstance(exp, torch.Tensor):
                moved = exp.to(device)
            else:
                moved = exp
            on_device.append(moved)

        return on_device

    def evolvable_attributes(
        self,
        networks_only: bool = False,
    ) -> EvolvableAttributeDict:
        """Return the attributes related to the evolvable networks in the algorithm. Includes
        attributes that are either EvolvableModule or ModuleDict objects, as well as the optimizers
        associated with the networks.

        :param networks_only: If True, only include evolvable networks, defaults to False
        :type networks_only: bool, optional

        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """

        def is_evolvable(attr: str, obj: Any) -> bool:
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
        """Wrap the models in the algorithm with the accelerator."""
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
            msg = "No accelerator has been set for the algorithm."
            raise AttributeError(msg)

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
        self,
        index: int | None = None,
        wrap: bool = True,
    ) -> Self:
        """Create a clone of the algorithm.

        :param index: The index of the clone, defaults to None
        :type index: int | None, optional
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
            optim_cls = opt_config.get_optimizer_cls()
            if getattr(orig_optimizer, "use_llm_param_groups", False) and isinstance(
                opt_config.lr,
                tuple,
            ):
                opt = OptimizerWrapper(
                    optim_cls,
                    networks=networks,
                    lr=getattr(clone, opt_config.lr[0]),
                    lr_critic=getattr(clone, opt_config.lr[1]),
                    use_llm_param_groups=True,
                    network_names=opt_config.networks,
                    lr_name=opt_config.lr,
                    optimizer_kwargs=opt_config.optimizer_kwargs,
                )
            else:
                opt = OptimizerWrapper(
                    optim_cls,
                    networks=networks,
                    lr=getattr(clone, opt_config.lr),
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
        """Save a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        torch.save(
            get_checkpoint_dict(self),
            path,
            pickle_module=dill,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint: dict[str, Any] = torch.load(
            path,
            map_location=self.device,
            pickle_module=dill,
            weights_only=False,
        )

        # Recreate evolvable modules
        network_info: dict[str, dict[str, Any]] = checkpoint["network_info"]
        network_names = network_info["network_names"]
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info["modules"].items() if k.startswith(name)
            }

            module_cls = net_dict.get(f"{name}_cls")
            if module_cls is None:
                # This allows us to super this method in the LLMAlgorithm class
                # as we don't want to reinstantiate the network in this class
                break
            init_dict = net_dict[f"{name}_init_dict"]

            module_dict_cls = net_dict.get(f"{name}_module_dict_cls")
            if isinstance(module_cls, dict):
                loaded_modules = {}
                for agent_id, mod in module_cls.items():
                    init_dict[agent_id]["device"] = self.device
                    loaded_modules[agent_id] = mod(**init_dict[agent_id])

                setattr(self, name, module_dict_cls(loaded_modules))
            else:
                init_dict["device"] = self.device
                loaded_module: EvolvableModuleProtocol = module_cls(**init_dict)
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
            if isinstance(loaded_module, ModuleDictProtocol):
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
            use_llm_param_groups = opt_dict.get(f"{name}_use_llm_param_groups", False)
            networks = [getattr(self, net) for net in opt_networks]
            optimizer = OptimizerWrapper(
                optimizer_cls=optimizer_cls,
                networks=networks,
                lr=getattr(self, opt_lr),
                optimizer_kwargs=opt_kwargs,
                network_names=opt_networks,
                lr_name=opt_lr,
                use_llm_param_groups=use_llm_param_groups,
                lr_critic=getattr(self, "lr_critic", None),
            )

            # Load optimizer state
            optimizer.load_state_dict(opt_dict[f"{name}_state_dict"])
            setattr(self, name, optimizer)

        # Check loaded registry is consistent with the algorithm
        if checkpoint["registry"] != self.registry:
            msg = (
                "Loaded registry does not match the algorithm's registry. Please make "
                "sure you are loading the checkpoint with the correct algorithm."
            )
            raise ValueError(
                msg,
            )

        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(state_dict=checkpoint["lr_scheduler"])
            checkpoint.pop("lr_scheduler")

        # Load other attributes
        checkpoint.pop("network_info")
        for attribute, value in checkpoint.items():
            if isinstance(value, torch.Tensor) and isinstance(
                getattr(self, attribute, None), torch.Tensor
            ):
                value = value.to(getattr(self, attribute).device)
            setattr(self, attribute, value)

        # Wrap models / compile if necessary
        if self.accelerator is not None:
            self.wrap_models()
        elif self.torch_compiler:
            torch.set_float32_matmul_precision("high")
            self.recompile()

    @classmethod
    def load(
        cls,
        path: str,
        device: DeviceType = "cpu",
        accelerator: Accelerator | None = None,
    ) -> Self:
        """Load an algorithm from a checkpoint.

        :param path: Location to load checkpoint from.
        :type path: string
        :param device: Device to load the algorithm on, defaults to 'cpu'
        :type device: str, optional
        :param accelerator: Accelerator object for distributed computing, defaults to None
        :type accelerator: Accelerator | None, optional

        :return: An instance of the algorithm
        :rtype: RLAlgorithm
        """
        from agilerl.modules import EvolvableModule, ModuleDict

        checkpoint: dict[str, Any] = torch.load(
            path,
            map_location=device,
            pickle_module=dill,
            weights_only=False,
        )

        # Reconstruct evolvable modules in algorithm
        network_info: dict[str, dict[str, Any]] | None = checkpoint.get("network_info")
        if network_info is None:
            msg = (
                "Network info not found in checkpoint. You may be loading a checkpoint from "
                "an older version of AgileRL. Since v2.0, we require AgileRL algorithms to "
                "have a specific structure to simplify evolutionary hyperparameter optimization. "
                "Please downgrade to v1.0.30 to load checkpoints from before this change."
            )
            raise ValueError(
                msg,
            )

        network_names = network_info["network_names"]
        loaded_modules: dict[str, EvolvableAttributeType] = {}
        for name in network_names:
            net_dict = {
                k: v for k, v in network_info["modules"].items() if k.startswith(name)
            }

            # Add device to init dict
            init_dict = net_dict.get(f"{name}_init_dict")
            if init_dict is None:
                msg = f"Init dict for {name} not found in checkpoint."
                raise ValueError(msg)

            init_dict = chkpt_attribute_to_device(init_dict, device)

            # Reconstruct the module dict class if necessary
            module_dict_cls = net_dict.get(f"{name}_module_dict_cls")
            if module_dict_cls is not None:
                loaded_modules[name] = module_dict_cls()

            # Reconstruct the modules
            module_cls: type[EvolvableModule] | dict[str, type[EvolvableModule]] = (
                net_dict[f"{name}_cls"]
            )
            if isinstance(module_cls, dict):
                for agent_id, mod_cls in module_cls.items():
                    d = filter_init_dict(init_dict[agent_id], mod_cls)
                    d["device"] = device
                    mod: EvolvableModule = mod_cls(**d)
                    loaded_modules[name][agent_id] = mod
            else:
                init_dict = filter_init_dict(init_dict, module_cls)
                init_dict["device"] = device
                module = module_cls(**init_dict)
                loaded_modules[name] = module

        # Reconstruct the algorithm
        checkpoint["accelerator"] = accelerator
        checkpoint["device"] = device
        class_init_dict = filter_init_dict(checkpoint, cls)
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
            loaded_module: EvolvableModule | ModuleDict = getattr(self, name)
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
                opt_dict[f"{name}_state_dict"],
                device,
            )
            optimizer.load_state_dict(state_dict)
            loaded_optimizers[name] = optimizer

        # Assign loaded modules and optimizers to the algorithm
        for name, module in loaded_modules.items():
            setattr(self, name, module)

        for name, optimizer in loaded_optimizers.items():
            setattr(self, name, optimizer)

        # Assign other attributes to the algorithm
        for attribute in EvolvableAlgorithm.inspect_attributes(self):
            if attribute not in checkpoint:
                warnings.warn(
                    f"Attribute {attribute} not found in checkpoint. Skipping.",
                    stacklevel=2,
                )
                continue

            value = checkpoint.get(attribute)
            if isinstance(value, torch.Tensor) and isinstance(
                getattr(self, attribute, None), torch.Tensor
            ):
                value = value.to(getattr(self, attribute).device)
            setattr(self, attribute, value)

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

    def clean_up(self) -> None:
        """Clean up the algorithm by deleting the networks and optimizers.

        :return: None
        :rtype: None
        """
        for attr_name in self.evolvable_attributes():
            delattr(self, attr_name)


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
    :type device: str | torch.device, optional
    :param accelerator: Accelerator object for distributed computing, defaults to None.
    :type accelerator: Accelerator | None, optional
    :param normalize_images: If True, normalize images, defaults to True.
    :type normalize_images: bool, optional
    :param name: Name of the algorithm, defaults to the class name.
    :type name: str | None, optional
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        index: int,
        hp_config: HyperparameterConfig | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        torch_compiler: Any | None = None,
        normalize_images: bool = True,
        name: str | None = None,
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
    :type observation_spaces: list[spaces.Space] | spaces.Dict
    :param action_spaces: The action spaces of the agent environments.
    :type action_spaces: list[spaces.Space] | spaces.Dict
    :param index: The index of the individual in the population.
    :type index: int.
    :param agent_ids: The agent IDs of the agents in the environment.
    :type agent_ids: list[int] | None, optional
    :param learn_step: Learning frequency, defaults to 2048
    :type learn_step: int, optional
    :param device: Device to run the algorithm on, defaults to "cpu"
    :type device: str, optional
    :param accelerator: Accelerator object for distributed computing, defaults to None
    :type accelerator: Accelerator | None, optional
    :param torch_compiler: The torch compiler mode to use, defaults to None
    :type torch_compiler: Any | None, optional
    :param normalize_images: If True, normalize images, defaults to True
    :type normalize_images: bool, optional
    :param placeholder_value: The value to use as placeholder for missing observations, defaults to -1.
    :type placeholder_value: Any | None, optional
    :param name: Name of the algorithm, defaults to the class name
    :type name: str | None, optional
    """

    possible_observation_spaces: dict[str, spaces.Space]
    possible_action_spaces: dict[str, spaces.Space]

    shared_agent_ids: list[str]
    grouped_agents: dict[str, list[str]]
    unique_observation_spaces: dict[str, spaces.Space]
    unique_action_spaces: dict[str, spaces.Space]

    def __init__(
        self,
        observation_spaces: Iterable[spaces.Space] | spaces.Dict,
        action_spaces: Iterable[spaces.Space] | spaces.Dict,
        index: int,
        agent_ids: Iterable[int] | None = None,
        hp_config: HyperparameterConfig | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        torch_compiler: Any | None = None,
        normalize_images: bool = True,
        placeholder_value: Any | None = -1,
        name: str | None = None,
    ) -> None:

        super().__init__(index, hp_config, device, accelerator, torch_compiler, name)

        assert type(observation_spaces) is type(action_spaces), (
            "Observation spaces and action spaces must be the same type. "
            f"Got {type(observation_spaces)} and {type(action_spaces)}."
        )

        if isinstance(observation_spaces, (list, tuple)):
            assert isinstance(
                agent_ids,
                (tuple, list),
            ), "Agent IDs must be specified if observation spaces are passed as a list."
            assert len(agent_ids) == len(
                observation_spaces,
            ), "Number of agent IDs must match number of observation spaces."

            self.possible_observation_spaces = spaces.Dict(
                dict(zip(agent_ids, observation_spaces, strict=False)),
            )
            self.possible_action_spaces = spaces.Dict(
                dict(zip(agent_ids, action_spaces, strict=False)),
            )
        elif isinstance(observation_spaces, (spaces.Dict, dict)):
            if isinstance(observation_spaces, dict):
                observation_spaces = spaces.Dict(observation_spaces)
                action_spaces = spaces.Dict(action_spaces)

            self.possible_observation_spaces = observation_spaces
            self.possible_action_spaces = action_spaces
        else:
            msg = f"Observation spaces must be a list or dictionary of spaces.Space objects. Got {type(observation_spaces)}."
            raise TypeError(msg)

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
                msg = f"Unknown observation space type: {type(obs_space)}"
                raise ValueError(msg)

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

        from agilerl.modules import ModuleDict

        # Additional check to ensure multi-agent networks are initialized with valid keys
        for name, network in self.evolvable_attributes(networks_only=True).items():
            if isinstance(network, ModuleDict):
                for key in network:
                    if (key not in self.agent_ids) and (
                        key not in self.shared_agent_ids
                    ):
                        msg = (
                            f"Network '{name}' contains key '{key}' which is not present in `self.agent_ids` "
                            f"or `self.shared_agent_ids`. Please initialize multi-agent networks through agilerl.modules.ModuleDict "
                            "objects with the agent or group/shared IDs as keys."
                        )
                        raise ValueError(
                            msg,
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
        self,
        observation: ObservationType,
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
        """Extract action masks from info dictionary.

        :param infos: Info dict
        :type infos: dict[str, dict[...]]

        :return: Action masks
        :rtype: dict[str, np.ndarray]
        """
        # Get dict of form {"agent_id" : [1, 0, 0, 0]...} etc
        return {
            agent: info.get("action_mask", None) if isinstance(info, dict) else None
            for agent, info in infos.items()
            if agent in self.agent_ids
        }

    def extract_agent_masks(
        self,
        infos: InfosDict | None = None,
    ) -> tuple[ArrayDict, ArrayDict]:
        """Extract env_defined_actions from info dictionary and determine agent masks.

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
            for agent_id, action_val in list(env_defined_actions.items()):
                val = action_val
                # Handle None if environment isn't vectorized
                if val is None:
                    if not isinstance(
                        self.possible_action_spaces[agent_id],
                        spaces.Discrete,
                    ):
                        nan_arr = np.empty(self.action_dims[agent_id])
                        nan_arr[:] = np.nan
                    else:
                        nan_arr = np.array([np.nan])

                    env_defined_actions[agent_id] = nan_arr
                    val = nan_arr

                # Handle discrete actions + env not vectorized
                if isinstance(val, (int, float)):
                    val = np.array([val])
                    env_defined_actions[agent_id] = val

                agent_masks[agent_id] = np.where(
                    np.isnan(env_defined_actions[agent_id]),
                    0,
                    1,
                ).astype(bool)

        return env_defined_actions, agent_masks

    def build_net_config(
        self,
        net_config: NetConfigType | None = None,
        flatten: bool = True,
        return_encoders: bool = False,
    ) -> NetConfigType | tuple[NetConfigType, dict[str, NetConfigType]]:
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
        :type net_config: NetConfigType | None
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

            if config_key not in encoder_configs or (
                isinstance(config, MlpNetConfig)
                and len(config["hidden_size"])
                > len(
                    encoder_configs["mlp_config"]["hidden_size"],
                )
            ):
                encoder_configs[config_key] = asdict(config)

        # Helper function to check if any agent ID exists in the net_config
        def _has_agent_ids(config: NetConfigType) -> bool:
            return any(
                (agent_id in self.agent_ids) or (agent_id in self.shared_agent_ids)
                for agent_id in config
            )

        # Helper function to get or create encoder config for an agent
        def _get_encoder_config(config: NetConfigType, agent_id: str) -> NetConfigType:
            encoder_config = config.get("encoder_config")
            simba = config.get("simba", False)
            if encoder_config is None:
                encoder_config = get_default_encoder_config(
                    observation_spaces[agent_id],
                    simba,
                )
                config["encoder_config"] = encoder_config

            return encoder_config

        # 1. net_config is None -> Automatically define an encoder for each sub-agent or group
        if net_config is None:
            net_config = defaultdict(OrderedDict)
            for agent_id in agent_ids:
                encoder_config = get_default_encoder_config(
                    observation_spaces[agent_id],
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
            agent_id in self.agent_ids and grouped_config for agent_id in net_config
        ):
            msg = (
                "Found key in net_config corresponding to an individual sub-agent in a grouped setting. "
                "Please specify the configuration for groups instead (e.g. {'agent': {...}, ...} rather than {'agent_0': {...}, ...})"
            )
            raise KeyError(
                msg,
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
                    observation_spaces[agent_id],
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

                if (
                    isinstance(self.possible_action_spaces[agent_id], spaces.Discrete)
                    and output_dict[agent_id].shape[-1] == 1
                ):
                    output_dict[agent_id] = output_dict[agent_id].squeeze(-1)

        return output_dict

    def sum_shared_rewards(self, rewards: ArrayDict) -> ArrayDict:
        """Sum the rewards for grouped agents.

        :param rewards: Reward dictionary from environment
        :type rewards: dict[str, np.ndarray]
        :return: Summed rewards dictionary
        :rtype: dict[str, np.ndarray]
        """
        reward_shape = next(iter(rewards.values()))
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
        self,
        agent_outputs: ArrayDict,
        vect_dim: int,
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
            group_agent_outputs = [
                agent_outputs[group]
                for group in self.grouped_agents[group_id]
                if group in agent_outputs
            ]

            if group_agent_outputs:
                # Stack outputs along first dimension
                stacked_outputs = np.stack(group_agent_outputs, axis=0)
                # Reshape into a form suitable for batch processing
                group_outputs[group_id] = np.reshape(
                    stacked_outputs,
                    (len(group_agent_outputs) * vect_dim, -1),
                )

        return group_outputs


class LLMAlgorithm(EvolvableAlgorithm, ABC):
    """Base object for all LLM algorithms in the AgileRL framework.

    :param index: The index of the algorithm.
    :type index: int
    :param batch_size: The batch size.
    :type batch_size: int
    :param lr: The learning rate.
    :type lr: float
    :param max_grad_norm: The maximum gradient norm.
    :type max_grad_norm: float
    :param clone: Whether to clone the model.
    :type clone: bool
    :param calc_position_embeddings: Whether to calculate position embeddings.
    :type calc_position_embeddings: bool
    :param seed: The seed.
    :type seed: int
    :param pad_token_id: The pad token id.
    :type pad_token_id: int
    :param pad_token: The pad token.
    :type pad_token: str
    :param use_liger_loss: Whether to use Liger loss.
    :type use_liger_loss: bool
    :param lora_config: The LoRA config.
    :type lora_config: LoraConfigProtocol | None
    :param use_separate_reference_adapter: Whether to use a separate reference adapter.
    :type use_separate_reference_adapter: bool
    :param use_value_head: Whether to use a separate value head.
    :type use_value_head: bool
    :param model_name: The name of the model.
    :type model_name: str | None
    :param actor_network: The actor network.
    :type actor_network: PreTrainedModelProtocol | None
    :param micro_batch_size_per_gpu: The micro batch size per GPU.
    :type micro_batch_size_per_gpu: int | None
    :param cosine_lr_schedule_config: The cosine LR schedule config.
    :type cosine_lr_schedule_config: CosineLRScheduleConfig | None
    :param hp_config: The hyperparameter configuration.
    :type hp_config: Optional[HyperparameterConfig]
    :param wrap: Whether to wrap the model.
    :type wrap: bool
    :param device: The device to run the algorithm on.
    :type device: str | torch.device
    :param accelerator: The accelerator to use.
    :type accelerator: Accelerator | None
    :param name: The name of the algorithm.
    :type name: str | None
    :param model_config: The configuration for the model.
    :type model_config: dict[str, Any] | PretrainedConfig | None
    :param gradient_checkpointing: Whether to use gradient checkpointing.
    :type gradient_checkpointing: bool
    :param torch_compiler: The torch compiler mode to use ('default',
        'reduce-overhead', or 'max-autotune'), defaults to None.
    :type torch_compiler: str | None, optional
    :param reduce_memory_peak: Deprecated. Previously hinted peak-memory batching;
        ignored. Configure ``micro_batch_size_per_gpu`` and DeepSpeed instead.
    :type reduce_memory_peak: bool, optional
    """

    _separate_reference_adapter_deprecation_emitted = False
    _allowed_adapters = frozenset({"actor", "reference", "critic"})

    # Typed as non-optional: the actor/optimizer are always assigned by
    # ``_initialize_actors`` before any learn/get_action call. :meth:`clean_up` rebinds
    # them to ``None`` purely to drop GPU refs — callers must not touch the algorithm
    # after that.
    actor: PeftModelProtocol
    optimizer: OptimizerWrapper
    lr_scheduler: SequentialLR | None

    def __init__(
        self,
        index: int,
        batch_size: int,
        lr: float,
        max_grad_norm: float,
        clone: bool,
        calc_position_embeddings: bool,
        seed: int,
        pad_token_id: int,
        pad_token: str,
        use_liger_loss: bool,
        lora_config: LoraConfig | None,
        use_separate_reference_adapter: bool = False,
        lr_critic: float | None = None,
        use_value_head: bool = False,
        use_vllm: bool = False,
        vllm_config: VLLMConfig | None = None,
        model_name: str | None = None,
        actor_network: PreTrainedModelProtocol | None = None,
        micro_batch_size_per_gpu: int | None = None,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None = None,
        hp_config: HyperparameterConfig | None = None,
        use_memory_efficient_params: bool = True,
        wrap: bool = True,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        name: str | None = None,
        model_config: dict[str, Any] | PretrainedConfigProtocol | None = None,
        gradient_checkpointing: bool = True,
        torch_compiler: str | None = None,
        reduce_memory_peak: bool = False,
    ) -> None:
        if not HAS_LLM_DEPENDENCIES:
            msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
            raise ImportError(msg)
        if reduce_memory_peak:
            warnings.warn(
                "reduce_memory_peak is deprecated and has no effect; configure batch "
                "size via micro_batch_size_per_gpu and DeepSpeed settings instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if use_liger_loss and not HAS_LIGER_KERNEL:
            warnings.warn(
                "use_liger_loss=True requested, but `liger-kernel` is not available on this platform/environment. "
                "Falling back to standard loss.",
                stacklevel=2,
            )
            use_liger_loss = False

        if model_name is None and actor_network is None:
            msg = "At least one of model_name or actor_network must be provided."
            raise ValueError(
                msg,
            )

        if lora_config is None:
            warnings.warn(
                "No LoRA config provided. AgileRL can only be used to finetune adapters at present. "
                "Using default LoRA configuration for RL finetuning: "
                "r=16, lora_alpha=64, target_modules='all-linear', task_type='CAUSAL_LM', lora_dropout=0.05."
                "To use a different LoRA configuration, please pass lora_config to the constructor.",
                stacklevel=2,
            )
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            )
        if use_liger_loss:
            warnings.warn(
                "Liger Loss used with LoRA, deactivating LoRA for the lm_head by setting exclude_modules to ['lm_head']",
                stacklevel=2,
            )
            lora_config.exclude_modules = ["lm_head"]

        if use_memory_efficient_params and use_vllm and not vllm_config.sleep_mode:
            warnings.warn(
                "Memory efficient params is only supported when using vLLM in sleep mode."
                "Setting use_memory_efficient_params to False.",
                stacklevel=2,
            )
            use_memory_efficient_params = False

        if use_memory_efficient_params and not use_vllm:
            warnings.warn(
                "Memory efficient params is only supported when using vLLM."
                "Setting use_memory_efficient_params to False.",
                stacklevel=2,
            )
            use_memory_efficient_params = False

        if vllm_config is not None and not use_vllm:
            warnings.warn(
                "vllm_config is provided but use_vllm is False. Setting vllm_config to None.",
                stacklevel=2,
            )
            vllm_config = None

        super().__init__(index, hp_config, device, accelerator, torch_compiler, name)
        self.gradient_checkpointing = gradient_checkpointing
        self.use_liger_loss = use_liger_loss
        self.zero_stage = None
        self.reference_update_tracker = 0  # Updated every time the reference policy is updated which is updated each time we pass through the train dataset
        self.calc_position_embeddings = calc_position_embeddings
        self.pad_token_id = pad_token_id
        self.pad_token = pad_token
        self.pretrained_model_name_or_path = (
            model_name
            if model_name is not None
            else get_model_name_or_path(actor_network)
        )
        self.model_config = model_config
        self._configure_batch_size_per_process(
            batch_size,
            micro_batch_size_per_gpu,
        )
        self.batch_size = batch_size
        self.lr = align_deepspeed_lr(float(lr), self.accelerator)
        self.lr_critic = lr_critic

        if self.accelerator is not None:
            ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
            if ds_plugin is not None:
                ds_config = ds_plugin.deepspeed_config
                if max_grad_norm is not None:
                    if accelerator.is_main_process:
                        warnings.warn(
                            "Argument 'max_grad_norm' will overwrite the equivalent value set for 'gradient_clipping' in the deepspeed config.",
                            stacklevel=2,
                        )
                    ds_config["gradient_clipping"] = max_grad_norm
                if (
                    cosine_lr_schedule_config is not None
                    and accelerator.is_main_process
                ):
                    warnings.warn(
                        "Cannot specify the optimizer in the DeepSpeed config and use AgileRL's LR scheduler. "
                        "If you want to use LR scheduling, please specify in the DeepSpeed config. "
                        "Setting LR scheduler to None.",
                        stacklevel=2,
                    )
                    cosine_lr_schedule_config = None
                self.register_mutation_hook(self._sync_deepspeed_gradient_clipping)
                self.zero_stage = ds_config["zero_optimization"]["stage"]
                if (
                    self.zero_stage is not None
                    and self.zero_stage > 2
                    and self.accelerator.is_main_process
                ):
                    warnings.warn(
                        "DeepSpeed ZeRO Stage 3 is nascent and may not work as expected, proceed with caution when using this feature.",
                        stacklevel=2,
                    )
            if self.accelerator.num_processes > 1:
                seed = broadcast_object_list([seed], from_process=0)[0]
            seed += self.accelerator.process_index
            set_seed(seed)

        # YAML / config loaders may supply LR as a string (e.g. "5e-5"); PyTorch optimizers require float.
        self.lora_config = lora_config
        self.use_vllm = use_vllm
        self.vllm_config = vllm_config
        self.max_grad_norm = max_grad_norm
        self.use_memory_efficient_params = use_memory_efficient_params
        self.memory_efficient_params_context = (
            self._memory_efficient_params
            if use_memory_efficient_params
            else nullcontext
        )
        self.wrap = wrap
        self.use_separate_reference_adapter = use_separate_reference_adapter
        self._warn_separate_reference_adapter_deprecation()

        selected_adapters = ("actor",)
        if use_separate_reference_adapter:
            selected_adapters += ("reference",)
        if use_value_head:
            selected_adapters += ("critic",)
        self.selected_adapters = selected_adapters

        self.cosine_lr_schedule_config = cosine_lr_schedule_config
        self.use_value_head = use_value_head
        self._uses_deepspeed = (
            self.accelerator is not None
            and getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self._vllm_awake = self.use_vllm and not self.vllm_config.sleep_mode
        self._vllm_moved = False
        self.rng = np.random.RandomState(seed)

    def preprocess_observation(self, observation: ObservationType) -> TorchObsType:
        """Preprocess observations (dummy) for forward pass through neural network.

        :param observations: Observations of environment
        :type observations: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or tuple[torch.Tensor[float], ...]
        """
        return cast("TorchObsType", observation)

    def _warn_separate_reference_adapter_deprecation(self) -> None:
        """Warn once per process about the pending adapter-mode deprecation."""
        if not self.use_separate_reference_adapter:
            return
        if LLMAlgorithm._separate_reference_adapter_deprecation_emitted:
            return
        warnings.warn(
            "`use_separate_reference_adapter=True` is deprecated and will be "
            "removed in a future release. Prefer using LoRA adapters while "
            "keeping the base model untouched.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        LLMAlgorithm._separate_reference_adapter_deprecation_emitted = True

    def save_checkpoint(
        self,
        path: str,
        lora_only: bool = True,
        save_optimizer: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save adapter weights and algorithm state to a directory.

        AgileRL never persists base-model weights when ``lora_only=True`` for
        LLM algorithms: a checkpoint is a directory containing

          * ``<adapter>/adapter_model.safetensors`` + ``adapter_config.json`` —
            one subdirectory per adapter in :attr:`selected_adapters` (always
            ``actor``, plus ``reference`` / ``critic`` when those adapters are
            configured). Written only when ``lora_only=True``.
          * ``attributes.pt`` — algorithm hyperparameters, plus (optionally)
            the actor state dict and/or optimizer state dict depending on the
            cell below. Always present.
          * ``save_checkpoint/`` — DeepSpeed ZeRO \u2265 2 sharded-checkpoint
            output. Present only when an :class:`~accelerate.Accelerator` is
            attached and ``save_optimizer=True``.

        Behaviour per cell of the ``(lora_only, save_optimizer, deepspeed)``
        grid:

          Plain (no accelerator):
            lora_only=T, save_optimizer=T  \u2192  PEFT adapter dirs on disk +
                                                 optimizer state in ``attributes.pt``
            lora_only=T, save_optimizer=F  \u2192  PEFT adapter dirs only
            lora_only=F, save_optimizer=T  \u2192  full actor state_dict +
                                                 optimizer state in ``attributes.pt``
            lora_only=F, save_optimizer=F  \u2192  full actor state_dict in ``attributes.pt``

          DeepSpeed:
            lora_only=T, save_optimizer=T  \u2192  engine tag dir (frozen params
                                                 excluded) + PEFT adapter dirs
            lora_only=T, save_optimizer=F  \u2192  PEFT adapter dirs only
            lora_only=F, save_optimizer=T  \u2192  engine tag dir (frozen params
                                                 included)
            lora_only=F, save_optimizer=F  \u2192  gathered (ZeRO-3 aware) actor
                                                 state_dict injected into
                                                 ``attributes.pt``

        :param path: Directory to write the checkpoint into.
        :type path: str
        :param lora_only: If ``True`` (default) only adapter weights are
            written to disk via ``save_pretrained``; the base model is shared
            across checkpoints and not serialised. If ``False``, the full
            actor state dict is persisted (into ``attributes.pt`` on the plain
            path, or into the DeepSpeed engine's tag dir / gathered dict on
            the distributed path).
        :type lora_only: bool
        :param save_optimizer: If ``True`` (default) also persist the
            optimizer and LR scheduler state so training can resume. On
            DeepSpeed ZeRO \u2265 2 this writes a sharded checkpoint into
            ``<path>/save_checkpoint``; otherwise optimizer state is included
            in ``attributes.pt``.
        :type save_optimizer: bool
        """
        if "weights_only" in kwargs:
            warnings.warn(
                "weights_only is deprecated and will be removed in a future release. Use lora_only instead.",
                stacklevel=2,
                category=DeprecationWarning,
            )
            lora_only = kwargs["weights_only"]
        if lora_only and not self.use_separate_reference_adapter:
            warnings.warn(
                "lora_only=True requested, but use_separate_reference_adapter is False; base model (reference) weights will not be saved.",
                stacklevel=2,
                category=UserWarning,
            )

        Path(path).mkdir(parents=True, exist_ok=True)

        # omit_actor_info: actor state goes into attributes.pt only when we
        # want a full-model torch save on the plain (non-deepspeed) path.
        #   * lora_only=True  → adapter weights saved via PEFT on disk; no actor in attrs.pt.
        #   * deepspeed        → actor state either lives in the engine's tag dir
        #                        (save_optimizer=True) or is gathered and injected
        #                        via the manual state_dict path below (F, F).
        #   * plain + lora_only=False → full state_dict round-trips through attrs.pt.
        omit_actor_info = lora_only or self._uses_deepspeed
        omit_optimizer_info = True
        state_dict = {}
        if save_optimizer:
            if self.accelerator is not None:
                # Save deepspeed checkpoint with lora_only=True
                self._save_distributed_actor(
                    path, tag="save_checkpoint", lora_only=lora_only
                )
            else:
                omit_optimizer_info = False

        if lora_only:
            model_ref = self._get_unwrapped_actor()
            with gather_if_zero3(self.zero_stage, list(model_ref.parameters())):
                model_ref.save_pretrained(
                    save_directory=path,
                    selected_adapters=self.selected_adapters,
                    is_main_process=self.accelerator is None
                    or self.accelerator.is_main_process,
                )

        elif self._uses_deepspeed and not save_optimizer:
            # (lora_only=False, save_optimizer=False, deepspeed): the ZeRO-3
            # shards aren't materialised in the default module loop, so gather
            # manually and inject the state_dict into attributes.pt.
            model_ref = self._get_unwrapped_actor()
            with gather_if_zero3(self.zero_stage, list(model_ref.parameters())):
                module_cls = model_ref.__class__
                state_dict = {
                    "actor_cls": module_cls,
                    "actor_init_dict": None,
                    "actor_state_dict": model_ref.state_dict(),
                    "actor_module_dict_cls": None,
                }

        # Build the checkpoint payload saved alongside adapter weights.
        checkpoint_dict = get_checkpoint_dict(
            self,
            omit_actor_info=omit_actor_info,
            omit_optimizer_info=omit_optimizer_info,
        )
        checkpoint_dict.pop("llm", None)
        checkpoint_dict.pop("tp_group", None)
        checkpoint_dict["_lora_only"] = lora_only
        if state_dict:
            checkpoint_dict["network_info"] = {}
            checkpoint_dict["network_info"]["modules"] = {}
            checkpoint_dict["network_info"]["modules"] = state_dict

        # Persist non-model attributes to ``attributes.pt``.
        # In distributed runs only the main process writes the file.
        if self.accelerator is None or self.accelerator.is_main_process:
            checkpoint_path = Path(path) / "attributes.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                checkpoint_dict,
                str(checkpoint_path),
                pickle_module=dill,
            )

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        overwrite_reference_adapter: bool = False,
        overwrite_critic_adapter: bool = True,
    ) -> None:
        """Load adapter weights and algorithm state from a checkpoint directory.

        Adapter roles restored on load:

          * ``actor``     — the trained policy. Always loaded.
          * ``reference`` — the fixed policy used for KL / comparison. The
            checkpoint's ``actor`` adapter is copied onto ``reference`` so
            that SFT \u2192 DPO \u2192 GRPO chains work out of the box: the stage-N
            actor becomes the stage-N+1 reference.
          * ``critic``    — optional value head. Loaded from disk if a
            ``critic/`` adapter is present, else copied from ``actor``, else
            left as the live fresh LoRA init.

        LoRA config reconciliation: when the checkpoint's config and the live
        algorithm's config disagree (e.g. after a rank mutation between
        runs), the two are merged non-destructively:

          * ``r`` (rank) \u2192 ``max(current, checkpoint)``; the smaller side's
            weights are padded into the top-left rank slice of the larger
            adapter (see :meth:`_pad_adapter_state_to_live_shape`).
          * ``target_modules`` / ``modules_to_save`` \u2192 union.
          * Any other mismatched field \u2192 current value wins, with a warning.

        Any adapter whose live config ends up differing from the merged
        result is rebuilt via :meth:`_reconfigure_adapters_to_match` before
        weights are loaded, so tensors always land in the correct shape.

        Dispatch per cell of the ``(lora_only-on-disk, load_optimizer, deepspeed)``
        grid, where ``lora_only-on-disk`` is read from ``attributes.pt``:

          Plain (no accelerator):
            lora_only=T, load_optimizer=T  \u2192  PEFT adapter load + optimizer
                                                 state from ``attributes.pt``
            lora_only=T, load_optimizer=F  \u2192  PEFT adapter load only
            lora_only=F, load_optimizer=T  \u2192  torch load of actor +
                                                 optimizer from ``attributes.pt``
            lora_only=F, load_optimizer=F  \u2192  torch load of actor only

          DeepSpeed:
            lora_only=T, load_optimizer=T  \u2192  DeepSpeed engine load from
                                                 ``<path>/save_checkpoint``
            lora_only=T, load_optimizer=F  \u2192  PEFT adapter load
            lora_only=F, load_optimizer=T  \u2192  DeepSpeed engine load from
                                                 ``<path>/save_checkpoint``
            lora_only=F, load_optimizer=F  \u2192  ``actor.load_state_dict(...)``
                                                 from ``attributes.pt``

        When ``load_optimizer=True`` but the checkpoint contains no optimizer
        state (e.g. it was saved with ``save_optimizer=False``), a
        ``UserWarning`` is emitted and a freshly-initialised optimizer is
        used.

        :param path: Directory containing a checkpoint written by
            :meth:`save_checkpoint`.
        :type path: str
        :param load_optimizer: If ``True`` (default) also load the optimizer
            and LR scheduler state so training can resume. On DeepSpeed ZeRO
            \u2265 2 this reads a sharded checkpoint from
            ``<path>/save_checkpoint``; otherwise optimizer state is read
            from ``attributes.pt``.
        :type load_optimizer: bool
        """
        pickle_module = dill if self.accelerator is None else pickle
        checkpoint = torch.load(
            str(Path(path) / "attributes.pt"),
            weights_only=False,
            pickle_module=pickle_module,
        )

        lora_only = checkpoint.pop("_lora_only", False) or checkpoint.pop(
            "_weights_only", False
        )
        if self._uses_deepspeed:
            if load_optimizer:
                self._load_distributed_actor(path, tag="save_checkpoint")
            elif lora_only:
                self._load_model_checkpoint(
                    path, overwrite_reference_adapter, overwrite_critic_adapter
                )
            else:
                model_ref = self._get_unwrapped_actor()
                with gather_if_zero3(self.zero_stage, list(model_ref.parameters())):
                    model_ref.load_state_dict(
                        checkpoint["network_info"]["modules"]["actor_state_dict"]
                    )

            self._restore_checkpoint_attributes(checkpoint)

        else:
            # ``get_checkpoint_dict`` always emits a ``network_info.optimizers``
            # key — empty dict means "no optimizer state was saved". Check
            # truthiness, not key presence.
            if (
                not checkpoint.get("network_info", {}).get("optimizers")
                and load_optimizer
            ):
                warnings.warn(
                    "Optimizer state not found in checkpoint. Training will proceed using a NEW optimizer instance with random/initial default state. ",
                    stacklevel=2,
                )

            super().load_checkpoint(path + "/attributes.pt")
            if lora_only:
                self._load_model_checkpoint(
                    path, overwrite_reference_adapter, overwrite_critic_adapter
                )

        if "lr_scheduler" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def _load_model_checkpoint(
        self,
        path: str,
        overwrite_reference_adapter: bool = False,
        overwrite_critic_adapter: bool = True,
    ) -> None:
        """Restore LoRA adapter weights from a checkpoint directory.

        Reconciles any LoRA config mismatch (e.g. rank mutation) between the checkpoint
        and the live algorithm via :meth:`_merge_lora_configs` / :meth:`_reconfigure_adapters_to_match`
        before loading weights. Reference and Critic LoRA adapters in the checkpoint can be overwritten by the Actor using the ``overwrite_reference_adapter`` and ``overwrite_critic_adapter`` flags.

        :param path: Checkpoint directory path.
        :type path: str
        :param overwrite_reference_adapter: If ``True`` do not overwrite the live reference
            adapter. Defaults to ``False``.
        :type overwrite_reference_adapter: bool
        """
        ckpt_lora_config = self._load_checkpoint_lora_config(path)
        if ckpt_lora_config is not None:
            self.lora_config = self._merge_lora_configs(
                self.lora_config, ckpt_lora_config
            )
            self._reconfigure_adapters_to_match(self.lora_config)

        for adapter in self.selected_adapters:
            if (Path(path) / adapter).exists():
                self._load_adapter_weights(path, adapter, ckpt_lora_config)

        if "reference" in self.selected_adapters and overwrite_reference_adapter:
            self._copy_adapter_weights(
                source_adapter="actor", target_adapter="reference"
            )

        if "critic" in self.selected_adapters and overwrite_critic_adapter:
            # Always overwrite the critic
            self._copy_adapter_weights(
                source_adapter="actor", target_adapter="critic"
            )

    def _restore_checkpoint_attributes(self, checkpoint: dict[str, Any]) -> None:
        """Restore algorithm attributes from payload.

        ``lora_config`` and ``selected_adapters`` are intentionally skipped \u2014 the current
        algorithm's values are authoritative, and any LoRA-shape reconciliation is done
        inside :meth:`_load_model_checkpoint`.

        :param checkpoint: Loaded attribute payload.
        :type checkpoint: dict[str, Any]
        :param checkpoint_type: The checkpoint type.
        :type checkpoint_type: Literal["peft", "deepspeed", "torch"]
        """
        skip_attrs = {"lr_scheduler", "lora_config", "selected_adapters"}
        for attr, value in checkpoint.items():
            if attr in skip_attrs:
                continue
            setattr(self, attr, value)

    def _rebuild_optimizer_after_load(self) -> None:
        """Recreate the optimizer wrapper after distributed checkpoint load.

        Distributed load restores model weights/engine state first, then this
        method rebuilds the wrapper metadata used by training paths.
        """
        self.optimizer = OptimizerWrapper(
            optimizer_cls=self._select_optim_class(),
            networks=[self.actor],
            network_names=["actor"],
            lr=self.lr,
            lr_critic=self.lr_critic,
            use_llm_param_groups=True,
            lr_name="lr" if self.lr_critic is None else ("lr_actor", "lr_critic"),
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: DeviceType = "cpu",
        accelerator: Accelerator | None = None,
    ) -> None:
        msg = (
            "The load class method is not supported for this algorithm class. "
            "To load a saved LLM, please load the model as follows, and then re-instantiate the GRPO/DPO/SFT "
            "class, using the pre-trained model.\n\n"
            "base_model = AutoModelForCausalLM.from_pretrained(\n"
            '    "Qwen/Qwen2.5-3B",\n'
            "    torch_dtype=torch.bfloat16,\n"
            '    device_map="auto"\n'
            ")\n"
            'tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")\n'
            "model = PeftModelProtocol.from_pretrained(base_model, path)\n"
            "where 'path' is the directory containing the saved LoRA adapter weights."
        )
        raise NotImplementedError(
            msg,
        )

    def wrap_models(self) -> None:
        """Wrap the models in the accelerator, DeepSpeed objects must be wrapped at the same time,
        not individually.
        """
        if self.accelerator is not None:
            assert self.optimizer is not None, (
                "Optimizer is set to None, please check that the optimizer is correctly defined."
            )
            # The below is true when an optimizer is defined in the deepspeed config.
            is_dummy_optimizer = isinstance(self.optimizer.optimizer, DummyOptimizer)
            self._restore_adapter_trainability(["actor", "critic"])

            # When prepare is called on the dummy optimizer, it is returned as a DummyOptimizer object.
            # In the cases where self.optimizer.optimizer is an optim.Adam object, it is returned as DeepSpeedOptimizer
            self.actor, optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.actor,
                self.optimizer.optimizer,
                self.lr_scheduler,
            )
            # If optimizer is a dummy optimizer, then the deepspeed engine has been initialized with
            # an optimizer in the config and the optimizer is therefore part of the engine. We point the
            # optimizer attribute of the OptimizerWrapper to the active optimizer.
            self.optimizer.optimizer = (
                optimizer if not is_dummy_optimizer else self.actor.optimizer
            )

            # Again, we retrospectively set the optimizer class to the type of the optimizer as returned by prepare.
            self.optimizer.optimizer_cls = (
                type(optimizer)
                if not is_dummy_optimizer
                else type(self.actor.optimizer)
            )
            if self.gradient_checkpointing:
                self._get_unwrapped_actor().gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
        else:
            assert self.actor is not None, (
                "Actor is set to None, please check that the actor is defined."
            )
            self.actor = self.actor.to(self.device)
            if self.gradient_checkpointing:
                self.actor.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )

    def clean_up(self) -> None:
        """Clean up the algorithm."""
        if self.accelerator is not None:
            # Free up GPU memory occupied by parameters
            if hasattr(self.actor, "empty_partition_cache"):
                self.actor.empty_partition_cache()
            if hasattr(self.actor, "destroy"):
                self.actor.destroy()
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
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
        elif torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()

    def clone(self, index: int | None = None, wrap: bool = True) -> Self:
        """Create a clone of the algorithm.

        :param index: The index of the clone, defaults to None
        :type index: int | None, optional
        :param wrap: If True, wrap the models in the clone with the accelerator, defaults to False
        :type wrap: bool, optional

        :return: A clone of the algorithm
        :rtype: EvolvableAlgorithm
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = self._resolve_clone_work_dir(temp_dir)
            self._save_clone_distributed_actor_state(work_dir)
            clone = self._create_clone_instance()
            clone.mutation_hook()
            clone = self._copy_clone_attributes(clone)
            self._restore_clone_optimizer_and_scheduler(clone)

            # Set the index
            if index is not None:
                clone.index = index

            clone.wrap_models()
            self._load_clone_distributed_actor_state(clone, work_dir)

            return clone

    def _resolve_clone_work_dir(self, temp_dir: str) -> str:
        """Resolve a clone workspace path visible to all ranks.

        :param temp_dir: Local temporary directory path.
        :type temp_dir: str
        :return: Shared working directory path for clone artifacts.
        :rtype: str
        """
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            return broadcast_object_list([temp_dir], from_process=0)[0]
        return temp_dir

    def _save_clone_distributed_actor_state(self, work_dir: str) -> None:
        """Save distributed actor state for ZeRO-2/3 clone workflows.

        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        if self.accelerator is None or self.zero_stage is None or self.zero_stage < 2:
            return

        self.accelerator.wait_for_everyone()
        self._save_distributed_actor(f"{work_dir}/agent_{self.index}")
        self.accelerator.wait_for_everyone()

    def _create_clone_instance(self) -> Self:
        """Instantiate a clone with cloned actor weights and runtime args.

        :return: Newly constructed clone instance.
        :rtype: Self
        """
        input_args = EvolvableAlgorithm.inspect_attributes(
            self,
            input_args_only=True,
        )
        input_args["wrap"] = False
        input_args["clone"] = True
        input_args["actor_network"] = self._clone_actor_network()
        input_args["accelerator"] = (
            Accelerator() if self.accelerator is not None else None
        )
        return type(self)(**input_args)

    def _clone_actor_network(self) -> PreTrainedModelProtocol:
        """Clone actor network while preserving value-head state when enabled.

        :return: Cloned actor network suitable for clone instantiation.
        :rtype: PreTrainedModelProtocol
        """
        actor = self._get_unwrapped_actor()

        if self.use_value_head:
            value_head_model = actor
            inner_peft = value_head_model.pretrained_model
            inner_sd = None
            if self.zero_stage is None or self.zero_stage < 2:
                inner_sd = clone_tensors_for_torch_save(inner_peft.state_dict())
            cloned_inner = clone_llm(inner_peft, self.zero_stage, state_dict=inner_sd)
            cloned_model = type(value_head_model)(cloned_inner)
            cloned_model.v_head.load_state_dict(value_head_model.v_head.state_dict())
            cloned_model.is_peft_model = True
            return cloned_model

        actor_state_dict = None
        if self.zero_stage is None or self.zero_stage < 2:
            actor_state_dict = clone_tensors_for_torch_save(actor.state_dict())
        return clone_llm(actor, self.zero_stage, state_dict=actor_state_dict)

    def _copy_clone_attributes(self, clone: Self) -> Self:
        """Copy non-network attributes while preserving clone runtime members.

        Keeps clone-owned accelerator/scheduler (and vLLM handles when used)
        intact while copying remaining algorithm attributes.

        :param clone: Clone instance to mutate.
        :type clone: Self
        :return: Updated clone instance.
        :rtype: Self
        """
        accelerator = clone.accelerator
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
        clone.lr_scheduler = cloned_lr_scheduler
        self.lr_scheduler = original_lr_scheduler

        if self.use_vllm:
            clone.llm = cloned_llm
            self.llm = original_llm
        return clone

    def _restore_clone_optimizer_and_scheduler(self, clone: Self) -> None:
        """Restore optimizer/scheduler state for non-accelerated clones.

        :param clone: Clone instance receiving optimizer/scheduler states.
        :type clone: Self
        """
        if self.accelerator is not None:
            return

        clone.optimizer.optimizer.load_state_dict(
            state_dict=self.optimizer.optimizer.state_dict(),
        )
        if self.lr_scheduler is not None and clone.lr_scheduler is not None:
            clone.lr_scheduler.load_state_dict(self.lr_scheduler.state_dict())

    def _load_clone_distributed_actor_state(self, clone: Self, work_dir: str) -> None:
        """Load saved distributed actor state into clone for ZeRO-2/3.

        :param clone: Clone instance receiving distributed actor state.
        :type clone: Self
        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        if self.zero_stage is not None and self.zero_stage >= 2:
            clone.accelerator.wait_for_everyone()
            clone._load_distributed_actor(f"{work_dir}/agent_{self.index}")
            clone.accelerator.wait_for_everyone()
        elif self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    @staticmethod
    def update_lr(
        optimizer: torch.optim.Optimizer,  # Deepspeed optimizers are subclasses of torch.optim.Optimizer
        lr: float,
        accelerator: Accelerator | None = None,
        scheduler_config: CosineLRScheduleConfig | None = None,
        lr_critic: float | None = None,
    ) -> tuple[Accelerator | None, SequentialLR | None]:
        """Update the learning rate of the optimizer.

        :param optimizer: Optimizer
        :type optimizer: Optimizer
        :param lr: Learning rate
        :type lr: float
        :param accelerator: Accelerator
        :type accelerator: Accelerator | None
        :param scheduler_config: Scheduler configuration
        :type scheduler_config: CosineLRScheduleConfig | None
        :param lr_critic: When param groups include ``group`` ``actor``/``critic``,
            set critic/value-head groups to this LR; actor groups use ``lr``.
        :type lr_critic: float | None

        :return: Tuple of accelerator and scheduler
        :return: Accelerator
        """
        split = lr_critic is not None and any(
            "group" in pg for pg in optimizer.param_groups
        )
        if split:
            for param_group in optimizer.param_groups:
                g = param_group.get("group")
                if g == "critic":
                    param_group["lr"] = lr_critic
                elif g == "actor":
                    param_group["lr"] = lr
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if accelerator is None:
            scheduler = (
                create_warmup_cosine_scheduler(optimizer, scheduler_config, 1e-8, lr)
                if scheduler_config is not None
                else None
            )
            return accelerator, scheduler

        ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is None:
            scheduler = (
                create_warmup_cosine_scheduler(optimizer, scheduler_config, 1e-8, lr)
                if scheduler_config is not None
                else None
            )
            return accelerator, scheduler

        ds_config = getattr(ds_plugin, "deepspeed_config", None)
        if ds_config is None:
            return accelerator, None

        if ds_config.get("scheduler", None) is not None:
            ds_config["scheduler"]["params"]["warmup_max_lr"] = lr

        if ds_config.get("optimizer", None) is not None:
            ds_config["optimizer"]["params"]["lr"] = lr

        return accelerator, None

    def set_reference_policy(self, reference_update_tracker: int) -> None:
        """Update the reference policy when the reference policy update tracker is greater than the current reference policy update tracker.

        :param reference_update_tracker: The reference policy update tracker
        :type reference_update_tracker: int
        """
        assert reference_update_tracker >= self.reference_update_tracker, (
            "Reference policy update tracker should be greater than or equal to the current reference policy update tracker."
        )
        if reference_update_tracker > self.reference_update_tracker:
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            # Merge adapter into base model
            # Update the reference update tracker
            if self.use_separate_reference_adapter:
                self._copy_adapter_weights(
                    source_adapter="actor", target_adapter="reference"
                )
            else:
                unwrapped = self._get_unwrapped_actor()
                if self.use_value_head:
                    merged = unwrapped.pretrained_model.merge_and_unload()
                    unwrapped.pretrained_model = get_peft_model(
                        merged, self.lora_config, adapter_name="actor"
                    )
                    unwrapped.is_peft_model = True
                else:
                    merged_base_model = unwrapped.merge_and_unload()
                    self.actor = None
                    self.actor = get_peft_model(
                        merged_base_model,
                        self.lora_config,
                        adapter_name="actor",
                    )
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                self.use_adapter("actor")

                # FIXME not convinced about this
                optim_class = self._select_optim_class()
                self.optimizer = OptimizerWrapper(
                    optim_class,
                    networks=[self.actor],
                    lr=self.lr,
                    lr_critic=self.lr_critic,
                    use_llm_param_groups=True,
                    network_names=["actor"],
                    lr_name="lr"
                    if self.lr_critic is None
                    else ("lr_actor", "lr_critic"),
                )
                self.wrap_models()
            self.reference_update_tracker += 1

    def use_adapter(self, adapter_name: str) -> None:
        """Switch the active PEFT adapter, handling all side-effects.

        For "reference": switches adapter and freezes reference params (never trained).
        For all others: switches adapter and restores requires_grad=True on all
        training adapter LoRA params so that DeepSpeed ZeRO-2 gradient bucket hooks
        keep firing correctly.

        :param adapter_name: Name of the adapter to activate ("actor", "critic", "reference").
        :type adapter_name: str
        """
        peft_model = self._peft_model
        peft_model.set_adapter(adapter_name)
        if adapter_name == "reference":
            for name, param in self.actor.named_parameters():
                if param is not None and "reference" in name:
                    param.requires_grad = False
        else:
            self._restore_adapter_trainability(["actor", "critic"])

    @contextmanager
    def select_adapter(self, adapter_name: str) -> None:
        """Temporarily switch adapter; restores the actor adapter on exit.

        :param adapter_name: Name of the adapter to activate ("actor", "critic", "reference").
        :type adapter_name: str
        """
        self.use_adapter(adapter_name)
        try:
            yield
        finally:
            self.use_adapter("actor")

    def _select_optim_class(self) -> type[OptimizerType | DummyOptimizer]:
        """Select the optimizer class based on the accelerator and deepspeed config.

        :return: Optimizer class
        :rtype: type[torch.optim.Optimizer] | type[DummyOptimizer]
        """
        if (
            self.accelerator is not None
            and self.accelerator.state.deepspeed_plugin is not None
            and self.accelerator.state.deepspeed_plugin.deepspeed_config.get(
                "optimizer",
                None,
            )
            is not None
        ):
            return DummyOptimizer
        return AdamW

    def _save_distributed_actor(
        self,
        path: str,
        tag: str = "intermediate_checkpoint",
        lora_only: bool = False,
    ) -> None:
        """Save actor/optimizer/scheduler state via DeepSpeed checkpointing.

        :param path: Output directory to save the checkpoint at
        :type path: str
        """
        if self.accelerator is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            assert self.actor is not None, (
                "Actor is not defined, please check that the actor is defined."
            )
            self._restore_adapter_trainability(self.selected_adapters)
            self.actor.save_checkpoint(
                path, tag=tag, exclude_frozen_parameters=lora_only
            )
            self.use_adapter("actor")
        else:
            warnings.warn(
                "Distributed actor save not supported for non-distributed training.",
                stacklevel=2,
            )

    def _load_distributed_actor(
        self,
        path: str,
        tag: str = "intermediate_checkpoint",
    ) -> None:
        """Override the load_checkpoint method to provide guidance on the correct method to use.

        :param path: Output directory to load the checkpoint from
        :type path: str
        """
        if self.accelerator is not None:
            deepspeed_dirs = sorted(Path(path).glob(tag))
            try:
                assert len(deepspeed_dirs) > 0
                load_path, _ = self.actor.load_checkpoint(
                    str(path),
                    tag=tag,
                    load_module_strict=False,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                )
                if load_path is None:
                    msg = (
                        "Load path is returned as None from deepspeed load_checkpoint."
                    )
                    raise ValueError(
                        msg,
                    )
                self.use_adapter("actor")

            except Exception as e:
                msg = f"Deepspeed failed to resume from checkpoint {path}"
                raise ValueError(
                    msg,
                ) from e
        else:
            warnings.warn(
                "Distributed actor load not supported for non-distributed training.",
                stacklevel=2,
            )

    def _warn_peft_model(
        self,
        peft_model: PeftModelProtocol,
        *,
        context: str,
    ) -> PreTrainedModelProtocol:
        """Merge active adapters into the base weights and drop the PEFT wrapper.

        Emits ``UserWarning`` so callers know adapter tensors are not preserved as
        separate PEFT adapters; forward behavior is kept in the merged dense model.
        """
        warnings.warn(
            f"{context}: A PeftModel was passed; calling merge_and_unload() to merge active adapter weights "
            "into the dense base model before attaching new randomly initialized AgileRL adapters.",
            UserWarning,
            stacklevel=2,
        )
        return peft_model.merge_and_unload()

    def _initialize_actors(
        self,
        base_model: PreTrainedModelProtocol | None,
        add_adapters: bool = True,
    ) -> None:
        """Initialize the actor network.

        If ``base_model`` is a user-supplied :class:`~peft.PeftModel` (with
        ``add_adapters`` True), active adapters are merged into the dense base and
        the PEFT wrapper is removed before attaching AgileRL adapters. The clone path
        (``add_adapters`` False) passes through the model unchanged.

        :param base_model: Base model
        :type base_model: PreTrainedModelProtocol
        :param add_adapters: Flag to indicate if adapters should be added to the model, defaults to True
        :type add_adapters: bool, optional
        """
        if base_model is None:
            base_model = create_model_from_name_or_path(
                self.pretrained_model_name_or_path,
                add_value_head=self.use_value_head,
                use_accelerator=self.accelerator is not None,
            )

        if add_adapters:
            if isinstance(base_model, PeftModelProtocol):
                base_model = self._warn_peft_model(
                    base_model,
                    context="actor_network",
                )

            if self.use_value_head and isinstance(
                getattr(base_model, "pretrained_model", None), PeftModelProtocol
            ):
                inner = base_model.pretrained_model
                base_model.pretrained_model = self._warn_peft_model(
                    inner,
                    context="actor_network.pretrained_model",
                )

            peft_target = (
                base_model.pretrained_model if self.use_value_head else base_model
            )
            # User Peft is merged to dense above; always attach AgileRL adapters here.
            peft_target = get_peft_model(
                peft_target,
                self.lora_config,
                adapter_name="actor",
            )

            # Add every adapter listed in ``selected_adapters`` beyond ``actor`` as a fresh
            # LoRA initialised from ``self.lora_config``. Downstream loads can overwrite
            # these (with padding for rank-mutation) via :meth:`_load_adapter_weights`.
            for name in self.selected_adapters:
                if name == "actor":
                    continue
                if name not in peft_target.peft_config:
                    peft_target.add_adapter(
                        adapter_name=name,
                        peft_config=self.lora_config,  # type: ignore[arg-type]
                    )

            # Drop any adapters we don't own (e.g. from a user-supplied PEFT model).
            for stray in list(peft_target.peft_config.keys()):
                if stray not in self.selected_adapters:
                    warnings.warn(
                        f"Adapter '{stray}' found in the model but is not listed in "
                        f"`selected_adapters={self.selected_adapters!r}`. It will be removed "
                        "and any weights will be lost.",
                        stacklevel=2,
                    )
                    peft_target.delete_adapter(stray)

            if self.use_value_head:
                base_model.pretrained_model = peft_target
                base_model.is_peft_model = True
                self.actor = base_model
            else:
                self.actor = peft_target
        else:
            self.actor = base_model

        self.use_adapter("actor")
        patch_lora_for_fused_forward(self.actor)

        if self.torch_compiler:
            if self._uses_deepspeed:
                warnings.warn(
                    "torch_compiler is not yet compatible with DeepSpeed; "
                    "compilation skipped for this run.",
                    stacklevel=2,
                )
            else:
                if self.gradient_checkpointing:
                    warnings.warn(
                        "torch_compiler is incompatible with gradient_checkpointing; "
                        "disabling gradient checkpointing for this run.",
                        stacklevel=2,
                    )
                    self.gradient_checkpointing = False
                self.actor = compile_model(self.actor, self.torch_compiler)

        if self.accelerator is None:
            self.actor = DummyEvolvable(module=self.actor, device=self.device)

        # If an optimizer is defined in the deepspeed config, then the optimizer is part of the engine when
        # accelerator.prepare() is called. Since we are yet to wrap the model, we pass a dummy optimizer to the OptimizerWrapper.
        # In all other cases optim.Adam is used.
        optim_class = self._select_optim_class()

        self.optimizer = OptimizerWrapper(
            optim_class,
            networks=[self.actor],
            lr=self.lr,
            lr_critic=self.lr_critic,
            use_llm_param_groups=True,
            network_names=["actor"],
            lr_name="lr" if self.lr_critic is None else ("lr_actor", "lr_critic"),
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

    @contextmanager
    def _memory_efficient_params(self) -> None:
        """Memory efficient params context manager.

        :param agent: Distributed agent
        :type agent: DistributedLLMAgent
        :return: None
        :rtype: None
        """
        if self.zero_stage == 3:
            warnings.warn(
                "Memory efficient params is not yet compatible with DeepSpeed ZeRO-3; "
                "memory efficient params will be disabled for this run.",
                stacklevel=2,
            )
            yield
            return
        unwrapped_model = self._get_unwrapped_actor()
        move_params_to_gpu(unwrapped_model, self.device)
        yield
        move_params_to_cpu(unwrapped_model)

    @contextmanager
    def _amp_ctx(self):
        """Yield a ``torch.amp.autocast`` context when running without an accelerator.

        When an ``Accelerator`` is present it already manages mixed-precision
        via its own autocast wrapper, so this is a no-op in that case.
        """
        if self.accelerator is not None:
            yield
        else:
            device_type = torch.device(self.device).type
            if device_type == "cuda" and torch.cuda.is_bf16_supported():
                with torch.amp.autocast(device_type, dtype=torch.bfloat16):
                    yield
            else:
                yield

    def _fused_model_pass(
        self,
        fused_ids: torch.Tensor,
        fused_mask: torch.Tensor,
        routing: list[str],
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model on a fused batch with per-sample adapter routing.

        When *batch_size* is ``None`` the full batch is processed in a single
        ``model.forward`` call — required when gradients are active so that
        gradient-checkpoint recomputation sees the same routing.  When set,
        the batch is iterated in micro-batches (safe under ``no_grad``).

        :return: ``(log_probs, values)`` where *log_probs* has shape
            ``(fused_ids.shape[0], seq_len - 1)`` and *values* matches that
            batch dimension when ``use_value_head`` is set, else ``None``.
        """
        unwrapped = self._get_unwrapped_actor()
        total = fused_ids.shape[0]

        position_ids = None
        if self.calc_position_embeddings:
            position_ids = fused_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(mask=(fused_mask == 0), value=1)

        chunks = (
            [(0, total)]
            if batch_size is None
            else [(s, min(s + batch_size, total)) for s in range(0, total, batch_size)]
        )

        all_logprobs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        for start, end in chunks:
            set_fused_adapter_routing(unwrapped, routing[start:end])
            model_kwargs: dict = {
                "input_ids": fused_ids[start:end],
                "attention_mask": fused_mask[start:end],
                "use_cache": False,
            }
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids[start:end]

            with self._amp_ctx():
                output = self.actor.forward(**model_kwargs)

            if isinstance(output, tuple):
                # Value-head models may return (loss, logits, value, ...); Peft/causal
                # paths may return shorter tuples — only index when present.
                logits = output[0]
                value = output[2] if len(output) > 2 else None
            else:
                logits = output.logits
                value = None

            del output
            logits = logits / self.temperature

            all_logprobs.append(
                LLMAlgorithm._memory_efficient_logits(
                    logits[:, :-1],
                    fused_ids[start:end, 1:],
                )
            )
            if self.use_value_head and value is not None:
                all_values.append(value[:, :-1])

        if self.use_value_head:
            values = torch.cat(all_values, dim=0) if len(chunks) > 1 else all_values[0]
        else:
            values = None
        logprobs = (
            torch.cat(all_logprobs, dim=0) if len(chunks) > 1 else all_logprobs[0]
        )
        return logprobs, values

    def _fused_forward(
        self,
        ids: torch.Tensor,
        batch_size: int,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Actor log-probs, and optionally critic values, in one forward.

        When ``use_value_head`` is set, the input is doubled (actor slice then
        critic slice) and routed so the base model runs once. Otherwise only
        the actor slice is run.

        The doubled batch (value-head path) is always processed in one
        ``model.forward`` call to preserve gradient-checkpoint correctness.

        .. note::

           The routing is **not** cleared here — it must remain active until
           after ``backward()`` completes (for gradient checkpoint
           recomputation).  Callers must call
           ``clear_fused_adapter_routing`` after the backward pass.

           Callers are responsible for ensuring the model is in training
           mode and adapter trainability is restored before entering the
           minibatch loop (see ``learn()`` in ``ppo_llm.py``).

        :param ids: Token IDs ``(B, seq_len)``.
        :param batch_size: Unused (kept for API symmetry).
        :param attention_mask: Optional attention mask matching *ids*.
        :return: ``(actor_log_probs, critic_values)`` with shapes ``(B, seq_len-1)``;
            *critic_values* is ``None`` when no value head is used.
        """
        B = ids.shape[0]
        if attention_mask is None:
            attention_mask = ids != self.pad_token_id
        if self.use_value_head:
            fused_ids = ids.repeat(2, 1)
            fused_mask = attention_mask.repeat(2, 1)
            routing = ["actor"] * B + ["critic"] * B
        else:
            fused_ids = ids
            fused_mask = attention_mask
            routing = ["actor"] * B

        log_probs, values = self._fused_model_pass(
            fused_ids,
            fused_mask,
            routing,
        )
        if self.use_value_head:
            return log_probs[:B], values[B:]
        return log_probs, None

    def _fused_forward_no_grad(
        self,
        ids: torch.Tensor,
        batch_size: int,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute reference log-probs, actor log-probs, and critic values in
        one forward pass (under ``torch.no_grad``).

        When ``use_separate_reference_adapter`` is ``True``, the batch is
        tripled (reference / actor / critic).  When ``False``, reference
        log-probs are computed separately (adapter layers disabled) and the
        actor/critic portion is double-fused.

        Unlike ``_fused_forward`` this method **can** micro-batch because no
        gradient checkpoint recomputation is involved.

        :param ids: Token IDs ``(B, seq_len)``.
        :param batch_size: Micro-batch size for memory-bounded iteration.
        :param attention_mask: Optional attention mask matching *ids*.
        :return: ``(reference_log_probs, actor_log_probs, critic_values)``
            each of shape ``(B, seq_len - 1)``.
        """
        B = ids.shape[0]
        if attention_mask is None:
            attention_mask = ids != self.pad_token_id

        self.actor.eval()

        with torch.inference_mode():
            if self.use_separate_reference_adapter:
                adapters = ["reference", "actor"]
            else:
                adapters = ["actor"]

            if self.use_value_head:
                adapters.append("critic")

            N = len(adapters)
            fused_ids = ids.repeat(N, 1)
            fused_mask = attention_mask.repeat(N, 1)
            routing: list[str] = []
            for adapter in adapters:
                routing.extend([adapter] * B)

            log_probs, values = self._fused_model_pass(
                fused_ids,
                fused_mask,
                routing,
                batch_size=batch_size,
            )
            clear_fused_adapter_routing(self._get_unwrapped_actor())
            critic_values = None
            if self.use_separate_reference_adapter:
                ref_logprobs = log_probs[:B]
                actor_logprobs = log_probs[B : 2 * B]
                if self.use_value_head:
                    critic_values = values[2 * B :]
            else:
                ref_logprobs = self._get_logprobs(
                    ids,
                    batch_size=batch_size,
                    use_reference=True,
                    eval_mode=True,
                    attention_mask=attention_mask,
                )
                actor_logprobs = log_probs[:B]
                if self.use_value_head:
                    critic_values = values[B:]

        return ref_logprobs, actor_logprobs, critic_values

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
        with self.select_adapter("reference" if use_reference else "actor"):
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
                with self._amp_ctx():
                    output = self.actor.forward(**batch_model_kwargs)
                logits = output[0] if isinstance(output, tuple) else output.logits
                logits = logits / self.temperature

                log_prob = LLMAlgorithm._memory_efficient_logits(
                    logits[:, :-1],
                    batch_ids[:, 1:],
                )

                batch_model_kwargs = None
                logits = None
                log_probs.append(log_prob)
        return torch.cat(log_probs, dim=0)

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Perform a backward pass and optimizer step.

        :param loss: Combined loss.
        """
        if self._uses_deepspeed:
            # uses_deepspeed = self.accelerator.state.deepspeed_plugin is not None

            self.accelerator.backward(loss)

            # FIXME fairly sure this does not need to be here as handled by accelerator
            # if not uses_deepspeed:
            #     for group in self.optimizer.optimizer.param_groups:
            #         clip_grad_norm_(group["params"], self.max_grad_norm)
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()

            # FIXMe lr scheduler needs a bit of work
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.lr = self.lr_scheduler.get_last_lr()[0]
        else:
            loss.backward()

            for group in self.optimizer.optimizer.param_groups:
                clip_grad_norm_(group["params"], self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.lr = self.lr_scheduler.get_last_lr()[0]

    @property
    def _peft_model(self) -> Any:
        """The PeftModel managing LoRA adapters.

        When ``use_value_head=True`` the PeftModel lives inside the
        value-head wrapper at ``self.actor.pretrained_model``.
        Otherwise ``self.actor`` itself is the PeftModel.
        """
        if self.use_value_head:
            return self.actor.pretrained_model
        return self.actor

    def _restore_adapter_trainability(self, selected_adapters: list[str]) -> None:
        """Restore requires_grad=True for all trainable parameters of specified adapters.

        PEFT's set_adapter() sets requires_grad=False on all non-active adapter
        weights. Under DeepSpeed ZeRO Stage 2, gradient bucket hooks are registered
        once at accelerator.prepare() time based on the requires_grad snapshot at
        that moment. If set_adapter() later toggles requires_grad=False on params
        that ZeRO-2 registered hooks for, those hooks never fire, the bucket never
        completes, and reduce-scatter never runs - the optimizer sees zero gradients.

        :param selected_adapters: LoRA adapter names whose params should be trainable.
        :type selected_adapters: list[str]
        """
        key = tuple(sorted(selected_adapters))
        cache = getattr(self, "_trainable_params_cache", None)
        if cache is not None and cache[0] == key:
            for param in cache[1]:
                param.requires_grad_(True)
            return

        model = self.actor.module if hasattr(self.actor, "module") else self.actor
        params: list[torch.nn.Parameter] = []
        for name, param in model.named_parameters():
            for adapter in selected_adapters:
                if adapter in name and "lora" in name:
                    params.append(param)
                    break
        for param in params:
            param.requires_grad_(True)
        self._trainable_params_cache = (key, params)

    def _move_model_to_vllm(self) -> None:
        """Move the deepspeed model to vllm."""
        if self._vllm_moved:
            return
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        model_ref = self._get_unwrapped_actor()
        peft_ref = model_ref.pretrained_model if self.use_value_head else model_ref
        self.use_adapter("actor")
        modules_to_skip = (
            "lora_",
            "original_module",
            "modules_to_save",
            "ia3_",
            "ranknum",
            "summary",
        )
        with gather_if_zero3(self.zero_stage, list(model_ref.parameters())):
            peft_ref.merge_adapter()
            weights_to_load = []
            try:
                for name, param in model_ref.named_parameters():
                    weight_name = name.removeprefix("module.")
                    weight_name = weight_name.removeprefix("pretrained_model.")
                    weight_name = weight_name.removeprefix("base_model.model.")
                    weight_name = weight_name.replace(".base_layer", "")
                    if peft_ref.prefix in weight_name:
                        continue
                    if any(tok in weight_name for tok in modules_to_skip):
                        continue
                    weights_to_load.append((weight_name, param.data))

                def _load_weights(model):
                    model.load_weights(weights_to_load)

                self.llm.apply_model(_load_weights)
            finally:
                model_ref.unmerge_adapter()
                model_ref.set_adapter("actor")
        self.llm.reset_prefix_cache()
        self._vllm_moved = True

    def _generate_with_vllm_colocate(
        self, prompts: list[dict[str, Any]], group_size: int, temperature: float | None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Generate completions with colocated vLLM for GRPO/LLMPPO-style batches.

        Each entry in ``prompts`` is repeated ``group_size`` times so vLLM receives
        a flat list of length ``len(prompts) * group_size`` (e.g. GRPO groups).

        **Prompt dict fields:** ``input_ids`` and usually ``text`` for decoding.
        For sliding-window multi-turn prompts, optionally set ``trajectory_input_ids``,
        ``trajectory_text`` (decoded string passed to vLLM), ``stitch_prefix_ids``, and
        ``initial_prompt_len`` (required when ``stitch_prefix_ids`` is
        non-empty). Action masks use the full logical prompt length from
        ``input_ids``, not only ``trajectory_input_ids``.

        :param prompts: Length-``N`` list of observation dicts for this rank.
        :type prompts: list[dict[str, Any]]
        :param group_size: Repeat factor per prompt (1 for plain PPO).
        :type group_size: int
        :param temperature: Temperature for sampling.
        :type temperature: float | None
        :return: Per-prompt completion token tensors and matching action masks.
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor]]
        """
        if SamplingParams is None:
            msg = "vLLM is required when use_vllm=True. Install AgileRL with vLLM support for this platform: `pip install agilerl[llm]`."
            raise ImportError(msg)

        def _trajectory_input_ids(prompt: dict[str, Any]) -> torch.Tensor:
            return cast(
                "torch.Tensor",
                prompt.get("trajectory_input_ids", prompt["input_ids"]),
            )

        def _token_prompt_for_vllm(prompt: dict[str, Any]) -> dict[str, list[int]]:
            ids = _trajectory_input_ids(prompt)
            return {"prompt_token_ids": ids.squeeze(0).tolist()}

        def _stitch_prefix(prompt: dict[str, Any], ref: torch.Tensor) -> torch.Tensor:
            st = prompt.get("stitch_prefix_ids")
            if st is None:
                return ref.new_zeros((ref.shape[0], 0))
            return cast("torch.Tensor", st)

        def _vllm_max_new_tokens(model_prompt_len: int) -> int:
            room = self.max_model_len - model_prompt_len
            if room <= 0:
                error_msg = f"Model prompt length ({model_prompt_len}) is greater than the model length ({self.max_model_len})"
                raise ValueError(error_msg)
            max_out = min(max_token_cap, room)
            if self.min_output_tokens is not None:
                max_out = max(max_out, min(self.min_output_tokens, room))
            return min(max_out, room)

        group_prompts = [prompt for prompt in prompts for _ in range(group_size)]
        prompts_ids = [_trajectory_input_ids(p) for p in group_prompts]
        stitch_prefixes = [
            _stitch_prefix(p, prompts_ids[i]) for i, p in enumerate(group_prompts)
        ]
        token_prompts = [_token_prompt_for_vllm(p) for p in group_prompts]
        max_token_cap = (
            self.max_output_tokens
            if self.max_output_tokens is not None
            else self.max_model_len
        )
        max_output_tokens = [
            _vllm_max_new_tokens(int(prompt_id.shape[1])) for prompt_id in prompts_ids
        ]

        if self.vllm_config.tensor_parallel_size > 1:
            orig_size = len(token_prompts)

            gathered_prompts_ids = [
                None for _ in range(self.vllm_config.tensor_parallel_size)
            ]
            gathered_token_prompts = [
                None for _ in range(self.vllm_config.tensor_parallel_size)
            ]
            gathered_stitch_prefixes = [
                None for _ in range(self.vllm_config.tensor_parallel_size)
            ]
            gathered_max_output_tokens = [
                None for _ in range(self.vllm_config.tensor_parallel_size)
            ]

            torch.distributed.all_gather_object(
                gathered_prompts_ids,
                prompts_ids,
                group=self.tp_group,
            )
            torch.distributed.all_gather_object(
                gathered_token_prompts,
                token_prompts,
                group=self.tp_group,
            )
            torch.distributed.all_gather_object(
                gathered_stitch_prefixes,
                stitch_prefixes,
                group=self.tp_group,
            )
            torch.distributed.all_gather_object(
                gathered_max_output_tokens,
                max_output_tokens,
                group=self.tp_group,
            )

            all_prompts_ids = [
                prompt_id for sublist in gathered_prompts_ids for prompt_id in sublist
            ]
            all_token_prompts = [
                prompt for sublist in gathered_token_prompts for prompt in sublist
            ]
            all_stitch_prefixes = [
                sp for sublist in gathered_stitch_prefixes for sp in sublist
            ]
            all_max_output_tokens = [
                max_out for sublist in gathered_max_output_tokens for max_out in sublist
            ]
        else:
            all_token_prompts = token_prompts
            all_prompts_ids = prompts_ids
            all_stitch_prefixes = stitch_prefixes
            all_max_output_tokens = max_output_tokens

        generation_kwargs = {
            "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
            "repetition_penalty": self.repetition_penalty,
            "temperature": temperature,
            "top_p": self.top_p,
            "top_k": -1 if (self.top_k is None or self.top_k == 0) else self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "min_tokens": (
                0 if self.min_output_tokens is None else self.min_output_tokens
            ),
            "presence_penalty": self.vllm_config.presence_penalty,
            "frequency_penalty": self.vllm_config.frequency_penalty,
        }
        if self.vllm_config.stop_sequences:
            generation_kwargs["stop"] = self.vllm_config.stop_sequences
        sampling_params = [
            SamplingParams(**generation_kwargs, max_tokens=max_output_token)
            for max_output_token in all_max_output_tokens
        ]

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        all_outputs = self.llm.generate(
            all_token_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        completion_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        if self.vllm_config.tensor_parallel_size > 1:
            # Slice completions for this rank within its TP group.
            # Each rank generates all outputs — we keep only our share.
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            tp_slice = slice(
                local_rank_in_group * orig_size,
                (local_rank_in_group + 1) * orig_size,
            )
            completion_ids = completion_ids[tp_slice]
            prompts_ids = all_prompts_ids[tp_slice]
            stitch_prefixes = all_stitch_prefixes[tp_slice]

        prompts_ids = [p.to(self.device, non_blocking=True) for p in prompts_ids]
        stitch_prefixes = [
            sp.to(self.device, non_blocking=True) for sp in stitch_prefixes
        ]

        completion_ids = [
            torch.cat(
                [
                    torch.cat(
                        prompts_ids[group_size * i : group_size * (i + 1)],
                        dim=0,
                    ).to(self.device),
                    stack_and_pad_experiences(
                        completion_ids[group_size * i : group_size * (i + 1)],
                        padding_values=[self.pad_token_id],
                        device=self.device,
                    )[0],
                ],
                dim=1,
            )
            for i in range(len(prompts))
        ]

        if any(int(sp.shape[1]) > 0 for sp in stitch_prefixes):
            completion_ids = stitch_completion_after_windowed_vllm_generate(
                completion_ids,
                stitch_prefixes,
                group_prompts,
                group_size,
                prompts,
            )

        num_input_tokens = [
            int(cast("torch.Tensor", prompts[i]["input_ids"]).shape[1])
            for i in range(len(prompts))
        ]
        completion_masks = []

        for i, completion_id in enumerate(completion_ids):
            completion_mask = torch.zeros_like(
                completion_id,
                dtype=torch.bool,
                device=self.device,
            )
            completion_mask[:, num_input_tokens[i] :] = True
            completion_mask[completion_id == self.pad_token_id] = False
            completion_mask = completion_mask[:, 1:]
            completion_masks.append(completion_mask)

        return completion_ids, completion_masks

    @staticmethod
    def _memory_efficient_logits(
        logits: torch.Tensor,
        index: torch.Tensor,
        _chunk_rows: int = 1,
    ) -> torch.Tensor:
        """Calculate log probabilities for previously generated token ids.

        Processes a few rows at a time so peak memory stays bounded to
        ``(_chunk_rows, seq_len, vocab_size)`` rather than the full batch,
        avoiding OOM on large-vocabulary models while reducing Python loop
        overhead compared to a strict row-by-row approach.

        :param logits: Logits of shape ``(B, seq_len, vocab_size)``.
        :type logits: torch.Tensor
        :param index: Token IDs of shape ``(B, seq_len)``.
        :type index: torch.Tensor
        :return: Log probabilities of the completion IDs, shape ``(B, seq_len)``.
        :rtype: torch.Tensor
        """
        # 1. Gather the raw logits for the specific token IDs.
        # Shape reduces from (B, seq_len, vocab_size) immediately to (B, seq_len)

        B = logits.shape[0]
        if B <= _chunk_rows:
            return (
                F.log_softmax(logits, dim=-1)
                .gather(dim=-1, index=index.unsqueeze(-1))
                .squeeze(-1)
            )

        per_token_logps = []
        for start in range(0, B, _chunk_rows):
            end = min(start + _chunk_rows, B)
            target_logits_chunk = (
                logits[start:end]
                .gather(dim=-1, index=index[start:end].unsqueeze(-1))
                .squeeze(-1)
            )
            log_z_chunk = torch.logsumexp(logits[start:end], dim=-1)
            per_token_logps_chunk = (target_logits_chunk - log_z_chunk).to(
                logits.dtype
            )  # Do we need to upcast to float 32 here??
            per_token_logps.append(per_token_logps_chunk)
        return torch.cat(per_token_logps, dim=0)

    def _configure_batch_size_per_process(
        self,
        batch_size: int,
        micro_batch_size_per_gpu: int | None,
    ) -> None:
        if self.accelerator is None:
            self.batch_size_per_process = batch_size
            if micro_batch_size_per_gpu is not None:
                self.micro_batch_size_per_gpu = int(micro_batch_size_per_gpu)
            else:
                self.micro_batch_size_per_gpu = batch_size
            return

        ds_plugin = self.accelerator.state.deepspeed_plugin
        if ds_plugin is None:
            err_msg = """DeepSpeed plugin is not initialized. If using an accelerator,
            ensure to launch your training script with `accelerate launch --num_processes <your_script.py>`."""
            raise ValueError(err_msg)
        ds_config = ds_plugin.deepspeed_config

        if batch_size % self.accelerator.num_processes != 0:
            msg = f"Batch size ({batch_size}) must be divisible by the number of processes ({self.accelerator.num_processes})."
            raise ValueError(
                msg,
            )

        self.batch_size_per_process = int(batch_size / self.accelerator.num_processes)

        if micro_batch_size_per_gpu is None:
            if (
                self.batch_size_per_process
                % ds_config.get("gradient_accumulation_steps", 1)
                != 0
            ):
                msg = (
                    f"Batch size ({batch_size}) must be divisible by the product of the number of processes ({self.accelerator.num_processes}) and gradient accumulation steps ({ds_config.get('gradient_accumulation_steps', 1)})."
                    "Gradient accumulation steps can be updated in the deepspeed config by changing the 'gradient_accumulation_steps' parameter."
                )
                raise ValueError(
                    msg,
                )

            gradient_accumulation_steps = ds_config.get(
                "gradient_accumulation_steps", 1
            )
            self.micro_batch_size_per_gpu = (
                self.batch_size_per_process // gradient_accumulation_steps
            )

            prev_micro = ds_config.get("train_micro_batch_size_per_gpu")
            if prev_micro is not None:
                warnings.warn(
                    "Overwriting DeepSpeed config train_micro_batch_size_per_gpu "
                    f"from {prev_micro!r} to {self.micro_batch_size_per_gpu} "
                    f"(batch_size_per_process={self.batch_size_per_process} "
                    f"// gradient_accumulation_steps={gradient_accumulation_steps}).",
                    stacklevel=2,
                )
            ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size_per_gpu
            return

        if micro_batch_size_per_gpu == 0:
            msg = (
                "micro_batch_size_per_gpu is equal to zero, which is not allowed. "
                "Please set micro_batch_size_per_gpu to a positive integer."
            )
            raise ValueError(msg)

        self.micro_batch_size_per_gpu = int(micro_batch_size_per_gpu)
        if (
            batch_size
            % (self.micro_batch_size_per_gpu * self.accelerator.num_processes)
            != 0
        ):
            msg = f"When specifying micro_batch_size_per_gpu, batch_size ({batch_size}) must be divisible by the product of the number of processes ({self.accelerator.num_processes}) and micro_batch_size_per_gpu ({self.micro_batch_size_per_gpu})."
            raise ValueError(
                msg,
            )
        prev_micro = ds_config.get("train_micro_batch_size_per_gpu")
        if prev_micro is not None:
            warnings.warn(
                "Overwriting DeepSpeed config train_micro_batch_size_per_gpu "
                f"from {prev_micro!r} to {self.micro_batch_size_per_gpu} ",
                stacklevel=2,
            )
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size_per_gpu
        gradient_accumulation_steps = (
            batch_size / self.accelerator.num_processes / self.micro_batch_size_per_gpu
        )
        warnings.warn(
            f"Overwriting deepspeed config gradient accumulation steps from {ds_config.get('gradient_accumulation_steps', 'auto')} to {gradient_accumulation_steps}",
            stacklevel=2,
        )
        ds_config["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
        return

    def recompile(self) -> None:
        """Recompile evolvable modules with ``torch.compile``.

        Iterates over ``evolvable_attributes`` and compiles each one.
        Skipped when DeepSpeed is active because ``DeepSpeedEngine`` is not
        compatible with ``OptimizedModule`` wrapping.
        """
        if self.torch_compiler is None or self._uses_deepspeed:
            return
        for name, obj in self.evolvable_attributes(networks_only=True).items():
            setattr(self, name, compile_model(obj, self.torch_compiler))

    def _update_existing_adapter(
        self,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        """Overwrite weights of an existing adapter in-place without creating new parameters.

        :param checkpoint_dir: Checkpoint directory
        :type checkpoint_dir: str
        :param adapter_name: Adapter name
        :type adapter_name: str.

        :return: None
        :rtype: None
        """
        unwrapped = self._get_unwrapped_actor()
        peft_model = unwrapped.pretrained_model if self.use_value_head else unwrapped

        adapter_path = f"{checkpoint_dir}/{adapter_name}/adapter_model.safetensors"
        adapter_state = load_file(adapter_path, device=self.device)

        with gather_if_zero3(
            self.zero_stage,
            list(unwrapped.parameters()),
            modifier_rank=0,
        ):
            with torch.no_grad():
                set_peft_model_state_dict(
                    peft_model,
                    adapter_state,
                    adapter_name=adapter_name,
                )
            peft_model.set_adapter(adapter_name)

            for name, param in unwrapped.named_parameters():
                if "reference" in name:
                    param.requires_grad = False
                elif "actor" in name or "critic" in name:
                    param.requires_grad = True
        self.accelerator.wait_for_everyone()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _copy_adapter_weights(self, source_adapter: str, target_adapter: str) -> None:
        """Copy LoRA weights from source adapter to target adapter."""
        source_params = {}
        target_params = {}
        for name, param in self.actor.named_parameters():
            if "lora" not in name:
                continue
            if f".{source_adapter}." in name:
                key = name.replace(f".{source_adapter}.", ".", 1)
                source_params[key] = param
            elif f".{target_adapter}." in name:
                key = name.replace(f".{target_adapter}.", ".", 1)
                target_params[key] = param

        if not source_params:
            msg = f"No LoRA tensors found for source adapter '{source_adapter}'."
            raise ValueError(
                msg,
            )
        if not target_params:
            msg = f"No LoRA tensors found for target adapter '{target_adapter}'."
            raise ValueError(
                msg,
            )

        missing = [key for key in source_params if key not in target_params]
        if missing:
            msg = (
                f"Target adapter '{target_adapter}' is missing {len(missing)} LoRA tensors "
                f"present in source adapter '{source_adapter}'."
            )
            raise ValueError(
                msg,
            )

        for key, src_param in source_params.items():
            target_params[key].data.copy_(src_param.data)

    @staticmethod
    def _load_checkpoint_lora_config(path: str) -> LoraConfig | None:
        """Load the ``actor`` adapter's LoRA config from a checkpoint directory, if present.

        :param path: Directory previously written by :meth:`save_checkpoint`.
        :type path: str
        :return: The ``LoraConfig`` stored alongside the actor adapter, or ``None`` if
            the checkpoint does not contain one (legacy checkpoint, or no ``actor/`` subdir).
        :rtype: peft.LoraConfig | None
        """
        config_path = Path(path) / "actor" / "adapter_config.json"
        if not config_path.is_file():
            return None
        return LoraConfig.from_pretrained(str(config_path.parent))

    @staticmethod
    def _merge_lora_configs(
        current: LoraConfig | None,
        checkpoint: LoraConfig,
    ) -> LoraConfig:
        """Reconcile a checkpoint's LoRA config with the current one, favouring the current
        where a choice must be made and warning on every mismatch.

        Rules:

        * ``r``: take ``max(current, checkpoint)`` (rank can grow via mutation).
        * ``target_modules``, ``modules_to_save``: take the union when both are iterable,
          otherwise keep current.
        * Everything else: keep current, warn on mismatch.

        :param current: The LoRA config the live algorithm was instantiated with. When
            ``None`` the checkpoint's config is returned as-is.
        :type current: peft.LoraConfig | None
        :param checkpoint: The LoRA config stored alongside the checkpoint's actor adapter.
        :type checkpoint: peft.LoraConfig
        :return: A new ``LoraConfig`` representing the reconciled settings.
        :rtype: peft.LoraConfig
        """
        if current is None:
            return checkpoint

        merged_kwargs = (
            current.to_dict() if hasattr(current, "to_dict") else dict(vars(current))
        )
        ckpt_kwargs = (
            checkpoint.to_dict()
            if hasattr(checkpoint, "to_dict")
            else dict(vars(checkpoint))
        )

        def _as_set(x: Any) -> set[str] | None:
            if x is None:
                return None
            if isinstance(x, str):
                return {x}
            try:
                return set(x)
            except TypeError:
                return None

        for key, ckpt_val in ckpt_kwargs.items():
            cur_val = merged_kwargs.get(key)
            if key == "r":
                cur_r = cur_val if isinstance(cur_val, int) else 0
                ckpt_r = ckpt_val if isinstance(ckpt_val, int) else 0
                new_r = max(cur_r, ckpt_r)
                if cur_r != ckpt_r:
                    warnings.warn(
                        f"LoRA rank mismatch (current={cur_r}, checkpoint={ckpt_r}); "
                        f"using max={new_r} and padding checkpoint weights into the extra rank slots.",
                        stacklevel=2,
                    )
                merged_kwargs[key] = new_r
                continue
            if key in ("target_modules", "modules_to_save"):
                cur_set = _as_set(cur_val)
                ckpt_set = _as_set(ckpt_val)
                if cur_set is None or ckpt_set is None:
                    if cur_val != ckpt_val:
                        warnings.warn(
                            f"LoRA '{key}' differs (current={cur_val!r}, checkpoint={ckpt_val!r}); "
                            "keeping the current value.",
                            stacklevel=2,
                        )
                    continue
                union = cur_set | ckpt_set
                if cur_set != ckpt_set:
                    warnings.warn(
                        f"LoRA '{key}' differs (current={sorted(cur_set)}, checkpoint={sorted(ckpt_set)}); "
                        f"using union={sorted(union)}.",
                        stacklevel=2,
                    )
                merged_kwargs[key] = sorted(union)
                continue
            if cur_val != ckpt_val:
                warnings.warn(
                    f"LoRA '{key}' differs (current={cur_val!r}, checkpoint={ckpt_val!r}); "
                    "keeping current value.",
                    stacklevel=2,
                )

        return LoraConfig(**merged_kwargs)

    @staticmethod
    def _lora_configs_equivalent(a: LoraConfig, b: LoraConfig) -> bool:
        """Structural equality for two ``LoraConfig`` instances.

        List/tuple/set-typed fields (``target_modules`` etc.) are normalised to sorted
        lists before comparison so insertion order does not matter.

        :param a: First config.
        :type a: peft.LoraConfig
        :param b: Second config.
        :type b: peft.LoraConfig
        :return: ``True`` iff every keyword field is equal after normalisation.
        :rtype: bool
        """
        a_dict = a.to_dict() if hasattr(a, "to_dict") else dict(vars(a))
        b_dict = b.to_dict() if hasattr(b, "to_dict") else dict(vars(b))
        for key in ("target_modules", "modules_to_save", "exclude_modules"):
            for d in (a_dict, b_dict):
                val = d.get(key)
                if isinstance(val, (list, tuple, set)):
                    d[key] = sorted(val)
        return a_dict == b_dict

    def _reconfigure_adapters_to_match(self, target_config: LoraConfig) -> None:
        """Ensure every adapter in :attr:`adapter_names` uses ``target_config``.

        If an adapter's live config already matches, it is left untouched. Otherwise it
        is rebuilt against ``target_config`` with freshly-initialised weights; callers
        are expected to subsequently load weights into it (with rank padding where
        needed).

        :param target_config: The merged LoRA config that all adapters should match.
        :type target_config: peft.LoraConfig
        :return: None. Mutates the live PEFT model in place.
        :rtype: None
        """
        peft_model = self._peft_model
        if not isinstance(peft_model, PeftModelProtocol):
            return

        current_adapter = (
            peft_model.active_adapter
            if hasattr(peft_model, "active_adapter")
            else "actor"
        )
        for name in self.selected_adapters:
            live_cfg = peft_model.peft_config.get(name)
            if live_cfg is not None and self._lora_configs_equivalent(
                live_cfg, target_config
            ):
                continue
            with gather_if_zero3(
                self.zero_stage, list(peft_model.parameters()), modifier_rank=0
            ):
                if name in peft_model.peft_config:
                    peft_model.delete_adapter(name)
                peft_model.add_adapter(adapter_name=name, peft_config=target_config)
        if current_adapter in peft_model.peft_config:
            peft_model.set_adapter(current_adapter)
        else:
            peft_model.set_adapter("actor")

    def _load_adapter_weights(
        self,
        checkpoint_dir: str,
        adapter_name: str,
        ckpt_lora_config: LoraConfig | None,
    ) -> None:
        """Overwrite a live adapter's weights from disk, padding smaller LoRA ranks into
        the current adapter shape where needed.

        :param checkpoint_dir: Directory written by :meth:`save_checkpoint`; must contain
            ``<adapter_name>/adapter_model.safetensors``.
        :type checkpoint_dir: str
        :param adapter_name: Name of the adapter to overwrite (must already exist on the
            live PEFT model).
        :type adapter_name: str
        :param ckpt_lora_config: The checkpoint's LoRA config, used to detect a rank
            mismatch that requires padding. Pass ``None`` to skip padding entirely.
        :type ckpt_lora_config: peft.LoraConfig | None
        :return: None. Mutates the live adapter's parameters in place.
        :rtype: None
        """
        unwrapped = self._get_unwrapped_actor()
        peft_model = unwrapped.pretrained_model if self.use_value_head else unwrapped

        adapter_path = f"{checkpoint_dir}/{adapter_name}/adapter_model.safetensors"
        adapter_state = load_file(adapter_path, device=str(self.device))

        with gather_if_zero3(
            self.zero_stage, list(unwrapped.parameters()), modifier_rank=0
        ):
            if (
                ckpt_lora_config is not None
                and self.lora_config is not None
                and getattr(ckpt_lora_config, "r", None)
                != getattr(self.lora_config, "r", None)
            ):
                adapter_state = self._pad_adapter_state_to_live_shape(
                    adapter_state, adapter_name, peft_model
                )

            with torch.no_grad():
                set_peft_model_state_dict(
                    peft_model, adapter_state, adapter_name=adapter_name
                )
            peft_model.set_adapter(adapter_name)

            for name, param in unwrapped.named_parameters():
                if "reference" in name:
                    param.requires_grad = False

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    @staticmethod
    def _pad_adapter_state_to_live_shape(
        adapter_state: dict[str, torch.Tensor],
        adapter_name: str,
        peft_model: Any,
    ) -> dict[str, torch.Tensor]:
        """Pad each checkpoint tensor into the live adapter's shape, copying into the
        top-left slice and leaving the rest at the fresh-init values PEFT populated when
        the adapter was (re-)created.

        :param adapter_state: Raw state dict loaded from an
            ``adapter_model.safetensors`` file.
        :type adapter_state: dict[str, torch.Tensor]
        :param adapter_name: Name of the live adapter whose shape should be matched.
        :type adapter_name: str
        :param peft_model: The underlying ``PeftModel``.
        :type peft_model: peft.PeftModel
        :return: A new state dict with every tensor reshaped to match the live adapter.
        :rtype: dict[str, torch.Tensor]
        """
        live_state = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
        padded: dict[str, torch.Tensor] = {}
        for key, ckpt_t in adapter_state.items():
            live_t = live_state.get(key)
            if live_t is None or tuple(live_t.shape) == tuple(ckpt_t.shape):
                padded[key] = ckpt_t
                continue
            if any(ck > lv for ck, lv in zip(ckpt_t.shape, live_t.shape, strict=False)):
                # Checkpoint rank > live rank shouldn't happen with max() merge, but
                # fall back to a straight load so PEFT raises a clear error.
                padded[key] = ckpt_t
                continue
            canvas = live_t.detach().clone()
            slices = tuple(slice(0, d) for d in ckpt_t.shape)
            canvas[slices] = ckpt_t.to(canvas.dtype).to(canvas.device)
            padded[key] = canvas
        return padded

    @staticmethod
    def _create_prompt_masks(
        prompt_lengths: list[int], max_length: int
    ) -> torch.Tensor:
        """Create a mask for the prompts based on the prompt lengths (vectorized).

        :param prompt_lengths: List of prompt lengths
        :type prompt_lengths: list[int]
        :param max_length: Maximum length of the prompts
        :type max_length: int
        :return: Mask tensor [batch_size, max_length]
        :rtype: torch.Tensor
        """
        prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.long)
        positions = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        return positions > prompt_lengths_tensor.unsqueeze(1)

    def _configure_vllm(self) -> None:
        """Configure vLLM for efficient inference during generation in 'get_action'."""
        if LLM is None:
            msg = "vLLM is required when use_vllm=True. Install AgileRL with vLLM support for this platform: `pip install agilerl[llm]`."
            raise ImportError(msg)
        if self.vllm_config is None:
            warnings.warn(
                "No VLLM config provided. Using default VLLM configuration for generation.",
                stacklevel=2,
            )
            self.vllm_config = VLLMConfig()
        num_processes = (
            self.accelerator.num_processes if self.accelerator is not None else 1
        )
        process_index = (
            self.accelerator.process_index if self.accelerator is not None else 0
        )
        local_process_index = (
            self.accelerator.local_process_index if self.accelerator is not None else 0
        )
        if num_processes % self.vllm_config.tensor_parallel_size != 0:
            msg = f"Tensor parallel size {self.vllm_config.tensor_parallel_size} must be a multiple of the number of processes {num_processes}."
            raise ValueError(
                msg,
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
                        ),
                    )
                    for i in range(
                        num_processes // self.vllm_config.tensor_parallel_size,
                    )
                ],
            )

        # vLLM requires the environment variables to be set for distributed training.
        os.environ["RANK"] = str(process_index)
        os.environ["LOCAL_RANK"] = str(local_process_index)
        os.environ["WORLD_SIZE"] = str(num_processes)
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

        llm_kwargs = {
            "model": self.pretrained_model_name_or_path,
            "tensor_parallel_size": self.vllm_config.tensor_parallel_size,
            "gpu_memory_utilization": self.vllm_config.gpu_memory_utilization,
            "max_num_seqs": self.vllm_config.max_num_seqs,
            "max_model_len": self.max_model_len,
            "distributed_executor_backend": "external_launcher",
            "seed": process_index // self.vllm_config.tensor_parallel_size,
            "max_num_batched_tokens": self.vllm_config.max_num_seqs
            * self.max_model_len,
            "model_impl": "vllm",
            "enable_sleep_mode": self.vllm_config.sleep_mode,
        }
        if self.vllm_config.dtype is not None:
            llm_kwargs["dtype"] = self.vllm_config.dtype
        if self.vllm_config.quantization is not None:
            llm_kwargs["quantization"] = self.vllm_config.quantization
        try:
            self.llm = LLM(**llm_kwargs)
        except ValueError as err:
            backend_env = os.environ.get("VLLM_ATTENTION_BACKEND")
            if backend_env is not None and "backend" in str(err).lower():
                msg = (
                    "vLLM initialization failed due to unsupported "
                    f"VLLM_ATTENTION_BACKEND={backend_env!r}. "
                    "Please unset VLLM_ATTENTION_BACKEND or set it to a backend "
                    "supported by your installed vLLM build."
                )
                raise ValueError(msg) from err
            raise

        if self.vllm_config.sleep_mode:
            self.llm.sleep(level=2)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _sync_deepspeed_gradient_clipping(self) -> None:
        """Synchronize max_grad_norm with DeepSpeed gradient_clipping config.
        Registered as a mutation hook to ensure consistency after mutations.
        """
        if self.accelerator is None or self.accelerator.state.deepspeed_plugin is None:
            return

        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is None:
            return

        ds_config = ds_plugin.deepspeed_config
        if "gradient_clipping" not in ds_config:
            return

        if ds_config["gradient_clipping"] != self.max_grad_norm:
            ds_config["gradient_clipping"] = self.max_grad_norm

        if hasattr(self.actor, "optimizer"):
            if hasattr(self.actor.optimizer, "grad_clip"):
                self.actor.optimizer.grad_clip = self.max_grad_norm
            if hasattr(self.actor.optimizer, "clip_grad"):
                self.actor.optimizer.clip_grad = self.max_grad_norm

    def _get_lm_head(self):
        """Locate the lm_head module, handling both raw and PEFT-wrapped models.

        :return: The lm_head (or embed_out) linear layer.
        :rtype: torch.nn.Module
        :raises AttributeError: If no lm_head can be found.
        """
        model = self.actor
        if hasattr(model, "base_model"):  # PeftModel → LoraModel
            model = model.base_model
        if hasattr(model, "model"):  # LoraModel → CausalLM
            model = model.model
        for attr in ("lm_head", "embed_out"):
            if hasattr(model, attr):
                return getattr(model, attr)
        err_msg = f"""Cannot find lm_head in {type(self.actor).__name__}.
        Set use_liger_loss=False.
        """
        raise AttributeError(err_msg)

    def _get_unwrapped_actor(self) -> Any:
        """Return actor unwrapped from Accelerate and DummyEvolvable layers."""
        actor = (
            self.accelerator.unwrap_model(self.actor)
            if self.accelerator is not None
            else self.actor
        )
        while isinstance(actor, DummyEvolvable):
            actor = actor.module
        return actor

    def _prepare_vllm_for_training(self) -> None:
        """Prepare vLLM for learning."""
        if self._vllm_awake and (
            self.accelerator is None or self.accelerator.is_main_process
        ):
            torch.cuda.empty_cache()
            self.llm.sleep(level=2)
            self._vllm_awake = False

        if self.use_vllm:
            self._vllm_moved = False

    def _prepare_vllm_for_generation(self) -> None:
        if not self._vllm_awake and (
            self.accelerator is None or self.accelerator.is_main_process
        ):
            torch.cuda.empty_cache()
            self.llm.wake_up()
            self._vllm_awake = True
        if self.use_memory_efficient_params:
            unwrapped_model = self._get_unwrapped_actor()
            move_params_to_cpu(unwrapped_model)
        self._move_model_to_vllm()
