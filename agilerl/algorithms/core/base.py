# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import gc
import inspect
import logging
import os
import pickle
import shutil
import tempfile
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NoReturn,
    Protocol,
    TypeVar,
    overload,
)

import dill
import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from gymnasium import spaces
from tensordict import TensorDict
from torch._dynamo import OptimizedModule
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from typing_extensions import Self

from agilerl import HAS_DEEPSPEED, HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES, HAS_VLLM

if HAS_LIGER_KERNEL:
    from liger_kernel.transformers import _apply_liger_kernel_to_instance
from agilerl.algorithms.core.llm_ops.fused_logprobs import (
    FusedLinearLogProbsFunction,
    fused_linear_logprobs_chunked,
)
from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    MutationHook,
    MutationRegistry,
    NetworkGroup,
    OptimizerConfig,
    OptimizerFactory,
)
from agilerl.metrics import AgentMetrics, MultiAgentMetrics
from agilerl.modules.configs import MlpNetConfig, NetConfig
from agilerl.modules.dummy import DummyEvolvable
from agilerl.protocols import (
    AgentWrapperProtocol,
    EvolvableAlgorithmProtocol,
    EvolvableModuleProtocol,
    ModuleDictProtocol,
    PeftModelProtocol,
    PretrainedConfigProtocol,
    PreTrainedModelProtocol,
)
from agilerl.typing import (
    ActionResult,
    ActionType,
    ArrayDict,
    CheckpointInfo,
    DeviceType,
    ExperiencesT,
    FitnessValue,
    GymSpaceType,
    InfosDict,
    LrNameType,
    MaybeActionMask,
    ModuleType,
    MultiAgentActionMasks,
    MultiAgentObservationType,
    MultiAgentSetup,
    MultiAgentSpacesType,
    NetConfigType,
    ObservationType,
    OptimizerType,
    ReasoningPrompts,
    TorchObsType,
    coerce_action_mask,
)
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    DummyOptimizer,
    VLLMConfig,
    _resolve_lr,
    check_supported_space,
    chkpt_attribute_to_device,
    clone_llm,
    concatenate_tensors,
    configure_tf32_precision,
    create_warmup_cosine_scheduler,
    filter_init_dict,
    get_input_size_from_space,
    get_output_size_from_space,
    isroutine,
    key_in_nested_dict,
    module_checkpoint_dict,
    needs_image_transpose,
    preprocess_observation,
    recursive_check_module_attrs,
    stack_and_pad_experiences,
    stack_experiences,
    transpose_image_space,
)
from agilerl.utils.evolvable_networks import (
    compile_model,
    config_from_dict,
    get_default_encoder_config,
    is_image_space,
    is_vector_space,
)
from agilerl.utils.llm_packing import (
    pack_padded_batch,
    unpack_logprobs,
    unpack_values,
)

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import SequentialLR
    from transformers import BitsAndBytesConfig

# Make imports visible to typechecker and import when required
if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file

    from agilerl.algorithms.core.llm_ops.fused_lora import (
        get_cached_lora_layers,
        patch_lora_for_fused_forward,
        set_fused_adapter_routing,
        unset_fused_adapter_routing,
    )
    from agilerl.algorithms.core.llm_ops.vllm_colocate import (
        patch_vllm_lora_keep_resident,
        patch_vllm_strip_multimodal_towers,
    )
    from agilerl.utils.algo_utils import clone_llm
    from agilerl.utils.llm_utils import (
        adapt_lora_config_for_model,
        align_deepspeed_lr,
        build_completion_mask,
        build_vllm_llm_init_kwargs,
        build_vllm_rollout_lora_request,
        create_model_from_name_or_path,
        gather_if_zero3,
        get_model_name_or_path,
        get_state_dict,
        log_cuda_memory_snapshot,
        move_params_to_cpu,
        move_params_to_gpu,
        offload_colocated_trainer_from_gpu,
        save_peft_adapter_for_vllm_rollout,
        stitch_completion_after_windowed_vllm_generate,
    )

if TYPE_CHECKING or HAS_DEEPSPEED:
    from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

if TYPE_CHECKING:
    from vllm import LLM, CompletionOutput, SamplingParams
elif HAS_VLLM:
    from vllm import LLM, CompletionOutput, SamplingParams
else:
    LLM = CompletionOutput = SamplingParams = None

__all__ = ["ActionResult", "EvolvableAlgorithm", "MultiAgentRLAlgorithm", "RLAlgorithm"]

logger = logging.getLogger(__name__)

SelfAgentWrapper = TypeVar("SelfAgentWrapper", bound=AgentWrapperProtocol)


class ClassicRLPopulationFactory(Protocol):
    index: int

    def __init__(
        self,
        observation_space: GymSpaceType | MultiAgentSpacesType,
        action_space: GymSpaceType | MultiAgentSpacesType,
        index: int,
        device: DeviceType = "cpu",
        **kwargs: Any,
    ) -> None: ...

    def load_checkpoint(self, path: str) -> None: ...


ClassicRLAlgoT = TypeVar("ClassicRLAlgoT", bound=ClassicRLPopulationFactory)


def build_classic_rl_population(
    cls: type[ClassicRLAlgoT],
    size: int,
    observation_space: GymSpaceType,
    action_space: GymSpaceType,
    device: DeviceType = "cpu",
    wrapper_cls: Callable[..., SelfAgentWrapper] | None = None,
    wrapper_kwargs: dict[str, Any] | None = None,
    resume_from_checkpoint: str | None = None,
    **kwargs: Any,
) -> list[ClassicRLAlgoT | SelfAgentWrapper]:
    """Build a population of classic RL algorithms (as opposed to LLM algorithms)."""
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    population: list[ClassicRLAlgoT | SelfAgentWrapper] = []
    for i in range(size):
        agent = cls(observation_space, action_space, index=i, device=device, **kwargs)
        if resume_from_checkpoint is not None:
            agent.load_checkpoint(resume_from_checkpoint)
            agent.index = i
        if wrapper_cls is not None:
            agent = wrapper_cls(agent, **wrapper_kwargs)
        population.append(agent)

    return population


# Generic so instantiating a concrete algorithm class types as that class,
# not as the EvolvableAlgorithm base.
AlgoT = TypeVar("AlgoT", bound="EvolvableAlgorithm")

# Bound to the structural interface shared by evolvable algorithms and the agent
# wrappers around them, so attribute-copying helpers accept either.
IndividualT = TypeVar(
    "IndividualT",
    bound="EvolvableAlgorithmProtocol | AgentWrapperProtocol[Any]",
)


class RegistryMeta(ABCMeta):
    """Metaclass that runs registry initialization on top of ABC support."""

    def __call__(
        cls: type[AlgoT],
        *args: Any,
        **kwargs: Any,
    ) -> AlgoT:
        # Create the instance
        instance: AlgoT = super().__call__(*args, **kwargs)

        # Initialize the MutationRegistry -> ensures that all of the networks and
        # optimizers are registered with the algorithm, and that the specified hyperparameters
        # to mutate have been set as attributes in the algorithm.
        if isinstance(instance, cls) and hasattr(instance, "_registry_init"):
            instance._registry_init()

        return instance


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

    # NOTE: this feels messy, refactor this to be more elegant
    if omit_actor_info and "actor" in attribute_dict:
        attribute_dict.pop("actor", None)
    if omit_optimizer_info and "optimizer" in attribute_dict:
        attribute_dict.pop("optimizer", None)
    lr_scheduler = attribute_dict.pop("lr_scheduler", None)
    if lr_scheduler is not None:
        attribute_dict["lr_scheduler"] = lr_scheduler.state_dict()

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
) -> OptimizerFactory | dict[str, OptimizerFactory]:
    """Return the optimizer class from the string or dictionary of optimizer classes.

    :param optimizer_cls: The optimizer class or dictionary of optimizer classes.
    :type optimizer_cls: str | dict[str, str]
    :return: The optimizer class or dictionary of optimizer classes.
    :rtype: OptimizerFactory | dict[str, OptimizerFactory]
    """
    if isinstance(optimizer_cls, dict):
        return {
            agent_id: getattr(torch.optim, cls_name)
            for agent_id, cls_name in optimizer_cls.items()
        }
    return getattr(torch.optim, optimizer_cls)


class EvolvableAlgorithm(ABC, Generic[ExperiencesT], metaclass=RegistryMeta):
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
    :type torch_compiler: str | None, optional
    :param name: Name of the algorithm, defaults to the class name.
    :type name: str | None, optional
    """

    metrics: AgentMetrics | MultiAgentMetrics
    # Optional LR scheduler, set by subclasses that use one (e.g. LLMAlgorithm).
    lr_scheduler: SequentialLR | None

    def __init__(
        self,
        index: int,
        hp_config: HyperparameterConfig | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        torch_compiler: str | None = None,
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
        self.algo = name or self.__class__.__name__
        self._mut = None
        self._index = index
        self.registry = MutationRegistry(hp_config)
        self.training = True
        self.subpopulation_id: int | None = None

    @property
    def index(self) -> int:
        """Return the index of the algorithm."""
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        """Set the index of the algorithm."""
        self._index = value

    @property
    def mut(self) -> str | None:
        """Return the mutation object of the algorithm."""
        return self._mut

    @mut.setter
    def mut(self, value: str | None) -> None:
        """Set the mutation object of the algorithm."""
        self._mut = value

    @property
    def hp_config(self) -> HyperparameterConfig:
        """Return the hyperparameter configuration for Evo-HPO mutations."""
        hp_config = self.registry.hp_config
        assert hp_config is not None  # MutationRegistry.__post_init__ guarantees this
        return hp_config

    @hp_config.setter
    def hp_config(self, value: HyperparameterConfig) -> None:
        """Set the hyperparameter configuration for Evo-HPO mutations."""
        self.registry.hp_config = value

    @property
    def steps(self) -> int:
        """Cumulative global step count."""
        return self.metrics.steps

    @steps.setter
    def steps(self, value: int) -> None:
        self.metrics.steps = value

    @property
    def scores(self) -> list[float | list[float]]:
        """Per-episode scores (per-group score rows for multi-agent metrics)."""
        return self.metrics.scores

    @scores.setter
    def scores(self, value: list[float | list[float]]) -> None:
        self.metrics.scores = value

    @property
    def fitness(self) -> list[FitnessValue]:
        """Fitness history (scalars, or per-sub-agent rows for multi-agent)."""
        return list(self.metrics.fitness)

    @fitness.setter
    def fitness(self, value: Iterable[FitnessValue]) -> None:
        maxlen = self.metrics.fitness.maxlen
        self.metrics.fitness = deque(value, maxlen=maxlen)

    def add_scores(self, scores: Sequence[float | list[float]]) -> None:
        """Add scores to the metrics.

        :param scores: List of scores (or per-agent score rows) to add.
        :type scores: Sequence[float | list[float]]
        """
        self.metrics.add_scores(scores)

    def init_training_step(self) -> None:
        """Initialize the training step for metrics tracking."""
        self.metrics.init_training_step()

    def finalize_training_step(self, num_steps: int) -> None:
        """Finalize the training step for metrics tracking.

        :param num_steps: Number of steps taken during the training step.
        :type num_steps: int
        """
        self.metrics.finalize_training_step(num_steps)

    @abstractmethod
    def preprocess_observation(
        self,
        observation: Any,  # noqa: ANN401 -- observation shape varies per algorithm (single obs vs per-agent mapping)
    ) -> TorchObsType | dict[str, TorchObsType]:
        """Preprocesses observations for forward pass through neural network.

        :param observation: Observations of environment
        :type observation: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]]
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, experiences: ExperiencesT) -> Any:  # noqa: ANN401 -- return type varies per algorithm (loss dict, tuple, etc.)
        """Abstract method for learning the algorithm."""
        raise NotImplementedError

    @abstractmethod
    def get_action(
        self,
        obs: ObservationType | MultiAgentObservationType,
        *args: Any,
        **kwargs: Any,
    ) -> ActionType | ActionResult | tuple[Any, ...]:
        """Abstract method for getting an action from the algorithm.

        :param obs: The observation to get an action for.
        :type obs: ObservationType | MultiAgentObservationType
        :param args: Additional arguments to pass to the action function.
        :type args: Any
        :param kwargs: Additional keyword arguments to pass to the action function.
        :type kwargs: Any
        :return: The action to take.
        :rtype: ActionType | ActionResult | tuple[Any, ...]
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, *args: Any, **kwargs: Any) -> float | npt.NDArray:
        """Abstract method for testing the algorithm."""
        raise NotImplementedError

    @staticmethod
    def get_state_dim(
        observation_space: spaces.Space | list[spaces.Space] | dict[str, spaces.Space],
    ) -> (
        tuple[int, ...]
        | dict[str, tuple[int, ...]]
        | tuple[tuple[int, ...] | dict[str, tuple[int, ...]], ...]
    ):
        """Return the dimension of the state space as it pertains to the underlying
        networks (i.e. the input size of the networks).

        :param observation_space: The observation space of the environment.
        :type observation_space: spaces.Space or list[spaces.Space].

        :return: The dimension of the state space.
        :rtype: tuple[int, ...] | dict[str, tuple[int, ...]]
        """
        warnings.warn(
            "This method is deprecated. Use get_input_size_from_space instead.",
            stacklevel=2,
            category=DeprecationWarning,
        )
        return get_input_size_from_space(observation_space)

    @staticmethod
    def get_action_dim(
        action_space: spaces.Space | list[spaces.Space] | dict[str, spaces.Space],
    ) -> int | dict[str, int] | tuple[int | dict[str, int], ...]:
        """Return the dimension of the action space as it pertains to the underlying
        networks (i.e. the output size of the networks).

        :param action_space: The action space of the environment.
        :type action_space: spaces.Space or list[spaces.Space].

        :return: The dimension of the action space.
        :rtype: int | dict[str, int] | tuple[int | dict[str, int], ...]
        """
        warnings.warn(
            "This method is deprecated. Use get_output_size_from_space instead.",
            stacklevel=2,
            category=DeprecationWarning,
        )
        return get_output_size_from_space(action_space)

    @staticmethod
    def inspect_attributes(
        agent: EvolvableAlgorithmProtocol | AgentWrapperProtocol[Any],
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
        agent: IndividualT,
        clone: IndividualT,
    ) -> IndividualT:
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

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401 -- __setattr__ accepts any attribute value
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

    def _wrap_attr(self, attr: Any) -> Any:  # noqa: ANN401 -- accelerator-wrapped network/optimizer has no static type
        """Wrap an evolvable attribute (network or optimizer) with the accelerator.

        :param attr: The attribute to wrap.
        :type attr: Any

        :return: The wrapped attribute.
        :rtype: Any
        """
        accelerator = self.accelerator
        assert accelerator is not None  # Guarded by wrap_models
        if isinstance(attr, OptimizerWrapper):
            if isinstance(attr.optimizer, dict):
                wrapped_opt = {
                    agent_id: accelerator.prepare(opt)
                    for agent_id, opt in attr._optimizers_by_agent().items()
                }
            else:
                wrapped_opt = accelerator.prepare(attr.optimizer)

            attr.optimizer = wrapped_opt
            return attr

        # Only wrap the model if its part of the computation graph
        return accelerator.prepare(attr) if attr.state_dict() else attr

    def _reinit_opt_from_config(
        self,
        config: OptimizerConfig,
    ) -> None:
        """Reinitializes an optimizer from its configuration.

        :param config: The optimizer configuration.
        :type config: OptimizerConfig
        """
        opt = getattr(self, config.name)
        optimizer = getattr(opt, "optimizer", None)

        if isinstance(self, LLMAlgorithm):
            if hasattr(self.actor, "optimizer"):
                # If the optimizer is defined in the deepspeed config, we do this
                optimizer = self.actor.optimizer
            else:
                optimizer = opt.optimizer

            lr = (
                tuple(getattr(self, lr_name) for lr_name in config.lr)
                if isinstance(config.lr, tuple)
                else getattr(self, config.lr)
            )
            self.accelerator, self.lr_scheduler = LLMAlgorithm.update_lr(
                optimizer,
                lr=lr,
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

    def get_lr_names(self) -> list[LrNameType]:
        """Return the learning-rate attribute name(s) of each optimizer."""
        return [opt.lr for opt in self.registry.optimizers]

    def register_network_group(self, group: NetworkGroup) -> None:
        """Set the evaluation network for the algorithm.

        :param name: The name of the evaluation network.
        :type name: str
        """
        self.registry.register_group(group)

    def register_mutation_hook(self, hook: MutationHook) -> None:
        """Register a hook to be executed after a mutation is performed on
        the algorithm.

        :param hook: The hook to be executed after mutation.
        :type hook: MutationHook
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
                return getattr(self, group.eval_network_name())

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
        on_device: list[TorchObsType] = []
        for exp in experiences:
            moved: TorchObsType
            # Check the Tensor leaf before the container arms so narrowing is exact.
            if isinstance(exp, torch.Tensor):
                moved = exp.to(device)
            elif isinstance(exp, dict):
                moved = {key: val.to(device) for key, val in exp.items()}
            elif isinstance(exp, (list, tuple)) and isinstance(exp[0], torch.Tensor):
                moved = tuple(val.to(device) for val in exp)
            else:
                moved = exp
            on_device.append(moved)

        return tuple(on_device)

    def evolvable_attributes(
        self,
        networks_only: bool = False,
    ) -> dict[str, Any]:
        """Return the attributes related to the evolvable networks in the algorithm. Includes
        attributes that are either EvolvableModule or ModuleDict objects, as well as the optimizers
        associated with the networks.

        :param networks_only: If True, only include evolvable networks, defaults to False
        :type networks_only: bool, optional

        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """

        def is_evolvable(attr: str, obj: object) -> bool:
            return (
                recursive_check_module_attrs(obj, networks_only)
                and not attr.startswith("_")
                and not attr.endswith("_")
            )

        # Inspect evolvable given specs
        evolvable_attrs: dict[str, Any] = {}
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
        cloned_modules: dict[str, Any] = {}
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
            lr_value, lr_critic_value = _resolve_lr(self, opt_config.lr)
            opt = OptimizerWrapper(
                optim_cls,
                networks=networks,
                lr=lr_value,
                lr_critic=lr_critic_value,
                is_llm_optimizer=getattr(orig_optimizer, "is_llm_optimizer", False),
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
            configure_tf32_precision()
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
        network_info: CheckpointInfo = checkpoint["network_info"]
        modules = network_info["modules"]
        optimizers = network_info["optimizers"]
        network_names = network_info["network_names"]
        for name in network_names:
            net_dict = {k: v for k, v in modules.items() if k.startswith(name)}

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

                assert module_dict_cls is not None, (
                    f"Missing '{name}_module_dict_cls' entry for multi-agent "
                    "network in checkpoint."
                )
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
            net_dict = {k: v for k, v in modules.items() if k.startswith(name)}
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
            opt_dict = {k: v for k, v in optimizers.items() if k.startswith(name)}

            # Initialize optimizer
            opt_kwargs = opt_dict[f"{name}_kwargs"]
            optimizer_cls = get_optimizer_cls(opt_dict[f"{name}_cls"])
            opt_networks = opt_dict[f"{name}_networks"]
            opt_lr = opt_dict[f"{name}_lr"]
            is_llm_optimizer = bool(opt_dict.get(f"{name}_is_llm_optimizer", False))
            lr, lr_critic = _resolve_lr(self, opt_lr)
            networks = [getattr(self, net) for net in opt_networks]
            optimizer = OptimizerWrapper(
                optimizer_cls=optimizer_cls,
                networks=networks,
                lr=lr,
                optimizer_kwargs=opt_kwargs,
                network_names=opt_networks,
                lr_name=opt_lr,
                lr_critic=lr_critic,
                is_llm_optimizer=is_llm_optimizer,
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
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state_dict=checkpoint["lr_scheduler"])
            checkpoint.pop("lr_scheduler")

        # Load other attributes
        checkpoint.pop("network_info")
        # Pre-2.8 checkpoints stored ``steps`` as a cumulative list; coerce to
        # the int expected by the metrics tracker.
        if isinstance(checkpoint.get("steps"), (list, tuple)):
            legacy_steps = checkpoint["steps"]
            checkpoint["steps"] = int(legacy_steps[-1]) if len(legacy_steps) else 0
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
            configure_tf32_precision()
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
        network_info: CheckpointInfo | None = checkpoint.get("network_info")
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

        modules = network_info["modules"]
        optimizers = network_info["optimizers"]
        network_names = network_info["network_names"]
        loaded_modules: dict[str, Any] = {}
        for name in network_names:
            net_dict = {k: v for k, v in modules.items() if k.startswith(name)}

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
            net_dict = {k: v for k, v in modules.items() if k.startswith(name)}
            loaded_module = getattr(self, name)
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
            opt_dict = {k: v for k, v in optimizers.items() if k.startswith(name)}

            # Add device to optimizer kwargs
            opt_kwargs = chkpt_attribute_to_device(opt_dict[f"{name}_kwargs"], device)
            lr = opt_dict[f"{name}_lr"]
            is_llm_optimizer = bool(opt_dict.get(f"{name}_is_llm_optimizer", False))
            optimizer_cls = get_optimizer_cls(opt_dict[f"{name}_cls"])
            opt_networks = opt_dict[f"{name}_networks"]
            lr_value, lr_critic_value = _resolve_lr(self, lr)
            networks = [loaded_modules[net] for net in opt_networks]
            optimizer = OptimizerWrapper(
                optimizer_cls=optimizer_cls,
                networks=networks,
                lr=lr_value,
                network_names=opt_networks,
                lr_name=lr,
                optimizer_kwargs=opt_kwargs,
                lr_critic=lr_critic_value,
                is_llm_optimizer=is_llm_optimizer,
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
            configure_tf32_precision()
            self.recompile()

        # Check for agent wrapper
        wrapper_cls = checkpoint.get("wrapper_cls")
        if wrapper_cls is not None:
            init_dict = checkpoint.get("wrapper_init_dict") or {}
            wrapper_attributes = checkpoint.get("wrapper_attrs") or {}
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


class RLAlgorithm(EvolvableAlgorithm[ExperiencesT], ABC, Generic[ExperiencesT]):
    """Base object for all single-agent algorithms in the AgileRL framework.

    :param observation_space: The observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: The action space of the environment.
    :type action_space: spaces.Space
    :param index: The index of the individual.
    :type index: int
    :param hp_config: Hyperparameter configuration for the algorithm, defaults to None.
    :type hp_config: HyperparameterConfig | None, optional
    :param device: Device to run the algorithm on, defaults to "cpu".
    :type device: str | torch.device, optional
    :param accelerator: Accelerator object for distributed computing, defaults to None.
    :type accelerator: Accelerator | None, optional
    :param torch_compiler: The torch compiler mode to use, defaults to None.
    :type torch_compiler: str | None, optional
    :param normalize_images: If True, normalize images, defaults to True.
    :type normalize_images: bool, optional
    :param name: Name of the algorithm, defaults to the class name.
    :type name: str | None, optional
    """

    @classmethod
    def population(
        cls,
        size: int,
        observation_space: GymSpaceType,
        action_space: GymSpaceType,
        device: DeviceType = "cpu",
        wrapper_cls: Callable[..., SelfAgentWrapper] | None = None,
        wrapper_kwargs: dict[str, Any] | None = None,
        resume_from_checkpoint: str | None = None,
        **kwargs: Any,
    ) -> list[Self | SelfAgentWrapper]:
        """Create a population of algorithms.

        :param size: The size of the population.
        :type size: int
        :param observation_space: The observation space.
        :type observation_space: GymSpaceType
        :param action_space: The action space.
        :type action_space: GymSpaceType
        :param device: Torch device. Defaults to ``"cpu"``.
        :type device: DeviceType
        :param wrapper_cls: Optional wrapper class to apply to each agent.
        :type wrapper_cls: type | None
        :param wrapper_kwargs: Keyword arguments for the wrapper class.
        :type wrapper_kwargs: dict[str, Any] | None
        :param resume_from_checkpoint: Path to checkpoint to resume from.
        :type resume_from_checkpoint: str | None
        :param kwargs: Additional keyword arguments to pass to the algorithm constructor.
        :type kwargs: Any
        :return: A list of algorithms.
        :rtype: list[RLAlgorithm]
        """
        return build_classic_rl_population(
            cls,
            size,
            observation_space,
            action_space,
            device,
            wrapper_cls,
            wrapper_kwargs,
            resume_from_checkpoint,
            **kwargs,
        )

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        index: int,
        hp_config: HyperparameterConfig | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        torch_compiler: str | None = None,
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
        self.swap_channels = needs_image_transpose(self.observation_space)
        self.env_observation_space = observation_space
        if self.swap_channels:
            logger.warning(
                "Found channels-last observation space. "
                "AgileRL automatically transposes images to be channels-first to support PyTorch convolutions.",
                stacklevel=2,
            )
            self.observation_space = transpose_image_space(self.observation_space)

        self.metrics = AgentMetrics()

    def preprocess_observation(self, observation: ObservationType) -> TorchObsType:
        """Preprocesses observations for forward pass through neural network.

        :param observation: Observations of environment
        :type observation: ObservationType
        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or tuple[torch.Tensor[float], ...]
        """
        return preprocess_observation(
            self.observation_space,
            observation=observation,
            device=self.device,
            normalize_images=self.normalize_images,
            swap_channels=self.swap_channels,
        )


class MultiAgentRLAlgorithm(
    EvolvableAlgorithm[ExperiencesT], ABC, Generic[ExperiencesT]
):
    """Base object for all multi-agent algorithms in the AgileRL framework.

    :param observation_spaces: The observation spaces of the agent environments.
    :type observation_spaces: MultiAgentSpacesType
    :param action_spaces: The action spaces of the agent environments.
    :type action_spaces: MultiAgentSpacesType
    :param index: The index of the individual in the population.
    :type index: int.
    :param agent_ids: The agent IDs of the agents in the environment.
    :type agent_ids: list[int] | None, optional
    :param hp_config: Hyperparameter configuration for the algorithm, defaults to None.
    :type hp_config: HyperparameterConfig | None, optional
    :param device: Device to run the algorithm on, defaults to "cpu"
    :type device: str, optional
    :param accelerator: Accelerator object for distributed computing, defaults to None
    :type accelerator: Accelerator | None, optional
    :param torch_compiler: The torch compiler mode to use, defaults to None
    :type torch_compiler: str | None, optional
    :param normalize_images: If True, normalize images, defaults to True
    :type normalize_images: bool, optional
    :param placeholder_value: The value to use as placeholder for missing observations, defaults to -1.
    :type placeholder_value: float | None, optional
    :param name: Name of the algorithm, defaults to the class name
    :type name: str | None, optional
    """

    metrics: MultiAgentMetrics

    possible_observation_spaces: spaces.Dict
    possible_action_spaces: spaces.Dict

    shared_agent_ids: list[str]
    grouped_agents: dict[str, list[str]]
    unique_observation_spaces: dict[str, spaces.Space]
    unique_action_spaces: dict[str, spaces.Space]

    @classmethod
    def population(
        cls,
        size: int,
        observation_space: GymSpaceType,
        action_space: GymSpaceType,
        device: DeviceType = "cpu",
        wrapper_cls: Callable[..., SelfAgentWrapper] | None = None,
        wrapper_kwargs: dict[str, Any] | None = None,
        resume_from_checkpoint: str | None = None,
        **kwargs: Any,
    ) -> list[Self | SelfAgentWrapper]:
        """Create a population of algorithms.

        :param size: The size of the population.
        :type size: int
        :param observation_space: The observation spaces of the agents.
        :type observation_space: GymSpaceType
        :param action_space: The action spaces of the agents.
        :type action_space: GymSpaceType
        :param device: Torch device. Defaults to ``"cpu"``.
        :type device: DeviceType
        :param wrapper_cls: Optional wrapper class to apply to each agent.
        :type wrapper_cls: type | None
        :param wrapper_kwargs: Keyword arguments for the wrapper class.
        :type wrapper_kwargs: dict[str, Any] | None
        :param resume_from_checkpoint: Path to checkpoint to resume from.
        :type resume_from_checkpoint: str | None
        :param kwargs: Additional keyword arguments to pass to the algorithm constructor.
        :type kwargs: Any
        :return: A list of algorithms.
        :rtype: list[MultiAgentRLAlgorithm]
        """
        return build_classic_rl_population(
            cls,
            size,
            observation_space,
            action_space,
            device,
            wrapper_cls,
            wrapper_kwargs,
            resume_from_checkpoint,
            **kwargs,
        )

    def __init__(
        self,
        observation_spaces: MultiAgentSpacesType,
        action_spaces: MultiAgentSpacesType,
        index: int,
        agent_ids: Iterable[str] | None = None,
        hp_config: HyperparameterConfig | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        torch_compiler: str | None = None,
        normalize_images: bool = True,
        placeholder_value: float | None = -1,
        name: str | None = None,
    ) -> None:

        super().__init__(index, hp_config, device, accelerator, torch_compiler, name)

        # Reject scalars/strings up front (a non-``isinstance(list)`` check so the
        # per-agent ``Iterable[spaces.Space]`` element type survives narrowing).
        if isinstance(observation_spaces, str) or not hasattr(
            observation_spaces, "__iter__"
        ):
            msg = "Observation spaces must be a list or dictionary."
            raise TypeError(msg)

        assert type(observation_spaces) is type(action_spaces), (
            "Observation spaces and action spaces must be the same type. "
            f"Got {type(observation_spaces)} and {type(action_spaces)}."
        )

        if isinstance(observation_spaces, spaces.Dict):
            assert isinstance(action_spaces, spaces.Dict), (
                "Action spaces must also be passed as a spaces.Dict."
            )
            self.possible_observation_spaces = observation_spaces
            self.possible_action_spaces = action_spaces
        elif isinstance(observation_spaces, Mapping):
            assert isinstance(action_spaces, Mapping), (
                "Action spaces must also be passed as a mapping."
            )
            self.possible_observation_spaces = spaces.Dict(
                dict(observation_spaces),
            )
            self.possible_action_spaces = spaces.Dict(dict(action_spaces))
        else:
            # A sequence of per-agent spaces paired with agent_ids. Excluding the
            # mapping cases above preserves the Iterable[spaces.Space] element
            # type that an isinstance(list, tuple) check would erase.
            assert agent_ids is not None, (
                "Agent IDs must be specified if observation spaces are passed as a list."
            )
            assert not isinstance(action_spaces, Mapping), (
                "Action spaces must also be passed as a list."
            )
            agent_id_list = list(agent_ids)
            obs_space_list = list(observation_spaces)
            action_space_list = list(action_spaces)
            assert len(agent_id_list) == len(obs_space_list), (
                "Number of agent IDs must match number of observation spaces."
            )
            self.possible_observation_spaces = spaces.Dict(
                dict(zip(agent_id_list, obs_space_list, strict=False)),
            )
            self.possible_action_spaces = spaces.Dict(
                dict(zip(agent_id_list, action_space_list, strict=False)),
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

        # Check if any observation space is channels-last and transpose if necessary
        self.swap_channels = needs_image_transpose(self.possible_observation_spaces)
        self.env_observation_spaces = self.possible_observation_spaces
        if self.swap_channels:
            logger.warning(
                "Found channels-last observation space. "
                "AgileRL automatically transposes images to be channels-first to support PyTorch convolutions.",
                stacklevel=2,
            )
            # transpose_image_space preserves the space structure, so a Dict
            # space transposes to a Dict space.
            transposed = transpose_image_space(self.possible_observation_spaces)
            assert isinstance(transposed, spaces.Dict)
            self.possible_observation_spaces = transposed

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

        # Track multi-agent metrics using the effective training IDs. In grouped
        # setups this corresponds to shared group IDs; otherwise raw agent IDs.
        self.metrics = MultiAgentMetrics(list(self.observation_space.keys()))

    def _registry_init(self) -> None:
        super()._registry_init()

        from agilerl.modules import ModuleDict

        # Additional check to ensure multi-agent networks are initialized with valid keys
        for name, network in self.evolvable_attributes(networks_only=True).items():
            if isinstance(network, ModuleDict):
                for key in network:
                    if key not in set(self.agent_ids + self.shared_agent_ids):
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

    def add_scores(self, scores: Sequence[float | list[float]]) -> None:
        """Add scores to the metrics, aggregating sub-agents into their groups.

        Multi-agent training loops collect non-summed score rows with one
        entry per environment agent. When agents share policies (grouped
        setups) the metrics track group IDs instead, so each row is reduced
        to the mean score per group before being recorded.

        :param scores: List of scores (or per-agent score rows) to add.
        :type scores: Sequence[float | list[float]]
        """
        is_nested = bool(scores) and isinstance(scores[0], (list, np.ndarray))
        # Grouped setups track metrics under group IDs, so per-env-agent rows
        # must be reduced to a per-group mean before being recorded.
        is_grouped = (
            is_nested
            and self.has_grouped_agents()
            and self.metrics.agent_ids == self.shared_agent_ids
        )
        if is_grouped:
            # ``is_nested`` established that rows are per-agent sequences; pin
            # each one so the per-group reduction can index it. The only shape
            # these loops produce is one entry per raw env agent; anything else
            # would mislabel group columns, so fail loudly rather than misrecord.
            score_rows: list[list[float] | np.ndarray] = []
            for row in scores:
                grouped_error = (
                    "Grouped multi-agent scores expected one entry per agent "
                    f"({len(self.agent_ids)} agents: {self.agent_ids})."
                )
                assert isinstance(row, (list, np.ndarray)), grouped_error
                assert len(row) == len(self.agent_ids), grouped_error
                score_rows.append(row)
            column = {aid: idx for idx, aid in enumerate(self.agent_ids)}
            group_columns = [
                [column[aid] for aid in self.grouped_agents[gid]]
                for gid in self.shared_agent_ids
            ]
            scores = [
                [float(np.mean([row[idx] for idx in cols])) for cols in group_columns]
                for row in score_rows
            ]
        super().add_scores(scores)

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
        observation: Mapping[str, ObservationType],
        group_ids: list[str] | None = None,
    ) -> dict[str, TorchObsType]:
        """Preprocesses observations for forward pass through neural network.

        :param observation: Per-agent observations of the environment.
        :type observation: Mapping[str, ObservationType]
        :param group_ids: Optional list of output IDs. When group IDs are provided
            (e.g., ``["agent", "other_agent"]``), observations are grouped and
            concatenated per group. Otherwise, observations are returned per
            agent ID for backwards compatibility.
        :type group_ids: list[str] | None

        :return: Preprocessed observations
        :rtype: dict[str, TorchObsType]
        """
        obs_dict = observation
        if group_ids is None:
            preprocessed: dict[str, TorchObsType] = {}
            for agent_id, agent_obs in obs_dict.items():
                preprocessed[agent_id] = preprocess_observation(
                    self.possible_observation_spaces.get(agent_id),
                    observation=agent_obs,
                    device=self.device,
                    normalize_images=self.normalize_images,
                    placeholder_value=self.placeholder_value,
                )
            return preprocessed

        buckets: dict[str, list[TorchObsType]] = {
            group_id: [] for group_id in group_ids
        }
        for agent_id, agent_obs in obs_dict.items():
            output_id = self.get_network_id(agent_id)
            if output_id not in buckets:
                buckets[output_id] = []

            buckets[output_id].append(
                preprocess_observation(
                    self.observation_space.get(output_id),
                    observation=agent_obs,
                    device=self.device,
                    normalize_images=self.normalize_images,
                    swap_channels=self.swap_channels,
                    placeholder_value=self.placeholder_value,
                )
            )
        # Populated buckets concatenate to a single tensor; empty buckets (a
        # group with no active agent) become an empty tensor so every supplied
        # group id is present in the output without widening the value type.
        return {
            output_id: (
                concatenate_tensors(obs_list)
                if obs_list
                else torch.empty(0, device=self.device)
            )
            for output_id, obs_list in buckets.items()
        }

    def extract_action_masks(self, infos: InfosDict) -> MultiAgentActionMasks:
        """Extract action masks from info dictionary.

        :param infos: Info dict
        :type infos: InfosDict

        :return: Action masks (``None`` for agents without one). The return is
            a read-only mapping so subclasses may specialise the value type:
            the base yields raw numpy masks; on-policy multi-agent subclasses
            (e.g. IPPO) stack them into per-group tensors.
        :rtype: MultiAgentActionMasks
        """
        # Get dict of form {"agent_id" : [1, 0, 0, 0]...} etc
        action_masks: dict[str, MaybeActionMask] = {}
        for agent, info in infos.items():
            if agent not in self.agent_ids:
                continue
            # Real envs occasionally hand back a non-mapping per-agent info
            # (e.g. a bare string); treat it as carrying no mask.
            mask = info.get("action_mask", None) if isinstance(info, Mapping) else None
            action_masks[agent] = coerce_action_mask(mask)
        return action_masks

    def extract_agent_masks(
        self,
        infos: InfosDict | None = None,
    ) -> tuple[ArrayDict | None, ArrayDict | None]:
        """Extract env_defined_actions from info dictionary and determine agent masks.

        :param infos: Info dict
        :type infos: InfosDict | None

        :return: Env defined actions and agent masks (both ``None`` when the
            info dict defines no actions). Actions are normalized to arrays.
        :rtype: tuple[ArrayDict | None, ArrayDict | None]
        """
        # Deal with case of no env_defined_actions defined in the info dict
        # Deal with empty info dicts for each sub agent
        if (
            infos is None
            or not key_in_nested_dict(infos, "env_defined_actions")
            or all(not info for agent, info in infos.items() if agent in self.agent_ids)
        ):
            return None, None

        raw_actions: dict[str, int | float | np.ndarray | torch.Tensor | None] = {}
        for agent, info in infos.items():
            if agent not in self.agent_ids:
                continue
            raw = info.get("env_defined_actions", None)
            if raw is None or isinstance(raw, (int, float, np.ndarray, torch.Tensor)):
                raw_actions[agent] = raw
            else:
                raw_actions[agent] = None
        env_defined_actions: ArrayDict = {}
        agent_masks: ArrayDict = {}
        for agent_id, action_val in raw_actions.items():
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

                val = nan_arr

            # Handle discrete actions + env not vectorized
            if isinstance(val, (int, float)):
                val = np.array([val])
            elif isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()

            env_defined_actions[agent_id] = val
            agent_masks[agent_id] = np.where(
                np.isnan(val),
                0,
                1,
            ).astype(bool)

        return env_defined_actions, agent_masks

    @overload
    def build_net_config(
        self,
        net_config: NetConfigType | None = ...,
        flatten: bool = ...,
        return_encoders: Literal[False] = ...,
    ) -> NetConfigType: ...

    @overload
    def build_net_config(
        self,
        net_config: NetConfigType | None = ...,
        flatten: bool = ...,
        *,
        return_encoders: Literal[True],
    ) -> tuple[NetConfigType, dict[str, NetConfigType]]: ...

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
        def _add_to_encoder_configs(config: NetConfigType, agent_id: str = "") -> None:
            net_config = config_from_dict(config)
            config_key = (
                "mlp_config" if isinstance(net_config, MlpNetConfig) else agent_id
            )

            if config_key not in encoder_configs or (
                isinstance(net_config, MlpNetConfig)
                and len(net_config["hidden_size"])
                > len(
                    encoder_configs["mlp_config"]["hidden_size"],
                )
            ):
                encoder_configs[config_key] = asdict(net_config)

        # Helper function to check if any agent ID exists in the net_config
        def _has_agent_ids(config: NetConfigType) -> bool:
            return any(
                (agent_id in self.agent_ids) or (agent_id in self.shared_agent_ids)
                for agent_id in config
            )

        # Helper function to get or create encoder config for an agent
        def _get_encoder_config(config: NetConfigType, agent_id: str) -> NetConfigType:
            simba = bool(config.get("simba", False))
            if "encoder_config" not in config or config.get("encoder_config") is None:
                encoder_config = get_default_encoder_config(
                    observation_spaces[agent_id],
                    simba,
                )
                config["encoder_config"] = encoder_config
                return encoder_config
            encoder_config = config["encoder_config"]
            if not isinstance(encoder_config, (dict, NetConfig)):
                msg = (
                    f"encoder_config for agent {agent_id!r} must be a dict or "
                    f"NetConfig, got {type(encoder_config).__name__}"
                )
                raise TypeError(msg)
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
        :rtype: str
        """
        return agent_id.rsplit("_", 1)[0] if isinstance(agent_id, str) else agent_id

    def get_network_id(self, agent_id: str) -> str:
        """Get the actor/critic network ID for an agent.

        :param agent_id: The agent ID
        :type agent_id: str
        :return: The network ID
        :rtype: str
        """
        return self.get_group_id(agent_id) if self.has_grouped_agents() else agent_id

    def assemble_shared_inputs(
        self,
        experience: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Preprocesses inputs by constructing dictionaries by shared agents.

        :param experience: per-agent experience to reshape from environment
        :type experience: Mapping[str, Any]

        :return: Preprocessed inputs, grouped by shared agents
        :rtype: dict[str, dict[str, Any]]
        """
        stacked_experience: dict[str, dict[str, Any]] = {
            group_id: {} for group_id in self.observation_space
        }
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
        :type group_outputs: dict[str, npt.NDArray]
        :param vect_dim: Vectorization dimension size, i.e. number of vect envs
        :type vect_dim: int
        :param grouped_agents: Dictionary of grouped agent IDs
        :type grouped_agents: dict[str, list[str]]
        :return: Assembled dictionary, e.g. {'agent_0': 4, 'agent_1': 7, 'agent_2': 8}
        :rtype: dict[str, npt.NDArray]
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

    def sum_shared_rewards(
        self, rewards: Mapping[str, npt.NDArray | float | int]
    ) -> ArrayDict:
        """Sum the rewards for grouped agents.

        :param rewards: Reward dictionary from environment. Vectorised envs
            provide arrays; a non-vectorised ``ParallelEnv`` provides scalars.
        :type rewards: dict[str, npt.NDArray | float]
        :return: Summed rewards dictionary
        :rtype: dict[str, npt.NDArray]
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
        :type agent_outputs: dict[str, npt.NDArray]
        :param vect_dim: Vectorization dimension size, i.e. number of vect envs
        :type vect_dim: int
        :return: Assembled dictionary with the form {'agent': [4, 7, 8]}
        :rtype: dict[str, npt.NDArray]
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


def _vllm_sampled_token_logprobs(output: CompletionOutput) -> list[float]:
    """Per-token logprob of the *sampled* token from a vLLM ``CompletionOutput``.

    With ``SamplingParams(logprobs=0)`` vLLM returns, per generated position, a
    dict that always contains the sampled token. Missing or non-finite entries
    fall back to ``0.0`` (yielding a unit importance-sampling ratio for that
    token, since the correction multiplies the loss by ``exp(old - sampling)``).

    :param output: A vLLM ``CompletionOutput`` (``token_ids`` + ``logprobs``).
    :type output: CompletionOutput
    :return: One sampled-token logprob per generated token.
    :rtype: list[float]
    """
    token_ids = output.token_ids
    logprobs = getattr(output, "logprobs", None)
    if not logprobs:
        return [0.0] * len(token_ids)
    out: list[float] = []
    for tok, lp_dict in zip(token_ids, logprobs, strict=False):
        entry = lp_dict.get(tok) if lp_dict else None
        val = float(entry.logprob) if entry is not None else 0.0
        # Only finite logprobs count (rejects NaN and ±inf).
        out.append(val if np.isfinite(val) else 0.0)
    return out


class LLMAlgorithm(EvolvableAlgorithm[ExperiencesT], ABC, Generic[ExperiencesT]):
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
    :param use_liger_loss: Whether to use Liger loss. Defaults to ``False``.
        Passing ``True`` without ``liger-kernel`` installed warns and falls
        back to ``False``.
    :type use_liger_loss: bool
    :param lora_config: The LoRA config.
    :type lora_config: LoraConfig | None
    :param use_separate_reference_adapter: Keep a dedicated ``reference`` LoRA
        adapter that reference-policy updates copy the actor adapter onto. When
        ``False`` (default) the reference is the base model with adapters
        disabled; since base weights are immutable, that reference stays the
        initial policy for the whole run.
    :type use_separate_reference_adapter: bool
    :param lr_critic: Critic/value-head learning rate. If ``None``, ``lr_actor`` is used.
    :type lr_critic: float | None, optional
    :param use_value_head: Whether to use a separate value head.
    :type use_value_head: bool
    :param use_vllm: Whether to route generation through vLLM.
    :type use_vllm: bool, optional
    :param vllm_config: vLLM runtime configuration.
    :type vllm_config: VLLMConfig | None, optional
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
    :param use_memory_efficient_params: For colocated vLLM, offload the trainer's
        own base to CPU during rollout (and bring it back for the training step)
        so the rollout engine and the trainer never both hold a base on the GPU.
        Defaults to True; inert without colocated vLLM.
    :type use_memory_efficient_params: bool, optional
    :param wrap: Whether to wrap the model.
    :type wrap: bool
    :param device: The device to run the algorithm on.
    :type device: str | torch.device
    :param accelerator: The accelerator to use.
    :type accelerator: Accelerator | None
    :param name: The name of the algorithm.
    :type name: str | None
    :param model_config: Keyword arguments for ``from_pretrained`` (not the HF
        ``PretrainedConfig`` object). AgileRL-only keys such as ``lora_target_scope``
        must not be placed here; use the dedicated ``lora_target_scope`` argument.
    :type model_config: dict[str, Any] | PretrainedConfig | None
    :param lora_target_scope: Optional PEFT LoRA path scope for multimodal models
        (e.g. ``"language_model"``). Passed to :func:`adapt_lora_config_for_model`.
    :type lora_target_scope: str | None, optional
    :param chunk_rows: Primary chunk-size knob for fused logit tiles used by
        both the standard fused-logprob path and the Liger fused-loss path.
        ``None`` (default) preserves each path's auto-tuned behavior.
    :type chunk_rows: int | None, optional
    :param vllm_importance_sampling_correction: When ``True`` (default) and
        ``use_vllm=True``, correct the rollout/trainer log-prob mismatch by
        weighting each training token by ``clamp(exp(trainer - sampling),
        max=vllm_importance_sampling_cap)``. Active only for training rollouts;
        inert on the HuggingFace path and at eval.
    :type vllm_importance_sampling_correction: bool, optional
    :param vllm_importance_sampling_cap: Upper clamp on the vLLM
        importance-sampling ratio (default ``2.0``), bounding the correction
        weight to limit variance from outlier tokens. Must be > 0.
    :type vllm_importance_sampling_cap: float, optional
    :param gradient_checkpointing: Whether to use gradient checkpointing.
    :type gradient_checkpointing: bool
    :param torch_compiler: The torch compiler mode to use ('default',
        'reduce-overhead', or 'max-autotune'), defaults to None.
    :type torch_compiler: str | None, optional
    :param reduce_memory_peak: Deprecated. Previously hinted peak-memory batching;
        ignored. Configure ``micro_batch_size_per_gpu`` and DeepSpeed instead.
    :type reduce_memory_peak: bool, optional
    :param cast_logprobs_to_fp32: When ``True`` (the default), the per-token
        log-probability reduction (``gather`` / ``logsumexp``) runs in fp32
        before being cast back to the input dtype, for numerically stable
        log-probs.

        Setting ``False`` runs the reduction in the input dtype, introducing a
        per-token bf16 quantisation error (~0.1 at ``V≈128k``) which can bias
        PPO/GRPO importance-sampling ratios. Use only if you've verified bf16 is
        acceptable for your vocab/shape — it saves ~6 MB on the
        fused-linear-logprob path.
    :type cast_logprobs_to_fp32: bool, optional
    :param quantization_config: Optional ``transformers.BitsAndBytesConfig`` for
        loading the base model in 4-/8-bit (QLoRA). ``lm_head`` is kept
        unquantized so the fused-linear-logprob path stays numerically exact.
    :type quantization_config: BitsAndBytesConfig | None, optional
    :param activation_offload: When ``True``, run the training forward inside
        ``torch.autograd.graph.save_on_cpu`` so tensors saved for backward live
        in pinned host RAM instead of GPU memory. Trades PCIe bandwidth for GPU
        memory (the win grows with sequence length); a no-op during rollout /
        reference forwards.
    :type activation_offload: bool, optional
    :param use_sequence_packing: Opt in to padding-free sequence packing for the
        gradient forward (sequences pack into one varlen / blockmask pass). Only
        honoured under a FlashAttention-2 / FlexAttention backend, otherwise
        inert; the no-grad reference/old-logprob pass stays padded.
    :type use_sequence_packing: bool, optional
    """

    _allowed_adapters = frozenset({"actor", "reference", "critic"})
    # Adapter exported to vLLM for rollout — always the actor (also the
    # ``LoRARequest`` name the llm_utils helpers default to).
    _vllm_rollout_adapter = "actor"

    # Runtime handles that cross HF/PEFT/DeepSpeed/vLLM wrapper boundaries;
    # their concrete types depend on the launch configuration (Accelerate
    # prepare(), DeepSpeed engines, DummyEvolvable, colocated vLLM), so they
    # are deliberately duck-typed. ``clean_up`` resets them to None.
    actor: Any
    optimizer: Any
    llm: Any
    tp_group: Any
    lr: float

    temperature: float
    repetition_penalty: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    max_model_len: int
    max_output_tokens: int | None
    min_output_tokens: int | None

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
        cast_logprobs_to_fp32: bool = True,
        quantization_config: BitsAndBytesConfig | None = None,
        activation_offload: bool = False,
        use_sequence_packing: bool = False,
        lora_target_scope: str | None = None,
        chunk_rows: int | None = None,
        vllm_importance_sampling_correction: bool = True,
        vllm_importance_sampling_cap: float = 2.0,
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
                "r=16, lora_alpha=32, target_modules='all-linear', task_type='CAUSAL_LM', lora_dropout=0.05."
                "To use a different LoRA configuration, please pass lora_config to the constructor.",
                stacklevel=2,
            )
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
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

        # ``use_memory_efficient_params`` offloads the trainer's own base to CPU
        # during rollout (and brings it back for the training step) so the
        # colocated rollout engine and the trainer never both hold a base on the
        # GPU. It is meaningful only for colocated vLLM; inert otherwise.
        if not use_vllm:
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
        if quantization_config is not None:
            # Both the (always-on) fused-linear-logprob path and the Liger path
            # run the lm_head matmul outside the model's quantized forward, so
            # the lm_head weight must stay unquantized to keep that matmul exact.
            skip = list(
                getattr(quantization_config, "llm_int8_skip_modules", None) or []
            )
            if "lm_head" not in skip:
                skip.append("lm_head")
                quantization_config.llm_int8_skip_modules = skip
        self.quantization_config = quantization_config
        self.activation_offload = activation_offload
        self.use_sequence_packing = bool(use_sequence_packing)
        self.lora_target_scope = lora_target_scope
        if isinstance(model_config, dict):
            # ``lora_target_scope`` is AgileRL-only; don't let it reach
            # ``from_pretrained``.
            model_dict = {
                k: v for k, v in dict(model_config).items() if k != "lora_target_scope"
            }
            if quantization_config is not None:
                model_dict.setdefault("quantization_config", quantization_config)
            model_config = model_dict
        elif model_config is None and quantization_config is not None:
            model_config = {"quantization_config": quantization_config}
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
                    if self.accelerator.is_main_process:
                        warnings.warn(
                            "Argument 'max_grad_norm' will overwrite the equivalent value set for 'gradient_clipping' in the deepspeed config.",
                            stacklevel=2,
                        )
                    ds_config["gradient_clipping"] = max_grad_norm
                if (
                    cosine_lr_schedule_config is not None
                    and self.accelerator.is_main_process
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
        self.cast_logprobs_to_fp32 = cast_logprobs_to_fp32
        if chunk_rows is not None and chunk_rows <= 0:
            msg = f"chunk_rows must be a positive int or None, got {chunk_rows}."
            raise ValueError(msg)
        self.chunk_rows = chunk_rows
        # vLLM sampling-mismatch correction (truncated importance sampling).
        # The rollout is drawn from vLLM but the loss treats the trainer's
        # recomputed ``old_log_probs`` as the behaviour policy; the two differ
        # because of engine/precision/LoRA/quantisation kernel differences even
        # when weights are shared. ``correction`` opts in to capturing vLLM's
        if vllm_importance_sampling_cap <= 0:
            msg = "vllm_importance_sampling_cap must be > 0."
            raise ValueError(msg)
        self.vllm_importance_sampling_correction = bool(
            vllm_importance_sampling_correction
        )
        self.vllm_importance_sampling_cap = float(vllm_importance_sampling_cap)
        # Kept on even when use_vllm=False: decoupled rollouts still sample
        # from a separate vLLM engine.
        self._is_correction_liger_warned = False
        # Warn-once flag for the canonical Liger + non-token importance-sampling
        # "not memory-bounded" warning (see :meth:`_warn_liger_non_token_is`).
        self._liger_non_token_warned = False
        # Warn-once flag for reference-update requests without a separate
        # reference adapter (the implicit base reference cannot move).
        self._frozen_reference_warned = False

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
        self._vllm_awake = self.use_vllm and not (
            self.vllm_config is not None and self.vllm_config.sleep_mode
        )
        self._vllm_moved = False
        self._vllm_lora_loaded = False
        self._vllm_lora_staging_dir: Path | None = None
        self._vllm_lora_staging_dir_is_temp = True
        self._vllm_rollout_lora_request: Any | None = None
        # Colocated vLLM (use_vllm=True) runs the rollout engine and the HF
        # trainer in one process. Each holds its OWN base: vLLM cycles its base
        # CPU<->GPU via native sleep/wake (vLLM >= 0.22 round-trips dense and
        # bnb 4-bit losslessly), and the trainer base is offloaded to CPU during
        # rollout (use_memory_efficient_params) so the two never coexist on the
        # GPU. Only LoRA adapters are synced to vLLM per rollout. The in-process
        # external_launcher engine is single-GPU, so tensor parallelism is not
        # yet available when colocated (use a non-colocated / async rollout for
        # TP today). NOTE: colocated tensor-parallel support is planned.
        if self.use_vllm and self.vllm_config is not None:
            tp = getattr(self.vllm_config, "tensor_parallel_size", 1)
            if tp != 1:
                msg = (
                    "Colocated vLLM requires tensor_parallel_size==1 (the "
                    f"in-process external_launcher engine is single-GPU), got "
                    f"{tp}. Use a non-colocated / async rollout for "
                    "tensor-parallel generation (colocated TP support is planned)."
                )
                raise ValueError(msg)
        self.rng = np.random.RandomState(seed)
        self.metrics = AgentMetrics()

    def preprocess_observation(self, observation: TorchObsType) -> TorchObsType:
        """Preprocess observations (dummy) for forward pass through neural network.

        :param observation: Observations of environment
        :type observation: torch.Tensor[float] or dict[str, torch.Tensor[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or tuple[torch.Tensor[float], ...]
        """
        # Dummy pass-through: LLM observations are already batched tensors.
        return observation

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

        **Plain (no accelerator):**

        - ``lora_only=T, save_optimizer=T`` -- PEFT adapter dirs on disk +
          optimizer state in ``attributes.pt``
        - ``lora_only=T, save_optimizer=F`` -- PEFT adapter dirs only
        - ``lora_only=F, save_optimizer=T`` -- full actor state_dict +
          optimizer state in ``attributes.pt``
        - ``lora_only=F, save_optimizer=F`` -- full actor state_dict in
          ``attributes.pt``

        **DeepSpeed:**

        - ``lora_only=T, save_optimizer=T`` -- engine tag dir (frozen params
          excluded) + PEFT adapter dirs
        - ``lora_only=T, save_optimizer=F`` -- PEFT adapter dirs only
        - ``lora_only=F, save_optimizer=T`` -- engine tag dir (frozen params
          included)
        - ``lora_only=F, save_optimizer=F`` -- gathered (ZeRO-3 aware) actor
          state_dict injected into ``attributes.pt``

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

        Path(path).mkdir(parents=True, exist_ok=True)

        # omit_actor_info: actor state goes into attributes.pt only when we
        # want a full-model torch save on the plain (non-deepspeed) path.
        #   * lora_only=True  → adapter weights saved via PEFT on disk; no actor in attrs.pt.
        #   * deepspeed        → actor state either lives in the engine's tag dir
        #                        (save_optimizer=True) or is gathered and injected
        #                        via the manual state_dict path below (F, F).
        #   * plain + lora_only=False → full state_dict round-trips through attrs.pt.
        omit_actor_info = lora_only or self.accelerator is not None
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
        load_optimizer: bool = False,
        overwrite_reference_adapter: bool = False,
        overwrite_critic_adapter: bool = True,
    ) -> None:
        """Load adapter weights and algorithm state from a checkpoint directory.

        Adapter roles restored on load:

          * ``actor``     — the trained policy. Always loaded.
          * ``reference`` — the fixed policy used for KL / comparison. The
            checkpoint's ``actor`` adapter is copied onto ``reference`` so
            that SFT -> DPO -> GRPO chains work out of the box: the stage-N
            actor becomes the stage-N+1 reference.
          * ``critic``    — optional value head. Loaded from disk if a
            ``critic/`` adapter is present, else copied from ``actor``, else
            left as the live fresh LoRA init.

        The checkpoint's LoRA config must match the live algorithm's config;
        a mismatch raises ``ValueError`` (re-create the agent with the
        checkpoint's LoRA config to load it).

        **No DeepSpeed:**

        - ``lora_only=T, load_optimizer=T`` -- PEFT adapter load + optimizer
          state from ``attributes.pt``
        - ``lora_only=T, load_optimizer=F`` -- PEFT adapter load only
        - ``lora_only=F, load_optimizer=T`` -- torch load of actor +
          optimizer from ``attributes.pt``
        - ``lora_only=F, load_optimizer=F`` -- torch load of actor only

        **DeepSpeed:**

        - ``lora_only=T, load_optimizer=T`` -- DeepSpeed engine load from
          ``<path>/save_checkpoint``
        - ``lora_only=T, load_optimizer=F`` -- PEFT adapter load
        - ``lora_only=F, load_optimizer=T`` -- DeepSpeed engine load from
          ``<path>/save_checkpoint``
        - ``lora_only=F, load_optimizer=F`` -- ``actor.load_state_dict(...)``
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
                # DeepSpeed restore resumes actor/optimizer shards. For LoRA-only
                # checkpoints also load adapter dirs so reference/critic adapters
                # are refreshed from PEFT artifacts.
                if lora_only:
                    self._load_model_checkpoint(
                        path,
                        overwrite_reference_adapter,
                        overwrite_critic_adapter,
                    )
            elif lora_only:
                self._load_model_checkpoint(
                    path,
                    overwrite_reference_adapter,
                    overwrite_critic_adapter,
                )
            else:
                actor_state_dict = (
                    checkpoint.get("network_info", {})
                    .get("modules", {})
                    .get("actor_state_dict")
                )
                if actor_state_dict is None:
                    # DeepSpeed full-model checkpoints saved with
                    # save_optimizer=True persist module weights in the
                    # save_checkpoint tag directory (not attributes.pt).
                    self._load_distributed_actor(
                        path,
                        tag="save_checkpoint",
                        load_optimizer_states=False,
                        load_lr_scheduler_states=False,
                    )
                else:
                    model_ref = self._get_unwrapped_actor()
                    with gather_if_zero3(self.zero_stage, list(model_ref.parameters())):
                        model_ref.load_state_dict(actor_state_dict)

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
            if lora_only:
                self._load_model_checkpoint(
                    path,
                    overwrite_reference_adapter,
                    overwrite_critic_adapter,
                )
            # ``super().load_checkpoint`` restores every attribute from the
            # checkpoint, which would clobber the live ``lora_config`` /
            # ``selected_adapters``. Stash and restore, mirroring the deepspeed
            # branch's ``_restore_checkpoint_attributes`` skip-list.
            live_lora_config = self.lora_config
            live_selected_adapters = self.selected_adapters
            super().load_checkpoint(path + "/attributes.pt")
            self.lora_config = live_lora_config
            self.selected_adapters = live_selected_adapters

        if "lr_scheduler" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def _load_model_checkpoint(
        self,
        path: str,
        overwrite_reference_adapter: bool = False,
        overwrite_critic_adapter: bool = True,
    ) -> None:
        """Restore LoRA adapter weights from a checkpoint directory.

        The checkpoint's LoRA config must match the live algorithm's; a mismatch
        (e.g. a different rank) raises ``ValueError``. Reference and Critic
        LoRA adapters in the checkpoint can be overwritten by the Actor using the
        ``overwrite_reference_adapter`` and ``overwrite_critic_adapter`` flags.

        :param path: Checkpoint directory path.
        :type path: str
        :param overwrite_reference_adapter: If ``True`` do not overwrite the live reference
            adapter. Defaults to ``False``.
        :type overwrite_reference_adapter: bool
        """
        ckpt_lora_config = self._load_checkpoint_lora_config(path)
        if ckpt_lora_config is not None:
            if self.lora_config is None or self._lora_configs_equivalent(
                self.lora_config, ckpt_lora_config
            ):
                self.lora_config = ckpt_lora_config
            else:
                raise ValueError(
                    self._format_lora_config_mismatch_error(
                        self.lora_config, ckpt_lora_config
                    )
                )

        for adapter in self.selected_adapters:
            if (Path(path) / adapter).exists():
                self._load_adapter_weights(path, adapter)

        if "reference" in self.selected_adapters and overwrite_reference_adapter:
            self._copy_adapter_weights(
                source_adapter="actor", target_adapter="reference"
            )

        if "critic" in self.selected_adapters and overwrite_critic_adapter:
            # Always overwrite the critic
            self._copy_adapter_weights(source_adapter="actor", target_adapter="critic")

        # The value head (PPO's ``v_head`` Linear) is a non-LoRA module saved
        # alongside the adapters; the adapter load above never touches it.
        if self.use_value_head:
            self._restore_value_head(path)

    def _restore_value_head(self, path: str) -> None:
        """Restore the ``v_head`` weights saved next to the LoRA adapters.

        ``AutoModelForCausalLMWithValueHead.save_pretrained`` writes the value
        head into ``pytorch_model.bin`` (the ``v_head.*`` keys of its combined
        state dict), but the LoRA-adapter load path never reads it back, so the
        value head would otherwise stay at its fresh init after
        :meth:`load_checkpoint`. Mirrors what ``from_pretrained`` does on
        construction (``post_init``), for the load-into-existing-agent path.

        :param path: Checkpoint directory written by :meth:`save_checkpoint`.
        :type path: str
        """
        wrapper = self._get_unwrapped_actor()
        loader = getattr(type(wrapper), "_maybe_load_resume_state_dict", None)
        if loader is None or not hasattr(wrapper, "post_init"):
            return
        resume_sd = loader(path)
        if resume_sd is not None and any(k.startswith("v_head.") for k in resume_sd):
            wrapper.post_init(resume_sd)

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
            is_llm_optimizer=True,
            lr_name="lr" if self.lr_critic is None else ("lr_actor", "lr_critic"),
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: DeviceType = "cpu",
        accelerator: Accelerator | None = None,
    ) -> NoReturn:
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

    @classmethod
    def population(
        cls,
        size: int,
        accelerator: Accelerator | None = None,
        device: DeviceType = "cpu",
        resume_from_checkpoint: str | None = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Create a population of LLM algorithms.

        Builds agent 0 fully (loading the model from disk), then clones the actor
        network for agents 1..N using :func:`clone_llm`. Each agent beyond the
        first receives a fresh ``Accelerator`` instance to avoid sharing the same
        DeepSpeed distributed context.

        :param size: The size of the population.
        :type size: int
        :param accelerator: HuggingFace ``Accelerator`` instance for agent 0.
        :type accelerator: Accelerator | None
        :param device: Torch device. Defaults to ``"cpu"``.
        :type device: DeviceType
        :param resume_from_checkpoint: Path to checkpoint to resume from.
        :type resume_from_checkpoint: str | None
        :param kwargs: Additional keyword arguments to pass to the algorithm constructor.
        :type kwargs: Any
        :return: A list of LLM algorithms.
        :rtype: list[LLMAlgorithm]
        """
        agent_0 = cls(index=0, accelerator=accelerator, device=device, **kwargs)
        if resume_from_checkpoint is not None:
            agent_0.load_checkpoint(resume_from_checkpoint)
            agent_0.index = 0

        population: list[Self] = [agent_0]
        for i in range(1, size):
            agent_accelerator = Accelerator() if accelerator is not None else None
            cloned_actor = clone_llm(
                agent_0.actor,
                zero_stage=0,
                state_dict=(
                    agent_0.actor.state_dict()
                    if accelerator is None
                    else get_state_dict(agent_0.actor)
                ),
            )
            clone_kwargs = dict(kwargs)
            clone_kwargs.pop("actor_network", None)
            agent = cls(
                index=i,
                accelerator=agent_accelerator,
                device=device,
                actor_network=cloned_actor,
                **clone_kwargs,
            )
            if resume_from_checkpoint is not None:
                agent.load_checkpoint(resume_from_checkpoint)
                agent.index = i
            population.append(agent)

        return population

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
        staging_dir = getattr(self, "_vllm_lora_staging_dir", None)
        if (
            staging_dir is not None
            and getattr(self, "_vllm_lora_staging_dir_is_temp", True)
            and staging_dir.is_dir()
        ):
            # Only remove staging dirs we created; a user-configured
            # ``VLLMConfig.lora_staging_dir`` is left in place.
            shutil.rmtree(staging_dir, ignore_errors=True)
        self._vllm_lora_staging_dir = None
        self._vllm_lora_loaded = False
        self._vllm_rollout_lora_request = None
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

    def _uses_quantized_clone_rebuild(self) -> bool:
        """Whether clone should rebuild the actor from pretrained + BitsAndBytes.

        QLoRA / bitsandbytes bases store packed ``Params4bit`` tensors. A dense
        ``clone_llm`` shell cannot load those shapes from a DeepSpeed checkpoint,
        so quantized clones reload the base via ``from_pretrained`` and transfer
        only adapter (and optimizer) state.
        """
        return self.quantization_config is not None

    def _save_clone_adapter_weights(self, work_dir: str) -> None:
        """Persist PEFT adapters for a quantized rebuild-from-pretrained clone.

        Used when ZeRO-2/3 DeepSpeed sharding is not available (or not needed)
        to move adapter weights onto a freshly loaded quantized base.

        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        adapter_dir = f"{work_dir}/adapters"
        model_ref = self._get_unwrapped_actor()
        with gather_if_zero3(self.zero_stage, list(model_ref.parameters())):
            model_ref.save_pretrained(
                save_directory=adapter_dir,
                selected_adapters=self.selected_adapters,
                is_main_process=self.accelerator is None
                or self.accelerator.is_main_process,
            )
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _load_clone_adapter_weights(self, work_dir: str) -> None:
        """Load PEFT adapters saved by :meth:`_save_clone_adapter_weights`.

        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        adapter_dir = f"{work_dir}/adapters"
        for adapter_name in self.selected_adapters:
            self._load_adapter_weights(adapter_dir, adapter_name)

    def _save_clone_distributed_actor_state(self, work_dir: str) -> None:
        """Save distributed actor state for ZeRO-2/3 clone workflows.

        Quantized clones also write PEFT adapters when ZeRO stage is below 2 so
        the rebuild-from-pretrained path can restore LoRA weights without a
        DeepSpeed module load of packed nf4 base tensors.

        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        quant_rebuild = self._uses_quantized_clone_rebuild()
        if quant_rebuild and (
            self.accelerator is None or self.zero_stage is None or self.zero_stage < 2
        ):
            self._save_clone_adapter_weights(work_dir)

        if self.accelerator is None or self.zero_stage is None or self.zero_stage < 2:
            return

        self.accelerator.wait_for_everyone()
        # Quantized: exclude frozen base so DeepSpeed never round-trips packed
        # Params4bit into a freshly loaded nf4 shell (adapters + opt only).
        self._save_distributed_actor(
            f"{work_dir}/agent_{self.index}",
            lora_only=quant_rebuild,
        )
        self.accelerator.wait_for_everyone()

    def _create_clone_instance(self) -> Self:
        """Instantiate a clone with cloned actor weights and runtime args.

        Quantized clones pass ``actor_network=None`` so init reloads the base
        with ``BitsAndBytesConfig`` and re-attaches adapters; adapter weights
        are restored after ``wrap_models`` via DeepSpeed (ZeRO≥2) or PEFT files.

        :return: Newly constructed clone instance.
        :rtype: Self
        """
        input_args = EvolvableAlgorithm.inspect_attributes(
            self,
            input_args_only=True,
        )
        input_args["wrap"] = False
        input_args["clone"] = True
        if self._uses_quantized_clone_rebuild():
            # Rebuild via from_pretrained + quantization_config; adapters are
            # attached because actor_network is None (see _setup_actors).
            input_args["actor_network"] = None
            input_args["model_name"] = self.pretrained_model_name_or_path
        else:
            input_args["actor_network"] = self._clone_actor_network()
        input_args["accelerator"] = (
            Accelerator() if self.accelerator is not None else None
        )
        return type(self)(**input_args)

    def _clone_actor_network(self) -> Any:  # noqa: ANN401 -- returns a heterogeneous HF/PEFT/value-head actor wrapper
        """Clone actor network while preserving value-head state when enabled.

        :return: Cloned actor network (HF/PEFT/value-head wrapper) suitable
            for clone instantiation.
        :rtype: Any
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
        sleep_mode = bool(
            self.use_vllm
            and self.vllm_config is not None
            and self.vllm_config.sleep_mode
        )
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
            if sleep_mode:
                # CuMem is process-global: transfer the single sleep-mode engine
                # to the clone. Tournament selection cleans up the parent next.
                clone.llm = original_llm
                self.llm = None
                for attr in (
                    "_vllm_awake",
                    "_vllm_moved",
                    "_vllm_lora_loaded",
                    "_vllm_lora_staging_dir",
                    "_vllm_lora_staging_dir_is_temp",
                    "_vllm_rollout_lora_request",
                    "_vllm_rollout_adapter",
                    "tp_group",
                ):
                    if hasattr(self, attr):
                        setattr(clone, attr, getattr(self, attr))
                # Prevent parent ``clean_up`` from deleting the staging dir the
                # clone still owns.
                self._vllm_lora_staging_dir = None
                self._vllm_lora_loaded = False
                self._vllm_rollout_lora_request = None
            else:
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

        Quantized clones without ZeRO≥2 restore adapters from the PEFT files
        written by :meth:`_save_clone_distributed_actor_state`.

        :param clone: Clone instance receiving distributed actor state.
        :type clone: Self
        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        if self.zero_stage is not None and self.zero_stage >= 2:
            assert clone.accelerator is not None  # ZeRO >= 2 implies an accelerator
            clone.accelerator.wait_for_everyone()
            clone._load_distributed_actor(f"{work_dir}/agent_{self.index}")
            clone.accelerator.wait_for_everyone()
        elif self._uses_quantized_clone_rebuild():
            clone._load_clone_adapter_weights(work_dir)
            if self.use_value_head:
                clone_actor = clone._get_unwrapped_actor()
                parent_actor = self._get_unwrapped_actor()
                clone_actor.v_head.load_state_dict(parent_actor.v_head.state_dict())
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
        elif self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    @staticmethod
    def update_lr(
        optimizer: torch.optim.Optimizer,  # Deepspeed optimizers are subclasses of torch.optim.Optimizer
        lr: float | tuple[float, float],
        accelerator: Accelerator | None = None,
        scheduler_config: CosineLRScheduleConfig | None = None,
    ) -> tuple[Accelerator | None, SequentialLR | None]:
        """Update the learning rate of the optimizer.

        :param optimizer: Optimizer
        :type optimizer: Optimizer
        :param lr: Learning rate value, or actor/critic pair.
        :type lr: float | tuple[float, float]
        :param accelerator: Accelerator
        :type accelerator: Accelerator | None
        :param scheduler_config: Scheduler configuration
        :type scheduler_config: CosineLRScheduleConfig | None

        :return: Tuple of accelerator and scheduler
        :return: Accelerator
        :rtype: tuple[Accelerator | None, SequentialLR | None]
        """
        if isinstance(lr, tuple):
            lr_actor, lr_critic = lr
            lr = lr_actor
        else:
            lr_critic = None

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
        """Update the reference policy when the tracker advances past the stored value.

        Base weights are immutable in AgileRL's LoRA-only training: with
        ``use_separate_reference_adapter=True`` the actor adapter is copied onto
        the ``reference`` adapter; without one the implicit reference (the base
        model with adapters disabled) cannot move, so the update request is
        acknowledged with a one-time warning and the KL anchor stays the initial
        policy.

        :param reference_update_tracker: The reference policy update tracker
        :type reference_update_tracker: int
        """
        assert reference_update_tracker >= self.reference_update_tracker, (
            "Reference policy update tracker should be greater than or equal to the current reference policy update tracker."
        )
        if reference_update_tracker > self.reference_update_tracker:
            if self.use_separate_reference_adapter:
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                self._copy_adapter_weights(
                    source_adapter="actor", target_adapter="reference"
                )
            elif not self._frozen_reference_warned:
                warnings.warn(
                    "A reference-policy update was requested but "
                    "use_separate_reference_adapter is False, so the reference "
                    "stays the initial base policy. Set "
                    "use_separate_reference_adapter=True for an updating "
                    "reference.",
                    stacklevel=2,
                    category=UserWarning,
                )
                self._frozen_reference_warned = True
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
        if adapter_name == "reference":
            if self.use_separate_reference_adapter:
                peft_model.set_adapter("reference")
                for name, param in self.actor.named_parameters():
                    if param is not None and "reference" in name:
                        param.requires_grad = False
            else:
                peft_model.base_model.disable_adapter_layers()
        else:
            if self.use_separate_reference_adapter:
                peft_model.set_adapter(adapter_name)
            else:
                peft_model.base_model.enable_adapter_layers()
        self._restore_adapter_trainability(["actor", "critic"])

    @contextmanager
    def select_adapter(self, adapter_name: str) -> Generator[None, None, None]:
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
            # Keep reference adapter frozen in DeepSpeed checkpoints so frozen
            # param fragments are emitted consistently on save/load roundtrips.
            trainable_adapters = [
                name for name in self.selected_adapters if name != "reference"
            ]
            self._restore_adapter_trainability(trainable_adapters)
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
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
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
                    load_optimizer_states=load_optimizer_states,
                    load_lr_scheduler_states=load_lr_scheduler_states,
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

    @staticmethod
    def _position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """Left-padding-safe ``position_ids`` from an attention mask.

        Cumulative real-token count minus one, with padded positions pinned to
        ``1`` so the rotary embedding sees a valid (ignored) index.

        :param mask: ``(B, T)`` attention mask (1 = real token, 0 = pad).
        :type mask: torch.Tensor
        :return: ``(B, T)`` position ids in ``long``.
        :rtype: torch.Tensor
        """
        position_ids = mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask=(mask == 0), value=1)
        return position_ids

    def _fused_logprob_fn_and_head(
        self,
    ) -> tuple[Callable, torch.Tensor, torch.Tensor | None]:
        """Resolve the fused per-token-logprob fn and the lm_head weight/bias.

        Fused-linear-logprob path (the only path): the lm_head is identity-
        patched for the forward (which returns the last hidden state), then
        per-token logprobs are computed via a chunked matmul over the lm_head
        weight, never materializing ``(B, T, V)``. Under grad the matmul is
        routed through a gradient-checkpointed autograd Function so the backward
        stays bounded too; under no_grad the lighter static method is used.

        :return: ``(fused_fn, lm_head_weight, lm_head_bias)``.
        :rtype: tuple[Callable, torch.Tensor, torch.Tensor | None]
        """
        fused_fn = (
            LLMAlgorithm._logprobs_from_hidden_fused_grad
            if torch.is_grad_enabled()
            else LLMAlgorithm._logprobs_from_hidden_fused
        )
        lm_head = self._get_lm_head()
        return fused_fn, lm_head.weight, lm_head.bias

    def _warn_liger_non_token_is(
        self,
        level: str,
        algo_name: str,
        *,
        once_attr: str = "_liger_non_token_warned",
    ) -> None:
        """Warn once that Liger + non-token importance sampling is not memory-bounded.

        The combination (``use_liger_loss=True`` with a turn-/trajectory-level
        importance-sampling ``level``) is permitted but unbounded in memory: the
        token-flatten trick only applies at token level, so the fused kernel
        processes one whole sequence per chunk. Guards on a per-instance flag
        (``once_attr``) so repeated calls emit the canonical message at most once.

        :param level: the importance-sampling level (e.g. ``"turn"``/``"trajectory"``).
        :type level: str
        :param algo_name: human-readable algorithm name for the message prefix.
        :type algo_name: str
        :param once_attr: per-instance attribute used to suppress duplicates.
        :type once_attr: str, optional
        """
        if getattr(self, once_attr, False):
            return
        warnings.warn(
            f"{algo_name} with use_liger_loss=True at importance_sampling_level="
            f"'{level}' is permitted but NOT memory-bounded: the fused kernel "
            "processes one whole sequence per chunk and materializes a "
            "(seq_len, vocab) logits tensor per trajectory (the token-flatten "
            "trick only applies at token level). For bounded memory set "
            "use_liger_loss=False — the standard path is always memory-bounded "
            "via the fused-linear-logprob path.",
            stacklevel=2,
        )
        setattr(self, once_attr, True)

    def _align_sampling_logprobs(
        self,
        sampling_logps: list[torch.Tensor | None] | None,
        action_masks: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, int]:
        """Scatter per-row flat vLLM logprobs onto the ``(B, T-1)`` action frame.

        ``sampling_logps`` is one 1-D tensor per row (the generated-token
        logprobs in order; single-turn = one rollout, multi-turn = concatenated
        across turns), parallel to the stacked ``completion_ids`` rows. Each is
        scattered onto the ``True`` positions of that row's action mask. Rows
        whose token count doesn't match the mask (e.g. env truncation) keep the
        ``old_log_probs`` value there, so their importance ratio is 1 (no
        correction) instead of crashing.

        :param sampling_logps: Per-row flat logprobs, or ``None``.
        :type sampling_logps: list[torch.Tensor | None] | None
        :param action_masks: ``(B, T-1)`` action-token mask.
        :type action_masks: torch.Tensor
        :param old_log_probs: ``(B, T-1)`` trainer old-policy logprobs (the
            fallback where data is missing → unit ratio).
        :type old_log_probs: torch.Tensor
        :return: ``(aligned (B, T-1) or None, n_rows_skipped)``.
        :rtype: tuple[torch.Tensor | None, int]
        """
        if not sampling_logps:
            return None, 0
        out = old_log_probs.clone()
        mask_bool = action_masks.to(torch.bool)
        n_rows = out.shape[0]
        n_skipped = 0
        for b in range(n_rows):
            flat = sampling_logps[b] if b < len(sampling_logps) else None
            if flat is None:
                n_skipped += 1
                continue
            pos = mask_bool[b].nonzero(as_tuple=True)[0]
            if pos.numel() != flat.numel():
                n_skipped += 1
                continue
            out[b, pos] = flat.to(device=out.device, dtype=out.dtype)
        return out, n_skipped

    def _sampling_mismatch_metrics(
        self,
        old_log_probs: torch.Tensor,
        sampling_log_probs: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> dict[str, float]:
        """Summarise the vLLM-vs-trainer logprob divergence over action tokens.

        ``vllm_is_delta_mean`` is ``mean|old - sampling|``; the ratio stats
        describe ``clamp(exp(old - sampling), max=cap)`` (mean, p95, and the
        fraction hitting the clamp). All are detached batch-level diagnostics,
        computed regardless of whether the correction is applied to the loss.
        """
        with torch.no_grad():
            mask = action_masks.to(torch.bool)
            mask_f = mask.to(torch.float32)
            denom = mask_f.sum().clamp(min=1.0)
            log_diff = (old_log_probs - sampling_log_probs) * mask_f
            delta_mean = (log_diff.abs().sum() / denom).item()
            ratio = torch.exp(log_diff).clamp(max=self.vllm_importance_sampling_cap)
            sel = ratio[mask]
            metrics = {"vllm_is_delta_mean": delta_mean}
            if sel.numel() > 0:
                metrics["vllm_is_ratio_mean"] = sel.mean().item()
                metrics["vllm_is_ratio_p95"] = torch.quantile(sel.float(), 0.95).item()
                metrics["vllm_is_frac_clamped"] = (
                    (sel >= self.vllm_importance_sampling_cap).float().mean().item()
                )
            return metrics

    def _aligned_sampling_logprobs_and_metrics(
        self,
        sampling_logps: list[torch.Tensor | None] | None,
        action_masks: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Align captured vLLM sampling logprobs and summarise the mismatch.

        Aligns the per-row logprobs captured at rollout onto the ``(B, T-1)``
        action frame (:meth:`_align_sampling_logprobs`), computes the
        vLLM-vs-trainer mismatch metrics whenever logprobs were captured —
        independently of whether the correction is applied to the loss — and
        warns when rows were skipped for a token-count mismatch. Shared by the
        GRPO/PPO/REINFORCE ``learn`` implementations.

        :param sampling_logps: Per-row vLLM sampling logprobs from rollout, or
            ``None`` when none were captured.
        :type sampling_logps: list[torch.Tensor | None] | None
        :param action_masks: ``(B, T-1)`` action-token mask.
        :type action_masks: torch.Tensor
        :param old_log_probs: ``(B, T-1)`` frozen-policy logprobs.
        :type old_log_probs: torch.Tensor
        :return: ``(aligned_logprobs_or_None, metrics)``.
        :rtype: tuple[torch.Tensor | None, dict[str, float]]
        """
        is_metrics: dict[str, float] = {}
        sampling_log_probs, n_skipped = self._align_sampling_logprobs(
            sampling_logps, action_masks, old_log_probs
        )
        if sampling_log_probs is not None:
            is_metrics = self._sampling_mismatch_metrics(
                old_log_probs, sampling_log_probs, action_masks
            )
            if n_skipped:
                is_metrics["vllm_is_rows_skipped"] = float(n_skipped)
                warnings.warn(
                    f"{n_skipped}/{action_masks.shape[0]} rows had a token-count "
                    "mismatch between captured vLLM logprobs and the action "
                    "mask; their importance ratio defaults to 1 (no "
                    "correction). Check rollout/trainer tokenisation if this "
                    "is large.",
                    stacklevel=2,
                )
        return sampling_log_probs, is_metrics

    def _setup_actors(
        self,
        actor_network: object | None,
        *,
        clone: bool,
    ) -> None:
        """Build the actor(s), routing through the colocated-vLLM path when enabled.

        ``clone=True`` with a pre-built PEFT ``actor_network`` reuses it (no new
        adapters). Quantized clones pass ``actor_network=None`` so the base is
        reloaded with ``BitsAndBytesConfig``; adapters must still be attached
        before the LoRA-only DeepSpeed / PEFT weight restore.
        """
        # Rebuild-from-pretrained (actor_network is None) always needs adapters,
        # including when clone=True for the quantized path.
        add_adapters = (not clone) or (actor_network is None)
        if self.use_vllm:
            self._initialize_colocated_vllm_and_actors(
                actor_network, add_adapters=add_adapters, clone=clone
            )
        else:
            self._initialize_actors(actor_network, add_adapters)

    def _initialize_colocated_vllm_and_actors(
        self,
        base_model: object | None,
        add_adapters: bool = True,
        *,
        clone: bool = False,
    ) -> None:
        """Initialize a colocated vLLM rollout engine + the HF/PEFT trainer.

        vLLM and the trainer each hold their OWN base (no zero-copy aliasing).
        Across the rollout<->train cycle vLLM round-trips its base CPU<->GPU via
        native sleep/wake (vLLM >= 0.22 restores both dense and bnb 4-bit
        losslessly) and the trainer base is offloaded to CPU during rollout
        (``use_memory_efficient_params``), so the two bases never coexist on the
        GPU. Only LoRA adapters are synced to vLLM per rollout (see
        :meth:`_move_lora_to_vllm`).

        **Init ordering is CUDA-safe.** A bitsandbytes trainer quantizes on the
        GPU during ``from_pretrained``; starting vLLM first (even slept) can
        leave the CUDA allocator in a state where the trainer's bnb device
        copies segfault. So for a fresh bnb trainer under ``sleep_mode`` the
        trainer is built first, offloaded to CPU, then vLLM starts. A dense
        trainer (or ``sleep_mode`` off, or a clone) is CUDA-safe vLLM-first.

        ``base_model`` is ``None`` for a fresh algorithm (the trainer base is
        loaded from ``pretrained_model_name_or_path``) or a fully-built,
        already-adapted actor copy for a clone (``add_adapters=False``), reused
        as-is.

        **Sleep-mode clones** do not construct a second ``LLM``: CuMem is
        process-global (one sleep-mode engine per process). The parent's engine
        is transferred in :meth:`_copy_clone_attributes` after construction.
        """
        # Sleep-mode CuMem forbids a second in-process engine. Build the
        # trainer only; ``clone()`` moves the parent's ``llm`` onto this instance.
        if clone and self.vllm_config is not None and self.vllm_config.sleep_mode:
            self.llm = None
            self._initialize_actors(base_model, add_adapters)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            return

        main = self.accelerator is None or self.accelerator.process_index == 0
        if self._trainer_should_load_before_vllm(base_model):
            if main:
                warnings.warn(
                    "colocated init: trainer-first order (bnb trainer + vLLM "
                    "sleep mode); trainer built, offloaded to CPU, then vLLM "
                    "starts. vLLM cycles its base via native sleep/wake.",
                    stacklevel=2,
                )
            self._initialize_actors(base_model, add_adapters)
            self._offload_trainer_to_cpu_for_colocated_vllm()
            self._configure_vllm()
        else:
            if main:
                warnings.warn(
                    "colocated init: vLLM-first order (dense trainer or "
                    "sleep_mode off); each side holds its own base, vLLM cycles "
                    "via native sleep/wake, trainer offloaded during rollout.",
                    stacklevel=2,
                )
            self._configure_vllm()
            self._initialize_actors(base_model, add_adapters)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _trainer_should_load_before_vllm(
        self,
        base_model: object | None,
    ) -> bool:
        """Whether the HF trainer must be built before colocated vLLM starts.

        bitsandbytes runs GPU quantization during ``from_pretrained``. Starting
        vLLM first (even in sleep mode) can leave the CUDA allocator in a state
        where subsequent device copies during the trainer's bnb load segfault,
        so a fresh quantized trainer under sleep mode is loaded first.
        """
        return (
            self.use_vllm
            and self.vllm_config is not None
            and self.vllm_config.sleep_mode
            and self.quantization_config is not None
            and base_model is None
        )

    def _offload_trainer_to_cpu_for_colocated_vllm(self) -> None:  # pragma: no cover
        """Move the HF trainer off the GPU before colocated vLLM ``LLM()`` init.

        Trainer-side bitsandbytes quantization runs on the GPU during
        ``from_pretrained`` even with ``device_map="cpu"``. Offloading after
        load keeps the trainer-first ordering (which avoids post-vLLM bnb
        segfaults) while freeing the GPU for vLLM startup (profile / compile /
        CUDA-graph capture).
        """
        if not getattr(self, "actor", None):
            warnings.warn(
                "colocated init: trainer CPU offload skipped (actor not set)",
                stacklevel=2,
            )
            return
        main = self.accelerator is None or self.accelerator.process_index == 0
        if main:
            log_cuda_memory_snapshot("colocated init: before trainer CPU offload")
        remaining_cuda_bytes = offload_colocated_trainer_from_gpu(
            self._get_unwrapped_actor()
        )
        if main:
            log_cuda_memory_snapshot("colocated init: after trainer CPU offload")
            if remaining_cuda_bytes > 0:
                warnings.warn(
                    f"colocated init: trainer still has "
                    f"{remaining_cuda_bytes / (1024**2):.2f} MiB on CUDA after "
                    "offload",
                    stacklevel=2,
                )

    def _initialize_actors(
        self,
        base_model: Any | None,  # noqa: ANN401 -- base HF model or trl-style value-head wrapper, dereferenced dynamically
        add_adapters: bool = True,
    ) -> None:
        """Initialize the actor network.

        A user-supplied :class:`~peft.PeftModel` is rejected (with
        ``add_adapters`` True): AgileRL manages its own adapters on an immutable
        base, so pass the base model instead. The clone path (``add_adapters``
        False) passes through the model unchanged.

        :param base_model: Base model
        :type base_model: PreTrainedModelProtocol
        :param add_adapters: Flag to indicate if adapters should be added to the model, defaults to True
        :type add_adapters: bool, optional
        """
        if base_model is None:
            model_config = (
                dict(self.model_config) if isinstance(self.model_config, dict) else None
            )
            # Colocated sleep mode: keep HF weights on CPU until accelerator.prepare().
            # When trainer bnb quant is enabled, _initialize_colocated_vllm_and_actors
            # loads the trainer before vLLM so bnb kernels do not run after vLLM sleep.
            if (
                self.use_vllm
                and self.vllm_config is not None
                and self.vllm_config.sleep_mode
            ):
                if model_config is None:
                    model_config = {}
                model_config.setdefault("device_map", "cpu")
            if (
                self.use_vllm
                and getattr(self, "llm", None) is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
            base_model = create_model_from_name_or_path(
                self.pretrained_model_name_or_path,
                model_config=model_config,
                add_value_head=self.use_value_head,
                use_accelerator=self.accelerator is not None,
            )

        if add_adapters:
            if isinstance(base_model, PeftModelProtocol):
                msg = (
                    "actor_network: a PeftModel was passed, but AgileRL manages its "
                    "own LoRA adapters on an immutable base model. Pass the "
                    "base model instead (merge your adapters first via PEFT's "
                    "merge_and_unload() if you want to keep their effect)."
                )
                raise ValueError(msg)
            if self.use_value_head and isinstance(
                getattr(base_model, "pretrained_model", None), PeftModelProtocol
            ):
                msg = (
                    "actor_network.pretrained_model: a PeftModel was passed, but AgileRL manages its "
                    "own LoRA adapters on an immutable base model. Pass the "
                    "base model instead (merge your adapters first via PEFT's "
                    "merge_and_unload() if you want to keep their effect)."
                )
                raise ValueError(msg)

            # The PEFT target crosses HF/PEFT wrapper types; keep it untyped.
            peft_target: Any = (
                base_model.pretrained_model if self.use_value_head else base_model
            )
            # A bitsandbytes-quantized base must go through PEFT's kbit
            # preprocessing before adapters are attached: it casts layernorms
            # (and lm_head) to fp32 for training stability and registers the
            # input-embedding hook that gradient checkpointing relies on.
            # Detected via the transformers bnb load flags so a user-supplied
            # quantized actor_network is handled even without quantization_config.
            quantized_base = (
                getattr(peft_target, "is_loaded_in_8bit", False)
                or getattr(peft_target, "is_loaded_in_4bit", False)
                or getattr(peft_target, "is_quantized", False)
            )
            if quantized_base:
                peft_target = prepare_model_for_kbit_training(
                    peft_target,
                    use_gradient_checkpointing=self.gradient_checkpointing,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            # Gemma 4 etc.: LoRA must target inner ``.linear`` inside *ClippableLinear.
            lora_config = adapt_lora_config_for_model(
                peft_target,
                self.lora_config,
                lora_target_scope=self.lora_target_scope,
            )
            self.lora_config = lora_config
            peft_target = get_peft_model(
                peft_target,
                lora_config,
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
                        peft_config=self.lora_config,
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

            # Apply Liger Kernel optimizations (fused RMSNorm, RoPE, SwiGLU,
            # CrossEntropy) to the inner causal-LM *after* PEFT wrapping.
            # PeftModel is not a PreTrainedModel, so AutoLigerKernelForCausalLM
            # cannot be used at load time; instance-level patching works on the
            # already-constructed modules instead.
            if HAS_LIGER_KERNEL:
                inner_model = (
                    peft_target.base_model.model
                    if hasattr(peft_target, "base_model")
                    else peft_target
                )
                # Skip if already patched (by us or upstream) — double-patching
                # is undefined behavior.
                already_patched = getattr(
                    inner_model, "_agilerl_liger_patched", False
                ) or any(
                    type(m).__module__.startswith("liger_kernel")
                    for m in inner_model.modules()
                )
                if already_patched:
                    logger.info(
                        "Liger Kernel patches already present on %s; skipping.",
                        type(inner_model).__name__,
                    )
                try:
                    if not already_patched:
                        _apply_liger_kernel_to_instance(model=inner_model)
                        inner_model._agilerl_liger_patched = True
                        logger.info(
                            "Liger Kernel instance-level patches applied to %s.",
                            type(inner_model).__name__,
                        )
                except (KeyError, AttributeError, TypeError):
                    logger.warning(
                        "Liger Kernel does not support %s; "
                        "falling back to stock HF modules.",
                        type(inner_model).__name__,
                    )

            # The actor crosses HF/PEFT/compile/DummyEvolvable representations
            # below, so build it in an untyped local before storing it.
            actor: Any
            if self.use_value_head:
                # The value-head wrapper is duck-typed (trl-style).
                vh_wrapper: Any = base_model
                vh_wrapper.pretrained_model = peft_target
                vh_wrapper.is_peft_model = True
                actor = vh_wrapper
            else:
                actor = peft_target
        else:
            actor = base_model

        self.actor = actor
        self.use_adapter("actor")
        patch_lora_for_fused_forward(actor)

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
                actor = compile_model(actor, self.torch_compiler)
                self.actor = actor

        if self.accelerator is None:
            actor = DummyEvolvable(module=actor, device=str(self.device))
            self.actor = actor

        # If an optimizer is defined in the deepspeed config, then the optimizer is part of the engine when
        # accelerator.prepare() is called. Since we are yet to wrap the model, we pass a dummy optimizer to the OptimizerWrapper.
        # In all other cases optim.Adam is used.
        optim_class = self._select_optim_class()

        self.optimizer = OptimizerWrapper(
            optim_class,
            networks=[self.actor],
            lr=self.lr,
            lr_critic=self.lr_critic,
            is_llm_optimizer=True,
            network_names=["actor"],
            lr_name="lr" if self.lr_critic is None else ("lr_actor", "lr_critic"),
        )

        if self.cosine_lr_schedule_config is not None:
            if self.optimizer.optimizer_cls != DummyOptimizer:
                sched_optimizer = self.optimizer._single_optimizer()
            else:
                # With a DummyOptimizer, DeepSpeed owns the real optimizer and
                # attaches it to the actor engine at wrap time.
                sched_optimizer = actor.optimizer  # pragma: no cover - needs a live DeepSpeed engine to attach actor.optimizer
            # ``actor.optimizer`` resolves through ``nn.Module.__getattr__`` to a
            # dynamic member; both branches yield a real optimizer at runtime.
            assert isinstance(sched_optimizer, torch.optim.Optimizer)
            self.lr_scheduler = create_warmup_cosine_scheduler(
                sched_optimizer,
                self.cosine_lr_schedule_config,
                1e-8,
                self.lr,
            )
        else:
            self.lr_scheduler = None

    @contextmanager
    def _amp_ctx(self) -> Generator[None, None, None]:
        """Yield a ``torch.amp.autocast`` context when running without an accelerator.

        When an ``Accelerator`` is present it already manages mixed-precision
        via its own autocast wrapper, so this is a no-op in that case.
        """
        if self.accelerator is not None:
            yield
        else:
            device_type = torch.device(self.device).type
            if device_type == "cuda" and torch.cuda.is_bf16_supported():
                with (
                    torch.amp.autocast(device_type, dtype=torch.bfloat16),
                    self._lora_input_cast_ctx(),
                ):
                    yield
            else:
                yield

    @contextmanager
    def _lora_input_cast_ctx(self) -> Generator[None, None, None]:
        """Skip PEFT's fp32 cast of every LoRA input while autocast is active."""
        if self.actor is None:
            yield
            return

        layers = get_cached_lora_layers(self.actor)
        previous = [layer.cast_input_dtype_enabled for layer in layers]
        for layer in layers:
            layer.cast_input_dtype_enabled = False
        try:
            yield
        finally:
            for layer, was_enabled in zip(layers, previous, strict=True):
                layer.cast_input_dtype_enabled = was_enabled

    @contextmanager
    def _activation_offload_ctx(self) -> Generator[None, None, None]:
        """Offload tensors saved for backward to pinned host RAM.

        When ``activation_offload`` is set, the training forward pass is run
        inside :func:`torch.autograd.graph.save_on_cpu`, so the activations
        kept between forward and backward (with gradient checkpointing, the
        checkpoint-boundary tensors) live in host memory instead of GPU
        memory. This trades PCIe bandwidth for GPU memory; the win grows with
        sequence length, which makes it the lever for long-context training.

        A no-op when offload is disabled or grads are inactive (rollout /
        reference forwards save nothing for backward). Purely trainer-side, so
        it composes with a co-located or a decoupled rollout engine alike.
        """
        if self.activation_offload and torch.is_grad_enabled():
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
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
        :rtype: tuple[torch.Tensor, torch.Tensor | None]
        """
        unwrapped = self._get_unwrapped_actor()
        total = fused_ids.shape[0]
        seq_len_out = fused_ids.shape[1] - 1

        position_ids = None
        if self.calc_position_embeddings:
            position_ids = self._position_ids_from_mask(fused_mask)

        chunks = (
            [(0, total)]
            if batch_size is None
            else [(s, min(s + batch_size, total)) for s in range(0, total, batch_size)]
        )

        fused_fn, lm_head_weight, lm_head_bias = self._fused_logprob_fn_and_head()

        def _process_chunk(
            start: int, end: int
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            set_fused_adapter_routing(unwrapped, routing[start:end])
            model_kwargs: dict = {
                "input_ids": fused_ids[start:end],
                "attention_mask": fused_mask[start:end],
                "use_cache": False,
            }
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids[start:end]

            with (
                self._patch_lm_head_to_identity(),
                self._amp_ctx(),
                self._activation_offload_ctx(),
            ):
                output = self.actor.forward(**model_kwargs)

            if isinstance(output, tuple):
                # Value-head models may return (loss, logits, value, ...); Peft/causal
                # paths may return shorter tuples — only index when present. With
                # lm_head identity-patched, output[0] is the last hidden state.
                first = output[0]
                value = output[2] if len(output) > 2 else None
            else:
                first = output.logits
                value = None
            del output

            chunk_lp = fused_fn(
                first[:, :-1],
                lm_head_weight,
                lm_head_bias,
                fused_ids[start:end, 1:],
                temperature=self.temperature,
                cast_to_fp32=self.cast_logprobs_to_fp32,
                chunk_rows=self.chunk_rows,
            )
            del first

            chunk_v = (
                value[:, :-1] if (self.use_value_head and value is not None) else None
            )
            return chunk_lp, chunk_v

        # Single-chunk fast path: skip the buffer + copy entirely.
        if len(chunks) == 1:
            return _process_chunk(0, total)

        # Multi-chunk path: pre-allocate output buffers once and write each
        # chunk in place via copy_(). Avoids holding the full list of chunk
        # tensors plus the concatenated buffer in memory at the same time
        # (which doubles peak memory in the torch.cat path).
        logprobs_out: torch.Tensor | None = None
        values_out: torch.Tensor | None = None

        for start, end in chunks:
            chunk_lp, chunk_v = _process_chunk(start, end)

            # Lazy-allocate on the first chunk so we inherit dtype/device
            # from the model output rather than guessing up front.
            if logprobs_out is None:
                logprobs_out = torch.empty(
                    (total, seq_len_out),
                    dtype=chunk_lp.dtype,
                    device=chunk_lp.device,
                )
            logprobs_out[start:end].copy_(chunk_lp)
            del chunk_lp

            if chunk_v is not None:
                if values_out is None:
                    values_out = torch.empty(
                        (total, seq_len_out),
                        dtype=chunk_v.dtype,
                        device=chunk_v.device,
                    )
                values_out[start:end].copy_(chunk_v)
                del chunk_v

        assert logprobs_out is not None  # chunks is never empty
        return logprobs_out, values_out

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
           ``unset_fused_adapter_routing`` after the backward pass.

           Callers are responsible for ensuring the model is in training
           mode and adapter trainability is restored before entering the
           minibatch loop (see ``learn()`` in ``ppo_llm.py``).

        :param ids: Token IDs ``(B, seq_len)``.
        :type ids: torch.Tensor
        :param batch_size: Unused (kept for API symmetry).
        :type batch_size: int
        :param attention_mask: Optional attention mask matching *ids*.
        :type attention_mask: torch.Tensor | None, optional
        :return: ``(actor_log_probs, critic_values)`` with shapes ``(B, seq_len-1)``;
            *critic_values* is ``None`` when no value head is used.
        :rtype: tuple[torch.Tensor, torch.Tensor | None]
        """
        B = ids.shape[0]
        if attention_mask is None:
            attention_mask = ids != self.pad_token_id

        # Packed path for the gradient forward only; no-grad passes stay padded.
        if torch.is_grad_enabled() and self._packing_mode() is not None:
            return self._fused_packed_forward(ids, attention_mask)

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
            assert values is not None  # value-head models return values
            return log_probs[:B], values[B:]
        return log_probs, None

    def _fused_packed_forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Padding-free packed variant of the gradient :meth:`_fused_forward`.

        Actor and critic see identical token ids, so the ``(B, T)`` batch is
        packed **once** into a single padding-free row, then that row is
        repeated per active adapter (``actor`` [+ ``critic``]) to form an
        ``(n_adapters, N)`` batch. Per-row fused LoRA routing applies the actor
        adapter to row 0 and the critic adapter to the last row; both rows carry
        the same per-segment ``position_ids`` (``attention_mask=None``), so
        transformers builds a block-diagonal varlen / blockmask forward — no
        cross-sequence attention, sliding windows preserved (see
        :meth:`_packing_mode`). A **single** ``model.forward`` keeps
        gradient-checkpoint recomputation consistent with one persistent routing
        (the routing is *not* cleared here — see the :meth:`_fused_forward`
        note).

        :param ids: Token IDs ``(B, seq_len)``.
        :type ids: torch.Tensor
        :param attention_mask: Mask matching *ids* (non-zero marks real tokens).
        :type attention_mask: torch.Tensor
        :return: ``(actor_log_probs, critic_values)`` each ``(B, seq_len - 1)``;
            *critic_values* is ``None`` when no value head is used.
        :rtype: tuple[torch.Tensor, torch.Tensor | None]
        """
        unwrapped = self._get_unwrapped_actor()
        packed = pack_padded_batch(ids, attention_mask)

        adapters = ["actor"] + (["critic"] if self.use_value_head else [])
        n_adapters = len(adapters)
        fused_ids = packed.input_ids.repeat(n_adapters, 1)  # (n_adapters, N)
        fused_position_ids = packed.position_ids.repeat(n_adapters, 1)
        set_fused_adapter_routing(unwrapped, adapters)

        fused_fn, lm_head_weight, lm_head_bias = self._fused_logprob_fn_and_head()
        with (
            self._patch_lm_head_to_identity(),
            self._amp_ctx(),
            self._activation_offload_ctx(),
        ):
            output = self.actor.forward(
                input_ids=fused_ids,
                position_ids=fused_position_ids,
                use_cache=False,
            )

        if isinstance(output, tuple):
            hidden = output[0]
            value = output[2] if len(output) > 2 else None
        else:
            hidden = output.logits
            value = None

        # Actor log-probs from row 0 (actor adapter): the fused matmul
        # consumes the (1, N, H) hidden + (1, N-1) next-token targets exactly as
        # the padded path does; unpack scatters back to the (B, T-1) frame and
        # drops the cross-segment boundary prediction.
        packed_lp = fused_fn(
            hidden[:1][:, :-1],
            lm_head_weight,
            lm_head_bias,
            packed.input_ids[:, 1:],
            temperature=self.temperature,
            cast_to_fp32=self.cast_logprobs_to_fp32,
            chunk_rows=self.chunk_rows,
        )
        log_probs = unpack_logprobs(packed_lp, packed)

        values = None
        if self.use_value_head and value is not None:
            # Critic values from the last row (critic adapter): per-token scalars
            # mapped back to the (B, T-1) frame (no boundary drop, see
            # unpack_values). Pad positions are zero and masked downstream.
            values = unpack_values(value[-1], packed)

        return log_probs, values

    def _fused_forward_no_grad(
        self,
        ids: torch.Tensor,
        batch_size: int,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute reference log-probs, actor log-probs, and critic values in
        one forward pass (under ``torch.no_grad``).

        The batch is tripled (reference / actor / critic rows) and routed per
        row, so the frozen base runs in a single fused pass.  When
        ``use_separate_reference_adapter`` is ``True`` the reference rows use
        the ``"reference"`` adapter; when ``False`` they are routed to PEFT's
        reserved ``"__base__"`` name, which applies no LoRA delta (the frozen
        base is the reference policy).

        This method micro-batches because no gradient checkpoint recomputation
        is involved.

        :param ids: Token IDs ``(B, seq_len)``.
        :type ids: torch.Tensor
        :param batch_size: Micro-batch size for memory-bounded iteration.
        :type batch_size: int
        :param attention_mask: Optional attention mask matching *ids*.
        :type attention_mask: torch.Tensor | None, optional
        :return: ``(reference_log_probs, actor_log_probs, critic_values)``
            each of shape ``(B, seq_len - 1)``.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        """
        B = ids.shape[0]
        if attention_mask is None:
            attention_mask = ids != self.pad_token_id

        self.actor.eval()

        with torch.inference_mode():
            reference_adapter = (
                "reference" if self.use_separate_reference_adapter else "__base__"
            )
            adapters = [reference_adapter, "actor"]
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
            unset_fused_adapter_routing(self._get_unwrapped_actor())
            ref_logprobs = log_probs[:B]
            actor_logprobs = log_probs[B : 2 * B]
            if self.use_value_head:
                assert values is not None  # value-head models return values
                critic_values = values[2 * B :]
            else:
                critic_values = None

        return ref_logprobs, actor_logprobs, critic_values

    def _resolve_attn_implementation(self) -> str | None:
        """Best-effort read of the actor's active attention backend."""
        try:
            cfg = getattr(self._get_unwrapped_actor(), "config", None)
            impl = getattr(cfg, "_attn_implementation", None)
            if impl is not None:
                return impl
        except Exception:
            # Best-effort probe: introspecting a wrapped actor's config can fail
            # in exotic setups (accelerate/DeepSpeed wrappers, slotted modules).
            # Non-fatal — fall back to model_config / None below.
            logger.debug("attn-implementation probe failed", exc_info=True)
        model_config = getattr(self, "model_config", None)
        if isinstance(model_config, dict):
            return model_config.get("attn_implementation")
        return None

    def _packing_mode(self) -> str | None:
        """Resolve how (if at all) to pack the forward, given the backend.

        Packing passes the model only per-sequence ``position_ids`` (no mask);
        transformers detects the packed format and AND-composes a block-diagonal
        constraint onto each layer's native (causal / sliding-window) mask. We
        only gate on backends where that composed mask stays sparse:

        * ``"varlen"`` — FlashAttention-2 turns the packed ``position_ids`` into
          ``cu_seqlens`` (+ per-layer ``window_size``); no mask is materialized.
          Memory/throughput-optimal and handles dynamic shapes with no recompile.
        * ``"blockmask"`` — FlexAttention builds a sparse block-diagonal
          ``BlockMask`` (active tiles + compiled ``mask_mod``, windowed on
          sliding layers). Needs no ``flash_attn`` build, but recompiles per
          packed length.

        Both preserve sliding-window attention per layer, so packing is correct
        for SWA models (e.g. gemma), not just full-attention ones. Dense backends
        (SDPA / eager) would build a *dense* ``O(N^2)`` block-diagonal mask —
        correct but defeating the memory win — so packing is **not** enabled on
        them and falls back to padding (warning once). Returns ``None`` when
        packing is off or the backend is unsupported.
        """
        if not getattr(self, "use_sequence_packing", False):
            return None
        impl = self._resolve_attn_implementation()
        if impl == "flash_attention_2":
            return "varlen"
        if impl == "flex_attention":
            return "blockmask"
        if not getattr(self, "_packing_backend_warned", False):
            warnings.warn(
                "use_sequence_packing=True needs a varlen/block-sparse attention "
                "backend (flash_attention_2 or flex_attention); got "
                f"{impl!r}, which has no sparse path. Falling back to the padded "
                "forward (no packing).",
                stacklevel=2,
            )
            self._packing_backend_warned = True
        return None

    def _sequence_packing_active(self) -> bool:
        """``True`` when sequence packing should drive this forward."""
        return self._packing_mode() is not None

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
        grad_enabled = torch.is_grad_enabled()
        with self.select_adapter("reference" if use_reference else "actor"):
            self.actor.train(mode=not eval_mode)
            num_samples = ids.shape[0]
            if attention_mask is None:
                # TODO this calc is avoided when using PreferenceGym, need to make ReasoningGym do the same
                attention_mask = ids != self.pad_token_id
            if self.calc_position_embeddings:
                position_ids = self._position_ids_from_mask(attention_mask)

            fused_fn, lm_head_weight, lm_head_bias = self._fused_logprob_fn_and_head()
            # Pack only the gradient forward (the per-epoch hot path). The
            # no-grad old/reference passes stay padded so they are mutually
            # consistent; the packed-vs-padded gap for the current policy is the
            # tiny varlen/dense-vs-padded numerical difference.
            packing_mode = self._packing_mode() if grad_enabled else None

            # Split the sample into batches
            log_probs = []
            for batch in range(0, num_samples, batch_size):
                end_idx = min((batch + batch_size), num_samples)
                batch_ids = ids[batch:end_idx, :]
                batch_attention_mask = attention_mask[batch:end_idx, :]

                packed = None
                if packing_mode is not None:
                    packed = pack_padded_batch(batch_ids, batch_attention_mask)

                if packed is not None:
                    # Per-sequence position_ids (no mask): transformers detects
                    # the packed format and keeps sequences attention-isolated
                    # per layer (sliding-window safe).
                    batch_model_kwargs = {
                        "input_ids": packed.input_ids,
                        "position_ids": packed.position_ids,
                        "use_cache": False,
                    }
                else:
                    batch_model_kwargs = {
                        "input_ids": batch_ids,
                        "attention_mask": batch_attention_mask,
                        "use_cache": False,
                    }
                    if self.calc_position_embeddings:
                        batch_model_kwargs["position_ids"] = position_ids[
                            batch:end_idx, :
                        ]

                with (
                    self._patch_lm_head_to_identity(),
                    self._amp_ctx(),
                    self._activation_offload_ctx(),
                ):
                    output = self.actor.forward(**batch_model_kwargs)
                first = output[0] if isinstance(output, tuple) else output.logits

                if packed is not None:
                    packed_lp = fused_fn(
                        first[:, :-1],
                        lm_head_weight,
                        lm_head_bias,
                        packed.input_ids[:, 1:],
                        temperature=self.temperature,
                        cast_to_fp32=self.cast_logprobs_to_fp32,
                        chunk_rows=self.chunk_rows,
                    )
                    # Map back to the dense (mb, T-1) frame so the loss path is
                    # unchanged; cross-segment boundary predictions are dropped.
                    # unpack_logprobs reshapes the packed logprobs internally.
                    log_prob = unpack_logprobs(packed_lp, packed)
                else:
                    log_prob = fused_fn(
                        first[:, :-1],
                        lm_head_weight,
                        lm_head_bias,
                        batch_ids[:, 1:],
                        temperature=self.temperature,
                        cast_to_fp32=self.cast_logprobs_to_fp32,
                        chunk_rows=self.chunk_rows,
                    )

                first = None
                batch_model_kwargs = None
                log_probs.append(log_prob)
        return torch.cat(log_probs, dim=0)

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Perform a backward pass and optimizer step.

        :param loss: Combined loss.
        :type loss: torch.Tensor
        """
        if self._uses_deepspeed:
            assert self.accelerator is not None  # _uses_deepspeed implies one
            self.accelerator.backward(loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.lr = float(self.lr_scheduler.get_last_lr()[0])
        else:
            loss.backward()

            for group in self.optimizer.optimizer.param_groups:
                clip_grad_norm_(group["params"], self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.lr = float(self.lr_scheduler.get_last_lr()[0])

    @property
    def _peft_model(self) -> Any:  # noqa: ANN401 -- PeftModel lives at a wrapper-specific attribute; concrete type varies
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

    def _get_peft_model_for_vllm_sync(self) -> Any:  # noqa: ANN401 -- unwrapped PEFT model type varies (value-head wrapper vs bare PeftModel)
        """Unwrapped PEFT model used for vLLM weight / adapter sync."""
        model_ref = self._get_unwrapped_actor()
        return model_ref.pretrained_model if self.use_value_head else model_ref

    def _ensure_vllm_lora_staging_dir(self) -> Path:
        """Resolve (once) the dir the rollout LoRA adapter is exported to.

        The staging dir is always process-private: each rank exports its own
        adapter copy and reads it back locally, so ranks never race on shared
        files. Honours ``VLLMConfig.lora_staging_dir`` when set — e.g. a known
        path that orchestrated deployments expect the adapter under — staging
        in a ``rank_<process_index>`` subdirectory of that root when
        distributed. The dir is created (parents included) and marked
        non-temporary so ``clean_up`` never deletes it. Otherwise falls back
        to a process-private ``mkdtemp`` that ``clean_up`` removes.
        Idempotent: both the colocated init (``_configure_vllm``) and every
        adapter sync (``_move_lora_to_vllm``) call this, so the same directory
        is used throughout the agent's life.

        :return: The resolved staging directory.
        :rtype: pathlib.Path
        """
        if self._vllm_lora_staging_dir is None:
            configured = getattr(self.vllm_config, "lora_staging_dir", None)
            if configured is not None:
                staging_dir = Path(configured)
                if self.accelerator is not None and self.accelerator.num_processes > 1:
                    staging_dir = staging_dir / f"rank_{self.accelerator.process_index}"
                staging_dir.mkdir(parents=True, exist_ok=True)
                self._vllm_lora_staging_dir = staging_dir
                self._vllm_lora_staging_dir_is_temp = False
            else:
                self._vllm_lora_staging_dir = Path(
                    tempfile.mkdtemp(prefix="agilerl_vllm_lora_")
                )
                self._vllm_lora_staging_dir_is_temp = True
        return self._vllm_lora_staging_dir

    def _move_lora_to_vllm(self) -> None:
        """Export the actor LoRA adapter to disk and register it with vLLM.

        Adapter-only sync (colocated vLLM always serves LoRA): vLLM keeps its
        own base and only the LoRA delta is synced per rollout via
        ``llm_engine.add_lora``. Compatible with vLLM-side weight quantization
        (e.g. ``bitsandbytes`` for QLoRA rollouts).

        **Does not touch base weights.** vLLM owns its base across the
        native sleep/wake cycle; the trainer holds (and offloads) its own.
        """
        peft_ref = self._get_peft_model_for_vllm_sync()
        peft_ref.set_adapter(self._vllm_rollout_adapter)

        staging_dir = self._ensure_vllm_lora_staging_dir()
        with gather_if_zero3(self.zero_stage, list(peft_ref.parameters())):
            if self.lora_config is None:
                msg = "lora_config is required for vLLM LoRA adapter export."
                raise ValueError(msg)
            target_modules = self.lora_config.target_modules
            assert target_modules is not None, (
                "lora_config.target_modules is required for vLLM LoRA adapter export."
            )
            adapter_path = save_peft_adapter_for_vllm_rollout(
                peft_ref,
                staging_dir,
                self._vllm_rollout_adapter,
                target_modules=target_modules,
            )
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        # One-shot refresh of the resident slot. ``load_inplace`` forces vLLM to
        # re-read the (updated) adapter weights from disk; required from the second
        # sync onward, when the slot already holds the previous step's adapter.
        refresh_request = build_vllm_rollout_lora_request(
            adapter_path,
            load_inplace=self._vllm_lora_loaded,
        )
        lora_device = torch.device(self.device)
        if lora_device.type == "cuda":
            # Pin the CUDA context to this agent's device: vLLM's LoRA copy
            # kernels otherwise launch on the process-default device.
            with torch.cuda.device(lora_device):
                loaded = self.llm.llm_engine.add_lora(refresh_request)
        else:
            loaded = self.llm.llm_engine.add_lora(refresh_request)
        if not loaded:
            msg = (
                "vLLM failed to load LoRA adapter from "
                f"{adapter_path}. Check max_lora_rank / target module "
                "names match the trainer."
            )
            raise RuntimeError(msg)

        # The request handed to ``generate()`` must NOT carry ``load_inplace``:
        # vLLM re-evaluates active LoRAs every decode step, and load_inplace would
        # reparse the full adapter from disk each step (disk-bound rollouts). The
        # one-shot add_lora above already refreshed the resident slot.
        self._vllm_rollout_lora_request = build_vllm_rollout_lora_request(
            adapter_path,
            load_inplace=False,
        )
        self._vllm_lora_loaded = True

    def _sync_actor_to_vllm(self) -> None:
        """Sync the trainer's actor LoRA adapter into the colocated vLLM engine.

        Colocated vLLM keeps its own base and always serves LoRA via
        ``add_lora``, so only the adapter is synced — see
        :meth:`_move_lora_to_vllm`. The bases are not shared.
        Idempotent within a rollout cycle: gated by ``self._vllm_moved``, which
        the wake path clears.
        """
        if self._vllm_moved:
            return
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        self._move_lora_to_vllm()

        self.llm.reset_prefix_cache()
        self._vllm_moved = True

    def _generate_with_vllm_colocate(
        self,
        prompts: Sequence[ReasoningPrompts],
        group_size: int,
        temperature: float | None,
        capture_sampling_logps: bool = False,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None] | None
    ]:
        """Generate completions with colocated vLLM for GRPO/LLMPPO-style batches.

        Each entry in ``prompts`` is repeated ``group_size`` times so vLLM receives
        a flat list of length ``len(prompts) * group_size`` (e.g. GRPO groups).

        **Prompt dict fields:** ``input_ids`` and usually ``text`` for decoding.
        For sliding-window multi-turn prompts, optionally set ``trajectory_input_ids``,
        ``trajectory_text`` (decoded string passed to vLLM), ``stitch_prefix_ids``, and
        ``initial_prompt_len`` (required when ``stitch_prefix_ids`` is
        non-empty). Action masks use the full logical prompt length from
        ``input_ids``, not only ``trajectory_input_ids``.

        :param prompts: Length-``N`` sequence of observation mappings for this rank.
        :type prompts: Sequence[ReasoningPrompts]
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
        vllm_config = self.vllm_config
        assert vllm_config is not None, (
            "vllm_config must be configured for colocated vLLM generation."
        )

        max_token_cap = (
            self.max_output_tokens
            if self.max_output_tokens is not None
            else self.max_model_len
        )

        def _trajectory_input_ids(prompt: ReasoningPrompts) -> torch.Tensor:
            traj = prompt.get("trajectory_input_ids")
            if traj is None:
                return prompt["input_ids"]
            return traj

        def _token_prompt_for_vllm(ids: torch.Tensor) -> dict[str, list[int]]:
            return {"prompt_token_ids": ids.squeeze(0).tolist()}

        def _stitch_prefix(prompt: ReasoningPrompts, ref: torch.Tensor) -> torch.Tensor:
            st = prompt.get("stitch_prefix_ids")
            if st is None:
                return ref.new_zeros((ref.shape[0], 0))
            return st

        def _vllm_max_new_tokens(model_prompt_len: int) -> int:
            room = self.max_model_len - model_prompt_len
            if room <= 0:
                error_msg = f"Model prompt length ({model_prompt_len}) is greater than the model length ({self.max_model_len})"
                raise ValueError(error_msg)
            max_out = min(max_token_cap, room)
            if self.min_output_tokens is not None:
                max_out = max(max_out, min(self.min_output_tokens, room))
            return min(max_out, room)

        # Compute the per-prompt work once per *unique* prompt (N items),
        # then alias by reference across each group (N·G items)
        unique_ids = [_trajectory_input_ids(p) for p in prompts]
        unique_tokens = [_token_prompt_for_vllm(ids) for ids in unique_ids]
        unique_max = [_vllm_max_new_tokens(int(ids.shape[1])) for ids in unique_ids]
        unique_stitch = [
            _stitch_prefix(p, ids) for p, ids in zip(prompts, unique_ids, strict=True)
        ]

        # Replicate by reference for the flat vLLM batch. Entries within a
        # group of `group_size` are aliased references to the same tensor / dict
        # — safe because downstream use is read-only is read-only w.r.t. these objects.
        # Do not introduce in-place ops on these aliases.
        group_prompts = [p for p in prompts for _ in range(group_size)]
        prompts_ids = [ids for ids in unique_ids for _ in range(group_size)]
        token_prompts = [tp for tp in unique_tokens for _ in range(group_size)]
        max_output_tokens = [m for m in unique_max for _ in range(group_size)]
        stitch_prefixes = [sp for sp in unique_stitch for _ in range(group_size)]

        if vllm_config.tensor_parallel_size > 1:
            orig_size = len(token_prompts)

            gathered_prompts_ids: list[Any] = [
                None for _ in range(vllm_config.tensor_parallel_size)
            ]
            gathered_token_prompts: list[Any] = [
                None
            ] * vllm_config.tensor_parallel_size
            gathered_stitch_prefixes: list[Any] = [
                None
            ] * vllm_config.tensor_parallel_size
            gathered_max_output_tokens: list[Any] = [
                None
            ] * vllm_config.tensor_parallel_size

            for gathered, obj in zip(
                (
                    gathered_prompts_ids,
                    gathered_token_prompts,
                    gathered_stitch_prefixes,
                    gathered_max_output_tokens,
                ),
                (prompts_ids, token_prompts, stitch_prefixes, max_output_tokens),
                strict=True,
            ):
                torch.distributed.all_gather_object(gathered, obj, group=self.tp_group)

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

        # The windowed stitch path reorders tokens, so sampling-logprob capture
        # (for the vLLM mismatch correction) is excluded there.
        stitch_active = any(int(sp.shape[1]) > 0 for sp in stitch_prefixes)
        capture_sampling_logps = capture_sampling_logps and not stitch_active
        generation_kwargs: dict[str, Any] = {
            "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
            "repetition_penalty": self.repetition_penalty,
            "temperature": temperature,
            "top_p": self.top_p,
            "top_k": -1 if (self.top_k is None or self.top_k == 0) else self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "min_tokens": (
                0 if self.min_output_tokens is None else self.min_output_tokens
            ),
            "presence_penalty": vllm_config.presence_penalty,
            "frequency_penalty": vllm_config.frequency_penalty,
        }
        if capture_sampling_logps:
            # logprobs=0 → vLLM returns the sampled token's logprob only.
            generation_kwargs["logprobs"] = 0
        if vllm_config.stop_sequences:
            generation_kwargs["stop"] = vllm_config.stop_sequences
        sampling_params = [
            SamplingParams(**generation_kwargs, max_tokens=max_output_token)
            for max_output_token in all_max_output_tokens
        ]

        generate_kwargs: dict[str, Any] = {
            "sampling_params": sampling_params,
            "use_tqdm": False,
        }
        if self.vllm_config is not None and self._vllm_rollout_lora_request is not None:
            generate_kwargs["lora_request"] = self._vllm_rollout_lora_request

        all_outputs = self.llm.generate(all_token_prompts, **generate_kwargs)

        completion_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        # Flat per-completion lists of the sampled tokens' logprobs (one float
        # per generated token), parallel to ``completion_ids``. Empty when not
        # capturing.
        sampling_logps_flat: list[list[float]] = (
            [
                _vllm_sampled_token_logprobs(output)
                for outputs in all_outputs
                for output in outputs.outputs
            ]
            if capture_sampling_logps
            else []
        )
        if vllm_config.tensor_parallel_size > 1:
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
            if capture_sampling_logps:
                sampling_logps_flat = sampling_logps_flat[tp_slice]

        # Transfer fromn host-to-device once per unique prompt, then re-alias across the group.
        unique_prompts_ids_dev = [
            prompts_ids[group_size * i].to(self.device, non_blocking=True)
            for i in range(len(prompts))
        ]
        unique_stitch_dev = [
            stitch_prefixes[group_size * i].to(self.device, non_blocking=True)
            for i in range(len(prompts))
        ]
        prompts_ids = [ids for ids in unique_prompts_ids_dev for _ in range(group_size)]
        stitch_prefixes = [sp for sp in unique_stitch_dev for _ in range(group_size)]

        completion_ids = [
            torch.cat(
                [
                    torch.cat(
                        prompts_ids[group_size * i : group_size * (i + 1)],
                        dim=0,
                    ),
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

        sampling_logps: list[torch.Tensor | None] | None = (
            [
                torch.tensor(lp, dtype=torch.float32, device=self.device)
                for lp in sampling_logps_flat
            ]
            if capture_sampling_logps
            else None
        )

        if any(int(sp.shape[1]) > 0 for sp in stitch_prefixes):
            completion_ids = stitch_completion_after_windowed_vllm_generate(
                completion_ids,
                stitch_prefixes,
                group_prompts,
                group_size,
                prompts,
            )

        # Prompt mappings type their values as `Any`, so `input_ids` reads back
        # as `Any`; pin it to `torch.Tensor` to read the sequence-length dimension.
        num_input_tokens = [
            int(prompts[i]["input_ids"].shape[1]) for i in range(len(prompts))
        ]
        completion_masks = [
            build_completion_mask(completion_id, num_input_tokens[i], self.pad_token_id)
            for i, completion_id in enumerate(completion_ids)
        ]

        return completion_ids, completion_masks, sampling_logps

    @staticmethod
    def _logprobs_from_logits(
        logits: torch.Tensor,
        index: torch.Tensor,
        cast_to_fp32: bool = True,
        chunk_rows: int = 1,
    ) -> torch.Tensor:
        """Calculate log probabilities for previously generated token ids.

        Processes ``chunk_rows`` rows at a time so peak memory stays bounded to
        ``(chunk_rows, seq_len, vocab_size)`` rather than the full batch, avoiding
        OOM on large-vocabulary models. Default ``chunk_rows=1`` minimizes the
        fp32 workspace at the cost of more kernel launches; raise to amortize
        launch overhead when memory headroom allows.

        With ``cast_to_fp32=True``, the per-chunk reduction (``amax`` /
        ``gather`` / ``logsumexp``) runs in fp32 then casts the
        ``(B, seq_len)`` output back to *logits* dtype. Matches the precision
        of ``F.log_softmax`` over the same inputs to within the final bf16
        cast. With ``cast_to_fp32=False`` the reduction stays in *logits*
        dtype throughout — faster and lower peak (no fp32 workspace) at the
        cost of bf16-quantisation error in the reduction.

        Logits are max-centered per row before ``logsumexp``, matching
        ``F.log_softmax`` stability either way.

        :param logits: Logits of shape ``(B, seq_len, vocab_size)``.
        :type logits: torch.Tensor
        :param index: Token IDs of shape ``(B, seq_len)``.
        :type index: torch.Tensor
        :param cast_to_fp32: Promote each chunk to fp32 before the reduction.
        :type cast_to_fp32: bool
        :return: Log probabilities of the completion IDs, shape ``(B, seq_len)``.
        :rtype: torch.Tensor
        """
        orig_dtype = logits.dtype
        B = logits.shape[0]

        def _logprobs_chunk(lg: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            if cast_to_fp32:
                lg = lg.float()
            max_lg = lg.amax(dim=-1, keepdim=True)
            shifted = lg - max_lg
            target = shifted.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
            log_z = torch.logsumexp(shifted, dim=-1)
            result = target - log_z
            return result.to(orig_dtype) if cast_to_fp32 else result

        if B <= chunk_rows:
            return _logprobs_chunk(logits, index)

        per_token_logps = []
        for start in range(0, B, chunk_rows):
            end = min(start + chunk_rows, B)
            per_token_logps.append(
                _logprobs_chunk(logits[start:end], index[start:end]),
            )
        return torch.cat(per_token_logps, dim=0)

    @staticmethod
    def _resolve_fused_chunk_rows(vocab_size: int, explicit: int | None = None) -> int:
        """Rows per fused ``(chunk_rows, vocab)`` logit tile.

        Shared by the fused-linear-logprob (standard) path and the Liger
        fused-loss path so both bound their per-chunk logit workspace
        identically. A positive ``explicit`` overrides; ``None``
        auto-tunes to a ~256 MB fp32 logit workspace (fewer rows at larger
        vocab), clamped to ``[128, 4096]``.

        :param vocab_size: lm_head output dim (rows of the logit tile's V axis).
        :type vocab_size: int
        :param explicit: Explicit override, or ``None`` to auto-tune.
        :type explicit: int | None
        :return: Rows per chunk.
        :rtype: int
        """
        if explicit is not None:
            return explicit
        workspace_bytes = 256 * 1024 * 1024
        return min(max(workspace_bytes // max(1, vocab_size * 4), 128), 4096)

    @staticmethod
    def _logprobs_from_hidden_fused(
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        target_ids: torch.Tensor,
        temperature: float = 1.0,
        cast_to_fp32: bool = True,
        chunk_rows: int | None = None,
    ) -> torch.Tensor:
        """Per-token target logprobs without materializing the full ``(B, T, V)``
        logits tensor.

        Tiles flat over ``(B*T)`` with workspace bounded to ``(chunk_rows, V)``
        per iteration. Counterpart of :meth:`_logprobs_from_logits` for
        callers that hold hidden states and the lm_head separately. **No-grad
        only** — gradients won't flow to ``lm_head_weight`` from this fn. The
        gradient-aware counterpart is :meth:`_logprobs_from_hidden_fused_grad`.

        Numerical contract matches :meth:`_logprobs_from_logits` when fed
        equivalent inputs (``logits = (hidden @ Wᵀ + b) / T``): same
        ``cast_to_fp32`` semantics, same final-cast-back-to-input-dtype, same
        max-shift ``gather - logsumexp`` formulation. Default ``cast_to_fp32=True``
        keeps the two paths bit-comparable.

        :param hidden: ``(B, T, H)`` last-hidden-state.
        :type hidden: torch.Tensor
        :param lm_head_weight: ``(V, H)``.
        :type lm_head_weight: torch.Tensor
        :param lm_head_bias: ``(V,)`` or ``None``.
        :type lm_head_bias: torch.Tensor | None
        :param target_ids: ``(B, T)`` (caller does the ``[:, :-1]``/``[:, 1:]``
            shift before calling).
        :type target_ids: torch.Tensor
        :param temperature: scalar; logits divided by this before log_softmax
            (skipped when ``1.0``).
        :type temperature: float, optional
        :param cast_to_fp32: when True (default), run the per-chunk reduction
            in fp32 then cast back. Same semantics as
            :meth:`_logprobs_from_logits`.
        :type cast_to_fp32: bool, optional
        :param chunk_rows: rows of the flattened ``(B*T)`` workspace per
            iteration; trades launch count vs ``chunk_rows * V`` peak. When
            ``None`` (default) it is resolved from the vocab size via
            a ~256 MB fp32 workspace heuristic.
        :type chunk_rows: int | None, optional
        :return: ``(B, T)`` per-token logprobs in ``hidden.dtype``.
        :rtype: torch.Tensor
        """
        chunk_rows = LLMAlgorithm._resolve_fused_chunk_rows(
            lm_head_weight.shape[0], chunk_rows
        )
        return fused_linear_logprobs_chunked(
            hidden,
            lm_head_weight,
            lm_head_bias,
            target_ids,
            temperature=temperature,
            cast_to_fp32=cast_to_fp32,
            chunk_rows=chunk_rows,
        )

    @staticmethod
    def _logprobs_from_hidden_fused_grad(
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        target_ids: torch.Tensor,
        temperature: float = 1.0,
        cast_to_fp32: bool = True,
        chunk_rows: int | None = None,
    ) -> torch.Tensor:
        """Gradient-aware version of :meth:`_logprobs_from_hidden_fused`.

        Routes through :class:`FusedLinearLogProbsFunction` so the per-token
        logprobs are differentiable w.r.t. ``hidden`` (and ``lm_head_weight`` /
        bias when they require grad) while never materializing the full
        ``(B, T, V)`` logits tensor in the forward *or* backward pass — the
        lm_head matmul is gradient-checkpointed and recomputed chunk-by-chunk.

        Forward values are bit-comparable to :meth:`_logprobs_from_hidden_fused`
        (and hence :meth:`_logprobs_from_logits`); the gradient equals the exact
        ``log_softmax`` gradient.

        :param hidden: ``(B, T, H)`` last-hidden-state (typically requires grad).
        :type hidden: torch.Tensor
        :param lm_head_weight: ``(V, H)``.
        :type lm_head_weight: torch.Tensor
        :param lm_head_bias: ``(V,)`` or ``None``.
        :type lm_head_bias: torch.Tensor | None
        :param target_ids: ``(B, T)`` (caller does the shift before calling).
        :type target_ids: torch.Tensor
        :param temperature: logits divided by this before log_softmax.
        :type temperature: float, optional
        :param cast_to_fp32: run the per-chunk reduction in fp32.
        :type cast_to_fp32: bool, optional
        :param chunk_rows: rows of the flattened ``(B*T)`` workspace per chunk.
            When ``None`` (default) it is resolved from the vocab size via
            a ~256 MB fp32 workspace heuristic.
        :type chunk_rows: int | None, optional
        :return: ``(B, T)`` per-token logprobs in ``hidden.dtype``.
        :rtype: torch.Tensor
        """
        chunk_rows = LLMAlgorithm._resolve_fused_chunk_rows(
            lm_head_weight.shape[0], chunk_rows
        )
        return FusedLinearLogProbsFunction.apply(
            hidden,
            lm_head_weight,
            lm_head_bias,
            target_ids,
            temperature,
            cast_to_fp32,
            chunk_rows,
        )

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
        adapter_state = load_file(adapter_path, device=str(self.device))

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
    def _format_lora_config_mismatch_error(
        current: LoraConfig,
        checkpoint: LoraConfig,
    ) -> str:
        """Format a user-facing error for mismatched LoRA configs.

        :param current: LoRA config from the live loading agent.
        :type current: peft.LoraConfig
        :param checkpoint: LoRA config persisted in the checkpoint.
        :type checkpoint: peft.LoraConfig
        :return: Error string with mismatch context and remediation.
        :rtype: str
        """

        def summarize(cfg: LoraConfig) -> dict[str, Any]:
            """Summarize key LoRA config fields for mismatch messages."""
            cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg))
            summary_keys = (
                "r",
                "lora_alpha",
                "target_modules",
                "modules_to_save",
                "bias",
                "task_type",
            )
            summary = {key: cfg_dict.get(key) for key in summary_keys}
            for key in ("target_modules", "modules_to_save"):
                value = summary.get(key)
                if isinstance(value, (set, tuple)):
                    summary[key] = sorted(value)
            return summary

        current_summary = summarize(current)
        checkpoint_summary = summarize(checkpoint)
        return (
            "LoRA configs differ; refusing to load the checkpoint.\n"
            f"Current config: {current_summary}\n"
            f"Checkpoint config: {checkpoint_summary}\n"
            "Resolution: re-create the agent with the checkpoint's LoRA config "
            "before calling load_checkpoint."
        )

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
        ignore_keys = {"inference_mode"}
        ordered_keys = ("target_modules", "modules_to_save", "exclude_modules")
        a_dict = a.to_dict() if hasattr(a, "to_dict") else dict(vars(a))
        b_dict = b.to_dict() if hasattr(b, "to_dict") else dict(vars(b))
        for key in ordered_keys:
            for d in (a_dict, b_dict):
                val = d.get(key)
                if isinstance(val, (list, tuple, set)):
                    d[key] = sorted(val)
        for key in ignore_keys:
            a_dict.pop(key, None)
            b_dict.pop(key, None)
        return a_dict == b_dict

    def _load_adapter_weights(
        self,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        """Overwrite a live adapter's weights from disk.

        :param checkpoint_dir: Directory written by :meth:`save_checkpoint`; must contain
            ``<adapter_name>/adapter_model.safetensors``.
        :type checkpoint_dir: str
        :param adapter_name: Name of the adapter to overwrite (must already exist on the
            live PEFT model).
        :type adapter_name: str
        :return: None. Mutates the live adapter's parameters in place.
        :rtype: None
        """
        unwrapped = self._get_unwrapped_actor()
        peft_model = unwrapped.pretrained_model if self.use_value_head else unwrapped

        adapter_path = f"{checkpoint_dir}/{adapter_name}/adapter_model.safetensors"
        adapter_state = load_file(adapter_path, device=str(self.device))

        with gather_if_zero3(
            self.zero_stage,
            list(unwrapped.parameters()),
            modifier_rank=0,
        ):
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

        lora_rank = getattr(self.lora_config, "r", None) if self.lora_config else None
        llm_kwargs = build_vllm_llm_init_kwargs(
            self.vllm_config,
            trainer_model_name_or_path=self.pretrained_model_name_or_path,
            max_model_len=self.max_model_len,
            process_index=process_index,
            lora_rank=lora_rank,
        )
        self._ensure_vllm_lora_staging_dir()
        if self.accelerator is None or self.accelerator.process_index == 0:
            warnings.warn(
                f"colocated init: starting vLLM LLM() with "
                f"max_num_batched_tokens={llm_kwargs.get('max_num_batched_tokens')} "
                f"(max_num_seqs={self.vllm_config.max_num_seqs} "
                f"max_model_len={self.max_model_len})",
                stacklevel=2,
            )

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

        # Keep the persistent rollout-adapter slot resident (vLLM V1 otherwise
        # zeroes it on dummy batches and never re-copies it, so the trained
        # adapter would contribute nothing); see ``patch_vllm_lora_keep_resident``.
        # Must run after the in-process engine (and its LoRA layers) exist.
        patched = patch_vllm_lora_keep_resident(self.llm)
        if (
            self.accelerator is None or self.accelerator.process_index == 0
        ) and patched:
            warnings.warn(
                f"colocated init: kept {patched} vLLM LoRA slots resident "
                "(works around vLLM zeroing the rollout adapter slot).",
                stacklevel=2,
            )

        strip_towers = getattr(self.vllm_config, "strip_multimodal_towers", False)
        if strip_towers:
            # Free unused vision/audio towers on multimodal bases (text-only RL
            # never runs them); see ``patch_vllm_strip_multimodal_towers``.
            freed = patch_vllm_strip_multimodal_towers(
                self.llm,
                tower_attrs=strip_towers if isinstance(strip_towers, list) else None,
            )
            if (
                self.accelerator is None or self.accelerator.process_index == 0
            ) and freed:
                total_params = sum(freed.values())
                detail = ", ".join(
                    f"{path}={count / 1e6:.1f}M" for path, count in freed.items()
                )
                warnings.warn(
                    f"colocated init: stripped multimodal towers "
                    f"({total_params / 1e6:.1f}M params freed: {detail}).",
                    stacklevel=2,
                )

        if self.vllm_config.sleep_mode:
            # Native sleep: back the base up to CPU and free the KV cache, so
            # the trainer's own base can use the GPU during the training step.
            self._sleep_vllm_after_init()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _sleep_vllm_after_init(self) -> None:
        """Put the colocated engine to sleep once after construction.

        Native ``sleep(level=sleep_mode_level)``: vLLM cycles its allocator
        state based on the configured sleep level; ``wake_up()`` restores the
        engine allocations.
        """
        assert self.vllm_config is not None  # _configure_vllm guarantees a config
        self.llm.sleep(level=self.vllm_config.sleep_mode_level)
        self._vllm_awake = False
        if self.accelerator is None or self.accelerator.is_main_process:
            log_cuda_memory_snapshot("vLLM sleep complete")

    def _sync_deepspeed_gradient_clipping(self) -> None:
        """Synchronize max_grad_norm with DeepSpeed gradient_clipping config.
        Registered as a mutation hook to ensure consistency after mutations.
        """
        if self.accelerator is None:
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

    def _get_lm_head_parent(self) -> tuple[Any, str]:
        """Locate the parent module owning ``lm_head`` (or ``embed_out``).

        Walks through value-head, PEFT, and LoRA wrappers to the inner
        causal-LM that exposes the language-model head as an attribute.
        Returned so that callers can both read the head (``getattr(parent,
        attr)``) and replace it temporarily (``setattr(parent, attr, ...)``)
        — the latter is used by the no-grad fused-linear-logprob path.

        :return: ``(parent_module, attr_name)``.
        :rtype: tuple[Any, str]
        :raises AttributeError: If no lm_head can be found.
        """
        model = self.actor
        if self.use_value_head and hasattr(model, "pretrained_model"):
            # Value-head wrapper (e.g. AutoModelForCausalLMWithValueHead) →
            # the PEFT/causal-LM inner model.
            model = model.pretrained_model
        if hasattr(model, "base_model"):  # PeftModel → LoraModel
            model = model.base_model
        if hasattr(model, "model"):  # LoraModel → CausalLM
            model = model.model
        for attr in ("lm_head", "embed_out"):
            if hasattr(model, attr):
                return model, attr
        err_msg = (
            f"Cannot find lm_head (or embed_out) in {type(self.actor).__name__}. "
            "The fused-linear-logprob path needs the output embedding layer to "
            "compute per-token log-probs without materializing full logits."
        )
        raise AttributeError(err_msg)

    def _get_lm_head(self) -> torch.nn.Linear:
        """Locate the lm_head module, handling value-head, PEFT and LoRA wrappers.

        :return: The lm_head (or embed_out) linear layer.
        :rtype: torch.nn.Linear
        :raises AttributeError: If no lm_head can be found.
        """
        parent, attr = self._get_lm_head_parent()
        return getattr(parent, attr)

    @contextmanager
    def _patch_lm_head_to_identity(self) -> Generator[torch.nn.Module, None, None]:
        """Temporarily replace ``lm_head`` with ``nn.Identity``.

        With the head identity-patched, the model's ``output.logits`` becomes
        the post-final-norm hidden state ``(B, T, H)`` instead of the full
        ``(B, T, V)`` logits — which is what the no-grad fused-linear-logprob
        kernel consumes directly. The original module is always restored,
        even if the wrapped block raises.
        """
        model, attr = self._get_lm_head_parent()
        original = getattr(model, attr)
        setattr(model, attr, torch.nn.Identity())
        try:
            yield original
        finally:
            setattr(model, attr, original)

    def _get_unwrapped_actor(self) -> Any:  # noqa: ANN401 -- actor spans PEFT/DeepSpeed/value-head/DummyEvolvable wrappers
        """Return actor unwrapped from Accelerate and DummyEvolvable layers."""
        actor = (
            self.accelerator.unwrap_model(self.actor)
            if self.accelerator is not None
            else self.actor
        )
        while isinstance(actor, DummyEvolvable):
            actor = actor.module
        return actor

    @contextmanager
    def _memory_efficient_params(
        self,
    ) -> Generator[None, None, None]:  # pragma: no cover
        """Hold the trainer base on GPU only for the wrapped (training) block.

        Used by the colocated path (``use_memory_efficient_params``): the
        trainer's own base normally rests on CPU so the rollout engine owns the
        GPU; this moves it onto the GPU for the forward/backward and back to CPU
        afterwards, so the two bases never coexist on the GPU (no 2x-base peak).
        Disabled under DeepSpeed ZeRO-3 (params are already sharded).
        """
        if self.zero_stage == 3:
            warnings.warn(
                "Memory efficient params is not yet compatible with DeepSpeed "
                "ZeRO-3; memory efficient params will be disabled for this run.",
                stacklevel=2,
            )
            yield
            return
        unwrapped_model = self._get_unwrapped_actor()
        move_params_to_gpu(unwrapped_model, torch.device(self.device))
        try:
            yield
        finally:
            # Always move the base back on CPU on error
            move_params_to_cpu(unwrapped_model)

    def _prepare_vllm_for_training(self) -> None:
        """Prepare vLLM for learning."""
        if not self.use_vllm:
            return
        assert self.vllm_config is not None  # _configure_vllm guarantees a config
        # Every rank holds its own colocated engine (external_launcher), so
        # every rank must sleep it — not just the main process.
        if self.vllm_config.sleep_mode and self._vllm_awake:
            torch.cuda.empty_cache()
            self.llm.sleep(level=self.vllm_config.sleep_mode_level)
            self._vllm_awake = False

        self._vllm_moved = False

    def _prepare_vllm_for_generation(self) -> None:
        assert self.vllm_config is not None  # _configure_vllm guarantees a config
        if self.use_memory_efficient_params and not self._vllm_awake:
            # Colocated: park the trainer's own base on CPU *before* waking vLLM
            # so the rollout engine owns the GPU and the two bases never coexist
            # on-device. The training step brings it back via
            # ``memory_efficient_params_context``. While the engine is already
            # awake (mid-rollout turns) the base is still parked, so the offload
            # and its log are skipped alongside the wake below.
            move_params_to_cpu(self._get_unwrapped_actor())
            if self.accelerator is None or self.accelerator.is_main_process:
                log_cuda_memory_snapshot(
                    "trainer base offloaded to CPU (before vLLM wake)"
                )
        # Every rank holds its own colocated engine, and _sleep_vllm_after_init
        # slept them all; we wake them all here.
        if self.vllm_config.sleep_mode and not self._vllm_awake:
            torch.cuda.empty_cache()
            self.llm.wake_up()
            self._vllm_awake = True
            if self.accelerator is None or self.accelerator.is_main_process:
                log_cuda_memory_snapshot("vLLM base restored on GPU (after wake)")
        self._sync_actor_to_vllm()
