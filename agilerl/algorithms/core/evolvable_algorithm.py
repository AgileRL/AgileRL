# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    cast,
)

import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator
from gymnasium import spaces
from torch.utils.hooks import RemovableHandle
from typing_extensions import Self

from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    MutationHook,
    MutationRegistry,
    NetworkGroup,
    OptimizerConfig,
)
from agilerl.metrics import AgentMetrics, MultiAgentMetrics
from agilerl.modules import EvolvableModule, ModuleDict
from agilerl.protocols import (
    EvolvableModuleProtocol,
)
from agilerl.typing import (
    ActionResult,
    ActionType,
    BackwardHook,
    DeviceType,
    ExperiencesT,
    FitnessValue,
    GradInput,
    GraMaScores,
    GymSpaceType,
    LrNameType,
    MultiAgentObservationType,
    ObservationType,
    TorchObsType,
)
from agilerl.utils.algo_utils import (
    check_supported_space,
    get_input_size_from_space,
    get_output_size_from_space,
    needs_image_transpose,
    preprocess_observation,
    recursive_check_module_attrs,
    transpose_image_space,
)
from agilerl.utils.evolvable_networks import (
    compile_model,
)
from agilerl.utils.mutation_utils import target_activations

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import SequentialLR




from agilerl.algorithms.core.evolvable_checkpoint import EvolvableCheckpointMixin
from agilerl.algorithms.core.evolvable_helpers import (
    RegistryMeta,
    SelfAgentWrapper,
    _is_readonly_property,
    _per_neuron_grad,
    build_classic_rl_population,
)

logger = logging.getLogger(__name__)


class EvolvableAlgorithm(EvolvableCheckpointMixin, ABC, Generic[ExperiencesT], metaclass=RegistryMeta):
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
        self.grama_scores: GraMaScores | None = None
        self._grama_handles: list[RemovableHandle] = []
        self._grama_latest: GraMaScores | None = None

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

    def init_training_step(self, capture_grama: bool = False) -> None:
        """Open the agent's training block: metrics tracking, and GraMa capture.

        Hooks are registered afresh each cycle, so they follow the agent
        through architecture mutations, checkpoint reloads and accelerator
        re-wrapping. Opening a block implicitly closes one that an earlier call
        left open.

        :param capture_grama: Whether to register GraMa capture hooks for this
            training step. Defaults to False since the LLM finetuners never run
            ReGraMa.
        :type capture_grama: bool
        :return: None.
        :rtype: None
        """
        self.metrics.init_training_step()
        self._set_grama_capture(capture_grama)

    def finalize_training_step(self, num_steps: int) -> None:
        """Close the agent's training block, storing any captured GraMa scores.

        :param num_steps: Number of steps taken during the training step.
        :type num_steps: int
        :return: None.
        :rtype: None
        """
        self.metrics.finalize_training_step(num_steps)
        self._release_grama_capture()

    def _set_grama_capture(self, capture_grama: bool) -> None:
        """Close any open GraMa capture, then open a new one if requested.

        :param capture_grama: Whether to register GraMa capture hooks for the
            training step being opened.
        :type capture_grama: bool
        :return: None.
        :rtype: None
        """
        self._release_grama_capture()
        if capture_grama:
            self._register_grama_capture()

    def _register_grama_capture(self) -> None:
        """Hook every measured activation of every evaluation network.

        A full backward hook on an activation is handed grad_{z_i}L, the
        gradient w.r.t. its input, so the GraMa metric is measured for free
        during the real training backward pass.

        :return: None.
        :rtype: None
        """
        latest: GraMaScores = []
        self._grama_latest = latest
        for net_idx, (_network_id, network) in enumerate(self.unrolled_eval_networks()):
            targets = target_activations(network)
            latest.append([None] * len(targets))
            for mod_idx, module in enumerate(targets):
                handle = module.register_full_backward_hook(
                    self._grama_hook(latest, net_idx, mod_idx),
                )
                self._grama_handles.append(handle)

    @staticmethod
    def _grama_hook(
        latest: GraMaScores,
        net_idx: int,
        mod_idx: int,
    ) -> BackwardHook:
        """Build the backward hook recording one measured activation's gradient.

        :param latest: The open capture's snapshot, written in place.
        :type latest: GraMaScores
        :param net_idx: Position of the layer's network in the snapshot.
        :type net_idx: int
        :param mod_idx: Position of the layer within that network's snapshot.
        :type mod_idx: int
        :return: The hook to register on that activation.
        :rtype: BackwardHook
        """

        def hook(
            _module: torch.nn.Module,
            grad_input: GradInput,
            _grad_output: GradInput,
        ) -> None:
            gradient = _per_neuron_grad(grad_input)
            if gradient is None:
                return
            latest[net_idx][mod_idx] = gradient

        return hook

    def _release_grama_capture(self) -> None:
        """Close any open GraMa capture, storing its snapshot and removing its hooks.

        :return: None.
        :rtype: None
        """
        latest = self._grama_latest
        if latest is None:
            return
        self.grama_scores = [list(net_latest) for net_latest in latest]
        self._remove_grama_handles()
        self._grama_latest = None

    def _remove_grama_handles(self) -> None:
        """Detach every backward hook this capture registered.

        :return: None.
        :rtype: None
        """
        for handle in self._grama_handles:
            handle.remove()
        self._grama_handles = []

    def get_eval_modules(
        self,
        cloning: bool = True,
    ) -> tuple[dict[str, EvolvableModule], dict[str, EvolvableModule]]:
        """Get the offsprings of all of the evaluation modules in the individual.

        :param cloning: Whether to clone each evaluation module before returning it,
            defaults to True.
        :type cloning: bool, optional

        :return: Tuple of offspring policy and the rest of the evaluation modules
        :rtype: tuple[dict[str, EvolvableModule], dict[str, EvolvableModule]]
        """
        offspring_modules: dict[str, EvolvableModule] = {}
        offspring_policy: dict[str, EvolvableModule] = {}
        for group in self.registry.groups:
            eval_name = group.eval_network_name()
            eval_module: EvolvableModule = getattr(self, eval_name)

            # Clone the offspring prior to applying mutations
            offspring = eval_module.clone() if cloning else eval_module
            if group.policy:
                offspring_policy[eval_name] = offspring
            else:
                offspring_modules[eval_name] = offspring

        return offspring_policy, offspring_modules

    def unrolled_eval_networks(self) -> list[tuple[str | None, torch.nn.Module]]:
        """Return the agent's evaluation networks as (network_id, network) pairs.

        :return: One (network_id, network) pair per measured network.
        :rtype: list[tuple[str | None, torch.nn.Module]]
        """
        offspring_policy, offspring_modules = self.get_eval_modules(cloning=False)

        accelerator = self.accelerator
        pairs: list[tuple[str | None, torch.nn.Module]] = []
        for eval_net in chain(offspring_policy.values(), offspring_modules.values()):
            if accelerator is not None:
                eval_net = accelerator.unwrap_model(eval_net)
            if isinstance(eval_net, ModuleDict):
                sub_networks = cast("ModuleDict[torch.nn.Module]", eval_net)
                pairs.extend(sub_networks.items())
            else:
                pairs.append((None, eval_net))
        return pairs

    def eval_policy_network_ids(self) -> set[int]:
        """Return the id of every evaluation network in the agent's policy group.

        :return: Identities of the policy's evaluation networks.
        :rtype: set[int]
        """
        policy_name = self.registry.policy()
        if not isinstance(policy_name, str):
            return set()
        policy = getattr(self, policy_name, None)
        if policy is None:
            return set()
        if isinstance(policy, dict) and not isinstance(policy, ModuleDict):
            return {
                id(
                    self.accelerator.unwrap_model(module)
                    if self.accelerator is not None
                    else module
                )
                for module in policy.values()
            }
        if self.accelerator is not None:
            policy = self.accelerator.unwrap_model(policy)
        if isinstance(policy, ModuleDict):
            return {id(module) for _key, module in policy.items()}
        return {id(policy)}

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

        from agilerl.algorithms.core.llm_algorithm import LLMAlgorithm

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

        evolvable_attrs: dict[str, Any] = {}
        for attr in dir(self):
            if _is_readonly_property(self, attr):
                continue
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
