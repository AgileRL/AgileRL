# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import inspect
import logging
import warnings
from collections.abc import Iterable
from importlib.metadata import version
from typing import (
    Any,
)

import dill
import numpy as np
import torch
from accelerate import Accelerator
from tensordict import TensorDict
from torch._dynamo import OptimizedModule
from typing_extensions import Self

from agilerl.algorithms.core.evolvable_helpers import (
    IndividualT,
    _is_readonly_property,
    get_optimizer_cls,
)
from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.algorithms.core.registry import (
    MutationRegistry,
)
from agilerl.modules import EvolvableModule, ModuleDict
from agilerl.protocols import (
    AgentWrapperProtocol,
    EvolvableAlgorithmProtocol,
    EvolvableModuleProtocol,
    ModuleDictProtocol,
)
from agilerl.typing import (
    CheckpointInfo,
    DeviceType,
)
from agilerl.utils.algo_utils import (
    _resolve_lr,
    chkpt_attribute_to_device,
    configure_tf32_precision,
    filter_init_dict,
    isroutine,
    module_checkpoint_dict,
)
from agilerl.utils.constructor_kwargs import (
    constructor_kwargs_from_flat,
    constructor_kwargs_from_obj,
    with_runtime_wrap,
)
from agilerl.utils.constructor_kwargs import (
    from_hparams as construct_from_hparams,
)

logger = logging.getLogger(__name__)


class EvolvableCheckpointMixin:
    """Clone, save, and load for :class:`EvolvableAlgorithm`."""

    @staticmethod
    def inspect_attributes(
        agent: EvolvableAlgorithmProtocol | AgentWrapperProtocol[Any],
        input_args_only: bool = False,
        exclude: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Inspect and retrieve the attributes of the current object, excluding attributes related to the
        underlying evolvable networks (i.e. `EvolvableModule`, `torch.optim.Optimizer`) and with
        an option to include only the attributes that are input arguments to the constructor.

        :param input_args_only: If True, only include attributes that are input arguments to the constructor.
                                Defaults to False.
        :type input_args_only: bool
        :param exclude: Extra attribute names to drop from the result, on top of the standard exclusions
            below. For a caller-specific reason to leave an attribute out of its own view.
        :type exclude: Iterable[str], optional
        :return: A dictionary of attribute names and their values.
        :rtype: dict[str, Any]
        """
        names = [n for n in dir(agent) if not _is_readonly_property(agent, n)]
        attributes = [
            (n, val) for n in names if not isroutine(val := getattr(agent, n))
        ]

        excluded_names = list(agent.evolvable_attributes().keys())
        excluded_names += [
            attr for attr, val in attributes if isinstance(val, TensorDict)
        ]
        excluded_names += list(exclude)

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
                if k not in excluded_names and k in constructor_params
            }
        else:
            # Remove the algo specific guarded variables (if specified)
            attributes = {k: v for k, v in attributes if k not in excluded_names}
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
        from agilerl.algorithms.core.evolvable_algorithm import EvolvableAlgorithm

        for attribute in EvolvableCheckpointMixin.inspect_attributes(agent):
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
        input_args = with_runtime_wrap(constructor_kwargs_from_obj(self), wrap)

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
        clone = EvolvableCheckpointMixin.copy_attributes(self, clone)
        if index is not None:
            clone.index = index

        return clone

    @classmethod
    def from_hparams(cls, *args: Any, **hparams: Any) -> Self:
        """Build an instance from positional spaces and a flat hyperparameter mapping."""
        return construct_from_hparams(cls, *args, **hparams)

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

    def load_weights(self, path: str) -> None:
        """Load only the network weights from a checkpoint.

        Warm-starts a new run from prior weights; optimizer, LR schedule,
        training progress and hyperparameters are not loaded.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint: dict[str, Any] = torch.load(
            path,
            map_location=self.device,
            pickle_module=dill,
            weights_only=False,
        )
        self._load_torch_checkpoint(checkpoint)

        if self.accelerator is not None:
            self.wrap_models()
        elif self.torch_compiler:
            configure_tf32_precision()
            self.recompile()

    def _load_torch_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Recreate the evolvable modules and load their state dicts.

        :param checkpoint: Deserialized checkpoint dictionary.
        :type checkpoint: dict[str, Any]
        """
        network_info: CheckpointInfo = checkpoint["network_info"]
        modules = network_info["modules"]
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

    def load_checkpoint(self, path: str) -> None:
        """Load saved agent properties and network weights from checkpoint.

        Restores full training state (weights, optimizer, LR schedule,
        hyperparameters) to resume a run; :meth:`load_weights` takes weights only.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint: dict[str, Any] = torch.load(
            path,
            map_location=self.device,
            pickle_module=dill,
            weights_only=False,
        )

        self._load_torch_checkpoint(checkpoint)

        network_info: CheckpointInfo = checkpoint["network_info"]
        optimizers = network_info["optimizers"]
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
            # Checkpoint carries the writer's device; the live agent owns placement.
            if attribute == "device":
                continue
            if _is_readonly_property(self, attribute):
                continue
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
        class_init_dict = constructor_kwargs_from_flat(cls, checkpoint)
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

        for attribute in EvolvableCheckpointMixin.inspect_attributes(
            self, exclude=("grama_scores",)
        ):
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

def get_checkpoint_dict(
    agent: EvolvableAlgorithmProtocol,
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
    attribute_dict = EvolvableCheckpointMixin.inspect_attributes(agent)
    attribute_dict["agilerl_version"] = version("agilerl")
    attribute_dict.pop("accelerator", None)
    attribute_dict.pop("rollout_buffer", None)
    attribute_dict.pop("grama_scores", None)

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
