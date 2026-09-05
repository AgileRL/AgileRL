# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABCMeta
from collections.abc import Callable
from typing import (
    Any,
    Protocol,
    TypeVar,
)

import torch

from agilerl.algorithms.core.registry import (
    OptimizerFactory,
)
from agilerl.protocols import (
    AgentWrapperProtocol,
    EvolvableAlgorithmProtocol,
)
from agilerl.typing import (
    DeviceType,
    GradInput,
    GymSpaceType,
    MultiAgentSpacesType,
)
from agilerl.utils.constructor_kwargs import (
    assemble_init_kwargs,
    own_init_has_var_params,
)

logger = logging.getLogger(__name__)


def _is_readonly_property(obj: object, name: str) -> bool:
    """Return True when ``name`` is a property without a setter on ``obj``'s type.

    Derived attributes (e.g. GRPO's ``aux_metric_name``) must not be persisted or
    restored via ``setattr`` — checkpoint restore would raise
    ``AttributeError: can't set attribute``.

    :param obj: Instance whose class MRO is inspected.
    :type obj: object
    :param name: Attribute name.
    :type name: str
    :return: Whether ``name`` is a read-only property.
    :rtype: bool
    """
    for cls in type(obj).__mro__:
        attr = vars(cls).get(name)
        if isinstance(attr, property):
            return attr.fset is None
        if attr is not None:
            return False
    return False


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
AlgoT = TypeVar("AlgoT", bound=EvolvableAlgorithmProtocol)

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
        if own_init_has_var_params(cls):
            instance: AlgoT = super().__call__(*args, **kwargs)
        else:
            instance = super().__call__(**assemble_init_kwargs(cls, args, kwargs))

        # Initialize the MutationRegistry -> ensures that all of the networks and
        # optimizers are registered with the algorithm, and that the specified hyperparameters
        # to mutate have been set as attributes in the algorithm.
        if isinstance(instance, cls) and hasattr(instance, "_registry_init"):
            instance._registry_init()

        return instance


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


def _per_neuron_grad(grad_input: GradInput) -> torch.Tensor | None:
    """Reduce an activation's grad_input to one |grad_{z_i}L| per neuron.

    The first element of the tuple a full backward hook receives is the gradient
    of the loss w.r.t. the module's input, i.e. the pre-activation gradient.
    Dense gradients have shape (batch, H) and are averaged over the batch;
    convolutional gradients have shape (batch, C, *spatial) and are averaged
    over the batch and spatial dimensions.

    :param grad_input: The gradient a full backward hook was handed.
    :type grad_input: GradInput
    :return: One mean absolute gradient per neuron, or None if none flowed.
    :rtype: torch.Tensor | None
    """
    if isinstance(grad_input, (tuple, list)):
        grad = grad_input[0] if len(grad_input) > 0 else None
    else:
        grad = grad_input
    if grad is None:
        return None
    magnitude = grad.detach().abs()
    if magnitude.dim() <= 1:
        return magnitude
    reduce_dims = [dim for dim in range(magnitude.dim()) if dim != 1]
    return magnitude.mean(dim=reduce_dims)
