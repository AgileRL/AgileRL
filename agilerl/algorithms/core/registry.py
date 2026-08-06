# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import ItemsView, Iterator
from dataclasses import dataclass, field
from types import FunctionType, MethodType
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import torch

from agilerl.protocols import EvolvableAlgorithmProtocol, NamedCallable
from agilerl.typing import LrNameType, NetworkType
from agilerl.utils.algo_utils import DummyOptimizer

# An optimizer "class": a torch ``Optimizer`` subclass or an optimizer-like
# constructor such as ``DummyOptimizer`` or an Accelerate wrapper type.
OptimizerFactory = NamedCallable

# A mutation hook: a zero-argument function or bound method registered by name
# and looked up on the algorithm when applied.
MutationHook = FunctionType | MethodType


@dataclass
class NetworkConfig:
    """Dataclass for storing the configuration of an evolvable network
    within an `EvolvableAlgorithm`.

    :param name: The name of the attribute where the network is stored.
    :type name: str
    :param eval_network: Whether the network is an evaluation network. This implies
    that the network is optimized during training. Default is False.

    :type eval_network: bool
    :param optimizer: The name of the optimizer that updates the network.
    :type optimizer: str | None
    """

    name: str
    eval_network: bool = field(default=False)
    optimizer: str | None = field(default=None)

    def __post_init__(self) -> None:
        if self.eval_network and self.optimizer is None:
            msg = "Evaluation network must have an optimizer associated with it."
            raise ValueError(
                msg,
            )


@dataclass
class OptimizerConfig:
    """Dataclass for storing the configuration of an optimizer within an `EvolvableAlgorithm`. Usually
    an optimizer will be used to update the parameters of a single evaluation network, but in some cases
    it may be used to update the parameters of multiple networks simultaneously. Here we provide the
    flexibility to specify such configurations for PyTorch optimizers.

    :param name: The name of the attribute where the optimizer is stored.
    :type name: str
    :param networks: The list of network attribute names that the optimizer will update.
    :type networks: list[str]
    :param lr: Attribute name(s) for learning rate on the algorithm: ``str`` or
        ``("lr_actor", "lr_critic")`` for split LLM optimizers.
    :type lr: str | tuple[str, str]
    :param optimizer_cls: The optimizer class/es to be used. Stored as the class
        name/s (``str`` or ``dict[str, str]``) after ``__post_init__`` for
        serialization.
    :type optimizer_cls: OptimizerFactory | dict[str, OptimizerFactory]
    :param optimizer_kwargs: The keyword arguments to be passed to the optimizer.
    :type optimizer_kwargs: dict[str, Any]
    """

    name: str
    networks: list[str]
    lr: LrNameType
    optimizer_cls: OptimizerFactory | dict[str, OptimizerFactory] | str | dict[str, str]
    optimizer_kwargs: dict[str, Any] | list[dict[str, Any]]

    def __post_init__(self) -> None:
        # Save optimizer_cls as string for serialization. Excluding the scalar
        # members isolates the per-agent mapping with its ``str`` keys intact and
        # each value narrowed to ``NamedCallable | str``.
        if not isinstance(self.optimizer_cls, (str, NamedCallable)):
            self.optimizer_cls = {
                agent_id: cls if isinstance(cls, str) else cls.__name__
                for agent_id, cls in self.optimizer_cls.items()
            }
        elif not isinstance(self.optimizer_cls, str):
            self.optimizer_cls = self.optimizer_cls.__name__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptimizerConfig):
            return False
        return self.name == other.name and self.networks == other.networks

    def get_optimizer_cls(self) -> OptimizerFactory | dict[str, OptimizerFactory]:
        """Get the optimizer class/es from the stored configuration.

        :return: The optimizer class/es from the stored configuration.
        :rtype: OptimizerFactory | dict[str, OptimizerFactory]
        """
        name_to_cls: dict[str, OptimizerFactory] = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "RMSprop": torch.optim.RMSprop,
            "Adadelta": torch.optim.Adadelta,
            "Adagrad": torch.optim.Adagrad,
            "Adamax": torch.optim.Adamax,
            "ASGD": torch.optim.ASGD,
            "LBFGS": torch.optim.LBFGS,
            "Rprop": torch.optim.Rprop,
            "DummyOptimizer": DummyOptimizer,
        }
        # ``__post_init__`` serializes optimizer_cls to name string/s.
        if not isinstance(self.optimizer_cls, (str, NamedCallable)):
            result: dict[str, OptimizerFactory] = {}
            for agent_id, cls_name in self.optimizer_cls.items():
                assert isinstance(cls_name, str), (
                    "Optimizer classes are serialized to name strings in __post_init__."
                )
                result[agent_id] = name_to_cls[cls_name]
            return result

        cls_name = self.optimizer_cls
        assert isinstance(cls_name, str)
        return name_to_cls[cls_name]


@dataclass
class RLParameter:
    """Dataclass for storing the configuration of a hyperparameter that will be mutated during
    training. The hyperparameter is defined by a range of values that it can take, and the
    shrink and grow factors that will be used to mutate the hyperparameter value.

    :param min: The minimum value that the hyperparameter can take. For numpy arrays, this will be broadcast.
    :type min: float
    :param max: The maximum value that the hyperparameter can take. For numpy arrays, this will be broadcast.
    :type max: float
    :param shrink_factor: The factor by which the hyperparameter will be shrunk during mutation. Default is 0.8.
    :type shrink_factor: float
    :param grow_factor: The factor by which the hyperparameter will be grown during mutation. Default is 1.2.
    :type grow_factor: float
    :param dtype: The data type of the hyperparameter. Default is float.
    :type dtype: type[float] | type[int] | type[npt.NDArray]
    :param value: The current value of the hyperparameter. Default is None.
    :type value: int | float | npt.NDArray | None
    """

    min: float
    max: float
    shrink_factor: float = 0.8
    grow_factor: float = 1.2
    dtype: type[float] | type[int] | type[npt.NDArray] = float
    value: int | float | npt.NDArray | None = field(default=None, init=False)

    def mutate(self) -> int | float | npt.NDArray:
        """Mutate the hyperparameter value by either growing or shrinking it.

        For scalar values (int/float), the mutation applies the grow/shrink factor uniformly.
        For numpy arrays, the mutation is applied element-wise, with proper broadcasting
        of min/max constraints and preservation of the original array's dtype.

        :return: The mutated hyperparameter value.
        :rtype: int | float | npt.NDArray
        """
        value = self.value
        assert value is not None, "Hyperparameter value is not set"

        # Equal probability of growing or shrinking; the value is clipped to
        # [min, max] either way, so applying the factor then clipping is
        # equivalent to the bounded grow/shrink update.
        factor = self.shrink_factor if torch.rand(1).item() < 0.5 else self.grow_factor

        if isinstance(value, np.ndarray):
            # Element-wise update, preserving the original array's dtype
            self.value = np.clip(value * factor, self.min, self.max).astype(value.dtype)
        else:
            new_value = min(max(value * factor, self.min), self.max)
            # Cast the new value to the correct dtype (scalar values are
            # int or float; ndarray values take the branch above)
            self.value = int(new_value) if self.dtype is int else float(new_value)

        return self.value


class HyperparameterConfig:
    """Stores the RL hyperparameters that will be mutated during training. For each
    hyperparameter, we store the name of the attribute where the hyperparameter is
    stored, and the range of values that the hyperparameter can take.
    """

    def __init__(self, **kwargs: RLParameter) -> None:
        self.config = kwargs
        for key, value in kwargs.items():
            if not isinstance(value, RLParameter):
                msg = "Expected RLParameter object for hyperparameter configuration."
                raise TypeError(msg)

            setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            "HyperparameterConfig(\n"
            + "\n".join([f"{key}: {value}" for key, value in self.config.items()])
            + "\n)"
        )

    def __bool__(self) -> bool:
        """Return False if the config is empty, True otherwise.

        :return: Whether the config contains any hyperparameters
        :rtype: bool
        """
        return bool(self.config)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperparameterConfig):
            return False
        return set(self.names()) == set(other.names())

    def __iter__(self) -> Iterator[str]:
        return iter(self.config)

    def __getitem__(self, key: str) -> RLParameter:
        return self.config[key]

    def names(self) -> list[str]:
        return list(self.config.keys())

    def items(self) -> ItemsView[str, RLParameter]:
        return self.config.items()

    def sample(self) -> tuple[str, RLParameter]:
        """Sample a hyperparameter from the configuration.

        :return: The name of the hyperparameter and its configuration.
        :rtype: tuple[str, RLParameter]
        """
        key = torch.randperm(len(self.config))[0]
        return list(self.config.keys())[key], list(self.config.values())[key]


def make_default_hp_config(**kwargs: float) -> HyperparameterConfig:
    """Create a default HyperparameterConfig with bounds derived from the current values.

    For floats (e.g. learning rates), bounds are set to ``[value / 10, value * 10]``.
    For ints (e.g. batch_size, learn_step), bounds are set to ``[value // 4, value * 4]``
    (clamped to a minimum of 1).

    :param kwargs: Mapping of hyperparameter names to their current values.
    :returns: A HyperparameterConfig with sensible mutation ranges.
    :rtype: HyperparameterConfig
    """
    params: dict[str, RLParameter] = {}
    for name, value in kwargs.items():
        if isinstance(value, float):
            params[name] = RLParameter(min=value / 10, max=value * 10)
        elif isinstance(value, int):
            params[name] = RLParameter(
                min=max(1, value // 4),
                max=value * 4,
                grow_factor=1.5,
                shrink_factor=0.75,
            )
    return HyperparameterConfig(**params)


@dataclass
class NetworkGroup:
    """Dataclass for storing a group of networks. This consists of an evaluation network (i.e.
    a network that is optimized during training) and, optionally, some other networks that
    share parameters with the evaluation network (e.g. the target network in DQN). If the
    networks are passed as an `agilerl.modules.base.ModuleDict` object, we assume that the networks
    are part of a multiagent setting.

    :param eval_network: The evaluation network. Replaced by the network's
        attribute name (``str``) in ``__post_init__``.
    :type eval_network: NetworkType | str
    :param shared_networks: The list of shared networks. Replaced by the
        networks' attribute names (``list[str]``) in ``__post_init__``.
    :type shared_networks: NetworkType | list[NetworkType] | str | list[str] | None
    :param policy: Whether the network is a policy (e.g. the network used to get the actions
        of the agent). There must be one network group in an algorithm which sets this to True.
        Default is False.
    :type policy: bool
    """

    eval_network: NetworkType | str
    shared_networks: NetworkType | list[NetworkType] | str | list[str] | None = field(
        default=None,
    )
    policy: bool = field(default=False)

    def __post_init__(self) -> None:
        # Check that the shared networks are of the same type as the eval network
        if self.shared_networks is not None:
            eval_cls = type(self.eval_network)
            if isinstance(self.shared_networks, list):
                assert all(isinstance(net, eval_cls) for net in self.shared_networks), (
                    f"Expected a list of {eval_cls.__name__} objects for the "
                    f"shared argument in the network group. Found {type(self.shared_networks[0])}."
                )
            else:
                assert isinstance(self.shared_networks, eval_cls), (
                    f"Expected a {eval_cls.__name__} object for the "
                    f"shared argument in the network group. Found {type(self.shared_networks)}."
                )

        # Identify the names of the attributes where the networks are stored
        container = self._infer_parent_container()
        self.eval_network = self._infer_attribute_names(container, self.eval_network)[0]
        if self.shared_networks is not None:
            shared = (
                self.shared_networks
                if isinstance(self.shared_networks, list)
                else [self.shared_networks]
            )
            self.shared_networks = self._infer_attribute_names(container, shared)

    def __hash__(self) -> int:
        return hash((self.eval_network, self.shared_networks, self.policy))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NetworkGroup):
            return False
        return (
            self.eval_network == other.eval_network
            and self.shared_networks == other.shared_networks
            and self.policy == other.policy
        )

    def eval_network_name(self) -> str:
        """Attribute name of the evaluation network.

        :return: The attribute name resolved by ``__post_init__``.
        :rtype: str
        """
        name = self.eval_network
        assert isinstance(name, str), (
            "NetworkGroup attribute names are resolved in __post_init__."
        )
        return name

    def shared_network_names(self) -> list[str]:
        """Attribute names of the shared networks (empty when there are none).

        :return: The attribute names resolved by ``__post_init__``.
        :rtype: list[str]
        """
        shared = self.shared_networks
        if shared is None:
            return []
        items = shared if isinstance(shared, list) else [shared]
        names: list[str] = []
        for item in items:
            assert isinstance(item, str), (
                "NetworkGroup attribute names are resolved in __post_init__."
            )
            names.append(item)
        return names

    def _infer_parent_container(self) -> EvolvableAlgorithmProtocol:
        """Infer the parent container dynamically using the stack frame.

        :return: The parent container object
        :rtype: EvolvableAlgorithm
        """
        # NOTE: Here the assumption is that NetworkGroup is used inside the __init__
        # method of the implemented algorithm, such that we can access the defined locals
        # and extract the corresponding attribute names to the passed networks.
        # Walk three frames up (this method -> caller -> ... -> algorithm __init__);
        # each ``f_back`` is Optional, so narrow before dereferencing.
        frame = inspect.currentframe()
        for _ in range(3):
            assert frame is not None, "expected an active caller frame"
            frame = frame.f_back
        assert frame is not None, "expected the algorithm __init__ frame"
        return frame.f_locals["self"]

    def _infer_attribute_names(
        self,
        container: object,
        objects: object | list[object],
    ) -> list[str]:
        """Infer attribute names of the networks being optimized.

        :param container: The container object to inspect.
        :type container: object
        :param objects: The objects to match.
        :type objects: object | list[object]

        :return: List of attribute names for the networks
        :rtype: list[str]
        """

        def _match_condition(attr_value: object) -> bool:
            if isinstance(objects, list):
                return any(id(attr_value) == id(obj) for obj in objects)
            return id(attr_value) == id(objects)

        return [
            attr_name
            for attr_name, attr_value in vars(container).items()
            if _match_condition(attr_value)
        ]


def make_network_group(
    eval_network: str,
    shared_networks: str | list[str] | None,
    policy: bool = False,
) -> NetworkGroup:
    """Make a network group from a given eval network and, optionally, some network/s that
    share parameters with the eval network.

    :param eval_network: The evaluation network.
    :type eval_network: str
    :param shared_networks: The list of shared networks.
    :type shared_networks: str | list[str] | None
    :param policy: Whether the network is a policy (e.g. the network used to get the actions
    of the agent). There must be one network group in an algorithm which sets this to True.
    Default is False.
    :type policy: bool

    :return: NetworkGroup object with the passed configuration.
    :rtype: NetworkGroup
    """
    return NetworkGroup(
        eval_network=eval_network,
        shared_networks=shared_networks,
        policy=policy,
    )


@dataclass
class MutationRegistry:
    """Registry to keep track of the components of an algorithms that may evolve during training.
    This is interpreted by a :class:`Mutations <agilerl.hpo.mutations.Mutations>` object
    when performing evolutionary hyperparameter optimization. This includes:

    1. The hyperparameter configuration of the algorithm.
    2. The network groups of the algorithm.
    3. The optimizers of the algorithm.
    4. The mutation hooks of the algorithm (i.e. functions that are called after a mutation is performed).

    :param hp_config: The hyperparameter configuration of the algorithm.
    :type hp_config: HyperparameterConfig
    """

    hp_config: HyperparameterConfig | None = field(default=None)

    def __post_init__(self) -> None:
        self.groups: list[NetworkGroup] = []
        self.optimizers: list[OptimizerConfig] = []
        # Hooks are stored by name and looked up on the algorithm when applied.
        self.hooks: list[str] = []

        if self.hp_config is None:
            self.hp_config = HyperparameterConfig()

    def __repr__(self) -> str:
        groups_str = "\n".join(
            [
                f"Eval: '{group.eval_network}', Shared: {group.shared_networks}"
                for group in self.groups
            ],
        )
        optimizers_str = "\n".join(
            [
                f"{opt.optimizer_cls}: '{opt.name}', Networks: {opt.networks}"
                for opt in self.optimizers
            ],
        )
        return f"Network Groups:\n{groups_str}\n\nOptimizers:\n{optimizers_str}"

    def __eq__(self, other: object) -> bool:
        """Check if two MutationRegistry objects are equal. This involves checking
        that the network groups and optimizer configurations are the same.

        :param other: The other object to compare with.
        :type other: object

        :return: True if the two MutationRegistry objects are equal, False otherwise.
        :rtype: bool
        """
        if not isinstance(other, MutationRegistry):
            return False
        return self.groups == other.groups and self.optimizers == other.optimizers

    @property
    def optimizer_networks(self) -> dict[str, list[str]]:
        """Get a dictionary of optimizer names and the network attribute names that they update.

        :return: A dictionary of optimizer names and the network attribute names that they update.
        :rtype: dict[str, list[str]]
        """
        return {config.name: config.networks for config in self.optimizers}

    @overload
    def policy(self, return_group: Literal[False] = ...) -> str | None: ...

    @overload
    def policy(self, return_group: Literal[True]) -> NetworkGroup | None: ...

    @overload
    def policy(self, return_group: bool) -> str | NetworkGroup | None: ...

    def policy(self, return_group: bool = False) -> str | NetworkGroup | None:
        """Get the name of the policy network in the registry.

        :param return_group: Whether to return the network group instead of just the name.
        :type return_group: bool

        :return: The name of the policy network in the registry.
        :rtype: str | NetworkGroup | None
        """
        for group in self.groups:
            if group.policy:
                return group if return_group else group.eval_network_name()
        return None

    def all_registered(self) -> set[str]:
        """Return all of the members in the registry.

        :return: The names of all the members in the registry.
        :rtype: set[str]
        """
        all_registered = {group.eval_network_name() for group in self.groups}
        for group in self.groups:
            all_registered.update(group.shared_network_names())
        all_registered.update(opt.name for opt in self.optimizers)
        return all_registered

    def networks(self) -> list[NetworkConfig]:
        """Get a list of network configurations in the registry.

        :return: A list of network configurations in the registry. This includes
            the evaluation and shared networks.
        :rtype: list[NetworkConfig]
        """
        # Match with optimizers (only eval networks can have optimizers by definition)
        optimizer_eval = {}
        for opt_name, nets in self.optimizer_networks.items():
            for net in nets:
                optimizer_eval[net] = opt_name

        # Fetch evaluation and shared networks
        eval_networks = [
            NetworkConfig(
                name=group.eval_network_name(),
                eval_network=True,
                optimizer=optimizer_eval.get(group.eval_network_name()),
            )
            for group in self.groups
        ]
        shared_networks = [
            NetworkConfig(name=shared, eval_network=False)
            for group in self.groups
            for shared in group.shared_network_names()
        ]

        return eval_networks + shared_networks

    def register_group(self, group: NetworkGroup) -> None:
        """Register a network configuration in the registry.

        :param group: The network group to be registered.
        :type group: NetworkGroup
        """
        self.groups.append(group)

    def register_optimizer(self, optimizer: OptimizerConfig) -> None:
        """Register an optimizer configuration in the registry.

        :param optimizer: The optimizer configuration to be registered.
        :type optimizer: OptimizerConfig
        """
        self.optimizers.append(optimizer)

    def register_hook(self, hook: MutationHook) -> None:
        """Register a hook in the registry as its name. This is used to store the names of the
        mutation hooks that will be applied after a mutation is performed.

        :param hook: The hook to be registered.
        :type hook: MutationHook
        """
        self.hooks.append(hook.__name__)
