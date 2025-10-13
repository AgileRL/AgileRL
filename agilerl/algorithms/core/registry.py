import inspect
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch.optim import Optimizer

from agilerl.protocols import EvolvableAlgorithm
from agilerl.typing import NetworkType
from agilerl.utils.llm_utils import DummyOptimizer


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
    :type optimizer: Optional[str]
    """

    name: str
    eval_network: bool = field(default=False)
    optimizer: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.eval_network and self.optimizer is None:
            raise ValueError(
                "Evaluation network must have an optimizer associated with it."
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
    :param lr: The learning rate of the optimizer.
    :type lr: str
    :param optimizer_cls: The optimizer class to be used.
    :type optimizer_cls: type[Optimizer]
    :param optimizer_kwargs: The keyword arguments to be passed to the optimizer.
    :type optimizer_kwargs: dict[str, Any]
    """

    name: str
    networks: Union[str, list[str]]
    lr: str
    optimizer_cls: Union[type[Optimizer], list[type[Optimizer]]]
    optimizer_kwargs: Union[dict[str, Any], list[dict[str, Any]]]

    def __post_init__(self):
        # Save optimizer_cls as string for serialization
        if isinstance(self.optimizer_cls, dict):
            self.optimizer_cls = {
                agent_id: cls.__name__ for agent_id, cls in self.optimizer_cls.items()
            }
        else:
            self.optimizer_cls = self.optimizer_cls.__name__

    def __eq__(self, other: "OptimizerConfig") -> bool:
        return self.name == other.name and self.networks == other.networks

    def get_optimizer_cls(self) -> Union[type[Optimizer], dict[str, type[Optimizer]]]:
        """Get the optimizer object/s from the stored configuration.

        :return: The optimizer object/s from the stored configuration.
        :rtype: Union[Optimizer, dict[str, Optimizer]]
        """
        name_to_cls = {
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
        if isinstance(self.optimizer_cls, dict):
            return {
                agent_id: name_to_cls[cls_name]
                for agent_id, cls_name in self.optimizer_cls.items()
            }

        return name_to_cls[self.optimizer_cls]


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
    :type dtype: Union[type[float], type[int], type[np.ndarray]]
    :param value: The current value of the hyperparameter. Default is None.
    :type value: Optional[Union[Number, np.ndarray]]
    """

    min: float
    max: float
    shrink_factor: float = 0.8
    grow_factor: float = 1.2
    dtype: Union[type[float], type[int], type[np.ndarray]] = float
    value: Optional[Union[Number, np.ndarray]] = field(default=None, init=False)

    def mutate(self) -> Union[Number, np.ndarray]:
        """Mutate the hyperparameter value by either growing or shrinking it.

        For scalar values (int/float), the mutation applies the grow/shrink factor uniformly.
        For numpy arrays, the mutation is applied element-wise, with proper broadcasting
        of min/max constraints and preservation of the original array's dtype.

        :return: The mutated hyperparameter value.
        :rtype: Union[Number, np.ndarray]
        """
        assert self.value is not None, "Hyperparameter value is not set"

        # Equal probability of growing or shrinking
        if torch.rand(1).item() < 0.5:
            # Shrinking
            if isinstance(self.value, np.ndarray):
                new_value = np.where(
                    self.value * self.shrink_factor > self.min,
                    self.value * self.shrink_factor,
                    self.min,
                )
            else:
                if self.value * self.shrink_factor > self.min:
                    new_value = self.value * self.shrink_factor
                else:
                    new_value = self.min
        else:
            # Growing
            if isinstance(self.value, np.ndarray):
                new_value = np.where(
                    self.value * self.grow_factor < self.max,
                    self.value * self.grow_factor,
                    self.max,
                )
            else:
                if self.value * self.grow_factor < self.max:
                    new_value = self.value * self.grow_factor
                else:
                    new_value = self.max

        # Clip the new value to the min and max
        if isinstance(new_value, np.ndarray):
            new_value = np.clip(new_value, self.min, self.max)
        else:
            new_value = min(max(new_value, self.min), self.max)

        # Cast the new value to the correct dtype
        if isinstance(new_value, np.ndarray):
            # Preserve the original array's dtype
            new_value = new_value.astype(self.value.dtype)
            self.value = new_value
        else:
            self.value = self.dtype(new_value)

        return self.value


class HyperparameterConfig:
    """Stores the RL hyperparameters that will be mutated during training. For each
    hyperparameter, we store the name of the attribute where the hyperparameter is
    stored, and the range of values that the hyperparameter can take."""

    def __init__(self, **kwargs: dict[str, RLParameter]):
        self.config = kwargs
        for key, value in kwargs.items():
            if not isinstance(value, RLParameter):
                raise ValueError(
                    "Expected RLParameter object for hyperparameter configuration."
                )

            setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            "HyperparameterConfig(\n"
            + "\n".join([f"{key}: {value}" for key, value in self.config.items()])
            + "\n)"
        )

    def __bool__(self) -> bool:
        """Returns False if the config is empty, True otherwise.

        :return: Whether the config contains any hyperparameters
        :rtype: bool
        """
        return bool(self.config)

    def __eq__(self, other: "HyperparameterConfig") -> bool:
        return set(self.names()) == set(other.names())

    def __iter__(self):
        return iter(self.config)

    def __getitem__(self, key: str) -> RLParameter:
        return self.config[key]

    def names(self) -> list[str]:
        return list(self.config.keys())

    def items(self) -> dict[str, Any]:
        return self.config.items()

    def sample(self) -> tuple[str, RLParameter]:
        """Sample a hyperparameter from the configuration.

        :return: The name of the hyperparameter and its configuration.
        :rtype: tuple[str, RLParameter]
        """
        key = torch.randperm(len(self.config))[0]
        return list(self.config.keys())[key], list(self.config.values())[key]


@dataclass
class NetworkGroup:
    """Dataclass for storing a group of networks. This consists of an evaluation network (i.e.
    a network that is optimized during training) and, optionally, some other networks that
    share parameters with the evaluation network (e.g. the target network in DQN). If the
    networks are passed as an agilerl.modules.base.ModuleDict, we assume that the networks
    are part of a multiagent setting.

    :param eval_network: The evaluation network.
    :type eval_network: NetworkType
    :param shared_networks: The list of shared networks.
    :type shared_networks: Optional[NetworkType]
    :param policy: Whether the network is a policy (e.g. the network used to get the actions
        of the agent). There must be one network group in an algorithm which sets this to True.
        Default is False.
    :type policy: bool
    """

    eval_network: NetworkType
    shared_networks: Optional[NetworkType] = field(default=None)
    policy: bool = field(default=False)

    def __post_init__(self):
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
                    f"shared argument in the network group. Found {type(self.shared_networks[0])}."
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

    def _infer_parent_container(self) -> EvolvableAlgorithm:
        """
        Infer the parent container dynamically using the stack frame.

        :return: The parent container object
        :rtype: EvolvableAlgorithm
        """
        # NOTE: Here the assumption is that NetworkGroup is used inside the __init__
        # method of the implemented algorithm, such that we can access the defined locals
        # and extract the corresponding attribute names to the passed networks.
        current_frame = inspect.currentframe()
        return current_frame.f_back.f_back.f_back.f_locals["self"]

    def _infer_attribute_names(
        self, container: object, objects: Union[object, list[object]]
    ) -> list[str]:
        """
        Infer attribute names of the networks being optimized.

        :param container: The container object to inspect.
        :type container: object
        :param objects: The objects to match.
        :type objects: Union[object, list[object]]

        :return: List of attribute names for the networks
        :rtype: list[str]
        """

        def _match_condition(attr_value: Any) -> bool:
            if isinstance(objects, list):
                return any(id(attr_value) == id(obj) for obj in objects)
            else:
                return id(attr_value) == id(objects)

        return [
            attr_name
            for attr_name, attr_value in vars(container).items()
            if _match_condition(attr_value)
        ]


def make_network_group(
    eval_network: str,
    shared_networks: Optional[Union[str, list[str]]],
    policy: bool = False,
) -> NetworkGroup:
    """Make a network group from a given eval network and, optionally, some network/s that
    share parameters with the eval network.

    :param eval_network: The evaluation network.
    :type eval_network: str
    :param shared_networks: The list of shared networks.
    :type shared_networks: Optional[Union[str, list[str]]]
    :param policy: Whether the network is a policy (e.g. the network used to get the actions
    of the agent). There must be one network group in an algorithm which sets this to True.
    Default is False.
    :type policy: bool

    :return: NetworkGroup object with the passed configuration.
    :rtype: NetworkGroup
    """
    return NetworkGroup(
        eval_network=eval_network, shared_networks=shared_networks, policy=policy
    )


@dataclass
class MutationRegistry:
    """Registry to keep track of the components of an algorithms that may evolve during training
    in a structured way to be interpreted by a :class:`Mutations <agilerl.hpo.mutations.Mutations>` object
    when performing evolutionary hyperparameter optimization. This includes:

    1. The hyperparameter configuration of the algorithm.
    2. The network groups of the algorithm.
    3. The optimizers of the algorithm.
    4. The mutation hooks of the algorithm (i.e. functions that are called after a mutation is performed).

    :param hp_config: The hyperparameter configuration of the algorithm.
    :type hp_config: HyperparameterConfig
    """

    hp_config: Optional[HyperparameterConfig] = field(default=None)

    def __post_init__(self):
        self.groups: list[NetworkGroup] = []
        self.optimizers: list[OptimizerConfig] = []
        self.hooks: list[Callable] = []

        if self.hp_config is None:
            self.hp_config = HyperparameterConfig()

    def __repr__(self) -> str:
        groups_str = "\n".join(
            [
                f"Eval: '{group.eval_network}', Shared: {group.shared_networks}"
                for group in self.groups
            ]
        )
        optimizers_str = "\n".join(
            [
                f"{opt.optimizer_cls}: '{opt.name}', Networks: {opt.networks}"
                for opt in self.optimizers
            ]
        )
        return f"Network Groups:\n{groups_str}\n\nOptimizers:\n{optimizers_str}"

    def __eq__(self, other: Optional["MutationRegistry"]) -> bool:
        """Check if two MutationRegistry objects are equal. This involves checking
        that the network groups and optimizer configurations are the same.

        :param other: The other MutationRegistry object to compare with.
        :type other: Optional[MutationRegistry]

        :return: True if the two MutationRegistry objects are equal, False otherwise.
        :rtype: bool
        """
        return self.groups == other.groups and self.optimizers == other.optimizers

    @property
    def optimizer_networks(self) -> dict[str, list[str]]:
        """Get a dictionary of optimizer names and the network attribute names that they update.

        :return: A dictionary of optimizer names and the network attribute names that they update.
        :rtype: dict[str, list[str]]
        """
        return {config.name: config.networks for config in self.optimizers}

    def policy(self, return_group: bool = False) -> Optional[Union[str, NetworkGroup]]:
        """Get the name of the policy network in the registry.

        :param return_group: Whether to return the network group instead of just the name.
        :type return_group: bool

        :return: The name of the policy network in the registry.
        :rtype: Optional[Union[str, NetworkGroup]]
        """
        for group in self.groups:
            if group.policy:
                return group.eval_network if not return_group else group
        return

    def all_registered(self) -> list[str]:
        """Returns all of the members in the registry.

        :return: A list of all the members in the registry.
        :rtype: list[str]
        """
        all_registered = {group.eval_network for group in self.groups}
        all_registered.update(
            shared
            for group in self.groups
            if group.shared_networks is not None
            for shared in (
                group.shared_networks
                if isinstance(group.shared_networks, list)
                else [group.shared_networks]
            )
        )
        all_registered.update(opt.name for opt in self.optimizers)
        return all_registered

    def networks(self) -> list[NetworkConfig]:
        """Get a list of network configurations in the registry.

        :return: A list of network configurations in the registry. This includes the evaluation
        and shared networks.
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
                name=group.eval_network,
                eval_network=True,
                optimizer=optimizer_eval.get(group.eval_network),
            )
            for group in self.groups
        ]
        shared_networks = [
            NetworkConfig(name=shared, eval_network=False)
            for group in self.groups
            if group.shared_networks is not None
            for shared in (
                group.shared_networks
                if isinstance(group.shared_networks, list)
                else [group.shared_networks]
            )
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

    def register_hook(self, hook: Callable) -> None:
        """Register a hook in the registry as its name. This is used to store the names of the
        mutation hooks that will be applied after a mutation is performed.

        :param hook: The hook to be registered.
        :type hook: Callable
        """
        self.hooks.append(hook.__name__)
