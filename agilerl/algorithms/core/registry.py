import inspect
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.optim import Optimizer

from agilerl.protocols import EvolvableAlgorithm, EvolvableModule


@dataclass
class NetworkConfig:
    """Dataclass for storing the configuration of an evolvable network
    within an `EvolvableAlgorithm`.

    :param name: The name of the attribute where the network is stored.
    :type name: str
    :param eval: Whether the network is an evaluation network. This implies
    that the network is optimized during training. Default is False.

    :type eval: bool
    :param optimizer: The name of the optimizer that updates the network.
    :type optimizer: Optional[str]
    """

    name: str
    eval: bool = field(default=False)
    optimizer: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.eval and self.optimizer is None:
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
    :type networks: List[str]
    :param optimizer_cls: The optimizer class to be used.
    :type optimizer_cls: Type[Optimizer]
    :param optimizer_kwargs: The keyword arguments to be passed to the optimizer.
    :type optimizer_kwargs: Dict[str, Any]
    """

    name: str
    networks: Union[str, List[str]]
    lr: str
    optimizer_cls: Union[Type[Optimizer], List[Type[Optimizer]]]
    optimizer_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]]
    multiagent: bool = field(default=False)

    def __post_init__(self):
        # Save optimizer_cls as string for serialization
        if isinstance(self.optimizer_cls, list):
            self.optimizer_cls = [cls.__name__ for cls in self.optimizer_cls]
        else:
            self.optimizer_cls = self.optimizer_cls.__name__

    def get_optimizer_cls(self) -> Union[Optimizer, List[Optimizer]]:
        """Get the optimizer object/s from the stored configuration.

        :return: The optimizer object/s from the stored configuration.
        :rtype: Union[Optimizer, List[Optimizer]]
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
        }
        if isinstance(self.optimizer_cls, list):
            return [name_to_cls[cls_name] for cls_name in self.optimizer_cls]

        return name_to_cls[self.optimizer_cls]


@dataclass
class RLParameter:
    """Dataclass for storing the configuration of a hyperparameter that will be mutated during
    training. The hyperparameter is defined by a range of values that it can take, and the
    shrink and grow factors that will be used to mutate the hyperparameter value.

    :param min: The minimum value that the hyperparameter can take.
    :type min: float
    :param max: The maximum value that the hyperparameter can take.
    :type max: float
    :param shrink_factor: The factor by which the hyperparameter will be shrunk during mutation. Default is 0.8.
    :type shrink_factor: float
    :param grow_factor: The factor by which the hyperparameter will be grown during mutation. Default is 1.2.
    :type grow_factor: float
    :param dtype: The data type of the hyperparameter. Default is float.
    :type dtype: Union[Type[float], Type[int]]
    :param value: The current value of the hyperparameter. Default is None.
    :type value: Optional[Number]
    """

    min: float
    max: float
    shrink_factor: float = 0.8
    grow_factor: float = 1.2
    dtype: Union[Type[float], Type[int]] = float
    value: Optional[Number] = field(default=None, init=False)

    def mutate(self) -> Number:
        """Mutate the hyperparameter value by either growing or shrinking it.

        :return: The mutated hyperparameter value.
        :rtype: Number
        """
        assert self.value is not None, "Hyperparameter value is not set"

        # Equal probability of growing or shrinking
        if torch.rand(1).item() < 0.5:
            if self.value * self.shrink_factor > self.min:
                new_value = self.value * self.shrink_factor
            else:
                new_value = self.min
        else:
            if self.value * self.grow_factor < self.max:
                new_value = self.value * self.grow_factor
            else:
                new_value = self.max

        new_value = min(max(new_value, self.min), self.max)
        self.value = self.dtype(new_value)
        return self.value


class HyperparameterConfig:
    """Stores the RL hyperparameters that will be mutated during training. For each
    hyperparameter, we store the name of the attribute where the hyperparameter is
    stored, and the range of values that the hyperparameter can take."""

    def __init__(self, **kwargs: Dict[str, RLParameter]):
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

    def __iter__(self):
        return iter(self.config)

    def __getitem__(self, key: str) -> RLParameter:
        return self.config[key]

    def names(self) -> List[str]:
        return list(self.config.keys())

    def items(self) -> Dict[str, Any]:
        return self.config.items()

    def sample(self) -> Tuple[str, RLParameter]:
        """Sample a hyperparameter from the configuration.

        :return: The name of the hyperparameter and its configuration.
        :rtype: Tuple[str, RLHyperparameter]
        """
        key = torch.randperm(len(self.config))[0]
        return list(self.config.keys())[key], list(self.config.values())[key]


@dataclass
class NetworkGroup:
    """Dataclass for storing a group of networks. This consists of an evaluation network (i.e.
    a network that is optimized during training) and, optionally, some other networks that
    share parameters with the evaluation network (e.g. the target network in DQN).

    :param eval: The evaluation network.
    :type eval: str
    :param shared: The list of shared networks.
    :type shared: str, List[str]
    :param policy: Whether the network is a policy (e.g. the network used to get the actions
        of the agent). There must be one network group in an algorithm which sets this to True.
        Default is False.
    :type policy: bool
    :param multiagent: Whether the network group is used in a multiagent setting. Default is False.
    :type multiagent: bool
    """

    eval: EvolvableModule
    shared: Optional[Union[EvolvableModule, List[EvolvableModule]]] = field(
        default=None
    )
    policy: bool = field(default=False)
    multiagent: bool = field(default=False)

    def __post_init__(self):
        if self.multiagent:
            assert isinstance(self.eval, list) and isinstance(
                self.eval[0], EvolvableModule
            ), (
                "Multiagent algorithms should specify a list of EvolvableModule objects "
                "for the evaluation argument in the network group.",
            )

            if self.shared is not None:
                assert isinstance(self.shared, list), (
                    "Multiagent algorithms should specify a list of EvolvableModule objects "
                    "for the shared argument in the network group.",
                )

        # Identify the names of the attributes where the networks are stored
        container = self._infer_parent_container()
        eval = self.eval if isinstance(self.eval, list) else [self.eval]
        self.eval = self._infer_attribute_names(container, eval)[0]
        if self.shared is not None:
            shared = self.shared if isinstance(self.shared, list) else [self.shared]

            if self.multiagent and isinstance(shared[0], list):
                self.shared = [
                    self._infer_attribute_names(container, shared) for shared in shared
                ]
            else:
                assert isinstance(
                    shared[0], EvolvableModule
                ), "Expected a list of EvolvableModule objects for the shared argument in the network group."

                self.shared = self._infer_attribute_names(container, shared)

    def _infer_parent_container(self) -> EvolvableAlgorithm:
        """
        Infer the parent container dynamically using the stack frame.

        :return: The parent container object
        """
        # NOTE: Here the assumption is that NetworkGroup is used inside the __init__
        # method of the implemented algorithm, such that we can access the defined locals
        # and extract the corresponding attribute names to the passed networks.
        current_frame = inspect.currentframe()
        return current_frame.f_back.f_back.f_back.f_locals["self"]

    def _infer_attribute_names(
        self, container: object, objects: List[object]
    ) -> List[str]:
        """
        Infer attribute names of the networks being optimized.

        :return: List of attribute names for the networks
        """

        def _match_condition(attr_value: Any) -> bool:
            if not self.multiagent:
                return any(id(attr_value) == id(net) for net in objects)
            return id(attr_value) == id(objects)

        return [
            attr_name
            for attr_name, attr_value in vars(container).items()
            if _match_condition(attr_value)
        ]


def make_network_group(
    eval: str, shared: Optional[Union[str, List[str]]], policy: bool = False
) -> NetworkGroup:
    """Make a network group from a given eval network and, optionally, some network/s that
    share parameters with the eval network.

    :param eval: The evaluation network.
    :type eval: str
    :param shared: The list of shared networks.
    :type shared: str, List[str]
    :param policy: Whether the network is a policy (e.g. the network used to get the actions
    of the agent). There must be one network group in an algorithm which sets this to True.
    Default is False.
    :type policy: bool

    :return: NetworkGroup object with the passed configuration.
    :rtype: NetworkGroup
    """
    return NetworkGroup(eval=eval, shared=shared, policy=policy)


@dataclass
class MutationRegistry:
    """Registry for storing the evolvable modules and optimizers of an `EvolvableAlgorithm`
    in a structured way to be interpreted by a `Mutations` object when performing evolutionary
    hyperparameter optimization.

    :param hp_config: The hyperparameter configuration of the algorithm.
    :type hp_config: HyperparameterConfig"""

    hp_config: HyperparameterConfig

    def __post_init__(self):
        self.groups: List[NetworkGroup] = []
        self.optimizers: List[OptimizerConfig] = []
        self.hooks: List[Callable] = []

    def __repr__(self) -> str:
        groups_str = "\n".join(
            [f"Eval: '{group.eval}', Shared: {group.shared}" for group in self.groups]
        )
        optimizers_str = "\n".join(
            [
                f"{opt.optimizer_cls}: '{opt.name}', Networks: {opt.networks}"
                for opt in self.optimizers
            ]
        )
        return f"Network Groups:\n{groups_str}\n\nOptimizers:\n{optimizers_str}"

    def __eq__(self, other: "MutationRegistry") -> bool:
        return (
            self.hp_config == other.hp_config
            and self.groups == other.groups
            and self.optimizers == other.optimizers
            and self.hooks == other.hooks
        )

    @property
    def optimizer_networks(self) -> Dict[str, List[str]]:
        """Get a dictionary of optimizer names and the network attribute names that they update.

        :return: A dictionary of optimizer names and the network attribute names that they update.
        :rtype: Dict[str, List[str]]
        """
        return {config.name: config.networks for config in self.optimizers}

    @property
    def policy(self) -> Optional[str]:
        """Get the name of the policy network in the registry.

        :return: The name of the policy network in the registry.
        :rtype: Optional[str]
        """
        for group in self.groups:
            if group.policy:
                return group.eval
        return None

    def all_registered(self) -> List[str]:
        """Returns all of the members in the registry."""
        all_registered = {group.eval for group in self.groups}
        all_registered.update(
            shared
            for group in self.groups
            if group.shared is not None
            for shared in (
                group.shared if isinstance(group.shared, list) else [group.shared]
            )
        )
        all_registered.update(opt.name for opt in self.optimizers)
        return all_registered

    def networks(self) -> List[NetworkConfig]:
        """Get a list of network configurations in the registry.

        :return: A list of network configurations in the registry.
        :rtype: List[NetworkConfig]
        """
        # Match with optimizers (only eval networks can have optimizers by definition)
        optimizer_eval = {}
        for opt_name, nets in self.optimizer_networks.items():
            for net in nets:
                optimizer_eval[net] = opt_name

        # Fetch evaluation and shared networks
        eval_networks = [
            NetworkConfig(
                name=group.eval, eval=True, optimizer=optimizer_eval.get(group.eval)
            )
            for group in self.groups
        ]
        shared_networks = [
            NetworkConfig(name=shared, eval=False)
            for group in self.groups
            if group.shared is not None
            for shared in (
                group.shared if isinstance(group.shared, list) else [group.shared]
            )
        ]

        return eval_networks + shared_networks

    def register_group(self, group: NetworkGroup) -> None:
        """Register a network configuration in the registry.

        :param config: The network configuration to be registered.
        :type config: NetworkConfig
        """
        self.groups.append(group)

    def register_optimizer(self, optimizer: OptimizerConfig) -> None:
        """Register an optimizer configuration in the registry.

        :param config: The optimizer configuration to be registered.
        :type config: OptimizerConfig
        """
        self.optimizers.append(optimizer)

    def register_hook(self, hook: Callable) -> None:
        """Register a hook in the registry as its name.

        :param hook: The hook to be registered.
        :type hook: Callable
        """
        self.hooks.append(hook.__name__)
