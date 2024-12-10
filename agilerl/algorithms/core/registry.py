from typing import Optional, List, Dict, Union, Type, Any, Callable
import inspect
from dataclasses import dataclass, field
from torch.optim import Optimizer
import torch

from agilerl.protocols import EvolvableModule, EvolvableAlgorithm

@dataclass
class NetworkConfig:
    """Dataclass for storing the configuration of a en evolvable network 
    within an `EvolvableAlgorithm`.

    :param name: The name of the attribute where the network is stored.
    :type name: str
    :param eval: Whether the network is an evaluation network. This implies that the network

    :type eval: bool
    :param optimizer: The name of the optimizer that updates the network.
    :type optimizer: Optional[str] 
    """
    name: str
    eval: bool = field(default=False)
    optimizer: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.eval and self.optimizer is None:
            raise ValueError("Evaluation network must have an optimizer associated with it.") 


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
    optimizer_cls: Union[Type[Optimizer], List[Type[Optimizer]]]
    optimizer_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]]

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
            "Rprop": torch.optim.Rprop
        }
        if isinstance(self.optimizer_cls, list):
            return [name_to_cls[cls_name] for cls_name in self.optimizer_cls]
        
        return name_to_cls[self.optimizer_cls]


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
    """
    eval: EvolvableModule
    shared: Optional[Union[EvolvableModule, List[EvolvableModule]]] = field(default=None)
    policy: bool = field(default=False)
    multiagent: bool = field(default=False)

    def __post_init__(self):
        if self.multiagent:
            assert (
                isinstance(self.eval, list) and isinstance(self.eval[0], EvolvableModule),
                "Multiagent algorithms should specify a list of EvolvableModule objects "
                "for the evaluation argument in the network group."
            )
            if self.shared is not None:
                assert (
                    isinstance(self.shared, list),
                    "Multiagent algorithms should specify a list of EvolvableModule objects "
                    "for the shared argument in the network group."
                )

        # Identify the names of the attributes where the networks are stored
        container = self._infer_parent_container()
        eval = self.eval if isinstance(self.eval, list) else [self.eval]
        self.eval = self._infer_attribute_names(container, eval)[0]
        if self.shared is not None:
            shared = self.shared if isinstance(self.shared, list) else [self.shared]

            if self.multiagent and isinstance(shared[0], list):
                self.shared = [self._infer_attribute_names(container, shared) for shared in shared]
            else:
                assert (
                    isinstance(shared[0], EvolvableModule),
                    "Expected a list of EvolvableModule objects for the shared argument in the network group."
                )
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
        return current_frame.f_back.f_back.f_back.f_locals['self']

    def _infer_attribute_names(self, container: object, objects: List[object]) -> List[str]:
        """
        Infer attribute names of the networks being optimized.

        :return: List of attribute names for the networks
        """
        def _match_condition(attr_value: Any) -> bool:
            if not self.multiagent:
                return any(id(attr_value) == id(net) for net in objects)
            return id(attr_value) == id(objects)
    
        return [
            attr_name for attr_name, attr_value in vars(container).items()
            if _match_condition(attr_value)
        ]
    

def make_network_group(
        eval: str,
        shared: Optional[Union[str, List[str]]],
        policy: bool = False
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
    hyperparameter optimization."""

    def __post_init__(self):
        self.groups: List[NetworkGroup] = []
        self.optimizers: List[OptimizerConfig] = []
        self.hooks: List[Callable] = []

    def __repr__(self) -> str:
        groups_str = "\n".join(
            [
                f"Eval: '{group.eval}', Shared: {group.shared}" 
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
            shared for group in self.groups if group.shared is not None
            for shared in (group.shared if isinstance(group.shared, list) else [group.shared])
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
            NetworkConfig(name=group.eval, eval=True, optimizer=optimizer_eval.get(group.eval)) 
            for group in self.groups
            ]
        shared_networks = [
            NetworkConfig(name=shared, eval=False) for group in self.groups 
            if group.shared is not None 
            for shared in (group.shared if isinstance(group.shared, list) else [group.shared])
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

