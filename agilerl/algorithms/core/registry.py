from typing import Optional, List, Dict, Union, Type
from dataclasses import dataclass, field
from torch.optim import Optimizer

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
    """
    optimizer_cls: Type[Optimizer]
    name: str
    networks: List[str] = field(default_factory=list)


@dataclass
class NetworkGroup:
    """Dataclass for storing a group of networks. This consists of an evaluation network (i.e. 
    a network that is optimized during training) and, optionally, some other networks that 
    share parameters with the evaluation network (e.g. the target network in DQN).
    
    :param eval: The evaluation network.
    :type eval: str
    :param shared: The list of shared networks.
    :type shared: str, List[str]
    """
    eval: str
    shared: Optional[Union[str, List[str]]] = field(default=None)

def make_network_group(eval: str, shared: Optional[Union[str, List[str]]]) -> NetworkGroup:
    """Make a network group from a given eval network and, optionally, some network/s that 
    share parameters with the eval network.

    :param eval: The evaluation network.
    :type eval: str
    :param shared: The list of shared networks.
    :type shared: str, List[str]

    :return: NetworkGroup object with the passed configuration.
    :rtype: NetworkGroup
    """
    return NetworkGroup(eval=eval, shared=shared)


@dataclass
class Registry:
    """Registry for storing the evolvable modules and optimizers of an `EvolvableAlgorithm`
    in a structured way to be interpreted by a `Mutations` object when performing evolutionary 
    hyperparameter optimization."""

    def __post_init__(self):
        self.groups: List[NetworkGroup] = []
        self.optimizers: List[OptimizerConfig] = []

    def __repr__(self) -> str:
        groups_str = "\n".join(
            [
                f"Eval: '{group.eval}', Shared: {group.shared}" 
                for group in self.groups
            ]
            )
        optimizers_str = "\n".join(
            [
                f"{opt.optimizer_cls.__name__}: '{opt.name}', Networks: {opt.networks}" 
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
    
    def all_registered(self) -> List[str]:
        """Returns all of the members in the registry."""
        all_registered = {group.eval for group in self.groups}
        all_registered.update(
            shared for group in self.groups if group.shared is not None
            for shared in (group.shared if isinstance(group.shared, list) else [group.shared])
        )
        all_registered.update(opt.name for opt in self.optimizers)
        return all_registered

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
