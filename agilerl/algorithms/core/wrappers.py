from typing import Any, Dict, List, Union
from torch.optim import Optimizer
import torch.nn as nn


class OptimizerWrapper:
    """Wrapper to initialize optimizer and store metadata relevant for 
    evolutionary hyperparameter optimization.
    
    :param optimizer_cls: The optimizer class to be initialized.
    :type optimizer_cls: Type[torch.optim.Optimizer]
    :param networks: The list of networks that the optimizer will update.
    :type networks: List[EvolvableModule]
    :param optimizer_kwargs: The keyword arguments to be passed to the optimizer.
    :type optimizer_kwargs: Dict[str, Any]
    """
    def __init__(
            self,
            optimizer_cls: Optimizer,
            networks: Union[nn.Module, List[nn.Module]],
            network_attr_names: Union[str, List[str]],
            optimizer_kwargs: Dict[str, Any]):
        
        self.optimizer_cls = optimizer_cls
        self.networks = networks
        self.optimizer_kwargs = optimizer_kwargs
        self.network_names = network_attr_names if isinstance(network_attr_names, list) else [network_attr_names]

        if isinstance(networks, nn.Module):
            assert (
                isinstance(network_attr_names, str),
                "If networks is a single network, network_attr_names must be a string."
            )
            networks = [networks]
            network_attr_names = [network_attr_names]

        # Initialize the optimizer
        optimizer_kwargs = [{"params": net.parameters(), **optimizer_kwargs} for net in networks]
        self.optimizer = optimizer_cls(optimizer_kwargs)

        # Wrap all of the optimizer methods and attributes
        for attr in dir(self.optimizer):
            if not attr.startswith("_"):
                setattr(self, attr, getattr(self.optimizer, attr))

