from typing import Any, Dict, List, Union, Optional
import inspect
import torch.nn as nn
from torch.optim import Optimizer

from agilerl.typing import OptimizerType
from agilerl.protocols import EvolvableAlgorithm
from agilerl.modules.base import EvolvableModule

_Optimizer = Union[OptimizerType, List[OptimizerType]]
_Module = Union[EvolvableModule, List[EvolvableModule]]

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
    optimizer: _Optimizer

    def __init__(
            self,
            optimizer_cls: _Optimizer,
            networks: _Module,
            optimizer_kwargs: Dict[str, Any],
            network_names: Optional[List[str]] = None
            ) -> None:
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

        if isinstance(networks, nn.Module):
            self.networks = [networks]
        
        self.parent_container = self._infer_parent_container()

        # NOTE: This should be passed when reintializing the optimizer
        # when mutating an individual.
        if network_names is not None:
            self.network_names = network_names
        else:
            self.network_names = self._infer_network_attr_names()

        assert self.network_names, "No networks found in the parent container."

        # Initialize the optimizer/s
        multiple_attrs = len(self.network_names) > 1
        multiple_networks = len(self.networks) > 1

        # NOTE: For multi-agent algorithms, we expect multiple EvolvableModules to be stored 
        # in a single attribute, where we want to initialize a separate optimizer for each
        # network.
        if multiple_networks and not multiple_attrs:
            self.optimizer = []
            for i, net in enumerate(self.networks):
                optimizer = optimizer_cls[i] if isinstance(optimizer_cls, list) else optimizer_cls
                kwargs = optimizer_kwargs[i] if isinstance(optimizer_kwargs, list) else optimizer_kwargs
                self.optimizer.append(optimizer(net.parameters(), **kwargs))

        # Single-agent algorithms with multiple networks for a single optimizer
        elif multiple_networks and multiple_attrs:
            assert len(self.networks) == len(self.network_names), (
                "Number of networks and network attribute names do not match."
            )
            assert isinstance(optimizer_cls, type), (
                "Expected a single optimizer class for multiple networks."
            )
            # Initialize a single optimizer from the combination of network parameters
            opt_args = []
            for i, net in enumerate(self.networks):
                kwargs = optimizer_kwargs[i] if isinstance(optimizer_kwargs, list) else optimizer_kwargs
                opt_args.append({"params": net.parameters(), **kwargs})

            self.optimizer = optimizer_cls(opt_args)

        # Single-agent algorithms with a single network for a single optimizer
        else:
            assert isinstance(optimizer_cls, type), (
                "Expected a single optimizer class for a single network."
            )
            assert isinstance(optimizer_kwargs, dict), (
                "Expected a single dictionary of optimizer keyword arguments."
            )
            self.optimizer = optimizer_cls(self.networks[0].parameters(), **optimizer_kwargs)
    
    def __getitem__(self, index: int) -> Optimizer:
        try: 
            return self.optimizer[index]
        except TypeError:
            raise TypeError(f"Can't access item of a single {type(self.optimizer)} object.")

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.optimizer, name)

    def _infer_parent_container(self) -> EvolvableAlgorithm:
        """
        Infer the parent container dynamically using the stack frame.

        :return: The parent container object
        """
        # NOTE: Here the assumption is that OptimizerWrapper is used inside the __init__ 
        # method of the implemented algorithm, such that we can access the defined locals
        # and extract the corresponding attribute names to the passed networks.
        current_frame = inspect.currentframe()
        return current_frame.f_back.f_back.f_locals['self']

    def _infer_network_attr_names(self) -> List[str]:
        """
        Infer attribute names of the networks being optimized.

        :return: List of attribute names for the networks
        """
        return [
            attr_name for attr_name, attr_value in vars(self.parent_container).items()
            if any(id(attr_value) == id(net) for net in self.networks)
        ]
