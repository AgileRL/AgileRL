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
            network_names: Optional[List[str]] = None,
            multiagent: bool = False
            ) -> None:
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.multiagent = multiagent

        if isinstance(networks, nn.Module):
            self.networks = [networks]
        else:
            self.networks = networks

        # NOTE: This should be passed when reintializing the optimizer
        # when mutating an individual.
        if network_names is not None:
            self.network_names = network_names
        else:
            parent_container = self._infer_parent_container()
            self.network_names = self._infer_network_attr_names(parent_container)

        assert self.network_names, "No networks found in the parent container."

        # Initialize the optimizer/s
        # NOTE: For multi-agent algorithms, we want to have a different optimizer for each of the 
        # networks.
        multiple_attrs = len(self.network_names) > 1
        multiple_networks = len(self.networks) > 1
        if multiagent:
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
    
    def __iter__(self):
        return iter(self.optimizer)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
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

    def _infer_network_attr_names(self, container: Any) -> List[str]:
        """
        Infer attribute names of the networks being optimized.

        :return: List of attribute names for the networks
        """
        def _match_condition(attr_value: Any) -> bool:
            if not self.multiagent:
                return any(id(attr_value) == id(net) for net in self.networks)
            return id(attr_value) == id(self.networks)
    
        return [
            attr_name for attr_name, attr_value in vars(container).items()
            if _match_condition(attr_value)
        ]
    

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state of the optimizer from the passed state dictionary.

        :param state_dict: State dictionary of the optimizer.
        :type state_dict: Dict[str, Any]
        """
        if self.multiagent:
            raise TypeError(
                "Loading optimizer state dict directly is not supported for multi-agent optimizers."
                )
        
        self.optimizer.load_state_dict(state_dict)
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state of the optimizer as a dictionary.

        :return: State dictionary of the optimizer.
        :rtype: Dict[str, Any]
        """
        if self.multiagent:
            raise TypeError(
                "Getting optimizer state dict directly is not supported for multi-agent optimizers."
                )
        
        return self.optimizer.state_dict()
