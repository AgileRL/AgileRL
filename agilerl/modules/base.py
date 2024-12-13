from typing import Any, Dict, List, Callable, Optional
import copy
from functools import wraps
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from agilerl.protocols import MutationType, MutationMethod
from agilerl.modules.custom_components import NoisyLinear
from agilerl.utils.evolvable_networks import get_activation

def register_mutation_fn(mutation_type: MutationType) -> Callable[[Callable], MutationMethod]:
    """Decorator to register a method as a mutation function of a specific type.
    
    :param mutation_type: The type of mutation function.
    :type mutation_type: MutationType
    :return: The decorator function.
    :rtype: Callable[[Callable], MutationMethod]
    """
    def decorator(func: Callable[[Any], Optional[Dict[str, Any]]]) -> MutationMethod:
        f"""{func.__doc__}"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        
        # Explicitly set the mutation type attribute on the wrapper function
        wrapper._mutation_type = mutation_type
        return wrapper
    
    return decorator

class EvolvableModule(nn.Module, ABC):
    """Base class for evolvable neural networks. Provides methods that allow for 
    seamless network mutations."""

    model: nn.Module

    def __init__(self, device: str) -> None:
        nn.Module.__init__(self)
        self._init_mutation_methods()

        self.device = device

    @property
    @abstractmethod
    def init_dict(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "init_dict property must be implemented in order to store the configuration of the evolvable module."
            )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("forward method must be implemented in order to use the evolvable module.")
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    @staticmethod
    def preserve_parameters(old_net: nn.Module, new_net: nn.Module) -> nn.Module:
        """Returns new neural network with copied parameters from old network. Specifically, it 
        handles tensors with different sizes by copying the minimum number of elements.

        :param old_net: Old neural network
        :type old_net: nn.Module
        :param new_net: New neural network
        :type new_net: nn.Module
        :return: New neural network with copied parameters
        :rtype: nn.Module
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                old_param = old_net_dict[key]
                old_size = old_param.data.size()
                new_size = param.data.size()

                if old_size == new_size:
                    # If the sizes are the same, just copy the parameter
                    param.data = old_param.data
                else:
                    if "norm" not in key:
                        # Create a slicing index to handle tensors with varying sizes
                        slice_index = tuple(slice(0, min(o, n)) for o, n in zip(old_size, new_size))
                        param.data[slice_index] = old_param.data[slice_index]

        return new_net

    @staticmethod
    def shrink_preserve_parameters(old_net: nn.Module, new_net: nn.Module) -> nn.Module:
        """Returns shrunk new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module
        :param new_net: New neural network
        :type new_net: nn.Module
        :return: Shrunk new neural network with copied parameters
        :rtype: nn.Module
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if "norm" not in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        min_0 = min(old_size[0], new_size[0])
                        if len(param.data.size()) == 1:
                            param.data[:min_0] = old_net_dict[key].data[:min_0]
                        else:
                            min_1 = min(old_size[1], new_size[1])
                            param.data[:min_0, :min_1] = old_net_dict[key].data[:min_0, :min_1]

        return new_net

    @staticmethod
    def reset_noise(*networks: nn.Module) -> None:
        """Reset noise for all NoisyLinear layers in the network.
        
        param networks: The networks to reset noise for.
        :type networks: nn.Module
        """
        for net in networks:
            for layer in net.modules():
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()
    
    def _init_mutation_methods(self) -> None:
        # Initialize dictionaries to store mutation methods by type
        self._layer_mutation_methods = {}
        self._node_mutation_methods = {}
        for name, method in vars(self.__class__).items():
            if isinstance(method, MutationMethod):
                if method._mutation_type == MutationType.LAYER:
                    self._layer_mutation_methods[name] = method
                elif method._mutation_type == MutationType.NODE:
                    self._node_mutation_methods[name] = method
        
        self._mutation_methods = (
            list(self._layer_mutation_methods.values()) +  
            list(self._node_mutation_methods.values())
        )

    def get_mutation_methods(self) -> Dict[str, MutationMethod]:
        """Get all mutation methods.

        return: A dictionary of mutation methods.
        """
        return {method.__name__: method for method in self._mutation_methods}
    
    def filter_mutation_methods(self, remove: str) -> None:
        """Filter out mutation methods that contain the specified string in their name.
        
        param remove: The string to remove.
        type remove: str
        """
        self._layer_mutation_methods = {
            name: method for name, method in self._layer_mutation_methods.items() if remove not in name
        }
        self._node_mutation_methods = {
            name: method for name, method in self._node_mutation_methods.items() if remove not in name
        }
        self._mutation_methods = [
            method for method in self._mutation_methods if remove not in method.__name__
        ]

    def get_mutation_probs(self, new_layer_prob: float) -> List[float]:
        """Get the mutation probabilities for each mutation method.
        
        param new_layer_prob: The probability of selecting a layer mutation method.
        type new_layer_prob: float
        return: A list of probabilities for each mutation method.
        """
        num_layer_fns = len(self._layer_mutation_methods)
        num_node_fns = len(self._node_mutation_methods)

        probs = []
        for fn in self.get_mutation_methods().values():
            if fn._mutation_type == MutationType.LAYER:
                prob = new_layer_prob / num_layer_fns
            elif fn._mutation_type == MutationType.NODE:
                prob = (1 - new_layer_prob) / num_node_fns
            
            probs.append(prob)
        
        return probs

    def clone(self) -> "EvolvableModule":
        """Returns clone of neural net with identical parameters."""
        clone = self.__class__(**copy.deepcopy(self.init_dict))

        # Load state dict if the network has been trained
        if self.state_dict():
            clone.load_state_dict(self.state_dict())

        return clone