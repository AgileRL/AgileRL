from typing import Any, Dict, List, Callable, Optional, TypeVar, Iterable, Tuple
import copy
from functools import wraps
from abc import ABC, ABCMeta
from numpy.random import Generator
import numpy as np
import torch
import torch.nn as nn

from agilerl.protocols import MutationType, MutationMethod
from agilerl.modules.custom_components import NoisyLinear
from agilerl.utils.algo_utils import recursive_check_module_attrs

SelfEvolvableModule = TypeVar("SelfEvolvableModule", bound="EvolvableModule")

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

class _ModuleMeta(type):
    """Metaclass to wrap registry information after algorithm is done 
    intiializing with specified network groups and optimizers."""
    def __call__(cls, *args, **kwargs):
        # Create the instance
        instance: SelfEvolvableModule = super().__call__(*args, **kwargs)

        # Call the base class post_init_hook after all initialization
        if isinstance(instance, cls) and hasattr(instance, "_init_underlying_methods"):
            instance._init_underlying_methods()

        return instance
    
class ModuleMeta(_ModuleMeta, ABCMeta):
    ...

class EvolvableModule(nn.Module, ABC, metaclass=ModuleMeta):
    """Base class for evolvable neural networks."""

    def __init__(self, device: str) -> None:
        nn.Module.__init__(self)
        self._init_surface_methods()
        self.device = device

    @property
    def init_dict(self) -> Dict[str, Any]:
        return {"device": self.device}
    
    @property
    def mutation_methods(self) -> List[str]:
        return self._mutation_methods
    
    @property
    def layer_mutation_methods(self) -> List[str]:
        return self._layer_mutation_methods
    
    @property
    def node_mutation_methods(self) -> List[str]:
        return self._node_mutation_methods
    
    @property
    def activation(self) -> Optional[str]:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "forward method must be implemented in order to use the evolvable module."
            )
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        return self.forward(x)
    
    def change_activation(self, activation: str, output: bool) -> None:
        """Set the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        """
        raise NotImplementedError(
            "change_activation method must be implemented in order to set the activation function."
            )


    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            mut_method = self.get_mutation_methods().get(name)
            if mut_method is not None:
                return mut_method
            raise e

    
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

    def modules(self) -> Dict[str, "EvolvableModule"]:
        """Returns the attributes related to the evolvable networks in the algorithm. Includes 
        attributes that are either evolvable networks or a list of evolvable networks, as well 
        as the optimizers associated with the networks.

        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """
        def is_evolvable(attr: str, obj: Any):
            return (
                recursive_check_module_attrs(obj, networks_only=True)
                and not attr.startswith("_") and not attr.endswith("_")
            )
        # Inspect evolvable
        evolvable_attrs = {}
        for attr in dir(self):
            obj = getattr(self, attr)
            if is_evolvable(attr, obj):
                evolvable_attrs[attr] = obj

        return evolvable_attrs

    def _init_surface_methods(self) -> None:
        # Check mutation methods in class
        self._layer_mutation_methods = []
        self._node_mutation_methods = []
        for method in vars(self.__class__).values():
            if isinstance(method, MutationMethod):
                if method._mutation_type == MutationType.LAYER:
                    self._layer_mutation_methods.append(method.__name__)
                elif method._mutation_type == MutationType.NODE:
                    self._node_mutation_methods.append(method.__name__)
        
        # Check mutation methods in superclasses
        for base in self.__class__.__bases__:
            for method in vars(base).values():
                if isinstance(method, MutationMethod):
                    if method._mutation_type == MutationType.LAYER:
                        self._layer_mutation_methods.append(method.__name__)
                    elif method._mutation_type == MutationType.NODE:
                        self._node_mutation_methods.append(method.__name__)

        self._mutation_methods = (
            self._layer_mutation_methods + self._node_mutation_methods
        )
    
    def _init_underlying_methods(self) -> None:
        # After module has been initialized, we can identify 
        # any additional methods of underlying EvolvableModule objects
        # and add them to the registry
        layer_fns = []
        node_fns = []
        for attr, module in self.modules().items():
            for name, method in module.get_mutation_methods().items():
                method_name = ".".join([attr, name])
                method_type = method._mutation_type
                if method_type == MutationType.LAYER:
                    layer_fns.append(method_name)
                elif method_type == MutationType.NODE:
                    node_fns.append(method_name)
                else:
                    raise ValueError(f"Invalid mutation type: {method_type}")
        
        extra_methods = layer_fns + node_fns
        self._mutation_methods += extra_methods
        self._layer_mutation_methods += layer_fns
        self._node_mutation_methods += node_fns

    def get_mutation_methods(self) -> Dict[str, MutationMethod]:
        """Get all mutation methods for the network.

        :return: A dictionary of mutation methods.
        :rtype: Dict[str, MutationMethod]
        """
        def get_method_from_name(name: str) -> MutationMethod:
            if "." not in name:
                return getattr(self, name)

            attr = name.split(".")[0]
            method = ".".join(name.split(".")[1:])
            return getattr(getattr(self, attr), method)

        return {name: get_method_from_name(name) for name in self.mutation_methods}
    
    def filter_mutation_methods(self, remove: str) -> None:
        """Filter out mutation methods that contain the specified string in their name.
        
        param remove: The string to remove.
        type remove: str
        """
        def filter_methods(methods: List[str]) -> List[str]:
            return [method for method in methods if remove not in method]
        
        self._layer_mutation_methods = filter_methods(self._layer_mutation_methods)
        self._node_mutation_methods = filter_methods(self._node_mutation_methods)
        self._mutation_methods = filter_methods(self._mutation_methods)

    def get_mutation_probs(self, new_layer_prob: float) -> List[float]:
        """Get the mutation probabilities for each mutation method.
        
        param new_layer_prob: The probability of selecting a layer mutation method.
        type new_layer_prob: float
        return: A list of probabilities for each mutation method.
        rtype: List[float]
        """
        num_layer_fns = len(self.layer_mutation_methods)
        num_node_fns = len(self.node_mutation_methods)

        probs = []
        for fn in self.get_mutation_methods().values():
            if fn._mutation_type == MutationType.LAYER:
                prob = new_layer_prob / num_layer_fns
            elif fn._mutation_type == MutationType.NODE:
                prob = (1 - new_layer_prob) / num_node_fns
            
            probs.append(prob)
        
        return probs
    
    def sample_mutation_method(self, new_layer_prob: float, rng: Optional[Generator] = None) -> MutationMethod:
        """Sample a mutation method based on the mutation probabilities.
        
        param new_layer_prob: The probability of selecting a layer mutation method.
        type new_layer_prob: float
        return: The sampled mutation method.
        rtype: MutationMethod
        """
        if rng is None:
            rng = np.random.default_rng()

        probs = self.get_mutation_probs(new_layer_prob)
        return rng.choice(self.mutation_methods, p=probs, size=1)

    def clone(self) -> "EvolvableModule":
        """Returns clone of neural net with identical parameters."""
        clone = self.__class__(**copy.deepcopy(self.init_dict))

        # Load state dict if the network has been trained
        if self.state_dict():
            clone.load_state_dict(self.state_dict())

        return clone
    
class ModuleDict(EvolvableModule, nn.ModuleDict):
    """Analogous to nn.ModuleDict, but allows for the recursive inheritance of the 
    mutation methods of underlying evolvable modules.
    """
    @property
    def mutation_methods(self) -> List[str]:
        return [
            f"{name}.{method}" for name, module in self.items() 
            for method in module.mutation_methods
        ]

    @property
    def layer_mutation_methods(self) -> List[str]:
        return [
            f"{name}.{method}" for name, module in self.items() 
            for method in module.layer_mutation_methods
        ]
    
    @property
    def node_mutation_methods(self) -> List[str]:
        return [
            f"{name}.{method}" for name, module in self.items() 
            for method in module.node_mutation_methods
        ]

    def values(self) -> Iterable[EvolvableModule]:
        return super().values()
    
    def items(self) -> Iterable[Tuple[str, EvolvableModule]]:
        return super().items()
    
    def get_mutation_methods(self) -> Dict[str, MutationMethod]:
        """Get all mutation methods for the network.

        :return: A dictionary of mutation methods.
        :rtype: Dict[str, MutationMethod]
        """
        def get_method_from_name(name: str) -> MutationMethod:
            if "." not in name:
                return getattr(self, name)

            key = name.split(".")[0]
            method = ".".join(name.split(".")[1:])
            return getattr(self[key], method)

        return {name: get_method_from_name(name) for name in self.mutation_methods}
    