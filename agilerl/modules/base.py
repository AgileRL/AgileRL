from typing import Any, Dict, List, Callable, Optional, TypeVar, Iterable, Tuple
import copy
import inspect
from functools import wraps
from abc import ABC, ABCMeta, abstractmethod
from numpy.random import Generator
import numpy as np
import torch
import torch.nn as nn

from agilerl.protocols import MutationType, MutationMethod
from agilerl.modules.custom_components import NoisyLinear
from agilerl.utils.algo_utils import recursive_check_module_attrs

SelfEvolvableModule = TypeVar("SelfEvolvableModule", bound="EvolvableModule")

def is_evolvable(attr: str, obj: Any) -> bool:
    """Check if an attribute of a module is evolvable.

    :param attr: The attribute name.
    :type attr: str
    :param obj: The attribute object.
    :type obj: Any
    :return: True if the attribute is evolvable, False otherwise.
    :rtype: bool
    """
    return (
        recursive_check_module_attrs(obj, networks_only=True)
        and not attr.startswith("_") and not attr.endswith("_")
    )

def register_mutation_fn(mutation_type: MutationType, **recreate_kwargs) -> Callable[[Callable], MutationMethod]:
    """Decorator to register a method as a mutation function of a specific type.
    
    :param mutation_type: The type of mutation function.
    :type mutation_type: MutationType
    :return: The decorator function.
    :rtype: Callable[[Callable], MutationMethod]
    """
    def decorator(func: Callable[[Any], Optional[Dict[str, Any]]]) -> MutationMethod:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        # Explicitly set the mutation type attribute on the wrapper function
        wrapper._mutation_type = mutation_type
        wrapper._recreate_kwargs = recreate_kwargs
        return wrapper
    
    return decorator

def mutation_wrapper(
        mod: SelfEvolvableModule,
        mut_method: MutationMethod,
        mut_attr: str
        ) -> MutationMethod:
    """Wrapper function to apply a mutation method and recreate the network.

    :param mod: The evolvable module.
    :type mod: SelfEvolvableModule
    :param mut_method: The mutation method.
    :type mut_method: MutationMethod
    :param mut_attr: The mutation attribute.
    :type mut_attr: str

    :return: The wrapped mutation method.
    :rtype: MutationMethod
    """
    @wraps(mut_method)
    def wrapped(*args, **kwargs):
        result = mut_method(*args, **kwargs)
        print(mut_attr)
        mod.last_mutation = mut_method
        mod.last_mutation_attr = mut_attr

        # Inspect the keyword arguments of `recreate_network` and match with 
        # the specified kwargs in the called mutation method
        if mut_method._recreate_kwargs:
            recreation_kwargs = inspect.signature(mod.recreate_network).parameters
            rec_kwargs = {
                k: v for k, v in mut_method._recreate_kwargs.items() if k in recreation_kwargs
                }
        else:
            rec_kwargs = {}

        mod.recreate_network(**rec_kwargs)
        return result

    return wrapped

class _ModuleMeta(type):
    """Metaclass to parse the mutation methods of an EvolvableModule instance 
    and its superclasses."""
    def __call__(cls, *args, **kwargs):
        instance: SelfEvolvableModule = super().__call__(*args, **kwargs)

        # Parse and log mutation methods from the instance
        instance._init_underlying_methods()

        return instance

class ModuleMeta(_ModuleMeta, ABCMeta):
    pass

class EvolvableModule(nn.Module, ABC, metaclass=ModuleMeta):
    """Base class for evolvable neural networks."""

    def __init__(self, device: str) -> None:
        nn.Module.__init__(self)
        self._init_surface_methods()
        self.device = device
        self._last_mutation = None
        self._last_mutation_attr = None

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
    
    @property
    def last_mutation(self) -> Optional[MutationMethod]:
        return self._last_mutation
    
    @last_mutation.setter
    def last_mutation(self, value: MutationMethod) -> None:
        self._last_mutation = value

    @property
    def last_mutation_attr(self) -> Optional[str]:
        return self._last_mutation_attr
    
    @last_mutation_attr.setter
    def last_mutation_attr(self, value: str) -> None:
        self._last_mutation_attr = value

    @abstractmethod
    def recreate_network(self, **kwargs) -> None:
        """Recreate the network after a mutation has been applied."""
        raise NotImplementedError(
            "An EvolvableModule must implement the recreate_network method, which is called after " \
            "a mutation has been applied."
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "forward method must be implemented in order to use the evolvable module."
            )
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the network."""
        return self.forward(*args, **kwargs)
    
    def change_activation(self, activation: str, output: bool) -> None:
        """Set the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        """
        raise NotImplementedError(
            "change_activation method must be implemented in order to set the activation function."
            )
    
    def __getattribute__(self, name: str) -> Any:
        """Get attribute of the network. This handles the case where a mutation method is trying to be 
        fetched from the top-most class of the network (i.e. not an underlying module).

        :param name: The name of the attribute.
        :type name: str
        :return: The attribute of the network.
        :rtype: Any
        """
        attr = super().__getattribute__(name)

        if not callable(attr):
            return attr

        return (
            mutation_wrapper(self, attr, name) 
            if isinstance(attr, MutationMethod) else attr
        )

    def __getattr__(self, name: str) -> Any:
        """Get attribute of the network. If the attribute is a mutation method, return the
        method (from an underlying EvolvableModule attribute also). Otherwise, raise an 
        AttributeError.
        
        :param name: The name of the attribute.
        :type name: str
        :return: The attribute of the network.
        :rtype: Any
        """
        try:
            attr = super().__getattr__(name)
            return (
                mutation_wrapper(self, attr, name) 
                if isinstance(attr, MutationMethod) else attr
                )
            
        except AttributeError as e:
            mut_method = self.get_mutation_methods().get(name)
            if mut_method is not None:
                return mut_method

            raise e

    def get_output_dense(self) -> Optional[nn.Module]:
        """Get the output dense layer of the network.

        :return: The output dense layer.
        :rtype: nn.Module
        """
        return
    
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
                elif "norm" not in key:
                    # Create a slicing index to handle tensors with varying sizes
                    slice_index = tuple(slice(0, min(o, n)) for o, n in zip(old_size, new_size))
                    param.data[slice_index] = old_param.data[slice_index]

        return new_net

    @staticmethod
    def reset_noise(*networks: nn.Module) -> None:
        """Reset noise for all NoisyLinear layers in the network.
        
        :param networks: The networks to reset noise for.
        :type networks: nn.Module
        """
        for net in networks:
            for layer in net.modules():
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

    @staticmethod
    def init_weights_gaussian(module: nn.Module, std_coeff: float) -> None:
        """Initialize the weights of the neural network using a Gaussian distribution.

        :param module: The neural network module.
        :type module: nn.Module
        :param std_coeff: The standard deviation coefficient.
        :type std_coeff: float
        """
        def init_weights(m):
            if isinstance(m, nn.Linear):
                hidden_size = m.weight.size(1)
                nn.init.normal_(m.weight, mean=0, std=std_coeff / hidden_size)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if isinstance(module, nn.Linear):
            init_weights(module)
            return

        layers = [m for m in module.children()]
        for layer in layers:
            init_weights(layer)
    
    def disable_mutations(self, mut_type: Optional[MutationType] = None) -> None:
        """Make the network unevolvable."""
        if mut_type is None:
            self._layer_mutation_methods = []
            self._node_mutation_methods = []
            self._mutation_methods = []
        elif mut_type == MutationType.LAYER:
            self._mutation_methods = [
                method for method in self._mutation_methods if method not in self._layer_mutation_methods
                ]
            self._layer_mutation_methods = []
        elif mut_type == MutationType.NODE:
            self._mutation_methods = [
                method for method in self._mutation_methods if method not in self._node_mutation_methods
                ]
            self._node_mutation_methods = []
        else:
            raise ValueError(f"Invalid mutation type: {mut_type}")

    def modules(self) -> Dict[str, "EvolvableModule"]:
        """Returns the attributes related to the evolvable modules in the algorithm. Includes 
        attributes that are either evolvable modules or a list of evolvable modules, as well 
        as the optimizers associated with the networks.

        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """
        # Inspect evolvable
        evolvable_attrs = {}
        for attr in dir(self):
            obj = getattr(self, attr)
            if is_evolvable(attr, obj):
                evolvable_attrs[attr] = obj

        return evolvable_attrs

    def _init_surface_methods(self) -> None:
        # Check mutation methods in class
        layer_methods = []
        node_methods = []
        for method in vars(self.__class__).values():
            if isinstance(method, MutationMethod):
                if method._mutation_type == MutationType.LAYER:
                    layer_methods.append(method.__name__)
                elif method._mutation_type == MutationType.NODE:
                    node_methods.append(method.__name__)
        
        # Check mutation methods in superclasses
        def check_base_methods(cls) -> None:
            for base in cls.__bases__:
                if base is EvolvableModule:
                    return
                for method in vars(base).values():
                    if isinstance(method, MutationMethod):
                        if method._mutation_type == MutationType.LAYER:
                            layer_methods.append(method.__name__)
                        elif method._mutation_type == MutationType.NODE:
                            node_methods.append(method.__name__)

                check_base_methods(base)

        check_base_methods(self.__class__)

        # We want the unique set of mutation methods across the class and its superclasses
        self._layer_mutation_methods = list(set(layer_methods))
        self._node_mutation_methods = list(set(node_methods))

        self._mutation_methods = (
            self._layer_mutation_methods + self._node_mutation_methods
        )
    
    def _init_underlying_methods(self) -> None:
        # After module has been initialized, we can identify 
        # any additional methods of underlying EvolvableModule attributes
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
        
        all_methods = layer_fns + node_fns
        self._mutation_methods += all_methods
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

        if num_layer_fns == 0 or num_node_fns == 0:
            return [1 / len(self.mutation_methods) for _ in self.mutation_methods]

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
        param rng: The random number generator.
        type rng: Optional[Generator]
        return: The sampled mutation method.
        rtype: MutationMethod
        """
        if rng is None:
            rng = np.random.default_rng()

        probs = self.get_mutation_probs(new_layer_prob)
        return rng.choice(self.mutation_methods, p=probs, size=1)[0]

    def clone(self: SelfEvolvableModule) -> SelfEvolvableModule:
        """Returns clone of neural net with identical parameters."""
        clone = self.__class__(**copy.deepcopy(self.init_dict))

        # Load state dict if the network has been trained
        if self.state_dict():
            clone.load_state_dict(self.state_dict())

        return clone
    
class EvolvableWrapper(EvolvableModule):
    """Wrapper class for evolvable neural networks. It takes in an EvolvableModule and 
    inherits its mutation methods as class methods."""

    def __init__(self, module: EvolvableModule) -> None:
        super().__init__(module.device)
        self._wrapped = module

    @property
    def wrapped(self) -> Optional["EvolvableModule"]:
        return self._wrapped

    def _init_underlying_methods(self) -> None:
        super()._init_underlying_methods()

        layer_fns = self.wrapped.layer_mutation_methods
        node_fns = self.wrapped.node_mutation_methods
        all_methods = layer_fns + node_fns

        conflicting_methods = [method for method in self._mutation_methods if method in all_methods]
        if conflicting_methods:
            raise ValueError(
            f"Mutation methods in the wrapped module conflict with the wrapper's methods: {conflicting_methods}"
            )

        self._node_mutation_methods += node_fns
        self._layer_mutation_methods += layer_fns
        self._mutation_methods += all_methods

    def modules(self) -> Dict[str, "EvolvableModule"]:
        """Returns the attributes related to the evolvable modules in the algorithm. Includes 
        attributes that are either evolvable modules or a list of evolvable modules, as well 
        as the optimizers associated with the networks.

        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """
        # Inspect evolvable
        evolvable_attrs = {}
        for attr in dir(self):
            obj = getattr(self, attr)
            if is_evolvable(attr, obj) and attr != "wrapped":
                evolvable_attrs[attr] = obj

        return evolvable_attrs

    def get_mutation_methods(self) -> Dict[str, MutationMethod]:
        """Get all mutation methods for the network.

        :return: A dictionary of mutation methods.
        :rtype: Dict[str, MutationMethod]
        """
        def get_method_from_name(name: str) -> MutationMethod:
            if "." not in name:
                try:
                    return getattr(self.wrapped, name)
                except AttributeError:
                    pass

                return getattr(self, name)

            attr = name.split(".")[0]
            method = ".".join(name.split(".")[1:])
            return getattr(getattr(self, attr), method)

        return {name: get_method_from_name(name) for name in self.mutation_methods}

    def recreate_network(self, **kwargs) -> None:
        """Recreate the network after a mutation has been applied.
        
        :param shrink_params: Whether to shrink the network parameters.
        :type shrink_params: bool
        """
        return self.wrapped.recreate_network(**kwargs)

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
    
    def __getitem__(self, key: str) -> EvolvableModule:
        return super().__getitem__(key)

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
    
    def recreate_network(self, key: str, kwargs: Dict[str, Any]) -> None:
        """Recreate the network after a mutation has been applied.
        
        :param kwargs: The keyword arguments for recreating the network.
        :type kwargs: Dict[str, Dict[str, Any]]
        """
        self[key].recreate_network(**kwargs)