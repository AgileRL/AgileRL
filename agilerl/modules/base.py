import copy
import inspect
from collections import OrderedDict
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from numpy.random import Generator
from torch._dynamo import OptimizedModule

from agilerl.modules.custom_components import NoisyLinear
from agilerl.protocols import MutationMethod, MutationType

SelfEvolvableModule = TypeVar("SelfEvolvableModule", bound="EvolvableModule")


def mutation(
    mutation_type: MutationType, **recreate_kwargs
) -> Callable[[Callable], MutationMethod]:
    """Decorator to register a method as a mutation function of a specific type. This signals
    that the module should be recreated after the function has been called on the module.

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

        # Each mutation method might want to call different arguments when recreating the network
        wrapper._recreate_kwargs = recreate_kwargs
        return wrapper

    return decorator


class MutationContext:
    """Tracks nested mutation method calls. This allows us to automatically call a modules
    ``recreate_network()`` method after the outermost mutation method has been called. It handles
    cases where we fall back on another mutation method when a limit has been reached e.g. in
    the EvolvableMLP ``add_layer()`` method, after the maximum number of layers has been reached we
    instead call ``add_node()``. In cases like this, we want to avoid redundant recreations of the
    network.

    :param module: The evolvable module.
    :type module: EvolvableModule
    :param method: The mutation method.
    :type method: MutationMethod
    :param attribute: The mutation attribute.
    :type attribute: str
    """

    def __init__(
        self, module: SelfEvolvableModule, method: MutationMethod, attribute: str
    ):
        self.module = module
        self.method = method
        self.method_name = attribute

    def __enter__(self) -> None:
        self.module._mutation_depth += 1
        self.module.last_mutation = self.method
        self.module.last_mutation_attr = self.method_name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager. If the mutation depth is 0, then we have exited the outermost
        mutation method and need to recreate the network. Otherwise, we have exited a nested
        mutation method and do not need to do anything.
        """
        self.module._mutation_depth -= 1
        if self.module._mutation_depth == 0:
            # Identify the mutation method that was actually applied
            final_mutation_attr = self._resolve_final_mutation_attr()
            self.module.last_mutation_attr = final_mutation_attr

            if final_mutation_attr is not None:
                self.module.last_mutation = getattr(self.module, final_mutation_attr)

                # Recreate the network whose architecture was mutated
                if "." not in final_mutation_attr and not isinstance(
                    self.module, EvolvableWrapper
                ):
                    if self.method._recreate_kwargs:
                        recreation_kwargs = inspect.signature(
                            self.module.recreate_network
                        ).parameters
                        rec_kwargs = {
                            k: v
                            for k, v in self.method._recreate_kwargs.items()
                            if k in recreation_kwargs
                        }
                    else:
                        rec_kwargs = {}

                    self.module.recreate_network(**rec_kwargs)

            # Apply mutation hook if specified
            for hook in self.module._mutation_hooks:
                hook()

    def _resolve_final_mutation_attr(self) -> Optional[str]:
        """Resolve the mutation method that was applied last. This is necessary during
        hyperparameter optimization in RL algorithms because it is often convenient to
        apply the same architecture mutation to the different networks in an algorithm
        because the tasks they solve are of similar complexity, and their networks will
        therefore require similar capacities.

        :return: The final mutation attribute.
        :rtype: Optional[str]
        """
        if (
            self.module.last_mutation_attr is not None
            and "." in self.module.last_mutation_attr
        ):
            parts = self.module.last_mutation_attr.split(".")
            mutated_module = self.module
            for part in parts[:-1]:
                mutated_module: SelfEvolvableModule = getattr(mutated_module, part)

            if mutated_module.last_mutation_attr is None:
                return

            return ".".join(parts[:-1] + [mutated_module.last_mutation_attr])

        elif isinstance(self.module, EvolvableWrapper):
            return self.module.wrapped.last_mutation_attr

        return self.module.last_mutation_attr


def _mutation_wrapper(
    module: SelfEvolvableModule, method: MutationMethod, attribute: str
) -> Callable:
    """Wraps mutation methods to use context manager.

    :param module: The evolvable module.
    :type module: EvolvableModule
    :param method: The mutation method.
    :type method: MutationMethod
    :param attribute: The mutation attribute.
    :type attribute: str

    :return: The wrapped mutation method.
    :rtype: Callable
    """

    @wraps(method)
    def wrapped(*args, **kwargs):
        with MutationContext(module, method, attribute):
            # This handles the case of an `EvolvableWrapper`
            if attribute not in module.mutation_methods:
                module.last_mutation_attr = None
                module.last_mutation = None
                return

            return method(*args, **kwargs)

    return wrapped


def _get_filtered_methods(
    module: SelfEvolvableModule, mut_type: MutationType
) -> List[str]:
    """Gets the mutation methods of a given type for the module.

    :param module: The evolvable module.
    :type module: EvolvableModule
    :param mut_type: The type of mutation method.
    :type mut_type: MutationType
    :return: The filtered mutation methods.
    :rtype: List[str]
    """
    _fetch = (
        "layer_mutation_methods"
        if mut_type == MutationType.LAYER
        else "node_mutation_methods"
    )

    init_methods: List[str] = getattr(module, "_" + _fetch)
    _filtered = []
    for method in init_methods:
        if "." not in method:
            _filtered.append(method)
            continue

        comps = method.split(".")
        mod: EvolvableModule = getattr(module, comps[0])
        layer_muts = getattr(mod, _fetch)
        if any(".".join([comps[0], m]) == method for m in layer_muts):
            _filtered.append(method)

    return _filtered


class ModuleMeta(type):
    """Metaclass to parse the mutation methods of an EvolvableModule instance
    and its superclasses. Allows us to dynamically keep track of the last mutation
    method applied to an EvolvableModule instance, and automatically recreate the
    relevant network after a mutation has been applied.

    :param cls: The class to be metaclassed.
    :type cls: type
    :param args: The arguments to pass to the class constructor.
    :type args: tuple
    :param kwargs: The keyword arguments to pass to the class constructor.
    :type kwargs: dict
    """

    def __call__(
        cls: Type[SelfEvolvableModule], *args, **kwargs
    ) -> SelfEvolvableModule:
        instance: SelfEvolvableModule = super().__call__(*args, **kwargs)

        # Wrap mutation methods to use context manager
        for name, method in instance.get_mutation_methods().items():
            setattr(instance, name, _mutation_wrapper(instance, method, name))

        # Use the same random number generator for all of the evolvable modules
        for module in instance._evolvable_modules.values():
            module._rng = instance._rng

        return instance


class EvolvableModule(nn.Module, metaclass=ModuleMeta):
    """Base class for evolvable neural networks.

    :param device: The device to run the network on.
    :type device: str
    :param random_seed: The random seed to use for the network.
    :type random_seed: Optional[int]
    """

    _evolvable_modules: Dict[str, Optional["EvolvableModule"]]
    _mutation_hooks: List[Callable]

    def __init__(self, device: str, random_seed: Optional[int] = None) -> None:
        nn.Module.__init__(self)
        self._init_surface_methods()
        self.device = device
        self.random_seed = random_seed

        self._rng = np.random.default_rng(seed=random_seed)
        self._last_mutation = None
        self._last_mutation_attr = None
        self._mutation_depth = 0

        super().__setattr__("_evolvable_modules", {})
        super().__setattr__("_mutation_hooks", [])

    @property
    def init_dict(self) -> Dict[str, Any]:
        return self.get_init_dict()

    @property
    def net_config(self) -> Dict[str, Any]:
        return self.get_init_dict()

    @property
    def mutation_methods(self) -> List[str]:
        return self.layer_mutation_methods + self.node_mutation_methods

    @property
    def layer_mutation_methods(self) -> List[str]:
        return _get_filtered_methods(self, MutationType.LAYER)

    @property
    def node_mutation_methods(self) -> List[str]:
        return _get_filtered_methods(self, MutationType.NODE)

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

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._rng = value
        for module in self.modules().values():
            module.rng = value

    def recreate_network(self, **kwargs) -> None:
        """Recreate the network after a mutation has been applied. If the mutation methods of
        an `EvolvableModule` are only attributed to its nested modules, then the `recreate_network`
        method should be implemented in the nested modules and it is not required on the parent.

        :param kwargs: Keyword arguments to pass to the network constructor.
        :type kwargs: Dict[str, Any]
        """
        if any("." not in method for method in self.mutation_methods):
            raise NotImplementedError(
                "An EvolvableModule must implement the recreate_network method whenever it includes "
                "unique mutation methods."
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "forward method must be implemented in order to use the evolvable module."
        )

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the network."""
        return self.forward(*args, **kwargs)

    def get_init_dict(self) -> Dict[str, Any]:
        """Get the dictionary of constructor arguments for the network.

        :return: The dictionary of constructor arguments.
        :rtype: Dict[str, Any]
        """
        constructor_args = inspect.signature(self.__init__).parameters

        try:
            return {k: getattr(self, k) for k in constructor_args.keys()}
        except AttributeError as e:
            raise AttributeError(
                "Custom EvolvableModule objects must be explicit about their "
                "constructor arguments (i.e. don't use *args and **kwargs). Error: "
                + str(e)
            )

    def change_activation(self, activation: str, output: bool) -> None:
        """Set the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: Whether to set the activation function for the output layer.
        :type output: bool
        """
        raise NotImplementedError(
            "change_activation method must be implemented in order to set the activation function."
        )

    def __setattr__(self, name: str, value: Union[Any, SelfEvolvableModule]) -> None:
        """Set attribute of the network. If the attribute is a module, add its mutation methods
        to the parent module. Otherwise, set the attribute as usual.

        :param name: The name of the attribute.
        :type name: str
        :param value: The value of the attribute.
        :type value: Any
        """
        # Add mutation methods to the network
        if isinstance(value, EvolvableModule):
            if name not in self._evolvable_modules:
                layer_fns = []
                node_fns = []
                for mut_name, method in value.get_mutation_methods().items():
                    method_name = ".".join([name, mut_name])
                    method_type = method._mutation_type
                    if method_type == MutationType.LAYER:
                        layer_fns.append(method_name)
                    elif method_type == MutationType.NODE:
                        node_fns.append(method_name)
                    else:
                        raise ValueError(f"Invalid mutation type: {method_type}")

                self._layer_mutation_methods += layer_fns
                self._node_mutation_methods += node_fns
            else:
                value._layer_mutation_methods = self._evolvable_modules[
                    name
                ]._layer_mutation_methods
                value._node_mutation_methods = self._evolvable_modules[
                    name
                ]._node_mutation_methods

            self._evolvable_modules[name] = value

        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """Get attribute of the network. If the attribute is a mutation method, return the
        method (also one from a nested module). Otherwise, raise an AttributeError.

        :param name: The name of the attribute.
        :type name: str
        :return: The attribute of the network.
        :rtype: Any
        """
        try:
            attr = super().__getattr__(name)
            return attr
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
                    slice_index = tuple(
                        slice(0, min(o, n)) for o, n in zip(old_size, new_size)
                    )
                    param.data[slice_index] = old_param.data[slice_index]

        return new_net

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

    def _init_surface_methods(self) -> None:
        """Initialize the surface mutation methods of the network. We parse the class and its
        superclasses to find all mutation methods and add them to the `_layer_mutation_methods`
        and `_node_mutation_methods` lists."""

        def _fetch_methods(cls: Type) -> List[str]:
            return [
                method
                for method in vars(cls).values()
                if isinstance(method, MutationMethod)
            ]

        # Check mutation methods in class
        class_methods: List[MutationMethod] = _fetch_methods(self.__class__)
        layer_methods = []
        node_methods = []
        for method in class_methods:
            if method._mutation_type == MutationType.LAYER:
                layer_methods.append(method.__name__)
            elif method._mutation_type == MutationType.NODE:
                node_methods.append(method.__name__)

        # Check mutation methods in superclasses
        def check_base_methods(cls) -> None:
            for base in cls.__bases__:
                if base is EvolvableModule:
                    return

                base_methods: List[MutationMethod] = _fetch_methods(base)
                for method in base_methods:
                    if method._mutation_type == MutationType.LAYER:
                        layer_methods.append(method.__name__)
                    elif method._mutation_type == MutationType.NODE:
                        node_methods.append(method.__name__)

                check_base_methods(base)

        check_base_methods(self.__class__)

        # We want the unique set of mutation methods across the class and its superclasses
        self._layer_mutation_methods = list(set(layer_methods))
        self._node_mutation_methods = list(set(node_methods))

    def reset_noise(self) -> None:
        """Reset noise for all NoisyLinear layers in the network."""
        for layer in self.torch_modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def register_mutation_hook(self, hook: Callable) -> None:
        """Register a hook to be called after a mutation has been applied to a
        nested evolvable module. The hook function should not take any arguments.

        :param hook: The hook function.
        :type hook: Callable
        """
        self._mutation_hooks.append(hook)

    def disable_mutations(self, mut_type: Optional[MutationType] = None) -> None:
        """Disable all or some mutation methods from the evolvable module. It recursively
        disables the mutation methods of nested evolvable modules as well.

        :param mut_type: The type of mutation method to disable.
        :type mut_type: Optional[MutationType]
        """
        if mut_type is None:
            self._layer_mutation_methods = []
            self._node_mutation_methods = []
        elif mut_type == MutationType.LAYER:
            self._layer_mutation_methods = []
        elif mut_type == MutationType.NODE:
            self._node_mutation_methods = []
        else:
            raise ValueError(f"Invalid mutation type: {mut_type}")

        # Recursively disable mutations in nested evolvable modules
        for module in self.modules().values():
            module.disable_mutations(mut_type)

    def modules(self) -> Dict[str, "EvolvableModule"]:
        """Returns the nested evolvable modules in the network.

        .. warning:: This overrides the behavior of `nn.Module.modules()` and only returns
            the evolvable modules. If you need the torch modules, use :meth:`torch_modules()` instead.

        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """
        return self._evolvable_modules

    def torch_modules(self) -> Iterator[nn.Module]:
        """Returns an iterator over the `nn.Module`s in the network.

        :return: An iterator over the `nn.Module`s in the network.
        :rtype: Iterator[nn.Module]
        """
        return super().modules()

    def get_mutation_methods(self) -> Dict[str, MutationMethod]:
        """Get all mutation methods for the network as dictionary of method names to
        mutation methods.

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

        :param remove: The string to remove.
        :type remove: str
        """

        def filter_methods(methods: List[str]) -> List[str]:
            return [method for method in methods if remove not in method]

        self._layer_mutation_methods = filter_methods(self._layer_mutation_methods)
        self._node_mutation_methods = filter_methods(self._node_mutation_methods)

    def get_mutation_probs(self, new_layer_prob: float) -> List[float]:
        """Get the mutation probabilities for each mutation method.

        :param new_layer_prob: The probability of selecting a layer mutation method.
        :type new_layer_prob: float
        return: A list of probabilities for each mutation method.
        rtype: List[float]
        """
        num_layer_fns = len(self.layer_mutation_methods)
        num_node_fns = len(self.node_mutation_methods)

        num_total = num_layer_fns + num_node_fns
        if num_layer_fns == 0 or num_node_fns == 0:
            return [1 / num_total for _ in range(num_total)]

        probs = []
        for fn in self.get_mutation_methods().values():
            if fn._mutation_type == MutationType.LAYER:
                prob = new_layer_prob / num_layer_fns
            elif fn._mutation_type == MutationType.NODE:
                prob = (1 - new_layer_prob) / num_node_fns

            probs.append(prob)

        return probs

    def sample_mutation_method(
        self, new_layer_prob: float, rng: Optional[Generator] = None
    ) -> MutationMethod:
        """Sample a mutation method based on the mutation probabilities.

        :param new_layer_prob: The probability of selecting a layer mutation method.
        type new_layer_prob: float
        :param rng: The random number generator.
        type rng: Optional[Generator]
        return: The sampled mutation method.
        rtype: MutationMethod
        """
        if not self.mutation_methods:
            raise ValueError(
                "No mutation methods available. Please use the @mutation decorator to register methods."
            )

        if rng is None:
            rng = np.random.default_rng()

        probs = self.get_mutation_probs(new_layer_prob)
        return rng.choice(self.mutation_methods, p=probs, size=1)[0]

    def clone(self: SelfEvolvableModule) -> SelfEvolvableModule:
        """Returns clone of an `EvolvableModule` with identical parameters.

        :return: A clone of the `EvolvableModule`.
        :rtype: SelfEvolvableModule
        """
        clone = self.__class__(**copy.deepcopy(self.get_init_dict()))
        clone.rng = self.rng

        clone._layer_mutation_methods = self._layer_mutation_methods
        clone._node_mutation_methods = self._node_mutation_methods
        for name, module in clone._evolvable_modules.items():
            module._layer_mutation_methods = self._evolvable_modules[
                name
            ]._layer_mutation_methods
            module._node_mutation_methods = self._evolvable_modules[
                name
            ]._node_mutation_methods

        # Load state dict if the network has been trained
        try:
            clone.load_state_dict(self.state_dict(), strict=False)
        except RuntimeError as e:
            print(
                f"Warning: Failed to load state_dict during unpickling of {self.__class__.__name__} to device {self.device}. Error: {e}"
            )
            pass
        return clone


class EvolvableWrapper(EvolvableModule):
    """Wrapper class for evolvable neural networks. Can be used to provide some
    additional functionality to an `EvolvableModule` while maintaining its mutation methods
    at the top-level.

    :param module: The evolvable module to wrap.
    :type module: EvolvableModule
    """

    def __init__(self, module: EvolvableModule) -> None:
        super().__init__(module.device, module.random_seed)

        self._init_wrapped_methods(module, MutationType.LAYER)
        self._init_wrapped_methods(module, MutationType.NODE)

        # Disable mutations in the wrapped module since these are now assigned to the wrapper
        self._wrapped = module
        self.filter_mutation_methods("_wrapped")

    @property
    def wrapped(self) -> EvolvableModule:
        return self._wrapped

    def change_activation(self, activation: str, output: bool) -> None:
        """Change the activation function for the network.

        :param activation: The activation function to use.
        :type activation: str
        :param output: Whether to set the activation function for the output layer.
        :type output: bool
        """
        self._wrapped.change_activation(activation, output)

    def _init_wrapped_methods(
        self, module: EvolvableModule, mut_type: MutationType
    ) -> None:
        """Initialize the mutation methods of the wrapped module.

        :param module: The wrapped module.
        :type module: EvolvableModule
        :param mut_type: The type of mutation method.
        :type mut_type: MutationType
        """
        _fetch = (
            "layer_mutation_methods"
            if mut_type == MutationType.LAYER
            else "node_mutation_methods"
        )

        for method in getattr(module, _fetch):
            if method not in self.mutation_methods:
                setattr(self, method, getattr(module, method))
                current_methods: List[str] = getattr(self, f"_{_fetch}")
                current_methods.append(method)
                setattr(self, f"_{_fetch}", current_methods)
            else:
                raise AttributeError(f"Duplicate mutation method: {method}")


ModuleType = TypeVar("ModuleType", bound=Union[EvolvableModule, nn.Module])


class ModuleDict(EvolvableModule, nn.ModuleDict, Generic[ModuleType]):
    """Analogous to ``nn.ModuleDict``, but allows for the inheritance of the
    mutation methods of nested evolvable modules.

    :param modules: The modules to add to the dictionary.
    :type modules: Optional[Dict[str, EvolvableModule]]
    :param device: The device to use for the modules.
    :type device: str
    """

    def __init__(
        self, modules: Optional[Dict[str, EvolvableModule]] = None, device: str = "cpu"
    ) -> None:
        super().__init__(device)
        if modules is not None:
            for name, module in modules.items():
                self.add_module(name, module)

    @property
    def layer_mutation_methods(self) -> List[str]:
        return [
            f"{name}.{method}"
            for name, module in self.modules().items()
            for method in module.layer_mutation_methods
        ]

    @property
    def node_mutation_methods(self) -> List[str]:
        return [
            f"{name}.{method}"
            for name, module in self.modules().items()
            for method in module.node_mutation_methods
        ]

    @property
    def activation(self) -> Optional[str]:
        nested_activations = [module.activation for module in self.modules().values()]
        if len(set(nested_activations)) == 1:
            return nested_activations[0]
        return None

    def __getitem__(self, key: str) -> ModuleType:
        return super().__getitem__(key)

    def values(self) -> Iterable[ModuleType]:
        return super().values()

    def items(self) -> Iterable[Tuple[str, ModuleType]]:
        return super().items()

    def change_activation(self, activation: str, output: bool) -> None:
        """Change the activation function for the network.

        :param activation: The activation function to use.
        :type activation: str
        :param output: Whether to set the activation function for the output layer.
        :type output: bool
        """
        for module in self.modules().values():
            module.change_activation(activation, output)

    def filter_mutation_methods(self, remove: str) -> None:
        """Filter out mutation methods that contain the specified string in their name.

        param remove: The string to remove.
        type remove: str
        """
        for module in self.modules().values():
            module.filter_mutation_methods(remove)

    def modules(self) -> Dict[str, EvolvableModule]:
        """Returns the nested evolvable modules in the network.

        .. warning:: This overrides the behavior of `nn.Module.modules()` and only returns
            the evolvable modules. If you need the torch modules, use :meth:`torch_modules()` instead.

        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """
        evo_modules = OrderedDict()
        for name, module in self.items():
            orig_mod = (
                module._orig_mod if isinstance(module, OptimizedModule) else module
            )
            if isinstance(orig_mod, EvolvableModule):
                evo_modules[name] = module

        return evo_modules

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

    def clone(self) -> "ModuleDict":
        """Returns clone of an `ModuleDict` with identical parameters.

        :return: A clone of the `ModuleDict`.
        :rtype: ModuleDict
        """
        return ModuleDict({key: module.clone() for key, module in self.items()})
