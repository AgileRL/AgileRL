import pytest
from unittest.mock import Mock
import numpy as np
import torch
from torch import nn

from agilerl.modules.base import (
    EvolvableModule,
    EvolvableWrapper,
    ModuleDict,
    MutationContext,
    _mutation_wrapper,
    mutation,
)
from agilerl.modules.custom_components import NoisyLinear
from agilerl.protocols import MutationType


def test_register_mutation_fn():
    @mutation(MutationType.NODE)
    def dummy_mutation(self):
        return {"mutation": "dummy"}

    assert dummy_mutation._mutation_type == MutationType.NODE


def test_evolvable_module_initialization():
    class DummyEvolvableModule(EvolvableModule):
        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = DummyEvolvableModule(device="cpu")
    assert module.device == "cpu"
    assert module.get_init_dict() == {"device": "cpu", "random_seed": None}


def test_evolvable_module_get_mutation_methods():
    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_mutation(self):
            return {"mutation": "dummy"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = DummyEvolvableModule(device="cpu")
    mutation_methods = module.get_mutation_methods()
    assert "dummy_mutation" in mutation_methods
    assert mutation_methods["dummy_mutation"]._mutation_type == MutationType.NODE


def test_evolvable_module_clone():
    class DummyEvolvableModule(EvolvableModule):
        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = DummyEvolvableModule(device="cpu")
    clone = module.clone()
    assert clone.device == module.device
    assert clone.get_init_dict() == module.get_init_dict()


def test_evolvable_module_make_unevolvable():
    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_mutation(self):
            return {"mutation": "dummy"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = DummyEvolvableModule(device="cpu")
    module.disable_mutations()
    assert module.mutation_methods == []
    assert module.layer_mutation_methods == []
    assert module.node_mutation_methods == []


def test_evolvable_module_sample_mutation_method():
    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_mutation(self):
            return {"mutation": "dummy"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = DummyEvolvableModule(device="cpu")
    sampled_method = module.sample_mutation_method(new_layer_prob=0.5)
    assert sampled_method == "dummy_mutation"


def test_evolvable_module_sample_mutation_method_raises_when_no_methods():
    """sample_mutation_method raises ValueError when no mutation methods are registered."""

    class NoMutationModule(EvolvableModule):
        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = NoMutationModule(device="cpu")
    with pytest.raises(ValueError, match="No mutation methods available"):
        module.sample_mutation_method(new_layer_prob=0.5)


@pytest.mark.parametrize(
    "mut_type, expect_error", [(99, ValueError), ("invalid", ValueError)]
)
def test_evolvable_module_disable_mutations_invalid_type(mut_type, expect_error):
    """disable_mutations raises ValueError for invalid mutation type."""

    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_mutation(self):
            return {"mutation": "dummy"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = DummyEvolvableModule(device="cpu")
    with pytest.raises(expect_error, match="Invalid mutation type"):
        module.disable_mutations(mut_type)


def test_inherited_evolvable_module_mutation_methods():
    class BaseEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def base_mutation(self):
            return {"mutation": "base"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    class InheritedEvolvableModule(BaseEvolvableModule):
        @mutation(MutationType.LAYER)
        def inherited_mutation(self):
            return {"mutation": "inherited"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = InheritedEvolvableModule(device="cpu")
    mutation_methods = module.get_mutation_methods()
    assert "base_mutation" in mutation_methods
    assert "inherited_mutation" in mutation_methods
    assert mutation_methods["base_mutation"]._mutation_type == MutationType.NODE
    assert mutation_methods["inherited_mutation"]._mutation_type == MutationType.LAYER


def test_evolvable_module_with_evolvable_attributes():
    class AttributeEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def attribute_mutation(self):
            return {"mutation": "attribute"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    class ParentEvolvableModule(EvolvableModule):
        def __init__(self, device="cpu"):
            super().__init__(device)
            self.attribute_module = AttributeEvolvableModule(device=device)

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = ParentEvolvableModule(device="cpu")
    mutation_methods = module.get_mutation_methods()
    assert "attribute_module.attribute_mutation" in mutation_methods
    assert (
        mutation_methods["attribute_module.attribute_mutation"]._mutation_type
        == MutationType.NODE
    )


def test_mutation_calls_recreate_network():
    """Test that the mutation decorator calls recreate_network after mutation."""

    class MockEvolvableModule(EvolvableModule):
        def __init__(self, device="cpu"):
            super().__init__(device)

        @mutation(MutationType.NODE)
        def node_mutation(self):
            return {"mutation": "node"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    # Create a module instance
    module = MockEvolvableModule(device="cpu")

    # Replace recreate_network with a mock
    module.recreate_network = Mock()

    # Call the mutation method, which should trigger recreate_network via the wrapper
    module.node_mutation()

    # Verify recreate_network was called once
    module.recreate_network.assert_called_once_with()


def test_mutation_with_recreate_kwargs():
    """Test that the kwargs are correctly passed to the mutation decorator."""
    # Instead of testing the actual call to recreate_network with shrink_params,
    # test that the mutation decorator correctly sets the _recreate_kwargs attribute

    @mutation(MutationType.NODE, shrink_params=True)
    def test_mutation(self):
        pass

    # Check that the decorator correctly set the _recreate_kwargs attribute
    assert test_mutation._recreate_kwargs == {"shrink_params": True}


def test_mutation_with_args_kwargs():
    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def mutation_with_args(self, arg1, arg2=None):
            return {"arg1": arg1, "arg2": arg2}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    module = DummyEvolvableModule(device="cpu")
    module.recreate_network = Mock()

    result = module.mutation_with_args("test", arg2="value")

    assert result == {"arg1": "test", "arg2": "value"}
    module.recreate_network.assert_called_once()


######### Test ModuleDict #########
def test_evolvable_wrapper_duplicate_mutation_method_raises():
    """EvolvableWrapper raises AttributeError when wrapped module has same method name in both layer and node lists."""

    class ModuleWithDuplicateMethodName(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_mutation(self):
            return {}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    mod = ModuleWithDuplicateMethodName(device="cpu")
    mod._layer_mutation_methods.append("dummy_mutation")  # same name in both lists
    with pytest.raises(AttributeError, match="Duplicate mutation method"):
        EvolvableWrapper(mod)


def test_module_dict_initialization():
    """Test ModuleDict initialization with evolvable modules."""

    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_node_mutation(self):
            return {"mutation": "node"}

        @mutation(MutationType.LAYER)
        def dummy_layer_mutation(self):
            return {"mutation": "layer"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    # Create a ModuleDict with device parameter first
    module_dict = ModuleDict(device="cpu")

    # Add modules after initialization
    module_dict["module1"] = DummyEvolvableModule(device="cpu")
    module_dict["module2"] = DummyEvolvableModule(device="cpu")

    # Check that the ModuleDict contains the modules
    assert "module1" in module_dict
    assert "module2" in module_dict
    assert isinstance(module_dict["module1"], DummyEvolvableModule)
    assert isinstance(module_dict["module2"], DummyEvolvableModule)


def test_module_dict_mutation_methods():
    """Test that ModuleDict correctly exposes mutation methods from contained modules."""

    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_node_mutation(self):
            return {"mutation": "node"}

        @mutation(MutationType.LAYER)
        def dummy_layer_mutation(self):
            return {"mutation": "layer"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    # Create a ModuleDict
    module_dict = ModuleDict(device="cpu")

    # Add modules after initialization
    module_dict["module1"] = DummyEvolvableModule(device="cpu")
    module_dict["module2"] = DummyEvolvableModule(device="cpu")

    # Check that mutation methods are correctly exposed
    mutation_methods = module_dict.get_mutation_methods()

    # Check that all expected mutation methods are present
    assert "module1.dummy_node_mutation" in mutation_methods
    assert "module1.dummy_layer_mutation" in mutation_methods
    assert "module2.dummy_node_mutation" in mutation_methods
    assert "module2.dummy_layer_mutation" in mutation_methods

    # Check mutation types
    assert (
        mutation_methods["module1.dummy_node_mutation"]._mutation_type
        == MutationType.NODE
    )
    assert (
        mutation_methods["module1.dummy_layer_mutation"]._mutation_type
        == MutationType.LAYER
    )


def test_module_dict_layer_node_methods():
    """Test the layer_mutation_methods and node_mutation_methods properties of ModuleDict."""

    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_node_mutation(self):
            return {"mutation": "node"}

        @mutation(MutationType.LAYER)
        def dummy_layer_mutation(self):
            return {"mutation": "layer"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    # Create a ModuleDict
    module_dict = ModuleDict(device="cpu")

    # Add modules after initialization
    module_dict["module1"] = DummyEvolvableModule(device="cpu")
    module_dict["module2"] = DummyEvolvableModule(device="cpu")

    # Check layer mutation methods
    layer_methods = module_dict.layer_mutation_methods
    assert "module1.dummy_layer_mutation" in layer_methods
    assert "module2.dummy_layer_mutation" in layer_methods
    assert "module1.dummy_node_mutation" not in layer_methods

    # Check node mutation methods
    node_methods = module_dict.node_mutation_methods
    assert "module1.dummy_node_mutation" in node_methods
    assert "module2.dummy_node_mutation" in node_methods
    assert "module1.dummy_layer_mutation" not in node_methods


def test_module_dict_access_methods():
    """Test the various methods to access modules in ModuleDict."""

    class DummyEvolvableModule(EvolvableModule):
        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    # Create a ModuleDict
    module_dict = ModuleDict(device="cpu")

    # Add modules after initialization
    module_dict["module1"] = DummyEvolvableModule(device="cpu")
    module_dict["module2"] = DummyEvolvableModule(device="cpu")

    # Test __getitem__
    assert isinstance(module_dict["module1"], DummyEvolvableModule)

    # Test values()
    values = list(module_dict.values())
    assert len(values) == 2
    assert all(isinstance(module, DummyEvolvableModule) for module in values)

    # Test items()
    items = list(module_dict.items())
    assert len(items) == 2
    assert items[0][0] == "module1" or items[1][0] == "module1"
    assert all(isinstance(item[1], DummyEvolvableModule) for item in items)

    # Test modules()
    modules = module_dict.modules()
    assert len(modules) == 2
    assert "module1" in modules
    assert "module2" in modules


######### Test EvolvableWrapper #########
def test_evolvable_wrapper_initialization():
    """Test EvolvableWrapper initialization with an evolvable module."""

    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_node_mutation(self):
            return {"mutation": "node"}

        @mutation(MutationType.LAYER)
        def dummy_layer_mutation(self):
            return {"mutation": "layer"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    # Create an evolvable module
    module = DummyEvolvableModule(device="cpu")

    # Wrap the module
    wrapper = EvolvableWrapper(module)

    # Check that the wrapper contains the module
    assert wrapper.wrapped == module
    assert wrapper.device == module.device


def test_evolvable_wrapper_mutation_methods():
    """Test that EvolvableWrapper correctly exposes mutation methods from wrapped module."""

    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_node_mutation(self):
            return {"mutation": "node"}

        @mutation(MutationType.LAYER)
        def dummy_layer_mutation(self):
            return {"mutation": "layer"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    # Create an evolvable module
    module = DummyEvolvableModule(device="cpu")

    # Wrap the module
    wrapper = EvolvableWrapper(module)

    # Check that mutation methods are correctly exposed
    wrapper_methods = wrapper.get_mutation_methods()

    # Check that all expected mutation methods are present
    assert "dummy_node_mutation" in wrapper_methods
    assert "dummy_layer_mutation" in wrapper_methods

    # Check that mutation types are preserved
    assert wrapper_methods["dummy_node_mutation"]._mutation_type == MutationType.NODE
    assert wrapper_methods["dummy_layer_mutation"]._mutation_type == MutationType.LAYER

    # Check that the wrapped module's mutations are disabled
    assert not any("_wrapped" in method for method in wrapper.mutation_methods)


class MockMethod:
    _mutation_type = MutationType.NODE
    _recreate_kwargs = {}

    def __call__(self, *args, **kwargs):
        return {"mutation": "mock"}


def test_evolvable_wrapper_mutation_execution():
    """Test that mutations on the wrapper are correctly executed."""
    # In EvolvableWrapper, the wrapper takes over the mutation methods
    # and the original module's methods are disabled

    class DummyEvolvableModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def dummy_mutation(self):
            return {"mutation": "dummy"}

        def forward(self, x):
            pass

        def recreate_network(self):
            pass

    # Create an evolvable module with a mock recreate_network
    module = DummyEvolvableModule(device="cpu")

    # Wrap the module
    wrapper = EvolvableWrapper(module)

    # The wrapper should have adopted the method but the original method is disabled
    # We need to mock it in the wrapper directly
    wrapper.dummy_mutation = MockMethod()

    # Call the mutation method on the wrapper
    result = wrapper.dummy_mutation()

    # Verify the result
    assert result == {"mutation": "mock"}


def test_evolvable_wrapper_forward():
    """Test the forward method being correctly proxied to the wrapped module."""

    class DummyEvolvableModule(EvolvableModule):
        def forward(self, x):
            return "forwarded"

        def recreate_network(self):
            pass

    # Create an evolvable module
    module = DummyEvolvableModule(device="cpu")

    # Wrap the module
    wrapper = EvolvableWrapper(module)

    # Define a forward method that calls the wrapped module
    def forward(self, x):
        return self.wrapped.forward(x)

    # Add the forward method to the wrapper
    wrapper.forward = forward.__get__(wrapper)

    # Call the forward method
    result = wrapper.forward("input")

    # Verify the result
    assert result == "forwarded"


def test_mutation_context_filters_recreate_kwargs_and_calls_hooks():
    class Dummy(EvolvableModule):
        def __init__(self, device="cpu"):
            super().__init__(device)
            self.recreated_with = None

        @mutation(MutationType.NODE, shrink_params=True, ignored=True)
        def mut(self):
            return {}

        def forward(self, x):
            return x

        def recreate_network(self, shrink_params=False):
            self.recreated_with = shrink_params

    module = Dummy(device="cpu")
    hook = Mock()
    module.register_mutation_hook(hook)
    module.mut()
    assert module.recreated_with is True
    hook.assert_called_once()


def test_mutation_context_resolve_nested_and_wrapper_paths():
    class Child(EvolvableModule):
        @mutation(MutationType.NODE)
        def mut(self):
            return {}

        def forward(self, x):
            return x

        def recreate_network(self):
            pass

    class Parent(EvolvableModule):
        def __init__(self, device="cpu"):
            super().__init__(device)
            self.child = Child(device=device)

        def forward(self, x):
            return x

        def recreate_network(self):
            pass

    parent = Parent(device="cpu")
    ctx = MutationContext(parent, parent.child.mut, "child.mut")
    parent.last_mutation_attr = "child.mut"
    parent.child.last_mutation_attr = None
    assert ctx._resolve_final_mutation_attr() is None

    simple_child = Child(device="cpu")
    wrapper = EvolvableWrapper(simple_child)
    wrapper.wrapped.last_mutation_attr = "mut"
    wctx = MutationContext(wrapper, wrapper.mut, "mut")
    assert wctx._resolve_final_mutation_attr() == "mut"


def test_mutation_wrapper_returns_none_for_disabled_attr():
    class Dummy(EvolvableModule):
        @mutation(MutationType.NODE)
        def mut(self):
            return {}

        def forward(self, x):
            return x

        def recreate_network(self):
            pass

    module = Dummy(device="cpu")
    wrapped = _mutation_wrapper(module, module.mut, "missing_attr")
    assert wrapped() is None
    assert module.last_mutation is None
    assert module.last_mutation_attr is None


def test_base_properties_setters_and_errors():
    class Child(EvolvableModule):
        def forward(self, x):
            return x

        def recreate_network(self):
            pass

    class Parent(EvolvableModule):
        @mutation(MutationType.NODE)
        def mut(self):
            return {}

        def __init__(self, device="cpu"):
            super().__init__(device)
            self.child = Child(device=device)

        def forward(self, x):
            return x

    module = Parent(device="cpu")
    assert module.net_config == module.get_init_dict()
    assert module.activation is None
    module.rng = np.random.default_rng(123)
    module.device = "cpu:new"
    assert module.child.device == "cpu:new"
    assert module.child.rng is module.rng

    with pytest.raises(
        NotImplementedError, match="must implement the recreate_network"
    ):
        module.recreate_network()

    class NoForward(EvolvableModule):
        def recreate_network(self):
            pass

    with pytest.raises(NotImplementedError, match="forward method must be implemented"):
        NoForward(device="cpu").forward(torch.tensor([1.0]))

    class ArgsKw(EvolvableModule):
        def __init__(self, device="cpu", *args, **kwargs):
            super().__init__(device)

        def forward(self, x):
            return x

        def recreate_network(self):
            pass

    with pytest.raises(AttributeError, match="constructor arguments"):
        ArgsKw(device="cpu").get_init_dict()

    class NoChange(EvolvableModule):
        def forward(self, x):
            return x

        def recreate_network(self):
            pass

    with pytest.raises(
        NotImplementedError, match="change_activation method must be implemented"
    ):
        NoChange(device="cpu").change_activation("ReLU", output=False)


def test_setattr_invalid_mutation_type_raises():
    class BadMethod:
        _mutation_type = "invalid"

    class BadChild(EvolvableModule):
        def forward(self, x):
            return x

        def recreate_network(self):
            pass

        def get_mutation_methods(self):
            return {"bad": BadMethod()}

    class Parent(EvolvableModule):
        def forward(self, x):
            return x

        def recreate_network(self):
            pass

    parent = Parent(device="cpu")
    with pytest.raises(ValueError, match="Invalid mutation type"):
        parent.bad = BadChild(device="cpu")


def test_module_utilities_and_noise_and_probs():
    class MutModule(EvolvableModule):
        @mutation(MutationType.NODE)
        def node_mut(self):
            return {}

        @mutation(MutationType.LAYER)
        def layer_mut(self):
            return {}

        def __init__(self, device="cpu"):
            super().__init__(device)
            self.noisy = NoisyLinear(4, 4)

        def forward(self, x):
            return x

        def recreate_network(self):
            pass

    module = MutModule(device="cpu")
    assert module.get_output_dense() is None
    assert len(module.get_mutation_probs(0.6)) == 2

    module.disable_mutations(MutationType.LAYER)
    assert module.layer_mutation_methods == []
    module._layer_mutation_methods = ["layer_mut"]
    module.disable_mutations(MutationType.NODE)
    assert module.node_mutation_methods == []

    reset_mock = Mock()
    module.noisy.reset_noise = reset_mock
    module.reset_noise()
    reset_mock.assert_called_once()

    it = module.torch_modules()
    assert hasattr(it, "__iter__")


def test_preserve_parameters_and_init_weights_gaussian():
    old = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    new = nn.Sequential(nn.Linear(5, 2), nn.LayerNorm(2))
    new = EvolvableModule.preserve_parameters(old, new)
    assert new[0].weight.shape == torch.Size([2, 5])

    lin = nn.Linear(4, 2)
    EvolvableModule.init_weights_gaussian(lin, std_coeff=1.0)
    seq = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))
    EvolvableModule.init_weights_gaussian(seq, std_coeff=1.0)


def test_clone_nested_mutation_methods_and_wrapper_change_activation():
    class Child(EvolvableModule):
        @mutation(MutationType.NODE)
        def child_mut(self):
            return {}

        def forward(self, x):
            return x

        def recreate_network(self):
            pass

    class Parent(EvolvableModule):
        @mutation(MutationType.LAYER)
        def layer_mut(self):
            return {}

        def __init__(self, device="cpu"):
            super().__init__(device)
            self.child = Child(device=device)

        def forward(self, x):
            return x

        def recreate_network(self):
            pass

        def change_activation(self, activation, output):
            self.changed = (activation, output)

    module = Parent(device="cpu")
    clone = module.clone()
    assert clone.child._node_mutation_methods == module.child._node_mutation_methods

    wrapper = EvolvableWrapper(Child(device="cpu"))
    wrapper.wrapped.changed = None
    wrapper.wrapped.change_activation = lambda activation, output: setattr(
        wrapper.wrapped,
        "changed",
        (activation, output),
    )
    wrapper.change_activation("ReLU", output=True)
    assert wrapper.wrapped.changed == ("ReLU", True)


def test_module_dict_additional_branches():
    class A(EvolvableModule):
        @mutation(MutationType.NODE)
        def mut(self):
            return {}

        def __init__(self, device="cpu", activation="ReLU"):
            super().__init__(device)
            self._activation = activation
            self.filtered = False
            self.changed = False

        @property
        def activation(self):
            return self._activation

        def forward(self, x):
            return x

        def recreate_network(self):
            pass

        def filter_mutation_methods(self, remove):
            self.filtered = True
            super().filter_mutation_methods(remove)

        def change_activation(self, activation, output):
            self.changed = True
            self._activation = activation

    m1, m2 = A(device="cpu", activation="ReLU"), A(device="cpu", activation="Tanh")
    md = ModuleDict({"m1": m1, "m2": m2})
    assert md.activation is None
    methods = md.get_mutation_methods()
    assert "m1.mut" in methods
    md.change_activation("Sigmoid", output=False)
    md.filter_mutation_methods("mut")
    assert md["m1"].changed and md["m2"].changed
    assert md["m1"].filtered and md["m2"].filtered
