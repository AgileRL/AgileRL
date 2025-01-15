import pytest
from unittest.mock import MagicMock
from agilerl.modules.base import EvolvableModule, is_evolvable, mutation
from agilerl.protocols import MutationType
from torch import nn

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
    assert module.init_dict == {"device": "cpu"}

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
    assert clone.init_dict == module.init_dict

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
    assert mutation_methods["attribute_module.attribute_mutation"]._mutation_type == MutationType.NODE