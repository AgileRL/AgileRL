import pytest
import torch
from torch import nn

from agilerl.modules.dummy import DummyEvolvable


class TestDummyEvolvableInit:
    def test_from_module(self):
        module = DummyEvolvable(module=nn.Linear(10, 10), device="cpu")
        assert module.device == "cpu"
        assert module.module.weight.shape == (10, 10)
        assert module.module.bias.shape == (10,)

    def test_from_module_fn(self):
        module = DummyEvolvable(module_fn=lambda: nn.Linear(10, 10), device="cpu")
        assert module.device == "cpu"
        assert module.module.weight.shape == (10, 10)
        assert module.module.bias.shape == (10,)

    def test_module_fn_without_kwargs_sets_default(self):
        """Covers module_fn with module_kwargs=None -> sets module_kwargs={}."""
        module = DummyEvolvable(
            module_fn=lambda: nn.Linear(8, 4),
            device="cpu",
        )
        assert module.module.weight.shape == (4, 8)

    def test_from_module_path(self):
        """Covers init when module is provided (not module_fn)."""
        mod = nn.Linear(6, 2)
        module = DummyEvolvable(module=mod, device="cpu")
        assert module.module is mod
        assert module.module.weight.shape == (2, 6)

    def test_raises(self):
        with pytest.raises(ValueError) as e:
            DummyEvolvable(module_fn=None, module=None, device="cpu")
        assert "Either module or module_fn must be provided." in str(e.value)

    def test_from_module_fn_and_kwargs(self):
        module = DummyEvolvable(
            module_fn=lambda: nn.Linear(10, 10),
            module_kwargs={},
            device="cpu",
        )
        assert module.device == "cpu"
        assert module.module.weight.shape == (10, 10)


class TestDummyEvolvableGetattr:
    def test_getattr_module(self):
        """Covers __getattr__ when name == 'module'."""
        module = DummyEvolvable(module=nn.Linear(4, 4), device="cpu")
        assert module.module.weight.shape == (4, 4)
        m = getattr(module, "module")
        assert m is module.module

    def test_getattr_delegates_to_inner(self):
        """Covers __getattr__ delegation to inner module."""
        module = DummyEvolvable(module=nn.Linear(4, 4), device="cpu")
        assert hasattr(module, "weight")
        assert module.weight.shape == (4, 4)


class TestDummyEvolvableChangeActivation:
    def test_change_activation(self):
        module = DummyEvolvable(module=nn.Linear(4, 4), device="cpu")
        module.change_activation("ReLU", output=False)


class TestDummyEvolvableForward:
    def test_forward(self):
        module = DummyEvolvable(module=nn.Linear(4, 4), device="cpu")
        x = torch.randn(2, 4)
        out = module.forward(x)
        assert out.shape == (2, 4)
