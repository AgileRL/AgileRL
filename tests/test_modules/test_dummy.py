import pytest
import torch
from torch import nn

from agilerl.modules.dummy import DummyEvolvable


def test_dummy_evolvable_from_module():
    module = DummyEvolvable(module=nn.Linear(10, 10), device="cpu")
    assert module.device == "cpu"
    assert module.module.weight.shape == (10, 10)
    assert module.module.bias.shape == (10,)


def test_dummy_evolvable_from_module_fn():
    module = DummyEvolvable(module_fn=lambda: nn.Linear(10, 10), device="cpu")
    assert module.device == "cpu"
    assert module.module.weight.shape == (10, 10)
    assert module.module.bias.shape == (10,)


def test_dummy_evolvable_module_fn_without_kwargs_sets_default():
    """Covers module_fn with module_kwargs=None -> sets module_kwargs={}."""
    module = DummyEvolvable(
        module_fn=lambda: nn.Linear(8, 4),
        device="cpu",
    )
    assert module.module.weight.shape == (4, 8)


def test_dummy_evolvable_from_module_path():
    """Covers init when module is provided (not module_fn)."""
    mod = nn.Linear(6, 2)
    module = DummyEvolvable(module=mod, device="cpu")
    assert module.module is mod
    assert module.module.weight.shape == (2, 6)


def test_dummy_evolvable_raises():
    with pytest.raises(ValueError) as e:
        DummyEvolvable(module_fn=None, module=None, device="cpu")
    assert "Either module or module_fn must be provided." in str(e.value)


def test_dummy_evolvable_from_module_fn_and_kwargs():
    module = DummyEvolvable(
        module_fn=lambda: nn.Linear(10, 10),
        module_kwargs={},
        device="cpu",
    )
    assert module.device == "cpu"
    assert module.module.weight.shape == (10, 10)


def test_dummy_evolvable_getattr_module():
    """Covers __getattr__ when name == 'module'."""
    module = DummyEvolvable(module=nn.Linear(4, 4), device="cpu")
    assert module.module.weight.shape == (4, 4)
    m = getattr(module, "module")
    assert m is module.module


def test_dummy_evolvable_getattr_delegates_to_inner():
    """Covers __getattr__ delegation to inner module."""
    module = DummyEvolvable(module=nn.Linear(4, 4), device="cpu")
    assert hasattr(module, "weight")
    assert module.weight.shape == (4, 4)


def test_dummy_evolvable_change_activation():
    module = DummyEvolvable(module=nn.Linear(4, 4), device="cpu")
    module.change_activation("ReLU", output=False)


def test_dummy_evolvable_forward():
    module = DummyEvolvable(module=nn.Linear(4, 4), device="cpu")
    x = torch.randn(2, 4)
    out = module.forward(x)
    assert out.shape == (2, 4)
