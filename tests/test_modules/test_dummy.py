import pytest
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
