"""Evolvable network modules.

Submodules are loaded lazily so leaf imports such as
``agilerl.modules.configs`` do not pull in ``agilerl.typing`` (and create an
import cycle).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "EvolvableBERT",
    "EvolvableCNN",
    "EvolvableGPT",
    "EvolvableLSTM",
    "EvolvableMLP",
    "EvolvableModule",
    "EvolvableMultiInput",
    "EvolvableResNet",
    "EvolvableSimBa",
    "EvolvableWrapper",
    "GumbelSoftmax",
    "ModuleDict",
    "NewGELU",
    "NoisyLinear",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "EvolvableBERT": (".bert", "EvolvableBERT"),
    "EvolvableCNN": (".cnn", "EvolvableCNN"),
    "EvolvableGPT": (".gpt", "EvolvableGPT"),
    "EvolvableLSTM": (".lstm", "EvolvableLSTM"),
    "EvolvableMLP": (".mlp", "EvolvableMLP"),
    "EvolvableModule": (".base", "EvolvableModule"),
    "EvolvableMultiInput": (".multi_input", "EvolvableMultiInput"),
    "EvolvableResNet": (".resnet", "EvolvableResNet"),
    "EvolvableSimBa": (".simba", "EvolvableSimBa"),
    "EvolvableWrapper": (".base", "EvolvableWrapper"),
    "GumbelSoftmax": (".custom_components", "GumbelSoftmax"),
    "ModuleDict": (".base", "ModuleDict"),
    "NewGELU": (".custom_components", "NewGELU"),
    "NoisyLinear": (".custom_components", "NoisyLinear"),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401 -- lazy re-export of heterogeneous symbols
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from exc
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
