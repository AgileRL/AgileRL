# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Algorithm implementations.

Re-exports are lazy; the public surface is declared in ``__init__.pyi``. LLM
algorithms are withheld unless the ``llm`` extra is installed.
"""

from collections.abc import Callable

import lazy_loader as lazy

from agilerl import HAS_LLM_DEPENDENCIES

_LLM_ALGORITHMS = frozenset(
    {"CISPO", "DPO", "GRPO", "GSPO", "LLMPPO", "LLMREINFORCE", "SFT"},
)


def _attach_gated() -> tuple[
    Callable[[str], object], Callable[[], list[str]], list[str]
]:
    """Build the lazy module hooks, hiding LLM algorithms without the extra.

    :return: The ``__getattr__``, ``__dir__`` and ``__all__`` module hooks.
    :rtype: tuple[Callable[[str], object], Callable[[], list[str]], list[str]]
    """
    lazy_getattr, _lazy_dir, lazy_all = lazy.attach_stub(__name__, __file__)
    exported = [
        *sorted(
            lazy_all if HAS_LLM_DEPENDENCIES else set(lazy_all) - _LLM_ALGORITHMS,
        ),
    ]

    def module_getattr(name: str) -> object:
        if name in _LLM_ALGORITHMS and not HAS_LLM_DEPENDENCIES:
            msg = (
                f"{name} requires the LLM extra for this platform: "
                "`pip install agilerl[llm]`."
            )
            raise AttributeError(msg)
        return lazy_getattr(name)

    def module_dir() -> list[str]:
        return exported

    return module_getattr, module_dir, exported


__getattr__, __dir__, __all__ = _attach_gated()
