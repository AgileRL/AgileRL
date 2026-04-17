"""CISPO algorithm variant built on top of GRPO."""

from __future__ import annotations

from functools import partial
from inspect import Signature, signature
from typing import Any

from agilerl.algorithms.grpo import GRPO


def _signatures_without_loss_type() -> tuple[Signature, Signature]:
    """Build class and ``__init__`` signatures without ``loss_type``."""
    grpo_sig = signature(GRPO.__init__)
    class_params = [
        param
        for param in grpo_sig.parameters.values()
        if param.name not in {"self", "loss_type"}
    ]
    init_params = [
        param for param in grpo_sig.parameters.values() if param.name != "loss_type"
    ]
    return (
        grpo_sig.replace(parameters=class_params),
        grpo_sig.replace(parameters=init_params),
    )


class CISPO(GRPO):
    """CISPO loss variant of :class:`agilerl.algorithms.grpo.GRPO`."""

    _init_with_cispo = partial(GRPO.__init__, loss_type="cispo")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a CISPO agent with fixed ``loss_type``."""
        self._init_with_cispo(self, *args, **kwargs)


_CISPO_CLASS_SIG, _CISPO_INIT_SIG = _signatures_without_loss_type()
CISPO.__signature__ = _CISPO_CLASS_SIG
CISPO.__init__.__signature__ = _CISPO_INIT_SIG
