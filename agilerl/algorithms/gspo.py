"""GSPO algorithm variant built on top of GRPO."""

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


class GSPO(GRPO):
    """GSPO loss variant of :class:`agilerl.algorithms.grpo.GRPO`."""

    _init_with_gspo = partial(GRPO.__init__, loss_type="gspo")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a GSPO agent with fixed ``loss_type``."""
        self._init_with_gspo(self, *args, **kwargs)


_GSPO_CLASS_SIG, _GSPO_INIT_SIG = _signatures_without_loss_type()
GSPO.__signature__ = _GSPO_CLASS_SIG
GSPO.__init__.__signature__ = _GSPO_INIT_SIG
