"""CISPO algorithm variant built on top of GRPO."""

from __future__ import annotations

from typing import Any

from agilerl.algorithms.grpo import GRPO, _signatures_without_loss_type


class CISPO(GRPO):
    """CISPO loss variant of :class:`agilerl.algorithms.grpo.GRPO`

    Paper: https://arxiv.org/abs/2506.13585
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a CISPO agent with fixed ``loss_type``."""
        super().__init__(*args, loss_type="cispo", **kwargs)


_CISPO_CLASS_SIG, _CISPO_INIT_SIG = _signatures_without_loss_type()
CISPO.__signature__ = _CISPO_CLASS_SIG
CISPO.__init__.__signature__ = _CISPO_INIT_SIG
