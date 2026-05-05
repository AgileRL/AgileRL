"""GSPO algorithm variant built on top of GRPO."""

from __future__ import annotations

from typing import Any

from agilerl.algorithms.grpo import GRPO, _signatures_without_loss_type


class GSPO(GRPO):
    """GSPO loss variant of :class:`agilerl.algorithms.grpo.GRPO`.

    Paper: https://arxiv.org/abs/2507.18071
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a GSPO agent with fixed ``loss_type``."""
        super().__init__(*args, loss_type="gspo", **kwargs)


_GSPO_CLASS_SIG, _GSPO_INIT_SIG = _signatures_without_loss_type()
GSPO.__signature__ = _GSPO_CLASS_SIG
GSPO.__init__.__signature__ = _GSPO_INIT_SIG
