# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU-offload optimizer wrapper for FSDP2 training."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
from torch.distributed.tensor import DTensor


class CPUOffloadOptimizer:
    """Wrap an optimizer so optimizer states (AdamW m+v) stay on CPU.

    States are moved to GPU only for ``step()``, then back to CPU.  Parameters
    and gradients remain on GPU throughout, so compute is unaffected.  With
    activation checkpointing, activations and optimizer states are never on
    GPU at the same time: peak memory becomes ``max(activations, opt_states)``
    instead of ``sum``.

    Handles FSDP2 ``DTensor`` optimizer states by swapping ``_local_tensor``
    between CPU and GPU while preserving the DTensor wrapper.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, pin_memory: bool = True
    ) -> None:
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._initialized = False

    def _to_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        if device == "cpu":
            moved = tensor.to("cpu")
            if self.pin_memory and not moved.is_pinned():
                return moved.pin_memory()
            return moved
        return tensor.to(device, non_blocking=True)

    def _move_states(self, device: str) -> None:
        # Non-fused, non-capturable Adam reads ``step`` on CPU; a device copy
        # forces a host sync per parameter in ``step()``.
        step_on_device = bool(
            self.optimizer.defaults.get("fused")
            or self.optimizer.defaults.get("capturable")
        )
        for p in self.optimizer.state:
            state = self.optimizer.state[p]
            for k, v in state.items():
                if k == "step" and not step_on_device:
                    continue
                if isinstance(v, DTensor):
                    new_dt = copy.copy(v)
                    new_dt._local_tensor = self._to_device(v._local_tensor, device)
                    state[k] = new_dt
                elif isinstance(v, torch.Tensor):
                    state[k] = self._to_device(v, device)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if not self._initialized:
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            self._initialized = True
            return result
        self._move_states("cuda")
        result = self.optimizer.step(closure)
        self._move_states("cpu")
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        if self._initialized:
            self._move_states("cuda")
        sd = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
        self.offload_states()

    def offload_states(self) -> None:
        """Move populated optimizer states to CPU; later steps round-trip them."""
        self._move_states("cpu")
        self._initialized = True

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        self.optimizer.param_groups = value

    @property
    def state(self) -> dict[torch.Tensor, Any]:
        return self.optimizer.state

    @property
    def base_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer
