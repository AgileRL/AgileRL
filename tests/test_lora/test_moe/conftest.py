# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch.nn.functional as F

from agilerl.lora.moe import grouped_gemm as moe_gemm
from agilerl.lora.moe import recompute as moe_recompute
from agilerl.lora.moe import routed as moe_routed


class ReferenceExpertKernels:
    """CPU stand-ins for the Triton expert kernels that record which ones ran."""

    def __init__(self) -> None:
        self.used: set[str] = set()
        self.kernels = moe_gemm.ExpertKernels(
            self.matmul, self.weight_grad, self.scatter_add
        )

    def matmul(self, x, weight, offs):
        self.used.add("matmul")
        return F.grouped_mm(x, weight, offs=offs)

    def weight_grad(self, x, grad, offs):
        self.used.add("weight_grad")
        return F.grouped_mm(x.t(), grad, offs=offs)

    def scatter_add(self, out, index, rows, weights):
        self.used.add("scatter_add")
        values = rows.float() if weights is None else rows.float() * weights.float()
        out.index_add_(0, index, values)


@pytest.fixture
def reference_expert_kernels(monkeypatch):
    """Route every expert GEMM and recompute scatter through :class:`ReferenceExpertKernels`."""
    reference = ReferenceExpertKernels()
    monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: False)
    monkeypatch.setattr(
        moe_gemm, "unaligned_gemm_kernels", lambda _x: reference.kernels
    )
    for module in (moe_gemm, moe_recompute, moe_routed):
        monkeypatch.setattr(module, "expert_kernels", lambda _x: reference.kernels)
    return reference
