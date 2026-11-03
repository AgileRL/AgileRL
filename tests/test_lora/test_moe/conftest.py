# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from agilerl.lora.moe import grouped_gemm as moe_gemm
from agilerl.lora.moe import recompute as moe_recompute
from agilerl.lora.moe import routed as moe_routed


class ReferenceExpertKernels:
    """CPU stand-ins for the Triton expert kernels that record which ones ran."""

    def __init__(self) -> None:
        self.used: set[str] = set()
        self.kernels = moe_gemm.ExpertKernels(
            self.matmul,
            self.weight_grad,
            self.scatter_add,
            self.gather_scaled_rows,
            self.swiglu,
            self.swiglu_backward,
        )

    def matmul(self, x, weight, offs, lora):
        if lora is None:
            self.used.add("matmul")
            return F.grouped_mm(x, weight, offs=offs)
        self.used.add("lora_matmul")
        xa, lora_b = lora
        return F.grouped_mm(x, weight, offs=offs) + F.grouped_mm(xa, lora_b, offs=offs)

    def weight_grad(self, x, grad, offs):
        self.used.add("weight_grad")
        return F.grouped_mm(x.t(), grad, offs=offs)

    def scatter_add(self, out, index, rows, weights):
        self.used.add("scatter_add")
        values = rows.float() if weights is None else rows.float() * weights.float()
        out.index_add_(0, index, values)

    def gather_scaled_rows(self, grad, index, rows, weights):
        self.used.add("gather_scaled_rows")
        gathered = grad.index_select(0, index)
        grad_rows = (gathered * weights.float()).to(rows.dtype)
        return grad_rows, (gathered * rows.float()).sum(-1, keepdim=True)

    def swiglu(self, projected):
        self.used.add("swiglu")
        gate, up = projected.chunk(2, dim=-1)
        return F.silu(gate) * up

    def swiglu_backward(self, grad, projected):
        self.used.add("swiglu_backward")
        gate, up = projected.chunk(2, dim=-1)
        sigmoid = torch.sigmoid(gate)
        grad_gate = grad * up * sigmoid * (1 + gate * (1 - sigmoid))
        return torch.cat([grad_gate, grad * gate * sigmoid], dim=-1)


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
