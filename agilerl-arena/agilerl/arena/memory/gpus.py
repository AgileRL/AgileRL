# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Known accelerators and the free-form names that resolve to them."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory.specs import DeviceSpec, GiB


class GpuInfo(BaseModel):
    """One accelerator model a resource class can provide."""

    model_config = ConfigDict(frozen=True)

    # Canonical CUDA device name, matching ``torch.cuda.get_device_name`` so
    # the per-device CUDA-context constants resolve.
    name: str
    total_gib: float
    cc_major: int
    cc_minor: int
    bf16_tflops: float | None = None
    memory_gb_per_second: float | None = None
    all_reduce_gb_per_second: float | None = None
    all_reduce_latency_us: float | None = None
    # GPUs of one node talk over NVLink, not PCIe.
    nvlink: bool = False

    def device_spec(self) -> DeviceSpec:
        return DeviceSpec(
            total_bytes=int(self.total_gib * GiB),
            name=self.name,
            compute_capability=(self.cc_major, self.cc_minor),
            nvlink=self.nvlink,
            peak_bf16_flops=(
                None if self.bf16_tflops is None else self.bf16_tflops * 1e12
            ),
            memory_bytes_per_second=(
                None
                if self.memory_gb_per_second is None
                else self.memory_gb_per_second * 1e9
            ),
            all_reduce_bytes_per_second=(
                None
                if self.all_reduce_gb_per_second is None
                else self.all_reduce_gb_per_second * 1e9
            ),
            all_reduce_latency_seconds=(
                None
                if self.all_reduce_latency_us is None
                else self.all_reduce_latency_us * 1e-6
            ),
        )


# First substring match. A token that contains another stays above it.
# ``RTXPRO6000`` does not contain ``RTX6000``. ``total_gib`` is CUDA-visible
# memory. Throughput is datasheet dense BF16. All-reduce is one 8-GPU node;
# B200 bandwidth and every latency are assumed. Below SM 8.0 there are none.
GPU_CATALOGUE: tuple[tuple[str, GpuInfo], ...] = (
    (
        "GB300",
        # GB300 NVL72 GPU: 279 GB HBM3e at 8 TB/s. Dense BF16 is half of 5 PFLOPS.
        GpuInfo(
            name="NVIDIA GB300",
            total_gib=279,
            cc_major=10,
            cc_minor=0,
            bf16_tflops=2500,
            memory_gb_per_second=8000,
            all_reduce_gb_per_second=700,
            all_reduce_latency_us=10,
            nvlink=True,
        ),
    ),
    (
        "B300",
        # 288 GB is the HGX HBM3e capacity. Dense BF16 is the B200 peak.
        # All-reduce is assumed the same way as B200.
        GpuInfo(
            name="NVIDIA B300",
            total_gib=288,
            cc_major=10,
            cc_minor=0,
            bf16_tflops=2250,
            memory_gb_per_second=8000,
            all_reduce_gb_per_second=700,
            all_reduce_latency_us=10,
            nvlink=True,
        ),
    ),
    (
        "B200",
        GpuInfo(
            name="NVIDIA B200",
            total_gib=180,
            cc_major=10,
            cc_minor=0,
            bf16_tflops=2250,
            memory_gb_per_second=8000,
            all_reduce_gb_per_second=700,
            all_reduce_latency_us=10,
            nvlink=True,
        ),
    ),
    (
        "H200",
        GpuInfo(
            name="NVIDIA H200",
            total_gib=141,
            cc_major=9,
            cc_minor=0,
            bf16_tflops=989,
            memory_gb_per_second=4800,
            all_reduce_gb_per_second=350,
            all_reduce_latency_us=10,
            nvlink=True,
        ),
    ),
    (
        "H100",
        GpuInfo(
            name="NVIDIA H100 80GB HBM3",
            total_gib=79.11,
            cc_major=9,
            cc_minor=0,
            bf16_tflops=989,
            memory_gb_per_second=3350,
            all_reduce_gb_per_second=350,
            all_reduce_latency_us=10,
            nvlink=True,
        ),
    ),
    (
        "A100-80",
        GpuInfo(
            name="NVIDIA A100-SXM4-80GB",
            total_gib=79.25,
            cc_major=8,
            cc_minor=0,
            bf16_tflops=312,
            memory_gb_per_second=2039,
            all_reduce_gb_per_second=230,
            all_reduce_latency_us=10,
            nvlink=True,
        ),
    ),
    (
        "A100",
        GpuInfo(
            name="NVIDIA A100-SXM4-40GB",
            total_gib=40,
            cc_major=8,
            cc_minor=0,
            bf16_tflops=312,
            memory_gb_per_second=1555,
            all_reduce_gb_per_second=230,
            all_reduce_latency_us=10,
            nvlink=True,
        ),
    ),
    (
        "A10",
        # NVIDIA publishes no A10G datasheet; these are the A10's figures.
        GpuInfo(
            name="NVIDIA A10G",
            total_gib=24,
            cc_major=8,
            cc_minor=6,
            bf16_tflops=125,
            memory_gb_per_second=600,
            all_reduce_gb_per_second=20,
            all_reduce_latency_us=20,
        ),
    ),
    (
        "L40",
        GpuInfo(
            name="NVIDIA L40S",
            total_gib=48,
            cc_major=8,
            cc_minor=9,
            bf16_tflops=362,
            memory_gb_per_second=864,
            all_reduce_gb_per_second=20,
            all_reduce_latency_us=20,
        ),
    ),
    (
        "L4",
        GpuInfo(
            name="NVIDIA L4",
            total_gib=22.03,
            cc_major=8,
            cc_minor=9,
            bf16_tflops=121,
            memory_gb_per_second=300,
            all_reduce_gb_per_second=20,
            all_reduce_latency_us=20,
        ),
    ),
    (
        "RTXPRO6000",
        # Datasheet BF16 tensor peak is 1 PFLOP with 2:4 sparsity; dense is 500.
        # All-reduce is not on the datasheet (PCIe, no NVLink).
        GpuInfo(
            name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            total_gib=96,
            cc_major=12,
            cc_minor=0,
            bf16_tflops=500,
            memory_gb_per_second=1597,
            all_reduce_gb_per_second=20,
            all_reduce_latency_us=20,
        ),
    ),
    (
        "RTX6000ADA",
        # Dense BF16 tensor peak is 364.2. PCIe, no NVLink; all-reduce is not
        # on the datasheet.
        GpuInfo(
            name="NVIDIA RTX 6000 Ada Generation",
            total_gib=48,
            cc_major=8,
            cc_minor=9,
            bf16_tflops=364.2,
            memory_gb_per_second=960,
            all_reduce_gb_per_second=20,
            all_reduce_latency_us=20,
        ),
    ),
    (
        "QUADRORTX6000",
        GpuInfo(name="Quadro RTX 6000", total_gib=24, cc_major=7, cc_minor=5),
    ),
    (
        "RTX6000",
        GpuInfo(
            name="NVIDIA RTX 6000 Blackwell Server Edition",
            total_gib=96,
            cc_major=12,
            cc_minor=0,
            bf16_tflops=500,
            memory_gb_per_second=1597,
            all_reduce_gb_per_second=20,
            all_reduce_latency_us=20,
        ),
    ),
    ("T4", GpuInfo(name="Tesla T4", total_gib=16, cc_major=7, cc_minor=5)),
    (
        "V100",
        GpuInfo(name="Tesla V100-SXM2-16GB", total_gib=16, cc_major=7, cc_minor=0),
    ),
)


def lookup_gpu(gpu_type: str) -> GpuInfo | None:
    """Resolve a free-form ``gpu_type`` ("NVIDIA L4", "a100 80gb") to a known GPU."""
    # Longest catalogue token first; "A100-80" matches any A100 string with an "80".
    collapsed = re.sub(r"[^A-Z0-9]", "", gpu_type.upper().replace("NVIDIA", ""))
    for token, info in GPU_CATALOGUE:
        if token == "A100-80":
            if "A100" in collapsed and "80" in collapsed:
                return info
        elif token in collapsed:
            return info
    return None
