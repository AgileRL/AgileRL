# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU and memory held back from a GPU worker.

A GPU worker runs the kubelet, a few system DaemonSets, and one workload pod.
``kubeReserved`` follows that pod count. The CPU reservation stays under one
core, and a resource class counts whole vCPUs, so one vCPU is held back.
Two GiB covers kube-reserved at that pod count, the 100 Mi eviction threshold,
those DaemonSets, and the gap between the advertised instance size and the
memory the OS reports.
"""

KUBELET_RESERVED_CPUS = 1
KUBELET_RESERVED_MEMORY_GIB = 2
