# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ray and trainer ports a job service opens between its instances."""

from __future__ import annotations

from dataclasses import dataclass

# Every Ray port sits in 61001-61999: above the SPCS ephemeral range
# (32768-60999) and the trainer spans a job can accept.
RAY_GCS_PORT = 61001
RAY_CLIENT_SERVER_PORT = 61002
RAY_DASHBOARD_PORT = 61003
RAY_OBJECT_MANAGER_PORT = 61011
RAY_NODE_MANAGER_PORT = 61012
RAY_RUNTIME_ENV_AGENT_PORT = 61013
RAY_DASHBOARD_AGENT_GRPC_PORT = 61014
RAY_DASHBOARD_AGENT_LISTEN_PORT = 61015
RAY_METRICS_EXPORT_PORT = 61016
RAY_MIN_WORKER_PORT = 61031
RAY_MAX_WORKER_PORT = 61999

# Trainer bases from agilerl_ray/constants.py. Torch probes 16 ports from
# 29700 + agent * 16. vLLM slot agent * 64 + engine probes 32 ports from
# 10000 + slot * 32 and scans its data-parallel master 100 ports above
# 20000 + slot * 32.
DIST_PORT_BASE = 29700
DIST_PORT_PROBE = 16
VLLM_PORT_BASE = 10000
VLLM_PORT_PROBE = 32
VLLM_AGENT_STRIDE = 64
VLLM_DP_MASTER_PORT_BASE = 20000
VLLM_DP_SCAN_OFFSET = 100

# Only the head's Ray client server binds RAY_CLIENT_SERVER_PORT, and nothing
# connects to it from another instance.
RAY_PORTS = {
    "ray-gcs": RAY_GCS_PORT,
    "ray-dash": RAY_DASHBOARD_PORT,
    "ray-object": RAY_OBJECT_MANAGER_PORT,
    "ray-node": RAY_NODE_MANAGER_PORT,
    "ray-runtime": RAY_RUNTIME_ENV_AGENT_PORT,
    "ray-agent-grpc": RAY_DASHBOARD_AGENT_GRPC_PORT,
    "ray-agent-http": RAY_DASHBOARD_AGENT_LISTEN_PORT,
    "ray-metrics": RAY_METRICS_EXPORT_PORT,
}


@dataclass(frozen=True)
class TrainerPorts:
    """Population members and rollout engines per member the trainer binds for."""

    agents: int
    engines_per_agent: int


def trainer_port_spans(trainer: TrainerPorts) -> list[tuple[int, int]]:
    """Inclusive port spans the trainer may bind on any instance.

    :param trainer: Population size and engines per member.
    :return: ``(start, end)`` for vLLM, vLLM data-parallel masters, and torch.
    :raises ValueError: A span reaches the Ray ports.
    """
    last_slot = (trainer.agents - 1) * VLLM_AGENT_STRIDE + trainer.engines_per_agent - 1
    spans = [
        (VLLM_PORT_BASE, VLLM_PORT_BASE + (last_slot + 1) * VLLM_PORT_PROBE - 1),
        (
            VLLM_DP_MASTER_PORT_BASE,
            VLLM_DP_MASTER_PORT_BASE
            + last_slot * VLLM_PORT_PROBE
            + VLLM_DP_SCAN_OFFSET
            + VLLM_PORT_PROBE
            - 1,
        ),
        (DIST_PORT_BASE, DIST_PORT_BASE + trainer.agents * DIST_PORT_PROBE - 1),
    ]
    for start, end in spans:
        if start <= RAY_MAX_WORKER_PORT and end >= RAY_GCS_PORT:
            msg = (
                f"{trainer.agents} population members with "
                f"{trainer.engines_per_agent} rollout engines each need ports "
                f"{start}-{end}, which reach the Ray ports "
                f"{RAY_GCS_PORT}-{RAY_MAX_WORKER_PORT}. "
                "Lower training.pop_size or training.rollout_engines_per_agent."
            )
            raise ValueError(msg)
    return spans


def service_endpoints(trainer: TrainerPorts | None) -> list[dict[str, object]]:
    """TCP endpoints the job spec opens between instances.

    Ray's fixed ports keep their own endpoints. The Ray worker range, the
    ephemeral range, and, when ``trainer`` is set, the trainer spans are
    merged so no two overlap. The head service passes ``None``.

    :param trainer: Population size and engines per member, or ``None`` when
        the service opens no trainer spans.
    :return: Endpoint mappings for the service specification.
    :raises ValueError: A trainer span reaches the Ray ports.
    """
    endpoints: list[dict[str, object]] = [
        {"name": name, "port": port, "protocol": "TCP", "public": False}
        for name, port in RAY_PORTS.items()
    ]
    spans = [
        (RAY_MIN_WORKER_PORT, RAY_MAX_WORKER_PORT),
        (32768, 60999),
    ]
    if trainer is not None:
        spans.extend(trainer_port_spans(trainer))
    endpoints.extend(
        {
            "name": f"ports-{index}",
            "portRange": f"{start}-{end}",
            "protocol": "TCP",
            "public": False,
        }
        for index, (start, end) in enumerate(_merge(spans))
    )
    return endpoints


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
