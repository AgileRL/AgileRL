# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GPU ZeRO-3 packed-MoE gather/release: leaves partition and memory does not grow.

DeepSpeed ``init_distributed`` / ZeRO-3 leave process-global state, so the
scenario runs in a subprocess.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from agilerl.utils.zero3_patches import install_zero3_patches

NUM_MICROS = 4
NUM_EXPERTS = 4
NUM_LAYERS = 2
HIDDEN = 128
INTERMEDIATE = 256
BATCH = 2
SEQ = 8
MEMORY_SLACK_BYTES = 32 * 1024 * 1024
MIN_EXPERT_NUMEL = 10_000


class PackedMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE) * 0.02
        )

    def forward(
        self, hidden_states: torch.Tensor, expert_index: torch.Tensor
    ) -> torch.Tensor:
        flat = hidden_states.reshape(-1, hidden_states.size(-1))
        idx = expert_index.reshape(-1)
        up = self.up_proj[idx]
        down = self.down_proj[idx]
        gated = F.silu(torch.bmm(up, flat.unsqueeze(-1)).squeeze(-1))
        out = torch.bmm(down, gated.unsqueeze(-1)).squeeze(-1)
        return out.view_as(hidden_states)


class PackedMoEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixer = nn.Linear(HIDDEN, HIDDEN)
        self.experts = PackedMoE()

    def forward(
        self, hidden_states: torch.Tensor, expert_index: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.mixer(hidden_states)
        return hidden_states + self.experts(hidden_states, expert_index)


class TinyPackedMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([PackedMoEBlock() for _ in range(NUM_LAYERS)])
        self.head = nn.Linear(HIDDEN, HIDDEN)

    def forward(
        self, hidden_states: torch.Tensor, expert_index: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = checkpoint(
                layer,
                hidden_states,
                expert_index,
                use_reentrant=False,
            )
        return self.head(hidden_states)


def _packed_expert_snapshot(module: nn.Module) -> list[dict[str, object]]:
    """Status of packed expert ZeRO-3 params (the gather/release surface)."""
    rows: list[dict[str, object]] = []
    for name, param in module.named_parameters():
        if "experts" not in name:
            continue
        if not hasattr(param, "ds_status"):
            continue
        if getattr(param, "ds_persist", False):
            continue
        if param.ds_numel < MIN_EXPERT_NUMEL:
            continue
        status = param.ds_status
        status_name = status.name if hasattr(status, "name") else str(status)
        active = getattr(param, "ds_active_sub_modules", None)
        rows.append(
            {
                "name": name,
                "status": status_name,
                "data_numel": int(param.data.numel()),
                "active": sorted(active) if active else [],
            }
        )
    return rows


def _all_released(rows: list[dict[str, object]]) -> bool:
    return all(
        row["status"] == "NOT_AVAILABLE"
        and row["data_numel"] == 0
        and row["active"] == []
        for row in rows
    )


def _run_packed_moe_zero3_release() -> dict[str, object]:
    """Train a tiny packed-MoE ZeRO-3 model; report leaf status and memory."""
    # DeepSpeed; this GPU subprocess is the only caller.
    import deepspeed
    from deepspeed.utils import get_z3_leaf_modules

    torch.manual_seed(0)
    device = torch.device("cuda")

    ds_config = {
        "train_batch_size": BATCH * NUM_MICROS,
        "train_micro_batch_size_per_gpu": BATCH,
        "gradient_accumulation_steps": NUM_MICROS,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
        "zero_optimization": {
            "stage": 3,
            "leaf_module": {
                "name_suffixes": ["experts"],
                "classes": ["PackedMoE"],
            },
            "stage3_param_persistence_threshold": 1024,
            "stage3_prefetch_bucket_size": 0,
            "stage3_max_reuse_distance": 0,
            "stage3_max_live_parameters": 1_000_000_000,
        },
        "bf16": {"enabled": False},
        "fp16": {"enabled": False},
    }

    install_zero3_patches(ds_config, num_partitions=1)
    deepspeed.init_distributed(dist_backend="nccl")

    model = TinyPackedMoE()
    for name, param in model.named_parameters():
        if "experts" in name:
            param.requires_grad = False

    probe = torch.randn(BATCH, SEQ, HIDDEN)
    idx0 = torch.zeros(BATCH, SEQ, dtype=torch.long)
    idx1 = torch.ones(BATCH, SEQ, dtype=torch.long)
    with torch.no_grad():
        experts_differ = not torch.allclose(
            model.layers[0].experts(probe, idx0),
            model.layers[0].experts(probe, idx1),
        )

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[param for param in model.parameters() if param.requires_grad],
        config=ds_config,
    )
    engine.train()

    leaf_modules = set(get_z3_leaf_modules(engine.module))
    leaf_names = sorted(
        name for name, module in engine.module.named_modules() if module in leaf_modules
    )

    experts_used: set[int] = set()
    after_forward: list[list[dict[str, object]]] = []
    after_backward: list[list[dict[str, object]]] = []
    allocated_after_forward: list[int] = []
    allocated_after_backward: list[int] = []

    hidden = torch.randn(BATCH, SEQ, HIDDEN, device=device)
    for micro in range(NUM_MICROS):
        primary = micro % NUM_EXPERTS
        expert_index = torch.full(
            (BATCH, SEQ), primary, dtype=torch.long, device=device
        )
        expert_index[:, 1] = (primary + 1) % NUM_EXPERTS
        experts_used.update(int(value) for value in expert_index.unique().tolist())

        torch.cuda.synchronize()
        output = engine(hidden, expert_index)
        torch.cuda.synchronize()
        after_forward.append(_packed_expert_snapshot(engine.module))
        allocated_after_forward.append(int(torch.cuda.memory_allocated()))

        loss = output.float().pow(2).mean()
        engine.backward(loss)
        torch.cuda.synchronize()
        after_backward.append(_packed_expert_snapshot(engine.module))
        allocated_after_backward.append(int(torch.cuda.memory_allocated()))

    engine.step()

    forward_released = [_all_released(rows) for rows in after_forward]
    backward_released = [_all_released(rows) for rows in after_backward]
    forward_baseline = allocated_after_forward[0]
    backward_baseline = allocated_after_backward[0]
    memory_grew = any(
        allocated > forward_baseline + MEMORY_SLACK_BYTES
        for allocated in allocated_after_forward[1:]
    ) or any(
        allocated > backward_baseline + MEMORY_SLACK_BYTES
        for allocated in allocated_after_backward[1:]
    )

    return {
        "experts_differ": experts_differ,
        "leaf_names": leaf_names,
        "experts_used": sorted(experts_used),
        "forward_released": forward_released,
        "backward_released": backward_released,
        "allocated_after_forward": allocated_after_forward,
        "allocated_after_backward": allocated_after_backward,
        "after_forward": after_forward,
        "after_backward": after_backward,
        "memory_grew": memory_grew,
    }


@pytest.mark.gpu
class TestPackedMoeZero3LeafRelease:
    def test_leaves_partition_after_micros_and_memory_does_not_grow(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        pytest.importorskip("deepspeed")

        # Arrange
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        spoke_root = Path(__file__).resolve().parents[2]
        env = os.environ | {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "PYTHONPATH": os.pathsep.join(
                [str(spoke_root), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        }

        # Act
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        assert proc.returncode == 0, (
            f"packed-MoE ZeRO-3 release subprocess failed\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        result_line = next(
            line for line in proc.stdout.splitlines() if line.startswith("RESULT ")
        )
        result = json.loads(result_line.removeprefix("RESULT "))

        # Assert
        assert result["experts_differ"] is True
        assert any(name.endswith("experts") for name in result["leaf_names"]), result
        assert result["experts_used"] == list(range(NUM_EXPERTS)), result
        assert result["after_forward"], result
        assert all(result["after_forward"]), result
        assert result["after_backward"], result
        assert all(result["after_backward"]), result
        assert result["forward_released"] == [True] * NUM_MICROS, result
        assert result["backward_released"] == [True] * NUM_MICROS, result
        assert result["memory_grew"] is False, result


if __name__ == "__main__":
    print("RESULT " + json.dumps(_run_packed_moe_zero3_release()), flush=True)
