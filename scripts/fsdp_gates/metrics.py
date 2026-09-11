# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Gate-run metrics: schemas, CUDA peaks, and nvidia-smi sampling."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    """One training-step sample (in-process)."""

    step: int
    loss: float | None = None
    eval_metric: float | None = None
    step_ms: float | None = None
    tokens_per_sec: float | None = None
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    max_allocated_bytes: int | None = None
    max_reserved_bytes: int | None = None


@dataclass
class SmiSample:
    """One external nvidia-smi poll."""

    t_s: float
    gpu_index: int
    memory_used_mib: float
    memory_total_mib: float
    utilization_gpu: float | None = None


@dataclass
class RunSummary:
    """Roll-up written to ``summary.json`` for compare.py."""

    job_id: str
    backend: str
    seed: int
    algo: str | None = None
    status: str = "unknown"
    wall_s: float | None = None
    steps: int = 0
    final_loss: float | None = None
    final_eval: float | None = None
    mean_step_ms: float | None = None
    p95_step_ms: float | None = None
    mean_tokens_per_sec: float | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    peak_smi_used_mib: float | None = None
    git_sha: str | None = None
    worktree: str | None = None
    launch_cmd: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def artifact_dir(root: Path, job_id: str, backend: str, seed: int = 0) -> Path:
    """Return ``root/job_id/backend/seedN`` and create it."""
    path = root / job_id / backend / f"seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON object as a line."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def cuda_memory_snapshot(device: int | None = None) -> dict[str, int | None]:
    """Return current and peak CUDA allocator stats (bytes)."""
    try:
        import torch
    except ImportError:
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "max_allocated_bytes": None,
            "max_reserved_bytes": None,
        }
    if not torch.cuda.is_available():
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "max_allocated_bytes": None,
            "max_reserved_bytes": None,
        }
    if device is None:
        device = torch.cuda.current_device()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def reset_cuda_peak_stats(device: int | None = None) -> None:
    """Reset ``max_memory_*`` counters after warmup."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def query_nvidia_smi() -> list[SmiSample]:
    """Poll nvidia-smi once; empty list if unavailable."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    now = time.time()
    samples: list[SmiSample] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        util = (
            float(parts[3])
            if len(parts) > 3 and parts[3] not in {"", "[N/A]"}
            else None
        )
        samples.append(
            SmiSample(
                t_s=now,
                gpu_index=int(parts[0]),
                memory_used_mib=float(parts[1]),
                memory_total_mib=float(parts[2]),
                utilization_gpu=util,
            )
        )
    return samples


class NvidiaSmiSampler:
    """Background nvidia-smi sampler for fair cross-backend VRAM profiles."""

    def __init__(self, out_path: Path, interval_s: float = 1.0) -> None:
        self.out_path = out_path
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_used_mib = 0.0
        self._t0 = 0.0

    def start(self) -> None:
        """Start sampling on a daemon thread."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.out_path.exists():
            self.out_path.unlink()
        self._t0 = time.time()
        self._thread = threading.Thread(
            target=self._loop, name="nvidia-smi-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> float:
        """Stop sampler; return peak used MiB across GPUs/samples."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3)
        return self._peak_used_mib

    def _loop(self) -> None:
        while not self._stop.is_set():
            for sample in query_nvidia_smi():
                sample.t_s = sample.t_s - self._t0
                self._peak_used_mib = max(self._peak_used_mib, sample.memory_used_mib)
                append_jsonl(self.out_path, asdict(sample))
            self._stop.wait(self.interval_s)


class StepMetricsWriter:
    """Append per-step records and build a :class:`RunSummary`."""

    def __init__(self, out_dir: Path, summary: RunSummary) -> None:
        self.out_dir = out_dir
        self.summary = summary
        self.metrics_path = out_dir / "metrics.jsonl"
        self._step_ms: list[float] = []
        self._tps: list[float] = []
        if self.metrics_path.exists():
            self.metrics_path.unlink()

    def record(self, rec: StepRecord) -> None:
        """Write one step and update running aggregates."""
        mem = cuda_memory_snapshot()
        if rec.allocated_bytes is None:
            rec.allocated_bytes = mem["allocated_bytes"]
        if rec.reserved_bytes is None:
            rec.reserved_bytes = mem["reserved_bytes"]
        if rec.max_allocated_bytes is None:
            rec.max_allocated_bytes = mem["max_allocated_bytes"]
        if rec.max_reserved_bytes is None:
            rec.max_reserved_bytes = mem["max_reserved_bytes"]
        append_jsonl(self.metrics_path, asdict(rec))
        self.summary.steps = max(self.summary.steps, rec.step + 1)
        if rec.loss is not None:
            self.summary.final_loss = rec.loss
        if rec.eval_metric is not None:
            self.summary.final_eval = rec.eval_metric
        if rec.step_ms is not None:
            self._step_ms.append(rec.step_ms)
        if rec.tokens_per_sec is not None:
            self._tps.append(rec.tokens_per_sec)
        if rec.max_allocated_bytes is not None:
            prev = self.summary.peak_allocated_bytes or 0
            self.summary.peak_allocated_bytes = max(prev, rec.max_allocated_bytes)
        if rec.max_reserved_bytes is not None:
            prev = self.summary.peak_reserved_bytes or 0
            self.summary.peak_reserved_bytes = max(prev, rec.max_reserved_bytes)

    def finalize(self, *, status: str, wall_s: float | None = None) -> RunSummary:
        """Flush summary.json."""
        self.summary.status = status
        if wall_s is not None:
            self.summary.wall_s = wall_s
        if self._step_ms:
            ordered = sorted(self._step_ms)
            self.summary.mean_step_ms = sum(ordered) / len(ordered)
            self.summary.p95_step_ms = ordered[
                min(len(ordered) - 1, int(0.95 * len(ordered)))
            ]
        if self._tps:
            self.summary.mean_tokens_per_sec = sum(self._tps) / len(self._tps)
        write_json(self.out_dir / "summary.json", asdict(self.summary))
        return self.summary


def default_artifact_root() -> Path:
    """Resolve artifact root from env or `/tmp`."""
    return Path(os.environ.get("AGILERL_GATE_ARTIFACT_ROOT", "/tmp/agilerl-fsdp-gates"))


def git_sha(worktree: Path) -> str | None:
    """Best-effort HEAD sha for manifests."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(worktree),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None
