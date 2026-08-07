# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Gate-run logging tooling for comprehensive FSDP2 / DP experiment data.

Provides components that wrap training runs without modifying
``finetune_llm_multiturn`` or algorithm classes:

- :class:`EnvManifest`         — capture system state at run start
- :class:`StdoutTee`          — tee stdout/stderr to artifact-dir log files
- :class:`JsonlLogger`        — AgileRL ``Logger`` subclass writing per-step metrics to JSONL
- :class:`LossCurveParser`    — parse captured stdout for per-step loss/eval (fallback)
- :class:`RankHealthMonitor`  — allreduce-based NaN/Inf / divergence detection
- :class:`ParamChecksum`      — gather LoRA params and SHA256-hash for parity checks
- :class:`RunArtifactCollector` — bundle all artifacts into a structured directory
- :func:`patched_init_loggers` — context manager injecting ``JsonlLogger`` into training
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator

from metrics import (  # noqa: E402
    RunSummary,
    StepMetricsWriter,
    StepRecord,
    append_jsonl,
    cuda_memory_snapshot,
    git_sha,
    query_nvidia_smi,
    write_json,
)


# ---------------------------------------------------------------------------
# EnvManifest
# ---------------------------------------------------------------------------


@dataclass
class EnvManifest:
    """System state captured at run start; written to ``manifest.json``."""

    torch_version: str | None = None
    cuda_version: str | None = None
    cuda_home: str | None = None
    python_version: str | None = None
    hostname: str | None = None
    os_release: str | None = None
    git_sha: str | None = None
    git_branch: str | None = None
    git_diff_stat: str | None = None
    gpus: list[dict[str, Any]] = field(default_factory=list)
    driver_version: str | None = None
    nproc: int = 1
    env_vars: dict[str, str] = field(default_factory=dict)
    fsdp_config: dict[str, Any] | None = None
    package_versions: dict[str, str] = field(default_factory=dict)

    def capture(self, repo: Path, fsdp_config: dict | None = None) -> "EnvManifest":
        """Populate fields from the live system."""
        try:
            import torch

            self.torch_version = torch.__version__
            self.cuda_version = getattr(torch.version, "cuda", None)
        except ImportError:
            pass

        self.cuda_home = os.environ.get("CUDA_HOME")
        self.python_version = sys.version.split()[0] if sys.version else None
        self.hostname = os.environ.get("HOSTNAME") or _hostname_fallback()
        self.os_release = _os_release_fallback()
        self.git_sha = git_sha(repo)
        self.git_branch = _git_branch(repo)
        self.git_diff_stat = _git_diff_stat(repo)

        smi = query_nvidia_smi()
        if smi:
            self.driver_version = _nvidia_driver_version()
            for s in smi:
                self.gpus.append(
                    {"index": s.gpu_index, "memory_total_mib": s.memory_total_mib}
                )

        self.nproc = int(os.environ.get("WORLD_SIZE", "1"))
        for var in (
            "CUDA_VISIBLE_DEVICES", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
            "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE",
        ):
            if var in os.environ:
                self.env_vars[var] = os.environ[var]

        if fsdp_config is not None:
            self.fsdp_config = dict(fsdp_config)

        self.package_versions = _package_versions()
        return self

    def write(self, out_dir: Path) -> None:
        """Write ``manifest.json`` to *out_dir*."""
        write_json(out_dir / "manifest.json", asdict(self))


def _hostname_fallback() -> str | None:
    try:
        return subprocess.run(
            ["hostname"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:
        return None


def _os_release_fallback() -> str | None:
    try:
        p = Path("/etc/os-release")
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip('"')
    except Exception:
        pass
    return None


def _git_branch(repo: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, check=False,
        )
        return proc.stdout.strip() or None if proc.returncode == 0 else None
    except Exception:
        return None


def _git_diff_stat(repo: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(repo), capture_output=True, text=True, check=False,
        )
        return proc.stdout.strip() or None if proc.returncode == 0 else None
    except Exception:
        return None


def _nvidia_driver_version() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import importlib.metadata as md

        for pkg in ("torch", "vllm", "transformers", "deepspeed", "peft", "accelerate"):
            try:
                versions[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                pass
    except Exception:
        pass
    return versions

# ---------------------------------------------------------------------------
# StdoutTee
# ---------------------------------------------------------------------------


class _TeeStream:
    """File-like wrapper that writes to both the original stream and a file."""

    def __init__(self, original: Any, file: io.TextIOBase, lock: threading.Lock) -> None:
        self._original = original
        self._file = file
        self._lock = lock

    def write(self, data: str) -> int:
        with self._lock:
            self._file.write(data)
            self._file.flush()
        return self._original.write(data)

    def flush(self) -> None:
        self._file.flush()
        flush = getattr(self._original, "flush", None)
        if flush:
            flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class StdoutTee:
    """Context manager: tee ``sys.stdout`` / ``sys.stderr`` to files.

    Lines go to the original streams AND to the log files so console
    output is unchanged while everything is captured.
    """

    def __init__(self, stdout_path: Path, stderr_path: Path) -> None:
        self._stdout_path = stdout_path
        self._stderr_path = stderr_path
        self._lock = threading.Lock()
        self._stdout_file: io.TextIOBase | None = None
        self._stderr_file: io.TextIOBase | None = None
        self._old_stdout: Any = None
        self._old_stderr: Any = None

    def __enter__(self) -> "StdoutTee":
        self._stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout_file = self._stdout_path.open("w", encoding="utf-8")
        self._stderr_file = self._stderr_path.open("w", encoding="utf-8")
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _TeeStream(self._old_stdout, self._stdout_file, self._lock)
        sys.stderr = _TeeStream(self._old_stderr, self._stderr_file, self._lock)
        return self

    def __exit__(self, *exc: Any) -> None:
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        if self._stdout_file:
            self._stdout_file.close()
        if self._stderr_file:
            self._stderr_file.close()

# ---------------------------------------------------------------------------
# JsonlLogger — AgileRL Logger subclass
# ---------------------------------------------------------------------------

_Logger: Any = None
try:
    from agilerl.logger import Logger as _Logger
except Exception:
    pass


_JsonlBase = _Logger if _Logger is not None else object


class JsonlLogger(_JsonlBase):  # type: ignore[misc, valid-type]
    """AgileRL ``Logger`` that appends ``MetricsReport.to_dict()`` to a JSONL file.

    Installed via :func:`patched_init_loggers` so ``finetune_llm_*``
    calls ``write`` on every ``population.report_metrics()`` without
    changes to training code.
    """

    def __init__(self, path: Path) -> None:
        if _Logger is not None:
            super().__init__()  # type: ignore[misc]
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")

    def write(self, report: Any) -> None:
        """Append one metrics dict as a JSON line."""
        data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        append_jsonl(self._path, data)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# patched_init_loggers context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patched_init_loggers(
    jsonl_path: Path,
) -> Generator[None, None, None]:
    """Patch ``init_loggers`` to inject a :class:`JsonlLogger` into training.

    ``finetune_llm_*`` imports ``init_loggers`` at module level, so the
    binding in the importing module must be patched, not just the
    definition in ``agilerl.utils.utils``. This patches both the source
    module and every training module that has its own import binding.

    Usage::

        with patched_init_loggers(out_dir / "train_metrics.jsonl"):
            finetune_llm_multiturn(...)

    The patch is reverted on exit, even if the training call raises.
    """
    import agilerl.utils.utils as agilerl_utils

    original = agilerl_utils.init_loggers
    logger = JsonlLogger(jsonl_path)

    def patched(*args: Any, **kwargs: Any) -> list:
        loggers = original(*args, **kwargs)
        loggers.append(logger)
        return loggers

    # Patch the source module and every training module that imported
    # init_loggers as a top-level name binding.
    patched_modules = [agilerl_utils]
    for mod_path in (
        "agilerl.training.llm.multiturn",
        "agilerl.training.llm.sft",
        "agilerl.training.llm.preference",
        "agilerl.training.llm.reasoning",
        "agilerl.training.train_off_policy",
        "agilerl.training.train_on_policy",
        "agilerl.training.train_bandits",
        "agilerl.training.train_multi_agent_off_policy",
        "agilerl.training.train_multi_agent_on_policy",
        "agilerl.training.train_offline",
    ):
        try:
            mod = __import__(mod_path, fromlist=["init_loggers"])
            if hasattr(mod, "init_loggers"):
                patched_modules.append(mod)
        except Exception:
            pass

    originals = {mod: getattr(mod, "init_loggers") for mod in patched_modules}
    for mod in patched_modules:
        setattr(mod, "init_loggers", patched)
    try:
        yield
    finally:
        for mod, orig in originals.items():
            setattr(mod, "init_loggers", orig)

# ---------------------------------------------------------------------------
# LossCurveParser — stdout fallback
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_RE = re.compile(r"[\u2500-\u257f]")  # Rich table box-drawing chars
_FLOAT_RE = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf|-inf)"
_LOSS_PATTERNS = [
    re.compile(r"(?:train/)?loss\s*" + _FLOAT_RE, re.IGNORECASE),
    re.compile(r"loss:\s*" + _FLOAT_RE, re.IGNORECASE),
    re.compile(r"Loss\s+" + _FLOAT_RE, re.IGNORECASE),
]
_EVAL_PATTERNS = [
    re.compile(r"(?:eval|accuracy|hit.?rate)\s*" + _FLOAT_RE, re.IGNORECASE),
    re.compile(r"eval:\s*" + _FLOAT_RE, re.IGNORECASE),
]
_STEP_PATTERNS = [
    re.compile(r"(?:step|global.?step)\s*[:=]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)/(\d+)\s*steps", re.IGNORECASE),
    re.compile(r"global\s*steps:\s*([\d_]+)", re.IGNORECASE),
]


class LossCurveParser:
    """Parse captured stdout lines to extract per-step metrics.

    Fallback for when :class:`JsonlLogger` is not available. Parses
    plain-text and Rich-table output for loss / eval / step values.
    """

    def __init__(self, writer: StepMetricsWriter) -> None:
        self._writer = writer
        self._records: list[dict[str, Any]] = []
        self._last_step = -1

    def feed_line(self, line: str) -> None:
        """Process one stdout line; emit a StepRecord if metrics are found."""
        clean = _ANSI_RE.sub("", line)
        clean = _BOX_RE.sub(" ", clean).strip()
        if not clean:
            return

        step = self._extract_step(clean)
        loss = self._extract_float(clean, _LOSS_PATTERNS)
        eval_m = self._extract_float(clean, _EVAL_PATTERNS)

        if step is not None or loss is not None:
            s = step if step is not None else self._last_step + 1
            self._last_step = s
            rec = {"step": s, "loss": loss, "eval_metric": eval_m}
            self._records.append(rec)
            self._writer.record(
                StepRecord(step=s, loss=loss, eval_metric=eval_m)
            )

    def feed_text(self, text: str) -> None:
        """Process a block of text line by line."""
        for line in text.splitlines():
            self.feed_line(line)

    def flush(self) -> list[dict[str, Any]]:
        """Return accumulated records and clear the buffer."""
        records = list(self._records)
        self._records.clear()
        return records

    @staticmethod
    def _extract_step(line: str) -> int | None:
        for pat in _STEP_PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    return int(m.group(1).replace("_", ""))
                except (ValueError, IndexError):
                    pass
        return None

    @staticmethod
    def _extract_float(line: str, patterns: list[re.Pattern[str]]) -> float | None:
        for pat in patterns:
            m = pat.search(line)
            if m:
                try:
                    return float(m.group(1).lower())
                except (ValueError, IndexError):
                    pass
        return None

# ---------------------------------------------------------------------------
# RankHealthMonitor
# ---------------------------------------------------------------------------


class RankHealthMonitor:
    """Allreduce-based per-rank health check after each ``learn()``.

    Detects NaN/Inf on any rank (Gate 2 hard fail) and cross-rank
    divergence (warning if max/min loss ratio > 2x). No-op if
    ``world_size == 1``.
    """

    def __init__(self, enabled: bool = True, out_path: Path | None = None) -> None:
        self._enabled = enabled
        self._out_path = out_path
        self._records: list[dict[str, Any]] = []

    def check(self, step: int, loss: float | None) -> dict[str, Any]:
        """Run an allreduce health check; return a status dict."""
        if not self._enabled or loss is None:
            return {"step": step, "enabled": self._enabled}

        try:
            import torch
            import torch.distributed as dist

            if not (dist.is_available() and dist.is_initialized()):
                return {"step": step, "enabled": False, "loss": loss}

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loss_t = torch.tensor(float(loss), device=device)

            is_finite = torch.isfinite(loss_t).float()
            all_finite = dist.all_reduce(is_finite, op=dist.ReduceOp.PRODUCT).item()
            has_nan = not bool(all_finite)

            min_loss = loss_t.clone()
            max_loss = loss_t.clone()
            dist.all_reduce(min_loss, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_loss, op=dist.ReduceOp.MAX)
            denom = max(abs(min_loss.item()), 1e-8)
            ratio = float(max_loss.item() / denom)
            diverged = ratio > 2.0

            status = {
                "step": step,
                "loss": float(loss),
                "has_nan": has_nan,
                "diverged": diverged,
                "min_loss": float(min_loss.item()),
                "max_loss": float(max_loss.item()),
                "ratio": ratio,
            }
        except Exception as exc:
            status = {"step": step, "error": str(exc), "loss": loss}

        self._records.append(status)
        if self._out_path is not None:
            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self._out_path, status)
        return status

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

# ---------------------------------------------------------------------------
# ParamChecksum
# ---------------------------------------------------------------------------


class ParamChecksum:
    """Gather LoRA params and SHA256-hash them for parity checks.

    Used by the clone-check gate to verify two agents initialised from
    the same seed produce identical parameter tensors, and by the
    save/load roundtrip gate to verify checkpoint fidelity.
    """

    def __init__(self, out_path: Path | None = None) -> None:
        self._out_path = out_path
        self._snapshots: dict[str, str] = {}

    def snapshot(self, name: str, model: Any) -> str:
        """Compute a SHA256 over all named parameters of *model*.

        Only floats are hashed; tensor dtype/shape are included so
        dtype-mismatch regressions are caught.
        """
        h = hashlib.sha256()
        try:
            for pname, p in model.named_parameters():
                h.update(pname.encode("utf-8"))
                h.update(str(tuple(p.shape)).encode("utf-8"))
                h.update(str(p.dtype).encode("utf-8"))
                h.update(p.detach().cpu().numpy().tobytes())
        except Exception as exc:
            h.update(f"error:{exc}".encode("utf-8"))
        digest = h.hexdigest()
        self._snapshots[name] = digest
        if self._out_path is not None:
            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self._out_path, self._snapshots)
        return digest

    def compare(self, name_a: str, name_b: str) -> bool:
        """Return True if two prior snapshots match exactly."""
        return self._snapshots.get(name_a) == self._snapshots.get(name_b)

    def snapshots(self) -> dict[str, str]:
        return dict(self._snapshots)

# ---------------------------------------------------------------------------
# RunArtifactCollector
# ---------------------------------------------------------------------------


@dataclass
class RunArtifactCollector:
    """Bundle all run artifacts into a structured directory.

    Layout::

        <out_dir>/
          manifest.json
          stdout.log
          stderr.log
          train_metrics.jsonl   (JsonlLogger output, per-step metrics dict)
          metrics.jsonl          (StepMetricsWriter output, StepRecord rows)
          rank_health.jsonl
          param_checksums.json
          summary.json
          checkpoints/   (copied if present)
    """

    out_dir: Path
    repo: Path
    fsdp_config: dict | None = None
    job_id: str = "gate"
    backend: str = "fsdp2"
    seed: int = 0
    algo: str | None = None

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = EnvManifest()
        self._run_summary = RunSummary(
            job_id=self.job_id,
            backend=self.backend,
            seed=self.seed,
            algo=self.algo,
        )
        self._step_writer = StepMetricsWriter(self.out_dir, self._run_summary)
        self._stdout_tee: StdoutTee | None = None
        self._loss_parser: LossCurveParser | None = None
        self._rank_health = RankHealthMonitor(
            out_path=self.out_dir / "rank_health.jsonl"
        )
        self._param_checksum = ParamChecksum(
            out_path=self.out_dir / "param_checksums.json"
        )
        self._start_time = 0.0

    # -- lifecycle --------------------------------------------------------

    def open(self) -> None:
        """Capture manifest, start stdout tee, and record start time."""
        self._manifest.capture(Path(self.repo), self.fsdp_config)
        self._manifest.write(self.out_dir)
        self._run_summary.git_sha = self._manifest.git_sha
        self._run_summary.worktree = str(self.repo)
        self._stdout_tee = StdoutTee(
            self.out_dir / "stdout.log", self.out_dir / "stderr.log"
        )
        self._stdout_tee.__enter__()
        self._loss_parser = LossCurveParser(self._step_writer)
        self._start_time = time.monotonic()

    def close(self, status: str = "ok") -> None:
        """Tear down tee, write run summary, and finalise step metrics."""
        if self._stdout_tee is not None:
            self._stdout_tee.__exit__(None, None, None)
            self._stdout_tee = None
        self._populate_final_loss_from_jsonl()
        elapsed = time.monotonic() - self._start_time
        self._step_writer.finalize(status=status, wall_s=elapsed)

    def _populate_final_loss_from_jsonl(self) -> None:
        """Read the last loss from train_metrics.jsonl into the run summary."""
        path = self.jsonl_path
        if not path.exists():
            return
        last_loss = None
        last_eval = None
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in ("train/mean_loss", "train/loss", "loss"):
                if k in row and row[k] is not None:
                    try:
                        last_loss = float(row[k])
                    except (ValueError, TypeError):
                        pass
                    break
            for k in ("train/mean_score", "train/accuracy", "eval"):
                if k in row and row[k] is not None:
                    try:
                        last_eval = float(row[k])
                    except (ValueError, TypeError):
                        pass
                    break
        if last_loss is not None:
            self._run_summary.final_loss = last_loss
        if last_eval is not None:
            self._run_summary.final_eval = last_eval

    # -- accessors --------------------------------------------------------

    @property
    def jsonl_path(self) -> Path:
        return self.out_dir / "train_metrics.jsonl"

    @property
    def rank_health(self) -> RankHealthMonitor:
        return self._rank_health

    @property
    def param_checksum(self) -> ParamChecksum:
        return self._param_checksum

    @property
    def loss_parser(self) -> LossCurveParser | None:
        return self._loss_parser

    @property
    def step_writer(self) -> StepMetricsWriter:
        return self._step_writer

    @property
    def run_summary(self) -> RunSummary:
        return self._run_summary

    # -- helpers ----------------------------------------------------------

    def collect_checkpoint(self, src: Path, name: str | None = None) -> Path | None:
        """Copy a checkpoint dir/file into ``<out_dir>/checkpoints/``."""
        if not src.exists():
            return None
        dst_dir = self.out_dir / "checkpoints"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_name = name or src.name
        dst = dst_dir / dst_name
        if src.is_dir():
            subprocess.run(
                ["cp", "-r", str(src), str(dst)], check=False,
            )
        else:
            subprocess.run(["cp", str(src), str(dst)], check=False)
        return dst if dst.exists() else None

    def __enter__(self) -> "RunArtifactCollector":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()