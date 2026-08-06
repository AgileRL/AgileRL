# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate FSDP2 gate suite jobs (dry-run by default)."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from metrics import (  # noqa: E402
    NvidiaSmiSampler,
    RunSummary,
    artifact_dir,
    default_artifact_root,
    git_sha,
    write_json,
)

PROFILE_RANK = {"t4_2x16": 0, "l4_2x24": 1, "prod_multi": 2}


def _load_suite(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _profile_ok(job: dict[str, Any], active: str) -> tuple[bool, str | None]:
    """Return (should_run, waiver_reason)."""
    required = job.get("requires_profile_at_least")
    waiver = job.get("waiver_on")
    if waiver == active:
        return False, f"waived on profile {active}"
    if required is None:
        return True, None
    if PROFILE_RANK.get(active, 0) < PROFILE_RANK.get(required, 99):
        return False, f"needs profile>={required}, active={active}"
    return True, None


def _fsdp_probe_cmd(
    *,
    fsdp_root: Path,
    job: dict[str, Any],
    backend: str,
    seed: int,
    out_dir: Path,
) -> list[str]:
    nproc = int(job.get("nproc", 2))
    probe = fsdp_root / "scripts/fsdp_gates/run_probe.py"
    cfg = job.get("probe_config")
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(probe),
        "--job-id",
        str(job["id"]),
        "--backend",
        backend,
        "--seed",
        str(seed),
        "--artifact-dir",
        str(out_dir),
    ]
    if cfg:
        cmd.extend(["--config", str(fsdp_root / cfg)])
    if job.get("fsdp") or backend == "fsdp2":
        cmd.append("--fsdp")
    if job.get("max_steps") is not None:
        cmd.extend(["--max-steps", str(job["max_steps"])])
    if job.get("algo"):
        cmd.extend(["--algo", str(job["algo"])])
    for key, value in (job.get("init_hp_overrides") or {}).items():
        cmd.extend(["--init-hp", f"{key}={value}"])
    for arg in job.get("args") or []:
        cmd.append(str(arg))
    return cmd


def _ds_probe_cmd(
    *,
    fsdp_root: Path,
    zero3_root: Path,
    job: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> list[str]:
    nproc = int(job.get("nproc", 2))
    accelerate_cfg = fsdp_root / "docs/migration/gates/accelerate_ds_2gpu.yaml"
    probe = fsdp_root / "scripts/fsdp_gates/run_probe_ds.py"
    cfg = job.get("probe_config")
    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        str(accelerate_cfg),
        "--num_processes",
        str(nproc),
        str(probe),
        "--job-id",
        str(job["id"]),
        "--backend",
        "deepspeed",
        "--seed",
        str(seed),
        "--artifact-dir",
        str(out_dir),
        "--zero3-root",
        str(zero3_root),
    ]
    if cfg:
        # Configs are mirrored under demos/llm/debugging/configs in both trees
        cmd.extend(["--config", str(zero3_root / cfg)])
    if job.get("max_steps") is not None:
        cmd.extend(["--max-steps", str(job["max_steps"])])
    if job.get("algo"):
        cmd.extend(["--algo", str(job["algo"])])
    for key, value in (job.get("init_hp_overrides") or {}).items():
        cmd.extend(["--init-hp", f"{key}={value}"])
    return cmd


def _generic_cmd(
    *,
    root: Path,
    job: dict[str, Any],
) -> list[str] | None:
    if job.get("manual") or job.get("kind") == "review":
        return None
    if job.get("command"):
        return ["bash", "-c", str(job["command"]).strip()]
    entry = job.get("entry") or job.get("entry_fsdp2")
    if not entry:
        return None
    launch = job.get("launch", "python")
    nproc = int(job.get("nproc", 1))
    path = root / entry
    extra = [str(a) for a in (job.get("args") or job.get("entry_args") or [])]
    if launch == "torchrun":
        return [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
            str(path),
            *extra,
        ]
    return [sys.executable, str(path), *extra]


def build_commands(
    suite: dict[str, Any],
    job: dict[str, Any],
    *,
    seed: int,
    artifact_root: Path,
) -> list[tuple[str, list[str], Path]]:
    """Return list of (backend, cmd, out_dir)."""
    fsdp_root = Path(suite["worktrees"]["fsdp"])
    zero3_root = Path(suite["worktrees"]["zero3"])
    built: list[tuple[str, list[str], Path]] = []
    for backend in job.get("backends", ["fsdp"]):
        out = artifact_dir(artifact_root, str(job["id"]), backend, seed)
        use_probe = bool(job.get("probe_config")) or str(job.get("entry", "")).endswith(
            "run_probe.py"
        )
        if backend in {"fsdp", "fsdp2"} and use_probe and backend != "deepspeed":
            probe_backend = "fsdp2" if (job.get("fsdp") or backend == "fsdp2") else backend
            cmd = _fsdp_probe_cmd(
                fsdp_root=fsdp_root,
                job=job,
                backend=probe_backend,
                seed=seed,
                out_dir=out,
            )
            built.append((backend, cmd, out))
            continue
        if backend == "deepspeed" and use_probe:
            cmd = _ds_probe_cmd(
                fsdp_root=fsdp_root,
                zero3_root=zero3_root,
                job=job,
                seed=seed,
                out_dir=out,
            )
            built.append((backend, cmd, out))
            continue
        if backend == "fsdp2" and job.get("entry_fsdp2"):
            root = fsdp_root
            j = dict(job)
            j["entry"] = job["entry_fsdp2"]
            cmd = _generic_cmd(root=root, job=j)
            if cmd:
                built.append((backend, cmd, out))
            continue
        root = fsdp_root if backend.startswith("fsdp") else zero3_root
        cmd = _generic_cmd(root=root, job=job)
        if cmd is None:
            continue
        built.append((backend, cmd, out))
    return built


def run_one(
    *,
    backend: str,
    cmd: list[str],
    out_dir: Path,
    job: dict[str, Any],
    worktree: Path,
    seed: int,
    sample_interval_s: float,
    execute: bool,
) -> int:
    """Dry-run or execute one backend command with nvidia-smi sampling."""
    manifest = {
        "job_id": job["id"],
        "backend": backend,
        "seed": seed,
        "cmd": cmd,
        "cwd": str(worktree),
        "git_sha": git_sha(worktree),
        "note": job.get("note"),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"[{job['id']}/{backend}/seed{seed}] {shlex.join(cmd)}")
    if not execute:
        print("  (dry-run)")
        return 0

    env = os.environ.copy()
    env["AGILERL_GATE_ARTIFACT_DIR"] = str(out_dir)
    if backend == "deepspeed":
        # Prefer zero3 checkout for imports
        env["PYTHONPATH"] = str(worktree) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

    sampler = NvidiaSmiSampler(out_dir / "smi_timeseries.jsonl", interval_s=sample_interval_s)
    sampler.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(worktree), env=env, check=False)
        status = "ok" if proc.returncode == 0 else "failed"
        code = proc.returncode
    except KeyboardInterrupt:
        status = "interrupted"
        code = 130
    wall = time.time() - t0
    peak_smi = sampler.stop()

    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        summary = json_load(summary_path)
        summary["peak_smi_used_mib"] = peak_smi
        summary["wall_s"] = summary.get("wall_s") or wall
        summary["status"] = status if status != "ok" else summary.get("status", status)
        summary["launch_cmd"] = cmd
        write_json(summary_path, summary)
    else:
        write_json(
            summary_path,
            RunSummary(
                job_id=str(job["id"]),
                backend=backend,
                seed=seed,
                algo=job.get("algo"),
                status=status,
                wall_s=wall,
                peak_smi_used_mib=peak_smi,
                git_sha=git_sha(worktree),
                worktree=str(worktree),
                launch_cmd=cmd,
            ).__dict__,
        )
    return code


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("docs/migration/gates/suite.yaml"),
        help="Suite YAML path (relative to wt-fsdp or absolute)",
    )
    parser.add_argument("--ids", nargs="*", help="Subset of job ids")
    parser.add_argument("--gate", type=int, help="Only jobs for this gate number")
    parser.add_argument(
        "--profile",
        default=None,
        help="Hardware profile name from suite (default: suite.defaults.profile)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run jobs (default is dry-run)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="After execute, run compare.py on paired jobs",
    )
    args = parser.parse_args(argv)

    suite_path = args.suite
    if not suite_path.is_absolute():
        suite_path = Path.cwd() / suite_path
        if not suite_path.is_file():
            suite_path = Path(__file__).resolve().parents[2] / args.suite

    suite = _load_suite(suite_path)
    profile = args.profile or suite.get("defaults", {}).get("profile", "t4_2x16")
    artifact_root = Path(
        os.environ.get("AGILERL_GATE_ARTIFACT_ROOT", suite.get("artifact_root", default_artifact_root()))
    )
    sample_interval = float(suite.get("defaults", {}).get("sample_interval_s", 1.0))
    fsdp_root = Path(os.environ.get("AGILERL_GATE_FSDP", suite["worktrees"]["fsdp"]))
    zero3_root = Path(os.environ.get("AGILERL_GATE_ZERO3", suite["worktrees"]["zero3"]))
    suite["worktrees"]["fsdp"] = str(fsdp_root)
    suite["worktrees"]["zero3"] = str(zero3_root)

    jobs = suite["jobs"]
    if args.gate is not None:
        jobs = [j for j in jobs if int(j.get("gate", -1)) == args.gate]
    if args.ids:
        wanted = set(args.ids)
        jobs = [j for j in jobs if j["id"] in wanted]

    exit_code = 0
    for job in jobs:
        ok, reason = _profile_ok(job, profile)
        if not ok:
            print(f"[{job['id']}] SKIP: {reason}")
            write_json(
                artifact_dir(artifact_root, str(job["id"]), "skipped", 0) / "summary.json",
                {
                    "job_id": job["id"],
                    "status": "skipped",
                    "reason": reason,
                    "profile": profile,
                },
            )
            continue
        if job.get("optional") and not args.execute:
            print(f"[{job['id']}] optional (listed in dry-run)")
        seeds = [int(s) for s in job.get("seeds", suite.get("defaults", {}).get("seeds", [0]))]
        for seed in seeds:
            for backend, cmd, out_dir in build_commands(
                suite, job, seed=seed, artifact_root=artifact_root
            ):
                worktree = fsdp_root if backend.startswith("fsdp") else zero3_root
                if backend == "deepspeed":
                    worktree = zero3_root
                code = run_one(
                    backend=backend,
                    cmd=cmd,
                    out_dir=out_dir,
                    job=job,
                    worktree=worktree,
                    seed=seed,
                    sample_interval_s=sample_interval,
                    execute=args.execute,
                )
                if code != 0:
                    exit_code = code
            if (
                args.execute
                and args.compare
                and set(job.get("backends", [])) >= {"deepspeed", "fsdp2"}
            ):
                cmp = subprocess.run(
                    [
                        sys.executable,
                        str(_SCRIPTS / "compare.py"),
                        str(artifact_root / job["id"]),
                    ],
                    check=False,
                )
                if cmp.returncode != 0:
                    exit_code = cmp.returncode
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
