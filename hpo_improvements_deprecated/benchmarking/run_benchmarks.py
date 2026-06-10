r"""Benchmarking suite runner for HPO algorithm comparison.

Orchestrates the 10 benchmark scenarios defined in configs/ by invoking
``agilerl.train`` as a subprocess for each selected benchmark.  All W&B runs
are logged to the ``hpo_improvements`` project with descriptive run names.

Example usage::

    # Run all benchmarks on GPU with the baseline HPO algorithm
    uv run python hpo_improvements/benchmarking/run_benchmarks.py \\
        --benchmarks all --hpo-name EvoHPO --device cuda

    # Run only the two DDPG benchmarks on CPU
    uv run python hpo_improvements/benchmarking/run_benchmarks.py \\
        -b 1 2 --hpo-name EvoHPO -d cpu

    # Dry-run to inspect the commands that would be executed
    uv run python hpo_improvements/benchmarking/run_benchmarks.py \\
        --benchmarks all --hpo-name MyHPO --device mps --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

BENCHMARKS: dict[int, dict[str, str]] = {
    1: {"config": "01_ddpg_simple.yaml", "algo": "DDPG", "complexity": "simple"},
    2: {"config": "02_ddpg_complex.yaml", "algo": "DDPG", "complexity": "complex"},
    3: {"config": "03_ppo_simple.yaml", "algo": "PPO", "complexity": "simple"},
    4: {"config": "04_ppo_complex.yaml", "algo": "PPO", "complexity": "complex"},
    5: {"config": "05_maddpg_simple.yaml", "algo": "MADDPG", "complexity": "simple"},
    6: {"config": "06_maddpg_complex.yaml", "algo": "MADDPG", "complexity": "complex"},
    7: {"config": "07_ippo_simple.yaml", "algo": "IPPO", "complexity": "simple"},
    8: {"config": "08_ippo_complex.yaml", "algo": "IPPO", "complexity": "complex"},
    9: {"config": "09_grpo_simple.yaml", "algo": "GRPO", "complexity": "simple"},
    10: {"config": "10_grpo_complex.yaml", "algo": "GRPO", "complexity": "complex"},
}

WANDB_PROJECT = "hpo_improvements"


def build_run_name(algo: str, complexity: str, hpo_name: str) -> str:
    """Build a descriptive W&B run name.

    Format: ``{Algorithm}-{complexity}-{hpo_name}-{YYYY-MM-DD_HH-MM-SS}``
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{algo}-{complexity}-{hpo_name}-{ts}"


def build_command(
    config_path: Path,
    run_name: str,
    device: str,
    *,
    save_elite: bool = True,
) -> list[str]:
    """Construct the ``agilerl.train`` CLI invocation for a single benchmark."""
    cmd = [
        sys.executable,
        "-m",
        "agilerl.train",
        "-m",
        str(config_path),
        "--wb",
        "--wandb-project",
        WANDB_PROJECT,
        "--wandb-run-name",
        run_name,
        "--device",
        device,
    ]
    if save_elite:
        cmd.append("--save-elite")
    return cmd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Run HPO benchmarking suite for AgileRL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_benchmark_table(),
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        required=True,
        help=(
            "Benchmark numbers to run (1-10), space-separated. "
            "Use 'all' to run every benchmark."
        ),
    )
    parser.add_argument(
        "--hpo-name",
        type=str,
        required=True,
        help="Name of the HPO algorithm being tested (appears in W&B run name).",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Torch device: cpu, cuda, mps (default: cpu).",
    )
    parser.add_argument(
        "--save-elite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the elite agent after training (default: on).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def resolve_benchmark_ids(raw: list[str]) -> list[int]:
    """Convert CLI input to a sorted list of unique benchmark IDs."""
    if "all" in [r.lower() for r in raw]:
        return sorted(BENCHMARKS.keys())

    ids: set[int] = set()
    for token in raw:
        try:
            bid = int(token)
        except ValueError:
            print(f"[WARNING] Ignoring invalid benchmark id: {token!r}")
            continue
        if bid not in BENCHMARKS:
            print(f"[WARNING] Benchmark {bid} does not exist (valid: 1-10), skipping.")
            continue
        ids.add(bid)

    if not ids:
        print("[ERROR] No valid benchmarks selected.")
        sys.exit(1)
    return sorted(ids)


def _benchmark_table() -> str:
    lines = ["\nAvailable benchmarks:\n"]
    for bid, info in sorted(BENCHMARKS.items()):
        lines.append(
            f"  {bid:>2}  {info['algo']:<8} {info['complexity']:<8} {info['config']}"
        )
    return "\n".join(lines)


def main() -> None:
    """Run selected benchmarks and print execution summary."""
    args = parse_args()
    benchmark_ids = resolve_benchmark_ids(args.benchmarks)

    print(f"\n{'=' * 60}")
    print("  HPO Benchmarking Suite")
    print(f"  HPO algorithm : {args.hpo_name}")
    print(f"  Device        : {args.device}")
    print(f"  Benchmarks    : {benchmark_ids}")
    print(f"  Save elite    : {args.save_elite}")
    print(f"  Dry run       : {args.dry_run}")
    print(f"{'=' * 60}\n")

    results: list[dict[str, object]] = []

    for bid in benchmark_ids:
        info = BENCHMARKS[bid]
        config_path = CONFIGS_DIR / info["config"]
        run_name = build_run_name(info["algo"], info["complexity"], args.hpo_name)
        cmd = build_command(
            config_path,
            run_name,
            args.device,
            save_elite=args.save_elite,
        )

        print(f"[{bid:>2}/{len(benchmark_ids)}] {info['algo']} ({info['complexity']})")
        print(f"       Config : {config_path}")
        print(f"       W&B run: {run_name}")
        print(f"       Command: {' '.join(cmd)}\n")

        if args.dry_run:
            results.append({"id": bid, "status": "dry-run"})
            continue

        t0 = time.monotonic()
        # Command is assembled from static templates and validated benchmark IDs.
        proc = subprocess.run(cmd, check=False)  # noqa: S603
        elapsed = time.monotonic() - t0

        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        results.append({"id": bid, "status": status, "elapsed_s": elapsed})
        print(f"       Result  : {status}  ({elapsed:.1f}s)\n")

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for r in results:
        elapsed = f"  ({r['elapsed_s']:.1f}s)" if "elapsed_s" in r else ""
        binfo = BENCHMARKS[r["id"]]
        print(
            f"  [{r['id']:>2}] {binfo['algo']:<8} {binfo['complexity']:<8} => {r['status']}{elapsed}"
        )
    print()


if __name__ == "__main__":
    main()
