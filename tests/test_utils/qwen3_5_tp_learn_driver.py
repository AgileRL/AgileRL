# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Head-node driver: run the Qwen3.5 tensor-parallel learn on two GPUs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import ray


@ray.remote(num_gpus=2)
def run() -> int:
    """Restrict the actor to two GPUs and return the pytest exit code."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    root = Path(__file__).resolve().parents[2]
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, agilerl; print(agilerl.__file__); print(torch.cuda.device_count())",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=root,
    )
    print(check.stdout, check.stderr, flush=True)
    lines = [line for line in check.stdout.splitlines() if line]
    if check.returncode != 0 or len(lines) != 2:
        return check.returncode or 2
    if "site-packages" in lines[0]:
        print(f"agilerl is the image copy: {lines[0]}", flush=True)
        return 2
    if lines[1] != "2":
        print(f"device_count is {lines[1]}", flush=True)
        return 2
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n0",
            "-m",
            "gpu",
            "-k",
            "qwen3_5 and not moe",
            "tests/test_algorithms/test_llms/test_parallel_learn.py",
            "-q",
        ],
        check=False,
        cwd=root,
    )
    return proc.returncode


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    os.chdir(root)
    ray.init(address="auto")
    from tests.test_utils.qwen3_5_tp_learn_driver import run as remote_run

    return int(ray.get(remote_run.remote()))


if __name__ == "__main__":
    raise SystemExit(main())
