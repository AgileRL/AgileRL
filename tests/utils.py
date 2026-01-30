"""Test utilities for AgileRL, including GPU memory management."""

import functools
import gc
import os
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Callable

import cloudpickle
import torch
from typing_extensions import ParamSpec

AGILERL_PATH = Path(__file__).parent.parent

_P = ParamSpec("_P")


def spawn_new_process_for_each_test(f: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to spawn a new process for each test function."""

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # Check if we're already in a subprocess
        if os.environ.get("RUNNING_IN_SUBPROCESS") == "1":
            # If we are, just run the function directly
            return f(*args, **kwargs)

        import torch.multiprocessing as mp

        with suppress(RuntimeError):
            mp.set_start_method("spawn")

        # Create a process with environment variable set
        env = os.environ.copy()
        env["RUNNING_IN_SUBPROCESS"] = "1"
        env["COVERAGE_PROCESS_START"] = os.path.join(
            str(AGILERL_PATH), "pyproject.toml"
        )

        with tempfile.TemporaryDirectory() as tempdir:
            output_filepath = os.path.join(tempdir, "new_process.tmp")

            # `cloudpickle` allows pickling complex functions directly
            payload = (f, args, kwargs, output_filepath)
            input_bytes = cloudpickle.dumps(payload)

            repo_root = str(AGILERL_PATH.resolve())

            env = dict(env or os.environ)
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

            cmd = [sys.executable, "-m", "tests.subprocess_runner"]

            returned = subprocess.run(
                cmd, input=input_bytes, capture_output=True, env=env
            )

            # check if the subprocess is successful
            try:
                returned.check_returncode()
            except Exception as e:
                # wrap raised exception to provide more information
                raise RuntimeError(
                    f"Error raised in subprocess:\n{returned.stderr.decode()}"
                ) from e

    return wrapper


def get_physical_device_indices(devices: list[int]) -> list[int]:
    """
    Map logical CUDA device indices to physical device indices.

    When CUDA_VISIBLE_DEVICES is set, the logical device indices (0, 1, ...)
    may not match the physical GPU indices. This function performs the mapping.

    Args:
        devices: List of logical CUDA device indices.

    Returns:
        List of corresponding physical device indices.
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return devices
    visible_indices = [int(x) for x in visible_devices.split(",")]
    index_mapping = {i: physical for i, physical in enumerate(visible_indices)}
    return [index_mapping[i] for i in devices if i in index_mapping]


def force_gpu_memory_release() -> None:
    """
    Aggressively release GPU memory using all available methods.

    This combines multiple cleanup strategies used by vLLM and DeepSpeed:
    1. Unfreeze GC to allow collection of frozen objects
    2. Multiple garbage collection cycles
    3. CUDA cache clearing
    4. Host memory cache clearing (PyTorch 2.5+)
    """
    if not torch.cuda.is_available():
        return

    # Unfreeze GC - critical for vLLM which freezes objects during CUDA graph capture
    gc.unfreeze()

    # Multiple GC cycles to handle reference chains
    for _ in range(3):
        gc.collect()

    # Clear CUDA caches
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Clear IPC memory handles
    torch.cuda.ipc_collect()

    # Clear host memory cache (PyTorch 2.5+)
    try:
        torch._C._host_emptyCache()
    except AttributeError:
        pass  # Not available in older PyTorch versions

    # Reset memory stats
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except Exception:
        pass


def wait_for_gpu_memory_to_clear(
    *,
    devices: list[int] | None = None,
    threshold_bytes: int | None = None,
    threshold_ratio: float | None = None,
    timeout_s: float = 60,
) -> None:
    """
    Wait for GPU memory to be cleared below a threshold.

    Uses NVML for accurate memory measurement instead of PyTorch's view,
    which can be inaccurate due to caching and delayed deallocation.

    This is particularly important for tests that use DeepSpeed and vLLM,
    where GPU memory may not be immediately released after cleanup calls.

    Args:
        devices: List of CUDA device indices. If None, uses all available devices.
        threshold_bytes: Absolute memory threshold in bytes. If set, threshold_ratio is ignored.
        threshold_ratio: Memory usage ratio threshold (e.g., 0.4 = 40% of total memory).
            Default is 0.4 if neither threshold is specified. Note that CUDA context
            alone uses 1-2GB, so thresholds below 0.2 may be unrealistic.
        timeout_s: Timeout in seconds before raising an error.

    Raises:
        ValueError: If memory doesn't clear within timeout.

    Example:
        >>> wait_for_gpu_memory_to_clear(threshold_ratio=0.4, timeout_s=30)
        >>> wait_for_gpu_memory_to_clear(threshold_bytes=5 * 2**30)  # 5 GiB
    """
    if not torch.cuda.is_available():
        return

    # Set default threshold - 40% accounts for CUDA context overhead
    if threshold_bytes is None and threshold_ratio is None:
        threshold_ratio = 0.4

    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlInit,
            nvmlShutdown,
        )
    except ImportError:
        # Fallback if pynvml not available - just do aggressive cleanup
        force_gpu_memory_release()
        return

    if devices is None:
        devices = list(range(torch.cuda.device_count()))

    if not devices:
        return

    physical_devices = get_physical_device_indices(devices)

    nvmlInit()
    try:
        start_time = time.time()
        while True:
            # Try to release memory on each iteration
            force_gpu_memory_release()

            memory_status: dict[int, tuple[float, float]] = {}

            for device in physical_devices:
                handle = nvmlDeviceGetHandleByIndex(device)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                gb_used = mem_info.used / (2**30)
                gb_total = mem_info.total / (2**30)
                memory_status[device] = (gb_used, gb_total)

            # Check if all devices are below threshold
            if threshold_bytes is not None:
                threshold_gb = threshold_bytes / (2**30)
                all_clear = all(
                    used <= threshold_gb for used, _ in memory_status.values()
                )
                threshold_str = f"{threshold_gb:.1f} GiB"
            else:
                all_clear = all(
                    used / total <= threshold_ratio
                    for used, total in memory_status.values()
                )
                threshold_str = f"{threshold_ratio:.0%}"

            elapsed = time.time() - start_time

            if all_clear:
                print(f"GPU memory cleared on devices {devices} after {elapsed:.1f}s")
                break

            if elapsed >= timeout_s:
                status_str = ", ".join(
                    f"GPU{d}: {u:.2f}/{t:.2f} GiB ({u/t:.0%})"
                    for d, (u, t) in memory_status.items()
                )
                raise ValueError(
                    f"GPU memory not cleared after {timeout_s}s. "
                    f"Status: {status_str}. Threshold: {threshold_str}"
                )

            time.sleep(2)
    finally:
        nvmlShutdown()
