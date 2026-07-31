# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Device-level peak-memory sampling via NVML.

``torch.cuda.max_memory_allocated`` only sees the torch caching allocator.
Colocated vLLM allocates its weights and KV pool through CuMem (to support
sleep/wake), which bypasses torch entirely — so the generation-phase peak is
only visible at the device level. A background thread polls
``nvmlDeviceGetMemoryInfo`` and records the max used-bytes over the window.
"""

from __future__ import annotations

import threading
import time
from types import TracebackType

from typing_extensions import Self


def wait_for_idle(
    device_index: int = 0,
    timeout_s: float = 120.0,
    settle_s: float = 3.0,
    poll_s: float = 0.5,
) -> int:
    """Block until device memory stops falling; return the settled floor.

    A sweep runs each point in its own subprocess, and the driver reclaims a
    dead process's memory asynchronously — so the next point can sample its
    baseline while its predecessor is still resident, contaminating
    ``device_peak_bytes`` for the whole point. Returns the floor even on
    timeout: a device genuinely busy with someone else's work should slow the
    sweep down, not fail it.

    :param device_index: CUDA device index to watch.
    :param timeout_s: Give up waiting after this long and return what is read.
    :param settle_s: Quiet period with no decrease that counts as settled.
    :param poll_s: Polling period.
    """
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        def used() -> int:
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)

        deadline = time.monotonic() + timeout_s
        floor = used()
        quiet_since = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(poll_s)
            current = used()
            if current < floor:
                floor, quiet_since = current, time.monotonic()
            elif time.monotonic() - quiet_since >= settle_s:
                return floor
        return min(floor, used())
    finally:
        pynvml.nvmlShutdown()


class NvmlPeakSampler:
    """Context manager recording device-level peak used-bytes over its scope.

    :param device_index: CUDA device index to watch.
    :param interval_s: Polling period. 10 ms resolves the allocation ramps of
        multi-second phases; short spikes between samples are missed, which
        is acceptable because torch-side spikes are cross-checked against
        ``max_memory_allocated``.
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.01) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self.peak_bytes = 0
        self.baseline_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        while not self._stop.is_set():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if info.used > self.peak_bytes:
                self.peak_bytes = int(info.used)
            time.sleep(self.interval_s)

    def __enter__(self) -> Self:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self.baseline_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        self.peak_bytes = self.baseline_bytes
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        import pynvml

        # Take one final sample so a peak right at scope exit is not missed.
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if info.used > self.peak_bytes:
            self.peak_bytes = int(info.used)
        pynvml.nvmlShutdown()
