# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Host GPU linker env for framework unit tests.

Must run before ``import torch``. GCP images export ``NCCL_NET=gIB`` and put
``/usr/local/gib/lib64`` first on ``LD_LIBRARY_PATH``. FSDP2 CUDA collectives
then fail to load a net plugin, and the CUDA 13 / cuDNN wheel libs stay
hidden from bitsandbytes, vLLM sleep mode, and LSTM.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# Overwrite, do not setdefault: the host already exports NCCL_NET=gIB.
_NCCL_TEST_ENV = {
    "NCCL_NET": "Socket",
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_NET_PLUGIN": "none",
    "NCCL_PROFILER_PLUGIN": "none",
    "NCCL_TUNER_PLUGIN": "none",
}

_PRELOAD_NAMES = (
    "libnvJitLink.so.13",
    "libnvrtc.so.13",
    "libcudnn.so.9",
)


def _site_packages() -> Path | None:
    for entry in map(Path, sys.path):
        if (entry / "nvidia").is_dir():
            return entry
    fallback = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    return fallback if (fallback / "nvidia").is_dir() else None


def _without_gib(ld_path: str) -> list[str]:
    return [
        part
        for part in ld_path.split(":")
        if part and "/gib/" not in part.replace("\\", "/").lower()
    ]


def _nvidia_lib_dirs(site: Path) -> list[str]:
    dirs = []
    for rel in ("nvidia/cu13/lib", "nvidia/cudnn/lib", "nvidia/nccl/lib"):
        path = site / rel
        if path.is_dir():
            dirs.append(str(path))
    return dirs


def apply() -> None:
    """Force Socket NCCL and put the wheel CUDA libs ahead of gIB."""
    for key, value in _NCCL_TEST_ENV.items():
        os.environ[key] = value

    tuner = os.environ.get("NCCL_TUNER_CONFIG_PATH", "")
    if "gib" in tuner.lower():
        os.environ.pop("NCCL_TUNER_CONFIG_PATH", None)

    site = _site_packages()
    nvidia_dirs = _nvidia_lib_dirs(site) if site is not None else []
    existing = _without_gib(os.environ.get("LD_LIBRARY_PATH", ""))
    ordered = []
    seen: set[str] = set()
    for part in [*nvidia_dirs, *existing]:
        if part not in seen:
            ordered.append(part)
            seen.add(part)
    if ordered:
        os.environ["LD_LIBRARY_PATH"] = ":".join(ordered)

    for directory in nvidia_dirs:
        for name in _PRELOAD_NAMES:
            lib = Path(directory) / name
            if lib.is_file():
                ctypes.CDLL(str(lib), mode=os.RTLD_GLOBAL)
