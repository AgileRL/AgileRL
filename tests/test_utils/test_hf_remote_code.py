# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cross-process Hugging Face remote-code lock."""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from multiprocessing.pool import AsyncResult, Pool
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from filelock import FileLock, Timeout
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, dynamic_module_utils

from agilerl.utils.hf_remote_code import (
    REMOTE_CODE_LOCK_NAME,
    load_remote_code,
    remote_code_lock,
)

NUM_LOADERS = 6

PROJ_WEIGHT = [[1.0, 2.0], [3.0, 4.0]]

SIGNAL_TIMEOUT_S = 120

CONFIG_SOURCE = """
from transformers import PretrainedConfig


class TinyRemoteConfig(PretrainedConfig):
    model_type = "tiny_remote"

    def __init__(self, tiny_field=0, ready_path=None, release_path=None, **kwargs):
        self.tiny_field = tiny_field
        self.ready_path = ready_path
        self.release_path = release_path
        super().__init__(**kwargs)
"""

# The model pauses in __init__, which from_pretrained runs before loading weights.
MODELING_SOURCE = """
import time
from pathlib import Path

from torch import nn
from transformers import PreTrainedModel

from .configuration_tiny_remote import TinyRemoteConfig


class TinyRemoteForCausalLM(PreTrainedModel):
    config_class = TinyRemoteConfig
    base_model_prefix = "tiny"

    def __init__(self, config):
        super().__init__(config)
        self.proj = nn.Linear(2, 2)
        self.post_init()
        if config.ready_path is not None:
            Path(config.ready_path).touch()
            deadline = time.monotonic() + 120
            while not Path(config.release_path).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("weight load was never released")
                time.sleep(0.05)

    def forward(self, x):
        return self.proj(x)
"""


def write_remote_code_checkpoint(
    model_dir: Path,
    ready_path: Path | None = None,
    release_path: Path | None = None,
) -> None:
    """Write a tiny checkpoint whose config and model classes live in checkpoint code.

    :param model_dir: Directory to write the checkpoint into.
    :type model_dir: Path
    :param ready_path: File the model touches when its weight load starts.
    :type ready_path: Path | None
    :param release_path: File the model waits for before its weight load continues.
    :type release_path: Path | None
    """
    model_dir.mkdir()
    # Padding widens the window in which transformers' copy is half-written.
    padding = "\n".join(f"# padding line {i}" for i in range(20_000))
    (model_dir / "configuration_tiny_remote.py").write_text(
        CONFIG_SOURCE + padding + "\n"
    )
    (model_dir / "modeling_tiny_remote.py").write_text(MODELING_SOURCE)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "tiny_remote",
                "architectures": ["TinyRemoteForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_tiny_remote.TinyRemoteConfig",
                    "AutoModelForCausalLM": (
                        "modeling_tiny_remote.TinyRemoteForCausalLM"
                    ),
                },
                "tiny_field": 7,
                "ready_path": None if ready_path is None else str(ready_path),
                "release_path": None if release_path is None else str(release_path),
            }
        )
    )
    save_file(
        {"proj.weight": torch.tensor(PROJ_WEIGHT), "proj.bias": torch.zeros(2)},
        model_dir / "model.safetensors",
        metadata={"format": "pt"},
    )


def load_remote_config(model_dir: str, start: threading.Barrier) -> tuple[str, int]:
    """Load the checkpoint config under the lock once every loader is ready.

    :param model_dir: Checkpoint directory.
    :type model_dir: str
    :param start: Barrier releasing every loader at once.
    :type start: threading.Barrier
    :return: The loaded config's class name and ``tiny_field``.
    :rtype: tuple[str, int]
    """
    start.wait()
    with remote_code_lock(True):
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    return type(config).__name__, config.tiny_field


def load_remote_model(model_dir: str) -> tuple[str, list[list[float]]]:
    """Load the checkpoint code under the lock, then the model weights outside it.

    :param model_dir: Checkpoint directory.
    :type model_dir: str
    :return: The loaded model's class name and ``proj`` weight.
    :rtype: tuple[str, list[list[float]]]
    """
    load_remote_code(model_dir, AutoModelForCausalLM, True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    return type(model).__name__, model.proj.weight.tolist()


def load_remote_model_together(
    model_dir: str, start: threading.Barrier
) -> tuple[str, list[list[float]]]:
    """Run :func:`load_remote_model` once every loader is ready.

    :param model_dir: Checkpoint directory.
    :type model_dir: str
    :param start: Barrier releasing every loader at once.
    :type start: threading.Barrier
    :return: The loaded model's class name and ``proj`` weight.
    :rtype: tuple[str, list[list[float]]]
    """
    start.wait()
    return load_remote_model(model_dir)


def lock_is_held(lock_path: Path) -> bool:
    """Whether another holder has the lock file, probed without blocking.

    :param lock_path: Lock file to probe.
    :type lock_path: Path
    :return: ``True`` when the probe cannot acquire the lock.
    :rtype: bool
    """
    try:
        with FileLock(lock_path, timeout=0):
            return False
    except Timeout:
        return True


def wait_for_file(path: Path, pending: AsyncResult) -> None:
    """Block until ``path`` exists, re-raising the loader's error if it finishes first.

    :param path: File to wait for.
    :type path: Path
    :param pending: The loader writing ``path``.
    :type pending: AsyncResult
    """
    deadline = time.monotonic() + SIGNAL_TIMEOUT_S
    while not path.exists():
        if pending.ready():
            pending.get()
            pytest.fail(f"loader finished without writing {path}")
        if time.monotonic() > deadline:
            pytest.fail(f"{path} never appeared")
        time.sleep(0.05)


def stop_pool(pool: Pool) -> None:
    """Let ``pool``'s workers exit on their own before the ``with`` block ends.

    Exiting the block terminates live workers with SIGTERM, which can hang a
    worker inside coverage's SIGTERM handler.

    :param pool: Pool whose tasks have all returned.
    :type pool: Pool
    """
    pool.close()
    pool.join()


class TestRemoteCodeLock:
    def test_concurrent_cold_cache_loads_return_custom_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        model_dir = tmp_path / "tiny_remote"
        write_remote_code_checkpoint(model_dir)
        # Spawned children read the cache dir from the environment when transformers imports.
        monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "modules"))
        ctx = multiprocessing.get_context("spawn")

        # Act
        with ctx.Manager() as manager, ctx.Pool(NUM_LOADERS) as pool:
            start = manager.Barrier(NUM_LOADERS)
            results = pool.starmap(
                load_remote_config,
                [(str(model_dir), start)] * NUM_LOADERS,
            )
            stop_pool(pool)

        # Assert
        assert results == [("TinyRemoteConfig", 7)] * NUM_LOADERS

    def test_nested_lock_reenters_and_releases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path))
        lock_path = tmp_path / REMOTE_CODE_LOCK_NAME

        with remote_code_lock(True), remote_code_lock(True):
            held_inside = lock_is_held(lock_path)

        assert held_inside
        assert not lock_is_held(lock_path)

    def test_releases_lock_after_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path))
        lock_path = tmp_path / REMOTE_CODE_LOCK_NAME
        held_inside: list[bool] = []
        error = "load failed"

        def fail_under_lock() -> None:
            with remote_code_lock(True):
                held_inside.append(lock_is_held(lock_path))
                raise RuntimeError(error)

        # Act
        with pytest.raises(RuntimeError, match="load failed"):
            fail_under_lock()

        # Assert
        assert held_inside == [True]
        assert not lock_is_held(lock_path)

    def test_skips_lock_without_remote_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path))

        with remote_code_lock(False):
            held_inside = lock_is_held(tmp_path / REMOTE_CODE_LOCK_NAME)

        assert not held_inside


class TestLoadRemoteCode:
    def test_concurrent_cold_cache_model_loads_return_custom_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        model_dir = tmp_path / "tiny_remote"
        write_remote_code_checkpoint(model_dir)
        monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "modules"))
        ctx = multiprocessing.get_context("spawn")

        # Act
        with ctx.Manager() as manager, ctx.Pool(NUM_LOADERS) as pool:
            start = manager.Barrier(NUM_LOADERS)
            results = pool.starmap(
                load_remote_model_together,
                [(str(model_dir), start)] * NUM_LOADERS,
            )
            stop_pool(pool)

        # Assert
        assert results == [("TinyRemoteForCausalLM", PROJ_WEIGHT)] * NUM_LOADERS

    def test_weight_load_runs_outside_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        ready_path = tmp_path / "weight_load_started"
        release_path = tmp_path / "weight_load_released"
        model_dir = tmp_path / "tiny_remote"
        write_remote_code_checkpoint(model_dir, ready_path, release_path)
        modules_cache = tmp_path / "modules"
        monkeypatch.setenv("HF_MODULES_CACHE", str(modules_cache))
        ctx = multiprocessing.get_context("spawn")

        # Act
        with ctx.Pool(1) as pool:
            pending = pool.apply_async(load_remote_model, (str(model_dir),))
            wait_for_file(ready_path, pending)
            held_during_weight_load = lock_is_held(
                modules_cache / REMOTE_CODE_LOCK_NAME
            )
            release_path.touch()
            result = pending.get(timeout=SIGNAL_TIMEOUT_S)
            stop_pool(pool)

        # Assert
        assert not held_during_weight_load
        assert result == ("TinyRemoteForCausalLM", PROJ_WEIGHT)

    def test_without_remote_code_leaves_cache_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_dir = tmp_path / "tiny_remote"
        write_remote_code_checkpoint(model_dir)
        modules_cache = tmp_path / "modules"
        monkeypatch.setattr(
            dynamic_module_utils, "HF_MODULES_CACHE", str(modules_cache)
        )

        load_remote_code(str(model_dir), AutoModelForCausalLM, False)

        assert not modules_cache.exists()
