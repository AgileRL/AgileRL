"""Tests for ``reproducibility.seed_everything``.

The function mutates process-global RNG/thread/env state; the autouse
``_isolate_global_state`` fixture in ``conftest.py`` snapshots and restores it.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

import reproducibility
from reproducibility import seed_everything


class TestDeterminism:
    def test_same_seed_reproduces_random_numpy_torch(self):
        seed_everything(123)
        a = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
        seed_everything(123)
        b = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
        assert a[0] == b[0]
        assert a[1] == b[1]
        assert a[2] == b[2]

    def test_different_seeds_differ(self):
        seed_everything(1)
        a = np.random.rand(5).tolist()
        seed_everything(2)
        b = np.random.rand(5).tolist()
        assert a != b


class TestThreadPinning:
    def test_num_threads_pinned_to_one(self):
        seed_everything(0)
        assert torch.get_num_threads() == 1

    def test_idempotent_no_raise_on_repeated_calls(self):
        # set_num_interop_threads raises on the second call; it must be swallowed.
        seed_everything(0)
        seed_everything(0)  # must not raise
        seed_everything(7)  # must not raise


class TestDeterministicFlag:
    def test_default_sets_cublas_and_cudnn(self, monkeypatch):
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        seed_everything(0)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_false_does_not_set_cublas_in_clean_env(self, monkeypatch):
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        seed_everything(0, deterministic=False)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ

    def test_deterministic_is_keyword_only(self):
        with pytest.raises(TypeError):
            seed_everything(0, True)  # type: ignore[misc]


class TestCudaBranch:
    # ``set_global_seed`` (agilerl) calls ``torch.cuda.manual_seed_all``
    # unconditionally, so stub it out to isolate ``seed_everything``'s own
    # ``if torch.cuda.is_available()`` branch.
    def test_cuda_seeded_when_available(self, monkeypatch):
        calls = []
        monkeypatch.setattr(reproducibility, "set_global_seed", lambda s: None)
        monkeypatch.setattr(reproducibility.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            reproducibility.torch.cuda,
            "manual_seed_all",
            lambda s: calls.append(s),
        )
        seed_everything(99)
        assert calls == [99]

    def test_cuda_skipped_when_unavailable(self, monkeypatch):
        calls = []
        monkeypatch.setattr(reproducibility, "set_global_seed", lambda s: None)
        monkeypatch.setattr(reproducibility.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            reproducibility.torch.cuda,
            "manual_seed_all",
            lambda s: calls.append(s),
        )
        seed_everything(99)
        assert calls == []
