# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time

import pytest
import torch
import torch.distributed as dist

from tests.rank_launch import rank_context, run_ranks

MODULE_STATE = "pristine"


def group_identity_worker(rank: int, world_size: int) -> tuple[int, int, int, int]:
    return rank, world_size, dist.get_rank(), dist.get_world_size()


def all_reduce_worker(rank: int, _world_size: int) -> float:
    value = torch.tensor([float(rank + 1)])
    dist.all_reduce(value)
    return value.item()


def extra_args_worker(rank: int, _world_size: int, scale: int, label: str) -> str:
    return f"{label}{rank * scale}"


def environ_worker(_rank: int, _world_size: int) -> tuple[str | None, str, str]:
    return (
        os.environ.get("AGILERL_RANK_LAUNCH_PROBE"),
        os.environ["RANK"],
        os.environ["WORLD_SIZE"],
    )


def module_state_worker(_rank: int, _world_size: int) -> str:
    return MODULE_STATE


def group_initialized_worker(_rank: int, _world_size: int) -> bool:
    return dist.is_initialized()


def failing_worker(rank: int, _world_size: int) -> None:
    if rank == 1:
        msg = "boom on rank one"
        raise ValueError(msg)


def dying_worker(rank: int, _world_size: int) -> None:
    if rank == 0:
        os._exit(3)
    time.sleep(60)


def silent_worker(_rank: int, _world_size: int) -> None:
    time.sleep(60)


class TestRankContext:
    @pytest.mark.skipif(sys.platform == "win32", reason="Windows has no forkserver")
    def test_linux_without_cuda_forks_from_the_forkserver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Act
        ctx = rank_context()

        # Assert
        assert ctx.get_start_method() == "forkserver"

    @pytest.mark.parametrize(
        ("platform", "cuda"),
        [("darwin", False), ("linux", True)],
        ids=["macos", "linux-cuda"],
    )
    def test_spawns_where_forking_is_unsafe(
        self, monkeypatch: pytest.MonkeyPatch, platform: str, cuda: bool
    ) -> None:
        # Arrange
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)

        # Act
        ctx = rank_context()

        # Assert
        assert ctx.get_start_method() == "spawn"


@pytest.mark.multi_rank
class TestRunRanks:
    def test_returns_each_ranks_result_in_rank_order(self) -> None:
        results = run_ranks(group_identity_worker, 2)

        assert results == [(0, 2, 0, 2), (1, 2, 1, 2)]

    def test_ranks_share_one_process_group(self) -> None:
        results = run_ranks(all_reduce_worker, 4)

        assert results == [10.0, 10.0, 10.0, 10.0]

    def test_passes_extra_args_after_rank_and_world_size(self) -> None:
        results = run_ranks(extra_args_worker, 2, 5, "x")

        assert results == ["x0", "x5"]

    def test_ranks_see_the_callers_environment_and_rendezvous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setenv("AGILERL_RANK_LAUNCH_PROBE", "set-by-test")

        # Act
        results = run_ranks(environ_worker, 2)

        # Assert
        assert results == [("set-by-test", "0", "2"), ("set-by-test", "1", "2")]

    def test_ranks_do_not_inherit_in_process_test_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(sys.modules[__name__], "MODULE_STATE", "mutated")

        # Act
        results = run_ranks(module_state_worker, 2)

        # Assert
        assert results == ["pristine", "pristine"]

    def test_backend_none_skips_the_process_group(self) -> None:
        results = run_ranks(group_initialized_worker, 1, backend=None)

        assert results == [False]

    def test_worker_error_names_the_rank_and_carries_its_traceback(self) -> None:
        with pytest.raises(AssertionError, match="rank 1 failed") as error:
            run_ranks(failing_worker, 2, backend=None)

        assert "ValueError: boom on rank one" in str(error.value)

    def test_rank_that_dies_fails_without_waiting_for_the_timeout(self) -> None:
        # Arrange
        start = time.monotonic()

        # Act
        with pytest.raises(RuntimeError, match="rank 0 exited with code 3"):
            run_ranks(dying_worker, 2, backend=None, timeout=120.0)

        # Assert
        assert time.monotonic() - start < 30.0

    def test_silent_rank_times_out_and_is_killed(self) -> None:
        # Arrange
        start = time.monotonic()

        # Act
        with pytest.raises(TimeoutError, match=r"ranks \[0\] did not finish"):
            run_ranks(silent_worker, 1, backend=None, timeout=1.0)

        # Assert
        assert time.monotonic() - start < 30.0
