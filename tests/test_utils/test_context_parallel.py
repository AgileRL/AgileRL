# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for context-parallel validation, sequence shard, and gather."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from agilerl.distributed import FSDPConfig
from agilerl.distributed.context_parallel import (
    CP_SUPPORTED_ALGOS,
    gather_for_cp,
    gather_for_cp_wo_grad,
    reject_unsupported_cp,
    shard_for_cp,
    shard_training_bundle,
    validate_cp_config,
    validate_cp_degree,
    validate_cp_ep_mix,
    validate_cp_heads,
    validate_cp_seq_len,
    validate_cp_style,
)
from agilerl.distributed.expert_parallel import build_parallel_mesh
from agilerl.utils.llm_packing import pack_padded_batch, pad_packed_row_for_cp

_DIST_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def _clean_dist_state():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    saved = {var: os.environ.pop(var, None) for var in _DIST_ENV}
    yield
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


def _gloo_available() -> bool:
    if sys.platform == "win32":
        return False
    return dist.is_available()


requires_gloo = pytest.mark.skipif(not _gloo_available(), reason="gloo unavailable")


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_ranks(worker, world_size: int = 2, timeout: float = 300.0) -> None:
    port = _free_port()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(rank, world_size, port, queue))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=timeout) for _ in range(world_size)]
    for proc in procs:
        proc.join(timeout=timeout)
        assert proc.exitcode == 0, f"rank exited {proc.exitcode}"
    for rank, status, err in sorted(results):
        assert status == "ok", f"rank {rank}: {err}"


class TestValidateCpDegree:
    def test_one_passes(self) -> None:
        validate_cp_degree(1)

    @pytest.mark.parametrize("cp", [0, -2])
    def test_rejects_non_positive(self, cp: int) -> None:
        with pytest.raises(ValueError, match="cp must be >= 1"):
            validate_cp_degree(cp)

    @pytest.mark.parametrize("cp", [True, 2.0, "2"])
    def test_rejects_non_int(self, cp: object) -> None:
        not_an_int: Any = cp
        with pytest.raises(TypeError, match="cp must be an int"):
            validate_cp_degree(not_an_int)


class TestValidateCpStyle:
    def test_ulysses_passes(self) -> None:
        assert validate_cp_style("ulysses") == "ulysses"

    @pytest.mark.parametrize("style", ["ring", "zigzag", ""])
    def test_rejects_other_styles(self, style: str) -> None:
        with pytest.raises(ValueError, match="only 'ulysses'"):
            validate_cp_style(style)


class TestValidateCpEpMix:
    def test_ep1_with_any_cp_passes(self) -> None:
        validate_cp_ep_mix(1, 4)

    def test_ep_divisible_by_cp_passes(self) -> None:
        validate_cp_ep_mix(4, 2, world_size=8)

    def test_ep_not_divisible_by_cp_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible by cp"):
            validate_cp_ep_mix(4, 3)

    def test_world_not_divisible_by_ep_raises(self) -> None:
        with pytest.raises(ValueError, match="must be divisible by ep"):
            validate_cp_ep_mix(4, 2, world_size=6)


class TestValidateCpHeads:
    def test_cp1_skips_head_checks(self) -> None:
        validate_cp_heads(7, 3, 1)

    def test_even_heads_pass(self) -> None:
        validate_cp_heads(32, 8, 4)

    def test_uneven_query_heads_raise(self) -> None:
        with pytest.raises(ValueError, match="num_attention_heads"):
            validate_cp_heads(30, 8, 4)

    def test_kv_replicate_path_passes(self) -> None:
        validate_cp_heads(32, 2, 8)

    def test_kv_neither_split_nor_replicate_raises(self) -> None:
        with pytest.raises(ValueError, match="num_key_value_heads"):
            validate_cp_heads(32, 3, 8)

    def test_missing_kv_heads_defaults_to_query_heads(self) -> None:
        validate_cp_heads(32, None, 4)
        with pytest.raises(ValueError, match="num_attention_heads"):
            validate_cp_heads(30, None, 4)


class TestValidateCpSeqLen:
    def test_cp1_skips_seq_checks(self) -> None:
        validate_cp_seq_len(7, 1)

    def test_divisible_passes(self) -> None:
        validate_cp_seq_len(8, 2)

    def test_indivisible_raises(self) -> None:
        with pytest.raises(ValueError, match="must be divisible by cp"):
            validate_cp_seq_len(7, 2)


class TestValidateCpConfig:
    def _config(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "cp": 2,
            "cp_style": "ulysses",
            "fsdp_config": FSDPConfig(),
            "world_size": 2,
            "attn_implementation": "flash_attention_2",
        }
        base.update(overrides)
        return base

    def test_cp1_returns_ulysses_without_checks(self) -> None:
        assert (
            validate_cp_config(
                cp=1,
                cp_style="ulysses",
                fsdp_config=None,
                world_size=1,
            )
            == "ulysses"
        )

    def test_valid_dense_config(self) -> None:
        assert validate_cp_config(**self._config()) == "ulysses"

    def test_cp_needs_fsdp(self) -> None:
        with pytest.raises(ValueError, match="requires fsdp_config"):
            validate_cp_config(**self._config(fsdp_config=None))

    def test_cp_needs_divisible_world(self) -> None:
        with pytest.raises(ValueError, match="divisible by cp"):
            validate_cp_config(**self._config(world_size=3))

    def test_cp_rejects_bad_style(self) -> None:
        with pytest.raises(ValueError, match="only 'ulysses'"):
            validate_cp_config(**self._config(cp_style="ring"))

    def test_cp_rejects_liger_loss(self) -> None:
        with pytest.raises(ValueError, match="use_liger_loss"):
            validate_cp_config(**self._config(use_liger_loss=True))

    def test_cp_allows_liger_loss_at_token_level(self) -> None:
        assert (
            validate_cp_config(
                **self._config(use_liger_loss=True, liger_cp_level="token")
            )
            == "ulysses"
        )

    def test_cp_allows_sequence_packing(self) -> None:
        assert (
            validate_cp_config(**self._config(use_sequence_packing=True)) == "ulysses"
        )

    def test_cp_rejects_non_flash_attention(self) -> None:
        with pytest.raises(ValueError, match="flash_attention_2"):
            validate_cp_config(**self._config(attn_implementation="sdpa"))

    def test_cp_rejects_undividable_heads(self) -> None:
        with pytest.raises(ValueError, match="num_attention_heads"):
            validate_cp_config(
                **self._config(
                    cp=4,
                    world_size=4,
                    num_attention_heads=30,
                    num_key_value_heads=8,
                )
            )

    def test_cp_rejects_bad_ep_mix(self) -> None:
        with pytest.raises(ValueError, match="not divisible by cp"):
            validate_cp_config(**self._config(world_size=12, ep=3))


class TestUnsupportedCpRejects:
    def test_cp1_does_not_raise(self) -> None:
        reject_unsupported_cp("PPO", 1)
        assert CP_SUPPORTED_ALGOS == (
            "GRPO",
            "SFT",
            "PPO",
            "REINFORCE",
            "DPO",
        )

    def test_helper_raises_with_supported_names(self) -> None:
        with pytest.raises(ValueError, match="GRPO, SFT, PPO, REINFORCE, and DPO"):
            reject_unsupported_cp("ILQL", 2)


class TestShardForCp:
    def test_cp1_returns_input(self) -> None:
        data = torch.arange(6).reshape(1, 6)

        assert torch.equal(shard_for_cp(data, 0, 1), data)

    def test_even_split(self) -> None:
        data = torch.arange(8).reshape(2, 4)

        assert torch.equal(shard_for_cp(data, 0, 2), data[:, :2])
        assert torch.equal(shard_for_cp(data, 1, 2), data[:, 2:])

    def test_concat_restores_sequence(self) -> None:
        data = torch.arange(12).reshape(3, 4)
        restored = torch.cat(
            [shard_for_cp(data, 0, 2), shard_for_cp(data, 1, 2)], dim=1
        )

        assert torch.equal(restored, data)

    def test_same_rows(self) -> None:
        data = torch.arange(8).reshape(2, 4)

        assert shard_for_cp(data, 0, 2).shape[0] == data.shape[0]
        assert shard_for_cp(data, 1, 2).shape[0] == data.shape[0]

    def test_uneven_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible by"):
            shard_for_cp(torch.zeros(1, 7), 0, 2)


class TestShardTrainingBundle:
    def _tensors(self) -> dict[str, Any]:
        batch, action_len = 2, 8
        frames = torch.arange(batch * action_len, dtype=torch.float32).reshape(
            batch, action_len
        )
        return {
            "token_ids": torch.arange(batch * (action_len + 1)).reshape(
                batch, action_len + 1
            ),
            "mask": frames > 0,
            "advantages": frames,
            "old_log_probs": frames,
            "reference_log_probs": frames,
            "turn_ids": frames.to(torch.long),
            "sampling_ratios": frames,
        }

    def test_cp1_shifts_without_cutting(self) -> None:
        tensors = self._tensors()

        bundle = shard_training_bundle(**tensors, cp_rank=0, cp_size=1)

        assert torch.equal(bundle.query_ids, tensors["token_ids"][:, :-1])
        assert torch.equal(bundle.label_ids, tensors["token_ids"][:, 1:])
        assert torch.equal(bundle.mask, tensors["mask"])

    def test_shards_share_one_cut(self) -> None:
        tensors = self._tensors()

        first = shard_training_bundle(**tensors, cp_rank=0, cp_size=2)
        second = shard_training_bundle(**tensors, cp_rank=1, cp_size=2)

        assert first.query_ids.shape[1] == 4
        assert second.query_ids.shape[1] == 4
        assert torch.equal(
            torch.cat([first.query_ids, second.query_ids], dim=1),
            tensors["token_ids"][:, :-1],
        )
        assert torch.equal(
            torch.cat([first.label_ids, second.label_ids], dim=1),
            tensors["token_ids"][:, 1:],
        )
        assert torch.equal(torch.cat([first.mask, second.mask], dim=1), tensors["mask"])

    def test_boundary_target_stays_in_shard(self) -> None:
        tensors = self._tensors()

        first = shard_training_bundle(**tensors, cp_rank=0, cp_size=2)

        assert int(first.label_ids[0, -1].item()) == int(
            tensors["token_ids"][0, 4].item()
        )

    def test_trajectory_advantages_pass_through(self) -> None:
        tensors = self._tensors()
        tensors["advantages"] = torch.ones(2, 1)

        bundle = shard_training_bundle(**tensors, cp_rank=1, cp_size=2)

        assert torch.equal(bundle.advantages, tensors["advantages"])

    def test_none_optionals_stay_none(self) -> None:
        tensors = self._tensors()
        tensors["turn_ids"] = None
        tensors["sampling_ratios"] = None

        bundle = shard_training_bundle(**tensors, cp_rank=0, cp_size=2)

        assert bundle.turn_ids is None
        assert bundle.sampling_ratios is None

    def test_undividable_action_frame_raises(self) -> None:
        tensors = self._tensors()
        tensors["token_ids"] = torch.zeros(2, 8)

        with pytest.raises(ValueError, match="must be divisible by cp"):
            shard_training_bundle(**tensors, cp_rank=0, cp_size=2)

    def test_disagreeing_tensor_raises(self) -> None:
        tensors = self._tensors()
        tensors["mask"] = torch.zeros(2, 7, dtype=torch.bool)

        with pytest.raises(ValueError, match="boundary disagreement"):
            shard_training_bundle(**tensors, cp_rank=0, cp_size=2)


class TestPadPackedRowForCp:
    def test_cp1_returns_packed_row(self) -> None:
        ids = torch.arange(6).reshape(2, 3)
        mask = torch.ones_like(ids)
        packed = pack_padded_batch(ids, mask)

        padded = pad_packed_row_for_cp(packed, pad_token_id=0, cp=1)

        assert torch.equal(padded.input_ids, packed.input_ids)
        assert padded.max_seqlen == packed.max_seqlen

    def test_already_aligned_row_is_unchanged(self) -> None:
        ids = torch.arange(4).reshape(1, 4)
        packed = pack_padded_batch(ids, torch.ones_like(ids))

        padded = pad_packed_row_for_cp(packed, pad_token_id=0, cp=2)

        assert padded.input_ids.shape[1] == 4
        assert torch.equal(padded.input_ids, packed.input_ids)

    def test_tail_is_own_segment(self) -> None:
        ids = torch.arange(5).reshape(1, 5)
        packed = pack_padded_batch(ids, torch.ones_like(ids))

        padded = pad_packed_row_for_cp(packed, pad_token_id=9, cp=2)

        assert padded.input_ids.shape[1] % 2 == 0
        assert int(padded.input_ids[0, -1].item()) == 9
        assert padded.cu_seqlens[-1] - padded.cu_seqlens[-2] == 1


def _mesh_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = build_parallel_mesh(world_size=world_size, cp=2, device_type="cpu")

        assert mesh is not None
        assert mesh.cp is not None
        group = mesh.cp_group
        assert dist.get_world_size(group) == 2

        local = torch.full((2, 4), float(rank), requires_grad=True)
        gathered = gather_for_cp(local, group)
        assert gathered.shape == (2, 8)
        assert torch.equal(gathered[:, :4], torch.zeros(2, 4))
        assert torch.equal(gathered[:, 4:], torch.ones(2, 4))

        nograd = gather_for_cp_wo_grad(local.detach(), 2, group)
        assert torch.equal(nograd, gathered.detach())

        gathered.pow(2).sum().backward()
        assert local.grad is not None
        torch.testing.assert_close(local.grad, 2 * local.detach())

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestContextParallelGather:
    def test_cp_mesh_and_gather_roundtrip(self):
        _spawn_ranks(_mesh_worker)
