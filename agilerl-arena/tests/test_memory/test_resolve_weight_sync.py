# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""One async weight sync is priced from its bytes, rounds and link."""

from __future__ import annotations

import pytest

from agilerl.arena.memory.lora_formulas import (
    NcclReshardLayout,
    lora_adapter_tensor_count,
    lora_sync_buffer_bytes,
    merged_lora_sync_tensor_bytes,
    nccl_reshard_sync_linears,
)
from agilerl.arena.memory.manifest import lookup_gpu
from agilerl.arena.memory.resolve import weight_sync
from agilerl.arena.memory.resolve.weight_sync import (
    ADAPTER_FSDP_GATHER_SECONDS_PER_TENSOR,
    ADAPTER_SWAP_BYTES_PER_SECOND,
    ADAPTER_SYNC_SECONDS,
    ENGINE_LOAD_ALL_REDUCE_SHARE,
    FSDP_GATHER_SECONDS_PER_TENSOR,
    LINK_EGRESS_BYTES_PER_SECOND,
    MERGED_SYNC_BARRIER_SECONDS,
    MERGED_SYNC_ENGINE_ROUND_SECONDS,
    MERGED_SYNC_PAUSE_SECONDS,
    MERGED_SYNC_TRAINER_ROUND_SECONDS,
    SM80_NCCL_RESHARD_COSTS,
    SM90_NCCL_RESHARD_COSTS,
    adapter_sync_seconds,
    merged_sync_rounds,
    merged_sync_seconds,
    nccl_reshard_bulk_seconds,
    nccl_reshard_costs,
    nccl_reshard_sync_seconds,
)
from agilerl.arena.memory.specs import DeviceSpec, GiB, MiB, ModelArch, ModelSpec
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


def _model(hub_id: str) -> ModelSpec:
    return ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config),
    )


def _device(name: str) -> DeviceSpec:
    gpu = lookup_gpu(name)
    assert gpu is not None
    return gpu.device_spec()


def _merged_sizes(model: ModelSpec) -> list[int]:
    return merged_lora_sync_tensor_bytes(model.arch, "all-linear", 0, 2)


# Medians of weight_sync_s over steps 5-30 on 8x H100 SXM, CISPO on
# GuessTheNumber (2 turns), all-linear r=16, 4 trainers and 2 engines at TP 2.
# The planner trained Qwen3.5-9B and Qwen3.6-35B-A3B FSDP, the rest DDP.
H100_SYNCS = [
    # (model, FSDP, adapter, merged NIXL, nccl_reshard)
    ("Qwen/Qwen3.5-2B", False, 0.109, 0.169, 0.111),
    ("Qwen/Qwen3.5-4B", False, 0.141, 0.320, 0.179),
    ("Qwen/Qwen3.5-9B", True, 0.294, 0.729, 0.353),
    ("Qwen/Qwen3.8-27B", False, 0.328, 2.607, 0.675),
    ("Qwen/Qwen3.6-35B-A3B", True, 0.512, 0.509, None),
]
H100_IDS = ["2b", "4b", "9b", "27b", "35b-a3b"]


class TestMergedSyncRounds:
    def test_a_tensor_that_overfills_the_round_starts_the_next(self) -> None:
        # Arrange: four 150 MiB tensors in 256 MiB rounds.
        sizes = [150 * MiB] * 4

        # Act
        rounds = merged_sync_rounds(sizes)

        # Assert
        assert rounds == 4

    def test_a_round_holds_tensors_that_fill_it_exactly(self) -> None:
        # Arrange
        sizes = [128 * MiB] * 4

        # Act
        rounds = merged_sync_rounds(sizes)

        # Assert
        assert rounds == 2

    @pytest.mark.parametrize(
        ("hub_id", "expected"),
        [
            ("Qwen/Qwen3.5-2B", 11),
            ("Qwen/Qwen3.5-4B", 30),
            ("Qwen/Qwen3.5-9B", 64),
            ("Qwen/Qwen3.8-27B", 256),
            ("Qwen/Qwen3.6-35B-A3B", 11),
        ],
    )
    def test_matches_the_rounds_the_h100_runs_pushed(
        self, hub_id: str, expected: int
    ) -> None:
        # Act
        rounds = merged_sync_rounds(_merged_sizes(_model(hub_id)))

        # Assert
        assert rounds == expected


class TestMergedSyncSeconds:
    @pytest.mark.parametrize(
        ("hub_id", "fsdp", "measured"),
        [(hub_id, fsdp, merged) for hub_id, fsdp, _, merged, _ in H100_SYNCS],
        ids=H100_IDS,
    )
    def test_matches_the_h100_merged_nixl_syncs(
        self, hub_id: str, fsdp: bool, measured: float
    ) -> None:
        # Act
        seconds = merged_sync_seconds(
            _model(hub_id), "all-linear", 0, 2, 2, fsdp, _device("H100-80"), "cuda_ipc"
        )

        # Assert
        assert seconds == pytest.approx(measured, rel=0.12)

    def test_matches_the_a100_merged_nixl_sync(self) -> None:
        # Arrange: 4x A100-SXM4-80GB, 2 DDP trainers, 1 engine at TP 2.
        model = _model("Qwen/Qwen3.5-4B")

        # Act
        seconds = merged_sync_seconds(
            model, "all-linear", 0, 1, 2, False, _device("A100-80"), "cuda_ipc"
        )

        # Assert
        assert seconds == pytest.approx(0.377, rel=0.07)

    def test_engine_bound_rounds_cost_the_engine_side_after_the_first_merge(
        self,
    ) -> None:
        # Arrange: Qwen3-4B all-linear moves 7.27 GB to four TP 1 engines.
        model = _model("Qwen/Qwen3-4B")
        sizes = _merged_sizes(model)
        rounds = merged_sync_rounds(sizes)

        # Act
        seconds = merged_sync_seconds(
            model, "all-linear", 0, 4, 1, False, _device("A100-80"), "cuda_ipc"
        )

        # Assert
        assert sum(sizes) == pytest.approx(7.27e9, rel=0.01)
        assert seconds == pytest.approx(
            MERGED_SYNC_PAUSE_SECONDS
            + MERGED_SYNC_TRAINER_ROUND_SECONDS
            + rounds * MERGED_SYNC_ENGINE_ROUND_SECONDS
            + 4 * sum(sizes) / LINK_EGRESS_BYTES_PER_SECOND["cuda_ipc"]
            + MERGED_SYNC_BARRIER_SECONDS
        )

    def test_trainer_bound_rounds_cost_the_merge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: a 50 ms merge outlasts every engine round over NVLink.
        model = _model("Qwen/Qwen3-4B")
        sizes = _merged_sizes(model)
        rounds = merged_sync_rounds(sizes)
        monkeypatch.setattr(weight_sync, "MERGED_SYNC_TRAINER_ROUND_SECONDS", 0.05)

        # Act
        seconds = merged_sync_seconds(
            model, "all-linear", 0, 4, 1, False, _device("A100-80"), "cuda_ipc"
        )

        # Assert: every merge, then the last round's receive.
        assert seconds == pytest.approx(
            MERGED_SYNC_PAUSE_SECONDS
            + rounds * 0.05
            + MERGED_SYNC_ENGINE_ROUND_SECONDS
            + 4 * sum(sizes) / LINK_EGRESS_BYTES_PER_SECOND["cuda_ipc"] / rounds
            + MERGED_SYNC_BARRIER_SECONDS
        )

    def test_fsdp_trainers_gather_every_tensor_before_its_merge(self) -> None:
        # Arrange: the gathers make every round trainer-bound.
        model = _model("Qwen/Qwen3-4B")
        sizes = _merged_sizes(model)
        rounds = merged_sync_rounds(sizes)
        receive = 4 * sum(sizes) / LINK_EGRESS_BYTES_PER_SECOND["cuda_ipc"]

        # Act
        seconds = merged_sync_seconds(
            model, "all-linear", 0, 4, 1, True, _device("A100-80"), "cuda_ipc"
        )

        # Assert
        assert seconds == pytest.approx(
            MERGED_SYNC_PAUSE_SECONDS
            + rounds * MERGED_SYNC_TRAINER_ROUND_SECONDS
            + len(sizes) * FSDP_GATHER_SECONDS_PER_TENSOR
            + MERGED_SYNC_ENGINE_ROUND_SECONDS
            + receive / rounds
            + MERGED_SYNC_BARRIER_SECONDS
        )

    def test_each_engine_reads_once_and_loads_at_a_share_of_all_reduce(
        self,
    ) -> None:
        # Arrange
        model = _model("Qwen/Qwen3-4B")
        device = _device("H100-80")
        sizes = _merged_sizes(model)
        fixed = (
            MERGED_SYNC_PAUSE_SECONDS
            + MERGED_SYNC_TRAINER_ROUND_SECONDS
            + merged_sync_rounds(sizes) * MERGED_SYNC_ENGINE_ROUND_SECONDS
            + MERGED_SYNC_BARRIER_SECONDS
        )

        # Act
        tp2 = merged_sync_seconds(model, "all-linear", 0, 4, 2, False, device, "tcp")
        tp8 = merged_sync_seconds(model, "all-linear", 0, 4, 8, False, device, "tcp")

        # Assert
        assert tp8 == pytest.approx(tp2)
        assert tp2 - fixed == pytest.approx(
            4 * sum(sizes) / LINK_EGRESS_BYTES_PER_SECOND["tcp"]
            + sum(sizes)
            / (ENGINE_LOAD_ALL_REDUCE_SHARE * device.all_reduce_bytes_per_second)
        )

    def test_a100_loads_slower_than_h100(self) -> None:
        # Arrange
        model = _model("Qwen/Qwen3.5-9B")

        # Act
        a100 = merged_sync_seconds(
            model, "all-linear", 0, 2, 2, False, _device("A100-80"), "cuda_ipc"
        )
        h100 = merged_sync_seconds(
            model, "all-linear", 0, 2, 2, False, _device("H100-80"), "cuda_ipc"
        )

        # Assert
        assert a100 > h100

    def test_tensor_parallel_engine_without_all_reduce_figures_raises(self) -> None:
        # Arrange
        device = DeviceSpec(total_bytes=80 * GiB, name="Exotic GPU")

        # Act / Assert
        with pytest.raises(ValueError, match="Exotic GPU has no all-reduce bandwidth"):
            merged_sync_seconds(
                _model("Qwen/Qwen3-4B"), "all-linear", 0, 1, 2, False, device, "tcp"
            )

    def test_tcp_takes_over_a_minute_for_four_engines(self) -> None:
        # Act
        seconds = merged_sync_seconds(
            _model("Qwen/Qwen3-4B"),
            "all-linear",
            0,
            4,
            1,
            False,
            _device("A100-80"),
            "tcp",
        )

        # Assert: 4 x 7.27 GB at 0.4 GB/s.
        assert seconds > 70


class TestAdapterSyncSeconds:
    @pytest.mark.parametrize(
        ("hub_id", "fsdp", "measured", "tolerance"),
        [
            (hub_id, fsdp, adapter, 0.35 if fsdp else 0.03)
            for hub_id, fsdp, adapter, _, _ in H100_SYNCS
        ],
        ids=H100_IDS,
    )
    def test_matches_the_h100_adapter_syncs(
        self, hub_id: str, fsdp: bool, measured: float, tolerance: float
    ) -> None:
        # Act: one gather rate prices Qwen3.5-9B 22% high and Qwen3.6-35B-A3B
        # 30% low.
        seconds = adapter_sync_seconds(
            _model(hub_id), 16, "all-linear", 0, 2, 2, fsdp, "cuda_ipc"
        )

        # Assert
        assert seconds == pytest.approx(measured, rel=tolerance)

    def test_nvlink_costs_the_engine_rpcs_and_the_swap(self) -> None:
        # Arrange
        model = _model("Qwen/Qwen3-4B")
        adapter = lora_sync_buffer_bytes(model.arch, 16, "all-linear", 0, 2)

        # Act
        seconds = adapter_sync_seconds(
            model, 16, "all-linear", 0, 4, 1, False, "cuda_ipc"
        )

        # Assert: 4 x 66 MB is under 1 ms at 275 GB/s.
        assert seconds == pytest.approx(
            ADAPTER_SYNC_SECONDS + adapter / ADAPTER_SWAP_BYTES_PER_SECOND, abs=1e-3
        )

    def test_fsdp_trainers_gather_every_lora_factor(self) -> None:
        # Arrange
        model = _model("Qwen/Qwen3-4B")
        factors = lora_adapter_tensor_count(model.arch, "all-linear", 0)

        # Act
        ddp = adapter_sync_seconds(model, 16, "all-linear", 0, 2, 2, False, "cuda_ipc")
        fsdp = adapter_sync_seconds(model, 16, "all-linear", 0, 2, 2, True, "cuda_ipc")

        # Assert
        assert factors == 2 * 7 * 36
        assert fsdp - ddp == pytest.approx(
            factors * ADAPTER_FSDP_GATHER_SECONDS_PER_TENSOR
        )

    def test_every_engine_rank_reads_the_adapter(self) -> None:
        # Arrange
        model = _model("Qwen/Qwen3-4B")
        adapter = lora_sync_buffer_bytes(model.arch, 16, "all-linear", 0, 2)

        # Act
        seconds = adapter_sync_seconds(model, 16, "all-linear", 0, 2, 2, False, "tcp")

        # Assert
        assert seconds == pytest.approx(
            ADAPTER_SYNC_SECONDS
            + adapter / ADAPTER_SWAP_BYTES_PER_SECOND
            + 4 * adapter / LINK_EGRESS_BYTES_PER_SECOND["tcp"]
        )

    def test_scales_with_tensor_parallel_width(self) -> None:
        # Arrange
        model = _model("Qwen/Qwen3-4B")
        adapter = lora_sync_buffer_bytes(model.arch, 16, "all-linear", 0, 2)

        # Act
        one_tp4 = adapter_sync_seconds(model, 16, "all-linear", 0, 1, 4, False, "tcp")
        one_tp8 = adapter_sync_seconds(model, 16, "all-linear", 0, 1, 8, False, "tcp")

        # Assert
        assert one_tp8 - one_tp4 == pytest.approx(
            4 * adapter / LINK_EGRESS_BYTES_PER_SECOND["tcp"]
        )


class TestNcclReshardCosts:
    @pytest.mark.parametrize("gpu", ["A100-80", "A100"])
    def test_a100_takes_the_sm80_rates(self, gpu: str) -> None:
        # Act
        costs = nccl_reshard_costs(_device(gpu))

        # Assert
        assert costs == SM80_NCCL_RESHARD_COSTS

    @pytest.mark.parametrize("gpu", ["H100-80", "H200", "B200"])
    def test_hopper_and_newer_take_the_sm90_rates(self, gpu: str) -> None:
        # Act
        costs = nccl_reshard_costs(_device(gpu))

        # Assert
        assert costs == SM90_NCCL_RESHARD_COSTS

    def test_nvlink_below_hopper_scales_the_a100_rates_by_all_reduce_bandwidth(
        self,
    ) -> None:
        # Arrange: capability (8, 9) has no measured row. All-reduce at
        # 700 GB/s against A100's 230 GB/s sets the slowdown.
        slowdown = 230 / 700
        device = DeviceSpec(
            total_bytes=80 * GiB,
            compute_capability=(8, 9),
            nvlink=True,
            all_reduce_bytes_per_second=700e9,
        )

        # Act
        costs = nccl_reshard_costs(device)

        # Assert
        assert costs is not None
        assert costs.bulk_engine_seconds_per_gib == pytest.approx(0.00315 * slowdown)
        assert costs.bulk_rank_seconds_per_gib == pytest.approx(0.00183 * slowdown)
        assert costs.misc_bytes_per_second == pytest.approx(2.385e10 / slowdown)
        assert costs.control_seconds == SM80_NCCL_RESHARD_COSTS.control_seconds

    @pytest.mark.parametrize("gpu", ["NVIDIA L4", "A10G"])
    def test_pcie_gpu_has_no_rates(self, gpu: str) -> None:
        # Act
        costs = nccl_reshard_costs(_device(gpu))

        # Assert
        assert costs is None

    def test_nvlink_gpu_without_all_reduce_figures_has_no_rates(self) -> None:
        # Arrange
        device = DeviceSpec(total_bytes=80 * GiB, nvlink=True)

        # Act
        costs = nccl_reshard_costs(device)

        # Assert
        assert costs is None


class TestNcclReshardBulkSeconds:
    def test_one_gib_matches_the_a100_microbenchmark(self) -> None:
        # Arrange: the 0.984 GiB block, 1 engine at TP 2 is 4.0 ms; 2 engines
        # at TP 1 is 9.8 ms.
        block = int(0.984 * GiB)
        costs = SM80_NCCL_RESHARD_COSTS

        # Act
        one_tp2 = nccl_reshard_bulk_seconds(block, 1, 2, costs)
        two_tp1 = nccl_reshard_bulk_seconds(block, 2, 1, costs)

        # Assert
        assert one_tp2 == pytest.approx(0.004, rel=1e-3)
        assert two_tp1 == pytest.approx(0.0098, rel=1e-3)

    @pytest.mark.parametrize(
        ("engines", "engine_width", "measured"),
        [(1, 2, 0.00274), (2, 2, 0.00450), (1, 4, 0.00188)],
        ids=["1xtp2", "2xtp2", "1xtp4"],
    )
    def test_one_gib_matches_the_h100_microbenchmark(
        self, engines: int, engine_width: int, measured: float
    ) -> None:
        # Arrange: the 0.984 GiB FFN block over NVSwitch.
        block = int(0.984 * GiB)

        # Act
        seconds = nccl_reshard_bulk_seconds(
            block, engines, engine_width, SM90_NCCL_RESHARD_COSTS
        )

        # Assert
        assert seconds == pytest.approx(measured, rel=0.11)


class TestNcclReshardSyncSeconds:
    def test_full_refit_matches_the_a100_runs(self) -> None:
        # Arrange: 4x A100-SXM4-80GB, 2 DDP trainers, 1 engine TP 2. Medians
        # of weight_sync_s, excluding the first sync.
        costs = SM80_NCCL_RESHARD_COSTS

        # Act
        small = nccl_reshard_sync_seconds(
            _model("Qwen/Qwen3.5-0.8B"), "all-linear", 1, 2, 1, False, costs
        )
        four = nccl_reshard_sync_seconds(
            _model("Qwen/Qwen3.5-4B"), "all-linear", 1, 2, 1, False, costs
        )
        nine = nccl_reshard_sync_seconds(
            _model("Qwen/Qwen3.5-9B"), "all-linear", 1, 2, 1, False, costs
        )

        # Assert
        assert small == pytest.approx(0.1485, abs=0.001)
        assert four == pytest.approx(0.221, rel=0.07)
        assert nine == pytest.approx(0.306, abs=0.001)

    @pytest.mark.parametrize(
        ("hub_id", "fsdp", "measured", "tolerance"),
        [
            (hub_id, fsdp, nccl, 0.14 if fsdp else 0.02)
            for hub_id, fsdp, _, _, nccl in H100_SYNCS
            if nccl is not None
        ],
        ids=H100_IDS[:4],
    )
    def test_full_refit_matches_the_h100_runs(
        self, hub_id: str, fsdp: bool, measured: float, tolerance: float
    ) -> None:
        # Act
        seconds = nccl_reshard_sync_seconds(
            _model(hub_id),
            "all-linear",
            2,
            2,
            4 if fsdp else 1,
            fsdp,
            SM90_NCCL_RESHARD_COSTS,
        )

        # Assert
        assert seconds == pytest.approx(measured, rel=tolerance)

    def test_fsdp_trainers_gather_every_misc_tensor(self) -> None:
        # Arrange
        model = _model("Qwen/Qwen3.5-9B")
        _, misc = nccl_reshard_sync_linears(
            model.arch, "all-linear", NcclReshardLayout(4, 2)
        )
        costs = SM90_NCCL_RESHARD_COSTS

        # Act
        ddp = nccl_reshard_sync_seconds(model, "all-linear", 2, 2, 4, False, costs)
        fsdp = nccl_reshard_sync_seconds(model, "all-linear", 2, 2, 4, True, costs)

        # Assert
        assert fsdp - ddp == pytest.approx(
            FSDP_GATHER_SECONDS_PER_TENSOR * sum(linear.count for linear in misc)
        )
