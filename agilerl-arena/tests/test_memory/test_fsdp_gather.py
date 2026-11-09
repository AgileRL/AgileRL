# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""FSDP gather-count tests: golden numbers on known geometries."""

from agilerl.arena.memory import formulas, fsdp_gather
from agilerl.arena.memory.specs import GiB, ModelArch
from tests.test_memory.test_formulas import (
    GEMMA4_E4B,
    GRANITE_MOE,
    MOE_TINY,
    NEMOTRON_H,
    NEMOTRON_H_MOE,
    QWEN_05B,
)

GATHER_KWARGS = {
    "n_gpus": 4,
    "reshard_after_forward": True,
}


class TestFSDPGatheredParams:
    def test_reshard_keeps_a_prefetch_window_and_not_the_embedding(self):
        counts = formulas.param_counts(QWEN_05B)
        block = fsdp_gather.largest_block_params(counts, QWEN_05B)

        gathered = fsdp_gather.fsdp_gathered_params(
            counts,
            QWEN_05B,
            n_gpus=4,
            reshard_after_forward=True,
        )

        assert gathered == 2 * block
        assert gathered < counts.total - formulas.token_embedding_params(QWEN_05B)

    def test_no_reshard_gathers_every_sharded_parameter(self):
        counts = formulas.param_counts(QWEN_05B)
        sharded = counts.total - formulas.token_embedding_params(QWEN_05B)

        gathered = fsdp_gather.fsdp_gathered_params(
            counts,
            QWEN_05B,
            n_gpus=4,
            reshard_after_forward=False,
        )

        assert gathered == sharded

    def test_untied_head_stays_gathered(self):
        untied = QWEN_05B.model_copy(update={"tied_embeddings": False})
        counts = formulas.param_counts(untied)
        block = fsdp_gather.largest_block_params(counts, untied)

        gathered = fsdp_gather.fsdp_gathered_params(
            counts,
            untied,
            n_gpus=4,
            reshard_after_forward=True,
        )

        assert gathered == 2 * block + counts.lm_head

    def test_one_gpu_gathers_the_same_copy_as_a_sharded_group(self):
        counts = formulas.param_counts(QWEN_05B)
        block = fsdp_gather.largest_block_params(counts, QWEN_05B)
        sharded = counts.total - formulas.token_embedding_params(QWEN_05B)

        resharded = fsdp_gather.fsdp_gathered_params(
            counts, QWEN_05B, n_gpus=1, reshard_after_forward=True
        )
        held = fsdp_gather.fsdp_gathered_params(
            counts, QWEN_05B, n_gpus=1, reshard_after_forward=False
        )

        assert resharded == 2 * block
        assert held == sharded

    def test_block_exclusive_hybrid_window_uses_the_widest_layer_kind(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)
        counts = formulas.param_counts(arch)
        block = fsdp_gather.largest_block_params(counts, arch)
        mlp = formulas.mlp_params_per_layer(arch)

        gathered = fsdp_gather.fsdp_gathered_params(
            counts,
            arch,
            n_gpus=4,
            reshard_after_forward=True,
        )

        assert arch.block_exclusive_layers
        assert arch.mlp_layers
        assert not arch.tied_embeddings
        assert block == mlp
        assert block > formulas.mamba_params_per_layer(arch)
        assert block > formulas.attention_params_per_layer(arch)
        # Every MLP layer neighbours a Mamba layer, so a two-unit window
        # holds one of each.
        assert (
            gathered == block + formulas.mamba_params_per_layer(arch) + counts.lm_head
        )

    def test_moe_window_holds_one_expert_block_and_its_neighbour(self):
        # Arrange
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        counts = formulas.param_counts(arch)
        experts, router = formulas.moe_params_per_layer(arch)
        mamba = formulas.mamba_params_per_layer(arch)

        # Act
        gathered = fsdp_gather.fsdp_gathered_params(
            counts,
            arch,
            n_gpus=4,
            reshard_after_forward=True,
        )

        # Assert: MoE blocks never sit side by side, so the second unit is a
        # Mamba block, not a second expert stack.
        assert arch.layer_kinds[:3] == ("mamba", "moe", "mamba")
        assert gathered == experts + router + mamba + counts.lm_head


class TestFSDPReshardedSideParams:
    def test_tied_dense_model_has_no_side_unit(self):
        counts = formulas.param_counts(QWEN_05B)

        assert fsdp_gather.fsdp_resharded_side_params(counts, QWEN_05B) == 0

    def test_untied_head_is_a_side_unit(self):
        untied = QWEN_05B.model_copy(update={"tied_embeddings": False})
        counts = formulas.param_counts(untied)

        assert fsdp_gather.fsdp_resharded_side_params(counts, untied) == counts.lm_head

    def test_per_layer_embeddings_are_a_side_unit_and_the_root_is_not(self):
        # Arrange
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)
        counts = formulas.param_counts(arch, GEMMA4_E4B.num_params)
        per_layer = counts.embedding - formulas.token_embedding_params(arch)

        # Act
        side = fsdp_gather.fsdp_resharded_side_params(counts, arch)

        # Assert: the tower root and unattributed parameters stay gathered.
        assert arch.tied_embeddings
        assert counts.unattributed > 0
        assert side == per_layer
        assert 5.2 * GiB < side * 2 < 5.3 * GiB


class TestFSDPGatherPoolParams:
    def test_dense_model_pools_a_prefetch_window(self):
        counts = formulas.param_counts(QWEN_05B)
        block = fsdp_gather.largest_block_params(counts, QWEN_05B)

        pool = fsdp_gather.fsdp_gather_pool_params(counts, QWEN_05B, 1, 4)

        assert pool == 2 * block

    def test_tensor_parallel_plan_leaves_only_norms_to_gather(self):
        arch = QWEN_05B.model_copy(update={"tensor_parallel_plan": True})
        counts = formulas.param_counts(arch)
        norms_per_layer = counts.norms // arch.n_layers

        on_mesh = fsdp_gather.fsdp_gather_pool_params(counts, arch, 1, 4, 2)
        off_mesh = fsdp_gather.fsdp_gather_pool_params(counts, arch, 1, 4)

        assert not arch.layer_kinds
        assert on_mesh == 2 * norms_per_layer
        assert off_mesh == 2 * fsdp_gather.largest_block_params(counts, arch)

    def test_per_layer_embeddings_pool_beside_the_first_unit(self):
        # Arrange
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)
        counts = formulas.param_counts(arch, GEMMA4_E4B.num_params)
        per_layer = counts.embedding - formulas.token_embedding_params(arch)
        block = fsdp_gather.largest_block_params(counts, arch)

        # Act
        pool = fsdp_gather.fsdp_gather_pool_params(counts, arch, 1, 4)

        # Assert: 4-GPU Gemma 4 E4B held 5.43 GiB on its all-gather stream.
        assert pool == per_layer + block
        assert 5.3 * GiB < pool * 2 < 5.5 * GiB

    def test_untied_head_pools_beside_the_last_unit(self):
        # Arrange
        arch = ModelArch.from_hf_config(NEMOTRON_H)
        counts = formulas.param_counts(arch)

        # Act
        pool = fsdp_gather.fsdp_gather_pool_params(counts, arch, 1, 4)

        # Assert
        assert arch.layer_kinds[-1] == "mlp"
        assert pool == counts.lm_head + formulas.mlp_params_per_layer(arch)

    def test_expert_parallel_gathers_only_a_ranks_expert_share(self):
        # Arrange
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        counts = formulas.param_counts(arch)
        experts, router = formulas.moe_params_per_layer(arch)
        inter = arch.expert_intermediate_size or arch.intermediate_size
        routed = arch.n_experts * 2 * arch.expert_width * inter
        unit_without_routed = experts - routed + router

        # Act
        dense = fsdp_gather.fsdp_gather_pool_params(counts, arch, 1, 8)
        ep_only = fsdp_gather.fsdp_gather_pool_params(counts, arch, 8, 8)
        ep_with_dp = fsdp_gather.fsdp_gather_pool_params(counts, arch, 8, 16)

        # Assert: the head pools beside the last (MoE) unit. EP over the whole
        # shard group leaves no rank to gather experts from.
        assert not arch.gated_mlp
        assert arch.layer_kinds[-1] == "moe"
        assert dense == counts.lm_head + experts + router == 1_649_786_880
        assert ep_only == counts.lm_head + unit_without_routed == 372_621_312
        assert ep_with_dp == ep_only + routed // 8 == 532_267_008


class TestLargestBlockParams:
    def test_expert_parallel_leaves_the_routed_experts_out_of_the_block(self):
        # Arrange
        counts = formulas.param_counts(MOE_TINY)
        routed = formulas.routed_expert_params_per_layer(MOE_TINY)

        # Act
        local = fsdp_gather.largest_block_params(counts, MOE_TINY)
        parallel = fsdp_gather.largest_block_params(counts, MOE_TINY, expert_parallel=2)

        # Assert
        assert not MOE_TINY.block_exclusive_layers
        assert routed > 0
        assert parallel == local - routed


class TestFSDPGatheredParamsExpertParallel:
    def test_ep_filling_the_shard_group_gathers_no_routed_experts(self):
        # Arrange
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        counts = formulas.param_counts(arch)
        experts, router = formulas.moe_params_per_layer(arch)
        routed = formulas.routed_expert_params_per_layer(arch)
        mamba = formulas.mamba_params_per_layer(arch)
        attention = formulas.attention_params_per_layer(arch)

        # Act
        gathered = fsdp_gather.fsdp_gathered_params(
            counts, arch, expert_parallel=4, **GATHER_KWARGS
        )

        # Assert: the MoE block unit keeps its router and shared experts only,
        # so a Mamba + attention pair can be the widest window.
        assert 0 < routed < experts
        block_moe = experts - routed + router
        assert gathered == max(block_moe, attention) + mamba + counts.lm_head

    def test_leftover_ranks_gather_one_layer_of_local_experts(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        counts = formulas.param_counts(arch)
        routed = formulas.routed_expert_params_per_layer(arch)

        filled = fsdp_gather.fsdp_gathered_params(
            counts, arch, expert_parallel=4, **GATHER_KWARGS
        )
        leftover = fsdp_gather.fsdp_gathered_params(
            counts, arch, expert_parallel=2, **GATHER_KWARGS
        )

        assert leftover == filled + routed // 2

    def test_no_reshard_holds_every_local_expert_layer(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        counts = formulas.param_counts(arch)
        routed = formulas.routed_expert_params_per_layer(arch)
        sharded = counts.total - formulas.token_embedding_params(arch)
        kwargs = {**GATHER_KWARGS, "reshard_after_forward": False}

        gathered = fsdp_gather.fsdp_gathered_params(
            counts, arch, expert_parallel=2, **kwargs
        )

        assert gathered == sharded - arch.moe_layers * routed // 2

    def test_tensor_parallel_uniform_moe_stack_keeps_experts_gathered(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        counts = formulas.param_counts(arch)
        block_moe = fsdp_gather.block_moe_params_per_layer(arch)

        gathered = fsdp_gather.fsdp_gathered_params(
            counts, arch, tensor_parallel=2, **GATHER_KWARGS
        )

        assert not arch.layer_kinds
        assert gathered >= 2 * block_moe
