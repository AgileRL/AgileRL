# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The fused Liger loss runs wherever its kernel expresses the objective."""

from typing import Any

import pytest

from agilerl.arena.memory.resolve.liger import (
    liger_head_grad,
    liger_kernel_arch,
    liger_loss_blocker,
    liger_loss_layout,
    liger_whole_sequence_loss,
)
from agilerl.arena.memory.specs import ModelArch

TOKEN_LEVEL_RUN: dict[str, Any] = {
    "importance_sampling_level": "token",
    "records_sampling_supports": False,
    "kl_advantage_shaping": False,
    "final_logit_softcap": None,
    "reweights_tokens": False,
    "advantage_granularity": None,
}


def blocker(algorithm: str, **changes: Any) -> str | None:
    """Liger blocker for a token-level run with ``changes`` applied."""
    return liger_loss_blocker(algorithm, **{**TOKEN_LEVEL_RUN, **changes})


class TestLigerLossBlocker:
    @pytest.mark.parametrize(
        "algorithm", ["grpo", "cispo", "ppo", "reinforce", "dpo", "sft"]
    )
    def test_token_level_is_served(self, algorithm: str) -> None:
        assert blocker(algorithm) is None

    def test_unset_level_is_token_level(self) -> None:
        assert blocker("reinforce", importance_sampling_level=None) is None

    def test_token_level_fuses_the_sampling_correction(self) -> None:
        assert blocker("ppo", reweights_tokens=True) is None
        assert blocker("grpo", reweights_tokens=True) is None

    def test_auto_ppo_level_defers_to_the_batch(self) -> None:
        # Act
        config_reason = blocker(
            "ppo", importance_sampling_level="auto", reweights_tokens=True
        )
        batch_reason = blocker(
            "ppo", importance_sampling_level="turn", reweights_tokens=True
        )

        # Assert
        assert config_reason is None
        assert batch_reason == (
            "the ppo Liger kernel weights tokens by vLLM sampling log-probs only "
            "at token-level importance sampling, not 'turn'"
        )

    @pytest.mark.parametrize("algorithm", ["ppo", "reinforce"])
    @pytest.mark.parametrize("level", ["turn", "trajectory"])
    def test_ppo_and_reinforce_fuse_above_token_level(
        self, algorithm: str, level: str
    ) -> None:
        assert blocker(algorithm, importance_sampling_level=level) is None

    @pytest.mark.parametrize("algorithm", ["ppo", "reinforce"])
    @pytest.mark.parametrize("level", ["turn", "trajectory"])
    def test_ppo_and_reinforce_reweighting_above_token_level_is_blocked(
        self, algorithm: str, level: str
    ) -> None:
        # Act
        reason = blocker(
            algorithm, importance_sampling_level=level, reweights_tokens=True
        )

        # Assert
        assert reason == (
            f"the {algorithm} Liger kernel weights tokens by vLLM sampling "
            f"log-probs only at token-level importance sampling, not '{level}'"
        )

    def test_grpo_turn_level_is_blocked(self) -> None:
        # Act
        reason = blocker("grpo", importance_sampling_level="turn")

        # Assert
        assert reason == "the grpo Liger kernel has no 'turn' importance-sampling level"

    @pytest.mark.parametrize("level", ["turn", "trajectory"])
    def test_cispo_above_token_level_is_blocked(self, level: str) -> None:
        # Act
        reason = blocker("cispo", importance_sampling_level=level)

        # Assert
        assert reason == (
            f"the cispo Liger kernel has no '{level}' importance-sampling level"
        )

    @pytest.mark.parametrize(
        ("algorithm", "level"), [("grpo", "trajectory"), ("gspo", None)]
    )
    def test_grpo_trajectory_level_is_served(
        self, algorithm: str, level: str | None
    ) -> None:
        assert (
            blocker(
                algorithm,
                importance_sampling_level=level,
                advantage_granularity="trajectory",
            )
            is None
        )

    @pytest.mark.parametrize("algorithm", ["grpo", "gspo"])
    def test_grpo_trajectory_reweighting_is_blocked(self, algorithm: str) -> None:
        # Act
        reason = blocker(
            algorithm, importance_sampling_level="trajectory", reweights_tokens=True
        )

        # Assert
        assert reason == (
            f"the {algorithm} Liger kernel weights tokens by vLLM sampling "
            "log-probs only at token-level importance sampling, not 'trajectory'"
        )

    def test_gspo_per_turn_advantages_are_blocked(self) -> None:
        # Act
        reason = blocker("gspo", advantage_granularity="turn")

        # Assert
        assert reason == (
            "the gspo Liger kernel takes one advantage per sequence at "
            "trajectory level, not per-turn advantages"
        )

    def test_grpo_auto_advantage_granularity_defers_to_the_batch(self) -> None:
        assert blocker("gspo", advantage_granularity="auto") is None

    @pytest.mark.parametrize("level", ["token", "trajectory"])
    def test_sampling_supports_block_grpo_but_not_ppo(self, level: str) -> None:
        # Act
        grpo_reason = blocker(
            "grpo", importance_sampling_level=level, records_sampling_supports=True
        )
        ppo_reason = blocker(
            "ppo", importance_sampling_level=level, records_sampling_supports=True
        )

        # Assert
        assert grpo_reason == (
            "the GRPO Liger kernel normalizes over the full vocabulary, not "
            "vLLM's sampling support (top_k > 0 and temperature > 0)"
        )
        assert ppo_reason is None

    def test_kl_advantage_shaping_blocks_grpo(self) -> None:
        # Act
        reason = blocker("grpo", kl_advantage_shaping=True)

        # Assert
        assert reason == "KL advantage shaping has no fused Liger kernel"

    def test_softcap_blocks_kernels_that_do_not_apply_it(self) -> None:
        # Act
        grpo_reason = blocker("grpo", final_logit_softcap=30.0)
        dpo_reason = blocker("dpo", final_logit_softcap=30.0)
        ppo_reason = blocker(
            "ppo", importance_sampling_level="turn", final_logit_softcap=30.0
        )
        sft_reason = blocker("sft", final_logit_softcap=30.0)

        # Assert
        assert grpo_reason == (
            "the grpo Liger kernel does not apply the model's final-logit "
            "softcap (30.0)"
        )
        assert dpo_reason == (
            "the dpo Liger kernel does not apply the model's final-logit softcap (30.0)"
        )
        assert ppo_reason is None
        assert sft_reason is None


class TestLigerWholeSequenceLoss:
    @pytest.mark.parametrize("algorithm", ["ppo", "reinforce"])
    @pytest.mark.parametrize("level", ["turn", "trajectory"])
    def test_ppo_and_reinforce_chunk_whole_rows_above_token_level(
        self, algorithm: str, level: str
    ) -> None:
        assert liger_whole_sequence_loss(algorithm, level) is True

    @pytest.mark.parametrize(
        ("algorithm", "level"), [("grpo", "trajectory"), ("gspo", None)]
    )
    def test_group_relative_kernel_chunks_one_sequence_at_trajectory_level(
        self, algorithm: str, level: str | None
    ) -> None:
        assert liger_whole_sequence_loss(algorithm, level) is True

    def test_dpo_chunks_chosen_and_rejected_pairs(self) -> None:
        assert liger_whole_sequence_loss("dpo", None) is True

    @pytest.mark.parametrize(
        ("algorithm", "level"),
        [
            ("ppo", "token"),
            ("reinforce", None),
            ("grpo", "token"),
            ("cispo", None),
            ("sft", None),
        ],
    )
    def test_other_runs_chunk_token_tiles(
        self, algorithm: str, level: str | None
    ) -> None:
        assert liger_whole_sequence_loss(algorithm, level) is False


class TestLigerLossLayout:
    @pytest.mark.parametrize("algorithm", ["ppo", "reinforce"])
    def test_served_above_token_level_holds_whole_sequences(
        self, algorithm: str
    ) -> None:
        assert liger_loss_layout(
            algorithm, **{**TOKEN_LEVEL_RUN, "importance_sampling_level": "turn"}
        ) == (True, True)

    def test_served_at_token_level_tiles_tokens(self) -> None:
        assert liger_loss_layout("ppo", **TOKEN_LEVEL_RUN) == (True, False)

    def test_blocked_at_the_resolved_level_runs_the_standard_loss(self) -> None:
        # Act
        layout = liger_loss_layout(
            "ppo",
            **{
                **TOKEN_LEVEL_RUN,
                "importance_sampling_level": "turn",
                "reweights_tokens": True,
            },
        )

        # Assert
        assert layout == (False, False)

    def test_grpo_trajectory_level_holds_whole_sequences(self) -> None:
        assert liger_loss_layout(
            "grpo", **{**TOKEN_LEVEL_RUN, "importance_sampling_level": "trajectory"}
        ) == (True, True)


class TestLigerHeadGrad:
    def test_token_tiles_keep_the_head_grad(self) -> None:
        assert liger_head_grad("grpo", True, False) is True

    @pytest.mark.parametrize(
        ("algorithm", "liger_loss", "whole_sequence_loss"),
        [("ppo", True, True), ("ppo", False, False), ("sft", True, False)],
        ids=["whole-sequence-tile", "standard-path", "sft-frozen-head"],
    )
    def test_other_losses_hold_no_head_grad(
        self, algorithm: str, liger_loss: bool, whole_sequence_loss: bool
    ) -> None:
        assert liger_head_grad(algorithm, liger_loss, whole_sequence_loss) is False


class TestLigerKernelArch:
    def test_liger_layers_keep_the_checkpoint_geometry(self) -> None:
        arch = ModelArch(
            n_layers=2,
            hidden_size=64,
            intermediate_size=128,
            n_heads=4,
            n_kv_heads=4,
            head_dim=16,
            vocab_size=1000,
        )

        assert liger_kernel_arch(arch, True) is arch

    def test_stock_layers_keep_fp32_norm_inputs_and_the_activated_gate(self) -> None:
        arch = ModelArch(
            n_layers=2,
            hidden_size=64,
            intermediate_size=128,
            n_heads=4,
            n_kv_heads=4,
            head_dim=16,
            vocab_size=1000,
        )

        stock = liger_kernel_arch(arch, False)

        assert stock.fp32_norm_inputs is True
        assert stock.stock_gated_mlp is True
