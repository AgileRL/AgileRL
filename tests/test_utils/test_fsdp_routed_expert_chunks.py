# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Routed-expert chunk budget set by ``materialize_fsdp2_from_cpu_state``."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import NemotronHConfig, NemotronHForCausalLM

from agilerl.distributed import FSDPConfig, init_distributed
from agilerl.distributed.fsdp import materialize_fsdp2_from_cpu_state
from agilerl.lora.fused import patch_lora_for_fused_forward
from agilerl.lora.moe import (
    RoutedExpertsLoraWrapper,
    upgrade_moe_param_wrappers,
)

VOCAB = 64


def tiny_hybrid_moe() -> NemotronHForCausalLM:
    """Mamba, attention, MoE and dense-MLP blocks, fp32, seeded."""
    config = NemotronHConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        layers_block_type=["mamba", "attention", "moe", "mlp"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=32,
        use_mamba_kernels=False,
        ssm_state_size=8,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=1,
        conv_kernel=2,
        expand=2,
        chunk_size=8,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=64,
        use_cache=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return NemotronHForCausalLM(config).float()


def tiny_hybrid_moe_with_lora() -> nn.Module:
    """``tiny_hybrid_moe`` with dense and packed-expert LoRA, fused routing patched."""
    model = get_peft_model(
        tiny_hybrid_moe(),
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "o_proj", "in_proj", "up_proj", "down_proj"],
            target_parameters=["mixer.experts.up_proj", "mixer.experts.down_proj"],
        ),
    )
    torch.manual_seed(1)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.1)
    upgrade_moe_param_wrappers(model)
    patch_lora_for_fused_forward(model)
    return model


@pytest.fixture
def world_size_one(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Single-process gloo group so FSDP2 can shard on CPU."""
    if sys.platform == "win32":
        pytest.skip("torch gloo process groups are unsupported on Windows wheels")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(29850 + os.getpid() % 100))
    # FSDP2's default mesh uses the host accelerator; shard on CPU everywhere.
    monkeypatch.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
    assert init_distributed() is True
    yield
    dist.destroy_process_group()


class TestMaterializeFsdp2FromCpuStateRoutedExpertChunks:
    @pytest.mark.usefixtures("world_size_one")
    def test_sets_the_chunk_budget_on_every_routed_wrapper(self) -> None:
        # Arrange
        model = tiny_hybrid_moe_with_lora()

        # Act
        materialize_fsdp2_from_cpu_state(
            model, "cpu", FSDPConfig(routed_expert_chunk_mib=256)
        )

        # Assert
        wrappers = [
            module
            for module in model.modules()
            if isinstance(module, RoutedExpertsLoraWrapper)
        ]
        assert wrappers
        assert all(wrapper.chunk_bytes == 256 * 1024 * 1024 for wrapper in wrappers)
