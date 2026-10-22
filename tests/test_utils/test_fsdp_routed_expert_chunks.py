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
from torch.utils.checkpoint import CheckpointPolicy, checkpoint
from transformers import NemotronHConfig, NemotronHForCausalLM

from agilerl.distributed import FSDPConfig, init_distributed
from agilerl.distributed.ep_forward import (
    COPYING_ROUTED_COUNTS,
    TokenExchange,
    _routed_counts,
)
from agilerl.distributed.fsdp import materialize_fsdp2_from_cpu_state
from agilerl.distributed.fsdp_blocks import (
    _block_checkpoint_contexts,
    _save_routing_policy,
)
from agilerl.lora.fused import patch_lora_for_fused_forward
from agilerl.lora.moe import (
    RoutedExpertsLoraWrapper,
    upgrade_moe_param_wrappers,
)
from agilerl.lora.moe.grouped_gemm import expert_row_counts

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


class TestSaveRoutingPolicy:
    def test_saves_router_topk(self) -> None:
        policy = _save_routing_policy(None, torch.ops.aten.topk.default)

        assert policy is CheckpointPolicy.MUST_SAVE

    def test_saves_cpu_device_copy(self) -> None:
        policy = _save_routing_policy(
            None,
            torch.ops.aten._to_copy.default,
            device=torch.device("cpu"),
        )

        assert policy is CheckpointPolicy.MUST_SAVE

    def test_recomputes_routed_counts_copy(self) -> None:
        copying = COPYING_ROUTED_COUNTS.set(True)

        policy = _save_routing_policy(
            None,
            torch.ops.aten._to_copy.default,
            device=torch.device("cpu"),
        )

        COPYING_ROUTED_COUNTS.reset(copying)
        assert policy is CheckpointPolicy.PREFER_RECOMPUTE

    def test_recomputes_other_ops(self) -> None:
        policy = _save_routing_policy(None, torch.ops.aten.add.Tensor)

        assert policy is CheckpointPolicy.PREFER_RECOMPUTE


class TestBlockCheckpointContexts:
    @pytest.mark.usefixtures("world_size_one")
    def test_recompute_matches_forward_after_routed_counts_copy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        tolist = torch.Tensor.tolist
        # CUDA ``.tolist()`` copies to host through ``_to_copy``; CPU tensors skip it.
        monkeypatch.setattr(
            torch.Tensor,
            "tolist",
            lambda t: tolist(
                torch.ops.aten._to_copy.default(t, device=torch.device("cpu"))
            ),
        )
        exchange = TokenExchange(dist.group.WORLD, 1, 1, None, 1 << 20)
        torch.manual_seed(0)
        weight = torch.randn(4, 3, requires_grad=True)
        hidden = torch.randn(5, 4)
        splits = torch.tensor([2, 3])

        def block(x: torch.Tensor) -> torch.Tensor:
            scores, experts = (x @ weight).topk(2, -1)
            send = expert_row_counts(experts.reshape(-1), 3).unsqueeze(0)
            _routed_counts(send, exchange)
            chunks = scores.split(splits.tolist())
            return torch.cat([chunk.pow(i + 2) for i, chunk in enumerate(chunks)])

        upstream = torch.randn(5, 2)
        (block(hidden) * upstream).sum().backward()
        expected = weight.grad.clone()
        weight.grad = None

        # Act
        out = checkpoint(
            block,
            hidden,
            use_reentrant=False,
            context_fn=_block_checkpoint_contexts(True),
        )
        (out * upstream).sum().backward()

        # Assert
        assert torch.equal(weight.grad, expected)


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
