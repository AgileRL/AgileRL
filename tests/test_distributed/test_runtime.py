# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from agilerl.distributed.fsdp import FSDPConfig, canonical_fsdp_param_fqn
from agilerl.distributed.runtime import FSDPRuntime
from tests.parallel_helpers import (
    assert_same_state,
    broadcast_module,
    lora_runtime,
    with_lora,
)
from tests.rank_launch import run_ranks
from tests.test_architectures.qwen3.parallel_helpers import LORA_TARGETS, shard_data
from tests.test_architectures.qwen3.test_parallel import Qwen3BlockStack, _forward


def actor_with(config: PretrainedConfig, attn_implementation: str) -> nn.Module:
    config._attn_implementation = attn_implementation
    actor = nn.Linear(1, 1)
    actor.config = config
    return actor


def prepare_with_cp(actor: nn.Module, cp: int = 2, tp: int = 1) -> None:
    runtime = FSDPRuntime(
        FSDPConfig(cp=cp, tp=tp, reshard_after_forward=True, optim_cpu_offload=False)
    )
    runtime.prepare_actor(actor, "cpu", None, 1e-5, None, lambda _names: None)


def _no_grad_forward_resume_worker(
    rank: int, world_size: int, checkpoint_dir: str
) -> None:
    """Optimizer state exports and resumes after a no-grad forward left units unsharded."""
    # Arrange
    torch.manual_seed(0)
    dense = with_lora(Qwen3BlockStack(), LORA_TARGETS, std=0.1)
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    model, runtime, optimizer = lora_runtime(
        dense, world_size, 1, 2, 1, reshard_after_forward=False
    )
    hidden, probe = shard_data(rank // 2)
    loss = (_forward(model, hidden) * probe).sum()
    runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=model, max_grad_norm=1e6
    )
    trainable = {
        canonical_fsdp_param_fqn(name)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    with torch.no_grad():
        _forward(model, hidden)

    # Act
    exported = runtime.export_optimizer_state(model, optimizer)
    if rank == 0:
        torch.save(exported, f"{checkpoint_dir}/optimizer.pt")
    dist.barrier()
    checkpoint = torch.load(f"{checkpoint_dir}/optimizer.pt", weights_only=False)
    resumed, resumed_runtime, resumed_optimizer = lora_runtime(
        dense, world_size, 1, 2, 1, reshard_after_forward=False
    )
    with torch.no_grad():
        _forward(resumed, hidden)
    resumed_runtime.import_optimizer_state(resumed, resumed_optimizer, checkpoint)
    reexported = resumed_runtime.export_optimizer_state(resumed, resumed_optimizer)

    # Assert
    if rank != 0:
        return
    assert set(checkpoint["state"]) == trainable
    assert_same_state(reexported, checkpoint, "optimizer")


class TestFSDPRuntimeOptimizerState:
    @pytest.mark.multi_rank
    def test_export_and_import_after_no_grad_forward(self, tmp_path: Path) -> None:
        run_ranks(_no_grad_forward_resume_worker, 4, str(tmp_path), timeout=300)


class TestFSDPRuntimePrepareActor:
    def test_context_parallel_accepts_deepseek_v4_on_flex(self) -> None:
        actor = actor_with(DeepseekV4Config(), "flex_attention")

        # Validation passes; building the mesh then needs a process group.
        with pytest.raises(RuntimeError, match="initialised process group"):
            prepare_with_cp(actor)

    def test_context_parallel_refuses_a_backend_ulysses_does_not_patch(
        self,
    ) -> None:
        actor = actor_with(Qwen3Config(), "sdpa")

        with pytest.raises(
            ValueError, match="needs flash_attention_2 or flex_attention"
        ):
            prepare_with_cp(actor)

    def test_context_parallel_splits_tp_local_heads(self) -> None:
        config = Qwen3Config(num_attention_heads=4, num_key_value_heads=4)
        actor = actor_with(config, "flex_attention")

        with pytest.raises(
            ValueError, match=r"num_attention_heads \(2\) divisible by cp \(4\)"
        ):
            prepare_with_cp(actor, cp=4, tp=2)
