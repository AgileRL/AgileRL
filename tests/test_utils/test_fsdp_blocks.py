# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Router top-k replay in activation-checkpointed gpt-oss blocks."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import CheckpointError, checkpoint
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter

from agilerl.distributed import FSDPConfig, init_distributed
from agilerl.distributed.fsdp_blocks import _block_checkpoint_contexts, apply_fsdp2
from agilerl.lora.moe import upgrade_moe_param_wrappers

VOCAB = 64
HIDDEN = 32


class DriftingRouter(GptOssTopKRouter):
    """gpt-oss router sending tokens to experts 0-1, but token 0 to 2-3 on odd calls."""

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__(config)
        self.calls = 0

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        odd = self.calls % 2
        self.calls += 1
        # Same ops every call: selective checkpointing replays saved ops by order.
        drift = torch.zeros(hidden_states.shape[0], self.num_experts)
        drift[:, :2] = 100.0
        drift[0, :2] = 100.0 * (1 - odd)
        drift[0, 2:] = 100.0 * odd
        router_logits = F.linear(hidden_states, self.weight, self.bias) + drift
        top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_scores = F.softmax(top_value, dim=1, dtype=top_value.dtype)
        return router_logits, router_scores, router_indices


def tiny_gpt_oss_with_lora() -> nn.Module:
    """Two-layer gpt-oss with drifting routers, attention and packed-expert LoRA, fp32."""
    config = GptOssConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "full_attention"],
        max_position_embeddings=64,
        use_cache=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = GptOssForCausalLM(config).float()
    for layer in model.model.layers:
        router = DriftingRouter(config)
        router.load_state_dict(layer.mlp.router.state_dict())
        layer.mlp.router = router
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "o_proj"],
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        ),
    )
    torch.manual_seed(1)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.1)
    upgrade_moe_param_wrappers(model)
    return model


def tiny_gpt_oss_mlp_with_lora() -> nn.Module:
    """First-layer MoE MLP of :func:`tiny_gpt_oss_with_lora`."""
    return tiny_gpt_oss_with_lora().get_submodule("base_model.model.model.layers.0.mlp")


def trainable_grads(module: nn.Module) -> dict[str, torch.Tensor]:
    """Full gradient of each trainable parameter, keyed by its unwrapped name."""
    return {
        name.replace("_checkpoint_wrapped_module.", ""): param.grad.full_tensor()
        if isinstance(param.grad, DTensor)
        else param.grad
        for name, param in module.named_parameters()
        if param.requires_grad
    }


@pytest.fixture
def world_size_one(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Single-process gloo group so FSDP2 can shard on CPU."""
    if sys.platform == "win32":
        pytest.skip("torch gloo process groups are unsupported on Windows wheels")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(29750 + os.getpid() % 100))
    # FSDP2's default mesh uses the host accelerator; shard on CPU everywhere.
    monkeypatch.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
    assert init_distributed() is True
    yield
    dist.destroy_process_group()


class TestBlockCheckpointContexts:
    def test_saved_router_topk_keeps_gpt_oss_forward_routing(self) -> None:
        # Arrange
        reference = tiny_gpt_oss_mlp_with_lora()
        mlp = tiny_gpt_oss_mlp_with_lora()
        torch.manual_seed(2)
        hidden = torch.randn(2, 6, HIDDEN)
        upstream = torch.randn(2, 6, HIDDEN)
        reference_hidden = hidden.clone().requires_grad_()
        (reference(reference_hidden)[0] * upstream).sum().backward()
        hidden.requires_grad_()

        # Act
        out, _scores = checkpoint(
            mlp,
            hidden,
            use_reentrant=False,
            context_fn=_block_checkpoint_contexts(True),
        )
        (out * upstream).sum().backward()

        # Assert
        expected = trainable_grads(reference)
        grads = trainable_grads(mlp)
        assert grads.keys() == expected.keys()
        for name, grad in grads.items():
            assert torch.equal(grad, expected[name]), name
        assert torch.equal(hidden.grad, reference_hidden.grad)

    def test_recomputed_router_reroutes_gpt_oss_tokens(self) -> None:
        # Arrange
        mlp = tiny_gpt_oss_mlp_with_lora()
        torch.manual_seed(2)
        hidden = torch.randn(2, 6, HIDDEN, requires_grad=True)
        out, _scores = checkpoint(
            mlp,
            hidden,
            use_reentrant=False,
            context_fn=_block_checkpoint_contexts(False),
        )

        # Act / Assert
        with pytest.raises(CheckpointError, match="different metadata"):
            out.sum().backward()


class TestApplyFsdp2:
    @pytest.mark.usefixtures("world_size_one")
    def test_checkpointed_gpt_oss_blocks_keep_forward_routing(self) -> None:
        # Arrange
        reference = tiny_gpt_oss_with_lora()
        model = tiny_gpt_oss_with_lora()
        torch.manual_seed(2)
        input_ids = torch.randint(0, VOCAB, (2, 6))
        upstream = torch.randn(2, 6, VOCAB)
        (reference(input_ids=input_ids).logits * upstream).sum().backward()

        # Act
        apply_fsdp2(
            model, FSDPConfig(param_dtype="float32"), gradient_checkpointing=True
        )
        (model(input_ids=input_ids).logits * upstream).sum().backward()

        # Assert
        expected = trainable_grads(reference)
        grads = trainable_grads(model)
        assert grads.keys() == expected.keys()
        for name, grad in grads.items():
            assert torch.equal(grad, expected[name]), name
