# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Regional ``torch.compile`` of dense transformer-block submodules."""

from __future__ import annotations

import copy
import os
import sys
from collections.abc import Generator

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch import nn
from torch._dynamo.utils import counters
from torch.distributed.tensor import DTensor
from transformers import NemotronHConfig, NemotronHForCausalLM

from agilerl.distributed import FSDPConfig, init_distributed
from agilerl.distributed.fsdp import (
    canonical_fsdp_param_fqn,
    compile_dense_block_modules,
    materialize_fsdp2_from_cpu_state,
)
from agilerl.lora.fused import (
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
)
from agilerl.lora.moe import (
    RoutedExpertsLoraWrapper,
    upgrade_moe_param_wrappers,
)

# Compile on CPU without inductor codegen; AOTAutograd still builds the
# compiled forward and backward graphs.
BACKEND = "aot_eager"
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


def assert_forward_backward_match(
    model: nn.Module, reference: nn.Module, input_ids: torch.Tensor
) -> None:
    """Logits and every gradient of ``model`` equal ``reference`` on one batch."""
    logits = model(input_ids=input_ids).logits
    expected = reference(input_ids=input_ids).logits
    logits.square().mean().backward()
    expected.square().mean().backward()

    # aot_eager runs the same aten kernels as eager, so default fp32 tolerances hold.
    torch.testing.assert_close(logits, expected)
    grads = {
        name: param.grad
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    expected_grads = {
        name: param.grad
        for name, param in reference.named_parameters()
        if param.grad is not None
    }
    assert grads.keys() == expected_grads.keys()
    assert grads
    for name, grad in grads.items():
        torch.testing.assert_close(grad, expected_grads[name], msg=name)
    model.zero_grad()
    reference.zero_grad()


def compiled_names(model: nn.Module) -> set[str]:
    """Names of submodules with a compiled ``__call__``."""
    return {
        name
        for name, module in model.named_modules()
        if module._compiled_call_impl is not None
    }


@pytest.fixture(autouse=True)
def fresh_dynamo() -> Generator[None, None, None]:
    """Drop dynamo caches and counters so graphs from other tests do not count."""
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()


class TestCompileDenseBlockModules:
    def test_compiles_norms_and_mlps(self) -> None:
        # Arrange
        model = tiny_hybrid_moe()

        # Act
        units = compile_dense_block_modules(model, BACKEND)

        # Assert
        names = {id(module): name for name, module in model.named_modules()}
        expected = {
            "model.layers.0.norm",
            "model.layers.1.norm",
            "model.layers.2.norm",
            "model.layers.2.mixer.shared_experts",
            "model.layers.3.norm",
            "model.layers.3.mixer",
        }
        assert {names[id(unit)] for unit in units} == expected
        assert compiled_names(model) == expected

    def test_leaves_moe_mixers_and_projections_eager(self) -> None:
        # Arrange
        model = tiny_hybrid_moe_with_lora()
        layers = model.base_model.model.model.layers
        experts = layers[2].mixer.experts

        # Act
        compile_dense_block_modules(model, BACKEND)

        # Assert
        assert isinstance(experts, RoutedExpertsLoraWrapper)
        eager = [
            experts,
            *experts.modules(),
            layers[2].mixer,
            layers[2].mixer.gate,
            layers[2].mixer.fc1_latent_proj,
            layers[0].mixer,
            layers[0].mixer.in_proj,
            layers[0].mixer.norm,
            layers[1].mixer,
            layers[1].mixer.q_proj,
            layers[1].mixer.k_proj,
        ]
        assert all(module._compiled_call_impl is None for module in eager)
        assert layers[2].mixer.shared_experts._compiled_call_impl is not None

    def test_outputs_and_grads_match_eager(self) -> None:
        # Arrange
        model = tiny_hybrid_moe()
        reference = copy.deepcopy(model)
        compile_dense_block_modules(model, BACKEND)
        input_ids = torch.randint(0, VOCAB, (2, 5))

        # Act / Assert
        assert_forward_backward_match(model, reference, input_ids)
        assert counters["stats"]["unique_graphs"] > 0

    def test_new_sequence_length_reuses_graphs(self) -> None:
        # Arrange
        model = tiny_hybrid_moe()
        reference = copy.deepcopy(model)
        compile_dense_block_modules(model, BACKEND)
        assert_forward_backward_match(model, reference, torch.randint(0, VOCAB, (2, 5)))
        graphs = counters["stats"]["unique_graphs"]

        # Act
        assert_forward_backward_match(model, reference, torch.randint(0, VOCAB, (2, 9)))

        # Assert
        assert counters["stats"]["unique_graphs"] == graphs

    def test_fused_lora_routing_matches_eager_without_graph_breaks(self) -> None:
        # Arrange
        model = tiny_hybrid_moe_with_lora()
        reference = copy.deepcopy(model)
        compile_dense_block_modules(model, BACKEND)
        set_fused_adapter_routing(model, ["default", "default"])
        set_fused_adapter_routing(reference, ["default", "default"])
        input_ids = torch.randint(0, VOCAB, (2, 6))

        # Act / Assert
        assert_forward_backward_match(model, reference, input_ids)
        assert not counters["graph_break"]

    def test_model_without_blocks_compiles_nothing(self) -> None:
        # Arrange
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))

        # Act
        units = compile_dense_block_modules(model, BACKEND)

        # Assert
        assert units == []
        assert compiled_names(model) == set()


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


class TestMaterializeFsdp2FromCpuStateCompileBlocks:
    @pytest.mark.usefixtures("world_size_one")
    def test_compile_blocks_compiles_dense_units_under_fsdp(self) -> None:
        # Arrange
        model = tiny_hybrid_moe()
        reference = copy.deepcopy(model)
        config = FSDPConfig(
            compile_blocks=True,
            compile_backend=BACKEND,
            param_dtype="float32",
        )

        # Act
        materialize_fsdp2_from_cpu_state(model, "cpu", config)

        # Assert
        assert compiled_names(model) == {
            "model.layers.0.norm",
            "model.layers.1.norm",
            "model.layers.2.norm",
            "model.layers.2.mixer.shared_experts",
            "model.layers.3.norm",
            "model.layers.3.mixer",
        }
        input_ids = torch.randint(0, VOCAB, (2, 5))
        torch.testing.assert_close(
            model(input_ids=input_ids).logits, reference(input_ids=input_ids).logits
        )

    @pytest.mark.usefixtures("world_size_one")
    def test_compile_blocks_with_checkpointing_matches_eager(self) -> None:
        # Arrange
        model = tiny_hybrid_moe()
        reference = copy.deepcopy(model)
        config = FSDPConfig(
            compile_blocks=True,
            compile_backend=BACKEND,
            param_dtype="float32",
            reduce_dtype="float32",
        )
        materialize_fsdp2_from_cpu_state(
            model, "cpu", config, gradient_checkpointing=True
        )
        input_ids = torch.randint(0, VOCAB, (2, 5))

        # Act
        logits = model(input_ids=input_ids).logits
        expected = reference(input_ids=input_ids).logits
        logits.square().mean().backward()
        expected.square().mean().backward()

        # Assert
        torch.testing.assert_close(logits, expected)
        expected_grads = {
            name: param.grad for name, param in reference.named_parameters()
        }
        grads = {
            canonical_fsdp_param_fqn(name): (
                param.grad.full_tensor()
                if isinstance(param.grad, DTensor)
                else param.grad
            )
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        assert grads
        for name, grad in grads.items():
            torch.testing.assert_close(grad, expected_grads[name], msg=name)

    @pytest.mark.usefixtures("world_size_one")
    def test_compile_blocks_off_compiles_nothing(self) -> None:
        # Arrange
        model = tiny_hybrid_moe()

        # Act
        materialize_fsdp2_from_cpu_state(model, "cpu", FSDPConfig())

        # Assert
        assert compiled_names(model) == set()
