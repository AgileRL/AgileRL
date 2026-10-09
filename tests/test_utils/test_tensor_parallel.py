# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel dense layers on the tensor-parallel mesh."""

from __future__ import annotations

import copy
import os
import sys
import traceback
from functools import partial
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor
from transformers import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention,
    NemotronHMLP,
    NemotronHMoE,
)

from agilerl.distributed import FSDPConfig
from agilerl.distributed import tensor_parallel as tp_mod
from agilerl.distributed.expert_parallel import build_parallel_mesh
from agilerl.distributed.fsdp import (
    materialize_dtensors,
    materialize_fsdp2_from_cpu_state,
)
from agilerl.distributed.fsdp_blocks import apply_fsdp2
from agilerl.distributed.process import sync_grads
from agilerl.distributed.tensor_parallel import (
    SharedSeedDropout,
    apply_tensor_parallel,
    copy_input_to_region,
)
from agilerl.utils.llm_utils import get_lora_named_params
from tests.dist_ports import get_free_port, gloo_rank_env

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
    os.environ.update(gloo_rank_env(rank, world_size, port))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _spawn_ranks(worker, world_size: int = 2, timeout: float = 300.0) -> None:
    port = get_free_port()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(rank, world_size, port, queue))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    try:
        results = [queue.get(timeout=timeout) for _ in range(world_size)]
        for proc in procs:
            proc.join(timeout=timeout)
            assert proc.exitcode == 0, f"rank exited {proc.exitcode}"
        for rank, status, err in sorted(results):
            assert status == "ok", f"rank {rank}: {err}"
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)


def _broadcast_module(module: nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(param.data, src=0)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=1e-4, rtol=1e-4):
        diff = (actual - expected).abs().max().item()
        msg = f"max abs diff {diff}"
        raise AssertionError(msg)


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    # Sharded fp32 sums reduce in another order, so a near-zero element can be
    # off by a few ulps of the tensor's largest element.
    scale = max(expected.abs().max().item(), 1.0)
    if not torch.allclose(actual, expected, atol=1e-5 * scale, rtol=1e-5):
        diff = (actual - expected).abs().max().item()
        msg = f"{label}: max abs diff {diff}"
        raise AssertionError(msg)


def _with_lora(module: nn.Module, targets: list[str]) -> nn.Module:
    inject_adapter_in_model(LoraConfig(r=2, target_modules=targets), module)
    module.requires_grad_(True)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.5)
    return module


def _assert_grads_match(module: nn.Module, reference: nn.Module) -> None:
    """Every trainable grad equals this rank's share of the unsharded grad."""
    reference_params = dict(reference.named_parameters())
    for name, param in module.named_parameters():
        full = reference_params[name].grad
        if isinstance(param, DTensor):
            local = distribute_tensor(
                full, param.device_mesh, param.placements, src_data_rank=None
            )
            _assert_exact(param.grad.to_local(), local.to_local(), name)
        else:
            _assert_exact(param.grad, full, name)


def _forward_backward(
    module: nn.Module, source: torch.Tensor, probe: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = source.clone().requires_grad_(True)
    output = _hidden(module(inputs))
    (output * probe).sum().backward()
    return output, inputs.grad


def _hidden(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if not isinstance(output, torch.Tensor):
        msg = f"expected a tensor, got {type(output).__name__}"
        raise TypeError(msg)
    return output


def _tp_mesh(world_size: int, tp: int):
    mesh = build_parallel_mesh(world_size=world_size, tp=tp, device_type="cpu")
    if mesh is None:
        msg = "expected a tensor-parallel mesh"
        raise AssertionError(msg)
    return mesh


def _attention(
    num_attention_heads: int,
    num_key_value_heads: int,
    *,
    head_dim: int = 4,
    hidden_size: int = 16,
) -> NemotronHAttention:
    config = NemotronHConfig(
        hidden_size=hidden_size,
        vocab_size=32,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        attn_implementation="eager",
        layers_block_type=["full_attention"],
    )
    config._attn_implementation = "eager"
    module = NemotronHAttention(config, layer_idx=0)
    module.eval()
    return module


def _fill(module: nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.normal_()


def _attention_parity_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        tp = 2
        head_dim = 4
        attn = _attention(4, 2, head_dim=head_dim, hidden_size=16)
        _with_lora(attn, ["q_proj", "k_proj", "v_proj", "o_proj"])
        _broadcast_module(attn)
        reference = copy.deepcopy(attn)
        source = torch.randn(2, 4, attn.config.hidden_size)
        probe = torch.randn(2, 4, attn.config.hidden_size)
        dist.broadcast(source, src=0)
        dist.broadcast(probe, src=0)
        mesh = _tp_mesh(world_size, tp)
        apply_tensor_parallel(attn, mesh.tp)
        q_weight = attn.q_proj.weight
        k_weight = attn.k_proj.weight
        o_weight = attn.o_proj.weight
        if not isinstance(q_weight, DTensor) or not isinstance(k_weight, DTensor):
            msg = "q_proj and k_proj weights must be DTensors when kv heads divide tp"
            raise AssertionError(msg)
        if not isinstance(o_weight, DTensor) or o_weight.placements[0].dim != 1:
            msg = "o_proj weight must be Shard(1)"
            raise AssertionError(msg)
        if k_weight.to_local().shape[0] * tp != k_weight.shape[0]:
            msg = "k_proj local rows are not an even partition"
            raise AssertionError(msg)

        expected, expected_input_grad = _forward_backward(reference, source, probe)
        actual, actual_input_grad = _forward_backward(attn, source, probe)

        _assert_exact(actual, expected, "output")
        _assert_exact(actual_input_grad, expected_input_grad, "input grad")
        _assert_grads_match(attn, reference)
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _shared_kv_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        tp = 4
        head_dim = 4
        attn = _attention(8, 2, head_dim=head_dim, hidden_size=16)
        _with_lora(attn, ["q_proj", "k_proj", "v_proj", "o_proj"])
        _broadcast_module(attn)
        reference = copy.deepcopy(attn)
        source = torch.randn(2, 4, attn.config.hidden_size)
        probe = torch.randn(2, 4, attn.config.hidden_size)
        dist.broadcast(source, src=0)
        dist.broadcast(probe, src=0)
        mesh = _tp_mesh(world_size, tp)
        apply_tensor_parallel(attn, mesh.tp)
        k_weight = attn.k_proj.weight
        v_weight = attn.v_proj.weight
        if isinstance(k_weight, DTensor) or isinstance(v_weight, DTensor):
            msg = "shared key/value weights must stay plain tensors"
            raise AssertionError(msg)
        if k_weight.shape[0] != 2 * head_dim or v_weight.shape[0] != 2 * head_dim:
            msg = f"expected both kv heads on every rank, got {tuple(k_weight.shape)}"
            raise AssertionError(msg)
        replicated = getattr(attn.k_proj.base_layer, "tp_replicated_params", ())
        if "weight" not in replicated:
            msg = "shared k_proj.weight must be in tp_replicated_params"
            raise AssertionError(msg)

        expected, expected_input_grad = _forward_backward(reference, source, probe)
        actual, actual_input_grad = _forward_backward(attn, source, probe)

        _assert_exact(actual, expected, "output")
        _assert_exact(actual_input_grad, expected_input_grad, "input grad")
        _assert_grads_match(attn, reference)
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _sgd_step(module: nn.Module, lr: float) -> None:
    with torch.no_grad():
        for param in module.parameters():
            if param.grad is not None:
                param.sub_(lr * param.grad)


def _assert_lora_export_matches(module: nn.Module, reference: nn.Module) -> None:
    """Gathered LoRA tensors equal the unsharded model's, row for row."""
    named = get_lora_named_params(module)
    reference_params = dict(reference.named_parameters())
    with materialize_dtensors(*[param for _, param in named]) as dense:
        for (name, _), exported in zip(named, dense, strict=True):
            _assert_exact(exported, reference_params[name].detach(), name)


def _lora_export_worker(
    tp: int, rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        # Arrange
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        attn = _with_lora(
            _attention(8, 2, head_dim=4, hidden_size=16),
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        _broadcast_module(attn)
        reference = copy.deepcopy(attn)
        source = torch.randn(2, 4, 16)
        probe = torch.randn(2, 4, 16)
        dist.broadcast(source, src=0)
        dist.broadcast(probe, src=0)
        apply_tensor_parallel(attn, _tp_mesh(world_size, tp).tp)

        # Act
        for module in (reference, attn):
            _forward_backward(module, source, probe)
            _sgd_step(module, lr=0.1)

        # Assert
        _assert_lora_export_matches(attn, reference)
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _shared_kv_load_worker(
    checkpoint_dir: str, rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        # Arrange
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        head_dim = 4
        dense = AttentionStack(_attention(8, 2, head_dim=head_dim, hidden_size=16))
        if rank == 0:
            save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
        dist.barrier()
        with torch.device("meta"):
            attn = _attention(8, 2, head_dim=head_dim, hidden_size=16)
        attn.config._name_or_path = checkpoint_dir
        model = AttentionStack(attn)
        model.config = attn.config
        source = torch.randn(2, 4, 16)
        dist.broadcast(source, src=0)
        config = FSDPConfig(
            tp=world_size,
            wrap_every_n_blocks=1,
            param_persistence_threshold=0,
            param_dtype="float32",
        )

        # Act
        materialize_fsdp2_from_cpu_state(
            model, "cpu", config, parallel_mesh=_tp_mesh(world_size, world_size)
        )

        # Assert
        loaded = model.layers[0].attn
        expected = dense.layers[0].attn
        _assert_exact(loaded.k_proj.weight, expected.k_proj.weight, "k_proj.weight")
        _assert_exact(loaded.v_proj.weight, expected.v_proj.weight, "v_proj.weight")
        with torch.no_grad():
            _assert_exact(_hidden(loaded(source)), _hidden(expected(source)), "output")
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _mixed_precision_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        # Arrange
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        tp = 2
        attn = _attention(4, 1)
        inject_adapter_in_model(
            LoraConfig(r=2, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
            attn,
        )
        model = AttentionStack(attn)
        _broadcast_module(model)
        for name, param in model.named_parameters():
            param.requires_grad_("lora_" in name)
        mesh = _tp_mesh(world_size, tp)
        apply_tensor_parallel(model, mesh.tp)
        apply_fsdp2(
            model,
            FSDPConfig(
                tp=tp,
                wrap_every_n_blocks=1,
                param_persistence_threshold=0,
                param_dtype="bfloat16",
                reduce_dtype="float32",
            ),
            mesh=mesh.hsdp,
        )
        forward_dtypes: list[torch.dtype] = []
        for name, module in attn.named_modules():
            if ".lora_" in name and isinstance(module, nn.Linear):
                module.register_forward_pre_hook(
                    lambda mod, _args: forward_dtypes.append(mod.weight.dtype)
                )
        trainable = [param for param in model.parameters() if param.requires_grad]
        torch.manual_seed(rank)
        source = torch.randn(2, 4, attn.config.hidden_size, dtype=torch.bfloat16)

        # Act
        output = _hidden(attn(source))
        output.float().sum().backward()
        mesh.sync_grads(trainable, torch.float32)
        sync_grads([param for param in trainable if not isinstance(param, DTensor)])

        # Assert
        assert output.dtype == torch.bfloat16
        assert len(forward_dtypes) == 8
        assert set(forward_dtypes) == {torch.bfloat16}
        assert {param.dtype for param in trainable} == {torch.float32}
        assert {param.grad.dtype for param in trainable} == {torch.float32}
        assert attn.q_proj.base_layer.weight.dtype == torch.bfloat16
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _heads_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        attn = _attention(4, 2)
        mesh = _tp_mesh(world_size, tp=3)
        with pytest.raises(ValueError, match=r"num_attention_heads \(4\).*tp \(3\)"):
            apply_tensor_parallel(attn, mesh.tp)
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class LatentPair(nn.Module):
    def __init__(self, hidden: int, latent: int) -> None:
        super().__init__()
        self.fc1_latent_proj = nn.Linear(hidden, latent, bias=False)
        self.fc2_latent_proj = nn.Linear(latent, hidden, bias=True)
        self.experts = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1_latent_proj(hidden_states)
        hidden_states = self.experts(hidden_states)
        return self.fc2_latent_proj(hidden_states)


class VisionAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.query = nn.Linear(hidden, inner, bias=False)
        self.key = nn.Linear(hidden, inner, bias=False)
        self.value = nn.Linear(hidden, inner, bias=False)
        self.proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        query = self.query(hidden_states).view(
            batch, seq, self.num_heads, self.head_dim
        )
        key = self.key(hidden_states).view(batch, seq, self.num_heads, self.head_dim)
        value = self.value(hidden_states).view(
            batch, seq, self.num_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        weights = torch.softmax(
            query @ key.transpose(-1, -2) * (self.head_dim**-0.5),
            dim=-1,
        )
        out = (weights @ value).transpose(1, 2).contiguous()
        return self.proj(out.view(batch, seq, -1))


class VisionMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, *, bias: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=False)
        self.fc2 = nn.Linear(intermediate, hidden, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(hidden_states)))


class VisionTower(nn.Module):
    def __init__(self, attn: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.attn = attn
        self.vision_model.mlp = mlp


class CausalHead(nn.Module):
    def __init__(self, vocab: int, hidden: int, *, tie: bool) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=tie)
        if tie:
            self.lm_head.weight = self.embed.weight


def _dense_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        tp = 2
        mesh = _tp_mesh(world_size, tp)

        mlp_config = NemotronHConfig(
            hidden_size=8,
            intermediate_size=8,
            vocab_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            layers_block_type=["mlp"],
            mlp_bias=False,
        )
        mlp = NemotronHMLP(mlp_config, intermediate_size=8)
        mlp.eval()
        _broadcast_module(mlp)
        mlp_reference = copy.deepcopy(mlp)
        mlp_source = torch.randn(2, 4, 8)
        dist.broadcast(mlp_source, src=0)
        apply_tensor_parallel(mlp, mesh.tp)
        _assert_close(mlp(mlp_source), mlp_reference(mlp_source))

        latent = LatentPair(hidden=8, latent=4)
        _broadcast_module(latent)
        latent_reference = copy.deepcopy(latent)
        latent_source = torch.randn(2, 4, 8)
        dist.broadcast(latent_source, src=0)
        apply_tensor_parallel(latent, mesh.tp)
        latent_output = latent(latent_source)
        _assert_close(latent_output, latent_reference(latent_source))
        latent_output.sum().backward()
        fc2_grad = latent.fc2_latent_proj.weight.grad
        if isinstance(fc2_grad, DTensor):
            fc2_grad = fc2_grad.to_local()
        if not isinstance(fc2_grad, torch.Tensor) or not torch.isfinite(fc2_grad).all():
            msg = "fc2_latent_proj.weight.grad is not finite"
            raise AssertionError(msg)

        moe_config = NemotronHConfig(
            hidden_size=8,
            vocab_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            moe_intermediate_size=8,
            moe_shared_expert_intermediate_size=8,
            moe_latent_size=4,
            n_routed_experts=4,
            num_experts_per_tok=1,
            n_group=1,
            topk_group=1,
            layers_block_type=["moe"],
            mlp_bias=False,
        )
        moe = NemotronHMoE(moe_config, layer_idx=0)
        _fill(moe)
        moe.eval()
        _broadcast_module(moe)
        moe_reference = copy.deepcopy(moe)
        moe_source = torch.randn(2, 3, 8)
        dist.broadcast(moe_source, src=0)
        apply_tensor_parallel(moe, mesh.tp)
        if not isinstance(moe.fc1_latent_proj.weight, DTensor):
            msg = "NemotronHMoE.fc1_latent_proj.weight must be a DTensor"
            raise AssertionError(msg)
        if isinstance(moe.experts.up_proj, DTensor) or isinstance(
            moe.gate.weight, DTensor
        ):
            msg = "routed experts and the router must stay plain tensors"
            raise AssertionError(msg)
        _assert_close(moe(moe_source), moe_reference(moe_source))

        vision_attn = VisionAttention(hidden=8, num_heads=4, head_dim=2)
        vision_mlp = VisionMLP(hidden=8, intermediate=8)
        tower = VisionTower(vision_attn, vision_mlp)
        tower.eval()
        _broadcast_module(tower)
        vision_reference = copy.deepcopy(tower)
        vision_source = torch.randn(2, 4, 8)
        dist.broadcast(vision_source, src=0)
        apply_tensor_parallel(tower, mesh.tp)
        _assert_close(
            tower.vision_model.attn(vision_source),
            vision_reference.vision_model.attn(vision_source),
        )
        _assert_close(
            tower.vision_model.mlp(vision_source),
            vision_reference.vision_model.mlp(vision_source),
        )

        biased_mlp = VisionMLP(hidden=8, intermediate=8, bias=True)
        biased_tower = VisionTower(
            VisionAttention(hidden=8, num_heads=4, head_dim=2),
            biased_mlp,
        )
        biased_tower.eval()
        _broadcast_module(biased_tower)
        biased_reference = copy.deepcopy(biased_tower)
        biased_source = torch.randn(2, 4, 8)
        dist.broadcast(biased_source, src=0)
        apply_tensor_parallel(biased_tower, mesh.tp)
        stale = biased_mlp.fc2.bias
        replaced = nn.Parameter(
            stale.detach().clone() + 1, requires_grad=stale.requires_grad
        )
        biased_mlp.fc2.bias = replaced
        ref_bias = biased_reference.vision_model.mlp.fc2.bias
        ref_bias.data.copy_(replaced.detach())
        biased_out = biased_tower.vision_model.mlp(biased_source)
        _assert_close(
            biased_out,
            biased_reference.vision_model.mlp(biased_source),
        )
        biased_out.sum().backward()
        biased_reference.vision_model.mlp(biased_source).sum().backward()
        bias_grad = biased_mlp.fc2.bias.grad
        ref_grad = biased_reference.vision_model.mlp.fc2.bias.grad
        if bias_grad is None or ref_grad is None:
            msg = "fc2.bias.grad is missing"
            raise AssertionError(msg)
        _assert_close(bias_grad, ref_grad)

        odd = CausalHead(vocab=7, hidden=8, tie=False)
        with pytest.raises(ValueError, match=r"vocab_size \(7\).*tp \(2\)"):
            apply_tensor_parallel(odd, mesh.tp)

        tied = CausalHead(vocab=16, hidden=8, tie=True)
        apply_tensor_parallel(tied, mesh.tp)
        if isinstance(tied.lm_head.weight, DTensor):
            msg = "tied lm_head.weight must stay a plain tensor"
            raise AssertionError(msg)
        if tied.lm_head.weight is not tied.embed.weight:
            msg = "tied lm_head.weight must stay the embedding weight"
            raise AssertionError(msg)

        head = CausalHead(vocab=16, hidden=8, tie=False)
        _broadcast_module(head)
        head_reference = copy.deepcopy(head)
        head_source = torch.randn(2, 4, 8)
        dist.broadcast(head_source, src=0)
        apply_tensor_parallel(head, mesh.tp)
        _assert_close(head.lm_head(head_source), head_reference.lm_head(head_source))
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _latent_grad_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        # Arrange
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        mesh = _tp_mesh(world_size, tp=2)
        latent = _with_lora(
            LatentPair(hidden=8, latent=4), ["fc1_latent_proj", "fc2_latent_proj"]
        )
        mlp_config = NemotronHConfig(
            hidden_size=8,
            intermediate_size=8,
            vocab_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            layers_block_type=["mlp"],
            mlp_bias=False,
        )
        mlp = _with_lora(
            NemotronHMLP(mlp_config, intermediate_size=8), ["up_proj", "down_proj"]
        )
        modules = [latent, mlp]
        for module in modules:
            _broadcast_module(module)
        references = [copy.deepcopy(module) for module in modules]
        source = torch.randn(2, 4, 8)
        probe = torch.randn(2, 4, 8)
        dist.broadcast(source, src=0)
        dist.broadcast(probe, src=0)
        for module in modules:
            apply_tensor_parallel(module, mesh.tp)

        for module, reference in zip(modules, references, strict=True):
            # Act
            expected, expected_input_grad = _forward_backward(reference, source, probe)
            actual, actual_input_grad = _forward_backward(module, source, probe)

            # Assert
            label = type(module).__name__
            _assert_exact(actual, expected, f"{label} output")
            _assert_exact(actual_input_grad, expected_input_grad, f"{label} input")
            _assert_grads_match(module, reference)
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _dropout_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        torch.manual_seed(rank)
        latent = LatentPair(hidden=8, latent=4)
        config = LoraConfig(
            r=2,
            target_modules=["fc1_latent_proj", "fc2_latent_proj"],
            lora_dropout=0.5,
        )
        inject_adapter_in_model(config, latent)
        mesh = _tp_mesh(world_size, tp=2)
        apply_tensor_parallel(latent, mesh.tp)
        column = latent.fc1_latent_proj.lora_dropout["default"]
        row = latent.fc2_latent_proj.lora_dropout["default"]
        column.train()

        mask = column(torch.ones(256))
        masks = [torch.empty_like(mask) for _ in range(world_size)]
        dist.all_gather(masks, mask)

        assert isinstance(column, SharedSeedDropout)
        assert isinstance(row, nn.Dropout)
        assert torch.equal(masks[0], masks[1])
        assert bool((mask == 0).any())
        assert bool((mask == 2).any())
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class AttentionBlock(nn.Module):
    def __init__(self, attn: NemotronHAttention) -> None:
        super().__init__()
        self.attn = attn
        self.lin = nn.Linear(attn.config.hidden_size, attn.config.hidden_size)


class AttentionStack(nn.Module):
    _no_split_modules: ClassVar = ["AttentionBlock"]

    def __init__(self, attn: NemotronHAttention) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(attn)])


def _fsdp_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        tp = 2
        attn = _attention(4, 2)
        model = AttentionStack(attn)
        _broadcast_module(model)
        mesh = _tp_mesh(world_size, tp)
        apply_tensor_parallel(model, mesh.tp)
        before = {
            id(param)
            for param in model.parameters()
            if isinstance(param, DTensor) and param.device_mesh is mesh.tp
        }
        if not before:
            msg = "expected attention DTensors on the tp mesh"
            raise AssertionError(msg)
        apply_fsdp2(
            model,
            FSDPConfig(tp=tp, wrap_every_n_blocks=1, param_persistence_threshold=0),
            mesh=mesh.hsdp,
        )
        after = {
            id(param)
            for param in model.parameters()
            if isinstance(param, DTensor) and param.device_mesh is mesh.tp
        }
        if after != before:
            msg = "attention DTensors must stay on the tp mesh"
            raise AssertionError(msg)
        q_weight = model.layers[0].attn.q_proj.weight
        if not isinstance(q_weight, DTensor) or q_weight.device_mesh is not mesh.tp:
            msg = "q_proj.weight must stay on the tp mesh"
            raise AssertionError(msg)
        lin = model.layers[0].lin.weight
        if not isinstance(lin, DTensor) or lin.device_mesh is mesh.tp:
            msg = "non-TP linear must be on the FSDP mesh"
            raise AssertionError(msg)
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _lm_head_fsdp_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        tp = 2
        head = CausalHead(16, 8, tie=False)
        mesh = _tp_mesh(world_size, tp)
        apply_tensor_parallel(head, mesh.tp)
        weight = head.lm_head.weight
        if not isinstance(weight, DTensor) or weight.device_mesh is not mesh.tp:
            msg = "lm_head.weight must be on the tp mesh before FSDP"
            raise AssertionError(msg)
        apply_fsdp2(
            head,
            FSDPConfig(tp=tp, wrap_every_n_blocks=1, param_persistence_threshold=0),
            mesh=mesh.hsdp,
        )
        weight = head.lm_head.weight
        if not isinstance(weight, DTensor) or weight.device_mesh is not mesh.tp:
            msg = "lm_head.weight must stay on the tp mesh"
            raise AssertionError(msg)
        result_queue.put((rank, "ok", None))
    except BaseException:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestApplyTensorParallel:
    def test_attention_forward_and_grads_match_unsharded(self):
        _spawn_ranks(_attention_parity_worker)

    def test_shared_kv_head_forward_and_grads_match_unsharded(self):
        _spawn_ranks(_shared_kv_worker, world_size=4)

    @pytest.mark.parametrize("tp", [2, 4])
    def test_lora_export_after_step_matches_unsharded(self, tp):
        _spawn_ranks(partial(_lora_export_worker, tp), world_size=tp)

    def test_shared_kv_head_load_matches_checkpoint(self, tmp_path):
        _spawn_ranks(partial(_shared_kv_load_worker, str(tmp_path)), world_size=4)

    def test_lora_runs_in_param_dtype_and_grads_sync_in_reduce_dtype(self):
        _spawn_ranks(_mixed_precision_worker, world_size=4)

    def test_latent_and_mlp_grads_match_unsharded(self):
        _spawn_ranks(_latent_grad_worker)

    def test_column_lora_dropout_mask_matches_across_ranks(self):
        _spawn_ranks(_dropout_worker)

    def test_attention_heads_not_divisible_by_tp_raises(self):
        _spawn_ranks(_heads_worker, world_size=3)

    def test_mlp_latent_vision_and_lm_head_match_unsharded(self):
        _spawn_ranks(_dense_worker)

    def test_fsdp_leaves_attention_dtensors_on_tp_mesh(self):
        _spawn_ranks(_fsdp_worker)

    def test_fsdp_leaves_lm_head_on_tp_mesh(self):
        _spawn_ranks(_lm_head_fsdp_worker)


class TestTensorParallelGuards:
    def test_base_linear_reads_a_base_layer_attribute(self):
        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = nn.Linear(2, 3)

        wrap = Wrap()

        assert tp_mod._base_linear(wrap) is wrap.base_layer

    def test_shard_weight_returns_when_the_weight_is_missing(self):
        tp_mod._shard_weight(nn.Module(), MagicMock(), tp_mod.Shard(0))

    def test_lora_linears_reads_a_plain_dict_bank(self):
        linear = nn.Linear(2, 2)
        wrapper = nn.Module()
        wrapper.lora_A = {"actor": linear, "skip": "nope"}

        assert tp_mod._lora_linears(wrapper, "lora_A") == [linear]

    def test_shared_seed_dropout_is_identity_in_eval(self):
        drop = SharedSeedDropout(0.5, seed=0)
        drop.eval()
        x = torch.ones(3, 3)

        assert drop(x) is x

    def test_validate_gqa_rejects_heads_that_do_not_divide_kv(self):
        with pytest.raises(ValueError, match="must be divisible by"):
            tp_mod._validate_gqa(4, 3, tp=1)

    def test_validate_gqa_rejects_a_query_span_that_crosses_a_kv_group(self):
        with pytest.raises(ValueError, match="cross a key-value group"):
            tp_mod._validate_gqa(12, 4, tp=3)

    def test_shard_kv_returns_when_the_weight_is_already_a_dtensor(self):
        linear = nn.Linear(4, 4)
        with patch("agilerl.distributed.tensor_parallel.DTensor", nn.Parameter):
            tp_mod._shard_kv(linear, MagicMock(), num_q=4, num_kv=2, head_dim=2)

    def test_shard_kv_rejects_row_count_that_does_not_match_kv_heads(self):
        linear = nn.Linear(4, 8)
        mesh = MagicMock()
        mesh.size.return_value = 4

        with pytest.raises(ValueError, match="do not match"):
            tp_mod._shard_kv(linear, mesh, num_q=8, num_kv=2, head_dim=2)

    def test_copy_input_to_region_rewrites_hidden_states_kwarg(self):
        hidden = torch.ones(2, 2)
        with patch.object(
            tp_mod.CopyToTPRegion, "apply", side_effect=lambda tensor, _group: tensor
        ):
            args, kwargs = copy_input_to_region((), {"hidden_states": hidden}, object())

        assert args == ()
        assert kwargs["hidden_states"] is hidden

    def test_live_row_bias_is_none_without_a_plain_bias(self):
        linear = nn.Linear(2, 2, bias=False)

        assert tp_mod._live_row_bias(None) is None
        assert tp_mod._live_row_bias(linear) is None

    def test_tp_group_raises_without_a_process_group(self):
        with pytest.raises(RuntimeError, match="no process group"):
            tp_mod._tp_group(nn.Linear(2, 2))

    def test_all_reduce_hidden_adds_bias_to_a_tuple_output(self):
        hidden = torch.ones(2, 2)
        extra = object()
        bias = nn.Parameter(torch.full((2,), 3.0))
        with patch.object(
            tp_mod.ReduceFromTPRegion,
            "apply",
            side_effect=lambda tensor, _group: tensor,
        ):
            out = tp_mod._all_reduce_hidden((hidden, extra), object(), bias)

        assert torch.equal(out[0], hidden + 3)
        assert out[1] is extra

    def test_all_reduce_hidden_rejects_a_non_tensor_tuple(self):
        with pytest.raises(TypeError, match="did not return a tensor"):
            tp_mod._all_reduce_hidden((object(),), object(), None)

    def test_all_reduce_hidden_rejects_a_non_tensor_output(self):
        with pytest.raises(TypeError, match="did not return a tensor"):
            tp_mod._all_reduce_hidden("nope", object(), None)

    def test_install_forwards_are_idempotent(self):
        module = nn.Linear(2, 2)
        object.__setattr__(module, "_agilerl_dense_tp", True)

        tp_mod._install_block_forward(module, None)
        tp_mod.install_reduce_forward(module, None)
        tp_mod._install_gather_forward(module)
        tp_mod._install_split_forward(module, None)

        assert "forward" not in module.__dict__

    def test_gather_forward_rejects_a_non_tensor_output(self):
        linear = nn.Linear(2, 2)
        object.__setattr__(linear, "_tp_group", MagicMock(spec=dist.ProcessGroup))
        object.__setattr__(linear, "_tp_mesh", MagicMock(get_local_rank=lambda: 0))
        tp_mod._install_gather_forward(linear)
        with (
            patch(
                "agilerl.distributed.tensor_parallel.copy_input_to_region",
                return_value=((torch.ones(2),), {}),
            ),
            patch(
                "agilerl.distributed.tensor_parallel._call_local_forward",
                return_value="nope",
            ),
        ):
            with pytest.raises(TypeError, match="column-parallel gather"):
                linear(torch.ones(2))

    def test_split_forward_rejects_a_non_tensor_input(self):
        linear = nn.Linear(2, 2)
        object.__setattr__(linear, "_tp_group", MagicMock(spec=dist.ProcessGroup))
        object.__setattr__(linear, "_tp_mesh", MagicMock(get_local_rank=lambda: 0))
        object.__setattr__(linear, "_tp_degree", 2)
        tp_mod._install_split_forward(linear, None)

        with pytest.raises(TypeError, match="expected a tensor input"):
            linear("nope")

    def test_split_forward_rejects_a_non_tensor_output(self):
        linear = nn.Linear(2, 2)
        object.__setattr__(linear, "_tp_group", MagicMock(spec=dist.ProcessGroup))
        object.__setattr__(linear, "_tp_mesh", MagicMock(get_local_rank=lambda: 0))
        object.__setattr__(linear, "_tp_degree", 2)
        tp_mod._install_split_forward(linear, None)
        with (
            patch(
                "agilerl.distributed.tensor_parallel._slice_last_dim",
                side_effect=lambda tensor, *_rest: tensor,
            ),
            patch(
                "agilerl.distributed.tensor_parallel._call_local_forward",
                return_value="nope",
            ),
        ):
            with pytest.raises(
                TypeError, match="row-parallel linear expected a tensor"
            ):
                linear(torch.ones(2, 2))

    def test_vision_head_count_raises_without_heads(self):
        with pytest.raises(ValueError, match="has no num_heads"):
            tp_mod._vision_head_count(nn.Linear(2, 2))

    def test_shard_vision_attention_rejects_query_rows_that_do_not_divide(self):
        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(4, 6)

        attn = Attn()
        attn.__dict__["num_heads"] = 4
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            tp_mod._shard_vision_attention(attn, MagicMock())

    def test_shard_latent_returns_false_when_projections_are_not_linear(self):
        class Latent(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1_latent_proj = nn.Identity()
                self.fc2_latent_proj = nn.Identity()

        assert tp_mod._shard_latent(Latent(), MagicMock()) is False

    def test_weight_is_embedding_finds_a_tied_embedding(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 4)
                self.lm_head = nn.Linear(4, 8, bias=False)
                self.lm_head.weight = self.embed.weight

        model = Model()

        assert tp_mod._weight_is_embedding(model, model.lm_head.weight) is True

    def test_iter_lm_heads_finds_a_nested_linear_head(self):
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 8)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = Inner()

        found = tp_mod._iter_lm_heads(Model())

        assert [name for name, _module in found] == ["decoder.lm_head"]

    def test_lm_head_is_tied_when_the_parent_config_ties_embeddings(self):
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 8)
                self.config = SimpleNamespace(tie_word_embeddings=True)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = Parent()
                self.config = SimpleNamespace(tie_word_embeddings=False)

        model = Model()

        assert tp_mod._lm_head_is_tied(model, "decoder.lm_head", model.decoder.lm_head)

    def test_shard_lm_heads_installs_gather_on_an_already_sharded_head(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 8)
                self.config = SimpleNamespace(tie_word_embeddings=False)

        model = Model()
        mesh = MagicMock()
        mesh.size.return_value = 2
        mesh.get_group.return_value = object()
        with patch("agilerl.distributed.tensor_parallel.DTensor", nn.Parameter):
            count = tp_mod._shard_lm_heads(model, mesh)

        assert count == 1
        assert getattr(model.lm_head, "_agilerl_dense_tp", False)

    def test_apply_tensor_parallel_is_a_no_op_without_a_mesh(self):
        assert apply_tensor_parallel(nn.Linear(2, 2), None) == 0

    def test_apply_tensor_parallel_is_a_no_op_when_tp_is_one(self):
        mesh = MagicMock()
        mesh.size.return_value = 1

        assert apply_tensor_parallel(nn.Linear(2, 2), mesh) == 0
