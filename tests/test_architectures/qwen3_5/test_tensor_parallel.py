# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Qwen3.5 blocks on a gloo mesh."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard
from transformers import AutoModel
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5VisionAttention,
    Qwen3_5VisionBlock,
    Qwen3_5VisionMLP,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionPatchMerger,
)

from agilerl.architectures.qwen3_5 import tensor_parallel as qwen3_5_tp
from agilerl.architectures.qwen3_5.tensor_parallel import (
    GDN_PERMUTED_PARAMS,
    VISION_PERMUTED_PARAMS,
    _gdn_rank_ranges,
    _vision_rank_ranges,
    apply_qwen3_5_tensor_parallel,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import (
    FSDPConfig,
    ParallelMesh,
    build_parallel_mesh,
    fsdp_blocks,
)
from agilerl.distributed.context_parallel.sequence import (
    shard_for_cp,
)
from agilerl.distributed.fsdp import (
    materialize_fsdp2_from_cpu_state,
)
from agilerl.distributed.mesh import MeshAxisName
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.distributed.tensor_parallel import is_lora_b_weight, sum_tp_partial_grads
from tests.parallel_helpers import (
    assert_exact,
    assert_lora,
    assert_same_state,
    broadcast_module,
    broadcast_tensor,
    lora_runtime,
    on_tp_mesh,
    plain,
    publish_varlen_offsets,
    with_lora,
    world_held_numel,
)
from tests.rank_launch import run_ranks
from tests.test_architectures.qwen3_5.parallel_helpers import (
    BATCH,
    DECAY_PARAMS,
    HEAD_DIM,
    HIDDEN,
    SEQ,
    assert_sequence_roundtrip,
    cp_rank_loss,
    fake_fla_kernels,
    forward_blocks,
)

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "qkv",
    "proj",
    "linear_fc1",
    "linear_fc2",
]
BLOCK_CLASSES = {
    Qwen3_5GatedDeltaNet,
    Qwen3_5Attention,
    Qwen3_5MLP,
    Qwen3_5VisionAttention,
    Qwen3_5VisionMLP,
}
NORM_CLASSES = {Qwen3_5RMSNorm, Qwen3_5RMSNormGated, nn.LayerNorm}


def _text_config(linear_key_heads: int = 2) -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        linear_num_key_heads=linear_key_heads,
        # One row per value head. A 2-row local GEMM does not match a 4-row GEMM.
        linear_num_value_heads=8,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=4,
        vocab_size=32,
        layer_types=["linear_attention", "full_attention"],
    )
    config._attn_implementation = "eager"
    return config


def _vision_config() -> Qwen3_5VisionConfig:
    config = Qwen3_5VisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_heads=4,
        depth=1,
        out_hidden_size=HIDDEN,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    return config


class QwenMergerBlock(nn.Module):
    """FSDP unit around the vision merger."""

    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.merger = Qwen3_5VisionPatchMerger(config, use_postshuffle_norm=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.merger(hidden)


class QwenBlockStack(nn.Module):
    """One linear-attention layer, one full-attention layer, and one vision block."""

    _no_split_modules = (
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
        "QwenMergerBlock",
    )

    def __init__(self, linear_key_heads: int = 2) -> None:
        super().__init__()
        text = _text_config(linear_key_heads)
        vision = _vision_config()
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(text, 0), Qwen3_5DecoderLayer(text, 1)]
        )
        self.vision = Qwen3_5VisionBlock(vision)
        self.merger = QwenMergerBlock(vision)
        self.config = SimpleNamespace(model_type="qwen3_5")


def _reference_params(reference: nn.Module) -> dict[str, nn.Parameter]:
    return {plain(name): param for name, param in reference.named_parameters()}


def _permuted_full(
    model: nn.Module, fqn: str, full: torch.Tensor, tp: int
) -> torch.Tensor:
    """``full`` in the per-rank row order its sharded parameter stores."""
    owner: nn.Module | None = None
    owner_name = ""
    for name, module in model.named_modules():
        if not isinstance(module, (Qwen3_5GatedDeltaNet, Qwen3_5VisionAttention)):
            continue
        candidate = plain(name)
        if fqn != candidate and not fqn.startswith(candidate + "."):
            continue
        if owner is None or len(candidate) > len(owner_name):
            owner = module
            owner_name = candidate
    if owner is None:
        return full
    relative = fqn[len(owner_name) + 1 :] if owner_name else fqn
    if isinstance(owner, Qwen3_5GatedDeltaNet) and (
        relative in GDN_PERMUTED_PARAMS or is_lora_b_weight(relative, "in_proj_qkv")
    ):
        ranges = [
            _gdn_rank_ranges(
                int(owner.num_k_heads),
                int(owner.head_k_dim),
                int(owner.num_v_heads),
                int(owner.head_v_dim),
                tp,
                rank,
            )
            for rank in range(tp)
        ]
    elif isinstance(owner, Qwen3_5VisionAttention) and (
        relative in VISION_PERMUTED_PARAMS or is_lora_b_weight(relative, "qkv")
    ):
        ranges = [
            _vision_rank_ranges(int(owner.num_heads), int(owner.head_dim), tp, rank)
            for rank in range(tp)
        ]
    else:
        return full
    return torch.cat(
        [full[start:stop] for spans in ranges for start, stop in spans], dim=0
    )


def _assert_grads(
    model: nn.Module, reference: nn.Module, cp_group: dist.ProcessGroup | None
) -> None:
    """Pre-sync grads match the reference; FSDP grads also sum over ``cp_group``."""
    reference_params = _reference_params(reference)
    errors: list[str] = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None:
                errors.append(f"{name} has no grad")
                continue
            full = reference_params[plain(name)].grad
            if isinstance(param, DTensor):
                mesh = param.device_mesh
                if on_tp_mesh(param):
                    tp = int(mesh.size(mesh.mesh_dim_names.index(MeshAxisName.TP)))
                    full = _permuted_full(model, plain(name), full, tp)
                if cp_group is not None:
                    full = full.clone()
                    dist.all_reduce(full, group=cp_group)
                local = distribute_tensor(
                    full, mesh, param.placements, src_data_rank=None
                ).to_local()
                got = param.grad.to_local()
            else:
                local = full
                got = param.grad
            if plain(name).endswith(DECAY_PARAMS) and local.numel() > 0:
                scale = max(local.abs().max().item(), 1.0)
                if not torch.allclose(got, local, atol=1e-4 * scale, rtol=1e-4):
                    diff = (got - local).abs().max().item()
                    errors.append(f"{name}: max abs diff {diff}")
                continue
            try:
                assert_exact(got, local, name)
            except AssertionError as exc:
                errors.append(str(exc))
    if errors:
        raise AssertionError("\n".join(errors))


def _assert_tp_placements(
    model: nn.Module, reference: nn.Module, tp: int, fsdp_size: int
) -> None:
    reference_params = _reference_params(reference)
    saw_tp = False
    for name, param in model.named_parameters():
        full = reference_params[plain(name)]
        if not isinstance(param, DTensor):
            continue
        if on_tp_mesh(param):
            saw_tp = True
            assert param.device_mesh.mesh_dim_names[-1] == MeshAxisName.TP, name
            assert param.device_mesh.size() == tp * fsdp_size, name
            copies = tp if param.placements[-1].is_replicate() else 1
            assert world_held_numel(param) == full.numel() * copies, name
        else:
            assert param.device_mesh.size() == fsdp_size, name
            assert world_held_numel(param) == full.numel() * tp, name
    assert saw_tp


def _assert_permuted_lora_b_sharded(model: nn.Module) -> None:
    """GDN ``in_proj_qkv`` and vision ``qkv`` LoRA B are ``Shard(0)`` over TP."""
    lora_b = [
        (name, param)
        for name, param in model.named_parameters()
        if ".in_proj_qkv.lora_B." in name or ".attn.qkv.lora_B." in name
    ]
    assert len(lora_b) == 2
    for name, param in lora_b:
        assert isinstance(param, DTensor), name
        assert on_tp_mesh(param), name
        assert param.placements[-1] == Shard(0), name


def _liger_mlp_class() -> type[nn.Module]:
    # liger-kernel is not installed on the CPU gloo runner.
    try:
        from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP
    except ImportError:

        class LigerQwen3MoeSwiGLUMLP(nn.Module):
            pass

        return LigerQwen3MoeSwiGLUMLP
    return LigerQwen3MoeSwiGLUMLP


def _reported_mlp_name(self: nn.Module) -> str:
    return qwen3_5_tp.Qwen3_5MLP.__name__


def _point_plan_at_liger_mlp(stack: nn.Module) -> None:
    """Point the plan at Liger's MLP class, then keep the HF forward."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    liger_mlp = _liger_mlp_class()
    modeling.Qwen3_5MLP = liger_mlp
    qwen3_5_tp.Qwen3_5MLP = liger_mlp
    for module in stack.modules():
        if type(module).__name__ != "Qwen3_5MLP":
            continue
        module.__dict__["forward"] = Qwen3_5MLP.forward.__get__(
            module, module.__class__
        )
        module.__dict__["_get_name"] = _reported_mlp_name.__get__(
            module, module.__class__
        )


def _assert_language_mlp_on_tp_mesh(
    model: nn.Module, reference: nn.Module, tp: int
) -> None:
    if tp <= 1:
        return
    reference_params = _reference_params(reference)
    weights = [
        (name, param)
        for name, param in model.named_parameters()
        if plain(name).endswith(
            (
                "mlp.gate_proj.base_layer.weight",
                "mlp.up_proj.base_layer.weight",
                "mlp.down_proj.base_layer.weight",
            )
        )
    ]
    assert len(weights) == 6
    for name, param in weights:
        assert isinstance(param, DTensor), name
        assert on_tp_mesh(param), name
        assert isinstance(param.placements[-1], Shard), name
        assert world_held_numel(param) == reference_params[plain(name)].numel(), name


def _worker(rank: int, world_size: int, cp: int, tp: int) -> None:
    torch.manual_seed(0)
    if cp > 1:
        fake_fla_kernels()
    stack = with_lora(QwenBlockStack(), LORA_TARGETS)
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    _point_plan_at_liger_mlp(stack)
    assert qwen3_5_tp.Qwen3_5MLP is not Qwen3_5MLP
    mesh = build_parallel_mesh(world_size=world_size, tp=tp, cp=cp, device_type="cpu")
    if tp == 1 and cp == 1:
        assert mesh is None
        # fully_shard without a mesh uses the accelerator, not this gloo group.
        fsdp_mesh = ParallelMesh(
            dp_shard=init_device_mesh(
                "cpu", (world_size,), mesh_dim_names=("dp_shard",)
            )
        )
    else:
        assert mesh is not None
        assert mesh.dp_shard.size() == world_size // tp
        fsdp_mesh = mesh
    gdn_hidden = broadcast_tensor(torch.randn(BATCH, SEQ, HIDDEN))
    attn_hidden = broadcast_tensor(torch.randn(BATCH, SEQ, HIDDEN))
    vision_hidden = broadcast_tensor(torch.randn(SEQ, HIDDEN))
    cp_rank = 0 if mesh is None or mesh.cp is None else int(mesh.cp.get_local_rank())
    gdn_hidden = shard_for_cp(gdn_hidden, cp_rank, cp)
    gdn_probe = broadcast_tensor(torch.randn_like(gdn_hidden))
    attn_probe = broadcast_tensor(torch.randn_like(attn_hidden))
    vision_probe = broadcast_tensor(torch.randn_like(vision_hidden))
    merger_probe = broadcast_tensor(torch.randn_like(vision_hidden))
    config = FSDPConfig(
        reshard_after_forward=True,
        tp=tp,
        cp=cp,
        param_dtype="float32",
        reduce_dtype="float32",
        routed_expert_chunk_mib=64,
        checkpoint_every_n_blocks=1,
    )
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack,
            "cpu",
            config,
            parallel_mesh=fsdp_mesh,
            gradient_checkpointing=True,
        )
    if tp > 1 and cp > 1:
        with torch.no_grad():
            assert_sequence_roundtrip(stack, reference, cp, mesh)
    if cp > 1:
        publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)

    gdn_in = gdn_hidden.detach().requires_grad_(True)
    attn_in = attn_hidden.detach().requires_grad_(True)
    vision_in = vision_hidden.detach().requires_grad_(True)
    outputs = forward_blocks(stack, gdn_in, attn_in, vision_in)
    if tp > 1:
        _assert_tp_placements(stack, reference, tp, world_size // tp)
        _assert_language_mlp_on_tp_mesh(stack, reference, tp)
        _assert_permuted_lora_b_sharded(stack)
    ref_gdn = gdn_hidden.detach().requires_grad_(True)
    ref_attn = attn_hidden.detach().requires_grad_(True)
    ref_vision = vision_hidden.detach().requires_grad_(True)
    reference_outputs = forward_blocks(reference, ref_gdn, ref_attn, ref_vision)
    labels = ("gdn", "attention", "vision", "merger")
    probes = (gdn_probe, attn_probe, vision_probe, merger_probe)
    loss = torch.zeros(())
    reference_loss = torch.zeros(())
    for label, output, expected, probe in zip(
        labels, outputs, reference_outputs, probes, strict=True
    ):
        assert_exact(output, expected, label)
        loss = loss + (output * probe).sum()
        reference_loss = reference_loss + (expected * probe).sum()
    loss.backward()
    reference_loss.backward()
    if tp > 1:
        sum_tp_partial_grads(stack, mesh.tp.get_group())

    _assert_grads(stack, reference, None if cp == 1 else mesh.cp.get_group())
    assert_lora(stack, reference)


def _meta_load_worker(
    rank: int, world_size: int, cp: int, tp: int, checkpoint_dir: str
) -> None:
    """A meta stack loads each rank's dp_shard x cp x tp piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    fake_fla_kernels()
    dense = QwenBlockStack()
    if rank == 0:
        save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = QwenBlockStack()
    stack.config = Qwen3_5Config(name_or_path=checkpoint_dir)
    mesh = build_parallel_mesh(world_size=world_size, tp=tp, cp=cp, device_type="cpu")
    config = FSDPConfig(
        reshard_after_forward=True,
        tp=tp,
        cp=cp,
        param_dtype="float32",
        routed_expert_chunk_mib=64,
    )

    # Act
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(stack, "cpu", config, parallel_mesh=mesh)

    # Assert
    expected = dict(dense.named_parameters())
    strided = False
    for name, param in stack.named_parameters():
        assert isinstance(param, DTensor), name
        target = expected[plain(name)]
        if on_tp_mesh(param):
            target = _permuted_full(stack, plain(name), target, tp)
            strided = strided or any(
                isinstance(placement, _StridedShard) for placement in param.placements
            )
        assert_exact(param.full_tensor(), target, name)
    assert strided


def _resume_worker(
    rank: int, world_size: int, save_tp: int, load_tp: int, checkpoint_dir: str
) -> None:
    """A LoRA checkpoint saved at one TP degree resumes unchanged at another."""
    # Arrange
    cp = 2
    torch.manual_seed(0)
    fake_fla_kernels()
    dense = with_lora(QwenBlockStack(linear_key_heads=4), LORA_TARGETS)
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
    saved, runtime, optimizer = lora_runtime(dense, world_size, cp, save_tp, 1)
    mesh = runtime.parallel_mesh
    assert mesh is not None
    assert mesh.cp is not None
    cp_rank = int(mesh.cp.get_local_rank())
    loss = cp_rank_loss(saved, rank // (cp * save_tp), cp_rank, cp)
    runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=saved, max_grad_norm=1e6
    )
    model_state = runtime.export_model_state(saved)
    optimizer_state = runtime.export_optimizer_state(saved, optimizer)
    if rank == 0:
        torch.save(
            {"model": model_state, "optimizer": optimizer_state},
            f"{checkpoint_dir}/checkpoint.pt",
        )
    dist.barrier()
    checkpoint = torch.load(f"{checkpoint_dir}/checkpoint.pt", weights_only=False)
    resumed, resumed_runtime, resumed_optimizer = lora_runtime(
        dense, world_size, cp, load_tp, 1
    )

    # Act
    resumed_runtime.import_model_state(resumed, checkpoint["model"], strict=True)
    resumed_runtime.import_optimizer_state(
        resumed, resumed_optimizer, checkpoint["optimizer"]
    )
    reexported_model = resumed_runtime.export_model_state(resumed)
    reexported_optimizer = resumed_runtime.export_optimizer_state(
        resumed, resumed_optimizer
    )

    # Assert
    if rank != 0:
        return
    assert checkpoint["optimizer"]["state"]
    for key, value in checkpoint["model"].items():
        if "lora_" not in key:
            assert_exact(value, natural[key], key)
    assert_same_state(reexported_model, checkpoint["model"], "model")
    assert_same_state(reexported_optimizer, checkpoint["optimizer"], "optimizer")


def _step_worker(rank: int, world_size: int, cp: int, tp: int) -> None:
    """One FSDPRuntime SGD step on dp x cp x tp matches the mean data-shard step."""
    # Arrange
    torch.manual_seed(0)
    fake_fla_kernels()
    stack = with_lora(QwenBlockStack(), LORA_TARGETS)
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    dp = world_size // (cp * tp)
    mesh = build_parallel_mesh(world_size=world_size, tp=tp, cp=cp, device_type="cpu")
    assert mesh is not None
    assert mesh.cp is not None
    config = FSDPConfig(
        reshard_after_forward=True,
        tp=tp,
        cp=cp,
        param_dtype="float32",
        reduce_dtype="float32",
        routed_expert_chunk_mib=64,
        checkpoint_every_n_blocks=1,
    )
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
    reference_loss = sum(
        cp_rank_loss(reference, dp_rank, cp_rank, cp)
        for dp_rank in range(dp)
        for cp_rank in range(cp)
    )
    (reference_loss / dp).backward()
    expected = {
        plain(name): param.detach() - param.grad
        for name, param in reference.named_parameters()
        if param.grad is not None
    }
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    by_mesh: dict[Any, list[nn.Parameter]] = {}
    for param in stack.parameters():
        if param.requires_grad:
            key = param.device_mesh if isinstance(param, DTensor) else None
            by_mesh.setdefault(key, []).append(param)
    sgd = torch.optim.SGD([{"params": group} for group in by_mesh.values()], lr=1.0)
    optimizer = SimpleNamespace(
        _single_optimizer=lambda: sgd, step=sgd.step, zero_grad=sgd.zero_grad
    )
    loss = cp_rank_loss(stack, rank // (cp * tp), int(mesh.cp.get_local_rank()), cp)

    # Act
    step = runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=stack, max_grad_norm=1e6
    )

    # Assert
    assert step is not None
    assert len(expected) == sum(1 for _ in stack.parameters())
    errors: list[str] = []
    with torch.no_grad():
        for name, param in stack.named_parameters():
            target = expected[plain(name)]
            value = param
            if isinstance(param, DTensor):
                value = param.full_tensor()
                if on_tp_mesh(param):
                    target = _permuted_full(stack, plain(name), target, tp)
            if plain(name).endswith(DECAY_PARAMS):
                scale = max(target.abs().max().item(), 1.0)
                if not torch.allclose(value, target, atol=1e-4 * scale, rtol=1e-4):
                    errors.append(
                        f"{name}: max abs diff {(value - target).abs().max()}"
                    )
                continue
            try:
                assert_exact(value, target, name)
            except AssertionError as exc:
                errors.append(str(exc))
    if errors:
        raise AssertionError("\n".join(errors))


class TestQwen35TensorParallelPlan:
    def test_qkv_rows_partition_the_projection(self) -> None:
        covered: list[int] = []
        for rank in range(2):
            covered.extend(
                index
                for start, stop in _gdn_rank_ranges(2, 4, 4, 4, 2, rank)
                for index in range(start, stop)
            )

        assert sorted(covered) == list(range(2 * 4 * 2 + 4 * 4))

    def test_vision_qkv_rows_partition_the_projection(self) -> None:
        covered: list[int] = []
        for rank in range(2):
            covered.extend(
                index
                for start, stop in _vision_rank_ranges(4, 4, 2, rank)
                for index in range(start, stop)
            )

        assert sorted(covered) == list(range(3 * 16))

    def test_checkpoint_block_classes_are_the_plan(self) -> None:
        config = Qwen3_5Config.from_dict(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.8-27B"].config
        )

        with torch.device("meta"):
            model = AutoModel.from_config(config)

        found: set[type] = set()
        for module in model.modules():
            if not isinstance(module, (Qwen3_5DecoderLayer, Qwen3_5VisionBlock)):
                continue
            for child in module.children():
                if type(child) in NORM_CLASSES:
                    continue
                assert type(child) in BLOCK_CLASSES
                found.add(type(child))
        assert found == BLOCK_CLASSES
        assert any(
            isinstance(module, Qwen3_5VisionPatchMerger) for module in model.modules()
        )
        assert any(
            isinstance(module, Qwen3_5VisionPatchEmbed) for module in model.modules()
        )

    def test_tiny_stack_runs(self) -> None:
        stack = QwenBlockStack()
        hidden = torch.randn(BATCH, SEQ, HIDDEN)
        vision = torch.randn(SEQ, HIDDEN)

        gdn_out, attn_out, vision_out, merger_out = forward_blocks(
            stack, hidden, hidden, vision
        )

        assert gdn_out.shape == hidden.shape
        assert attn_out.shape == hidden.shape
        assert vision_out.shape == vision.shape
        assert merger_out.shape == vision.shape

    def test_heads_that_do_not_divide_tp_raise(self) -> None:
        mesh = MagicMock()
        mesh.size.return_value = 3

        with pytest.raises(ValueError, match="linear_num_key_heads"):
            apply_qwen3_5_tensor_parallel(QwenBlockStack(), mesh)

    def test_vision_hidden_that_does_not_divide_heads_raises(self) -> None:
        attn = QwenBlockStack().vision.attn
        attn.dim = attn.num_heads * HEAD_DIM + 1
        mesh = MagicMock()
        mesh.size.return_value = 2

        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            qwen3_5_tp._shard_vision_attention(attn, mesh)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "world"),
        [(1, 1, 2), (1, 2, 4), (2, 1, 4), (2, 2, 8)],
    )
    def test_forward_and_grads_match_unsharded(
        self, cp: int, tp: int, world: int
    ) -> None:
        run_ranks(_worker, world, cp, tp, timeout=180)

    @pytest.mark.multi_rank
    def test_meta_load_fills_dp_cp_tp_shards_in_permuted_row_order(
        self, tmp_path: Path
    ) -> None:
        run_ranks(_meta_load_worker, 8, 2, 2, str(tmp_path), timeout=180)

    @pytest.mark.multi_rank
    def test_dp_cp_tp_step_matches_the_mean_data_shard_step(self) -> None:
        run_ranks(_step_worker, 8, 2, 2, timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(("save_tp", "load_tp"), [(2, 4), (4, 2)])
    def test_checkpoint_resumes_unchanged_at_another_tp_degree(
        self, save_tp: int, load_tp: int, tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save_tp, load_tp, str(tmp_path), timeout=300)
