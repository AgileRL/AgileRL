# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor and expert parallel GLM-5.3 blocks on a gloo mesh."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from transformers import AutoModel, AutoModelForImageTextToText
from transformers.models.glm5_next import modeling_glm5_next
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextRMSNorm,
    Glm5NextTextAttention,
    Glm5NextTextDecoderLayer,
    Glm5NextTextExperts,
    Glm5NextTextForgetGate,
    Glm5NextTextHyperConnection,
    Glm5NextTextIndexer,
    Glm5NextTextLinearAttention,
    Glm5NextTextMLP,
    Glm5NextTextRMSNorm,
    Glm5NextTextTopkRouter,
    Glm5NextVisionAttention,
    Glm5NextVisionMLP,
    Glm5NextVisionModel,
    Glm5NextVisionPatchEmbed,
    Glm5NextVisionPatchMerger,
)

from agilerl.architectures.glm5_next.tensor_parallel import (
    ATTENTION_CLASS,
    GLM5_NEXT_TENSOR_PARALLEL_PLAN,
    LINEAR_ATTENTION_CLASS,
    MLP_CLASS,
    apply_glm5_next_tensor_parallel,
)
from agilerl.arena.models.model_info import (
    LORA_EXCLUDED_MODULES,
    SUPPORTED_MODEL_INFO,
)
from agilerl.distributed import fsdp_blocks
from agilerl.distributed.fsdp import materialize_fsdp2_from_cpu_state
from agilerl.distributed.mesh import MeshAxisName
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.lora.moe import upgrade_moe_param_wrappers
from tests.parallel_helpers import (
    assert_exact,
    assert_lora,
    assert_resharded,
    assert_same_state,
    broadcast_module,
    fsdp_config,
    lora_runtime,
    on_ep_mesh,
    on_tp_mesh,
    parallel_mesh,
    plain,
    sgd,
)
from tests.rank_launch import run_ranks

HUB_ID = "zai-org/GLM-5.3-Flash"
HIDDEN = 16
HC_MULT = 4
BATCH = 2
# Four k-pools of two tokens, so the DSA indexer keeps two plus the tail.
SEQ = 8
TP_BLOCK_CLASSES = {
    Glm5NextTextLinearAttention,
    Glm5NextTextAttention,
    Glm5NextTextMLP,
}
VISION_TP_BLOCK_CLASSES = {
    Glm5NextVisionAttention,
    Glm5NextVisionMLP,
    Glm5NextVisionPatchMerger,
}
EP_BLOCK_CLASSES = {Glm5NextTextExperts}
# The untied lm_head is sharded by the generic head shard, not the family plan.
HEAD_CLASSES = {Glm5NextForConditionalGeneration}
# Norms and embeddings stay whole; the router and mHC mixers are tiny and read
# the whole residual stream. The vision patch embed and spatial-merge conv are
# small and run outside the blocks on the whole patch stream.
SKIPPED_CLASSES = {
    Glm5NextTextRMSNorm,
    nn.Embedding,
    Glm5NextTextTopkRouter,
    Glm5NextTextHyperConnection,
    Glm5NextRMSNorm,
    Glm5NextVisionPatchEmbed,
    nn.Conv2d,
}
CHECKPOINT_PLANNED_CLASSES = {
    "vl": TP_BLOCK_CLASSES | VISION_TP_BLOCK_CLASSES | EP_BLOCK_CLASSES | HEAD_CLASSES,
    "text": TP_BLOCK_CLASSES | EP_BLOCK_CLASSES,
}
# KDA mixers, DSA attentions, and 3 dense plus 42 shared-expert MLPs.
TEXT_TP_BLOCKS = {LINEAR_ATTENTION_CLASS: 34, ATTENTION_CLASS: 11, MLP_CLASS: 45}
CHECKPOINT_TP_BLOCKS = {
    "text": TEXT_TP_BLOCKS,
    # Plus 24 vision blocks and the patch merger.
    "vl": {
        **TEXT_TP_BLOCKS,
        "Glm5NextVisionAttention": 24,
        "Glm5NextVisionMLP": 24,
        "Glm5NextVisionPatchMerger": 1,
    },
}
# One weight of each class the TP plan shards, by stack FQN suffix.
TP_PROBE_WEIGHTS = (
    "layers.0.self_attn.q_proj.base_layer.weight",
    "layers.0.self_attn.conv1d.weight",
    "layers.0.self_attn.forget_gate.A_log",
    "layers.1.self_attn.q_b_proj.base_layer.weight",
    "layers.1.self_attn.kv_b_proj.base_layer.weight",
    "layers.1.mlp.up_proj.base_layer.weight",
    "layers.2.mlp.shared_experts.up_proj.base_layer.weight",
)
# A 4x4 and a 2x4 patch grid, so vision attention runs two images.
GRID_THW = ((1, 4, 4), (1, 2, 4))
SPATIAL_MERGE = 2
PATCH_PIXELS = 3 * 2 * 2
VISION_PROBE_WEIGHTS = (
    "visual.blocks.0.attn.qkv.base_layer.weight",
    "visual.blocks.0.attn.qkv.base_layer.bias",
    "visual.blocks.0.attn.proj.weight",
    "visual.blocks.1.mlp.up_proj.base_layer.weight",
    "visual.merger.up_proj.base_layer.weight",
)
VISION_ROW_ORDERS = [
    f"visual.blocks.{block}.attn.qkv.{name}"
    for block in range(2)
    for name in ("bias", "weight")
]


def _config(linear_heads: int) -> Glm5NextTextConfig:
    config = Glm5NextTextConfig(
        vocab_size=32,
        hidden_size=HIDDEN,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        layer_types=["linear_attention", "indexed_attention", "indexed_attention"],
        mlp_layer_types=["sparse", "dense", "sparse"],
        indexer_types=["full", "full", "shared"],
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        index_kpool=2,
        linear_num_heads=linear_heads,
        linear_head_dim=4,
        linear_conv_kernel_dim=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        hc_mult=HC_MULT,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    return config


class Glm5NextBlockStack(nn.Module):
    """A KDA layer with MoE, a DSA layer with a dense MLP, and a shared-index DSA layer with MoE."""

    _no_split_modules = ("Glm5NextTextDecoderLayer",)

    def __init__(self, linear_heads: int = 4) -> None:
        super().__init__()
        config = _config(linear_heads)
        self.layers = nn.ModuleList(
            [Glm5NextTextDecoderLayer(config, i) for i in range(3)]
        )
        self.config = SimpleNamespace(model_type="glm5_next_text")
        # Gates, mHC mixers and experts are torch.empty and the router is zeros
        # outside post_init. Small scales keep the three-layer chain's gradient
        # norm far below the clip.
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, Glm5NextTextForgetGate):
                    module.A_log.uniform_(-1.0, 1.0)
                    module.dt_bias.normal_(std=0.5)
                elif isinstance(module, Glm5NextTextHyperConnection):
                    module.fn.normal_(std=0.1)
                    module.base.normal_(std=0.5)
                    module.scale.normal_()
                elif isinstance(module, Glm5NextTextExperts):
                    module.gate_up_proj.normal_(std=0.1)
                    module.down_proj.normal_(std=0.1)
                elif isinstance(module, Glm5NextTextTopkRouter):
                    module.weight.normal_()

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        topk = None
        for layer in self.layers:
            hidden, topk = layer(hidden, attention_mask=mask, prev_topk_indices=topk)
        return hidden


def _vision_config(num_heads: int) -> Glm5NextVisionConfig:
    config = Glm5NextVisionConfig(
        depth=2,
        hidden_size=HIDDEN,
        num_heads=num_heads,
        intermediate_size=32,
        out_hidden_size=HIDDEN,
        projection_intermediate_size=32,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=SPATIAL_MERGE,
        attention_bias=True,
    )
    config._attn_implementation = "sdpa"
    return config


class Glm5NextVisionStack(nn.Module):
    """A two-block GLM-5.3 vision tower with its spatial-merge conv and patch merger."""

    _no_split_modules = ("Glm5NextVisionBlock", "Glm5NextVisionPatchMerger")

    def __init__(self, num_heads: int = 4) -> None:
        super().__init__()
        self.visual = Glm5NextVisionModel(_vision_config(num_heads))
        self.config = SimpleNamespace(model_type="glm5_next_vision")
        # post_init zeroes biases; nonzero ones reach the output through the
        # row-parallel bias path.
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith("bias"):
                    param.normal_(std=0.1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.visual(pixels, grid_thw=torch.tensor(GRID_THW)).pooler_output


def _with_tower_lora(stack: nn.Module) -> nn.Module:
    inject_adapter_in_model(
        LoraConfig(
            r=2, target_modules=sorted(SUPPORTED_MODEL_INFO[HUB_ID].tower_lora_modules)
        ),
        stack,
    )
    stack.requires_grad_(True)
    with torch.no_grad():
        for name, param in stack.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.1)
    return stack


def _vision_data(dp_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch pixels and a merged-token output probe for one data-parallel rank."""
    generator = torch.Generator().manual_seed(300 + dp_rank)
    patches = sum(t * h * w for t, h, w in GRID_THW)
    tokens = patches // SPATIAL_MERGE**2
    return (
        torch.randn(patches, PATCH_PIXELS, generator=generator),
        torch.randn(tokens, HIDDEN, generator=generator),
    )


def _vision_loss(stack: nn.Module, dp_rank: int) -> torch.Tensor:
    pixels, probe = _vision_data(dp_rank)
    return (stack(pixels) * probe).sum()


def _with_moe_lora(module: nn.Module) -> nn.Module:
    info = SUPPORTED_MODEL_INFO[HUB_ID]
    inject_adapter_in_model(
        LoraConfig(
            r=2,
            target_modules=sorted(info.default_lora_modules),
            exclude_modules=sorted(LORA_EXCLUDED_MODULES["glm5_next_text"]),
            target_parameters=sorted(info.parameters),
        ),
        module,
    )
    assert upgrade_moe_param_wrappers(module) == 2
    module.requires_grad_(True)
    # The indexer runs under no_grad, so it gets no gradient.
    for child in module.modules():
        if isinstance(child, Glm5NextTextIndexer):
            child.requires_grad_(False)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.1)
    return module


def _apply_liger_swaps(stack: nn.Module) -> None:
    """Swap the class Liger reassigns for glm5_next and set MLP forwards per instance."""
    modeling_glm5_next.Glm5NextTextRMSNorm = type("LigerRMSNorm", (nn.Module,), {})
    # Liger's SwiGLU kernel needs a GPU, so the instance forward is the stock one.
    for module in stack.modules():
        if isinstance(module, Glm5NextTextMLP):
            module.forward = MethodType(Glm5NextTextMLP.forward, module)


def _mask() -> torch.Tensor:
    return torch.ones(BATCH, SEQ, dtype=torch.bool)


def _shard_data(dp_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """mHC hidden streams and an output probe for one data-parallel rank."""
    generator = torch.Generator().manual_seed(200 + dp_rank)
    shape = (BATCH, SEQ, HC_MULT, HIDDEN)
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    )


def _rank_loss(stack: nn.Module, dp_rank: int) -> torch.Tensor:
    hidden, probe = _shard_data(dp_rank)
    return (stack(hidden, _mask()) * probe).sum()


def _assert_tp_blocks_sharded(
    model: nn.Module, tp: int, probes: tuple[str, ...]
) -> None:
    """Each probe weight holds 1/tp of its elements per TP rank."""
    params = {plain(name): param for name, param in model.named_parameters()}
    for suffix in probes:
        param = params[suffix]
        assert isinstance(param, DTensor), suffix
        assert on_tp_mesh(param), suffix
        assert isinstance(param.placements[-1], Shard), suffix
        tp_local = param.redistribute(
            param.device_mesh,
            [Replicate()] * (param.device_mesh.ndim - 1) + [param.placements[-1]],
        ).to_local()
        assert tp_local.numel() * tp == param.numel(), suffix


def _assert_experts_on_ep(model: nn.Module, ep: int) -> None:
    """Packed experts split their expert axis ``ep`` ways."""
    experts = [
        param
        for module in model.modules()
        if isinstance(module, Glm5NextTextExperts)
        for param in (module.gate_up_proj, module.down_proj)
    ]
    assert len(experts) == 4
    for param in experts:
        assert isinstance(param, DTensor)
        assert on_ep_mesh(param)
        index = param.device_mesh.mesh_dim_names.index(MeshAxisName.EP)
        assert param.placements[index] == Shard(0)
        assert int(param.device_mesh.size(index)) == ep


def _permuted_full(
    orders: dict[str, torch.Tensor], name: str, full: torch.Tensor
) -> torch.Tensor:
    """``full`` in the per-rank row order the plan stores for ``name``."""
    order = orders.get(name)
    return full if order is None else full.index_select(0, order)


def _assert_stepped(
    model: nn.Module, expected: dict[str, torch.Tensor], orders: dict[str, torch.Tensor]
) -> None:
    """Every gathered parameter equals its unsharded value after one SGD step."""
    assert len(expected) == sum(1 for _ in model.parameters())
    errors: list[str] = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            value = param.full_tensor()
            target = expected[plain(name)].reshape(value.shape)
            try:
                assert_exact(value, _permuted_full(orders, name, target), name)
            except AssertionError as exc:
                errors.append(str(exc))
    if errors:
        raise AssertionError("\n".join(errors))


def _stepped_reference(reference: nn.Module) -> dict[str, torch.Tensor]:
    """Reference parameters after an SGD step with ``lr=1``."""
    return {
        plain(name): param.detach() - param.grad
        if param.requires_grad
        else param.detach()
        for name, param in reference.named_parameters()
    }


def _worker(rank: int, world_size: int, tp: int, ep: int) -> None:
    """Forward, reshard, LoRA gather and one SGD step match the unsharded stack."""
    # Arrange
    torch.manual_seed(0)
    stack = _with_moe_lora(Glm5NextBlockStack())
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    _apply_liger_swaps(stack)
    dp = world_size // tp
    mesh = parallel_mesh(world_size, 1, tp, ep)
    config = fsdp_config(1, tp, ep)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    hidden, probe = _shard_data(rank // tp)
    with torch.no_grad():
        full_output = reference(hidden, _mask())
    reference_loss = sum(_rank_loss(reference, dp_rank) for dp_rank in range(dp))
    (reference_loss / dp).backward()
    expected = _stepped_reference(reference)
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    optimizer = sgd(stack, lr=1.0)

    # Act
    output = stack(hidden, _mask())
    assert_resharded(stack, reference, tp, world_size // tp)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp, TP_PROBE_WEIGHTS)
    if ep > 1:
        _assert_experts_on_ep(stack, ep)
    assert_lora(stack, reference)
    step = runtime.backward(
        (output * probe).sum(),
        optimizer,
        gradient_accumulation_steps=1,
        actor=stack,
        max_grad_norm=1e6,
    )

    # Assert
    assert_exact(output, full_output, "output")
    assert step is not None
    orders = GLM5_NEXT_TENSOR_PARALLEL_PLAN.row_orders(stack)
    assert bool(orders) == (tp > 1)
    _assert_stepped(stack, expected, orders)


def _vision_worker(rank: int, world_size: int, tp: int) -> None:
    """Vision forward, reshard, LoRA gather and one SGD step match the unsharded tower."""
    # Arrange
    torch.manual_seed(0)
    stack = _with_tower_lora(Glm5NextVisionStack())
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    dp = world_size // tp
    mesh = parallel_mesh(world_size, 1, tp, 1)
    config = fsdp_config(1, tp, 1)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    pixels, probe = _vision_data(rank // tp)
    with torch.no_grad():
        full_output = reference(pixels)
    reference_loss = sum(_vision_loss(reference, dp_rank) for dp_rank in range(dp))
    (reference_loss / dp).backward()
    expected = _stepped_reference(reference)
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    optimizer = sgd(stack, lr=1.0)

    # Act
    output = stack(pixels)
    assert_resharded(stack, reference, tp, dp)
    _assert_tp_blocks_sharded(stack, tp, VISION_PROBE_WEIGHTS)
    assert_lora(stack, reference)
    step = runtime.backward(
        (output * probe).sum(),
        optimizer,
        gradient_accumulation_steps=1,
        actor=stack,
        max_grad_norm=1e6,
    )

    # Assert
    assert_exact(output, full_output, "output")
    assert step is not None
    orders = GLM5_NEXT_TENSOR_PARALLEL_PLAN.row_orders(stack)
    assert sorted(plain(name) for name in orders) == [
        name.replace("qkv.", "qkv.base_layer.") for name in VISION_ROW_ORDERS
    ]
    _assert_stepped(stack, expected, orders)


def _meta_load_worker(
    rank: int, world_size: int, tp: int, ep: int, checkpoint_dir: str
) -> None:
    """A meta stack loads each rank's dp_shard x tp x ep piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    dense = Glm5NextBlockStack()
    if rank == 0:
        save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = Glm5NextBlockStack()
    stack.config = Glm5NextTextConfig(name_or_path=checkpoint_dir)
    mesh = parallel_mesh(world_size, 1, tp, ep)
    config = fsdp_config(1, tp, ep)

    # Act
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(stack, "cpu", config, parallel_mesh=mesh)

    # Assert
    expected = dict(dense.named_parameters())
    orders = GLM5_NEXT_TENSOR_PARALLEL_PLAN.row_orders(stack)
    assert sorted(orders) == ["layers.0.self_attn.conv1d.weight"]
    strided = False
    for name, param in stack.named_parameters():
        assert isinstance(param, DTensor), name
        if on_tp_mesh(param):
            strided = strided or any(
                isinstance(placement, _StridedShard) for placement in param.placements
            )
        target = _permuted_full(orders, name, expected[plain(name)])
        assert_exact(param.full_tensor(), target, name)
    assert strided
    _assert_experts_on_ep(stack, ep)


def _vision_meta_load_worker(
    rank: int, world_size: int, tp: int, checkpoint_dir: str
) -> None:
    """A meta vision tower loads each rank's dp_shard x tp piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    dense = Glm5NextVisionStack()
    if rank == 0:
        save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    pixels, _probe = _vision_data(0)
    with torch.no_grad():
        full_output = dense(pixels)
    with torch.device("meta"):
        stack = Glm5NextVisionStack()
    stack.config = Glm5NextVisionConfig(name_or_path=checkpoint_dir)
    mesh = parallel_mesh(world_size, 1, tp, 1)
    config = fsdp_config(1, tp, 1)

    # Act
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(stack, "cpu", config, parallel_mesh=mesh)

    # Assert
    expected = dict(dense.named_parameters())
    orders = GLM5_NEXT_TENSOR_PARALLEL_PLAN.row_orders(stack)
    assert sorted(plain(name) for name in orders) == VISION_ROW_ORDERS
    strided = False
    for name, param in stack.named_parameters():
        assert isinstance(param, DTensor), name
        if on_tp_mesh(param):
            strided = strided or any(
                isinstance(placement, _StridedShard) for placement in param.placements
            )
        target = _permuted_full(orders, name, expected[plain(name)])
        assert_exact(param.full_tensor(), target, name)
    assert strided
    # The merger unit stays unsharded after a no-grad forward; read parameters first.
    with torch.no_grad():
        assert_exact(stack(pixels), full_output, "output")


def _resume_worker(
    rank: int,
    world_size: int,
    save_tp: int,
    load_tp: int,
    ep: int,
    checkpoint_dir: str,
) -> None:
    """A LoRA checkpoint saved at one TP degree resumes unchanged at another."""
    # Arrange
    torch.manual_seed(0)
    # FSDP x TP splits A_log and b_proj rows over all 8 ranks.
    dense = _with_moe_lora(Glm5NextBlockStack(linear_heads=8))
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    saved, runtime, optimizer = lora_runtime(dense, world_size, 1, save_tp, ep)
    runtime.backward(
        _rank_loss(saved, rank // save_tp),
        optimizer,
        gradient_accumulation_steps=1,
        actor=saved,
        max_grad_norm=1e6,
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
        dense, world_size, 1, load_tp, ep
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
    assert any("experts" in key and "lora_" in key for key in checkpoint["model"])
    for key, value in checkpoint["model"].items():
        if "lora_" not in key:
            assert_exact(value, natural[key], key)
    assert_same_state(reexported_model, checkpoint["model"], "model")
    assert_same_state(reexported_optimizer, checkpoint["optimizer"], "optimizer")


def _indivisible_heads_worker(rank: int, world_size: int) -> None:
    stack = Glm5NextBlockStack()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    with pytest.raises(ValueError, match=r"linear_num_heads \(4\) .* tp \(3\)"):
        apply_glm5_next_tensor_parallel(stack, mesh)


def _indivisible_vision_heads_worker(rank: int, world_size: int) -> None:
    stack = Glm5NextVisionStack()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    with pytest.raises(ValueError, match=r"vision num_heads \(4\) .* tp \(3\)"):
        apply_glm5_next_tensor_parallel(stack, mesh)


def _checkpoint_meta_model(kind: str) -> nn.Module:
    config = Glm5NextConfig.from_dict(SUPPORTED_MODEL_INFO[HUB_ID].config)
    with torch.device("meta"):
        if kind == "text":
            return AutoModel.from_config(config.text_config)
        return AutoModelForImageTextToText.from_config(config)


def _meta_checkpoint_shard_worker(rank: int, world_size: int, kind: str) -> None:
    model = _checkpoint_meta_model(kind)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    count = apply_glm5_next_tensor_parallel(model, mesh)

    expected = CHECKPOINT_TP_BLOCKS[kind]
    sharded = Counter(
        type(module).__name__
        for module in model.modules()
        if getattr(module, "_tp_degree", 1) == world_size
        and type(module).__name__ in expected
    )
    assert sharded == expected
    assert count == sum(expected.values())


@pytest.mark.parametrize("kind", ["vl", "text"])
class TestGlm5NextCheckpointCoverage:
    def test_every_parameter_is_planned_or_skipped(self, kind: str) -> None:
        # Arrange
        model = _checkpoint_meta_model(kind)
        modules = dict(model.named_modules())
        known = (
            TP_BLOCK_CLASSES
            | VISION_TP_BLOCK_CLASSES
            | EP_BLOCK_CLASSES
            | HEAD_CLASSES
            | SKIPPED_CLASSES
        )

        # Act
        owners: dict[str, type] = {}
        for name, _ in model.named_parameters():
            path = name.rsplit(".", 1)[0]
            while type(modules[path]) not in known and path:
                path = path.rpartition(".")[0]
            owners[name] = type(modules[path])

        # Assert
        unmatched = sorted(name for name, owner in owners.items() if owner not in known)
        assert unmatched == []
        planned = set(owners.values()) - SKIPPED_CLASSES
        assert planned == CHECKPOINT_PLANNED_CLASSES[kind]

    @pytest.mark.multi_rank
    def test_tp_plan_shards_every_checkpoint_block(self, kind: str) -> None:
        run_ranks(_meta_checkpoint_shard_worker, 2, kind, timeout=300)


class TestGlm5NextParallel:
    def test_tiny_stack_runs(self) -> None:
        stack = _with_moe_lora(Glm5NextBlockStack())
        hidden, _probe = _shard_data(0)

        output = stack(hidden, _mask())

        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()

    @pytest.mark.usefixtures("gloo_process_group")
    def test_reapplying_the_plan_leaves_shards_and_forwards_in_place(self) -> None:
        # Arrange
        stack = Glm5NextBlockStack()
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
        count = apply_glm5_next_tensor_parallel(stack, mesh)
        params = dict(stack.named_parameters())
        forwards = {
            name: vars(module).get("forward") for name, module in stack.named_modules()
        }

        # Act
        recount = apply_glm5_next_tensor_parallel(stack, mesh)

        # Assert
        assert recount == count
        assert all(param is params[name] for name, param in stack.named_parameters())
        assert all(
            vars(module).get("forward") is forwards[name]
            for name, module in stack.named_modules()
        )

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("tp", "ep", "world"),
        [(1, 1, 2), (1, 2, 2), (2, 2, 4)],
    )
    def test_forward_and_step_match_unsharded(
        self, tp: int, ep: int, world: int
    ) -> None:
        run_ranks(_worker, world, tp, ep, timeout=300)

    @pytest.mark.multi_rank
    def test_meta_load_fills_dp_tp_ep_shards_in_permuted_row_order(
        self, tmp_path: Path
    ) -> None:
        run_ranks(_meta_load_worker, 4, 2, 2, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(("save_tp", "load_tp"), [(2, 4), (4, 2)])
    def test_checkpoint_resumes_unchanged_at_another_tp_degree(
        self, save_tp: int, load_tp: int, tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save_tp, load_tp, 4, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    def test_indivisible_linear_heads_raise(self) -> None:
        run_ranks(_indivisible_heads_worker, 3, timeout=300)


class TestGlm5NextVisionParallel:
    def test_tiny_tower_runs(self) -> None:
        stack = _with_tower_lora(Glm5NextVisionStack())
        pixels, probe = _vision_data(0)

        output = stack(pixels)

        assert output.shape == probe.shape
        assert torch.isfinite(output).all()

    @pytest.mark.multi_rank
    def test_forward_and_step_match_unsharded(self) -> None:
        run_ranks(_vision_worker, 4, 2, timeout=300)

    @pytest.mark.multi_rank
    def test_meta_load_fills_dp_tp_shards_in_permuted_row_order(
        self, tmp_path: Path
    ) -> None:
        run_ranks(_vision_meta_load_worker, 4, 2, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    def test_indivisible_vision_heads_raise(self) -> None:
        run_ranks(_indivisible_vision_heads_worker, 3, timeout=300)
