# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor, context, and expert parallel Qwen3.5-MoE blocks on a gloo mesh."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from transformers import AutoModel
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeExperts,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeMLP,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTopKRouter,
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionBlock,
    Qwen3_5MoeVisionMLP,
    Qwen3_5MoeVisionPatchEmbed,
    Qwen3_5MoeVisionPatchMerger,
)

from agilerl.architectures.qwen3_5.tensor_parallel import (
    GDN_PERMUTED_PARAMS,
    VISION_PERMUTED_PARAMS,
    _gdn_rank_ranges,
    _vision_rank_ranges,
    apply_qwen3_5_tensor_parallel,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import (
    fsdp_blocks,
)
from agilerl.distributed.context_parallel.sequence import shard_for_cp
from agilerl.distributed.fsdp import (
    materialize_fsdp2_from_cpu_state,
)
from agilerl.distributed.mesh import MeshAxisName
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.distributed.tensor_parallel import is_lora_b_weight
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
    publish_varlen_offsets,
    sgd,
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
    shard_data,
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
EXPERT_LORA_PARAMS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
TP_BLOCK_CLASSES = {
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeAttention,
    Qwen3_5MoeMLP,
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionMLP,
    Qwen3_5MoeVisionPatchMerger,
}
EP_BLOCK_CLASSES = {Qwen3_5MoeExperts}
# Norms and embeddings stay whole; the router and the hidden -> 1 shared-expert
# gate are tiny; vLLM serves the Conv3d patch embedding without LoRA.
SKIPPED_CLASSES = {
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    nn.LayerNorm,
    nn.Embedding,
    Qwen3_5MoeTopKRouter,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeVisionPatchEmbed,
}
# One weight of each class the TP plan shards, by stack FQN suffix.
TP_PROBE_WEIGHTS = (
    "layers.0.linear_attn.in_proj_qkv.base_layer.weight",
    "layers.1.self_attn.q_proj.base_layer.weight",
    "layers.0.mlp.shared_expert.gate_proj.base_layer.weight",
    "vision.attn.qkv.base_layer.weight",
    "vision.mlp.linear_fc1.base_layer.weight",
    "merger.merger.linear_fc1.base_layer.weight",
)
# Blocks the TP plan shards on Qwen3.6-35B-A3B: 30 GatedDeltaNets, 10
# attentions, 40 shared experts, 27 vision attentions and MLPs, one merger.
CHECKPOINT_TP_BLOCKS = 30 + 10 + 40 + 27 + 27 + 1


def _text_config(linear_key_heads: int = 2) -> Qwen3_5MoeTextConfig:
    config = Qwen3_5MoeTextConfig(
        hidden_size=HIDDEN,
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
        moe_intermediate_size=8,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        layer_types=["linear_attention", "full_attention"],
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def _vision_config() -> Qwen3_5MoeVisionConfig:
    config = Qwen3_5MoeVisionConfig(
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


class QwenMoeMergerBlock(nn.Module):
    """FSDP unit around the vision merger."""

    def __init__(self, config: Qwen3_5MoeVisionConfig) -> None:
        super().__init__()
        self.merger = Qwen3_5MoeVisionPatchMerger(config, use_postshuffle_norm=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.merger(hidden)


class QwenMoeBlockStack(nn.Module):
    """One linear-attention and one full-attention MoE layer, plus one vision block."""

    _no_split_modules = (
        "Qwen3_5MoeDecoderLayer",
        "Qwen3_5MoeVisionBlock",
        "QwenMoeMergerBlock",
    )

    def __init__(self, linear_key_heads: int = 2) -> None:
        super().__init__()
        text = _text_config(linear_key_heads)
        vision = _vision_config()
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(text, 0), Qwen3_5MoeDecoderLayer(text, 1)]
        )
        self.vision = Qwen3_5MoeVisionBlock(vision)
        self.merger = QwenMoeMergerBlock(vision)
        self.config = SimpleNamespace(model_type="qwen3_5_moe")
        # Experts are torch.empty and the router is zeros outside post_init.
        with torch.no_grad():
            for layer in self.layers:
                layer.mlp.experts.gate_up_proj.normal_(std=0.5)
                layer.mlp.experts.down_proj.normal_(std=0.5)
                layer.mlp.gate.weight.normal_()


def _with_moe_lora(module: nn.Module) -> nn.Module:
    inject_adapter_in_model(
        LoraConfig(
            r=2, target_modules=LORA_TARGETS, target_parameters=EXPERT_LORA_PARAMS
        ),
        module,
    )
    assert upgrade_moe_param_wrappers(module) == 2
    module.requires_grad_(True)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.5)
    return module


def _apply_liger_swaps(stack: nn.Module) -> None:
    """Swap the classes Liger reassigns for qwen3_5_moe, keeping the HF forwards."""
    modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm = type(
        "LigerRMSNormForQwen3Next", (nn.Module,), {}
    )
    modeling_qwen3_5_moe.Qwen3_5MoeExperts = type("LigerExperts", (nn.Module,), {})
    for module in stack.modules():
        if type(module).__name__ == "Qwen3_5MoeMLP":
            module.__dict__["_get_name"] = lambda: "LigerQwen3MoeSwiGLUMLP"


def _permuted_full(
    model: nn.Module, fqn: str, full: torch.Tensor, tp: int
) -> torch.Tensor:
    """``full`` in the per-rank row order its sharded parameter stores."""
    owner: nn.Module | None = None
    owner_name = ""
    for name, module in model.named_modules():
        if not isinstance(module, Qwen3_5MoeGatedDeltaNet | Qwen3_5MoeVisionAttention):
            continue
        candidate = plain(name)
        if fqn != candidate and not fqn.startswith(candidate + "."):
            continue
        if owner is None or len(candidate) > len(owner_name):
            owner = module
            owner_name = candidate
    if owner is None:
        return full
    relative = fqn[len(owner_name) + 1 :]
    if isinstance(owner, Qwen3_5MoeGatedDeltaNet) and (
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
    elif isinstance(owner, Qwen3_5MoeVisionAttention) and (
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


def _assert_tp_blocks_sharded(model: nn.Module, tp: int) -> None:
    """One weight of every TP-planned class holds 1/tp of its rows per TP rank."""
    params = {plain(name): param for name, param in model.named_parameters()}
    for suffix in TP_PROBE_WEIGHTS:
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
        if isinstance(module, Qwen3_5MoeExperts)
        for param in (module.gate_up_proj, module.down_proj)
    ]
    assert len(experts) == 4
    for param in experts:
        assert isinstance(param, DTensor)
        assert on_ep_mesh(param)
        index = param.device_mesh.mesh_dim_names.index(MeshAxisName.EP)
        assert param.placements[index] == Shard(0)
        assert int(param.device_mesh.size(index)) == ep


def _worker(rank: int, world_size: int, cp: int, tp: int, ep: int) -> None:
    """Forward, reshard, LoRA gather and one SGD step match the unsharded stack."""
    # Arrange
    torch.manual_seed(0)
    fake_fla_kernels()
    stack = _with_moe_lora(QwenMoeBlockStack())
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    _apply_liger_swaps(stack)
    dp = world_size // (cp * tp)
    mesh = parallel_mesh(world_size, cp, tp, ep)
    # The FSDP mesh folds cp into dp_shard.
    assert mesh.dp_shard.size() == 2 * cp
    config = fsdp_config(cp, tp, ep)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    if tp > 1 and cp > 1:
        with torch.no_grad():
            assert_sequence_roundtrip(stack, reference, cp, mesh)
    if cp > 1:
        publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
    cp_rank = 0 if mesh.cp is None else int(mesh.cp.get_local_rank())
    gdn, attn, vision, probes = shard_data(rank // (cp * tp))
    gdn = shard_for_cp(gdn, cp_rank, cp)
    gdn_probe, *other_probes = probes
    with torch.no_grad():
        expected_outputs = forward_blocks(reference, gdn, attn, vision)
    reference_loss = sum(
        cp_rank_loss(reference, dp_rank, rank_in_cp, cp)
        for dp_rank in range(dp)
        for rank_in_cp in range(cp)
    )
    (reference_loss / dp).backward()
    expected = {
        plain(name): param.detach() - param.grad
        for name, param in reference.named_parameters()
    }
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    optimizer = sgd(stack, lr=1.0)

    # Act
    outputs = forward_blocks(stack, gdn, attn, vision)
    assert_resharded(stack, reference, tp, world_size // tp)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp)
    if ep > 1:
        _assert_experts_on_ep(stack, ep)
    assert_lora(stack, reference)
    loss = (outputs[0] * shard_for_cp(gdn_probe, cp_rank, cp)).sum()
    for output, probe in zip(outputs[1:], other_probes, strict=True):
        loss = loss + (output * probe).sum() / cp
    step = runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=stack, max_grad_norm=1e6
    )

    # Assert
    for label, output, target in zip(
        ("gdn", "attention", "vision", "merger"),
        outputs,
        expected_outputs,
        strict=True,
    ):
        assert_exact(output, target, label)
    assert step is not None
    assert len(expected) == sum(1 for _ in stack.parameters())
    errors: list[str] = []
    with torch.no_grad():
        for name, param in stack.named_parameters():
            value = param.full_tensor()
            target = expected[plain(name)].reshape(value.shape)
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


def _meta_load_worker(
    rank: int, world_size: int, cp: int, tp: int, ep: int, checkpoint_dir: str
) -> None:
    """A meta stack loads each rank's dp_shard x cp x tp x ep piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    fake_fla_kernels()
    dense = QwenMoeBlockStack()
    if rank == 0:
        save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = QwenMoeBlockStack()
    stack.config = Qwen3_5MoeConfig(name_or_path=checkpoint_dir)
    mesh = parallel_mesh(world_size, cp, tp, ep)
    config = fsdp_config(cp, tp, ep)

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
    _assert_experts_on_ep(stack, ep)


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
    cp = 2
    torch.manual_seed(0)
    fake_fla_kernels()
    dense = _with_moe_lora(QwenMoeBlockStack(linear_key_heads=4))
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
    saved, runtime, optimizer = lora_runtime(dense, world_size, cp, save_tp, ep)
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
        dense, world_size, cp, load_tp, ep
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


def _meta_checkpoint_shard_worker(rank: int, world_size: int) -> None:
    config = Qwen3_5MoeConfig.from_dict(
        SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
    )
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    count = apply_qwen3_5_tensor_parallel(model, mesh)

    assert count == CHECKPOINT_TP_BLOCKS


class TestQwen35MoeCheckpointCoverage:
    def test_every_parameter_is_planned_or_skipped(self) -> None:
        # Arrange
        config = Qwen3_5MoeConfig.from_dict(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        )
        with torch.device("meta"):
            model = AutoModel.from_config(config)
        modules = dict(model.named_modules())
        known = TP_BLOCK_CLASSES | EP_BLOCK_CLASSES | SKIPPED_CLASSES

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
        assert planned == TP_BLOCK_CLASSES | EP_BLOCK_CLASSES

    @pytest.mark.multi_rank
    def test_tp_plan_shards_every_checkpoint_block(self) -> None:
        run_ranks(_meta_checkpoint_shard_worker, 2, timeout=300)


class TestQwen35MoeParallel:
    def test_tiny_stack_runs(self) -> None:
        stack = _with_moe_lora(QwenMoeBlockStack())
        hidden = torch.randn(BATCH, SEQ, HIDDEN)
        vision = torch.randn(SEQ, HIDDEN)

        outputs = forward_blocks(stack, hidden, hidden, vision)

        assert [output.shape for output in outputs] == [
            hidden.shape,
            hidden.shape,
            vision.shape,
            vision.shape,
        ]

    def test_heads_that_do_not_divide_tp_raise(self) -> None:
        mesh = MagicMock()
        mesh.size.return_value = 3

        with pytest.raises(ValueError, match="linear_num_key_heads"):
            apply_qwen3_5_tensor_parallel(QwenMoeBlockStack(), mesh)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "ep", "world"),
        [
            (1, 1, 1, 2),
            (2, 1, 1, 4),
            (1, 1, 2, 2),
            (1, 2, 2, 4),
            (2, 1, 2, 4),
            (2, 2, 2, 8),
        ],
    )
    def test_forward_and_step_match_unsharded(
        self, cp: int, tp: int, ep: int, world: int
    ) -> None:
        run_ranks(_worker, world, cp, tp, ep, timeout=300)

    @pytest.mark.multi_rank
    def test_meta_load_fills_dp_cp_tp_ep_shards_in_permuted_row_order(
        self, tmp_path: Path
    ) -> None:
        run_ranks(_meta_load_worker, 8, 2, 2, 2, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(("save_tp", "load_tp"), [(2, 4), (4, 2)])
    def test_checkpoint_resumes_unchanged_at_another_tp_degree(
        self, save_tp: int, load_tp: int, tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save_tp, load_tp, 4, str(tmp_path), timeout=300)
