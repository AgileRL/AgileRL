# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor, context, and expert parallel Nemotron-H blocks on a gloo mesh."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
from transformers import AutoModelForCausalLM, NemotronHConfig
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention,
    NemotronHBlock,
    NemotronHExperts,
    NemotronHForCausalLM,
    NemotronHMamba2Mixer,
    NemotronHMLP,
    NemotronHRMSNorm,
    NemotronHTopkRouter,
)

from agilerl.algorithms.core.sequence_layout import ContextParallelSequence
from agilerl.architectures.nemotron_h.tensor_parallel import (
    NEMOTRON_H_TENSOR_PARALLEL_PLAN,
    apply_nemotron_h_tensor_parallel,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import fsdp_blocks
from agilerl.distributed.context_parallel.sequence import (
    CP_SCAN_CHUNK,
    pad_right_for_cp,
    shard_for_cp,
)
from agilerl.distributed.context_parallel.ulysses import (
    clear_ulysses_params,
)
from agilerl.distributed.fsdp import materialize_fsdp2_from_cpu_state
from agilerl.distributed.mesh import MeshAxisName
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.lora.moe import upgrade_moe_param_wrappers
from tests.parallel_helpers import (
    CollectiveRecorder,
    assert_exact,
    assert_lora,
    assert_resharded,
    assert_same_state,
    broadcast_module,
    broadcast_tensor,
    fsdp_config,
    lora_runtime,
    on_ep_mesh,
    on_tp_mesh,
    parallel_mesh,
    plain,
    publish_varlen_offsets,
    sgd,
    unwrap,
)
from tests.rank_launch import run_ranks

LIGHTNING = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
HIDDEN = 16
BATCH = 2
SEQ = 4
# PEFT rejects LoRA on the Mamba out_proj for nemotron_h.
LORA_TARGETS = [
    "in_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
]
EXPERT_LORA_PARAMS = ["mixer.experts.up_proj", "mixer.experts.down_proj"]
TP_BLOCK_CLASSES = {NemotronHMamba2Mixer, NemotronHAttention, NemotronHMLP}
EP_BLOCK_CLASSES = {NemotronHExperts}
# The untied lm_head is sharded by the generic head shard, not the family plan.
HEAD_CLASSES = {NemotronHForCausalLM}
# Norms and embeddings stay whole; the router is tiny.
SKIPPED_CLASSES = {NemotronHRMSNorm, nn.Embedding, NemotronHTopkRouter}
# One weight of each class the TP plan shards, by stack FQN suffix.
TP_PROBE_WEIGHTS = (
    "layers.0.mixer.in_proj.base_layer.weight",
    "layers.1.mixer.q_proj.base_layer.weight",
    "layers.2.mixer.shared_experts.up_proj.base_layer.weight",
    "layers.3.mixer.up_proj.base_layer.weight",
)
# Blocks the TP plan shards per checkpoint: Mamba mixers, attentions, and MLPs
# (shared-expert MLPs on Lightning, plain MLP blocks on Nano).
CHECKPOINT_TP_BLOCKS = {LIGHTNING: 23 + 6 + 23, NANO: 21 + 4 + 17}
CHECKPOINT_PLANNED_CLASSES = {
    LIGHTNING: TP_BLOCK_CLASSES | EP_BLOCK_CLASSES | HEAD_CLASSES,
    NANO: TP_BLOCK_CLASSES | HEAD_CLASSES,
}


def _config() -> NemotronHConfig:
    config = NemotronHConfig(
        hidden_size=HIDDEN,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        mamba_num_heads=16,
        mamba_head_dim=4,
        n_groups=8,
        ssm_state_size=4,
        chunk_size=4,
        use_conv_bias=True,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        intermediate_size=8,
        layers_block_type=["linear_attention", "full_attention", "moe", "mlp"],
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


class NemotronBlockStack(nn.Module):
    """One Mamba, one attention, one MoE, and one MLP block."""

    _no_split_modules = ("NemotronHBlock",)

    def __init__(self) -> None:
        super().__init__()
        config = _config()
        self.layers = nn.ModuleList([NemotronHBlock(config, i) for i in range(4)])
        self.config = SimpleNamespace(model_type="nemotron_h")
        # Experts are torch.empty and the router is zeros outside post_init.
        moe = self.layers[2].mixer
        with torch.no_grad():
            moe.experts.up_proj.normal_(std=0.5)
            moe.experts.down_proj.normal_(std=0.5)
            moe.gate.weight.normal_()


def _with_moe_lora(module: nn.Module) -> nn.Module:
    inject_adapter_in_model(
        LoraConfig(
            r=2, target_modules=LORA_TARGETS, target_parameters=EXPERT_LORA_PARAMS
        ),
        module,
    )
    assert upgrade_moe_param_wrappers(module) == 1
    module.requires_grad_(True)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.5)
    return module


def _apply_liger_swaps() -> None:
    """Swap the class Liger reassigns for nemotron_h, keeping the HF forwards."""
    modeling_nemotron_h.NemotronHRMSNorm = type(
        "LigerNemotronHRMSNorm", (nn.Module,), {}
    )


def _forward_blocks(
    stack: nn.Module,
    mamba_hidden: torch.Tensor,
    attn_hidden: torch.Tensor,
    moe_hidden: torch.Tensor,
    mlp_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        stack.layers[0](mamba_hidden),
        stack.layers[1](attn_hidden),
        stack.layers[2](moe_hidden),
        stack.layers[3](mlp_hidden),
    )


def _shard_data(dp_rank: int) -> tuple[torch.Tensor, ...]:
    """Mamba, attention, MoE and MLP inputs, then one output probe for each."""
    generator = torch.Generator().manual_seed(200 + dp_rank)
    return tuple(torch.randn(BATCH, SEQ, HIDDEN, generator=generator) for _ in range(8))


def _rank_loss(stack: nn.Module, dp_rank: int, cp_rank: int, cp: int) -> torch.Tensor:
    """This rank's Mamba, MoE and MLP slice losses plus 1/cp of the attention loss."""
    mamba, attn, moe, mlp, mamba_probe, attn_probe, moe_probe, mlp_probe = _shard_data(
        dp_rank
    )
    mamba_out, attn_out, moe_out, mlp_out = _forward_blocks(
        stack,
        shard_for_cp(mamba, cp_rank, cp),
        attn,
        shard_for_cp(moe, cp_rank, cp),
        shard_for_cp(mlp, cp_rank, cp),
    )
    return (
        (mamba_out * shard_for_cp(mamba_probe, cp_rank, cp)).sum()
        + (attn_out * attn_probe).sum() / cp
        + (moe_out * shard_for_cp(moe_probe, cp_rank, cp)).sum()
        + (mlp_out * shard_for_cp(mlp_probe, cp_rank, cp)).sum()
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
        if isinstance(module, NemotronHExperts)
        for param in (module.up_proj, module.down_proj)
    ]
    assert len(experts) == 2
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


def _assert_sequence_roundtrip(
    model: nn.Module, reference: nn.Module, cp: int, mesh: object
) -> None:
    """The shared expert through ``prepare_forward`` and ``scoring_hidden`` matches unsharded."""
    layout = ContextParallelSequence(
        pad_token_id=0,
        cp=cp,
        group=mesh.cp.get_group(),
        rank=int(mesh.cp.get_local_rank()),
        round_to=cp * CP_SCAN_CHUNK,
    )
    ids = broadcast_tensor(torch.randint(1, 32, (BATCH, 5)))
    mask = torch.ones_like(ids)
    table = broadcast_tensor(torch.randn(32, HIDDEN))
    aligned, _aligned_mask = pad_right_for_cp(
        ids, cp * CP_SCAN_CHUNK, layout.pad_token_id, mask
    )
    assert aligned.shape[1] - 1 == cp * CP_SCAN_CHUNK
    block = model.layers[2]
    mlp = unwrap(block).mixer.shared_experts
    block.unshard()
    with layout.prepare_forward(ids, mask) as forward:
        restored = forward.scoring_hidden(mlp(table[forward.kwargs["input_ids"]]))
    block.reshard()
    expected = reference.layers[2].mixer.shared_experts(table[aligned[:, :-1]])
    expected = shard_for_cp(expected, layout.rank, cp)
    assert_exact(restored, expected, "cp sequence")


def _worker(rank: int, world_size: int, cp: int, tp: int, ep: int) -> None:
    """Forward, reshard, LoRA gather and one SGD step match the unsharded stack."""
    # Arrange
    torch.manual_seed(0)
    stack = _with_moe_lora(NemotronBlockStack())
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    _apply_liger_swaps()
    dp = world_size // (cp * tp)
    mesh = parallel_mesh(world_size, cp, tp, ep)
    # The FSDP mesh folds cp into dp_shard.
    assert mesh.dp_shard.size() == 2 * cp
    config = fsdp_config(cp, tp, ep)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    if cp > 1:
        with torch.no_grad():
            _assert_sequence_roundtrip(stack, reference, cp, mesh)
    cp_rank = 0 if mesh.cp is None else int(mesh.cp.get_local_rank())
    mamba, attn, moe, mlp, mamba_probe, attn_probe, moe_probe, mlp_probe = _shard_data(
        rank // (cp * tp)
    )
    with torch.no_grad():
        full_outputs = _forward_blocks(reference, mamba, attn, moe, mlp)
    reference_loss = 0
    for dp_rank in range(dp):
        outputs = _forward_blocks(reference, *_shard_data(dp_rank)[:4])
        probes = _shard_data(dp_rank)[4:]
        for output, probe in zip(outputs, probes, strict=True):
            reference_loss = reference_loss + (output * probe).sum()
    (reference_loss / dp).backward()
    expected = {
        plain(name): param.detach() - param.grad
        for name, param in reference.named_parameters()
    }
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    optimizer = sgd(stack, lr=1.0)

    # Act
    if cp > 1:
        publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
    outputs = _forward_blocks(
        stack,
        shard_for_cp(mamba, cp_rank, cp),
        attn,
        shard_for_cp(moe, cp_rank, cp),
        shard_for_cp(mlp, cp_rank, cp),
    )
    assert_resharded(stack, reference, tp, world_size // tp)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp)
    if ep > 1:
        _assert_experts_on_ep(stack, ep)
    assert_lora(stack, reference)
    loss = (
        (outputs[0] * shard_for_cp(mamba_probe, cp_rank, cp)).sum()
        + (outputs[1] * attn_probe).sum() / cp
        + (outputs[2] * shard_for_cp(moe_probe, cp_rank, cp)).sum()
        + (outputs[3] * shard_for_cp(mlp_probe, cp_rank, cp)).sum()
    )
    step = runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=stack, max_grad_norm=1e6
    )
    clear_ulysses_params()

    # Assert
    full_mamba, full_attn, full_moe, full_mlp = full_outputs
    for label, output, target in zip(
        ("mamba", "attention", "moe", "mlp"),
        outputs,
        (
            shard_for_cp(full_mamba, cp_rank, cp),
            full_attn,
            shard_for_cp(full_moe, cp_rank, cp),
            shard_for_cp(full_mlp, cp_rank, cp),
        ),
        strict=True,
    ):
        assert_exact(output, target, label)
    assert step is not None
    assert len(expected) == sum(1 for _ in stack.parameters())
    orders = NEMOTRON_H_TENSOR_PARALLEL_PLAN.row_orders(stack)
    errors: list[str] = []
    with torch.no_grad():
        for name, param in stack.named_parameters():
            value = param.full_tensor()
            target = expected[plain(name)].reshape(value.shape)
            try:
                assert_exact(value, _permuted_full(orders, name, target), name)
            except AssertionError as exc:
                errors.append(str(exc))
    if errors:
        raise AssertionError("\n".join(errors))


def _checkpointed_moe_worker(rank: int, world_size: int, tp: int, ep: int) -> None:
    """The checkpointed MoE block matches the uncheckpointed one; with saved collectives its recompute sends nothing extra."""
    # Arrange
    torch.manual_seed(0)
    stack = _with_moe_lora(NemotronBlockStack())
    broadcast_module(stack)
    _apply_liger_swaps()
    data = _shard_data(rank // tp)
    moe, probe = data[2], data[6]

    # Act
    results = {}
    for checkpointed, save in ((False, None), (True, False), (True, True)):
        model = copy.deepcopy(stack)
        mesh = parallel_mesh(world_size, 1, tp, ep)
        with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
            materialize_fsdp2_from_cpu_state(
                model,
                "cpu",
                replace(fsdp_config(1, tp, ep), save_moe_collectives=save),
                parallel_mesh=mesh,
                gradient_checkpointing=checkpointed,
            )
        hidden = moe.clone().requires_grad_(True)
        with CollectiveRecorder() as forward:
            out = model.layers[2](hidden)
        with CollectiveRecorder() as backward:
            (out * probe).sum().backward()
        grads = {
            plain(name): param.grad.to_local()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        results[checkpointed, save] = (
            out.detach(),
            hidden.grad,
            grads,
            forward,
            backward,
        )

    # Assert
    plain_out, plain_hidden_grad, plain_grads, plain_forward, plain_backward = results[
        False, None
    ]
    for key in ((True, False), (True, True)):
        out, hidden_grad, grads, _, _ = results[key]
        assert torch.equal(out, plain_out), key
        assert torch.equal(hidden_grad, plain_hidden_grad), key
        assert grads.keys() == plain_grads.keys()
        assert grads
        for name, grad in grads.items():
            assert torch.equal(grad, plain_grads[name]), (key, name)
    # Token all-to-alls and, under TP, the row gather are saved. The shared
    # expert's all-reduce ends the block, past where the recompute stops.
    _, _, _, saved_forward, saved_backward = results[True, True]
    saved_kinds = {kind for kind, functional, _, _ in saved_forward.calls if functional}
    eager_kinds = [
        kind for kind, functional, _, _ in saved_forward.calls if not functional
    ]
    assert saved_kinds == ({"all_to_all", "all_gather"} if tp > 1 else {"all_to_all"})
    assert eager_kinds == (["all_reduce"] if tp > 1 else [])
    assert saved_forward.kinds == plain_forward.kinds
    assert saved_backward.kinds == plain_backward.kinds
    # Off, the forward stays eager and the recompute repeats the routed collectives.
    _, _, _, eager_forward, recomputed_backward = results[True, False]
    assert eager_forward.calls == plain_forward.calls
    assert not any(functional for _, functional, _, _ in recomputed_backward.calls)
    assert recomputed_backward.rows("all_to_all") > plain_backward.rows("all_to_all")
    if tp > 1:
        # The recompute stops before the shared expert, so only the row gather repeats.
        assert recomputed_backward.kinds.count(
            "all_gather"
        ) > plain_backward.kinds.count("all_gather")


def _meta_load_worker(
    rank: int, world_size: int, cp: int, tp: int, ep: int, checkpoint_dir: str
) -> None:
    """A meta stack loads each rank's dp_shard x cp x tp x ep piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    dense = NemotronBlockStack()
    if rank == 0:
        save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = NemotronBlockStack()
    stack.config = NemotronHConfig(name_or_path=checkpoint_dir)
    mesh = parallel_mesh(world_size, cp, tp, ep)
    config = fsdp_config(cp, tp, ep)

    # Act
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(stack, "cpu", config, parallel_mesh=mesh)

    # Assert
    expected = dict(dense.named_parameters())
    orders = NEMOTRON_H_TENSOR_PARALLEL_PLAN.row_orders(stack)
    assert orders
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
    dense = _with_moe_lora(NemotronBlockStack())
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    saved, runtime, optimizer = lora_runtime(dense, world_size, cp, save_tp, ep)
    mesh = runtime.parallel_mesh
    assert mesh is not None
    assert mesh.cp is not None
    publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
    loss = _rank_loss(saved, rank // (cp * save_tp), int(mesh.cp.get_local_rank()), cp)
    runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=saved, max_grad_norm=1e6
    )
    clear_ulysses_params()
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


def _checkpoint_meta_model(model_id: str) -> nn.Module:
    config = NemotronHConfig.from_dict(SUPPORTED_MODEL_INFO[model_id].config)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def _meta_checkpoint_shard_worker(rank: int, world_size: int, model_id: str) -> None:
    model = _checkpoint_meta_model(model_id)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    count = apply_nemotron_h_tensor_parallel(model, mesh)

    assert count == CHECKPOINT_TP_BLOCKS[model_id]


@pytest.mark.parametrize("model_id", [LIGHTNING, NANO])
class TestNemotronHCheckpointCoverage:
    def test_every_parameter_is_planned_or_skipped(self, model_id: str) -> None:
        # Arrange
        model = _checkpoint_meta_model(model_id)
        modules = dict(model.named_modules())
        known = TP_BLOCK_CLASSES | EP_BLOCK_CLASSES | HEAD_CLASSES | SKIPPED_CLASSES

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
        assert planned == CHECKPOINT_PLANNED_CLASSES[model_id]

    @pytest.mark.multi_rank
    def test_tp_plan_shards_every_checkpoint_block(self, model_id: str) -> None:
        run_ranks(_meta_checkpoint_shard_worker, 2, model_id, timeout=300)


class TestNemotronHParallel:
    def test_tiny_stack_runs(self) -> None:
        stack = _with_moe_lora(NemotronBlockStack())
        hidden = torch.randn(BATCH, SEQ, HIDDEN)

        outputs = _forward_blocks(stack, hidden, hidden, hidden, hidden)

        assert [output.shape for output in outputs] == [hidden.shape] * 4

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
    @pytest.mark.parametrize(("tp", "ep", "world"), [(1, 2, 2), (2, 2, 4)])
    def test_checkpointed_moe_block_matches_and_saved_collectives_skip_recompute(
        self, tp: int, ep: int, world: int
    ) -> None:
        run_ranks(_checkpointed_moe_worker, world, tp, ep, timeout=300)

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
