# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor- and context-parallel Qwen3 blocks on a gloo mesh."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from transformers import AutoModelForCausalLM
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    position_ids_from_mask,
)
from agilerl.architectures.catalog import family_runtime
from agilerl.architectures.qwen3.tensor_parallel import apply_qwen3_tensor_parallel
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import ParallelMesh, fsdp_blocks
from agilerl.distributed.context_parallel import (
    clear_ulysses_params,
)
from agilerl.distributed.context_parallel.sequence import (
    CP_SCAN_CHUNK,
    pad_right_for_cp,
    shard_for_cp,
    validate_cp_heads,
)
from agilerl.distributed.expert_parallel import iter_packed_expert_modules
from agilerl.distributed.fsdp import materialize_fsdp2_from_cpu_state
from agilerl.distributed.mesh import MeshAxisName
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.distributed.tensor_parallel import apply_tensor_parallel
from tests.parallel_helpers import (
    assert_exact,
    assert_lora,
    assert_resharded,
    assert_same_state,
    broadcast_module,
    broadcast_tensor,
    fsdp_config,
    lora_runtime,
    on_tp_mesh,
    parallel_mesh,
    plain,
    publish_varlen_offsets,
    sgd,
    unwrap,
    use_ulysses_flash,
    with_lora,
    world_held_numel,
)
from tests.rank_launch import run_ranks
from tests.test_architectures.qwen3.parallel_helpers import (
    BATCH,
    HIDDEN,
    LORA_TARGETS,
    SEQ,
    eager_mask,
    shard_data,
)

HEAD_DIM = 4
HEADS = 8
KV_HEADS = 4
CHECKPOINTS = ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")
# Legal tp degrees from 16 q / 8 kv heads and 32 q / 8 kv heads.
LEGAL_TP = {
    "Qwen/Qwen3-1.7B": [2, 4, 8, 16],
    "Qwen/Qwen3-4B": [2, 4, 8, 16, 32],
}
# Ulysses flash exchanges sequence for heads around attention.
CP_BLOCK_CLASSES = {Qwen3Attention}
# Token-wise blocks see only their own positions, so a sequence shard needs no exchange.
TOKENWISE_CLASSES = {Qwen3MLP}
TP_BLOCK_CLASSES = {Qwen3Attention, Qwen3MLP}
# Norms and the tied embedding stay whole.
SKIPPED_CLASSES = {Qwen3RMSNorm, nn.Embedding}
# One weight of each part the TP plan shards, by stack FQN without LoRA wrappers.
TP_PROBE_WEIGHTS = (
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.1.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.1.mlp.down_proj.weight",
)
# Per-head norms each TP rank keeps whole.
TP_REPLICATED_WEIGHTS = (
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
)


def _config() -> Qwen3Config:
    config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=32,
    )
    config._attn_implementation = "eager"
    return config


class Qwen3BlockStack(nn.Module):
    """Two Qwen3 decoder layers."""

    _no_split_modules = ("Qwen3DecoderLayer",)

    def __init__(self) -> None:
        super().__init__()
        config = _config()
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, 0), Qwen3DecoderLayer(config, 1)]
        )
        self.config = SimpleNamespace(model_type="qwen3")


def _apply_liger_swaps() -> None:
    """Swap the classes Liger reassigns for qwen3, keeping the HF forwards."""
    modeling_qwen3.Qwen3RMSNorm = type("LigerRMSNorm", (nn.Module,), {})
    modeling_qwen3.Qwen3MLP = type("LigerSwiGLUMLP", (nn.Module,), {})


def _rope(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = Qwen3RotaryEmbedding(_config())
    return rotary(torch.zeros(*positions.shape, HIDDEN), positions)


def _forward(stack: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Run both layers on the full sequence with an eager causal mask."""
    rope = _rope(torch.arange(SEQ).expand(BATCH, SEQ))
    for layer in stack.layers:
        hidden = layer(hidden, attention_mask=eager_mask(), position_embeddings=rope)
    return hidden


def _forward_cp_shard(
    stack: nn.Module, hidden: torch.Tensor, local: slice
) -> torch.Tensor:
    """Run both layers on one CP sequence shard; Ulysses gathers the sequence."""
    rope = _rope(torch.arange(SEQ).expand(BATCH, SEQ)[:, local])
    for layer in stack.layers:
        hidden = layer(hidden, attention_mask=None, position_embeddings=rope)
    return hidden


def _assert_sequence_roundtrip(
    model: nn.Module, reference: nn.Module, cp: int, mesh: ParallelMesh
) -> None:
    """Attention through ContextParallelSequence and scoring_hidden matches the local shard."""
    group = mesh.cp.get_group()
    layout = ContextParallelSequence(
        pad_token_id=0,
        cp=cp,
        group=group,
        rank=int(mesh.cp.get_local_rank()),
        round_to=cp * CP_SCAN_CHUNK,
    )
    ids = broadcast_tensor(torch.randint(1, 32, (BATCH, 5)))
    mask = torch.ones_like(ids)
    table = broadcast_tensor(torch.randn(32, HIDDEN))
    aligned, aligned_mask = pad_right_for_cp(
        ids, cp * CP_SCAN_CHUNK, layout.pad_token_id, mask
    )
    assert aligned.shape[1] - 1 == cp * CP_SCAN_CHUNK
    block = model.layers[1]
    attention = unwrap(block).self_attn
    block.unshard()
    with layout.prepare_forward(ids, mask) as forward:
        local = table[forward.kwargs["input_ids"]]
        rope = _rope(forward.kwargs["position_ids"])
        restored = forward.scoring_hidden(attention(local, rope, None)[0])
    block.reshard()
    length = aligned.shape[1] - 1
    positions = position_ids_from_mask(aligned_mask)[:, :length]
    expected = reference.layers[1].self_attn(
        table[aligned[:, :-1]], _rope(positions), eager_mask(length)
    )[0]
    expected = shard_for_cp(expected, layout.rank, cp)
    assert_exact(restored, expected, "cp sequence")


def _assert_tp_blocks_sharded(model: nn.Module, tp: int) -> None:
    """Each probed weight holds 1/tp of its elements per TP rank; norms stay whole."""
    params = {
        plain(name).replace(".base_layer", ""): param
        for name, param in model.named_parameters()
    }
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
    for suffix in TP_REPLICATED_WEIGHTS:
        param = params[suffix]
        assert isinstance(param, DTensor), suffix
        assert on_tp_mesh(param), suffix
        assert param.placements[-1].is_replicate(), suffix


def _worker(rank: int, world_size: int, cp: int, tp: int) -> None:
    """Forward, reshard, LoRA gather and one SGD step match the unsharded stack."""
    # Arrange
    torch.manual_seed(0)
    stack = with_lora(Qwen3BlockStack(), LORA_TARGETS, std=0.1)
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    _apply_liger_swaps()
    dp = world_size // (cp * tp)
    mesh = parallel_mesh(world_size, cp, tp, 1)
    # The FSDP mesh folds cp into dp_shard.
    assert mesh.dp_shard.size() == 2 * cp
    config = fsdp_config(cp, tp, 1)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    local = slice(None)
    if cp > 1:
        assert mesh.cp is not None
        use_ulysses_flash(unwrap(stack.layers[0]).self_attn.config, mesh.cp.get_group())
        with torch.no_grad():
            _assert_sequence_roundtrip(stack, reference, cp, mesh)
        publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
        cp_rank = int(mesh.cp.get_local_rank())
        local = slice(cp_rank * SEQ // cp, (cp_rank + 1) * SEQ // cp)
    hidden, probe = shard_data(rank // (cp * tp))
    with torch.no_grad():
        expected_output = _forward(reference, hidden)[:, local]
    reference_loss = sum(
        (_forward(reference, data) * data_probe).sum()
        for data, data_probe in map(shard_data, range(dp))
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
    if cp > 1:
        output = _forward_cp_shard(stack, hidden[:, local], local)
    else:
        output = _forward(stack, hidden)
    assert_resharded(stack, reference, tp, world_size // tp)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp)
    assert_lora(stack, reference)
    loss = (output * probe[:, local]).sum()
    step = runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=stack, max_grad_norm=1e6
    )
    clear_ulysses_params()

    # Assert
    assert_exact(output, expected_output, "output")
    assert step is not None
    assert len(expected) == sum(1 for _ in stack.parameters())
    with torch.no_grad():
        for name, param in stack.named_parameters():
            assert_exact(param.full_tensor(), expected[plain(name)], name)


def _frozen_tp_worker(rank: int, world_size: int, shard_frozen_tp: bool) -> None:
    """Frozen TP shards are resident or FSDP-sharded; LoRA training matches unsharded."""
    # Arrange
    torch.manual_seed(0)
    tp = 2
    dp = world_size // tp
    dense = with_lora(Qwen3BlockStack(), LORA_TARGETS, std=0.1)
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    reference = copy.deepcopy(dense)
    stack = copy.deepcopy(dense)
    mesh = parallel_mesh(world_size, 1, tp, 1)
    config = fsdp_config(1, tp, 1, shard_frozen_tp=shard_frozen_tp)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    optimizer = sgd(stack, lr=1.0)
    hidden, probe = shard_data(rank // tp)
    with torch.no_grad():
        expected_output = _forward(reference, hidden)
    reference_loss = sum(
        (_forward(reference, data) * data_probe).sum()
        for data, data_probe in map(shard_data, range(dp))
    )
    (reference_loss / dp).backward()
    expected_lora = {
        plain(name): param.detach() - param.grad
        for name, param in reference.named_parameters()
        if param.requires_grad
    }
    params = {
        plain(name).replace(".base_layer", ""): param
        for name, param in stack.named_parameters()
    }

    # Act
    output = _forward(stack, hidden)
    loss = (output * probe).sum()
    runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=stack, max_grad_norm=1e6
    )
    model_state = runtime.export_model_state(stack)
    shared = [model_state]
    dist.broadcast_object_list(shared, src=0)
    resumed, resumed_runtime, _ = lora_runtime(
        dense, world_size, 1, tp, 1, shard_frozen_tp=shard_frozen_tp
    )
    resumed_runtime.import_model_state(resumed, shared[0], strict=True)
    reexported = resumed_runtime.export_model_state(resumed)

    # Assert
    for suffix in TP_PROBE_WEIGHTS:
        frozen = params[suffix]
        assert not frozen.requires_grad, suffix
        if shard_frozen_tp:
            assert frozen.device_mesh.size() == world_size, suffix
            assert world_held_numel(frozen) == frozen.numel(), suffix
        else:
            assert frozen.device_mesh.mesh_dim_names == (MeshAxisName.TP,), suffix
            assert world_held_numel(frozen) == frozen.numel() * dp, suffix
    assert_exact(output, expected_output, "output")
    assert_exact(loss, (expected_output * probe).sum(), "loss")
    with torch.no_grad():
        for name, param in stack.named_parameters():
            if param.requires_grad:
                assert_exact(param.full_tensor(), expected_lora[plain(name)], name)
            else:
                assert_exact(param.full_tensor(), natural[plain(name)], name)
    if rank != 0:
        return
    assert model_state.keys() == natural.keys()
    for key, value in model_state.items():
        expected = expected_lora[key] if "lora_" in key else natural[key]
        assert_exact(value, expected, key)
    assert_same_state(reexported, model_state, "model")


def _meta_load_worker(
    rank: int, world_size: int, cp: int, tp: int, checkpoint_dir: str
) -> None:
    """A meta stack loads each rank's dp_shard x cp x tp piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    dense = Qwen3BlockStack()
    if rank == 0:
        save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = Qwen3BlockStack()
    stack.config = Qwen3Config(name_or_path=checkpoint_dir)
    mesh = parallel_mesh(world_size, cp, tp, 1)
    config = fsdp_config(cp, tp, 1)

    # Act
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(stack, "cpu", config, parallel_mesh=mesh)

    # Assert
    expected = dict(dense.named_parameters())
    for name, param in stack.named_parameters():
        assert isinstance(param, DTensor), name
        mesh_size = world_size if on_tp_mesh(param) else world_size // tp
        assert param.device_mesh.size() == mesh_size, name
        assert_exact(param.full_tensor(), expected[plain(name)], name)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp)


def _resume_worker(
    rank: int,
    world_size: int,
    save: tuple[int, int],
    load: tuple[int, int],
    checkpoint_dir: str,
) -> None:
    """A LoRA checkpoint saved at one ``(cp, tp)`` mesh resumes unchanged at another."""
    # Arrange
    torch.manual_seed(0)
    dense = with_lora(Qwen3BlockStack(), LORA_TARGETS, std=0.1)
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    save_cp, save_tp = save
    load_cp, load_tp = load
    saved, runtime, optimizer = lora_runtime(dense, world_size, save_cp, save_tp, 1)
    hidden, probe = shard_data(rank // (save_cp * save_tp))
    loss = (_forward(saved, hidden) * probe).sum()
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
        dense, world_size, load_cp, load_tp, 1
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


def _meta_checkpoint_shard_worker(rank: int, world_size: int, model_id: str) -> None:
    config = Qwen3Config.from_dict(SUPPORTED_MODEL_INFO[model_id].config)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    count = apply_tensor_parallel(model, mesh)

    # Attention and MLP in every layer; the tied LM head stays whole.
    assert count == 2 * config.num_hidden_layers
    layer = model.model.layers[0]
    q_rows = config.num_attention_heads * config.head_dim
    assert layer.self_attn.q_proj.weight.to_local().shape[0] == q_rows // world_size
    mlp_rows = config.intermediate_size // world_size
    assert layer.mlp.gate_proj.weight.to_local().shape[0] == mlp_rows
    assert layer.mlp.down_proj.weight.to_local().shape[1] == mlp_rows
    assert not isinstance(model.lm_head.weight, DTensor)


class TestQwen3CheckpointCoverage:
    @pytest.mark.parametrize("model_id", CHECKPOINTS)
    def test_every_parameter_is_planned_or_skipped(self, model_id: str) -> None:
        # Arrange
        config = Qwen3Config.from_dict(SUPPORTED_MODEL_INFO[model_id].config)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        modules = dict(model.named_modules())
        known = (
            CP_BLOCK_CLASSES | TOKENWISE_CLASSES | TP_BLOCK_CLASSES | SKIPPED_CLASSES
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
        assert set(owners.values()) == known
        assert set(owners.values()) - SKIPPED_CLASSES == TP_BLOCK_CLASSES

    @pytest.mark.parametrize("model_id", CHECKPOINTS)
    def test_tensor_and_context_parallel_apply(self, model_id: str) -> None:
        # Arrange
        config = Qwen3Config.from_dict(SUPPORTED_MODEL_INFO[model_id].config)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        runtime = family_runtime(model_id)

        # Act
        experts = iter_packed_expert_modules(model)

        # Assert
        assert experts == []
        assert runtime.tensor_parallel_plan == (
            "agilerl.architectures.qwen3.tensor_parallel:QWEN3_TENSOR_PARALLEL_PLAN"
        )
        assert model.lm_head.weight is model.model.embed_tokens.weight
        for cp in (2, 4):
            validate_cp_heads(
                config.num_attention_heads, config.num_key_value_heads, cp
            )

    @pytest.mark.parametrize("model_id", CHECKPOINTS)
    def test_legal_tp_degrees_follow_head_counts(self, model_id: str) -> None:
        # Arrange
        config = Qwen3Config.from_dict(SUPPORTED_MODEL_INFO[model_id].config)
        heads = config.num_attention_heads
        heads_per_kv = heads // config.num_key_value_heads

        # Act
        legal = [
            tp
            for tp in range(2, heads + 1)
            if heads % tp == 0
            and (heads_per_kv % (heads // tp) == 0 or (heads // tp) % heads_per_kv == 0)
            and config.intermediate_size % tp == 0
        ]

        # Assert
        assert legal == LEGAL_TP[model_id]

    @pytest.mark.multi_rank
    @pytest.mark.parametrize("model_id", CHECKPOINTS)
    def test_tp_plan_shards_every_checkpoint_block(self, model_id: str) -> None:
        run_ranks(_meta_checkpoint_shard_worker, 2, model_id, timeout=300)


class LigerSwiGLUMLP(Qwen3MLP):
    """Stands in for Liger's MLP: another class name, the same projections."""


def _liger_mlp_worker(rank: int, world_size: int) -> None:
    # Arrange
    stack = Qwen3BlockStack()
    for layer in stack.layers:
        layer.mlp.__class__ = LigerSwiGLUMLP
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    # Act
    count = apply_qwen3_tensor_parallel(stack, mesh)

    # Assert
    assert count == 4
    for layer in stack.layers:
        down = layer.mlp.down_proj.weight
        assert isinstance(down, DTensor)
        assert down.placements == (Shard(1),)


class TestQwen3Parallel:
    @pytest.mark.multi_rank
    def test_class_swapped_liger_mlp_still_shards(self) -> None:
        run_ranks(_liger_mlp_worker)

    def test_tiny_stack_runs(self) -> None:
        stack = with_lora(Qwen3BlockStack(), LORA_TARGETS, std=0.1)
        hidden = torch.randn(BATCH, SEQ, HIDDEN)

        output = _forward(stack, hidden)

        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()

    def test_heads_that_do_not_divide_tp_raise(self) -> None:
        mesh = MagicMock()
        mesh.size.return_value = 3

        with pytest.raises(ValueError, match="num_attention_heads"):
            apply_qwen3_tensor_parallel(Qwen3BlockStack(), mesh)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "world"), [(1, 1, 2), (1, 2, 4), (2, 1, 4), (2, 2, 8)]
    )
    def test_forward_and_step_match_unsharded(
        self, cp: int, tp: int, world: int
    ) -> None:
        run_ranks(_worker, world, cp, tp, timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize("shard_frozen_tp", [False, True])
    def test_frozen_tp_weights_resident_or_sharded_train_lora_unchanged(
        self, shard_frozen_tp: bool
    ) -> None:
        run_ranks(_frozen_tp_worker, 4, shard_frozen_tp, timeout=300)

    @pytest.mark.multi_rank
    def test_meta_load_fills_dp_cp_shards(self, tmp_path: Path) -> None:
        run_ranks(_meta_load_worker, 4, 2, 1, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    def test_meta_load_fills_dp_cp_tp_shards(self, tmp_path: Path) -> None:
        run_ranks(_meta_load_worker, 8, 2, 2, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(("save", "load"), [((2, 1), (4, 1)), ((4, 1), (2, 1))])
    def test_checkpoint_resumes_unchanged_at_another_cp_degree(
        self, save: tuple[int, int], load: tuple[int, int], tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save, load, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(("save", "load"), [((1, 2), (1, 4)), ((1, 4), (1, 2))])
    def test_checkpoint_resumes_unchanged_at_another_tp_degree(
        self, save: tuple[int, int], load: tuple[int, int], tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save, load, str(tmp_path), timeout=300)
