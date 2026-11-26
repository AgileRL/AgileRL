# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-, context- and expert-parallel DeepSeek-V4 blocks on a gloo mesh."""

from __future__ import annotations

import copy
import math
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
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, create_mask
from transformers import AutoModel, AutoModelForCausalLM
from transformers.models.deepseek_v4 import modeling_deepseek_v4
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    DeepseekV4Experts,
    DeepseekV4HashRouter,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4MLP,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4TopKRouter,
)

from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    position_ids_from_mask,
)
from agilerl.architectures.deepseek_v4 import patches as deepseek_v4_module
from agilerl.architectures.deepseek_v4.context_parallel import (
    install_deepseek_v4_context_parallel,
    wrap_deepseek_v4_context_parallel,
)
from agilerl.architectures.deepseek_v4.patches import (
    CONTEXT_PARALLEL_GROUP,
    document_keys,
    flex_compressed_attention,
    sequence_shard_keys,
)
from agilerl.architectures.deepseek_v4.tensor_parallel import (
    _install_group_forward,
    apply_deepseek_v4_tensor_parallel,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import ParallelMesh, fsdp_blocks
from agilerl.distributed.context_parallel import (
    clear_ulysses_params,
)
from agilerl.distributed.context_parallel.sequence import (
    gather_sequence_for_cp,
    pad_right_for_cp,
)
from agilerl.distributed.expert_parallel import (
    iter_packed_expert_modules,
    packed_expert_counts,
)
from agilerl.distributed.fsdp import materialize_fsdp2_from_cpu_state
from agilerl.distributed.mesh import MeshAxisName
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.distributed.tensor_parallel import apply_tensor_parallel
from agilerl.lora.moe import upgrade_moe_param_wrappers
from tests.parallel_helpers import (
    CollectiveRecorder,
    assert_exact,
    assert_lora,
    assert_resharded,
    assert_same_state,
    broadcast_module,
    broadcast_tensor,
    dense_flex_attention,
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

HUB_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
HIDDEN = 16
HEADS = 8
O_GROUPS = 4
HC_MULT = 4
EXPERTS = 4
TOP_K = 2
VOCAB = 32
BATCH = 2
# Six CSA entries, so the indexer selects, and three HCA entries. Split over
# cp=2, an HCA window, a CSA overlap pair and the sliding window cross the shards.
SEQ = 12
WINDOW = 4
INIT_STD = 0.3
LORA_TARGETS = sorted(SUPPORTED_MODEL_INFO[HUB_ID].default_lora_modules)
EXPERT_LORA_PARAMS = sorted(SUPPORTED_MODEL_INFO[HUB_ID].parameters)
EXPERT_PARAMS = ("gate_up_proj", "down_proj")
TP_BLOCK_CLASSES = {DeepseekV4Attention, DeepseekV4MLP}
EP_BLOCK_CLASSES = {DeepseekV4Experts}
# Norms and the embedding stay whole. Routers and hyper-connections are tiny and
# feed every head, so they stay whole too.
SKIPPED_CLASSES = {
    DeepseekV4RMSNorm,
    nn.Embedding,
    DeepseekV4TopKRouter,
    DeepseekV4HashRouter,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
}
# One weight of each part the TP plan shards, by stack FQN without LoRA wrappers.
TP_PROBE_WEIGHTS = (
    "layers.0.self_attn.q_b_proj.weight",
    "layers.0.self_attn.sinks",
    "layers.1.self_attn.o_a_proj.weight",
    "layers.2.self_attn.o_b_proj.weight",
    "layers.1.mlp.shared_experts.gate_proj.weight",
    "layers.2.mlp.shared_experts.down_proj.weight",
)
CHECKPOINT_LAYERS = 43
CHECKPOINT_EXPERTS = 256


def _config() -> DeepseekV4Config:
    config = DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_groups=O_GROUPS,
        o_lora_rank=4,
        n_routed_experts=EXPERTS,
        n_shared_experts=1,
        num_experts_per_tok=TOP_K,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        mlp_layer_types=["hash_moe", "moe", "moe"],
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        sliding_window=WINDOW,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=2,
        hc_mult=HC_MULT,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


class DeepseekV4BlockStack(nn.Module):
    """Sliding, compressed sparse, and heavily compressed DeepSeek-V4 decoder layers."""

    _no_split_modules = ("DeepseekV4DecoderLayer",)

    def __init__(self) -> None:
        super().__init__()
        config = _config()
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, index) for index in range(3)]
        )
        self.config = SimpleNamespace(model_type="deepseek_v4")


def _init_weights(stack: DeepseekV4BlockStack) -> DeepseekV4BlockStack:
    """Fill the ``torch.empty`` params and the hash table ``post_init`` would load."""
    with torch.no_grad():
        for param in stack.parameters():
            param.normal_(std=INIT_STD)
        table = torch.stack([torch.randperm(EXPERTS)[:TOP_K] for _ in range(VOCAB)])
        stack.layers[0].mlp.gate.tid2eid.copy_(table)
    return stack


def _with_moe_lora(module: nn.Module) -> nn.Module:
    inject_adapter_in_model(
        LoraConfig(
            r=2, target_modules=LORA_TARGETS, target_parameters=EXPERT_LORA_PARAMS
        ),
        module,
    )
    assert upgrade_moe_param_wrappers(module) == 3
    module.requires_grad_(True)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.1)
    return module


def _apply_liger_swaps() -> None:
    """Swap the class Liger reassigns for deepseek_v4, keeping the HF forwards."""
    modeling_deepseek_v4.DeepseekV4RMSNorm = type("LigerRMSNorm", (nn.Module,), {})


def _use_flex() -> None:
    """Run attention through the patched flex forward on a CPU flex kernel."""
    stock = DeepseekV4Attention.__dict__["forward"]
    DeepseekV4Attention.forward = flex_compressed_attention(stock)
    deepseek_v4_module.compile_friendly_flex_attention = dense_flex_attention


def _eager_mask() -> torch.Tensor:
    """Additive sliding-window causal mask for eager attention."""
    query = torch.arange(SEQ)[:, None]
    key = torch.arange(SEQ)[None, :]
    allowed = (key <= query) & (key > query - WINDOW)
    blocked = torch.finfo(torch.float32).min
    mask = torch.where(allowed, 0.0, blocked)[None, None]
    return mask.expand(BATCH, 1, SEQ, SEQ)


def _flex_mask(length: int = SEQ) -> BlockMask:
    return create_block_mask(
        lambda _b, _h, q_idx, kv_idx: (kv_idx <= q_idx) & (kv_idx > q_idx - WINDOW),
        B=BATCH,
        H=None,
        Q_LEN=length,
        KV_LEN=length,
        device="cpu",
    )


def _run_layers(
    stack: nn.Module,
    hidden: torch.Tensor,
    ids: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor | BlockMask | None,
) -> torch.Tensor:
    rotary = DeepseekV4RotaryEmbedding(_config())
    hc_input = hidden[..., 0, :]
    rope = {
        layer_type: rotary(hc_input, positions, layer_type=layer_type)
        for layer_type in ("main", "compress")
    }
    for layer in stack.layers:
        hidden = layer(
            hidden,
            input_ids=ids,
            position_embeddings=rope,
            position_ids=positions,
            attention_mask=mask,
        )
    return hidden


def _forward(
    stack: nn.Module,
    hidden: torch.Tensor,
    ids: torch.Tensor,
    flex: bool,
    local: slice | None = None,
) -> torch.Tensor:
    """Run every layer on the full sequence, or on the ``local`` CP shard of it."""
    positions = torch.arange(SEQ).expand(BATCH, SEQ)
    if local is not None:
        return _run_layers(
            stack, hidden[:, local], ids[:, local], positions[:, local], None
        )
    mask = _flex_mask() if flex else _eager_mask()
    return _run_layers(stack, hidden, ids, positions, mask)


def _assert_sequence_roundtrip(
    model: nn.Module, reference: nn.Module, cp: int, mesh: ParallelMesh
) -> None:
    """Every layer's gathered output under ContextParallelSequence matches the full sequence."""
    assert mesh.cp is not None
    # Flex rows pad to whole 128-token blocks.
    round_to = math.lcm(cp, 128)
    layout = ContextParallelSequence(
        pad_token_id=0,
        cp=cp,
        group=mesh.cp.get_group(),
        rank=int(mesh.cp.get_local_rank()),
        round_to=round_to,
    )
    ids = broadcast_tensor(torch.randint(1, VOCAB, (BATCH, 100)))
    mask = torch.ones_like(ids)
    # A right-padded row: the second ends early.
    ids[1, 90:] = 0
    mask[1, 90:] = 0
    table = broadcast_tensor(torch.randn(VOCAB, HC_MULT, HIDDEN))
    aligned, aligned_mask = pad_right_for_cp(ids, round_to, layout.pad_token_id, mask)
    length = aligned.shape[1] - 1
    assert length == round_to

    with layout.prepare_forward(ids, mask) as forward:
        shard_ids = forward.kwargs["input_ids"]
        output = _run_layers(
            model, table[shard_ids], shard_ids, forward.kwargs["position_ids"], None
        )
        gathered = gather_sequence_for_cp(output.flatten(2), mesh.cp.get_group())[
            :, : ids.shape[1]
        ]

    full_ids = aligned[:, :-1]
    positions = position_ids_from_mask(aligned_mask)[:, :length]
    expected = _run_layers(
        reference, table[full_ids], full_ids, positions, _flex_mask(length)
    )
    assert_exact(gathered, expected.flatten(2)[:, : ids.shape[1]], "cp sequence")


def _shard_data(dp_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hyper-connection streams, token ids, and an output probe for one data shard."""
    generator = torch.Generator().manual_seed(100 + dp_rank)
    hidden = torch.randn(BATCH, SEQ, HC_MULT, HIDDEN, generator=generator)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator)
    probe = torch.randn(BATCH, SEQ, HC_MULT, HIDDEN, generator=generator)
    return hidden, ids, probe


def _assert_tp_blocks_sharded(model: nn.Module, tp: int) -> None:
    """Each probed weight holds 1/tp of its elements per TP rank."""
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


def _assert_experts_on_ep(model: nn.Module, ep: int) -> None:
    """Expert weights split their expert axis ``ep`` ways."""
    params = [
        getattr(module, name)
        for module in model.modules()
        if isinstance(module, DeepseekV4Experts)
        for name in EXPERT_PARAMS
    ]
    assert len(params) == 3 * len(EXPERT_PARAMS)
    for param in params:
        assert isinstance(param, DTensor)
        assert on_ep_mesh(param)
        index = param.device_mesh.mesh_dim_names.index(MeshAxisName.EP)
        assert param.placements[index] == Shard(0)
        assert int(param.device_mesh.size(index)) == ep


def _worker(rank: int, world_size: int, cp: int, tp: int, ep: int, flex: bool) -> None:
    """Forward, reshard, LoRA gather and one SGD step match the unsharded stack."""
    # Arrange
    torch.manual_seed(0)
    if flex:
        _use_flex()
    stack = _with_moe_lora(_init_weights(DeepseekV4BlockStack()))
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    _apply_liger_swaps()
    dp = world_size // (cp * tp)
    mesh = parallel_mesh(world_size, cp, tp, ep)
    config = fsdp_config(cp, tp, ep)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    local = None
    if cp > 1:
        assert mesh.cp is not None
        if tp > 1:
            with torch.no_grad():
                _assert_sequence_roundtrip(stack, reference, cp, mesh)
        publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
        cp_rank = int(mesh.cp.get_local_rank())
        local = slice(cp_rank * SEQ // cp, (cp_rank + 1) * SEQ // cp)
    hidden, ids, probe = _shard_data(rank // (cp * tp))
    with torch.no_grad():
        expected_output = _forward(reference, hidden, ids, flex)
        if local is not None:
            expected_output = expected_output[:, local]
            probe = probe[:, local]
    reference_loss = sum(
        (_forward(reference, data, data_ids, flex) * data_probe).sum()
        for data, data_ids, data_probe in map(_shard_data, range(dp))
    )
    (reference_loss / dp).backward()
    expected = {
        plain(name): param.detach()
        if param.grad is None
        else param.detach() - param.grad
        for name, param in reference.named_parameters()
    }
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    optimizer = sgd(stack, lr=1.0)

    # Act
    output = _forward(stack, hidden, ids, flex, local)
    assert_resharded(stack, reference, tp, world_size // tp)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp)
    if ep > 1:
        _assert_experts_on_ep(stack, ep)
    assert_lora(stack, reference)
    loss = (output * probe).sum()
    step = runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=stack, max_grad_norm=1e6
    )
    clear_ulysses_params()

    # Assert
    assert_exact(output, expected_output, "output")
    assert step is not None
    assert len(expected) == sum(1 for _ in stack.parameters())
    errors: list[str] = []
    with torch.no_grad():
        for name, param in stack.named_parameters():
            value = param.full_tensor()
            target = expected[plain(name)].reshape(value.shape)
            try:
                assert_exact(value, target, name)
            except AssertionError as exc:
                errors.append(str(exc))
    if errors:
        raise AssertionError("\n".join(errors))


def _shard_keys_worker(rank: int, world_size: int) -> None:
    """Each shard's compressed keys, mask rows and gradients match the unsharded sequence's."""
    group = dist.group.WORLD
    local = slice(rank * SEQ // world_size, (rank + 1) * SEQ // world_size)
    positions = torch.arange(SEQ).unsqueeze(0)
    for layer_idx in (1, 2):
        # Arrange
        torch.manual_seed(layer_idx)
        attention = DeepseekV4Attention(_config(), layer_idx)
        with torch.no_grad():
            for param in attention.parameters():
                param.normal_(std=INIT_STD)
        broadcast_module(attention)
        hidden = broadcast_tensor(torch.randn(BATCH, SEQ, HIDDEN))
        q_residual = broadcast_tensor(torch.randn(BATCH, SEQ, 8))
        kv = broadcast_tensor(torch.randn(BATCH, 1, SEQ, 8))
        full_hidden = hidden.clone().requires_grad_(True)
        expected_keys, expected_mask = document_keys(
            attention,
            full_hidden,
            q_residual,
            kv,
            positions,
            None,
            _flex_mask(),
            (SEQ,),
        )
        probe = broadcast_tensor(torch.randn_like(expected_keys))
        (expected_keys * probe).sum().backward()
        expected_grads = [
            param.grad.clone()
            for param in attention.compressor.parameters()
            if param.grad is not None
        ]
        attention.zero_grad(set_to_none=True)
        shard_hidden = hidden[:, local].clone().requires_grad_(True)

        # Act
        with CollectiveRecorder() as recorder:
            keys, mask = sequence_shard_keys(
                attention,
                group,
                shard_hidden,
                q_residual[:, local],
                kv[:, :, local],
                positions[:, local],
            )
        (keys * probe).sum().div(world_size).backward()
        grads = [
            param.grad
            for param in attention.compressor.parameters()
            if param.grad is not None
        ]
        for grad in grads:
            dist.all_reduce(grad)

        # Assert
        label = f"layer {layer_idx}"
        assert recorder.kinds.count("all_gather") == 2, label
        assert_exact(keys, expected_keys, f"{label} keys")
        key_len = keys.shape[2]
        dense = create_mask(mask.mask_mod, BATCH, 1, SEQ // world_size, key_len, "cpu")
        expected_dense = create_mask(
            expected_mask.mask_mod, BATCH, 1, SEQ, key_len, "cpu"
        )
        assert torch.equal(dense, expected_dense[:, :, local]), label
        assert shard_hidden.grad is not None
        assert full_hidden.grad is not None
        assert_exact(
            shard_hidden.grad, full_hidden.grad[:, local], f"{label} input grad"
        )
        assert len(grads) == len(expected_grads) > 0, label
        for grad, expected_grad in zip(grads, expected_grads, strict=True):
            assert_exact(grad, expected_grad, f"{label} grad")


def _meta_load_worker(
    rank: int, world_size: int, cp: int, tp: int, ep: int, checkpoint_dir: str
) -> None:
    """A meta stack loads each rank's dp_shard x cp x tp x ep piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    dense = _init_weights(DeepseekV4BlockStack())
    if rank == 0:
        save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = DeepseekV4BlockStack()
    stack.config = DeepseekV4Config(name_or_path=checkpoint_dir)
    mesh = parallel_mesh(world_size, cp, tp, ep)
    config = fsdp_config(cp, tp, ep)

    # Act
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(stack, "cpu", config, parallel_mesh=mesh)

    # Assert
    expected = dict(dense.named_parameters())
    for name, param in stack.named_parameters():
        assert isinstance(param, DTensor), name
        assert_exact(param.full_tensor(), expected[plain(name)], name)
    expected_buffers = dict(dense.named_buffers())
    for name, buffer in stack.named_buffers():
        assert torch.equal(buffer, expected_buffers[plain(name)]), name
    _assert_tp_blocks_sharded(stack, tp)
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
    cp = 1
    torch.manual_seed(0)
    dense = _with_moe_lora(_init_weights(DeepseekV4BlockStack()))
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    saved, runtime, optimizer = lora_runtime(dense, world_size, cp, save_tp, ep)
    hidden, ids, probe = _shard_data(rank // save_tp)
    loss = (_forward(saved, hidden, ids, False) * probe).sum()
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


def _checkpoint_config() -> DeepseekV4Config:
    return DeepseekV4Config.from_dict(SUPPORTED_MODEL_INFO[HUB_ID].config)


def _meta_checkpoint_shard_worker(rank: int, world_size: int) -> None:
    config = _checkpoint_config()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    count = apply_tensor_parallel(model, mesh)

    # Every attention block and shared-expert MLP, plus the untied LM head.
    assert count == 2 * CHECKPOINT_LAYERS + 1
    sharded: dict[str, int] = {}
    for module in model.modules():
        name = type(module).__name__
        weight = {
            "DeepseekV4Attention": "q_b_proj",
            "DeepseekV4MLP": "down_proj",
        }.get(name)
        if weight is None:
            continue
        param = module.get_submodule(weight).weight
        assert isinstance(param, DTensor), name
        assert isinstance(param.placements[0], Shard), name
        sharded[name] = sharded.get(name, 0) + 1
    assert sharded == {
        "DeepseekV4Attention": CHECKPOINT_LAYERS,
        "DeepseekV4MLP": CHECKPOINT_LAYERS,
    }
    attention = model.model.layers[0].self_attn
    heads = int(config.num_attention_heads)
    assert attention.sinks.to_local().shape == (heads // world_size,)
    assert attention.q_b_proj.weight.to_local().shape[0] == (
        heads * int(config.head_dim) // world_size
    )
    o_a_rows = int(config.o_groups) * int(config.o_lora_rank)
    assert attention.o_a_proj.weight.to_local().shape[0] == o_a_rows // world_size
    assert attention.o_b_proj.weight.to_local().shape[1] == o_a_rows // world_size
    assert model.lm_head.weight.to_local().shape[0] == (
        int(config.vocab_size) // world_size
    )


class TestDeepseekV4CheckpointCoverage:
    def test_every_parameter_is_planned_or_skipped(self) -> None:
        # Arrange
        with torch.device("meta"):
            model = AutoModel.from_config(_checkpoint_config())
        modules = dict(model.named_modules())
        known = TP_BLOCK_CLASSES | EP_BLOCK_CLASSES | SKIPPED_CLASSES

        # Act
        owners: dict[str, type] = {}
        for name, _ in model.named_parameters():
            path = name.rsplit(".", 1)[0] if "." in name else ""
            while type(modules[path]) not in known and path:
                path = path.rpartition(".")[0]
            owners[name] = type(modules[path])

        # Assert
        unmatched = sorted(name for name, owner in owners.items() if owner not in known)
        assert unmatched == []
        planned = set(owners.values()) - SKIPPED_CLASSES
        assert planned == TP_BLOCK_CLASSES | EP_BLOCK_CLASSES

    def test_expert_parallel_finds_every_checkpoint_expert_block(self) -> None:
        # Arrange
        with torch.device("meta"):
            model = AutoModel.from_config(_checkpoint_config())

        # Act
        found = iter_packed_expert_modules(model)

        # Assert
        assert [type(module) for _, module in found] == [
            DeepseekV4Experts
        ] * CHECKPOINT_LAYERS
        assert packed_expert_counts(model) == [CHECKPOINT_EXPERTS] * CHECKPOINT_LAYERS

    def test_cp_plan_wraps_every_checkpoint_attention_once(self) -> None:
        # Arrange
        with torch.device("meta"):
            model = AutoModel.from_config(_checkpoint_config())
        mesh = MagicMock()

        # Act
        first = wrap_deepseek_v4_context_parallel(model, mesh)
        second = wrap_deepseek_v4_context_parallel(model, mesh)

        # Assert
        assert first == CHECKPOINT_LAYERS
        assert second == 0

    @pytest.mark.multi_rank
    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_tp_plan_shards_every_checkpoint_block(self, tp: int) -> None:
        run_ranks(_meta_checkpoint_shard_worker, tp, timeout=300)

    @pytest.mark.parametrize(
        ("tp", "match"), [(3, "num_attention_heads"), (16, "o_groups")]
    )
    def test_checkpoint_degrees_that_split_a_group_raise(
        self, tp: int, match: str
    ) -> None:
        with torch.device("meta"):
            model = AutoModel.from_config(_checkpoint_config())
        mesh = MagicMock()
        mesh.size.return_value = tp

        with pytest.raises(ValueError, match=match):
            apply_deepseek_v4_tensor_parallel(model, mesh)


class TestDeepseekV4Parallel:
    def test_tiny_stack_runs(self) -> None:
        stack = _with_moe_lora(_init_weights(DeepseekV4BlockStack()))
        hidden, ids, _ = _shard_data(0)

        output = _forward(stack, hidden, ids, False)

        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()

    def test_group_forward_installs_once(self) -> None:
        attention = DeepseekV4Attention(_config(), 0)
        _install_group_forward(attention, 1)
        installed = attention.forward

        _install_group_forward(attention, 1)

        assert attention.forward is installed

    def test_cp_wrapper_passes_the_group_only_while_a_shard_trains(self) -> None:
        # Arrange
        class Recorder(nn.Module):
            def forward(self, hidden: torch.Tensor, **kwargs: object) -> object:
                return kwargs.get(CONTEXT_PARALLEL_GROUP)

        attention = Recorder()
        mesh = MagicMock()
        install_deepseek_v4_context_parallel(attention, mesh)
        hidden = torch.zeros(1)

        # Act
        outside = attention(hidden)
        publish_varlen_offsets(torch.tensor([0, SEQ], dtype=torch.int32))
        try:
            inside = attention(hidden)
        finally:
            clear_ulysses_params()

        # Assert
        assert outside is None
        assert inside is mesh.get_group.return_value

    @pytest.mark.parametrize("cp", [2, 4])
    def test_shard_keys_match_the_whole_sequence_in_two_gathers(self, cp: int) -> None:
        run_ranks(_shard_keys_worker, cp, timeout=300)

    @pytest.mark.parametrize(
        ("tp", "match"), [(3, "num_attention_heads"), (8, "o_groups")]
    )
    def test_degrees_that_split_a_group_raise(self, tp: int, match: str) -> None:
        mesh = MagicMock()
        mesh.size.return_value = tp

        with pytest.raises(ValueError, match=match):
            apply_deepseek_v4_tensor_parallel(DeepseekV4BlockStack(), mesh)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "ep", "world", "flex"),
        [
            (1, 1, 1, 2, False),
            (1, 1, 2, 2, False),
            (1, 2, 2, 2, False),
            (1, 2, 2, 4, False),
            (1, 2, 2, 4, True),
            (2, 1, 1, 4, True),
            (2, 1, 2, 4, True),
            (2, 1, 4, 4, True),
            (4, 1, 1, 8, True),
            (2, 2, 2, 8, True),
            (2, 2, 4, 8, True),
        ],
        ids=[
            "fsdp",
            "ep2",
            "tp2-ep2-dp1",
            "tp2-ep2",
            "tp2-ep2-flex",
            "cp2",
            "cp2-ep2",
            "cp2-ep4",
            "cp4",
            "cp2-tp2-ep2",
            "cp2-tp2-ep4",
        ],
    )
    def test_forward_and_step_match_unsharded(
        self, cp: int, tp: int, ep: int, world: int, flex: bool
    ) -> None:
        run_ranks(_worker, world, cp, tp, ep, flex, timeout=600)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "ep", "world"),
        [(1, 2, 2, 4), (2, 2, 2, 8)],
        ids=["tp2-ep2", "cp2-tp2-ep2"],
    )
    def test_meta_load_fills_dp_cp_tp_ep_shards(
        self, cp: int, tp: int, ep: int, world: int, tmp_path: Path
    ) -> None:
        run_ranks(_meta_load_worker, world, cp, tp, ep, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(("save_tp", "load_tp"), [(2, 4), (4, 2)])
    def test_checkpoint_resumes_unchanged_at_another_tp_degree(
        self, save_tp: int, load_tp: int, tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save_tp, load_tp, 4, str(tmp_path), timeout=300)
