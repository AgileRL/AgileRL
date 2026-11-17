# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-, context- and expert-parallel MiMo-V2 blocks on a gloo mesh."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)
from transformers import AutoModelForCausalLM
from transformers.integrations import flex_attention as flex_integration
from transformers.models.mimo_v2_flash import modeling_mimo_v2_flash
from transformers.models.mimo_v2_flash.configuration_mimo_v2_flash import (
    MiMoV2FlashConfig,
)
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
    MiMoV2FlashDecoderLayer,
    MiMoV2FlashExperts,
    MiMoV2FlashForCausalLM,
    MiMoV2FlashMLP,
    MiMoV2FlashMoE,
    MiMoV2FlashRMSNorm,
    MiMoV2FlashRotaryEmbedding,
    MiMoV2FlashTopkRouter,
)

from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    position_ids_from_mask,
)
from agilerl.architectures import register_mimo_v2_native_loading
from agilerl.architectures.catalog import family_runtime
from agilerl.architectures.mimo_v2 import attention as mimo_v2_attention
from agilerl.architectures.mimo_v2.attention import (
    install_mimo_v2_fp32_lse_flex_attention,
)
from agilerl.architectures.mimo_v2.config import MiMoV2Config
from agilerl.architectures.mimo_v2.tensor_parallel import (
    apply_mimo_v2_tensor_parallel,
)
from agilerl.arena.memory.resolve.attention import (
    context_parallel_blocker,
    resolve_attn_implementation,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import ParallelMesh, fsdp_blocks
from agilerl.distributed.context_parallel import (
    attention_head_counts,
    clear_ulysses_params,
    substitute_hf_ulysses_attn,
)
from agilerl.distributed.context_parallel.sequence import (
    gather_sequence_for_cp,
    pad_right_for_cp,
    validate_cp_heads,
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
    unwrap,
)
from tests.rank_launch import run_ranks
from tests.test_architectures.mimo_v2.checkpoint_helpers import tiny_hub_config

CHECKPOINT = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
HIDDEN = 16
HEADS = 8
# The checkpoint's 1:2 full-to-sliding KV head ratio: tp=2 splits both, tp=4
# replicates the full-attention KV heads.
KV_HEADS = 2
SWA_KV_HEADS = 4
# Q/K and V head dims differ, as on the checkpoint (192 / 128).
HEAD_DIM = 8
V_HEAD_DIM = 4
EXPERTS = 4
VOCAB = 32
BATCH = 2
SEQ = 8
FULL_POSITIONS = torch.arange(SEQ).expand(BATCH, SEQ)
# Shorter than one cp=2 shard, so sliding layers mask inside and across shards.
WINDOW = 3
# Layer 0 is full attention with a dense MLP; layer 1 is sliding-window attention with sinks and MoE.
LAYER_TYPES = ("full_attention", "sliding_attention")
WINDOWS = {"full_attention": None, "sliding_attention": WINDOW}
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
EXPERT_LORA_PARAMS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
EXPERT_PARAMS = ("gate_up_proj", "down_proj")
TP_BLOCK_CLASSES = {MiMoV2FlashAttention, MiMoV2FlashMLP}
EP_BLOCK_CLASSES = {MiMoV2FlashExperts}
# The untied lm_head is the causal LM's own linear; the generic LM-head pass splits it.
LM_HEAD_CLASSES = {MiMoV2FlashForCausalLM}
# Norms and the token embedding stay whole; the router is tiny and stays whole.
SKIPPED_CLASSES = {MiMoV2FlashRMSNorm, nn.Embedding, MiMoV2FlashTopkRouter}
# One weight of each part the TP plan shards, by stack FQN without LoRA wrappers.
TP_PROBE_WEIGHTS = (
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.1.self_attn.q_proj.weight",
    "layers.1.self_attn.k_proj.weight",
    "layers.1.self_attn.v_proj.weight",
    "layers.1.self_attn.o_proj.weight",
    "layers.1.self_attn.sinks",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "lm_head.weight",
)
FULL_KV_WEIGHTS = (
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
)
CHECKPOINT_LAYERS = 48
CHECKPOINT_SWA_LAYERS = 39
CHECKPOINT_DENSE_LAYERS = 1
CHECKPOINT_EXPERTS = 256


def _config() -> MiMoV2Config:
    raw = tiny_hub_config()
    raw.update(
        {
            "hidden_size": HIDDEN,
            "num_attention_heads": HEADS,
            "swa_num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "swa_num_key_value_heads": SWA_KV_HEADS,
            "head_dim": HEAD_DIM,
            "swa_head_dim": HEAD_DIM,
            "v_head_dim": V_HEAD_DIM,
            "swa_v_head_dim": V_HEAD_DIM,
            "intermediate_size": 8,
            "moe_intermediate_size": 4,
            "n_routed_experts": EXPERTS,
            "num_experts_per_tok": 2,
            "num_hidden_layers": len(LAYER_TYPES),
            "hybrid_layer_pattern": [0, 1],
            "moe_layer_freq": [0, 1],
            "sliding_window": WINDOW,
            "vocab_size": VOCAB,
            "pad_token_id": 0,
        }
    )
    config = MiMoV2Config.from_dict(raw)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


class MiMoV2BlockStack(nn.Module):
    """A full-attention dense layer and a sliding-window MoE layer, then an untied LM head."""

    _no_split_modules = ("MiMoV2FlashDecoderLayer",)

    def __init__(self) -> None:
        super().__init__()
        config = _config()
        self.layers = nn.ModuleList(
            [MiMoV2FlashDecoderLayer(config, index) for index in range(2)]
        )
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.config = SimpleNamespace(model_type="mimo_v2", tie_word_embeddings=False)
        # Experts and sinks are torch.empty and the router is zeros outside post_init.
        with torch.no_grad():
            moe = self.layers[1].mlp
            moe.experts.gate_up_proj.normal_(std=0.5)
            moe.experts.down_proj.normal_(std=0.5)
            moe.gate.weight.normal_()
            moe.gate.e_score_correction_bias.normal_(std=0.1)
            self.layers[1].self_attn.sinks.normal_()


def _with_lora(module: nn.Module) -> nn.Module:
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
                param.normal_(std=0.1)
    return module


def _apply_liger_swaps(stack: nn.Module) -> None:
    """Swap the RMSNorm class and patch dense-MLP forwards as Liger does, keeping the HF math."""
    modeling_mimo_v2_flash.MiMoV2FlashRMSNorm = type(
        "LigerRMSNorm", (MiMoV2FlashRMSNorm,), {}
    )
    for module in stack.modules():
        if type(module).__name__ == "MiMoV2FlashMLP":
            module.forward = MethodType(MiMoV2FlashMLP.forward, module)


def _eager_mask(window: int | None, length: int = SEQ) -> torch.Tensor:
    """Additive causal mask, windowed on sliding layers, for eager attention."""
    query = torch.arange(length)[:, None]
    key = torch.arange(length)[None, :]
    allowed = key <= query
    if window is not None:
        allowed &= key > query - window
    blocked = torch.finfo(torch.float32).min
    return torch.where(allowed, 0.0, blocked)[None, None]


def _block_mask(window: int | None) -> BlockMask:
    def mask_mod(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        allowed = kv_idx <= q_idx
        if window is not None:
            allowed = allowed & (kv_idx > q_idx - window)
        return allowed

    return create_block_mask(mask_mod, None, None, SEQ, SEQ, device="cpu")


def _rope(
    positions: torch.Tensor, layer_type: str
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = MiMoV2FlashRotaryEmbedding(_config())
    return rotary(torch.zeros(*positions.shape, HIDDEN), positions, layer_type)


def _forward(
    stack: nn.Module,
    hidden: torch.Tensor,
    masks: dict[str, torch.Tensor | BlockMask | None],
    positions: torch.Tensor = FULL_POSITIONS,
) -> torch.Tensor:
    """Run both layers on the positions ``hidden`` holds."""
    for layer, layer_type in zip(stack.layers, LAYER_TYPES, strict=True):
        hidden = layer(
            hidden,
            attention_mask=masks[layer_type],
            position_embeddings=_rope(positions, layer_type),
        )
    return hidden


def _eager_masks() -> dict[str, torch.Tensor | BlockMask]:
    return {kind: _eager_mask(window) for kind, window in WINDOWS.items()}


def _token_log_probs(
    stack: nn.Module, hidden: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits from the untied head, and each position's log-prob of ``ids``."""
    logits = stack.lm_head(hidden)
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, ids.unsqueeze(-1))
    return logits, log_probs.squeeze(-1)


def _shard_data(dp_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Input hidden states, token ids and an output probe for one data shard."""
    generator = torch.Generator().manual_seed(100 + dp_rank)
    hidden = torch.randn(BATCH, SEQ, HIDDEN, generator=generator)
    ids = torch.randint(1, VOCAB, (BATCH, SEQ), generator=generator)
    probe = torch.randn(BATCH, SEQ, HIDDEN, generator=generator)
    return hidden, ids, probe


def _assert_tp_blocks_sharded(model: nn.Module, tp: int) -> None:
    """Each probed weight holds 1/tp of its elements per TP rank."""
    params = {
        plain(name).replace(".base_layer", ""): param
        for name, param in model.named_parameters()
    }
    # Full-attention layers have half the sliding layers' KV heads; tp=4 replicates them.
    split_kv = KV_HEADS % tp == 0
    sharded = TP_PROBE_WEIGHTS + (FULL_KV_WEIGHTS if split_kv else ())
    for suffix in sharded:
        param = params[suffix]
        assert isinstance(param, DTensor), suffix
        assert on_tp_mesh(param), suffix
        assert isinstance(param.placements[-1], Shard), suffix
        tp_local = param.redistribute(
            param.device_mesh,
            [Replicate()] * (param.device_mesh.ndim - 1) + [param.placements[-1]],
        ).to_local()
        assert tp_local.numel() * tp == param.numel(), suffix
    if split_kv:
        return
    for suffix in FULL_KV_WEIGHTS:
        param = params[suffix]
        assert isinstance(param, DTensor), suffix
        assert on_tp_mesh(param), suffix
        assert param.placements[-1].is_replicate(), suffix


def _assert_experts_on_ep(model: nn.Module, ep: int) -> None:
    """Packed expert weights split their expert axis ``ep`` ways."""
    params = [
        getattr(module, name)
        for module in model.modules()
        if isinstance(module, MiMoV2FlashExperts)
        for name in EXPERT_PARAMS
    ]
    assert len(params) == len(EXPERT_PARAMS)
    for param in params:
        assert isinstance(param, DTensor)
        assert on_ep_mesh(param)
        index = param.device_mesh.mesh_dim_names.index(MeshAxisName.EP)
        assert param.placements[index] == Shard(0)
        assert int(param.device_mesh.size(index)) == ep


def _dense(param: torch.Tensor) -> torch.Tensor:
    return param.full_tensor() if isinstance(param, DTensor) else param


def _materialize(stack: nn.Module, world_size: int, cp: int, tp: int, ep: int) -> Any:
    mesh = parallel_mesh(world_size, cp, tp, ep)
    # The FSDP mesh folds cp into dp_shard.
    assert mesh.dp_shard.size() == 2 * cp
    config = fsdp_config(cp, tp, ep)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    return mesh, config


def _use_ulysses_flex(stack: nn.Module, cp_group: dist.ProcessGroup) -> None:
    """Route the stack through Ulysses over the family flex backend and the dense CPU kernel."""
    install_mimo_v2_fp32_lse_flex_attention()
    mimo_v2_attention.compile_friendly_flex_attention = dense_flex_attention
    flex_integration.compile_friendly_flex_attention = dense_flex_attention
    unwrap(stack.layers[0]).self_attn.config._attn_implementation = "flex_attention"
    substitute_hf_ulysses_attn(cp_group, "flex_attention")


def _assert_sequence_roundtrip(
    model: nn.Module, reference: nn.Module, cp: int, mesh: ParallelMesh
) -> None:
    """Attention's gathered output under ContextParallelSequence matches the full sequence."""
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
    ids = broadcast_tensor(torch.randint(1, VOCAB, (BATCH, 5)))
    mask = torch.ones_like(ids)
    table = broadcast_tensor(torch.randn(VOCAB, HIDDEN))
    aligned, aligned_mask = pad_right_for_cp(ids, round_to, layout.pad_token_id, mask)
    length = aligned.shape[1] - 1
    assert length == round_to
    positions = position_ids_from_mask(aligned_mask)[:, :length]
    for index, layer_type in enumerate(LAYER_TYPES):
        block = model.layers[index]
        attention = unwrap(block).self_attn
        block.unshard()
        with layout.prepare_forward(ids, mask) as forward:
            local = table[forward.kwargs["input_ids"]]
            rope = _rope(forward.kwargs["position_ids"], layer_type)
            output = attention(local, rope, None)[0]
            gathered = gather_sequence_for_cp(output, mesh.cp.get_group())[
                :, : ids.shape[1]
            ]
        block.reshard()
        expected = reference.layers[index].self_attn(
            table[aligned[:, :-1]],
            _rope(positions, layer_type),
            _eager_mask(WINDOWS[layer_type], length),
        )[0]
        assert_exact(gathered, expected[:, : ids.shape[1]], f"cp sequence {layer_type}")


def _worker(
    rank: int,
    world_size: int,
    cp: int,
    tp: int,
    ep: int,
    liger_first: bool = False,
) -> None:
    """Forward, reshard, LoRA gather and one SGD step match the unsharded stack."""
    # Arrange
    torch.manual_seed(0)
    if liger_first:
        modeling_mimo_v2_flash.MiMoV2FlashRMSNorm = type(
            "LigerRMSNorm", (MiMoV2FlashRMSNorm,), {}
        )
    stack = _with_lora(MiMoV2BlockStack())
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    if liger_first:
        assert type(stack.layers[0].input_layernorm).__name__ == "LigerRMSNorm"
    _apply_liger_swaps(stack)
    dp = world_size // (cp * tp)
    mesh, config = _materialize(stack, world_size, cp, tp, ep)
    local = slice(None)
    masks: dict[str, torch.Tensor | BlockMask | None] = _eager_masks()
    if cp > 1:
        assert mesh.cp is not None
        _use_ulysses_flex(stack, mesh.cp.get_group())
        with torch.no_grad():
            _assert_sequence_roundtrip(stack, reference, cp, mesh)
        publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
        cp_rank = int(mesh.cp.get_local_rank())
        local = slice(cp_rank * SEQ // cp, (cp_rank + 1) * SEQ // cp)
        # Ulysses builds the masks for the gathered sequence.
        masks = dict.fromkeys(WINDOWS)
    hidden, ids, probe = _shard_data(rank // (cp * tp))
    with torch.no_grad():
        expected_output = _forward(reference, hidden, _eager_masks())
        expected_logits, expected_log_probs = _token_log_probs(
            reference, expected_output, ids
        )
    reference_loss = 0
    for data, data_ids, data_probe in map(_shard_data, range(dp)):
        data_output = _forward(reference, data, _eager_masks())
        _, data_log_probs = _token_log_probs(reference, data_output, data_ids)
        reference_loss = (
            reference_loss + (data_output * data_probe).sum() + data_log_probs.sum()
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
    output = _forward(stack, hidden[:, local], masks, FULL_POSITIONS[:, local])
    logits, log_probs = _token_log_probs(stack, output, ids[:, local])
    assert_resharded(stack.layers, reference.layers, tp, world_size // tp)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp)
    if ep > 1:
        _assert_experts_on_ep(stack, ep)
    assert_lora(stack, reference)
    loss = (output * probe[:, local]).sum() + log_probs.sum()
    step = runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=stack, max_grad_norm=1e6
    )
    clear_ulysses_params()

    # Assert
    assert_exact(output, expected_output[:, local], "output")
    assert_exact(logits, expected_logits[:, local], "logits")
    assert_exact(log_probs, expected_log_probs[:, local], "log-probs")
    assert step is not None
    assert len(expected) == sum(1 for _ in stack.parameters())
    errors: list[str] = []
    with torch.no_grad():
        for name, param in stack.named_parameters():
            value = _dense(param)
            target = expected[plain(name)].reshape(value.shape)
            try:
                assert_exact(value, target, name)
            except AssertionError as exc:
                errors.append(str(exc))
    if errors:
        raise AssertionError("\n".join(errors))


def _eager_flex(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    training: bool,
    **kwargs: Any,
) -> Any:
    """Uncompiled flex: compiled flex on CPU cannot return the log-sum-exp."""
    return flex_attention(query, key, value, **kwargs)


def _flex_worker(rank: int, world_size: int, tp: int) -> None:
    """The family flex backend on TP-local heads and sinks matches unsharded eager attention."""
    # Arrange
    torch.manual_seed(0)
    stack = _with_lora(MiMoV2BlockStack())
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    _apply_liger_swaps(stack)
    _materialize(stack, world_size, 1, tp, tp)
    install_mimo_v2_fp32_lse_flex_attention()
    mimo_v2_attention.compile_friendly_flex_attention = _eager_flex
    flex_integration.compile_friendly_flex_attention = _eager_flex
    unwrap(stack.layers[0]).self_attn.config._attn_implementation = "flex_attention"
    hidden, _, _ = _shard_data(rank // tp)
    flex_masks = {kind: _block_mask(window) for kind, window in WINDOWS.items()}

    # Act
    # CPU flex has no backward and rejects inputs that require grad.
    with torch.no_grad():
        output = _forward(stack, hidden, flex_masks)
        expected = _forward(reference, hidden, _eager_masks())

    # Assert
    assert reference.layers[0].self_attn.config._attn_implementation == "eager"
    _assert_tp_blocks_sharded(stack, tp)
    assert_exact(output, expected, "flex output")


def _meta_load_worker(
    rank: int, world_size: int, cp: int, tp: int, ep: int, checkpoint_dir: str
) -> None:
    """A meta stack loads each rank's dp_shard x cp x tp x ep piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    dense = MiMoV2BlockStack()
    if rank == 0:
        save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = MiMoV2BlockStack()
    stack.config = MiMoV2FlashConfig(
        name_or_path=checkpoint_dir, tie_word_embeddings=False
    )
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
    buffers = dict(dense.named_buffers())
    for name, buffer in stack.named_buffers():
        assert_exact(buffer, buffers[plain(name)], name)
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
    dense = _with_lora(MiMoV2BlockStack())
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    saved, runtime, optimizer = lora_runtime(dense, world_size, cp, save_tp, ep)
    hidden, ids, probe = _shard_data(rank // save_tp)
    output = _forward(saved, hidden, _eager_masks())
    _, log_probs = _token_log_probs(saved, output, ids)
    loss = (output * probe).sum() + log_probs.sum()
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


def _checkpoint_model() -> nn.Module:
    register_mimo_v2_native_loading()
    config = MiMoV2Config.from_dict(SUPPORTED_MODEL_INFO[CHECKPOINT].config)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def _meta_checkpoint_shard_worker(rank: int, world_size: int) -> None:
    model = _checkpoint_model()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    counts: dict[str, int] = {}
    for module in model.modules():
        name = type(module).__name__
        counts[name] = counts.get(name, 0) + 1

    count = apply_tensor_parallel(model, mesh)

    # Every attention block, the dense MLP, and the untied LM head.
    assert counts["MiMoV2FlashAttention"] == CHECKPOINT_LAYERS
    assert counts["MiMoV2FlashMLP"] == CHECKPOINT_DENSE_LAYERS
    assert count == CHECKPOINT_LAYERS + CHECKPOINT_DENSE_LAYERS + 1
    sharded = {"MiMoV2FlashAttention": 0, "MiMoV2FlashMLP": 0}
    for module in model.modules():
        name = type(module).__name__
        if name == "MiMoV2FlashAttention":
            weights = [module.q_proj.weight, module.o_proj.weight]
            if module.sinks is not None:
                weights.append(module.sinks)
        elif name == "MiMoV2FlashMLP":
            weights = [module.gate_proj.weight, module.down_proj.weight]
        else:
            continue
        for weight in weights:
            assert isinstance(weight, DTensor), name
            assert weight.to_local().numel() * world_size == weight.numel(), name
        sharded[name] += 1
    assert sharded == {
        "MiMoV2FlashAttention": CHECKPOINT_LAYERS,
        "MiMoV2FlashMLP": CHECKPOINT_DENSE_LAYERS,
    }
    head = model.lm_head.weight
    assert isinstance(head, DTensor)
    assert head.to_local().numel() * world_size == head.numel()
    sinks = model.model.layers[1].self_attn.sinks
    assert sinks.to_local().shape == (64 // world_size,)
    for layer in model.model.layers:
        if isinstance(layer.mlp, MiMoV2FlashMoE):
            assert not isinstance(layer.mlp.gate.weight, DTensor)
            assert not isinstance(layer.mlp.experts.gate_up_proj, DTensor)
    assert not isinstance(model.model.embed_tokens.weight, DTensor)


class TestMiMoV2CheckpointCoverage:
    def test_every_parameter_is_planned_or_skipped(self) -> None:
        # Arrange
        model = _checkpoint_model()
        modules = dict(model.named_modules())
        known = TP_BLOCK_CLASSES | EP_BLOCK_CLASSES | LM_HEAD_CLASSES | SKIPPED_CLASSES

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
        root_owned = sorted(
            name for name, owner in owners.items() if owner in LM_HEAD_CLASSES
        )
        assert root_owned == ["lm_head.weight"]
        assert model.config.tie_word_embeddings is False

    def test_trainer_builds_the_native_causal_lm(self) -> None:
        model = _checkpoint_model()

        assert type(model) is MiMoV2FlashForCausalLM
        assert family_runtime(CHECKPOINT).tensor_parallel_plan == (
            "agilerl.architectures.mimo_v2.tensor_parallel:MIMO_V2_TENSOR_PARALLEL_PLAN"
        )
        assert resolve_attn_implementation("mimo_v2", True) == "flex_attention"

    def test_expert_parallel_finds_every_checkpoint_expert_block(self) -> None:
        # Arrange
        model = _checkpoint_model()

        # Act
        found = iter_packed_expert_modules(model)

        # Assert
        moe_layers = CHECKPOINT_LAYERS - CHECKPOINT_DENSE_LAYERS
        assert [type(module) for _, module in found] == [
            MiMoV2FlashExperts
        ] * moe_layers
        assert packed_expert_counts(model) == [CHECKPOINT_EXPERTS] * moe_layers

    def test_legal_tp_degrees_follow_every_layer(self) -> None:
        # Arrange
        model = _checkpoint_model()
        attention = [layer.self_attn for layer in model.model.layers]
        heads = {
            (module.config.num_attention_heads, module.num_key_value_groups)
            for module in attention
        }
        sliding = sum(module.sinks is not None for module in attention)
        width = model.config.intermediate_size

        # Act
        legal = [
            tp
            for tp in range(2, 9)
            if all(
                num_q % tp == 0
                and (groups % (num_q // tp) == 0 or (num_q // tp) % groups == 0)
                for num_q, groups in heads
            )
            and width % tp == 0
        ]

        # Assert
        assert heads == {(64, 16), (64, 8)}
        assert sliding == CHECKPOINT_SWA_LAYERS
        assert {(module.head_dim, module.v_head_dim) for module in attention} == {
            (192, 128)
        }
        assert legal == [2, 4, 8]

    def test_context_parallel_runs_generic_ulysses_flex(self) -> None:
        # Arrange
        model = _checkpoint_model()
        backend = resolve_attn_implementation("mimo_v2", True)

        # Act
        heads, kv_heads = attention_head_counts(model)

        # Assert
        assert family_runtime(CHECKPOINT).context_parallel_plan is None
        assert context_parallel_blocker(backend, 2, "mimo_v2") is None
        assert (heads, kv_heads) == (64, 4)
        for cp in (2, 4, 8):
            validate_cp_heads(heads, kv_heads, cp)

    def test_legal_cp_tp_meshes_split_local_heads(self) -> None:
        # Arrange
        attention = [layer.self_attn for layer in _checkpoint_model().model.layers]
        kv_heads = {
            module.k_proj.out_features // module.head_dim for module in attention
        }
        meshes = [
            (cp, tp) for tp in (1, 2, 4, 8) for cp in (1, 2, 4, 8) if (cp, tp) != (1, 1)
        ]

        # Act
        legal = []
        for cp, tp in meshes:
            try:
                for kv in kv_heads:
                    validate_cp_heads(64 // tp, max(kv // tp, 1), cp)
            except ValueError:
                continue
            legal.append((cp, tp))

        # Assert
        assert kv_heads == {4, 8}
        assert legal == meshes

    def test_heads_that_do_not_divide_tp_raise(self) -> None:
        mesh = MagicMock()
        mesh.size.return_value = 3

        with pytest.raises(ValueError, match="num_attention_heads"):
            apply_mimo_v2_tensor_parallel(MiMoV2BlockStack(), mesh)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize("world", [2, 8])
    def test_tp_plan_shards_every_checkpoint_block(self, world: int) -> None:
        run_ranks(_meta_checkpoint_shard_worker, world, timeout=300)


class TestMiMoV2Parallel:
    def test_tiny_stack_runs(self) -> None:
        stack = _with_lora(MiMoV2BlockStack())
        hidden, ids, _ = _shard_data(0)

        output = _forward(stack, hidden, _eager_masks())
        _, log_probs = _token_log_probs(stack, output, ids)

        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()
        assert torch.isfinite(log_probs).all()

    def test_tiny_stack_matches_checkpoint_layout(self) -> None:
        stack = MiMoV2BlockStack()

        attention = [layer.self_attn for layer in stack.layers]

        assert [module.sliding_window for module in attention] == [None, WINDOW]
        assert [module.num_key_value_groups for module in attention] == [4, 2]
        assert attention[0].sinks is None
        assert attention[1].sinks.shape == (HEADS,)
        assert type(stack.layers[0].mlp) is MiMoV2FlashMLP
        assert type(stack.layers[1].mlp.experts) is MiMoV2FlashExperts

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "ep", "world"),
        [
            (1, 1, 1, 2),
            (1, 1, 2, 2),
            (1, 2, 2, 4),
            (1, 4, 4, 8),
            (2, 1, 1, 4),
            (2, 1, 2, 4),
            # tp=2 leaves one full-attention KV head per rank; Ulysses repeats it.
            (2, 2, 2, 8),
            # Four ranks share two full-attention KV heads.
            (4, 1, 1, 8),
        ],
    )
    def test_forward_and_step_match_unsharded(
        self, cp: int, tp: int, ep: int, world: int
    ) -> None:
        run_ranks(_worker, world, cp, tp, ep, timeout=300)

    @pytest.mark.multi_rank
    def test_liger_classes_built_before_wrap_are_sharded(self) -> None:
        run_ranks(_worker, 4, 1, 2, 2, True, timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(("tp", "world"), [(2, 4), (4, 8)])
    def test_family_flex_attention_matches_unsharded(self, tp: int, world: int) -> None:
        run_ranks(_flex_worker, world, tp, timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "ep", "world"), [(1, 2, 2, 4), (2, 2, 2, 8)], ids=["tp", "cp_tp"]
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
