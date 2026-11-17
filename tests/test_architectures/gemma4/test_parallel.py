# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor- and context-parallel Gemma 4 blocks on a gloo mesh."""

from __future__ import annotations

import copy
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, inject_adapter_in_model
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from transformers import AutoModelForImageTextToText
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4 import modeling_gemma4
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4Config,
    Gemma4TextConfig,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel,
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextMLP,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    Gemma4VisionModel,
    repeat_kv,
)

from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    position_ids_from_mask,
)
from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS, family_runtime
from agilerl.architectures.gemma4 import tensor_parallel as gemma4_tensor_parallel
from agilerl.architectures.gemma4.tensor_parallel import apply_gemma4_tensor_parallel
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import ParallelMesh, fsdp_blocks
from agilerl.distributed.context_parallel import (
    clear_ulysses_params,
    substitute_hf_ulysses_attn,
)
from agilerl.distributed.context_parallel.sequence import (
    CP_SCAN_CHUNK,
    pad_right_for_cp,
    shard_for_cp,
    validate_cp_heads,
)
from agilerl.distributed.expert_parallel import iter_packed_expert_modules
from agilerl.distributed.fsdp import (
    materialize_dtensors,
    materialize_fsdp2_from_cpu_state,
)
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.distributed.tensor_parallel import apply_tensor_parallel
from tests.parallel_helpers import (
    assert_exact,
    assert_lora,
    assert_resharded,
    assert_same_state,
    broadcast_module,
    broadcast_tensor,
    cpu_flex_attention,
    fsdp_config,
    lora_runtime,
    on_tp_mesh,
    parallel_mesh,
    plain,
    publish_varlen_offsets,
    sgd,
    unwrap,
)
from tests.rank_launch import run_ranks
from tests.test_architectures.gemma4.parallel_helpers import model_config

CHECKPOINT = "google/gemma-4-E4B-it"
# Float32 Gemma 4 reference misses 1e-5 against float64 by ~1e-3, so this file compares in float64.
DTYPE = torch.float64
FSDP_DTYPE = "float64"
HIDDEN = 16
HEADS = 8
# The checkpoint's KV head count: cp=2 splits it, cp=4 replicates it.
KV_HEADS = 2
SLIDING_HEAD_DIM = 4
FULL_HEAD_DIM = 8
PER_LAYER_DIM = 4
VOCAB = 32
# Gemma 4's final logit softcap.
SOFTCAP = 30.0
BATCH = 2
SEQ = 8
# Shorter than one CP shard, so sliding layers mask inside and across shards.
WINDOW = 3
# The last two layers reuse the first two layers' K/V, as the checkpoint's last 18 do.
LAYER_TYPES = (
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
)
WINDOWS = {"sliding_attention": WINDOW, "full_attention": None}
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
# Legal tp degrees from 8 q / 2 kv heads on every layer and the 10240-wide MLP.
LEGAL_TP = [2, 4, 8]
# Legal (cp, tp) with cp or tp above 1: Ulysses splits the query heads left on each TP rank.
LEGAL_CP_TP = [(2, 1), (4, 1), (8, 1), (1, 2), (2, 2), (4, 2), (1, 4), (2, 4), (1, 8)]
# Ulysses flex exchanges sequence for heads around attention.
CP_BLOCK_CLASSES = {Gemma4TextAttention}
# Token-wise blocks see only their own positions: the MLP, and the per-layer
# input gate and projections on the decoder layer and text model.
TOKENWISE_CLASSES = {Gemma4TextMLP, Gemma4TextDecoderLayer, Gemma4TextModel}
TP_BLOCK_CLASSES = {Gemma4TextAttention, Gemma4TextMLP}
# The per-layer input gate and projections stay whole on every TP rank.
TP_WHOLE_CLASSES = {Gemma4TextDecoderLayer, Gemma4TextModel}
# Norms stay whole. The token and per-layer embeddings share this class and the
# tied head; TP splits their columns outside the block plan.
SKIPPED_CLASSES = {Gemma4RMSNorm, Gemma4TextScaledWordEmbedding}
# cp > 1 on this VL checkpoint requires strip_multimodal_towers=True, so the
# towers never run in a context-parallel forward.
TOWER_CLASSES = {Gemma4VisionModel, Gemma4AudioModel, Gemma4MultimodalEmbedder}
# One weight of each part the TP plan shards, by stack FQN without LoRA wrappers:
# sliding and full attention, a KV-shared layer, the MLP, and the per-layer embedding.
TP_PROBE_WEIGHTS = (
    "embed_tokens_per_layer.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.1.self_attn.o_proj.weight",
    "layers.2.self_attn.q_proj.weight",
    "layers.3.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.1.mlp.down_proj.weight",
)
KV_WEIGHTS = (
    "layers.0.self_attn.k_proj.weight",
    "layers.1.self_attn.v_proj.weight",
)
# Per-head norms each TP rank keeps whole.
TP_REPLICATED_WEIGHTS = (
    "layers.0.self_attn.q_norm.weight",
    "layers.1.self_attn.k_norm.weight",
    "layers.2.self_attn.q_norm.weight",
)


def _config() -> Gemma4TextConfig:
    config = Gemma4TextConfig(
        hidden_size=HIDDEN,
        intermediate_size=8,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=SLIDING_HEAD_DIM,
        global_head_dim=FULL_HEAD_DIM,
        sliding_window=WINDOW,
        layer_types=list(LAYER_TYPES),
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=PER_LAYER_DIM,
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
    )
    config._attn_implementation = "eager"
    return config


class Gemma4BlockStack(nn.Module):
    """Gemma 4 text decoder layers between a tied token embedding and LM head.

    Sliding and full layers, then two that share their K/V, fed by a per-layer embedding.
    """

    _no_split_modules = ("Gemma4TextDecoderLayer",)

    def __init__(self) -> None:
        super().__init__()
        config = _config()
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            VOCAB, HIDDEN, config.pad_token_id, embed_scale=HIDDEN**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, index) for index in range(len(LAYER_TYPES))]
        )
        self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
            VOCAB,
            len(LAYER_TYPES) * PER_LAYER_DIM,
            config.pad_token_id,
            embed_scale=PER_LAYER_DIM**0.5,
        )
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.tie_weights()
        self.config = SimpleNamespace(
            model_type="gemma4_text", tie_word_embeddings=True
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def tie_weights(self) -> None:
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self, hidden: torch.Tensor, ids: torch.Tensor, per_layer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token and per-layer embedding lookups added to their context terms."""
        looked_up = self.embed_tokens_per_layer(ids)
        per_layer = per_layer + looked_up.reshape(
            *ids.shape, len(LAYER_TYPES), PER_LAYER_DIM
        )
        return hidden + self.embed_tokens(ids), per_layer


def _token_log_probs(
    stack: nn.Module, hidden: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softcapped logits from the tied head, and each position's log-prob of ``ids``."""
    logits = torch.tanh(stack.lm_head(hidden) / SOFTCAP) * SOFTCAP
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, ids.unsqueeze(-1))
    return logits, log_probs.squeeze(-1)


def _with_lora(module: nn.Module) -> nn.Module:
    inject_adapter_in_model(LoraConfig(r=2, target_modules=LORA_TARGETS), module)
    module.requires_grad_(True)
    # FSDP leaves the tied embedding out of its units, so it stays frozen as under LoRA.
    module.embed_tokens.requires_grad_(False)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.1)
    return module


def _apply_liger_swaps() -> None:
    """Swap the classes Liger reassigns for gemma4, keeping the HF forwards."""
    norm = type("LigerRMSNormForGemma4", (Gemma4RMSNorm,), {})
    mlp = type("LigerGEGLUMLPForGemma4", (Gemma4TextMLP,), {})
    modeling_gemma4.Gemma4RMSNorm = norm
    modeling_gemma4.Gemma4TextMLP = mlp
    gemma4_tensor_parallel.Gemma4TextMLP = mlp


def _use_ulysses_flex(stack: nn.Module, cp_group: dist.ProcessGroup) -> None:
    """Route the stack's attention through Ulysses over the CPU flex stand-in."""
    unwrap(stack.layers[0]).self_attn.config._attn_implementation = "flex_attention"
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = cpu_flex_attention
    substitute_hf_ulysses_attn(cp_group, "flex_attention")


def _eager_mask(window: int | None, length: int = SEQ) -> torch.Tensor:
    """Additive causal mask, windowed on sliding layers, for eager attention."""
    query = torch.arange(length)[:, None]
    key = torch.arange(length)[None, :]
    allowed = key <= query
    if window is not None:
        allowed &= key > query - window
    blocked = torch.finfo(torch.float32).min
    return torch.where(allowed, 0.0, blocked)[None, None]


def _float64_eager_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hugging Face Gemma 4 eager attention with the softmax kept in the input dtype."""
    if scaling is None:
        scaling = module.head_dim**-0.5
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    weights = (query @ key.transpose(2, 3)) * scaling
    if softcap is not None:
        weights = torch.tanh(weights / softcap) * softcap
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = torch.softmax(weights, dim=-1)
    weights = nn.functional.dropout(weights, p=dropout, training=module.training)
    return (weights @ value).transpose(1, 2).contiguous(), weights


def _float64_rms_norm(self: Gemma4RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    """Hugging Face Gemma 4 RMSNorm in the input dtype."""
    normed = self._norm(hidden_states)
    return normed * self.weight if self.with_scale else normed


def _use_float64() -> None:
    """Build in float64; HF Gemma 4 rounds its norms and eager softmax through float32."""
    torch.set_default_dtype(DTYPE)
    modeling_gemma4.eager_attention_forward = _float64_eager_attention
    Gemma4RMSNorm.forward = _float64_rms_norm


def _rope(
    positions: torch.Tensor, layer_type: str
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = Gemma4TextRotaryEmbedding(_config())
    return rotary(torch.zeros(*positions.shape, HIDDEN), positions, layer_type)


def _forward(
    stack: nn.Module, hidden: torch.Tensor, ids: torch.Tensor, per_layer: torch.Tensor
) -> torch.Tensor:
    """Run every layer on the full sequence with eager masks."""
    positions = torch.arange(SEQ).expand(BATCH, SEQ)
    hidden, per_layer = stack(hidden, ids, per_layer)
    shared_kv = UserDict()
    for index, (layer, layer_type) in enumerate(
        zip(stack.layers, LAYER_TYPES, strict=True)
    ):
        hidden = layer(
            hidden,
            per_layer[:, :, index],
            shared_kv_states=shared_kv,
            position_embeddings=_rope(positions, layer_type),
            attention_mask=_eager_mask(WINDOWS[layer_type]),
        )
    return hidden


def _forward_cp_shard(
    stack: nn.Module,
    hidden: torch.Tensor,
    ids: torch.Tensor,
    per_layer: torch.Tensor,
    local: slice,
) -> torch.Tensor:
    """Run every layer on one CP sequence shard; Ulysses flex builds the masks."""
    positions = torch.arange(SEQ).expand(BATCH, SEQ)[:, local]
    hidden, per_layer = stack(hidden, ids, per_layer)
    shared_kv = UserDict()
    for index, (layer, layer_type) in enumerate(
        zip(stack.layers, LAYER_TYPES, strict=True)
    ):
        hidden = layer(
            hidden,
            per_layer[:, :, index],
            shared_kv_states=shared_kv,
            position_embeddings=_rope(positions, layer_type),
            attention_mask=None,
        )
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
    length = aligned.shape[1] - 1
    positions = position_ids_from_mask(aligned_mask)[:, :length]
    for index in (0, 1):
        layer_type = LAYER_TYPES[index]
        block = model.layers[index]
        attention = unwrap(block).self_attn
        block.unshard()
        with layout.prepare_forward(ids, mask) as forward:
            local = table[forward.kwargs["input_ids"]]
            rope = _rope(forward.kwargs["position_ids"], layer_type)
            output = attention(local, rope, None, shared_kv_states=UserDict())[0]
            restored = forward.scoring_hidden(output)
        block.reshard()
        expected = reference.layers[index].self_attn(
            table[aligned[:, :-1]],
            _rope(positions, layer_type),
            _eager_mask(WINDOWS[layer_type], length),
            shared_kv_states=UserDict(),
        )[0]
        expected = shard_for_cp(expected, layout.rank, cp)
        assert_exact(restored, expected, f"cp sequence {layer_type}")


def _shard_data(
    dp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hidden states, token ids, per-layer context and an output probe for one data shard."""
    generator = torch.Generator().manual_seed(100 + dp_rank)
    hidden = torch.randn(BATCH, SEQ, HIDDEN, generator=generator)
    ids = torch.randint(1, VOCAB, (BATCH, SEQ), generator=generator)
    per_layer = torch.randn(
        BATCH, SEQ, len(LAYER_TYPES), PER_LAYER_DIM, generator=generator
    )
    probe = torch.randn(BATCH, SEQ, HIDDEN, generator=generator)
    return hidden, ids, per_layer, probe


def _assert_tp_blocks_sharded(model: nn.Module, tp: int) -> None:
    """Each probed weight holds 1/tp of its elements per TP rank; norms stay whole."""
    params = {
        plain(name).replace(".base_layer", ""): param
        for name, param in model.named_parameters()
    }
    # Two KV heads split for tp=2; at tp=4 each pair of ranks reads one whole head.
    sharded = TP_PROBE_WEIGHTS + (KV_WEIGHTS if KV_HEADS % tp == 0 else ())
    replicated = TP_REPLICATED_WEIGHTS + (() if KV_HEADS % tp == 0 else KV_WEIGHTS)
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
    for suffix in replicated:
        param = params[suffix]
        assert isinstance(param, DTensor), suffix
        assert on_tp_mesh(param), suffix
        assert param.placements[-1].is_replicate(), suffix


def _assert_tied_embedding(stack: nn.Module, reference: nn.Module, tp: int) -> None:
    """Embedding and head share one weight; TP splits its hidden columns."""
    weight = stack.embed_tokens.weight
    assert stack.lm_head.weight is weight
    if tp == 1:
        assert not isinstance(weight, DTensor)
        assert_exact(weight, reference.embed_tokens.weight, "embed_tokens")
        return
    assert isinstance(weight, DTensor)
    assert on_tp_mesh(weight)
    assert weight.device_mesh.size() == tp
    assert weight.placements == (Shard(1),)
    assert weight.to_local().numel() * tp == weight.numel()
    assert_exact(weight.full_tensor(), reference.embed_tokens.weight, "embed_tokens")


def _dense(param: torch.Tensor) -> torch.Tensor:
    return param.full_tensor() if isinstance(param, DTensor) else param


def _worker(
    rank: int, world_size: int, cp: int, tp: int, liger_first: bool = False
) -> None:
    """Forward, reshard, LoRA gather and one SGD step match the unsharded stack."""
    # Arrange
    _use_float64()
    torch.manual_seed(0)
    if liger_first:
        _apply_liger_swaps()
    stack = _with_lora(Gemma4BlockStack())
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    if liger_first:
        assert type(unwrap(stack.layers[0]).mlp).__name__ == "LigerGEGLUMLPForGemma4"
    else:
        _apply_liger_swaps()
    dp = world_size // (cp * tp)
    mesh = parallel_mesh(world_size, cp, tp, 1)
    # The FSDP mesh folds cp into dp_shard.
    assert mesh.dp_shard.size() == 2 * cp
    config = fsdp_config(cp, tp, 1, FSDP_DTYPE)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    local = slice(None)
    if cp > 1:
        assert mesh.cp is not None
        _use_ulysses_flex(stack, mesh.cp.get_group())
        with torch.no_grad():
            _assert_sequence_roundtrip(stack, reference, cp, mesh)
        publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
        cp_rank = int(mesh.cp.get_local_rank())
        local = slice(cp_rank * SEQ // cp, (cp_rank + 1) * SEQ // cp)
    hidden, ids, per_layer, probe = _shard_data(rank // (cp * tp))
    with torch.no_grad():
        expected_output = _forward(reference, hidden, ids, per_layer)
        expected_logits, expected_log_probs = _token_log_probs(
            reference, expected_output, ids
        )
    reference_loss = 0
    for data, data_ids, data_per_layer, data_probe in map(_shard_data, range(dp)):
        data_output = _forward(reference, data, data_ids, data_per_layer)
        _, data_log_probs = _token_log_probs(reference, data_output, data_ids)
        reference_loss = (
            reference_loss + (data_output * data_probe).sum() + data_log_probs.sum()
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
    if cp > 1:
        output = _forward_cp_shard(
            stack, hidden[:, local], ids[:, local], per_layer[:, local], local
        )
    else:
        output = _forward(stack, hidden, ids, per_layer)
    logits, log_probs = _token_log_probs(stack, output, ids[:, local])
    with torch.no_grad(), materialize_dtensors(stack.lm_head.weight) as (head,):
        fused_logits = torch.tanh(output @ head.T / SOFTCAP) * SOFTCAP
    for part in ("layers", "embed_tokens_per_layer"):
        assert_resharded(
            getattr(stack, part), getattr(reference, part), tp, world_size // tp
        )
    _assert_tied_embedding(stack, reference, tp)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp)
    assert_lora(stack, reference)
    loss = (output * probe[:, local]).sum() + log_probs.sum()
    step = runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=stack, max_grad_norm=1e6
    )
    clear_ulysses_params()

    # Assert
    assert_exact(output, expected_output[:, local], "output")
    assert_exact(logits, expected_logits[:, local], "logits")
    assert_exact(fused_logits, expected_logits[:, local], "fused logits")
    assert_exact(log_probs, expected_log_probs[:, local], "log-probs")
    assert step is not None
    assert len(expected) == sum(1 for _ in stack.parameters())
    with torch.no_grad():
        for name, param in stack.named_parameters():
            assert_exact(_dense(param), expected[plain(name)], name)


def _meta_load_worker(
    rank: int, world_size: int, cp: int, tp: int, checkpoint_dir: str
) -> None:
    """A meta stack loads each rank's dp_shard x cp x tp piece from safetensors."""
    # Arrange
    _use_float64()
    torch.manual_seed(0)
    dense = Gemma4BlockStack()
    if rank == 0:
        # The tied head is not its own checkpoint tensor.
        state = {k: v for k, v in dense.state_dict().items() if k != "lm_head.weight"}
        save_file(state, f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = Gemma4BlockStack()
    stack.config = Gemma4TextConfig(name_or_path=checkpoint_dir)
    mesh = parallel_mesh(world_size, cp, tp, 1)
    config = fsdp_config(cp, tp, 1, FSDP_DTYPE)

    # Act
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(stack, "cpu", config, parallel_mesh=mesh)

    # Assert
    expected = dict(dense.named_parameters())
    _assert_tied_embedding(stack, dense, tp)
    for name, param in stack.named_parameters():
        if name == "embed_tokens.weight":
            continue
        assert isinstance(param, DTensor), name
        mesh_size = world_size if on_tp_mesh(param) else world_size // tp
        assert param.device_mesh.size() == mesh_size, name
        assert_exact(param.full_tensor(), expected[plain(name)], name)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp)
    buffers = dict(dense.named_buffers())
    for name, buffer in stack.named_buffers():
        assert_exact(buffer, buffers[plain(name)], name)


def _resume_worker(
    rank: int,
    world_size: int,
    save: tuple[int, int],
    load: tuple[int, int],
    checkpoint_dir: str,
) -> None:
    """A LoRA checkpoint saved at one ``(cp, tp)`` mesh resumes unchanged at another."""
    # Arrange
    _use_float64()
    torch.manual_seed(0)
    dense = _with_lora(Gemma4BlockStack())
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    save_cp, save_tp = save
    load_cp, load_tp = load
    saved, runtime, optimizer = lora_runtime(
        dense, world_size, save_cp, save_tp, 1, FSDP_DTYPE
    )
    hidden, ids, per_layer, probe = _shard_data(rank // (save_cp * save_tp))
    output = _forward(saved, hidden, ids, per_layer)
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
        dense, world_size, load_cp, load_tp, 1, FSDP_DTYPE
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
    _assert_tied_embedding(resumed, dense, load_tp)

    # Assert
    if rank != 0:
        return
    assert checkpoint["optimizer"]["state"]
    for key, value in checkpoint["model"].items():
        if "lora_" not in key:
            assert_exact(value, natural[key], key)
    assert_same_state(reexported_model, checkpoint["model"], "model")
    assert_same_state(reexported_optimizer, checkpoint["optimizer"], "optimizer")


def _checkpoint_model() -> nn.Module:
    config = Gemma4Config.from_dict(SUPPORTED_MODEL_INFO[CHECKPOINT].config)
    with torch.device("meta"):
        return AutoModelForImageTextToText.from_config(config)


def _multimodal_forward_worker(rank: int, world_size: int) -> None:
    config = model_config()
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    dense = Gemma4ForConditionalGeneration(config).to(DTYPE)
    model = copy.deepcopy(dense)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    ids = torch.randint(1, 28, (BATCH, SEQ), generator=torch.Generator().manual_seed(0))
    expected = dense(input_ids=ids).logits

    apply_tensor_parallel(model, mesh)
    logits = model(input_ids=ids).logits

    embed = model.model.language_model.embed_tokens.weight
    assert model.lm_head.weight is embed
    assert isinstance(embed, DTensor)
    assert embed.placements == (Shard(1),)
    assert_exact(logits, expected, "multimodal logits")


def _meta_checkpoint_shard_worker(rank: int, world_size: int, peft: bool) -> None:
    model = _checkpoint_model()
    text = model.model.language_model
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    text_lora = rf".*language_model.*\.({'|'.join(LORA_TARGETS)})"
    target = (
        get_peft_model(model, LoraConfig(r=2, target_modules=text_lora))
        if peft
        else model
    )

    count = apply_tensor_parallel(target, mesh)

    # Attention and MLP in every text layer, the per-layer embedding, and the tied head.
    assert count == 2 * len(text.layers) + 2
    embed = text.embed_tokens.weight
    assert model.lm_head.weight is embed
    assert isinstance(embed, DTensor)
    assert embed.placements == (Shard(1),)
    assert embed.to_local().numel() * world_size == embed.numel()
    ple = text.embed_tokens_per_layer.weight
    assert isinstance(ple, DTensor)
    assert ple.placements == (Shard(1),)
    assert ple.to_local().numel() * world_size == ple.numel()
    for layer in text.layers:
        attention = layer.self_attn
        q_rows = attention.config.num_attention_heads * attention.head_dim
        assert attention.q_proj.weight.to_local().shape[0] == q_rows // world_size
        assert attention.o_proj.weight.to_local().shape[1] == q_rows // world_size
        if not attention.is_kv_shared_layer:
            kv_rows = attention.k_proj.weight.shape[0]
            assert attention.k_proj.weight.to_local().shape[0] == kv_rows // world_size
        mlp_rows = layer.mlp.intermediate_size // world_size
        assert layer.mlp.gate_proj.weight.to_local().shape[0] == mlp_rows
        assert layer.mlp.down_proj.weight.to_local().shape[1] == mlp_rows
        assert not isinstance(layer.per_layer_input_gate.weight, DTensor)
        assert not isinstance(layer.per_layer_projection.weight, DTensor)
    assert not isinstance(text.per_layer_model_projection.weight, DTensor)
    for tower in (model.model.vision_tower, model.model.audio_tower):
        assert not any(isinstance(param, DTensor) for param in tower.parameters())


class TestGemma4CheckpointCoverage:
    def test_every_parameter_is_planned_or_skipped(self) -> None:
        # Arrange
        model = _checkpoint_model()
        modules = dict(model.named_modules())
        known = (
            CP_BLOCK_CLASSES
            | TOKENWISE_CLASSES
            | TP_BLOCK_CLASSES
            | SKIPPED_CLASSES
            | TOWER_CLASSES
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
        tp_owners = set(owners.values()) - SKIPPED_CLASSES - TOWER_CLASSES
        assert tp_owners - TP_WHOLE_CLASSES == TP_BLOCK_CLASSES

    def test_tensor_and_context_parallel_apply(self) -> None:
        # Arrange
        model = _checkpoint_model()
        runtime = family_runtime(CHECKPOINT)
        text = model.model.language_model

        # Act
        experts = iter_packed_expert_modules(model)
        heads = {
            (
                attention.q_proj.out_features // attention.head_dim,
                attention.config.per_layer_config[
                    attention.layer_idx
                ].num_key_value_heads,
            )
            for attention in (layer.self_attn for layer in text.layers)
        }

        # Assert
        assert experts == []
        assert runtime.tensor_parallel_plan == (
            "agilerl.architectures.gemma4.tensor_parallel:GEMMA4_TENSOR_PARALLEL_PLAN"
        )
        assert (
            FAMILY_RUNTIME_CONFIGS["gemma4_text"].tensor_parallel_plan
            == runtime.tensor_parallel_plan
        )
        assert runtime.context_parallel_plan is None
        assert model.lm_head.weight is text.embed_tokens.weight
        assert heads == {(8, 2)}
        for cp in (2, 4, 8):
            validate_cp_heads(8, 2, cp)

    def test_heads_that_do_not_divide_cp_raise(self) -> None:
        with pytest.raises(ValueError, match="num_attention_heads"):
            validate_cp_heads(8, 2, 16)

    def test_legal_tp_degrees_follow_every_layer(self) -> None:
        # Arrange
        text = _checkpoint_model().model.language_model
        attention = [layer.self_attn for layer in text.layers]
        heads = {
            (module.config.num_attention_heads, module.num_key_value_groups)
            for module in attention
        }
        widths = {layer.mlp.intermediate_size for layer in text.layers}

        # Act
        legal = [
            tp
            for tp in range(2, 9)
            if all(
                num_q % tp == 0
                and (groups % (num_q // tp) == 0 or (num_q // tp) % groups == 0)
                for num_q, groups in heads
            )
            and all(width % tp == 0 for width in widths)
        ]

        # Assert
        assert heads == {(8, 4)}
        assert widths == {10240}
        assert {module.head_dim for module in attention} == {256, 512}
        assert legal == LEGAL_TP

    def test_legal_cp_tp_meshes_split_local_heads(self) -> None:
        # Arrange
        meshes = [(cp, tp) for tp in (1, 2, 4, 8) for cp in (1, 2, 4, 8)]

        # Act
        legal = []
        for cp, tp in meshes:
            if (cp, tp) == (1, 1):
                continue
            try:
                validate_cp_heads(HEADS // tp, max(KV_HEADS // tp, 1), cp)
            except ValueError:
                continue
            legal.append((cp, tp))

        # Assert
        assert legal == LEGAL_CP_TP

    def test_heads_that_do_not_divide_tp_raise(self) -> None:
        mesh = MagicMock()
        mesh.size.return_value = 3

        with pytest.raises(ValueError, match="num_attention_heads"):
            apply_gemma4_tensor_parallel(Gemma4BlockStack(), mesh)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize("peft", [False, True], ids=["bare", "peft"])
    def test_tp_plan_shards_every_checkpoint_block(self, peft: bool) -> None:
        run_ranks(_meta_checkpoint_shard_worker, 2, peft, timeout=300)


def _biased_down_proj_worker(rank: int, world_size: int) -> None:
    stack = Gemma4BlockStack()
    mlp = stack.layers[0].mlp
    mlp.down_proj = nn.Linear(mlp.intermediate_size, HIDDEN, bias=True)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    with pytest.raises(ValueError, match="down_proj has a bias"):
        apply_gemma4_tensor_parallel(stack, mesh)


class TestGemma4Parallel:
    @pytest.mark.multi_rank
    def test_mlp_down_proj_bias_raises(self) -> None:
        run_ranks(_biased_down_proj_worker)

    def test_tiny_stack_runs(self) -> None:
        stack = _with_lora(Gemma4BlockStack())
        hidden, ids, per_layer, _ = _shard_data(0)

        output = _forward(stack, hidden, ids, per_layer)

        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()

    def test_tiny_stack_matches_checkpoint_layout(self) -> None:
        stack = Gemma4BlockStack()

        attention = [layer.self_attn for layer in stack.layers]

        assert [module.head_dim for module in attention] == [4, 8, 4, 8]
        assert [module.is_kv_shared_layer for module in attention] == [
            False,
            False,
            True,
            True,
        ]
        assert [module.sliding_window for module in attention] == [
            WINDOW,
            None,
            WINDOW,
            None,
        ]
        assert all(module.scaling == 1.0 for module in attention)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "world"),
        [(1, 1, 2), (2, 1, 4), (4, 1, 8), (1, 2, 4), (2, 2, 8), (1, 4, 8)],
    )
    def test_forward_and_step_match_unsharded(
        self, cp: int, tp: int, world: int
    ) -> None:
        run_ranks(_worker, world, cp, tp, timeout=300)

    @pytest.mark.multi_rank
    def test_multimodal_forward_matches_unsharded(self) -> None:
        run_ranks(_multimodal_forward_worker, 2, timeout=300)

    @pytest.mark.multi_rank
    def test_liger_classes_built_before_wrap_are_sharded(self) -> None:
        run_ranks(_worker, 4, 1, 2, True, timeout=300)

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
