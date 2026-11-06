# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-, context- and expert-parallel gpt-oss blocks on a gloo mesh."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from transformers import AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt_oss import modeling_gpt_oss
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
    GptOssTopKRouter,
)

from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    position_ids_from_mask,
)
from agilerl.architectures.gptoss.tensor_parallel import apply_gpt_oss_tensor_parallel
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
    cpu_flex_attention,
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

HIDDEN = 16
HEAD_DIM = 4
HEADS = 8
KV_HEADS = 4
EXPERTS = 4
BATCH = 2
SEQ = 8
# Shorter than one CP shard, so sliding layers mask inside and across shards.
WINDOW = 3
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXPERT_LORA_PARAMS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
EXPERT_PARAMS = ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias")
TP_BLOCK_CLASSES = {GptOssAttention}
EP_BLOCK_CLASSES = {GptOssExperts}
# Norms and embeddings stay whole; the router is tiny and stays whole.
SKIPPED_CLASSES = {GptOssRMSNorm, nn.Embedding, GptOssTopKRouter}
# One weight of each attention part the TP plan shards, by stack FQN without LoRA wrappers.
TP_PROBE_WEIGHTS = (
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.sinks",
    "layers.1.self_attn.o_proj.weight",
)
CHECKPOINT_LAYERS = 24


def _config() -> GptOssConfig:
    config = GptOssConfig(
        hidden_size=HIDDEN,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_local_experts=EXPERTS,
        num_experts_per_tok=2,
        vocab_size=32,
        sliding_window=WINDOW,
        layer_types=["sliding_attention", "full_attention"],
        attention_bias=True,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


class GptOssBlockStack(nn.Module):
    """One sliding-attention and one full-attention gpt-oss decoder layer."""

    _no_split_modules = ("GptOssDecoderLayer",)

    def __init__(self) -> None:
        super().__init__()
        config = _config()
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, 0), GptOssDecoderLayer(config, 1)]
        )
        self.config = SimpleNamespace(model_type="gpt_oss")
        # Experts and sinks are torch.empty and the router is zeros outside post_init.
        with torch.no_grad():
            for layer in self.layers:
                experts = layer.mlp.experts
                experts.gate_up_proj.normal_(std=0.5)
                experts.down_proj.normal_(std=0.5)
                experts.gate_up_proj_bias.normal_(std=0.1)
                experts.down_proj_bias.normal_(std=0.1)
                layer.mlp.router.weight.normal_()
                layer.mlp.router.bias.normal_()
                layer.self_attn.sinks.normal_()


def _with_moe_lora(module: nn.Module) -> nn.Module:
    inject_adapter_in_model(
        LoraConfig(
            r=2, target_modules=LORA_TARGETS, target_parameters=EXPERT_LORA_PARAMS
        ),
        module,
    )
    assert upgrade_moe_param_wrappers(module) == 2
    module.requires_grad_(True)
    # At std 0.5 attention grads reach 1e4 and float32 misses the 1e-5 tolerance.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.1)
    return module


def _apply_liger_swaps() -> None:
    """Swap the class Liger reassigns for gpt_oss, keeping the HF forwards."""
    modeling_gpt_oss.GptOssRMSNorm = type("LigerRMSNorm", (nn.Module,), {})


def _eager_mask(window: int | None, length: int = SEQ) -> torch.Tensor:
    """Additive causal mask, windowed on sliding layers, for eager attention."""
    query = torch.arange(length)[:, None]
    key = torch.arange(length)[None, :]
    allowed = key <= query
    if window is not None:
        allowed &= key > query - window
    blocked = torch.finfo(torch.float32).min
    return torch.where(allowed, 0.0, blocked)[None, None]


def _rope(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = GptOssRotaryEmbedding(_config())
    return rotary(torch.zeros(*positions.shape, HIDDEN), positions)


def _forward(stack: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Run both layers on the full sequence with eager masks."""
    rope = _rope(torch.arange(SEQ).expand(BATCH, SEQ))
    masks = (_eager_mask(WINDOW), _eager_mask(None))
    for layer, mask in zip(stack.layers, masks, strict=True):
        hidden = layer(hidden, attention_mask=mask, position_embeddings=rope)
    return hidden


def _forward_cp_shard(
    stack: nn.Module, hidden: torch.Tensor, local: slice
) -> torch.Tensor:
    """Run both layers on one CP sequence shard; Ulysses flex builds the masks."""
    rope = _rope(torch.arange(SEQ).expand(BATCH, SEQ)[:, local])
    for layer in stack.layers:
        hidden = layer(hidden, attention_mask=None, position_embeddings=rope)
    return hidden


def _use_ulysses_flex(stack: nn.Module, cp_group: dist.ProcessGroup) -> None:
    """Route the stack's attention through Ulysses over the CPU flex stand-in."""
    stack.layers[0].self_attn.config._attn_implementation = "flex_attention"
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = cpu_flex_attention
    substitute_hf_ulysses_attn(cp_group, "flex_attention")


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
        table[aligned[:, :-1]], _rope(positions), _eager_mask(None, length)
    )[0]
    expected = shard_for_cp(expected, layout.rank, cp)
    assert_exact(restored, expected, "cp sequence")


def _shard_data(dp_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Input hidden states and an output probe for one data shard."""
    generator = torch.Generator().manual_seed(100 + dp_rank)
    hidden = torch.randn(BATCH, SEQ, HIDDEN, generator=generator)
    probe = torch.randn(BATCH, SEQ, HIDDEN, generator=generator)
    return hidden, probe


def _assert_tp_blocks_sharded(model: nn.Module, tp: int) -> None:
    """Each probed attention weight holds 1/tp of its elements per TP rank."""
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
    """Expert weights and biases split their expert axis ``ep`` ways."""
    params = [
        getattr(module, name)
        for module in model.modules()
        if isinstance(module, GptOssExperts)
        for name in EXPERT_PARAMS
    ]
    assert len(params) == 2 * len(EXPERT_PARAMS)
    for param in params:
        assert isinstance(param, DTensor)
        assert on_ep_mesh(param)
        index = param.device_mesh.mesh_dim_names.index(MeshAxisName.EP)
        assert param.placements[index] == Shard(0)
        assert int(param.device_mesh.size(index)) == ep


def _worker(rank: int, world_size: int, cp: int, tp: int, ep: int) -> None:
    """Forward, reshard, LoRA gather and one SGD step match the unsharded stack."""
    # Arrange
    torch.manual_seed(0)
    stack = _with_moe_lora(GptOssBlockStack())
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
    local = slice(None)
    if cp > 1:
        assert mesh.cp is not None
        _use_ulysses_flex(stack, mesh.cp.get_group())
        if tp > 1:
            with torch.no_grad():
                _assert_sequence_roundtrip(stack, reference, cp, mesh)
        publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
        cp_rank = int(mesh.cp.get_local_rank())
        local = slice(cp_rank * SEQ // cp, (cp_rank + 1) * SEQ // cp)
    hidden, probe = _shard_data(rank // (cp * tp))
    with torch.no_grad():
        expected_output = _forward(reference, hidden)[:, local]
    reference_loss = sum(
        (_forward(reference, data) * data_probe).sum()
        for data, data_probe in map(_shard_data, range(dp))
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
    if ep > 1:
        _assert_experts_on_ep(stack, ep)
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


def _grouped_mm_ep_worker(rank: int, world_size: int) -> None:
    """EP experts without LoRA match the eager stack under HF's grouped_mm experts."""
    # Arrange
    torch.manual_seed(0)
    stack = GptOssBlockStack()
    inject_adapter_in_model(LoraConfig(r=2, target_modules=LORA_TARGETS), stack)
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    for layer in stack.layers:
        layer.mlp.experts.config._experts_implementation = "grouped_mm"
    mesh = parallel_mesh(world_size, 1, 1, 2)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", fsdp_config(1, 1, 2), parallel_mesh=mesh
        )
    hidden, probe = _shard_data(rank)
    reference_hidden = hidden.clone().requires_grad_(True)
    hidden.requires_grad_(True)
    expected_output = _forward(reference, reference_hidden)
    (expected_output * probe).sum().backward()

    # Act
    output = _forward(stack, hidden)
    (output * probe).sum().backward()

    # Assert
    _assert_experts_on_ep(stack, 2)
    assert_exact(output, expected_output.detach(), "output")
    assert_exact(hidden.grad, reference_hidden.grad, "input grad")


def _disabled_expert_lora_ep_worker(rank: int, world_size: int) -> None:
    """EP experts with LoRA disabled match the plain stack under HF's grouped_mm experts."""
    # Arrange
    torch.manual_seed(0)
    stack = GptOssBlockStack()
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    _with_moe_lora(stack)
    for module in stack.modules():
        if isinstance(module, BaseTunerLayer):
            module.enable_adapters(False)
    for layer in stack.layers:
        layer.mlp.experts.get_base_layer().config._experts_implementation = "grouped_mm"
    mesh = parallel_mesh(world_size, 1, 1, 2)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", fsdp_config(1, 1, 2), parallel_mesh=mesh
        )
    hidden, probe = _shard_data(rank)
    reference_hidden = hidden.clone().requires_grad_(True)
    hidden.requires_grad_(True)
    expected_output = _forward(reference, reference_hidden)
    (expected_output * probe).sum().backward()

    # Act
    output = _forward(stack, hidden)
    (output * probe).sum().backward()

    # Assert
    _assert_experts_on_ep(stack, 2)
    assert_exact(output, expected_output.detach(), "output")
    assert_exact(hidden.grad, reference_hidden.grad, "input grad")


def _meta_load_worker(
    rank: int, world_size: int, cp: int, tp: int, ep: int, checkpoint_dir: str
) -> None:
    """A meta stack loads each rank's dp_shard x cp x tp x ep piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    dense = GptOssBlockStack()
    if rank == 0:
        save_file(dense.state_dict(), f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = GptOssBlockStack()
    stack.config = GptOssConfig(name_or_path=checkpoint_dir)
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
    dense = _with_moe_lora(GptOssBlockStack())
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    saved, runtime, optimizer = lora_runtime(dense, world_size, cp, save_tp, ep)
    hidden, probe = _shard_data(rank // save_tp)
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
    config = GptOssConfig.from_dict(SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].config)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    count = apply_tensor_parallel(model, mesh)

    # Every attention block plus the untied LM head.
    assert count == CHECKPOINT_LAYERS + 1
    sinks = model.model.layers[0].self_attn.sinks
    assert isinstance(sinks, DTensor)
    assert sinks.to_local().shape == (64 // world_size,)


class TestGptOssCheckpointCoverage:
    def test_every_parameter_is_planned_or_skipped(self) -> None:
        # Arrange
        config = GptOssConfig.from_dict(
            SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].config
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

    def test_expert_parallel_finds_every_checkpoint_expert_block(self) -> None:
        # Arrange
        config = GptOssConfig.from_dict(
            SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].config
        )
        with torch.device("meta"):
            model = AutoModel.from_config(config)

        # Act
        found = iter_packed_expert_modules(model)

        # Assert
        assert [type(module) for _, module in found] == [
            GptOssExperts
        ] * CHECKPOINT_LAYERS
        assert packed_expert_counts(model) == [32] * CHECKPOINT_LAYERS

    @pytest.mark.multi_rank
    def test_tp_plan_shards_every_checkpoint_block(self) -> None:
        run_ranks(_meta_checkpoint_shard_worker, 2, timeout=300)


class TestGptOssParallel:
    def test_tiny_stack_runs(self) -> None:
        stack = _with_moe_lora(GptOssBlockStack())
        hidden = torch.randn(BATCH, SEQ, HIDDEN)

        output = _forward(stack, hidden)

        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()

    def test_heads_that_do_not_divide_tp_raise(self) -> None:
        mesh = MagicMock()
        mesh.size.return_value = 3

        with pytest.raises(ValueError, match="num_attention_heads"):
            apply_gpt_oss_tensor_parallel(GptOssBlockStack(), mesh)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "ep", "world"),
        [
            (1, 1, 1, 2),
            (1, 1, 2, 2),
            (1, 2, 2, 4),
            (2, 1, 1, 4),
            (2, 1, 2, 4),
            (2, 2, 2, 8),
        ],
    )
    def test_forward_and_step_match_unsharded(
        self, cp: int, tp: int, ep: int, world: int
    ) -> None:
        run_ranks(_worker, world, cp, tp, ep, timeout=300)

    @pytest.mark.multi_rank
    def test_ep_without_expert_lora_matches_unsharded_under_grouped_mm(self) -> None:
        run_ranks(_grouped_mm_ep_worker, 2, timeout=300)

    @pytest.mark.multi_rank
    def test_ep_with_expert_lora_disabled_matches_unsharded_under_grouped_mm(
        self,
    ) -> None:
        run_ranks(_disabled_expert_lora_ep_worker, 2, timeout=300)

    @pytest.mark.multi_rank
    def test_meta_load_fills_dp_cp_tp_ep_shards(self, tmp_path: Path) -> None:
        run_ranks(_meta_load_worker, 8, 2, 2, 2, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(("save_tp", "load_tp"), [(2, 4), (4, 2)])
    def test_checkpoint_resumes_unchanged_at_another_tp_degree(
        self, save_tp: int, load_tp: int, tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save_tp, load_tp, 4, str(tmp_path), timeout=300)
