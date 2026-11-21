# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-, expert- and context-parallel Granite-3.1 MoE on a gloo mesh."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, inject_adapter_in_model
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from transformers import AutoModelForCausalLM
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeAttention,
    GraniteMoeExperts,
    GraniteMoeForCausalLM,
    GraniteMoeRMSNorm,
    GraniteMoeTopKRouter,
)

from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    position_ids_from_mask,
)
from agilerl.architectures.catalog import family_runtime
from agilerl.architectures.granitemoe.tensor_parallel import (
    apply_granitemoe_tensor_parallel,
)
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
from agilerl.distributed.expert_parallel import (
    iter_packed_expert_modules,
    packed_expert_counts,
)
from agilerl.distributed.fsdp import (
    materialize_dtensors,
    materialize_fsdp2_from_cpu_state,
)
from agilerl.distributed.mesh import MeshAxisName
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.distributed.tensor_parallel import apply_tensor_parallel
from agilerl.lora.moe import upgrade_moe_param_wrappers
from tests.parallel_helpers import (
    assert_exact,
    assert_lora,
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
    use_ulysses_flash,
    world_held_numel,
)
from tests.rank_launch import run_ranks

MODEL_ID = "ibm-granite/granite-3.1-3b-a800m-instruct"
HIDDEN = 16
HEADS = 4
KV_HEADS = 2
EXPERTS = 4
VOCAB = 32
BATCH = 2
SEQ = 8
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXPERT_LORA_PARAMS = [
    "block_sparse_moe.experts.gate_up_proj",
    "block_sparse_moe.experts.down_proj",
]
EP_BLOCK_CLASSES = {GraniteMoeExperts}
# Ulysses flash exchanges sequence for heads around attention.
CP_BLOCK_CLASSES = {GraniteMoeAttention}
TP_BLOCK_CLASSES = {GraniteMoeAttention}
# Norms stay whole and the router is tiny and stays whole. The tied embedding
# and head split their columns outside the block plan.
SKIPPED_CLASSES = {GraniteMoeRMSNorm, nn.Embedding, GraniteMoeTopKRouter}
# One weight of each attention projection the TP plan shards, by FQN without LoRA wrappers.
TP_PROBE_WEIGHTS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.1.self_attn.v_proj.weight",
    "model.layers.1.self_attn.o_proj.weight",
)
CHECKPOINT_LAYERS = 32
CHECKPOINT_EXPERTS = 40
CHECKPOINT_HEADS = 24
CHECKPOINT_KV_HEADS = 8
TRAINERS = 8
# Legal tp at 8 trainers: 24 query heads / tp keeps whole KV groups of 3.
LEGAL_TP = [2, 4, 8]
# Legal (cp, tp, ep) at 8 trainers with dp_shard >= 2: Ulysses splits each TP
# rank's query heads, and ep divides 40 experts and the 8-rank EP region.
LEGAL_CP_TP_EP = [
    (cp, tp, ep)
    for cp, tp in [(1, 2), (2, 2), (4, 2), (1, 4), (2, 4)]
    for ep in (1, 2, 4, 8)
]


def _config() -> GraniteMoeConfig:
    config = GraniteMoeConfig(
        hidden_size=HIDDEN,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_local_experts=EXPERTS,
        num_experts_per_tok=2,
        vocab_size=VOCAB,
        # Not 1/sqrt(head_dim), so a dropped softmax scale fails the compare.
        attention_multiplier=0.3,
        embedding_multiplier=12.0,
        residual_multiplier=0.22,
        logits_scaling=6.0,
        tie_word_embeddings=True,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def _new_stack() -> GraniteMoeForCausalLM:
    """Two Granite MoE decoder layers with the tied embedding and LM head."""
    model = GraniteMoeForCausalLM(_config())
    # post_init draws std 0.02, which leaves attention uniform and experts near zero.
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_(std=0.1)
        for layer in model.model.layers:
            attention = layer.self_attn
            for linear in (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.o_proj,
            ):
                linear.weight.normal_(std=0.5)
            moe = layer.block_sparse_moe
            moe.experts.gate_up_proj.normal_(std=0.5)
            moe.experts.down_proj.normal_(std=0.5)
            moe.router.weight.normal_()
    return model


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
                param.normal_(std=0.1)
    return module


def _forward(stack: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Logits for the full sequence."""
    return stack(input_ids=ids, use_cache=False).logits


def _capture_final_hidden(stack: nn.Module) -> list[torch.Tensor]:
    """Record the final norm output the LM head reads."""
    captured: list[torch.Tensor] = []
    stack.model.norm.register_forward_hook(
        lambda _module, _args, output: captured.append(output.detach())
    )
    return captured


def _forward_cp_shard(
    stack: nn.Module, ids: torch.Tensor, local: slice
) -> torch.Tensor:
    """Logits for one CP sequence shard; Ulysses gathers the sequence."""
    positions = torch.arange(SEQ).expand(BATCH, SEQ)[:, local]
    return stack(input_ids=ids, position_ids=positions, use_cache=False).logits


def _assert_sequence_roundtrip(
    model: nn.Module, reference: nn.Module, cp: int, mesh: ParallelMesh
) -> None:
    """The model through ContextParallelSequence and scoring_hidden matches the local shard."""
    layout = ContextParallelSequence(
        pad_token_id=0,
        cp=cp,
        group=mesh.cp.get_group(),
        rank=int(mesh.cp.get_local_rank()),
        round_to=cp * CP_SCAN_CHUNK,
    )
    ids = broadcast_tensor(torch.randint(1, VOCAB, (BATCH, 5)))
    mask = torch.ones_like(ids)
    aligned, aligned_mask = pad_right_for_cp(
        ids, cp * CP_SCAN_CHUNK, layout.pad_token_id, mask
    )
    assert aligned.shape[1] - 1 == cp * CP_SCAN_CHUNK
    with layout.prepare_forward(ids, mask) as forward:
        restored = forward.scoring_hidden(model(**forward.kwargs).logits)
    length = aligned.shape[1] - 1
    positions = position_ids_from_mask(aligned_mask)[:, :length]
    expected = reference(
        input_ids=aligned[:, :-1],
        # Without a mask HF reads the repeated pad positions as packed rows;
        # Ulysses runs one causal row over the pad.
        attention_mask=torch.ones_like(aligned_mask[:, :length]),
        position_ids=positions,
        use_cache=False,
    ).logits
    expected = shard_for_cp(expected, layout.rank, cp)
    assert_exact(restored, expected, "cp sequence")


def _shard_data(dp_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Input ids and a logits probe for one data shard."""
    generator = torch.Generator().manual_seed(100 + dp_rank)
    ids = torch.randint(1, VOCAB, (BATCH, SEQ), generator=generator)
    probe = torch.randn(BATCH, SEQ, VOCAB, generator=generator)
    return ids, probe


def _full(param: torch.Tensor) -> torch.Tensor:
    return param.full_tensor() if isinstance(param, DTensor) else param


def _assert_resharded(model: nn.Module, reference: nn.Module, tp: int) -> None:
    """Each parameter is split across ranks again; TP weights stay on the TP mesh."""
    reference_params = {
        plain(name): param for name, param in reference.named_parameters()
    }
    world = dist.get_world_size()
    embedding = model.model.embed_tokens.weight
    for name, param in model.named_parameters():
        if param is embedding:
            continue
        full = reference_params[plain(name)]
        assert isinstance(param, DTensor), name
        held = world_held_numel(param)
        if on_ep_mesh(param):
            # Expert LoRA shards on ep only and is replicated over edp_shard.
            split = 1
            for dim, placement in enumerate(param.placements):
                if not placement.is_replicate():
                    split *= int(param.device_mesh.size(dim))
            assert held == full.numel() * (world // split), name
        elif on_tp_mesh(param):
            assert param.device_mesh.mesh_dim_names[-1] == MeshAxisName.TP, name
            assert param.device_mesh.size() == world, name
            copies = tp if param.placements[-1].is_replicate() else 1
            assert held == full.numel() * copies, name
        else:
            assert param.device_mesh.size() == world // tp, name
            assert held == full.numel() * tp, name


def _assert_tied_embedding(stack: nn.Module, reference: nn.Module, tp: int) -> None:
    """Embedding and head share one weight; TP splits its hidden columns."""
    weight = stack.model.embed_tokens.weight
    assert stack.lm_head.weight is weight
    expected = reference.model.embed_tokens.weight
    if tp == 1:
        assert not isinstance(weight, DTensor)
        assert_exact(weight, expected, "embed_tokens")
        return
    assert isinstance(weight, DTensor)
    assert on_tp_mesh(weight)
    assert weight.device_mesh.size() == tp
    assert weight.placements == (Shard(1),)
    assert weight.to_local().numel() * tp == weight.numel()
    assert_exact(weight.full_tensor(), expected, "embed_tokens")


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
        if KV_HEADS % tp != 0 and suffix.endswith(("k_proj.weight", "v_proj.weight")):
            # Several TP ranks read one whole KV head, so every rank keeps all of them.
            assert param.placements[-1].is_replicate(), suffix
            continue
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
        if isinstance(module, GraniteMoeExperts)
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
    """Forward, reshard, LoRA gather and one SGD step match the unsharded model."""
    # Arrange
    torch.manual_seed(0)
    stack = _with_moe_lora(_new_stack())
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    dp = world_size // (cp * tp)
    mesh = parallel_mesh(world_size, cp, tp, ep)
    # The FSDP mesh folds cp into dp_shard.
    assert mesh.dp_shard.size() == 2 * cp
    config = fsdp_config(cp, tp, ep)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    final_hidden = _capture_final_hidden(stack)
    local = slice(None)
    if cp > 1:
        assert mesh.cp is not None
        use_ulysses_flash(stack.config, mesh.cp.get_group())
        with torch.no_grad():
            _assert_sequence_roundtrip(stack, reference, cp, mesh)
        publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
        cp_rank = int(mesh.cp.get_local_rank())
        local = slice(cp_rank * SEQ // cp, (cp_rank + 1) * SEQ // cp)
    ids, probe = _shard_data(rank // (cp * tp))
    with torch.no_grad():
        expected_output = _forward(reference, ids)[:, local]
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
        output = _forward_cp_shard(stack, ids[:, local], local)
    else:
        output = _forward(stack, ids)
    with torch.no_grad(), materialize_dtensors(stack.lm_head.weight) as (head,):
        gathered_logits = final_hidden[-1] @ head.T / stack.config.logits_scaling
    _assert_resharded(stack, reference, tp)
    _assert_tied_embedding(stack, reference, tp)
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
    assert_exact(output, expected_output, "logits")
    assert_exact(gathered_logits, expected_output, "gathered head logits")
    assert step is not None
    assert len(expected) == sum(1 for _ in stack.parameters())
    errors: list[str] = []
    with torch.no_grad():
        for name, param in stack.named_parameters():
            value = _full(param)
            target = expected[plain(name)].reshape(value.shape)
            try:
                assert_exact(value, target, name)
            except AssertionError as exc:
                errors.append(str(exc))
    if errors:
        raise AssertionError("\n".join(errors))


def _meta_load_worker(
    rank: int, world_size: int, cp: int, tp: int, ep: int, checkpoint_dir: str
) -> None:
    """A meta model loads each rank's dp_shard x cp x tp x ep piece from safetensors."""
    # Arrange
    torch.manual_seed(0)
    dense = _new_stack()
    if rank == 0:
        # Hub checkpoints store the tied head once, under the embedding.
        state = {
            key: value
            for key, value in dense.state_dict().items()
            if key != "lm_head.weight"
        }
        save_file(state, f"{checkpoint_dir}/model.safetensors")
    dist.barrier()
    with torch.device("meta"):
        stack = GraniteMoeForCausalLM(_config())
    stack.config.name_or_path = checkpoint_dir
    mesh = parallel_mesh(world_size, cp, tp, ep)
    config = fsdp_config(cp, tp, ep)

    # Act
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(stack, "cpu", config, parallel_mesh=mesh)

    # Assert
    expected = dict(dense.named_parameters())
    for name, param in stack.named_parameters():
        assert_exact(_full(param), expected[plain(name)], name)
    _assert_tied_embedding(stack, dense, tp)
    if tp > 1:
        _assert_tp_blocks_sharded(stack, tp)
    _assert_experts_on_ep(stack, ep)


def _resume_worker(
    rank: int,
    world_size: int,
    save: tuple[int, int, int],
    load: tuple[int, int, int],
    checkpoint_dir: str,
) -> None:
    """A LoRA checkpoint saved at one ``(cp, tp, ep)`` mesh resumes unchanged at another."""
    # Arrange
    torch.manual_seed(0)
    dense = _with_moe_lora(_new_stack())
    for name, param in dense.named_parameters():
        param.requires_grad_("lora_" in name)
    broadcast_module(dense)
    natural = {key: value.clone() for key, value in dense.state_dict().items()}
    save_cp, save_tp, _ = save
    load_tp = load[1]
    saved, runtime, optimizer = lora_runtime(dense, world_size, *save)
    _assert_tied_embedding(saved, dense, save_tp)
    ids, probe = _shard_data(rank // (save_cp * save_tp))
    loss = (_forward(saved, ids) * probe).sum()
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
    resumed, resumed_runtime, resumed_optimizer = lora_runtime(dense, world_size, *load)

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
    assert any("experts" in key and "lora_" in key for key in checkpoint["model"])
    for key, value in checkpoint["model"].items():
        if "lora_" not in key:
            assert_exact(value, natural[key], key)
    assert_same_state(reexported_model, checkpoint["model"], "model")
    assert_same_state(reexported_optimizer, checkpoint["optimizer"], "optimizer")


def _checkpoint_model() -> nn.Module:
    config = GraniteMoeConfig.from_dict(SUPPORTED_MODEL_INFO[MODEL_ID].config)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def _meta_checkpoint_shard_worker(rank: int, world_size: int, peft: bool) -> None:
    model = _checkpoint_model()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    target = (
        get_peft_model(model, LoraConfig(r=2, target_modules=LORA_TARGETS))
        if peft
        else model
    )

    count = apply_tensor_parallel(target, mesh)

    # Every attention block plus the tied head.
    assert count == CHECKPOINT_LAYERS + 1
    embed = model.model.embed_tokens.weight
    assert model.lm_head.weight is embed
    assert isinstance(embed, DTensor)
    assert embed.placements == (Shard(1),)
    assert embed.to_local().numel() * world_size == embed.numel()
    for layer in model.model.layers:
        attention = layer.self_attn
        q_rows = CHECKPOINT_HEADS * attention.head_dim
        q = attention.q_proj.weight
        assert q.to_local().shape[0] * world_size == q_rows
        assert attention.o_proj.weight.to_local().shape[1] * world_size == q_rows
        kv = attention.k_proj.weight
        assert kv.to_local().shape[0] * world_size == kv.shape[0]
        assert not isinstance(layer.block_sparse_moe.router.weight, DTensor)
        assert not isinstance(layer.block_sparse_moe.experts.gate_up_proj, DTensor)
        assert not isinstance(layer.input_layernorm.weight, DTensor)


class TestGraniteMoeCheckpointCoverage:
    def test_every_parameter_is_planned_or_skipped(self) -> None:
        # Arrange
        model = _checkpoint_model()
        modules = dict(model.named_modules())
        known = TP_BLOCK_CLASSES | EP_BLOCK_CLASSES | CP_BLOCK_CLASSES | SKIPPED_CLASSES

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

    def test_expert_parallel_finds_every_checkpoint_expert_block(self) -> None:
        # Arrange
        model = _checkpoint_model()

        # Act
        found = iter_packed_expert_modules(model)

        # Assert
        assert [type(module) for _, module in found] == [
            GraniteMoeExperts
        ] * CHECKPOINT_LAYERS
        assert packed_expert_counts(model) == [CHECKPOINT_EXPERTS] * CHECKPOINT_LAYERS

    def test_tensor_parallel_plan_is_registered(self) -> None:
        # Arrange
        model = _checkpoint_model()

        # Act
        runtime = family_runtime(MODEL_ID)

        # Assert
        assert runtime.tensor_parallel_plan == (
            "agilerl.architectures.granitemoe.tensor_parallel:"
            "GRANITEMOE_TENSOR_PARALLEL_PLAN"
        )
        assert runtime.context_parallel_plan is None
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_checkpoint_head_counts(self) -> None:
        config = SUPPORTED_MODEL_INFO[MODEL_ID].config

        assert config["num_attention_heads"] == CHECKPOINT_HEADS
        assert config["num_key_value_heads"] == CHECKPOINT_KV_HEADS
        assert config["hidden_size"] == 1536
        assert config["num_local_experts"] == CHECKPOINT_EXPERTS
        assert config["tie_word_embeddings"]

    def test_legal_tp_degrees_keep_whole_kv_groups(self) -> None:
        # Arrange
        groups = CHECKPOINT_HEADS // CHECKPOINT_KV_HEADS

        # Act
        legal = [
            tp
            for tp in range(2, TRAINERS + 1)
            if TRAINERS % tp == 0
            and CHECKPOINT_HEADS % tp == 0
            and (
                groups % (CHECKPOINT_HEADS // tp) == 0
                or (CHECKPOINT_HEADS // tp) % groups == 0
            )
        ]

        # Assert
        assert legal == LEGAL_TP

    def test_legal_cp_tp_ep_meshes_at_eight_trainers(self) -> None:
        # Arrange
        degrees = (1, 2, 4, 8)

        # Act
        legal = []
        for tp in (2, 4, 8):
            for cp in degrees:
                # dp_shard folds in cp and must stay at 2 or more.
                if TRAINERS // tp < 2 or (TRAINERS // tp) % cp != 0:
                    continue
                try:
                    validate_cp_heads(
                        CHECKPOINT_HEADS // tp,
                        max(CHECKPOINT_KV_HEADS // tp, 1),
                        cp,
                    )
                except ValueError:
                    continue
                legal.extend(
                    (cp, tp, ep)
                    for ep in degrees
                    if TRAINERS % ep == 0 and CHECKPOINT_EXPERTS % ep == 0
                )

        # Assert
        assert sorted(legal) == sorted(LEGAL_CP_TP_EP)

    def test_heads_that_do_not_divide_tp_raise(self) -> None:
        mesh = MagicMock()
        mesh.size.return_value = 3

        with pytest.raises(ValueError, match="num_attention_heads"):
            apply_granitemoe_tensor_parallel(_new_stack(), mesh)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize("peft", [False, True], ids=["bare", "peft"])
    def test_tp_plan_shards_every_checkpoint_block(self, peft: bool) -> None:
        run_ranks(_meta_checkpoint_shard_worker, 2, peft, timeout=300)

    def test_liger_has_no_granitemoe_patch(self) -> None:
        monkey_patch = pytest.importorskip("liger_kernel.transformers.monkey_patch")
        granite_liger = pytest.importorskip("agilerl.architectures.granite_liger")

        applier = monkey_patch.MODEL_TYPE_TO_APPLY_LIGER_FN.get("granitemoe")

        assert applier in (None, granite_liger.apply_liger_kernel_to_granitemoe)

    def test_checkpoint_heads_divide_context_parallel_degrees(self) -> None:
        # Arrange
        config = SUPPORTED_MODEL_INFO[MODEL_ID].config

        # Act
        for cp in (2, 4, 8):
            validate_cp_heads(
                config["num_attention_heads"], config["num_key_value_heads"], cp
            )

        # Assert
        with pytest.raises(ValueError, match="divisible"):
            validate_cp_heads(
                config["num_attention_heads"], config["num_key_value_heads"], 16
            )


class TestGraniteMoeParallel:
    def test_tiny_stack_runs(self) -> None:
        stack = _with_moe_lora(_new_stack())
        ids, _ = _shard_data(0)

        output = _forward(stack, ids)

        assert output.shape == (BATCH, SEQ, VOCAB)
        assert torch.isfinite(output).all()

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "ep", "world"),
        [
            (1, 1, 1, 2),
            (1, 1, 2, 2),
            (2, 1, 1, 4),
            (2, 1, 2, 4),
            (1, 2, 2, 4),
            (2, 2, 2, 8),
        ],
    )
    def test_forward_and_step_match_unsharded(
        self, cp: int, tp: int, ep: int, world: int
    ) -> None:
        run_ranks(_worker, world, cp, tp, ep, timeout=300)

    @pytest.mark.multi_rank
    def test_meta_load_fills_dp_cp_ep_shards(self, tmp_path: Path) -> None:
        run_ranks(_meta_load_worker, 4, 2, 1, 2, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    def test_meta_load_fills_dp_cp_tp_ep_shards(self, tmp_path: Path) -> None:
        run_ranks(_meta_load_worker, 8, 2, 2, 2, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("save", "load"), [((1, 1, 2), (1, 1, 4)), ((1, 1, 4), (1, 1, 2))]
    )
    def test_checkpoint_resumes_unchanged_at_another_ep_degree(
        self, save: tuple[int, int, int], load: tuple[int, int, int], tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save, load, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("save", "load"), [((2, 1, 2), (4, 1, 2)), ((4, 1, 2), (2, 1, 2))]
    )
    def test_checkpoint_resumes_unchanged_at_another_cp_degree(
        self, save: tuple[int, int, int], load: tuple[int, int, int], tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save, load, str(tmp_path), timeout=300)

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("save", "load"), [((1, 2, 4), (1, 4, 4)), ((1, 4, 4), (1, 2, 4))]
    )
    def test_checkpoint_resumes_unchanged_at_another_tp_degree(
        self, save: tuple[int, int, int], load: tuple[int, int, int], tmp_path: Path
    ) -> None:
        run_ranks(_resume_worker, 8, save, load, str(tmp_path), timeout=300)
