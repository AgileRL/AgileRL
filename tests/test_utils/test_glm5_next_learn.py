# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CISPO learn: GLM-5.3 (glm5_next) TP, CP and EP against a same-world FSDP reference."""

from __future__ import annotations

import gc
import math
import os
import traceback
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, PeftModel
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from torch.distributed.tensor import DTensor, Replicate
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
)

from agilerl.algorithms.cispo import CISPO
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import FSDPConfig
from tests.dist_ports import get_free_port

HUB_ID = "zai-org/GLM-5.3-Flash"
HIDDEN = 64
HEAD_DIM = 16
# fla's TileLang KDA backward fails layout inference at head_dim 16.
LINEAR_HEAD_DIM = 128
# Four k-pools of four tokens, so the DSA indexer keeps two plus the tail.
SEQ = 16
GROUP = 2
SEED = 0
LOSS_FLOOR = 5e-2
TP_PROBES = frozenset(
    {
        "linear_attention.q_proj",
        "dsa_attention.q_b_proj",
        "shared_experts.gate_proj",
        "vision_attention.qkv",
    }
)

requires_2_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs 2 CUDA devices",
)
requires_4_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="needs 4 CUDA devices",
)


def _model_config() -> Glm5NextConfig:
    text = Glm5NextTextConfig(
        vocab_size=32,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "indexed_attention",
            "linear_attention",
            "indexed_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        indexer_types=["full"] * 4,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_rope_head_dim=0,
        qk_nope_head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        index_n_heads=4,
        index_head_dim=HEAD_DIM,
        index_topk=8,
        index_kpool=4,
        linear_num_heads=4,
        linear_head_dim=LINEAR_HEAD_DIM,
        linear_conv_kernel_dim=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        hc_mult=4,
        pad_token_id=0,
    )
    vision = Glm5NextVisionConfig(
        depth=1,
        hidden_size=HIDDEN,
        num_heads=4,
        intermediate_size=64,
        out_hidden_size=HIDDEN,
        projection_intermediate_size=64,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        image_size=8,
    )
    config = Glm5NextConfig(text_config=text, vision_config=vision)
    config._attn_implementation = "sdpa"
    return config


def _new_model() -> Glm5NextForConditionalGeneration:
    torch.manual_seed(SEED)
    # The trainer loads Hub checkpoints in bf16.
    return Glm5NextForConditionalGeneration(_model_config()).to(torch.bfloat16)


def _batch(
    prompts: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    rows = prompts * GROUP
    token_ids = [
        torch.randint(1, 28, (1, SEQ), generator=generator, dtype=torch.long)
        for _ in range(rows)
    ]
    masks = [torch.ones(1, SEQ - 1, dtype=torch.bool) for _ in range(rows)]
    rewards = torch.tensor([1.0, 0.0] * prompts, dtype=torch.float32)
    return token_ids, masks, rewards


def _local_batch(
    batch: tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor],
    rank: int,
    cp: int,
    tp: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    token_ids, masks, rewards = batch
    replicas = cp * tp
    per_rank = len(token_ids) // (dist.get_world_size() // replicas)
    start = (rank // replicas) * per_rank
    stop = start + per_rank
    return token_ids[start:stop], masks[start:stop], rewards[start:stop]


def _lora_config() -> LoraConfig:
    info = SUPPORTED_MODEL_INFO[HUB_ID]
    return LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=sorted(info.default_lora_modules),
        target_parameters=sorted(info.parameters),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _linear_weight(module: nn.Linear | LoraLinear) -> nn.Parameter:
    if isinstance(module, LoraLinear):
        return module.get_base_layer().weight
    return module.weight


def _expert_weight(experts: nn.Module) -> torch.Tensor:
    if isinstance(experts, BaseTunerLayer):
        experts = experts.get_base_layer()
    return experts.gate_up_proj


def _probe_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    text = model.model.language_model
    return {
        "token_embedding": text.embed_tokens.weight,
        "linear_attention.q_proj": _linear_weight(text.layers[0].self_attn.q_proj),
        "dsa_attention.q_b_proj": _linear_weight(text.layers[1].self_attn.q_b_proj),
        "shared_experts.gate_proj": _linear_weight(
            text.layers[1].mlp.shared_experts.gate_proj
        ),
        "experts.gate_up_proj": _expert_weight(text.layers[1].mlp.experts),
        # The actor keeps the tower; only the rollout engine strips it.
        "vision_attention.qkv": _linear_weight(model.model.visual.blocks[0].attn.qkv),
    }


def _held_numel(weight: torch.Tensor, rank: int) -> int:
    """Elements of ``weight`` held across the world; a plain tensor counts once."""
    if not isinstance(weight, DTensor):
        return weight.numel()
    held = torch.tensor([weight.to_local().numel()], device=f"cuda:{rank}")
    dist.all_reduce(held)
    return int(held.item())


def _assert_sharded(
    actor: nn.Module, full: dict[str, int], tp: int, ep: int, rank: int
) -> None:
    """EP splits the packed experts, TP the KDA, DSA, MLP, and vision probes; FSDP the rest."""
    base = actor.get_base_model() if isinstance(actor, PeftModel) else actor
    probes = _probe_weights(base)
    experts = probes.pop("experts.gate_up_proj")
    if ep > 1:
        assert isinstance(experts, DTensor)
        mesh_names = tuple(experts.device_mesh.mesh_dim_names or ())
        axis = mesh_names.index("ep")
        assert int(experts.device_mesh.size(axis)) == ep
        placements = [Replicate()] * experts.device_mesh.ndim
        placements[axis] = experts.placements[axis]
        ep_local = experts.redistribute(experts.device_mesh, placements).to_local()
        assert ep_local.numel() * ep == full["experts.gate_up_proj"], ep_local.numel()
    else:
        if isinstance(experts, DTensor):
            assert "ep" not in tuple(experts.device_mesh.mesh_dim_names or ())
        assert _held_numel(experts, rank) == full["experts.gate_up_proj"]
    for name, weight in probes.items():
        mesh_names = (
            tuple(weight.device_mesh.mesh_dim_names or ())
            if isinstance(weight, DTensor)
            else ()
        )
        assert "ep" not in mesh_names, name
        if tp > 1 and name in TP_PROBES:
            assert "tp" in mesh_names, (name, mesh_names)
            # World equals tp here, so FSDP splits nothing further.
            assert weight.to_local().numel() * tp == full[name], name
        else:
            assert "tp" not in mesh_names, (name, mesh_names)
            assert _held_numel(weight, rank) == full[name], name


def _lora_snapshot(actor: nn.Module) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    for name, param in actor.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        local = param.to_local() if isinstance(param, DTensor) else param
        values.append(local.detach().clone())
    assert values, "actor has no trainable LoRA parameters"
    return values


def _lora_moved(before: list[torch.Tensor], actor: nn.Module) -> bool:
    after = _lora_snapshot(actor)
    assert len(after) == len(before), (len(before), len(after))
    pairs = zip(before, after, strict=True)
    return any(not torch.equal(left, right) for left, right in pairs)


def _reduce(value: float, rank: int, op: dist.ReduceOp) -> float:
    tensor = torch.tensor([value], dtype=torch.float32, device=f"cuda:{rank}")
    dist.all_reduce(tensor, op=op)
    if op == dist.ReduceOp.SUM:
        tensor /= dist.get_world_size()
    return float(tensor.item())


def _learn(
    rank: int,
    batch: tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor],
    cp: int,
    tp: int,
    ep: int,
    model: Glm5NextForConditionalGeneration,
) -> float:
    full = {name: weight.numel() for name, weight in _probe_weights(model).items()}
    agent = CISPO(
        actor_network=model,
        pad_token_id=0,
        pad_token="<pad>",
        beta=0.0,
        lr=1e-2,
        group_size=GROUP,
        batch_size=len(batch[0]) // GROUP,
        seed=SEED,
        max_model_len=256,
        lora_config=_lora_config(),
        fsdp_config=FSDPConfig(
            reshard_after_forward=True,
            activation_offload=False,
            cp=cp,
            tp=tp,
            ep=ep,
            param_dtype="bfloat16",
            reduce_dtype="float32",
            routed_expert_chunk_mib=64,
            optim_cpu_offload=False,
            cpu_offload=False,
            checkpoint_every_n_blocks=1,
        ),
        use_separate_reference_adapter=False,
        gradient_checkpointing=True,
        vllm_config=None,
        rollout_strip_multimodal_towers=True,
    )
    try:
        _assert_sharded(agent.actor, full, tp, ep, rank)
        before = _lora_snapshot(agent.actor)
        token_ids, masks, rewards = _local_batch(batch, rank, cp, tp)
        result = agent.learn((token_ids, masks, rewards))
        assert "grad_norm_pre" in result, sorted(result)
        loss = float(result["loss"])
        finite = _reduce(1.0 if math.isfinite(loss) else 0.0, rank, dist.ReduceOp.MIN)
        grad = _reduce(float(result["grad_norm_pre"]), rank, dist.ReduceOp.MIN)
        moved = _reduce(
            1.0 if _lora_moved(before, agent.actor) else 0.0,
            rank,
            dist.ReduceOp.MIN,
        )
        if rank == 0:
            print(
                f"cp={cp} tp={tp} ep={ep} loss={loss:.6f} grad_norm_pre={grad:.6f}",
                flush=True,
            )
        assert finite == 1.0, loss
        assert grad > 0.0, grad
        assert moved == 1.0
        return _reduce(loss, rank, dist.ReduceOp.SUM)
    finally:
        agent.clean_up()
        del agent
        gc.collect()
        torch.cuda.empty_cache()


def _worker(
    rank: int,
    world_size: int,
    port: int,
    cp: int,
    tp: int,
    ep: int,
    result_queue: Any,
) -> None:
    try:
        os.environ.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
            }
        )
        torch.cuda.set_device(rank)
        # One group per reference rank, so every rank sees a nonzero advantage.
        batch = _batch(prompts=world_size)
        # Liger patches the modeling classes during the first init.
        models = [_new_model() for _ in range(3)]
        ref_a = _learn(rank, batch, 1, 1, 1, models[0])
        ref_b = _learn(rank, batch, 1, 1, 1, models[1])
        parallel = _learn(rank, batch, cp, tp, ep, models[2])
        gap = abs(ref_a - ref_b)
        limit = max(10 * gap, LOSS_FLOOR)
        if rank == 0:
            print(
                f"cp={cp} tp={tp} ep={ep} world={world_size} "
                f"loss_ref={ref_a:.6f} loss_ref2={ref_b:.6f} d={gap:.6f} "
                f"loss_parallel={parallel:.6f} limit={limit:.6f}",
                flush=True,
            )
        dist.barrier()
        assert abs(parallel - ref_a) <= limit, (parallel, ref_a, ref_b, gap, limit)
        dist.barrier()
        result_queue.put((rank, "ok", None))
    except Exception:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(world_size: int, cp: int, tp: int, ep: int) -> None:
    port = get_free_port()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, port, cp, tp, ep, queue))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    try:
        results = [queue.get(timeout=900) for _ in range(world_size)]
        for proc in procs:
            proc.join(timeout=120)
            assert proc.exitcode in (0, None), proc.exitcode
        for rank, status, err in sorted(results):
            assert status == "ok", f"rank {rank}: {err}"
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)


@pytest.mark.gpu
class TestGlm5NextParallelLearn:
    @requires_2_cuda
    @pytest.mark.parametrize(
        ("cp", "tp", "ep"),
        [(1, 2, 2), (1, 1, 2), (2, 1, 1)],
        ids=["tpep", "ep", "cp"],
    )
    def test_learn_matches_fsdp_reference(self, cp: int, tp: int, ep: int) -> None:
        _spawn(2, cp, tp, ep)

    @requires_4_cuda
    def test_cp_reshard_learn_matches_fsdp_reference(self) -> None:
        _spawn(4, 2, 1, 1)
