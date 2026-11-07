# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CISPO learn: DeepSeek-V4 TP, CP and EP against a same-world FSDP reference."""

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
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
)

from agilerl.algorithms.cispo import CISPO
from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.deepseek_v4 import install_deepseek_v4_patches
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import FSDPConfig
from tests.dist_ports import get_free_port

HUB_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
VOCAB = 32
EXPERTS = 4
TOP_K = 2
# One heavily compressed (ratio 128) entry needs 128 tokens.
SEQ = 136
GROUP = 2
SEED = 0
LOSS_FLOOR = 5e-2
TP_SPLIT_PROBES = ("attention.q_b_proj", "shared_experts.gate_proj")
TP_REPLICATED_PROBES = ("attention.q_a_proj", "compressor.kv_proj")

requires_2_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs 2 CUDA devices",
)
requires_4_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="needs 4 CUDA devices",
)


def _model_config() -> DeepseekV4Config:
    config = DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        n_routed_experts=EXPERTS,
        n_shared_experts=1,
        num_experts_per_tok=TOP_K,
        num_hash_layers=1,
        # Sliding, compressed sparse, and heavily compressed attention.
        compress_ratios=[0, 4, 128],
        sliding_window=32,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=4,
        hc_mult=4,
        pad_token_id=0,
    )
    config._attn_implementation = "flex_attention"
    return config


def _new_model() -> DeepseekV4ForCausalLM:
    torch.manual_seed(SEED)
    model = DeepseekV4ForCausalLM(_model_config())
    # Checkpoints ship the hash table; init leaves every token on expert 0.
    generator = torch.Generator().manual_seed(SEED)
    for layer in model.model.layers:
        if layer.mlp.is_hash:
            table = torch.stack(
                [
                    torch.randperm(EXPERTS, generator=generator)[:TOP_K]
                    for _ in range(VOCAB)
                ]
            )
            layer.mlp.gate.tid2eid.copy_(table)
    # The trainer loads Hub checkpoints in bf16.
    return model.to(torch.bfloat16)


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
    layers = model.model.layers
    return {
        "token_embedding": model.model.embed_tokens.weight,
        "attention.q_a_proj": _linear_weight(layers[0].self_attn.q_a_proj),
        "attention.q_b_proj": _linear_weight(layers[0].self_attn.q_b_proj),
        "compressor.kv_proj": _linear_weight(layers[1].self_attn.compressor.kv_proj),
        "shared_experts.gate_proj": _linear_weight(
            layers[0].mlp.shared_experts.gate_proj
        ),
        "experts.gate_up_proj": _expert_weight(layers[0].mlp.experts),
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
    """EP splits the experts and TP the split probes; attention keeps whole weights on every TP rank."""
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
        mesh_names: tuple[str, ...] = ()
        if isinstance(weight, DTensor):
            mesh_names = tuple(weight.device_mesh.mesh_dim_names or ())
        assert "ep" not in mesh_names, name
        if tp > 1 and name in TP_SPLIT_PROBES:
            assert isinstance(weight, DTensor), name
            assert mesh_names[-1] == "tp", (name, mesh_names)
            replicated = [Replicate()] * (weight.device_mesh.ndim - 1)
            tp_local = weight.redistribute(
                weight.device_mesh, [*replicated, weight.placements[-1]]
            ).to_local()
            assert tp_local.numel() * tp == full[name], (name, tp_local.numel())
            assert _held_numel(weight, rank) == full[name], name
        elif tp > 1 and name in TP_REPLICATED_PROBES:
            assert isinstance(weight, DTensor), name
            assert mesh_names[-1] == "tp", (name, mesh_names)
            assert weight.placements[-1] == Replicate(), (name, weight.placements)
            assert _held_numel(weight, rank) == full[name] * tp, name
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
    model: DeepseekV4ForCausalLM,
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
        # The patches enable flex on DeepSeek-V4; the trainer installs them before loading.
        install_deepseek_v4_patches(FAMILY_RUNTIME_CONFIGS["deepseek_v4"].patch)
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
class TestDeepseekV4ParallelLearn:
    @requires_2_cuda
    # MoE meshes need ep >= tp, so the TP run also splits experts.
    @pytest.mark.parametrize(
        ("cp", "tp", "ep"),
        [(1, 2, 2), (1, 1, 2), (2, 1, 1)],
        ids=["tp", "ep", "cp"],
    )
    def test_learn_matches_fsdp_reference(self, cp: int, tp: int, ep: int) -> None:
        _spawn(2, cp, tp, ep)

    @requires_4_cuda
    def test_cp_reshard_learn_matches_fsdp_reference(self) -> None:
        _spawn(4, 2, 1, 1)
