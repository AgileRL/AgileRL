# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""One CISPO learn step under cp x tp x ep, checked against a same-world FSDP reference."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from agilerl.algorithms.cispo import CISPO
from agilerl.distributed import FSDPConfig
from agilerl.distributed.mesh import MeshAxisName
from tests.parallel_helpers import plain

SEQ = 8
GROUP = 2
SEED = 0


@dataclass(frozen=True)
class FamilyCase:
    """A tiny model of one family, its LoRA targets, and the meshes it learns on.

    ``probes`` returns weights the plan shards; keys starting ``experts.`` split
    on ``ep``, the rest on ``tp``. ``meshes`` holds ``(world, cp, tp, ep)``.
    """

    model: Callable[[], nn.Module]
    target_modules: tuple[str, ...]
    target_parameters: tuple[str, ...]
    probes: Callable[[nn.Module], dict[str, torch.Tensor]]
    param_dtype: str
    atol: float
    rtol: float
    meshes: tuple[tuple[int, int, int, int], ...]
    strip_towers: bool = False


@dataclass(frozen=True)
class LearnResult:
    """Rank 0's exported LoRA before and after one step, and the pre-clip grad norm."""

    before: dict[str, torch.Tensor]
    after: dict[str, torch.Tensor]
    grad_norm: float


def linear_weight(module: nn.Linear | LoraLinear) -> nn.Parameter:
    if isinstance(module, LoraLinear):
        return module.get_base_layer().weight
    return module.weight


def expert_weight(experts: nn.Module, name: str) -> torch.Tensor:
    if isinstance(experts, BaseTunerLayer):
        experts = experts.get_base_layer()
    return getattr(experts, name)


def global_batch(
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


def local_batch(
    batch: tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor],
    rank: int,
    replica: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """This rank's data shard; the ``replica`` = tp * cp ranks of a shard see the same rows."""
    token_ids, masks, rewards = batch
    per_rank = len(token_ids) // (dist.get_world_size() // replica)
    start = (rank // replica) * per_rank
    stop = start + per_rank
    return token_ids[start:stop], masks[start:stop], rewards[start:stop]


def assert_axis_split(
    name: str, weight: torch.Tensor, axis: str, degree: int, full: int
) -> None:
    """``weight`` splits ``degree`` ways on mesh axis ``axis``, or not at all when degree is 1."""
    names = ()
    if isinstance(weight, DTensor):
        names = tuple(weight.device_mesh.mesh_dim_names or ())
    if degree == 1:
        assert axis not in names, (name, names)
        return
    assert axis in names, (name, names)
    index = names.index(axis)
    placements = [Replicate()] * weight.device_mesh.ndim
    placements[index] = weight.placements[index]
    local = weight.redistribute(weight.device_mesh, placements).to_local()
    assert local.numel() * degree == full, (name, local.numel(), full)


def exported_lora(agent: CISPO) -> dict[str, torch.Tensor]:
    """Adapters in checkpoint row order, as rank 0 exports them; other ranks get none."""
    state = agent.shard_runtime.export_model_state(agent.actor)
    return {plain(key): value.float() for key, value in state.items() if "lora_" in key}


def learn_step(
    case: FamilyCase,
    model: nn.Module,
    batch: tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor],
    rank: int,
    mesh: tuple[int, int, int],
) -> LearnResult:
    cp, tp, ep = mesh
    full = {name: weight.numel() for name, weight in case.probes(model).items()}
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
        lora_config=LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=list(case.target_modules),
            target_parameters=list(case.target_parameters) or None,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        fsdp_config=FSDPConfig(
            reshard_after_forward=True,
            activation_offload=False,
            cp=cp,
            tp=tp,
            ep=ep,
            param_dtype=case.param_dtype,
            reduce_dtype="float32",
            routed_expert_chunk_mib=64,
            optim_cpu_offload=False,
            cpu_offload=False,
            checkpoint_every_n_blocks=1,
        ),
        use_separate_reference_adapter=False,
        gradient_checkpointing=True,
        vllm_config=None,
        rollout_strip_multimodal_towers=case.strip_towers,
    )
    try:
        # AdamW's first step is lr * g / (|g| + eps). Clipping to max_grad_norm
        # 0.1 keeps every |g| well under eps 1, so the step follows g, not sign(g).
        for group in agent.optimizer.optimizer.param_groups:
            group["eps"] = 1.0
        base = (
            agent.actor.get_base_model()
            if isinstance(agent.actor, PeftModel)
            else agent.actor
        )
        for name, weight in case.probes(base).items():
            if name.startswith("experts."):
                assert_axis_split(name, weight, MeshAxisName.EP, ep, full[name])
            else:
                assert_axis_split(name, weight, MeshAxisName.TP, tp, full[name])
        before = exported_lora(agent)
        result = agent.learn(local_batch(batch, rank, cp * tp))
        after = exported_lora(agent)
        return LearnResult(before, after, float(result["grad_norm_pre"]))
    finally:
        agent.clean_up()
        del agent
        gc.collect()
        torch.cuda.empty_cache()


def learn_worker(
    rank: int,
    world_size: int,
    case: FamilyCase,
    cp: int,
    tp: int,
    ep: int,
) -> None:
    """One learn step on ``(cp, tp, ep)`` matches the pure-FSDP step on the same world and batch."""
    # TF32 off: fp32 cases then differ from the reference only in reduction order.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(rank)
    # One group per reference rank, so every rank sees a nonzero advantage.
    batch = global_batch(prompts=world_size)
    # Liger patches the modeling classes during the first init. FSDP does not
    # cast weights it leaves unwrapped, so the actor starts in the case dtype.
    dtype = getattr(torch, case.param_dtype)
    models = [case.model().to(dtype) for _ in range(2)]

    reference = learn_step(case, models[0], batch, rank, (1, 1, 1))
    parallel = learn_step(case, models[1], batch, rank, (cp, tp, ep))

    assert torch.isclose(
        torch.tensor(parallel.grad_norm),
        torch.tensor(reference.grad_norm),
        atol=case.atol,
        rtol=case.rtol,
    ), (parallel.grad_norm, reference.grad_norm)
    if rank != 0:
        return
    assert reference.before, "actor has no LoRA parameters"
    assert any(
        not torch.equal(reference.after[name], value)
        for name, value in reference.before.items()
    ), "the reference step left every LoRA parameter unchanged"
    assert parallel.before.keys() == reference.before.keys()
    # EP exports packed-expert lora_B as [out, r, E], a view of the reference [out, E*r].
    errors: list[str] = []
    for name, expected in reference.before.items():
        if not torch.equal(parallel.before[name].reshape(expected.shape), expected):
            errors.append(f"{name}: initial LoRA differs from the reference")
    expected_steps = {
        name: expected - reference.before[name]
        for name, expected in reference.after.items()
    }
    # Reduction-order noise scales with the model's largest grad, not with each
    # tensor: rarely routed experts take near-zero steps.
    atol = case.rtol * max(step.abs().max().item() for step in expected_steps.values())
    for name, expected_step in expected_steps.items():
        step = (parallel.after[name] - parallel.before[name]).reshape(
            expected_step.shape
        )
        if not torch.allclose(step, expected_step, atol=atol, rtol=case.rtol):
            diff = (step - expected_step).abs().max().item()
            errors.append(f"{name}: step max abs diff {diff}, atol {atol}")
    assert not errors, "\n".join(errors)
