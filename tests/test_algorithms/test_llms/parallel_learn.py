# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""One CISPO or LLM-PPO learn step under cp x tp x ep, checked against a same-world FSDP reference."""

from __future__ import annotations

import gc
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from agilerl.algorithms.cispo import CISPO
from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.distributed import FSDPConfig
from agilerl.distributed.mesh import MeshAxisName
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from tests.parallel_helpers import plain
from tests.test_algorithms.test_llms.llm_helpers import rollout_logps

SEQ = 8
GROUP = 2
SEED = 0
IMAGE_SEQ = 72
# Image runs of the two image-row layouts: one across each CP shard start
# (column 36 under flash, 64 once flex rounds the frame to 128), and two in the
# first shard. Each completion starts after its row's last image.
IMAGE_STARTS = ((34, 62), (1, 10))
COMPLETION_STARTS = (66, 14)
TEXT_COMPLETION_START = 20


@dataclass(frozen=True)
class ImageCase:
    """Image token id and the pixel layout of one image in a family's tiny model."""

    image_token_id: int
    grid: tuple[int, int]
    patch_dim: int
    tokens_per_image: int


@dataclass(frozen=True)
class FamilyCase:
    """A tiny model of one family, its LoRA targets, and the meshes it learns on.

    ``probes`` returns weights the plan shards; keys starting ``experts.`` split
    on ``ep``, the rest on ``tp``. ``meshes`` holds ``(world, cp, tp, ep)``.
    A string ``target_modules`` is a PEFT full-match regex. ``images`` gives the
    first half of the batch rows images, so the last data rank of a mesh with
    ``dp_shard > 1`` learns on text only.
    """

    model: Callable[[], nn.Module]
    target_modules: tuple[str, ...] | str
    target_parameters: tuple[str, ...]
    probes: Callable[[nn.Module], dict[str, torch.Tensor]]
    param_dtype: str
    atol: float
    rtol: float
    meshes: tuple[tuple[int, int, int, int], ...]
    strip_towers: bool = False
    images: ImageCase | None = None


@dataclass(frozen=True)
class LearnBatch:
    """Rows of one learn and the vision rows of their images, in row order."""

    token_ids: list[torch.Tensor]
    masks: list[torch.Tensor]
    rewards: torch.Tensor
    pixel_values: torch.Tensor | None = None
    image_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class LearnResult:
    """Rank 0's exported LoRA before and after one step, and this rank's step stats.

    ``tower_rows`` counts rows into the vision tower's first submodule, one per
    image on Gemma 4; ``images`` is this rank's image count.
    """

    before: dict[str, torch.Tensor]
    after: dict[str, torch.Tensor]
    grad_norm: float
    critic_grad_norm: float | None
    tower_rows: int
    images: int


def linear_weight(module: nn.Linear | LoraLinear) -> nn.Parameter:
    if isinstance(module, LoraLinear):
        return module.get_base_layer().weight
    return module.weight


def expert_weight(experts: nn.Module, name: str) -> torch.Tensor:
    if isinstance(experts, BaseTunerLayer):
        experts = experts.get_base_layer()
    return getattr(experts, name)


def global_batch(prompts: int, images: ImageCase | None) -> LearnBatch:
    generator = torch.Generator().manual_seed(SEED)
    rows = prompts * GROUP
    rewards = torch.tensor([1.0, 0.0] * prompts, dtype=torch.float32)
    if images is None:
        token_ids = [
            torch.randint(1, 28, (1, SEQ), generator=generator, dtype=torch.long)
            for _ in range(rows)
        ]
        masks = [torch.ones(1, SEQ - 1, dtype=torch.bool) for _ in range(rows)]
        return LearnBatch(token_ids, masks, rewards)
    token_ids, masks, counts = [], [], []
    for row in range(rows):
        starts, completion = (), TEXT_COMPLETION_START
        if row < rows // 2:
            starts = IMAGE_STARTS[row % 2]
            completion = COMPLETION_STARTS[row % 2]
        ids = torch.randint(1, 28, (1, IMAGE_SEQ), generator=generator)
        for start in starts:
            ids[0, start : start + images.tokens_per_image] = images.image_token_id
        mask = torch.zeros(1, IMAGE_SEQ - 1, dtype=torch.bool)
        mask[0, completion - 1 :] = True
        token_ids.append(ids)
        masks.append(mask)
        counts.append(len(starts))
    pixel_values = torch.randn(
        sum(counts), *images.grid, images.patch_dim, generator=generator
    )
    return LearnBatch(token_ids, masks, rewards, pixel_values, tuple(counts))


def local_batch(batch: LearnBatch, rank: int, replica: int) -> LearnBatch:
    """This rank's data shard; the ``replica`` = tp * cp ranks of a shard see the same rows."""
    per_rank = len(batch.token_ids) // (dist.get_world_size() // replica)
    start = (rank // replica) * per_rank
    stop = start + per_rank
    counts = batch.image_counts[start:stop]
    pixel_values = None
    if sum(counts):
        first = sum(batch.image_counts[:start])
        pixel_values = batch.pixel_values[first : first + sum(counts)]
    return LearnBatch(
        batch.token_ids[start:stop],
        batch.masks[start:stop],
        batch.rewards[start:stop],
        pixel_values,
        counts,
    )


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


def exported_lora(agent: CISPO | LLMPPO) -> dict[str, torch.Tensor]:
    """Adapters in checkpoint row order, as rank 0 exports them; other ranks get none."""
    state = agent.shard_runtime.export_model_state(agent.actor)
    return {plain(key): value.float() for key, value in state.items() if "lora_" in key}


def learn_step(
    case: FamilyCase,
    model: nn.Module,
    batch: LearnBatch,
    rank: int,
    mesh: tuple[int, int, int],
    ppo: bool = False,
    vision_cache: bool = True,
) -> LearnResult:
    """One learn of CISPO, or of LLM-PPO with a value head, on ``mesh``.

    :param vision_cache: Whether forwards of the step reuse vision tower outputs.
    """
    cp, tp, ep = mesh
    full = {name: weight.numel() for name, weight in case.probes(model).items()}
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=(
            case.target_modules
            if isinstance(case.target_modules, str)
            else list(case.target_modules)
        ),
        target_parameters=list(case.target_parameters) or None,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["summary"] if ppo else None,
    )
    # The value head draws its init from the global RNG, which earlier steps advance.
    torch.manual_seed(SEED)
    algorithm_kwargs = (
        {
            "actor_network": AutoModelForCausalLMWithValueHead(model),
            "lr_actor": 1e-2,
            "lr_critic": 1e-2,
            "batch_size": len(batch.token_ids),
            "update_epochs": 1,
        }
        if ppo
        else {
            "actor_network": model,
            "lr": 1e-2,
            "group_size": GROUP,
            "batch_size": len(batch.token_ids) // GROUP,
        }
    )
    agent = (LLMPPO if ppo else CISPO)(
        **algorithm_kwargs,
        pad_token_id=0,
        pad_token="<pad>",
        beta=0.0,
        seed=SEED,
        max_model_len=256,
        lora_config=lora_config,
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
        if not vision_cache:
            agent._vision_cache.step = nullcontext
        base = agent.actor.pretrained_model if ppo else agent.actor
        if isinstance(base, PeftModel):
            base = base.get_base_model()
        tower_rows: list[int] = []
        if case.images is not None:
            tower = base.get_encoder(modality="image")
            next(tower.children()).register_forward_pre_hook(
                lambda _module, args: tower_rows.append(args[0].shape[0])
            )
        for name, weight in case.probes(base).items():
            if name.startswith("experts."):
                assert_axis_split(name, weight, MeshAxisName.EP, ep, full[name])
            else:
                assert_axis_split(name, weight, MeshAxisName.TP, tp, full[name])
        before = exported_lora(agent)
        local = local_batch(batch, rank, cp * tp)
        result = agent.learn(
            (local.token_ids, local.masks, local.rewards),
            pixel_values=local.pixel_values,
            pixel_image_counts=(
                None if local.pixel_values is None else list(local.image_counts)
            ),
            image_token_id=None if case.images is None else case.images.image_token_id,
            sampling_logps=rollout_logps(local.masks) if ppo else None,
        )
        after = exported_lora(agent)
        return LearnResult(
            before,
            after,
            float(result["grad_norm_pre"]),
            float(result["critic_grad_norm_pre"]) if ppo else None,
            sum(tower_rows),
            sum(local.image_counts),
        )
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
    batch = global_batch(world_size, case.images)
    # Liger patches the modeling classes during the first init. FSDP does not
    # cast weights it leaves unwrapped, so the actor starts in the case dtype.
    dtype = getattr(torch, case.param_dtype)
    models = [case.model().to(dtype) for _ in range(2)]

    reference = learn_step(case, models[0], batch, rank, (1, 1, 1))
    parallel = learn_step(case, models[1], batch, rank, (cp, tp, ep))

    assert_same_step(case, parallel, reference, rank)


def ppo_learn_worker(rank: int, world_size: int, case: FamilyCase, cp: int) -> None:
    """LLM-PPO on ``cp`` matches the pure-FSDP step, and its vision cache runs each image once."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(rank)
    batch = global_batch(world_size, case.images)
    dtype = getattr(torch, case.param_dtype)
    models = [case.model().to(dtype) for _ in range(3)]

    reference = learn_step(case, models[0], batch, rank, (1, 1, 1), ppo=True)
    cached = learn_step(case, models[1], batch, rank, (cp, 1, 1), ppo=True)
    uncached = learn_step(
        case, models[2], batch, rank, (cp, 1, 1), ppo=True, vision_cache=False
    )

    assert_same_step(case, cached, reference, rank)
    assert_same_step(case, uncached, cached, rank)
    if cached.images:
        # Every forward of the step names the same images; only the first runs the tower.
        assert cached.tower_rows == cached.images, (cached.tower_rows, cached.images)
        assert uncached.tower_rows > cached.tower_rows


def assert_same_step(
    case: FamilyCase, parallel: LearnResult, reference: LearnResult, rank: int
) -> None:
    """``parallel`` took ``reference``'s step: grad norms, and on rank 0 every LoRA step."""
    for label, got, want in (
        ("grad_norm", parallel.grad_norm, reference.grad_norm),
        ("critic_grad_norm", parallel.critic_grad_norm, reference.critic_grad_norm),
    ):
        if want is None:
            continue
        assert torch.isclose(
            torch.tensor(got), torch.tensor(want), atol=case.atol, rtol=case.rtol
        ), (label, got, want)
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
