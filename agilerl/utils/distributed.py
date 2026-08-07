"""Torch-native distributed helpers for AgileRL.

AgileRL is single-device by default. Multi-GPU LLM training initialises
``torch.distributed`` from the standard launcher environment variables
(``RANK``/``LOCAL_RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT``),
which are set by ``torchrun`` or by an orchestration layer such as Ray
actors. There is no wrapper object: ``torch.distributed`` itself is the
single source of truth for rank/world topology, and these helpers no-op on
a single device.
"""

from __future__ import annotations

import copy
import datetime
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

DTensor: Any
try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    DTensor = None

CPUOffloadPolicy: Any
MixedPrecisionPolicy: Any
fully_shard: Any
register_fsdp_forward_method: Any
try:
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
        register_fsdp_forward_method,
    )

    HAS_FSDP2 = True
except ImportError:  # pragma: no cover -- torch built without distributed support
    CPUOffloadPolicy = None
    MixedPrecisionPolicy = None
    fully_shard = None
    register_fsdp_forward_method = None
    HAS_FSDP2 = False

_LAUNCHER_ENV_VARS = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


def distributed_env_present() -> bool:
    """Whether launcher-style rendezvous env vars are set (torchrun / Ray)."""
    return all(var in os.environ for var in _LAUNCHER_ENV_VARS)


def is_distributed() -> bool:
    """Whether ``torch.distributed`` is available and initialised."""
    return dist.is_available() and dist.is_initialized()


def init_distributed(timeout_seconds: int = 1800) -> bool:
    """Initialise ``torch.distributed`` from launcher env vars.

    No-op (returns ``False``) on a single device with no launcher env. Safe
    to call repeatedly; if a process group already exists (e.g. initialised
    by a Ray orchestration layer) it is reused.

    :param timeout_seconds: Collective timeout for the process group.
    :type timeout_seconds: int
    :return: ``True`` when distributed training is active.
    :rtype: bool
    """
    if is_distributed():
        if torch.cuda.is_available():
            torch.cuda.set_device(get_local_rank())
        return True
    if not distributed_env_present():
        return False
    # Compound backend so CPU tensors (object broadcasts, metric
    # aggregation) go over gloo while CUDA tensors use nccl.
    backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    return True


def get_rank() -> int:
    """Global rank (0 on a single device)."""
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """Rank within the node, used for device selection."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return get_rank() % torch.cuda.device_count()
    return 0


def get_world_size() -> int:
    """Number of processes (1 on a single device)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """Whether this is rank 0 (always ``True`` on a single device)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronise all processes; no-op on a single device."""
    if is_distributed():
        dist.barrier()


def broadcast_object_list(objects: list, src: int = 0) -> list:
    """Broadcast a list of picklable objects from ``src`` to all ranks.

    Mutates ``objects`` in place on non-source ranks and returns it. No-op
    on a single device.

    :param objects: Objects to broadcast (same length on every rank).
    :type objects: list
    :param src: Source rank.
    :type src: int
    :return: The broadcast list.
    :rtype: list
    """
    if is_distributed():
        dist.broadcast_object_list(objects, src=src)
    return objects


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks in place; no-op on a single device."""
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def sync_grads(params) -> None:
    """Average gradients across ranks (manual data-parallel synchronisation).

    AgileRL's LLM trainer is LoRA-only, so the trainable gradient set is
    tiny; averaging it explicitly at the gradient-accumulation boundary
    replaces a DDP wrapper outright — no module wrapping, no reducer
    hooks, no unused-parameter bookkeeping. No-op on a single device.

    :param params: Parameters whose ``.grad`` should be averaged.
    """
    if not is_distributed():
        return
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | torch.device | None = None) -> str:
    """Pick the training device.

    Distributed CUDA runs are pinned to ``cuda:<local_rank>``; otherwise the
    requested device (or the best available) is used.

    :param requested: Device requested by the caller, if any.
    :type requested: str | torch.device | None
    :return: Device string.
    :rtype: str
    """
    if torch.cuda.is_available() and (is_distributed() or distributed_env_present()):
        return f"cuda:{get_local_rank()}"
    if requested is not None:
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class FSDPConfig:
    """Settings for sharding the LLM actor with PyTorch FSDP2 (``fully_shard``).

    :param reshard_after_forward: Free gathered parameters after each
        module's forward (ZeRO-3-like memory profile).
    :type reshard_after_forward: bool
    :param cpu_offload: Offload sharded parameters/gradients to CPU.
    :type cpu_offload: bool
    :param optim_cpu_offload: Offload optimizer states (AdamW m+v) to CPU,
        moving them to GPU only during ``step()``.  Parameters and gradients
        stay on GPU.  Mutually exclusive with ``cpu_offload``.  Defaults to
        ``True`` — benchmarked as memory-neutral and faster with colocated vLLM.
    :type optim_cpu_offload: bool
    :param param_dtype: Parameter dtype for the mixed-precision policy.
        Defaults to ``bfloat16`` when ``None`` (Prime-RL style).
    :type param_dtype: torch.dtype | None
    :param reduce_dtype: Gradient-reduction dtype for the mixed-precision
        policy. Defaults to ``float32`` when ``None``.
    :type reduce_dtype: torch.dtype | None
    """

    reshard_after_forward: bool = True
    cpu_offload: bool = False
    optim_cpu_offload: bool = True
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


def _transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Find per-layer transformer blocks to shard individually.

    Uses HuggingFace's ``_no_split_modules`` class-name convention where
    available; falls back to an empty list (root-only sharding).
    """
    no_split: set[str] = set()
    for module in model.modules():
        names = getattr(module, "_no_split_modules", None)
        if names:
            no_split.update(names)
    if not no_split:
        return []
    return [m for m in model.modules() if type(m).__name__ in no_split]


def _resolve_causal_lm(model: nn.Module) -> nn.Module:
    """Unwrap value-head / PEFT shells to the HuggingFace causal LM."""
    causal = model
    if hasattr(causal, "pretrained_model"):
        causal = causal.pretrained_model
    get_base = getattr(causal, "get_base_model", None)
    if callable(get_base):
        try:
            return get_base()
        except Exception:  # pragma: no cover -- PEFT API edge cases
            pass
    if hasattr(causal, "base_model"):
        causal = causal.base_model
        inner = getattr(causal, "model", None)
        if inner is not None and (
            hasattr(inner, "lm_head")
            or hasattr(inner, "embed_out")
            or hasattr(inner, "model")
        ):
            return inner
    return causal


def _language_model(causal: nn.Module) -> nn.Module | None:
    """Return the transformer body that owns ``embed_tokens`` / ``layers``."""
    inner = getattr(causal, "model", None)
    if inner is not None and (
        hasattr(inner, "embed_tokens")
        or hasattr(inner, "embeddings")
        or hasattr(inner, "layers")
    ):
        return inner
    if hasattr(causal, "embed_tokens") or hasattr(causal, "layers"):
        return causal
    return None


def _shard_embed_and_lm_head(model: nn.Module, shard_kwargs: dict) -> None:
    """Shard token embeddings (and untied ``lm_head``) as their own FSDP units.

    Matches Prime-RL's layout: embeddings reshards after use; an untied
    ``lm_head`` is a separate unit so it is not held unsharded with the root.
    ``lm_head`` is sharded alone (not joint with final norm) so an
    identity-patched backbone forward does not all-gather the head when only
    the norm runs. Tied embeddings skip the head unit so tying stays intact.
    """
    assert fully_shard is not None
    causal = _resolve_causal_lm(model)
    language = _language_model(causal)
    if language is None:
        return

    embed = getattr(language, "embed_tokens", None) or getattr(
        language, "embeddings", None
    )
    if embed is not None:
        fully_shard(embed, **shard_kwargs)

    config = getattr(causal, "config", None)
    if config is not None and bool(getattr(config, "tie_word_embeddings", False)):
        return

    lm_head = getattr(causal, "lm_head", None) or getattr(causal, "embed_out", None)
    if lm_head is not None:
        fully_shard(lm_head, **shard_kwargs)


def apply_fsdp2(
    model: nn.Module,
    config: FSDPConfig | None = None,
    ep_mesh: Any | None = None,
) -> nn.Module:
    """Shard ``model`` with FSDP2: blocks, embed/(untied) lm_head, then root.

    Parameters become DTensors in place, so any optimizer must be (re)built
    after this call. Callers must not pass a dense full-model replica on
    CUDA — use :func:`materialize_fsdp2_from_cpu_state` so weights stay on
    CPU/meta until only local shards are allocated on the compute device.

    When ``ep_mesh`` is set (``ep > 1``), packed-expert modules are
    ``fully_shard``'d on ``ep_mesh.dp_mod_ep`` first, then transformer blocks /
    embeddings / root use ``ep_mesh.hsdp`` (Prime-RL EP×FSDP compose). With
    ``ep_mesh is None`` (``ep == 1``), uses today's flat process-group FSDP.

    :param model: Model to shard (CPU or meta parameters).
    :type model: nn.Module
    :param config: Sharding settings; defaults to :class:`FSDPConfig`'s
        defaults.
    :type config: FSDPConfig | None
    :param ep_mesh: Optional Expert Parallel mesh views for EP×FSDP compose.
    :type ep_mesh: ExpertParallelMesh | None
    :return: The sharded model (same object).
    :rtype: nn.Module
    """
    if not HAS_FSDP2:
        msg = "FSDP2 requires a torch build with distributed support."
        raise RuntimeError(msg)
    assert fully_shard is not None
    if not is_distributed():
        msg = (
            "FSDP2 sharding requires an initialised process group. Launch "
            "with torchrun (or have your orchestration layer set the "
            "rendezvous env vars) so init_distributed() succeeds."
        )
        raise RuntimeError(msg)
    config = config or FSDPConfig()

    # Prime-RL style: bf16 params for compute, fp32 reduction. Callers can
    # override via FSDPConfig; ``None`` means use these defaults.
    param_dtype = (
        config.param_dtype if config.param_dtype is not None else torch.bfloat16
    )
    reduce_dtype = (
        config.reduce_dtype if config.reduce_dtype is not None else torch.float32
    )
    kwargs: dict = {
        "reshard_after_forward": config.reshard_after_forward,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
    }
    if config.cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()

    if ep_mesh is not None:
        from agilerl.utils.expert_parallel import iter_packed_expert_modules

        expert_kwargs = {**kwargs, "mesh": ep_mesh.dp_mod_ep}
        for _name, experts in iter_packed_expert_modules(model):
            fully_shard(experts, **expert_kwargs)
        kwargs = {**kwargs, "mesh": ep_mesh.hsdp}

    for block in _transformer_blocks(model):
        fully_shard(block, **kwargs)
    _shard_embed_and_lm_head(model, kwargs)
    fully_shard(model, **kwargs)
    # PEFT ``generate`` delegates to ``base_model.generate`` and never enters
    # the FSDP-rooted ``forward``, so root shards (e.g. embed_tokens) stay
    # DTensors against plain ``input_ids``. Register ``generate`` so FSDP2
    # all-gathers the same way as ``forward``.
    if hasattr(model, "generate"):
        assert register_fsdp_forward_method is not None
        register_fsdp_forward_method(model, "generate")
    return model


def _restore_after_to_empty(model: nn.Module) -> None:
    """Restore tying and modules that ``to_empty`` leaves unbound."""
    seen: set[int] = set()
    for module in model.modules():
        tie = getattr(module, "tie_weights", None)
        if not callable(tie):
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        try:
            tie()
        except Exception:  # pragma: no cover -- model-specific tie edge cases
            pass


def materialize_fsdp2_from_cpu_state(
    model: nn.Module,
    device: str | torch.device,
    config: FSDPConfig | None = None,
    ep_mesh: Any | None = None,
) -> nn.Module:
    """Shard a CPU-resident model without placing a dense full replica on GPU.

    Captures a CPU state dict, moves parameters to meta, optionally applies
    Expert Parallel ``Shard(0)`` on packed experts, applies FSDP2, allocates
    empty sharded storages on ``device`` (or CPU when ``config.cpu_offload``),
    then scatters the CPU state into DTensor shards.

    :param model: Dense actor on CPU (base + LoRA already attached).
    :type model: nn.Module
    :param device: Compute device for sharded parameter storage.
    :type device: str | torch.device
    :param config: FSDP2 settings.
    :type config: FSDPConfig | None
    :param ep_mesh: Optional :class:`~agilerl.utils.expert_parallel.ExpertParallelMesh`
        from ``build_expert_parallel_mesh``. When set, packed experts (and
        stacked expert LoRA) are ``Shard``'d on the EP dim before FSDP2. Flat
        PG FSDP is used for ``ep=1`` (``ep_mesh is None``).
    :type ep_mesh: ExpertParallelMesh | None
    :return: The sharded model (same object).
    :rtype: nn.Module
    """
    from agilerl.utils.llm_utils import load_full_state_dict

    config = config or FSDPConfig()
    cpu_state = {
        key: value.detach().to("cpu").contiguous()
        for key, value in model.state_dict().items()
    }
    model.to_empty(device="meta")
    if ep_mesh is not None:
        from agilerl.utils.expert_parallel import apply_expert_parallel

        # Parallelize on meta modules so EP placements exist before FSDP2;
        # ``load_full_state_dict`` then scatters dense CPU weights into the
        # EP (+ FSDP) local shards without a CUDA full-MoE densify.
        apply_expert_parallel(model, ep_mesh.ep)
    apply_fsdp2(model, config, ep_mesh=ep_mesh)
    target = torch.device("cpu") if config.cpu_offload else torch.device(device)
    model.to_empty(device=target)
    _restore_after_to_empty(model)
    load_full_state_dict(model, cpu_state, strict=False)
    del cpu_state
    return model


def shard_dataloader_kwargs(dataset, shuffle: bool = True) -> dict[str, Any]:
    """DataLoader kwargs that shard ``dataset`` across ranks.

    Returns ``{"sampler": DistributedSampler(...)}`` when distributed (the
    caller must not also pass ``shuffle``), or ``{"shuffle": shuffle}`` on a
    single device.

    :param dataset: Map-style dataset to shard.
    :param shuffle: Whether to shuffle.
    :type shuffle: bool
    :return: Keyword arguments for ``torch.utils.data.DataLoader``.
    :rtype: dict
    """
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler

        return {
            "sampler": DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=shuffle,
            )
        }
    return {"shuffle": shuffle}


class CPUOffloadOptimizer:
    """Wrap an optimizer so optimizer states (AdamW m+v) stay on CPU.

    States are moved to GPU only for ``step()``, then back to CPU.  Parameters
    and gradients remain on GPU throughout, so compute is unaffected.  With
    activation checkpointing, activations and optimizer states are never on
    GPU at the same time: peak memory becomes ``max(activations, opt_states)``
    instead of ``sum``.

    Handles FSDP2 ``DTensor`` optimizer states by swapping ``_local_tensor``
    between CPU and GPU while preserving the DTensor wrapper.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, pin_memory: bool = True):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._initialized = False

    def _move_states(self, device: str) -> None:
        for p in self.optimizer.state:
            state = self.optimizer.state[p]
            for k, v in state.items():
                if isinstance(v, DTensor):
                    local = v._local_tensor
                    if device == "cpu":
                        new_local = local.to("cpu")
                        if self.pin_memory and not new_local.is_pinned():
                            new_local = new_local.pin_memory()
                    else:
                        new_local = local.to(device, non_blocking=True)
                    new_dt = copy.copy(v)
                    new_dt._local_tensor = new_local
                    state[k] = new_dt
                elif isinstance(v, torch.Tensor):
                    if device == "cpu":
                        new_v = v.to("cpu")
                        if self.pin_memory and not new_v.is_pinned():
                            new_v = new_v.pin_memory()
                        state[k] = new_v
                    else:
                        state[k] = v.to(device, non_blocking=True)

    def step(self, closure=None):
        if not self._initialized:
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            self._initialized = True
            return result
        self._move_states("cuda")
        torch.cuda.synchronize()
        result = self.optimizer.step(closure)
        self._move_states("cpu")
        return result

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()
        sd = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
        return sd

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self._move_states("cpu")
        self._initialized = True

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    @property
    def state(self):
        return self.optimizer.state

    @property
    def base_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer
