# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Actor shard runtime: prepare, export/import, and adapter tensor ops for dense vs FSDP2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import SequentialLR

from agilerl.distributed.context_parallel import validate_cp_heads
from agilerl.distributed.expert_parallel import (
    ParallelMesh,
    build_parallel_mesh,
    iter_packed_expert_modules,
    tp_data_parallel_size,
    validate_actor_ep,
)
from agilerl.distributed.fsdp import (
    CPUOffloadOptimizer,
    FSDPConfig,
    _ep_expert_live_keys,
    _scatter_ep_expert_slices,
    canonical_fsdp_param_fqn,
    materialize_dtensors,
    materialize_fsdp2_from_cpu_state,
    set_full_model_state_dict,
)
from agilerl.distributed.process import (
    get_world_size,
    raise_on_any_rank,
    sync_grads,
)
from agilerl.utils.phase_timer import PhaseTimer

if TYPE_CHECKING:
    from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
    from agilerl.utils.algo_utils import CosineLRScheduleConfig


@dataclass
class PrepareResult:
    """Actor, optimizer, and scheduler after :meth:`BaseRuntime.prepare_actor`."""

    actor: nn.Module
    optimizer: OptimizerWrapper
    lr_scheduler: SequentialLR | None


@runtime_checkable
class PeftBaseModel(Protocol):
    """PEFT wrapper that exposes the inner pretrained module."""

    def get_base_model(self) -> nn.Module: ...


@runtime_checkable
class GradientCheckpointable(Protocol):
    """Module that implements activation checkpointing."""

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, bool]
    ) -> None: ...


def _supports_gradient_checkpointing(module: nn.Module) -> bool:
    """True when the class sets Hugging Face's checkpointing flag."""
    # Class attribute. Absent means the module does not support checkpointing.
    flag = getattr(type(module), "supports_gradient_checkpointing", False)
    return flag is True


def _checkpointable_module(module: nn.Module) -> GradientCheckpointable | None:
    """Causal LM, or its ``language_model`` child, when the class opts in."""
    current = module
    if isinstance(current, PeftBaseModel):
        base = current.get_base_model()
        if isinstance(base, nn.Module):
            current = base
    if isinstance(current, GradientCheckpointable) and _supports_gradient_checkpointing(
        current
    ):
        return current
    language_model = getattr(current, "language_model", None)
    if isinstance(
        language_model, GradientCheckpointable
    ) and _supports_gradient_checkpointing(language_model):
        return language_model
    return None


def gradient_checkpointing_module(model: nn.Module) -> GradientCheckpointable:
    """Module that implements ``gradient_checkpointing_enable``."""
    found = _checkpointable_module(model)
    if found is not None:
        return found
    # Value-head shells keep the causal LM (often PEFT-wrapped) on this attribute.
    shell = model
    if isinstance(model, PeftBaseModel):
        base = model.get_base_model()
        if isinstance(base, nn.Module):
            shell = base
    pretrained = getattr(shell, "pretrained_model", None)
    if isinstance(pretrained, nn.Module):
        found = _checkpointable_module(pretrained)
        if found is not None:
            return found
    msg = f"{type(model).__name__} does not support gradient checkpointing"
    raise TypeError(msg)


def _optimizer_state_as_dict(optimizer: OptimizerWrapper) -> dict[str, Any]:
    """``optimizer.state_dict()``; refuse list-shaped multi-opt state."""
    state = optimizer.state_dict()
    if not isinstance(state, dict):
        msg = "optimizer.state_dict() must return a dict"
        raise TypeError(msg)
    return state


def _place_param_optimizer_state(
    param: torch.Tensor,
    saved_state: dict[str, Any],
    *,
    step_on_device: bool,
) -> dict[str, Any]:
    """Move one parameter's full optimizer state onto that parameter's placement."""
    param_state: dict[str, Any] = {}
    for key, value in saved_state.items():
        # Adam keeps ``step`` on CPU unless the group is fused
        # or capturable, as ``Optimizer.load_state_dict`` does.
        if not isinstance(value, torch.Tensor):
            param_state[key] = value
        elif key == "step" and not step_on_device:
            param_state[key] = value.cpu()
        elif isinstance(param, DTensor) and value.dim() > 0:
            param_state[key] = distribute_tensor(
                value.to(param.device_mesh.device_type),
                param.device_mesh,
                param.placements,
            )
        else:
            param_state[key] = value.to(param.device)
    return param_state


def _lora_adapter_pairs(
    actor: nn.Module,
    source_adapter: str,
    target_adapter: str,
) -> dict[str, tuple[nn.Parameter, nn.Parameter]]:
    """Map each target LoRA FQN to ``(source_param, target_param)``."""
    source_params: dict[str, nn.Parameter] = {}
    target_params: dict[str, tuple[str, nn.Parameter]] = {}
    for name, param in actor.named_parameters():
        if "lora" not in name:
            continue
        if f".{source_adapter}." in name:
            source_params[name.replace(f".{source_adapter}.", ".", 1)] = param
        elif f".{target_adapter}." in name:
            key = name.replace(f".{target_adapter}.", ".", 1)
            target_params[key] = (name, param)
    if not source_params:
        msg = f"No LoRA tensors found for source adapter '{source_adapter}'."
        raise ValueError(msg)
    if not target_params:
        msg = f"No LoRA tensors found for target adapter '{target_adapter}'."
        raise ValueError(msg)
    missing = [key for key in source_params if key not in target_params]
    if missing:
        msg = (
            f"Target adapter '{target_adapter}' is missing {len(missing)} LoRA tensors "
            f"present in source adapter '{source_adapter}'."
        )
        raise ValueError(msg)
    return {
        target_params[key][0]: (source, target_params[key][1])
        for key, source in source_params.items()
    }


def _scalar_grad_norm(norm: torch.Tensor) -> torch.Tensor:
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    return norm.detach().reshape(()).float()


def clip_param_group_grad_norm_(
    params: list[nn.Parameter], max_norm: float
) -> torch.Tensor:
    """Clip one optimizer group when FSDP2 mixes DTensor and Tensor grads.

    ``fully_shard(..., ignored_params=...)`` leaves tiny parameters replicated.
    ``torch.nn.utils.clip_grad_norm_`` then foreach-norms a mixed list and
    raises. Clip each kind, then scale every grad by the joint total.

    :param params: Parameters of one optimizer group.
    :type params: list[nn.Parameter]
    :param max_norm: Maximum total gradient norm.
    :type max_norm: float
    :return: Total gradient norm before clipping.
    :rtype: torch.Tensor
    """
    trainable = [param for param in params if param.grad is not None]
    if not trainable:
        return torch.zeros((), dtype=torch.float32)

    sharded = [param for param in trainable if isinstance(param, DTensor)]
    replicated = [param for param in trainable if not isinstance(param, DTensor)]
    if not sharded:
        return clip_grad_norm_(replicated, max_norm=max_norm)
    if not replicated:
        return clip_grad_norm_(sharded, max_norm=max_norm)

    shard_norm = _scalar_grad_norm(clip_grad_norm_(sharded, max_norm=float("inf")))
    repl_norm = _scalar_grad_norm(clip_grad_norm_(replicated, max_norm=float("inf")))
    total = torch.sqrt(shard_norm * shard_norm + repl_norm * repl_norm)
    clip_coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for param in trainable:
        grad = param.grad
        if grad is not None:
            grad.mul_(clip_coef.to(grad.device))
    return total


@dataclass(frozen=True)
class OptimizerStep:
    """Global gradient norms and learning rate from one optimizer step."""

    grad_norm_pre: float
    grad_norm_post: float
    lr: float | None


def clip_param_groups(
    param_groups: list[dict[str, Any]],
    max_grad_norm: float | None,
    clip_fn: Callable[[list[nn.Parameter], float], torch.Tensor],
) -> tuple[float, float]:
    """Clip every group by one coefficient from the L2 norm over all groups.

    Matches ``torch.nn.utils.clip_grad_norm_`` over the union of all grads.
    Groups may sit on different DTensor meshes, so each is measured on its own.

    :param param_groups: Optimizer param groups.
    :type param_groups: list[dict[str, Any]]
    :param max_grad_norm: Clip threshold for the total norm, or ``None`` to
        only measure.
    :type max_grad_norm: float | None
    :param clip_fn: Clips one group in place and returns its pre-clip norm;
        called with ``max_norm=inf`` so it only measures.
    :type clip_fn: Callable[[list[nn.Parameter], float], torch.Tensor]
    :return: ``(pre, post)`` L2 norms over every group.
    :rtype: tuple[float, float]
    """
    group_norms = [
        float(_scalar_grad_norm(clip_fn(group["params"], float("inf"))))
        for group in param_groups
    ]
    total = sum(norm * norm for norm in group_norms) ** 0.5
    if max_grad_norm is None:
        return total, total
    # min(nan, 1.0) is nan, so a non-finite total reaches the grads as in torch.
    clip_coef = min(max_grad_norm / (total + 1e-6), 1.0)
    for group in param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.mul_(clip_coef)
    return total, total * clip_coef


def _step_result(
    grad_norm_pre: float,
    grad_norm_post: float,
    lr_scheduler: SequentialLR | None,
) -> OptimizerStep:
    if lr_scheduler is None:
        return OptimizerStep(grad_norm_pre, grad_norm_post, lr=None)
    lr_scheduler.step()
    return OptimizerStep(
        grad_norm_pre, grad_norm_post, lr=float(lr_scheduler.get_last_lr()[0])
    )


class BaseRuntime(ABC):
    """Behaviors that differ between unsharded and FSDP2-sharded actors."""

    # Backward calls since the last optimizer step.
    _pending_micro_batches: int = 0

    def __init__(self) -> None:
        """Create the phase timer :meth:`backward` marks while a learn window is open."""
        self.phase_timer = PhaseTimer()

    def micro_batches_until_step(self, gradient_accumulation_steps: int) -> int:
        """Backward calls up to and including the one that steps the optimizer.

        :param gradient_accumulation_steps: Micro-batches per optimizer step.
        :type gradient_accumulation_steps: int
        :return: Micro-batches the next optimizer step still waits for.
        :rtype: int
        """
        return max(gradient_accumulation_steps - self._pending_micro_batches, 1)

    def _count_micro_batch(self, gradient_accumulation_steps: int) -> bool:
        """Count one backward call; return whether it steps the optimizer.

        :param gradient_accumulation_steps: Micro-batches per optimizer step.
        :type gradient_accumulation_steps: int
        :return: Whether this micro-batch closes the accumulation window.
        :rtype: bool
        """
        self._pending_micro_batches += 1
        if self._pending_micro_batches < gradient_accumulation_steps:
            return False
        self._pending_micro_batches = 0
        return True

    @property
    @abstractmethod
    def is_sharded(self) -> bool:
        """Whether this runtime shards actor parameters."""

    def data_parallel_world(self, process_world_size: int) -> int:
        """Data-parallel replica count for a process-group world of this size."""
        return process_world_size

    def data_parallel_rank(self, process_rank: int) -> int:
        """Data-parallel rank for this process rank."""
        return process_rank

    @abstractmethod
    def prepare_actor(
        self,
        actor: nn.Module,
        device: str | torch.device,
        colocated: bool,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None,
        lr: float,
        lr_critic: float | None,
        restore_adapter_trainability: Callable[[list[str]], None],
        gradient_checkpointing: bool = False,
    ) -> PrepareResult:
        """Place the actor for training and return optimizer plus scheduler.

        :param actor: Actor module (dense or FSDP2-sharded).
        :type actor: nn.Module
        :param device: Training device.
        :type device: str | torch.device
        :param colocated: Whether vLLM generation is colocated with training.
        :type colocated: bool
        :param cosine_lr_schedule_config: Warmup-cosine schedule, or ``None`` for constant LR.
        :type cosine_lr_schedule_config: CosineLRScheduleConfig | None
        :param lr: Actor learning rate.
        :type lr: float
        :param lr_critic: Critic learning rate, or ``None`` without a critic.
        :type lr_critic: float | None
        :param restore_adapter_trainability: Re-enables grads on the named adapters.
        :type restore_adapter_trainability: Callable[[list[str]], None]
        :param gradient_checkpointing: Enable HF gradient checkpointing.
        :type gradient_checkpointing: bool
        :return: Placed actor, optimizer and scheduler.
        :rtype: PrepareResult
        """

    @abstractmethod
    def export_model_state(self, model: nn.Module) -> dict[str, Any]:
        """Return a full model state dict (FSDP2: on CPU, rank 0 only).

        :param model: Model to export.
        :type model: nn.Module
        :return: Full state dict.
        :rtype: dict[str, Any]
        """

    @abstractmethod
    def import_model_state(
        self, model: nn.Module, state: dict[str, Any], strict: bool = False
    ) -> None:
        """Load a full model state dict onto ``model``.

        :param model: Model to load into.
        :type model: nn.Module
        :param state: Full state dict.
        :type state: dict[str, Any]
        :param strict: Raise on missing or unexpected keys.
        :type strict: bool
        """

    @abstractmethod
    def export_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
    ) -> dict[str, Any]:
        """Return a full optimizer state dict (FSDP2: on CPU, rank 0 only).

        :param actor: Actor module (dense or FSDP2-sharded).
        :type actor: nn.Module
        :param optimizer: Optimizer to export.
        :type optimizer: OptimizerWrapper
        :return: Full optimizer state dict.
        :rtype: dict[str, Any]
        """

    @abstractmethod
    def import_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        optimizer_state: dict[str, Any],
    ) -> None:
        """Load a full optimizer state dict onto ``optimizer``.

        :param actor: Actor module (dense or FSDP2-sharded).
        :type actor: nn.Module
        :param optimizer: Optimizer to load into.
        :type optimizer: OptimizerWrapper
        :param optimizer_state: Full optimizer state dict.
        :type optimizer_state: dict[str, Any]
        """

    @abstractmethod
    def gather_layer(
        self,
        layer: nn.Linear,
        device: torch.device | str,
    ) -> AbstractContextManager[tuple[torch.Tensor, torch.Tensor | None]]:
        """Yield full local ``(weight, bias)`` for ``layer`` on ``device``.

        :param layer: Linear layer to gather.
        :type layer: nn.Linear
        :param device: Device for the yielded tensors.
        :type device: torch.device | str
        """

    @abstractmethod
    def copy_adapter_tensors(
        self,
        actor: nn.Module,
        source_adapter: str,
        target_adapter: str,
    ) -> None:
        """Copy LoRA tensors from ``source_adapter`` to ``target_adapter``.

        :param actor: Actor module (dense or FSDP2-sharded).
        :type actor: nn.Module
        :param source_adapter: Adapter to copy from.
        :type source_adapter: str
        :param target_adapter: Adapter to overwrite.
        :type target_adapter: str
        """

    @abstractmethod
    def import_adapter_tensors(
        self,
        actor: nn.Module,
        peft_model: nn.Module,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        """Write LoRA tensors from ``checkpoint_dir`` onto ``actor``.

        :param actor: Actor module (dense or FSDP2-sharded).
        :type actor: nn.Module
        :param peft_model: PEFT model that owns the adapter.
        :type peft_model: nn.Module
        :param checkpoint_dir: Directory holding ``<adapter>/adapter_model.safetensors``.
        :type checkpoint_dir: str
        :param adapter_name: Adapter to load.
        :type adapter_name: str
        """

    @abstractmethod
    def actor_compute_device(
        self, actor: nn.Module, fallback: torch.device
    ) -> torch.device:
        """Device for HF generate inputs.

        :param actor: Actor module (dense or FSDP2-sharded).
        :type actor: nn.Module
        :param fallback: Device used when the actor has no parameters or is sharded.
        :type fallback: torch.device
        :return: Device for generate inputs.
        :rtype: torch.device
        """

    @abstractmethod
    def backward(
        self,
        loss: torch.Tensor,
        optimizer: OptimizerWrapper,
        gradient_accumulation_steps: int,
        actor: nn.Module,
        max_grad_norm: float | None = None,
        lr_scheduler: SequentialLR | None = None,
    ) -> OptimizerStep | None:
        """Backward one micro-batch; step on the accumulation boundary.

        :param loss: Micro-batch loss.
        :type loss: torch.Tensor
        :param optimizer: Optimizer to step on the boundary.
        :type optimizer: OptimizerWrapper
        :param gradient_accumulation_steps: Micro-batches per optimizer step.
        :type gradient_accumulation_steps: int
        :param actor: Actor module (dense or FSDP2-sharded).
        :type actor: nn.Module
        :param max_grad_norm: Clip threshold, or ``None`` to skip clipping.
        :type max_grad_norm: float | None
        :param lr_scheduler: Scheduler to step on the boundary.
        :type lr_scheduler: SequentialLR | None
        :return: Norms and new learning rate on a step boundary, else ``None``.
        :rtype: OptimizerStep | None
        """


class DPRuntime(BaseRuntime):
    """Single-device and data-parallel (unsharded) runtime."""

    @property
    def is_sharded(self) -> bool:
        return False

    def prepare_actor(
        self,
        actor: nn.Module,
        device: str | torch.device,
        colocated: bool,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None,
        lr: float,
        lr_critic: float | None,
        restore_adapter_trainability: Callable[[list[str]], None],
        gradient_checkpointing: bool = False,
    ) -> PrepareResult:
        # heavy stack: llm_utils (optimizer/scheduler); export/import/gather do not load it
        from agilerl.utils.llm_utils import (  # lazy import: optimizer/scheduler unused by export/import/gather
            make_llm_optimizer,
            make_llm_scheduler,
        )

        target = torch.device(device)
        param = next(actor.parameters(), None)
        placed: nn.Module = actor
        if (
            param is None
            or param.device.type != target.type
            or (target.index is not None and param.device.index != target.index)
        ):
            moved = actor.to(target)
            if not isinstance(moved, nn.Module):
                msg = "actor.to() must return nn.Module"
                raise TypeError(msg)
            placed = moved
        if gradient_checkpointing:
            checkpoint_module = gradient_checkpointing_module(placed)
            checkpoint_module.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        optimizer = make_llm_optimizer(placed, lr, lr_critic)
        return PrepareResult(
            actor=placed,
            optimizer=optimizer,
            lr_scheduler=make_llm_scheduler(optimizer, cosine_lr_schedule_config, lr),
        )

    def export_model_state(self, model: nn.Module) -> dict[str, Any]:
        return model.state_dict()

    def import_model_state(
        self, model: nn.Module, state: dict[str, Any], strict: bool = False
    ) -> None:
        model.load_state_dict(state, strict=strict)

    def export_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
    ) -> dict[str, Any]:
        return _optimizer_state_as_dict(optimizer)

    def import_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        optimizer_state: dict[str, Any],
    ) -> None:
        optimizer.load_state_dict(optimizer_state)

    @contextmanager
    def gather_layer(
        self,
        layer: nn.Linear,
        device: torch.device | str,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor | None], None, None]:
        yield layer.weight, layer.bias

    def copy_adapter_tensors(
        self,
        actor: nn.Module,
        source_adapter: str,
        target_adapter: str,
    ) -> None:
        pairs = _lora_adapter_pairs(actor, source_adapter, target_adapter)
        for source, target in pairs.values():
            target.data.copy_(source.data)

    def import_adapter_tensors(
        self,
        actor: nn.Module,
        peft_model: nn.Module,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        # optional extra: peft, safetensors (adapter checkpoint load)
        from peft import set_peft_model_state_dict  # lazy import of optional peft extra
        from safetensors.torch import load_file  # lazy import of optional extra

        load_device = str(next(actor.parameters()).device)
        adapter_state = load_file(
            f"{checkpoint_dir}/{adapter_name}/adapter_model.safetensors",
            device=load_device,
        )
        with torch.no_grad():
            set_peft_model_state_dict(
                peft_model, adapter_state, adapter_name=adapter_name
            )

    def actor_compute_device(
        self, actor: nn.Module, fallback: torch.device
    ) -> torch.device:
        param = next(actor.parameters(), None)
        return fallback if param is None else param.device

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: OptimizerWrapper,
        gradient_accumulation_steps: int,
        actor: nn.Module,
        max_grad_norm: float | None = None,
        lr_scheduler: SequentialLR | None = None,
    ) -> OptimizerStep | None:
        """Average LoRA grads across ranks when unsharded."""
        is_step_boundary = self._count_micro_batch(gradient_accumulation_steps)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        self.phase_timer.mark("backward")
        if not is_step_boundary:
            return None

        inner = optimizer._single_optimizer()
        sync_grads([param for group in inner.param_groups for param in group["params"]])
        self.phase_timer.mark("grad_sync")
        grad_norm_pre, grad_norm_post = clip_param_groups(
            inner.param_groups, max_grad_norm, clip_grad_norm_
        )
        optimizer.step()
        optimizer.zero_grad()
        self.phase_timer.mark("optim")
        return _step_result(grad_norm_pre, grad_norm_post, lr_scheduler)


class FSDPRuntime(BaseRuntime):
    """FSDP2-sharded runtime: wrap, gathers, and adapter writes."""

    def __init__(self, config: FSDPConfig) -> None:
        """Initialize the FSDP2 shard runtime.

        :param config: The FSDP configuration.
        """
        super().__init__()
        self.config = config
        self.parallel_mesh: ParallelMesh | None = None

    def data_parallel_world(self, process_world_size: int) -> int:
        folded = tp_data_parallel_size(process_world_size, self.config.tp)
        cp = self.config.cp
        if cp == 1:
            return folded
        if folded % cp != 0:
            msg = (
                f"world_size ({process_world_size}) must be divisible by "
                f"cp ({cp}) for context parallel."
            )
            raise ValueError(msg)
        return folded // cp

    def data_parallel_rank(self, process_rank: int) -> int:
        return process_rank // (self.config.tp * self.config.cp)

    @property
    def is_sharded(self) -> bool:
        return True

    @raise_on_any_rank()
    def prepare_actor(
        self,
        actor: nn.Module,
        device: str | torch.device,
        colocated: bool,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None,
        lr: float,
        lr_critic: float | None,
        restore_adapter_trainability: Callable[[list[str]], None],
        gradient_checkpointing: bool = False,
    ) -> PrepareResult:
        # heavy stack: llm_utils (optimizer/scheduler); export/import/gather do not load it
        from agilerl.utils.llm_utils import (  # lazy import: optimizer/scheduler unused by export/import/gather
            make_llm_optimizer,
            make_llm_scheduler,
        )

        if self.config.cpu_offload and self.config.optim_cpu_offload:
            msg = (
                "FSDPConfig.cpu_offload and optim_cpu_offload are mutually "
                "exclusive: cpu_offload already moves optimizer states to "
                "CPU. Set only one of them."
            )
            raise ValueError(msg)
        if self.config.cpu_offload and not colocated:
            msg = (
                "FSDP2 full cpu_offload requires colocated vLLM generation "
                "(vllm_config set). HuggingFace generate is not supported "
                "with CPUOffloadPolicy because it assumes model params live "
                "on the compute device. Set vllm_config or use "
                "optim_cpu_offload instead (params stay on GPU)."
            )
            raise ValueError(msg)

        restore_adapter_trainability(["actor", "critic"])
        if self.config.ep > 1:
            validate_actor_ep(actor, self.config.ep, get_world_size())
        if self.config.cp > 1:
            from agilerl.algorithms.core.llm_ops.ulysses_attn import (  # cycle: algorithms.core imports this module via base
                attention_head_counts,
                model_attention_backend,
            )

            heads, kv_heads = attention_head_counts(actor)
            validate_cp_heads(heads, kv_heads, self.config.cp)
            backend = model_attention_backend(actor)
            if backend != "flash_attention_2":
                msg = (
                    f"cp={self.config.cp} requires the actor's attention backend "
                    f"to be 'flash_attention_2', got {backend!r}: the Ulysses "
                    "substitution patches the flash-attention-2 forward."
                )
                raise ValueError(msg)
        self.parallel_mesh = build_parallel_mesh(
            ep=self.config.ep,
            tp=self.config.tp,
            cp=self.config.cp,
            shard_group_size=self.config.shard_group_size,
            device_type=torch.device(device).type,
        )
        wrapped = materialize_fsdp2_from_cpu_state(
            actor,
            device,
            self.config,
            parallel_mesh=self.parallel_mesh,
            gradient_checkpointing=gradient_checkpointing,
        )
        if self.parallel_mesh is not None:
            object.__setattr__(wrapped, "_agilerl_parallel_mesh", self.parallel_mesh)
        if self.config.cp > 1:
            from agilerl.algorithms.core.llm_ops.ulysses_attn import (  # cycle: algorithms.core imports this module via base
                substitute_hf_ulysses_attn,
            )

            if self.parallel_mesh is None or self.parallel_mesh.cp is None:
                msg = f"cp={self.config.cp} built no context-parallel mesh."
                raise RuntimeError(msg)
            substitute_hf_ulysses_attn(self.parallel_mesh.cp_group)

        optimizer = make_llm_optimizer(wrapped, lr, lr_critic)
        if self.config.optim_cpu_offload:
            inner = optimizer._single_optimizer()
            if isinstance(inner, CPUOffloadOptimizer):
                msg = "optimizer is already CPU-offloaded"
                raise TypeError(msg)
            optimizer.optimizer = CPUOffloadOptimizer(inner)
        return PrepareResult(
            actor=wrapped,
            optimizer=optimizer,
            lr_scheduler=make_llm_scheduler(optimizer, cosine_lr_schedule_config, lr),
        )

    @raise_on_any_rank()
    def export_model_state(self, model: nn.Module) -> dict[str, Any]:
        return get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    @raise_on_any_rank()
    def import_model_state(
        self, model: nn.Module, state: dict[str, Any], strict: bool = False
    ) -> None:
        packed = list(iter_packed_expert_modules(model))
        expert_keys = _ep_expert_live_keys(model, packed) if packed else frozenset()
        if expert_keys:
            _scatter_ep_expert_slices(model, state, expert_keys)
        set_full_model_state_dict(model, state, strict=strict, skip=expert_keys)

    @raise_on_any_rank()
    def export_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
    ) -> dict[str, Any]:
        if not isinstance(actor, FSDPModule):
            return _optimizer_state_as_dict(optimizer)
        inner_optimizer = optimizer._single_optimizer()
        if isinstance(inner_optimizer, CPUOffloadOptimizer):
            inner_optimizer = inner_optimizer.optimizer
        return get_optimizer_state_dict(
            actor,
            inner_optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    @raise_on_any_rank()
    def import_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        optimizer_state: dict[str, Any],
    ) -> None:
        if not isinstance(actor, FSDPModule):
            optimizer.load_state_dict(optimizer_state)
            return

        offload = optimizer._single_optimizer()
        inner_optimizer = (
            offload.optimizer if isinstance(offload, CPUOffloadOptimizer) else offload
        )
        param_fqns = {
            param: canonical_fsdp_param_fqn(name)
            for name, param in actor.named_parameters()
        }
        saved_state = optimizer_state.get("state", {})

        inner_optimizer.state.clear()
        with torch.no_grad():
            for group in inner_optimizer.param_groups:
                step_on_device = bool(group.get("fused") or group.get("capturable"))
                for param in group["params"]:
                    if not param.requires_grad:
                        continue
                    fqn = param_fqns.get(param)
                    if fqn is None:
                        msg = "Trainable optimizer parameter is not a named parameter on the actor"
                        raise RuntimeError(msg)
                    if fqn not in saved_state:
                        msg = f"Optimizer resume is missing state for trainable parameter {fqn!r}"
                        raise RuntimeError(msg)
                    inner_optimizer.state[param] = _place_param_optimizer_state(
                        param, saved_state[fqn], step_on_device=step_on_device
                    )

        if isinstance(offload, CPUOffloadOptimizer):
            offload.offload_states()

    @contextmanager
    def gather_layer(
        self,
        layer: nn.Linear,
        device: torch.device | str,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor | None], None, None]:
        target = torch.device(device)
        with materialize_dtensors(layer.weight, layer.bias) as (w, b):
            if w.device != target:
                w = w.to(target, non_blocking=True)
            if b is not None and b.device != target:
                b = b.to(target, non_blocking=True)
            yield w, b

    @raise_on_any_rank()
    def copy_adapter_tensors(
        self,
        actor: nn.Module,
        source_adapter: str,
        target_adapter: str,
    ) -> None:
        pairs = _lora_adapter_pairs(actor, source_adapter, target_adapter)
        with torch.no_grad():
            dense_sources = {
                name: source.full_tensor() if isinstance(source, DTensor) else source
                for name, (source, _) in pairs.items()
            }
        set_full_model_state_dict(actor, dense_sources, strict=False)

    @raise_on_any_rank()
    def import_adapter_tensors(
        self,
        actor: nn.Module,
        peft_model: nn.Module,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        # heavy stack: llm_utils (adapter load); unused by the rest of this runtime
        from agilerl.utils.llm_utils import (  # lazy import: adapter load unused by the rest of this runtime
            load_lora_adapters,
        )

        device = str(next(actor.parameters()).device)
        load_lora_adapters(actor, checkpoint_dir, adapter_name, device=device)

    def actor_compute_device(
        self, actor: nn.Module, fallback: torch.device
    ) -> torch.device:
        return fallback

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: OptimizerWrapper,
        gradient_accumulation_steps: int,
        actor: nn.Module,
        max_grad_norm: float | None = None,
        lr_scheduler: SequentialLR | None = None,
    ) -> OptimizerStep | None:
        """Accumulate, clip, and step.

        FSDP2 reduce-scatters its own DTensors. Replicated params
        (``ignored_params``, LoRA) and EP / TP shards outside FSDP need an
        explicit all-reduce. Under ``cp > 1`` those replicated grads sum
        across the context-parallel group, then divide by the data-parallel
        size (world / cp).
        """
        is_step_boundary = self._count_micro_batch(gradient_accumulation_steps)

        if self.config.defer_grad_sync:
            if not isinstance(actor, FSDPModule):
                msg = "defer_grad_sync requires an FSDP2-sharded actor"
                raise TypeError(msg)
            actor.set_requires_gradient_sync(is_step_boundary, recurse=True)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        self.phase_timer.mark("backward")
        if not is_step_boundary:
            return None

        inner = optimizer._single_optimizer()
        params = [param for group in inner.param_groups for param in group["params"]]
        replicated = [param for param in params if not isinstance(param, DTensor)]
        if self.config.cp > 1:
            sync_grads(
                replicated,
                divide_factor=self.data_parallel_world(get_world_size()),
            )
        else:
            sync_grads(replicated)
        if self.parallel_mesh is not None:
            self.parallel_mesh.sync_grads(
                params, getattr(torch, self.config.reduce_dtype)
            )
        self.phase_timer.mark("grad_sync")
        grad_norm_pre, grad_norm_post = clip_param_groups(
            inner.param_groups, max_grad_norm, clip_param_group_grad_norm_
        )
        optimizer.step()
        optimizer.zero_grad()
        self.phase_timer.mark("optim")
        return _step_result(grad_norm_pre, grad_norm_post, lr_scheduler)
