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
from torch.optim.lr_scheduler import LambdaLR

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
    lr_scheduler: LambdaLR | None


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


def _host_copy(value: object) -> object:
    """Copy every tensor in a nested dict / list / tuple to a new CPU tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _host_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_host_copy(item) for item in value)
    return value


def _copy_shared_storage(value: object, live_storage: set[int]) -> object:
    """Clone every tensor in a nested dict / list / tuple whose storage is in ``live_storage``."""
    if isinstance(value, torch.Tensor):
        if value.untyped_storage().data_ptr() in live_storage:
            return value.detach().clone()
        return value
    if isinstance(value, dict):
        return {
            key: _copy_shared_storage(item, live_storage) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_shared_storage(item, live_storage) for item in value)
    return value


def _detach_from_optimizer(
    state: dict[str, Any], optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    """Clone exported tensors that alias ``optimizer``'s live state (e.g. CPU ``step``)."""
    live_storage: set[int] = set()
    for param_state in optimizer.state.values():
        for value in param_state.values():
            local = value.to_local() if isinstance(value, DTensor) else value
            if isinstance(local, torch.Tensor):
                live_storage.add(local.untyped_storage().data_ptr())
    return {
        key: _copy_shared_storage(item, live_storage) for key, item in state.items()
    }


def _optimizer_state_as_dict(optimizer: OptimizerWrapper) -> dict[str, Any]:
    """Host copy of ``optimizer.state_dict()``; refuse list-shaped multi-opt state."""
    state = optimizer.state_dict()
    if not isinstance(state, dict):
        msg = "optimizer.state_dict() must return a dict"
        raise TypeError(msg)
    return {key: _host_copy(value) for key, value in state.items()}


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
    """Gradient norms from one optimizer step.

    A group's post-clip norm is its pre-clip norm times its clip coefficient.
    """

    # Pre-clip L2 norm of each optimizer param group, in group order.
    group_grad_norms: tuple[float, ...]
    # Coefficient each param group's grads were scaled by, in group order.
    clip_coefs: tuple[float, ...]

    @property
    def grad_norm_pre(self) -> float:
        """Pre-clip L2 norm over every param group."""
        return sum(norm * norm for norm in self.group_grad_norms) ** 0.5

    @property
    def grad_norm_post(self) -> float:
        """Post-clip L2 norm over every param group."""
        return (
            sum(
                (norm * coef) ** 2
                for norm, coef in zip(
                    self.group_grad_norms, self.clip_coefs, strict=True
                )
            )
            ** 0.5
        )


def clip_param_groups(
    param_groups: list[dict[str, Any]],
    max_grad_norm: float | None,
    clip_fn: Callable[[list[nn.Parameter], float], torch.Tensor],
    clip_groups: tuple[frozenset[str], ...] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Clip each set of groups by one coefficient from the L2 norm over that set.

    With ``clip_groups=None`` every group forms one set, matching
    ``torch.nn.utils.clip_grad_norm_`` over the union of all grads. Groups
    may sit on different DTensor meshes, so each is measured on its own.

    :param param_groups: Optimizer param groups.
    :type param_groups: list[dict[str, Any]]
    :param max_grad_norm: Clip threshold for each set's norm, or ``None`` to
        only measure.
    :type max_grad_norm: float | None
    :param clip_fn: Clips one group in place and returns its pre-clip norm;
        called with ``max_norm=inf`` so it only measures.
    :type clip_fn: Callable[[list[nn.Parameter], float], torch.Tensor]
    :param clip_groups: Sets of ``group["group"]`` names clipped by their own
        norm. Every param group must fall in exactly one set.
    :type clip_groups: tuple[frozenset[str], ...] | None
    :return: Pre-clip L2 norm of each group, and the coefficient each group's
        grads were scaled by.
    :rtype: tuple[tuple[float, ...], tuple[float, ...]]
    """
    group_norms = tuple(
        float(_scalar_grad_norm(clip_fn(group["params"], float("inf"))))
        for group in param_groups
    )
    if max_grad_norm is None:
        return group_norms, (1.0,) * len(param_groups)
    if clip_groups is None:
        clip_sets = [list(range(len(param_groups)))]
    else:
        clip_sets = [
            [i for i, group in enumerate(param_groups) if group["group"] in names]
            for names in clip_groups
        ]
        if sorted(i for indices in clip_sets for i in indices) != list(
            range(len(param_groups))
        ):
            names = [group["group"] for group in param_groups]
            msg = f"clip_groups {clip_groups} must hold each of {names} exactly once"
            raise ValueError(msg)
    clip_coefs = [1.0] * len(param_groups)
    for indices in clip_sets:
        total = sum(group_norms[i] ** 2 for i in indices) ** 0.5
        # min(nan, 1.0) is nan, so a non-finite total reaches the grads as in torch.
        clip_coef = min(max_grad_norm / (total + 1e-6), 1.0)
        for i in indices:
            clip_coefs[i] = clip_coef
            for param in param_groups[i]["params"]:
                if param.grad is not None:
                    param.grad.mul_(clip_coef)
    return group_norms, tuple(clip_coefs)


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
        """Return a full optimizer state dict on CPU (FSDP2: rank 0 only).

        No tensor in the result shares storage with the live optimizer state,
        so later optimizer steps do not change it.

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
        clip_groups: tuple[frozenset[str], ...] | None = None,
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
        :param clip_groups: Param group name sets clipped by their own norm;
            ``None`` clips every group by the global norm.
        :type clip_groups: tuple[frozenset[str], ...] | None
        :return: Norms on a step boundary, else ``None``.
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
            lr_scheduler=make_llm_scheduler(optimizer, cosine_lr_schedule_config),
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
        clip_groups: tuple[frozenset[str], ...] | None = None,
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
        group_grad_norms, clip_coefs = clip_param_groups(
            inner.param_groups, max_grad_norm, clip_grad_norm_, clip_groups
        )
        optimizer.step()
        optimizer.zero_grad()
        self.phase_timer.mark("optim")
        return OptimizerStep(group_grad_norms, clip_coefs)


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
        return tp_data_parallel_size(process_world_size, self.config.tp)

    def data_parallel_rank(self, process_rank: int) -> int:
        return process_rank // self.config.tp

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

        if self.config.optim_cpu_offload is None:
            msg = (
                "FSDPConfig.optim_cpu_offload is unset. LLMAlgorithm.wrap_models "
                "resolves it from the memory estimate; pass True or False here."
            )
            raise ValueError(msg)
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
        self.parallel_mesh = build_parallel_mesh(
            ep=self.config.ep,
            tp=self.config.tp,
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

        # Fused AdamW steps CPU-offloaded state slower than the default kernel.
        on_gpu = torch.device(device).type == "cuda" and not (
            self.config.cpu_offload or self.config.optim_cpu_offload
        )
        optimizer = make_llm_optimizer(wrapped, lr, lr_critic, fused=on_gpu)
        if self.config.optim_cpu_offload:
            inner = optimizer._single_optimizer()
            if isinstance(inner, CPUOffloadOptimizer):
                msg = "optimizer is already CPU-offloaded"
                raise TypeError(msg)
            optimizer.optimizer = CPUOffloadOptimizer(inner)
        return PrepareResult(
            actor=wrapped,
            optimizer=optimizer,
            lr_scheduler=make_llm_scheduler(optimizer, cosine_lr_schedule_config),
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
        state = get_optimizer_state_dict(
            actor,
            inner_optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        return _detach_from_optimizer(state, inner_optimizer)

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
        saved_params = {
            fqn
            for group in optimizer_state.get("param_groups", [])
            for fqn in group["params"]
        }

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
                        # A saved param with no state never stepped; AdamW
                        # creates its state on the first step.
                        if fqn in saved_params:
                            continue
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
        load_lora_adapters(peft_model, checkpoint_dir, adapter_name, device=device)

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
        clip_groups: tuple[frozenset[str], ...] | None = None,
    ) -> OptimizerStep | None:
        """Accumulate, clip, and step.

        FSDP2 reduce-scatters its own DTensors. Replicated params
        (``ignored_params``, LoRA) and EP / TP shards outside FSDP need an
        explicit all-reduce.
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
        sync_grads([param for param in params if not isinstance(param, DTensor)])
        if self.parallel_mesh is not None:
            self.parallel_mesh.sync_grads(
                params, getattr(torch, self.config.reduce_dtype)
            )
        self.phase_timer.mark("grad_sync")
        group_grad_norms, clip_coefs = clip_param_groups(
            inner.param_groups,
            max_grad_norm,
            clip_param_group_grad_norm_,
            clip_groups,
        )
        optimizer.step()
        optimizer.zero_grad()
        self.phase_timer.mark("optim")
        return OptimizerStep(group_grad_norms, clip_coefs)
