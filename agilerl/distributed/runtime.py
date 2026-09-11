# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Actor shard runtime: prepare, export/import, and adapter tensor ops for dense vs FSDP2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
)
from torch.distributed.tensor import distribute_tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from agilerl.distributed.fsdp_profile import timed_fsdp
from agilerl.distributed.process import (
    CPUOffloadOptimizer,
    FSDPConfig,
    gather_params,
    get_full_model_state_dict,
    is_fsdp_sharded,
    materialize_dtensors,
    materialize_fsdp2_from_cpu_state,
    raise_on_any_rank,
    set_full_model_state_dict,
    sync_grads,
)

if TYPE_CHECKING:
    from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
    from agilerl.utils.algo_utils import CosineLRScheduleConfig


@dataclass
class PrepareResult:
    """Actor, optimizer, and scheduler after :meth:`BaseRuntime.prepare_actor`."""

    actor: nn.Module
    optimizer: OptimizerWrapper
    lr_scheduler: object | None


def _rebuild_llm_optimizer(
    actor: nn.Module,
    lr: float,
    lr_critic: float | None,
) -> OptimizerWrapper:
    from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper

    return OptimizerWrapper(
        optimizer_cls=AdamW,
        networks=[actor],
        network_names=["actor"],
        lr=lr,
        lr_critic=lr_critic,
        is_llm_optimizer=True,
        lr_name="lr" if lr_critic is None else ("lr_actor", "lr_critic"),
    )


def _lora_adapter_param_maps(
    actor: nn.Module,
    source_adapter: str,
    target_adapter: str,
) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter], dict[str, str]]:
    source_params: dict[str, nn.Parameter] = {}
    target_params: dict[str, nn.Parameter] = {}
    target_full_names: dict[str, str] = {}
    for name, param in actor.named_parameters():
        if "lora" not in name:
            continue
        if f".{source_adapter}." in name:
            key = name.replace(f".{source_adapter}.", ".", 1)
            source_params[key] = param
        elif f".{target_adapter}." in name:
            key = name.replace(f".{target_adapter}.", ".", 1)
            target_params[key] = param
            target_full_names[key] = name
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
    return source_params, target_params, target_full_names


class BaseRuntime(ABC):
    """Behaviors that differ between unsharded and FSDP2-sharded actors."""

    _micro_batch_count: int = 0

    @property
    @abstractmethod
    def is_sharded(self) -> bool:
        """Whether this runtime shards actor parameters."""

    @abstractmethod
    def prepare_actor(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        lr_scheduler: object | None,
        *,
        device: str | torch.device,
        use_vllm: bool,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None,
        lr: float,
        lr_critic: float | None,
        restore_adapter_trainability: Callable[[list[str]], None],
        gradient_checkpointing: bool = False,
    ) -> PrepareResult:
        """Prepare the actor for training and return remapped opt/scheduler."""

    @abstractmethod
    def export_model_state(
        self, model: nn.Module, *, cpu_offload: bool = True
    ) -> dict[str, Any]:
        """Return a full model state dict (gathered under FSDP2)."""

    @abstractmethod
    def import_model_state(
        self, model: nn.Module, state: dict[str, Any], *, strict: bool = False
    ) -> None:
        """Load a full model state dict onto ``model``."""

    @abstractmethod
    def export_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        *,
        cpu_offload: bool = True,
    ) -> dict[str, Any]:
        """Return a full optimizer state dict (gathered under FSDP2)."""

    @abstractmethod
    def import_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        optimizer_state: dict[str, Any],
    ) -> None:
        """Load a full optimizer state dict onto ``optimizer``."""

    @abstractmethod
    def gather_layer(
        self,
        layer: nn.Linear,
        *,
        device: torch.device | str,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor | None], None, None]:
        """Yield full local ``(weight, bias)`` for ``layer`` on ``device``."""

    @abstractmethod
    def copy_adapter_tensors(
        self,
        actor: nn.Module,
        source_adapter: str,
        target_adapter: str,
    ) -> None:
        """Copy LoRA tensors from ``source_adapter`` to ``target_adapter``."""

    @abstractmethod
    def import_adapter_tensors(
        self,
        actor: nn.Module,
        peft_model: nn.Module,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        """Write LoRA tensors from ``checkpoint_dir`` onto ``actor``."""

    @abstractmethod
    def actor_compute_device(
        self, actor: nn.Module, fallback: torch.device
    ) -> torch.device:
        """Device for HF generate inputs."""

    @contextmanager
    def timed(self, stage: str, **fields: Any) -> Generator[None, None, None]:
        """No-op timing slot; :class:`FSDPRuntime` logs GPU-synced INFO lines."""
        yield

    @abstractmethod
    def backward(
        self,
        loss: torch.Tensor,
        optimizer: OptimizerWrapper,
        micro_batch_size: int,
        mini_batch_size: int,
        gradient_accumulation_steps: int | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        """Implemented by :class:`DPRuntime` and :class:`FSDPRuntime`."""


class DPRuntime(BaseRuntime):
    """Single-device and data-parallel (unsharded) runtime."""

    @property
    def is_sharded(self) -> bool:
        return False

    def prepare_actor(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        lr_scheduler: object | None,
        *,
        device: str | torch.device,
        use_vllm: bool,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None,
        lr: float,
        lr_critic: float | None,
        restore_adapter_trainability: Callable[[list[str]], None],
        gradient_checkpointing: bool = False,
    ) -> PrepareResult:
        target = torch.device(device)
        param = next(actor.parameters(), None)
        if (
            param is None
            or param.device.type != target.type
            or (target.index is not None and param.device.index != target.index)
        ):
            actor = actor.to(target)
        return PrepareResult(
            actor=actor, optimizer=optimizer, lr_scheduler=lr_scheduler
        )

    def export_model_state(
        self, model: nn.Module, *, cpu_offload: bool = True
    ) -> dict[str, Any]:
        return model.state_dict()

    def import_model_state(
        self, model: nn.Module, state: dict[str, Any], *, strict: bool = False
    ) -> None:
        model.load_state_dict(state, strict=strict)

    def export_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        *,
        cpu_offload: bool = True,
    ) -> dict[str, Any]:
        return optimizer.state_dict()

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
        *,
        device: torch.device | str,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor | None], None, None]:
        yield layer.weight, layer.bias

    def copy_adapter_tensors(
        self,
        actor: nn.Module,
        source_adapter: str,
        target_adapter: str,
    ) -> None:
        source_params, target_params, _ = _lora_adapter_param_maps(
            actor, source_adapter, target_adapter
        )
        for key, src_param in source_params.items():
            target_params[key].data.copy_(src_param.data)

    def import_adapter_tensors(
        self,
        actor: nn.Module,
        peft_model: nn.Module,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

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
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None = None,
        *,
        lr_scheduler: object | None = None,
        actor: nn.Module,
    ) -> tuple[float | None, float | None]:
        """Average LoRA grads across ranks when unsharded."""
        params = [
            param
            for group in optimizer.optimizer.param_groups
            for param in group["params"]
        ]
        self._micro_batch_count += 1
        is_step_boundary = self._micro_batch_count % gradient_accumulation_steps == 0

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        new_lr = None
        grad_norm = None

        if not is_step_boundary:
            return new_lr, grad_norm

        sync_grads(params)
        if max_grad_norm is not None:
            for group in optimizer.optimizer.param_groups:
                grad_norm = clip_grad_norm_(
                    group["params"], max_norm=max_grad_norm
                ).item()
        optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()
            new_lr = lr_scheduler.get_last_lr()[0]

        return new_lr, grad_norm


class FSDPRuntime(BaseRuntime):
    """FSDP2-sharded runtime: wrap, gathers, and adapter writes."""

    def __init__(self, config: FSDPConfig) -> None:
        """Initialize the FSDP2 shard runtime.

        :param config: The FSDP configuration.
        """
        self.config = config

    @property
    def is_sharded(self) -> bool:
        return True

    @raise_on_any_rank()
    def prepare_actor(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        lr_scheduler: object | None,
        *,
        device: str | torch.device,
        use_vllm: bool,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None,
        lr: float,
        lr_critic: float | None,
        restore_adapter_trainability: Callable[[list[str]], None],
        gradient_checkpointing: bool = False,
    ) -> PrepareResult:
        if self.config.cpu_offload and self.config.optim_cpu_offload:
            msg = (
                "FSDPConfig.cpu_offload and optim_cpu_offload are mutually "
                "exclusive: cpu_offload already moves optimizer states to "
                "CPU. Set only one of them."
            )
            raise ValueError(msg)
        if self.config.cpu_offload and not use_vllm:
            msg = (
                "FSDP2 full cpu_offload requires vLLM for generation "
                "(use_vllm=True). HuggingFace generate is not supported "
                "with CPUOffloadPolicy because it assumes model params live "
                "on the compute device. Either enable vLLM or use "
                "optim_cpu_offload instead (params stay on GPU)."
            )
            raise ValueError(msg)

        restore_adapter_trainability(["actor", "critic"])
        wrapped = materialize_fsdp2_from_cpu_state(
            actor,
            device,
            self.config,
            gradient_checkpointing=gradient_checkpointing,
        )
        new_optimizer = _rebuild_llm_optimizer(wrapped, lr, lr_critic)
        if self.config.optim_cpu_offload:
            inner = new_optimizer._single_optimizer()
            new_optimizer.optimizer = CPUOffloadOptimizer(inner)
        if cosine_lr_schedule_config is None:
            new_scheduler = None
        else:
            from agilerl.utils.algo_utils import create_warmup_cosine_scheduler

            new_scheduler = create_warmup_cosine_scheduler(
                new_optimizer._single_optimizer(),
                cosine_lr_schedule_config,
                1e-8,
                lr,
            )
        return PrepareResult(
            actor=wrapped, optimizer=new_optimizer, lr_scheduler=new_scheduler
        )

    @raise_on_any_rank()
    def export_model_state(
        self, model: nn.Module, *, cpu_offload: bool = True
    ) -> dict[str, Any]:
        return get_full_model_state_dict(model, cpu_offload=cpu_offload)

    @raise_on_any_rank()
    def import_model_state(
        self, model: nn.Module, state: dict[str, Any], *, strict: bool = False
    ) -> None:
        set_full_model_state_dict(model, state, strict=strict)

    @raise_on_any_rank()
    def export_optimizer_state(
        self,
        actor: nn.Module,
        optimizer: OptimizerWrapper,
        *,
        cpu_offload: bool = True,
    ) -> dict[str, Any]:
        if not is_fsdp_sharded(actor):
            return optimizer.state_dict()
        if not cpu_offload:
            msg = (
                "export_optimizer_state(cpu_offload=False) is not "
                "supported for FSDP2 models: consolidating optimizer state on "
                "GPU on every rank undermines sharded training. Use "
                "cpu_offload=True."
            )
            raise ValueError(msg)
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
        if not is_fsdp_sharded(actor):
            optimizer.load_state_dict(optimizer_state)
            return

        inner_optimizer = optimizer._single_optimizer()
        if isinstance(inner_optimizer, CPUOffloadOptimizer):
            inner_optimizer = inner_optimizer.optimizer
        fqn_to_param = dict(actor.named_parameters())
        saved_state = optimizer_state.get("state", {})

        inner_optimizer.state.clear()

        with torch.no_grad():
            for group in inner_optimizer.param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue
                    fqn = None
                    for name, mapped in fqn_to_param.items():
                        if mapped is param:
                            fqn = name
                            break
                    if fqn is None or fqn not in saved_state:
                        continue
                    saved_entry = saved_state[fqn]
                    inner_optimizer.state[param] = {}
                    for sk in ("step", "exp_avg", "exp_avg_sq"):
                        if sk not in saved_entry:
                            continue
                        saved_val = saved_entry[sk]
                        if hasattr(param, "device_mesh") and saved_val.dim() > 0:
                            target_device = param.device_mesh.device_type
                            saved_val = saved_val.to(target_device)
                            sharded = distribute_tensor(
                                saved_val, param.device_mesh, param.placements
                            )
                            inner_optimizer.state[param][sk] = sharded
                        else:
                            inner_optimizer.state[param][sk] = saved_val.to(
                                param.device
                            )

        offload = optimizer._single_optimizer()
        if isinstance(offload, CPUOffloadOptimizer):
            offload._move_states("cpu")
            offload._initialized = True

    @contextmanager
    def gather_layer(
        self,
        layer: nn.Linear,
        *,
        device: torch.device | str,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor | None], None, None]:
        target = torch.device(device)
        with materialize_dtensors(layer.weight, layer.bias) as gathered_tensors:
            w, b = gathered_tensors
            if w is not None and w.device != target:
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
        source_params, _, target_full_names = _lora_adapter_param_maps(
            actor, source_adapter, target_adapter
        )
        dense_sources: dict[str, torch.Tensor] = {}
        source_keys = list(source_params.keys())
        with gather_params(list(source_params.values())) as gathered_tensors:
            for key, dense in zip(source_keys, gathered_tensors, strict=True):
                if dense is None:
                    continue
                dense_sources[key] = dense.detach().clone()
        set_full_model_state_dict(
            actor,
            {target_full_names[key]: tensor for key, tensor in dense_sources.items()},
            strict=False,
        )

    @raise_on_any_rank()
    def import_adapter_tensors(
        self,
        actor: nn.Module,
        peft_model: nn.Module,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        from agilerl.utils.llm_utils import load_lora_adapters

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
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None = None,
        *,
        lr_scheduler: object | None = None,
        actor: nn.Module,
    ) -> tuple[float | None, float | None]:
        """Accumulate, clip, and step. FSDP2 owns the reduce-scatter."""
        self._micro_batch_count += 1
        is_step_boundary = self._micro_batch_count % gradient_accumulation_steps == 0

        if self.config.defer_grad_sync:
            actor.set_requires_gradient_sync(is_step_boundary, recurse=True)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        with timed_fsdp(
            "microbatch_backward",
            micro_batch=self._micro_batch_count,
            step_boundary=is_step_boundary,
            grad_sync=is_step_boundary if self.config.defer_grad_sync else True,
        ):
            loss.backward()

        new_lr = None
        grad_norm = None
        if not is_step_boundary:
            return new_lr, grad_norm

        if max_grad_norm is not None:
            with timed_fsdp("clip_grad", micro_batch=self._micro_batch_count):
                for group in optimizer.optimizer.param_groups:
                    grad_norm = clip_grad_norm_(
                        group["params"], max_norm=max_grad_norm
                    ).item()
        with timed_fsdp("optimizer_step", micro_batch=self._micro_batch_count):
            optimizer.step()
        with timed_fsdp("zero_grad", micro_batch=self._micro_batch_count):
            optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()
            new_lr = lr_scheduler.get_last_lr()[0]

        return new_lr, grad_norm

    @contextmanager
    def timed(self, stage: str, **fields: Any) -> Generator[None, None, None]:
        with timed_fsdp(stage, micro_batch=self._micro_batch_count, **fields):
            yield


def make_shard_runtime(config: FSDPConfig | None) -> BaseRuntime:
    """Build the shard runtime for ``config`` (dense when ``config`` is ``None``)."""
    if config is None:
        return DPRuntime()
    return FSDPRuntime(config)
