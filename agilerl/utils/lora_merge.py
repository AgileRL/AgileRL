# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""ZeRO-aware layer-wise LoRA merge into a Hugging Face weight directory."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from accelerate import Accelerator, init_empty_weights
from huggingface_hub.serialization import split_torch_state_dict_into_shards
from peft import PeftConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Conv1d, Conv2d, Conv3d, Embedding, Linear, LoraLayer
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

from agilerl.utils.llm_utils import gather_if_zero3, zero3_full_shape_views
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

DEFAULT_MAX_SHARD_SIZE = "5GB"
DEFAULT_TORCH_DTYPE = torch.bfloat16


class MergedExportError(RuntimeError):
    """Layer-wise merge or HF write failed; every rank has left collectives."""


@dataclass(frozen=True)
class ModuleCopy:
    """One unmodified tensor to write under a Hugging Face parameter name."""

    hf_name: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class ModuleLoraMerged:
    """One LoRA-wrapped module whose base weight or bias is merged."""

    hf_name: str
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d
    adapter_name: str
    part: Literal["weight", "bias"]
    reference: torch.Tensor


ExportItem = ModuleCopy | ModuleLoraMerged


class BaseWeightStore:
    """Lazy CPU tensors from a safetensors Hugging Face checkpoint."""

    def __init__(self, key_files: dict[str, str]) -> None:
        self.key_files = key_files

    @classmethod
    def from_checkpoint(cls, model_path: str | Path) -> BaseWeightStore:
        """Index ``model.safetensors`` or a sharded weight map under ``model_path``."""
        source = str(model_path)
        index_file = cached_file(
            source,
            SAFE_WEIGHTS_INDEX_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        if index_file is not None:
            weight_map = json.loads(Path(index_file).read_text(encoding="utf-8"))[
                "weight_map"
            ]
            key_files: dict[str, str] = {}
            shard_paths: dict[str, str] = {}
            for key, filename in weight_map.items():
                if filename not in shard_paths:
                    shard = cached_file(
                        source,
                        filename,
                        _raise_exceptions_for_missing_entries=False,
                    )
                    if shard is None:
                        msg = f"Replay export missing shard {filename} under {source}"
                        raise FileNotFoundError(msg)
                    shard_paths[filename] = shard
                key_files[key] = shard_paths[filename]
            return cls(key_files)

        weights_file = cached_file(
            source,
            SAFE_WEIGHTS_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        if weights_file is None:
            msg = f"Replay export needs safetensors weights under {source}"
            raise FileNotFoundError(msg)
        with safe_open(weights_file, framework="pt") as handle:
            return cls(dict.fromkeys(handle.keys(), weights_file))

    def has(self, key: str) -> bool:
        """Return whether ``key`` is in the checkpoint."""
        return key in self.key_files

    def load(self, key: str) -> torch.Tensor:
        """Load one tensor onto CPU. Does not cache."""
        path = self.key_files.get(key)
        if path is None:
            msg = f"Checkpoint has no tensor {key!r}"
            raise KeyError(msg)
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def export_merged_pretrained(
    output_dir: str | Path,
    *,
    model: nn.Module | None = None,
    adapter_path: str | Path | None = None,
    base_model_name_or_path: str | Path | None = None,
    adapter_name: str = "actor",
    zero_stage: int | None = None,
    accelerator: Accelerator | None = None,
    max_shard_size: str | int = DEFAULT_MAX_SHARD_SIZE,
    torch_dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> None:
    """Write adapter-free Hugging Face weights without assembling a full state dict.

    Live-model shape: pass ``model`` (a PEFT-wrapped causal LM, or a value-head
    wrapper whose ``pretrained_model`` is PEFT). Replay shape: pass ``adapter_path``
    (the ``actor/`` PEFT directory) and ``base_model_name_or_path``.

    :param output_dir: Directory for ``config.json``, safetensors, and optional
        tokenizer / ``generation_config.json``.
    :param model: Live PEFT module currently on the training ranks.
    :param adapter_path: Saved PEFT adapter directory for replay.
    :param base_model_name_or_path: Base model id or local HF directory for replay.
    :param adapter_name: PEFT adapter to fold in, defaults to ``actor``.
    :param zero_stage: DeepSpeed ZeRO stage; ``3`` gathers one module at a time.
    :param accelerator: Optional Accelerate handle so every rank enters gathers
        and a failed write cannot leave ranks stuck in a collective.
    :param max_shard_size: Hugging Face shard budget (``"5GB"`` default).
    :param torch_dtype: Dtype of written weights, defaults to ``bfloat16``.
    :param tokenizer: Optional tokenizer written beside the weights.
    """
    error: BaseException | None = None
    try:
        if model is not None:
            if adapter_path is not None or base_model_name_or_path is not None:
                msg = (
                    "export_merged_pretrained expects either model=... (live) or "
                    "adapter_path=... and base_model_name_or_path=... (replay)"
                )
                raise ValueError(msg)
            _export_live_model(
                model,
                output_dir=output_dir,
                adapter_name=adapter_name,
                zero_stage=zero_stage,
                accelerator=accelerator,
                max_shard_size=max_shard_size,
                torch_dtype=torch_dtype,
                tokenizer=tokenizer,
                weight_store=None,
            )
        elif adapter_path is not None:
            if base_model_name_or_path is None:
                msg = "Replay export requires base_model_name_or_path with adapter_path"
                raise ValueError(msg)
            if accelerator is None or accelerator.is_main_process:
                replay_model, weight_store = _load_replay_peft(
                    adapter_path=adapter_path,
                    base_model_name_or_path=base_model_name_or_path,
                    adapter_name=adapter_name,
                )
                _export_live_model(
                    replay_model,
                    output_dir=output_dir,
                    adapter_name=adapter_name,
                    zero_stage=None,
                    accelerator=None,
                    max_shard_size=max_shard_size,
                    torch_dtype=torch_dtype,
                    tokenizer=tokenizer,
                    weight_store=weight_store,
                )
        elif base_model_name_or_path is not None:
            msg = "Replay export requires adapter_path with base_model_name_or_path"
            raise ValueError(msg)
        else:
            msg = (
                "export_merged_pretrained expects either model=... (live) or "
                "adapter_path=... and base_model_name_or_path=... (replay)"
            )
            raise ValueError(msg)
    except Exception as exc:
        error = exc
    _reraise_if_any_rank_failed(accelerator, error)


def _any_rank_failed(
    accelerator: Accelerator | None,
    error: BaseException | None,
) -> bool:
    """Return whether this rank or any other rank has already failed the export."""
    if accelerator is None or accelerator.num_processes <= 1:
        return error is not None

    accelerator.wait_for_everyone()
    flag = torch.zeros(1, dtype=torch.int64, device=accelerator.device)
    if error is not None:
        flag[0] = 1
    flag = accelerator.reduce(flag, reduction="sum")
    return int(flag.item()) != 0


def _reraise_if_any_rank_failed(
    accelerator: Accelerator | None,
    error: BaseException | None,
) -> None:
    """Raise on every rank when any rank failed, after collectives have exited."""
    if accelerator is None or accelerator.num_processes <= 1:
        if error is None:
            return
        if isinstance(error, (MergedExportError, TypeError, ValueError)):
            raise error
        raise MergedExportError(str(error)) from error

    if not _any_rank_failed(accelerator, error):
        return
    if isinstance(error, MergedExportError):
        raise error
    msg = "Merged HF export failed on at least one rank"
    raise MergedExportError(msg) from error


def _load_replay_peft(
    adapter_path: str | Path,
    base_model_name_or_path: str | Path,
    adapter_name: str,
) -> tuple[nn.Module, BaseWeightStore]:
    """Build a meta PEFT skeleton and a store for on-demand base tensors."""
    config = AutoConfig.from_pretrained(str(base_model_name_or_path))
    adapter_config = PeftConfig.from_pretrained(str(adapter_path))
    with init_empty_weights(include_buffers=True):
        base = AutoModelForCausalLM.from_config(config)
        peft_model = get_peft_model(base, adapter_config, adapter_name=adapter_name)
    _assign_adapter_weights(peft_model, adapter_path, adapter_name)
    return peft_model, BaseWeightStore.from_checkpoint(base_model_name_or_path)


def _assign_adapter_weights(
    model: nn.Module,
    adapter_path: str | Path,
    adapter_name: str,
) -> None:
    """Move saved LoRA tensors onto the meta skeleton as CPU parameters."""
    weights_file = cached_file(
        str(adapter_path),
        "adapter_model.safetensors",
        _raise_exceptions_for_missing_entries=False,
    )
    if weights_file is None:
        msg = f"Replay export needs adapter_model.safetensors under {adapter_path}"
        raise FileNotFoundError(msg)
    tensors = load_file(weights_file, device="cpu")
    used: set[str] = set()
    for name, _ in model.named_parameters():
        key = name.replace(f".{adapter_name}.", ".", 1)
        tensor = tensors.get(key)
        if tensor is None:
            continue
        _replace_named_tensor(model, name, tensor)
        used.add(key)
    leftover = set(tensors) - used
    if leftover:
        preview = ", ".join(sorted(leftover)[:4])
        msg = f"Adapter tensors were not on the PEFT model: {preview}"
        raise ValueError(msg)
    if not used:
        msg = f"No adapter tensors matched parameters under {adapter_path}"
        raise ValueError(msg)


def _replace_named_tensor(root: nn.Module, name: str, tensor: torch.Tensor) -> object:
    """Replace a nested parameter or buffer with a CPU ``Parameter``; return the old value."""
    parts = name.split(".")
    module = root.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else root
    return _replace_tensor(module, parts[-1], tensor)


def _replace_tensor(module: nn.Module, attr: str, tensor: torch.Tensor) -> object:
    """Replace ``module.attr`` with a CPU parameter; return the previous value."""
    try:
        previous = module.get_parameter(attr)
    except AttributeError:
        previous = module.get_buffer(attr)
    setattr(
        module,
        attr,
        nn.Parameter(
            tensor.detach().contiguous().cpu(),
            requires_grad=previous.requires_grad,
        ),
    )
    return previous


def _as_peft_model(model: nn.Module) -> PeftModel:
    """Unwrap a value-head wrapper to the PEFT causal LM."""
    if isinstance(model, AutoModelForCausalLMWithValueHead):
        model = model.pretrained_model
    if not isinstance(model, PeftModel):
        msg = "export_merged_pretrained requires a PEFT-wrapped causal LM"
        raise TypeError(msg)
    return model


def _peft_lora_module(
    module: nn.Module,
) -> Linear | Embedding | Conv1d | Conv2d | Conv3d:
    """Return a PEFT LoRA module that can produce a merge delta."""
    if isinstance(module, (Linear, Embedding, Conv1d, Conv2d, Conv3d)):
        return module
    msg = f"{type(module).__name__} cannot produce a LoRA delta"
    raise TypeError(msg)


def _as_lora_base(
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d,
) -> nn.Linear | nn.Embedding | Conv1D | nn.Conv1d | nn.Conv2d | nn.Conv3d:
    """Return the Linear, Embedding, or Conv module LoRA wraps."""
    base = layer.get_base_layer()
    if isinstance(base, nn.Embedding):
        return base
    if isinstance(base, (nn.Linear, Conv1D, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return base
    msg = f"{type(layer).__name__} wraps {type(base).__name__}, which is not a LoRA base module"
    raise TypeError(msg)


def _base_bias(
    base: nn.Linear | nn.Embedding | Conv1D | nn.Conv1d | nn.Conv2d | nn.Conv3d,
) -> torch.Tensor | None:
    """Return the base module bias, if it has one."""
    if isinstance(base, nn.Embedding):
        return None
    return base.bias


def _export_live_model(
    model: nn.Module,
    *,
    output_dir: str | Path,
    adapter_name: str,
    zero_stage: int | None,
    accelerator: Accelerator | None,
    max_shard_size: str | int,
    torch_dtype: torch.dtype,
    tokenizer: PreTrainedTokenizerBase | None,
    weight_store: BaseWeightStore | None = None,
) -> None:
    """Merge ``adapter_name`` layer-wise from a live PEFT module into ``output_dir``."""
    peft_model = _as_peft_model(model)
    if adapter_name not in peft_model.peft_config:
        msg = f"Adapter {adapter_name!r} is not on the PEFT model"
        raise ValueError(msg)
    pretrained = peft_model.get_base_model()
    if not isinstance(pretrained, PreTrainedModel):
        msg = (
            "export_merged_pretrained requires a transformers model, "
            f"got {type(pretrained).__name__}"
        )
        raise TypeError(msg)
    specs = _collect_export_specs(pretrained, adapter_name, weight_store)
    shard_plan = _plan_shards(specs, max_shard_size)
    output_path = Path(output_dir)
    error: BaseException | None = None
    is_main = accelerator is None or accelerator.is_main_process
    if is_main:
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = exc

    for filename, hf_names in shard_plan:
        if _any_rank_failed(accelerator, error):
            break
        shard_tensors = _materialize_shard(
            specs,
            hf_names,
            zero_stage=zero_stage,
            torch_dtype=torch_dtype,
            keep_tensors=is_main,
            weight_store=weight_store,
        )
        if is_main:
            try:
                save_file(shard_tensors, str(output_path / filename))
            except Exception as exc:
                error = exc
        del shard_tensors

    if error is not None:
        raise error

    if not is_main:
        return

    if len(shard_plan) > 1:
        weight_map = {
            hf_name: filename
            for filename, hf_names in shard_plan
            for hf_name in hf_names
        }
        total_size = sum(_spec_nbytes(spec, torch_dtype) for spec in specs)
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        (output_path / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2),
            encoding="utf-8",
        )

    pretrained.config.save_pretrained(output_path)
    if pretrained.generation_config is not None:
        pretrained.generation_config.save_pretrained(output_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)


def _tied_weight_names(pretrained: PreTrainedModel) -> set[str]:
    """HF names that share storage with another parameter and must not be written."""
    tied = pretrained._tied_weights_keys
    if tied is None:
        return set()
    if isinstance(tied, dict):
        return {str(key) for key in tied}
    if isinstance(tied, str):
        return {tied}
    return {str(name) for name in tied}


def _collect_export_specs(
    pretrained: PreTrainedModel,
    adapter_name: str,
    weight_store: BaseWeightStore | None = None,
) -> list[ExportItem]:
    """Build the HF-name → source list without materialising parameter values."""
    lora_prefixes = tuple(
        name
        for name, module in pretrained.named_modules()
        if isinstance(module, LoraLayer)
    )
    tied = _tied_weight_names(pretrained)
    specs: list[ExportItem] = []
    seen: set[str] = set()

    for module_name, module in pretrained.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        layer = _peft_lora_module(module)
        hf_weight = f"{module_name}.weight" if module_name else "weight"
        base = _as_lora_base(layer)
        if hf_weight not in tied and hf_weight not in seen:
            specs.append(
                ModuleLoraMerged(hf_weight, layer, adapter_name, "weight", base.weight)
            )
            seen.add(hf_weight)
        base_bias = _base_bias(base)
        if base_bias is None:
            continue
        hf_bias = f"{module_name}.bias" if module_name else "bias"
        if hf_bias not in tied and hf_bias not in seen:
            specs.append(
                ModuleLoraMerged(hf_bias, layer, adapter_name, "bias", base_bias)
            )
            seen.add(hf_bias)

    for name, tensor in _parameters_and_buffers(pretrained):
        if name in tied or name in seen:
            continue
        if any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in lora_prefixes
        ):
            continue
        if (
            weight_store is not None
            and tensor.device.type == "meta"
            and not weight_store.has(name)
        ):
            continue
        specs.append(ModuleCopy(name, tensor))
        seen.add(name)
    return specs


def _parameters_and_buffers(
    module: nn.Module,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield parameters then buffers in Hugging Face ``state_dict`` order."""
    yield from module.named_parameters()
    yield from module.named_buffers()


def _full_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    """Unsharded shape: ``ds_shape`` under ZeRO-3, else ``tensor.shape``."""
    with zero3_full_shape_views([tensor]):
        return tuple(tensor.shape)


def _spec_reference_tensor(spec: ExportItem) -> torch.Tensor:
    """A tensor whose shape/dtype describe the exported value (possibly sharded)."""
    if isinstance(spec, ModuleCopy):
        return spec.tensor
    return spec.reference


def _spec_nbytes(spec: ExportItem, torch_dtype: torch.dtype) -> int:
    """Byte size of the exported tensor at ``torch_dtype``."""
    reference = _spec_reference_tensor(spec)
    numel = 1
    for dim in _full_shape(reference):
        numel *= int(dim)
    dtype = torch_dtype if reference.is_floating_point() else reference.dtype
    return numel * dtype.itemsize


def _plan_shards(
    specs: Sequence[ExportItem],
    max_shard_size: str | int,
) -> list[tuple[str, list[str]]]:
    """Assign HF names to safetensors files without holding real weights."""
    meta_state = {
        spec.hf_name: torch.empty(
            _full_shape(_spec_reference_tensor(spec)),
            dtype=_spec_reference_tensor(spec).dtype,
            device="meta",
        )
        for spec in specs
    }
    split = split_torch_state_dict_into_shards(
        meta_state,
        max_shard_size=max_shard_size,
    )
    return list(split.filename_to_tensors.items())


def _layer_gather_params(
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d,
) -> list[torch.Tensor]:
    """Parameters that must be gathered to merge one LoRA module."""
    params: list[torch.Tensor] = []
    seen: set[int] = set()

    def _add(tensor: object) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        tid = id(tensor)
        if tid in seen:
            return
        seen.add(tid)
        params.append(tensor)

    base_layer = _as_lora_base(layer)
    _add(base_layer.weight)
    _add(_base_bias(base_layer))
    for _, tensor in (*layer.named_parameters(), *layer.named_buffers()):
        _add(tensor)
    return params


def _adapter_on_layer(
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d, adapter_name: str
) -> bool:
    """Return whether ``adapter_name`` has LoRA weights on ``layer``."""
    if adapter_name in layer.lora_A:
        return True
    return adapter_name in layer.lora_embedding_A


def _delta_weight(
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d, adapter_name: str
) -> torch.Tensor:
    """LoRA delta for ``adapter_name``.

    PEFT's CPU bf16 ``get_delta_weight`` writes the upcast-then-cast tensors
    back onto ``.data``; restore the gathered storage so ZeRO can re-shard.
    """
    delta_fn = layer.get_delta_weight
    if not callable(delta_fn):
        msg = f"{type(layer).__name__} cannot produce a LoRA delta"
        raise TypeError(msg)
    snapshots = [(param, param.data) for param in layer.parameters()]
    saved_embedding: list[tuple[nn.ParameterDict, str, torch.Tensor]] = []
    if adapter_name in layer.lora_embedding_A:
        saved_embedding.append(
            (layer.lora_embedding_A, adapter_name, layer.lora_embedding_A[adapter_name])
        )
    if adapter_name in layer.lora_embedding_B:
        saved_embedding.append(
            (layer.lora_embedding_B, adapter_name, layer.lora_embedding_B[adapter_name])
        )
    try:
        return delta_fn(adapter_name)
    finally:
        for param, data in snapshots:
            param.data = data
        for store, key, value in saved_embedding:
            store[key] = value


def _merged_weight_and_bias(
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d,
    adapter_name: str,
    torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fold LoRA into clones of the base weight and bias; do not write the module."""
    base_layer = _as_lora_base(layer)
    base_weight = base_layer.weight.detach()
    if _adapter_on_layer(layer, adapter_name):
        delta = _delta_weight(layer, adapter_name)
        weight = base_weight + delta.to(dtype=base_weight.dtype)
    else:
        weight = base_weight
    bias_tensor = _base_bias(base_layer)
    bias: torch.Tensor | None = None
    if bias_tensor is not None:
        if adapter_name in layer.lora_B:
            lora_b_mod = layer.lora_B[adapter_name]
            if isinstance(lora_b_mod, nn.Linear) and isinstance(
                lora_b_mod.bias, torch.Tensor
            ):
                scaling = layer.scaling[adapter_name]
                bias = bias_tensor.detach() + lora_b_mod.bias.detach() * scaling
            else:
                bias = bias_tensor.detach()
        else:
            bias = bias_tensor.detach()
    weight_cpu = weight.detach().to(dtype=torch_dtype, device="cpu").contiguous()
    bias_cpu = (
        None
        if bias is None
        else bias.detach().to(dtype=torch_dtype, device="cpu").contiguous()
    )
    return weight_cpu, bias_cpu


def _copy_to_cpu(tensor: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    """Detach ``tensor`` to contiguous CPU memory; cast floats to ``torch_dtype``."""
    detached = tensor.detach()
    if detached.is_floating_point():
        return detached.to(dtype=torch_dtype, device="cpu").contiguous()
    return detached.to(device="cpu").contiguous()


def _merged_checkpoint_names(hf_name: str) -> tuple[str, str | None]:
    """Base checkpoint keys for the wrapped layer's weight and bias."""
    if hf_name == "weight" or hf_name.endswith(".weight"):
        stem = hf_name[: -len("weight")].rstrip(".")
        return hf_name, f"{stem}.bias" if stem else "bias"
    if hf_name == "bias" or hf_name.endswith(".bias"):
        stem = hf_name[: -len("bias")].rstrip(".")
        return (f"{stem}.weight" if stem else "weight"), hf_name
    return hf_name, None


@contextmanager
def _base_layer_from_store(
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d,
    spec: ModuleLoraMerged,
    weight_store: BaseWeightStore | None,
) -> Generator[None, None, None]:
    """Temporarily place checkpoint tensors on a meta base layer."""
    if weight_store is None:
        yield
        return
    base_layer = _as_lora_base(layer)
    weight_name, bias_name = _merged_checkpoint_names(spec.hf_name)
    restored: list[tuple[nn.Module, str, object]] = []
    weight = base_layer.weight
    if weight.device.type == "meta":
        previous = _replace_tensor(base_layer, "weight", weight_store.load(weight_name))
        restored.append((base_layer, "weight", previous))
    bias = _base_bias(base_layer)
    if (
        bias is not None
        and bias.device.type == "meta"
        and bias_name is not None
        and weight_store.has(bias_name)
    ):
        previous = _replace_tensor(base_layer, "bias", weight_store.load(bias_name))
        restored.append((base_layer, "bias", previous))
    try:
        yield
    finally:
        for module, attr, previous in restored:
            setattr(module, attr, previous)


def _materialize_shard(
    specs: Sequence[ExportItem],
    hf_names: Sequence[str],
    *,
    zero_stage: int | None,
    torch_dtype: torch.dtype,
    keep_tensors: bool,
    weight_store: BaseWeightStore | None = None,
) -> dict[str, torch.Tensor]:
    """Gather and merge only the tensors that belong to one safetensors file."""
    by_name = {spec.hf_name: spec for spec in specs}
    shard: dict[str, torch.Tensor] = {}
    merged_done: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}

    for hf_name in hf_names:
        spec = by_name[hf_name]
        if isinstance(spec, ModuleCopy):
            source = spec.tensor
            if source.device.type == "meta":
                if weight_store is None:
                    msg = f"Cannot copy meta tensor {hf_name} without a weight store"
                    raise RuntimeError(msg)
                source = weight_store.load(hf_name)
            with gather_if_zero3(zero_stage, [source]):
                copied = _copy_to_cpu(source, torch_dtype)
            if keep_tensors:
                shard[hf_name] = copied
            continue

        layer_id = id(spec.layer)
        if layer_id not in merged_done:
            gather_params = _layer_gather_params(spec.layer)
            with gather_if_zero3(zero_stage, gather_params):
                with _base_layer_from_store(spec.layer, spec, weight_store):
                    merged_done[layer_id] = _merged_weight_and_bias(
                        spec.layer, spec.adapter_name, torch_dtype
                    )
        weight, bias = merged_done[layer_id]
        if not keep_tensors:
            continue
        if spec.part == "weight":
            shard[hf_name] = weight
        elif bias is not None:
            shard[hf_name] = bias

    return shard
