# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Layer-wise LoRA merge into a Hugging Face weight directory via FSDP2/DTensor gather."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Container, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import torch
from accelerate import init_empty_weights
from huggingface_hub.serialization import split_torch_state_dict_into_shards
from peft import PeftConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import (
    Conv1d,
    Conv2d,
    Conv3d,
    Embedding,
    Linear,
    LoraLayer,
    ParamWrapper,
)
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, PreTrainedModel
from transformers.core_model_loading import revert_weight_conversion
from transformers.pytorch_utils import Conv1D
from transformers.tokenization_utils_base import (
    ADDED_TOKENS_FILE,
    FULL_TOKENIZER_FILE,
    SPECIAL_TOKENS_MAP_FILE,
    TOKENIZER_CONFIG_FILE,
    PreTrainedTokenizerBase,
)
from transformers.utils import (
    CHAT_TEMPLATE_FILE,
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    IMAGE_PROCESSOR_NAME,
    LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
    PROCESSOR_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    VIDEO_PROCESSOR_NAME,
    cached_file,
)

from agilerl.architectures import trainer_model_class
from agilerl.arena.memory.formulas import DEFAULT_ROUTED_CHUNK_BYTES
from agilerl.distributed import (
    gather_params,
    is_main_process,
    materialize_dtensors,
    raise_on_any_rank,
)
from agilerl.distributed.fsdp import canonical_fsdp_param_fqn
from agilerl.distributed.fsdp_meta import (
    ConverterRoute,
    build_safetensors_key_files,
    checkpoint_quantization_config,
    convert_checkpoint_route,
    converter_checkpoint_routes,
    dequantizing_transform_groups,
)
from agilerl.distributed.tensor_parallel import (
    tensor_parallel_row_orders,
    tensor_rows_in_order,
)
from agilerl.lora.moe.adapters import wrapper_chain
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from agilerl.utils.stochastic_round import copy_synced_master_

DEFAULT_MAX_SHARD_SIZE = "5GB"
DEFAULT_TORCH_DTYPE = torch.bfloat16
ARENA_ARTIFACT_MANIFEST_FILENAME = "arena_artifact_manifest.json"
MERGED_COMPLETE_MARKER = ".complete"
HF_MERGED_FORMAT = "hf_merged"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"


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


@dataclass(frozen=True)
class ParamLoraMerged:
    """One ``ParamWrapper``-targeted parameter (e.g. packed experts) whose LoRA is merged."""

    hf_name: str
    wrapper: ParamWrapper
    adapter_name: str
    reference: torch.Tensor


ExportItem = ModuleCopy | ModuleLoraMerged | ParamLoraMerged


def export_merged_pretrained(
    output_dir: str | Path,
    *,
    model: nn.Module | None = None,
    adapter_path: str | Path | None = None,
    base_model_name_or_path: str | Path | None = None,
    adapter_name: str = "actor",
    max_shard_size: str | int = DEFAULT_MAX_SHARD_SIZE,
    torch_dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> None:
    """Write adapter-free Hugging Face weights without assembling a full state dict.

    Live-model shape: pass ``model`` (a PEFT-wrapped causal LM, or a value-head
    wrapper whose ``pretrained_model`` is PEFT). Replay shape: pass ``adapter_path``
    (the ``actor/`` PEFT directory) and ``base_model_name_or_path``; the merge
    keeps the base checkpoint's tensor names, files and quantization, so it
    loads wherever the base checkpoint does.

    :param output_dir: Directory for ``config.json``, safetensors, optional
        tokenizer / ``generation_config.json``, ``arena_artifact_manifest.json``,
        and ``.complete``.
    :param model: Live PEFT module currently on the training ranks.
    :param adapter_path: Saved PEFT adapter directory for replay.
    :param base_model_name_or_path: Base model id or local HF directory for replay.
    :param adapter_name: PEFT adapter to fold in, defaults to ``actor``.
    :param max_shard_size: Hugging Face shard budget (``"5GB"`` default); replay
        keeps the base checkpoint's files.
    :param torch_dtype: Dtype of written weights, defaults to ``bfloat16``;
        replay of a quantized checkpoint keeps its stored dtypes unless the
        adapter changes a quantized tensor, which dequantizes the checkpoint.
    :param tokenizer: Optional tokenizer written beside the weights.
    """
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
                max_shard_size=max_shard_size,
                torch_dtype=torch_dtype,
                tokenizer=tokenizer,
            )
            return
        if adapter_path is not None:
            if base_model_name_or_path is None:
                msg = "Replay export requires base_model_name_or_path with adapter_path"
                raise ValueError(msg)
            _export_replay(
                Path(output_dir),
                adapter_path,
                str(base_model_name_or_path),
                adapter_name,
                torch_dtype,
                tokenizer,
            )
            return
        if base_model_name_or_path is not None:
            msg = "Replay export requires adapter_path with base_model_name_or_path"
            raise ValueError(msg)
        msg = (
            "export_merged_pretrained expects either model=... (live) or "
            "adapter_path=... and base_model_name_or_path=... (replay)"
        )
        raise ValueError(msg)
    except (MergedExportError, TypeError, ValueError):
        raise
    except Exception as exc:
        raise MergedExportError(str(exc)) from exc


def _sha256_and_size(path: Path) -> tuple[str, int]:
    """Return the hex sha256 and byte length of ``path``."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def write_merged_completeness_files(output_dir: str | Path) -> None:
    """Write Arena ``inspect_merged`` markers for a successful HF export.

    :param output_dir: Directory that already holds merged Hugging Face files.
    """
    output_path = Path(output_dir)
    skip = {
        MERGED_COMPLETE_MARKER,
        ARENA_ARTIFACT_MANIFEST_FILENAME,
        ADAPTER_CONFIG_FILENAME,
    }
    files: list[dict[str, str | int]] = []
    for path in sorted(output_path.rglob("*")):
        if not path.is_file() or path.name in skip:
            continue
        rel = path.relative_to(output_path).as_posix()
        sha256, size = _sha256_and_size(path)
        files.append({"key": rel, "sha256": sha256, "bytes": size})
    manifest = {"format": HF_MERGED_FORMAT, "files": files}
    (output_path / ARENA_ARTIFACT_MANIFEST_FILENAME).write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    (output_path / MERGED_COMPLETE_MARKER).write_text("1", encoding="utf-8")


def merged_export_is_complete(merged_dir: str | Path) -> bool:
    """Return whether ``merged_dir`` has ``.complete`` and a non-empty ``hf_merged`` manifest.

    :param merged_dir: Directory written by :func:`export_merged_pretrained`.
    :return: True when the export finished.
    """
    merged = Path(merged_dir)
    manifest_path = merged / ARENA_ARTIFACT_MANIFEST_FILENAME
    if not (merged / MERGED_COMPLETE_MARKER).is_file() or not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload["format"] == HF_MERGED_FORMAT and len(payload["files"]) > 0


def _load_replay_peft(
    adapter_path: str | Path,
    base_model_name_or_path: str,
    adapter_name: str,
) -> PreTrainedModel:
    """Build a meta skeleton of the base with the saved adapter injected, its weights on CPU."""
    config = AutoConfig.from_pretrained(base_model_name_or_path)
    adapter_config = PeftConfig.from_pretrained(str(adapter_path))
    with init_empty_weights(include_buffers=True):
        base = trainer_model_class(base_model_name_or_path).from_config(config)
        peft_model = get_peft_model(base, adapter_config, adapter_name=adapter_name)
    _assign_adapter_weights(peft_model, adapter_path, adapter_name)
    return base


def _export_replay(
    output_dir: Path,
    adapter_path: str | Path,
    base_model_name_or_path: str,
    adapter_name: str,
    torch_dtype: torch.dtype,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    """Copy the base checkpoint's tensors under their stored names, adding the adapter's deltas.

    Each delta is computed under the module's name and mapped onto the stored
    keys with the model's weight conversions reversed, as ``save_pretrained``
    does. A delta on a quantized tensor dequantizes the whole checkpoint. One
    base file is read and written at a time.
    """
    if not is_main_process():
        return
    pretrained = _load_replay_peft(adapter_path, base_model_name_or_path, adapter_name)
    key_files = build_safetensors_key_files(base_model_name_or_path)
    quantization = checkpoint_quantization_config(pretrained.config.to_dict())
    routes = (
        converter_checkpoint_routes(
            key_files, dequantizing_transform_groups([("", [])], quantization)
        )
        if quantization is not None
        else {}
    )
    delta_specs = _replay_delta_specs(pretrained, adapter_name)
    delta_keys = _stored_delta_keys(
        pretrained, delta_specs, key_files.keys() | routes.keys()
    )
    dequantize = any(key in routes for key in delta_keys)
    outputs = _replay_outputs(key_files, routes if dequantize else {})
    unstored = sorted(key for key in delta_keys if key not in outputs)
    if unstored:
        msg = (
            f"Adapter {adapter_name!r} changes tensors {base_model_name_or_path} "
            f"does not store: {', '.join(unstored[:8])}"
        )
        raise ValueError(msg)

    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _write_replay_files(
        output_dir,
        pretrained,
        key_files,
        outputs,
        delta_specs,
        delta_keys,
        torch_dtype if quantization is None or dequantize else None,
    )
    pretrained.config.save_pretrained(output_dir)
    if dequantize:
        _drop_quantization_config(output_dir / CONFIG_NAME)
    _copy_base_files(base_model_name_or_path, output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    write_merged_completeness_files(output_dir)


def _write_replay_files(
    output_dir: Path,
    pretrained: PreTrainedModel,
    key_files: dict[str, str],
    outputs: dict[str, tuple[str, ConverterRoute | None]],
    delta_specs: dict[str, ReplayDelta],
    delta_keys: dict[str, str],
    cast_dtype: torch.dtype | None,
) -> None:
    """Write each base file's tensors with the adapter's deltas added, then the index.

    :param cast_dtype: Floating-point dtype for every written tensor, or ``None`` to keep stored dtypes.
    """
    pending: dict[str, dict[str, torch.Tensor]] = {}
    weight_map: dict[str, str] = {}
    total_size = 0
    for path in sorted({path for path, _ in outputs.values()}):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key, (key_path, route) in outputs.items():
                if key_path != path:
                    continue
                if route is None:
                    tensors[key] = handle.get_tensor(key)
                else:
                    tensors[key] = convert_checkpoint_route(
                        route, route.sources, key_files, pretrained.config
                    )[key]
        for key, tensor in tensors.items():
            if key in delta_keys:
                delta = _pop_stored_delta(
                    pretrained,
                    delta_specs[delta_keys[key]],
                    delta_keys[key],
                    key,
                    pending,
                )
                tensors[key] = _add_delta(
                    key, tensor, delta, cast_dtype or tensor.dtype
                )
            elif cast_dtype is not None and tensor.is_floating_point():
                tensors[key] = tensor.to(cast_dtype)
        filename = Path(path).name
        save_file(tensors, str(output_dir / filename))
        weight_map.update(dict.fromkeys(tensors, filename))
        total_size += sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        del tensors

    if set(weight_map.values()) != {SAFE_WEIGHTS_NAME}:
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        (output_dir / SAFE_WEIGHTS_INDEX_NAME).write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )


def _copy_base_files(base_model_name_or_path: str, output_dir: Path) -> None:
    """Copy the base's generation config, processor and tokenizer files that it has."""
    # vLLM builds a vision-language model's processor, tokenizer included, from the model directory.
    base_files = (
        GENERATION_CONFIG_NAME,
        PROCESSOR_NAME,
        IMAGE_PROCESSOR_NAME,
        VIDEO_PROCESSOR_NAME,
        CHAT_TEMPLATE_FILE,
        LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
        FULL_TOKENIZER_FILE,
        TOKENIZER_CONFIG_FILE,
        SPECIAL_TOKENS_MAP_FILE,
        ADDED_TOKENS_FILE,
        "tokenizer.model",
    )
    for name in base_files:
        path = cached_file(
            base_model_name_or_path,
            name,
            _raise_exceptions_for_missing_entries=False,
        )
        if path is not None:
            shutil.copyfile(path, output_dir / name)


@dataclass(frozen=True)
class ReplayDelta:
    """The adapter's change to one module-named weight or bias, computed on demand."""

    shape: tuple[int, ...]
    compute: Callable[[], torch.Tensor]


def _replay_delta_specs(
    pretrained: PreTrainedModel, adapter_name: str
) -> dict[str, ReplayDelta]:
    """Module-named weights and biases ``adapter_name`` changes."""
    specs: dict[str, ReplayDelta] = {}
    for spec in _collect_export_specs(pretrained, adapter_name):
        if isinstance(spec, ModuleCopy):
            continue
        shape = _full_shape(spec.reference)
        if isinstance(spec, ParamLoraMerged):
            if adapter_name in spec.wrapper.lora_A:
                specs[spec.hf_name] = ReplayDelta(
                    shape, partial(_param_delta, spec.wrapper, adapter_name)
                )
            continue
        if spec.part == "weight":
            if _adapter_on_layer(spec.layer, adapter_name):
                specs[spec.hf_name] = ReplayDelta(
                    shape, partial(_delta_weight, spec.layer, adapter_name)
                )
            continue
        bias = _lora_bias(spec.layer, adapter_name)
        if bias is not None:
            scaling = spec.layer.scaling[adapter_name]
            specs[spec.hf_name] = ReplayDelta(
                shape, partial(torch.mul, bias.detach(), scaling)
            )
    return specs


def _stored_delta_keys(
    pretrained: PreTrainedModel,
    delta_specs: dict[str, ReplayDelta],
    stored: Container[str],
) -> dict[str, str]:
    """Stored checkpoint key to the module-named weight whose delta it receives.

    A weight maps to its reverted names, or to its own name when the checkpoint
    stores it that way (packed experts saved in the in-memory layout).
    """
    delta_keys: dict[str, str] = {}
    for hf_name, delta in delta_specs.items():
        meta = {hf_name: torch.empty(delta.shape, device="meta")}
        reverted = list(revert_weight_conversion(pretrained, meta))
        if hf_name in stored and not all(key in stored for key in reverted):
            reverted = [hf_name]
        for key in reverted:
            delta_keys[key] = hf_name
    return delta_keys


def _replay_outputs(
    key_files: dict[str, str], routes: dict[str, ConverterRoute]
) -> dict[str, tuple[str, ConverterRoute | None]]:
    """Written key to the base file it goes beside, and the dequantize route that builds it."""
    consumed = {key for route in routes.values() for _, key in route.sources}
    outputs: dict[str, tuple[str, ConverterRoute | None]] = {
        key: (path, None) for key, path in key_files.items() if key not in consumed
    }
    for target, route in routes.items():
        outputs[target] = (key_files[route.sources[0][1]], route)
    return outputs


def _pop_stored_delta(
    pretrained: PreTrainedModel,
    delta: ReplayDelta,
    hf_name: str,
    key: str,
    pending: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    """The fp32 delta for stored ``key``; the other stored keys of ``hf_name`` wait in ``pending``."""
    if key == hf_name:
        return delta.compute().float()
    if hf_name not in pending:
        pending[hf_name] = revert_weight_conversion(
            pretrained, {hf_name: delta.compute().float()}
        )
    stored = pending[hf_name]
    value = stored.pop(key)
    if not stored:
        del pending[hf_name]
    return value


def _param_delta(wrapper: ParamWrapper, adapter_name: str) -> torch.Tensor:
    """fp32 LoRA delta of a wrapped parameter from its factors; the base may be meta."""
    weight_a = wrapper.lora_A[adapter_name].get_parameter("weight").float()
    weight_b = wrapper.lora_B[adapter_name].get_parameter("weight").float()
    scaling = wrapper.scaling[adapter_name]
    n_experts = wrapper.num_experts
    if n_experts == 1:
        return weight_b @ weight_a * scaling
    # PEFT's stacked layouts: A is [E*r, in] or [E, r, in]; B is [out, E*r] or [out, r, E].
    weight_a = weight_a.reshape(n_experts, -1, weight_a.shape[-1])
    weight_b = weight_b.reshape(weight_b.shape[0], -1, n_experts).permute(2, 0, 1)
    delta = torch.bmm(weight_b, weight_a) * scaling
    return delta if wrapper._did_swap_in_out_features else delta.mT


def _lora_bias(
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d, adapter_name: str
) -> torch.Tensor | None:
    """The LoRA B bias ``adapter_name`` trains on ``layer``, if any."""
    if adapter_name not in layer.lora_B:
        return None
    lora_b = layer.lora_B[adapter_name]
    if isinstance(lora_b, nn.Linear) and isinstance(lora_b.bias, torch.Tensor):
        return lora_b.bias
    return None


def _add_delta(
    key: str, base: torch.Tensor, delta: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """``base + delta`` in fp32, cast once to ``dtype``."""
    if not base.is_floating_point() or base.element_size() == 1:
        msg = f"Cannot add a LoRA delta to {base.dtype} tensor {key!r}"
        raise ValueError(msg)
    if tuple(base.shape) != tuple(delta.shape):
        msg = (
            f"LoRA delta for {key!r} has shape {tuple(delta.shape)}, "
            f"the stored tensor {tuple(base.shape)}"
        )
        raise ValueError(msg)
    return (base.float() + delta).to(dtype)


def _drop_quantization_config(config_path: Path) -> None:
    """Remove ``quantization_config`` from a written ``config.json`` and its text config."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("quantization_config", None)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config.pop("quantization_config", None)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _adapter_checkpoint_key(param_name: str, adapter_name: str) -> str:
    """Map a live PEFT parameter FQN to the adapter safetensors key.

    PEFT writes Linear factors as ``lora_A.weight`` (adapter name was the
    second-to-last segment) and Embedding factors as ``lora_embedding_A``
    (adapter name was the last segment, no ``.weight``).
    """
    adapter_suffix = f".{adapter_name}"
    if param_name.endswith(adapter_suffix):
        return param_name.removesuffix(adapter_suffix)
    prefix, sep, suffix = param_name.rpartition(".")
    if sep and prefix.endswith(adapter_suffix):
        return f"{prefix.removesuffix(adapter_suffix)}.{suffix}"
    return param_name


def _assign_adapter_weights(
    model: nn.Module,
    adapter_path: str | Path,
    adapter_name: str,
) -> None:
    """Move saved LoRA tensors onto the meta skeleton as CPU parameters.

    Tied modules (``embed_tokens`` / ``lm_head``) share one Parameter.
    ``named_parameters()`` lists that object once; this walk includes
    duplicates so ``lm_head.base_layer.weight`` in the adapter file is
    consumed when it aliases the embedding.
    """
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
    seen_params: set[int] = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        key = canonical_fsdp_param_fqn(_adapter_checkpoint_key(name, adapter_name))
        tensor = tensors.get(key)
        if tensor is None:
            continue
        if id(param) not in seen_params:
            _replace_named_tensor(model, name, tensor)
            seen_params.add(id(param))
        used.add(key)
    leftover = set(tensors) - used
    if leftover:
        keys = sorted(leftover)
        preview = ", ".join(keys[:12])
        if len(keys) > 12:
            preview = f"{preview}, ... ({len(keys)} total)"
        msg = (
            "Adapter tensors were not on the PEFT model rebuilt from "
            f"adapter_config.json (adapter {adapter_name!r} under "
            f"{adapter_path}): {preview}"
        )
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
    max_shard_size: str | int,
    torch_dtype: torch.dtype,
    tokenizer: PreTrainedTokenizerBase | None,
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
    specs = _collect_export_specs(pretrained, adapter_name)
    shard_plan = _plan_shards(specs, max_shard_size)
    hf_row_orders = {
        id(model.get_parameter(fqn)): torch.argsort(order)
        for fqn, order in tensor_parallel_row_orders(model).items()
    }
    output_path = Path(output_dir)
    is_main = is_main_process()

    with raise_on_any_rank():
        if is_main:
            if output_path.is_dir():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True)

    for filename, hf_names in shard_plan:
        with raise_on_any_rank():
            shard_tensors = _materialize_shard(
                specs,
                hf_names,
                hf_row_orders,
                torch_dtype=torch_dtype,
                keep_tensors=is_main,
            )
            if is_main:
                save_file(shard_tensors, str(output_path / filename))
            del shard_tensors

    with raise_on_any_rank():
        if is_main:
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
            write_merged_completeness_files(output_path)


def _collect_export_specs(
    pretrained: PreTrainedModel,
    adapter_name: str,
) -> list[ExportItem]:
    """Build the HF-name → source list without materialising parameter values."""
    lora_prefixes = tuple(
        name
        for name, module in pretrained.named_modules()
        if isinstance(module, LoraLayer)
    )
    inner_links = {
        id(module.base_layer)
        for module in pretrained.modules()
        if isinstance(module, ParamWrapper)
    }
    # Respects ``tie_word_embeddings``: an untied model keeps its ``lm_head``.
    tied = set(pretrained.all_tied_weights_keys)
    specs: list[ExportItem] = []
    seen: set[str] = set()

    for live_name, module in pretrained.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        module_name = canonical_fsdp_param_fqn(live_name)
        if isinstance(module, ParamWrapper):
            if id(module) not in inner_links:
                specs.extend(_param_wrapper_specs(module_name, module, adapter_name))
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

    for live_name, tensor in _parameters_and_buffers(pretrained):
        if any(
            live_name == prefix or live_name.startswith(f"{prefix}.")
            for prefix in lora_prefixes
        ):
            continue
        name = canonical_fsdp_param_fqn(live_name)
        if name in tied or name in seen:
            continue
        specs.append(ModuleCopy(name, tensor))
        seen.add(name)
    return specs


def _param_wrapper_specs(
    module_name: str, wrapper: ParamWrapper, adapter_name: str
) -> list[ParamLoraMerged | ModuleCopy]:
    """Specs for every tensor of the module a (nested) ``ParamWrapper`` chain wraps.

    Wrapped parameters merge their own link's LoRA; the rest (e.g. expert
    biases) copy unchanged. Names drop the chain's ``base_layer`` segments.
    """
    links = wrapper_chain(wrapper)
    prefix = f"{module_name}." if module_name else ""
    specs: list[ParamLoraMerged | ModuleCopy] = []
    for name, tensor in _parameters_and_buffers(wrapper.get_base_layer()):
        link = links.get(name)
        if link is None:
            specs.append(ModuleCopy(f"{prefix}{name}", tensor))
        else:
            specs.append(ParamLoraMerged(f"{prefix}{name}", link, adapter_name, tensor))
    return specs


def _parameters_and_buffers(
    module: nn.Module,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield parameters then persistent buffers, the tensors ``state_dict`` holds."""
    yield from module.named_parameters()
    for owner_name, owner in module.named_modules():
        prefix = f"{owner_name}." if owner_name else ""
        for name, buffer in owner.named_buffers(recurse=False):
            if name not in owner._non_persistent_buffers_set:
                yield f"{prefix}{name}", buffer


def _full_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    """Unsharded shape (a ``DTensor`` reports its global shape)."""
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
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d | ParamWrapper,
    adapter_name: str,
) -> torch.Tensor:
    """LoRA delta for ``adapter_name``.

    PEFT's CPU bf16 ``get_delta_weight`` writes the upcast-then-cast tensors
    back onto ``.data``; restore the gathered storage so shards can be reinstalled.
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
    weight_order: torch.Tensor | None,
    bias_order: torch.Tensor | None,
    delta_order: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fold LoRA into the gathered base weight and bias, in HF row order; do not write the module."""
    base_layer = _as_lora_base(layer)
    base_weight = tensor_rows_in_order(
        _dense_local(base_layer.weight).detach(), weight_order
    )
    if _adapter_on_layer(layer, adapter_name):
        delta = tensor_rows_in_order(
            _dense_local(_delta_weight(layer, adapter_name)), delta_order
        )
        weight = base_weight + delta.to(dtype=base_weight.dtype)
    else:
        weight = base_weight
    bias_tensor = _base_bias(base_layer)
    bias: torch.Tensor | None = None
    if bias_tensor is not None:
        bias = tensor_rows_in_order(_dense_local(bias_tensor), bias_order).detach()
        lora_bias = _lora_bias(layer, adapter_name)
        if lora_bias is not None:
            scaling = layer.scaling[adapter_name]
            bias = bias + _dense_local(lora_bias).detach() * scaling
    return weight, bias


def _param_gather_params(
    wrapper: ParamWrapper, adapter_name: str
) -> list[torch.Tensor]:
    """The wrapped parameter and this link's LoRA factors for ``adapter_name``."""
    params = [wrapper.get_param()]
    if adapter_name in wrapper.lora_A:
        params.extend(wrapper.lora_A[adapter_name].parameters())
        params.extend(wrapper.lora_B[adapter_name].parameters())
    return params


def _merged_param(
    wrapper: ParamWrapper, adapter_name: str, order: torch.Tensor | None
) -> torch.Tensor:
    """Fold LoRA into the gathered wrapped parameter, in HF row order; do not write the module."""
    base = tensor_rows_in_order(_dense_local(wrapper.get_param()).detach(), order)
    if adapter_name not in wrapper.lora_A:
        return base
    delta = _dense_local(_delta_weight(wrapper, adapter_name))
    return base + delta.to(dtype=base.dtype)


def _dense_local(tensor: torch.Tensor) -> torch.Tensor:
    """Plain tensor for one parameter; ``full_tensor`` if it is still a DTensor."""
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


def _copy_to_cpu(tensor: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    """Offload one unsharded tensor to owned CPU storage; cast floats to ``torch_dtype``."""
    dense = _dense_local(tensor)
    detached = dense.detach()
    if detached.is_floating_point():
        copied = detached.to(dtype=torch_dtype, device="cpu").contiguous()
    else:
        copied = detached.to(device="cpu").contiguous()
    copied = copied.clone()
    if dense is not tensor:
        del dense
    return copied


def _copied_tensor(
    spec: ModuleCopy,
    hf_row_orders: dict[int, torch.Tensor],
    torch_dtype: torch.dtype,
    keep_tensors: bool,
) -> torch.Tensor | None:
    """Gather one non-LoRA tensor in HF row order; the CPU copy, or ``None`` when not kept."""
    if spec.tensor.device.type == "meta":
        msg = f"Cannot export meta tensor {spec.hf_name}"
        raise RuntimeError(msg)
    with materialize_dtensors(spec.tensor) as gathered:
        dense = gathered[0]
        if dense is None:
            msg = f"gathered tensor for {spec.hf_name} is missing"
            raise RuntimeError(msg)
        if not keep_tensors:
            return None
        dense = tensor_rows_in_order(dense, hf_row_orders.get(id(spec.tensor)))
        return _copy_to_cpu(dense, torch_dtype)


def _materialize_shard(
    specs: Sequence[ExportItem],
    hf_names: Sequence[str],
    hf_row_orders: dict[int, torch.Tensor],
    *,
    torch_dtype: torch.dtype,
    keep_tensors: bool,
) -> dict[str, torch.Tensor]:
    """Gather and merge one safetensors file: one module (or tensor) at a time.

    Each ``gather_params`` unshards that FSDP unit, rank 0 copies to CPU, then
    the context restores DTensor shards so the rest of the model stays sharded.
    ``hf_row_orders`` maps ``id`` of a TP-permuted parameter to the rows that
    put its gathered value in HF order.
    """
    by_name = {spec.hf_name: spec for spec in specs}
    shard: dict[str, torch.Tensor] = {}
    merged_cpu: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
    gathered_layers: set[int] = set()

    for hf_name in hf_names:
        spec = by_name[hf_name]
        if isinstance(spec, ModuleCopy):
            copied = _copied_tensor(spec, hf_row_orders, torch_dtype, keep_tensors)
            if copied is not None:
                shard[hf_name] = copied
            continue

        if isinstance(spec, ParamLoraMerged):
            wrapper = spec.wrapper
            order = hf_row_orders.get(id(spec.reference))
            with gather_params(
                wrapper, _param_gather_params(wrapper, spec.adapter_name)
            ):
                merged = _merged_param(wrapper, spec.adapter_name, order)
                if keep_tensors:
                    shard[hf_name] = _copy_to_cpu(merged, torch_dtype)
                del merged
            continue

        layer_id = id(spec.layer)
        if layer_id not in gathered_layers:
            base_layer = _as_lora_base(spec.layer)
            weight_order = hf_row_orders.get(id(base_layer.weight))
            bias_order = hf_row_orders.get(id(_base_bias(base_layer)))
            lora_b = spec.layer.lora_B
            delta_order = (
                hf_row_orders.get(id(lora_b[spec.adapter_name].weight))
                if spec.adapter_name in lora_b
                else None
            )
            layer_params = _layer_gather_params(spec.layer)
            with gather_params(spec.layer, layer_params):
                weight, bias = _merged_weight_and_bias(
                    spec.layer,
                    spec.adapter_name,
                    weight_order,
                    bias_order,
                    delta_order,
                )
                if keep_tensors:
                    merged_cpu[layer_id] = (
                        _copy_to_cpu(weight, torch_dtype),
                        None if bias is None else _copy_to_cpu(bias, torch_dtype),
                    )
            gathered_layers.add(layer_id)
        if not keep_tensors:
            continue
        weight, bias = merged_cpu[layer_id]
        if spec.part == "weight":
            shard[hf_name] = weight
        elif bias is not None:
            shard[hf_name] = bias

    return shard


@dataclass(frozen=True)
class MergedSyncSpec:
    """One weight merged sync pushes, with the rows that put its merge in Hugging Face order.

    ``weight_order`` reorders the gathered base rows and ``delta_order`` the
    gathered LoRA B rows; ``None`` keeps the stored order.
    """

    item: ModuleLoraMerged | ParamLoraMerged
    weight_order: torch.Tensor | None
    delta_order: torch.Tensor | None


def merged_sync_specs(model: nn.Module, adapter_name: str) -> list[MergedSyncSpec]:
    """Weights ``adapter_name`` changes, under Hugging Face checkpoint names, in module order.

    Biases are left out: the adapters train no LoRA bias. Row orders of
    tensor-parallel-permuted weights sit on their parameter's device.

    :param model: PEFT-wrapped causal LM, or a value-head wrapper around one.
    :param adapter_name: Adapter whose merged weights are pushed.
    :return: One spec per adapted weight or packed-expert parameter.
    :raises ValueError: If the adapter is missing, adapts no weight, or is DoRA.
    """
    peft_model = _as_peft_model(model)
    if adapter_name not in peft_model.peft_config:
        msg = f"Adapter {adapter_name!r} is not on the PEFT model"
        raise ValueError(msg)
    pretrained = peft_model.get_base_model()
    if not isinstance(pretrained, PreTrainedModel):
        msg = f"Merged sync requires a transformers model, got {type(pretrained).__name__}"
        raise TypeError(msg)
    hf_row_orders: dict[int, torch.Tensor] = {}
    for fqn, order in tensor_parallel_row_orders(peft_model).items():
        param = peft_model.get_parameter(fqn)
        hf_row_orders[id(param)] = torch.argsort(order).to(param.device)
    specs: list[MergedSyncSpec] = []
    for item in _collect_export_specs(pretrained, adapter_name):
        if isinstance(item, ParamLoraMerged) and adapter_name in item.wrapper.lora_A:
            # Tensor-parallel plans permute rows of linear layers only.
            specs.append(MergedSyncSpec(item, None, None))
        elif (
            isinstance(item, ModuleLoraMerged)
            and item.part == "weight"
            and _adapter_on_layer(item.layer, adapter_name)
        ):
            # The merge pushes W + scale * B @ A, without DoRA's magnitude.
            if item.layer.use_dora.get(adapter_name):
                msg = f"Merged sync does not support DoRA adapters ({item.hf_name})"
                raise ValueError(msg)
            lora_b = item.layer.lora_B
            delta_order = (
                hf_row_orders.get(id(lora_b[adapter_name].weight))
                if adapter_name in lora_b
                else None
            )
            specs.append(
                MergedSyncSpec(item, hf_row_orders.get(id(item.reference)), delta_order)
            )
    if not specs:
        msg = f"Adapter {adapter_name!r} adapts no weight"
        raise ValueError(msg)
    return specs


@torch.no_grad()
def write_merged_weight(
    spec: MergedSyncSpec, destination: torch.Tensor | None
) -> Exception | None:
    """Gather one adapted module and write ``W + scale * B @ A`` into ``destination``.

    The merge runs in one fp32 buffer, in Hugging Face row order, and casts
    once to ``destination``'s dtype. Only the base weight and this adapter's
    LoRA factors are gathered. Every rank calls this for the same specs in the
    same order; a rank passing ``None`` only joins the gather. A failed gather
    raises; a merge failing after it is returned, so the caller can still join
    the gathers that follow.

    :param spec: Spec from :func:`merged_sync_specs`.
    :param destination: Tensor of the weight's full shape, or ``None``.
    :return: The error the merge raised, or ``None``.
    """
    item = spec.item
    if isinstance(item, ParamLoraMerged):
        gathered = gather_params(
            item.wrapper, _param_gather_params(item.wrapper, item.adapter_name)
        )
    else:
        gathered = gather_params(
            item.layer,
            [
                _as_lora_base(item.layer).weight,
                *_adapter_factors(item.layer, item.adapter_name),
            ],
        )
    with gathered:
        if destination is None:
            return None
        try:
            _write_gathered_merge(spec, destination)
        except Exception as exc:
            return exc
    return None


def _write_gathered_merge(spec: MergedSyncSpec, destination: torch.Tensor) -> None:
    """Write the merge of one gathered module into ``destination``."""
    item = spec.item
    if isinstance(item, ParamLoraMerged):
        _write_merged_param(item.wrapper, item.adapter_name, destination)
        return
    layer = item.layer
    adapter_name = item.adapter_name
    base = tensor_rows_in_order(
        _dense_local(_as_lora_base(layer).weight), spec.weight_order
    )
    if type(layer) is Linear and adapter_name not in layer.lora_variant:
        _write_merged_linear(layer, adapter_name, base, spec.delta_order, destination)
        return
    with _fp32_lora_factors(layer, adapter_name):
        delta = tensor_rows_in_order(
            layer.get_delta_weight(adapter_name), spec.delta_order
        )
    copy_synced_master_(destination, base.float() + delta)


def _adapter_factors(
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d, adapter_name: str
) -> list[nn.Parameter]:
    """``adapter_name``'s LoRA parameters on ``layer``."""
    return [
        param
        for name, param in layer.named_parameters()
        if name.startswith("lora_") and adapter_name in name.split(".")
    ]


def _write_merged_linear(
    layer: Linear,
    adapter_name: str,
    base: torch.Tensor,
    delta_order: torch.Tensor | None,
    destination: torch.Tensor,
) -> None:
    """Accumulate ``scale * B @ A`` onto an fp32 copy of the gathered base, then write it."""
    weight_a = _dense_local(layer.get_parameter(f"lora_A.{adapter_name}.weight"))
    weight_b = _dense_local(layer.get_parameter(f"lora_B.{adapter_name}.weight"))
    weight_a = weight_a.float()
    weight_b = tensor_rows_in_order(weight_b, delta_order).float()
    merged = base.to(torch.float32, copy=True)
    # A fan-in-fan-out base stores [in, out]; accumulate through its transpose.
    target = merged.T if layer.fan_in_fan_out else merged
    target.addmm_(weight_b, weight_a, alpha=layer.scaling[adapter_name])
    copy_synced_master_(destination, merged)


@contextmanager
def _fp32_lora_factors(
    layer: Linear | Embedding | Conv1d | Conv2d | Conv3d, adapter_name: str
) -> Generator[None, None, None]:
    """Hold ``adapter_name``'s LoRA factors on ``layer`` in fp32, restoring their storage after."""
    factors = _adapter_factors(layer, adapter_name)
    saved = [param.data for param in factors]
    for param in factors:
        param.data = _dense_local(param.data).float()
    try:
        yield
    finally:
        for param, data in zip(factors, saved, strict=True):
            param.data = data


def _write_merged_param(
    wrapper: ParamWrapper, adapter_name: str, destination: torch.Tensor
) -> None:
    """Merge a gathered packed-expert parameter in fp32, a bounded chunk of experts at a time."""
    base = _dense_local(wrapper.get_param())
    weight_a = _dense_local(wrapper.lora_A[adapter_name].get_parameter("weight"))
    weight_b = _dense_local(wrapper.lora_B[adapter_name].get_parameter("weight"))
    scaling = wrapper.scaling[adapter_name]
    n_experts = wrapper.num_experts
    if n_experts == 1:
        merged = base.to(torch.float32, copy=True)
        merged.addmm_(weight_b.float(), weight_a.float(), alpha=scaling)
        copy_synced_master_(destination, merged)
        return
    # PEFT's stacked layouts: A is [E*r, in] or [E, r, in]; B is [out, E*r] or [out, r, E].
    weight_a = weight_a.reshape(n_experts, -1, weight_a.shape[-1])
    weight_b = weight_b.reshape(weight_b.shape[0], -1, n_experts).permute(2, 0, 1)
    out_major = wrapper._did_swap_in_out_features
    chunk = max(1, DEFAULT_ROUTED_CHUNK_BYTES // (base[0].numel() * 4))
    for start in range(0, n_experts, chunk):
        stop = min(start + chunk, n_experts)
        expert_a = weight_a[start:stop].float()
        expert_b = weight_b[start:stop].float()
        merged = base[start:stop].to(torch.float32, copy=True)
        if out_major:
            merged.baddbmm_(expert_b, expert_a, alpha=scaling)
        else:
            merged.baddbmm_(expert_a.mT, expert_b.mT, alpha=scaling)
        copy_synced_master_(destination[start:stop], merged)
