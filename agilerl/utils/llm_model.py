# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import logging
import re
import warnings
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator
from torch import nn

from agilerl import HAS_DEEPSPEED, HAS_LLM_DEPENDENCIES

if TYPE_CHECKING:
    from peft import LoraConfig
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.tokenization_utils_base import (
        PreTrainedTokenizerBase,
    )

else:
    PreTrainedTokenizerBase = object

logger = logging.getLogger(__name__)

if HAS_LLM_DEPENDENCIES:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.modeling_utils import PreTrainedModel

    from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
else:
    # Sentinels for missing optional LLM dependencies. All uses are gated on
    # HAS_LLM_DEPENDENCIES, so these are never reached at runtime.
    PreTrainedModel = Any
    Dataset = Any
    AutoModelForCausalLM: Any = None
    AutoModelForCausalLMWithValueHead: Any = None
    BitsAndBytesConfig: Any = None

# Sentinel when DeepSpeed is absent; overwritten by the real enum otherwise.
ZeroParamStatus = None
if HAS_DEEPSPEED:
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

DEPRECATED_LLM_ENV_NAMES = frozenset(("apply_chat_template",))

# Accepted spellings per bitsandbytes quantization preset
BNB_QUANT_NONE_ALIASES = frozenset({"none"})
BNB_QUANT_INT8_ALIASES = frozenset({"int8"})
BNB_QUANT_NF4_ALIASES = frozenset(
    {"nf4", "4bit", "4-bit", "bnb-4bit", "bnb_4bit"},
)
BNB_QUANT_PRESETS = (
    BNB_QUANT_NONE_ALIASES | BNB_QUANT_INT8_ALIASES | BNB_QUANT_NF4_ALIASES
)

# Gemma 4 wraps projections in *ClippableLinear; PEFT must target the inner ``.linear``
# submodule via regex (see https://github.com/huggingface/peft/issues/3129).
_CLIPPABLE_LINEAR_WRAPPER_SUFFIX = "ClippableLinear"



def _as_optional_int(value: object | None) -> int | None:
    """Parse HF token ids that arrive as ``int`` or numeric ``str``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _eos_id_set(eos_id: object | None) -> frozenset[int]:
    """Parse HF ``eos_token_id`` as a set of ints (scalar or sequence)."""
    if eos_id is None or isinstance(eos_id, bool):
        return frozenset()
    if isinstance(eos_id, (int, str)):
        parsed = _as_optional_int(eos_id)
        return frozenset({parsed}) if parsed is not None else frozenset()
    if isinstance(eos_id, (list, tuple)):
        ids: set[int] = set()
        for item in eos_id:
            parsed = _as_optional_int(item)
            if parsed is not None:
                ids.add(parsed)
        return frozenset(ids)
    return frozenset()


def _coerce_distinct_pad_id(pad_id: object | None, eos_id: object | None) -> int | None:
    """Return ``pad_id`` as int when set and distinct from every ``eos_id``."""
    pad_int = _as_optional_int(pad_id)
    if pad_int is None:
        return None
    if pad_int in _eos_id_set(eos_id):
        return None
    return pad_int


def resolve_pad_token_id(
    tokenizer: PreTrainedTokenizerBase,
    *,
    model_config: object | None = None,
    generation_config: object | None = None,
) -> tuple[int, str]:
    """Resolve a pad token id that prefers not to alias ``tokenizer.eos_token_id``.

    Priority when each candidate is set and ``!= eos_token_id``:

    1. ``model_config.pad_token_id``
    2. ``generation_config.pad_token_id``
    3. ``tokenizer.pad_token_id``
    4. ``tokenizer.unk_token_id``
    5. ``tokenizer.eos_token_id`` (last resort; warns)

    :param tokenizer: Hugging Face tokenizer.
    :type tokenizer: PreTrainedTokenizerBase
    :param model_config: Optional model config with ``pad_token_id``.
    :type model_config: object | None
    :param generation_config: Optional generation config with ``pad_token_id``.
    :type generation_config: object | None
    :return: ``(pad_token_id, source_label)``.
    :rtype: tuple[int, str]
    """
    eos_id = tokenizer.eos_token_id

    candidates: list[tuple[object | None, str]] = []
    if model_config is not None:
        candidates.append((getattr(model_config, "pad_token_id", None), "model.config"))
    if generation_config is not None:
        candidates.append(
            (getattr(generation_config, "pad_token_id", None), "generation_config")
        )
    candidates.append((tokenizer.pad_token_id, "tokenizer.pad_token_id"))
    candidates.append((tokenizer.unk_token_id, "tokenizer.unk_token_id"))

    for pad_id, source in candidates:
        resolved = _coerce_distinct_pad_id(pad_id, eos_id)
        if resolved is not None:
            return resolved, source

    eos_ids = _eos_id_set(eos_id)
    if not eos_ids:
        msg = "Tokenizer has no eos_token_id; cannot resolve a pad token id."
        raise ValueError(msg)

    warnings.warn(
        "No pad token id distinct from eos_token_id; using eos_token_id as pad. "
        "Mid-sequence EOS tokens (e.g. ChatML turn boundaries) will be masked "
        "as padding when attention uses ids != pad_token_id.",
        UserWarning,
        stacklevel=2,
    )
    return min(eos_ids), "tokenizer.eos_token_id"


def apply_pad_token_id(tokenizer: PreTrainedTokenizerBase, pad_token_id: int) -> None:
    """Set ``tokenizer.pad_token`` / ``pad_token_id`` from the resolved id.

    :param tokenizer: Hugging Face tokenizer to update.
    :type tokenizer: PreTrainedTokenizerBase
    :param pad_token_id: Resolved pad token id.
    :type pad_token_id: int
    """
    pad_token_id = int(pad_token_id)
    token_str: str | None = None

    eos_ids = _eos_id_set(tokenizer.eos_token_id)
    unk_id = _as_optional_int(tokenizer.unk_token_id)
    if pad_token_id in eos_ids:
        eos_tok = tokenizer.eos_token
        if isinstance(eos_tok, str):
            token_str = eos_tok
    if token_str is None and unk_id is not None and pad_token_id == unk_id:
        unk_tok = tokenizer.unk_token
        if isinstance(unk_tok, str):
            token_str = unk_tok
    if token_str is None:
        try:
            converted = tokenizer.convert_ids_to_tokens(pad_token_id)
        except Exception:
            converted = None
        if isinstance(converted, str):
            token_str = converted

    if token_str is not None:
        tokenizer.pad_token = token_str
    tokenizer.pad_token_id = pad_token_id


def load_pad_token_configs(
    model_name_or_path: str | None,
) -> tuple[object | None, object | None]:
    """Load model and generation configs for pad-token resolution.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str | None
    :return: ``(model_config, generation_config)``; either may be ``None``.
    :rtype: tuple[object | None, object | None]
    """
    if not HAS_LLM_DEPENDENCIES or not model_name_or_path:
        return None, None

    model_config: object | None = None
    generation_config: object | None = None
    try:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(model_name_or_path)
    except Exception:
        model_config = None
    try:
        from transformers import GenerationConfig

        generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    except Exception:
        generation_config = None
    return model_config, generation_config



def gather_tensor(
    tensor: torch.Tensor | npt.NDArray | float,
    accelerator: Accelerator,
) -> torch.Tensor:
    """Gather tensors from gpus.

    :param tensor: Tensor (or array/scalar convertible to one) to gather
    :type tensor: torch.Tensor | npt.NDArray | float
    :param accelerator: Accelerator object
    :type accelerator: accelerate.Accelerator
    :return: Stacked tensors
    :rtype: torch.Tensor
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device=accelerator.device)
    tensor = tensor.to(accelerator.device)
    return accelerator.gather(tensor)


def needs_cross_rank_seq_padding(algo: object, *, world_size: int) -> bool:
    """Return whether ranks must sync completion seq lengths before ``learn()``.

    Multi-rank Liger token-level losses chunk over ``B * (T - 1)`` and issue one
    NCCL allreduce per chunk (DAPO/CISPO normaliser). ZeRO-3 parameter gathers
    also require identical per-rank ``T`` so every rank issues the same NCCL
    collectives. Divergent per-rank ``T`` after local ``stack_and_pad`` therefore
    deadlocks.
    """
    if world_size <= 1:
        return False
    zero_stage = getattr(algo, "zero_stage", 0)
    if zero_stage == 3 or zero_stage == "3":
        return True
    if not getattr(algo, "use_liger_loss", False):
        return False
    return getattr(algo, "importance_sampling_level", "token") == "token"


def allreduce_minmax_int(value: int, accelerator: Accelerator) -> tuple[int, int]:
    """Return ``(min, max)`` of ``value`` across Accelerate ranks.

    Uses :meth:`Accelerator.gather` so the reduction participates in the same
    process-group bookkeeping as the rest of the Accelerate/DeepSpeed run
    (plain ``torch.distributed.all_reduce`` on the default group is easy to
    desync from DeepSpeed's communicator set).
    """
    t = torch.tensor([int(value)], device=accelerator.device, dtype=torch.long)
    gathered = accelerator.gather(t)
    return int(gathered.min().item()), int(gathered.max().item())


def pad_completion_batch_to_seq_len(
    token_ids: torch.Tensor,
    action_masks: torch.Tensor,
    *,
    target_seq_len: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad stacked completions/masks to ``target_seq_len`` / ``target-1``.

    Pad positions use ``pad_token_id`` and ``False`` so they contribute nothing
    to the masked CISPO/Liger objective.
    """
    required_dims = 2
    if token_ids.dim() != required_dims:
        msg = f"token_ids must be (B, T), got shape {tuple(token_ids.shape)}"
        raise ValueError(msg)
    if action_masks.dim() != required_dims:
        msg = f"action_masks must be (B, T-1), got shape {tuple(action_masks.shape)}"
        raise ValueError(msg)

    batch, seq_len = token_ids.shape
    mask_len = action_masks.shape[1]
    if mask_len != seq_len - 1:
        msg = (
            f"action_masks length ({mask_len}) must be token_ids length ({seq_len}) - 1"
        )
        raise ValueError(msg)
    if target_seq_len < seq_len:
        msg = f"target_seq_len ({target_seq_len}) must be >= local seq_len ({seq_len})"
        raise ValueError(msg)
    if target_seq_len == seq_len:
        return token_ids, action_masks

    pad_t = target_seq_len - seq_len
    token_ids = torch.nn.functional.pad(
        token_ids,
        (0, pad_t),
        value=pad_token_id,
    )
    action_masks = torch.nn.functional.pad(
        action_masks,
        (0, pad_t),
        value=False,
    )
    if token_ids.shape != (batch, target_seq_len):
        msg = (
            f"padded completions shape {tuple(token_ids.shape)} != "
            f"({batch}, {target_seq_len})"
        )
        raise RuntimeError(msg)
    if action_masks.shape != (batch, target_seq_len - 1):
        msg = (
            f"padded masks shape {tuple(action_masks.shape)} != "
            f"({batch}, {target_seq_len - 1})"
        )
        raise RuntimeError(msg)
    return token_ids, action_masks


def _local_batch_and_seq_len(
    completion_ids: list[torch.Tensor] | torch.Tensor,
) -> tuple[int, int]:
    """Return ``(B, T)`` from a stacked batch or a list of variable-length rows."""
    if isinstance(completion_ids, list):
        if not completion_ids:
            return 0, 0
        return len(completion_ids), max(int(t.shape[-1]) for t in completion_ids)
    if not isinstance(completion_ids, torch.Tensor):
        msg = f"completion_ids must be a tensor or list, got {type(completion_ids)}"
        raise TypeError(msg)
    if completion_ids.dim() != 2:
        msg = f"completion_ids must be (B, T), got shape {tuple(completion_ids.shape)}"
        raise ValueError(msg)
    return int(completion_ids.shape[0]), int(completion_ids.shape[1])


def align_completion_batch_shapes_across_ranks(
    completion_ids: Any,  # noqa: ANN401 -- cross-rank batch of variable-length sequences; element type varies by paradigm
    action_masks: Any,  # noqa: ANN401 -- see completion_ids
    rewards: Any,  # noqa: ANN401 -- see completion_ids
    *,
    pad_token_id: int,
    accelerator: Accelerator,
    minmax_fn: Callable[[int], tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sync ``B``/``T`` across ranks before heavy local pad/stack work.

    Collective metadata sync happens first; pad/stack follows. A DP barrier runs
    after pad so no rank enters ZeRO ``learn`` collectives while peers are still
    padding. Call immediately before ``learn()`` when
    :func:`needs_cross_rank_seq_padding` is true. Shorter ranks are right-padded
    to the global max ``T`` so Liger token-level chunk collectives stay in
    lockstep.
    """
    # Lazy import avoids a circular dependency with algo_utils -> llm_utils.
    from agilerl.utils.algo_utils import stack_and_pad_experiences

    local_b, local_t = _local_batch_and_seq_len(completion_ids)
    reduce_fn = minmax_fn or (lambda value: allreduce_minmax_int(value, accelerator))

    min_b, max_b = reduce_fn(local_b)
    if min_b != max_b:
        msg = (
            f"Completion batch row counts diverge across ranks "
            f"(local B={local_b}, min={min_b}, max={max_b}); refusing to pad "
            f"rows. Check dataloader / data_batch_size_per_gpu sharding."
        )
        raise RuntimeError(msg)

    _min_t, max_t = reduce_fn(local_t)

    completion_ids, action_masks, rewards = stack_and_pad_experiences(
        completion_ids,
        action_masks,
        rewards,
        padding_values=[pad_token_id, False, 0.0],
    )
    if int(completion_ids.shape[1]) < max_t:
        completion_ids, action_masks = pad_completion_batch_to_seq_len(
            completion_ids,
            action_masks,
            target_seq_len=max_t,
            pad_token_id=pad_token_id,
        )
    if int(completion_ids.shape[1]) != max_t:
        msg = (
            f"Cross-rank seq align failed: local T={completion_ids.shape[1]} "
            f"!= global max T={max_t}"
        )
        raise RuntimeError(msg)

    accelerator.wait_for_everyone()
    return completion_ids, action_masks, rewards


def aggregate_metrics_across_gpus(
    accelerator: Accelerator | None,
    metric_tensor: torch.Tensor | npt.NDArray | float,
) -> float:
    """Aggregate gathered tensors.

    :param accelerator: Accelerator object
    :type accelerator: accelerate.Accelerator | None
    :param metric_tensor: Metrics
    :type metric_tensor: torch.Tensor | npt.NDArray | float
    :return: Mean metric
    :rtype: float
    """
    if accelerator is None:
        if isinstance(metric_tensor, torch.Tensor):
            return metric_tensor.float().mean().item()
        return float(metric_tensor)
    all_metrics = gather_tensor(metric_tensor, accelerator)
    return all_metrics.mean().item()


def safe_aggregate_metrics(
    accelerator: Accelerator | None,
    metrics: torch.Tensor | npt.NDArray | float,
) -> float:
    """Aggregate metrics generically, handling both when an accelerator is being used and when it isn't.

    :param accelerator: Accelerator object
    :type accelerator: Accelerator | None
    :param metrics: Metrics
    :type metrics: torch.Tensor | npt.NDArray | float
    :return: Mean metric
    :rtype: float
    """
    if accelerator is None:
        if isinstance(metrics, (torch.Tensor, np.ndarray)):
            return float(
                np.mean(metrics)
                if isinstance(metrics, np.ndarray)
                else metrics.float().mean().item()
            )
        return float(metrics)
    return aggregate_metrics_across_gpus(accelerator, metrics)


def aggregate_metrics_dict(
    accelerator: Accelerator | None,
    metrics: dict[str, torch.Tensor | npt.NDArray | float],
) -> dict[str, float]:
    """Aggregate all values in a metrics dict across GPUs (or locally if no accelerator).

    :param accelerator: Accelerator object (or None for single-device).
    :type accelerator: Accelerator | None
    :param metrics: Dictionary mapping metric names to raw values.
    :type metrics: dict[str, torch.Tensor | npt.NDArray | float]
    :return: Dictionary with all values aggregated to floats.
    :rtype: dict[str, float]
    """
    return {k: safe_aggregate_metrics(accelerator, v) for k, v in metrics.items()}


@contextmanager
def gather_if_zero3(
    zero_stage: int | None,
    params: list[torch.Tensor],
    modifier_rank: int | None = None,
) -> Generator[None, None, None]:
    """Conditional context manager for setting the zero stage for the model.

    :param zero_stage: The zero stage
    :type zero_stage: int | None
    :param params: The parameters to gather
    :type params: list[torch.Tensor]
    :param modifier_rank: The modifier rank
    :type modifier_rank: int | None
    """
    if zero_stage == 3:
        if not HAS_DEEPSPEED:
            msg = (
                "DeepSpeed is required for ZeRO stage 3 parameter gathering, but it "
                "is not installed."
            )
            raise ImportError(msg)
        # Lazy: deepspeed is an optional dependency and only ZeRO-3 needs it.
        import deepspeed

        with deepspeed.zero.GatheredParameters(
            params=params,
            modifier_rank=modifier_rank,
        ):
            yield
    else:
        yield


@contextmanager
def zero3_full_shape_views(
    params: list[torch.Tensor],
) -> Generator[None, None, None]:
    """Expose full ``ds_shape`` views on partitioned ZeRO-3 params for shape-only reads.

    Each partitioned param temporarily swaps its placeholder ``data`` for a
    zero-storage view (a scalar expanded to ``ds_shape``), so shape, dtype and
    device reads inside the block see the full tensor without an all-gather.
    Values must not be read inside the block; the placeholder is restored on
    exit. Params without ``ds_shape``, or already ``AVAILABLE``, pass through
    untouched.

    :param params: Candidate parameters; only partitioned ZeRO-3 params get views.
    :type params: list[torch.Tensor]
    """
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    seen: set[int] = set()
    try:
        for param in params:
            if id(param) in seen:
                continue
            ds_shape = getattr(param, "ds_shape", None)
            if ds_shape is None:
                continue
            status = getattr(param, "ds_status", None)
            if getattr(status, "name", None) == "AVAILABLE":
                continue
            seen.add(id(param))
            saved.append((param, param.data))
            param.data = torch.empty((), dtype=param.dtype, device=param.device).expand(
                tuple(ds_shape)
            )
        yield
    finally:
        for param, data in saved:
            param.data = data


@contextmanager
def gather_if_ds_param(
    *tensors: torch.Tensor | None,
    modifier_rank: int | None = 0,
) -> Generator[None, None, None]:
    """Allgather ZeRO-3 params for the duration of the block.

    No-op when none of ``tensors`` carry a DeepSpeed ``ds_id``, or when every
    such param is already ``AVAILABLE`` (tied embeddings owned by
    ``embed_tokens``). Duplicate references are gathered once (by identity).
    Defaults to ``modifier_rank=0`` so DeepSpeed releases the gathered buffer
    after the block.

    The gather must wrap only the matmul / fused loss that reads the weight —
    not a module ``forward`` — because ZeRO-3's post-forward hooks
    re-partition the param and free the gathered buffer.

    :param tensors: Candidate weight tensors; only those with ``ds_id`` gather.
    :param modifier_rank: Passed to DeepSpeed ``GatheredParameters``.
    """
    seen: set[int] = set()
    params: list[torch.Tensor] = []
    for t in tensors:
        if t is None or not hasattr(t, "ds_id"):
            continue
        tid = id(t)
        if tid in seen:
            continue
        seen.add(tid)
        if (
            ZeroParamStatus is not None
            and hasattr(t, "ds_status")
            and t.ds_status == ZeroParamStatus.AVAILABLE
        ):
            continue
        params.append(t)
    if not params:
        yield
        return
    with gather_if_zero3(3, params, modifier_rank=modifier_rank):
        yield


def adapter_checkpoint_params(model: nn.Module) -> list[torch.Tensor]:
    """Return the parameters PEFT's adapter checkpoint I/O reads and writes.

    Scopes ZeRO-3 gathers around ``save_pretrained`` /
    ``get_peft_model_state_dict`` / ``set_peft_model_state_dict`` so base
    parameters stay sharded instead of being materialised on every rank.

    :param model: The model to collect adapter checkpoint parameters from.
    :type model: nn.Module
    :return: LoRA A/B and DoRA magnitude parameters, plus the
        ``modules_to_save`` copies PEFT writes into the same checkpoint (an LLM
        PPO value head).
    :rtype: list[torch.Tensor]
    """
    return [
        p for n, p in model.named_parameters() if "lora" in n or "modules_to_save" in n
    ]


def get_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Get the state dict of the model for zero3.

    :param model: The model to get the state dict of.
    :type model: nn.Module
    :return: The state dict of the model.
    :rtype: dict[str, torch.Tensor]
    """
    with gather_if_zero3(3, list(model.parameters()), modifier_rank=0):
        return model.state_dict()


def build_bnb_quantization_config(
    spec: str | dict[str, Any] | None,
) -> BitsAndBytesConfig | None:
    """Build a trainer-side ``BitsAndBytesConfig`` from a YAML-friendly spec.

    Lets ``QUANTIZATION`` be set declaratively in a config file / ``INIT_HP``
    instead of constructing a :class:`~transformers.BitsAndBytesConfig` in
    Python. Accepted forms:

    * ``None`` or ``"none"`` -- no quantization (BF16 baseline); returns ``None``.
    * ``"int8"`` -- LLM.int8() 8-bit weights.
    * ``"nf4"``, ``"4bit"``, ``"4-bit"``, ``"bnb-4bit"`` or ``"bnb_4bit"`` --
      4-bit NF4 with bf16 compute, bf16 quant storage and double quantisation
      (the QLoRA recipe, ZeRO-3 / FSDP compatible).
    * ``dict`` -- forwarded verbatim as ``BitsAndBytesConfig(**spec)`` for full
      control; ``bnb_4bit_compute_dtype`` / ``bnb_4bit_quant_storage`` may be
      given as dtype strings (e.g. ``"bfloat16"``), which transformers resolves.

    Preset names are matched case-insensitively with whitespace removed.

    :param spec: Quantization preset name, ``BitsAndBytesConfig`` kwargs dict,
        or ``None``.
    :type spec: str | dict[str, Any] | None
    :return: A configured ``BitsAndBytesConfig``, or ``None`` for no quantization.
    :rtype: BitsAndBytesConfig | None
    """
    if spec is None:
        return None
    if not HAS_LLM_DEPENDENCIES:
        msg = "Quantization requires optional LLM dependencies (install agilerl[llm])."
        raise ImportError(msg)
    if isinstance(spec, BitsAndBytesConfig):
        return spec
    if isinstance(spec, dict):
        return BitsAndBytesConfig(**spec)
    if isinstance(spec, str):
        mode = "".join(spec.split()).lower()
        if mode in BNB_QUANT_NONE_ALIASES:
            return None
        if mode in BNB_QUANT_INT8_ALIASES:
            return BitsAndBytesConfig(load_in_8bit=True)
        if mode in BNB_QUANT_NF4_ALIASES:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        msg = (
            f"Unknown quantization preset {spec!r}; expected one of "
            f"{sorted(BNB_QUANT_PRESETS)} or a BitsAndBytesConfig kwargs dict."
        )
        raise ValueError(msg)
    msg = (
        f"QUANTIZATION must be a preset name, a BitsAndBytesConfig kwargs dict, "
        f"or None; got {type(spec).__name__}."
    )
    raise TypeError(msg)


def model_has_clippable_linear_wrappers(model: nn.Module) -> bool:
    """Return True when the model uses *ClippableLinear projection wrappers."""
    return any(
        mod.__class__.__name__.endswith(_CLIPPABLE_LINEAR_WRAPPER_SUFFIX)
        for mod in model.modules()
    )


def discover_clippable_projection_leaf_names(model: nn.Module) -> list[str]:
    """Return leaf names (e.g. ``q_proj``) of *ClippableLinear wrapper modules."""
    names: set[str] = set()
    for name, mod in model.named_modules():
        if mod.__class__.__name__.endswith(_CLIPPABLE_LINEAR_WRAPPER_SUFFIX):
            names.add(name.rsplit(".", 1)[-1])
    return sorted(names)


def discover_clippable_inner_linear_module_keys(model: nn.Module) -> list[str]:
    """Return full ``named_modules`` keys for inner ``.linear`` weights in wrappers."""
    named = dict(model.named_modules())
    keys: list[str] = []
    for name, mod in model.named_modules():
        if not mod.__class__.__name__.endswith(_CLIPPABLE_LINEAR_WRAPPER_SUFFIX):
            continue
        inner_key = f"{name}.linear"
        inner = named.get(inner_key)
        if inner is not None and _is_peft_adaptable_linear(inner):
            keys.append(inner_key)
    return sorted(keys)


def _is_peft_adaptable_linear(module: nn.Module) -> bool:
    """Return True when PEFT LoRA can wrap this weight module."""
    if isinstance(module, nn.Linear):
        return True
    return module.__class__.__name__ in ("Linear4bit", "Linear8bitLt")


def peft_target_key_matches(key: str, target_modules: str | list[str]) -> bool:
    """Return ``True`` when a module key matches a PEFT ``target_modules`` spec.

    Mirrors PEFT's own matching rules so AgileRL selects exactly the modules
    PEFT would wrap: a string spec is a regex (``fullmatch``); a list matches
    a key exactly or by ``.suffix``. Used to filter LoRA state dicts for vLLM
    export consistently with how the adapters were attached.

    :param key: Dotted module key (e.g. ``model.layers.0.self_attn.q_proj``).
    :type key: str
    :param target_modules: PEFT ``target_modules`` regex or suffix list.
    :type target_modules: str | list[str]
    :return: Whether the key matches.
    :rtype: bool
    """
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, key) is not None
    if key in target_modules:
        return True
    return any(key.endswith(f".{target_key}") for target_key in target_modules)


def _peft_key_is_excluded(key: str, exclude_modules: list[str] | None) -> bool:
    if not exclude_modules:
        return False
    return any(
        key == excluded or key.endswith(f".{excluded}") or excluded in key
        for excluded in exclude_modules
    )


def list_peft_matched_module_keys(
    model: nn.Module,
    target_modules: str | list[str],
    *,
    exclude_modules: list[str] | None = None,
) -> list[str]:
    """List module keys that PEFT would adapt for the given target spec."""
    matched: list[str] = []
    for key, _ in model.named_modules():
        if _peft_key_is_excluded(key, exclude_modules):
            continue
        if peft_target_key_matches(key, target_modules):
            matched.append(key)
    return matched


def _looks_like_peft_target_regex(spec: str) -> bool:
    """Heuristic: the user passed a regex ``target_modules`` spec.

    Needed by :func:`adapt_lora_config_for_model` to pass regex specs through
    untouched instead of rewriting them like plain suffix names.
    """
    return spec.startswith(".*") or r"\." in spec or "(" in spec


def build_clippable_linear_lora_target_regex(
    projection_names: list[str],
) -> str:
    r"""Build a PEFT regex that targets inner ``.linear`` inside *ClippableLinear.

    Example::

        .*\.(q_proj|k_proj|v_proj)\.linear

    For projections under a multimodal scope, use
    :func:`build_scoped_lora_target_regex` instead (it also matches plain
    ``nn.Linear`` projections under the scope).

    See https://github.com/huggingface/peft/issues/3129
    """
    if not projection_names:
        msg = "At least one projection name is required for the LoRA target regex."
        raise ValueError(msg)
    alts = "|".join(re.escape(name) for name in sorted(set(projection_names)))
    # Optional prefix so top-level and nested keys both fullmatch.
    return rf"(?:.*\.)?({alts})\.linear"


def build_scoped_lora_target_regex(
    projection_names: list[str],
    scope: str,
) -> str:
    r"""PEFT regex for projections under a multimodal scope (e.g. ``language_model``).

    Matches both:

    * *ClippableLinear* inner weights (``…q_proj.linear``) on vision/audio towers
    * Plain ``nn.Linear`` leaves (``…q_proj``) on the language model in Gemma 4

    PEFT uses ``re.fullmatch`` on ``named_modules()`` keys (peft#3129).
    """
    if not projection_names:
        msg = "At least one projection name is required for the LoRA target regex."
        raise ValueError(msg)
    alts = "|".join(re.escape(name) for name in sorted(set(projection_names)))
    scope_esc = re.escape(scope.strip("."))
    return rf".*\.{scope_esc}.*\.(?:{alts})(?:\.linear)?$"


def _normalize_projection_leaf_name(name: str) -> str:
    """Strip a trailing ``.linear`` so YAML can use either ``q_proj`` or ``q_proj.linear``."""
    return name.removesuffix(".linear")


def _projection_names_for_clippable_lora(
    model: nn.Module, raw_targets: str | Iterable[str] | None
) -> list[str] | None:
    """Resolve projection leaf names, or ``None`` if ``raw_targets`` is already regex."""
    if raw_targets is None:
        return None
    raw_list = [raw_targets] if isinstance(raw_targets, str) else list(raw_targets)
    if any(_looks_like_peft_target_regex(t) for t in raw_list):
        return None

    names: list[str] = []
    for target in raw_list:
        if target == "all-linear":
            names.extend(discover_clippable_projection_leaf_names(model))
        else:
            names.append(_normalize_projection_leaf_name(target))
    return sorted(set(names))


def build_clippable_linear_lora_target_suffixes(
    projection_names: list[str],
) -> list[str]:
    """Build PEFT list targets ``q_proj.linear`` (suffix match, avoids regex pitfalls)."""
    return [f"{name}.linear" for name in sorted(set(projection_names))]


def _infer_clippable_lora_scope(model: nn.Module) -> str | None:
    """Infer ``language_model`` / ``audio_tower`` scope only when all inner keys live under it."""
    inner_keys = discover_clippable_inner_linear_module_keys(model)
    if not inner_keys:
        return None
    for scope in ("language_model", "audio_tower"):
        if all(
            key == scope or key.startswith(f"{scope}.") or f".{scope}." in key
            for key in inner_keys
        ):
            return scope
    return None


def _clone_lora_config_with_targets(
    lora_config: LoraConfig, target_modules: str | list[str]
) -> LoraConfig:
    """Return a new ``LoraConfig`` with updated ``target_modules``."""
    if hasattr(lora_config, "to_dict"):
        cfg_dict = lora_config.to_dict()
        cfg_dict["target_modules"] = target_modules
        return lora_config.__class__(**cfg_dict)
    adapted = copy.deepcopy(lora_config)
    adapted.target_modules = target_modules
    return adapted


def _example_module_keys_for_lora_scope(
    model: nn.Module,
    scope: str,
    *,
    limit: int = 3,
) -> list[str]:
    """Sample ``named_modules`` keys under a scope (for error messages)."""
    scope_token = scope.strip(".")
    keys = [
        key
        for key, _ in model.named_modules()
        if f".{scope_token}." in key or key.startswith(f"{scope_token}.")
    ]
    return keys[:limit]


def adapt_lora_config_for_model(
    model: nn.Module,
    lora_config: LoraConfig,
    *,
    lora_target_scope: str | None = None,
) -> LoraConfig:
    r"""Rewrite ``LoraConfig.target_modules`` for PEFT (regex or suffix list).

    For *ClippableLinear* models (Gemma 4 vision/audio), short names like ``q_proj``
    substring-match the wrapper and fail injection; use regex targets on the inner
    ``.linear`` (https://github.com/huggingface/peft/issues/3129).

    When ``lora_target_scope`` is set (e.g. ``language_model``), only modules under
    that path are targeted. The scoped regex matches plain ``nn.Linear`` language
    layers and ``.linear`` inside ClippableLinear towers. Unscoped fallbacks are
    not used when the scope is explicit.
    """
    raw_targets = lora_config.target_modules
    projection_names = _projection_names_for_clippable_lora(model, raw_targets)
    if projection_names is None:
        return lora_config

    exclude_modules = list(getattr(lora_config, "exclude_modules", None) or [])

    explicit_scope = lora_target_scope is not None
    scope = lora_target_scope if explicit_scope else _infer_clippable_lora_scope(model)

    if scope is not None:
        if not projection_names:
            msg = (
                "lora_target_scope is set but no projection names were resolved "
                f"from target_modules={raw_targets!r}."
            )
            raise ValueError(msg)
        target_spec = build_scoped_lora_target_regex(projection_names, scope)
        matched = list_peft_matched_module_keys(
            model, target_spec, exclude_modules=exclude_modules
        )
        if not matched:
            examples = _example_module_keys_for_lora_scope(model, scope)
            msg = (
                f"No modules matched scoped LoRA target_modules for scope={scope!r} "
                f"(regex={target_spec!r}). "
                f"Example keys under scope: {examples}. "
                "Check LORA_TARGET_SCOPE and TARGET_MODULES match the model layout."
            )
            raise ValueError(msg)
        adapted = _clone_lora_config_with_targets(lora_config, target_spec)
        logger.info(
            "Adapted LoRA target_modules via scoped regex (%s) (%d modules), e.g. %s",
            scope,
            len(matched),
            ", ".join(matched[:2]) + ("..." if len(matched) > 2 else ""),
        )
        return adapted

    if not model_has_clippable_linear_wrappers(model):
        return lora_config

    if not projection_names:
        msg = (
            "Model uses ClippableLinear wrappers but no projection names were "
            "resolved for LoRA targeting."
        )
        raise ValueError(msg)

    suffix_targets = build_clippable_linear_lora_target_suffixes(projection_names)
    candidate_specs: list[tuple[str, str | list[str]]] = [
        ("suffix list", suffix_targets),
        ("regex", build_clippable_linear_lora_target_regex(projection_names)),
    ]

    for label, target_spec in candidate_specs:
        matched = list_peft_matched_module_keys(
            model, target_spec, exclude_modules=exclude_modules
        )
        if not matched:
            continue
        if raw_targets == target_spec:
            return lora_config
        adapted = _clone_lora_config_with_targets(lora_config, target_spec)
        logger.info(
            "Adapted LoRA target_modules for ClippableLinear wrappers via %s "
            "(%d modules), e.g. %s",
            label,
            len(matched),
            ", ".join(matched[:2]) + ("..." if len(matched) > 2 else ""),
        )
        return adapted

    sample_inner = discover_clippable_inner_linear_module_keys(model)[:5]
    tried = [f"{label}={spec!r}" for label, spec in candidate_specs]
    msg = (
        "No modules matched LoRA target_modules for ClippableLinear layers. "
        f"Tried: {'; '.join(tried)}. "
        f"Example inner linear keys: {sample_inner}. "
        "Set LORA_TARGET_SCOPE in INIT_HP if the tower path differs, or pass a "
        "custom target_modules regex."
    )
    raise ValueError(msg)


def log_cuda_memory_snapshot(label: str, device_index: int = 0) -> None:
    """Log allocated/reserved CUDA memory (GiB) for colocated vLLM/HF debugging."""
    if not torch.cuda.is_available():
        return
    device = torch.device(f"cuda:{device_index}")
    alloc_gib = torch.cuda.memory_allocated(device) / (1024**3)
    reserved_gib = torch.cuda.memory_reserved(device) / (1024**3)
    logger.info(
        "%s: CUDA[%s] allocated=%.2f GiB reserved=%.2f GiB",
        label,
        device_index,
        alloc_gib,
        reserved_gib,
    )


def format_colocated_vllm_oom_hint(
    device_index: int = 0,
    *,
    kv_cache_memory_bytes: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    trainer_on_gpu: bool = True,
) -> str:
    """Build a human-readable VRAM summary after vLLM ``wake_up`` OOM."""
    if not torch.cuda.is_available():
        return "CUDA is not available on this host."

    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    alloc_bytes = torch.cuda.memory_allocated(device_index)
    free_gib = free_bytes / (1024**3)
    total_gib = total_bytes / (1024**3)
    alloc_gib = alloc_bytes / (1024**3)

    lines = [
        (
            f"CUDA device {device_index}: {total_gib:.2f} GiB total, "
            f"{alloc_gib:.2f} GiB torch-allocated, {free_gib:.2f} GiB free (driver)."
        ),
    ]
    if kv_cache_memory_bytes is not None:
        kv_gib = kv_cache_memory_bytes / (1024**3)
        shortfall = max(0.0, kv_gib - free_gib)
        lines.append(
            f"vLLM kv_cache_memory_bytes requests {kv_gib:.2f} GiB on wake_up; "
            f"≈{shortfall:.2f} GiB more free VRAM is needed if that allocation failed."
        )
    if gpu_memory_utilization is not None:
        lines.append(
            f"vLLM gpu_memory_utilization={gpu_memory_utilization} is also checked at "
            "init (requires that fraction of total VRAM to appear free)."
        )
    if max_model_len is not None:
        lines.append(
            f"max_model_len={max_model_len} sets the KV slot length cap (long contexts "
            "need large KV pools even when max_num_seqs=1)."
        )
    if trainer_on_gpu:
        lines.append(
            "The DeepSpeed trainer was still on GPU when wake_up ran. With sleep_mode, "
            "AgileRL moves trainer weights to CPU before wake_up; ZeRO optimizer state "
            "may still remain on GPU."
        )
    lines.append(
        "Check nvidia-smi right before the failure. To reduce peak: unset "
        "kv_cache_memory_bytes, lower max_model_len, lower vllm gpu_memory_utilization, "
        "or use a smaller max_num_seqs / group_size."
    )
    return "\n".join(lines)


def resolve_attn_implementation(requested: str | None = None) -> str:
    """Pick the most memory-efficient attention implementation available.

    Long-context RL rollouts make attention the memory bottleneck. SDPA's flash
    backend is only used when there is no explicit attention mask, but
    sliding-window models (e.g. Gemma) pass an explicit mask, so SDPA falls back
    to a backend that materialises the full TxT scores/bias (OOM at ~30k
    tokens). ``flash_attention_2`` (the ``flash_attn`` package) uses native
    windowed-causal attention with O(T) memory and no TxT mask, so prefer it
    when installed; otherwise fall back to SDPA.

    To force a specific backend, pass it explicitly — e.g.
    ``model_config={"attn_implementation": "flex_attention"}`` on the algorithm
    for PyTorch's built-in FlexAttention (block-sparse masked attention with
    O(T) memory and no TxT mask, which handles sliding-window models at long
    context without needing the ``flash_attn`` package).

    :param requested: An explicit choice from the caller. Anything other than
        ``None`` / ``"auto"`` is returned unchanged (caller stays authoritative).
    :type requested: str | None
    :return: The attention implementation string for ``from_pretrained`` /
        ``from_config``.
    :rtype: str
    """
    if requested is not None and requested != "auto":
        return requested
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def flex_decode_kernel_options(
    query: torch.Tensor, attention_mask: torch.Tensor | BlockMask | None = None
) -> dict[str, Any] | None:
    """Return a ``BLOCK_M`` that keeps short-query flex forwards compilable.

    Queries under 128 tokens hit inductor's flex-decoding kernel, whose default
    ``BLOCK_M`` (``next_power_of_2(seq_len_q * gqa_shared_heads)``) can exceed
    the mask's Q block size; every candidate config then fails a divisibility
    filter and compilation raises ``NoValidChoicesError``. Pinning ``BLOCK_M``
    to the mask's Q block size avoids that on every GPU generation.

    :param query: Query tensor, shaped ``[batch, heads, seq_len_q, head_dim]``.
    :type query: torch.Tensor
    :param attention_mask: Mask passed to flex attention; a ``BlockMask``
        carries the Q block size, anything else uses torch's 128 default.
    :type attention_mask: torch.Tensor | BlockMask | None
    :return: ``{"BLOCK_M": ...}`` for short queries, else ``None``.
    :rtype: dict[str, Any] | None
    """
    shape = getattr(query, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    seq_len_q = shape[-2]
    if not isinstance(seq_len_q, int) or seq_len_q >= 128:
        return None

    q_block_size = 128
    block_size = getattr(attention_mask, "BLOCK_SIZE", None)
    if isinstance(block_size, tuple) and block_size:
        # Maskless calls get one giant block; cap at torch's 128 default.
        q_block_size = min(int(block_size[0]), 128)
    return {"BLOCK_M": q_block_size}


def patch_flex_attention_kernel_options(options: dict[str, Any] | None = None) -> None:
    """Inject SRAM-safe Triton ``kernel_options`` into transformers' flex-attn path.

    FlexAttention's default Triton config for large head dims (e.g. Gemma's
    head_dim=256) needs more shared memory than an A100 has (~208 KB required vs
    ~163 KB available), so the autotuner finds "no valid triton configs"
    (OutOfMemoryError: out of resource: triton_tem_fused_flex_attention_0).
    The flex attention function accepts ``kernel_options`` to shrink the block
    sizes / pipeline stages; this registers a wrapper over the
    ``"flex_attention"`` entry that supplies safe defaults when the caller
    passes none. Idempotent; no-op if transformers/flex is unavailable.

    **Auto-detect**: when ``options`` is ``None``, A100 (SM80) and earlier get
    the SRAM-safe small blocks; Hopper (SM90+) fits the stock tiles, so only
    short-query forwards get a ``BLOCK_M`` there (see
    :func:`flex_decode_kernel_options`) and the autotuner keeps everything else.

    :param options: Override the default kernel options (forward + backward
        block sizes, ``num_warps``, ``num_stages``). Installed unconditionally
        when given.
    :type options: dict[str, Any] | None
    """
    try:
        from transformers.integrations.flex_attention import flex_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    if getattr(flex_attention_forward, "_agilerl_kernel_opts_patched", False):
        return

    # Small blocks exist only to fit pre-SM90 SRAM; Hopper keeps the
    # autotuner's blocks and gets a decode-safe BLOCK_M per call instead.
    needs_sram_safe_blocks = True
    if options is None and torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            capability = (0, 0)
        needs_sram_safe_blocks = capability < (9, 0)

    # head_dim=256 makes the Q/K/V tiles (BLOCK x head_dim) the dominant SRAM
    # cost, so use small 32-wide blocks to fit the A100's ~163 KB shared memory.
    opts = options
    if opts is None and needs_sram_safe_blocks:
        opts = {
            # Forward kernel blocks.
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            # Backward kernel blocks (training).
            "BLOCK_M1": 16,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 16,
            "num_warps": 4,
            "num_stages": 2,
        }

    def _flex_with_opts(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | BlockMask,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        resolved = (
            opts
            if opts is not None
            else flex_decode_kernel_options(query, attention_mask)
        )
        if resolved is not None:
            kwargs.setdefault("kernel_options", resolved)
        return flex_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )

    # Dynamic marker attribute on the wrapper function (checked via getattr
    # above); function attributes are not statically declarable.
    _flex_with_opts._agilerl_kernel_opts_patched = True  # ty: ignore[unresolved-attribute]
    try:
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = _flex_with_opts
    except Exception:
        ALL_ATTENTION_FUNCTIONS.register("flex_attention", _flex_with_opts)


def create_model_from_name_or_path(
    model_name_or_path: str,
    model_config: dict[str, Any] | None = None,
    add_value_head: bool = False,
    use_accelerator: bool = False,
) -> PreTrainedModel:
    """Create a model from a name or path.

    :param model_name_or_path: The name or path of the model to create.
    :type model_name_or_path: str
    :param model_config: Extra keyword arguments forwarded to ``from_pretrained``
        (e.g. a ``quantization_config``). ``torch_dtype`` and
        ``attn_implementation`` are filled in as defaults when not already
        present, so passing a config never silently disables SDPA attention.
    :type model_config: dict[str, Any ] | None
    :param use_value_head: Flag to indicate if a value head should be added to the model, defaults to False
    :type use_value_head: bool, optional
    :param use_accelerator: Flag to indicate if the model should be created with the accelerator, defaults to False
    :type use_accelerator: bool, optional
    :return: The created model.
    :rtype: PreTrainedModel
    """
    # Start from the caller's config (if any) and fill in our SDPA + dtype
    # defaults with ``setdefault``, so any explicit caller value stays
    # authoritative.
    model_config = dict(model_config) if model_config else {}
    model_config.setdefault(
        "torch_dtype", torch.bfloat16 if not use_accelerator else torch.float16
    )
    # Auto-select the best available attention backend (flash_attention_2 when
    # the flash_attn package is installed, else sdpa). ``resolve_*`` treats an
    # explicit caller value (incl. "flex_attention") as authoritative.
    model_config["attn_implementation"] = resolve_attn_implementation(
        model_config.get("attn_implementation")
    )
    if model_config["attn_implementation"] == "flex_attention":
        patch_flex_attention_kernel_options()
    if add_value_head:
        return AutoModelForCausalLMWithValueHead.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            **model_config,
        )
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_config,
    )


