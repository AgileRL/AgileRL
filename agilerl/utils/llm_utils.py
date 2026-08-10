# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import gc
import logging
import random
import re
import shutil
import textwrap
import warnings
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator
from torch import nn

from agilerl import HAS_DEEPSPEED, HAS_LLM_DEPENDENCIES
from agilerl.typing import (
    JSONValue,
    PopulationType,
    PreferencePrompts,
    RolloutPrompt,
    SFTPrompts,
)

if TYPE_CHECKING:
    from accelerate.utils import DeepSpeedPlugin
    from peft import LoraConfig, PeftModel
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.tokenization_utils_base import (
        BatchEncoding,
        PreTrainedTokenizerBase,
    )

    from agilerl.algorithms.core.base import LLMAlgorithm
    from agilerl.utils.algo_utils import VLLMConfig
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

_DEPRECATED_LLM_ENV_NAMES = frozenset(("apply_chat_template",))

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


def __getattr__(name: str) -> Any:  # noqa: ANN401 -- lazy module re-export resolves attributes dynamically
    """Lazy re-exports from ``llm_envs`` with a deprecation warning."""
    if name in _DEPRECATED_LLM_ENV_NAMES:
        warnings.warn(
            (
                f"Importing {name} from agilerl.utils.llm_utils is deprecated; "
                "it has moved to agilerl.llm_envs. Import from "
                "agilerl.llm_envs instead; importing from "
                "agilerl.utils.llm_utils will be removed in a future release."
            ),
            FutureWarning,
            stacklevel=2,
        )
        import agilerl.llm_envs as _llm_envs

        return getattr(_llm_envs, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DEPRECATED_LLM_ENV_NAMES))


def validate_llm_context_lengths(
    max_model_len: int,
    max_output_tokens: int | None,
) -> None:
    """Reject configs that leave no prompt room for multi-turn rollouts.

    :param max_model_len: Total context length (prompt + completion ceiling).
    :type max_model_len: int
    :param max_output_tokens: Per-generation token cap; skipped when ``None``.
    :type max_output_tokens: int | None
    :raises ValueError: If ``max_output_tokens >= max_model_len``.
    """
    if max_output_tokens is None:
        return
    if max_output_tokens >= max_model_len:
        msg = (
            f"max_output_tokens ({max_output_tokens}) must be less than "
            f"max_model_len ({max_model_len}); equal or larger values leave no "
            "prompt budget for multi-turn rollouts "
            f"(max_prompt_tokens={max_prompt_tokens_for_model_len(max_model_len, max_output_tokens)})."
        )
        raise ValueError(msg)


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
    import importlib.util

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


def fill_outside_mask(
    values: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Replace the entries of ``values`` outside ``mask`` with a finite constant.

    Masked reductions weight by the mask, and IEEE ``nan * 0`` is ``nan``, so one
    non-finite entry at a padding position would otherwise poison the reduction.

    :param values: Tensor to fill, broadcastable with ``mask``.
    :type values: torch.Tensor
    :param mask: Mask whose ``True``/non-zero entries are kept.
    :type mask: torch.Tensor
    :param fill_value: Value written outside the mask, defaults to ``0.0``.
    :type fill_value: float, optional
    :return: ``values`` with the out-of-mask entries replaced.
    :rtype: torch.Tensor
    """
    keep = mask.to(device=values.device, dtype=torch.bool)
    if keep.shape != values.shape:
        keep = keep.expand_as(values)
    return values.masked_fill(~keep, fill_value)


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: int | None = None
) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(dim=axis) / mask.sum(dim=axis)
    return (values * mask).sum() / mask.sum()


def masked_var(
    values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum <= 1:
            msg = (
                "Unbiased masked variance requires at least 2 unmasked values; "
                "increase `mini_batch_size` or `gradient_accumulation_steps`."
            )
            raise ValueError(msg)
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(
    values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True
) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def pool_by_turns(
    token_values: torch.Tensor,
    turn_ids: torch.Tensor,
    num_turns: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Aggregate per-token values into per-turn scalars.

    :param token_values: [batch, seq_len] per-token scalars.
    :type token_values: torch.Tensor
    :param turn_ids: [batch, seq_len] turn index per token, -1 for non-action.
    :type turn_ids: torch.Tensor
    :param num_turns: Total number of turns (max turn_id + 1).
    :type num_turns: int
    :param reduction: ``"mean"`` (default) for mean-pooling,
        ``"sum"`` for sum-pooling (e.g. to aggregate log-ratios),
        ``"final_value"`` to select the last token value per turn.
    :type reduction: str
    :return: [batch, num_turns] aggregated values per turn.
    :rtype: torch.Tensor
    """
    batch_size, seq_len = token_values.shape
    turn_values = torch.zeros(batch_size, num_turns, device=token_values.device)
    token_positions = (
        torch.arange(seq_len, device=token_values.device)
        .unsqueeze(0)
        .expand_as(turn_ids)
    )
    for t in range(num_turns):
        mask_t = (turn_ids == t).float()
        summed = (token_values * mask_t).sum(dim=1)
        if reduction == "mean":
            count = mask_t.sum(dim=1).clamp(min=1)
            turn_values[:, t] = summed / count
        elif reduction == "sum":
            turn_values[:, t] = summed
        elif reduction == "final_value":
            masked_positions = torch.where(
                mask_t.bool(),
                token_positions,
                torch.full_like(token_positions, -1),
            )
            final_pos = masked_positions.max(dim=1).values
            has_turn = final_pos >= 0
            safe_final_pos = final_pos.clamp(min=0)
            final_vals = token_values[
                torch.arange(batch_size, device=token_values.device),
                safe_final_pos,
            ]
            turn_values[:, t] = torch.where(
                has_turn,
                final_vals,
                torch.zeros_like(final_vals),
            )
        else:
            msg = (
                f"Invalid reduction: {reduction}. Must be 'mean', 'sum', 'final_value'."
            )
            raise ValueError(msg)
    return turn_values


def baseline_free_turn_cells(turn_mask: torch.Tensor, group_size: int) -> torch.Tensor:
    """Mark played ``(sample, turn)`` cells whose group holds fewer than two members.

    :param turn_mask: Boolean ``[batch, num_turns]`` mask of played turns.
    :type turn_mask: torch.Tensor
    :param group_size: Members per group, in contiguous blocks along the batch.
    :type group_size: int
    :return: Boolean ``[batch, num_turns]`` mask of cells lacking a baseline.
    :rtype: torch.Tensor
    """
    batch, num_turns = turn_mask.shape
    count = turn_mask.reshape(-1, group_size, num_turns).sum(dim=1, keepdim=True)
    lone = (count <= 1).expand(-1, group_size, -1).reshape(batch, num_turns)
    return turn_mask & lone


def validate_importance_sampling_level(level: str, *, allow_auto: bool) -> None:
    """Raise ``ValueError`` unless ``level`` is a recognised IS pooling level.

    Valid levels are ``"token"``, ``"turn"`` and ``"trajectory"``; ``"auto"`` is
    additionally accepted when ``allow_auto`` is ``True``.

    :param level: The importance-sampling pooling level to validate.
    :type level: str
    :param allow_auto: Whether ``"auto"`` is an accepted value.
    :type allow_auto: bool
    :raises ValueError: If ``level`` is not in the valid set.
    """
    valid = {"token", "turn", "trajectory"}
    if allow_auto:
        valid.add("auto")
    if level not in valid:
        msg = (
            f"importance_sampling_level must be one of {sorted(valid)}, got {level!r}."
        )
        raise ValueError(msg)


def pool_log_ratio_by_level(
    token_log_ratio: torch.Tensor,
    action_mask: torch.Tensor,
    turn_ids: torch.Tensor | None,
    level: str,
    num_turns: int | None = None,
    turn_reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool per-token log-ratios to a token/turn/trajectory importance-sampling unit.

    Returns the per-unit pooled log-ratio (the log importance weights) and the
    matching unit mask, mirroring the pooling used by :func:`clipped_is_surrogate`:

    * ``"token"``    — identity: per-token log-ratio, ``action_mask`` as units.
    * ``"turn"``     — per-turn pooling of token log-ratios (``turn_ids``
      required); ``turn_reduction="mean"`` gives a length-normalized mean
      (geometric-mean ratio), ``"sum"`` gives a sum (product ratio). A turn is
      active when any of its tokens appear.
    * ``"trajectory"`` — length-normalized mean over the completion's action
      tokens; the trajectory is active when it has at least one action token.

    Operates only on ``(B, T)`` tensors (no vocab axis), so it is memory-bounded.

    :param token_log_ratio: ``(B, T)`` per-token ``log pi - log pi_old``.
    :type token_log_ratio: torch.Tensor
    :param action_mask: ``(B, T)`` action-token mask.
    :type action_mask: torch.Tensor
    :param turn_ids: ``(B, T)`` turn index per token (``-1`` non-action);
        required for the turn level.
    :type turn_ids: torch.Tensor | None
    :param level: ``"token"`` / ``"turn"`` / ``"trajectory"``.
    :type level: str
    :param num_turns: Number of turns for the turn level; inferred from
        ``turn_ids`` when ``None``.
    :type num_turns: int | None
    :param turn_reduction: Turn-level pooling reduction for ``level="turn"``,
        one of ``"mean"`` or ``"sum"``.
    :type turn_reduction: str
    :return: ``(log_importance_weights, unit_mask)`` at the requested level.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: If ``level`` is unknown or turn ids are missing.
    """
    if level == "token":
        return token_log_ratio, action_mask
    if level == "turn":
        if turn_ids is None:
            msg = "turn-level surrogate requires turn_ids."
            raise ValueError(msg)
        if turn_reduction not in {"mean", "sum"}:
            msg = (
                "turn_reduction must be one of ['mean', 'sum'], got "
                f"{turn_reduction!r}."
            )
            raise ValueError(msg)
        if num_turns is None:
            num_turns = int(turn_ids.max().item()) + 1
        # Pool only over action tokens, exactly like the trajectory branch: a
        # non-action token (``action_mask == 0``) belongs to no turn, so the
        # per-turn mean stays mask-aware and a single full-completion turn
        # collapses to the trajectory-level mean. Callers that already mark
        # non-action tokens with ``turn_ids == -1`` are unaffected (no-op).
        effective_turn_ids = torch.where(
            action_mask.to(torch.bool),
            turn_ids,
            torch.full_like(turn_ids, -1),
        )
        log_importance_weights = pool_by_turns(
            token_log_ratio, effective_turn_ids, num_turns, reduction=turn_reduction
        )
        # Per-turn action-token counts via the same pooling; a turn is active
        # iff it owns at least one action token.
        turn_token_counts = pool_by_turns(
            torch.ones_like(token_log_ratio),
            effective_turn_ids,
            num_turns,
            reduction="sum",
        )
        unit_mask = (turn_token_counts > 0).to(log_importance_weights.dtype)
        return log_importance_weights, unit_mask
    if level == "trajectory":
        mask_f = action_mask.to(token_log_ratio.dtype)
        count = mask_f.sum(dim=-1, keepdim=True)
        log_importance_weights = (token_log_ratio * mask_f).sum(
            dim=-1, keepdim=True
        ) / count.clamp(min=1.0)
        unit_mask = (count > 0).to(token_log_ratio.dtype)
        return log_importance_weights, unit_mask
    msg = (
        f"Unknown importance_sampling_level '{level}'. "
        "Expected one of ['token', 'turn', 'trajectory']."
    )
    raise ValueError(msg)


def clipped_min_surrogate(
    log_importance_weights: torch.Tensor,
    advantages: torch.Tensor,
    unit_mask: torch.Tensor,
    clip_min: float,
    clip_max: float,
    loss_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clipped PPO min-surrogate from per-unit log importance weights.

    Exponentiates the log importance weights, clamps the ratio to
    ``[clip_min, clip_max]``, and reduces the masked pessimistic (min)
    surrogate over the active units.

    :param log_importance_weights: Per-unit pooled log-ratio.
    :type log_importance_weights: torch.Tensor
    :param advantages: Per-unit advantages (same shape).
    :type advantages: torch.Tensor
    :param unit_mask: Mask selecting active units (same shape).
    :type unit_mask: torch.Tensor
    :param clip_min: Lower clip bound for the ratio (e.g. ``1 - clip_coef``).
    :type clip_min: float
    :param clip_max: Upper clip bound for the ratio (e.g. ``1 + clip_coef``).
    :type clip_max: float
    :param loss_weight: Optional per-unit, **detached**, non-negative weight that
        multiplies each unit's surrogate before the masked mean — e.g. the vLLM
        sampling-mismatch (truncated-IS) correction ratio. ``None`` leaves the
        surrogate unweighted. Since the weight is ``>= 0`` it commutes with the
        ``max`` (pessimistic) reduction, so it is a faithful per-unit reweight.
    :type loss_weight: torch.Tensor | None
    :return: ``(pg_loss, clipfrac)`` scalars.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    ratio = torch.exp(log_importance_weights)
    clipped_ratio = torch.clamp(ratio, clip_min, clip_max)
    clipfrac = masked_mean((ratio != clipped_ratio).float(), unit_mask)
    surrogate = torch.max(-advantages * ratio, -advantages * clipped_ratio)
    if loss_weight is not None:
        surrogate = surrogate * loss_weight
    pg_loss = masked_mean(surrogate, unit_mask)
    return pg_loss, clipfrac


def clipped_is_surrogate(
    token_log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    turn_ids: torch.Tensor | None,
    importance_sampling_level: str,
    clip_coef: float,
    loss_weight: torch.Tensor | None = None,
    turn_reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clipped PPO-style policy surrogate at a token/turn/trajectory IS level.

    The importance ratio (and the advantage paired with it) is pooled to
    ``importance_sampling_level``:

    * ``"token"``    — per-token ratio, clip per token.
    * ``"turn"``     — per-turn pooling of token log-ratios, clip per turn
      (``turn_ids`` required). ``turn_reduction="mean"`` uses a length-
      normalized mean (geometric-mean ratio), ``"sum"`` uses a sum (product
      ratio).
    * ``"trajectory"`` — length-normalized mean over the whole completion, clip
      per trajectory.

    Thin composition of :func:`pool_log_ratio_by_level` (ratio + advantage
    pooling) and :func:`clipped_min_surrogate` (clip + min-surrogate). Shared by
    the non-Liger PPO and REINFORCE paths. Operates only on ``(B, T)`` tensors
    (no vocab axis), so it is memory-bounded.

    :param token_log_ratio: ``(B, T)`` per-token ``log pi - log pi_old``.
    :type token_log_ratio: torch.Tensor
    :param advantages: ``(B, T)`` per-token advantages.
    :type advantages: torch.Tensor
    :param action_mask: ``(B, T)`` action-token mask.
    :type action_mask: torch.Tensor
    :param turn_ids: ``(B, T)`` turn index per token (``-1`` non-action);
        required for turn level.
    :type turn_ids: torch.Tensor | None
    :param importance_sampling_level: ``"token"`` / ``"turn"`` / ``"trajectory"``.
    :type importance_sampling_level: str
    :param clip_coef: Symmetric clip coefficient (clip to ``[1-c, 1+c]``).
    :type clip_coef: float
    :param turn_reduction: Turn-level pooling reduction when
        ``importance_sampling_level="turn"``, one of ``"mean"`` or ``"sum"``.
    :type turn_reduction: str
    :return: ``(pg_loss, clipfrac)`` scalars.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    num_turns = (
        int(turn_ids.max().item()) + 1
        if importance_sampling_level == "turn" and turn_ids is not None
        else None
    )
    log_importance_weights, unit_mask = pool_log_ratio_by_level(
        token_log_ratio,
        action_mask,
        turn_ids,
        importance_sampling_level,
        num_turns,
        turn_reduction=turn_reduction,
    )
    adv, _ = pool_log_ratio_by_level(
        advantages,
        action_mask,
        turn_ids,
        importance_sampling_level,
        num_turns,
        turn_reduction="mean",
    )
    pooled_loss_weight = None
    if loss_weight is not None:
        pooled_loss_weight, _ = pool_log_ratio_by_level(
            loss_weight,
            action_mask,
            turn_ids,
            importance_sampling_level,
            num_turns,
            turn_reduction="mean",
        )
    return clipped_min_surrogate(
        log_importance_weights,
        adv,
        unit_mask,
        1 - clip_coef,
        1 + clip_coef,
        loss_weight=pooled_loss_weight,
    )


def create_llm_accelerator(
    *,
    deepspeed_plugin: DeepSpeedPlugin | None = None,
) -> Accelerator | None:
    """Create an :class:`Accelerator` for LLM training with DeepSpeed.

    This helper enforces a strict DeepSpeed contract for LLM workloads:

    * **0 GPUs** — returns ``None`` (the ``accelerator=None`` code-path
      in :class:`~agilerl.algorithms.core.base.LLMAlgorithm` handles
      CPU-only training).
    * When ``deepspeed_plugin`` is provided, returns an
      ``Accelerator(deepspeed_plugin=...)``.
    * Otherwise, instantiates ``Accelerator()`` and requires that a
      DeepSpeed plugin is already present (for example via
      ``accelerate config`` + ``accelerate launch``).
      If no plugin is detected, raises ``RuntimeError`` with setup
      instructions.

    :param deepspeed_plugin: Explicit DeepSpeed plugin instance. If
        omitted, this function expects a launch-configured plugin to be
        present in ``Accelerator.state``.
    :type deepspeed_plugin: DeepSpeedPlugin | None
    :return: A configured ``Accelerator``, or ``None`` when no GPU is
        available.
    """
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        logger.info("No GPUs detected — returning None (CPU-only path).")
        return None

    if deepspeed_plugin is not None:
        return Accelerator(deepspeed_plugin=deepspeed_plugin)

    accelerator = Accelerator()
    if accelerator.state.deepspeed_plugin is None:
        msg = (
            "DeepSpeed is required for create_llm_accelerator(), but no "
            "DeepSpeed plugin was detected. Use one of: "
            "(1) run `accelerate config` and launch with `accelerate launch ...`; "
            "(2) pass `deepspeed_plugin=` explicitly to create_llm_accelerator()."
        )
        raise RuntimeError(msg)
    return accelerator


def get_llm_accelerator(
    base_accelerator: Accelerator | None,
    idx: int,
) -> Accelerator | None:
    """Return a per-agent accelerator from a base accelerator.

    ``idx == 0`` reuses ``base_accelerator``. For additional agents this helper
    creates a fresh ``Accelerator`` instance so each LLM algorithm owns an
    independent accelerator/engine reference.

    :param base_accelerator: Accelerator passed into population creation.
    :type base_accelerator: Accelerator | None
    :param idx: Agent index in the population.
    :type idx: int
    :return: Accelerator for the specific agent, or ``None``.
    :rtype: Accelerator | None
    """
    if idx < 0:
        msg = f"Population index must be non-negative, got {idx}."
        raise ValueError(msg)

    if base_accelerator is None:
        return None

    if idx == 0:
        return base_accelerator

    return Accelerator()


def cuda_tensor_bytes_in_module(module: torch.nn.Module) -> int:
    """Sum nbytes of parameters and buffers still on a CUDA device."""
    total = 0
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.is_cuda:
            total += tensor.numel() * tensor.element_size()
    return total


def collect_trainable_param_stats(pop: PopulationType) -> dict[str, Any]:
    """Best-effort LoRA / trainable-param accounting for a population's first agent.

    Recorded once at init so runs can be correlated against LoRA size.
    Wrapped in a broad except: introspecting peft-wrapped accelerator-managed
    models can fail in odd ways, and a logging-only field shouldn't fault training.

    :param pop: Population of LLM algorithms; only ``pop[0]`` is inspected.
    :type pop: PopulationType
    :return: ``trainable_params`` / ``total_params`` / ``trainable_param_ratio``
        fields, or an empty dict when the actor is missing or introspection fails.
    :rtype: dict[str, Any]
    """
    out: dict[str, Any] = {}
    try:
        agent = pop[0]
        actor = getattr(agent, "actor", None)
        if actor is None:
            return out
        # Unwrap accelerate / peft / DDP layers if present.
        inner = actor
        for attr in ("module", "model"):
            unwrapped = getattr(inner, attr, None)
            if unwrapped is not None:
                inner = unwrapped
        total = 0
        trainable = 0
        for param in inner.parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
        if total > 0:
            out["trainable_params"] = trainable
            out["total_params"] = total
            out["trainable_param_ratio"] = trainable / total
    except Exception:  # pragma: no cover — best-effort
        pass
    return out


def resolve_llm_device(
    accelerator: Accelerator | None,
    device: str | torch.device | None = None,
) -> str:
    """Resolve the training device for an LLM algorithm.

    The accelerator outranks *device*: under ``accelerate``/DeepSpeed each rank
    must own its own GPU, so a caller passing a bare ``"cuda"`` cannot be allowed
    to collapse every rank onto device 0.

    :param accelerator: Accelerator object, or ``None`` for single-process runs.
    :type accelerator: accelerate.Accelerator | None
    :param device: Caller-requested device, or ``None`` to auto-detect.
    :type device: str | torch.device | None
    :return: The rank's device under an accelerator, else *device*, else the best
        locally available device.
    :rtype: str
    """
    if accelerator is not None:
        return f"cuda:{accelerator.process_index}"
    if device is not None:
        return str(device)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def offload_colocated_trainer_from_gpu(unwrapped_model: torch.nn.Module) -> int:
    """Force the trainer module tree onto CPU before colocated vLLM ``LLM()`` init.

    Unlike :func:`move_params_to_cpu`, always calls ``.to("cpu")`` even when the
    first parameter already reports CPU (bitsandbytes / ``device_map="cpu"`` can
    leave most weights on GPU). Returns remaining CUDA tensor bytes afterward.
    """
    unwrapped_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    return cuda_tensor_bytes_in_module(unwrapped_model)


def move_params_to_gpu(unwrapped_model: torch.nn.Module, device: torch.device) -> None:
    """Move params to GPU.

    :param agent: Distributed agent
    :type agent: DistributedLLMAgent
    :return: None
    :rtype: None
    """
    if next(unwrapped_model.parameters()).device != device:
        unwrapped_model.to(device, non_blocking=True)
        torch.cuda.synchronize()


def move_params_to_cpu(unwrapped_model: torch.nn.Module) -> bool:
    """Move params to CPU, returning True when a device transfer was needed."""
    if next(unwrapped_model.parameters()).device.type != "cpu":
        unwrapped_model.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True
    return False


def attention_mask_from_padded_ids(
    ids: torch.Tensor,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Boolean attention mask covering everything but the trailing padding run.

    ``ids != pad`` would also hole out pad-id tokens that are real transcript
    content — with pad == eos (every pad-less model family) that is each turn
    terminator the sampler attended to, shifting positions and attention at
    learn time. Only the right-side padding added when stacking rows to a
    rectangle is non-content, so only the trailing run is masked.

    :param ids: Token tensor of shape ``(B, T)``, right-padded per row.
    :type ids: torch.Tensor
    :param pad_token_id: Pad token id; ``None`` means nothing is padding.
    :type pad_token_id: int | None
    :return: Boolean mask of shape ``(B, T)``.
    :rtype: torch.Tensor
    """
    if pad_token_id is None:
        return torch.ones_like(ids, dtype=torch.bool)
    is_pad = (ids == pad_token_id).to(torch.int64)
    trailing = is_pad.flip(-1).cumprod(-1).flip(-1)
    return trailing == 0


def build_completion_mask(
    token_ids: torch.Tensor,
    prompt_len: int | None,
    pad_token_id: int,
) -> torch.Tensor:
    """Build the boolean action mask marking the completion within a token sequence.

    Returns ``True`` at positions that are (a) past the prompt and (b) not
    pad tokens, dropping the leading position to align with the
    next-token-prediction shift used downstream.

    :param token_ids: Token tensor of shape ``(B, seq_len)`` containing
        the prompt followed by generated tokens.
    :type token_ids: torch.Tensor
    :param prompt_len: Number of leading tokens to mask out (the full
        prompt length). ``None`` means "no prompt prefix" — every non-pad
        token is part of the completion, so the entire dim is set before
        pads are zeroed back out.
    :type prompt_len: int | None
    :param pad_token_id: Pad token id used to suppress padding positions.
    :type pad_token_id: int
    :return: Boolean mask of shape ``(B, seq_len - 1)``.
    :rtype: torch.Tensor
    """
    non_pad = token_ids != pad_token_id
    if prompt_len is None or prompt_len == 0:
        mask = non_pad
    else:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        mask = (positions.unsqueeze(0) >= prompt_len) & non_pad
    return mask[:, 1:]


def prepare_prompt_hf_generate(
    prompt: RolloutPrompt, device: torch.device
) -> dict[str, torch.Tensor]:
    """Turn one rollout prompt into HuggingFace ``generate`` inputs.

    ``attention_mask`` is taken from the prompt when present (e.g. a padded
    batch) and derived as all-ones otherwise (a single unpadded row, the
    ``RolloutHarness`` prompt shape).

    :param prompt: The prompt to prepare.
    :type prompt: RolloutPrompt
    :param device: The device to move the tensors to.
    :type device: torch.device
    :return: ``input_ids`` / ``attention_mask`` moved to ``device``.
    :rtype: dict[str, torch.Tensor]
    """
    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt.get("attention_mask")
    return {
        "input_ids": input_ids,
        "attention_mask": (
            torch.ones_like(input_ids)
            if attention_mask is None
            else attention_mask.to(device)
        ),
    }


def get_model_name_or_path(model: PreTrainedModel) -> str:
    """Get the name or path of a model.

    :param model: The model to get the name or path of.
    :type model: PreTrainedModel
    :return: The name or path of the model.
    :rtype: str
    """
    if hasattr(model, "name_or_path"):
        # transformers types ``name_or_path`` as ``str | None``; it is always set
        # on a loaded model, but coerce to keep the return a plain ``str``.
        return str(model.name_or_path)

    if hasattr(model, "pretrained_model") and hasattr(
        model.pretrained_model, "name_or_path"
    ):
        return model.pretrained_model.name_or_path

    if hasattr(model, "base_model") and hasattr(model.base_model, "name_or_path"):
        return model.base_model.name_or_path

    if hasattr(model, "base_model") and hasattr(model.base_model, "pretrained_model"):
        return model.base_model.pretrained_model.name_or_path

    msg = "Model name or path not found."
    raise ValueError(msg)


ACTIVATION_CHECKPOINTING_KEY = "activation_checkpointing"


def _named_activation_checkpointing_keys(block: Any) -> str:  # noqa: ANN401 -- DeepSpeed config values are untyped
    """Describe the settings a rejected activation-checkpointing section carries.

    :param block: Value found under the activation-checkpointing key.
    :type block: Any
    :return: Human-readable list of the keys, or a description of the value.
    :rtype: str
    """
    if isinstance(block, Mapping):
        if not block:
            return "no keys"
        return ", ".join(sorted(str(key) for key in block))
    return repr(block)


def assert_no_activation_checkpointing_config(
    deepspeed_config: Any,  # noqa: ANN401 -- DeepSpeed configs arrive as opaque mappings
    *,
    source: str,
) -> None:
    """Reject a DeepSpeed config carrying an activation-checkpointing section.

    :param deepspeed_config: DeepSpeed config mapping to inspect.
    :type deepspeed_config: Any
    :param source: Where the config came from, quoted back in the message.
    :type source: str
    :return: None
    :rtype: None
    :raises TypeError: If the config is not a mapping and so cannot be checked.
    :raises RuntimeError: If the activation-checkpointing section is present.
    """
    if deepspeed_config is None:
        return
    if not isinstance(deepspeed_config, Mapping):
        msg = (
            f"DeepSpeed config from {source} is {type(deepspeed_config).__name__}, "
            f"not a mapping, so its {ACTIVATION_CHECKPOINTING_KEY} section cannot "
            "be checked."
        )
        raise TypeError(msg)
    if ACTIVATION_CHECKPOINTING_KEY not in deepspeed_config:
        return
    block = deepspeed_config[ACTIVATION_CHECKPOINTING_KEY]
    msg = (
        f"DeepSpeed config from {source} sets {ACTIVATION_CHECKPOINTING_KEY} "
        f"({_named_activation_checkpointing_keys(block)}), which is not honoured "
        "on this training path. DeepSpeed only reads that section for checkpoints "
        "routed through deepspeed.checkpointing.checkpoint. Gradient checkpointing "
        "here is enabled through HuggingFace's gradient_checkpointing_enable, which "
        "binds torch.utils.checkpoint.checkpoint, and that never consults the "
        "DeepSpeed config. Routing it through DeepSpeed instead is not available "
        "on this stack: partition_activations shards along the model-parallel "
        "group, which is size 1 here, so it saves nothing; "
        "contiguous_memory_optimization requires partition_activations plus a "
        "fixed layer count; and deepspeed.checkpointing.checkpoint is a reentrant "
        "autograd Function with no use_reentrant argument, which the LoRA plus "
        "ZeRO-3 recompute path needs to control. Remove the section rather than "
        "carry settings that do nothing."
    )
    raise RuntimeError(msg)


def align_deepspeed_lr(lr: float, accelerator: Accelerator | None) -> float:
    """Align the learning rate for DeepSpeed.

    :param lr: The learning rate to align.
    :type lr: float
    :param accelerator: The accelerator to align the learning rate for.
    :type accelerator: Accelerator | None
    :return: The aligned learning rate.
    :rtype: float
    """
    if accelerator is not None:
        optim_lr = (
            accelerator.state.deepspeed_plugin.deepspeed_config.get("optimizer", {})
            .get("params", {})
            .get("lr", None)
        )
        if optim_lr is not None and optim_lr != lr:
            warnings.warn(
                f"DeepSpeed learning rate is set to {optim_lr} but the argument 'lr' is set to {lr}. "
                "Overwriting deepspeed learning rate with the argument 'lr'.",
                stacklevel=2,
            )
            accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"][
                "lr"
            ] = lr
    return lr


def sample_eval_prompts(
    env: Any,  # noqa: ANN401 -- gym env duck-typed across SFT/Preference/other gyms
    n: int = 5,
    seed: int = 0,
) -> list[tuple[str, str | None, str | None]]:
    """Randomly sample *n* ``(prompt, chosen, rejected)`` triples from
    *env*'s held-out test dataset.

    Columns are resolved automatically per dataset ``objective``:

    * ``objective="sft"`` — ``chosen`` is ``env.response_column``; ``rejected``
      is ``None`` (SFT has no negative example).
    * ``objective="preference"`` — ``chosen`` and ``rejected`` map to the
      dataset's ``"chosen"`` / ``"rejected"`` columns.
    * Any other env — both are ``None``.

    :param env: AgileRL gym environment with a ``test_dataloader`` attribute.
    :type env: Any
    :param n: Number of samples to draw, defaults to 5.
    :type n: int, optional
    :param seed: Random seed for reproducible sampling, defaults to 0.
    :type seed: int, optional
    :return: List of ``(prompt, chosen, rejected)`` tuples; unused fields are
        ``None``.
    :rtype: list[tuple[str, str | None, str | None]]
    """
    dataset = env.test_dataloader.dataset
    indices = random.Random(seed).sample(range(len(dataset)), min(n, len(dataset)))

    chosen_col: str | None = None
    rejected_col: str | None = None
    objective = getattr(env, "objective", None)
    if objective == "sft":
        chosen_col = env.response_column
    elif objective == "preference":
        chosen_col = "chosen"
        rejected_col = "rejected"

    return [
        (
            dataset[i]["prompt"],
            dataset[i][chosen_col] if chosen_col else None,
            dataset[i][rejected_col] if rejected_col else None,
        )
        for i in indices
    ]


def compare_responses(
    agent: LLMAlgorithm,
    tokenizer: Any,  # noqa: ANN401 -- HF tokenizer; typed decode() returns str|list[str], which this single-sequence path would have to narrow
    samples: list[tuple[str, str | None, str | None]],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    do_sample: bool = False,
    skip_special_tokens: bool = True,
    show_base_model: bool = True,
) -> None:
    """Run each prompt through the base model and the fine-tuned LoRA model,
    printing a formatted comparison to the terminal one sample at a time.

    After each sample the user is prompted to press **Enter** to continue or
    **q + Enter** to quit early.  Intended to be called at the end of a
    training script for a quick qualitative sanity-check.

    Works with any LoRA-adapted
    :class:`~agilerl.algorithms.core.base.LLMAlgorithm` (``SFT``, ``DPO``,
    …).  When the model has no LoRA adapter the base-model column is omitted
    and only the current model's output is shown.

    :param agent: Trained AgileRL LLM agent exposing ``agent.actor`` and
        ``agent.device``.
    :type agent: LLMAlgorithm
    :param tokenizer: HuggingFace tokenizer matching the model.
    :type tokenizer: Any
    :param samples: ``(prompt, chosen, rejected)`` triples as returned by
        :func:`sample_eval_prompts`.  ``None`` fields are silently skipped.
    :type samples: list[tuple[str, str | None, str | None]]
    :param max_new_tokens: Maximum tokens to generate per response, defaults
        to 200.
    :type max_new_tokens: int, optional
    :param temperature: Sampling temperature, defaults to 1.0.
    :type temperature: float, optional
    :param do_sample: Use sampling instead of greedy decoding, defaults to
        False.  Set ``True`` together with a ``temperature`` != 1.0 for
        stochastic outputs.
    :type do_sample: bool, optional
    :param skip_special_tokens: Strip special tokens when decoding, defaults
        to True.
    :type skip_special_tokens: bool, optional
    :param show_base_model: If ``False``, skip the base-model generation block
        (only the current model output is printed).  Useful when the adapter is
        merged or base vs. adapter outputs are identical.
    :type show_base_model: bool, optional
    """
    model = agent.actor
    device = agent.device
    width = min(shutil.get_terminal_size(fallback=(100, 40)).columns, 120)
    divider = "─" * width
    has_adapter = hasattr(model, "disable_adapter")

    def _generate(prompt_text: str, *, use_base: bool) -> str:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs: dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
        model.eval()
        with torch.no_grad():
            if use_base and has_adapter:
                with model.disable_adapter():
                    output_ids = model.generate(**gen_kwargs)
            else:
                output_ids = model.generate(**gen_kwargs)
        new_tokens = output_ids[0][prompt_len:]
        return tokenizer.decode(
            new_tokens, skip_special_tokens=skip_special_tokens
        ).strip()

    def _wrap(text: str, indent: int = 2) -> str:
        prefix = " " * indent
        return textwrap.fill(
            text,
            width=width - indent,
            initial_indent=prefix,
            subsequent_indent=prefix,
        )

    total = len(samples)
    for i, (prompt, chosen, rejected) in enumerate(samples, 1):
        header = f"  SAMPLE {i} / {total}  "
        padding = max(0, width - len(header))
        left_pad = padding // 2
        right_pad = padding - left_pad
        print(f"\n{'═' * left_pad}{header}{'═' * right_pad}")

        print(f"\nPROMPT\n{divider}")
        print(_wrap(prompt))

        if chosen is not None:
            print(f"\nDATASET RESPONSE (CHOSEN)\n{divider}")
            print(_wrap(chosen))

        if rejected is not None:
            print(f"\nDATASET RESPONSE (REJECTED)\n{divider}")
            print(_wrap(rejected))

        if has_adapter and show_base_model:
            print(f"\nBASE MODEL\n{divider}")
            print(_wrap(_generate(prompt, use_base=True)))

        label = "FINE-TUNED MODEL" if has_adapter else "MODEL RESPONSE"
        print(f"\n{label}\n{divider}")
        print(_wrap(_generate(prompt, use_base=False)))

        if i < total:
            nav = "  [Enter] next sample   [q + Enter] quit  "
            nav_padding = max(0, width - len(nav))
            print(
                f"\n{'─' * (nav_padding // 2)}{nav}{'─' * (nav_padding - nav_padding // 2)}"
            )
            try:
                if input().strip().lower() == "q":
                    break
            except EOFError:
                break

    print(f"\n{'═' * width}\n")


def calculate_k3_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """K3 estimator of ``KL[q || p]`` (Schulman 2020).

    ``exp(log_p - log_q) - (log_p - log_q) - 1`` — always-positive,
    lower-variance than the naive ``log_p - log_q`` estimator.
    """
    diff = log_p - log_q
    return torch.exp(diff) - diff - 1.0


# ---------------------------------------------------------------------------
# vLLM / CUDA capability helpers (benchmark CLI, colocated rollout)
# ---------------------------------------------------------------------------


def resolve_vllm_max_lora_rank(
    vllm_max_lora_rank: int,
    lora_rank: int | None,
) -> int:
    """Ensure vLLM ``max_lora_rank`` is at least the trainer LoRA rank."""
    trainer_rank = int(lora_rank or 0)
    return max(int(vllm_max_lora_rank), trainer_rank)


def resolve_vllm_max_num_batched_tokens(
    max_num_seqs: int,
    max_model_len: int,
    explicit: int | None = None,
) -> int:
    """Resolve vLLM ``max_num_batched_tokens`` for colocated rollout.

    Do not use ``max_num_seqs * max_model_len`` (e.g. 8x32768 = 262144): that
    drives torch.compile inductor benchmark tensors of ~5+ GiB during ``LLM()``
    init and multimodal encoder-cache budgets, which OOMs on a 40GB GPU even
    when the trainer is offloaded. Default caps prefill batching while keeping
    at least one full ``max_model_len`` context for chunked prefill.
    """
    if explicit is not None:
        return int(explicit)
    worst_case = max_num_seqs * max_model_len
    # Allow concurrent prefills up to 8k tokens per slot unless that exceeds one
    # max-length context (then cap at max_model_len).
    concurrent_budget = max(max_model_len, max_num_seqs * 8192)
    return min(worst_case, concurrent_budget)


def build_vllm_llm_init_kwargs(
    vllm_config: VLLMConfig,
    *,
    trainer_model_name_or_path: str,
    max_model_len: int,
    process_index: int = 0,
    lora_rank: int | None = None,
) -> dict[str, Any]:
    """Build kwargs for ``vllm.LLM`` from :class:`~agilerl.utils.algo_utils.VLLMConfig`."""
    vllm_model = (
        vllm_config.vllm_model_name_or_path
        if vllm_config.vllm_model_name_or_path is not None
        else trainer_model_name_or_path
    )
    kwargs: dict[str, Any] = {
        "model": vllm_model,
        "tensor_parallel_size": vllm_config.tensor_parallel_size,
        "gpu_memory_utilization": vllm_config.gpu_memory_utilization,
        "max_num_seqs": vllm_config.max_num_seqs,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "external_launcher",
        "seed": process_index // vllm_config.tensor_parallel_size,
        "max_num_batched_tokens": resolve_vllm_max_num_batched_tokens(
            vllm_config.max_num_seqs,
            max_model_len,
            getattr(vllm_config, "max_num_batched_tokens", None),
        ),
        "model_impl": "vllm",
        "enable_sleep_mode": vllm_config.sleep_mode,
    }
    if vllm_config.dtype is not None:
        kwargs["dtype"] = vllm_config.dtype
    if vllm_config.quantization is not None:
        kwargs["quantization"] = vllm_config.quantization
    if vllm_config.kv_cache_dtype is not None:
        kwargs["kv_cache_dtype"] = vllm_config.kv_cache_dtype
    if getattr(vllm_config, "kv_cache_memory_bytes", None) is not None:
        kwargs["kv_cache_memory_bytes"] = vllm_config.kv_cache_memory_bytes
    if getattr(vllm_config, "enforce_eager", None) is not None:
        # Force vLLM to skip CUDA-graph capture. Saves the ~2 GiB CUDA-graph
        # private pool (useful for colocated trainer/rollout setups with a
        # tight GPU budget) at the cost of slightly slower per-step decode.
        kwargs["enforce_eager"] = vllm_config.enforce_eager
    # Colocated vLLM always serves LoRA: the trainer shares vLLM's base and syncs
    # only the adapter per rollout.
    kwargs["enable_lora"] = True
    kwargs["max_lora_rank"] = resolve_vllm_max_lora_rank(
        vllm_config.max_lora_rank,
        lora_rank,
    )
    kwargs["max_loras"] = vllm_config.max_loras
    return kwargs


def build_vllm_rollout_lora_request(
    lora_path: str | Path,
    *,
    load_inplace: bool = False,
    lora_name: str = "actor",
    lora_int_id: int = 1,
) -> Any:  # noqa: ANN401 -- returns vllm.LoRARequest (optional Linux-only dependency)
    """Build a vLLM :class:`~vllm.lora.request.LoRARequest` for rollout."""
    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name=lora_name,
        lora_int_id=lora_int_id,
        lora_path=str(lora_path),
        load_inplace=load_inplace,
    )


def peft_lora_state_dict_key_to_module_key(key: str) -> str:
    """Strip PEFT LoRA weight suffixes so ``target_modules`` matching applies."""
    for marker in (
        ".lora_A.",
        ".lora_B.",
        ".lora_embedding_A.",
        ".lora_embedding_B.",
    ):
        if marker in key:
            return key.split(marker, maxsplit=1)[0]
    return key


def remap_peft_lora_key_for_vllm(key: str) -> str:
    """Normalize PEFT keys (e.g. ClippableLinear ``.linear.lora_A``) for vLLM."""
    key = key.replace(".linear.lora_A.", ".lora_A.").replace(
        ".linear.lora_B.", ".lora_B."
    )
    return key.replace(".base_layer.", ".")


def expert_lora_vllm_key_map(peft_model: nn.Module) -> dict[str, str]:
    """Map packed-experts LoRA module keys to the paths vLLM's fused-MoE loader reads (down on ``<experts>``, gate/up on ``<experts>.base_layer``)."""
    from peft.tuners.lora.layer import ParamWrapper

    key_map: dict[str, str] = {}
    unmapped: list[str] = []
    for name, module in peft_model.named_modules():
        if not isinstance(module, ParamWrapper):
            continue
        experts_path = name
        while experts_path.endswith(".base_layer"):
            experts_path = experts_path.removesuffix(".base_layer")
        parameter_name = module.parameter_name
        parent, _, leaf = experts_path.rpartition(".")
        sibling_prefix = f"{parent}." if parent else ""
        if parameter_name in ("gate_up_proj", "up_proj"):
            key_map[name] = f"{experts_path}.base_layer"
        elif parameter_name == "down_proj":
            key_map[name] = experts_path
        elif parameter_name == "weight" and leaf == "input_linear":
            key_map[name] = f"{sibling_prefix}experts.base_layer"
        elif parameter_name == "weight" and leaf == "output_linear":
            key_map[name] = f"{sibling_prefix}experts"
        else:
            unmapped.append(f"{name} ({parameter_name})")
    if unmapped:
        msg = (
            "No vLLM fused-MoE LoRA mapping for wrapped parameters "
            f"{unmapped}; rollout would silently diverge from the trainer "
            "policy. Restrict target_parameters to supported experts modules."
        )
        raise ValueError(msg)
    return key_map


def filter_peft_state_dict_for_vllm_lora(
    state_dict: dict[str, torch.Tensor],
    target_modules: str | list[str] | None,
    *,
    expert_key_map: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Keep LoRA tensors whose modules match the trainer ``target_modules`` spec or expert map."""
    filtered: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        module_key = peft_lora_state_dict_key_to_module_key(key)
        if expert_key_map is not None and module_key in expert_key_map:
            suffix = key.removeprefix(module_key)
            filtered[f"{expert_key_map[module_key]}{suffix}"] = tensor
            continue
        if target_modules is None or not peft_target_key_matches(
            module_key, target_modules
        ):
            continue
        filtered[remap_peft_lora_key_for_vllm(key)] = tensor
    return filtered


def _json_safe_value(obj: object) -> JSONValue:
    """Recursively convert PEFT config values to JSON-serializable types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, set):
        return [_json_safe_value(v) for v in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple)):
        return [_json_safe_value(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe_value(v) for k, v in obj.items()}
    return str(obj)


def save_peft_adapter_for_vllm_rollout(
    peft_model: PeftModel,
    staging_dir: Path | str,
    adapter_name: str,
    *,
    target_modules: str | list[str] | None,
    expert_key_map: dict[str, str] | None = None,
) -> Path:
    """Export a PEFT adapter checkpoint that vLLM can load for colocated rollout.

    Keeps only tensors that match the same ``target_modules`` spec used for PEFT
    training (from :func:`adapt_lora_config_for_model`) or the packed-experts
    ``expert_key_map`` (from :func:`expert_lora_vllm_key_map`). Rewrites
    ClippableLinear ``.linear`` suffixes in keys for vLLM. ``staging_dir`` must
    be process-private (AgileRL stages per-rank when distributed): every caller
    writes the adapter files.
    """
    if not HAS_LLM_DEPENDENCIES:
        msg = "save_peft_adapter_for_vllm_rollout requires peft and transformers."
        raise ImportError(msg)

    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    adapter_path = Path(staging_dir) / adapter_name
    state = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
    n_before = len(state)
    state = filter_peft_state_dict_for_vllm_lora(
        state, target_modules, expert_key_map=expert_key_map
    )
    if not state:
        msg = (
            f"No LoRA tensors left for vLLM export after filtering with "
            f"target_modules={target_modules!r} (had {n_before} tensors). "
            "Ensure adapt_lora_config_for_model ran with the intended "
            "LORA_TARGET_SCOPE before training."
        )
        raise ValueError(msg)
    if n_before != len(state):
        logger.info(
            "vLLM LoRA export: kept %d / %d tensors (target_modules=%r)",
            len(state),
            n_before,
            target_modules,
        )

    adapter_path.mkdir(parents=True, exist_ok=True)
    save_file(state, adapter_path / "adapter_model.safetensors")

    peft_cfg = peft_model.peft_config[adapter_name]
    cfg_dict: dict[str, JSONValue] = {
        str(key): _json_safe_value(value) for key, value in peft_cfg.to_dict().items()
    }
    cfg_dict["target_modules"] = _json_safe_value(target_modules or [])

    import json

    (adapter_path / "adapter_config.json").write_text(
        json.dumps(cfg_dict, indent=2),
        encoding="utf-8",
    )
    return adapter_path


def apply_chat_template(
    conversation_template: list[dict[str, str]],
    question: str,
    answer: str,
    tokenizer: PreTrainedTokenizerBase,
) -> BatchEncoding:
    """Create and tokenize a chat template for a reasoning task.

    :param conversation_template: The conversation template to be tokenized.
    :type conversation_template: list[dict[str, str]]
    :param question: The question to be tokenized.
    :type question: str
    :param answer: The answer to be tokenized.
    :type answer: str
    :param tokenizer: The tokenizer to be used.
    :type tokenizer: PreTrainedTokenizerBase
    :return: The tokenized prompt.
    :rtype: BatchEncoding
    """
    updated_prompt = render_chat_template(
        conversation_template,
        tokenizer,
        question=question,
        answer=answer,
    )
    # The rendered template already carries its own special tokens (BOS where
    # the template emits one); adding them again would double the BOS.
    return tokenizer(
        [updated_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
        add_special_tokens=False,
    )


def render_chat_template(
    conversation_template: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    **format_kwargs: Any,
) -> str:
    """Format each template message and render it through the chat template.

    :param conversation_template: Messages whose ``content`` holds
        ``str.format`` placeholders.
    :type conversation_template: list[dict[str, str]]
    :param tokenizer: Tokenizer providing the chat template.
    :type tokenizer: PreTrainedTokenizerBase
    :param chat_template_kwargs: Extra kwargs for the ``apply_chat_template``
        render (e.g. ``{"enable_thinking": False}``).
    :type chat_template_kwargs: dict[str, Any] | None
    :param format_kwargs: Values interpolated into each message's content.
    :return: The rendered (untokenized) prompt text.
    :rtype: str
    """
    formatted_conversation = [
        {
            "role": msg["role"],
            "content": msg["content"].format(**format_kwargs),
        }
        for msg in conversation_template
    ]
    render_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
    # ``continue_final_message`` is only meaningful for an assistant prefill; a
    # template ending on a user/system message would otherwise render with the
    # final turn left open and no assistant header at all.
    if formatted_conversation[-1]["role"] == "assistant":
        if formatted_conversation[-1]["content"]:
            render_kwargs.setdefault("continue_final_message", True)
        else:
            # An empty prefill makes continue_final_message's trim ill-defined;
            # opening a fresh assistant turn renders the same intent.
            formatted_conversation = formatted_conversation[:-1]
            render_kwargs.setdefault("add_generation_prompt", True)
    else:
        render_kwargs.setdefault("add_generation_prompt", True)
    rendered = tokenizer.apply_chat_template(
        formatted_conversation,
        tokenize=False,
        **render_kwargs,
    )
    assert isinstance(rendered, str), "tokenize=False renders to a single string"
    return rendered


def max_prompt_tokens_for_model_len(
    max_model_len: int,
    max_output_tokens: int | None,
) -> int:
    """Upper bound on prompt tokens so at least one completion token can be generated.

    Reserve generation headroom while keeping prompt budget as large as possible.
    When ``max_output_tokens`` is provided, reserve up to that many tokens
    (capped by ``max_model_len``). When it is ``None``, reserve exactly one
    token so generation remains possible without collapsing prompt budget.

    :param max_model_len: Engine context length (prompt + completion ceiling).
    :type max_model_len: int
    :param max_output_tokens: Configured completion cap; if ``None``, reserve
        one token of generation headroom.
    :type max_output_tokens: int | None
    :return: Largest allowed prompt length under that headroom (may be 0).
    :rtype: int
    """
    gen_reserve = (
        max(1, min(max_output_tokens, max_model_len))
        if max_output_tokens is not None
        else 1
    )
    return max(0, max_model_len - gen_reserve)


def normalize_prompt_batch(
    prompts: RolloutPrompt | list[RolloutPrompt],
) -> list[RolloutPrompt]:
    """Normalize rollout prompts into a list-of-dicts per sample.

    Supports both a list of per-sample dicts and a single stacked dict whose
    tensor/list values are batched on dimension 0.

    :param prompts: The prompts to normalize.
    :type prompts: RolloutPrompt | list[RolloutPrompt]
    :return: One prompt dict per sample.
    :rtype: list[RolloutPrompt]
    """
    if isinstance(prompts, list):
        return prompts

    input_ids = prompts["input_ids"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() == 1:
        return [prompts]

    batch_size = int(input_ids.shape[0])
    if batch_size == 0:
        return []

    # Inspect each key once and write it into every output dict in one pass.
    # Keys not declared on ``RolloutPrompt`` (caller-supplied metadata) are
    # copied through unchanged, which a key-by-key typed construction can't do.
    samples: list[dict[str, object]] = [{} for _ in range(batch_size)]
    for key, value in prompts.items():
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > 0
            and value.shape[0] == batch_size
        ):
            chunks = value.unbind(0) if value.dim() == 1 else value.split(1, dim=0)
            for sample, chunk in zip(samples, chunks, strict=True):
                sample[key] = chunk
        elif isinstance(value, list) and len(value) == batch_size:
            for sample, item in zip(samples, value, strict=True):
                sample[key] = item
        else:
            for sample in samples:
                sample[key] = value
    # Open dicts preserve undeclared metadata; the closed TypedDict return can't
    # name those keys, and a TypeGuard pass would only add a Python loop.
    return samples  # ty: ignore[invalid-return-type]


def is_rollout_prompt(obs: Mapping[str, object]) -> TypeGuard[RolloutPrompt]:
    """Check whether a mapping is a tokenized rollout prompt.

    :param obs: A prompt mapping returned by a rollout env.
    :type obs: Mapping[str, object]
    :return: ``True`` when the mapping carries prompt tokens.
    :rtype: TypeGuard[RolloutPrompt]
    """
    return isinstance(obs.get("input_ids"), torch.Tensor)


def is_preference_prompts(batch: Mapping[str, object]) -> TypeGuard[PreferencePrompts]:
    """Check whether a collated batch carries the chosen/rejected pair DPO needs.

    :param batch: A batch collated by an ``objective="preference"`` ``DatasetEnv``.
    :type batch: Mapping[str, object]
    :return: ``True`` when the batch carries both preference encodings.
    :rtype: TypeGuard[PreferencePrompts]
    """
    return isinstance(batch.get("chosen_input_ids"), torch.Tensor) and isinstance(
        batch.get("rejected_input_ids"), torch.Tensor
    )


def is_sft_prompts(batch: Mapping[str, object]) -> TypeGuard[SFTPrompts]:
    """Check whether a collated batch carries the prompt/response pair SFT needs.

    :param batch: A batch collated by an ``objective="sft"`` ``DatasetEnv``.
    :type batch: Mapping[str, object]
    :return: ``True`` when the batch carries the teacher-forced encoding.
    :rtype: TypeGuard[SFTPrompts]
    """
    return isinstance(batch.get("input_ids"), torch.Tensor) and isinstance(
        batch.get("response"), list
    )
