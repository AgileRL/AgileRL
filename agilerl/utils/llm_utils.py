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
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from torch import nn

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.protocols import DTensorLike
from agilerl.typing import (
    HFGeneratePrompts,
    JSONValue,
    PopulationType,
    ReasoningPrompts,
)
from agilerl.utils.distributed import (
    all_reduce_mean,
    barrier,
    get_world_size,
    is_distributed,
    resolve_device,
)

FSDPModule: Any
FullyShardedDataParallel: Any
try:
    from torch.distributed.fsdp import FSDPModule, FullyShardedDataParallel

    HAS_FSDP = True
except ImportError:  # pragma: no cover -- torch built without distributed support
    FSDPModule = None
    FullyShardedDataParallel = None
    HAS_FSDP = False

DTensor: Any
try:
    from torch.distributed.tensor import DTensor

    HAS_DTENSOR = True
except ImportError:  # pragma: no cover -- torch built without distributed support
    DTensor = None
    HAS_DTENSOR = False

if TYPE_CHECKING:
    from peft import LoraConfig, PeftModel
    from torch.nn.attention.flex_attention import BlockMask
    from transformers import PreTrainedTokenizerBase

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

_DEPRECATED_LLM_ENV_NAMES = frozenset(
    ("apply_chat_template", "ReasoningGym", "PreferenceGym", "SFTGym"),
)

# Named bitsandbytes quantization presets
_BNB_QUANT_PRESETS = frozenset({"none", "int8", "nf4"})

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


def max_prompt_tokens_for_sliding_window(
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


def validate_llm_context_lengths(
    max_model_len: int,
    max_output_tokens: int | None,
) -> None:
    """Reject configs that leave no prompt room under sliding-window rollouts.

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
            f"(max_prompt_tokens={max_prompt_tokens_for_sliding_window(max_model_len, max_output_tokens)})."
        )
        raise ValueError(msg)


def is_reasoning_prompts(obs: Mapping[str, object]) -> TypeGuard[ReasoningPrompts]:
    """Check whether a mapping is a tokenized ``ReasoningPrompts`` observation.

    :param obs: An observation mapping returned by a tokenized multi-turn env.
    :type obs: Mapping[str, object]
    :return: ``True`` when the mapping carries prompt tensors.
    :rtype: TypeGuard[ReasoningPrompts]
    """
    return isinstance(obs.get("input_ids"), torch.Tensor) and isinstance(
        obs.get("attention_mask"), torch.Tensor
    )


def normalize_reasoning_prompt_batch(
    prompts: ReasoningPrompts | list[ReasoningPrompts],
) -> list[ReasoningPrompts]:
    """Normalize reasoning prompts into a list-of-dicts per sample.

    Supports both legacy list-of-dicts and stacked dict formats where tensor/list
    values are batched on dimension 0.
    :param prompts: The prompts to normalize.
    :type prompts: ReasoningPrompts | list[ReasoningPrompts]
    :return: The normalized prompts.
    :rtype: list[ReasoningPrompts]
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
    # Keys not declared on ``ReasoningPrompts`` (caller-supplied metadata) are
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


def gather_tensor(
    tensor: torch.Tensor | npt.NDArray | float,
) -> torch.Tensor:
    """Gather a tensor from every rank (identity on a single device).

    Prefer ``torch.distributed`` when a process group is initialised.

    :param tensor: Tensor (or array/scalar convertible to one) to gather.
    :return: Stacked / concatenated tensors from all ranks.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if not is_distributed():
        return tensor
    tensor = tensor.detach().to(torch.device(resolve_device()))
    gathered = [torch.empty_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered) if tensor.dim() == 0 else torch.cat(gathered)


def needs_cross_rank_seq_padding(algo: object, *, world_size: int) -> bool:
    """Return whether ranks must sync completion seq lengths before ``learn()``.

    Multi-rank Liger token-level losses chunk over ``B * (T - 1)`` and issue one
    NCCL allreduce per chunk (DAPO/CISPO normaliser). FSDP2 full-shard (and any
    collective-heavy DP path) also requires identical per-rank ``T`` so every
    rank issues the same NCCL collectives.
    """
    if world_size <= 1:
        return False
    if getattr(algo, "fsdp_config", None) is not None:
        return True
    if getattr(algo, "distributed", False):
        return True
    if not getattr(algo, "use_liger_loss", False):
        return False
    return getattr(algo, "importance_sampling_level", "token") == "token"


def allreduce_minmax_int(value: int) -> tuple[int, int]:
    """Return ``(min, max)`` of ``value`` across ranks via torch.distributed."""
    if not is_distributed():
        v = int(value)
        return v, v
    t = torch.tensor(
        [int(value)], device=torch.device(resolve_device()), dtype=torch.long
    )
    gathered = gather_tensor(t)
    return int(gathered.min().item()), int(gathered.max().item())


def pad_completion_batch_to_seq_len(
    completion_ids: torch.Tensor,
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
    if completion_ids.dim() != required_dims:
        msg = f"completion_ids must be (B, T), got shape {tuple(completion_ids.shape)}"
        raise ValueError(msg)
    if action_masks.dim() != required_dims:
        msg = f"action_masks must be (B, T-1), got shape {tuple(action_masks.shape)}"
        raise ValueError(msg)

    batch, seq_len = completion_ids.shape
    mask_len = action_masks.shape[1]
    if mask_len != seq_len - 1:
        msg = (
            f"action_masks length ({mask_len}) must be completion_ids length "
            f"({seq_len}) - 1"
        )
        raise ValueError(msg)
    if target_seq_len < seq_len:
        msg = f"target_seq_len ({target_seq_len}) must be >= local seq_len ({seq_len})"
        raise ValueError(msg)
    if target_seq_len == seq_len:
        return completion_ids, action_masks

    pad_t = target_seq_len - seq_len
    completion_ids = torch.nn.functional.pad(
        completion_ids,
        (0, pad_t),
        value=pad_token_id,
    )
    action_masks = torch.nn.functional.pad(
        action_masks,
        (0, pad_t),
        value=False,
    )
    if completion_ids.shape != (batch, target_seq_len):
        msg = (
            f"padded completions shape {tuple(completion_ids.shape)} != "
            f"({batch}, {target_seq_len})"
        )
        raise RuntimeError(msg)
    if action_masks.shape != (batch, target_seq_len - 1):
        msg = (
            f"padded masks shape {tuple(action_masks.shape)} != "
            f"({batch}, {target_seq_len - 1})"
        )
        raise RuntimeError(msg)
    return completion_ids, action_masks


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
    minmax_fn: Callable[[int], tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sync ``B``/``T`` across ranks before heavy local pad/stack work.

    Collective metadata sync happens first; pad/stack follows. A DP barrier runs
    after pad so no rank enters sharded ``learn`` collectives while peers are
    still padding. Call immediately before ``learn()`` when
    :func:`needs_cross_rank_seq_padding` is true. Shorter ranks are right-padded
    to the global max ``T`` so Liger token-level chunk collectives stay in
    lockstep.
    """
    # Lazy import avoids a circular dependency with algo_utils -> llm_utils.
    from agilerl.utils.algo_utils import stack_and_pad_experiences

    local_b, local_t = _local_batch_and_seq_len(completion_ids)
    reduce_fn = minmax_fn or allreduce_minmax_int

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

    barrier()
    return completion_ids, action_masks, rewards


def aggregate_metrics_across_gpus(
    metric_tensor: torch.Tensor | npt.NDArray | float,
) -> float:
    """Average a metric across ranks (local mean on a single device)."""
    if not is_distributed():
        if isinstance(metric_tensor, torch.Tensor):
            return metric_tensor.float().mean().item()
        if isinstance(metric_tensor, np.ndarray):
            return float(np.mean(metric_tensor))
        return float(metric_tensor)
    if not isinstance(metric_tensor, torch.Tensor):
        metric_tensor = torch.as_tensor(metric_tensor)
    local_mean = metric_tensor.detach().float().mean()
    return all_reduce_mean(local_mean.to(torch.device(resolve_device()))).item()


def aggregate_metrics_dict(
    metrics: dict[str, torch.Tensor | npt.NDArray | float],
) -> dict[str, float]:
    """Aggregate all values in a metrics dict across GPUs (or locally)."""
    return {k: aggregate_metrics_across_gpus(v) for k, v in metrics.items()}


def is_fsdp_sharded(model: nn.Module) -> bool:
    """Return ``True`` when ``model`` contains FSDP-sharded submodules.

    Detects both FSDP2 (``fully_shard``) and legacy FSDP1 wrappers; AgileRL
    only supports FSDP2 for training.
    """
    if not HAS_FSDP or not isinstance(model, nn.Module):
        return False
    return any(
        isinstance(module, (FSDPModule, FullyShardedDataParallel))
        for module in model.modules()
    )


def _parameter_owner(param: torch.Tensor) -> tuple[nn.Module, str] | None:
    """Return ``(module, attr_name)`` for a parameter registered on a module."""
    for referrer in gc.get_referrers(param):
        if not isinstance(referrer, dict):
            continue
        attr_names = [
            key
            for key, value in referrer.items()
            if value is param and isinstance(key, str)
        ]
        if not attr_names:
            continue
        for obj in gc.get_referrers(referrer):
            if (
                isinstance(obj, nn.Module)
                and getattr(obj, "_parameters", None) is referrer
            ):
                return obj, attr_names[0]
    return None


def _is_dtensor(tensor: torch.Tensor) -> TypeGuard[DTensorLike]:
    return HAS_DTENSOR and isinstance(tensor, DTensor)


@contextmanager
def materialize_dtensors(
    *tensors: torch.Tensor | None,
) -> Generator[list[torch.Tensor | None], None, None]:
    """All-gather ``DTensor`` shards to dense locals without swapping module params.

    Prefer this for ephemeral matmuls (fused lm_head logprobs). Use
    :func:`gather_params` when in-module reads must see dense weights
    (``state_dict`` / PEFT ``save_pretrained`` / Liger). All ranks must enter
    and exit together. Yields a list parallel to ``tensors``.
    """
    locals_: list[torch.Tensor | None] = []
    for tensor in tensors:
        if tensor is None or not _is_dtensor(tensor):
            locals_.append(tensor)
        else:
            locals_.append(tensor.full_tensor())
    yield locals_


@contextmanager
def gather_params(
    params: Sequence[torch.Tensor | None],
) -> Generator[list[torch.Tensor | None], None, None]:
    """Materialize full (unsharded) views of ``params`` for the duration of the context.

    Plain tensors are left unchanged. For FSDP2 ``DTensor`` parameters, each
    tensor is all-gathered with :meth:`~torch.distributed.tensor.DTensor.full_tensor`
    and temporarily installed on its owning module so in-module reads (e.g.
    ``state_dict`` / PEFT ``save_pretrained``) see dense weights. Original
    shards are restored on exit.

    Yields a list parallel to ``params``: dense locals for any gathered
    ``DTensor``, and the original handles otherwise. Callers that hold
    pre-gather tensor references must use the yielded list for math — those
    references still point at the shard. For matmul-only gathers prefer
    :func:`materialize_dtensors` (no module Parameter install).

    Only the tensors in ``params`` are gathered — pass
    :func:`get_lora_params` for adapter-only save/export/copy. All ranks must
    enter and exit together.

    .. warning::
        Gathered parameters are **read-only**: writes are discarded when the
        sharded ``DTensor`` is restored. Use :func:`load_full_state_dict` to
        write weights into a sharded model.
    """
    restores: list[tuple[nn.Module, str, DTensorLike]] = []
    locals_: list[torch.Tensor | None] = []
    try:
        for param in params:
            if param is None or not _is_dtensor(param):
                locals_.append(param)
                continue
            full = param.full_tensor()
            locals_.append(full)
            owner = _parameter_owner(param)
            if owner is None:
                continue
            module, name = owner
            restores.append((module, name, param))
            owned: dict[str, Any] = module._parameters
            owned[name] = nn.Parameter(full, requires_grad=bool(param.requires_grad))
        yield locals_
    finally:
        for module, name, original in reversed(restores):
            owned: dict[str, Any] = module._parameters
            owned[name] = original


def load_full_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """Load a full (unsharded) state dict into a model that may be FSDP-sharded.

    For non-sharded models this is a plain ``load_state_dict``. For FSDP2
    models, the full state dict is distributed onto the sharded parameters via
    ``torch.distributed.checkpoint``.
    """
    if is_fsdp_sharded(model):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        model_state_dict: Any = state_dict
        set_model_state_dict(
            model,
            model_state_dict,
            options=StateDictOptions(full_state_dict=True, strict=strict),
        )
    else:
        model.load_state_dict(state_dict, strict=strict)


@contextmanager
def gather_if_zero3(
    zero_stage: int | None,
    params: list[torch.Tensor],
    modifier_rank: int | None = None,
) -> Generator[list[torch.Tensor | None], None, None]:
    """Gather ``params`` for the duration of the context.

    Delegates to :func:`gather_params`. ``zero_stage`` and ``modifier_rank``
    are accepted for call-site compatibility and ignored.
    """
    del zero_stage, modifier_rank
    with gather_params(params) as gathered:
        yield gathered


@contextmanager
def gather_if_ds_param(
    *tensors: torch.Tensor | None,
    modifier_rank: int | None = 0,
) -> Generator[list[torch.Tensor | None], None, None]:
    """Gather the given tensors for ephemeral matmuls (no module Parameter swap).

    Delegates to :func:`materialize_dtensors`. ``modifier_rank`` is accepted for
    call-site compatibility and ignored. Yields dense locals; pre-gather
    handles stay sharded. Use :func:`gather_params` when in-module reads need
    dense weights installed.
    """
    del modifier_rank
    with materialize_dtensors(*tensors) as gathered:
        yield gathered


def get_lora_params(model: nn.Module) -> list[torch.Tensor]:
    """Return adapter parameters for scoped gathers / export.

    Pass the result to :func:`gather_params` for adapter-only save, export, or
    copy so base-model shards stay unmaterialised.
    """
    return [p for n, p in model.named_parameters() if "lora" in n]


def save_lora_adapters(
    model: nn.Module,
    path: str | Path,
    selected_adapters: Sequence[str],
    use_value_head: bool = False,
    is_main: bool = True,
) -> None:
    r"""Save LoRA adapter weights and configs in PEFT-compatible format.

    Gathers FSDP2-sharded adapter parameters to CPU *before* handing them to
    safetensors, avoiding the invalid-storage error that occurs when
    ``full_tensor()`` results are installed as live GPU ``nn.Parameter``\ s
    and then serialised via PEFT's ``save_pretrained``.

    All ranks must call this together (``full_tensor()`` is a collective),
    but only ``is_main`` writes to disk.
    """
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    base_path = Path(path)

    is_dist = dist.is_available() and dist.is_initialized()

    for adapter_name in selected_adapters:
        adapter_dir = base_path / adapter_name
        if is_main:
            adapter_dir.mkdir(parents=True, exist_ok=True)

        # get_peft_model_state_dict filters by adapter and strips the
        # adapter-name segment from the keys, producing the exact format
        # that set_peft_model_state_dict expects on load.
        raw_state = get_peft_model_state_dict(model, adapter_name=adapter_name)
        cpu_state: dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            cpu_state[key] = value.to("cpu").contiguous()

        if is_main:
            save_file(
                cpu_state,
                str(adapter_dir / "adapter_model.safetensors"),
                metadata={"format": "pt"},
            )
            if hasattr(model, "peft_config") and adapter_name in model.peft_config:
                model.peft_config[adapter_name].save_pretrained(str(adapter_dir))

        del cpu_state
        if is_dist:
            dist.barrier()

    # Save the value head (PPO's v_head Linear) as pytorch_model.bin so
    # _restore_value_head can find it on load.
    if use_value_head:
        v_head_state: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if "v_head" in name:
                full = param.full_tensor() if hasattr(param, "full_tensor") else param
                v_head_state[name] = full.to("cpu").contiguous()
        if is_main and v_head_state:
            torch.save(v_head_state, str(base_path / "pytorch_model.bin"))
        if is_dist:
            dist.barrier()


def load_lora_adapters(
    model: nn.Module,
    path: str | Path,
    adapter_name: str,
    device: torch.device | str = "cpu",
) -> None:
    """Load LoRA adapter weights from a PEFT-compatible checkpoint directory.

    Handles FSDP2-sharded models by scattering loaded full tensors into DTensor
    local shards via ``distribute_tensor``. For non-sharded models, falls
    back to a plain ``copy_``.

    All ranks must call this together (``distribute_tensor`` is a collective).
    """
    from safetensors.torch import load_file as safe_load_file

    adapter_path = Path(path) / adapter_name / "adapter_model.safetensors"
    adapter_state = safe_load_file(str(adapter_path), device=str(device))

    is_dist = dist.is_available() and dist.is_initialized()

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Only touch parameters belonging to this adapter.
            if f".{adapter_name}." not in name:
                continue
            # Map model param name → loaded state dict key by removing the
            # adapter-name segment: ...lora_A.actor.weight → ...lora_A.weight
            loaded_key = name.replace(f".{adapter_name}.", ".")
            if loaded_key not in adapter_state:
                continue

            full_tensor = adapter_state[loaded_key].to(device)

            if hasattr(param, "device_mesh"):
                # FSDP2 DTensor: scatter the full tensor onto the mesh
                # and copy the local shard into the parameter's local shard.
                from torch.distributed.tensor import distribute_tensor

                sharded = distribute_tensor(
                    full_tensor, param.device_mesh, param.placements
                )
                param.data._local_tensor.copy_(sharded._local_tensor)
            else:
                param.data.copy_(full_tensor)

    if is_dist:
        dist.barrier()


def get_state_dict(
    model: nn.Module, cpu_offload: bool = True
) -> dict[str, torch.Tensor]:
    """Get the full state dict of a model, gathering FSDP-sharded parameters.

    For FSDP2 models the state dict is materialized via
    ``torch.distributed.checkpoint`` so the returned tensors own their storage
    (plain unshard views would be freed on reshard).

    Under FSDP2, ``cpu_offload`` must be ``True``: only rank 0 receives the
    full state dict on CPU; other ranks get an empty dict. A GPU full state
    dict would place the entire model on every rank and is rejected.
    """
    if is_fsdp_sharded(model):
        if not cpu_offload:
            msg = (
                "get_state_dict(cpu_offload=False) is not supported for FSDP2 "
                "models: a full GPU state dict would place the entire model on "
                "every rank. Use cpu_offload=True for checkpoints, or "
                "adapter-scoped helpers for clones."
            )
            raise ValueError(msg)
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        full_state: Any = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        return full_state
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
    * ``"nf4"`` -- 4-bit NF4 with bf16 compute, bf16 quant storage and double
      quantisation (the QLoRA recipe, ZeRO-3 compatible).
    * ``dict`` -- forwarded verbatim as ``BitsAndBytesConfig(**spec)`` for full
      control; ``bnb_4bit_compute_dtype`` / ``bnb_4bit_quant_storage`` may be
      given as dtype strings (e.g. ``"bfloat16"``), which transformers resolves.

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
        mode = spec.strip().lower()
        if mode in ("none", ""):
            return None
        if mode == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        if mode == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        msg = (
            f"Unknown quantization preset {spec!r}; expected one of "
            f"{sorted(_BNB_QUANT_PRESETS)} or a BitsAndBytesConfig kwargs dict."
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

    **Auto-detect**: when ``options`` is ``None``, the visible GPU's compute
    capability is checked. Hopper (SM90+ / H100, H200) has ~228 KB usable
    shared memory per SM — enough for the stock kernel's ~208 KB tiles — so
    the patch is skipped and FlexAttention's autotuner picks larger blocks
    for better throughput. On A100 (SM80) and earlier the SRAM-safe small
    blocks are installed. Pass an explicit ``options`` dict to override the
    auto-decision either way.

    :param options: Override the default kernel options (forward + backward
        block sizes, ``num_warps``, ``num_stages``). When given, the
        capability auto-skip is bypassed and the supplied options are
        installed unconditionally.
    :type options: dict[str, Any] | None
    """
    try:
        from transformers.integrations.flex_attention import flex_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    if getattr(flex_attention_forward, "_agilerl_kernel_opts_patched", False):
        return

    # Auto-skip on Hopper+. The stock kernel's ~208 KB tiles fit in Hopper's
    # ~228 KB usable shared memory, so leaving the autotuner alone lets it
    # pick larger, faster blocks. Below SM90 (A100 ~164 KB, Ampere consumer
    # ~100 KB) the stock config OOMs the kernel launch, so fall through to
    # install the small-block safe defaults.
    if options is None and torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            capability = (0, 0)
        if capability >= (9, 0):
            return

    # head_dim=256 makes the Q/K/V tiles (BLOCK x head_dim) the dominant SRAM
    # cost, so use small 32-wide blocks to fit the A100's ~163 KB shared memory.
    opts = options or {
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
        kwargs.setdefault("kernel_options", opts)
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
    use_distributed: bool = False,
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
    :param use_distributed: Whether the model is created for a distributed
        run, defaults to False
    :type use_distributed: bool, optional
    :return: The created model.
    :rtype: PreTrainedModel
    """
    # Start from the caller's config (if any) and fill in our SDPA + dtype
    # defaults with ``setdefault``, so any explicit caller value stays
    # authoritative.
    model_config = dict(model_config) if model_config else {}
    model_config.setdefault("torch_dtype", torch.bfloat16)
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
    device: str | torch.device | None = None,
) -> str:
    """Resolve the training device for an LLM algorithm.

    Pins to ``cuda:<local_rank>`` under distributed training via
    :func:`agilerl.utils.distributed.resolve_device`.
    """
    return resolve_device(device)


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


def stitch_completion_after_windowed_hf_generate(
    completion_id: torch.Tensor,
    stitch: torch.Tensor | None,
    initial_len: int | None,
) -> tuple[torch.Tensor, int | None]:
    """Reinsert dropped middle tokens after HF ``generate`` on a windowed prompt.

    ``completion_id`` is ``concat(model_input_ids, new_tokens)``. The full
    chronological sequence is
    ``concat(completion_id[:, :initial_len], stitch, completion_id[:, initial_len:])``.

    :param completion_id: Output of ``generate`` on the truncated prompt,
        shape ``(1, seq_len)``.
    :type completion_id: torch.Tensor
    :param stitch: Middle segment removed for context windowing, shape ``(1, K)``.
    :type stitch: torch.Tensor | None
    :param initial_len: ``model_window_initial_len`` (length of the initial
        user segment within ``model_input_ids``).
    :type initial_len: int | None
    :return: Full prompt plus generation with stitch restored.
    :rtype: torch.Tensor
    """
    if stitch is None:
        return completion_id, initial_len
    # A windowed prompt always pairs a stitch tensor with its initial length.
    assert initial_len is not None
    stitch = stitch.to(completion_id.device, non_blocking=True)
    stitch_len = stitch.shape[1]
    full_prompt_len = initial_len + stitch_len
    return (
        torch.cat(
            [
                completion_id[:, :initial_len],
                stitch,
                completion_id[:, initial_len:],
            ],
            dim=1,
        ),
        full_prompt_len,
    )


def build_completion_mask(
    completion_id: torch.Tensor,
    prompt_len: int | None,
    pad_token_id: int,
    completion_len: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the boolean action mask for a completion tensor.

    Returns ``True`` at positions that are part of the generated completion
    (i.e. past the prompt), dropping the leading position to align with the
    next-token-prediction shift used downstream.

    When ``completion_len`` is provided (per-row generated-token counts) the
    completion span is marked by position — ``positions in
    [prompt_len, prompt_len + completion_len)`` — so the mask is correct even
    when ``pad_token_id == eos_token_id`` (a real generated EOS is included, only
    the trailing padding is excluded). When ``None`` the span is inferred from
    non-pad tokens, which miscounts a generated EOS when pad aliases eos.

    :param completion_id: Token tensor of shape ``(B, seq_len)`` containing
        the prompt followed by generated tokens.
    :type completion_id: torch.Tensor
    :param prompt_len: Number of leading tokens to mask out (the full
        prompt length, possibly after sliding-window stitching). ``None``
        means "no prompt prefix" — every non-pad token is part of the
        completion. This matches the legacy slice semantics where
        ``mask[:, None:] = True`` set the entire dim before pads were
        zeroed back out.
    :type prompt_len: int | None
    :param pad_token_id: Pad token id used to suppress padding positions.
    :type pad_token_id: int
    :param completion_len: Per-row count of generated tokens. When provided,
        the completion span is marked by position (robust to
        ``pad_token_id == eos_token_id``); otherwise it is inferred from
        non-pad tokens. Shape ``(B,)``, same device as ``completion_id``.
    :type completion_len: torch.Tensor | None
    :return: Boolean mask of shape ``(B, seq_len - 1)``.
    :rtype: torch.Tensor
    """
    if completion_len is not None and (prompt_len is None or prompt_len == 0):
        msg = "completion_len requires a non-zero prompt_len to locate the span."
        raise ValueError(msg)
    if completion_len is not None:
        positions = torch.arange(completion_id.shape[1], device=completion_id.device)
        end = prompt_len + completion_len.to(device=completion_id.device)
        mask = (positions.unsqueeze(0) >= prompt_len) & (
            positions.unsqueeze(0) < end.unsqueeze(-1)
        )
        return mask[:, 1:]
    non_pad = completion_id != pad_token_id
    if prompt_len is None or prompt_len == 0:
        mask = non_pad
    else:
        positions = torch.arange(completion_id.shape[1], device=completion_id.device)
        mask = (positions.unsqueeze(0) >= prompt_len) & non_pad
    return mask[:, 1:]


def hf_completion_lengths(
    completion_id: torch.Tensor,
    prompt_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Recover the true per-row generated-token count from an HF ``generate`` output.

    HF ``generate`` pads finished rows with ``pad_token_id`` up to the batch's
    longest generation. When ``pad_token_id == eos_token_id`` the padded
    positions and a real stopping EOS share an id, so a token-id scan can't
    tell them apart. This helper disambiguates by position: a row's generated
    region is ``[prompt_len, prompt_len + G)`` where ``G`` is the batch max. The
    first token equal to ``pad_token_id`` within that region is the stopping
    EOS (so the row generated ``first_pad - prompt_len + 1`` tokens, EOS
    included); if no such token exists the row ran to ``G`` (no EOS, hit the
    cap). At most one EOS can appear because ``generate`` stops a row at its
    first EOS.

    :param completion_id: ``generate`` output, shape ``(B, prompt_len + G)``.
    :type completion_id: torch.Tensor
    :param prompt_len: Number of leading prompt tokens.
    :type prompt_len: int
    :param pad_token_id: Pad token id (may equal the eos id).
    :type pad_token_id: int
    :return: Per-row generated-token counts, shape ``(B,)``, on
        ``completion_id``'s device.
    :rtype: torch.Tensor
    """
    B, L = completion_id.shape
    device = completion_id.device
    G = L - prompt_len
    if G <= 0:
        return torch.zeros(B, dtype=torch.long, device=device)
    gen_region = completion_id[:, prompt_len:]
    is_pad = gen_region == pad_token_id
    has_pad = is_pad.any(dim=1)
    first_pad = is_pad.int().argmax(dim=1)
    first_pad = torch.where(has_pad, first_pad, torch.full_like(first_pad, G))
    gen_len = torch.where(first_pad < G, first_pad + 1, first_pad)
    return gen_len.to(torch.long)


def build_hf_completion_mask(
    completion_id: torch.Tensor,
    input_ids_len: int,
    initial_prompt_len: int | None,
    stitch_ids: torch.Tensor | None,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch a windowed HF ``generate`` output and build its action mask.

    Wraps :func:`stitch_completion_after_windowed_hf_generate` and
    :func:`build_completion_mask` so the mask is correct when
    ``pad_token_id == eos_token_id``. The generated span is marked by position
    using the true per-row generation length recovered from the raw output,
    so a real stopping EOS (which aliases the pad id) is included and only the
    trailing padding is excluded — for both the current turn and any prior
    turns retained in the windowed prompt suffix.

    The windowed prompt is ``[initial_segment | recent_suffix]`` of length
    ``input_ids_len``; ``generate`` appends ``new_tokens``; stitching
    reinserts the dropped middle to give
    ``[initial_segment | stitch | recent_suffix | new_tokens | pad]``. The
    action span is ``[full_prompt_len, full_prompt_len + recent_suffix_len +
    new_token_len)`` where ``recent_suffix_len = input_ids_len -
    initial_prompt_len``.

    :param completion_id: Raw ``generate`` output, shape ``(B, input_ids_len
        + G)``.
    :type completion_id: torch.Tensor
    :param input_ids_len: Length of the prompt fed to ``generate`` (the
        windowed prompt length).
    :type input_ids_len: int
    :param initial_prompt_len: Length of the initial segment within the
        windowed prompt (``None`` when no windowing).
    :type initial_prompt_len: int | None
    :param stitch_ids: Dropped middle segment to reinsert (``None`` when no
        windowing).
    :type stitch_ids: torch.Tensor | None
    :param pad_token_id: Pad token id (may equal the eos id).
    :type pad_token_id: int
    :return: ``(stitched_completion_id, action_mask)`` where the mask has
        shape ``(B, seq_len - 1)``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    new_token_len = hf_completion_lengths(completion_id, input_ids_len, pad_token_id)
    completion_id, full_prompt_len = stitch_completion_after_windowed_hf_generate(
        completion_id, stitch_ids, initial_prompt_len
    )
    if stitch_ids is None:
        # Non-windowed path: prompt length equals the generate input length.
        completion_len = new_token_len
        full_prompt_len = input_ids_len
    else:
        recent_suffix_len = input_ids_len - initial_prompt_len
        completion_len = new_token_len + recent_suffix_len
    mask = build_completion_mask(
        completion_id, full_prompt_len, pad_token_id, completion_len=completion_len
    )
    return completion_id, mask


def stitch_completion_after_windowed_vllm_generate(
    completion_ids: list[torch.Tensor],
    stitch_prefixes: list[torch.Tensor],
    group_prompts: Sequence[ReasoningPrompts],
    group_size: int,
    prompts: Sequence[ReasoningPrompts],
) -> list[torch.Tensor]:
    """Reinsert dropped middle segments into ``model_prompt | generation`` tensors.

    For each logical prompt ``i``, ``block`` is
    ``concat(trajectory_input_ids, new_tokens)`` (batched over ``group_size``).
    When ``stitch_prefix_ids`` is non-empty, the full chronological sequence is
    ``concat(block[:, :I], stitch, block[:, I:], dim=1)`` with
    ``I = initial_prompt_len`` from the corresponding prompt dict.

    :param completion_ids: One tensor per logical prompt: prompt+gen per row.
    :type completion_ids: list[torch.Tensor]
    :param stitch_prefixes: Parallel to expanded ``group_prompts``; empty
        tensors when no windowing for that slot.
    :type stitch_prefixes: list[torch.Tensor]
    :param group_prompts: ``prompts`` expanded so each original prompt repeats
        ``group_size`` times (same order as vLLM batch).
    :type group_prompts: Sequence[ReasoningPrompts]
    :param group_size: Number of repeated entries per logical prompt.
    :type group_size: int
    :param prompts: Original batch of observation mappings (length ``N``).
    :type prompts: Sequence[ReasoningPrompts]
    :return: Same length as ``completion_ids``, with middle stitch applied
        where ``stitch_prefixes`` is non-empty.
    :rtype: list[torch.Tensor]
    """
    if group_size != 1:
        error_msg = f"vLLM sliding-window stitch is only implemented for group_size=1 (got {group_size})."
        raise ValueError(error_msg)
    stitched: list[torch.Tensor] = []
    for i, _ in enumerate(prompts):
        completion_i = completion_ids[i]
        stitch_i = stitch_prefixes[group_size * i]
        if stitch_i.shape[1] == 0:
            stitched.append(completion_i)
            continue
        initial_prompt_len_raw = group_prompts[group_size * i].get("initial_prompt_len")
        if initial_prompt_len_raw is None:
            msg = "initial_prompt_len required when stitch_prefix_ids is non-empty"
            raise ValueError(
                msg,
            )
        if isinstance(initial_prompt_len_raw, torch.Tensor):
            initial_prompt_len_i = int(initial_prompt_len_raw.item())
        elif isinstance(initial_prompt_len_raw, list):
            if not initial_prompt_len_raw:
                msg = "initial_prompt_len list is empty"
                raise ValueError(msg)
            initial_prompt_len_i = int(initial_prompt_len_raw[0])
        else:
            initial_prompt_len_i = int(initial_prompt_len_raw)
        group_size_i = completion_i.shape[0]
        stitch_group_i = stitch_i.expand(group_size_i, -1)
        stitched.append(
            torch.cat(
                [
                    completion_i[:, :initial_prompt_len_i],
                    stitch_group_i,
                    completion_i[:, initial_prompt_len_i:],
                ],
                dim=1,
            ),
        )
    return stitched


def prepare_prompt_hf_generate(
    prompt_dict: ReasoningPrompts, device: torch.device
) -> HFGeneratePrompts:
    """Prepare a prompt dictionary for HuggingFace generate.

    :param prompt_dict: The prompt dictionary to prepare.
    :type prompt_dict: ReasoningPrompts
    :param device: The device to move the prompt dictionary to.
    :type device: torch.device
    :return: The prepared prompt dictionary.
    :rtype: HFGeneratePrompts
    """
    # Trajectory keys may be absent or explicitly None (first turn); both fall
    # back to the initial prompt tensors.
    input_ids = prompt_dict.get("trajectory_input_ids")
    if input_ids is None:
        input_ids = prompt_dict["input_ids"]
    attention_mask = prompt_dict.get("trajectory_attention_mask")
    if attention_mask is None:
        attention_mask = prompt_dict["attention_mask"]
    stitched = prompt_dict.get("stitch_prefix_ids")
    initial_prompt_len = prompt_dict.get("initial_prompt_len")
    if isinstance(initial_prompt_len, torch.Tensor):
        initial_prompt_len = (
            int(initial_prompt_len.item()) if initial_prompt_len.numel() == 1 else None
        )
    elif isinstance(initial_prompt_len, list):
        initial_prompt_len = initial_prompt_len[0] if initial_prompt_len else None

    result: HFGeneratePrompts = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "stitch_prefix_ids": stitched,
        "initial_prompt_len": initial_prompt_len,
    }
    return result


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


def sample_eval_prompts(
    env: Any,  # noqa: ANN401 -- gym env duck-typed across SFT/Preference/other gyms
    n: int = 5,
    seed: int = 0,
) -> list[tuple[str, str | None, str | None]]:
    """Randomly sample *n* ``(prompt, chosen, rejected)`` triples from
    *env*'s held-out test dataset.

    Columns are resolved automatically per gym type:

    * :class:`SFTGym` — ``chosen`` is ``env.response_column``; ``rejected``
      is ``None`` (SFT has no negative example).
    * :class:`PreferenceGym` — ``chosen`` and ``rejected`` map to the
      dataset's ``"chosen"`` / ``"rejected"`` columns.
    * Any other gym — both are ``None``.

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
    if hasattr(env, "response_column"):  # SFTGym
        chosen_col = env.response_column
    elif "chosen" in dataset.features:  # PreferenceGym
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
    # FSDP2 keeps adapter params as sharded ``DTensor``s; ``gather_params``
    # may not install dense locals on the module for every param, so
    # materialise any remaining ``DTensor`` entries here before safetensors
    # serialisation (which cannot read a DTensor storage pointer).
    state = {k: (v.full_tensor() if _is_dtensor(v) else v) for k, v in state.items()}
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
