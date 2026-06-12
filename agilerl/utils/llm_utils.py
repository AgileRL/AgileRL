from __future__ import annotations

import logging
import random
import shutil
import textwrap
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.typing import ReasoningPrompts

try:
    from torch.distributed.fsdp import FSDPModule, FullyShardedDataParallel

    HAS_FSDP = True
except ImportError:  # pragma: no cover -- torch built without distributed support
    FSDPModule = None
    FullyShardedDataParallel = None
    HAS_FSDP = False

logger = logging.getLogger(__name__)

if HAS_LLM_DEPENDENCIES:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel

    from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
else:
    AutoTokenizer = Any
    PreTrainedModel = Any
    Dataset = Any
    AutoModelForCausalLM = Any  # type: ignore[assignment,misc]
    AutoModelForCausalLMWithValueHead = Any  # type: ignore[assignment,misc]

_DEPRECATED_LLM_ENV_NAMES = frozenset(
    ("apply_chat_template", "ReasoningGym", "PreferenceGym", "SFTGym"),
)


def __getattr__(name: str) -> Any:
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

    # Inspect each key once and write into all output dicts in one pass
    result: list[ReasoningPrompts] = [{} for _ in range(batch_size)]
    for key, value in prompts.items():
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > 0
            and value.shape[0] == batch_size
        ):
            chunks: tuple[torch.Tensor, ...] = (
                value.unbind(0) if value.dim() == 1 else value.split(1, dim=0)
            )
            for sample, chunk in zip(result, chunks, strict=True):
                sample[key] = chunk
        elif isinstance(value, list) and len(value) == batch_size:
            for sample, v in zip(result, value, strict=True):
                sample[key] = v
        else:
            for sample in result:
                sample[key] = value
    return result


def is_fsdp_sharded(model: nn.Module) -> bool:
    """Return ``True`` when ``model`` contains FSDP-sharded submodules.

    Detects both FSDP2 (``fully_shard``) and legacy FSDP1 wrappers; only FSDP2
    is supported by AgileRL (see :func:`gather_full_params`).

    :param model: The model to inspect.
    :type model: nn.Module
    :return: Whether any submodule is FSDP-sharded.
    :rtype: bool
    """
    if not HAS_FSDP:
        return False
    return any(
        isinstance(module, (FSDPModule, FullyShardedDataParallel))
        for module in model.modules()
    )


@contextmanager
def gather_full_params(model: nn.Module) -> Generator[None, None, None]:
    """Materialize full (unsharded) parameters for the duration of the context.

    No-op for non-sharded models (single process, DDP). For FSDP2
    (``fully_shard``) models, parameters are unsharded on entry and resharded
    on exit.

    .. warning::
        Under FSDP2 the gathered parameters are **read-only**: writes target
        the all-gather buffer, which is discarded on reshard. Use
        :func:`load_full_state_dict` to write weights into a sharded model.

    :param model: The model whose parameters should be gathered.
    :type model: nn.Module
    """
    fsdp2_modules = (
        [module for module in model.modules() if isinstance(module, FSDPModule)]
        if HAS_FSDP
        else []
    )
    if not fsdp2_modules:
        if HAS_FSDP and any(
            isinstance(module, FullyShardedDataParallel) for module in model.modules()
        ):
            msg = (
                "Legacy FSDP1 wrapping detected. AgileRL only supports FSDP2 "
                "(torch.distributed.fsdp.fully_shard); shard the model with "
                "agilerl.utils.distributed.apply_fsdp2 instead."
            )
            raise NotImplementedError(msg)
        yield
        return

    for module in fsdp2_modules:
        module.unshard()
    try:
        yield
    finally:
        for module in reversed(fsdp2_modules):
            module.reshard()


def load_full_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """Load a full (unsharded) state dict into a model that may be FSDP-sharded.

    For non-sharded models this is a plain ``load_state_dict``. For FSDP2
    models, the full state dict is distributed onto the sharded parameters via
    ``torch.distributed.checkpoint``.

    :param model: The (possibly sharded) model to load weights into.
    :type model: nn.Module
    :param state_dict: Full state dict keyed like ``model.state_dict()``.
    :type state_dict: dict[str, torch.Tensor]
    :param strict: Whether to require an exact key match.
    :type strict: bool
    """
    if is_fsdp_sharded(model):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            model,
            state_dict,
            options=StateDictOptions(full_state_dict=True, strict=strict),
        )
    else:
        model.load_state_dict(state_dict, strict=strict)


def get_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Get the full state dict of a model, gathering FSDP-sharded parameters.

    For FSDP2 models the state dict is materialized via
    ``torch.distributed.checkpoint`` so the returned tensors own their storage
    (plain unshard views would be freed on reshard).

    :param model: The model to get the state dict of.
    :type model: nn.Module
    :return: The state dict of the model.
    :rtype: dict[str, torch.Tensor]
    """
    if is_fsdp_sharded(model):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        return get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    return model.state_dict()


def create_model_from_name_or_path(
    model_name_or_path: str,
    model_config: dict[str, Any] | None = None,
    add_value_head: bool = False,
    use_distributed: bool = False,
) -> PreTrainedModel:
    """Create a model from a name or path.

    :param model_name_or_path: The name or path of the model to create.
    :type model_name_or_path: str
    :param model_config: The configuration of the model to create.
    :type model_config: dict[str, Any ] | None
    :param use_value_head: Flag to indicate if a value head should be added to the model, defaults to False
    :type use_value_head: bool, optional
    :param use_distributed: Whether the model is created for a distributed
        run, defaults to False
    :type use_distributed: bool, optional
    :return: The created model.
    :rtype: PreTrainedModel
    """
    if model_config is None:
        model_config = {
            "torch_dtype": torch.bfloat16 if not use_distributed else torch.float16,
            "attn_implementation": "sdpa",
        }
    if add_value_head:
        return AutoModelForCausalLMWithValueHead.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            **model_config,
        )
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_config,
    )


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: bool | None = None
) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
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
    :param turn_ids: [batch, seq_len] turn index per token, -1 for non-action.
    :param num_turns: Total number of turns (max turn_id + 1).
    :param reduction: ``"mean"`` (default) for mean-pooling,
        ``"sum"`` for sum-pooling (e.g. to aggregate log-ratios),
        ``"final_value"`` to select the last token value per turn.
    :return: [batch, num_turns] aggregated values per turn.
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


def move_params_to_cpu(unwrapped_model: torch.nn.Module) -> None:
    """Move params to CPU.

    :param agent: Distributed agent
    :type agent: DistributedLLMAgent
    :return: None
    :rtype: None
    """
    if next(unwrapped_model.parameters()).device != "cpu":
        unwrapped_model.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def stitch_completion_after_windowed_hf_generate(
    completion_id: torch.Tensor,
    stitch: torch.Tensor,
    initial_len: int,
) -> tuple[torch.Tensor, int]:
    """Reinsert dropped middle tokens after HF ``generate`` on a windowed prompt.

    ``completion_id`` is ``concat(model_input_ids, new_tokens)``. The full
    chronological sequence is
    ``concat(completion_id[:, :initial_len], stitch, completion_id[:, initial_len:])``.

    :param completion_id: Output of ``generate`` on the truncated prompt,
        shape ``(1, seq_len)``.
    :type completion_id: torch.Tensor
    :param stitch: Middle segment removed for context windowing, shape ``(1, K)``.
    :type stitch: torch.Tensor
    :param initial_len: ``model_window_initial_len`` (length of the initial
        user segment within ``model_input_ids``).
    :type initial_len: int
    :return: Full prompt plus generation with stitch restored.
    :rtype: torch.Tensor
    """
    if stitch is None:
        return completion_id, initial_len
    stitch = stitch.to(completion_id.device, non_blocking=True)
    stitch_len = stitch.shape[1] if stitch is not None else 0
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
) -> torch.Tensor:
    """Build the boolean action mask for a completion tensor.

    Returns ``True`` at positions that are (a) past the prompt and (b) not
    pad tokens, dropping the leading position to align with the
    next-token-prediction shift used downstream.

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
    :return: Boolean mask of shape ``(B, seq_len - 1)``.
    :rtype: torch.Tensor
    """
    non_pad = completion_id != pad_token_id
    if prompt_len is None or prompt_len == 0:
        mask = non_pad
    else:
        positions = torch.arange(completion_id.shape[1], device=completion_id.device)
        mask = (positions.unsqueeze(0) >= prompt_len) & non_pad
    return mask[:, 1:]


def stitch_completion_after_windowed_vllm_generate(
    completion_ids: list[torch.Tensor],
    stitch_prefixes: list[torch.Tensor],
    group_prompts: list[dict[str, Any]],
    group_size: int,
    prompts: list[dict[str, Any]],
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
    :type group_prompts: list[dict[str, Any]]
    :param group_size: Number of repeated entries per logical prompt.
    :type group_size: int
    :param prompts: Original batch of observation dicts (length ``N``).
    :type prompts: list[dict[str, Any]]
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
) -> dict[str, torch.Tensor | int]:
    """Prepare a prompt dictionary for HuggingFace generate.

    :param prompt_dict: The prompt dictionary to prepare.
    :type prompt_dict: ReasoningPrompts
    :param device: The device to move the prompt dictionary to.
    :type device: torch.device
    :return: The prepared prompt dictionary.
    :rtype: dict[str, torch.Tensor | int]
    """
    input_ids = prompt_dict.get("trajectory_input_ids", prompt_dict["input_ids"])
    attention_mask = prompt_dict.get(
        "trajectory_attention_mask",
        prompt_dict["attention_mask"],
    )
    stitched = prompt_dict.get("stitch_prefix_ids")
    initial_prompt_len = prompt_dict.get("initial_prompt_len")
    if isinstance(initial_prompt_len, torch.Tensor) and initial_prompt_len.numel() == 1:
        initial_prompt_len = int(initial_prompt_len.item())

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "stitch_prefix_ids": stitched,
        "initial_prompt_len": initial_prompt_len,
    }


def get_model_name_or_path(model: PreTrainedModel) -> str:
    """Get the name or path of a model.

    :param model: The model to get the name or path of.
    :type model: PreTrainedModel
    :return: The name or path of the model.
    :rtype: str
    """
    if hasattr(model, "name_or_path"):
        return model.name_or_path

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
    env: Any, n: int = 5, seed: int = 0
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
    agent: Any,
    tokenizer: Any,
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
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
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
