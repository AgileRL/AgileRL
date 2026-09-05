# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import shutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
)

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from tensordict import TensorDict
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.modules.dummy import DummyEvolvable
from agilerl.typing import (
    TorchObsType,
)

if TYPE_CHECKING:
    from agilerl.algorithms.core.base import EvolvableAlgorithm

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import PeftConfig, PeftModel, get_peft_model
    from transformers import PreTrainedModel

    PreTrainedModelType = PeftModel | PreTrainedModel
else:
    # Annotations referencing PreTrainedModelType are evaluated at function
    # definition time, so provide a runtime placeholder when the LLM
    # dependencies are not installed.
    PeftConfig = None
    PeftModel = None
    PreTrainedModel = None
    PreTrainedModelType = Any
    get_peft_model = None


from agilerl.utils.algo_obs import maybe_add_batch_dim
from agilerl.utils.algo_spaces import is_str_keyed_dict


@dataclass
class CosineLRScheduleConfig:
    """Data class to configure a cosine LR scheduler."""

    num_epochs: int
    warmup_proportion: float


@dataclass
class VLLMConfig:
    """Data class to configure a colocated vLLM instance.

    :param tensor_parallel_size: Number of GPUs for tensor parallelism, defaults to 1.
    :type tensor_parallel_size: int, optional
    :param gpu_memory_utilization: Fraction of GPU memory to reserve for vLLM KV cache,
        defaults to 0.3.
    :type gpu_memory_utilization: float, optional
    :param max_num_seqs: Maximum number of sequences processed concurrently.  For GRPO,
        set this to at least ``group_size`` to avoid request queuing, defaults to 8.
    :type max_num_seqs: int, optional
    :param max_num_batched_tokens: Cap on tokens vLLM may process in one scheduler
        iteration (prefill batching / compile profiling).  ``None`` uses
        :func:`~agilerl.utils.llm_utils.resolve_vllm_max_num_batched_tokens`
        (not ``max_num_seqs * max_model_len``, which OOMs long-context colocated
        init).  Set explicitly when you need full parallel max-length prefills.
    :type max_num_batched_tokens: int | None, optional
    :param sleep_mode: Put vLLM to sleep between ``get_action`` calls to free GPU memory
        for training.  Cannot be used with agent populations on a single device,
        defaults to False.
    :type sleep_mode: bool, optional
    :param sleep_mode_level: Sleep level passed to ``llm.sleep(level=...)`` when
        ``sleep_mode`` is enabled. ``1`` backs the base weights up to CPU and
        drops the KV cache; ``2`` discards the weights entirely, so it is only
        safe when new base weights are pushed into vLLM after every wake. The
        colocated LoRA-only sync never re-pushes the base, so it requires
        level 1. Defaults to 1.
    :type sleep_mode_level: int, optional
    :param dtype: Model weight dtype passed to the vLLM ``LLM`` constructor
        (e.g. ``"bfloat16"``, ``"float16"``).  ``None`` lets vLLM choose,
        defaults to None.
    :type dtype: str | None, optional
    :param quantization: Quantization method passed to the vLLM ``LLM`` constructor
        (e.g. ``"awq"``, ``"gptq"``).  ``None`` disables quantization, defaults to None.
    :type quantization: str | None, optional
    :param vllm_model_name_or_path: Optional HF id or path for the vLLM engine only.
        When set, the trainer may use a different ``model_name`` (e.g. bnb NF4 base)
        while rollout loads this checkpoint (e.g. an AWQ export).  ``None`` uses the
        trainer model path, defaults to None.
    :type vllm_model_name_or_path: str | None, optional
    :param kv_cache_dtype: Bare passthrough to vLLM's ``kv_cache_dtype`` kwarg
        (e.g. ``"fp8"`` on Hopper+ / Ada / Blackwell, ``"auto"``).  AgileRL does
        not validate any value — the string is forwarded verbatim and vLLM
        emits its own hardware errors / warnings.  ``None`` (the default) omits
        the kwarg so vLLM keeps its own default.  FP8 KV requires compute
        capability 8.9+; on A100 leave this unset.
    :type kv_cache_dtype: str | None, optional
    :param stop_sequences: List of strings that terminate generation early (e.g.
        ``["</answer>"]``).  Passed as ``stop`` to ``SamplingParams``, defaults to None.
    :type stop_sequences: list[str] | None, optional
    :param presence_penalty: Penalise tokens that have already appeared in the output;
        positive values discourage repetition.  Passed to ``SamplingParams``,
        defaults to 0.0 (disabled).
    :type presence_penalty: float, optional
    :param frequency_penalty: Penalise tokens proportionally to how often they have
        appeared so far.  Passed to ``SamplingParams``, defaults to 0.0 (disabled).
    :type frequency_penalty: float, optional
    :param max_lora_rank: Maximum LoRA rank passed to the vLLM ``LLM`` constructor.
        Should be at least the trainer's ``lora_config.r``.  Defaults to 16.
    :type max_lora_rank: int, optional
    :param max_loras: Maximum number of LoRA adapters vLLM can hold concurrently.
        Defaults to 1 (actor rollout only).
    :type max_loras: int, optional
    :param kv_cache_memory_bytes: Manually pin KV cache size in bytes instead of
        letting vLLM auto-size from ``gpu_memory_utilization``.  When set, vLLM
        uses this exact value for the KV cache and skips the auto-sizing path
        in ``determine_available_memory`` — but ``gpu_memory_utilization`` is
        **still honoured** by the upfront ``free_memory >= total_memory *
        gpu_memory_utilization`` startup check in
        ``vllm/v1/worker/gpu_worker.py:init_device``.  When running multiple
        vLLM processes concurrently you must keep ``gpu_memory_utilization``
        small enough that every worker's startup check passes.

        **Required for safe parallel/colocated vLLM**: vLLM's startup
        ``determine_available_memory`` profile run asserts that GPU free-memory
        does not increase between the pre- and post-profile snapshots.  When
        peer processes on the same GPU release memory mid-profile (concurrent
        xdist workers, sibling CI containers sharing one GPU), the assertion
        fires with ``Error in memory profiling. Initial free memory ... current
        free memory ...``.  Setting ``kv_cache_memory_bytes`` triggers vLLM's
        early-return path in ``determine_available_memory`` and skips that
        assertion entirely.  CI tests set this to a small value (e.g. 32 MiB)
        on the tiny test fixture; production deployments running a single
        vLLM should leave it unset.  Defaults to None.
    :type kv_cache_memory_bytes: int | None, optional
    :param strip_multimodal_towers: Free the GPU memory held by a multimodal
        base's unused towers after engine init (text-only RL). ``True`` strips
        the standard HF attribute names (``vision_tower``, ``audio_tower``,
        ``multi_modal_projector``, ``embed_vision``, ``embed_audio``); a list
        of attribute names strips those instead, for models that mount
        unwanted modalities elsewhere. Defaults to ``False``.
    :type strip_multimodal_towers: bool | list[str], optional
    :param lora_staging_dir: Root directory where the trained LoRA adapter is
        exported for vLLM to (re)load each sync. Staging is always
        process-private: in distributed runs each rank stages under a
        ``rank_<process_index>`` subdirectory of this root. Set explicitly
        when the adapter must live at a known path (e.g. orchestrated/arena
        deployments); user-supplied directories are created if missing and
        never deleted by AgileRL. ``None`` (default) uses a fresh
        process-private temporary directory, removed on ``clean_up``.
    :type lora_staging_dir: str | None, optional
    """

    # Colocate mode parameters
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.3
    max_num_seqs: int = 8
    max_num_batched_tokens: int | None = None
    enforce_eager: bool | None = None
    sleep_mode: bool = False
    sleep_mode_level: int = 1
    dtype: str | None = None
    quantization: str | None = None
    vllm_model_name_or_path: str | None = None
    kv_cache_dtype: str | None = None
    max_lora_rank: int = 16
    max_loras: int = 1
    strip_multimodal_towers: bool | list[str] = False
    stop_sequences: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # See class docstring above. Required to avoid vLLM's memory-profiling
    # assertion when running multiple vLLM processes on a shared GPU.
    kv_cache_memory_bytes: int | None = None
    lora_staging_dir: str | None = None

    def __post_init__(self) -> None:
        if self.sleep_mode_level not in (1, 2):
            msg = (
                "vllm sleep_mode_level must be either 1 or 2, got "
                f"{self.sleep_mode_level}."
            )
            raise ValueError(msg)

        # sleep_mode toggles the native vLLM sleep/wake cycle (base backed up to
        # host RAM, KV freed) between rollout and training for a single colocated
        # agent; it is not usable with a population on one device.
        if self.sleep_mode:
            warnings.warn(
                "VLLM sleep mode cannot be used with populations of agents on a "
                "single device. To use sleep mode, ensure you are training a "
                "single agent or, alternatively, use a different device for "
                "each agent.",
                stacklevel=2,
            )
            if self.gpu_memory_utilization <= 0.5:
                warnings.warn(
                    f"vLLM sleep_mode=True with gpu_memory_utilization="
                    f"{self.gpu_memory_utilization} — conservative for rollout "
                    f"after sleep, but vLLM still allocates its KV pool during "
                    f"``LLM()`` init before ``sleep()`` frees GPU memory. On "
                    f"smaller GPUs or long context, cap init with "
                    f"kv_cache_memory_bytes or a lower gpu_memory_utilization.",
                    stacklevel=2,
                )


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: CosineLRScheduleConfig,
    min_lr: float,
    max_lr: float,
) -> SequentialLR:
    """Create cosine annealing lr scheduler with warm-up.

    :param optimizer: Optimizer
    :type optimizer: torch.optim.Optimizer
    :param config: LR scheduler config
    :type config: CosineLRScheduleConfig
    :param min_lr: Minimum learning rate
    :type min_lr: float
    :param max_lr: Maximum learning rate
    :type max_lr: float
    :return: Return sequential learning rate scheduler
    :rtype: SequentialLR
    """
    num_epochs = config.num_epochs
    warmup_proportion = config.warmup_proportion
    warmup_epochs = int(num_epochs * warmup_proportion)
    remaining_epochs = num_epochs - warmup_epochs
    for param_group in optimizer.param_groups:
        param_group["lr"] = max_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=min_lr / max_lr,  # Start factor to get from min_lr to max_lr
        end_factor=1.0,  # End with the full max_lr
        total_iters=warmup_epochs,
    )
    # Decay scheduler: Cosine decay from max_lr to min_lr
    # Double T_max to ensure we only use the first half of the cosine curve (strictly decreasing)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=remaining_epochs * 2,  # Doubled to ensure strictly decreasing LR
        eta_min=min_lr,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def remove_nested_files(files: list[str]) -> None:
    """Remove nested files from a list of files.

    :param files: List of files to remove nested files from
    :type files: list[str]
    :param depth: Depth of the nested files, defaults to 0
    :type depth: int, optional
    """
    for f in files:
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)


def vectorize_experiences_by_agent(
    experiences: dict[str, Any],
    dim: int = 1,
) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...]:
    """Reorganizes experiences into a tensor, vectorized by time step.

    Example input:
    {'agent_0': [[1, 2, 3, 4]], 'agent_1': [[5, 6, 7, 8]]}
    Example output:
    torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    :param experiences: Dictionaries containing experiences indexed by agent_id that share a policy agent.
    :type experiences: dict[str, ObservationType]
    :param dim: New dimension to stack along
    :type dim: int
    :return: Tensor, dict of tensors, or tuple of tensors of experiences, stacked along provided dimension
    :rtype: torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...]
    """
    if not experiences:
        return torch.tensor([])

    # Get a sample value to determine the type
    sample_value = next(iter(experiences.values()))

    if isinstance(sample_value, dict):
        # Handle dictionary observations
        keys = sample_value.keys()
        vectorized_dict: dict[str, Any] = {
            k: vectorize_experiences_by_agent(
                {agent_id: experiences[agent_id][k] for agent_id in experiences},
                dim=dim,
            )
            for k in keys
        }
        return vectorized_dict
    if isinstance(sample_value, tuple):
        # Handle tuple observations
        tuple_length = len(sample_value)
        vectorized_tuple: tuple[Any, ...] = tuple(
            vectorize_experiences_by_agent(
                {agent_id: experiences[agent_id][i] for agent_id in experiences},
                dim=dim,
            )
            for i in range(tuple_length)
        )
        return vectorized_tuple
    # Original implementation for array/tensor observations
    tensors: list[torch.Tensor] = []
    for experience in experiences.values():
        if experience is None:
            continue
        tensors.append(torch.Tensor(np.array(experience)))

    # Check if all tensors have the same shape
    if all(t.shape == tensors[0].shape for t in tensors):
        stacked_tensor = torch.stack(tensors, dim=dim)
    else:
        # Concatenate along the specified dimension
        stacked_tensor = torch.cat(tensors)

    return stacked_tensor


def vectorize_agent_experiences_flat(
    experiences: dict[str, Any],
    dim: int = 1,
) -> torch.Tensor:
    """Vectorize flat per-agent experiences (log-probs, rewards, dones, values).

    :param experiences: Per-agent experiences indexed by agent id.
    :type experiences: dict[str, Any]
    :param dim: Dimension to stack along.
    :type dim: int
    :return: The stacked experiences as a single tensor.
    :rtype: torch.Tensor
    :raises TypeError: If the experiences vectorize to a structured container
        rather than a flat tensor.
    """
    vectorized = vectorize_experiences_by_agent(experiences, dim=dim)
    if not isinstance(vectorized, torch.Tensor):
        msg = (
            "vectorize_agent_experiences_flat expects flat per-agent scalars, "
            "but the experiences vectorized to a structured observation container."
        )
        raise TypeError(msg)
    return vectorized


def experience_to_tensors(
    experience: Any,  # noqa: ANN401 -- nested heterogeneous experience (dict/tuple/array-like) forwarded to np.array
    space: spaces.Space,
    actions: bool = False,
) -> TorchObsType:
    """Convert experience to tensors matching the structure of the given space.

    :param experience: Experience to convert (dict, tuple, or array-like)
    :type experience: dict[str, Any] | tuple[Any, ...] | npt.NDArray | Any
    :param space: Space to convert experience to
    :type space: spaces.Space
    :param actions: Whether the experience is an action, defaults to False
    :type actions: bool, optional
    :return: Tensor(s) of the experience
    :rtype: TorchObsType
    """
    if isinstance(experience, dict):
        assert isinstance(space, spaces.Dict), (
            f"Expected Dict space for dict experience, got {type(space)}"
        )
        tensor_dict: dict[str, Any] = {
            key: experience_to_tensors(value, space[key])
            for key, value in experience.items()
        }
        return tensor_dict
    if isinstance(experience, tuple):
        assert isinstance(space, spaces.Tuple), (
            f"Expected Tuple space for tuple experience, got {type(space)}"
        )
        tensor_tuple: tuple[Any, ...] = tuple(
            experience_to_tensors(exp, space[i]) for i, exp in enumerate(experience)
        )
        return tensor_tuple
    array = np.array(experience)

    # Ensure experience has a batch dimension
    array = maybe_add_batch_dim(array, space, actions)
    return torch.from_numpy(array)


def concatenate_tensors(tensors: list[TorchObsType]) -> TorchObsType:
    """Concatenate tensors along first dimension.

    :param tensors: List of tensors to concatenate
    :type tensors: list[TorchObsType]
    :return: Concatenated tensor
    :rtype: TorchObsType
    """
    first = tensors[0]
    if isinstance(first, dict):
        # Homogeneous by construction: all entries share the first entry's structure
        concat_dict: dict[str, Any] = {}
        for key in first:
            column: list[TorchObsType] = []
            for t in tensors:
                assert isinstance(t, dict)
                assert not isinstance(t, torch.Tensor)
                value = t[key]
                assert isinstance(value, torch.Tensor)
                column.append(value)
            concat_dict[key] = concatenate_tensors(column)
        return concat_dict
    if isinstance(first, tuple):
        concat_tuple: tuple[Any, ...] = tuple(
            _concatenate_tuple_column(tensors, i) for i in range(len(first))
        )
        return concat_tuple
    tensor_list: list[torch.Tensor] = []
    for t in tensors:
        assert isinstance(t, torch.Tensor)
        tensor_list.append(t)
    return torch.cat(tensor_list, dim=0)


def _concatenate_tuple_column(tensors: list[TorchObsType], i: int) -> TorchObsType:
    """Concatenate the i-th positional entry across a list of tuple observations."""
    column: list[TorchObsType] = []
    for t in tensors:
        assert isinstance(t, tuple)
        value = t[i]
        assert isinstance(value, torch.Tensor)
        column.append(value)
    return concatenate_tensors(column)


def reshape_from_space(tensor: TorchObsType, space: spaces.Space) -> TorchObsType:
    """Reshape tensor from space.

    :param tensor: Tensor to reshape
    :type tensor: TorchObsType
    :param space: Space to reshape tensor to
    :type space: spaces.Space
    :return: Reshaped tensor
    :rtype: TorchObsType
    """
    if isinstance(tensor, (torch.Tensor, TensorDict)):
        space_shape = space.shape
        assert space_shape is not None, (
            f"{type(space)} spaces have no shape to reshape to."
        )
        reshaped = tensor.reshape(-1, *space_shape)
        for squeeze_dim in [0, -1]:
            if reshaped.size(squeeze_dim) == 1:
                reshaped = reshaped.squeeze(squeeze_dim)

        return reshaped
    if isinstance(tensor, dict):
        assert isinstance(space, spaces.Dict), (
            f"Expected Dict space for dict tensor, got {type(space)}"
        )
        reshaped_dict: dict[str, Any] = {
            key: reshape_from_space(value, space[key]) for key, value in tensor.items()
        }
        return reshaped_dict
    if isinstance(tensor, tuple):
        assert isinstance(space, spaces.Tuple), (
            f"Expected Tuple space for tuple tensor, got {type(space)}"
        )
        reshaped_tuple: tuple[Any, ...] = tuple(
            reshape_from_space(value, space[i]) for i, value in enumerate(tensor)
        )
        return reshaped_tuple

    msg = f"Unsupported tensor type: {type(tensor)}"
    raise TypeError(msg)


def concatenate_experiences_into_batches(
    experiences: dict[str, Any],
    space: spaces.Space,
    actions: bool = False,
) -> TorchObsType:
    """Reorganizes experiences into a batched tensor.

    Example input:
    {'agent_0': [[[...1], [...2]], [[...5], [...6]]],
        'agent_1': [[[...3], [...4]], [[...7], [...8]]]}

    Example output:
    torch.Tensor([...1], [...2], [...3], [...4], [...5], [...6], [...7], [...8])

    :param experiences: Dictionaries containing experiences indexed by agent_id that share a policy agent.
    :type experiences: dict[str, ObservationType]
    :param space: Observation/action/etc space to maintain
    :type space: spaces.Space
    :param actions: Whether the experiences are actions, defaults to False
    :type actions: bool, optional
    :return: Tensor, dict of tensors, or tuple of tensors of experiences, stacked along first dimension, with shape (num_experiences, *shape)
    :rtype: torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...]
    """
    tensors: list[TorchObsType] = []
    for agent_id in experiences:
        exp = experience_to_tensors(experiences[agent_id], space, actions)
        tensors.append(exp)

    stacked_tensor = concatenate_tensors(tensors)
    return reshape_from_space(stacked_tensor, space)


def is_peft_model(model: nn.Module) -> bool:
    """Check if a model is a PEFT model.

    :param model: Model to check
    :type model: nn.Module
    :return: True if the model is a PEFT model, False otherwise
    :rtype: bool
    """
    return isinstance(model, PeftModel)


def _rename_peft_primary_adapter_keys_in_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    old_adapter: str,
    new_adapter: str,
) -> dict[str, torch.Tensor]:
    """Rewrite state-dict keys when the primary PEFT adapter is renamed (e.g. to ``actor``)."""
    if old_adapter == new_adapter:
        return state_dict
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k.replace(f".{old_adapter}.", f".{new_adapter}.")
        nk = nk.replace(f"lora_{old_adapter}", f"lora_{new_adapter}")
        out[nk] = v
    return out


def clone_llm(
    original_model: PreTrainedModelType | DummyEvolvable,
    zero_stage: int | None,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> PreTrainedModelType:
    """Clone the actor.

    :param original_model: Model to clone
    :type original_model: PreTrainedModelType
    :param zero_stage: Zero stage to use, defaults to 0
    :type zero_stage: int | None, optional
    :param state_dict: State dict to load, defaults to None
    :type state_dict: dict[str, torch.Tensor] | None, optional
    :return: Cloned model
    """
    match original_model:
        case PeftModel() | PreTrainedModel():
            source_model = original_model
        case DummyEvolvable():
            # DummyEvolvable wraps an arbitrary module; the RL-clone path is only
            # reached with a pretrained model inside it.
            inner_model = original_model.module
            assert isinstance(inner_model, (PeftModel, PreTrainedModel))
            source_model = inner_model
        case _:
            msg = f"Invalid 'original_model' type: {type(original_model)}"
            raise ValueError(msg)
    model_config = source_model.config
    base_model = source_model.model
    assert isinstance(base_model, nn.Module)
    model: nn.Module = type(base_model)(model_config)
    adapter_names: list[str] = []

    # Any model carrying peft_config has adapters to copy, including
    # wrappers that are not PeftModel subclasses. The attribute is dynamic,
    # so pin the adapter-name/config pairs to their concrete peft types.
    if hasattr(source_model, "peft_config"):
        raw_peft_config = source_model.peft_config
        assert is_str_keyed_dict(raw_peft_config)
        peft_configs: dict[str, PeftConfig] = {
            name: config
            for name, config in raw_peft_config.items()
            if isinstance(config, PeftConfig)
        }
        adapter_names = list(peft_configs.keys())

        if len(adapter_names) > 1:
            warnings.warn(
                "Multiple adapters detected. Only the first adapter will be used for RL finetuning.",
                stacklevel=2,
            )
        # AgileRL standardizes on adapter name "actor" for the primary adapter.
        first_adapter = adapter_names[0]
        keep_adapter_base_dtype = zero_stage == 3
        model = get_peft_model(
            model,
            peft_configs[first_adapter],
            adapter_name="actor",
            autocast_adapter_dtype=not keep_adapter_base_dtype,
        )

        # Add remaining adapters using add_adapter
        for adapter_name in adapter_names[1:]:
            model.add_adapter(
                peft_config=peft_configs[adapter_name],
                adapter_name=adapter_name,
                autocast_adapter_dtype=not keep_adapter_base_dtype,
            )
        if keep_adapter_base_dtype:
            for name, param in model.named_parameters():
                if "lora" in name and param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)
        expert_targets = getattr(peft_configs[first_adapter], "target_parameters", None)
        if isinstance(expert_targets, (list, tuple)) and expert_targets:
            # Lazy import avoids a circular dependency with algorithms -> registry -> algo_utils.
            from agilerl.algorithms.core.llm_ops.moe_lora import (
                upgrade_moe_param_wrappers,
            )

            upgrade_moe_param_wrappers(model)
        model.disable_adapter()

    if state_dict is not None:
        sd = state_dict
        if adapter_names and adapter_names[0] != "actor":
            sd = _rename_peft_primary_adapter_keys_in_state_dict(
                sd,
                old_adapter=adapter_names[0],
                new_adapter="actor",
            )
        model.load_state_dict(sd, strict=False)
    return model


class DummyOptimizer:
    """Placeholder optimizer class to pass to the OptimizerWrapper when the optimizer is defined in the deepspeed config."""

    def __init__(self, params: list[torch.Tensor], **kwargs: Any) -> None:
        """Sentinel class to use for the optimizer when the optimizer is defined in the deepspeed config.

        :param params: Parameters to optimize.
        :type params: list[torch.Tensor]
        """

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> NoReturn:
        msg = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        raise RuntimeError(
            msg,
        )

    def zero_grad(self) -> NoReturn:
        msg = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        raise RuntimeError(
            msg,
        )

    def state_dict(self) -> NoReturn:
        msg = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        raise RuntimeError(
            msg,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> NoReturn:
        msg = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        raise RuntimeError(
            msg,
        )


def _match_action_ndims(
    reference: npt.NDArray, other: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Prepend singleton axes until continuous action arrays share the same ndim."""
    while other.ndim < reference.ndim:
        other = np.expand_dims(other, 0)
    while reference.ndim < other.ndim:
        reference = np.expand_dims(reference, 0)
    return reference, other


def _reconcile_shapes(
    reference: npt.NDArray, other: npt.NDArray, discrete_actions: bool
) -> tuple[npt.NDArray, npt.NDArray]:
    """Squeeze and broadcast `other` to match `reference` shape where possible.

    :param reference: Reference array to match shape to.
    :type reference: npt.NDArray
    :param other: Array to reconcile shape of.
    :type other: npt.NDArray
    :param discrete_actions: Whether the actions are discrete, defaults to False
    :type discrete_actions: bool, optional
    :return: Tuple of reconciled arrays.
    :rtype: tuple[npt.NDArray, npt.NDArray]
    """
    if reference.shape == other.shape:
        return reference, other

    if np.prod(other.shape) == np.prod(reference.shape):
        if discrete_actions:
            if other.ndim < reference.ndim:
                reference = reference.squeeze()
            else:
                other = other.squeeze()
        else:
            reference, other = _match_action_ndims(reference, other)

    return reference, np.broadcast_to(other, reference.shape)


def apply_env_defined_actions(
    agent_ids: list[str],
    action_dict: dict[str, npt.NDArray],
    env_defined_actions: dict[str, npt.NDArray],
    agent_masks: dict[str, npt.NDArray],
    discrete_actions: bool,
) -> dict[str, npt.NDArray]:
    """Apply env-defined actions to agent actions where the agent mask is True.

    :param agent_ids: Agent identifiers to process.
    :type agent_ids: list[str]
    :param action_dict: Mutable mapping of agent id → action array.
    :type action_dict: dict[str, npt.NDArray]
    :param env_defined_actions: Mapping of agent id → override action array.
    :type env_defined_actions: dict[str, npt.NDArray]
    :param agent_masks: Mapping of agent id → boolean mask array.
    :type agent_masks: dict[str, npt.NDArray]
    :param discrete_actions: Whether the actions are discrete, defaults to False
    :type discrete_actions: bool, optional
    :return: `action_dict` with overrides applied in-place.
    :rtype: dict[str, npt.NDArray]
    """
    for agent_id in agent_ids:
        action = action_dict[agent_id]
        override = env_defined_actions[agent_id]
        mask = agent_masks[agent_id]
        action, override = _reconcile_shapes(action, override, discrete_actions)
        action, mask = _reconcile_shapes(action, mask, discrete_actions)
        action[mask] = override[mask]
        action_dict[agent_id] = action
    return action_dict


def _resolve_lr(
    agent: "EvolvableAlgorithm", lr: str | tuple[str, str]
) -> tuple[Any, Any | None]:
    """Resolve the learning-rate value(s) from the agent attribute name(s).

    :param lr: Learning-rate attribute name, or a (actor, critic) pair of names
    :type lr: str | tuple[str, str]
    :return: Learning-rate value(s); the second element is None for a single name
    :rtype: tuple[Any, Any | None]
    """
    if isinstance(lr, tuple):
        return getattr(agent, lr[0]), getattr(agent, lr[1])
    return getattr(agent, lr), None


def inherit_init_signature(
    parent: type, fixed: set[str] | None = None
) -> Callable[[type], type]:
    """Class decorator giving a subclass its ``parent``'s ``__init__`` signature.

    A subclass that pins some of its parent's constructor arguments via
    ``*args``/``**kwargs`` loses its introspectable signature
    (``inspect.signature`` would just report ``(self, *args, **kwargs)``).
    AgileRL reads ``inspect.signature(agent.__init__).parameters`` to build the
    clone/checkpoint ``init_dict`` (see :class:`EvolvableAlgorithm`), so this
    restores the parent's real parameters — minus the ones the subclass fixes —
    on both the class and its ``__init__``.

    :param parent: Parent class whose ``__init__`` signature to inherit.
    :type parent: type
    :param fixed: Parameter names the subclass pins internally and therefore must
        not accept (excluded from the inherited signature), defaults to ``None``.
    :type fixed: set[str] | None, optional
    :return: A class decorator.
    :rtype: Callable[[type], type]
    """
    fixed = fixed or set()
    parent_sig = inspect.signature(parent.__init__)
    kept = [p for p in parent_sig.parameters.values() if p.name not in fixed]

    def decorate(cls: type) -> type:
        if "__init__" not in cls.__dict__:
            msg = (
                f"{cls.__name__} must define its own __init__ before "
                "@inherit_init_signature (otherwise the parent's signature is "
                "mutated)."
            )
            raise TypeError(msg)
        # inspect.signature(cls) is the constructor *call* — drop ``self``.
        cls.__signature__ = parent_sig.replace(  # ty: ignore[unresolved-attribute]  # metaprogramming: __signature__ is set dynamically on the class; typeshed does not model it
            parameters=[p for p in kept if p.name != "self"]
        )
        # inspect.signature(cls.__init__) is the *method* — keep ``self``. This
        # is the one EvolvableAlgorithm reads to build the clone/checkpoint dict.
        cls.__init__.__signature__ = parent_sig.replace(parameters=kept)  # ty: ignore[unresolved-attribute]  # metaprogramming: __signature__ is set dynamically on the function object; typeshed does not model it
        return cls

    return decorate
