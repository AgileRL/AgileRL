# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Class-level workarounds for a catalog Mamba2 mixer.

Every patch installs once at the class level and is idempotent. The mixer
class is resolved from a dotted path when a patch runs, not when this module
is imported: an absent target is a no-op with a warning, and a present class
with the wrong shape raises.
"""

from __future__ import annotations

import functools
import logging
import sys
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.utils.checkpoint
from torch.distributed.tensor import DTensor

from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.llm_packing import RESETS_AT_DOCUMENT_BOUNDARY, packed_seq_idx
from agilerl.utils.patching import class_is_patched, try_import

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from peft import PeftModel
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

__all__ = [
    "block_type_mask_mapping",
    "install_mamba_patches",
    "patch_nemotron_mamba_fused_path",
    "patch_nemotron_mamba_packed_sequences",
    "patch_nemotron_mamba_stream_ordering",
]

MEM_EFF_ATTR = "use_mem_eff_path"
EAGER_SCAN_CHUNK = 64

STREAM_PATCHED_FLAG = "_agilerl_mamba_stream_patched"
FUSED_PATH_PATCHED_FLAG = "_agilerl_mamba_fused_path_patched"
KERNEL_FULL_TENSOR_FLAG = "_agilerl_kernel_full_tensor"
MAMBA_TP_SHARD_ATTR = "agilerl_mamba_tp_shard"
SDPA_NAN_PATCHED_FLAG = "_agilerl_sdpa_nan_patched"
SDPA_NAME = "scaled_dot_product_attention"
RMSNORM_FN = "rmsnorm_fn"
SSM_KERNEL_NAMES = (
    "causal_conv1d_fn",
    "mamba_chunk_scan_combined",
    "mamba_split_conv1d_scan_combined",
)


def block_type_mask_mapping(
    module: object,
    embeds: torch.Tensor,
    attention_mask: torch.Tensor | dict[str, object] | None,
    past_key_values: object | None,
    position_ids: torch.Tensor | None,
) -> tuple[dict[str, object], torch.Tensor | None]:
    """HF keys for ``full_attention`` / ``linear_attention`` masks."""
    if position_ids is None:
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        past_seen = get_seq_length() if callable(get_seq_length) else 0
        position_ids = torch.arange(embeds.shape[1], device=embeds.device) + past_seen
        position_ids = position_ids.unsqueeze(0)
    masking = try_import("transformers.masking_utils")
    mask_kwargs = {
        "config": getattr(module, "config", None),
        "inputs_embeds": embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    mapping: dict[str, object] = {}
    if masking is not None:
        create_causal = getattr(masking, "create_causal_mask", None)
        create_linear = getattr(masking, "create_recurrent_attention_mask", None)
        if callable(create_causal):
            mapping["full_attention"] = create_causal(**mask_kwargs)
        if callable(create_linear):
            mapping["linear_attention"] = create_linear(**mask_kwargs)
    return mapping, position_ids


def _resolve_mixer_class(mixer: str) -> type | None:
    """Resolve the mixer class at ``mixer``, or None when its module is absent.

    :param mixer: Dotted path ``module.Class``.
    :type mixer: str
    :return: The mixer class, or None.
    :rtype: type | None
    """
    module_path, sep, class_name = mixer.rpartition(".")
    if not sep or not module_path or not class_name:
        message = f"[mamba] invalid mixer path {mixer!r}"
        raise RuntimeError(message)
    module = try_import(module_path)
    if module is None:
        return None
    mixer_cls = getattr(module, class_name, None)
    if mixer_cls is None:
        message = f"[mamba] {module_path} is present but missing {class_name}"
        raise RuntimeError(message)
    return mixer_cls


def _assigns_mem_eff_attr(original_init: Callable[..., None]) -> bool:
    """Whether the mixer's ``__init__`` sets the fused-path attribute.

    :param original_init: Mixer ``__init__`` to inspect.
    :type original_init: Callable[..., None]
    :return: True when the attribute name appears among the names it touches.
    :rtype: bool
    """
    code = getattr(original_init, "__code__", None)
    return MEM_EFF_ATTR in set(getattr(code, "co_names", ()))


def _ada_ssm_gpu(module: torch.nn.Module) -> bool:
    """Whether this mixer's weights sit on a GPU whose SSM kernels return NaN."""
    in_proj = cast("torch.nn.Linear", module.in_proj)
    return _ada_capability(in_proj.weight.device)


def _ada_capability(device: torch.device) -> bool:
    """Whether SSM kernels on this device return NaN."""
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device) == (8, 9)


def _forward_calls_rmsnorm(forward: object) -> bool:
    """Whether this forward calls the Triton gated RMSNorm."""
    code = getattr(forward, "__code__", None)
    if code is None:
        return False
    names = set(code.co_names)
    names.update(getattr(code, "co_freevars", ()))
    return RMSNORM_FN in names


def _gated_rms_pytorch(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """Gated RMSNorm in fp32. The gate is applied before the norm."""
    dtype = hidden_states.dtype
    values = hidden_states.float()
    if gate is not None:
        values = values * torch.nn.functional.silu(gate.float())
    grouped = values.reshape(*values.shape[:-1], -1, group_size)
    scale = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (grouped * scale).reshape(*values.shape) * weight.float()
    return normalized.to(dtype)


def _install_pytorch_gated_rms(module_cls: type) -> None:
    """Run this gated RMSNorm in PyTorch on compute capability 8.9."""
    cls: Any = module_cls
    forward = cls.__dict__.get("forward")
    announced = False

    @functools.wraps(forward)
    def pytorch_rms(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal announced
        if _ada_capability(cast("torch.Tensor", self.weight).device):
            if not announced:
                logger.debug(
                    "[mamba-fused-path] %s runs in PyTorch on compute capability 8.9",
                    type(self).__name__,
                )
                announced = True
            norm: Any = self
            return _gated_rms_pytorch(
                hidden_states,
                norm.weight,
                gate,
                norm.variance_epsilon,
                norm.group_size,
            )
        return forward(self, hidden_states, gate)

    cls.forward = pytorch_rms


def _patch_ada_gated_rms(model: PreTrainedModel | PeftModel) -> None:
    """Patch gated RMSNorm modules whose forward calls the Triton kernel."""
    patched: set[type] = set()
    for module in model.modules():
        module_cls = type(module)
        if module_cls in patched:
            continue
        forward = module_cls.__dict__.get("forward")
        if not _forward_calls_rmsnorm(forward):
            continue
        _install_pytorch_gated_rms(module_cls)
        patched.add(module_cls)


def _zero_padding_states(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Zero padded positions before the eager scan.

    ``torch_forward`` leaves those positions intact when the batch size is 1.

    :param args: Positional mixer arguments.
    :type args: tuple[Any, ...]
    :param kwargs: Keyword mixer arguments.
    :type kwargs: dict[str, Any]
    :return: Arguments with padded hidden states set to zero.
    :rtype: tuple[tuple[Any, ...], dict[str, Any]]
    """
    if not args or not isinstance(args[0], torch.Tensor):
        return args, kwargs
    mask = kwargs.get("attention_mask")
    if mask is None and len(args) >= 4:
        mask = args[3]
    hidden = args[0]
    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 2
        or mask.shape != hidden.shape[:2]
    ):
        return args, kwargs
    zeroed = hidden * mask.to(dtype=hidden.dtype)[:, :, None]
    return (zeroed, *args[1:]), kwargs


def _checkpointed_eager_scan(
    module: torch.nn.Module,
    eager: Callable[..., object],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    limit: object,
) -> object:
    """Recompute the eager scan with a smaller chunk size."""
    mixer: Any = module
    if not args or not isinstance(args[0], torch.Tensor):
        return eager(mixer, *args, **kwargs)
    hidden = args[0]
    rest = args[1:]
    clamped = mixer.time_step_limit

    def scan(hidden_states: torch.Tensor) -> object:
        # The recompute runs after the outer forward restores the limit.
        mixer.time_step_limit = clamped
        chunk = mixer.chunk_size
        if chunk > EAGER_SCAN_CHUNK:
            mixer.chunk_size = EAGER_SCAN_CHUNK
        try:
            return eager(mixer, hidden_states, *rest, **kwargs)
        finally:
            mixer.time_step_limit = limit
            mixer.chunk_size = chunk

    return torch.utils.checkpoint.checkpoint(scan, hidden, use_reentrant=False)


def _install_fused_scan(module_cls: type) -> None:
    """Gather SSM parameters and clamp an open dt limit for this mixer class."""
    if class_is_patched(module_cls, FUSED_PATH_PATCHED_FLAG):
        return
    _gather_ssm_kernel_parameters(module_cls)
    cls: Any = module_cls
    forward = cls.__dict__.get("forward")
    if forward is None:
        setattr(module_cls, FUSED_PATH_PATCHED_FLAG, True)
        return
    announced = False

    @functools.wraps(forward)
    def fused_scan(self: torch.nn.Module, *args: Any, **kwargs: Any) -> object:
        nonlocal announced
        # None and (0, inf) pass dt_limit=None into the fused kernel.
        mixer: Any = self
        limit = mixer.time_step_limit
        min_dt = mixer.time_step_min
        max_dt = mixer.time_step_max
        if (
            (limit is None or limit == (0.0, float("inf")))
            and min_dt is not None
            and max_dt is not None
        ):
            mixer.time_step_limit = (min_dt, max_dt)
        try:
            # SSM kernels on compute capability 8.9 return NaN.
            eager = type(self).__dict__.get("torch_forward")
            if _ada_ssm_gpu(self) and callable(eager):
                if not announced:
                    logger.debug(
                        "[mamba-fused-path] %s forward uses torch_forward "
                        "on compute capability 8.9",
                        type(self).__name__,
                    )
                    announced = True
                if "hidden_states" in kwargs:
                    args = (kwargs.pop("hidden_states"), *args)
                args, kwargs = _zero_padding_states(args, kwargs)
                return _checkpointed_eager_scan(mixer, eager, args, kwargs, limit)
            return forward(self, *args, **kwargs)
        finally:
            mixer.time_step_limit = limit

    cls.forward = fused_scan
    setattr(module_cls, FUSED_PATH_PATCHED_FLAG, True)
    logger.debug(
        "[mamba-fused-path] %s forward uses the fused scan with gathered parameters",
        module_cls.__name__,
    )


def _forward_calls_fused_scan(forward: object) -> bool:
    code = getattr(forward, "__code__", None)
    names = set(getattr(code, "co_names", ()))
    names.update(getattr(code, "co_freevars", ()))
    return "mamba_split_conv1d_scan_combined" in names


def _class_calls_fused_scan(module_cls: type) -> bool:
    for cls in module_cls.__mro__:
        for attr in cls.__dict__.values():
            if _forward_calls_fused_scan(attr):
                return True
    return False


def mark_mamba_tp_shard(param: torch.Tensor) -> None:
    """Mark a tensor-parallel mixer parameter so SSM kernels read its local shard.

    :param param: Mixer parameter placed on the expert-parallel mesh.
    :type param: torch.Tensor
    :return: None
    :rtype: None
    """
    setattr(param, MAMBA_TP_SHARD_ATTR, True)


def _full_parameter_tensor(value: object) -> object:
    """Local tensor for a tensor-parallel mixer shard; gathered tensor otherwise."""
    if isinstance(value, DTensor):
        if getattr(value, MAMBA_TP_SHARD_ATTR, False):
            return value.to_local()
        return value.full_tensor()
    return value


def _ssm_scan_in_fp32(value: object) -> bool:
    """Ada bf16 SSM kernels return NaN, so those inputs run in fp32."""
    if not isinstance(value, torch.Tensor):
        return False
    if not value.is_floating_point() or value.dtype == torch.float32:
        return False
    return _ada_capability(value.device)


def _cast_floating_to(value: object, dtype: torch.dtype) -> object:
    """Cast floating tensors in *value* back to *dtype*."""
    if torch.is_tensor(value) and value.is_floating_point() and value.dtype != dtype:
        return value.to(dtype)
    if isinstance(value, tuple):
        return tuple(_cast_floating_to(item, dtype) for item in value)
    return value


def _call_kernel_with_full_tensors(kernel: object) -> object:
    """Gather sharded arguments before *kernel* reads them by pointer."""
    announced = False

    def gathered(*args: object, **kwargs: object) -> object:
        nonlocal announced
        call: Any = kernel
        dtype: torch.dtype | None = None

        def prepare(value: object) -> object:
            nonlocal dtype
            tensor = _full_parameter_tensor(value)
            if not torch.is_tensor(tensor):
                return tensor
            tensor = tensor.contiguous()
            if _ssm_scan_in_fp32(tensor):
                if dtype is None:
                    dtype = tensor.dtype
                tensor = tensor.float()
            return tensor

        result = call(
            *(prepare(arg) for arg in args),
            **{key: prepare(val) for key, val in kwargs.items()},
        )
        if dtype is None:
            return result
        if not announced:
            logger.debug(
                "[mamba-fused-path] SSM kernels run floating inputs in fp32 on this GPU"
            )
            announced = True
        return _cast_floating_to(result, dtype)

    setattr(gathered, KERNEL_FULL_TENSOR_FLAG, True)
    return gathered


def _gather_ssm_kernel_parameters(module_cls: type) -> None:
    """Make SSM kernels in this mixer's module read gathered parameters."""
    module = sys.modules.get(module_cls.__module__)
    if module is None:
        return
    for name in SSM_KERNEL_NAMES:
        if name not in vars(module):
            continue
        kernel = vars(module)[name]
        if kernel is None:
            # Missing optional dependency; the module leaves the name as None.
            continue
        if getattr(kernel, KERNEL_FULL_TENSOR_FLAG, False):
            continue
        setattr(module, name, _call_kernel_with_full_tensors(kernel))


def _forward_calls_sdpa(fn: object) -> bool:
    """Whether *fn* calls scaled-dot-product attention."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return False
    names = set(code.co_names)
    names.update(getattr(code, "co_freevars", ()))
    return SDPA_NAME in names


def patch_sdpa_fully_masked_rows(model: PreTrainedModel | PeftModel) -> None:
    """Replace NaN attention outputs from fully masked SDPA rows.

    A query row whose mask is all ``-inf`` makes
    ``scaled_dot_product_attention`` return NaN, and that NaN spreads through
    the rest of the block.

    :param model: Model whose attention modules are swept.
    :type model: PreTrainedModel | PeftModel
    :return: None
    :rtype: None
    """
    patched: set[type] = set()
    for module in model.modules():
        module_cls = type(module)
        if module_cls in patched or class_is_patched(module_cls, SDPA_NAN_PATCHED_FLAG):
            continue
        forward = module_cls.__dict__.get("forward")
        if not _forward_calls_sdpa(forward):
            continue
        _patch_sdpa_forward(module_cls)
        patched.add(module_cls)


def _patch_sdpa_forward(module_cls: type) -> None:
    """Zero NaN values in this attention class's forward output."""
    cls: Any = module_cls
    forward = cls.forward

    @functools.wraps(forward)
    def finite_forward(self: torch.nn.Module, *args: Any, **kwargs: Any) -> object:
        output = forward(self, *args, **kwargs)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (torch.nan_to_num(output[0]), *output[1:])
        if isinstance(output, torch.Tensor):
            return torch.nan_to_num(output)
        return output

    target: Any = module_cls
    target.forward = finite_forward
    setattr(module_cls, SDPA_NAN_PATCHED_FLAG, True)
    logger.debug(
        "[sdpa-mask] %s forward replaces NaN rows from a fully masked attention",
        module_cls.__name__,
    )


def _keep_fused_scan_on_instances(
    mixer_cls: type,
    model: PreTrainedModel | PeftModel,
) -> int:
    """Install the fused-scan wrapper on mixer classes already built in *model*.

    A same-name class is included when it sets the fused-path attribute or its
    methods call the fused scan. An open dt limit is replaced with
    ``time_step_min`` and ``time_step_max`` for that forward. A same-name class
    with neither is left alone.

    :param mixer_cls: The patched mixer class.
    :type mixer_cls: type
    :param model: Model whose submodules are swept.
    :type model: PreTrainedModel | PeftModel
    :return: Number of mixer instances kept on the fused scan.
    :rtype: int
    """
    kept = 0
    patched_remote: set[type] = set()
    for module in model.modules():
        module_cls = type(module)
        if isinstance(module, mixer_cls):
            kept += 1
            continue
        if module_cls.__name__ != mixer_cls.__name__:
            continue
        init = module_cls.__dict__.get("__init__")
        assigns = init is not None and _assigns_mem_eff_attr(init)
        calls_fused_scan = _class_calls_fused_scan(module_cls)
        if not hasattr(module, MEM_EFF_ATTR) and not assigns and not calls_fused_scan:
            continue
        if module_cls not in patched_remote:
            _install_fused_scan(module_cls)
            patched_remote.add(module_cls)
        kept += 1
    return kept


def patch_nemotron_mamba_fused_path(
    mixer: str,
    enabled: bool = True,
    model: PreTrainedModel | PeftModel | None = None,
) -> None:
    """Keep the fused SSM kernel and make its parameter reads safe.

    Training calls ``mamba_split_conv1d_scan_combined``. Sharded parameters are
    gathered before that kernel reads them. An open ``time_step_limit``
    (``None`` or ``(0, inf)``) is replaced with ``time_step_min`` and
    ``time_step_max`` for the forward. On compute capability 8.9 the forward
    uses ``torch_forward`` because the SSM kernels return NaN, and the gated
    RMSNorm runs in PyTorch. ``conv1d`` and ``out_proj`` stay inside the kernel
    on other GPUs; LoRA does not target them.

    :param mixer: Dotted path of the mixer class to patch.
    :type mixer: str
    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
    :param model: Already-built model whose mixers are also wrapped,
        defaults to None.
    :type model: PreTrainedModel | PeftModel | None, optional
    :return: None
    :rtype: None
    """
    if not enabled:
        logger.info("[mamba-fused-path] disabled by caller; forward left unpatched")
        return

    mixer_cls = _resolve_mixer_class(mixer)
    if mixer_cls is None:
        logger.warning(
            "[mamba-fused-path] %s unavailable; forward left unpatched",
            mixer.rpartition(".")[2],
        )
        return

    _install_fused_scan(mixer_cls)

    if model is None:
        return
    _patch_ada_gated_rms(model)
    kept = _keep_fused_scan_on_instances(mixer_cls, model)
    if kept:
        logger.debug(
            "[mamba-fused-path] fused scan kept on %d mixers",
            kept,
        )


def _is_cuda_tensor(value: object) -> bool:
    """Whether *value* is a CUDA tensor the caching allocator can track.

    :param value: Candidate tensor.
    :type value: object
    :return: True when the value is on CUDA and supports ``record_stream``.
    :rtype: bool
    """
    return bool(getattr(value, "is_cuda", False)) and hasattr(value, "record_stream")


def _cuda_tensors(value: object) -> Iterator[Any]:
    """CUDA tensors held directly by *value*.

    :param value: Tensor, or a tuple or list that may contain tensors.
    :type value: object
    :return: Iterator over the CUDA tensors found.
    :rtype: Iterator[Any]
    """
    if _is_cuda_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            if _is_cuda_tensor(item):
                yield item


def _record_streams(value: object, *streams: Any) -> None:
    """Record *value*'s CUDA tensors against every stream that touches them.

    :param value: Tensor, or a tuple or list that may contain tensors.
    :type value: object
    :param streams: Streams to record against.
    :type streams: Any
    :return: None
    :rtype: None
    """
    for tensor in _cuda_tensors(value):
        for stream in streams:
            tensor.record_stream(stream)


def _make_patched_forward(
    original_forward: Callable[..., Any],
) -> Callable[..., Any]:
    """Build the mixer ``forward`` wrapper that joins the caller and default streams.

    :param original_forward: Mixer ``forward`` to call through to.
    :type original_forward: Callable[..., Any]
    :return: Replacement ``forward``.
    :rtype: Callable[..., Any]
    """

    @functools.wraps(original_forward)
    def patched_forward(
        self: object,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        if not torch.cuda.is_available() or not _is_cuda_tensor(hidden_states):
            call_args, kwargs = _zero_padding_states((hidden_states, *args), kwargs)
            return original_forward(self, *call_args, **kwargs)

        device = hidden_states.device
        current = torch.cuda.current_stream(device)
        default = torch.cuda.default_stream(device)
        call_args, kwargs = _zero_padding_states((hidden_states, *args), kwargs)
        hidden_states, args = call_args[0], call_args[1:]
        if current == default:
            return original_forward(self, hidden_states, *args, **kwargs)

        default.wait_stream(current)
        _record_streams(hidden_states, default)
        outputs = original_forward(self, hidden_states, *args, **kwargs)
        current.wait_stream(default)
        _record_streams(outputs, default, current)
        return outputs

    return patched_forward


def patch_nemotron_mamba_stream_ordering(
    mixer: str,
    enabled: bool = True,
    model: PreTrainedModel | PeftModel | None = None,
) -> None:
    """Order the Mamba2 mixer's default-stream kernels against its caller.

    The mixer's ``forward`` runs the mamba and causal-conv1d kernels inside
    ``torch.cuda.stream(default_stream)``. The block enters that stream before
    the mixer runs. Parameter all-gathers complete on the stream that was
    current when the fetch was issued, so when that stream is not the default
    one the kernels carry no dependency on it and can read a parameter buffer
    that is still being filled. The block wrapper makes the default stream
    wait on the current stream before that switch. The mixer wrapper does the
    same when it is entered on a non-default stream, and makes the current
    stream wait on the default stream afterwards so downstream compute sees
    finished results.
    Inputs and outputs are recorded against both streams so the caching
    allocator keeps their blocks reserved until every stream that touches them
    is done. When the caller is already on the default stream the ordering is
    redundant and the wrapper calls straight through. The first CUDA call logs
    whether the caller stream matched the default stream. A same-name mixer
    class already built on ``model`` gets the same wrapper.

    :param mixer: Dotted path of the mixer class to patch.
    :type mixer: str
    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
    :param model: Already-built model whose same-name mixer classes are also
        wrapped, defaults to None.
    :type model: PreTrainedModel | PeftModel | None, optional
    :return: None
    :rtype: None
    """
    if not enabled:
        logger.info("[mamba-stream] disabled by caller; forward left unpatched")
        return

    mixer_cls = _resolve_mixer_class(mixer)
    if mixer_cls is None:
        logger.warning(
            "[mamba-stream] %s unavailable; forward left unpatched",
            mixer.rpartition(".")[2],
        )
        return
    _patch_stream_class(mixer_cls)
    if model is None:
        return
    _patch_default_stream_blocks(model)
    seen: set[type] = set()
    for module in model.modules():
        module_cls = type(module)
        if module_cls is mixer_cls or module_cls.__name__ != mixer_cls.__name__:
            continue
        if module_cls in seen:
            continue
        seen.add(module_cls)
        if "forward" not in module_cls.__dict__:
            continue
        _patch_stream_class(module_cls)


def _forward_uses_default_stream(forward: object) -> bool:
    """Whether this forward enters the default CUDA stream."""
    code = getattr(forward, "__code__", None)
    if code is None:
        return False
    names = set(code.co_names)
    names.update(getattr(code, "co_freevars", ()))
    return "default_stream" in names


def _patch_default_stream_blocks(model: PreTrainedModel | PeftModel) -> None:
    """Wait for the caller stream before a block switches to the default stream."""
    seen: set[type] = set()
    for module in model.modules():
        module_cls = type(module)
        if module_cls in seen:
            continue
        seen.add(module_cls)
        forward = module_cls.__dict__.get("forward")
        if not _forward_uses_default_stream(forward):
            continue
        _patch_stream_class(module_cls)


def _patch_stream_class(mixer_cls: type) -> None:
    """Wrap this mixer's ``forward`` so default-stream kernels wait on the caller."""
    if class_is_patched(mixer_cls, STREAM_PATCHED_FLAG):
        return
    original_forward = mixer_cls.__dict__.get("forward")
    if original_forward is None:
        message = f"[mamba-stream] {mixer_cls.__name__} lacks forward"
        raise RuntimeError(message)
    mixer_target: Any = mixer_cls
    mixer_target.forward = _make_patched_forward(original_forward)
    setattr(mixer_cls, STREAM_PATCHED_FLAG, True)
    logger.info(
        "[mamba-stream] %s.forward patched; default-stream kernels ordered "
        "against the calling stream",
        mixer_cls.__name__,
    )


def _per_document(
    run: Callable[[torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,
    seq_idx: torch.Tensor,
) -> torch.Tensor:
    """Run *run* on each packed document of each row on its own and rejoin the rows.

    :param run: Mixer forward over one ``(1, L_doc, H)`` document.
    :type run: Callable[[torch.Tensor], torch.Tensor]
    :param hidden_states: ``(B, L, H)`` packed rows.
    :type hidden_states: torch.Tensor
    :param seq_idx: ``(B, L)`` document index per token.
    :type seq_idx: torch.Tensor
    :return: ``(B, L, H)`` outputs.
    :rtype: torch.Tensor
    """
    rows = []
    for row, row_seq_idx in zip(hidden_states.split(1), seq_idx, strict=True):
        lengths = torch.unique_consecutive(row_seq_idx, return_counts=True)[1]
        documents = row.split(lengths.tolist(), dim=1)
        rows.append(torch.cat([run(document) for document in documents], dim=1))
    return torch.cat(rows)


def _packed_split_scan(
    mixer: torch.nn.Module,
    kernel: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    seq_idx: torch.Tensor,
) -> torch.Tensor:
    """Fused Mamba2 forward whose conv and scan restart wherever ``seq_idx`` changes.

    :param mixer: Mamba2 mixer.
    :type mixer: torch.nn.Module
    :param kernel: ``mamba_split_conv1d_scan_combined`` from the mixer's module.
    :type kernel: Callable[..., torch.Tensor]
    :param hidden_states: ``(B, L, H)`` packed rows.
    :type hidden_states: torch.Tensor
    :param seq_idx: ``(B, L)`` int32 document index per token.
    :type seq_idx: torch.Tensor
    :return: ``(B, L, H)`` mixer output.
    :rtype: torch.Tensor
    """
    module: Any = mixer
    limit = module.time_step_limit
    dt_limit = {} if limit is None else {"dt_limit": limit}
    return kernel(
        module.in_proj(hidden_states),
        module.conv1d.weight.squeeze(1),
        module.conv1d.bias,
        module.dt_bias,
        -torch.exp(module.A_log.float()),
        D=module.D,
        chunk_size=module.chunk_size,
        seq_idx=seq_idx,
        activation=module.activation,
        rmsnorm_weight=module.norm.weight,
        rmsnorm_eps=module.norm.variance_epsilon,
        outproj_weight=module.out_proj.weight,
        outproj_bias=module.out_proj.bias,
        headdim=module.head_dim,
        ngroups=module.n_groups,
        norm_before_gate=False,
        **dt_limit,
    )


def _install_packed_mixer(mixer_cls: type) -> None:
    """Give this mixer's forward a ``seq_idx`` argument that resets state per document."""
    cls: Any = mixer_cls
    forward = cls.__dict__["forward"]
    torch_forward = cls.__dict__["torch_forward"]
    cuda_kernels_forward = cls.__dict__["cuda_kernels_forward"]
    # Mixer ``__init__`` assigns these module globals, so read them per call.
    kernels = vars(sys.modules[mixer_cls.__module__])

    @functools.wraps(torch_forward)
    def packed_torch_forward(
        self: torch.nn.Module,
        input_states: torch.Tensor,
        cache_params: object | None = None,
        attention_mask: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seq_idx is None:
            return torch_forward(self, input_states, cache_params, attention_mask)
        return _per_document(
            lambda document: torch_forward(self, document), input_states, seq_idx
        )

    @functools.wraps(cuda_kernels_forward)
    def packed_cuda_kernels_forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        cache_params: object | None = None,
        attention_mask: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seq_idx is None:
            return cuda_kernels_forward(
                self, hidden_states, cache_params, attention_mask
            )
        kernel = kernels["mamba_split_conv1d_scan_combined"]
        return _packed_split_scan(self, kernel, hidden_states, seq_idx)

    @functools.wraps(forward)
    def packed_forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        cache_params: object | None = None,
        attention_mask: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if seq_idx is None:
            return forward(self, hidden_states, cache_params, attention_mask, **kwargs)
        mixer: Any = self
        if (
            kernels["is_fast_path_available"]
            and mixer.in_proj.weight.device.type == "cuda"
            and not torch.compiler.is_compiling()
        ):
            with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
                return mixer.cuda_kernels_forward(hidden_states, seq_idx=seq_idx)
        return mixer.torch_forward(hidden_states, seq_idx=seq_idx)

    cls.torch_forward = packed_torch_forward
    cls.cuda_kernels_forward = packed_cuda_kernels_forward
    cls.forward = packed_forward


def _install_packed_block(block_cls: type, mixer_cls: type) -> None:
    """Pass a packed row's ``seq_idx`` from this block's ``position_ids`` to its mixer."""
    cls: Any = block_cls
    forward = cls.__dict__["forward"]

    @functools.wraps(forward)
    def packed_block_forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        past_key_values: object | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        block: Any = self
        seq_idx = None
        # A padding mask or a cache means the row is not packed.
        if (
            isinstance(block.mixer, mixer_cls)
            and attention_mask is None
            and past_key_values is None
            and position_ids is not None
        ):
            seq_idx = packed_seq_idx(position_ids)
        if seq_idx is None:
            return forward(
                self,
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
            )
        residual = hidden_states
        normed = block.norm(hidden_states.to(dtype=block.norm.weight.dtype))
        return residual + block.mixer(normed, seq_idx=seq_idx)

    cls.forward = packed_block_forward


def patch_nemotron_mamba_packed_sequences(mixer: str, block: str) -> None:
    """Reset the mixer's conv and scan state at every packed-document boundary.

    The block derives ``seq_idx`` from ``position_ids`` (a document starts
    wherever positions do not step by one) and passes it to its mixer. On CUDA
    the fused kernel takes ``seq_idx`` directly; ``torch_forward`` runs each
    document on its own. Unpacked rows take the unpatched path. The mixer class
    is then marked with
    :data:`~agilerl.utils.llm_packing.RESETS_AT_DOCUMENT_BOUNDARY`.

    :param mixer: Dotted path of the mixer class.
    :type mixer: str
    :param block: Dotted path of the decoder block class that calls the mixer.
    :type block: str
    :return: None
    :rtype: None
    """
    mixer_cls = _resolve_mixer_class(mixer)
    block_cls = _resolve_mixer_class(block)
    if mixer_cls is None or block_cls is None:
        logger.warning(
            "[mamba-packed] %s unavailable; packed rows are not supported",
            mixer.rpartition(".")[2],
        )
        return
    if class_is_patched(mixer_cls, RESETS_AT_DOCUMENT_BOUNDARY):
        return
    _install_packed_mixer(mixer_cls)
    _install_packed_block(block_cls, mixer_cls)
    setattr(mixer_cls, RESETS_AT_DOCUMENT_BOUNDARY, True)
    logger.debug(
        "[mamba-packed] %s resets state at packed-document boundaries",
        mixer_cls.__name__,
    )


def install_mamba_patches(
    patch: PatchRuntimeConfig,
    model: PreTrainedModel | PeftModel | None = None,
) -> None:
    """Install catalog Mamba2 mixer workarounds.

    ``block`` routes packed-row ``seq_idx`` into the mixer so its state resets
    at each document boundary.
    ``fused_path`` gathers SSM kernel parameters and clamps an open dt limit.
    ``stream_ordering``
    wraps mixer ``forward`` so default-stream scan/conv kernels wait on the
    caller stream. Those kernels launch on the default stream, so a caller on
    another stream can race with a parameter all-gather. When ``model`` is
    set, attention modules whose forward calls scaled-dot-product attention
    replace NaN rows from a fully masked mask.

    :param patch: Catalog patch config; mamba flags and mixer class path.
    :type patch: PatchRuntimeConfig
    :param model: Already-built model the patches also apply to, or None.
    :type model: PreTrainedModel | PeftModel | None
    :return: None
    :rtype: None
    """
    if patch.mamba is None:
        return
    # Installed first so the fused-path and stream wrappers wrap the packed forward.
    if patch.mamba.block is not None:
        patch_nemotron_mamba_packed_sequences(
            mixer=patch.mamba.mixer, block=patch.mamba.block
        )
    if patch.mamba.fused_path:
        patch_nemotron_mamba_fused_path(mixer=patch.mamba.mixer, model=model)
    if patch.mamba.stream_ordering:
        patch_nemotron_mamba_stream_ordering(mixer=patch.mamba.mixer, model=model)
    if model is not None:
        patch_sdpa_fully_masked_rows(model)
