# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Class-level workarounds for a catalog Mamba2 mixer.

Both patches install once at the class level, are idempotent, and take
``enabled`` so a caller can turn them off. The mixer class is resolved from a
dotted path when a patch runs, not when this module is imported: an absent
target is a no-op with a warning, and a present class with the wrong shape
raises.
"""

from __future__ import annotations

import functools
import logging
import sys
from typing import TYPE_CHECKING, Any

import torch

from agilerl.architectures.runtime import PatchRuntimeConfig
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
    "patch_nemotron_mamba_stream_ordering",
]

MEM_EFF_ATTR = "use_mem_eff_path"

STREAM_PATCHED_FLAG = "_agilerl_mamba_stream_patched"
FUSED_PATH_PATCHED_FLAG = "_agilerl_mamba_fused_path_patched"
KERNEL_FULL_TENSOR_FLAG = "_agilerl_kernel_full_tensor"
SDPA_NAN_PATCHED_FLAG = "_agilerl_sdpa_nan_patched"
SDPA_NAME = "scaled_dot_product_attention"
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

    @functools.wraps(forward)
    def fused_scan(self: torch.nn.Module, *args: Any, **kwargs: Any) -> object:
        # None and (0, inf) pass dt_limit=None into the fused kernel.
        limit = getattr(self, "time_step_limit", None)
        min_dt = getattr(self, "time_step_min", None)
        max_dt = getattr(self, "time_step_max", None)
        mixer: Any = self
        if (
            (limit is None or limit == (0.0, float("inf")))
            and min_dt is not None
            and max_dt is not None
        ):
            mixer.time_step_limit = (min_dt, max_dt)
        try:
            return forward(self, *args, **kwargs)
        finally:
            if hasattr(mixer, "time_step_limit"):
                mixer.time_step_limit = limit

    cls.forward = fused_scan
    setattr(module_cls, FUSED_PATH_PATCHED_FLAG, True)
    logger.info(
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


def _full_parameter_tensor(value: object) -> object:
    """Return the gathered tensor when *value* is an FSDP ``DTensor``."""
    if type(value).__name__ != "DTensor":
        return value
    full = getattr(value, "full_tensor", None)
    if not callable(full):
        return value
    return full()


def _ssm_scan_in_fp32(value: object) -> bool:
    """Ada bf16 SSM kernels return NaN, so those inputs run in fp32."""
    device = getattr(value, "device", None)
    if getattr(device, "type", None) != "cuda":
        return False
    is_floating = getattr(value, "is_floating_point", None)
    if not callable(is_floating) or not is_floating():
        return False
    if getattr(value, "dtype", None) == torch.float32:
        return False
    capability = torch.cuda.get_device_capability(device)
    return capability == (8, 9)


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
            logger.info(
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
        kernel = getattr(module, name, None)
        if kernel is None or getattr(kernel, KERNEL_FULL_TENSOR_FLAG, False):
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
        if module_cls in patched or getattr(module_cls, SDPA_NAN_PATCHED_FLAG, False):
            continue
        forward = getattr(module_cls, "forward", None)
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
    logger.info(
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
        init = getattr(module_cls, "__init__", None)
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
    ``time_step_max`` for the forward. On compute capability 8.9, floating
    kernel inputs run in fp32. ``conv1d`` and ``out_proj`` stay inside the
    kernel; LoRA does not target them.

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
    kept = _keep_fused_scan_on_instances(mixer_cls, model)
    if kept:
        logger.info(
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
            return original_forward(self, hidden_states, *args, **kwargs)

        device = hidden_states.device
        current = torch.cuda.current_stream(device)
        default = torch.cuda.default_stream(device)
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
    ``torch.cuda.stream(default_stream)``. Parameter all-gathers complete
    on the stream that was current when the fetch was issued, so when that
    stream is not the default one the kernels carry no dependency on it and
    can read a parameter buffer that is still being filled. The wrapper makes
    the default stream wait on the current stream before the call, which
    transitively covers the all-gather, and makes the current stream wait on the
    default stream afterwards so downstream compute sees finished results.
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
    seen: set[type] = set()
    for module in model.modules():
        module_cls = type(module)
        if module_cls is mixer_cls or module_cls.__name__ != mixer_cls.__name__:
            continue
        if module_cls in seen:
            continue
        seen.add(module_cls)
        if getattr(module_cls, "forward", None) is None:
            continue
        _patch_stream_class(module_cls)


def _patch_stream_class(mixer_cls: type) -> None:
    """Wrap this mixer's ``forward`` so default-stream kernels wait on the caller."""
    if class_is_patched(mixer_cls, STREAM_PATCHED_FLAG):
        return
    original_forward = getattr(mixer_cls, "forward", None)
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


def install_mamba_patches(
    patch: PatchRuntimeConfig,
    model: PreTrainedModel | PeftModel | None = None,
) -> None:
    """Install catalog Mamba2 mixer workarounds.

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
    if patch.mamba.fused_path:
        patch_nemotron_mamba_fused_path(mixer=patch.mamba.mixer, model=model)
    if patch.mamba.stream_ordering:
        patch_nemotron_mamba_stream_ordering(mixer=patch.mamba.mixer, model=model)
    if model is not None:
        patch_sdpa_fully_masked_rows(model)
