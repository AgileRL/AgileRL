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
    "install_mamba_patches",
    "patch_nemotron_mamba_fused_path",
    "patch_nemotron_mamba_stream_ordering",
]

MEM_EFF_ATTR = "use_mem_eff_path"

STREAM_PATCHED_FLAG = "_agilerl_mamba_stream_patched"
FUSED_PATH_PATCHED_FLAG = "_agilerl_mamba_fused_path_patched"


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


def _make_patched_init(original_init: Callable[..., None]) -> Callable[..., None]:
    """Build the ``__init__`` wrapper that clears the fused-path attribute.

    :param original_init: Mixer ``__init__`` to call through to.
    :type original_init: Callable[..., None]
    :return: Replacement ``__init__``.
    :rtype: Callable[..., None]
    """

    @functools.wraps(original_init)
    def patched_init(self: object, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        setattr(self, MEM_EFF_ATTR, False)

    return patched_init


def _drop_fused_path_on_instances(
    mixer_cls: type,
    model: PreTrainedModel | PeftModel,
) -> int:
    """Clear the fused-path attribute on every existing mixer in *model*.

    Raises on a module whose class only shares the mixer's name (e.g. a
    ``trust_remote_code`` copy) — an instance no patch can reach.

    :param mixer_cls: The patched mixer class.
    :type mixer_cls: type
    :param model: Model whose submodules are swept.
    :type model: PreTrainedModel | PeftModel
    :return: Number of mixer instances cleared.
    :rtype: int
    """
    cleared = 0
    for module in model.modules():
        if isinstance(module, mixer_cls):
            setattr(module, MEM_EFF_ATTR, False)
            cleared += 1
        elif type(module).__name__ == mixer_cls.__name__:
            message = (
                f"[mamba-fused-path] {type(module).__module__}."
                f"{type(module).__qualname__} is not the patched "
                f"{mixer_cls.__module__}.{mixer_cls.__qualname__}; the model "
                "was built from a different mixer class (e.g. via "
                "trust_remote_code), so the fused-path patch cannot reach it"
            )
            raise RuntimeError(message)
    return cleared


def patch_nemotron_mamba_fused_path(
    *,
    mixer: str,
    enabled: bool = True,
    model: PreTrainedModel | PeftModel | None = None,
) -> None:
    """Keep every catalog Mamba2 mixer on its decomposed forward path.

    The mixer's ``cuda_kernels_forward`` takes a fused branch when
    ``use_mem_eff_path`` is set and the batch is unpadded and the mixer is in
    training mode. That branch hands ``conv1d.weight``, ``conv1d.bias``,
    ``norm.weight``, ``out_proj.weight`` and ``out_proj.bias`` to
    ``mamba_split_conv1d_scan_combined`` as raw tensors, so those submodules are
    never called: their ZeRO-3 pre-forward gather hooks do not fire, leaving the
    parameters to deepspeed's residency-dependent fallback all-gather, which
    ranks can disagree about and deadlock on; the set of traced submodules also
    shifts with training mode and per-rank padding; and the LoRA delta on
    ``out_proj`` is dropped because the kernel reads the base weight. Clearing
    the attribute on every instance as it is constructed removes the branch, so
    ``self.norm`` and ``self.out_proj`` run as modules. This wraps ``__init__``,
    so it only covers mixers built afterwards; pass ``model`` to sweep mixers
    that already exist.

    :param mixer: Dotted path of the mixer class to patch.
    :type mixer: str
    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
    :param model: Already-built model whose mixers are also cleared,
        defaults to None.
    :type model: PreTrainedModel | PeftModel | None, optional
    :return: None
    :rtype: None
    """
    if not enabled:
        logger.info("[mamba-fused-path] disabled by caller; __init__ left unpatched")
        return

    mixer_cls = _resolve_mixer_class(mixer)
    if mixer_cls is None:
        logger.warning(
            "[mamba-fused-path] %s unavailable; __init__ left unpatched",
            mixer.rpartition(".")[2],
        )
        return

    if not class_is_patched(mixer_cls, FUSED_PATH_PATCHED_FLAG):
        original_init = getattr(mixer_cls, "__init__", None)
        if original_init is None or not _assigns_mem_eff_attr(original_init):
            message = (
                f"[mamba-fused-path] {mixer_cls.__name__}.__init__ does not set "
                f"{MEM_EFF_ATTR}"
            )
            raise RuntimeError(message)

        mixer_target: Any = mixer_cls
        mixer_target.__init__ = _make_patched_init(original_init)
        setattr(mixer_cls, FUSED_PATH_PATCHED_FLAG, True)
        logger.info(
            "[mamba-fused-path] %s.__init__ patched; %s cleared on every mixer",
            mixer_cls.__name__,
            MEM_EFF_ATTR,
        )

    if model is None:
        return
    cleared = _drop_fused_path_on_instances(mixer_cls, model)
    if cleared:
        logger.info(
            "[mamba-fused-path] %s cleared on %d existing mixers",
            MEM_EFF_ATTR,
            cleared,
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
    *,
    mixer: str,
    enabled: bool = True,
    model: PreTrainedModel | PeftModel | None = None,
) -> None:
    """Order the Mamba2 mixer's default-stream kernels against its caller.

    The mixer's ``forward`` runs the mamba and causal-conv1d kernels inside
    ``torch.cuda.stream(default_stream)``. A ZeRO-3 parameter all-gather
    completes on the stream that was current when the fetch was issued, so when
    that stream is not the default one the kernels carry no dependency on it and
    can read a parameter buffer that is still being filled. The wrapper makes
    the default stream wait on the current stream before the call, which
    transitively covers the all-gather, and makes the current stream wait on the
    default stream afterwards so downstream compute sees finished results.
    Inputs and outputs are recorded against both streams so the caching
    allocator keeps their blocks reserved until every stream that touches them
    is done. When the caller is already on the default stream the ordering is
    redundant and the wrapper calls straight through. The first CUDA call logs
    whether the caller stream matched the default stream.

    :param mixer: Dotted path of the mixer class to patch.
    :type mixer: str
    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
    :param model: Unused; accepted for family-dispatch parity, defaults to None.
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
    *,
    model: PreTrainedModel | PeftModel | None = None,
) -> None:
    """Install catalog Mamba2 mixer workarounds.

    ``fused_path`` clears the mixer's fused-path attribute. ``stream_ordering``
    wraps mixer ``forward`` so default-stream scan/conv kernels wait on the
    caller stream. Those kernels launch on the default stream, so a caller on
    another stream can race regardless of ZeRO stage.

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
