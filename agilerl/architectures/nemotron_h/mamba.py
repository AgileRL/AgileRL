# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Class-level workarounds for the Nemotron-H Mamba2 mixer under ZeRO-3.

Both patches install once at the class level, are idempotent, and take
``enabled`` so a caller can turn them off. ``NemotronHMamba2Mixer`` is resolved
when a patch runs, not when this module is imported: an absent target is a
no-op with a warning, and a present class with the wrong shape raises.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import torch

from agilerl.utils.patching import class_is_patched, try_import

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from agilerl.protocols import PreTrainedModelProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "patch_nemotron_mamba_fused_path",
    "patch_nemotron_mamba_stream_ordering",
]

NEMOTRON_H_MODULE = "transformers.models.nemotron_h.modeling_nemotron_h"

MEM_EFF_ATTR = "use_mem_eff_path"

STREAM_PATCHED_FLAG = "_agilerl_mamba_stream_patched"
FUSED_PATH_PATCHED_FLAG = "_agilerl_mamba_fused_path_patched"


def _resolve_mixer_class() -> type | None:
    """Resolve ``NemotronHMamba2Mixer``, or None when the modeling module is absent.

    :return: The mixer class, or None.
    :rtype: type | None
    """
    module = try_import(NEMOTRON_H_MODULE)
    if module is None:
        return None
    mixer = getattr(module, "NemotronHMamba2Mixer", None)
    if mixer is None:
        message = (
            f"[mamba] {NEMOTRON_H_MODULE} is present but missing NemotronHMamba2Mixer"
        )
        raise RuntimeError(message)
    return mixer


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
    model: PreTrainedModelProtocol,
) -> int:
    """Clear the fused-path attribute on every existing mixer in *model*.

    Raises on a module whose class only shares the mixer's name (e.g. a
    ``trust_remote_code`` copy) — an instance no patch can reach.

    :param mixer_cls: The patched mixer class.
    :type mixer_cls: type
    :param model: Model whose submodules are swept.
    :type model: PreTrainedModelProtocol
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
    enabled: bool = True,
    model: PreTrainedModelProtocol | None = None,
) -> None:
    """Keep every Nemotron-H Mamba2 mixer on its decomposed forward path.

    ``NemotronHMamba2Mixer.cuda_kernels_forward`` takes a fused branch when
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

    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
    :param model: Already-built model whose mixers are also cleared,
        defaults to None.
    :type model: PreTrainedModelProtocol | None, optional
    :return: None
    :rtype: None
    """
    if not enabled:
        logger.info("[mamba-fused-path] disabled by caller; __init__ left unpatched")
        return

    mixer_cls = _resolve_mixer_class()
    if mixer_cls is None:
        logger.warning(
            "[mamba-fused-path] NemotronHMamba2Mixer unavailable; "
            "__init__ left unpatched",
        )
        return

    if not class_is_patched(mixer_cls, FUSED_PATH_PATCHED_FLAG):
        original_init = getattr(mixer_cls, "__init__", None)
        if original_init is None or not _assigns_mem_eff_attr(original_init):
            message = (
                f"[mamba-fused-path] NemotronHMamba2Mixer.__init__ does not set "
                f"{MEM_EFF_ATTR}"
            )
            raise RuntimeError(message)

        mixer_target: Any = mixer_cls
        mixer_target.__init__ = _make_patched_init(original_init)
        setattr(mixer_cls, FUSED_PATH_PATCHED_FLAG, True)
        logger.info(
            "[mamba-fused-path] NemotronHMamba2Mixer.__init__ patched; %s cleared "
            "on every mixer",
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
    enabled: bool = True,
    model: PreTrainedModelProtocol | None = None,
) -> None:
    """Order the Nemotron-H Mamba2 mixer's default-stream kernels against its caller.

    ``NemotronHMamba2Mixer.forward`` runs the mamba and causal-conv1d kernels
    inside ``torch.cuda.stream(default_stream)``. A ZeRO-3 parameter all-gather
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

    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
    :param model: Unused; accepted for family-dispatch parity, defaults to None.
    :type model: PreTrainedModelProtocol | None, optional
    :return: None
    :rtype: None
    """
    if not enabled:
        logger.info("[mamba-stream] disabled by caller; forward left unpatched")
        return

    mixer_cls = _resolve_mixer_class()
    if mixer_cls is None:
        logger.warning(
            "[mamba-stream] NemotronHMamba2Mixer unavailable; forward left unpatched",
        )
        return
    if class_is_patched(mixer_cls, STREAM_PATCHED_FLAG):
        return

    original_forward = getattr(mixer_cls, "forward", None)
    if original_forward is None:
        message = "[mamba-stream] NemotronHMamba2Mixer lacks forward"
        raise RuntimeError(message)

    mixer_target: Any = mixer_cls
    mixer_target.forward = _make_patched_forward(original_forward)
    setattr(mixer_cls, STREAM_PATCHED_FLAG, True)
    logger.info(
        "[mamba-stream] NemotronHMamba2Mixer.forward patched; default-stream "
        "kernels ordered against the calling stream",
    )
