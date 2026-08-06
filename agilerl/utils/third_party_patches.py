# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Class-level workarounds for third-party behaviour that ZeRO-3 training cannot use.

* :func:`patch_zero3_fetch_trace` keeps deepspeed's parameter-fetch trace from
  replaying a submodule order the current step never produced.
* :func:`patch_zero3_param_persistence` keeps small ZeRO-3 parameters
  replicated on every rank by setting ``zero.Init`` persistence thresholds.
* :func:`patch_nemotron_mamba_fused_path` keeps the Nemotron-H Mamba2 mixer on
  the forward path that runs ``conv1d``, ``norm`` and ``out_proj`` as modules.
* :func:`patch_nemotron_mamba_stream_ordering` orders that mixer's
  default-stream kernels against the stream its caller is running on.

Every patch installs once at the class level, is idempotent, and takes
``enabled`` so a caller can turn it off (or an env kill-switch for
persistence). Third-party symbols are resolved when a patch runs, not when
this module is imported. An absent target is a no-op with a warning; a present
class with the wrong shape raises for wrap-style patches and fails soft for
attribute patches. Install them before the model is built.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "patch_nemotron_mamba_fused_path",
    "patch_nemotron_mamba_stream_ordering",
    "patch_zero3_fetch_trace",
    "patch_zero3_param_persistence",
]

_ZERO3_COORDINATOR_MODULE = "deepspeed.runtime.zero.partitioned_param_coordinator"
_ZERO3_PARTITION_MODULE = "deepspeed.runtime.zero.partition_parameters"
_NEMOTRON_H_MODULE = "transformers.models.nemotron_h.modeling_nemotron_h"

_MANGLE_PREFIX = "_PartitionedParameterCoordinator"
_SNAPSHOT_ATTR = "_agilerl_zero3_trace_snapshot"
_MEM_EFF_ATTR = "use_mem_eff_path"

_TRACE_PATCHED_FLAG = "_agilerl_zero3_trace_patched"
_STREAM_PATCHED_FLAG = "_agilerl_mamba_stream_patched"
_FUSED_PATH_PATCHED_FLAG = "_agilerl_mamba_fused_path_patched"

_TRACE_ATTRS = (
    "__trace_mode",
    "__submodule_order",
    "__param_order",
)

_STREAM_RELATION_LOGGED = False


def _try_import(module_path: str) -> ModuleType | None:
    """Import *module_path*, or None when the package cannot be loaded.

    :param module_path: Importable module path.
    :type module_path: str
    :return: The imported module, or None.
    :rtype: ModuleType | None
    """
    try:
        return importlib.import_module(module_path)
    except Exception:
        logger.debug("[patches] %s could not be imported", module_path)
        return None


def _resolve_zero3_targets() -> tuple[type | None, type | None]:
    """Resolve the ZeRO-3 coordinator class and trace-mode enum.

    Both are None when deepspeed is absent. A loaded module that lacks either
    symbol is version skew and raises.

    :return: ``(coordinator_cls, trace_mode)``, or ``(None, None)``.
    :rtype: tuple[type | None, type | None]
    """
    module = _try_import(_ZERO3_COORDINATOR_MODULE)
    if module is None:
        return None, None
    coordinator = getattr(module, "PartitionedParameterCoordinator", None)
    trace_mode = getattr(module, "ZeRoTraceMode", None)
    missing = [
        name
        for name, value in (
            ("PartitionedParameterCoordinator", coordinator),
            ("ZeRoTraceMode", trace_mode),
        )
        if value is None
    ]
    if missing:
        message = (
            "[zero3-trace] "
            f"{_ZERO3_COORDINATOR_MODULE} is present but missing "
            f"{', '.join(missing)}"
        )
        raise RuntimeError(message)
    return coordinator, trace_mode


def _resolve_mixer_class() -> type | None:
    """Resolve ``NemotronHMamba2Mixer``, or None when the modeling module is absent.

    :return: The mixer class, or None.
    :rtype: type | None
    """
    module = _try_import(_NEMOTRON_H_MODULE)
    if module is None:
        return None
    mixer = getattr(module, "NemotronHMamba2Mixer", None)
    if mixer is None:
        message = (
            f"[mamba] {_NEMOTRON_H_MODULE} is present but missing NemotronHMamba2Mixer"
        )
        raise RuntimeError(message)
    return mixer


def _resolve_zero3_init() -> type | None:
    """Resolve deepspeed ``zero.Init``, or None when it cannot be loaded.

    :return: The ``Init`` class, or None.
    :rtype: type | None
    """
    module = _try_import(_ZERO3_PARTITION_MODULE)
    if module is None:
        return None
    return getattr(module, "Init", None)


def _class_is_patched(cls: type, flag: str) -> bool:
    """Whether *cls* itself carries *flag*, ignoring any inherited value.

    :param cls: Class to inspect.
    :type cls: type
    :param flag: Marker attribute name.
    :type flag: str
    :return: True when this exact class has already been patched.
    :rtype: bool
    """
    return bool(vars(cls).get(flag, False))


def _mangled(name: str) -> str:
    """Name-mangle *name* as deepspeed's coordinator class does.

    :param name: Double-underscore attribute name.
    :type name: str
    :return: The mangled attribute name.
    :rtype: str
    """
    return f"{_MANGLE_PREFIX}{name}"


def _routes_to_conditional_submodules(deepspeed_config: object) -> bool:
    """Whether the config describes a model that picks submodules from the data.

    A ``leaf_module`` entry is how a mixture-of-experts block is declared, and
    such a block runs only the experts its router selected.

    :param deepspeed_config: Resolved DeepSpeed config, or None.
    :type deepspeed_config: object
    :return: True when a recorded submodule order cannot be replayed.
    :rtype: bool
    """
    if not isinstance(deepspeed_config, Mapping):
        return False
    zero_optimization = deepspeed_config.get("zero_optimization")
    if not isinstance(zero_optimization, Mapping):
        return False
    return bool(zero_optimization.get("leaf_module"))


def _copy_trace_value(value: object) -> object:
    """Copy a trace container one level deep, sharing the objects it references.

    :param value: Trace attribute value.
    :type value: object
    :return: Value that no longer aliases live coordinator state.
    :rtype: object
    """
    if isinstance(value, list):
        return list(value)
    return value


def _snapshot_trace_state(coordinator: object) -> dict[str, Any]:
    """Copy the coordinator's trace identity attributes.

    :param coordinator: Live parameter coordinator.
    :type coordinator: object
    :return: Mangled attribute name to copied value.
    :rtype: dict[str, Any]
    """
    snapshot: dict[str, Any] = {}
    for name in _TRACE_ATTRS:
        attr = _mangled(name)
        if not hasattr(coordinator, attr):
            continue
        snapshot[attr] = _copy_trace_value(getattr(coordinator, attr))
    return snapshot


def _restore_trace_state(coordinator: object, snapshot: dict[str, Any]) -> None:
    """Write a trace snapshot back onto the coordinator.

    :param coordinator: Live parameter coordinator.
    :type coordinator: object
    :param snapshot: Mangled attribute name to value.
    :type snapshot: dict[str, Any]
    :return: None
    :rtype: None
    """
    for attr, value in snapshot.items():
        setattr(coordinator, attr, value)


def _missing_trace_attrs(coordinator_cls: type) -> list[str]:
    """Trace attributes the coordinator's ``__init__`` does not assign.

    :param coordinator_cls: Coordinator class to inspect.
    :type coordinator_cls: type
    :return: Unmangled names of attributes this deepspeed version lacks.
    :rtype: list[str]
    """
    code = getattr(getattr(coordinator_cls, "__init__", None), "__code__", None)
    assigned = set(getattr(code, "co_names", ()))
    return [name for name in _TRACE_ATTRS if _mangled(name) not in assigned]


def _make_patched_reset_step(
    original_reset_step: Callable[[Any], None],
    invalid_mode: object,
    always_invalid: bool,
) -> Callable[[Any], None]:
    """Build the ``reset_step`` wrapper that guards the trace identity.

    :param original_reset_step: Unbound ``reset_step`` to call through to.
    :type original_reset_step: Callable[[Any], None]
    :param invalid_mode: The coordinator's ``INVALID`` trace mode member.
    :type invalid_mode: object
    :param always_invalid: Force on-demand fetch on every step.
    :type always_invalid: bool
    :return: Replacement ``reset_step``.
    :rtype: Callable[[Any], None]
    """
    trace_mode_attr = _mangled("__trace_mode")

    @functools.wraps(original_reset_step)
    def patched_reset_step(self: object) -> None:
        if always_invalid:
            original_reset_step(self)
            setattr(self, trace_mode_attr, invalid_mode)
            return

        if not torch.is_grad_enabled():
            if getattr(self, _SNAPSHOT_ATTR, None) is None:
                setattr(self, _SNAPSHOT_ATTR, _snapshot_trace_state(self))
                logger.info(
                    "[zero3-trace] no-grad forward: trace held at %s, "
                    "fetching on demand",
                    getattr(self, trace_mode_attr, None),
                )
            original_reset_step(self)
            setattr(self, trace_mode_attr, invalid_mode)
            return

        snapshot = getattr(self, _SNAPSHOT_ATTR, None)
        if snapshot is not None:
            _restore_trace_state(self, snapshot)
            setattr(self, _SNAPSHOT_ATTR, None)
            logger.info(
                "[zero3-trace] grad forward: trace restored to %s",
                getattr(self, trace_mode_attr, None),
            )
        original_reset_step(self)

    return patched_reset_step


def patch_zero3_fetch_trace(
    deepspeed_config: Mapping[str, Any] | None = None,
    *,
    enabled: bool = True,
) -> None:
    """Keep no-grad forwards from disturbing the ZeRO-3 parameter-fetch trace.

    Wraps ``PartitionedParameterCoordinator.reset_step`` once at the class
    level. Every call runs the original exactly once, so the inflight-param
    registry, fetch queue, step id, leaf-module fetch events and profiler are
    reset as deepspeed expects. Around that call the wrapper preserves only the
    trace identity: a no-grad forward snapshots it and then runs with the mode
    forced to ``INVALID`` (on-demand fetch, deterministic across ranks for
    forward-only passes), and the next grad-enabled call restores it, so the
    training forward and backward passes form a contiguous chain of matching
    steps and the trace reaches ``COMPLETE``. A config declaring
    ``leaf_module`` describes a model that picks submodules from the data, so
    every step fetches on demand and no trace is recorded.

    :param deepspeed_config: Resolved DeepSpeed config, or None.
    :type deepspeed_config: Mapping[str, Any] | None
    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
    :return: None
    :rtype: None
    """
    if not enabled:
        logger.info("[zero3-trace] disabled by caller; reset_step left unpatched")
        return

    coordinator_cls, trace_mode = _resolve_zero3_targets()
    if coordinator_cls is None or trace_mode is None:
        logger.warning(
            "[zero3-trace] deepspeed ZeRO-3 coordinator unavailable; "
            "reset_step left unpatched",
        )
        return
    if _class_is_patched(coordinator_cls, _TRACE_PATCHED_FLAG):
        return

    original_reset_step = getattr(coordinator_cls, "reset_step", None)
    invalid_mode = getattr(trace_mode, "INVALID", None)
    if original_reset_step is None or invalid_mode is None:
        message = "[zero3-trace] coordinator lacks reset_step or an INVALID trace mode"
        raise RuntimeError(message)

    missing = _missing_trace_attrs(coordinator_cls)
    if missing:
        message = f"[zero3-trace] coordinator does not define {', '.join(missing)}"
        raise RuntimeError(message)

    always_invalid = _routes_to_conditional_submodules(deepspeed_config)
    coordinator_target: Any = coordinator_cls
    coordinator_target.reset_step = _make_patched_reset_step(
        original_reset_step, invalid_mode, always_invalid
    )
    setattr(coordinator_cls, _TRACE_PATCHED_FLAG, True)
    if always_invalid:
        logger.info(
            "[zero3-trace] reset_step patched; leaf_module declared, so every "
            "step fetches parameters on demand",
        )
        return
    logger.info(
        "[zero3-trace] reset_step patched; no-grad forwards preserve %d trace "
        "identity attributes",
        len(_TRACE_ATTRS),
    )


def patch_zero3_param_persistence(
    param_persistence_threshold: int,
    *,
    model_persistence_threshold: int | None = None,
    num_partitions: int = 1,
) -> None:
    """Keep small ZeRO-3 parameters replicated on every rank.

    ``zero.Init`` marks a parameter persistent, and so exempt from partitioning
    and from per-forward all-gathers, only when its class-level
    ``apply_param_persistence`` flag is set and the parameter fits under both
    thresholds. Setting the flag and the thresholds here decides persistence for
    every parameter created afterwards, so this must run before the model is
    constructed. The model threshold is a per-rank budget, hence the floor
    division by *num_partitions*. Disabled by
    ``AGILERL_ZERO3_PERSISTENCE_PATCH=0``; a no-op when ``zero.Init`` is
    unavailable or lacks the attributes involved.

    :param param_persistence_threshold: Largest parameter element count kept persistent.
    :type param_persistence_threshold: int
    :param model_persistence_threshold: Total persistent element budget across the model.
    :type model_persistence_threshold: int | None
    :param num_partitions: Number of ZeRO-3 parameter partitions (trainer world size).
    :type num_partitions: int
    :return: None
    :rtype: None
    """
    if os.environ.get("AGILERL_ZERO3_PERSISTENCE_PATCH", "1") == "0":
        logger.warning(
            "[zero3-persist] disabled by AGILERL_ZERO3_PERSISTENCE_PATCH=0",
        )
        return

    init_cls = _resolve_zero3_init()
    if init_cls is None:
        logger.warning(
            "[zero3-persist] deepspeed zero.Init unavailable; "
            "parameter persistence left untouched",
        )
        return

    partitions = int(num_partitions)
    if partitions < 1:
        logger.warning(
            "[zero3-persist] num_partitions=%s is not positive; treating it as 1",
            num_partitions,
        )
        partitions = 1

    targets: dict[str, Any] = {
        "apply_param_persistence": True,
        "param_persistence_threshold": int(param_persistence_threshold),
    }
    if model_persistence_threshold is not None:
        targets["model_persistence_threshold"] = (
            int(model_persistence_threshold) // partitions
        )

    missing = [name for name in targets if not hasattr(init_cls, name)]
    if missing:
        logger.warning(
            "[zero3-persist] zero.Init does not define %s; "
            "parameter persistence left untouched",
            ", ".join(missing),
        )
        return

    for name, value in targets.items():
        setattr(init_cls, name, value)

    logger.info(
        "[zero3-persist] persistence enabled: param_threshold=%d "
        "model_threshold=%s num_partitions=%d",
        targets["param_persistence_threshold"],
        getattr(init_cls, "model_persistence_threshold", None),
        partitions,
    )


def _assigns_mem_eff_attr(original_init: Callable[..., None]) -> bool:
    """Whether the mixer's ``__init__`` sets the fused-path attribute.

    :param original_init: Mixer ``__init__`` to inspect.
    :type original_init: Callable[..., None]
    :return: True when the attribute name appears among the names it touches.
    :rtype: bool
    """
    code = getattr(original_init, "__code__", None)
    return _MEM_EFF_ATTR in set(getattr(code, "co_names", ()))


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
        setattr(self, _MEM_EFF_ATTR, False)

    return patched_init


def patch_nemotron_mamba_fused_path(*, enabled: bool = True) -> None:
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
    so it must be installed before the model is built.

    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
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
    if _class_is_patched(mixer_cls, _FUSED_PATH_PATCHED_FLAG):
        return

    original_init = getattr(mixer_cls, "__init__", None)
    if original_init is None or not _assigns_mem_eff_attr(original_init):
        message = (
            f"[mamba-fused-path] NemotronHMamba2Mixer.__init__ does not set "
            f"{_MEM_EFF_ATTR}"
        )
        raise RuntimeError(message)

    mixer_target: Any = mixer_cls
    mixer_target.__init__ = _make_patched_init(original_init)
    setattr(mixer_cls, _FUSED_PATH_PATCHED_FLAG, True)
    logger.info(
        "[mamba-fused-path] NemotronHMamba2Mixer.__init__ patched; %s cleared "
        "on every mixer",
        _MEM_EFF_ATTR,
    )


def _log_stream_relation(same_stream: bool) -> None:
    """Log the caller/default stream relation once for GPU A/B grepping.

    :param same_stream: Whether the calling stream is the default stream.
    :type same_stream: bool
    :return: None
    :rtype: None
    """
    global _STREAM_RELATION_LOGGED
    if _STREAM_RELATION_LOGGED:
        return
    _STREAM_RELATION_LOGGED = True
    logger.info(
        "[mamba-stream] caller stream is%s the default stream",
        "" if same_stream else " not",
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
        _log_stream_relation(current == default)
        if current == default:
            return original_forward(self, hidden_states, *args, **kwargs)

        default.wait_stream(current)
        _record_streams(hidden_states, default)
        outputs = original_forward(self, hidden_states, *args, **kwargs)
        current.wait_stream(default)
        _record_streams(outputs, default, current)
        return outputs

    return patched_forward


def patch_nemotron_mamba_stream_ordering(*, enabled: bool = True) -> None:
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
    if _class_is_patched(mixer_cls, _STREAM_PATCHED_FLAG):
        return

    original_forward = getattr(mixer_cls, "forward", None)
    if original_forward is None:
        message = "[mamba-stream] NemotronHMamba2Mixer lacks forward"
        raise RuntimeError(message)

    mixer_target: Any = mixer_cls
    mixer_target.forward = _make_patched_forward(original_forward)
    setattr(mixer_cls, _STREAM_PATCHED_FLAG, True)
    logger.info(
        "[mamba-stream] NemotronHMamba2Mixer.forward patched; default-stream "
        "kernels ordered against the calling stream",
    )
