# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Class-level workarounds for third-party behaviour that ZeRO-3 training cannot use.

Call :func:`install_zero3_third_party_hooks` before the model is built; it
selects the patches that apply to the DeepSpeed config.

* :func:`patch_zero3_fetch_trace` keeps deepspeed's parameter-fetch trace from
  replaying a submodule order the current step never produced.
* :func:`patch_zero3_param_persistence` keeps small ZeRO-3 parameters
  replicated on every rank by setting ``zero.Init`` persistence thresholds.

:func:`try_import` and :func:`class_is_patched` are exported for downstream
packages that install their own architecture-scoped patches.

Every patch installs once at the class level, is idempotent, and takes
``enabled`` so a caller can turn it off. Third-party symbols are resolved when
a patch runs, not when this module is imported. An absent target is a no-op
with a warning; a present class with the wrong shape raises for wrap-style
patches and fails soft for attribute patches.
"""

from __future__ import annotations

import functools
import importlib
import logging
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "class_is_patched",
    "install_zero3_third_party_hooks",
    "patch_zero3_fetch_trace",
    "patch_zero3_param_persistence",
    "try_import",
]

ZERO3_COORDINATOR_MODULE = "deepspeed.runtime.zero.partitioned_param_coordinator"
ZERO3_PARTITION_MODULE = "deepspeed.runtime.zero.partition_parameters"

MANGLE_PREFIX = "_PartitionedParameterCoordinator"
SNAPSHOT_ATTR = "_agilerl_zero3_trace_snapshot"

TRACE_PATCHED_FLAG = "_agilerl_zero3_trace_patched"

TRACE_ATTRS = (
    "__trace_mode",
    "__submodule_order",
    "__param_order",
)


def install_zero3_third_party_hooks(
    deepspeed_config: Mapping[str, Any] | None,
    *,
    num_partitions: int = 1,
) -> None:
    """Install the ZeRO-3 third-party patches that apply to this run.

    Always installs the fetch-trace workaround. Parameter persistence installs
    when the config declares ``stage3_param_persistence_threshold``. Call
    before the model is built.

    :param deepspeed_config: Resolved DeepSpeed config, or None.
    :type deepspeed_config: Mapping[str, Any] | None
    :param num_partitions: ZeRO-3 partition count (trainer world size).
    :type num_partitions: int
    :return: None
    :rtype: None
    """
    patch_zero3_fetch_trace(deepspeed_config)

    if not isinstance(deepspeed_config, Mapping):
        return
    zero_optimization = deepspeed_config.get("zero_optimization")
    if not isinstance(zero_optimization, Mapping):
        return
    param_threshold = zero_optimization.get("stage3_param_persistence_threshold")
    if param_threshold is None:
        return
    patch_zero3_param_persistence(
        int(param_threshold),
        model_persistence_threshold=zero_optimization.get(
            "stage3_model_persistence_threshold",
        ),
        num_partitions=num_partitions,
    )


def try_import(module_path: str) -> ModuleType | None:
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
    module = try_import(ZERO3_COORDINATOR_MODULE)
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
            f"{ZERO3_COORDINATOR_MODULE} is present but missing "
            f"{', '.join(missing)}"
        )
        raise RuntimeError(message)
    return coordinator, trace_mode


def _resolve_zero3_init() -> type | None:
    """Resolve deepspeed ``zero.Init``, or None when it cannot be loaded.

    :return: The ``Init`` class, or None.
    :rtype: type | None
    """
    module = try_import(ZERO3_PARTITION_MODULE)
    if module is None:
        return None
    return getattr(module, "Init", None)


def class_is_patched(cls: type, flag: str) -> bool:
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
    return f"{MANGLE_PREFIX}{name}"


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
    for name in TRACE_ATTRS:
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
    return [name for name in TRACE_ATTRS if _mangled(name) not in assigned]


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
            if getattr(self, SNAPSHOT_ATTR, None) is None:
                setattr(self, SNAPSHOT_ATTR, _snapshot_trace_state(self))
            original_reset_step(self)
            setattr(self, trace_mode_attr, invalid_mode)
            return

        snapshot = getattr(self, SNAPSHOT_ATTR, None)
        if snapshot is not None:
            _restore_trace_state(self, snapshot)
            setattr(self, SNAPSHOT_ATTR, None)
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
    if class_is_patched(coordinator_cls, TRACE_PATCHED_FLAG):
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
    setattr(coordinator_cls, TRACE_PATCHED_FLAG, True)
    if always_invalid:
        logger.info(
            "[zero3-trace] reset_step patched; leaf_module declared, so every "
            "step fetches parameters on demand",
        )
        return
    logger.info(
        "[zero3-trace] reset_step patched; no-grad forwards preserve %d trace "
        "identity attributes",
        len(TRACE_ATTRS),
    )


def patch_zero3_param_persistence(
    param_persistence_threshold: int,
    *,
    model_persistence_threshold: int | None = None,
    num_partitions: int = 1,
    enabled: bool = True,
) -> None:
    """Keep small ZeRO-3 parameters replicated on every rank.

    ``zero.Init`` marks a parameter persistent, and so exempt from partitioning
    and from per-forward all-gathers, only when its class-level
    ``apply_param_persistence`` flag is set and the parameter fits under both
    thresholds. Setting the flag and the thresholds here decides persistence for
    every parameter created afterwards, so this must run before the model is
    constructed. The model threshold is a per-rank budget, hence the floor
    division by *num_partitions*. A no-op when ``zero.Init`` is unavailable or
    lacks the attributes involved.

    :param param_persistence_threshold: Largest parameter element count kept persistent.
    :type param_persistence_threshold: int
    :param model_persistence_threshold: Total persistent element budget across the model.
    :type model_persistence_threshold: int | None
    :param num_partitions: Number of ZeRO-3 parameter partitions (trainer world size).
    :type num_partitions: int
    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
    :return: None
    :rtype: None
    """
    if not enabled:
        logger.info(
            "[zero3-persist] disabled by caller; parameter persistence left untouched",
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
