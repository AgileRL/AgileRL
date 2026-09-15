# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Class-level DeepSpeed workarounds that ZeRO-3 training needs.

Call :func:`install_zero3_patches` before the model is built; it selects the
patches that apply to the DeepSpeed config and model identity.
Architecture-scoped patches live in :mod:`agilerl.architectures` and are
dispatched from here by family.

* :func:`patch_zero3_fetch_trace` keeps deepspeed's parameter-fetch trace from
  replaying a submodule order the current step never produced.
* :func:`patch_zero3_param_persistence` keeps small ZeRO-3 parameters
  replicated on every rank by setting ``zero.Init`` persistence thresholds.
* :func:`patch_zero3_persistent_release` keeps parameters already marked
  persistent resident when a submodule releases under an incomplete fetch
  trace, so persistence holds on models that fetch on demand every step.

Resolution and idempotence primitives live in :mod:`agilerl.utils.patching`,
below both this module and the per-family patch modules.

Every patch installs once at the class level, is idempotent, and takes
``enabled`` so a caller can turn it off. Third-party symbols are resolved when
a patch runs, not when this module is imported. An absent target is a no-op
with a warning; a present class with the wrong shape raises for wrap-style
patches and fails soft for attribute patches.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from agilerl.architectures import install_family_patches, pretrained_model_type
from agilerl.utils.patching import class_is_patched, try_import

if TYPE_CHECKING:
    from collections.abc import Callable

    from peft import PeftModel
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

__all__ = [
    "install_model_patches",
    "install_zero3_patches",
    "patch_zero3_fetch_trace",
    "patch_zero3_param_persistence",
    "patch_zero3_persistent_release",
]

ZERO3_COORDINATOR_MODULE = "deepspeed.runtime.zero.partitioned_param_coordinator"
ZERO3_PARTITION_MODULE = "deepspeed.runtime.zero.partition_parameters"

MANGLE_PREFIX = "_PartitionedParameterCoordinator"
SNAPSHOT_ATTR = "_agilerl_zero3_trace_snapshot"

TRACE_PATCHED_FLAG = "_agilerl_zero3_trace_patched"
PERSIST_RELEASE_PATCHED_FLAG = "_agilerl_zero3_persist_release_patched"

PACKED_LEAF_NAME_SUFFIXES = frozenset({"experts"})
PACKED_LEAF_CLASS_SUFFIX = "MoE"

TRACE_ATTRS = (
    "__trace_mode",
    "__submodule_order",
    "__param_order",
)


def install_model_patches(
    zero_stage: int,
    deepspeed_config: Mapping[str, Any] | None,
    *,
    model_name_or_path: str | None = None,
    model: PreTrainedModel | PeftModel | None = None,
    num_partitions: int = 1,
) -> None:
    """Install family patches, then ZeRO-3 wraps when ``zero_stage`` is 3.

    :param zero_stage: DeepSpeed ZeRO stage for this run.
    :type zero_stage: int
    :param deepspeed_config: Resolved DeepSpeed config, or None.
    :type deepspeed_config: Mapping[str, Any] | None
    :param model_name_or_path: Hugging Face id or local path of the actor.
    :type model_name_or_path: str | None
    :param model: Already-built actor model, or None.
    :type model: PreTrainedModel | PeftModel | None
    :param num_partitions: ZeRO-3 partition count (trainer world size).
    :type num_partitions: int
    :return: None
    :rtype: None
    """
    if model is not None:
        model_type = model.config.model_type
    elif model_name_or_path:
        model_type = pretrained_model_type(model_name_or_path)
    else:
        model_type = None
    install_family_patches(model_type, model)
    if zero_stage == 3:
        install_zero3_patches(
            deepspeed_config,
            num_partitions=num_partitions,
        )


def install_zero3_patches(
    deepspeed_config: Mapping[str, Any] | None,
    *,
    num_partitions: int = 1,
) -> None:
    """Install the ZeRO-3 patches that apply to this run.

    Always installs the fetch-trace workaround and the persistent-release
    guard. Parameter persistence installs when the config declares
    ``stage3_param_persistence_threshold``. Call before the model is built.

    :param deepspeed_config: Resolved DeepSpeed config, or None.
    :type deepspeed_config: Mapping[str, Any] | None
    :param num_partitions: ZeRO-3 partition count (trainer world size).
    :type num_partitions: int
    :return: None
    :rtype: None
    """
    patch_zero3_fetch_trace(deepspeed_config)
    patch_zero3_persistent_release()

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


def _resolve_zero3_release_targets() -> tuple[
    type | None,
    Callable[..., Any] | None,
    Callable[[Any], bool] | None,
    object | None,
]:
    """Resolve the ZeRO-3 coordinator class and its param-iteration helpers.

    All None when deepspeed is absent. A loaded module that lacks any symbol
    is version skew and raises.

    :return: ``(coordinator_cls, iter_params, z3_leaf_module, available_status)``,
        or all None.
    :rtype: tuple[type | None, Callable | None, Callable | None, object | None]
    """
    module = try_import(ZERO3_COORDINATOR_MODULE)
    if module is None:
        return None, None, None, None
    partition_module = try_import(ZERO3_PARTITION_MODULE)
    coordinator = getattr(module, "PartitionedParameterCoordinator", None)
    iter_params = getattr(module, "iter_params", None)
    z3_leaf_module = getattr(module, "z3_leaf_module", None)
    zero_param_status = (
        getattr(partition_module, "ZeroParamStatus", None)
        if partition_module is not None
        else None
    )
    available_status = getattr(zero_param_status, "AVAILABLE", None)
    missing = [
        name
        for name, value in (
            ("PartitionedParameterCoordinator", coordinator),
            ("iter_params", iter_params),
            ("z3_leaf_module", z3_leaf_module),
            ("ZeroParamStatus.AVAILABLE", available_status),
        )
        if value is None
    ]
    if missing:
        message = (
            "[zero3-persist-release] "
            f"{ZERO3_COORDINATOR_MODULE} is present but missing "
            f"{', '.join(missing)}"
        )
        raise RuntimeError(message)
    return coordinator, iter_params, z3_leaf_module, available_status


def _resolve_zero3_init() -> type | None:
    """Resolve deepspeed ``zero.Init``, or None when it cannot be loaded.

    :return: The ``Init`` class, or None.
    :rtype: type | None
    """
    module = try_import(ZERO3_PARTITION_MODULE)
    if module is None:
        return None
    return getattr(module, "Init", None)


def _mangled(name: str) -> str:
    """Name-mangle *name* as deepspeed's coordinator class does.

    :param name: Double-underscore attribute name.
    :type name: str
    :return: The mangled attribute name.
    :rtype: str
    """
    return f"{MANGLE_PREFIX}{name}"


def _is_packed_experts_leaf(leaf_module: Mapping[Any, Any]) -> bool:
    """Whether ``leaf_module`` names a packed experts block every rank always enters.

    :param leaf_module: DeepSpeed ``zero_optimization.leaf_module`` mapping.
    :type leaf_module: Mapping[Any, Any]
    :return: True for packed ``experts`` suffixes or packed ``*MoE`` classes.
    :rtype: bool
    """
    suffixes = leaf_module.get("name_suffixes")
    if isinstance(suffixes, (list, tuple, set, frozenset)):
        if any(suffix in PACKED_LEAF_NAME_SUFFIXES for suffix in suffixes):
            return True
    classes = leaf_module.get("classes")
    if isinstance(classes, (list, tuple, set, frozenset)):
        for cls in classes:
            if not isinstance(cls, str):
                continue
            simple_name = cls.rsplit(".", 1)[-1]
            if simple_name.endswith(PACKED_LEAF_CLASS_SUFFIX):
                return True
    return False


def _routes_to_conditional_submodules(deepspeed_config: object) -> bool:
    """Whether a recorded submodule fetch order cannot be replayed.

    Packed ``experts`` is one module every rank always enters; a recorded
    fetch order is valid. Per-expert ModuleList children with rank-divergent
    visit order cannot replay a recorded order.

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
    leaf_module = zero_optimization.get("leaf_module")
    if not isinstance(leaf_module, Mapping) or not leaf_module:
        return False
    if _is_packed_experts_leaf(leaf_module):
        return False
    suffixes = leaf_module.get("name_suffixes")
    classes = leaf_module.get("classes")
    has_suffixes = isinstance(suffixes, (list, tuple)) and len(suffixes) > 0
    classes_listed = isinstance(classes, (list, tuple)) and len(classes) > 0
    return has_suffixes or classes_listed


def _copy_trace_value(value: object) -> object:
    """Copy a trace container one level deep, sharing the objects it references.

    :param value: Trace attribute value.
    :type value: object
    :return: Copy that does not alias live coordinator state.
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
    steps and the trace reaches ``COMPLETE``. A ``leaf_module`` of per-expert
    child modules with rank-divergent visit order cannot replay a recorded
    fetch sequence, so every step fetches on demand. Packed ``experts`` leaves
    are one module every rank always enters, so they keep the RECORD →
    COMPLETE path.

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


def _partition_unreleased_leaf_params(
    coordinator: object,
    submodule: object,
    iter_params: Callable[..., Any],
    available_status: object,
) -> None:
    """Partition leaf params blocked only by stale leftover owner ids.

    :param coordinator: Live parameter coordinator.
    :type coordinator: object
    :param submodule: ZeRO-3 leaf submodule whose params were just released.
    :type submodule: object
    :param iter_params: deepspeed's parameter iterator for a submodule.
    :type iter_params: Callable[..., Any]
    :param available_status: ``ZeroParamStatus.AVAILABLE``.
    :type available_status: object
    :return: None
    :rtype: None
    """
    live_owners = set(getattr(coordinator, _mangled("__active_backward_submodules")))
    n_available_attr = _mangled("__n_available_params")
    for param in iter_params(submodule, recurse=True):
        if getattr(param, "ds_persist", False):
            continue
        if getattr(param, "is_external_param", False):
            continue
        if getattr(param, "ds_status", None) != available_status:
            continue
        active = getattr(param, "ds_active_sub_modules", None)
        if not active or active & live_owners:
            continue
        active.clear()
        param.partition()
        setattr(
            coordinator,
            n_available_attr,
            getattr(coordinator, n_available_attr) - param.ds_numel,
        )


def _make_patched_release_sub_module(
    original_release_sub_module: Callable[..., None],
    iter_params: Callable[..., Any],
    z3_leaf_module: Callable[[Any], bool],
    available_status: object,
) -> Callable[..., None]:
    """Build the ``release_sub_module`` wrapper that pins persist and frees leaves.

    :param original_release_sub_module: Unbound ``release_sub_module`` to call
        through to.
    :type original_release_sub_module: Callable[..., None]
    :param iter_params: deepspeed's parameter iterator for a submodule.
    :type iter_params: Callable[..., Any]
    :param z3_leaf_module: Predicate for deepspeed leaf modules.
    :type z3_leaf_module: Callable[[Any], bool]
    :param available_status: ``ZeroParamStatus.AVAILABLE``.
    :type available_status: object
    :return: Replacement ``release_sub_module``.
    :rtype: Callable[..., None]
    """

    @functools.wraps(original_release_sub_module)
    def patched_release_sub_module(
        self: object, submodule: object, *args: Any, **kwargs: Any
    ) -> None:
        is_leaf = z3_leaf_module(submodule)
        pinned = [
            (param, getattr(param, "is_external_param", False))
            for param in iter_params(submodule, recurse=is_leaf)
            if getattr(param, "ds_persist", False)
        ]
        for param, _ in pinned:
            param.is_external_param = True
        try:
            original_release_sub_module(self, submodule, *args, **kwargs)
        finally:
            for param, was_external in pinned:
                param.is_external_param = was_external
        if "forward" in kwargs:
            is_forward = kwargs["forward"]
        elif args:
            is_forward = args[0]
        else:
            is_forward = False
        if is_leaf and not is_forward:
            _partition_unreleased_leaf_params(
                self, submodule, iter_params, available_status
            )

    return patched_release_sub_module


def patch_zero3_persistent_release(*, enabled: bool = True) -> None:
    """Keep persistent ZeRO-3 parameters resident when a submodule releases.

    ``release_sub_module`` excludes persistent parameters from release only
    when its recorded fetch trace is complete; while the trace is recording or
    invalid it releases them like any other parameter, and each one is
    re-gathered on its next use. The wrapper flags the submodule's persistent
    parameters as external for the duration of each call, which every guard in
    the release loop honours (both the partition call and the leaf-module data
    swap), then restores the flags. After a backward leaf release, leftover
    owner ids that are not in the active backward set are stale; those
    non-persist params are partitioned and ``__n_available_params`` is
    decremented. Empty-active AVAILABLE params and params a live backward
    owner still holds stay gathered.
    Whole-model releases (``release_and_reset_all``) ignore the external flag,
    so optimizer-step and teardown partitioning still cover every parameter.
    A no-op when deepspeed is unavailable.

    :param enabled: Install the patch, defaults to True.
    :type enabled: bool, optional
    :return: None
    :rtype: None
    """
    if not enabled:
        logger.info(
            "[zero3-persist-release] disabled by caller; "
            "release_sub_module left unpatched",
        )
        return

    coordinator_cls, iter_params, z3_leaf_module, available_status = (
        _resolve_zero3_release_targets()
    )
    if (
        coordinator_cls is None
        or iter_params is None
        or z3_leaf_module is None
        or available_status is None
    ):
        logger.warning(
            "[zero3-persist-release] deepspeed ZeRO-3 coordinator unavailable; "
            "release_sub_module left unpatched",
        )
        return
    if class_is_patched(coordinator_cls, PERSIST_RELEASE_PATCHED_FLAG):
        return

    original_release_sub_module = getattr(coordinator_cls, "release_sub_module", None)
    if original_release_sub_module is None:
        message = "[zero3-persist-release] coordinator lacks release_sub_module"
        raise RuntimeError(message)

    coordinator_target: Any = coordinator_cls
    coordinator_target.release_sub_module = _make_patched_release_sub_module(
        original_release_sub_module,
        iter_params,
        z3_leaf_module,
        available_status,
    )
    setattr(coordinator_cls, PERSIST_RELEASE_PATCHED_FLAG, True)
    logger.info(
        "[zero3-persist-release] release_sub_module patched; persistent "
        "parameters stay resident while the fetch trace is incomplete",
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
