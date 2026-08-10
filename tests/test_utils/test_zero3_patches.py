# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deepspeed class-level patches and family dispatch.

The third-party classes are replaced with doubles, so these run without
deepspeed or a GPU.
"""

import collections
import enum
import logging

import pytest
import torch

from agilerl.utils import zero3_patches
from agilerl.utils.zero3_patches import (
    install_zero3_patches,
    patch_zero3_fetch_trace,
    patch_zero3_param_persistence,
)

_DEFAULT_PARAM_THRESHOLD = 100_000
_DEFAULT_MODEL_THRESHOLD = 1_000_000_000

_DROPPED_ATTRS = (
    "__param_queue",
    "__most_recent_step_id_param_fetched_for",
    "__step_id_module_fetched_for",
    "__step_id",
    "__n_available_params",
)

_LEAF_MODULE_CONFIG = {
    "zero_optimization": {
        "stage": 3,
        "leaf_module": {"name_suffixes": ["experts"]},
    },
}


class _FakeTraceMode(enum.Enum):
    RECORD = 1
    COMPLETE = 2
    INVALID = 3


class _FakeModule:
    def __init__(self, ds_id):
        self.ds_id = ds_id


def _make_coordinator_class():
    """Fresh coordinator double; the class name reproduces deepspeed's mangling."""

    class PartitionedParameterCoordinator:
        def __init__(self):
            self.__trace_mode = _FakeTraceMode.COMPLETE
            self.__step_id = 0
            self.__submodule_order = [_FakeModule(1), _FakeModule(2)]
            self.__param_order = [("p0", 0), ("p1", 1)]
            self.__param_queue = collections.deque()
            self.__most_recent_step_id_param_fetched_for = collections.defaultdict(
                lambda: -1,
            )
            self.__step_id_module_fetched_for = collections.defaultdict(
                collections.deque
            )
            self.__n_available_params = 0
            self.__ongoing_fetch_leaf_module_events = set()
            self.reset_step_calls = 0
            self.trace_modes_at_reset = []

        def reset_step(self):
            """Per-step bookkeeping only, mirroring deepspeed's own ``reset_step``."""
            self.reset_step_calls += 1
            self.trace_modes_at_reset.append(self.__trace_mode)
            self.__param_queue = collections.deque(self.__param_order)
            self.__most_recent_step_id_param_fetched_for = collections.defaultdict(
                lambda: -1,
            )
            self.__step_id_module_fetched_for = collections.defaultdict(
                collections.deque
            )
            self.__step_id = 0
            self.__n_available_params = 0
            self.__ongoing_fetch_leaf_module_events.clear()

    return PartitionedParameterCoordinator


def _attr(coordinator, name):
    return getattr(coordinator, zero3_patches._mangled(name))


def _set_attr(coordinator, name, value):
    setattr(coordinator, zero3_patches._mangled(name), value)


def _simulate_no_grad_forward(coordinator):
    _attr(coordinator, "__submodule_order").append(_FakeModule(99))
    _set_attr(coordinator, "__step_id", 41)
    _set_attr(coordinator, "__n_available_params", 4096)
    _attr(coordinator, "__step_id_module_fetched_for")[1].append(41)
    _attr(coordinator, "__ongoing_fetch_leaf_module_events").add(7)
    _attr(coordinator, "__param_queue").clear()


@pytest.fixture
def coordinator_cls(monkeypatch):
    cls = _make_coordinator_class()
    monkeypatch.setattr(
        zero3_patches,
        "_resolve_zero3_targets",
        lambda: (cls, _FakeTraceMode),
    )
    return cls


class TestImportDoesNotPullThirdParty:
    """Symbols are resolved inside patch_*; nothing is cached at import time."""

    def test_module_has_no_eager_third_party_bindings(self):
        for name in (
            "_COORDINATOR_CLASS",
            "_TRACE_MODE",
            "_MIXER_CLASS",
            "_INIT_CLASS",
        ):
            assert not hasattr(zero3_patches, name)
        assert callable(zero3_patches._resolve_zero3_targets)
        assert callable(zero3_patches._resolve_zero3_init)


class TestInstallZero3ThirdPartyHooks:
    def test_always_installs_fetch_trace(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            zero3_patches,
            "patch_zero3_fetch_trace",
            lambda config: calls.append("fetch"),
        )
        monkeypatch.setattr(
            zero3_patches,
            "install_family_zero3_patches",
            lambda _name: calls.append("families"),
        )
        monkeypatch.setattr(
            zero3_patches,
            "patch_zero3_param_persistence",
            lambda *args, **kwargs: calls.append("persist"),
        )

        install_zero3_patches({"zero_optimization": {"stage": 3}})

        assert calls == ["fetch", "families"]

    def test_family_dispatch_receives_the_checkpoint_id(self, monkeypatch):
        seen: list[str | None] = []
        monkeypatch.setattr(
            zero3_patches,
            "patch_zero3_fetch_trace",
            lambda config: None,
        )
        monkeypatch.setattr(
            zero3_patches,
            "install_family_zero3_patches",
            seen.append,
        )

        install_zero3_patches(
            {"zero_optimization": {"stage": 3}},
            model_name_or_path="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        )

        assert seen == ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"]

    def test_persistence_when_threshold_set(self, monkeypatch):
        persist_kwargs: dict = {}
        monkeypatch.setattr(
            zero3_patches,
            "patch_zero3_fetch_trace",
            lambda config: None,
        )
        monkeypatch.setattr(
            zero3_patches,
            "install_family_zero3_patches",
            lambda _name: None,
        )

        def _persist(threshold, **kwargs):
            persist_kwargs["threshold"] = threshold
            persist_kwargs.update(kwargs)

        monkeypatch.setattr(
            zero3_patches,
            "patch_zero3_param_persistence",
            _persist,
        )

        install_zero3_patches(
            {
                "zero_optimization": {
                    "stage": 3,
                    "stage3_param_persistence_threshold": 50_000,
                    "stage3_model_persistence_threshold": 1_000_000,
                }
            },
            num_partitions=4,
        )

        assert persist_kwargs == {
            "threshold": 50_000,
            "model_persistence_threshold": 1_000_000,
            "num_partitions": 4,
        }

    @pytest.mark.parametrize(
        "config",
        [
            None,
            "not-a-mapping",
            {},
            {"zero_optimization": []},
            {"zero_optimization": {"stage": 3}},
        ],
        ids=[
            "none",
            "non_mapping",
            "empty",
            "zero_opt_not_mapping",
            "threshold_absent",
        ],
    )
    def test_skips_persistence_without_threshold(self, monkeypatch, config) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            zero3_patches,
            "patch_zero3_fetch_trace",
            lambda _config: calls.append("fetch"),
        )
        monkeypatch.setattr(
            zero3_patches,
            "patch_zero3_param_persistence",
            lambda *args, **kwargs: calls.append("persist"),
        )

        install_zero3_patches(config)

        assert calls == ["fetch"]


class TestZero3Resolvers:
    def test_resolve_zero3_targets_returns_none_pair_when_import_fails(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(zero3_patches, "try_import", lambda _path: None)

        assert zero3_patches._resolve_zero3_targets() == (None, None)

    def test_resolve_zero3_targets_raises_when_symbols_missing(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            zero3_patches,
            "try_import",
            lambda _path: type("M", (), {})(),
        )

        with pytest.raises(RuntimeError, match="missing"):
            zero3_patches._resolve_zero3_targets()

    def test_resolve_zero3_init_returns_none_when_module_missing(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(zero3_patches, "try_import", lambda _path: None)

        assert zero3_patches._resolve_zero3_init() is None

    def test_snapshot_skips_attributes_the_coordinator_lacks(self) -> None:
        assert zero3_patches._snapshot_trace_state(object()) == {}

    def test_resolve_zero3_init_returns_none_when_attribute_missing(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            zero3_patches,
            "try_import",
            lambda _path: type("M", (), {})(),
        )

        assert zero3_patches._resolve_zero3_init() is None


class TestRoutesToConditionalSubmodules:
    """Only a declared ``leaf_module`` marks a model as data-routed."""

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"zero_optimization": {}},
            {"zero_optimization": {"stage": 3}},
            {"zero_optimization": {"leaf_module": {}}},
            {"zero_optimization": None},
            {"zero_optimization": [("leaf_module", True)]},
            {"leaf_module": {"name_suffixes": ["experts"]}},
            "not-a-mapping",
        ],
    )
    def test_absent_or_empty_leaf_module_is_false(self, config):
        assert zero3_patches._routes_to_conditional_submodules(config) is False

    def test_declared_leaf_module_is_true(self):
        assert zero3_patches._routes_to_conditional_submodules(_LEAF_MODULE_CONFIG)

    def test_leaf_module_by_classes_is_true(self):
        config = {"zero_optimization": {"leaf_module": {"classes": ["NemotronHMoE"]}}}
        assert zero3_patches._routes_to_conditional_submodules(config)

    def test_not_exported_in_all(self):
        assert "routes_to_conditional_submodules" not in zero3_patches.__all__
        assert not hasattr(zero3_patches, "routes_to_conditional_submodules")


class TestLeafModuleForcesOnDemandFetch:
    """A data-routed model records no trace, so nothing is replayed."""

    def test_grad_forward_never_records_a_trace(self, coordinator_cls):
        patch_zero3_fetch_trace(_LEAF_MODULE_CONFIG)
        coordinator = coordinator_cls()
        _set_attr(coordinator, "__trace_mode", _FakeTraceMode.RECORD)

        coordinator.reset_step()

        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.INVALID

    def test_mode_holds_across_repeated_grad_steps(self, coordinator_cls):
        patch_zero3_fetch_trace(_LEAF_MODULE_CONFIG)
        coordinator = coordinator_cls()

        for _ in range(3):
            coordinator.reset_step()

        assert coordinator.reset_step_calls == 3
        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.INVALID

    def test_no_grad_forward_takes_no_snapshot(self, coordinator_cls):
        patch_zero3_fetch_trace(_LEAF_MODULE_CONFIG)
        coordinator = coordinator_cls()

        with torch.no_grad():
            coordinator.reset_step()

        assert getattr(coordinator, zero3_patches.SNAPSHOT_ATTR, None) is None
        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.INVALID

    def test_per_step_bookkeeping_still_resets(self, coordinator_cls):
        patch_zero3_fetch_trace(_LEAF_MODULE_CONFIG)
        coordinator = coordinator_cls()
        _simulate_no_grad_forward(coordinator)

        coordinator.reset_step()

        assert coordinator.reset_step_calls == 1
        assert _attr(coordinator, "__step_id") == 0
        assert _attr(coordinator, "__n_available_params") == 0
        assert _attr(coordinator, "__ongoing_fetch_leaf_module_events") == set()

    def test_config_without_leaf_module_keeps_trace_recording(self, coordinator_cls):
        patch_zero3_fetch_trace({"zero_optimization": {"stage": 2}})
        coordinator = coordinator_cls()
        _set_attr(coordinator, "__trace_mode", _FakeTraceMode.RECORD)

        coordinator.reset_step()

        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.RECORD

    def test_omitted_config_keeps_trace_recording(self, coordinator_cls):
        patch_zero3_fetch_trace()
        coordinator = coordinator_cls()
        _set_attr(coordinator, "__trace_mode", _FakeTraceMode.RECORD)

        coordinator.reset_step()

        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.RECORD


class TestZero3TraceSnapshotBranch:
    """Without ``leaf_module`` the trace identity survives no-grad forwards."""

    def test_no_grad_entry_calls_original_reset_step(self, coordinator_cls):
        patch_zero3_fetch_trace()
        coordinator = coordinator_cls()
        _simulate_no_grad_forward(coordinator)

        with torch.no_grad():
            coordinator.reset_step()

        assert coordinator.reset_step_calls == 1
        assert _attr(coordinator, "__step_id") == 0
        assert _attr(coordinator, "__n_available_params") == 0
        assert _attr(coordinator, "__ongoing_fetch_leaf_module_events") == set()
        assert not _attr(coordinator, "__step_id_module_fetched_for")
        assert list(_attr(coordinator, "__param_queue")) == [("p0", 0), ("p1", 1)]

    def test_no_grad_entry_snapshots_and_forces_invalid(self, coordinator_cls):
        patch_zero3_fetch_trace()
        coordinator = coordinator_cls()

        with torch.no_grad():
            coordinator.reset_step()

        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.INVALID
        assert coordinator.trace_modes_at_reset == [_FakeTraceMode.COMPLETE]
        snapshot = getattr(coordinator, zero3_patches.SNAPSHOT_ATTR)
        assert snapshot is not None
        mangled_trace_mode = zero3_patches._mangled("__trace_mode")
        assert snapshot[mangled_trace_mode] is _FakeTraceMode.COMPLETE
        assert set(snapshot) == {
            zero3_patches._mangled(name) for name in zero3_patches.TRACE_ATTRS
        }

    def test_snapshot_excludes_per_step_bookkeeping(self, coordinator_cls):
        patch_zero3_fetch_trace()
        coordinator = coordinator_cls()

        with torch.no_grad():
            coordinator.reset_step()

        snapshot = getattr(coordinator, zero3_patches.SNAPSHOT_ATTR)
        for name in _DROPPED_ATTRS:
            assert name not in zero3_patches.TRACE_ATTRS
            assert zero3_patches._mangled(name) not in snapshot

    def test_snapshot_containers_are_not_aliased(self, coordinator_cls):
        patch_zero3_fetch_trace()
        coordinator = coordinator_cls()

        with torch.no_grad():
            coordinator.reset_step()
        _simulate_no_grad_forward(coordinator)

        snapshot = getattr(coordinator, zero3_patches.SNAPSHOT_ATTR)
        assert len(snapshot[zero3_patches._mangled("__submodule_order")]) == 2
        assert len(snapshot[zero3_patches._mangled("__param_order")]) == 2

    def test_frozen_tuple_orders_round_trip(self, coordinator_cls):
        patch_zero3_fetch_trace()
        coordinator = coordinator_cls()
        submodules = tuple(_attr(coordinator, "__submodule_order"))
        params = tuple(_attr(coordinator, "__param_order"))
        _set_attr(coordinator, "__submodule_order", submodules)
        _set_attr(coordinator, "__param_order", params)

        with torch.no_grad():
            coordinator.reset_step()
        _set_attr(coordinator, "__submodule_order", [_FakeModule(99)])
        _set_attr(coordinator, "__param_order", [("p9", 9)])
        coordinator.reset_step()

        restored = _attr(coordinator, "__submodule_order")
        assert isinstance(restored, tuple)
        assert restored == submodules
        assert _attr(coordinator, "__param_order") == params
        assert list(_attr(coordinator, "__param_queue")) == list(params)

    def test_repeated_no_grad_entries_keep_first_snapshot(self, coordinator_cls):
        patch_zero3_fetch_trace()
        coordinator = coordinator_cls()

        with torch.no_grad():
            coordinator.reset_step()
            _simulate_no_grad_forward(coordinator)
            coordinator.reset_step()

        snapshot = getattr(coordinator, zero3_patches.SNAPSHOT_ATTR)
        assert len(snapshot[zero3_patches._mangled("__submodule_order")]) == 2
        assert (
            snapshot[zero3_patches._mangled("__trace_mode")] is _FakeTraceMode.COMPLETE
        )
        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.INVALID
        assert coordinator.reset_step_calls == 2

    def test_grad_entry_after_no_grad_restores_and_calls_through(self, coordinator_cls):
        patch_zero3_fetch_trace()
        coordinator = coordinator_cls()
        expected_modules = list(_attr(coordinator, "__submodule_order"))
        expected_params = list(_attr(coordinator, "__param_order"))

        with torch.no_grad():
            coordinator.reset_step()
        _simulate_no_grad_forward(coordinator)
        coordinator.reset_step()

        assert coordinator.reset_step_calls == 2
        assert coordinator.trace_modes_at_reset[-1] is _FakeTraceMode.COMPLETE
        assert _attr(coordinator, "__submodule_order") == expected_modules
        assert _attr(coordinator, "__param_order") == expected_params
        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.COMPLETE
        assert _attr(coordinator, "__step_id") == 0
        assert list(_attr(coordinator, "__param_queue")) == expected_params
        assert getattr(coordinator, zero3_patches.SNAPSHOT_ATTR) is None

    def test_grad_entry_without_snapshot_calls_through(self, coordinator_cls):
        patch_zero3_fetch_trace()
        coordinator = coordinator_cls()
        _set_attr(coordinator, "__step_id", 17)

        coordinator.reset_step()

        assert coordinator.reset_step_calls == 1
        assert _attr(coordinator, "__step_id") == 0
        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.COMPLETE
        assert getattr(coordinator, zero3_patches.SNAPSHOT_ATTR, None) is None


class TestZero3TraceInstallation:
    def test_apply_is_idempotent(self, coordinator_cls, caplog):
        with caplog.at_level(logging.INFO):
            patch_zero3_fetch_trace()
            first_reset_step = coordinator_cls.reset_step
            patch_zero3_fetch_trace()

        assert coordinator_cls.reset_step is first_reset_step
        install_lines = [
            record.message
            for record in caplog.records
            if "reset_step patched" in record.message
        ]
        assert len(install_lines) == 1

        coordinator = coordinator_cls()
        with torch.no_grad():
            coordinator.reset_step()
        coordinator.reset_step()
        assert coordinator.reset_step_calls == 2

    def test_disabled_leaves_reset_step_unpatched(self, coordinator_cls, caplog):
        original_reset_step = coordinator_cls.reset_step

        with caplog.at_level(logging.INFO):
            patch_zero3_fetch_trace(_LEAF_MODULE_CONFIG, enabled=False)

        assert coordinator_cls.reset_step is original_reset_step
        assert any("disabled by caller" in record.message for record in caplog.records)

        coordinator = coordinator_cls()
        with torch.no_grad():
            coordinator.reset_step()
        assert _attr(coordinator, "__trace_mode") is _FakeTraceMode.COMPLETE

    def test_missing_attribute_raises(self, monkeypatch):
        class PartitionedParameterCoordinator:
            def __init__(self):
                self.__trace_mode = _FakeTraceMode.COMPLETE
                self.__submodule_order = []
                self.reset_step_calls = 0

            def reset_step(self):
                self.reset_step_calls += 1

        monkeypatch.setattr(
            zero3_patches,
            "_resolve_zero3_targets",
            lambda: (PartitionedParameterCoordinator, _FakeTraceMode),
        )

        with pytest.raises(RuntimeError, match="does not define"):
            patch_zero3_fetch_trace()

        assert not getattr(
            PartitionedParameterCoordinator,
            zero3_patches.TRACE_PATCHED_FLAG,
            False,
        )

    def test_missing_reset_step_raises(self, monkeypatch):
        class PartitionedParameterCoordinator:
            def __init__(self):
                self.__trace_mode = _FakeTraceMode.COMPLETE
                self.__submodule_order = []
                self.__param_order = []

        monkeypatch.setattr(
            zero3_patches,
            "_resolve_zero3_targets",
            lambda: (PartitionedParameterCoordinator, _FakeTraceMode),
        )

        with pytest.raises(RuntimeError, match="lacks reset_step"):
            patch_zero3_fetch_trace()

    def test_missing_coordinator_class_is_a_no_op(self, monkeypatch, caplog):
        monkeypatch.setattr(
            zero3_patches,
            "_resolve_zero3_targets",
            lambda: (None, None),
        )

        with caplog.at_level(logging.WARNING):
            patch_zero3_fetch_trace()

        assert any(
            "coordinator unavailable" in record.message for record in caplog.records
        )


class _FakeZero3Init:
    """Fake ``zero.Init`` exposing only the persistence class attributes."""

    param_persistence_threshold = _DEFAULT_PARAM_THRESHOLD
    model_persistence_threshold = _DEFAULT_MODEL_THRESHOLD
    num_persisted_parameters = 0
    num_persisted_elements = 0
    apply_param_persistence = False


@pytest.fixture
def zero3_init_cls(monkeypatch):
    monkeypatch.setattr(
        zero3_patches,
        "_resolve_zero3_init",
        lambda: _FakeZero3Init,
    )
    yield _FakeZero3Init
    _FakeZero3Init.apply_param_persistence = False
    _FakeZero3Init.param_persistence_threshold = _DEFAULT_PARAM_THRESHOLD
    _FakeZero3Init.model_persistence_threshold = _DEFAULT_MODEL_THRESHOLD


class TestZero3ParamPersistence:
    def test_sets_flag_and_param_threshold(self, zero3_init_cls, caplog):
        with caplog.at_level(logging.INFO):
            patch_zero3_param_persistence(50_000)

        assert zero3_init_cls.apply_param_persistence is True
        assert zero3_init_cls.param_persistence_threshold == 50_000
        assert any(
            "param_threshold=50000" in record.message for record in caplog.records
        )

    def test_model_threshold_is_divided_by_num_partitions(self, zero3_init_cls):
        patch_zero3_param_persistence(
            100_000,
            model_persistence_threshold=1_000_000_001,
            num_partitions=4,
        )

        assert zero3_init_cls.model_persistence_threshold == 250_000_000

    def test_model_threshold_untouched_when_not_supplied(self, zero3_init_cls):
        patch_zero3_param_persistence(100_000, num_partitions=8)

        assert zero3_init_cls.model_persistence_threshold == _DEFAULT_MODEL_THRESHOLD
        assert zero3_init_cls.apply_param_persistence is True

    def test_apply_is_idempotent(self, zero3_init_cls):
        for _ in range(3):
            patch_zero3_param_persistence(
                100_000,
                model_persistence_threshold=_DEFAULT_MODEL_THRESHOLD,
                num_partitions=2,
            )

        assert zero3_init_cls.apply_param_persistence is True
        assert zero3_init_cls.param_persistence_threshold == 100_000
        assert (
            zero3_init_cls.model_persistence_threshold == _DEFAULT_MODEL_THRESHOLD // 2
        )

    def test_missing_attribute_leaves_init_untouched(self, monkeypatch, caplog):
        class Init:
            param_persistence_threshold = _DEFAULT_PARAM_THRESHOLD
            model_persistence_threshold = _DEFAULT_MODEL_THRESHOLD

        monkeypatch.setattr(
            zero3_patches,
            "_resolve_zero3_init",
            lambda: Init,
        )

        with caplog.at_level(logging.WARNING):
            patch_zero3_param_persistence(
                50_000,
                model_persistence_threshold=800,
                num_partitions=2,
            )

        assert not hasattr(Init, "apply_param_persistence")
        assert Init.param_persistence_threshold == _DEFAULT_PARAM_THRESHOLD
        assert Init.model_persistence_threshold == _DEFAULT_MODEL_THRESHOLD
        warnings = [
            record.message
            for record in caplog.records
            if "does not define" in record.message
        ]
        assert warnings
        assert "apply_param_persistence" in warnings[0]

    def test_missing_init_class_is_a_no_op(self, monkeypatch, caplog):
        monkeypatch.setattr(
            zero3_patches,
            "_resolve_zero3_init",
            lambda: None,
        )

        with caplog.at_level(logging.WARNING):
            patch_zero3_param_persistence(50_000)

        assert any(
            "zero.Init unavailable" in record.message for record in caplog.records
        )

    def test_non_positive_num_partitions_falls_back_to_one(
        self, zero3_init_cls, caplog
    ):
        with caplog.at_level(logging.WARNING):
            patch_zero3_param_persistence(
                100_000,
                model_persistence_threshold=900,
                num_partitions=0,
            )

        assert zero3_init_cls.model_persistence_threshold == 900
        assert any("not positive" in record.message for record in caplog.records)

    def test_disabled_leaves_persistence_untouched(self, zero3_init_cls, caplog):
        with caplog.at_level(logging.INFO):
            patch_zero3_param_persistence(
                50_000,
                model_persistence_threshold=800,
                num_partitions=2,
                enabled=False,
            )

        assert zero3_init_cls.apply_param_persistence is False
        assert zero3_init_cls.param_persistence_threshold == _DEFAULT_PARAM_THRESHOLD
        assert zero3_init_cls.model_persistence_threshold == _DEFAULT_MODEL_THRESHOLD
        assert any("disabled by caller" in record.message for record in caplog.records)
