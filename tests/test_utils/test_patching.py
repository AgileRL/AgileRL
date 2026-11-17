# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared class-level patch primitives."""

import importlib
from importlib.metadata import version

import pytest
from packaging.specifiers import SpecifierSet

from agilerl.utils import patching
from agilerl.utils.patching import (
    LibraryShim,
    UnsupportedLibraryVersionError,
    library_shim,
)

# Every framework module that declares a library shim.
SHIM_MODULES = (
    "agilerl.algorithms.core.llm_ops.vllm_colocate",
    "agilerl.architectures.catalog",
    "agilerl.architectures.flex_attention",
    "agilerl.architectures.nemotron_h.liger",
    "agilerl.architectures.nemotron_h.mamba",
    "agilerl.architectures.qwen3_5.packed",
    "agilerl.distributed.fsdp_blocks",
    "agilerl.lora.fused",
)


@pytest.fixture
def empty_registry(monkeypatch: pytest.MonkeyPatch) -> list[LibraryShim]:
    registry: list[LibraryShim] = []
    monkeypatch.setattr(patching, "LIBRARY_SHIMS", registry)
    return registry


class TestTryImport:
    def test_returns_the_module_when_it_imports(self) -> None:
        assert patching.try_import("json") is not None

    def test_returns_none_on_failure(self, monkeypatch) -> None:
        def _boom(_name: str):
            msg = "missing"
            raise ImportError(msg)

        monkeypatch.setattr(patching.importlib, "import_module", _boom)

        assert patching.try_import("not.a.real.module") is None


class TestClassIsPatched:
    def test_false_when_the_flag_is_absent(self) -> None:
        assert not patching.class_is_patched(type("C", (), {}), "_flag")

    def test_true_when_the_class_carries_the_flag(self) -> None:
        assert patching.class_is_patched(type("C", (), {"_flag": True}), "_flag")

    def test_inherited_flag_does_not_count_as_patched(self) -> None:
        base = type("Base", (), {"_flag": True})

        assert not patching.class_is_patched(type("Child", (base,), {}), "_flag")


class TestLibraryShim:
    def test_runs_the_shim_inside_its_range(
        self, empty_registry: list[LibraryShim]
    ) -> None:
        @library_shim("packaging", f"=={version('packaging')}")
        def shim(value: int) -> int:
            return value + 1

        assert shim(1) == 2

    def test_refuses_to_run_outside_its_range(
        self, empty_registry: list[LibraryShim]
    ) -> None:
        # Arrange
        calls: list[int] = []

        @library_shim("packaging", "<1")
        def shim() -> None:
            calls.append(1)

        # Act / Assert
        with pytest.raises(
            UnsupportedLibraryVersionError,
            match=rf"shim was checked against packaging<1, but packaging "
            rf"{version('packaging')} is installed",
        ):
            shim()
        assert calls == []

    def test_runs_the_shim_when_the_library_is_not_installed(
        self, empty_registry: list[LibraryShim]
    ) -> None:
        @library_shim("agilerl-no-such-distribution", "==1.0.*")
        def shim() -> str:
            return "ran"

        assert shim() == "ran"

    def test_registers_the_shim_under_its_qualified_name(
        self, empty_registry: list[LibraryShim]
    ) -> None:
        @library_shim("vllm", "==0.32.*")
        def shim() -> None: ...

        assert empty_registry == [
            LibraryShim(
                name=f"{__name__}.TestLibraryShim."
                "test_registers_the_shim_under_its_qualified_name.<locals>.shim",
                package="vllm",
                versions=SpecifierSet("==0.32.*"),
            )
        ]

    @pytest.mark.parametrize("installed", ["0.32.1.dev3+g1234abc", "2.13.0+cu130"])
    def test_minor_range_covers_dev_and_local_builds(
        self, monkeypatch: pytest.MonkeyPatch, installed: str
    ) -> None:
        monkeypatch.setattr(patching, "version", lambda _package: installed)
        release = ".".join(installed.split(".")[:2])
        shim = LibraryShim("shim", "torch", SpecifierSet(f"=={release}.*"))

        shim.check()


class TestLibraryShimsRegistry:
    def test_every_shim_covers_the_installed_release(self) -> None:
        # Arrange
        for module in SHIM_MODULES:
            importlib.import_module(module)

        # Act
        stale = [
            f"{shim.name}: {shim.package}{shim.versions}, installed "
            f"{shim.installed_version()}"
            for shim in patching.LIBRARY_SHIMS
            if not shim.covers_installed()
        ]

        # Assert
        assert stale == [], (
            "Re-check these shims on the installed releases, then widen their "
            "range or delete them:\n" + "\n".join(stale)
        )
