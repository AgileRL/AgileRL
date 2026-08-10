# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared class-level patch primitives."""

from agilerl.utils import patching


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
