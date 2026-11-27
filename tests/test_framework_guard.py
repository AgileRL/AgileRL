# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.machinery
import subprocess
import sys

import pytest

import agilerl
from agilerl import _framework_guard


class TestIsFrameworkModule:
    @pytest.mark.parametrize(
        "name",
        [
            "agilerl.algorithms",
            "agilerl.algorithms.core.base",
            "agilerl.training.train_off_policy",
            "agilerl.utils.utils",
            "agilerl.wrappers.agent",
        ],
    )
    def test_2x_framework_paths_match(self, name):
        assert _framework_guard.is_framework_module(name)

    @pytest.mark.parametrize(
        "name",
        [
            "agilerl",
            "agilerl.utils",
            "agilerl.models.env",
            "agilerl.cli",
            "agilerl.arena.cli",
            "agilerl_ray.core.algorithms",
            "algorithms",
        ],
    )
    def test_other_paths_do_not_match(self, name):
        assert not _framework_guard.is_framework_module(name)


class TestIsRenamedModule:
    @pytest.mark.parametrize(
        "name",
        [
            "agilerl.arena",
            "agilerl.arena.cli",
            "agilerl.arena.client",
            "agilerl.arena.models.env",
        ],
    )
    def test_agilerl_arena_paths_match(self, name):
        assert _framework_guard.is_renamed_module(name)

    @pytest.mark.parametrize("name", ["agilerl.arenas", "agilerl.client", "arena"])
    def test_other_paths_do_not_match(self, name):
        assert not _framework_guard.is_renamed_module(name)


class TestAgilerlArenaImports:
    """agilerl-arena 1.x imports name the agilerl 3 import that replaced them."""

    @pytest.mark.parametrize(
        "statement",
        ["import agilerl.arena", "from agilerl.arena import ArenaClient"],
    )
    def test_1x_import_names_the_rename(self, statement):
        with pytest.raises(ModuleNotFoundError) as raised:
            exec(statement, {})

        message = str(raised.value)
        assert "The SDK is now the agilerl package (formerly agilerl-arena)" in message
        assert "import 'agilerl' instead of 'agilerl.arena'" in message
        assert "from agilerl import ArenaClient" in message
        assert "agilerl<3" not in message


class TestFrameworkImportGuard:
    @pytest.mark.parametrize(
        "name", ["agilerl.algorithms", "agilerl.algorithms.ppo", "agilerl.utils.utils"]
    )
    def test_2x_import_names_the_framework_pin(self, name):
        with pytest.raises(ModuleNotFoundError, match=r"pip install 'agilerl<3'"):
            importlib.import_module(name)

    def test_2x_from_import_names_the_framework_pin(self):
        with pytest.raises(ModuleNotFoundError, match=r"agilerl<3") as raised:
            exec("from agilerl.utils.utils import create_population", {})

        assert raised.value.name == "agilerl"

    def test_pin_advice_restores_the_arena_command_in_place(self):
        message = _framework_guard.pin_message("No module named 'agilerl.algorithms'")

        assert "python -m venv .venv-agilerl2" in message
        assert "pip install --force-reinstall --no-deps 'agilerl-arena<2'" in message

    @pytest.mark.parametrize(
        "name", ["agilerl.cli", "agilerl.utils", "agilerl.models.env", "agilerl.typing"]
    )
    def test_cli_modules_still_import(self, name):
        assert importlib.import_module(name).__name__ == name

    def test_unknown_module_keeps_the_plain_error(self):
        with pytest.raises(ModuleNotFoundError) as raised:
            importlib.import_module("agilerl.not_a_module")

        assert "agilerl<3" not in str(raised.value)

    @pytest.mark.parametrize(
        "statement",
        [
            "from agilerl import algorithms",
            "from agilerl import training",
            "import agilerl.algorithms",
            "from agilerl.algorithms import GRPO",
        ],
    )
    def test_2x_name_from_import_names_the_framework_pin(self, statement):
        with pytest.raises(ModuleNotFoundError, match=r"pip install 'agilerl<3'"):
            exec(statement, {})

    @pytest.mark.parametrize("name", ["training", "algorithms"])
    def test_2x_attribute_is_missing_for_hasattr(self, name):
        assert not hasattr(agilerl, name)

    def test_2x_attribute_access_names_the_framework_pin(self):
        with pytest.raises(AttributeError, match=r"agilerl<3") as raised:
            _ = agilerl.training

        assert raised.value.name == "training"

    def test_3x_names_still_import(self):
        exec("from agilerl import ArenaClient", {})
        exec("from agilerl.typing import JSONValue", {})
        exec("from agilerl.utils import prepare_env_upload", {})

    def test_unknown_agilerl_name_keeps_the_plain_error(self):
        with pytest.raises(ImportError) as raised:
            exec("from agilerl import not_a_name", {})

        assert "agilerl<3" not in str(raised.value)

    @pytest.mark.parametrize("module", ["agilerl.typing", "agilerl.utils"])
    def test_unknown_names_on_cli_modules_are_missing_quietly(self, module):
        assert not hasattr(importlib.import_module(module), "ArrayOrTensor")

    def test_unknown_agilerl_attribute_is_missing_quietly(self):
        assert not hasattr(importlib.import_module("agilerl"), "not_a_name")

    def test_import_agilerl_installs_the_guard_once_ahead_of_pathfinder(self):
        # Arrange
        code = (
            "import sys, importlib, importlib.machinery, agilerl\n"
            "from agilerl import _framework_guard\n"
            "importlib.reload(agilerl)\n"
            "guards = [i for i, f in enumerate(sys.meta_path) "
            "if isinstance(f, _framework_guard.FrameworkImportGuard)]\n"
            "path_at = sys.meta_path.index(importlib.machinery.PathFinder)\n"
            "print(len(guards), guards[0] == path_at - 1)\n"
        )

        # Act
        ran = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        # Assert
        assert ran.stdout == "1 True\n"

    def test_install_leaves_an_existing_guard_in_place(self):
        before = list(sys.meta_path)

        _framework_guard.install()

        assert sys.meta_path == before

    def test_install_appends_when_pathfinder_is_missing(self, monkeypatch):
        meta = [
            finder
            for finder in sys.meta_path
            if finder is not importlib.machinery.PathFinder
            and not isinstance(finder, _framework_guard.FrameworkImportGuard)
        ]
        monkeypatch.setattr(sys, "meta_path", meta)

        _framework_guard.install()

        assert isinstance(sys.meta_path[-1], _framework_guard.FrameworkImportGuard)

    def test_leftover_1x_arena_tree_names_the_rename(self, tmp_path, monkeypatch):
        # Arrange
        root = tmp_path / "agilerl"
        arena = root / "arena"
        arena.mkdir(parents=True)
        (arena / "__init__.py").write_text("LEGACY = True\n")
        (arena / "client.py").write_text("LEGACY_CLIENT = True\n")
        monkeypatch.setattr(agilerl, "__path__", [str(root), *agilerl.__path__])
        for name in ("agilerl.arena", "agilerl.arena.client"):
            sys.modules.pop(name, None)

        # Act / Assert
        try:
            with pytest.raises(
                ModuleNotFoundError,
                match=r"import 'agilerl' instead of 'agilerl\.arena'",
            ):
                importlib.import_module("agilerl.arena")
            with pytest.raises(
                ModuleNotFoundError,
                match=r"import 'agilerl' instead of 'agilerl\.arena'",
            ):
                importlib.import_module("agilerl.arena.client")
            assert "agilerl.arena.client" not in sys.modules
        finally:
            for name in ("agilerl.arena", "agilerl.arena.client"):
                sys.modules.pop(name, None)

    def test_namespace_arena_without_init_names_the_rename(self, tmp_path, monkeypatch):
        # Arrange
        root = tmp_path / "agilerl"
        (root / "arena").mkdir(parents=True)
        monkeypatch.setattr(agilerl, "__path__", [str(root), *agilerl.__path__])
        sys.modules.pop("agilerl.arena", None)

        # Act / Assert
        try:
            with pytest.raises(
                ModuleNotFoundError,
                match=r"import 'agilerl' instead of 'agilerl\.arena'",
            ):
                importlib.import_module("agilerl.arena")
        finally:
            sys.modules.pop("agilerl.arena", None)

    def test_arena_submodule_without_package_init_names_the_rename(
        self, tmp_path, monkeypatch
    ):
        # Arrange
        arena = tmp_path / "agilerl" / "arena"
        arena.mkdir(parents=True)
        (arena / "client.py").write_text("LEGACY_CLIENT = True\n")
        monkeypatch.setattr(
            agilerl, "__path__", [str(tmp_path / "agilerl"), *agilerl.__path__]
        )
        for name in ("agilerl.arena", "agilerl.arena.client"):
            sys.modules.pop(name, None)

        # Act / Assert
        try:
            with pytest.raises(
                ModuleNotFoundError,
                match=r"import 'agilerl' instead of 'agilerl\.arena'",
            ):
                importlib.import_module("agilerl.arena.client")
            assert "agilerl.arena.client" not in sys.modules
        finally:
            for name in ("agilerl.arena", "agilerl.arena.client"):
                sys.modules.pop(name, None)

    def test_framework_module_present_on_the_path_imports(self, tmp_path, monkeypatch):
        # Arrange
        root = tmp_path / "agilerl"
        algorithms = root / "algorithms"
        algorithms.mkdir(parents=True)
        (algorithms / "__init__.py").write_text("PRESENT = True\n")
        monkeypatch.setattr(agilerl, "__path__", [str(root), *agilerl.__path__])
        sys.modules.pop("agilerl.algorithms", None)

        # Act
        try:
            loaded = importlib.import_module("agilerl.algorithms")
        finally:
            sys.modules.pop("agilerl.algorithms", None)

        # Assert
        assert loaded.PRESENT is True
