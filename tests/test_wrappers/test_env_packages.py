# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for installing the dependencies an env declares via ``env_packages``."""

from __future__ import annotations

import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest

from agilerl.llm_envs.env_packages import ensure_importable, package_list


class TestPackageList:
    """``env_packages`` is a Ray-shaped runtime env; both key styles name the same thing."""

    @pytest.mark.parametrize(
        ("env_packages", "expected"),
        [
            ({"uv": ["gem-llm"]}, ["gem-llm"]),
            ({"pip": ["gem-llm", "numpy"]}, ["gem-llm", "numpy"]),
            ({"uv": {"packages": ["gem-llm"]}}, ["gem-llm"]),
            ({"uv": ["a"], "pip": ["b"]}, ["a", "b"]),
        ],
    )
    def test_flattens_both_key_styles(
        self, env_packages: dict[str, Any], expected: list[str]
    ) -> None:
        assert package_list(env_packages) == expected

    @pytest.mark.parametrize("env_packages", [{}, {"uv": []}, {"conda": ["x"]}])
    def test_naming_no_packages_is_an_error(self, env_packages: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="named no packages"):
            package_list(env_packages)

    def test_string_value_is_one_package(self) -> None:
        assert package_list({"uv": "gem-llm"}) == ["gem-llm"]
        assert package_list({"pip": "numpy"}) == ["numpy"]

    def test_leading_dash_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="starts with '-'"):
            package_list({"uv": ["--index-url", "http://evil/simple", "x"]})

    def test_non_sequence_value_is_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"env_packages\.uv"):
            package_list({"uv": 1})

    def test_non_string_item_in_a_sequence_is_rejected(self) -> None:
        # A YAML list can hold anything; a non-string would reach the installer
        # command line unquoted.
        with pytest.raises(TypeError, match="sequence of strings"):
            package_list({"uv": ["gem-llm", 7]})

    def test_non_string_item_names_the_offending_type(self) -> None:
        with pytest.raises(TypeError, match="got dict"):
            package_list({"pip": [{"name": "numpy"}]})


class TestEnsureImportable:
    """Packages are installed only when the env actually cannot be imported."""

    def test_importable_entrypoint_installs_nothing(self) -> None:
        with patch("agilerl.llm_envs.env_packages._install") as install:
            ensure_importable(
                "agilerl.llm_envs.env_packages:package_list", {"uv": ["x"]}
            )
        install.assert_not_called()

    def test_missing_entrypoint_installs_into_this_interpreter(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("subprocess.run") as run,
            patch("importlib.invalidate_caches") as invalidate,
        ):
            run.return_value = subprocess.CompletedProcess([], returncode=0)
            ensure_importable("no_such_module:Env", {"uv": ["gem-llm"]})
        assert run.call_args.args[0] == [
            "/usr/bin/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--",
            "gem-llm",
        ]
        # A fresh install goes into a directory the import system has already
        # listed, so it stays invisible until the caches are dropped.
        invalidate.assert_called_once()

    def test_an_entrypoint_missing_its_attribute_is_not_an_install_problem(
        self,
    ) -> None:
        with patch("agilerl.llm_envs.env_packages._install") as install:
            with pytest.raises(AttributeError):
                ensure_importable("agilerl.llm_envs.env_packages:nope", {"uv": ["x"]})
        install.assert_not_called()

    def test_without_uv_it_says_to_install_them_yourself(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="uv, which is not on PATH"):
                ensure_importable("no_such_module:Env", {"uv": ["gem-llm"]})

    def test_unresolvable_packages_point_at_a_dedicated_env_host(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess([], returncode=1)
            with pytest.raises(RuntimeError, match="needs its own environment") as err:
                ensure_importable("no_such_module:Env", {"uv": ["gem-llm"]})
        assert "env_url" in str(err.value)
