# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for agilerl.utils.env_utils."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agilerl.utils import env_utils


class TestEscapeNonFormatBraces:
    def test_preserves_format_keys(self):
        template = "Q: {question} A: {answer} meta: {other}"
        result = env_utils.escape_non_format_braces(template)
        assert "{question}" in result
        assert "{answer}" in result
        assert "{{other}}" in result

    def test_custom_format_keys(self):
        template = "keep {foo} escape {bar}"
        result = env_utils.escape_non_format_braces(template, format_keys=["foo"])
        assert "{foo}" in result
        assert "{{bar}}" in result

    def test_empty_braces_survive_format(self):
        template = "reply with {} when done, {question}?"
        result = env_utils.escape_non_format_braces(template)
        assert result.format(question="q") == "reply with {} when done, q?"

    def test_lone_braces_survive_format(self):
        template = "open { and close } alone, {question}"
        result = env_utils.escape_non_format_braces(template)
        assert result.format(question="q") == "open { and close } alone, q"

    def test_nested_json_survives_format(self):
        template = 'return {"result": {"answer": 1}} for {question}'
        result = env_utils.escape_non_format_braces(template)
        assert result.format(question="q") == 'return {"result": {"answer": 1}} for q'


class TestParseEntrypoint:
    def test_rejects_missing_colon(self):
        with pytest.raises(ValueError, match="Invalid entrypoint format"):
            env_utils._parse_entrypoint("NoColonHere")

    def test_rejects_empty_module_or_target(self):
        with pytest.raises(ValueError, match="both module and target"):
            env_utils._parse_entrypoint(":OnlyTarget")
        with pytest.raises(ValueError, match="both module and target"):
            env_utils._parse_entrypoint("only_module:")

    def test_parses_valid_entrypoint(self):
        assert env_utils._parse_entrypoint("env.py:MyEnv") == ("env.py", "MyEnv")


class TestGetRewardFn:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "missing.py"
        with pytest.raises(ValueError, match="not found"):
            env_utils.get_reward_fn("reward", str(missing))

    def test_import_error_wrapped(self, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text(
            "raise RuntimeError('intentional import failure')\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Error importing reward function"):
            env_utils.get_reward_fn("reward", str(bad_file))

    def test_spec_none_raises(self, tmp_path):
        reward_file = tmp_path / "reward.py"
        reward_file.write_text("def reward(): return 1.0\n", encoding="utf-8")
        with patch(
            "agilerl.utils.env_utils.importlib_util.spec_from_file_location",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="Could not create spec"):
                env_utils.get_reward_fn("reward", str(reward_file))

    def test_loads_callable(self, tmp_path):
        reward_file = tmp_path / "reward.py"
        reward_file.write_text("def my_reward(obs): return 1.0\n", encoding="utf-8")
        fn = env_utils.get_reward_fn("my_reward", str(reward_file))
        assert callable(fn)
        assert fn({}) == 1.0


class TestResolveWrapper:
    def test_invalid_string_wrapper_raises(self):
        with pytest.raises(ValueError, match="Invalid wrapper"):
            env_utils._resolve_wrapper("not_a_valid_wrapper_spec")

    def test_non_callable_resolved_wrapper_raises(self, tmp_path):
        target_file = tmp_path / "mod.py"
        target_file.write_text("VALUE = 1\n", encoding="utf-8")
        with pytest.raises(TypeError, match="non-callable"):
            env_utils._resolve_wrapper(f"{target_file.name}:VALUE", path=str(tmp_path))

    def test_dotted_import_path(self):
        fn, kwargs = env_utils._resolve_wrapper("os.path.join")
        assert fn is not None
        assert kwargs == {}

    def test_entrypoint_resolution(self, tmp_path):
        target_file = tmp_path / "wrap.py"
        target_file.write_text(
            "def wrap(env, scale=1):\n    return env\n",
            encoding="utf-8",
        )
        fn, kwargs = env_utils._resolve_wrapper(
            (f"{target_file.name}:wrap", {"scale": 2}),
            path=str(tmp_path),
        )
        assert callable(fn)
        assert kwargs == {"scale": 2}


class TestLoadModuleFromPath:
    def test_spec_without_loader_raises(self, tmp_path):
        script = tmp_path / "script.py"
        script.write_text("x = 1\n", encoding="utf-8")
        with patch(
            "agilerl.utils.env_utils.importlib_util.spec_from_file_location",
            return_value=MagicMock(loader=None),
        ):
            with pytest.raises(ImportError, match="Could not load module"):
                env_utils._load_module_from_path("script", script)
