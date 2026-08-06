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

    def test_windows_drive_path_splits_on_last_colon(self):
        r"""``C:\...\file.py:Class`` splits at the trailing colon, keeping the drive."""
        module_ref, target = env_utils._parse_entrypoint(r"C:\tmp\disk_env.py:DiskEnv")
        assert module_ref == r"C:\tmp\disk_env.py"
        assert target == "DiskEnv"


class TestGetRubricFactory:
    def test_module_executes_once_across_builds(self, tmp_path):
        """The factory pays the module's import cost once, however many it builds."""
        rubric_file = tmp_path / "counting_rubric.py"
        rubric_file.write_text(
            "from openenv.core.rubrics.base import Rubric\n"
            "EXECUTIONS = []\n"
            "EXECUTIONS.append(1)\n"
            "class Counting(Rubric):\n"
            "    def forward(self, action, observation):\n"
            "        return 1.0\n",
            encoding="utf-8",
        )
        factory = env_utils.get_rubric_factory("Counting", str(rubric_file))
        built = [factory() for _ in range(3)]
        assert len({id(rubric) for rubric in built}) == 3
        assert type(built[0]).forward.__globals__["EXECUTIONS"] == [1]

    def test_reward_callable_builds_fresh_instances(self, tmp_path):
        rubric_file = tmp_path / "reward.py"
        rubric_file.write_text(
            "def reward(completion, answer, question):\n    return 1.0\n",
            encoding="utf-8",
        )
        factory = env_utils.get_rubric_factory("reward", str(rubric_file))
        assert factory() is not factory()

    def test_module_level_instance_is_shared(self, tmp_path):
        """An author's own singleton rubric is handed back as-is, not copied."""
        rubric_file = tmp_path / "singleton.py"
        rubric_file.write_text(
            "from openenv.core.rubrics.base import Rubric\n"
            "class Single(Rubric):\n"
            "    def forward(self, action, observation):\n"
            "        return 1.0\n"
            "RUBRIC = Single()\n",
            encoding="utf-8",
        )
        factory = env_utils.get_rubric_factory("RUBRIC", str(rubric_file))
        assert factory() is factory()


class TestGetRubric:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "missing.py"
        with pytest.raises(ValueError, match="not found"):
            env_utils.get_rubric("rubric", str(missing))

    def test_import_error_wrapped(self, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text(
            "raise RuntimeError('intentional import failure')\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Error importing rubric"):
            env_utils.get_rubric("rubric", str(bad_file))

    def test_spec_none_raises(self, tmp_path):
        rubric_file = tmp_path / "rubric.py"
        rubric_file.write_text("def rubric(): return 1.0\n", encoding="utf-8")
        with patch(
            "agilerl.utils.env_utils.importlib_util.spec_from_file_location",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="Could not create spec"):
                env_utils.get_rubric("rubric", str(rubric_file))

    def test_loads_rubric_instance(self, tmp_path):
        rubric_file = tmp_path / "rubric.py"
        rubric_file.write_text(
            "from agilerl.llm_envs.rubrics import reward_fn_to_rubric\n"
            "def my_reward(obs): return 1.0\n"
            "RUBRIC = reward_fn_to_rubric(my_reward)\n",
            encoding="utf-8",
        )
        rubric = env_utils.get_rubric("RUBRIC", str(rubric_file))
        from openenv.core.rubrics.base import Rubric

        assert isinstance(rubric, Rubric)

    def test_loads_rubric_subclass(self, tmp_path):
        rubric_file = tmp_path / "rubric_cls.py"
        rubric_file.write_text(
            "from openenv.core.rubrics.base import Rubric\n"
            "class MyRubric(Rubric):\n"
            "    def forward(self, action, observation):\n"
            "        return 1.0\n",
            encoding="utf-8",
        )
        rubric = env_utils.get_rubric("MyRubric", str(rubric_file))
        from openenv.core.rubrics.base import Rubric

        assert isinstance(rubric, Rubric)

    def test_loads_legacy_reward_callable(self, tmp_path):
        """Arena reward modules export a callable; wrap with RewardFnRubric."""
        reward_file = tmp_path / "reward.py"
        reward_file.write_text(
            "def combined_rewards(completion, answer, question):\n"
            "    del answer, question\n"
            "    return 1.0 if completion else 0.0\n",
            encoding="utf-8",
        )
        rubric = env_utils.get_rubric("combined_rewards", str(reward_file))
        from openenv.core.rubrics.base import Rubric

        from agilerl.llm_envs.rubrics import RewardFnRubric

        assert isinstance(rubric, Rubric)
        assert isinstance(rubric, RewardFnRubric)

    def test_loads_weighted_sum_rubric_instance(self, tmp_path):
        """OpenEnv composition exports (e.g. WeightedSum) must not be re-wrapped."""
        reward_file = tmp_path / "reward.py"
        reward_file.write_text(
            "from openenv.core.rubrics.base import Rubric\n"
            "from openenv.core.rubrics import WeightedSum\n"
            "\n"
            "class TestsPassRubric(Rubric):\n"
            "    def forward(self, action, observation) -> float:\n"
            "        return 1.0\n"
            "\n"
            "class StyleRubric(Rubric):\n"
            "    def forward(self, action, observation) -> float:\n"
            "        return 0.6\n"
            "\n"
            "reward = WeightedSum(\n"
            "    [TestsPassRubric(), StyleRubric()],\n"
            "    weights=[0.7, 0.3],\n"
            ")\n",
            encoding="utf-8",
        )
        rubric = env_utils.get_rubric("reward", str(reward_file))
        from openenv.core.rubrics import WeightedSum

        from agilerl.llm_envs.rubrics import RewardFnRubric

        assert isinstance(rubric, WeightedSum)
        assert not isinstance(rubric, RewardFnRubric)
        assert list(rubric.named_children())  # leaves registered for metrics

    def test_non_rubric_export_raises(self, tmp_path):
        rubric_file = tmp_path / "not_rubric.py"
        rubric_file.write_text("NOT_A_RUBRIC = 42\n", encoding="utf-8")
        with pytest.raises(TypeError, match="must be a Rubric"):
            env_utils.get_rubric("NOT_A_RUBRIC", str(rubric_file))


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
