# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for agilerl.utils.env_utils."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from agilerl.arena.models.env import JudgeSpec
from agilerl.llm_envs import judge as judge_module
from agilerl.utils import env_utils

LOCAL_JUDGE = JudgeSpec(model="m", base_url="http://judge:8000")


class TestLoadPettingzooParallelEnv:
    def test_retired_cooperative_pong_v5_returns_v6_factory(self):
        loaded = env_utils.load_pettingzoo_parallel_env(
            "pettingzoo.butterfly.cooperative_pong_v5",
        )

        v6 = import_module("pettingzoo.butterfly.cooperative_pong_v6")
        assert loaded is v6.parallel_env

    def test_installed_v5_module_is_not_replaced(self, monkeypatch):
        # Arrange
        module_name = "pettingzoo.butterfly.cooperative_pong_v5"
        installed = ModuleType(module_name)
        installed.parallel_env = object()
        monkeypatch.setitem(sys.modules, module_name, installed)

        # Act
        loaded = env_utils.load_pettingzoo_parallel_env(module_name)

        # Assert
        assert loaded is installed.parallel_env

    def test_missing_module_without_alias_raises(self):
        with pytest.raises(
            ModuleNotFoundError,
            match=r"pettingzoo\.butterfly\.not_a_real_env_v0",
        ):
            env_utils.load_pettingzoo_parallel_env(
                "pettingzoo.butterfly.not_a_real_env_v0",
            )

    def test_missing_dependency_is_not_replaced(self, monkeypatch):
        # Arrange
        def broken_import(module_name: str):
            message = "No module named 'pygame' while importing " + module_name
            raise ModuleNotFoundError(message, name="pygame")

        monkeypatch.setattr(env_utils, "import_module", broken_import)

        # Act / Assert
        with pytest.raises(ModuleNotFoundError, match="pygame"):
            env_utils.load_pettingzoo_parallel_env(
                "pettingzoo.butterfly.cooperative_pong_v5",
            )

    def test_module_without_parallel_env_raises(self, monkeypatch):
        module = ModuleType("pettingzoo.butterfly.missing_parallel_v1")
        monkeypatch.setitem(sys.modules, module.__name__, module)

        with pytest.raises(AttributeError, match="has no 'parallel_env'"):
            env_utils.load_pettingzoo_parallel_env(module.__name__)


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

    def test_module_level_instance_is_copied_per_call(self, tmp_path):
        """Each call gets its own copy, including nested WeightedSum children."""
        rubric_file = tmp_path / "singleton.py"
        rubric_file.write_text(
            "from openenv.core.rubrics.base import Rubric\n"
            "from openenv.core.rubrics.containers import WeightedSum\n"
            "class Leaf(Rubric):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.n = 0\n"
            "    def forward(self, action, observation):\n"
            "        self.n += 1\n"
            "        return float(self.n)\n"
            "RUBRIC = WeightedSum([Leaf()], weights=[1.0])\n",
            encoding="utf-8",
        )
        factory = env_utils.get_rubric_factory("RUBRIC", str(rubric_file))
        first, second = factory(), factory()
        assert first is not second
        assert next(iter(first.children())) is not next(iter(second.children()))
        first.last_score = 1.0
        assert second.last_score is None

    def test_a_judge_is_handed_to_the_rubric_class(self, tmp_path, monkeypatch):
        monkeypatch.setattr(judge_module, "JUDGE_CLIENTS", {})
        rubric_file = tmp_path / "judged.py"
        rubric_file.write_text(
            "from openenv.core.rubrics.base import Rubric\n"
            "class Judged(Rubric):\n"
            "    def __init__(self, judge):\n"
            "        super().__init__()\n"
            "        self.judge = judge\n"
            "    def forward(self, action, observation):\n"
            "        return 1.0\n",
            encoding="utf-8",
        )

        factory = env_utils.get_rubric_factory("Judged", str(rubric_file), LOCAL_JUDGE)

        first, second = factory(), factory()
        assert first is not second
        assert first.judge is judge_module.JUDGE_CLIENTS[LOCAL_JUDGE]
        assert second.judge is first.judge

    def test_an_uncopyable_rubric_instance_raises_at_load(self, tmp_path):
        rubric_file = tmp_path / "held.py"
        rubric_file.write_text(
            "import threading\n"
            "from openenv.core.rubrics.base import Rubric\n"
            "class Held(Rubric):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.lock = threading.Lock()\n"
            "    def forward(self, action, observation):\n"
            "        return 1.0\n"
            "RUBRIC = Held()\n",
            encoding="utf-8",
        )

        with pytest.raises(TypeError, match="cannot be copied"):
            env_utils.get_rubric_factory("RUBRIC", str(rubric_file))

    def test_a_judge_with_a_reward_callable_raises(self, tmp_path):
        rubric_file = tmp_path / "reward.py"
        rubric_file.write_text(
            "def reward(completion, answer, question):\n    return 1.0\n",
            encoding="utf-8",
        )

        with pytest.raises(TypeError, match="__init__ takes judge"):
            env_utils.get_rubric_factory("reward", str(rubric_file), LOCAL_JUDGE)


class TestGetRubricFactoryLoading:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "missing.py"
        with pytest.raises(ValueError, match="not found"):
            env_utils.get_rubric_factory("rubric", str(missing))()

    def test_import_error_wrapped(self, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text(
            "raise RuntimeError('intentional import failure')\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Error importing rubric"):
            env_utils.get_rubric_factory("rubric", str(bad_file))()

    def test_spec_none_raises(self, tmp_path):
        rubric_file = tmp_path / "rubric.py"
        rubric_file.write_text("def rubric(): return 1.0\n", encoding="utf-8")
        with patch(
            "agilerl.utils.env_utils.importlib_util.spec_from_file_location",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="Could not load module"):
                env_utils.get_rubric_factory("rubric", str(rubric_file))()

    def test_loads_rubric_instance(self, tmp_path):
        rubric_file = tmp_path / "rubric.py"
        rubric_file.write_text(
            "from agilerl.llm_envs.rubrics import reward_fn_to_rubric\n"
            "def my_reward(obs): return 1.0\n"
            "RUBRIC = reward_fn_to_rubric(my_reward)\n",
            encoding="utf-8",
        )
        rubric = env_utils.get_rubric_factory("RUBRIC", str(rubric_file))()
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
        rubric = env_utils.get_rubric_factory("MyRubric", str(rubric_file))()
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
        rubric = env_utils.get_rubric_factory("combined_rewards", str(reward_file))()
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
        rubric = env_utils.get_rubric_factory("reward", str(reward_file))()
        from openenv.core.rubrics import WeightedSum

        from agilerl.llm_envs.rubrics import RewardFnRubric

        assert isinstance(rubric, WeightedSum)
        assert not isinstance(rubric, RewardFnRubric)
        assert list(rubric.named_children())  # leaves registered for metrics

    def test_non_rubric_export_raises(self, tmp_path):
        rubric_file = tmp_path / "not_rubric.py"
        rubric_file.write_text("NOT_A_RUBRIC = 42\n", encoding="utf-8")
        with pytest.raises(TypeError, match="must be a Rubric"):
            env_utils.get_rubric_factory("NOT_A_RUBRIC", str(rubric_file))()

    def test_spec_without_loader_raises(self, tmp_path):
        reward_file = tmp_path / "reward.py"
        reward_file.write_text("def reward(): return 1.0\n", encoding="utf-8")
        with patch(
            "agilerl.utils.env_utils.importlib_util.spec_from_file_location",
            return_value=MagicMock(loader=None),
        ):
            with pytest.raises(ValueError, match="Could not load module"):
                env_utils.get_rubric_factory("reward", str(reward_file))


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


class TestConstructEntrypointEnv:
    def test_constructs_entrypoint_with_kwargs(self, tmp_path):
        module = tmp_path / "ctor.py"
        module.write_text(
            "class Env:\n    def __init__(self, n=0):\n        self.n = n\n",
            encoding="utf-8",
        )

        env = env_utils.construct_entrypoint_env(
            f"{module.name}:Env", {"n": 4}, path=str(tmp_path)
        )

        assert env.n == 4

    def test_factory_receives_entrypoint_as_first_arg(self, tmp_path):
        module = tmp_path / "lib.py"
        module.write_text(
            "def make(env_id, n=0):\n"
            "    return type('Env', (), {'env_id': env_id, 'n': n})()\n",
            encoding="utf-8",
        )

        env = env_utils.construct_entrypoint_env(
            "game:GuessTheNumber-v0-easy",
            {"n": 2},
            factory=f"{module.name}:make",
            path=str(tmp_path),
        )

        assert env.env_id == "game:GuessTheNumber-v0-easy"
        assert env.n == 2

    def test_non_callable_factory_raises(self, tmp_path):
        module = tmp_path / "lib.py"
        module.write_text("make = 3\n", encoding="utf-8")

        with pytest.raises(TypeError, match="Factory"):
            env_utils.construct_entrypoint_env(
                "game:GuessTheNumber-v0-easy",
                factory=f"{module.name}:make",
                path=str(tmp_path),
            )
