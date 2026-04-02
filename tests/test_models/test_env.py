from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.models.env import GymEnvSpec, LLMEnvSpec, LLMEnvType, PzEnvSpec


def _write_module(tmp_path, name: str, contents: str) -> None:
    module_path = tmp_path / f"{name}.py"
    module_path.write_text(contents)


class TestGymEnvSpec:
    def test_custom_env_with_config_and_wrappers(self, tmp_path):
        _write_module(
            tmp_path,
            "custom_gym_env",
            """
class CustomGymEnv:
    def __init__(self, value=0):
        self.value = value
""",
        )

        call_order = []

        def wrapper_one(env):
            call_order.append("wrapper_one")
            env.wrapper_trace = ["wrapper_one"]
            return env

        def wrapper_two(env, tag):
            call_order.append("wrapper_two")
            env.wrapper_trace.append(tag)
            return env

        make_env = GymEnvSpec.constuct_custom_env_fn(
            entrypoint="custom_gym_env:CustomGymEnv",
            path=str(tmp_path),
            config={"value": 12},
            wrappers=[wrapper_one, (wrapper_two, {"tag": "wrapper_two"})],
        )
        env = make_env()

        assert env.value == 12
        assert env.wrapper_trace == ["wrapper_one", "wrapper_two"]
        assert call_order == ["wrapper_one", "wrapper_two"]

    def test_custom_env_without_env_path(self, tmp_path, monkeypatch):
        _write_module(
            tmp_path,
            "cwd_env",
            """
class CwdEnv:
    def __init__(self, value=3):
        self.value = value
""",
        )
        monkeypatch.chdir(tmp_path)

        make_env = GymEnvSpec.constuct_custom_env_fn(
            entrypoint="cwd_env:CwdEnv",
            path=None,
            config={"value": 99},
        )
        env = make_env()
        assert env.value == 99

    def test_custom_env_with_invalid_entrypoint(self):
        make_env = GymEnvSpec.constuct_custom_env_fn(entrypoint="invalid-entrypoint")
        with pytest.raises(ValueError, match="Invalid entrypoint format"):
            make_env()

    def test_custom_env_with_missing_module(self):
        make_env = GymEnvSpec.constuct_custom_env_fn(entrypoint="does_not_exist:Env")
        with pytest.raises(ModuleNotFoundError, match="Could not resolve module"):
            make_env()

    def test_custom_env_with_missing_target(self, tmp_path):
        _write_module(
            tmp_path,
            "missing_target",
            """
class OtherEnv:
    pass
""",
        )
        make_env = GymEnvSpec.constuct_custom_env_fn(
            entrypoint="missing_target:WantedEnv",
            path=str(tmp_path),
        )
        with pytest.raises(AttributeError, match="does not define 'WantedEnv'"):
            make_env()

    def test_make_env_with_registered_gym_env(self):
        spec = GymEnvSpec(name="CartPole-v1", num_envs=4)

        with patch("agilerl.utils.utils.make_vect_envs") as make_vect_mock:
            make_vect_mock.return_value = "gym_vec_env"
            result = spec.make_env()

        assert result == "gym_vec_env"
        assert make_vect_mock.call_args.kwargs["env_name"] == "CartPole-v1"
        assert make_vect_mock.call_args.kwargs["num_envs"] == 4
        assert make_vect_mock.call_args.kwargs["make_env"] is None
        assert make_vect_mock.call_args.kwargs["should_async_vector"] is True

    def test_make_env_with_custom_factory(self, tmp_path):
        _write_module(
            tmp_path,
            "gym_for_make_env",
            """
class MyGymEnv:
    def __init__(self, value=0):
        self.value = value
""",
        )
        spec = GymEnvSpec(
            name="unused",
            num_envs=2,
            entrypoint="gym_for_make_env:MyGymEnv",
            path=str(tmp_path),
            config={"value": 7},
            sync=True,
        )

        with patch("agilerl.utils.utils.make_vect_envs") as make_vect_mock:
            make_vect_mock.return_value = "custom_vec_env"
            result = spec.make_env()

        assert result == "custom_vec_env"
        assert make_vect_mock.call_args.kwargs["should_async_vector"] is False
        make_env = make_vect_mock.call_args.kwargs["make_env"]
        env = make_env()
        assert env.value == 7

    def test_make_env_with_gym_env(self):
        spec = GymEnvSpec(name="CartPole-v1", num_envs=1, sync=True)
        env = spec.make_env()
        try:
            assert env.num_envs == 1
            obs, info = env.reset()
            assert obs is not None
            assert isinstance(info, dict)
        finally:
            env.close()


class TestPzEnvSpec:
    def test_custom_env_with_construct_and_wrappers(self, tmp_path):
        _write_module(
            tmp_path,
            "custom_pz_env",
            """
class DummyPzEnv:
    def __init__(self, size=0):
        self.size = size
        self.tag = None

def build_env(size=0):
    return DummyPzEnv(size=size)
""",
        )

        def wrapper(env, tag):
            env.tag = tag
            return env

        spec = PzEnvSpec(
            name="unused",
            num_envs=3,
            entrypoint="custom_pz_env:build_env",
            path=str(tmp_path),
            config={"size": 5},
            wrappers=[(wrapper, {"tag": "wrapped"})],
        )

        with patch("agilerl.utils.utils.make_multi_agent_vect_envs") as make_multi_mock:
            make_multi_mock.return_value = "vector_env"
            result = spec.make_env()

        assert result == "vector_env"
        assert make_multi_mock.call_args.kwargs["num_envs"] == 3

        constructor = make_multi_mock.call_args.kwargs["env"]
        env = constructor()
        assert env.size == 5
        assert env.tag == "wrapped"

    def test_env_without_entrypoint_with_parallel_env_constructor(
        self, tmp_path, monkeypatch
    ):
        _write_module(
            tmp_path,
            "pz_pkg",
            """
class DummyPzEnv:
    def __init__(self, value=0):
        self.value = value

def parallel_env(value=0):
    return DummyPzEnv(value=value)
""",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        spec = PzEnvSpec(
            name="pz_pkg",
            num_envs=2,
            config={"value": 42},
        )

        with patch("agilerl.utils.utils.make_multi_agent_vect_envs") as make_multi_mock:
            make_multi_mock.return_value = "pz_vec_env"
            result = spec.make_env()

        assert result == "pz_vec_env"
        constructor = make_multi_mock.call_args.kwargs["env"]
        env = constructor()
        assert env.value == 42

    def test_custom_env_wrapper_with_string_entrypoint(self, tmp_path):
        _write_module(
            tmp_path,
            "custom_pz_env_with_wrapper",
            """
class DummyPzEnv:
    def __init__(self, value=0):
        self.value = value
        self.wrapped = False

def build_env(value=0):
    return DummyPzEnv(value=value)

def mark_wrapped(env):
    env.wrapped = True
    return env
""",
        )

        spec = PzEnvSpec(
            name="unused",
            num_envs=1,
            entrypoint="custom_pz_env_with_wrapper:build_env",
            path=str(tmp_path),
            config={"value": 3},
            wrappers=["custom_pz_env_with_wrapper:mark_wrapped"],
        )

        with patch("agilerl.utils.utils.make_multi_agent_vect_envs") as make_multi_mock:
            make_multi_mock.return_value = "pz_vec_env"
            result = spec.make_env()

        assert result == "pz_vec_env"
        constructor = make_multi_mock.call_args.kwargs["env"]
        env = constructor()
        assert env.value == 3
        assert env.wrapped is True

    def test_make_env_with_registered_pettingzoo_environment(self):
        spec = PzEnvSpec(
            name="pettingzoo.mpe.simple_speaker_listener_v4",
            num_envs=1,
            env_config={"max_cycles": 5, "continuous_actions": False},
        )

        env = spec.make_env()
        try:
            observations, infos = env.reset(seed=0)
            assert env.num_envs == 1
            assert len(env.agents) > 0
            assert isinstance(observations, dict)
            assert isinstance(infos, dict)
            assert all(agent in observations for agent in env.agents)
        finally:
            env.close()


class TestLLMEnvSpec:
    def test_reasoning_requires_reward_file_path(self):
        with pytest.raises(ValueError, match="reward_file_path is required"):
            LLMEnvSpec(env_type=LLMEnvType.REASONING, reward_file_path=None)

    def test_preference_does_not_require_reward_file_path(self):
        spec = LLMEnvSpec(env_type=LLMEnvType.PREFERENCE, reward_file_path=None)
        assert spec.reward_file_path is None

    def test_default_fields(self):
        spec = LLMEnvSpec(env_type=LLMEnvType.REASONING)
        assert spec.train_test_split == 0.9
        assert spec.reward_file_path == "reward.py"
        assert spec.dataset_path == "dataset.parquet"
        assert spec.columns is None
        assert spec.prompt_template is None
        assert spec.max_reward is None

    def test_is_standalone_model(self):
        spec = LLMEnvSpec(env_type=LLMEnvType.REASONING)
        assert not hasattr(spec, "name")
        assert not hasattr(spec, "num_envs")

    def test_custom_fields(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            columns={"question": "input", "answer": "output"},
            prompt_template={"role": "user", "content": "{question}"},
            max_reward=5.0,
            train_test_split=0.8,
            reward_file_path="my_reward.py",
            dataset_path="data/train.parquet",
        )
        assert spec.columns == {"question": "input", "answer": "output"}
        assert spec.prompt_template == {"role": "user", "content": "{question}"}
        assert spec.max_reward == 5.0
        assert spec.train_test_split == 0.8
        assert spec.reward_file_path == "my_reward.py"
        assert spec.dataset_path == "data/train.parquet"

    def test_train_test_split_bounds(self):
        with pytest.raises(Exception):
            LLMEnvSpec(env_type=LLMEnvType.REASONING, train_test_split=1.5)
        with pytest.raises(Exception):
            LLMEnvSpec(env_type=LLMEnvType.REASONING, train_test_split=-0.1)

    @patch("agilerl.models.env._resolve_entrypoint_target")
    @patch("agilerl.models.env.LLMEnvSpec._load_dataset")
    def test_make_env_reasoning(self, mock_load, mock_resolve):
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_load.return_value = (mock_train_ds, mock_test_ds)
        mock_resolve.return_value = lambda *a, **kw: 1.0
        mock_tokenizer = MagicMock()

        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            reward_file_path="reward.py",
            prompt_template={"role": "user", "content": "{q}"},
        )

        with patch("agilerl.utils.llm_utils.ReasoningGym") as MockGym:
            MockGym.return_value = "reasoning_gym"
            result = spec.make_env(tokenizer=mock_tokenizer)

        assert result == "reasoning_gym"
        MockGym.assert_called_once_with(
            train_dataset=mock_train_ds,
            test_dataset=mock_test_ds,
            tokenizer=mock_tokenizer,
            reward_fn=mock_resolve.return_value,
            conversation_template={"role": "user", "content": "{q}"},
            accelerator=None,
        )

    @patch("agilerl.models.env.LLMEnvSpec._load_dataset")
    def test_make_env_preference(self, mock_load):
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_load.return_value = (mock_train_ds, mock_test_ds)
        mock_tokenizer = MagicMock()

        spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE,
            reward_file_path=None,
        )

        with patch("agilerl.utils.llm_utils.PreferenceGym") as MockGym:
            MockGym.return_value = "preference_gym"
            result = spec.make_env(tokenizer=mock_tokenizer)

        assert result == "preference_gym"
        MockGym.assert_called_once_with(
            train_dataset=mock_train_ds,
            test_dataset=mock_test_ds,
            tokenizer=mock_tokenizer,
            accelerator=None,
        )

    def test_serialization_roundtrip(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            columns={"q": "question"},
            max_reward=10.0,
            dataset_path="data.parquet",
        )
        data = spec.model_dump()
        restored = LLMEnvSpec.model_validate(data)
        assert restored.env_type == LLMEnvType.REASONING
        assert restored.columns == {"q": "question"}
        assert restored.max_reward == 10.0
        assert restored.dataset_path == "data.parquet"
