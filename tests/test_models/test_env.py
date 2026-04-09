from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import numpy as np

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.models.env import (
    BanditEnvSpec,
    GymEnvSpec,
    LLMEnvSpec,
    LLMEnvType,
    PzEnvSpec,
)


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
        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            reward_file_path="reward.py",
            reward_fn_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )
        assert spec.train_test_split == 0.9
        assert spec.reward_file_path == "reward.py"
        assert spec.dataset_path == "dataset.parquet"
        assert spec.columns is None
        assert spec.max_reward is None

    def test_is_standalone_model(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            reward_file_path="reward.py",
            reward_fn_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )
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
            reward_fn_name="my_reward",
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

    @patch("agilerl.models.env.make_conversation_template")
    @patch("agilerl.models.env.get_reward_fn")
    @patch.object(LLMEnvSpec, "load_dataset")
    def test_make_env_reasoning(self, mock_load, mock_reward_fn, mock_conv_tmpl):
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_load.return_value = (mock_train_ds, mock_test_ds)
        mock_reward_fn.return_value = lambda *a, **kw: 1.0
        mock_conv_tmpl.return_value = [{"role": "user", "content": "{q}"}]
        mock_tokenizer = MagicMock()

        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            reward_file_path="reward.py",
            reward_fn_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )

        with patch("agilerl.utils.llm_utils.ReasoningGym") as MockGym:
            MockGym.return_value = "reasoning_gym"
            result = spec.make_env(tokenizer=mock_tokenizer)

        assert result == "reasoning_gym"
        mock_reward_fn.assert_called_once_with(
            reward_fn_name="reward_fn", file_path="reward.py"
        )
        mock_conv_tmpl.assert_called_once()
        MockGym.assert_called_once()

    @patch.object(LLMEnvSpec, "load_dataset")
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
        MockGym.assert_called_once()

    def test_serialization_roundtrip(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            columns={"q": "question"},
            max_reward=10.0,
            dataset_path="data.parquet",
            reward_file_path="reward.py",
            reward_fn_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )
        data = spec.model_dump()
        restored = LLMEnvSpec.model_validate(data)
        assert restored.env_type == LLMEnvType.REASONING
        assert restored.columns == {"q": "question"}
        assert restored.max_reward == 10.0
        assert restored.dataset_path == "data.parquet"


class TestBanditEnvSpec:
    """Tests for BanditEnvSpec validation and make_env."""

    def _make_dataset(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        features = pd.DataFrame(rng.standard_normal((50, 4)).astype(np.float32))
        targets = pd.DataFrame(rng.integers(0, 3, size=(50, 1)))
        return features, targets

    def test_dataset_mode_from_dataframes(self):
        features, targets = self._make_dataset()
        spec = BanditEnvSpec(features=features, targets=targets)
        env = spec.make_env()

        assert hasattr(env, "single_observation_space")
        assert hasattr(env, "single_action_space")
        assert hasattr(env, "num_envs")
        state = env.reset()
        assert state.shape[0] == env.arms

    def test_dataset_mode_from_csv(self, tmp_path):
        features, targets = self._make_dataset()
        feat_path = tmp_path / "features.csv"
        tgt_path = tmp_path / "targets.csv"
        features.to_csv(feat_path, index=False)
        targets.to_csv(tgt_path, index=False)

        spec = BanditEnvSpec(
            name="TestCSV",
            features=str(feat_path),
            targets=str(tgt_path),
        )
        env = spec.make_env()
        state = env.reset()
        assert state.dtype == np.float32

    def test_entrypoint_mode(self, tmp_path):
        _write_module(
            tmp_path,
            "custom_bandit",
            """\
import numpy as np
from gymnasium import spaces

class FakeBandit:
    def __init__(self, n_arms=2, context_dim=4):
        self.arms = n_arms
        self.num_envs = 1
        self.single_observation_space = spaces.Box(-1, 1, shape=(context_dim,))
        self.single_action_space = spaces.Discrete(n_arms)

    def reset(self):
        return np.zeros(self.single_observation_space.shape, dtype=np.float32)

    def step(self, k):
        return self.reset(), float(k == 0)
""",
        )

        spec = BanditEnvSpec(
            name="FakeBandit",
            entrypoint="custom_bandit:FakeBandit",
            path=str(tmp_path),
            config={"n_arms": 3, "context_dim": 8},
        )
        env = spec.make_env()
        assert env.arms == 3
        assert env.single_action_space.n == 3
        assert env.single_observation_space.shape == (8,)

    def test_validation_requires_dataset_or_entrypoint(self):
        with pytest.raises(ValueError, match="requires either"):
            BanditEnvSpec(name="empty")

    def test_validation_rejects_both(self):
        features, targets = self._make_dataset()
        with pytest.raises(ValueError, match="not both"):
            BanditEnvSpec(
                features=features,
                targets=targets,
                entrypoint="some_module:SomeClass",
            )

    def test_validation_requires_both_features_and_targets(self):
        features, _ = self._make_dataset()
        with pytest.raises(ValueError, match="together"):
            BanditEnvSpec(features=features)

    def test_serialization_roundtrip(self):
        spec = BanditEnvSpec(
            name="IRIS",
            features="data/features.csv",
            targets="data/targets.csv",
        )
        data = spec.model_dump(mode="json")
        restored = BanditEnvSpec.model_validate(data)
        assert restored.name == "IRIS"
        assert restored.features == "data/features.csv"
        assert restored.targets == "data/targets.csv"

    def test_entrypoint_serialization_roundtrip(self):
        spec = BanditEnvSpec(
            name="Custom",
            entrypoint="my_module:MyBandit",
            config={"n_arms": 5},
        )
        data = spec.model_dump(mode="json")
        restored = BanditEnvSpec.model_validate(data)
        assert restored.entrypoint == "my_module:MyBandit"
        assert restored.config == {"n_arms": 5}
        assert restored.features is None
