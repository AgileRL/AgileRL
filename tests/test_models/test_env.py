# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from agilerl.models.env import (
    BanditEnvSpec,
    GymEnvSpec,
    LLMEnvSpec,
    LLMEnvType,
    OfflineEnvSpec,
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

        make_env = GymEnvSpec.construct_custom_env_fn(
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

        make_env = GymEnvSpec.construct_custom_env_fn(
            entrypoint="cwd_env:CwdEnv",
            path=None,
            config={"value": 99},
        )
        env = make_env()
        assert env.value == 99

    def test_custom_env_with_invalid_entrypoint(self):
        make_env = GymEnvSpec.construct_custom_env_fn(entrypoint="invalid-entrypoint")
        with pytest.raises(ValueError, match="Invalid entrypoint format"):
            make_env()

    def test_custom_env_with_missing_module(self):
        make_env = GymEnvSpec.construct_custom_env_fn(entrypoint="does_not_exist:Env")
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
        make_env = GymEnvSpec.construct_custom_env_fn(
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
            name="mpe2.simple_speaker_listener_v4",
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
            LLMEnvSpec(
                env_type=LLMEnvType.REASONING, dataset="ds", reward_file_path=None
            )

    def test_preference_does_not_require_reward_file_path(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE, dataset="ds", reward_file_path=None
        )
        assert spec.reward_file_path is None

    def test_default_fields(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            dataset="dataset.parquet",
            reward_file_path="reward.py",
            reward_fn_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )
        assert spec.train_test_split == 0.9
        assert spec.reward_file_path == "reward.py"
        assert spec.dataset == "dataset.parquet"
        assert spec.columns is None
        assert spec.max_reward is None

    def test_is_standalone_model(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            dataset="dataset.parquet",
            reward_file_path="reward.py",
            reward_fn_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )
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
            dataset="data/train.parquet",
        )
        assert spec.columns == {"question": "input", "answer": "output"}
        assert spec.prompt_template == {"role": "user", "content": "{question}"}
        assert spec.max_reward == 5.0
        assert spec.train_test_split == 0.8
        assert spec.reward_file_path == "my_reward.py"
        assert spec.dataset == "data/train.parquet"

    def test_train_test_split_bounds(self):
        with pytest.raises(ValidationError):
            LLMEnvSpec(
                env_type=LLMEnvType.REASONING, dataset="ds", train_test_split=1.5
            )
        with pytest.raises(ValidationError):
            LLMEnvSpec(
                env_type=LLMEnvType.REASONING, dataset="ds", train_test_split=-0.1
            )

    @patch("agilerl.models.env.make_conversation_template")
    @patch("agilerl.models.env.get_reward_fn")
    @patch.object(LLMEnvSpec, "_load_dataset")
    def test_make_env_reasoning(self, mock_load, mock_reward_fn, mock_conv_tmpl):
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_load.return_value = (mock_train_ds, mock_test_ds)
        mock_reward_fn.return_value = lambda *a, **kw: 1.0
        mock_conv_tmpl.return_value = [{"role": "user", "content": "{q}"}]
        mock_tokenizer = MagicMock()

        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            dataset="dataset.parquet",
            reward_file_path="reward.py",
            reward_fn_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )

        with patch("agilerl.wrappers.llm_envs.ReasoningGym") as MockGym:
            MockGym.return_value = "reasoning_gym"
            result = spec.make_env(tokenizer=mock_tokenizer)

        assert result == "reasoning_gym"
        mock_reward_fn.assert_called_once_with(
            reward_fn_name="reward_fn", file_path="reward.py"
        )
        mock_conv_tmpl.assert_called_once()
        MockGym.assert_called_once()

    @patch.object(LLMEnvSpec, "_load_dataset")
    def test_make_env_preference(self, mock_load):
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_load.return_value = (mock_train_ds, mock_test_ds)
        mock_tokenizer = MagicMock()

        spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE,
            dataset="dataset.parquet",
            reward_file_path=None,
        )

        with patch("agilerl.wrappers.llm_envs.PreferenceGym") as MockGym:
            MockGym.return_value = "preference_gym"
            result = spec.make_env(tokenizer=mock_tokenizer)

        assert result == "preference_gym"
        MockGym.assert_called_once()

    def test_serialization_roundtrip(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            columns={"q": "question"},
            max_reward=10.0,
            dataset="data.parquet",
            reward_file_path="reward.py",
            reward_fn_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )
        data = spec.model_dump()
        restored = LLMEnvSpec.model_validate(data)
        assert restored.env_type == LLMEnvType.REASONING
        assert restored.columns == {"q": "question"}
        assert restored.max_reward == 10.0
        assert restored.dataset == "data.parquet"


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


# ---------------------------------------------------------------------------
# LLM SFT
# ---------------------------------------------------------------------------
class TestLLMEnvSpecSFT:
    def test_sft_spec_valid_construction(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.SFT,
            dataset="my_dataset.parquet",
            response_column="answer",
        )
        assert spec.env_type == LLMEnvType.SFT
        assert spec.dataset == "my_dataset.parquet"
        assert spec.response_column == "answer"
        assert spec.reward_file_path is None

    def test_sft_default_response_column(self):
        spec = LLMEnvSpec(env_type=LLMEnvType.SFT, dataset="ds")
        assert spec.response_column == "response"

    @patch.object(LLMEnvSpec, "_load_dataset")
    def test_sft_make_env_creates_sft_gym(self, mock_load):
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_load.return_value = (mock_train_ds, mock_test_ds)
        mock_tokenizer = MagicMock()

        spec = LLMEnvSpec(
            env_type=LLMEnvType.SFT,
            dataset="sft_data.parquet",
            response_column="completion",
        )

        with patch("agilerl.wrappers.llm_envs.SFTGym") as MockGym:
            MockGym.return_value = "sft_gym"
            result = spec.make_env(tokenizer=mock_tokenizer)

        assert result == "sft_gym"
        MockGym.assert_called_once()
        call_kwargs = MockGym.call_args.kwargs
        assert call_kwargs["train_dataset"] is mock_train_ds
        assert call_kwargs["test_dataset"] is mock_test_ds
        assert call_kwargs["response_column"] == "completion"

    def test_sft_rejects_reward_file_path(self):
        with pytest.raises(ValueError, match="not supported for SFT"):
            LLMEnvSpec(
                env_type=LLMEnvType.SFT,
                dataset="ds",
                reward_file_path="reward.py",
            )

    def test_preference_rejects_reward_file_path(self):
        with pytest.raises(ValueError, match="not supported for preference"):
            LLMEnvSpec(
                env_type=LLMEnvType.PREFERENCE,
                dataset="ds",
                reward_file_path="reward.py",
            )

    def test_dataset_alias_name(self):
        spec = LLMEnvSpec.model_validate(
            {
                "name": "my_dataset",
                "env_type": "reasoning",
                "reward_file_path": "r.py",
                "reward_fn_name": "fn",
                "prompt_template": {"role": "user", "content": "{q}"},
            }
        )
        assert spec.dataset == "my_dataset"


# ---------------------------------------------------------------------------
# GymEnvSpec - make_single_env + extra_wrappers
# ---------------------------------------------------------------------------
class TestGymEnvSpecSingleEnv:
    def test_make_single_env_registered(self):
        spec = GymEnvSpec(name="CartPole-v1")
        env = spec.make_single_env()
        try:
            assert hasattr(env, "reset")
            assert hasattr(env, "step")
        finally:
            env.close()

    def test_make_single_env_custom(self, tmp_path):
        _write_module(
            tmp_path,
            "single_env_mod",
            """\
class SingleEnv:
    def __init__(self, val=0):
        self.val = val
""",
        )
        spec = GymEnvSpec(
            name="unused",
            entrypoint="single_env_mod:SingleEnv",
            path=str(tmp_path),
            config={"val": 42},
        )
        env = spec.make_single_env()
        assert env.val == 42

    def test_make_env_extra_wrappers(self):
        spec = GymEnvSpec(name="CartPole-v1", num_envs=2)

        with patch("agilerl.utils.utils.make_vect_envs") as mock_make:
            mock_make.return_value = "vec_env"
            spec.make_env(extra_wrappers=["SomeWrapper"])

        assert mock_make.call_args.kwargs["extra_wrappers"] == ["SomeWrapper"]


# ---------------------------------------------------------------------------
# PzEnvSpec - make_single_env + error paths + extra_wrappers
# ---------------------------------------------------------------------------
class TestPzEnvSpecSingleEnv:
    def test_make_single_env_custom(self, tmp_path):
        _write_module(
            tmp_path,
            "pz_single_mod",
            """\
class PzSingleEnv:
    def __init__(self, size=0):
        self.size = size
""",
        )
        spec = PzEnvSpec(
            name="unused",
            entrypoint="pz_single_mod:PzSingleEnv",
            path=str(tmp_path),
            config={"size": 7},
        )
        env = spec.make_single_env()
        assert env.size == 7

    def test_make_single_env_module_with_parallel_env(self, tmp_path, monkeypatch):
        _write_module(
            tmp_path,
            "pz_parallel_single",
            """\
class DummyPzEnv:
    def __init__(self, value=0):
        self.value = value

def parallel_env(value=0):
    return DummyPzEnv(value=value)
""",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        spec = PzEnvSpec(name="pz_parallel_single", config={"value": 99})
        env = spec.make_single_env()
        assert env.value == 99

    def test_make_single_env_module_missing_parallel_env(self, tmp_path, monkeypatch):
        _write_module(
            tmp_path,
            "pz_no_parallel",
            """\
class SomeEnv:
    pass
""",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        spec = PzEnvSpec(name="pz_no_parallel", num_envs=1)
        with pytest.raises(AttributeError, match="no 'parallel_env'"):
            spec.make_single_env()

    def test_make_env_extra_wrappers(self, tmp_path):
        _write_module(
            tmp_path,
            "pz_ew",
            """\
class PzEw:
    pass
""",
        )
        spec = PzEnvSpec(
            name="unused",
            num_envs=1,
            entrypoint="pz_ew:PzEw",
            path=str(tmp_path),
        )

        with patch("agilerl.utils.utils.make_multi_agent_vect_envs") as mock_make:
            mock_make.return_value = "pz_vec"
            spec.make_env(extra_wrappers=["Wrapper"])

        assert mock_make.call_args.kwargs["extra_wrappers"] == ["Wrapper"]


# ---------------------------------------------------------------------------
# OfflineEnvSpec
# ---------------------------------------------------------------------------
class TestOfflineEnvSpec:
    def test_requires_dataset_source(self):
        with pytest.raises(ValueError, match="requires either"):
            OfflineEnvSpec(name="CartPole-v1")

    def test_valid_with_dataset_path(self):
        spec = OfflineEnvSpec(name="CartPole-v1", dataset_path="/tmp/data.h5")
        assert spec.dataset_path == "/tmp/data.h5"
        assert spec.minari_dataset_id is None

    def test_valid_with_minari_id(self):
        spec = OfflineEnvSpec(name="CartPole-v1", minari_dataset_id="cartpole-v0")
        assert spec.minari_dataset_id == "cartpole-v0"
        assert spec.dataset_path is None

    def test_valid_with_remote(self):
        spec = OfflineEnvSpec(
            name="CartPole-v1",
            minari_dataset_id="cartpole-v0",
            remote=True,
        )
        assert spec.remote is True

    def test_inherits_gym_fields(self):
        spec = OfflineEnvSpec(
            name="CartPole-v1",
            dataset_path="/tmp/data.h5",
            num_envs=4,
            sync=True,
        )
        assert spec.num_envs == 4
        assert spec.sync is True


# ---------------------------------------------------------------------------
# BanditEnvSpec - parquet loading + serializer
# ---------------------------------------------------------------------------
class TestBanditEnvSpecExtended:
    def test_parquet_loading(self, tmp_path):
        import pandas as pd

        features = pd.DataFrame(np.random.randn(10, 3).astype(np.float32))
        targets = pd.DataFrame(np.random.randint(0, 2, size=(10, 1)))
        feat_path = tmp_path / "features.parquet"
        tgt_path = tmp_path / "targets.parquet"
        features.to_parquet(feat_path)
        targets.to_parquet(tgt_path)

        spec = BanditEnvSpec(
            features=str(feat_path),
            targets=str(tgt_path),
        )
        env = spec.make_env()
        assert hasattr(env, "arms")
        state = env.reset()
        assert state is not None

    def test_serializer_with_dataframe(self):
        import pandas as pd

        features = pd.DataFrame(np.random.randn(5, 2))
        targets = pd.DataFrame(np.random.randint(0, 2, size=(5, 1)))
        spec = BanditEnvSpec(features=features, targets=targets)
        data = spec.model_dump(mode="json")
        assert data["features"] is None
        assert data["targets"] is None

    def test_make_env_dataset_mode_missing_targets_raises(self):
        import pandas as pd

        features = pd.DataFrame(np.random.randn(5, 3).astype(np.float32))
        # The "features + targets together" validator normally blocks this;
        # model_construct bypasses it to reach make_env's dataset-mode guard.
        spec = BanditEnvSpec.model_construct(features=features)
        assert spec.targets is None
        assert spec.entrypoint is None
        with pytest.raises(
            ValueError, match="Both 'features' and 'targets' are required"
        ):
            spec.make_env()


# ---------------------------------------------------------------------------
# LLM Multiturn
# ---------------------------------------------------------------------------
class TestLLMEnvSpecMultiturn:
    """Verify LLMEnvSpec validation and factory creation for multiturn."""

    def test_multiturn_requires_env_name_or_entrypoint(self):
        with pytest.raises(ValueError, match="Exactly one of env_name or entrypoint"):
            LLMEnvSpec(env_type=LLMEnvType.MULTITURN, max_turns=5)

    def test_multiturn_rejects_both_env_name_and_entrypoint(self):
        with pytest.raises(ValueError, match="Exactly one of env_name or entrypoint"):
            LLMEnvSpec(
                env_type=LLMEnvType.MULTITURN,
                env_name="game:Test-v0",
                entrypoint="my_mod:make",
                max_turns=5,
            )

    def test_env_config_rejected_with_env_name(self):
        with pytest.raises(ValueError, match="env_config is only used with entrypoint"):
            LLMEnvSpec(
                env_type=LLMEnvType.MULTITURN,
                env_name="game:Test-v0",
                env_config={"difficulty": "easy"},
                max_turns=5,
            )

    def test_multiturn_valid_spec_gem(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            env_name="game:GuessTheNumber-v0-easy",
            max_turns=10,
        )
        assert spec.env_type == LLMEnvType.MULTITURN
        assert spec.max_turns == 10
        assert spec.env_name == "game:GuessTheNumber-v0-easy"
        assert spec.name == "game:GuessTheNumber-v0-easy"

    def test_multiturn_valid_spec_entrypoint(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            entrypoint="my_mod:MyEnv",
            env_config={"difficulty": "easy"},
            max_turns=10,
        )
        assert spec.env_type == LLMEnvType.MULTITURN
        assert spec.entrypoint == "my_mod:MyEnv"
        assert spec.env_config == {"difficulty": "easy"}
        assert spec.name == "my_mod:MyEnv"

    def test_multiturn_max_turns_optional(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            env_name="game:GuessTheNumber-v0-easy",
        )
        assert spec.max_turns is None

    def test_multiturn_make_env_raises(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            env_name="game:GuessTheNumber-v0-easy",
            max_turns=5,
        )
        with pytest.raises(TypeError, match="make_multiturn_env_factory"):
            spec.make_env(tokenizer=MagicMock())

    def test_gem_missing_gives_helpful_error(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            env_name="game:Test-v0",
            max_turns=5,
        )
        with (
            patch.dict("sys.modules", {"gem": None}),
            pytest.raises(ImportError, match="pip install gem-llm"),
        ):
            spec.make_multiturn_env_factory(MagicMock())

    def test_gem_factory_creates_wrapper(self):
        mock_env = MagicMock()
        mock_gem = MagicMock()
        mock_gem.make = MagicMock(return_value=mock_env)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            env_name="game:Test-v0",
            max_turns=5,
        )

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()

        with (
            patch.dict("sys.modules", {"gem": mock_gem}),
            patch("agilerl.llm_envs.TokenObservationWrapper", mock_wrapper_cls),
        ):
            factory = spec.make_multiturn_env_factory(
                mock_tokenizer, max_model_len=512, max_output_tokens=128
            )
            assert callable(factory)
            factory()

        mock_gem.make.assert_called_with("game:Test-v0")
        mock_wrapper_cls.assert_called_once_with(
            env=mock_env,
            tokenizer=mock_tokenizer,
            max_turns=5,
            pad_id=0,
            apply_chat_template=True,
            max_model_len=512,
            max_output_tokens=128,
            strict_chat_template_boundary=True,
        )

    def test_entrypoint_factory_creates_wrapper(self):
        mock_env = MagicMock()
        mock_constructor = MagicMock(return_value=mock_env)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            entrypoint="my_mod:MyEnv",
            env_config={"difficulty": "easy"},
            max_turns=5,
        )

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()

        with (
            patch(
                "agilerl.models.env.resolve_entrypoint_target",
                return_value=mock_constructor,
            ),
            patch("agilerl.llm_envs.TokenObservationWrapper", mock_wrapper_cls),
        ):
            factory = spec.make_multiturn_env_factory(
                mock_tokenizer, max_model_len=512, max_output_tokens=128
            )
            assert callable(factory)
            factory()

        mock_constructor.assert_called_with(difficulty="easy")
        mock_wrapper_cls.assert_called_once_with(
            env=mock_env,
            tokenizer=mock_tokenizer,
            max_turns=5,
            pad_id=0,
            apply_chat_template=True,
            max_model_len=512,
            max_output_tokens=128,
            strict_chat_template_boundary=True,
        )

    @pytest.mark.parametrize("strict", [True, False])
    def test_factory_forwards_strict_chat_template_boundary(self, strict):
        mock_env = MagicMock()
        mock_gem = MagicMock()
        mock_gem.make = MagicMock(return_value=mock_env)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            env_name="game:Test-v0",
            max_turns=5,
            strict_chat_template_boundary=strict,
        )
        assert spec.strict_chat_template_boundary is strict

        mock_wrapper_cls = MagicMock()
        with (
            patch.dict("sys.modules", {"gem": mock_gem}),
            patch("agilerl.llm_envs.TokenObservationWrapper", mock_wrapper_cls),
        ):
            spec.make_multiturn_env_factory(mock_tokenizer)()

        assert (
            mock_wrapper_cls.call_args.kwargs["strict_chat_template_boundary"] is strict
        )

    def test_strict_chat_template_boundary_defaults_to_true(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            env_name="game:Test-v0",
            max_turns=5,
        )
        assert spec.strict_chat_template_boundary is True

    def test_gem_factory_probes_max_turns(self):
        mock_env = MagicMock()
        mock_env.max_turns = 7
        mock_gem = MagicMock()
        mock_gem.make = MagicMock(return_value=mock_env)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            env_name="game:Test-v0",
        )
        assert spec.max_turns is None

        with patch.dict("sys.modules", {"gem": mock_gem}):
            spec.make_multiturn_env_factory(mock_tokenizer)

        assert spec.max_turns == 7

    def test_entrypoint_factory_probes_max_turns(self):
        mock_env = MagicMock()
        mock_env.max_turns = 12
        mock_constructor = MagicMock(return_value=mock_env)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            entrypoint="my_mod:MyEnv",
        )
        assert spec.max_turns is None

        with patch(
            "agilerl.models.env.resolve_entrypoint_target",
            return_value=mock_constructor,
        ):
            spec.make_multiturn_env_factory(mock_tokenizer)

        assert spec.max_turns == 12

    def test_factory_requires_env_name_or_entrypoint(self):
        # A validated multiturn spec always has exactly one of env_name /
        # entrypoint; model_construct bypasses that validator so we can reach
        # the factory's own defensive guard for the missing-both state.
        spec = LLMEnvSpec.model_construct(env_type=LLMEnvType.MULTITURN)
        assert spec.env_name is None
        assert spec.entrypoint is None
        with pytest.raises(ValueError, match="Exactly one of env_name or entrypoint"):
            spec.make_multiturn_env_factory(MagicMock())

    def test_dataset_still_required_for_reasoning(self):
        with pytest.raises(ValueError, match="dataset is required"):
            LLMEnvSpec(
                env_type=LLMEnvType.REASONING,
                reward_file_path="r.py",
                reward_fn_name="fn",
                prompt_template={"user_0": "Q: {question}"},
            )

    def test_dataset_not_required_for_multiturn(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            env_name="game:Test-v0",
        )
        assert spec.dataset is None


class TestLLMEnvSpecLoadDataset:
    """Defensive ``dataset is None`` guards in the private dataset loaders."""

    def _spec_without_dataset(self) -> LLMEnvSpec:
        spec = LLMEnvSpec(env_type=LLMEnvType.MULTITURN, env_name="game:Test-v0")
        assert spec.dataset is None
        return spec

    def test_load_dataset_requires_dataset(self):
        spec = self._spec_without_dataset()
        with pytest.raises(ValueError, match="dataset is required to load"):
            spec._load_dataset()

    def test_load_dataset_hf_requires_dataset(self):
        spec = self._spec_without_dataset()
        with pytest.raises(ValueError, match="dataset is required to load"):
            spec._load_dataset_hf()

    def test_load_dataset_file_requires_dataset(self):
        spec = self._spec_without_dataset()
        with pytest.raises(ValueError, match="dataset is required to load"):
            spec._load_dataset_file()


class TestLLMEnvSpecMakeReasoningEnv:
    """Defensive guard in ``_make_reasoning_env`` for missing reward fields."""

    def test_missing_reward_fields_raises(self):
        # A multiturn spec has reward_fn_name / reward_file_path /
        # prompt_template all None; the reasoning builder's guard rejects it
        # before the (mocked) datasets/tokenizer are ever used.
        spec = LLMEnvSpec(env_type=LLMEnvType.MULTITURN, env_name="game:Test-v0")
        with pytest.raises(
            ValueError,
            match="reward_fn_name, reward_file_path, and prompt_template",
        ):
            spec._make_reasoning_env(MagicMock(), MagicMock(), MagicMock())


class TestRequireDatasets:
    """_require_datasets ImportError when HuggingFace datasets is missing."""

    def test_require_datasets_import_error(self, monkeypatch):
        import builtins

        from agilerl.models.env import _require_datasets

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datasets":
                message = "No module named 'datasets'"
                raise ImportError(message)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="pip install 'agilerl\\[llm\\]'"):
            _require_datasets()
