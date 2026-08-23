# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import agilerl.llm_envs  # noqa: F401
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


class _StubTextEnv:
    """Minimal text env, resolvable as an entrypoint target."""

    max_turns = 2

    def reset(self, seed=None):
        return "hi", {}

    def step(self, action):
        return "", 0.0, True, False, {}


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

        with patch("agilerl.models.env.make_vect_envs") as make_vect_mock:
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

        with patch("agilerl.models.env.make_vect_envs") as make_vect_mock:
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

        with patch("agilerl.models.env.make_multi_agent_vect_envs") as make_multi_mock:
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

        with patch("agilerl.models.env.make_multi_agent_vect_envs") as make_multi_mock:
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

        with patch("agilerl.models.env.make_multi_agent_vect_envs") as make_multi_mock:
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
    def test_reasoning_requires_rubric_file_path(self):
        with pytest.raises(ValueError, match="rubric_file_path is required"):
            LLMEnvSpec(env_type=LLMEnvType.ROLLOUT, dataset="ds", rubric_file_path=None)

    def test_accepts_legacy_reward_keys(self):
        """Arena manifests may still send reward_file_path / reward_fn_name."""
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            dataset="ds",
            reward_file_path="reward.py",
            reward_fn_name="combined_rewards",
            prompt_template={"user_0": "{question}"},
        )
        assert spec.rubric_file_path == "reward.py"
        assert spec.rubric_name == "combined_rewards"

    def test_dataset_does_not_require_rubric_file_path(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="preference",
            dataset="ds",
            rubric_file_path=None,
        )
        assert spec.rubric_file_path is None

    def test_default_fields(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            dataset="dataset.parquet",
            rubric_file_path="reward.py",
            rubric_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )
        assert spec.train_test_split == 0.9
        assert spec.rubric_file_path == "reward.py"
        assert spec.dataset == "dataset.parquet"
        assert spec.columns is None
        assert spec.max_reward is None
        assert spec.chat_template_kwargs == {}

    def test_is_standalone_model(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            dataset="dataset.parquet",
            rubric_file_path="reward.py",
            rubric_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )
        assert not hasattr(spec, "num_envs")

    def test_custom_fields(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            columns={"question": "input", "answer": "output"},
            prompt_template={"role": "user", "content": "{question}"},
            max_reward=5.0,
            train_test_split=0.8,
            rubric_file_path="my_reward.py",
            rubric_name="my_reward",
            dataset="data/train.parquet",
        )
        assert spec.columns == {"question": "input", "answer": "output"}
        assert spec.prompt_template == {"role": "user", "content": "{question}"}
        assert spec.max_reward == 5.0
        assert spec.train_test_split == 0.8
        assert spec.rubric_file_path == "my_reward.py"
        assert spec.dataset == "data/train.parquet"

    def test_train_test_split_bounds(self):
        with pytest.raises(ValidationError):
            LLMEnvSpec(env_type=LLMEnvType.ROLLOUT, dataset="ds", train_test_split=1.5)
        with pytest.raises(ValidationError):
            LLMEnvSpec(env_type=LLMEnvType.ROLLOUT, dataset="ds", train_test_split=-0.1)

    @patch("agilerl.models.env.make_conversation_template")
    @patch("agilerl.models.env.get_rubric_factory")
    @patch.object(LLMEnvSpec, "_load_dataset")
    def test_dataset_backed_rollout_factory(
        self, mock_load, mock_get_rubric_factory, mock_conv_tmpl
    ):
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_load.return_value = (mock_train_ds, mock_test_ds)
        rubric = mock_get_rubric_factory.return_value.return_value
        mock_conv_tmpl.return_value = [{"role": "user", "content": "{question}"}]
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            dataset="dataset.parquet",
            rubric_file_path="reward.py",
            rubric_name="reward_fn",
            prompt_template={"user_0": "{question}"},
        )
        # A dataset-backed rollout is single-turn by construction.
        assert spec.max_turns == 1

        mock_rollout_cls = MagicMock()
        with patch("agilerl.models.env.RolloutHarness", mock_rollout_cls):
            factory = spec.make_rollout_env_factory(mock_tokenizer)
            # Building the factory reads nothing.
            mock_load.assert_not_called()
            mock_get_rubric_factory.assert_not_called()
            factory()

        mock_get_rubric_factory.assert_called_once_with(
            rubric_name="reward_fn", file_path="reward.py"
        )
        mock_conv_tmpl.assert_called_once()
        mock_rollout_cls.from_dataset.assert_called_once()
        args, kwargs = mock_rollout_cls.from_dataset.call_args
        assert args[0] is mock_train_ds
        assert args[1] is rubric
        assert kwargs["test_dataset"] is mock_test_ds
        assert kwargs["apply_chat_template"] is False

        # The builder renders the template for a row, so it can interpolate any
        # column -- not just the question.
        mock_tokenizer.apply_chat_template.return_value = "<rendered>"
        assert kwargs["prompt_builder"]({"question": "2+2", "answer": "4"}) == (
            "<rendered>"
        )
        (messages,) = mock_tokenizer.apply_chat_template.call_args.args
        assert messages == [{"role": "user", "content": "2+2"}]

    @patch("agilerl.models.env.make_conversation_template")
    @patch("agilerl.models.env.get_rubric_factory")
    @patch.object(LLMEnvSpec, "_load_dataset")
    def test_chat_template_kwargs_reach_dataset_factory_and_prompt_builder(
        self, mock_load, mock_get_rubric_factory, mock_conv_tmpl
    ):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_conv_tmpl.return_value = [{"role": "user", "content": "{question}"}]
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            dataset="dataset.parquet",
            rubric_file_path="reward.py",
            rubric_name="reward_fn",
            prompt_template={"user_0": "{question}"},
            chat_template_kwargs={"enable_thinking": False},
        )
        mock_rollout_cls = MagicMock()
        with patch("agilerl.models.env.RolloutHarness", mock_rollout_cls):
            spec.make_rollout_env_factory(mock_tokenizer)()

        kwargs = mock_rollout_cls.from_dataset.call_args.kwargs
        # apply_chat_template is False on this env, so the kwargs would be dead
        # there; the prompt_builder is what renders, so they must reach its call.
        assert "chat_template_kwargs" not in kwargs
        mock_tokenizer.apply_chat_template.return_value = "<rendered>"
        kwargs["prompt_builder"]({"question": "2+2"})
        render_kwargs = mock_tokenizer.apply_chat_template.call_args.kwargs
        assert render_kwargs["enable_thinking"] is False

    def test_chat_template_kwargs_reach_url_factory(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            env_url="ws://envs.internal:8000",
            max_turns=3,
            chat_template_kwargs={"enable_thinking": False},
        )
        mock_rollout_cls = MagicMock()
        with patch("agilerl.models.env.RolloutHarness", mock_rollout_cls):
            spec.make_rollout_env_factory(mock_tokenizer)()
        assert mock_rollout_cls.call_args.kwargs["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    def test_env_packages_are_installed_before_the_entrypoint_is_resolved(self):
        """An entrypoint whose deps are declared gets them installed first."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="tests.test_models.test_env:_StubTextEnv",
            env_packages={"uv": ["some-env-lib"]},
            max_turns=2,
        )
        with (
            patch("agilerl.models.env.ensure_importable") as mock_ensure,
            patch("agilerl.models.env.RolloutHarness", MagicMock()),
        ):
            spec.make_rollout_env_factory(mock_tokenizer)()

        mock_ensure.assert_called_once_with(
            "tests.test_models.test_env:_StubTextEnv", {"uv": ["some-env-lib"]}
        )

    def test_chat_template_kwargs_reach_entrypoint_factory(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="tests.test_models.test_env:_StubTextEnv",
            max_turns=2,
            chat_template_kwargs={"enable_thinking": False},
        )
        mock_rollout_cls = MagicMock()
        with patch("agilerl.models.env.RolloutHarness", mock_rollout_cls):
            spec.make_rollout_env_factory(mock_tokenizer)()
        assert mock_rollout_cls.local.call_args.kwargs["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    @patch.object(LLMEnvSpec, "_load_dataset")
    def test_make_env_preference(self, mock_load):
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_load.return_value = (mock_train_ds, mock_test_ds)
        mock_tokenizer = MagicMock()

        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="preference",
            dataset="dataset.parquet",
            rubric_file_path=None,
            chat_template_kwargs={"enable_thinking": False},
        )

        with patch("agilerl.models.env.DatasetEnv") as MockEnv:
            MockEnv.return_value = "dataset_env"
            result = spec.make_dataset_env(tokenizer=mock_tokenizer)

        assert result == "dataset_env"
        MockEnv.assert_called_once()
        assert MockEnv.call_args.kwargs["objective"] == "preference"
        assert MockEnv.call_args.kwargs["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    def test_serialization_roundtrip(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            columns={"q": "question"},
            max_reward=10.0,
            dataset="data.parquet",
            rubric_file_path="reward.py",
            rubric_name="reward_fn",
            prompt_template={"role": "user", "content": "{q}"},
        )
        data = spec.model_dump()
        restored = LLMEnvSpec.model_validate(data)
        assert restored.env_type == LLMEnvType.ROLLOUT
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


class TestLLMEnvSpecDatasetSplit:
    """Every rank splits the dataset for itself, so the split must not be random."""

    @staticmethod
    def _write_rows(tmp_path, n=40):
        path = tmp_path / "rows.parquet"
        pd.DataFrame(
            {
                "question": [f"q{i}" for i in range(n)],
                "answer": [f"a{i}" for i in range(n)],
            }
        ).to_parquet(path)
        return path

    @staticmethod
    def _spec(path, seed):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            dataset=str(path),
            rubric_file_path="reward.py",
            rubric_name="reward_fn",
            prompt_template={"user_0": "{question}"},
        )
        spec.seed = seed
        return spec

    def test_same_seed_gives_every_rank_the_same_rows(self, tmp_path):
        """Row ``n`` must be the same row on every rank, or index shards overlap."""
        path = self._write_rows(tmp_path)

        train_a, test_a = self._spec(path, 7)._load_dataset()
        train_b, test_b = self._spec(path, 7)._load_dataset()

        assert train_a["question"] == train_b["question"]
        assert test_a["question"] == test_b["question"]

    def test_unset_seed_still_splits_deterministically(self, tmp_path):
        """A manifest that names no seed must not fall back to a random split."""
        path = self._write_rows(tmp_path)

        train_a, _ = self._spec(path, None)._load_dataset()
        train_b, _ = self._spec(path, None)._load_dataset()

        assert train_a["question"] == train_b["question"]

    def test_different_seeds_give_different_splits(self, tmp_path):
        path = self._write_rows(tmp_path)

        train_a, _ = self._spec(path, 7)._load_dataset()
        train_c, _ = self._spec(path, 8)._load_dataset()

        assert train_a["question"] != train_c["question"]


# ---------------------------------------------------------------------------
# LLM SFT
# ---------------------------------------------------------------------------
class TestLLMEnvSpecSFT:
    def test_sft_spec_valid_construction(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset="my_dataset.parquet",
            response_column="answer",
        )
        assert spec.env_type == LLMEnvType.DATASET
        assert spec.dataset == "my_dataset.parquet"
        assert spec.response_column == "answer"
        assert spec.rubric_file_path is None

    def test_sft_default_response_column(self):
        spec = LLMEnvSpec(env_type=LLMEnvType.DATASET, objective="sft", dataset="ds")
        assert spec.response_column == "response"

    @patch.object(LLMEnvSpec, "_load_dataset")
    def test_sft_make_env_creates_sft_gym(self, mock_load):
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_load.return_value = (mock_train_ds, mock_test_ds)
        mock_tokenizer = MagicMock()

        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset="sft_data.parquet",
            response_column="completion",
        )

        with patch("agilerl.models.env.DatasetEnv") as MockEnv:
            MockEnv.return_value = "dataset_env"
            result = spec.make_dataset_env(tokenizer=mock_tokenizer)

        assert result == "dataset_env"
        MockEnv.assert_called_once()
        call_kwargs = MockEnv.call_args.kwargs
        assert call_kwargs["train_dataset"] is mock_train_ds
        assert call_kwargs["test_dataset"] is mock_test_ds
        assert call_kwargs["objective"] == "sft"
        assert call_kwargs["response_column"] == "completion"

    def test_sft_rejects_rubric_file_path(self):
        with pytest.raises(ValueError, match="not supported for dataset"):
            LLMEnvSpec(
                env_type=LLMEnvType.DATASET,
                objective="sft",
                dataset="ds",
                rubric_file_path="reward.py",
            )

    def test_preference_rejects_rubric_file_path(self):
        with pytest.raises(ValueError, match="not supported for dataset"):
            LLMEnvSpec(
                env_type=LLMEnvType.DATASET,
                objective="preference",
                dataset="ds",
                rubric_file_path="reward.py",
            )

    def test_dataset_alias_name(self):
        spec = LLMEnvSpec.model_validate(
            {
                "name": "my_dataset",
                "env_type": "rollout",
                "rubric_file_path": "r.py",
                "rubric_name": "fn",
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

        with patch("agilerl.models.env.make_vect_envs") as mock_make:
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

        with patch("agilerl.models.env.make_multi_agent_vect_envs") as mock_make:
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
# LLM Rollout
# ---------------------------------------------------------------------------
class TestLLMEnvSpecRollout:
    """Verify LLMEnvSpec validation and factory creation for rollout."""

    def test_rollout_requires_a_source(self):
        with pytest.raises(ValueError, match="Exactly one of dataset, entrypoint"):
            LLMEnvSpec(env_type=LLMEnvType.ROLLOUT, max_turns=5)

    def test_rollout_rejects_two_sources(self):
        with pytest.raises(ValueError, match="Exactly one of dataset, entrypoint"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                entrypoint="my_mod:make",
                env_url="http://envs:8000",
                max_turns=5,
            )

    def test_dataset_backed_rollout_rejects_multi_turn(self):
        with pytest.raises(ValueError, match="single-turn; max_turns must be 1"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                dataset="ds",
                rubric_file_path="r.py",
                rubric_name="fn",
                prompt_template={"user_0": "Q: {question}"},
                max_turns=4,
            )

    def test_env_backed_rollout_rejects_rubric_file_path(self):
        with pytest.raises(ValueError, match="not supported for env-backed"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                entrypoint="my_mod:make",
                rubric_file_path="r.py",
                max_turns=5,
            )

    def test_env_config_rejected_without_entrypoint(self):
        with pytest.raises(ValueError, match="env_config is only used with entrypoint"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                env_url="http://envs:8000",
                env_config={"difficulty": "easy"},
                max_turns=5,
            )

    def test_rollout_valid_spec_library_factory(self):
        # A library's own factory is just an entrypoint; its env id rides in
        # env_config like any other constructor argument.
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="gem:make",
            env_config={"env_id": "game:GuessTheNumber-v0-easy"},
            max_turns=10,
        )
        assert spec.env_type == LLMEnvType.ROLLOUT
        assert spec.max_turns == 10
        assert spec.entrypoint == "gem:make"
        assert spec.name == "gem:make"

    def test_rollout_valid_spec_entrypoint(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:MyEnv",
            env_config={"difficulty": "easy"},
            max_turns=10,
        )
        assert spec.env_type == LLMEnvType.ROLLOUT
        assert spec.entrypoint == "my_mod:MyEnv"
        assert spec.env_config == {"difficulty": "easy"}
        assert spec.name == "my_mod:MyEnv"

    def test_rollout_valid_spec_env_url(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            env_url="http://env-host:8000",
            max_turns=4,
        )
        assert spec.env_url == "http://env-host:8000"
        assert spec.name == "http://env-host:8000"
        assert spec.max_turns == 4

    def test_env_url_requires_max_turns(self):
        with pytest.raises(ValueError, match="max_turns is required with env_url"):
            LLMEnvSpec(env_type=LLMEnvType.ROLLOUT, env_url="http://h:8000")

    def test_env_url_is_mutually_exclusive_with_other_sources(self):
        with pytest.raises(ValueError, match="Exactly one of dataset, entrypoint"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                env_url="http://h:8000",
                entrypoint="my_mod:make",
                max_turns=1,
            )

    def test_mcp_tool_requires_env_url(self):
        with pytest.raises(
            ValueError,
            match="mcp_tool / request_timeout_s",
        ):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                entrypoint="my_mod:make",
                max_turns=1,
                mcp_tool="run_code",
            )

    def test_action_field_applies_to_entrypoint_envs(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:make",
            max_turns=1,
            action_field="code",
        )
        assert spec.action_field == "code"

    def test_action_field_rejected_for_dataset_backed_rollout(self):
        with pytest.raises(ValueError, match="no such env"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                dataset="my_dataset",
                rubric_file_path="rubric.py",
                rubric_name="rubric",
                prompt_template={"user_prompt": "{question}"},
                action_field="code",
            )

    def test_observation_field_applies_to_entrypoint_envs(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:make",
            max_turns=1,
            observation_field="board",
        )
        assert spec.observation_field == "board"

    def test_observation_field_and_processor_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                entrypoint="my_mod:make",
                max_turns=1,
                observation_field="board",
                observation_processor="proc.py:render",
            )

    def test_observation_processor_rejected_for_dataset_backed_rollout(self):
        with pytest.raises(ValueError, match="builds its own prompts"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                dataset="my_dataset",
                rubric_file_path="rubric.py",
                rubric_name="rubric",
                prompt_template={"user_prompt": "{question}"},
                observation_processor="proc.py:render",
            )

    def test_env_url_factory_builds_session_client(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            env_url="http://env-host:8000",
            max_turns=4,
            mcp_tool="run_code",
            request_timeout_s=30.0,
        )
        mock_rollout_cls = MagicMock()
        mock_session_cls = MagicMock()
        with (
            patch("agilerl.models.env.RolloutHarness", mock_rollout_cls),
            patch("agilerl.llm_envs.openenv.RemoteEnvClient", mock_session_cls),
        ):
            spec.make_rollout_env_factory(
                mock_tokenizer, max_model_len=512, max_output_tokens=128
            )()
        # env_url opens a WebSocket session; the transport params bake into the client.
        mock_session_cls.assert_called_once_with(
            "http://env-host:8000",
            mcp_tool="run_code",
            action_field="message",
            timeout_s=30.0,
        )
        mock_rollout_cls.assert_called_once_with(
            mock_session_cls.return_value,
            mock_tokenizer,
            max_turns=4,
            observation_field=None,
            observation_processor=None,
            apply_chat_template=True,
            chat_template_kwargs={},
            max_model_len=512,
            max_output_tokens=128,
            strict_chat_template_boundary=True,
        )
        mock_rollout_cls.local.assert_not_called()

    def test_rollout_max_turns_optional(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:make",
        )
        assert spec.max_turns is None

    def test_rollout_make_env_raises(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:make",
            max_turns=5,
        )
        with pytest.raises(TypeError, match="make_rollout_env_factory"):
            spec.make_dataset_env(tokenizer=MagicMock())

    def test_entrypoint_factory_creates_wrapper(self):
        mock_env = MagicMock()
        mock_constructor = MagicMock(return_value=mock_env)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:MyEnv",
            env_config={"difficulty": "easy"},
            max_turns=5,
        )

        mock_rollout_cls = MagicMock()

        with (
            patch(
                "agilerl.models.env.resolve_entrypoint_target",
                return_value=mock_constructor,
            ),
            patch("agilerl.models.env.RolloutHarness", mock_rollout_cls),
        ):
            factory = spec.make_rollout_env_factory(
                mock_tokenizer, max_model_len=512, max_output_tokens=128
            )
            assert callable(factory)
            factory()

        mock_constructor.assert_called_with(difficulty="easy")
        mock_rollout_cls.local.assert_called_once_with(
            mock_env,
            mock_tokenizer,
            max_turns=5,
            action_field="message",
            observation_field=None,
            observation_processor=None,
            apply_chat_template=True,
            chat_template_kwargs={},
            max_model_len=512,
            max_output_tokens=128,
            strict_chat_template_boundary=True,
        )

    def test_entrypoint_factory_does_not_forward_system_prompt_to_constructor(self):
        mock_env = MagicMock()
        mock_constructor = MagicMock(return_value=mock_env)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:MyEnv",
            env_config={"difficulty": "easy", "system_prompt": "be terse"},
            max_turns=5,
        )
        mock_rollout_cls = MagicMock()

        with (
            patch(
                "agilerl.models.env.resolve_entrypoint_target",
                return_value=mock_constructor,
            ),
            patch("agilerl.models.env.RolloutHarness", mock_rollout_cls),
        ):
            spec.make_rollout_env_factory(
                mock_tokenizer, max_model_len=512, max_output_tokens=128
            )()

        mock_constructor.assert_called_with(difficulty="easy")
        assert mock_env.system_prompt == "be terse"

    def test_observation_processor_resolves_and_reaches_the_harness(self, tmp_path):
        proc_file = tmp_path / "proc.py"
        proc_file.write_text(
            "def render(payload):\n    return 'obs:' + str(payload['info_state'])\n"
        )
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="tests.test_models.test_env:_StubTextEnv",
            max_turns=2,
            observation_processor=f"{proc_file}:render",
        )
        mock_rollout_cls = MagicMock()
        with patch("agilerl.models.env.RolloutHarness", mock_rollout_cls):
            spec.make_rollout_env_factory(mock_tokenizer)()
        kwargs = mock_rollout_cls.local.call_args.kwargs
        assert kwargs["observation_field"] is None
        assert kwargs["observation_processor"]({"info_state": [1, 2]}) == "obs:[1, 2]"

    def test_observation_processor_must_resolve_to_a_callable(self, tmp_path):
        proc_file = tmp_path / "proc.py"
        proc_file.write_text("render = 3\n")
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="tests.test_models.test_env:_StubTextEnv",
            max_turns=1,
            observation_processor=f"{proc_file}:render",
        )
        with pytest.raises(TypeError, match="non-callable"):
            spec.make_rollout_env_factory(mock_tokenizer)

    def test_entrypoint_factory_probes_max_turns(self):
        mock_env = MagicMock()
        mock_env.max_turns = 12
        mock_constructor = MagicMock(return_value=mock_env)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:MyEnv",
        )
        assert spec.max_turns is None

        with patch(
            "agilerl.models.env.resolve_entrypoint_target",
            return_value=mock_constructor,
        ):
            spec.make_rollout_env_factory(mock_tokenizer)

        assert spec.max_turns == 12

    def test_factory_rejects_zero_prompt_budget(self):
        """``max_output_tokens >= max_model_len`` leaves no prompt room (budget
        0), so every rollout episode is truncated at reset and training can never
        advance. The factory must reject it at setup rather than build envs that
        silently produce a non-terminating loop.
        """
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:MyEnv",
            max_turns=3,
        )

        with pytest.raises(ValueError, match="must be less than"):
            spec.make_rollout_env_factory(
                mock_tokenizer, max_model_len=512, max_output_tokens=512
            )

    def test_rubric_fields_alone_are_not_a_source(self):
        with pytest.raises(ValueError, match="Exactly one of dataset, entrypoint"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                rubric_file_path="r.py",
                rubric_name="fn",
                prompt_template={"user_0": "Q: {question}"},
            )

    def test_dataset_not_required_for_rollout(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:make",
        )
        assert spec.dataset is None

    def test_env_packages_rides_along_with_an_entrypoint(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="gem:make",
            env_config={"env_id": "game:Sudoku-v0-easy"},
            env_packages={"uv": ["gem-llm"]},
        )
        assert spec.env_packages == {"uv": ["gem-llm"]}

    @pytest.mark.parametrize(
        "source",
        [{"env_url": "http://env:8000"}, {"dataset": "rows.parquet"}],
    )
    def test_env_packages_needs_an_entrypoint_to_install_for(self, source):
        with pytest.raises(ValueError, match="nothing to install for"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                max_turns=1,
                env_packages={"uv": ["gem-llm"]},
                **source,
            )

    def test_env_packages_naming_nothing_is_rejected_at_validation(self):
        with pytest.raises(ValueError, match="named no packages"):
            LLMEnvSpec(
                env_type=LLMEnvType.ROLLOUT,
                entrypoint="gem:make",
                env_packages={"uv": []},
            )


class TestLLMEnvSpecLoadDataset:
    """Defensive ``dataset is None`` guards in the private dataset loaders."""

    def _spec_without_dataset(self) -> LLMEnvSpec:
        spec = LLMEnvSpec(env_type=LLMEnvType.ROLLOUT, entrypoint="my_mod:make")
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


class TestDatasetBackedRolloutRewardFields:
    """Defensive guard for a dataset-backed rollout missing its reward fields."""

    def test_missing_reward_fields_raises(self):
        # A validated dataset-backed rollout always carries rubric_name /
        # rubric_file_path / prompt_template; model_construct bypasses that
        # validator so we can reach the factory's own defensive guard, which
        # fires on first resolve — before any dataset or tokenizer is touched.
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        spec = LLMEnvSpec.model_construct(
            env_type=LLMEnvType.ROLLOUT, dataset="dummy/dataset"
        )
        assert spec.rubric_name is None

        factory = spec.make_rollout_env_factory(mock_tokenizer)
        with pytest.raises(
            ValueError,
            match="rubric_name, rubric_file_path, and prompt_template",
        ):
            factory()


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


class TestLLMEnvSpecFactoryGuards:
    def test_make_dataset_env_requires_objective_even_when_mutated_away(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            dataset="d.parquet",
            objective="sft",
        )
        spec.objective = None
        with pytest.raises(ValueError, match="objective is required"):
            spec.make_dataset_env(MagicMock())

    def test_factory_requires_a_source_even_when_mutated_away(self):
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="pkg.mod:Env",
            max_turns=2,
        )
        spec.entrypoint = None
        with pytest.raises(ValueError, match="An entrypoint is required"):
            spec.make_rollout_env_factory(MagicMock())

    def test_non_callable_entrypoint_target_is_rejected(self):
        # json.__version__ resolves fine but is a string, not an env factory.
        spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="json:__version__",
            max_turns=2,
        )
        with pytest.raises(TypeError, match="non-callable"):
            spec.make_rollout_env_factory(MagicMock())
