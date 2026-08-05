# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests targeting uncovered lines in agilerl/models/ Pydantic models.

Covers validators, edge-case coercion, optional field handling, and
algorithm-specific training function resolution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from agilerl import HAS_ARENA_DEPENDENCIES, HAS_LLM_DEPENDENCIES
from agilerl.models.algorithms.ppo import PPOSpec
from agilerl.models.env import BanditEnvSpec, GymEnvSpec, LLMEnvSpec, LLMEnvType
from agilerl.models.hpo import MutationSpec
from agilerl.models.manifest import TrainingManifest
from agilerl.models.networks import (
    CnnSpec,
    FinetuningNetworkSpec,
    LoraConfigDict,
    LstmSpec,
    MlpSpec,
    MultiInputSpec,
    NetworkSpec,
    SimbaSpec,
    min_max_validator,
    normalize_manifest_network,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec

requires_arena = pytest.mark.skipif(
    not HAS_ARENA_DEPENDENCIES, reason="agilerl-arena is not installed"
)


class TestNormalizeManifestNetwork:
    """normalize_manifest_network arch placement and validation."""

    def test_non_dict_passthrough(self):
        assert normalize_manifest_network(42) == 42
        assert normalize_manifest_network("hello") == "hello"

    def test_top_level_arch_without_encoder_config(self):
        normalized = normalize_manifest_network(
            {
                "arch": "mlp",
                "latent_dim": 64,
                "head_config": {"hidden_size": [64]},
            },
        )
        assert normalized["encoder_config"] == {"arch": "mlp"}

    def test_missing_arch_deferred(self):
        # Deferred: no arch declared anywhere, so the data is returned
        # unchanged rather than raising; the trainer resolves the arch later.
        data = {
            "encoder_config": {"hidden_size": [64]},
            "head_config": {"hidden_size": [64]},
        }
        normalized = normalize_manifest_network(data)
        assert "arch" not in normalized["encoder_config"]
        assert normalized == data


class TestMinMaxValidator:
    """min_max_validator raises on invalid range."""

    def test_raises_when_min_exceeds_max(self):
        model = MlpSpec(hidden_size=[64], min_hidden_layers=1, max_hidden_layers=6)
        model.min_mlp_nodes = 999
        model.max_mlp_nodes = 8
        with pytest.raises(ValueError, match="must be less than or equal to"):
            min_max_validator("min_mlp_nodes", "max_mlp_nodes")(model)


class TestMlpSpec:
    """Covers MlpSpec validators and model_dump."""

    def test_model_dump_removes_none_output_activation(self):
        """output_activation=None is stripped from dump."""
        spec = MlpSpec(hidden_size=[64])
        d = spec.model_dump()
        assert "output_activation" not in d

    def test_model_dump_keeps_output_activation(self):
        spec = MlpSpec(hidden_size=[64], output_activation="Tanh")
        d = spec.model_dump()
        assert d["output_activation"] == "Tanh"

    def test_hidden_size_below_min_hidden_layers(self):
        """hidden_size length < min_hidden_layers."""
        with pytest.raises(
            ValueError,
            match="hidden_size must be greater than or equal to min_hidden_layers",
        ):
            MlpSpec(hidden_size=[64], min_hidden_layers=3)

    def test_hidden_size_above_max_hidden_layers(self):
        """hidden_size length > max_hidden_layers."""
        with pytest.raises(
            ValueError,
            match="hidden_size must be less than or equal to max_hidden_layers",
        ):
            MlpSpec(hidden_size=[32] * 8, max_hidden_layers=3)

    def test_node_below_min_mlp_nodes(self):
        """hidden_size element < min_mlp_nodes."""
        with pytest.raises(
            ValueError,
            match="hidden_size elements must be greater than or equal to min_mlp_nodes",
        ):
            MlpSpec(hidden_size=[2], min_mlp_nodes=8)

    def test_node_above_max_mlp_nodes(self):
        """hidden_size element > max_mlp_nodes."""
        with pytest.raises(
            ValueError,
            match="hidden_size elements must be less than or equal to max_mlp_nodes",
        ):
            MlpSpec(hidden_size=[512], max_mlp_nodes=256)


class TestSimbaSpec:
    """Covers SimbaSpec validators."""

    def test_hidden_size_below_min_mlp_nodes(self):
        """hidden_size < min_mlp_nodes."""
        with pytest.raises(
            ValueError,
            match="hidden_size must be greater than or equal to min_mlp_nodes",
        ):
            SimbaSpec(hidden_size=4, num_blocks=1, min_mlp_nodes=8)

    def test_hidden_size_above_max_mlp_nodes(self):
        """hidden_size > max_mlp_nodes."""
        with pytest.raises(
            ValueError, match="hidden_size must be less than or equal to max_mlp_nodes"
        ):
            SimbaSpec(hidden_size=512, num_blocks=1, max_mlp_nodes=256)


class TestCnnSpec:
    """Covers CnnSpec validators."""

    def test_channel_size_outside_layer_range(self):
        """channel_size length out of [min, max] hidden layers."""
        with pytest.raises(ValueError, match="hidden_layers must be between"):
            CnnSpec(
                channel_size=[16] * 8,
                kernel_size=[3] * 8,
                stride_size=[1] * 8,
                max_hidden_layers=3,
            )

    def test_channel_below_min(self):
        """channel_size element < min_channel_size."""
        with pytest.raises(
            ValueError,
            match="channel_size must be greater than or equal to min_channel_size",
        ):
            CnnSpec(
                channel_size=[2],
                kernel_size=[3],
                stride_size=[1],
                min_channel_size=8,
            )

    def test_channel_above_max(self):
        """channel_size element > max_channel_size."""
        with pytest.raises(
            ValueError,
            match="channel_size must be less than or equal to max_channel_size",
        ):
            CnnSpec(
                channel_size=[512],
                kernel_size=[3],
                stride_size=[1],
                max_channel_size=256,
            )

    def test_mismatched_sizes(self):
        """channel/kernel/stride length mismatch."""
        with pytest.raises(ValueError, match="must have the same length"):
            CnnSpec(channel_size=[16, 32], kernel_size=[3], stride_size=[1, 1])


class TestFinetuningNetworkSpec:
    """PEFT LoRA coercion, resolution, and serialization in networks.py."""

    def test_coerce_non_dict_passthrough(self):
        """_coerce_peft_lora with non-dict data returns input unchanged."""
        payload = "not-a-dict"
        assert FinetuningNetworkSpec._coerce_peft_lora(payload) is payload

    @pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
    def test_coerce_peft_lora_config_instance(self):
        """PEFT LoraConfig is converted before Pydantic validation."""
        from types import SimpleNamespace

        from peft import LoraConfig, TaskType

        from agilerl.models.networks import _peft_lora_config_to_manifest_dict

        peft_lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        manifest_dict = _peft_lora_config_to_manifest_dict(peft_lora)
        assert manifest_dict["target_modules"] == ["q_proj", "v_proj"]
        assert manifest_dict["task_type"] == "CAUSAL_LM"

        enum_task = SimpleNamespace(value="SEQ_CLS")
        manifest_dict = _peft_lora_config_to_manifest_dict(
            SimpleNamespace(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules={"b", "a"},
                task_type=enum_task,
            )
        )
        assert manifest_dict["task_type"] == "SEQ_CLS"
        assert manifest_dict["target_modules"] == ["a", "b"]

        spec = FinetuningNetworkSpec(
            pretrained_model_name_or_path="gpt2",
            max_context_length=512,
            lora_config=peft_lora,
        )
        assert spec.lora_config is not None
        assert spec.lora_config.r == 8

    @pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
    def test_coerce_peft_lora_none_target_modules(self):
        """PEFT LoraConfig with target_modules=None maps to all-linear."""
        from peft import LoraConfig, TaskType

        peft_lora = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=None,
            task_type=TaskType.CAUSAL_LM,
        )
        spec = FinetuningNetworkSpec(
            pretrained_model_name_or_path="gpt2",
            max_context_length=512,
            lora_config=peft_lora,
        )
        dumped = spec.model_dump(mode="json")
        assert dumped["lora_config"]["target_modules"] == "all-linear"

    def test_resolve_lora_config_without_llm_dependencies(self):
        """LoraConfigDict resolution raises when LLM extras are absent."""
        with patch("agilerl.models.networks.HAS_LLM_DEPENDENCIES", False):
            with pytest.raises(ImportError, match="LLM dependencies are required"):
                FinetuningNetworkSpec(
                    pretrained_model_name_or_path="gpt2",
                    max_context_length=512,
                    lora_config=LoraConfigDict(),
                )

    @pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
    def test_serialize_lora_config_non_json_mode(self):
        """_serialize_lora_config returns the live object outside json mode."""
        from peft import LoraConfig, TaskType

        peft_lora = LoraConfig(r=4, lora_alpha=8, task_type=TaskType.CAUSAL_LM)
        spec = FinetuningNetworkSpec(
            pretrained_model_name_or_path="gpt2",
            max_context_length=512,
            lora_config=peft_lora,
        )
        result = FinetuningNetworkSpec._serialize_lora_config(
            spec, peft_lora, MagicMock(mode="python")
        )
        assert result is peft_lora

    def test_serialize_lora_none(self):
        """_serialize_lora_config with None value."""
        spec = FinetuningNetworkSpec(
            pretrained_model_name_or_path="gpt2",
            max_context_length=512,
            lora_config=None,
        )
        dumped = spec.model_dump(mode="json")
        assert dumped["lora_config"] is None

    def test_serialize_lora_config_dict_json(self):
        """_serialize_lora_config with LoraConfigDict in json mode."""
        lora = LoraConfigDict()
        spec = FinetuningNetworkSpec.__new__(FinetuningNetworkSpec)
        object.__setattr__(
            spec,
            "__dict__",
            {
                "pretrained_model_name_or_path": "gpt2",
                "max_context_length": 512,
                "lora_config": lora,
            },
        )
        object.__setattr__(
            spec,
            "__pydantic_fields_set__",
            {"pretrained_model_name_or_path", "max_context_length", "lora_config"},
        )
        result = FinetuningNetworkSpec._serialize_lora_config(
            spec, lora, MagicMock(mode="json")
        )
        assert isinstance(result, dict)

    def test_lora_config_dict_serialize_set_target_modules(self):
        """Line 363, 368, 372 of networks.py: _serialize_target_modules with set."""
        lc = LoraConfigDict(target_modules={"a", "b"})
        d = lc.model_dump(mode="json")
        assert d["target_modules"] == ["a", "b"]


class TestLLMEnvSpec:
    """Covers LLMEnvSpec validators and properties."""

    def test_name_returns_dataset(self):
        """name property returns dataset."""
        spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE,
            dataset="my_dataset",
        )
        assert spec.name == "my_dataset"

    def test_reasoning_missing_reward_fn_name(self):
        """reasoning without reward_fn_name."""
        with pytest.raises(ValueError, match="reward_fn_name is required"):
            LLMEnvSpec(
                env_type=LLMEnvType.REASONING,
                dataset="ds",
                reward_file_path="some/path.py",
                prompt_template={"system": "hi"},
            )

    def test_reasoning_missing_prompt_template(self):
        """reasoning without prompt_template."""
        with pytest.raises(ValueError, match="Prompt template is required"):
            LLMEnvSpec(
                env_type=LLMEnvType.REASONING,
                dataset="ds",
                reward_file_path="some/path.py",
                reward_fn_name="my_fn",
            )

    def test_preference_missing_dataset(self):
        """preference without dataset."""
        with pytest.raises(ValueError, match="dataset is required for preference"):
            LLMEnvSpec(env_type=LLMEnvType.PREFERENCE)

    def test_sft_missing_dataset(self):
        """SFT without dataset."""
        with pytest.raises(ValueError, match="dataset is required for SFT"):
            LLMEnvSpec(env_type=LLMEnvType.SFT)

    def test_load_dataset_hf(self):
        """_load_dataset_hf with columns rename."""
        spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE,
            dataset="my_ds",
            columns={"old": "new"},
        )
        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.rename_columns.return_value = mock_ds
        mock_ds.train_test_split.return_value = {
            "train": "train_split",
            "test": "test_split",
        }

        with patch("datasets.load_dataset", return_value=mock_ds):
            train, test = spec._load_dataset_hf()

        mock_ds.rename_columns.assert_called_once_with({"old": "new"})
        assert train == "train_split"
        assert test == "test_split"

    def test_load_dataset_file_with_columns(self):
        """_load_dataset_file with columns rename."""
        spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE,
            dataset="data.parquet",
            columns={"old": "new"},
        )
        mock_df = MagicMock()
        mock_df.rename.return_value = mock_df
        mock_split = {"train": "train_split", "test": "test_split"}
        mock_hf_ds = MagicMock()
        mock_hf_ds.train_test_split.return_value = mock_split

        with (
            patch("agilerl.models.env.pd.read_parquet", return_value=mock_df),
            patch("datasets.Dataset.from_pandas", return_value=mock_hf_ds),
        ):
            _train, _test = spec._load_dataset_file()

        mock_df.rename.assert_called_once_with(columns={"old": "new"})

    def test_load_dataset_dispatches_hf(self):
        """_load_dataset dispatches to HF path for non-parquet."""
        spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE,
            dataset="some/hf/dataset",
        )
        with patch.object(spec, "_load_dataset_hf", return_value=("t", "v")) as m:
            _train, _test = spec._load_dataset()
        m.assert_called_once()

    def test_load_dataset_dispatches_parquet(self):
        """_load_dataset dispatches to the parquet loader for .parquet paths."""
        spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE,
            dataset="data/train.parquet",
        )
        with patch.object(spec, "_load_dataset_file", return_value=("t", "v")) as m:
            _train, _test = spec._load_dataset()
        m.assert_called_once()

    def test_make_env_invalid_type(self):
        """make_env with an unexpected env_type."""
        spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE,
            dataset="ds",
        )
        # Monkey-patch env_type to something that passes validators but doesn't
        # match any if-branch in make_env
        with (
            patch.object(
                spec, "_load_dataset", return_value=(MagicMock(), MagicMock())
            ),
            patch.object(spec, "_make_reasoning_env"),
            patch.object(spec, "_make_preference_env"),
            patch.object(spec, "_make_sft_env"),
        ):
            object.__setattr__(spec, "env_type", "unknown")
            with pytest.raises(ValueError, match="Invalid environment type"):
                spec.make_env(tokenizer=MagicMock())

    def test_multiturn_entrypoint_non_callable(self):
        """make_multiturn_env_factory with non-callable entrypoint."""
        spec = LLMEnvSpec(
            env_type=LLMEnvType.MULTITURN,
            entrypoint="some.module:Cls",
        )
        with patch(
            "agilerl.models.env.resolve_entrypoint_target", return_value="not_callable"
        ):
            with pytest.raises(TypeError, match="resolved to non-callable"):
                spec.make_multiturn_env_factory(tokenizer=MagicMock())


class TestBanditEnvSpec:
    """Covers BanditEnvSpec validators and _load_dataframe edge cases."""

    def test_hdf5_loading(self):
        """_load_dataframe with .h5 extension."""
        import pandas as pd

        spec = BanditEnvSpec(
            features=pd.DataFrame({"a": [1]}),
            targets=pd.DataFrame({"b": [2]}),
        )
        with patch(
            "agilerl.models.env.pd.read_hdf", return_value=pd.DataFrame({"x": [1]})
        ):
            df = spec._load_dataframe("data.h5")
        assert "x" in df.columns

    def test_unsupported_file_type(self):
        """_load_dataframe with unsupported extension."""
        import pandas as pd

        spec = BanditEnvSpec(
            features=pd.DataFrame({"a": [1]}),
            targets=pd.DataFrame({"b": [2]}),
        )
        with pytest.raises(ValueError, match="Unsupported file type"):
            spec._load_dataframe("data.xlsx")

    def test_entrypoint_non_callable(self):
        """make_env entrypoint resolves to non-callable."""
        spec = BanditEnvSpec(entrypoint="some.module:Cls")
        with patch(
            "agilerl.models.env.resolve_entrypoint_target", return_value="not_callable"
        ):
            with pytest.raises(TypeError, match="resolved to non-callable"):
                spec.make_env()


class TestGymEnvSpecNonCallable:
    """Lines 120-121 in env.py: construct_custom_env_fn with non-callable entrypoint."""

    def test_non_callable_entrypoint(self):
        from agilerl.models.env import GymEnvSpec

        with patch("agilerl.models.env.resolve_entrypoint_target", return_value=42):
            factory = GymEnvSpec.construct_custom_env_fn("some:entry")
            with pytest.raises(TypeError, match="resolved to non-callable"):
                factory()


class TestPzEnvSpecNonCallable:
    """Lines 219-220, 247-248, 275-276 in env.py."""

    def test_non_callable_entrypoint(self):
        from agilerl.models.env import PzEnvSpec

        with patch("agilerl.models.env.resolve_entrypoint_target", return_value=42):
            factory = PzEnvSpec.construct_custom_env_fn("some:entry")
            with pytest.raises(TypeError, match="resolved to non-callable"):
                factory()

    def test_make_single_env_no_parallel_env(self):
        """fallback path for PZ module without parallel_env."""
        from agilerl.models.env import PzEnvSpec

        spec = PzEnvSpec(name="fake_pz_module")
        mock_module = MagicMock(spec=[])  # no parallel_env attr
        with patch("agilerl.models.env.import_module", return_value=mock_module):
            with pytest.raises(AttributeError, match="has no 'parallel_env'"):
                spec.make_single_env()

    def test_make_env_no_entrypoint_no_parallel_env(self):
        """make_env inner closure raises on missing parallel_env."""
        from agilerl.models.env import PzEnvSpec

        spec = PzEnvSpec(name="fake_pz_module")
        mock_module = MagicMock(spec=[])
        with (
            patch("agilerl.models.env.import_module", return_value=mock_module),
            patch("agilerl.utils.utils.make_multi_agent_vect_envs") as mock_vect,
        ):
            mock_vect.side_effect = lambda env, **kw: env()
            with pytest.raises(AttributeError, match="has no 'parallel_env'"):
                spec.make_env()


class TestAlgorithmRegistry:
    def test_registry_override_warning(self):
        """warning on re-registering same algorithm name."""
        from agilerl.models.algo import ALGO_REGISTRY, AlgorithmSpec

        class _DummySpec(AlgorithmSpec):
            pass

        ALGO_REGISTRY.add("__test_dup__", _DummySpec)
        with patch("agilerl.models.algo.logger") as mock_logger:
            ALGO_REGISTRY.add("__test_dup__", _DummySpec)
            mock_logger.warning.assert_called_once()

    def test_build_algorithm_not_implemented(self):
        """AlgorithmSpec.build_algorithm raises."""
        from agilerl.models.algo import AlgorithmSpec

        spec = AlgorithmSpec.__new__(AlgorithmSpec)
        with pytest.raises(
            NotImplementedError, match="must implement a build_algorithm"
        ):
            spec.build_algorithm()

    def test_get_training_fn_not_implemented(self):
        """AlgorithmSpec.get_training_fn raises."""
        from agilerl.models.algo import AlgorithmSpec

        with pytest.raises(NotImplementedError, match="must implement get_training_fn"):
            AlgorithmSpec.get_training_fn()


class TestAlgoSpecClassVars:
    """Lines 302, 335-336, 400, 455 in algo.py."""

    def test_llm_spec_num_epochs_kwarg(self):
        """get_training_kwargs includes num_epochs for LLM."""
        from agilerl.models.algo import LLMAlgorithmSpec
        from agilerl.models.env import LLMEnvSpec
        from agilerl.models.training import TrainingSpec

        spec = LLMAlgorithmSpec.__new__(LLMAlgorithmSpec)
        object.__setattr__(
            spec,
            "__dict__",
            {
                "batch_size": 8,
                "hp_config": None,
            },
        )
        training = TrainingSpec(
            num_epochs=3, checkpoint_steps=100, evaluation_interval=10
        )
        env_spec = LLMEnvSpec(env_type="preference", dataset="dummy", max_reward=1.0)
        kwargs = spec.get_training_kwargs(training=training, env_spec=env_spec)
        assert kwargs["num_epochs"] == 3
        assert kwargs["max_reward"] == 1.0
        assert kwargs["checkpoint_steps"] == 100

    def test_llm_spec_forwards_checkpoint_path(self):
        from agilerl.models.algo import LLMAlgorithmSpec
        from agilerl.models.training import TrainingSpec

        spec = LLMAlgorithmSpec.__new__(LLMAlgorithmSpec)
        object.__setattr__(spec, "__dict__", {"batch_size": 8, "hp_config": None})
        training = TrainingSpec(checkpoint_path="/ckpts", evaluation_interval=10)
        env_spec = MagicMock()
        env_spec.max_reward = None
        kwargs = spec.get_training_kwargs(training=training, env_spec=env_spec)
        assert kwargs["checkpoint_path"] == "/ckpts"

    def test_llm_spec_warns_on_unsupported_training_fields(self):
        from agilerl.models.algo import LLMAlgorithmSpec
        from agilerl.models.training import TrainingSpec

        spec = LLMAlgorithmSpec.__new__(LLMAlgorithmSpec)
        object.__setattr__(spec, "__dict__", {"batch_size": 8, "hp_config": None})
        training = TrainingSpec(
            target_score=200.0, eval_steps=100, evaluation_interval=10
        )
        env_spec = MagicMock()
        env_spec.max_reward = None
        with pytest.warns(UserWarning, match="target_score, eval_steps"):
            kwargs = spec.get_training_kwargs(training=training, env_spec=env_spec)
        assert "target_score" not in kwargs
        assert "eval_steps" not in kwargs

    def test_llm_spec_no_warning_for_default_training_fields(self, recwarn):
        from agilerl.models.algo import LLMAlgorithmSpec
        from agilerl.models.training import TrainingSpec

        spec = LLMAlgorithmSpec.__new__(LLMAlgorithmSpec)
        object.__setattr__(spec, "__dict__", {"batch_size": 8, "hp_config": None})
        # Setting an unsupported field to its default (as the trainer does for
        # overwrite_checkpoints) must not warn
        training = TrainingSpec(evaluation_interval=10, overwrite_checkpoints=False)
        env_spec = MagicMock()
        env_spec.max_reward = None
        spec.get_training_kwargs(training=training, env_spec=env_spec)
        assert not any("not supported by LLM" in str(w.message) for w in recwarn.list)

    def test_offline_spec_dataset_path(self):
        """get_training_kwargs for offline algo with dataset_path."""
        from agilerl.models.algo import RLAlgorithmSpec, offline
        from agilerl.models.env import OfflineEnvSpec

        @offline()
        class _OffSpec(RLAlgorithmSpec):
            pass

        spec = _OffSpec()
        env_spec = OfflineEnvSpec(name="test", minari_dataset_id="cart-v0", remote=True)
        training = TrainingSpec()
        kwargs = spec.get_training_kwargs(training=training, env_spec=env_spec)
        assert kwargs["minari_dataset_id"] == "cart-v0"
        assert kwargs["remote"] is True

    def test_offline_spec_dataset_path_uses_h5py(self, tmp_path):
        """get_training_kwargs opens an offline HDF5 dataset when no Minari id."""
        from agilerl.models.algo import RLAlgorithmSpec, offline
        from agilerl.models.env import OfflineEnvSpec

        @offline()
        class _OffSpec(RLAlgorithmSpec):
            pass

        dataset_path = tmp_path / "offline.h5"
        dataset_path.write_bytes(b"")
        spec = _OffSpec()
        env_spec = OfflineEnvSpec(name="test", dataset_path=str(dataset_path))
        training = TrainingSpec()
        mock_file = MagicMock()

        with patch("h5py.File", return_value=mock_file) as mock_h5:
            kwargs = spec.get_training_kwargs(training=training, env_spec=env_spec)

        mock_h5.assert_called_once_with(str(dataset_path), "r")
        assert kwargs["dataset"] is mock_file

    def test_rl_spec_resume_from_checkpoint(self):
        """RLAlgorithmSpec.build_algorithm with resume."""
        from agilerl.models.algo import RLAlgorithmSpec

        spec = RLAlgorithmSpec(learn_step=1)
        mock_algo_cls = MagicMock()
        mock_algo = MagicMock()
        mock_algo_cls.return_value = mock_algo
        mock_algo.load_checkpoint.side_effect = lambda _p: setattr(
            mock_algo, "index", 3
        )
        with patch.object(type(spec), "algo_class", return_value=mock_algo_cls):
            spec.build_algorithm(
                observation_space=MagicMock(),
                action_space=MagicMock(),
                index=2,
                resume_from_checkpoint="/some/path",
            )
        mock_algo.load_checkpoint.assert_called_once_with("/some/path")
        # The caller's slot assignment must survive the restore
        assert mock_algo.index == 2

    def test_multi_agent_resume_from_checkpoint(self):
        """MultiAgentRLAlgorithmSpec.build_algorithm with resume."""
        from agilerl.models.algo import MultiAgentRLAlgorithmSpec

        spec = MultiAgentRLAlgorithmSpec()
        mock_algo_cls = MagicMock()
        mock_algo = MagicMock()
        mock_algo_cls.return_value = mock_algo
        mock_algo.load_checkpoint.side_effect = lambda _p: setattr(
            mock_algo, "index", 3
        )
        with patch.object(type(spec), "algo_class", return_value=mock_algo_cls):
            spec.build_algorithm(
                observation_spaces={"a": MagicMock()},
                action_spaces={"a": MagicMock()},
                index=2,
                resume_from_checkpoint="/some/path",
            )
        mock_algo.load_checkpoint.assert_called_once_with("/some/path")
        assert mock_algo.index == 2


class TestBuildAlgorithmMissingArgsRaise:
    """build_algorithm overrides reject missing required inputs."""

    def test_rl_spec_requires_spaces_and_index(self):
        from agilerl.models.algo import RLAlgorithmSpec

        spec = RLAlgorithmSpec(learn_step=1)
        with pytest.raises(ValueError, match="observation_space"):
            spec.build_algorithm()

    def test_multi_agent_spec_requires_spaces_and_index(self):
        from agilerl.models.algo import MultiAgentRLAlgorithmSpec

        spec = MultiAgentRLAlgorithmSpec()
        with pytest.raises(ValueError, match="observation_spaces"):
            spec.build_algorithm()

    def test_llm_spec_requires_tokenizer(self):
        from agilerl.models.algo import LLMAlgorithmSpec

        spec = LLMAlgorithmSpec.__new__(LLMAlgorithmSpec)
        with pytest.raises(ValueError, match="requires a tokenizer"):
            spec.build_algorithm()


class TestBuildAlgorithmForwardsOnlySetFields:
    """Unset spec fields must fall through to the algorithm's own defaults."""

    def test_rl_spec_forwards_only_set_fields(self):
        from agilerl.models.algo import RLAlgorithmSpec

        spec = RLAlgorithmSpec(learn_step=2)
        mock_algo_cls = MagicMock()
        with patch.object(type(spec), "algo_class", return_value=mock_algo_cls):
            spec.build_algorithm(
                observation_space=MagicMock(),
                action_space=MagicMock(),
                index=0,
            )
        kwargs = mock_algo_cls.call_args.kwargs
        assert kwargs["learn_step"] == 2
        assert "gamma" not in kwargs
        assert "batch_size" not in kwargs

    def test_multi_agent_spec_forwards_only_set_fields(self):
        from agilerl.models.algo import MultiAgentRLAlgorithmSpec

        spec = MultiAgentRLAlgorithmSpec(gamma=0.9)
        mock_algo_cls = MagicMock()
        with patch.object(type(spec), "algo_class", return_value=mock_algo_cls):
            spec.build_algorithm(
                observation_spaces={"a": MagicMock()},
                action_spaces={"a": MagicMock()},
                index=0,
            )
        kwargs = mock_algo_cls.call_args.kwargs
        assert kwargs["gamma"] == 0.9
        assert "learn_step" not in kwargs
        assert "batch_size" not in kwargs

    def test_llm_spec_forwards_only_set_fields(self):
        from agilerl.models.algorithms.grpo import GRPOSpec

        spec = GRPOSpec(pretrained_model_name_or_path="gpt2", beta=0.05, group_size=4)
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_algo_cls = MagicMock()
        with patch.object(type(spec), "algo_class", return_value=mock_algo_cls):
            spec.build_algorithm(tokenizer=mock_tokenizer, index=0)
        kwargs = mock_algo_cls.call_args.kwargs
        assert kwargs["beta"] == 0.05
        assert kwargs["model_name"] == "gpt2"
        # GRPO's own defaults (e.g. lr=5e-7) must apply when unset
        assert "lr" not in kwargs
        assert "batch_size" not in kwargs

    def test_spec_built_algorithm_matches_direct_construction(self):
        import gymnasium as gym

        from agilerl.algorithms import DQN
        from agilerl.models.algorithms.dqn import DQNSpec

        observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        from_spec = DQNSpec().build_algorithm(
            observation_space=observation_space,
            action_space=action_space,
            index=0,
        )
        direct = DQN(
            observation_space=observation_space,
            action_space=action_space,
        )
        assert from_spec.lr == direct.lr
        assert from_spec.batch_size == direct.batch_size
        assert from_spec.gamma == direct.gamma
        assert from_spec.learn_step == direct.learn_step


class TestLLMAlgorithmSpecBuild:
    """Lines 531-533, 554 in algo.py."""

    def test_micro_batch_size_per_gpu_forwarded_only_when_set(self):
        """Explicit micro batch reaches the constructor; unset leaves the algorithm's default."""
        from agilerl.models.algorithms.grpo import GRPOSpec

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.eos_token = "<|endoftext|>"
        accelerator = MagicMock()
        accelerator.num_processes = 2

        spec = GRPOSpec(
            pretrained_model_name_or_path="gpt2",
            group_size=4,
            batch_size=8,
            micro_batch_size_per_gpu=1,
        )
        mock_algo_cls = MagicMock()
        with patch.object(type(spec), "algo_class", return_value=mock_algo_cls):
            spec.build_algorithm(
                tokenizer=mock_tokenizer, index=0, accelerator=accelerator
            )
        assert mock_algo_cls.call_args.kwargs["micro_batch_size_per_gpu"] == 1

        # Unset: the algorithm derives it (e.g. from gradient accumulation),
        # matching direct construction.
        derived = GRPOSpec(
            pretrained_model_name_or_path="gpt2", group_size=4, batch_size=8
        )
        mock_algo_cls.reset_mock()
        with patch.object(type(derived), "algo_class", return_value=mock_algo_cls):
            derived.build_algorithm(
                tokenizer=mock_tokenizer, index=0, accelerator=accelerator
            )
        assert "micro_batch_size_per_gpu" not in mock_algo_cls.call_args.kwargs

    def test_chunk_rows_forwarded_to_constructor(self):
        """The manifest chunk_rows knob reaches the algorithm constructor."""
        from agilerl.models.algorithms.grpo import GRPOSpec

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.eos_token = "<|endoftext|>"

        spec = GRPOSpec(
            pretrained_model_name_or_path="gpt2", group_size=4, chunk_rows=128
        )
        mock_algo_cls = MagicMock()
        with patch.object(type(spec), "algo_class", return_value=mock_algo_cls):
            spec.build_algorithm(tokenizer=mock_tokenizer, index=0)
        assert mock_algo_cls.call_args.kwargs["chunk_rows"] == 128

    def test_vllm_config_dict_coerced(self):
        """build_algorithm coerces dict vllm_config."""
        from agilerl.models.algorithms.grpo import GRPOSpec

        spec = GRPOSpec(
            pretrained_model_name_or_path="gpt2",
            use_vllm=True,
            vllm_config={"tensor_parallel_size": 1},
            group_size=4,
        )
        # Force vllm_config back to a raw dict so the coercion path is hit
        object.__setattr__(spec, "vllm_config", {"tensor_parallel_size": 1})

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_algo_cls = MagicMock()
        mock_algo = MagicMock()
        mock_algo_cls.return_value = mock_algo

        with (
            patch.object(type(spec), "algo_class", return_value=mock_algo_cls),
            patch("agilerl.utils.algo_utils.VLLMConfig") as mock_vllm,
        ):
            mock_vllm.return_value = "coerced_config"
            spec.build_algorithm(tokenizer=mock_tokenizer, index=0)

        mock_vllm.assert_called_once_with(tensor_parallel_size=1)

    @pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
    def test_build_algorithm_quantization_and_attn_implementation(self):
        """build_algorithm resolves nf4 quantization and attn_implementation."""
        from agilerl.models.algorithms.grpo import GRPOSpec
        from agilerl.utils.llm_utils import build_bnb_quantization_config

        spec = GRPOSpec(
            pretrained_model_name_or_path="gpt2",
            group_size=4,
            quantization="nf4",
            attn_implementation="sdpa",
        )
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_algo_cls = MagicMock()

        with patch.object(type(spec), "algo_class", return_value=mock_algo_cls):
            spec.build_algorithm(tokenizer=mock_tokenizer, index=0)

        kwargs = mock_algo_cls.call_args.kwargs
        assert "quantization" not in kwargs
        assert kwargs["quantization_config"] == build_bnb_quantization_config("nf4")
        assert kwargs["model_config"] == {"attn_implementation": "sdpa"}

    def test_resume_from_checkpoint(self):
        """build_algorithm with resume_from_checkpoint."""
        from agilerl.models.algo import LLMAlgorithmSpec

        spec = LLMAlgorithmSpec.__new__(LLMAlgorithmSpec)
        attrs = {
            "batch_size": 4,
            "pretrained_model_name_or_path": "gpt2",
            "lora_config": None,
            "hp_config": None,
            "use_vllm": False,
            "max_model_len": 512,
            "beta": 0.01,
            "max_grad_norm": 0.1,
            "update_epochs": 1,
            "reduce_memory_peak": False,
            "use_separate_reference_adapter": False,
            "calc_position_embeddings": True,
            "gradient_checkpointing": True,
            "use_liger_loss": False,
            "seed": 42,
        }
        object.__setattr__(spec, "__dict__", attrs)
        object.__setattr__(spec, "__pydantic_fields_set__", set(attrs.keys()))

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_algo_cls = MagicMock()
        mock_algo = MagicMock()
        mock_algo_cls.return_value = mock_algo
        mock_algo.load_checkpoint.side_effect = lambda _p: setattr(
            mock_algo, "index", 3
        )

        with patch.object(type(spec), "algo_class", return_value=mock_algo_cls):
            spec.build_algorithm(
                tokenizer=mock_tokenizer,
                index=2,
                resume_from_checkpoint="/ckpt",
            )

        mock_algo.load_checkpoint.assert_called_once_with("/ckpt")
        assert mock_algo.index == 2


class TestManifestResolveAlgorithm:
    """Lines 57-58, 69-71, 74-75 in manifest.py."""

    def test_resolve_non_dict_non_spec(self):
        """_resolve_algorithm with invalid type."""
        from agilerl.models.manifest import _resolve_algorithm

        with pytest.raises(TypeError, match="Expected a dict or AlgorithmSpec"):
            _resolve_algorithm(42)

    def test_resolve_llm_without_dependencies(self):
        """LLM algo without LLM deps raises ImportError."""
        from agilerl.models.manifest import _resolve_algorithm

        with patch("agilerl.models.manifest.HAS_LLM_DEPENDENCIES", False):
            with pytest.raises(ImportError, match="LLM dependencies"):
                _resolve_algorithm({"name": "GRPO", "group_size": 4})


class TestManifestHelpers:
    """Helper resolvers and trainer-spec coercion in manifest.py."""

    def test_coerce_environment_invalid_type(self):
        from agilerl.models.manifest import _coerce_environment

        with pytest.raises(TypeError, match="Expected a dict or environment spec"):
            _coerce_environment(42)

    def test_resolve_network_finetuning_spec(self):
        from agilerl.models.manifest import _resolve_network

        network = FinetuningNetworkSpec(
            pretrained_model_name_or_path="gpt2",
            max_context_length=512,
        )
        resolved = _resolve_network(network)
        assert resolved["pretrained_model_name_or_path"] == "gpt2"
        assert resolved["max_context_length"] == 512

    @pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
    def test_network_from_algorithm_llm_agent(self):
        from agilerl.models.algorithms.grpo import GRPOSpec
        from agilerl.models.manifest import TrainingManifest

        manifest = TrainingManifest.from_trainer_specs(
            algorithm=GRPOSpec(
                group_size=4,
                pretrained_model_name_or_path="gpt2",
                max_model_len=256,
            ),
            environment=GymEnvSpec(name="dummy"),
            training=TrainingSpec(max_steps=10),
        )
        assert manifest.network is not None
        assert manifest.network["pretrained_model_name_or_path"] == "gpt2"
        assert manifest.network["max_context_length"] == 256


@requires_arena
class TestManifestFromTrainerSpecsForeignModels:
    """Foreign BaseModel inputs are dumped before manifest validation."""

    def test_from_trainer_specs_coerces_foreign_training_model(self):
        from pydantic import BaseModel

        from agilerl.arena.models.env import EnvSpec as ArenaEnvSpec

        class ForeignTraining(BaseModel):
            max_steps: int = 300
            evo_steps: int = 50
            pop_size: int = 2

        manifest = TrainingManifest.from_trainer_specs(
            algorithm=PPOSpec(learn_step=64),
            environment=ArenaEnvSpec(name="CartPole-v1"),
            training=ForeignTraining(),
        )
        assert manifest.training.max_steps == 300

    def test_resolve_foreign_arena_algorithm(self):
        from agilerl.arena.models.algorithms.ppo import PPOSpec as ArenaPPOSpec
        from agilerl.models.manifest import _resolve_algorithm

        resolved = _resolve_algorithm(ArenaPPOSpec(learn_step=48))
        assert isinstance(resolved, PPOSpec)
        assert resolved.learn_step == 48


class TestTrainingSpec:
    """Lines 247-248 in training.py."""

    def test_eval_steps_must_exceed_evaluation_interval(self):
        with pytest.raises(
            ValueError, match="eval_steps must be greater than evaluation_interval"
        ):
            TrainingSpec(eval_steps=5, evaluation_interval=10)

    def test_max_wall_seconds_accepted(self):
        assert TrainingSpec(max_wall_seconds=3600).max_wall_seconds == 3600
        assert TrainingSpec().max_wall_seconds is None
        with pytest.raises(ValueError, match="greater than"):
            TrainingSpec(max_wall_seconds=0)


class TestTournamentSelectionSpec:
    def test_tournament_size_of_one_is_valid(self):
        from agilerl.models.hpo import TournamentSelectionSpec

        assert TournamentSelectionSpec(tournament_size=1).tournament_size == 1
        with pytest.raises(ValueError, match="greater than"):
            TournamentSelectionSpec(tournament_size=0)


class TestReplayBufferSpecNStep:
    """Lines 140-146 in training.py."""

    def test_init_n_step_buffer_with_per(self):
        """init_n_step_buffer when per + n_step enabled."""
        from agilerl.models.algorithms.rainbow_dqn import RainbowDQNSpec

        spec = ReplayBufferSpec(per_buffer=True, n_step_buffer=True)
        algo = RainbowDQNSpec(
            net_config=None,
        )
        buf = spec.init_n_step_buffer(algo, device="cpu")
        assert buf is not None

    def test_init_n_step_buffer_no_gamma(self):
        """init_n_step_buffer raises when gamma missing."""
        spec = ReplayBufferSpec(per_buffer=True, n_step_buffer=True)
        algo = MagicMock(spec=[])  # no gamma attribute
        with pytest.raises(ValueError, match="Gamma must be specified"):
            spec.init_n_step_buffer(algo, device="cpu")

    def test_init_buffer_per_with_n_step_builds_per_main_memory(self):
        """Combined PER + n-step: main memory is PER, n-step is secondary."""
        from agilerl.components.replay_buffer import (
            MultiStepReplayBuffer,
            PrioritizedReplayBuffer,
        )
        from agilerl.models.algorithms.rainbow_dqn import RainbowDQNSpec

        spec = ReplayBufferSpec(per_buffer=True, n_step_buffer=True)
        algo = RainbowDQNSpec(net_config=None)
        memory = spec.init_buffer(algo, device="cpu")
        n_step_memory = spec.init_n_step_buffer(algo, device="cpu")
        assert isinstance(memory, PrioritizedReplayBuffer)
        assert isinstance(n_step_memory, MultiStepReplayBuffer)

    def test_init_buffer_n_step_only_builds_multi_step_main_memory(self):
        """n-step without PER keeps the main memory as the n-step buffer."""
        from agilerl.components.replay_buffer import MultiStepReplayBuffer
        from agilerl.models.algorithms.rainbow_dqn import RainbowDQNSpec

        spec = ReplayBufferSpec(n_step_buffer=True)
        algo = RainbowDQNSpec(net_config=None)
        memory = spec.init_buffer(algo, device="cpu")
        assert isinstance(memory, MultiStepReplayBuffer)
        assert spec.init_n_step_buffer(algo, device="cpu") is None


class TestLLMPPOSpec:
    """Lines 42-45, 56-61 in llmppo.py."""

    def test_vllm_required_when_use_vllm(self):
        from agilerl.models.algorithms.llmppo import LLMPPOSpec

        with pytest.raises(ValueError, match="VLLM config is not set"):
            LLMPPOSpec(use_vllm=True, vllm_config=None)

    def test_vllm_config_accepted_when_use_vllm(self):
        from agilerl.models.algorithms.llmppo import LLMPPOSpec
        from agilerl.utils.algo_utils import VLLMConfig

        spec = LLMPPOSpec(use_vllm=True, vllm_config=VLLMConfig())
        assert spec.use_vllm is True
        assert spec.vllm_config is not None

    def test_get_training_fn_multiturn(self):
        from agilerl.models.algorithms.llmppo import LLMPPOSpec

        fn = LLMPPOSpec.get_training_fn(multiturn=True)
        from agilerl.training.llm import finetune_llm_multiturn

        assert fn is finetune_llm_multiturn

    def test_get_training_fn_reasoning(self):
        from agilerl.models.algorithms.llmppo import LLMPPOSpec

        fn = LLMPPOSpec.get_training_fn(multiturn=False)
        from agilerl.training.llm import finetune_llm_reasoning

        assert fn is finetune_llm_reasoning


class TestLLMREINFORCESpec:
    """Lines 39-42, 53-58 in llmreinforce.py."""

    def test_vllm_required_when_use_vllm(self):
        from agilerl.models.algorithms.llmreinforce import LLMREINFORCESpec

        with pytest.raises(ValueError, match="VLLM config is not set"):
            LLMREINFORCESpec(use_vllm=True, vllm_config=None)

    def test_vllm_config_accepted_when_use_vllm(self):
        from agilerl.models.algorithms.llmreinforce import LLMREINFORCESpec
        from agilerl.utils.algo_utils import VLLMConfig

        spec = LLMREINFORCESpec(use_vllm=True, vllm_config=VLLMConfig())
        assert spec.use_vllm is True
        assert spec.vllm_config is not None

    def test_get_training_fn_multiturn(self):
        from agilerl.models.algorithms.llmreinforce import LLMREINFORCESpec

        fn = LLMREINFORCESpec.get_training_fn(multiturn=True)
        from agilerl.training.llm import finetune_llm_multiturn

        assert fn is finetune_llm_multiturn

    def test_get_training_fn_reasoning(self):
        from agilerl.models.algorithms.llmreinforce import LLMREINFORCESpec

        fn = LLMREINFORCESpec.get_training_fn(multiturn=False)
        from agilerl.training.llm import finetune_llm_reasoning

        assert fn is finetune_llm_reasoning


class TestCISPOSpec:
    """Lines 25-30 in cispo.py."""

    def test_get_training_fn_multiturn(self):
        from agilerl.models.algorithms.cispo import CISPOSpec

        fn = CISPOSpec.get_training_fn(multiturn=True)
        from agilerl.training.llm import finetune_llm_multiturn

        assert fn is finetune_llm_multiturn

    def test_get_training_fn_reasoning(self):
        from agilerl.models.algorithms.cispo import CISPOSpec

        fn = CISPOSpec.get_training_fn(multiturn=False)
        from agilerl.training.llm import finetune_llm_reasoning

        assert fn is finetune_llm_reasoning


class TestGSPOSpec:
    """Lines 25-30 in gspo.py."""

    def test_get_training_fn_multiturn(self):
        from agilerl.models.algorithms.gspo import GSPOSpec

        fn = GSPOSpec.get_training_fn(multiturn=True)
        from agilerl.training.llm import finetune_llm_multiturn

        assert fn is finetune_llm_multiturn

    def test_get_training_fn_reasoning(self):
        from agilerl.models.algorithms.gspo import GSPOSpec

        fn = GSPOSpec.get_training_fn(multiturn=False)
        from agilerl.training.llm import finetune_llm_reasoning

        assert fn is finetune_llm_reasoning


class TestRainbowDQNSpec:
    """Lines 40-41 in rainbow_dqn.py."""

    def test_v_min_gte_v_max(self):
        from agilerl.models.algorithms.rainbow_dqn import RainbowDQNSpec

        with pytest.raises(ValueError, match="v_min must be less than v_max"):
            RainbowDQNSpec(v_min=200, v_max=100)


class TestSFTSpec:
    """Lines 31-33 in sft.py."""

    def test_get_training_fn(self):
        from agilerl.models.algorithms.sft import SFTSpec

        fn = SFTSpec.get_training_fn()
        from agilerl.training.llm import finetune_llm_sft

        assert fn is finetune_llm_sft


class TestGRPOSpec:
    """Lines 39-40 in grpo.py."""

    def test_vllm_required_when_use_vllm(self):
        from agilerl.models.algorithms.grpo import GRPOSpec

        with pytest.raises(ValueError, match="VLLM config is not set"):
            GRPOSpec(group_size=4, use_vllm=True, vllm_config=None)


class TestMutationSpecExtraForbid:
    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            MutationSpec(no_mutation=0.5)

    def test_valid_nested_structure(self):
        spec = MutationSpec(probabilities={"no_mut": 0.5})
        assert spec.probabilities.no_mut == 0.5


class TestNetworkSpecUpperBound:
    def test_latent_dim_exceeds_max(self):
        with pytest.raises(ValidationError):
            NetworkSpec(
                latent_dim=200,
                max_latent_dim=128,
                encoder_config={"arch": "mlp", "hidden_size": [64]},
                head_config={"hidden_size": [64]},
            )


class TestSimbaSpecUpperBound:
    def test_num_blocks_exceeds_max(self):
        with pytest.raises(ValidationError):
            SimbaSpec(hidden_size=64, num_blocks=10, max_blocks=4)


class TestLstmSpecUpperBound:
    def test_hidden_state_size_exceeds_max(self):
        with pytest.raises(ValidationError):
            LstmSpec(hidden_state_size=512, max_hidden_state_size=256)

    def test_num_layers_exceeds_max(self):
        with pytest.raises(ValidationError):
            LstmSpec(hidden_state_size=64, num_layers=10, max_layers=6)


class TestMultiInputSpecUpperBound:
    def test_latent_dim_exceeds_max(self):
        with pytest.raises(ValidationError):
            MultiInputSpec(latent_dim=200, max_latent_dim=128)


class TestTrainingSpecValidations:
    def test_evo_steps_greater_than_max_steps_raises(self):
        with pytest.raises(ValidationError):
            TrainingSpec(max_steps=1000, evo_steps=5000)

    def test_eps_start_less_than_eps_end_raises(self):
        with pytest.raises(ValidationError):
            TrainingSpec(eps_start=0.01, eps_end=1.0)

    def test_valid_evo_steps(self):
        spec = TrainingSpec(max_steps=5000, evo_steps=1000)
        assert spec.max_steps == 5000
        assert spec.evo_steps == 1000

    def test_valid_eps(self):
        spec = TrainingSpec(eps_start=1.0, eps_end=0.01)
        assert spec.eps_start == 1.0
        assert spec.eps_end == 0.01


# Helper for Python < 3.10 that might not have contextlib.nullcontext
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield
