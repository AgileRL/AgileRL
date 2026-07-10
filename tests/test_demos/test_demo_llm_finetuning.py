"""Smoke tests for ``demos/llm/demo_llm_finetuning.py``.

The demo is the only in-tree consumer of the LLM training manifests, so it
silently rotted when the configs moved to the structured manifest schema. These
tests build a trainer the way the demo does -- with the model/tokenizer/env
construction stubbed out -- so a schema drift fails here instead of at runtime.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agilerl import HAS_LLM_DEPENDENCIES

pytestmark = pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = PROJECT_ROOT / "demos" / "llm" / "demo_llm_finetuning.py"


def _load_demo():
    """Import the demo by path -- ``demos/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("demo_llm_finetuning", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo():
    return _load_demo()


@pytest.fixture
def stub_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.mark.parametrize(
    ("mode", "expected_algo", "expected_objective"),
    [("sft", "SFT", "sft"), ("dpo", "DPO", "preference")],
)
def test_demo_builds_a_trainer_from_its_default_config(
    demo, stub_tokenizer, tmp_path, mode, expected_algo, expected_objective
):
    """The demo's default config for each mode drives a dataset trainer."""
    from agilerl.training.train_llm import train_llm_dataset
    from agilerl.training.trainer import LocalTrainer

    config_path = PROJECT_ROOT / demo.CONFIG_DIR / f"{mode}.yaml"
    assert config_path.exists(), f"demo default config missing: {config_path}"

    manifest = demo.build_manifest(str(config_path), None, str(tmp_path))
    assert manifest["algorithm"]["name"] == expected_algo

    with (
        patch.object(LocalTrainer, "_make_tokenizer", return_value=stub_tokenizer),
        patch.object(LocalTrainer, "_make_env", return_value=MagicMock()),
        patch(
            "agilerl.training.trainer.create_population_from_spec",
            return_value=[MagicMock()],
        ),
    ):
        trainer = LocalTrainer.from_manifest(manifest)

    assert trainer.algorithm_spec.name == expected_algo
    assert str(trainer.env_spec.env_type) == "dataset"
    assert trainer.env_spec.objective == expected_objective
    # Teacher-forced training: an instantiated env, not a rollout factory.
    assert trainer.train_fn is train_llm_dataset
    assert trainer.env_factory is None


def test_build_manifest_repoints_model_for_warm_start(demo, tmp_path):
    """``--load-path`` swaps the manifest's base model for the merged adapter."""
    config_path = PROJECT_ROOT / demo.CONFIG_DIR / "dpo.yaml"
    dest = str(tmp_path / "merged")

    with patch.object(
        demo, "_merge_adapter_into_base", return_value=dest
    ) as mock_merge:
        manifest = demo.build_manifest(str(config_path), "some/adapter", dest)

    mock_merge.assert_called_once_with("some/adapter", dest)
    assert manifest["network"]["pretrained_model_name_or_path"] == dest


def test_build_manifest_leaves_model_alone_without_warm_start(demo, tmp_path):
    """Without ``--load-path`` the manifest's own base model is used."""
    config_path = PROJECT_ROOT / demo.CONFIG_DIR / "dpo.yaml"

    with patch.object(demo, "_merge_adapter_into_base") as mock_merge:
        manifest = demo.build_manifest(str(config_path), None, str(tmp_path))

    mock_merge.assert_not_called()
    assert manifest["network"]["pretrained_model_name_or_path"] == "Qwen/Qwen2.5-0.5B"
