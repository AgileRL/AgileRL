# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

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
    demo, stub_tokenizer, mode, expected_algo, expected_objective
):
    """The demo's default config for each mode drives a dataset trainer."""
    from agilerl.training.llm import train_llm_dataset
    from agilerl.training.trainer import LocalTrainer

    config_path = PROJECT_ROOT / demo.CONFIG_DIR / f"{mode}.yaml"
    assert config_path.exists(), f"demo default config missing: {config_path}"

    manifest = demo.build_manifest(str(config_path))
    assert manifest["algorithm"]["name"] == expected_algo

    with (
        patch.object(LocalTrainer, "_make_tokenizer", return_value=stub_tokenizer),
        patch.object(LocalTrainer, "_make_env", return_value=MagicMock()),
        patch("agilerl.training.trainer.create_llm_accelerator", return_value=None),
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


def test_demo_warm_starts_with_load_weights_from(demo, stub_tokenizer):
    """``--load-path`` is a warm start, not a resume: weights only, manifest rules.

    The base model stays exactly what the manifest names -- nothing is merged into
    it -- and the checkpoint is passed as ``load_weights_from``, so the optimizer
    starts fresh and the manifest keeps its hyperparameters.
    """
    from agilerl.training.trainer import LocalTrainer

    config_path = PROJECT_ROOT / demo.CONFIG_DIR / "dpo.yaml"
    manifest = demo.build_manifest(str(config_path))
    assert manifest["network"]["pretrained_model_name_or_path"] == "Qwen/Qwen2.5-0.5B"

    with (
        patch.object(LocalTrainer, "_make_tokenizer", return_value=stub_tokenizer),
        patch.object(LocalTrainer, "_make_env", return_value=MagicMock()),
        patch("agilerl.training.trainer.create_llm_accelerator", return_value=None),
        patch("agilerl.training.trainer.create_population_from_spec") as mock_create,
    ):
        mock_create.return_value = [MagicMock()]
        LocalTrainer.from_manifest(manifest, load_weights_from="outputs/sft_run")

    kwargs = mock_create.call_args.kwargs
    assert kwargs["load_weights_from"] == "outputs/sft_run"
    assert kwargs["resume_from_checkpoint"] is None


def test_demo_never_merges_adapters_into_the_base(demo):
    """The demo must not fold LoRA weights into base weights."""
    source = DEMO_PATH.read_text()
    assert "merge_and_unload" not in source
