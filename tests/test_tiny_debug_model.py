# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for demos/llm/debugging/tiny_model.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer

LLAMA_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

TINY_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "demos"
    / "llm"
    / "debugging"
    / "tiny_model.py"
)


def _load_tiny_model():
    spec = importlib.util.spec_from_file_location(
        "tiny_model_under_test",
        TINY_MODEL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestBuildTinyHfTokenizer:
    def test_encodes_digits_to_the_same_ids_as_tiny_digit_tokenizer(self):
        tiny = _load_tiny_model()
        hf = tiny.build_tiny_hf_tokenizer()
        digits = tiny.TinyDigitTokenizer()

        assert hf.encode("3", add_special_tokens=False) == digits.encode("3")
        assert hf.encode("12", add_special_tokens=False) == digits.encode("12")
        assert hf.pad_token_id == digits.pad_token_id
        assert hf.eos_token_id == digits.eos_token_id


class TestBuildTinyActorNetwork:
    def test_writes_a_local_checkpoint_vllm_can_point_at(self):
        tiny = _load_tiny_model()

        actor = tiny.build_tiny_actor_network()
        checkpoint = Path(actor.name_or_path)

        assert checkpoint.is_dir()
        assert (checkpoint / "config.json").is_file()
        config = AutoConfig.from_pretrained(checkpoint)
        assert config.hidden_size // config.num_attention_heads == 32
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        assert tokenizer.encode("3", add_special_tokens=False) == [3]

    def test_value_head_actor_shares_the_checkpoint_path(self):
        tiny = _load_tiny_model()

        actor = tiny.build_tiny_actor_network(use_value_head=True)
        checkpoint = Path(actor.name_or_path)

        assert checkpoint.is_dir()
        assert (checkpoint / "config.json").is_file()


class TestDebugLoraTargets:
    def test_configs_and_matrix_name_llama_linears(self):
        debugging = TINY_MODEL_PATH.parent
        for path in sorted((debugging / "configs").glob("*.yaml")):
            targets = yaml.safe_load(path.read_text(encoding="utf-8"))["DEBUG"]["lora"][
                "target_modules"
            ]
            assert targets == LLAMA_LORA_TARGET_MODULES

        matrix = (debugging / "debugging_llm_training_matrix.py").read_text(
            encoding="utf-8"
        )
        assert "c_attn" not in matrix
        assert "q_proj" in matrix

    def test_peft_accepts_the_llama_targets_on_the_tiny_checkpoint(self):
        tiny = _load_tiny_model()
        actor = tiny.build_tiny_actor_network()

        wrapped = get_peft_model(
            actor,
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=LLAMA_LORA_TARGET_MODULES,
                task_type="CAUSAL_LM",
            ),
        )

        lora_names = [
            name for name, _param in wrapped.named_parameters() if "lora_" in name
        ]
        assert any("q_proj" in name for name in lora_names)
        assert any("down_proj" in name for name in lora_names)


def _load_matrix():
    debugging = TINY_MODEL_PATH.parent
    if str(debugging) not in sys.path:
        sys.path.insert(0, str(debugging))
    spec = importlib.util.spec_from_file_location(
        "debugging_llm_training_matrix_under_test",
        debugging / "debugging_llm_training_matrix.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTinyPreferenceEnv:
    def test_loss_mask_covers_non_pad_tokens_after_the_prompt(self):
        # Arrange
        matrix = _load_matrix()
        env = matrix.TinyPreferenceEnv(
            tokenizer=matrix.TinyDigitTokenizer(),
            data_batch_size_per_gpu=2,
            dataset_size=4,
        )

        # Act
        batch = env.step()

        # Assert
        prompt_lengths = [
            len(env.tokenizer.encode(prompt)) for prompt in batch["prompt"]
        ]
        for side in ("chosen", "rejected"):
            attention = batch[f"{side}_attention_mask"]
            positions = torch.arange(attention.shape[1]).unsqueeze(0)
            after_prompt = positions >= torch.tensor(prompt_lengths).unsqueeze(1)
            expected = after_prompt & attention.bool()
            assert batch[f"{side}_loss_mask"].dtype is torch.bool
            assert torch.equal(batch[f"{side}_loss_mask"], expected)
            assert batch[f"{side}_loss_mask"].shape == batch[f"{side}_input_ids"].shape
        assert "prompt_lengths" not in batch


class TestSaveTinyVllmCheckpoint:
    def test_writes_config_weights_and_tokenizer(self, tmp_path):
        tiny = _load_tiny_model()
        model = tiny.build_tiny_actor_network()

        out = tiny.save_tiny_vllm_checkpoint(model, tmp_path)

        assert out == tmp_path
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["vocab_size"] == tiny.TINY_VOCAB_SIZE
        assert (tmp_path / "tokenizer.json").is_file()
        tokenizer = AutoTokenizer.from_pretrained(tmp_path)
        assert tokenizer.encode("0", add_special_tokens=False) == [0]
