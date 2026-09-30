# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for demos/llm/debugging/tiny_model.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

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
