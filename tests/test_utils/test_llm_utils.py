# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("datasets", reason="LLM dependencies not installed")

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from datasets import Dataset as Datasets
from torch import nn
from transformers import AutoTokenizer

from agilerl.llm_envs import DatasetEnv
from agilerl.utils import llm_utils as llm_utils_module
from agilerl.utils.algo_utils import DummyOptimizer
from agilerl.utils.llm_utils import (
    adapt_lora_config_for_model,
    adapter_checkpoint_params,
    align_deepspeed_lr,
    apply_pad_token_id,
    attention_mask_from_padded_ids,
    baseline_free_turn_cells,
    build_bnb_quantization_config,
    build_clippable_linear_lora_target_regex,
    build_clippable_linear_lora_target_suffixes,
    build_completion_mask,
    build_scoped_lora_target_regex,
    build_vllm_llm_init_kwargs,
    build_vllm_rollout_lora_request,
    calculate_k3_kl,
    clipped_is_surrogate,
    collect_trainable_param_stats,
    compare_responses,
    create_llm_accelerator,
    create_model_from_name_or_path,
    cuda_tensor_bytes_in_module,
    discover_clippable_inner_linear_module_keys,
    discover_clippable_projection_leaf_names,
    fill_outside_mask,
    filter_peft_state_dict_for_vllm_lora,
    flex_decode_kernel_options,
    format_colocated_vllm_oom_hint,
    gather_if_ds_param,
    gather_if_zero3,
    get_llm_accelerator,
    get_model_name_or_path,
    get_state_dict,
    list_peft_matched_module_keys,
    load_pad_token_configs,
    log_cuda_memory_snapshot,
    masked_mean,
    masked_var,
    model_has_clippable_linear_wrappers,
    move_params_to_cpu,
    move_params_to_gpu,
    normalize_prompt_batch,
    offload_colocated_trainer_from_gpu,
    patch_flex_attention_kernel_options,
    peft_lora_state_dict_key_to_module_key,
    peft_target_key_matches,
    pool_by_turns,
    pool_log_ratio_by_level,
    remap_peft_lora_key_for_vllm,
    resolve_attn_implementation,
    resolve_llm_device,
    resolve_pad_token_id,
    resolve_vllm_max_lora_rank,
    resolve_vllm_max_num_batched_tokens,
    sample_eval_prompts,
    save_peft_adapter_for_vllm_rollout,
    validate_importance_sampling_level,
)
from tests import TINY_LLM_FIXTURE_PATH


class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 1000

    def batch_decode(self, *args, **kwargs):
        return ["This is a test completion."]

    def apply_chat_template(self, *args, **kwargs):
        return "This is a test completion."

    def __call__(self, *args, **kwargs):
        return torch.tensor([1, 2, 3, 4, 5])


class Info:
    def __init__(self, name):
        self.dataset_name = name


class DummyPreferenceDataset:
    def __init__(self, num_samples):
        self.prompt = [f"This is prompt {i}." for i in range(num_samples)]
        self.chosen = [f"This is chosen {i}." for i in range(num_samples)]
        self.rejected = [f"This is rejected {i}." for i in range(num_samples)]
        self.features = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
        self.info = Info("dummy_dataset")

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, index):
        return {
            "prompt": self.prompt[index],
            "chosen": self.chosen[index],
            "rejected": self.rejected[index],
        }

    def filter(self, fn):
        keep_indices = [
            i
            for i in range(len(self))
            if fn(
                {
                    "prompt": self.prompt[i],
                    "chosen": self.chosen[i],
                    "rejected": self.rejected[i],
                }
            )
        ]
        filtered = DummyPreferenceDataset(0)
        filtered.prompt = [self.prompt[i] for i in keep_indices]
        filtered.chosen = [self.chosen[i] for i in keep_indices]
        filtered.rejected = [self.rejected[i] for i in keep_indices]
        filtered.features = {
            "prompt": filtered.prompt,
            "chosen": filtered.chosen,
            "rejected": filtered.rejected,
        }
        filtered.info = self.info
        return filtered


@pytest.fixture
def accelerator_factory():
    def generate_accelerator(use_accelerator):
        AcceleratorState._reset_state(True)
        return Accelerator() if use_accelerator else None

    return generate_accelerator


@pytest.fixture
def preference_dataset(num_samples):
    train_dataset = DummyPreferenceDataset(int(num_samples * 0.8))
    test_dataset = DummyPreferenceDataset(int(num_samples * 0.2))
    return train_dataset, test_dataset


class TestDummyOptimizerInit:
    def test_dummy_optimizer_init(self):
        """Test DummyOptimizer initialization."""
        params = [torch.tensor([1.0, 2.0, 3.0])]
        lr = 0.001
        optimizer = DummyOptimizer(params, lr=lr)
        assert optimizer is not None


class TestDummyOptimizerStep:
    def test_dummy_optimizer_step(self):
        """Test DummyOptimizer step method raises RuntimeError."""
        params = [torch.tensor([1.0, 2.0, 3.0])]
        lr = 0.001
        optimizer = DummyOptimizer(params, lr=lr)

        with pytest.raises(RuntimeError) as exc_info:
            optimizer.step()

        expected_message = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        assert str(exc_info.value) == expected_message


class TestDummyOptimizerZeroGrad:
    def test_dummy_optimizer_zero_grad(self):
        """Test DummyOptimizer zero_grad method raises RuntimeError."""
        params = [torch.tensor([1.0, 2.0, 3.0])]
        lr = 0.001
        optimizer = DummyOptimizer(params, lr=lr)

        with pytest.raises(RuntimeError) as exc_info:
            optimizer.zero_grad()

        expected_message = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        assert str(exc_info.value) == expected_message


class TestDummyOptimizerStateDict:
    def test_dummy_optimizer_state_dict(self):
        """Test DummyOptimizer state_dict method raises RuntimeError."""
        params = [torch.tensor([1.0, 2.0, 3.0])]
        lr = 0.001
        optimizer = DummyOptimizer(params, lr=lr)

        with pytest.raises(RuntimeError) as exc_info:
            optimizer.state_dict()

        expected_message = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        assert str(exc_info.value) == expected_message


class TestDummyOptimizerLoadStateDict:
    def test_dummy_optimizer_load_state_dict(self):
        """Test DummyOptimizer load_state_dict method raises RuntimeError."""
        params = [torch.tensor([1.0, 2.0, 3.0])]
        lr = 0.001
        optimizer = DummyOptimizer(params, lr=lr)

        with pytest.raises(RuntimeError) as exc_info:
            optimizer.load_state_dict({})

        expected_message = (
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )
        assert str(exc_info.value) == expected_message


class TestGatherIfZero3:
    @pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
    def test_gather_if_zero3(self, zero_stage):
        """Test gather_if_zero3 context manager."""
        # ``patch("deepspeed.zero.GatheredParameters", ...)`` resolves its
        # target on ``__enter__`` (not at collection), so the patch blows up
        # for *every* zero_stage on platforms without deepspeed (Windows: see
        # ``deepspeed~=0.17.1; sys_platform != 'win32'`` in pyproject.toml),
        # not just stage 3. Skip the whole parametrized test in that case;
        # ``test_gather_if_zero3_stage_not_three_noop`` below covers the
        # deepspeed-free stages without the patch.
        pytest.importorskip("deepspeed", reason="gather_if_zero3 requires deepspeed.")
        params = [torch.tensor([1.0, 2.0, 3.0])]

        @contextmanager
        def dummy_gather_parameters(*args, **kwargs):
            yield

        with (
            patch(
                "deepspeed.zero.GatheredParameters",
                side_effect=dummy_gather_parameters,
            ) as mock_gathered_parameters,
            gather_if_zero3(zero_stage, params),
        ):
            assert mock_gathered_parameters.call_count == (zero_stage == 3)

    def test_gather_if_zero3_stage_not_three_noop(self):
        """ZeRO stages other than 3 should be a no-op context manager."""
        with gather_if_zero3(1, []):
            assert True

    def test_gather_if_zero3_stage_three_without_deepspeed_raises(self):
        """ZeRO-3 gathering requires deepspeed; raise a clear error when absent."""
        with patch("agilerl.utils.llm_utils.HAS_DEEPSPEED", False):
            with pytest.raises(ImportError, match="DeepSpeed is required for ZeRO"):
                with gather_if_zero3(3, []):
                    pass

    def test_gather_if_ds_param_noops_without_ds_id(self):
        weight = torch.randn(4, 2)
        entered = False
        with gather_if_ds_param(weight, None):
            entered = True
        assert entered

    def test_gather_if_ds_param_gathers_when_ds_id_present(self):
        pytest.importorskip(
            "deepspeed", reason="gather_if_ds_param requires deepspeed."
        )
        weight = torch.randn(4, 2)
        weight.ds_id = 0
        calls: list[list] = []

        @contextmanager
        def capture_gather(params=None, modifier_rank=None):
            calls.append(list(params))
            yield

        with patch(
            "deepspeed.zero.GatheredParameters",
            side_effect=capture_gather,
        ):
            with gather_if_ds_param(weight, None):
                pass
        assert len(calls) == 1
        assert calls[0] == [weight]

    def test_gather_if_ds_param_uses_modifier_rank_zero(self):
        """ZeRO-3 gather must pass modifier_rank=0 for reliable release."""
        pytest.importorskip(
            "deepspeed", reason="gather_if_ds_param requires deepspeed."
        )
        weight = torch.randn(4, 2)
        weight.ds_id = 0
        captured: list[int | None] = []

        @contextmanager
        def capture_gather(params=None, modifier_rank=None):
            captured.append(modifier_rank)
            yield

        with patch(
            "deepspeed.zero.GatheredParameters",
            side_effect=capture_gather,
        ):
            with gather_if_ds_param(weight):
                pass
        assert captured == [0]

    def test_gather_if_ds_param_skips_available_tied_weight(self):
        """Tied embeddings already AVAILABLE must not be re-gathered."""
        pytest.importorskip(
            "deepspeed", reason="gather_if_ds_param requires deepspeed."
        )
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        weight = torch.randn(4, 2)
        weight.ds_id = 0
        weight.ds_status = ZeroParamStatus.AVAILABLE
        calls = 0

        @contextmanager
        def capture_gather(params=None, modifier_rank=None):
            nonlocal calls
            calls += 1
            yield

        with patch(
            "deepspeed.zero.GatheredParameters",
            side_effect=capture_gather,
        ):
            with gather_if_ds_param(weight):
                pass
        assert calls == 0

    def test_gather_if_ds_param_gathers_not_available(self):
        """NOT_AVAILABLE ZeRO-3 shards still need GatheredParameters."""
        pytest.importorskip(
            "deepspeed", reason="gather_if_ds_param requires deepspeed."
        )
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        weight = torch.randn(4, 2)
        weight.ds_id = 0
        weight.ds_status = ZeroParamStatus.NOT_AVAILABLE
        calls: list[list] = []

        @contextmanager
        def capture_gather(params=None, modifier_rank=None):
            calls.append(list(params))
            yield

        with patch(
            "deepspeed.zero.GatheredParameters",
            side_effect=capture_gather,
        ):
            with gather_if_ds_param(weight):
                pass
        assert len(calls) == 1
        assert calls[0][0] is weight

    def test_gather_if_ds_param_dedupes_by_identity(self):
        """Duplicate tensor references must gather once (id-based dedupe)."""
        pytest.importorskip(
            "deepspeed", reason="gather_if_ds_param requires deepspeed."
        )
        weight = torch.randn(4, 2)
        weight.ds_id = 0
        calls: list[list] = []

        @contextmanager
        def capture_gather(params=None, modifier_rank=None):
            calls.append(list(params))
            yield

        with patch(
            "deepspeed.zero.GatheredParameters",
            side_effect=capture_gather,
        ):
            with gather_if_ds_param(weight, weight):
                pass
        assert len(calls) == 1
        assert len(calls[0]) == 1
        assert calls[0][0] is weight


def test_get_state_dict():
    # ``get_state_dict`` unconditionally wraps ``model.state_dict()`` in
    # ``gather_if_zero3(3, ...)`` (see agilerl/utils/llm_utils.py:166), which
    # requires deepspeed at runtime regardless of whether the model is actually
    # ZeRO-3-wrapped. On Windows deepspeed isn't installed (pyproject.toml:
    # ``deepspeed~=0.17.1; sys_platform != 'win32'``) and the call raises
    # ``ImportError: DeepSpeed is required for ZeRO stage 3 parameter
    # gathering``. In production the function is gated behind
    # ``HAS_LLM_DEPENDENCIES`` (only imported in agilerl/utils/utils.py when
    # the LLM extras are installed), so this codepath is never reached on
    # Windows in real usage either.
    pytest.importorskip("deepspeed", reason="get_state_dict requires deepspeed.")
    model = nn.Linear(10, 10)
    state_dict = get_state_dict(model)
    assert isinstance(state_dict, dict)
    for key, value in state_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)


def test_adapter_checkpoint_params_filters_adapter_params_only():
    """adapter_checkpoint_params returns only adapter params, never base params."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Linear(10, 10),
    )

    # Manually register LoRA-style named params by wrapping in a module
    # that uses the "lora" naming convention.
    class FakeLora(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = nn.Linear(10, 10, bias=False)
            self.lora_B = nn.Linear(10, 10, bias=False)

    wrapper = nn.ModuleDict({"base": model, "lora_adapter": FakeLora()})

    lora_params = adapter_checkpoint_params(wrapper)
    lora_names = {n for n, p in wrapper.named_parameters() if "lora" in n}
    expected_count = sum(1 for n, _ in wrapper.named_parameters() if "lora" in n)

    assert len(lora_params) == expected_count
    assert all(p is not None for p in lora_params)
    # Base params must NOT appear in the filtered set
    base_params = {p for n, p in wrapper.named_parameters() if "lora" not in n}
    for lp in lora_params:
        assert not any(lp is bp for bp in base_params)
    # Sanity: lora_names are non-empty (the test setup is valid)
    assert len(lora_names) > 0


def test_adapter_checkpoint_params_empty_model():
    """adapter_checkpoint_params on a model with no adapter params returns []."""
    model = nn.Linear(10, 10)
    assert adapter_checkpoint_params(model) == []


def test_adapter_checkpoint_params_includes_dora_magnitude():
    """DoRA magnitude vectors are written to the adapter checkpoint."""
    model = nn.ModuleDict(
        {
            "base": nn.Linear(10, 10),
            "lora_magnitude_vector": nn.ParameterDict(
                {"default": nn.Parameter(torch.ones(10))}
            ),
        }
    )

    gathered = adapter_checkpoint_params(model)

    assert len(gathered) == 1
    assert gathered[0] is model["lora_magnitude_vector"]["default"]


def test_adapter_checkpoint_params_includes_modules_to_save():
    """PEFT saves ``modules_to_save`` copies, so ZeRO-3 must gather them too."""
    value_head = nn.Linear(10, 1, bias=False)
    model = nn.ModuleDict(
        {
            "base": nn.Linear(10, 10),
            "summary": nn.ModuleDict(
                {"modules_to_save": nn.ModuleDict({"default": value_head})}
            ),
        }
    )

    gathered = adapter_checkpoint_params(model)

    assert any(p is value_head.weight for p in gathered)
    assert not any(p is model["base"].weight for p in gathered)


def test_adapter_checkpoint_params_covers_every_peft_adapter_checkpoint_key():
    """Every parameter PEFT writes to an adapter checkpoint must be gathered."""
    pytest.importorskip(
        "peft", reason="adapter_checkpoint_params covers PEFT adapter keys."
    )
    from peft import LoraConfig, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict

    class _ValueHeadModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_1 = nn.Linear(8, 8)
            self.summary = nn.Linear(8, 1)

        def forward(self, x):
            return self.summary(self.linear_1(x))

    peft_model = get_peft_model(
        _ValueHeadModel(),
        LoraConfig(r=2, target_modules=["linear_1"], modules_to_save=["summary"]),
    )

    gathered = adapter_checkpoint_params(peft_model)
    saved = get_peft_model_state_dict(peft_model)

    assert any("modules_to_save" in n for n, _ in peft_model.named_parameters())
    gathered_storage = {p.data_ptr() for p in gathered}
    assert len(gathered) == len(saved)
    for key, tensor in saved.items():
        assert tensor.data_ptr() in gathered_storage, key


def _make_tokenizer(vocab_size: int = 100, prompt_len: int = 3) -> MagicMock:
    """Return a mock tokenizer compatible with compare_responses."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    # tokenizer(text, return_tensors="pt") → encoding with .to(device)
    encoding = MagicMock()
    encoding.__getitem__ = lambda self, key: (
        torch.zeros(1, prompt_len, dtype=torch.long)
        if key == "input_ids"
        else torch.ones(1, prompt_len, dtype=torch.long)
    )
    encoding.to.return_value = {
        "input_ids": torch.zeros(1, prompt_len, dtype=torch.long),
        "attention_mask": torch.ones(1, prompt_len, dtype=torch.long),
    }
    tokenizer.return_value = encoding
    tokenizer.decode.return_value = "decoded response"
    return tokenizer


def _make_agent(has_adapter: bool, device: str = "cpu") -> MagicMock:
    """Return a mock agent with actor and device attributes."""
    agent = MagicMock()
    agent.device = device
    model = MagicMock()
    model.generate.return_value = torch.zeros(1, 5, dtype=torch.long)
    if has_adapter:
        # disable_adapter() must work as a context manager
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=None)
        cm.__exit__ = MagicMock(return_value=False)
        model.disable_adapter = MagicMock(return_value=cm)
    else:
        del model.disable_adapter  # hasattr() returns False
    agent.actor = model
    return agent


class TestCompareResponses:
    def test_compare_responses_no_adapter_with_reference(self, capsys):
        """Without an adapter only the model response section is printed; reference is shown."""
        agent = _make_agent(has_adapter=False)
        tokenizer = _make_tokenizer()
        samples = [("What is 2+2?", "It is 4.", "It is 5.")]

        compare_responses(agent, tokenizer, samples)

        captured = capsys.readouterr().out
        assert "PROMPT" in captured
        assert "DATASET RESPONSE (CHOSEN)" in captured
        assert "DATASET RESPONSE (REJECTED)" in captured
        assert "MODEL RESPONSE" in captured
        assert "BASE MODEL" not in captured
        assert "FINE-TUNED MODEL" not in captured
        # generate called exactly once (no base model pass)
        assert agent.actor.generate.call_count == 1

    def test_compare_responses_keeps_zero_pad_token_id(self, capsys):
        """A legitimate pad id of 0 is passed to generate, not swapped for eos."""
        agent = _make_agent(has_adapter=False)
        tokenizer = _make_tokenizer()
        samples = [("What is 2+2?", None, None)]

        compare_responses(agent, tokenizer, samples)

        assert agent.actor.generate.call_args.kwargs["pad_token_id"] == 0

    def test_compare_responses_no_adapter_no_reference(self, capsys):
        """When reference is None the DATASET RESPONSE section is skipped."""
        agent = _make_agent(has_adapter=False)
        tokenizer = _make_tokenizer()
        samples = [("What is 2+2?", None, None)]

        compare_responses(agent, tokenizer, samples)

        captured = capsys.readouterr().out
        assert "PROMPT" in captured
        assert "DATASET RESPONSE" not in captured
        assert "MODEL RESPONSE" in captured

    def test_compare_responses_with_adapter_shows_base_and_finetuned(self, capsys):
        """With an adapter both BASE MODEL and FINE-TUNED MODEL sections are printed."""
        agent = _make_agent(has_adapter=True)
        tokenizer = _make_tokenizer()
        samples = [
            (
                "Tell me a joke.",
                "Why did the chicken cross the road?",
                "To get to the bar.",
            ),
        ]

        compare_responses(agent, tokenizer, samples)

        captured = capsys.readouterr().out
        assert "DATASET RESPONSE (REJECTED)" in captured
        assert "BASE MODEL" in captured
        assert "FINE-TUNED MODEL" in captured
        assert "MODEL RESPONSE" not in captured
        # generate called twice: once inside disable_adapter, once without
        assert agent.actor.generate.call_count == 2
        agent.actor.disable_adapter.assert_called_once()

    def test_compare_responses_multiple_samples_enter_continues(self, capsys):
        """Pressing Enter (empty string) advances to the next sample."""
        agent = _make_agent(has_adapter=False)
        tokenizer = _make_tokenizer()
        samples = [
            ("Q1", "A1", None),
            ("Q2", "A2", "rejected for Q2"),
            ("Q3", "A3", None),
        ]

        with patch("builtins.input", return_value=""):
            compare_responses(agent, tokenizer, samples)

        captured = capsys.readouterr().out
        assert "DATASET RESPONSE (REJECTED)" in captured
        # Navigation prompt appears between samples (not after the last one)
        assert captured.count("[Enter] next sample") == len(samples) - 1
        # All three samples were generated
        assert agent.actor.generate.call_count == len(samples)

    def test_compare_responses_quit_early(self, capsys):
        """Pressing 'q' stops processing remaining samples."""
        agent = _make_agent(has_adapter=False)
        tokenizer = _make_tokenizer()
        samples = [("Q1", "A1", None), ("Q2", "A2", None), ("Q3", "A3", None)]

        with patch("builtins.input", return_value="q"):
            compare_responses(agent, tokenizer, samples)

        # Only the first sample's generation runs; loop exits before Q2 and Q3
        assert agent.actor.generate.call_count == 1

    def test_compare_responses_eof_breaks_loop(self, capsys):
        """An EOFError from input() (non-interactive environment) stops the loop gracefully."""
        agent = _make_agent(has_adapter=False)
        tokenizer = _make_tokenizer()
        samples = [("Q1", "A1", None), ("Q2", "A2", None)]

        with patch("builtins.input", side_effect=EOFError):
            compare_responses(agent, tokenizer, samples)

        # Only the first sample is generated; EOFError prevents further iteration
        assert agent.actor.generate.call_count == 1

    def test_compare_responses_single_sample_no_input_prompt(self, capsys):
        """With a single sample the navigation prompt and input() are never shown/called."""
        agent = _make_agent(has_adapter=False)
        tokenizer = _make_tokenizer()
        samples = [("Only prompt", "Only response", None)]

        with patch("builtins.input") as mock_input:
            compare_responses(agent, tokenizer, samples)

        mock_input.assert_not_called()

    @pytest.mark.parametrize(("do_sample", "temperature"), [(False, 1.0), (True, 0.7)])
    def test_compare_responses_generation_kwargs_forwarded(
        self, do_sample, temperature
    ):
        """do_sample and temperature are forwarded to model.generate."""
        agent = _make_agent(has_adapter=False)
        tokenizer = _make_tokenizer()
        samples = [("prompt", None, None)]

        compare_responses(
            agent,
            tokenizer,
            samples,
            max_new_tokens=50,
            temperature=temperature,
            do_sample=do_sample,
        )

        _, call_kwargs = agent.actor.generate.call_args
        assert call_kwargs["max_new_tokens"] == 50
        assert call_kwargs["temperature"] == temperature
        assert call_kwargs["do_sample"] == do_sample

    def test_compare_responses_skip_special_tokens_forwarded(self):
        """skip_special_tokens is forwarded to tokenizer.decode."""
        agent = _make_agent(has_adapter=False)
        tokenizer = _make_tokenizer()
        samples = [("prompt", None, None)]

        compare_responses(agent, tokenizer, samples, skip_special_tokens=False)

        _, decode_kwargs = tokenizer.decode.call_args
        assert decode_kwargs["skip_special_tokens"] is False


class TestSampleEvalPrompts:
    def test_sample_eval_prompts_sft_style_response_column(self):
        """A ``objective="sft"`` env resolves ``chosen`` from ``response_column``."""
        from types import SimpleNamespace

        ds = Datasets.from_dict(
            {"prompt": ["p0", "p1"], "response": ["r0", "r1"]},
        )
        env = SimpleNamespace(
            objective="sft",
            response_column="response",
            test_dataloader=SimpleNamespace(dataset=ds),
        )
        rows = sample_eval_prompts(env, n=2, seed=0)
        assert len(rows) == 2
        assert {r[0] for r in rows} == {"p0", "p1"}
        # chosen comes from response_column; rejected is None for SFT.
        for p, c, r in rows:
            assert c == ("r0" if p == "p0" else "r1")
            assert r is None

    def test_sample_eval_prompts_preference_style_chosen_rejected(self):
        """A ``objective="preference"`` env resolves ``chosen`` / ``rejected`` columns."""
        from types import SimpleNamespace

        ds = Datasets.from_dict(
            {
                "prompt": ["p0", "p1"],
                "chosen": ["c0", "c1"],
                "rejected": ["x0", "x1"],
            },
        )
        env = SimpleNamespace(
            objective="preference", test_dataloader=SimpleNamespace(dataset=ds)
        )
        rows = sample_eval_prompts(env, n=2, seed=0)
        assert len(rows) == 2
        prompts = {r[0] for r in rows}
        assert prompts == {"p0", "p1"}
        for p, c, r in rows:
            if p == "p0":
                assert (c, r) == ("c0", "x0")
            else:
                assert (c, r) == ("c1", "x1")


class TestDatasetEnvPreferenceInit:
    def test_preference_max_context_length_warning(self):
        train_dataset = Datasets.from_dict(
            {
                "prompt": [
                    "This is a prompt that is longer than the max context length. This prompt really is a lot longer than the other one.",
                    "This is a prompt that is shorter.",
                ],
                "chosen": ["This is an answer.", "This is an answer."],
                "rejected": ["This is an answer.", "This is an answer."],
            },
        )
        test_dataset = Datasets.from_dict(
            {
                "prompt": ["This is a normal length prompt"],
                "chosen": ["This is an answer."],
                "rejected": ["This is an answer."],
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        with pytest.warns(
            UserWarning,
            match=r"1 samples were filtered out of the train dataset due to the max context length constraint.",
        ):
            env = DatasetEnv(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                objective="preference",
                data_batch_size_per_gpu=data_batch_size,
                max_context_length=10,
                min_completion_length=1,
            )
        assert len(env.train_dataloader) == 1
        assert len(env.test_dataloader) == 1


def test_llm_utils_fallback_types_when_no_llm_dependencies():
    """Test that llm_utils sets fallback sentinels when HAS_LLM_DEPENDENCIES is False:"""
    import agilerl.utils as agilerl_utils_pkg

    # Remove the module from cache to force reimport
    original_module = sys.modules.pop("agilerl.utils.llm_utils", None)

    try:
        # Patch HAS_LLM_DEPENDENCIES before reimporting
        with patch("agilerl.HAS_LLM_DEPENDENCIES", False):
            # Reimport the module - it will see HAS_LLM_DEPENDENCIES as False
            import agilerl.utils.llm_utils as llm_utils_reloaded

            # Verify the fallback sentinels
            assert llm_utils_reloaded.PreTrainedModel is Any
            assert llm_utils_reloaded.Dataset is Any
            assert llm_utils_reloaded.AutoModelForCausalLM is None
            assert llm_utils_reloaded.AutoModelForCausalLMWithValueHead is None
            assert llm_utils_reloaded.BitsAndBytesConfig is None
    finally:
        # Restore original module to avoid affecting other tests. Both the
        # sys.modules entry AND the parent-package attribute have to be
        # restored — ``from agilerl.utils import llm_utils`` resolves through
        # the package attribute, not sys.modules, so leaving the reloaded
        # (Any-bound) module bound on ``agilerl.utils`` leaks symbols into
        # any tests that import this way on the same xdist worker.
        if original_module is not None:
            sys.modules["agilerl.utils.llm_utils"] = original_module
            agilerl_utils_pkg.llm_utils = original_module
        else:
            sys.modules.pop("agilerl.utils.llm_utils", None)


class TestCreateLlmAccelerator:
    def test_create_llm_accelerator_no_gpus_returns_none(self):
        with patch("torch.cuda.device_count", return_value=0):
            result = create_llm_accelerator()
        assert result is None

    def test_create_llm_accelerator_uses_explicit_plugin_when_provided(self):
        AcceleratorState._reset_state(True)
        explicit_plugin = MagicMock(name="explicit_plugin")
        expected_accelerator = MagicMock(spec=Accelerator)
        mock_ctor = MagicMock(return_value=expected_accelerator)
        with (
            patch("torch.cuda.device_count", return_value=1),
            patch.dict(create_llm_accelerator.__globals__, {"Accelerator": mock_ctor}),
        ):
            result = create_llm_accelerator(deepspeed_plugin=explicit_plugin)
        assert result is expected_accelerator
        mock_ctor.assert_called_once_with(deepspeed_plugin=explicit_plugin)

    def test_create_llm_accelerator_uses_launch_configured_plugin_when_available(self):
        AcceleratorState._reset_state(True)
        launch_plugin = object()
        launch_accelerator = MagicMock(spec=Accelerator)
        launch_accelerator.state = MagicMock()
        launch_accelerator.state.deepspeed_plugin = launch_plugin
        mock_ctor = MagicMock(return_value=launch_accelerator)
        with (
            patch("torch.cuda.device_count", return_value=1),
            patch.dict(create_llm_accelerator.__globals__, {"Accelerator": mock_ctor}),
        ):
            result = create_llm_accelerator()
        assert result is launch_accelerator
        mock_ctor.assert_called_once_with()

    def test_create_llm_accelerator_raises_without_explicit_or_launch_plugin(self):
        AcceleratorState._reset_state(True)
        launch_accelerator = MagicMock(spec=Accelerator)
        launch_accelerator.state = MagicMock()
        launch_accelerator.state.deepspeed_plugin = None
        mock_ctor = MagicMock(return_value=launch_accelerator)
        with (
            patch("torch.cuda.device_count", return_value=1),
            patch.dict(create_llm_accelerator.__globals__, {"Accelerator": mock_ctor}),
            pytest.raises(RuntimeError, match="DeepSpeed is required"),
        ):
            create_llm_accelerator()


class TestGetLlmAccelerator:
    def test_get_llm_accelerator_none_base_returns_none(self):
        assert get_llm_accelerator(None, idx=0) is None
        assert get_llm_accelerator(None, idx=3) is None

    def test_get_llm_accelerator_returns_base_for_first_index(self):
        base = MagicMock(spec=Accelerator)
        assert get_llm_accelerator(base, idx=0) is base

    def test_get_llm_accelerator_creates_new_plain_accelerator_for_nonzero_index(self):
        base = MagicMock(spec=Accelerator)
        base.state = MagicMock()
        base.state.deepspeed_plugin = None
        fresh = MagicMock(spec=Accelerator)
        mock_ctor = MagicMock(return_value=fresh)
        with patch.dict(get_llm_accelerator.__globals__, {"Accelerator": mock_ctor}):
            out = get_llm_accelerator(base, idx=1)
        assert out is fresh
        mock_ctor.assert_called_once_with()

    def test_get_llm_accelerator_creates_new_plain_accelerator_with_plugin_for_nonzero_index(
        self,
    ):
        base = MagicMock(spec=Accelerator)
        plugin = object()
        base.state = MagicMock()
        base.state.deepspeed_plugin = plugin
        fresh = MagicMock(spec=Accelerator)
        mock_ctor = MagicMock(return_value=fresh)
        with patch.dict(get_llm_accelerator.__globals__, {"Accelerator": mock_ctor}):
            out = get_llm_accelerator(base, idx=2)
        assert out is fresh
        mock_ctor.assert_called_once_with()

    def test_get_llm_accelerator_negative_index_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            get_llm_accelerator(None, idx=-1)


def test_normalize_prompt_batch_stacked_dict_to_per_sample_list():
    prompts = {
        "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "question": ["q0", "q1"],
        "meta": {"constant": True},
    }
    out = normalize_prompt_batch(prompts)
    assert isinstance(out, list)
    assert len(out) == 2
    assert torch.equal(out[0]["input_ids"], torch.tensor([[1, 2]], dtype=torch.long))
    assert out[1]["question"] == "q1"
    assert out[0]["meta"] == {"constant": True}


def test_masked_stats_and_pool_by_turns_helpers():
    values = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert masked_mean(values, mask) == pytest.approx(2.0)
    assert masked_var(values, mask, unbiased=False) == pytest.approx(1.0)

    token_values = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    turn_ids = torch.tensor([[0, 0, 1, -1]])
    pooled = pool_by_turns(token_values, turn_ids, num_turns=2, reduction="mean")
    assert pooled.shape == (1, 2)
    assert pooled[0, 0].item() == pytest.approx(2.0)
    assert pooled[0, 1].item() == pytest.approx(5.0)
    pooled_final = pool_by_turns(
        token_values,
        turn_ids,
        num_turns=2,
        reduction="final_value",
    )
    assert pooled_final.shape == (1, 2)
    assert pooled_final[0, 0].item() == pytest.approx(3.0)
    assert pooled_final[0, 1].item() == pytest.approx(5.0)


def test_masked_var_unbiased_requires_at_least_two_unmasked_values():
    values = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    with pytest.raises(
        ValueError, match="Unbiased masked variance requires at least 2 unmasked values"
    ):
        masked_var(values, mask, unbiased=True)


class TestRenderChatTemplate:
    def test_forwards_chat_template_kwargs_to_render(self):
        from agilerl.utils.llm_utils import render_chat_template

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<rendered>"
        out = render_chat_template(
            [{"role": "user", "content": "{question}"}],
            tokenizer,
            chat_template_kwargs={"enable_thinking": False},
            question="2+2",
        )
        assert out == "<rendered>"
        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["enable_thinking"] is False
        (messages,) = tokenizer.apply_chat_template.call_args.args
        assert messages == [{"role": "user", "content": "2+2"}]

    def test_defaults_render_without_extra_kwargs(self):
        from agilerl.utils.llm_utils import render_chat_template

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<rendered>"
        render_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenizer,
        )
        assert tokenizer.apply_chat_template.call_args.kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
        }

    def test_assistant_prefill_continues_the_final_message(self):
        from agilerl.utils.llm_utils import render_chat_template

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<rendered>"
        render_chat_template(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "Let me think:"},
            ],
            tokenizer,
        )
        assert tokenizer.apply_chat_template.call_args.kwargs == {
            "tokenize": False,
            "continue_final_message": True,
        }

    def test_empty_assistant_prefill_opens_a_fresh_turn(self):
        # continue_final_message on an empty prefill is ill-defined; the empty
        # message is dropped and a generation prompt opens the turn instead.
        from agilerl.utils.llm_utils import render_chat_template

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<rendered>"
        render_chat_template(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": ""},
            ],
            tokenizer,
        )
        assert tokenizer.apply_chat_template.call_args.kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        (messages,) = tokenizer.apply_chat_template.call_args.args
        assert messages == [{"role": "user", "content": "q"}]


class TestMaxPromptTokensForModelLen:
    def test_reserves_one_token_when_max_output_tokens_none(self):
        from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len

        assert max_prompt_tokens_for_model_len(128, None) == 127

    def test_reserves_max_output_tokens_when_set(self):
        from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len

        assert max_prompt_tokens_for_model_len(128, 32) == 96

    def test_caps_reservation_at_max_model_len(self):
        from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len

        # max_output_tokens > max_model_len → reserve max_model_len, return 0
        assert max_prompt_tokens_for_model_len(64, 256) == 0

    def test_clamps_at_zero_when_no_room(self):
        from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len

        assert max_prompt_tokens_for_model_len(0, None) == 0


class TestValidateLlmContextLengths:
    def test_skips_when_max_output_tokens_none(self):
        from agilerl.utils.llm_utils import validate_llm_context_lengths

        validate_llm_context_lengths(32768, None)

    def test_accepts_strictly_smaller_max_output_tokens(self):
        from agilerl.utils.llm_utils import validate_llm_context_lengths

        validate_llm_context_lengths(32768, 1024)

    def test_raises_when_max_output_equals_max_model_len(self):
        from agilerl.utils.llm_utils import validate_llm_context_lengths

        with pytest.raises(ValueError, match="max_output_tokens \\(32768\\)"):
            validate_llm_context_lengths(32768, 32768)

    def test_raises_when_max_output_exceeds_max_model_len(self):
        from agilerl.utils.llm_utils import validate_llm_context_lengths

        with pytest.raises(ValueError, match="max_prompt_tokens=0"):
            validate_llm_context_lengths(64, 256)


class TestNormalizeReasoningPromptBatch:
    def test_passes_list_through(self):
        from agilerl.utils.llm_utils import normalize_prompt_batch

        original = [{"input_ids": torch.tensor([1, 2])}]
        assert normalize_prompt_batch(original) is original

    def test_one_d_input_ids_treated_as_single_sample(self):
        from agilerl.utils.llm_utils import normalize_prompt_batch

        prompts = {"input_ids": torch.tensor([1, 2, 3])}
        out = normalize_prompt_batch(prompts)
        assert len(out) == 1
        assert out[0] is prompts

    def test_empty_batch_returns_empty_list(self):
        from agilerl.utils.llm_utils import normalize_prompt_batch

        prompts = {"input_ids": torch.zeros((0, 4), dtype=torch.long)}
        assert normalize_prompt_batch(prompts) == []

    def test_non_tensor_input_ids_returns_single_sample(self):
        from agilerl.utils.llm_utils import normalize_prompt_batch

        prompts = {"input_ids": "not a tensor"}
        out = normalize_prompt_batch(prompts)
        assert out == [prompts]


class TestApplyChatTemplateReexport:
    def test_apply_chat_template_is_reexported_from_llm_envs(self):
        import agilerl.utils.llm_utils as llm_utils_module
        from agilerl.llm_envs import apply_chat_template as reexport

        # Canonical home is llm_utils; llm_envs re-exports the same object.
        assert llm_utils_module.apply_chat_template is reexport

    def test_unknown_name_raises_attribute_error(self):
        import agilerl.utils.llm_utils as llm_utils_module

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = llm_utils_module._nope_definitely_not_here

    def test_dir_includes_apply_chat_template(self):
        import agilerl.utils.llm_utils as llm_utils_module

        assert "apply_chat_template" in dir(llm_utils_module)


class TestResolveLlmDevice:
    """Accelerator outranks an explicit device, which outranks auto-detection."""

    def test_accelerator_gives_the_ranks_device(self):
        accelerator = MagicMock()
        accelerator.process_index = 3
        assert resolve_llm_device(accelerator) == "cuda:3"

    def test_accelerator_outranks_an_explicit_device(self):
        # A bare "cuda" from the caller would otherwise collapse every rank
        # onto device 0.
        accelerator = MagicMock()
        accelerator.process_index = 2
        assert resolve_llm_device(accelerator, "cuda") == "cuda:2"

    def test_explicit_device_used_without_accelerator(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_llm_device(None, "cpu") == "cpu"

    def test_torch_device_is_stringified(self):
        assert resolve_llm_device(None, torch.device("cuda", 1)) == "cuda:1"

    def test_cuda_preferred_when_nothing_requested(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_llm_device(None) == "cuda"

    def test_mps_used_when_cuda_unavailable(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert resolve_llm_device(None) == "mps"

    def test_cpu_when_no_accelerator_available(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert resolve_llm_device(None) == "cpu"


def test_move_params_helpers_call_model_move_and_cuda_sync():
    model_gpu = MagicMock()
    gpu_param = MagicMock()
    gpu_param.device = torch.device("cpu")
    model_gpu.parameters.return_value = iter([gpu_param])
    with patch("torch.cuda.synchronize") as sync:
        move_params_to_gpu(model_gpu, torch.device("cuda:0"))
    model_gpu.to.assert_called_once_with(torch.device("cuda:0"), non_blocking=True)
    sync.assert_called_once()

    model_cpu = MagicMock()
    cpu_param = MagicMock()
    cpu_param.device = torch.device("cuda:0")
    model_cpu.parameters.return_value = iter([cpu_param])
    with (
        patch("torch.cuda.synchronize") as sync,
        patch("torch.cuda.empty_cache") as empty_cache,
    ):
        assert move_params_to_cpu(model_cpu) is True
    model_cpu.to.assert_called_once_with("cpu", non_blocking=True)
    sync.assert_called_once()
    empty_cache.assert_called_once()


def test_move_params_to_cpu_skips_when_already_on_cpu():
    """No move, sync, or empty_cache when the model already lives on CPU."""
    model = MagicMock()
    param = MagicMock()
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])
    with (
        patch("torch.cuda.synchronize") as sync,
        patch("torch.cuda.empty_cache") as empty_cache,
    ):
        assert move_params_to_cpu(model) is False
    model.to.assert_not_called()
    sync.assert_not_called()
    empty_cache.assert_not_called()


def test_get_model_name_or_path_and_align_deepspeed_lr_helpers():
    class _DirectModel:
        name_or_path = "direct_name"

    model = _DirectModel()
    assert get_model_name_or_path(model) == "direct_name"

    class _Inner:
        name_or_path = "inner_name"

    class _Nested:
        pretrained_model = _Inner()

    nested = _Nested()
    assert get_model_name_or_path(nested) == "inner_name"

    class _Missing:
        pass

    missing = _Missing()
    with pytest.raises(ValueError, match="Model name or path not found"):
        get_model_name_or_path(missing)

    accelerator = MagicMock()
    accelerator.state.deepspeed_plugin.deepspeed_config = {
        "optimizer": {"params": {"lr": 1e-3}}
    }
    with pytest.warns(UserWarning, match="DeepSpeed learning rate is set to"):
        out = align_deepspeed_lr(2e-3, accelerator)
    assert out == pytest.approx(2e-3)
    assert accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"][
        "lr"
    ] == pytest.approx(2e-3)


def test_k3_helper_matches_torch() -> None:
    """K3 estimator helper is the same formula Liger ships."""
    torch.manual_seed(5)
    log_p = torch.randn(3, 4) * 0.1
    log_q = torch.randn(3, 4) * 0.1
    # Reference: torch implementation of the same formula.
    ref = torch.exp(log_p - log_q) - (log_p - log_q) - 1.0
    assert torch.allclose(calculate_k3_kl(log_p, log_q), ref)


class TestFillOutsideMask:
    """:func:`fill_outside_mask` keeps masked reductions finite when padding
    slots hold NaN/Inf, which ``values * mask`` cannot do (``nan * 0 == nan``).
    """

    def test_non_finite_padding_no_longer_poisons_masked_sum(self) -> None:
        values = torch.tensor([[1.0, float("nan")], [2.0, float("inf")]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        assert torch.isnan(values * mask).any()

        filled = fill_outside_mask(values, mask)

        assert filled.tolist() == [[1.0, 0.0], [2.0, 0.0]]
        assert torch.isfinite((filled * mask).sum())

    def test_in_mask_non_finite_values_are_preserved(self) -> None:
        values = torch.tensor([[float("nan"), 1.0]])
        mask = torch.tensor([[True, True]])

        filled = fill_outside_mask(values, mask)

        assert torch.isnan(filled[0, 0])

    def test_broadcasts_a_narrower_mask(self) -> None:
        values = torch.full((2, 3, 4), float("nan"))
        mask = torch.zeros(2, 3, 1, dtype=torch.bool)

        filled = fill_outside_mask(values, mask)

        assert filled.shape == (2, 3, 4)
        assert torch.equal(filled, torch.zeros(2, 3, 4))

    def test_custom_fill_value(self) -> None:
        values = torch.tensor([[float("nan"), 5.0]])
        mask = torch.tensor([[0.0, 1.0]])

        assert fill_outside_mask(values, mask, -1.0).tolist() == [[-1.0, 5.0]]

    def test_gradient_does_not_flow_through_filled_positions(self) -> None:
        values = torch.tensor([[1.0, 2.0]], requires_grad=True)
        mask = torch.tensor([[1.0, 0.0]])

        fill_outside_mask(values, mask).sum().backward()

        assert values.grad.tolist() == [[1.0, 0.0]]


class TestMaskedMeanAxis:
    """Cover the axis-reduction branch in :func:`masked_mean`."""

    def test_per_row_axis_reduction(self) -> None:
        values = torch.tensor([[1.0, 3.0], [4.0, 8.0]])
        mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        result = masked_mean(values, mask, axis=1)
        assert result.shape == (2,)
        assert result.tolist() == pytest.approx([2.0, 8.0])


class TestMaskedWhitenShiftMean:
    """Cover the ``shift_mean=False`` branch in :func:`masked_whiten`."""

    def test_shift_mean_false_adds_mean_back(self) -> None:
        from agilerl.utils.llm_utils import masked_whiten

        values = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        whitened_no_shift = masked_whiten(values, mask, shift_mean=False)
        whitened_shift = masked_whiten(values, mask, shift_mean=True)
        # The two outputs differ by exactly the masked mean (=4.0).
        diff = whitened_no_shift - whitened_shift
        assert torch.allclose(diff, torch.full_like(diff, 4.0), atol=1e-5)


class TestBaselineFreeTurnCells:
    """``baseline_free_turn_cells`` finds played cells with no group baseline."""

    def test_group_counts_expand_back_to_every_member(self) -> None:
        turn_mask = torch.tensor(
            [
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [True, True, True],
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, False, False],
            ],
        )

        sparse = baseline_free_turn_cells(turn_mask, 4)

        expected = torch.tensor(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, True, True],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
        )
        assert torch.equal(sparse, expected)

    def test_unplayed_cells_are_never_marked(self) -> None:
        turn_mask = torch.zeros(4, 6, dtype=torch.bool)
        turn_mask[3, :4] = True

        sparse = baseline_free_turn_cells(turn_mask, 4)

        assert not sparse[~turn_mask].any()
        assert torch.equal(sparse, turn_mask)


class TestPoolByTurnsBadReduction:
    """``pool_by_turns`` should raise a clear ValueError for unknown reductions."""

    def test_unknown_reduction_raises(self) -> None:
        token_values = torch.tensor([[1.0, 2.0]])
        turn_ids = torch.tensor([[0, 0]])
        with pytest.raises(ValueError, match="Invalid reduction: unsupported"):
            pool_by_turns(token_values, turn_ids, num_turns=1, reduction="unsupported")


class TestPoolLogRatioByLevelBadReduction:
    """``pool_log_ratio_by_level`` rejects unknown turn reductions at the turn level."""

    def test_unknown_turn_reduction_raises(self) -> None:
        token_log_ratio = torch.zeros(1, 2)
        action_mask = torch.ones(1, 2)
        turn_ids = torch.tensor([[0, 0]])
        with pytest.raises(ValueError, match=r"turn_reduction must be one of"):
            pool_log_ratio_by_level(
                token_log_ratio,
                action_mask,
                turn_ids,
                level="turn",
                turn_reduction="unsupported",
            )


class TestClippedIsSurrogate:
    """The shared token/turn/sequence clipped surrogate used by the non-Liger
    PPO and REINFORCE paths. token/turn/sequence are points on one
    ratio-pooling axis, so the level collapses cleanly at the limits.
    """

    @staticmethod
    def _setup():
        torch.manual_seed(0)
        B, T = 3, 6
        token_log_ratio = torch.randn(B, T)
        advantages = torch.randn(B, T)
        mask = torch.ones(B, T)
        mask[0, 5:] = 0  # ragged sequence
        return token_log_ratio, advantages, mask, B, T

    def test_turn_each_token_own_turn_equals_token(self):
        tlr, adv, mask, _B, T = self._setup()
        turn_each = torch.where(
            mask.bool(),
            torch.arange(T).unsqueeze(0).expand_as(mask).long(),
            torch.full_like(mask, -1, dtype=torch.long),
        )
        pg_tok, cf_tok = clipped_is_surrogate(tlr, adv, mask, turn_each, "token", 0.2)
        pg_turn, cf_turn = clipped_is_surrogate(tlr, adv, mask, turn_each, "turn", 0.2)
        assert torch.allclose(pg_tok, pg_turn, atol=1e-5)
        assert torch.allclose(cf_tok, cf_turn, atol=1e-5)

    def test_single_turn_equals_sequence(self):
        tlr, adv, mask, B, T = self._setup()
        turn_one = torch.where(
            mask.bool(),
            torch.zeros(B, T, dtype=torch.long),
            torch.full((B, T), -1, dtype=torch.long),
        )
        pg_turn, _ = clipped_is_surrogate(tlr, adv, mask, turn_one, "turn", 0.2)
        pg_seq, _ = clipped_is_surrogate(tlr, adv, mask, turn_one, "trajectory", 0.2)
        assert torch.allclose(pg_turn, pg_seq, atol=1e-5)

    def test_sequence_pools_ratio_and_advantage(self):
        """Trajectory level uses the length-normalized mean log-ratio and the
        mean advantage over a completion's action tokens.
        """
        tlr = torch.tensor([[0.2, 0.4, 1.0, -5.0]])
        adv = torch.tensor([[1.0, 3.0, 2.0, 99.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # last token non-action
        pg, _ = clipped_is_surrogate(tlr, adv, mask, None, "trajectory", 0.2)
        mean_lr = (0.2 + 0.4 + 1.0) / 3
        mean_adv = (1.0 + 3.0 + 2.0) / 3
        ratio = torch.exp(torch.tensor(mean_lr))
        clipped = torch.clamp(ratio, 0.8, 1.2)
        expected = torch.max(-mean_adv * ratio, -mean_adv * clipped)
        assert torch.allclose(pg, expected, atol=1e-5)

    def test_turn_sum_reduction_pools_ratio_by_product_but_advantage_by_mean(self):
        """With ``turn_reduction="sum"`` the turn ratio is the product of token
        ratios (nightly/Turn-PPO), while the (broadcast) advantage is still
        recovered by the turn mean — i.e. it is *not* rescaled by turn length.
        """
        # One sample, one turn, two action tokens; advantage is the per-turn
        # value broadcast across both tokens (as the GAE code produces).
        tlr = torch.tensor([[0.1, 0.2]])
        adv = torch.tensor([[2.0, 2.0]])
        mask = torch.ones(1, 2)
        turn_ids = torch.zeros(1, 2, dtype=torch.long)
        # Wide clip so the ratio is never clamped, isolating the pooling logic.
        pg, _ = clipped_is_surrogate(
            tlr, adv, mask, turn_ids, "turn", 10.0, turn_reduction="sum"
        )
        ratio = torch.exp(torch.tensor(0.1 + 0.2))  # product ratio
        expected = -2.0 * ratio  # mean-pooled advantage, NOT 4.0
        assert torch.allclose(pg, expected, atol=1e-5)

    def test_turn_requires_turn_ids(self):
        tlr, adv, mask, _B, _T = self._setup()
        with pytest.raises(ValueError, match="turn-level surrogate requires turn_ids"):
            clipped_is_surrogate(tlr, adv, mask, None, "turn", 0.2)

    def test_unknown_level_raises(self):
        tlr, adv, mask, _B, _T = self._setup()
        with pytest.raises(ValueError, match="Unknown importance_sampling_level"):
            clipped_is_surrogate(tlr, adv, mask, None, "bogus", 0.2)

    def test_gradient_flows_to_log_ratio(self):
        """All levels are differentiable w.r.t. the token log-ratio."""
        for level, turn_ids in (
            ("token", None),
            ("trajectory", None),
            ("turn", torch.zeros(3, 6, dtype=torch.long)),
        ):
            tlr, adv, mask, _B, _T = self._setup()
            tlr = tlr.clone().requires_grad_(True)
            pg, _ = clipped_is_surrogate(tlr, adv, mask, turn_ids, level, 0.2)
            pg.backward()
            assert tlr.grad is not None
            assert torch.isfinite(tlr.grad).all()

    def test_loss_weight_scales_per_unit_surrogate(self):
        """A detached per-token loss weight reweights the surrogate; uniform
        weights of 1 are a no-op and uniform weights of w scale pg_loss by w.
        """
        tlr, adv, mask, _B, _T = self._setup()
        pg_base, cf_base = clipped_is_surrogate(tlr, adv, mask, None, "token", 0.2)
        pg_ones, cf_ones = clipped_is_surrogate(
            tlr, adv, mask, None, "token", 0.2, loss_weight=torch.ones_like(tlr)
        )
        assert torch.allclose(pg_base, pg_ones, atol=1e-6)
        assert torch.allclose(cf_base, cf_ones, atol=1e-6)
        pg_half, _ = clipped_is_surrogate(
            tlr, adv, mask, None, "token", 0.2, loss_weight=torch.full_like(tlr, 0.5)
        )
        assert torch.allclose(pg_half, 0.5 * pg_base, atol=1e-6)


class TestCreateModelFromNameOrPathValueHead:
    """``create_model_from_name_or_path`` should route through ``AutoModelForCausalLMWithValueHead``
    when a value head is requested.
    """

    def test_add_value_head_calls_value_head_loader(self) -> None:
        from agilerl.utils import llm_utils as llm_utils_module

        # When agilerl[llm] isn't installed (e.g. Windows CI without vllm),
        # the symbol collapses to ``typing.Any``. Inspect the bound symbol
        # directly rather than the ``HAS_LLM_DEPENDENCIES`` flag — another
        # test in this module force-reloads ``llm_utils`` with
        # ``HAS_LLM_DEPENDENCIES=False`` and the package attribute restore
        # can leak through xdist worker reuse.
        if llm_utils_module.AutoModelForCausalLMWithValueHead is Any:
            pytest.skip(
                "AutoModelForCausalLMWithValueHead unavailable without agilerl[llm]."
            )

        sentinel_model = object()
        with patch.object(
            llm_utils_module.AutoModelForCausalLMWithValueHead,
            "from_pretrained",
            return_value=sentinel_model,
        ) as mock_loader:
            out = llm_utils_module.create_model_from_name_or_path(
                "some/model",
                add_value_head=True,
                use_accelerator=False,
            )
        assert out is sentinel_model
        # Default model_config is built when not supplied; verify the loader saw
        # the model name plus the synthesized dtype/attn keys.
        call_kwargs = mock_loader.call_args.kwargs
        assert call_kwargs["pretrained_model_name_or_path"] == "some/model"
        assert call_kwargs["attn_implementation"] == "sdpa"


class TestGetModelNameOrPathBaseModelBranches:
    """``get_model_name_or_path`` falls through several optional attribute
    chains: ``base_model.name_or_path`` then ``base_model.pretrained_model.name_or_path``.
    These branches don't fire when ``name_or_path`` exists at the top level.
    """

    def test_base_model_name_or_path(self) -> None:
        class _Base:
            name_or_path = "via_base"

        class _Wrapper:
            base_model = _Base()

        wrapper = _Wrapper()
        # Sanity: no top-level ``name_or_path`` and no ``pretrained_model``.
        assert not hasattr(wrapper, "name_or_path")
        assert not hasattr(wrapper, "pretrained_model")
        assert get_model_name_or_path(wrapper) == "via_base"

    def test_base_model_pretrained_model_name_or_path(self) -> None:
        class _PretrainedInBase:
            name_or_path = "via_base_pretrained"

        class _Base:
            pretrained_model = _PretrainedInBase()
            # Intentionally no ``name_or_path`` here so the helper has to fall
            # through to ``base_model.pretrained_model.name_or_path``.

        class _Wrapper:
            base_model = _Base()

        wrapper = _Wrapper()
        assert get_model_name_or_path(wrapper) == "via_base_pretrained"


GiB = 1024**3


# ---------------------------------------------------------------------------
# Fixture models for the ClippableLinear / LoRA-targeting helpers
# ---------------------------------------------------------------------------


class _FakeClippableLinear(nn.Module):
    """Stand-in for Gemma's *ClippableLinear projection wrappers.

    Only the class-name suffix matters to the helpers under test.
    """

    def __init__(self, inner: nn.Module | None = None):
        super().__init__()
        self.linear = nn.Linear(4, 4) if inner is None else inner


class _ClippableTower(nn.Module):
    """Vision-tower-like container of *ClippableLinear projection wrappers."""

    def __init__(self):
        super().__init__()
        self.q_proj = _FakeClippableLinear()
        self.k_proj = _FakeClippableLinear()


class _RootClippableModel(nn.Module):
    """ClippableLinear wrappers outside any language_model/audio_tower scope."""

    def __init__(self):
        super().__init__()
        self.tower = _ClippableTower()


class _ScopedClippableModel(nn.Module):
    """ClippableLinear wrappers nested under ``model.language_model``."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _ClippableTower()


class _AudioScopedClippableModel(nn.Module):
    """ClippableLinear wrappers nested under ``model.audio_tower``."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.audio_tower = _ClippableTower()


class _LanguageScopedLinearModel(nn.Module):
    """Plain ``nn.Linear`` projections under a nested ``language_model`` scope."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.q_proj = nn.Linear(4, 4)
        self.model.language_model.mlp = nn.Linear(4, 4)


class _PlainLinearModel(nn.Module):
    """No ClippableLinear wrappers at all."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)


class _PlainLoraConfig:
    """Minimal LoraConfig stand-in without ``to_dict`` (deepcopy clone path)."""

    def __init__(self, target_modules, exclude_modules=None):
        self.target_modules = target_modules
        self.exclude_modules = exclude_modules


class _DictLoraConfig:
    """LoraConfig stand-in with ``to_dict`` (reconstruction clone path)."""

    def __init__(self, target_modules=None, exclude_modules=None, r=8):
        self.target_modules = target_modules
        self.exclude_modules = exclude_modules
        self.r = r

    def to_dict(self):
        return {
            "target_modules": self.target_modules,
            "exclude_modules": self.exclude_modules,
            "r": self.r,
        }


class TestBuildBnbQuantizationConfig:
    """YAML-friendly QUANTIZATION spec -> BitsAndBytesConfig resolution."""

    class _FakeBnbConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    @pytest.fixture(autouse=True)
    def _fake_bnb(self, monkeypatch):
        monkeypatch.setattr(llm_utils_module, "HAS_LLM_DEPENDENCIES", True)
        monkeypatch.setattr(llm_utils_module, "BitsAndBytesConfig", self._FakeBnbConfig)

    def test_none_spec_returns_none(self):
        assert build_bnb_quantization_config(None) is None

    @pytest.mark.parametrize("spec", ["none", "  NONE  "])
    def test_none_preset_strings_return_none(self, spec):
        assert build_bnb_quantization_config(spec) is None

    @pytest.mark.parametrize("spec", ["", "null", "false", "0", "bitsandbytes"])
    def test_dropped_aliases_raise_value_error(self, spec):
        with pytest.raises(ValueError, match=r"Unknown quantization preset"):
            build_bnb_quantization_config(spec)

    def test_int8_preset(self):
        out = build_bnb_quantization_config("int8")
        assert out.kwargs == {"load_in_8bit": True}

    def test_nf4_preset_matches_qlora_recipe(self):
        out = build_bnb_quantization_config("nf4")
        assert out.kwargs == {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_quant_storage": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
        }

    @pytest.mark.parametrize(
        "alias",
        [
            "nf4",
            "4bit",
            "4-bit",
            "4 bit",
            "bnb-4bit",
            "bnb_4bit",
            "BNB_4BIT",
        ],
    )
    def test_every_4bit_alias_builds_the_same_config(self, alias):
        assert (
            build_bnb_quantization_config(alias).kwargs
            == build_bnb_quantization_config("nf4").kwargs
        )

    @pytest.mark.parametrize("alias", ["nf4", "4bit", "bnb_4bit", "bnb-4bit"])
    def test_every_4bit_alias_sets_bf16_quant_storage(self, alias):
        kwargs = build_bnb_quantization_config(alias).kwargs
        assert kwargs["bnb_4bit_quant_storage"] is torch.bfloat16

    def test_dict_spec_forwarded_verbatim(self):
        spec = {"load_in_8bit": True, "llm_int8_threshold": 5.0}
        out = build_bnb_quantization_config(spec)
        assert out.kwargs == spec

    def test_existing_config_instance_passthrough(self):
        spec = self._FakeBnbConfig(load_in_4bit=True)
        assert build_bnb_quantization_config(spec) is spec

    def test_unknown_preset_raises_value_error(self):
        with pytest.raises(
            ValueError, match=r"Unknown quantization preset 'fp4'"
        ) as exc_info:
            build_bnb_quantization_config("fp4")
        message = str(exc_info.value)
        for alias in ("int8", "nf4", "4bit", "bnb_4bit", "none"):
            assert repr(alias) in message

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="got int"):
            build_bnb_quantization_config(42)

    def test_missing_llm_dependencies_raises_import_error(self, monkeypatch):
        monkeypatch.setattr(llm_utils_module, "HAS_LLM_DEPENDENCIES", False)
        with pytest.raises(ImportError, match=r"install agilerl\[llm\]"):
            build_bnb_quantization_config("int8")

    def test_none_spec_skips_dependency_check(self, monkeypatch):
        monkeypatch.setattr(llm_utils_module, "HAS_LLM_DEPENDENCIES", False)
        assert build_bnb_quantization_config(None) is None


class TestClippableLinearDiscovery:
    def test_model_has_clippable_linear_wrappers(self):
        assert model_has_clippable_linear_wrappers(_RootClippableModel())
        assert not model_has_clippable_linear_wrappers(_PlainLinearModel())

    def test_discover_projection_leaf_names_sorted(self):
        assert discover_clippable_projection_leaf_names(_RootClippableModel()) == [
            "k_proj",
            "q_proj",
        ]
        assert discover_clippable_projection_leaf_names(_PlainLinearModel()) == []

    def test_discover_inner_linear_module_keys(self):
        keys = discover_clippable_inner_linear_module_keys(_RootClippableModel())
        assert keys == ["tower.k_proj.linear", "tower.q_proj.linear"]

    def test_discover_inner_keys_skips_non_adaptable_inner_modules(self):
        model = nn.Module()
        model.q_proj = _FakeClippableLinear(inner=nn.SiLU())
        assert discover_clippable_inner_linear_module_keys(model) == []

    def test_discover_inner_keys_accepts_bnb_quantized_linear_names(self):
        class Linear4bit(nn.Module):
            """Class name mimics the bitsandbytes 4-bit linear layer."""

        model = nn.Module()
        model.q_proj = _FakeClippableLinear(inner=Linear4bit())
        assert discover_clippable_inner_linear_module_keys(model) == ["q_proj.linear"]

    def test_is_peft_adaptable_linear(self):
        class Linear8bitLt(nn.Module):
            """Class name mimics the bitsandbytes 8-bit linear layer."""

        assert llm_utils_module._is_peft_adaptable_linear(nn.Linear(2, 2))
        assert llm_utils_module._is_peft_adaptable_linear(Linear8bitLt())
        assert not llm_utils_module._is_peft_adaptable_linear(nn.SiLU())


class TestPeftTargetKeyMatching:
    def test_regex_spec_uses_fullmatch(self):
        spec = r"(?:.*\.)?(q_proj)\.linear"
        assert peft_target_key_matches("model.layers.0.q_proj.linear", spec)
        assert peft_target_key_matches("q_proj.linear", spec)
        assert not peft_target_key_matches("model.layers.0.q_proj", spec)

    def test_list_spec_exact_and_suffix_match(self):
        assert peft_target_key_matches("q_proj", ["q_proj"])
        assert peft_target_key_matches("model.layers.0.q_proj", ["q_proj"])
        assert not peft_target_key_matches("model.layers.0.q_proj_extra", ["q_proj"])

    def test_list_matched_module_keys_respects_exclusions(self):
        model = _RootClippableModel()
        targets = ["q_proj.linear", "k_proj.linear"]
        assert list_peft_matched_module_keys(model, targets) == [
            "tower.q_proj.linear",
            "tower.k_proj.linear",
        ]
        assert list_peft_matched_module_keys(
            model, targets, exclude_modules=["k_proj"]
        ) == ["tower.q_proj.linear"]


class TestLoraTargetRegexBuilders:
    def test_clippable_regex_targets_inner_linear(self):
        regex = build_clippable_linear_lora_target_regex(["q_proj", "k_proj", "q_proj"])
        assert regex == r"(?:.*\.)?(k_proj|q_proj)\.linear"
        assert re.fullmatch(regex, "model.layers.0.q_proj.linear")
        assert re.fullmatch(regex, "k_proj.linear")
        assert not re.fullmatch(regex, "model.layers.0.q_proj")

    def test_clippable_regex_requires_projection_names(self):
        with pytest.raises(ValueError, match="At least one projection name"):
            build_clippable_linear_lora_target_regex([])

    def test_scoped_regex_matches_plain_and_wrapped_projections(self):
        regex = build_scoped_lora_target_regex(["q_proj"], "language_model")
        assert re.fullmatch(regex, "model.language_model.layers.0.q_proj")
        assert re.fullmatch(regex, "model.language_model.layers.0.q_proj.linear")
        assert not re.fullmatch(regex, "model.vision_tower.layers.0.q_proj")

    def test_scoped_regex_requires_projection_names(self):
        with pytest.raises(ValueError, match="At least one projection name"):
            build_scoped_lora_target_regex([], "language_model")

    def test_suffix_targets_sorted_and_deduped(self):
        assert build_clippable_linear_lora_target_suffixes(
            ["q_proj", "k_proj", "q_proj"]
        ) == ["k_proj.linear", "q_proj.linear"]


class TestProjectionNameResolution:
    def test_regex_spec_short_circuits_to_none(self):
        assert (
            llm_utils_module._projection_names_for_clippable_lora(
                _PlainLinearModel(), r".*\.q_proj"
            )
            is None
        )

    def test_none_targets_returns_none(self):
        assert (
            llm_utils_module._projection_names_for_clippable_lora(
                _PlainLinearModel(), None
            )
            is None
        )

    def test_looks_like_peft_target_regex_heuristic(self):
        looks = llm_utils_module._looks_like_peft_target_regex
        assert looks(".*foo")
        assert looks(r"foo\.bar")
        assert looks("foo(bar)")
        assert not looks("q_proj")
        assert not looks("q_proj.linear")

    def test_all_linear_expands_to_discovered_wrapper_names(self):
        names = llm_utils_module._projection_names_for_clippable_lora(
            _RootClippableModel(), "all-linear"
        )
        assert names == ["k_proj", "q_proj"]

    def test_explicit_names_normalized_sorted_deduped(self):
        names = llm_utils_module._projection_names_for_clippable_lora(
            _PlainLinearModel(), ["v_proj.linear", "q_proj", "q_proj"]
        )
        assert names == ["q_proj", "v_proj"]


class TestInferClippableLoraScope:
    def test_language_model_scope_inferred(self):
        assert (
            llm_utils_module._infer_clippable_lora_scope(_ScopedClippableModel())
            == "language_model"
        )

    def test_audio_tower_scope_inferred(self):
        assert (
            llm_utils_module._infer_clippable_lora_scope(_AudioScopedClippableModel())
            == "audio_tower"
        )

    def test_unscoped_wrappers_infer_none(self):
        assert (
            llm_utils_module._infer_clippable_lora_scope(_RootClippableModel()) is None
        )

    def test_no_wrappers_infer_none(self):
        assert llm_utils_module._infer_clippable_lora_scope(_PlainLinearModel()) is None


class TestExampleModuleKeysForLoraScope:
    def test_returns_keys_under_scope_only(self):
        model = _LanguageScopedLinearModel()
        keys = llm_utils_module._example_module_keys_for_lora_scope(
            model, "language_model"
        )
        assert keys == [
            "model.language_model.q_proj",
            "model.language_model.mlp",
        ]

    def test_limit_truncates_examples(self):
        model = _LanguageScopedLinearModel()
        keys = llm_utils_module._example_module_keys_for_lora_scope(
            model, "language_model", limit=1
        )
        assert keys == ["model.language_model.q_proj"]


class TestAdaptLoraConfigForModel:
    def test_regex_targets_returned_unchanged(self):
        cfg = _PlainLoraConfig(target_modules=r".*\.q_proj")
        assert adapt_lora_config_for_model(_RootClippableModel(), cfg) is cfg

    def test_plain_model_short_names_returned_unchanged(self):
        cfg = _PlainLoraConfig(target_modules=["q_proj"])
        assert adapt_lora_config_for_model(_PlainLinearModel(), cfg) is cfg

    def test_explicit_scope_rewrites_to_scoped_regex(self):
        model = _LanguageScopedLinearModel()
        cfg = _PlainLoraConfig(target_modules=["q_proj"])
        adapted = adapt_lora_config_for_model(
            model, cfg, lora_target_scope="language_model"
        )
        expected = build_scoped_lora_target_regex(["q_proj"], "language_model")
        assert adapted is not cfg
        assert adapted.target_modules == expected
        # Original config untouched (deepcopy clone path).
        assert cfg.target_modules == ["q_proj"]
        assert re.fullmatch(expected, "model.language_model.q_proj")

    def test_explicit_scope_with_to_dict_config_reconstructs_class(self):
        model = _LanguageScopedLinearModel()
        cfg = _DictLoraConfig(target_modules=["q_proj"], r=16)
        adapted = adapt_lora_config_for_model(
            model, cfg, lora_target_scope="language_model"
        )
        assert isinstance(adapted, _DictLoraConfig)
        assert adapted is not cfg
        assert adapted.r == 16
        assert adapted.target_modules == build_scoped_lora_target_regex(
            ["q_proj"], "language_model"
        )

    def test_explicit_scope_without_matches_raises_with_examples(self):
        model = _LanguageScopedLinearModel()
        cfg = _PlainLoraConfig(target_modules=["v_proj"])
        with pytest.raises(
            ValueError, match="No modules matched scoped LoRA target_modules"
        ) as exc_info:
            adapt_lora_config_for_model(model, cfg, lora_target_scope="language_model")
        assert "Example keys under scope" in str(exc_info.value)
        assert "model.language_model.q_proj" in str(exc_info.value)

    def test_explicit_scope_with_empty_targets_raises(self):
        cfg = _PlainLoraConfig(target_modules=[])
        with pytest.raises(ValueError, match="lora_target_scope is set but no"):
            adapt_lora_config_for_model(
                _LanguageScopedLinearModel(), cfg, lora_target_scope="language_model"
            )

    def test_inferred_language_model_scope_with_all_linear(self):
        model = _ScopedClippableModel()
        cfg = _PlainLoraConfig(target_modules="all-linear")
        adapted = adapt_lora_config_for_model(model, cfg)
        assert adapted.target_modules == build_scoped_lora_target_regex(
            ["k_proj", "q_proj"], "language_model"
        )

    def test_unscoped_clippable_rewrites_to_suffix_list(self):
        model = _RootClippableModel()
        cfg = _PlainLoraConfig(target_modules=["q_proj"])
        adapted = adapt_lora_config_for_model(model, cfg)
        assert adapted is not cfg
        assert adapted.target_modules == ["q_proj.linear"]
        assert cfg.target_modules == ["q_proj"]

    def test_unscoped_clippable_with_matching_suffix_targets_is_identity(self):
        model = _RootClippableModel()
        cfg = _PlainLoraConfig(target_modules=["q_proj.linear"])
        assert adapt_lora_config_for_model(model, cfg) is cfg

    def test_unscoped_clippable_with_empty_targets_raises(self):
        cfg = _PlainLoraConfig(target_modules=[])
        with pytest.raises(
            ValueError, match="ClippableLinear wrappers but no projection names"
        ):
            adapt_lora_config_for_model(_RootClippableModel(), cfg)

    def test_unscoped_clippable_with_unmatched_projection_raises(self):
        model = _RootClippableModel()
        cfg = _PlainLoraConfig(target_modules=["v_proj"])
        with pytest.raises(
            ValueError,
            match="No modules matched LoRA target_modules for ClippableLinear",
        ) as exc_info:
            adapt_lora_config_for_model(model, cfg)
        assert "LORA_TARGET_SCOPE" in str(exc_info.value)


class TestLogCudaMemorySnapshot:
    def test_noop_without_cuda(self, monkeypatch, caplog):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with caplog.at_level(logging.INFO, logger="agilerl.utils.llm_utils"):
            log_cuda_memory_snapshot("label")
        assert caplog.text == ""

    def test_logs_allocated_and_reserved_gib(self, monkeypatch, caplog):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 2 * GiB)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 3 * GiB)
        with caplog.at_level(logging.INFO, logger="agilerl.utils.llm_utils"):
            log_cuda_memory_snapshot("after wake_up", device_index=0)
        assert "after wake_up" in caplog.text
        assert "allocated=2.00 GiB" in caplog.text
        assert "reserved=3.00 GiB" in caplog.text


class TestFormatColocatedVllmOomHint:
    @staticmethod
    def _fake_cuda(monkeypatch, free_gib=10, total_gib=40, alloc_gib=25):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "mem_get_info",
            lambda device_index: (free_gib * GiB, total_gib * GiB),
        )
        monkeypatch.setattr(
            torch.cuda, "memory_allocated", lambda device_index: alloc_gib * GiB
        )

    def test_no_cuda_short_circuit(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert format_colocated_vllm_oom_hint() == "CUDA is not available on this host."

    def test_full_summary_includes_all_sections(self, monkeypatch):
        self._fake_cuda(monkeypatch)
        hint = format_colocated_vllm_oom_hint(
            kv_cache_memory_bytes=12 * GiB,
            gpu_memory_utilization=0.55,
            max_model_len=32768,
        )
        assert "40.00 GiB total" in hint
        assert "25.00 GiB torch-allocated" in hint
        assert "10.00 GiB free (driver)" in hint
        assert "requests 12.00 GiB" in hint
        assert "≈2.00 GiB more free VRAM" in hint
        assert "gpu_memory_utilization=0.55" in hint
        assert "max_model_len=32768" in hint
        assert "DeepSpeed trainer" in hint  # trainer_on_gpu defaults to True
        assert "Check nvidia-smi" in hint

    def test_optional_sections_omitted(self, monkeypatch):
        self._fake_cuda(monkeypatch)
        hint = format_colocated_vllm_oom_hint(trainer_on_gpu=False)
        # Only the device summary and the closing advice remain.
        assert "requests" not in hint  # kv_cache_memory_bytes line
        assert "is also checked at" not in hint  # gpu_memory_utilization line
        assert "KV slot length cap" not in hint  # max_model_len line
        assert "DeepSpeed trainer" not in hint
        assert "Check nvidia-smi" in hint

    def test_kv_shortfall_clamped_at_zero(self, monkeypatch):
        self._fake_cuda(monkeypatch, free_gib=10)
        hint = format_colocated_vllm_oom_hint(kv_cache_memory_bytes=5 * GiB)
        assert "≈0.00 GiB more free VRAM" in hint


class TestResolveAttnImplementation:
    def test_explicit_choice_is_authoritative(self):
        assert resolve_attn_implementation("flex_attention") == "flex_attention"
        assert resolve_attn_implementation("eager") == "eager"

    def test_auto_prefers_flash_attention_2_when_installed(self, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
        assert resolve_attn_implementation(None) == "flash_attention_2"
        assert resolve_attn_implementation("auto") == "flash_attention_2"

    def test_auto_falls_back_to_sdpa(self, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        assert resolve_attn_implementation(None) == "sdpa"


class _RegisterOnlyRegistry:
    """Attention-function registry that rejects item assignment."""

    def __init__(self):
        self.registered = {}

    def __setitem__(self, key, value):
        msg = "immutable registry"
        raise TypeError(msg)

    def register(self, key, value):
        self.registered[key] = value


class TestPatchFlexAttentionKernelOptions:
    @staticmethod
    def _install_fake_flex(monkeypatch, *, already_patched=False, registry=None):
        """Install fake transformers flex-attention modules into sys.modules."""
        calls = []

        def fake_flex_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        ):
            calls.append(kwargs)
            return "flex-out"

        if already_patched:
            fake_flex_attention_forward._agilerl_kernel_opts_patched = True
        flex_mod = types.ModuleType("transformers.integrations.flex_attention")
        flex_mod.flex_attention_forward = fake_flex_attention_forward
        modeling_mod = types.ModuleType("transformers.modeling_utils")
        modeling_mod.ALL_ATTENTION_FUNCTIONS = registry if registry is not None else {}
        monkeypatch.setitem(
            sys.modules, "transformers.integrations.flex_attention", flex_mod
        )
        monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_mod)
        return modeling_mod.ALL_ATTENTION_FUNCTIONS, calls

    def test_returns_silently_when_flex_import_unavailable(self, monkeypatch):
        flex_mod = types.ModuleType("transformers.integrations.flex_attention")
        # Intentionally missing ``flex_attention_forward``.
        modeling_mod = types.ModuleType("transformers.modeling_utils")
        modeling_mod.ALL_ATTENTION_FUNCTIONS = {}
        monkeypatch.setitem(
            sys.modules, "transformers.integrations.flex_attention", flex_mod
        )
        monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_mod)
        patch_flex_attention_kernel_options()
        assert modeling_mod.ALL_ATTENTION_FUNCTIONS == {}

    def test_double_patch_guard_skips_reinstall(self, monkeypatch):
        registry, _ = self._install_fake_flex(monkeypatch, already_patched=True)
        patch_flex_attention_kernel_options()
        assert registry == {}

    @staticmethod
    def _query(seq_len_q):
        return torch.zeros(2, 8, seq_len_q, 64)

    def test_hopper_skips_sram_safe_blocks_on_long_queries(self, monkeypatch):
        registry, calls = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
        patch_flex_attention_kernel_options()
        wrapper = registry["flex_attention"]
        wrapper(None, self._query(512), "k", "v", None)
        # Autotuner keeps full control of the training forward/backward blocks.
        assert "kernel_options" not in calls[0]

    def test_hopper_sets_decode_safe_block_m_on_short_queries(self, monkeypatch):
        registry, calls = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
        patch_flex_attention_kernel_options()
        wrapper = registry["flex_attention"]
        wrapper(None, self._query(111), "k", "v", None)
        assert calls[0]["kernel_options"] == {"BLOCK_M": 128}

    def test_installs_sram_safe_defaults_without_cuda(self, monkeypatch):
        registry, calls = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        patch_flex_attention_kernel_options()
        wrapper = registry["flex_attention"]
        assert wrapper._agilerl_kernel_opts_patched is True
        out = wrapper(None, "q", "k", "v", None)
        assert out == "flex-out"
        opts = calls[0]["kernel_options"]
        assert opts["BLOCK_M"] == 32
        assert opts["BLOCK_N"] == 32
        assert opts["BLOCK_M1"] == 16
        assert opts["num_warps"] == 4
        assert opts["num_stages"] == 2

    def test_capability_probe_failure_installs_safe_defaults(self, monkeypatch):
        registry, _ = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def _raise_capability_error():
            msg = "no device"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            torch.cuda, "get_device_capability", _raise_capability_error
        )
        patch_flex_attention_kernel_options()
        assert "flex_attention" in registry

    def test_explicit_options_bypass_hopper_autoskip(self, monkeypatch):
        registry, calls = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
        patch_flex_attention_kernel_options(options={"BLOCK_M": 64})
        wrapper = registry["flex_attention"]
        wrapper(None, "q", "k", "v", None)
        assert calls[0]["kernel_options"] == {"BLOCK_M": 64}

    def test_caller_kernel_options_take_precedence(self, monkeypatch):
        registry, calls = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        patch_flex_attention_kernel_options()
        wrapper = registry["flex_attention"]
        wrapper(None, "q", "k", "v", None, kernel_options={"BLOCK_M": 128})
        assert calls[0]["kernel_options"] == {"BLOCK_M": 128}

    def test_falls_back_to_registry_register_method(self, monkeypatch):
        registry = _RegisterOnlyRegistry()
        self._install_fake_flex(monkeypatch, registry=registry)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        patch_flex_attention_kernel_options()
        assert "flex_attention" in registry.registered

    def test_sram_safe_block_m_divides_the_mask_q_block(self, monkeypatch):
        registry, calls = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        patch_flex_attention_kernel_options()
        registry["flex_attention"](None, self._query(111), "k", "v", None)
        # Pre-Hopper defaults must satisfy the same flex-decoding divisibility
        # check that the Hopper path gets from flex_decode_kernel_options.
        assert 128 % calls[0]["kernel_options"]["BLOCK_M"] == 0


class TestFlexDecodeKernelOptions:
    @staticmethod
    def _query(seq_len_q):
        return torch.zeros(2, 8, seq_len_q, 64)

    def test_long_query_leaves_block_size_to_inductor(self):
        assert flex_decode_kernel_options(self._query(128)) is None

    @pytest.mark.parametrize("seq_len_q", [1, 111, 127])
    def test_short_query_pins_block_m_to_the_default_q_block(self, seq_len_q):
        assert flex_decode_kernel_options(self._query(seq_len_q)) == {"BLOCK_M": 128}

    def test_narrower_mask_q_block_lowers_block_m(self):
        mask = types.SimpleNamespace(BLOCK_SIZE=(64, 64))
        assert flex_decode_kernel_options(self._query(111), mask) == {"BLOCK_M": 64}

    def test_maskless_giant_block_is_capped(self):
        # torch's empty BlockMask spans the sequence in one 2**30-wide block.
        mask = types.SimpleNamespace(BLOCK_SIZE=(1 << 30, 1 << 30))
        assert flex_decode_kernel_options(self._query(111), mask) == {"BLOCK_M": 128}

    @pytest.mark.parametrize("query", ["q", None, torch.zeros(4)])
    def test_unshaped_query_returns_none(self, query):
        assert flex_decode_kernel_options(query) is None


class TestCreateModelFromNameOrPathDefaults:
    @staticmethod
    def _fake_loader(captured):
        class _Loader:
            @staticmethod
            def from_pretrained(pretrained_model_name_or_path, **kwargs):
                captured["name"] = pretrained_model_name_or_path
                captured["kwargs"] = kwargs
                return "model-sentinel"

        return _Loader

    def test_defaults_to_bf16_and_sdpa(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            llm_utils_module, "AutoModelForCausalLM", self._fake_loader(captured)
        )
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        out = create_model_from_name_or_path("org/tiny")
        assert out == "model-sentinel"
        assert captured["name"] == "org/tiny"
        assert captured["kwargs"]["torch_dtype"] is torch.bfloat16
        assert captured["kwargs"]["attn_implementation"] == "sdpa"

    def test_accelerator_defaults_to_fp16(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            llm_utils_module, "AutoModelForCausalLM", self._fake_loader(captured)
        )
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        create_model_from_name_or_path("org/tiny", use_accelerator=True)
        assert captured["kwargs"]["torch_dtype"] is torch.float16

    def test_caller_dtype_stays_authoritative(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            llm_utils_module, "AutoModelForCausalLM", self._fake_loader(captured)
        )
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        create_model_from_name_or_path(
            "org/tiny", model_config={"torch_dtype": torch.float32}
        )
        assert captured["kwargs"]["torch_dtype"] is torch.float32

    def test_flex_attention_triggers_kernel_options_patch(self, monkeypatch):
        captured = {}
        patch_calls = []
        monkeypatch.setattr(
            llm_utils_module, "AutoModelForCausalLM", self._fake_loader(captured)
        )
        monkeypatch.setattr(
            llm_utils_module,
            "patch_flex_attention_kernel_options",
            lambda *a, **k: patch_calls.append(1),
        )
        caller_config = {"attn_implementation": "flex_attention"}
        create_model_from_name_or_path("org/tiny", model_config=caller_config)
        assert captured["kwargs"]["attn_implementation"] == "flex_attention"
        assert patch_calls == [1]
        # The caller's dict is copied, not mutated.
        assert caller_config == {"attn_implementation": "flex_attention"}


class TestValidateImportanceSamplingLevel:
    @pytest.mark.parametrize("level", ["token", "turn", "trajectory"])
    @pytest.mark.parametrize("allow_auto", [True, False])
    def test_valid_levels_pass(self, level, allow_auto):
        validate_importance_sampling_level(level, allow_auto=allow_auto)

    def test_auto_accepted_only_when_allowed(self):
        validate_importance_sampling_level("auto", allow_auto=True)
        with pytest.raises(ValueError, match="got 'auto'"):
            validate_importance_sampling_level("auto", allow_auto=False)

    def test_legacy_sequence_level_rejected_after_rename(self):
        with pytest.raises(ValueError, match="got 'sequence'") as exc_info:
            validate_importance_sampling_level("sequence", allow_auto=True)
        assert "trajectory" in str(exc_info.value)


class TestPoolLogRatioByLevel:
    def test_token_level_is_identity(self):
        tlr = torch.tensor([[1.0, 2.0]])
        mask = torch.tensor([[1.0, 0.0]])
        weights, unit_mask = pool_log_ratio_by_level(tlr, mask, None, "token")
        assert weights is tlr
        assert unit_mask is mask

    def test_turn_level_infers_num_turns_from_turn_ids(self):
        tlr = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.ones(1, 4)
        turn_ids = torch.tensor([[0, 0, 1, 1]])
        weights, unit_mask = pool_log_ratio_by_level(
            tlr, mask, turn_ids, "turn", num_turns=None
        )
        assert weights.shape == (1, 2)
        assert weights[0].tolist() == pytest.approx([1.5, 3.5])
        assert unit_mask.tolist() == [[1.0, 1.0]]

    def test_turn_level_masks_out_turns_without_action_tokens(self):
        tlr = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        turn_ids = torch.tensor([[0, 0, 1, 1]])
        weights, unit_mask = pool_log_ratio_by_level(tlr, mask, turn_ids, "turn")
        assert unit_mask.tolist() == [[1.0, 0.0]]
        assert weights[0, 1].item() == pytest.approx(0.0)

    def test_turn_level_sum_reduction_matches_product_ratio_log_pooling(self):
        tlr = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.ones(1, 4)
        turn_ids = torch.tensor([[0, 0, 1, 1]])
        weights, unit_mask = pool_log_ratio_by_level(
            tlr, mask, turn_ids, "turn", turn_reduction="sum"
        )
        assert weights.shape == (1, 2)
        assert weights[0].tolist() == pytest.approx([3.0, 7.0])
        assert unit_mask.tolist() == [[1.0, 1.0]]

    def test_trajectory_level_masks_rows_without_action_tokens(self):
        tlr = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        weights, unit_mask = pool_log_ratio_by_level(tlr, mask, None, "trajectory")
        assert unit_mask.tolist() == [[1.0], [0.0]]
        assert weights[0, 0].item() == pytest.approx(1.5)
        assert weights[1, 0].item() == pytest.approx(0.0)


class TestAttentionMaskFromPaddedIds:
    def test_masks_only_the_trailing_padding_run(self):
        ids = torch.tensor([[5, 0, 7, 0, 0], [1, 2, 3, 4, 0]])
        mask = attention_mask_from_padded_ids(ids, 0)
        assert mask.tolist() == [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]

    def test_none_pad_id_masks_nothing(self):
        ids = torch.tensor([[1, 2, 3]])
        assert attention_mask_from_padded_ids(ids, None).all()

    def test_unpadded_rows_stay_fully_attended(self):
        ids = torch.tensor([[1, 2, 0, 3]])
        mask = attention_mask_from_padded_ids(ids, 0)
        assert mask.tolist() == [[True, True, True, True]]


class TestBuildCompletionMask:
    def test_masks_prompt_and_pad_positions_with_shift(self):
        completion = torch.tensor([[5, 6, 7, 0, 0]])
        mask = build_completion_mask(completion, prompt_len=2, pad_token_id=0)
        # Positions >= 2 and non-pad -> [F,F,T,F,F]; leading position dropped.
        assert mask.shape == (1, 4)
        assert mask.tolist() == [[False, True, False, False]]

    @pytest.mark.parametrize("prompt_len", [None, 0])
    def test_no_prompt_prefix_keeps_all_non_pad_tokens(self, prompt_len):
        completion = torch.tensor([[5, 6, 7, 0, 0]])
        mask = build_completion_mask(completion, prompt_len=prompt_len, pad_token_id=0)
        assert mask.tolist() == [[True, True, False, False]]


class TestCudaTensorBytesInModule:
    def test_cpu_module_reports_zero(self):
        assert cuda_tensor_bytes_in_module(nn.Linear(4, 4)) == 0

    def test_sums_only_cuda_tensors_across_params_and_buffers(self):
        class _FakeDeviceTensor:
            def __init__(self, numel, element_size, is_cuda):
                self.is_cuda = is_cuda
                self._numel = numel
                self._element_size = element_size

            def numel(self):
                return self._numel

            def element_size(self):
                return self._element_size

        module = SimpleNamespace(
            parameters=lambda: iter(
                [
                    _FakeDeviceTensor(10, 2, is_cuda=True),
                    _FakeDeviceTensor(100, 4, is_cuda=False),
                ]
            ),
            buffers=lambda: iter([_FakeDeviceTensor(5, 4, is_cuda=True)]),
        )
        assert cuda_tensor_bytes_in_module(module) == 10 * 2 + 5 * 4


class TestCollectTrainableParamStats:
    @staticmethod
    def _mixed_grad_net():
        net = nn.Module()
        net.trainable = nn.Linear(4, 4)  # 20 params
        net.frozen = nn.Linear(4, 2)  # 10 params
        net.frozen.requires_grad_(False)
        return net

    def test_counts_trainable_and_total_params(self):
        agent = SimpleNamespace(actor=self._mixed_grad_net())
        stats = collect_trainable_param_stats([agent])
        assert stats["trainable_params"] == 20
        assert stats["total_params"] == 30
        assert stats["trainable_param_ratio"] == pytest.approx(20 / 30)

    def test_unwraps_module_then_model_attributes(self):
        inner = self._mixed_grad_net()
        actor = SimpleNamespace(module=SimpleNamespace(model=inner))
        stats = collect_trainable_param_stats([SimpleNamespace(actor=actor)])
        assert stats["trainable_params"] == 20
        assert stats["total_params"] == 30

    def test_actor_none_returns_empty_dict(self):
        assert collect_trainable_param_stats([SimpleNamespace(actor=None)]) == {}

    def test_zero_param_actor_returns_empty_dict(self):
        assert collect_trainable_param_stats([SimpleNamespace(actor=nn.Module())]) == {}

    def test_empty_population_swallowed_as_empty_dict(self):
        assert collect_trainable_param_stats([]) == {}

    def test_introspection_failure_swallowed_as_empty_dict(self):
        class _ExplodingActor:
            def parameters(self):
                msg = "parameter introspection failed"
                raise RuntimeError(msg)

        agent = SimpleNamespace(actor=_ExplodingActor())
        assert collect_trainable_param_stats([agent]) == {}


class _RecordingToModule(nn.Module):
    """nn.Module that records every ``.to()`` call."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class TestOffloadColocatedTrainerFromGpu:
    def test_forces_cpu_move_even_when_already_on_cpu(self, monkeypatch):
        model = _RecordingToModule()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        remaining = offload_colocated_trainer_from_gpu(model)
        assert remaining == 0
        assert (("cpu",), {}) in model.to_calls

    def test_synchronizes_and_clears_cache_when_cuda_available(self, monkeypatch):
        model = _RecordingToModule()
        calls = {"sync": 0, "empty": 0}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "synchronize",
            lambda: calls.__setitem__("sync", calls["sync"] + 1),
        )
        monkeypatch.setattr(
            torch.cuda,
            "empty_cache",
            lambda: calls.__setitem__("empty", calls["empty"] + 1),
        )
        remaining = offload_colocated_trainer_from_gpu(model)
        assert remaining == 0
        assert calls == {"sync": 1, "empty": 1}


class TestResolveVllmMaxLoraRank:
    def test_no_trainer_rank_keeps_configured_value(self):
        assert resolve_vllm_max_lora_rank(16, None) == 16

    def test_trainer_rank_raises_floor(self):
        assert resolve_vllm_max_lora_rank(16, 32) == 32

    def test_configured_value_wins_when_larger(self):
        assert resolve_vllm_max_lora_rank(16, 8) == 16


class TestResolveVllmMaxNumBatchedTokens:
    def test_explicit_value_is_authoritative(self):
        assert resolve_vllm_max_num_batched_tokens(8, 32768, explicit=4096) == 4096

    def test_small_worst_case_used_directly(self):
        # 2 * 1024 = 2048 is below the 8k-per-slot concurrent budget.
        assert resolve_vllm_max_num_batched_tokens(2, 1024) == 2048

    def test_concurrent_budget_caps_long_context_batches(self):
        # 8 * 32768 = 262144 worst case capped at max(32768, 8 * 8192) = 65536.
        assert resolve_vllm_max_num_batched_tokens(8, 32768) == 65536

    def test_keeps_at_least_one_full_context(self):
        # Budget never drops below one max_model_len context.
        assert resolve_vllm_max_num_batched_tokens(1, 32768) == 32768


def _vllm_config(**overrides):
    """SimpleNamespace mirroring the VLLMConfig fields read by the builder."""
    base = {
        "vllm_model_name_or_path": None,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_num_seqs": 8,
        "sleep_mode": True,
        "dtype": None,
        "quantization": None,
        "kv_cache_dtype": None,
        "kv_cache_memory_bytes": None,
        "enforce_eager": None,
        "max_lora_rank": 16,
        "max_loras": 1,
        "max_num_batched_tokens": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestBuildVllmLlmInitKwargs:
    def test_defaults_fall_back_to_trainer_model(self):
        kwargs = build_vllm_llm_init_kwargs(
            _vllm_config(),
            trainer_model_name_or_path="org/base",
            max_model_len=32768,
        )
        assert kwargs["model"] == "org/base"
        assert kwargs["tensor_parallel_size"] == 1
        assert kwargs["gpu_memory_utilization"] == 0.8
        assert kwargs["max_num_seqs"] == 8
        assert kwargs["max_model_len"] == 32768
        assert kwargs["distributed_executor_backend"] == "external_launcher"
        assert kwargs["seed"] == 0
        assert kwargs["max_num_batched_tokens"] == 65536
        assert kwargs["model_impl"] == "vllm"
        assert kwargs["enable_sleep_mode"] is True
        # Colocated vLLM always serves the trainer's LoRA adapter.
        assert kwargs["enable_lora"] is True
        assert kwargs["max_lora_rank"] == 16
        assert kwargs["max_loras"] == 1
        for absent in (
            "dtype",
            "quantization",
            "kv_cache_dtype",
            "kv_cache_memory_bytes",
            "enforce_eager",
        ):
            assert absent not in kwargs

    def test_optional_fields_and_lora_rank_floor(self):
        kwargs = build_vllm_llm_init_kwargs(
            _vllm_config(
                vllm_model_name_or_path="org/quantized",
                dtype="bfloat16",
                quantization="bitsandbytes",
                kv_cache_dtype="fp8",
                kv_cache_memory_bytes=123456,
                enforce_eager=True,
                max_num_batched_tokens=4096,
                tensor_parallel_size=2,
            ),
            trainer_model_name_or_path="org/base",
            max_model_len=8192,
            process_index=5,
            lora_rank=64,
        )
        assert kwargs["model"] == "org/quantized"
        assert kwargs["dtype"] == "bfloat16"
        assert kwargs["quantization"] == "bitsandbytes"
        assert kwargs["kv_cache_dtype"] == "fp8"
        assert kwargs["kv_cache_memory_bytes"] == 123456
        assert kwargs["enforce_eager"] is True
        assert kwargs["max_num_batched_tokens"] == 4096
        assert kwargs["seed"] == 2  # process_index // tensor_parallel_size
        assert kwargs["max_lora_rank"] == 64  # trainer rank outranks the config


class TestBuildVllmRolloutLoraRequest:
    def test_builds_request_with_defaults_and_str_path(self, monkeypatch, tmp_path):
        class _FakeLoRARequest:
            def __init__(self, lora_name, lora_int_id, lora_path, load_inplace):
                self.lora_name = lora_name
                self.lora_int_id = lora_int_id
                self.lora_path = lora_path
                self.load_inplace = load_inplace

        fake_mod = types.ModuleType("vllm.lora.request")
        fake_mod.LoRARequest = _FakeLoRARequest
        monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_mod)

        adapter_path = tmp_path / "adapter"
        req = build_vllm_rollout_lora_request(adapter_path, load_inplace=True)
        assert req.lora_name == "actor"
        assert req.lora_int_id == 1
        assert isinstance(req.lora_path, str)
        assert req.lora_path == str(adapter_path)
        assert req.load_inplace is True


class TestPeftLoraKeyHelpers:
    @pytest.mark.parametrize(
        "marker",
        [".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B."],
    )
    def test_module_key_strips_lora_weight_suffixes(self, marker):
        key = f"base.layers.0.q_proj{marker}weight"
        assert peft_lora_state_dict_key_to_module_key(key) == "base.layers.0.q_proj"

    def test_module_key_passthrough_without_marker(self):
        assert (
            peft_lora_state_dict_key_to_module_key("base.layers.0.q_proj.weight")
            == "base.layers.0.q_proj.weight"
        )

    def test_remap_strips_clippable_inner_linear_suffix(self):
        assert (
            remap_peft_lora_key_for_vllm("model.layers.0.q_proj.linear.lora_A.weight")
            == "model.layers.0.q_proj.lora_A.weight"
        )
        assert (
            remap_peft_lora_key_for_vllm("model.layers.0.q_proj.linear.lora_B.weight")
            == "model.layers.0.q_proj.lora_B.weight"
        )

    def test_remap_strips_base_layer_segment(self):
        assert (
            remap_peft_lora_key_for_vllm("model.layers.0.q_proj.base_layer.weight")
            == "model.layers.0.q_proj.weight"
        )

    def test_remap_passthrough_for_plain_keys(self):
        key = "model.layers.0.q_proj.lora_A.weight"
        assert remap_peft_lora_key_for_vllm(key) == key


class TestFilterPeftStateDictForVllmLora:
    def test_keeps_matching_modules_and_remaps_keys(self):
        t_keep = torch.zeros(1)
        t_drop = torch.ones(1)
        state = {
            "model.layers.0.q_proj.linear.lora_A.weight": t_keep,
            "model.layers.0.out_proj.lora_A.weight": t_drop,
        }
        out = filter_peft_state_dict_for_vllm_lora(state, ["q_proj.linear"])
        assert list(out) == ["model.layers.0.q_proj.lora_A.weight"]
        assert out["model.layers.0.q_proj.lora_A.weight"] is t_keep

    def test_regex_target_modules_supported(self):
        state = {"model.layers.0.q_proj.linear.lora_A.weight": torch.zeros(1)}
        out = filter_peft_state_dict_for_vllm_lora(state, r"(?:.*\.)?(q_proj)\.linear")
        assert list(out) == ["model.layers.0.q_proj.lora_A.weight"]

    def test_no_matches_yields_empty_dict(self):
        state = {"model.layers.0.out_proj.lora_A.weight": torch.zeros(1)}
        assert filter_peft_state_dict_for_vllm_lora(state, ["q_proj"]) == {}


class TestJsonSafeValue:
    def test_primitives_pass_through(self):
        assert llm_utils_module._json_safe_value(None) is None
        assert llm_utils_module._json_safe_value("x") == "x"
        assert llm_utils_module._json_safe_value(3) == 3
        assert llm_utils_module._json_safe_value(1.5) == 1.5
        assert llm_utils_module._json_safe_value(True) is True

    def test_sets_become_sorted_lists(self):
        assert llm_utils_module._json_safe_value({"b", "a"}) == ["a", "b"]

    def test_tuples_become_lists(self):
        assert llm_utils_module._json_safe_value((1, "two")) == [1, "two"]

    def test_nested_dicts_get_string_keys(self):
        assert llm_utils_module._json_safe_value({1: {"s": {"b", "a"}}}) == {
            "1": {"s": ["a", "b"]}
        }

    def test_arbitrary_objects_stringified(self):
        assert llm_utils_module._json_safe_value(torch.bfloat16) == "torch.bfloat16"


class TestSavePeftAdapterForVllmRollout:
    class _FakePeftConfig:
        def to_dict(self):
            return {
                "r": 8,
                "target_modules": {"q_proj"},
                "lora_dtype": torch.bfloat16,
            }

    def _install_fakes(self, monkeypatch, state):
        """Install fake peft/safetensors modules and return the call recorder."""
        calls = {}
        fake_peft = types.ModuleType("peft")

        def fake_get_state(model, adapter_name):
            calls["adapter_name"] = adapter_name
            return dict(state)

        fake_peft.get_peft_model_state_dict = fake_get_state
        fake_st = types.ModuleType("safetensors.torch")

        def fake_save_file(tensors, path):
            calls["saved_tensors"] = tensors
            calls["saved_path"] = Path(path)
            Path(path).write_bytes(b"")

        fake_st.save_file = fake_save_file
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st)
        monkeypatch.setattr(llm_utils_module, "HAS_LLM_DEPENDENCIES", True)
        return calls

    def _peft_model(self):
        return SimpleNamespace(peft_config={"actor": self._FakePeftConfig()})

    def test_requires_llm_dependencies(self, monkeypatch, tmp_path):
        monkeypatch.setattr(llm_utils_module, "HAS_LLM_DEPENDENCIES", False)
        with pytest.raises(ImportError, match="requires peft and transformers"):
            save_peft_adapter_for_vllm_rollout(
                MagicMock(), tmp_path, "actor", target_modules=["q_proj"]
            )

    def test_exports_filtered_remapped_adapter_with_config(self, monkeypatch, tmp_path):
        t_keep = torch.zeros(2)
        state = {
            "model.layers.0.q_proj.linear.lora_A.weight": t_keep,
            "model.layers.0.out_proj.lora_A.weight": torch.ones(2),
        }
        calls = self._install_fakes(monkeypatch, state)
        out = save_peft_adapter_for_vllm_rollout(
            self._peft_model(),
            tmp_path,
            "actor",
            target_modules=["q_proj.linear"],
        )
        assert out == tmp_path / "actor"
        assert calls["adapter_name"] == "actor"
        assert list(calls["saved_tensors"]) == ["model.layers.0.q_proj.lora_A.weight"]
        assert calls["saved_tensors"]["model.layers.0.q_proj.lora_A.weight"] is t_keep
        assert calls["saved_path"] == tmp_path / "actor" / "adapter_model.safetensors"
        cfg = json.loads(
            (tmp_path / "actor" / "adapter_config.json").read_text(encoding="utf-8")
        )
        assert cfg["target_modules"] == ["q_proj.linear"]
        assert cfg["r"] == 8
        assert cfg["lora_dtype"] == "torch.bfloat16"

    def test_raises_when_filter_drops_every_tensor(self, monkeypatch, tmp_path):
        state = {"model.layers.0.out_proj.lora_A.weight": torch.zeros(2)}
        self._install_fakes(monkeypatch, state)
        with pytest.raises(ValueError, match="No LoRA tensors left for vLLM export"):
            save_peft_adapter_for_vllm_rollout(
                self._peft_model(),
                tmp_path,
                "actor",
                target_modules=["q_proj"],
            )


class TestCrossRankLigerAlign:
    def test_needs_cross_rank_seq_padding_gates_on_liger_token_is(self):
        assert not llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(use_liger_loss=True, importance_sampling_level="token"),
            world_size=1,
        )
        assert not llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(use_liger_loss=False, importance_sampling_level="token"),
            world_size=2,
        )
        assert not llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(
                use_liger_loss=True, importance_sampling_level="trajectory"
            ),
            world_size=2,
        )
        # Missing attrs: use_liger_loss defaults False; IS level defaults "token".
        assert not llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(),
            world_size=2,
        )
        assert llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(use_liger_loss=True, importance_sampling_level="token"),
            world_size=2,
        )
        assert llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(use_liger_loss=True),  # default IS level is token
            world_size=2,
        )

    def test_needs_cross_rank_seq_padding_gates_on_zero_stage_3(self):
        assert llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(zero_stage=3, use_liger_loss=False),
            world_size=2,
        )
        assert llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(zero_stage="3", use_liger_loss=False),
            world_size=2,
        )
        assert not llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(zero_stage=3, use_liger_loss=False),
            world_size=1,
        )
        assert not llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(zero_stage=2, use_liger_loss=False),
            world_size=2,
        )
        assert not llm_utils_module.needs_cross_rank_seq_padding(
            SimpleNamespace(use_liger_loss=False),
            world_size=2,
        )

    def test_allreduce_minmax_int_uses_accelerator_gather(self):
        acc = MagicMock()
        acc.device = torch.device("cpu")
        acc.gather.side_effect = lambda t: torch.tensor([2, 5], dtype=t.dtype)

        min_v, max_v = llm_utils_module.allreduce_minmax_int(3, acc)
        assert (min_v, max_v) == (2, 5)
        acc.gather.assert_called_once()
        gathered_arg = acc.gather.call_args.args[0]
        assert gathered_arg.tolist() == [3]
        assert gathered_arg.dtype == torch.long

    def test_pad_completion_batch_to_seq_len_happy_and_noop(self):
        ids = torch.ones(2, 3, dtype=torch.long)
        masks = torch.ones(2, 2, dtype=torch.bool)

        same_ids, same_masks = llm_utils_module.pad_completion_batch_to_seq_len(
            ids, masks, target_seq_len=3, pad_token_id=0
        )
        assert same_ids is ids
        assert same_masks is masks

        pad_ids, pad_masks = llm_utils_module.pad_completion_batch_to_seq_len(
            ids, masks, target_seq_len=5, pad_token_id=9
        )
        assert pad_ids.shape == (2, 5)
        assert pad_masks.shape == (2, 4)
        assert torch.all(pad_ids[:, 3:] == 9)
        assert torch.all(~pad_masks[:, 2:])

    @pytest.mark.parametrize(
        ("ids", "masks", "target", "match"),
        [
            (
                torch.ones(2, dtype=torch.long),
                torch.ones(2, 1, dtype=torch.bool),
                2,
                "token_ids must be \\(B, T\\)",
            ),
            (
                torch.ones(2, 3, dtype=torch.long),
                torch.ones(2, dtype=torch.bool),
                3,
                "action_masks must be \\(B, T-1\\)",
            ),
            (
                torch.ones(2, 3, dtype=torch.long),
                torch.ones(2, 1, dtype=torch.bool),
                3,
                "action_masks length",
            ),
            (
                torch.ones(2, 3, dtype=torch.long),
                torch.ones(2, 2, dtype=torch.bool),
                2,
                "target_seq_len",
            ),
        ],
    )
    def test_pad_completion_batch_to_seq_len_validation_errors(
        self, ids, masks, target, match
    ):
        with pytest.raises(ValueError, match=match):
            llm_utils_module.pad_completion_batch_to_seq_len(
                ids, masks, target_seq_len=target, pad_token_id=0
            )

    def test_pad_completion_batch_to_seq_len_post_pad_shape_guards(self):
        ids = torch.ones(2, 3, dtype=torch.long)
        masks = torch.ones(2, 2, dtype=torch.bool)

        with patch(
            "torch.nn.functional.pad",
            side_effect=[
                torch.ones(2, 4, dtype=torch.long),  # wrong T for target=5
                torch.ones(2, 4, dtype=torch.bool),
            ],
        ):
            with pytest.raises(RuntimeError, match="padded completions shape"):
                llm_utils_module.pad_completion_batch_to_seq_len(
                    ids, masks, target_seq_len=5, pad_token_id=0
                )

        with patch(
            "torch.nn.functional.pad",
            side_effect=[
                torch.ones(2, 5, dtype=torch.long),
                torch.ones(2, 3, dtype=torch.bool),  # wrong mask len
            ],
        ):
            with pytest.raises(RuntimeError, match="padded masks shape"):
                llm_utils_module.pad_completion_batch_to_seq_len(
                    ids, masks, target_seq_len=5, pad_token_id=0
                )

    def test_local_batch_and_seq_len_from_list(self):
        rows = [
            torch.ones(1, 3, dtype=torch.long),
            torch.ones(1, 7, dtype=torch.long),
        ]
        assert llm_utils_module._local_batch_and_seq_len(rows) == (2, 7)

    def test_local_batch_and_seq_len_from_tensor(self):
        batch = torch.ones(4, 5, dtype=torch.long)
        assert llm_utils_module._local_batch_and_seq_len(batch) == (4, 5)

    def test_local_batch_and_seq_len_empty_list(self):
        assert llm_utils_module._local_batch_and_seq_len([]) == (0, 0)

    def test_local_batch_and_seq_len_rejects_bad_input(self):
        with pytest.raises(TypeError, match="tensor or list"):
            llm_utils_module._local_batch_and_seq_len("nope")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"\(B, T\)"):
            llm_utils_module._local_batch_and_seq_len(torch.ones(3, dtype=torch.long))

    def test_align_runs_minmax_before_stack_and_pad(self, monkeypatch):
        events: list[str] = []

        def fake_stack_and_pad(*args, **kwargs):
            events.append("stack")
            completions = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
            masks = torch.tensor([[True, True], [True, False]])
            rewards = torch.tensor([[1.0], [0.0]])
            return completions, masks, rewards

        def fake_minmax(value):
            events.append(f"minmax:{value}")
            return value, value

        accelerator = MagicMock()

        def wait_for_everyone():
            events.append("barrier")

        accelerator.wait_for_everyone.side_effect = wait_for_everyone

        monkeypatch.setattr(
            "agilerl.utils.algo_utils.stack_and_pad_experiences",
            fake_stack_and_pad,
        )

        completion_ids = [
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[4, 5]], dtype=torch.long),
        ]
        action_masks = [
            torch.tensor([[True, True]]),
            torch.tensor([[True]]),
        ]
        rewards = torch.tensor([[1.0], [0.0]])

        out_ids, out_masks, out_rewards = (
            llm_utils_module.align_completion_batch_shapes_across_ranks(
                completion_ids,
                action_masks,
                rewards,
                pad_token_id=0,
                accelerator=accelerator,
                minmax_fn=fake_minmax,
            )
        )

        assert events[0:2] == ["minmax:2", "minmax:3"]
        assert events[2] == "stack"
        assert events[-1] == "barrier"
        assert accelerator.wait_for_everyone.call_count == 1
        assert out_ids.shape == (2, 3)
        assert out_masks.shape == (2, 2)
        assert out_rewards.shape == (2, 1)

    def test_align_completion_batch_shapes_pads_short_rank(self):
        short_ids = [
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.long),
        ]
        short_mask = [
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        ]
        rewards = torch.zeros(2, dtype=torch.float32)
        accelerator = MagicMock()

        def fake_minmax(value):
            # Local max T=4 before stack; pretend peer has T=6.
            return (4, 6) if value == 4 else (value, value)

        out_ids, out_mask, out_rewards = (
            llm_utils_module.align_completion_batch_shapes_across_ranks(
                short_ids,
                short_mask,
                rewards,
                pad_token_id=0,
                accelerator=accelerator,
                minmax_fn=fake_minmax,
            )
        )
        assert out_ids.shape == (2, 6)
        assert out_mask.shape == (2, 5)
        assert out_rewards.shape == (2,)
        assert torch.all(out_ids[:, 4:] == 0)
        assert torch.all(~out_mask[:, 3:])
        accelerator.wait_for_everyone.assert_called_once()

    def test_align_completion_batch_shapes_noop_when_t_already_global_max(self):
        ids = [torch.ones(1, 4, dtype=torch.long)]
        masks = [torch.ones(1, 3, dtype=torch.bool)]
        rewards = torch.zeros(1, dtype=torch.float32)
        accelerator = MagicMock()

        def fake_minmax(value):
            return (value, value)

        out_ids, out_mask, _ = (
            llm_utils_module.align_completion_batch_shapes_across_ranks(
                ids,
                masks,
                rewards,
                pad_token_id=0,
                accelerator=accelerator,
                minmax_fn=fake_minmax,
            )
        )
        assert out_ids.shape == (1, 4)
        assert out_mask.shape == (1, 3)
        accelerator.wait_for_everyone.assert_called_once()

    def test_align_completion_batch_shapes_raises_on_b_diverge(self):
        ids = [torch.ones(1, 3, dtype=torch.long)]
        masks = [torch.ones(1, 2, dtype=torch.bool)]
        rewards = torch.zeros(1, dtype=torch.float32)

        def fake_minmax(value):
            # Local B=1 from row count; pretend peer has B=2.
            return (1, 2) if value == 1 else (value, value)

        with pytest.raises(RuntimeError, match="row counts diverge"):
            llm_utils_module.align_completion_batch_shapes_across_ranks(
                ids,
                masks,
                rewards,
                pad_token_id=0,
                accelerator=MagicMock(),
                minmax_fn=fake_minmax,
            )

    def test_align_completion_batch_shapes_raises_when_t_mismatch_after_sync(self):
        ids = [torch.ones(1, 4, dtype=torch.long)]
        masks = [torch.ones(1, 3, dtype=torch.bool)]
        rewards = torch.zeros(1, dtype=torch.float32)

        def fake_minmax(value):
            # B agrees; claimed global max T is shorter than local stacked T,
            # so no pad runs and the post-align guard fires.
            if value == 1:
                return (1, 1)
            return (2, 2)

        with pytest.raises(RuntimeError, match="Cross-rank seq align failed"):
            llm_utils_module.align_completion_batch_shapes_across_ranks(
                ids,
                masks,
                rewards,
                pad_token_id=0,
                accelerator=MagicMock(),
                minmax_fn=fake_minmax,
            )

    def test_align_invokes_wait_for_everyone_after_pad(self, monkeypatch):
        def fake_stack_and_pad(*args, **kwargs):
            completions = torch.tensor([[1, 2, 3]], dtype=torch.long)
            masks = torch.tensor([[True, True]])
            rewards = torch.tensor([[1.0]])
            return completions, masks, rewards

        monkeypatch.setattr(
            "agilerl.utils.algo_utils.stack_and_pad_experiences",
            fake_stack_and_pad,
        )
        accelerator = MagicMock()
        llm_utils_module.align_completion_batch_shapes_across_ranks(
            [torch.tensor([[1, 2, 3]], dtype=torch.long)],
            [torch.tensor([[True, True]])],
            torch.tensor([[1.0]]),
            pad_token_id=0,
            accelerator=accelerator,
            minmax_fn=lambda value: (value, value),
        )
        assert accelerator.wait_for_everyone.call_count == 1

    def test_align_uses_allreduce_minmax_when_minmax_fn_omitted(self):
        ids = [torch.ones(1, 3, dtype=torch.long)]
        masks = [torch.ones(1, 2, dtype=torch.bool)]
        rewards = torch.zeros(1, dtype=torch.float32)
        acc = MagicMock()

        with patch.object(
            llm_utils_module,
            "allreduce_minmax_int",
            side_effect=lambda value, _acc: (value, value),
        ) as mock_minmax:
            out_ids, _, _ = llm_utils_module.align_completion_batch_shapes_across_ranks(
                ids,
                masks,
                rewards,
                pad_token_id=0,
                accelerator=acc,
            )
        assert out_ids.shape == (1, 3)
        assert mock_minmax.call_count == 2  # B then T
        assert mock_minmax.call_args_list[0].args[1] is acc
        acc.wait_for_everyone.assert_called_once()


class TestResolvePadTokenId:
    """Canonical pad-token resolution prefers a pad id distinct from eos."""

    @staticmethod
    def _tokenizer(
        *,
        eos_id: int | list[int] = 2,
        pad_id: int | None = 11,
        unk_id: int | None = 0,
        eos_token: str = "</s>",
        pad_token: str | None = "<|im_end|>",
        unk_token: str | None = "<unk>",
    ) -> SimpleNamespace:
        eos_ids = eos_id if isinstance(eos_id, list) else [eos_id]
        id_to_token: dict[int, str] = {
            **({unk_id: unk_token} if unk_id is not None and unk_token else {}),
            **({pad_id: pad_token} if pad_id is not None and pad_token else {}),
        }
        for eid in eos_ids:
            id_to_token.setdefault(
                eid, eos_token if eid == eos_ids[0] else f"<eos_{eid}>"
            )

        def convert_ids_to_tokens(token_id: int) -> str:
            return id_to_token[int(token_id)]

        return SimpleNamespace(
            eos_token_id=eos_id,
            eos_token=eos_token,
            pad_token_id=pad_id,
            pad_token=pad_token,
            unk_token_id=unk_id,
            unk_token=unk_token,
            convert_ids_to_tokens=convert_ids_to_tokens,
        )

    def test_model_config_pad_wins_over_tokenizer_im_end(self):
        tokenizer = self._tokenizer(pad_id=11, unk_id=0)
        model_config = SimpleNamespace(pad_token_id=0)

        pad_id, source = resolve_pad_token_id(tokenizer, model_config=model_config)

        assert pad_id == 0
        assert source == "model.config"

    def test_distinct_tokenizer_pad_kept(self):
        tokenizer = self._tokenizer(eos_id=2, pad_id=0, unk_id=None, pad_token="<pad>")

        pad_id, source = resolve_pad_token_id(tokenizer)

        assert pad_id == 0
        assert source == "tokenizer.pad_token_id"

    def test_unk_fallback_when_pad_aliases_eos(self):
        tokenizer = self._tokenizer(
            eos_id=2, pad_id=2, unk_id=0, pad_token="</s>", unk_token="<unk>"
        )

        pad_id, source = resolve_pad_token_id(tokenizer)

        assert pad_id == 0
        assert source == "tokenizer.unk_token_id"

    def test_eos_last_resort_warns(self):
        tokenizer = self._tokenizer(
            eos_id=2, pad_id=2, unk_id=2, pad_token="</s>", unk_token="</s>"
        )

        with pytest.warns(UserWarning, match="eos_token_id"):
            pad_id, source = resolve_pad_token_id(tokenizer)

        assert pad_id == 2
        assert source == "tokenizer.eos_token_id"

    def test_generation_config_before_tokenizer_pad(self):
        tokenizer = self._tokenizer(pad_id=11, unk_id=0)
        generation_config = SimpleNamespace(pad_token_id=0)

        pad_id, source = resolve_pad_token_id(
            tokenizer, generation_config=generation_config
        )

        assert pad_id == 0
        assert source == "generation_config"

    def test_list_eos_rejects_pad_alias_without_model_config(self):
        tokenizer = self._tokenizer(eos_id=[2, 11], pad_id=11, unk_id=0)

        pad_id, source = resolve_pad_token_id(tokenizer)

        assert pad_id == 0
        assert source == "tokenizer.unk_token_id"

    def test_list_eos_model_config_pad_wins(self):
        tokenizer = self._tokenizer(eos_id=[2, 11], pad_id=11, unk_id=0)
        model_config = SimpleNamespace(pad_token_id=0)

        pad_id, source = resolve_pad_token_id(tokenizer, model_config=model_config)

        assert pad_id == 0
        assert source == "model.config"

    def test_scalar_eos_pad_alias_falls_back_to_unk(self):
        tokenizer = self._tokenizer(eos_id=11, pad_id=11, unk_id=0)

        pad_id, source = resolve_pad_token_id(tokenizer)

        assert pad_id == 0
        assert source == "tokenizer.unk_token_id"

    def test_apply_pad_token_id_sets_unk_string(self):
        tokenizer = self._tokenizer(pad_id=11, unk_id=0)
        apply_pad_token_id(tokenizer, 0)
        assert tokenizer.pad_token_id == 0
        assert tokenizer.pad_token == "<unk>"

    def test_multiturn_mask_keeps_im_end_when_pad_is_unk(self):
        """Trailing pad id 0 is masked; mid-sequence ``<|im_end|>`` (11) stays on."""
        tokenizer = self._tokenizer(eos_id=2, pad_id=11, unk_id=0)
        model_config = SimpleNamespace(pad_token_id=0)
        pad_id, _ = resolve_pad_token_id(tokenizer, model_config=model_config)
        apply_pad_token_id(tokenizer, pad_id)
        assert tokenizer.pad_token_id == 0

        # content, <|im_end|>, content, <|im_end|>, trailing pads
        ids = torch.tensor([[5, 6, 11, 7, 11, 0, 0, 0]], dtype=torch.long)
        mask = (ids != tokenizer.pad_token_id).long()

        assert mask[0, 2].item() == 1
        assert mask[0, 4].item() == 1
        assert mask[0, 5].item() == 0
        assert mask[0, 7].item() == 0
        # Old force-eos/im_end-as-pad behavior would zero the mid-sequence 11s.
        assert (ids == 11).any()
        assert not ((ids == 11) & (mask == 0)).any()

    def test_raises_when_tokenizer_has_no_eos(self) -> None:
        tokenizer = self._tokenizer(pad_id=None, unk_id=None)
        tokenizer.eos_token_id = None

        with pytest.raises(ValueError, match="no eos_token_id"):
            resolve_pad_token_id(tokenizer)

    def test_accepts_string_token_ids(self) -> None:
        tokenizer = self._tokenizer(eos_id=2, pad_id=None, unk_id=None)
        tokenizer.pad_token_id = "0"
        tokenizer.eos_token_id = "2"

        pad_id, source = resolve_pad_token_id(tokenizer)

        assert pad_id == 0
        assert source == "tokenizer.pad_token_id"

    def test_apply_pad_sets_id_only_when_convert_ids_fails(self) -> None:
        tokenizer = self._tokenizer(pad_id=11, unk_id=0, pad_token=None)
        tokenizer.pad_token = None

        def _boom(_token_id: int) -> str:
            msg = "lookup failed"
            raise RuntimeError(msg)

        tokenizer.convert_ids_to_tokens = _boom

        apply_pad_token_id(tokenizer, 5)

        assert tokenizer.pad_token_id == 5
        assert tokenizer.pad_token is None

    def test_apply_pad_uses_eos_string_when_pad_aliases_eos(self) -> None:
        tokenizer = self._tokenizer(eos_id=2, pad_id=11, eos_token="</s>")

        apply_pad_token_id(tokenizer, 2)

        assert tokenizer.pad_token_id == 2
        assert tokenizer.pad_token == "</s>"

    def test_apply_pad_converts_the_id_when_pad_is_neither_eos_nor_unk(self) -> None:
        tokenizer = self._tokenizer(eos_id=2, pad_id=None, unk_id=None)
        tokenizer.convert_ids_to_tokens = lambda token_id: f"<tok_{int(token_id)}>"

        apply_pad_token_id(tokenizer, 7)

        assert tokenizer.pad_token_id == 7
        assert tokenizer.pad_token == "<tok_7>"

    def test_load_pad_token_configs_short_circuits_without_a_model_path(self) -> None:
        assert load_pad_token_configs(None) == (None, None)
        assert load_pad_token_configs("") == (None, None)


class TestAsOptionalIntAndEosIdSet:
    def test_as_optional_int_rejects_bool_and_bad_strings(self) -> None:
        from agilerl.utils.llm_utils import _as_optional_int

        assert _as_optional_int(None) is None
        assert _as_optional_int(True) is None
        assert _as_optional_int(False) is None
        assert _as_optional_int("nope") is None
        assert _as_optional_int(3.5) is None
        assert _as_optional_int(7) == 7
        assert _as_optional_int("7") == 7

    def test_eos_id_set_empty_for_bool_and_garbage(self) -> None:
        from agilerl.utils.llm_utils import _eos_id_set

        assert _eos_id_set(True) == frozenset()
        assert _eos_id_set("nope") == frozenset()
        assert _eos_id_set(object()) == frozenset()
        assert _eos_id_set([1, "x", 2]) == frozenset({1, 2})


class TestDeprecatedLlmEnvReExports:
    """``llm_utils`` re-exports names that moved to ``llm_envs``, with a warning."""

    def test_moved_name_warns_and_returns_the_llm_envs_symbol(self):
        import agilerl.llm_envs as llm_envs
        import agilerl.utils.llm_utils as llm_utils

        with pytest.warns(FutureWarning, match="has moved to agilerl.llm_envs"):
            resolved = llm_utils.__getattr__("apply_chat_template")

        assert resolved is llm_envs.apply_chat_template

    def test_unknown_name_still_raises_attribute_error(self):
        import agilerl.utils.llm_utils as llm_utils

        with pytest.raises(AttributeError, match="has no attribute"):
            llm_utils.__getattr__("definitely_not_a_symbol")
