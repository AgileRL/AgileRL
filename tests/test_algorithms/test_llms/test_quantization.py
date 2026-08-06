# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLM memory-optimization plumbing (quantization +
activation offload) through LLMAlgorithm + VLLMConfig.

Most tests are pure-Python: no real model load, no CUDA. The ``gpu``-marked
classes at the bottom are the exception — they load the tiny fixture for real
and verify quantization actually happened. Validates that:
  * `create_model_from_name_or_path` forwards a ``model_config`` (incl. a
    ``quantization_config``) into transformers' ``from_pretrained`` while
    still applying the SDPA + dtype defaults.
  * `build_bnb_quantization_config` resolves YAML-friendly presets / dicts.
  * `_prepare_llm_algo_kwargs` wires ``INIT_HP['QUANTIZATION']`` and
    ``INIT_HP['ACTIVATION_OFFLOAD']`` through.
  * `LLMAlgorithm._activation_offload_ctx` gates ``save_on_cpu`` correctly.
  * `LLMAlgorithm` adds ``lm_head`` to ``llm_int8_skip_modules`` when a fused
    log-prob path is active.
  * `LLMAlgorithm` does NOT touch the skip list when no fused path is on.
  * `VLLMConfig` defaults preserve backwards compatibility (e.g.
    ``vllm_model_name_or_path`` is off by default).
"""

from __future__ import annotations

import json
import types
from unittest import mock

import pytest

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")
# bitsandbytes is a linux-only optional dependency (see pyproject `llm` extra).
# Constructing a 4-bit BitsAndBytesConfig calls post_init(), which looks up the
# installed bnb package version — so the whole quantization test module needs
# bnb actually installed, not just transformers' config class importable.
pytest.importorskip(
    "bitsandbytes",
    reason="quantization tests require bitsandbytes (linux-only optional dep).",
)

import bitsandbytes as bnb
import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.reinforce_llm import REINFORCE
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import (
    _json_safe_value,
    adapt_lora_config_for_model,
    build_bnb_quantization_config,
    build_clippable_linear_lora_target_regex,
    build_clippable_linear_lora_target_suffixes,
    build_scoped_lora_target_regex,
    build_vllm_llm_init_kwargs,
    build_vllm_rollout_lora_request,
    create_model_from_name_or_path,
    cuda_tensor_bytes_in_module,
    filter_peft_state_dict_for_vllm_lora,
    list_peft_matched_module_keys,
    offload_colocated_trainer_from_gpu,
    peft_target_key_matches,
    remap_peft_lora_key_for_vllm,
    resolve_vllm_max_lora_rank,
    resolve_vllm_max_num_batched_tokens,
)
from agilerl.utils.utils import _prepare_llm_algo_kwargs
from tests import TINY_LLM_FIXTURE_PATH


class TestCreateModelFromNameOrPath:
    def test_forwards_quantization_config(self):
        # The production path (LLMAlgorithm.__init__) folds quantization_config
        # into the model_config dict; verify it reaches from_pretrained intact.
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        captured = {}

        def _fake_from_pretrained(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch(
            "agilerl.utils.llm_utils.AutoModelForCausalLM.from_pretrained",
            side_effect=_fake_from_pretrained,
        ):
            create_model_from_name_or_path(
                "dummy/path", model_config={"quantization_config": cfg}
            )

        assert captured.get("quantization_config") is cfg

    def test_quant_config_keeps_sdpa_and_dtype(self):
        # Regression: a model_config carrying a quantization_config must NOT
        # suppress the SDPA + dtype defaults. Eager attention and fp32
        # non-quantized weights at long context erase the quantization win.
        cfg = BitsAndBytesConfig(load_in_4bit=True)
        captured = {}

        def _fake_from_pretrained(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch(
            "agilerl.utils.llm_utils.AutoModelForCausalLM.from_pretrained",
            side_effect=_fake_from_pretrained,
        ):
            create_model_from_name_or_path(
                "dummy/path", model_config={"quantization_config": cfg}
            )

        assert captured.get("attn_implementation") == "sdpa"
        assert "torch_dtype" in captured
        assert captured.get("quantization_config") is cfg

    def test_explicit_model_config_values_win(self):
        # setdefault semantics: a caller-supplied attn_implementation is kept.
        captured = {}

        def _fake_from_pretrained(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch(
            "agilerl.utils.llm_utils.AutoModelForCausalLM.from_pretrained",
            side_effect=_fake_from_pretrained,
        ):
            create_model_from_name_or_path(
                "dummy/path", model_config={"attn_implementation": "eager"}
            )

        assert captured.get("attn_implementation") == "eager"

    def test_none_quantization_does_not_inject_key(self):
        captured = {}

        def _fake_from_pretrained(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch(
            "agilerl.utils.llm_utils.AutoModelForCausalLM.from_pretrained",
            side_effect=_fake_from_pretrained,
        ):
            create_model_from_name_or_path("dummy/path")

        assert "quantization_config" not in captured

    def test_value_head_variant_forwards_quantization_config(self):
        cfg = BitsAndBytesConfig(load_in_4bit=True)
        captured = {}

        def _fake_from_pretrained(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch(
            "agilerl.utils.llm_utils.AutoModelForCausalLMWithValueHead.from_pretrained",
            side_effect=_fake_from_pretrained,
        ):
            create_model_from_name_or_path(
                "dummy/path",
                add_value_head=True,
                model_config={"quantization_config": cfg},
            )

        assert captured.get("quantization_config") is cfg


class TestBuildBnbQuantizationConfig:
    """The YAML / INIT_HP -> BitsAndBytesConfig resolver."""

    def test_none_returns_none(self):
        assert build_bnb_quantization_config(None) is None

    @pytest.mark.parametrize("spec", ["none", "NONE", " none ", ""])
    def test_none_like_strings_return_none(self, spec):
        assert build_bnb_quantization_config(spec) is None

    def test_int8_preset(self):
        cfg = build_bnb_quantization_config("int8")
        assert isinstance(cfg, BitsAndBytesConfig)
        assert cfg.load_in_8bit is True
        assert cfg.load_in_4bit is False

    def test_nf4_preset(self):
        cfg = build_bnb_quantization_config("nf4")
        assert isinstance(cfg, BitsAndBytesConfig)
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.bnb_4bit_use_double_quant is True
        assert cfg.bnb_4bit_compute_dtype == torch.bfloat16

    def test_preset_is_case_insensitive(self):
        assert build_bnb_quantization_config("NF4").load_in_4bit is True

    def test_dict_spec_forwarded_verbatim(self):
        cfg = build_bnb_quantization_config(
            {"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"}
        )
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_compute_dtype == torch.bfloat16

    def test_bitsandbytes_config_passthrough(self):
        original = BitsAndBytesConfig(load_in_8bit=True)
        assert build_bnb_quantization_config(original) is original

    def test_unknown_preset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown quantization preset"):
            build_bnb_quantization_config("fp4")

    def test_bad_type_raises_type_error(self):
        with pytest.raises(TypeError, match="QUANTIZATION must be"):
            build_bnb_quantization_config(4)


class TestInitHpQuantizationWiring:
    """`_prepare_llm_algo_kwargs` resolves INIT_HP['QUANTIZATION']."""

    def _prepare(self, algo_kwargs, init_hp):
        return _prepare_llm_algo_kwargs(
            algo_kwargs,
            tokenizer=None,
            model_name="dummy/path",
            lora_config=None,
            vllm_config=None,
            INIT_HP=init_hp,
        )

    def test_preset_resolved_into_quantization_config(self):
        merged = self._prepare({}, {"QUANTIZATION": "int8"})
        cfg = merged.get("quantization_config")
        assert isinstance(cfg, BitsAndBytesConfig)
        assert cfg.load_in_8bit is True

    def test_no_quantization_key_leaves_kwargs_clean(self):
        merged = self._prepare({}, {})
        assert "quantization_config" not in merged

    def test_none_preset_does_not_inject_key(self):
        merged = self._prepare({}, {"QUANTIZATION": "none"})
        assert "quantization_config" not in merged

    def test_explicit_algo_kwarg_wins_over_init_hp(self):
        explicit = BitsAndBytesConfig(load_in_4bit=True)
        merged = self._prepare(
            {"quantization_config": explicit}, {"QUANTIZATION": "int8"}
        )
        assert merged["quantization_config"] is explicit

    def test_activation_offload_resolved_from_init_hp(self):
        merged = self._prepare({}, {"ACTIVATION_OFFLOAD": True})
        assert merged["activation_offload"] is True

    def test_activation_offload_absent_leaves_kwargs_clean(self):
        merged = self._prepare({}, {})
        assert "activation_offload" not in merged

    def test_explicit_activation_offload_kwarg_wins(self):
        merged = self._prepare(
            {"activation_offload": False}, {"ACTIVATION_OFFLOAD": True}
        )
        assert merged["activation_offload"] is False


class TestActivationOffloadContext:
    """`LLMAlgorithm._activation_offload_ctx` gates ``save_on_cpu``.

    Offload only matters for tensors saved for backward, so the context is a
    no-op when offload is disabled or grads are inactive (rollout / reference
    forwards). Called as an unbound method on a stub to avoid constructing a
    full LLMAlgorithm (no model load, no CUDA).
    """

    def _ctx(self, enabled):
        stub = types.SimpleNamespace(activation_offload=enabled)
        return LLMAlgorithm._activation_offload_ctx(stub)

    def test_disabled_does_not_call_save_on_cpu(self):
        with mock.patch("torch.autograd.graph.save_on_cpu") as m:
            with self._ctx(False):
                pass
        m.assert_not_called()

    def test_enabled_calls_save_on_cpu_with_pinned_memory(self):
        with mock.patch("torch.autograd.graph.save_on_cpu") as m:
            with self._ctx(True):
                pass
        m.assert_called_once_with(pin_memory=True)

    def test_enabled_under_no_grad_does_not_call_save_on_cpu(self):
        with mock.patch("torch.autograd.graph.save_on_cpu") as m:
            with torch.no_grad(), self._ctx(True):
                pass
        m.assert_not_called()


class TestVLLMConfigDefaults:
    def test_quantization_defaults_off(self):
        cfg = VLLMConfig()
        assert cfg.quantization is None
        assert cfg.vllm_model_name_or_path is None
        assert cfg.max_lora_rank == 16
        assert cfg.max_loras == 1
        assert cfg.kv_cache_dtype is None

    def test_lora_fields_can_be_set(self):
        cfg = VLLMConfig(
            max_lora_rank=32,
            max_loras=2,
            vllm_model_name_or_path="TheBloke/Llama-2-7B-AWQ",
            quantization="awq",
        )
        assert cfg.max_lora_rank == 32
        assert cfg.max_loras == 2
        assert cfg.vllm_model_name_or_path == "TheBloke/Llama-2-7B-AWQ"
        assert cfg.quantization == "awq"

    def test_kv_cache_dtype_defaults_to_none(self):
        cfg = VLLMConfig()
        assert cfg.kv_cache_dtype is None

    def test_kv_cache_dtype_passthrough(self):
        # AgileRL forwards kv_cache_dtype verbatim to vLLM without curating
        # values — vLLM emits its own hardware errors on bad values.
        for dtype in ("fp8", "fp8_e4m3", "fp8_e5m2", "auto"):
            cfg = VLLMConfig(kv_cache_dtype=dtype)
            assert cfg.kv_cache_dtype == dtype


# These tests touch the LLMAlgorithm constructor with quantization paths.
# Real instantiation requires vllm import-time wiring; we exercise
# only the bnb-skip-list logic via a stripped-down helper that mirrors the
# production block in base.py. Keeping the algo constructor out of scope lets
# these tests run quickly without GPU.


class TestLmHeadAutoSkipLogic:
    """Mirror of the lm_head skip-modules block from LLMAlgorithm.__init__.

    The production logic lives at agilerl/algorithms/core/base.py around the
    'Auto-skip lm_head from bnb quantization' comment. This test reproduces
    that block in isolation so we can assert its behaviour without paying the
    cost of constructing a full LLMAlgorithm.
    """

    @staticmethod
    def _apply_skip(
        qc: BitsAndBytesConfig, use_liger: bool, use_fused: bool
    ) -> BitsAndBytesConfig:
        if qc is not None and (use_liger or use_fused):
            skip = list(getattr(qc, "llm_int8_skip_modules", None) or [])
            if "lm_head" not in skip:
                skip.append("lm_head")
                qc.llm_int8_skip_modules = skip
        return qc

    def test_liger_adds_lm_head(self):
        qc = BitsAndBytesConfig(load_in_8bit=True)
        out = self._apply_skip(qc, use_liger=True, use_fused=False)
        assert "lm_head" in (out.llm_int8_skip_modules or [])

    def test_fused_logprobs_adds_lm_head(self):
        qc = BitsAndBytesConfig(load_in_4bit=True)
        out = self._apply_skip(qc, use_liger=False, use_fused=True)
        assert "lm_head" in (out.llm_int8_skip_modules or [])

    def test_neither_does_not_add_lm_head(self):
        qc = BitsAndBytesConfig(load_in_8bit=True)
        out = self._apply_skip(qc, use_liger=False, use_fused=False)
        assert "lm_head" not in (out.llm_int8_skip_modules or [])

    def test_does_not_duplicate_existing_lm_head(self):
        qc = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"])
        out = self._apply_skip(qc, use_liger=True, use_fused=False)
        assert (out.llm_int8_skip_modules or []).count("lm_head") == 1

    def test_preserves_existing_skip_modules(self):
        qc = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_skip_modules=["embed_tokens"]
        )
        out = self._apply_skip(qc, use_liger=True, use_fused=False)
        skip = out.llm_int8_skip_modules or []
        assert "embed_tokens" in skip
        assert "lm_head" in skip


class TestJsonSafeValue:
    def test_converts_sets_for_adapter_config_dump(self):
        payload = _json_safe_value({"target_modules": {"q_proj", "k_proj"}, "r": 16})
        json.dumps(payload)
        assert payload["target_modules"] == ["k_proj", "q_proj"]


class TestFilterPeftStateDictForVllmLora:
    def test_keeps_only_tensors_matching_target_modules(self):
        target = build_scoped_lora_target_regex(
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            "language_model",
        )
        state = {
            "base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(
                1
            ),
            "base_model.model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.lora_A.weight": torch.zeros(
                1
            ),
        }
        filtered = filter_peft_state_dict_for_vllm_lora(state, target)
        assert len(filtered) == 1
        assert "language_model" in next(iter(filtered))

    def test_remaps_clippable_linear_suffix(self):
        key = "base_model.model.language_model.layers.0.self_attn.q_proj.linear.lora_A.weight"
        assert (
            remap_peft_lora_key_for_vllm(key)
            == "base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        )


class TestResolveVllmMaxLoraRank:
    def test_takes_max_of_config_and_trainer(self):
        assert resolve_vllm_max_lora_rank(16, 32) == 32
        assert resolve_vllm_max_lora_rank(64, 16) == 64


class TestConfigureVllmKwargs:
    """Verify build_vllm_llm_init_kwargs wires VLLMConfig into LLM()."""

    def test_default_config_enables_lora(self):
        cfg = VLLMConfig()
        kwargs = build_vllm_llm_init_kwargs(
            cfg, trainer_model_name_or_path="trainer/model", max_model_len=1024
        )
        assert kwargs["model"] == "trainer/model"
        assert kwargs["enable_lora"] is True
        assert kwargs["max_lora_rank"] == 16
        assert kwargs["max_loras"] == 1
        assert "quantization" not in kwargs
        # Default kv_cache_dtype is None -> the builder omits the key entirely.
        assert "kv_cache_dtype" not in kwargs

    def test_kv_cache_dtype_injected_when_overridden(self):
        cfg = VLLMConfig(kv_cache_dtype="fp8")
        kwargs = build_vllm_llm_init_kwargs(
            cfg, trainer_model_name_or_path="trainer/model", max_model_len=1024
        )
        assert kwargs["kv_cache_dtype"] == "fp8"

    def test_decoupled_quant_rollout_injects_lora_and_quantization(self):
        cfg = VLLMConfig(
            quantization="awq",
            vllm_model_name_or_path="TheBloke/Llama-2-7B-AWQ",
            max_lora_rank=32,
            max_loras=4,
        )
        kwargs = build_vllm_llm_init_kwargs(
            cfg,
            trainer_model_name_or_path="trainer/llama-7b",
            max_model_len=2048,
            lora_rank=16,
        )
        assert kwargs["model"] == "TheBloke/Llama-2-7B-AWQ"
        assert kwargs["quantization"] == "awq"
        assert kwargs["enable_lora"] is True
        assert kwargs["max_lora_rank"] == 32
        assert kwargs["max_loras"] == 4

    def test_vllm_model_fallback_to_trainer_model(self):
        cfg = VLLMConfig(quantization=None)
        kwargs = build_vllm_llm_init_kwargs(
            cfg,
            trainer_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=1024,
        )
        assert kwargs["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert kwargs["enable_lora"] is True

    def test_trainer_rank_bumps_max_lora_rank(self):
        cfg = VLLMConfig(max_lora_rank=8)
        kwargs = build_vllm_llm_init_kwargs(
            cfg,
            trainer_model_name_or_path="m",
            max_model_len=512,
            lora_rank=32,
        )
        assert kwargs["max_lora_rank"] == 32

    def test_max_num_batched_tokens_not_full_cartesian_product(self):
        cfg = VLLMConfig(max_num_seqs=8)
        kwargs = build_vllm_llm_init_kwargs(
            cfg,
            trainer_model_name_or_path="m",
            max_model_len=32768,
        )
        assert kwargs["max_num_batched_tokens"] == 65536
        assert kwargs["max_num_batched_tokens"] < 8 * 32768

    def test_resolve_vllm_max_num_batched_tokens_explicit(self):
        assert resolve_vllm_max_num_batched_tokens(8, 32768, 4096) == 4096


class TestBuildVllmRolloutLoraRequest:
    def test_builds_request_with_path(self, tmp_path):
        pytest.importorskip("vllm", reason="requires vllm")
        req = build_vllm_rollout_lora_request(tmp_path, load_inplace=True)
        assert req.lora_name == "actor"
        assert req.lora_int_id == 1
        assert req.lora_path == str(tmp_path)
        assert req.load_inplace is True


class TestOffloadColocatedTrainerFromGpu:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_forces_cpu_even_when_first_param_on_cpu(self):
        """Regression: move_params_to_cpu skips when param[0] is CPU but others are not."""
        first = torch.nn.Parameter(torch.zeros(2, device="cpu"))
        second = torch.nn.Parameter(torch.ones(2, device="cuda"))
        model = torch.nn.Module()
        model.register_parameter("first", first)
        model.register_parameter("second", second)

        remaining = offload_colocated_trainer_from_gpu(model)

        assert remaining == 0
        assert cuda_tensor_bytes_in_module(model) == 0
        assert model.second.device.type == "cpu"


class TestColocatedInitOrdering:
    """Colocated vLLM and the trainer each hold their own base.

    A fresh bnb trainer under sleep mode is built FIRST (then offloaded to CPU)
    before vLLM starts, to avoid post-vLLM bnb CUDA-allocator segfaults; every
    other case is CUDA-safe vLLM-first. There is no shared base, and a
    user-supplied ``base_model`` is used directly by ``_initialize_actors``.
    """

    @staticmethod
    def _stub_agent(**kwargs: object) -> LLMAlgorithm:
        # LLMAlgorithm is abstract (get_action/learn/test). Build a concrete
        # no-op subclass and bypass __init__ so we can unit-test the
        # colocated-init flow in isolation. ``object.__new__`` on the abstract
        # class itself raises, so the concrete subclass is required.
        class _ConcreteLLMAlgorithm(LLMAlgorithm):
            def get_action(self, *args: object, **kwargs: object) -> None:
                raise NotImplementedError

            def learn(self, *args: object, **kwargs: object) -> None:
                raise NotImplementedError

            def test(self, *args: object, **kwargs: object) -> None:
                raise NotImplementedError

        agent = object.__new__(_ConcreteLLMAlgorithm)
        agent.use_vllm = kwargs.get("use_vllm", True)
        agent.vllm_config = kwargs.get("vllm_config", VLLMConfig(sleep_mode=True))
        agent.quantization_config = kwargs.get("quantization_config")
        agent.accelerator = kwargs.get("accelerator")
        return agent

    def test_trainer_first_for_fresh_bnb_trainer_under_sleep_mode(self):
        # bnb trainer + sleep_mode + no base_model: the trainer is built first,
        # offloaded to CPU, then vLLM starts. ``_initialize_actors`` builds the
        # trainer's own base (base_model None).
        agent = self._stub_agent(quantization_config=object())
        assert agent._trainer_should_load_before_vllm(None) is True
        call_order: list[str] = []

        def _actors(base, _add):
            call_order.append("actors" if base is None else "actors_wrong")

        with (
            mock.patch.object(
                agent, "_configure_vllm", side_effect=lambda: call_order.append("vllm")
            ),
            mock.patch.object(agent, "_initialize_actors", side_effect=_actors),
            mock.patch.object(
                agent,
                "_offload_trainer_to_cpu_for_colocated_vllm",
                side_effect=lambda: call_order.append("offload"),
            ),
        ):
            agent._initialize_colocated_vllm_and_actors(None, True)

        assert call_order == ["actors", "offload", "vllm"]

    def test_vllm_first_for_dense_trainer(self):
        # No quantization_config: CUDA-safe vLLM-first. vLLM is configured
        # before the trainer actors are built, and no CPU offload runs.
        agent = self._stub_agent(quantization_config=None)
        assert agent._trainer_should_load_before_vllm(None) is False
        call_order: list[str] = []

        with (
            mock.patch.object(
                agent, "_configure_vllm", side_effect=lambda: call_order.append("vllm")
            ),
            mock.patch.object(
                agent,
                "_initialize_actors",
                side_effect=lambda base, _add: call_order.append("actors"),
            ),
            mock.patch.object(
                agent, "_offload_trainer_to_cpu_for_colocated_vllm"
            ) as offload,
        ):
            agent._initialize_colocated_vllm_and_actors(None, True)

        assert call_order == ["vllm", "actors"]
        offload.assert_not_called()

    def test_vllm_first_when_sleep_mode_off(self):
        # sleep_mode off disables the trainer-first ordering even with bnb.
        agent = self._stub_agent(
            vllm_config=VLLMConfig(sleep_mode=False), quantization_config=object()
        )
        assert agent._trainer_should_load_before_vllm(None) is False

    def test_user_supplied_base_model_is_used_directly(self):
        # A non-None base_model (clone / in-memory actor) skips the trainer-first
        # ordering and is passed straight through to ``_initialize_actors``.
        agent = self._stub_agent(quantization_config=object())
        supplied = mock.MagicMock()
        assert agent._trainer_should_load_before_vllm(supplied) is False

        with (
            mock.patch.object(agent, "_configure_vllm"),
            mock.patch.object(agent, "_initialize_actors") as actors,
            mock.patch.object(
                agent, "_offload_trainer_to_cpu_for_colocated_vllm"
            ) as offload,
        ):
            agent._initialize_colocated_vllm_and_actors(supplied, False)

        actors.assert_called_once_with(supplied, False)
        offload.assert_not_called()


class TestPrepareVllmForGenerationOffload:
    """The trainer-base offload runs every call, but its snapshot log (and the
    synchronize/empty_cache inside ``move_params_to_cpu``) fire only when the
    base actually moved — once per engine wake in practice, not once per turn.
    """

    @staticmethod
    def _stub_agent(sleep_mode: bool = True) -> LLMAlgorithm:
        agent = TestColocatedInitOrdering._stub_agent(
            vllm_config=VLLMConfig(sleep_mode=sleep_mode)
        )
        agent.use_memory_efficient_params = True
        agent._vllm_awake = not sleep_mode
        agent._vllm_moved = False
        agent.llm = mock.MagicMock()
        return agent

    def test_log_fires_only_when_base_actually_moves(self):
        agent = self._stub_agent()
        with (
            mock.patch(
                "agilerl.algorithms.core.base.move_params_to_cpu"
            ) as move_to_cpu,
            mock.patch(
                "agilerl.algorithms.core.base.log_cuda_memory_snapshot"
            ) as log_snapshot,
            mock.patch("torch.cuda.empty_cache"),
            mock.patch.object(agent, "_get_unwrapped_actor"),
            mock.patch.object(agent, "_sync_actor_to_vllm"),
        ):
            move_to_cpu.side_effect = [True, False, False]
            agent._prepare_vllm_for_generation()
            assert log_snapshot.call_count == 2  # offload + wake snapshots
            assert agent._vllm_awake is True

            # Mid-rollout turns: base already parked, so no further logs.
            agent._prepare_vllm_for_generation()
            agent._prepare_vllm_for_generation()
            assert move_to_cpu.call_count == 3
            assert log_snapshot.call_count == 2
            agent.llm.wake_up.assert_called_once()

    def test_offload_still_runs_with_sleep_mode_off(self):
        # sleep_mode off initializes _vllm_awake=True; the trainer base must
        # still be parked before the first rollout.
        agent = self._stub_agent(sleep_mode=False)
        with (
            mock.patch(
                "agilerl.algorithms.core.base.move_params_to_cpu",
                return_value=True,
            ) as move_to_cpu,
            mock.patch(
                "agilerl.algorithms.core.base.log_cuda_memory_snapshot"
            ) as log_snapshot,
            mock.patch.object(agent, "_get_unwrapped_actor"),
            mock.patch.object(agent, "_sync_actor_to_vllm"),
        ):
            agent._prepare_vllm_for_generation()

        move_to_cpu.assert_called_once()
        assert log_snapshot.call_count == 1
        agent.llm.wake_up.assert_not_called()


class TestAdaptLoraConfigForClippableLinear:
    @staticmethod
    def _gemma4_style_block():
        class Gemma4ClippableLinear(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias=False)

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = Gemma4ClippableLinear(8, 8)
                self.k_proj = Gemma4ClippableLinear(8, 8)
                self.o_proj = Gemma4ClippableLinear(8, 8)

        return Block()

    def test_no_op_on_plain_linear_model(self):
        from peft import LoraConfig

        model = torch.nn.Module()
        model.layer = torch.nn.Linear(4, 4)
        cfg = LoraConfig(r=4, target_modules=["layer"])
        assert adapt_lora_config_for_model(model, cfg) is cfg

    def test_all_linear_becomes_suffix_targets(self):
        from peft import LoraConfig

        model = self._gemma4_style_block()
        cfg = LoraConfig(r=4, target_modules="all-linear")
        adapted = adapt_lora_config_for_model(model, cfg)
        # peft normalizes target_modules to a set, so compare order-insensitively
        # against the canonical (sorted) suffix list.
        assert sorted(adapted.target_modules) == (
            build_clippable_linear_lora_target_suffixes(["k_proj", "o_proj", "q_proj"])
        )
        assert sorted(list_peft_matched_module_keys(model, adapted.target_modules)) == [
            "k_proj.linear",
            "o_proj.linear",
            "q_proj.linear",
        ]

    def test_explicit_proj_names_become_suffix_targets(self):
        from peft import LoraConfig

        model = self._gemma4_style_block()
        cfg = LoraConfig(r=4, target_modules=["q_proj", "o_proj"])
        adapted = adapt_lora_config_for_model(model, cfg)
        assert sorted(adapted.target_modules) == ["o_proj.linear", "q_proj.linear"]
        # A targeted projection's inner .linear matches; the wrapper alone does not.
        assert peft_target_key_matches(
            "model.layers.0.self_attn.q_proj.linear", adapted.target_modules
        )
        assert not peft_target_key_matches(
            "model.layers.0.self_attn.q_proj", adapted.target_modules
        )

    def test_nested_wrapper_matches_via_suffix_list(self):
        from peft import LoraConfig

        class Gemma4ClippableLinear(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias=False)

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = torch.nn.Module()
                self.attn.q_proj = Gemma4ClippableLinear(8, 8)

        model = Block()
        cfg = LoraConfig(r=4, target_modules="all-linear")
        adapted = adapt_lora_config_for_model(model, cfg)
        assert sorted(adapted.target_modules) == ["q_proj.linear"]
        assert list_peft_matched_module_keys(model, adapted.target_modules) == [
            "attn.q_proj.linear"
        ]

    def test_unscoped_regex_fullmatches_nested_keys(self):
        import re

        pattern = build_clippable_linear_lora_target_regex(["q_proj"])
        assert re.fullmatch(pattern, "model.layers.0.self_attn.q_proj.linear")
        assert re.fullmatch(pattern, "q_proj.linear")
        assert re.fullmatch(pattern, "model.layers.0.self_attn.q_proj") is None

    def test_user_regex_is_left_unchanged(self):
        from peft import LoraConfig

        model = self._gemma4_style_block()
        custom = r".*\.language_model.*\.(q_proj|k_proj)\.linear"
        cfg = LoraConfig(r=4, target_modules=custom)
        assert adapt_lora_config_for_model(model, cfg) is cfg

    def test_language_model_scope_only_when_all_inners_under_scope(self):
        import re

        from peft import LoraConfig

        # Realistic Gemma-4 layout: the language model is nested under ``model``
        # and uses plain nn.Linear projections (ClippableLinear is only on the
        # vision/audio towers). The scoped regex requires the scope to be nested.
        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleList(
                    [torch.nn.Module()]
                )
                self.model.language_model.layers[0].q_proj = torch.nn.Linear(
                    4, 4, bias=False
                )

        model = Block()
        cfg = LoraConfig(r=4, target_modules=["q_proj"])
        adapted = adapt_lora_config_for_model(
            model, cfg, lora_target_scope="language_model"
        )
        scoped = build_scoped_lora_target_regex(["q_proj"], "language_model")
        assert isinstance(adapted.target_modules, str)
        assert adapted.target_modules == scoped
        assert re.fullmatch(scoped, "model.language_model.layers.0.q_proj")
        assert list_peft_matched_module_keys(model, adapted.target_modules) == [
            "model.language_model.layers.0.q_proj"
        ]

    def test_scoped_regex_targets_plain_lm_and_not_vision_clippable(self):
        import re

        from peft import LoraConfig

        class Gemma4ClippableLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, bias=False)

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                layer.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
                layer.mlp = torch.nn.Module()
                layer.mlp.gate_proj = torch.nn.Linear(4, 8, bias=False)
                self.model.language_model.layers = torch.nn.ModuleList([layer])
                vt_layer = torch.nn.Module()
                vt_layer.self_attn = torch.nn.Module()
                vt_layer.self_attn.q_proj = Gemma4ClippableLinear()
                self.model.vision_tower = torch.nn.Module()
                self.model.vision_tower.encoder = torch.nn.Module()
                self.model.vision_tower.encoder.layers = torch.nn.ModuleList([vt_layer])

        model = Block()
        cfg = LoraConfig(r=4, target_modules=["q_proj", "gate_proj"])
        adapted = adapt_lora_config_for_model(
            model, cfg, lora_target_scope="language_model"
        )
        matched = list_peft_matched_module_keys(model, adapted.target_modules)
        assert matched == [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.mlp.gate_proj",
        ]
        assert not any("vision_tower" in key for key in matched)
        scoped = build_scoped_lora_target_regex(["q_proj"], "language_model")
        assert re.fullmatch(scoped, "model.language_model.layers.0.self_attn.q_proj")
        assert (
            re.fullmatch(
                scoped, "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear"
            )
            is None
        )


def _require_bf16_cuda() -> None:
    """Skip when no bf16-capable CUDA device is usable.

    The shipped presets quantize with bf16 compute/storage, so pre-Ampere
    GPUs are out of scope rather than a failure. Mirrors the guarded probe in
    ``tests/test_algorithms/test_llms/conftest.py`` (``is_bf16_supported``
    raises when the driver is loaded but no device is visible).
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("bnb preset tests need a CUDA device.")
    try:
        bf16 = torch.cuda.is_bf16_supported()
    except RuntimeError:
        bf16 = False
    if not bf16:
        pytest.skip("bnb preset tests need a bf16-capable CUDA device.")


def _assert_finite_logits_forward(model: torch.nn.Module, vocab_size: int) -> None:
    device = next(model.parameters()).device
    input_ids = torch.randint(0, vocab_size, (2, 8), device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    assert logits.shape == (2, 8, vocab_size)
    assert torch.isfinite(logits.float()).all()


@pytest.mark.gpu
class TestQuantizedModelLoadOnGpu:
    """Real bnb loads of the tiny fixture: quantization must actually happen.

    The wiring tests above only check that configs are *forwarded*; these load
    the checkpoint with each preset and assert the linear projections were
    really replaced by bnb modules, the packed weights live on GPU, and the
    quantized model still produces finite logits.
    """

    _PROJECTIONS = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @staticmethod
    def _load(preset):
        return create_model_from_name_or_path(
            TINY_LLM_FIXTURE_PATH,
            model_config={"quantization_config": build_bnb_quantization_config(preset)},
        )

    def test_nf4_preset_quantizes_projections(self):
        _require_bf16_cuda()
        model = self._load("nf4")

        quantized_leaves = {
            name.rsplit(".", 1)[-1]
            for name, mod in model.named_modules()
            if isinstance(mod, bnb.nn.Linear4bit)
        }
        assert quantized_leaves.issuperset(self._PROJECTIONS)

        q_proj = model.model.layers[0].self_attn.q_proj
        assert isinstance(q_proj.weight, bnb.nn.Params4bit)
        assert q_proj.weight.quant_state is not None
        assert q_proj.weight.device.type == "cuda"
        # Packed 4-bit storage holds fewer stored elements than logical weights.
        assert q_proj.weight.numel() < q_proj.in_features * q_proj.out_features

        # lm_head stays dense: the fused logprob paths run it unquantized.
        assert not isinstance(model.lm_head, bnb.nn.Linear4bit)
        _assert_finite_logits_forward(model, model.config.vocab_size)

    def test_int8_preset_quantizes_projections(self):
        _require_bf16_cuda()
        model = self._load("int8")

        quantized_leaves = {
            name.rsplit(".", 1)[-1]
            for name, mod in model.named_modules()
            if isinstance(mod, bnb.nn.Linear8bitLt)
        }
        assert quantized_leaves.issuperset(self._PROJECTIONS)

        q_proj = model.model.layers[0].self_attn.q_proj
        assert q_proj.weight.dtype == torch.int8
        assert q_proj.weight.device.type == "cuda"

        assert not isinstance(model.lm_head, bnb.nn.Linear8bitLt)
        _assert_finite_logits_forward(model, model.config.vocab_size)


@pytest.mark.gpu
class TestReinforceQuantizedInit:
    """Algorithm-level nf4 (the decoupled-trainer QLoRA path, no vLLM)."""

    def test_nf4_base_with_lora_adapters_and_dense_lm_head(self):
        _require_bf16_cuda()
        agent = REINFORCE(
            model_name=TINY_LLM_FIXTURE_PATH,
            pad_token_id=151643,
            pad_token="<|endoftext|>",
            lora_config=LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
                lora_dropout=0.0,
            ),
            quantization_config=build_bnb_quantization_config("nf4"),
            device="cuda",
            batch_size=2,
            micro_batch_size_per_gpu=1,
            max_output_tokens=8,
            max_model_len=32,
            use_liger_loss=False,
            wrap=False,
        )

        # __init__ forces lm_head into the bnb skip list (fused logprob paths
        # run the head matmul outside the quantized forward).
        assert "lm_head" in agent.quantization_config.llm_int8_skip_modules
        assert type(agent._get_lm_head()).__name__ not in (
            "Linear4bit",
            "Linear8bitLt",
        )

        # The PEFT-wrapped actor sits on a genuinely 4-bit base. NB: the actor
        # is a DummyEvolvable, whose ``modules()`` is the EvolvableModule
        # registry API, not torch's recursive walk — use ``named_modules()``.
        assert any(
            isinstance(m, bnb.nn.Linear4bit) for _, m in agent.actor.named_modules()
        )
        # ... with LoRA adapters attached to the quantized projections.
        lora_param_names = [
            name for name, _ in agent.actor.named_parameters() if "lora_A" in name
        ]
        assert lora_param_names
        assert all("q_proj" in name or "v_proj" in name for name in lora_param_names)

        _assert_finite_logits_forward(
            agent.actor, agent._get_unwrapped_actor().config.vocab_size
        )
