# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for FSDP2/DTensor LoRA merge to Hugging Face format."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from peft.tuners.lora.layer import LoraLayer, ParamWrapper
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    GenerationMixin,
    GPT2Config,
    GPT2LMHeadModel,
    GraniteMoeHybridConfig,
    GraniteMoeHybridForCausalLM,
    NemotronHConfig,
    NemotronHForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from agilerl.distributed import build_parallel_mesh, gather_params
from agilerl.distributed.expert_parallel import apply_expert_parallel
from agilerl.distributed.tensor_parallel import (
    apply_tensor_parallel,
    tensor_parallel_row_orders,
)
from agilerl.lora.moe import (
    RoutedExpertsLoraWrapper,
    SortedExpertsLoraWrapper,
    TransposedExpertsLoraWrapper,
    moe_expert_target_parameters,
    upgrade_moe_param_wrappers,
)
from agilerl.utils import lora_merge as lora_merge_mod
from agilerl.utils.llm_utils import (
    adapt_lora_config_for_model,
    directory_opener,
    get_lora_params,
    load_lora_adapters,
    snapshot_lora_adapters,
    vllm_lora_state_dict,
    write_lora_adapters,
)
from agilerl.utils.lora_merge import (
    ARENA_ARTIFACT_MANIFEST_FILENAME,
    MERGED_COMPLETE_MARKER,
    BaseWeightStore,
    MergedExportError,
    MergedSyncSpec,
    ModuleCopy,
    ParamLoraMerged,
    export_merged_pretrained,
    merged_sync_specs,
    write_merged_completeness_files,
    write_merged_weight,
)
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from tests.rank_launch import run_ranks

# Reloaded float32 logits vs live PEFT; merge is an affine add of the LoRA delta.
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5
# fp16 / bf16 write adds cast error on top of the same affine add.
FP16_LOGIT_ATOL = 2e-3
BF16_LOGIT_ATOL = 2e-2


def _tiny_gpt2() -> GPT2LMHeadModel:
    """Build a CPU GPT-2 small enough for unit tests."""
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_inner=32,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


def _lora_config(
    target_modules: list[str] | None = None,
    fan_in_fan_out: bool = True,
) -> LoraConfig:
    """LoRA config with nonzero init so merge is not a no-op."""
    return LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=target_modules or ["c_attn"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        init_lora_weights=False,
        fan_in_fan_out=fan_in_fan_out,
    )


def _tiny_opt() -> OPTForCausalLM:
    """Build a CPU OPT small enough for unit tests."""
    return OPTForCausalLM(
        OPTConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            ffn_dim=32,
            num_attention_heads=2,
            max_position_embeddings=16,
            word_embed_proj_dim=16,
        )
    )


def _tiny_qwen3_5_vl() -> nn.Module:
    """Build a CPU Qwen3.5 VL model: one linear-attention and one full-attention layer."""
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "layer_types": ["linear_attention", "full_attention"],
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
        },
        tie_word_embeddings=True,
    )
    return AutoModelForImageTextToText.from_config(config)


def _tiny_peft(**lora_kwargs: Any) -> nn.Module:
    """PEFT-wrapped tiny GPT-2 with adapter name ``actor``."""
    torch.manual_seed(0)
    return get_peft_model(
        _tiny_gpt2(), _lora_config(**lora_kwargs), adapter_name="actor"
    )


def _snapshot_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU clones of every parameter for mutation checks."""
    return {
        name: param.detach().cpu().clone() for name, param in model.named_parameters()
    }


def _assert_unmutated(model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    """Assert every parameter matches the snapshot."""
    after = _snapshot_tensors(model)
    assert after.keys() == before.keys()
    for name, tensor in after.items():
        assert torch.equal(tensor, before[name]), name


def _assert_arena_completeness(merged: Path) -> dict[str, object]:
    """Assert Arena inspect_merged files exist and hashes match file bytes."""
    marker = merged / MERGED_COMPLETE_MARKER
    manifest_path = merged / ARENA_ARTIFACT_MANIFEST_FILENAME
    assert marker.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["format"] == "hf_merged"
    files = payload["files"]
    assert files
    keys = [entry["key"] for entry in files]
    assert "adapter_config.json" not in keys
    for entry in files:
        rel = entry["key"]
        assert not rel.startswith("/")
        assert ".." not in Path(rel).parts
        data = (merged / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert entry["bytes"] == len(data)
    return payload


def _peft_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward logits from a PEFT or HF causal LM."""
    model.eval()
    with torch.no_grad():
        return model(input_ids).logits.float()


class TestExportMergedPretrainedLive:
    def test_reload_logits_match_adapter_plus_base(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        before = _snapshot_tensors(peft_model)
        out = tmp_path / "merged"
        input_ids = torch.randint(0, 32, (2, 8))

        export_merged_pretrained(
            out,
            model=peft_model,
            torch_dtype=torch.float32,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(out)

        assert torch.allclose(
            _peft_logits(peft_model, input_ids),
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )
        _assert_unmutated(peft_model, before)
        weights = load_file(out / "model.safetensors")
        assert all("lora" not in key for key in weights)
        assert not (out / "adapter_config.json").exists()
        assert (out / "config.json").exists()
        assert (out / "generation_config.json").exists()
        _assert_arena_completeness(out)

    def test_writes_completeness_files_matching_hashes(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        out = tmp_path / "merged"

        export_merged_pretrained(out, model=peft_model, torch_dtype=torch.float32)

        payload = _assert_arena_completeness(out)
        listed = {entry["key"] for entry in payload["files"]}
        assert "config.json" in listed
        assert "model.safetensors" in listed
        assert MERGED_COMPLETE_MARKER not in listed
        assert ARENA_ARTIFACT_MANIFEST_FILENAME not in listed

    def test_clears_leftover_shards_before_write(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        out = tmp_path / "merged"
        out.mkdir()
        leftover = out / "model-00001-of-00002.safetensors"
        leftover.write_bytes(b"stale")

        export_merged_pretrained(out, model=peft_model, torch_dtype=torch.float32)

        payload = _assert_arena_completeness(out)
        listed = {entry["key"] for entry in payload["files"]}
        assert leftover.name not in listed
        assert not leftover.exists()

    def test_default_dtype_is_bfloat16(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()

        export_merged_pretrained(tmp_path / "merged", model=peft_model)

        weights = load_file(tmp_path / "merged" / "model.safetensors")
        assert next(iter(weights.values())).dtype == torch.bfloat16

    def test_unwraps_value_head_wrapper(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()

        export_merged_pretrained(
            tmp_path / "merged",
            model=AutoModelForCausalLMWithValueHead(peft_model),
            torch_dtype=torch.float32,
        )

        assert (tmp_path / "merged" / "model.safetensors").exists()

    def test_writes_tokenizer(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()

        class FakeTokenizer:
            def save_pretrained(self, path: str | Path) -> None:
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "tokenizer_config.json").write_text(
                    "{}", encoding="utf-8"
                )

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            tokenizer=FakeTokenizer(),
            torch_dtype=torch.float32,
        )

        assert (tmp_path / "merged" / "tokenizer_config.json").exists()

    def test_merges_lora_bias_when_configured(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        layer = next(
            module
            for module in peft_model.get_base_model().modules()
            if isinstance(module, LoraLayer)
        )
        lora_b = layer.lora_B["actor"]
        lora_b.bias = nn.Parameter(torch.ones(lora_b.out_features) * 0.01)
        input_ids = torch.randint(0, 32, (1, 4))

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            torch_dtype=torch.float32,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")

        assert torch.allclose(
            _peft_logits(peft_model, input_ids),
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )

    def test_sharded_write_never_holds_full_state_dict(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        shard_lengths: list[int] = []

        def spy_save(tensors: dict[str, torch.Tensor], path: str) -> None:
            shard_lengths.append(len(tensors))
            save_file(tensors, path)

        with patch("agilerl.utils.lora_merge.save_file", spy_save):
            export_merged_pretrained(
                tmp_path / "merged",
                model=peft_model,
                torch_dtype=torch.float32,
                max_shard_size="1KB",
            )

        assert (tmp_path / "merged" / "model.safetensors.index.json").exists()
        index = json.loads(
            (tmp_path / "merged" / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        n_tensors = len(index["weight_map"])
        assert shard_lengths
        assert max(shard_lengths) < n_tensors
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")
        input_ids = torch.randint(0, 32, (1, 4))
        assert torch.allclose(
            _peft_logits(peft_model, input_ids),
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )

    def test_gathers_one_module_not_the_full_model(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        n_params = sum(1 for _ in peft_model.parameters())
        gather_sizes: list[int] = []

        @contextmanager
        def recording_gather(_root, params):
            gather_sizes.append(len(params))
            yield list(params)

        with patch("agilerl.utils.lora_merge.gather_params", recording_gather):
            export_merged_pretrained(
                tmp_path / "merged",
                model=peft_model,
                torch_dtype=torch.float32,
            )

        assert gather_sizes
        assert all(size < n_params for size in gather_sizes)
        assert any(size > 0 for size in gather_sizes)

    def test_copy_to_cpu_uses_yielded_gather_not_pre_gather_handle(self) -> None:
        shard_tensor = torch.ones(2, 2)
        dense_tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        spec = ModuleCopy("w", shard_tensor)

        @contextmanager
        def yield_dense(*tensors):
            assert tensors[0] is shard_tensor
            yield [dense_tensor]

        with patch("agilerl.utils.lora_merge.materialize_dtensors", yield_dense):
            shard = lora_merge_mod._materialize_shard(
                [spec],
                ["w"],
                {},
                torch_dtype=torch.float32,
                keep_tensors=True,
            )

        assert torch.equal(shard["w"], dense_tensor.cpu())
        assert not torch.equal(shard["w"], shard_tensor.cpu())

    def test_keep_tensors_false_returns_empty_shard(self) -> None:
        spec = ModuleCopy("w", torch.ones(2, 2))

        shard = lora_merge_mod._materialize_shard(
            [spec],
            ["w"],
            {},
            torch_dtype=torch.float32,
            keep_tensors=False,
        )

        assert shard == {}

    def test_copy_to_cpu_owns_storage_for_save_file(self, tmp_path: Path) -> None:
        view = torch.ones((), dtype=torch.float32).expand(4, 4)

        copied = lora_merge_mod._copy_to_cpu(view, torch.float32)

        save_file({"w": copied}, str(tmp_path / "t.safetensors"))
        assert copied.data_ptr() != 0

    def test_copy_to_cpu_gathers_dtensor_shard(self) -> None:
        class FakeDTensor:
            def __init__(self, local: torch.Tensor):
                self._local = local

            def full_tensor(self) -> torch.Tensor:
                return self._local

        dense = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        with patch.object(lora_merge_mod, "DTensor", FakeDTensor):
            copied = lora_merge_mod._copy_to_cpu(FakeDTensor(dense), torch.float32)

        assert torch.equal(copied, dense)
        assert copied.device.type == "cpu"

    def test_embedding_lora_and_non_float_buffer(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft(target_modules=["wte"], fan_in_fan_out=False)
        peft_model.get_base_model().register_buffer(
            "token_type_ids",
            torch.zeros(2, dtype=torch.int64),
        )

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            torch_dtype=torch.float32,
        )

        weights = load_file(tmp_path / "merged" / "model.safetensors")
        assert "token_type_ids" in weights
        assert weights["token_type_ids"].dtype == torch.int64
        assert any("wte.weight" in key for key in weights)
        assert all("lora" not in key for key in weights)

    def test_copies_base_when_layer_lacks_named_adapter(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        peft_model = get_peft_model(
            _tiny_gpt2(),
            _lora_config(target_modules=["c_attn"]),
            adapter_name="actor",
        )
        peft_model.add_adapter("critic", _lora_config(target_modules=["c_proj"]))
        input_ids = torch.randint(0, 32, (1, 4))
        before = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            adapter_name="actor",
            torch_dtype=torch.float32,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")

        assert torch.allclose(
            before,
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )


class TestExportMergedPretrainedVisionLanguage:
    @pytest.mark.parametrize("shape", ["live", "replay"])
    def test_qwen3_5_export_keeps_architecture_and_towers(
        self, tmp_path: Path, shape: str
    ) -> None:
        # Arrange
        torch.manual_seed(0)
        base = _tiny_qwen3_5_vl()
        base.save_pretrained(tmp_path / "base")
        visual = {
            name: tensor.clone()
            for name, tensor in base.state_dict().items()
            if name.startswith("model.visual.")
        }
        lora = adapt_lora_config_for_model(
            base, _lora_config(["q_proj", "in_proj_qkv"], fan_in_fan_out=False)
        )
        peft_model = get_peft_model(base, lora, adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")

        # Act
        if shape == "live":
            export_merged_pretrained(
                tmp_path / "merged",
                model=peft_model,
                torch_dtype=torch.float32,
            )
        else:
            export_merged_pretrained(
                tmp_path / "merged",
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
                torch_dtype=torch.float32,
            )

        # Assert
        config = json.loads((tmp_path / "merged" / "config.json").read_text())
        assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
        assert config["model_type"] == "qwen3_5"
        merged = load_file(tmp_path / "merged" / "model.safetensors")
        assert visual
        assert all(torch.equal(merged[name], tensor) for name, tensor in visual.items())
        assert not any("lora_" in name for name in merged)


class TestWriteMergedCompletenessFiles:
    def test_skips_adapter_config_and_writes_marker_last(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged"
        merged.mkdir()
        (merged / "config.json").write_text("{}", encoding="utf-8")
        (merged / "adapter_config.json").write_text("{}", encoding="utf-8")
        nested = merged / "tokenizer"
        nested.mkdir()
        (nested / "tokenizer.json").write_bytes(b"tok")

        write_merged_completeness_files(merged)

        payload = _assert_arena_completeness(merged)
        keys = [entry["key"] for entry in payload["files"]]
        assert keys == ["config.json", "tokenizer/tokenizer.json"]
        assert (merged / "adapter_config.json").is_file()


class TestExportMergedPretrainedReplay:
    def test_reload_from_actor_dir_and_base(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base_dir = tmp_path / "base"
        base.save_pretrained(base_dir)
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        ckpt = tmp_path / "ckpt"
        peft_model.save_pretrained(ckpt)
        actor_dir = ckpt / "actor"
        input_ids = torch.randint(0, 32, (2, 8))
        live_logits = _peft_logits(peft_model, input_ids)

        out = tmp_path / "merged"
        export_merged_pretrained(
            out,
            adapter_path=actor_dir,
            base_model_name_or_path=base_dir,
            adapter_name="actor",
            torch_dtype=torch.float32,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(out)

        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )
        _assert_arena_completeness(out)

    def test_non_main_replay_skips_write(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        out = tmp_path / "merged"

        with patch("agilerl.utils.lora_merge.is_main_process", return_value=False):
            export_merged_pretrained(
                out,
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
                adapter_name="actor",
                torch_dtype=torch.float32,
            )

        assert not out.exists()

    def test_does_not_call_from_pretrained(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")

        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("replay must not load the full model"),
        ):
            export_merged_pretrained(
                tmp_path / "merged",
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
                adapter_name="actor",
                torch_dtype=torch.float32,
            )

        assert (tmp_path / "merged" / "model.safetensors").exists()

    def test_base_weights_are_meta_after_export(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        skeleton, store = lora_merge_mod._load_replay_peft(
            tmp_path / "ckpt" / "actor",
            tmp_path / "base",
            "actor",
        )
        base_params = [
            param
            for name, param in skeleton.get_base_model().named_parameters()
            if "lora_" not in name
        ]
        assert all(param.device.type == "meta" for param in base_params)
        assert any(
            "lora_" in name and param.device.type == "cpu"
            for name, param in skeleton.named_parameters()
        )

        lora_merge_mod._export_live_model(
            skeleton,
            output_dir=tmp_path / "merged",
            adapter_name="actor",
            max_shard_size="1KB",
            torch_dtype=torch.float32,
            tokenizer=None,
            weight_store=store,
        )

        assert all(param.device.type == "meta" for param in base_params)

    def test_loads_base_tensors_per_shard(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        loads: list[list[str]] = []
        batch: list[str] = []
        orig_load = BaseWeightStore.load
        orig_materialize = lora_merge_mod._materialize_shard

        def spy_load(self: BaseWeightStore, key: str) -> torch.Tensor:
            batch.append(key)
            return orig_load(self, key)

        def spy_materialize(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            batch.clear()
            tensors = orig_materialize(*args, **kwargs)
            loads.append(list(batch))
            return tensors

        with (
            patch.object(BaseWeightStore, "load", spy_load),
            patch.object(lora_merge_mod, "_materialize_shard", spy_materialize),
        ):
            export_merged_pretrained(
                tmp_path / "merged",
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
                adapter_name="actor",
                torch_dtype=torch.float32,
                max_shard_size="1KB",
            )

        store = BaseWeightStore.from_checkpoint(tmp_path / "base")
        assert loads
        assert max(len(keys) for keys in loads) < len(store.key_files)

    def test_sharded_base_checkpoint(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base", max_shard_size="1KB")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        input_ids = torch.randint(0, 32, (1, 4))
        live_logits = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "merged",
            adapter_path=tmp_path / "ckpt" / "actor",
            base_model_name_or_path=tmp_path / "base",
            adapter_name="actor",
            torch_dtype=torch.float32,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")

        assert (tmp_path / "base" / "model.safetensors.index.json").exists()
        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )

    def test_missing_base_safetensors_raises(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        (tmp_path / "base" / "model.safetensors").unlink()
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")

        with pytest.raises(MergedExportError, match="safetensors weights"):
            export_merged_pretrained(
                tmp_path / "merged",
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
                adapter_name="actor",
            )

    def test_missing_adapter_safetensors_raises(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        (tmp_path / "ckpt" / "actor" / "adapter_model.safetensors").unlink()

        with pytest.raises(MergedExportError, match=r"adapter_model\.safetensors"):
            export_merged_pretrained(
                tmp_path / "merged",
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
                adapter_name="actor",
            )

    def test_missing_index_shard_raises(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"w": "missing.safetensors"}}),
            encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError, match="missing shard"):
            BaseWeightStore.from_checkpoint(tmp_path)

    def test_store_load_unknown_key_raises(self) -> None:
        store = BaseWeightStore({})

        with pytest.raises(KeyError, match="no tensor"):
            store.load("missing.weight")

    def test_adapter_leftover_key_raises(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        extra = {"not_a_lora_key": torch.ones(1)}
        with patch(
            "agilerl.utils.lora_merge.load_file",
            return_value=extra,
        ):
            with pytest.raises(ValueError, match="were not on the PEFT model"):
                export_merged_pretrained(
                    tmp_path / "merged",
                    adapter_path=tmp_path / "ckpt" / "actor",
                    base_model_name_or_path=tmp_path / "base",
                    adapter_name="actor",
                )

    def test_empty_adapter_file_raises(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        with patch("agilerl.utils.lora_merge.load_file", return_value={}):
            with pytest.raises(ValueError, match="No adapter tensors matched"):
                export_merged_pretrained(
                    tmp_path / "merged",
                    adapter_path=tmp_path / "ckpt" / "actor",
                    base_model_name_or_path=tmp_path / "base",
                    adapter_name="actor",
                )

    def test_copy_meta_tensor_without_store_raises(self) -> None:
        spec = ModuleCopy("w", torch.empty(2, 2, device="meta"))

        with pytest.raises(RuntimeError, match="without a weight store"):
            lora_merge_mod._materialize_shard(
                [spec],
                ["w"],
                {},
                torch_dtype=torch.float32,
                keep_tensors=True,
            )

    def test_skips_meta_copy_missing_from_checkpoint(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        skeleton, store = lora_merge_mod._load_replay_peft(
            tmp_path / "ckpt" / "actor",
            tmp_path / "base",
            "actor",
        )
        skeleton.get_base_model().register_buffer(
            "not_in_ckpt", torch.empty(2, device="meta")
        )

        specs = lora_merge_mod._collect_export_specs(
            skeleton.get_base_model(), "actor", store
        )

        assert all(spec.hf_name != "not_in_ckpt" for spec in specs)

    def test_merged_checkpoint_names(self) -> None:
        assert lora_merge_mod._merged_checkpoint_names("weight") == ("weight", "bias")
        assert lora_merge_mod._merged_checkpoint_names("bias") == ("weight", "bias")
        assert lora_merge_mod._merged_checkpoint_names("h.0.weight") == (
            "h.0.weight",
            "h.0.bias",
        )
        assert lora_merge_mod._merged_checkpoint_names("h.0.bias") == (
            "h.0.weight",
            "h.0.bias",
        )
        assert lora_merge_mod._merged_checkpoint_names("foo") == ("foo", None)


class TestReplaceTensor:
    def test_replaces_parameter_with_cpu_clone(self) -> None:
        module = nn.Linear(2, 3, bias=False)
        original = module.weight
        new = torch.ones(3, 2)

        previous = lora_merge_mod._replace_tensor(module, "weight", new)

        assert previous is original
        assert isinstance(module.weight, nn.Parameter)
        assert module.weight.device.type == "cpu"
        assert torch.equal(module.weight.detach(), new)
        assert module.weight.requires_grad is True

    def test_replaces_buffer_with_cpu_parameter(self) -> None:
        module = nn.Module()
        original = torch.zeros(4)
        module.register_buffer("scale", original)
        new = torch.arange(4, dtype=torch.float32)

        previous = lora_merge_mod._replace_tensor(module, "scale", new)

        assert previous is original
        assert isinstance(module.scale, nn.Parameter)
        assert module.scale.device.type == "cpu"
        assert torch.equal(module.scale.detach(), new)
        assert module.scale.requires_grad is False


class TestExportMergedPretrainedFailures:
    def test_neither_live_nor_replay_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="either model="):
            export_merged_pretrained(tmp_path)

    def test_both_live_and_replay_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="either model="):
            export_merged_pretrained(
                tmp_path,
                model=_tiny_peft(),
                adapter_path=tmp_path / "actor",
                base_model_name_or_path=tmp_path / "base",
            )

    def test_adapter_path_without_base_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="base_model_name_or_path"):
            export_merged_pretrained(tmp_path, adapter_path=tmp_path / "actor")

    def test_base_without_adapter_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="adapter_path"):
            export_merged_pretrained(
                tmp_path,
                base_model_name_or_path=tmp_path / "base",
            )

    def test_non_peft_model_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="PEFT-wrapped"):
            export_merged_pretrained(tmp_path, model=nn.Linear(4, 4))

    def test_base_model_not_pretrained_raises(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        peft_model.get_base_model = lambda: nn.Linear(2, 2)

        with pytest.raises(TypeError, match="transformers model"):
            export_merged_pretrained(
                tmp_path / "merged",
                model=peft_model,
                torch_dtype=torch.float32,
            )

    def test_non_peft_module_cannot_produce_delta(self) -> None:
        with pytest.raises(TypeError, match="cannot produce a LoRA delta"):
            lora_merge_mod._peft_lora_module(nn.Linear(2, 2))

    def test_missing_gather_raises(self) -> None:
        spec = ModuleCopy("w", torch.ones(2, 2))

        @contextmanager
        def yield_none(*_tensors):
            yield [None]

        with patch("agilerl.utils.lora_merge.materialize_dtensors", yield_none):
            with pytest.raises(RuntimeError, match="is missing"):
                lora_merge_mod._materialize_shard(
                    [spec],
                    ["w"],
                    {},
                    torch_dtype=torch.float32,
                    keep_tensors=True,
                )

    def test_unsupported_lora_base_raises(self) -> None:
        layer = next(
            module
            for module in _tiny_peft().modules()
            if isinstance(module, PeftLoraLinear)
        )
        layer.base_layer = nn.ReLU()

        with pytest.raises(TypeError, match="not a LoRA base module"):
            lora_merge_mod._as_lora_base(layer)

    def test_missing_adapter_name_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Adapter 'critic'"):
            export_merged_pretrained(
                tmp_path,
                model=_tiny_peft(),
                adapter_name="critic",
            )

    def test_failed_write_does_not_mutate_and_raises(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        before = _snapshot_tensors(peft_model)

        with patch(
            "agilerl.utils.lora_merge.save_file",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(MergedExportError, match="disk full"):
                export_merged_pretrained(
                    tmp_path / "merged",
                    model=peft_model,
                    torch_dtype=torch.float32,
                    max_shard_size="1KB",
                )

        _assert_unmutated(peft_model, before)

    def test_reraises_merged_export_error_unchanged(self, tmp_path: Path) -> None:
        with patch(
            "agilerl.utils.lora_merge._export_live_model",
            side_effect=MergedExportError("already wrapped"),
        ):
            with pytest.raises(MergedExportError, match="already wrapped"):
                export_merged_pretrained(tmp_path, model=_tiny_peft())

    def test_failed_collective_does_not_mutate_and_raises(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        before = _snapshot_tensors(peft_model)

        def failing_gather(*_args: object, **_kwargs: object) -> None:
            msg = "collective failed"
            raise RuntimeError(msg)

        with patch("agilerl.utils.lora_merge.gather_params", failing_gather):
            with pytest.raises(MergedExportError, match="collective failed"):
                export_merged_pretrained(
                    tmp_path / "merged",
                    model=peft_model,
                    torch_dtype=torch.float32,
                )

        _assert_unmutated(peft_model, before)

    def test_missing_get_delta_weight_raises(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        layer = next(
            module
            for module in peft_model.get_base_model().modules()
            if isinstance(module, LoraLayer)
        )
        layer.get_delta_weight = None

        with pytest.raises(TypeError, match="cannot produce a LoRA delta"):
            export_merged_pretrained(
                tmp_path / "merged",
                model=peft_model,
                torch_dtype=torch.float32,
            )

    def test_non_main_live_skips_files(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        out = tmp_path / "merged"

        with patch("agilerl.utils.lora_merge.is_main_process", return_value=False):
            export_merged_pretrained(
                out,
                model=peft_model,
                torch_dtype=torch.float32,
            )

        assert not out.exists()


class TestExportMergedPretrainedDistributedFailure:
    @contextmanager
    def _distributed(self, is_main: bool = True, all_reduce: Any = None):
        def _noop_all_reduce(_flag_tensor: torch.Tensor, **_kwargs: Any) -> None:
            pass

        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.utils.lora_merge.is_main_process", return_value=is_main),
            patch(
                "agilerl.distributed.process.dist.all_reduce",
                all_reduce if all_reduce is not None else _noop_all_reduce,
            ),
        ):
            yield

    def test_reduce_propagates_failure_to_every_rank(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        reduced: list[torch.Tensor] = []

        def fake_all_reduce(flag_tensor: torch.Tensor, **_kwargs: Any) -> None:
            reduced.append(flag_tensor.clone())

        with (
            self._distributed(all_reduce=fake_all_reduce),
            patch(
                "agilerl.utils.lora_merge.save_file",
                side_effect=OSError("disk full"),
            ),
        ):
            with pytest.raises(MergedExportError, match="disk full"):
                export_merged_pretrained(
                    tmp_path / "merged",
                    model=peft_model,
                    torch_dtype=torch.float32,
                )

        assert any(int(flag.item()) == 1 for flag in reduced)

    def test_reduce_success_when_no_rank_failed(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()

        def fake_all_reduce(flag_tensor: torch.Tensor, **_kwargs: Any) -> None:
            flag_tensor.zero_()

        with self._distributed(all_reduce=fake_all_reduce):
            export_merged_pretrained(
                tmp_path / "merged",
                model=peft_model,
                torch_dtype=torch.float32,
            )

        assert (tmp_path / "merged" / "model.safetensors").exists()

    def test_reraises_merged_export_error_from_other_rank(self, tmp_path: Path) -> None:
        def fake_all_reduce(flag_tensor: torch.Tensor, **_kwargs: Any) -> None:
            flag_tensor.fill_(1)

        with (
            self._distributed(all_reduce=fake_all_reduce),
            patch(
                "agilerl.utils.lora_merge._export_live_model",
                side_effect=MergedExportError("already wrapped"),
            ),
        ):
            with pytest.raises(MergedExportError, match="already wrapped"):
                export_merged_pretrained(
                    tmp_path / "merged",
                    model=_tiny_peft(),
                )

    def test_save_file_failure_on_first_shard_skips_later_gathers(
        self, tmp_path: Path
    ) -> None:
        peft_model = _tiny_peft()
        gather_entered: list[int] = []
        gathers_after_abort: list[int] = []
        materialize_count = 0
        aborting = False
        orig_materialize = lora_merge_mod._materialize_shard
        n_shards = len(
            lora_merge_mod._plan_shards(
                lora_merge_mod._collect_export_specs(
                    peft_model.get_base_model(), "actor"
                ),
                "1KB",
            )
        )
        assert n_shards > 1

        def spy_materialize(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            nonlocal materialize_count
            materialize_count += 1
            return orig_materialize(*args, **kwargs)

        def fake_all_reduce(flag_tensor: torch.Tensor, **_kwargs: Any) -> None:
            nonlocal aborting
            if materialize_count > 0:
                aborting = True
                flag_tensor.fill_(1)

        @contextmanager
        def recording_gather(_root: Any, params: Any):
            gather_entered.append(len(params))
            if aborting:
                gathers_after_abort.append(len(params))
            yield list(params)

        @contextmanager
        def recording_materialize(*tensors: Any):
            gather_entered.append(len(tensors))
            if aborting:
                gathers_after_abort.append(len(tensors))
            yield list(tensors)

        with (
            self._distributed(is_main=False, all_reduce=fake_all_reduce),
            patch("agilerl.utils.lora_merge._materialize_shard", spy_materialize),
            patch("agilerl.utils.lora_merge.gather_params", recording_gather),
            patch(
                "agilerl.utils.lora_merge.materialize_dtensors", recording_materialize
            ),
        ):
            with pytest.raises(
                MergedExportError, match="Peer rank failed in shard runtime collective"
            ):
                export_merged_pretrained(
                    tmp_path / "merged",
                    model=peft_model,
                    torch_dtype=torch.float32,
                    max_shard_size="1KB",
                )

        assert materialize_count == 1
        assert gather_entered
        assert gathers_after_abort == []

    def test_mkdir_failure_skips_all_gathers(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        blocker = tmp_path / "merged"
        blocker.write_text("not a directory", encoding="utf-8")
        gather_entered: list[int] = []

        def fake_all_reduce(_flag_tensor: torch.Tensor, **_kwargs: Any) -> None:
            return None

        @contextmanager
        def recording_gather(_root: Any, params: Any):
            gather_entered.append(len(params))
            yield list(params)

        @contextmanager
        def recording_materialize(*tensors: Any):
            gather_entered.append(len(tensors))
            yield list(tensors)

        with (
            self._distributed(all_reduce=fake_all_reduce),
            patch("agilerl.utils.lora_merge.gather_params", recording_gather),
            patch(
                "agilerl.utils.lora_merge.materialize_dtensors", recording_materialize
            ),
        ):
            with pytest.raises(MergedExportError):
                export_merged_pretrained(
                    blocker,
                    model=peft_model,
                    torch_dtype=torch.float32,
                    max_shard_size="1KB",
                )

        assert gather_entered == []


class TestExportMergedPretrainedTiedKeys:
    def test_tied_lm_head_is_skipped(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()

        export_merged_pretrained(
            tmp_path / "merged", model=peft_model, torch_dtype=torch.float32
        )

        weights = load_file(tmp_path / "merged" / "model.safetensors")
        assert "lm_head.weight" not in weights
        assert "transformer.wte.weight" in weights

    def test_tied_target_held_as_separate_parameter_is_skipped(
        self, tmp_path: Path
    ) -> None:
        # ``to_empty`` leaves a tied head as its own Parameter until re-tied.
        peft_model = _tiny_peft()
        head = peft_model.get_base_model().lm_head
        head.weight = nn.Parameter(head.weight.detach().clone())

        export_merged_pretrained(
            tmp_path / "merged", model=peft_model, torch_dtype=torch.float32
        )

        weights = load_file(tmp_path / "merged" / "model.safetensors")
        assert "lm_head.weight" not in weights
        assert "transformer.wte.weight" in weights

    def test_untied_lm_head_is_written(self, tmp_path: Path) -> None:
        # Arrange
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.config.tie_word_embeddings = False
        untied = GPT2LMHeadModel(base.config)
        peft_model = get_peft_model(untied, _lora_config(), adapter_name="actor")
        input_ids = torch.randint(0, 32, (2, 8))

        # Act
        export_merged_pretrained(
            tmp_path / "merged", model=peft_model, torch_dtype=torch.float32
        )

        # Assert
        weights = load_file(tmp_path / "merged" / "model.safetensors")
        assert "lm_head.weight" in weights
        assert torch.equal(weights["lm_head.weight"], untied.lm_head.weight)
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")
        assert torch.allclose(
            _peft_logits(peft_model, input_ids),
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )


def _tiny_qwen3_5_gdn_lm() -> PreTrainedModel:
    """Qwen3.5 text LM with one Gated DeltaNet and one attention layer."""
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
    )
    return Qwen3_5ForCausalLM(config)


def _tiny_nemotron_h_lm() -> PreTrainedModel:
    """NemotronH LM with one Mamba2, one attention and one MLP layer."""
    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=16,
        layers_block_type=["mamba", "attention", "mlp"],
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=16,
        use_mamba_kernels=False,
        ssm_state_size=4,
        mamba_num_heads=4,
        mamba_head_dim=4,
        n_groups=2,
        conv_kernel=2,
        chunk_size=8,
        use_bias=True,
        use_conv_bias=True,
        use_cache=False,
    )
    return NemotronHForCausalLM(config)


def _tiny_granitemoehybrid_lm() -> PreTrainedModel:
    """Granite 4 LM with one Mamba2 layer (one SSM group) and one attention layer."""
    config = GraniteMoeHybridConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        shared_intermediate_size=8,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=0,
        position_embedding_type="nope",
        mamba_n_heads=8,
        mamba_d_head=4,
        mamba_n_groups=1,
        mamba_d_state=4,
        mamba_d_conv=4,
        mamba_chunk_size=4,
        tie_word_embeddings=True,
        pad_token_id=0,
    )
    return GraniteMoeHybridForCausalLM(config)


# PEFT refuses LoRA on the Mamba2 out_proj and conv1d.
TP_EXPORT_MODELS = {
    "qwen3_5": (_tiny_qwen3_5_gdn_lm, "all-linear"),
    "nemotron_h": (
        _tiny_nemotron_h_lm,
        ["in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    ),
    "granitemoehybrid": (
        _tiny_granitemoehybrid_lm,
        [
            "in_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "input_linear",
            "output_linear",
        ],
    ),
}


def _tp_export_peft_model(family: str) -> nn.Module:
    build, targets = TP_EXPORT_MODELS[family]
    return get_peft_model(
        build().float(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=targets,
            lora_dropout=0.0,
            init_lora_weights=False,
        ),
        adapter_name="actor",
    )


def _tp_export_worker(rank: int, world_size: int, family: str, out_dir: str) -> int:
    """Export a tp-sharded and an unsharded copy of one seeded PEFT model."""
    torch.manual_seed(0)
    model = _tp_export_peft_model(family)
    reference = copy.deepcopy(model)
    mesh = build_parallel_mesh(world_size=world_size, tp=world_size, device_type="cpu")
    assert mesh is not None
    apply_tensor_parallel(model, mesh.tp)

    export_merged_pretrained(f"{out_dir}/tp", model=model, torch_dtype=torch.float32)
    export_merged_pretrained(
        f"{out_dir}/reference", model=reference, torch_dtype=torch.float32
    )

    return len(tensor_parallel_row_orders(model))


@pytest.mark.multi_rank
class TestExportMergedPretrainedTensorParallel:
    @pytest.mark.parametrize("family", sorted(TP_EXPORT_MODELS))
    def test_writes_the_unsharded_merge(self, tmp_path: Path, family: str) -> None:
        # Act
        permuted = run_ranks(_tp_export_worker, 2, family, str(tmp_path))

        # Assert
        assert permuted[0] > 0
        written = load_file(tmp_path / "tp" / "model.safetensors")
        expected = load_file(tmp_path / "reference" / "model.safetensors")
        assert written.keys() == expected.keys()
        for name, tensor in expected.items():
            assert torch.equal(written[name], tensor), name


def _assert_same_tensors(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), name


def _tp_adapter_worker(rank: int, world_size: int, family: str, out_dir: str) -> None:
    """LoRA-only exports of a tp-sharded model equal the unsharded model's."""
    # Arrange
    torch.manual_seed(0)
    model = _tp_export_peft_model(family)
    reference = copy.deepcopy(model)
    torch.manual_seed(1)
    resumed = _tp_export_peft_model(family)
    mesh = build_parallel_mesh(world_size=world_size, tp=world_size, device_type="cpu")
    assert mesh is not None
    apply_tensor_parallel(model, mesh.tp)
    apply_tensor_parallel(resumed, mesh.tp)
    saved_dir = f"{out_dir}/rank{rank}"
    expected = snapshot_lora_adapters(reference, ["actor"]).adapters["actor"]

    # Act
    snapshot = snapshot_lora_adapters(model, ["actor"])
    with gather_params(model, get_lora_params(model)):
        exported = vllm_lora_state_dict(model, "actor")
    write_lora_adapters(snapshot, directory_opener(saved_dir))
    load_lora_adapters(resumed, saved_dir, "actor")
    reloaded = snapshot_lora_adapters(resumed, ["actor"]).adapters["actor"]

    # Assert
    assert any(".lora_B." in name for name in tensor_parallel_row_orders(model))
    _assert_same_tensors(snapshot.adapters["actor"], expected)
    _assert_same_tensors(exported, vllm_lora_state_dict(reference, "actor"))
    _assert_same_tensors(reloaded, expected)


@pytest.mark.multi_rank
class TestLoraAdapterExportTensorParallel:
    @pytest.mark.parametrize("family", sorted(TP_EXPORT_MODELS))
    def test_adapter_exports_match_the_unsharded_model(
        self, tmp_path: Path, family: str
    ) -> None:
        run_ranks(_tp_adapter_worker, 2, family, str(tmp_path))


MOE_VOCAB = 32
MOE_HIDDEN = 8
MOE_INTERMEDIATE = 8
NUM_EXPERTS = 4
TOP_K = 2


class RoutedExperts(nn.Module):
    """Self-routing packed experts: gated ``gate_up_proj`` (Qwen3-MoE) or ungated ``up_proj`` (NemotronH)."""

    def __init__(self, gated: bool) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.gated = gated
        up_rows = 2 * MOE_INTERMEDIATE if gated else MOE_INTERMEDIATE
        self.register_parameter(
            "gate_up_proj" if gated else "up_proj",
            nn.Parameter(torch.randn(NUM_EXPERTS, up_rows, MOE_HIDDEN) * 0.1),
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, MOE_HIDDEN, MOE_INTERMEDIATE) * 0.1
        )
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        up_weight = self.gate_up_proj if self.gated else self.up_proj
        out = torch.zeros_like(hidden_states)
        for expert in range(self.num_experts):
            token_idx, slot = torch.where(top_k_index == expert)
            projected = F.linear(hidden_states[token_idx], up_weight[expert])
            if self.gated:
                gate, up = projected.chunk(2, dim=-1)
                inner = self.act_fn(gate) * up
            else:
                inner = self.act_fn(projected)
            rows = F.linear(inner, self.down_proj[expert])
            out.index_add_(0, token_idx, rows * top_k_weights[token_idx, slot, None])
        return out


class TransposedExperts(nn.Module):
    """Packed ``x @ W[e] + b[e]`` experts with ``[E, in, out]`` weights (GPT-OSS layout)."""

    is_transposed = True

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.gate_up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, MOE_HIDDEN, 2 * MOE_INTERMEDIATE) * 0.1
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.randn(NUM_EXPERTS, 2 * MOE_INTERMEDIATE) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, MOE_INTERMEDIATE, MOE_HIDDEN) * 0.1
        )
        self.down_proj_bias = nn.Parameter(torch.randn(NUM_EXPERTS, MOE_HIDDEN) * 0.1)
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        out = torch.zeros_like(hidden_states)
        for expert in range(self.num_experts):
            token_idx, slot = torch.where(router_indices == expert)
            gate_up = (
                hidden_states[token_idx] @ self.gate_up_proj[expert]
                + self.gate_up_proj_bias[expert]
            )
            gate = gate_up[..., ::2].clamp(max=self.limit)
            up = gate_up[..., 1::2].clamp(min=-self.limit, max=self.limit)
            inner = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
            rows = inner @ self.down_proj[expert] + self.down_proj_bias[expert]
            out.index_add_(0, token_idx, rows * routing_weights[token_idx, slot, None])
        return out


class SortedExperts(nn.Module):
    """Grouped linear over expert-sorted rows, ``weight`` ``[E, out, in]`` (GraniteMoe ``ParallelExperts``)."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.weight = nn.Parameter(
            torch.randn(NUM_EXPERTS, output_size, input_size) * 0.1
        )

    def forward(self, inputs, expert_size):
        rows = inputs.split(expert_size, dim=0)
        return torch.cat(
            [F.linear(rows[e], self.weight[e]) for e in range(self.num_experts)]
        )


class RoutedMoeBlock(nn.Module):
    """Top-k router feeding a self-routing packed-experts module."""

    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        self.router = nn.Linear(MOE_HIDDEN, NUM_EXPERTS, bias=False)
        self.experts = experts

    def forward(self, hidden_states):
        probs = torch.softmax(self.router(hidden_states), dim=-1)
        top_k_weights, top_k_index = probs.topk(TOP_K, dim=-1)
        return self.experts(hidden_states, top_k_index, top_k_weights)


class SortedMoeBlock(nn.Module):
    """Top-k router, then grouped ``input_linear`` / ``output_linear`` over sorted rows."""

    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(MOE_HIDDEN, NUM_EXPERTS, bias=False)
        self.input_linear = SortedExperts(MOE_HIDDEN, 2 * MOE_INTERMEDIATE)
        self.output_linear = SortedExperts(MOE_INTERMEDIATE, MOE_HIDDEN)

    def forward(self, hidden_states):
        probs = torch.softmax(self.router(hidden_states), dim=-1)
        top_k_weights, top_k_index = probs.topk(TOP_K, dim=-1)
        order = top_k_index.flatten().argsort(stable=True)
        token_idx = order // TOP_K
        expert_size = torch.bincount(
            top_k_index.flatten(), minlength=NUM_EXPERTS
        ).tolist()
        gate, up = self.input_linear(hidden_states[token_idx], expert_size).chunk(
            2, dim=-1
        )
        rows = self.output_linear(F.silu(gate) * up, expert_size)
        rows = rows * top_k_weights.flatten()[order, None]
        return torch.zeros_like(hidden_states).index_add(0, token_idx, rows)


class TinyMoeConfig(PretrainedConfig):
    model_type = "tiny_moe_lora_merge"

    def __init__(self, layout: str = "routed_gated", **kwargs: Any) -> None:
        self.layout = layout
        super().__init__(tie_word_embeddings=False, **kwargs)


class TinyMoeForCausalLM(PreTrainedModel, GenerationMixin):
    """One packed-MoE residual block between an embedding and an LM head."""

    config_class = TinyMoeConfig

    def __init__(self, config: TinyMoeConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(MOE_VOCAB, MOE_HIDDEN)
        blocks = {
            "routed_gated": lambda: RoutedMoeBlock(RoutedExperts(gated=True)),
            "routed_ungated": lambda: RoutedMoeBlock(RoutedExperts(gated=False)),
            "transposed": lambda: RoutedMoeBlock(TransposedExperts()),
            "sorted": SortedMoeBlock,
        }
        self.mlp = blocks[config.layout]()
        self.lm_head = nn.Linear(MOE_HIDDEN, MOE_VOCAB, bias=False)
        self.post_init()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids).reshape(-1, MOE_HIDDEN)
        return self.lm_head(hidden + self.mlp(hidden))


# Merged model runs one GEMM per expert projection where the split-LoRA
# wrapper runs three; float32 reassociation stays far below this.
MOE_LOGIT_ATOL = 1e-5
EXPERT_LAYOUTS = [
    pytest.param("routed_gated", RoutedExpertsLoraWrapper, id="routed_gated"),
    pytest.param("routed_ungated", RoutedExpertsLoraWrapper, id="routed_ungated"),
    pytest.param("transposed", TransposedExpertsLoraWrapper, id="transposed"),
    pytest.param("sorted", SortedExpertsLoraWrapper, id="sorted"),
]


def _expert_lora_model(
    layout: str, target_parameters: list[str] | None = None
) -> tuple[TinyMoeForCausalLM, nn.Module]:
    """Base model and a split-LoRA PEFT copy (``lm_head`` plus packed experts), upgraded as the trainer does."""
    torch.manual_seed(0)
    base = TinyMoeForCausalLM(TinyMoeConfig(layout=layout))
    lora = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["lm_head"],
        target_parameters=target_parameters or moe_expert_target_parameters(base),
        lora_dropout=0.0,
        init_lora_weights=False,
    )
    peft_model = get_peft_model(copy.deepcopy(base), lora, adapter_name="actor")
    upgrade_moe_param_wrappers(peft_model)
    return base, peft_model


def _moe_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Eval-mode logits of a tiny MoE model or its PEFT wrapper."""
    model.eval()
    with torch.no_grad():
        return model(input_ids=input_ids)


def _load_tiny_moe(config: TinyMoeConfig, merged: Path) -> TinyMoeForCausalLM:
    """Strictly load an export: every model tensor present, nothing extra."""
    model = TinyMoeForCausalLM(config)
    model.load_state_dict(load_file(merged / "model.safetensors"))
    return model


class TestExportMergedPretrainedExpertLora:
    @pytest.mark.parametrize(("layout", "wrapper_cls"), EXPERT_LAYOUTS)
    def test_merged_forward_matches_split_lora_forward(
        self, tmp_path: Path, layout: str, wrapper_cls: type
    ) -> None:
        # Arrange
        base, peft_model = _expert_lora_model(layout)
        input_ids = torch.randint(0, MOE_VOCAB, (2, 6))
        live_logits = _moe_logits(peft_model, input_ids)
        before = _snapshot_tensors(peft_model)

        # Act
        export_merged_pretrained(
            tmp_path / "merged", model=peft_model, torch_dtype=torch.float32
        )

        # Assert
        assert any(type(module) is wrapper_cls for module in peft_model.modules())
        merged = _load_tiny_moe(base.config, tmp_path / "merged")
        assert torch.allclose(
            live_logits, _moe_logits(merged, input_ids), atol=MOE_LOGIT_ATOL
        )
        assert not torch.allclose(live_logits, _moe_logits(base, input_ids), atol=1e-2)
        for name in moe_expert_target_parameters(base):
            assert not torch.allclose(
                merged.get_parameter(name), base.get_parameter(name)
            ), name
        _assert_unmutated(peft_model, before)

    def test_untargeted_expert_tensors_copy_unchanged(self, tmp_path: Path) -> None:
        # Arrange
        base, peft_model = _expert_lora_model(
            "transposed", target_parameters=["mlp.experts.down_proj"]
        )
        input_ids = torch.randint(0, MOE_VOCAB, (2, 6))
        live_logits = _moe_logits(peft_model, input_ids)

        # Act
        export_merged_pretrained(
            tmp_path / "merged", model=peft_model, torch_dtype=torch.float32
        )

        # Assert
        weights = load_file(tmp_path / "merged" / "model.safetensors")
        experts = base.mlp.experts
        for name in ("gate_up_proj", "gate_up_proj_bias", "down_proj_bias"):
            assert torch.equal(weights[f"mlp.experts.{name}"], getattr(experts, name))
        assert not torch.allclose(weights["mlp.experts.down_proj"], experts.down_proj)
        merged = _load_tiny_moe(base.config, tmp_path / "merged")
        assert torch.allclose(
            live_logits, _moe_logits(merged, input_ids), atol=MOE_LOGIT_ATOL
        )

    def test_copies_experts_when_wrapper_lacks_named_adapter(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        base, peft_model = _expert_lora_model("routed_gated")
        peft_model.add_adapter(
            "critic",
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["lm_head"],
                lora_dropout=0.0,
                init_lora_weights=False,
            ),
        )
        peft_model.set_adapter("critic")
        input_ids = torch.randint(0, MOE_VOCAB, (2, 6))
        critic_logits = _moe_logits(peft_model, input_ids)

        # Act
        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            adapter_name="critic",
            torch_dtype=torch.float32,
        )

        # Assert
        weights = load_file(tmp_path / "merged" / "model.safetensors")
        for name in ("gate_up_proj", "down_proj"):
            assert torch.equal(
                weights[f"mlp.experts.{name}"], getattr(base.mlp.experts, name)
            )
        merged = _load_tiny_moe(base.config, tmp_path / "merged")
        assert torch.allclose(
            critic_logits, _moe_logits(merged, input_ids), atol=MOE_LOGIT_ATOL
        )

    @pytest.mark.parametrize("layout", ["routed_gated", "routed_ungated", "sorted"])
    def test_gathers_expert_parallel_dtensors(
        self, tmp_path: Path, gloo_process_group: None, layout: str
    ) -> None:
        # Arrange
        base, peft_model = _expert_lora_model(layout)
        input_ids = torch.randint(0, MOE_VOCAB, (2, 6))
        live_logits = _moe_logits(peft_model, input_ids)
        ep_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
        apply_expert_parallel(peft_model, ep_mesh)
        wrappers = [
            module
            for module in peft_model.modules()
            if isinstance(module, ParamWrapper)
        ]
        lora_b = wrappers[0].lora_B["actor"].weight

        # Act
        export_merged_pretrained(
            tmp_path / "merged", model=peft_model, torch_dtype=torch.float32
        )

        # Assert
        assert isinstance(lora_b, DTensor)
        assert lora_b.ndim == 3
        assert all(isinstance(w.get_param(), DTensor) for w in wrappers)
        assert all(isinstance(w.lora_A["actor"].weight, DTensor) for w in wrappers)
        merged = _load_tiny_moe(base.config, tmp_path / "merged")
        assert torch.allclose(
            live_logits, _moe_logits(merged, input_ids), atol=MOE_LOGIT_ATOL
        )


def _tiny_qwen3_moe() -> Qwen3MoeForCausalLM:
    """CPU Qwen3-MoE with packed ``mlp.experts`` and an untied ``lm_head``."""
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=64,
        head_dim=16,
    )
    return Qwen3MoeForCausalLM(config).float()


def _save_packed_checkpoint(model: PreTrainedModel, path: Path) -> None:
    """Write in-memory tensor names (packed experts), as Qwen3.5/3.6-MoE checkpoints store them."""
    path.mkdir(parents=True)
    state = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    model.config.save_pretrained(path)


class TestExportMergedPretrainedQwen3Moe:
    @pytest.mark.parametrize("shape", ["live", "replay"])
    def test_attention_and_expert_lora_reload_matches_adapter(
        self, tmp_path: Path, shape: str
    ) -> None:
        # Arrange
        torch.manual_seed(0)
        base = _tiny_qwen3_moe()
        _save_packed_checkpoint(base, tmp_path / "base")
        lora = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            target_parameters=moe_expert_target_parameters(base),
            lora_dropout=0.0,
            init_lora_weights=False,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(copy.deepcopy(base), lora, adapter_name="actor")
        assert upgrade_moe_param_wrappers(peft_model) == 2
        peft_model.save_pretrained(tmp_path / "ckpt")
        input_ids = torch.randint(0, 64, (2, 10))
        live_logits = _peft_logits(peft_model, input_ids)

        # Act
        if shape == "live":
            export_merged_pretrained(
                tmp_path / "merged", model=peft_model, torch_dtype=torch.float32
            )
        else:
            export_merged_pretrained(
                tmp_path / "merged",
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
                torch_dtype=torch.float32,
            )

        # Assert
        weights = load_file(tmp_path / "merged" / "model.safetensors")
        assert "lm_head.weight" in weights
        assert "model.layers.1.mlp.experts.gate_up_proj" in weights
        assert all("lora" not in key and "base_layer" not in key for key in weights)
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")
        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=MOE_LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )

    def test_checkpoint_wrapped_layers_export_base_state_dict_keys(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        torch.manual_seed(0)
        base = _tiny_qwen3_moe()
        lora = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            target_parameters=moe_expert_target_parameters(base),
            lora_dropout=0.0,
            init_lora_weights=False,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(copy.deepcopy(base), lora, adapter_name="actor")
        upgrade_moe_param_wrappers(peft_model)
        layers = peft_model.get_base_model().model.layers
        for index, layer in enumerate(layers):
            layers[index] = checkpoint_wrapper(layer)
        input_ids = torch.randint(0, 64, (2, 10))
        live_logits = _peft_logits(peft_model, input_ids)

        # Act
        export_merged_pretrained(
            tmp_path / "merged", model=peft_model, torch_dtype=torch.float32
        )

        # Assert
        weights = load_file(tmp_path / "merged" / "model.safetensors")
        assert set(weights) == set(base.state_dict())
        reloaded = _tiny_qwen3_moe()
        reloaded.load_state_dict(weights)
        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=MOE_LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )

    def test_replay_without_packed_expert_keys_raises(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_qwen3_moe()
        # save_pretrained writes per-expert names, not the packed parameter.
        base.save_pretrained(tmp_path / "base")
        lora = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[],
            target_parameters=moe_expert_target_parameters(base),
            lora_dropout=0.0,
            init_lora_weights=False,
            task_type="CAUSAL_LM",
        )
        get_peft_model(base, lora, adapter_name="actor").save_pretrained(
            tmp_path / "ckpt"
        )

        with pytest.raises(MergedExportError, match="has no tensor"):
            export_merged_pretrained(
                tmp_path / "merged",
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
            )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    """Byte size of a dense tensor."""
    return int(tensor.numel() * tensor.element_size())


class TestExportMergedPretrainedArchitectures:
    @pytest.mark.parametrize(
        ("make_base", "targets", "fan_in_fan_out"),
        [
            (_tiny_gpt2, ["c_attn"], True),
            (_tiny_opt, ["q_proj", "v_proj"], False),
        ],
        ids=["gpt2", "opt"],
    )
    def test_live_and_replay_match_saved_checkpoint(
        self,
        tmp_path: Path,
        make_base: Any,
        targets: list[str],
        fan_in_fan_out: bool,
    ) -> None:
        torch.manual_seed(0)
        base = make_base()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(
            base,
            _lora_config(target_modules=targets, fan_in_fan_out=fan_in_fan_out),
            adapter_name="actor",
        )
        peft_model.save_pretrained(tmp_path / "ckpt")
        input_ids = torch.randint(0, 32, (2, 8))
        live_logits = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "live",
            model=peft_model,
            torch_dtype=torch.float32,
        )
        export_merged_pretrained(
            tmp_path / "replay",
            adapter_path=tmp_path / "ckpt" / "actor",
            base_model_name_or_path=tmp_path / "base",
            adapter_name="actor",
            torch_dtype=torch.float32,
        )
        live_reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "live")
        replay_reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "replay")

        assert torch.allclose(
            live_logits,
            _peft_logits(live_reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )
        assert torch.allclose(
            live_logits,
            _peft_logits(replay_reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )
        live_keys = set(load_file(tmp_path / "live" / "model.safetensors"))
        replay_keys = set(load_file(tmp_path / "replay" / "model.safetensors"))
        assert all("lora" not in key for key in live_keys | replay_keys)
        # Replay writes checkpoint tensors only; live may include computed buffers.
        assert replay_keys <= live_keys

    def test_replay_sharded_tied_checkpoint(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base", max_shard_size="1KB")
        peft_model = get_peft_model(
            base,
            _lora_config(),
            adapter_name="actor",
        )
        peft_model.save_pretrained(tmp_path / "ckpt")
        input_ids = torch.randint(0, 32, (1, 4))
        live_logits = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "merged",
            adapter_path=tmp_path / "ckpt" / "actor",
            base_model_name_or_path=tmp_path / "base",
            adapter_name="actor",
            torch_dtype=torch.float32,
            max_shard_size="1KB",
        )
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")
        weight_map = json.loads(
            (tmp_path / "merged" / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )["weight_map"]

        assert (tmp_path / "base" / "model.safetensors.index.json").exists()
        assert "lm_head.weight" not in weight_map
        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )


class TestExportMergedPretrainedMemory:
    def test_store_does_not_retain_loaded_tensors(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        _tiny_gpt2().save_pretrained(tmp_path / "base")
        store = BaseWeightStore.from_checkpoint(tmp_path / "base")
        key = next(iter(store.key_files))

        tensor = store.load(key)
        ref = weakref.ref(tensor)
        nbytes = _tensor_nbytes(tensor)
        del tensor
        gc.collect()

        assert nbytes > 0
        assert ref() is None

    def test_replay_shard_loads_stay_below_full_checkpoint(
        self, tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        shard_nbytes: list[int] = []
        running: list[int] = []
        orig_load = BaseWeightStore.load
        orig_materialize = lora_merge_mod._materialize_shard

        def spy_load(self: BaseWeightStore, key: str) -> torch.Tensor:
            tensor = orig_load(self, key)
            running.append(_tensor_nbytes(tensor))
            return tensor

        def spy_materialize(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            running.clear()
            tensors = orig_materialize(*args, **kwargs)
            shard_nbytes.append(sum(running))
            return tensors

        with (
            patch.object(BaseWeightStore, "load", spy_load),
            patch.object(lora_merge_mod, "_materialize_shard", spy_materialize),
        ):
            export_merged_pretrained(
                tmp_path / "merged",
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
                adapter_name="actor",
                torch_dtype=torch.float32,
                max_shard_size="1KB",
            )

        store = BaseWeightStore.from_checkpoint(tmp_path / "base")
        total = sum(_tensor_nbytes(store.load(key)) for key in store.key_files)
        assert shard_nbytes
        assert max(shard_nbytes) < total
        assert all(nbytes > 0 for nbytes in shard_nbytes)

    def test_replay_adapter_is_small_and_base_stays_meta(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        skeleton, store = lora_merge_mod._load_replay_peft(
            tmp_path / "ckpt" / "actor",
            tmp_path / "base",
            "actor",
        )
        adapter_bytes = sum(
            _tensor_nbytes(param)
            for param in skeleton.parameters()
            if param.device.type != "meta"
        )
        base_bytes = sum(_tensor_nbytes(store.load(key)) for key in store.key_files)
        base_params = [
            param
            for name, param in skeleton.get_base_model().named_parameters()
            if "lora_" not in name
        ]

        lora_merge_mod._export_live_model(
            skeleton,
            output_dir=tmp_path / "merged",
            adapter_name="actor",
            max_shard_size="1KB",
            torch_dtype=torch.float32,
            tokenizer=None,
            weight_store=store,
        )

        assert adapter_bytes < base_bytes
        assert all(param.device.type == "meta" for param in base_params)

    def test_sharded_write_releases_shard_tensors(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        refs: list[weakref.ref] = []
        orig_save = save_file

        def spy_save(tensors: dict[str, torch.Tensor], path: str) -> None:
            refs.extend(weakref.ref(tensor) for tensor in tensors.values())
            orig_save(tensors, path)

        with patch("agilerl.utils.lora_merge.save_file", spy_save):
            export_merged_pretrained(
                tmp_path / "merged",
                model=peft_model,
                torch_dtype=torch.float32,
                max_shard_size="1KB",
            )
        gc.collect()

        assert refs
        assert all(ref() is None for ref in refs)


class TestExportMergedPretrainedDevices:
    def test_cpu_float16_live_export_logits(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft().to(dtype=torch.float16)
        input_ids = torch.randint(0, 32, (2, 8))
        live_logits = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            torch_dtype=torch.float16,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(
            tmp_path / "merged", torch_dtype=torch.float16
        )

        assert (
            next(
                iter(load_file(tmp_path / "merged" / "model.safetensors").values())
            ).dtype
            == torch.float16
        )
        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=FP16_LOGIT_ATOL,
            rtol=FP16_LOGIT_ATOL,
        )

    def test_replay_from_bf16_saved_base(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2().to(dtype=torch.bfloat16)
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        input_ids = torch.randint(0, 32, (2, 8))
        live_logits = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "merged",
            adapter_path=tmp_path / "ckpt" / "actor",
            base_model_name_or_path=tmp_path / "base",
            adapter_name="actor",
            torch_dtype=torch.bfloat16,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(
            tmp_path / "merged", torch_dtype=torch.bfloat16
        )

        assert (
            next(
                iter(load_file(tmp_path / "merged" / "model.safetensors").values())
            ).dtype
            == torch.bfloat16
        )
        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=BF16_LOGIT_ATOL,
            rtol=BF16_LOGIT_ATOL,
        )


@pytest.mark.gpu
class TestExportMergedPretrainedGPU:
    @pytest.fixture(autouse=True)
    def _full_fp32_cuda_matmul(self):
        # LoRA is two GEMMs; merged is one. TF32 makes those disagree past 1e-5.
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32

    def test_live_export_logits_match(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft().cuda()
        before = _snapshot_tensors(peft_model)
        input_ids = torch.randint(0, 32, (2, 8), device="cuda")
        live_logits = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            torch_dtype=torch.float32,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged").cuda()
        weights = load_file(tmp_path / "merged" / "model.safetensors")

        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )
        _assert_unmutated(peft_model, before)
        assert all(param.device.type == "cuda" for param in peft_model.parameters())
        assert all(tensor.device.type == "cpu" for tensor in weights.values())
        assert all("lora" not in key for key in weights)

    def test_linear_lora_live_and_replay_logits_match(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_opt()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(
            base,
            _lora_config(target_modules=["q_proj", "v_proj"], fan_in_fan_out=False),
            adapter_name="actor",
        )
        peft_model.save_pretrained(tmp_path / "ckpt")
        input_ids = torch.randint(0, 32, (2, 8), device="cuda")
        live_logits = _peft_logits(peft_model.cuda(), input_ids)

        export_merged_pretrained(
            tmp_path / "live",
            model=peft_model,
            torch_dtype=torch.float32,
        )
        export_merged_pretrained(
            tmp_path / "replay",
            adapter_path=tmp_path / "ckpt" / "actor",
            base_model_name_or_path=tmp_path / "base",
            adapter_name="actor",
            torch_dtype=torch.float32,
        )
        live_reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "live").cuda()
        replay_reloaded = AutoModelForCausalLM.from_pretrained(
            tmp_path / "replay"
        ).cuda()

        assert torch.allclose(
            live_logits,
            _peft_logits(live_reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )
        assert torch.allclose(
            live_logits,
            _peft_logits(replay_reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )

    def test_live_export_bf16_logits_match(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft().to(device="cuda", dtype=torch.bfloat16)
        input_ids = torch.randint(0, 32, (2, 8), device="cuda")
        live_logits = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            torch_dtype=torch.bfloat16,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(
            tmp_path / "merged", torch_dtype=torch.bfloat16
        ).cuda()

        assert (
            next(
                iter(load_file(tmp_path / "merged" / "model.safetensors").values())
            ).dtype
            == torch.bfloat16
        )
        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=BF16_LOGIT_ATOL,
            rtol=BF16_LOGIT_ATOL,
        )
        assert all(param.device.type == "cuda" for param in peft_model.parameters())

    def test_live_export_sharded_logits_match(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft().cuda()
        input_ids = torch.randint(0, 32, (1, 4), device="cuda")
        live_logits = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            torch_dtype=torch.float32,
            max_shard_size="1KB",
        )
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged").cuda()

        assert (tmp_path / "merged" / "model.safetensors.index.json").exists()
        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )

    def test_live_export_on_cuda(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft().cuda()
        input_ids = torch.randint(0, 32, (2, 8), device="cuda")
        live_logits = _peft_logits(peft_model, input_ids)

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            torch_dtype=torch.float32,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged").cuda()

        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )

    def test_replay_keeps_base_meta_and_matches_logits(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        base = _tiny_gpt2()
        base.save_pretrained(tmp_path / "base")
        peft_model = get_peft_model(base, _lora_config(), adapter_name="actor")
        peft_model.save_pretrained(tmp_path / "ckpt")
        input_ids = torch.randint(0, 32, (2, 8), device="cuda")
        live_logits = _peft_logits(peft_model.cuda(), input_ids)

        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("replay must not load the full model"),
        ):
            export_merged_pretrained(
                tmp_path / "merged",
                adapter_path=tmp_path / "ckpt" / "actor",
                base_model_name_or_path=tmp_path / "base",
                adapter_name="actor",
                torch_dtype=torch.float32,
            )

        skeleton, _ = lora_merge_mod._load_replay_peft(
            tmp_path / "ckpt" / "actor",
            tmp_path / "base",
            "actor",
        )
        base_params = [
            param
            for name, param in skeleton.get_base_model().named_parameters()
            if "lora_" not in name
        ]
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged").cuda()

        assert all(param.device.type == "meta" for param in base_params)
        assert torch.allclose(
            live_logits,
            _peft_logits(reloaded, input_ids),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )


# The sync merge and PEFT's merge run the same fp32 ops in a different order.
FP32_MERGE_ATOL = 1e-6
FP32_MERGE_RTOL = 1e-6
DENSE_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
GDN_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]


def _tiny_qwen3() -> Qwen3ForCausalLM:
    """CPU dense Qwen3 with two decoder layers."""
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
    )
    return Qwen3ForCausalLM(config)


def _tiny_qwen3_5_moe() -> Qwen3_5MoeForCausalLM:
    """CPU Qwen3.5-MoE: one gated-delta layer, one full-attention layer, packed experts."""
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        max_position_embeddings=64,
    )
    return Qwen3_5MoeForCausalLM(config)


def _sync_lora(
    target_modules: list[str],
    target_parameters: list[str] | None = None,
    lora_alpha: int = 8,
) -> LoraConfig:
    return LoraConfig(
        r=4,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        target_parameters=target_parameters,
        lora_dropout=0.0,
        init_lora_weights=False,
        task_type="CAUSAL_LM",
    )


def _merged_sync_state(
    peft_model: nn.Module, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Every merged weight the sync pushes, written at ``dtype``."""
    state = {}
    for spec in merged_sync_specs(peft_model, "actor"):
        destination = torch.empty(spec.item.reference.shape, dtype=dtype)
        assert write_merged_weight(spec, destination) is None
        state[spec.item.hf_name] = destination
    return state


def _dense_peft(dtype: torch.dtype) -> nn.Module:
    torch.manual_seed(0)
    return get_peft_model(
        _tiny_qwen3().to(dtype), _sync_lora(DENSE_TARGETS), adapter_name="actor"
    )


def _two_dimensional_parameter_peft() -> nn.Module:
    """One 2-D parameter targeted directly, which PEFT counts as a single expert."""
    torch.manual_seed(0)
    return get_peft_model(
        _tiny_qwen3(),
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[],
            target_parameters=["model.layers.0.self_attn.q_proj.weight"],
            lora_dropout=0.0,
            init_lora_weights=False,
            task_type="CAUSAL_LM",
        ),
        adapter_name="actor",
    )


def _gdn_moe_peft() -> nn.Module:
    torch.manual_seed(0)
    base = _tiny_qwen3_5_moe()
    return get_peft_model(
        base,
        _sync_lora(GDN_TARGETS, moe_expert_target_parameters(base)),
        adapter_name="actor",
    )


class TestMergedSyncSpecs:
    def test_dense_model_names_every_adapted_linear_weight(self) -> None:
        # Act
        names = [
            spec.item.hf_name
            for spec in merged_sync_specs(_dense_peft(torch.float32), "actor")
        ]

        # Assert
        expected = [
            f"model.layers.{layer}.{module}.weight"
            for layer in range(2)
            for module in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        ]
        assert names == expected

    def test_gated_delta_moe_model_names_projections_and_packed_experts(self) -> None:
        # Act
        names = {
            spec.item.hf_name for spec in merged_sync_specs(_gdn_moe_peft(), "actor")
        }

        # Assert: checkpoint names, which vLLM's loader packs into
        # in_proj_qkvz / in_proj_ba / qkv_proj and its fused experts.
        gdn = [
            f"model.layers.0.linear_attn.{name}.weight"
            for name in (
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            )
        ]
        attention = [
            f"model.layers.1.self_attn.{name}.weight"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        experts = [
            f"model.layers.{layer}.mlp.experts.{name}"
            for layer in range(2)
            for name in ("gate_up_proj", "down_proj")
        ]
        assert names == {*gdn, *attention, *experts}

    def test_missing_adapter_raises(self) -> None:
        with pytest.raises(
            ValueError, match="Adapter 'critic' is not on the PEFT model"
        ):
            merged_sync_specs(_dense_peft(torch.float32), "critic")

    def test_adapter_on_no_layer_raises(self) -> None:
        # Arrange: strip the adapter's LoRA from every layer.
        peft_model = _dense_peft(torch.float32)
        for module in peft_model.modules():
            if isinstance(module, LoraLayer) and "actor" in module.lora_A:
                del module.lora_A["actor"]

        # Act / Assert
        with pytest.raises(ValueError, match="Adapter 'actor' adapts no weight"):
            merged_sync_specs(peft_model, "actor")

    def test_dora_adapter_raises(self) -> None:
        # Arrange
        torch.manual_seed(0)
        config = _sync_lora(["q_proj"])
        config.use_dora = True
        peft_model = get_peft_model(_tiny_qwen3(), config, adapter_name="actor")

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=r"does not support DoRA adapters \(model\.layers\.0\.self_attn",
        ):
            merged_sync_specs(peft_model, "actor")

    def test_non_pretrained_base_raises(self) -> None:
        peft_model = _dense_peft(torch.float32)
        peft_model.get_base_model = lambda: nn.Linear(2, 2)

        with pytest.raises(
            TypeError, match="Merged sync requires a transformers model"
        ):
            merged_sync_specs(peft_model, "actor")


class TestWriteMergedWeight:
    def test_dense_merge_matches_peft_merge_and_unload(self) -> None:
        # Arrange
        peft_model = _dense_peft(torch.float32)
        reference = copy.deepcopy(peft_model).merge_and_unload().state_dict()

        # Act
        merged = _merged_sync_state(peft_model, torch.float32)

        # Assert
        assert merged
        for name, tensor in merged.items():
            assert torch.allclose(
                tensor, reference[name], atol=FP32_MERGE_ATOL, rtol=FP32_MERGE_RTOL
            ), name

    def test_gated_delta_moe_merge_matches_peft_merge_and_unload(self) -> None:
        # Arrange: training upgrades the expert wrappers; PEFT merges plain ones.
        peft_model = _gdn_moe_peft()
        reference = copy.deepcopy(peft_model).merge_and_unload().state_dict()
        assert upgrade_moe_param_wrappers(peft_model) == 2

        # Act
        merged = _merged_sync_state(peft_model, torch.float32)

        # Assert
        assert "model.layers.1.mlp.experts.gate_up_proj" in merged
        for name, tensor in merged.items():
            assert torch.allclose(
                tensor, reference[name], atol=FP32_MERGE_ATOL, rtol=FP32_MERGE_RTOL
            ), name

    def test_expert_chunks_merge_like_one_pass(self) -> None:
        # Arrange: a one-byte chunk budget merges one expert at a time.
        peft_model = _gdn_moe_peft()
        whole = _merged_sync_state(peft_model, torch.float32)

        # Act
        with patch.object(lora_merge_mod, "DEFAULT_ROUTED_CHUNK_BYTES", 1):
            chunked = _merged_sync_state(peft_model, torch.float32)

        # Assert
        for name, tensor in whole.items():
            assert torch.equal(chunked[name], tensor), name

    def test_bf16_merge_rounds_once_from_fp32(self) -> None:
        # Arrange: a bf16 base with PEFT's fp32 adapter.
        peft_model = _dense_peft(torch.bfloat16)
        upcast = copy.deepcopy(peft_model).float()
        fp32_reference = upcast.merge_and_unload().state_dict()

        # Act
        merged = _merged_sync_state(peft_model, torch.bfloat16)

        # Assert: the fp32 merge of bf16 factors, rounded to bf16 once.
        for name, tensor in merged.items():
            assert torch.equal(tensor, fp32_reference[name].to(torch.bfloat16)), name

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_merge_leaves_the_trainer_weights_unchanged(
        self, dtype: torch.dtype
    ) -> None:
        # Arrange
        peft_model = _gdn_moe_peft().to(dtype)
        before = _snapshot_tensors(peft_model)

        # Act
        _merged_sync_state(peft_model, dtype)

        # Assert
        _assert_unmutated(peft_model, before)

    def test_none_destination_joins_the_module_gather(self) -> None:
        # Arrange: a rank that only joins the gather passes no destination.
        peft_model = _dense_peft(torch.float32)
        before = _snapshot_tensors(peft_model)
        spec = next(iter(merged_sync_specs(peft_model, "actor")))

        # Act
        error = write_merged_weight(spec, None)

        # Assert
        assert error is None
        _assert_unmutated(peft_model, before)

    @pytest.mark.parametrize(
        "build",
        [lambda: _dense_peft(torch.float32), _two_dimensional_parameter_peft],
        ids=["module", "param"],
    )
    def test_merge_failure_after_the_gather_is_returned(self, build: Any) -> None:
        # Arrange: a destination of the wrong shape fails the write, not the gather.
        peft_model = build()
        before = _snapshot_tensors(peft_model)
        spec = next(iter(merged_sync_specs(peft_model, "actor")))

        # Act
        error = write_merged_weight(spec, torch.empty(1, 3))

        # Assert
        assert isinstance(error, RuntimeError)
        _assert_unmutated(peft_model, before)

    def test_gather_failure_raises(self) -> None:
        # Arrange
        spec = next(iter(merged_sync_specs(_dense_peft(torch.float32), "actor")))

        @contextmanager
        def failed_gather(*_args: Any) -> Any:
            msg = "all-gather failed"
            raise RuntimeError(msg)
            yield

        # Act / Assert
        with (
            patch.object(lora_merge_mod, "gather_params", failed_gather),
            pytest.raises(RuntimeError, match="all-gather failed"),
        ):
            write_merged_weight(spec, torch.empty(spec.item.reference.shape))

    def test_embedding_merge_matches_peft_and_restores_factor_storage(self) -> None:
        # Arrange: an embedding is not a plain Linear, so the merge runs PEFT's delta.
        torch.manual_seed(0)
        peft_model = get_peft_model(
            _tiny_qwen3(), _sync_lora(["embed_tokens"]), adapter_name="actor"
        )
        peft_model.to(torch.bfloat16)
        reference = copy.deepcopy(peft_model).float().merge_and_unload().state_dict()
        factors = {
            name: param.data_ptr()
            for name, param in peft_model.named_parameters()
            if "lora_embedding" in name
        }

        # Act
        merged = _merged_sync_state(peft_model, torch.float32)

        # Assert
        name = "model.embed_tokens.weight"
        assert set(merged) == {name}
        assert torch.allclose(
            merged[name], reference[name], atol=FP32_MERGE_ATOL, rtol=FP32_MERGE_RTOL
        )
        assert factors
        for factor_name, param in peft_model.named_parameters():
            if factor_name in factors:
                assert param.data_ptr() == factors[factor_name], factor_name
                assert param.dtype == torch.bfloat16, factor_name

    @pytest.mark.parametrize(("layout", "wrapper_cls"), EXPERT_LAYOUTS)
    def test_packed_expert_layouts_match_the_unfused_merge(
        self, layout: str, wrapper_cls: type
    ) -> None:
        # Arrange
        _, peft_model = _expert_lora_model(layout)
        specs = merged_sync_specs(peft_model, "actor")
        expected = {spec.item.hf_name: _unfused_merge(spec) for spec in specs}

        # Act
        merged = _merged_sync_state(peft_model, torch.float32)

        # Assert
        assert any(
            isinstance(spec.item, ParamLoraMerged)
            and isinstance(spec.item.wrapper, wrapper_cls)
            for spec in specs
        )
        for name, tensor in expected.items():
            assert torch.allclose(
                merged[name], tensor, atol=FP32_MERGE_ATOL, rtol=FP32_MERGE_RTOL
            ), name

    def test_two_dimensional_parameter_merges_as_one_expert(self) -> None:
        # Arrange: a 2-D parameter is one expert, so the merge is one matmul.
        peft_model = _two_dimensional_parameter_peft()
        reference = copy.deepcopy(peft_model).merge_and_unload().state_dict()
        specs = merged_sync_specs(peft_model, "actor")

        # Act
        merged = _merged_sync_state(peft_model, torch.float32)

        # Assert
        assert len(specs) == 1
        assert isinstance(specs[0].item, ParamLoraMerged)
        assert specs[0].item.wrapper.num_experts == 1
        name = "model.layers.0.self_attn.q_proj.weight"
        assert set(merged) == {name}
        assert torch.allclose(
            merged[name], reference[name], atol=FP32_MERGE_ATOL, rtol=FP32_MERGE_RTOL
        )


# BLAS may fuse the scale into the accumulate, skipping one rounding: two
# fp32 ulps at magnitudes below 32.
ACCUMULATE_ATOL = 2.0**-18


def _odd_scale_dense_peft() -> nn.Module:
    """Bf16 Qwen3 whose LoRA scale, 1.5, makes ``scale * B @ A`` round."""
    torch.manual_seed(0)
    return get_peft_model(
        _tiny_qwen3().to(torch.bfloat16),
        _sync_lora(DENSE_TARGETS, lora_alpha=6),
        adapter_name="actor",
    )


def _odd_scale_fan_in_fan_out_peft() -> nn.Module:
    """Bf16 GPT-2, whose ``Conv1D`` stores ``[in, out]``, at LoRA scale 1.5."""
    torch.manual_seed(0)
    return get_peft_model(
        _tiny_gpt2().to(torch.bfloat16),
        LoraConfig(
            r=2,
            lora_alpha=3,
            target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.0,
            init_lora_weights=False,
            fan_in_fan_out=True,
        ),
        adapter_name="actor",
    )


def _odd_scale_packed_experts_peft() -> nn.Module:
    """Bf16 Qwen3.5-MoE with upgraded packed-expert wrappers at LoRA scale 1.5."""
    torch.manual_seed(0)
    base = _tiny_qwen3_5_moe().to(torch.bfloat16)
    peft_model = get_peft_model(
        base,
        _sync_lora(GDN_TARGETS, moe_expert_target_parameters(base), lora_alpha=6),
        adapter_name="actor",
    )
    upgrade_moe_param_wrappers(peft_model)
    return peft_model


def _unfused_merge(spec: MergedSyncSpec) -> torch.Tensor:
    """``W + scale * B @ A`` as a separate fp32 matmul, scale and add."""
    item = spec.item
    if isinstance(item, ParamLoraMerged):
        wrapper = item.wrapper
        weight_a = wrapper.lora_A["actor"].get_parameter("weight").float()
        weight_b = wrapper.lora_B["actor"].get_parameter("weight").float()
        n_experts = wrapper.num_experts
        equation = (
            "o r e, e r i -> e o i"
            if wrapper._did_swap_in_out_features
            else "o r e, e r i -> e i o"
        )
        delta = torch.einsum(
            equation,
            weight_b.reshape(weight_b.shape[0], -1, n_experts),
            weight_a.reshape(n_experts, -1, weight_a.shape[-1]),
        )
        return wrapper.get_param().float() + delta * wrapper.scaling["actor"]
    layer = item.layer
    delta = layer.lora_B["actor"].weight.float() @ layer.lora_A["actor"].weight.float()
    if layer.fan_in_fan_out:
        delta = delta.T
    return item.reference.float() + delta * layer.scaling["actor"]


class TestWriteMergedWeightAccumulate:
    @pytest.mark.parametrize(
        "build",
        [
            _odd_scale_dense_peft,
            _odd_scale_fan_in_fan_out_peft,
            _odd_scale_packed_experts_peft,
        ],
        ids=["dense", "fan_in_fan_out", "packed_experts"],
    )
    def test_matches_the_unfused_merge(self, build: Any) -> None:
        # Arrange
        peft_model = build()
        expected = {
            spec.item.hf_name: _unfused_merge(spec)
            for spec in merged_sync_specs(peft_model, "actor")
        }

        # Act
        merged = _merged_sync_state(peft_model, torch.float32)

        # Assert
        assert merged.keys() == expected.keys()
        for name, tensor in expected.items():
            assert tensor.abs().max() < 32, name
            assert torch.allclose(merged[name], tensor, rtol=0, atol=ACCUMULATE_ATOL), (
                name
            )


def _tp_merged_sync_worker(rank: int, world_size: int, family: str) -> int:
    """Merged-sync writes of a tp-sharded model equal the unsharded model's."""
    # Arrange
    torch.manual_seed(0)
    model = _tp_export_peft_model(family)
    reference = copy.deepcopy(model)
    mesh = build_parallel_mesh(world_size=world_size, tp=world_size, device_type="cpu")
    assert mesh is not None
    apply_tensor_parallel(model, mesh.tp)
    expected = _merged_sync_state(reference, torch.float32)

    # Act
    merged = _merged_sync_state(model, torch.float32)

    # Assert
    _assert_same_tensors(merged, expected)
    return sum(
        spec.weight_order is not None and spec.delta_order is not None
        for spec in merged_sync_specs(model, "actor")
    )


@pytest.mark.multi_rank
class TestWriteMergedWeightTensorParallel:
    @pytest.mark.parametrize("family", sorted(TP_EXPORT_MODELS))
    def test_writes_the_unsharded_merge_in_hugging_face_row_order(
        self, family: str
    ) -> None:
        # Act
        reordered = run_ranks(_tp_merged_sync_worker, 2, family)

        # Assert: some written weights were stored in tensor-parallel row order.
        assert reordered[0] > 0
