# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for FSDP2/DTensor LoRA merge to Hugging Face format."""

from __future__ import annotations

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
from torch import nn

pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from peft.tuners.lora.layer import LoraLayer
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    OPTConfig,
    OPTForCausalLM,
)

from agilerl.utils import lora_merge as lora_merge_mod
from agilerl.utils.lora_merge import (
    ARENA_ARTIFACT_MANIFEST_FILENAME,
    MERGED_COMPLETE_MARKER,
    BaseWeightStore,
    MergedExportError,
    ModuleCopy,
    export_merged_pretrained,
    write_merged_completeness_files,
)
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

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
    *,
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
        def recording_gather(params):
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
        def yield_dense(params):
            assert params[0] is shard_tensor
            yield [dense_tensor]

        with patch("agilerl.utils.lora_merge.gather_params", yield_dense):
            shard = lora_merge_mod._materialize_shard(
                [spec],
                ["w"],
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
            torch_dtype=torch.float32,
            keep_tensors=False,
        )

        assert shard == {}

    def test_copy_to_cpu_owns_storage_for_save_file(self, tmp_path: Path) -> None:
        view = torch.ones((), dtype=torch.float32).expand(4, 4)

        copied = lora_merge_mod._copy_to_cpu(view, torch.float32)

        save_file({"w": copied}, str(tmp_path / "t.safetensors"))
        assert copied.data_ptr() != 0

    def test_uses_ds_shape_when_planning_shards(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        weight = peft_model.get_base_model().transformer.wte.weight
        weight.ds_shape = tuple(weight.shape)

        export_merged_pretrained(
            tmp_path / "merged",
            model=peft_model,
            torch_dtype=torch.float32,
        )

        assert (tmp_path / "merged" / "model.safetensors").exists()

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
            "agilerl.utils.lora_merge.AutoModelForCausalLM.from_pretrained",
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
    def _distributed(self, *, is_main: bool = True, all_reduce: Any = None):
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
        def recording_gather(params: Any):
            gather_entered.append(len(params))
            if aborting:
                gathers_after_abort.append(len(params))
            yield list(params)

        with (
            self._distributed(is_main=False, all_reduce=fake_all_reduce),
            patch("agilerl.utils.lora_merge._materialize_shard", spy_materialize),
            patch("agilerl.utils.lora_merge.gather_params", recording_gather),
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
        def recording_gather(params: Any):
            gather_entered.append(len(params))
            yield list(params)

        with (
            self._distributed(all_reduce=fake_all_reduce),
            patch("agilerl.utils.lora_merge.gather_params", recording_gather),
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
    def test_list_and_string_tied_keys_are_skipped(self, tmp_path: Path) -> None:
        peft_model = _tiny_peft()
        base = peft_model.get_base_model()
        base._tied_weights_keys = ["transformer.wpe.weight"]

        export_merged_pretrained(
            tmp_path / "list",
            model=peft_model,
            torch_dtype=torch.float32,
        )
        listed = load_file(tmp_path / "list" / "model.safetensors")
        assert "transformer.wpe.weight" not in listed

        base._tied_weights_keys = "transformer.wpe.weight"
        export_merged_pretrained(
            tmp_path / "string",
            model=peft_model,
            torch_dtype=torch.float32,
        )
        named = load_file(tmp_path / "string" / "model.safetensors")
        assert "transformer.wpe.weight" not in named

        base._tied_weights_keys = None
        export_merged_pretrained(
            tmp_path / "none",
            model=peft_model,
            torch_dtype=torch.float32,
        )
        assert (tmp_path / "none" / "model.safetensors").exists()


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
            "agilerl.utils.lora_merge.AutoModelForCausalLM.from_pretrained",
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
