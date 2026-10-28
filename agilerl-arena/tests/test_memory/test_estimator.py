# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Estimator behaviour: breakdown structure, monotonicity, budget checks."""

import pytest
from pydantic import ValidationError

from agilerl.arena.memory import formulas, vision
from agilerl.arena.memory.advice import advise
from agilerl.arena.memory.estimator import (
    estimate_generation,
    estimate_run,
    estimate_training,
    generation_can_serve,
    training_breakdown,
)
from agilerl.arena.memory.lora_placement import (
    lora_tensor_placement,
    split_moe_lora_recompute_bytes,
)
from agilerl.arena.memory.nccl import (
    TP_RUNTIME_BYTES,
    nccl_on_node_fraction,
    trainer_nccl_bytes,
)
from agilerl.arena.memory.resolve.chunk import resolve_chunk_rows
from agilerl.arena.memory.resolve.fsdp import resolve_fsdp_memory
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GenerationSettings,
    GiB,
    MiB,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingSettings,
    WeightVariant,
)
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from tests.test_memory.test_formulas import (
    FALCON_H1,
    MOE_TINY,
    NEMOTRON_H_MOE,
    NEMOTRON_NANO_MAMBA,
    QWEN_05B,
    SUPER_VL,
)

# One layer, wide MLP: attention LoRA stays under the 100k persistence
# threshold, the intermediate-sized LoRA matrix does not.
WIDE = ModelArch(
    n_layers=1,
    hidden_size=4096,
    intermediate_size=11008,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    vocab_size=128,
    tied_embeddings=True,
)


@pytest.fixture
def model():
    return ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        arch=QWEN_05B,
    )


@pytest.fixture
def device():
    return DeviceSpec(total_bytes=24 * GiB, name="test-24g")


def component(breakdown, key):
    return next(c for c in breakdown.components if c.key == key)


def _with_extra_params(model, fraction):
    analytic = formulas.param_counts(model.arch).total
    return model.model_copy(update={"n_params": int(analytic / (1.0 - fraction))})


class TestEstimateTraining:
    def test_breakdown_structure(self, model, device):
        breakdown = estimate_training(model, device, TrainingSettings())
        keys = [c.key for c in breakdown.components]
        assert keys == [
            "base_weights",
            "adapters",
            "grads",
            "activations",
            "logits_workspace",
            "optimizer_state",
            "overhead",
            "allocator_reserve",
        ]
        assert breakdown.total_bytes > 0
        assert breakdown.fits
        # LoRA-only grads scale with adapter params, under 1/20 of a dense
        # fine-tune's 12 bytes/param.
        full_ft_equivalent = 12 * formulas.param_counts(model.arch).total
        assert component(breakdown, "grads").n_bytes < full_ft_equivalent / 20
        assert (
            component(breakdown, "grads").n_bytes
            < component(breakdown, "base_weights").n_bytes
        )
        # Chunked loss: tiles, one fp32 head, and the weight-dtype head grad.
        head_upcast = model.arch.vocab_size * model.arch.hidden_size * 4
        head_grad = model.arch.vocab_size * model.arch.hidden_size * 2
        assert (
            component(breakdown, "logits_workspace").n_bytes
            <= 512 * MiB + head_upcast + head_grad
        )

    def test_monotonic_in_seq_len_and_update_size(self, model, device):
        base = estimate_training(
            model, device, TrainingSettings(max_model_len=1024)
        ).total_bytes
        longer = estimate_training(
            model, device, TrainingSettings(max_model_len=4096)
        ).total_bytes
        assert longer > base

        bigger_update = estimate_training(
            model, device, TrainingSettings(trajectories_per_update=256)
        ).total_bytes
        default = estimate_training(
            model, device, TrainingSettings(trajectories_per_update=32)
        ).total_bytes
        assert bigger_update > default

    def test_checkpointing_changes_training_memory(self, model, device):
        settings = TrainingSettings(max_model_len=2048)
        checkpointed = estimate_training(model, device, settings)
        unchunked = estimate_training(
            model,
            device,
            settings.model_copy(update={"gradient_checkpointing": False}),
        )
        assert unchunked.total_bytes > checkpointed.total_bytes
        assert any("gradient_checkpointing" in w for w in unchunked.warnings)

    def test_beta_zero_skips_the_reference_forward_but_keeps_the_adapter(
        self, model, device
    ):
        # beta=0 keeps the reference adapter; the reference rows run as their own
        # micro-batch forwards, so the no-grad peak does not move.
        with_kl = TrainingSettings(beta=0.001, max_model_len=4096)
        without_kl = TrainingSettings(beta=0.0, max_model_len=4096)

        assert with_kl.n_resident_adapters == without_kl.n_resident_adapters

        hot = estimate_training(model, device, with_kl)
        cold = estimate_training(model, device, without_kl)

        assert (
            component(cold, "activations").detail["nograd_peak"]
            == (component(hot, "activations").detail["nograd_peak"])
        )
        assert component(cold, "adapters").n_bytes == component(hot, "adapters").n_bytes
        assert cold.total_bytes <= hot.total_bytes

        # Single-row micro-batches let the fused loss bind over backward.
        detail = component(hot, "activations").detail
        assert detail["loss_peak"] >= detail["backward_peak"]
        assert detail["loss_peak"] >= detail["nograd_peak"]
        # Activations is the binding instant net of workspace and gradients.
        logits = component(hot, "logits_workspace").n_bytes
        grads = component(hot, "grads").n_bytes
        assert (
            component(hot, "activations").n_bytes
            == detail["loss_peak"] - logits - grads
        )

        assert any("beta=0" in w for w in cold.warnings)

    def test_packed_adapters_charge_fp32(self, device):
        model = ModelSpec(model_id="moe-test", arch=MOE_TINY)
        train = TrainingSettings(lora_packed_target_matrices=2)

        packed = estimate_training(model, device, train)
        linear_only = estimate_training(
            model, device, train.model_copy(update={"lora_packed_target_matrices": 0})
        )

        packed_params = formulas.packed_lora_param_count(MOE_TINY, train.lora_rank, 2)
        assert (
            component(packed, "adapters").n_bytes
            - component(linear_only, "adapters").n_bytes
            == packed_params * train.n_resident_adapters * 4
        )

    def test_allocator_reserve_marks_up_torch_terms_only(self, model, device):
        # Training is charged the torch caching-allocator reserve; vLLM's CuMem pool is reserved up front.
        breakdown = estimate_training(model, device, TrainingSettings())
        by_key = {c.key: c.n_bytes for c in breakdown.components}
        assert "allocator_reserve" in by_key

        torch_side = sum(
            v for k, v in by_key.items() if k not in ("overhead", "allocator_reserve")
        )
        assert by_key["allocator_reserve"] == pytest.approx(
            torch_side * formulas.ALLOCATOR_RESERVE_FRACTION, abs=1
        )

        generation = estimate_generation(model, device, GenerationSettings())
        assert "allocator_reserve" not in {c.key for c in generation.components}

    def test_exact_checkpoint_count_moves_the_weights_component(self, model, device):
        baseline = component(
            estimate_training(model, device, TrainingSettings()), "base_weights"
        )
        exact = component(
            estimate_training(
                _with_extra_params(model, 0.05), device, TrainingSettings()
            ),
            "base_weights",
        )
        assert exact.n_bytes > baseline.n_bytes

    @pytest.mark.parametrize(
        "phase",
        [estimate_training, estimate_generation],
        ids=["training", "generation"],
    )
    def test_both_phases_warn_when_geometry_misses_the_checkpoint(
        self, model, device, phase
    ):
        # Warn when checkpoint params exceed the geometry by ~5%.
        settings = (
            TrainingSettings() if phase is estimate_training else GenerationSettings()
        )
        assert not any(
            "not accounted for" in w for w in phase(model, device, settings).warnings
        )
        gappy = _with_extra_params(model, 0.05)
        assert any(
            "not accounted for" in w for w in phase(gappy, device, settings).warnings
        )

    def test_geometry_gap_warning_skips_empty_counts(self):
        empty = formulas.ParamCounts(
            embedding=0,
            lm_head=0,
            attention=0,
            mlp=0,
            norms=0,
            multimodal_towers=0,
        )

        assert empty.total == 0
        assert formulas.geometry_gap_warning(empty) is None


class TestEstimateGeneration:
    def test_kv_pool_is_budget_remainder(self, model, device):
        settings = GenerationSettings(gpu_memory_utilization=0.5, max_model_len=1024)
        breakdown = estimate_generation(model, device, settings)
        budget = int(0.5 * device.total_bytes)
        engine_side = sum(
            component(breakdown, key).n_bytes
            for key in (
                "weights",
                "kv_cache",
                "activation_peak",
                "cuda_graphs",
                "lora_slots",
            )
        )
        engine_runtime = component(breakdown, "overhead").detail["engine_runtime"]
        assert engine_side + engine_runtime == pytest.approx(budget, rel=0.01)

        pinned = estimate_generation(
            model,
            device,
            settings.model_copy(update={"kv_cache_memory_bytes": 123456789}),
        )
        assert component(pinned, "kv_cache").n_bytes == 123456789

    def test_enforce_eager_drops_graph_pool(self, model, device):
        with_graphs = estimate_generation(model, device, GenerationSettings())
        eager = estimate_generation(
            model, device, GenerationSettings(enforce_eager=True)
        )
        assert (
            component(with_graphs, "cuda_graphs").n_bytes
            == formulas.CUDA_GRAPH_POOL_BYTES
        )
        assert component(eager, "cuda_graphs").n_bytes == 0

    def test_kv_demand_warning(self, model, device):
        breakdown = estimate_generation(
            model,
            device,
            GenerationSettings(
                max_model_len=32768,
                max_num_seqs=64,
                kv_cache_memory_bytes=1 * GiB,
            ),
        )
        # 64 sequences at 32k tokens want ~25 GiB of KV against a 1 GiB pool.
        assert any("preempt" in w for w in breakdown.warnings)

    def test_warns_when_budget_too_small_for_weights(self, device):
        big = ModelSpec(
            model_id="big",
            arch=QWEN_05B.model_copy(update={"n_layers": 200, "hidden_size": 4096}),
        )
        breakdown = estimate_generation(
            big, device, GenerationSettings(gpu_memory_utilization=0.2)
        )
        assert component(breakdown, "kv_cache").n_bytes == 0
        assert any("fail at init" in w for w in breakdown.warnings)

    def test_empty_kv_pool_does_not_fit(self, model, device):
        # Arrange: weights and overhead use the whole utilization budget.
        settings = GenerationSettings(gpu_memory_utilization=0.01)

        # Act
        breakdown = estimate_generation(model, device, settings)

        # Assert
        assert component(breakdown, "kv_cache").detail["pool"] == 0
        assert breakdown.total_bytes <= breakdown.device_usable_bytes
        assert not breakdown.fits

    def test_pool_shorter_than_one_sequence_does_not_fit(self, model, device):
        # Arrange: vLLM refuses to start without room for one max_model_len sequence.
        settings = GenerationSettings(
            max_model_len=32768, max_num_seqs=2, kv_cache_memory_bytes=1
        )

        # Act
        breakdown = estimate_generation(model, device, settings)

        # Assert
        kv = component(breakdown, "kv_cache")
        assert 0 < kv.detail["startup_pool"] < kv.detail["full_sequence_bytes"]
        assert not breakdown.fits
        assert any("cannot hold one 32768-token" in w for w in breakdown.warnings)

    def test_pool_holding_one_sequence_still_fits_and_warns_of_preemption(
        self, model, device
    ):
        # Arrange: one sequence fits, all 64 do not.
        settings = GenerationSettings(
            max_model_len=32768, max_num_seqs=64, kv_cache_memory_bytes=1 * GiB
        )

        # Act
        breakdown = estimate_generation(model, device, settings)

        # Assert
        kv = component(breakdown, "kv_cache")
        assert kv.detail["full_sequence_bytes"] <= kv.detail["startup_pool"]
        assert kv.detail["pool"] < kv.detail["worst_case_demand"]
        assert breakdown.fits
        assert any("preempt" in w for w in breakdown.warnings)

    def test_hybrid_mamba_state_shares_the_start_up_pool(self):
        # Arrange: vLLM 0.25.1 started this on an A100-80 with 7.2 GiB for KV
        # and Mamba state, though 64 slots of Mamba state alone exceed it.
        info = SUPPORTED_MODEL_INFO["nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"]
        model = ModelSpec(
            model_id="nano",
            arch=ModelArch.from_hf_config(info.config),
            n_params=info.num_params,
        )
        a100 = DeviceSpec(total_bytes=int(79.25 * GiB), name="NVIDIA A100-SXM4-80GB")
        settings = GenerationSettings(
            gpu_memory_utilization=0.2, max_model_len=32768, max_num_seqs=64
        )

        # Act
        breakdown = estimate_generation(model, a100, settings)

        # Assert
        kv = component(breakdown, "kv_cache")
        assert kv.detail["pool"] == 0
        assert kv.detail["full_sequence_bytes"] <= kv.detail["startup_pool"]
        assert breakdown.fits
        assert not any("fail at init" in w for w in breakdown.warnings)

    def test_moe_warns_about_fused_kernel_caches(self, device):
        model = ModelSpec(model_id="moe-test", arch=MOE_TINY)
        breakdown = estimate_generation(model, device, GenerationSettings())

        assert any("fused-MoE kernel" in w for w in breakdown.warnings)

    def test_multimodal_warns_about_tower_residency(self, model, device):
        towered = model.model_copy(
            update={
                "arch": model.arch.model_copy(update={"multimodal_tower_params": 1_000})
            }
        )
        breakdown = estimate_generation(towered, device, GenerationSettings())

        assert any("Multimodal engine residency" in w for w in breakdown.warnings)

    def test_hybrid_warns_about_aligned_kv_pages(self, device):
        model = ModelSpec(model_id="falcon", arch=ModelArch.from_hf_config(FALCON_H1))
        breakdown = estimate_generation(model, device, GenerationSettings())
        block_size = formulas.aligned_kv_block_size(model.arch, "bf16")

        assert any("recurrent layers" in w for w in breakdown.warnings)
        assert any(str(block_size) in w for w in breakdown.warnings)

    def test_a_full_budget_on_a_large_card_fits_the_whole_card(self, model):
        # Arrange: 0.95 of an 80 GiB card plus the CUDA context and runtime
        # exceeds 95% of the card but not the card.
        a100 = DeviceSpec(total_bytes=80 * GiB, name="NVIDIA A100-SXM4-80GB")
        settings = GenerationSettings(gpu_memory_utilization=0.95, max_model_len=4096)

        # Act
        breakdown = estimate_generation(model, a100, settings)

        # Assert
        assert breakdown.device_usable_bytes == a100.total_bytes
        assert a100.usable_bytes < breakdown.total_bytes <= a100.total_bytes
        assert breakdown.fits

    def test_moe_engine_charges_fused_expert_lora_stacks(self, device):
        # Arrange
        moe = ModelSpec(model_id="moe-test", arch=MOE_TINY)
        dense_rank = formulas.lora_param_count(MOE_TINY, 16)
        settings = GenerationSettings(max_lora_rank=16, max_loras=2)
        packed = settings.model_copy(update={"lora_packed_target_matrices": 2})

        # Act
        layout_2d = estimate_generation(moe, device, settings)
        layout_3d = estimate_generation(moe, device, packed)

        # Assert
        stacks_2d = formulas.engine_expert_lora_param_count(MOE_TINY, 16, 1, False)
        stacks_3d = formulas.engine_expert_lora_param_count(MOE_TINY, 16, 1, True)
        assert component(layout_2d, "lora_slots").n_bytes == 2 * 2 * (
            dense_rank + stacks_2d
        )
        assert component(layout_3d, "lora_slots").n_bytes == 2 * 2 * (
            dense_rank + stacks_3d
        )
        assert component(layout_3d, "kv_cache").n_bytes - component(
            layout_2d, "kv_cache"
        ).n_bytes == 2 * 2 * (stacks_2d - stacks_3d)

    def test_model_declaring_3d_expert_weights_uses_the_3d_layout(self, device):
        # Arrange
        declared = ModelSpec(
            model_id="moe-3d",
            arch=MOE_TINY.model_copy(update={"vllm_3d_moe_lora": True}),
        )
        settings = GenerationSettings(max_lora_rank=16)

        # Act
        breakdown = estimate_generation(declared, device, settings)

        # Assert
        stacks_3d = formulas.engine_expert_lora_param_count(MOE_TINY, 16, 1, True)
        assert component(breakdown, "lora_slots").n_bytes == 2 * (
            formulas.lora_param_count(MOE_TINY, 16) + stacks_3d
        )
        assert ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        ).vllm_3d_moe_lora
        assert not ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["ibm-granite/granite-3.1-3b-a800m-instruct"].config
        ).vllm_3d_moe_lora

    def test_engine_runtime_comes_out_of_the_kv_budget_on_every_rank(self):
        # Arrange
        info = SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"]
        qwen36 = ModelSpec(
            model_id="Qwen/Qwen3.6-35B-A3B",
            arch=ModelArch.from_hf_config(info.config),
            n_params=info.num_params,
        )
        a100 = DeviceSpec(total_bytes=int(79.25 * GiB), name="NVIDIA A100-SXM4-80GB")
        one_rank = GenerationSettings(gpu_memory_utilization=0.95, max_model_len=4096)
        two_ranks = one_rank.model_copy(update={"tensor_parallel_size": 2})

        # Act
        tp1 = component(estimate_generation(qwen36, a100, one_rank), "overhead")
        tp2 = component(estimate_generation(qwen36, a100, two_ranks), "overhead")

        # Assert: vLLM 0.32 logged 2.49 GiB (TP=1) and 2.31 GiB (TP=2).
        assert tp1.detail["engine_runtime"] == tp2.detail["engine_runtime"]
        assert tp1.detail["engine_runtime"] / GiB == pytest.approx(2.4, abs=0.1)
        assert tp1.n_bytes == (a100.context_bytes + tp1.detail["engine_runtime"])

    def test_kept_vision_tower_charges_the_encoder_profile(self):
        # Arrange
        info = SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"]
        arch = ModelArch.from_hf_config(info.config)
        qwen36 = ModelSpec(
            model_id="Qwen/Qwen3.6-35B-A3B",
            arch=arch,
            n_params=info.num_params,
            variants=(
                WeightVariant(),
                WeightVariant(name="engine", stripped_multimodal=True),
            ),
        )
        a100 = DeviceSpec(total_bytes=80 * GiB, name="NVIDIA A100-SXM4-80GB")
        kept = GenerationSettings(
            gpu_memory_utilization=0.9, max_model_len=4096, tensor_parallel_size=2
        )
        stripped = kept.model_copy(update={"weight_variant": "engine"})

        # Act
        with_tower = estimate_generation(qwen36, a100, kept)
        without_tower = estimate_generation(qwen36, a100, stripped)

        # Assert
        encoder, _ = vision.encoder_profile_bytes(arch, 2.0, 2, 1)
        peak = component(with_tower, "activation_peak")
        assert peak.n_bytes == encoder
        assert peak.detail["startup_peak"] == encoder
        assert (
            component(without_tower, "activation_peak").detail["startup_peak"] < encoder
        )

    def test_large_token_cap_charges_several_profile_images(self):
        # Arrange
        info = SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"]
        arch = ModelArch.from_hf_config(info.config)
        qwen36 = ModelSpec(
            model_id="Qwen/Qwen3.6-35B-A3B", arch=arch, n_params=info.num_params
        )
        a100 = DeviceSpec(total_bytes=80 * GiB, name="NVIDIA A100-SXM4-80GB")
        settings = GenerationSettings(
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            max_num_batched_tokens=32768,
            tensor_parallel_size=2,
        )

        # Act
        breakdown = estimate_generation(qwen36, a100, settings)

        # Assert
        encoder, _ = vision.encoder_profile_bytes(arch, 2.0, 2, 2)
        assert component(breakdown, "activation_peak").detail["startup_peak"] == encoder

    def test_training_keeps_the_fragmentation_band(self, model, device):
        # Act
        breakdown = estimate_training(model, device, TrainingSettings(algorithm="grpo"))

        # Assert
        assert (
            breakdown.device_usable_bytes
            == device.usable_bytes
            == int(device.total_bytes * 0.95)
        )


class TestEngineCompileCopy:
    def gemma(self) -> ModelSpec:
        info = SUPPORTED_MODEL_INFO["google/gemma-4-E4B-it"]
        return ModelSpec(
            model_id="gemma",
            arch=ModelArch.from_hf_config(info.config),
            n_params=info.num_params,
        )

    def test_gemma4_compiling_engine_fails_on_an_l4_at_0_85(self):
        # Arrange: the CISPO-DATASET-ROLLOUT engine. vLLM had -1.04 GiB of KV.
        settings = GenerationSettings(
            gpu_memory_utilization=0.85,
            max_num_seqs=16,
            max_model_len=512,
            max_num_batched_tokens=2048,
            max_loras=1,
            max_lora_rank=16,
        )
        l4 = DeviceSpec(total_bytes=int(22.03 * GiB), name="NVIDIA L4")

        # Act
        compiled = generation_can_serve(estimate_generation(self.gemma(), l4, settings))
        eager = generation_can_serve(
            estimate_generation(
                self.gemma(), l4, settings.model_copy(update={"enforce_eager": True})
            )
        )

        # Assert: the compile pass's per-layer embedding copy is what runs out.
        assert not compiled
        assert eager

    def test_model_without_per_layer_embeddings_pays_nothing(self, model):
        settings = GenerationSettings(gpu_memory_utilization=0.5, max_model_len=1024)
        device = DeviceSpec(total_bytes=24 * GiB, name="NVIDIA L4")

        compiled = component(
            estimate_generation(model, device, settings), "kv_cache"
        ).n_bytes
        eager = component(
            estimate_generation(
                model, device, settings.model_copy(update={"enforce_eager": True})
            ),
            "kv_cache",
        ).n_bytes

        assert eager - compiled == formulas.CUDA_GRAPH_POOL_BYTES


class TestGenerationCanServe:
    def test_requires_the_pool_to_cover_demand(self, model, device):
        fits = estimate_generation(
            model,
            device,
            GenerationSettings(gpu_memory_utilization=0.5, max_model_len=1024),
        )
        assert generation_can_serve(fits)

        starved = estimate_generation(
            model,
            device,
            GenerationSettings(
                max_model_len=32768,
                max_num_seqs=64,
                kv_cache_memory_bytes=1 * GiB,
            ),
        )
        assert starved.fits
        assert not generation_can_serve(starved)

    def test_rejects_a_phase_that_does_not_fit(self, model):
        tiny = DeviceSpec(total_bytes=1 * MiB)
        breakdown = estimate_generation(model, tiny, GenerationSettings())

        assert not breakdown.fits
        assert not generation_can_serve(breakdown)

    def test_rejects_generation_with_no_kv_component(self, model, device):
        estimate = estimate_run(
            RunConfig(
                model=model,
                train_device=device,
                gen_device=device,
                training=TrainingSettings(algorithm="sft"),
            )
        )

        assert estimate.generation.components == ()
        assert not generation_can_serve(estimate.generation)

    def test_margin_must_stay_free_after_demand(self, model, device):
        # Arrange
        breakdown = estimate_generation(model, device, GenerationSettings())
        kv = component(breakdown, "kv_cache").detail
        spare = kv["pool"] - kv["worst_case_demand"]

        # Act / Assert
        assert generation_can_serve(breakdown, spare)
        assert not generation_can_serve(breakdown, spare + 1)


class TestEstimateRun:
    def test_sizes_each_phase_on_its_own_device(self, model, device):
        gen = DeviceSpec(total_bytes=16 * GiB)
        estimate = estimate_run(
            RunConfig(model=model, train_device=device, gen_device=gen)
        )

        assert estimate.training.device_total_bytes == device.total_bytes
        assert estimate.generation.device_total_bytes == gen.total_bytes
        assert estimate.fits
        assert estimate.training.headroom_bytes == (
            estimate.training.device_usable_bytes - estimate.training.total_bytes
        )
        assert [c.key for c in estimate.generation.components] == [
            "weights",
            "kv_cache",
            "activation_peak",
            "cuda_graphs",
            "mamba_state",
            "lora_slots",
            "overhead",
        ]

    def test_serializes_with_bytes_alias(self, model, device):
        gen = DeviceSpec(total_bytes=16 * GiB)
        estimate = estimate_run(
            RunConfig(model=model, train_device=device, gen_device=gen)
        )
        payload = estimate.model_dump(mode="json", by_alias=True)
        first = payload["training"]["components"][0]
        assert "bytes" in first
        assert "bytes_" not in first


class TestDeviceSpecContextBytesProperty:
    def test_prefers_named_device_then_default(self):
        assert (
            DeviceSpec(total_bytes=24 * GiB, name="NVIDIA L4").context_bytes
            == 694 * MiB
        )
        assert (
            DeviceSpec(total_bytes=80 * GiB, name="NVIDIA H100 80GB HBM3").context_bytes
            == 1440 * MiB
        )
        assert (
            DeviceSpec(total_bytes=24 * GiB, name="mystery-card").context_bytes
            == 1268 * MiB
        )
        assert (
            DeviceSpec(
                total_bytes=24 * GiB, name="NVIDIA L4", cuda_context_bytes=123
            ).context_bytes
            == 123
        )

    def test_usable_bytes_uses_the_available_override(self):
        pinned = DeviceSpec(total_bytes=24 * GiB, available_bytes=10 * GiB)

        assert pinned.usable_bytes == 10 * GiB
        assert DeviceSpec(total_bytes=24 * GiB).usable_bytes == int(24 * GiB * 0.95)


class TestDeviceSpecGenerationUsableBytesProperty:
    def test_is_the_whole_card(self):
        assert DeviceSpec(total_bytes=80 * GiB).generation_usable_bytes == 80 * GiB

    def test_uses_the_available_override(self):
        pinned = DeviceSpec(total_bytes=80 * GiB, available_bytes=60 * GiB)

        assert pinned.generation_usable_bytes == 60 * GiB


class TestTrainingSettingsInit:
    def test_chunk_rows_must_be_positive(self):
        with pytest.raises(ValidationError, match="chunk_rows"):
            TrainingSettings(chunk_rows=0)

    def test_fuse_actor_critic_pass_is_ppo_only(self):
        assert TrainingSettings(algorithm="ppo", fuse_actor_critic_pass=True)

        with pytest.raises(
            ValidationError,
            match="fuse_actor_critic_pass=True requires algorithm='ppo', got 'grpo'",
        ):
            TrainingSettings(algorithm="grpo", fuse_actor_critic_pass=True)

    def test_micro_batch_size_must_be_positive(self):
        with pytest.raises(ValidationError, match="micro_batch_size"):
            TrainingSettings(micro_batch_size=0)

    def test_sft_drops_the_unspecified_policy_head_grad(self):
        assert TrainingSettings(algorithm="sft").liger_weight_grad is False
        assert TrainingSettings().liger_weight_grad is True
        assert (
            TrainingSettings(algorithm="sft", liger_weight_grad=True).liger_weight_grad
            is True
        )


class TestGenerationSettings:
    def test_concurrency_caps_at_max_num_seqs(self):
        capped = GenerationSettings(max_num_seqs=8, concurrent_requests=16)
        below = GenerationSettings(max_num_seqs=8, concurrent_requests=4)

        assert capped.concurrency == 8
        assert below.concurrency == 4
        assert GenerationSettings(max_num_seqs=8).concurrency == 8

    def test_rejects_utilization_outside_the_unit_interval(self):
        with pytest.raises(ValidationError, match="gpu_memory_utilization"):
            GenerationSettings(gpu_memory_utilization=0.0)
        with pytest.raises(ValidationError, match="gpu_memory_utilization"):
            GenerationSettings(gpu_memory_utilization=1.1)


class TestModelSpecVariant:
    def test_raises_for_an_unknown_name(self, model):
        with pytest.raises(KeyError, match="nf4"):
            model.variant("nf4")

        assert model.variant("base") == WeightVariant()


class TestDistributedTerms:
    """Flat data parallel and FSDP2."""

    def test_flat_dp_keeps_full_grads_and_adam(self, model, device):
        single = estimate_training(model, device, TrainingSettings())
        group = estimate_training(model, device, TrainingSettings(n_training_gpus=4))

        assert component(group, "base_weights").n_bytes == (
            component(single, "base_weights").n_bytes
        )
        assert component(group, "grads").n_bytes == component(single, "grads").n_bytes
        trained = formulas.lora_param_count(model.arch, 16)
        assert component(single, "optimizer_state").n_bytes == trained * 8
        assert component(group, "optimizer_state").n_bytes == trained * 8
        assert any("data parallel" in w for w in group.warnings)

    def test_fsdp_offload_flags_are_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            TrainingSettings(fsdp=FSDPConfig(cpu_offload=True, optim_cpu_offload=True))

    def test_fsdp_on_one_gpu_keeps_the_base_and_stores_lora_in_bf16(
        self, model, device
    ):
        flat = estimate_training(model, device, TrainingSettings())
        sharded = estimate_training(model, device, TrainingSettings(fsdp=FSDPConfig()))

        assert component(sharded, "base_weights").n_bytes == (
            component(flat, "base_weights").n_bytes
        )
        assert component(sharded, "adapters").n_bytes == pytest.approx(
            component(flat, "adapters").n_bytes / 2, abs=1
        )

    def test_fsdp_reshard_divides_the_body_and_keeps_the_embedding(self, model, device):
        counts = formulas.param_counts(model.arch)
        token = formulas.token_embedding_params(model.arch)
        gathered = formulas.fsdp_gathered_params(
            counts,
            model.arch,
            n_gpus=4,
            reshard_after_forward=True,
            prefetch_units=1,
            wrap_every_n_blocks=1,
            cpu_offload=False,
        )
        expected = (token + (counts.total - token) / 4 + gathered) * 2

        sharded = estimate_training(
            model,
            device,
            TrainingSettings(fsdp=FSDPConfig(), n_training_gpus=4),
        )

        assert component(sharded, "base_weights").n_bytes == pytest.approx(
            expected, abs=1
        )
        assert any("resharded" in w for w in sharded.warnings)

    def test_fsdp_slack_adds_the_all_gather_pool(self, model, device):
        # Arrange
        counts = formulas.param_counts(model.arch)
        block = formulas.largest_block_params(counts, model.arch)

        # Act
        sharded = estimate_training(
            model,
            device,
            TrainingSettings(fsdp=FSDPConfig(), n_training_gpus=4),
        )
        flat = estimate_training(model, device, TrainingSettings(n_training_gpus=4))

        # Assert: a tied dense model pools its two-block prefetch window.
        slack = component(sharded, "allocator_reserve")
        assert slack.detail["all_gather_pool"] == 2 * block * 2
        assert slack.n_bytes == pytest.approx(
            slack.detail["compute_stream"] + slack.detail["all_gather_pool"], abs=1
        )
        assert component(flat, "allocator_reserve").detail["all_gather_pool"] == 0

    @pytest.mark.parametrize("max_model_len", [16384, 2048])
    def test_untied_head_counts_only_above_the_activations(self, device, max_model_len):
        # Arrange
        untied = ModelSpec(
            model_id="untied",
            arch=QWEN_05B.model_copy(update={"tied_embeddings": False}),
        )
        counts = formulas.param_counts(untied.arch)
        token = formulas.token_embedding_params(untied.arch)
        gathered = formulas.fsdp_gathered_params(
            counts,
            untied.arch,
            n_gpus=4,
            reshard_after_forward=True,
            prefetch_units=1,
            wrap_every_n_blocks=1,
            cpu_offload=False,
        )
        settings = TrainingSettings(
            fsdp=FSDPConfig(), n_training_gpus=4, max_model_len=max_model_len
        )

        # Act
        sharded = estimate_training(untied, device, settings)

        # Assert
        activations = component(sharded, "activations").n_bytes
        head_above = max(counts.lm_head * 2 - activations, 0)
        assert component(sharded, "base_weights").n_bytes == pytest.approx(
            (token + (counts.total - token) / 4 + gathered - counts.lm_head) * 2
            + head_above,
            abs=1,
        )
        assert (head_above == 0) == (max_model_len == 16384)

    def test_fsdp_without_reshard_holds_the_gathered_body(self, model, device):
        freed = estimate_training(
            model,
            device,
            TrainingSettings(fsdp=FSDPConfig(), n_training_gpus=4),
        )
        kept = estimate_training(
            model,
            device,
            TrainingSettings(
                fsdp=FSDPConfig(reshard_after_forward=False),
                n_training_gpus=4,
            ),
        )
        counts = formulas.param_counts(model.arch)
        token = formulas.token_embedding_params(model.arch)
        expected = (token + (counts.total - token) * (1 + 1 / 4)) * 2

        assert component(kept, "base_weights").n_bytes == pytest.approx(expected, abs=1)
        assert component(kept, "base_weights").n_bytes > (
            component(freed, "base_weights").n_bytes
        )
        assert any("stay gathered" in w for w in kept.warnings)

    def test_fsdp_replicates_small_lora_and_shards_the_wide_matrices(self, device):
        model = ModelSpec(model_id="wide", arch=WIDE)
        flat = estimate_training(model, device, TrainingSettings(lora_rank=16))
        sharded = estimate_training(
            model,
            device,
            TrainingSettings(
                fsdp=FSDPConfig(),
                n_training_gpus=4,
                lora_rank=16,
                trajectories_per_update=4,
            ),
        )
        replicated, sharded_params = lora_tensor_placement(
            WIDE, 16, "all-linear", 0, 100_000
        )

        # Actor and reference adapters are resident. No-grad can be the binding
        # instant, so gradient bytes come from the backward_grads detail.
        assert component(sharded, "adapters").n_bytes == pytest.approx(
            2 * (replicated * 2 + sharded_params * 2 / 4), abs=1
        )
        assert component(sharded, "activations").detail["backward_grads"] == (
            pytest.approx(replicated * 2 + sharded_params * 2 / 4, abs=1)
        )
        assert sharded_params == 3 * 16 * 11008
        assert component(sharded, "adapters").n_bytes < (
            component(flat, "adapters").n_bytes
        )

    def test_deferred_grad_sync_holds_unsharded_grads(self, device):
        model = ModelSpec(model_id="wide", arch=WIDE)
        held = estimate_training(
            model,
            device,
            TrainingSettings(
                fsdp=FSDPConfig(),
                n_training_gpus=4,
                lora_rank=16,
                group_size=8,
            ),
        )
        synced = estimate_training(
            model,
            device,
            TrainingSettings(
                fsdp=FSDPConfig(defer_grad_sync=False),
                n_training_gpus=4,
                lora_rank=16,
                group_size=8,
            ),
        )
        replicated, sharded_params = lora_tensor_placement(
            WIDE, 16, "all-linear", 0, 100_000
        )

        assert component(held, "activations").detail["backward_grads"] == pytest.approx(
            replicated * 2 + sharded_params * 4, abs=1
        )
        assert component(synced, "activations").detail["backward_grads"] == (
            pytest.approx(replicated * 2 + sharded_params * 2 / 4, abs=1)
        )

    def test_held_grads_take_the_reduce_dtype(self, device):
        model = ModelSpec(model_id="wide", arch=WIDE)
        settings = TrainingSettings(
            fsdp=FSDPConfig(reduce_dtype="bfloat16"),
            n_training_gpus=4,
            lora_rank=16,
            group_size=8,
        )
        replicated, sharded_params = lora_tensor_placement(
            WIDE, 16, "all-linear", 0, 100_000
        )

        held = estimate_training(model, device, settings)

        assert component(held, "activations").detail["backward_grads"] == pytest.approx(
            (replicated + sharded_params) * 2, abs=1
        )

    def test_optim_cpu_offload_charges_adam_only_at_the_step(self, model, device):
        offloaded = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(optim_cpu_offload=True))
        )
        resident = estimate_training(
            model,
            device,
            TrainingSettings(fsdp=FSDPConfig(optim_cpu_offload=False)),
        )
        trained = formulas.lora_param_count(model.arch, 16)

        assert component(offloaded, "optimizer_state").n_bytes == 0
        # Two bf16 moments per FSDP adapter parameter.
        assert component(offloaded, "optimizer_state").detail["step_bytes"] == (
            trained * 4
        )
        assert component(resident, "optimizer_state").n_bytes == trained * 4

    def test_optimizer_step_is_the_peak_when_adam_lives_on_cpu(self, device):
        model = ModelSpec(model_id="wide", arch=WIDE)
        breakdown = estimate_training(
            model,
            device,
            TrainingSettings(
                fsdp=FSDPConfig(optim_cpu_offload=True),
                max_model_len=8,
                lora_rank=64,
                lora_target_scope="all-linear",
            ),
        )
        detail = component(breakdown, "activations").detail

        assert component(breakdown, "activations").n_bytes == 0
        assert component(breakdown, "logits_workspace").n_bytes == 0
        assert component(breakdown, "grads").n_bytes == detail["step_grads"]
        assert detail["step_grads"] == detail["optimizer_peak"]

    def test_cpu_offload_keeps_only_the_gather_and_embedding(self, model, device):
        counts = formulas.param_counts(model.arch)
        token = formulas.token_embedding_params(model.arch)
        gathered = formulas.fsdp_gathered_params(
            counts,
            model.arch,
            n_gpus=4,
            reshard_after_forward=True,
            prefetch_units=1,
            wrap_every_n_blocks=1,
            cpu_offload=True,
        )

        sharded = estimate_training(
            model,
            device,
            TrainingSettings(
                fsdp=FSDPConfig(cpu_offload=True, optim_cpu_offload=False),
                n_training_gpus=4,
            ),
        )

        assert component(sharded, "base_weights").n_bytes == pytest.approx(
            (token + gathered) * 2, abs=1
        )
        assert component(sharded, "optimizer_state").n_bytes == 0
        assert component(sharded, "optimizer_state").detail["step_bytes"] == 0

    def test_dp_shards_the_update_rows(self):
        settings = TrainingSettings(
            n_training_gpus=4,
            trajectories_per_update=128,
        )
        # 128 rows over 4 learner shards = 32 per GPU.
        assert settings.trajectories == 32

    def test_contracted_packed_moe_charges_split_lora(self, device):
        model = ModelSpec(model_id="moe-test", arch=MOE_TINY)
        settings = TrainingSettings(
            lora_packed_target_matrices=2,
            packed_moe_dispatch="contracted",
            max_model_len=1024,
        )

        bar = estimate_training(model, device, settings)

        assert component(bar, "activations").detail["split_moe_lora"] == (
            split_moe_lora_recompute_bytes(
                MOE_TINY, 1, 1024, 2, "contracted", 2, settings.lora_rank
            )
        )

    def test_fp32_adapter_input_casts_are_flat_data_parallel_only(self, device):
        model = ModelSpec(model_id="qwen", arch=QWEN_05B)

        flat = estimate_training(model, device, TrainingSettings(n_training_gpus=4))
        sharded = estimate_training(
            model, device, TrainingSettings(n_training_gpus=4, fsdp=FSDPConfig())
        )

        assert component(flat, "activations").detail["lora_fp32_input_casts"] > 0
        assert component(sharded, "activations").detail["lora_fp32_input_casts"] == 0

    def test_block_exclusive_block_pays_its_own_layers_dropout(self, device):
        # Nano's widest block is a Mamba layer; its only adapted input is
        # in_proj's residual stream.
        arch = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)
        model = ModelSpec(model_id="nano", arch=arch)
        settings = TrainingSettings(
            n_training_gpus=4,
            fsdp=FSDPConfig(),
            lora_dropout=0.05,
            max_model_len=16384,
        )

        detail = component(estimate_training(model, device, settings), "activations")

        assert detail.detail["block_recompute"] == formulas.mamba_block_bytes(
            arch, 1, 16384, 2.0, backward=True
        )
        assert detail.detail["lora_dropout"] == formulas.lora_dropout_bytes(
            arch, 1, 16384, 2.0, layer="mamba"
        )
        assert detail.detail["lora_dropout"] < formulas.lora_dropout_bytes(
            arch, 1, 16384, 2.0
        )

    def test_lora_dropout_charges_the_adapter_input_copies(self, device):
        model = ModelSpec(model_id="qwen", arch=QWEN_05B)

        without = estimate_training(model, device, TrainingSettings())
        with_dropout = estimate_training(
            model, device, TrainingSettings(lora_dropout=0.05)
        )

        assert component(without, "activations").detail["lora_dropout"] == 0
        assert component(with_dropout, "activations").detail["lora_dropout"] == (
            formulas.lora_dropout_bytes(QWEN_05B, 1, 1024, 4.0)
        )

    def test_activation_offload_drops_checkpoint_saves(self, model, device):
        on_device = estimate_training(
            model, device, TrainingSettings(activation_offload=False)
        )
        offloaded = estimate_training(
            model, device, TrainingSettings(activation_offload=True)
        )

        assert component(on_device, "activations").detail["checkpoint_boundaries"] > 0
        assert component(offloaded, "activations").detail["checkpoint_boundaries"] == 0


class TestOrchestratedOverhead:
    def test_orchestrated_charges_the_job_overhead_on_both_phases(self, model, device):
        plain_t = estimate_training(model, device, TrainingSettings())
        orchestrated_t = estimate_training(
            model, device, TrainingSettings(), orchestrated=True
        )
        assert (
            component(orchestrated_t, "overhead").n_bytes
            - component(plain_t, "overhead").n_bytes
        ) == formulas.JOB_OVERHEAD_BYTES
        plain_g = estimate_generation(model, device, GenerationSettings())
        orchestrated_g = estimate_generation(
            model, device, GenerationSettings(), orchestrated=True
        )
        assert (
            component(orchestrated_g, "overhead").n_bytes
            - component(plain_g, "overhead").n_bytes
        ) == formulas.JOB_OVERHEAD_BYTES


class TestDistributedTrainerOverhead:
    def test_multi_rank_trainer_charges_nccl_instead_of_the_library_term(
        self, model, device
    ):
        # Act
        single = estimate_training(model, device, TrainingSettings(n_training_gpus=1))
        data_parallel = estimate_training(
            model, device, TrainingSettings(n_training_gpus=4)
        )
        fsdp = estimate_training(
            model,
            device,
            TrainingSettings(n_training_gpus=4, fsdp=FSDPConfig()),
        )

        # Assert
        one_rank = component(single, "overhead").detail
        assert one_rank["trainer_lib_overhead"] == formulas.TRAINER_LIB_OVERHEAD_BYTES
        assert one_rank["distributed_trainer_overhead"] == 0
        world = trainer_nccl_bytes(device.nccl_group_bytes, 4, 4)
        for estimate in (data_parallel, fsdp):
            overhead = component(estimate, "overhead")
            assert overhead.detail["trainer_lib_overhead"] == 0
            assert overhead.detail["distributed_trainer_overhead"] == world
            assert overhead.n_bytes == sum(overhead.detail.values())

    def test_mesh_process_groups_add_nccl_overhead(self, model, device):
        # Arrange
        common = {"n_training_gpus": 4, "max_model_len": 128}
        group = device.nccl_group_bytes

        # Act
        fsdp = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(), **common)
        )
        hsdp = estimate_training(
            model,
            device,
            TrainingSettings(fsdp=FSDPConfig(shard_group_size=2), **common),
        )
        tensor_parallel = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(tp=2), **common)
        )

        # Assert: HSDP adds its shard and replicate groups; TP adds its group,
        # the TP-replica group and the TP runtime, even without a TP plan.
        def nccl(breakdown):
            return component(breakdown, "overhead").detail[
                "distributed_trainer_overhead"
            ]

        two_rank_groups = int(group * 2 * nccl_on_node_fraction(2))
        assert nccl(hsdp) - nccl(fsdp) == pytest.approx(two_rank_groups, abs=1)
        assert nccl(tensor_parallel) - nccl(fsdp) == pytest.approx(
            two_rank_groups + TP_RUNTIME_BYTES, abs=1
        )

    def test_ranks_across_nodes_hold_less_nccl_than_one_node(self, model, device):
        # Arrange
        settings = TrainingSettings(n_training_gpus=8, fsdp=FSDPConfig())
        two_nodes = device.model_copy(update={"gpus_per_node": 4})

        # Act
        one_node_bar = estimate_training(model, device, settings)
        two_node_bar = estimate_training(model, two_nodes, settings)

        # Assert
        def nccl(breakdown):
            return component(breakdown, "overhead").detail[
                "distributed_trainer_overhead"
            ]

        assert nccl(two_node_bar) < nccl(one_node_bar)
        assert nccl(two_node_bar) == trainer_nccl_bytes(
            device.nccl_group_bytes, 8, 8, gpus_per_node=4
        )

    def test_nccl_group_bytes_follow_the_named_device_then_default(self):
        assert DeviceSpec(
            total_bytes=80 * GiB, name="NVIDIA A100-SXM4-80GB"
        ).nccl_group_bytes == (820 * MiB)
        assert DeviceSpec(
            total_bytes=80 * GiB, name="NVIDIA H100 80GB HBM3"
        ).nccl_group_bytes == (950 * MiB)
        assert (
            DeviceSpec(total_bytes=24 * GiB, name="NVIDIA L4").nccl_group_bytes
            == 125 * MiB
        )
        assert DeviceSpec(
            total_bytes=24 * GiB, name="mystery-card"
        ).nccl_group_bytes == (820 * MiB)
        assert DeviceSpec(total_bytes=24 * GiB).nccl_group_bytes == 820 * MiB


class TestAlgorithmTerms:
    """PPO, SFT, and DPO structure."""

    def test_ppo_carries_a_critic_adapter_and_value_head(self, model, device):
        # PPO fuses reference, actor, and critic rows; trains actor + critic plus a Linear(hidden -> 1) value head.
        grpo = TrainingSettings(algorithm="grpo")
        ppo = TrainingSettings(algorithm="ppo")

        assert ppo.n_resident_adapters == 3
        assert ppo.n_trained_adapters == 2
        assert grpo.n_trained_adapters == 1

        ppo_bd = estimate_training(model, device, ppo)
        grpo_bd = estimate_training(model, device, grpo)
        assert (
            component(ppo_bd, "adapters").n_bytes
            > component(grpo_bd, "adapters").n_bytes
        )
        assert component(ppo_bd, "grads").n_bytes > component(grpo_bd, "grads").n_bytes
        # The critic rows run as their own micro-batch forwards.
        assert (
            component(ppo_bd, "activations").detail["nograd_peak"]
            == component(grpo_bd, "activations").detail["nograd_peak"]
        )
        assert ppo_bd.total_bytes > grpo_bd.total_bytes

    def test_ppo_value_head_shards_below_the_persistence_threshold(self, model, device):
        replicated = estimate_training(
            model,
            device,
            TrainingSettings(
                algorithm="ppo",
                n_training_gpus=4,
                fsdp=FSDPConfig(param_persistence_threshold=100_000),
            ),
        )
        sharded = estimate_training(
            model,
            device,
            TrainingSettings(
                algorithm="ppo",
                n_training_gpus=4,
                fsdp=FSDPConfig(param_persistence_threshold=0),
            ),
        )

        assert (
            component(sharded, "adapters").n_bytes
            < component(replicated, "adapters").n_bytes
        )

    def test_ppo_split_pass_that_exceeds_the_device_does_not_fit(self, model):
        small = DeviceSpec(total_bytes=4 * GiB)

        bar = estimate_training(model, small, TrainingSettings(algorithm="ppo"))

        assert bar.total_bytes > small.usable_bytes
        assert not bar.fits

    def test_dpo_holds_both_preference_graphs(self, model, device):
        grpo = TrainingSettings(algorithm="grpo")
        dpo = TrainingSettings(algorithm="dpo")
        assert dpo.grad_graph_rows == 2
        assert grpo.grad_graph_rows == 1
        # Chosen + rejected graphs share the backward instant.
        g = estimate_training(model, device, grpo)
        d = estimate_training(model, device, dpo)
        assert (
            component(d, "activations").detail["backward_peak"]
            > component(g, "activations").detail["backward_peak"]
        )

    def test_dpo_liger_loss_tiles_a_whole_pair_of_sequences(self, model, device):
        # Arrange: Liger's preference loss chunks over chosen/rejected pairs.
        token = TrainingSettings(algorithm="dpo", max_model_len=1024)
        pair = token.model_copy(update={"liger_sequence_chunks": True})

        # Act
        token_bar = estimate_training(model, device, token)
        pair_bar = estimate_training(model, device, pair)

        # Assert
        vocab = model.arch.vocab_size
        tile = 2 * 1024 * vocab * formulas.LIGER_SEQUENCE_LOGIT_BYTES
        assert (
            component(pair_bar, "activations").detail["loss_peak"]
            > component(token_bar, "activations").detail["loss_peak"]
        )
        assert component(pair_bar, "logits_workspace").n_bytes >= tile

    def test_sft_has_no_nograd_instant_and_no_engine(self, model, device):
        settings = TrainingSettings(algorithm="sft")
        assert not settings.has_nograd_pass
        assert not settings.uses_generation_engine
        breakdown = estimate_training(model, device, settings)
        assert component(breakdown, "activations").detail["nograd_peak"] == 0

    def test_sft_has_no_reference_or_critic(self):
        settings = TrainingSettings(algorithm="sft")
        assert settings.n_resident_adapters == 1
        assert not settings.uses_reference

    def test_engineless_algorithms_get_an_empty_generation_bar(self, model, device):
        config = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(algorithm="sft"),
        )
        estimate = estimate_run(config)
        assert estimate.generation.components == ()
        assert estimate.generation.fits
        assert not generation_can_serve(estimate.generation)


class TestActivationPeakLossPeak:
    """Loss instant: the gradient pass's logit workspace."""

    def test_support_scored_loss_swaps_vocab_tiles_for_support_rows(
        self, model, device
    ):
        # Arrange
        vocab, h = QWEN_05B.vocab_size, QWEN_05B.hidden_size
        chunk = resolve_chunk_rows(vocab, None)
        full = TrainingSettings(max_model_len=1024)
        supported = full.model_copy(
            update={"sampling_support_width": 20, "support_scored_loss": True}
        )

        # Act
        full_estimate = estimate_training(model, device, full)
        full_peak = component(full_estimate, "activations")
        support_peak = component(
            estimate_training(model, device, supported), "activations"
        )

        # Assert: support rows also drop the fused kernel's lm_head grad.
        vocab_workspace = chunk * vocab * 8 + vocab * h * 4
        head_grad = component(full_estimate, "logits_workspace").detail[
            "weight_dtype_head_grad"
        ]
        support_rows = formulas.support_rows_bytes(QWEN_05B, chunk, 20, 2.0)
        assert head_grad > 0
        assert full_peak.detail["loss_peak"] - support_peak.detail["loss_peak"] == (
            vocab_workspace + head_grad - support_rows
        )

    def test_a_full_vocabulary_loss_keeps_its_workspace(self, model, device):
        # Arrange
        full = TrainingSettings(algorithm="ppo", max_model_len=1024)
        liger = full.model_copy(update={"sampling_support_width": 20})

        # Act
        full_peak = component(estimate_training(model, device, full), "activations")
        liger_peak = component(estimate_training(model, device, liger), "activations")

        # Assert
        assert liger_peak.detail["loss_peak"] == full_peak.detail["loss_peak"]
        assert liger_peak.detail["nograd_peak"] < full_peak.detail["nograd_peak"]


class TestActivationPeakNogradPeak:
    """No-grad logprob instant: one micro-batch forward, then the head."""

    def test_block_binds_when_it_outweighs_the_fp32_head(self, device):
        # Arrange
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)
        settings = TrainingSettings(max_model_len=32768, algorithm="ppo")

        # Act
        detail = component(estimate_training(model, device, settings), "activations")

        # Assert
        assert detail.detail["nograd_peak"] == formulas.nograd_forward_bytes(
            arch, 1, 32768, 2.0
        )

    def test_head_binds_at_short_context(self, model, device):
        # Arrange
        seq_len = 1024
        settings = TrainingSettings(max_model_len=seq_len)
        h, vocab = QWEN_05B.hidden_size, QWEN_05B.vocab_size
        tile = resolve_chunk_rows(vocab, settings.chunk_rows) * vocab * 4

        # Act
        detail = component(estimate_training(model, device, settings), "activations")

        # Assert
        assert detail.detail["nograd_peak"] == seq_len * h * 2 + tile + vocab * h * 4

    def test_supports_score_the_head_without_an_fp32_copy(self, model, device):
        # Arrange
        seq_len = 1024
        settings = TrainingSettings(max_model_len=seq_len, sampling_support_width=20)
        h, vocab = QWEN_05B.hidden_size, QWEN_05B.vocab_size
        rows = formulas.support_rows_bytes(
            QWEN_05B, resolve_chunk_rows(vocab, None), 20, 2.0
        )

        # Act
        detail = component(estimate_training(model, device, settings), "activations")

        # Assert
        assert detail.detail["nograd_peak"] == seq_len * h * 2 + rows

    def test_fused_adapter_rows_do_not_widen_the_forward(self, device):
        # Arrange
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)
        grpo = TrainingSettings(max_model_len=16384, beta=0.0)
        ppo = grpo.model_copy(update={"algorithm": "ppo", "beta": 0.001})

        # Act
        one_adapter = estimate_training(model, device, grpo)
        three_adapters = estimate_training(model, device, ppo)

        # Assert
        assert (
            component(one_adapter, "activations").detail["nograd_peak"]
            == component(three_adapters, "activations").detail["nograd_peak"]
        )

    def test_vision_rows_of_two_chunks_are_on_the_device(self, device):
        # Arrange: 4 images of 1280x2048, 60 MiB of fp32 pixels each.
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)
        text = TrainingSettings(max_model_len=32768, micro_batch_size=2)
        images = text.model_copy(
            update={"images_per_row": 4, "image_width": 1280, "image_height": 2048}
        )

        # Act
        text_detail = component(estimate_training(model, device, text), "activations")
        image_detail = component(
            estimate_training(model, device, images), "activations"
        )

        # Assert
        assert image_detail.detail["nograd_vision_rows"] == 2 * 2 * 4 * 60 * MiB
        assert (
            image_detail.detail["nograd_peak"] - text_detail.detail["nograd_peak"]
            == 2 * 2 * 4 * 60 * MiB
        )


class TestActivationPeakSegmentBalancePeak:
    """Balancing segment rows holds one row to and from each lockstep peer."""

    def test_holds_a_row_each_way_per_peer(self, device):
        # Arrange
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)
        settings = TrainingSettings(
            n_training_gpus=4,
            images_per_row=4,
            image_width=1280,
            image_height=2048,
            segmented_rows=True,
        )

        # Act
        detail = component(estimate_training(model, device, settings), "activations")

        # Assert
        assert detail.detail["segment_balance_peak"] == 2 * 3 * 4 * 60 * MiB

    def test_binds_when_it_outweighs_the_learn_passes(self, device):
        # Arrange: 8 lockstep ranks of 16 images of 1280x2048 at 1k tokens.
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)
        settings = TrainingSettings(
            max_model_len=1024,
            n_training_gpus=8,
            images_per_row=16,
            image_width=1280,
            image_height=2048,
            segmented_rows=True,
        )

        # Act
        bar = estimate_training(model, device, settings)

        # Assert
        assert component(bar, "activations").n_bytes == 2 * 7 * 16 * 60 * MiB

    def test_unsegmented_rows_stay_put(self, device):
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)
        settings = TrainingSettings(
            n_training_gpus=4, images_per_row=4, image_width=1280, image_height=2048
        )

        detail = component(estimate_training(model, device, settings), "activations")

        assert detail.detail["segment_balance_peak"] == 0


class TestFusedActorCriticPass:
    """PPO actor and critic rows in one gradient forward and backward."""

    def test_fused_backward_charges_the_critic_row(self, model, device):
        # Arrange
        seq_len = 16384
        split = TrainingSettings(
            algorithm="ppo", max_model_len=seq_len, lora_dropout=0.05
        )
        fused = split.model_copy(update={"fuse_actor_critic_pass": True})
        h, n_layers = QWEN_05B.hidden_size, QWEN_05B.n_layers
        extra_row = (
            seq_len * h * n_layers * 2  # checkpoint boundaries
            + seq_len * h * 2  # loss hidden state
            + formulas.block_recompute_bytes(QWEN_05B, 1, seq_len, 2.0, backward=True)
            + formulas.lora_input_cast_bytes(QWEN_05B, 1, seq_len)
            + formulas.lora_dropout_bytes(QWEN_05B, 1, seq_len, 4.0)
        )

        # Act
        split_bar = estimate_training(model, device, split)
        fused_bar = estimate_training(model, device, fused)

        # Assert
        split_detail = component(split_bar, "activations").detail
        fused_detail = component(fused_bar, "activations").detail
        assert fused_bar.total_bytes > split_bar.total_bytes
        assert (
            fused_detail["backward_peak"] - split_detail["backward_peak"] == extra_row
        )
        assert fused_detail["checkpoint_boundaries"] == 2 * seq_len * h * n_layers * 2
        assert fused_detail["loss_hidden_state"] == 2 * seq_len * h * 2
        assert fused_detail["block_recompute"] == formulas.block_recompute_bytes(
            QWEN_05B, 2, seq_len, 2.0, backward=True
        )
        assert fused_detail["lora_fp32_input_casts"] == (
            formulas.lora_input_cast_bytes(QWEN_05B, 2, seq_len)
        )
        assert fused_detail["lora_dropout"] == formulas.lora_dropout_bytes(
            QWEN_05B, 2, seq_len, 4.0
        )
        # The loss instant adds the critic row's saves; logit tiles stay the same.
        assert fused_detail["loss_peak"] - split_detail["loss_peak"] == (
            seq_len * h * n_layers * 2 + seq_len * h * 2
        )

    def test_config_fits_split_but_not_fused(self, model):
        # Arrange
        budget = DeviceSpec(total_bytes=8 * GiB, available_bytes=6 * GiB)
        split = TrainingSettings(algorithm="ppo", max_model_len=16384)
        fused = split.model_copy(update={"fuse_actor_critic_pass": True})

        # Act
        split_bar = estimate_training(model, budget, split)
        fused_bar = estimate_training(model, budget, fused)

        # Assert
        assert split_bar.fits
        assert not fused_bar.fits

    def test_two_row_split_micro_batch_matches_one_row_fused_gradients(
        self, model, device
    ):
        # Arrange
        seq_len = 4096
        split = TrainingSettings(
            algorithm="ppo",
            max_model_len=seq_len,
            lora_dropout=0.05,
            micro_batch_size=2,
        )
        fused = split.model_copy(
            update={"micro_batch_size": 1, "fuse_actor_critic_pass": True}
        )

        # Act
        split_bar = estimate_training(model, device, split)
        fused_bar = estimate_training(model, device, fused)

        # Assert
        h, n_layers = QWEN_05B.hidden_size, QWEN_05B.n_layers
        split_detail = dict(component(split_bar, "activations").detail)
        fused_detail = dict(component(fused_bar, "activations").detail)
        assert split.grad_forward_rows == fused.grad_forward_rows == 2
        # Two-row no-grad forwards score one more row at the fp32 head.
        assert split_detail.pop("nograd_peak") - fused_detail.pop("nograd_peak") == (
            seq_len * h * 2
        )
        assert split_detail == fused_detail
        assert split_bar.total_bytes == fused_bar.total_bytes
        assert split_detail["checkpoint_boundaries"] == 2 * seq_len * h * n_layers * 2
        assert split_detail["block_recompute"] == formulas.block_recompute_bytes(
            QWEN_05B, 2, seq_len, 2.0, backward=True
        )

    @pytest.mark.parametrize(
        ("algorithm", "graphs"),
        [("grpo", 1), ("cispo", 1), ("ppo", 1), ("dpo", 2)],
    )
    def test_micro_batch_scales_the_gradient_rows(
        self, model, device, algorithm, graphs
    ):
        # Arrange: DPO keeps the chosen and rejected graphs until one loss.
        seq_len = 1024
        h, n_layers = QWEN_05B.hidden_size, QWEN_05B.n_layers
        one_row = TrainingSettings(algorithm=algorithm, max_model_len=seq_len)
        four_rows = one_row.model_copy(update={"micro_batch_size": 4})

        # Act
        one_bar = estimate_training(model, device, one_row)
        four_bar = estimate_training(model, device, four_rows)
        one = component(one_bar, "activations")
        four = component(four_bar, "activations")

        # Assert
        assert one.detail["checkpoint_boundaries"] == (
            graphs * seq_len * h * n_layers * 2
        )
        assert four.detail["checkpoint_boundaries"] == (
            4 * graphs * seq_len * h * n_layers * 2
        )
        assert four.detail["lora_fp32_input_casts"] == (
            formulas.lora_input_cast_bytes(QWEN_05B, 4, seq_len)
        )
        # At 1k tokens the fp32 head outweighs a block; it scores 3 more rows.
        assert four.detail["nograd_peak"] - one.detail["nograd_peak"] == (
            3 * seq_len * h * 2
        )
        assert (
            component(four_bar, "logits_workspace").n_bytes
            == component(one_bar, "logits_workspace").n_bytes
        )
        assert (
            component(four_bar, "base_weights").n_bytes
            == component(one_bar, "base_weights").n_bytes
        )
        assert (
            component(four_bar, "grads").n_bytes == component(one_bar, "grads").n_bytes
        )
        assert (
            component(four_bar, "optimizer_state").n_bytes
            == component(one_bar, "optimizer_state").n_bytes
        )

    def test_fused_contracted_moe_charges_split_lora_for_both_rows(self, device):
        model = ModelSpec(model_id="moe-test", arch=MOE_TINY)
        settings = TrainingSettings(
            algorithm="ppo",
            fuse_actor_critic_pass=True,
            lora_packed_target_matrices=2,
            packed_moe_dispatch="contracted",
            max_model_len=1024,
        )

        bar = estimate_training(model, device, settings)

        assert component(bar, "activations").detail["split_moe_lora"] == (
            split_moe_lora_recompute_bytes(
                MOE_TINY, 2, 1024, 2, "contracted", 2, settings.lora_rank
            )
        )


class TestMicroBatchAdvice:
    def test_over_budget_run_suggests_a_one_row_micro_batch(self, model):
        # Arrange
        device = DeviceSpec(total_bytes=8 * GiB, available_bytes=6 * GiB)
        config = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(max_model_len=16384, micro_batch_size=16),
        )

        # Act
        tips = advise(config)

        # Assert
        match = next(
            tip
            for tip in tips
            if tip.phase == "training" and "micro_batch_size_per_gpu" in tip.action
        )
        assert "16 -> 1" in match.action
        assert match.saves_bytes > 0

    def test_one_row_micro_batch_is_not_suggested(self, model, device):
        config = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(micro_batch_size=1),
        )

        actions = [tip.action for tip in advise(config)]

        assert all("micro_batch_size_per_gpu" not in action for action in actions)


class TestStripTowersAdvice:
    @staticmethod
    def _towered(model, variants):
        return model.model_copy(
            update={
                "arch": model.arch.model_copy(
                    update={"multimodal_tower_params": 2_000_000_000}
                ),
                "variants": variants,
            }
        )

    def test_over_budget_engine_suggests_stripping_the_towers(self, model):
        # Arrange: 4 GB of tower weights overrun a 4 GiB card's engine budget.
        device = DeviceSpec(total_bytes=4 * GiB)
        engine = WeightVariant(name="engine", stripped_multimodal=True)
        towered = self._towered(model, (WeightVariant(), engine))
        config = RunConfig(model=towered, train_device=device, gen_device=device)
        stripped = config.model_copy(
            update={
                "generation": config.generation.model_copy(
                    update={"weight_variant": "engine"}
                )
            }
        )
        expected = (
            estimate_run(config).generation.total_bytes
            - estimate_run(stripped).generation.total_bytes
        )

        # Act
        tips = advise(config)

        # Assert
        match = next(tip for tip in tips if "strip_multimodal_towers" in tip.action)
        assert match.phase == "generation"
        assert match.saves_bytes == expected
        assert expected > 0

    def test_no_stripped_variant_means_no_strip_advice(self, model):
        device = DeviceSpec(total_bytes=4 * GiB)
        towered = self._towered(model, (WeightVariant(),))
        config = RunConfig(model=towered, train_device=device, gen_device=device)

        actions = [tip.action for tip in advise(config)]

        assert all("strip_multimodal_towers" not in action for action in actions)

    def test_an_already_stripped_engine_gets_no_strip_advice(self, model):
        device = DeviceSpec(total_bytes=4 * GiB)
        engine = WeightVariant(name="engine", stripped_multimodal=True)
        towered = self._towered(model, (WeightVariant(), engine))
        config = RunConfig(
            model=towered,
            train_device=device,
            gen_device=device,
            generation=GenerationSettings(weight_variant="engine"),
        )

        actions = [tip.action for tip in advise(config)]

        assert all("strip_multimodal_towers" not in action for action in actions)

    def test_image_rows_get_no_strip_advice(self, model):
        device = DeviceSpec(total_bytes=4 * GiB)
        engine = WeightVariant(name="engine", stripped_multimodal=True)
        towered = self._towered(model, (WeightVariant(), engine))
        config = RunConfig(
            model=towered,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(
                images_per_row=1, image_width=1280, image_height=720
            ),
        )

        actions = [tip.action for tip in advise(config)]

        assert all("strip_multimodal_towers" not in action for action in actions)


class TestUnfusedMambaScan:
    def test_nemotron_h_on_compute_capability_8_9_adds_the_unfused_backward(self):
        # Arrange
        model = ModelSpec(
            model_id="nano", arch=ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)
        )
        settings = TrainingSettings(
            n_training_gpus=4, fsdp=FSDPConfig(), max_model_len=16384
        )
        ada = DeviceSpec(
            total_bytes=22 * GiB, name="NVIDIA L4", compute_capability=(8, 9)
        )
        fused = ada.model_copy(update={"compute_capability": (8, 0)})

        # Act
        unfused_bar = estimate_training(model, ada, settings)
        fused_bar = estimate_training(model, fused, settings)

        # Assert: 4x L4 FSDP at 16k measured 16.25 GiB and fit.
        assert unfused_bar.fits
        assert 2 * GiB < unfused_bar.total_bytes - fused_bar.total_bytes < 5 * GiB

    def test_other_families_and_unknown_capability_keep_the_fused_scan(self):
        nano = ModelSpec(
            model_id="nano", arch=ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)
        )
        falcon = ModelSpec(model_id="falcon", arch=ModelArch.from_hf_config(FALCON_H1))
        settings = TrainingSettings(max_model_len=4096)
        ada = DeviceSpec(
            total_bytes=22 * GiB, name="NVIDIA L4", compute_capability=(8, 9)
        )
        unknown = ada.model_copy(update={"compute_capability": None})

        assert not falcon.arch.unfused_mamba_scan_family
        assert (
            estimate_training(falcon, ada, settings).total_bytes
            == estimate_training(falcon, unknown, settings).total_bytes
        )
        # At 16k the backward is the peak, so the unfused path shows.
        long = settings.model_copy(update={"max_model_len": 16384})
        assert (
            estimate_training(nano, unknown, long).total_bytes
            < estimate_training(nano, ada, long).total_bytes
        )


# MoE and attention only, so the routed-expert block sets the backward peak.
MOE_MODEL = ModelSpec(
    model_id="nemotron-moe",
    arch=ModelArch.from_hf_config(
        {
            **NEMOTRON_H_MOE,
            "num_hidden_layers": 3,
            "layers_block_type": ["moe", "attention", "moe"],
        }
    ),
)


def moe_settings(**fsdp: object) -> TrainingSettings:
    """Expert-LoRA EP 8 SFT at 16k tokens; with no no-grad pass the backward binds."""
    return TrainingSettings(
        algorithm="sft",
        max_model_len=16384,
        lora_rank=8,
        lora_packed_target_matrices=2,
        packed_moe_dispatch="contracted",
        n_training_gpus=8,
        fsdp=FSDPConfig(ep=8, **fsdp),
    )


def host(breakdown, key):
    return next(c for c in breakdown.host if c.key == key)


def budget_device(usable_bytes: int) -> DeviceSpec:
    return DeviceSpec(total_bytes=80 * GiB, available_bytes=usable_bytes, name="budget")


class TestRoutedChunkTerm:
    def test_training_peak_grows_with_the_chunk(self, device):
        totals = [
            estimate_training(
                MOE_MODEL,
                device,
                moe_settings(routed_expert_chunk_mib=mib, optim_cpu_offload=True),
            ).total_bytes
            for mib in (64, 256, 1024)
        ]

        assert totals[0] < totals[1] < totals[2]

    def test_activations_detail_reports_the_chunk_term(self, device):
        # Arrange
        settings = moe_settings(routed_expert_chunk_mib=256, optim_cpu_offload=True)

        # Act
        breakdown = estimate_training(MOE_MODEL, device, settings)

        # Assert
        expected = formulas.routed_chunk_bytes(
            MOE_MODEL.arch,
            1,
            16384,
            2.0,
            256 * MiB,
            expert_lora=True,
            contracted=True,
            ep=8,
        )
        detail = component(breakdown, "activations").detail
        assert detail["routed_expert_chunks"] == expected > 0

    def test_recurrent_block_keeps_the_chunks_of_the_experts_beside_it(self, device):
        # Arrange
        info = SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"]
        qwen36 = ModelSpec(
            model_id="Qwen/Qwen3.6-35B-A3B",
            arch=ModelArch.from_hf_config(info.config),
            n_params=info.num_params,
        )
        settings = moe_settings(routed_expert_chunk_mib=256, optim_cpu_offload=True)

        # Act
        breakdown = estimate_training(qwen36, device, settings)

        # Assert
        detail = component(breakdown, "activations").detail
        assert detail["block_recompute"] == formulas.block_recompute_bytes(
            qwen36.arch, 1, 16384, 2.0, backward=True, layer="mamba"
        )
        assert detail["routed_expert_chunks"] == formulas.routed_chunk_bytes(
            qwen36.arch,
            1,
            16384,
            2.0,
            256 * MiB,
            expert_lora=True,
            contracted=True,
            ep=8,
        )
        assert detail["routed_expert_chunks"] > 0

    def test_tensor_parallel_ranks_split_the_routed_tokens(self, device):
        # Arrange
        settings = moe_settings(
            routed_expert_chunk_mib=256, optim_cpu_offload=True, tp=2
        ).model_copy(update={"n_training_gpus": 16})

        # Act
        breakdown = estimate_training(MOE_MODEL, device, settings)

        # Assert
        detail = component(breakdown, "activations").detail
        assert detail["routed_expert_chunks"] == formulas.routed_chunk_bytes(
            MOE_MODEL.arch,
            1,
            8192,
            2.0,
            256 * MiB,
            expert_lora=True,
            contracted=True,
            ep=8,
        )


class TestSavedRouterTopkTerm:
    def test_checkpointed_moe_training_charges_router_topk(self, device):
        settings = moe_settings(optim_cpu_offload=True)

        breakdown = estimate_training(MOE_MODEL, device, settings)

        detail = component(breakdown, "activations").detail
        # 16k tokens x top-6 x 2 MoE layers, 4 + 8 bytes per routed slot.
        assert detail["saved_router_topk"] == 16384 * 6 * 2 * 12

    def test_activation_offload_keeps_router_topk_on_device(self, device):
        settings = moe_settings(optim_cpu_offload=True).model_copy(
            update={"activation_offload": True}
        )

        breakdown = estimate_training(MOE_MODEL, device, settings)

        detail = component(breakdown, "activations").detail
        assert detail["checkpoint_boundaries"] == 0
        assert detail["saved_router_topk"] == formulas.saved_router_topk_bytes(
            MOE_MODEL.arch, 1, 16384
        )

    def test_data_parallel_checkpointing_does_not_charge_router_topk(self, device):
        settings = TrainingSettings(max_model_len=16384, n_training_gpus=8)

        breakdown = estimate_training(MOE_MODEL, device, settings)

        detail = component(breakdown, "activations").detail
        assert detail["checkpoint_boundaries"] > 0
        assert detail["saved_router_topk"] == 0

    def test_dense_or_uncheckpointed_training_charges_nothing(self, model, device):
        dense = estimate_training(model, device, TrainingSettings())
        uncheckpointed = estimate_training(
            MOE_MODEL,
            device,
            moe_settings(optim_cpu_offload=True).model_copy(
                update={"gradient_checkpointing": False}
            ),
        )

        assert component(dense, "activations").detail["saved_router_topk"] == 0
        assert component(uncheckpointed, "activations").detail["saved_router_topk"] == 0


class TestOptimizerPlacement:
    def test_data_parallel_charges_the_foreach_workspace(self, model, device):
        breakdown = estimate_training(model, device, TrainingSettings())

        trained = formulas.lora_param_count(model.arch, 16)
        detail = component(breakdown, "optimizer_state").detail
        assert detail["foreach_workspace"] == trained * 4

    def test_fsdp_gpu_optimizer_keeps_adam_resident_and_steps_fused(
        self, model, device
    ):
        breakdown = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(optim_cpu_offload=False))
        )

        trained = formulas.lora_param_count(model.arch, 16)
        optimizer = component(breakdown, "optimizer_state")
        assert optimizer.n_bytes == trained * 4
        assert optimizer.detail["foreach_workspace"] == 0

    def test_gpu_optimizer_costs_more_device_memory_than_offload(self, device):
        offloaded = estimate_training(
            MOE_MODEL,
            device,
            moe_settings(routed_expert_chunk_mib=64, optim_cpu_offload=True),
        )
        resident = estimate_training(
            MOE_MODEL,
            device,
            moe_settings(routed_expert_chunk_mib=64, optim_cpu_offload=False),
        )

        assert resident.total_bytes > offloaded.total_bytes


class TestTrainingHostBreakdown:
    def test_offloaded_optimizer_pins_moments_params_and_grads(self, model, device):
        # Arrange
        settings = TrainingSettings(fsdp=FSDPConfig(optim_cpu_offload=True))

        # Act
        breakdown = estimate_training(model, device, settings)

        # Assert
        adam = component(breakdown, "optimizer_state").detail["step_bytes"]
        step_grads = component(breakdown, "activations").detail["step_grads"]
        assert host(breakdown, "optimizer_offload").n_bytes == adam + 2 * step_grads

    def test_gpu_optimizer_pins_nothing(self, model, device):
        breakdown = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(optim_cpu_offload=False))
        )

        assert host(breakdown, "optimizer_offload").n_bytes == 0

    def test_vision_cache_holds_each_image_tower_output(self, device):
        # Arrange: Qwen3.5, 8 rows x 2 screenshots of 44 x 80 patches.
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        vision_model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)
        screenshots = TrainingSettings(
            images_per_row=2, image_width=1280, image_height=720
        )

        # Act
        without = estimate_training(vision_model, device, TrainingSettings())
        with_images = estimate_training(vision_model, device, screenshots)

        # Assert
        cache = host(with_images, "vision_feature_cache").n_bytes
        # Pinned blocks: 3520 x 768 bf16 rounds to 8 MiB, 880 x 1024 to 2 MiB.
        assert cache == 8 * 2 * (8 + 2) * MiB
        assert host(without, "vision_feature_cache").n_bytes == 0
        assert with_images.total_bytes == without.total_bytes

    def test_vision_rows_live_once_beside_two_pinned_staging_buffers(self, device):
        # Arrange: 8 rows x 4 images of 1280x2048, 60 MiB of fp32 pixels each.
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        vision_model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)
        settings = TrainingSettings(
            images_per_row=4, image_width=1280, image_height=2048
        )

        # Act
        bar = estimate_training(vision_model, device, settings)

        # Assert
        assert host(bar, "vision_rows").n_bytes == 8 * 4 * 60 * MiB
        # One micro-batch row is 240 MiB, in a 256 MiB pinned block.
        assert host(bar, "vision_row_staging").n_bytes == 2 * 256 * MiB

    def test_text_rows_hold_no_vision_rows(self, device):
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        vision_model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)

        bar = estimate_training(vision_model, device, TrainingSettings())

        assert host(bar, "vision_rows").n_bytes == 0
        assert host(bar, "vision_row_staging").n_bytes == 0

    def test_tower_lora_bypasses_the_cache_and_holds_tower_activations(self, device):
        # Arrange
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        )
        vision_model = ModelSpec(model_id="Qwen/Qwen3.5-0.8B", arch=arch)
        frozen = TrainingSettings(
            images_per_row=2, image_width=1280, image_height=2048, lora_dropout=0.05
        )
        trained = frozen.model_copy(update={"vision_tower_lora": True})

        # Act
        frozen_bar = estimate_training(vision_model, device, frozen)
        trained_bar = estimate_training(vision_model, device, trained)

        # Assert
        frozen_detail = component(frozen_bar, "activations").detail
        trained_detail = component(trained_bar, "activations").detail
        assert host(trained_bar, "vision_feature_cache").n_bytes == 0
        assert host(frozen_bar, "vision_feature_cache").n_bytes > 0
        # fp32 adapters: 480 MiB of tower tensors plus 150 MiB of merger dropout.
        assert frozen_detail["vision_tower_lora"] == 0
        assert trained_detail["vision_tower_lora"] == 630 * MiB
        assert (
            trained_detail["backward_peak"] - frozen_detail["backward_peak"]
            == 630 * MiB
        )
        assert trained_detail["loss_peak"] - frozen_detail["loss_peak"] == 630 * MiB

    def test_catalog_vision_family_charges_no_feature_cache(self, device):
        # Arrange
        vision_model = ModelSpec(
            model_id="super-vl", arch=ModelArch.from_hf_config(SUPER_VL.config)
        )
        screenshots = TrainingSettings(
            images_per_row=4, image_width=1280, image_height=720
        )

        # Act
        breakdown = estimate_training(vision_model, device, screenshots)

        # Assert
        assert host(breakdown, "vision_feature_cache").n_bytes == 0

    @pytest.mark.parametrize(
        ("hub_id", "row_bytes"),
        [
            ("Qwen/Qwen3.5-0.8B", 44 * 80 * 1536 * 4),
            ("google/gemma-4-E4B-it", 36 * 66 * 768 * 4),
            ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", 46 * 80 * 768 * 4),
        ],
    )
    def test_async_rollout_holds_the_next_updates_pixel_rows(
        self, device, hub_id, row_bytes
    ):
        # Arrange: 8 rows x 4 screenshots at the 1280x720 viewport.
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config)
        vision_model = ModelSpec(model_id=hub_id, arch=arch)
        text = TrainingSettings(async_rollout=True)
        screenshots = text.model_copy(
            update={"images_per_row": 4, "image_width": 1280, "image_height": 720}
        )

        # Act
        without = estimate_training(vision_model, device, text)
        with_images = estimate_training(vision_model, device, screenshots)

        # Assert
        pixels = (
            host(with_images, "prefetched_rollout").n_bytes
            - host(without, "prefetched_rollout").n_bytes
        )
        assert pixels == 8 * 4 * row_bytes

    def test_checkpoint_snapshot_adds_adam_when_checkpointing_the_optimizer(
        self, model, device
    ):
        # Arrange
        trained = formulas.lora_param_count(model.arch, 16)

        # Act
        weights_only = estimate_training(model, device, TrainingSettings())
        with_optimizer = estimate_training(
            model, device, TrainingSettings(checkpoint_optimizer=True)
        )

        # Assert: flat data parallel keeps fp32 adapters.
        assert host(weights_only, "checkpoint_snapshot").n_bytes == trained * 4
        assert host(with_optimizer, "checkpoint_snapshot").n_bytes == trained * 12

    def test_async_rollout_holds_the_next_rollout_batch(self, model, device):
        held = estimate_training(model, device, TrainingSettings())
        prefetched = estimate_training(
            model, device, TrainingSettings(async_rollout=True)
        )

        rollout = component(held, "overhead").detail["rollout_tensors"]
        assert host(held, "prefetched_rollout").n_bytes == 0
        assert host(prefetched, "prefetched_rollout").n_bytes == rollout
        assert prefetched.host_total_bytes > held.host_total_bytes


class TestResolveFsdpMemory:
    def peak(self, chunk: int, offload: bool) -> int:
        return estimate_training(
            MOE_MODEL,
            budget_device(80 * GiB),
            moe_settings(routed_expert_chunk_mib=chunk, optim_cpu_offload=offload),
        ).total_bytes

    def test_set_values_are_kept(self):
        settings = moe_settings(routed_expert_chunk_mib=128, optim_cpu_offload=True)

        resolved = resolve_fsdp_memory(
            MOE_MODEL, budget_device(80 * GiB), settings, training_breakdown
        )

        assert resolved.routed_expert_chunk_mib == 128
        assert resolved.optim_cpu_offload is True

    def test_ample_budget_picks_the_largest_chunk_and_a_gpu_optimizer(self):
        resolved = resolve_fsdp_memory(
            MOE_MODEL, budget_device(80 * GiB), moe_settings(), training_breakdown
        )

        assert resolved.routed_expert_chunk_mib == 512
        assert resolved.optim_cpu_offload is False

    def test_unset_chunk_takes_the_largest_that_fits(self):
        # Arrange: 128 MiB fits with the optimizer offloaded, 256 MiB does not.
        usable = (self.peak(128, True) + self.peak(256, True)) // 2

        # Act
        resolved = resolve_fsdp_memory(
            MOE_MODEL, budget_device(usable), moe_settings(), training_breakdown
        )

        # Assert
        assert resolved.routed_expert_chunk_mib == 128

    def test_gpu_optimizer_needs_the_underprediction_buffer(self):
        # Arrange
        settings = moe_settings(routed_expert_chunk_mib=64)
        gpu_peak = self.peak(64, False)
        offload_budget = (self.peak(64, True) + gpu_peak) // 2
        buffered_budget = int(gpu_peak / (1 - formulas.MAX_UNDERPREDICTION)) + 1

        # Act
        tight = resolve_fsdp_memory(
            MOE_MODEL, budget_device(offload_budget), settings, training_breakdown
        )
        unbuffered = resolve_fsdp_memory(
            MOE_MODEL, budget_device(gpu_peak), settings, training_breakdown
        )
        roomy = resolve_fsdp_memory(
            MOE_MODEL, budget_device(buffered_budget), settings, training_breakdown
        )

        # Assert
        assert tight.optim_cpu_offload is True
        assert unbuffered.optim_cpu_offload is True
        assert roomy.optim_cpu_offload is False

    def test_nothing_fits_takes_the_smallest_chunk_and_offload(self):
        resolved = resolve_fsdp_memory(
            MOE_MODEL, budget_device(GiB), moe_settings(), training_breakdown
        )

        assert resolved.routed_expert_chunk_mib == 64
        assert resolved.optim_cpu_offload is True

    def test_full_cpu_offload_never_offloads_the_optimizer_again(self):
        settings = TrainingSettings(fsdp=FSDPConfig(cpu_offload=True))

        resolved = resolve_fsdp_memory(
            MOE_MODEL, budget_device(80 * GiB), settings, training_breakdown
        )

        assert resolved.optim_cpu_offload is False

    def test_same_inputs_pick_the_same_values(self):
        usable = (self.peak(128, True) + self.peak(256, True)) // 2

        picks = [
            resolve_fsdp_memory(
                MOE_MODEL, budget_device(usable), moe_settings(), training_breakdown
            )
            for _ in range(3)
        ]

        assert picks[0] == picks[1] == picks[2]

    def test_estimate_training_sizes_the_resolved_config_and_names_it(self):
        # Arrange
        device = budget_device(80 * GiB)

        # Act
        auto = estimate_training(MOE_MODEL, device, moe_settings())
        explicit = estimate_training(
            MOE_MODEL,
            device,
            moe_settings(routed_expert_chunk_mib=512, optim_cpu_offload=False),
        )

        # Assert
        assert auto.total_bytes == explicit.total_bytes
        assert auto.warnings[-1] == (
            "Picked from this estimate: routed_expert_chunk_mib=512, "
            "optim_cpu_offload=False (GPU, fused AdamW)."
        )

    def test_needs_an_fsdp_config(self, model, device):
        with pytest.raises(ValueError, match=r"needs settings\.fsdp"):
            resolve_fsdp_memory(model, device, TrainingSettings(), training_breakdown)

    def test_needs_a_model_when_a_device_is_set(self, device):
        # Arrange
        settings = TrainingSettings(fsdp=FSDPConfig())

        # Act / Assert
        with pytest.raises(ValueError, match="needs a model when device is set"):
            resolve_fsdp_memory(
                None,
                device,
                settings,
                training_breakdown,
            )


SUPER_VL_MODEL = ModelSpec(
    model_id="super-vl", arch=ModelArch.from_hf_config(SUPER_VL.config)
)
# torch ``total_memory`` of an A100 80GB; the H100 80GB value is assumed.
A100_80GB_CUDA_BYTES = int(79.15 * GiB)
H100_80GB_CUDA_BYTES = int(79.11 * GiB)


def super_vl_cispo(
    max_model_len: int, n_training_gpus: int, **fsdp: object
) -> TrainingSettings:
    """Super-VL CISPO with expert LoRA, EP 8 and FSDP inside groups of 8 GPUs."""
    return TrainingSettings(
        algorithm="cispo",
        trajectories_per_update=128,
        max_model_len=max_model_len,
        lora_rank=8,
        lora_packed_target_matrices=2,
        packed_moe_dispatch="contracted",
        beta=0.05,
        use_separate_reference_adapter=False,
        n_training_gpus=n_training_gpus,
        checkpoint_optimizer=True,
        async_rollout=True,
        fsdp=FSDPConfig(ep=8, shard_group_size=8, reduce_dtype="bfloat16", **fsdp),
    )


def training_gpu(name: str, cuda_bytes: int) -> DeviceSpec:
    """Training GPU on an 8-GPU node. The budget is DeviceSpec.usable_bytes."""
    return DeviceSpec(total_bytes=cuda_bytes, name=name, gpus_per_node=8)


class TestSuperVlCalibration:
    """Super-VL learn steps on A100 80GB, optimizer offloaded, rows up to 30,892."""

    @pytest.fixture
    def a100(self):
        return training_gpu("NVIDIA A100-SXM4-80GB", A100_80GB_CUDA_BYTES)

    def estimate(self, device: DeviceSpec, chunk_mib: int) -> float:
        settings = super_vl_cispo(
            30892, 8, routed_expert_chunk_mib=chunk_mib, optim_cpu_offload=True
        )
        return estimate_training(SUPER_VL_MODEL, device, settings).total_bytes / GiB

    @pytest.mark.parametrize(
        ("chunk_mib", "rank0_peak_gib"), [(64, 77.68), (256, 77.95), (1024, 78.09)]
    )
    def test_estimate_tracks_the_busiest_rank_above_rank0(
        self, a100, chunk_mib, rank0_peak_gib
    ):
        # Arrange: the busiest rank peaked at 79.2 GiB at 64 MiB and is taken
        # to rise with chunk size as rank 0 did.
        busiest_peak_gib = 79.2 + (rank0_peak_gib - 77.68)

        # Act
        estimate_gib = self.estimate(a100, chunk_mib)

        # Assert
        assert estimate_gib > rank0_peak_gib
        assert estimate_gib == pytest.approx(busiest_peak_gib, abs=1.0)

    def test_busiest_rank_stays_inside_the_runtime_headroom(self, a100):
        # Arrange: the node's busiest rank peaked at 79.2 GiB at 64 MiB.
        busiest_peak_gib = 79.2
        headroom_gib = (a100.total_bytes - a100.usable_bytes) / GiB

        # Act
        excess_gib = busiest_peak_gib - self.estimate(a100, 64)

        # Assert
        assert 0 < excess_gib < 1.0
        assert excess_gib < headroom_gib

    def test_busiest_rank_at_256_mib_stays_inside_the_runtime_headroom(self, a100):
        # Arrange: rank 0 rose 0.27 GiB from 64 to 256 MiB; the busiest rank
        # is taken to rise by the same amount from its 79.2 GiB at 64 MiB.
        busiest_peak_gib = 79.2 + (77.95 - 77.68)
        headroom_gib = (a100.total_bytes - a100.usable_bytes) / GiB

        # Act
        excess_gib = busiest_peak_gib - self.estimate(a100, 256)

        # Assert
        assert 0 < excess_gib < headroom_gib


class TestSuperVlVwaPicks:
    """VWA launch shape: 16 trainer GPUs, rows up to 24,000 + 5,120 tokens."""

    @pytest.mark.parametrize(
        ("name", "cuda_bytes"),
        [
            ("NVIDIA A100-SXM4-80GB", A100_80GB_CUDA_BYTES),
            ("NVIDIA H100 80GB HBM3", H100_80GB_CUDA_BYTES),
        ],
    )
    def test_picks_the_smallest_chunk_with_the_optimizer_offloaded(
        self, name, cuda_bytes
    ):
        resolved = resolve_fsdp_memory(
            SUPER_VL_MODEL,
            training_gpu(name, cuda_bytes),
            super_vl_cispo(29120, 16),
            training_breakdown,
        )

        assert resolved.routed_expert_chunk_mib == 64
        assert resolved.optim_cpu_offload is True

    def test_h100_estimate_at_512_mib_covers_the_measured_busiest_rank(self):
        # Arrange: H100 VWA busiest rank peaked at 76.47 GiB (NVML) at 512 MiB.
        device = training_gpu("NVIDIA H100 80GB HBM3", H100_80GB_CUDA_BYTES)
        settings = super_vl_cispo(
            29120, 16, routed_expert_chunk_mib=512, optim_cpu_offload=True
        )

        # Act
        estimate_gib = (
            estimate_training(SUPER_VL_MODEL, device, settings).total_bytes / GiB
        )

        # Assert
        assert 76.47 <= estimate_gib < device.total_bytes / GiB

    def test_h100_pick_leaves_room_for_the_busiest_rank(self):
        # Arrange: EP routing spread A100 ranks 1.5 GiB over one learn step.
        rank_spread = int(1.5 * GiB)
        device = training_gpu("NVIDIA H100 80GB HBM3", H100_80GB_CUDA_BYTES)
        settings = super_vl_cispo(29120, 16)
        resolved = resolve_fsdp_memory(
            SUPER_VL_MODEL, device, settings, training_breakdown
        )

        # Act
        peak = estimate_training(
            SUPER_VL_MODEL, device, settings.model_copy(update={"fsdp": resolved})
        ).total_bytes

        # Assert
        assert peak + rank_spread < H100_80GB_CUDA_BYTES

    def test_max_model_len_rows_fit_no_chunk(self):
        # Arrange
        device = training_gpu("NVIDIA A100-SXM4-80GB", A100_80GB_CUDA_BYTES)
        settings = super_vl_cispo(
            32768, 16, routed_expert_chunk_mib=64, optim_cpu_offload=True
        )

        # Act
        breakdown = estimate_training(SUPER_VL_MODEL, device, settings)

        # Assert
        assert not breakdown.fits


class TestTrainingSettingsShardGpusProperty:
    def test_shard_group_caps_the_shard_count(self):
        settings = super_vl_cispo(29120, 16)

        assert settings.shard_gpus == 8

    def test_without_a_shard_group_every_training_gpu_shards(self):
        settings = TrainingSettings(n_training_gpus=16, fsdp=FSDPConfig(ep=8))

        assert settings.shard_gpus == 16

    def test_replicated_groups_shard_weights_like_one_group(self):
        # Arrange
        device = training_gpu("NVIDIA A100-SXM4-80GB", A100_80GB_CUDA_BYTES)
        fsdp = {"routed_expert_chunk_mib": 256, "optim_cpu_offload": True}

        # Act
        two_groups = estimate_training(
            SUPER_VL_MODEL, device, super_vl_cispo(29120, 16, **fsdp)
        )
        one_group = estimate_training(
            SUPER_VL_MODEL, device, super_vl_cispo(29120, 8, **fsdp)
        )

        # Assert
        for key in ("base_weights", "adapters", "grads"):
            assert (
                component(two_groups, key).n_bytes == component(one_group, key).n_bytes
            )


class TestTrainingTensorParallel:
    def test_linears_are_resident_at_one_over_tp_and_the_head_is_gathered_for_logits(
        self, device
    ):
        # Arrange
        model = ModelSpec(
            model_id="nano", arch=ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)
        )
        common = {"n_training_gpus": 4, "max_model_len": 128}

        # Act
        plain = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(), **common)
        )
        parallel = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(tp=2), **common)
        )

        # Assert
        assert (
            component(parallel, "base_weights").n_bytes
            > component(plain, "base_weights").n_bytes
        )
        assert (
            component(parallel, "activations").detail["block_recompute"]
            < (component(plain, "activations").detail["block_recompute"])
        )
        assert (
            component(parallel, "activations").detail["checkpoint_boundaries"]
            == (component(plain, "activations").detail["checkpoint_boundaries"])
        )
        plain_logits = component(plain, "logits_workspace").detail
        parallel_logits = component(parallel, "logits_workspace").detail
        assert parallel_logits["chunk_tiles"] == plain_logits["chunk_tiles"]
        assert parallel_logits["tp_head_gather"] == model.arch.vocab_size * (
            model.arch.hidden_size * 2
        )
        assert not any("not sized" in note for note in parallel.warnings)

    def test_family_without_a_plan_prices_weights_and_activations_as_fsdp(
        self, model, device
    ):
        # Arrange
        common = {"n_training_gpus": 4, "max_model_len": 128}

        # Act
        plain = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(), **common)
        )
        parallel = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(tp=2), **common)
        )

        # Assert
        assert not model.arch.tensor_parallel_plan
        assert component(parallel, "base_weights").n_bytes == (
            component(plain, "base_weights").n_bytes
        )
        assert (
            component(parallel, "activations").detail["block_recompute"]
            == (component(plain, "activations").detail["block_recompute"])
        )

    def test_family_without_a_plan_shards_an_untied_head(self, device):
        # Arrange
        arch = QWEN_05B.model_copy(update={"tied_embeddings": False})
        model = ModelSpec(model_id="untied", arch=arch)
        head_bytes = formulas.param_counts(arch).lm_head * 2
        common = {"n_training_gpus": 4, "max_model_len": 128}

        # Act
        plain = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(), **common)
        )
        parallel = estimate_training(
            model, device, TrainingSettings(fsdp=FSDPConfig(tp=2), **common)
        )

        # Assert: the head leaves its 1/4 FSDP shard and full gather for a 1/2 TP
        # shard, and the logprob loop gathers it to a dense copy at the loss.
        assert component(parallel, "base_weights").n_bytes == pytest.approx(
            component(plain, "base_weights").n_bytes - 0.75 * head_bytes, abs=1
        )
        assert (
            component(parallel, "activations").detail["loss_peak"]
            == component(plain, "activations").detail["loss_peak"] + head_bytes
        )
        assert (
            component(parallel, "activations").detail["block_recompute"]
            == (component(plain, "activations").detail["block_recompute"])
        )

    def test_trajectories_divide_by_data_parallel_ranks(self):
        settings = TrainingSettings(
            n_training_gpus=4,
            trajectories_per_update=128,
            fsdp=FSDPConfig(tp=2),
        )

        assert settings.data_parallel_gpus == 2
        assert settings.trajectories == 64


class TestTrainingExpertParallel:
    def test_ep_filling_the_group_drops_the_expert_gather(self, device):
        # Arrange
        common = {"n_training_gpus": 8, "max_model_len": 1024, "algorithm": "sft"}
        routed = formulas.routed_expert_params_per_layer(MOE_MODEL.arch)

        # Act
        plain = estimate_training(
            MOE_MODEL, device, TrainingSettings(fsdp=FSDPConfig(), **common)
        )
        parallel = estimate_training(
            MOE_MODEL, device, TrainingSettings(fsdp=FSDPConfig(ep=8), **common)
        )

        # Assert: the plain window gathers two neighbouring units, one of
        # them a full expert layer; EP gathers none.
        saved = (
            component(plain, "base_weights").n_bytes
            - component(parallel, "base_weights").n_bytes
        )
        assert saved >= routed * 2

    def test_leftover_ranks_add_one_local_expert_layer(self, device):
        common = {"n_training_gpus": 8, "max_model_len": 1024, "algorithm": "sft"}
        routed = formulas.routed_expert_params_per_layer(MOE_MODEL.arch)

        filled = estimate_training(
            MOE_MODEL, device, TrainingSettings(fsdp=FSDPConfig(ep=8), **common)
        )
        leftover = estimate_training(
            MOE_MODEL, device, TrainingSettings(fsdp=FSDPConfig(ep=4), **common)
        )

        assert component(leftover, "base_weights").n_bytes == pytest.approx(
            component(filled, "base_weights").n_bytes + routed // 4 * 2, abs=8
        )

    def test_expert_lora_state_splits_over_the_ep_group(self, device):
        # Arrange
        common = {
            "n_training_gpus": 8,
            "max_model_len": 1024,
            "algorithm": "sft",
            "lora_packed_target_matrices": 2,
        }
        expert_lora = formulas.packed_lora_param_count(MOE_MODEL.arch, 16, 2)

        # Act
        ep8 = estimate_training(
            MOE_MODEL, device, TrainingSettings(fsdp=FSDPConfig(ep=8), **common)
        )
        ep4 = estimate_training(
            MOE_MODEL, device, TrainingSettings(fsdp=FSDPConfig(ep=4), **common)
        )

        # Assert: bf16 Adam moments over an eighth vs a quarter of the experts.
        adam_gap = (
            component(ep4, "optimizer_state").n_bytes
            - component(ep8, "optimizer_state").n_bytes
        )
        assert adam_gap == pytest.approx(
            expert_lora * formulas.ADAM_MOMENTS_PER_PARAM * 2 * (1 / 4 - 1 / 8),
            rel=1e-6,
        )

    def test_token_exchange_is_reported_and_halves_under_tp(self, device):
        common = {"n_training_gpus": 8, "max_model_len": 1024, "algorithm": "sft"}

        ep = estimate_training(
            MOE_MODEL, device, TrainingSettings(fsdp=FSDPConfig(ep=4), **common)
        )
        ep_tp = estimate_training(
            MOE_MODEL, device, TrainingSettings(fsdp=FSDPConfig(ep=4, tp=2), **common)
        )
        plain = estimate_training(
            MOE_MODEL, device, TrainingSettings(fsdp=FSDPConfig(), **common)
        )

        exchange = component(ep, "activations").detail["ep_token_exchange"]
        assert exchange == formulas.ep_exchange_bytes(MOE_MODEL.arch, 1, 1024, 2.0, 4)
        assert component(ep_tp, "activations").detail["ep_token_exchange"] * 2 == (
            exchange
        )
        assert component(plain, "activations").detail["ep_token_exchange"] == 0


class TestFp8TensorParallelEngine:
    def test_weights_are_one_byte_and_split_across_the_group(self, model, device):
        # Arrange / Act
        full = estimate_generation(model, device, GenerationSettings())
        sharded = estimate_generation(
            model,
            device,
            GenerationSettings(tensor_parallel_size=4, weight_quantization="fp8"),
        )

        # Assert: fp8 is half of bf16, and four ranks share the copy.
        # This model has 2 KV heads, so TP 4 keeps one head on each rank.
        assert component(sharded, "weights").n_bytes * 8 == (
            component(full, "weights").n_bytes
        )
        assert (
            component(sharded, "kv_cache").detail["worst_case_demand"] * 2
            == (component(full, "kv_cache").detail["worst_case_demand"])
        )

    def test_super_vl_keeps_one_kv_head_and_quarters_mamba_state(self, device):
        # Arrange: Nick's engine, TP 4 and FP8, 192 sequences at 32k.
        common = {
            "max_num_seqs": 192,
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.8,
        }

        # Act
        full = estimate_generation(SUPER_VL_MODEL, device, GenerationSettings(**common))
        sharded = estimate_generation(
            SUPER_VL_MODEL,
            device,
            GenerationSettings(
                tensor_parallel_size=4, weight_quantization="fp8", **common
            ),
        )

        # Assert
        assert component(sharded, "weights").n_bytes * 8 == (
            component(full, "weights").n_bytes
        )
        assert (
            component(sharded, "kv_cache").detail["worst_case_demand"] * 2
            == (component(full, "kv_cache").detail["worst_case_demand"])
        )
        assert component(sharded, "mamba_state").n_bytes * 4 == (
            component(full, "mamba_state").n_bytes
        )

    def test_zero_kv_heads_leaves_demand_unscaled(self, device):
        # Arrange: n_kv_heads == 0 cannot scale by heads_per_rank / n_kv_heads.
        arch = ModelArch.from_hf_config(FALCON_H1).model_copy(update={"n_kv_heads": 0})
        model = ModelSpec(model_id="falcon-no-kv", arch=arch)
        settings = GenerationSettings(tensor_parallel_size=4, max_model_len=1024)

        # Act
        breakdown = estimate_generation(model, device, settings)

        # Assert
        demand = formulas.kv_cache_demand_bytes(
            arch, settings.weight_dtype, settings.concurrency, settings.max_model_len, 0
        )
        assert demand == 0
        assert component(breakdown, "kv_cache").detail["worst_case_demand"] == 0


class TestLigerSequenceLossTile:
    def test_sequence_chunks_tile_a_whole_row_at_the_loss(self, model, device):
        # Arrange: 16384-token rows outgrow the token-level tile's rows.
        token_rows = resolve_chunk_rows(model.arch.vocab_size, None)
        common = {"max_model_len": 16384}

        # Act
        token = estimate_training(model, device, TrainingSettings(**common))
        sequence = estimate_training(
            model, device, TrainingSettings(liger_sequence_chunks=True, **common)
        )

        # Assert
        token_detail = component(token, "activations").detail
        sequence_detail = component(sequence, "activations").detail
        head_grad = model.arch.vocab_size * model.arch.hidden_size * 2
        assert token_rows < 16384
        assert (
            sequence_detail["loss_peak"] - token_detail["loss_peak"]
            == (16384 * formulas.LIGER_SEQUENCE_LOGIT_BYTES - token_rows * 8)
            * model.arch.vocab_size
            - head_grad
        )
        assert sequence_detail["nograd_peak"] == token_detail["nograd_peak"]

    def test_token_loss_keeps_the_weight_dtype_head_grad(self, model, device):
        # Arrange: a short row, so the loss instant is the peak.
        token = estimate_training(model, device, TrainingSettings(max_model_len=1024))
        sequence = estimate_training(
            model,
            device,
            TrainingSettings(max_model_len=1024, liger_sequence_chunks=True),
        )
        head_grad = model.arch.vocab_size * model.arch.hidden_size * 2

        # Act
        token_logits = component(token, "logits_workspace").detail
        sequence_logits = component(sequence, "logits_workspace").detail

        # Assert
        assert token_logits["weight_dtype_head_grad"] == head_grad
        assert sequence_logits["weight_dtype_head_grad"] == 0

    def test_the_standard_path_drops_the_weight_dtype_head_grad(self, model, device):
        # Arrange: a short row, so the loss instant is the peak.
        fused = TrainingSettings(max_model_len=1024, liger_weight_grad=True)
        standard = TrainingSettings(max_model_len=1024, liger_weight_grad=False)
        head_grad = model.arch.vocab_size * model.arch.hidden_size * 2

        # Act
        fused_bar = estimate_training(model, device, fused)
        standard_bar = estimate_training(model, device, standard)

        # Assert
        assert (
            component(standard_bar, "logits_workspace").detail["weight_dtype_head_grad"]
            == 0
        )
        assert (
            component(fused_bar, "activations").detail["loss_peak"]
            - component(standard_bar, "activations").detail["loss_peak"]
            == head_grad
        )
