# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Setting inversion: largest value that still fits."""

import json
from pathlib import Path

import pytest

from agilerl.arena.memory import estimator as estimator_module
from agilerl.arena.memory import formulas, kv_cache, lora_formulas
from agilerl.arena.memory import solver as solver_module
from agilerl.arena.memory.estimator import (
    estimate_generation,
    estimate_run,
    generation_can_serve,
)
from agilerl.arena.memory.generation import KV_MARGIN_BYTES, sequences_held
from agilerl.arena.memory.manifest import lookup_gpu
from agilerl.arena.memory.solver import (
    DEFAULT_CONTEXT_LIMIT,
    CannotSolve,
    FieldSpec,
    _apply,
    architectural_context_limit,
    inference_run_config,
    size_serving_engine,
    solve,
)
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GenerationSettings,
    GiB,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingSettings,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from tests.test_memory.test_estimator import component
from tests.test_memory.test_formulas import MOE_TINY, QWEN_05B

ASSETS = Path(__file__).parent / "assets"
TINY_CONFIG = json.loads((ASSETS / "tiny_llm" / "config.json").read_text())


def lockstep_max_model_len(config, length):
    return config.model_copy(
        update={
            "training": config.training.model_copy(update={"max_model_len": length}),
            "generation": config.generation.model_copy(
                update={"max_model_len": length}
            ),
        }
    )


@pytest.fixture
def model():
    return ModelSpec(model_id="Qwen/Qwen2.5-0.5B-Instruct", arch=QWEN_05B)


@pytest.fixture
def device():
    return DeviceSpec(total_bytes=24 * GiB, name="NVIDIA L4")


@pytest.fixture
def nine_b():
    hub_id = "Qwen/Qwen3.5-9B"
    return ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config),
    )


@pytest.fixture
def nine_b_engine():
    return GenerationSettings(
        gpu_memory_utilization=0.9,
        max_num_seqs=16,
        max_lora_rank=64,
        lora_sync_scope="all-linear",
        lora_dual_stream=True,
    )


def engine_run(model, card_gib, generation):
    card = DeviceSpec(total_bytes=int(card_gib * GiB), name="card")
    return RunConfig(
        model=model, train_device=card, gen_device=card, generation=generation
    )


class TestArchitecturalContextLimit:
    def test_reads_max_position(self):
        assert architectural_context_limit(TINY_CONFIG) == 32768
        assert (
            architectural_context_limit(
                {"text_config": {"max_position_embeddings": 8192}}
            )
            == 8192
        )

    def test_no_position_limit_uses_the_default(self):
        assert architectural_context_limit({}) == DEFAULT_CONTEXT_LIMIT


class TestSolveGenerationOnly:
    def test_max_model_len_is_the_last_value_that_serves(self, model, device):
        settings = GenerationSettings(gpu_memory_utilization=0.07, max_num_seqs=8)
        result = solve(
            inference_run_config(model, device, settings),
            "max_model_len",
            hi=32768,
            phases=("generation",),
        )
        assert result.value % 16 == 0
        assert result.limited_by == "memory"
        assert generation_can_serve(result.estimate.generation)

        over = estimate_generation(
            model,
            device,
            result.config.generation.model_copy(
                update={"max_model_len": result.value + 16}
            ),
        )
        assert not generation_can_serve(over)

    def test_max_num_seqs_grows_until_the_pool_is_full(self, model, device):
        settings = GenerationSettings(gpu_memory_utilization=0.07, max_model_len=2048)
        result = solve(
            inference_run_config(model, device, settings),
            "max_num_seqs",
            hi=64,
            phases=("generation",),
        )
        over = estimate_generation(
            model,
            device,
            result.config.generation.model_copy(
                update={"max_num_seqs": result.value + 1}
            ),
        )
        assert result.value == 7
        assert result.limited_by == "memory"
        assert generation_can_serve(result.estimate.generation)
        assert not generation_can_serve(over)

    def test_generation_search_does_not_price_training(
        self, model, device, monkeypatch
    ):
        # Arrange
        settings = GenerationSettings(gpu_memory_utilization=0.07, max_model_len=2048)
        priced = []
        real = estimator_module.estimate_training

        def spy(*args, **kwargs):
            priced.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(solver_module, "estimate_training", spy)
        monkeypatch.setattr(estimator_module, "estimate_training", spy)

        # Act
        result = solve(
            inference_run_config(model, device, settings),
            "max_num_seqs",
            hi=64,
            phases=("generation",),
        )

        # Assert: one training bar, for the returned estimate.
        assert result.value == 7
        assert result.estimate.training.phase == "training"
        assert len(priced) == 1

    def test_higher_concurrency_shortens_max_model_len(self, model, device):
        def solved(seqs: int) -> int:
            settings = GenerationSettings(
                gpu_memory_utilization=0.07, max_num_seqs=seqs
            )
            return solve(
                inference_run_config(model, device, settings),
                "max_model_len",
                hi=32768,
                phases=("generation",),
            ).value

        assert solved(4) == 3856
        assert solved(8) == 1872

    def test_tiny_model_on_l4_hits_the_rope_cap(self, device):
        tiny = ModelSpec(model_id="tiny", arch=ModelArch.from_hf_config(TINY_CONFIG))
        result = solve(
            inference_run_config(tiny, device),
            "max_model_len",
            hi=architectural_context_limit(TINY_CONFIG),
            phases=("generation",),
        )
        assert result.value == 32768
        assert result.limited_by == "bound"

    def test_cannot_solve_when_the_card_cannot_hold_the_weights(self, device):
        huge = ModelSpec(
            model_id="huge",
            arch=QWEN_05B.model_copy(update={"n_layers": 400, "hidden_size": 8192}),
        )
        with pytest.raises(CannotSolve, match="no max_model_len value up to"):
            solve(
                inference_run_config(
                    huge, DeviceSpec(total_bytes=1 * GiB), GenerationSettings()
                ),
                "max_model_len",
                hi=1024,
                phases=("generation",),
            )

    def test_upper_bound_below_the_minimum_is_rejected(self, model, device):
        config = inference_run_config(model, device)

        with pytest.raises(ValueError, match="below the minimum"):
            solve(config, "max_model_len", hi=1, phases=("generation",))

    def test_two_fields_on_one_group_both_update(self, model, device):
        config = inference_run_config(model, device)
        spec = FieldSpec(
            name="both",
            group="generation",
            field="max_num_seqs",
            lo=1,
            default_hi=8,
            sync=(
                ("generation", "max_model_len"),
                ("generation", "max_num_seqs"),
            ),
        )

        updated = _apply(config, spec, 16)

        assert updated.generation.max_model_len == 16
        assert updated.generation.max_num_seqs == 16

    def test_unknown_field_stays_a_value_error(self, model, device):
        # Arrange
        config = inference_run_config(model, device)

        # Act + Assert
        with pytest.raises(ValueError, match="Unknown field"):
            solve(config, "learning_rate", phases=("generation",))


class TestInferenceRunConfig:
    def test_inference_engine_holds_no_lora_slot_or_dual_stream(self, model, device):
        # Arrange
        settings = GenerationSettings(lora_dual_stream=True)
        training = RunConfig(
            model=model, train_device=device, gen_device=device, generation=settings
        )

        # Act
        served = estimate_run(inference_run_config(model, device, settings))
        trained = estimate_run(training)

        # Assert
        slot = 2 * lora_formulas.engine_lora_param_count(QWEN_05B, 16)
        assert component(served.generation, "lora_slots").n_bytes == 0
        assert component(trained.generation, "lora_slots").n_bytes == slot
        assert (
            component(served.generation, "cuda_graphs").n_bytes
            == formulas.CUDA_GRAPH_POOL_BYTES
        )
        assert (
            component(trained.generation, "cuda_graphs").n_bytes
            == formulas.CUDA_GRAPH_POOL_BYTES + formulas.LORA_DUAL_STREAM_BYTES
        )
        assert (
            component(served.generation, "kv_cache").detail["pool"]
            == component(trained.generation, "kv_cache").detail["pool"]
            + slot
            + formulas.LORA_DUAL_STREAM_BYTES
        )

    def test_moe_inference_drops_the_expert_lora_stacks(self, device):
        # Arrange
        moe = ModelSpec(model_id="moe-test", arch=MOE_TINY)
        settings = GenerationSettings()
        training = RunConfig(
            model=moe, train_device=device, gen_device=device, generation=settings
        )

        # Act
        served = estimate_run(inference_run_config(moe, device, settings))
        trained = estimate_run(training)

        # Assert
        stacks = lora_formulas.engine_expert_lora_param_count(MOE_TINY, 16, 1, False)
        assert component(served.generation, "lora_slots").n_bytes == 0
        assert component(trained.generation, "lora_slots").n_bytes == 2 * (
            lora_formulas.engine_lora_param_count(MOE_TINY, 16) + stacks
        )


def catalogued(hub_id):
    return ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config),
    )


def serving_settings(max_model_len):
    return GenerationSettings(
        gpu_memory_utilization=0.9, max_num_seqs=8, max_model_len=max_model_len
    )


class TestSizeServingEngine:
    def test_dense_model_takes_every_gpu_of_an_h100_node(self):
        # Arrange: Qwen3-4B has 32 attention and 8 KV heads.
        h100 = lookup_gpu("H100-80").device_spec()

        # Act
        serving = size_serving_engine(
            catalogued("Qwen/Qwen3-4B"), h100, 8, serving_settings(4096)
        )

        # Assert
        assert serving.tensor_parallel_size == 8
        assert serving.fits
        assert serving.serves_all
        assert serving.full_context_sequences == 8
        assert serving.reason is None

    def test_width_matches_the_inference_run_config_estimate(self):
        # Arrange
        model = catalogued("Qwen/Qwen3-4B")
        h100 = lookup_gpu("H100-80").device_spec()
        settings = serving_settings(4096)

        # Act
        serving = size_serving_engine(model, h100, 8, settings)

        # Assert
        config = inference_run_config(
            model, h100, settings.model_copy(update={"tensor_parallel_size": 8})
        )
        bar = estimate_generation(config.model, config.gen_device, config.generation)
        assert serving.full_context_sequences == sequences_held(bar, 8)
        assert serving.serves_all == generation_can_serve(bar, KV_MARGIN_BYTES)

    @pytest.mark.parametrize(
        ("hub_id", "num_gpus", "width"),
        [
            # 32 heads split over 4 but not 5 or 6.
            ("Qwen/Qwen3-4B", 6, 4),
            # 14 heads split over 2 but not 4 or 8.
            ("Qwen/Qwen2.5-0.5B-Instruct", 8, 2),
        ],
    )
    def test_falls_back_to_the_widest_width_the_heads_split_over(
        self, hub_id, num_gpus, width
    ):
        h100 = lookup_gpu("H100-80").device_spec()

        serving = size_serving_engine(
            catalogued(hub_id), h100, num_gpus, serving_settings(4096)
        )

        assert serving.tensor_parallel_size == width
        assert serving.fits

    def test_skips_a_width_the_mamba_heads_do_not_split_over(self):
        # Arrange: attention splits over 8; 36 recurrent heads only over 4.
        model = catalogued("Qwen/Qwen3.5-9B")
        model = model.model_copy(
            update={"arch": model.arch.model_copy(update={"mamba_n_heads": 36})}
        )
        h100 = lookup_gpu("H100-80").device_spec()

        # Act
        serving = size_serving_engine(model, h100, 8, serving_settings(4096))

        # Assert
        assert model.arch.n_heads % 8 == 0
        assert not model.arch.splits_heads(8)
        assert serving.tensor_parallel_size == 4

    def test_long_context_holds_some_sequences_at_full_length(self):
        l4 = lookup_gpu("NVIDIA L4").device_spec()

        serving = size_serving_engine(
            catalogued("Qwen/Qwen3-4B"), l4, 1, serving_settings(32768)
        )

        assert serving.tensor_parallel_size == 1
        assert serving.fits
        assert not serving.serves_all
        assert serving.full_context_sequences == 2
        assert serving.reason is None

    def test_model_larger_than_the_card_does_not_fit(self):
        l4 = lookup_gpu("NVIDIA L4").device_spec()

        serving = size_serving_engine(
            catalogued("Qwen/Qwen3.8-27B"), l4, 1, serving_settings(4096)
        )

        assert serving.tensor_parallel_size == 1
        assert not serving.fits
        assert not serving.serves_all
        assert serving.full_context_sequences == 0
        assert serving.reason.startswith("the engine needs ")
        assert serving.reason.endswith(
            "tensor_parallel_size=1; the GPU has 22.0 GiB usable"
        )

    def test_kv_pool_without_room_for_one_sequence_does_not_fit(self):
        l4 = lookup_gpu("NVIDIA L4").device_spec()

        serving = size_serving_engine(
            catalogued("Qwen/Qwen3-4B"), l4, 1, serving_settings(131072)
        )

        assert not serving.fits
        assert serving.full_context_sequences == 0
        assert serving.reason == (
            "the KV cache cannot hold one 131072-token sequence at "
            "tensor_parallel_size=1"
        )


class TestSolve:
    def test_max_model_len_keeps_training_and_generation_in_lockstep(
        self, model, device
    ):
        config = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(max_model_len=512),
            generation=GenerationSettings(
                gpu_memory_utilization=0.3,
                max_model_len=512,
                max_num_seqs=4,
            ),
        )
        result = solve(config, "max_model_len", hi=2048)
        assert result.value == 2048
        assert result.config.training.max_model_len == 2048
        assert result.config.generation.max_model_len == 2048

    def test_max_num_seqs_on_an_engine_run_checks_only_generation(self, model, device):
        # Arrange: the trainer does not fit its 1 GiB card; the engine has an L4.
        config = RunConfig(
            model=model,
            train_device=DeviceSpec(total_bytes=GiB),
            gen_device=device,
            generation=GenerationSettings(
                gpu_memory_utilization=0.07, max_model_len=2048
            ),
        )

        # Act
        result = solve(config, "max_num_seqs", hi=64)
        over = estimate_generation(
            model,
            device,
            result.config.generation.model_copy(update={"max_num_seqs": 7}),
        )

        # Assert
        assert result.value == 6
        assert result.limited_by == "memory"
        assert generation_can_serve(result.estimate.generation)
        assert not generation_can_serve(over)
        assert result.unchecked_over_budget == ("training",)


class TestSolveEngineReserve:
    """vLLM reserves its gpu_memory_utilization budget at any context length."""

    def test_no_context_fits_when_the_reserve_outgrows_the_card(self, model):
        # Arrange: 0.9 of 8 GiB plus the CUDA context is more than the card.
        device = DeviceSpec(total_bytes=8 * GiB)
        settings = GenerationSettings(
            gpu_memory_utilization=0.9,
            max_num_seqs=8,
            concurrent_requests=2,
            max_num_batched_tokens=32768,
        )
        config = inference_run_config(model, device, settings)

        # Act + Assert
        with pytest.raises(CannotSolve, match="no max_model_len value up to 32768"):
            solve(config, "max_model_len", hi=32768, phases=("generation",))

    def test_the_engine_bar_does_not_move_with_context(self, model):
        # Arrange
        device = DeviceSpec(total_bytes=8 * GiB)

        def engine_bar(context: int) -> int:
            settings = GenerationSettings(
                gpu_memory_utilization=0.9,
                max_num_seqs=8,
                concurrent_requests=2,
                max_model_len=context,
                max_num_batched_tokens=32768,
            )
            run = estimate_run(inference_run_config(model, device, settings))
            return run.generation.total_bytes

        # Act
        bars = [engine_bar(context) for context in (16, 4096, 32768)]

        # Assert: the start-up profile's peak stays reserved after it frees.
        assert bars[0] == bars[1] == bars[2]
        assert bars[0] > device.generation_usable_bytes


class TestSolveWithoutAnEngine:
    @pytest.mark.parametrize("algorithm", ["sft", "dpo"])
    def test_generation_only_field_is_a_usage_error(self, model, device, algorithm):
        # Arrange
        config = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(algorithm=algorithm),
            generation=GenerationSettings(),
        )

        # Act + Assert: a usage error, not CannotSolve.
        with pytest.raises(ValueError, match="starts no engine") as exc_info:
            solve(config, "max_num_seqs", hi=64)

        assert not isinstance(exc_info.value, CannotSolve)

    def test_max_model_len_on_sft_checks_the_training_bar(self, model, device):
        # Arrange
        config = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(algorithm="sft"),
            generation=GenerationSettings(),
        )

        # Act
        result = solve(config, "max_model_len", hi=2048)

        # Assert
        assert result.config.training.max_model_len == result.value
        assert result.config.generation.max_model_len == result.value
        assert result.unchecked_over_budget == ()
        assert result.estimate.training.fits_with_buffer

    def test_training_limited_solve_stops_at_the_buffer(self, model):
        # Arrange: a card small enough that context search hits training memory.
        device = DeviceSpec(total_bytes=int(4.5 * GiB))
        config = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(algorithm="sft"),
            generation=GenerationSettings(),
        )

        # Act
        result = solve(config, "max_model_len", hi=32768)
        over = estimate_run(
            lockstep_max_model_len(
                result.config, result.value + kv_cache.KV_BLOCK_SIZE_DEFAULT
            )
        )

        # Assert
        assert result.value == 10944
        assert result.limited_by == "memory"
        assert result.estimate.training.fits_with_buffer
        assert over.training.fits
        assert not over.training.fits_with_buffer


class TestSolveRebuild:
    def test_micro_batch_resolved_per_candidate_reaches_the_single_row_top(self, model):
        # Arrange: the rebuild picks the largest micro-batch that fits, as a
        # manifest with an omitted micro-batch does.
        device = DeviceSpec(total_bytes=int(4.5 * GiB))
        base = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(
                algorithm="sft", micro_batch_size=8, trajectories_per_update=8
            ),
            generation=GenerationSettings(),
        )

        def at(length: int, micro: int) -> RunConfig:
            training = base.training.model_copy(update={"micro_batch_size": micro})
            return lockstep_max_model_len(
                base.model_copy(update={"training": training}), length
            )

        def rebuild(length: int) -> RunConfig:
            fitting = (
                micro
                for micro in (8, 4, 2)
                if estimate_run(at(length, micro)).training.fits_with_buffer
            )
            return at(length, next(fitting, 1))

        # Act
        held = solve(base, "max_model_len", hi=32768)
        rebuilt = solve(base, "max_model_len", hi=32768, rebuild=rebuild)
        over = estimate_run(at(rebuilt.value + kv_cache.KV_BLOCK_SIZE_DEFAULT, 1))

        # Assert
        assert held.value == 1360
        assert rebuilt.value == 10944
        assert rebuilt.limited_by == "memory"
        assert rebuilt.config.training.micro_batch_size == 1
        assert rebuilt.config.training.max_model_len == 10944
        assert not over.training.fits_with_buffer

    def test_a_training_miss_ends_the_scan(self, model):
        # Arrange: no context fits the trainer on a 1 GiB card.
        device = DeviceSpec(total_bytes=GiB)
        base = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            training=TrainingSettings(algorithm="sft"),
            generation=GenerationSettings(),
        )
        rebuilt = []

        def rebuild(length: int) -> RunConfig:
            rebuilt.append(length)
            return lockstep_max_model_len(base, length)

        # Act
        with pytest.raises(CannotSolve, match="no max_model_len value up to 4096"):
            solve(base, "max_model_len", hi=4096, rebuild=rebuild)

        # Assert
        assert rebuilt == [kv_cache.KV_BLOCK_SIZE_DEFAULT]

    def test_a_generation_miss_keeps_scanning(self, model, device):
        # Arrange: below 64 tokens the rebuild puts the engine on a card too
        # small to serve; the trainer fits throughout.
        small = DeviceSpec(total_bytes=GiB, name="small")
        base = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            generation=GenerationSettings(gpu_memory_utilization=0.3, max_num_seqs=4),
        )

        def rebuild(length: int) -> RunConfig:
            gen_device = small if length < 64 else device
            return base.model_copy(update={"gen_device": gen_device})

        # Act
        result = solve(base, "max_model_len", hi=128, rebuild=rebuild)

        # Assert
        assert result.value == 128
        assert result.limited_by == "bound"
        assert result.config.gen_device == device


class TestSolveWidestEngineWidth:
    def test_a_generation_miss_at_the_widest_engine_ends_the_scan(
        self, nine_b, nine_b_engine
    ):
        # Arrange: one 9B engine GPU of 20 GiB serves no context.
        base = engine_run(nine_b, 20, nine_b_engine)
        rebuilt = []

        def rebuild(length: int) -> RunConfig:
            rebuilt.append(length)
            return base

        # Act
        with pytest.raises(CannotSolve, match="no max_model_len value up to 128"):
            solve(
                base,
                "max_model_len",
                hi=128,
                phases=("generation",),
                rebuild=rebuild,
                widest_engine_width=1,
            )

        # Assert
        assert rebuilt == [kv_cache.KV_BLOCK_SIZE_DEFAULT]

    def test_a_miss_below_the_widest_engine_keeps_scanning(self, nine_b, nine_b_engine):
        # Arrange: from 64 tokens the rebuild serves on two 20 GiB GPUs.
        base = engine_run(nine_b, 20, nine_b_engine)

        def rebuild(length: int) -> RunConfig:
            width = 1 if length < 64 else 2
            return engine_run(
                nine_b,
                20,
                nine_b_engine.model_copy(update={"tensor_parallel_size": width}),
            )

        # Act
        result = solve(
            base,
            "max_model_len",
            hi=128,
            phases=("generation",),
            rebuild=rebuild,
            widest_engine_width=2,
        )

        # Assert
        assert result.value == 128
        assert result.limited_by == "bound"
        assert result.config.generation.tensor_parallel_size == 2

    def test_a_miss_the_other_sync_mode_clears_keeps_scanning(
        self, nine_b, nine_b_engine
    ):
        # Arrange: on 24 GiB only merged sync serves; the rebuild picks it from 64 tokens.
        base = engine_run(nine_b, 24, nine_b_engine)
        merged = engine_run(
            nine_b, 24, nine_b_engine.model_copy(update={"merged_lora_sync": True})
        )

        # Act
        result = solve(
            base,
            "max_model_len",
            hi=128,
            phases=("generation",),
            rebuild=lambda length: base if length < 64 else merged,
            widest_engine_width=1,
        )

        # Assert
        assert result.value == 128
        assert result.limited_by == "bound"
        assert result.config.generation.merged_lora_sync

    def test_a_miss_the_lean_engine_clears_keeps_scanning(self):
        # Arrange: on 23.5 GiB a multi-turn Gemma engine with n-gram drafting
        # and dual-stream LoRA misses in both sync modes; from 64 tokens the
        # rebuild drops those fast paths.
        hub_id = "google/gemma-4-E4B-it"
        gemma = ModelSpec(
            model_id=hub_id,
            arch=ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config),
        )
        engine = GenerationSettings(
            gpu_memory_utilization=0.9,
            max_num_seqs=16,
            multi_turn=True,
            max_lora_rank=64,
            lora_sync_scope="all-linear",
            lora_dual_stream=True,
        )
        base = engine_run(gemma, 23.5, engine)
        lean = engine_run(
            gemma,
            23.5,
            engine.model_copy(
                update={
                    "lora_dual_stream": False,
                    "speculative_drafting": False,
                    "max_num_batched_tokens": kv_cache.VLLM_ENGINE_MAX_NUM_BATCHED_TOKENS,
                    "merged_lora_sync": True,
                }
            ),
        )

        # Act
        result = solve(
            base,
            "max_model_len",
            hi=128,
            phases=("generation",),
            rebuild=lambda length: base if length < 64 else lean,
            widest_engine_width=1,
        )

        # Assert
        assert result.value == 128
        assert result.limited_by == "bound"
        assert result.config.generation.speculative_drafting is False
        assert result.config.generation.lora_dual_stream is False


class TestBareSolveOnAnInferenceConfig:
    def test_bare_solve_caps_on_the_dummy_training_bar(self, model):
        # Arrange
        device = DeviceSpec(total_bytes=int(5 * GiB))
        settings = GenerationSettings(gpu_memory_utilization=0.5, max_num_seqs=2)
        config = inference_run_config(model, device, settings)

        # Act
        bare = solve(config, "max_model_len", hi=32768)
        dedicated = solve(config, "max_model_len", hi=32768, phases=("generation",))

        # Assert
        assert bare.value == 14880
        assert bare.limited_by == "memory"
        assert bare.estimate.training.fits_with_buffer
        assert dedicated.value == 32768
        assert dedicated.limited_by == "bound"
        assert dedicated.value > bare.value
        over = estimate_run(
            lockstep_max_model_len(
                bare.config, bare.value + kv_cache.KV_BLOCK_SIZE_DEFAULT
            )
        )
        assert over.training.fits
        assert not over.training.fits_with_buffer
