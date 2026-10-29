# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Setting inversion: largest value that still fits."""

import json
from pathlib import Path

import pytest

from agilerl.arena.memory import formulas
from agilerl.arena.memory.estimator import (
    estimate_generation,
    estimate_run,
    generation_can_serve,
)
from agilerl.arena.memory.solver import (
    DEFAULT_CONTEXT_LIMIT,
    CannotSolve,
    FieldSpec,
    _apply,
    architectural_context_limit,
    inference_run_config,
    solve,
    solve_inference,
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
from tests.test_memory.test_formulas import QWEN_05B

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


class TestSolveInference:
    def test_max_model_len_is_the_last_value_that_serves(self, model, device):
        settings = GenerationSettings(gpu_memory_utilization=0.07, max_num_seqs=8)
        result = solve_inference(
            inference_run_config(model, device, settings), "max_model_len", hi=32768
        )
        assert result.value % 16 == 0
        assert result.limited_by == "memory"
        assert generation_can_serve(result.estimate.generation)

        over = estimate_generation(
            model,
            device,
            settings.model_copy(update={"max_model_len": result.value + 16}),
        )
        assert not generation_can_serve(over)

    def test_max_num_seqs_grows_until_the_pool_is_full(self, model, device):
        settings = GenerationSettings(gpu_memory_utilization=0.07, max_model_len=2048)
        result = solve_inference(
            inference_run_config(model, device, settings), "max_num_seqs", hi=64
        )
        over = estimate_generation(
            model,
            device,
            settings.model_copy(update={"max_num_seqs": result.value + 1}),
        )
        assert result.value == 15
        assert result.limited_by == "memory"
        assert generation_can_serve(result.estimate.generation)
        assert not generation_can_serve(over)

    def test_higher_concurrency_shortens_max_model_len(self, model, device):
        def solved(seqs: int) -> int:
            settings = GenerationSettings(
                gpu_memory_utilization=0.07, max_num_seqs=seqs
            )
            return solve_inference(
                inference_run_config(model, device, settings),
                "max_model_len",
                hi=32768,
            ).value

        assert solved(4) == 8336
        assert solved(8) == 4112

    def test_tiny_model_on_l4_hits_the_rope_cap(self, device):
        tiny = ModelSpec(model_id="tiny", arch=ModelArch.from_hf_config(TINY_CONFIG))
        result = solve_inference(
            inference_run_config(tiny, device),
            "max_model_len",
            hi=architectural_context_limit(TINY_CONFIG),
        )
        assert result.value == 32768
        assert result.limited_by == "bound"

    def test_cannot_solve_when_the_card_cannot_hold_the_weights(self, device):
        huge = ModelSpec(
            model_id="huge",
            arch=QWEN_05B.model_copy(update={"n_layers": 400, "hidden_size": 8192}),
        )
        with pytest.raises(CannotSolve, match="no max_model_len value up to"):
            solve_inference(
                inference_run_config(
                    huge, DeviceSpec(total_bytes=1 * GiB), GenerationSettings()
                ),
                "max_model_len",
                hi=1024,
            )

    def test_upper_bound_below_the_minimum_is_rejected(self, model, device):
        config = inference_run_config(model, device)

        with pytest.raises(ValueError, match="below the minimum"):
            solve_inference(config, "max_model_len", hi=1)

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
            solve_inference(config, "learning_rate")


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
            solve_inference(config, "max_model_len", hi=32768)

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
                result.config, result.value + formulas.KV_BLOCK_SIZE_DEFAULT
            )
        )

        # Assert
        assert result.value == 432
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
        over = estimate_run(at(rebuilt.value + formulas.KV_BLOCK_SIZE_DEFAULT, 1))

        # Assert
        assert held.value == 48
        assert rebuilt.value == 432
        assert rebuilt.limited_by == "memory"
        assert rebuilt.config.training.micro_batch_size == 1
        assert rebuilt.config.training.max_model_len == 432
        assert not over.training.fits_with_buffer


class TestBareSolveOnAnInferenceConfig:
    def test_bare_solve_caps_on_the_dummy_training_bar(self, model):
        # Arrange
        device = DeviceSpec(total_bytes=int(5 * GiB))
        settings = GenerationSettings(gpu_memory_utilization=0.5, max_num_seqs=2)
        config = inference_run_config(model, device, settings)

        # Act
        bare = solve(config, "max_model_len", hi=32768)
        dedicated = solve_inference(config, "max_model_len", hi=32768)

        # Assert
        assert bare.value == 3456
        assert bare.limited_by == "memory"
        assert bare.estimate.training.fits_with_buffer
        assert dedicated.value == 32768
        assert dedicated.limited_by == "bound"
        assert dedicated.value > bare.value
        over = estimate_run(
            lockstep_max_model_len(
                bare.config, bare.value + formulas.KV_BLOCK_SIZE_DEFAULT
            )
        )
        assert over.training.fits
        assert not over.training.fits_with_buffer
