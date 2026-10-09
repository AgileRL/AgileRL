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
        pool = 256 * 1024 * 1024
        settings = GenerationSettings(
            gpu_memory_utilization=0.9,
            max_num_seqs=8,
            kv_cache_memory_bytes=pool,
        )
        result = solve_inference(
            inference_run_config(model, device, settings), "max_model_len", hi=32768
        )
        per_token = formulas.kv_cache_bytes_per_token(model.arch, "bf16")
        expected = (int(pool // (8 * per_token)) // 16) * 16
        assert result.value == expected
        assert result.limited_by == "memory"
        assert generation_can_serve(result.estimate.generation)

        over = estimate_generation(
            model,
            device,
            settings.model_copy(update={"max_model_len": expected + 16}),
        )
        assert not generation_can_serve(over)

    def test_max_num_seqs_grows_until_the_pool_is_full(self, model, device):
        settings = GenerationSettings(
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            kv_cache_memory_bytes=256 * 1024 * 1024,
        )
        result = solve_inference(
            inference_run_config(model, device, settings), "max_num_seqs", hi=64
        )
        over = estimate_generation(
            model,
            device,
            settings.model_copy(update={"max_num_seqs": result.value + 1}),
        )
        assert result.value == 10
        assert result.limited_by == "memory"
        assert generation_can_serve(result.estimate.generation)
        assert not generation_can_serve(over)

    def test_higher_concurrency_shortens_max_model_len(self, model, device):
        def solved(seqs: int) -> int:
            settings = GenerationSettings(
                gpu_memory_utilization=0.9,
                max_num_seqs=seqs,
                kv_cache_memory_bytes=256 * 1024 * 1024,
            )
            return solve_inference(
                inference_run_config(model, device, settings),
                "max_model_len",
                hi=32768,
            ).value

        assert solved(4) == 5456
        assert solved(8) == 2720

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


class TestSolveSingleValley:
    """Generation fits() dips mid-range; the minimum failing must not end it."""

    def test_returns_the_run_top_when_the_minimum_does_not_fit(self, model):
        # Arrange
        device = DeviceSpec(total_bytes=8 * GiB)
        settings = GenerationSettings(
            gpu_memory_utilization=0.9,
            max_num_seqs=8,
            concurrent_requests=2,
        )
        config = inference_run_config(model, device, settings)

        # Act
        result = solve_inference(config, "max_model_len", hi=32768)

        # Assert
        assert result.value == 21120
        assert result.limited_by == "memory"

    def test_the_minimum_fails_while_mid_range_fits(self, model):
        # Arrange
        device = DeviceSpec(total_bytes=8 * GiB)

        def serves(context: int) -> bool:
            settings = GenerationSettings(
                gpu_memory_utilization=0.9,
                max_num_seqs=8,
                concurrent_requests=2,
                max_model_len=context,
            )
            bar = estimate_run(inference_run_config(model, device, settings))
            return generation_can_serve(bar.generation)

        # Act + Assert
        assert not serves(16)
        assert serves(4096)
        assert not serves(32768)


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
        assert bare.value == 9008
        assert bare.limited_by == "memory"
        assert bare.estimate.training.fits_with_buffer
        assert dedicated.value == 22256
        assert dedicated.value > bare.value
        over = estimate_run(
            lockstep_max_model_len(
                bare.config, bare.value + formulas.KV_BLOCK_SIZE_DEFAULT
            )
        )
        assert over.training.fits
        assert not over.training.fits_with_buffer
