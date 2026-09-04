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
    CannotSolve,
    architectural_context_limit,
    inference_run_config,
    solve,
    solve_inference,
)
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GenerationKnobs,
    GiB,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingKnobs,
)
from tests.test_memory.test_formulas import QWEN_05B

ASSETS = Path(__file__).parent / "assets"
TINY_CONFIG = json.loads((ASSETS / "tiny_llm" / "config.json").read_text())


@pytest.fixture
def model():
    return ModelSpec(model_id="Qwen/Qwen2.5-0.5B-Instruct", arch=QWEN_05B)


@pytest.fixture
def device():
    return DeviceSpec(total_bytes=24 * GiB, name="test-24g")


def test_architectural_context_limit_reads_max_position():
    assert architectural_context_limit(TINY_CONFIG) == 32768
    assert (
        architectural_context_limit({"text_config": {"max_position_embeddings": 8192}})
        == 8192
    )


def test_solve_max_model_len_is_the_last_value_that_serves(model, device):
    pool = 256 * 1024 * 1024
    knobs = GenerationKnobs(
        gpu_memory_utilization=0.9,
        max_num_seqs=8,
        kv_cache_memory_bytes=pool,
    )
    result = solve_inference(
        inference_run_config(model, device, knobs), "max_model_len", hi=32768
    )
    per_token = formulas.kv_cache_bytes_per_token(model.arch, "auto", "bf16")
    expected = (int(pool // (8 * per_token)) // 16) * 16
    assert result.value == expected
    assert result.limited_by == "memory"
    assert generation_can_serve(result.estimate.generation)

    over = estimate_generation(
        model,
        device,
        knobs.model_copy(update={"max_model_len": expected + 16}),
    )
    assert not generation_can_serve(over)


def test_higher_concurrency_shortens_max_model_len(model, device):
    def solved(seqs: int) -> int:
        knobs = GenerationKnobs(
            gpu_memory_utilization=0.9,
            max_num_seqs=seqs,
            kv_cache_memory_bytes=256 * 1024 * 1024,
        )
        return solve_inference(
            inference_run_config(model, device, knobs), "max_model_len", hi=32768
        ).value

    assert solved(8) <= solved(4)


def test_tiny_model_on_l4_hits_the_rope_cap(device):
    tiny = ModelSpec(model_id="tiny", arch=ModelArch.from_hf_config(TINY_CONFIG))
    result = solve_inference(
        inference_run_config(tiny, device),
        "max_model_len",
        hi=architectural_context_limit(TINY_CONFIG),
    )
    assert result.value == 32768
    assert result.limited_by == "bound"


def test_cannot_solve_when_the_card_cannot_hold_the_weights(device):
    huge = ModelSpec(
        model_id="huge",
        arch=QWEN_05B.model_copy(update={"n_layers": 400, "hidden_size": 8192}),
    )
    with pytest.raises(CannotSolve, match="already does not fit"):
        solve_inference(
            inference_run_config(
                huge, DeviceSpec(total_bytes=1 * GiB), GenerationKnobs()
            ),
            "max_model_len",
            hi=1024,
        )


def test_solve_max_num_seqs_grows_until_the_pool_is_full(model, device):
    knobs = GenerationKnobs(
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        kv_cache_memory_bytes=256 * 1024 * 1024,
    )
    result = solve_inference(
        inference_run_config(model, device, knobs), "max_num_seqs", hi=64
    )
    assert 1 <= result.value <= 64
    assert generation_can_serve(result.estimate.generation)
    over = estimate_generation(
        model,
        device,
        knobs.model_copy(update={"max_num_seqs": result.value + 1}),
    )
    if result.value < 64:
        assert not generation_can_serve(over)


def test_solve_micro_batch_size_is_the_last_value_that_fits(model, device):
    config = RunConfig(
        model=model,
        train_device=device,
        training=TrainingKnobs(max_model_len=2048, micro_batch_size_per_gpu=1),
        generation=GenerationKnobs(max_model_len=2048),
    )
    result = solve(config, "micro_batch_size_per_gpu", hi=256)
    assert result.value >= 1
    assert result.config.training.micro_batch_size_per_gpu == result.value
    over = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"micro_batch_size_per_gpu": result.value + 1}
            )
        }
    )
    if result.limited_by == "memory":
        assert not estimate_run(over).training.fits


def test_unknown_knob_is_a_value_error(model, device):
    with pytest.raises(ValueError, match="Unknown knob"):
        solve(inference_run_config(model, device), "learning_rate")


def test_solve_max_model_len_keeps_training_and_generation_in_lockstep(model, device):
    config = RunConfig(
        model=model,
        train_device=device,
        training=TrainingKnobs(max_model_len=512, micro_batch_size_per_gpu=1),
        generation=GenerationKnobs(
            gpu_memory_utilization=0.3,
            max_model_len=512,
            max_num_seqs=4,
        ),
    )
    result = solve(config, "max_model_len", hi=2048)
    assert result.config.training.max_model_len == result.value
    assert result.config.generation.max_model_len == result.value
    assert result.value >= 512
