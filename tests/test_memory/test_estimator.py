"""Estimator behaviour: breakdown structure, monotonicity, budget checks."""

import pytest

from agilerl.memory.advice import advise
from agilerl.memory.estimator import (
    estimate_generation,
    estimate_run,
    estimate_training,
)
from agilerl.memory.specs import (
    DeviceSpec,
    GenerationKnobs,
    GiB,
    ModelSpec,
    RunConfig,
    TrainingKnobs,
    WeightVariant,
)
from tests.test_memory.test_formulas import QWEN_05B


@pytest.fixture
def model():
    return ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        arch=QWEN_05B,
        variants=(WeightVariant(), WeightVariant(name="nf4", quantization="nf4")),
    )


@pytest.fixture
def device():
    return DeviceSpec(total_bytes=24 * GiB, name="test-24g")


def component(breakdown, key):
    return next(c for c in breakdown.components if c.key == key)


def test_training_breakdown_structure(model, device):
    breakdown = estimate_training(model, device, TrainingKnobs())
    keys = [c.key for c in breakdown.components]
    assert keys == [
        "base_weights",
        "adapters",
        "grads_optimizer",
        "activations",
        "logits_workspace",
        "vllm_residual",
        "overhead",
    ]
    assert breakdown.total_bytes > 0
    assert breakdown.fits
    # LoRA-only: grads+optimizer are far smaller than the frozen base.
    assert (
        component(breakdown, "grads_optimizer").bytes_
        < component(breakdown, "base_weights").bytes_ / 10
    )
    # Chunked loss caps the logit workspace at 2 tiles <= 512 MiB.
    assert component(breakdown, "logits_workspace").bytes_ <= 512 * 1024**2


def test_training_monotonic_in_seq_len_and_batch(model, device):
    base = estimate_training(
        model, device, TrainingKnobs(max_model_len=1024)
    ).total_bytes
    longer = estimate_training(
        model, device, TrainingKnobs(max_model_len=4096)
    ).total_bytes
    assert longer > base

    bigger_batch = estimate_training(
        model, device, TrainingKnobs(micro_batch_size_per_gpu=32)
    ).total_bytes
    default = estimate_training(
        model, device, TrainingKnobs(micro_batch_size_per_gpu=4)
    ).total_bytes
    assert bigger_batch > default


def test_training_checkpointing_and_quantization_levers(model, device):
    knobs = TrainingKnobs(max_model_len=2048, micro_batch_size_per_gpu=8)
    checkpointed = estimate_training(model, device, knobs)
    unchunked = estimate_training(
        model,
        device,
        knobs.model_copy(update={"gradient_checkpointing": False}),
    )
    assert unchunked.total_bytes > checkpointed.total_bytes
    assert any("gradient_checkpointing" in w for w in unchunked.warnings)

    nf4 = estimate_training(
        model,
        device,
        knobs.model_copy(update={"quantization": "nf4"}),
        trainer_variant="nf4",
    )
    assert (
        component(nf4, "base_weights").bytes_
        < component(checkpointed, "base_weights").bytes_
    )


def test_beta_is_memory_neutral(model, device):
    # beta is not a TrainingKnobs field at all: the reference forward and
    # adapter exist regardless, so there is nothing for it to change.
    assert "beta" not in TrainingKnobs.model_fields


def test_generation_kv_pool_is_budget_remainder(model, device):
    knobs = GenerationKnobs(gpu_memory_utilization=0.5, max_model_len=1024)
    breakdown = estimate_generation(model, device, knobs)
    budget = int(0.5 * device.total_bytes)
    engine_side = sum(
        component(breakdown, key).bytes_
        for key in (
            "weights",
            "kv_cache",
            "activation_peak",
            "cuda_graphs",
            "lora_slots",
        )
    )
    assert engine_side == pytest.approx(budget, rel=0.01)

    pinned = estimate_generation(
        model,
        device,
        knobs.model_copy(update={"kv_cache_memory_bytes": 123456789}),
    )
    assert component(pinned, "kv_cache").bytes_ == 123456789


def test_generation_enforce_eager_drops_graph_pool(model, device):
    with_graphs = estimate_generation(model, device, GenerationKnobs())
    eager = estimate_generation(model, device, GenerationKnobs(enforce_eager=True))
    assert component(with_graphs, "cuda_graphs").bytes_ == 2 * GiB
    assert component(eager, "cuda_graphs").bytes_ == 0


def test_generation_warns_when_budget_too_small_for_weights(device):
    big = ModelSpec(
        model_id="big",
        arch=QWEN_05B.model_copy(update={"n_layers": 200, "hidden_size": 4096}),
    )
    breakdown = estimate_generation(
        big, device, GenerationKnobs(gpu_memory_utilization=0.2)
    )
    assert component(breakdown, "kv_cache").bytes_ == 0
    assert any("fail at init" in w for w in breakdown.warnings)


def test_generation_kv_demand_warning(model, device):
    breakdown = estimate_generation(
        model,
        device,
        GenerationKnobs(
            max_model_len=32768,
            max_num_seqs=64,
            kv_cache_memory_bytes=1 * GiB,
        ),
    )
    # 64 sequences at 32k tokens want ~25 GiB of KV against a 1 GiB pool.
    assert any("preempt" in w for w in breakdown.warnings)


def test_colocated_run_includes_trainer_residual(model, device):
    config = RunConfig(model=model, train_device=device)
    estimate = estimate_run(config)
    assert config.colocated
    residual = component(estimate.generation, "trainer_residual")
    assert residual.bytes_ > 0

    split = RunConfig(
        model=model,
        train_device=device,
        gen_device=DeviceSpec(total_bytes=16 * GiB),
    )
    split_estimate = estimate_run(split)
    assert component(split_estimate.generation, "trainer_residual").bytes_ == 0


def test_sleeping_engine_residual_is_small_and_utilization_independent(model, device):
    # Measured on vLLM 0.23: sleep level 1 hands the engine's weights and KV
    # pool back to the device, so the trainer does NOT pay the engine's
    # gpu_memory_utilization budget. The residual is a small constant and
    # must not scale with utilization, or a profile would be pinned to the
    # single utilization it was measured at.
    def residual_at(utilization: float) -> int:
        estimate = estimate_run(
            RunConfig(
                model=model,
                train_device=device,
                generation=GenerationKnobs(gpu_memory_utilization=utilization),
            )
        )
        return component(estimate.training, "vllm_residual").bytes_

    assert residual_at(0.3) == residual_at(0.8)
    assert residual_at(0.45) < GiB
    assert residual_at(0.45) < int(0.1 * device.total_bytes)

    # A dedicated training device carries no engine residual at all.
    split = estimate_run(
        RunConfig(
            model=model,
            train_device=device,
            gen_device=DeviceSpec(total_bytes=16 * GiB),
            generation=GenerationKnobs(gpu_memory_utilization=0.45),
        )
    )
    assert component(split.training, "vllm_residual").bytes_ == 0


def test_over_budget_advice_is_ranked_and_actionable(model):
    tiny = DeviceSpec(total_bytes=2 * GiB)
    config = RunConfig(
        model=model,
        train_device=tiny,
        training=TrainingKnobs(max_model_len=4096, micro_batch_size_per_gpu=16),
    )
    estimate = estimate_run(config)
    assert not estimate.training.fits

    suggestions = advise(config)
    assert suggestions
    savings = [s.saves_bytes for s in suggestions]
    assert savings == sorted(savings, reverse=True)
    assert all(s.saves_bytes > 0 for s in suggestions)


def test_estimate_serializes_with_bytes_alias(model, device):
    estimate = estimate_run(RunConfig(model=model, train_device=device))
    payload = estimate.model_dump(mode="json", by_alias=True)
    first = payload["training"]["components"][0]
    assert "bytes" in first
    assert "bytes_" not in first
