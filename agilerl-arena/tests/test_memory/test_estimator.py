# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Estimator behaviour: breakdown structure, monotonicity, budget checks."""

import pytest

from agilerl.arena.memory import formulas
from agilerl.arena.memory.advice import advise
from agilerl.arena.memory.estimator import (
    estimate_generation,
    estimate_run,
    estimate_training,
    generation_can_serve,
)
from agilerl.arena.memory.specs import (
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
        "zero3_state",
        "overhead",
        "allocator_reserve",
    ]
    assert breakdown.total_bytes > 0
    assert breakdown.fits
    # LoRA-only: grads + optimizer state scale with adapter parameters, so
    # they land far below the ~12 bytes/param a full fine-tune would pay on
    # the whole model — the inversion of the usual "optimizer state is 3x
    # your weights" intuition.
    full_ft_equivalent = 12 * formulas.param_counts(model.arch).total
    assert component(breakdown, "grads_optimizer").bytes_ < full_ft_equivalent / 20
    assert (
        component(breakdown, "grads_optimizer").bytes_
        < component(breakdown, "base_weights").bytes_
    )
    # Chunked loss caps the tile pair at <= 512 MiB (the auto-tune budget);
    # the fused path (fused_logprobs.py) additionally hoists one fp32
    # (vocab, hidden) head copy for the loop, so the workspace is
    # tiles + head, never a slab.
    head_upcast = formulas.fused_head_upcast_bytes(model.arch)
    assert (
        component(breakdown, "logits_workspace").bytes_ <= 512 * 1024**2 + head_upcast
    )


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


def test_beta_zero_skips_the_reference_forward_but_keeps_the_adapter(model, device):
    # beta does not change what is *resident*: the reference adapter is
    # created at init either way. What it changes is whether the fused
    # no-grad pass carries a reference row, which is activation memory.
    with_kl = TrainingKnobs(beta=0.001, max_model_len=4096)
    without_kl = TrainingKnobs(beta=0.0, max_model_len=4096)

    assert with_kl.n_adapter_rows == 2
    assert without_kl.n_adapter_rows == 1
    assert with_kl.n_resident_adapters == without_kl.n_resident_adapters

    hot = estimate_training(model, device, with_kl)
    cold = estimate_training(model, device, without_kl)

    # The no-grad instant halves, since it drops one of its two fused rows.
    assert (
        component(cold, "activations").detail["nograd_peak"]
        < (component(hot, "activations").detail["nograd_peak"])
    )
    assert component(cold, "adapters").bytes_ == component(hot, "adapters").bytes_
    assert cold.total_bytes <= hot.total_bytes

    # But the bar only moves when the no-grad instant is the binding one.
    # Under gradient checkpointing the backward carries a saved hidden state
    # per layer and usually dominates, masking the beta=0 saving — worth
    # knowing before promising users that beta=0 frees memory.
    detail = component(hot, "activations").detail
    assert detail["backward_peak"] >= detail["nograd_peak"]
    # The instant totals carry the gradients that are resident at each, but the
    # bar reports those under grads_optimizer, so activations is the binding
    # instant net of them.
    gradients = component(hot, "grads_optimizer").detail["gradients"]
    assert component(hot, "activations").bytes_ == detail["backward_peak"] - gradients

    # And the saving is only real once the framework short-circuits that row.
    assert any("beta=0" in w for w in cold.warnings)


def test_ppo_carries_a_critic_adapter_and_value_head(model, device):
    # PPO fuses three rows (reference, actor, critic), keeps three adapters
    # resident, and trains two of them plus a Linear(hidden -> 1) value head.
    grpo = TrainingKnobs(algorithm="grpo")
    ppo = TrainingKnobs(algorithm="ppo")

    assert ppo.n_adapter_rows == 3
    assert ppo.n_resident_adapters == 3
    assert ppo.n_trained_adapters == 2
    assert grpo.n_trained_adapters == 1

    ppo_bd = estimate_training(model, device, ppo)
    grpo_bd = estimate_training(model, device, grpo)
    assert component(ppo_bd, "adapters").bytes_ > component(grpo_bd, "adapters").bytes_
    assert (
        component(ppo_bd, "grads_optimizer").bytes_
        > component(grpo_bd, "grads_optimizer").bytes_
    )
    # PPO's third fused row widens the no-grad instant, which can make that
    # instant rather than the backward the binding one. When it does, the
    # "activations" component reports what is live *then* — which excludes
    # the block recompute and so reads lower than GRPO's, even though PPO's
    # total is higher. Compare totals, not components, across algorithms.
    assert (
        component(ppo_bd, "activations").detail["nograd_peak"]
        > component(grpo_bd, "activations").detail["nograd_peak"]
    )
    assert ppo_bd.total_bytes > grpo_bd.total_bytes


def test_sft_has_no_reference_or_critic():
    knobs = TrainingKnobs(algorithm="sft")
    assert knobs.n_adapter_rows == 1
    assert knobs.n_resident_adapters == 1
    assert not knobs.uses_reference


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
    assert (
        component(with_graphs, "cuda_graphs").bytes_ == formulas.CUDA_GRAPH_POOL_BYTES
    )
    assert component(eager, "cuda_graphs").bytes_ == 0


def _with_extra_params(model, fraction):
    analytic = formulas.param_counts(model.arch).total
    return model.model_copy(update={"n_params": int(analytic / (1.0 - fraction))})


@pytest.mark.parametrize(
    "phase", [estimate_training, estimate_generation], ids=["training", "generation"]
)
def test_both_phases_warn_when_geometry_misses_the_checkpoint(model, device, phase):
    # Gemma 4's per-layer-embedding and tower blocks reach 3.7-5.0%; a stale
    # parse of the same config reached 68%, silently, which is what this
    # catches.
    knobs = TrainingKnobs() if phase is estimate_training else GenerationKnobs()
    assert not any(
        "not accounted for" in w for w in phase(model, device, knobs).warnings
    )
    gappy = _with_extra_params(model, 0.05)
    assert any("not accounted for" in w for w in phase(gappy, device, knobs).warnings)


def test_exact_checkpoint_count_moves_the_weights_component(model, device):
    baseline = component(
        estimate_training(model, device, TrainingKnobs()), "base_weights"
    )
    exact = component(
        estimate_training(_with_extra_params(model, 0.05), device, TrainingKnobs()),
        "base_weights",
    )
    assert exact.bytes_ > baseline.bytes_


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


def test_generation_can_serve_requires_the_pool_to_cover_demand(model, device):
    fits = estimate_generation(
        model,
        device,
        GenerationKnobs(gpu_memory_utilization=0.5, max_model_len=1024),
    )
    assert generation_can_serve(fits)

    starved = estimate_generation(
        model,
        device,
        GenerationKnobs(
            max_model_len=32768,
            max_num_seqs=64,
            kv_cache_memory_bytes=1 * GiB,
        ),
    )
    assert starved.fits
    assert not generation_can_serve(starved)


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


def test_eager_is_told_to_switch_attention_backend(device):
    # eager builds a rows x heads x S x S score matrix, quadratic in context
    # while every other term is linear, so switching is the only fix that
    # leaves batch, context and optimisation untouched.
    def run(impl, arch):
        return RunConfig(
            model=ModelSpec(model_id=impl, arch=arch),
            train_device=device,
            training=TrainingKnobs(
                max_model_len=8192,
                micro_batch_size_per_gpu=8,
                attn_implementation=impl,
            ),
        )

    suggestions = advise(run("eager", QWEN_05B))
    offered = [s for s in suggestions if "flex_attention" in s.action]
    assert offered, "no backend switch offered for eager"
    assert offered[0].saves_bytes == max(s.saves_bytes for s in suggestions)

    # SDPA is never offered it, windowed or not: measured within 156 MiB of
    # flex_attention on both Gemma 4 sizes at 4096 x 8.
    windowed = QWEN_05B.model_copy(update={"sliding_window": 512})
    assert not [
        s for s in advise(run("sdpa", windowed)) if "flex_attention" in s.action
    ]
    assert not [
        s for s in advise(run("sdpa", QWEN_05B)) if "flex_attention" in s.action
    ]


def test_estimate_serializes_with_bytes_alias(model, device):
    estimate = estimate_run(RunConfig(model=model, train_device=device))
    payload = estimate.model_dump(mode="json", by_alias=True)
    first = payload["training"]["components"][0]
    assert "bytes" in first
    assert "bytes_" not in first


def test_allocator_reserve_marks_up_torch_terms_only(model, device):
    # Training runs through the torch caching allocator, so the device is
    # charged reserved rather than allocated bytes. Generation runs inside
    # vLLM's CuMem pool, which is reserved up front at gpu_memory_utilization
    # -- there is no incremental slack to charge, and the leave-one-model-out
    # optimum there is 0.0% for all eight profiled models.
    breakdown = estimate_training(model, device, TrainingKnobs(), colocated=True)
    by_key = {c.key: c.bytes_ for c in breakdown.components}
    assert "allocator_reserve" in by_key

    torch_side = sum(
        v
        for k, v in by_key.items()
        if k not in ("vllm_residual", "overhead", "allocator_reserve")
    )
    assert by_key["allocator_reserve"] == pytest.approx(
        torch_side * formulas.ALLOCATOR_RESERVE_FRACTION, abs=1
    )

    generation = estimate_generation(model, device, GenerationKnobs(), colocated=True)
    assert "allocator_reserve" not in {c.key for c in generation.components}


class TestDistributedTerms:
    """DP / ZeRO sharding: logical terms, no measured ground truth yet."""

    def test_zero2_shards_optimizer_but_not_weights(self, model, device):
        single = estimate_training(model, device, TrainingKnobs())
        sharded = estimate_training(
            model,
            device,
            TrainingKnobs(distributed="deepspeed", n_training_gpus=4, zero_stage=2),
        )
        assert component(sharded, "base_weights").bytes_ == (
            component(single, "base_weights").bytes_
        )
        # The framework's resolved DeepSpeed config pins offload_optimizer to
        # CPU, so the AdamW state costs the device nothing under DeepSpeed —
        # measured on the Nemotron ZeRO-3 sweep (0 CUDA bytes, host-resident
        # moments). Gradients still live on-device, sharded.
        opt = component(sharded, "grads_optimizer").detail["adamw_state"]
        assert opt == 0
        assert component(single, "grads_optimizer").detail["adamw_state"] > 0

    def test_zero3_divides_the_frozen_base(self, model, device):
        single = estimate_training(model, device, TrainingKnobs())
        sharded = estimate_training(
            model,
            device,
            TrainingKnobs(distributed="deepspeed", n_training_gpus=8, zero_stage=3),
        )
        base_single = component(single, "base_weights").bytes_
        base_sharded = component(sharded, "base_weights").bytes_
        counts = formulas.param_counts(model.arch, model.n_params)
        gather = formulas.zero3_gather_bytes(counts, model.arch, "bf16")
        assert base_sharded == pytest.approx(base_single / 8 + gather, rel=0.01)
        assert any("ZeRO-3" in w for w in sharded.warnings)

    def test_dp_shards_the_update_rows(self, device, model):
        knobs = TrainingKnobs(
            distributed="deepspeed",
            n_training_gpus=4,
            trajectories_per_update=128,
            micro_batch_size_per_gpu=64,
        )
        # 128 rows over 4 learner shards = 32 per GPU, capping the micro-batch.
        assert knobs.trajectories == 32
        assert knobs.grad_rows == 32

    def test_without_deepspeed_nothing_shards(self, model, device):
        alone = TrainingKnobs(n_training_gpus=4)
        assert alone.dp_world_size == 1


class TestRayOverhead:
    def test_orchestrated_charges_the_worker_overhead_on_both_phases(
        self, model, device
    ):
        plain_t = estimate_training(model, device, TrainingKnobs())
        ray_t = estimate_training(model, device, TrainingKnobs(), orchestrated=True)
        assert (
            component(ray_t, "overhead").bytes_ - component(plain_t, "overhead").bytes_
        ) == formulas.RAY_ACTOR_OVERHEAD_BYTES
        plain_g = estimate_generation(model, device, GenerationKnobs())
        ray_g = estimate_generation(model, device, GenerationKnobs(), orchestrated=True)
        assert (
            component(ray_g, "overhead").bytes_ - component(plain_g, "overhead").bytes_
        ) == formulas.RAY_ACTOR_OVERHEAD_BYTES


class TestAlgorithmTerms:
    """PPO/SFT/DPO structure, modelled from their code paths (unmeasured)."""

    def test_dpo_holds_both_preference_graphs(self, model, device):
        grpo = TrainingKnobs(algorithm="grpo", micro_batch_size_per_gpu=4)
        dpo = TrainingKnobs(algorithm="dpo", micro_batch_size_per_gpu=4)
        assert dpo.grad_graph_rows == 2 * dpo.grad_rows
        assert grpo.grad_graph_rows == grpo.grad_rows
        # Chosen + rejected graphs live together: the backward instant carries
        # twice the saved state at the same micro-batch.
        g = estimate_training(model, device, grpo)
        d = estimate_training(model, device, dpo)
        assert (
            component(d, "activations").detail["backward_peak"]
            > component(g, "activations").detail["backward_peak"]
        )

    def test_dpo_reference_survives_beta_zero_and_does_not_fuse(self):
        knobs = TrainingKnobs(algorithm="dpo", beta=0.0)
        assert knobs.uses_reference
        assert knobs.n_adapter_rows == 1  # sequential passes, not fused rows

    def test_sft_has_no_nograd_instant_and_no_engine(self, model, device):
        knobs = TrainingKnobs(algorithm="sft")
        assert not knobs.has_nograd_pass
        assert not knobs.uses_generation_engine
        breakdown = estimate_training(model, device, knobs, colocated=True)
        assert component(breakdown, "activations").detail["nograd_peak"] == 0

    def test_engineless_algorithms_get_an_empty_generation_bar(self, model, device):
        config = RunConfig(
            model=model,
            train_device=device,
            training=TrainingKnobs(algorithm="sft"),
        )
        estimate = estimate_run(config)
        assert estimate.generation.components == ()
        assert estimate.generation.fits
        # No engine also means no sleeping-engine residual in the training bar.
        assert component(estimate.training, "vllm_residual").bytes_ == 0

    def test_unmeasured_algorithms_say_so(self, model, device):
        # PPO's one training sweep never exercised multi-row updates (the
        # algorithm samples one completion per prompt and the rig sent one
        # prompt), so its training bar still says so. SFT and DPO are
        # measured — 21-point A100 sweeps each, promoted into the corpus —
        # and must NOT claim otherwise.
        breakdown = estimate_training(model, device, TrainingKnobs(algorithm="ppo"))
        assert any("no measured ground truth" in w for w in breakdown.warnings)
        for algorithm in ("sft", "dpo"):
            breakdown = estimate_training(
                model, device, TrainingKnobs(algorithm=algorithm)
            )
            assert not any(
                "no measured ground truth" in w for w in breakdown.warnings
            ), algorithm


class TestContextExtrapolation:
    """Every stored measurement stops at 4096 tokens; past that the bar says so."""

    def test_within_the_measured_ceiling_is_silent(self, model, device):
        for phase in (
            estimate_training(model, device, TrainingKnobs(max_model_len=4096)),
            estimate_generation(model, device, GenerationKnobs(max_model_len=4096)),
        ):
            assert not any("extrapolation" in w for w in phase.warnings)

    def test_beyond_it_both_phases_warn(self, model, device):
        for phase in (
            estimate_training(model, device, TrainingKnobs(max_model_len=32768)),
            estimate_generation(model, device, GenerationKnobs(max_model_len=32768)),
        ):
            assert any("extrapolation" in w for w in phase.warnings), phase.phase
