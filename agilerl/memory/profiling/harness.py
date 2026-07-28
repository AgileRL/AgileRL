"""Measure one profiling point: real GRPO generation + training peaks.

Each point instantiates a colocated GRPO agent with the point's knobs, runs
one rollout and one learn, and records per-phase peaks:

- generation: NVML device peak (the only signal that sees vLLM's CuMem
  allocations) around ``get_action``;
- training: NVML device peak plus ``torch.cuda.max_memory_allocated`` /
  ``max_memory_reserved`` around ``learn``.

Requires a CUDA device with the ``llm`` extra installed; import stays lazy so
the sweep planner and fitter remain importable everywhere.

Runs as a module (``python -m agilerl.memory.profiling.harness``) so the sweep
can spawn one fresh process per point: vLLM's CuMem allocator (which backs
sleep mode) is process-global and permits only one engine per process, so a
multi-point sweep must isolate each point in its own process.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from agilerl.memory.calibration import MeasuredPoint
from agilerl.memory.specs import GenerationKnobs, TrainingKnobs


@dataclass(frozen=True)
class SweepPoint:
    """One knob combination of the profiling sweep."""

    seq_len: int
    micro_batch: int
    group_size: int
    lora_rank: int
    quantization: str = "none"
    #: Prompts sent per rollout; the update carries ``n_prompts x group_size``
    #: completion rows.
    n_prompts: int = 1
    #: Restricts which submodules LoRA wraps. Required for multimodal
    #: checkpoints: ``all-linear`` otherwise wraps the vision/audio towers
    #: too, and vLLM refuses the resulting adapter because those module names
    #: are not in its supported LoRA target set.
    lora_target_scope: str | None = None
    #: Attention backend forwarded to the model config. ``auto`` leaves the
    #: framework's own resolution in place (FlashAttention-2 if the package
    #: is importable, else SDPA), which is what a stock install gets.
    attn_implementation: str = "auto"
    #: Algorithm under test. Only GRPO and PPO are measured: they bracket the
    #: adapter structure (2 fused rows vs 3, one trained adapter vs two), and
    #: the rest reduce to one or the other.
    algorithm: str = "grpo"
    #: Explicit LoRA targets, comma-separated. ``all-linear`` is PEFT's
    #: shorthand but is not applied for every architecture — on OLMoE it was
    #: iterated character-wise — and wrapping every expert projection on an
    #: MoE is pathological regardless, so MoE sweeps name the attention
    #: projections instead.
    lora_target_modules: str = "all-linear"
    #: Engine budget fraction. A real sweep axis, not a fixed setting: the
    #: colocated training floor is driven by it, so holding out points at
    #: other utilizations is what validates that the floor is modelled
    #: analytically rather than absorbed into the fitted intercept.
    gpu_memory_utilization: float = 0.45

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "seq_len": self.seq_len,
            "micro_batch": self.micro_batch,
            "group_size": self.group_size,
            "lora_rank": self.lora_rank,
            "quantization": self.quantization,
            "n_prompts": self.n_prompts,
            "lora_target_scope": self.lora_target_scope or "",
            "attn_implementation": self.attn_implementation,
            "algorithm": self.algorithm,
            "lora_target_modules": self.lora_target_modules,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }

    @classmethod
    def from_dict(cls, knobs: dict[str, float | int | str | bool]) -> SweepPoint:
        """Rebuild a point from a stored :class:`MeasuredPoint`'s knobs, so
        fixtures can be re-fitted offline without re-measuring.
        """
        return cls(
            seq_len=int(knobs["seq_len"]),
            micro_batch=int(knobs["micro_batch"]),
            group_size=int(knobs["group_size"]),
            lora_rank=int(knobs["lora_rank"]),
            quantization=str(knobs.get("quantization", "none")),
            n_prompts=int(knobs.get("n_prompts", 1)),
            lora_target_scope=str(knobs.get("lora_target_scope") or "") or None,
            attn_implementation=str(knobs.get("attn_implementation") or "auto"),
            algorithm=str(knobs.get("algorithm") or "grpo"),
            lora_target_modules=str(knobs.get("lora_target_modules") or "all-linear"),
            gpu_memory_utilization=float(knobs.get("gpu_memory_utilization", 0.45)),
        )

    def with_utilization(self, gpu_memory_utilization: float) -> SweepPoint:
        return replace(self, gpu_memory_utilization=gpu_memory_utilization)

    @property
    def scope(self) -> str:
        """Estimator-side LoRA scope implied by the targeted modules.

        The distinction matters enormously on an MoE: ``all-linear`` sizes
        adapters across every expert's projections, which on a 64-expert
        model over-predicts by ~10 GiB at rank 64 against an attention-only
        adapter.
        """
        if self.lora_target_modules == "all-linear":
            return "all-linear"
        mlp_markers = ("gate", "up_proj", "down_proj", "expert", "fc")
        targeted = self.lora_target_modules.lower()
        return (
            "all-linear"
            if any(marker in targeted for marker in mlp_markers)
            else "attention-only"
        )

    def training_knobs(self) -> TrainingKnobs:
        return TrainingKnobs(
            micro_batch_size_per_gpu=self.micro_batch,
            batch_size=self.micro_batch,
            group_size=self.group_size,
            # The harness sends one prompt, so the update carries exactly
            # ``group_size`` completion rows.
            trajectories_per_update=self.group_size * self.n_prompts,
            max_model_len=self.seq_len,
            lora_rank=self.lora_rank,
            lora_target_scope=self.scope,  # type: ignore[arg-type]
            quantization=self.quantization,  # type: ignore[arg-type]
            attn_implementation=self.attn_implementation,  # type: ignore[arg-type]
            algorithm=self.algorithm,  # type: ignore[arg-type]
        )

    @property
    def prompt_len(self) -> int:
        """Prompt length used for this point, leaving the rest of the context
        budget for completions.
        """
        return max(self.seq_len // 4, 8)

    def generation_knobs(self) -> GenerationKnobs:
        return GenerationKnobs(
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.group_size,
            max_model_len=self.seq_len,
            max_prompt_len=self.prompt_len,
            max_lora_rank=self.lora_rank,
            concurrent_requests=self.group_size * self.n_prompts,
        )


def measure_point(
    model_name: str,
    point: SweepPoint,
    device_index: int = 0,
    prompt_len: int | None = None,
    snapshot_path: str | None = None,
) -> tuple[MeasuredPoint | None, MeasuredPoint]:
    """Run one sweep point and return (generation, training) measurements.

    The prompt length defaults to a quarter of ``seq_len`` so completions can
    exercise the remaining context budget.

    ``snapshot_path`` additionally records a torch allocator history over the
    training phase and dumps it for https://pytorch.org/memory_viz. The sweep
    validates predicted *totals*; a snapshot is how the per-component split
    gets checked, because it carries a stack trace per allocation. It cannot
    replace the NVML measurement — vLLM allocates through CuMem, which the
    torch allocator never sees — and it is far too heavy for every sweep
    point, so it is opt-in and used on single configurations.
    """
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer

    from agilerl.memory.profiling.nvml import NvmlPeakSampler
    from agilerl.utils.algo_utils import VLLMConfig

    if point.algorithm == "ppo":
        from agilerl.algorithms.ppo_llm import PPO as Algorithm
    elif point.algorithm == "sft":
        from agilerl.algorithms.sft import SFT as Algorithm
    elif point.algorithm == "dpo":
        from agilerl.algorithms.dpo import DPO as Algorithm
    else:
        from agilerl.algorithms.grpo import GRPO as Algorithm

    if snapshot_path:
        # Record from before the model exists: allocations predating the
        # recording carry no stack, and the weights are allocated during
        # construction. Keep it cheap — python-only frames and a bounded
        # entry count. The default (all C++ frames, unbounded) costs many GiB
        # of host RAM across vLLM init and took a 23 GiB box off the network.
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=100_000
        )

    prompt_len = prompt_len or point.prompt_len
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = None
    if point.quantization == "nf4":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    targets = (
        "all-linear"
        if point.lora_target_modules == "all-linear"
        else [m.strip() for m in point.lora_target_modules.split(",") if m.strip()]
    )
    # SFT and DPO train from a fixed dataset: they hard-set use_vllm=False and
    # so have no generation phase to measure at all.
    rollout_based = point.algorithm in ("grpo", "ppo")
    algorithm_kwargs: dict[str, object] = {}
    if point.algorithm == "grpo":
        # Only GRPO groups completions; the others batch trajectories.
        algorithm_kwargs["group_size"] = point.group_size
    if rollout_based:
        algorithm_kwargs["use_vllm"] = True
        algorithm_kwargs["vllm_config"] = VLLMConfig(
            gpu_memory_utilization=point.gpu_memory_utilization,
            max_num_seqs=point.group_size,
            max_lora_rank=point.lora_rank,
            sleep_mode=True,
        )
        algorithm_kwargs["max_output_tokens"] = point.seq_len - prompt_len

    agent = Algorithm(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        pad_token=tokenizer.pad_token or tokenizer.eos_token,
        model_name=model_name,
        model_config=(
            {"attn_implementation": point.attn_implementation}
            if point.attn_implementation != "auto"
            else None
        ),
        micro_batch_size_per_gpu=point.micro_batch,
        batch_size=point.micro_batch,
        max_model_len=point.seq_len,
        lora_config=LoraConfig(
            r=point.lora_rank,
            lora_alpha=2 * point.lora_rank,
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
        lora_target_scope=point.lora_target_scope,
        quantization_config=quantization_config,
        **algorithm_kwargs,
    )

    # Fixed-content prompts of controlled token length; content is irrelevant
    # to memory, shape is everything.
    prompt_ids = torch.randint(
        low=10, high=tokenizer.vocab_size - 10, size=(1, prompt_len)
    )
    prompts = [
        {
            "input_ids": prompt_ids.clone(),
            "attention_mask": torch.ones_like(prompt_ids),
            "text": tokenizer.decode(prompt_ids[0]),
        }
        for _ in range(point.n_prompts)
    ]

    try:
        if rollout_based:
            with NvmlPeakSampler(device_index) as generation_sampler:
                completion_ids, action_masks, sampling_logps = agent.get_action(
                    prompts, training=True
                )

            # Sleep the engine *before* opening the training window. ``learn``
            # sleeps it internally, so a window opened around ``learn`` would
            # sample the still-awake engine and report the rollout footprint
            # as the training peak. Idempotent: guarded by the awake flag.
            agent._prepare_vllm_for_training()

            # One reward per trajectory row; completions arrive as one
            # (group_size, seq) tensor per prompt.
            n_trajectories = sum(ids.shape[0] for ids in completion_ids)
            experiences = (
                completion_ids,
                action_masks,
                torch.randn(n_trajectories),
            )
            learn_kwargs = {"sampling_logps": sampling_logps}
            generation_peak = generation_sampler.peak_bytes
        else:
            # SFT and DPO train from a dataset, so there is no rollout to
            # measure. Synthesise a batch of the right shape: content is
            # irrelevant to memory, only shapes matter.
            rows = point.micro_batch
            ids = torch.randint(
                low=10,
                high=tokenizer.vocab_size - 10,
                size=(rows, point.seq_len),
                device=agent.device,
            )
            mask = torch.ones_like(ids)
            if point.algorithm == "sft":
                experiences = {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "prompt_lengths": torch.full((rows,), prompt_len),
                }
            else:
                # DPO scores a chosen and a rejected sequence per row, so its
                # effective batch is twice the nominal one.
                experiences = (
                    ids,
                    ids.clone(),
                    mask,
                    mask.clone(),
                    torch.randn(rows),
                )
            learn_kwargs = {}
            generation_peak = None

        torch.cuda.reset_peak_memory_stats(device_index)
        reserved_at_entry = int(torch.cuda.memory_reserved(device_index))
        if snapshot_path:
            # Scope the trace to the training window. Otherwise the largest
            # peak in the trace belongs to vLLM's start-up KV allocation and
            # the training phase — the thing being attributed — never shows
            # up as the maximum. The trainer's base weights are moved back
            # onto the device inside ``learn``, so they are still captured.
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=100_000,
                clear_history=True,
            )
        with NvmlPeakSampler(device_index) as training_sampler:
            agent.learn(experiences, **learn_kwargs)
        if snapshot_path:
            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"wrote allocator snapshot to {snapshot_path}", flush=True)
        torch_allocated = int(torch.cuda.max_memory_allocated(device_index))
        torch_reserved = int(torch.cuda.max_memory_reserved(device_index))
    finally:
        agent.clean_up()

    generation = (
        MeasuredPoint(
            knobs=point.as_dict(),
            phase="generation",
            device_peak_bytes=generation_peak,
            nvml_polled_bytes=generation_peak,
        )
        if generation_peak is not None
        else None
    )
    # NVML polling can miss the brief backward spike, so correct with torch's
    # exact high-water mark — but only its *growth* across the window.
    # Absolute torch stats are unusable here: vLLM's CuMem pages stay on
    # torch's books after sleep releases them physically (measured: torch
    # reports ~9 GiB allocated while the device shows ~0.9 GiB), so the
    # phantom cancels in the delta but not in the absolute.
    torch_growth = max(torch_reserved - reserved_at_entry, 0)
    training = MeasuredPoint(
        knobs=point.as_dict(),
        phase="training",
        device_peak_bytes=max(
            training_sampler.peak_bytes,
            training_sampler.baseline_bytes + torch_growth,
        ),
        nvml_polled_bytes=training_sampler.peak_bytes,
        # Device memory the sleeping engine leaves resident (CUDA context +
        # engine structures) — the floor the trainer builds on.
        sleeping_baseline_bytes=training_sampler.baseline_bytes,
        torch_max_allocated_bytes=torch_allocated,
        torch_max_reserved_bytes=torch_reserved,
    )
    return generation, training


#: Attribute names the framework strips for text-only RL on a multimodal
#: base, matching ``VLLMConfig.strip_multimodal_towers``.
MULTIMODAL_TOWER_ATTRS = (
    "vision_tower",
    "audio_tower",
    "multi_modal_projector",
    "embed_vision",
    "embed_audio",
)


def _storage_bytes(module: object) -> int:
    """Realised bytes of a module's parameters and buffers.

    Sums distinct storages rather than trusting nominal dtypes: quantized
    layers report their packed size, and tied weights are counted once.
    """
    total = 0
    seen: set[int] = set()
    for tensor in list(module.parameters()) + list(module.buffers()):  # type: ignore[attr-defined]
        storage = tensor.untyped_storage()
        if storage.data_ptr() in seen:
            continue
        seen.add(storage.data_ptr())
        total += storage.nbytes()
    return total


def measure_realised_weight_bytes(
    model_name: str,
    quantization: str = "none",
    device_index: int = 0,
) -> dict[str, int]:
    """Measure realised weight bytes for every variant this checkpoint offers.

    Always returns the full model. When the checkpoint carries multimodal
    towers it also returns the text-only size, because stripping them is a
    real deployment option for text-only RL and the saving is not derivable
    from the text config alone. Nominal bits-per-weight cannot be trusted
    either: quantization scales, zero points, and layers held at higher
    precision all eat into the notional saving, which is why this loads the
    model rather than computing.
    """
    import torch

    from agilerl.utils.llm_utils import create_model_from_name_or_path

    model_config: dict[str, object] | None = None
    if quantization == "nf4":
        from transformers import BitsAndBytesConfig

        model_config = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        }
    model = create_model_from_name_or_path(model_name, model_config=model_config)

    sizes = {"full": _storage_bytes(model)}
    towers = {
        attr: getattr(model, attr, None) or getattr(model.base_model, attr, None)
        for attr in MULTIMODAL_TOWER_ATTRS
        if getattr(model, attr, None) is not None
        or getattr(getattr(model, "base_model", None), attr, None) is not None
    }
    if towers:
        tower_bytes = sum(_storage_bytes(module) for module in towers.values())
        sizes["stripped"] = sizes["full"] - tower_bytes
        sizes["towers"] = tower_bytes

    del model
    torch.cuda.empty_cache()
    return sizes


def main(argv: list[str] | None = None) -> int:
    """Measure one point (and optionally its realised weight bytes) in this
    process, writing the result JSON to ``--out``. Invoked one process per
    point by :func:`agilerl.memory.profiling.sweep.run_sweep`.
    """
    parser = argparse.ArgumentParser(prog="agilerl.memory.profiling.harness")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--quantization", default="none")
    parser.add_argument(
        "--algorithm", default="grpo", choices=["grpo", "ppo", "sft", "dpo"]
    )
    parser.add_argument(
        "--lora-target-modules",
        default="all-linear",
        help="Comma-separated module names, or the all-linear shorthand",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="auto | eager | sdpa | flash_attention_2 | flex_attention",
    )
    parser.add_argument(
        "--lora-target-scope",
        default=None,
        help="Restrict LoRA targeting (e.g. language_model for multimodal)",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Only measure realised weight bytes for --quantization",
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help=(
            "Also dump a torch allocator snapshot of the training phase to "
            "this path, for pytorch.org/memory_viz. Attributes memory per "
            "call site; blind to vLLM's CuMem allocations."
        ),
    )
    args = parser.parse_args(argv)

    if args.weights_only:
        result = {
            "realised_weight_bytes": measure_realised_weight_bytes(
                args.model, args.quantization, args.device_index
            )
        }
    else:
        point = SweepPoint(
            seq_len=args.seq_len,
            micro_batch=args.micro_batch,
            group_size=args.group_size,
            lora_rank=args.lora_rank,
            quantization=args.quantization,
            lora_target_scope=args.lora_target_scope,
            attn_implementation=args.attn_implementation,
            algorithm=args.algorithm,
            lora_target_modules=args.lora_target_modules,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        generation, training = measure_point(
            args.model,
            point,
            device_index=args.device_index,
            snapshot_path=args.snapshot,
        )
        result = {
            "generation": (generation.model_dump(mode="json") if generation else None),
            "training": training.model_dump(mode="json"),
        }
    Path(args.out).write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
