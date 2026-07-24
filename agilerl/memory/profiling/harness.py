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
from dataclasses import dataclass
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

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "seq_len": self.seq_len,
            "micro_batch": self.micro_batch,
            "group_size": self.group_size,
            "lora_rank": self.lora_rank,
            "quantization": self.quantization,
        }

    def training_knobs(self) -> TrainingKnobs:
        return TrainingKnobs(
            micro_batch_size_per_gpu=self.micro_batch,
            batch_size=self.micro_batch,
            group_size=self.group_size,
            max_model_len=self.seq_len,
            lora_rank=self.lora_rank,
            quantization=self.quantization,  # type: ignore[arg-type]
        )

    def generation_knobs(self, gpu_memory_utilization: float) -> GenerationKnobs:
        return GenerationKnobs(
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=self.group_size,
            max_model_len=self.seq_len,
            max_lora_rank=self.lora_rank,
            concurrent_requests=self.group_size,
        )


def measure_point(
    model_name: str,
    point: SweepPoint,
    device_index: int = 0,
    gpu_memory_utilization: float = 0.45,
    prompt_len: int | None = None,
    n_prompts: int = 1,
) -> tuple[MeasuredPoint, MeasuredPoint]:
    """Run one sweep point and return (generation, training) measurements.

    The prompt length defaults to a quarter of ``seq_len`` so completions can
    exercise the remaining context budget.
    """
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer

    from agilerl.algorithms.grpo import GRPO
    from agilerl.memory.profiling.nvml import NvmlPeakSampler
    from agilerl.utils.algo_utils import VLLMConfig

    prompt_len = prompt_len or max(point.seq_len // 4, 8)
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

    agent = GRPO(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        pad_token=tokenizer.pad_token or tokenizer.eos_token,
        model_name=model_name,
        group_size=point.group_size,
        micro_batch_size_per_gpu=point.micro_batch,
        batch_size=point.micro_batch,
        max_model_len=point.seq_len,
        max_output_tokens=point.seq_len - prompt_len,
        lora_config=LoraConfig(
            r=point.lora_rank,
            lora_alpha=2 * point.lora_rank,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        quantization_config=quantization_config,
        use_vllm=True,
        vllm_config=VLLMConfig(
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=point.group_size,
            max_lora_rank=point.lora_rank,
            sleep_mode=True,
        ),
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
        for _ in range(n_prompts)
    ]

    try:
        with NvmlPeakSampler(device_index) as generation_sampler:
            completion_ids, action_masks, sampling_logps = agent.get_action(
                prompts, training=True
            )

        # One reward per trajectory row; completions arrive as one
        # (group_size, seq) tensor per prompt.
        n_trajectories = sum(ids.shape[0] for ids in completion_ids)
        rewards = torch.randn(n_trajectories)
        torch.cuda.reset_peak_memory_stats(device_index)
        reserved_at_entry = int(torch.cuda.memory_reserved(device_index))
        with NvmlPeakSampler(device_index) as training_sampler:
            agent.learn(
                (completion_ids, action_masks, rewards),
                sampling_logps=sampling_logps,
            )
        torch_allocated = int(torch.cuda.max_memory_allocated(device_index))
        torch_reserved = int(torch.cuda.max_memory_reserved(device_index))
    finally:
        agent.clean_up()

    generation = MeasuredPoint(
        knobs=point.as_dict(),
        phase="generation",
        device_peak_bytes=generation_sampler.peak_bytes,
        nvml_polled_bytes=generation_sampler.peak_bytes,
    )
    # NVML polling can miss the brief backward spike; torch's exact reserved
    # high-water mark plus the non-torch resident memory (vLLM sleeping
    # residual + CUDA context) is a tighter lower bound on the device peak.
    nontorch_baseline = max(training_sampler.baseline_bytes - reserved_at_entry, 0)
    training = MeasuredPoint(
        knobs=point.as_dict(),
        phase="training",
        device_peak_bytes=max(
            training_sampler.peak_bytes, nontorch_baseline + torch_reserved
        ),
        nvml_polled_bytes=training_sampler.peak_bytes,
        torch_max_allocated_bytes=torch_allocated,
        torch_max_reserved_bytes=torch_reserved,
    )
    return generation, training


def measure_realised_weight_bytes(
    model_name: str, quantization: str = "none", device_index: int = 0
) -> int:
    """Load the trainer-side model alone and measure realised weight bytes.

    Sums parameter and buffer storage directly (robust to allocator
    rounding); quantized layers report their packed storage sizes.
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
    total = 0
    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += tensor.untyped_storage().nbytes()
    del model
    torch.cuda.empty_cache()
    return total


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
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Only measure realised weight bytes for --quantization",
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
        )
        generation, training = measure_point(
            args.model,
            point,
            device_index=args.device_index,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        result = {
            "generation": generation.model_dump(mode="json"),
            "training": training.model_dump(mode="json"),
        }
    Path(args.out).write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
