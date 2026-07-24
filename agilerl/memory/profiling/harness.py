"""Measure one profiling point: real GRPO generation + training peaks.

Each point instantiates a colocated GRPO agent with the point's knobs, runs
one rollout and one learn, and records per-phase peaks:

- generation: NVML device peak (the only signal that sees vLLM's CuMem
  allocations) around ``get_action``;
- training: NVML device peak plus ``torch.cuda.max_memory_allocated`` /
  ``max_memory_reserved`` around ``learn``.

Requires a CUDA device with the ``llm`` extra installed; import stays lazy so
the sweep planner and fitter remain importable everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        low=10, high=tokenizer.vocab_size - 10, size=(prompt_len,)
    )
    prompts = [
        {
            "input_ids": prompt_ids.clone(),
            "attention_mask": torch.ones_like(prompt_ids),
        }
        for _ in range(n_prompts)
    ]

    try:
        with NvmlPeakSampler(device_index) as generation_sampler:
            completion_ids, action_masks, sampling_logps = agent.get_action(
                prompts, training=True
            )

        rewards = torch.randn(len(completion_ids))
        torch.cuda.reset_peak_memory_stats(device_index)
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
        nvml_peak_bytes=generation_sampler.peak_bytes,
    )
    training = MeasuredPoint(
        knobs=point.as_dict(),
        phase="training",
        nvml_peak_bytes=training_sampler.peak_bytes,
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

    kwargs: dict[str, object] = {}
    if quantization == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = create_model_from_name_or_path(model_name, **kwargs)
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
