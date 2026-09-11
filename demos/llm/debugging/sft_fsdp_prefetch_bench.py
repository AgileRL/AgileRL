#!/usr/bin/env python3
# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""SFT + FSDP2 prefetch bench on Qwen2.5-0.5B (matches oss sft.yaml model size).

Captures ``agilerl.fsdp_profile`` INFO timers from one warmup ``learn()``
plus several measured steps. Rank 0 prints a stage summary.

Launch::

    CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens7 NCCL_NET=Socket \\
      PYTHONUNBUFFERED=1 \\
      torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:29502 \\
      demos/llm/debugging/sft_fsdp_prefetch_bench.py --no-gc --seq-len 1024
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from peft import LoraConfig
from transformers import AutoTokenizer

from agilerl.algorithms.sft import SFT
from agilerl.distributed import FSDPConfig, distributed_env_present, get_rank
from agilerl.distributed.fsdp_profile import logger as fsdp_logger

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH = 2
PROMPT_LEN = 32
WARMUP = 1
STEPS = 4


class ProfileCapture(logging.Handler):
    """Keep parsed ``fsdp_profile`` INFO fields."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.rows: list[dict[str, str]] = []
        self.enabled = False

    def emit(self, record: logging.LogRecord) -> None:
        if not self.enabled:
            return
        msg = record.getMessage()
        if not msg.startswith("fsdp_profile "):
            return
        fields: dict[str, str] = {}
        for part in msg.split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
        self.rows.append(fields)


def _batch(
    device: torch.device, vocab_size: int, *, seq_len: int
) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (BATCH, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_lengths = torch.full((BATCH,), PROMPT_LEN, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": prompt_lengths,
    }


def _mean_ms(rows: list[dict[str, str]], stage: str) -> float | None:
    values = [
        float(row["elapsed_ms"])
        for row in rows
        if row.get("stage") == stage and row.get("rank") == "0"
    ]
    if not values:
        return None
    return statistics.mean(values)


def _sum_ms(
    rows: list[dict[str, str]], stage: str, *, parent: str | None = None
) -> float:
    total = 0.0
    for row in rows:
        if row.get("rank") != "0" or row.get("stage") != stage:
            continue
        if parent is None:
            if "parent" in row:
                continue
        elif row.get("parent") != parent:
            continue
        total += float(row["elapsed_ms"])
    return total


def _mib(nbytes: int) -> int:
    return int(nbytes / (1024 * 1024))


def _print_summary(label: str, rows: list[dict[str, str]], wall_s: float) -> None:
    stages = (
        "microbatch_forward",
        "microbatch_backward",
        "clip_grad",
        "optimizer_step",
        "zero_grad",
    )
    print(f"BENCH_LABEL={label}", flush=True)
    print(f"BENCH_WALL_S={wall_s:.3f}", flush=True)
    print(
        f"BENCH_PEAK_ALLOC_MIB={_mib(torch.cuda.max_memory_allocated())}",
        flush=True,
    )
    print(
        f"BENCH_PEAK_RESERVED_MIB={_mib(torch.cuda.max_memory_reserved())}",
        flush=True,
    )
    for stage in stages:
        mean = _mean_ms(rows, stage)
        if mean is None:
            print(f"BENCH_MEAN {stage}=n/a", flush=True)
            continue
        print(f"BENCH_MEAN {stage}={mean:.1f}", flush=True)
    print(
        "BENCH_SUM wait_unshard parent=microbatch_forward="
        f"{_sum_ms(rows, 'wait_unshard', parent='microbatch_forward'):.1f}",
        flush=True,
    )
    print(
        "BENCH_SUM wait_unshard parent=microbatch_backward="
        f"{_sum_ms(rows, 'wait_unshard', parent='microbatch_backward'):.1f}",
        flush=True,
    )
    print(
        "BENCH_SUM wait_unshard parent=none="
        f"{_sum_ms(rows, 'wait_unshard', parent=None):.1f}",
        flush=True,
    )
    print(
        "BENCH_SUM all_gather_unshard parent=microbatch_forward="
        f"{_sum_ms(rows, 'all_gather_unshard', parent='microbatch_forward'):.1f}",
        flush=True,
    )
    print(
        "BENCH_SUM all_gather_unshard parent=microbatch_backward="
        f"{_sum_ms(rows, 'all_gather_unshard', parent='microbatch_backward'):.1f}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--no-gc", action="store_true")
    parser.add_argument("--bwd-prefetch", choices=("on", "off"), default="on")
    return parser.parse_args()


def _disable_backward_prefetch(model: torch.nn.Module) -> None:
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPParamGroup,
    )

    FSDPParamGroup._backward_prefetch = lambda self: None
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        module._get_fsdp_state()._states_to_backward_prefetch = []


def main() -> int:
    if not torch.cuda.is_available():
        print("Need CUDA", file=sys.stderr)
        return 2
    if not distributed_env_present():
        print("Launch with torchrun so FSDP2 can initialise.", file=sys.stderr)
        return 2

    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    fsdp_logger.setLevel(logging.INFO)
    capture = ProfileCapture()
    fsdp_logger.addHandler(capture)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = int(tokenizer.vocab_size)

    agent = SFT(
        model_name=args.model,
        pad_token_id=int(tokenizer.pad_token_id),
        pad_token=tokenizer.pad_token,
        lora_config=LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        ),
        fsdp_config=FSDPConfig(reshard_after_forward=True),
        device=device,
        micro_batch_size_per_gpu=BATCH,
        batch_size=BATCH * 2,
        use_liger_loss=False,
        gradient_checkpointing=not args.no_gc,
        update_epochs=1,
    )
    if args.bwd_prefetch == "off":
        _disable_backward_prefetch(agent.actor)
    if get_rank() == 0:
        print(
            f"model={args.model} seq={args.seq_len} batch={BATCH} "
            f"gc={not args.no_gc} bwd_prefetch={args.bwd_prefetch} "
            f"reshard_after_forward={agent.fsdp_config.reshard_after_forward}",
            flush=True,
        )

    batch = _batch(device, vocab_size, seq_len=args.seq_len)
    for _ in range(WARMUP):
        agent.learn(batch, training=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    capture.rows.clear()
    capture.enabled = True
    start = time.perf_counter()
    for _ in range(STEPS):
        agent.learn(batch, training=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - start
    capture.enabled = False

    if get_rank() == 0:
        _print_summary(
            f"sft:{args.model}:bwd={args.bwd_prefetch}", capture.rows, wall_s
        )
        print("bench ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
