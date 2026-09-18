#!/usr/bin/env python3
# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""SFT + FSDP2 prefetch bench on Qwen2.5-0.5B (matches oss sft.yaml model size).

One warmup ``learn()`` plus several measured steps. Rank 0 prints wall time
and CUDA peak memory.

Launch::

    CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens7 NCCL_NET=Socket \\
      PYTHONUNBUFFERED=1 \\
      torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:29502 \\
      demos/llm/debugging/sft_fsdp_prefetch_bench.py --no-gc --seq-len 1024
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from peft import LoraConfig
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from transformers import AutoTokenizer

from agilerl.algorithms.sft import SFT
from agilerl.distributed import FSDPConfig, distributed_env_present, get_rank

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH = 2
PROMPT_LEN = 32
WARMUP = 1
STEPS = 4


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


def _mib(nbytes: int) -> int:
    return int(nbytes / (1024 * 1024))


def _print_summary(label: str, wall_s: float) -> None:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--no-gc", action="store_true")
    parser.add_argument("--bwd-prefetch", choices=("on", "off"), default="on")
    return parser.parse_args()


def _disable_backward_prefetch(model: torch.nn.Module) -> None:
    backward_prefetch = "_backward_prefetch"
    get_fsdp_state = "_get_fsdp_state"
    states_to_prefetch = "_states_to_backward_prefetch"
    object.__setattr__(FSDPParamGroup, backward_prefetch, lambda _self: None)
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        state = getattr(module, get_fsdp_state)()
        object.__setattr__(state, states_to_prefetch, [])


def main() -> int:
    if not torch.cuda.is_available():
        print("Need CUDA", file=sys.stderr)
        return 2
    if not distributed_env_present():
        print("Launch with torchrun so FSDP2 can initialise.", file=sys.stderr)
        return 2

    args = _parse_args()

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
    start = time.perf_counter()
    for _ in range(STEPS):
        agent.learn(batch, training=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - start

    if get_rank() == 0:
        _print_summary(f"sft:{args.model}:bwd={args.bwd_prefetch}", wall_s)
        print("bench ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
