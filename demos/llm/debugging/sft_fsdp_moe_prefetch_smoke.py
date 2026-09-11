#!/usr/bin/env python3
# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""SFT + FSDP2 + Liger prefetch smoke on the tiny Qwen3-MoE fixture.

Must complete one ``learn()`` with prefetch on and print ``learn ok``.
A regression of the Liger/PEFT false-root bug raises:

    AttributeError: 'FSDPCommContext' object has no attribute
    'all_gather_copy_in_stream'

Launch::

    CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens7 NCCL_NET=Socket \\
      torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:29501 \\
      demos/llm/debugging/sft_fsdp_moe_prefetch_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from peft import LoraConfig

from agilerl.algorithms.sft import SFT
from agilerl.distributed import FSDPConfig, distributed_env_present, get_rank
from demos.llm.debugging.fixtures.build_tiny_qwen3_moe import (
    VOCAB_SIZE,
    build_model,
)

SEQ_LEN = 16
BATCH = 2
PROMPT_LEN = 4


def _batch(device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_lengths = torch.full((BATCH,), PROMPT_LEN, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": prompt_lengths,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("Need CUDA", file=sys.stderr)
        return 2
    if not distributed_env_present():
        print("Launch with torchrun so FSDP2 can initialise.", file=sys.stderr)
        return 2

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = build_model().to("cpu")
    agent = SFT(
        actor_network=model,
        pad_token_id=5,
        pad_token="[PAD]",
        lora_config=LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
        fsdp_config=FSDPConfig(reshard_after_forward=True),
        device=device,
        micro_batch_size_per_gpu=BATCH,
        use_liger_loss=True,
        gradient_checkpointing=False,
        update_epochs=1,
    )

    inner = agent.actor.base_model.model
    if get_rank() == 0:
        print(
            f"reshard_after_forward={agent.fsdp_config.reshard_after_forward}",
            flush=True,
        )
        print(f"CausalLM.forward={inner.forward}", flush=True)
        print(
            f"liger_patched={getattr(inner, '_agilerl_liger_patched', False)}",
            flush=True,
        )

    try:
        metrics = agent.learn(_batch(device), training=True)
    except AttributeError as err:
        import traceback

        print(f"RANK {get_rank()} AttributeError: {err}", flush=True)
        traceback.print_exc()
        if "all_gather_copy_in_stream" in str(err):
            print("REPRODUCED prefetch crash", flush=True)
            return 1
        raise
    if get_rank() == 0:
        print(f"learn ok: {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
