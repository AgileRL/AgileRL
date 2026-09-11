# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import os
import random
from importlib.util import find_spec

import numpy as np
import pytest
import torch
import torch.distributed as dist

from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from tests.utils import (
    force_gpu_memory_release,
    wait_for_gpu_memory_to_clear,
)


def _cuda_bf16_available() -> bool:
    """True only when a usable CUDA device supports bf16.

    ``torch.cuda.is_bf16_supported()`` calls ``current_device()`` internally,
    which raises ``RuntimeError`` when the CUDA driver is loaded but no device
    is visible (e.g. ``CUDA_VISIBLE_DEVICES=`` on the CPU CI runner, where
    ``is_available()`` can still report True). Guard against that so model
    construction falls back to fp32 instead of crashing.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except RuntimeError:
        return False


@pytest.fixture(autouse=True)
def cleanup_after_test(request):
    if torch.cuda.is_available() and (num_gpus := torch.cuda.device_count()) > 0:
        # Under xdist, multiple LLM workers share the same GPU, so the global
        # free-memory threshold is meaningless — peer workers' allocations
        # would never clear. Just release this worker's own memory and proceed.
        # In single-process runs we can still gate on the global threshold.
        if os.environ.get("PYTEST_XDIST_WORKER") is None:
            wait_for_gpu_memory_to_clear(
                devices=list(range(num_gpus)), threshold_ratio=0.4
            )
        else:
            force_gpu_memory_release()

    yield

    # vLLM distributed cleanup (only when vLLM tests ran and deps exist). Imported
    # lazily so `pytest -m "not llm"` can collect this package on hosts without vllm.
    if "vllm" in request.node.name and find_spec("vllm") is not None:
        from vllm.distributed import cleanup_dist_env_and_memory
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        cleanup_dist_env_and_memory()

    torch._dynamo.reset()
    force_gpu_memory_release()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def generate_model(pretrained_model_name_or_path, add_value_head=False):
    """Build a dense base model for ``actor_network``.

    AgileRL attaches and manages its own LoRA adapters (PeftModel inputs are
    rejected), so the factory returns the unwrapped base model.
    """
    pytest.importorskip("peft", reason="LLM tests require peft.")
    pytest.importorskip("transformers", reason="LLM tests require transformers.")
    from transformers import AutoModelForCausalLM

    if add_value_head:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.gradient_checkpointing_enable()
        return model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        dtype=torch.bfloat16 if _cuda_bf16_available() else torch.float32,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    return model


@pytest.fixture
def model_factory():
    return generate_model
