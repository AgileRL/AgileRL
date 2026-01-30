import gc
import os
import random

import deepspeed.comm.comm as ds_comm
import deepspeed.utils.groups as ds_groups
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin
from peft import LoraConfig, get_peft_model
from torch._inductor.utils import fresh_cache
from transformers import AutoModelForCausalLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import destroy_model_parallel

from tests.utils import (
    force_gpu_memory_release,
    wait_for_gpu_memory_to_clear,
)


def _should_wait_for_gpu_memory() -> bool:
    """Check if GPU memory wait is enabled via environment variable."""
    return os.getenv("AGILERL_TEST_CLEAN_GPU_MEMORY", "1") == "1"


@pytest.fixture(autouse=True)
def use_fresh_cache():
    """Use a fresh inductor cache for each test."""
    with fresh_cache():
        yield


@pytest.fixture(autouse=True)
def cleanup_after_test(request):
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    if _should_wait_for_gpu_memory() and torch.cuda.is_available():
        wait_for_gpu_memory_to_clear(threshold_ratio=0.4, timeout_s=120)

    yield

    if "vllm" in request.node.name:
        # vLLM-specific cleanup
        destroy_model_parallel()
        cleanup_dist_env_and_memory()
        for attr in dir(ds_groups):
            if attr.startswith("_") and attr.endswith("_GROUP"):
                setattr(ds_groups, attr, None)
        ds_comm.cdb = None

    torch._dynamo.reset()
    force_gpu_memory_release()
    AcceleratorState._reset_state(True)


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    yield


@pytest.fixture(scope="function")
def accelerator_factory():
    def generate_accelerator(use_deepspeed_optimizer, config):
        gc.collect()
        torch.cuda.empty_cache()
        AcceleratorState._reset_state(True)
        if use_deepspeed_optimizer and (config is not None):
            config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4,  # Smaller learning rate
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                },
            }
        return (
            Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config))
            if config is not None
            else None
        )

    return generate_accelerator


@pytest.fixture(scope="function")
def model_factory():
    def generate_model(pretrained_model_name_or_path):
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, peft_config)
        return model

    return generate_model
