import gc
import random
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin

try:
    import deepspeed.comm.comm as ds_comm
    import deepspeed.utils.groups as ds_groups
except ImportError:
    ds_comm = None
    ds_groups = None

try:
    cleanup_dist_env_and_memory = import_module(
        "vllm.distributed"
    ).cleanup_dist_env_and_memory
    destroy_model_parallel = import_module(
        "vllm.distributed.parallel_state"
    ).destroy_model_parallel
except ImportError:
    cleanup_dist_env_and_memory = None
    destroy_model_parallel = None

from tests.utils import (
    force_gpu_memory_release,
    wait_for_gpu_memory_to_clear,
)


@pytest.fixture(autouse=True)
def cleanup_after_test(request):
    if torch.cuda.is_available() and (num_gpus := torch.cuda.device_count()) > 0:
        # All GPU-touching tests are pinned to one xdist worker (see
        # tests/conftest.py), so this process owns the GPU and we can gate on
        # the global free-memory threshold.
        wait_for_gpu_memory_to_clear(
            devices=list(range(num_gpus)), threshold_ratio=0.4
        )

    yield

    if (
        "vllm" in request.node.name
        and destroy_model_parallel is not None
        and cleanup_dist_env_and_memory is not None
        and ds_groups is not None
        and ds_comm is not None
    ):
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def generate_accelerator(use_deepspeed_optimizer, config):
    if config is not None and not torch.cuda.is_available():
        pytest.skip("DeepSpeed-configured LLM tests require CUDA support.")
    if config is not None and find_spec("deepspeed") is None:
        pytest.skip("DeepSpeed-configured LLM tests require deepspeed.")

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


@pytest.fixture(scope="function")
def accelerator_factory():
    return generate_accelerator


def generate_model(pretrained_model_name_or_path):
    pytest.importorskip("peft", reason="LLM tests require peft.")
    pytest.importorskip("transformers", reason="LLM tests require transformers.")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

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
        dtype=(
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float32
        ),
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    return model


@pytest.fixture(scope="function")
def model_factory():
    return generate_model
