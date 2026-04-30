import gc
import os
import random
from importlib.util import find_spec

import numpy as np
import pytest
import torch
from accelerate.state import AcceleratorState
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

from tests.utils import (
    force_gpu_memory_release,
    wait_for_gpu_memory_to_clear,
)


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

    # vLLM + DeepSpeed cleanup (only when vLLM tests ran and deps exist). Imported
    # lazily so `pytest -m "not llm"` can collect this package on hosts without vllm.
    if "vllm" in request.node.name and find_spec("vllm") is not None:
        try:
            import deepspeed.comm.comm as ds_comm
        except ImportError:
            ds_comm = None
        if ds_comm is not None:
            import deepspeed.utils.groups as ds_groups
            from vllm.distributed import cleanup_dist_env_and_memory
            from vllm.distributed.parallel_state import destroy_model_parallel

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


def generate_model(pretrained_model_name_or_path, add_value_head=False):
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
    if add_value_head:
        peft_config.modules_to_save = ["summary"]
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, peft_config)
        return model
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
