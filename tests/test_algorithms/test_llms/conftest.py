import gc
import socket

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

dist_env = dict(
    ACCELERATE_USE_DEEPSPEED="true",
    MASTER_ADDR="localhost",
    MASTER_PORT="10999",
    RANK="0",
    LOCAL_RANK="0",
    WORLD_SIZE="1",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
)


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def deepspeed_env():
    import os

    dynamic_dist_env = dist_env.copy()
    dynamic_dist_env["MASTER_PORT"] = str(get_free_port())
    existing_vars = {}
    for key, value in dynamic_dist_env.items():
        key = key.upper()
        if key in os.environ:
            existing_vars[key] = os.environ[key]
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key in dynamic_dist_env:
            key = key.upper()
            if key in existing_vars:
                # restore previous value
                os.environ[key] = existing_vars[key]
            else:
                os.environ.pop(key, None)
        gc.collect()
        torch.cuda.empty_cache()


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
