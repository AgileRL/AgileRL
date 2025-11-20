import gc
import random

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def cleanup_vllm_instances():
    """Clean up vLLM LLM instances and engines"""
    import vllm
    import vllm.engine.llm_engine

    # Clean global engine
    if hasattr(vllm, "_global_llm_engine"):
        try:
            vllm._global_llm_engine.shutdown()
        except Exception:
            pass
        del vllm._global_llm_engine

    # Clean any cached engines
    if hasattr(vllm.engine.llm_engine, "_cached_engines"):
        for engine in vllm.engine.llm_engine._cached_engines.values():
            try:
                engine.shutdown()
            except Exception:
                pass
        vllm.engine.llm_engine._cached_engines.clear()

    # Clean LLM class instances
    if hasattr(vllm, "LLM"):
        # Clear any class-level caches
        if hasattr(vllm.LLM, "_instances"):
            for instance in vllm.LLM._instances:
                try:
                    if hasattr(instance, "llm_engine"):
                        instance.llm_engine.shutdown()
                except Exception:
                    pass
            vllm.LLM._instances.clear()


@pytest.fixture(autouse=True)
def cleanup_after_test(request):
    test_name = request.node.name
    before = torch.cuda.memory_allocated() / 1e9
    yield
    cleanup_vllm_instances()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    AcceleratorState._reset_state(True)
    after = torch.cuda.memory_allocated() / 1e9
    delta = after - before
    if delta > 0.5:  # More than 500MB leaked
        print(f"\n⚠️  {test_name}: {before:.2f}GB → {after:.2f}GB (Δ {delta:+.2f}GB)")


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
