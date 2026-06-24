import gc
import torch
import pytest
from importlib.util import find_spec
from accelerate.state import AcceleratorState
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin


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
