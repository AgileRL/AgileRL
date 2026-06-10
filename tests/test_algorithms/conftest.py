import gc
import os
import torch
import pytest
from accelerate.state import AcceleratorState
from accelerate import Accelerator


def generate_accelerator(mode, gradient_accumulation_steps=None):
    """Build an ``Accelerator`` for LLM algorithm tests.

    :param mode: ``None`` (no accelerator), ``"ddp"`` (plain torch-native
        accelerator; DDP under multi-process launch, single-process here) or
        ``"fsdp2"`` (FSDP2 sharding via ``fully_shard``, world size 1).
    :param gradient_accumulation_steps: Optional accumulation steps to
        configure on the accelerator.

    Accelerated modes need CUDA. Setting ``AGILERL_TEST_CPU_ACCELERATOR=1``
    lets the ``"ddp"`` mode run on a CPU-only machine (gloo backend) for
    local debugging; ``"fsdp2"`` always requires CUDA.
    """
    if mode is not None and not torch.cuda.is_available():
        cpu_ddp_ok = (
            mode == "ddp" and os.environ.get("AGILERL_TEST_CPU_ACCELERATOR") == "1"
        )
        if not cpu_ddp_ok:
            pytest.skip("Accelerated LLM tests require CUDA support.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)

    if mode is None:
        return None

    kwargs = {}
    if gradient_accumulation_steps is not None:
        kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps

    if mode == "fsdp2":
        from accelerate import FullyShardedDataParallelPlugin

        os.environ["ACCELERATE_USE_FSDP"] = "true"
        kwargs["fsdp_plugin"] = FullyShardedDataParallelPlugin(fsdp_version=2)
    else:
        os.environ.pop("ACCELERATE_USE_FSDP", None)

    return Accelerator(**kwargs)


@pytest.fixture(scope="function")
def accelerator_factory():
    return generate_accelerator
