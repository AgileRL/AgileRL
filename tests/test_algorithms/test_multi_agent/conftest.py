import os

import pytest
import torch.distributed as dist
from accelerate.state import AcceleratorState

_DIST_ENV_VARS = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "LOCAL_WORLD_SIZE",
)


@pytest.fixture(autouse=True)
def reset_distributed_state():
    """Tear down any leaked ``torch.distributed`` process group and clear
    the environment variables that ``PartialState`` inspects when deciding
    whether to initialise a distributed backend.

    Without this, DeepSpeed / LLM tests that run earlier in the session can
    leave ``torch.distributed`` initialised and env-vars like ``WORLD_SIZE``
    set, which causes subsequent ``Accelerator()`` calls to attempt a
    multi-worker rendezvous (hanging on macOS / Windows, or wrapping models
    in DDP on Linux)."""
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

    for var in _DIST_ENV_VARS:
        os.environ.pop(var, None)

    AcceleratorState._reset_state(True)
