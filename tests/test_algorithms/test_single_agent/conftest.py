import os

import pytest
import torch.distributed as dist

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
    the environment variables that are inspected when deciding whether to
    initialise a distributed backend.

    Without this, distributed LLM tests that run earlier on the same xdist
    worker can leave ``torch.distributed`` initialised and env-vars like
    ``WORLD_SIZE`` set, which can cause subsequent tests to attempt a
    multi-worker rendezvous or wrap networks in DDP (e.g.
    ``'DistributedDataParallel' object has no attribute 'encoder'``).
    Mirrors ``tests/test_algorithms/test_multi_agent/conftest.py``.
    """
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

    for var in _DIST_ENV_VARS:
        os.environ.pop(var, None)
