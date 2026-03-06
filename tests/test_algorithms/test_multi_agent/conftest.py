import pytest
from accelerate.state import AcceleratorState


@pytest.fixture(autouse=True)
def reset_accelerator_state():
    """Reset the AcceleratorState singleton before each test so that
    distributed-training state leaked by earlier tests (e.g. DeepSpeed
    leaving torch.distributed initialised) does not cause the Accelerator
    to wrap models in DDP unexpectedly."""
    AcceleratorState._reset_state(True)
    yield
    AcceleratorState._reset_state(True)
