import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def cleanup():
    yield  # Run the test first
    torch.cuda.empty_cache()  # Free up GPU memory
    gc.collect()  # Collect garbage
