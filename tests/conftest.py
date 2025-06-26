import gc

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from tests.helper_functions import (
    gen_multi_agent_dict_or_tuple_spaces,
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_multi_agent_multidiscrete_spaces,
    generate_multidiscrete_space,
    generate_random_box_space,
)


# Only clear CUDA cache when actually needed
@pytest.fixture(autouse=True, scope="function")
def cleanup():
    yield  # Run the test first
    # Only clear CUDA cache if CUDA was actually used
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    # Only collect garbage periodically, not after every test
    if hasattr(cleanup, "call_count"):
        cleanup.call_count += 1
    else:
        cleanup.call_count = 1

    # Only run garbage collection every 10 tests
    if cleanup.call_count % 10 == 0:
        gc.collect()


# Shared device fixture to avoid repeated device checks
@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Common observation spaces (session-scoped for reuse)
@pytest.fixture(scope="session")
def vector_space():
    return generate_random_box_space(shape=(4,))


@pytest.fixture(scope="session")
def discrete_space():
    return generate_discrete_space(2)


@pytest.fixture(scope="session")
def dict_space():
    return generate_dict_or_tuple_space(2, 2, dict_space=True)


@pytest.fixture(scope="session")
def multidiscrete_space():
    return generate_multidiscrete_space(2, 2)


@pytest.fixture(scope="session")
def multibinary_space():
    return spaces.MultiBinary(4)


@pytest.fixture(scope="session")
def image_space():
    return generate_random_box_space(shape=(3, 32, 32), low=0, high=255)


# Common multi-agent spaces
@pytest.fixture(scope="session")
def ma_vector_space():
    return generate_multi_agent_box_spaces(3, (6,))


@pytest.fixture(scope="session")
def ma_discrete_space():
    return generate_multi_agent_discrete_spaces(3, 2)


@pytest.fixture(scope="session")
def ma_multidiscrete_space():
    return generate_multi_agent_multidiscrete_spaces(3, 2)


@pytest.fixture(scope="session")
def ma_multibinary_space():
    return [spaces.MultiBinary(2) for _ in range(3)]


@pytest.fixture(scope="session")
def ma_image_space():
    return generate_multi_agent_box_spaces(3, (3, 32, 32), low=0, high=255)


@pytest.fixture(scope="session")
def ma_dict_space():
    return gen_multi_agent_dict_or_tuple_spaces(3, 2, 2, dict_space=True)


# Simple network fixtures (function-scoped to avoid state issues)
@pytest.fixture(scope="function")
def simple_mlp():
    return nn.Sequential(
        nn.Linear(4, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        nn.Softmax(dim=-1),
    )


@pytest.fixture(scope="function")
def simple_mlp_critic():
    return nn.Sequential(
        nn.Linear(6, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Tanh(),
    )


@pytest.fixture(scope="function")
def simple_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
        nn.Softmax(dim=-1),
    )
