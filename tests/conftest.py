import gc

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
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
    yield

    # Only clear CUDA cache if CUDA was actually used
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()

    # Only collect garbage periodically, not after every test
    if hasattr(cleanup, "call_count"):
        cleanup.call_count += 1
    else:
        cleanup.call_count = 1

    # Only run garbage collection every 10 tests
    if cleanup.call_count % 5 == 0:
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
def tuple_space():
    return generate_dict_or_tuple_space(2, 2, dict_space=False)


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


@pytest.fixture(scope="session")
def ma_dict_space_small():
    return gen_multi_agent_dict_or_tuple_spaces(3, 1, 1, dict_space=True)


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


########################################################
################# MUTATIONS ############################
########################################################


@pytest.fixture(scope="session")
def ac_hp_config():
    return HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )


@pytest.fixture(scope="session")
def default_hp_config():
    yield HyperparameterConfig(
        lr=RLParameter(min=6.25e-5, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=1, max=10, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )


@pytest.fixture(scope="session")
def grpo_hp_config():
    yield HyperparameterConfig(
        lr=RLParameter(min=6.25e-5, max=1e-2),
    )


@pytest.fixture(scope="session")
def encoder_mlp_config():
    yield {
        "latent_dim": 8,
        "min_latent_dim": 1,
        "encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
        "head_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
    }


@pytest.fixture(scope="session")
def encoder_simba_config():
    yield {
        "latent_dim": 8,
        "min_latent_dim": 1,
        "simba": True,
        "encoder_config": {
            "hidden_size": 64,
            "num_blocks": 3,
        },
        "head_config": {"hidden_size": [8], "min_mlp_nodes": 1},
    }


@pytest.fixture(scope="session")
def encoder_cnn_config():
    yield {
        "latent_dim": 8,
        "min_latent_dim": 1,
        "encoder_config": {
            "channel_size": [5, 5],
            "kernel_size": [3, 3],
            "stride_size": [1, 1],
            "min_channel_size": 1,
            "max_channel_size": 10,
        },
        "head_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
    }


@pytest.fixture(scope="session")
def encoder_multi_input_config():
    yield {
        "latent_dim": 8,
        "min_latent_dim": 1,
        "encoder_config": {
            "cnn_config": {
                "channel_size": [5],
                "kernel_size": [3],
                "stride_size": [1],
                "min_channel_size": 1,
                "max_channel_size": 10,
            },
            "mlp_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
        },
        "head_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
    }


class EvoDummyRNG:
    rng = np.random.default_rng(seed=42)

    def choice(self, a, size=None, replace=True, p=None):
        return 1

    def integers(self, low=0, high=None):
        return self.rng.integers(low, high)


@pytest.fixture(scope="session")
def dummy_rng():
    return EvoDummyRNG()
