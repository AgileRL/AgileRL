"""Lightweight test runner for subprocess-isolated tests.

Replaces the pytest entrypoint with a direct python3 invocation to eliminate
pytest overhead (plugin discovery, collection, fixture resolution) and avoid
running all parametrized variants when only one is intended.

Usage:
    python -m tests.subprocess_runner \
        --module path/to/test_file.py \
        --test test_function_name \
        --params '{"key": "value", ...}' \
        --fixtures '["fixture_name", ...]'
"""

import argparse
import importlib.util
import json
import os
import random
import socket
import sys
import tempfile
import traceback

import numpy as np
import torch
from accelerate.state import AcceleratorState
from torch._inductor.utils import fresh_cache


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def setup_deepspeed_env():
    """Set up DeepSpeed distributed environment variables."""
    env_vars = {
        "ACCELERATE_USE_DEEPSPEED": "true",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(get_free_port()),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for key, value in env_vars.items():
        os.environ[key] = value


def setup_test_env_vars():
    """Set pytest ini env vars that would normally be set by pytest."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("AGILERL_TEST_CLEAN_GPU_MEMORY", "1")


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def wait_for_gpu_memory():
    """Wait for GPU memory to clear before running the test."""
    if torch.cuda.is_available() and (num_gpus := torch.cuda.device_count()) > 0:
        from tests.utils import wait_for_gpu_memory_to_clear

        wait_for_gpu_memory_to_clear(devices=list(range(num_gpus)), threshold_ratio=0.2)


def cleanup_after_test(test_name):
    """Run cleanup after the test, replicating conftest autouse fixtures."""
    import deepspeed.comm.comm as ds_comm
    import deepspeed.utils.groups as ds_groups

    from tests.utils import force_gpu_memory_release

    if "vllm" in test_name:
        from vllm.distributed import cleanup_dist_env_and_memory
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        cleanup_dist_env_and_memory()
        for attr in dir(ds_groups):
            if attr.startswith("_") and attr.endswith("_GROUP"):
                setattr(ds_groups, attr, None)
        ds_comm.cdb = None

    torch._dynamo.reset()
    force_gpu_memory_release()
    AcceleratorState._reset_state(True)


def import_module_from_path(module_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location("_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_module"] = module
    spec.loader.exec_module(module)
    return module


class _NodeStub:
    """Minimal stub for pytest's request.node, providing .name."""

    def __init__(self, name):
        self.name = name


class _RequestStub:
    """Minimal stub for pytest's FixtureRequest.

    Provides the subset of the interface that conftest fixtures rely on,
    primarily ``request.node.name`` used by ``cleanup_after_test``.
    """

    def __init__(self, test_name):
        self.node = _NodeStub(test_name)


def resolve_fixtures(fixture_names, test_module, test_name):
    """Build fixture values for the given fixture names.

    Returns a dict mapping fixture name to its resolved value.
    """
    fixtures = {}

    for name in fixture_names:
        if name == "deepspeed_env":
            setup_deepspeed_env()
            fixtures[name] = None
        elif name == "accelerator_factory":
            from tests.test_algorithms.test_llms.conftest import generate_accelerator

            fixtures[name] = generate_accelerator
        elif name == "model_factory":
            from tests.test_algorithms.test_llms.conftest import generate_model

            fixtures[name] = generate_model
        elif name == "grpo_factory":
            fixtures[name] = getattr(test_module, "generate_grpo")
        elif name == "dpo_factory":
            fixtures[name] = getattr(test_module, "generate_dpo")
        elif name == "preference_dataset_factory":
            fixtures[name] = getattr(test_module, "make_preference_gym")
        elif name == "request":
            fixtures[name] = _RequestStub(test_name)
        elif name == "tmpdir":
            fixtures[name] = tempfile.mkdtemp()
        else:
            raise ValueError(
                f"Unknown fixture '{name}'. Register it in subprocess_runner.py"
            )

    return fixtures


def main():
    parser = argparse.ArgumentParser(description="Run a single test function directly")
    parser.add_argument("--module", required=True, help="Path to test module file")
    parser.add_argument("--test", required=True, help="Test function name")
    parser.add_argument(
        "--params", required=True, help="JSON-serialized parametrized kwargs"
    )
    parser.add_argument(
        "--fixtures", required=True, help="JSON list of fixture names needed"
    )
    args = parser.parse_args()

    params = json.loads(args.params)
    fixture_names = json.loads(args.fixtures)

    setup_test_env_vars()

    module = import_module_from_path(args.module)

    # Unwrap only one level: past spawn_new_process_for_each_test but keep
    # any @patch wrappers intact so they inject mocks automatically.
    test_func = getattr(module, args.test)
    if hasattr(test_func, "__wrapped__"):
        test_func = test_func.__wrapped__

    fixtures = resolve_fixtures(fixture_names, module, args.test)

    kwargs = {**fixtures, **params}

    set_seed()
    wait_for_gpu_memory()

    with fresh_cache():
        try:
            test_func(**kwargs)
        finally:
            cleanup_after_test(args.test)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
