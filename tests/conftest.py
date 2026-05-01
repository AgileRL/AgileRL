import gc
import os
import shutil
import socket
import sys
import tempfile

# Give each xdist worker its own torch inductor cache dir BEFORE torch is
# imported. Parallel workers sharing the default cache race on precompiled
# headers (mtime checks fail on macOS clang++ and can cause flaky rebuilds
# elsewhere). When the CI presets TORCHINDUCTOR_CACHE_DIR for restoration,
# we nest each worker under it so cache reuse across runs still works.
# This is a no-op when running without xdist.
_xdist_worker_id = os.environ.get("PYTEST_XDIST_WORKER")
if _xdist_worker_id:
    _inductor_base = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or tempfile.gettempdir()
    _worker_cache = os.path.join(_inductor_base, f"worker_{_xdist_worker_id}")

    def _writable(path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path, delete=True):
                pass
            return True
        except OSError:
            return False

    # A stale dir left by a previous run (e.g. created inside a container as
    # root, or with restrictive perms after a tmpwatch sweep on /var/tmp) can
    # be unwritable by the current user, causing torch.compile to crash with
    # PermissionError. Probe both the worker dir and the inner ``cache/``
    # subdir torch.compile creates; wipe and retry on failure, falling back
    # to ``mkdtemp`` so the run can always proceed.
    _inner = os.path.join(_worker_cache, "cache")
    if not (
        _writable(_worker_cache) and (not os.path.exists(_inner) or _writable(_inner))
    ):
        shutil.rmtree(_worker_cache, ignore_errors=True)
        if not _writable(_worker_cache):
            _worker_cache = tempfile.mkdtemp(
                prefix=f"torchinductor_{_xdist_worker_id}_"
            )

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _worker_cache

# Force HF libs offline during tests. Any test that tries to download from the
# Hub fails loudly instead of silently fetching (and getting CI rate-limited).
# Tests that need an LLM use tests/assets/tiny_llm/ via TINY_LLM_FIXTURE_PATH.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Tests that construct ``Accelerator()`` directly (instead of via the
# ``deepspeed_env`` fixture) inherit torch's default MASTER_PORT. Parallel
# xdist workers then race to bind the same port and one fails with
# ``EADDRINUSE``. Give each worker a deterministic, unique MASTER_PORT here
# so any later distributed init lands on a non-colliding port.
if _xdist_worker_id:
    _worker_num = int("".join(c for c in _xdist_worker_id if c.isdigit()) or "0")
    os.environ.setdefault("MASTER_PORT", str(29500 + _worker_num))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from accelerate.state import AcceleratorState, PartialState  # noqa: E402
from gymnasium import spaces  # noqa: E402
from torch import nn  # noqa: E402

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter  # noqa: E402
from tests.helper_functions import (  # noqa: E402
    gen_multi_agent_dict_or_tuple_spaces,
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_multi_agent_multidiscrete_spaces,
    generate_multidiscrete_space,
    generate_random_box_space,
)

if not torch.cuda.is_available():
    os.environ.setdefault("ACCELERATE_USE_CPU", "true")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Pin tests that share mutable global state to dedicated xdist workers so
    they never run in parallel with each other.

    - ``vllm``- and ``gpu``-marked tests share a single pool of four
      ``gputest0``..``gputest3`` xdist groups, sized to cap **total
      concurrent GPU-touching tests at 4** regardless of the runner's
      CPU count. With ``--dist loadgroup`` and ``-n auto`` on the 8-core
      Linux runner, four workers consume the GPU pool in parallel and the
      remaining four fan out across CPU-only tests — so a single ``pytest``
      invocation handles both. The cap is set by:

      1. **GPU memory.** Each container has a dedicated ~14.6 GiB GPU. Peak
         per-test usage: ~4.4 GiB (``test_grpo_move_model_to_vllm``), ~3.0
         GiB (``test_grpo_learn``), ~2.5 GiB
         (``test_grpo_clone_with_accelerator_vllm``); median ~1.5 GiB.
         With every test factory constructing ``VLLMConfig`` with
         ``gpu_memory_utilization≈0.2`` and
         ``kv_cache_memory_bytes=32 * 1024 * 1024``, a vLLM worker reserves
         ~3.2 GiB and ``gpu`` (DeepSpeed) tests use ~0.5 GiB; the worst-case
         "4 vLLM workers + small DeepSpeed share" still fits in ~13 GiB.
      2. **Port races.** Each worker's ``deepspeed_env`` fixture allocates a
         free ``MASTER_PORT`` via the standard bind-to-port-0 / close /
         return dance, which is TOCTOU. Above ~4 concurrent workers the
         collision rate produces ``EADDRINUSE`` during
         ``torch.distributed.init_process_group``.

      ``vllm`` tests run in ``subprocess_runner.py``-spawned subprocesses, so
      worker-process state is reset between them. ``gpu`` tests run
      in-process and can leak DeepSpeed groups / accelerator state to the
      next test sharing the same group; the per-fixture cleanup
      (``AcceleratorState._reset_state(True)`` etc.) handles this in
      practice for the test sets in this repo, but **don't add many more
      ``gpu``-marked tests without re-checking** — DeepSpeed has no clean
      ``destroy_process_group`` path so sharing a worker between two
      DeepSpeed-init tests can surface ``Group <ProcessGroup ...> is not
      registered`` or ``EADDRINUSE``-on-MASTER_PORT.
    - ``test_minari_utils``: tests create/delete shared Minari datasets on disk.

    Uses ``tryfirst=True`` so the ``xdist_group`` markers below are attached
    before xdist's own ``pytest_collection_modifyitems`` (in ``xdist/remote.py``)
    reads them and appends ``@group`` suffixes to nodeids for loadgroup
    scheduling.

    Background on ``kv_cache_memory_bytes``: vLLM's
    ``determine_available_memory`` profile run asserts that GPU free-memory
    does not increase between the pre- and post-profile snapshots. When peer
    processes on the same GPU (concurrent xdist workers, sibling CI
    containers sharing one GPU) release memory mid-profile, the assertion
    fires with ``Error in memory profiling. Initial free memory ... current
    free memory ...``. Setting ``kv_cache_memory_bytes`` triggers vLLM's
    early-return path in ``determine_available_memory`` and skips the
    assertion entirely. When adding new vLLM-using tests, route them through
    the existing ``generate_grpo`` / ``generate_reinforce`` factories — or
    set both ``kv_cache_memory_bytes`` and a small ``gpu_memory_utilization``
    on the ``VLLMConfig`` directly.
    """
    # Single shared pool: vllm-marked + gpu-marked tests round-robin into
    # the same 4 xdist groups, capping total GPU-test concurrency at 4
    # regardless of -n auto's worker count. See docstring above.
    gputest_groups = [pytest.mark.xdist_group(f"gputest{i}") for i in range(4)]
    minari_group = pytest.mark.xdist_group("minari")
    gputest_count = 0
    for item in items:
        if item.get_closest_marker("vllm") or item.get_closest_marker("gpu"):
            item.add_marker(gputest_groups[gputest_count % len(gputest_groups)])
            gputest_count += 1
        elif "test_minari_utils" in item.nodeid:
            item.add_marker(minari_group)


# Only clear CUDA cache when actually needed
@pytest.fixture(autouse=True, scope="function")
def cleanup():
    # Reset the process-wide ``AcceleratorState`` / ``PartialState`` singletons
    # **before** every test. Both are accelerate's shared-state caches keyed by
    # device, so once any test instantiates an ``Accelerator()`` the device is
    # frozen for the rest of the worker's lifetime — a later test asking for a
    # different device (typically ``cpu=True`` on macOS/MPS workers, set via
    # ``ACCELERATE_USE_CPU=true`` above) then hits ``_check_initialized`` and
    # fails with ``AcceleratorState has already been initialized ...``.
    #
    # Resetting at setup (vs. teardown) is robust to fixtures that swallow
    # exceptions, tests that create accelerators inside ``with`` blocks that
    # raise, and ordering with subdirectory conftests like
    # ``tests/test_algorithms/test_llms/conftest.py`` that already reset on
    # teardown — those will continue to work and just be redundant on the next
    # test's setup. ``.clear()`` is a no-op when state is empty, so this is
    # cheap.
    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()

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


@pytest.fixture(autouse=True)
def skip_cuda_parametrization_when_unavailable(request):
    """Skip CUDA-specific parametrized tests on environments without CUDA."""
    callspec = getattr(request.node, "callspec", None)
    if (
        callspec is not None
        and callspec.params.get("device") == "cuda"
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA parametrization skipped because CUDA is unavailable.")


@pytest.fixture(autouse=True)
def skip_torch_compile_parametrization_when_windows_compiler_unavailable(request):
    """Skip torch.compile parametrized tests on Windows without MSVC compiler."""
    callspec = getattr(request.node, "callspec", None)
    if callspec is None or sys.platform != "win32":
        return

    compile_mode = callspec.params.get("compile_mode")
    if compile_mode is not None and shutil.which("cl") is None:
        pytest.skip(
            "torch.compile parametrization skipped: MSVC compiler (`cl`) is unavailable on Windows runner."
        )


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


@pytest.fixture(scope="session")
def ac_hp_config():
    return HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=20,
            max=200,
            dtype=int,
            grow_factor=1.5,
            shrink_factor=0.75,
        ),
    )


@pytest.fixture(scope="session")
def default_hp_config():
    return HyperparameterConfig(
        lr=RLParameter(min=6.25e-5, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=1,
            max=10,
            dtype=int,
            grow_factor=1.5,
            shrink_factor=0.75,
        ),
    )


@pytest.fixture(scope="session")
def grpo_hp_config():
    return HyperparameterConfig(
        lr=RLParameter(min=0.00001, max=1),
    )


@pytest.fixture(scope="session")
def encoder_mlp_config():
    return {
        "latent_dim": 8,
        "min_latent_dim": 1,
        "encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
        "head_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
    }


@pytest.fixture(scope="session")
def encoder_simba_config():
    return {
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
    return {
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
    return {
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


dist_env = {
    "ACCELERATE_USE_DEEPSPEED": "true",
    "MASTER_ADDR": "localhost",
    "MASTER_PORT": "10999",
    "RANK": "0",
    "LOCAL_RANK": "0",
    "WORLD_SIZE": "1",
    "CUDA_VISIBLE_DEVICES": "0",
}


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def deepspeed_env():

    dynamic_dist_env = dist_env.copy()
    dynamic_dist_env["MASTER_PORT"] = str(get_free_port())
    existing_vars = {}
    for key, value in dynamic_dist_env.items():
        key = key.upper()
        if key in os.environ:
            existing_vars[key] = os.environ[key]
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key in dynamic_dist_env:
            key = key.upper()
            if key in existing_vars:
                # restore previous value
                os.environ[key] = existing_vars[key]
            else:
                os.environ.pop(key, None)
