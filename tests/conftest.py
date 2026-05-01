import gc
import os
import shutil
import socket
import sys
import tempfile
from importlib.util import find_spec

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

    - ``vllm`` tests are spread across ``vllm0``..``vllm{N-1}`` xdist groups
      so up to N vLLM tests can run concurrently. ``N`` is sized to be larger
      than the worker count any current runner is likely to produce under
      pyproject's ``-n auto`` setting, so the group count itself is never the
      bottleneck — actual concurrency is governed by the runner's CPU count
      and (implicitly) by GPU memory pressure.

      Background: vLLM's ``determine_available_memory`` profile run asserts
      that GPU free-memory does not increase between the pre- and post-profile
      snapshots. When peer processes on the same GPU (concurrent xdist
      workers, or sibling CI containers sharing one GPU) release memory
      mid-profile, the assertion fires with ``Error in memory profiling.
      Initial free memory ... current free memory ...``.

      The fix: every test factory that constructs a ``VLLMConfig`` sets
      ``kv_cache_memory_bytes`` (a small value, e.g. 32 MiB on the tiny test
      fixture). vLLM's ``determine_available_memory`` returns early when this
      field is set, skipping the assertion entirely. **Without this flag,
      these xdist groups would have to be collapsed back to a single ``vllm``
      group (serial execution).** When adding new vLLM-using tests, make sure
      they go through one of the existing factories — or set the flag
      directly on the ``VLLMConfig`` they construct.
    - ``gpu`` tests are spread across ``gpu0``..``gpu{N-1}`` xdist groups
      sized the same way. These don't initialise real vLLM, so they're safe
      to run alongside each other.
    - ``test_minari_utils``: tests create/delete shared Minari datasets on disk.

    Uses ``tryfirst=True`` so the ``xdist_group`` markers below are attached
    before xdist's own ``pytest_collection_modifyitems`` (in ``xdist/remote.py``)
    reads them and appends ``@group`` suffixes to nodeids for loadgroup
    scheduling.
    """
    # Number of xdist groups for ``vllm``- and ``gpu``-marked tests. Picked to
    # comfortably exceed the worker count ``-n auto`` produces on typical
    # self-hosted runners (≤ 16 cores), so test concurrency is gated by CPU
    # count (and GPU memory) rather than by group count. Increase if runners
    # ever ship with > 16 cores and the vLLM/GPU phases start under-utilising.
    _N_PARALLEL_GROUPS = 16
    vllm_groups = [
        pytest.mark.xdist_group(f"vllm{i}") for i in range(_N_PARALLEL_GROUPS)
    ]
    gpu_groups = [pytest.mark.xdist_group(f"gpu{i}") for i in range(_N_PARALLEL_GROUPS)]
    minari_group = pytest.mark.xdist_group("minari")
    vllm_count = 0
    gpu_count = 0
    for item in items:
        if item.get_closest_marker("vllm"):
            item.add_marker(vllm_groups[vllm_count % len(vllm_groups)])
            vllm_count += 1
        elif item.get_closest_marker("gpu"):
            item.add_marker(gpu_groups[gpu_count % len(gpu_groups)])
            gpu_count += 1
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

    # Same idea, but for DeepSpeed's module-level ``_*_GROUP`` caches in
    # ``deepspeed.utils.groups`` (and its comm backend ``ds_comm.cdb``).
    # ``DeepSpeedEngine.__init__`` reads ``_get_sequence_data_parallel_group()``
    # — which returns ``_SEQUENCE_DATA_PARALLEL_GROUP`` — and calls
    # ``dist.broadcast(..., group=...)`` on it. If a previous test on this
    # xdist worker created an Accelerator+DeepSpeedEngine, populated those
    # caches, and was then GC'd (which tears down its ``ProcessGroup``), the
    # next DeepSpeed init re-reads the now-stale cached ``ProcessGroup``
    # reference and ``dist.get_global_rank`` raises ``Group <...> is not
    # registered, please create group with torch.distributed.new_group API``
    # because the underlying group is no longer in
    # ``torch.distributed._world.pg_group_ranks``.
    #
    # Setting all module-level ``_*_GROUP`` attributes back to ``None`` and
    # clearing ``ds_comm.cdb`` forces the next DeepSpeed init to repopulate
    # both from scratch against the current ``torch.distributed`` state. This
    # mirrors the existing per-vLLM-test cleanup in
    # ``tests/test_algorithms/test_llms/conftest.py`` but applies it to **every**
    # test, so non-vLLM DeepSpeed tests (``test_core_base.py``'s
    # ``TestLLMDeepspeedCheckpointSaveLoad`` etc.) also get a clean slate.
    # Gated on ``find_spec`` so it stays a no-op on Windows where DeepSpeed
    # isn't installed (``deepspeed~=0.17.1; sys_platform != 'win32'`` in
    # ``pyproject.toml``).
    if find_spec("deepspeed") is not None:
        import deepspeed.comm.comm as ds_comm
        import deepspeed.utils.groups as ds_groups

        for attr in dir(ds_groups):
            if attr.startswith("_") and attr.endswith("_GROUP"):
                setattr(ds_groups, attr, None)
        ds_comm.cdb = None

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
