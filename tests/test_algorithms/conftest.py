import gc
import os

import pytest
import torch
import torch.distributed as dist

from agilerl.utils.distributed import init_distributed, is_distributed


def generate_distributed_mode(mode):
    """Initialise ``torch.distributed`` for LLM algorithm tests.

    :param mode: ``None`` (single device, no process group), ``"dist"``
        (single-process ``torch.distributed`` rendezvous via the
        ``distributed_env`` fixture's env vars; gloo on CPU, nccl on CUDA)
        or ``"fsdp2"`` (same process group at world size 1; tests
        additionally pass ``FSDPConfig()`` to the algorithm).

    Distributed modes need CUDA. Setting ``AGILERL_TEST_CPU_DISTRIBUTED=1``
    lets the ``"dist"`` mode run on a CPU-only machine (gloo backend) for
    local debugging; ``"fsdp2"`` always requires CUDA.

    :return: ``True`` when a process group is active.
    """
    if mode is not None and not torch.cuda.is_available():
        cpu_dist_ok = (
            mode == "dist" and os.environ.get("AGILERL_TEST_CPU_DISTRIBUTED") == "1"
        )
        if not cpu_dist_ok:
            pytest.skip("Distributed LLM tests require CUDA support.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if mode is None:
        return False

    # The ``distributed_env`` fixture probes a free port for MASTER_PORT,
    # but another xdist worker can grab it before init_process_group binds
    # its TCPStore (EADDRINUSE race) — re-probe and retry.
    for attempt in range(5):
        try:
            initialised = init_distributed()
            break
        except RuntimeError as err:
            if "address already in use" not in str(err).lower() or attempt == 4:
                raise
            from tests.conftest import get_free_port

            os.environ["MASTER_PORT"] = str(get_free_port())
    assert initialised, (
        "Distributed mode requires the `distributed_env` fixture to set the "
        "rendezvous env vars (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT)."
    )
    return True


@pytest.fixture
def dist_mode_factory(request):
    """Factory fixture wrapping :func:`generate_distributed_mode`.

    Distributed modes lazily pull in the ``distributed_env`` fixture so the
    rendezvous env vars are only set (and restored) when a process group is
    actually wanted — single-device (``None``) variants must not see them,
    otherwise the algorithm's own ``init_distributed()`` call would create a
    process group. Destroys the process group on teardown when this
    fixture's factory call initialised it, so distributed state never leaks
    to the next test.
    """
    pre_existing = is_distributed()

    def _factory(mode):
        if mode is not None:
            request.getfixturevalue("distributed_env")
        return generate_distributed_mode(mode)

    yield _factory

    if not pre_existing and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
