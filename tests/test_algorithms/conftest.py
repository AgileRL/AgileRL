# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import gc
from importlib.util import find_spec

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin


def _reset_deepspeed_comm() -> None:
    """Clear DeepSpeed's cached comm backend + group handles.

    DeepSpeed caches its comm backend (``deepspeed.comm.cdb``) and cloned
    process groups (``deepspeed.utils.groups._*_GROUP``) in module globals;
    accelerate's ``AcceleratorState._reset_state`` does not touch them. Nulling
    them forces the next ``deepspeed.initialize`` to rebuild distributed from
    scratch. Only call this once the world group has actually been torn down
    (see :func:`generate_accelerator`): while the group is alive DeepSpeed's
    cache is valid, and clearing it would re-clone the group every build,
    leaking NCCL communicators and OOM-ing concurrent GPU workers.
    """
    try:
        import deepspeed.comm.comm as ds_comm
        from deepspeed.utils import groups
    except ImportError:
        return
    ds_comm.cdb = None
    for name in dir(groups):
        if name.endswith("GROUP"):
            setattr(groups, name, None)


def generate_accelerator(use_deepspeed_optimizer, config):
    if config is not None and not torch.cuda.is_available():
        pytest.skip("DeepSpeed-configured LLM tests require CUDA support.")
    if config is not None and find_spec("deepspeed") is None:
        pytest.skip("DeepSpeed-configured LLM tests require deepspeed.")

    gc.collect()
    torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)
    # Only when a prior test tore down the world group (so DeepSpeed's cached
    # clones now dangle) do we clear its caches; otherwise reuse them, since
    # re-cloning every build leaks NCCL communicators (OOM under concurrent
    # xdist GPU workers). See ``_reset_deepspeed_comm``.
    if (
        config is not None
        and torch.distributed.is_available()
        and not torch.distributed.is_initialized()
    ):
        _reset_deepspeed_comm()
    if use_deepspeed_optimizer and (config is not None):
        config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,  # Smaller learning rate
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        }
    return (
        Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config))
        if config is not None
        else None
    )


@pytest.fixture
def accelerator_factory():
    return generate_accelerator
