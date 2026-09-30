# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Identify tests that spawn multiprocessing vector-env workers.

``AsyncPettingZooVecEnv`` and gymnasium ``AsyncVectorEnv`` start subprocesses.
pytest-xdist workers are also processes. On spawn start methods (macOS,
Windows) concurrent tests in that nested tree EOFError / BrokenPipeError the
pipes. ``--dist loadgroup`` plus one ``async_vec`` group keeps them on a
single worker.
"""

ASYNC_VEC_XDIST_GROUP = "async_vec"

# Substrings of pytest nodeids. Keep in sync with tests that construct a live
# async vector env (not mocks, not SyncVectorEnv-only helpers).
ASYNC_VEC_NODEID_PARTS = (
    "test_vector/test_vector.py",
    "test_make_multi_agent_vect_envs",
    "TestMakeVectEnvs",
    "test_make_skill_vect_envs",
    "rgb_vectorized",
    "TestFunctionPreservingTrainerWiring::test_multi_agent",
)


def nodeid_spawns_async_vector_env(nodeid: str) -> bool:
    """Return whether this collected item starts async vector-env subprocesses.

    :param nodeid: Pytest item nodeid
    :return: True when the test must share the ``async_vec`` xdist group
    """
    if any(part in nodeid for part in ASYNC_VEC_NODEID_PARTS):
        return True
    return "test_train/test_trainer.py" in nodeid and (
        "from_manifest" in nodeid or "test_deferred_encoder" in nodeid
    )
