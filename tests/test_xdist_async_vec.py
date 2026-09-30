# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from tests.xdist_async_vec import nodeid_spawns_async_vector_env


class TestNodeidSpawnsAsyncVectorEnv:
    def test_vector_module_is_grouped(self):
        nodeid = (
            "tests/test_vector/test_vector.py::TestAsyncPettingZooVecEnvStep::test_step"
        )

        assert nodeid_spawns_async_vector_env(nodeid)

    def test_make_multi_agent_vect_envs_is_grouped(self):
        nodeid = (
            "tests/test_utils/test_utils.py::"
            "test_make_multi_agent_vect_envs_returns_asyncvectorenv_object"
        )

        assert nodeid_spawns_async_vector_env(nodeid)

    def test_make_vect_envs_async_class_is_grouped(self):
        nodeid = "tests/test_utils/test_utils.py::TestMakeVectEnvs::test_with_make_env"

        assert nodeid_spawns_async_vector_env(nodeid)

    def test_rgb_vectorized_train_is_grouped(self):
        nodeid = (
            "tests/test_train/test_train.py::TestTrainMultiAgentOffPolicy::"
            "test_train_multi_agent_off_policy_rgb_vectorized"
        )

        assert nodeid_spawns_async_vector_env(nodeid)

    def test_function_preserving_multi_agent_is_grouped(self):
        nodeid = (
            "tests/test_train/test_train.py::"
            "TestFunctionPreservingTrainerWiring::test_multi_agent_on_policy"
        )

        assert nodeid_spawns_async_vector_env(nodeid)

    def test_local_trainer_from_manifest_is_grouped(self):
        nodeid = (
            "tests/test_train/test_trainer.py::"
            "test_from_manifest_infers_multiinput_when_arch_absent"
        )

        assert nodeid_spawns_async_vector_env(nodeid)

    def test_deferred_encoder_is_grouped(self):
        nodeid = (
            "tests/test_train/test_trainer.py::"
            "test_deferred_encoder_uses_spec_defaults_not_dataclass[cnn]"
        )

        assert nodeid_spawns_async_vector_env(nodeid)

    def test_unrelated_cpu_test_is_not_grouped(self):
        nodeid = "tests/test_algorithms/test_dqn.py::test_learn"

        assert not nodeid_spawns_async_vector_env(nodeid)
