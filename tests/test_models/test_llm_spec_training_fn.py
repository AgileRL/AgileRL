# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest

from agilerl.algorithms.core import LLMAlgorithm
from agilerl.models.algo import LLMAlgorithmSpec
from agilerl.models.algorithms.dpo import DPOSpec
from agilerl.models.algorithms.sft import SFTSpec


class TestAlgoClass:
    @pytest.mark.parametrize(("spec", "expected"), [(DPOSpec, "DPO"), (SFTSpec, "SFT")])
    def test_resolves_llm_algorithm_subclass(self, spec, expected):
        resolved = spec.algo_class()
        assert resolved.__name__ == expected
        assert issubclass(resolved, LLMAlgorithm)


class TestGetTrainingFn:
    """The spec's declared env type selects the loop; there is no ``multiturn`` flag."""

    @pytest.mark.parametrize("spec", [DPOSpec, SFTSpec])
    def test_dataset_specs_return_the_dataset_loop(self, spec):
        from agilerl.training.llm import train_llm_dataset

        assert spec.get_training_fn() is train_llm_dataset

    def test_rollout_spec_returns_the_rollout_loop(self):
        from agilerl.models.algorithms.grpo import GRPOSpec
        from agilerl.training.llm import train_llm_rollout

        assert GRPOSpec.get_training_fn() is train_llm_rollout

    def test_base_default_not_implemented(self):
        with pytest.raises(NotImplementedError, match="must implement get_training_fn"):
            LLMAlgorithmSpec.get_training_fn()
