import pytest

from agilerl.models.algo import LLMAlgorithmSpec
from agilerl.models.algorithms.dpo import DPOSpec
from agilerl.models.algorithms.sft import SFTSpec


class TestGetTrainingFn:
    def test_dpo_rejects_multiturn(self):
        with pytest.raises(ValueError, match="DPO does not support multi-turn"):
            DPOSpec.get_training_fn(multiturn=True)

    def test_sft_rejects_multiturn(self):
        with pytest.raises(ValueError, match="SFT does not support multi-turn"):
            SFTSpec.get_training_fn(multiturn=True)

    def test_base_multiturn_raises_value_error(self):
        with pytest.raises(ValueError, match="does not support multi-turn"):
            LLMAlgorithmSpec.get_training_fn(multiturn=True)

    def test_base_default_not_implemented(self):
        with pytest.raises(NotImplementedError, match="must implement get_training_fn"):
            LLMAlgorithmSpec.get_training_fn()
