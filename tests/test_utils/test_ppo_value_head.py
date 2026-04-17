from types import SimpleNamespace

import pytest
import torch
from torch import nn

from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

pytestmark = pytest.mark.llm


class _DummyPretrainedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.name_or_path = "dummy-model"
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace()
        self._saved_args = None
        self._saved_kwargs = None
        self.backbone = nn.Linear(4, 4)

    def save_pretrained(self, *args, **kwargs):  # noqa: ANN001
        self._saved_args = args
        self._saved_kwargs = kwargs
        return None

    def prepare_inputs_for_generation(self, *args, **kwargs):  # noqa: ANN001
        return {}


def test_save_pretrained_non_peft_forwards_state_dict_to_base_model(tmp_path):
    model = _DummyPretrainedModel()
    wrapped = AutoModelForCausalLMWithValueHead(model)

    wrapped.save_pretrained(str(tmp_path))

    assert model._saved_args == (str(tmp_path),)
    assert "state_dict" in model._saved_kwargs
    saved_state = model._saved_kwargs["state_dict"]
    assert any(key.startswith("v_head.") for key in saved_state)


def test_save_pretrained_peft_writes_bin_and_omits_state_dict_in_base_call(tmp_path):
    model = _DummyPretrainedModel()
    wrapped = AutoModelForCausalLMWithValueHead(model)
    wrapped.is_peft_model = True

    wrapped.save_pretrained(str(tmp_path))

    assert (tmp_path / "pytorch_model.bin").exists()
    assert model._saved_args == (str(tmp_path),)
    assert "state_dict" not in model._saved_kwargs
