from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from agilerl.utils.ppo_value_head import (
    AutoModelForCausalLMWithValueHead,
    PPOValueHead,
    _pop_value_head_kwargs,
    _resolve_hidden_size,
)

class DummyConfig(PretrainedConfig):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 11, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class DummyPretrainedModel(PreTrainedModel):
    config_class = DummyConfig
    base_model_prefix = "dummy"

    def __init__(
        self,
        config: DummyConfig | None = None,
        *,
        return_hidden_states: bool = True,
        include_loss: bool = True,
        include_past: bool = False,
        hidden_dtype: torch.dtype = torch.float32,
    ) -> None:
        config = config or DummyConfig()
        super().__init__(config)
        hs = config.hidden_size
        vs = config.vocab_size
        self.embed = nn.Embedding(vs, hs)
        self.proj = nn.Linear(hs, hs)
        self.lm_head = nn.Linear(hs, vs)
        self.return_hidden_states = return_hidden_states
        self.include_loss = include_loss
        self.include_past = include_past
        self.hidden_dtype = hidden_dtype
        self.name_or_path = "dummy-model"
        self.generation_config = SimpleNamespace(temperature=1.0)
        self._saved_args = None
        self._saved_kwargs = None
        self._forward_kwargs = None

    def save_pretrained(self, *args, **kwargs):  # noqa: ANN001
        self._saved_args = args
        self._saved_kwargs = kwargs
        return None

    def prepare_inputs_for_generation(self, *args, **kwargs):  # noqa: ANN001
        return {}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        self._forward_kwargs = dict(kwargs)
        del attention_mask, kwargs
        if input_ids is None:
            msg = "input_ids must be provided"
            raise ValueError(msg)
        x = self.embed(input_ids)
        hidden = torch.relu(self.proj(x))
        logits = self.lm_head(hidden)
        hidden_states = (
            (hidden.to(self.hidden_dtype),) if self.return_hidden_states else None
        )
        loss = torch.tensor(0.5) if self.include_loss else None
        past_key_values = ((torch.zeros(1),),) if self.include_past else None
        if not return_dict:
            return (logits,)
        if not output_hidden_states:
            hidden_states = None
        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
        )

    def generate(self, *args, **kwargs):  # noqa: ANN001
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            msg = "input_ids must be provided"
            raise ValueError(msg)
        appended = torch.zeros(
            (input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat((input_ids, appended), dim=1)


class TestResolveHiddenSize:
    def test_prefers_word_embed_proj_dim(self):
        cfg = SimpleNamespace(word_embed_proj_dim=123, hidden_size=8)
        assert _resolve_hidden_size(cfg) == 123

    def test_falls_back_to_hidden_size(self):
        cfg = SimpleNamespace(hidden_size=16)
        assert _resolve_hidden_size(cfg) == 16

    def test_raises_without_hidden_dimensions(self):
        cfg = SimpleNamespace()
        with pytest.raises(ValueError, match="Cannot infer value-head hidden size"):
            _resolve_hidden_size(cfg)


class TestPopValueHeadKwargs:
    def test_extracts_only_value_head_keys(self):
        kwargs = {
            "summary_dropout_prob": 0.1,
            "v_head_initializer_range": 0.2,
            "v_head_init_strategy": "normal",
            "other": 7,
        }
        out = _pop_value_head_kwargs(kwargs)
        assert out == {
            "summary_dropout_prob": 0.1,
            "v_head_initializer_range": 0.2,
            "v_head_init_strategy": "normal",
        }
        assert kwargs == {"other": 7}

    def test_returns_empty_dict_when_no_value_head_keys_present(self):
        kwargs = {"alpha": 1, "beta": 2}
        out = _pop_value_head_kwargs(kwargs)
        assert out == {}
        assert kwargs == {"alpha": 1, "beta": 2}


class TestPPOValueHeadForward:
    def test_returns_projected_values_with_expected_shape(self):
        cfg = SimpleNamespace(hidden_size=4)
        head = PPOValueHead(cfg)
        hidden_states = torch.randn(2, 3, 4)
        out = head(hidden_states)
        assert out.shape == (2, 3, 1)


class TestAutoModelForCausalLMWithValueHeadForward:
    def test_returns_logits_loss_values_contract(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
        logits, loss, values = wrapped(input_ids=input_ids)
        assert logits.shape == (2, 5, model.config.vocab_size)
        assert isinstance(loss, torch.Tensor)
        assert values.shape == (2, 5)

    def test_returns_past_key_values_when_requested(self):
        model = DummyPretrainedModel(include_past=True)
        wrapped = AutoModelForCausalLMWithValueHead(model)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
        out = wrapped(input_ids=input_ids, return_past_key_values=True)
        assert len(out) == 4
        assert out[3] is not None

    def test_raises_when_hidden_states_missing(self):
        model = DummyPretrainedModel(return_hidden_states=False)
        wrapped = AutoModelForCausalLMWithValueHead(model)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
        with pytest.raises(RuntimeError, match="did not return hidden_states"):
            wrapped(input_ids=input_ids)

    def test_casts_hidden_state_dtype_to_value_head_dtype(self):
        model = DummyPretrainedModel(hidden_dtype=torch.float32)
        wrapped = AutoModelForCausalLMWithValueHead(model)
        wrapped.v_head.summary = wrapped.v_head.summary.to(torch.float64)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 3))
        _, _, values = wrapped(input_ids=input_ids)
        assert values.dtype == torch.float64

    def test_accepts_past_key_values_argument(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 3))
        past = ((torch.ones(1),),)
        wrapped(input_ids=input_ids, past_key_values=past)
        assert model._forward_kwargs["past_key_values"] is past


class TestAutoModelForCausalLMWithValueHeadInit:
    def test_accepts_summary_dropout_prob_without_error(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model, summary_dropout_prob=0.1)
        assert isinstance(wrapped.v_head, PPOValueHead)

    def test_initializes_value_head_with_normal_strategy(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(
            model,
            v_head_init_strategy="normal",
            v_head_initializer_range=0.0,
        )
        assert torch.allclose(
            wrapped.v_head.summary.weight,
            torch.zeros_like(wrapped.v_head.summary.weight),
        )
        assert torch.allclose(
            wrapped.v_head.summary.bias,
            torch.zeros_like(wrapped.v_head.summary.bias),
        )


class TestAutoModelForCausalLMWithValueHeadStateDict:
    def test_non_peft_includes_backbone_and_v_head(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        state_dict = wrapped.state_dict()
        assert any(key.startswith("v_head.") for key in state_dict)
        assert any(key.startswith("embed.") for key in state_dict)

    def test_peft_includes_only_v_head_prefix(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        wrapped.is_peft_model = True
        state_dict = wrapped.state_dict()
        assert len(state_dict) > 0
        assert all(key.startswith("v_head.") for key in state_dict)


class TestAutoModelForCausalLMWithValueHeadPostInit:
    def test_loads_v_head_prefixed_weights(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        target_w = torch.full_like(wrapped.v_head.summary.weight, 0.25)
        target_b = torch.full_like(wrapped.v_head.summary.bias, -0.5)
        wrapped.post_init(
            {
                "v_head.summary.weight": target_w,
                "v_head.summary.bias": target_b,
            }
        )
        assert torch.allclose(wrapped.v_head.summary.weight, target_w)
        assert torch.allclose(wrapped.v_head.summary.bias, target_b)


class TestAutoModelForCausalLMWithValueHeadSavePretrained:
    def test_non_peft_forwards_state_dict_to_base_model(self, tmp_path):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        wrapped.save_pretrained(str(tmp_path))
        assert model._saved_args == (str(tmp_path),)
        assert "state_dict" in model._saved_kwargs
        saved_state = model._saved_kwargs["state_dict"]
        assert any(key.startswith("v_head.") for key in saved_state)

    def test_peft_writes_bin_and_omits_state_dict_in_base_call(self, tmp_path):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        wrapped.is_peft_model = True
        wrapped.save_pretrained(str(tmp_path))
        assert (tmp_path / "pytorch_model.bin").exists()
        assert model._saved_args == (str(tmp_path),)
        assert "state_dict" not in model._saved_kwargs

    def test_peft_accepts_save_directory_keyword(self, tmp_path):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        wrapped.is_peft_model = True
        wrapped.save_pretrained(save_directory=str(tmp_path))
        assert (tmp_path / "pytorch_model.bin").exists()
        assert model._saved_args == ()
        assert model._saved_kwargs["save_directory"] == str(tmp_path)
        assert "state_dict" not in model._saved_kwargs

    def test_peft_raises_when_save_directory_missing(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        wrapped.is_peft_model = True
        with pytest.raises(ValueError, match="requires a save directory"):
            wrapped.save_pretrained()


class TestAutoModelForCausalLMWithValueHeadFromPretrained:
    def test_with_model_instance_returns_working_wrapper(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 3))
        logits, loss, values = wrapped(input_ids=input_ids)
        assert logits.shape == (1, 3, model.config.vocab_size)
        assert isinstance(loss, torch.Tensor)
        assert values.shape == (1, 3)

    def test_invalid_input_type_raises_type_error(self):
        with pytest.raises(TypeError, match="must be str or PreTrainedModel"):
            AutoModelForCausalLMWithValueHead.from_pretrained(12345)  # type: ignore[arg-type]

    def test_string_path_applies_resume_v_head_state(self, tmp_path, monkeypatch):
        base_model = DummyPretrainedModel()
        target_w = torch.full((1, base_model.config.hidden_size), 0.33)
        target_b = torch.tensor([-0.25])
        torch.save(
            {
                "v_head.summary.weight": target_w,
                "v_head.summary.bias": target_b,
            },
            tmp_path / "pytorch_model.bin",
        )
        monkeypatch.setattr(
            "agilerl.utils.ppo_value_head.AutoModelForCausalLM.from_pretrained",
            lambda *args, **kwargs: base_model,
        )
        wrapped = AutoModelForCausalLMWithValueHead.from_pretrained(str(tmp_path))
        assert torch.allclose(wrapped.v_head.summary.weight, target_w)
        assert torch.allclose(wrapped.v_head.summary.bias, target_b)

    def test_raises_import_error_when_peft_config_provided_without_peft(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "agilerl.utils.ppo_value_head.AutoModelForCausalLM.from_pretrained",
            lambda *args, **kwargs: DummyPretrainedModel(),
        )
        monkeypatch.setattr("agilerl.utils.ppo_value_head.get_peft_model", None)
        with pytest.raises(ImportError, match="peft is required"):
            AutoModelForCausalLMWithValueHead.from_pretrained(
                str(tmp_path), peft_config=object()
            )

    def test_applies_peft_to_pretrained_model_instance_when_available(
        self, monkeypatch
    ):
        base_model = DummyPretrainedModel()
        wrapped_model = DummyPretrainedModel()
        marker = object()
        captured = {}

        def fake_get_peft_model(model, peft_config):
            captured["model"] = model
            captured["peft_config"] = peft_config
            return wrapped_model

        monkeypatch.setattr(
            "agilerl.utils.ppo_value_head.get_peft_model", fake_get_peft_model
        )
        wrapped = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model, peft_config=marker
        )
        assert captured["model"] is base_model
        assert captured["peft_config"] is marker
        assert wrapped.pretrained_model is wrapped_model

    def test_applies_peft_to_string_path_when_available(self, tmp_path, monkeypatch):
        base_model = DummyPretrainedModel()
        peft_wrapped_model = DummyPretrainedModel()
        marker = object()
        captured = {}

        monkeypatch.setattr(
            "agilerl.utils.ppo_value_head.AutoModelForCausalLM.from_pretrained",
            lambda *args, **kwargs: base_model,
        )

        def fake_get_peft_model(model, peft_config):
            captured["model"] = model
            captured["peft_config"] = peft_config
            return peft_wrapped_model

        monkeypatch.setattr(
            "agilerl.utils.ppo_value_head.get_peft_model", fake_get_peft_model
        )
        wrapped = AutoModelForCausalLMWithValueHead.from_pretrained(
            str(tmp_path), peft_config=marker
        )
        assert captured["model"] is base_model
        assert captured["peft_config"] is marker
        assert wrapped.pretrained_model is peft_wrapped_model


class TestMaybeLoadResumeStateDict:
    def test_returns_none_for_non_directory(self, tmp_path):
        not_a_dir = tmp_path / "not_a_dir"
        out = AutoModelForCausalLMWithValueHead._maybe_load_resume_state_dict(
            str(not_a_dir)
        )
        assert out is None

    def test_loads_bin_checkpoint(self, tmp_path):
        ckpt = {
            "v_head.summary.weight": torch.randn(1, 4),
            "v_head.summary.bias": torch.randn(1),
        }
        torch.save(ckpt, tmp_path / "pytorch_model.bin")
        out = AutoModelForCausalLMWithValueHead._maybe_load_resume_state_dict(
            str(tmp_path)
        )
        assert out is not None
        assert "v_head.summary.weight" in out
        assert "v_head.summary.bias" in out

    def test_prefers_safetensors_over_bin(self, tmp_path):
        safetensors = pytest.importorskip("safetensors.torch")
        safe_w = torch.full((1, 4), 0.77)
        safe_b = torch.tensor([0.5])
        bin_w = torch.full((1, 4), -0.77)
        bin_b = torch.tensor([-0.5])
        safetensors.save_file(
            {
                "v_head.summary.weight": safe_w,
                "v_head.summary.bias": safe_b,
            },
            str(tmp_path / "model.safetensors"),
        )
        torch.save(
            {
                "v_head.summary.weight": bin_w,
                "v_head.summary.bias": bin_b,
            },
            tmp_path / "pytorch_model.bin",
        )
        out = AutoModelForCausalLMWithValueHead._maybe_load_resume_state_dict(
            str(tmp_path)
        )
        assert out is not None
        assert torch.allclose(out["v_head.summary.weight"], safe_w)
        assert torch.allclose(out["v_head.summary.bias"], safe_b)

    def test_returns_none_when_loader_errors(self, tmp_path):
        (tmp_path / "pytorch_model.bin").write_text(
            "not a torch checkpoint", encoding="utf-8"
        )
        out = AutoModelForCausalLMWithValueHead._maybe_load_resume_state_dict(
            str(tmp_path)
        )
        assert out is None

    def test_returns_none_when_directory_has_no_resume_checkpoint(self, tmp_path):
        out = AutoModelForCausalLMWithValueHead._maybe_load_resume_state_dict(
            str(tmp_path)
        )
        assert out is None


class TestAutoModelForCausalLMWithValueHeadNameOrPathProperty:
    def test_property_round_trip(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        wrapped.name_or_path = "updated-path"
        assert wrapped.name_or_path == "updated-path"
        assert model.name_or_path == "updated-path"


class TestAutoModelForCausalLMWithValueHeadGenerationConfigProperty:
    def test_property_round_trip(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        cfg = SimpleNamespace(temperature=0.2)
        wrapped.generation_config = cfg
        assert wrapped.generation_config is cfg
        assert model.generation_config is cfg


class TestAutoModelForCausalLMWithValueHeadDeviceProperty:
    def test_returns_base_model_device(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        assert wrapped.device == model.device


class TestAutoModelForCausalLMWithValueHeadGenerate:
    def test_forwards_generate_to_pretrained_model(self):
        model = DummyPretrainedModel()
        wrapped = AutoModelForCausalLMWithValueHead(model)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
        generated = wrapped.generate(input_ids=input_ids)
        assert generated.shape == (1, 5)
        assert torch.equal(generated[:, :4], input_ids)
        assert torch.equal(generated[:, 4], torch.zeros(1, dtype=input_ids.dtype))
