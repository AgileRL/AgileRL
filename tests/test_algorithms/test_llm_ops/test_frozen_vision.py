# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import (
    CLIPVisionConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlavaConfig,
    LlavaForConditionalGeneration,
)

from agilerl.algorithms.core.llm_ops.frozen_vision import install_frozen_vision_no_grad

IMAGE_TOKEN_ID = 63
LANGUAGE_LORA_TARGETS = r".*language_model.*\.(q_proj|v_proj)"
TOWER_PATH = "base_model.model.model.vision_tower"


def tiny_text_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
    )


def tiny_llava_peft(target_modules: str | list[str]) -> PeftModel:
    """Tiny Llava with HF gradient checkpointing and random-init LoRA, in train mode."""
    torch.manual_seed(0)
    config = LlavaConfig(
        vision_config=CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
            projection_dim=16,
        ),
        text_config=tiny_text_config(),
        image_token_id=IMAGE_TOKEN_ID,
        image_seq_length=4,
        vision_feature_layer=-1,
    )
    model = LlavaForConditionalGeneration(config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    peft_model = get_peft_model(
        model,
        LoraConfig(r=2, target_modules=target_modules, init_lora_weights=False),
    )
    peft_model.train()
    return peft_model


def vision_tower(model: PeftModel) -> nn.Module:
    return model.get_base_model().get_encoder(modality="image")


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    image_tokens = [IMAGE_TOKEN_ID] * 4
    return {
        "input_ids": torch.tensor([[1, *image_tokens, 5, 6, 7]]),
        "pixel_values": torch.randn(1, 3, 8, 8),
    }


def loss_and_grads(model: PeftModel) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = model(**batch()).logits
    loss = logits.float().pow(2).mean()
    loss.backward()
    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    return loss.detach(), grads


def capture_tower_output(model: PeftModel) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    vision_tower(model).register_forward_hook(
        lambda _module, _args, output: captured.update(hidden=output.last_hidden_state)
    )
    return captured


def count_first_layer_calls(model: PeftModel) -> list[int]:
    calls = [0]
    layer = vision_tower(model).encoder.layers[0]
    layer.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    return calls


class TestInstallFrozenVisionNoGrad:
    def test_frozen_tower_returns_its_path(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)

        # Act
        path = install_frozen_vision_no_grad(model)

        # Assert
        assert path == TOWER_PATH

    def test_frozen_tower_outputs_carry_no_autograd(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        install_frozen_vision_no_grad(model)
        captured = capture_tower_output(model)

        # Act
        model(**batch())

        # Assert
        assert captured["hidden"].grad_fn is None
        assert captured["hidden"].requires_grad is False

    def test_grads_match_the_autograd_tower(self) -> None:
        # Arrange
        reference = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        no_grad_tower = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        install_frozen_vision_no_grad(no_grad_tower)

        # Act
        reference_loss, reference_grads = loss_and_grads(reference)
        loss, grads = loss_and_grads(no_grad_tower)

        # Assert
        assert torch.equal(loss, reference_loss)
        assert grads.keys() == reference_grads.keys()
        assert len(grads) == 8
        for name, grad in grads.items():
            assert torch.equal(grad, reference_grads[name]), name

    def test_checkpointed_tower_is_not_recomputed(self) -> None:
        # Arrange
        reference = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        no_grad_tower = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        install_frozen_vision_no_grad(no_grad_tower)
        reference_calls = count_first_layer_calls(reference)
        calls = count_first_layer_calls(no_grad_tower)

        # Act
        loss_and_grads(reference)
        loss_and_grads(no_grad_tower)

        # Assert
        assert reference_calls == [2]
        assert calls == [1]

    def test_deepcopy_runs_the_copied_tower(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        install_frozen_vision_no_grad(model)
        clone = copy.deepcopy(model)
        with torch.no_grad():
            for param in vision_tower(clone).parameters():
                param.zero_()
        original = capture_tower_output(model)
        copied = capture_tower_output(clone)

        # Act
        model(**batch())
        clone(**batch())

        # Assert
        assert not torch.equal(original["hidden"], copied["hidden"])
        assert copied["hidden"].grad_fn is None

    def test_install_twice_keeps_one_wrapper(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        install_frozen_vision_no_grad(model)
        wrapped_forward = vision_tower(model).forward

        # Act
        path = install_frozen_vision_no_grad(model)

        # Assert
        assert path == TOWER_PATH
        assert vision_tower(model).forward is wrapped_forward

    def test_tower_with_trainable_param_keeps_autograd(self) -> None:
        # Arrange
        model = tiny_llava_peft(["q_proj", "v_proj"])
        captured = capture_tower_output(model)

        # Act
        path = install_frozen_vision_no_grad(model)
        model(**batch())

        # Assert
        assert path is None
        assert captured["hidden"].grad_fn is not None

    def test_text_only_model_returns_none(self) -> None:
        # Arrange
        model = LlamaForCausalLM(tiny_text_config())

        # Act
        path = install_frozen_vision_no_grad(model)

        # Assert
        assert path is None

    def test_plain_module_returns_none(self) -> None:
        # Arrange
        model = nn.Linear(4, 4)

        # Act
        path = install_frozen_vision_no_grad(model)

        # Assert
        assert path is None
