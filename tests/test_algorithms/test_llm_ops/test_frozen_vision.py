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

from agilerl.algorithms.core.llm_ops.frozen_vision import (
    VisionFeatureCache,
    install_frozen_vision_no_grad,
)

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


def tiny_llava_peft(
    target_modules: str | list[str], vision_attention_dropout: float = 0.0
) -> PeftModel:
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
            attention_dropout=vision_attention_dropout,
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


def images(count: int) -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(count, 3, 8, 8)


def image_logits(model: PeftModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """Logits of one row holding every image of ``pixel_values``."""
    image_tokens = [IMAGE_TOKEN_ID] * 4 * pixel_values.shape[0]
    input_ids = torch.tensor([[1, *image_tokens, 5, 6, 7]])
    with torch.no_grad():
        return model(input_ids=input_ids, pixel_values=pixel_values).logits


def count_first_layer_calls(model: PeftModel) -> list[int]:
    calls = [0]
    layer = vision_tower(model).encoder.layers[0]
    layer.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    return calls


def cached_llava() -> tuple[
    PeftModel, VisionFeatureCache, torch.Tensor, torch.Tensor, list[int]
]:
    """Llava with a cached frozen tower, two images, their uncached logits and a tower call counter."""
    model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
    cache = VisionFeatureCache()
    install_frozen_vision_no_grad(model, cache)
    pixel_values = images(2)
    expected = image_logits(model, pixel_values)
    return model, cache, pixel_values, expected, count_first_layer_calls(model)


class TestInstallFrozenVisionNoGrad:
    def test_frozen_tower_returns_its_path(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)

        # Act
        path = install_frozen_vision_no_grad(model, VisionFeatureCache())

        # Assert
        assert path == TOWER_PATH

    def test_frozen_tower_outputs_carry_no_autograd(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        install_frozen_vision_no_grad(model, VisionFeatureCache())
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
        install_frozen_vision_no_grad(no_grad_tower, VisionFeatureCache())

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
        install_frozen_vision_no_grad(no_grad_tower, VisionFeatureCache())
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
        install_frozen_vision_no_grad(model, VisionFeatureCache())
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
        cache = VisionFeatureCache()
        install_frozen_vision_no_grad(model, cache)
        wrapped_forward = vision_tower(model).forward

        # Act
        path = install_frozen_vision_no_grad(model, cache)

        # Assert
        assert path == TOWER_PATH
        assert vision_tower(model).forward is wrapped_forward

    def test_tower_with_trainable_param_keeps_autograd(self) -> None:
        # Arrange
        model = tiny_llava_peft(["q_proj", "v_proj"])
        captured = capture_tower_output(model)

        # Act
        path = install_frozen_vision_no_grad(model, VisionFeatureCache())
        model(**batch())

        # Assert
        assert path is None
        assert captured["hidden"].grad_fn is not None

    def test_text_only_model_returns_none(self) -> None:
        # Arrange
        model = LlamaForCausalLM(tiny_text_config())

        # Act
        path = install_frozen_vision_no_grad(model, VisionFeatureCache())

        # Assert
        assert path is None

    def test_plain_module_returns_none(self) -> None:
        # Arrange
        model = nn.Linear(4, 4)

        # Act
        path = install_frozen_vision_no_grad(model, VisionFeatureCache())

        # Assert
        assert path is None


class TestVisionFeatureCache:
    def test_step_reuses_the_tower_output_of_keyed_images(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        cache = VisionFeatureCache()
        install_frozen_vision_no_grad(model, cache)
        pixel_values = images(2)
        expected = image_logits(model, pixel_values)
        calls = count_first_layer_calls(model)

        # Act
        with cache.step(), cache.images(torch.arange(2)):
            first = image_logits(model, pixel_values)
            second = image_logits(model, pixel_values)

        # Assert
        assert calls == [1]
        assert torch.equal(first, expected)
        assert torch.equal(second, expected)

    def test_reordered_keys_return_entries_in_call_order(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        cache = VisionFeatureCache()
        install_frozen_vision_no_grad(model, cache)
        pixel_values = images(2)
        expected = image_logits(model, pixel_values.flip(0))
        calls = count_first_layer_calls(model)

        # Act
        with cache.step():
            with cache.images(torch.tensor([0, 1])):
                image_logits(model, pixel_values)
            with cache.images(torch.tensor([1, 0])):
                reordered = image_logits(model, pixel_values.flip(0))

        # Assert: fp32 tower GEMMs over a reordered batch may round differently
        # in the last bits.
        assert calls == [1]
        assert torch.allclose(reordered, expected, rtol=0.0, atol=1e-6)

    def test_partly_cached_call_runs_the_tower(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        cache = VisionFeatureCache()
        install_frozen_vision_no_grad(model, cache)
        pixel_values = images(3)
        expected = image_logits(model, pixel_values[1:])
        calls = count_first_layer_calls(model)

        # Act
        with cache.step():
            with cache.images(torch.tensor([0, 1])):
                image_logits(model, pixel_values[:2])
            with cache.images(torch.tensor([1, 2])):
                partial_hit = image_logits(model, pixel_values[1:])

        # Assert
        assert calls == [2]
        assert torch.equal(partial_hit, expected)

    def test_train_mode_reuses_eval_mode_output(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS)
        cache = VisionFeatureCache()
        install_frozen_vision_no_grad(model, cache)
        pixel_values = images(2)
        calls = count_first_layer_calls(model)

        # Act
        with cache.step(), cache.images(torch.arange(2)):
            model.eval()
            image_logits(model, pixel_values)
            model.train()
            image_logits(model, pixel_values)

        # Assert
        assert cache.mode_dependent is False
        assert calls == [1]

    def test_dropout_tower_keeps_eval_and_train_outputs_apart(self) -> None:
        # Arrange
        model = tiny_llava_peft(LANGUAGE_LORA_TARGETS, vision_attention_dropout=0.1)
        cache = VisionFeatureCache()
        install_frozen_vision_no_grad(model, cache)
        pixel_values = images(2)
        calls = count_first_layer_calls(model)

        # Act
        with cache.step(), cache.images(torch.arange(2)):
            model.eval()
            image_logits(model, pixel_values)
            model.train()
            image_logits(model, pixel_values)
            image_logits(model, pixel_values)

        # Assert
        assert cache.mode_dependent is True
        assert calls == [2]

    def test_calls_outside_a_step_run_the_tower(self) -> None:
        # Arrange
        model, cache, pixel_values, expected, calls = cached_llava()

        # Act
        with cache.images(torch.arange(2)):
            outputs = [image_logits(model, pixel_values) for _ in range(2)]

        # Assert
        assert calls == [2]
        assert all(torch.equal(output, expected) for output in outputs)

    def test_calls_without_keys_run_the_tower(self) -> None:
        # Arrange
        model, cache, pixel_values, expected, calls = cached_llava()

        # Act
        with cache.step():
            outputs = [image_logits(model, pixel_values) for _ in range(2)]

        # Assert
        assert calls == [2]
        assert all(torch.equal(output, expected) for output in outputs)

    def test_calls_with_one_key_per_two_images_run_the_tower(self) -> None:
        # Arrange
        model, cache, pixel_values, expected, calls = cached_llava()

        # Act
        with cache.step(), cache.images(torch.arange(1)):
            outputs = [image_logits(model, pixel_values) for _ in range(2)]

        # Assert
        assert calls == [2]
        assert all(torch.equal(output, expected) for output in outputs)

    def test_entries_end_with_their_step(self) -> None:
        # Arrange
        model, cache, pixel_values, expected, calls = cached_llava()

        # Act
        outputs = []
        for _ in range(2):
            with cache.step(), cache.images(torch.arange(2)):
                outputs.append(image_logits(model, pixel_values))

        # Assert
        assert calls == [2]
        assert all(torch.equal(output, expected) for output in outputs)

    def test_tower_with_adapter_layers_runs_uncached(self) -> None:
        # Arrange
        model = tiny_llava_peft(["q_proj", "v_proj"])
        for param in vision_tower(model).parameters():
            param.requires_grad_(False)
        cache = VisionFeatureCache()
        path = install_frozen_vision_no_grad(model, cache)
        calls = count_first_layer_calls(model)

        # Act
        with cache.step(), cache.images(torch.arange(2)):
            image_logits(model, images(2))
            image_logits(model, images(2))

        # Assert
        assert path == TOWER_PATH
        assert calls == [2]
