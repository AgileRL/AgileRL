# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for image observation encoding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agilerl.llm_envs.observation import (
    ESCAPED_IMAGE_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    ImageTokens,
    encode_image_training_inputs,
    image_placeholder_logit_bias,
    model_image_token,
    patch_grid_rows,
)

QWEN_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class ChatTemplateProcessor:
    """Processor whose chat template frames user content and writes ``image_text`` per image item."""

    def __init__(self, image_text: str, image_token: str) -> None:
        self.image_text = image_text
        self.image_token = image_token

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        (message,) = messages
        content = message["content"]
        assert isinstance(content, list)
        body = "".join(
            self.image_text if item["type"] == "image" else item["text"]
            for item in content
        )
        return f"<|im_start|>user\n{body}<|im_end|>\n"


class TestImageTokensFromProcessor:
    @pytest.mark.parametrize(
        ("image_text", "image_token"),
        [
            (QWEN_PLACEHOLDER, "<|image_pad|>"),
            ("<|image|>", "<|image|>"),
            (IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER),
        ],
    )
    def test_reads_the_text_one_image_item_renders(
        self, image_text: str, image_token: str
    ) -> None:
        processor = ChatTemplateProcessor(image_text, image_token)

        tokens = ImageTokens.from_processor(processor)

        assert tokens == ImageTokens(placeholder=image_text, image_token=image_token)


class TestImageTokensRender:
    def test_swaps_each_canonical_placeholder_for_the_model_placeholder(self) -> None:
        tokens = ImageTokens(QWEN_PLACEHOLDER, "<|image_pad|>")

        rendered = tokens.render(f"{IMAGE_PLACEHOLDER}\npage\n{IMAGE_PLACEHOLDER}")

        assert rendered == f"{QWEN_PLACEHOLDER}\npage\n{QWEN_PLACEHOLDER}"

    def test_escapes_the_model_image_token_already_in_the_text(self) -> None:
        tokens = ImageTokens(QWEN_PLACEHOLDER, "<|image_pad|>")

        rendered = tokens.render(f"{IMAGE_PLACEHOLDER}\nan <|image_pad|> tag")

        assert rendered == f"{QWEN_PLACEHOLDER}\nan &lt;|image_pad|&gt; tag"
        assert rendered.count("<|image_pad|>") == 1

    def test_model_placeholder_around_the_canonical_image_token(self) -> None:
        tokens = ImageTokens(f"<img>{IMAGE_PLACEHOLDER}</img>", IMAGE_PLACEHOLDER)

        rendered = tokens.render(f"{IMAGE_PLACEHOLDER}\npage\n{IMAGE_PLACEHOLDER}")

        assert rendered == (
            f"<img>{IMAGE_PLACEHOLDER}</img>\npage\n<img>{IMAGE_PLACEHOLDER}</img>"
        )

    def test_canonical_tokens_leave_text_unchanged(self) -> None:
        text = f"{IMAGE_PLACEHOLDER}\na {ESCAPED_IMAGE_PLACEHOLDER} tag"

        assert ImageTokens().render(text) == text


class TestModelImageToken:
    def test_reads_the_tokenizer_image_token(self) -> None:
        tokenizer = SimpleNamespace(image_token="<|image_pad|>")

        assert model_image_token(tokenizer) == "<|image_pad|>"

    def test_tokenizer_without_an_image_token_uses_the_canonical_placeholder(
        self,
    ) -> None:
        assert model_image_token(SimpleNamespace()) == IMAGE_PLACEHOLDER


class TestImagePlaceholderLogitBias:
    def test_suppresses_the_placeholder_id(self) -> None:
        assert image_placeholder_logit_bias({"<image>": 7, "a": 1}, "<image>") == {
            7: -100.0
        }

    def test_suppresses_the_model_image_token(self) -> None:
        vocab = {"<image>": 7, "<|image_pad|>": 9}

        assert image_placeholder_logit_bias(vocab, "<|image_pad|>") == {9: -100.0}

    def test_vocab_without_the_image_token_gets_no_bias(self) -> None:
        assert image_placeholder_logit_bias({"a": 1}, "<image>") is None


class TestEncodeImageTrainingInputs:
    def test_keeps_per_image_pixels_without_a_patch_grid(self) -> None:
        pixel_values = torch.randn(2, 3, 4, 4)

        def processor(**kwargs: object) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([[1, 2]]), "pixel_values": pixel_values}

        input_ids, encoded = encode_image_training_inputs(
            text="<image>", image=[object(), object()], processor=processor
        )

        assert input_ids.tolist() == [[1, 2]]
        assert encoded is pixel_values

    def test_folds_packed_patches_into_one_row_per_image(self) -> None:
        # Arrange
        packed = torch.arange(2 * 2 * 3 * 5, dtype=torch.float32).reshape(12, 5)

        def processor(**kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": packed,
                "image_grid_hw": torch.tensor([[2, 3], [2, 3]]),
            }

        # Act
        _, rows = encode_image_training_inputs(
            text="<image><image>", image=[object(), object()], processor=processor
        )

        # Assert
        assert rows.shape == (2, 2, 3, 5)
        assert torch.equal(rows[1, 0, 0], packed[6])
        assert torch.equal(rows.reshape(-1, 5), packed)

    def test_folds_temporal_grid_patches_into_one_row_per_image(self) -> None:
        # Arrange
        packed = torch.arange(2 * 2 * 3 * 5, dtype=torch.float32).reshape(12, 5)

        def processor(**kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": packed,
                "image_grid_thw": torch.tensor([[1, 2, 3], [1, 2, 3]]),
            }

        # Act
        _, rows = encode_image_training_inputs(
            text="", image=[object(), object()], processor=processor
        )

        # Assert
        assert rows.shape == (2, 2, 3, 5)
        assert torch.equal(rows.reshape(-1, 5), packed)

    def test_multi_frame_temporal_grid_raises(self) -> None:
        def processor(**kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[1]]),
                "pixel_values": torch.zeros(12, 5),
                "image_grid_thw": torch.tensor([[2, 2, 3]]),
            }

        with pytest.raises(ValueError, match="one temporal patch"):
            encode_image_training_inputs(text="", image=object(), processor=processor)

    def test_places_padded_patches_by_position(self) -> None:
        # Arrange
        grid = [(x, y) for y in range(2) for x in range(3)]
        order = [4, 0, 5, 2, 1, 3]
        positions = torch.tensor(
            [[grid[i] for i in order] + [(-1, -1)] * 2] * 2, dtype=torch.long
        )
        patches = torch.randn(2, 8, 4)

        def processor(**kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[1]]),
                "pixel_values": patches,
                "image_position_ids": positions,
            }

        # Act
        _, rows = encode_image_training_inputs(
            text="", image=[object(), object()], processor=processor
        )

        # Assert
        assert rows.shape == (2, 2, 3, 4)
        for slot, index in enumerate(order):
            x, y = grid[index]
            assert torch.equal(rows[:, y, x], patches[:, slot])

    def test_padded_images_of_different_grids_raise(self) -> None:
        positions = torch.tensor(
            [
                [(0, 0), (1, 0), (0, 1), (1, 1)],
                [(0, 0), (1, 0), (2, 0), (-1, -1)],
            ]
        )

        def processor(**kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[1]]),
                "pixel_values": torch.zeros(2, 4, 3),
                "image_position_ids": positions,
            }

        with pytest.raises(ValueError, match="one full grid per call"):
            encode_image_training_inputs(
                text="", image=[object(), object()], processor=processor
            )


class TestPatchGridRows:
    def test_images_with_different_grids_raise(self) -> None:
        packed = torch.zeros(4 + 6, 5)

        with pytest.raises(ValueError, match=r"patch grids \[\(2, 2\), \(2, 3\)\]"):
            patch_grid_rows(packed, torch.tensor([[2, 2], [2, 3]]))
