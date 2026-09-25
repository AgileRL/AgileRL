# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""RolloutHarness vision observation wiring."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from agilerl.llm_envs.collector import RolloutCollector
from agilerl.llm_envs.harness import RolloutHarness
from agilerl.llm_envs.observation import IMAGE_USER_CONTENT_PREFIX
from tests.helpers.rollout_doubles import FakeEnvClient, MiniTokenizer


class ChatTemplateTokenizer(MiniTokenizer):
    """Tokenizer whose chat template returns a fixed render."""

    def __init__(self, rendered: object) -> None:
        self.rendered = rendered

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> object:
        del messages, add_generation_prompt, kwargs
        if tokenize:
            return [1, 2, 3]
        return self.rendered


class NoneInputIdsTokenizer(MiniTokenizer):
    def __call__(self, texts: list[str], **_: Any) -> dict[str, None]:
        del texts
        return {"input_ids": None, "attention_mask": None}


class NonStrDecodeTokenizer(MiniTokenizer):
    def decode(self, ids: Any, **_: Any) -> int:
        del ids
        return 0


class ImageResetClient(FakeEnvClient):
    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[object, dict[str, Any]]:
        del seed, row_index
        self.reset_calls += 1
        self._episode_steps = 0
        return {"text": "digit 7", "image": object()}, {}


class TestRolloutHarnessVision:
    def test_string_observation_uses_token_ids_prompt_only(self) -> None:
        # Arrange
        harness = RolloutHarness(
            FakeEnvClient(),
            MiniTokenizer(),
            max_turns=1,
            apply_chat_template=False,
        )

        # Act
        prompt, _info = harness.reset()

        # Assert
        assert "input_ids" in prompt
        assert "image" not in prompt
        assert "prompt" not in prompt
        assert harness._episode_pixel_values is None

    def test_image_observation_keeps_multimodal_prompt(self) -> None:
        # Arrange
        encoded_text: list[str] = []

        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del images, return_tensors
            encoded_text.append(text)
            return {
                "input_ids": torch.tensor([[11, 12, 13, 14]]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

        harness = RolloutHarness(
            ImageResetClient(),
            MiniTokenizer(),
            max_turns=1,
            apply_chat_template=False,
            vision_processor=fake_processor,
        )

        # Act
        prompt, _info = harness.reset()

        # Assert
        assert "input_ids" in prompt
        assert torch.equal(prompt["input_ids"], torch.tensor([[11, 12, 13, 14]]))
        assert IMAGE_USER_CONTENT_PREFIX in prompt["prompt"]
        assert "digit 7" in prompt["prompt"]
        assert prompt["image"] is not None
        assert prompt["prompt_token_len"] == 4
        assert encoded_text == [prompt["prompt"]]
        assert torch.equal(
            prompt["pixel_values"],
            torch.ones(1, 3, 2, 2),
        )

    def test_get_episode_data_before_reset_raises_never_called(self) -> None:
        # Arrange
        harness = RolloutHarness(
            ImageResetClient(),
            MiniTokenizer(),
            max_turns=1,
            apply_chat_template=False,
        )

        # Act / Assert
        with pytest.raises(
            RuntimeError, match="No episode data: reset\\(\\) was never called"
        ):
            harness.get_episode_data()

    def test_get_episode_data_after_image_reset_before_step_raises_no_completion(
        self,
    ) -> None:
        # Arrange
        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

        harness = RolloutHarness(
            ImageResetClient(),
            MiniTokenizer(),
            max_turns=1,
            apply_chat_template=False,
            vision_processor=fake_processor,
        )
        harness.reset()

        # Act / Assert
        with pytest.raises(
            RuntimeError, match="No episode data: episode has no completion yet"
        ):
            harness.get_episode_data()

    def test_step_episode_uses_vllm_prompt_token_len_for_image_turn(self) -> None:
        # Arrange
        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

        def env_factory() -> RolloutHarness:
            return RolloutHarness(
                ImageResetClient(),
                MiniTokenizer(),
                max_turns=1,
                apply_chat_template=False,
                vision_processor=fake_processor,
            )

        collector = RolloutCollector(env_factory, batch_size=1, group_size=1)
        episode_id = "ep-vision"
        collector.reset_episode(episode_id, 0, seed=0)
        completion_ids = torch.arange(14, dtype=torch.long).unsqueeze(0)

        # Act
        collector.step_episode(episode_id, completion_ids, prompt_token_len=10)

        # Assert
        harness = collector.envs[0]
        assert harness.turn_boundaries[0][0] == 10

    def test_image_reset_renders_the_chat_template(self) -> None:
        # Arrange
        seen: list[str] = []

        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del images, return_tensors
            seen.append(text)
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

        harness = RolloutHarness(
            ImageResetClient(),
            ChatTemplateTokenizer("<chat>digit 7"),
            max_turns=1,
            vision_processor=fake_processor,
        )

        # Act
        prompt, _info = harness.reset()

        # Assert
        assert seen == ["<chat>digit 7"]
        assert prompt["prompt"] == "<chat>digit 7"

    def test_image_reset_rejects_non_str_chat_template(self) -> None:
        harness = RolloutHarness(
            ImageResetClient(),
            ChatTemplateTokenizer(5),
            max_turns=1,
            vision_processor=lambda **_kwargs: {},
        )

        with pytest.raises(TypeError, match="must return str"):
            harness.reset()

    def test_empty_image_text_uses_instruction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.llm_envs.harness.observation_text_and_image",
            lambda _payload: ("", object()),
        )
        seen: list[str] = []

        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del images, return_tensors
            seen.append(text)
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

        harness = RolloutHarness(
            ImageResetClient(),
            MiniTokenizer(),
            max_turns=1,
            apply_chat_template=False,
            instruction="look",
            vision_processor=fake_processor,
        )

        harness.reset()

        assert seen == ["look"]

    def test_image_reset_requires_vision_processor(self) -> None:
        harness = RolloutHarness(
            ImageResetClient(),
            MiniTokenizer(),
            max_turns=1,
            apply_chat_template=False,
        )

        with pytest.raises(RuntimeError, match="vision_processor"):
            harness.reset()

    def test_reset_raises_when_tokenize_returns_no_ids(self) -> None:
        harness = RolloutHarness(
            FakeEnvClient(),
            NoneInputIdsTokenizer(),
            max_turns=1,
            apply_chat_template=False,
        )

        with pytest.raises(RuntimeError, match="left no prompt token ids"):
            harness.reset()

    def test_image_step_rejects_non_str_decode(self) -> None:
        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

        harness = RolloutHarness(
            ImageResetClient(),
            NonStrDecodeTokenizer(),
            max_turns=1,
            apply_chat_template=False,
            vision_processor=fake_processor,
        )
        harness.reset()

        with pytest.raises(TypeError, match="decode\\(\\) of one sequence returns str"):
            harness.step(torch.arange(8).unsqueeze(0))

    def test_text_step_rejects_non_str_decode(self) -> None:
        harness = RolloutHarness(
            FakeEnvClient(),
            NonStrDecodeTokenizer(),
            max_turns=1,
            apply_chat_template=False,
        )
        harness.reset()

        with pytest.raises(TypeError, match="decode\\(\\) of one sequence returns str"):
            harness.step(torch.arange(8).unsqueeze(0))

    def test_step_apply_requires_prompt_ids(self) -> None:
        harness = RolloutHarness(
            FakeEnvClient(),
            MiniTokenizer(),
            max_turns=2,
            apply_chat_template=False,
        )
        harness.full_ids = None

        with pytest.raises(RuntimeError, match="reset\\(\\) must run before step"):
            harness._step_apply(("next", "user", 0.0, False, False, {}))

    def test_image_step_records_sampling_logps(self) -> None:
        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

        harness = RolloutHarness(
            ImageResetClient(),
            MiniTokenizer(),
            max_turns=1,
            apply_chat_template=False,
            vision_processor=fake_processor,
        )
        harness.reset()
        logps = torch.tensor([0.1, 0.2])

        harness.step(torch.arange(8).unsqueeze(0), sampling_logps=logps)

        assert harness.sampling_logps == [logps]
