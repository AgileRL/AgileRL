# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""RolloutHarness vision observation wiring."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agilerl.llm_envs.collector import RolloutCollector
from agilerl.llm_envs.harness import RolloutHarness, TranscriptContinuityError
from agilerl.llm_envs.observation import (
    ESCAPED_IMAGE_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    IMAGE_USER_CONTENT_PREFIX,
    ImageProcessorCall,
    encode_image_training_inputs,
)
from tests import TINY_LLM_FIXTURE_PATH
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


class ImageTurnClient(ImageResetClient):
    def step(self, action: Any) -> tuple[object, float, bool, bool, dict[str, Any]]:
        del action
        self.step_calls += 1
        return {"text": "next screen", "image": object()}, 0.0, False, False, {}


class ActionTextTokenizer(MiniTokenizer):
    def decode(self, ids: Any, **_: Any) -> str:
        del ids
        return "act"

    def encode(self, text: str, **_: Any) -> list[int]:
        if text == "act":
            return [7, 8]
        return [1]


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
        collector.reset_episode(episode_id, task=collector.assign_group_task(0))
        completion_ids = torch.arange(14, dtype=torch.long).unsqueeze(0)

        # Act
        collector.step_episode(episode_id, completion_ids, prompt_token_len=10)

        # Assert
        harness = collector.envs[0]
        assert harness.full_ids is not None
        assert harness.full_ids.tolist() == [[1, 2, 3, 4, 10, 11, 12, 13]]
        assert harness.turn_boundaries[0] == (4, 8, 0)

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
            harness._step_apply(
                ([{"role": "user", "content": "next"}], None, 0.0, False, False, {})
            )

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

    def test_second_image_turn_keeps_the_first_turn_in_place(self) -> None:
        # Arrange
        prompts = iter([[11, 12, 13, 14], [5, 6], [8]])

        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([next(prompts)]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

        harness = RolloutHarness(
            ImageTurnClient(),
            ActionTextTokenizer(),
            max_turns=3,
            apply_chat_template=False,
            vision_processor=fake_processor,
        )
        harness.reset()

        # Act
        harness.step(torch.tensor([[11, 12, 13, 14, 50, 51, 99]]))
        harness.step(torch.tensor([[11, 12, 13, 14, 50, 51, 99, 5, 6, 60]]))

        # Assert
        assert harness.turn_boundaries == [(4, 7, 0), (9, 10, 1)]
        assert harness.current_prompt["input_ids"].tolist() == [
            [11, 12, 13, 14, 50, 51, 99, 5, 6, 60, 8]
        ]

    def test_second_image_turn_rejects_a_sample_that_drops_the_sampled_turn(
        self,
    ) -> None:
        # Arrange
        prompts = iter([[11, 12, 13, 14], [5, 6]])

        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([next(prompts)]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

        harness = RolloutHarness(
            ImageTurnClient(),
            ActionTextTokenizer(),
            max_turns=3,
            apply_chat_template=False,
            vision_processor=fake_processor,
        )
        harness.reset()
        harness.step(torch.tensor([[11, 12, 13, 14, 50, 51, 99]]))

        # Act / Assert
        with pytest.raises(
            TranscriptContinuityError,
            match="Turn 1 prompt does not extend the sampled transcript: "
            "tokens diverge at position 4 of 7",
        ):
            harness.step(torch.tensor([[11, 12, 13, 14, 7, 8, 2, 5, 6, 60]]))

    def test_training_ids_follow_processor_not_expanded_prompt(self) -> None:
        # Arrange
        def fake_processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([[10, 10, 10]]),
                "pixel_values": torch.ones(2, 3, 2, 2),
            }

        harness = RolloutHarness(
            ImageResetClient(),
            ActionTextTokenizer(),
            max_turns=1,
            apply_chat_template=False,
            vision_processor=fake_processor,
        )
        harness.reset()
        harness._last_full_prompt_token_len = 5

        # Act
        harness.step(torch.tensor([[9, 9, 9, 9, 9, 50, 51]]))
        full_ids, _mask, _turns, _rewards, _logps, pixel_values, _segments = (
            harness.get_episode_data()
        )

        # Assert
        assert full_ids.tolist() == [[10, 10, 10, 50, 51]]
        assert harness.turn_boundaries == [(3, 5, 0)]
        assert pixel_values is not None
        assert pixel_values.shape == (2, 3, 2, 2)


class TextThenImageClient(FakeEnvClient):
    """Text reset, then an image observation on every step."""

    def __init__(self, image: object) -> None:
        super().__init__()
        self.image = image

    def step(self, action: Any) -> tuple[object, float, bool, bool, dict[str, Any]]:
        del action
        self.step_calls += 1
        return {"text": "next screen", "image": self.image}, 0.0, False, False, {}


def two_token_processor(
    *, text: str, images: object, return_tensors: str
) -> dict[str, torch.Tensor]:
    del text, images, return_tensors
    return {
        "input_ids": torch.tensor([[5, 6]]),
        "pixel_values": torch.ones(1, 3, 2, 2),
    }


class TestRolloutHarnessStep:
    def test_first_image_after_a_text_reset_is_sent_alone(self) -> None:
        # Arrange
        image = object()
        harness = RolloutHarness(
            TextThenImageClient(image),
            MiniTokenizer(),
            max_turns=2,
            apply_chat_template=False,
            vision_processor=two_token_processor,
        )
        prompt, _info = harness.reset()

        # Act
        prompt, *_ = harness.step(
            torch.cat([prompt["input_ids"], torch.tensor([[9]])], dim=1)
        )

        # Assert
        assert prompt["image"] is image
        assert prompt["prompt"] == f"go{IMAGE_USER_CONTENT_PREFIX}next screen"
        assert prompt["input_ids"].tolist() == [[1, 2, 3, 9, 5, 6]]

    def test_image_observation_without_a_vision_processor_raises(self) -> None:
        # Arrange
        harness = RolloutHarness(
            TextThenImageClient(object()),
            MiniTokenizer(),
            max_turns=2,
            apply_chat_template=False,
        )
        prompt, _info = harness.reset()

        # Act / Assert
        with pytest.raises(
            RuntimeError,
            match="Image observations require vision_processor on RolloutHarness",
        ):
            harness.step(torch.cat([prompt["input_ids"], torch.tensor([[9]])], dim=1))

    def test_image_turn_without_a_template_feedback_boundary_raises(self) -> None:
        # Arrange
        harness = RolloutHarness(
            ImageTurnClient(),
            ChatTemplateTokenizer(f"{IMAGE_PLACEHOLDER}digit 7"),
            max_turns=2,
            vision_processor=two_token_processor,
        )
        harness.reset()

        # Act / Assert
        with pytest.raises(
            RuntimeError,
            match="could not render a 'user' feedback turn boundary for the image "
            "transcript",
        ):
            harness.step(torch.tensor([[5, 6, 9]]))

    def test_step_without_training_ids_raises(self) -> None:
        # Arrange
        harness = RolloutHarness(
            FakeEnvClient(),
            MiniTokenizer(),
            max_turns=2,
            apply_chat_template=False,
        )
        harness.reset()
        harness.full_ids = None

        # Act / Assert
        with pytest.raises(
            RuntimeError,
            match="step\\(\\) requires a prior reset\\(\\) or step\\(\\) that "
            "built a prompt",
        ):
            harness.step(torch.tensor([[1, 2, 3, 9]]))


# Super VL's chat template, reduced to what a text transcript exercises: a
# prefilled open <think> in the generation prompt, and history reasoning dropped
# before the last user turn.
THINKING_CHAT_TEMPLATE = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined "
    "else true %}"
    "{%- set truncate_history_thinking = truncate_history_thinking "
    "if truncate_history_thinking is defined else true %}"
    "{%- set ns = namespace(last_user_idx=-1) %}"
    "{%- for m in messages %}{%- if m['role'] == 'user' %}"
    "{%- set ns.last_user_idx = loop.index0 %}{%- endif %}{%- endfor %}"
    "{%- for m in messages %}"
    "{%- if m['role'] == 'assistant' %}"
    "{%- set c = m['content'] %}"
    "{%- if '<think>' not in c and '</think>' not in c %}"
    "{%- set c = '<think></think>' ~ c %}{%- endif %}"
    "{%- if truncate_history_thinking and loop.index0 < ns.last_user_idx "
    "and '<think>' in c and '</think>' in c %}"
    "{%- set c = '<think></think>' ~ c.split('</think>')[-1] %}{%- endif %}"
    "{{- '<|im_start|>assistant\\n' ~ (c | trim) ~ '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>' ~ m['role'] ~ '\\n' ~ m['content'] ~ '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{%- if enable_thinking %}{{- '<|im_start|>assistant\\n<think>\\n' }}"
    "{%- else %}{{- '<|im_start|>assistant\\n<think></think>' }}{%- endif %}"
    "{%- endif %}"
)

INSTRUCTION = "Reply with one call."


class ScreenTurnClient(FakeEnvClient):
    """Image env that records every action it is sent."""

    def __init__(self) -> None:
        super().__init__()
        self.actions: list[str] = []

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[object, dict[str, Any]]:
        del seed, row_index
        return {"text": "page 0", "image": object()}, {}

    def step(self, action: Any) -> tuple[object, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        page = {"text": f"page {len(self.actions)}", "image": object()}
        return page, 0.0, False, False, {}


@pytest.fixture(scope="module")
def thinking_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    tokenizer.chat_template = THINKING_CHAT_TEMPLATE
    return tokenizer


def thinking_harness(
    tokenizer: PreTrainedTokenizerBase, client: ScreenTurnClient
) -> RolloutHarness:
    def processor(
        *, text: str, images: object, return_tensors: str
    ) -> dict[str, torch.Tensor]:
        count = len(images) if isinstance(images, list) else 1
        encoded = tokenizer(
            text, add_special_tokens=False, return_tensors=return_tensors
        )
        return {"input_ids": encoded["input_ids"], "pixel_values": torch.zeros(count)}

    return RolloutHarness(
        client,
        tokenizer,
        max_turns=3,
        system_prompt=INSTRUCTION,
        vision_processor=processor,
    )


def vllm_turn(
    prompt: dict[str, Any], sampled: list[int]
) -> tuple[list[int], torch.Tensor]:
    """What vLLM conditions on for ``prompt``, and its ``prompt + sampled`` output."""
    prompt_ids = prompt["prompt_token_ids"][0].tolist()
    return prompt_ids, torch.tensor([prompt_ids + sampled])


def one_char_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """``text`` as one token per character, a split BPE never produces itself."""
    return [
        token
        for char in text
        for token in tokenizer.encode(char, add_special_tokens=False)
    ]


def newline_first_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """A sampled newline token, then ``text``."""
    return [
        *tokenizer.encode("\n", add_special_tokens=False),
        *tokenizer.encode(text, add_special_tokens=False),
    ]


class TestRolloutHarnessThinkingTranscript:
    def test_each_turn_trains_its_sampled_tokens_in_their_sampled_context(
        self, thinking_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = thinking_tokenizer
        client = ScreenTurnClient()
        harness = thinking_harness(tokenizer, client)
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        completions = [
            "The lamp row has three prices.</think>\n\nclick('12')",
            "Now on the item page.</think>\n\nscroll(0, 200)\n",
            "The price is $5.</think>\n\nsend_msg_to_user('$5')",
        ]
        prompt, _info = harness.reset()
        turns: list[tuple[list[int], list[int]]] = []

        # Act
        for completion in completions:
            sampled = [*tokenizer.encode(completion, add_special_tokens=False), end]
            prompt_ids, output = vllm_turn(prompt, sampled)
            turns.append((prompt_ids, sampled))
            prompt, *_ = harness.step(output)
        full_ids, action_mask, turn_ids, *_ = harness.get_episode_data()

        # Assert
        ids = full_ids[0].tolist()
        for turn_idx, (prompt_ids, sampled) in enumerate(turns):
            trained = (action_mask[0] & (turn_ids[0] == turn_idx)).nonzero()
            start, stop = int(trained[0]) + 1, int(trained[-1]) + 2
            assert ids[start:stop] == sampled
            assert ids[:start] == prompt_ids
        assert client.actions == [
            "click('12')",
            "scroll(0, 200)",
            "send_msg_to_user('$5')",
        ]

    def test_instruction_follows_every_page_and_stays_in_history(
        self, thinking_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = thinking_tokenizer
        harness = thinking_harness(tokenizer, ScreenTurnClient())
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = tokenizer.encode("ok</think>noop()", add_special_tokens=False)
        prompt, _info = harness.reset()

        # Act
        for _turn in range(2):
            _prompt_ids, output = vllm_turn(prompt, [*sampled, end])
            prompt, *_ = harness.step(output)

        # Assert
        for page in range(3):
            assert prompt["prompt"].count(f"page {page}\n{INSTRUCTION}") == 1
        assert prompt["prompt"].endswith("<|im_start|>assistant\n<think>\n")

    @pytest.mark.parametrize("split", [one_char_tokens, newline_first_tokens])
    def test_next_prompt_extends_sampled_ids_that_do_not_retokenize(
        self,
        thinking_tokenizer: PreTrainedTokenizerBase,
        split: Callable[[PreTrainedTokenizerBase, str], list[int]],
    ) -> None:
        # Arrange
        tokenizer = thinking_tokenizer
        harness = thinking_harness(tokenizer, ScreenTurnClient())
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*split(tokenizer, "Done.</think>click('12')"), end]
        prompt, _info = harness.reset()
        prompt_ids, output = vllm_turn(prompt, sampled)
        context = prompt_ids + sampled
        retokenized = tokenizer.encode(
            prompt["prompt"] + tokenizer.decode(sampled), add_special_tokens=False
        )
        assert retokenized[: len(context)] != context

        # Act
        next_prompt, *_ = harness.step(output)
        prompt = next_prompt
        for _turn in range(2):
            _prompt_ids, output = vllm_turn(prompt, sampled)
            prompt, *_ = harness.step(output)
        full_ids, action_mask, turn_ids, *_ = harness.get_episode_data()

        # Assert
        assert next_prompt["prompt_token_ids"][0, : len(context)].tolist() == context
        assert next_prompt["input_ids"][0, : len(context)].tolist() == context
        trained = (action_mask[0] & (turn_ids[0] == 0)).nonzero()
        start, stop = int(trained[0]) + 1, int(trained[-1]) + 2
        assert full_ids[0, :stop].tolist() == context
        assert start == len(prompt_ids)

    def test_rejects_a_turn_sampled_from_a_different_transcript(
        self, thinking_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = thinking_tokenizer
        harness = thinking_harness(tokenizer, ScreenTurnClient())
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*tokenizer.encode("ok</think>noop()", add_special_tokens=False), end]
        prompt, _info = harness.reset()
        prompt_ids, output = vllm_turn(prompt, sampled)
        prompt, *_ = harness.step(output)
        _next_prompt_ids, output = vllm_turn(prompt, sampled)
        output[0, len(prompt_ids)] += 1

        # Act / Assert
        with pytest.raises(
            TranscriptContinuityError,
            match=f"Turn 1 prompt does not extend the sampled transcript: "
            f"tokens diverge at position {len(prompt_ids)} of "
            f"{len(prompt_ids) + len(sampled)}",
        ):
            harness.step(output)


class PageTextClient(ScreenTurnClient):
    """Image env whose pages carry a fixed text."""

    def __init__(self, page_text: str) -> None:
        super().__init__()
        self.page_text = page_text

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[object, dict[str, Any]]:
        del seed, row_index
        return {"text": self.page_text, "image": object()}, {}

    def step(self, action: Any) -> tuple[object, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        return {"text": self.page_text, "image": object()}, 0.0, False, False, {}


def expanding_processor(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Processor that expands each placeholder the way Super VL's does."""

    def processor(
        *, text: str, images: object, return_tensors: str
    ) -> dict[str, torch.Tensor]:
        images = images if isinstance(images, list) else [images]
        image_num_tokens = torch.full((len(images),), 2)
        pieces = text.split(IMAGE_PLACEHOLDER)
        expanded = pieces[0]
        for index, piece in enumerate(pieces[1:]):
            n_tokens = int(image_num_tokens[index])
            expanded += "<img>" + "." * n_tokens + "</img>" + piece
        encoded = tokenizer(
            expanded, add_special_tokens=False, return_tensors=return_tensors
        )
        return {
            "input_ids": encoded["input_ids"],
            "pixel_values": torch.zeros(len(images)),
        }

    return processor


def vllm_image_turn(prompt: dict[str, Any], sampled: list[int]) -> torch.Tensor:
    """vLLM's ``prompt + sampled`` output, each image expanded as the processor does."""
    return torch.tensor([prompt["input_ids"][0].tolist() + sampled])


class TestRolloutHarnessImagePlaceholder:
    def test_page_text_that_spells_the_placeholder_is_not_an_image(
        self, thinking_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = thinking_tokenizer
        processor = expanding_processor(tokenizer)
        harness = RolloutHarness(
            PageTextClient("a <image> tag"),
            tokenizer,
            max_turns=3,
            system_prompt=INSTRUCTION,
            vision_processor=processor,
        )
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = tokenizer.encode("ok</think>noop()", add_special_tokens=False)
        prompt, _info = harness.reset()
        output = vllm_image_turn(prompt, [*sampled, end])

        # Act
        prompt, _reward, _terminated, truncated, _info = harness.step(output)

        # Assert
        assert not truncated
        assert prompt["prompt"].count(IMAGE_PLACEHOLDER) == 2
        assert prompt["prompt"].count(f"a {ESCAPED_IMAGE_PLACEHOLDER} tag") == 2

    def test_sampled_placeholder_ends_the_episode_with_its_tokens_trained(
        self, thinking_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = thinking_tokenizer
        client = PageTextClient("page")
        processor = expanding_processor(tokenizer)
        harness = RolloutHarness(
            client,
            tokenizer,
            max_turns=3,
            system_prompt=INSTRUCTION,
            vision_processor=processor,
        )
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        completion = "The <image> shows a lamp.</think>click('12')"
        sampled = [*tokenizer.encode(completion, add_special_tokens=False), end]
        prompt, _info = harness.reset()
        output = vllm_image_turn(prompt, sampled)

        # Act
        prompt, _reward, terminated, truncated, _info = harness.step(output)
        full_ids, action_mask, *_ = harness.get_episode_data()

        # Assert
        assert truncated
        assert not terminated
        assert prompt == {}
        assert client.actions == ["click('12')"]
        trained = action_mask[0].nonzero()
        start, stop = int(trained[0]) + 1, int(trained[-1]) + 2
        assert full_ids[0, start:stop].tolist() == sampled


class NumberedScreenClient(ScreenTurnClient):
    """Image env whose page ``n`` image is a tensor filled with ``n``."""

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[object, dict[str, Any]]:
        del seed, row_index
        return {"text": "page 0", "image": torch.full((2,), 0.0)}, {}

    def step(self, action: Any) -> tuple[object, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        page = len(self.actions)
        image = torch.full((2,), float(page))
        return {"text": f"page {page}", "image": image}, 0.0, False, False, {}


@pytest.fixture
def image_token_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    tokenizer.chat_template = THINKING_CHAT_TEMPLATE
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [IMAGE_PLACEHOLDER, "<img>", "</img>"]}
    )
    return tokenizer


def context_token_processor(
    tokenizer: PreTrainedTokenizerBase, tokens_per_image: int
) -> Callable[..., dict[str, torch.Tensor]]:
    """Processor that expands each placeholder into ``tokens_per_image`` context tokens."""

    def processor(
        *, text: str, images: object, return_tensors: str
    ) -> dict[str, torch.Tensor]:
        images = images if isinstance(images, list) else [images]
        expanded = text.replace(
            IMAGE_PLACEHOLDER,
            "<img>" + IMAGE_PLACEHOLDER * tokens_per_image + "</img>",
        )
        encoded = tokenizer(
            expanded, add_special_tokens=False, return_tensors=return_tensors
        )
        return {
            "input_ids": encoded["input_ids"],
            "pixel_values": torch.stack(images),
        }

    return processor


class TestRolloutHarnessImageTokens:
    def test_image_tokens_match_pixel_values_on_every_turn(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        start_id = tokenizer.convert_tokens_to_ids("<img>")
        harness = RolloutHarness(
            NumberedScreenClient(),
            tokenizer,
            max_turns=3,
            system_prompt=INSTRUCTION,
            vision_processor=context_token_processor(tokenizer, tokens_per_image=3),
        )
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok</think>noop()"), end]
        prompt, _info = harness.reset()
        prompts = []

        # Act
        for _turn in range(3):
            prompts.append(prompt)
            prompt, *_ = harness.step(vllm_image_turn(prompt, sampled))

        # Assert
        for page, turn_prompt in enumerate(prompts):
            engine_ids = turn_prompt["prompt_token_ids"][0].tolist()
            train_ids = turn_prompt["input_ids"][0].tolist()
            assert engine_ids.count(image_id) == page + 1
            assert start_id not in engine_ids
            assert train_ids.count(image_id) == 3 * (page + 1)
            assert turn_prompt["prompt_token_len"] == len(train_ids)
        pixel_values = harness.get_episode_data()[5]
        assert pixel_values is not None
        assert pixel_values[:, 0].tolist() == [0.0, 1.0, 2.0]


def numbered_screen_harness(
    tokenizer: PreTrainedTokenizerBase, **harness_kwargs: Any
) -> RolloutHarness:
    """Three-turn screen harness; each image expands to 3 context tokens."""
    return RolloutHarness(
        NumberedScreenClient(),
        tokenizer,
        max_turns=3,
        system_prompt=INSTRUCTION,
        vision_processor=context_token_processor(tokenizer, tokens_per_image=3),
        **harness_kwargs,
    )


def run_image_episode(
    harness: RolloutHarness, sampled: list[int]
) -> list[dict[str, Any]]:
    """Answer every turn with ``sampled``; return the prompts the policy saw."""
    prompt, _info = harness.reset()
    prompts = []
    while not harness.done:
        prompts.append(prompt)
        prompt, *_ = harness.step(
            vllm_image_turn(prompt, sampled), sampling_logps=torch.zeros(len(sampled))
        )
    return prompts


def chat_text_harness(
    tokenizer: PreTrainedTokenizerBase, **harness_kwargs: Any
) -> RolloutHarness:
    """Three-turn text harness rendered through the chat template."""
    return RolloutHarness(
        FakeEnvClient(),
        tokenizer,
        max_turns=3,
        system_prompt=INSTRUCTION,
        **harness_kwargs,
    )


def run_text_episode(
    harness: RolloutHarness, sampled: list[int]
) -> list[dict[str, Any]]:
    """Answer every turn with ``sampled``; return the prompts the policy saw."""
    prompt, _info = harness.reset()
    prompts = []
    while not harness.done:
        prompts.append(prompt)
        prompt, *_ = harness.step(
            torch.cat([prompt["input_ids"], torch.tensor([sampled])], dim=1),
            sampling_logps=torch.zeros(len(sampled)),
        )
    return prompts


def split_segments(
    full_ids: torch.Tensor, token_lengths: torch.Tensor
) -> list[list[int]]:
    """Each segment's ids, cut from the episode row."""
    ids = full_ids[0].tolist()
    ends = list(itertools.accumulate(token_lengths.tolist()))
    return [ids[start:end] for start, end in zip([0, *ends[:-1]], ends, strict=True)]


class TestRolloutHarnessStepSampledImagePlaceholder:
    @pytest.mark.parametrize("turn", [0, 2])
    def test_a_sampled_image_placeholder_fails_the_turn(
        self, image_token_tokenizer: PreTrainedTokenizerBase, turn: int
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok</think>noop()"), end]
        harness = numbered_screen_harness(tokenizer)
        prompt, _info = harness.reset()
        for _ in range(turn):
            prompt, *_ = harness.step(
                vllm_image_turn(prompt, sampled),
                sampling_logps=torch.zeros(len(sampled)),
            )
        bad = [*sampled[:-1], image_id, end]

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=rf"Turn {turn} generated the image placeholder '<image>' "
            rf"\(id {image_id}\) 1 time\(s\)",
        ):
            harness.step(
                vllm_image_turn(prompt, bad), sampling_logps=torch.zeros(len(bad))
            )

    def test_text_harness_ignores_the_placeholder_id(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok"), image_id, end]
        harness = chat_text_harness(tokenizer)
        prompt, _info = harness.reset()

        # Act
        harness.step(
            torch.cat([prompt["input_ids"], torch.tensor([sampled])], dim=1),
            sampling_logps=torch.zeros(len(sampled)),
        )

        # Assert
        assert harness.full_ids is not None
        assert int((harness.full_ids == image_id).sum()) == 1


class TestRolloutHarnessSegmentRestart:
    def test_image_segments_tile_the_row_and_its_pixel_rows(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok</think>noop()"), end]
        system_prefix = tokenizer.encode(
            f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n", add_special_tokens=False
        )
        reference = run_image_episode(numbered_screen_harness(tokenizer), sampled)
        threshold = reference[1]["prompt_token_len"] - 1
        harness = numbered_screen_harness(tokenizer, segment_prompt_tokens=threshold)

        # Act
        prompts = run_image_episode(harness, sampled)
        full_ids, mask, _turns, _rewards, logps, pixel_values, segments = (
            harness.get_episode_data()
        )

        # Assert
        assert segments is not None
        assert segments.pixel_rows is not None
        assert pixel_values is not None
        assert logps is not None
        rows = segments.pixel_rows.tolist()
        assert rows == [1, 1, 1]
        assert int(segments.token_lengths.sum()) == full_ids.shape[1]
        assert sum(rows) == pixel_values.shape[0]
        assert pixel_values[:, 0].tolist() == [0.0, 1.0, 2.0]
        pieces = split_segments(full_ids, segments.token_lengths)
        assert [piece.count(image_id) for piece in pieces] == [3 * r for r in rows]
        assert all(piece[: len(system_prefix)] == system_prefix for piece in pieces)
        segment_starts = segments.token_lengths.cumsum(0)[:-1]
        assert not mask[0, segment_starts - 1].any()
        assert int(mask.sum()) == 3 * len(sampled) == logps.numel()
        assert len(harness.turn_rewards) == 3
        assert all(prompt["prompt_token_len"] <= threshold for prompt in prompts[1:])
        assert (
            "Previous actions:\n1. noop()\n2. noop()\n\npage 2" in prompts[2]["prompt"]
        )
        assert prompts[2]["prompt"].count(IMAGE_PLACEHOLDER) == 1

    def test_text_segments_restart_behind_the_system_prompt(
        self, thinking_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = thinking_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*tokenizer.encode("ok</think>noop()", add_special_tokens=False), end]
        system_prefix = tokenizer.encode(
            f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n", add_special_tokens=False
        )
        reference = run_text_episode(chat_text_harness(tokenizer), sampled)
        threshold = int(reference[1]["input_ids"].shape[1]) - 1
        harness = chat_text_harness(tokenizer, segment_prompt_tokens=threshold)

        # Act
        prompts = run_text_episode(harness, sampled)
        full_ids, mask, _turns, _rewards, logps, pixel_values, segments = (
            harness.get_episode_data()
        )

        # Assert
        assert segments is not None
        assert segments.pixel_rows is None
        assert pixel_values is None
        assert logps is not None
        assert segments.token_lengths.numel() == 3
        assert int(segments.token_lengths.sum()) == full_ids.shape[1]
        pieces = split_segments(full_ids, segments.token_lengths)
        assert all(piece[: len(system_prefix)] == system_prefix for piece in pieces)
        assert all(piece[-len(sampled) :] == sampled for piece in pieces)
        assert int(mask.sum()) == 3 * len(sampled) == logps.numel()
        assert all(p["input_ids"].shape[1] <= threshold for p in prompts[1:])
        restarted = tokenizer.decode(prompts[2]["input_ids"][0])
        assert "Previous actions:\n1. noop()\n2. noop()\n\nfeedback" in restarted

    def test_image_restart_over_budget_continues_while_the_context_fits(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Quoting escapes every ', so the history entry outgrows the sampled turn.
        sampled = [*newline_first_tokens(tokenizer, "say: " + "'" * 60 + "x"), end]
        continued = run_image_episode(numbered_screen_harness(tokenizer), sampled)
        continued_len = continued[1]["prompt_token_len"]
        threshold = continued_len - 1
        restarted = run_image_episode(
            numbered_screen_harness(tokenizer, segment_prompt_tokens=threshold),
            sampled,
        )
        assert restarted[1]["prompt_token_len"] > continued_len
        harness = numbered_screen_harness(
            tokenizer,
            segment_prompt_tokens=threshold,
            max_model_len=continued_len + 1,
        )

        # Act
        prompts = run_image_episode(harness, sampled)
        *_rest, pixel_values, segments = harness.get_episode_data()

        # Assert
        assert [prompt["prompt"] for prompt in prompts] == [
            prompt["prompt"] for prompt in continued[:2]
        ]
        assert harness.done
        assert segments is None
        assert pixel_values is not None
        assert pixel_values[:, 0].tolist() == [0.0, 1.0]

    def test_image_restart_and_continued_prompt_over_budget_truncate(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "say: " + "'" * 60 + "x"), end]
        continued = run_image_episode(numbered_screen_harness(tokenizer), sampled)
        continued_len = continued[1]["prompt_token_len"]
        harness = numbered_screen_harness(
            tokenizer,
            segment_prompt_tokens=continued_len - 1,
            max_model_len=continued_len,
        )

        # Act
        prompts = run_image_episode(harness, sampled)
        *_rest, pixel_values, segments = harness.get_episode_data()

        # Assert
        assert len(prompts) == 1
        assert harness.done
        assert segments is None
        assert pixel_values is not None
        assert pixel_values[:, 0].tolist() == [0.0]

    @pytest.mark.parametrize(
        ("segment_max_images", "expected_rows"), [(2, [2, 1]), (1, [1, 1, 1])]
    )
    def test_image_limit_restarts_before_a_prompt_exceeds_it(
        self,
        image_token_tokenizer: PreTrainedTokenizerBase,
        segment_max_images: int,
        expected_rows: list[int],
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok</think>noop()"), end]
        harness = numbered_screen_harness(
            tokenizer, segment_max_images=segment_max_images
        )

        # Act
        prompts = run_image_episode(harness, sampled)
        *_rest, pixel_values, segments = harness.get_episode_data()

        # Assert
        assert segments is not None
        assert segments.pixel_rows is not None
        assert segments.pixel_rows.tolist() == expected_rows
        assert pixel_values is not None
        assert pixel_values[:, 0].tolist() == [0.0, 1.0, 2.0]
        assert all(
            prompt["prompt"].count(IMAGE_PLACEHOLDER) <= segment_max_images
            for prompt in prompts
        )
        assert prompts[2]["prompt"].count(IMAGE_PLACEHOLDER) == 1
        assert (
            "Previous actions:\n1. noop()\n2. noop()\n\npage 2"
            in (prompts[2]["prompt"])
        )

    def test_restart_keeps_the_browsergym_goal(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok</think>noop()"), end]
        goal = "Find the red car."
        screen_processor = context_token_processor(tokenizer, tokens_per_image=3)

        class GoalScreenClient(ScreenTurnClient):
            def reset(
                self, seed: int | None = None, *, row_index: int | None = None
            ) -> tuple[object, dict[str, Any]]:
                del seed, row_index
                return {"goal": goal, "text": "page 0", "screenshot": [[0]]}, {}

            def step(
                self, action: Any
            ) -> tuple[object, float, bool, bool, dict[str, Any]]:
                self.actions.append(action)
                page = len(self.actions)
                obs = {"goal": goal, "text": f"page {page}", "screenshot": [[page]]}
                return obs, 0.0, False, False, {}

        def processor(
            *, text: str, images: object, return_tensors: str
        ) -> dict[str, torch.Tensor]:
            images = images if isinstance(images, list) else [images]
            pixels = [
                torch.tensor(image.getpixel((0, 0)), dtype=torch.float32)
                for image in images
            ]
            return screen_processor(
                text=text, images=pixels, return_tensors=return_tensors
            )

        harness = RolloutHarness(
            GoalScreenClient(),
            tokenizer,
            max_turns=3,
            system_prompt=INSTRUCTION,
            vision_processor=processor,
            segment_max_images=1,
        )

        # Act
        prompts = run_image_episode(harness, sampled)

        # Assert
        assert len(prompts) == 3
        restarted = prompts[2]["prompt"]
        assert f"Previous actions:\n1. noop()\n2. noop()\n\n{goal}\n\npage 2" in (
            restarted
        )
        assert restarted.endswith(
            f"Question:\n{goal}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )

    def test_image_limit_restart_over_budget_raises(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Quoting escapes every ', so the history entry outgrows the sampled turn.
        sampled = [*newline_first_tokens(tokenizer, "say: " + "'" * 60 + "x"), end]
        continued = run_image_episode(numbered_screen_harness(tokenizer), sampled)
        continued_len = continued[1]["prompt_token_len"]
        harness = numbered_screen_harness(
            tokenizer, segment_max_images=1, max_model_len=continued_len + 1
        )
        prompt, _info = harness.reset()

        # Act / Assert
        with pytest.raises(
            RuntimeError,
            match=r"Turn 1 needs a context restart to stay within "
            r"segment_max_images=1, but the restarted prompt is over the prompt budget",
        ):
            harness.step(
                vllm_image_turn(prompt, sampled),
                sampling_logps=torch.zeros(len(sampled)),
            )

    @pytest.mark.parametrize("segment_max_images", [0, -1])
    def test_rejects_a_non_positive_image_limit(
        self, image_token_tokenizer: PreTrainedTokenizerBase, segment_max_images: int
    ) -> None:
        with pytest.raises(
            ValueError,
            match=f"segment_max_images must be a positive int or None, got {segment_max_images}",
        ):
            numbered_screen_harness(
                image_token_tokenizer, segment_max_images=segment_max_images
            )

    @pytest.mark.parametrize("segment_prompt_tokens", [None, 1_000_000])
    def test_without_a_reachable_threshold_matches_an_unsegmented_run(
        self,
        image_token_tokenizer: PreTrainedTokenizerBase,
        segment_prompt_tokens: int | None,
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok</think>noop()"), end]
        baseline = numbered_screen_harness(tokenizer)
        harness = numbered_screen_harness(
            tokenizer, segment_prompt_tokens=segment_prompt_tokens
        )

        # Act
        run_image_episode(baseline, sampled)
        run_image_episode(harness, sampled)
        expected = baseline.get_episode_data()
        got = harness.get_episode_data()

        # Assert
        assert expected[6] is None
        assert got[6] is None
        for want, have in zip(expected[:6], got[:6], strict=True):
            assert torch.equal(have, want)


def rebuilt_pixel_values(
    calls: list[ImageProcessorCall],
    processor: Callable[..., dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Rerun each recorded call through the processor, rows in call order."""
    return torch.cat(
        [
            encode_image_training_inputs(
                text=call.text, image=list(call.images), processor=processor
            )[1]
            for call in calls
        ]
    )


class TestRolloutHarnessEpisodeImageCalls:
    def test_calls_rebuild_the_pixel_values_of_every_turn(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok</think>noop()"), end]
        harness = numbered_screen_harness(tokenizer)
        processor = context_token_processor(tokenizer, tokens_per_image=3)
        run_image_episode(harness, sampled)

        # Act
        calls = harness.episode_image_calls()

        # Assert
        pixel_values = harness.get_episode_data()[5]
        assert pixel_values is not None
        assert [len(call.images) for call in calls] == [1, 1, 1]
        assert torch.equal(rebuilt_pixel_values(calls, processor), pixel_values)

    def test_calls_rebuild_the_pixel_rows_across_restarted_segments(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok</think>noop()"), end]
        reference = run_image_episode(numbered_screen_harness(tokenizer), sampled)
        harness = numbered_screen_harness(
            tokenizer, segment_prompt_tokens=reference[1]["prompt_token_len"] - 1
        )
        processor = context_token_processor(tokenizer, tokens_per_image=3)
        run_image_episode(harness, sampled)

        # Act
        calls = harness.episode_image_calls()

        # Assert
        _ids, _mask, _turns, _rewards, _logps, pixel_values, segments = (
            harness.get_episode_data()
        )
        assert segments is not None
        assert segments.pixel_rows is not None
        assert pixel_values is not None
        assert segments.pixel_rows.tolist() == [1, 1, 1]
        assert all("Previous actions:" in call.text for call in calls[1:])
        assert torch.equal(rebuilt_pixel_values(calls, processor), pixel_values)

    def test_reset_starts_a_new_call_list(
        self, image_token_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        tokenizer = image_token_tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(tokenizer, "ok</think>noop()"), end]
        harness = numbered_screen_harness(tokenizer)
        run_image_episode(harness, sampled)

        # Act
        harness.reset()

        # Assert
        calls = harness.episode_image_calls()
        assert len(calls) == 1
        assert torch.equal(calls[0].images[0], torch.full((2,), 0.0))

    def test_text_episode_has_no_calls(
        self, thinking_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        harness = chat_text_harness(thinking_tokenizer)
        end = thinking_tokenizer.convert_tokens_to_ids("<|im_end|>")
        sampled = [*newline_first_tokens(thinking_tokenizer, "ok"), end]

        # Act
        run_text_episode(harness, sampled)

        # Assert
        assert harness.episode_image_calls() == []


class TestImageProcessorCallFromInputs:
    def test_wraps_one_image(self) -> None:
        image = torch.zeros(2)

        call = ImageProcessorCall.from_inputs(text="t", image=image)

        assert call.text == "t"
        assert call.images == (image,)

    def test_keeps_every_image_of_a_list(self) -> None:
        first, second = torch.zeros(2), torch.ones(2)

        call = ImageProcessorCall.from_inputs(text="t", image=[first, second])

        assert call.images == (first, second)


class TestRolloutCollectorEpisodeImageCalls:
    def test_returns_the_active_episode_calls_until_finalize(self) -> None:
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
        collector.reset_episode(episode_id, task=collector.assign_group_task(0))
        collector.step_episode(
            episode_id,
            torch.arange(14, dtype=torch.long).unsqueeze(0),
            prompt_token_len=10,
        )

        # Act
        calls = collector.episode_image_calls(episode_id)
        collector.finalize_episode(episode_id)

        # Assert
        assert len(calls) == 1
        assert len(calls[0].images) == 1
        with pytest.raises(KeyError, match="ep-vision"):
            collector.episode_image_calls(episode_id)
