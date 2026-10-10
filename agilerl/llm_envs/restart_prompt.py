# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Kept-turn types and restart-prompt assembly for a context restart."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import torch
from PIL import Image

from agilerl.llm_envs.observation import (
    IMAGE_USER_CONTENT_PREFIX,
    ImageProcessorCall,
    encode_image_training_inputs,
    goal_images_text,
)


class KeptTurn(NamedTuple):
    """One turn as its segment held it: the sampled ids and the observation after them.

    :param gen_ids: The turn's sampled ids, 1-D.
    :param last_token_id: Last id of the transcript the observation extended.
    :param obs_text: The observation's rendered text.
    :param role: Chat role the observation spoke as.
    :param image: The observation's image, or ``None``.
    :param older_obs_text: Text shown in place of ``obs_text`` once a later
        turn is kept, or ``None`` to show ``obs_text``.
    """

    gen_ids: torch.Tensor
    last_token_id: int
    obs_text: str
    role: str
    image: object | None
    older_obs_text: str | None


class KeptTurnPiece(NamedTuple):
    """A kept turn as a restarted prompt carries it: sampled ids, then the observation.

    ``train_ids`` hold the observation with its image expanded by the processor;
    ``engine_ids`` leave the placeholder unexpanded. Text and image fields are
    empty on a text prompt.
    """

    gen_ids: torch.Tensor
    train_ids: torch.Tensor
    engine_ids: torch.Tensor
    gen_text: str
    feedback_text: str
    image: object | None
    pixel_values: torch.Tensor | None


class RestartPrompt(NamedTuple):
    """The prompt that opens a new segment, and the image state it carries."""

    prompt_ids: torch.Tensor
    multimodal_turn: dict[str, Any] | None
    pixel_values: torch.Tensor | None
    images: list[object]
    image_calls: list[ImageProcessorCall]


class RestartPromptMixin:
    """Restart-prompt builders used by :class:`~agilerl.llm_envs.harness.RolloutHarness`."""

    _kept_turns: deque[KeptTurn]
    _goal_images: list[Image.Image]
    _segment_prompt_tokens: int | None
    _segment_max_images: int | None
    _restart_older_images: bool
    _tokenize_initial_prompt: Callable[[str], torch.Tensor]
    _chat_prompt_string: Callable[[str], str]
    _engine_ids: Callable[[str], torch.Tensor]
    _prompt_limit: Callable[[], int | None]
    _text_feedback_ids: Callable[[str, str, int], torch.Tensor]
    _decode: Callable[..., str]
    _image_feedback_turn_text: Callable[[str, str, int], str]
    _require_vision_processor: Callable[[], Callable[..., Mapping[str, Any]]]

    def _action_list_prompt(
        self, actions: str, obs_text: str, image: object | None
    ) -> RestartPrompt | None:
        """Restart prompt of the action list then the latest observation, in one user turn.

        :param image: The observation's image or images in placeholder order, or ``None``.
        :return: ``None`` when the prompt is over the prompt limit.
        """
        body = obs_text.removeprefix(IMAGE_USER_CONTENT_PREFIX)
        restart_text = (
            f"{obs_text[: len(obs_text) - len(body)]}"
            f"Previous actions:\n{actions}\n\n{body}"
        )
        if image is None:
            prompt_ids = self._tokenize_initial_prompt(restart_text)
            restart = RestartPrompt(prompt_ids, None, None, [], [])
        else:
            prompt_str = self._chat_prompt_string(restart_text)
            prompt_ids, pixel_values = encode_image_training_inputs(
                text=prompt_str,
                image=image,
                processor=self._require_vision_processor(),
            )
            multimodal_turn = {
                "prompt": prompt_str,
                "image": image,
                "prompt_token_len": int(prompt_ids.shape[-1]),
                "input_ids": prompt_ids,
                "prompt_token_ids": self._engine_ids(prompt_str),
            }
            restart = RestartPrompt(
                prompt_ids,
                multimodal_turn,
                pixel_values,
                list(image) if isinstance(image, list) else [image],
                [ImageProcessorCall.from_inputs(text=prompt_str, image=image)],
            )
        max_pt = self._prompt_limit()
        if max_pt is not None and int(prompt_ids.shape[-1]) > max_pt:
            return None
        return restart

    def _kept_turns_prompt(
        self, actions: str, image: object | None
    ) -> RestartPrompt | None:
        """Restart prompt of the action list, then the kept turns as they were sampled.

        The goal images follow the action list. The last kept turn's
        observation is the latest one; earlier ones show their
        ``older_obs_text`` when set, and their images only with
        ``_restart_older_images``. Kept images are taken newest first, at
        most ``segment_max_images - 1`` with the goal images, so the next
        turn's image fits. The oldest turns are dropped until the prompt
        plus one more turn as long as the latest fits ``segment_prompt_tokens``
        and the model budget, so the next turn does not restart again.

        :return: ``None`` when no turns are kept or the latest does not fit.
        """
        if not self._kept_turns:
            return None
        head_text = f"Previous actions:\n{actions}"
        goal_images = self._goal_images
        image_path = image is not None or bool(goal_images)
        head_pixel_values = None
        if not image_path:
            head_str = ""
            head_ids = head_engine_ids = self._tokenize_initial_prompt(head_text)
        elif goal_images:
            head_str = self._chat_prompt_string(
                head_text + goal_images_text(len(goal_images))
            )
            head_ids, head_pixel_values = encode_image_training_inputs(
                text=head_str,
                image=list(goal_images),
                processor=self._require_vision_processor(),
            )
            head_engine_ids = self._engine_ids(head_str)
        else:
            head_str = self._chat_prompt_string(head_text)
            head_ids = head_engine_ids = self._engine_ids(head_str)
        token_limit = self._prompt_limit()
        # The latest observation keeps its image even at segment_max_images=1.
        images_left = (
            None
            if self._segment_max_images is None
            else max(self._segment_max_images - 1 - len(goal_images), 1)
        )
        pieces = self._select_kept_turn_pieces(
            image_path=image_path,
            prompt_len=int(head_ids.shape[1]),
            token_limit=token_limit,
            images_left=images_left,
        )
        if not pieces:
            return None
        if not image_path:
            return self._restart_prompt_from_kept_pieces(head_ids, head_ids, pieces)
        return self._restart_prompt_from_kept_pieces(
            head_ids, head_engine_ids, pieces, head_str, head_pixel_values
        )

    def _select_kept_turn_pieces(
        self,
        image_path: bool,
        prompt_len: int,
        token_limit: int | None,
        images_left: int | None,
    ) -> list[KeptTurnPiece]:
        """Newest kept turns that fit the token and image budgets, oldest first."""
        pieces: list[KeptTurnPiece] = []
        for turn in reversed(self._kept_turns):
            older = bool(pieces)
            piece = self._kept_turn_piece(
                turn,
                image_path=image_path,
                with_image=(images_left is None or images_left > 0)
                and (not older or self._restart_older_images),
                older=older,
            )
            piece_len = int(piece.gen_ids.shape[1] + piece.train_ids.shape[1])
            if token_limit is not None and not pieces:
                token_limit -= piece_len
            if token_limit is not None and prompt_len + piece_len > token_limit:
                break
            prompt_len += piece_len
            pieces.append(piece)
            if piece.image is not None and images_left is not None:
                images_left -= 1
        pieces.reverse()
        return pieces

    def _restart_prompt_from_kept_pieces(
        self,
        head_ids: torch.Tensor,
        head_engine_ids: torch.Tensor,
        pieces: list[KeptTurnPiece],
        head_str: str | None = None,
        head_pixel_values: torch.Tensor | None = None,
    ) -> RestartPrompt:
        """Concatenate the action-list head with the kept-turn pieces.

        :param head_str: The head's chat text on an image prompt; ``None`` on a
            text prompt.
        :param head_pixel_values: The goal images' pixels when the head carries them.
        """
        device = head_ids.device
        train_ids = torch.cat(
            [head_ids]
            + [
                ids.to(device)
                for piece in pieces
                for ids in (piece.gen_ids, piece.train_ids)
            ],
            dim=1,
        )
        if head_str is None:
            return RestartPrompt(train_ids, None, None, [], [])
        engine_ids = torch.cat(
            [head_engine_ids]
            + [
                ids.to(device)
                for piece in pieces
                for ids in (piece.gen_ids, piece.engine_ids)
            ],
            dim=1,
        )
        goal_images = self._goal_images if head_pixel_values is not None else []
        images = [
            *goal_images,
            *(piece.image for piece in pieces if piece.image is not None),
        ]
        pixel_parts = [
            pixels
            for pixels in (
                head_pixel_values,
                *(piece.pixel_values for piece in pieces),
            )
            if pixels is not None
        ]
        image_calls = [
            ImageProcessorCall.from_inputs(text=piece.feedback_text, image=img)
            for piece in pieces
            if (img := piece.image) is not None
        ]
        if goal_images:
            image_calls.insert(
                0, ImageProcessorCall.from_inputs(text=head_str, image=goal_images)
            )
        multimodal_turn = {
            "prompt": head_str
            + "".join(piece.gen_text + piece.feedback_text for piece in pieces),
            "image": images[0] if len(images) == 1 else images,
            "prompt_token_len": int(train_ids.shape[1]),
            "input_ids": train_ids,
            "prompt_token_ids": engine_ids,
        }
        return RestartPrompt(
            train_ids,
            multimodal_turn,
            torch.cat(pixel_parts, dim=0),
            images,
            image_calls,
        )

    def _kept_turn_piece(
        self, turn: KeptTurn, image_path: bool, with_image: bool, older: bool
    ) -> KeptTurnPiece:
        """One kept turn's sampled ids and the observation framed after them.

        :param turn: The kept turn.
        :param image_path: Whether the restarted prompt is an image prompt.
        :param with_image: Whether the observation may keep its image.
        :param older: Whether a later turn is kept after this one.
        """
        gen_ids = turn.gen_ids.unsqueeze(0)
        obs_text = turn.obs_text
        if older and turn.older_obs_text is not None:
            prefix = IMAGE_USER_CONTENT_PREFIX if turn.image is not None else ""
            obs_text = prefix + turn.older_obs_text
        keep_image = image_path and with_image and turn.image is not None
        if turn.image is not None and not keep_image:
            obs_text = obs_text.removeprefix(IMAGE_USER_CONTENT_PREFIX)
        if not image_path:
            ids = self._text_feedback_ids(obs_text, turn.role, turn.last_token_id)
            return KeptTurnPiece(gen_ids, ids, ids, "", "", None, None)
        gen_text = self._decode(gen_ids[0].tolist(), skip_special_tokens=False)
        feedback_text = self._image_feedback_turn_text(
            obs_text, turn.role, turn.last_token_id
        )
        engine_ids = self._engine_ids(feedback_text)
        if not keep_image:
            return KeptTurnPiece(
                gen_ids, engine_ids, engine_ids, gen_text, feedback_text, None, None
            )
        train_ids, pixel_values = encode_image_training_inputs(
            text=feedback_text,
            image=turn.image,
            processor=self._require_vision_processor(),
        )
        return KeptTurnPiece(
            gen_ids,
            train_ids,
            engine_ids,
            gen_text,
            feedback_text,
            turn.image,
            pixel_values,
        )
