# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Render env observation payloads to prompt text and chat roles."""

from __future__ import annotations

import base64
import io
import os
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from PIL import Image

from agilerl.utils.algo_utils import is_str_keyed_dict

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "DEFAULT_OBSERVATION_ROLE",
    "ESCAPED_IMAGE_PLACEHOLDER",
    "IMAGE_PLACEHOLDER",
    "IMAGE_USER_CONTENT_PREFIX",
    "OBSERVATION_ROLES",
    "QUESTION_AFTER_CONTEXT",
    "ImageProcessorCall",
    "ImageTokens",
    "apply_info_prefix_suffix",
    "encode_image_training_inputs",
    "goal_images_text",
    "image_placeholder_logit_bias",
    "image_to_base64_png",
    "model_image_token",
    "observation_goal_images",
    "observation_role",
    "observation_text_and_image",
    "process_observation",
]

#: Prompt text the vision processor (and vLLM) expands into one image's tokens.
IMAGE_PLACEHOLDER = "<image>"

IMAGE_USER_CONTENT_PREFIX = f"{IMAGE_PLACEHOLDER}\n"

#: Env text that spells the placeholder would be expanded as an extra image.
ESCAPED_IMAGE_PLACEHOLDER = "&lt;image&gt;"

# Splits long observation text from the question restated after it, so the
# instruction can sit between them.
QUESTION_AFTER_CONTEXT = "\n\nQuestion:\n"

#: Chat role an env observation speaks as when it does not say.
DEFAULT_OBSERVATION_ROLE = "user"

#: Roles an env may claim for an observation. ``assistant`` is excluded: those
#: turns are the policy's own generations, and letting an env inject them would
#: put untrained tokens inside a trained span.
OBSERVATION_ROLES = frozenset({"user", "tool", "system"})

DEFAULT_SCREENSHOT_PROMPT = "Complete the task shown in the image."


def _screenshot_to_pil_rgb(screenshot: object) -> Image.Image:
    """Decode a screenshot sent as base64 image bytes or a nested pixel list."""
    if isinstance(screenshot, str):
        with Image.open(io.BytesIO(base64.b64decode(screenshot))) as image:
            return image.convert("RGB")
    arr = np.asarray(screenshot)
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return Image.fromarray(arr.astype(np.uint8))
    msg = f"screenshot must have shape (H, W) or (H, W, 3), got {arr.shape}"
    raise TypeError(msg)


def image_to_base64_png(image: Image.Image) -> str:
    """PNG-encode an image as base64 text for a JSON observation."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decoded_image(image: object) -> object:
    """PIL image for each base64 string an OpenEnv server sent; others unchanged."""
    if isinstance(image, list):
        return [_decoded_image(item) for item in image]
    if isinstance(image, str):
        return _screenshot_to_pil_rgb(image)
    return image


def _prompt_text_for_screenshot_obs(obs: Mapping[str, object]) -> str:
    goal = obs.get("goal")
    text = obs.get("text")
    if isinstance(goal, str) and isinstance(text, str):
        if goal:
            return f"{goal}\n\n{text}{QUESTION_AFTER_CONTEXT}{goal}"
        return f"{goal}\n\n{text}"
    for key in ("goal", "text", "prompt"):
        value = obs.get(key)
        if isinstance(value, str):
            return value
    return DEFAULT_SCREENSHOT_PROMPT


def observation_text_and_image(
    obs: object, *, screenshot_field: str = "screenshot"
) -> tuple[str, object | None]:
    """Render observation text and an optional single image for VL rollouts.

    :param obs: The raw observation payload.
    :param screenshot_field: Observation key holding the page image. ``""``
        does not read one.
    """
    if isinstance(obs, str):
        return obs, None
    if is_str_keyed_dict(obs) and obs.get("image") is not None:
        text = obs.get("text")
        if text is None:
            text = obs.get("prompt")
        if not isinstance(text, str):
            msg = f"VL observation text must be str, got {type(text).__name__}."
            raise TypeError(msg)
        return _image_turn_text(text), _decoded_image(obs["image"])
    if (
        screenshot_field
        and is_str_keyed_dict(obs)
        and obs.get(screenshot_field) is not None
    ):
        text = _prompt_text_for_screenshot_obs(obs)
        return _image_turn_text(text), _screenshot_to_pil_rgb(obs[screenshot_field])
    return process_observation(obs), None


def observation_goal_images(
    obs: object, *, goal_images_field: str = "goal_images"
) -> list[Image.Image]:
    """Decode the task images an observation carries.

    Each entry is base64 image bytes or a pixel list, as the page image is.

    :param obs: The raw observation payload.
    :param goal_images_field: Observation key holding the images. ``""`` reads none.
    :return: The goal images as RGB, in order; empty when there are none.
    """
    images = (
        obs.get(goal_images_field)
        if goal_images_field and is_str_keyed_dict(obs)
        else None
    )
    if not isinstance(images, list):
        return []
    return [_screenshot_to_pil_rgb(image) for image in images]


def goal_images_text(count: int) -> str:
    """One placeholder line per goal image, appended after a turn's text.

    :param count: Number of goal images.
    :rtype: str
    """
    return f"\n{IMAGE_PLACEHOLDER}" * count


def apply_info_prefix_suffix(text: str, info: Mapping[str, Any]) -> str:
    """Wrap prompt text in the env's ``info`` prefix/suffix when present.

    Game envs in the gem convention return instructions as the observation and
    the task body as ``info["suffix"]``; gem's own observation wrapper
    concatenates them the same way.

    :param text: The rendered observation text.
    :param info: The env turn's info mapping.
    :return: ``text`` with the prefix prepended and the suffix appended.
    """
    prefix = info.get("prefix")
    suffix = info.get("suffix")
    if isinstance(prefix, str) and prefix:
        text = prefix + text
    if isinstance(suffix, str) and suffix:
        text = text + suffix
    return text


def _image_turn_text(text: str) -> str:
    """Lead with exactly one image placeholder; escape any the env text spells."""
    body = text.removeprefix(IMAGE_USER_CONTENT_PREFIX)
    escaped = body.replace(IMAGE_PLACEHOLDER, ESCAPED_IMAGE_PLACEHOLDER)
    return f"{IMAGE_USER_CONTENT_PREFIX}{escaped}"


def encode_image_training_inputs(
    *,
    text: str,
    image: object | list[object],
    processor: Callable[..., Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a checkpoint processor on one or more images for trainer ``pixel_values``."""
    if isinstance(image, list):
        images = image[0] if len(image) == 1 else image
    else:
        images = image
    encoded = processor(text=text, images=images, return_tensors="pt")
    input_ids = encoded["input_ids"]
    pixel_values = encoded["pixel_values"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(
        pixel_values, torch.Tensor
    ):
        msg = "processor must return torch input_ids and pixel_values tensors"
        raise TypeError(msg)
    image_grid_hw = encoded.get("image_grid_hw")
    image_grid_thw = encoded.get("image_grid_thw")
    image_position_ids = encoded.get("image_position_ids")
    if isinstance(image_grid_hw, torch.Tensor):
        pixel_values = patch_grid_rows(pixel_values, image_grid_hw)
    elif isinstance(image_grid_thw, torch.Tensor):
        if (image_grid_thw[:, 0] != 1).any():
            msg = f"Images must have one temporal patch, got grids {image_grid_thw.tolist()}"
            raise ValueError(msg)
        pixel_values = patch_grid_rows(pixel_values, image_grid_thw[:, 1:])
    elif isinstance(image_position_ids, torch.Tensor):
        pixel_values = position_grid_rows(pixel_values, image_position_ids)
    return input_ids, pixel_values


def patch_grid_rows(
    pixel_values: torch.Tensor, image_grid_hw: torch.Tensor
) -> torch.Tensor:
    """Fold packed ``(total_patches, patch_dim)`` patches into one ``(h, w, patch_dim)`` row per image.

    :param pixel_values: Row-major patches of every image, concatenated.
    :param image_grid_hw: ``(N, 2)`` patch grid of each image.
    :return: ``(N, h, w, patch_dim)`` patch-grid rows.
    :raises ValueError: If the images resize to different patch grids.
    """
    grids = {tuple(grid) for grid in image_grid_hw.tolist()}
    if len(grids) != 1:
        msg = (
            f"Images of one processor call resize to patch grids {sorted(grids)}; "
            "vision rows need one grid per call."
        )
        raise ValueError(msg)
    ((grid_height, grid_width),) = grids
    return pixel_values.reshape(
        len(image_grid_hw), grid_height, grid_width, pixel_values.shape[-1]
    )


def position_grid_rows(
    pixel_values: torch.Tensor, image_position_ids: torch.Tensor
) -> torch.Tensor:
    """Place padded ``(N, max_patches, patch_dim)`` patches into one ``(h, w, patch_dim)`` row per image.

    :param pixel_values: Each image's patches, padded to a common length.
    :param image_position_ids: ``(N, max_patches, 2)`` ``(x, y)`` patch position,
        ``-1`` on padding.
    :return: ``(N, h, w, patch_dim)`` patch-grid rows.
    :raises ValueError: If the images cover different patch grids.
    """
    valid = (image_position_ids >= 0).all(dim=-1)
    xs, ys = image_position_ids[..., 0], image_position_ids[..., 1]
    grids = {
        (
            int(ys[image][keep].max()) + 1,
            int(xs[image][keep].max()) + 1,
            int(keep.sum()),
        )
        for image, keep in enumerate(valid)
    }
    if len(grids) != 1 or any(h * w != count for h, w, count in grids):
        msg = (
            f"Images of one processor call cover patch grids {sorted(grids)} "
            "(height, width, patches); vision rows need one full grid per call."
        )
        raise ValueError(msg)
    ((grid_height, grid_width, _),) = grids
    images = torch.arange(len(valid)).unsqueeze(-1).expand_as(valid)
    rows = pixel_values.new_zeros(
        len(valid), grid_height, grid_width, pixel_values.shape[-1]
    )
    rows[images[valid], ys[valid], xs[valid]] = pixel_values[valid]
    return rows


class ImagePromptProcessor(Protocol):
    """The checkpoint processor surface :meth:`ImageTokens.from_processor` reads."""

    image_token: str

    def apply_chat_template(
        self, conversation: list[dict[str, Any]], /, **kwargs: Any
    ) -> str: ...


@dataclass(frozen=True)
class ImageTokens:
    """The prompt text one image takes in a model's chat format.

    Prompts are built with :data:`IMAGE_PLACEHOLDER`; :meth:`render` swaps in
    the model's own placeholder before the processor or vLLM sees the text.

    :param placeholder: Text the chat template writes for one image.
    :param image_token: Token the processor expands into the image's feature
        positions.
    """

    placeholder: str = IMAGE_PLACEHOLDER
    image_token: str = IMAGE_PLACEHOLDER

    @classmethod
    def from_processor(cls, processor: ImagePromptProcessor) -> ImageTokens:
        """Read the placeholder a checkpoint processor's chat template writes for one image."""
        sentinel = "IMAGE_TOKENS_SENTINEL"

        def render(content: list[dict[str, str]]) -> str:
            return processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=False,
            )

        with_image = render([{"type": "image"}, {"type": "text", "text": sentinel}])
        text_only = render([{"type": "text", "text": sentinel}])
        start = len(os.path.commonprefix([with_image, text_only]))
        return cls(
            placeholder=with_image[start : with_image.index(sentinel, start)],
            image_token=processor.image_token,
        )

    def render(self, text: str) -> str:
        """``text`` with each canonical placeholder replaced by the model's.

        A model image token already in ``text`` is escaped first, so only the
        placeholders become images.
        """
        if self.placeholder == IMAGE_PLACEHOLDER:
            return text
        return self.placeholder.join(
            self.escape(part) for part in text.split(IMAGE_PLACEHOLDER)
        )

    def escape(self, text: str) -> str:
        """``text`` with the model image token spelled so no processor expands it."""
        escaped_token = self.image_token.replace("<", "&lt;").replace(">", "&gt;")
        return text.replace(self.image_token, escaped_token)


def model_image_token(tokenizer: PreTrainedTokenizerBase) -> str:
    """Token a checkpoint's processor expands into image features.

    Processors read ``tokenizer.image_token`` and use :data:`IMAGE_PLACEHOLDER`
    when the checkpoint's tokenizer names none.
    """
    return getattr(tokenizer, "image_token", None) or IMAGE_PLACEHOLDER


def image_placeholder_logit_bias(
    vocab: Mapping[str, int], image_token: str
) -> dict[int, float] | None:
    """Logit bias that keeps rollouts from sampling the model's image token.

    A sampled image token takes an image feature row in the trainer's VL
    forward, and the harness fails any turn that holds one.

    :param vocab: Tokenizer vocabulary.
    :param image_token: Token the model's processor expands into image features.
    :return: ``{image_token_id: -100.0}``, or ``None`` when the vocabulary has
        no such token.
    """
    if image_token not in vocab:
        return None
    return {vocab[image_token]: -100.0}


@dataclass(frozen=True)
class ImageProcessorCall:
    """Inputs of one :func:`encode_image_training_inputs` call whose pixels an episode kept."""

    text: str
    images: tuple[object, ...]

    @classmethod
    def from_inputs(
        cls, *, text: str, image: object | list[object]
    ) -> ImageProcessorCall:
        """Record the ``text`` and ``image`` arguments of one processor call."""
        images = tuple(image) if isinstance(image, list) else (image,)
        return cls(text=text, images=images)


def _field(value: object, name: str) -> object:
    """``name`` of a dict (hosted env) or an attribute (in-process env), else ``None``."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _mcp_result_text(result: object) -> str | None:
    """Text from an MCP tool result's content blocks or ``data``, else ``None``."""
    content = _field(result, "content")
    blocks = content if isinstance(content, list) else []
    texts = [text for block in blocks if isinstance(text := _field(block, "text"), str)]
    if texts:
        return "\n".join(texts)
    data = _field(result, "data")
    return data if isinstance(data, str) else None


def process_observation(obs: object, observation_field: str | None = None) -> str:
    """Render an OpenEnv observation payload to prompt text.

    The default observation processor. A named ``observation_field`` wins: an
    env that carries its text somewhere only its author knows should be told,
    not guessed at. Otherwise the specified shapes are tried — an MCP tool
    result, our own ``TextObservation`` ``prompt``, then ``error``. Warns
    rather than returning silently empty text, since an empty prompt otherwise
    surfaces as an inexplicably broken episode much later.
    """
    if isinstance(obs, str):
        return obs
    if not is_str_keyed_dict(obs):
        warnings.warn(
            f"OpenEnv observation is a {type(obs).__name__}, not text or a "
            "mapping; the episode gets an empty prompt.",
            stacklevel=2,
        )
        return ""
    if observation_field is not None:
        named = obs.get(observation_field)
        if isinstance(named, str):
            return named
    result = obs.get("result")
    if result is not None:
        rendered = _mcp_result_text(result)
        if rendered is not None:
            return rendered
    prompt = obs.get("prompt")
    if isinstance(prompt, str):
        return prompt
    error = obs.get("error")
    if error:
        return f"Error: {_field(error, 'message') or error}"
    asked = (
        f" No '{observation_field}' either, which the manifest named."
        if observation_field is not None
        else ""
    )
    warnings.warn(
        "OpenEnv observation carried no text the episode could use, so it gets "
        f"an empty prompt. Keys present: {sorted(obs)}. Expected a 'prompt' "
        "(TextObservation), a 'result' (MCP tool call), or a named "
        "observation_field. An observation with no text fields at all (board "
        f"states, screens) needs an observation_processor to render it.{asked}",
        stacklevel=2,
    )
    return ""


def observation_role(payload: object, info: Mapping[str, Any] | None = None) -> str:
    """The chat role an env claims for one observation.

    An env labels a turn by putting ``role`` on the step metadata (``info``) or on
    the observation payload itself; ``info`` wins. Anything outside
    :data:`OBSERVATION_ROLES` is rejected rather than quietly rendered as a user
    message — a tool result presented as "the user said X" is exactly the
    confusion the label exists to prevent.

    :param payload: The raw observation payload from the env.
    :param info: Step metadata, where an env that separates the two puts the role.
    :return: One of :data:`OBSERVATION_ROLES`.
    :rtype: str
    """
    raw: object = None
    if info is not None:
        raw = info.get("role")
    if raw is None and isinstance(payload, Mapping):
        raw = payload.get("role")
    if raw is None:
        return DEFAULT_OBSERVATION_ROLE
    role = str(raw)
    if role not in OBSERVATION_ROLES:
        msg = (
            f"Env labelled an observation with role {role!r}; expected one of "
            f"{sorted(OBSERVATION_ROLES)}."
        )
        raise ValueError(msg)
    return role
