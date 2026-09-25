# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Single-turn digit recognition env with a rendered PIL image observation."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw


class DigitImageEnv:
    """Gym-style env: one image question per episode, scalar reward on the answer."""

    def __init__(self, *, image_size: int = 512) -> None:
        self._image_size = image_size
        self._digit: int | None = None

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        rng = random.Random(seed)
        digit = rng.randint(0, 9)
        self._digit = digit
        image = self._render_digit(digit)
        return (
            {
                "text": "What digit is shown? Answer with the digit.",
                "image": image,
            },
            {},
        )

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        digit = self._digit
        if digit is None:
            msg = "step() before reset()"
            raise RuntimeError(msg)
        reward = 1.0 if str(digit) in action else 0.0
        return "", reward, True, False, {}

    def _render_digit(self, digit: int) -> Image.Image:
        glyph = Image.new("L", (16, 16), 255)
        ImageDraw.Draw(glyph).text((2, 1), str(digit), fill=0)
        return glyph.resize(
            (self._image_size, self._image_size),
            Image.Resampling.NEAREST,
        ).convert("RGB")


def make(**kwargs: Any) -> DigitImageEnv:
    """Build a :class:`DigitImageEnv` (entrypoint for rollout manifests)."""
    return DigitImageEnv(**kwargs)
