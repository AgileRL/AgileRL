# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Arena's prompt template over dataset rows, and the placeholders one may use."""

from __future__ import annotations

import re
from typing import Any

# A single-braced ``{name}``; doubled braces are literal text.
PROMPT_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_]\w*)\}(?!\})")
ROW_PLACEHOLDERS = ("question", "answer")
# The template Arena's starting manifest prompts each dataset row with. The
# system prompt names no <think> tags: on a model with native thinking those
# strings are control tokens, and a prompt that contains them puts the model
# inside its thinking block.
DEFAULT_PROMPT_TEMPLATE = {
    "system_0": (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then gives the final answer inside "
        "<answer> </answer> tags, i.e., <answer> answer here </answer>."
    ),
    "user_1": (
        "Answer the following question as accurately, clearly, and concisely "
        "as possible.\n\nQuestion: {question}\n"
    ),
}


def check_row_placeholders(
    prompt_template: dict[str, Any], columns: dict[str, str] | None
) -> None:
    """Reject a ``{name}`` the trainer never fills: over dataset rows only {question} and {answer} are.

    :param prompt_template: The environment's prompt template.
    :param columns: The environment's column mapping, or None when the names already match.
    :raises ValueError: A message is not text, or names a placeholder that is not a row field.
    """
    columns = columns or {}
    for key, text in prompt_template.items():
        if not isinstance(text, str):
            msg = f"prompt_template.{key} must be text, got {type(text).__name__}."
            raise ValueError(msg)
        for name in PROMPT_PLACEHOLDER.findall(text):
            if name in ROW_PLACEHOLDERS:
                continue
            msg = (
                f"prompt_template.{key} uses {{{name}}}, which no dataset "
                "column fills, so the model would see it as literal text. "
                "A prompt over dataset rows can use {question} and {answer}: "
                f"they read the dataset's {columns.get('question', 'question')!r} "
                f"and {columns.get('answer', 'answer')!r} columns. Map a "
                "different column onto either one in environment.columns."
            )
            raise ValueError(msg)
