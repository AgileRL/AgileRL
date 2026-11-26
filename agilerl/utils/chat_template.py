# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Chat template text and the render kwargs a tokenizer applies to every render."""

from __future__ import annotations

import uuid
from types import MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

CHAT_TEMPLATE_DEFAULT_KWARGS_ATTR = "chat_template_default_kwargs"
CHAT_TEMPLATE_ORIGINAL_APPLY_ATTR = "chat_template_apply_orig"

# Jinja template for tokenizers that ship without a chat template.
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'].capitalize() + ': ' + message['content'] + '\\n\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'Assistant: ' }}"
    "{% endif %}"
)


def _apply_chat_template(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- binds to a tokenizer and forwards its render signature
    for key, value in getattr(self, CHAT_TEMPLATE_DEFAULT_KWARGS_ATTR).items():
        kwargs.setdefault(key, value)
    return getattr(self, CHAT_TEMPLATE_ORIGINAL_APPLY_ATTR)(self, *args, **kwargs)


def resolve_chat_template_kwargs(tokenizer: Any) -> dict[str, Any]:  # noqa: ANN401 -- tokenizers carry these kwargs as dynamically attached attributes
    """Read the chat-template render kwargs configured on *tokenizer*.

    :param tokenizer: Tokenizer carrying the configured render kwargs.
    :type tokenizer: Any
    :return: Configured kwargs, empty when none are configured.
    :rtype: dict[str, Any]
    """
    defaults = getattr(tokenizer, CHAT_TEMPLATE_DEFAULT_KWARGS_ATTR, None)
    return dict(defaults) if defaults else {}


def inject_chat_template_kwargs(
    tokenizer: Any,  # noqa: ANN401 -- the wrap is attached to the tokenizer instance
    chat_template_kwargs: dict[str, Any] | None,
) -> Any:  # noqa: ANN401 -- returns the tokenizer it was handed
    """Fill missing ``apply_chat_template`` kwargs from *chat_template_kwargs*.

    Explicit caller kwargs win. The defaults are stored on the tokenizer as
    ``chat_template_default_kwargs`` and the unbound original as
    ``chat_template_apply_orig``.

    :param tokenizer: Tokenizer whose renders take the configured kwargs.
    :type tokenizer: Any
    :param chat_template_kwargs: Kwargs every render defaults to; empty or
        ``None`` leaves the tokenizer untouched.
    :type chat_template_kwargs: dict[str, Any] | None
    :return: The tokenizer.
    :rtype: Any
    """
    if not chat_template_kwargs:
        return tokenizer

    defaults = dict(chat_template_kwargs)
    method = tokenizer.apply_chat_template
    already_wrapped = (
        isinstance(method, MethodType) and method.__func__ is _apply_chat_template
    )
    if (
        already_wrapped
        and getattr(tokenizer, CHAT_TEMPLATE_DEFAULT_KWARGS_ATTR, None) == defaults
    ):
        return tokenizer

    if not already_wrapped:
        setattr(
            tokenizer,
            CHAT_TEMPLATE_ORIGINAL_APPLY_ATTR,
            getattr(method, "__func__", method),
        )

    setattr(tokenizer, CHAT_TEMPLATE_DEFAULT_KWARGS_ATTR, defaults)
    tokenizer.apply_chat_template = MethodType(_apply_chat_template, tokenizer)
    return tokenizer


def ensure_chat_template_kwargs_injected(tokenizer: Any) -> Any:  # noqa: ANN401 -- returns the tokenizer it was handed
    """Rebind the render wrap from the defaults attribute, which survives pickling.

    :param tokenizer: Tokenizer whose bound wrap may have been dropped.
    :type tokenizer: Any
    :return: The tokenizer.
    :rtype: Any
    """
    defaults = getattr(tokenizer, CHAT_TEMPLATE_DEFAULT_KWARGS_ATTR, None)
    if not defaults:
        return tokenizer
    return inject_chat_template_kwargs(tokenizer, defaults)


def special_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Ids of the tokenizer's special tokens, added special tokens included.

    Harmony-format tokenizers (gpt-oss) register their turn markers as special
    added tokens without listing them in ``all_special_ids``.

    :param tokenizer: Tokenizer to read.
    :return: Special token ids.
    """
    return set(tokenizer.all_special_ids) | {
        token_id
        for token_id, token in tokenizer.added_tokens_decoder.items()
        if token.special
    }


def feedback_frame_parts(
    tokenizer: PreTrainedTokenizerBase,
    roles: tuple[str, ...],
    chat_template_kwargs: dict[str, Any],
) -> tuple[str, ...] | None:
    """Text the chat template puts around the messages that follow an assistant turn.

    One part before each message plus a closing suffix that ends in the
    generation prompt: ``(prefix, suffix)`` for one message. Sliced at uuid4
    placeholders (render verbatim, can't collide); the dummy user message
    satisfies strict-alternation templates.

    :param tokenizer: Tokenizer whose chat template renders the frame.
    :param roles: Chat role of each message after the assistant turn.
    :param chat_template_kwargs: Render kwargs, ``tools`` included.
    :return: The frame parts, or ``None`` when the template cannot render them.
    """
    assistant_ph = uuid.uuid4().hex
    feedback_phs = [uuid.uuid4().hex for _ in roles]
    messages = [
        {"role": "user", "content": "."},
        {"role": "assistant", "content": assistant_ph},
        *(
            {"role": role, "content": placeholder}
            for role, placeholder in zip(roles, feedback_phs, strict=True)
        ),
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
    except Exception:
        return None
    if not isinstance(rendered, str):
        return None

    cursor = rendered.rfind(assistant_ph)
    if cursor < 0:
        return None
    cursor += len(assistant_ph)
    parts: list[str] = []
    for placeholder in feedback_phs:
        start = rendered.find(placeholder, cursor)
        if start < 0:
            return None
        parts.append(rendered[cursor:start])
        cursor = start + len(placeholder)
    parts.append(rendered[cursor:])
    if not parts[0] or not parts[-1]:
        return None
    return tuple(parts)
