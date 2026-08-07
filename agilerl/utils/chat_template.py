# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Chat template text and the render kwargs a tokenizer applies to every render."""

from __future__ import annotations

from types import MethodType
from typing import Any

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
