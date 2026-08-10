# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for chat-template render kwargs configured on a tokenizer."""

from __future__ import annotations

from typing import ClassVar

from jinja2 import Template

from agilerl.utils.chat_template import (
    DEFAULT_CHAT_TEMPLATE,
    ensure_chat_template_kwargs_injected,
    inject_chat_template_kwargs,
    resolve_chat_template_kwargs,
)


class _RecordingTokenizer:
    """Tokenizer stand-in recording the kwargs each render was called with."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        return "rendered"


def _render(tokenizer, **kwargs):
    return tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], **kwargs)


class TestResolveChatTemplateKwargs:
    def test_unconfigured_tokenizer_resolves_to_nothing(self) -> None:
        assert resolve_chat_template_kwargs(_RecordingTokenizer()) == {}

    def test_configured_kwargs_are_copied(self) -> None:
        tokenizer = inject_chat_template_kwargs(
            _RecordingTokenizer(),
            {"enable_thinking": False},
        )

        resolved = resolve_chat_template_kwargs(tokenizer)
        resolved["enable_thinking"] = True

        assert resolve_chat_template_kwargs(tokenizer) == {"enable_thinking": False}

    def test_kwargs_resolve_from_the_attribute_alone(self) -> None:
        tokenizer = _RecordingTokenizer()
        tokenizer.chat_template_default_kwargs = {"enable_thinking": False}

        assert resolve_chat_template_kwargs(tokenizer) == {"enable_thinking": False}


class TestInjectChatTemplateKwargs:
    def test_no_kwargs_leaves_the_tokenizer_untouched(self) -> None:
        tokenizer = _RecordingTokenizer()

        assert inject_chat_template_kwargs(tokenizer, None) is tokenizer
        assert inject_chat_template_kwargs(tokenizer, {}) is tokenizer
        assert "apply_chat_template" not in vars(tokenizer)

    def test_defaults_fill_kwargs_the_caller_omits(self) -> None:
        tokenizer = inject_chat_template_kwargs(
            _RecordingTokenizer(),
            {"enable_thinking": False},
        )

        _render(tokenizer, add_generation_prompt=True)

        assert tokenizer.calls == [
            {"add_generation_prompt": True, "enable_thinking": False},
        ]

    def test_explicit_caller_kwargs_win(self) -> None:
        tokenizer = inject_chat_template_kwargs(
            _RecordingTokenizer(),
            {"enable_thinking": False},
        )

        _render(tokenizer, enable_thinking=True)

        assert tokenizer.calls == [{"enable_thinking": True}]

    def test_reinjecting_the_same_defaults_keeps_one_wrap(self) -> None:
        tokenizer = inject_chat_template_kwargs(
            _RecordingTokenizer(),
            {"enable_thinking": False},
        )
        wrapped = vars(tokenizer)["apply_chat_template"]

        inject_chat_template_kwargs(tokenizer, {"enable_thinking": False})

        assert vars(tokenizer)["apply_chat_template"] is wrapped
        assert (
            tokenizer.chat_template_apply_orig
            is _RecordingTokenizer.apply_chat_template
        )

    def test_new_defaults_replace_the_old_ones_without_stacking(self) -> None:
        tokenizer = inject_chat_template_kwargs(
            _RecordingTokenizer(),
            {"enable_thinking": False},
        )
        inject_chat_template_kwargs(tokenizer, {"mark_effort": True})

        _render(tokenizer)

        assert tokenizer.calls == [{"mark_effort": True}]
        assert (
            tokenizer.chat_template_apply_orig
            is _RecordingTokenizer.apply_chat_template
        )


class TestEnsureChatTemplateKwargsInjected:
    def test_an_unconfigured_tokenizer_is_left_alone(self) -> None:
        tokenizer = _RecordingTokenizer()

        assert ensure_chat_template_kwargs_injected(tokenizer) is tokenizer
        assert "apply_chat_template" not in vars(tokenizer)

    def test_the_wrap_is_rebuilt_from_the_defaults_attribute(self) -> None:
        tokenizer = _RecordingTokenizer()
        tokenizer.chat_template_default_kwargs = {"enable_thinking": False}

        ensure_chat_template_kwargs_injected(tokenizer)
        _render(tokenizer, add_generation_prompt=True)

        assert tokenizer.calls == [
            {"add_generation_prompt": True, "enable_thinking": False},
        ]

    def test_an_already_wrapped_tokenizer_keeps_its_wrap(self) -> None:
        tokenizer = inject_chat_template_kwargs(
            _RecordingTokenizer(),
            {"enable_thinking": False},
        )
        wrapped = vars(tokenizer)["apply_chat_template"]

        ensure_chat_template_kwargs_injected(tokenizer)

        assert vars(tokenizer)["apply_chat_template"] is wrapped


class TestDefaultChatTemplate:
    MESSAGES: ClassVar[list[dict[str, str]]] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]

    def test_renders_capitalised_roles_separated_by_blank_lines(self) -> None:
        rendered = Template(DEFAULT_CHAT_TEMPLATE).render(
            messages=self.MESSAGES,
            add_generation_prompt=False,
        )

        assert rendered == "User: hi\n\nAssistant: yo\n\n"

    def test_generation_prompt_appends_the_assistant_turn(self) -> None:
        rendered = Template(DEFAULT_CHAT_TEMPLATE).render(
            messages=self.MESSAGES,
            add_generation_prompt=True,
        )

        assert rendered == "User: hi\n\nAssistant: yo\n\nAssistant: "
