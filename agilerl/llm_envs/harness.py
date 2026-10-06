# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-level rollout env: tokenisation + turn loop over a text env client."""

from __future__ import annotations

import ast
import re
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch

from agilerl.components.llm_rollout_data import (
    EpisodeSegments,
    validate_episode_segments,
)
from agilerl.llm_envs.env_sources import is_url, spec_to_factory
from agilerl.llm_envs.observation import (
    DEFAULT_OBSERVATION_ROLE,
    ESCAPED_IMAGE_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    IMAGE_USER_CONTENT_PREFIX,
    QUESTION_AFTER_CONTEXT,
    encode_image_training_inputs,
    observation_role,
    observation_text_and_image,
    process_observation,
)
from agilerl.protocols import EnvClientProtocol, TextEnvProtocol
from agilerl.utils.algo_utils import is_str_keyed_dict
from agilerl.utils.env_utils import construct_entrypoint_env
from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len

__all__ = ["RolloutHarness", "TranscriptContinuityError", "env_action_text"]

# Every restart prompt repeats each entry; an unclosed reasoning block is the whole generation.
ACTION_HISTORY_MAX_CHARS = 300


class TranscriptContinuityError(RuntimeError):
    """A turn was sampled from a context that does not extend the tokens sampled so far."""


def env_action_text(gen_text: str) -> str:
    """Text the env executes: what follows the first ``</think>``, else the whole text.

    A template that prefills an open ``<think>`` yields ``reasoning</think>action``
    with no opening tag. A reasoning block that never closes passes through whole.
    """
    _reasoning, closed, action = gen_text.partition("</think>")
    if not closed:
        return _quote_unparsed_call(gen_text)
    return _quote_unparsed_call(action.strip())


def _quote_unparsed_call(text: str) -> str:
    """Wrap the payload of a call the env cannot parse as a string argument."""
    raw = text.strip()
    matched = re.match(r"\A([A-Za-z_][A-Za-z0-9_]*)\b(.*)\Z", raw, re.DOTALL)
    if matched is None:
        return text
    name, rest = matched.group(1), matched.group(2).strip()
    # A tag after the call is not a payload to quote.
    if not rest or rest[0] not in "({:" or "<" in raw:
        return text
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        tree = None
    if (
        tree is not None
        and len(tree.body) == 1
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Call)
    ):
        return raw
    payload = rest
    while payload and payload[0] in ":{( ":
        payload = payload[1:]
    while payload and payload[-1] in ")} ":
        payload = payload[:-1]
    if len(payload) >= 2 and payload[0] == payload[-1] and payload[0] in "'\"":
        payload = payload[1:-1]
    if not payload:
        return text
    escaped = payload.replace("\\", "\\\\").replace("'", "\\'")
    return f"{name}('{escaped}')"


if TYPE_CHECKING:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.rubrics.base import Rubric
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _coerced_system_prompt(value: object) -> str | None:
    """Non-empty system prompt text, or ``None``."""
    if isinstance(value, str) and value:
        return value
    return None


class RolloutHarness:
    """Token-level rollout env: tokenisation + turn loop over a text env client.

    Assembles the multi-turn transcript and the provenance mask (only policy-generated
    tokens train, via :meth:`get_episode_data`); terminates on ``max_model_len`` overflow.
    """

    def __init__(
        self,
        env_client: str | Callable[[], str] | EnvClientProtocol,
        tokenizer: PreTrainedTokenizerBase,
        max_turns: int = 1,
        *,
        timeout_s: float | None = None,
        mcp_tool: str | None = None,
        action_field: str = "message",
        observation_field: str | None = None,
        observation_processor: Callable[[Any], str] | None = None,
        instruction: str = "",
        strict_chat_template_boundary: bool = True,
        apply_chat_template: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_model_len: int | None = None,
        system_prompt: str | None = None,
        vision_processor: Callable[..., Mapping[str, Any]] | None = None,
        tasks: Sequence[Mapping[str, Any]] | None = None,
        eval_tasks: Sequence[Mapping[str, Any]] | None = None,
        segment_prompt_tokens: int | None = None,
        segment_max_images: int | None = None,
    ) -> None:
        """Drive a text env at the token level over ``env_client`` (a URL or a client object).

        :param env_client: URL string or zero-arg URL provider (either opens a
            WebSocket session, the provider re-resolved per dial) or an
            ``EnvClientProtocol``.
        :param tokenizer: Encodes prompts/feedback and applies the chat template.
        :param max_turns: Max generation turns per episode.
        :param timeout_s: Per-request OpenEnv timeout; ``None`` leaves requests unbounded.
        :param mcp_tool: MCP tool the text is sent to as ``call_tool``; ``None`` sends plain text.
        :param action_field: Action field the model's text goes into — the env's own
            name for it (``message`` by default, but ``code`` or ``action_str``
            for envs that name it otherwise), or the MCP tool's argument name.
        :param observation_field: Observation field the env's text comes back in.
            ``None`` reads the specified shapes and, failing those, the one text
            field the payload carries.
        :param observation_processor: Renders an observation payload to prompt
            text, replacing the default (:func:`process_observation`) for envs
            whose observations need more than a field lookup. Runs on the
            collector's I/O threads, so it must be thread-safe.
        :param instruction: Prompt used when reset's observation renders empty.
        :param strict_chat_template_boundary: Raise when the chat template cannot
            render a multi-turn boundary; ``False`` warns and falls back to
            ChatML markers, which malform the transcript on a non-ChatML tokenizer.
        :param apply_chat_template: Render prompts through the chat template vs raw encoding.
        :param chat_template_kwargs: Extra kwargs for every ``apply_chat_template``
            render (e.g. ``{"enable_thinking": False}``); the env's tool schemas
            are set on top under ``tools``.
        :param max_model_len: Engine context length; enables stop-on-overflow when set.
        :param system_prompt: Rendered as a leading ``system`` message ahead of the
            env's first observation. Requires ``apply_chat_template``; a template
            without a system slot raises at reset rather than dropping it silently.
        :param vision_processor: Callable ``(text=..., images=..., return_tensors='pt')``
            that returns ``input_ids`` and ``pixel_values`` for image observations.
            Each image turn's new text is processed alone with that turn's image;
            ``pixel_values`` concatenate along dim 0 across turns.
        :param tasks: Per-row reset kwargs forwarded to a URL-backed client.
        :param eval_tasks: Held-out per-row reset kwargs for eval mode on a URL-backed client.
        :param segment_prompt_tokens: Restart the episode's context when the next
            prompt would exceed this many tokens. The restarted prompt holds the
            system prompt, the actions taken so far and the latest observation;
            :meth:`get_episode_data` returns every segment. When that prompt is
            over the model budget the context continues unrestarted while it
            fits. ``None`` never restarts.
        :param segment_max_images: Restart the episode's context when the next
            prompt would carry more than this many images (match the engine's
            ``limit_mm_per_prompt``). A restarted prompt carries one image.
            ``None`` puts no image limit on a segment.
        :ivar full_ids: Running token sequence of the current segment (prompt +
            generations + feedback).
        :ivar turn_boundaries: ``(start, end, turn_idx)`` spans of policy-generated
            tokens in the current segment.
        :ivar turn_rewards: Per-turn rewards from the env.
        :ivar done: Whether the episode has terminated.
        :ivar current_prompt: Latest policy-ready prompt; ``{}`` once done.
        :ivar sampling_logps: Per-turn vLLM sampling logprobs (empty on the HF path).
        """
        self.max_turns = max_turns
        if segment_prompt_tokens is not None and segment_prompt_tokens < 1:
            msg = (
                "segment_prompt_tokens must be a positive int or None, "
                f"got {segment_prompt_tokens}."
            )
            raise ValueError(msg)
        self._segment_prompt_tokens = segment_prompt_tokens
        if segment_max_images is not None and segment_max_images < 1:
            msg = (
                "segment_max_images must be a positive int or None, "
                f"got {segment_max_images}."
            )
            raise ValueError(msg)
        self._segment_max_images = segment_max_images
        # Finished segments of the episode: ids through the last generation,
        # segment-relative turn boundaries, and the segment's pixel rows.
        self._segments: list[
            tuple[torch.Tensor, list[tuple[int, int, int]], torch.Tensor | None]
        ] = []
        # Text the env executed on each turn, capped; empty when restarts are off.
        self._action_history: list[str] = []
        if observation_field is not None and observation_processor is not None:
            msg = (
                "observation_field names the field the default observation "
                "processor reads; a custom observation_processor replaces that "
                "default. Set one or the other."
            )
            raise ValueError(msg)
        self._observation_processor: Callable[[Any], str] = (
            observation_processor
            or partial(process_observation, observation_field=observation_field)
        )
        self._instruction = instruction
        self._strict_chat_template_boundary = strict_chat_template_boundary
        if isinstance(env_client, EnvClientProtocol):
            if (
                timeout_s is not None
                or mcp_tool is not None
                or action_field != "message"
                or tasks is not None
                or eval_tasks is not None
            ):
                msg = (
                    "timeout_s / mcp_tool / action_field / tasks / "
                    "eval_tasks configure the transport and have no effect "
                    "on an already-built env client; set them on the client instead."
                )
                raise ValueError(msg)
            self._env_client: EnvClientProtocol = env_client
        else:
            from agilerl.llm_envs.openenv import RemoteEnvClient  # optional extra: llm

            self._env_client = RemoteEnvClient(
                env_client,
                timeout_s=timeout_s,
                mcp_tool=mcp_tool,
                action_field=action_field,
                tasks=tasks,
                eval_tasks=eval_tasks,
            )
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template
        self.chat_template_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
        # Tool schemas fetched lazily on first use (see :attr:`tools`).
        self._tools: list[Any] | None = None
        self._tools_known = False
        self._max_model_len = max_model_len
        self.full_ids: torch.Tensor | None = None
        self.turn_boundaries: list[tuple[int, int, int]] = []
        self.turn_rewards: list[float] = []
        self.rubric_score_sums: dict[str, float] = {}
        self._turn_idx = 0
        self._prompt_text: str = ""
        # Image episodes: the last prompt plus its sampled tokens, as text and as
        # the ids vLLM generates from (image placeholders unexpanded).
        self._transcript: tuple[str, torch.Tensor] | None = None
        # Image episodes: the last turn's sampled sequence, as the engine returned it.
        self._sampled_ids: torch.Tensor | None = None
        self._episode_images: list[object] = []
        self._last_full_prompt_token_len: int | None = None
        # Cached chat-template frame around a feedback turn, per observation role
        # (rendered once each): a tool result and a user message get different frames.
        self._boundary_parts: dict[str, tuple[str, str] | None] = {}
        self._system_prompt = system_prompt or None
        if self._system_prompt is not None and not self.apply_chat_template:
            msg = (
                "system_prompt is rendered through the chat template; with "
                "apply_chat_template=False the env's text is already fully "
                "rendered, so prepend the system turn there instead."
            )
            raise ValueError(msg)
        self.done: bool = False
        self.current_prompt: dict[str, Any] = {}
        self.sampling_logps: list[torch.Tensor] = []
        self._special_ids_cache: frozenset[int] | None = None
        self._vision_processor = vision_processor
        # The VL forward scatters one image feature row into every id equal to
        # the placeholder, generated ones included.
        self._image_placeholder_id: int | None = (
            None
            if vision_processor is None
            else tokenizer.get_vocab().get(IMAGE_PLACEHOLDER)
        )
        self._multimodal_turn: dict[str, Any] | None = None
        self._episode_pixel_values: torch.Tensor | None = None

    @classmethod
    def local(
        cls,
        env: TextEnvProtocol | Environment,
        tokenizer: PreTrainedTokenizerBase,
        max_turns: int = 1,
        *,
        action_field: str = "message",
        instruction: str = "",
        **kwargs: Any,
    ) -> RolloutHarness:
        """Drive a local env **in-process** (no HTTP) via a :class:`InProcessEnvClient`.

        :param env: A plain-text env or an OpenEnv ``Environment``.
        :param tokenizer: Tokenizer for the token-level loop.
        :param max_turns: Generation turns per episode.
        :param action_field: Action field the model's text goes into, on the
            env's declared action class.
        :param instruction: Prompt returned when the env's reset obs renders empty.
        :param kwargs: Forwarded to :class:`RolloutHarness`.
        :rtype: RolloutHarness
        """
        from agilerl.llm_envs.openenv import InProcessEnvClient  # optional extra: llm

        env_client = InProcessEnvClient(env, action_field=action_field)
        if _coerced_system_prompt(kwargs.get("system_prompt")) is None:
            env_prompt = _coerced_system_prompt(getattr(env, "system_prompt", None))
            if env_prompt is not None:
                kwargs["system_prompt"] = env_prompt
        return cls(
            env_client,
            tokenizer,
            max_turns=max_turns,
            instruction=instruction,
            **kwargs,
        )

    @classmethod
    def from_spec(
        cls,
        spec: str | Callable[..., TextEnvProtocol],
        env_config: dict[str, Any] | None,
        tokenizer: PreTrainedTokenizerBase,
        max_turns: int = 1,
        *,
        factory: str | None = None,
        **kwargs: Any,
    ) -> RolloutHarness:
        """Build a ``RolloutHarness`` from an env ``spec``.

        A **URL** is driven remotely; anything else is built with ``env_config``
        and driven in-process. ``factory`` is the optional ``module:attr`` callable
        that receives the entrypoint as its first argument; when unset, ``spec``
        itself is the constructor.

        :param spec: A URL, an env constructor callable, or an entrypoint /
            registry id.
        :param env_config: Kwargs for the factory / constructor (ignored for a
            URL, except ``system_prompt``). ``system_prompt`` is set on the built
            env and passed to the harness so the chat template renders it.
        :param tokenizer: Tokenizer for the token-level loop.
        :param max_turns: Generation turns per episode.
        :param factory: Optional ``module:attr`` builder that receives ``spec``.
        :param kwargs: Forwarded to :class:`RolloutHarness`.
        :rtype: RolloutHarness
        """
        config = dict(env_config or {})
        # A library factory takes the env id and its own kwargs, so a manifest
        # cannot pass ``system_prompt`` through it -- ``gem.make`` rejects the
        # kwarg outright. Override it on the built env instead, which is what a
        # manifest naming a prompt for a registry env means. The harness reads
        # that attribute (or an explicit kwarg) so the chat template renders it.
        system_prompt = config.pop("system_prompt", None)
        if isinstance(spec, str) and is_url(spec):
            if "system_prompt" not in kwargs:
                kwargs["system_prompt"] = system_prompt
            return cls(spec, tokenizer, max_turns=max_turns, **kwargs)
        if factory is not None:
            if not isinstance(spec, str):
                msg = "factory requires an entrypoint string, not a callable spec."
                raise TypeError(msg)
            env = cast(
                "TextEnvProtocol",
                construct_entrypoint_env(spec, config, factory=factory),
            )
        else:
            builder = spec_to_factory(spec) if isinstance(spec, str) else spec
            env = builder(**config)
        if system_prompt is not None:
            env.system_prompt = system_prompt
        if system_prompt is not None and "system_prompt" not in kwargs:
            kwargs["system_prompt"] = system_prompt
        return cls.local(env, tokenizer, max_turns=max_turns, **kwargs)

    @classmethod
    def from_dataset(
        cls,
        dataset: Sequence[Mapping[str, Any]],
        rubric: Rubric,
        tokenizer: PreTrainedTokenizerBase,
        *,
        test_dataset: Sequence[Mapping[str, Any]] | None = None,
        prompt_builder: Callable[[Mapping[str, Any]], str] | None = None,
        question_column: str = "question",
        answer_column: str = "answer",
        **kwargs: Any,
    ) -> RolloutHarness:
        """Build a single-turn ``RolloutHarness`` from (question, answer) rows and a rubric.

        Rows + rubric are served in-process as a single-turn
        :class:`~agilerl.llm_envs.qa_dataset.QADatasetEnv` (``max_turns`` fixed
        to 1); ``test_dataset`` rows are served under :meth:`eval_mode`. Wrap a
        ``(completion, answer, question) -> float`` callable with
        :func:`~agilerl.llm_envs.rubrics.reward_fn_to_rubric` first.

        :param dataset: Train rows, indexable to mappings with the question/answer keys.
        :param rubric: OpenEnv ``Rubric`` scoring each completion.
        :param tokenizer: Tokenizer for the token-level loop.
        :param test_dataset: Held-out rows served under eval mode (falls back to ``dataset``).
        :param prompt_builder: Maps a row to the served prompt text; ``None`` serves
            ``str(row[question_column])``.
        :param question_column: Row key for the question.
        :param answer_column: Row key for the answer.
        :param kwargs: Forwarded to :class:`RolloutHarness`.
        :rtype: RolloutHarness
        """
        from agilerl.llm_envs.qa_dataset import QADatasetEnv  # optional extra: llm
        from agilerl.llm_envs.rubrics import _require_rubric  # optional extra: llm

        if "max_turns" in kwargs:
            msg = "from_dataset builds a single-turn env; max_turns is fixed to 1"
            raise TypeError(msg)
        env = QADatasetEnv(
            dataset,
            rubric=_require_rubric(rubric),
            test_dataset=test_dataset,
            prompt_builder=prompt_builder,
            question_column=question_column,
            answer_column=answer_column,
        )
        return cls.local(env, tokenizer, max_turns=1, **kwargs)

    def _adopt_system_prompt(self, info: Mapping[str, Any]) -> None:
        """Take a system prompt from reset metadata when the harness has none."""
        if self._system_prompt is not None:
            return
        incoming = _coerced_system_prompt(info.get("system_prompt"))
        if incoming is None:
            return
        if not self.apply_chat_template:
            msg = (
                "system_prompt is rendered through the chat template; with "
                "apply_chat_template=False the env's text is already fully "
                "rendered, so prepend the system turn there instead."
            )
            raise ValueError(msg)
        self._system_prompt = incoming

    def _user_turn_chat_template_inputs(
        self, obs_text: str
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Messages and kwargs for one user turn through the chat template."""
        chat_template_kwargs = dict(self.chat_template_kwargs)
        if self.tools is not None:
            chat_template_kwargs["tools"] = self.tools
        messages: list[dict[str, str]] = []
        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append(
            {"role": "user", "content": self._with_trailing_instruction(obs_text)}
        )
        return messages, chat_template_kwargs

    def _with_trailing_instruction(self, content: str) -> str:
        """Put the instruction after the observation text and leave the question last."""
        if self._system_prompt is None:
            return content
        if QUESTION_AFTER_CONTEXT in content:
            context, question = content.rsplit(QUESTION_AFTER_CONTEXT, 1)
            return f"{context}\n{self._system_prompt}{QUESTION_AFTER_CONTEXT}{question}"
        return f"{content}\n{self._system_prompt}"

    def _chat_prompt_string(self, obs_text: str) -> str:
        """Render one user turn (plus optional system) through the chat template."""
        if not self.apply_chat_template:
            return obs_text
        messages, chat_template_kwargs = self._user_turn_chat_template_inputs(obs_text)
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if not isinstance(rendered, str):
            msg = "apply_chat_template(tokenize=False) must return str"
            raise TypeError(msg)
        return rendered

    def _image_feedback_turn_text(
        self, feedback_text: str, role: str, last_token_id: int
    ) -> str:
        """Text that extends the image transcript by one feedback turn.

        The instruction follows every observation and stays in history, so each
        prompt extends the previous one verbatim.

        :param feedback_text: The env's rendered observation for this turn.
        :param role: Chat role the observation speaks as.
        :param last_token_id: Last token of the transcript being extended.
        """
        content = self._with_trailing_instruction(feedback_text)
        if not self.apply_chat_template:
            return content
        parts = self._feedback_boundary_parts(role)
        if parts is None:
            msg = (
                f"The tokenizer's chat template could not render a {role!r} "
                "feedback turn boundary for the image transcript."
            )
            raise RuntimeError(msg)
        prefix, suffix = parts
        # A sampled end-of-turn token already closes the transcript.
        if last_token_id in self._special_ids():
            end_text = self._decode([last_token_id], skip_special_tokens=False)
            prefix = prefix.removeprefix(end_text)
        return prefix + content + suffix

    def _require_transcript_continues(
        self, transcript_ids: torch.Tensor, prompt_ids: torch.Tensor
    ) -> None:
        """Raise unless ``prompt_ids`` starts with ``transcript_ids``, token for token.

        The trainer scores every earlier turn inside the last prompt, so each
        turn must be sampled from the exact tokens sampled before it.
        """
        expected = transcript_ids[0].tolist()
        head = prompt_ids[0, : len(expected)].tolist()
        if head == expected:
            return
        diverged = next(
            (
                idx
                for idx, (a, b) in enumerate(zip(head, expected, strict=False))
                if a != b
            ),
            len(head),
        )
        msg = (
            f"Turn {self._turn_idx} prompt does not extend the sampled transcript: "
            f"tokens diverge at position {diverged} of {len(expected)}."
        )
        raise TranscriptContinuityError(msg)

    def _engine_ids(self, text: str) -> torch.Tensor:
        """``(1, T)`` ids of ``text`` with each image placeholder left unexpanded."""
        return torch.tensor(
            [self.tokenizer.encode(text, add_special_tokens=False)], dtype=torch.long
        )

    def _decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        """Text of one id sequence."""
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        if not isinstance(text, str):
            msg = "decode() of one sequence returns str"
            raise TypeError(msg)
        return text

    def _require_vision_processor(self) -> Callable[..., Mapping[str, Any]]:
        """The vision processor; raises when the harness has none."""
        if self._vision_processor is None:
            msg = "Image observations require vision_processor on RolloutHarness"
            raise RuntimeError(msg)
        return self._vision_processor

    def _prompt_image_payload(self) -> object | list[object]:
        """``prompt['image']`` value for the current episode images."""
        if len(self._episode_images) == 1:
            return self._episode_images[0]
        return self._episode_images

    def _tokenize_initial_prompt(self, obs_text: str) -> torch.Tensor:
        """Tokenize the initial observation, optionally with chat template."""
        if self.apply_chat_template:
            messages, chat_template_kwargs = self._user_turn_chat_template_inputs(
                obs_text
            )
            result: Any = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            token_ids = result["input_ids"] if isinstance(result, Mapping) else result
            if (
                isinstance(token_ids, list)
                and token_ids
                and isinstance(token_ids[0], list)
            ):
                token_ids = token_ids[0]
            return torch.tensor([token_ids], dtype=torch.long)

        # ``apply_chat_template=False`` means the text is already fully rendered
        # (specials included, e.g. a template-baked BOS); adding them again would
        # double the BOS on Llama/Gemma/Mistral-family tokenizers.
        encoded = self.tokenizer(
            [obs_text],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_attention_mask=True,
            add_special_tokens=False,
        )
        return encoded["input_ids"]

    def _tokenize_feedback(
        self,
        feedback_text: str,
        role: str = DEFAULT_OBSERVATION_ROLE,
    ) -> torch.Tensor:
        """Tokenize the feedback turn via the cached chat-template frame (ChatML fallback).

        :param feedback_text: The env's rendered observation for this turn.
        :param role: Chat role the observation speaks as — ``user`` for an env
            message, ``tool`` for a tool result, ``system`` for an injected
            directive. The frame is rendered per role, so a tool result is not
            passed off to the model as something the user said.
        """
        if not self.apply_chat_template:
            return torch.tensor(
                [self.tokenizer.encode(feedback_text, add_special_tokens=False)],
                dtype=torch.long,
            )

        boundary_ids = self._chat_template_boundary_ids(feedback_text, role)
        if boundary_ids is not None:
            return boundary_ids

        msg = (
            f"The tokenizer's chat template could not render a {role!r} feedback "
            "turn boundary; falling back to ChatML markers "
            "(<|im_end|>/<|im_start|>). For a non-ChatML tokenizer the multi-turn "
            "transcript will be malformed."
        )
        if self._strict_chat_template_boundary:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)
        # Fallback: ChatML-style markers.
        turn_boundary = (
            f"<|im_end|>\n<|im_start|>{role}\n"
            + feedback_text
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        return torch.tensor(
            [self.tokenizer.encode(turn_boundary, add_special_tokens=False)],
            dtype=torch.long,
        )

    def _special_ids(self) -> frozenset[int]:
        """The tokenizer's special-token id set (cached; empty when undeclared)."""
        if self._special_ids_cache is None:
            self._special_ids_cache = frozenset(
                int(i) for i in (getattr(self.tokenizer, "all_special_ids", None) or [])
            )
        return self._special_ids_cache

    def _feedback_boundary_parts(
        self,
        role: str = DEFAULT_OBSERVATION_ROLE,
    ) -> tuple[str, str] | None:
        """Cached ``(prefix, suffix)`` the chat template wraps a ``role`` feedback turn in.

        Sliced at two uuid4 placeholders (render verbatim, can't collide); the dummy
        user message satisfies strict-alternation templates. ``None`` -> ChatML fallback.

        :param role: Chat role of the observation turn being framed.
        """
        if role in self._boundary_parts:
            return self._boundary_parts[role]
        self._boundary_parts[role] = None
        assistant_ph = uuid.uuid4().hex
        feedback_ph = uuid.uuid4().hex
        messages = [
            {"role": "user", "content": "."},
            {"role": "assistant", "content": assistant_ph},
            {"role": role, "content": feedback_ph},
        ]
        chat_template_kwargs = dict(self.chat_template_kwargs)
        if self.tools is not None:
            chat_template_kwargs["tools"] = self.tools
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except Exception:
            return None
        if not isinstance(rendered, str):
            return None

        assistant_end = rendered.rfind(assistant_ph)
        feedback_start = rendered.rfind(feedback_ph)
        if assistant_end < 0 or feedback_start <= assistant_end:
            return None
        assistant_end += len(assistant_ph)
        prefix = rendered[assistant_end:feedback_start]
        suffix = rendered[feedback_start + len(feedback_ph) :]
        if not prefix or not suffix:
            return None
        self._boundary_parts[role] = (prefix, suffix)
        return self._boundary_parts[role]

    def _chat_template_boundary_ids(
        self,
        feedback_text: str,
        role: str = DEFAULT_OBSERVATION_ROLE,
    ) -> torch.Tensor | None:
        """Token ids for the templated turn boundary carrying ``feedback_text`` (``None`` if no frame)."""
        parts = self._feedback_boundary_parts(role)
        if parts is None:
            return None
        prefix, suffix = parts
        encoded = self.tokenizer.encode(
            prefix + feedback_text + suffix, add_special_tokens=False
        )
        if not encoded:
            return None
        return torch.tensor([encoded], dtype=torch.long)

    def _prompt_budget(self) -> int | None:
        """Max prompt tokens before context overflow, or ``None`` when no model length is set."""
        if self._max_model_len is None:
            return None
        return max_prompt_tokens_for_model_len(self._max_model_len)

    def _policy_prompt_from_state(self) -> dict[str, Any]:
        """Build the ``get_action`` prompt dict from harness state."""
        if self._multimodal_turn is not None:
            prompt_len = int(self._multimodal_turn["prompt_token_len"])
            self._last_full_prompt_token_len = prompt_len
            prompt: dict[str, Any] = dict(self._multimodal_turn)
            if self._episode_pixel_values is not None:
                prompt["pixel_values"] = self._episode_pixel_values
            return prompt
        if self.full_ids is None:
            msg = "No prompt: reset() was never called"
            raise RuntimeError(msg)
        self._last_full_prompt_token_len = int(self.full_ids.shape[1])
        return {"input_ids": self.full_ids}

    @property
    def tools(self) -> list[Any] | None:
        """Tool schemas the env advertises (``None`` when none); fetched once and cached."""
        if not self._tools_known:
            self._tools = self._env_client.tools or None
            self._tools_known = True
            if self._tools and self.apply_chat_template:
                template = getattr(self.tokenizer, "chat_template", None)
                if template and "tools" not in template:
                    warnings.warn(
                        f"The env advertises {len(self._tools)} tool(s) but the "
                        "tokenizer's chat template never references 'tools'; the "
                        "schemas will not be rendered into any prompt.",
                        stacklevel=2,
                    )
        return self._tools

    @tools.setter
    def tools(self, value: list[Any] | None) -> None:
        """Override the advertised tool schemas (``None`` / ``[]`` -> no ``tools=``)."""
        self._tools = value or None
        self._tools_known = True

    @property
    def dataset_size(self) -> int:
        """Rows in the env's dataset (``0`` for a procedural env)."""
        return self._env_client.dataset_size

    @property
    def rubric_components(self) -> tuple[str, ...]:
        """Leaf rubric names for component metrics (empty when none)."""
        return tuple(self._env_client.rubric_components)

    @contextmanager
    def eval_mode(self) -> Iterator[None]:
        """Serve the env client's held-out split for the duration of the block."""
        with self._env_client.eval_mode():
            yield

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a fresh episode and return the policy-ready prompt plus info.

        A prompt over the context budget ends truncated at turn 0 (empty ``current_prompt``).

        :param seed: Reset seed forwarded to the env client.
        :param row_index: Dataset row index to serve (dataset-backed envs only).
        """
        obs_text, image, info = self._reset_fetch(seed, row_index=row_index)
        return self._reset_apply(obs_text, info, image=image)

    def _render_observation(self, payload: object) -> str:
        """Render one observation payload to prompt text via the processor."""
        text = self._observation_processor(payload)
        if not isinstance(text, str):
            name = getattr(
                self._observation_processor, "__name__", "observation_processor"
            )
            msg = f"{name} must return the prompt text, got {type(text).__name__}."
            raise TypeError(msg)
        return text

    def _reset_fetch(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[str, object | None, dict[str, Any]]:
        """Pull and render the initial prompt from the env backend — the parallelizable I/O.

        No tokenizer work; touching :attr:`tools` warms its cache to overlap too.
        """
        _ = self.tools
        payload, info = self._env_client.reset(seed=seed, row_index=row_index)
        obs_text, image = observation_text_and_image(payload)
        if image is None:
            obs_text = self._render_observation(payload) or self._instruction
        elif not obs_text:
            obs_text = self._instruction
        return obs_text, image, info

    def _reset_apply(
        self,
        obs_text: str,
        info: dict[str, Any],
        *,
        image: object | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Tokenize the initial prompt and start the episode (truncates at turn 0 if over budget)."""
        self._adopt_system_prompt(info)
        self._multimodal_turn = None
        self._episode_pixel_values = None
        self._episode_images = [image] if image is not None else []
        if image is not None:
            processor = self._require_vision_processor()
            prompt_str = self._chat_prompt_string(obs_text)
            train_ids, pixel_values = encode_image_training_inputs(
                text=prompt_str,
                image=image,
                processor=processor,
            )
            prompt_token_len = int(train_ids.shape[-1])
            self.full_ids = None
            self._episode_pixel_values = pixel_values
            self._multimodal_turn = {
                "prompt": prompt_str,
                "image": image,
                "prompt_token_len": prompt_token_len,
                "input_ids": train_ids,
                "prompt_token_ids": self._engine_ids(prompt_str),
            }
        else:
            self.full_ids = self._tokenize_initial_prompt(obs_text)
        self.turn_boundaries = []
        self.turn_rewards = []
        self.rubric_score_sums = {}
        self._turn_idx = 0
        self._prompt_text = obs_text
        self._transcript = None
        self._sampled_ids = None
        self.sampling_logps = []
        self._segments = []
        self._action_history = []

        max_pt = self._prompt_budget()
        if self._multimodal_turn is not None:
            prompt_len = int(self._multimodal_turn["prompt_token_len"])
        else:
            full_ids = self.full_ids
            if full_ids is None:
                msg = "reset() left no prompt token ids"
                raise RuntimeError(msg)
            prompt_len = int(full_ids.shape[1])
        if max_pt is not None and prompt_len > max_pt:
            self.done = True
            self.current_prompt = {}
            self._last_full_prompt_token_len = prompt_len
            return self.current_prompt, info

        self.done = False
        self.current_prompt = self._policy_prompt_from_state()
        return self.current_prompt, info

    def _step_prepare(
        self,
        token_ids: torch.Tensor,
        sampling_logps: torch.Tensor | None = None,
    ) -> str:
        """Decode and record this turn's generation; return its text (tokenizer phase)."""
        if self._last_full_prompt_token_len is None:
            msg = "step() requires a prior reset() or step() that built a prompt"
            raise RuntimeError(msg)
        prompt_len = self._last_full_prompt_token_len
        sequence = token_ids if token_ids.dim() > 1 else token_ids.unsqueeze(0)
        turn = self._multimodal_turn
        if turn is not None:
            if self._sampled_ids is not None:
                self._require_transcript_continues(self._sampled_ids, sequence)
            self._sampled_ids = sequence.detach()
            gen_ids = sequence[0, prompt_len:].detach()
            self._reject_image_placeholder_ids(gen_ids)
            gen_text = self._decode(gen_ids.tolist(), skip_special_tokens=True)
            if sampling_logps is not None:
                self.sampling_logps.append(sampling_logps)
            sampled_text = self._decode(gen_ids.tolist(), skip_special_tokens=False)
            engine_ids = turn["prompt_token_ids"]
            self._transcript = (
                turn["prompt"] + sampled_text,
                torch.cat(
                    [engine_ids, gen_ids.unsqueeze(0).to(engine_ids.device)], dim=1
                ),
            )
            self._multimodal_turn = None
            # pixel_values was built from the processor ids. The image-token
            # count in the training sequence has to match that tensor.
            processor_ids = turn["input_ids"].detach()
            self.full_ids = torch.cat(
                [processor_ids.to(gen_ids.device), gen_ids.unsqueeze(0)],
                dim=1,
            )
            gen_end = int(self.full_ids.shape[1])
            self.turn_boundaries.append(
                (int(processor_ids.shape[1]), gen_end, self._turn_idx)
            )
            self._record_action(gen_text)
            return gen_text
        full_ids = self.full_ids
        if full_ids is None:
            msg = "step() requires a prior reset() or step() that built a prompt"
            raise RuntimeError(msg)
        # Only the new suffix crosses devices; the prefix is byte-identical to ``full_ids``.
        gen_ids = sequence[0, prompt_len:].detach().to(full_ids.device)
        self._reject_image_placeholder_ids(gen_ids)
        gen_text = self._decode(gen_ids.tolist(), skip_special_tokens=True)
        if sampling_logps is not None:
            self.sampling_logps.append(sampling_logps)
        self.full_ids = torch.cat([full_ids, gen_ids.unsqueeze(0)], dim=1)
        gen_end = self.full_ids.shape[1]
        self.turn_boundaries.append((prompt_len, gen_end, self._turn_idx))
        self._record_action(gen_text)
        return gen_text

    def _reject_image_placeholder_ids(self, gen_ids: torch.Tensor) -> None:
        """Fail a turn whose sampled ids hold the image placeholder id.

        :param gen_ids: This turn's generated ids.
        :raises ValueError: If any generated id is the image placeholder id;
            training would scatter image features into it with no pixel rows.
        """
        if self._image_placeholder_id is None:
            return
        count = int((gen_ids == self._image_placeholder_id).sum())
        if count:
            msg = (
                f"Turn {self._turn_idx} generated the image placeholder "
                f"{IMAGE_PLACEHOLDER!r} (id {self._image_placeholder_id}) {count} "
                "time(s); its training row would have more image positions than "
                "pixel rows. Suppress the id at sampling time."
            )
            raise ValueError(msg)

    def _record_action(self, gen_text: str) -> None:
        """Add this turn's env action to the restart history, capped in length."""
        if self._segment_prompt_tokens is None and self._segment_max_images is None:
            return
        action = env_action_text(gen_text)
        if len(action) > ACTION_HISTORY_MAX_CHARS:
            action = action[:ACTION_HISTORY_MAX_CHARS] + "…"
        self._action_history.append(action)

    def _step_env(
        self, gen_text: str
    ) -> tuple[str, str, object | None, float, bool, bool, dict[str, Any]]:
        """Round-trip the env backend and render its observation — the parallelizable phase.

        Carries the observation's chat role alongside its text so :meth:`_step_apply`
        frames a tool result as a tool turn rather than as something the user said.
        """
        payload, reward, terminated, truncated, info = self._env_client.step(
            env_action_text(gen_text)
        )
        if is_str_keyed_dict(payload) and (
            payload.get("image") is not None or payload.get("screenshot") is not None
        ):
            obs_text, image = observation_text_and_image(payload)
        else:
            image = None
            obs_text = self._render_observation(payload)
        return (
            obs_text,
            observation_role(payload, info),
            image,
            reward,
            terminated,
            truncated,
            info,
        )

    def _step_apply(
        self,
        env_result: tuple[str, str, object | None, float, bool, bool, dict[str, Any]],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply the env round-trip result: rewards, truncation, feedback tokens."""
        next_obs, next_role, next_image, reward, terminated, truncated, info = (
            env_result
        )
        self.turn_rewards.append(float(reward))
        for name, value in (info.get("rubric_scores") or {}).items():
            self.rubric_score_sums[name] = self.rubric_score_sums.get(
                name, 0.0
            ) + float(value)
        self._turn_idx += 1

        if not (terminated or truncated) and self._turn_idx >= self.max_turns:
            truncated = True

        prompt: dict[str, Any] = {}
        if not (terminated or truncated):
            full_ids = self.full_ids
            if full_ids is None:
                msg = "reset() must run before step()"
                raise RuntimeError(msg)
            feedback_text = next_obs
            if next_image is not None:
                processor = self._require_vision_processor()
                if self._transcript is None:
                    # Text-only so far: the ids hold no expanded image tokens.
                    transcript_text = self._decode(
                        full_ids[0].tolist(), skip_special_tokens=False
                    )
                    transcript_engine_ids = full_ids
                else:
                    transcript_text, transcript_engine_ids = self._transcript
                # A sampled turn that spells the placeholder would be expanded as
                # another image, and its trained ids cannot change: end here.
                if transcript_text.count(IMAGE_PLACEHOLDER) != len(
                    self._episode_images
                ):
                    truncated = True
                else:
                    turn_text = self._image_feedback_turn_text(
                        feedback_text, next_role, int(full_ids[0, -1])
                    )
                    # Only the new turn is tokenized: decode/encode does not
                    # round-trip every id sequence the model samples.
                    turn_ids, turn_pixel_values = encode_image_training_inputs(
                        text=turn_text,
                        image=next_image,
                        processor=processor,
                    )
                    train_ids = torch.cat(
                        [full_ids, turn_ids.to(full_ids.device)], dim=1
                    )
                    engine_ids = torch.cat(
                        [
                            transcript_engine_ids,
                            self._engine_ids(turn_text).to(
                                transcript_engine_ids.device
                            ),
                        ],
                        dim=1,
                    )
                    prompt_token_len = int(train_ids.shape[-1])
                    max_pt = self._prompt_budget()
                    over_images = (
                        self._segment_max_images is not None
                        and len(self._episode_images) + 1 > self._segment_max_images
                    )
                    restarted = (
                        self._over_segment_tokens(prompt_token_len) or over_images
                    ) and self._restart_context(full_ids, feedback_text, next_image)
                    if restarted:
                        pass
                    elif over_images:
                        msg = (
                            f"Turn {self._turn_idx} needs a context restart to stay "
                            f"within segment_max_images={self._segment_max_images}, "
                            "but the restarted prompt is over the prompt budget."
                        )
                        raise RuntimeError(msg)
                    elif max_pt is not None and prompt_token_len > max_pt:
                        truncated = True
                    else:
                        self._episode_images.append(next_image)
                        if self._episode_pixel_values is not None:
                            turn_pixel_values = torch.cat(
                                [
                                    self._episode_pixel_values,
                                    turn_pixel_values.to(
                                        self._episode_pixel_values.device
                                    ),
                                ],
                                dim=0,
                            )
                        self._episode_pixel_values = turn_pixel_values
                        self._multimodal_turn = {
                            "prompt": transcript_text + turn_text,
                            "image": self._prompt_image_payload(),
                            "prompt_token_len": prompt_token_len,
                            "input_ids": train_ids,
                            "prompt_token_ids": engine_ids,
                        }
                        self.full_ids = None
            else:
                self._transcript = None
                self._sampled_ids = None
                feedback_ids = self._tokenize_feedback(feedback_text, next_role).to(
                    full_ids.device
                )
                # The transcript keeps the sampled end-of-turn token (it is trained),
                # so drop the boundary frame's duplicate terminator when both are
                # present; a turn truncated at max_tokens still gets the frame's one.
                if (
                    feedback_ids.shape[1] > 1
                    and int(full_ids[0, -1]) == int(feedback_ids[0, 0])
                    and int(feedback_ids[0, 0]) in self._special_ids()
                ):
                    feedback_ids = feedback_ids[:, 1:]
                self.full_ids = torch.cat([full_ids, feedback_ids], dim=1)

                prompt_len = int(self.full_ids.shape[1])
                restarted = self._over_segment_tokens(
                    prompt_len
                ) and self._restart_context(full_ids, feedback_text, None)
                if (
                    not restarted
                    and (max_pt := self._prompt_budget()) is not None
                    and prompt_len > max_pt
                ):
                    truncated = True

            if not truncated:
                prompt = self._policy_prompt_from_state()

        self.done = bool(terminated or truncated)
        self.current_prompt = prompt
        return prompt, reward, terminated, truncated, info

    def _over_segment_tokens(self, prompt_len: int) -> bool:
        """Whether a ``prompt_len``-token prompt restarts the context."""
        return (
            self._segment_prompt_tokens is not None
            and prompt_len > self._segment_prompt_tokens
        )

    def _restart_context(
        self, segment_ids: torch.Tensor, obs_text: str, image: object | None
    ) -> bool:
        """Close the current segment and start a fresh prompt from the action history.

        :param segment_ids: The current segment's ids through its last generation.
        :param obs_text: The env's rendered observation that opens the new segment.
        :param image: That observation's image, or ``None``.
        :return: ``False``, with no state changed, when the new prompt is over budget.
        """
        body = obs_text.removeprefix(IMAGE_USER_CONTENT_PREFIX)
        # A spelled placeholder in an action would be expanded as another image.
        actions = "\n".join(
            f"{index}. {action.replace(IMAGE_PLACEHOLDER, ESCAPED_IMAGE_PLACEHOLDER)}"
            for index, action in enumerate(self._action_history, start=1)
        )
        restart_text = (
            f"{obs_text[: len(obs_text) - len(body)]}"
            f"Previous actions:\n{actions}\n\n{body}"
        )
        multimodal_turn: dict[str, Any] | None = None
        pixel_values: torch.Tensor | None = None
        if image is None:
            prompt_ids = self._tokenize_initial_prompt(restart_text)
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
        max_pt = self._prompt_budget()
        if max_pt is not None and int(prompt_ids.shape[-1]) > max_pt:
            return False

        self._segments.append(
            (segment_ids, self.turn_boundaries, self._episode_pixel_values)
        )
        self.turn_boundaries = []
        self._transcript = None
        self._sampled_ids = None
        self._episode_images = [image] if image is not None else []
        self._episode_pixel_values = pixel_values
        self._multimodal_turn = multimodal_turn
        self.full_ids = prompt_ids if multimodal_turn is None else None
        return True

    def step(
        self,
        token_ids: torch.Tensor,
        sampling_logps: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Decode the generation, round-trip the env, and apply the result.

        Single-env convenience over the ``_step_*`` phases; ``sampling_logps`` is this
        turn's vLLM logprobs (or ``None``).
        """
        gen_text = self._step_prepare(token_ids, sampling_logps)
        return self._step_apply(self._step_env(gen_text))

    def get_episode_data(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        EpisodeSegments | None,
    ]:
        """Build the episode row for training, its segments back to back.

        The position predicting each later segment's first token is masked out.

        :return: ``full_ids``, ``action_mask``, ``turn_ids``, ``turn_rewards``,
            ``sampling_logps``, ``pixel_values``, and ``segments`` (``None``
            unless the context restarted).
        """
        if self.full_ids is None:
            if self._multimodal_turn is not None:
                msg = "No episode data: episode has no completion yet"
                raise RuntimeError(msg)
            msg = "No episode data: reset() was never called"
            raise RuntimeError(msg)

        segments = [
            *self._segments,
            (self.full_ids, self.turn_boundaries, self._episode_pixel_values),
        ]
        full_ids = (
            self.full_ids
            if len(segments) == 1
            else torch.cat([ids for ids, _bounds, _pixels in segments], dim=1)
        )
        seq_len = full_ids.shape[1]
        action_mask = torch.zeros(1, seq_len - 1, dtype=torch.bool)
        turn_ids = torch.full((1, seq_len - 1), -1, dtype=torch.long)

        offset = 0
        for ids, boundaries, _pixels in segments:
            for gen_start, gen_end, tidx in boundaries:
                mask_start = offset + gen_start - 1
                mask_end = offset + gen_end - 1
                if mask_start >= 0 and mask_end <= seq_len - 1:
                    action_mask[0, mask_start:mask_end] = True
                    turn_ids[0, mask_start:mask_end] = tidx
            offset += int(ids.shape[1])

        turn_rewards = list(self.turn_rewards)
        while len(turn_rewards) < self.max_turns:
            turn_rewards.append(0.0)

        pixel_values = self._episode_pixel_values
        layout: EpisodeSegments | None = None
        if len(segments) > 1:
            pixel_parts = [pixels for _ids, _bounds, pixels in segments]
            present = [pixels for pixels in pixel_parts if pixels is not None]
            pixel_values = torch.cat(present, dim=0) if present else None
            layout = EpisodeSegments(
                token_lengths=torch.tensor(
                    [int(ids.shape[1]) for ids, _bounds, _pixels in segments],
                    dtype=torch.long,
                ),
                pixel_rows=None
                if pixel_values is None
                else torch.tensor(
                    [
                        0 if pixels is None else int(pixels.shape[0])
                        for pixels in pixel_parts
                    ],
                    dtype=torch.long,
                ),
            )
            validate_episode_segments(layout, int(seq_len))

        return (
            full_ids,
            action_mask,
            turn_ids,
            torch.tensor(turn_rewards, dtype=torch.float),
            torch.cat(self.sampling_logps) if self.sampling_logps else None,
            pixel_values,
            layout,
        )

    def close(self) -> None:
        """Close the env client, releasing whatever backend it owns."""
        self._env_client.close()
