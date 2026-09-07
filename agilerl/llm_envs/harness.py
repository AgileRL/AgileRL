# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-level rollout env: tokenisation + turn loop over a text env client."""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from agilerl.llm_envs.env_specs import is_url, spec_to_factory
from agilerl.llm_envs.observation import (
    DEFAULT_OBSERVATION_ROLE,
    observation_role,
    process_observation,
)
from agilerl.protocols import EnvClientProtocol, TextEnvProtocol
from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len

__all__ = ["RolloutHarness"]

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
        :ivar full_ids: Running episode token sequence (prompt + generations + feedback).
        :ivar turn_boundaries: ``(start, end, turn_idx)`` spans of policy-generated tokens.
        :ivar turn_rewards: Per-turn rewards from the env.
        :ivar done: Whether the episode has terminated.
        :ivar current_prompt: Latest policy-ready prompt; ``{}`` once done.
        :ivar sampling_logps: Per-turn vLLM sampling logprobs (empty on the HF path).
        """
        self.max_turns = max_turns
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
            ):
                msg = (
                    "timeout_s / mcp_tool / action_field configure the transport "
                    "and have no effect on an already-built env client; set them "
                    "on the client instead."
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
        self._gen_texts: list[str] = []
        self._feedback_texts: list[str] = []
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
        **kwargs: Any,
    ) -> RolloutHarness:
        """Build a ``RolloutHarness`` from an env ``spec``.

        A **URL** is driven remotely; anything else is built with ``env_config``
        and driven in-process: an **env factory callable** directly, otherwise a
        ``module:attr`` / ``path.py:attr`` entrypoint naming a callable that
        returns a text env. For rows + a rubric instead of an env, see
        :meth:`from_dataset`.

        :param spec: A URL, an env factory callable, or a ``module:attr`` /
            ``path.py:attr`` entrypoint.
        :param env_config: Kwargs for the factory / entrypoint (ignored for a URL,
            except ``system_prompt``). ``system_prompt`` is set on the built env
            and passed to the harness so the chat template renders it.
        :param tokenizer: Tokenizer for the token-level loop.
        :param max_turns: Generation turns per episode.
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
        factory = spec_to_factory(spec) if isinstance(spec, str) else spec
        env = factory(**config)
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

    def _tokenize_initial_prompt(self, obs_text: str) -> torch.Tensor:
        """Tokenize the initial observation, optionally with chat template."""
        if self.apply_chat_template:
            chat_template_kwargs = dict(self.chat_template_kwargs)
            if self.tools is not None:
                chat_template_kwargs["tools"] = self.tools
            messages: list[dict[str, str]] = []
            if self._system_prompt is not None:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": obs_text})
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
        """Build the ``get_action`` prompt dict from the current ``full_ids``."""
        if self.full_ids is None:
            msg = "No prompt: reset() was never called"
            raise RuntimeError(msg)
        self._last_full_prompt_token_len = int(self.full_ids.shape[1])
        # Single unpadded row; HF-generate derives an all-ones mask from ``input_ids``.
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
        obs_text, info = self._reset_fetch(seed, row_index=row_index)
        return self._reset_apply(obs_text, info)

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
    ) -> tuple[str, dict[str, Any]]:
        """Pull and render the initial prompt from the env backend — the parallelizable I/O.

        No tokenizer work; touching :attr:`tools` warms its cache to overlap too.
        """
        _ = self.tools
        payload, info = self._env_client.reset(seed=seed, row_index=row_index)
        return (self._render_observation(payload) or self._instruction), info

    def _reset_apply(
        self,
        obs_text: str,
        info: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Tokenize the initial prompt and start the episode (truncates at turn 0 if over budget)."""
        self._adopt_system_prompt(info)
        self.full_ids = self._tokenize_initial_prompt(obs_text)
        self.turn_boundaries = []
        self.turn_rewards = []
        self.rubric_score_sums = {}
        self._turn_idx = 0
        self._prompt_text = obs_text
        self._gen_texts = []
        self._feedback_texts = []
        self.sampling_logps = []

        max_pt = self._prompt_budget()
        if max_pt is not None and int(self.full_ids.shape[1]) > max_pt:
            self.done = True
            self.current_prompt = {}
            # Step slices generation from this episode's tokenized prompt length.
            self._last_full_prompt_token_len = int(self.full_ids.shape[1])
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
        if self._last_full_prompt_token_len is None or self.full_ids is None:
            msg = "step() requires a prior reset() or step() that built a prompt"
            raise RuntimeError(msg)
        prompt_len = self._last_full_prompt_token_len
        full_ids = self.full_ids
        sequence = token_ids if token_ids.dim() > 1 else token_ids.unsqueeze(0)
        # Only the new suffix crosses devices; the prefix is byte-identical to ``full_ids``.
        gen_ids = sequence[0, prompt_len:].detach().to(full_ids.device)
        gen_text = self.tokenizer.decode(
            gen_ids.tolist(),
            skip_special_tokens=True,
        )
        if not isinstance(gen_text, str):
            msg = "decode() of one sequence returns str"
            raise TypeError(msg)
        if sampling_logps is not None:
            self.sampling_logps.append(sampling_logps)
        self.full_ids = torch.cat([full_ids, gen_ids.unsqueeze(0)], dim=1)
        gen_end = self.full_ids.shape[1]
        self.turn_boundaries.append((prompt_len, gen_end, self._turn_idx))
        self._gen_texts.append(gen_text)
        return gen_text

    def _step_env(
        self, gen_text: str
    ) -> tuple[str, str, float, bool, bool, dict[str, Any]]:
        """Round-trip the env backend and render its observation — the parallelizable phase.

        Carries the observation's chat role alongside its text so :meth:`_step_apply`
        frames a tool result as a tool turn rather than as something the user said.
        """
        payload, reward, terminated, truncated, info = self._env_client.step(gen_text)
        return (
            self._render_observation(payload),
            observation_role(payload, info),
            reward,
            terminated,
            truncated,
            info,
        )

    def _step_apply(
        self,
        env_result: tuple[str, str, float, bool, bool, dict[str, Any]],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply the env round-trip result: rewards, truncation, feedback tokens."""
        next_obs, next_role, reward, terminated, truncated, info = env_result
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
            self._feedback_texts.append(feedback_text)
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

            if (max_pt := self._prompt_budget()) is not None:
                prompt_len = int(self.full_ids.shape[1])
                if prompt_len > max_pt:
                    truncated = True

            if not truncated:
                prompt = self._policy_prompt_from_state()

        self.done = bool(terminated or truncated)
        self.current_prompt = prompt
        return prompt, reward, terminated, truncated, info

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
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        """Build the episode row ``(full_ids, action_mask, turn_ids, turn_rewards, sampling_logps)``.

        ``sampling_logps`` is the per-turn vLLM logprobs concatenated, or ``None`` (HF path).
        """
        if self.full_ids is None:
            msg = "No episode data: reset() was never called"
            raise RuntimeError(msg)

        seq_len = self.full_ids.shape[1]
        action_mask = torch.zeros(1, seq_len - 1, dtype=torch.bool)
        turn_ids = torch.full((1, seq_len - 1), -1, dtype=torch.long)

        for gen_start, gen_end, tidx in self.turn_boundaries:
            mask_start = gen_start - 1
            mask_end = gen_end - 1
            if mask_start >= 0 and mask_end <= seq_len - 1:
                action_mask[0, mask_start:mask_end] = True
                turn_ids[0, mask_start:mask_end] = tidx

        turn_rewards = list(self.turn_rewards)
        while len(turn_rewards) < self.max_turns:
            turn_rewards.append(0.0)

        return (
            self.full_ids,
            action_mask,
            turn_ids,
            torch.tensor(turn_rewards, dtype=torch.float),
            torch.cat(self.sampling_logps) if self.sampling_logps else None,
        )

    def close(self) -> None:
        """Close the env client, releasing whatever backend it owns."""
        self._env_client.close()
