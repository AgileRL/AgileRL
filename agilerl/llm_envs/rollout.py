# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-level rollout envs for generative LLM tasks over an env client.

``RolloutHarness`` runs the tokenisation + turn loop for one episode; ``RolloutCollector``
steps a batch of them in lock-step, a shared ``TaskAssigner`` giving each GRPO group
a common task (dataset row or reset seed).
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from agilerl.llm_envs.env_specs import is_url, spec_to_factory
from agilerl.protocols import EnvClientProtocol, TextEnvProtocol
from agilerl.utils.algo_utils import is_str_keyed_dict
from agilerl.utils.llm_utils import (
    is_rollout_prompts,
    max_prompt_tokens_for_model_len,
)

__all__ = [
    "RolloutCollector",
    "RolloutHarness",
    "TaskAssigner",
    "process_observation",
]

if TYPE_CHECKING:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.rubrics.base import Rubric
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from agilerl.typing import RolloutPrompts


def process_observation(obs: object, observation_field: str | None = None) -> str:
    """Render an OpenEnv observation payload to prompt text.

    The default observation processor. A named ``observation_field`` wins: an
    env that carries its text somewhere only its author knows should be told,
    not guessed at. Otherwise the two specified shapes are tried — an MCP tool
    result and our own ``TextObservation`` — then the one non-bookkeeping text
    field the payload carries, which warns, naming the field to set. Warns
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
    if isinstance(result, dict):
        raw_blocks = result.get("content")
        blocks = raw_blocks if isinstance(raw_blocks, list) else []
        texts: list[str] = []
        for block in blocks:
            if not is_str_keyed_dict(block):
                continue
            block_text = block.get("text")
            if isinstance(block_text, str):
                texts.append(block_text)
        if texts:
            return "\n".join(texts)
        data = result.get("data")
        if isinstance(data, str):
            return data
    prompt = obs.get("prompt")
    if isinstance(prompt, str):
        return prompt
    candidates = [
        (key, value)
        for key, value in obs.items()
        if isinstance(value, str)
        and value.strip()
        and key
        not in {
            "done",
            "episode_id",
            "error",
            "id",
            "metadata",
            "result",
            "reward",
            "rubric_scores",
            "step_count",
            "terminated",
            "tool_name",
            "truncated",
            "type",
        }
    ]
    if len(candidates) == 1:
        found, found_text = candidates[0]
        warnings.warn(
            f"OpenEnv observation matched no specified shape; read its text "
            f"from '{found}', the only field carrying any. Set "
            f"observation_field='{found}' in the manifest to make that "
            "explicit.",
            stacklevel=2,
        )
        return found_text
    error = obs.get("error")
    if error:
        return f"Error: {error}"
    asked = (
        f" No '{observation_field}' either, which the manifest named."
        if observation_field is not None
        else ""
    )
    warnings.warn(
        "OpenEnv observation carried no text the episode could use, so it gets "
        f"an empty prompt. Keys present: {sorted(obs)}. Expected a 'prompt' "
        "(TextObservation), a 'result' (MCP tool call), or exactly one other "
        "field carrying text. An observation with no text fields at all (board "
        f"states, screens) needs an observation_processor to render it.{asked}",
        stacklevel=2,
    )
    return ""


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
        apply_chat_template: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_model_len: int | None = None,
        max_output_tokens: int | None = None,
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
        :param apply_chat_template: Render prompts through the chat template vs raw encoding.
        :param chat_template_kwargs: Extra kwargs for every ``apply_chat_template``
            render (e.g. ``{"enable_thinking": False}``); the env's tool schemas
            are set on top under ``tools``.
        :param max_model_len: Engine context length; enables stop-on-overflow when set.
        :param max_output_tokens: Generation budget reserved under ``max_model_len``.
        :ivar full_ids: Running episode token sequence (prompt + generations + feedback).
        :ivar turn_boundaries: ``(start, end, turn_idx)`` spans of policy-generated tokens.
        :ivar turn_rewards: Per-turn rewards from the env.
        :ivar done: Whether the episode has terminated.
        :ivar current_prompt: Latest policy-ready observation; ``{}`` once done.
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
            # Lazy: OpenEnv backend needs the optional ``openenv`` package.
            from agilerl.llm_envs.openenv import RemoteEnvClient

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
        self._max_output_tokens = max_output_tokens
        self.full_ids: torch.Tensor | None = None
        self.turn_boundaries: list[tuple[int, int, int]] = []
        self.turn_rewards: list[float] = []
        self.rubric_score_sums: dict[str, float] = {}
        self._turn_idx = 0
        self._prompt_text: str = ""
        self._gen_texts: list[str] = []
        self._feedback_texts: list[str] = []
        self._last_full_prompt_token_len: int | None = None
        # Cached chat-template frame around a feedback turn (rendered once).
        self._boundary_parts: tuple[str, str] | None = None
        self._boundary_parts_known = False
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
        from agilerl.llm_envs.openenv import InProcessEnvClient

        env_client = InProcessEnvClient(env, action_field=action_field)
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
        :param env_config: Kwargs for the factory / entrypoint (ignored for a URL).
        :param tokenizer: Tokenizer for the token-level loop.
        :param max_turns: Generation turns per episode.
        :param kwargs: Forwarded to :class:`RolloutHarness`.
        :rtype: RolloutHarness
        """
        if isinstance(spec, str):
            if is_url(spec):
                return cls(spec, tokenizer, max_turns=max_turns, **kwargs)
            env = spec_to_factory(spec)(**(env_config or {}))
        else:
            env = spec(**(env_config or {}))
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
        :class:`~agilerl.llm_envs.prompt_dataset.PromptDatasetEnv` (``max_turns`` fixed
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
        from agilerl.llm_envs.prompt_dataset import PromptDatasetEnv
        from agilerl.llm_envs.rubrics import _require_rubric

        if "max_turns" in kwargs:
            msg = "from_dataset builds a single-turn env; max_turns is fixed to 1"
            raise TypeError(msg)
        env = PromptDatasetEnv(
            dataset,
            rubric=_require_rubric(rubric),
            test_dataset=test_dataset,
            prompt_builder=prompt_builder,
            question_column=question_column,
            answer_column=answer_column,
        )
        return cls.local(env, tokenizer, max_turns=1, **kwargs)

    def _tokenize_initial_prompt(self, obs_text: str) -> torch.Tensor:
        """Tokenize the initial observation, optionally with chat template."""
        if self.apply_chat_template:
            chat_template_kwargs = dict(self.chat_template_kwargs)
            if self.tools is not None:
                chat_template_kwargs["tools"] = self.tools
            result: Any = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": obs_text}],
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

    def _tokenize_feedback(self, feedback_text: str) -> torch.Tensor:
        """Tokenize the feedback turn via the cached chat-template frame (ChatML fallback)."""
        if not self.apply_chat_template:
            return torch.tensor(
                [self.tokenizer.encode(feedback_text, add_special_tokens=False)],
                dtype=torch.long,
            )

        boundary_ids = self._chat_template_boundary_ids(feedback_text)
        if boundary_ids is not None:
            return boundary_ids

        warnings.warn(
            "The tokenizer's chat template could not render a feedback turn "
            "boundary; falling back to ChatML markers (<|im_end|>/<|im_start|>). "
            "For a non-ChatML tokenizer the multi-turn transcript will be "
            "malformed.",
            stacklevel=2,
        )
        # Fallback: ChatML-style markers.
        turn_boundary = (
            "<|im_end|>\n<|im_start|>user\n"
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

    def _feedback_boundary_parts(self) -> tuple[str, str] | None:
        """Cached ``(prefix, suffix)`` the chat template wraps a feedback turn in.

        Sliced at two uuid4 placeholders (render verbatim, can't collide); the dummy
        user message satisfies strict-alternation templates. ``None`` -> ChatML fallback.
        """
        if self._boundary_parts_known:
            return self._boundary_parts
        self._boundary_parts_known = True
        assistant_ph = uuid.uuid4().hex
        feedback_ph = uuid.uuid4().hex
        messages = [
            {"role": "user", "content": "."},
            {"role": "assistant", "content": assistant_ph},
            {"role": "user", "content": feedback_ph},
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
        self._boundary_parts = (prefix, suffix)
        return self._boundary_parts

    def _chat_template_boundary_ids(
        self,
        feedback_text: str,
    ) -> torch.Tensor | None:
        """Token ids for the templated turn boundary carrying ``feedback_text`` (``None`` if no frame)."""
        parts = self._feedback_boundary_parts()
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
        return max_prompt_tokens_for_model_len(
            self._max_model_len,
            self._max_output_tokens,
        )

    def _policy_observation_from_state(self) -> dict[str, Any]:
        """Build observation dict for ``get_action`` from current ``full_ids``."""
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
        """Create a fresh episode and return the policy-ready observation plus info.

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
            return self.current_prompt, info

        self.done = False
        self.current_prompt = self._policy_observation_from_state()
        return self.current_prompt, info

    def _step_prepare(
        self,
        full_completion_ids: torch.Tensor,
        sampling_logps: torch.Tensor | None = None,
    ) -> str:
        """Decode and record this turn's generation; return its text (tokenizer phase)."""
        if self._last_full_prompt_token_len is None or self.full_ids is None:
            msg = (
                "step() requires a prior reset() or step() "
                "that built a input context observation"
            )
            raise RuntimeError(msg)
        prompt_len = self._last_full_prompt_token_len
        full_ids = self.full_ids
        completion = (
            full_completion_ids
            if full_completion_ids.dim() > 1
            else full_completion_ids.unsqueeze(0)
        )
        # Only the new suffix crosses devices; the prefix is byte-identical to ``full_ids``.
        gen_ids = completion[0, prompt_len:].detach().to(full_ids.device)
        gen_text = self.tokenizer.decode(
            gen_ids.tolist(),
            skip_special_tokens=True,
        )
        assert isinstance(gen_text, str), "decode() of one sequence returns str"
        if sampling_logps is not None:
            self.sampling_logps.append(sampling_logps)
        self.full_ids = torch.cat([full_ids, gen_ids.unsqueeze(0)], dim=1)
        gen_end = self.full_ids.shape[1]
        self.turn_boundaries.append((prompt_len, gen_end, self._turn_idx))
        self._gen_texts.append(gen_text)
        return gen_text

    def _step_env(self, gen_text: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Round-trip the env backend and render its observation — the parallelizable phase."""
        payload, reward, terminated, truncated, info = self._env_client.step(gen_text)
        return self._render_observation(payload), reward, terminated, truncated, info

    def _step_apply(
        self,
        env_result: tuple[str, float, bool, bool, dict[str, Any]],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply the env round-trip result: rewards, truncation, feedback tokens."""
        next_obs, reward, terminated, truncated, info = env_result
        self.turn_rewards.append(float(reward))
        for name, value in (info.get("rubric_scores") or {}).items():
            self.rubric_score_sums[name] = self.rubric_score_sums.get(
                name, 0.0
            ) + float(value)
        self._turn_idx += 1

        if not (terminated or truncated) and self._turn_idx >= self.max_turns:
            truncated = True

        prompt_dict: dict[str, Any] = {}
        if not (terminated or truncated):
            full_ids = self.full_ids
            assert full_ids is not None, "reset() must run before step()"
            feedback_text = next_obs
            self._feedback_texts.append(feedback_text)
            feedback_ids = self._tokenize_feedback(feedback_text).to(full_ids.device)
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
                prompt_dict = self._policy_observation_from_state()

        self.done = bool(terminated or truncated)
        self.current_prompt = prompt_dict
        return prompt_dict, reward, terminated, truncated, info

    def step(
        self,
        full_completion_ids: torch.Tensor,
        sampling_logps: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Decode the generation, round-trip the env, and apply the result.

        Single-env convenience over the ``_step_*`` phases; ``sampling_logps`` is this
        turn's vLLM logprobs (or ``None``).
        """
        gen_text = self._step_prepare(full_completion_ids, sampling_logps)
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


def _mix_seed(value: int) -> int:
    """Spread a task seed via splitmix64: deterministic, and injective over 64-bit inputs."""
    z = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (z ^ (z >> 31)) & ((1 << 63) - 1)


class TaskAssigner:
    """Assign each episode a ``(seed, row_index)`` task; a GRPO group shares one.

    Dataset rows are reshuffled each epoch and split into one equal shard per
    data-parallel rank; a procedural env has no rows and is seeded instead.

    :param dataset_size: Rows in the env's dataset; ``0`` for a procedural env.
    :param seed: Seed for the per-epoch shuffle (``None`` -> a fixed default).
    :param rank: This process's shard index in ``[0, world_size)``.
    :param world_size: Number of data-parallel shards (``1`` = no sharding).
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Build an assigner over this rank's shard with a seeded per-epoch shuffle."""
        if world_size < 1:
            msg = f"world_size must be >= 1, got {world_size}."
            raise ValueError(msg)
        if not 0 <= rank < world_size:
            msg = f"rank must be in [0, {world_size}), got {rank}."
            raise ValueError(msg)
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._shard_size = self.dataset_size // self.world_size
        self._shard_start = self.rank * self._shard_size
        if self.dataset_size > 0 and self._shard_size == 0:
            msg = (
                f"rank {rank} of {world_size} gets an empty shard of a "
                f"{dataset_size}-row dataset; reduce world_size."
            )
            raise ValueError(msg)
        #: Completed full passes over this rank's shard of the dataset rows.
        self.num_epochs = 0
        self._generator = torch.Generator().manual_seed(
            seed if seed is not None else 42
        )
        self._epoch_order: list[int] = []
        self._pos = 0

    def next_row(self) -> int:
        """Next row from the epoch-reshuffled shard stream (bumps :attr:`num_epochs`)."""
        if self._pos >= len(self._epoch_order):  # epoch boundary (and first call)
            if self._epoch_order:
                self.num_epochs += 1
            self._epoch_order = (
                torch.randperm(self._shard_size, generator=self._generator)
                + self._shard_start
            ).tolist()
            self._pos = 0
        row = self._epoch_order[self._pos]
        self._pos += 1
        return row

    def assign(
        self,
        batch_size: int,
        group_size: int,
        *,
        base_seed: int | None = None,
        seed_offset: int = 0,
    ) -> list[tuple[int | None, int | None]]:
        """Tasks for one batch: ``batch_size * group_size`` ``(seed, row_index)`` pairs.

        Item ``i``'s seed is ``base_seed + seed_offset + i`` spread through
        :func:`_mix_seed`, so consecutive windows land far apart in an env's
        seed space rather than walking it linearly — an env whose seed-to-task
        mapping has short-period structure would otherwise recycle tasks on a
        fixed cycle. The pair is repeated ``group_size`` times so the whole
        group shares one task. Callers must keep group seeds unique across
        batches (advance ``base_seed`` or ``seed_offset``).
        """
        out: list[tuple[int | None, int | None]] = []
        for item in range(batch_size):
            seed = (
                None
                if base_seed is None
                else _mix_seed(int(base_seed) + int(seed_offset) + item)
            )
            row = self.next_row() if self.dataset_size > 0 else None
            out.extend([(seed, row)] * group_size)
        return out


class RolloutCollector:
    """Batched in-process collector over ``batch_size * group_size`` :class:`RolloutHarness` slots.

    Driven lock-step (``reset``/``step``/``get_trajectories``, the colocated path,
    single-caller) or per-episode (``reset_episode``/``step_episode``/
    ``finalize_episode``, the async path — a slot is held from reset to finalize).
    The per-episode methods are synchronous and thread-safe; asyncio callers
    offload them via ``asyncio.to_thread``. A ``step_episode`` whose episode is
    finalized while its env round-trip is in flight raises rather than applying
    the stale result to the slot's next episode.
    """

    def __init__(
        self,
        env_factory: Callable[..., RolloutHarness],
        batch_size: int,
        group_size: int,
        env_config: dict[str, Any] | None = None,
        *,
        io_timeout_s: float | None = 600.0,
        base_seed: int | None = None,
        slot_acquire_timeout_s: float | None = 300.0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Create ``batch_size * group_size`` independent env wrappers.

        :param env_factory: Factory that builds one :class:`RolloutHarness`.
        :param batch_size: Number of logical batch items.
        :param group_size: Grouped rollouts per batch item.
        :param env_config: Optional kwargs passed to ``env_factory``.
        :param io_timeout_s: Deadline (seconds) for one concurrent round of env round-trips
            (raises ``TimeoutError`` rather than blocking); the only bound on an in-process
            env. ``None`` disables it; keep it above any per-env ``timeout_s``.
        :param base_seed: Base seed for the per-episode path's task assignment (the
            lock-step path takes its seed per ``reset`` call).
        :param slot_acquire_timeout_s: How long ``reset_episode`` waits for a free slot
            before raising ``TimeoutError``; ``None`` waits forever.
        :param rank: This process's data-parallel shard index, handed to the
            :class:`TaskAssigner` (from the runtime, never manifest config).
        :param world_size: Number of data-parallel shards.
        """
        if batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}."
            raise ValueError(msg)
        if group_size <= 0:
            msg = f"group_size must be > 0, got {group_size}."
            raise ValueError(msg)
        if env_config is None:
            env_config = {}
        self.env_factory = env_factory
        self.env_config = env_config
        self.num_envs = batch_size * group_size
        self.batch_size = batch_size
        self.group_size = group_size
        self._io_timeout_s = io_timeout_s
        self.envs: list[RolloutHarness] = []
        # Hands each group its task (dataset row / seed); built on first reset.
        self._task_assigner: TaskAssigner | None = None
        # Component key set for ``reward_*`` metrics, frozen when the envs are built.
        self.rubric_component_names: tuple[str, ...] = ()
        self._rank = int(rank)
        self._world_size = int(world_size)
        # --- per-episode state (untouched by the lock-step path) ---
        self._base_seed = int(base_seed) if base_seed is not None else None
        self._slot_acquire_timeout_s = slot_acquire_timeout_s
        self._seed_offset = 0
        # One (seed, row_index) per logical slot, built lazily per rollout window.
        self._assignment: list[tuple[int | None, int | None]] | None = None
        self._free_slots: queue.Queue[int] | None = None
        self._episode_to_slot: dict[str, int] = {}
        # Monotonic per-slot activation tokens: bumped each time a slot is
        # (re)assigned, so a step that outlives its episode is detectable.
        self._slot_activations: list[int] = [0] * self.num_envs
        # Guards the slot queue + episode map (mutated from many caller threads);
        # re-entrant because a failed _ensure_slots calls close() while holding it.
        self._slot_lock = threading.RLock()
        # Serializes tokenizer work and env-state mutation across caller threads;
        # env I/O runs outside it so round-trips still overlap.
        self._tokenizer_lock = threading.Lock()

    @property
    def _is_initialized(self) -> bool:
        """``True`` once every env slot has been created."""
        return len(self.envs) == self.num_envs

    @property
    def num_epochs(self) -> int:
        """Completed passes over the dataset rows (``0`` when not dataset-backed)."""
        return self._task_assigner.num_epochs if self._task_assigner is not None else 0

    def _active_envs(self) -> list[RolloutHarness]:
        """Non-terminal envs, in their stable batch/group (list) order."""
        return [env for env in self.envs if not env.done]

    def _get_prompts(self) -> list[RolloutPrompts] | None:
        """Prompts for active envs, or ``None`` when all are terminal."""
        active = self._active_envs()
        if not active:
            return None
        prompts: list[RolloutPrompts] = []
        for env in active:
            obs = env.current_prompt
            assert is_rollout_prompts(obs), "an active env always holds a prompt"
            prompts.append(obs)
        return prompts

    def reset(
        self,
        seed: int | None = None,
    ) -> list[RolloutPrompts] | None:
        """Reset all env wrappers, building them on the first call.

        :meth:`TaskAssigner.assign` picks each group's task; prompts return
        in stable list order. If building envs fails partway, the built ones are closed
        and the batch left empty so a retried ``reset`` starts clean.

        :param seed: Optional base seed for deterministic rollouts.
        :return: Active prompt dictionaries after reset.
        """
        self._build_envs_and_assigner(seed)
        assert self._task_assigner is not None
        assignments = self._task_assigner.assign(
            self.batch_size, self.group_size, base_seed=seed
        )
        # Phase 1 (concurrent — pure I/O): fetch every env's initial prompt.
        fetches = self._map_env_io(
            [
                partial(env._reset_fetch, env_seed, row_index=row)
                for env, (env_seed, row) in zip(self.envs, assignments, strict=False)
            ]
        )
        # Phase 2 (sequential — tokenizer): apply each prompt.
        for env, (obs_text, info) in zip(self.envs, fetches, strict=False):
            env._reset_apply(obs_text, info)

        return self._get_prompts()

    def get_rubric_score_means(self) -> dict[str, float]:
        """Mean per-episode component sums over the frozen component key set.

        A component missing from every episode reports ``nan`` rather than ``0``,
        so a never-scored criterion is not logged as a zero reward.
        """
        means: dict[str, float] = {}
        for name in self.rubric_component_names:
            values = [
                env.rubric_score_sums[name]
                for env in self.envs
                if name in env.rubric_score_sums
            ]
            means[name] = sum(values) / len(values) if values else float("nan")
        return means

    def step(
        self,
        completion_ids: list[torch.Tensor],
        sampling_logps: list[torch.Tensor | None] | None = None,
    ) -> list[RolloutPrompts] | None:
        """Step each active env with its corresponding completion.

        :param completion_ids: One completion tensor per active env.
        :param sampling_logps: vLLM logprobs parallel to ``completion_ids``; entries or
            the whole list may be ``None``.
        :return: Next active prompt dictionaries after stepping.
        """
        active = self._active_envs()
        if len(completion_ids) != len(active):
            msg = (
                "Number of completions does not match number of active envs: "
                f"{len(completion_ids)} != {len(active)}"
            )
            raise RuntimeError(msg)
        if sampling_logps is not None and len(sampling_logps) != len(active):
            msg = (
                "Number of sampling logprobs does not match number of active "
                f"envs: {len(sampling_logps)} != {len(active)}"
            )
            raise RuntimeError(msg)
        slps = sampling_logps if sampling_logps is not None else [None] * len(active)
        # Phase 1 (sequential — tokenizer): decode + record each generation.
        gen_texts = [
            env._step_prepare(full, sampling_logps=slp)
            for env, full, slp in zip(active, completion_ids, slps, strict=False)
        ]
        # Phase 2 (concurrent — pure I/O): round-trip each env backend at once.
        results = self._map_env_io(
            [
                partial(env._step_env, gt)
                for env, gt in zip(active, gen_texts, strict=False)
            ]
        )
        # Phase 3 (sequential — tokenizer): apply each result to its own env.
        for env, result in zip(active, results, strict=False):
            env._step_apply(result)
        return self._get_prompts()

    def _map_env_io(self, thunks: list[Callable[[], Any]]) -> list[Any]:
        """Run each zero-arg thunk on its own daemon thread, returning results in order.

        Bounded by ``io_timeout_s``: on completion the first thunk exception (in
        order) propagates; past the deadline ``TimeoutError`` is raised and the
        stragglers abandoned (daemon threads, so a hung env never occupies a shared
        pool or blocks interpreter exit). A single thunk runs inline only when
        unbounded.
        """
        if not thunks:
            return []
        if self._io_timeout_s is None and len(thunks) == 1:
            return [thunks[0]()]
        results: list[Any] = [None] * len(thunks)
        errors: list[BaseException | None] = [None] * len(thunks)
        finished = threading.Semaphore(0)

        def run(index: int, thunk: Callable[[], Any]) -> None:
            try:
                results[index] = thunk()
            except BaseException as exc:  # repropagated on the caller thread
                errors[index] = exc
            finally:
                finished.release()

        for index, thunk in enumerate(thunks):
            threading.Thread(
                target=run,
                args=(index, thunk),
                name=f"rollout-env-io-{index}",
                daemon=True,
            ).start()
        deadline = (
            None
            if self._io_timeout_s is None
            else time.monotonic() + self._io_timeout_s
        )
        for done_count in range(len(thunks)):
            remaining = None if deadline is None else deadline - time.monotonic()
            if not finished.acquire(timeout=remaining):
                msg = (
                    f"{len(thunks) - done_count}/{len(thunks)} env round-trips did "
                    f"not finish within io_timeout_s={self._io_timeout_s}s; a hung "
                    "env or a stalled transport blocked the batch."
                )
                raise TimeoutError(msg)
        for error in errors:
            if error is not None:
                raise error
        return results

    def close(self) -> None:
        """Close every env wrapper and clear the slot state; a later reset rebuilds."""
        with self._slot_lock:
            self._episode_to_slot.clear()
            self._free_slots = None
        for env in self.envs:
            env.close()
        self.envs = []

    def get_trajectories(
        self,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        int,
        list[torch.Tensor | None] | None,
    ]:
        """Collect complete episode tensors from all envs, in list order.

        :return: ``(completion_ids_list, action_masks_list, all_turn_ids, all_rewards,
            batch_steps, all_sampling_logps)`` where ``batch_steps`` is the summed turn
            count. ``all_sampling_logps`` is ``None`` when no vLLM logprobs were captured,
            else one 1-D tensor per env (``None`` for envs that captured none).
        """
        completion_ids_list: list[torch.Tensor] = []
        action_masks_list: list[torch.Tensor] = []
        all_turn_ids: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        all_sampling_logps: list[torch.Tensor | None] = []
        batch_steps = 0
        for env in self.envs:
            (
                ep_ids,
                action_mask,
                turn_ids,
                turn_rewards_t,
                sampling_logps,
            ) = env.get_episode_data()
            completion_ids_list.append(ep_ids)
            action_masks_list.append(action_mask)
            all_turn_ids.append(turn_ids)
            all_rewards.append(turn_rewards_t)
            batch_steps += len(env.turn_boundaries)
            all_sampling_logps.append(sampling_logps)

        return (
            completion_ids_list,
            action_masks_list,
            all_turn_ids,
            all_rewards,
            batch_steps,
            # Collapse to a single ``None`` when nothing was captured.
            (
                all_sampling_logps
                if any(logps is not None for logps in all_sampling_logps)
                else None
            ),
        )

    # --- per-episode API (the async path) ------------------------------------

    def set_group_seed(self, group_seed: int) -> None:
        """Start a rollout window: set the grouped seed offset and invalidate the assignment.

        Group seeds must stay unique across windows, so the caller advances
        ``group_seed`` between windows while :attr:`_base_seed` stays fixed
        (see :meth:`TaskAssigner.assign`).
        """
        self._seed_offset = int(group_seed)
        self._assignment = None

    def update_rollout_geometry(
        self,
        *,
        rollout_batch_size: int,
        group_size: int,
    ) -> None:
        """Change the window's batch/group split without recreating envs.

        Capacity-capped: the env list built at construction is the slot pool, so the
        new window may not exceed ``num_envs`` episodes.
        """
        with self._slot_lock:
            if self._episode_to_slot:
                msg = "cannot update rollout geometry while episodes are still active"
                raise RuntimeError(msg)
        new_batch_size = max(1, int(rollout_batch_size))
        new_group_size = max(1, int(group_size))
        if new_batch_size * new_group_size > self.num_envs:
            msg = (
                f"requested rollout geometry ({new_batch_size} x {new_group_size} = "
                f"{new_batch_size * new_group_size} episodes) exceeds the slot pool "
                f"of {self.num_envs}; growing beyond the initial allocation requires "
                "recreating the collector"
            )
            raise ValueError(msg)
        self.batch_size = new_batch_size
        self.group_size = new_group_size
        self._assignment = None

    def active_episode_count(self) -> int:
        """Count the episodes currently holding a slot (thread-safe)."""
        with self._slot_lock:
            return len(self._episode_to_slot)

    def active_episode_ids(self, max_ids: int = 16) -> list[str]:
        """Snapshot up to ``max_ids`` active episode IDs, for diagnostics (thread-safe)."""
        with self._slot_lock:
            return [
                str(episode_id) for episode_id in list(self._episode_to_slot)[:max_ids]
            ]

    def _build_envs_and_assigner(self, seed: int | None) -> None:
        """Build the env pool once and, with it, the task assigner seeded by ``seed``."""
        if not self._is_initialized:
            try:
                while len(self.envs) < self.num_envs:
                    self.envs.append(self.env_factory(**self.env_config))
            except Exception:
                self.close()
                raise
        if self._task_assigner is None:
            self._task_assigner = TaskAssigner(
                self.envs[0].dataset_size,
                seed=seed,
                rank=self._rank,
                world_size=self._world_size,
            )
            self.rubric_component_names = tuple(self.envs[0].rubric_components)

    def _ensure_slots(self) -> None:
        """Build the envs, the task assigner and the free-slot queue once (thread-safe)."""
        with self._slot_lock:
            if self._free_slots is not None:
                return
            self._build_envs_and_assigner(self._base_seed)
            free_slots: queue.Queue[int] = queue.Queue()
            for slot in range(self.num_envs):
                free_slots.put_nowait(slot)
            self._free_slots = free_slots

    def _episode_assignment(
        self,
        logical_slot: int | None,
    ) -> tuple[int | None, int | None]:
        """The window's ``(seed, row_index)`` for ``logical_slot`` (``(None, None)`` unpinned)."""
        if logical_slot is None:
            return None, None
        with self._slot_lock:
            assert self._task_assigner is not None, "_ensure_slots builds the assigner"
            if self._assignment is None:
                self._assignment = self._task_assigner.assign(
                    self.batch_size,
                    self.group_size,
                    base_seed=self._base_seed,
                    seed_offset=self._seed_offset,
                )
            if not 0 <= int(logical_slot) < len(self._assignment):
                msg = (
                    f"logical_slot {logical_slot} is outside the current rollout "
                    f"window of {len(self._assignment)} episodes."
                )
                raise IndexError(msg)
            return self._assignment[int(logical_slot)]

    def _slot_and_activation(self, episode_id: str) -> tuple[int, int]:
        """The ``(slot, activation)`` ``episode_id`` holds; ``KeyError`` when it is not active."""
        with self._slot_lock:
            slot = self._episode_to_slot.get(episode_id)
            if slot is None:
                msg = f"Episode {episode_id!r} is not active."
                raise KeyError(msg)
            return slot, self._slot_activations[slot]

    def _require_current(self, episode_id: str, slot: int, activation: int) -> None:
        """Raise unless ``episode_id`` still holds ``slot`` on the same activation."""
        with self._slot_lock:
            current_slot = self._episode_to_slot.get(episode_id)
            current_activation = self._slot_activations[slot]
        if current_slot != slot or current_activation != activation:
            msg = (
                f"Episode {episode_id!r} no longer owns its env slot: it was "
                "finalized (and the slot possibly reused) while this step was "
                "in flight."
            )
            raise RuntimeError(msg)

    def reset_episode(
        self,
        episode_id: str,
        logical_slot: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Acquire a free slot and reset one episode at its window-assigned task.

        Blocking and thread-safe — the offload target for an asyncio caller. The slot
        is held until :meth:`get_episode_data` / :meth:`finalize_episode`. ``logical_slot``
        indexes the window's group-contiguous ``(seed, row_index)`` assignment; ``None``
        leaves the task unpinned.

        :param episode_id: Caller-unique id naming this episode in later calls.
        :param logical_slot: Position in the rollout window, pinning the group task.
        :return: ``(prompt, info)`` — the policy-ready observation (empty when the
            episode truncated at turn 0).
        """
        self._ensure_slots()
        assert self._free_slots is not None
        seed, row_index = self._episode_assignment(logical_slot)
        try:
            slot = self._free_slots.get(timeout=self._slot_acquire_timeout_s)
        except queue.Empty:
            msg = (
                f"Timed out after {self._slot_acquire_timeout_s}s acquiring a free env "
                f"slot for episode {episode_id!r}; {self.num_envs} slots are all held "
                "by unfinalized episodes."
            )
            raise TimeoutError(msg) from None
        acquired = False
        try:
            env = self.envs[slot]
            obs_text, info = env._reset_fetch(seed, row_index=row_index)
            with self._tokenizer_lock:
                prompts, info = env._reset_apply(obs_text, info)
            with self._slot_lock:
                # Checked under the same lock as the insert, so racing duplicate
                # ids cannot both claim a slot.
                if episode_id in self._episode_to_slot:
                    msg = f"Episode {episode_id!r} is already active."
                    raise RuntimeError(msg)
                self._episode_to_slot[episode_id] = slot
                self._slot_activations[slot] += 1
            acquired = True
            return prompts, info
        finally:
            # A failed reset must release the slot it took so the pool is not starved.
            if not acquired:
                self._free_slots.put_nowait(slot)

    def step_episode(
        self,
        episode_id: str,
        completion_ids: torch.Tensor,
        sampling_logps: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one episode a turn: decode, round-trip its env, apply the result.

        Thread-safe; only the env round-trip runs outside the tokenizer lock, so
        interleaved episodes overlap on I/O and stay serial on tokenizer work.

        :param episode_id: The episode to step (from a prior :meth:`reset_episode`).
        :param completion_ids: This turn's full completion tensor.
        :param sampling_logps: This turn's vLLM sampling logprobs, or ``None``.
        :return: The :meth:`RolloutHarness.step` 5-tuple.
        :raises RuntimeError: If the episode was finalized (and its slot possibly
            reused) while the step was in flight — the stale result is never
            applied to the slot's successor episode.
        """
        slot, activation = self._slot_and_activation(episode_id)
        env = self.envs[slot]
        with self._tokenizer_lock:
            self._require_current(episode_id, slot, activation)
            gen_text = env._step_prepare(completion_ids, sampling_logps=sampling_logps)
        env_result = env._step_env(gen_text)
        with self._tokenizer_lock:
            self._require_current(episode_id, slot, activation)
            return env._step_apply(env_result)

    def get_episode_data(
        self,
        episode_id: str,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        """Build one episode's tensors and release its slot.

        :param episode_id: The episode to finalize; ``KeyError`` when not active.
        :return: The :meth:`RolloutHarness.get_episode_data` 5-tuple.
        """
        result = self.finalize_episode(episode_id, missing_ok=False)
        assert result is not None, "missing_ok=False raises instead of returning None"
        return result

    def finalize_episode(
        self,
        episode_id: str,
        *,
        missing_ok: bool = True,
    ) -> (
        tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
        ]
        | None
    ):
        """Finalize and release one episode slot exactly once.

        Idempotent when ``missing_ok`` — safe for cancellation cleanup paths that may
        race with normal finalize.
        """
        with self._slot_lock:
            slot = self._episode_to_slot.pop(episode_id, None)
        if slot is None:
            if missing_ok:
                return None
            msg = f"Episode {episode_id!r} is not active."
            raise KeyError(msg)
        try:
            with self._tokenizer_lock:
                return self.envs[slot].get_episode_data()
        finally:
            assert self._free_slots is not None, "an active episode implies slots exist"
            self._free_slots.put_nowait(slot)
