"""Token-level rollout env for generative LLM tasks (multi-turn agentic or single-turn reasoning).

A :class:`RolloutEnv` owns the tokenisation + turn loop and talks to its env through an
**env client** exposing the OpenEnv ``reset`` / ``step`` calls. The env client is
either an :class:`~agilerl.llm_envs.openenv.OpenEnvClient` (the env is hosted over a URL
— a remote Space or a local :class:`~agilerl.llm_envs.openenv.OpenEnvServer`) or a
:class:`~agilerl.llm_envs.openenv.LocalEnvClient` (the env runs in-process, no HTTP —
e.g. inside a Ray actor). :meth:`RolloutEnv.from_spec` picks the right one from a URL or
a ``module:Class`` entrypoint. ``BatchRolloutEnv`` maintains independent groups of these
rollouts over a batch, sharing a :class:`BatchPointer` dataset cursor.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

import torch

from agilerl.llm_envs.openenv import (
    LocalEnvClient,
    OpenEnvClient,
    ServedEnvClient,
    _name_from_spec,
    is_url,
    load_env,
)
from agilerl.protocols import EnvClientProtocol, TextEnvProtocol
from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len

if TYPE_CHECKING:
    from agilerl.typing import RolloutPrompts


class RolloutEnv:
    """Token-level rollout env: tokenisation + turn loop over a text env client.

    It lets a model that produces tokens interact with a text world reached through an
    env client — an :class:`~agilerl.llm_envs.openenv.OpenEnvClient` over a URL, or an
    in-process :class:`~agilerl.llm_envs.openenv.LocalEnvClient`: ``reset`` pulls the
    initial prompt and ``step`` sends the decoded action and reads back the next
    observation + reward — the env may score it, run a tool, or hit a sandbox. On top of
    that it exposes the token-level surface the trainer / rollout engine drive:

    * ``reset()`` / ``step(completion_ids, sampling_logps=None)`` return a
      tokenised **observation dict** (``input_ids`` / ``attention_mask``);
    * it assembles the multi-turn **transcript** (``full_ids`` = prompt + gen0 +
      feedback0 + ... with per-turn ``turn_boundaries``);
    * it builds the **provenance mask** and the complete per-episode row via
      :meth:`get_episode_data` -> ``(full_ids, action_mask, turn_ids,
      turn_rewards, sampling_logps)``, where only policy-generated tokens train;
    * it renders the **chat template** (optionally with ``tools=`` schemas) and
      enforces the **context budget**: when ``max_model_len`` is set and the next
      turn would overflow, the episode terminates rather than dropping turns.

    It is the in-flight, mutable builder of one episode's row: the owning
    :class:`BatchRolloutEnv` keeps a ``list`` of these and reads :attr:`done` /
    :attr:`current_prompt` to drive the lockstep loop.
    """

    def __init__(
        self,
        env_client: str | EnvClientProtocol,
        tokenizer: Any,
        max_turns: int = 1,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
        mcp_tool: str | None = None,
        pad_id: int | None = None,
        apply_chat_template: bool = True,
        max_model_len: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        """Drive a text env at the token level over ``env_client`` (a URL or a client object).

        :param env_client: A URL string (builds an HTTP ``OpenEnvClient``) or an
            :class:`~agilerl.protocols.EnvClientProtocol` implementation
            used as-is. See :meth:`local` / :meth:`from_spec` for the common cases.
        :type env_client: str | EnvClientProtocol
        :param tokenizer: Tokenizer used to encode prompts/feedback and apply the
            chat template.
        :type tokenizer: Any
        :param max_turns: Maximum number of generation turns per episode.
        :type max_turns: int
        :param headers: Optional HTTP headers (e.g. auth) sent on every request.
        :type headers: dict[str, str] | None
        :param timeout_s: Per-request timeout in seconds for the OpenEnv client.
            ``None`` (the default) leaves requests unbounded; supply the value from
            the run manifest.
        :type timeout_s: float | None
        :param mcp_tool: For an external MCP server, the tool the model's text is sent
            to as a ``call_tool``; ``None`` (default) sends plain text actions.
        :type mcp_tool: str | None
        :param pad_id: Padding token id used when assembling token tensors,
            defaults to ``None``.
        :type pad_id: int | None
        :param apply_chat_template: Render prompts through the tokenizer's chat
            template (vs. raw encoding), defaults to ``True``.
        :type apply_chat_template: bool
        :param max_model_len: Engine context length (prompt + completion). When
            set, enables the stop-on-overflow check above, defaults to ``None``.
        :type max_model_len: int | None
        :param max_output_tokens: Generation budget reserved under
            ``max_model_len`` when checking for overflow, defaults to ``None``.
        :type max_output_tokens: int | None
        :ivar full_ids: The episode's running token sequence (prompt + each
            generation + each feedback turn), or ``None`` before ``reset``.
        :vartype full_ids: torch.Tensor | None
        :ivar turn_boundaries: ``(start, end, turn_idx)`` spans of the
            policy-generated tokens within ``full_ids`` — the masking provenance.
        :vartype turn_boundaries: list[tuple[int, int, int]]
        :ivar turn_rewards: Per-turn rewards returned by the wrapped env.
        :vartype turn_rewards: list[float]
        :ivar done: Whether this episode has terminated (set by ``step``).
        :vartype done: bool
        :ivar current_prompt: The latest policy-ready observation — what the
            owning ``BatchRolloutEnv`` hands to the next ``get_action``; ``{}``
            once the episode is done.
        :vartype current_prompt: dict[str, Any]
        :ivar sampling_logps: Per-turn vLLM sampling logprobs (one 1-D tensor per
            turn), captured when the policy provides them; empty on the HF path.
        :vartype sampling_logps: list[torch.Tensor]
        """
        self.max_turns = max_turns
        # ``env_client`` is either a URL string (build an HTTP client) or an already-built
        # client object (an ``OpenEnvClient`` / ``LocalEnvClient`` / injected double) —
        # the latter is how ``.local`` / ``.from_spec`` plug in an in-process or custom
        # transport. ``headers`` / ``timeout_s`` / ``mcp_tool`` apply only to the URL case.
        if isinstance(env_client, str):
            self._env_client = OpenEnvClient(
                env_client,
                headers=headers,
                timeout_s=timeout_s,
                mcp_tool=mcp_tool,
            )
        else:
            self._env_client = env_client
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.apply_chat_template = apply_chat_template
        # Tool schemas come from whatever the env advertises through the env client,
        # fetched lazily on first use (see :attr:`tools`) so construction never
        # requires a live server.
        self._tools: list[Any] | None = None
        self._tools_known = False
        self._max_model_len = max_model_len
        self._max_output_tokens = max_output_tokens
        self.full_ids: torch.Tensor | None = None
        self.turn_boundaries: list[tuple[int, int, int]] = []
        self.turn_rewards: list[float] = []
        self._turn_idx = 0
        self._prompt_text: str = ""
        self._gen_texts: list[str] = []
        self._feedback_texts: list[str] = []
        self._last_full_prompt_token_len: int | None = None
        # Cached chat-template frame around a feedback turn (rendered lazily once;
        # the frame is constant for a given tokenizer/template).
        self._boundary_parts: tuple[str, str] | None = None
        self._boundary_parts_known = False
        # Per-episode state owned by this wrapper (the in-flight row builder):
        self.done: bool = False
        self.current_prompt: dict[str, Any] = {}
        self.sampling_logps: list[torch.Tensor] = []

    @classmethod
    def serving(
        cls,
        make_env: Callable[[], Any],
        tokenizer: Any,
        max_turns: int = 1,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
        mcp_tool: str | None = None,
        env_name: str | None = None,
        **kwargs: Any,
    ) -> RolloutEnv:
        """Host one OpenEnv server for a fresh env and drive it over HTTP.

        Calls ``make_env()`` for a fresh local env and binds the ``RolloutEnv`` to a
        :class:`~agilerl.llm_envs.openenv.ServedEnvClient` — a backend that hosts the
        env on its own :class:`~agilerl.llm_envs.openenv.OpenEnvServer` and owns the
        whole server+client pair, torn down by :meth:`close`. A served env handles
        one episode at a time, so use this as the per-rollout ``env_factory`` to give
        a :class:`BatchRolloutEnv` one isolated server per concurrent rollout::

            env_factory = lambda: RolloutEnv.serving(make_env, tokenizer, max_turns=4)

        ``BatchRolloutEnv`` calls the factory ``batch_size * group_size`` times, so the
        number of servers is determined by the batch (one OS thread + port each). For
        an already-running external server (a hosted Space), pass its ``url`` to the
        :class:`RolloutEnv` constructor directly.

        :param make_env: Zero-arg factory returning a fresh local env to host.
        :param tokenizer: Tokenizer for the token-level loop.
        :param max_turns: Generation turns per episode.
        :param headers: Optional HTTP headers sent on every client request.
        :param timeout_s: Per-request client timeout in seconds (``None`` = unbounded).
        :param mcp_tool: Optional MCP transport adapter (see
            :class:`~agilerl.llm_envs.openenv.OpenEnvClient`).
        :param env_name: Name advertised in the served env's OpenEnv metadata.
        :param kwargs: Forwarded to :class:`RolloutEnv` (e.g. ``pad_id``,
            ``apply_chat_template``, ``max_model_len``).
        :rtype: RolloutEnv
        """
        env_client = ServedEnvClient(
            make_env(),
            env_name=env_name,
            headers=headers,
            timeout_s=timeout_s,
            mcp_tool=mcp_tool,
        )
        return cls(env_client, tokenizer, max_turns=max_turns, **kwargs)

    @classmethod
    def local(
        cls,
        env: TextEnvProtocol,
        tokenizer: Any,
        max_turns: int = 1,
        *,
        instruction: str = "",
        **kwargs: Any,
    ) -> RolloutEnv:
        """Drive a local env **in-process** (no HTTP) via a :class:`LocalEnvClient`.

        The no-server counterpart to :meth:`serving`: the env runs in this process (e.g.
        inside a Ray actor), so there is no uvicorn thread, port, or loopback request.
        Reach for :meth:`serving` only when the env must actually be hosted over HTTP.

        :param env: A local env (plain-text ``reset`` / ``step``).
        :param tokenizer: Tokenizer for the token-level loop.
        :param max_turns: Generation turns per episode.
        :param instruction: Prompt returned when the env's reset obs renders empty.
        :param kwargs: Forwarded to :class:`RolloutEnv` (e.g. ``pad_id``, ``max_model_len``).
        :rtype: RolloutEnv
        """
        env_client = LocalEnvClient(env, instruction=instruction)
        return cls(env_client, tokenizer, max_turns=max_turns, **kwargs)

    @classmethod
    def from_spec(
        cls,
        spec: str,
        env_config: dict[str, Any] | None,
        tokenizer: Any,
        max_turns: int = 1,
        *,
        serve: bool = False,
        **kwargs: Any,
    ) -> RolloutEnv:
        """Build a ``RolloutEnv`` from an env ``spec`` — a URL or a ``module:Class`` entrypoint.

        * a **URL** -> the env is already hosted elsewhere; drive it over an
          :class:`OpenEnvClient` (HTTP).
        * a **``module:Class`` / ``path.py:Class`` entrypoint** -> load + build the env
          with ``env_config`` and drive it **in-process** via :class:`LocalEnvClient`
          (no HTTP). Pass ``serve=True`` to instead host it on an in-process
          :class:`OpenEnvServer` (HTTP loopback) — e.g. to also expose it as a Space.

        Owns whatever it builds (the local env client, or the served server); :meth:`close`
        releases it.

        :param spec: A URL or a ``module:Class`` / ``path.py:Class`` entrypoint.
        :param env_config: Kwargs forwarded to the entrypoint constructor (ignored for a URL).
        :param tokenizer: Tokenizer for the token-level loop.
        :param max_turns: Generation turns per episode.
        :param serve: Host an entrypoint env over HTTP instead of running it in-process.
        :param kwargs: Forwarded to :class:`RolloutEnv`.
        :rtype: RolloutEnv
        """
        if is_url(spec):
            return cls(spec, tokenizer, max_turns=max_turns, **kwargs)
        env = load_env(spec, env_config)
        if serve:
            return cls.serving(
                lambda: env,
                tokenizer,
                max_turns=max_turns,
                env_name=_name_from_spec(spec),
                **kwargs,
            )
        return cls.local(env, tokenizer, max_turns=max_turns, **kwargs)

    def _tokenize_initial_prompt(self, obs_text: str) -> dict[str, torch.Tensor]:
        """Tokenize the initial observation, optionally with chat template."""
        if self.apply_chat_template:
            chat_template_kwargs: dict[str, Any] = {}
            if self.tools is not None:
                chat_template_kwargs["tools"] = self.tools
            result = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": obs_text}],
                tokenize=True,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            # Transformers v5 apply_chat_template returns a dict
            token_ids = result["input_ids"]
            if (
                isinstance(token_ids, list)
                and token_ids
                and isinstance(token_ids[0], list)
            ):
                token_ids = token_ids[0]
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

        encoded = self.tokenizer(
            [obs_text],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def _tokenize_feedback(self, feedback_text: str) -> torch.Tensor:
        """Tokenize the assistant→user→assistant boundary plus feedback text.

        Uses the tokenizer's chat template (via the cached frame from
        :meth:`_feedback_boundary_parts`) so this works for any chat-templated
        model (Gemma, Llama, Qwen, Mistral, ...), with ChatML markers as the
        fallback for tokenizers whose templates don't render the placeholders
        verbatim (rare).
        """
        if not self.apply_chat_template:
            return torch.tensor(
                [self.tokenizer.encode(feedback_text)],
                dtype=torch.long,
            )

        boundary_ids = self._chat_template_boundary_ids(feedback_text)
        if boundary_ids is not None:
            return boundary_ids

        # Fallback: ChatML-style markers (works for Qwen and derivatives).
        turn_boundary = (
            "<|im_end|>\n<|im_start|>user\n"
            + feedback_text
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        return torch.tensor(
            [self.tokenizer.encode(turn_boundary)],
            dtype=torch.long,
        )

    def _feedback_boundary_parts(self) -> tuple[str, str] | None:
        """The ``(prefix, suffix)`` strings the chat template wraps a feedback turn in.

        Rendered once per env and cached — the frame around the feedback text is
        constant for a given tokenizer/template, so per-turn work reduces to one
        ``encode``. The render is ``[user("."), assistant(placeholder),
        user(placeholder)]`` with ``add_generation_prompt=True``, using two
        per-render unique tokens (``uuid4`` hexes) that render verbatim and cannot
        collide with template text: slicing at them yields exactly the bytes that
        close the assistant turn (``prefix`` up to the feedback slot) and close the
        user turn / open the next assistant turn (``suffix``). The dummy leading
        user message keeps strict-alternation templates (e.g. some Mistral
        variants) happy.

        Returns ``None`` when the placeholders cannot be located in the render
        (caller falls back to ChatML markers).
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
        chat_template_kwargs: dict[str, Any] = {}
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
        """Token ids for the templated turn boundary carrying ``feedback_text``.

        Wraps the feedback in the cached chat-template frame from
        :meth:`_feedback_boundary_parts` and encodes it. Returns ``None`` when the
        frame is unavailable (caller falls back to ChatML markers).
        """
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
        return {
            "input_ids": self.full_ids,
            "attention_mask": torch.ones_like(self.full_ids),
        }

    @property
    def tools(self) -> list[Any] | None:
        """Tool schemas the env advertises (``None`` when none) — passed to the
        chat template as ``tools=``.

        Fetched from the env client on first access and cached; over HTTP the
        fetch is strict (see
        :meth:`~agilerl.llm_envs.openenv.OpenEnvClient._fetch_state`), so a
        broken server fails loudly here rather than silently training without
        tool schemas.
        """
        if not self._tools_known:
            self._tools = getattr(self._env_client, "tools", None) or None
            self._tools_known = True
        return self._tools

    @tools.setter
    def tools(self, value: list[Any] | None) -> None:
        """Override the advertised tool schemas (``None`` / ``[]`` -> no ``tools=``)."""
        self._tools = value or None
        self._tools_known = True

    @property
    def dataset_size(self) -> int:
        """Number of rows in the env client (0 if not dataset-backed)."""
        return getattr(self._env_client, "dataset_size", 0)

    @contextmanager
    def eval_mode(self):
        """Serve the env client's held-out split for the duration of the block."""
        inner = getattr(self._env_client, "eval_mode", None)
        if callable(inner):
            with inner():
                yield
        else:
            with nullcontext():
                yield

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a fresh episode and return the policy-ready observation plus info.

        ``seed`` and ``row_index`` are forwarded to ``env_client.reset``: a
        dataset-backed env client uses ``row_index`` to select its row, while an env
        that picks its own task (e.g. a remote OpenEnv server) ignores it.

        :param seed: Optional reset seed forwarded to the env client.
        :type seed: int | None
        :param row_index: Dataset row chosen by the owning ``BatchRolloutEnv``,
            forwarded to ``env_client.reset``.
        :type row_index: int | None
        :raises RuntimeError: When ``max_model_len`` is set and the initial
            prompt already exceeds the context budget (the owning
            ``BatchRolloutEnv`` skips such rows with a warning) — failing here
            beats the same condition killing the run inside the generation
            engine mid-training.
        """
        obs_text, info = self._env_client.reset(seed=seed, row_index=row_index)

        encoded = self._tokenize_initial_prompt(obs_text)
        if (max_pt := self._prompt_budget()) is not None:
            prompt_len = int(encoded["input_ids"].shape[1])
            if prompt_len > max_pt:
                msg = (
                    f"initial prompt is {prompt_len} tokens but the context budget "
                    f"allows at most {max_pt} "
                    f"(max_model_len={self._max_model_len}, "
                    f"max_output_tokens={self._max_output_tokens}"
                    + (f", row_index={row_index}" if row_index is not None else "")
                    + ")"
                )
                raise RuntimeError(msg)
        self.full_ids = encoded["input_ids"]
        self.turn_boundaries = []
        self.turn_rewards = []
        self._turn_idx = 0
        self._prompt_text = obs_text
        self._gen_texts = []
        self._feedback_texts = []
        self.done = False
        self.sampling_logps = []

        self.current_prompt = self._policy_observation_from_state()
        return self.current_prompt, info

    def _step(
        self,
        full_completion_ids: torch.Tensor,
        gen_text: str,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Record a generation and step the underlying environment."""
        prompt_len = self.full_ids.shape[1]
        self.full_ids = full_completion_ids.detach().to(self.full_ids.device)
        gen_end = self.full_ids.shape[1]
        self.turn_boundaries.append((prompt_len, gen_end, self._turn_idx))
        self._gen_texts.append(gen_text)

        next_obs, reward, terminated, truncated, info = self._env_client.step(gen_text)
        self.turn_rewards.append(float(reward))
        self._turn_idx += 1

        # The env may be happy to continue, but this episode's turn budget is
        # spent: truncate here so no driver (eval loop, external collector) has
        # to impose the cap itself — and no dangling feedback turn is appended.
        if not (terminated or truncated) and self._turn_idx >= self.max_turns:
            truncated = True
            info = {**info, "agilerl_max_turns_reached": int(self.max_turns)}

        prompt_dict: dict[str, Any] = {}
        if not (terminated or truncated):
            feedback_text = next_obs
            self._feedback_texts.append(feedback_text)
            feedback_ids = self._tokenize_feedback(feedback_text).to(
                self.full_ids.device
            )
            self.full_ids = torch.cat([self.full_ids, feedback_ids], dim=1)

            # Strict overflow check: if the cumulative prompt would no
            # longer fit under ``max_model_len`` with room for at least one
            # generation, terminate the trajectory cleanly.
            if (max_pt := self._prompt_budget()) is not None:
                prompt_len = int(self.full_ids.shape[1])
                if prompt_len > max_pt:
                    truncated = True
                    info = {
                        **info,
                        "agilerl_context_overflow": {
                            "full_prompt_len": prompt_len,
                            "max_prompt_tokens": int(max_pt),
                            "max_model_len": int(self._max_model_len),
                            "max_output_tokens": (
                                int(self._max_output_tokens)
                                if self._max_output_tokens is not None
                                else None
                            ),
                        },
                    }

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
        """Decode the generation from completion IDs and step the environment.

        ``sampling_logps`` is this turn's per-token vLLM rollout logprobs (or
        ``None`` on the HF path); when provided it is appended to
        :attr:`sampling_logps` so the assembled episode owns the complete row.
        """
        if self._last_full_prompt_token_len is None:
            msg = (
                "step() requires a prior reset() or step() "
                "that built a input context observation"
            )
            raise RuntimeError(msg)
        pl = self._last_full_prompt_token_len
        gen_tokens = full_completion_ids[0, pl:]
        gen_text = self.tokenizer.decode(
            gen_tokens.tolist(),
            skip_special_tokens=True,
        )
        if sampling_logps is not None:
            self.sampling_logps.append(sampling_logps)
        return self._step(full_completion_ids, gen_text)

    def get_episode_data(
        self,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        """Build and return the complete episode row for learning.

        Returns ``(full_ids, action_mask, turn_ids, turn_rewards,
        sampling_logps)``, where ``sampling_logps`` is this episode's per-turn
        vLLM logprobs concatenated across turns, or ``None`` when none were
        captured (the HF path).
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

        if self.pad_id is not None:
            pad_positions = self.full_ids[0, 1:] == self.pad_id
            action_mask[0, pad_positions] = False
            turn_ids[0, pad_positions] = -1

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
        """Close the env client, releasing whatever backend it owns.

        Ownership lives in the client: a
        :class:`~agilerl.llm_envs.openenv.ServedEnvClient` stops its server and
        releases its connection pool, an
        :class:`~agilerl.llm_envs.openenv.OpenEnvClient` sends ``/close`` and
        releases its pool, a :class:`~agilerl.llm_envs.openenv.LocalEnvClient`
        closes the wrapped env.
        """
        if hasattr(self._env_client, "close"):
            self._env_client.close()

    def get_debug_info(self) -> dict[str, Any]:
        """Return a dict of human-readable debug information for the episode."""
        if self.full_ids is None:
            return {"error": "No episode data"}

        full_ids, action_mask, _turn_ids, turn_rewards, _logps = self.get_episode_data()
        full_text = self.tokenizer.decode(
            full_ids[0].tolist(), skip_special_tokens=False
        )

        turn_details = []
        for gen_start, gen_end, tidx in self.turn_boundaries:
            gen_token_ids = full_ids[0, gen_start:gen_end].tolist()
            gen_text_decoded = self.tokenizer.decode(
                gen_token_ids, skip_special_tokens=True
            )
            turn_details.append(
                {
                    "turn": tidx,
                    "gen_start": gen_start,
                    "gen_end": gen_end,
                    "gen_len": gen_end - gen_start,
                    "gen_text_sent_to_env": self._gen_texts[tidx]
                    if tidx < len(self._gen_texts)
                    else None,
                    "gen_text_decoded_from_ids": gen_text_decoded,
                    "gen_token_ids": gen_token_ids[:50],
                    "reward": self.turn_rewards[tidx]
                    if tidx < len(self.turn_rewards)
                    else None,
                }
            )

        n_action_tokens = action_mask.sum().item()
        n_total_tokens = full_ids.shape[1]

        return {
            "n_turns": len(self.turn_boundaries),
            "n_total_tokens": n_total_tokens,
            "n_action_tokens": n_action_tokens,
            "action_fraction": n_action_tokens / max(n_total_tokens - 1, 1),
            "turn_rewards_raw": list(self.turn_rewards),
            "turn_rewards_padded": turn_rewards.tolist(),
            "prompt_text": self._prompt_text[:200],
            "full_text_preview": full_text[:500],
            "turn_details": turn_details,
            "feedback_texts": self._feedback_texts,
        }


class BatchPointer:
    """Group-consistent dataset cursor for batched rollouts.

    Owns a per-epoch reshuffled row stream (a fresh full permutation each epoch, from one
    seeded generator) and hands out one dataset row per batch item, reused across that
    item's group so a GRPO group shares its prompt. Both the in-process
    :class:`BatchRolloutEnv` and the distributed ``RayBatchRolloutEnv`` (the Ray-actor
    collector in the integration repo) build a ``BatchPointer``, so the dataset order +
    group pinning match across the colocated and async rollout paths.

    :param dataset_size: Number of rows the env serves (``0`` -> :meth:`next_row` unused).
    :param seed: Seed for the per-epoch shuffle (``None`` -> a fixed default).
    """

    def __init__(self, dataset_size: int, *, seed: int | None = None) -> None:
        """Build a cursor over ``dataset_size`` rows with a seeded per-epoch shuffle."""
        self.dataset_size = int(dataset_size)
        #: Completed full passes over the dataset rows (drives e.g. the trainer's
        #: per-epoch reference-policy refresh).
        self.num_epochs = 0
        self._generator = torch.Generator().manual_seed(
            seed if seed is not None else 42
        )
        self._epoch_order: list[int] = []
        self._pos = 0

    def next_row(self) -> int:
        """Next dataset row in an infinite reshuffled stream (full perm per epoch)."""
        if self._pos >= len(self._epoch_order):  # epoch boundary (and first call)
            if self._epoch_order:
                self.num_epochs += 1
            self._epoch_order = torch.randperm(
                self.dataset_size, generator=self._generator
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
        """One ``(seed, row_index)`` per slot, group-contiguous; a group shares both.

        Returns ``batch_size * group_size`` pairs in slot order. ``seed`` is
        ``base_seed + seed_offset + batch_item`` (``None`` when ``base_seed`` is ``None``);
        ``row_index`` is the next reshuffled row per batch item (``None`` when
        ``dataset_size == 0``), reused across the item's group. The distributed collector
        calls this to drive its per-episode workers with matching seeds + rows.
        """
        out: list[tuple[int | None, int | None]] = []
        for item in range(batch_size):
            seed = (
                None if base_seed is None else int(base_seed) + int(seed_offset) + item
            )
            row = self.next_row() if self.dataset_size > 0 else None
            out.extend([(seed, row)] * group_size)
        return out


class BatchRolloutEnv:
    """Batched in-process collector of LLM rollout episodes — the colocated path.

    Holds ``batch_size * group_size`` independent :class:`RolloutEnv`
    instances in ``self.envs`` (laid out group-contiguous, position implicit in
    list order) and steps all active ones in lock-step using policy completions.
    Each wrapper owns its own in-flight episode state — transcript, ``done``,
    ``current_prompt`` and ``sampling_logps``.

    Envs are stepped sequentially, so per-turn latency is the sum of the envs'
    step times — negligible for in-process backends, but a real cost for
    high-latency HTTP envs. Rollout concurrency is the distributed (Ray)
    collector's job; this class is the simple colocated counterpart sharing the
    same :class:`BatchPointer` semantics.
    """

    def __init__(
        self,
        env_factory: Callable[..., RolloutEnv],
        batch_size: int,
        group_size: int,
        env_config: dict[str, Any] | None = None,
    ):
        """Create ``batch_size * group_size`` independent env wrappers.

        :param env_factory: Factory that builds one :class:`RolloutEnv`.
        :type env_factory: Callable[..., RolloutEnv]
        :param batch_size: Number of logical batch items.
        :type batch_size: int
        :param group_size: Number of grouped rollouts per batch item.
        :type group_size: int
        :param env_config: Optional kwargs passed to ``env_factory``.
        :type env_config: dict[str, Any] | None
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
        self.envs: list[RolloutEnv] = []
        # Shared dataset cursor, built on the first reset from the first env's
        # ``dataset_size`` (0 -> it only derives per-item seeds, no row pinning). The
        # same :class:`BatchPointer` drives the distributed ``RayBatchRolloutEnv``, so
        # dataset order + group pinning match across the colocated and async paths.
        self._pointer: BatchPointer | None = None

    @property
    def _is_initialized(self) -> bool:
        """``True`` once every env slot has been created."""
        return len(self.envs) == self.num_envs

    @property
    def num_epochs(self) -> int:
        """Completed passes over the dataset rows (``0`` when not dataset-backed)."""
        return self._pointer.num_epochs if self._pointer is not None else 0

    def _active_envs(self) -> list[RolloutEnv]:
        """Non-terminal envs, in their stable batch/group (list) order."""
        return [env for env in self.envs if not env.done]

    def _get_prompts(self) -> list[RolloutPrompts] | None:
        """Prompts for active envs, or ``None`` when all are terminal."""
        active = self._active_envs()
        if not active:
            return None
        return [env.current_prompt for env in active]

    def _reset_env(
        self,
        *,
        seed: int | None,
        env_idx: int,
        row_index: int | None = None,
    ) -> None:
        """Reset one existing env in place; the wrapper rebinds its own state.

        :param seed: Optional reset seed passed to the env wrapper.
        :type seed: int | None
        :param env_idx: Index into ``self.envs`` to reset.
        :type env_idx: int
        :param row_index: Dataset row chosen by this ``BatchRolloutEnv``.
        :type row_index: int | None
        """
        if env_idx < 0 or env_idx >= len(self.envs):
            msg = f"env_idx out of bounds: {env_idx} not in [0, {len(self.envs) - 1}]"
            raise IndexError(msg)
        env = self.envs[env_idx]
        if row_index is not None:
            env.reset(seed=seed, row_index=row_index)
        else:
            env.reset(seed=seed)

    def reset(
        self,
        seed: int | None = None,
    ) -> list[RolloutPrompts] | None:
        """Reset all env wrappers, building them on the first call.

        Seed/row assignment is delegated to :meth:`BatchPointer.assign` — the one
        policy shared with the distributed collector: one seed per batch row (same
        across the row's group) and one dataset row per batch row (advancing once
        per row, re-shuffling each epoch), so every group is row-consistent.
        Prompts are returned in stable list (batch/group) order. Envs that are not
        dataset-backed (``dataset_size == 0``) get seeds but no ``row_index``.

        A dataset row whose initial prompt exceeds the context budget
        is skipped with a warning and replaced by the next row in the stream.

        If building the envs fails partway (e.g. a server fails to start), the
        already-built envs are closed and the batch is left empty, so a retried
        ``reset`` starts clean.

        :param seed: Optional base seed for deterministic rollouts.
        :type seed: int | None
        :return: Active prompt dictionaries after reset.
        :rtype: list[RolloutPrompts] | None
        """
        if not self._is_initialized:
            try:
                while len(self.envs) < self.num_envs:
                    self.envs.append(self.env_factory(**self.env_config))
            except Exception:
                self.close()
                self.envs = []
                raise
            self._pointer = BatchPointer(
                int(getattr(self.envs[0], "dataset_size", 0) or 0), seed=seed
            )

        assignments = self._pointer.assign(
            self.batch_size, self.group_size, base_seed=seed
        )
        for batch_idx in range(self.batch_size):
            lead_idx = batch_idx * self.group_size
            env_seed, row_index = assignments[lead_idx]
            row_index = self._reset_lead(self.envs[lead_idx], env_seed, row_index)
            for group_idx in range(1, self.group_size):
                env = self.envs[lead_idx + group_idx]
                if row_index is not None:
                    env.reset(seed=env_seed, row_index=row_index)
                else:
                    env.reset(seed=env_seed)

        return self._get_prompts()

    def _reset_lead(
        self,
        env: RolloutEnv,
        seed: int | None,
        row_index: int | None,
    ) -> int | None:
        """Reset a group-lead env, skipping over-budget dataset rows with a warning.

        Returns the row the group settles on (the assigned row, or the next
        in-budget row from the shared cursor). Re-raised when the env is not
        dataset-backed (no cursor to advance) or no in-budget row turns up
        within 100 attempts.
        """
        assert self._pointer is not None
        attempts = min(max(self._pointer.dataset_size, 1), 100)
        for _ in range(attempts):
            error = self._reset_or_overflow(env, seed, row_index)
            if error is None:
                return row_index
            if row_index is None:
                raise error
            warnings.warn(
                f"Skipping dataset row {row_index}: {error}",
                stacklevel=2,
            )
            row_index = self._pointer.next_row()
        msg = f"no dataset row fit the context budget after {attempts} attempts"
        raise RuntimeError(msg) from error

    @staticmethod
    def _reset_or_overflow(
        env: RolloutEnv,
        seed: int | None,
        row_index: int | None,
    ) -> RuntimeError | None:
        """Reset ``env``, returning its ``RuntimeError`` (an over-budget prompt)
        instead of raising, so the caller can skip the row.
        """
        try:
            if row_index is not None:
                env.reset(seed=seed, row_index=row_index)
            else:
                env.reset(seed=seed)
        except RuntimeError as exc:
            return exc
        return None

    def step(
        self,
        completion_ids: list[torch.Tensor],
        sampling_logps: list[torch.Tensor | None] | None = None,
    ) -> list[RolloutPrompts] | None:
        """Step each active env with its corresponding completion.

        Each wrapper records the turn's ``sampling_logps`` and updates its own
        ``done`` / ``current_prompt``, so the collector only routes completions.

        :param completion_ids: One completion tensor per active env.
        :type completion_ids: list[torch.Tensor]
        :param sampling_logps: Sampling logprobs from vLLM rollout for this
            turn, parallel to ``completion_ids``; entries (or the whole list)
            may be ``None`` when nothing was captured.
        :type sampling_logps: list[torch.Tensor | None] | None
        :return: Next active prompt dictionaries after stepping.
        :rtype: list[RolloutPrompts] | None
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
        for idx, (env, completion) in enumerate(
            zip(active, completion_ids, strict=False)
        ):
            full_completion = completion
            if full_completion.dim() == 1:
                full_completion = full_completion.unsqueeze(0)
            slp = sampling_logps[idx] if sampling_logps is not None else None
            env.step(full_completion, sampling_logps=slp)
        return self._get_prompts()

    def close(self) -> None:
        """Close all env wrappers."""
        seen: set[int] = set()
        for env in self.envs:
            env_id = id(env)
            if env_id in seen:
                continue
            seen.add(env_id)
            if hasattr(env, "close"):
                env.close()

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

        :return: ``(completion_ids_list, action_masks_list, all_turn_ids,
            all_rewards, batch_steps, all_sampling_logps)`` where ``batch_steps``
            is the summed number of recorded turn boundaries across envs.
            ``all_sampling_logps`` is ``None`` when no vLLM logprobs were captured
            this rollout; otherwise it holds one 1-D tensor of generated-token
            logprobs per env (concatenated across turns), with ``None`` for any
            env that captured none.
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int, list[torch.Tensor | None] | None]
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
            batch_steps += len(getattr(env, "turn_boundaries", []))
            all_sampling_logps.append(sampling_logps)

        return (
            completion_ids_list,
            action_masks_list,
            all_turn_ids,
            all_rewards,
            batch_steps,
            # Collapse to a single ``None`` when nothing was captured, so the
            # caller needs only an ``is not None`` check (no per-row re-scan).
            (
                all_sampling_logps
                if any(logps is not None for logps in all_sampling_logps)
                else None
            ),
        )
