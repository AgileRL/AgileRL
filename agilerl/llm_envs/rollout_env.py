"""Environment for generative LLM tasks, e.g. multi-turn agentic tasks or single-turn reasoning.

A :class:`RolloutEnv`, produces prompts (usually backed by a dataset) that are sent to the model, which generates a completion, that is scored by the env's ``reward_fn``. Reasoning is the ``max_turns=1`` configuration.
``BatchRolloutEnv`` is a wrapper to maintain independent groups of rollouts on batches of prompts.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

import torch

from agilerl.llm_envs.base import LLMEnv
from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len

if TYPE_CHECKING:
    from agilerl.typing import RolloutPrompts


class RolloutEnv(LLMEnv):
    """Generation rollout env: dataset-seeded prompt in, scored text out.

    Text observation / text action. With ``max_turns=1`` (the default) this is the
    reasoning env: the model produces one completion to a dataset-seeded prompt and
    the env scores it via ``reward_fn(completion, answer, question)`` on the decoded
    generation.

    :param max_turns: Number of generation turns before the episode terminates.
    :type max_turns: int
    :param tools: Optional tool schemas available to the policy.
    :type tools: list | None
    :param questions: Per-row question strings.
    :type questions: list[str] | None
    :param answers: Per-row training answer strings.
    :type answers: list[str] | None
    :param reward_fn: ``(completion, answer, question) -> float`` scorer.
    :type reward_fn: Callable[[str, str, str], float] | None
    :param prompt_builder: Maps a question to the prompt text shown to the model.
    :type prompt_builder: Callable[[str], str] | None
    :param test_questions: Held-out question strings used under ``eval_mode``.
    :type test_questions: list[str] | None
    :param test_answers: Held-out answer strings used under ``eval_mode``.
    :type test_answers: list[str] | None
    """

    def __init__(
        self,
        *,
        max_turns: int = 1,
        tools: list | None = None,
        questions: list[str] | None = None,
        answers: list[str] | None = None,
        reward_fn: Callable[[str, str, str], float] | None = None,
        prompt_builder: Callable[[str], str] | None = None,
        test_questions: list[str] | None = None,
        test_answers: list[str] | None = None,
    ) -> None:
        self.max_turns = max_turns
        self.tools = list(tools) if tools else []
        self.questions = questions
        self.answers = answers
        self.reward_fn = reward_fn
        self.prompt_builder = prompt_builder
        self.test_questions = test_questions
        self.test_answers = test_answers
        self.evaluation_mode = False
        self._turn = 0
        self._question: str = ""
        self._answer: str = ""
        # Dataset cursor owned by ``BatchRolloutEnv`` when batched; on the
        # standalone / eval path the env walks its active split sequentially.
        self._cursor = 0
        self._cursor_split = ""

    @property
    def dataset_size(self) -> int:
        """Number of training rows backing this env (0 when not dataset-backed)."""
        return len(self.questions) if self.questions else 0

    def _active_rows(self) -> tuple[list[str], list[str]]:
        """Return the (questions, answers) for the current train/eval split."""
        if self.evaluation_mode and self.test_questions is not None:
            return self.test_questions, self.test_answers
        return self.questions, self.answers

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Select a dataset row and return its prompt text plus info.

        :param seed: Optional reset seed (unused on the standalone path, which
            walks its split sequentially).
        :type seed: int | None
        :param row_index: Dataset row chosen by the owning ``BatchRolloutEnv``;
            when ``None`` the env resolves a row from its own per-split cursor.
        :type row_index: int | None
        """
        self._turn = 0
        questions, answers = self._active_rows()
        if row_index is None:
            split = (
                "eval"
                if self.evaluation_mode and self.test_questions is not None
                else "train"
            )
            if split != self._cursor_split:
                self._cursor = 0
                self._cursor_split = split
            row_index = self._cursor
            self._cursor += 1
        row = row_index % len(questions)
        self._question = questions[row]
        self._answer = answers[row]
        return self.prompt_builder(self._question), {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Score the completion against the current row; terminate at ``max_turns``."""
        self._turn += 1
        reward = float(self.reward_fn(action, self._answer, self._question))
        terminated = self._turn >= self.max_turns
        return "", reward, terminated, False, {}


class RolloutEnvWrapper:
    """Token-level wrapper around a text :class:`RolloutEnv`.

    The *wrapper* is the adapter that lets a model which produces tokens interact
    with a text-based environment. The wrapped :class:`RolloutEnv` speaks strings
    (``reset()`` -> prompt text, ``step(action_text)`` -> reward); this wrapper
    exposes the token contract the trainer / rollout engine drive:

    * ``reset()`` / ``step(completion_ids, sampling_logps=None)`` return a
      tokenised **observation dict** (``input_ids`` / ``attention_mask`` /
      ``text``);
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
        env: RolloutEnv,
        tokenizer: Any,
        max_turns: int,
        pad_id: int | None = None,
        apply_chat_template: bool = True,
        max_model_len: int | None = None,
        max_output_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Wrap ``env`` so it can be driven at the token level.

        When ``max_model_len`` is set, the cumulative prompt is never
        truncated: if appending the next turn's feedback would push the prompt
        past ``max_model_len - max_output_tokens``, the trajectory terminates
        with ``truncated=True`` and an ``agilerl_context_overflow`` breadcrumb
        in ``info`` (generation stops rather than dropping older turns).

        :param env: The text-level rollout environment to wrap.
        :type env: RolloutEnv
        :param tokenizer: Tokenizer used to encode prompts/feedback and apply the
            chat template.
        :type tokenizer: Any
        :param max_turns: Maximum number of generation turns per episode.
        :type max_turns: int
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
        :param tools: Optional OpenAI/JSON tool schemas forwarded to the chat
            template (``tools=``) so the policy can emit tool calls. ``None``
            (default) preserves the exact pre-tool behaviour (no ``tools=`` kwarg
            is passed).
        :type tools: list[dict] | None
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
        self._env = env
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.apply_chat_template = apply_chat_template
        self.tools = tools
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
        # Per-episode state owned by this wrapper (the in-flight row builder):
        self.done: bool = False
        self.current_prompt: dict[str, Any] = {}
        self.sampling_logps: list[torch.Tensor] = []

    @staticmethod
    def _format_obs(obs: str, info: dict[str, Any] | None) -> str:
        """Apply prefix/suffix from info dict to an observation string."""
        text = str(obs)
        if not info:
            return text
        prefix = info.get("prefix", "")
        suffix = info.get("suffix", "")
        if prefix:
            text = f"{prefix}{text}"
        if suffix:
            text = f"{text}\n{suffix}"
        return text

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

        Uses ``tokenizer.apply_chat_template`` to compute the boundary so this
        works for any chat-templated model (Gemma, Llama, Qwen, Mistral, ...).
        We render two short conversations — one ending after a placeholder
        assistant turn, one continuing with the new user feedback turn and a
        fresh generation prompt — then take the suffix difference. Falling
        back to ChatML markers preserves prior behaviour for tokenizers whose
        templates don't preserve the prefix string (rare).
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

    def _chat_template_boundary_ids(
        self,
        feedback_text: str,
    ) -> torch.Tensor | None:
        """Compute boundary token ids via the tokenizer's chat template.

        Renders ``[user("."), assistant(placeholder), user(feedback)]`` with
        ``add_generation_prompt=True`` and slices the rendered string from
        the end of ``placeholder`` to the end. This yields exactly the bytes
        that close the assistant turn, write a user turn containing
        ``feedback_text``, and open the next assistant turn — for whatever
        chat template the tokenizer carries. The dummy leading user message
        keeps strict-alternation templates (e.g. some Mistral variants)
        happy. The placeholder is a per-render unique token (a ``uuid4`` hex)
        chosen so it renders verbatim and cannot collide with anything the
        template or ``feedback_text`` might already contain.

        Returns ``None`` if the placeholder cannot be located in the render
        (caller should fall back to ChatML markers).
        """
        placeholder = uuid.uuid4().hex
        messages = [
            {"role": "user", "content": "."},
            {"role": "assistant", "content": placeholder},
            {"role": "user", "content": feedback_text},
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

        idx = rendered.rfind(placeholder)
        if idx < 0:
            return None

        boundary_text = rendered[idx + len(placeholder) :]
        if not boundary_text:
            return None

        encoded = self.tokenizer.encode(boundary_text, add_special_tokens=False)
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
        prompt_ids_1d = self.full_ids[0]
        obs: dict[str, Any] = {
            "input_ids": self.full_ids,
            "attention_mask": torch.ones_like(self.full_ids),
            "text": self.tokenizer.decode(
                prompt_ids_1d.tolist(),
                skip_special_tokens=True,
            ),
        }
        return obs

    @property
    def dataset_size(self) -> int:
        """Number of training rows backing the wrapped env (0 if not dataset-backed)."""
        return getattr(self._env, "dataset_size", 0)

    @property
    def evaluation_mode(self) -> bool:
        """Whether the wrapped env is currently serving its held-out split."""
        return bool(getattr(self._env, "evaluation_mode", False))

    @evaluation_mode.setter
    def evaluation_mode(self, value: bool) -> None:
        if hasattr(self._env, "evaluation_mode"):
            self._env.evaluation_mode = value

    @contextmanager
    def eval_mode(self):
        """Serve the wrapped env's held-out split for the duration of the block."""
        inner = getattr(self._env, "eval_mode", None)
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

        ``seed`` is forwarded to the wrapped env; ``row_index`` is forwarded only
        when set, so non-dataset envs (e.g. constant/probe envs whose ``reset``
        takes no ``row_index``) still work — they are never assigned a row.

        :param seed: Optional reset seed forwarded to the wrapped env.
        :type seed: int | None
        :param row_index: Dataset row chosen by the owning ``BatchRolloutEnv``,
            forwarded to the wrapped env's ``reset`` only when not ``None``.
        :type row_index: int | None
        """
        reset_kwargs: dict[str, Any] = {"seed": seed}
        if row_index is not None:
            reset_kwargs["row_index"] = row_index
        obs_text, info = self._env.reset(**reset_kwargs)
        obs_text = self._format_obs(obs_text, info)

        encoded = self._tokenize_initial_prompt(obs_text)
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

        next_obs, reward, terminated, truncated, info = self._env.step(gen_text)
        self.turn_rewards.append(float(reward))
        self._turn_idx += 1

        prompt_dict: dict[str, Any] = {}
        if not (terminated or truncated):
            feedback_text = self._format_obs(next_obs, info)
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
                "that built a policy observation"
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
        """Close the wrapped environment when supported."""
        if hasattr(self._env, "close"):
            self._env.close()

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


class BatchRolloutEnv:
    """Batched in-process collector of LLM rollout episodes.

    Holds ``batch_size * group_size`` independent :class:`RolloutEnvWrapper`
    instances in ``self.envs`` (laid out group-contiguous, position implicit in
    list order) and steps all active ones in lock-step using policy completions.
    Each wrapper owns its own in-flight episode state — transcript, ``done``,
    ``current_prompt`` and ``sampling_logps``.
    """

    def __init__(
        self,
        env_factory: Callable[..., RolloutEnvWrapper],
        batch_size: int,
        group_size: int,
        env_config: dict[str, Any] | None = None,
    ):
        """Create ``batch_size * group_size`` independent env wrappers.

        :param env_factory: Factory that builds one :class:`RolloutEnvWrapper`.
        :type env_factory: Callable[..., RolloutEnvWrapper]
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
        self.envs: list[RolloutEnvWrapper] = []
        self._dataset_size = 0
        self._epoch_order: list[int] = []
        self._pos = 0
        self._generator: torch.Generator | None = None

    def _next_row(self) -> int:
        """Next dataset row in an infinite reshuffled stream (full perm per epoch)."""
        if self._pos >= len(self._epoch_order):  # epoch boundary (and first call)
            self._epoch_order = torch.randperm(
                self._dataset_size, generator=self._generator
            ).tolist()
            self._pos = 0
        row = self._epoch_order[self._pos]
        self._pos += 1
        return row

    @property
    def _is_initialized(self) -> bool:
        """``True`` once every env slot has been created."""
        return len(self.envs) == self.num_envs

    def _active_envs(self) -> list[RolloutEnvWrapper]:
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

        Seeds are assigned per batch row (same seed across groups). The shared
        dataset cursor resolves a single ``row_index`` per batch row (advancing
        once per row, re-shuffling each epoch), and every group env of that row is
        reset with both the row seed and that one row, so the group is
        row-consistent. Prompts are returned in stable list (batch/group) order.
        Envs that are not dataset-backed (``dataset_size == 0``) skip the cursor
        entirely.

        :param seed: Optional base seed for deterministic rollouts.
        :type seed: int | None
        :return: Active prompt dictionaries after reset.
        :rtype: list[RolloutPrompts] | None
        """
        ds_size = 0
        seed_base = seed
        for batch_idx in range(self.batch_size):
            batch_seed = None if seed_base is None else seed_base + batch_idx
            row_index: int | None = None
            for group_idx in range(self.group_size):
                env_idx = batch_idx * self.group_size + group_idx
                if not self._is_initialized:
                    env_i = self.env_factory(**self.env_config)
                    # Size the shared shuffle from the first env's dataset, before
                    # resolving any row, so all rows draw from one permutation
                    # stream. Envs that aren't dataset-backed (no ``dataset_size``,
                    # or 0) get no shuffle and a reset without a ``row_index``.
                    ds_size = getattr(env_i, "dataset_size", 0)
                    if self._dataset_size == 0 and ds_size > 0:
                        self._dataset_size = ds_size
                        self._generator = torch.Generator().manual_seed(
                            seed if seed is not None else 42
                        )
                    if group_idx == 0 and self._dataset_size > 0:
                        row_index = self._next_row()
                    if row_index is not None:
                        env_i.reset(seed=batch_seed, row_index=row_index)
                    else:
                        env_i.reset(seed=batch_seed)
                    self.envs.append(env_i)
                else:
                    if group_idx == 0 and self._dataset_size > 0:
                        row_index = self._next_row()
                    self._reset_env(
                        env_idx=env_idx,
                        seed=batch_seed,
                        row_index=row_index,
                    )

        return self._get_prompts()

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
