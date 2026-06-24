"""Token-level rollout harness for multi-turn LLM environments."""

from __future__ import annotations

import uuid
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from agilerl.llm_envs.rollout_env import RolloutEnv
from agilerl.utils.llm_utils import max_prompt_tokens_for_model_len


class RolloutHarness:
    """Token-level rollout harness around a text :class:`RolloutEnv`.

    The *harness* is the adapter that lets a model which produces tokens
    interact with a text-based environment. The wrapped :class:`RolloutEnv` speaks
    strings (``reset()`` -> prompt text, ``step(action_text)`` -> reward); this
    harness exposes the token contract the trainer/rollout engine and the env:

    * ``reset()`` / ``step(completion_ids)`` return a tokenised **observation
      dict** (``input_ids`` / ``attention_mask`` / ``text``);
    * it assembles the multi-turn **transcript** (``full_ids`` = prompt + gen0 +
      feedback0 + ... with per-turn ``turn_boundaries``);
    * it builds the **provenance mask** via :meth:`get_episode_data` ->
      ``(full_ids, action_mask, turn_ids, turn_rewards)``, where only
      policy-generated tokens train;
    * it renders the **chat template** (optionally with ``tools=`` schemas) and
      enforces the **context budget**: when ``max_model_len`` is set and the next
      turn would overflow, the episode terminates rather than dropping turns.
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
        """
        self.max_turns = max_turns
        self._env = env
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.apply_chat_template = apply_chat_template
        # Preserve the None-vs-list distinction the chat-template path keys on
        # (``None`` => omit the ``tools=`` kwarg entirely).
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

        return self._policy_observation_from_state(), info

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

        return prompt_dict, reward, terminated, truncated, info

    def step(
        self,
        full_completion_ids: torch.Tensor,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Decode the generation from completion IDs and step the environment."""
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
        return self._step(full_completion_ids, gen_text)

    def get_episode_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build and return the full episode data for learning."""
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
        )

    def close(self) -> None:
        """Close the wrapped environment when supported."""
        if hasattr(self._env, "close"):
            self._env.close()

    def get_debug_info(self) -> dict[str, Any]:
        """Return a dict of human-readable debug information for the episode."""
        if self.full_ids is None:
            return {"error": "No episode data"}

        full_ids, action_mask, _turn_ids, turn_rewards = self.get_episode_data()
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
