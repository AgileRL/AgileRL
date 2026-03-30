"""Token-level observation wrapper for multi-turn GEM environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from gem.core import Env as GemEnv


class TokenObservationWrapper:
    """Token-level observation wrapper for multi-turn GEM environments.

    Wraps a GEM environment factory and provides a Gymnasium-like
    ``reset`` / ``step`` interface that operates on token IDs.  Maintains the
    growing token sequence, per-token action mask, and turn boundary tracking
    needed for LLMPPO's turn-level GAE.

    Each call to :meth:`reset` creates a fresh underlying environment via
    ``env_fn``, so the wrapper is reused across episodes just like a normal
    Gymnasium env.

    When ``apply_chat_template=True`` (the default for instruct models),
    observations are formatted as proper chat messages so the model receives
    input in the ``<|im_start|>user / assistant`` format it was trained on.

    :param env_fn: Factory that returns a fresh GEM environment.
    :type env_fn: Callable[[], GemEnv]
    :param tokenizer: Tokenizer for encoding/decoding text.
    :type tokenizer: Any
    :param max_turns: Maximum number of interaction turns per episode.
    :type max_turns: int
    :param pad_id: Pad token ID used to mask padding positions, or ``None``.
    :type pad_id: int | None
    :param apply_chat_template: Whether to format observations using the
        tokenizer's chat template. Defaults to ``True``.
    :type apply_chat_template: bool
    """

    def __init__(
        self,
        env_fn: Callable[[], GemEnv],
        tokenizer: Any,
        max_turns: int,
        pad_id: int | None = None,
        apply_chat_template: bool = True,
    ) -> None:
        self.env_fn = env_fn
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.pad_id = pad_id
        self.apply_chat_template = apply_chat_template

        self._env: GemEnv | None = None
        self.full_ids: torch.Tensor | None = None
        self.turn_boundaries: list[tuple[int, int, int]] = []
        self.turn_rewards: list[float] = []
        self._turn_idx = 0

        self._prompt_text: str = ""
        self._gen_texts: list[str] = []
        self._feedback_texts: list[str] = []

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
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": obs_text}],
                tokenize=True,
                add_generation_prompt=True,
            )
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
        """Tokenize feedback for the next turn, with chat turn boundaries."""
        if self.apply_chat_template:
            turn_boundary = (
                "<|im_end|>\n<|im_start|>user\n"
                + feedback_text
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            return torch.tensor(
                [self.tokenizer.encode(turn_boundary)],
                dtype=torch.long,
            )

        return torch.tensor(
            [self.tokenizer.encode(feedback_text)],
            dtype=torch.long,
        )

    def reset(self) -> dict[str, torch.Tensor]:
        """Create a fresh environment and return the tokenized initial prompt.

        :return: Prompt dict with ``input_ids`` and ``attention_mask``.
        :rtype: dict[str, torch.Tensor]
        """
        self._env = self.env_fn()
        obs_text, info = self._env.reset()
        obs_text = self._format_obs(obs_text, info)

        encoded = self._tokenize_initial_prompt(obs_text)
        self.full_ids = encoded["input_ids"]
        self.turn_boundaries = []
        self.turn_rewards = []
        self._turn_idx = 0
        self._prompt_text = obs_text
        self._gen_texts = []
        self._feedback_texts = []

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def step(
        self,
        full_completion_ids: torch.Tensor,
        gen_text: str,
    ) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, Any]]:
        """Record a generation and step the underlying environment.

        :param full_completion_ids: Full token IDs (prompt + gen) of shape
            ``[1, seq_len]`` as returned by ``get_action``.
        :type full_completion_ids: torch.Tensor
        :param gen_text: Decoded generation text sent to the environment.
        :type gen_text: str
        :return: ``(prompt_dict, reward, terminated, truncated, info)``
        :rtype: tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, Any]]
        """
        prompt_len = self.full_ids.shape[1]
        self.full_ids = full_completion_ids.to(self.full_ids.device)
        gen_end = self.full_ids.shape[1]
        self.turn_boundaries.append((prompt_len, gen_end, self._turn_idx))
        self._gen_texts.append(gen_text)

        next_obs, reward, terminated, truncated, info = self._env.step(gen_text)
        self.turn_rewards.append(float(reward))
        self._turn_idx += 1

        prompt_dict: dict[str, torch.Tensor] = {}
        if not (terminated or truncated):
            feedback_text = self._format_obs(next_obs, info)
            self._feedback_texts.append(feedback_text)
            feedback_ids = self._tokenize_feedback(feedback_text).to(
                self.full_ids.device
            )
            self.full_ids = torch.cat([self.full_ids, feedback_ids], dim=1)

            prompt_ids_1d = self.full_ids[0]
            prompt_dict = {
                "input_ids": self.full_ids,
                "attention_mask": torch.ones_like(self.full_ids),
                "text": self.tokenizer.decode(
                    prompt_ids_1d.tolist(), skip_special_tokens=True
                ),
            }

        return prompt_dict, reward, terminated, truncated, info

    def get_episode_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build and return the full episode data for learning.

        :return: ``(full_ids, action_mask, turn_ids, turn_rewards)``
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
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
        )

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
