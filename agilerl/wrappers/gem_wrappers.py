"""Token-level observation wrapper for multi-turn GEM environments."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import requests
import torch
from gem.tools.base_tool import BaseTool

from agilerl.utils.llm_utils import max_prompt_tokens_for_sliding_window

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

    :param env: GEM environment.
    :type env: GemEnv
    :param tokenizer: Tokenizer for encoding/decoding text.
    :type tokenizer: Any
    :param max_turns: Maximum number of interaction turns per episode.
    :type max_turns: int
    :param pad_id: Pad token ID used to mask padding positions, or ``None``.
    :type pad_id: int | None
    :param apply_chat_template: Whether to format observations using the
        tokenizer's chat template. Defaults to ``True``.
    :type apply_chat_template: bool
    :param max_model_len: Context length for sliding-window prompts; if ``None``,
        observations skip merging :meth:`build_model_prompt_fields`.
    :type max_model_len: int | None
    :param max_output_tokens: Max new tokens cap (same meaning as the policy);
        only used when ``max_model_len`` is set.
    :type max_output_tokens: int | None
    """

    def __init__(
        self,
        env: GemEnv,
        tokenizer: Any,
        max_turns: int,
        pad_id: int | None = None,
        apply_chat_template: bool = True,
        max_model_len: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self._env = env
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.pad_id = pad_id
        self.apply_chat_template = apply_chat_template
        self._sw_max_model_len = max_model_len
        self._sw_max_output_tokens = max_output_tokens
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

    def _policy_observation_from_state(self) -> dict[str, Any]:
        """Build observation dict for ``get_action`` from current ``full_ids``."""
        if self.full_ids is None:
            msg = "No prompt: reset() was never called"
            raise RuntimeError(
                msg,
            )
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
        if self._sw_max_model_len is not None:
            max_pt = max_prompt_tokens_for_sliding_window(
                self._sw_max_model_len,
                self._sw_max_output_tokens,
            )
            obs.update(self.build_model_prompt_fields(max_pt))
        return obs

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a fresh episode and return the policy-ready observation plus info.

        The observation includes ``input_ids``, ``attention_mask``, ``text``, and
        when ``max_model_len`` was set at construction, sliding-window fields from
        :meth:`build_model_prompt_fields`. Sets ``_initial_prompt_len`` and
        ``_last_full_prompt_token_len`` for :meth:`step`.

        :param seed: Optional RNG seed forwarded to the underlying env ``reset`` so
            parallel GRPO rollouts can share the same stochastic initial state.
        :type seed: int | None
        :return: ``(observation, info)`` with ``info`` from the underlying GEM env.
        :rtype: tuple[dict[str, Any], dict[str, Any]]
        """
        if seed is not None:
            try:
                obs_text, info = self._env.reset(seed=seed)
            except TypeError:
                obs_text, info = self._env.reset()
        else:
            obs_text, info = self._env.reset()
        obs_text = self._format_obs(obs_text, info)

        encoded = self._tokenize_initial_prompt(obs_text)
        self.full_ids = encoded["input_ids"]
        self._initial_prompt_len = int(encoded["input_ids"].shape[1])
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
        """Record a generation and step the underlying environment.

        :param full_completion_ids: Full token IDs (prompt + gen) of shape
            ``[1, seq_len]`` as returned by ``get_action``.
        :type full_completion_ids: torch.Tensor
        :param gen_text: Decoded generation text sent to the environment.
        :type gen_text: str
        :return: ``(next_observation, reward, terminated, truncated, info)``.
            ``next_observation`` is empty when the episode ended; otherwise it is
            the policy-ready dict from :meth:`_policy_observation_from_state`.
        :rtype: tuple[dict[str, Any], float, bool, bool, dict[str, Any]]
        """
        prompt_len = self.full_ids.shape[1]
        self.full_ids = full_completion_ids.to(self.full_ids.device)
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
            prompt_dict = self._policy_observation_from_state()

        return prompt_dict, reward, terminated, truncated, info

    def step(
        self,
        full_completion_ids: torch.Tensor,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Decode the generation from ``full_completion_ids`` and call :meth:`step`.

        Uses ``_last_full_prompt_token_len`` from the latest observation built by
        :meth:`reset` or :meth:`step`.

        :param full_completion_ids: Prompt + generated tokens, shape ``[1, seq]``.
        :type full_completion_ids: torch.Tensor
        :return: Same tuple as :meth:`_step`.
        :rtype: tuple[dict[str, Any], float, bool, bool, dict[str, Any]]
        """
        if self._last_full_prompt_token_len is None:
            msg = (
                "step() requires a prior reset() or step() "
                "that built a policy observation"
            )
            raise RuntimeError(
                msg,
            )
        pl = self._last_full_prompt_token_len
        gen_tokens = full_completion_ids[0, pl:]
        gen_text = self.tokenizer.decode(
            gen_tokens.tolist(),
            skip_special_tokens=True,
        )
        return self._step(full_completion_ids, gen_text)

    def build_model_prompt_fields(
        self,
        max_prompt_tokens: int,
    ) -> dict[str, Any]:
        """Build truncated prompt tensors for the LLM and a stitch prefix for full trajectories.

        Drops the oldest completed assistant+feedback turns after the initial
        user prompt until ``max_prompt_tokens`` is respected (or only the
        initial segment remains).

        :param max_prompt_tokens: Maximum number of tokens in the prompt passed
            to the model (after truncation).
        :type max_prompt_tokens: int
        :return: Dict with ``trajectory_input_ids``, ``trajectory_attention_mask``,
            ``trajectory_text``, ``stitch_prefix_ids`` (tokens removed between the
            initial segment and the kept tail), and ``model_window_initial_len``.
            Chronological reconstruction is
            ``cat(trajectory_input_ids[:, :I], stitch_prefix_ids, trajectory_input_ids[:, I:], dim=1)``
            where ``I`` is ``model_window_initial_len``.
        :rtype: dict[str, Any]
        """
        if self.full_ids is None:
            msg = "No prompt: reset() was never called"
            raise RuntimeError(
                msg,
            )

        full = self.full_ids
        initial_len = self._initial_prompt_len
        seq_len = full.shape[1]
        boundaries = self.turn_boundaries
        n = len(boundaries)

        if initial_len > max_prompt_tokens:
            msg = (
                f"Initial prompt ({initial_len} tokens) exceeds "
                f"max_prompt_tokens ({max_prompt_tokens})."
            )
            raise RuntimeError(
                msg,
            )

        k = 0
        while True:
            if k < n:
                drop_from = boundaries[k][0]
            elif n == 0:
                drop_from = initial_len
            else:
                drop_from = seq_len
            if drop_from >= seq_len:
                trunc = full[:, :initial_len].clone()
            else:
                trunc = torch.cat(
                    [full[:, :initial_len], full[:, drop_from:]],
                    dim=1,
                )
            if trunc.shape[1] <= max_prompt_tokens or k >= n:
                break
            k += 1

        if trunc.shape[1] > max_prompt_tokens:
            msg = (
                "Could not fit prompt even after dropping all post-initial turns; "
                f"trunc_len={trunc.shape[1]}, max_prompt_tokens={max_prompt_tokens}."
            )
            raise RuntimeError(
                msg,
            )

        if k < n:
            drop_from_final = boundaries[k][0]
        elif n == 0:
            drop_from_final = initial_len
        else:
            drop_from_final = seq_len
        stitch = full[:, initial_len:drop_from_final]

        prompt_ids_1d = trunc[0]
        trajectory_text = self.tokenizer.decode(
            prompt_ids_1d.tolist(),
            skip_special_tokens=True,
        )
        return {
            "trajectory_input_ids": trunc,
            "trajectory_attention_mask": torch.ones_like(trunc),
            "trajectory_text": trajectory_text,
            "stitch_prefix_ids": stitch,
            "initial_prompt_len": initial_len,
        }

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


# Adapted from https://github.com/PeterGriffinJin/Search-R1

# Timeout for search request in seconds
TIMEOUT = 5


class SearchTool(BaseTool):
    tool_type = "search"

    def __init__(self, num_workers=1, search_url=None, topk=3, timeout=TIMEOUT):
        super().__init__(num_workers)
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self._search_url_resolved = self.search_url is not None

    def _parse_action(self, action: str) -> tuple[str, str, bool]:
        """Parse the action string to extract the <search> content and the full matched tag.
        Returns (content, parsed_action, is_valid)
        """
        # only take the first match
        pattern = r"<search>(.*?)</search>"
        match = re.search(pattern, action, re.DOTALL)
        if match:
            parsed_query = match.group(1).strip()
            parsed_action = action[: match.end()]  # including thinking process
            return parsed_query, parsed_action, True
        return "", "", False

    def _search(self, query: str):
        """Perform a search using the configured search_url.
        Returns a formatted string of search results.
        """
        if not self._search_url_resolved:
            self.search_url = self.search_url or os.environ.get("SEARCH_URL")
            self._search_url_resolved = True

        if not self.search_url:
            msg = "search_url must be provided for SearchTool."
            raise ValueError(msg)

        payload = {"q": query, "format": "json"}

        try:
            response = requests.get(
                self.search_url,
                params=payload,
                timeout=self.timeout,
            ).json()
            result = response["results"][: self.topk]
            response_string = ""
            for r in result:
                response_string += f"  {r.get('content', '')}\n"
            return response_string
        except Exception as e:
            return f"[SearchTool Error: {e}]"

    def _passages2string(self, result):
        format_reference = ""
        for idx, doc_item in enumerate(result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return format_reference

    def instruction_string(self) -> str:
        return (
            "You have access to a search engine to help answer questions.\n\n"
            "Additional instructions:\n"
            "- If your initial reasoning in <think> shows you lack some knowledge, explain what you need to find next inside a new <think> block.\n"
            "- Then issue a search query using:\n"
            "  <search> your query here </search>\n"
            "- The search engine will provide results inside:\n"
            "  <information> ... </information>\n"
            "- You may repeat the <think> and <search> steps as many times as needed.\n"
            "- When you are ready, give your final answer in:\n"
            "  <answer> your answer here </answer>"
        )

    def execute_action(self, action: str):
        """Execute the parsed action for the SearchTool.

        Args:
            action: The raw action string, typically containing a search query
                within <search>...</search> tags.

        Returns:
            observation: The formatted search result, or an empty string if invalid.
            done: Always False for search tool (search does not terminate the episode).
            valid: True if a valid search query was found and executed, False otherwise.
        """
        parsed_query, parsed_action, is_valid = self._parse_action(action)
        if not is_valid:
            # observation = "No valid search query found. Please provide your query within <search>...</search> tags."
            observation = ""
            valid = False
            has_error = True
        else:
            search_result = self._search(parsed_query)
            observation = f"\n\n<information>{search_result}</information>\n\n"
            valid = True
            has_error = "[SearchTool Error:" in search_result
        return valid, has_error, observation, parsed_action


class FormatRewardWrapper:
    """Wraps a GEM environment to give a small bonus for producing <answer> tags.

    Without this, the model gets zero reward when it answers without using
    the correct format, making it impossible to learn the format through RL
    alone (classic sparse-reward problem).
    """

    def __init__(self, env, format_bonus: float = 0.1):
        self._env = env
        self._format_bonus = format_bonus

    @property
    def format_bonus(self):
        return self._format_bonus

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: str, **kwargs):
        obs, reward, terminated, truncated, info = self._env.step(action, **kwargs)
        if terminated and "<answer>" in action and not info.get("correct", False):
            reward += self._format_bonus
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)
