"""Search and formatting wrappers for multi-turn LLM environments."""

from __future__ import annotations

import os
import re
from typing import Any

import requests

# Timeout for search request in seconds
TIMEOUT = 5


class SearchTool:
    tool_type = "search"

    def __init__(self, num_workers=1, search_url=None, topk=3, timeout=TIMEOUT):
        self.num_workers = num_workers
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self._search_url_resolved = self.search_url is not None

    def _parse_action(self, action: str) -> tuple[str, str, bool]:
        """Parse action string and extract <search> payload."""
        pattern = r"<search>(.*?)</search>"
        match = re.search(pattern, action, re.DOTALL)
        if match:
            parsed_query = match.group(1).strip()
            parsed_action = action[: match.end()]
            return parsed_query, parsed_action, True
        return "", "", False

    def _search(self, query: str):
        """Perform a search using the configured search_url."""
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
        """Execute parsed action for SearchTool."""
        parsed_query, parsed_action, is_valid = self._parse_action(action)
        if not is_valid:
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
    """Wraps a multi-turn environment to give a small bonus for <answer> tags."""

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
