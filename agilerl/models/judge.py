# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The LLM a rubric's judge calls."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

ENV_VAR_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


class JudgeSpec(BaseModel):
    """The LLM a rubric's judge calls: provider, model, endpoint and limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai", "anthropic", "bedrock"] = Field(
        default="openai",
        description=(
            "API the judge speaks. openai covers any OpenAI-compatible endpoint: "
            "OpenRouter, Together, Groq or self-hosted vLLM, through base_url. "
            "bedrock calls Amazon Bedrock's Converse API with the machine's own "
            "AWS credentials."
        ),
    )
    model: str = Field(
        min_length=1,
        description="Judge model id; for bedrock, the Bedrock model id (deepseek.v3.2).",
    )
    region: str | None = Field(
        default=None,
        min_length=1,
        description="AWS region a bedrock judge's model runs in (us-east-1).",
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "http(s) endpoint. A URL with a path is the API root, used as given "
            "(https://openrouter.ai/api/v1); a bare host gets the client's default "
            "path. Unset uses the provider's hosted API, which needs api_key_env."
        ),
    )
    api_key_env: str | None = Field(
        default=None,
        pattern=ENV_VAR_NAME_PATTERN,
        description=(
            "Name of the environment variable holding the API key; never the key "
            "itself. Unset sends no key (a local endpoint)."
        ),
    )
    max_concurrency: int = Field(
        default=16, ge=1, description="Judge requests in flight per process."
    )
    timeout_s: float = Field(
        default=120.0,
        gt=0.0,
        description="Seconds one request attempt may take; the provider SDK retries it.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Judge sampling temperature. Anthropic judges take none: leave it unset.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description=(
            "Verdict token budget. A thinking judge spends it on its thinking "
            "too, so a low budget cuts the verdict off."
        ),
    )

    @model_validator(mode="after")
    def _check_endpoint(self) -> Self:
        if self.provider == "bedrock":
            if self.region is None:
                msg = "A bedrock judge needs region."
                raise ValueError(msg)
            if self.base_url is not None or self.api_key_env is not None:
                msg = (
                    "A bedrock judge takes no base_url or api_key_env: it signs "
                    "with the machine's AWS credentials."
                )
                raise ValueError(msg)
            return self
        if self.region is not None:
            msg = f"region is for bedrock judges, not {self.provider}."
            raise ValueError(msg)
        if self.base_url is None:
            if self.api_key_env is None:
                msg = f"The hosted {self.provider} API needs api_key_env."
                raise ValueError(msg)
        else:
            url = urlsplit(self.base_url)
            if url.scheme not in ("http", "https") or not url.hostname:
                msg = f"judge base_url must be an http(s) URL, got {self.base_url!r}."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_temperature(self) -> Self:
        if self.provider == "anthropic" and self.temperature != 0.0:
            msg = "Anthropic judges take no temperature; leave it unset."
            raise ValueError(msg)
        return self
