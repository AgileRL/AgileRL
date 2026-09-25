# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS packed-expert layout and split-LoRA forward."""

from __future__ import annotations

from agilerl.architectures.gptoss.experts import (
    GptOssExpertsLoraWrapper,
    is_gpt_oss_experts_module,
)

__all__ = [
    "GptOssExpertsLoraWrapper",
    "is_gpt_oss_experts_module",
]
