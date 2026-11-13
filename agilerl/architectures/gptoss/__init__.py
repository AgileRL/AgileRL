# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS."""

from __future__ import annotations

from agilerl.architectures.runtime import ModelRuntimeConfig, ModelTypes
from agilerl.arena.models.model_info import REASONING_PARSERS

FAMILY = ModelRuntimeConfig(
    name="gpt_oss",
    model_types=ModelTypes(text=("gpt_oss",)),
    chat_format="harmony",
    reasoning_parser=REASONING_PARSERS["gpt_oss"],
    tensor_parallel_plan="agilerl.architectures.gptoss.tensor_parallel:TENSOR_PARALLEL_PLAN",
)
