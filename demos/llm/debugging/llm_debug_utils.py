# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from peft import LoraConfig


def lora_config_from_dict(lora: dict) -> LoraConfig:
    kw: dict = {
        "r": lora["r"],
        "lora_alpha": lora["lora_alpha"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if lora.get("target_modules") is not None:
        kw["target_modules"] = list(lora["target_modules"])
    if lora.get("target_parameters") is not None:
        kw["target_parameters"] = list(lora["target_parameters"])
    if "lora_dropout" in lora:
        kw["lora_dropout"] = float(lora["lora_dropout"])
    if lora.get("modules_to_save"):
        kw["modules_to_save"] = list(lora["modules_to_save"])
    return LoraConfig(**kw)
