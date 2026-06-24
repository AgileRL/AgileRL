from __future__ import annotations

from peft import LoraConfig


def lora_config_from_dict(lora: dict) -> LoraConfig:
    kw: dict = {
        "r": lora["r"],
        "lora_alpha": lora["lora_alpha"],
        "target_modules": list(lora["target_modules"]),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if lora.get("modules_to_save"):
        kw["modules_to_save"] = list(lora["modules_to_save"])
    return LoraConfig(**kw)
