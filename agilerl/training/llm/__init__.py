# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLM finetuning entry points.

Two loops, matching the LLM env taxonomy:

* :func:`train_llm_rollout` -- generative rollout RL over a ``RolloutHarness``
  (GRPO / PPO / REINFORCE; reasoning is ``max_turns=1``).
* :func:`train_llm_dataset` -- teacher-forced dataloader training over a
  ``DatasetEnv`` (preference / SFT).
"""

from agilerl.training.llm.dataset import train_llm_dataset
from agilerl.training.llm.rollout import train_llm_rollout

__all__ = ["train_llm_dataset", "train_llm_rollout"]
