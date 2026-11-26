# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""A verifiers v1 env for tests: guess each task's secret word in up to three tries."""

from __future__ import annotations

import verifiers.v1 as vf

__all__ = ["GuessEnv", "GuessTaskset"]

SECRETS = ["apple", "pear"]
SYSTEM_PROMPT = "Answer with one lowercase word."
MAX_TRIES = 3


class GuessData(vf.TaskData):
    info: dict


class GuessTask(vf.Task[GuessData, vf.State, vf.TaskConfig]):
    pass


class GuessEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig()


class GuessEnv(vf.Env[GuessEnvConfig]):
    async def run(self, task, agents):
        secret = task.data.info["secret"]
        async with agents.player.interaction(task) as interaction:
            segment = await interaction.turn()
            for _ in range(MAX_TRIES - 1):
                if segment.terminated or segment.last_reply == secret:
                    break
                segment = await interaction.turn("Wrong, guess again.")
        trace = interaction.trace
        trace.record_reward("correct", float(trace.last_reply == secret))


class GuessTaskset(vf.Taskset[GuessTask, vf.TasksetConfig]):
    def load(self):
        for idx, secret in enumerate(SECRETS):
            yield GuessTask(
                GuessData(
                    idx=idx,
                    name=f"guess#{idx}",
                    prompt=f"Guess fruit number {idx}.",
                    system_prompt=SYSTEM_PROMPT,
                    info={"secret": secret},
                ),
                self.config.task,
            )
