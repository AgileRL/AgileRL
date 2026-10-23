# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""A verifiers v1 env with a tool, for tests: add with the ``add`` tool, then answer."""

from __future__ import annotations

import verifiers.v1 as vf

__all__ = ["CalcTaskset", "CalcToolset"]

PROMPT = "What is 2 + 3 + 4? Use the add tool, then answer with the number."
ANSWER = "9"
SYSTEM_PROMPT = "Answer with one number."


class CalcToolset(vf.Toolset[vf.ToolsetConfig, vf.State]):
    TOOL_PREFIX = "calc"

    @vf.tool
    async def add(self, a: int, b: int) -> int:
        """Add two integers."""
        return a + b


class CalcTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class CalcTask(vf.Task[vf.TaskData, vf.State, CalcTaskConfig]):
    @classmethod
    def toolsets(cls, config: CalcTaskConfig) -> list[vf.Toolset]:
        return [CalcToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return float(trace.last_reply.strip() == ANSWER)


class CalcTasksetConfig(vf.TasksetConfig):
    task: CalcTaskConfig = CalcTaskConfig()


class CalcTaskset(vf.Taskset[CalcTask, CalcTasksetConfig]):
    def load(self):
        yield CalcTask(
            vf.TaskData(
                idx=0, name="calc#0", prompt=PROMPT, system_prompt=SYSTEM_PROMPT
            ),
            self.config.task,
        )


if __name__ == "__main__":
    CalcToolset.run()
