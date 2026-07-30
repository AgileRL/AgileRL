# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Throwaway: host a reasoning env on an OpenEnv server and print its URL.

Training reaches the env by URL only — no env code in the trainer. Define your own
env (``ReasoningServerEnv`` below is the whole contract) and serve it:

    # terminal 1 — the env server
    python scripts/local/serve_openenv_reasoning.py
    # -> OpenEnv server hosting a reasoning env at http://127.0.0.1:PORT

    # terminal 2 — set ``env_url`` in the run manifest to that URL, then train
    accelerate launch -m agilerl.train configs/training/llm_finetuning/grpo_env.yaml

One shared env means one ``/ws`` session at a time — fine for a single rollout
(a GRPO group sharing one prompt). For a wide batch of different prompts, pass
``make_env`` + ``max_concurrent_envs`` to :class:`OpenEnvServer`.
"""

import time

from agilerl.llm_envs.openenv import OpenEnvServer

QUESTIONS = ["What is 2+2?", "Capital of France?", "What is 10-3?"]
ANSWERS = ["4", "Paris", "7"]


class ReasoningServerEnv:
    """A tiny prompt-dataset env — the whole contract the OpenEnv server needs.

    ``reset`` serves a prompt for ``row_index``; ``step`` scores the completion.
    Define your own env the same way and serve it; ``RolloutEnv(url)`` drives it.
    """

    def __init__(self, questions: list[str], answers: list[str]) -> None:
        self.questions = questions
        self.answers = answers
        self._answer = ""

    @property
    def dataset_size(self) -> int:
        return len(self.questions)

    def reset(self, seed=None, *, row_index=0, evaluation=None):
        del seed, evaluation
        i = (row_index or 0) % len(self.questions)
        self._answer = self.answers[i]
        return f"Question: {self.questions[i]}\nAnswer:", {}

    def step(self, action: str):
        reward = float(self._answer.lower() in action.lower())
        return "", reward, True, False, {}


def main() -> None:
    """Serve a tiny reasoning env over OpenEnv and block until Ctrl-C."""
    server = OpenEnvServer(ReasoningServerEnv(QUESTIONS, ANSWERS)).start()
    print(f"OpenEnv server hosting a reasoning env at {server.base_url}", flush=True)
    print(
        "Set AGILERL_OPENENV_URL to this and run the trainer; Ctrl-C to stop.",
        flush=True,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
