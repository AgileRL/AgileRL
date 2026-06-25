"""Throwaway: host a reasoning env on an OpenEnv server and print its URL.

The point: the env runs as its own OpenEnv ``HTTPEnvServer`` process, and training
reaches it *purely by URL* — no env code in the trainer. There is no built-in dataset
env; you define your own (the ``ReasoningServerEnv`` below is the whole contract the
server needs) and serve it. Run this, then point the trainer at the printed URL:

    # terminal 1 — the env server
    python scripts/local/serve_openenv_reasoning.py
    # -> OpenEnv server hosting a reasoning env at http://127.0.0.1:PORT

    # terminal 2 — train against that URL (same box)
    AGILERL_OPENENV_URL=http://127.0.0.1:PORT BATCH_SIZE=1 \
      accelerate launch --config_file configs/accelerate/bench_accelerate_config.yaml \
      --num_processes 1 benchmarking/benchmarking_llm_reasoning.py

Note: one server hosts one env instance, so over the REST client it is correct for a
single in-flight rollout (``BATCH_SIZE=1``, or a GRPO group that shares one prompt).
For a wide batch of *different* prompts, give each rollout its own server, or use
OpenEnv WebSocket sessions for per-session isolation.
"""

import time

from agilerl.llm_envs import OpenEnvServer

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
