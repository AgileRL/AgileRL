"""Throwaway: host a reasoning env on an OpenEnv server and print its URL.

The point: the env runs as its own OpenEnv ``HTTPEnvServer`` process, and training
reaches it *purely by URL* — no env code in the trainer. Run this, then point the
trainer at the printed URL:

    # terminal 1 — the env server
    python scripts/local/serve_openenv_reasoning.py
    # -> OpenEnv server hosting ReasoningEnv at http://127.0.0.1:PORT

    # terminal 2 — train against that URL (same box)
    AGILERL_OPENENV_URL=http://127.0.0.1:PORT BATCH_SIZE=1 \
      accelerate launch --config_file configs/accelerate/bench_accelerate_config.yaml \
      --num_processes 1 benchmarking/benchmarking_llm_reasoning.py

Note: one server hosts one env instance, so over the REST client it is correct for a
single in-flight rollout (``BATCH_SIZE=1``, or a GRPO group that shares one prompt).
For a wide batch of *different* prompts, give each rollout its own server (the
``AGILERL_OPENENV_HTTP=1`` path serves one per ``env_factory()`` call) or use OpenEnv
WebSocket sessions for per-session isolation.
"""

import time

from agilerl.llm_envs import ReasoningEnv, serve


def reward_fn(completion: str, answer: str, question: str) -> float:
    """+1 if the gold answer appears in the completion."""
    del question
    return float(answer.lower() in completion.lower())


def main() -> None:
    """Serve a tiny reasoning env over OpenEnv and block until Ctrl-C."""
    env = ReasoningEnv(
        questions=["What is 2+2?", "Capital of France?", "What is 10-3?"],
        answers=["4", "Paris", "7"],
        reward_fn=reward_fn,
        prompt_builder=lambda q: f"Question: {q}\nAnswer:",
    )
    server = serve(env)  # OpenEnv HTTPEnvServer (uvicorn, ephemeral port)
    print(f"OpenEnv server hosting ReasoningEnv at {server.base_url}", flush=True)
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
