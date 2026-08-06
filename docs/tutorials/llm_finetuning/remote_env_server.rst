.. _remote_env_server_tutorial:

Serving Environments Remotely (WebSocket Sessions)
==================================================

Your environment does not have to live on the GPU that trains against it. AgileRL can drive a
:class:`~agilerl.llm_envs.rollout.RolloutHarness` over a URL, so you can **host the environment on
one machine and train on another** — for example, run the env on your laptop (or a CPU box, or a
container that ships a proprietary simulator) and launch the training job on a rented GPU server that
connects back to it.

The key property is that a **single URL serves the whole rollout group concurrently**. When the trainer
needs ``batch_size * group_size`` rollouts, it opens that many independent **WebSocket sessions** to the
one server (plus one more for the evaluation env, built lazily at the first evaluation), and the server
spins up one isolated environment per session — automatically. You never orchestrate the fan-out
yourself.

This tutorial hosts ``game:Sudoku-v0-easy`` (from `GEM <https://github.com/axon-rl/gem>`_) on a laptop
and trains ``Qwen/Qwen2.5-1.5B-Instruct`` with CISPO on a GPU server against it.

When to host the environment remotely
-------------------------------------

- The env has heavy or awkward dependencies you don't want on the training box (a game engine, a browser,
  a proprietary simulator, a tool/MCP server).
- You want to iterate on env code without redeploying the trainer, or share one hosted env across runs.
- The env is genuinely a service (an already-hosted `OpenEnv <https://github.com/meta-pytorch/OpenEnv>`_
  Space or container). In that case you simply point ``env_url`` at it.

How one server hosts many environments
--------------------------------------

This is worth understanding, because it is what makes a single URL enough. It is configured entirely on
the :class:`~agilerl.llm_envs.openenv.OpenEnvServer` we host with — you give it two things:

- ``make_env`` — a zero-argument factory that builds **one fresh environment**.
- ``max_concurrent_envs`` — the maximum number of live sessions (environments) at once.

``OpenEnvServer`` hands those to OpenEnv's application, which serves a persistent ``/ws`` WebSocket
endpoint. From there the behaviour is automatic:

- Each client that connects to ``/ws`` starts its **own session**. The server calls your ``make_env()``
  **once** for that session to build a fresh, isolated environment, and routes every ``reset`` / ``step``
  message on that connection to *that* environment for the life of the connection.
- When the connection closes, the server tears the environment down (an idle-session reaper also reclaims
  ones that go quiet). ``max_concurrent_envs`` bounds how many can be live simultaneously.

So the connection *is* the episode's identity. That is the difference from plain HTTP/REST, where each
request is independent and one shared environment can only track a single episode at a time. (OpenEnv's
REST ``/reset`` · ``/step`` · ``/state`` routes also exist only in *simulation* mode, whereas ``/ws`` is
available in production too — so the session client is also the only one that can reach a real production
deployment.)

The upshot for training:

.. code-block:: text

    trainer (GPU box)                         env server (laptop)
    ┌───────────────────────┐                 ┌──────────────────────────────┐
    │ RolloutCollector        │  ws session 0   │ make_env() -> Sudoku board 0 │
    │  (batch_size *         │ ═══════════════▶│ make_env() -> Sudoku board 1 │
    │   group_size = 8       │  ...            │        ...                   │
    │   rollout slots)       │  ws session 7   │ make_env() -> Sudoku board 7 │
    └───────────────────────┘ ═══════════════▶└──────────────────────────────┘
              one URL, max_concurrent_envs >= 8

Group members share a seed, so every session in a group resets to the *same* puzzle — exactly what
GRPO/CISPO need — while different groups get different puzzles.

Step 1 — Host the environment
-----------------------------

On the machine that will host the env (here, a laptop), install the environment package and AgileRL's LLM
extra, then serve the env with :class:`~agilerl.llm_envs.openenv.OpenEnvServer`:

.. code-block:: bash

    pip install "agilerl[llm]" gem-llm    # gem-llm provides the Sudoku env

.. code-block:: python

    # host_env.py — run this on the machine that owns the environment.
    import threading

    import gem

    from agilerl.llm_envs.openenv import OpenEnvServer


    def make_env():
        # Called once per WebSocket session -> one fresh, isolated board each.
        return gem.make("game:Sudoku-v0-easy")


    server = OpenEnvServer(
        make_env=make_env,
        host="127.0.0.1",       # loopback; exposed to the GPU box via an SSH tunnel below
        port=8000,
        env_name="sudoku-easy",
        max_concurrent_envs=16,  # >= batch_size * group_size + 1 (eval) on the trainer
    ).start()

    print(f"Serving {server.base_url} (up to 16 concurrent sessions)")
    try:
        threading.Event().wait()  # serve until Ctrl-C
    except KeyboardInterrupt:
        server.stop()

Run it:

.. code-block:: bash

    python host_env.py
    # Serving http://127.0.0.1:8000 (up to 16 concurrent sessions)

.. note::

    Set ``max_concurrent_envs`` to at least ``batch_size * group_size + 1`` — the ``+ 1`` covers the
    lazily built evaluation env's session. If the trainer opens more sessions than the server allows,
    the extra connections are rejected.

Step 2 — Expose the URL to the GPU server
-----------------------------------------

The training job runs on a different machine, so it needs to reach the env server. The simplest, most
private option is an **SSH reverse tunnel**: it forwards a port on the GPU box back to the laptop, so no
service is exposed to the public internet.

.. code-block:: bash

    # From the laptop: forward the GPU box's localhost:8000 to the laptop's env server.
    ssh -R 8000:127.0.0.1:8000 gpu-server

Leave that session open. Now, on ``gpu-server``, ``http://127.0.0.1:8000`` reaches the laptop's env
server. (For a LAN or a shared server, bind the host with ``host="0.0.0.0"`` and use its address
directly, or use a tunnelling service — but treat exposing an env server the same as exposing any other
service.)

Step 3 — Point the training run at the URL
------------------------------------------

In the run manifest, the environment is declared with ``env_url`` instead of an in-process ``entrypoint``.
Because a remote env's turn budget can't be probed, ``max_turns`` is required. Everything else is an
ordinary rollout config:

.. code-block:: yaml

    algorithm:
        name: CISPO
        batch_size: 2
        group_size: 4          # 2 * 4 = 8 rollout sessions (+1 for eval)
        lr: 0.00002
        max_output_tokens: 64
        use_vllm: true
        quantization: nf4
        vllm_config:
            gpu_memory_utilization: 0.7   # give vLLM room on a small GPU
            max_num_seqs: 8
            sleep_mode: true

    environment:
        env_type: rollout
        env_url: http://127.0.0.1:8000    # reached over the SSH reverse tunnel
        max_turns: 4

    training:
        max_steps: 16
        pop_size: 1

    network:
        pretrained_model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
        lora_config:
            lora_r: 8
            lora_alpha: 32
            target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]
            task_type: CAUSAL_LM

Under the hood, the ``env_url`` field builds each rollout slot as
``RolloutHarness(RemoteEnvClient(url, ...))`` — one WebSocket session per slot. Each message over a
session is bounded by ``request_timeout_s`` (300 seconds unless the manifest sets it; ``0`` disables
the bound).

Step 4 — Launch training on the GPU server
------------------------------------------

LLM training uses DeepSpeed, so launch through ``accelerate`` with a DeepSpeed config (a single-GPU one
ships in ``configs/accelerate/``):

.. code-block:: bash

    accelerate launch \
        --config_file configs/accelerate/bench_accelerate_config.yaml \
        -m agilerl.train path/to/manifest.yaml --device cuda

As the run starts you will see the env server log a burst of sessions opening — one per rollout slot — as
the first rollout begins:

.. code-block:: text

    Serving http://127.0.0.1:8000 (up to 16 concurrent sessions)
    ... training loads the model, starts vLLM, then:
    Algorithm: CISPO | Env: http://127.0.0.1:8000 | Pop size: 1 | Steps: 16 | Device: cuda

Each training step now generates completions on the GPU, sends the decoded moves to the laptop over the
tunnel, and scores them there — all eight Sudoku boards advancing independently, in lock-step turns, from
one URL.

Troubleshooting
---------------

- **"No available memory for the cache blocks"** — vLLM could not fit its KV cache. On a small GPU raise
  ``vllm_config.gpu_memory_utilization`` (0.7 is a good start on a 24 GB card); the trainer is offloaded
  while vLLM runs, so it can take a large share.
- **Sessions rejected / capacity errors** — raise ``max_concurrent_envs`` on the server to at least
  ``batch_size * group_size + 1``.
- **Connection refused from the GPU box** — the tunnel or server isn't up. Verify with a quick client on
  the GPU box: ``python -c "from agilerl.llm_envs.openenv import RemoteEnvClient as C; print(C('http://127.0.0.1:8000').reset(seed=0)[0][:80])"``.

See also
--------

- :ref:`env_grpo_ppo_tutorial` — the same GEM-based multi-turn task run in-process.
- :ref:`grpo_tutorial` — GRPO fine-tuning fundamentals.
