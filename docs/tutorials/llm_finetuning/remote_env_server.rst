.. _remote_env_server_tutorial:

Serving Environments Remotely (WebSocket Sessions)
==================================================

Your environment does not have to live on the GPU that trains against it. AgileRL can drive a
:class:`~agilerl.llm_envs.rollout.RolloutHarness` over a URL, so you can **host the environment on
one machine and train on another** — for example, run the env on a CPU box (or your laptop, or a
container that ships a proprietary simulator) and launch the training job on a rented GPU server
that connects back to it.

The key property is that a **single URL serves the whole rollout group concurrently**. When the trainer
needs ``batch_size * group_size`` rollouts, it opens that many independent **WebSocket sessions** to the
one server (plus one more for the evaluation env, built lazily at the first evaluation), and the server
spins up one isolated environment per session — automatically. You never orchestrate the fan-out
yourself.

This tutorial hosts a **coding environment** — ``code:Taco8k`` from
`GEM <https://github.com/axon-rl/gem>`_ — on a separate machine and trains
``Qwen/Qwen2.5-1.5B-Instruct`` with CISPO on a GPU server against it. A coding env is the case
where remote hosting stops being a convenience and starts being the point.

Why a coding environment wants its own machine
-----------------------------------------------

``code:Taco8k`` hands the model a competitive-programming problem, extracts the code from its
answer, and **runs that code** against the problem's test cases — one subprocess per test, fanned
out over a thread pool. Three consequences, and each one is a reason to keep it off the trainer:

- **It executes model output.** Whatever the policy emits gets run. Early in training that is
  malformed code; it is never code you reviewed.
- **Its real sandbox is Linux-only.** GEM's ``sandbox_type="bwrap"`` runs each test under
  `bubblewrap <https://github.com/containers/bubblewrap>`_ with ``--unshare-all`` and read-only
  binds. The default, ``sandbox_type="none"``, is *not* a sandbox — it is a plain subprocess
  behind a string-match check for a handful of import names, which is a guard against accidents
  rather than against anything deliberate. **Set ``bwrap`` explicitly**, and put it somewhere it
  exists.
- **It is CPU-hungry and dataset-heavy.** Up to 12 test executions per rollout, plus a
  multi-thousand-row dataset resident in memory — none of which wants to compete with vLLM for
  the training box.

The turn budget is ``max_turns: 1``: the model answers once and the environment scores it. That is
a rollout like any other, not a separate kind of environment — the same machinery that drives a
50-turn game drives a one-shot scored answer.

.. note::

   This tutorial demonstrates the **transport and the training loop**, not a learning result.
   Pass/fail on competitive-programming tests is a sparse reward, and a 1.5B model solves very few
   TACO problems, so expect reward to sit near zero over a short run. Use it to prove the path
   works end to end; scale the model and the step count before reading anything into the curve.

How one server hosts many environments
--------------------------------------

:func:`~agilerl.llm_envs.openenv_server.resolve_env` takes the same
``entrypoint`` and ``env_config`` you would put in a local manifest, starts an
:class:`~agilerl.llm_envs.openenv.OpenEnvServer`, and returns its URL.
``max_concurrent_envs`` is how many live sessions it will hold at once.

Each client that connects to ``/ws`` gets its **own session**. The server builds
one fresh, isolated environment for that connection and routes every ``reset`` /
``step`` on it until the connection closes. You do not orchestrate the fan-out.

.. code-block:: text

    trainer (GPU box)                          env server (CPU box)
    ┌────────────────────────┐                 ┌──────────────────────────────┐
    │ RolloutCollector       │  ws session 0   │ gem.make() -> TACO problem 0 │
    │  (batch_size *         │ ═══════════════▶│ gem.make() -> TACO problem 1 │
    │   group_size = 8       │  ...            │        ...                   │
    │   rollout slots)       │  ws session 7   │ gem.make() -> TACO problem 7 │
    └────────────────────────┘ ═══════════════▶└──────────────────────────────┘
              one URL, max_concurrent_envs >= 9   (8 rollouts + 1 eval)

Group members share a seed, so every session in a group resets to the *same* problem — exactly what
GRPO/CISPO need, since the group's completions have to be comparable — while different groups draw
different problems.

Step 1 — Host the environment
-----------------------------

On the machine that will host the env, install the environment packages and AgileRL's LLM extra.
GEM's coding envs need ``datasets`` as well as ``gem-llm`` — it is not pulled in for you:

.. code-block:: bash

    pip install "agilerl[llm]" gem-llm datasets
    # Linux, for the real sandbox:
    sudo apt-get install -y bubblewrap

Then serve it. ``env_id`` and ``sandbox_type`` are the same ``env_config`` you would
use in-process; ``resolve_env`` starts the server:

.. code-block:: python

    # host_env.py — run this on the machine that owns the environment.
    import threading

    from agilerl.llm_envs.openenv_server import resolve_env

    url, server = resolve_env(
        "gem:make",
        env_config={"env_id": "code:Taco8k", "sandbox_type": "bwrap"},
        host="127.0.0.1",        # loopback; exposed to the GPU box via an SSH tunnel below
        port=8000,
        max_concurrent_envs=16,  # >= batch_size * group_size + 1 (eval) on the trainer
    )

    print(f"Serving {url} (up to 16 concurrent sessions)")
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

    ``sandbox_type="bwrap"`` is the one that actually isolates; GEM's default (``"none"``) runs the
    model's code in a plain subprocess. Put that in ``env_config``, not in trainer code.

Step 2 — Expose the URL to the GPU server
-----------------------------------------

The training job runs on a different machine, so it needs to reach the env server. The simplest, most
private option is an **SSH reverse tunnel**: it forwards a port on the GPU box back to the env host, so no
service is exposed to the public internet.

.. code-block:: bash

    # From the env host: forward the GPU box's localhost:8000 to this machine's env server.
    ssh -R 8000:127.0.0.1:8000 gpu-server

Leave that session open. Now, on ``gpu-server``, ``http://127.0.0.1:8000`` reaches the env
server. (For a LAN or a shared server, bind the host with ``host="0.0.0.0"`` and use its address
directly, or use a tunnelling service — but treat exposing an env server the same as exposing any other
service. An env that executes code deserves more care here, not less.)

Step 3 — Point the training run at the URL
------------------------------------------

In the run manifest, the environment is declared with ``env_url`` instead of an in-process ``entrypoint``.
Because a remote env's turn budget can't be probed, ``max_turns`` is required — here it is ``1``, because
the env scores a single answer. Everything else is an ordinary rollout config:

.. code-block:: yaml

    algorithm:
        name: CISPO
        batch_size: 2
        group_size: 4          # 2 * 4 = 8 rollout sessions (+1 for eval)
        lr: 0.00002
        max_output_tokens: 512  # room for a whole program, not one move
        use_vllm: true
        quantization: nf4
        vllm_config:
            gpu_memory_utilization: 0.7   # give vLLM room on a small GPU
            max_num_seqs: 8
            sleep_mode: true

    environment:
        env_type: rollout
        env_url: http://127.0.0.1:8000    # reached over the SSH reverse tunnel
        max_turns: 1
        max_reward: 1.0
        request_timeout_s: 600            # running the tests takes longer than a game move

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
the bound). A coding env's ``step`` is slower than a game's, because it runs the tests before it
answers, so give it more headroom than the default.

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

Each training step now generates candidate programs on the GPU, sends them to the env host over the
tunnel, and runs them against the test cases there — eight problems scored in parallel, from one URL,
with none of that execution touching the training box.

Troubleshooting
---------------

- **"No available memory for the cache blocks"** — vLLM could not fit its KV cache. On a small GPU raise
  ``vllm_config.gpu_memory_utilization`` (0.7 is a good start on a 24 GB card); the trainer is offloaded
  while vLLM runs, so it can take a large share.
- **Sessions rejected / capacity errors** — raise ``max_concurrent_envs`` on the server to at least
  ``batch_size * group_size + 1``.
- **"bwrap: command not found"** — bubblewrap is not installed, or the host is not Linux. Install it, or
  move the env to a Linux box. Falling back to ``sandbox_type="none"`` means model-generated code runs
  unconfined on that machine.
- **Every reward is 0.0** — expected for a small model on TACO (see the note above). To confirm the path
  rather than the policy, check the env host's logs for sessions opening and test executions running.
- **Timeouts on step** — running a program against a dozen tests is slower than a game move. Raise
  ``request_timeout_s``.
- **Connection refused from the GPU box** — the tunnel or server isn't up. Verify with a quick client on
  the GPU box: ``python -c "from agilerl.llm_envs.openenv import RemoteEnvClient as C; print(C('http://127.0.0.1:8000').reset(seed=0)[0][:80])"``.

See also
--------

- :ref:`env_grpo_ppo_tutorial` — a multi-turn game environment run in-process.
- :ref:`llm_environments` — the environment interface, and every way a manifest can name one.
- :ref:`grpo_tutorial` — GRPO fine-tuning fundamentals.
