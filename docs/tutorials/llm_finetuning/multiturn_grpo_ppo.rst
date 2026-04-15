.. _multiturn_grpo_ppo_tutorial:

Multi-turn finetuning with LLMPPO, LLMReinforce, and GRPO
=========================================================

In this tutorial, we train three LLM reinforcement learning agents on the same multi-turn GEM task:
``LLMPPO``, ``LLMReinforce``, and ``GRPO``. The environment, model, tokenizer, and training loop are kept fixed so you can
compare algorithm behavior directly.

The task is ``game:GuessTheNumber-v0-easy``, where the agent has to guess a number and gets iterative feedback over multiple turns, 
with the goal of converging to the correct answer. The task is simple but useful for illustrating how LLMs can be fine tuned for
multi-turn agentic tasks.

Credit assignment: MDP vs bandit formulations
----------------------------------------------

The core difference between these algorithms is how they answer the question:
*which actions caused the outcome?*

MDP formulation — ``LLMPPO`` and ``LLMReinforce``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These treat multi-turn interaction as a sequential decision process. Each
turn is a timestep, each agent response is an action, and rewards are assigned
back through the trajectory using temporal structure.

**LLMPPO** uses ``turn_ids`` to broadcast per-turn rewards to their constituent
tokens, then fits a turn-level value function V(s_t) and computes GAE returns
across turn transitions:

.. math::

   A_t = \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta_{t+k}, \quad
   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)

This means the policy receives a *differentiated* gradient signal and early turns
that led to a good outcome are credited separately from late turns that executed
it. The value function explicitly models how good a conversational state is,
independent of what actually happened afterward.

**LLMReinforce** maps per-turn rewards through ``turn_ids`` and normalises them
with return batch normalisation (ReBN) rather than fitting a value function.
Cheaper to run than PPO, but higher variance, there is no baseline to reduce
the noise in early-turn credit estimates.

Both approaches make a strong assumption: that the transition structure
matters and that knowing you are at turn 3 of 5, having received a tool error
at turn 2, is useful signal for the update. This is the right assumption for
long agentic trajectories where early decisions constrain later ones.

Bandit formulation — ``GRPO``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRPO collapses the multi-turn trajectory into a single episode return and
optimises relative to a group of peers sampled from the same prompt:

.. math::

   A_i = \frac{R_i - \text{mean}(R_{1..G})}{\text{std}(R_{1..G}) + \epsilon}

There is no value function, no GAE, and no turn-level structure. The entire
conversation is treated as a single bandit arm. The gradient signal is
*undifferentiated*: every token in the episode receives the same advantage,
whether it was a pivotal early decision or a closing punctuation mark.

This is theoretically a mismatch for multi-turn tasks, but works in practice
because:

* The group-relative baseline (mean/std over ``GROUP_SIZE`` peers) is a much
  lower-variance estimator than a single-sample return, compensating for the
  lack of temporal structure.
* For tasks where the correct action at each turn is relatively unambiguous
  given the context, undifferentiated credit is sufficient and the policy just
  needs to know whether the episode succeeded, not *why*.
* No critic means no value function fitting overhead, which matters at scale.

The practical tradeoff
~~~~~~~~~~~~~~~~~~~~~~

+---------------------+-----------------------------+---------------------------+
| Property            | MDP (PPO / Reinforce)       | Bandit (GRPO)             |
+=====================+=============================+===========================+
| Credit assignment   | Per-turn via GAE / ReBN     | Episode-level, uniform    |
+---------------------+-----------------------------+---------------------------+
| Variance            | Lower (value baseline)      | Higher per-sample,        |
|                     |                             | lower via group averaging |
+---------------------+-----------------------------+---------------------------+
| Critic required     | Yes (PPO) / No (Reinforce)  | No                        |
+---------------------+-----------------------------+---------------------------+
| Best for            | Long trajectories, sparse   | Short–medium episodes,    |
|                     | rewards, early turns matter | dense or terminal reward  |
+---------------------+-----------------------------+---------------------------+
| Main failure mode   | Value fn divergence on      | Reward hacking via        |
|                     | long horizons               | episode-level shortcuts   |
+---------------------+-----------------------------+---------------------------+

For agentic tasks with tool use and multi-step reasoning, where a wrong
decision at turn 2 makes success at turn 5 structurally impossible, the MDP
formulation is the more principled choice. GRPO becomes competitive when
episodes are short enough that the bandit approximation is tight, or when
sampling efficiency (many group peers per prompt) compensates for the coarser
credit signal.

Dependencies
------------

.. code-block:: bash

    pip install -U agilerl[llm] gem


.. code-block:: python

    import gem
    import yaml
    from transformers import AutoTokenizer
    from agilerl.training.train_llm import finetune_llm_multiturn
    from agilerl.utils.algo_utils import VLLMConfig
    from agilerl.utils.llm_utils import create_llm_accelerator
    from agilerl.utils.utils import create_population
    from agilerl.wrappers.multiturn_wrappers import TokenObservationWrapper

Shared setup
------------

All runs use:

* Environment: ``game:GuessTheNumber-v0-easy``
* Model: ``Qwen/Qwen2.5-0.5B-Instruct``
* Wrapper: :class:`TokenObservationWrapper <agilerl.wrappers.multiturn_wrappers.TokenObservationWrapper>`
* Training loop: :meth:`finetune_llm_multiturn() <agilerl.training.train_llm.finetune_llm_multiturn>`
* Population size: ``1``
* Evolution/HPO: disabled

.. collapse:: Core setup code

    .. code-block:: python

        MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
        ENV_NAME = "game:GuessTheNumber-v0-easy"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        env_probe = gem.make(ENV_NAME)
        max_turns = env_probe.max_turns
        if hasattr(env_probe, "close"):
            env_probe.close()

        def env_factory():
            env = gem.make(ENV_NAME)
            return TokenObservationWrapper(
                env=env,
                tokenizer=tokenizer,
                max_turns=max_turns,
                pad_id=tokenizer.pad_token_id,
                apply_chat_template=True,
                max_model_len=INIT_HP.get("MAX_MODEL_LEN"),
                max_output_tokens=INIT_HP.get("MAX_OUTPUT_TOKENS"),
            )

        accelerator = create_llm_accelerator()
        vllm_config = VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_num_seqs=16,
            sleep_mode=True,
        )

Run LLMPPO baseline
-------------------

Use the LLMPPO multiturn config as a base and keep ``ALGO=LLMPPO``.

.. code-block:: bash

    python tutorials/llm_finetuning/multiturn_grpo_ppo.py \
      --algo LLMPPO \
      --config configs/training/llm_finetuning/ppo_llm.yaml \
      --max-steps 4096 \
      --evaluation-interval 10 \
      --output-dir saved_llms/multiturn_ppo

Run LLMReinforce baseline
-------------------------

Use the LLMReinforce config and set ``ALGO=LLMReinforce``.

.. code-block:: bash

    python tutorials/llm_finetuning/multiturn_grpo_ppo.py \
      --algo LLMReinforce \
      --config configs/training/llm_finetuning/reinforce_llm.yaml \
      --max-steps 4096 \
      --evaluation-interval 10 \
      --output-dir saved_llms/multiturn_reinforce

Run GRPO baseline
-----------------

Use the GRPO multiturn config and set ``ALGO=GRPO``.

.. code-block:: bash

    python tutorials/llm_finetuning/multiturn_grpo_ppo.py \
      --algo GRPO \
      --config configs/training/llm_finetuning/grpo_multiturn.yaml \
      --max-steps 4096 \
      --evaluation-interval 10 \
      --output-dir saved_llms/multiturn_grpo

Starter hyperparameters (good first run values)
-----------------------------------------------

These values are intentionally conservative and align with the shipped configs:

.. collapse:: Suggested ``INIT_HP`` starting points

    .. code-block:: python

        # LLMPPO
        INIT_HP_PPO = {
            "ALGO": "LLMPPO",
            "BATCH_SIZE": 32,
            "LR_ACTOR": 5e-6,
            "LR_CRITIC": 5e-5,
            "BETA": 0.01,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "VF_COEF": 0.5,
            "UPDATE_EPOCHS": 2,
            "MAX_MODEL_LEN": 1024,
            "MAX_OUTPUT_TOKENS": 64,
            "USE_VLLM": True,
            "MICRO_BATCH_SIZE_PER_GPU": 32,
        }

        # LLMReinforce
        INIT_HP_REINFORCE = {
            "ALGO": "LLMReinforce",
            "BATCH_SIZE": 32,
            "LR": 5e-6,
            "BETA": 0.01,
            "GAMMA": 0.9,
            "UPDATE_EPOCHS": 2,
            "MAX_MODEL_LEN": 1024,
            "MAX_OUTPUT_TOKENS": 64,
            "USE_VLLM": True,
            "MICRO_BATCH_SIZE_PER_GPU": 32,
        }

        # GRPO multiturn
        INIT_HP_GRPO = {
            "ALGO": "GRPO",
            "BATCH_SIZE": 16,
            "GROUP_SIZE": 4,
            "LR": 3e-4,
            "BETA": 5e-4,
            "UPDATE_EPOCHS": 2,
            "TEMPERATURE": 0.85,
            "MAX_MODEL_LEN": 2048,
        }

.. note::

   For GRPO, ``BATCH_SIZE`` and ``GROUP_SIZE`` must satisfy divisibility constraints in
   :meth:`finetune_llm_multiturn() <agilerl.training.train_llm.finetune_llm_multiturn>`.

Train call (no evo/HPO)
-----------------------

The key training call is the same for both algorithms. Evolutionary fields are explicitly disabled:

.. code-block:: python

    finetune_llm_multiturn(
        pop=[agent],
        max_turns=max_turns,
        env_factory=env_factory,
        init_hp=INIT_HP,
        max_steps=4096,
        save_elite=True,
        elite_path="saved_llms/multiturn_tutorial",
        wb=False,
        evo_steps=None,
        tournament=None,
        mutation=None,
        evaluation_interval=10,
        max_reward=1.0,
        verbose=True,
        accelerator=accelerator,
    )

Full training code
------------------

.. collapse:: Full code

   .. literalinclude:: ../../../tutorials/llm_finetuning/multiturn_grpo_ppo.py
      :language: python
