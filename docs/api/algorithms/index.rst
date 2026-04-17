.. _algorithms:

Algorithms
==========

AgileRL already includes state-of-the-art evolvable on-policy, off-policy, offline and multi-agent reinforcement learning algorithms with distributed training. We are constantly adding more algorithms, with a view to add hierarchical algorithms soon.

Core algorithm tools:

.. toctree::
   :maxdepth: 1

   base
   registry
   wrappers

Observation Spaces
------------------

AgileRL currently supports processing the following observation spaces:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Box (Continuous)
     - Discrete
     - Dict
     - Tuple
   * - ✅
     - ✅
     - ✅
     - ✅

Single-Agent Algorithms
----------------------

Action Spaces
~~~~~~~~~~~~~

.. list-table::
   :widths: 30 17 17 17 17
   :header-rows: 1

   * - Algorithm
     - ``Discrete``
     - ``Box``
     - ``MultiDiscrete``
     - ``MultiBinary``
   * - :ref:`CQL <cql>`
     - ✅
     - ❌
     - ❌
     - ❌
   * - :ref:`DDPG <ddpg>`
     - ❌
     - ✅
     - ❌
     - ❌
   * - :ref:`DQN <dqn>`
     - ✅
     - ❌
     - ❌
     - ❌
   * - :ref:`DQN Rainbow <dqn_rainbow>`
     - ✅
     - ❌
     - ❌
     - ❌
   * - :ref:`ILQL <ilql>`
     - ✅
     - ❌
     - ❌
     - ❌
   * - :ref:`PPO <ppo>`
     - ✅
     - ✅
     - ✅
     - ✅
   * - :ref:`TD3 <td3>`
     - ❌
     - ✅
     - ❌
     - ❌

.. toctree::
   :maxdepth: 1
   :hidden:

   cql
   ddpg
   dqn
   dqn_rainbow
   ilql
   ppo
   td3
   grpo
   dpo

Multi-Agent Algorithms
---------------------

Action Spaces
~~~~~~~~~~~~~

.. list-table::
   :widths: 30 17 17 17 17
   :header-rows: 1

   * - Algorithm
     - ``Discrete``
     - ``Box``
     - ``MultiDiscrete``
     - ``MultiBinary``
   * - :ref:`IPPO <ippo>`
     - ✅
     - ✅
     - ✅
     - ✅
   * - :ref:`MADDPG <maddpg>`
     - ✅
     - ✅
     - ❌
     - ❌
   * - :ref:`MATD3 <matd3>`
     - ✅
     - ✅
     - ❌
     - ❌

.. toctree::
   :maxdepth: 1
   :hidden:

   ippo
   maddpg
   matd3

Bandit Algorithms
----------------

.. toctree::
   :maxdepth: 1

   neural_ucb
   neural_ts

LLM Algorithms
--------------

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Algorithm
     - Training Mode
     - Notes
   * - :ref:`GRPO <grpo>`
     - On-policy RL
     - Group-relative policy optimization for LLMs.
   * - :ref:`CISPO <cispo>`
     - On-policy RL
     - GRPO specialization with fixed CISPO loss objective.
   * - :ref:`GSPO <gspo>`
     - On-policy RL
     - GRPO specialization with fixed GSPO sequence-level objective.
   * - :ref:`DPO <dpo>`
     - Preference optimization
     - Offline preference learning without reward-model RL loop.
   * - :ref:`LLM PPO <llmppo>`
     - On-policy RL
     - PPO-style actor-critic objective for token-level trajectories.
   * - :ref:`LLM REINFORCE <llmreinforce>`
     - On-policy RL
     - REINFORCE-style policy-gradient objective for LLM finetuning.

.. toctree::
   :maxdepth: 1

   grpo
   cispo
   gspo
   dpo
   llmppo
   llmreinforce
