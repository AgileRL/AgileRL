.. _mf_pbt_tutorial:

Lunar Lander with PPO & MF-PBT
==============================

In this tutorial we train a population of PPO agents on the Gymnasium ``LunarLander-v3``
environment, but instead of the default :ref:`tournament selection <tournament_selection>` we drive
the evolutionary loop with **multi-frequency selection**: the selection strategy that, combined
with AgileRL's :ref:`mutation <mutations>` step, yields **Multiple-Frequencies Population-Based
Training** (MF-PBT, `Doulazmi et al. <https://arxiv.org/abs/2506.03225>`_).

Multi-frequency selection is a drop-in replacement for tournament selection, so everything you
already know about AgileRL's HPO loop carries over: you only swap the selection object. We use PPO
here because its on-policy loop is the simplest to read, but the operator is algorithm-agnostic:
the exact same wiring works for DQN, DDPG, IPPO, bandits and offline algorithms.

We show **two** ways to run it:

#. **From a YAML manifest**: let :class:`~agilerl.training.trainer.LocalTrainer` build the
   population, the selection strategy, the mutations and the training loop for you.
#. **In Python**: construct the environment, population, selection strategy and mutations yourself,
   then hand them to :func:`~agilerl.training.train_on_policy.train_on_policy`, which runs the
   training loop.

Multi-Frequency PBT Overview
----------------------------

Rather than evolving the whole population together, MF-PBT splits it into ``n_subpopulations``
subpopulations that each evolve at their **own frequency**: subpopulation ``i`` only evolves every
``evolution_frequency_ratios[i]`` evaluation cycles. Keeping several frequencies alive at once
counteracts the greediness of single-frequency PBT: the slower subpopulations resist premature
convergence, while the faster ones refine promising hyperparameter schedules.

Each time a subpopulation evolves, its members are ranked by fitness into four brackets: **winners**,
**survivors**, **open-for-migration** and **losers**. Every loser is replaced by a clone of a
winner (and only those clones are later perturbed by mutation); winners and survivors are kept
unchanged; and the open-for-migration slots may be filled by stronger agents **migrating** in from
other subpopulations.

MF-PBT shines with **16 or more agents**, where there is room for several subpopulations at
different frequencies. For small populations, prefer :ref:`tournament selection <tournament_selection>`.
See :ref:`multi_frequency_selection` for the full description of the brackets, migration rules and
configuration surface.

Dependencies
------------

.. code-block:: python

    import os

    import imageio
    import gymnasium as gym
    import numpy as np
    import torch

    from agilerl import LocalTrainer
    from agilerl.algorithms import PPO
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.hpo.multi_frequency import MultiFrequencySelection
    from agilerl.hpo.mutation import Mutations
    from agilerl.training.train_on_policy import train_on_policy
    from agilerl.utils.utils import make_vect_envs

Approach 1: From a YAML manifest
--------------------------------

The recommended way to run MF-PBT is through a YAML manifest and
:class:`~agilerl.training.trainer.LocalTrainer`, which handles population creation, rollout
collection, the evolutionary HPO loop and the training loop for you.

Multi-frequency and tournament selection share the single ``selection_strategy`` manifest block
(also accepted under its former name, ``tournament_selection``), disambiguated by a ``strategy``
field that defaults to ``tournament``. To use MF-PBT, set
``strategy: multi_frequency`` and describe the subpopulation layout in that same block. The
population size lives in the ``training`` block as ``pop_size`` (mandatory for MF-PBT); the operator
derives the per-subpopulation size as ``pop_size // n_subpopulations``.

.. collapse:: ppo_mfpbt.yaml

    .. code-block:: yaml

        algorithm:
            name: PPO
            batch_size: 128
            lr: 0.001
            learn_step: 2048
            gamma: 0.99
            gae_lambda: 0.95
            action_std_init: 0.6
            clip_coef: 0.2
            ent_coef: 0.01
            vf_coef: 0.5
            max_grad_norm: 0.5
            update_epochs: 4
            share_encoders: true

        environment:
            name: LunarLander-v3
            num_envs: 16

        network:
            latent_dim: 64
            arch: mlp
            encoder_config:
                hidden_size: [64]
                activation: ReLU
                min_mlp_nodes: 64
                max_mlp_nodes: 500
                layer_norm: true
            head_config:
                hidden_size: [64]
                activation: ReLU
                min_hidden_layers: 1
                max_hidden_layers: 3
                min_mlp_nodes: 64
                max_mlp_nodes: 500
                output_vanish: true
                layer_norm: true

        mutation:
            probabilities:
                no_mut: 0.4
                arch_mut: 0.2
                new_layer: 0.2
                params_mut: 0.2
                act_mut: 0.0
                rl_hp_mut: 0.2
            rl_hp_selection:
                lr:
                    min: 0.0001
                    max: 0.01
                batch_size:
                    min: 8
                    max: 1024
                learn_step:
                    min: 256
                    max: 8192
                ent_coef:
                    min: 0.001
                    max: 0.1
                update_epochs:
                    min: 1
                    max: 10
            mutation_sd: 0.1
            rand_seed: 42

        # MF-PBT replaces tournament selection; both regimes share this block and
        # are disambiguated by `strategy`.
        selection_strategy:
            strategy: multi_frequency
            n_subpopulations: 2                 # >= 2
            n_winners: 2                        # >= 1
            n_survivors: 0                      # >= 0
            n_open_for_migration: 2             # >= 1
            n_losers: 4                         # >= 1
            evolution_frequency_ratios: [1, 5]  # strictly increasing ints, one per subpopulation

        training:
            max_steps: 1_000_000  # trimmed for the tutorial; scale up for real runs
            target_score: 250.0
            pop_size: 16          # >= 6, a multiple of n_subpopulations (2 subpopulations of 8)
            evo_steps: 10_240
            eval_steps:
            eval_loop: 1

Training is then:

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         trainer = LocalTrainer.from_manifest("ppo_mfpbt.yaml")
         population, fitnesses = trainer.train()

   .. tab-item:: CLI

      .. code-block:: bash

         python -m agilerl.train ppo_mfpbt.yaml

Approach 2: Running MF-PBT from Python
--------------------------------------

If you would rather stay in Python, you can build every piece yourself and hand them to
:func:`~agilerl.training.train_on_policy.train_on_policy`, which runs the evolutionary training
loop. The only MF-PBT-specific part is the selection object, everything else is a standard AgileRL
on-policy run.

Defining Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

As with any AgileRL run, we collect the algorithm hyperparameters in one dictionary and declare which
of them are eligible for mutation via a :class:`~agilerl.algorithms.core.registry.HyperparameterConfig`.

.. collapse:: Hyperparameter Configuration

    .. code-block:: python

        # A simple MLP with two 64-node hidden layers
        net_config = {"head_config": {"hidden_size": [64, 64]}}

        # RL hyperparameters that mutation is allowed to tune, with their search ranges
        hp_config = HyperparameterConfig(
            lr=RLParameter(min=1e-4, max=1e-2),
            batch_size=RLParameter(min=8, max=1024),
            learn_step=RLParameter(min=256, max=8192, grow_factor=1.5, shrink_factor=0.75),
            ent_coef=RLParameter(min=1e-3, max=1e-1, grow_factor=1.0, shrink_factor=0.9),
        )

        # Algorithm hyperparameters
        init_hp = {
            "batch_size": 128,
            "lr": 0.001,
            "learn_step": 2048,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "action_std_init": 0.6,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
            "update_epochs": 4,
            "num_envs": num_envs,
            "hp_config": hp_config,
            "net_config": net_config,
        }

Create the Environment
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    num_envs = 16
    env = make_vect_envs("LunarLander-v3", num_envs=num_envs)  # Create environment

    observation_space = env.single_observation_space
    action_space = env.single_action_space

Create a Population of Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MF-PBT needs a reasonably large population; we use 16 agents so the two subpopulations each hold 8.

.. code-block:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    population_size = 16
    pop = PPO.population(
        size=population_size,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **init_hp,
    )

Create the MF-PBT Selection Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the only line that differs from a tournament-selection run: we build a
:class:`~agilerl.hpo.multi_frequency.MultiFrequencySelection` instead of a
:class:`~agilerl.hpo.tournament.TournamentSelection`. Its constructor arguments map one-to-one onto
the manifest block above.

.. code-block:: python

    mf_selection = MultiFrequencySelection(
        population_size=population_size,
        n_subpopulations=2,
        n_winners=2,
        n_survivors=0,
        n_open_for_migration=2,
        n_losers=4,
        evolution_frequency_ratios=[1, 5],
        seed=42,
    )

A few things worth knowing:

* **Subpopulations are assigned automatically.** The operator tags each agent with its subpopulation
  on the first call to :meth:`~agilerl.hpo.multi_frequency.MultiFrequencySelection.select`,so you do
  not need to tag the agents yourself.
* **The seed is explicit here.** It seeds the RNG used to pick which winner replaces each loser. In
  the manifest path the trainer derives this seed from ``mutation.rand_seed`` instead.
* **Constraints.** ``population_size`` must be ``>= 6`` and a multiple of ``n_subpopulations``, and
  the four bracket sizes must sum to ``population_size // n_subpopulations`` (here ``2 + 0 + 2 + 4 = 8``).
  ``evolution_frequency_ratios`` must be strictly increasing integers ``>= 1``, one per subpopulation.

Create the Mutations Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mutation is unchanged from any other AgileRL run. MF-PBT nominates *which* agents to perturb, and
:class:`~agilerl.hpo.mutation.Mutations` decides *how*.

.. code-block:: python

    mutations = Mutations(
        no_mutation=0.4,
        architecture=0.2,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0.0,
        rl_hp=0.2,
        mutation_sd=0.1,
        rand_seed=42,
        device=device,
    )

Training with ``train_on_policy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the pieces in hand, :func:`~agilerl.training.train_on_policy.train_on_policy` orchestrates the
whole evolutionary run: it collects rollouts, trains each agent, evaluates the population and runs
the selection-plus-mutation step every ``evo_steps``. Passing ``selection_strategy=mf_selection`` is
the single switch that makes it MF-PBT (swap in a
:class:`~agilerl.hpo.tournament.TournamentSelection` and the exact same call runs tournament HPO
instead).

.. code-block:: python

    save_path = "MFPBT_PPO_trained_agent.pt"

    trained_pop, pop_fitnesses = train_on_policy(
        env=env,
        env_name="LunarLander-v3",
        algo="PPO",
        pop=pop,
        init_hp=init_hp,
        max_steps=200_000,
        evo_steps=10_240,
        eval_steps=None,
        eval_loop=1,
        selection_strategy=mf_selection,  # MF-PBT; the one line that differs from tournament HPO
        mutation=mutations,
        wb=False,             # set True to log to Weights & Biases
        save_elite=True,      # save the best agent at the end of training
        elite_path=save_path,
    )

Under the hood, each evolution cycle calls ``mf_selection.select(...)``, which returns the evolved
population and the indices of the winner-clones that replaced losers; only those clones are then
mutated (winners, survivors and migrants pass through untouched). The per-frequency scheduling is
internal to ``select``,it keeps a per-subpopulation counter and only evolves a subpopulation once
its counter reaches that subpopulation's ``evolution_frequency_ratio``.

.. note::

   For even finer control you can replace ``train_on_policy`` with your own training loop and call
   :func:`~agilerl.utils.utils.run_selection_and_mutation` once per evolution cycle; it performs the
   same selection-plus-mutation step.

Loading an Agent for Inference
------------------------------

Whichever path you took, the elite agent is a standard PPO checkpoint, loaded and run like any other.

.. code-block:: python

    ppo = PPO.load(save_path, device=device)

    test_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    rewards = []
    frames = []
    testing_eps = 7
    max_testing_steps = 1000
    with torch.no_grad():
        for ep in range(testing_eps):
            obs = test_env.reset()[0]  # Reset environment at start of episode
            score = 0
            for step in range(max_testing_steps):
                action, *_ = ppo.get_action(obs)
                action = action.squeeze()

                frames.append(test_env.render())  # Capture the frame
                obs, reward, terminated, truncated, _ = test_env.step(action)
                score += reward

                if terminated or truncated:
                    break

            rewards.append(score)
            print("-" * 15, f"Episode: {ep}", "-" * 15)
            print("Episodic Reward: ", rewards[-1])

        test_env.close()

    # Save the test episodes as a gif
    gif_path = "./videos/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimwrite(os.path.join(gif_path, "mfpbt_ppo_lunar_lander.gif"), frames, loop=0)
    print("Mean fitness:", np.mean(rewards))
