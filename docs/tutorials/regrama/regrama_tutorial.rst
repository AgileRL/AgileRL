.. _regrama_tutorial:

Lunar Lander with PPO & ReGraMa
===============================

In this tutorial we train a population of PPO agents on the Gymnasium ``LunarLander-v3``
environment with **ReGraMa** dormant-neuron resets enabled: the stage of AgileRL's
:ref:`network-parameter mutation <regrama>` that re-initialises the units which have stopped
learning, from *"Measure gradients, not activations!"*
(`Liu et al. <https://arxiv.org/abs/2505.24061>`_).

ReGraMa is not a replacement for anything, it is an extra stage *inside* the parameter mutation, so
everything you already know about AgileRL's HPO loop carries over unchanged. You keep your selection
strategy, your mutation probabilities and your training call, and turn the resets on with a single
flag. We use PPO here because its on-policy loop is the simplest to read, but the operator works
across every non-LLM algorithm family: DQN, DDPG, TD3, IPPO, MADDPG, MATD3, the bandits and offline
algorithms.

We show **two** ways to run it:

#. **From a YAML manifest**: let :class:`~agilerl.training.trainer.LocalTrainer` build the
   population, the selection strategy, the mutations and the training loop for you.
#. **In Python**: construct the environment, population, selection strategy and mutations yourself,
   then hand them to :func:`~agilerl.training.train_on_policy.train_on_policy`, which runs the
   training loop.

ReGraMa Overview
----------------

Deep RL networks steadily lose **plasticity**. As training goes on, more and more units stop
receiving meaningful gradient: they no longer contribute to the output, and because no gradient
reaches them they cannot recover on their own. The network keeps its nominal size while its
*effective* capacity shrinks, and it becomes progressively worse at fitting anything new. The stock
Gaussian parameter mutation cannot help here, because it perturbs randomly chosen weights with no
idea which units have gone quiet.

ReGraMa finds them. Each neuron is scored by the mean absolute gradient of the training loss with
respect to its **pre-activation**, divided by its own layer's mean. A neuron is *dormant* when that
normalised score falls at or below ``dormant_threshold``. Because the score is normalised within its
layer, the threshold is a fraction of the layer average rather than an absolute gradient magnitude,
so one setting works across layers and architectures of very different scales.

Measuring the *gradient* rather than the activation is the whole point. The two are related by
``grad_z L = grad_h L * act'(z)``, and ``act'`` is the only activation-dependent term — drop it and a
permanently-off ReLU unit, or a saturated Tanh unit, looks exactly like a healthy one. The gradient
form asks the question that actually matters: *can this unit still be updated?*

The scores are read from the **real training backward pass** through a hook on each activation, so
there is no extra forward or backward pass and no observation batch to collect. Only the final
minibatch of each cycle is kept, because the reset acts on the network as it stands at the end of
that cycle.

Each dormant neuron then gets fresh Xavier-uniform incoming weights, a zero bias, a small freshly
drawn set of outgoing weights, and a neutral normalisation entry. The outgoing weights are
deliberately small but **non-zero**: zeroing them would leave the revived neuron with exactly zero
gradient, so it would be re-flagged dormant forever and its incoming weights would never move.
Output layers of head networks are never reset — those units carry fixed meanings such as action
logits or a state value.

See :ref:`regrama` for the full description and for guidance on choosing a threshold.

Dependencies
------------

.. code-block:: python

    import torch

    from agilerl import LocalTrainer
    from agilerl.algorithms import PPO
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.training.train_on_policy import train_on_policy
    from agilerl.utils.utils import make_vect_envs

Option 1: From a manifest
-------------------------

The shipped ``configs/training/ppo/ppo_regrama.yaml`` is the stock PPO manifest with three fields
changed on its ``mutation`` block:

.. code-block:: yaml

    mutation:
        probabilities:
            no_mut: 0.4
            arch_mut: 0.2
            new_layer: 0.2
            params_mut: 0.2
            act_mut: 0.0
            rl_hp_mut: 0.2
        mutation_sd: 0.1
        rand_seed: 42
        super_param_mut: false
        regrama_param_mut: true
        dormant_threshold: 0.01

``regrama_param_mut`` turns the resets on and defaults to ``false``, so every existing manifest keeps
its current behaviour until you opt in. ``dormant_threshold`` defaults to ``0.01`` and must be
``>= 0``. ``super_param_mut`` defaults to ``true``; it is switched off in this tutorial.

Then run it:

.. code-block:: python

    from agilerl import LocalTrainer

    population, fitnesses = LocalTrainer.from_manifest(
        "configs/training/ppo/ppo_regrama.yaml"
    ).train()

One ``*_regrama.yaml`` ships for each non-LLM trainer family — ``ppo/ppo_regrama.yaml``,
``dqn/dqn_regrama.yaml``, ``cqn_regrama.yaml``, ``bandit/neural_ucb_regrama.yaml``,
``multi_agent/ippo_regrama.yaml`` and ``multi_agent/maddpg_regrama.yaml``.

Option 2: In Python
-------------------

Build the population and selection strategy exactly as you would for any AgileRL run. Selection is
untouched as ReGraMa lives on the mutation object:

.. code-block:: python

    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=population_size,
    )

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
        regrama_param_mut=True,   # reset dormant neurons before the Gaussian pass
        super_param_mut=False,    # drop the amplified band
        dormant_threshold=0.01,   # normalised score at or below which a neuron is dormant
    )

and hand them to the trainer as usual:

.. code-block:: python

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
        selection_strategy=tournament,
        mutation=mutations,
        wb=False,
        save_elite=True,
        elite_path="ReGraMa_PPO_trained_agent.pt",
    )

There is nothing else to wire up. Every AgileRL trainer enables the gradient capture as soon as it
sees a ReGraMa-configured ``Mutations`` object, and enables nothing at all when ReGraMa is off, so
the feature costs nothing when unused.

The complete runnable script is at ``tutorials/regrama/regrama.py``.
