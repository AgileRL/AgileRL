.. _metrics_and_logging:

Metrics and Logging
===================

AgileRL provides a structured metrics and logging system built on three layers:

.. graphviz::
   :align: center

   digraph metrics_flow {
       rankdir=LR;
       bgcolor="transparent";
       node [shape=box, style="rounded,filled", fontsize=12, margin="0.3,0.15", fontcolor="white"];
       edge [fontname="Helvetica", fontsize=10, color="#cc5500", fontcolor="#cc5500"];

       agent [label=<
           <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="2">
           <TR><TD><FONT FACE="Courier" POINT-SIZE="13">Agent.log</FONT></TD></TR>
           <TR><TD><FONT FACE="Helvetica" POINT-SIZE="10">accumulate<BR/>registered metrics</FONT></TD></TR>
           </TABLE>
       >, fillcolor="#468082", color="#468082", penwidth=1.5];
       population [label=<
           <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="2">
           <TR><TD><FONT FACE="Courier" POINT-SIZE="13">Population</FONT></TD></TR>
           <TR><TD><FONT FACE="Helvetica" POINT-SIZE="10">gather population-<BR/>level metrics</FONT></TD></TR>
           </TABLE>
       >, fillcolor="#468082", color="#468082", penwidth=1.5];
       report [label=<
           <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="2">
           <TR><TD><FONT FACE="Courier" POINT-SIZE="13">MetricsReport</FONT></TD></TR>
           <TR><TD><FONT FACE="Helvetica" POINT-SIZE="10">format + dispatch</FONT></TD></TR>
           </TABLE>
       >, fillcolor="#468082", color="#468082", penwidth=1.5];

       agent -> population [label="  collect  "];
       population -> report [label="  build  "];

       subgraph cluster_loggers {
           labeljust="c";
           style="dashed,rounded";
           color="#888888";
           fontname="Helvetica";
           fontsize=11;
           fontcolor="#333333";

           stdout [label=<<FONT FACE="Courier" POINT-SIZE="11">StdOutLogger</FONT>>, fillcolor="#468082", color="#468082", penwidth=1.5, margin="0.15,0.1"];
           wandb [label=<<FONT FACE="Courier" POINT-SIZE="11">WandbLogger</FONT>>, fillcolor="#468082", color="#468082", penwidth=1.5, margin="0.15,0.1"];
           csv [label=<<FONT FACE="Courier" POINT-SIZE="11">CSVLogger</FONT>>, fillcolor="#468082", color="#468082", penwidth=1.5, margin="0.15,0.1"];
           tb [label=<<FONT FACE="Courier" POINT-SIZE="11">TensorboardLogger</FONT>>, fillcolor="#468082", color="#468082", penwidth=1.5, margin="0.15,0.1"];
       }

       report -> stdout;
       report -> wandb;
       report -> csv;
       report -> tb;
   }

Each algorithm instance owns a :class:`~agilerl.metrics.AgentMetrics` (or
:class:`~agilerl.metrics.MultiAgentMetrics`) object that logs steps,
scores, fitness, and any additional algorithm-specific scalars or histograms.
:class:`~agilerl.population.Population` collects these per-agent metrics into
a :class:`~agilerl.population.MetricsReport` every evolution step and dispatches
it to all configured logger backends simultaneously.

.. note::

   If you are using :ref:`LocalTrainer <local_trainer>` or the off-the-shelf training functions, all of this is handled
   automatically. The sections below are relevant when writing custom training
   loops or when you need fine-grained control over what gets logged.


Logger Backends
---------------

All loggers implement a simple ``write(report)`` / ``close()`` interface. You
can use any combination simultaneously:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Backend
     - Description
   * - :class:`~agilerl.logger.StdOutLogger`
     - Renders a Rich table to the console via ``tqdm.write()``.
   * - :class:`~agilerl.logger.WandbLogger`
     - Logs the flat metrics dict to Weights & Biases. Auto-initialises a run if needed.
   * - :class:`~agilerl.logger.CSVLogger`
     - Appends one row per evolution step to a CSV file for offline analysis.
   * - :class:`~agilerl.logger.TensorboardLogger`
     - Writes scalars and histogram distributions to TensorBoard event files.


Example
-------

Below is a minimal training loop showing how to configure metrics and logging
with a :class:`~agilerl.population.Population`:

.. code-block:: python

   import numpy as np
   import torch

   from agilerl.algorithms import DQN
   from agilerl.components import ReplayBuffer
   from agilerl.components.data import Transition
   from agilerl.hpo.mutation import Mutations
   from agilerl.hpo.tournament import TournamentSelection
   from agilerl.logger import StdOutLogger, WandbLogger, CSVLogger
   from agilerl.population import Population
   from agilerl.utils.utils import default_progress_bar, make_vect_envs

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Environment
   num_envs = 8
   env = make_vect_envs("LunarLander-v3", num_envs=num_envs)

   # Algorithm hyperparameters
   init_hp = {
       "batch_size": 128,
       "lr": 1e-3,
       "gamma": 0.99,
       "learn_step": 1,
       "tau": 1e-3,
   }

   # Initialize population
   population_size = 4
   pop = DQN.population(
       size=population_size,
       observation_space=env.single_observation_space,
       action_space=env.single_action_space,
       device=device,
       **init_hp,
   )

   # Replay buffer
   memory = ReplayBuffer(max_size=100_000, device=device)

   # Evo-HPO
   tournament = TournamentSelection(
    tournament_size=2,
    elitism=True,
    population_size=population_size,
   )
   mutation = Mutations(
       no_mutation=0.4,
       architecture=0.2,
       new_layer_prob=0.5,
       parameters=0.2,
       activation=0.1,
       rl_hp=0.1,
       device=device,
   )

   # Configure loggers and population
   max_steps = 200_000
   evo_steps = 5_000
   pbar = default_progress_bar(max_steps)

  # The following loggers will:
  # - Render a Rich table with training progress to the console via ``tqdm.write()``.
  # - Log the flat metrics dict to Weights & Biases. Auto-initialises a run if needed.
  # - Append one row per evolution step to a CSV file for offline analysis.
   loggers = [
       StdOutLogger(pbar=pbar),
       WandbLogger(project="AgileRL"),
       CSVLogger("training_metrics.csv"),
   ]

   population = Population(agents=pop, loggers=loggers)

   # Training loop
   epsilon = 1.0
   while population.all_below(max_steps):
       for agent in population.agents:
           agent.init_training_step()

           obs, info = env.reset()
           scores = np.zeros(num_envs)
           completed_episode_scores = []
           steps = 0

           for idx_step in range(evo_steps // num_envs):
               action = agent.get_action(obs, epsilon)
               epsilon = max(0.1, epsilon * 0.999)
               next_obs, reward, done, trunc, info = env.step(action)

               scores += np.array(reward)
               for idx, (d, t) in enumerate(zip(done, trunc)):
                   if d or t:
                       completed_episode_scores.append(scores[idx])
                       scores[idx] = 0

               # Store transition in replay buffer
               transition = Transition(
                   obs=obs, action=action, reward=reward,
                   next_obs=next_obs, done=done,
               ).to_tensordict()
               transition.batch_size = [num_envs]
               memory.add(transition)

               # Learn from buffer
               if len(memory) >= agent.batch_size:
                   experiences = memory.sample(agent.batch_size)
                   agent.learn(experiences)

               obs = next_obs
               steps += num_envs

           agent.add_scores(completed_episode_scores)
           agent.finalize_training_step(steps)
           pbar.update(evo_steps // population.size)

       # Evaluate
       for agent in population.agents:
           agent.test(env, max_steps=1000, loop=3)

       # Report metrics to all backends, then evolve
       population.increment_evo_step()
       population.report_metrics(clear=True)

       if population.should_stop(target=200.0):
           break

       elite, new_pop, _ = tournament.select(population.agents)
       population.update(mutation.mutation(new_pop))

   population.finish()
   pbar.close()
   env.close()

The :func:`~agilerl.utils.utils.init_loggers` utility can also construct all
backends from simple flags, matching what the built-in ``train_*`` functions
use internally:

.. code-block:: python

   from agilerl.utils.utils import init_loggers

   loggers = init_loggers(
       algo="DQN",
       env_name="LunarLander-v3",
       pbar=pbar,
       verbose=True,
       wb=True,
       tensorboard=True,
       csv=True,
   )


.. seealso::

   - :doc:`../api/metrics` -- API reference for ``AgentMetrics`` and ``MultiAgentMetrics``
   - :doc:`../api/logger` -- API reference for all logger backends
   - :doc:`../api/population` -- API reference for ``Population``, ``PopulationMetrics``, and ``MetricsReport``
   - :ref:`trainers` -- The ``LocalTrainer`` handles metrics and logging automatically
