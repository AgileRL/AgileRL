===============

.. raw:: html

   <style>
   h1:first-of-type {
       display: none;
   }
   </style>

.. raw:: html

   <div align="center">

.. figure:: https://user-images.githubusercontent.com/47857277/222710068-e09a4e3c-368c-458a-9e01-b68674806887.png
   :height: 120
   :align: center

.. raw:: html

   <p align="center"><b>Reinforcement learning streamlined.</b><br>Easier and faster reinforcement learning with RLOps.
   Visit our <a href="https://agilerl.com">website</a>. View <a href="https://docs.agilerl.com">documentation</a>.
   Join the <a href="https://discord.gg/eB8HyTA2ux">Discord Server</a> for questions, help and collaboration.
   Train super-fast for free on <a href="https://arena.agilerl.com">Arena</a>, the RLOps platform from AgileRL.</p>

   <br>

   <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
   <a href="https://docs.agilerl.com/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/agilerl/badge/?version=latest" alt="Documentation Status"></a>
   <a href="https://pypi.python.org/pypi/agilerl/"><img src="https://static.pepy.tech/badge/agilerl" alt="Downloads"></a>
   <a href="https://discord.gg/eB8HyTA2ux"><img src="https://dcbadge.vercel.app/api/server/eB8HyTA2ux?style=flat" alt="Discord"></a>
   <a href="https://arena.agilerl.com"><img src="../_static/arena-github-badge.svg" alt="Arena"></a>
   <br><br>
   <h3><i>🚀 <b>Train super-fast for free on <a href="https://arena.agilerl.com">Arena</a>, the RLOps platform from AgileRL 🚀</b></i></h3>

   </div>

**AgileRL** is a Deep Reinforcement Learning library focused on improving development by introducing RLOps - MLOps for reinforcement learning.

This library is initially focused on reducing the time taken for training models and hyperparameter optimization (HPO) by pioneering
`evolutionary HPO techniques <../evo_hyperparam_opt/index.html>`_ for reinforcement learning. Evolutionary HPO has been shown to drastically reduce
overall training times by automatically converging on optimal hyperparameters, without requiring numerous training runs.

We are constantly adding more algorithms and features. AgileRL already includes state-of-the-art evolvable `on-policy <../on_policy/index.html>`_, `off-policy <../off_policy/index.html>`_, `offline <../offline_training/index.html>`_, `multi-agent <../multi_agent_training/index.html>`_ and `contextual multi-armed bandit <../bandits/index.html>`_ reinforcement learning algorithms with `distributed training <../distributed_training/index.html>`_.

.. figure:: https://user-images.githubusercontent.com/47857277/236407686-21363eb3-ffcf-419f-b019-0be4ddf1ed4a.gif
   :width: 100%
   :align: center

   AgileRL offers 10x faster hyperparameter optimization than SOTA.

.. raw:: html

   <h2 id="benchmarks">Benchmarks</h2>

Reinforcement learning algorithms and libraries are usually benchmarked once the optimal hyperparameters for training are known, but it often takes hundreds or thousands of experiments to discover these. This is unrealistic and does not reflect the true, total time taken for training. What if we could remove the need to conduct all these prior experiments?

In the charts below, a single AgileRL run, which automatically tunes hyperparameters, is benchmarked against Optuna's multiple training runs traditionally required for hyperparameter optimization, demonstrating the real time savings possible. Global steps is the sum of every step taken by any agent in the environment, including across an entire population.

.. figure:: https://user-images.githubusercontent.com/47857277/227481592-27a9688f-7c0a-4655-ab32-90d659a71c69.png
   :width: 600px
   :align: center

   AgileRL offers an order of magnitude speed up in hyperparameter optimization vs popular reinforcement learning training frameworks combined with Optuna. Remove the need for multiple training runs and save yourself hours.

AgileRL also supports multi-agent reinforcement learning using the Petting Zoo-style (parallel API). The charts below highlight the performance of our MADDPG and MATD3 algorithms with evolutionary hyper-parameter optimisation (HPO), benchmarked against epymarl's MADDPG algorithm with grid-search HPO for the simple speaker listener and simple spread environments.

.. figure:: https://github-production-user-asset-6210df.s3.amazonaws.com/118982716/264712154-4965ea5f-b777-423c-989b-e4db86eda3bd.png
   :width: 700px
   :align: center


.. raw:: html

   <h3 id="citing-agilerl">Citing AgileRL</h3>

If you use AgileRL in your work, please cite the repository:

.. code-block:: bibtex

   @software{Ustaran-Anderegg_AgileRL,
   author = {Ustaran-Anderegg, Nicholas and Pratt, Michael and Sabal-Bermudez, Jaime},
   license = {Apache-2.0},
   title = {{AgileRL}},
   url = {https://github.com/AgileRL/AgileRL}
   }

.. raw:: html

   <h3 id="contents">Contents</h3>

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   get_started/index
   releases/index


.. toctree::
   :maxdepth: 2
   :caption: Training

   evo_hyperparam_opt/index
   off_policy/index
   on_policy/index
   pomdp/index
   offline_training/index
   multi_agent_training/index
   llm_finetuning/index
   bandits/index
   distributed_training/index
   evolvable_networks/index
   custom_algorithms/index
   debugging_rl/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/gymnasium/index
   tutorials/pettingzoo/index
   tutorials/skills/index
   tutorials/llm_finetuning/index
   tutorials/bandits/index
   tutorials/custom_networks/index

.. toctree::
   :maxdepth: 1
   :caption: API

   api/algorithms/index
   api/components/index
   api/hpo/index
   api/modules/index
   api/networks/index
   api/train
   api/rollouts/index
   api/utils/index
   api/vector/index
   api/wrappers/index

.. toctree::
   :caption: Development

   GitHub <https://github.com/AgileRL/AgileRL>
   Discord <https://discord.com/invite/eB8HyTA2ux>
   Contribute to AgileRL <https://github.com/AgileRL/AgileRL/blob/main/CONTRIBUTING.md>
