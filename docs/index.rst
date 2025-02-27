.. figure:: https://user-images.githubusercontent.com/47857277/222710068-e09a4e3c-368c-458a-9e01-b68674806887.png
   :height: 120
   :align: center

   |License| |Docs status| |PyPI download total| |Discord|

.. |License| image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0

.. |Docs status| image:: https://readthedocs.org/projects/agilerl/badge/?version=latest
   :target: https://agilerl.readthedocs.io/en/latest/?badge=latest

.. |PyPI download total| image:: https://static.pepy.tech/badge/agilerl
   :target: https://pypi.python.org/pypi/agilerl/

.. |Discord| image:: https://dcbadge.vercel.app/api/server/eB8HyTA2ux?style=flat
   :target: https://discord.gg/eB8HyTA2ux

Streamlining reinforcement learning.
====================================

.. highlights::

   **✨ NEW: AgileRL 2.0 is here! Check out the latest powerful** :ref:`updates <agilerl2changes>` **✨**

   **🚀 Train super-fast for free on** `Arena <https://arena.agilerl.com>`_ **, the RLOps platform from AgileRL 🚀**

**AgileRL** is a Deep Reinforcement Learning library focused on improving development by introducing RLOps - MLOps for reinforcement learning.

This library is initially focused on reducing the time taken for training models and hyperparameter optimisation (HPO) by pioneering evolutionary HPO techniques for reinforcement learning.
Evolutionary HPO has been shown to drastically reduce overall training times by automatically converging on optimal hyperparameters, without requiring numerous training runs.
We are constantly adding more algorithms and features. AgileRL already includes state-of-the-art evolvable on-policy, off-policy, offline and multi-agent reinforcement learning algorithms with distributed training.

Join the AgileRL `Discord server <https://discord.com/invite/eB8HyTA2ux>`_ to ask questions, get help, and learn more about reinforcement learning.

.. figure:: https://user-images.githubusercontent.com/47857277/236407686-21363eb3-ffcf-419f-b019-0be4ddf1ed4a.gif
   :width: 900px
   :align: center

   AgileRL offers 10x faster hyperparameter optimization than SOTA.

   Global steps is the sum of every step taken by any agent in the environment, including across an entire population, during the entire hyperparameter optimization process.

Installation
------------

Install as a package with pip:

.. code-block:: bash

   pip install agilerl

Or install in development mode:

.. code-block:: bash

   git clone https://github.com/AgileRL/AgileRL.git && cd AgileRL
   pip install -e .


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   get_started/index
   get_started/agilerl2changes


.. toctree::
   :maxdepth: 2
   :caption: Training

   evo_hyperparam_opt/index
   off_policy/index
   on_policy/index
   offline_training/index
   multi_agent_training/index
   bandits/index
   distributed_training/index
   custom_architecture/index
   custom_algorithms/index
   debugging_rl/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/gymnasium/index
   tutorials/pettingzoo/index
   tutorials/skills/index
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
   api/utils/index
   api/vector/index
   api/wrappers/index

.. toctree::
   :caption: Development

   GitHub <https://github.com/AgileRL/AgileRL>
   Discord <https://discord.com/invite/eB8HyTA2ux>
   Contribute to AgileRL <https://github.com/AgileRL/AgileRL/blob/main/CONTRIBUTING.md>

Benchmarks
----------

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
