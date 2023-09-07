Welcome to AgileRL's documentation!
===================================

.. image:: https://user-images.githubusercontent.com/47857277/222710068-e09a4e3c-368c-458a-9e01-b68674806887.png
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

**AgileRL** is a Deep Reinforcement Learning library focused on improving development by introducing RLOps - MLOps for reinforcement learning.

This library is initially focused on reducing the time taken for training models and hyperparameter optimisation (HPO) by pioneering evolutionary HPO techniques for reinforcement learning.
Evolutionary HPO has been shown to drastically reduce overall training times by automatically converging on optimal hyperparameters, without requiring numerous training runs.
We are constantly adding more algorithms, with a view to add hierarchical and multi-agent algorithms soon.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 3

   get_started/index
   online_training/index
   offline_training/index
   multi_agent_training/index
   distributed_training/index
   api/index
