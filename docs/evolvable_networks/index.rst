.. _evolvable_networks:

Evolvable Neural Networks in AgileRL
====================================

Other than the hyperparemeters pertaining to the specific algorithm you're using to optimize your agent, a large source of variance in
the performance of your agent is the choice network architecture. Tuning the architecture of your network is usually a very time-consuming and tedious task,
requiring multiple training runs that can take days or even weeks to execute. AgileRL allows you to automatically tune the architecture of your network in
a single training run through :ref:`evolutionary hyperparameter optimization <evo_hyperparam_opt>`.

In order to mutate the architecture of neural networks seamlessly, we define the :class:`~agilerl.modules.base.EvolvableModule` base class as a building block
for all networks used in AgileRL. This is nothing but a wrapper around :class:`~torch.nn.Module` that allows us to keep track of the methods that mutate a network
in networks with nested evolvable modules.

`EvolvableModule`
~~~~~~~~~~~~~~~~~
