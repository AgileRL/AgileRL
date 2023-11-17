Mutation
========

Mutation is periodically used to explore the hyperparameter space, allowing different hyperparameter combinations to be trialled during training. If certain hyperparameters
prove relatively beneficial to training, then that agent is more likely to be preserved in the next generation, and so those characteristics are more likely to remain in the
population.

The ``Mutations()`` class is used to mutate agents with pre-set probabilities. The available mutations currently implemented are:
    * No mutation
    * Network architecture mutation - adding layers or nodes. Trained weights are reused and new weights are initialized randomly.
    * Network parameters mutation - mutating weights with Gaussian noise.
    * Network activation layer mutation - change of activation layer.
    * RL algorithm mutation - mutation of learning hyperparameter, such as learning rate or batch size.

``Mutations.mutation()`` returns a mutated population.

Tournament selection and mutation should be applied sequentially to fully evolve a population between evaluation and learning cycles.

.. code-block:: python

  from agilerl.hpo.mutation import Mutations
  import torch

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  mutations = Mutations(algo='DQN',                           # Algorithm
                        no_mutation=0.4,                      # No mutation
                        architecture=0.2,                     # Architecture mutation
                        new_layer_prob=0.2,                   # New layer mutation
                        parameters=0.2,                       # Network parameters mutation
                        activation=0,                         # Activation layer mutation
                        rl_hp=0.2,                            # Learning HP mutation
                        rl_hp_selection=['lr', 'batch_size'], # Learning HPs to choose from
                        mutation_sd=0.1,                      # Mutation strength
                        arch=NET_CONFIG['arch'],              # Network architecture
                        rand_seed=1,                          # Random seed
                        device=device)


Parameters
----------

.. autoclass:: agilerl.hpo.mutation.Mutations
  :members:
  :inherited-members:
