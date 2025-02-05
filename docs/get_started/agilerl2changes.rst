.. _agilerl2changes:

AgileRL 2.0 Changes
===================

Temporarily pasted:

Support for Complex Spaces:
---------------------------

This includes Dictionary and Tuple observation spaces, and is done through the EvolvableMultiInput module, which takes in such a
(single-level) space and assigns an EvolvableCNN to each underlying image subspace. Observations from vector / discrete spaces
are simply concatenated to the image encodings by default, but users can specify if they want these to be processed by an EvolvableMLP
before concatenating.

EvolvableModule Abstraction
---------------------------

A wrapper around nn.Module that allows us to keep track of the mutation methods in complex networks with nested modules. We use
the @mutation decorator to signal mutation methods and these are registered automatically as such. Such modules should implement
a recreate_network() method that is called automatically after any mutation method is used to modify the network's architecture.


EvolvableNetwork Abstraction
----------------------------

Towards a more general API for algorithm implementation, where complex observation spaces should be inherently supported, networks
inheriting from EvolvableNetwork automatically create an appropriate encoder from a given observation space. Custom networks simply
have to specify the head to the network that maps the observation encodings to a number of outputs. As part of this update we
implement common networks used in the already implemented algorithms e.g. QNetwork, ValueFunction, DeterministicActor, StochasticActor
(and more!)


EvolvableAlgorithm Abstraction
------------------------------

We create a class hierarchy for algorithms with a focus on evolutionary hyperparameter optimization. The EvolvableAlgorithm base
class implements common methods across any RL algorithm e.g. save_checkpoint(), load(), but also methods pertaining specifically
to mutations e.g. clone(). Under-the-hood, it initializes a MutationRegistry that users should use to register "network groups".
The registry also keeps track of the RL hyperparameters users wish to mutate during training and the optimizers. Users wishing to
create custom algorithms should now only need to worry about implementing get_action(), learn(), and (for now) test() methods.
