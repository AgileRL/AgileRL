import copy
import gc
import os
from typing import List
from unittest import mock

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin
from gymnasium import spaces
from peft import LoraConfig

from agilerl.algorithms.core import EvolvableAlgorithm
from agilerl.algorithms.grpo import GRPO
from agilerl.hpo.mutation import MutationError, Mutations
from agilerl.modules import EvolvableBERT, ModuleDict
from agilerl.utils.utils import create_population
from agilerl.wrappers.agent import AsyncAgentsWrapper, RSNorm
from tests.helper_functions import assert_state_dicts_equal
from tests.test_algorithms.test_grpo import create_module

# Shared HP dict that can be used by any algorithm
SHARED_INIT_HP = {
    "POPULATION_SIZE": 2,
    "DOUBLE": True,
    "BATCH_SIZE": 32,
    "CUDAGRAPHS": False,
    "LR": 1e-3,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "GAMMA": 0.99,
    "LEARN_STEP": 1,
    "TAU": 1e-3,
    "BETA": 0.4,
    "PRIOR_EPS": 0.000001,
    "NUM_ATOMS": 51,
    "V_MIN": 0,
    "V_MAX": 200,
    "N_STEP": 3,
    "POLICY_FREQ": 10,
    "GAE_LAMBDA": 0.95,
    "ACTION_STD_INIT": 0.6,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "TARGET_KL": None,
    "UPDATE_EPOCHS": 4,
    "AGENT_IDS": ["agent_0", "agent_1", "other_agent_0"],
    "LAMBDA": 1.0,
    "REG": 0.000625,
    "CHANNELS_LAST": False,
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
}

SHARED_INIT_HP_MA = SHARED_INIT_HP.copy()


def create_bert_network(device):
    return EvolvableBERT([12], [12], device=device)


def create_bert_networks_multi_agent(device):
    return ModuleDict(
        {
            "agent_0": create_bert_network(device),
            "agent_1": create_bert_network(device),
            "other_agent_0": create_bert_network(device),
        }
    )


@pytest.fixture(scope="function")
def bert_network(device):
    return create_bert_network(device)


@pytest.fixture(scope="function")
def bert_networks_multi_agent(device):
    return create_bert_networks_multi_agent(device)


@pytest.fixture(scope="function")
def bert_matd3_critic_networks(device):
    return [
        create_bert_networks_multi_agent(device),
        create_bert_networks_multi_agent(device),
    ]


@pytest.fixture(scope="function")
def init_pop(
    algo,
    observation_space,
    action_space,
    net_config,
    INIT_HP,
    population_size,
    device,
    accelerator,
    hp_config,
    torch_compiler,
    request,
    actor_network=None,
    critic_network=None,
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

    if hp_config is not None:
        hp_config = request.getfixturevalue(hp_config)

    if actor_network is not None:
        actor_network = request.getfixturevalue(actor_network)
    if critic_network is not None:
        critic_network = request.getfixturevalue(critic_network)

    pop = create_population(
        algo=algo,
        observation_space=observation_space,
        action_space=action_space,
        hp_config=hp_config,
        net_config=request.getfixturevalue(net_config),
        INIT_HP=INIT_HP,
        population_size=population_size,
        device=device,
        accelerator=accelerator,
        actor_network=actor_network,
        critic_network=critic_network,
        torch_compiler=torch_compiler,
    )
    yield pop
    del pop
    gc.collect()
    torch.cuda.empty_cache()


# The constructor initializes all the attributes of the Mutations class correctly.
def test_constructor_initializes_attributes():
    no_mutation = 0.1
    architecture = 0.2
    new_layer_prob = 0.3
    parameters = 0.4
    activation = 0.5
    rl_hp = 0.6
    mutation_sd = 0.7
    activation_selection = ["ReLU", "Sigmoid"]
    mutate_elite = True
    rand_seed = 12345
    device = "cpu"
    accelerator = None

    mutations = Mutations(
        no_mutation,
        architecture,
        new_layer_prob,
        parameters,
        activation,
        rl_hp,
        mutation_sd,
        activation_selection,
        mutate_elite,
        rand_seed,
        device,
        accelerator,
    )

    assert mutations.rng is not None
    assert mutations.no_mut == no_mutation
    assert mutations.architecture_mut == architecture
    assert mutations.new_layer_prob == new_layer_prob
    assert mutations.parameters_mut == parameters
    assert mutations.activation_mut == activation
    assert mutations.rl_hp_mut == rl_hp
    assert mutations.mutation_sd == mutation_sd
    assert mutations.activation_selection == activation_selection
    assert mutations.mutate_elite == mutate_elite
    assert mutations.device == device
    assert mutations.accelerator == accelerator

    del mutations


# Checks no mutations if all probabilities set to zero
@pytest.mark.parametrize("algo", ["DQN"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [("vector_space", "encoder_mlp_config")],
)
@pytest.mark.parametrize("action_space", ["discrete_space"])
@pytest.mark.parametrize("accelerator", [None])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", ["default_hp_config"])
def test_mutation_no_options(init_pop, device):
    pre_training_mut = True
    population = init_pop
    mutations = Mutations(0, 0, 0, 0, 0, 0, 0.1, device=device)

    new_population = [agent.clone() for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert_state_dicts_equal(old.actor.state_dict(), individual.actor.state_dict())

    del mutations, mutated_population, new_population


#### Single-agent algorithm mutations ####
# The mutation method applies random mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, hp_config, action_space",
    [
        ("DQN", "default_hp_config", "discrete_space"),
        ("Rainbow DQN", "default_hp_config", "discrete_space"),
        ("DDPG", "ac_hp_config", "vector_space"),
        ("TD3", "ac_hp_config", "vector_space"),
        ("PPO", "default_hp_config", "discrete_space"),
        ("CQN", "default_hp_config", "discrete_space"),
        ("NeuralUCB", "default_hp_config", "discrete_space"),
        ("NeuralTS", "default_hp_config", "discrete_space"),
    ],
)
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize(
    "observation_space, net_config", [("vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_random_mutations(algo, init_pop, device, accelerator):
    population = init_pop
    pre_training_mut = True

    population = init_pop

    mutations = Mutations(
        0,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        mutate_elite=False,
        device=device,
        accelerator=accelerator,
    )

    # Unwrap models if using accelerator
    if accelerator is not None:
        for agent in population:
            agent.unwrap_models()

    mutated_population = mutations.mutation(population, pre_training_mut)

    assert len(mutated_population) == len(population)
    assert mutated_population[0].mut == "None"  # Satisfies mutate_elite=False condition
    for individual in mutated_population:
        policy = getattr(individual, individual.registry.policy())
        assert individual.mut in [
            "None",
            "batch_size",
            "lr",
            "lr_actor",
            "lr_critic",
            "learn_step",
            "act",
            "param",
            policy.last_mutation_attr,
        ]

    del mutations, mutated_population


# The mutation method applies no mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space",
    [
        ("DQN", "discrete_space"),
        ("Rainbow DQN", "discrete_space"),
        ("DDPG", "vector_space"),
        ("TD3", "vector_space"),
        ("PPO", "discrete_space"),
        ("CQN", "discrete_space"),
        ("NeuralUCB", "discrete_space"),
        ("NeuralTS", "discrete_space"),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config", [("vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_no_mutations(init_pop, device, accelerator):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        1,
        0,
        0,
        0,
        0,
        0,
        0.1,
        device=device,
        accelerator=accelerator,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None"]
        assert old.index == individual.index
        assert old.actor != individual.actor
        assert_state_dicts_equal(old.actor.state_dict(), individual.actor.state_dict())

    del mutations, mutated_population, new_population


# The mutation method applies no mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space",
    [
        ("DQN", "discrete_space"),
        ("Rainbow DQN", "discrete_space"),
        ("DDPG", "vector_space"),
        ("TD3", "vector_space"),
        ("PPO", "discrete_space"),
        ("CQN", "discrete_space"),
        ("NeuralUCB", "discrete_space"),
        ("NeuralTS", "discrete_space"),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config", [("vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_no_mutations_pre_training_mut(init_pop, device, accelerator):
    pre_training_mut = True
    population = init_pop

    # Set all mutation probabilities to 0
    mutations = Mutations(
        1,
        0,
        0,
        0,
        0,
        1,
        0.1,
        device=device,
        accelerator=accelerator,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in [
            "None",
            "batch_size",
            "lr",
            "lr_actor",
            "lr_critic",
            "learn_step",
        ]
        assert old.index == individual.index
        assert old.actor != individual.actor
        assert_state_dicts_equal(old.actor.state_dict(), individual.actor.state_dict())

    del mutations, mutated_population, new_population


# The mutation method applies RL hyperparameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, hp_config, action_space",
    [
        ("DQN", "default_hp_config", "discrete_space"),
        ("Rainbow DQN", "default_hp_config", "discrete_space"),
        ("DDPG", "ac_hp_config", "vector_space"),
        ("TD3", "ac_hp_config", "vector_space"),
        ("PPO", "default_hp_config", "discrete_space"),
        ("CQN", "default_hp_config", "discrete_space"),
        ("NeuralUCB", "default_hp_config", "discrete_space"),
        ("NeuralTS", "default_hp_config", "discrete_space"),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config", [("vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_rl_hp_mutations(
    init_pop, device, accelerator, hp_config, request
):
    pre_training_mut = False
    population = init_pop
    mutations = Mutations(
        0,
        0,
        0,
        0,
        0,
        1,
        0.1,
        device=device,
        accelerator=accelerator,
    )
    hp_config = request.getfixturevalue(hp_config)

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        available_mutations = hp_config.names()
        assert individual.mut in available_mutations

        new_value = getattr(individual, individual.mut)
        min_value = hp_config[individual.mut].min
        max_value = hp_config[individual.mut].max
        assert min_value <= new_value <= max_value
        assert old.index == individual.index

    del mutations, mutated_population, new_population


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space",
    [
        ("DQN", "discrete_space"),
        ("Rainbow DQN", "discrete_space"),
        ("DDPG", "vector_space"),
        ("TD3", "vector_space"),
        ("PPO", "discrete_space"),
        ("CQN", "discrete_space"),
        ("NeuralUCB", "discrete_space"),
        ("NeuralTS", "discrete_space"),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        ("vector_space", "encoder_mlp_config"),
        ("image_space", "encoder_cnn_config"),
        ("dict_space", "encoder_multi_input_config"),
    ],
)
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_activation_mutations(
    init_pop, observation_space, device, accelerator
):
    pre_training_mut = False
    population = init_pop

    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        activation_selection = ["ReLU", "ELU", "GELU"]
    else:
        activation_selection = ["Tanh", "ReLU", "ELU", "GELU"]

    mutations = Mutations(
        0,
        0,
        0,
        0,
        1,
        0,
        0.1,
        activation_selection=activation_selection,
        device=device,
        accelerator=accelerator,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None", "act"]
        if individual.mut == "act":
            assert old.actor.activation != individual.actor.activation
            assert individual.actor.activation in activation_selection
        assert old.index == individual.index

    del mutations, mutated_population, new_population


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        ("vector_space", "encoder_mlp_config"),
        ("image_space", "encoder_cnn_config"),
        ("dict_space", "encoder_multi_input_config"),
        ("discrete_space", "encoder_mlp_config"),
    ],
)
@pytest.mark.parametrize("algo, action_space", [("DDPG", "vector_space")])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_activation_mutations_no_skip(init_pop, device, accelerator):
    pre_training_mut = False
    population = init_pop
    mutations = Mutations(
        0,
        0,
        0,
        0,
        1,
        0,
        0.1,
        device=device,
        accelerator=accelerator,
    )

    for individual in population:
        individual.algo = None
        individual.lr = 1e-3
    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None", "act"]
        if individual.mut == "act":
            assert old.actor.activation != individual.actor.activation
            assert individual.actor.activation in ["ReLU", "ELU", "GELU"]
        assert old.index == individual.index

    del mutations, mutated_population, new_population


# The mutation method applies parameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space, wrapper_cls",
    [
        ("DQN", "discrete_space", None),
        ("Rainbow DQN", "discrete_space", None),
        ("DDPG", "vector_space", None),
        ("DDPG", "vector_space", RSNorm),
        ("TD3", "vector_space", None),
        ("PPO", "discrete_space", None),
        ("CQN", "discrete_space", None),
        ("NeuralUCB", "discrete_space", None),
        ("NeuralTS", "discrete_space", None),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config", [("vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_parameter_mutations(
    algo, device, accelerator, init_pop, wrapper_cls
):
    pre_training_mut = False

    population = init_pop

    if wrapper_cls is not None:
        population = [wrapper_cls(agent) for agent in population]

    mutations = Mutations(
        0,
        0,
        0,
        1,
        0,
        0,
        0.5,
        device=device,
        accelerator=accelerator,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "param"
        # Due to randomness, sometimes parameters are not different
        # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
        assert old.index == individual.index

        # Compare state dictionaries of the actor (or network)
        policy_name = old.registry.policy()
        old_policy = getattr(old, policy_name)
        new_policy = getattr(individual, policy_name)
        old_sd = old_policy.state_dict()
        new_sd = new_policy.state_dict()
        mutation_found = False
        for key in old_sd.keys():
            if "norm" in key:  # Skip normalization layers
                continue
            diff_norm = (old_sd[key] - new_sd[key]).norm().item()
            if diff_norm > 1e-6:
                mutation_found = True
                break

        assert mutation_found, f"Mutation not applied for agent index {old.index}"

    del mutations, mutated_population, new_population


# The mutation method applies architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space, wrapper_cls",
    [
        ("DQN", "discrete_space", None),
        ("Rainbow DQN", "discrete_space", None),
        ("DDPG", "vector_space", None),
        ("DDPG", "vector_space", RSNorm),
        ("TD3", "vector_space", None),
        ("PPO", "discrete_space", None),
        ("CQN", "discrete_space", None),
        ("NeuralUCB", "discrete_space", None),
        ("NeuralTS", "discrete_space", None),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        ("vector_space", "encoder_mlp_config"),
        ("image_space", "encoder_cnn_config"),
        ("dict_space", "encoder_multi_input_config"),
    ],
)
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_architecture_mutations(
    init_pop, device, accelerator, wrapper_cls
):
    population: List[EvolvableAlgorithm] = init_pop
    if wrapper_cls is not None:
        population = [wrapper_cls(agent) for agent in population]

    mutations = Mutations(
        0,
        1,
        0.5,
        0,
        0,
        0,
        0.5,
        device=device,
        accelerator=accelerator,
    )

    mut_methods = population[0].actor.mutation_methods

    # Change EvolvableModule random number generator to test mutation methods
    class EvoDummyRNG:
        rng = np.random.default_rng(seed=42)

        def choice(self, a, size=None, replace=True, p=None):
            return 1

        def integers(self, low=0, high=None):
            return self.rng.integers(low, high)

    for individual in population:
        for name, network in individual.evolvable_attributes(
            networks_only=True
        ).items():
            network.rng = EvoDummyRNG()

    applied_mutations = set()
    for mut_method in mut_methods:

        class DummyRNG:
            def choice(self, a, size=None, replace=True, p=None):
                return [mut_method]

        mutations.rng = DummyRNG()

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = [
            mutations.architecture_mutate(agent) for agent in new_population
        ]
        for individual in mutated_population:
            individual.mutation_hook()

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population):
            policy_name = old.registry.policy()
            policy = getattr(individual, policy_name)
            # old_policy = getattr(old, policy_name)
            assert individual.mut == (policy.last_mutation_attr or "None")

            if policy.last_mutation_attr is not None:
                applied_mutations.add(policy.last_mutation_attr)
                # assert str(old_policy.state_dict()) != str(policy.state_dict())
                for group in old.registry.groups:
                    if group.eval_network != policy_name:
                        eval_module = getattr(individual, group.eval_network)
                        # old_eval_module = getattr(old, group.eval_network)
                        assert eval_module.last_mutation_attr is not None
                        assert (
                            eval_module.last_mutation_attr == policy.last_mutation_attr
                        )
                        # assert str(old_eval_module.state_dict()) != str(eval_module.state_dict())

            assert old.index == individual.index

        # assert_equal_state_dict(population, mutated_population)

    assert all(mut in mut_methods for mut in applied_mutations), set(mut_methods) - set(
        applied_mutations
    )

    del mutations, mutated_population, new_population


# The mutation method applies BERT architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, actor_network, critic_network", [("DDPG", "bert_network", "bert_network")]
)
@pytest.mark.parametrize(
    "observation_space, net_config", [("vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("action_space", ["vector_space"])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=True)])
@pytest.mark.parametrize(
    "mut_method",
    [
        [
            "add_encoder_layer",
            "remove_encoder_layer",
            "add_decoder_layer",
            "remove_decoder_layer",
        ],
        ["add_node", "remove_node"],
    ],
)
def test_mutation_applies_bert_architecture_mutations_single_agent(
    algo,
    observation_space,
    action_space,
    device,
    accelerator,
    mut_method,
    actor_network,
    critic_network,
    init_pop,
    request,
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

    # Pass the network parameters to init_pop through the test
    actual_actor_network = (
        request.getfixturevalue(actor_network) if actor_network else None
    )
    actual_critic_network = (
        request.getfixturevalue(critic_network) if critic_network else None
    )

    # Create a custom population with the BERT networks
    from agilerl.utils.utils import create_population

    population = create_population(
        algo=algo,
        observation_space=observation_space,
        action_space=action_space,
        hp_config=None,
        net_config=request.getfixturevalue("encoder_mlp_config"),
        INIT_HP=SHARED_INIT_HP,
        population_size=1,
        device=device,
        accelerator=accelerator,
        actor_network=actual_actor_network,
        critic_network=actual_critic_network,
    )

    mutations = Mutations(
        0,
        1,
        0.5,
        0,
        0,
        0,
        0.5,
        device=device,
        accelerator=accelerator,
    )

    class DummyRNG:
        def choice(self, a, size=None, replace=True, p=None):
            return [np.random.choice(mut_method)]

    mutations.rng = DummyRNG()

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = [
        mutations.architecture_mutate(agent) for agent in new_population
    ]

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        policy = getattr(individual, individual.registry.policy())
        assert individual.mut == policy.last_mutation_attr
        # Due to randomness and constraints on size, sometimes architectures are not different
        # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
        assert old.index == individual.index

    # assert_equal_state_dict(population, mutated_population)

    del mutations, mutated_population, new_population


#### Multi-agent algorithm mutations ####
# The mutation method applies random mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3", "IPPO"])
@pytest.mark.parametrize(
    "observation_space, net_config", [("ma_vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("action_space", ["ma_discrete_space"])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_random_mutations_multi_agent(init_pop, device, accelerator):
    pre_training_mut = False
    population = init_pop

    # Random mutations
    mutations = Mutations(
        0,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        device=device,
        accelerator=accelerator,
    )

    if accelerator is not None:
        for agent in population:
            agent.unwrap_models()

    mutated_population = mutations.mutation(population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for individual in mutated_population:
        policy = getattr(individual, individual.registry.policy())
        if policy.last_mutation_attr is not None:
            sampled_mutation = ".".join(policy.last_mutation_attr.split(".")[1:])
        else:
            sampled_mutation = "None"

        assert individual.mut in [
            "None",
            "batch_size",
            "lr",
            "lr_actor",
            "lr_critic",
            "learn_step",
            "act",
            "param",
            sampled_mutation,
        ]

    del mutations, mutated_population


# The mutation method applies no mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3", "IPPO"])
@pytest.mark.parametrize(
    "observation_space, net_config", [("ma_vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("action_space", ["ma_discrete_space"])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
def test_mutation_applies_no_mutations_multi_agent(init_pop, device, accelerator):
    pre_training_mut = False
    population = init_pop

    mutations = Mutations(
        1,
        0,
        0,
        0,
        0,
        0,
        0.1,
        device=device,
        accelerator=accelerator,
    )

    if accelerator is not None:
        for agent in population:
            agent.unwrap_models()

    mutated_population = mutations.mutation(population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None"]
        assert old.index == individual.index
        assert old.actors == individual.actors

    del mutations
    del population
    del mutated_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies RL hyperparameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, hp_config",
    [
        ("MADDPG", "ac_hp_config"),
        ("MATD3", "ac_hp_config"),
        ("IPPO", "default_hp_config"),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config", [("ma_vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("action_space", ["ma_discrete_space"])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_rl_hp_mutations_multi_agent(
    init_pop, device, accelerator, hp_config, request
):
    pre_training_mut = False
    population = init_pop

    mutations = Mutations(
        0,
        0,
        0,
        0,
        0,
        1,
        0.1,
        device=device,
        accelerator=accelerator,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    hp_config = request.getfixturevalue(hp_config)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        available_mutations = hp_config.names()
        assert individual.mut in available_mutations

        new_value = getattr(individual, individual.mut)
        min_value = hp_config[individual.mut].min
        max_value = hp_config[individual.mut].max
        assert min_value <= new_value <= max_value
        assert old.index == individual.index

    del mutations, mutated_population, new_population


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3", "IPPO"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        ("ma_vector_space", "encoder_mlp_config"),
        ("ma_image_space", "encoder_cnn_config"),
        ("ma_dict_space_small", "encoder_multi_input_config"),
    ],
)
@pytest.mark.parametrize("action_space", ["ma_discrete_space"])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_activation_mutations_multi_agent(
    init_pop, device, accelerator
):
    pre_training_mut = False
    population = init_pop

    mutations = Mutations(
        0,
        0,
        0,
        0,
        1,
        0,
        0.1,
        device=device,
        accelerator=accelerator,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None", "act"]
        if individual.mut == "act":
            for old_actor, actor in zip(old.actors, individual.actors):
                assert old_actor.activation != actor.activation
                assert individual.actors[0].activation in [
                    "ReLU",
                    "ELU",
                    "GELU",
                ]
        assert old.index == individual.index

    del mutations, mutated_population, new_population


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3", "IPPO"])
@pytest.mark.parametrize(
    "observation_space, net_config", [("ma_vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("action_space", ["ma_discrete_space"])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_activation_mutations_multi_agent_no_skip(
    init_pop, device, accelerator
):
    pre_training_mut = False
    population = init_pop

    mutations = Mutations(
        0,
        0,
        0,
        0,
        1,
        0,
        0.1,
        device=device,
        accelerator=accelerator,
    )

    for individual in population:
        individual.algo = None
    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None", "act"]
        if individual.mut == "act":
            for old_actor, actor in zip(
                old.actors.values(), individual.actors.values()
            ):
                assert old_actor.activation != actor.activation
                assert actor.activation in [
                    "ReLU",
                    "ELU",
                    "GELU",
                ]
        assert old.index == individual.index

    del mutations, mutated_population, new_population


# The mutation method applies parameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, wrapper_cls",
    [
        ("MADDPG", None),
        ("MATD3", None),
        ("IPPO", None),
        ("IPPO", AsyncAgentsWrapper),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config", [("ma_vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("action_space", ["ma_discrete_space"])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("torch_compiler", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_parameter_mutations_multi_agent(
    init_pop, device, accelerator, wrapper_cls
):
    pre_training_mut = False
    population = init_pop

    if wrapper_cls is not None:
        population = [wrapper_cls(agent) for agent in population]

    mutations = Mutations(
        0,
        0,
        0,
        1,
        0,
        0,
        0.5,
        device=device,
        accelerator=accelerator,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "param"
        # Due to randomness, sometimes parameters are not different
        # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
        assert old.index == individual.index

        # Compare state dictionaries of the actor (or network)
        policy_name = old.registry.policy()
        old_policy = getattr(old, policy_name)
        new_policy = getattr(individual, policy_name)
        old_sd = old_policy.state_dict()
        new_sd = new_policy.state_dict()
        mutation_found = False
        for key in old_sd.keys():
            if "norm" in key:  # Skip normalization layers
                continue
            diff_norm = (old_sd[key] - new_sd[key]).norm().item()
            if diff_norm > 1e-6:
                mutation_found = True
                break

        assert mutation_found, f"Mutation not applied for agent index {old.index}"

    del mutations, mutated_population, new_population


# The mutation method applies architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, wrapper_cls",
    [
        ("MADDPG", None),
        ("MATD3", None),
        ("IPPO", None),
        ("IPPO", AsyncAgentsWrapper),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        ("ma_vector_space", "encoder_mlp_config"),
        ("ma_image_space", "encoder_cnn_config"),
        # ("ma_dict_space_small", "encoder_multi_input_config"), NOTE: Takes too long to run
    ],
)
@pytest.mark.parametrize("action_space", ["ma_discrete_space"])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("torch_compiler", [None, "default"])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_architecture_mutations_multi_agent(
    algo, init_pop, device, accelerator, wrapper_cls
):
    population: List[EvolvableAlgorithm] = init_pop
    mutations = Mutations(
        0,
        1,
        0.5,
        0,
        0,
        0,
        0.5,
        device=device,
        accelerator=accelerator,
    )

    # Change EvolvableModule random number generator to test mutation methods
    class EvoDummyRNG:
        rng = np.random.default_rng(seed=42)

        def choice(self, a, size=None, replace=True, p=None):
            return 1

        def integers(self, low=0, high=None):
            return self.rng.integers(low, high)

    if wrapper_cls is not None:
        population = [wrapper_cls(agent) for agent in population]

    for individual in population:
        for network in individual.evolvable_attributes(networks_only=True).values():
            network.rng = EvoDummyRNG()

    test_agent = "agent_0" if algo != "IPPO" else "agent"
    mut_methods = population[0].actors[test_agent].mutation_methods
    applied_mutations = set()
    for mut_method in mut_methods:

        class DummyRNG:
            def choice(self, a, size=None, replace=True, p=None):
                return [".".join([test_agent, mut_method])]

        mutations.rng = DummyRNG()

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = [
            mutations.architecture_mutate(agent) for agent in new_population
        ]

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population):
            policy_name = individual.registry.policy()
            policy = getattr(individual, policy_name)
            # old_policy = getattr(old, policy_name)
            if policy.last_mutation_attr is not None:
                sampled_mutation = ".".join(policy.last_mutation_attr.split(".")[1:])
                applied_mutations.add(sampled_mutation)
            else:
                sampled_mutation = None

            assert individual.mut == sampled_mutation or "None"

            if sampled_mutation is not None:
                for group in old.registry.groups:
                    if group.eval_network != policy_name:
                        eval_module = getattr(individual, group.eval_network)
                        # old_eval_module = getattr(old, group.eval_network)
                        for _, module in eval_module.items():
                            bottom_eval_mut = module.last_mutation_attr.split(".")[-1]
                            bottom_policy_mut = policy.last_mutation_attr.split(".")[-1]
                            assert module.last_mutation_attr is not None
                            assert bottom_eval_mut == bottom_policy_mut

            assert old.index == individual.index

        del new_population, mutated_population
        gc.collect()
        torch.cuda.empty_cache()

    del mutations

    assert all(mut in applied_mutations for mut in mut_methods), set(mut_methods) - set(
        applied_mutations
    )


# The mutation method applies BERT architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, actor_network, critic_network",
    [
        ("MADDPG", "bert_networks_multi_agent", "bert_networks_multi_agent"),
        ("MATD3", "bert_networks_multi_agent", "bert_matd3_critic_networks"),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config", [("ma_vector_space", "encoder_mlp_config")]
)
@pytest.mark.parametrize("action_space", ["ma_discrete_space"])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("torch_compiler", [None])
def test_mutation_applies_bert_architecture_mutations_multi_agent(
    algo,
    device,
    accelerator,
    init_pop,
    observation_space,
    action_space,
    request,
    actor_network,
    critic_network,
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

    # Pass the network parameters to init_pop through the test
    actual_actor_network = (
        request.getfixturevalue(actor_network) if actor_network else None
    )
    actual_critic_network = (
        request.getfixturevalue(critic_network) if critic_network else None
    )

    # Create a custom population with the BERT networks
    from agilerl.utils.utils import create_population

    population = create_population(
        algo=algo,
        observation_space=observation_space,
        action_space=action_space,
        hp_config=None,
        net_config=request.getfixturevalue("encoder_mlp_config"),
        INIT_HP=SHARED_INIT_HP_MA,
        population_size=1,
        device=device,
        accelerator=accelerator,
        actor_network=actual_actor_network,
        critic_network=actual_critic_network,
    )

    mutations = Mutations(
        0,
        1,
        0.5,
        0,
        0,
        0,
        0.5,
        device=device,
        accelerator=accelerator,
    )

    test_agent = "agent_0"
    mut_methods = population[0].actors[test_agent].mutation_methods
    for mut_method in mut_methods:

        class DummyRNG:
            def choice(self, a, size=None, replace=True, p=None):
                return [".".join([test_agent, mut_method])]

        mutations.rng = DummyRNG()

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = [
            mutations.architecture_mutate(agent) for agent in new_population
        ]

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population):
            policy_name = individual.registry.policy()
            policy = getattr(individual, policy_name)
            # old_policy = getattr(old, policy_name)
            if policy.last_mutation_attr is not None:
                sampled_mutation = ".".join(policy.last_mutation_attr.split(".")[1:])
            else:
                sampled_mutation = None

            assert individual.mut == sampled_mutation or "None"

            if sampled_mutation is not None:
                for group in old.registry.groups:
                    if group.eval_network != policy_name:
                        eval_module = getattr(individual, group.eval_network)
                        # old_eval_module = getattr(old, group.eval_network)
                        for _, module in eval_module.items():
                            bottom_eval_mut = module.last_mutation_attr.split(".")[-1]
                            bottom_policy_mut = policy.last_mutation_attr.split(".")[-1]
                            assert module.last_mutation_attr is not None
                            assert bottom_eval_mut == bottom_policy_mut

            assert old.index == individual.index

        del new_population, mutated_population
        gc.collect()
        torch.cuda.empty_cache()

    del mutations, population


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_mutation_applies_rl_hp_mutation_llm_algorithm(
    request, grpo_hp_config, vector_space, monkeypatch, use_accelerator
):
    pre_training_mut = False

    with mock.patch.dict(os.environ, clear=True):
        if use_accelerator:
            AcceleratorState._reset_state(True)
            env_vars = {
                "ACCELERATE_USE_DEEPSPEED": "true",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "10999",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
            }
            for key, value in env_vars.items():
                monkeypatch.setenv(key, value)

            deepspeed_config = {
                "gradient_accumulation_steps": 1,
                "zero_optimization": {
                    "stage": 2,
                },
            }
            accelerator = Accelerator(
                deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=deepspeed_config),
            )
        else:
            accelerator = None
        try:
            population = [
                GRPO(
                    observation_space=vector_space,
                    action_space=copy.deepcopy(vector_space),
                    actor_network=create_module(
                        input_size=10,
                        max_tokens=20,
                        vocab_size=1000,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    ),
                    index=0,
                    hp_config=grpo_hp_config,
                    pad_token_id=1000 - 1,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    lora_config=LoraConfig(
                        r=16,
                        lora_alpha=64,
                        target_modules=["linear_1"],
                        task_type="CAUSAL_LM",
                        lora_dropout=0.05,
                    ),
                    accelerator=accelerator,
                )
            ]  # some sort of population

            mutations = Mutations(
                0,
                0,
                0,
                0,
                0,
                1,
                0.1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                accelerator=accelerator,
            )

            new_population = [agent.clone(wrap=False) for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                available_mutations = grpo_hp_config.names()
                assert individual.mut in available_mutations

                new_value = getattr(individual, individual.mut)
                min_value = grpo_hp_config[individual.mut].min
                max_value = grpo_hp_config[individual.mut].max
                assert min_value <= new_value <= max_value
                assert old.index == individual.index

            for agent in mutated_population:
                for param_group in agent.optimizer.optimizer.param_groups:
                    assert param_group["lr"] == agent.lr
        except Exception as e:
            raise e
        finally:
            # Cleanup
            if use_accelerator:
                accelerator.free_memory()
                AcceleratorState._reset_state(True)
            del mutations
            del population
            del mutated_population
            del new_population
            torch.cuda.empty_cache()


@pytest.mark.parametrize("mutation_type", ["architecture", "parameters", "activation"])
def test_mutations_warns_on_llm_algorithm(
    request, grpo_hp_config, vector_space, mutation_type
):
    pre_training_mut = False

    population = [
        GRPO(
            observation_space=vector_space,
            action_space=copy.deepcopy(vector_space),
            actor_network=create_module(
                input_size=10,
                max_tokens=20,
                vocab_size=1000,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            index=0,
            hp_config=grpo_hp_config,
            pad_token_id=1000 - 1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            lora_config=LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
        )
    ]  # some sort of population

    mutations = Mutations(
        0,
        1 if mutation_type == "architecture" else 0,
        0.5 if mutation_type == "architecture" else 0,
        1 if mutation_type == "parameters" else 0,
        1 if mutation_type == "activation" else 0,
        0,
        0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        accelerator=None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]

    if mutation_type == "architecture":
        with pytest.raises(MutationError):
            mutations.mutation(new_population, pre_training_mut)

        # Since MutationError is expected, create a dummy mutated_population for the assertions
        mutated_population = new_population
        for individual in mutated_population:
            individual.mut = "None"
    else:
        with pytest.warns(UserWarning):
            mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert old.mut is None
        assert individual.mut == "None"

    del mutations
    del population
    del mutated_population
    del new_population
