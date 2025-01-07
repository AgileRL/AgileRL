import copy

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from gymnasium import spaces

from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.hpo.mutation import Mutations
from agilerl.modules.bert import EvolvableBERT
from agilerl.utils.utils import create_population
from tests.helper_functions import (
    generate_random_box_space,
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces
)

# from pytest_mock import mocker

# Shared HP dict that can be used by any algorithm
SHARED_INIT_HP = {
    "POPULATION_SIZE": 4,
    "DOUBLE": True,
    "BATCH_SIZE": 128,
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
    "AGENT_IDS": ["agent1", "agent2"],
    "LAMBDA": 1.0,
    "REG": 0.000625,
    "CHANNELS_LAST": False,
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
}

SHARED_INIT_HP_MA = {
    "POPULATION_SIZE": 4,
    "DOUBLE": True,
    "BATCH_SIZE": 128,
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
    "AGENT_IDS": ["agent1", "agent2"],
    "LAMBDA": 1.0,
    "REG": 0.000625,
    "CHANNELS_LAST": False,
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
}


@pytest.fixture
def init_pop(
    algo,
    observation_space,
    action_space,
    net_config,
    INIT_HP,
    population_size,
    device,
    accelerator,
):
    return create_population(
        algo=algo,
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config,
        INIT_HP=INIT_HP,
        population_size=population_size,
        device=device,
        accelerator=accelerator,
    )


# The constructor initializes all the attributes of the Mutations class correctly.
def test_constructor_initializes_attributes():
    algo = {
        "actor": {
            "eval": "actor",
            "target": "actor_target",
            "optimizer": "optimizer",
        },
        "critics": [],
    }
    no_mutation = 0.1
    architecture = 0.2
    new_layer_prob = 0.3
    parameters = 0.4
    activation = 0.5
    rl_hp = 0.6
    rl_hp_selection = ["batch_size", "lr", "learn_step"]
    mutation_sd = 0.7
    activation_selection = ["ReLU", "Sigmoid"]
    min_lr = 0.0001
    max_lr = 0.01
    min_learn_step = 1
    max_learn_step = 120
    min_batch_size = 8
    max_batch_size = 1024
    agent_ids = None
    arch = "mlp"
    mutate_elite = True
    rand_seed = 12345
    device = "cpu"
    accelerator = None

    mutations = Mutations(
        algo,
        no_mutation,
        architecture,
        new_layer_prob,
        parameters,
        activation,
        rl_hp,
        rl_hp_selection,
        mutation_sd,
        activation_selection,
        min_lr,
        max_lr,
        min_learn_step,
        max_learn_step,
        min_batch_size,
        max_batch_size,
        agent_ids,
        arch,
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
    assert mutations.rl_hp_selection == rl_hp_selection
    assert mutations.mutation_sd == mutation_sd
    assert mutations.activation_selection == activation_selection
    assert mutations.mutate_elite == mutate_elite
    assert mutations.device == device
    assert mutations.accelerator == accelerator
    assert mutations.agent_ids == agent_ids
    assert mutations.min_batch_size == min_batch_size
    assert mutations.max_batch_size == max_batch_size
    assert mutations.min_lr == min_lr
    assert mutations.max_lr == max_lr
    assert mutations.min_learn_step == min_learn_step
    assert mutations.max_learn_step == max_learn_step

# Can regularize weight
def test_returns_regularize_weight():
    mutations = Mutations(
        "DQN", 0, 0, 0, 0, 0, 0, ["batch_size", "lr", "learn_step"], 0.1
    )

    weight = 10
    mag = 5
    result = mutations.regularize_weight(weight, mag)
    assert result == mag

    weight = -10
    mag = 5
    result = mutations.regularize_weight(weight, mag)
    assert result == -mag

    weight = 5
    mag = 5
    result = mutations.regularize_weight(weight, mag)
    assert result == weight


# Checks no mutations if all probabilities set to zero
@pytest.mark.parametrize(
    "algo, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        )
    ],
)
def test_mutation_no_options(
    algo,
    observation_space,
    action_space,
    net_config,
    INIT_HP,
    population_size,
    device,
    accelerator,
    init_pop,
):
    pre_training_mut = True

    population = init_pop

    mutations = Mutations(
        algo, 0, 0, 0, 0, 0, 0, ["batch_size", "lr", "learn_step"], 0.1, device=device
    )

    new_population = [agent.clone() for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert str(old.actor.state_dict()) == str(individual.actor.state_dict())


#### Single-agent algorithm mutations ####
# The mutation method applies random mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((4,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((4,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((4,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((4,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_random_mutations(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = True

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        ["batch_size", "lr", "learn_step"],
        0.1,
        mutate_elite=False,
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    for agent in population:
        if accelerator is not None:
            agent.unwrap_models()
    mutated_population = mutations.mutation(population, pre_training_mut)

    assert len(mutated_population) == len(population)
    assert mutated_population[0].mut == "None"  # Satisfies mutate_elite=False condition
    for individual in mutated_population:
        assert individual.mut in [
            "None",
            "bs",
            "lr",
            "lr_actor",
            "lr_critic",
            "ls",
            "act",
            "param",
            "arch",
        ]


# The mutation method applies no mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "ILQL",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "ILQL",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
    ],
)
def test_mutation_applies_no_mutations(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        1,
        0,
        0,
        0,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.1,
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None"]
        assert old.index == individual.index
        assert old.actor != individual.actor
        assert str(old.actor.state_dict()) == str(individual.actor.state_dict())


# The mutation method applies no mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "ILQL",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "ILQL",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(),
        ),
    ],
)
def test_mutation_applies_no_mutations_pre_training_mut(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = True

    population = init_pop

    mutations = Mutations(
        algo,
        1,
        0,
        0,
        0,
        0,
        1,
        ["batch_size", "lr", "learn_step"],
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
            "bs",
            "lr",
            "lr_actor",
            "lr_critic",
            "ls",
        ]
        assert old.index == individual.index
        assert old.actor != individual.actor
        assert str(old.actor.state_dict()) == str(individual.actor.state_dict())


# The mutation method applies RL hyperparameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "ILQL",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "ILQL",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_rl_hp_mutations(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    for rl_hp_mut in ["batch_size", "lr", "learn_step"]:
        mutations = Mutations(
            algo,
            0,
            0,
            0,
            0,
            0,
            1,
            [rl_hp_mut],
            0.1,
            device=device if not distributed else None,
            accelerator=accelerator if distributed else None,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population):
            assert individual.mut in [
                "None",
                "bs",
                "lr",
                "lr_actor",
                "lr_critic",
                "ls",
            ]
            if individual.mut == "bs":
                assert (
                    mutations.min_batch_size
                    <= individual.batch_size
                    <= mutations.max_batch_size
                )
            if individual.mut == "lr":
                assert mutations.min_lr <= individual.lr <= mutations.max_lr
            if individual.mut == "lr_actor":
                assert mutations.min_lr <= individual.lr_actor <= mutations.max_lr
            if individual.mut == "lr_critic":
                assert mutations.min_lr <= individual.lr_critic <= mutations.max_lr
            if individual.mut == "ls":
                assert (
                    mutations.min_learn_step
                    <= individual.learn_step
                    <= mutations.max_learn_step
                )
            assert old.index == individual.index


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "ILQL",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "ILQL",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_activation_mutations(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        0,
        1,
        0,
        ["batch_size", "lr", "learn_step"],
        0.1,
        activation_selection=["Tanh", "ReLU", "ELU", "GELU"],
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None", "act"]
        if individual.mut == "act":
            assert old.actor.activation != individual.actor.activation
            assert individual.actor.activation in ["Tanh", "ReLU", "ELU", "GELU"]
        assert old.index == individual.index


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_activation_mutations_no_skip(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        0,
        1,
        0,
        ["batch_size", "lr", "learn_step"],
        0.1,
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
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


# The mutation method applies CNN activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "ILQL",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "ILQL",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_cnn_activation_mutations(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        0,
        1,
        0,
        ["batch_size", "lr", "learn_step"],
        0.1,
        arch="cnn",
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None", "act"]
        if individual.mut == "act":
            assert old.actor.activation != individual.actor.activation
            assert individual.actor.activation in ["ReLU", "ELU", "GELU"]
        assert old.index == individual.index


# The mutation method applies parameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "ILQL",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "ILQL",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_parameter_mutations(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        1,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.5,
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "param"
        # Due to randomness, sometimes parameters are not different
        # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
        assert old.index == individual.index


# The mutation method applies CNN parameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "ILQL",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "ILQL",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_cnn_parameter_mutations(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        1,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.5,
        arch="cnn",
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "param"
        # Due to randomness, sometimes parameters are not different
        # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
        assert old.index == individual.index


# The mutation method applies architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_architecture_mutations(
    algo, distributed, device, accelerator, init_pop
):
    for _ in range(10):
        population = init_pop

        mutations = Mutations(
            algo,
            0,
            1,
            0.5,
            0,
            0,
            0,
            ["batch_size", "lr", "learn_step"],
            0.5,
            device=device if not distributed else None,
            accelerator=accelerator if distributed else None,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = [
            mutations.architecture_mutate(agent) for agent in new_population
        ]

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population):
            assert individual.mut == "arch"
            # Due to randomness and constraints on size, sometimes architectures are not different
            # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
            assert old.index == individual.index


# The mutation method applies CNN architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DQN",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "Rainbow DQN",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "TD3",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_random_box_space((2,), low=-1, high=1),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "PPO",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "CQN",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "ILQL",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "ILQL",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralUCB",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "NeuralTS",
            False,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            True,
            generate_random_box_space((3, 32, 32)),
            generate_discrete_space(2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_cnn_architecture_mutations(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = True

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        1,
        0.5,
        0,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.5,
        arch="cnn",
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "arch"
        # Due to randomness and constraints on size, sometimes architectures are not different
        # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
        assert old.index == individual.index


# The mutation method applies BERT architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator, mut_method",
    [
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
            [
                "add_encoder_layer",
                "remove_encoder_layer",
                "add_decoder_layer",
                "remove_decoder_layer",
            ],
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=True),
            [
                "add_encoder_layer",
                "remove_encoder_layer",
                "add_decoder_layer",
                "remove_decoder_layer",
            ],
        ),
        (
            "DDPG",
            False,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
            ["add_node", "remove_node"],
        ),
        (
            "DDPG",
            True,
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=True),
            ["add_node", "remove_node"],
        ),
    ],
)
def test_mutation_applies_bert_architecture_mutations_single_agent(
    algo, distributed, device, accelerator, init_pop, mut_method
):
    population = init_pop

    mutations = Mutations(
        algo,
        0,
        1,
        0.5,
        0,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.5,
        arch="bert",
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    class DummyRNG:
        def choice(self, a, size=None, replace=True, p=None):
            return [np.random.choice(mut_method)]

    mutations.rng = DummyRNG()

    for individual in population:


        individual.actor = EvolvableBERT([12], [12])
        individual.actor_target = copy.deepcopy(individual.actor)
        individual.critic = EvolvableBERT([12], [12])
        individual.critic_target = copy.deepcopy(individual.critic)

        individual.actor_optimizer = OptimizerWrapper(
            torch.optim.Adam,
            individual.actor,
            optimizer_kwargs={"lr": individual.lr_actor},
            network_names=individual.actor_optimizer.network_names
        )

        individual.critic_optimizer = OptimizerWrapper(
            torch.optim.Adam,
            individual.critic,
            optimizer_kwargs={"lr": individual.lr_critic},
            network_names=individual.critic_optimizer.network_names
        )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = [
        mutations.architecture_mutate(agent) for agent in new_population
    ]

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "arch"
        # Due to randomness and constraints on size, sometimes architectures are not different
        # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
        assert old.index == individual.index


#### Multi-agent algorithm mutations ####
# The mutation method applies random mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_random_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        ["batch_size", "lr", "learn_step"],
        0.1,
        agent_ids=SHARED_INIT_HP["AGENT_IDS"],
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    for agent in population:
        if accelerator is not None:
            agent.unwrap_models()
    mutated_population = mutations.mutation(population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for individual in mutated_population:
        assert individual.mut in [
            "None",
            "bs",
            "lr",
            "lr_actor",
            "lr_critic",
            "ls",
            "act",
            "param",
            "arch",
        ]


# The mutation method applies no mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_no_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        1,
        0,
        0,
        0,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.1,
        agent_ids=SHARED_INIT_HP["AGENT_IDS"],
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    for agent in population:
        if accelerator is not None:
            agent.unwrap_models()
    mutated_population = mutations.mutation(population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None"]
        assert old.index == individual.index
        assert old.actors == individual.actors


# The mutation method applies RL hyperparameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_rl_hp_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        0,
        0,
        1,
        ["batch_size", "lr", "learn_step"],
        0.1,
        agent_ids=SHARED_INIT_HP["AGENT_IDS"],
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in [
            "None",
            "bs",
            "lr",
            "lr_actor",
            "lr_critic",
            "ls",
        ]
        if individual.mut == "bs":
            assert (
                mutations.min_batch_size
                <= individual.batch_size
                <= mutations.max_batch_size
            )
        if individual.mut == "lr":
            assert mutations.min_lr <= individual.lr <= mutations.max_lr
        if individual.mut == "lr_actor":
            assert mutations.min_lr <= individual.lr_actor <= mutations.max_lr
        if individual.mut == "lr_critic":
            assert mutations.min_lr <= individual.lr_critic <= mutations.max_lr
        if individual.mut == "ls":
            assert (
                mutations.min_learn_step
                <= individual.learn_step
                <= mutations.max_learn_step
            )
        assert old.index == individual.index


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_activation_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        0,
        1,
        0,
        ["batch_size", "lr", "learn_step"],
        0.1,
        agent_ids=SHARED_INIT_HP["AGENT_IDS"],
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
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


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_activation_mutations_multi_agent_no_skip(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        0,
        1,
        0,
        ["batch_size", "lr", "learn_step"],
        0.1,
        agent_ids=SHARED_INIT_HP["AGENT_IDS"],
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    for individual in population:
        individual.algo = None
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


# The mutation method applies CNN activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_cnn_activation_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        0,
        1,
        0,
        ["batch_size", "lr", "learn_step"],
        0.1,
        agent_ids=SHARED_INIT_HP["AGENT_IDS"],
        arch="cnn",
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut in ["None", "act"]
        if individual.mut == "act":
            assert old.actor.activation != individual.actor.activation
            assert individual.actors[0].activation in [
                "ReLU",
                "ELU",
                "GELU",
            ]
        assert old.index == individual.index


# The mutation method applies parameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_parameter_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        1,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.5,
        agent_ids=SHARED_INIT_HP["AGENT_IDS"],
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "param"
        # Due to randomness, sometimes parameters are not different
        # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
        assert old.index == individual.index


# The mutation method applies CNN parameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_cnn_parameter_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        0,
        0,
        1,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.5,
        agent_ids=SHARED_INIT_HP["AGENT_IDS"],
        arch="cnn",
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "param"
        # Due to randomness, sometimes parameters are not different
        # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
        assert old.index == individual.index


# The mutation method applies architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_architecture_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop
):
    for _ in range(10):
        population = init_pop

        mutations = Mutations(
            algo,
            0,
            1,
            0.5,
            0,
            0,
            0,
            ["batch_size", "lr", "learn_step"],
            0.5,
            agent_ids=SHARED_INIT_HP["AGENT_IDS"],
            device=device if not distributed else None,
            accelerator=accelerator if distributed else None,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = [
            mutations.architecture_mutate(agent) for agent in new_population
        ]

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population):
            assert individual.mut == "arch"
            # Due to randomness and constraints on size, sometimes architectures are not different
            # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
            assert old.index == individual.index


# The mutation method applies architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(3, 32, 32)),
            generate_multi_agent_discrete_spaces(2, 2),
            {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
        ),
    ],
)
def test_mutation_applies_cnn_architecture_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop
):
    pre_training_mut = False

    population = init_pop

    mutations = Mutations(
        algo,
        0,
        1,
        0.5,
        0,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.5,
        agent_ids=SHARED_INIT_HP["AGENT_IDS"],
        arch="cnn",
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "arch"
        # Due to randomness and constraints on size, sometimes architectures are not different
        # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
        assert old.index == individual.index


# The mutation method applies BERT architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, distributed, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator, mut_method",
    [
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
            [
                "add_encoder_layer",
                "remove_encoder_layer",
                "add_decoder_layer",
                "remove_decoder_layer",
            ],
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
            [
                "add_encoder_layer",
                "remove_encoder_layer",
                "add_decoder_layer",
                "remove_decoder_layer",
            ],
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
            [
                "add_encoder_layer",
                "remove_encoder_layer",
                "add_decoder_layer",
                "remove_decoder_layer",
            ],
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
            ["add_node", "remove_node"],
        ),
        (
            "MADDPG",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
            ["add_node", "remove_node"],
        ),
        (
            "MADDPG",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
            ["add_node", "remove_node"],
        ),
        (
            "MATD3",
            False,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
            ["add_node", "remove_node"],
        ),
        (
            "MATD3",
            True,
            generate_multi_agent_box_spaces(2, shape=(4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            {"hidden_size": [8]},
            SHARED_INIT_HP_MA,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            Accelerator(device_placement=False),
            ["add_node", "remove_node"],
        ),
    ],
)
def test_mutation_applies_bert_architecture_mutations_multi_agent(
    algo, distributed, device, accelerator, init_pop, mut_method
):
    population = init_pop

    mutations = Mutations(
        algo,
        0,
        1,
        0.5,
        0,
        0,
        0,
        ["batch_size", "lr", "learn_step"],
        0.5,
        arch="bert",
        device=device if not distributed else None,
        accelerator=accelerator if distributed else None,
    )

    class DummyRNG:
        def choice(self, a, size=None, replace=True, p=None):
            return [np.random.choice(mut_method)]

    mutations.rng = DummyRNG()

    for individual in population:

        if distributed:
            adam_actors = [
                actor_optimizer.optimizer
                for actor_optimizer in individual.actor_optimizers
            ]
            if algo == "MATD3":
                adam_critics_1 = [
                    critic_optimizer_1.optimizer
                    for critic_optimizer_1 in individual.critic_1_optimizers
                ]
                adam_critics_2 = [
                    critic_optimizer_2.optimizer
                    for critic_optimizer_2 in individual.critic_2_optimizers
                ]
            else:
                adam_critics = [
                    critic_optimizer.optimizer
                    for critic_optimizer in individual.critic_optimizers
                ]
        else:
            adam_actors = [
                actor_optimizer for actor_optimizer in individual.actor_optimizers
            ]
            if algo == "MATD3":
                adam_critics_1 = [
                    critic_optimizer_1
                    for critic_optimizer_1 in individual.critic_1_optimizers
                ]
                adam_critics_2 = [
                    critic_optimizer_2
                    for critic_optimizer_2 in individual.critic_2_optimizers
                ]
            else:
                adam_critics = [
                    critic_optimizer
                    for critic_optimizer in individual.critic_optimizers
                ]

        if algo == "MADDPG":
            print(adam_critics)

        individual.actors = [EvolvableBERT([12], [12])]
        individual.actor_targets = copy.deepcopy(individual.actors)
        if algo == "MADDPG":
            individual.critics = [EvolvableBERT([12], [12])]
            individual.critic_targets = copy.deepcopy(individual.critics)


            individual.actor_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.actors,
                optimizer_kwargs={"lr": individual.lr_actor},
                network_names=individual.actor_optimizers.network_names,
                multiagent=True
            )
            individual.critic_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.critics,
                optimizer_kwargs={"lr": individual.lr_critic},
                network_names=individual.critic_optimizers.network_names,
                multiagent=True
            )

        else:
            individual.critics_1 = [EvolvableBERT([12], [12])]
            individual.critic_targets_1 = copy.deepcopy(individual.critics_1)
            individual.critics_2 = [EvolvableBERT([12], [12])]
            individual.critic_targets_2 = copy.deepcopy(individual.critics_2)
            individual.actor_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.actors,
                optimizer_kwargs={"lr": individual.lr_actor},
                network_names=individual.actor_optimizers.network_names,
                multiagent=True
            )
            individual.critic_1_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.critics_1,
                optimizer_kwargs={"lr": individual.lr_critic},
                network_names=individual.critic_1_optimizers.network_names,
                multiagent=True
            )

            individual.critic_2_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.critics_2,
                optimizer_kwargs={"lr": individual.lr_critic},
                network_names=individual.critic_2_optimizers.network_names,
                multiagent=True
            )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = [
        mutations.architecture_mutate(agent) for agent in new_population
    ]

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert individual.mut == "arch"
        # Due to randomness and constraints on size, sometimes architectures are not different
        # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
        assert old.index == individual.index


@pytest.mark.parametrize(
    "algo, observation_space, action_space, net_config, INIT_HP, population_size, device, accelerator",
    [
        (
            "DQN",
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "Rainbow DQN",
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8, 8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "DDPG",
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "TD3",
            generate_random_box_space((4,)),
            generate_random_box_space((2,), low=-1, high=1),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "PPO",
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "CQN",
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralUCB",
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
        (
            "NeuralTS",
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            {"hidden_size": [8]},
            SHARED_INIT_HP,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        ),
    ],
)
def test_reinit_opt(algo, init_pop):
    population = init_pop

    mutations = Mutations(
        algo,
        1,
        1,
        1,
        1,
        1,
        1,
        ["batch_size", "lr", "learn_step"],
        0.5,
    )

    new_population = [agent.clone() for agent in population]
    mutations.reinit_opt(new_population[0])

    opt_attr = new_population[0].registry.optimizers[0].name
    new_opt = getattr(new_population[0], opt_attr)
    old_opt = getattr(population[0], opt_attr)

    assert str(new_opt.state_dict()) == str(old_opt.state_dict())
