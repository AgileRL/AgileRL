import numpy as np
import pytest
import torch
from accelerate import Accelerator
from gymnasium import spaces

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.hpo.mutation import Mutations
from agilerl.modules.bert import EvolvableBERT
from agilerl.utils.utils import create_population
from tests.helper_functions import (
    assert_equal_state_dict,
    gen_multi_agent_dict_or_tuple_spaces,
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_random_box_space,
)

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
def ac_hp_config():
    return HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )


@pytest.fixture
def default_hp_config():
    return HyperparameterConfig(
        lr=RLParameter(min=6.25e-5, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=1, max=10, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )


@pytest.fixture
def encoder_mlp_config():
    return {"encoder_config": {"hidden_size": [8]}}


@pytest.fixture
def encoder_simba_config():
    return {
        "simba": True,
        "encoder_config": {
            "hidden_size": 64,
            "num_blocks": 3,
        },
    }


@pytest.fixture
def encoder_cnn_config():
    return {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }


@pytest.fixture
def encoder_multi_input_config():
    return {
        "encoder_config": {
            "cnn_config": {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            },
            "mlp_config": {
                "hidden_size": [8],
            },
            "lstm_config": {
                "hidden_size": 8,
            },
        }
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
    hp_config,
    request,
):
    if hp_config is not None:
        hp_config = request.getfixturevalue(hp_config)

    return create_population(
        algo=algo,
        observation_space=observation_space,
        action_space=action_space,
        hp_config=hp_config,
        net_config=request.getfixturevalue(net_config),
        INIT_HP=INIT_HP,
        population_size=population_size,
        device=device,
        accelerator=accelerator,
    )


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
    [(generate_random_box_space((4,)), "encoder_mlp_config")],
)
@pytest.mark.parametrize("action_space", [generate_discrete_space(2)])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", ["default_hp_config"])
def test_mutation_no_options(device, init_pop):
    pre_training_mut = True

    population = init_pop

    mutations = Mutations(0, 0, 0, 0, 0, 0, 0.1, device=device)

    new_population = [agent.clone() for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert str(old.actor.state_dict()) == str(individual.actor.state_dict())

    del mutations
    del population
    del mutated_population
    del new_population


#### Single-agent algorithm mutations ####
# The mutation method applies random mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, hp_config, action_space",
    [
        ("DQN", "default_hp_config", generate_discrete_space(2)),
        ("Rainbow DQN", "default_hp_config", generate_discrete_space(2)),
        ("DDPG", "ac_hp_config", generate_random_box_space((4,), low=-1, high=1)),
        ("TD3", "ac_hp_config", generate_random_box_space((4,), low=-1, high=1)),
        ("PPO", "default_hp_config", generate_discrete_space(2)),
        ("CQN", "default_hp_config", generate_discrete_space(2)),
        ("NeuralUCB", "default_hp_config", generate_discrete_space(2)),
        ("NeuralTS", "default_hp_config", generate_discrete_space(2)),
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_mlp_config"),
        (generate_random_box_space((3, 16, 16)), "encoder_cnn_config"),
        (generate_dict_or_tuple_space(1, 1), "encoder_multi_input_config"),
        (generate_discrete_space(4), "encoder_mlp_config"),
    ],
)
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_random_mutations(algo, device, accelerator, init_pop):
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

    for agent in population:
        if accelerator is not None:
            agent.unwrap_models()

    mutated_population = mutations.mutation(population, pre_training_mut)

    assert len(mutated_population) == len(population)
    assert mutated_population[0].mut == "None"  # Satisfies mutate_elite=False condition
    for individual in mutated_population:
        policy = getattr(individual, individual.registry.policy)
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

    del mutations
    del population
    del mutated_population

    torch.cuda.empty_cache()  # Free up GPU memory


@pytest.mark.parametrize(
    "algo, hp_config, action_space",
    [
        ("DQN", "default_hp_config", generate_discrete_space(2)),
        ("DDPG", "ac_hp_config", generate_random_box_space((4,), low=-1, high=1)),
        ("TD3", "ac_hp_config", generate_random_box_space((4,), low=-1, high=1)),
        ("PPO", "default_hp_config", generate_discrete_space(2)),
        ("CQN", "default_hp_config", generate_discrete_space(2)),
        ("NeuralUCB", "default_hp_config", generate_discrete_space(2)),
        ("NeuralTS", "default_hp_config", generate_discrete_space(2)),
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_simba_config"),
    ],
)
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_random_mutations_simba(algo, device, accelerator, init_pop):
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

    for agent in population:
        if accelerator is not None:
            agent.unwrap_models()

    mutated_population = mutations.mutation(population, pre_training_mut)

    assert len(mutated_population) == len(population)
    assert mutated_population[0].mut == "None"  # Satisfies mutate_elite=False condition
    for individual in mutated_population:
        policy = getattr(individual, individual.registry.policy)
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

    del mutations
    del population
    del mutated_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies no mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space",
    [
        ("DQN", generate_discrete_space(2)),
        ("Rainbow DQN", generate_discrete_space(2)),
        ("DDPG", generate_random_box_space((4,), low=-1, high=1)),
        ("TD3", generate_random_box_space((4,), low=-1, high=1)),
        ("PPO", generate_discrete_space(2)),
        ("CQN", generate_discrete_space(2)),
        ("NeuralUCB", generate_discrete_space(2)),
        ("NeuralTS", generate_discrete_space(2)),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_mlp_config"),
        (generate_random_box_space((3, 16, 16)), "encoder_cnn_config"),
        (generate_dict_or_tuple_space(1, 1), "encoder_multi_input_config"),
        (generate_discrete_space(4), "encoder_mlp_config"),
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_no_mutations(device, accelerator, init_pop):
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
        assert str(old.actor.state_dict()) == str(individual.actor.state_dict())

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies no mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space",
    [
        ("DQN", generate_discrete_space(2)),
        ("Rainbow DQN", generate_discrete_space(2)),
        ("DDPG", generate_random_box_space((4,), low=-1, high=1)),
        ("TD3", generate_random_box_space((4,), low=-1, high=1)),
        ("PPO", generate_discrete_space(2)),
        ("CQN", generate_discrete_space(2)),
        ("NeuralUCB", generate_discrete_space(2)),
        ("NeuralTS", generate_discrete_space(2)),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_mlp_config"),
        (generate_random_box_space((3, 16, 16)), "encoder_cnn_config"),
        (generate_dict_or_tuple_space(1, 1), "encoder_multi_input_config"),
        (generate_discrete_space(4), "encoder_mlp_config"),
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_no_mutations_pre_training_mut(device, accelerator, init_pop):
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
        assert str(old.actor.state_dict()) == str(individual.actor.state_dict())

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies RL hyperparameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, hp_config, action_space",
    [
        ("DQN", "default_hp_config", generate_discrete_space(2)),
        ("Rainbow DQN", "default_hp_config", generate_discrete_space(2)),
        ("DDPG", "ac_hp_config", generate_random_box_space((4,), low=-1, high=1)),
        ("TD3", "ac_hp_config", generate_random_box_space((4,), low=-1, high=1)),
        ("PPO", "default_hp_config", generate_discrete_space(2)),
        ("CQN", "default_hp_config", generate_discrete_space(2)),
        ("NeuralUCB", "default_hp_config", generate_discrete_space(2)),
        ("NeuralTS", "default_hp_config", generate_discrete_space(2)),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_mlp_config"),
        (generate_random_box_space((3, 16, 16)), "encoder_cnn_config"),
        (generate_dict_or_tuple_space(1, 1), "encoder_multi_input_config"),
        (generate_discrete_space(4), "encoder_mlp_config"),
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_rl_hp_mutations(
    device, accelerator, hp_config, init_pop, request
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

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space",
    [
        ("DQN", generate_discrete_space(2)),
        ("Rainbow DQN", generate_discrete_space(2)),
        ("DDPG", generate_random_box_space((4,), low=-1, high=1)),
        ("TD3", generate_random_box_space((4,), low=-1, high=1)),
        ("PPO", generate_discrete_space(2)),
        ("CQN", generate_discrete_space(2)),
        ("NeuralUCB", generate_discrete_space(2)),
        ("NeuralTS", generate_discrete_space(2)),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_mlp_config"),
        (generate_random_box_space((3, 16, 16)), "encoder_cnn_config"),
        (generate_dict_or_tuple_space(1, 1), "encoder_multi_input_config"),
        (generate_discrete_space(4), "encoder_mlp_config"),
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_activation_mutations(
    observation_space, device, accelerator, init_pop
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

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_mlp_config"),
        (generate_random_box_space((3, 16, 16)), "encoder_cnn_config"),
        (generate_dict_or_tuple_space(1, 1), "encoder_multi_input_config"),
        (generate_discrete_space(4), "encoder_mlp_config"),
    ],
)
@pytest.mark.parametrize(
    "algo, action_space", [("DDPG", generate_random_box_space((4,), low=-1, high=1))]
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_activation_mutations_no_skip(device, accelerator, init_pop):
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

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies parameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space",
    [
        ("DQN", generate_discrete_space(2)),
        ("Rainbow DQN", generate_discrete_space(2)),
        ("DDPG", generate_random_box_space((4,), low=-1, high=1)),
        ("TD3", generate_random_box_space((4,), low=-1, high=1)),
        ("PPO", generate_discrete_space(2)),
        ("CQN", generate_discrete_space(2)),
        ("NeuralUCB", generate_discrete_space(2)),
        ("NeuralTS", generate_discrete_space(2)),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_mlp_config"),
        (generate_random_box_space((3, 16, 16)), "encoder_cnn_config"),
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_parameter_mutations(algo, device, accelerator, init_pop):
    pre_training_mut = False

    population = init_pop

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
        policy_name = old.registry.policy
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

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize(
    "algo, action_space",
    [
        ("DQN", generate_discrete_space(2)),
        ("Rainbow DQN", generate_discrete_space(2)),
        ("DDPG", generate_random_box_space((4,), low=-1, high=1)),
        ("TD3", generate_random_box_space((4,), low=-1, high=1)),
        ("PPO", generate_discrete_space(2)),
        ("CQN", generate_discrete_space(2)),
        ("NeuralUCB", generate_discrete_space(2)),
        ("NeuralTS", generate_discrete_space(2)),
    ],
)
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_mlp_config"),
        (generate_random_box_space((3, 16, 16)), "encoder_cnn_config"),
        (generate_dict_or_tuple_space(1, 1), "encoder_multi_input_config"),
        (generate_discrete_space(4), "encoder_mlp_config"),
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_mutation_applies_architecture_mutations(algo, device, accelerator, init_pop):
    population = init_pop
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
    for mut_method in mut_methods:

        class DummyRNG:
            def choice(self, a, size=None, replace=True, p=None):
                return [mut_method]

        mutations.rng = DummyRNG()

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = [
            mutations.architecture_mutate(agent) for agent in new_population
        ]

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population):
            policy = getattr(individual, individual.registry.policy)
            assert individual.mut == policy.last_mutation_attr
            # Due to randomness and constraints on size, sometimes architectures are not different
            # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
            assert old.index == individual.index

        assert_equal_state_dict(population, mutated_population)

        torch.cuda.empty_cache()  # Free up GPU memory

    del mutations
    del population
    del mutated_population
    del new_population


# The mutation method applies BERT architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["DDPG"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [(generate_random_box_space((4,)), "encoder_mlp_config")],
)
@pytest.mark.parametrize(
    "action_space", [generate_random_box_space((2,), low=-1, high=1)]
)
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
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
    algo, device, accelerator, mut_method, init_pop
):
    population = init_pop

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

    for individual in population:
        individual.actor = EvolvableBERT([12], [12], device=device)
        individual.actor_target = EvolvableBERT([12], [12], device=device)
        individual.actor_target.load_state_dict(individual.actor.state_dict())
        individual.critic = EvolvableBERT([12], [12], device=device)
        individual.critic_target = EvolvableBERT([12], [12], device=device)
        individual.critic_target.load_state_dict(individual.critic.state_dict())

        individual.actor_optimizer = OptimizerWrapper(
            torch.optim.Adam,
            individual.actor,
            lr=individual.lr_actor,
            network_names=individual.actor_optimizer.network_names,
            lr_name=individual.actor_optimizer.lr_name,
        )

        individual.critic_optimizer = OptimizerWrapper(
            torch.optim.Adam,
            individual.critic,
            lr=individual.lr_critic,
            network_names=individual.critic_optimizer.network_names,
            lr_name=individual.critic_optimizer.lr_name,
        )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = [
        mutations.architecture_mutate(agent) for agent in new_population
    ]

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        policy = getattr(individual, individual.registry.policy)
        assert individual.mut == policy.last_mutation_attr
        # Due to randomness and constraints on size, sometimes architectures are not different
        # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
        assert old.index == individual.index

    # assert_equal_state_dict(population, mutated_population)

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


#### Multi-agent algorithm mutations ####
# The mutation method applies random mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_multi_agent_box_spaces(2, shape=(4,)), "encoder_mlp_config"),
        (generate_multi_agent_box_spaces(2, shape=(3, 16, 16)), "encoder_cnn_config"),
        (gen_multi_agent_dict_or_tuple_spaces(2, 1, 1), "encoder_multi_input_config"),
    ],
)
@pytest.mark.parametrize("action_space", [generate_multi_agent_discrete_spaces(2, 2)])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_random_mutations_multi_agent(
    algo, device, accelerator, init_pop
):
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

    for agent in population:
        if accelerator is not None:
            agent.unwrap_models()

    mutated_population = mutations.mutation(population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for individual in mutated_population:
        policy = getattr(individual, individual.registry.policy)
        assert individual.mut in [
            "None",
            "batch_size",
            "lr",
            "lr_actor",
            "lr_critic",
            "learn_step",
            "act",
            "param",
            policy[0].last_mutation_attr,
        ]

    del mutations
    del population
    del mutated_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies no mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [(generate_multi_agent_box_spaces(2, shape=(4,)), "encoder_mlp_config")],
)
@pytest.mark.parametrize("action_space", [generate_multi_agent_discrete_spaces(2, 2)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
def test_mutation_applies_no_mutations_multi_agent(algo, device, accelerator, init_pop):
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

    for agent in population:
        if accelerator is not None:
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
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [(generate_multi_agent_box_spaces(2, shape=(4,)), "encoder_mlp_config")],
)
@pytest.mark.parametrize("action_space", [generate_multi_agent_discrete_spaces(2, 2)])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
@pytest.mark.parametrize("hp_config", ["ac_hp_config"])
def test_mutation_applies_rl_hp_mutations_multi_agent(
    device, accelerator, init_pop, hp_config, request
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

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_multi_agent_box_spaces(2, shape=(4,)), "encoder_mlp_config"),
        (generate_multi_agent_box_spaces(2, shape=(3, 16, 16)), "encoder_cnn_config"),
        (gen_multi_agent_dict_or_tuple_spaces(2, 1, 1), "encoder_multi_input_config"),
    ],
)
@pytest.mark.parametrize("action_space", [generate_multi_agent_discrete_spaces(2, 2)])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_activation_mutations_multi_agent(
    algo, device, accelerator, init_pop
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

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies activation mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [(generate_multi_agent_box_spaces(2, shape=(4,)), "encoder_mlp_config")],
)
@pytest.mark.parametrize("action_space", [generate_multi_agent_discrete_spaces(2, 2)])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_activation_mutations_multi_agent_no_skip(
    algo, device, accelerator, init_pop
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
            for old_actor, actor in zip(old.actors, individual.actors):
                assert old_actor.activation != actor.activation
                assert individual.actors[0].activation in [
                    "ReLU",
                    "ELU",
                    "GELU",
                ]
        assert old.index == individual.index

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies parameter mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_multi_agent_box_spaces(2, shape=(4,)), "encoder_mlp_config"),
        (generate_multi_agent_box_spaces(2, shape=(3, 16, 16)), "encoder_cnn_config"),
        (gen_multi_agent_dict_or_tuple_spaces(2, 1, 1), "encoder_multi_input_config"),
    ],
)
@pytest.mark.parametrize("action_space", [generate_multi_agent_discrete_spaces(2, 2)])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_parameter_mutations_multi_agent(
    algo, device, accelerator, init_pop
):
    pre_training_mut = False
    population = init_pop

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
        policy_name = old.registry.policy
        old_policy = getattr(old, policy_name)
        new_policy = getattr(individual, policy_name)
        old_sd = old_policy[0].state_dict()
        new_sd = new_policy[0].state_dict()
        mutation_found = False
        for key in old_sd.keys():
            if "norm" in key:  # Skip normalization layers
                continue
            diff_norm = (old_sd[key] - new_sd[key]).norm().item()
            if diff_norm > 1e-6:
                mutation_found = True
                break

        assert mutation_found, f"Mutation not applied for agent index {old.index}"

    del mutations
    del population
    del mutated_population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory


# The mutation method applies architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_multi_agent_box_spaces(2, shape=(4,)), "encoder_mlp_config"),
        (generate_multi_agent_box_spaces(2, shape=(3, 16, 16)), "encoder_cnn_config"),
        (gen_multi_agent_dict_or_tuple_spaces(2, 1, 1), "encoder_multi_input_config"),
    ],
)
@pytest.mark.parametrize("action_space", [generate_multi_agent_discrete_spaces(2, 2)])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
def test_mutation_applies_architecture_mutations_multi_agent(
    algo, device, accelerator, init_pop
):
    population = init_pop
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

    mut_methods = population[0].actors[0].mutation_methods
    for mut_method in mut_methods:

        class DummyRNG:
            def choice(self, a, size=None, replace=True, p=None):
                return [mut_method]

        mutations.rng = DummyRNG()

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = [
            mutations.architecture_mutate(agent) for agent in new_population
        ]

        torch.cuda.empty_cache()  # Free up GPU memory

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population):
            policy = getattr(individual, individual.registry.policy)
            assert individual.mut == policy[0].last_mutation_attr
            # Due to randomness and constraints on size, sometimes architectures are not different
            # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
            assert old.index == individual.index

        assert_equal_state_dict(population, mutated_population)

    del mutations
    del population
    del mutated_population
    del new_population


# The mutation method applies BERT architecture mutations to the population and returns the mutated population.
@pytest.mark.parametrize("algo", ["MADDPG", "MATD3"])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_multi_agent_box_spaces(2, shape=(4,)), "encoder_mlp_config"),
        (generate_multi_agent_box_spaces(2, shape=(3, 16, 16)), "encoder_cnn_config"),
        (gen_multi_agent_dict_or_tuple_spaces(2, 1, 1), "encoder_multi_input_config"),
    ],
)
@pytest.mark.parametrize("action_space", [generate_multi_agent_discrete_spaces(2, 2)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
@pytest.mark.parametrize("population_size", [1])
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
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
def test_mutation_applies_bert_architecture_mutations_multi_agent(
    algo, device, accelerator, init_pop, mut_method
):
    population = init_pop

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

    for individual in population:
        individual.actors = [EvolvableBERT([12], [12], device=device)]
        individual.actor_targets = [EvolvableBERT([12], [12], device=device)]
        individual.actor_targets[0].load_state_dict(individual.actors[0].state_dict())
        if algo == "MADDPG":
            individual.critics = [EvolvableBERT([12], [12], device=device)]
            individual.critic_targets = [EvolvableBERT([12], [12], device=device)]
            individual.critic_targets[0].load_state_dict(
                individual.critics[0].state_dict()
            )

            individual.actor_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.actors,
                lr=individual.lr_actor,
                network_names=individual.actor_optimizers.network_names,
                lr_name=individual.actor_optimizers.lr_name,
                multiagent=True,
            )
            individual.critic_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.critics,
                lr=individual.lr_critic,
                network_names=individual.critic_optimizers.network_names,
                lr_name=individual.critic_optimizers.lr_name,
                multiagent=True,
            )

        else:
            individual.critics_1 = [EvolvableBERT([12], [12], device=device)]
            individual.critic_targets_1 = [EvolvableBERT([12], [12], device=device)]
            individual.critic_targets_1[0].load_state_dict(
                individual.critics_1[0].state_dict()
            )
            individual.critics_2 = [EvolvableBERT([12], [12], device=device)]
            individual.critic_targets_2 = [EvolvableBERT([12], [12], device=device)]
            individual.critic_targets_2[0].load_state_dict(
                individual.critics_2[0].state_dict()
            )
            individual.actor_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.actors,
                lr=individual.lr_actor,
                network_names=individual.actor_optimizers.network_names,
                lr_name=individual.actor_optimizers.lr_name,
                multiagent=True,
            )
            individual.critic_1_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.critics_1,
                lr=individual.lr_critic,
                network_names=individual.critic_1_optimizers.network_names,
                lr_name=individual.critic_1_optimizers.lr_name,
                multiagent=True,
            )

            individual.critic_2_optimizers = OptimizerWrapper(
                torch.optim.Adam,
                individual.critics_2,
                lr=individual.lr_critic,
                network_names=individual.critic_2_optimizers.network_names,
                lr_name=individual.critic_2_optimizers.lr_name,
                multiagent=True,
            )

    new_population = [agent.clone(wrap=False) for agent in population]
    mutated_population = [
        mutations.architecture_mutate(agent) for agent in new_population
    ]

    torch.cuda.empty_cache()  # Free up GPU memory

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        policy = getattr(individual, individual.registry.policy)
        assert individual.mut == policy[0].last_mutation_attr
        # Due to randomness and constraints on size, sometimes architectures are not different
        # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
        assert old.index == individual.index

    # assert_equal_state_dict(population, mutated_population)

    del mutations
    del population
    del mutated_population
    del new_population


@pytest.mark.parametrize(
    "algo, action_space",
    [
        ("DQN", generate_discrete_space(2)),
        ("Rainbow DQN", generate_discrete_space(2)),
        ("DDPG", generate_random_box_space((4,), low=-1, high=1)),
        ("TD3", generate_random_box_space((4,), low=-1, high=1)),
        ("PPO", generate_discrete_space(2)),
        ("CQN", generate_discrete_space(2)),
        ("NeuralUCB", generate_discrete_space(2)),
        ("NeuralTS", generate_discrete_space(2)),
    ],
)
@pytest.mark.parametrize(
    "device", [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (generate_random_box_space((4,)), "encoder_mlp_config"),
    ],
)
@pytest.mark.parametrize("hp_config", [None])
@pytest.mark.parametrize("population_size", [1])
def test_reinit_opt(algo, init_pop):
    population = init_pop
    mutations = Mutations(
        1,
        1,
        1,
        1,
        1,
        1,
        0.5,
    )

    new_population = [agent.clone() for agent in population]
    mutations.reinit_opt(new_population[0])

    opt_attr = new_population[0].registry.optimizers[0].name
    new_opt = getattr(new_population[0], opt_attr)
    old_opt = getattr(population[0], opt_attr)

    assert str(new_opt.state_dict()) == str(old_opt.state_dict())

    del mutations
    del population
    del new_population

    torch.cuda.empty_cache()  # Free up GPU memory
