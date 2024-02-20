import copy

import numpy as np
import torch
from accelerate import Accelerator

from agilerl.algorithms.cqn import CQN
from agilerl.algorithms.ddpg import DDPG
from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.algorithms.ilql import ILQL
from agilerl.algorithms.maddpg import MADDPG
from agilerl.algorithms.matd3 import MATD3
from agilerl.algorithms.neural_ts_bandit import NeuralTS
from agilerl.algorithms.neural_ucb_bandit import NeuralUCB
from agilerl.algorithms.ppo import PPO
from agilerl.algorithms.td3 import TD3
from agilerl.hpo.mutation import Mutations
from agilerl.networks.evolvable_bert import EvolvableBERT
from agilerl.utils.utils import initialPopulation

# from pytest_mock import mocker


# Shared HP dict that can be used by any algorithm
SHARED_INIT_HP = {
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
    "DISCRETE_ACTIONS": True,
    "GAE_LAMBDA": 0.95,
    "ACTION_STD_INIT": 0.6,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "TARGET_KL": None,
    "UPDATE_EPOCHS": 4,
    "MAX_ACTION": 1,
    "MIN_ACTION": -1,
    "N_AGENTS": 2,
    "AGENT_IDS": ["agent1", "agent2"],
    "LAMBDA": 1.0,
    "REG": 0.000625,
    "CHANNELS_LAST": False,
}

SHARED_INIT_HP_MA = copy.deepcopy(SHARED_INIT_HP)
SHARED_INIT_HP_MA["MAX_ACTION"] = [(1,), (1,)]
SHARED_INIT_HP_MA["MIN_ACTION"] = [(-1,), (-1,)]


# The constructor initializes all the attributes of the Mutations class correctly.
def test_constructor_initializes_attributes():
    algo = {
        "actor": {
            "eval": "actor",
            "target": "actor_target",
            "optimizer": "optimizer_type",
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
    assert mutations.arch == arch
    assert mutations.no_mut == no_mutation
    assert mutations.architecture_mut == architecture
    assert mutations.new_layer_prob == new_layer_prob
    assert mutations.parameters_mut == parameters
    assert mutations.activation_mut == activation
    assert mutations.rl_hp_mut == rl_hp
    assert mutations.rl_hp_selection == rl_hp_selection
    assert mutations.mutation_sd == mutation_sd
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
    assert mutations.multi_agent is False
    assert mutations.algo == algo


# The constructor initializes all the attributes of the Mutations class correctly for multi-agent.
def test_constructor_initializes_attributes_multi_agent():
    algo = "MATD3"
    algo_dict = {
        "actor": {
            "eval": "actors",
            "target": "actor_targets",
            "optimizer": "actor_optimizers_type",
        },
        "critics": [
            {
                "eval": "critics_1",
                "target": "critic_targets_1",
                "optimizer": "critic_1_optimizers_type",
            },
            {
                "eval": "critics_2",
                "target": "critic_targets_2",
                "optimizer": "critic_2_optimizers_type",
            },
        ],
    }
    no_mutation = 0.1
    architecture = 0.2
    new_layer_prob = 0.3
    parameters = 0.4
    activation = 0.5
    rl_hp = 0.6
    rl_hp_selection = ["batch_size", "lr", "learn_step"]
    mutation_sd = 0.7
    min_lr = 0.0001
    max_lr = 0.01
    min_learn_step = 1
    max_learn_step = 120
    min_batch_size = 8
    max_batch_size = 1024
    agent_ids = [1, 2, 3]
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
    assert mutations.arch == arch
    assert mutations.no_mut == no_mutation
    assert mutations.architecture_mut == architecture
    assert mutations.new_layer_prob == new_layer_prob
    assert mutations.parameters_mut == parameters
    assert mutations.activation_mut == activation
    assert mutations.rl_hp_mut == rl_hp
    assert mutations.rl_hp_selection == rl_hp_selection
    assert mutations.mutation_sd == mutation_sd
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
    assert mutations.multi_agent is True
    assert mutations.algo == algo_dict


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
def test_mutation_no_options():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    population_size = 1
    pre_training_mut = True

    population = initialPopulation(
        algo="DQN",
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        net_config=net_config,
        INIT_HP=SHARED_INIT_HP,
        population_size=population_size,
        device=device,
    )

    mutations = Mutations(
        "DQN", 0, 0, 0, 0, 0, 0, ["batch_size", "lr", "learn_step"], 0.1, device=device
    )

    new_population = [agent.clone() for agent in population]
    mutated_population = mutations.mutation(new_population, pre_training_mut)

    assert len(mutated_population) == len(population)
    for old, individual in zip(population, mutated_population):
        assert str(old.actor.state_dict()) == str(individual.actor.state_dict())


#### Single-agent algorithm mutations ####
# The mutation method applies random mutations to the population and returns the mutated population.
def test_mutation_applies_random_mutations():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = True

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            mutated_population = mutations.mutation(population, pre_training_mut)

            assert len(mutated_population) == len(population)
            assert (
                mutated_population[0].mut == "None"
            )  # Satisfies mutate_elite=False condition
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
def test_mutation_applies_no_mutations():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [4]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "ILQL": ILQL,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut in ["None"]
                assert old.index == individual.index
                assert old.actor != individual.actor
                assert str(old.actor.state_dict()) == str(individual.actor.state_dict())


# The mutation method applies no mutations to the population and returns the mutated population.
def test_mutation_applies_no_mutations_pre_training_mut():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [4]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = True

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "ILQL": ILQL,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            print(population, mutated_population)

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
def test_mutation_applies_rl_hp_mutations():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "ILQL": ILQL,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

                new_population = [agent.clone() for agent in population]
                mutated_population = mutations.mutation(
                    new_population, pre_training_mut
                )

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
                        assert (
                            mutations.min_lr <= individual.lr_actor <= mutations.max_lr
                        )
                    if individual.mut == "lr_critic":
                        assert (
                            mutations.min_lr <= individual.lr_critic <= mutations.max_lr
                        )
                    if individual.mut == "ls":
                        assert (
                            mutations.min_learn_step
                            <= individual.learn_step
                            <= mutations.max_learn_step
                        )
                    assert old.index == individual.index


# The mutation method applies activation mutations to the population and returns the mutated population.
def test_mutation_applies_activation_mutations():
    state_dim = [3]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "ILQL": ILQL,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut in ["None", "act"]
                if individual.mut == "act":
                    assert old.actor.mlp_activation != individual.actor.mlp_activation
                    assert individual.actor.mlp_activation in ["ReLU", "ELU", "GELU"]
                assert old.index == individual.index


# The mutation method applies activation mutations to the population and returns the mutated population.
def test_mutation_applies_activation_mutations_no_skip():
    state_dim = [3]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"DDPG": DDPG}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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
            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut in ["None", "act"]
                if individual.mut == "act":
                    assert old.actor.mlp_activation != individual.actor.mlp_activation
                    assert individual.actor.mlp_activation in ["ReLU", "ELU", "GELU"]
                assert old.index == individual.index


# The mutation method applies CNN activation mutations to the population and returns the mutated population.
def test_mutation_applies_cnn_activation_mutations():
    state_dim = [3, 32, 32]
    action_dim = 2
    one_hot = False
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "ILQL": ILQL,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut in ["None", "act"]
                if individual.mut == "act":
                    assert old.actor.mlp_activation != individual.actor.mlp_activation
                    assert individual.actor.mlp_activation in ["ReLU", "ELU", "GELU"]
                assert old.index == individual.index


# The mutation method applies parameter mutations to the population and returns the mutated population.
def test_mutation_applies_parameter_mutations():
    state_dim = [3]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "ILQL": ILQL,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut == "param"
                # Due to randomness, sometimes parameters are not different
                # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
                assert old.index == individual.index


# The mutation method applies CNN parameter mutations to the population and returns the mutated population.
def test_mutation_applies_cnn_parameter_mutations():
    state_dim = [3, 32, 32]
    action_dim = 2
    one_hot = False
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "ILQL": ILQL,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut == "param"
                # Due to randomness, sometimes parameters are not different
                # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
                assert old.index == individual.index


# The mutation method applies architecture mutations to the population and returns the mutated population.
def test_mutation_applies_architecture_mutations():
    state_dim = [3]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [32, 32]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "ILQL": ILQL,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            for mut_method in [
                ["add_mlp_layer", "remove_mlp_layer"],
                ["add_mlp_node", "remove_mlp_node"],
            ]:
                population = initialPopulation(
                    algo=algo,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    one_hot=one_hot,
                    net_config=net_config,
                    INIT_HP=SHARED_INIT_HP,
                    population_size=population_size,
                    device=device if not distributed else None,
                    accelerator=accelerator if distributed else None,
                )

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

                class DummyRNG:
                    def choice(self, a, size=None, replace=True, p=None):
                        return [np.random.choice(mut_method)]

                mutations.rng = DummyRNG()

                new_population = [agent.clone() for agent in population]
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
def test_mutation_applies_cnn_architecture_mutations():
    state_dim = [3, 32, 32]
    action_dim = 2
    one_hot = False
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = True

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "ILQL": ILQL,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut == "arch"
                # Due to randomness and constraints on size, sometimes architectures are not different
                # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
                assert old.index == individual.index


# The mutation method applies BERT architecture mutations to the population and returns the mutated population.
def test_mutation_applies_bert_architecture_mutations():
    state_dim = [3]
    action_dim = 2
    one_hot = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1

    algo_classes = {"DDPG": DDPG}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            for mut_method in [
                [
                    "add_encoder_layer",
                    "remove_encoder_layer",
                    "add_decoder_layer",
                    "remove_decoder_layer",
                ],
                ["add_node", "remove_node"],
            ]:
                population = initialPopulation(
                    algo=algo,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    one_hot=one_hot,
                    INIT_HP=SHARED_INIT_HP,
                    net_config={"arch": "mlp", "h_size": [32, 32]},
                    population_size=population_size,
                    device=device if not distributed else None,
                    accelerator=accelerator if distributed else None,
                )

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

                new_population = [agent.clone() for agent in population]
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
def test_mutation_applies_random_mutations_multi_agent():
    state_dim = [[4], [4]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP_MA,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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
def test_mutation_applies_no_mutations_multi_agent():
    state_dim = [[4], [4]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP_MA,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            mutated_population = mutations.mutation(population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut in ["None"]
                assert old.index == individual.index
                assert old.actors == individual.actors


# The mutation method applies RL hyperparameter mutations to the population and returns the mutated population.
def test_mutation_applies_rl_hp_mutations_multi_agent():
    state_dim = [[4], [4]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP_MA,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
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
def test_mutation_applies_activation_mutations_multi_agent():
    state_dim = [[4], [4]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP_MA,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut in ["None", "act"]
                if individual.mut == "act":
                    for old_actor, actor in zip(old.actors, individual.actors):
                        assert old_actor.mlp_activation != actor.mlp_activation
                        assert individual.actors[0].mlp_activation in [
                            "ReLU",
                            "ELU",
                            "GELU",
                        ]
                assert old.index == individual.index


# The mutation method applies activation mutations to the population and returns the mutated population.
def test_mutation_applies_activation_mutations_multi_agent_no_skip():
    state_dim = [[4], [4]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"MADDPG": MADDPG}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP_MA,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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
            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut in ["None", "act"]
                if individual.mut == "act":
                    for old_actor, actor in zip(old.actors, individual.actors):
                        assert old_actor.mlp_activation != actor.mlp_activation
                        assert individual.actors[0].mlp_activation in [
                            "ReLU",
                            "ELU",
                            "GELU",
                        ]
                assert old.index == individual.index


# The mutation method applies CNN activation mutations to the population and returns the mutated population.
def test_mutation_applies_cnn_activation_mutations_multi_agent():
    state_dim = [[3, 32, 32], [3, 32, 32]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP_MA,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut in ["None", "act"]
                if individual.mut == "act":
                    assert old.actor.mlp_activation != individual.actor.mlp_activation
                    assert individual.actors[0].mlp_activation in [
                        "ReLU",
                        "ELU",
                        "GELU",
                    ]
                assert old.index == individual.index


# The mutation method applies parameter mutations to the population and returns the mutated population.
def test_mutation_applies_parameter_mutations_multi_agent():
    state_dim = [[4], [4]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP_MA,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut == "param"
                # Due to randomness, sometimes parameters are not different
                # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
                assert old.index == individual.index


# The mutation method applies CNN parameter mutations to the population and returns the mutated population.
def test_mutation_applies_cnn_parameter_mutations_multi_agent():
    state_dim = [[3, 32, 32], [3, 32, 32]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP_MA,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut == "param"
                # Due to randomness, sometimes parameters are not different
                # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
                assert old.index == individual.index


# The mutation method applies architecture mutations to the population and returns the mutated population.
def test_mutation_applies_architecture_mutations_multi_agent():
    state_dim = [[4], [4]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            for mut_method in [
                ["add_mlp_layer", "remove_mlp_layer"],
                ["add_mlp_node", "remove_mlp_node"],
            ]:
                population = initialPopulation(
                    algo=algo,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    one_hot=one_hot,
                    net_config=net_config,
                    INIT_HP=SHARED_INIT_HP_MA,
                    population_size=population_size,
                    device=device if not distributed else None,
                    accelerator=accelerator if distributed else None,
                )

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

                class DummyRNG:
                    def choice(self, a, size=None, replace=True, p=None):
                        return [np.random.choice(mut_method)]

                mutations.rng = DummyRNG()

                new_population = [agent.clone() for agent in population]
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
def test_mutation_applies_cnn_architecture_mutations_multi_agent():
    state_dim = [[3, 32, 32], [3, 32, 32]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1
    pre_training_mut = False

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            population = initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP_MA,
                population_size=population_size,
                device=device if not distributed else None,
                accelerator=accelerator if distributed else None,
            )

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

            new_population = [agent.clone() for agent in population]
            mutated_population = mutations.mutation(new_population, pre_training_mut)

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population):
                assert individual.mut == "arch"
                # Due to randomness and constraints on size, sometimes architectures are not different
                # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
                assert old.index == individual.index


# The mutation method applies BERT architecture mutations to the population and returns the mutated population.
def test_mutation_applies_bert_architecture_mutations_multi_agent():
    state_dim = [[4], [4]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    population_size = 1

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for distributed in [False, True]:
        for algo in algo_classes.keys():
            for mut_method in [
                [
                    "add_encoder_layer",
                    "remove_encoder_layer",
                    "add_decoder_layer",
                    "remove_decoder_layer",
                ],
                ["add_node", "remove_node"],
            ]:
                population = initialPopulation(
                    algo=algo,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    one_hot=one_hot,
                    net_config=net_config,
                    INIT_HP=SHARED_INIT_HP_MA,
                    population_size=population_size,
                    device=device if not distributed else None,
                    accelerator=accelerator if distributed else None,
                )

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
                    individual.actors = [EvolvableBERT([12], [12])]
                    individual.actor_targets = copy.deepcopy(individual.actors)
                    if algo == "MADDPG":
                        individual.critics = [EvolvableBERT([12], [12])]
                        individual.critic_targets = copy.deepcopy(individual.critics)
                    else:
                        individual.critics_1 = [EvolvableBERT([12], [12])]
                        individual.critic_targets_1 = copy.deepcopy(
                            individual.critics_1
                        )
                        individual.critics_2 = [EvolvableBERT([12], [12])]
                        individual.critic_targets_2 = copy.deepcopy(
                            individual.critics_2
                        )

                new_population = [agent.clone() for agent in population]
                mutated_population = [
                    mutations.architecture_mutate(agent) for agent in new_population
                ]

                assert len(mutated_population) == len(population)
                for old, individual in zip(population, mutated_population):
                    assert individual.mut == "arch"
                    # Due to randomness and constraints on size, sometimes architectures are not different
                    # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
                    assert old.index == individual.index


def test_reinit_opt():
    state_dim = [3]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    population_size = 1

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
        "NeuralUCB": NeuralUCB,
        "NeuralTS": NeuralTS,
    }

    for algo in algo_classes.keys():
        population = initialPopulation(
            algo=algo,
            state_dim=state_dim,
            action_dim=action_dim,
            one_hot=one_hot,
            net_config=net_config,
            INIT_HP=SHARED_INIT_HP,
            population_size=population_size,
        )

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

        new_opt = getattr(
            new_population[0], mutations.algo["actor"]["optimizer"].replace("_type", "")
        )
        old_opt = getattr(
            population[0], mutations.algo["actor"]["optimizer"].replace("_type", "")
        )

        assert str(new_opt.state_dict()) == str(old_opt.state_dict())
