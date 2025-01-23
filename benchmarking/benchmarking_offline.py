import h5py
import torch
import torch.nn as nn
import yaml

from agilerl.algorithms.core.base import RLAlgorithm
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.wrappers.make_evolvable import MakeEvolvable
from agilerl.training.train_offline import train_offline
from agilerl.utils.algo_utils import observation_space_channels_to_first
from agilerl.utils.utils import (
    create_population,
    make_vect_envs,
    print_hyperparams
)

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')

class BasicNetActor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Softmax())  # Sigmoid activation

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============ AgileRL ============")
    print(f"DEVICE: {device}")

    env = make_vect_envs(INIT_HP["ENV_NAME"], num_envs=INIT_HP["NUM_ENVS"])

    observation_space = env.single_observation_space
    action_space = env.single_action_space
    if INIT_HP["CHANNELS_LAST"]:
        observation_space = observation_space_channels_to_first(observation_space)

    state_dim = RLAlgorithm.get_state_dim(observation_space)
    action_dim = RLAlgorithm.get_action_dim(action_space)
    
    actor = BasicNetActor(state_dim[0], [32, 32], action_dim)
    actor_network = MakeEvolvable(
        actor, input_tensor=torch.ones(state_dim), device=device
    )

    dataset = h5py.File(INIT_HP["DATASET"], "r")

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        INIT_HP["MEMORY_SIZE"], field_names=field_names, device=device
    )
    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )
    mutations = Mutations(
        no_mutation=MUTATION_PARAMS["NO_MUT"],
        architecture=MUTATION_PARAMS["ARCH_MUT"],
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],
        parameters=MUTATION_PARAMS["PARAMS_MUT"],
        activation=MUTATION_PARAMS["ACT_MUT"],
        rl_hp=MUTATION_PARAMS["RL_HP_MUT"],
        mutation_sd=MUTATION_PARAMS["MUT_SD"],
        min_lr=MUTATION_PARAMS["MIN_LR"],
        max_lr=MUTATION_PARAMS["MAX_LR"],
        min_batch_size=MUTATION_PARAMS["MAX_BATCH_SIZE"],
        max_batch_size=MUTATION_PARAMS["MAX_BATCH_SIZE"],
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    agent_pop = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        actor_network=actor_network,
        INIT_HP=INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=INIT_HP["NUM_ENVS"],
        device=device,
    )

    trained_pop, pop_fitnesses = train_offline(
        env,
        INIT_HP["ENV_NAME"],
        dataset,
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
    )

    print_hyperparams(trained_pop)
    # plot_population_score(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()

    env.close()


if __name__ == "__main__":
    with open("configs/training/cqn.yaml") as file:
        cqn_config = yaml.safe_load(file)
    INIT_HP = cqn_config["INIT_HP"]
    MUTATION_PARAMS = cqn_config["MUTATION_PARAMS"]
    NET_CONFIG = cqn_config["NET_CONFIG"]
    # DISTRIBUTED_TRAINING = cqn_config["DISTRIBUTED_TRAINING"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)
