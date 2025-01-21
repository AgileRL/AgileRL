import torch
import yaml

from agilerl.algorithms.core.base import RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.networks.custom_modules import RainbowMLP
from agilerl.training.train_off_policy import train_off_policy
from agilerl.utils.algo_utils import observation_space_channels_to_first
from agilerl.utils.utils import (
    create_population,
    make_vect_envs,
    print_hyperparams
)
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')

def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("============ AgileRL ============")
    print(f"DEVICE: {device}")

    env = make_vect_envs(INIT_HP["ENV_NAME"], num_envs=INIT_HP["NUM_ENVS"])

    observation_space = env.single_observation_space
    action_space = env.single_action_space
    if INIT_HP["CHANNELS_LAST"]:
        observation_space = observation_space_channels_to_first(observation_space)

    field_names = ["state", "action", "reward", "next_state", "done"]
    n_step_memory = None
    per = INIT_HP["PER"]
    n_step = True if INIT_HP["N_STEP"] > 1 else False
    if per:
        memory = PrioritizedReplayBuffer(
            memory_size=INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
            num_envs=INIT_HP["NUM_ENVS"],
            alpha=INIT_HP["ALPHA"],
            gamma=INIT_HP["GAMMA"],
            device=device,
        )
        if n_step:
            n_step_memory = MultiStepReplayBuffer(
                memory_size=INIT_HP["MEMORY_SIZE"],
                field_names=field_names,
                num_envs=INIT_HP["NUM_ENVS"],
                n_step=INIT_HP["N_STEP"],
                gamma=INIT_HP["GAMMA"],
                device=device,
            )
    elif n_step:
        memory = ReplayBuffer(
            memory_size=INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
            device=device,
        )
        n_step_memory = MultiStepReplayBuffer(
            memory_size=INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
            num_envs=INIT_HP["NUM_ENVS"],
            n_step=INIT_HP["N_STEP"],
            gamma=INIT_HP["GAMMA"],
            device=device,
        )
    else:
        memory = ReplayBuffer(
            memory_size=INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
            device=device,
        )

    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )
    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=MUTATION_PARAMS["NO_MUT"],
        architecture=MUTATION_PARAMS["ARCH_MUT"],
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],
        parameters=MUTATION_PARAMS["PARAMS_MUT"],
        activation=MUTATION_PARAMS["ACT_MUT"],
        rl_hp=MUTATION_PARAMS["RL_HP_MUT"],
        mutation_sd=MUTATION_PARAMS["MUT_SD"],
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    state_dim = RLAlgorithm.get_state_dim(observation_space)
    action_dim = RLAlgorithm.get_action_dim(action_space)
    if use_net:
        actor = RainbowMLP(
            num_inputs=state_dim[0],
            num_outputs=action_dim,
            output_vanish=True,
            layer_norm=True,
            num_atoms=51,
            support=torch.linspace(-200, 200, 51).to(device),
            device=device,
            hidden_size=[128, 128],
            activation="ReLU",
            output_activation="ReLU",
        )
    else:
        actor = None

    hp_config = HyperparameterConfig(
        lr = RLParameter(min=MUTATION_PARAMS['MIN_LR'], max=MUTATION_PARAMS['MAX_LR']),
        batch_size = RLParameter(
            min=MUTATION_PARAMS['MIN_BATCH_SIZE'],
            max=MUTATION_PARAMS['MAX_BATCH_SIZE'],
            dtype=int
            ),
        learn_step = RLParameter(
            min=MUTATION_PARAMS['MIN_LEARN_STEP'],
            max=MUTATION_PARAMS['MAX_LEARN_STEP'],
            dtype=int,
            grow_factor=1.5,
            shrink_factor=0.75
            )
    )

    agent_pop = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        hp_config=hp_config,
        actor_network=actor,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=INIT_HP["NUM_ENVS"],
        device=device,
    )

    trained_pop, pop_fitnesses = train_off_policy(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        n_step_memory=n_step_memory,
        n_step=n_step,
        per=per,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        learning_delay=INIT_HP["LEARNING_DELAY"],
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
        save_elite=True,
        elite_path="elite_rainbow.pt",
    )

    print_hyperparams(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()

    env.close()


if __name__ == "__main__":
    with open("configs/training/dqn_rainbow.yaml") as file:
        rainbow_dqn_config = yaml.safe_load(file)
    INIT_HP = rainbow_dqn_config["INIT_HP"]
    MUTATION_PARAMS = rainbow_dqn_config["MUTATION_PARAMS"]
    NET_CONFIG = rainbow_dqn_config["NET_CONFIG"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net=False)
