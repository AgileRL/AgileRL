import torch
import yaml

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.training.train_off_policy import train_off_policy
from agilerl.utils.utils import create_population, make_vect_envs, print_hyperparams

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============ AgileRL ============")
    print(f"DEVICE: {device}")

    env = make_vect_envs(INIT_HP["ENV_NAME"], num_envs=INIT_HP["NUM_ENVS"])

    try:
        state_dim = (env.single_observation_space.n,)
        one_hot = True
    except Exception:
        state_dim = env.single_observation_space.shape
        one_hot = False
    try:
        action_dim = env.single_action_space.n
    except Exception:
        action_dim = env.single_action_space.shape[0]

    if INIT_HP["CHANNELS_LAST"]:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    if INIT_HP["ALGO"] == "TD3":
        max_action = float(env.single_action_space.high[0])
        INIT_HP["MAX_ACTION"] = max_action

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=INIT_HP["MEMORY_SIZE"], field_names=field_names, device=device
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
        rl_hp_selection=MUTATION_PARAMS["RL_HP_SELECTION"],
        mutation_sd=MUTATION_PARAMS["MUT_SD"],
        min_lr=MUTATION_PARAMS["MIN_LR"],
        max_lr=MUTATION_PARAMS["MAX_LR"],
        min_batch_size=MUTATION_PARAMS["MAX_BATCH_SIZE"],
        max_batch_size=MUTATION_PARAMS["MAX_BATCH_SIZE"],
        min_learn_step=MUTATION_PARAMS["MIN_LEARN_STEP"],
        max_learn_step=MUTATION_PARAMS["MAX_LEARN_STEP"],
        arch=NET_CONFIG["arch"],
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    if use_net:
        # Currently set up for TD3
        actor = EvolvableMLP(
            num_inputs=state_dim[0],
            num_outputs=action_dim,
            output_vanish=False,
            init_layers=False,
            layer_norm=False,
            num_atoms=51,
            support=torch.linspace(-200, 200, 51).to(device),
            rainbow=True,
            device=device,
            hidden_size=[128, 128],
            mlp_activation="ReLU",
            mlp_output_activation="ReLU",
        )
        NET_CONFIG = None
        critic = [
            EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                device=device,
                hidden_size=[64, 64],
                mlp_activation="ReLU",
            )
            for _ in range(2)
        ]
    else:
        actor = None
        critic = None

    agent_pop = create_population(
        algo=INIT_HP["ALGO"],
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        actor_network=actor,
        critic_network=critic,
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
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        learning_delay=INIT_HP["LEARNING_DELAY"],
        eps_start=INIT_HP["EPS_START"] if "EPS_START" in INIT_HP else 1.0,
        eps_end=INIT_HP["EPS_END"] if "EPS_END" in INIT_HP else 0.01,
        eps_decay=INIT_HP["EPS_DECAY"] if "EPS_DECAY" in INIT_HP else 0.999,
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
    with open("configs/training/dqn.yaml") as file:
        config = yaml.safe_load(file)
    INIT_HP = config["INIT_HP"]
    MUTATION_PARAMS = config["MUTATION_PARAMS"]
    NET_CONFIG = config["NET_CONFIG"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net=False)
