import torch
import yaml
from ucimlrepo import fetch_ucirepo

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.training.train_bandits import train_bandits
from agilerl.utils.utils import initialPopulation, printHyperparams
from agilerl.wrappers.learning import BanditEnv

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Bandit Benchmarking =====")
    print(f"DEVICE: {device}")
    print(INIT_HP)
    print(MUTATION_PARAMS)
    print(NET_CONFIG)

    # Fetch data
    dataset = fetch_ucirepo(id=INIT_HP["UCI_REPO_ID"])
    features = dataset.data.features
    targets = dataset.data.targets

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_dim = env.arms

    if INIT_HP["CHANNELS_LAST"]:
        context_dim = (context_dim[2], context_dim[0], context_dim[1])

    field_names = ["context", "action"]
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
        actor = EvolvableMLP(
            num_inputs=context_dim[0],
            num_outputs=1,
            layer_norm=False,
            device=device,
            arch="mlp",
            hidden_size=[128],
        )
        NET_CONFIG = None
    else:
        actor = None

    agent_pop = initialPopulation(
        algo=INIT_HP["ALGO"],
        state_dim=context_dim,
        action_dim=action_dim,
        one_hot=None,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        actor_network=actor,
        population_size=INIT_HP["POP_SIZE"],
        device=device,
    )

    trained_pop, pop_fitnesses = train_bandits(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        max_steps=INIT_HP["MAX_STEPS"],
        episode_steps=INIT_HP["EPISODE_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
    )

    printHyperparams(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with open("configs/training/neural_ucb.yaml") as file:
        bandit_config = yaml.safe_load(file)
    INIT_HP = bandit_config["INIT_HP"]
    MUTATION_PARAMS = bandit_config["MUTATION_PARAMS"]
    NET_CONFIG = bandit_config["NET_CONFIG"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net=True)
