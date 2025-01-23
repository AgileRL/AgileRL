import gymnasium as gym
import supersuit as ss
import torch
import yaml
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from benchmarking.networks import (
    BasicNetActor,
    BasicNetCritic,
    ClipReward,
    SimpleCNNActor,
    SimpleCNNCritic,
    SoftmaxActor,
)
from pettingzoo.atari import pong_v3
from pettingzoo.mpe import simple_speaker_listener_v4

from agilerl.algorithms.core.base import RLAlgorithm, MultiAgentAlgorithm
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.modules.mlp import EvolvableMLP
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.training.train_off_policy import train_off_policy
from agilerl.training.train_on_policy import train_on_policy
from agilerl.wrappers.make_evolvable import MakeEvolvable
from agilerl.utils.algo_utils import observation_space_channels_to_first
from agilerl.utils.utils import (
    create_population,
    make_vect_envs,
    print_hyperparams
)

def main(INIT_HP, MUTATION_PARAMS, atari, multi=False, NET_CONFIG=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not multi:
        ####
        if not atari:
            env = make_vect_envs(INIT_HP["ENV_NAME"], num_envs=INIT_HP["NUM_ENVS"])
        else:
            env = gym.make(INIT_HP["ENV_NAME_ATARI"])
            env = AtariPreprocessing(env)
            env = ClipReward(env)
            env = ss.frame_stack_v1(env, 4)

        observation_space = env.single_observation_space
        action_space = env.single_action_space
        if INIT_HP["CHANNELS_LAST"]:
            observation_space = observation_space_channels_to_first(observation_space)

        if NET_CONFIG is not None:
            actor = None
            critic = None
        else:
            NET_CONFIG = None

            action_dims = RLAlgorithm.get_action_dim(action_space)
            state_dims = RLAlgorithm.get_state_dim(observation_space)
            if atari:
                # DQN
                network_actor = SimpleCNNActor(action_dims)
                actor = MakeEvolvable(
                    network_actor,
                    input_tensor=torch.ones(4, 84, 84).unsqueeze(0),
                    device=device,
                )
                critic = None
                if INIT_HP["ALGO"] in ["TD3", "DDPG"]:
                    pass
                elif INIT_HP["ALGO"] == "PPO":
                    network_critic = SimpleCNNCritic(1)
                    critic = MakeEvolvable(
                        network_critic,
                        input_tensor=torch.ones(4, 84, 84).unsqueeze(0),
                        device=device,
                    )

            else:
                # DQN
                if INIT_HP["ALGO"] == "DQN":
                    # network_actor_dqn = BasicNetActorDQN(
                    #     state_dims[0], [64, 64], action_dims
                    # )
                    # actor = MakeEvolvable(
                    #     network_actor_dqn,
                    #     input_tensor=torch.ones(state_dims[0]),
                    #     device=device,
                    # )

                    actor = EvolvableMLP(
                        num_inputs=state_dims[0],
                        num_outputs=action_dims,
                        device=device,
                        hidden_size=[64, 64],
                        mlp_activation="ReLU",
                    )

                    critic = None
                if INIT_HP["ALGO"] == "DDPG":
                    network_actor_ddpg = BasicNetActor(
                        state_dims[0], [64, 64], action_dims
                    )
                    actor = MakeEvolvable(
                        network_actor_ddpg,
                        input_tensor=torch.ones(state_dims[0]),
                        device=device,
                    )
                    network_critic = BasicNetCritic(
                        state_dims[0] + action_dims, [64, 64], 1
                    )
                    critic = MakeEvolvable(
                        network_critic,
                        torch.ones(state_dims[0] + action_dims),
                        device=device,
                    )

                    actor = EvolvableMLP(
                        num_inputs=state_dims[0],
                        num_outputs=action_dims,
                        device=device,
                        hidden_size=[64, 64],
                        mlp_activation="ReLU",
                        mlp_output_activation="Tanh",
                    )

                    critic = EvolvableMLP(
                        num_inputs=state_dims[0] + action_dims,
                        num_outputs=action_dims,
                        device=device,
                        hidden_size=[64, 64],
                        mlp_activation="ReLU",
                    )

                elif INIT_HP["ALGO"] == "TD3":
                    network_actor_td3 = BasicNetActor(
                        state_dims[0], [64, 64], action_dims
                    )
                    actor = MakeEvolvable(
                        network_actor_td3,
                        input_tensor=torch.ones(state_dims[0]),
                        device=device,
                    )
                    network_critic = BasicNetCritic(
                        state_dims[0] + action_dims, [64, 64], 1
                    )
                    critic_1 = MakeEvolvable(
                        network_critic,
                        torch.ones(state_dims[0] + action_dims),
                        device=device,
                    )
                    critic_2 = MakeEvolvable(
                        network_critic,
                        torch.ones(state_dims[0] + action_dims),
                        device=device,
                    )
                    critic = [critic_1, critic_2]
                elif INIT_HP["ALGO"] == "PPO":
                    network_actor_dqn = SoftmaxActor(
                        state_dims[0], [64, 64], action_dims
                    )
                    actor = MakeEvolvable(
                        network_actor_dqn,
                        input_tensor=torch.ones(state_dims[0]),
                        device=device,
                    )
                    network_critic = BasicNetCritic(state_dims[0], [32, 32], 1)
                    critic = MakeEvolvable(
                        network_critic, torch.ones(state_dims[0]), device=device
                    )

        if INIT_HP["ALGO"] != "PPO":
            field_names = ["state", "action", "reward", "next_state", "done"]
            memory = ReplayBuffer(
                INIT_HP["MEMORY_SIZE"],
                field_names=field_names,
                device=device,
            )

    else:
        ####
        if atari:
            env = pong_v3.parallel_env(num_players=2)

            # Environment processing for image based observations
            env = ss.frame_skip_v0(env, 4)
            env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
            env = ss.color_reduction_v0(env, mode="B")
            env = ss.resize_v1(env, x_size=84, y_size=84)
            env = ss.frame_stack_v1(env, 4)
        else:
            env = simple_speaker_listener_v4.parallel_env(
                continuous_actions=True, max_cycles=25
            )

        env.reset()

        # Configure the multi-agent algo input arguments
        observation_space = [env.observation_space(agent) for agent in env.agents]
        action_space = [env.action_space(agent) for agent in env.agents]
        if INIT_HP["CHANNELS_LAST"]:
            observation_space = [observation_space_channels_to_first(obs) for obs in observation_space]

        INIT_HP["N_AGENTS"] = env.num_agents
        INIT_HP["AGENT_IDS"] = [agent_id for agent_id in env.agents]
        if not atari:
            # MLPs
            state_dims = MultiAgentAlgorithm.get_state_dim(observation_space)
            action_dims = MultiAgentAlgorithm.get_action_dim(action_space)
            total_state_dims = sum(state_dim[0] for state_dim in state_dims)
            total_actions = sum(action_dims)
            actor = [
                MakeEvolvable(
                    SoftmaxActor(state_dim[0], [64, 64], action_dim),
                    input_tensor=torch.ones(state_dim[0]),
                    device=device,
                )
                for action_dim, state_dim in zip(action_dims, state_dims)
            ]
            if INIT_HP["ALGO"] == "MADDPG":
                critic = [
                    MakeEvolvable(
                        BasicNetCritic(total_state_dims + total_actions, [64, 64], 1),
                        input_tensor=torch.ones(total_state_dims + total_actions),
                        device=device,
                    )
                    for _ in range(INIT_HP["N_AGENTS"])
                ]
            elif INIT_HP["ALGO"] == "MATD3":
                critics_1 = [
                    MakeEvolvable(
                        BasicNetCritic(total_state_dims + total_actions, [64, 64], 1),
                        input_tensor=torch.ones(total_state_dims + total_actions),
                        device=device,
                    )
                    for _ in range(INIT_HP["N_AGENTS"])
                ]
                critics_2 = [
                    MakeEvolvable(
                        BasicNetCritic(total_state_dims + total_actions, [64, 64], 1),
                        input_tensor=torch.ones(total_state_dims + total_actions),
                        device=device,
                    )
                    for _ in range(INIT_HP["N_AGENTS"])
                ]
                critic = [critics_1, critics_2]

        field_names = ["state", "action", "reward", "next_state", "done"]
        memory = MultiAgentReplayBuffer(
            INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
            agent_ids=INIT_HP["AGENT_IDS"],
            device=device,
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
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    agent_pop = create_population(
        INIT_HP["ALGO"],
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        actor_network=actor,
        critic_network=critic,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=INIT_HP["NUM_ENVS"],
        device=device,
    )

    if INIT_HP["ALGO"] in ["MATD3", "MADDPG"]:
        trained_pop, pop_fitnesses = train_multi_agent(
            env,
            INIT_HP["ENV_NAME"],
            INIT_HP["ALGO"],
            agent_pop,
            memory=memory,
            INIT_HP=INIT_HP,
            MUT_P=MUTATION_PARAMS,
            net_config=NET_CONFIG,
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
        )
    elif INIT_HP["ALGO"] == "PPO":
        trained_pop, pop_fitnesses = train_on_policy(
            env,
            INIT_HP["ENV_NAME"],
            INIT_HP["ALGO"],
            agent_pop,
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
    elif INIT_HP["ALGO"] in ["DDPG", "DQN", "TD3"]:
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
    dqn = False
    ppo = False
    ddpg = True
    td3 = False
    maddpg = False
    matd3 = False
    standard = True
    atari = False

    if dqn:
        with open("configs/training/dqn.yaml") as file:
            dqn_config = yaml.safe_load(file)
        INIT_HP = dqn_config["INIT_HP"]
        MUTATION_PARAMS = dqn_config["MUTATION_PARAMS"]
        # net_config_mlp = dqn_config["MLP"]
        # net_config_cnn = dqn_config["CNN"]
        if standard:
            print("-" * 20, "DQN Lunar Lander using make evolvable", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=None)
            # print("-" * 20, "DQN Lunar Lander using net_config", "-" * 20)
            # main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=net_config_mlp)
        # if atari:
        #     print("-" * 20, "DQN Atari using make evolvable", "-" * 20)
        #     main(INIT_HP, MUTATION_PARAMS, atari=True, NET_CONFIG=None)
        #     print("-" * 20, "DQN Atari using net_config", "-" * 20)
        #     main(INIT_HP, MUTATION_PARAMS, atari=True, NET_CONFIG=net_config_cnn)

    if ppo:
        with open("configs/training/ppo.yaml") as file:
            ppo_config = yaml.safe_load(file)
        INIT_HP = ppo_config["INIT_HP"]
        MUTATION_PARAMS = ppo_config["MUTATION_PARAMS"]
        net_config_mlp = ppo_config["MLP"]
        net_config_cnn = ppo_config["CNN"]
        if standard:
            print("-" * 20, "PPO Lunar Lander using make evolvable", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=None)
            print("-" * 20, "PPO Lunar Lander using net_config", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=net_config_mlp)
        if atari:
            print("-" * 20, "PPO Atari using make evolvable", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=True, NET_CONFIG=None)
            print("-" * 20, "PPO Atari using net_config", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=True, NET_CONFIG=net_config_cnn)

    if ddpg:
        with open("configs/training/ddpg.yaml") as file:
            ddpg_config = yaml.safe_load(file)
        INIT_HP = ddpg_config["INIT_HP"]
        MUTATION_PARAMS = ddpg_config["MUTATION_PARAMS"]
        # net_config_mlp = ddpg_config["MLP"]
        if standard:
            print("-" * 20, "DDPG Lunar Lander using make evolvable", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, multi=False, NET_CONFIG=None)
            # print("-" * 20, "DDPG Lunar Lander using net_config", "-" * 20)
            # main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=net_config_mlp)

    if td3:
        with open("configs/training/td3.yaml") as file:
            td3_config = yaml.safe_load(file)
        INIT_HP = td3_config["INIT_HP"]
        MUTATION_PARAMS = td3_config["MUTATION_PARAMS"]
        # net_config_mlp = td3_config["MLP"]
        if standard:
            print("-" * 20, "TD3 Lunar Lander using make evolvable", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=None)
            print("-" * 20, "TD3 Lunar Lander using net_config", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=net_config_mlp)

    if maddpg:
        with open("configs/training/maddpg.yaml") as file:
            maddpg_config = yaml.safe_load(file)
        INIT_HP = maddpg_config["INIT_HP"]
        MUTATION_PARAMS = maddpg_config["MUTATION_PARAMS"]
        net_config_mlp = maddpg_config["MLP"]
        if standard:
            print(
                "-" * 20,
                "MADDPG simple speaker listener using make evolvable",
                "-" * 20,
            )
            main(INIT_HP, MUTATION_PARAMS, atari=False, multi=True, NET_CONFIG=None)
            print("-" * 20, "MADDPG simple speaker listener using net_config", "-" * 20)
            main(
                INIT_HP,
                MUTATION_PARAMS,
                atari=False,
                multi=True,
                NET_CONFIG=net_config_mlp,
            )
        if atari:
            print("-" * 20, "MADDPG Atari using make evolvable", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=True, multi=True, NET_CONFIG=None)
            print("-" * 20, "MADDPG Atari using net_config", "-" * 20)
            main(
                INIT_HP,
                MUTATION_PARAMS,
                atari=True,
                multi=True,
                NET_CONFIG=net_config_cnn,
            )

    if matd3:
        with open("configs/training/matd3.yaml") as file:
            matd3_config = yaml.safe_load(file)
        INIT_HP = matd3_config["INIT_HP"]
        MUTATION_PARAMS = matd3_config["MUTATION_PARAMS"]
        net_config_mlp = matd3_config["MLP"]
        if standard:
            print(
                "-" * 20, "MATD3 simple speaker listener using make evolvable", "-" * 20
            )
            main(INIT_HP, MUTATION_PARAMS, atari=False, multi=True, NET_CONFIG=None)
            print("-" * 20, "MATD3 simple speaker listener using net_config", "-" * 20)
            main(
                INIT_HP,
                MUTATION_PARAMS,
                atari=False,
                multi=True,
                NET_CONFIG=net_config_mlp,
            )
        if atari:
            print("-" * 20, "MATD3 Atari using make evolvable", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=True, multi=True, NET_CONFIG=None)
            print("-" * 20, "MATD3 Atari using net_config", "-" * 20)
            main(
                INIT_HP,
                MUTATION_PARAMS,
                atari=True,
                multi=True,
                NET_CONFIG=net_config_cnn,
            )
