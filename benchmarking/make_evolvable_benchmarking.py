import gymnasium as gym
import supersuit as ss
import torch
import yaml
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from networks import (
    BasicNetActor,
    BasicNetActorDQN,
    BasicNetCritic,
    ClipReward,
    SimpleCNNActor,
    SimpleCNNCritic,
    SoftmaxActor,
)
from pettingzoo.atari import pong_v3
from pettingzoo.mpe import simple_speaker_listener_v4

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.training.train_off_policy import train_off_policy
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import initialPopulation, makeVectEnvs, printHyperparams
from agilerl.wrappers.make_evolvable import MakeEvolvable

from agilerl.networks.evolvable_mlp import EvolvableMLP


def main(INIT_HP, MUTATION_PARAMS, atari, multi=False, NET_CONFIG=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not multi:
        ####
        if not atari:
            env = makeVectEnvs(INIT_HP["ENV_NAME"], num_envs=8)
            try:
                state_dims = env.single_observation_space.n
                one_hot = True
            except Exception:
                state_dims = env.single_observation_space.shape
                one_hot = False
            try:
                action_dims = env.single_action_space.n
            except Exception:
                action_dims = env.single_action_space.shape[0]
        else:
            env = gym.make(INIT_HP["ENV_NAME_ATARI"])
            env = AtariPreprocessing(env)
            env = ClipReward(env)
            env = ss.frame_stack_v1(env, 4)
            try:
                state_dims = env.observation_space.n
                one_hot = True
            except Exception:
                state_dims = env.observation_space.shape
                one_hot = False
            try:
                action_dims = env.action_space.n
            except Exception:
                action_dims = env.action_space.shape[0]

        if INIT_HP["CHANNELS_LAST"]:
            state_dims = (state_dims[2], state_dims[0], state_dims[1])

        if INIT_HP["ALGO"] == "TD3":
            max_action = float(env.single_action_space.high[0])
            INIT_HP["MAX_ACTION"] = max_action

        if INIT_HP["ALGO"] == "TD3":
            max_action = float(env.single_action_space.high[0])
            INIT_HP["MAX_ACTION"] = max_action

        if NET_CONFIG is not None:
            actor = None
            critic = None
        else:
            NET_CONFIG = None

            ####
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
                        mlp_activation="ReLU"
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
                action_dims,
                INIT_HP["MEMORY_SIZE"],
                field_names=field_names,
                device=device,
            )
        if NET_CONFIG is None:
            arch = actor.arch
        else:
            arch = NET_CONFIG["arch"]

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
        try:
            state_dims = [env.observation_space(agent).n for agent in env.agents]
            one_hot = True
        except Exception:
            state_dims = [env.observation_space(agent).shape for agent in env.agents]
            one_hot = False
        try:
            action_dims = [env.action_space(agent).n for agent in env.agents]
            INIT_HP["DISCRETE_ACTIONS"] = True
            INIT_HP["MAX_ACTION"] = None
            INIT_HP["MIN_ACTION"] = None
        except Exception:
            action_dims = [env.action_space(agent).shape[0] for agent in env.agents]
            INIT_HP["DISCRETE_ACTIONS"] = False
            INIT_HP["MAX_ACTION"] = [
                env.action_space(agent).high for agent in env.agents
            ]
            INIT_HP["MIN_ACTION"] = [
                env.action_space(agent).low for agent in env.agents
            ]

        if INIT_HP["CHANNELS_LAST"]:
            state_dims = [
                (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dims
            ]

        INIT_HP["N_AGENTS"] = env.num_agents
        INIT_HP["AGENT_IDS"] = [agent_id for agent_id in env.agents]

        ####
        if atari:
            pass

        else:
            # MLPs
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
        if NET_CONFIG is None:
            arch = actor[0].arch
        else:
            arch = NET_CONFIG["arch"]

    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVO_EPOCHS"],
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
        arch=arch,
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    agent_pop = initialPopulation(
        INIT_HP["ALGO"],
        state_dims,
        action_dims,
        one_hot,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        actor_network=actor,
        critic_network=critic,
        population_size=INIT_HP["POP_SIZE"],
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
            n_episodes=INIT_HP["EPISODES"],
            max_steps=500,
            evo_epochs=INIT_HP["EVO_EPOCHS"],
            evo_loop=1,
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
            n_episodes=INIT_HP["EPISODES"],
            max_steps=500,
            evo_epochs=INIT_HP["EVO_EPOCHS"],
            evo_loop=1,
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
            n_episodes=INIT_HP["EPISODES"],
            max_steps=500,
            evo_epochs=INIT_HP["EVO_EPOCHS"],
            evo_loop=1,
            target=INIT_HP["TARGET_SCORE"],
            tournament=tournament,
            mutation=mutations,
            wb=INIT_HP["WANDB"],
        )

    printHyperparams(trained_pop)
    # plotPopulationScore(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()

    env.close()


if __name__ == "__main__":
    dqn = True
    ppo = False
    ddpg = False
    td3 = False
    maddpg = False
    matd3 = False
    standard = True
    atari = False

    if dqn:
        with open("../configs/training/dqn.yaml") as file:
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
        with open("../configs/training/ppo.yaml") as file:
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
        with open("../configs/training/ddpg.yaml") as file:
            ddpg_config = yaml.safe_load(file)
        INIT_HP = ddpg_config["INIT_HP"]
        MUTATION_PARAMS = ddpg_config["MUTATION_PARAMS"]
        net_config_mlp = ddpg_config["MLP"]
        if standard:
            print("-" * 20, "DDPG Lunar Lander using make evolvable", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=None)
            print("-" * 20, "DDPG Lunar Lander using net_config", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=net_config_mlp)

    if td3:
        with open("../configs/training/td3.yaml") as file:
            td3_config = yaml.safe_load(file)
        INIT_HP = td3_config["INIT_HP"]
        MUTATION_PARAMS = td3_config["MUTATION_PARAMS"]
        net_config_mlp = td3_config["MLP"]
        if standard:
            print("-" * 20, "TD3 Lunar Lander using make evolvable", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=None)
            print("-" * 20, "TD3 Lunar Lander using net_config", "-" * 20)
            main(INIT_HP, MUTATION_PARAMS, atari=False, NET_CONFIG=net_config_mlp)

    if maddpg:
        with open("../configs/training/maddpg.yaml") as file:
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
        with open("../configs/training/matd3.yaml") as file:
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
