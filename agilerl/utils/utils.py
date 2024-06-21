import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from agilerl.algorithms.cqn import CQN
from agilerl.algorithms.ddpg import DDPG
from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.algorithms.maddpg import MADDPG
from agilerl.algorithms.matd3 import MATD3
from agilerl.algorithms.neural_ts_bandit import NeuralTS
from agilerl.algorithms.neural_ucb_bandit import NeuralUCB
from agilerl.algorithms.ppo import PPO
from agilerl.algorithms.td3 import TD3
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper


def make_vect_envs(env_name, num_envs=1):
    """Returns async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: gym.make(env_name) for i in range(num_envs)]
    )


def make_multi_agent_vect_envs(env, num_envs=1):
    """Returns async-vectorized PettingZoo parallel environments.

    :param env: PettingZoo parallel environment object
    :type env: pettingzoo.utils.env.ParallelEnv
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return PettingZooVectorizationParallelWrapper(env, num_envs)


def make_skill_vect_envs(env_name, skill, num_envs=1):
    """Returns async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param skill: Skill wrapper to apply to environment
    :type skill: agilerl.wrappers.learning.Skill
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: skill(gym.make(env_name)) for i in range(num_envs)]
    )


def create_population(
    algo,
    state_dim,
    action_dim,
    one_hot,
    net_config,
    INIT_HP,
    actor_network=None,
    critic_network=None,
    population_size=1,
    num_envs=1,
    device="cpu",
    accelerator=None,
):
    """Returns population of identical agents.

    :param algo: RL algorithm
    :type algo: str
    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding
    :type one_hot: bool
    :param INIT_HP: Initial hyperparameters
    :type INIT_HP: dict
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
    :param critic_network: Custom critic network, defaults to None
    :type critic_network: nn.Module, optional
    :param population_size: Number of agents in population, defaults to 1
    :type population_size: int, optional
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    """
    population = []

    if algo == "DQN":
        for idx in range(population_size):
            agent = DQN(
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                index=idx,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                double=INIT_HP["DOUBLE"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "Rainbow DQN":
        for idx in range(population_size):
            agent = RainbowDQN(
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                index=idx,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                beta=INIT_HP["BETA"],
                prior_eps=INIT_HP["PRIOR_EPS"],
                num_atoms=INIT_HP["NUM_ATOMS"],
                v_min=INIT_HP["V_MIN"],
                v_max=INIT_HP["V_MAX"],
                n_step=INIT_HP["N_STEP"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "DDPG":
        for idx in range(population_size):
            agent = DDPG(
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                max_action=INIT_HP["MAX_ACTION"],
                min_action=INIT_HP["MIN_ACTION"],
                O_U_noise=INIT_HP["O_U_NOISE"],
                expl_noise=INIT_HP["EXPL_NOISE"],
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP["MEAN_NOISE"],
                theta=INIT_HP["THETA"],
                dt=INIT_HP["DT"],
                index=idx,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr_actor=INIT_HP["LR_ACTOR"],
                lr_critic=INIT_HP["LR_CRITIC"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                policy_freq=INIT_HP["POLICY_FREQ"],
                actor_network=actor_network,
                critic_network=critic_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "PPO":
        for idx in range(population_size):
            agent = PPO(
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                discrete_actions=INIT_HP["DISCRETE_ACTIONS"],
                index=idx,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                gae_lambda=INIT_HP["GAE_LAMBDA"],
                action_std_init=INIT_HP["ACTION_STD_INIT"],
                clip_coef=INIT_HP["CLIP_COEF"],
                ent_coef=INIT_HP["ENT_COEF"],
                vf_coef=INIT_HP["VF_COEF"],
                max_grad_norm=INIT_HP["MAX_GRAD_NORM"],
                target_kl=INIT_HP["TARGET_KL"],
                update_epochs=INIT_HP["UPDATE_EPOCHS"],
                actor_network=actor_network,
                critic_network=critic_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "CQN":
        for idx in range(population_size):
            agent = CQN(
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                index=idx,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                double=INIT_HP["DOUBLE"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "TD3":
        for idx in range(population_size):
            agent = TD3(
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                max_action=INIT_HP["MAX_ACTION"],
                min_action=INIT_HP["MIN_ACTION"],
                O_U_noise=INIT_HP["O_U_NOISE"],
                expl_noise=INIT_HP["EXPL_NOISE"],
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP["MEAN_NOISE"],
                theta=INIT_HP["THETA"],
                dt=INIT_HP["DT"],
                index=idx,
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr_actor=INIT_HP["LR_ACTOR"],
                lr_critic=INIT_HP["LR_CRITIC"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                policy_freq=INIT_HP["POLICY_FREQ"],
                actor_network=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "MADDPG":
        for idx in range(population_size):
            agent = MADDPG(
                state_dims=state_dim,
                action_dims=action_dim,
                one_hot=one_hot,
                n_agents=INIT_HP["N_AGENTS"],
                agent_ids=INIT_HP["AGENT_IDS"],
                O_U_noise=INIT_HP["O_U_NOISE"],
                expl_noise=INIT_HP["EXPL_NOISE"],
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP["MEAN_NOISE"],
                theta=INIT_HP["THETA"],
                dt=INIT_HP["DT"],
                index=idx,
                max_action=INIT_HP["MAX_ACTION"],
                min_action=INIT_HP["MIN_ACTION"],
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr_actor=INIT_HP["LR_ACTOR"],
                lr_critic=INIT_HP["LR_CRITIC"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                discrete_actions=INIT_HP["DISCRETE_ACTIONS"],
                actor_networks=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "MATD3":
        for idx in range(population_size):
            agent = MATD3(
                state_dims=state_dim,
                action_dims=action_dim,
                one_hot=one_hot,
                n_agents=INIT_HP["N_AGENTS"],
                agent_ids=INIT_HP["AGENT_IDS"],
                O_U_noise=INIT_HP["O_U_NOISE"],
                expl_noise=INIT_HP["EXPL_NOISE"],
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP["MEAN_NOISE"],
                theta=INIT_HP["THETA"],
                dt=INIT_HP["DT"],
                index=idx,
                max_action=INIT_HP["MAX_ACTION"],
                min_action=INIT_HP["MIN_ACTION"],
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr_actor=INIT_HP["LR_ACTOR"],
                lr_critic=INIT_HP["LR_CRITIC"],
                policy_freq=INIT_HP["POLICY_FREQ"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                discrete_actions=INIT_HP["DISCRETE_ACTIONS"],
                actor_networks=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "NeuralUCB":
        for idx in range(population_size):
            agent = NeuralUCB(
                state_dim=state_dim,
                action_dim=action_dim,
                index=idx,
                net_config=net_config,
                gamma=INIT_HP["GAMMA"],
                lamb=INIT_HP["LAMBDA"],
                reg=INIT_HP["REG"],
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    elif algo == "NeuralTS":
        for idx in range(population_size):
            agent = NeuralTS(
                state_dim=state_dim,
                action_dim=action_dim,
                index=idx,
                net_config=net_config,
                gamma=INIT_HP["GAMMA"],
                lamb=INIT_HP["LAMBDA"],
                reg=INIT_HP["REG"],
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

    return population


def calculate_vectorized_scores(
    rewards, terminations, include_unterminated=False, only_first_episode=True
):
    episode_rewards = []
    num_envs, _ = rewards.shape

    for env_index in range(num_envs):
        # Find the indices where episodes terminate for the current environment
        termination_indices = np.where(terminations[env_index] == 1)[0]

        # If no terminations, sum the entire reward array for this environment
        if len(termination_indices) == 0:
            episode_reward = np.sum(rewards[env_index])
            episode_rewards.append(episode_reward)
            continue  # Skip to the next environment

        # Initialize the starting index for segmenting
        start_index = 0

        for termination_index in termination_indices:
            # Sum the rewards for the current episode
            episode_reward = np.sum(
                rewards[env_index, start_index : termination_index + 1]
            )

            # Store the episode reward
            episode_rewards.append(episode_reward)

            # If only the first episode is required, break after processing it
            if only_first_episode:
                break

            # Update the starting index for segmenting
            start_index = termination_index + 1

        # If include_unterminated is True, sum the rewards from the last termination index to the end
        if (
            not only_first_episode
            and include_unterminated
            and start_index < len(rewards[env_index])
        ):
            episode_reward = np.sum(rewards[env_index, start_index:])
            episode_rewards.append(episode_reward)

    return episode_rewards


def print_hyperparams(pop):
    """Prints current hyperparameters of agents in a population and their fitnesses.

    :param pop: Population of agents
    :type pop: list[object]
    """

    for agent in pop:
        print(
            "Agent ID: {}    Mean 5 Fitness: {:.2f}    Attributes: {}".format(
                agent.index, np.mean(agent.fitness[-5:]), agent.inspect_attributes()
            )
        )


def plot_population_score(pop):
    """Plots the fitness scores of agents in a population.

    :param pop: Population of agents
    :type pop: list[object]
    """
    plt.figure()
    for agent in pop:
        scores = agent.fitness
        steps = agent.steps[:-1]
        plt.plot(steps, scores)
    plt.title("Score History - Mutations")
    plt.xlabel("Steps")
    plt.ylim(bottom=-400)
    plt.show()
