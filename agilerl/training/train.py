import numpy as np
from tqdm import trange
import wandb
from datetime import datetime


def train(
        env,
        env_name,
        algo,
        pop,
        memory,
        swap_channels=False,
        n_episodes=2000,
        max_steps=500,
        evo_epochs=5,
        evo_loop=1,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=0.995,
        target=200.,
        tournament=None,
        mutation=None,
        checkpoint=None,
        checkpoint_path=None,
        wb=False,
        device='cpu'):
    """The general training function. Returns trained population of agents and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: List[object]
    :param memory: Experience Replay Buffer
    :type memory: object
    :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
    :type swap_channels: bool, optional
    :param n_episodes: Maximum number of training episodes, defaults to 2000
    :type n_episodes: int, optional
    :param max_steps: Maximum number of steps in environment per episode, defaults to 500
    :type max_steps: int, optional
    :param evo_epochs: Evolution frequency (episodes), defaults to 5
    :type evo_epochs: int, optional
    :param evo_loop: Number of evaluation episodes, defaults to 1
    :type evo_loop: int, optional
    :param eps_start: Maximum exploration - initial epsilon value, defaults to 1.0
    :type eps_start: float, optional
    :param eps_end: Minimum exploration - final epsilon value, defaults to 0.1
    :type eps_end: float, optional
    :param eps_decay: Epsilon decay per episode, defaults to 0.995
    :type eps_decay: float, optional
    :param target: Target score for early stopping, defaults to 200.
    :type target: float, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (episodes), defaults to None
    :type checkpoint: int, optional
    :param checkpoint_path: Location to save checkpoint, defaults to None
    :type checkpoint_path: str, optional
    :param wb: Weights & Biases tracking, defaults to False
    :type wb: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """
    if wb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="AgileRL",
            name="{}-EvoHPO-{}-{}".format(env_name, algo,
                                          datetime.now().strftime("%m%d%Y%H%M%S")),
            # track hyperparameters and run metadata
            config={
                "algo": "Evo HPO {}".format(algo),
                "env": env_name,
            }
        )

    save_path = checkpoint_path.split('.pt')[0] if checkpoint_path is not None else "{}-EvoHPO-{}-{}".format(
        env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S"))

    epsilon = eps_start

    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)

    pop_fitnesses = []
    total_steps = 0

    # RL training loop
    for idx_epi in pbar:
        for agent in pop:   # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            score = 0

            for idx_step in range(max_steps):
                if swap_channels:
                    state = np.moveaxis(state, [3], [1])
                # Get next action from agent
                action = agent.getAction(state, epsilon)
                next_state, reward, done, _, _ = env.step(
                    action)   # Act in environment

                # Save experience to replay buffer
                if swap_channels:
                    memory.save2memoryVectEnvs(
                        state, action, reward, np.moveaxis(next_state, [3], [1]), done)
                else:
                    memory.save2memoryVectEnvs(
                        state, action, reward, next_state, done)

                # Learn according to learning frequency
                if memory.counter % agent.learn_step == 0 and len(
                        memory) >= agent.batch_size:
                    experiences = memory.sample(
                        agent.batch_size)   # Sample replay buffer
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)

                state = next_state
                score += reward

            agent.scores.append(score)

            agent.steps[-1] += max_steps
            total_steps += max_steps

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

        # Now evolve if necessary
        if (idx_epi + 1) % evo_epochs == 0:

            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=swap_channels,
                    max_steps=max_steps,
                    loop=evo_loop) for agent in pop]
            pop_fitnesses.append(fitnesses)

            mean_scores = np.mean([agent.scores[-20:]
                                  for agent in pop], axis=1)

            if wb:
                wandb.log({"global_step": total_steps,
                           "eval/mean_score": np.mean(mean_scores),
                           "eval/mean_reward": np.mean(fitnesses),
                           "eval/best_fitness": np.max(fitnesses)})

            # Update step counter
            for agent in pop:
                agent.steps.append(agent.steps[-1])

            pbar.set_postfix_str(f'Fitness: {["%.2f"%fitness for fitness in fitnesses]}, 100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}, 100 score avgs: {["%.2f"%np.mean(agent.scores[-100:]) for agent in pop]}, agents: {[agent.index for agent in pop]}, steps: {[agent.steps[-1] for agent in pop]}, mutations: {[agent.mut for agent in pop]}')
            pbar.update(0)

            # Early stop if consistently reaches target
            if np.all(np.greater([np.mean(agent.fitness[-100:])
                      for agent in pop], target)) and idx_epi >= 100:
                if wb:
                    wandb.finish()
                return pop, pop_fitnesses

            if tournament and mutation is not None:
                # Tournament selection and population mutation
                elite, pop = tournament.select(pop)
                pop = mutation.mutation(pop)

        # Save model checkpoint
        if checkpoint is not None:
            if (idx_epi + 1) % checkpoint == 0:
                for i, agent in enumerate(pop):
                    agent.saveCheckpoint(f'{save_path}_{i}_{idx_epi+1}.pt')

    if wb:
        wandb.finish()
    return pop, pop_fitnesses
