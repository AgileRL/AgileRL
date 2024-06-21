import os
import warnings
from datetime import datetime

import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import trange

from agilerl.components.replay_data import ReplayDataset
from agilerl.components.sampler import Sampler
from agilerl.utils.utils import calculate_vectorized_scores


def train_off_policy(
    env,
    env_name,
    algo,
    pop,
    memory,
    INIT_HP=None,
    MUT_P=None,
    swap_channels=False,
    n_episodes=2000,
    max_steps=500,
    evo_epochs=5,
    evo_loop=1,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay=0.995,
    target=None,
    n_step=False,
    per=False,
    noisy=False,
    n_step_memory=None,
    tournament=None,
    mutation=None,
    checkpoint=None,
    checkpoint_path=None,
    save_elite=False,
    elite_path=None,
    wb=False,
    verbose=True,
    accelerator=None,
    wandb_api_key=None,
):
    """The general online RL training function. Returns trained population of agents
    and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[object]
    :param memory: Experience Replay Buffer
    :type memory: object
    :param INIT_HP: Dictionary containing initial hyperparameters, defaults to None
    :type INIT_HP: dict, optional
    :param MUT_P: Dictionary containing mutation parameters, defaults to None
    :type MUT_P: dict, optional
    :param swap_channels: Swap image channels dimension from last to first
        [H, W, C] -> [C, H, W], defaults to False
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
    :param target: Target score for early stopping, defaults to None
    :type target: float, optional
    :param n_step: Use multi-step experience replay buffer, defaults to False
    :type n_step: bool, optional
    :param per: Using prioritized experience replay buffer, defaults to False
    :type per: bool, optional
    :param noisy: Using noisy network exploration, defaults to False
    :type noisy: bool, optional
    :param memory: Multi-step Experience Replay Buffer to be used alongside Prioritized
        ERB, defaults to None
    :type memory: object, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (episodes), defaults to None
    :type checkpoint: int, optional
    :param checkpoint_path: Location to save checkpoint, defaults to None
    :type checkpoint_path: str, optional
    :param save_elite: Boolean flag indicating whether to save elite member at the end
        of training, defaults to False
    :type save_elite: bool, optional
    :param elite_path: Location to save elite agent, defaults to None
    :type elite_path: str, optional
    :param wb: Weights & Biases tracking, defaults to False
    :type wb: bool, optional
    :param verbose: Display training stats, defaults to True
    :type verbose: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wandb_api_key: API key for Weights & Biases, defaults to None
    :type wandb_api_key: str, optional
    """
    assert isinstance(
        algo, str
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(n_episodes, int), "Number of episodes must be an integer."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_epochs, int), "Evolution frequency must be an integer."
    assert isinstance(eps_start, float), "Starting epsilon must be a float."
    assert isinstance(eps_end, float), "Final value of epsilone must be a float."
    assert isinstance(eps_decay, float), "Epsilon decay rate must be a float."
    if target is not None:
        assert isinstance(
            target, (float, int)
        ), "Target score must be a float or an integer."
    assert isinstance(n_step, bool), "'n_step' must be a boolean."
    assert isinstance(per, bool), "'per' must be a boolean."
    if checkpoint is not None:
        assert isinstance(checkpoint, int), "Checkpoint must be an integer."
    assert isinstance(
        wb, bool
    ), "'wb' must be a boolean flag, indicating whether to record run with W&B"
    assert isinstance(verbose, bool), "Verbose must be a boolean."
    if save_elite is False and elite_path is not None:
        warnings.warn(
            "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
        )
    if checkpoint is None and checkpoint_path is not None:
        warnings.warn(
            "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
        )

    if wb:
        if not hasattr(wandb, "api"):
            if wandb_api_key is not None:
                wandb.login(key=wandb_api_key)
            else:
                warnings.warn("Must login to wandb with API key.")

        config_dict = {}
        if INIT_HP is not None:
            config_dict.update(INIT_HP)
        if MUT_P is not None:
            config_dict.update(MUT_P)

        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="AgileRL",
                    name="{}-EvoHPO-{}-{}".format(
                        env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
                    ),
                    # track hyperparameters and run metadata
                    config=config_dict,
                )
            accelerator.wait_for_everyone()
        else:
            wandb.init(
                # set the wandb project where this run will be logged
                project="AgileRL",
                name="{}-EvoHPO-{}-{}".format(
                    env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
                ),
                # track hyperparameters and run metadata
                config=config_dict,
            )

    if accelerator is not None:
        accel_temp_models_path = f"models/{env_name}"
        if accelerator.is_main_process:
            if not os.path.exists(accel_temp_models_path):
                os.makedirs(accel_temp_models_path)

    # Detect if environment is vectorised
    if hasattr(env, "num_envs"):
        is_vectorised = True
    else:
        is_vectorised = False

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
        )
    )

    if accelerator is not None:
        # Create dataloader from replay buffer
        replay_dataset = ReplayDataset(memory, pop[0].batch_size)
        replay_dataloader = DataLoader(replay_dataset, batch_size=None)
        replay_dataloader = accelerator.prepare(replay_dataloader)
        sampler = Sampler(
            distributed=True, dataset=replay_dataset, dataloader=replay_dataloader
        )
    else:
        sampler = Sampler(distributed=False, per=per, memory=memory)
        if n_step_memory is not None:
            n_step_sampler = Sampler(
                distributed=False, n_step=True, memory=n_step_memory
            )

    epsilon = eps_start

    if accelerator is not None:
        print(f"\nDistributed training on {accelerator.device}...")
    else:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if accelerator is not None:
        pbar = trange(
            n_episodes,
            unit="ep",
            bar_format=bar_format,
            ascii=True,
            disable=not accelerator.is_local_main_process,
        )
    else:
        pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None

    # Pre-training mutation
    if accelerator is None:
        if mutation is not None:
            pop = mutation.mutation(pop, pre_training_mut=True)

    # RL training loop
    for idx_epi in pbar:
        if accelerator is not None:
            accelerator.wait_for_everyone()
        for agent_idx, agent in enumerate(pop):  # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            rewards, terminations, truncs, losses = [], [], [], []
            score = 0

            if algo in ["DQN", "Rainbow DQN"]:
                train_actions_hist = [0] * agent.action_dim

            for idx_step in range(max_steps):
                if swap_channels:
                    state = np.moveaxis(state, [-1], [-3])
                # Get next action from agent
                if algo in ["DQN"]:
                    action = agent.get_action(state, epsilon)
                else:
                    action = agent.get_action(state)

                if algo in ["DQN", "Rainbow DQN"]:
                    for a in action:
                        if not isinstance(a, int):
                            a = int(a)
                        train_actions_hist[a] += 1

                if not is_vectorised:
                    action = action[0]
                next_state, reward, done, trunc, _ = env.step(
                    action
                )  # Act in environment

                # Save experience to replay buffer
                if n_step_memory is not None:
                    if swap_channels:
                        one_step_transition = n_step_memory.save_to_memory_vect_envs(
                            state,
                            action,
                            reward,
                            np.moveaxis(next_state, [-1], [-3]),
                            done,
                        )
                    else:
                        one_step_transition = n_step_memory.save_to_memory_vect_envs(
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                        )
                    if one_step_transition:
                        memory.save_to_memory_vect_envs(*one_step_transition)
                else:
                    if swap_channels:
                        memory.save_to_memory(
                            state,
                            action,
                            reward,
                            np.moveaxis(next_state, [-1], [-3]),
                            done,
                            is_vectorised=is_vectorised,
                        )
                    else:
                        memory.save_to_memory(
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                            is_vectorised=is_vectorised,
                        )

                if per:
                    # fraction = min((idx_step + 1)/ max_steps, 1.0)
                    fraction = min(
                        (total_steps + idx_step + 1) / (max_steps * n_episodes), 1.0
                    )  ####
                    agent.beta += fraction * (1.0 - agent.beta)

                # Learn according to learning frequency
                if (
                    memory.counter % agent.learn_step == 0
                    and len(memory) >= agent.batch_size
                ):
                    # Sample replay buffer
                    # Learn according to agent's RL algorithm
                    if per:
                        experiences = sampler.sample(agent.batch_size, agent.beta)
                        if n_step_memory is not None:
                            n_step_experiences = n_step_sampler.sample(experiences[6])
                            experiences += n_step_experiences
                        loss, idxs, priorities = agent.learn(
                            experiences, n_step=n_step, per=per
                        )
                        memory.update_priorities(idxs, priorities)
                    else:
                        experiences = sampler.sample(
                            agent.batch_size,
                            return_idx=True if n_step_memory is not None else False,
                        )
                        if n_step_memory is not None:
                            n_step_experiences = n_step_sampler.sample(experiences[5])
                            experiences += n_step_experiences
                            loss, *_ = agent.learn(experiences, n_step=n_step)
                        else:
                            loss = agent.learn(experiences)
                            if algo == "Rainbow DQN":
                                loss, *_ = loss

                if is_vectorised:
                    terminations.append(done)
                    rewards.append(reward)
                    truncs.append(trunc)
                else:
                    score += reward
                if loss is not None:
                    losses.append(loss)
                state = next_state

            if is_vectorised:
                scores = calculate_vectorized_scores(
                    np.array(rewards).transpose((1, 0)),
                    np.array(terminations).transpose((1, 0)),
                )
                score = np.mean(scores)

            agent.scores.append(score)
            if isinstance(losses[-1], tuple):
                actor_losses, critic_losses = list(zip(*losses))
                mean_loss = np.mean(
                    [loss for loss in actor_losses if loss is not None]
                ), np.mean(critic_losses)
            else:
                mean_loss = np.mean(losses)
            pop_loss[agent_idx].append(mean_loss)
            agent.steps[-1] += max_steps
            total_steps += max_steps

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

        # Now evolve if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                agent.test(
                    env, swap_channels=swap_channels, max_steps=max_steps, loop=evo_loop
                )
                for agent in pop
            ]
            pop_fitnesses.append(fitnesses)
            mean_scores = np.mean([agent.scores[-evo_epochs:] for agent in pop], axis=1)

            wandb_dict = {
                "global_step": (
                    total_steps * accelerator.state.num_processes
                    if accelerator is not None and accelerator.is_main_process
                    else total_steps
                ),
                "train/mean_score": np.mean(mean_scores),
                "eval/mean_fitness": np.mean(fitnesses),
                "eval/best_fitness": np.max(fitnesses),
            }

            # Create the loss dictionaries
            if algo in ["RainbowDQN", "DQN"]:
                actor_loss_dict = {
                    f"train/agent_{index}_actor_loss": np.mean(loss[-evo_epochs:])
                    for index, loss in enumerate(pop_loss)
                }
                wandb_dict.update(actor_loss_dict)
            elif algo in ["TD3", "DDPG"]:
                actor_loss_dict = {
                    f"train/agent_{index}_actor_loss": np.mean(
                        list(zip(*loss_list))[0][-evo_epochs:]
                    )
                    for index, loss_list in enumerate(pop_loss)
                }
                critic_loss_dict = {
                    f"train/agent_{index}_critic_loss": np.mean(
                        list(zip(*loss_list))[-1][-evo_epochs:]
                    )
                    for index, loss_list in enumerate(pop_loss)
                }
                wandb_dict.update(actor_loss_dict)
                wandb_dict.update(critic_loss_dict)

            if algo in ["DQN", "Rainbow DQN"]:
                train_actions_hist = [
                    freq / sum(train_actions_hist) for freq in train_actions_hist
                ]
                train_actions_dict = {
                    f"train/action_{index}": action
                    for index, action in enumerate(train_actions_hist)
                }
                wandb_dict.update(train_actions_dict)

            if wb:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        wandb.log(wandb_dict)
                    accelerator.wait_for_everyone()
                else:
                    wandb.log(wandb_dict)

            # Update step counter
            for agent in pop:
                agent.steps.append(agent.steps[-1])

            # Early stop if consistently reaches target
            if target is not None:
                if (
                    np.all(
                        np.greater(
                            [np.mean(agent.fitness[-100:]) for agent in pop], target
                        )
                    )
                    and idx_epi >= 100
                ):
                    if wb:
                        wandb.finish()
                    return pop, pop_fitnesses

            # Tournament selection and population mutation
            if tournament and mutation is not None:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.unwrap_models()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        elite, pop = tournament.select(pop)
                        pop = mutation.mutation(pop)
                        for pop_i, model in enumerate(pop):
                            model.save_checkpoint(
                                f"{accel_temp_models_path}/{algo}_{pop_i}.pt"
                            )
                    accelerator.wait_for_everyone()
                    if not accelerator.is_main_process:
                        for pop_i, model in enumerate(pop):
                            model.load_checkpoint(
                                f"{accel_temp_models_path}/{algo}_{pop_i}.pt"
                            )
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.wrap_models()
                else:
                    elite, pop = tournament.select(pop)
                    pop = mutation.mutation(pop)

                if save_elite and (idx_epi + 1 == n_episodes):
                    elite_save_path = (
                        elite_path.split(".pt")[0]
                        if elite_path is not None
                        else "{}-elite_{}-{}".format(
                            env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
                        )
                    )
                    elite.save_checkpoint(f"{elite_save_path}.pt")

            if verbose:
                fitness = ["%.2f" % fitness for fitness in fitnesses]
                avg_fitness = ["%.2f" % np.mean(agent.fitness[-100:]) for agent in pop]
                avg_score = ["%.2f" % np.mean(agent.scores[-100:]) for agent in pop]
                agents = [agent.index for agent in pop]
                num_steps = [agent.steps[-1] for agent in pop]
                muts = [agent.mut for agent in pop]
                pbar.update(0)

                print(
                    f"""
                    --- Epoch {idx_epi + 1} ---
                    Fitness:\t\t{fitness}
                    100 fitness avgs:\t{avg_fitness}
                    100 score avgs:\t{avg_score}
                    Agents:\t\t{agents}
                    Steps:\t\t{num_steps}
                    Mutations:\t\t{muts}
                    """,
                    end="\r",
                )

        # Save model checkpoint
        if checkpoint is not None:
            if (idx_epi + 1) % checkpoint == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.unwrap_models()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        for i, agent in enumerate(pop):
                            agent.save_checkpoint(f"{save_path}_{i}_{idx_epi+1}.pt")
                        print("Saved checkpoint.")
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.wrap_models()
                    accelerator.wait_for_everyone()
                else:
                    for i, agent in enumerate(pop):
                        agent.save_checkpoint(f"{save_path}_{i}_{idx_epi+1}.pt")
                    print("Saved checkpoint.")

    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.wait_for_everyone()
        else:
            wandb.finish()

    pbar.close()
    return pop, pop_fitnesses
