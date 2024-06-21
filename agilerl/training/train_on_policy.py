import os
import warnings
from datetime import datetime

import numpy as np
import wandb
from tqdm import trange


def train_on_policy(
    env,
    env_name,
    algo,
    pop,
    INIT_HP=None,
    MUT_P=None,
    swap_channels=False,
    max_steps=1000000,
    evo_steps=10000,
    eval_steps=None,
    eval_loop=1,
    target=None,
    tournament=None,
    mutation=None,
    checkpoint=None,
    checkpoint_path=None,
    overwrite_checkpoints=False,
    save_elite=False,
    elite_path=None,
    wb=False,
    verbose=True,
    accelerator=None,
    wandb_api_key=None,
):
    """The general on-policy RL training function. Returns trained population of agents
    and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[object]
    :param INIT_HP: Dictionary containing initial hyperparameters, defaults to None
    :type INIT_HP: dict, optional
    :param MUT_P: Dictionary containing mutation parameters, defaults to None
    :type MUT_P: dict, optional
    :param swap_channels: Swap image channels dimension from last to first
        [H, W, C] -> [C, H, W], defaults to False
    :type swap_channels: bool, optional
    :param max_steps: Maximum number of steps in environment, defaults to 1000000
    :type max_steps: int, optional
    :param evo_steps: Evolution frequency (steps), defaults to 10000
    :type evo_steps: int, optional
    :param eval_steps: Number of evaluation steps per episode. If None, will evaluate until
        environment terminates or truncates. Defaults to None
    :type eval_steps: int, optional
    :param eval_loop: Number of evaluation episodes, defaults to 1
    :type eval_loop: int, optional
    :param target: Target score for early stopping, defaults to None
    :type target: float, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (episodes), defaults to None
    :type checkpoint: int, optional
    :param checkpoint_path: Location to save checkpoint, defaults to None
    :type checkpoint_path: str, optional
    :param overwrite_checkpoints: Overwrite previous checkpoints during training, defaults to False
    :type overwrite_checkpoints: bool, optional
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
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    if target is not None:
        assert isinstance(
            target, (float, int)
        ), "Target score must be a float or an integer."
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
        num_envs = env.num_envs
    else:
        is_vectorised = False
        num_envs = 1

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
        )
    )

    if accelerator is not None:
        print(f"\nDistributed training on {accelerator.device}...")
    else:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if accelerator is not None:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process,
        )
    else:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None
    checkpoint_count = 0

    # Pre-training mutation
    if accelerator is None:
        if mutation is not None:
            pop = mutation.mutation(pop, pre_training_mut=True)

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        if accelerator is not None:
            accelerator.wait_for_everyone()
        pop_episode_scores = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0

            for _ in range(-(evo_steps // -agent.learn_step)):

                states = []
                actions = []
                log_probs = []
                rewards = []
                dones = []
                values = []
                truncs = []

                learn_steps = 0

                for idx_step in range(-(agent.learn_step // -num_envs)):

                    if swap_channels:
                        state = np.moveaxis(state, [-1], [-3])
                    # Get next action from agent
                    action, log_prob, _, value = agent.get_action(state)

                    if not is_vectorised:
                        action = action[0]
                        log_prob = log_prob[0]
                        value = value[0]
                    next_state, reward, done, trunc, _ = env.step(
                        action
                    )  # Act in environment

                    total_steps += num_envs
                    steps += num_envs
                    learn_steps += num_envs

                    states.append(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    dones.append(done)
                    values.append(value)
                    truncs.append(trunc)

                    state = next_state
                    scores += np.array(reward)

                    if not is_vectorised:
                        done = [done]
                        trunc = [trunc]

                    for idx, (d, t) in enumerate(zip(done, trunc)):
                        if d or t:
                            completed_episode_scores.append(scores[idx])
                            agent.scores.append(scores[idx])
                            scores[idx] = 0

                pbar.update(learn_steps // len(pop))

                if swap_channels:
                    next_state = np.moveaxis(next_state, [-1], [-3])

                experiences = (
                    states,
                    actions,
                    log_probs,
                    rewards,
                    dones,
                    values,
                    next_state,
                )
                # Learn according to agent's RL algorithm
                loss = agent.learn(experiences)
                pop_loss[agent_idx].append(loss)

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env, swap_channels=swap_channels, max_steps=eval_steps, loop=eval_loop
            )
            for agent in pop
        ]
        pop_fitnesses.append(fitnesses)
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        if wb:
            wandb_dict = {
                "global_step": (
                    total_steps * accelerator.state.num_processes
                    if accelerator is not None and accelerator.is_main_process
                    else total_steps
                ),
                "train/mean_score": np.mean(
                    [
                        mean_score
                        for mean_score in mean_scores
                        if not isinstance(mean_score, str)
                    ]
                ),
                "eval/mean_fitness": np.mean(fitnesses),
                "eval/best_fitness": np.max(fitnesses),
            }

            agent_loss_dict = {
                f"train/agent_{index}_loss": np.mean(loss_[-10:])
                for index, loss_ in enumerate(pop_loss)
            }
            wandb_dict.update(agent_loss_dict)

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
                    np.greater([np.mean(agent.fitness[-10:]) for agent in pop], target)
                )
                and len(pop[0].steps) >= 100
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

            if save_elite:
                elite_save_path = (
                    elite_path.split(".pt")[0]
                    if elite_path is not None
                    else f"{env_name}-elite_{algo}"
                )
                elite.save_checkpoint(f"{elite_save_path}.pt")

        if verbose:
            fitness = ["%.2f" % fitness for fitness in fitnesses]
            avg_fitness = ["%.2f" % np.mean(agent.fitness[-5:]) for agent in pop]
            avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
            agents = [agent.index for agent in pop]
            num_steps = [agent.steps[-1] for agent in pop]
            muts = [agent.mut for agent in pop]
            pbar.update(0)

            print(
                f"""
                --- Global Steps {total_steps} ---
                Fitness:\t\t{fitness}
                Score:\t\t{mean_scores}
                5 fitness avgs:\t{avg_fitness}
                10 score avgs:\t{avg_score}
                Agents:\t\t{agents}
                Steps:\t\t{num_steps}
                Mutations:\t\t{muts}
                """,
                end="\r",
            )

        # Save model checkpoint
        if checkpoint is not None:
            if pop[0].steps[-1] // checkpoint > checkpoint_count:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.unwrap_models()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        for i, agent in enumerate(pop):
                            current_checkpoint_path = (
                                f"{save_path}_{i}.pt"
                                if overwrite_checkpoints
                                else f"{save_path}_{i}_{agent.steps[-1]}.pt"
                            )
                            agent.save_checkpoint(current_checkpoint_path)
                        print("Saved checkpoint.")
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.wrap_models()
                    accelerator.wait_for_everyone()
                else:
                    for i, agent in enumerate(pop):
                        current_checkpoint_path = (
                            f"{save_path}_{i}.pt"
                            if overwrite_checkpoints
                            else f"{save_path}_{i}_{agent.steps[-1]}.pt"
                        )
                        agent.save_checkpoint(current_checkpoint_path)
                    print("Saved checkpoint.")
                checkpoint_count += 1

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
