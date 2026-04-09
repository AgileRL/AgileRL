import logging
import warnings
from datetime import datetime
from typing import Any

from accelerate import Accelerator
from tensordict import TensorDict
from torch.utils.data import DataLoader

from agilerl.algorithms import NeuralTS, NeuralUCB
from agilerl.components.data import ReplayDataset
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.sampler import Sampler
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.protocols import BanditEnvProtocol
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)

InitDictType = dict[str, Any] | None
PopulationType = list[NeuralTS | NeuralUCB]

logger = logging.getLogger(__name__)


def train_bandits(
    env: BanditEnvProtocol,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: ReplayBuffer,
    init_hp: InitDictType = None,
    mut_p: InitDictType = None,
    max_steps: int = 20000,
    episode_steps: int = 500,
    evo_steps: int = 2500,
    eval_steps: int = 500,
    eval_loop: int = 1,
    target: float | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    checkpoint: int | None = None,
    checkpoint_path: str | None = None,
    overwrite_checkpoints: bool = False,
    save_elite: bool = False,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
) -> tuple[PopulationType, list[float]]:
    """Run the general bandit training; returns trained population of agents
    and their fitnesses.

    :param env: The bandit environment to train in.
    :type env: BanditEnvProtocol
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[object]
    :param memory: Experience Replay Buffer
    :type memory: object
    :param init_hp: Dictionary containing initial hyperparameters, defaults to None
    :type init_hp: dict, optional
    :param mut_p: Dictionary containing mutation parameters, defaults to None
    :type mut_p: dict, optional
    :param max_steps: Maximum number of steps in environment, defaults to 20000
    :type max_steps: int, optional
    :param episode_steps: Number of steps in environment per episode, defaults to 500
    :type episode_steps: int, optional
    :param evo_steps: Evolution frequency (steps), defaults to 2500
    :type evo_steps: int, optional
    :param eval_steps: Number of evaluation steps per episode, defaults to 500
    :type eval_steps: int, optional
    :param eval_loop: Number of evaluation episodes, defaults to 1
    :type eval_loop: int, optional
    :param target: Target score for early stopping, defaults to None
    :type target: float, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (steps), defaults to None
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
    :param tensorboard: TensorBoard tracking, defaults to False
    :type tensorboard: bool, optional
    :param tensorboard_log_dir: Directory for TensorBoard logs, defaults to None
    :type tensorboard_log_dir: str, optional
    :param verbose: Display training stats, defaults to True
    :type verbose: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wandb_api_key: API key for Weights & Biases, defaults to None
    :type wandb_api_key: str, optional
    :param wandb_kwargs: Additional kwargs to pass to wandb.init()
    :type wandb_kwargs: dict, optional

    :return: Trained population of agents and their fitnesses
    :rtype: tuple[list[RLAlgorithm], list[float]]
    """
    assert isinstance(
        algo,
        str,
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    if target is not None:
        assert isinstance(
            target,
            (float, int),
        ), "Target score must be a float or an integer."
    if checkpoint is not None:
        assert isinstance(checkpoint, int), "Checkpoint must be an integer."
    assert isinstance(
        wb,
        bool,
    ), "'wb' must be a boolean flag, indicating whether to record run with W&B"
    assert isinstance(verbose, bool), "Verbose must be a boolean."
    if save_elite is False and elite_path is not None:
        warnings.warn(
            "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True.",
            stacklevel=2,
        )
    if checkpoint is None and checkpoint_path is not None:
        warnings.warn(
            "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined.",
            stacklevel=2,
        )

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name,
            algo,
            datetime.now().strftime("%m%d%Y%H%M%S"),
        )
    )

    if accelerator is not None:
        # Create dataloader from replay buffer
        replay_dataset = ReplayDataset(memory, pop[0].batch_size)
        replay_dataloader = DataLoader(replay_dataset, batch_size=None)
        replay_dataloader = accelerator.prepare(replay_dataloader)
        sampler = Sampler(dataset=replay_dataset, dataloader=replay_dataloader)
    else:
        sampler = Sampler(memory=memory)

    # Format progress bar
    pbar = default_progress_bar(max_steps, accelerator)

    loggers = init_loggers(
        algo=algo,
        env_name=env_name,
        pbar=pbar,
        verbose=verbose,
        wb=wb,
        tensorboard=tensorboard,
        tensorboard_log_dir=tensorboard_log_dir,
        accelerator=accelerator,
        wandb_api_key=wandb_api_key,
        wandb_kwargs=wandb_kwargs,
        init_hyperparams=init_hp,
        mutation_hyperparams=mut_p,
    )

    # Initialize population wrapper for metrics reporting
    population = Population(
        agents=pop,
        accelerator=accelerator,
        loggers=loggers,
    )

    # Pre-training mutation
    if accelerator is None and mutation is not None:
        population.update(mutation.mutation(population.agents, pre_training_mut=True))

    checkpoint_count = 0
    evo_count = 0

    # RL training loop
    while population.all_below(max_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

        for agent in population.agents:
            agent.set_training_mode(True)
            agent.init_evo_step()

            score = 0
            context = env.reset()
            for _idx_step in range(episode_steps):
                # Get next action from agent
                action = agent.get_action(context)
                next_context, reward = env.step(action)

                # Save experience to replay buffer
                transition = TensorDict({"obs": context[action], "reward": reward})
                transition = transition.unsqueeze(0)
                transition.batch_size = [1]
                memory.add(transition)

                # Learn according to learning frequency
                if len(memory) >= agent.batch_size:
                    for _ in range(agent.learn_step):
                        experiences = sampler.sample(agent.batch_size)
                        agent.learn(experiences)

                score += reward
                agent.regret.append(agent.regret[-1] + 1 - reward)

                context = next_context

            agent.add_scores([score])
            agent.finalize_evo_step(episode_steps)
            pbar.update(episode_steps // population.size)

        population.increment_evo_step()

        # Evaluate population
        for agent in population.agents:
            agent.test(
                env,
                max_steps=eval_steps,
                loop=eval_loop,
            )

        # Aggregate metrics from all agents and log -> clear metrics after reporting
        population.report_metrics(clear=True)

        # Check if target score has been reached
        if population.should_stop(target):
            logger.info("Target score has been reached. Stopping training.")
            population.finish()
            pbar.close()
            return population.agents, population.last_fitnesses

        # Tournament selection and population mutation
        if tournament and mutation is not None:
            if (population.local_step // evo_steps) > evo_count:
                population.update(
                    tournament_selection_and_mutation(
                        population=population.agents,
                        tournament=tournament,
                        mutation=mutation,
                        env_name=env_name,
                        algo=algo,
                        elite_path=elite_path,
                        save_elite=save_elite,
                        accelerator=accelerator,
                    ),
                )
                evo_count += 1

        # Save model checkpoint
        if checkpoint is not None:
            if population.local_step // checkpoint > checkpoint_count:
                save_population_checkpoint(
                    population=population.agents,
                    save_path=save_path,
                    overwrite_checkpoints=overwrite_checkpoints,
                    accelerator=accelerator,
                )
                checkpoint_count += 1

    population.finish()
    pbar.close()
    return population.agents, population.last_fitnesses
