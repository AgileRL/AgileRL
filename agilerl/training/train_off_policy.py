import logging
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from agilerl.algorithms import DDPG, DQN, TD3, RainbowDQN
from agilerl.components import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.data import ReplayDataset, Transition
from agilerl.components.sampler import Sampler
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.networks.actors import DeterministicActor
from agilerl.population import Population
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)
from agilerl.vector import DummyVecEnv

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensordict import TensorDict, TensorDictBase

    from agilerl.typing import ExperiencesType

InitDictType = dict[str, Any] | None
SupportedOffPolicy = DQN | RainbowDQN | DDPG | TD3
PopulationType = list[SupportedOffPolicy]
BufferType = ReplayBuffer | PrioritizedReplayBuffer | MultiStepReplayBuffer

logger = logging.getLogger(__name__)


def _learn_from_buffer(
    agent: SupportedOffPolicy,
    sampler: Sampler,
    memory: BufferType,
    n_step_memory: MultiStepReplayBuffer | None,
    n_step_sampler: Sampler | None,
    per: bool,
) -> None:
    """Execute a single learning step for the agent."""
    # `Sampler.sample` is bound to one of the sampling strategies at construction
    # time, so only its return type is statically known. The buffers hand back
    # `TensorDict` batches, which `ExperiencesType` (agilerl/typing.py) does not yet
    # cover - the casts to it below record that gap.
    sample = cast("Callable[..., TensorDict]", sampler.sample)

    # Prioritized and n-step replay are the preserve of the RainbowDQN-style
    # algorithms: only they anneal `beta`, take `n_experiences`/`per`, and return
    # the indices and priorities to write back. The suppressions below cover the
    # other members of the annotated population union, and the buffer protocol
    # (`update_priorities` lives on `PrioritizedReplayBuffer`), for a path that is
    # gated on the buffer the caller passed in.
    if per:
        experiences = sample(
            agent.batch_size,
            agent.beta,  # ty: ignore[unresolved-attribute]
        )
        n_step_experiences = (
            cast("Callable[..., TensorDict]", n_step_sampler.sample)(
                experiences["idxs"],
            )
            if n_step_sampler is not None
            else None
        )
        _loss, idxs, priorities = agent.learn(  # ty: ignore[invalid-assignment, not-iterable]
            cast("ExperiencesType", experiences),
            n_experiences=cast("ExperiencesType | None", n_step_experiences),  # ty: ignore[unknown-argument]
            per=per,  # ty: ignore[unknown-argument]
        )
        memory.update_priorities(idxs, priorities)  # ty: ignore[unresolved-attribute]
    else:
        experiences = sample(
            agent.batch_size,
            return_idx=n_step_memory is not None,
        )
        if n_step_sampler is not None:
            n_step_experiences = cast(
                "Callable[..., TensorDict]",
                n_step_sampler.sample,
            )(experiences["idxs"])
            agent.learn(
                cast("ExperiencesType", experiences),
                n_experiences=cast("ExperiencesType", n_step_experiences),  # ty: ignore[unknown-argument]
            )
        else:
            agent.learn(cast("ExperiencesType", experiences))


def train_off_policy(
    env: gym.Env | gym.vector.VectorEnv,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: BufferType,
    init_hp: InitDictType = None,
    mut_p: InitDictType = None,
    max_steps: int = 1000000,
    evo_steps: int = 10000,
    eval_steps: int | None = None,
    eval_loop: int = 1,
    learning_delay: int = 0,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.999,
    target: float | None = None,
    n_step_memory: MultiStepReplayBuffer | None = None,
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
    """Run the general online off-policy RL training; returns trained population
    of agents and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[RLAlgorithm]
    :param memory: Experience Replay Buffer
    :type memory: object
    :param init_hp: Dictionary containing initial hyperparameters, defaults to None
    :type init_hp: dict, optional
    :param mut_p: Dictionary containing mutation parameters, defaults to None
    :type mut_p: dict, optional
    :param max_steps: Maximum number of steps in environment, defaults to 1000000
    :type max_steps: int, optional
    :param evo_steps: Evolution frequency (steps), defaults to 10000
    :type evo_steps: int, optional
    :param eval_steps: Number of evaluation steps per episode. If None, will evaluate until
        environment terminates or truncates. Defaults to None
    :type eval_steps: int, optional
    :param eval_loop: Number of evaluation episodes, defaults to 1
    :type eval_loop: int, optional
    :param learning_delay: Steps in environment before starting learning, defaults to 0
    :type learning_delay: int, optional
    :param eps_start: Maximum exploration - initial epsilon value, defaults to 1.0
    :type eps_start: float, optional
    :param eps_end: Minimum exploration - final epsilon value, defaults to 0.1
    :type eps_end: float, optional
    :param eps_decay: Epsilon decay per episode, defaults to 0.995
    :type eps_decay: float, optional
    :param target: Target score for early stopping, defaults to None
    :type target: float, optional
    :param n_step_memory: Multi-step Experience Replay Buffer to be used alongside Prioritized
        ERB, defaults to None
    :type n_step_memory: object, optional
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
    assert isinstance(eps_start, float), "Starting epsilon must be a float."
    assert isinstance(eps_end, float), "Final value of epsilon must be a float."
    assert isinstance(eps_decay, float), "Epsilon decay rate must be a float."
    if target is not None:
        assert isinstance(
            target,
            (float, int),
        ), "Target score must be a float or an integer."

    per = isinstance(memory, PrioritizedReplayBuffer)
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

    # Ensure environment has vectorized interface. `DummyVecEnv` duck-types the
    # `VectorEnv` API rather than subclassing it, so the cast matches the annotations
    # of the algorithm methods it is handed to.
    vec_env = cast(
        "gym.vector.VectorEnv",
        env if hasattr(env, "num_envs") else DummyVecEnv(env),
    )

    num_envs = vec_env.num_envs

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
        # NOTE: n-step sampling requires index-based lookups which the distributed
        # sampler does not support (sample_distributed ignores return_idx).
        n_step_sampler = None
    else:
        sampler = Sampler(memory=memory)
        n_step_sampler = (
            Sampler(memory=n_step_memory) if n_step_memory is not None else None
        )

    # Format progress bar
    pbar = default_progress_bar(max_steps, accelerator)

    # Initialize loggers for metrics reporting
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

    # Initialize population for metrics reporting
    population = Population(
        agents=pop,
        accelerator=accelerator,
        loggers=loggers,
    )

    checkpoint_count = 0

    # Pre-training mutation
    if accelerator is None and mutation is not None:
        population.update(mutation.mutation(population.agents, pre_training_mut=True))

    # RL training loop
    while population.all_below(max_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

        for agent in population.agents:
            agent.set_training_mode(True)
            agent.init_training_step()

            obs, info = vec_env.reset()
            scores = np.zeros(num_envs)
            completed_episode_scores: list[float] = []
            steps = 0

            if isinstance(agent, DQN):
                epsilon = eps_start

            for idx_step in range(evo_steps // num_envs):
                # Get next action from agent
                if isinstance(agent, DQN):
                    action_mask = info.get("action_mask", None)
                    action = agent.get_action(obs, epsilon, action_mask=action_mask)
                    epsilon = max(eps_end, epsilon * eps_decay)
                elif isinstance(agent, RainbowDQN):
                    action_mask = info.get("action_mask", None)
                    action = agent.get_action(obs, action_mask=action_mask)
                else:
                    raw_action = agent.get_action(obs)

                    # Rescale action to action space bounds
                    action = DeterministicActor.rescale_action(
                        action=torch.from_numpy(raw_action),
                        low=agent.action_low,
                        high=agent.action_high,
                        output_activation=agent.actor.output_activation,
                    )
                    action = action.cpu().numpy()

                # Act in environment
                next_obs, reward, done, trunc, info = vec_env.step(action)
                scores += np.array(reward)

                reset_noise_indices = []
                for idx, (d, t) in enumerate(zip(done, trunc, strict=False)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                if isinstance(agent, (DDPG, TD3)):
                    # `reset_action_noise` fancy-indexes with its argument, so any
                    # sequence of indices works; widening its annotation to
                    # `Sequence[int] | np.ndarray` upstream drops this suppression.
                    agent.reset_action_noise(
                        reset_noise_indices,  # ty: ignore[invalid-argument-type]
                    )

                steps += num_envs

                # Save network output in buffer
                if isinstance(agent, (DDPG, TD3)):
                    action = raw_action

                transition: TensorDictBase = Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                )

                transition = transition.to_tensordict()
                transition.batch_size = [num_envs]
                if n_step_memory is not None:
                    one_step_transition = n_step_memory.add(transition)
                    if one_step_transition is not None:
                        memory.add(one_step_transition)
                else:
                    memory.add(transition)

                if per:
                    fraction = min(
                        ((agent.metrics.steps + idx_step + 1) * num_envs / max_steps),
                        1.0,
                    )
                    # `beta` is annealed by the algorithms that support prioritized
                    # replay (RainbowDQN), not by the population union as a whole.
                    agent.beta += fraction * (  # ty: ignore[unresolved-attribute]
                        1.0 - agent.beta  # ty: ignore[unresolved-attribute]
                    )

                # Learn according to learning frequency
                # Handle learn_step > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.size > learning_delay
                    ):
                        _learn_from_buffer(
                            agent,
                            sampler,
                            memory,
                            n_step_memory,
                            n_step_sampler,
                            per,
                        )

                elif len(memory) >= agent.batch_size and memory.size > learning_delay:
                    for _ in range(num_envs // agent.learn_step):
                        _learn_from_buffer(
                            agent,
                            sampler,
                            memory,
                            n_step_memory,
                            n_step_sampler,
                            per,
                        )

                obs = next_obs

            agent.add_scores(completed_episode_scores)
            agent.finalize_training_step(steps)
            pbar.update(evo_steps // population.size)

        if isinstance(agent, DQN):
            eps_start = epsilon

        # Evaluate population
        for agent in population.agents:
            agent.test(
                vec_env,
                max_steps=eval_steps,
                loop=eval_loop,
            )

        # Report progress
        population.increment_evo_step()
        population.report_metrics(clear=True)

        if population.should_stop(target):
            logger.info("Target score has been reached. Stopping training.")
            population.finish()
            pbar.close()
            # Single-agent fitnesses are scalars; `Population` types them as the
            # wider scalar-or-per-agent-dict row shared with multi-agent training.
            return population.agents, cast("list[float]", population.last_fitnesses)

        # Tournament selection and population mutation
        if tournament and mutation is not None:
            # `tournament_selection_and_mutation` takes and returns an invariant
            # `list[EvolvableAlgorithmProtocol]`, so a concrete population is assignable
            # in neither direction; making it generic in the agent type
            # (agilerl/utils/utils.py) removes both suppressions.
            population.update(
                tournament_selection_and_mutation(  # ty: ignore[invalid-argument-type]
                    population=population.agents,  # ty: ignore[invalid-argument-type]
                    tournament=tournament,
                    mutation=mutation,
                    env_name=env_name,
                    algo=algo,
                    elite_path=elite_path,
                    save_elite=save_elite,
                    accelerator=accelerator,
                ),
            )

        # Save model checkpoint
        if checkpoint is not None:
            if population.local_step // checkpoint > checkpoint_count:
                # `save_population_checkpoint` takes the same invariant list.
                save_population_checkpoint(
                    population=population.agents,  # ty: ignore[invalid-argument-type]
                    save_path=save_path,
                    overwrite_checkpoints=overwrite_checkpoints,
                    accelerator=accelerator,
                )
                checkpoint_count += 1

    population.finish()
    pbar.close()
    return population.agents, cast("list[float]", population.last_fitnesses)
