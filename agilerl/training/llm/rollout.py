# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Generative rollout training for LLM finetuning.

:func:`train_llm_rollout` drives on-policy RL (GRPO / PPO / REINFORCE) over a
:class:`~agilerl.llm_envs.RolloutHarness`; single-turn reasoning
(``max_turns=1``) and multi-turn share this one loop.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
from accelerate import Accelerator

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import GRPO
from agilerl.components.llm_rollout_data import LLMExperienceBatch
from agilerl.population import Population
from agilerl.training.configs import LLMRolloutRunConfig, LoggerExperiment
from agilerl.training.llm.common import (
    LLMCheckpointProgress,
    _validate_finetune_args,
    maybe_save_llm_step_checkpoint,
    save_llm_elite_if_requested,
)
from agilerl.utils.constructor_kwargs import accept_flat_kwargs
from agilerl.utils.llm_utils import (
    align_completion_batch_shapes_across_ranks,
    allreduce_minmax_int,
    needs_cross_rank_seq_padding,
    safe_aggregate_metrics,
)
from agilerl.utils.utils import (
    data_parallel_topology,
    default_progress_bar,
    init_loggers,
    resolve_selection_strategy,
    run_selection_and_mutation,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.algorithms import LLMPPO, LLMREINFORCE
    from agilerl.llm_envs import RolloutCollector, RolloutHarness
    from agilerl.rollouts.on_policy import collate_llm_rollouts, collect_rollouts_llm

    SupportedRollout = GRPO | LLMPPO | LLMREINFORCE
else:
    SupportedRollout = GRPO

if TYPE_CHECKING:
    from tqdm import tqdm


def _any_rank_flag(
    local: bool,
    accelerator: Accelerator | None,
) -> bool:
    """Whether any data-parallel rank has this flag set."""
    if accelerator is None:
        return local
    _, max_flag = allreduce_minmax_int(int(local), accelerator)
    return max_flag == 1


def _any_rank_empty_batch(
    local_empty: bool,
    accelerator: Accelerator | None,
) -> bool:
    """Whether any data-parallel rank has an empty rollout this step."""
    # Mixed empty/non-empty ranks would split learn() collectives.
    return _any_rank_flag(local_empty, accelerator)


@dataclass
class LLMRolloutSession:
    """Population, collector, and run config for one ``train_llm_rollout`` loop."""

    population: Population
    run: LLMRolloutRunConfig
    max_turns: int
    env_factory: Callable[[], RolloutHarness]
    init_hp: dict[str, Any]
    accelerator: Accelerator | None
    pbar: tqdm
    collector: RolloutCollector
    env_name: str
    batch_size: int
    data_increment: int
    group_size: int
    group_seed: int
    checkpoint_progress: LLMCheckpointProgress
    test_env: RolloutHarness | None = None
    total_steps: int = 0
    iteration: int = 0
    consecutive_stalls: int = 0
    wall_deadline: float | None = None


def _resolve_rollout_run(run: LLMRolloutRunConfig | None) -> LLMRolloutRunConfig:
    run = run or LLMRolloutRunConfig()
    evolution = run.evolution
    selection = resolve_selection_strategy(
        evolution.selection_strategy,
        evolution.tournament,
    )
    return replace(run, evolution=replace(evolution, selection_strategy=selection))


def _validate_rollout_run(
    pop: list[SupportedRollout],
    run: LLMRolloutRunConfig,
) -> None:
    loop = run.loop
    evolution = run.evolution
    _validate_finetune_args(
        loop.evo_steps,
        evolution.selection_strategy,
        evolution.mutation,
        None,
        loop.max_steps,
        pop,
        (LLMPPO, LLMREINFORCE, GRPO),
        (
            "The algorithm must be LLMPPO, LLMREINFORCE, or GRPO (including the "
            "GRPO variants CISPO and GSPO) for rollout finetuning. Got "
            f"{type(pop[0])} instead."
        ),
        checkpoint_steps=run.checkpoint.checkpoint_steps,
    )


def _start_llm_rollout(
    pop: list[SupportedRollout],
    max_turns: int,
    env_factory: Callable[[], RolloutHarness],
    init_hp: dict[str, Any] | None,
    run: LLMRolloutRunConfig,
    accelerator: Accelerator | None,
) -> LLMRolloutSession:
    """Validate args, build loggers, and open the rollout collector."""
    _validate_rollout_run(pop, run)
    if init_hp is None:
        init_hp = {
            "BATCH_SIZE_PER_GPU": pop[0].batch_size_per_process,
            "ALGO": pop[0].algo,
        }
    batch_size = init_hp.get("BATCH_SIZE", pop[0].batch_size)
    env_name = init_hp.get("env_name", "rollout")
    # Tensor-parallel ranks of one replica generate the same sequences, so the
    # data-parallel topology -- not the process group -- is what splits work.
    dp_rank, data_increment = data_parallel_topology(
        accelerator,
        getattr(getattr(pop[0], "vllm_config", None), "tensor_parallel_size", 1) or 1,
    )
    if run.logging.wb:
        init_hp["effective_data_batch_size"] = data_increment * batch_size
        init_hp["batch_size"] = batch_size
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path
        init_hp["max_turns"] = max_turns
    pbar = default_progress_bar(run.loop.max_steps, accelerator)
    loggers = init_loggers(
        experiment=LoggerExperiment(
            algo=init_hp.get("ALGO", pop[0].algo),
            env_name=env_name,
            init_hyperparams=init_hp,
        ),
        pbar=pbar,
        logging=run.logging,
        accelerator=accelerator,
    )
    group_size = getattr(pop[0], "group_size", 1)
    return LLMRolloutSession(
        population=Population(agents=pop, accelerator=accelerator, loggers=loggers),
        run=run,
        max_turns=max_turns,
        env_factory=env_factory,
        init_hp=init_hp,
        accelerator=accelerator,
        pbar=pbar,
        collector=RolloutCollector(
            env_factory,
            batch_size,
            group_size,
            io_timeout_s=run.loop.io_timeout_s,
            rank=dp_rank,
            world_size=data_increment,
        ),
        env_name=env_name,
        batch_size=batch_size,
        data_increment=data_increment,
        group_size=group_size,
        group_seed=int(pop[0].seed) + dp_rank * (1 << 31),
        checkpoint_progress=LLMCheckpointProgress(run.checkpoint.checkpoint_steps),
        wall_deadline=_rollout_wall_deadline(run.loop.max_wall_seconds),
    )


def _rollout_wall_deadline(max_wall_seconds: float | None) -> float | None:
    if max_wall_seconds is None or max_wall_seconds <= 0:
        return None
    return time.monotonic() + max_wall_seconds


def _wall_limit_reached(session: LLMRolloutSession) -> bool:
    if session.wall_deadline is None:
        return False
    # Every DP rank takes the same stop/continue branch; a rank-local
    # wall-clock check can leave a peer blocked in collect's barrier.
    if not _any_rank_flag(time.monotonic() >= session.wall_deadline, session.accelerator):
        return False
    if session.accelerator is None or session.accelerator.is_main_process:
        print(
            f"\nStopping rollout training: wall time limit "
            f"({session.run.loop.max_wall_seconds}s) reached.",
        )
    return True


def _collect_rollout_batch(
    session: LLMRolloutSession,
    agent: SupportedRollout,
) -> tuple[LLMExperienceBatch, int]:
    """Collect and collate one agent's rollout; updates ``group_seed``."""
    agent.set_reference_policy(session.collector.num_epochs)
    agent.init_training_step()
    (
        token_ids_list,
        action_masks_list,
        all_turn_ids,
        all_rewards,
        batch_steps,
        group_seed,
        all_sampling_logps,
    ) = collect_rollouts_llm(
        agent=agent,
        env=session.collector,
        n_steps=session.max_turns,
        batch_size=session.batch_size,
        group_seed=session.group_seed,
    )
    session.group_seed = group_seed
    # Collated a prompt group at a time, so the group-divisibility the
    # algorithms rely on holds by construction, and a misaligned
    # rollout fails here rather than inside a loss.
    batch = collate_llm_rollouts(
        token_ids_list,
        action_masks_list,
        all_turn_ids,
        all_rewards,
        all_sampling_logps,
        group_size=session.group_size,
    )
    return batch, batch_steps


def _experiences_for_learn(
    session: LLMRolloutSession,
    agent: SupportedRollout,
    batch: LLMExperienceBatch,
) -> tuple[tuple[object, object, object], torch.Tensor | None]:
    experiences = batch.experiences()
    turn_ids = batch.turn_ids
    if session.accelerator is None or not needs_cross_rank_seq_padding(
        agent, world_size=session.data_increment
    ):
        return experiences, turn_ids
    # Multi-rank Liger token-level losses allreduce per chunk, so
    # every rank must pad to one global sequence length.
    token_ids, action_masks, rewards = align_completion_batch_shapes_across_ranks(
        batch.token_ids,
        batch.action_masks,
        batch.rewards,
        pad_token_id=agent.pad_token_id,
        accelerator=session.accelerator,
    )
    if turn_ids is None:
        msg = "aligned batches are non-empty"
        raise RuntimeError(msg)
    target_mask_len = int(action_masks.shape[1])
    if int(turn_ids.shape[1]) < target_mask_len:
        turn_ids = torch.nn.functional.pad(
            turn_ids,
            (0, target_mask_len - int(turn_ids.shape[1])),
            value=-1,
        )
    return (token_ids, action_masks, rewards), turn_ids


def _log_rollout_agent_metrics(
    session: LLMRolloutSession,
    agent: SupportedRollout,
    batch: LLMExperienceBatch,
    mean_score: torch.Tensor,
) -> None:
    loop = session.run.loop
    if loop.max_reward is not None:
        if "accuracy" not in agent.metrics.additional_metrics:
            agent.metrics.register("accuracy")
        accuracy = (batch.rewards.sum(dim=1) >= loop.max_reward).float().mean()
        accuracy = accuracy.to(agent.device)
        agg_accuracy = safe_aggregate_metrics(session.accelerator, accuracy)
        if session.accelerator is None or session.accelerator.is_main_process:
            agent.metrics.log("accuracy", agg_accuracy)
    for name, mean_value in session.collector.get_rubric_score_means().items():
        metric = f"reward_{name}"
        if metric not in agent.metrics.additional_metrics:
            agent.metrics.register(metric)
        agg = safe_aggregate_metrics(
            session.accelerator, mean_score.new_tensor(mean_value)
        )
        if session.accelerator is None or session.accelerator.is_main_process:
            agent.metrics.log(metric, agg)


def _learn_nonempty_rollout(
    session: LLMRolloutSession,
    agent: SupportedRollout,
    batch: LLMExperienceBatch,
    batch_steps: int,
) -> int:
    episode_scores = batch.rewards.sum(dim=1)
    mean_score = episode_scores.mean().to(agent.device)
    experiences, turn_ids = _experiences_for_learn(session, agent, batch)
    agent.learn(
        experiences,
        turn_ids=turn_ids,
        sampling_logps=batch.sampling_logps,
    )
    agg_score = safe_aggregate_metrics(session.accelerator, mean_score)
    _log_rollout_agent_metrics(session, agent, batch, mean_score)
    effective_batch_steps = batch_steps * session.data_increment
    agent.finalize_training_step(batch_steps)
    session.total_steps += effective_batch_steps
    if session.accelerator is None or session.accelerator.is_main_process:
        agent.add_scores([float(agg_score)])
    return effective_batch_steps


def _train_one_rollout_agent(
    session: LLMRolloutSession,
    agent: SupportedRollout,
) -> int:
    """Collect, learn, and return effective batch steps for one agent."""
    batch, batch_steps = _collect_rollout_batch(session, agent)
    # Empty collated batches have no trajectories; learn() raises on them.
    # iteration_steps stays 0, so the stall guard after this loop fires.
    if _any_rank_empty_batch(
        batch_steps == 0 or batch.is_empty,
        session.accelerator,
    ):
        agent.finalize_training_step(0)
        return 0
    return _learn_nonempty_rollout(session, agent, batch, batch_steps)


def _maybe_eval_rollout_agent(
    session: LLMRolloutSession,
    agent: SupportedRollout,
) -> None:
    if (session.iteration + 1) % session.run.loop.evaluation_interval != 0:
        return
    if session.test_env is None:
        session.test_env = session.env_factory()
    agent.test(session.test_env, loop=session.run.loop.eval_loop)
    if session.accelerator is not None:
        session.accelerator.wait_for_everyone()


def _note_rollout_progress(session: LLMRolloutSession, iteration_steps: int) -> None:
    """Abort when rollouts repeatedly yield no usable turns."""
    if iteration_steps != 0:
        session.consecutive_stalls = 0
        return
    session.consecutive_stalls += 1
    if session.accelerator is None or session.accelerator.is_main_process:
        warnings.warn(
            "Rollout produced no usable turns this iteration "
            "(batch_steps == 0 for every agent), so training did not "
            "advance. Every prompt likely exceeds the context budget — "
            "check that max_output_tokens leaves room under "
            "max_context_length.",
            stacklevel=2,
        )
    if session.consecutive_stalls < 8:
        return
    msg = (
        f"Rollout training made no progress for {session.consecutive_stalls} "
        "consecutive iterations (batch_steps == 0 every time), so "
        "total_steps can never reach max_steps. Aborting instead of "
        "looping forever. Likely cause: the prompt budget is "
        "exhausted (max_output_tokens too close to "
        "max_context_length), or every sampled prompt exceeds the "
        "context length."
    )
    raise RuntimeError(msg)


def _report_rollout_metrics(session: LLMRolloutSession, iteration_steps: int) -> None:
    if session.accelerator is None or session.accelerator.is_main_process:
        session.pbar.update(iteration_steps // len(session.population.agents))
        session.population.report_metrics(clear=True)
    else:
        # Metrics accumulate on every rank; only main reports, so the
        # others must still clear or their stores grow for the whole run.
        session.population.clear_agent_metrics()
    if session.accelerator is not None:
        session.accelerator.wait_for_everyone()


def _evolve_llm_rollout(session: LLMRolloutSession) -> None:
    evolution = session.run.evolution
    checkpoint = session.run.checkpoint
    if session.accelerator is not None:
        session.accelerator.wait_for_everyone()
    session.population.update(
        run_selection_and_mutation(
            evolution.selection_strategy,
            population=session.population.agents,
            mutation=evolution.mutation,
            env_name=session.env_name,
            accelerator=session.accelerator,
            language_model=True,
            elite_path=checkpoint.elite_path,
            save_elite=bool(checkpoint.save_elite),
        ),
    )
    session.group_size = getattr(session.population.agents[0], "group_size", 1)
    session.collector.update_rollout_geometry(
        rollout_batch_size=session.batch_size,
        group_size=session.group_size,
    )
    if session.accelerator is not None:
        session.accelerator.wait_for_everyone()
    session.population.increment_evo_step()


def _evolve_or_checkpoint_rollout(session: LLMRolloutSession) -> None:
    evolution = session.run.evolution
    loop = session.run.loop
    evo_ready = (
        evolution.selection_strategy is not None
        and evolution.mutation is not None
        and loop.evo_steps is not None
    )
    if evo_ready and (session.iteration + 1) % loop.evo_steps == 0:
        _evolve_llm_rollout(session)
        return
    if evo_ready:
        return
    maybe_save_llm_step_checkpoint(
        session.population.agents,
        session.checkpoint_progress,
        session.run.checkpoint,
        session.total_steps,
        loop.max_steps,
    )


def _close_llm_rollout(session: LLMRolloutSession) -> None:
    try:
        session.collector.close()
    finally:
        if session.test_env is not None:
            session.test_env.close()


def _run_llm_rollout_loop(session: LLMRolloutSession) -> None:
    max_steps = session.run.loop.max_steps
    while session.total_steps < max_steps:
        if _wall_limit_reached(session):
            break
        iteration_steps = 0
        for agent in session.population.agents:
            iteration_steps += _train_one_rollout_agent(session, agent)
            _maybe_eval_rollout_agent(session, agent)
        _note_rollout_progress(session, iteration_steps)
        _report_rollout_metrics(session, iteration_steps)
        _evolve_or_checkpoint_rollout(session)
        session.iteration += 1


@accept_flat_kwargs
def train_llm_rollout(
    pop: list[SupportedRollout],
    max_turns: int,
    env_factory: Callable[[], RolloutHarness],
    init_hp: dict[str, Any] | None = None,
    run: LLMRolloutRunConfig | None = None,
    accelerator: Accelerator | None = None,
) -> tuple[list[SupportedRollout], Any]:
    """Train a population of LLM agents over rollout (generate-and-score) environments.

    Collects token-level episodes then runs turn-level updates. For a
    ``RolloutHarness`` with ``max_model_len`` set, a trajectory whose cumulative
    prompt would overflow the context is stopped with ``truncated=True``.

    :param pop: Population of LLMPPO, LLMREINFORCE or GRPO agents to finetune.
    :type pop: list[SupportedRollout]
    :param max_turns: Maximum interaction turns per episode.
    :type max_turns: int
    :param env_factory: Zero-arg factory that returns a fresh env for each
        trajectory rollout. Required to ensure trajectory state isolation.
    :type env_factory: Callable[[], RolloutHarness]
    :param init_hp: Initial hyperparameters.
    :type init_hp: dict, optional
    :param run: Loop, evolution, checkpoint, and logging settings.
    :type run: LLMRolloutRunConfig, optional
    :param accelerator: Hugging Face Accelerate instance, defaults to None.
    :type accelerator: Accelerator, optional
    :return: The finetuned population and its last recorded fitnesses.
    :rtype: tuple[list[SupportedRollout], Any]
    """
    run = _resolve_rollout_run(run)
    session = _start_llm_rollout(
        pop, max_turns, env_factory, init_hp, run, accelerator
    )
    try:
        _run_llm_rollout_loop(session)
        save_llm_elite_if_requested(session.population.agents, session.run.checkpoint)
    finally:
        _close_llm_rollout(session)
    session.population.finish()
    session.pbar.close()
    return session.population.agents, session.population.last_fitnesses
