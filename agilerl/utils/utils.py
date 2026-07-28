# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    IPPO,
    MADDPG,
    MATD3,
    PPO,
    TD3,
    NeuralTS,
    NeuralUCB,
    RainbowDQN,
)
from agilerl.algorithms.core import EvolvableAlgorithm, LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.logger import CSVLogger, StdOutLogger, TensorboardLogger, WandbLogger
from agilerl.protocols import EvolvableAlgorithmProtocol
from agilerl.typing import BPTTSequenceType, InfosDict, PopulationType
from agilerl.utils.algo_utils import CosineLRScheduleConfig, DummyOptimizer, clone_llm
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from agilerl.algorithms import CISPO, DPO, GRPO, GSPO, LLMPPO, LLMREINFORCE, SFT
    from agilerl.utils.llm_utils import get_llm_accelerator, get_state_dict

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers import PreTrainedTokenizerBase


AgentT = TypeVar("AgentT", bound=EvolvableAlgorithmProtocol)

SupportedObservationSpace = spaces.Box | spaces.Discrete | spaces.Dict | spaces.Tuple

_BOX2D_ENV_PREFIXES = (
    "LunarLander",
    "LunarLanderContinuous",
    "BipedalWalker",
    "BipedalWalkerHardcore",
    "CarRacing",
)


def _check_box2d_available(env_name: str) -> None:
    """Raise a helpful error if a Box2D environment is requested but box2d-py is missing."""
    if not any(env_name.startswith(prefix) for prefix in _BOX2D_ENV_PREFIXES):
        return
    try:
        import Box2D  # noqa: F401
    except ImportError:
        msg = (
            f"Environment '{env_name}' requires the Box2D physics engine.\n"
            "Install it with:  pip install agilerl[box2d]  (or  uv sync --extra box2d)\n"
            "Note: building box2d-py requires the system package 'swig'.\n"
            "  Ubuntu/Debian: sudo apt-get install swig\n"
            "  macOS:         brew install swig\n"
            "Alternatively, use a non-Box2D environment such as 'CartPole-v1'."
        )
        raise ImportError(msg) from None


def _normalize_algo_name(algo: str) -> str | None:
    """Map config-style names to internal algorithm keys."""
    return algo.upper().replace(" ", "").replace("-", "_")


def _lora_config_from_init_hp(INIT_HP: dict[str, Any]) -> "LoraConfig | None":
    """Build a ``peft.LoraConfig`` from INIT_HP keys, or return None."""
    modules = INIT_HP.get("LORA_TARGET_MODULES") or INIT_HP.get("TARGET_MODULES")
    if not modules:
        return None
    if not HAS_LLM_DEPENDENCIES:
        return None
    from peft import LoraConfig

    if isinstance(modules, str):
        modules = [modules]
    bias = str(INIT_HP.get("LORA_BIAS", "none"))
    assert bias in ("none", "all", "lora_only"), (
        f"LORA_BIAS must be one of 'none', 'all', 'lora_only'; got {bias!r}."
    )
    return LoraConfig(
        r=int(INIT_HP.get("LORA_R", 16)),
        lora_alpha=int(INIT_HP.get("LORA_ALPHA", 64)),
        target_modules=list(modules),
        lora_dropout=float(INIT_HP.get("LORA_DROPOUT", 0.0)),
        bias=bias,
        task_type=str(INIT_HP.get("LORA_TASK_TYPE", "CAUSAL_LM")),
    )


def _prepare_llm_algo_kwargs(
    algo_kwargs: dict[str, Any],
    *,
    tokenizer: "PreTrainedTokenizerBase | None",
    model_name: str | None,
    lora_config: object | None,
    vllm_config: object | None,
    INIT_HP: dict[str, Any],
    with_generation_defaults: bool = True,
) -> dict[str, Any]:
    """Merge tokenizer / model / LoRA defaults into ``algo_kwargs``.

    When ``with_generation_defaults`` is True (GRPO / LLMPPO / LLMREINFORCE), also
    merges vLLM-related defaults. When False (DPO), skips generation stack keys so
    callers do not inherit ``use_vllm`` / ``vllm_config``; reference-adapter
    default matches offline preference training (False unless ``INIT_HP`` says
    otherwise).
    """
    merged = dict(algo_kwargs)
    if tokenizer is not None:
        merged.setdefault("pad_token_id", tokenizer.pad_token_id)
        merged.setdefault("pad_token", tokenizer.pad_token)
    if model_name is not None:
        merged.setdefault("model_name", model_name)
    if INIT_HP.get("MODEL_NAME"):
        merged.setdefault("model_name", INIT_HP["MODEL_NAME"])
    if lora_config is not None:
        merged.setdefault("lora_config", lora_config)
    if merged.get("lora_config") is None:
        built = _lora_config_from_init_hp(INIT_HP)
        if built is not None:
            merged["lora_config"] = built
    if with_generation_defaults:
        if vllm_config is not None:
            merged.setdefault("vllm_config", vllm_config)
        merged.setdefault("use_vllm", bool(INIT_HP.get("USE_VLLM", False)))
        merged.setdefault(
            "use_separate_reference_adapter",
            INIT_HP.get("USE_SEPARATE_REFERENCE_ADAPTER", True),
        )
    else:
        merged.setdefault(
            "use_separate_reference_adapter",
            INIT_HP.get("USE_SEPARATE_REFERENCE_ADAPTER", False),
        )
    if "micro_batch_size_per_gpu" not in merged:
        batch_size = INIT_HP.get("BATCH_SIZE", 16)
        merged["micro_batch_size_per_gpu"] = INIT_HP.get(
            "MICRO_BATCH_SIZE_PER_GPU",
            batch_size,
        )  # NOTE we should take a look into deepspeed auto batch-sizing
    # Plain passthroughs: (merged_key, init_hp_key, caster, present_when_truthy).
    # reduce_memory_peak/activation_offload fire on key membership (so an explicit
    # False is honoured); lora_target_scope/chunk_rows fire only on a truthy value.
    _passthroughs = (
        ("reduce_memory_peak", "REDUCE_MEMORY_PEAK", bool, False),
        ("activation_offload", "ACTIVATION_OFFLOAD", bool, False),
        ("lora_target_scope", "LORA_TARGET_SCOPE", lambda v: v, True),
        ("chunk_rows", "CHUNK_ROWS", int, True),
    )
    for merged_key, init_hp_key, caster, present_when_truthy in _passthroughs:
        present = (
            bool(INIT_HP.get(init_hp_key))
            if present_when_truthy
            else init_hp_key in INIT_HP
        )
        if merged_key not in merged and present:
            merged[merged_key] = caster(INIT_HP[init_hp_key])
    # Trainer-side bitsandbytes quantization driven from config / INIT_HP.
    # An explicit quantization_config in algo_kwargs always wins; otherwise a
    # QUANTIZATION preset name or BitsAndBytesConfig kwargs dict is resolved.
    if "quantization_config" not in merged and INIT_HP.get("QUANTIZATION") is not None:
        from agilerl.utils.llm_utils import build_bnb_quantization_config

        quant_config = build_bnb_quantization_config(INIT_HP["QUANTIZATION"])
        if quant_config is not None:
            merged["quantization_config"] = quant_config
    # ATTN_IMPLEMENTATION: inject a non-"auto" value into model_config so the
    # algorithm's create_model call treats it as authoritative (overrides the
    # auto-pick and legacy AGILERL_ATTN_IMPLEMENTATION env var); "auto"/absent
    # leaves model_config alone so the auto-pick path still runs.
    attn_impl = INIT_HP.get("ATTN_IMPLEMENTATION")
    if attn_impl and attn_impl != "auto":
        mc = dict(merged.get("model_config") or {})
        mc.setdefault("attn_implementation", attn_impl)
        merged["model_config"] = mc
    return merged


def _validate_llm_kwargs(
    merged: dict[str, Any], *, actor_network: object | None
) -> None:
    if merged.get("pad_token_id") is None or merged.get("pad_token") is None:
        msg = (
            "LLM agents require pad_token_id and pad_token; pass tokenizer= to "
            "create_population or set them in algo_kwargs."
        )
        raise ValueError(msg)
    if merged.get("model_name") is None and actor_network is None:
        msg = (
            "LLM agents require model_name or actor_network; pass model_name=, "
            "set MODEL_NAME in INIT_HP, or pass actor_network=."
        )
        raise ValueError(msg)


def make_vect_envs(
    env_name: str | None = None,
    num_envs: int = 1,
    *,
    make_env: Callable[..., Any] | None = None,
    should_async_vector: bool = True,
    extra_wrappers: list[type] | None = None,
    **env_kwargs: Any,
) -> gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv:
    """Return async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param make_env: Function that creates a gym environment, defaults use gym.make(env_name)
    :type make_env: Callable, optional
    :param should_async_vector: Whether to asynchronous vectorized environments, defaults to True
    :type should_async_vector: bool, optional
    :param extra_wrappers: Optional list of wrapper classes to apply to each individual
        environment before vectorization.
    :type extra_wrappers: list[type] or None, optional
    :return: Vectorized gym environments
    :rtype: gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv
    """
    if env_name is not None:
        _check_box2d_available(env_name)

    vectorize = (
        gym.vector.AsyncVectorEnv if should_async_vector else gym.vector.SyncVectorEnv
    )

    if make_env is None:
        if env_name is None:
            msg = "Either env_name or make_env must be provided"
            raise ValueError(msg)
        env_id: str = env_name

        def make_env() -> gym.Env:
            return gym.make(env_id, **env_kwargs)

    if extra_wrappers is not None:
        _inner_make_env = make_env

        def make_env() -> gym.Env:
            env = _inner_make_env()
            for wrapper_cls in extra_wrappers:
                env = wrapper_cls(env)
            return env

    return vectorize([make_env for _ in range(num_envs)])


def make_multi_agent_vect_envs(
    env: Callable[..., ParallelEnv],
    num_envs: int = 1,
    *,
    extra_wrappers: list[type] | None = None,
    **env_kwargs: Any,
) -> AsyncPettingZooVecEnv:
    """Return async-vectorized PettingZoo parallel environments.

    :param env: PettingZoo parallel environment object
    :type env: pettingzoo.utils.env.ParallelEnv
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param extra_wrappers: Optional list of wrapper classes to apply to each individual
        environment before vectorization.
    :type extra_wrappers: list[type] or None, optional

    :return: Async-vectorized PettingZoo parallel environments
    :rtype: agilerl.vector.pz_async_vec_env.AsyncPettingZooVecEnv
    """
    if extra_wrappers is not None:
        _original_env = env

        def env(**kwargs: Any) -> ParallelEnv:
            e = _original_env(**kwargs)
            for wrapper_cls in extra_wrappers:
                e = wrapper_cls(e)
            return e

    env_fns = [lambda: env(**env_kwargs) for _ in range(num_envs)]
    return AsyncPettingZooVecEnv(env_fns=env_fns)


def make_skill_vect_envs(
    env_name: str,
    skill: Callable[..., gym.Env],
    num_envs: int = 1,
) -> gym.vector.AsyncVectorEnv:
    """Return async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param skill: Skill wrapper to apply to environment
    :type skill: agilerl.wrappers.learning.Skill
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: skill(gym.make(env_name)) for i in range(num_envs)],
    )


def suppress_verbose_logging() -> None:
    """Suppress verbose logging from DeepSpeed, Accelerate, and related libraries."""
    # Suppress DeepSpeed logging
    logging.getLogger("deepspeed").setLevel(logging.WARNING)

    # Suppress Accelerate logging
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    # Suppress specific DeepSpeed components
    logging.getLogger("deepspeed.runtime.engine").setLevel(logging.WARNING)
    logging.getLogger("deepspeed.runtime.zero").setLevel(logging.WARNING)
    logging.getLogger("deepspeed.checkpoint").setLevel(logging.WARNING)

    # Suppress JAX logging (if used)
    logging.getLogger("jax").setLevel(logging.WARNING)

    # Set root logger to INFO to avoid suppressing important messages
    logging.getLogger().setLevel(logging.INFO)


def default_progress_bar(
    max_steps: int,
    accelerator: Accelerator | None = None,
) -> tqdm.tqdm:
    """Return a default progress bar.

    :param max_steps: Maximum number of steps
    :type max_steps: int
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :return: Progress bar
    :rtype: tqdm.tqdm
    """
    bar_format = (
        "Training Progress │ "
        "{percentage:3.0f}% │ "
        "{bar:20} │ "
        "{n_fmt}/{total_fmt} steps │ "
        "⏱️ {elapsed} │ "
        "⏳ {remaining} │ "
        "{rate_fmt}"
        "{postfix}"
    )
    disable = (
        not accelerator.is_local_main_process if accelerator is not None else False
    )
    return tqdm.trange(
        max_steps,
        unit="step",
        bar_format=bar_format,
        ascii=False,
        dynamic_ncols=True,
        disable=disable,
    )


def create_population(
    algo: str,
    net_config: dict[str, Any] | None,
    INIT_HP: dict[str, Any],
    # The accepted space and network types depend on the ``algo`` string
    # (single spaces/networks vs per-agent dicts/lists); each algorithm
    # constructor validates them at runtime.
    observation_space: Any = None,  # noqa: ANN401 -- space type varies per algo (single space vs per-agent dict/list)
    action_space: Any = None,  # noqa: ANN401 -- space type varies per algo (single space vs per-agent dict/list)
    hp_config: HyperparameterConfig | None = None,
    actor_network: Any = None,  # noqa: ANN401 -- network type varies per algo constructor; state_dict() read on LLM path
    critic_network: Any = None,  # noqa: ANN401 -- network type varies per algo constructor
    agent_wrapper: Callable | None = None,
    wrapper_kwargs: dict[str, Any] | None = None,
    population_size: int = 1,
    num_envs: int = 1,
    device: str = "cpu",
    accelerator: Accelerator | None = None,
    torch_compiler: str | None = None,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    model_name: str | None = None,
    lora_config: object | None = None,
    vllm_config: object | None = None,
    algo_kwargs: dict[str, Any] | None = None,
) -> PopulationType:
    """Return population of identical agents.

    .. deprecated::
        Use ``Algorithm.population()`` instead (e.g. ``DQN.population(size=4, ...)``).

    :param algo: RL algorithm
    :type algo: str
    :param net_config: Network configuration
    :type net_config: dict or None
    :param INIT_HP: Initial hyperparameters
    :type INIT_HP: dict
    :param observation_space: Observation space
    :type observation_space: spaces.Space
    :param action_space: Action space
    :type action_space: spaces.Space
    :param hp_config: Choice of algorithm hyperparameters to mutate during training, defaults to None
    :type hp_config: HyperparameterConfig, optional
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
    :param torch_compiler: Torch compiler, defaults to None
    :type torch_compiler: Any, optional
    :param tokenizer: Hugging Face tokenizer; used to default ``pad_token_id`` /
        ``pad_token`` for GRPO / DPO / LLMPPO / LLMREINFORCE when not set in ``algo_kwargs``.
    :type tokenizer: PreTrainedTokenizerBase, optional
    :param model_name: HF model id or path; defaults ``algo_kwargs['model_name']``
        or ``INIT_HP['MODEL_NAME']`` for LLM agents.
    :type model_name: str, optional
    :param lora_config: ``peft.LoraConfig``; if omitted, built from
        ``LORA_*`` / ``TARGET_MODULES`` keys in ``INIT_HP`` when present.
    :type lora_config: Any, optional
    :param vllm_config: ``VLLMConfig`` for GRPO / LLMPPO / LLMREINFORCE (ignored for DPO).
    :type vllm_config: Any, optional
    :param algo_kwargs: Additional keyword arguments for the algorithm
    :type algo_kwargs: dict, optional
    :return: Population of agents
    :rtype: list[EvolvableAlgorithm]
    """
    algo_name = algo.replace(" ", "")
    warnings.warn(
        f"create_population() is deprecated. Use {algo_name}.population() instead "
        f"(e.g. {algo_name}.population(size=4, observation_space=obs, action_space=act, ...)).",
        DeprecationWarning,
        stacklevel=2,
    )
    if algo_kwargs is None:
        algo_kwargs = {}
    population: PopulationType = []
    if algo == "DQN":
        for idx in range(population_size):
            agent = DQN(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                double=INIT_HP.get("DOUBLE", False),
                cudagraphs=INIT_HP.get("CUDAGRAPHS", False),
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo == "Rainbow DQN":
        for idx in range(population_size):
            agent = RainbowDQN(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                beta=INIT_HP.get("BETA", 0.4),
                prior_eps=INIT_HP.get("PRIOR_EPS", 0.00001),
                num_atoms=INIT_HP.get("NUM_ATOMS", 51),
                v_min=INIT_HP.get("V_MIN", -100),
                v_max=INIT_HP.get("V_MAX", 100),
                n_step=INIT_HP.get("N_STEP", 3),
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo == "DDPG":
        for idx in range(population_size):
            agent = DDPG(
                observation_space=observation_space,
                action_space=action_space,
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                policy_freq=INIT_HP.get("POLICY_FREQ", 2),
                actor_network=actor_network,
                critic_network=critic_network,
                share_encoders=INIT_HP.get("SHARE_ENCODERS", True),
                device=device,
                accelerator=accelerator,
                **algo_kwargs,
            )

            agent = (
                agent_wrapper(agent, **(wrapper_kwargs or {}))
                if agent_wrapper is not None
                else agent
            )
            population.append(agent)

    elif algo == "PPO":
        for idx in range(population_size):
            agent = PPO(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 2048),
                gamma=INIT_HP.get("GAMMA", 0.99),
                gae_lambda=INIT_HP.get("GAE_LAMBDA", 0.95),
                action_std_init=INIT_HP.get("ACTION_STD_INIT", 0.6),
                clip_coef=INIT_HP.get("CLIP_COEF", 0.2),
                ent_coef=INIT_HP.get("ENT_COEF", 0.01),
                vf_coef=INIT_HP.get("VF_COEF", 0.5),
                max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 0.5),
                target_kl=INIT_HP.get("TARGET_KL"),
                update_epochs=INIT_HP.get("UPDATE_EPOCHS", 4),
                share_encoders=INIT_HP.get("SHARE_ENCODERS", True),
                recurrent=INIT_HP.get("RECURRENT", False),
                bptt_sequence_type=INIT_HP.get(
                    "BPTT_SEQUENCE_TYPE",
                    BPTTSequenceType.CHUNKED,
                ),
                max_seq_len=INIT_HP.get("MAX_SEQ_LEN"),
                actor_network=actor_network,
                critic_network=critic_network,
                device=device,
                accelerator=accelerator,
                num_envs=num_envs,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo == "CQN":
        for idx in range(population_size):
            agent = CQN(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                double=INIT_HP.get("DOUBLE", False),
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo == "TD3":
        for idx in range(population_size):
            agent = TD3(
                observation_space=observation_space,
                action_space=action_space,
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.005),
                policy_freq=INIT_HP.get("POLICY_FREQ", 2),
                actor_network=actor_network,
                critic_networks=critic_network,
                share_encoders=INIT_HP.get("SHARE_ENCODERS", True),
                device=device,
                accelerator=accelerator,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo == "MADDPG":
        for idx in range(population_size):
            agent = MADDPG(
                observation_spaces=observation_space,
                action_spaces=action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.95),
                tau=INIT_HP.get("TAU", 0.01),
                actor_networks=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
                torch_compiler=torch_compiler,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo == "MATD3":
        for idx in range(population_size):
            agent = MATD3(
                observation_spaces=observation_space,
                action_spaces=action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                policy_freq=INIT_HP.get("POLICY_FREQ", 2),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.95),
                tau=INIT_HP.get("TAU", 0.01),
                actor_networks=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
                torch_compiler=torch_compiler,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo == "IPPO":
        for idx in range(population_size):
            agent = IPPO(
                observation_spaces=observation_space,
                action_spaces=action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 2048),
                gamma=INIT_HP.get("GAMMA", 0.99),
                gae_lambda=INIT_HP.get("GAE_LAMBDA", 0.95),
                action_std_init=INIT_HP.get("ACTION_STD_INIT", 0.0),
                clip_coef=INIT_HP.get("CLIP_COEF", 0.2),
                ent_coef=INIT_HP.get("ENT_COEF", 0.01),
                vf_coef=INIT_HP.get("VF_COEF", 0.5),
                max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 0.5),
                target_kl=INIT_HP.get("TARGET_KL"),
                update_epochs=INIT_HP.get("UPDATE_EPOCHS", 4),
                actor_networks=actor_network,
                critic_networks=critic_network,
                action_batch_size=INIT_HP.get("ACTION_BATCH_SIZE"),
                device=device,
                accelerator=accelerator,
                torch_compiler=torch_compiler,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo == "NeuralUCB":
        for idx in range(population_size):
            agent = NeuralUCB(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                gamma=INIT_HP.get("GAMMA", 1),
                lamb=INIT_HP.get("LAMBDA", 1),
                reg=INIT_HP.get("REG", 0.000625),
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 2),
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo == "NeuralTS":
        for idx in range(population_size):
            agent = NeuralTS(
                observation_space=observation_space,
                action_space=action_space,
                index=idx,
                hp_config=hp_config,
                net_config=net_config,
                gamma=INIT_HP.get("GAMMA", 1),
                lamb=INIT_HP.get("LAMBDA", 1),
                reg=INIT_HP.get("REG", 0.000625),
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.003),
                learn_step=INIT_HP.get("LEARN_STEP", 2),
                actor_network=actor_network,
                device=device,
                accelerator=accelerator,
                **algo_kwargs,
            )
            population.append(agent)

    elif algo in ("GRPO", "CISPO", "GSPO"):
        if not HAS_LLM_DEPENDENCIES:
            msg = "GRPO/CISPO/GSPO require optional LLM dependencies (install agilerl[llm])."
            raise ImportError(msg)

        kwargs = _prepare_llm_algo_kwargs(
            algo_kwargs,
            tokenizer=tokenizer,
            model_name=model_name,
            lora_config=lora_config,
            vllm_config=vllm_config,
            INIT_HP=INIT_HP,
        )
        _validate_llm_kwargs(kwargs, actor_network=actor_network)
        cosine_cfg = INIT_HP.get("COSINE_lR_SCHEDULER")
        cosine_lr = (
            CosineLRScheduleConfig(**cosine_cfg) if cosine_cfg is not None else None
        )
        for idx in range(population_size):
            agent_accelerator = get_llm_accelerator(accelerator, idx)
            act = (
                (
                    clone_llm(
                        actor_network,
                        zero_stage=INIT_HP.get("ZERO_STAGE", 0),
                        state_dict=(
                            actor_network.state_dict()
                            if accelerator is None
                            else get_state_dict(actor_network)
                        ),
                    )
                    if idx != 0
                    else actor_network
                )
                if actor_network is not None
                else None
            )
            kw = dict(kwargs)
            kw.update(
                hp_config=hp_config,
                index=idx,
                batch_size=INIT_HP.get("BATCH_SIZE", 16),
                beta=INIT_HP.get("BETA", 0.001),
                lr=INIT_HP.get("LR", 5e-7),
                clip_coef=INIT_HP.get("CLIP_COEF", 0.2),
                max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 0.1),
                update_epochs=INIT_HP.get("UPDATE_EPOCHS", 1),
                group_size=INIT_HP.get("GROUP_SIZE", 8),
                temperature=INIT_HP.get("TEMPERATURE", 0.9),
                repetition_penalty=INIT_HP.get("REPETITION_PENALTY", 1.0),
                top_p=INIT_HP.get("TOP_P", 0.95),
                top_k=INIT_HP.get("TOP_K", 50),
                min_p=INIT_HP.get("MIN_P", 0.0),
                use_memory_efficient_params=INIT_HP.get(
                    "USE_MEMORY_EFFICIENT_PARAMS", True
                ),
                calc_position_embeddings=INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
                max_output_tokens=INIT_HP.get("MAX_OUTPUT_TOKENS"),
                min_output_tokens=INIT_HP.get("MIN_OUTPUT_TOKENS"),
                max_model_len=INIT_HP.get("MAX_MODEL_LEN", 1024),
                cosine_lr_schedule_config=cosine_lr,
                accelerator=agent_accelerator,
                gradient_checkpointing=INIT_HP.get("GRADIENT_CHECKPOINTING", True),
                actor_network=act,
                # Agents after the first receive a clone_llm copy that already
                # carries AgileRL's adapters — construct them via the clone
                # path (reuse as-is) rather than re-attaching adapters.
                clone=idx != 0 and act is not None,
                seed=INIT_HP.get("SEED", 42),
                use_liger_loss=INIT_HP.get("USE_LIGER_LOSS", False),
                cast_logprobs_to_fp32=INIT_HP.get("CAST_LOGPROBS_TO_FP32", True),
                use_kl_advantage_shaping=INIT_HP.get("USE_KL_ADVANTAGE_SHAPING", False),
                adv_norm=INIT_HP.get("ADV_NORM", "mean_std"),
                loss_type=INIT_HP.get("LOSS_TYPE", "grpo"),
                # ``None`` (no config key) lets GRPO resolve the default per
                # loss_type — "token", or "trajectory" for gspo — without
                # tripping the explicit-override warning.
                importance_sampling_level=INIT_HP.get("IMPORTANCE_SAMPLING_LEVEL"),
                advantage_granularity=INIT_HP.get(
                    "ADVANTAGE_GRANULARITY", INIT_HP.get("ACTION_GRANULARITY", "auto")
                ),
                whiten_advantages=INIT_HP.get("WHITEN_ADVANTAGES", False),
                adv_clip_range=INIT_HP.get("ADV_CLIP_RANGE"),
                filter_zero_adv=INIT_HP.get(
                    "FILTER_ZERO_ADV",
                    INIT_HP.get("FILTER_ZERO_ADVANTAGES", False),
                ),
                adv_filter_eps=INIT_HP.get(
                    "ADV_FILTER_EPS",
                    INIT_HP.get("ADVANTAGE_FILTER_EPS", 0.0),
                ),
                vllm_importance_sampling_correction=INIT_HP.get(
                    "VLLM_IMPORTANCE_SAMPLING_CORRECTION", True
                ),
                vllm_importance_sampling_cap=INIT_HP.get(
                    "VLLM_IMPORTANCE_SAMPLING_CAP", 2.0
                ),
                use_sequence_packing=INIT_HP.get("USE_SEQUENCE_PACKING", False),
            )
            if torch_compiler is not None:
                kw.setdefault("torch_compiler", torch_compiler)
            algo_cls = {"GRPO": GRPO, "CISPO": CISPO, "GSPO": GSPO}[algo]
            if algo in ("CISPO", "GSPO"):
                kw.pop("loss_type", None)
            agent = algo_cls(**kw)
            population.append(agent)
    elif algo == "SFT":
        if not HAS_LLM_DEPENDENCIES:
            msg = "SFT requires optional LLM dependencies (install agilerl[llm])."
            raise ImportError(msg)

        kwargs = _prepare_llm_algo_kwargs(
            algo_kwargs,
            tokenizer=tokenizer,
            model_name=model_name,
            lora_config=lora_config,
            vllm_config=vllm_config,
            INIT_HP=INIT_HP,
            with_generation_defaults=False,
        )
        _validate_llm_kwargs(kwargs, actor_network=actor_network)
        kwargs.pop("use_vllm", None)
        kwargs.pop("vllm_config", None)
        kwargs.pop("use_separate_reference_adapter", None)

        for idx in range(population_size):
            agent_accelerator = get_llm_accelerator(accelerator, idx)
            act = (
                (
                    clone_llm(
                        actor_network,
                        zero_stage=INIT_HP.get("ZERO_STAGE", 0),
                        state_dict=(
                            actor_network.state_dict()
                            if accelerator is None
                            else get_state_dict(actor_network)
                        ),
                    )
                    if idx != 0
                    else actor_network
                )
                if actor_network is not None
                else None
            )
            kw = dict(kwargs)
            kw.update(
                hp_config=hp_config,
                index=idx,
                batch_size=INIT_HP.get("BATCH_SIZE", 16),
                lr=INIT_HP.get("LR", 5e-5),
                max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 0.1),
                update_epochs=INIT_HP.get("UPDATE_EPOCHS", 1),
                calc_position_embeddings=INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
                accelerator=agent_accelerator,
                gradient_checkpointing=INIT_HP.get("GRADIENT_CHECKPOINTING", True),
                actor_network=act,
                clone=idx != 0 and act is not None,
                seed=INIT_HP.get("SEED", 42),
                use_liger_loss=INIT_HP.get("USE_LIGER_LOSS", False),
            )
            if torch_compiler is not None:
                kw.setdefault("torch_compiler", torch_compiler)
            agent = SFT(**kw)
            population.append(agent)
    elif algo == "DPO":
        if not HAS_LLM_DEPENDENCIES:
            msg = "DPO requires optional LLM dependencies (install agilerl[llm])."
            raise ImportError(msg)

        kwargs = _prepare_llm_algo_kwargs(
            algo_kwargs,
            tokenizer=tokenizer,
            model_name=model_name,
            lora_config=lora_config,
            vllm_config=vllm_config,
            INIT_HP=INIT_HP,
            with_generation_defaults=False,
        )
        _validate_llm_kwargs(kwargs, actor_network=actor_network)
        kwargs.pop("use_vllm", None)
        kwargs.pop("vllm_config", None)

        for idx in range(population_size):
            agent_accelerator = get_llm_accelerator(accelerator, idx)
            act = (
                (
                    clone_llm(
                        actor_network,
                        zero_stage=INIT_HP.get("ZERO_STAGE", 0),
                        state_dict=(
                            actor_network.state_dict()
                            if accelerator is None
                            else get_state_dict(actor_network)
                        ),
                    )
                    if idx != 0
                    else actor_network
                )
                if actor_network is not None
                else None
            )
            kw = dict(kwargs)
            kw.update(
                hp_config=hp_config,
                index=idx,
                batch_size=INIT_HP.get("BATCH_SIZE", 16),
                beta=INIT_HP.get("BETA", 0.001),
                lr=INIT_HP.get("LR", 0.000005),
                max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 0.1),
                update_epochs=INIT_HP.get("UPDATE_EPOCHS", 1),
                calc_position_embeddings=INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
                accelerator=agent_accelerator,
                gradient_checkpointing=INIT_HP.get("GRADIENT_CHECKPOINTING", True),
                actor_network=act,
                clone=idx != 0 and act is not None,
                seed=INIT_HP.get("SEED", 42),
                use_liger_loss=INIT_HP.get("USE_LIGER_LOSS", False),
            )
            if torch_compiler is not None:
                kw.setdefault("torch_compiler", torch_compiler)
            agent = DPO(**kw)
            population.append(agent)

    elif _normalize_algo_name(algo) == "LLMPPO":
        if not HAS_LLM_DEPENDENCIES:
            msg = "LLMPPO requires optional LLM dependencies (install agilerl[llm])."
            raise ImportError(msg)

        kwargs = _prepare_llm_algo_kwargs(
            algo_kwargs,
            tokenizer=tokenizer,
            model_name=model_name,
            lora_config=lora_config,
            vllm_config=vllm_config,
            INIT_HP=INIT_HP,
        )
        _validate_llm_kwargs(kwargs, actor_network=actor_network)
        cosine_cfg = INIT_HP.get("COSINE_lR_SCHEDULER")
        cosine_lr = (
            CosineLRScheduleConfig(**cosine_cfg) if cosine_cfg is not None else None
        )
        for idx in range(population_size):
            agent_accelerator = get_llm_accelerator(accelerator, idx)
            act = (
                (
                    clone_llm(
                        actor_network,
                        zero_stage=INIT_HP.get("ZERO_STAGE", 0),
                        state_dict=(
                            actor_network.state_dict()
                            if accelerator is None
                            else get_state_dict(actor_network)
                        ),
                    )
                    if idx != 0
                    else actor_network
                )
                if actor_network is not None
                else None
            )
            kw = dict(kwargs)
            kw.update(
                hp_config=hp_config,
                index=idx,
                batch_size=INIT_HP.get("BATCH_SIZE", 16),
                beta=INIT_HP.get("BETA", 0.01),
                vf_coef=INIT_HP.get("VF_COEF", 0.5),
                clip_coef=INIT_HP.get("CLIP_COEF", 0.2),
                gamma=INIT_HP.get("GAMMA", 1.0),
                gae_lambda=INIT_HP.get("GAE_LAMBDA", 1.0),
                advantage_granularity=INIT_HP.get(
                    "ADVANTAGE_GRANULARITY", INIT_HP.get("ACTION_GRANULARITY", "auto")
                ),
                importance_sampling_level=INIT_HP.get(
                    "IMPORTANCE_SAMPLING_LEVEL", "auto"
                ),
                turn_ratio_pooling=INIT_HP.get("TURN_RATIO_POOLING", "sum"),
                turn_level_clip=INIT_HP.get("TURN_LEVEL_CLIP", True),
                turn_value_reduction=INIT_HP.get("TURN_VALUE_REDUCTION", "final_value"),
                whiten_advantages=INIT_HP.get("WHITEN_ADVANTAGES", True),
                chunk_rows=INIT_HP.get("CHUNK_ROWS"),
                lr_actor=INIT_HP.get("LR_ACTOR", INIT_HP.get("LR", 5e-6)),
                lr_critic=INIT_HP.get("LR_CRITIC"),
                max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 1.0),
                update_epochs=INIT_HP.get("UPDATE_EPOCHS", 1),
                temperature=INIT_HP.get("TEMPERATURE", 1.0),
                max_output_tokens=INIT_HP.get("MAX_OUTPUT_TOKENS"),
                min_output_tokens=INIT_HP.get("MIN_OUTPUT_TOKENS"),
                max_model_len=INIT_HP.get("MAX_MODEL_LEN", 1024),
                use_memory_efficient_params=INIT_HP.get(
                    "USE_MEMORY_EFFICIENT_PARAMS", True
                ),
                calc_position_embeddings=INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
                cosine_lr_schedule_config=cosine_lr,
                accelerator=agent_accelerator,
                gradient_checkpointing=INIT_HP.get("GRADIENT_CHECKPOINTING", True),
                actor_network=act,
                clone=idx != 0 and act is not None,
                seed=INIT_HP.get("SEED", 42),
                use_liger_loss=INIT_HP.get("USE_LIGER_LOSS", False),
                cast_logprobs_to_fp32=INIT_HP.get("CAST_LOGPROBS_TO_FP32", True),
                vllm_importance_sampling_correction=INIT_HP.get(
                    "VLLM_IMPORTANCE_SAMPLING_CORRECTION", True
                ),
                vllm_importance_sampling_cap=INIT_HP.get(
                    "VLLM_IMPORTANCE_SAMPLING_CAP", 2.0
                ),
            )
            if torch_compiler is not None:
                kw.setdefault("torch_compiler", torch_compiler)
            agent = LLMPPO(**kw)
            population.append(agent)

    elif _normalize_algo_name(algo) == "LLMREINFORCE":
        if not HAS_LLM_DEPENDENCIES:
            msg = (
                "LLMREINFORCE requires optional LLM dependencies "
                "(install agilerl[llm])."
            )
            raise ImportError(msg)

        kwargs = _prepare_llm_algo_kwargs(
            algo_kwargs,
            tokenizer=tokenizer,
            model_name=model_name,
            lora_config=lora_config,
            vllm_config=vllm_config,
            INIT_HP=INIT_HP,
        )
        _validate_llm_kwargs(kwargs, actor_network=actor_network)
        cosine_cfg = INIT_HP.get("COSINE_lR_SCHEDULER")
        cosine_lr = (
            CosineLRScheduleConfig(**cosine_cfg) if cosine_cfg is not None else None
        )
        for idx in range(population_size):
            agent_accelerator = get_llm_accelerator(accelerator, idx)
            act = (
                (
                    clone_llm(
                        actor_network,
                        zero_stage=INIT_HP.get("ZERO_STAGE", 0),
                        state_dict=(
                            actor_network.state_dict()
                            if accelerator is None
                            else get_state_dict(actor_network)
                        ),
                    )
                    if idx != 0
                    else actor_network
                )
                if actor_network is not None
                else None
            )
            kw = dict(kwargs)
            kw.update(
                hp_config=hp_config,
                index=idx,
                batch_size=INIT_HP.get("BATCH_SIZE", 16),
                beta=INIT_HP.get("BETA", 0.01),
                clip_coef=INIT_HP.get("CLIP_COEF", 0.2),
                gamma=INIT_HP.get("GAMMA", 0.99),
                advantage_granularity=INIT_HP.get(
                    "ADVANTAGE_GRANULARITY", INIT_HP.get("ACTION_GRANULARITY", "auto")
                ),
                importance_sampling_level=INIT_HP.get(
                    "IMPORTANCE_SAMPLING_LEVEL", "token"
                ),
                turn_ratio_pooling=INIT_HP.get("TURN_RATIO_POOLING", "sum"),
                lr=INIT_HP.get("LR", 5e-7),
                max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 1.0),
                update_epochs=INIT_HP.get("UPDATE_EPOCHS", 1),
                temperature=INIT_HP.get("TEMPERATURE", 1.0),
                max_output_tokens=INIT_HP.get("MAX_OUTPUT_TOKENS"),
                min_output_tokens=INIT_HP.get("MIN_OUTPUT_TOKENS"),
                max_model_len=INIT_HP.get("MAX_MODEL_LEN"),
                calc_position_embeddings=INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
                use_memory_efficient_params=INIT_HP.get(
                    "USE_MEMORY_EFFICIENT_PARAMS", True
                ),
                cosine_lr_schedule_config=cosine_lr,
                accelerator=agent_accelerator,
                gradient_checkpointing=INIT_HP.get("GRADIENT_CHECKPOINTING", True),
                actor_network=act,
                clone=idx != 0 and act is not None,
                seed=INIT_HP.get("SEED", 42),
                use_liger_loss=INIT_HP.get("USE_LIGER_LOSS", False),
                cast_logprobs_to_fp32=INIT_HP.get("CAST_LOGPROBS_TO_FP32", True),
                vllm_importance_sampling_correction=INIT_HP.get(
                    "VLLM_IMPORTANCE_SAMPLING_CORRECTION", True
                ),
                vllm_importance_sampling_cap=INIT_HP.get(
                    "VLLM_IMPORTANCE_SAMPLING_CAP", 2.0
                ),
            )
            if torch_compiler is not None:
                kw.setdefault("torch_compiler", torch_compiler)
            agent = LLMREINFORCE(**kw)
            population.append(agent)
    return population


def save_population_checkpoint(
    population: list[AgentT],
    save_path: str,
    overwrite_checkpoints: bool,
    accelerator: Accelerator | None = None,
) -> None:
    """Save checkpoint of population of agents.

    :param population: Population of agents
    :type population: list[AgentT]
    :param save_path: Path to save checkpoint
    :type save_path: str
    :param overwrite_checkpoints: Flag to overwrite checkpoints
    :type overwrite_checkpoints: bool
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    """
    if accelerator is not None:
        # Need to unwrap models from acccelerator before saving
        accelerator.wait_for_everyone()
        for model in population:
            model.unwrap_models()
        accelerator.wait_for_everyone()

        # Save checkpoint on main process
        if accelerator.is_main_process:
            for i, agent in enumerate(population):
                current_checkpoint_path = (
                    f"{save_path}_{i}.pt"
                    if overwrite_checkpoints
                    else f"{save_path}_{i}_{agent.steps}.pt"
                )
                agent.save_checkpoint(current_checkpoint_path)
        accelerator.wait_for_everyone()

        # Load models back to accelerator processes
        for model in population:
            model.wrap_models()
        accelerator.wait_for_everyone()
    else:
        # Save checkpoint
        for i, agent in enumerate(population):
            current_checkpoint_path = (
                f"{save_path}_{i}.pt"
                if overwrite_checkpoints
                else f"{save_path}_{i}_{agent.steps}.pt"
            )
            agent.save_checkpoint(current_checkpoint_path)


def tournament_selection_and_mutation(
    population: list[AgentT],
    tournament: TournamentSelection,
    mutation: Mutations,
    env_name: str,
    algo: str | None = None,
    elite_path: str | None = None,
    save_elite: bool = False,
    accelerator: Accelerator | None = None,
    language_model: bool | None = False,
) -> list[AgentT]:
    """Perform tournament selection and mutation on a population of agents.

    :param population: Population of agents
    :type population: list[AgentT]
    :param tournament: Tournament selection object
    :type tournament: TournamentSelection
    :param mutation: Mutation object
    :type mutation: Mutations
    :param env_name: Environment name
    :type env_name: str
    :param elite_path: Path to save elite agent, defaults to None
    :type elite_path: str, optional
    :param save_elite: Flag to save elite agent, defaults to False
    :type save_elite: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param language_model: Flag to indicate if the environment is a language model, defaults to False
    :type language_model: bool, optional
    :return: Population of agents after tournament selection and mutation
    :rtype: list[AgentT]
    """
    if algo is None:
        algo = population[0].__class__.__name__

    if language_model:
        elite, population = tournament.select(population)
        if accelerator is None or accelerator.is_main_process:
            population = mutation.mutation(population)
        if accelerator is not None:
            accelerator.wait_for_everyone()
            # This branch only runs for LLM populations.
            consolidate_mutations(
                [agent for agent in population if isinstance(agent, LLMAlgorithm)]
            )
            accelerator.wait_for_everyone()
        if save_elite:
            assert isinstance(elite, LLMAlgorithm), (
                "LLM checkpoints require an LLMAlgorithm elite."
            )
            save_llm_checkpoint(elite, elite_path)
        return population

    elite = None
    if accelerator is not None:
        # Save temporary models for accelerator processes
        accel_temp_models_path = f"models/{env_name}"
        if accelerator.is_main_process:
            Path(accel_temp_models_path).mkdir(parents=True, exist_ok=True)
        # Need to unwrap models from acccelerator before selecting and mutating
        accelerator.wait_for_everyone()
        for model in population:
            model.unwrap_models()
        accelerator.wait_for_everyone()
        # Perform tournament selection and mutation on main process
        if accelerator.is_main_process:
            elite, population = tournament.select(population)
            population = mutation.mutation(population)
            for pop_i, model in enumerate(population):
                model.save_checkpoint(f"{accel_temp_models_path}/{algo}_{pop_i}.pt")
        accelerator.wait_for_everyone()

        # Load models back to accelerator processes
        if not accelerator.is_main_process:
            for pop_i, model in enumerate(population):
                model.load_checkpoint(f"{accel_temp_models_path}/{algo}_{pop_i}.pt")
        accelerator.wait_for_everyone()

        # Wrap models back to accelerator
        for model in population:
            model.wrap_models()
    else:
        # Perform tournament selection and mutation
        elite, population = tournament.select(population)
        population = mutation.mutation(population)

    if save_elite and elite is not None:
        elite_save_path = (
            elite_path.split(".pt")[0]
            if elite_path is not None
            else f"{env_name}-elite_{algo}"
        )
        elite.save_checkpoint(f"{elite_save_path}.pt")

    return population


def init_wandb(
    algo: str,
    env_name: str,
    init_hyperparams: dict[str, Any] | None = None,
    mutation_hyperparams: dict[str, Any] | None = None,
    wandb_api_key: str | None = None,
    accelerator: Accelerator | None = None,
    project: str = "AgileRL",
    addl_args: dict[str, Any] | None = None,
) -> None:
    """Initialize wandb for logging hyperparameters and run metadata.

    :param algo: RL algorithm
    :type algo: str
    :param env_name: Environment name
    :type env_name: str
    :param init_hyperparams: Initial hyperparameters, defaults to None
    :type init_hyperparams: dict, optional
    :param mutation_hyperparams: Mutation hyperparameters, defaults to None
    :type mutation_hyperparams: dict, optional
    :param wandb_api_key: Wandb API key, defaults to None
    :type wandb_api_key: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param addl_args: Additional kwargs to pass to wandb.init()
    :type addl_args: dict, optional
    """
    api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        warnings.warn(
            "No Weights & Biases API key found (pass wandb_api_key or set the "
            "WANDB_API_KEY environment variable); wandb may prompt interactively "
            "or fail to log.",
            UserWarning,
            stacklevel=2,
        )

    config_dict = {}
    if init_hyperparams is not None:
        config_dict.update(init_hyperparams)
    if mutation_hyperparams is not None:
        config_dict.update(mutation_hyperparams)

    # track hyperparameters and run metadata
    kwargs: dict[str, Any] = {
        "config": config_dict,
        "project": project,  # wandb project where this run will be logged
        "name": "{}-EvoHPO-{}-{}".format(
            env_name,
            algo,
            datetime.now().strftime("%m%d%Y%H%M%S"),
        ),
    }

    if addl_args is not None:
        kwargs.update(addl_args)

    if accelerator is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            wandb.init(**kwargs)
        accelerator.wait_for_everyone()
    else:
        wandb.init(**kwargs)


def init_loggers(
    *,
    algo: str,
    env_name: str,
    pbar: tqdm.tqdm,
    verbose: bool = True,
    wb: bool = False,
    tensorboard: bool = False,
    csv: bool = False,
    tensorboard_log_dir: str | None = None,
    csv_log_dir: str | None = None,
    accelerator: Accelerator | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    init_hyperparams: dict[str, Any] | None = None,
    mutation_hyperparams: dict[str, Any] | None = None,
) -> list:
    """Build the list of loggers for a training run.

    Consolidates the repeated logger-setup block shared by all training loops.

    :param algo: Algorithm name (used for wandb run name and TensorBoard experiment).
    :type algo: str
    :param env_name: Environment name.
    :type env_name: str
    :param pbar: ``tqdm`` progress bar for :class:`~agilerl.logger.StdOutLogger`.
    :type pbar: tqdm.tqdm
    :param verbose: Enable console logging via ``StdOutLogger``, defaults to True.
    :type verbose: bool
    :param wb: Enable Weights & Biases logging, defaults to False.
    :type wb: bool
    :param tensorboard: Enable TensorBoard logging, defaults to False.
    :type tensorboard: bool
    :param csv: Enable CSV logging, defaults to False.
    :type csv: bool
    :param tensorboard_log_dir: Directory for TensorBoard event files, defaults to None.
    :type tensorboard_log_dir: str | None
    :param csv_log_dir: Directory for CSV files, defaults to None.
    :type csv_log_dir: str | None
    :param accelerator: HuggingFace Accelerator for distributed training, defaults to None.
    :type accelerator: Accelerator | None
    :param wandb_api_key: API key for Weights & Biases, defaults to None.
    :type wandb_api_key: str | None
    :param wandb_kwargs: Extra kwargs merged into the ``init_wandb`` call (e.g.
        ``{"project": "AgileRLMultiAgent"}``), defaults to None.
    :type wandb_kwargs: dict[str, Any] | None
    :param init_hyperparams: Initial hyperparameters logged to wandb, defaults to None.
    :type init_hyperparams: dict[str, Any] | None
    :param mutation_hyperparams: Mutation hyperparameters logged to wandb, defaults to None.
    :type mutation_hyperparams: dict[str, Any] | None
    :returns: List of configured :class:`~agilerl.logger.Logger` instances.
    :rtype: list[Logger]
    """
    loggers = []
    if verbose:
        loggers.append(StdOutLogger(pbar, accelerator))

    if wb:
        init_wandb_kwargs: dict[str, Any] = {
            "algo": algo,
            "env_name": env_name,
            "init_hyperparams": init_hyperparams,
            "mutation_hyperparams": mutation_hyperparams,
            "wandb_api_key": wandb_api_key,
            "accelerator": accelerator,
        }
        if wandb_kwargs is not None:
            init_wandb_kwargs.update(wandb_kwargs)
        init_wandb(**init_wandb_kwargs)
        loggers.append(WandbLogger(accelerator))

    if tensorboard:
        experiment_name = f"{algo}-{env_name}"
        loggers.append(
            TensorboardLogger(
                log_dir=tensorboard_log_dir,
                experiment_name=experiment_name,
                accelerator=accelerator,
            )
        )
    if csv:
        if csv_log_dir is None:
            msg = "csv_log_dir must be provided when csv=True"
            raise ValueError(msg)
        loggers.append(CSVLogger(csv_log_dir))

    return loggers


def calculate_vectorized_scores(
    rewards: npt.NDArray,
    terminations: npt.NDArray,
    include_unterminated: bool = False,
    only_first_episode: bool = True,
) -> list[float]:
    """Calculate the vectorized scores for episodes based on rewards and terminations.

    :param rewards: Array of rewards for each environment.
    :type rewards: npt.NDArray
    :param terminations: Array indicating termination points for each environment.
    :type terminations: npt.NDArray
    :param include_unterminated: Whether to include rewards from unterminated episodes, defaults to False.
    :type include_unterminated: bool, optional
    :param only_first_episode: Whether to consider only the first episode, defaults to True.
    :type only_first_episode: bool, optional
    :return: List of episode rewards.
    :rtype: list[float]
    """
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
                rewards[env_index, start_index : termination_index + 1],
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


def print_hyperparams(pop: PopulationType) -> None:
    """Print current hyperparameters of agents in a population and their fitnesses.

    :param pop: Population of agents
    :type pop: list[EvolvableAlgorithm]
    """
    for agent in pop:
        mean_fitness = (
            np.mean([np.mean(f) for f in agent.fitness[-5:]]).item()
            if len(agent.fitness) > 0
            else float("nan")
        )
        attrs = EvolvableAlgorithm.inspect_attributes(agent)
        lines = [
            f"Agent ID: {agent.index}  |  Mean 5 Fitness: {mean_fitness:.2f}",
            "Attributes:",
            *[f"  {k}: {v}" for k, v in sorted(attrs.items())],
        ]
        print("\n".join(lines) + "\n")


def get_env_defined_actions(
    info: InfosDict,
    agents: list[str],
) -> dict[str, Any] | None:
    """Get the environment-defined actions for a list of agents.

    :param info: Info dictionary
    :type info: InfosDict
    :param agents: List of agents
    :type agents: list[str]
    :return: Environment-defined actions
    :rtype: dict[str, Any]
    """
    env_defined_actions = {
        agent: info[agent].get("env_defined_action", None) for agent in agents
    }

    if all(eda is None for eda in env_defined_actions.values()):
        return None

    return env_defined_actions


def save_llm_checkpoint(
    agent: LLMAlgorithm,
    checkpoint_path: str | None,
) -> None:
    """Checkpoint the LLM, saving LoRA adapter weights via HuggingFace ``save_pretrained``.

    The saved directory can be reloaded with::

        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained("<base-model-name>")
        model = PeftModel.from_pretrained(base_model, "<checkpoint_path>/actor/")

    :param agent: Agent
    :type agent: LLMAlgorithm
    :param checkpoint_path: Checkpoint path — used as-is (no algo sub-directory is appended).
        Defaults to ``"./saved_checkpoints"`` when ``None``.
    :type checkpoint_path: str
    """
    assert agent.actor is not None, "Actor is not initialized"
    path = "./saved_checkpoints" if checkpoint_path is None else checkpoint_path
    Path(path).mkdir(parents=True, exist_ok=True)
    if agent.accelerator is not None:
        agent.accelerator.wait_for_everyone()
        agent.save_checkpoint(path)
        agent.accelerator.wait_for_everyone()
    else:
        agent.save_checkpoint(path)


def consolidate_mutations(population: list[LLMAlgorithm]) -> None:
    """Consolidate mutations across processes during LLM fintuning.

    :param population: Population of agents
    :type population: list[EvolvableAlgorithm]
    """
    if not isinstance(population[0], LLMAlgorithm):
        warnings.warn(
            "Consolidate mutations is only supported for LLMAlgorithm.", stacklevel=2
        )
        return
    for agent in population:
        assert agent.actor is not None, "Actor is not initialized"
        index, mut, mut_value = broadcast_object_list(
            [
                agent.index,
                agent.mut,
                getattr(agent, agent.mut if agent.mut is not None else "None", "None"),
            ],
            from_process=0,
        )
        assert index == agent.index
        agent.mut = mut
        setattr(agent, mut, mut_value)

        if mut in ("lr", "critic_lr"):
            assert agent.optimizer is not None, "Optimizer is not initialized"
            opt = (
                agent.optimizer
                if not isinstance(agent.optimizer.optimizer, DummyOptimizer)
                # DeepSpeed engines expose the wrapped optimizer on the actor.
                else agent.actor.optimizer
            )
            lr = (
                (agent.lr, agent.lr_critic)
                if getattr(agent, "lr_critic", None) is not None
                else agent.lr
            )
            update_lr_kw: dict[str, Any] = {
                "optimizer": opt,
                "lr": lr,
                "accelerator": agent.accelerator,
                "scheduler_config": agent.cosine_lr_schedule_config,
            }
            agent.accelerator, agent.lr_scheduler = LLMAlgorithm.update_lr(
                **update_lr_kw
            )


def _distributed_world_size(accelerator: Accelerator | None) -> int:
    """World size for batch accounting: prefer Accelerate, else torch.distributed."""
    if accelerator is not None:
        return accelerator.num_processes
    return 1
