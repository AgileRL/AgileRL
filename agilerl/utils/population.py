# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Build an algorithm population from grouped constructor configs."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from accelerate import Accelerator

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
from agilerl.algorithms.core.registry import HyperparameterConfig
from agilerl.typing import BPTTSequenceType, PopulationType
from agilerl.utils.algo_utils import CosineLRScheduleConfig, clone_llm
from agilerl.utils.constructor_kwargs import accept_flat_kwargs
from agilerl.utils.llm_utils import (
    build_bnb_quantization_config,
    get_llm_accelerator,
    get_state_dict,
)

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from agilerl.algorithms import CISPO, DPO, GRPO, GSPO, LLMPPO, LLMREINFORCE, SFT

if TYPE_CHECKING:
    from peft import LoraConfig


@dataclass(frozen=True)
class PopulationSpaces:
    """Observation and action spaces; single-agent or per-agent."""

    observation_space: object | None = None
    action_space: object | None = None


@dataclass(frozen=True)
class PopulationNetworks:
    """Optional custom actor and critic modules."""

    actor_network: object | None = None
    critic_network: object | None = None


@dataclass(frozen=True)
class PopulationHPO:
    """Mutation config and extra algorithm constructor kwargs."""

    hp_config: HyperparameterConfig | None = None
    algo_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class PopulationRuntime:
    """Population size, vectorized env count, and device placement."""

    population_size: int = 1
    num_envs: int = 1
    device: str = "cpu"
    accelerator: Accelerator | None = None
    torch_compiler: str | None = None


@dataclass(frozen=True)
class PopulationWrappers:
    """Optional wrapper applied after construction (DDPG)."""

    agent_wrapper: Callable[..., object] | None = None
    wrapper_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class PopulationLLM:
    """Tokenizer, model id, LoRA, and vLLM settings for LLM algorithms."""

    tokenizer: object | None = None
    model_name: str | None = None
    lora_config: object | None = None
    vllm_config: object | None = None


@dataclass(frozen=True)
class PopulationRequest:
    """Inputs to one ``create_population`` call."""

    algo: str
    net_config: dict[str, Any] | None
    INIT_HP: dict[str, Any]
    spaces: PopulationSpaces
    networks: PopulationNetworks
    hpo: PopulationHPO
    runtime: PopulationRuntime
    wrappers: PopulationWrappers
    llm: PopulationLLM


def _normalize_algo_name(algo: str) -> str | None:
    """Map config-style names to internal algorithm keys."""
    return algo.upper().replace(" ", "").replace("-", "_")


def _lora_config_from_init_hp(INIT_HP: dict[str, Any]) -> LoraConfig | None:
    """Build a ``peft.LoraConfig`` from INIT_HP keys, or return None."""
    modules = INIT_HP.get("LORA_TARGET_MODULES") or INIT_HP.get("TARGET_MODULES")
    if not modules:
        return None
    if not HAS_LLM_DEPENDENCIES:
        return None
    from peft import LoraConfig  # optional extra: llm

    if isinstance(modules, str):
        modules = [modules]
    bias = INIT_HP.get("LORA_BIAS", "none")
    if bias not in ("none", "all", "lora_only"):
        msg = f"LORA_BIAS must be one of 'none', 'all', 'lora_only'; got {bias!r}."
        raise ValueError(msg)
    return LoraConfig(
        r=int(INIT_HP.get("LORA_R", 16)),
        lora_alpha=int(INIT_HP.get("LORA_ALPHA", 64)),
        target_modules=list(modules),
        lora_dropout=float(INIT_HP.get("LORA_DROPOUT", 0.0)),
        bias=bias,
        task_type=str(INIT_HP.get("LORA_TASK_TYPE", "CAUSAL_LM")),
    )


def _merge_generation_defaults(
    merged: dict[str, Any],
    *,
    vllm_config: object | None,
    INIT_HP: dict[str, Any],
    with_generation_defaults: bool,
) -> None:
    if with_generation_defaults:
        if vllm_config is not None:
            merged.setdefault("vllm_config", vllm_config)
        merged.setdefault("use_vllm", bool(INIT_HP.get("USE_VLLM", False)))
        merged.setdefault(
            "use_separate_reference_adapter",
            INIT_HP.get("USE_SEPARATE_REFERENCE_ADAPTER", True),
        )
        return
    merged.setdefault(
        "use_separate_reference_adapter",
        INIT_HP.get("USE_SEPARATE_REFERENCE_ADAPTER", False),
    )


def _apply_llm_init_hp_extras(merged: dict[str, Any], INIT_HP: dict[str, Any]) -> None:
    if "micro_batch_size_per_gpu" not in merged:
        batch_size = INIT_HP.get("BATCH_SIZE", 16)
        merged["micro_batch_size_per_gpu"] = INIT_HP.get(
            "MICRO_BATCH_SIZE_PER_GPU",
            batch_size,
        )  # NOTE we should take a look into deepspeed auto batch-sizing
    # Plain passthroughs: (merged_key, init_hp_key, caster, present_when_truthy).
    # activation_offload fires on key membership (so an explicit False is honoured);
    # lora_target_scope/chunk_rows fire only on a truthy value.
    passthroughs = (
        ("activation_offload", "ACTIVATION_OFFLOAD", bool, False),
        ("lora_target_scope", "LORA_TARGET_SCOPE", lambda v: v, True),
        ("chunk_rows", "CHUNK_ROWS", int, True),
    )
    for merged_key, init_hp_key, caster, present_when_truthy in passthroughs:
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
        quant_config = build_bnb_quantization_config(INIT_HP["QUANTIZATION"])
        if quant_config is not None:
            merged["quantization_config"] = quant_config
    # ATTN_IMPLEMENTATION: inject a non-"auto" value into model_config so the
    # algorithm's create_model call treats it as authoritative; "auto"/absent
    # leaves model_config alone so the auto-pick path still runs.
    attn_impl = INIT_HP.get("ATTN_IMPLEMENTATION")
    if attn_impl and attn_impl != "auto":
        mc = dict(merged.get("model_config") or {})
        mc.setdefault("attn_implementation", attn_impl)
        merged["model_config"] = mc


def _prepare_llm_algo_kwargs(
    algo_kwargs: dict[str, Any],
    *,
    tokenizer: object | None,
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
    _merge_generation_defaults(
        merged,
        vllm_config=vllm_config,
        INIT_HP=INIT_HP,
        with_generation_defaults=with_generation_defaults,
    )
    _apply_llm_init_hp_extras(merged, INIT_HP)
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


def _cosine_lr_from_init_hp(
    INIT_HP: dict[str, Any],
) -> CosineLRScheduleConfig | None:
    cosine_cfg = INIT_HP.get("COSINE_lR_SCHEDULER")
    if cosine_cfg is None:
        return None
    return CosineLRScheduleConfig(**cosine_cfg)


def _clone_llm_actor(
    actor_network: object | None,
    idx: int,
    INIT_HP: dict[str, Any],
    accelerator: Accelerator | None,
) -> object | None:
    if actor_network is None:
        return None
    if idx == 0:
        return actor_network
    if accelerator is None:
        state_dict = actor_network.state_dict()
    else:
        state_dict = get_state_dict(actor_network)
    return clone_llm(
        actor_network,
        zero_stage=INIT_HP.get("ZERO_STAGE", 0),
        state_dict=state_dict,
    )


def _wrap_agent(agent: object, wrappers: PopulationWrappers) -> object:
    if wrappers.agent_wrapper is None:
        return agent
    return wrappers.agent_wrapper(agent, **(wrappers.wrapper_kwargs or {}))


def _prepared_llm_kwargs(
    request: PopulationRequest, *, with_generation_defaults: bool = True
) -> dict[str, Any]:
    kwargs = _prepare_llm_algo_kwargs(
        request.hpo.algo_kwargs or {},
        tokenizer=request.llm.tokenizer,
        model_name=request.llm.model_name,
        lora_config=request.llm.lora_config,
        vllm_config=request.llm.vllm_config,
        INIT_HP=request.INIT_HP,
        with_generation_defaults=with_generation_defaults,
    )
    _validate_llm_kwargs(kwargs, actor_network=request.networks.actor_network)
    return kwargs


def _apply_torch_compiler(kw: dict[str, Any], torch_compiler: str | None) -> None:
    if torch_compiler is not None:
        kw.setdefault("torch_compiler", torch_compiler)


def _algo_kwargs(request: PopulationRequest) -> dict[str, Any]:
    return request.hpo.algo_kwargs or {}


def _build_dqn(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        population.append(
            DQN(
                observation_space=request.spaces.observation_space,
                action_space=request.spaces.action_space,
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                double=INIT_HP.get("DOUBLE", False),
                cudagraphs=INIT_HP.get("CUDAGRAPHS", False),
                actor_network=request.networks.actor_network,
                device=request.runtime.device,
                accelerator=request.runtime.accelerator,
                **_algo_kwargs(request),
            )
        )
    return population


def _build_rainbow_dqn(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        population.append(
            RainbowDQN(
                observation_space=request.spaces.observation_space,
                action_space=request.spaces.action_space,
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
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
                actor_network=request.networks.actor_network,
                device=request.runtime.device,
                accelerator=request.runtime.accelerator,
                **_algo_kwargs(request),
            )
        )
    return population


def _build_ddpg(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    population: PopulationType = []
    for idx in range(runtime.population_size):
        agent = DDPG(
            observation_space=request.spaces.observation_space,
            action_space=request.spaces.action_space,
            O_U_noise=INIT_HP.get("O_U_NOISE", True),
            expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
            vect_noise_dim=runtime.num_envs,
            mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
            theta=INIT_HP.get("THETA", 0.15),
            dt=INIT_HP.get("DT", 0.01),
            index=idx,
            hp_config=request.hpo.hp_config,
            net_config=request.net_config,
            batch_size=INIT_HP.get("BATCH_SIZE", 64),
            lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
            lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
            learn_step=INIT_HP.get("LEARN_STEP", 5),
            gamma=INIT_HP.get("GAMMA", 0.99),
            tau=INIT_HP.get("TAU", 0.001),
            policy_freq=INIT_HP.get("POLICY_FREQ", 2),
            actor_network=request.networks.actor_network,
            critic_network=request.networks.critic_network,
            share_encoders=INIT_HP.get("SHARE_ENCODERS", True),
            device=runtime.device,
            accelerator=runtime.accelerator,
            **_algo_kwargs(request),
        )
        population.append(_wrap_agent(agent, request.wrappers))
    return population


def _build_ppo(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    population: PopulationType = []
    for idx in range(runtime.population_size):
        population.append(
            PPO(
                observation_space=request.spaces.observation_space,
                action_space=request.spaces.action_space,
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
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
                actor_network=request.networks.actor_network,
                critic_network=request.networks.critic_network,
                device=runtime.device,
                accelerator=runtime.accelerator,
                num_envs=runtime.num_envs,
                **_algo_kwargs(request),
            )
        )
    return population


def _build_cqn(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        population.append(
            CQN(
                observation_space=request.spaces.observation_space,
                action_space=request.spaces.action_space,
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.0001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.001),
                double=INIT_HP.get("DOUBLE", False),
                actor_network=request.networks.actor_network,
                device=request.runtime.device,
                accelerator=request.runtime.accelerator,
                **_algo_kwargs(request),
            )
        )
    return population


def _build_td3(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    population: PopulationType = []
    for idx in range(runtime.population_size):
        population.append(
            TD3(
                observation_space=request.spaces.observation_space,
                action_space=request.spaces.action_space,
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=runtime.num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.99),
                tau=INIT_HP.get("TAU", 0.005),
                policy_freq=INIT_HP.get("POLICY_FREQ", 2),
                actor_network=request.networks.actor_network,
                critic_networks=request.networks.critic_network,
                share_encoders=INIT_HP.get("SHARE_ENCODERS", True),
                device=runtime.device,
                accelerator=runtime.accelerator,
                **_algo_kwargs(request),
            )
        )
    return population


def _build_maddpg(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    population: PopulationType = []
    for idx in range(runtime.population_size):
        population.append(
            MADDPG(
                observation_spaces=request.spaces.observation_space,
                action_spaces=request.spaces.action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=runtime.num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.95),
                tau=INIT_HP.get("TAU", 0.01),
                actor_networks=request.networks.actor_network,
                critic_networks=request.networks.critic_network,
                device=runtime.device,
                accelerator=runtime.accelerator,
                torch_compiler=runtime.torch_compiler,
                **_algo_kwargs(request),
            )
        )
    return population


def _build_matd3(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    population: PopulationType = []
    for idx in range(runtime.population_size):
        population.append(
            MATD3(
                observation_spaces=request.spaces.observation_space,
                action_spaces=request.spaces.action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
                O_U_noise=INIT_HP.get("O_U_NOISE", True),
                expl_noise=INIT_HP.get("EXPL_NOISE", 0.1),
                vect_noise_dim=runtime.num_envs,
                mean_noise=INIT_HP.get("MEAN_NOISE", 0.0),
                theta=INIT_HP.get("THETA", 0.15),
                dt=INIT_HP.get("DT", 0.01),
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr_actor=INIT_HP.get("LR_ACTOR", 0.0001),
                lr_critic=INIT_HP.get("LR_CRITIC", 0.001),
                policy_freq=INIT_HP.get("POLICY_FREQ", 2),
                learn_step=INIT_HP.get("LEARN_STEP", 5),
                gamma=INIT_HP.get("GAMMA", 0.95),
                tau=INIT_HP.get("TAU", 0.01),
                actor_networks=request.networks.actor_network,
                critic_networks=request.networks.critic_network,
                device=runtime.device,
                accelerator=runtime.accelerator,
                torch_compiler=runtime.torch_compiler,
                **_algo_kwargs(request),
            )
        )
    return population


def _build_ippo(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    population: PopulationType = []
    for idx in range(runtime.population_size):
        population.append(
            IPPO(
                observation_spaces=request.spaces.observation_space,
                action_spaces=request.spaces.action_space,
                agent_ids=INIT_HP["AGENT_IDS"],
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
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
                actor_networks=request.networks.actor_network,
                critic_networks=request.networks.critic_network,
                action_batch_size=INIT_HP.get("ACTION_BATCH_SIZE"),
                device=runtime.device,
                accelerator=runtime.accelerator,
                torch_compiler=runtime.torch_compiler,
                **_algo_kwargs(request),
            )
        )
    return population


def _build_neural_ucb(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        population.append(
            NeuralUCB(
                observation_space=request.spaces.observation_space,
                action_space=request.spaces.action_space,
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
                gamma=INIT_HP.get("GAMMA", 1),
                lamb=INIT_HP.get("LAMBDA", 1),
                reg=INIT_HP.get("REG", 0.000625),
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.001),
                learn_step=INIT_HP.get("LEARN_STEP", 2),
                actor_network=request.networks.actor_network,
                device=request.runtime.device,
                accelerator=request.runtime.accelerator,
                **_algo_kwargs(request),
            )
        )
    return population


def _build_neural_ts(request: PopulationRequest) -> PopulationType:
    INIT_HP = request.INIT_HP
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        population.append(
            NeuralTS(
                observation_space=request.spaces.observation_space,
                action_space=request.spaces.action_space,
                index=idx,
                hp_config=request.hpo.hp_config,
                net_config=request.net_config,
                gamma=INIT_HP.get("GAMMA", 1),
                lamb=INIT_HP.get("LAMBDA", 1),
                reg=INIT_HP.get("REG", 0.000625),
                batch_size=INIT_HP.get("BATCH_SIZE", 64),
                lr=INIT_HP.get("LR", 0.003),
                learn_step=INIT_HP.get("LEARN_STEP", 2),
                actor_network=request.networks.actor_network,
                device=request.runtime.device,
                accelerator=request.runtime.accelerator,
                **_algo_kwargs(request),
            )
        )
    return population


def _grpo_member_kwargs(
    request: PopulationRequest,
    idx: int,
    cosine_lr: CosineLRScheduleConfig | None,
    actor_network: object | None,
) -> dict[str, Any]:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    kw: dict[str, Any] = {
        "hp_config": request.hpo.hp_config,
        "index": idx,
        "batch_size": INIT_HP.get("BATCH_SIZE", 16),
        "beta": INIT_HP.get("BETA", 0.001),
        "lr": INIT_HP.get("LR", 5e-7),
        "clip_coef": INIT_HP.get("CLIP_COEF", 0.2),
        "max_grad_norm": INIT_HP.get("MAX_GRAD_NORM", 0.1),
        "update_epochs": INIT_HP.get("UPDATE_EPOCHS", 1),
        "group_size": INIT_HP.get("GROUP_SIZE", 8),
        "temperature": INIT_HP.get("TEMPERATURE", 0.9),
        "repetition_penalty": INIT_HP.get("REPETITION_PENALTY", 1.0),
        "top_p": INIT_HP.get("TOP_P", 0.95),
        "top_k": INIT_HP.get("TOP_K", 50),
        "min_p": INIT_HP.get("MIN_P", 0.0),
        "use_memory_efficient_params": INIT_HP.get(
            "USE_MEMORY_EFFICIENT_PARAMS", True
        ),
        "calc_position_embeddings": INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
        "max_output_tokens": INIT_HP.get("MAX_OUTPUT_TOKENS"),
        "min_output_tokens": INIT_HP.get("MIN_OUTPUT_TOKENS"),
        "max_model_len": INIT_HP.get("MAX_MODEL_LEN", 1024),
        "cosine_lr_schedule_config": cosine_lr,
        "accelerator": get_llm_accelerator(runtime.accelerator, idx),
        "gradient_checkpointing": INIT_HP.get("GRADIENT_CHECKPOINTING", True),
        "actor_network": actor_network,
        "clone": idx != 0 and actor_network is not None,
        "seed": INIT_HP.get("SEED", 42),
        "use_liger_loss": INIT_HP.get("USE_LIGER_LOSS", False),
        "cast_logprobs_to_fp32": INIT_HP.get("CAST_LOGPROBS_TO_FP32", True),
        "use_kl_advantage_shaping": INIT_HP.get("USE_KL_ADVANTAGE_SHAPING", False),
        "adv_norm": INIT_HP.get("ADV_NORM", "mean_std"),
        "loss_type": INIT_HP.get("LOSS_TYPE", "grpo"),
        "importance_sampling_level": INIT_HP.get("IMPORTANCE_SAMPLING_LEVEL"),
        "advantage_granularity": INIT_HP.get(
            "ADVANTAGE_GRANULARITY", INIT_HP.get("ACTION_GRANULARITY", "auto")
        ),
        "whiten_advantages": INIT_HP.get("WHITEN_ADVANTAGES", False),
        "adv_clip_range": INIT_HP.get("ADV_CLIP_RANGE"),
        "filter_zero_adv": INIT_HP.get(
            "FILTER_ZERO_ADV",
            INIT_HP.get("FILTER_ZERO_ADVANTAGES", False),
        ),
        "adv_filter_eps": INIT_HP.get(
            "ADV_FILTER_EPS",
            INIT_HP.get("ADVANTAGE_FILTER_EPS", 0.0),
        ),
        "vllm_importance_sampling_correction": INIT_HP.get(
            "VLLM_IMPORTANCE_SAMPLING_CORRECTION", True
        ),
        "vllm_importance_sampling_cap": INIT_HP.get(
            "VLLM_IMPORTANCE_SAMPLING_CAP", 2.0
        ),
        "use_sequence_packing": INIT_HP.get("USE_SEQUENCE_PACKING", False),
    }
    _apply_torch_compiler(kw, runtime.torch_compiler)
    return kw


def _build_grpo_family(request: PopulationRequest, algo: str) -> PopulationType:
    if not HAS_LLM_DEPENDENCIES:
        msg = "GRPO/CISPO/GSPO require optional LLM dependencies (install agilerl[llm])."
        raise ImportError(msg)
    kwargs = _prepared_llm_kwargs(request)
    cosine_lr = _cosine_lr_from_init_hp(request.INIT_HP)
    algo_cls = {"GRPO": GRPO, "CISPO": CISPO, "GSPO": GSPO}[algo]
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        act = _clone_llm_actor(
            request.networks.actor_network,
            idx,
            request.INIT_HP,
            request.runtime.accelerator,
        )
        kw = dict(kwargs)
        kw.update(_grpo_member_kwargs(request, idx, cosine_lr, act))
        if algo in ("CISPO", "GSPO"):
            kw.pop("loss_type", None)
        population.append(algo_cls(**kw))
    return population


def _sft_member_kwargs(
    request: PopulationRequest,
    idx: int,
    actor_network: object | None,
) -> dict[str, Any]:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    kw: dict[str, Any] = {
        "hp_config": request.hpo.hp_config,
        "index": idx,
        "batch_size": INIT_HP.get("BATCH_SIZE", 16),
        "lr": INIT_HP.get("LR", 5e-5),
        "max_grad_norm": INIT_HP.get("MAX_GRAD_NORM", 0.1),
        "update_epochs": INIT_HP.get("UPDATE_EPOCHS", 1),
        "calc_position_embeddings": INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
        "accelerator": get_llm_accelerator(runtime.accelerator, idx),
        "gradient_checkpointing": INIT_HP.get("GRADIENT_CHECKPOINTING", True),
        "actor_network": actor_network,
        "clone": idx != 0 and actor_network is not None,
        "seed": INIT_HP.get("SEED", 42),
        "use_liger_loss": INIT_HP.get("USE_LIGER_LOSS", False),
    }
    _apply_torch_compiler(kw, runtime.torch_compiler)
    return kw


def _build_sft(request: PopulationRequest) -> PopulationType:
    if not HAS_LLM_DEPENDENCIES:
        msg = "SFT requires optional LLM dependencies (install agilerl[llm])."
        raise ImportError(msg)
    kwargs = _prepared_llm_kwargs(request, with_generation_defaults=False)
    kwargs.pop("use_vllm", None)
    kwargs.pop("vllm_config", None)
    kwargs.pop("use_separate_reference_adapter", None)
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        act = _clone_llm_actor(
            request.networks.actor_network,
            idx,
            request.INIT_HP,
            request.runtime.accelerator,
        )
        kw = dict(kwargs)
        kw.update(_sft_member_kwargs(request, idx, act))
        population.append(SFT(**kw))
    return population


def _dpo_member_kwargs(
    request: PopulationRequest,
    idx: int,
    actor_network: object | None,
) -> dict[str, Any]:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    kw: dict[str, Any] = {
        "hp_config": request.hpo.hp_config,
        "index": idx,
        "batch_size": INIT_HP.get("BATCH_SIZE", 16),
        "beta": INIT_HP.get("BETA", 0.001),
        "lr": INIT_HP.get("LR", 0.000005),
        "max_grad_norm": INIT_HP.get("MAX_GRAD_NORM", 0.1),
        "update_epochs": INIT_HP.get("UPDATE_EPOCHS", 1),
        "calc_position_embeddings": INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
        "accelerator": get_llm_accelerator(runtime.accelerator, idx),
        "gradient_checkpointing": INIT_HP.get("GRADIENT_CHECKPOINTING", True),
        "actor_network": actor_network,
        "clone": idx != 0 and actor_network is not None,
        "seed": INIT_HP.get("SEED", 42),
        "use_liger_loss": INIT_HP.get("USE_LIGER_LOSS", False),
    }
    _apply_torch_compiler(kw, runtime.torch_compiler)
    return kw


def _build_dpo(request: PopulationRequest) -> PopulationType:
    if not HAS_LLM_DEPENDENCIES:
        msg = "DPO requires optional LLM dependencies (install agilerl[llm])."
        raise ImportError(msg)
    kwargs = _prepared_llm_kwargs(request, with_generation_defaults=False)
    kwargs.pop("use_vllm", None)
    kwargs.pop("vllm_config", None)
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        act = _clone_llm_actor(
            request.networks.actor_network,
            idx,
            request.INIT_HP,
            request.runtime.accelerator,
        )
        kw = dict(kwargs)
        kw.update(_dpo_member_kwargs(request, idx, act))
        population.append(DPO(**kw))
    return population


def _llmppo_member_kwargs(
    request: PopulationRequest,
    idx: int,
    cosine_lr: CosineLRScheduleConfig | None,
    actor_network: object | None,
) -> dict[str, Any]:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    kw: dict[str, Any] = {
        "hp_config": request.hpo.hp_config,
        "index": idx,
        "batch_size": INIT_HP.get("BATCH_SIZE", 16),
        "beta": INIT_HP.get("BETA", 0.01),
        "vf_coef": INIT_HP.get("VF_COEF", 0.5),
        "clip_coef": INIT_HP.get("CLIP_COEF", 0.2),
        "gamma": INIT_HP.get("GAMMA", 1.0),
        "gae_lambda": INIT_HP.get("GAE_LAMBDA", 1.0),
        "advantage_granularity": INIT_HP.get(
            "ADVANTAGE_GRANULARITY", INIT_HP.get("ACTION_GRANULARITY", "auto")
        ),
        "importance_sampling_level": INIT_HP.get(
            "IMPORTANCE_SAMPLING_LEVEL", "auto"
        ),
        "turn_ratio_pooling": INIT_HP.get("TURN_RATIO_POOLING", "sum"),
        "turn_level_clip": INIT_HP.get("TURN_LEVEL_CLIP", True),
        "turn_value_reduction": INIT_HP.get("TURN_VALUE_REDUCTION", "final_value"),
        "whiten_advantages": INIT_HP.get("WHITEN_ADVANTAGES", True),
        "chunk_rows": INIT_HP.get("CHUNK_ROWS"),
        "lr_actor": INIT_HP.get("LR_ACTOR", INIT_HP.get("LR", 5e-6)),
        "lr_critic": INIT_HP.get("LR_CRITIC"),
        "max_grad_norm": INIT_HP.get("MAX_GRAD_NORM", 1.0),
        "update_epochs": INIT_HP.get("UPDATE_EPOCHS", 1),
        "temperature": INIT_HP.get("TEMPERATURE", 1.0),
        "max_output_tokens": INIT_HP.get("MAX_OUTPUT_TOKENS"),
        "min_output_tokens": INIT_HP.get("MIN_OUTPUT_TOKENS"),
        "max_model_len": INIT_HP.get("MAX_MODEL_LEN", 1024),
        "use_memory_efficient_params": INIT_HP.get(
            "USE_MEMORY_EFFICIENT_PARAMS", True
        ),
        "calc_position_embeddings": INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
        "cosine_lr_schedule_config": cosine_lr,
        "accelerator": get_llm_accelerator(runtime.accelerator, idx),
        "gradient_checkpointing": INIT_HP.get("GRADIENT_CHECKPOINTING", True),
        "actor_network": actor_network,
        "clone": idx != 0 and actor_network is not None,
        "seed": INIT_HP.get("SEED", 42),
        "use_liger_loss": INIT_HP.get("USE_LIGER_LOSS", False),
        "cast_logprobs_to_fp32": INIT_HP.get("CAST_LOGPROBS_TO_FP32", True),
        "vllm_importance_sampling_correction": INIT_HP.get(
            "VLLM_IMPORTANCE_SAMPLING_CORRECTION", True
        ),
        "vllm_importance_sampling_cap": INIT_HP.get(
            "VLLM_IMPORTANCE_SAMPLING_CAP", 2.0
        ),
    }
    _apply_torch_compiler(kw, runtime.torch_compiler)
    return kw


def _build_llmppo(request: PopulationRequest) -> PopulationType:
    if not HAS_LLM_DEPENDENCIES:
        msg = "LLMPPO requires optional LLM dependencies (install agilerl[llm])."
        raise ImportError(msg)
    kwargs = _prepared_llm_kwargs(request)
    cosine_lr = _cosine_lr_from_init_hp(request.INIT_HP)
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        act = _clone_llm_actor(
            request.networks.actor_network,
            idx,
            request.INIT_HP,
            request.runtime.accelerator,
        )
        kw = dict(kwargs)
        kw.update(_llmppo_member_kwargs(request, idx, cosine_lr, act))
        population.append(LLMPPO(**kw))
    return population


def _llmreinforce_member_kwargs(
    request: PopulationRequest,
    idx: int,
    cosine_lr: CosineLRScheduleConfig | None,
    actor_network: object | None,
) -> dict[str, Any]:
    INIT_HP = request.INIT_HP
    runtime = request.runtime
    kw: dict[str, Any] = {
        "hp_config": request.hpo.hp_config,
        "index": idx,
        "batch_size": INIT_HP.get("BATCH_SIZE", 16),
        "beta": INIT_HP.get("BETA", 0.01),
        "clip_coef": INIT_HP.get("CLIP_COEF", 0.2),
        "gamma": INIT_HP.get("GAMMA", 0.99),
        "advantage_granularity": INIT_HP.get(
            "ADVANTAGE_GRANULARITY", INIT_HP.get("ACTION_GRANULARITY", "auto")
        ),
        "importance_sampling_level": INIT_HP.get(
            "IMPORTANCE_SAMPLING_LEVEL", "token"
        ),
        "turn_ratio_pooling": INIT_HP.get("TURN_RATIO_POOLING", "sum"),
        "lr": INIT_HP.get("LR", 5e-7),
        "max_grad_norm": INIT_HP.get("MAX_GRAD_NORM", 1.0),
        "update_epochs": INIT_HP.get("UPDATE_EPOCHS", 1),
        "temperature": INIT_HP.get("TEMPERATURE", 1.0),
        "max_output_tokens": INIT_HP.get("MAX_OUTPUT_TOKENS"),
        "min_output_tokens": INIT_HP.get("MIN_OUTPUT_TOKENS"),
        "max_model_len": INIT_HP.get("MAX_MODEL_LEN"),
        "calc_position_embeddings": INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
        "use_memory_efficient_params": INIT_HP.get(
            "USE_MEMORY_EFFICIENT_PARAMS", True
        ),
        "cosine_lr_schedule_config": cosine_lr,
        "accelerator": get_llm_accelerator(runtime.accelerator, idx),
        "gradient_checkpointing": INIT_HP.get("GRADIENT_CHECKPOINTING", True),
        "actor_network": actor_network,
        "clone": idx != 0 and actor_network is not None,
        "seed": INIT_HP.get("SEED", 42),
        "use_liger_loss": INIT_HP.get("USE_LIGER_LOSS", False),
        "cast_logprobs_to_fp32": INIT_HP.get("CAST_LOGPROBS_TO_FP32", True),
        "vllm_importance_sampling_correction": INIT_HP.get(
            "VLLM_IMPORTANCE_SAMPLING_CORRECTION", True
        ),
        "vllm_importance_sampling_cap": INIT_HP.get(
            "VLLM_IMPORTANCE_SAMPLING_CAP", 2.0
        ),
    }
    _apply_torch_compiler(kw, runtime.torch_compiler)
    return kw


def _build_llmreinforce(request: PopulationRequest) -> PopulationType:
    if not HAS_LLM_DEPENDENCIES:
        msg = (
            "LLMREINFORCE requires optional LLM dependencies "
            "(install agilerl[llm])."
        )
        raise ImportError(msg)
    kwargs = _prepared_llm_kwargs(request)
    cosine_lr = _cosine_lr_from_init_hp(request.INIT_HP)
    population: PopulationType = []
    for idx in range(request.runtime.population_size):
        act = _clone_llm_actor(
            request.networks.actor_network,
            idx,
            request.INIT_HP,
            request.runtime.accelerator,
        )
        kw = dict(kwargs)
        kw.update(_llmreinforce_member_kwargs(request, idx, cosine_lr, act))
        population.append(LLMREINFORCE(**kw))
    return population


def _build_grpo(request: PopulationRequest) -> PopulationType:
    return _build_grpo_family(request, "GRPO")


def _build_cispo(request: PopulationRequest) -> PopulationType:
    return _build_grpo_family(request, "CISPO")


def _build_gspo(request: PopulationRequest) -> PopulationType:
    return _build_grpo_family(request, "GSPO")


POPULATION_BUILDERS: dict[str, Callable[[PopulationRequest], PopulationType]] = {
    "DQN": _build_dqn,
    "Rainbow DQN": _build_rainbow_dqn,
    "DDPG": _build_ddpg,
    "PPO": _build_ppo,
    "CQN": _build_cqn,
    "TD3": _build_td3,
    "MADDPG": _build_maddpg,
    "MATD3": _build_matd3,
    "IPPO": _build_ippo,
    "NeuralUCB": _build_neural_ucb,
    "NeuralTS": _build_neural_ts,
    "GRPO": _build_grpo,
    "CISPO": _build_cispo,
    "GSPO": _build_gspo,
    "SFT": _build_sft,
    "DPO": _build_dpo,
    "LLMPPO": _build_llmppo,
    "LLMREINFORCE": _build_llmreinforce,
}


DEFAULT_POPULATION_SPACES = PopulationSpaces()
DEFAULT_POPULATION_NETWORKS = PopulationNetworks()
DEFAULT_POPULATION_HPO = PopulationHPO()
DEFAULT_POPULATION_RUNTIME = PopulationRuntime()
DEFAULT_POPULATION_WRAPPERS = PopulationWrappers()
DEFAULT_POPULATION_LLM = PopulationLLM()


@accept_flat_kwargs
def create_population(
    algo: str,
    net_config: dict[str, Any] | None,
    INIT_HP: dict[str, Any],
    spaces: PopulationSpaces = DEFAULT_POPULATION_SPACES,
    networks: PopulationNetworks = DEFAULT_POPULATION_NETWORKS,
    hpo: PopulationHPO = DEFAULT_POPULATION_HPO,
    runtime: PopulationRuntime = DEFAULT_POPULATION_RUNTIME,
    wrappers: PopulationWrappers = DEFAULT_POPULATION_WRAPPERS,
    llm: PopulationLLM = DEFAULT_POPULATION_LLM,
) -> PopulationType:
    """Return a population of identical agents.

    Flat keyword arguments matching the dataclass field names are accepted.

    .. deprecated::
        Use ``Algorithm.population()`` instead (e.g. ``DQN.population(size=4, ...)``).

    :param algo: RL algorithm name.
    :type algo: str
    :param net_config: Network configuration.
    :type net_config: dict or None
    :param INIT_HP: Initial hyperparameters.
    :type INIT_HP: dict
    :param spaces: Observation and action spaces.
    :type spaces: PopulationSpaces
    :param networks: Optional custom actor and critic modules.
    :type networks: PopulationNetworks
    :param hpo: Mutation config and extra algorithm kwargs.
    :type hpo: PopulationHPO
    :param runtime: Population size, env count, and device.
    :type runtime: PopulationRuntime
    :param wrappers: Optional post-construction wrapper (DDPG).
    :type wrappers: PopulationWrappers
    :param llm: Tokenizer, model id, LoRA, and vLLM settings.
    :type llm: PopulationLLM
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
    request = PopulationRequest(
        algo=algo,
        net_config=net_config,
        INIT_HP=INIT_HP,
        spaces=spaces,
        networks=networks,
        hpo=hpo,
        runtime=runtime,
        wrappers=wrappers,
        llm=llm,
    )
    builder = POPULATION_BUILDERS.get(algo) or POPULATION_BUILDERS.get(
        _normalize_algo_name(algo) or ""
    )
    if builder is None:
        return []
    return builder(request)
