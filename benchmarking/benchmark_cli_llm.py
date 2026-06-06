"""LLM-family CLI layer for AgileRL benchmarking scripts.

The LLM algorithms share a large, growing set of hyperparameters. Rather than
hand-maintain a parallel table of argparse flags, this module defines them once
as **dataclasses** — the dataclass is the single source of truth for a field's
type, default and config key — and reuses the generic :mod:`benchmark_cli`
bridge to render flat ``--lr`` / ``--group-size`` style flags from those fields.

The dataclasses form a hierarchy::

    LLMInitHP                     # common to every LLM algorithm
    ├── LLMRLInitHP               # online RL family (vLLM rollouts, advantages)
    │   ├── PPOInitHP             # LLMPPO
    │   ├── ReinforceInitHP       # LLMREINFORCE
    │   └── GRPOInitHP            # GRPO
    │       ├── CISPOInitHP       # CISPO
    │       └── GSPOInitHP        # GSPO
    ├── DPOInitHP                 # offline preference (DPO)
    └── SFTInitHP                 # supervised fine-tuning (SFT)

Two entry points wire it all together:

* :func:`parse_llm_benchmark_cli` — the online RL family (vLLM rollout flags,
  W&B verbosity tier), returning :class:`LLMBenchmarkConfig`.
* :func:`parse_offline_llm_cli` — DPO / SFT (no vLLM), returning
  :class:`OfflineLLMConfig`.

Both load the YAML config case-insensitively (legacy ``UPPER_SNAKE`` keys work
unchanged), generate typed flags for the algorithm in play, and layer CLI
overrides on top. Any config key not modelled by a dataclass passes straight
through to the resolved ``INIT_HP`` / ``MUTATION_PARAMS`` dict (e.g. the DPO/SFT
``LORA`` sub-dict), so nothing is lost.

Kept torch-free (standard library + PyYAML + :mod:`benchmark_cli`) so it imports
and unit-tests without the heavyweight LLM stack; the ``VLLMConfig`` / agent
population are built later inside each script's ``main()``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Union

import yaml

import benchmark_cli

# Mirrors ``agilerl.utils.utils.WB_LEVELS`` (kept inline to stay torch-free).
WB_LEVELS: tuple[str, ...] = ("off", "essential", "standard", "detailed", "debug")

# Trainer-side bitsandbytes quantization recipes (HF + PEFT, not vLLM).
TRAINER_QUANT_CHOICES: tuple[str, ...] = ("none", "int8", "nf4")

ADV_NORM = Literal["mean_only", "mean_std"]
LORA_BIAS = Literal["none", "all", "lora_only"]
# Advantage granularity (independent of the IS level). "token" is invalid for
# the group-relative GRPO family; the algorithm validates the combination.
ACTION_GRANULARITY = Literal["auto", "token", "turn", "trajectory"]
# IS / ratio-pooling level. "auto" is the PPO/REINFORCE legacy gate; the GRPO
# family accepts token / turn / sequence.
IS_LEVEL = Literal["auto", "token", "turn", "sequence"]


# --------------------------------------------------------------------------- #
# Hyperparameter dataclasses (INIT_HP)
# --------------------------------------------------------------------------- #
@dataclass
class LLMInitHP:
    """Hyperparameters common to *every* LLM algorithm.

    Field names are ``lower_snake``; :func:`benchmark_cli.dataclass_to_upper_dict`
    emits the ``UPPER_SNAKE`` keys the algorithms read. Fields typed
    ``Optional[...] = None`` are *omitted* when unset so the algorithm's own
    default applies (the dataclass never silently forces a value the config did
    not ask for). ``lr`` lives here because every algorithm except PPO (which
    has separate actor / critic LRs) uses a single learning rate.
    """

    algo: str = "GRPO"
    batch_size: int = 8
    lr: Optional[float] = None
    update_epochs: int = 1
    max_grad_norm: float = 1.0
    use_liger_loss: bool = False
    seed: Optional[int] = None
    # Population / evolutionary HPO
    pop_size: int = 1
    elitism: bool = True
    tourn_size: int = 1
    eval_loop: int = 1


@dataclass
class LLMRLInitHP(LLMInitHP):
    """Online RL family (PPO / REINFORCE / GRPO / CISPO / GSPO).

    Adds the vLLM rollout backend, generation controls, advantage / importance
    sampling knobs, flat LoRA fields and the trainer-side quantization / memory
    options consumed by ``create_population``.
    """

    algo: str = "GRPO"
    micro_batch_size_per_gpu: int = 1
    # Generation / context
    max_model_len: int = 2048
    max_output_tokens: int = 64
    temperature: float = 0.9
    # Optimisation
    beta: float = 1e-3
    clip_coef: Union[float, List[float]] = 0.2
    # Advantage / importance sampling (family-dependent; omit when unset)
    action_granularity: Optional[ACTION_GRANULARITY] = None
    importance_sampling_level: Optional[IS_LEVEL] = None
    # Rollout backend
    use_vllm: bool = True
    enable_sliding_window: bool = False
    use_memory_efficient_params: Optional[bool] = None
    # LoRA (flat form used by create_population)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: LORA_BIAS = "none"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_target_scope: Optional[str] = None
    # Trainer memory / quant / attention
    quantization: Literal["none", "int8", "nf4"] = "none"
    activation_offload: bool = False
    attn_implementation: Optional[str] = None
    liger_token_chunk_size: Optional[int] = None


@dataclass
class PPOInitHP(LLMRLInitHP):
    """LLMPPO: separate actor / critic LRs, GAE and a value-function coef.

    PPO does not use the inherited single ``lr``; pass ``--lr-actor`` /
    ``--lr-critic`` instead.
    """

    algo: str = "LLMPPO"
    lr_actor: float = 5e-6
    lr_critic: float = 5e-6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5


@dataclass
class ReinforceInitHP(LLMRLInitHP):
    """LLMREINFORCE: single LR, return-batch-normalised advantages (no groups)."""

    algo: str = "LLMREINFORCE"
    gamma: float = 1.0


@dataclass
class GRPOInitHP(LLMRLInitHP):
    """GRPO: single LR plus group-relative advantage knobs."""

    algo: str = "GRPO"
    group_size: int = 8
    adv_norm: Optional[ADV_NORM] = None
    filter_zero_adv: Optional[bool] = None
    use_kl_advantage_shaping: Optional[bool] = None


@dataclass
class CISPOInitHP(GRPOInitHP):
    """CISPO: GRPO with the CISPO loss variant (clip_coef as ``[lo, hi]``)."""

    algo: str = "CISPO"


@dataclass
class GSPOInitHP(GRPOInitHP):
    """GSPO: GRPO with sequence-level importance sampling."""

    algo: str = "GSPO"


@dataclass
class DPOInitHP(LLMInitHP):
    """Direct Preference Optimization (offline; reference policy, no vLLM)."""

    algo: str = "DPO"
    beta: float = 0.1
    nll_alpha: float = 1.0
    use_separate_reference_adapter: bool = True
    max_context_length: int = 512
    reduce_memory_peak: bool = False
    calc_position_embeddings: bool = True
    evo_steps: Optional[int] = None
    evaluation_interval: int = 1000
    num_batches: Optional[int] = None
    eval_n_samples: int = 10


@dataclass
class SFTInitHP(LLMInitHP):
    """Supervised Fine-Tuning (offline; no reference policy, no vLLM)."""

    algo: str = "SFT"
    max_context_length: int = 512
    reduce_memory_peak: bool = False
    calc_position_embeddings: bool = True
    evo_steps: Optional[int] = None
    evaluation_interval: int = 1000
    num_batches: Optional[int] = None
    eval_n_samples: int = 10


# Algorithm name -> hyperparameter schema.
INIT_HP_CLASSES: dict[str, type[LLMInitHP]] = {
    "LLMPPO": PPOInitHP,
    "LLMREINFORCE": ReinforceInitHP,
    "GRPO": GRPOInitHP,
    "CISPO": CISPOInitHP,
    "GSPO": GSPOInitHP,
    "DPO": DPOInitHP,
    "SFT": SFTInitHP,
}
RL_ALGOS: tuple[str, ...] = ("LLMPPO", "LLMREINFORCE", "GRPO", "CISPO", "GSPO")
OFFLINE_ALGOS: tuple[str, ...] = ("DPO", "SFT")

# Fields excluded from auto-generated INIT_HP flags for the RL family:
#   * ``algo`` has the dedicated top-level ``--algo`` (which also picks schema);
#   * ``quantization`` / ``activation_offload`` use the runtime ``--trainer-*``
#     options below instead of ``--quantization`` / ``--activation-offload``.
# All three are still config-loaded dataclass fields and serialise to INIT_HP.
_RL_FLAG_SKIP = frozenset({"algo", "quantization", "activation_offload"})
# Offline schemas (DPO/SFT) have no quantization field; only ``algo`` is skipped.
_OFFLINE_FLAG_SKIP = frozenset({"algo"})
# The historic flag for MICRO_BATCH_SIZE_PER_GPU is ``--micro-batch-per-gpu``.
_INIT_FLAG_OVERRIDES = {"micro_batch_size_per_gpu": "--micro-batch-per-gpu"}


def select_init_hp_class(algo: str) -> type[LLMInitHP]:
    """Return the hyperparameter dataclass for ``algo`` (raises on unknown)."""
    try:
        return INIT_HP_CLASSES[algo]
    except KeyError:
        supported = ", ".join(sorted(INIT_HP_CLASSES))
        msg = f"Unknown algorithm {algo!r}. Supported: {supported}."
        raise ValueError(msg) from None


# --------------------------------------------------------------------------- #
# Mutation parameters (MUTATION_PARAMS)
# --------------------------------------------------------------------------- #
@dataclass
class MutationParams:
    """Evolutionary HPO mutation parameters (the ``MUTATION_PARAMS`` section)."""

    no_mut: float = 0.1
    rl_hp_mut: float = 0.6
    mut_sd: float = 0.1
    rand_seed: int = 42
    min_lr: float = 1e-7
    max_lr: float = 1e-5
    min_beta: Optional[float] = None
    max_beta: Optional[float] = None
    # Group-size mutation bounds (GRPO family); omitted when unset.
    min_group_size: Optional[int] = None
    max_group_size: Optional[int] = None
    # Critic-LR mutation bounds (PPO with a value head); omitted when unset.
    min_lr_critic: Optional[float] = None
    max_lr_critic: Optional[float] = None


# --------------------------------------------------------------------------- #
# Shared resolution helpers
# --------------------------------------------------------------------------- #
def _resolve_algo(
    argv: list[str] | None,
    default_config: str,
    default_algo: str,
) -> tuple[str, dict, dict]:
    """Pre-parse ``--config`` / ``--algo`` to pick the hyperparameter schema."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default_config)
    pre.add_argument("--algo", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    raw = benchmark_cli.load_config(pre_args.config)
    raw_init = raw.get("INIT_HP", raw.get("init_hp", {})) or {}
    raw_mut = raw.get("MUTATION_PARAMS", raw.get("mutation_params", {})) or {}
    algo = pre_args.algo or raw_init.get("ALGO") or raw_init.get("algo") or default_algo
    return algo, raw_init, raw_mut


def _add_config_and_algo_args(
    parser: argparse.ArgumentParser,
    *,
    default_config: str,
    algo_choices: tuple[str, ...],
) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help=f"Path to the YAML config file (default: {default_config}).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=sorted(algo_choices),
        help="Algorithm; selects the hyperparameter schema. Default: config ALGO.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully-resolved INIT_HP / MUTATION_PARAMS as YAML and exit.",
    )


def _add_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--init-hp-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=benchmark_cli.parse_key_value,
        help="Additional INIT_HP override (VALUE parsed as YAML). Repeatable.",
    )
    parser.add_argument(
        "--mutation-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=benchmark_cli.parse_key_value,
        help="Additional MUTATION_PARAMS override (VALUE parsed as YAML). Repeatable.",
    )


def _init_hp_from_dataclass(
    init_cls: type[LLMInitHP],
    raw_init: dict,
    args: argparse.Namespace,
    algo: str,
    *,
    skip: frozenset[str],
) -> dict[str, Any]:
    """Config -> dataclass -> CLI overrides -> INIT_HP dict (+ passthrough keys)."""
    instance, unknown = benchmark_cli.dataclass_from_mapping(init_cls, raw_init)
    benchmark_cli.apply_dataclass_overrides(
        instance, args, dest_prefix="hp_", skip=skip
    )
    instance.algo = algo
    return {**unknown, **benchmark_cli.dataclass_to_upper_dict(instance)}


def _mutation_from_dataclass(raw_mut: dict, args: argparse.Namespace) -> dict[str, Any]:
    instance, unknown = benchmark_cli.dataclass_from_mapping(MutationParams, raw_mut)
    benchmark_cli.apply_dataclass_overrides(instance, args, dest_prefix="mut_")
    return {**unknown, **benchmark_cli.dataclass_to_upper_dict(instance)}


def _apply_kv_overrides(target: dict, items: list[tuple[str, Any]]) -> None:
    for key, value in items:
        target[key] = value


def _maybe_print_config(
    init_hp: dict, mutation_params: dict, args: argparse.Namespace
) -> None:
    if not args.print_config:
        return
    yaml.safe_dump(
        {"INIT_HP": init_hp, "MUTATION_PARAMS": mutation_params},
        sys.stdout,
        sort_keys=False,
        default_flow_style=False,
    )
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# Online RL family: runtime (vLLM) flags + entry point
# --------------------------------------------------------------------------- #
def add_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    vllm_defaults: dict[str, Any] | None = None,
) -> None:
    """Add the model / vLLM / trainer-quantization runtime flags.

    These are *not* hyperparameters — they configure the trainer + rollout
    engine and keep their established names so documented sweep commands keep
    working.

    :param vllm_defaults: Per-script overrides for the vLLM flag *defaults*
        (``gpu_memory_utilization``, ``max_num_seqs``, ``sleep_mode``). Lets a
        throughput-oriented benchmark (e.g. reasoning) keep its own rollout
        defaults while still exposing the same flags.
    """
    vllm_defaults = vllm_defaults or {}
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="HF model id or path for the trainer (and vLLM unless --vllm-model).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300_000,
        help="Stop training after this many environment steps (default 300k).",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Stop after at most this wall-clock time (training loop exits early).",
    )

    trainer = parser.add_argument_group("Trainer (Hugging Face + bitsandbytes)")
    trainer.add_argument(
        "--trainer-quantization",
        type=str,
        choices=TRAINER_QUANT_CHOICES,
        default=None,
        help=(
            "HF trainer weight quantization via bitsandbytes (overrides INIT_HP "
            "QUANTIZATION). Default: use the config value. See "
            "--vllm-quantization for rollout weights."
        ),
    )
    trainer.add_argument(
        "--trainer-activation-offload",
        action="store_true",
        help=(
            "Offload activations saved for backward to pinned host RAM "
            "(sets INIT_HP ACTIVATION_OFFLOAD)."
        ),
    )

    vllm = parser.add_argument_group("Rollout (colocated vLLM)")
    vllm.add_argument(
        "--vllm-quantization",
        type=str,
        default=None,
        metavar="METHOD",
        help=(
            "vLLM rollout weight quantization (forwarded verbatim). "
            "'bitsandbytes' is the validated path; pair with "
            "--vllm-dtype bfloat16 and --trainer-quantization nf4 for QLoRA."
        ),
    )
    vllm.add_argument(
        "--vllm-model",
        type=str,
        default=None,
        help="HF id/path for the vLLM engine only. Default: same as --model.",
    )
    vllm.add_argument(
        "--vllm-dtype",
        type=str,
        default=None,
        choices=("bfloat16", "float16", "float32", "auto"),
        help="vLLM weight dtype passed to LLM(dtype=...). Default: vLLM chooses.",
    )
    vllm.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=vllm_defaults.get("gpu_memory_utilization", 0.25),
        help="Fraction of GPU memory vLLM may reserve (weights + KV pool).",
    )
    vllm.add_argument(
        "--vllm-kv-cache-memory-gib",
        type=float,
        default=None,
        help="Optional pin for vLLM KV cache size in GiB (default: auto-sized).",
    )
    vllm.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=vllm_defaults.get("max_num_seqs", 1),
        help="vLLM max concurrent sequences during rollout.",
    )
    vllm.add_argument(
        "--vllm-sleep-mode",
        action=argparse.BooleanOptionalAction,
        default=vllm_defaults.get("sleep_mode", None),
        help="Put vLLM to sleep between rollouts. Default: on when POP_SIZE==1.",
    )
    vllm.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable vLLM CUDA-graph capture (frees the graph pool). Default: vLLM's.",
    )
    vllm.add_argument(
        "--vllm-strip-multimodal-towers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Free GPU memory used by vision/audio towers of a multimodal base "
            "after vLLM init. Text-only RL never invokes them."
        ),
    )
    vllm.add_argument(
        "--weight-sharing",
        action="store_true",
        help=(
            "Zero-copy base-weight sharing: the bnb QLoRA trainer aliases vLLM's "
            "already-quantized base (one base copy on the GPU; only LoRA adapters "
            "synced per step). Requires --vllm-quantization bitsandbytes, "
            "--trainer-quantization nf4|int8, and vLLM sleep mode."
        ),
    )


def add_llm_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    """Standard W&B flags plus the LLM-only ``--wb-level`` verbosity tier."""
    benchmark_cli.add_wandb_arguments(parser)
    parser.add_argument(
        "--wb-level",
        type=str,
        choices=WB_LEVELS,
        default=os.environ.get("WANDB_LEVEL", "standard"),
        help=(
            "W&B logging verbosity tier (fallback $WANDB_LEVEL, default 'standard'): "
            "off | essential | standard | detailed | debug."
        ),
    )


@dataclass
class LLMBenchmarkConfig:
    """Everything an online-RL benchmark ``main()`` needs after CLI resolution."""

    init_hp: dict[str, Any]
    mutation_params: dict[str, Any]
    algo: str
    wandb: benchmark_cli.WandbSettings
    args: argparse.Namespace

    def build_vllm_kwargs(self) -> dict[str, Any]:
        """Assemble the ``VLLMConfig`` kwargs from the runtime flags + INIT_HP.

        Returns a plain dict (torch-free); the caller constructs ``VLLMConfig``.
        ``sleep_mode`` defaults to on for a single agent (``POP_SIZE == 1``).
        """
        args = self.args
        pop_size = self.init_hp.get("POP_SIZE", 1)
        sleep_mode = (
            pop_size == 1 if args.vllm_sleep_mode is None else args.vllm_sleep_mode
        )
        kwargs: dict[str, Any] = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_num_seqs": args.vllm_max_num_seqs,
            "sleep_mode": sleep_mode,
            "weight_sharing": args.weight_sharing,
        }
        if args.vllm_kv_cache_memory_gib is not None:
            kwargs["kv_cache_memory_bytes"] = int(
                args.vllm_kv_cache_memory_gib * (1024**3)
            )
        if args.vllm_quantization and args.vllm_quantization.lower() != "none":
            kwargs["quantization"] = args.vllm_quantization
        if args.vllm_model is not None:
            kwargs["vllm_model_name_or_path"] = args.vllm_model
        if args.vllm_dtype is not None and args.vllm_dtype != "auto":
            kwargs["dtype"] = args.vllm_dtype
        if args.vllm_enforce_eager is not None:
            kwargs["enforce_eager"] = args.vllm_enforce_eager
        if args.vllm_strip_multimodal_towers is not None:
            kwargs["strip_multimodal_towers"] = args.vllm_strip_multimodal_towers
        return kwargs


def parse_llm_benchmark_cli(
    *,
    default_config: str,
    default_model: str,
    description: str,
    epilog: str | None = None,
    add_script_arguments: Any = None,
    vllm_defaults: dict[str, Any] | None = None,
    argv: list[str] | None = None,
) -> LLMBenchmarkConfig:
    """Parse an online-RL benchmark command line into a :class:`LLMBenchmarkConfig`.

    :param add_script_arguments: Optional ``callable(parser)`` for script-specific
        flags (e.g. ``--env-name`` on the multi-turn benchmark).
    :param vllm_defaults: Optional per-script overrides for the vLLM flag defaults
        (see :func:`add_runtime_arguments`).
    """
    algo, raw_init, raw_mut = _resolve_algo(argv, default_config, "GRPO")
    init_cls = select_init_hp_class(algo)

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_config_and_algo_args(
        parser, default_config=default_config, algo_choices=RL_ALGOS
    )
    benchmark_cli.add_dataclass_arguments(
        parser,
        init_cls,
        title="INIT_HP (hyperparameters)",
        dest_prefix="hp_",
        skip=_RL_FLAG_SKIP,
        flag_overrides=_INIT_FLAG_OVERRIDES,
    )
    benchmark_cli.add_dataclass_arguments(
        parser,
        MutationParams,
        title="MUTATION_PARAMS",
        flag_prefix="mut-",
        dest_prefix="mut_",
    )
    _add_override_args(parser)
    add_runtime_arguments(
        parser, default_model=default_model, vllm_defaults=vllm_defaults
    )
    add_llm_wandb_arguments(parser)
    if add_script_arguments is not None:
        add_script_arguments(parser)

    args = parser.parse_args(argv)

    init_hp = _init_hp_from_dataclass(
        init_cls, raw_init, args, algo, skip=_RL_FLAG_SKIP
    )
    if args.trainer_quantization is not None:
        init_hp["QUANTIZATION"] = args.trainer_quantization
    if getattr(args, "trainer_activation_offload", False):
        init_hp["ACTIVATION_OFFLOAD"] = True
    _apply_kv_overrides(init_hp, args.init_hp_override)

    mutation_params = _mutation_from_dataclass(raw_mut, args)
    _apply_kv_overrides(mutation_params, args.mutation_override)

    _maybe_print_config(init_hp, mutation_params, args)

    return LLMBenchmarkConfig(
        init_hp=init_hp,
        mutation_params=mutation_params,
        algo=algo,
        wandb=benchmark_cli.build_wandb_settings(args),
        args=args,
    )


# --------------------------------------------------------------------------- #
# Offline family (DPO / SFT): entry point
# --------------------------------------------------------------------------- #
def _add_offline_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    """W&B flags for DPO/SFT, which carry W&B settings inside INIT_HP.

    Defaults are ``None`` so the config's ``WANDB`` / ``WANDB_PROJECT`` values
    are respected unless the flag is given; ``--no-wandb`` forces logging off.
    """
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging (sets INIT_HP WANDB=false).",
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override INIT_HP WANDB_PROJECT.",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Override INIT_HP WANDB_ENTITY.",
    )
    group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Override INIT_HP WANDB_RUN_NAME.",
    )


@dataclass
class OfflineLLMConfig:
    """Everything a DPO / SFT benchmark ``main()`` needs after CLI resolution.

    W&B and LoRA settings live inside ``init_hp`` (the DPO/SFT scripts read them
    from there); ``args`` carries the runtime flags (``model``, ``save_path``).
    """

    init_hp: dict[str, Any]
    mutation_params: dict[str, Any]
    algo: str
    args: argparse.Namespace


def parse_offline_llm_cli(
    *,
    default_config: str,
    default_model: str,
    description: str,
    epilog: str | None = None,
    add_script_arguments: Any = None,
    argv: list[str] | None = None,
) -> OfflineLLMConfig:
    """Parse a DPO / SFT benchmark command line into an :class:`OfflineLLMConfig`."""
    algo, raw_init, raw_mut = _resolve_algo(argv, default_config, "DPO")
    init_cls = select_init_hp_class(algo)

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_config_and_algo_args(
        parser, default_config=default_config, algo_choices=OFFLINE_ALGOS
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="HF model id or path for the trainer.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory for the elite LoRA checkpoint (default: auto timestamped).",
    )
    benchmark_cli.add_dataclass_arguments(
        parser,
        init_cls,
        title="INIT_HP (hyperparameters)",
        dest_prefix="hp_",
        skip=_OFFLINE_FLAG_SKIP,
    )
    benchmark_cli.add_dataclass_arguments(
        parser,
        MutationParams,
        title="MUTATION_PARAMS",
        flag_prefix="mut-",
        dest_prefix="mut_",
    )
    _add_override_args(parser)
    _add_offline_wandb_arguments(parser)
    if add_script_arguments is not None:
        add_script_arguments(parser)

    args = parser.parse_args(argv)

    init_hp = _init_hp_from_dataclass(
        init_cls, raw_init, args, algo, skip=_OFFLINE_FLAG_SKIP
    )
    # W&B settings live in INIT_HP for DPO/SFT; fold the flags in.
    if args.no_wandb:
        init_hp["WANDB"] = False
    if args.wandb_project is not None:
        init_hp["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity is not None:
        init_hp["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_run_name is not None:
        init_hp["WANDB_RUN_NAME"] = args.wandb_run_name
    _apply_kv_overrides(init_hp, args.init_hp_override)

    mutation_params = _mutation_from_dataclass(raw_mut, args)
    _apply_kv_overrides(mutation_params, args.mutation_override)

    _maybe_print_config(init_hp, mutation_params, args)

    return OfflineLLMConfig(
        init_hp=init_hp,
        mutation_params=mutation_params,
        algo=algo,
        args=args,
    )
