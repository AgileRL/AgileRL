from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import argparse
import logging
import os
import warnings

import gem
import yaml
from transformers import AutoTokenizer

from agilerl.algorithms import CISPO, GRPO, GSPO, LLMPPO, LLMREINFORCE
from agilerl.llm_envs import TokenObservationWrapper
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import (
    build_bnb_quantization_config,
    create_llm_accelerator,
)
from agilerl.utils.utils import WB_LEVELS, create_population

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ENV_NAME = "game:Sudoku-v0-hard"

ALGO_REGISTRY = {
    "LLMPPO": LLMPPO,
    "LLMREINFORCE": LLMREINFORCE,
    "GRPO": GRPO,
    "CISPO": CISPO,
    "GSPO": GSPO,
}

_TRAINER_QUANT_CHOICES = ("none", "int8", "nf4")
_ALGO_CHOICES = tuple(ALGO_REGISTRY.keys())
_ADV_NORM_CHOICES = ("mean_only", "mean_std")
_LORA_BIAS_CHOICES = ("none", "all", "lora_only")
# IS / ratio-pooling level. "auto" is PPO/REINFORCE-only (legacy turn_level_clip
# gate); the GRPO family (GRPO/CISPO) accepts token/turn/sequence.
_IS_LEVEL_CHOICES = ("auto", "token", "turn", "sequence")

# INIT_HP keys already exposed under dedicated benchmark / trainer flags.
_INIT_HP_CLI_SKIP = frozenset(
    {
        "BATCH_SIZE",
        "MICRO_BATCH_SIZE_PER_GPU",
        "MAX_MODEL_LEN",
        "QUANTIZATION",
        "ACTIVATION_OFFLOAD",
    },
)

# (YAML key, mutation section, argparse kind, choices or None)
_CONFIG_ARG_SPECS: tuple[tuple[str, bool, str, tuple[str, ...] | None], ...] = (
    # MUTATION_PARAMS (cispo_quant_bench.yaml)
    ("NO_MUT", True, "float", None),
    ("RL_HP_MUT", True, "float", None),
    ("MUT_SD", True, "float", None),
    ("RAND_SEED", True, "int", None),
    ("MIN_LR", True, "float", None),
    ("MAX_LR", True, "float", None),
    ("MIN_BETA", True, "float", None),
    ("MAX_BETA", True, "float", None),
    ("MIN_GROUP_SIZE", True, "int", None),
    ("MAX_GROUP_SIZE", True, "int", None),
    # INIT_HP (cispo_quant_bench.yaml)
    ("ACTION_GRANULARITY", False, "str", None),
    ("IMPORTANCE_SAMPLING_LEVEL", False, "choices", _IS_LEVEL_CHOICES),
    ("ALGO", False, "choices", _ALGO_CHOICES),
    ("BETA", False, "float", None),
    ("CLIP_COEF", False, "float", None),
    ("ELITISM", False, "bool", None),
    ("EVAL_LOOP", False, "int", None),
    ("GROUP_SIZE", False, "int", None),
    ("ADV_NORM", False, "choices", _ADV_NORM_CHOICES),
    ("FILTER_ZERO_ADV", False, "bool", None),
    ("LR", False, "float", None),
    ("MAX_GRAD_NORM", False, "float", None),
    ("MAX_OUTPUT_TOKENS", False, "int", None),
    ("POP_SIZE", False, "int", None),
    ("LORA_TARGET_SCOPE", False, "str", None),
    ("TARGET_MODULES", False, "list", None),
    ("TEMPERATURE", False, "float", None),
    ("TOURN_SIZE", False, "int", None),
    ("UPDATE_EPOCHS", False, "int", None),
    ("USE_VLLM", False, "bool", None),
    ("USE_LIGER_LOSS", False, "bool", None),
    ("ENABLE_SLIDING_WINDOW", False, "bool", None),
    ("ATTN_IMPLEMENTATION", False, "str", None),
    ("LIGER_TOKEN_CHUNK_SIZE", False, "int", None),
    ("LORA_R", False, "int", None),
    ("LORA_ALPHA", False, "int", None),
    ("LORA_DROPOUT", False, "float", None),
    ("LORA_BIAS", False, "choices", _LORA_BIAS_CHOICES),
)


def _patch_sudoku_prompt(reset_method):
    r"""Wrap a gem Sudoku env ``reset`` to fix errors in its instruction prompt.

    gem's ``SudokuEnv._get_instructions`` renders rule 3 as
    ``"nine {scale}x{scale} subgrids"``, which is wrong: a 9x9 board (hard) has
    nine **3x3** boxes and a 4x4 board (easy) has **four 2x2** boxes. The worked
    example ``\boxed{1 1 5}`` is also out of range on the 4x4 (easy) board, whose
    digits only go up to 4. Each replacement matches only its own variant, so the
    wrapper is safe to apply to either difficulty.
    """

    def wrapper(*args, **kwargs):
        obs_text, info = reset_method(*args, **kwargs)
        obs_text = obs_text.replace("nine 9x9 subgrids", "nine 3x3 subgrids")
        obs_text = obs_text.replace("nine 4x4 subgrids", "four 2x2 subgrids")
        obs_text = obs_text.replace(r"\boxed{1 1 5}", r"\boxed{1 1 4}")
        return obs_text, info

    return wrapper


def _config_cli_flag(yaml_key: str, *, mutation: bool) -> str:
    prefix = "mut-" if mutation else ""
    return f"--{prefix}{yaml_key.lower().replace('_', '-')}"


def _config_dest_name(yaml_key: str, *, mutation: bool) -> str:
    prefix = "mut_" if mutation else "hp_"
    return f"{prefix}{yaml_key.lower()}"


def _parse_key_value_override(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        msg = f"expected KEY=VALUE, got {raw!r}"
        raise argparse.ArgumentTypeError(msg)
    key, _, value_raw = raw.partition("=")
    key = key.strip().upper()
    if not key:
        msg = f"empty INIT_HP / MUTATION_PARAMS key in {raw!r}"
        raise argparse.ArgumentTypeError(msg)
    value = yaml.safe_load(value_raw.strip())
    return key, value


def _register_config_override_args(parser: argparse.ArgumentParser) -> None:
    init_hp_group = parser.add_argument_group(
        "INIT_HP overrides (override values from the YAML config)",
    )
    mut_group = parser.add_argument_group(
        "MUTATION_PARAMS overrides (override values from the YAML config)",
    )
    for yaml_key, mutation, kind, choices in _CONFIG_ARG_SPECS:
        if not mutation and yaml_key in _INIT_HP_CLI_SKIP:
            continue
        flag = _config_cli_flag(yaml_key, mutation=mutation)
        dest = _config_dest_name(yaml_key, mutation=mutation)
        target = mut_group if mutation else init_hp_group
        help_text = (
            f"Override {'MUTATION_PARAMS' if mutation else 'INIT_HP'} {yaml_key}."
        )
        if kind == "bool":
            target.add_argument(
                flag,
                dest=dest,
                action=argparse.BooleanOptionalAction,
                default=None,
                help=help_text,
            )
        elif kind == "choices":
            target.add_argument(
                flag,
                dest=dest,
                type=str,
                choices=choices,
                default=None,
                help=help_text,
            )
        elif kind == "list":
            target.add_argument(
                flag,
                dest=dest,
                nargs="+",
                default=None,
                metavar="MODULE",
                help=f"{help_text} Pass one or more module names.",
            )
        elif kind == "int":
            target.add_argument(
                flag,
                dest=dest,
                type=int,
                default=None,
                help=help_text,
            )
        elif kind == "float":
            target.add_argument(
                flag,
                dest=dest,
                type=float,
                default=None,
                help=help_text,
            )
        else:
            target.add_argument(
                flag,
                dest=dest,
                type=str,
                default=None,
                help=help_text,
            )
    init_hp_group.add_argument(
        "--clip-coef-min",
        type=float,
        default=None,
        help=(
            "Override only the lower bound of CLIP_COEF; the upper bound is "
            "kept from the YAML / --clip-coef. Note: on the Liger CISPO "
            "token path the lower bound is unused (CISPO clips weights only "
            "from above), so this flag is a no-op for CISPO."
        ),
    )
    init_hp_group.add_argument(
        "--clip-coef-max",
        type=float,
        default=None,
        help=(
            "Override only the upper bound of CLIP_COEF; the lower bound is "
            "kept from the YAML / --clip-coef. On the Liger CISPO path this "
            "is passed directly as epsilon_high (no offset from 1.0)."
        ),
    )
    init_hp_group.add_argument(
        "--init-hp-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=_parse_key_value_override,
        help=(
            "Additional INIT_HP override; VALUE is parsed as YAML "
            "(e.g. GAE_LAMBDA=1, TARGET_MODULES=[q_proj]). Repeatable."
        ),
    )
    mut_group.add_argument(
        "--mutation-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=_parse_key_value_override,
        help="Additional MUTATION_PARAMS override (VALUE parsed as YAML). Repeatable.",
    )


def _apply_config_overrides(
    init_hp: dict,
    mut_p: dict,
    args: argparse.Namespace,
) -> None:
    """Apply CLI overrides onto loaded YAML INIT_HP / MUTATION_PARAMS dicts."""
    if args.max_model_len is not None:
        init_hp["MAX_MODEL_LEN"] = args.max_model_len
    if args.batch_size is not None:
        init_hp["BATCH_SIZE"] = args.batch_size
    if args.micro_batch_per_gpu is not None:
        init_hp["MICRO_BATCH_SIZE_PER_GPU"] = args.micro_batch_per_gpu
    init_hp["QUANTIZATION"] = args.trainer_quantization
    if args.trainer_activation_offload:
        init_hp["ACTIVATION_OFFLOAD"] = True
    if args.use_memory_efficient_params is not None:
        init_hp["USE_MEMORY_EFFICIENT_PARAMS"] = args.use_memory_efficient_params

    for yaml_key, mutation, _kind, _choices in _CONFIG_ARG_SPECS:
        if not mutation and yaml_key in _INIT_HP_CLI_SKIP:
            continue
        dest = _config_dest_name(yaml_key, mutation=mutation)
        value = getattr(args, dest, None)
        if value is not None:
            (mut_p if mutation else init_hp)[yaml_key] = value

    for key, value in args.init_hp_override:
        init_hp[key] = value
    for key, value in args.mutation_override:
        mut_p[key] = value

    # CLIP_COEF bound-specific overrides run last so they take precedence over
    # both --clip-coef and --init-hp-override CLIP_COEF=... above. Scalar
    # CLIP_COEF expands symmetrically to [1-x, 1+x] (matching the algorithm's
    # own parsing) before the override is applied, so users can sweep just one
    # bound without thinking about tuple syntax.
    if args.clip_coef_min is not None or args.clip_coef_max is not None:
        current = init_hp.get("CLIP_COEF", 0.2)
        if isinstance(current, (int, float)):
            lo, hi = 1.0 - float(current), 1.0 + float(current)
        else:
            lo, hi = float(current[0]), float(current[1])
        if args.clip_coef_min is not None:
            lo = float(args.clip_coef_min)
        if args.clip_coef_max is not None:
            hi = float(args.clip_coef_max)
        init_hp["CLIP_COEF"] = [lo, hi]


def _build_arg_parser() -> argparse.ArgumentParser:
    epilog = """
Quantization stacks (trainer HF vs rollout vLLM are separate copies in memory):

  Trainer (--trainer-quantization): Hugging Face + bitsandbytes via PEFT
    none  - bf16/fp16 base weights
    int8  - LLM.int8() weights
    nf4   - 4-bit NF4 QLoRA recipe (ZeRO-3 friendly)

  Rollout (--vllm-quantization): vLLM weight format at load time
    bitsandbytes - in-flight 4-bit (or load pre-quantized bnb Hub checkpoints;
      use --vllm-dtype bfloat16). This is the AgileRL-validated path; pair
      with --trainer-quantization nf4 for QLoRA rollouts.
    Other vLLM-supported methods (awq, gptq, ...) forward verbatim but are
    not validated by AgileRL — they typically require a pre-quantized
    checkpoint via --vllm-model.
"""
    parser = argparse.ArgumentParser(
        description="Multi-turn LLM benchmarking (HF trainer + optional vLLM rollout).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/llm_finetuning/ppo_llm.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "AgileRL"),
        help="W&B project (default: AgileRL or $WANDB_PROJECT)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity / team (optional; fallback $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("WANDB_RUN_NAME"),
        help="W&B run name (optional; fallback $WANDB_RUN_NAME)",
    )
    parser.add_argument(
        "--wb-level",
        type=str,
        choices=WB_LEVELS,
        default=os.environ.get("WANDB_LEVEL", "standard"),
        help=(
            "W&B logging verbosity tier (fallback $WANDB_LEVEL, default 'standard'). "
            "off=no W&B; essential=scalar metrics only; standard=adds GPU memory, "
            "per-step timing, throughput (cheap, default); detailed=adds histograms "
            "(KL / completion length / reward) with a small per-step host sync; "
            "debug=adds prompt/completion sample tables on every logged step."
        ),
    )

    trainer = parser.add_argument_group(
        "Trainer (Hugging Face + DeepSpeed + bitsandbytes)",
    )
    trainer.add_argument(
        "--trainer-quantization",
        type=str,
        choices=_TRAINER_QUANT_CHOICES,
        default="none",
        help=(
            "HF trainer weight quantization via bitsandbytes (not vLLM). "
            "See --vllm-quantization for rollout weights."
        ),
    )
    trainer.add_argument(
        "--trainer-activation-offload",
        action="store_true",
        help=(
            "HF trainer only: offload activations saved for backward to pinned "
            "host RAM (torch.autograd.graph.save_on_cpu)."
        ),
    )
    trainer.add_argument(
        "--quantization",
        type=str,
        choices=_TRAINER_QUANT_CHOICES,
        default=None,
        help=argparse.SUPPRESS,
    )
    trainer.add_argument(
        "--activation-offload",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    vllm = parser.add_argument_group("Rollout (colocated vLLM)")
    vllm.add_argument(
        "--vllm-quantization",
        type=str,
        default=None,
        metavar="METHOD",
        help=(
            "vLLM rollout weight quantization (forwarded verbatim to vLLM). "
            "'bitsandbytes' is the AgileRL-validated path — quantizes a dense "
            "Hub id at load; pair with --vllm-dtype bfloat16 and "
            "--trainer-quantization nf4 for QLoRA rollouts. Other vLLM "
            "backends (awq, gptq, ...) typically require a pre-quantized "
            "checkpoint via --vllm-model and are not validated by AgileRL."
        ),
    )
    vllm.add_argument(
        "--vllm-model",
        type=str,
        default=None,
        help=(
            "HF id/path for the vLLM engine only. Default: same as --model. "
            "Optional separate Hub id for vLLM only (e.g. AWQ export or same as "
            "--model with --vllm-quantization bitsandbytes)."
        ),
    )
    vllm.add_argument(
        "--vllm-dtype",
        type=str,
        default=None,
        choices=("bfloat16", "float16", "float32", "auto"),
        help="vLLM model weight dtype passed to LLM(dtype=...). Default: vLLM chooses.",
    )
    vllm.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.25,
        help="Fraction of GPU memory vLLM may reserve (weights + KV pool).",
    )
    vllm.add_argument(
        "--vllm-kv-cache-memory-gib",
        type=float,
        default=None,
        help=(
            "Optional pin for vLLM KV cache size in GiB. Default: unset (vLLM "
            "auto-sizes from gpu_memory_utilization). Only set for CI / debugging."
        ),
    )
    vllm.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=1,
        help="vLLM max concurrent sequences during rollout.",
    )
    vllm.add_argument(
        "--vllm-sleep-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Put vLLM to sleep between rollouts. Default: on when POP_SIZE==1 "
            "from config, else off."
        ),
    )
    vllm.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Disable vLLM CUDA-graph capture. Frees the ~2 GiB CUDA-graph "
            "private pool — useful on tight colocated GPU budgets — at a "
            "modest per-decode-step slowdown. Default: vLLM's default."
        ),
    )
    vllm.add_argument(
        "--vllm-strip-multimodal-towers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Free GPU memory used by vision/audio towers of a multimodal base "
            "after vLLM init. Text-only RL never invokes them; typically "
            "frees 1–3 GiB on Gemma-4-MM-class models. Stripped tower attrs "
            "raise loudly if anything ever calls them, so silent corruption "
            "is impossible. Checkpoints unaffected (LoRA-only save)."
        ),
    )
    vllm.add_argument(
        "--weight-sharing",
        action="store_true",
        help=(
            "Zero-copy base-weight sharing: the bnb QLoRA trainer aliases vLLM's "
            "already-quantized base instead of loading its own copy (one base "
            "copy on the GPU; only LoRA adapters are synced per step). Requires "
            "--vllm-quantization bitsandbytes, --trainer-quantization nf4|int8, "
            "and vLLM sleep mode (standby keeps the shared base resident). v1 "
            "shares the language model only."
        ),
    )
    trainer.add_argument(
        "--use-memory-efficient-params",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Move trainer params to CPU during vLLM rollout and back to GPU "
            "for learn. Requires --vllm-sleep-mode. Eliminates trainer + vLLM "
            "GPU coexistence. Default: on (AgileRL default)."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300_000,
        help="Stop training after this many environment steps (default 300k).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HF model id or path for the trainer (and vLLM unless --vllm-model).",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=DEFAULT_ENV_NAME,
        help=(
            f"GEM environment id passed to gem.make() (default: {DEFAULT_ENV_NAME})."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Overrides INIT_HP MAX_MODEL_LEN (sliding-window cap for multiturn RL).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=(
            "Max interaction turns per episode (wrapper + rollout). "
            "Default: GEM environment native max_turns."
        ),
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Stop benchmarking after at most this wall time (training loop exits early).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="finetune_llm_multiturn evaluation_interval (use a huge value to skip test()).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override INIT_HP BATCH_SIZE.",
    )
    parser.add_argument(
        "--micro-batch-per-gpu",
        type=int,
        default=None,
        help="Override INIT_HP MICRO_BATCH_SIZE_PER_GPU.",
    )
    _register_config_override_args(parser)
    return parser


def _normalize_deprecated_cli(args: argparse.Namespace) -> None:
    if args.quantization is not None:
        warnings.warn(
            "--quantization is deprecated; use --trainer-quantization.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.trainer_quantization = args.quantization
    if args.activation_offload:
        warnings.warn(
            "--activation-offload is deprecated; use --trainer-activation-offload.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.trainer_activation_offload = True


def main(
    init_hp,
    mut_p,
    *,
    wb: bool,
    wandb_api_key: str | None,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wb_level: str = "standard",
    trainer_quantization: str = "none",
    vllm_quantization: str | None = None,
    vllm_model: str | None = None,
    vllm_dtype: str | None = None,
    vllm_gpu_memory_utilization: float = 0.25,
    vllm_kv_cache_memory_gib: float | None = None,
    vllm_max_num_seqs: int = 1,
    vllm_sleep_mode: bool | None = None,
    vllm_enforce_eager: bool | None = None,
    vllm_strip_multimodal_towers: bool | None = None,
    weight_sharing: bool = False,
    activation_offload: bool = False,
    model_path: str = DEFAULT_MODEL,
    env_name: str = DEFAULT_ENV_NAME,
    max_steps: int = 300_000,
    max_model_len: int | None = None,
    max_turns: int | None = None,
    max_wall_seconds: float | None = None,
    evaluation_interval: int = 10,
    batch_size: int | None = None,
    micro_batch_per_gpu: int | None = None,
):
    algo_name = init_hp["ALGO"]
    algo_cls = ALGO_REGISTRY.get(algo_name)
    if algo_cls is None:
        msg = f"Unknown algorithm '{algo_name}'. Supported: {', '.join(ALGO_REGISTRY)}"
        raise ValueError(msg)

    quantization_config = build_bnb_quantization_config(trainer_quantization)

    init_hp = {**init_hp}
    if max_model_len is not None:
        init_hp["MAX_MODEL_LEN"] = max_model_len
    if batch_size is not None:
        init_hp["BATCH_SIZE"] = batch_size
    if micro_batch_per_gpu is not None:
        init_hp["MICRO_BATCH_SIZE_PER_GPU"] = micro_batch_per_gpu
    if activation_offload:
        init_hp["ACTIVATION_OFFLOAD"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_env = gem.make(env_name)
    rollout_max_turns = (
        max_turns
        if max_turns is not None
        else init_hp.get("MAX_TURNS", getattr(base_env, "max_turns", 32))
    )

    def env_factory():
        env = gem.make(env_name)
        if "sudoku" in env_name.lower():
            # Fix gem's Sudoku instruction-prompt bugs (subgrid wording + an
            # out-of-range worked example on the easy/4x4 board).
            env.reset = _patch_sudoku_prompt(env.reset)
        return TokenObservationWrapper(
            env,
            tokenizer,
            rollout_max_turns,
            tokenizer.pad_token_id,
            max_model_len=init_hp.get("MAX_MODEL_LEN", None),
            max_output_tokens=init_hp.get("MAX_OUTPUT_TOKENS", None),
            enable_sliding_window=init_hp.get("ENABLE_SLIDING_WINDOW", False),
        )

    accelerator = create_llm_accelerator()

    pop_size = init_hp.get("POP_SIZE", 1)
    if vllm_sleep_mode is None:
        vllm_sleep_mode = pop_size == 1

    vllm_kwargs: dict = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": vllm_gpu_memory_utilization,
        "max_num_seqs": vllm_max_num_seqs,
        "sleep_mode": vllm_sleep_mode,
        "weight_sharing": weight_sharing,
    }
    if vllm_kv_cache_memory_gib is not None:
        vllm_kwargs["kv_cache_memory_bytes"] = int(vllm_kv_cache_memory_gib * (1024**3))
    if vllm_quantization is not None and vllm_quantization.lower() != "none":
        vllm_kwargs["quantization"] = vllm_quantization
    if vllm_model is not None:
        vllm_kwargs["vllm_model_name_or_path"] = vllm_model
    if vllm_dtype is not None and vllm_dtype != "auto":
        vllm_kwargs["dtype"] = vllm_dtype
    if vllm_enforce_eager is not None:
        vllm_kwargs["enforce_eager"] = vllm_enforce_eager
    if vllm_strip_multimodal_towers is not None:
        vllm_kwargs["strip_multimodal_towers"] = vllm_strip_multimodal_towers

    use_vllm = init_hp.get("USE_VLLM", False)

    vllm_config = VLLMConfig(**vllm_kwargs) if use_vllm else None
    algo_kwargs: dict = {}
    if quantization_config is not None:
        algo_kwargs["quantization_config"] = quantization_config

    pop = create_population(
        algo=algo_name,
        net_config=None,
        INIT_HP=init_hp,
        population_size=pop_size,
        accelerator=accelerator,
        tokenizer=tokenizer,
        model_name=model_path,
        vllm_config=vllm_config,
        algo_kwargs=algo_kwargs if algo_kwargs else None,
    )

    finetune_llm_multiturn(
        pop=pop,
        max_turns=rollout_max_turns,
        init_hp=init_hp,
        wb=wb,
        wb_level=wb_level,
        wandb_api_key=wandb_api_key,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        save_elite=True,
        elite_path="saved_llms",
        evo_steps=None,
        mutation=None,
        tournament=None,
        evaluation_interval=evaluation_interval,
        max_reward=1.0,
        verbose=True,
        max_steps=max_steps,
        accelerator=accelerator,
        env_factory=env_factory,
        max_wall_seconds=max_wall_seconds,
    )
    if accelerator is not None:
        accelerator.end_training()


def _default_wandb_run_name(args: argparse.Namespace) -> str:
    parts = [f"tq-{args.trainer_quantization}"]
    if args.vllm_quantization and args.vllm_quantization.lower() != "none":
        parts.append(f"vq-{args.vllm_quantization}")
    return "_".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = _build_arg_parser()
    args = parser.parse_args()
    _normalize_deprecated_cli(args)

    with open(args.config) as file:
        config = yaml.safe_load(file)
    init_hp = dict(config["INIT_HP"])
    mut_p = dict(config["MUTATION_PARAMS"])
    _apply_config_overrides(init_hp, mut_p, args)
    wandb_key = os.environ.get("WANDB_API_KEY")

    run_name = args.wandb_run_name
    if run_name is None:
        run_name = _default_wandb_run_name(args)
    elif args.trainer_quantization != "none":
        run_name = f"{run_name}-tq-{args.trainer_quantization}"

    main(
        init_hp,
        mut_p,
        wb=not args.no_wandb,
        wandb_api_key=wandb_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=run_name,
        wb_level=args.wb_level,
        trainer_quantization=args.trainer_quantization,
        vllm_quantization=args.vllm_quantization,
        vllm_model=args.vllm_model,
        vllm_dtype=args.vllm_dtype,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_kv_cache_memory_gib=args.vllm_kv_cache_memory_gib,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_sleep_mode=args.vllm_sleep_mode,
        vllm_enforce_eager=args.vllm_enforce_eager,
        vllm_strip_multimodal_towers=args.vllm_strip_multimodal_towers,
        weight_sharing=args.weight_sharing,
        activation_offload=args.trainer_activation_offload,
        model_path=args.model,
        env_name=args.env_name,
        max_steps=args.max_steps,
        max_model_len=args.max_model_len,
        max_turns=args.max_turns,
        max_wall_seconds=args.max_wall_seconds,
        evaluation_interval=args.eval_interval,
        batch_size=args.batch_size,
        micro_batch_per_gpu=args.micro_batch_per_gpu,
    )
