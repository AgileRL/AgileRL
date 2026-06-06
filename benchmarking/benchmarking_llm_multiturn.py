from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import argparse
import logging

import gem
from transformers import AutoTokenizer

import benchmark_cli_llm
from agilerl.algorithms import CISPO, GRPO, GSPO, LLMPPO, LLMREINFORCE
from agilerl.llm_envs import TokenObservationWrapper
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import (
    build_bnb_quantization_config,
    create_llm_accelerator,
)
from agilerl.utils.utils import create_population

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ENV_NAME = "game:Sudoku-v0-hard"
DEFAULT_CONFIG = "configs/training/llm_finetuning/ppo_llm.yaml"

ALGO_REGISTRY = {
    "LLMPPO": LLMPPO,
    "LLMREINFORCE": LLMREINFORCE,
    "GRPO": GRPO,
    "CISPO": CISPO,
    "GSPO": GSPO,
}

EPILOG = """
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

Hyperparameter flags (--lr, --group-size, --batch-size, ...) and mutation flags
(--mut-no-mut, ...) are generated from the INIT_HP / MUTATION_PARAMS dataclasses
in benchmark_cli_llm; run with --help to see the full set for the selected algo,
or --print-config to dump the resolved config. Any unmodelled key is still
reachable via --init-hp-override KEY=VALUE.
"""


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


def _add_multiturn_arguments(parser: argparse.ArgumentParser) -> None:
    """Script-specific flags layered on top of the shared LLM benchmark CLI."""
    parser.add_argument(
        "--env-name",
        type=str,
        default=DEFAULT_ENV_NAME,
        help=f"GEM environment id passed to gem.make() (default: {DEFAULT_ENV_NAME}).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=(
            "Max interaction turns per episode (wrapper + rollout). "
            "Default: INIT_HP MAX_TURNS, else the GEM environment native max_turns."
        ),
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="finetune_llm_multiturn evaluation_interval (use a huge value to skip test()).",
    )


def _default_run_name(init_hp: dict, args: argparse.Namespace) -> str:
    parts = [f"tq-{init_hp.get('QUANTIZATION', 'none')}"]
    if args.vllm_quantization and args.vllm_quantization.lower() != "none":
        parts.append(f"vq-{args.vllm_quantization}")
    return "_".join(parts)


def main(resolved: benchmark_cli_llm.LLMBenchmarkConfig) -> None:
    args = resolved.args
    init_hp = resolved.init_hp

    algo_name = init_hp["ALGO"]
    if ALGO_REGISTRY.get(algo_name) is None:
        msg = f"Unknown algorithm '{algo_name}'. Supported: {', '.join(ALGO_REGISTRY)}"
        raise ValueError(msg)

    quantization_config = build_bnb_quantization_config(
        init_hp.get("QUANTIZATION", "none")
    )

    model_path = args.model
    env_name = args.env_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_env = gem.make(env_name)
    rollout_max_turns = (
        args.max_turns
        if args.max_turns is not None
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

    use_vllm = init_hp.get("USE_VLLM", False)
    vllm_config = VLLMConfig(**resolved.build_vllm_kwargs()) if use_vllm else None

    algo_kwargs: dict = {}
    if quantization_config is not None:
        algo_kwargs["quantization_config"] = quantization_config

    pop = create_population(
        algo=algo_name,
        net_config=None,
        INIT_HP=init_hp,
        population_size=init_hp.get("POP_SIZE", 1),
        accelerator=accelerator,
        tokenizer=tokenizer,
        model_name=model_path,
        vllm_config=vllm_config,
        algo_kwargs=algo_kwargs if algo_kwargs else None,
    )

    run_name = resolved.wandb.run_name
    quant = init_hp.get("QUANTIZATION", "none")
    if run_name is None:
        run_name = _default_run_name(init_hp, args)
    elif quant != "none":
        run_name = f"{run_name}-tq-{quant}"

    finetune_llm_multiturn(
        pop=pop,
        max_turns=rollout_max_turns,
        init_hp=init_hp,
        wb=resolved.wandb.enabled,
        wb_level=args.wb_level,
        wandb_api_key=resolved.wandb.api_key,
        wandb_project=resolved.wandb.project,
        wandb_entity=resolved.wandb.entity,
        wandb_run_name=run_name,
        save_elite=True,
        elite_path="saved_llms",
        evo_steps=None,
        mutation=None,
        tournament=None,
        evaluation_interval=args.eval_interval,
        max_reward=1.0,
        verbose=True,
        max_steps=args.max_steps,
        accelerator=accelerator,
        env_factory=env_factory,
        max_wall_seconds=args.max_wall_seconds,
    )
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resolved = benchmark_cli_llm.parse_llm_benchmark_cli(
        default_config=DEFAULT_CONFIG,
        default_model=DEFAULT_MODEL,
        description="Multi-turn LLM benchmarking (HF trainer + optional vLLM rollout).",
        epilog=EPILOG,
        add_script_arguments=_add_multiturn_arguments,
    )
    main(resolved)
