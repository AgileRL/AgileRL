from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    raise ImportError(
        "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`.",
    )

import argparse
import re

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import benchmark_cli_llm
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_llm import finetune_llm_reasoning
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import ReasoningGym, create_llm_accelerator
from agilerl.utils.utils import create_population

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
DEFAULT_CONFIG = "configs/training/llm_finetuning/ppo_llm.yaml"

# Reasoning is throughput-oriented (batched single-turn generation), so it runs
# vLLM with a larger memory budget and many concurrent sequences by default.
REASONING_VLLM_DEFAULTS = {
    "gpu_memory_utilization": 0.8,
    "max_num_seqs": 12,
    "sleep_mode": True,
}


def make_dataset(dataset_name: str) -> tuple[Dataset, Dataset]:
    raw_dataset = (
        load_dataset(dataset_name, split="train").shuffle(seed=42).select(range(50000))
    )
    raw_dataset = raw_dataset.rename_column("target", "answer")
    raw_dataset = raw_dataset.rename_column("nums", "question")
    train_test_split = raw_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset


def format_reward_func(completions, target, **kwargs):
    rewards = []
    for completion, _gt in zip(completions, target, strict=False):
        try:
            completion = "<think>" + completion
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            matches = re.search(regex, completion, re.DOTALL)
            if matches is None or len(matches.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:  # noqa: PERF203
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, nums, **kwargs):
    rewards = []

    for completion, gt, numbers in zip(completions, target, nums, strict=False):
        try:
            completion = "<think>" + completion
            answer_tags = re.findall(r"<answer>([\s\S]*?)<\/answer>", completion)

            if len(answer_tags) != 1:
                rewards.append(0.0)
                continue

            equation = answer_tags[0].strip()
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue

            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue

            result = eval(equation, {"__builtins__": None}, {})

            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def combined_rewards(completion, solution, prompt):
    return (
        equation_reward_func([completion], [solution], [prompt])[0]
        + format_reward_func([completion], [solution])[0]
    )


def _add_reasoning_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset id (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="finetune_llm_reasoning evaluation_interval.",
    )
    parser.add_argument(
        "--evo-steps",
        type=int,
        default=4,
        help="Evolution frequency (steps between tournament + mutation).",
    )


def main(resolved: benchmark_cli_llm.LLMBenchmarkConfig) -> None:
    args = resolved.args
    init_hp = resolved.init_hp
    mut_p = resolved.mutation_params
    model_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset, test_dataset = make_dataset(args.dataset)

    conversation_template = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.",
        },
        {
            "role": "user",
            "content": "Using each number in this list only once {question}, create an equation that equals {answer}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>.",
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]

    accelerator = create_llm_accelerator()
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=combined_rewards,
        conversation_template=conversation_template,
        data_batch_size_per_gpu=init_hp["BATCH_SIZE"],
        accelerator=accelerator,
        max_context_length=init_hp["MAX_MODEL_LEN"],
    )

    use_vllm = bool(init_hp.get("USE_VLLM", True))
    vllm_config = VLLMConfig(**resolved.build_vllm_kwargs()) if use_vllm else None
    hp_config = HyperparameterConfig(
        beta=RLParameter(min=mut_p["MIN_BETA"], max=mut_p["MAX_BETA"]),
        lr=RLParameter(min=mut_p["MIN_LR"], max=mut_p["MAX_LR"]),
        lr_critic=RLParameter(min=mut_p["MIN_LR_CRITIC"], max=mut_p["MAX_LR_CRITIC"]),
    )

    pop = create_population(
        algo=init_hp["ALGO"],
        net_config=None,
        INIT_HP=init_hp,
        hp_config=hp_config,
        population_size=init_hp["POP_SIZE"],
        accelerator=accelerator,
        tokenizer=tokenizer,
        model_name=model_path,
        vllm_config=vllm_config,
    )

    tournament = TournamentSelection(
        init_hp["TOURN_SIZE"],
        init_hp["ELITISM"],
        init_hp["POP_SIZE"],
        init_hp["EVAL_LOOP"],
    )
    mutations = Mutations(
        no_mutation=mut_p["NO_MUT"],
        architecture=mut_p.get("ARCH_MUT", 0.0),
        new_layer_prob=mut_p.get("NEW_LAYER", 0.0),
        parameters=mut_p.get("PARAMS_MUT", 0.0),
        activation=mut_p.get("ACT_MUT", 0.0),
        rl_hp=mut_p.get("RL_HP_MUT", 0.0),
        mutation_sd=mut_p.get("MUT_SD", 0.0),
        rand_seed=mut_p.get("RAND_SEED", 42),
        device=accelerator.device,
    )

    finetune_llm_reasoning(
        pop=pop,
        env=env,
        init_hp=init_hp,
        evaluation_interval=args.eval_interval,
        wb=resolved.wandb.enabled,
        wb_level=args.wb_level,
        wandb_api_key=resolved.wandb.api_key,
        wandb_project=resolved.wandb.project,
        wandb_entity=resolved.wandb.entity,
        wandb_run_name=resolved.wandb.run_name,
        save_elite=True,
        elite_path="saved_llms",
        max_reward=2.0,
        evo_steps=args.evo_steps,
        mutation=mutations,
        tournament=tournament,
        accelerator=accelerator,
        verbose=True,
    )
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    resolved = benchmark_cli_llm.parse_llm_benchmark_cli(
        default_config=DEFAULT_CONFIG,
        default_model=DEFAULT_MODEL,
        description="Reasoning (Countdown) LLM benchmarking with evolutionary HPO.",
        add_script_arguments=_add_reasoning_arguments,
        vllm_defaults=REASONING_VLLM_DEFAULTS,
    )
    main(resolved)
