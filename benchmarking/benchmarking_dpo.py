"""Benchmarking script for Direct Preference Optimization (DPO).

Runs DPO with default hyperparameters from ``configs/training/dpo.yaml``.
For full CLI options (custom save paths, checkpoint warm-starting, eval mode),
use the demo script instead::

    python demos/demo_llm_finetuning.py dpo --help

To run (single GPU, no accelerate):
    python benchmarking/benchmarking_dpo.py

To run with accelerate (multi-GPU / DeepSpeed):
    accelerate launch benchmarking/benchmarking_dpo.py
"""

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = (
        "LLM dependencies are not installed. "
        "Install them with `pip install agilerl[llm]`."
    )
    raise ImportError(msg)

from datetime import datetime

import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from agilerl.algorithms.dpo import DPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_llm import finetune_llm_preference
from agilerl.utils.llm_utils import (
    PreferenceGym,
    compare_responses,
    sample_eval_prompts,
)

MODEL_PATH = "Qwen/Qwen2.5-0.5B"
DATASET = "HumanLLMs/Human-Like-DPO-Dataset"


def make_dataset(dataset_name: str) -> tuple[Dataset, Dataset]:
    """Download and split the dataset into train / test subsets.

    :param dataset_name: HuggingFace dataset identifier.
    :type dataset_name: str
    :return: ``(train_dataset, test_dataset)`` HuggingFace ``Dataset`` objects.
    :rtype: tuple[Dataset, Dataset]
    """
    raw = load_dataset(dataset_name, split="train").shuffle(seed=42)
    splits = raw.train_test_split(test_size=0.1)
    return splits["train"], splits["test"]


def main(init_hp: dict, mut_p: dict, save_path: str = "outputs") -> None:
    """Run the DPO benchmarking loop.

    :param init_hp: Initial hyperparameter dict loaded from the YAML config.
    :type init_hp: dict
    :param mut_p: Mutation parameter dict loaded from the YAML config.
    :type mut_p: dict
    :param save_path: Directory to save elite LoRA checkpoint, defaults to ``"outputs"``.
    :type save_path: str
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)

    try:
        accelerator = Accelerator()
        if accelerator.state.deepspeed_plugin is None:
            accelerator = None
    except Exception:
        accelerator = None

    print("Setting up PreferenceGym environment...")
    env = PreferenceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=init_hp["BATCH_SIZE"],
        accelerator=accelerator,
        max_context_length=init_hp.get("MAX_CONTEXT_LENGTH"),
    )

    init_hp["PAD_TOKEN_ID"] = tokenizer.eos_token_id
    init_hp["PAD_TOKEN"] = tokenizer.eos_token

    lora_config = LoraConfig(**init_hp["LORA"])

    print("Setting up DPO agent population...")
    pop = [
        DPO(
            model_name=MODEL_PATH,
            pad_token_id=init_hp["PAD_TOKEN_ID"],
            pad_token=init_hp["PAD_TOKEN"],
            batch_size=init_hp["BATCH_SIZE"],
            lr=init_hp["LR"],
            beta=init_hp["BETA"],
            nll_alpha=init_hp.get("NLL_ALPHA", 1.0),
            update_epochs=init_hp["UPDATE_EPOCHS"],
            lora_config=lora_config,
            accelerator=accelerator,
            gradient_checkpointing=accelerator is not None,
            use_liger_loss=init_hp.get("USE_LIGER_LOSS", False),
        )
        for _ in range(init_hp["POP_SIZE"])
    ]

    evo_steps = init_hp.get("EVO_STEPS")
    if evo_steps is not None:
        tournament = TournamentSelection(
            init_hp["TOURN_SIZE"],
            init_hp["ELITISM"],
            init_hp["POP_SIZE"],
            init_hp["EVAL_LOOP"],
        )
        mutations = Mutations(
            no_mutation=mut_p["NO_MUT"],
            architecture=0,
            new_layer_prob=0,
            parameters=0,
            activation=0,
            rl_hp=mut_p["RL_HP_MUT"],
            mutation_sd=mut_p["MUT_SD"],
            rand_seed=mut_p["RAND_SEED"],
            accelerator=accelerator,
        )
    else:
        tournament = None
        mutations = None

    num_batches = init_hp.get("NUM_BATCHES")
    max_steps = num_batches * init_hp["BATCH_SIZE"] if num_batches is not None else None

    print("Fine-tuning DPO agents...")
    finetune_llm_preference(
        pop=pop,
        env=env,
        init_hp=init_hp,
        save_elite=True,
        elite_path=save_path,
        wb=init_hp.get("WANDB", False),
        evo_steps=evo_steps,
        tournament=tournament,
        mutation=mutations,
        wandb_api_key=init_hp.get("WANDB_API_KEY"),
        wandb_project=init_hp.get("WANDB_PROJECT", "AgileRL"),
        wandb_entity=init_hp.get("WANDB_ENTITY"),
        wandb_run_name=init_hp.get("WANDB_RUN_NAME"),
        evaluation_interval=init_hp.get("EVALUATION_INTERVAL", 200),
        accelerator=accelerator,
        max_steps=max_steps,
    )

    print("\nQualitative response comparison (elite agent):")
    elite = max(pop, key=lambda a: a.fitness[-1] if a.fitness else float("-inf"))
    eval_samples = sample_eval_prompts(env, n=init_hp.get("EVAL_N_SAMPLES", 5))
    compare_responses(elite, tokenizer, eval_samples)


if __name__ == "__main__":
    save_path = f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_DPO"
    with open("configs/training/dpo.yaml") as f:
        config = yaml.safe_load(f)
    main(
        init_hp=config["INIT_HP"],
        mut_p=config["MUTATION_PARAMS"],
        save_path=save_path,
    )
