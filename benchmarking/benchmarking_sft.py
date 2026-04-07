"""Benchmarking script for Supervised Fine-Tuning (SFT).

Pipeline
--------
Dataset  →  SFTGym  →  SFT.learn()  →  cross-entropy on response tokens only

SFT datasets contain plain ``(prompt, response)`` pairs — just prompts and
their desired target responses.  No rejected/negative responses are needed.

To run (single GPU, no accelerate):
    python benchmarking/benchmarking_sft.py

To run with accelerate (multi-GPU / DeepSpeed):
    accelerate launch benchmarking/benchmarking_sft.py
"""

from agilerl import HAS_LLM_DEPENDENCIES


if not HAS_LLM_DEPENDENCIES:
    msg = (
        "LLM dependencies are not installed. "
        "Install them with `pip install agilerl[llm]`."
    )
    raise ImportError(msg)

import argparse
import json
from datetime import datetime

import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agilerl.algorithms.sft import SFT
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_llm import finetune_llm_sft
from agilerl.utils.llm_utils import SFTGym, compare_responses, sample_eval_prompts

MODEL_PATH = "Qwen/Qwen2.5-0.5B"
DATASET = "HumanLLMs/Human-Like-DPO-Dataset"
# A DPO-style dataset reused here for its "prompt" + "chosen" columns.
# Any dataset with a "prompt" column and a (desirable) response column works with SFTGym.


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
    """Run the SFT benchmarking loop.

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

    # Use accelerate only when launched with `accelerate launch` else None
    try:
        accelerator = Accelerator()
        if accelerator.state.deepspeed_plugin is None:
            accelerator = None
    except Exception:
        accelerator = None

    print("Setting up SFTGym environment...")
    env = SFTGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=init_hp["BATCH_SIZE"],
        response_column="chosen",
        accelerator=accelerator,
        max_context_length=init_hp.get("MAX_CONTEXT_LENGTH"),
    )

    init_hp["PAD_TOKEN_ID"] = tokenizer.eos_token_id
    init_hp["PAD_TOKEN"] = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    print("Defining SFT agent population...")
    pop = [
        SFT(
            model_name=MODEL_PATH,
            pad_token_id=init_hp["PAD_TOKEN_ID"],
            pad_token=init_hp["PAD_TOKEN"],
            batch_size=init_hp["BATCH_SIZE"],
            update_epochs=init_hp["UPDATE_EPOCHS"],
            lora_config=lora_config,
            accelerator=accelerator,
            # gradient_checkpointing needs at least one float input with
            # requires_grad=True; disable for plain single-GPU runs where
            # memory pressure is not a concern.
            gradient_checkpointing=accelerator is not None,
            use_liger_loss=init_hp.get("USE_LIGER_LOSS", False),
        )
        for _ in range(init_hp["POP_SIZE"])
    ]

    # HPO objects — only constructed when evolution is enabled ---------------
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

    print("Finetuning SFT agents...")
    finetune_llm_sft(
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
        plot_path=init_hp.get("PLOT_PATH"),
    )

    print("\nQualitative response comparison (elite agent):")
    elite = max(pop, key=lambda a: a.fitness[-1] if a.fitness else float("-inf"))
    eval_samples = sample_eval_prompts(env, n=init_hp.get("EVAL_N_SAMPLES", 5))
    compare_responses(elite, tokenizer, eval_samples)


def eval_mode(load_path: str, max_new_tokens: int = 200) -> None:
    """Load a saved LoRA checkpoint and enter an interactive prompt loop.

    Loads the base model and LoRA adapter from *load_path*, then repeatedly
    reads a prompt from stdin and prints the model response (via
    :func:`~agilerl.utils.llm_utils.compare_responses`; base-model output is
    omitted in eval mode because it duplicates the merged adapter output).
    Type ``quit``, ``q``, ``exit``, or press **Ctrl+C** to exit.

    :param load_path: Path to a directory containing ``adapter_config.json``
        and the LoRA adapter weights saved by a prior SFT run.
    :type load_path: str
    :param max_new_tokens: Maximum tokens to generate per response, defaults to 200.
    :type max_new_tokens: int, optional
    """
    with open(f"{load_path}/adapter_config.json") as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    print(f"Loading tokenizer from {base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {base_model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    print(f"Applying LoRA adapter from {load_path} ...")
    actor_network = PeftModel.from_pretrained(base_model, load_path)

    agent = SFT(
        actor_network=actor_network,
        pad_token_id=tokenizer.eos_token_id,
        pad_token=tokenizer.eos_token,
    )

    print(f"\nEval mode ready  |  base: {base_model_name}  |  adapter: {load_path}")
    print("Enter a prompt and press Enter to generate a response.")
    print("Type 'quit', 'q', 'exit', or press Ctrl+C to quit.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting eval mode.")
            break

        if not prompt or prompt.lower() in {"quit", "q", "exit"}:
            print("Exiting eval mode.")
            break

        compare_responses(
            agent,
            tokenizer,
            [(prompt, None, None)],
            max_new_tokens=max_new_tokens,
            show_base_model=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT benchmarking script")
    parser.add_argument(
        "--save-path",
        default="outputs",
        help="Base directory to save elite LoRA checkpoint (default: outputs)",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable the timestamp sub-directory, overwriting any existing checkpoint at --save-path",
    )
    parser.add_argument(
        "--load-path",
        default=None,
        help=(
            "Path to a saved LoRA checkpoint to load. Required when --eval is set. "
            "Must contain adapter_model.safetensors / adapter_config.json."
        ),
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help=(
            "Eval mode: load an existing LoRA checkpoint (--load-path) and enter "
            "an interactive prompt loop instead of training."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per response in eval mode (default: 200)",
    )
    parser.add_argument(
        "--config",
        default="configs/training/sft.yaml",
        help="Path to YAML config file (default: configs/training/sft.yaml)",
    )
    args = parser.parse_args()

    if args.eval:
        if not args.load_path:
            parser.error("--load-path is required when --eval is set")
        eval_mode(load_path=args.load_path, max_new_tokens=args.max_new_tokens)
    else:
        save_path = (
            args.save_path
            if args.no_timestamp
            else f"{args.save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_SFT"
        )

        with open(args.config) as f:
            config = yaml.safe_load(f)
        main(
            init_hp=config["INIT_HP"],
            mut_p=config["MUTATION_PARAMS"],
            save_path=save_path,
        )
