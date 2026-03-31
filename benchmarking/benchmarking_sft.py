"""Benchmarking script for Supervised Fine-Tuning (SFT).

Pipeline
--------
Dataset  →  SFTGym  →  SFT.learn()  →  cross-entropy on response tokens only

SFT datasets contain plain ``(prompt, response)`` pairs — just prompts and
their desired target responses.  No rejected/negative responses are needed.

The dataset used here (``HumanLLMs/Human-Like-DPO-Dataset``) happens to be a
DPO-style preference dataset with ``prompt``, ``chosen``, and ``rejected``
columns.  We use only ``prompt`` + ``chosen`` by passing
``response_column="chosen"`` to ``SFTGym``; the ``rejected`` column is
completely ignored.

This is a natural fit for the typical two-stage alignment pipeline:

    Stage 1 — SFT (this script):  train on (prompt, chosen)
    Stage 2 — DPO              :  further align on (prompt, chosen, rejected)

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

import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from agilerl.algorithms.sft import SFT
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_llm import finetune_llm_sft
from agilerl.utils.llm_utils import SFTGym

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# A small model suitable for quick iteration.  Swap for a larger one as needed.
MODEL_PATH = "Qwen/Qwen2.5-0.5B"

# A DPO-style dataset reused here for its "prompt" + "chosen" columns.
# Any dataset with a "prompt" column and a response column works with SFTGym.
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


def main(init_hp: dict, mut_p: dict) -> None:
    """Run the SFT benchmarking loop.

    :param init_hp: Initial hyperparameter dict loaded from the YAML config.
    :type init_hp: dict
    :param mut_p: Mutation parameter dict loaded from the YAML config.
    :type mut_p: dict
    """
    # Tokenizer -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)

    # Use accelerate only when launched with `accelerate launch` (i.e. when
    # DeepSpeed / multi-GPU is configured).  For a plain single-GPU run via
    # `uv run python benchmarking/benchmarking_sft.py` pass accelerator=None
    # so the base class falls back to standard single-GPU training.
    try:
        accelerator = Accelerator()
        # Probe for DeepSpeed — if not configured the plugin is None and we
        # fall back to accelerator=None to avoid the deepspeed_config access
        # inside LLMAlgorithm._configure_batch_size.
        if accelerator.state.deepspeed_plugin is None:
            accelerator = None
    except Exception:
        accelerator = None

    # SFTGym expects (prompt, response) pairs.  This dataset uses "chosen" as
    # the column name for the good response, so we specify that explicitly.
    # A pure SFT dataset would typically use "response" (the default).
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

    # LoRA config ---------------------------------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    # Population ----------------------------------------------------------
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

    # Training loop -------------------------------------------------------
    # NUM_BATCHES=None → full epoch; NUM_BATCHES=N → exactly N batches.
    num_batches = init_hp.get("NUM_BATCHES")
    max_steps = num_batches * init_hp["BATCH_SIZE"] if num_batches is not None else None

    print("Finetuning SFT agents...")
    finetune_llm_sft(
        pop=pop,
        env=env,
        init_hp=init_hp,
        save_elite=True,
        elite_path="saved_llms/sft",
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


if __name__ == "__main__":
    with open("configs/training/sft.yaml") as f:
        config = yaml.safe_load(f)
    main(init_hp=config["INIT_HP"], mut_p=config["MUTATION_PARAMS"])
