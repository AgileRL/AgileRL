from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

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
from agilerl.utils.llm_utils import PreferenceGym

MODEL_PATH = "Qwen/Qwen2.5-0.5B"
DATASET = "HumanLLMs/Human-Like-DPO-Dataset"


def make_dataset(dataset_name: str) -> tuple[Dataset, Dataset]:
    raw_dataset = load_dataset(dataset_name, split="train").shuffle(seed=42)
    train_test_split = raw_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset


def main(init_hp, mut_p):
    # Instantiate the model and the associated tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)

    # Convert the HuggingFace dataset into a Gymnasium environment
    accelerator = Accelerator()
    env = PreferenceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=init_hp["BATCH_SIZE"],
        accelerator=accelerator,
    )

    init_hp["PAD_TOKEN_ID"] = tokenizer.eos_token_id
    init_hp["PAD_TOKEN"] = tokenizer.eos_token
    init_hp["ZERO_STAGE"] = accelerator.state.deepspeed_plugin.deepspeed_config[
        "zero_optimization"
    ]["stage"]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    pop = [
        DPO(
            model_name=MODEL_PATH,
            pad_token_id=init_hp["PAD_TOKEN_ID"],
            pad_token=init_hp["PAD_TOKEN"],
            batch_size=init_hp["BATCH_SIZE"],
            beta=init_hp["BETA"],
            update_epochs=init_hp["UPDATE_EPOCHS"],
            lora_config=lora_config,
            accelerator=accelerator if idx == 0 else Accelerator(),
        )
        for idx, _ in enumerate(range(init_hp["POP_SIZE"]))
    ]

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

    finetune_llm_preference(
        pop=pop,
        env=env,
        init_hp=init_hp,
        save_elite=True,
        elite_path="saved_llms",
        wb=True,
        evo_steps=5,
        tournament=tournament,
        mutation=mutations,
        wandb_api_key=None,
        evaluation_interval=10,
        accelerator=accelerator,
        num_epochs=1,
    )


if __name__ == "__main__":
    with open("configs/training/dpo.yaml") as file:
        config = yaml.safe_load(file)
    init_hp = config["INIT_HP"]
    mut_p = config["MUTATION_PARAMS"]
    main(init_hp, mut_p)
    main(init_hp=init_hp, mut_p=mut_p)
