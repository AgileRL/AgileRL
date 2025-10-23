import torch
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agilerl.algorithms.dpo import DPO
from agilerl.training.train_llm import finetune_llm_preference
from agilerl.utils.llm_utils import PreferenceGym

MODEL_PATH = "Qwen/Qwen2.5-3B"
DATASET = "HumanLLMs/Human-Like-DPO-Dataset"


def create_model(pretrained_model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config, adapter_name="actor")

    return model


def make_dataset(dataset_name: str) -> tuple[Dataset, Dataset]:
    raw_dataset = load_dataset(dataset_name, split="train").shuffle(seed=42)
    train_test_split = raw_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset


def main(init_hp, mut_p):
    # Instantiate the model and the associated tokenizer
    model = create_model(pretrained_model_name_or_path=MODEL_PATH)
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
    from gymnasium import spaces

    pop = [
        DPO(
            observation_space=spaces.Box(low=0, high=tokenizer.vocab_size - 1),
            action_space=spaces.Box(low=0, high=tokenizer.vocab_size - 1),
            actor_network=model,
            pad_token_id=init_hp["PAD_TOKEN_ID"],
            pad_token=init_hp["PAD_TOKEN"],
            batch_size=init_hp["BATCH_SIZE"],
            beta=init_hp["BETA"],
            update_epochs=init_hp["UPDATE_EPOCHS"],
            accelerator=accelerator,
        )
    ]

    # while True:
    #     import time
    #     time.sleep(1000000)

    finetune_llm_preference(
        pop=pop,
        env=env,
        init_hp=init_hp,
        save_elite=True,
        elite_path="saved_llms",
        wb=False,
        evo_steps=1000,
        tournament=None,
        mutation=None,
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
