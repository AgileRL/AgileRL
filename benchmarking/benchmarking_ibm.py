import re
from typing import Tuple

import torch
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.training.train_llm import finetune_llm
from agilerl.utils.llm_utils import HuggingFaceGym
from agilerl.utils.utils import create_population

MODEL_PATH = "Qwen/Qwen2.5-0.5B"  # "ibm-granite/granite-3.3-2b-instruct"
DATASET = "openai/gsm8k"


def create_model(pretrained_model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu",
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


def chat_template(q, a, tokenizer):
    conversation = [
        {
            "role": "system",
            "content": "\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>",
        },
        {"role": "user", "content": f"{q}"},
        {
            "role": "assistant",
            "content": "Let me solve this step by step.",
        },
    ]
    updated_prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, continue_final_message=True
    )
    tokenized_prompt = tokenizer(
        [updated_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    )
    return tokenized_prompt


def make_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    """

    Ensure there's only two columns: question and answer.
    """
    raw_dataset = (
        load_dataset(dataset_name, "main", split="train")
        .shuffle(seed=42)
        .select(range(7473))
    )
    # raw_dataset = raw_dataset.rename_column("target", "answer")
    # raw_dataset = raw_dataset.rename_column("nums", "question")
    train_test_split = raw_dataset.train_test_split(test_size=0.01)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset


XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    if "####" in answer:
        answer = answer.split("####")[1].strip().replace(",", "").replace("$", "")
    return answer.split()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


# Reward functions
def correctness_reward_func(completions, prompts, answer, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion for completion in completions]
    return [count_xml(c) for c in contents]


def reward_fn(completion, answer, question):

    answer = extract_hash_answer(answer)

    if answer is None or answer == "":
        return 0.0

    completion = [completion]
    answer = [answer]
    question = [question]

    correctness = correctness_reward_func(completion, question, answer)[0]
    int_r = int_reward_func(completion)[0]
    strict_format_r = strict_format_reward_func(completion)[0]
    soft_format_r = soft_format_reward_func(completion)[0]
    xmlcount_r = xmlcount_reward_func(completion)[0]

    return correctness + int_r + strict_format_r + soft_format_r + xmlcount_r


def main(init_hp, mut_p):
    # Instantiate the model and the associated tokenizer
    model = create_model(pretrained_model_name_or_path=MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)

    # Convert the HuggingFace dataset into a Gymnasium environment
    accelerator = Accelerator()

    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        apply_chat_template_fn=chat_template,
        data_batch_size_per_gpu=20,
        accelerator=accelerator,
        return_raw_completions=init_hp.get("USE_VLLM", False),
    )

    init_hp["PAD_TOKEN_ID"] = tokenizer.eos_token_id
    init_hp["PAD_TOKEN"] = tokenizer.eos_token

    hp_config = HyperparameterConfig(
        beta=RLParameter(min=mut_p["MIN_BETA"], max=mut_p["MAX_BETA"]),
        lr=RLParameter(min=mut_p["MIN_LR"], max=mut_p["MAX_LR"]),
        group_size=RLParameter(
            min=mut_p["MIN_GROUP_SIZE"], max=mut_p["MAX_GROUP_SIZE"], dtype=int
        ),
    )
    pop = create_population(
        algo=init_hp["ALGO"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_config=None,
        actor_network=model,
        INIT_HP=init_hp,
        hp_config=hp_config,
        population_size=init_hp["POP_SIZE"],
        accelerator=accelerator,
    )

    del model

    # tournament = TournamentSelection(
    #     init_hp["TOURN_SIZE"],
    #     init_hp["ELITISM"],
    #     init_hp["POP_SIZE"],
    #     init_hp["EVAL_LOOP"],
    # )

    # mutations = Mutations(
    #     no_mutation=mut_p["NO_MUT"],
    #     architecture=0,
    #     new_layer_prob=0,
    #     parameters=0,
    #     activation=0,
    #     rl_hp=mut_p["RL_HP_MUT"],
    #     mutation_sd=mut_p["MUT_SD"],
    #     rand_seed=mut_p["RAND_SEED"],
    #     accelerator=accelerator,
    # )

    finetune_llm(
        pop=pop,
        env=env,
        init_hp=init_hp,
        evaluation_interval=10,
        wb=False,
        save_elite=True,
        elite_path="saved_llms",
        max_reward=4.0,
        evo_steps=10,
        checkpoint_steps=1000,
        mutation=None,  # mutations,
        tournament=None,  # tournament,
        accelerator=accelerator,
        verbose=True,
        num_epochs=1,
    )
    accelerator.end_training()


if __name__ == "__main__":
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with open("configs/training/grpo.yaml") as file:
        config = yaml.safe_load(file)
    init_hp = config["INIT_HP"]
    mut_p = config["MUTATION_PARAMS"]
    main(init_hp, mut_p)
