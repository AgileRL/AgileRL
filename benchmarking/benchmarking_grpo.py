import re
from typing import Tuple
import yaml
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.algorithms.grpo import GRPO
from agilerl.training.train_llm import finetune_llm, finetune_evolvable_llm
from agilerl.utils.llm_utils import HuggingFaceGym
from agilerl.utils.algo_utils import CosineLRScheduleConfig
from agilerl.utils.utils import create_population
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

MODEL_PATH = "Qwen/Qwen2.5-1.5B"
DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"


def create_model(pretrained_model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    return model


def countdown_chat_template(q, a, tokenizer):
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.",
        },
        {
            "role": "user",
            "content": f"Using each number in this tensor only once {tuple(i.item() for i in q)}, create an equation that equals {a.item()}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>.",
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
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
    raw_dataset = (
        load_dataset(DATASET, split="train").shuffle(seed=42).select(range(50000))
    )
    raw_dataset = raw_dataset.rename_column("target", "answer")
    raw_dataset = raw_dataset.rename_column("nums", "question")
    train_test_split = raw_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset


def format_reward_func(completions, target, **kwargs):
    rewards = []

    for completion, gt in zip(completions, target):

        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, nums, **kwargs):
    rewards = []

    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            answer_tags = re.findall(r"<answer>([\s\S]*?)<\/answer>", completion)

            if len(answer_tags) != 1:
                rewards.append(0.0)
                continue

            equation = answer_tags[0].strip()
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            if sorted(used_numbers) != sorted(numbers.flatten().tolist()):
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
        except Exception as e:
            print(f"Equation error: {e}")
            rewards.append(0.0)
    return rewards


def combined_rewards(completion, solution, prompt):
    reward = (
        equation_reward_func([completion], [solution], [prompt])[0]
        + format_reward_func([completion], [solution])[0]
    )

    print(
        f"""
    ============================================ \n
    Completion: {completion}, \n
    Numbers: {prompt}, \n
    Correct Answer: {solution.item()} \n
    Reward: {reward}
    """
    )

    if reward == 2.0:
        with open("countdown_completions.txt", "a") as text_file:
            text_file.write(
                f"Prompt {prompt}" + "\n" + completion + "\n" + "=" * 50 + "\n"
            )

    return reward


def custom_collate_fn(batch):
    # Extract answers and questions
    answers = torch.tensor([item["answer"] for item in batch])

    # For questions of variable length, we need to pad them
    # First, find the maximum length
    max_len = max(len(item["question"]) for item in batch)

    # Create padded tensor
    questions = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        q_len = len(item["question"])
        questions[i, :q_len] = torch.tensor(item["question"])

    return {"answer": answers, "question": questions}


def main(init_hp, mut_p):
    # Instantiate the model and the associated tokenizer
    model = create_model(**{"pretrained_model_name_or_path": MODEL_PATH})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)
    # Convert the HuggingFace dataset into a Gymnasium environment
    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=combined_rewards,
        apply_chat_template_fn=countdown_chat_template,
        data_batch_size_per_gpu=2,
        custom_collate_fn=custom_collate_fn,
    )
    accelerators = [Accelerator() for _ in range(init_hp["POP_SIZE"])]
    init_hp["actor_network"] = model 
    init_hp["pad_token_id"] = tokenizer.eos_token_id

    hp_config = HyperparameterConfig(
        # lr=RLParameter(
        #     min=mut_p["MIN_LR"], max=mut_p["MAX_LR"]
        # ),
        lr=RLParameter(
            min=mut_p["MIN_LR"], max=mut_p["MAX_LR"]
        ),
    )



    print("creating create_population")
    pop = create_population(
        algo=init_hp["ALGO"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_config=None,
        INIT_HP=init_hp,
        hp_config=hp_config,
        population_size=init_hp["POP_SIZE"],
        accelerator=accelerators
    ) 

    tournament = TournamentSelection(
        init_hp["TOURN_SIZE"],
        init_hp["ELITISM"],
        init_hp["POP_SIZE"],
        init_hp["EVAL_LOOP"],
    )

    mutations = Mutations(
        no_mutation=mut_p["NO_MUT"],
        architecture=mut_p["ARCH_MUT"],
        new_layer_prob=mut_p["NEW_LAYER"],
        parameters=mut_p["PARAMS_MUT"],
        activation=mut_p["ACT_MUT"],
        rl_hp=mut_p["RL_HP_MUT"],
        mutation_sd=mut_p["MUT_SD"],
        rand_seed=mut_p["RAND_SEED"],
        accelerator=accelerators[0],
    )

    finetune_evolvable_llm(
        pop=pop,
        env=env,
        init_hp=init_hp,
        evaluation_interval=10,
        wb=False,
        checkpoint_interval=100,
        checkpoint_path="saved_llms",
        max_reward=2.0,
        evo_steps=1,
        mutation=mutations,
        tournament=tournament,
        accelerator=accelerators[0]
    )

if __name__ == "__main__":
    with open("configs/training/grpo.yaml") as file:
        config = yaml.safe_load(file)
    init_hp = config["INIT_HP"]
    mut_p = config["MUTATION_PARAMS"]
    main(init_hp, mut_p)
