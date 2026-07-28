import re

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from agilerl.algorithms import GRPO
from agilerl.training.llm import train_llm_rollout
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.llm_envs import RewardComponent, RolloutEnv

MODEL_PATH = "Qwen/Qwen2.5-0.5B"
DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
USE_VLLM = True
MAX_CONTEXT_LENGTH = 1024


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


def format_reward_func(completion: str, answer: str, question: str) -> float:
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 2:
            return 0.0
        return 1.0
    except Exception:  # noqa: PERF203
        return 0.0


def equation_reward_func(completion: str, answer: str, question: str) -> float:
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        answer_tags = re.findall(r"<answer>([\s\S]*?)<\/answer>", completion)

        if len(answer_tags) != 1:
            return 0.0

        equation = answer_tags[0].strip()
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
        numbers = (
            question.flatten().tolist()
            if hasattr(question, "flatten")
            else list(question)
        )

        if sorted(used_numbers) != sorted(numbers):
            return 0.0

        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        result = eval(equation, {"__builtins__": None}, {})
        return 1.0 if abs(float(result) - float(answer)) < 1e-5 else 0.0
    except Exception:
        return 0.0


def main():
    # Instantiate the model and the associated tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)

    # Convert the HuggingFace dataset into a Gymnasium environment
    accelerator = Accelerator()

    # Define the conversation template
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

    def prompt_builder(row: dict) -> str:
        parts = [
            m["content"].format(question=row["question"], answer=row["answer"])
            for m in conversation_template
        ]
        return "\n".join(p for p in parts if p)

    # Reasoning is a single-turn rollout env: (question, answer) rows + a reward fn,
    # no pre-packaged env needed — ``from_dataset`` bundles them into one. A plain
    # scalar reward_fn yields one learning reward with no per-component metrics; a
    # list of RewardComponent keeps that single learning reward (weighted sum) and
    # adds granular train/mean_reward/<name> reporting. The BatchRolloutEnv owns
    # the dataset cursor, giving each GRPO group a deterministic, group-consistent
    # dataset order.
    def env_factory(evaluation_mode: bool = False):
        env = RolloutEnv.from_dataset(
            train_dataset,
            [
                # Learning reward = sum(weight * fn(...)); raw values still reported
                # as train/mean_reward/<name>.
                RewardComponent("format", format_reward_func, weight=1.0),
                RewardComponent("equation", equation_reward_func, weight=1.0),
            ],
            tokenizer,
            test_dataset=test_dataset,
            prompt_builder=prompt_builder,
            pad_id=getattr(tokenizer, "pad_token_id", None),
            apply_chat_template=True,
            max_model_len=MAX_CONTEXT_LENGTH,
        )
        env.evaluation_mode = evaluation_mode
        return env

    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
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

    # Instantiate the grpo agent
    agent = GRPO(
        model_name=MODEL_PATH,
        pad_token_id=tokenizer.eos_token_id,
        pad_token=tokenizer.eos_token,
        lora_config=lora_config,
        batch_size=16,
        max_model_len=MAX_CONTEXT_LENGTH,
        group_size=8,
        accelerator=accelerator,
        use_vllm=USE_VLLM,
        vllm_config=VLLMConfig(sleep_mode=True, max_num_seqs=4),
    )
    train_llm_rollout(
        pop=[agent],
        max_turns=1,
        env_factory=env_factory,
        evaluation_interval=10,
        wb=True,
        save_elite=True,
        elite_path="checkpoints",
        max_reward=2.0,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    main()
