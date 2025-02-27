import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from agilerl.algorithms import GRPO
from agilerl.modules.dummy import to_evolvable
from agilerl.training.train_llm import finetune_llm
from agilerl.utils.llm_utils import HuggingFaceGym

MODEL_PATH = "Qwen/Qwen2-0.5B"
DATASET = "openai/gsm8k"


# def create_module(pretrained_model_name_or_path):
#     model = AutoModelForCausalLM.from_pretrained(
#         pretrained_model_name_or_path=pretrained_model_name_or_path,
#         device_map="cuda",
#         attn_implementation="flash_attention_2",
#         quantization_config=BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#         ),
#     )
#     lora_config = LoraConfig(
#         task_type="CAUSAL_LM",
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.1,
#         target_modules=["q_proj", "v_proj"],
#     )

#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()
#     return model


def create_module(pretrained_model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    # lora_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    # )

    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    return model


def format_reward(completion: str) -> float:
    """Reward function that checks if the completion has a specific format.

    :param completion: Prompt completion to be evaluated.
    :type completion: str
    :return: Reward for the format of the completion.
    :rtype: float
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    pattern_match = re.match(pattern, completion)
    return 1.0 if pattern_match else -1.0


def accuracy_reward(completion: str, solution: str) -> float:
    """Reward function that checks if the completion is the same as the ground truth.

    :param completion: Prompt completion to be evaluated.
    :type completion: str
    :param solution: Ground truth solution.
    :type solution: str
    :return: Reward for the accuracy of the completion.
    :rtype: float
    """
    # Obtain numerical answer
    pattern = re.compile(r"#### (\-?[0-9\.\,]+)")
    correct_answer = pattern.search(solution)
    correct_answer = correct_answer.group(1).strip()

    # Obtain our models answer
    pattern = r"\d+\.\d+|\d+/\d+|\d+"
    nums = re.findall(pattern, completion)
    if len(nums) == 0:
        return -1.0
    answer = nums[-1]
    return 3 if (answer == correct_answer) else -3


def reward_function(completion: str, solution: str) -> float:
    """Reward function that combines the format and accuracy rewards.

    :param completion: Prompt completion to be evaluated.
    :type completion: str
    :param solution: Ground truth solution.
    :type solution: str
    :return: Combined reward for the completion.
    :rtype: float
    """
    return accuracy_reward(completion, solution) + format_reward(completion)


def main():
    # Instantiate the model and the associated tokenizer
    print(torch.cuda.is_available())
    model = to_evolvable(
        module_fn=create_module,
        module_kwargs={"pretrained_model_name_or_path": MODEL_PATH},
        device="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Convert the HuggingFace dataset into a Gymnasium environment
    env = HuggingFaceGym(
        dataset_name=DATASET,
        tokenizer=tokenizer,
        reward_fn=reward_function,
        max_answer_tokens=200,
        data_batch_size=16,
    )
    # Instantiate the grpo agent
    agent = GRPO(
        env.observation_space,
        env.action_space,
        actor_network=model,
        pad_token_id=tokenizer.eos_token_id,
        device="cuda",
        batch_size=8,
        group_size=5
    )
    finetune_llm(
        agent=agent, env=env, INIT_HP={}, evaluation_interval=5, wb=True, checkpoint_interval=100, checkpoint_path="saved_llms"
    )  # Do we want to keep this the same?


if __name__ == "__main__":
    main()
