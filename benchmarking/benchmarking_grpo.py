import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from agilerl.algorithms import GRPO
from agilerl.modules.dummy import to_evolvable
from agilerl.training.train_llm import finetune_llm
from agilerl.utils.llm_utils import HuggingFaceGym, reward_function

MODEL_PATH = "Qwen/Qwen2-0.5B"
DATASET = "openai/gsm8k"


def create_module(pretrained_model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    # Instantiate the model and the associated tokenizer
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
        batch_size=4,
    )

    # Instantiate the grpo agent
    algo = GRPO(
        env.observation_space,
        env.action_space,
        actor_network=model,
        pad_token_id=tokenizer.pad_token_id,
        device="cuda",
    )
    finetune_llm(agent=algo, env=env, INIT_HP={})  # Do we want to keep this the same?


if __name__ == "__main__":
    main()
