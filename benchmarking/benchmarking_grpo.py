from transformers import AutoModelForCausalLM, AutoTokenizer

from agilerl.algorithms import GRPO
from agilerl.modules.dummy import to_evolvable
from agilerl.training.train_llm import finetune_llm
from agilerl.utils.llm_utils import HuggingFaceGym

MODEL_PATH = "Qwen/Qwen2-0.5B"
DATASET = "openai/gsm8k"


def main():
    # Instantiate the model and the associated tokenizer
    model = to_evolvable(
        module_fn=AutoModelForCausalLM.from_pretrained,
        module_kwargs={"pretrained_model_name_or_path": MODEL_PATH},
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Convert the HuggingFace dataset into a Gymnasium environment
    env = HuggingFaceGym(dataset_name=DATASET, tokenizer=tokenizer)

    # Instantiate the grpo agent
    algo = GRPO(
        env.observation_space,
        env.action_space,
        actor_network=model,
        pad_token_id=tokenizer.pad_token_id,
    )
    finetune_llm(agent=algo, env=env, INIT_HP={})  # Do we want to keep this the same?


if __name__ == "__main__":
    main()
