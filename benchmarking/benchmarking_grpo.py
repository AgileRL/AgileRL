from transformers import AutoModelForCausalLM, AutoTokenizer

from agilerl.algorithms import GRPO
from agilerl.modules.dummy import to_evolvable
from agilerl.utils.llm_utils import HuggingFaceGym

MODEL_PATH = "Qwen/Qwen2-0.5B"
DATASET = "openai/gsm8k"

model = to_evolvable(
    module_fn=AutoModelForCausalLM.from_pretrained,
    module_kwargs={"pretrained_model_name_or_path": MODEL_PATH},
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

env = HuggingFaceGym(dataset_name=DATASET, tokenizer=tokenizer)

algo = GRPO()
