"""LLM fine-tuning demo -- SFT and DPO with full CLI support.

Train, warm-start, or interactively evaluate LoRA-adapted language models.
Hyperparameters are loaded from YAML configs in ``configs/training/``.

Examples
--------
Train SFT::

    python demos/demo_llm_finetuning.py sft

Train DPO from the base model::

    python demos/demo_llm_finetuning.py dpo

Warm-start DPO from a prior SFT checkpoint::

    python demos/demo_llm_finetuning.py dpo --load-path outputs/sft/actor

Evaluate a saved checkpoint interactively::

    python demos/demo_llm_finetuning.py sft --eval --load-path outputs/sft/actor

Multi-GPU / DeepSpeed via accelerate::

    accelerate launch demos/demo_llm_finetuning.py sft
"""

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = (
        "LLM dependencies are not installed. "
        "Install them with `pip install agilerl[llm]`."
    )
    raise ImportError(msg)

import argparse
import json
from datetime import datetime

import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agilerl.algorithms.dpo import DPO
from agilerl.algorithms.sft import SFT
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_llm import finetune_llm_preference, finetune_llm_sft
from agilerl.utils.llm_utils import (
    PreferenceGym,
    SFTGym,
    compare_responses,
    sample_eval_prompts,
)

MODEL_PATH = "Qwen/Qwen2.5-0.5B"
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


def main(
    mode: str,
    init_hp: dict,
    mut_p: dict,
    save_path: str = "outputs",
    load_path: str | None = None,
) -> None:
    """Run an SFT or DPO fine-tuning loop.

    :param mode: ``"sft"`` or ``"dpo"``.
    :type mode: str
    :param init_hp: Initial hyperparameter dict from the YAML config.
    :type init_hp: dict
    :param mut_p: Mutation parameter dict from the YAML config.
    :type mut_p: dict
    :param save_path: Directory to save the elite LoRA checkpoint.
    :type save_path: str
    :param load_path: Optional path to a pre-trained LoRA checkpoint to warm-start from
        (e.g. an SFT adapter when running DPO).
    :type load_path: str | None
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)

    try:
        accelerator = Accelerator()
        if accelerator.state.deepspeed_plugin is None:
            accelerator = None
    except Exception:
        accelerator = None

    # --- Environment -------------------------------------------------------
    env_cls = SFTGym if mode == "sft" else PreferenceGym
    env_kwargs: dict = dict(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=init_hp["BATCH_SIZE"],
        accelerator=accelerator,
        max_context_length=init_hp.get("MAX_CONTEXT_LENGTH"),
    )
    if mode == "sft":
        env_kwargs["response_column"] = "chosen"

    print(f"Setting up {env_cls.__name__} environment...")
    env = env_cls(**env_kwargs)

    init_hp["PAD_TOKEN_ID"] = tokenizer.eos_token_id
    init_hp["PAD_TOKEN"] = tokenizer.eos_token

    lora_config = LoraConfig(**init_hp["LORA"])

    # --- Optional warm-start from a saved LoRA adapter ---------------------
    actor_network = None
    if load_path is not None:
        print(f"Loading pre-trained LoRA weights from {load_path} ...")
        with open(f"{load_path}/adapter_config.json") as f:
            base_model_name = json.load(f)["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        actor_network = PeftModel.from_pretrained(
            base_model, load_path, adapter_name="actor"
        )

    # --- Agent population --------------------------------------------------
    agent_cls = SFT if mode == "sft" else DPO
    agent_kwargs: dict = dict(
        model_name=MODEL_PATH if actor_network is None else None,
        actor_network=actor_network,
        pad_token_id=init_hp["PAD_TOKEN_ID"],
        pad_token=init_hp["PAD_TOKEN"],
        batch_size=init_hp["BATCH_SIZE"],
        lr=init_hp["LR"],
        update_epochs=init_hp["UPDATE_EPOCHS"],
        lora_config=lora_config if actor_network is None else None,
        accelerator=accelerator,
        gradient_checkpointing=accelerator is not None,
        use_liger_loss=init_hp.get("USE_LIGER_LOSS", False),
    )
    if mode == "dpo":
        agent_kwargs["beta"] = init_hp["BETA"]
        agent_kwargs["nll_alpha"] = init_hp.get("NLL_ALPHA", 1.0)

    print(f"Defining {mode.upper()} agent population...")
    pop = [agent_cls(**agent_kwargs) for _ in range(init_hp["POP_SIZE"])]

    # --- HPO (only when evolution is enabled) ------------------------------
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

    num_batches = init_hp.get("NUM_BATCHES")
    max_steps = num_batches * init_hp["BATCH_SIZE"] if num_batches is not None else None

    # --- Training ----------------------------------------------------------
    train_fn = finetune_llm_sft if mode == "sft" else finetune_llm_preference
    train_kwargs: dict = dict(
        pop=pop,
        env=env,
        init_hp=init_hp,
        save_elite=True,
        elite_path=save_path,
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
    )
    if mode == "sft":
        train_kwargs["plot_path"] = init_hp.get("PLOT_PATH")

    print(f"Fine-tuning {mode.upper()} agents...")
    train_fn(**train_kwargs)

    # --- Post-training qualitative eval ------------------------------------
    print("\nQualitative response comparison (elite agent):")
    elite = max(pop, key=lambda a: a.fitness[-1] if a.fitness else float("-inf"))
    eval_samples = sample_eval_prompts(env, n=init_hp.get("EVAL_N_SAMPLES", 5))
    compare_responses(elite, tokenizer, eval_samples)


def eval_mode(mode: str, load_path: str, max_new_tokens: int = 200) -> None:
    """Load a saved LoRA checkpoint and enter an interactive prompt loop.

    :param mode: ``"sft"`` or ``"dpo"`` — selects the agent class.
    :type mode: str
    :param load_path: Path to a directory containing ``adapter_config.json``
        and the LoRA adapter weights.
    :type load_path: str
    :param max_new_tokens: Maximum tokens to generate per response.
    :type max_new_tokens: int
    """
    with open(f"{load_path}/adapter_config.json") as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    print(f"Loading tokenizer from {base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {base_model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    print(f"Applying LoRA adapter from {load_path} ...")
    actor_network = PeftModel.from_pretrained(base_model, load_path)

    agent_cls = SFT if mode == "sft" else DPO
    agent = agent_cls(
        actor_network=actor_network,
        pad_token_id=tokenizer.eos_token_id,
        pad_token=tokenizer.eos_token,
    )

    print(f"\nEval mode ready  |  base: {base_model_name}  |  adapter: {load_path}")
    print("Enter a prompt and press Enter to generate a response.")
    print("Type 'quit', 'q', 'exit', or press Ctrl+C to quit.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting eval mode.")
            break

        if not prompt or prompt.lower() in {"quit", "q", "exit"}:
            print("Exiting eval mode.")
            break

        compare_responses(
            agent,
            tokenizer,
            [(prompt, None, None)],
            max_new_tokens=max_new_tokens,
            show_base_model=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM fine-tuning demo (SFT / DPO)")
    parser.add_argument("mode", choices=["sft", "dpo"], help="Fine-tuning algorithm")
    parser.add_argument(
        "--save-path",
        default="outputs",
        help="Base directory to save the elite LoRA checkpoint (default: outputs)",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable the timestamp sub-directory, overwriting any existing checkpoint",
    )
    parser.add_argument(
        "--load-path",
        default=None,
        help=(
            "Path to a LoRA checkpoint. In training mode, warm-starts from these weights "
            "(e.g. an SFT adapter for DPO). In eval mode (--eval), loads for interactive "
            "inference. Must contain adapter_model.safetensors / adapter_config.json."
        ),
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enter interactive eval mode instead of training (requires --load-path)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per response in eval mode (default: 200)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: configs/training/{mode}.yaml)",
    )
    args = parser.parse_args()

    if args.eval:
        if not args.load_path:
            parser.error("--load-path is required when --eval is set")
        eval_mode(
            mode=args.mode,
            load_path=args.load_path,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        config_path = args.config or f"configs/training/{args.mode}.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        save_path = (
            args.save_path
            if args.no_timestamp
            else f"{args.save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.mode.upper()}"
        )

        main(
            mode=args.mode,
            init_hp=config["INIT_HP"],
            mut_p=config["MUTATION_PARAMS"],
            save_path=save_path,
            load_path=args.load_path,
        )
