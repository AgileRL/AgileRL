from __future__ import annotations

import statistics
from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import yaml
from datasets import Dataset
from peft import LoraConfig
import torch
from accelerate import Accelerator
from agilerl.algorithms import LLMPPO
from agilerl.training import train_llm
from agilerl.training.train_llm import finetune_llm_reasoning
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import ReasoningGym
from benchmarking.tiny_model import (
    build_tiny_actor_network,
    TinyDigitTokenizer,
)

MAX_CONTEXT_LENGTH = 128
MAX_OUTPUT_TOKENS = 1
# Must be non-pad and non-eos for a meaningful smoke test.
TARGET_TOKEN_ID = 3
TARGET_TOKEN = str(TARGET_TOKEN_ID)


def make_dataset(
    train_size: int = 4096,
    test_size: int = 512,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    del seed

    def build_split(size: int) -> Dataset:
        # Keep prompts fixed and avoid the target token in context to create
        # a near-bandit objective: policy should learn to emit token 3.
        questions = ["11"] * size
        answers = [TARGET_TOKEN] * size
        return Dataset.from_dict({"question": questions, "answer": answers})

    return build_split(train_size), build_split(test_size)


def token_target_reward(completion: str, _answer: str, _question: str) -> float:
    # With max_new_tokens=1, reward only exact target-token generation.
    return 1.0 if completion == TARGET_TOKEN else -1.0


def evaluate_target_token_rate(
    agent: LLMPPO, env: ReasoningGym, batches: int = 4
) -> tuple[float, float]:
    total = 0
    hits = 0
    all_sq_errors = []
    with env.eval_mode(), torch.no_grad():
        prompts = env.reset(reset_dataloaders=True)
        for _ in range(batches):
            completion_ids, action_masks = agent.get_action(prompts, training=False)
            seq_rewards = []
            for prompt, group_completion in zip(prompts, completion_ids, strict=False):
                prompt_len = prompt["input_ids"].shape[1]
                first_generated = group_completion[:, prompt_len]
                total += first_generated.shape[0]
                hit = int((first_generated == TARGET_TOKEN_ID).sum().item())
                hits += hit
                reward = 1.0 if first_generated.item() == TARGET_TOKEN_ID else -1.0
                seq_rewards.append(reward)

            padded_ids, padded_masks = stack_and_pad_experiences(
                completion_ids,
                action_masks,
                padding_values=[agent.pad_token_id, False],
            )
            padded_ids = padded_ids.to(agent.device)
            padded_masks = padded_masks.to(agent.device)
            values = agent._get_values(
                padded_ids,
                batch_size=padded_ids.shape[0],
                eval_mode=True,
            )
            last_action_idx = padded_masks.long().cumsum(dim=-1).argmax(dim=-1)
            last_values = values.gather(1, last_action_idx.unsqueeze(1)).squeeze(1)
            rewards_t = torch.tensor(seq_rewards, device=last_values.device)
            all_sq_errors.append((last_values - rewards_t).pow(2).mean().item())

            prompts, _ = env.step(completion_ids)

    critic_mse = sum(all_sq_errors) / max(len(all_sq_errors), 1)
    return hits / max(total, 1), critic_mse


def evaluate_target_token_rate_greedy_like(
    agent: LLMPPO,
    env: ReasoningGym,
    batches: int = 4,
) -> tuple[float, float]:
    original_temperature = agent.generation_config.temperature
    original_top_k = agent.generation_config.top_k
    original_top_p = agent.generation_config.top_p
    agent.generation_config.temperature = 1e-3
    agent.generation_config.top_k = 1
    agent.generation_config.top_p = 1.0
    try:
        return evaluate_target_token_rate(agent, env, batches=batches)
    finally:
        agent.generation_config.temperature = original_temperature
        agent.generation_config.top_k = original_top_k
        agent.generation_config.top_p = original_top_p


def run_single_seed(init_hp: dict, seed: int) -> tuple[float, float]:
    accelerator = None  # Accelerator()
    torch.manual_seed(seed)
    actor_network = build_tiny_actor_network()
    tokenizer = TinyDigitTokenizer()
    if TARGET_TOKEN_ID in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        msg = (
            f"TARGET_TOKEN_ID={TARGET_TOKEN_ID} must not be pad/eos token "
            f"(pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id})."
        )
        raise ValueError(msg)
    train_dataset, test_dataset = make_dataset(seed=seed)

    conversation_template = [
        {"role": "system", "content": "Output one digit."},
        {"role": "user", "content": "{question}"},
        {"role": "assistant", "content": ""},
    ]

    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=token_target_reward,
        conversation_template=conversation_template,
        data_batch_size_per_gpu=init_hp["BATCH_SIZE"],
        accelerator=accelerator,
        max_context_length=MAX_CONTEXT_LENGTH,
        return_raw_completions=False,
        seed=seed,
    )

    llm_ppo = LLMPPO(
        model_name=None,
        actor_network=actor_network,
        lora_config=LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj", "c_fc"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
        micro_batch_size_per_gpu=min(8, init_hp["BATCH_SIZE"]),
        use_vllm=False,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        use_separate_reference_adapter=True,
        batch_size=init_hp["BATCH_SIZE"],
        beta=init_hp["BETA"],
        lr_actor=init_hp["LR_ACTOR"],
        lr_critic=init_hp["LR_CRITIC"],
        clip_coef=init_hp["CLIP_COEF"],
        max_grad_norm=init_hp["MAX_GRAD_NORM"],
        update_epochs=init_hp["UPDATE_EPOCHS"],
        temperature=init_hp["TEMPERATURE"],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_model_len=MAX_CONTEXT_LENGTH,
        accelerator=accelerator,
        vf_coef=init_hp["VF_COEF"],
        gamma=init_hp["GAMMA"],
        gae_lambda=init_hp["GAE_LAMBDA"],
        seed=seed,
        gradient_checkpointing=True,
    )

    pre_rate, pre_mse = evaluate_target_token_rate(llm_ppo, env, batches=4)
    pre_rate_greedy, pre_mse_g = evaluate_target_token_rate_greedy_like(
        llm_ppo, env, batches=4
    )
    print(
        f"[seed={seed}] pre-train token-{TARGET_TOKEN} hit rate "
        f"(sampled/greedy-like): {pre_rate:.3f}/{pre_rate_greedy:.3f}"
    )
    print(
        f"[seed={seed}] pre critic MSE (sampled/greedy-like): {pre_mse:.4f}/{pre_mse_g:.4f}"
    )

    original_save_checkpoint = train_llm.save_llm_checkpoint
    train_llm.save_llm_checkpoint = lambda *args, **kwargs: None
    try:
        finetune_llm_reasoning(
            pop=[llm_ppo],
            env=env,
            init_hp={"ALGO": "LLMPPO", **init_hp},
            evaluation_interval=20,
            wb=False,
            save_elite=False,
            elite_path="saved_llms",
            max_reward=1.0,
            evo_steps=None,
            mutation=None,
            tournament=None,
            accelerator=accelerator,
            checkpoint_steps=999999,
            verbose=True,
            max_steps=4096,
        )
    finally:
        train_llm.save_llm_checkpoint = original_save_checkpoint

    post_rate, post_mse = evaluate_target_token_rate(llm_ppo, env, batches=4)
    post_rate_greedy, post_mse_g = evaluate_target_token_rate_greedy_like(
        llm_ppo, env, batches=4
    )
    print(
        f"[seed={seed}] post-train token-{TARGET_TOKEN} hit rate "
        f"(sampled/greedy-like): {post_rate:.3f}/{post_rate_greedy:.3f}"
    )
    print(
        f"[seed={seed}] post critic MSE (sampled/greedy-like): {post_mse:.4f}/{post_mse_g:.4f}"
    )
    print(
        f"[seed={seed}] improvement (sampled/greedy-like): "
        f"{post_rate - pre_rate:+.3f}/{post_rate_greedy - pre_rate_greedy:+.3f}"
    )
    print(
        f"[seed={seed}] critic MSE change (sampled/greedy-like): "
        f"{post_mse - pre_mse:+.4f}/{post_mse_g - pre_mse_g:+.4f}"
    )
    return pre_rate, post_rate


def main(init_hp: dict, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    improvements = []
    for seed in seeds:
        pre_rate, post_rate = run_single_seed(dict(init_hp), seed)
        improvements.append(post_rate - pre_rate)
    mean_imp = statistics.mean(improvements)
    std_imp = statistics.stdev(improvements) if len(improvements) > 1 else 0.0
    print(
        f"[summary] mean improvement over {len(seeds)} seeds: "
        f"{mean_imp:+.3f} +/- {std_imp:.3f}"
    )


if __name__ == "__main__":
    with open("configs/training/llm_finetuning/ppo_llm.yaml") as file:
        config = yaml.safe_load(file)

    init_hp = config["INIT_HP"]
    # Explicit smoke-test settings to make policy improvement easier to observe.
    init_hp["BATCH_SIZE"] = 32
    init_hp["UPDATE_EPOCHS"] = 2
    init_hp["LR_ACTOR"] = 1e-3
    init_hp["LR_CRITIC"] = 1e-3
    init_hp["BETA"] = 0.0
    init_hp["TEMPERATURE"] = 0.5
    init_hp["VF_COEF"] = 0.5
    init_hp["GAMMA"] = 1.0
    init_hp["GAE_LAMBDA"] = 1.0
    main(init_hp, (0,))
