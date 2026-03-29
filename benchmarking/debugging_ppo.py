"""Stage-0 debugging: single-turn PPO with a constant target token.

Simplest possible probe -- the agent must learn to always emit a fixed
target digit ("3") from a fixed prompt ("11"), regardless of context.
Uses ConstantTargetEnv from agilerl.utils.probe_envs_llm (max_turns=1).
"""

from __future__ import annotations

import statistics

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed."
    raise ImportError(msg)

import torch
import yaml
from peft import LoraConfig

from agilerl.algorithms import LLMPPO
from agilerl.training import train_llm
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.utils.probe_envs_llm import ConstantTargetEnv
from benchmarking.tiny_model import build_tiny_actor_network, TinyDigitTokenizer

MAX_CONTEXT_LENGTH = 128
MAX_OUTPUT_TOKENS = 1
TARGET_TOKEN_ID = 3
TARGET_TOKEN = str(TARGET_TOKEN_ID)
EVAL_EPISODES = 128


def evaluate_hit_rate(
    agent: LLMPPO,
    tokenizer: TinyDigitTokenizer,
    num_episodes: int = EVAL_EPISODES,
    greedy: bool = False,
) -> float:
    original_temp = agent.generation_config.temperature
    original_top_k = agent.generation_config.top_k
    original_top_p = agent.generation_config.top_p
    if greedy:
        agent.generation_config.temperature = 1e-3
        agent.generation_config.top_k = 1
        agent.generation_config.top_p = 1.0

    hits = 0
    try:
        with torch.no_grad():
            for _ep in range(num_episodes):
                env = ConstantTargetEnv(target_digit=TARGET_TOKEN)
                obs, _info = env.reset()
                prompt_encoded = tokenizer(
                    [obs],
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    return_attention_mask=True,
                )
                prompt_dict = {
                    "input_ids": prompt_encoded["input_ids"],
                    "attention_mask": prompt_encoded["attention_mask"],
                }
                prompt_len = prompt_dict["input_ids"].shape[1]
                completion_ids, _ = agent.get_action([prompt_dict], training=False)
                full_ids = completion_ids[0]
                gen_tokens = full_ids[0, prompt_len:]
                gen_text = tokenizer.decode(
                    gen_tokens.tolist(), skip_special_tokens=True
                )
                if gen_text == TARGET_TOKEN:
                    hits += 1
    finally:
        agent.generation_config.temperature = original_temp
        agent.generation_config.top_k = original_top_k
        agent.generation_config.top_p = original_top_p

    return hits / max(num_episodes, 1)


def run_single_seed(init_hp: dict, seed: int) -> tuple[float, float]:
    accelerator = create_llm_accelerator()
    torch.manual_seed(seed)
    actor_network = build_tiny_actor_network()
    tokenizer = TinyDigitTokenizer()
    if TARGET_TOKEN_ID in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        msg = (
            f"TARGET_TOKEN_ID={TARGET_TOKEN_ID} must not be pad/eos token "
            f"(pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id})."
        )
        raise ValueError(msg)

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

    pre_rate = evaluate_hit_rate(llm_ppo, tokenizer, greedy=False)
    pre_rate_greedy = evaluate_hit_rate(llm_ppo, tokenizer, greedy=True)
    print(
        f"[seed={seed}] pre-train token-{TARGET_TOKEN} hit rate "
        f"(sampled/greedy): {pre_rate:.3f}/{pre_rate_greedy:.3f}"
    )

    batch_size = init_hp["BATCH_SIZE"]
    max_train_steps = (4096 + batch_size - 1) // batch_size
    env_fn = lambda: ConstantTargetEnv(target_digit=TARGET_TOKEN)
    eval_fn = lambda agent: evaluate_hit_rate(agent, tokenizer, greedy=True)

    original_save_checkpoint = train_llm.save_llm_checkpoint
    train_llm.save_llm_checkpoint = lambda *args, **kwargs: None
    try:
        finetune_llm_multiturn(
            pop=[llm_ppo],
            env_fn=env_fn,
            tokenizer=tokenizer,
            max_turns=1,
            init_hp={"ALGO": "LLMPPO", **init_hp},
            max_steps=max_train_steps,
            eval_fn=eval_fn,
            evaluation_interval=20,
            wb=False,
            save_elite=False,
            verbose=True,
            accelerator=accelerator,
        )
    finally:
        train_llm.save_llm_checkpoint = original_save_checkpoint

    post_rate = evaluate_hit_rate(llm_ppo, tokenizer, greedy=False)
    post_rate_greedy = evaluate_hit_rate(llm_ppo, tokenizer, greedy=True)
    print(
        f"[seed={seed}] post-train token-{TARGET_TOKEN} hit rate "
        f"(sampled/greedy): {post_rate:.3f}/{post_rate_greedy:.3f}"
    )
    print(
        f"[seed={seed}] improvement (sampled/greedy): "
        f"{post_rate - pre_rate:+.3f}/{post_rate_greedy - pre_rate_greedy:+.3f}"
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
