"""Stage-1 debugging: single-turn PPO with input-dependent target.

The agent receives a single-digit prompt from {1, 2, 3} and must learn
the mapping digit -> (digit % 3) + 1. Uses ConditionalTargetEnv from
agilerl.utils.probe_envs_llm (max_turns=1).
"""

from __future__ import annotations

import statistics
from random import Random

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed."
    raise ImportError(msg)

import torch
import yaml
from peft import LoraConfig

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.training import train_llm
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.llm_utils import create_llm_accelerator, masked_whiten
from agilerl.utils.probe_envs_llm import ConditionalTargetEnv
from benchmarking.tiny_model import TinyDigitTokenizer, build_tiny_actor_network

MAX_CONTEXT_LENGTH = 128
MAX_OUTPUT_TOKENS = 1
EVAL_EPISODES = 128
TARGET_TOKEN_IDS = (1, 2, 3)
TEST_POLICY_ONLY = False


def evaluate_accuracy(
    agent: LLMPPO,
    tokenizer: TinyDigitTokenizer,
    num_episodes: int = EVAL_EPISODES,
    greedy: bool = False,
) -> tuple[float, dict[int, float]]:
    """Returns (overall_accuracy, per_class_accuracy_dict)."""
    original_temp = agent.generation_config.temperature
    original_top_k = agent.generation_config.top_k
    original_top_p = agent.generation_config.top_p
    if greedy:
        agent.generation_config.temperature = 1e-3
        agent.generation_config.top_k = 1
        agent.generation_config.top_p = 1.0

    total = 0
    correct = 0
    class_total = {t: 0 for t in TARGET_TOKEN_IDS}
    class_correct = {t: 0 for t in TARGET_TOKEN_IDS}
    eval_rng = Random(12345)

    try:
        with torch.no_grad():
            for _ in range(num_episodes):
                env = ConditionalTargetEnv(seed=eval_rng.randint(0, 2**31))
                obs, _info = env.reset()
                input_digit = int(obs)

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

                _next_obs, reward, _terminated, _truncated, _step_info = env.step(
                    gen_text
                )

                total += 1
                class_total[input_digit] += 1
                if reward > 0:
                    correct += 1
                    class_correct[input_digit] += 1
    finally:
        agent.generation_config.temperature = original_temp
        agent.generation_config.top_k = original_top_k
        agent.generation_config.top_p = original_top_p

    per_class = {t: class_correct[t] / max(class_total[t], 1) for t in TARGET_TOKEN_IDS}
    return correct / max(total, 1), per_class


def enable_reinforce_style_advantages(agent: LLMPPO) -> None:
    def _reinforce_like_returns(
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del values
        returns = rewards
        advantages = masked_whiten(rewards, action_mask)
        return returns, advantages * action_mask

    agent._compute_gae_returns = _reinforce_like_returns  # type: ignore[method-assign]


def run_single_seed(init_hp: dict, seed: int) -> tuple[float, float]:
    accelerator = create_llm_accelerator()
    torch.manual_seed(seed)
    actor_network = build_tiny_actor_network()
    tokenizer = TinyDigitTokenizer()
    rng = Random(seed)

    llm_ppo = LLMPPO(
        model_name=None,
        actor_network=actor_network,
        lora_config=LoraConfig(
            r=8,
            lora_alpha=32,
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
        calc_position_embeddings=False,
        seed=seed,
        gradient_checkpointing=True,
        torch_compiler="default",
    )
    if TEST_POLICY_ONLY:
        enable_reinforce_style_advantages(llm_ppo)
        print(f"[seed={seed}] mode: test_policy_only=True (REINFORCE-style advantages)")
    else:
        print(f"[seed={seed}] mode: test_policy_only=False (PPO GAE advantages)")

    pre_acc, pre_class = evaluate_accuracy(
        llm_ppo, tokenizer, num_episodes=EVAL_EPISODES, greedy=False
    )
    pre_acc_g, pre_class_g = evaluate_accuracy(
        llm_ppo, tokenizer, num_episodes=EVAL_EPISODES, greedy=True
    )
    print(
        f"[seed={seed}] pre-train acc (sampled/greedy): {pre_acc:.3f}/{pre_acc_g:.3f}"
    )
    print(f"[seed={seed}] pre per-class sampled: {pre_class}")
    print(f"[seed={seed}] pre per-class greedy: {pre_class_g}")

    batch_size = init_hp["BATCH_SIZE"]
    max_train_steps = (4096 + batch_size - 1) // batch_size
    env_fn = lambda: ConditionalTargetEnv(seed=rng.randint(0, 2**31))
    eval_fn = lambda agent: evaluate_accuracy(
        agent, tokenizer, num_episodes=EVAL_EPISODES, greedy=True
    )[0]

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
            evaluation_interval=50,
            wb=False,
            save_elite=False,
            verbose=True,
            accelerator=accelerator,
        )
    finally:
        train_llm.save_llm_checkpoint = original_save_checkpoint

    post_acc, post_class = evaluate_accuracy(
        llm_ppo, tokenizer, num_episodes=EVAL_EPISODES, greedy=False
    )
    post_acc_g, post_class_g = evaluate_accuracy(
        llm_ppo, tokenizer, num_episodes=EVAL_EPISODES, greedy=True
    )
    print(
        f"[seed={seed}] post-train acc (sampled/greedy): "
        f"{post_acc:.3f}/{post_acc_g:.3f}"
    )
    print(f"[seed={seed}] post per-class sampled: {post_class}")
    print(f"[seed={seed}] post per-class greedy: {post_class_g}")
    print(
        f"[seed={seed}] improvement (sampled/greedy): "
        f"{post_acc - pre_acc:+.3f}/{post_acc_g - pre_acc_g:+.3f}"
    )
    return post_acc - pre_acc, post_acc_g - pre_acc_g


def main(init_hp: dict, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    sampled_improvements = []
    greedy_improvements = []
    for seed in seeds:
        sampled_imp, greedy_imp = run_single_seed(dict(init_hp), seed)
        sampled_improvements.append(sampled_imp)
        greedy_improvements.append(greedy_imp)

    sampled_mean = statistics.mean(sampled_improvements)
    sampled_std = (
        statistics.stdev(sampled_improvements) if len(sampled_improvements) > 1 else 0.0
    )
    greedy_mean = statistics.mean(greedy_improvements)
    greedy_std = (
        statistics.stdev(greedy_improvements) if len(greedy_improvements) > 1 else 0.0
    )
    print(
        f"[summary] sampled improvement over {len(seeds)} seeds: "
        f"{sampled_mean:+.3f} +/- {sampled_std:.3f}"
    )
    print(
        f"[summary] greedy improvement over {len(seeds)} seeds: "
        f"{greedy_mean:+.3f} +/- {greedy_std:.3f}"
    )


if __name__ == "__main__":
    with open("configs/training/llm_finetuning/ppo_llm.yaml") as file:
        config = yaml.safe_load(file)

    init_hp = config["INIT_HP"]
    init_hp["BATCH_SIZE"] = 64
    init_hp["UPDATE_EPOCHS"] = 2
    init_hp["LR_ACTOR"] = 1e-4
    init_hp["LR_CRITIC"] = 1e-4
    init_hp["BETA"] = 0.01
    init_hp["TEMPERATURE"] = 0.4
    init_hp["VF_COEF"] = 0.0 if TEST_POLICY_ONLY else 0.5
    init_hp["GAMMA"] = 1.0
    init_hp["GAE_LAMBDA"] = 1.0
    main(init_hp, (0,))
