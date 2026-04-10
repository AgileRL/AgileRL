"""Conditional single-digit target probe (``ConditionalTargetEnv``)."""

from __future__ import annotations

import statistics
from random import Random

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    raise ImportError("LLM dependencies are not installed.")

import torch

from agilerl.algorithms import GRPO, LLMPPO, LLMReinforce
from agilerl.training import train_llm
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.llm_utils import create_llm_accelerator, masked_whiten
from agilerl.utils.probe_envs_llm import ConditionalTargetEnv
from agilerl.utils.utils import create_population
from agilerl.wrappers.gem_wrappers import TokenObservationWrapper

from config_load import load_debug_config
from llm_debug_utils import lora_config_from_dict
from tiny_model import TinyDigitTokenizer, build_tiny_actor_network

TARGET_TOKEN_IDS = (1, 2, 3)


def evaluate_accuracy(
    agent: LLMPPO | LLMReinforce | GRPO,
    tokenizer: TinyDigitTokenizer,
    num_episodes: int,
    greedy: bool = False,
) -> tuple[float, dict[int, float]]:
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


def run_single_seed(cfg: dict, seed: int) -> tuple[float, float]:
    dbg = cfg["DEBUG"]
    init_hp = dict(cfg["INIT_HP"])
    if dbg.get("test_policy_only"):
        init_hp["VF_COEF"] = float(dbg["policy_only_vf_coef"])
    eval_eps = int(dbg["eval_episodes"])
    max_ctx = int(dbg["max_context_length"])
    max_new = int(dbg["max_output_tokens"])
    agent_dbg = dbg.get("agent") or {}
    agent_kw: dict = {}
    if "calc_position_embeddings" in agent_dbg:
        agent_kw["calc_position_embeddings"] = bool(agent_dbg["calc_position_embeddings"])
    if agent_dbg.get("torch_compiler") is not None:
        agent_kw["torch_compiler"] = agent_dbg["torch_compiler"]

    accelerator = create_llm_accelerator()
    torch.manual_seed(seed)
    tokenizer = TinyDigitTokenizer()

    init_hp.setdefault("ALGO", "LLMPPO")
    init_hp.setdefault("USE_VLLM", False)
    init_hp.setdefault("MAX_MODEL_LEN", max_ctx)
    init_hp.setdefault("MAX_OUTPUT_TOKENS", max_new)
    init_hp.setdefault("SEED", seed)
    if "LR" not in init_hp and "LR_ACTOR" in init_hp:
        init_hp["LR"] = init_hp["LR_ACTOR"]

    pop = create_population(
        algo=str(init_hp["ALGO"]),
        net_config=None,
        INIT_HP=init_hp,
        population_size=1,
        accelerator=accelerator,
        tokenizer=tokenizer,
        model_name=None,
        actor_network=build_tiny_actor_network(use_value_head=(init_hp["ALGO"] == "LLMPPO")),
        lora_config=lora_config_from_dict(dbg["lora"]),
        algo_kwargs=agent_kw,
    )
    agent = pop[0]

    if dbg.get("test_policy_only"):
        if isinstance(agent, LLMPPO):
            enable_reinforce_style_advantages(agent)
            print(f"[seed={seed}] test_policy_only=True (REINFORCE-style advantages)")
        else:
            print(
                f"[seed={seed}] test_policy_only ignored for algo={init_hp['ALGO']}"
            )
    else:
        print(f"[seed={seed}] test_policy_only=False (GAE)")

    pre_acc, pre_class = evaluate_accuracy(
        agent, tokenizer, num_episodes=eval_eps, greedy=False
    )
    pre_acc_g, pre_class_g = evaluate_accuracy(
        agent, tokenizer, num_episodes=eval_eps, greedy=True
    )
    print(
        f"[seed={seed}] pre-train acc (sampled/greedy): {pre_acc:.3f}/{pre_acc_g:.3f}"
    )
    print(f"[seed={seed}] pre per-class sampled: {pre_class}")
    print(f"[seed={seed}] pre per-class greedy: {pre_class_g}")

    env = TokenObservationWrapper(
        ConditionalTargetEnv(seed=seed),
        tokenizer,
        1,
        tokenizer.pad_token_id,
        apply_chat_template=False,
        max_model_len=max_ctx,
        max_output_tokens=max_new,
    )
    def env_factory() -> TokenObservationWrapper:
        return TokenObservationWrapper(
            ConditionalTargetEnv(seed=seed),
            tokenizer,
            1,
            tokenizer.pad_token_id,
            apply_chat_template=False,
            max_model_len=max_ctx,
            max_output_tokens=max_new,
        )
    eval_fn = lambda a: evaluate_accuracy(a, tokenizer, eval_eps, greedy=True)[0]

    original_save = train_llm.save_llm_checkpoint
    train_llm.save_llm_checkpoint = lambda *args, **kwargs: None
    try:
        finetune_llm_multiturn(
            pop=[agent],
            env=env,
            max_turns=1,
            init_hp=init_hp,
            max_steps=int(dbg["max_sample_steps"]),
            eval_fn=eval_fn,
            evaluation_interval=int(dbg["evaluation_interval"]),
            wb=False,
            save_elite=False,
            verbose=True,
            accelerator=accelerator,
            env_factory=env_factory,
        )
    finally:
        train_llm.save_llm_checkpoint = original_save

    post_acc, post_class = evaluate_accuracy(
        agent, tokenizer, num_episodes=eval_eps, greedy=False
    )
    post_acc_g, post_class_g = evaluate_accuracy(
        agent, tokenizer, num_episodes=eval_eps, greedy=True
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


def main(cfg: dict) -> None:
    seeds = tuple(int(s) for s in cfg["DEBUG"]["seeds"])
    sampled_improvements = []
    greedy_improvements = []
    for seed in seeds:
        sampled_imp, greedy_imp = run_single_seed(cfg, seed)
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
    main(load_debug_config("ppo_conditional_target.yaml"))
