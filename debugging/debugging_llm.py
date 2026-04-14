"""Constant-target single-turn PPO probe (``ConstantTargetEnv``)."""

from __future__ import annotations

import statistics

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    raise ImportError("LLM dependencies are not installed.")

import torch
from config_load import load_debug_config
from llm_debug_utils import lora_config_from_dict
from tiny_model import TinyDigitTokenizer, build_tiny_actor_network

from agilerl.algorithms import GRPO, LLMPPO, LLMReinforce
from agilerl.training import train_llm
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.utils.probe_envs_llm import ConstantTargetEnv
from agilerl.utils.utils import create_population
from agilerl.wrappers.gem_wrappers import TokenObservationWrapper


def evaluate_hit_rate(
    agent: LLMPPO | LLMReinforce | GRPO,
    tokenizer: TinyDigitTokenizer,
    target_token: str,
    num_episodes: int,
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
            for _ in range(num_episodes):
                env = ConstantTargetEnv(target_digit=target_token)
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
                if gen_text == target_token:
                    hits += 1
    finally:
        agent.generation_config.temperature = original_temp
        agent.generation_config.top_k = original_top_k
        agent.generation_config.top_p = original_top_p

    return hits / max(num_episodes, 1)


def run_single_seed(cfg: dict, seed: int) -> tuple[float, float]:
    dbg = cfg["DEBUG"]
    init_hp = dict(cfg["INIT_HP"])
    target_id = int(dbg["target_token_id"])
    target_token = str(target_id)
    eval_eps = int(dbg["eval_episodes"])
    max_ctx = int(dbg["max_context_length"])
    max_new = int(dbg["max_output_tokens"])

    accelerator = create_llm_accelerator()
    torch.manual_seed(seed)
    tokenizer = TinyDigitTokenizer()
    if target_id in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        raise ValueError(
            f"target_token_id={target_id} must not be pad/eos "
            f"(pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id})."
        )

    actor_network = build_tiny_actor_network(
        use_value_head=(init_hp["ALGO"] == "LLMPPO")
    )
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
        actor_network=actor_network,
        lora_config=lora_config_from_dict(dbg["lora"]),
    )
    agent = pop[0]

    pre_rate = evaluate_hit_rate(agent, tokenizer, target_token, eval_eps, greedy=False)
    pre_g = evaluate_hit_rate(agent, tokenizer, target_token, eval_eps, greedy=True)
    print(
        f"[seed={seed}] pre-train token-{target_token} hit rate "
        f"(sampled/greedy): {pre_rate:.3f}/{pre_g:.3f}"
    )

    def env_factory() -> TokenObservationWrapper:
        return TokenObservationWrapper(
            ConstantTargetEnv(target_digit=target_token),
            tokenizer,
            1,
            tokenizer.pad_token_id,
            apply_chat_template=False,
            max_model_len=max_ctx,
            max_output_tokens=max_new,
        )

    eval_fn = lambda a: evaluate_hit_rate(a, tokenizer, target_token, eval_eps, True)

    original_save = train_llm.save_llm_checkpoint
    train_llm.save_llm_checkpoint = lambda *args, **kwargs: None
    try:
        finetune_llm_multiturn(
            pop=[agent],
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

    post_rate = evaluate_hit_rate(
        agent, tokenizer, target_token, eval_eps, greedy=False
    )
    post_g = evaluate_hit_rate(agent, tokenizer, target_token, eval_eps, greedy=True)
    print(
        f"[seed={seed}] post-train token-{target_token} hit rate "
        f"(sampled/greedy): {post_rate:.3f}/{post_g:.3f}"
    )
    print(
        f"[seed={seed}] improvement (sampled/greedy): "
        f"{post_rate - pre_rate:+.3f}/{post_g - pre_g:+.3f}"
    )
    return pre_rate, post_rate


def main(cfg: dict) -> None:
    seeds = tuple(int(s) for s in cfg["DEBUG"]["seeds"])
    improvements = []
    for seed in seeds:
        pre_rate, post_rate = run_single_seed(cfg, seed)
        improvements.append(post_rate - pre_rate)
    mean_imp = statistics.mean(improvements)
    std_imp = statistics.stdev(improvements) if len(improvements) > 1 else 0.0
    print(
        f"[summary] mean improvement over {len(seeds)} seeds: "
        f"{mean_imp:+.3f} +/- {std_imp:.3f}"
    )


if __name__ == "__main__":
    main(load_debug_config("grpo_constant_target.yaml"))
