"""Multi-turn grid navigation probe (``GridNavigationEnv`` + ``TokenObservationWrapper``)."""

from __future__ import annotations

import statistics
from random import Random
from typing import Any

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    raise ImportError("LLM dependencies are not installed.")

import torch
from agilerl.wrappers.multiturn_wrappers import TokenObservationWrapper
from config_load import load_debug_config
from llm_debug_utils import lora_config_from_dict
from tiny_model import TinyDigitTokenizer, build_tiny_actor_network
from transformers import AutoTokenizer

from agilerl.algorithms import GRPO, LLMPPO, LLMReinforce
from agilerl.training import train_llm
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.utils.probe_envs_llm import GridNavigationEnv
from agilerl.utils.utils import create_population


def _prompt_dict_from_encoded(
    tokenizer: Any, prompt_encoded: dict[str, torch.Tensor]
) -> dict[str, Any]:
    del tokenizer
    input_ids = prompt_encoded["input_ids"]
    return {
        "input_ids": input_ids,
        "attention_mask": prompt_encoded["attention_mask"],
    }


def evaluate_accuracy(
    agent: LLMPPO | LLMReinforce | GRPO,
    tokenizer: Any,
    grid_size: int,
    max_turns: int,
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

    correct = 0
    total = 0
    eval_rng = Random(12345)

    try:
        with torch.no_grad():
            for _ in range(num_episodes):
                env = GridNavigationEnv(
                    grid_size=grid_size,
                    max_turns=max_turns,
                    seed=eval_rng.randint(0, 2**31),
                )
                obs, _info = env.reset()

                prompt_encoded = tokenizer(
                    [obs],
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    return_attention_mask=True,
                )

                for _turn_idx in range(max_turns):
                    prompt_dict = _prompt_dict_from_encoded(tokenizer, prompt_encoded)
                    prompt_len = prompt_dict["input_ids"].shape[1]

                    completion_ids, _ = agent.get_action([prompt_dict], training=False)
                    full_ids = completion_ids[0]
                    gen_tokens = full_ids[0, prompt_len:]
                    gen_text = tokenizer.decode(
                        gen_tokens.tolist(), skip_special_tokens=True
                    )

                    _next_obs, reward, terminated, truncated, _step_info = env.step(
                        gen_text
                    )

                    if terminated or truncated:
                        total += 1
                        if reward > 0:
                            correct += 1
                        break

                    feedback_ids = torch.tensor(
                        [tokenizer.encode(_next_obs)],
                        dtype=torch.long,
                        device=full_ids.device,
                    )
                    new_prompt_ids = torch.cat([full_ids, feedback_ids], dim=1)
                    prompt_encoded = {
                        "input_ids": new_prompt_ids,
                        "attention_mask": torch.ones_like(new_prompt_ids),
                    }
                else:
                    total += 1
    finally:
        agent.generation_config.temperature = original_temp
        agent.generation_config.top_k = original_top_k
        agent.generation_config.top_p = original_top_p

    return correct / max(total, 1)


def detailed_eval(
    agent: LLMPPO | LLMReinforce | GRPO,
    tokenizer: Any,
    grid_size: int,
    max_turns: int,
) -> float:

    orig_temp = agent.generation_config.temperature
    orig_top_k = agent.generation_config.top_k
    orig_top_p = agent.generation_config.top_p
    agent.generation_config.temperature = 1e-3
    agent.generation_config.top_k = 1
    agent.generation_config.top_p = 1.0

    action_names = {"1": "L", "2": "S", "3": "R"}
    results: dict[tuple[int, int], bool] = {}

    try:
        with torch.no_grad():
            for start in range(grid_size):
                for target in range(grid_size):
                    if start == target:
                        continue
                    env = GridNavigationEnv(
                        grid_size=grid_size,
                        max_turns=max_turns,
                        seed=0,
                    )
                    env.position = start
                    env.target = target
                    env.turn = 0

                    obs = f"{start}{target}"
                    prompt_encoded = tokenizer(
                        [obs],
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        return_attention_mask=True,
                    )

                    actions: list[str] = []
                    success = False
                    for _ in range(max_turns):
                        prompt_dict = _prompt_dict_from_encoded(
                            tokenizer, prompt_encoded
                        )
                        prompt_len = prompt_dict["input_ids"].shape[1]
                        completion_ids, _ = agent.get_action(
                            [prompt_dict],
                            training=False,
                        )
                        full_ids = completion_ids[0]
                        gen_tokens = full_ids[0, prompt_len:]
                        gen_text = tokenizer.decode(
                            gen_tokens.tolist(),
                            skip_special_tokens=True,
                        )
                        raw_tok = gen_tokens[0].item() if len(gen_tokens) > 0 else -1
                        actions.append(action_names.get(gen_text, f"?{raw_tok}"))

                        next_obs, reward, terminated, truncated, _ = env.step(
                            gen_text,
                        )
                        if terminated or truncated:
                            success = reward > 0
                            break

                        feedback_ids = torch.tensor(
                            [tokenizer.encode(next_obs)],
                            dtype=torch.long,
                            device=full_ids.device,
                        )
                        new_prompt_ids = torch.cat(
                            [full_ids, feedback_ids],
                            dim=1,
                        )
                        prompt_encoded = {
                            "input_ids": new_prompt_ids,
                            "attention_mask": torch.ones_like(new_prompt_ids),
                        }

                    tag = "OK" if success else "FAIL"
                    results[(start, target)] = success
                    direction = "R" if target > start else "L"
                    dist = abs(target - start)
                    print(
                        f"  {start}->{target} (d={dist},{direction})  "
                        f"actions=[{','.join(actions)}]  {tag}"
                    )
    finally:
        agent.generation_config.temperature = orig_temp
        agent.generation_config.top_k = orig_top_k
        agent.generation_config.top_p = orig_top_p

    n_ok = sum(results.values())
    n_total = len(results)
    print(f"  {n_ok}/{n_total} pairs solved")
    for d in range(1, grid_size):
        pairs = [(s, t) for (s, t) in results if abs(s - t) == d]
        if pairs:
            ok = sum(results[p] for p in pairs)
            print(f"  dist={d}: {ok}/{len(pairs)}")
    return n_ok / max(n_total, 1)


def run_single_seed(cfg: dict, seed: int) -> tuple[float, float]:
    dbg = cfg["DEBUG"]
    init_hp = dict(cfg["INIT_HP"])
    eval_eps = int(dbg["eval_episodes"])
    grid_size = int(dbg["grid_size"])
    max_turns = int(dbg["max_turns"])
    max_ctx = int(dbg["max_context_length"])
    max_new = int(dbg["max_output_tokens"])

    accelerator = create_llm_accelerator()
    torch.manual_seed(seed)
    model_name = init_hp.get("MODEL_NAME")
    if model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(init_hp.get("TRUST_REMOTE_CODE", False)),
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        actor_network = None
    else:
        tokenizer = TinyDigitTokenizer()
        actor_network = build_tiny_actor_network(
            use_value_head=(init_hp["ALGO"] == "LLMPPO"),
        )

    rng = Random(seed)

    init_hp.setdefault("ALGO", "LLMPPO")
    init_hp.setdefault("USE_VLLM", False)
    init_hp.setdefault("MAX_MODEL_LEN", max_ctx)
    init_hp.setdefault("MAX_OUTPUT_TOKENS", max_new)
    init_hp.setdefault("SEED", seed)
    if "LR" not in init_hp and "LR_ACTOR" in init_hp:
        init_hp["LR"] = init_hp["LR_ACTOR"]

    if init_hp["USE_VLLM"]:
        init_hp.setdefault(
            "VLLM_CONFIG",
            {
                "sleep_mode": False,
                "max_num_seqs": 16,
                "gpu_memory_utilization": 0.85,
            },
        )
    if init_hp.get("USE_VLLM", False):
        init_hp.setdefault("USE_MEMORY_EFFICIENT_PARAMS", True)
        if init_hp.get("USE_MEMORY_EFFICIENT_PARAMS", True):
            init_hp["VLLM_CONFIG"]["sleep_mode"] = True

    vllm_cfg = (
        VLLMConfig(**init_hp["VLLM_CONFIG"]) if init_hp.get("USE_VLLM", False) else None
    )

    pop = create_population(
        algo=str(init_hp["ALGO"]),
        net_config=None,
        INIT_HP=init_hp,
        population_size=1,
        accelerator=accelerator,
        tokenizer=tokenizer,
        model_name=model_name,
        actor_network=actor_network,
        lora_config=lora_config_from_dict(dbg["lora"]),
        vllm_config=vllm_cfg,
    )
    agent = pop[0]

    suppress = dbg.get("suppress_tokens")
    if suppress is not None:
        agent.generation_config.suppress_tokens = list(suppress)

    try:
        pre_acc = evaluate_accuracy(
            agent, tokenizer, grid_size, max_turns, eval_eps, greedy=False
        )
        pre_acc_g = evaluate_accuracy(
            agent, tokenizer, grid_size, max_turns, eval_eps, greedy=True
        )
        print(
            f"[seed={seed}] pre-train acc (sampled/greedy): {pre_acc:.3f}/{pre_acc_g:.3f}"
        )
        print("\nPre-training detailed eval:")
        detailed_eval(agent, tokenizer, grid_size, max_turns)

        def env_factory() -> TokenObservationWrapper:
            return TokenObservationWrapper(
                GridNavigationEnv(
                    grid_size=grid_size,
                    max_turns=max_turns,
                    seed=rng.randint(0, 2**31),
                ),
                tokenizer,
                max_turns,
                tokenizer.pad_token_id,
                apply_chat_template=False,
                max_model_len=max_ctx,
                max_output_tokens=max_new,
            )

        eval_fn = lambda a: evaluate_accuracy(
            a, tokenizer, grid_size, max_turns, eval_eps, greedy=True
        )

        original_save = train_llm.save_llm_checkpoint
        train_llm.save_llm_checkpoint = lambda *args, **kwargs: None
        try:
            finetune_llm_multiturn(
                pop=[agent],
                max_turns=max_turns,
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

        post_acc = evaluate_accuracy(
            agent, tokenizer, grid_size, max_turns, eval_eps, greedy=False
        )
        post_acc_g = evaluate_accuracy(
            agent, tokenizer, grid_size, max_turns, eval_eps, greedy=True
        )
        print(
            f"\n[seed={seed}] post-train acc (sampled/greedy): "
            f"{post_acc:.3f}/{post_acc_g:.3f}"
        )
        print(
            f"[seed={seed}] improvement (sampled/greedy): "
            f"{post_acc - pre_acc:+.3f}/{post_acc_g - pre_acc_g:+.3f}"
        )
        print("\nPost-training detailed eval:")
        detailed_eval(agent, tokenizer, grid_size, max_turns)
        return post_acc - pre_acc, post_acc_g - pre_acc_g
    finally:
        pass


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
    main(load_debug_config("ppo_grid_navigation.yaml"))
