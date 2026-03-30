"""Stage-3 debugging: multi-turn PPO with a 1D grid navigation GEM environment.

Tests that turn-level GAE correctly handles multi-turn trajectories where
each generation is treated as a single action.  The environment provides
meaningful state transitions and per-turn rewards, forcing the agent to
condition on environment feedback across turns.

Environment: GridNavigationEnv from ``agilerl.utils.probe_envs_llm``
  - 1D grid with positions 0..3.
  - Initial observation: "{position}{target}" (e.g. "03").
  - Action: "1" = left, "2" = stay, "3" = right.
  - Feedback observation: new position as a single digit.
  - Rewards: -0.1 step cost per turn, +1.0 on reaching target,
    -1.0 if max_turns exhausted.
"""

from __future__ import annotations

import statistics
from random import Random

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(msg)

import torch
import yaml
from peft import LoraConfig

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.training import train_llm
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.utils.probe_envs_llm import GridNavigationEnv
from benchmarking.tiny_model import TinyDigitTokenizer, build_tiny_actor_network

MAX_CONTEXT_LENGTH = 128
MAX_OUTPUT_TOKENS = 1
GRID_SIZE = 4
MAX_TURNS = 5
EVAL_EPISODES = 128


def evaluate_accuracy(
    agent: LLMPPO,
    tokenizer: TinyDigitTokenizer,
    grid_size: int,
    max_turns: int,
    num_episodes: int,
    greedy: bool = False,
) -> float:
    """Evaluate multi-turn accuracy (fraction reaching target) over many episodes."""
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
                    prompt_dict = {
                        "input_ids": prompt_encoded["input_ids"],
                        "attention_mask": prompt_encoded["attention_mask"],
                    }
                    prompt_len = prompt_dict["input_ids"].shape[1]

                    completion_ids, _ = agent.get_action(
                        [prompt_dict], training=False
                    )
                    full_ids = completion_ids[0]
                    gen_tokens = full_ids[0, prompt_len:]
                    gen_text = tokenizer.decode(
                        gen_tokens.tolist(), skip_special_tokens=True
                    )

                    next_obs, reward, terminated, truncated, _step_info = env.step(
                        gen_text
                    )

                    if terminated or truncated:
                        total += 1
                        if reward > 0:
                            correct += 1
                        break

                    feedback_ids = torch.tensor(
                        [tokenizer.encode(next_obs)], dtype=torch.long,
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
    agent: LLMPPO,
    tokenizer: TinyDigitTokenizer,
    grid_size: int,
    max_turns: int,
) -> float:
    """Run every (start, target) pair with greedy decoding, print per-turn actions."""
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
                        grid_size=grid_size, max_turns=max_turns, seed=0,
                    )
                    env.position = start
                    env.target = target
                    env.turn = 0

                    obs = f"{start}{target}"
                    prompt_encoded = tokenizer(
                        [obs], return_tensors="pt", padding=True,
                        padding_side="left", return_attention_mask=True,
                    )

                    actions: list[str] = []
                    success = False
                    for _ in range(max_turns):
                        prompt_dict = {
                            "input_ids": prompt_encoded["input_ids"],
                            "attention_mask": prompt_encoded["attention_mask"],
                        }
                        prompt_len = prompt_dict["input_ids"].shape[1]
                        completion_ids, _ = agent.get_action(
                            [prompt_dict], training=False,
                        )
                        full_ids = completion_ids[0]
                        gen_tokens = full_ids[0, prompt_len:]
                        gen_text = tokenizer.decode(
                            gen_tokens.tolist(), skip_special_tokens=True,
                        )
                        raw_tok = gen_tokens[0].item() if len(gen_tokens) > 0 else -1
                        actions.append(
                            action_names.get(gen_text, f"?{raw_tok}")
                        )

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
                            [full_ids, feedback_ids], dim=1,
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


def run_single_seed(init_hp: dict, seed: int) -> tuple[float, float]:
    accelerator = create_llm_accelerator()
    torch.manual_seed(seed)
    actor_network = build_tiny_actor_network()
    tokenizer = TinyDigitTokenizer()
    rng = Random(seed)

    grid_size = GRID_SIZE
    max_turns = MAX_TURNS

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
        seed=seed,
        gradient_checkpointing=True,
    )

    # Only allow valid action tokens during generation (1=left, 2=stay, 3=right).
    # Suppress PAD (5) and EOS (6) which are not valid actions and would waste
    # turns. Also suppress digit tokens 0 and 4 which aren't valid actions.
    llm_ppo.generation_config.suppress_tokens = [0, 4, 5, 6]

    pre_acc = evaluate_accuracy(
        llm_ppo, tokenizer, grid_size, max_turns, EVAL_EPISODES, greedy=False
    )
    pre_acc_g = evaluate_accuracy(
        llm_ppo, tokenizer, grid_size, max_turns, EVAL_EPISODES, greedy=True
    )
    print(
        f"[seed={seed}] pre-train acc (sampled/greedy): "
        f"{pre_acc:.3f}/{pre_acc_g:.3f}"
    )
    print("\nPre-training detailed eval:")
    detailed_eval(llm_ppo, tokenizer, grid_size, max_turns)

    max_samples = init_hp.get("MAX_STEPS", 4096 * 3)
    batch_size = init_hp["BATCH_SIZE"]
    evaluation_interval = init_hp.get("EVALUATION_INTERVAL", 50)
    # Legacy loop counted total environment samples; finetune_llm_multiturn's
    # max_steps is the number of PPO updates (one batch per update).
    max_train_steps = (max_samples + batch_size - 1) // batch_size

    env_fn = lambda: GridNavigationEnv(
        grid_size=grid_size,
        max_turns=max_turns,
        seed=rng.randint(0, 2**31),
    )
    eval_fn = lambda agent: evaluate_accuracy(
        agent, tokenizer, grid_size, max_turns, EVAL_EPISODES, greedy=True
    )

    original_save_checkpoint = train_llm.save_llm_checkpoint
    train_llm.save_llm_checkpoint = lambda *args, **kwargs: None
    try:
        finetune_llm_multiturn(
            pop=[llm_ppo],
            env_fn=env_fn,
            tokenizer=tokenizer,
            max_turns=max_turns,
            init_hp={"ALGO": "LLMPPO", **init_hp},
            max_steps=max_train_steps,
            eval_fn=eval_fn,
            evaluation_interval=evaluation_interval,
            wb=False,
            save_elite=False,
            verbose=True,
            accelerator=accelerator,
        )
    finally:
        train_llm.save_llm_checkpoint = original_save_checkpoint

    post_acc = evaluate_accuracy(
        llm_ppo, tokenizer, grid_size, max_turns, EVAL_EPISODES, greedy=False
    )
    post_acc_g = evaluate_accuracy(
        llm_ppo, tokenizer, grid_size, max_turns, EVAL_EPISODES, greedy=True
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
    detailed_eval(llm_ppo, tokenizer, grid_size, max_turns)
    return post_acc - pre_acc, post_acc_g - pre_acc_g


def main(init_hp: dict, seeds: tuple[int, ...] = (0,)) -> None:
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
    init_hp["BATCH_SIZE"] = 32
    init_hp["UPDATE_EPOCHS"] = 4
    init_hp["LR_ACTOR"] = 1e-4
    init_hp["LR_CRITIC"] = 1e-3
    init_hp["BETA"] = 0.05
    init_hp["TEMPERATURE"] = 0.7
    init_hp["VF_COEF"] = 0.5
    init_hp["GAMMA"] = 0.99
    init_hp["GAE_LAMBDA"] = 0.95
    init_hp["MAX_STEPS"] = 4096 * 8
    main(init_hp, (0,))
