"""Stage-3 debugging: multi-turn PPO with a 1D grid navigation GEM environment.

Tests that turn-level GAE correctly handles multi-turn trajectories where
each generation is treated as a single action.  The environment provides
meaningful state transitions and per-turn rewards, forcing the agent to
condition on environment feedback across turns.

Environment: GridNavigationEnv (extends gem.Env)
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

from tqdm import tqdm

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(msg)

try:
    from gem.core import Env as GemEnv
except ImportError:
    msg = "gem-llm is required for this script. Install with: pip install gem-llm"
    raise ImportError(msg) from None

import torch
import yaml
from peft import LoraConfig

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import create_llm_accelerator
from benchmarking.tiny_model import TinyDigitTokenizer, build_tiny_actor_network

MAX_CONTEXT_LENGTH = 128
MAX_OUTPUT_TOKENS = 1
GRID_SIZE = 4
MAX_TURNS = 5
STEP_COST = -0.1
EVAL_EPISODES = 128


class GridNavigationEnv(GemEnv):
    """1D grid navigation: move left/right to reach a target position.

    The agent sees its starting position and target once (initial obs),
    then receives only its current position as feedback after each move.
    It must remember the target from context and navigate toward it.

    Actions map to: "1" = left, "2" = stay, "3" = right.
    Positions are clamped to [0, grid_size-1].
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        max_turns: int = MAX_TURNS,
        step_cost: float = STEP_COST,
        seed: int = 42,
    ):
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.step_cost = step_cost
        self.rng = Random(seed)
        self.position = 0
        self.target = 0
        self.turn = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = Random(seed)
        self.position = self.rng.randint(0, self.grid_size - 1)
        others = [p for p in range(self.grid_size) if p != self.position]
        self.target = self.rng.choice(others)
        self.turn = 0
        obs = f"{self.position}{self.target}"
        return obs, {"position": self.position, "target": self.target}

    def step(self, action):
        self.turn += 1

        move = None
        for ch in str(action):
            if ch in "123":
                move = int(ch)
                break

        if move == 1:
            self.position = max(0, self.position - 1)
        elif move == 3:
            self.position = min(self.grid_size - 1, self.position + 1)
        # move == 2 or invalid: stay in place

        obs = str(self.position)

        if self.position == self.target:
            return obs, 1.0, True, False, {"success": True}
        if self.turn >= self.max_turns:
            return obs, -1.0, True, False, {"success": False}
        return obs, self.step_cost, False, False, {}


def rollout_multiturn(
    agent: LLMPPO,
    batch_size: int,
    tokenizer: TinyDigitTokenizer,
    max_turns: int,
    grid_size: int,
    rng: Random,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Roll out a batch of multi-turn episodes and build trajectory tensors.

    Returns per-episode lists of:
      completion_ids  [1, seq_len_i]
      action_masks    [1, seq_len_i - 1]
      turn_ids        [1, seq_len_i - 1]
      rewards         [max_turns]
    """
    all_completion_ids: list[torch.Tensor] = []
    all_action_masks: list[torch.Tensor] = []
    all_turn_ids: list[torch.Tensor] = []
    all_rewards: list[torch.Tensor] = []

    for _ in range(batch_size):
        env = GridNavigationEnv(
            grid_size=grid_size, max_turns=max_turns, seed=rng.randint(0, 2**31)
        )
        obs, _info = env.reset()

        prompt_encoded = tokenizer(
            [obs],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_attention_mask=True,
        )

        turn_boundaries: list[tuple[int, int, int]] = []
        turn_rewards: list[float] = []
        full_ids: torch.Tensor | None = None

        for turn_idx in range(max_turns):
            prompt_dict = {
                "input_ids": prompt_encoded["input_ids"],
                "attention_mask": prompt_encoded["attention_mask"],
            }
            prompt_len = prompt_dict["input_ids"].shape[1]

            completion_ids, _ = agent.get_action([prompt_dict], training=True)
            full_ids = completion_ids[0]

            gen_start = prompt_len
            gen_end = full_ids.shape[1]
            turn_boundaries.append((gen_start, gen_end, turn_idx))

            gen_tokens = full_ids[0, gen_start:]
            gen_text = tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)

            next_obs, reward, terminated, truncated, _step_info = env.step(gen_text)
            turn_rewards.append(float(reward))

            if terminated or truncated:
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

        assert full_ids is not None
        seq_len = full_ids.shape[1]
        action_mask = torch.zeros(1, seq_len - 1, dtype=torch.bool)
        turn_ids = torch.full((1, seq_len - 1), -1, dtype=torch.long)

        for gen_start, gen_end, tidx in turn_boundaries:
            mask_start = gen_start - 1
            mask_end = gen_end - 1
            if mask_start >= 0 and mask_end <= seq_len - 1:
                action_mask[0, mask_start:mask_end] = True
                turn_ids[0, mask_start:mask_end] = tidx

        for pos in range(seq_len - 1):
            if full_ids[0, pos + 1].item() == tokenizer.pad_token_id:
                action_mask[0, pos] = False
                turn_ids[0, pos] = -1

        while len(turn_rewards) < max_turns:
            turn_rewards.append(0.0)

        all_completion_ids.append(full_ids)
        all_action_masks.append(action_mask)
        all_turn_ids.append(turn_ids)
        all_rewards.append(torch.tensor(turn_rewards, dtype=torch.float))

    return all_completion_ids, all_action_masks, all_turn_ids, all_rewards


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

    num_steps = init_hp.get("MAX_STEPS", 4096 * 3)
    batch_size = init_hp["BATCH_SIZE"]
    evaluation_interval = init_hp.get("EVALUATION_INTERVAL", 50)
    total_samples = 0
    step_count = 0

    bar_format = (
        "{l_bar}{bar:10}| {n:4}/{total_fmt} "
        "[{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    )
    pbar = tqdm(
        total=num_steps,
        unit="sample",
        bar_format=bar_format,
        ascii=True,
        dynamic_ncols=True,
    )

    print("\nTraining...")
    while total_samples < num_steps:
        llm_ppo.set_reference_policy(0)

        comp_ids_list, action_masks_list, turn_ids_list, rewards_list = (
            rollout_multiturn(
                llm_ppo, batch_size, tokenizer, max_turns, grid_size, rng
            )
        )

        rewards_2d = torch.stack(rewards_list)

        (turn_ids_padded,) = stack_and_pad_experiences(
            turn_ids_list,
            padding_values=[-1],
        )

        experiences = (comp_ids_list, action_masks_list, rewards_2d)

        loss, kl, pg_loss, vf_loss, entropy = llm_ppo.learn(
            experiences, tokenizer, turn_ids=turn_ids_padded
        )

        episode_scores = [r.sum().item() for r in rewards_list]
        mean_score = sum(episode_scores) / max(len(episode_scores), 1)
        accuracy = (
            sum(1 for r in rewards_list if (r > 0).any())
            / max(len(rewards_list), 1)
        )

        total_samples += batch_size
        step_count += 1

        pbar.set_postfix(
            loss=f"{loss:.4f}",
            kl=f"{kl:.4f}",
            score=f"{mean_score:.3f}",
            acc=f"{accuracy:.3f}",
        )
        pbar.update(batch_size)

        if step_count % evaluation_interval == 0:
            eval_acc = evaluate_accuracy(
                llm_ppo, tokenizer, grid_size, max_turns, EVAL_EPISODES, greedy=True
            )

            banner_text = f"Step {step_count}  ({total_samples} samples)"
            banner_width = max(len(banner_text) + 8, 40)
            border = "=" * banner_width
            pbar.write(
                f"\n{border}\n"
                f"{banner_text.center(banner_width)}\n"
                f"{border}\n"
                f"Train score:\t\t{mean_score:.3f}\n"
                f"Train accuracy:\t\t{accuracy:.3f}\n"
                f"Eval accuracy (greedy):\t{eval_acc:.3f}\n"
                f"Loss:\t\t\t{loss:.4f}\n"
                f"KL-divergence:\t\t{kl:.4f}\n"
                f"PG loss:\t\t{pg_loss:.4f}\n"
                f"VF loss:\t\t{vf_loss:.4f}\n"
                f"Entropy:\t\t{entropy:.4f}\n"
                f"{border}"
            )

    pbar.close()

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
    init_hp["TEMPERATURE"] = 0.4
    init_hp["VF_COEF"] = 0.1
    init_hp["GAMMA"] = 0.99
    init_hp["GAE_LAMBDA"] = 0.95
    init_hp["MAX_STEPS"] = 4096 * 3
    main(init_hp, (0,))
