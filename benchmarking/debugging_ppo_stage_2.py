from __future__ import annotations

import statistics
from random import Random
from types import MethodType

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig

from agilerl.algorithms import LLMPPO
from agilerl.training import train_llm
from agilerl.training.train_llm import finetune_llm_reasoning
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import ReasoningGym, masked_whiten, create_llm_accelerator
from benchmarking.tiny_model import (
    build_tiny_actor_network,
    TinyDigitTokenizer,
)

from accelerate import Accelerator

MAX_CONTEXT_LENGTH = 128
MAX_OUTPUT_TOKENS = 1
EVAL_BATCHES = 8
TARGET_TOKEN_IDS = (1, 2, 3)
TEST_POLICY_ONLY = False


def target_from_question(question: str) -> str:
    # Map 2-digit prompt to one of 3 target tokens in {1, 2, 3}.
    total = sum(int(ch) for ch in question)
    return str((total % 3) + 1)


def make_dataset(
    train_size: int = 8192,
    test_size: int = 1024,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    rng = Random(seed)
    questions_space = [f"{a}{b}" for a in TARGET_TOKEN_IDS for b in TARGET_TOKEN_IDS]

    def build_split(size: int) -> Dataset:
        # Balanced class coverage by cycling through all question patterns.
        questions = [questions_space[i % len(questions_space)] for i in range(size)]
        rng.shuffle(questions)
        answers = [target_from_question(question) for question in questions]

        return Dataset.from_dict({"question": questions, "answer": answers})

    return build_split(train_size), build_split(test_size)


def conditional_reward(completion: str, answer: str, _question: str) -> float:
    return 1.0 if completion == answer else -1.0


def evaluate_accuracy(
    agent: LLMPPO,
    env: ReasoningGym,
    batches: int = EVAL_BATCHES,
    greedy_like: bool = False,
) -> tuple[float, dict[int, float], float]:
    original_temperature = agent.generation_config.temperature
    original_top_k = agent.generation_config.top_k
    original_top_p = agent.generation_config.top_p
    if greedy_like:
        agent.generation_config.temperature = 1e-3
        agent.generation_config.top_k = 1
        agent.generation_config.top_p = 1.0

    total = 0
    correct = 0
    class_total = {target: 0 for target in TARGET_TOKEN_IDS}
    class_correct = {target: 0 for target in TARGET_TOKEN_IDS}
    all_sq_errors = []
    try:
        with env.eval_mode(), torch.no_grad():
            prompts = env.reset(reset_dataloaders=True)
            for _ in range(batches):
                answers = [int(answer) for answer in env.answers]
                completion_ids, action_masks = agent.get_action(prompts, training=False)
                seq_rewards = []
                for prompt, group_completion, answer in zip(
                    prompts,
                    completion_ids,
                    answers,
                    strict=False,
                ):
                    prompt_len = prompt["input_ids"].shape[1]
                    preds = group_completion[:, prompt_len]
                    batch_correct = int((preds == answer).sum().item())
                    batch_total = preds.shape[0]
                    total += batch_total
                    correct += batch_correct
                    class_total[answer] += batch_total
                    class_correct[answer] += batch_correct
                    reward = 1.0 if preds.item() == answer else -1.0
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
    finally:
        agent.generation_config.temperature = original_temperature
        agent.generation_config.top_k = original_top_k
        agent.generation_config.top_p = original_top_p

    per_class = {
        target: class_correct[target] / max(class_total[target], 1)
        for target in TARGET_TOKEN_IDS
    }
    critic_mse = sum(all_sq_errors) / max(len(all_sq_errors), 1)
    return correct / max(total, 1), per_class, critic_mse


def enable_reinforce_style_advantages(agent: LLMPPO) -> None:
    def _reinforce_like_returns(
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del values
        # Policy-only mode: use token rewards directly as returns and
        # whiten masked rewards for REINFORCE-style advantages.
        returns = rewards
        advantages = masked_whiten(rewards, action_mask)
        return returns, advantages

    def _logprobs_no_critic(
        self: LLMPPO,
        ids: torch.Tensor,
        batch_size: int,
        use_reference: bool = False,
        eval_mode: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        with self.select_adapter("reference" if use_reference else "actor"):
            self.actor.train(mode=not eval_mode)
            if attention_mask is None:
                attention_mask = ids != self.pad_token_id
            if self.calc_position_embeddings:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

            num_samples = ids.shape[0]
            log_probs = []
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min((batch_start + batch_size), num_samples)
                batch_ids = ids[batch_start:batch_end, :]
                batch_attention_mask = attention_mask[batch_start:batch_end, :]
                model_kwargs = {
                    "input_ids": batch_ids,
                    "attention_mask": batch_attention_mask,
                    "use_cache": False,
                }
                if self.calc_position_embeddings:
                    model_kwargs["position_ids"] = position_ids[
                        batch_start:batch_end, :
                    ]

                # NOTE: use the underlying CausalLM directly to bypass value-head forward.
                output = self.actor.pretrained_model.forward(**model_kwargs)
                logits = output[0] if isinstance(output, tuple) else output.logits
                log_prob = self._memory_efficient_logits(
                    logits[:, :-1],
                    batch_ids[:, 1:],
                )
                log_probs.append(log_prob)

            full_log_probs = torch.cat(log_probs, dim=0)
            zero_values = torch.zeros_like(full_log_probs)
            return full_log_probs, zero_values, None

    agent._compute_gae_returns = _reinforce_like_returns  # type: ignore[method-assign]
    agent._get_logprobs_and_values = MethodType(  # type: ignore[method-assign]
        _logprobs_no_critic,
        agent,
    )


def run_single_seed(init_hp: dict, seed: int) -> tuple[float, float]:
    accelerator = create_llm_accelerator()
    torch.manual_seed(seed)
    actor_network = build_tiny_actor_network()
    tokenizer = TinyDigitTokenizer()
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
        reward_fn=conditional_reward,
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
        lr_actor=init_hp["LR"],
        lr_critic=init_hp.get("LR_CRITIC"),
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

    if TEST_POLICY_ONLY:
        enable_reinforce_style_advantages(llm_ppo)
        print(f"[seed={seed}] mode: test_policy_only=True (REINFORCE-style advantages)")
    else:
        print(f"[seed={seed}] mode: test_policy_only=False (PPO GAE advantages)")

    pre_acc, pre_class, pre_mse = evaluate_accuracy(llm_ppo, env, greedy_like=False)
    pre_acc_g, pre_class_g, pre_mse_g = evaluate_accuracy(
        llm_ppo, env, greedy_like=True
    )
    print(
        f"[seed={seed}] pre-train acc (sampled/greedy-like): "
        f"{pre_acc:.3f}/{pre_acc_g:.3f}"
    )
    print(f"[seed={seed}] pre per-class sampled: {pre_class}")
    print(f"[seed={seed}] pre per-class greedy-like: {pre_class_g}")
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
            evaluation_interval=50,
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
            max_steps=4096 * 3,
        )
    finally:
        train_llm.save_llm_checkpoint = original_save_checkpoint

    post_acc, post_class, post_mse = evaluate_accuracy(llm_ppo, env, greedy_like=False)
    post_acc_g, post_class_g, post_mse_g = evaluate_accuracy(
        llm_ppo, env, greedy_like=True
    )
    print(
        f"[seed={seed}] post-train acc (sampled/greedy-like): "
        f"{post_acc:.3f}/{post_acc_g:.3f}"
    )
    print(f"[seed={seed}] post per-class sampled: {post_class}")
    print(f"[seed={seed}] post per-class greedy-like: {post_class_g}")
    print(
        f"[seed={seed}] post critic MSE (sampled/greedy-like): {post_mse:.4f}/{post_mse_g:.4f}"
    )
    print(
        f"[seed={seed}] improvement (sampled/greedy-like): "
        f"{post_acc - pre_acc:+.3f}/{post_acc_g - pre_acc_g:+.3f}"
    )
    print(
        f"[seed={seed}] critic MSE change (sampled/greedy-like): "
        f"{post_mse - pre_mse:+.4f}/{post_mse_g - pre_mse_g:+.4f}"
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
        f"[summary] greedy-like improvement over {len(seeds)} seeds: "
        f"{greedy_mean:+.3f} +/- {greedy_std:.3f}"
    )


if __name__ == "__main__":
    with open("configs/training/llm_finetuning/ppo_llm.yaml") as file:
        config = yaml.safe_load(file)

    init_hp = config["INIT_HP"]
    # Stage-2 smoke-test overrides.
    init_hp["BATCH_SIZE"] = 32
    init_hp["UPDATE_EPOCHS"] = 4
    init_hp["LR"] = 1e-4
    init_hp["BETA"] = 0.05
    init_hp["TEMPERATURE"] = 0.4
    init_hp["VF_COEF"] = 0.1
    init_hp["GAMMA"] = 1.0
    init_hp["GAE_LAMBDA"] = 0.95
    main(init_hp, (0,))
