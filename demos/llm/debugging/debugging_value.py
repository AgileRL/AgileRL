# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Value head smoke test: constant reward through the LLMPPO learn path."""

from __future__ import annotations

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    raise ImportError("LLM dependencies are not installed.")

import torch
from config_load import load_debug_config
from datasets import Dataset
from llm_debug_utils import lora_config_from_dict
from tiny_model import TinyDigitTokenizer, build_tiny_actor_network
from torch.utils.data import DataLoader

from agilerl.algorithms import LLMPPO
from agilerl.llm_envs import apply_chat_template
from agilerl.utils.algo_utils import stack_and_pad_experiences


class ReasoningProbeEnv:
    """Single-turn reasoning probe that scores decoded completions via ``reward_fn``.

    Standalone (owns its own dataloader) because the value-head smoke test drives
    the batched tokenized-prompt surface directly at the token level: ``reset``
    returns a list of prompt dicts and ``step`` decodes the generated completions
    and returns ``(next_prompts, rewards)``. It only ever walks the train split.
    """

    def __init__(
        self,
        train_dataset,
        tokenizer,
        reward_fn,
        conversation_template,
        data_batch_size_per_gpu=8,
        seed=42,
    ) -> None:
        self.conversation_template = conversation_template
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        generator = torch.Generator().manual_seed(seed)
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=data_batch_size_per_gpu,
            shuffle=True,
            collate_fn=self._collate_fn,
            generator=generator,
        )
        self._iter = iter(self.dataloader)
        self.num_epochs = 0
        self.reset_called = False

    def _collate_fn(self, batch):
        """Collate ``(question, answer)`` rows into chat-templated token prompts."""
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        tokenized_prompts = [
            apply_chat_template(self.conversation_template, q, a, self.tokenizer)
            for q, a in zip(questions, answers, strict=False)
        ]
        return {
            "question": questions,
            "answer": answers,
            "tokenized_prompts": tokenized_prompts,
        }

    def reset(self, reset_dataloaders: bool = False):
        if reset_dataloaders:
            self._iter = iter(self.dataloader)
        self.reset_called = True
        prompts = self._next_prompts()
        self.last_tokenized_prompts = prompts
        return prompts

    def step(self, completions):
        self.reset_called = False
        rewards = self._decode_and_evaluate(completions)
        new_prompts = self._next_prompts()
        self.last_tokenized_prompts = new_prompts
        return new_prompts, rewards

    def _next_prompts(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self.num_epochs += 1
            self._iter = iter(self.dataloader)
            batch = next(self._iter)
        self.questions = batch["question"]
        self.answers = batch["answer"]
        return [
            {
                "input_ids": prompt["input_ids"],
                "attention_mask": prompt["attention_mask"],
                "text": None,
            }
            for prompt in batch["tokenized_prompts"]
        ]

    def _decode_and_evaluate(self, completions):
        total_rewards = []
        for idx, (group_completion, answer, question) in enumerate(
            zip(completions, self.answers, self.questions, strict=False),
        ):
            completion_to_decode = group_completion[
                :,
                self.last_tokenized_prompts[idx]["input_ids"].shape[1] :,
            ]
            decoded = self.tokenizer.batch_decode(
                completion_to_decode,
                skip_special_tokens=True,
            )
            total_rewards.append([self.reward_fn(c, answer, question) for c in decoded])
        return torch.tensor(total_rewards)


def constant_reward_factory(value: float):
    def _fn(_completion: str, _answer: str, _question: str) -> float:
        return value

    return _fn


def make_dataset(size: int) -> Dataset:
    return Dataset.from_dict({"question": ["11"] * size, "answer": ["1"] * size})


def get_terminal_values(
    agent: LLMPPO,
    completion_ids: list[torch.Tensor],
    action_masks: list[torch.Tensor],
) -> torch.Tensor:
    stacked_ids, stacked_masks = stack_and_pad_experiences(
        completion_ids,
        action_masks,
        padding_values=[agent.pad_token_id, False],
    )
    stacked_ids = stacked_ids.to(agent.device)
    stacked_masks = stacked_masks.to(agent.device)

    _, _, values = agent._fused_forward_no_grad(
        stacked_ids,
        batch_size=stacked_ids.shape[0],
    )

    last_action_idx = stacked_masks.long().cumsum(dim=1).argmax(dim=1)
    return values.gather(1, last_action_idx.unsqueeze(1)).squeeze(1)


def main(cfg: dict) -> None:
    dbg = cfg["DEBUG"]
    init_hp = cfg["INIT_HP"]
    lora = dbg["lora"]
    reward = float(dbg["constant_reward"])
    max_ctx = int(dbg["max_context_length"])
    max_new = int(dbg["max_output_tokens"])
    min_new = int(dbg["min_output_tokens"])

    torch.manual_seed(0)
    tokenizer = TinyDigitTokenizer()
    actor_network = build_tiny_actor_network(use_value_head=True)
    train_dataset, test_dataset = make_dataset(int(dbg["dataset_size"]))

    conversation_template = [
        {"role": "system", "content": "Output one digit."},
        {"role": "user", "content": "{question}"},
        {"role": "assistant", "content": ""},
    ]

    env = ReasoningProbeEnv(
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_fn=constant_reward_factory(reward),
        conversation_template=conversation_template,
        data_batch_size_per_gpu=int(dbg["data_batch_size_per_gpu"]),
        seed=0,
    )

    agent = LLMPPO(
        model_name=None,
        actor_network=actor_network,
        lora_config=lora_config_from_dict(lora),
        micro_batch_size_per_gpu=int(dbg["micro_batch_size_per_gpu"]),
        use_vllm=False,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        use_separate_reference_adapter=True,
        batch_size=int(init_hp["BATCH_SIZE"]),
        beta=float(init_hp["BETA"]),
        lr_actor=float(init_hp["LR_ACTOR"]),
        lr_critic=float(init_hp["LR_CRITIC"]),
        clip_coef=float(init_hp["CLIP_COEF"]),
        max_grad_norm=float(init_hp["MAX_GRAD_NORM"]),
        update_epochs=int(init_hp["UPDATE_EPOCHS"]),
        temperature=float(init_hp["TEMPERATURE"]),
        max_output_tokens=max_new,
        min_output_tokens=min_new,
        max_model_len=max_ctx,
        accelerator=None,
        vf_coef=float(init_hp["VF_COEF"]),
        gamma=float(init_hp["GAMMA"]),
        gae_lambda=float(init_hp["GAE_LAMBDA"]),
        seed=0,
    )

    num_steps = int(dbg["num_train_steps"])
    log_interval = int(dbg["log_interval"])
    tolerance = float(dbg["tolerance"])

    print(
        f"[value-debug] constant reward={reward}, gamma=1.0; "
        f"expected V(terminal) ≈ {reward:.4f}\n"
    )

    prompts = env.reset(reset_dataloaders=True)
    with torch.no_grad():
        init_result = agent.get_action(prompts, training=False)
        init_ids, init_masks = init_result.completion_ids, init_result.action_masks
        init_values = get_terminal_values(agent, init_ids, init_masks)

    init_mean = float(init_values.mean().item())
    print(
        f"[value-debug] initial V(terminal): "
        f"mean={init_mean:.4f}, std={init_values.std():.4f}"
    )

    prompts, _ = env.step(init_ids)

    for step in range(num_steps):
        agent.set_reference_policy(env.num_epochs)
        rollout = agent.get_action(prompts)
        completion_ids, action_masks = rollout.completion_ids, rollout.action_masks
        next_prompts, rewards = env.step(completion_ids)

        metrics = agent.learn(
            (completion_ids, action_masks, rewards),
            sampling_logps=rollout.sampling_logps,
        )
        vf_loss = metrics["vf_loss"]
        prompts = next_prompts

        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                snap = agent.get_action(prompts, training=False)
                snap_ids, snap_masks = snap.completion_ids, snap.action_masks
                snap_values = get_terminal_values(agent, snap_ids, snap_masks)
            print(
                f"[value-debug] step {step + 1:4d} | "
                f"vf_loss={learn_metrics['vf_loss']:.4f} | "
                f"V(terminal) mean={snap_values.mean():.4f}, "
                f"std={snap_values.std():.4f}"
            )

    with torch.no_grad():
        final_result = agent.get_action(prompts, training=False)
        final_ids, final_masks = (
            final_result.completion_ids,
            final_result.action_masks,
        )
        final_values = get_terminal_values(agent, final_ids, final_masks)

    final_mean = float(final_values.mean().item())
    init_error = abs(init_mean - reward)
    final_error = abs(final_mean - reward)

    print(
        f"\n[value-debug] final V(terminal): "
        f"mean={final_mean:.4f}, std={final_values.std():.4f}"
    )
    print(f"[value-debug] error: {init_error:.4f} -> {final_error:.4f}")

    improved = final_error < init_error
    converged = final_error < tolerance
    if improved and converged:
        print("[value-debug] PASS — value head converged toward constant return.")
    elif improved:
        print(
            f"[value-debug] FAIL — improved but error {final_error:.4f} > {tolerance}."
        )
    else:
        print(
            f"[value-debug] FAIL — no improvement "
            f"(init_error={init_error:.4f}, final_error={final_error:.4f})."
        )


if __name__ == "__main__":
    main(load_debug_config("ppo_value_head.yaml"))
