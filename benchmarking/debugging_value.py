"""Value-head smoke test through the full LLMPPO pipeline.

Every completion receives the same constant reward.  With gamma=1 and
gae_lambda=1 the GAE return at the terminal action position equals exactly
CONSTANT_REWARD.  After enough PPO ``learn()`` calls the value head should
predict V(terminal) ≈ CONSTANT_REWARD.

The test exercises the real code path: LoRA adapter wrapping, the two-pass
value-head hook in ``_get_logprobs_and_values``, ``_compute_token_rewards``,
``_compute_gae_returns``, and ``OptimizerWrapper``.
"""

from __future__ import annotations

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(msg)

import torch
from datasets import Dataset
from peft import LoraConfig

from agilerl.algorithms import LLMPPO
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import ReasoningGym
from benchmarking.tiny_model import build_tiny_actor_network, TinyDigitTokenizer

MAX_CONTEXT_LENGTH = 128
MAX_OUTPUT_TOKENS = 1
CONSTANT_REWARD = 1.0
NUM_TRAIN_STEPS = 200
LOG_INTERVAL = 1
TOLERANCE = 0.15


def constant_reward(_completion: str, _answer: str, _question: str) -> float:
    return CONSTANT_REWARD


def make_dataset(size: int = 4096) -> tuple[Dataset, Dataset]:
    data = Dataset.from_dict({"question": ["11"] * size, "answer": ["1"] * size})
    return data, data


def get_terminal_values(
    agent: LLMPPO,
    completion_ids: list[torch.Tensor],
    action_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Return value-head predictions at the terminal (reward) position of each sample."""
    stacked_ids, stacked_masks = stack_and_pad_experiences(
        completion_ids,
        action_masks,
        padding_values=[agent.pad_token_id, False],
    )
    stacked_ids = stacked_ids.to(agent.device)
    stacked_masks = stacked_masks.to(agent.device)

    _, values = agent._get_logprobs_and_values(
        stacked_ids,
        batch_size=stacked_ids.shape[0],
        use_reference=False,
        eval_mode=True,
    )

    last_action_idx = stacked_masks.long().cumsum(dim=1).argmax(dim=1)
    return values.gather(1, last_action_idx.unsqueeze(1)).squeeze(1)


def main() -> None:
    torch.manual_seed(0)
    tokenizer = TinyDigitTokenizer()
    actor_network = build_tiny_actor_network()
    train_dataset, test_dataset = make_dataset()

    conversation_template = [
        {"role": "system", "content": "Output one digit."},
        {"role": "user", "content": "{question}"},
        {"role": "assistant", "content": ""},
    ]

    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=constant_reward,
        conversation_template=conversation_template,
        data_batch_size_per_gpu=32,
        accelerator=None,
        max_context_length=MAX_CONTEXT_LENGTH,
        return_raw_completions=False,
        seed=0,
    )

    agent = LLMPPO(
        model_name=None,
        actor_network=actor_network,
        lora_config=LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj", "c_fc"],
            bias="none",
            modules_to_save=["summary"],
            task_type="CAUSAL_LM",
        ),
        micro_batch_size_per_gpu=8,
        use_vllm=False,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        use_separate_reference_adapter=True,
        batch_size=32,
        beta=0.0,
        lr_actor=1e-3,
        lr_critic=1e-3,
        clip_coef=0.2,
        max_grad_norm=1.0,
        update_epochs=4,
        temperature=1.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        min_output_tokens=MAX_OUTPUT_TOKENS,
        max_model_len=MAX_CONTEXT_LENGTH,
        accelerator=None,
        vf_coef=1.0,
        gamma=1.0,
        gae_lambda=1.0,
        seed=0,
    )

    print(
        f"[value-debug] smoke test: constant reward={CONSTANT_REWARD}, gamma=1.0\n"
        f"[value-debug] expected V(terminal) = {CONSTANT_REWARD:.4f}\n"
    )

    # --- Initial value predictions ---
    prompts = env.reset(reset_dataloaders=True)
    with torch.no_grad():
        init_ids, init_masks = agent.get_action(prompts, training=False)
        init_values = get_terminal_values(agent, init_ids, init_masks)

    init_mean = float(init_values.mean().item())
    print(
        f"[value-debug] initial V(terminal): "
        f"mean={init_mean:.4f}, std={init_values.std():.4f}"
    )

    # --- Training loop ---
    # Consume the initial batch so env state stays consistent.
    prompts, _ = env.step(init_ids)

    for step in range(NUM_TRAIN_STEPS):
        agent.set_reference_policy(env.num_epochs)
        completion_ids, action_masks = agent.get_action(prompts)
        next_prompts, rewards = env.step(completion_ids)

        loss, kl, pg_loss, vf_loss, entropy = agent.learn(
            (completion_ids, action_masks, rewards),
        )
        prompts = next_prompts

        if (step + 1) % LOG_INTERVAL == 0:
            with torch.no_grad():
                snap_ids, snap_masks = agent.get_action(prompts, training=False)
                snap_values = get_terminal_values(agent, snap_ids, snap_masks)
            print(
                f"[value-debug] step {step + 1:4d} | "
                f"vf_loss={vf_loss:.4f} | "
                f"V(terminal) mean={snap_values.mean():.4f}, "
                f"std={snap_values.std():.4f}"
            )

    # --- Final value predictions ---
    with torch.no_grad():
        final_ids, final_masks = agent.get_action(prompts, training=False)
        final_values = get_terminal_values(agent, final_ids, final_masks)

    final_mean = float(final_values.mean().item())
    init_error = abs(init_mean - CONSTANT_REWARD)
    final_error = abs(final_mean - CONSTANT_REWARD)

    print(
        f"\n[value-debug] final V(terminal): "
        f"mean={final_mean:.4f}, std={final_values.std():.4f}"
    )
    print(f"[value-debug] error: {init_error:.4f} -> {final_error:.4f}")

    improved = final_error < init_error
    converged = final_error < TOLERANCE
    if improved and converged:
        print(
            "[value-debug] PASS — value head learned the constant return "
            "through the full LLMPPO pipeline."
        )
    elif improved:
        print(
            f"[value-debug] FAIL — value head improved but did not converge "
            f"(error={final_error:.4f} > {TOLERANCE})."
        )
    else:
        print(
            f"[value-debug] FAIL — value head did not improve "
            f"(init_error={init_error:.4f}, final_error={final_error:.4f})."
        )


if __name__ == "__main__":
    main()
