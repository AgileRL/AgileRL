"""Value head smoke test: constant reward through the LLMPPO learn path."""

from __future__ import annotations

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    raise ImportError("LLM dependencies are not installed.")

import torch
from datasets import Dataset

from agilerl.algorithms import LLMPPO
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import ReasoningGym

from config_load import load_debug_config
from llm_debug_utils import lora_config_from_dict
from tiny_model import TinyDigitTokenizer, build_tiny_actor_network


def constant_reward_factory(value: float):
    def _fn(_completion: str, _answer: str, _question: str) -> float:
        return value

    return _fn


def make_dataset(size: int) -> tuple[Dataset, Dataset]:
    data = Dataset.from_dict({"question": ["11"] * size, "answer": ["1"] * size})
    return data, data


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

    _, values = agent._get_logprobs_and_values(
        stacked_ids,
        batch_size=stacked_ids.shape[0],
        use_reference=False,
        eval_mode=True,
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
    actor_network = build_tiny_actor_network()
    train_dataset, test_dataset = make_dataset(int(dbg["dataset_size"]))

    conversation_template = [
        {"role": "system", "content": "Output one digit."},
        {"role": "user", "content": "{question}"},
        {"role": "assistant", "content": ""},
    ]

    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=constant_reward_factory(reward),
        conversation_template=conversation_template,
        data_batch_size_per_gpu=int(dbg["data_batch_size_per_gpu"]),
        accelerator=None,
        max_context_length=max_ctx,
        return_raw_completions=False,
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
        init_ids, init_masks = agent.get_action(prompts, training=False)
        init_values = get_terminal_values(agent, init_ids, init_masks)

    init_mean = float(init_values.mean().item())
    print(
        f"[value-debug] initial V(terminal): "
        f"mean={init_mean:.4f}, std={init_values.std():.4f}"
    )

    prompts, _ = env.step(init_ids)

    for step in range(num_steps):
        agent.set_reference_policy(env.num_epochs)
        completion_ids, action_masks = agent.get_action(prompts)
        next_prompts, rewards = env.step(completion_ids)

        loss, kl, pg_loss, vf_loss, entropy = agent.learn(
            (completion_ids, action_masks, rewards),
        )
        prompts = next_prompts

        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                snap_ids, snap_masks = agent.get_action(prompts, training=False)
                snap_values = get_terminal_values(agent, snap_ids, snap_masks)
            print(
                f"[value-debug] step {step + 1:4d} | "
                f"vf_loss={vf_loss:.4f} | "
                f"V(terminal) mean={snap_values.mean():.4f}, "
                f"std={snap_values.std():.4f}"
            )

    with torch.no_grad():
        final_ids, final_masks = agent.get_action(prompts, training=False)
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
    main(load_debug_config("value_head_smoke.yaml"))
