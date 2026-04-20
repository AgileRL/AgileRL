# """Smoke tests for ``LLMPPO.test`` with ``TokenObservationWrapper`` (no GPU conftest)."""

# from __future__ import annotations

# import pytest
# import torch
# from peft import LoraConfig

# from agilerl.algorithms.ppo_llm import PPO as LLMPPO
# from agilerl.utils.probe_envs_llm import ConstantTargetEnv
# from agilerl.wrappers.gem_wrappers import TokenObservationWrapper
# from benchmarking.tiny_model import TinyDigitTokenizer, build_tiny_actor_network


# def _tiny_ppo() -> LLMPPO:
#     tokenizer = TinyDigitTokenizer()
#     return LLMPPO(
#         model_name=None,
#         actor_network=build_tiny_actor_network(),
#         lora_config=LoraConfig(
#             r=4,
#             lora_alpha=16,
#             target_modules=["c_attn", "c_proj", "c_fc"],
#             bias="none",
#             task_type="CAUSAL_LM",
#         ),
#         micro_batch_size_per_gpu=2,
#         use_vllm=False,
#         pad_token_id=tokenizer.pad_token_id,
#         pad_token=tokenizer.pad_token,
#         use_separate_reference_adapter=True,
#         batch_size=4,
#         beta=0.05,
#         lr_actor=1e-4,
#         lr_critic=1e-3,
#         clip_coef=0.2,
#         max_grad_norm=1.0,
#         update_epochs=2,
#         temperature=0.9,
#         max_output_tokens=8,
#         max_model_len=128,
#         accelerator=None,
#         vf_coef=0.5,
#         gamma=0.99,
#         gae_lambda=0.95,
#         seed=0,
#         gradient_checkpointing=False,
#     )


# def test_ppo_test_token_wrapper_concatenates_multi_episode_rewards() -> None:
#     """``loop`` outer iterations each run a full episode; rewards are concatenated."""
#     torch.manual_seed(0)
#     tokenizer = TinyDigitTokenizer()
#     env = TokenObservationWrapper(
#         ConstantTargetEnv(target_digit="3", prompt="11"),
#         tokenizer,
#         max_turns=1,
#         pad_id=tokenizer.pad_token_id,
#         apply_chat_template=False,
#         max_model_len=128,
#         max_output_tokens=8,
#     )
#     agent = _tiny_ppo()
#     out = agent.test(env, loop=3)
#     assert out.shape == (3,)
#     assert out.dtype == torch.float32


# def test_ppo_test_unknown_env_raises() -> None:
#     class _NotSupportedEnv:
#         pass

#     agent = _tiny_ppo()
#     with pytest.raises(TypeError, match="ReasoningGym"):
#         agent.test(_NotSupportedEnv(), loop=1)


# FIXME delete me!!!!!
