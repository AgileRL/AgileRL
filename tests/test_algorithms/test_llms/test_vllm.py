# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import pytest
import torch

pytest.importorskip("deepspeed", reason="LLM tests require deepspeed.")
pytest.importorskip("vllm", reason="LLM tests require vllm.")

from agilerl.llm_envs import RolloutHarness
from tests import TINY_LLM_FIXTURE_PATH
from tests.test_algorithms.test_llms.test_reinforce_llm import (
    generate_reinforce,
)
from tests.utils import (
    assert_vllm_get_action_contract,
    spawn_new_process_for_each_test,
)

pytestmark = pytest.mark.vllm


@pytest.fixture
def reinforce_factory():
    return generate_reinforce


def _minimal_reasoning_gym(
    device: str, vocab_size: int, input_size: int, batch_size: int
):
    del batch_size  # single-turn rollout env: test() steps one prompt at a time
    env = RolloutHarness.__new__(RolloutHarness)
    env.max_turns = 1
    env.done = False

    @contextmanager
    def eval_mode():
        yield

    env.eval_mode = eval_mode

    def _prompt():
        return {
            "input_ids": torch.randint(0, vocab_size, (1, input_size), device=device),
            "attention_mask": torch.ones(1, input_size, device=device),
            "text": "Solve the task briefly.",
        }

    def reset(seed=None, *, row_index=None):
        del seed, row_index
        env.done = False
        return _prompt(), {}

    def step(token_ids):
        del token_ids
        env.done = True
        return _prompt(), 1.0, True, False, {}

    env.reset = reset
    env.step = step
    return env


class TestREINFORCETest:
    @spawn_new_process_for_each_test
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("pretrained_model_name_or_path", [TINY_LLM_FIXTURE_PATH])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_vllm_methods(
        self,
        deepspeed_env,
        reinforce_factory,
        accelerator_factory,
        model_factory,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        del deepspeed_env
        rf = reinforce_factory(
            accelerator_factory=accelerator_factory,
            model_factory=model_factory,
            config=None,
            use_deepspeed_optimizer=False,
            vocab_size=vocab_size,
            input_size=input_size,
            max_tokens=max_tokens,
            use_vllm=True,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            # Keep the always-awake path covered here; the sleep/wake cycle is
            # exercised end-to-end by test_quantized_generate_survives_sleep_wake.
            sleep_mode=False,
        )

        assert rf.use_vllm
        assert rf.llm is not None
        assert not rf.vllm_config.sleep_mode

        batch_size = 4
        prompts = [
            {
                "input_ids": torch.randint(
                    0, vocab_size, (1, input_size), device=rf.device
                ),
                "attention_mask": torch.ones(1, input_size, device=rf.device),
                "text": "Write me a short story about a cat.",
            }
            for _ in range(batch_size - 1)
        ]
        prompts.append(
            {
                "input_ids": torch.randint(
                    0, vocab_size, (1, input_size), device=rf.device
                ),
                "attention_mask": torch.ones(1, input_size, device=rf.device),
                "text": "Continue the answer.",
            }
        )

        for training in (True, False):
            token_ids, action_masks, _ = rf.get_action(prompts, training=training)
            assert_vllm_get_action_contract(
                token_ids=token_ids,
                action_masks=action_masks,
                batch_size=batch_size,
                prompt_len=input_size,
                pad_token_id=rf.pad_token_id,
            )

        env = _minimal_reasoning_gym(
            device=rf.device,
            vocab_size=vocab_size,
            input_size=input_size,
            batch_size=2,
        )
        out = rf.test(env, loop=1)
        assert out.shape == ()

        rf.clean_up()

    @spawn_new_process_for_each_test
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("pretrained_model_name_or_path", [TINY_LLM_FIXTURE_PATH])
    def test_quantized_generate_survives_sleep_wake(
        self,
        deepspeed_env,
        reinforce_factory,
        accelerator_factory,
        model_factory,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
    ):
        """End-to-end bnb quantization: load → quantize → generate → sleep →
        wake → generate.

        vLLM loads the fixture with in-flight bitsandbytes quantization and the
        trainer holds its own base (loaded from the model name). vLLM native
        sleep/wake must round-trip the quantized base losslessly: greedy decode
        of the same prompts has to be identical before sleep and after wake. An
        earlier sleep path re-quantized bnb weights to garbage on reload — token
        equality is the regression check for that.
        """
        bnb = pytest.importorskip(
            "bitsandbytes",
            reason="quantized vLLM test requires bitsandbytes (linux-only).",
        )
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            # vLLM refuses bf16 below sm_80 (e.g. the T4 CI runners); torch's
            # default check includes emulated bf16 and stays True there.
            pytest.skip("bnb nf4 preset uses bf16 compute; GPU lacks native bf16.")
        from agilerl.utils.llm_utils import build_bnb_quantization_config

        del deepspeed_env
        rf = reinforce_factory(
            accelerator_factory=accelerator_factory,
            model_factory=model_factory,
            config=None,
            use_deepspeed_optimizer=False,
            vocab_size=vocab_size,
            input_size=input_size,
            max_tokens=max_tokens,
            use_vllm=True,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            micro_batch_size_per_gpu=None,
            sleep_mode=True,
            quantization_config=build_bnb_quantization_config("nf4"),
            # The fixture checkpoint is fp16; pin vLLM to bf16 so the engine's
            # quantization target matches the trainer's nf4 preset (bf16
            # compute/storage) and the shared base is dtype-consistent.
            vllm_config_overrides={
                "quantization": "bitsandbytes",
                "dtype": "bfloat16",
            },
            from_name=True,
            temperature=0.0,  # greedy → outputs comparable across sleep/wake
        )

        # The trainer holds its own 4-bit bnb base. NB: the actor is a
        # DummyEvolvable, whose ``modules()`` is the EvolvableModule registry
        # API, not torch's recursive walk — use ``named_modules()``.
        assert any(
            isinstance(m, bnb.nn.Linear4bit) for _, m in rf.actor.named_modules()
        )
        # Sleep mode is active and the engine was put to sleep after init.
        assert not rf._vllm_awake

        # Build prompts once and reuse them verbatim for both generations.
        batch_size = 2
        prompts = [
            {
                "input_ids": torch.randint(
                    0, vocab_size, (1, input_size), device=rf.device
                ),
                "attention_mask": torch.ones(1, input_size, device=rf.device),
                "text": "Write me a short story about a cat.",
            }
            for _ in range(batch_size)
        ]

        first_ids, first_masks, _ = rf.get_action(prompts, training=True)
        assert rf._vllm_awake
        assert_vllm_get_action_contract(
            token_ids=first_ids,
            action_masks=first_masks,
            batch_size=batch_size,
            prompt_len=input_size,
            pad_token_id=rf.pad_token_id,
        )

        # The algorithm's own pre-learn hook: native sleep frees the KV cache
        # and backs the base up to host RAM; wake must restore it losslessly.
        rf._prepare_vllm_for_training()
        assert not rf._vllm_awake

        second_ids, _, _ = rf.get_action(prompts, training=True)
        assert rf._vllm_awake
        for first, second in zip(first_ids, second_ids, strict=True):
            assert torch.equal(first, second), (
                "greedy completions changed across sleep/wake — the quantized "
                "base did not survive native sleep/wake"
            )

        rf.clean_up()

    @spawn_new_process_for_each_test
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("pretrained_model_name_or_path", [TINY_LLM_FIXTURE_PATH])
    def test_dense_generate_survives_sleep_wake(
        self,
        deepspeed_env,
        reinforce_factory,
        accelerator_factory,
        model_factory,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
    ):
        """Dense (unquantized) counterpart of the quantized sleep/wake test.

        vLLM native sleep/wake must round-trip the fp16 base losslessly: greedy
        decode of the same prompts has to be identical before sleep and after
        wake. fp16 (not bf16) so the test also runs on pre-Ampere CI GPUs.
        """
        del deepspeed_env
        rf = reinforce_factory(
            accelerator_factory=accelerator_factory,
            model_factory=model_factory,
            config=None,
            use_deepspeed_optimizer=False,
            vocab_size=vocab_size,
            input_size=input_size,
            max_tokens=max_tokens,
            use_vllm=True,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            micro_batch_size_per_gpu=None,
            sleep_mode=True,
            vllm_config_overrides={"dtype": "float16"},
            from_name=True,
            temperature=0.0,  # greedy → outputs comparable across sleep/wake
        )

        # Sleep mode is active and the engine was put to sleep after init.
        assert not rf._vllm_awake

        batch_size = 2
        prompts = [
            {
                "input_ids": torch.randint(
                    0, vocab_size, (1, input_size), device=rf.device
                ),
                "attention_mask": torch.ones(1, input_size, device=rf.device),
                "text": "Write me a short story about a cat.",
            }
            for _ in range(batch_size)
        ]

        first_ids, first_masks, _ = rf.get_action(prompts, training=True)
        assert rf._vllm_awake
        assert_vllm_get_action_contract(
            token_ids=first_ids,
            action_masks=first_masks,
            batch_size=batch_size,
            prompt_len=input_size,
            pad_token_id=rf.pad_token_id,
        )

        # Native sleep frees the KV cache and backs the base up to host RAM;
        # wake must restore it losslessly.
        rf._prepare_vllm_for_training()
        assert not rf._vllm_awake

        second_ids, _, _ = rf.get_action(prompts, training=True)
        assert rf._vllm_awake
        for first, second in zip(first_ids, second_ids, strict=True):
            assert torch.equal(first, second), (
                "greedy completions changed across sleep/wake — the dense "
                "base did not survive native sleep/wake"
            )

        rf.clean_up()
