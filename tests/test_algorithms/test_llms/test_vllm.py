from contextlib import contextmanager

import pytest
import torch

pytest.importorskip("deepspeed", reason="LLM tests require deepspeed.")
pytest.importorskip("vllm", reason="LLM tests require vllm.")

from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import ReasoningGym
from tests import TINY_LLM_FIXTURE_PATH
from tests.test_algorithms.test_llms.test_reinforce_llm import (
    generate_reinforce,
)
from tests.utils import (
    assert_vllm_get_action_contract,
    spawn_new_process_for_each_test,
)

pytestmark = pytest.mark.vllm


@pytest.fixture(scope="function")
def reinforce_factory():
    return generate_reinforce


def _minimal_reasoning_gym(
    device: str, vocab_size: int, input_size: int, batch_size: int
):
    env = ReasoningGym.__new__(ReasoningGym)

    @contextmanager
    def eval_mode():
        yield

    env.eval_mode = eval_mode

    def reset(reset_dataloaders=False):
        del reset_dataloaders
        return {
            "input_ids": torch.randint(
                0, vocab_size, (batch_size, input_size), device=device
            ),
            "attention_mask": torch.ones(batch_size, input_size, device=device),
            "question": [f"q_{i}" for i in range(batch_size)],
            "answer": [f"a_{i}" for i in range(batch_size)],
            "text": ["Solve the task briefly."] * batch_size,
        }

    def step(completion_ids):
        del completion_ids
        rewards = torch.ones(batch_size, device=device)
        return reset(), rewards

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
            sleep_mode=False,  # Sleep mode causes issues with tests so keep False for now
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
                "stitch_prefix_ids": torch.randint(
                    0, vocab_size, (1, 2), device=rf.device
                ),
                "initial_prompt_len": max(1, input_size // 2),
            }
        )

        for training in (True, False):
            completion_ids, action_masks = rf.get_action(prompts, training=training)
            assert_vllm_get_action_contract(
                completion_ids=completion_ids,
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
