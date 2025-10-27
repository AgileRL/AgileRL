import copy
import gc

import gymnasium as gym
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
from datasets import Dataset
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from peft import LoraConfig
from transformers import AutoTokenizer

from agilerl.algorithms.core.base import (
    OptimizerWrapper,
)
from agilerl.algorithms.dpo import DPO
from agilerl.utils.llm_utils import PreferenceGym
from tests.test_algorithms.test_llms.test_grpo import (
    create_module,
    deepspeed_config_stage_1,
    deepspeed_config_stage_2,
)


@pytest.fixture
def preference_dataset_factory():
    def make_preference_gym(
        num_samples: int,
        accelerator: Accelerator | None,
        tokenizer: AutoTokenizer,
        data_batch_size_per_gpu: int = 8,
    ):
        train_dataset = Dataset.from_dict(
            {
                "prompt": [f"Prompt {i}" for i in range(num_samples)],
                "chosen": [f"Chosen {i}" for i in range(num_samples)],
                "rejected": [f"Rejected {i}" for i in range(num_samples)],
            }
        )
        test_dataset = Dataset.from_dict(
            {
                "prompt": [f"Prompt {i}" for i in range(num_samples)],
                "chosen": [f"Chosen {i}" for i in range(num_samples)],
                "rejected": [f"Rejected {i}" for i in range(num_samples)],
            }
        )
        return PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            accelerator=accelerator,
        )

    return make_preference_gym


@pytest.fixture(scope="function")
def dpo_factory():
    def generate_dpo(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    ):
        gc.collect()
        torch.cuda.empty_cache()
        AcceleratorState._reset_state(True)

        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        if not use_deepspeed_optimizer and accelerator is not None:
            accelerator.state.deepspeed_plugin.deepspeed_config.pop("optimizer", None)
        observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
        action_space = gym.spaces.Box(
            low=0,
            high=vocab_size - 1,
            shape=(20,),
        )
        if pretrained_model_name_or_path is not None:
            actor = model_factory(pretrained_model_name_or_path)
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ]
        else:
            actor = create_module(
                input_size=input_size,
                max_tokens=max_tokens,
                vocab_size=vocab_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            target_modules = ["linear_1"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        dpo = DPO(
            observation_space=observation_space,
            action_space=action_space,
            actor_network=actor,
            pad_token_id=vocab_size - 1,
            pad_token="<pad>",
            lora_config=lora_config,
            accelerator=accelerator,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_separate_reference_adapter=use_separate_reference_adapter,
            reduce_memory_peak=reduce_memory_peak,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        )
        return dpo

    return generate_dpo


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (None, False),
        (deepspeed_config_stage_1, True),
        (deepspeed_config_stage_1, False),
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        # None,
    ],
)
@pytest.mark.parametrize("data_batch_size", [4])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_init_dpo(
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    assert isinstance(dpo.observation_space, gym.spaces.Box)
    assert isinstance(dpo.action_space, gym.spaces.Box)
    assert dpo.batch_size_per_process == 16 if not reduce_memory_peak else 1
    assert dpo.beta == 0.001
    assert dpo.lr == 1e-4 if use_deepspeed_optimizer else 1e-5, dpo.lr == 1e-4
    assert dpo.max_grad_norm is None if config is not None else 0.1
    assert dpo.update_epochs == 1
    assert dpo.temperature == 1
    assert dpo.calc_position_embeddings
    assert dpo.device == (
        dpo.accelerator.device
        if torch.cuda.is_available() and dpo.accelerator is not None
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    assert dpo.index == 0
    assert dpo.scores == []
    assert dpo.fitness == []
    assert dpo.steps == [0]
    if config is not None:
        assert isinstance(dpo.actor, DeepSpeedEngine)
        if not use_deepspeed_optimizer:
            assert isinstance(dpo.optimizer, OptimizerWrapper)
            assert isinstance(dpo.optimizer.optimizer, DeepSpeedOptimizerWrapper)
        else:
            assert isinstance(dpo.optimizer, OptimizerWrapper)
            assert isinstance(dpo.optimizer.optimizer, DeepSpeedZeroOptimizer)
            assert isinstance(dpo.actor.optimizer, DeepSpeedZeroOptimizer)
    else:
        assert isinstance(dpo.actor, torch.nn.Module)
    dpo.clean_up()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (None, False),
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    ],
)
@pytest.mark.parametrize("data_batch_size", [4])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_dpo_get_action(
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    with pytest.raises(NotImplementedError):
        dpo.get_action(obs=None)
    dpo.clean_up()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    ],
)
@pytest.mark.parametrize("data_batch_size", [2])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_dpo_learn(
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    train_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(100)],
            "chosen": [
                f"This prompt is better than the rejected prompt {i}"
                for i in range(100)
            ],
            "rejected": [f"Bad response {i}" for i in range(100)],
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(100)],
            "chosen": [
                f"This prompt is better than the rejected prompt {i}"
                for i in range(100)
            ],
            "rejected": [f"Bad response {i}" for i in range(100)],
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    env = PreferenceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=data_batch_size,
        accelerator=dpo.accelerator,
    )
    for name, param in dpo.actor.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and param is not None:
            param.data.normal_(mean=0, std=1.0)
    prompts = env.reset()
    pre_learn_actor_state_dict = copy.deepcopy(dpo.actor.state_dict())
    loss, chosen_reward, rejected_reward = dpo.learn(prompts)
    assert isinstance(loss, float)
    assert isinstance(chosen_reward, float)
    assert isinstance(rejected_reward, float)

    # Check that the actor network is updated
    for (param_name, param), (_, pre_learn_param) in zip(
        dpo.actor.state_dict().items(),
        pre_learn_actor_state_dict.items(),
    ):
        if "actor" in param_name:
            assert not torch.equal(param, pre_learn_param)

        elif "reference" in param_name:
            assert torch.equal(param, pre_learn_param)

        else:
            assert torch.equal(param, pre_learn_param)
    dpo.clean_up()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    ],
)
@pytest.mark.parametrize("data_batch_size", [2])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_dpo_test(
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    train_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(100)],
            "chosen": [
                f"This prompt is better than the rejected prompt {i}"
                for i in range(100)
            ],
            "rejected": [f"Bad response {i}" for i in range(100)],
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(100)],
            "chosen": [
                f"This prompt is better than the rejected prompt {i}"
                for i in range(100)
            ],
            "rejected": [f"Bad response {i}" for i in range(100)],
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    env = PreferenceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=data_batch_size,
        accelerator=dpo.accelerator,
    )
    fitness = dpo.test(env)
    assert isinstance(fitness, float)
    dpo.clean_up()
    AcceleratorState._reset_state(True)
