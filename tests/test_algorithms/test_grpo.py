import copy
import gc
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import gymnasium as gym
import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.scheduler import AcceleratedScheduler
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    get_peft_model,
    get_peft_model_state_dict,
)
from torch.optim.lr_scheduler import SequentialLR
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms import GRPO
from agilerl.algorithms.core.base import (
    EvolvableAlgorithm,
    LLMAlgorithm,
    OptimizerWrapper,
)
from agilerl.utils.algo_utils import CosineLRScheduleConfig, clone_llm

dist_env = dict(
    ACCELERATE_USE_DEEPSPEED="true",
    MASTER_ADDR="localhost",
    MASTER_PORT="10999",
    RANK="0",
    LOCAL_RANK="0",
    WORLD_SIZE="1",
)

deepspeed_base_config = {
    "fp32": {
        "enabled": True,
    },
    "auto_cast": True,
    "gradient_clipping": 0.5,
    "gradient_accumulation_steps": 1,
}

deepspeed_config_stage_1 = deepspeed_base_config | {
    "zero_optimization": {
        "stage": 1,
    },
}

deepspeed_config_stage_2 = deepspeed_base_config | {
    "zero_optimization": {
        "stage": 2,
    },
}

deepspeed_config_stage_3 = deepspeed_base_config | {
    "zero_optimization": {
        "stage": 3,
    },
}

deepspeed_config_stage_1_with_scheduler = deepspeed_base_config | {
    "zero_optimization": {
        "stage": 1,
    },
    "scheduler": {
        "params": {
            "warmup_max_lr": 0.001,
            "num_epochs": 10,
            "warmup_proportion": 0.05,
        }
    },
}


class DummyConfig(PretrainedConfig):
    def __init__(
        self,
        input_size=16,
        max_tokens=8,
        vocab_size=100,
        intermediate_size=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size


class DummyForwardOutput:
    def __init__(self, logits):
        self.logits = logits


class DummyMLPPreTrainedModel(PreTrainedModel):
    config_class = DummyConfig
    base_model_prefix = "dummy_mlp"

    def __init__(self, config: DummyConfig, device="cpu"):
        super().__init__(config)
        self.input_size = config.input_size
        self.max_tokens = config.max_tokens
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing_enabled = False
        self.datatype = (
            torch.bfloat16
            if deepspeed_base_config.get("bf16", {}).get("enabled", False)
            else (
                torch.float16
                if deepspeed_base_config.get("fp16", {}).get("enabled", False)
                else torch.float32
            )
        )
        self.linear_1 = nn.Linear(
            self.input_size + self.max_tokens, 32, device=device, dtype=self.datatype
        )
        self.linear_2 = nn.Linear(
            32,
            (self.input_size + self.max_tokens) * self.vocab_size,
            device=device,
            dtype=self.datatype,
        )

    @property
    def model(self):
        return self

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        input_ids = input_ids.to(self.datatype)
        output = self.linear_2(self.linear_1(input_ids)).reshape(
            input_ids.shape[0],
            self.input_size + self.max_tokens,
            self.vocab_size,
        )
        return DummyForwardOutput(
            logits=output,
        )

    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError("`input_ids` must be provided for generation.")
        input_shape = input_ids.shape
        group_size = input_shape[0]
        prompt_size = input_shape[1]
        # Simple generation: just return random tokens based on vocab size and desired length
        return torch.randint(
            0, self.vocab_size, (group_size, prompt_size + self.config.max_tokens)
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.gradient_checkpointing_enabled = True


class DummyHuggingFaceEnv:
    def __init__(self, vocab_size, input_size, data_batch_size):
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.data_batch_size = data_batch_size

    def reset(self, reset_dataloaders=False):
        states = [
            {
                "input_ids": torch.randint(0, self.vocab_size, (self.input_size,)),
                "attention_mask": torch.ones(*(self.input_size,)),
            }
            for _ in range(self.data_batch_size)
        ]
        return states

    def step(self, completion_ids):
        states = [
            {
                "input_ids": torch.randint(0, self.vocab_size, (self.input_size,)),
                "attention_mask": torch.ones(*(self.input_size,)),
            }
            for _ in range(self.data_batch_size)
        ]
        return (
            states,
            torch.cat([torch.tensor([1.0]) for _ in range(self.data_batch_size)]),
        )

    @contextmanager
    def eval_mode(self):
        try:
            yield
        finally:
            pass


def create_module(input_size, max_tokens, vocab_size, device):
    return DummyMLPPreTrainedModel(
        config=DummyConfig(
            input_size=input_size, max_tokens=max_tokens, vocab_size=vocab_size
        ),
        device=device,
    )


@pytest.fixture(autouse=True)
def deepspeed_env():
    import os

    existing_vars = {}
    for key, value in dist_env.items():
        key = key.upper()
        if key in os.environ:
            existing_vars[key] = os.environ[key]
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key in dist_env:
            key = key.upper()
            if key in existing_vars:
                # restore previous value
                os.environ[key] = existing_vars[key]
            else:
                os.environ.pop(key, None)
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def accelerator(request):
    gc.collect()
    torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)
    config = request.param.get("config", None)
    use_deepspeed_optimizer = request.param.get("use_deepspeed_optimizer", False)
    if use_deepspeed_optimizer and config is not None:
        config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,  # Smaller learning rate
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        }
    if config is None:
        yield None, False
    else:
        accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config))
        yield accelerator, use_deepspeed_optimizer
        config.pop("optimizer", None)
        accelerator.free_memory()


@pytest.fixture(scope="function")
def grpo(request, accelerator):
    gc.collect()
    torch.cuda.empty_cache()
    accelerator, use_deepspeed_optimizer = accelerator
    vocab_size = request.param.get("vocab_size", 1000)
    input_size = request.param.get("input_size", 10)
    max_tokens = request.param.get("max_tokens", 20)
    group_size = request.param.get("group_size", 5)
    set_reference_policy_adapter = request.param.get(
        "set_reference_policy_adapter", False
    )
    observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
    action_space = gym.spaces.Box(
        low=0,
        high=vocab_size - 1,
        shape=(20,),
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["linear_1"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    grpo = GRPO(
        observation_space,
        action_space,
        actor_network=create_module(
            input_size=input_size,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        lr=1e-5,
        pad_token_id=vocab_size - 1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        group_size=group_size,
        lora_config=lora_config,
        cosine_lr_schedule_config=(
            None
            if accelerator is not None
            else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
        ),
        accelerator=accelerator,
        use_separate_reference_adapter=set_reference_policy_adapter,
    )
    yield grpo
    del grpo


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_init_grpo_with_accelerator(
    grpo,
    accelerator,
    request,
):

    accelerator, use_deepspeed_optimizer = accelerator
    assert isinstance(grpo.observation_space, gym.spaces.Box)
    assert isinstance(grpo.action_space, gym.spaces.Box)
    assert grpo.batch_size == 1
    assert grpo.beta == 0.001
    assert grpo.lr == 1e-5 if not use_deepspeed_optimizer else 1e-4
    assert grpo.clip_coef == 0.2
    assert grpo.max_grad_norm is None
    assert grpo.update_epochs == 1
    assert grpo.group_size == request.node.callspec.params["grpo"]["group_size"]
    assert grpo.temperature == 0.9
    assert grpo.calc_position_embeddings
    assert grpo.device == accelerator.device
    assert grpo.index == 0
    assert grpo.scores == []
    assert grpo.fitness == []
    assert grpo.steps == [0]
    assert grpo.zero_stage == 2
    assert isinstance(grpo.generation_config, GenerationConfig)
    assert isinstance(grpo.actor, DeepSpeedEngine)
    if not use_deepspeed_optimizer:
        if accelerator is None:
            assert isinstance(
                grpo.lr_scheduler, AcceleratedScheduler
            ), grpo.lr_scheduler
            assert isinstance(
                grpo.cosine_lr_schedule_config, CosineLRScheduleConfig
            ), type(grpo.cosine_lr_schedule_config)
        assert isinstance(grpo.optimizer, OptimizerWrapper)
        assert isinstance(grpo.optimizer.optimizer, DeepSpeedOptimizerWrapper)
    else:
        assert isinstance(grpo.optimizer, OptimizerWrapper)
        assert isinstance(grpo.optimizer.optimizer, DeepSpeedZeroOptimizer)
        assert isinstance(grpo.actor.optimizer, DeepSpeedZeroOptimizer)
        assert grpo.lr_scheduler is None
        assert grpo.cosine_lr_schedule_config is None

    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_init_grpo_with_no_accelerator(
    grpo,
    accelerator,
    request,
):
    accelerator, _ = accelerator
    assert isinstance(grpo.observation_space, gym.spaces.Box)
    assert isinstance(grpo.action_space, gym.spaces.Box)
    assert grpo.batch_size == 1
    assert grpo.beta == 0.001
    assert grpo.lr == 1e-5
    assert grpo.clip_coef == 0.2
    assert grpo.max_grad_norm == 0.1
    assert grpo.update_epochs == 1
    assert grpo.group_size == 5
    assert grpo.temperature == 0.9
    assert grpo.calc_position_embeddings
    assert isinstance(grpo.cosine_lr_schedule_config, CosineLRScheduleConfig), type(
        grpo.cosine_lr_schedule_config
    )
    assert grpo.device == "cuda"
    assert grpo.index == 0
    assert grpo.scores == []
    assert grpo.fitness == []
    assert grpo.steps == [0]
    assert isinstance(grpo.generation_config, GenerationConfig)
    assert isinstance(grpo.actor, PeftModelForCausalLM)
    assert isinstance(grpo.optimizer, OptimizerWrapper)
    assert isinstance(grpo.lr_scheduler, SequentialLR), grpo.lr_scheduler
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator",
    [
        {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
        {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
    ],
    indirect=["accelerator"],
)
@pytest.mark.parametrize("set_reference_policy_adapter", [False, True])
def test_init_grpo_zero3_warning(accelerator, request, set_reference_policy_adapter):
    accelerator, use_deepspeed_optimizer = accelerator
    with pytest.warns(UserWarning):
        gc.collect()
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
        action_space = gym.spaces.Box(
            low=0,
            high=vocab_size - 1,
            shape=(20,),
        )
        GRPO(
            observation_space,
            action_space,
            actor_network=create_module(
                input_size=input_size,
                max_tokens=max_tokens,
                vocab_size=vocab_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            lr=0.1,
            pad_token_id=vocab_size - 1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            group_size=group_size,
            lora_config=LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
            cosine_lr_schedule_config=(
                None
                if accelerator is not None
                else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
            ),
            accelerator=accelerator,
            use_separate_reference_adapter=set_reference_policy_adapter,
        )
        gc.collect()
        torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator",
    [
        {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
        {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
    ],
    indirect=["accelerator"],
)
@pytest.mark.parametrize("set_reference_policy_adapter", [False, True])
def test_init_grpo_lr_warning(accelerator, request, set_reference_policy_adapter):
    accelerator, use_deepspeed_optimizer = accelerator
    with pytest.warns(UserWarning):
        gc.collect()
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
        action_space = gym.spaces.Box(
            low=0,
            high=vocab_size - 1,
            shape=(20,),
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        grpo = GRPO(
            observation_space,
            action_space,
            actor_network=create_module(
                input_size=input_size,
                max_tokens=max_tokens,
                vocab_size=vocab_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            lr=0.1,
            pad_token_id=vocab_size - 1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            group_size=group_size,
            lora_config=lora_config,
            cosine_lr_schedule_config=(
                None
                if accelerator is not None
                else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
            ),
            max_grad_norm=0.1,
            accelerator=accelerator,
            use_separate_reference_adapter=set_reference_policy_adapter,
        )
        assert grpo.lr == 1e-4 if use_deepspeed_optimizer else 0.1
        gc.collect()
        torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator",
    [
        {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
        {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
    ],
    indirect=["accelerator"],
)
@pytest.mark.parametrize("set_reference_policy_adapter", [False, True])
def test_init_grpo_max_grad_norm_warning(
    accelerator, request, set_reference_policy_adapter
):
    accelerator, use_deepspeed_optimizer = accelerator
    with pytest.warns(UserWarning):
        gc.collect()
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
        action_space = gym.spaces.Box(
            low=0,
            high=vocab_size - 1,
            shape=(20,),
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        GRPO(
            observation_space,
            action_space,
            actor_network=create_module(
                input_size=input_size,
                max_tokens=max_tokens,
                vocab_size=vocab_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            lr=0.1,
            pad_token_id=vocab_size - 1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            group_size=group_size,
            lora_config=lora_config,
            cosine_lr_schedule_config=(
                None
                if accelerator is not None
                else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
            ),
            max_grad_norm=0.1,
            accelerator=accelerator,
            use_separate_reference_adapter=set_reference_policy_adapter,
        )
        gc.collect()
        torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator",
    [
        {
            "config": deepspeed_config_stage_1_with_scheduler,
            "use_deepspeed_optimizer": True,
        },
    ],
    indirect=["accelerator"],
)
@pytest.mark.parametrize("set_reference_policy_adapter", [False, True])
def test_init_grpo_scheduler_warning(
    accelerator, request, set_reference_policy_adapter
):
    accelerator, use_deepspeed_optimizer = accelerator
    with pytest.warns(UserWarning):
        gc.collect()
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
        action_space = gym.spaces.Box(
            low=0,
            high=vocab_size - 1,
            shape=(20,),
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        GRPO(
            observation_space,
            action_space,
            actor_network=create_module(
                input_size=input_size,
                max_tokens=max_tokens,
                vocab_size=vocab_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            lr=0.1,
            pad_token_id=vocab_size - 1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            group_size=group_size,
            lora_config=lora_config,
            cosine_lr_schedule_config=CosineLRScheduleConfig(
                num_epochs=10, warmup_proportion=0.05
            ),
            max_grad_norm=0.1,
            accelerator=accelerator,
            use_separate_reference_adapter=set_reference_policy_adapter,
        )
        gc.collect()
        torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("data_batch_size", [8])
def test_get_action_grpo(grpo, accelerator, request, training, data_batch_size):
    vocab_size = request.node.callspec.params["grpo"]["vocab_size"]
    input_size = request.node.callspec.params["grpo"]["input_size"]
    max_tokens = request.node.callspec.params["grpo"]["max_tokens"]
    group_size = request.node.callspec.params["grpo"]["group_size"]
    accelerator, use_deepspeed_optimizer = accelerator
    states = [
        {
            "input_ids": torch.randint(0, vocab_size, (input_size,)),
            "attention_mask": torch.ones(*(input_size,)),
        }
        for _ in range(data_batch_size)
    ]
    completion_ids, action_masks = grpo.get_action(states, training)
    group_size = 1 if not training else group_size
    for ids in completion_ids:
        assert ids.shape == (group_size, input_size + max_tokens)
    for masks in action_masks:
        assert masks.shape == (group_size, input_size + max_tokens - 1)
    if grpo.accelerator is None:
        assert not grpo.actor.training
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize(
    "rewards",
    [
        torch.tensor([[2, 4, 6, 8, 20], [3, 6, 9, 12, 15]], dtype=torch.float32),
        torch.tensor([3, 6], dtype=torch.float32),
    ],
)
def test_calculate_advantage(grpo, accelerator, request, rewards):
    calculated_advantage = grpo._calculate_advantage(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.unsqueeze(0)
    mean_rewards = torch.mean(rewards, dim=1).unsqueeze(1)
    std_rewards = torch.std(rewards, dim=1).unsqueeze(1)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
    advantages = advantages.flatten().unsqueeze(1)
    assert torch.equal(advantages, calculated_advantage)
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [8])
def test_calculate_kl_divergence(grpo, accelerator, request, batch_size):
    normal_dist = torch.distributions.normal.Normal(0.0, 1.0)
    reference_log_probs = normal_dist.log_prob(torch.randn(batch_size))
    log_probs = normal_dist.log_prob(torch.randn(batch_size))
    kl = grpo._calculate_kl_divergence(log_probs, reference_log_probs)
    assert torch.all(kl >= 0.0)
    assert isinstance(kl, torch.Tensor)
    assert kl.shape == log_probs.shape
    assert kl.shape == reference_log_probs.shape
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_grpo_loss(grpo, accelerator, request):
    advantages = torch.arange(0, 10).unsqueeze(1)
    normal_dist = torch.distributions.normal.Normal(0.0, 1.0)
    reference_log_probs = normal_dist.log_prob(torch.randn(200)).reshape(10, -1)
    old_log_probs = normal_dist.log_prob(torch.randn(200)).reshape(10, -1)
    log_probs = normal_dist.log_prob(torch.randn(200)).reshape(10, -1)
    mask = torch.ones_like(log_probs)
    mask[:, -3:] = 0
    mask = mask.to(torch.bool)
    loss, kl = grpo._grpo_loss(
        mask, log_probs, old_log_probs, reference_log_probs, advantages
    )
    assert loss != 0
    assert kl != 0
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [2])
def test_grpo_learn(grpo, accelerator, request, batch_size):
    accelerator, use_deepspeed_optimizer = accelerator
    vocab_size = request.node.callspec.params["grpo"]["vocab_size"]
    input_size = request.node.callspec.params["grpo"]["input_size"]
    max_tokens = request.node.callspec.params["grpo"]["max_tokens"]
    group_size = request.node.callspec.params["grpo"]["group_size"]
    completions = [
        torch.randint(
            0, vocab_size, (group_size, input_size + max_tokens), device=grpo.device
        )
        for _ in range(batch_size)
    ]
    action_masks = [
        torch.randint(
            0, 2, (group_size, input_size + max_tokens - 1), device=grpo.device
        )
        for _ in range(batch_size)
    ]
    rewards = torch.stack(
        [
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
            * 0.1  # Small, increasing rewards scaled down
            for _ in range(batch_size)
        ],
        dim=0,
    )

    pre_learn_actor_adapter_state_dict = copy.deepcopy(
        get_peft_model_state_dict(grpo.actor, adapter_name="actor")
    )
    pre_learn_actor_state_dict = copy.deepcopy(grpo.actor.state_dict())
    mean_loss, mean_kl = grpo.learn((completions, action_masks, rewards))
    assert isinstance(mean_loss, float)
    assert isinstance(mean_kl, float)

    # Check that the actor network is updated and the reference actor is not
    for param, pre_learn_param in zip(
        get_peft_model_state_dict(grpo.actor, adapter_name="actor").values(),
        pre_learn_actor_adapter_state_dict.values(),
    ):
        assert not torch.equal(param, pre_learn_param)

    for param_name, pre_param_name in zip(
        grpo.actor.state_dict().keys(),
        pre_learn_actor_state_dict.keys(),
    ):
        if "lora" not in param_name.lower():
            assert torch.equal(
                grpo.actor.state_dict()[param_name],
                pre_learn_actor_state_dict[pre_param_name],
            )
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [8])
def test_get_logprobs(grpo, accelerator, request, batch_size):
    vocab_size = request.node.callspec.params["grpo"]["vocab_size"]
    input_size = request.node.callspec.params["grpo"]["input_size"]
    max_tokens = request.node.callspec.params["grpo"]["max_tokens"]
    batch_size = request.node.callspec.params["batch_size"]

    ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
        grpo.device
    )

    log_probs = grpo._get_logprobs(ids=ids)
    grpo.reduce_memory_peak = True
    log_probs_reduced_mem = grpo._get_logprobs(ids=ids)
    assert log_probs.shape == (ids.shape[0], ids.shape[1] - 1)
    assert log_probs_reduced_mem.shape == (ids.shape[0], ids.shape[1] - 1)
    assert torch.allclose(
        log_probs, log_probs_reduced_mem, atol=1e-3
    ), f"log_probs: {log_probs}, log_probs_reduced_mem: {log_probs_reduced_mem}"

    AcceleratorState._reset_state(True)


# FIXME add in zero 3 and work out why its not working
@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [8])
def test_grpo_value_error_with_nan_loss(grpo, accelerator, request, batch_size):
    vocab_size = request.node.callspec.params["grpo"]["vocab_size"]
    input_size = request.node.callspec.params["grpo"]["input_size"]
    max_tokens = request.node.callspec.params["grpo"]["max_tokens"]
    group_size = request.node.callspec.params["grpo"]["group_size"]
    completions = [
        torch.randint(
            0, vocab_size, (group_size, input_size + max_tokens), device=grpo.device
        )
        for _ in range(batch_size)
    ]
    action_masks = [
        torch.randint(
            0, 2, (group_size, input_size + max_tokens - 1), device=grpo.device
        ).bool()
        for _ in range(batch_size)
    ]
    rewards = torch.stack([torch.ones(group_size) for _ in range(batch_size)], dim=0)

    def mock_grpo_loss(*args, **kwargs):
        return torch.tensor(float("nan")), torch.tensor(1.0)

    with patch.object(grpo, "_grpo_loss", side_effect=mock_grpo_loss), pytest.raises(
        ValueError
    ):
        mean_loss, mean_kl = grpo.learn((completions, action_masks, rewards))


def test_grpo_load():
    with pytest.raises(NotImplementedError):
        GRPO.load("path")


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        # (
        #     {"config": deepspeed_config_stage_3},
        #     {
        #         "vocab_size": 1000,
        #         "input_size": 10,
        #         "max_tokens": 20,
        #         "group_size": 5,
        #         "set_reference_policy_adapter": False,
        #     },
        # ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        # (
        #     {"config": deepspeed_config_stage_3},
        #     {
        #         "vocab_size": 1000,
        #         "input_size": 10,
        #         "max_tokens": 20,
        #         "group_size": 5,
        #         "set_reference_policy_adapter": True,
        #     },
        # ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_grpo_save_load_checkpoint(grpo, accelerator, request, tmpdir):
    with tempfile.TemporaryDirectory() as tmpdir:
        grpo.save_checkpoint(tmpdir)
        vocab_size = request.node.callspec.params["grpo"]["vocab_size"]
        input_size = request.node.callspec.params["grpo"]["input_size"]
        max_tokens = request.node.callspec.params["grpo"]["max_tokens"]
        group_size = request.node.callspec.params["grpo"]["group_size"]
        set_reference_policy_adapter = request.node.callspec.params["grpo"][
            "set_reference_policy_adapter"
        ]
        accelerator, use_deepspeed_optimizer = accelerator
        print("ACCELERATOR BEING USED: ", accelerator)
        new_grpo = GRPO(
            gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,)),
            gym.spaces.Box(low=0, high=vocab_size - 1),
            actor_network=create_module(
                input_size=input_size,
                max_tokens=max_tokens,
                vocab_size=vocab_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            pad_token_id=vocab_size - 1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            group_size=group_size,
            lora_config=LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
            cosine_lr_schedule_config=(
                None
                if accelerator is not None
                else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
            ),
            accelerator=accelerator,
            use_separate_reference_adapter=set_reference_policy_adapter,
        )
        new_grpo.load_checkpoint(tmpdir)

        for attr in EvolvableAlgorithm.inspect_attributes(grpo):
            if not attr.startswith("_") and not attr.startswith("__"):
                if attr == "rng":
                    assert hasattr(new_grpo, attr)
                elif attr == "actor":
                    for param, new_param in zip(
                        grpo.actor.parameters(), new_grpo.actor.parameters()
                    ):
                        assert torch.equal(param, new_param)
                elif attr == "optimizer":
                    for param, new_param in zip(
                        grpo.optimizer.parameters(), new_grpo.optimizer.parameters()
                    ):
                        assert torch.equal(param, new_param)
                elif attr == "accelerator" or attr == "lr_scheduler":
                    assert (
                        getattr(new_grpo, attr).__class__.__name__
                        == getattr(grpo, attr).__class__.__name__
                    )
                elif not isinstance(getattr(grpo, attr), torch.Tensor):
                    assert getattr(new_grpo, attr) == getattr(
                        grpo, attr
                    ), f"Attribute {attr} is not equal"
                else:
                    assert torch.equal(getattr(new_grpo, attr), getattr(grpo, attr))


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [8])
def test_save_load_distributed_actor_no_accelerator(
    grpo, accelerator, request, batch_size, tmpdir
):
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    with pytest.warns(UserWarning):
        grpo._save_distributed_actor(checkpoint_path)

    with pytest.warns(UserWarning):
        grpo._load_distributed_actor(checkpoint_path)
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_grpo_save_load_distributed_actor(grpo, accelerator, request, tmpdir):
    vocab_size = request.node.callspec.params["grpo"]["vocab_size"]
    input_size = request.node.callspec.params["grpo"]["input_size"]
    max_tokens = request.node.callspec.params["grpo"]["max_tokens"]
    group_size = request.node.callspec.params["grpo"]["group_size"]
    set_reference_policy_adapter = request.node.callspec.params["grpo"][
        "set_reference_policy_adapter"
    ]
    accelerator, use_deepspeed_optimizer = accelerator
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    grpo._save_distributed_actor(checkpoint_path)
    grpo_optim_state_dict = (
        grpo.optimizer.state_dict()
        if not use_deepspeed_optimizer
        else grpo.actor.optimizer.state_dict()
    )
    grpo_optim_state_dict.pop("loss_scaler", None)
    new_grpo = GRPO(
        gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,)),
        gym.spaces.Box(low=0, high=vocab_size - 1),
        actor_network=create_module(
            input_size=input_size,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        pad_token_id=vocab_size - 1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        group_size=group_size,
        lora_config=LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        ),
        cosine_lr_schedule_config=(
            None
            if accelerator is not None
            else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
        ),
        accelerator=accelerator,
        use_separate_reference_adapter=set_reference_policy_adapter,
    )
    new_grpo._load_distributed_actor(checkpoint_path)

    if use_deepspeed_optimizer:
        opt = grpo.actor.optimizer
        new_opt = new_grpo.actor.optimizer
    else:
        opt = grpo.optimizer
        new_opt = new_grpo.optimizer

    if not use_deepspeed_optimizer and accelerator is None:
        assert (
            new_opt.optimizer.loss_scaler.cur_scale
            == opt.optimizer.loss_scaler.cur_scale
        )
    assert new_opt.state_dict().keys() == opt.state_dict().keys()

    # Check that the actor network is updated and the reference actor is not
    for param, pre_learn_param in zip(
        new_grpo.actor.parameters(), grpo.actor.parameters()
    ):
        assert torch.equal(param, pre_learn_param)

    for key in new_opt.state_dict().keys():
        if key == "loss_scaler":
            continue
        assert str(new_opt.state_dict()[key]) == str(grpo_optim_state_dict[key])
    AcceleratorState._reset_state(True)


# NOTE cloning does not work with deepspeed v3
@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        # (
        #     {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
        #     {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5, "set_reference_policy_adapter": False},
        # ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        # (
        #     {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
        #     {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5, "set_reference_policy_adapter": False},
        # ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        # (
        #     {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
        #     {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5, "set_reference_policy_adapter": True},
        # ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        # (
        #     {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
        #     {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5, "set_reference_policy_adapter": True},
        # ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_grpo_clone_with_accelerator(grpo, accelerator, request, tmpdir):
    accelerator, use_deepspeed_optimizer = accelerator
    grpo_accelerator = grpo.accelerator
    grpo_lr_scheduler = grpo.lr_scheduler
    grpo.fitness = [1, 2, 3]
    original_actor_state_dict = (
        grpo.actor.state_dict()
        if accelerator is None
        else accelerator.unwrap_model(grpo.actor).state_dict()
    )
    new_grpo = grpo.clone(index=1)

    # Check that the actor network is updated and the reference actor is not
    for (name, cloned_param), param in zip(
        new_grpo.actor.state_dict().items(),
        original_actor_state_dict.values(),
    ):
        assert torch.equal(cloned_param, param)

    assert new_grpo.index == 1
    if grpo.accelerator is not None:
        assert new_grpo.accelerator != grpo_accelerator
    if grpo.lr_scheduler is not None:
        assert new_grpo.lr_scheduler != grpo_lr_scheduler

    if use_deepspeed_optimizer:
        opt = grpo.actor.optimizer
        new_opt = new_grpo.actor.optimizer
    else:
        opt = grpo.optimizer
        new_opt = new_grpo.optimizer

    for pg1, pg2 in zip(
        opt.param_groups,
        new_opt.param_groups,
    ):
        assert pg1["lr"] == pg2["lr"]
        assert pg1["weight_decay"] == pg2["weight_decay"]
        assert pg1["betas"] == pg2["betas"]
        assert pg1["eps"] == pg2["eps"]

    assert new_grpo.lr == grpo.lr
    assert new_grpo.batch_size == grpo.batch_size
    assert new_grpo.clip_coef == grpo.clip_coef
    assert new_grpo.update_epochs == grpo.update_epochs
    assert new_grpo.group_size == grpo.group_size
    assert new_grpo.beta == grpo.beta
    assert new_grpo.pad_token_id == grpo.pad_token_id
    assert new_grpo.calc_position_embeddings == grpo.calc_position_embeddings
    assert new_grpo.generation_config == grpo.generation_config
    assert new_grpo.cosine_lr_schedule_config == grpo.cosine_lr_schedule_config
    assert new_grpo.wrap == grpo.wrap
    assert new_grpo.device == grpo.device
    assert new_grpo.fitness == grpo.fitness
    AcceleratorState._reset_state(True)
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [8])
def test_grpo_test(grpo, accelerator, request, batch_size):
    vocab_size = request.node.callspec.params["grpo"]["vocab_size"]
    input_size = request.node.callspec.params["grpo"]["input_size"]
    env = DummyHuggingFaceEnv(vocab_size, input_size, batch_size)
    fitnesses = grpo.test(env)
    assert isinstance(fitnesses, torch.Tensor)
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
def test_clone_llm_peft(vocab_size, input_size, max_tokens):
    # Create a dummy config
    config = DummyConfig(
        input_size=input_size,
        max_tokens=max_tokens,
        vocab_size=vocab_size,
        intermediate_size=128,
    )

    # Create the base model
    base_model = DummyMLPPreTrainedModel(config)

    # Create PEFT config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["linear_1", "linear_2"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Create PEFT model
    peft_model = get_peft_model(base_model, peft_config)

    # Clone the PEFT model
    cloned_model = clone_llm(peft_model, peft_model.state_dict())

    # Verify the cloned model is a PEFT model
    assert isinstance(cloned_model, type(peft_model))

    # Verify the configurations match
    assert cloned_model.config == peft_model.config
    assert cloned_model.peft_config == peft_model.peft_config

    # Verify the parameters match
    for (name1, param1), (name2, param2) in zip(
        cloned_model.named_parameters(), peft_model.named_parameters()
    ):
        assert name1 == name2
        assert torch.equal(param1, param2)

    # Verify the model structure
    assert isinstance(cloned_model.model, type(base_model))

    # Verify the PEFT adapter is properly cloned
    assert cloned_model.active_adapter == peft_model.active_adapter
    assert cloned_model.peft_config[cloned_model.active_adapter] == peft_config


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [8])
def test_grpo_clean_up(grpo, accelerator, request, batch_size):
    grpo.clean_up()
    assert grpo.actor is None
    assert grpo.optimizer is None
    assert grpo.lr_scheduler is None
    del grpo
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        )
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [8])
def test_grpo_preprocess_observation(grpo, accelerator, request, batch_size):
    obs = grpo.preprocess_observation(
        orig_obs := torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    )
    assert torch.equal(obs, orig_obs)
    del grpo
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        )
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [8])
def test_load_distributed_actor_warning(grpo, accelerator, request, batch_size):
    accelerator = MagicMock(spec=Accelerator)
    accelerator.state = MagicMock(spec=AcceleratorState)
    grpo.accelerator = accelerator
    with pytest.raises(ValueError):
        grpo._load_distributed_actor(None)
    del grpo
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "accelerator",
    [
        {"config": None},
    ],
    indirect=["accelerator"],
)
def test_init_grpo_lora_config_warning(accelerator, request):
    accelerator, use_deepspeed_optimizer = accelerator
    with pytest.warns(UserWarning):
        gc.collect()
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
        action_space = gym.spaces.Box(
            low=0,
            high=vocab_size - 1,
            shape=(20,),
        )
        with pytest.raises(ValueError):
            GRPO(
                observation_space,
                action_space,
                actor_network=create_module(
                    input_size=input_size,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                lr=0.1,
                pad_token_id=vocab_size - 1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=(
                    None
                    if accelerator is not None
                    else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
                ),
                accelerator=accelerator,
            )
        gc.collect()
        torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator",
    [
        {"config": None},
    ],
    indirect=["accelerator"],
)
def test_init_grpo_multiple_adapters(accelerator, request):
    """Test GRPO initialization with a PEFT model containing multiple adapters."""
    accelerator, use_deepspeed_optimizer = accelerator
    with pytest.warns(
        UserWarning, match="AgileRL RL finetuning is only compatible with one adapter."
    ):

        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        # Set up test parameters
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5

        # Create spaces
        observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
        action_space = gym.spaces.Box(
            low=0,
            high=vocab_size - 1,
            shape=(20,),
        )

        # Create base model
        base_model = create_module(
            input_size=input_size,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Create first adapter
        lora_config_1 = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        peft_model = get_peft_model(base_model, lora_config_1, adapter_name="adapter1")

        # Add second adapter
        lora_config_2 = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        peft_model.add_adapter(adapter_name="adapter2", peft_config=lora_config_2)

        # Initialize GRPO with the multi-adapter model
        grpo = GRPO(
            observation_space,
            action_space,
            actor_network=peft_model,
            lr=0.1,
            pad_token_id=vocab_size - 1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            group_size=group_size,
            cosine_lr_schedule_config=(
                None
                if accelerator is not None
                else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
            ),
            accelerator=accelerator,
            clone=False,
        )

        # Verify that only the first adapter is used
        assert len(grpo.actor.peft_config) == 1
        assert "actor" in grpo.actor.peft_config
        assert (
            grpo.actor.peft_config["actor"].r == lora_config_1.r
        )  # Check that first adapter's config is used

        # Test that the model still functions
        test_input = torch.randint(0, vocab_size - 1, (1, input_size))
        test_attention_mask = torch.ones_like(test_input)
        test_state = {"input_ids": test_input, "attention_mask": test_attention_mask}

        # Test get_action
        completion_ids, action_masks = grpo.get_action([test_state], training=True)
        assert isinstance(completion_ids, list)
        assert isinstance(action_masks, list)
        assert len(completion_ids) == 1
        assert len(action_masks) == 1

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
        AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {
                "config": deepspeed_config_stage_1_with_scheduler,
                "use_deepspeed_optimizer": True,
            },
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {
                "config": deepspeed_config_stage_1_with_scheduler,
                "use_deepspeed_optimizer": True,
            },
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_update_lr(grpo, accelerator, request):
    accelerator, use_deepspeed_optimizer = accelerator
    opt = (
        grpo.optimizer.optimizer
        if not use_deepspeed_optimizer
        else grpo.actor.optimizer
    )
    grpo.accelerator, grpo.lr_scheduler = LLMAlgorithm.update_lr(
        opt, 0.5, grpo.accelerator, grpo.cosine_lr_schedule_config
    )
    for param_group in opt.param_groups:
        assert param_group["lr"] == 0.5

    if use_deepspeed_optimizer:
        grpo.accelerator.deepspeed_plugin.deepspeed_config["optimizer"]["params"][
            "lr"
        ] = 0.5

        if (
            grpo.accelerator.deepspeed_plugin.deepspeed_config.get("scheduler", None)
            is not None
        ):
            grpo.accelerator.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "warmup_max_lr"
            ] = 0.5
            grpo.accelerator.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "num_epochs"
            ] = 10
            grpo.accelerator.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "warmup_proportion"
            ] = 0.05


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": False,
            },
        ),
        (
            {"config": None},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
        (
            {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
            {
                "vocab_size": 1000,
                "input_size": 10,
                "max_tokens": 20,
                "group_size": 5,
                "set_reference_policy_adapter": True,
            },
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_set_reference_policy(grpo, accelerator, request):
    input_size = request.node.callspec.params["grpo"]["input_size"]
    max_tokens = request.node.callspec.params["grpo"]["max_tokens"]
    reference_update_tracker = 0
    grpo.set_reference_policy(reference_update_tracker)
    input_ids = torch.tensor([[i + 1 for i in range(input_size + max_tokens)]]).to(
        grpo.device
    )
    action_masks = torch.tensor([[1 for _ in range(input_size + max_tokens)]]).to(
        grpo.device
    )
    output_before = grpo.actor(
        **{
            "input_ids": input_ids,
            "attention_mask": action_masks,
        }
    ).logits
    assert grpo.reference_update_tracker == reference_update_tracker
    reference_update_tracker += 1
    grpo.set_reference_policy(reference_update_tracker)

    output_after = grpo.actor(
        **{
            "input_ids": input_ids,
            "attention_mask": action_masks,
        }
    ).logits
    assert torch.allclose(output_before, output_after)
    assert grpo.reference_update_tracker == reference_update_tracker


@pytest.mark.parametrize(
    "accelerator",
    [
        {"config": None},
        {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
        {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
        # {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
        {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
        {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
        # {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
    ],
    indirect=["accelerator"],
)  # Test that ref actor is the same as actor after learning - i.e. its frozen
def test_grpo_ref_actor_is_same_as_actor_after_learning_reference_adapater(
    accelerator, request
):
    accelerator, use_deepspeed_optimizer = accelerator
    gc.collect()
    vocab_size = 1000
    input_size = 10
    max_tokens = 20
    group_size = 5
    observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
    action_space = gym.spaces.Box(
        low=0,
        high=vocab_size - 1,
        shape=(20,),
    )
    grpo = GRPO(
        observation_space,
        action_space,
        actor_network=create_module(
            input_size=input_size,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        lr=0.1,
        pad_token_id=vocab_size - 1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        group_size=group_size,
        lora_config=LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        ),
        cosine_lr_schedule_config=(
            None
            if accelerator is not None
            else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
        ),
        accelerator=accelerator,
        use_separate_reference_adapter=True,
    )

    # Ensure adapters have different params
    grpo.actor.set_adapter("actor")
    for name, param in grpo.actor.named_parameters():
        if "actor" in name:
            param.data *= 2
    assert not check_ref_adapater_is_same_as_actor_after_learning(grpo)
    grpo.set_reference_policy(reference_update_tracker=1)
    assert check_ref_adapater_is_same_as_actor_after_learning(grpo)
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator",
    [
        {"config": None},
        {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": False},
        {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": False},
        # {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": False},
        {"config": deepspeed_config_stage_1, "use_deepspeed_optimizer": True},
        {"config": deepspeed_config_stage_2, "use_deepspeed_optimizer": True},
        # {"config": deepspeed_config_stage_3, "use_deepspeed_optimizer": True},
    ],
    indirect=["accelerator"],
)
def test_grpo_set_reference_policy_with_wrong_adapter_name(accelerator, request):
    accelerator, use_deepspeed_optimizer = accelerator
    with pytest.raises(ValueError):
        gc.collect()
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
        action_space = gym.spaces.Box(
            low=0,
            high=vocab_size - 1,
            shape=(20,),
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        grpo = GRPO(
            observation_space,
            action_space,
            actor_network=create_module(
                input_size=input_size,
                max_tokens=max_tokens,
                vocab_size=vocab_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            lr=0.1,
            pad_token_id=vocab_size - 1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            group_size=group_size,
            lora_config=lora_config,
            cosine_lr_schedule_config=(
                None
                if accelerator is not None
                else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
            ),
            accelerator=accelerator,
            use_separate_reference_adapter=True,
        )
        grpo.actor.add_adapter("wrong_adapter", peft_config=lora_config)
        grpo.set_reference_policy(reference_update_tracker=1)
    AcceleratorState._reset_state(True)


def check_ref_adapater_is_same_as_actor_after_learning(grpo):
    ref_param = None
    actor_param = None
    for name, param in grpo.actor.named_parameters():
        if "lora" in name:
            if "reference" in name:
                ref_param = param
            elif "actor" in name:
                actor_param = param
            else:
                pass
        if ref_param is not None and actor_param is not None:
            if not torch.equal(ref_param, actor_param):
                return False
            ref_param = None
            actor_param = None
    return True
