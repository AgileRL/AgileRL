import copy
import gc
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple
from unittest import mock
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
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import SequentialLR
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel

from agilerl.algorithms import GRPO
from agilerl.algorithms.core.base import OptimizerWrapper
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
    "fp32": {"enabled": True},
    "gradient_clipping": 1.5,
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
    def eval(self):
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


@pytest.fixture
def accelerator(request):
    gc.collect()
    torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)
    config = request.param.get("config", None)
    if config is None:
        yield None
    else:
        accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config))
        yield accelerator
        accelerator.free_memory()


@pytest.fixture
def grpo(request, accelerator, monkeypatch):
    gc.collect()
    torch.cuda.empty_cache()
    with mock.patch.dict(os.environ, clear=True):
        env_vars = {
            "ACCELERATE_USE_DEEPSPEED": "true",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "10999",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
        vocab_size = request.param.get("vocab_size", 1000)
        input_size = request.param.get("input_size", 10)
        max_tokens = request.param.get("max_tokens", 20)
        group_size = request.param.get("group_size", 5)
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
            cosine_lr_schedule_config=CosineLRScheduleConfig(
                num_epochs=10, warmup_proportion=0.05
            ),
            accelerator=accelerator,
        )
        yield grpo


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_init_grpo_with_accelerator(
    grpo,
    accelerator,
    request,
):
    assert isinstance(grpo.observation_space, gym.spaces.Box)
    assert isinstance(grpo.action_space, gym.spaces.Box)
    assert grpo.batch_size == 1
    assert grpo.beta == 0.001
    assert grpo.lr == 0.1
    assert grpo.clip_coef == 0.2
    assert grpo.max_grad_norm is None
    assert grpo.update_epochs == 1
    assert grpo.group_size == request.node.callspec.params["grpo"]["group_size"]
    assert grpo.temperature == 0.9
    assert grpo.calc_position_embeddings
    assert isinstance(grpo.cosine_lr_schedule_config, CosineLRScheduleConfig), type(
        grpo.cosine_lr_schedule_config
    )
    assert grpo.device == accelerator.device
    assert grpo.index == 0
    assert grpo.scores == []
    assert grpo.fitness == []
    assert grpo.steps == [0]
    assert grpo.zero_stage == 2
    assert isinstance(grpo.generation_config, GenerationConfig)
    assert isinstance(grpo.actor, DeepSpeedEngine)
    assert isinstance(grpo.optimizer, DeepSpeedOptimizerWrapper)
    assert isinstance(grpo.lr_scheduler, AcceleratedScheduler), grpo.lr_scheduler
    assert not isinstance(grpo.reference_actor, DummyMLPPreTrainedModel)
    for ref_param, param in zip(
        grpo.reference_actor.parameters(), grpo.actor.parameters()
    ):
        assert torch.equal(ref_param, param)
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_init_grpo_with_no_accelerator(
    grpo,
    accelerator,
    request,
):
    assert isinstance(grpo.observation_space, gym.spaces.Box)
    assert isinstance(grpo.action_space, gym.spaces.Box)
    assert grpo.batch_size == 1
    assert grpo.beta == 0.001
    assert grpo.lr == 0.1
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
    assert isinstance(grpo.actor, DummyMLPPreTrainedModel)
    assert isinstance(grpo.optimizer, OptimizerWrapper)
    assert isinstance(grpo.lr_scheduler, SequentialLR), grpo.lr_scheduler
    for ref_param, param in zip(
        grpo.reference_actor.parameters(), grpo.actor.parameters()
    ):
        assert torch.equal(ref_param, param)
    assert not grpo.reference_actor.gradient_checkpointing_enabled
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator",
    [
        {"config": deepspeed_config_stage_3},
    ],
    indirect=["accelerator"],
)
def test_init_grpo_zero3_warning(monkeypatch, accelerator, request):
    with pytest.warns(UserWarning), mock.patch.dict(os.environ, clear=True):
        env_vars = {
            "ACCELERATE_USE_DEEPSPEED": "true",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "10999",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
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
            cosine_lr_schedule_config=CosineLRScheduleConfig(
                num_epochs=10, warmup_proportion=0.05
            ),
            accelerator=accelerator,
        )
        gc.collect()
        torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator",
    [
        {"config": deepspeed_config_stage_2},
    ],
    indirect=["accelerator"],
)
def test_init_grpo_max_grad_norm_warning(monkeypatch, accelerator, request):
    with pytest.warns(UserWarning), mock.patch.dict(os.environ, clear=True):
        env_vars = {
            "ACCELERATE_USE_DEEPSPEED": "true",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "10999",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
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
            cosine_lr_schedule_config=CosineLRScheduleConfig(
                num_epochs=10, warmup_proportion=0.05
            ),
            max_grad_norm=0.1,
            accelerator=accelerator,
        )
        gc.collect()
        torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [2])
def test_grpo_learn(grpo, accelerator, request, batch_size):
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
            torch.randint(0, 10, (group_size,), dtype=torch.float32)
            for _ in range(batch_size)
        ],
        dim=0,
    )

    pre_learn_actor_state_dict = copy.deepcopy(grpo.actor.state_dict())
    pre_learn_reference_actor_state_dict = copy.deepcopy(
        grpo.reference_actor.state_dict()
    )

    mean_loss, mean_kl = grpo.learn((completions, action_masks, rewards))
    assert isinstance(mean_loss, float)
    assert isinstance(mean_kl, float)

    # Check that the actor network is updated and the reference actor is not
    for param, pre_learn_param in zip(
        grpo.actor.state_dict().values(), pre_learn_actor_state_dict.values()
    ):
        assert not torch.equal(param, pre_learn_param)
    for param, pre_learn_param in zip(
        grpo.reference_actor.state_dict().values(),
        pre_learn_reference_actor_state_dict.values(),
    ):
        assert torch.equal(param, pre_learn_param)
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
    assert torch.allclose(log_probs, log_probs_reduced_mem, atol=1e-3)
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_grpo_load_checkpoint(grpo, accelerator, request):
    with pytest.raises(NotImplementedError):
        grpo.load_checkpoint("path")
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_grpo_save_checkpoint(grpo, accelerator, request):
    with pytest.raises(NotImplementedError):
        grpo.save_checkpoint("path")
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_grpo_save_load_checkpoint(grpo, accelerator, request, tmpdir):
    vocab_size = request.node.callspec.params["grpo"]["vocab_size"]
    input_size = request.node.callspec.params["grpo"]["input_size"]
    max_tokens = request.node.callspec.params["grpo"]["max_tokens"]
    group_size = request.node.callspec.params["grpo"]["group_size"]
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    grpo._save_distributed_actor(checkpoint_path)
    grpo_optimizer = grpo.optimizer
    grpo_optim_state_dict = grpo.optimizer.state_dict()
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
        cosine_lr_schedule_config=CosineLRScheduleConfig(
            num_epochs=10, warmup_proportion=0.05
        ),
        accelerator=accelerator,
    )
    new_grpo._load_distributed_actor(checkpoint_path)

    assert (
        new_grpo.optimizer.optimizer.loss_scaler.cur_scale
        == grpo_optimizer.optimizer.loss_scaler.cur_scale
    )
    assert new_grpo.optimizer.state_dict().keys() == grpo_optimizer.state_dict().keys()

    # Check that the actor network is updated and the reference actor is not
    for param, pre_learn_param in zip(
        new_grpo.actor.parameters(), grpo.actor.parameters()
    ):
        assert torch.equal(param, pre_learn_param)

    for key in new_grpo.optimizer.state_dict().keys():
        if key == "loss_scaler":
            continue
        assert str(new_grpo.optimizer.state_dict()[key]) == str(
            grpo_optim_state_dict[key]
        )
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "accelerator, grpo",
    [
        (
            {"config": None},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
    ],
    indirect=["accelerator", "grpo"],
)
def test_grpo_clone_with_accelerator(grpo, accelerator, request, tmpdir):
    grpo_accelerator = grpo.accelerator
    grpo_lr_scheduler = grpo.lr_scheduler
    grpo_optimizer = grpo.optimizer
    grpo.fitness = [1, 2, 3]
    new_grpo = grpo.clone(index=1)
    for param, pre_learn_param in zip(
        new_grpo.actor.parameters(), grpo.actor.parameters()
    ):
        assert torch.equal(param, pre_learn_param)
    for param, pre_learn_param in zip(
        new_grpo.reference_actor.parameters(), grpo.reference_actor.parameters()
    ):
        assert torch.equal(param, pre_learn_param)
    assert new_grpo.index == 1
    if grpo.accelerator is not None:
        assert new_grpo.accelerator != grpo_accelerator
    assert new_grpo.lr_scheduler != grpo_lr_scheduler
    for pg1, pg2 in zip(
        grpo_optimizer.optimizer.param_groups,
        new_grpo.optimizer.optimizer.param_groups,
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
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_1},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_2},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
        (
            {"config": deepspeed_config_stage_3},
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
        ),
    ],
    indirect=["accelerator", "grpo"],
)
@pytest.mark.parametrize("batch_size", [8])
def test_grpo_clean_up(grpo, accelerator, request, batch_size):
    # # @pytest.mark
    # @pytest.mark.parametrize("vocab_size", [1000])
    # @pytest.mark.parametrize("input_size", [10])
    # @pytest.mark.parametrize("max_tokens", [20])
    # @pytest.mark.parametrize("group_size", [5])
    # @pytest.mark.parametrize("batch_size", [8])
    # def test_grpo_clean_up(
    #     vocab_size, input_size, max_tokens, group_size, batch_size, tmpdir
    # ):
    #     observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
    #     action_space = gym.spaces.Box(
    #         low=0,
    #         high=vocab_size - 1,
    #         shape=(20,),
    #     )
    #     grpo = GRPO(
    #         observation_space,
    #         action_space,
    #         actor_network=create_module(
    #             input_size=input_size,
    #             max_tokens=max_tokens,
    #             vocab_size=vocab_size,
    #             device="cuda" if torch.cuda.is_available() else "cpu",
    #         ),
    #         pad_token_id=vocab_size - 1,
    #         device="cuda" if torch.cuda.is_available() else "cpu",
    #         group_size=group_size,
    #         cosine_lr_schedule_config=CosineLRScheduleConfig(
    #             num_epochs=10, warmup_proportion=0.05
    #         ),
    #         accelerator=None,
    #     )
    #     mock_accelerator = MagicMock(spec=Accelerator)
    #     mock_accelerator.free_memory = lambda *args: (None,) * len(args)
    #     grpo.accelerator = mock_accelerator
    grpo.clean_up()
    assert grpo.actor is None
    assert grpo.reference_actor is None
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
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
            {"vocab_size": 1000, "input_size": 10, "max_tokens": 20, "group_size": 5},
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
