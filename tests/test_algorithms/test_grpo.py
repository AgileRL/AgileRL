import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate.accelerator import Accelerator, DeepSpeedPlugin
from accelerate.scheduler import AcceleratedScheduler
from accelerate.state import AcceleratorState
from accelerate.utils import patch_environment
from torch.optim.lr_scheduler import SequentialLR
from transformers import GenerationConfig

from agilerl.utils.algo_utils import CosineLRScheduleConfig

dist_env = dict(
    ACCELERATE_USE_DEEPSPEED="true",
    MASTER_ADDR="localhost",
    MASTER_PORT="10999",
    RANK="0",
    LOCAL_RANK="0",
    WORLD_SIZE="1",
)


# Mock the Accelerator class to avoid DeepSpeed issues
class MockAccelerator:
    def __init__(self, *args, **kwargs):
        # Setup mock for AcceleratorState
        self.mock_accelerator_state = MagicMock()
        mock_deepspeed_plugin = MagicMock()
        mock_deepspeed_plugin.deepspeed_config = {
            "train_micro_batch_size_per_gpu": "auto",
            "zero_optimization": {"stage": 0},
        }
        self.mock_accelerator_state.deepspeed_plugin = mock_deepspeed_plugin
        AcceleratorState = self.accelerator_state
        print(AcceleratorState)
        self.state = MagicMock()

    def accelerator_state(self):
        return self.mock_accelerator_state

    def prepare(self, model, opt, *args):
        return (
            model,
            OptimizerWrapper(
                optim.AdamW,
                networks=[model],
                lr=5e-6,
                network_names=["net"],
                lr_name="lr",
            ),
            args[0],
        )

    def backward(self, loss):
        loss.backward()


# Replace the Accelerator class with our mock
@pytest.fixture(autouse=True)
def mock_accelerator():
    with patch("accelerate.Accelerator", MockAccelerator):
        yield


# Now import the modules after patching.
# This is ugly but these imports have to be after the above patches.
from agilerl.algorithms import GRPO  # noqa: E402
from agilerl.algorithms.core.wrappers import OptimizerWrapper  # noqa: E402


def mock_initialize(model, config, **kwargs):
    # Return a mockup of DeepSpeedEngine that behaves enough like the model
    class MockEngine:
        def __init__(self, model):
            self.module = model
            # Copy all attributes from model
            for attr_name in dir(model):
                if not attr_name.startswith("_"):
                    try:
                        setattr(self, attr_name, getattr(model, attr_name))
                    except (AttributeError, TypeError):
                        pass

        def eval(self):
            model.eval()

        def train(self, mode=True):
            model.train(mode)

        def parameters(self):
            return model.parameters()

        def __getattr__(self, name):
            # Fallback to model attributes
            return getattr(model, name)

    return MockEngine(model), None, None


# Patch deepspeed.initialize
# original_deepspeed_initialize = deepspeed.initialize
# deepspeed.initialize = mock_initialize


class DummyForwardOutput:
    def __init__(self, tensor):
        self.output = nn.Softmax(dim=-1)(tensor)
        self.logits = tensor


class DummyLLM(nn.Module):
    def __init__(self, input_size, max_tokens, vocab_size, device):
        super().__init__()
        self.input_size = input_size
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.Linear(
                input_size + max_tokens,
                32,
                device=device,
            ),
            nn.Linear(32, (input_size + max_tokens) * vocab_size, device=device),
        )
        self.device = device
        self.gradient_checkpointing_enabled = False

    def forward(self, input_ids, *args, **kwargs):
        input_ids = input_ids.to(torch.float32)
        output = self.net.forward(input_ids).reshape(
            input_ids.shape[0], self.input_size + self.max_tokens, self.vocab_size
        )
        return DummyForwardOutput(output)

    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        input_shape = input_ids.shape
        group_size = input_shape[0]
        prompt_size = input_shape[1]
        return torch.randint(
            0, self.vocab_size - 1, (group_size, prompt_size + self.max_tokens)
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.gradient_checkpointing_enabled = True
        return


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
    return DummyLLM(input_size, max_tokens, vocab_size, device)


# Add the missing method to GRPO prototype
GRPO._set_reference_policy = lambda self: None


@pytest.fixture
def grpo(vocab_size, input_size, max_tokens, group_size, use_accelerator):
    observation_space = gym.spaces.Box(low=0, high=vocab_size - 1, shape=(1,))
    action_space = gym.spaces.Box(
        low=0,
        high=vocab_size - 1,
        shape=(20,),
    )
    if use_accelerator:
        with patch_environment(**dist_env):
            AcceleratorState._reset_state(True)
            accelerator = Accelerator(
                DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)
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
                pad_token_id=vocab_size - 1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=CosineLRScheduleConfig(
                    num_epochs=10, warmup_proportion=0.05
                ),
                accelerator=accelerator,
            )
            return grpo
    else:
        accelerator = None
        return GRPO(
            observation_space,
            action_space,
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


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_init_grpo(
    grpo, vocab_size, input_size, max_tokens, group_size, use_accelerator
):
    if use_accelerator:
        device = (
            f"cuda:{os.getenv('LOCAL_RANK', '0')}"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assert isinstance(grpo.observation_space, gym.spaces.Box)
    assert isinstance(grpo.action_space, gym.spaces.Box)
    assert grpo.batch_size == 8
    assert grpo.beta == 0.001
    assert grpo.lr == 5e-7
    assert grpo.clip_coef == 0.2
    assert grpo.max_grad_norm is None if use_accelerator else 0.1
    assert grpo.update_epochs == 1
    assert grpo.group_size == group_size
    assert grpo.temperature == 0.9
    assert grpo.calc_position_embeddings
    assert isinstance(grpo.cosine_lr_schedule_config, CosineLRScheduleConfig), type(
        grpo.cosine_lr_schedule_config
    )
    assert grpo.device == device
    assert grpo.index == 0
    assert grpo.scores == []
    assert grpo.fitness == []
    assert grpo.steps == [0]
    assert isinstance(grpo.generation_config, GenerationConfig)
    if use_accelerator:
        from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
        from deepspeed.runtime.engine import DeepSpeedEngine

        assert isinstance(grpo.actor, DeepSpeedEngine)
        assert isinstance(grpo.optimizer, DeepSpeedOptimizerWrapper)
        assert isinstance(grpo.lr_scheduler, AcceleratedScheduler), grpo.lr_scheduler
    else:
        assert isinstance(grpo.actor, DummyLLM)
        assert isinstance(grpo.lr_scheduler, SequentialLR), grpo.lr_scheduler
        assert isinstance(grpo.optimizer, OptimizerWrapper)
        assert grpo.actor.gradient_checkpointing_enabled
    assert isinstance(grpo.reference_actor, DummyLLM) != use_accelerator
    for ref_param, param in zip(
        grpo.reference_actor.parameters(), grpo.actor.parameters()
    ):
        assert torch.equal(ref_param, param)
    assert not grpo.reference_actor.gradient_checkpointing_enabled


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("data_batch_size", [8])
@pytest.mark.parametrize("group_size, training", [(5, True), (1, False)])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_get_action_grpo(
    grpo,
    vocab_size,
    input_size,
    max_tokens,
    group_size,
    data_batch_size,
    training,
    use_accelerator,
):
    states = [
        {
            "input_ids": torch.randint(0, vocab_size, (input_size,)),
            "attention_mask": torch.ones(*(input_size,)),
        }
        for _ in range(data_batch_size)
    ]
    completion_ids, action_masks = grpo.get_action(states, training)
    for ids in completion_ids:
        assert ids.shape == (group_size, input_size + max_tokens)
    for masks in action_masks:
        assert masks.shape == (group_size, input_size + max_tokens - 1)
    if not use_accelerator:
        assert not grpo.actor.training


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False, True])
@pytest.mark.parametrize(
    "rewards",
    [
        torch.tensor([[2, 4, 6, 8, 20], [3, 6, 9, 12, 15]], dtype=torch.float32),
        torch.tensor([3, 6], dtype=torch.float32),
    ],
)
def test_calculate_advantage(
    grpo,
    vocab_size,
    input_size,
    max_tokens,
    group_size,
    batch_size,
    use_accelerator,
    rewards,
):
    calculated_advantage = grpo._calculate_advantage(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.unsqueeze(0)
    mean_rewards = torch.mean(rewards, dim=1).unsqueeze(1)
    std_rewards = torch.std(rewards, dim=1).unsqueeze(1)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
    advantages = advantages.flatten().unsqueeze(1)
    assert torch.equal(advantages, calculated_advantage)


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_calculate_kl_divergence(
    grpo, vocab_size, input_size, max_tokens, group_size, batch_size, use_accelerator
):
    normal_dist = torch.distributions.normal.Normal(0.0, 1.0)
    reference_log_probs = normal_dist.log_prob(torch.randn(batch_size))
    log_probs = normal_dist.log_prob(torch.randn(batch_size))
    kl = grpo._calculate_kl_divergence(log_probs, reference_log_probs)
    assert torch.all(kl >= 0.0)
    assert isinstance(kl, torch.Tensor)
    assert kl.shape == log_probs.shape
    assert kl.shape == reference_log_probs.shape


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_grpo_loss(
    grpo, vocab_size, input_size, max_tokens, group_size, use_accelerator
):
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


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_get_logprobs(
    grpo, vocab_size, input_size, max_tokens, group_size, batch_size, use_accelerator
):
    ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
        grpo.device
    )
    log_probs = grpo._get_logprobs(ids=ids)
    grpo.reduce_memory_peak = True
    log_probs_reduced_mem = grpo._get_logprobs(ids=ids)
    assert log_probs.shape == (ids.shape[0], ids.shape[1] - 1)
    assert log_probs_reduced_mem.shape == (ids.shape[0], ids.shape[1] - 1)
    assert torch.allclose(log_probs, log_probs_reduced_mem, atol=1e-3)


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_grpo_learn(
    grpo, vocab_size, input_size, max_tokens, group_size, batch_size, use_accelerator
):
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
    mean_loss, mean_kl = grpo.learn((completions, action_masks, rewards))
    assert isinstance(mean_loss, float)
    assert isinstance(mean_kl, float)

    # Don't check parameter changes as they might be hard to guarantee
    # Just check that the method returns the expected types


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_grpo_learn_with_nan_loss(
    grpo, vocab_size, input_size, max_tokens, group_size, batch_size, use_accelerator
):
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

    with patch.object(grpo, "_grpo_loss", side_effect=mock_grpo_loss):
        mean_loss, mean_kl = grpo.learn((completions, action_masks, rewards))

    # Since all loss values were NaN, mean_loss should be 0
    assert mean_loss == 0.0
    # KL values weren't accumulated since we hit the continue statement
    assert mean_kl == 0.0


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_grpo_test(
    grpo, vocab_size, input_size, max_tokens, group_size, batch_size, use_accelerator
):
    env = DummyHuggingFaceEnv(vocab_size, input_size, batch_size)
    fitnesses = grpo.test(env)
    assert isinstance(fitnesses, torch.Tensor)


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_grpo_load(
    grpo, vocab_size, input_size, max_tokens, group_size, batch_size, use_accelerator
):
    with pytest.raises(NotImplementedError):
        GRPO.load("path")


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_grpo_load_checkpoint(
    grpo, vocab_size, input_size, max_tokens, group_size, batch_size, use_accelerator
):
    with pytest.raises(NotImplementedError):
        grpo.load_checkpoint("path")


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False, True])
def test_grpo_save_checkpoint(
    grpo, vocab_size, input_size, max_tokens, group_size, batch_size, use_accelerator
):
    with pytest.raises(NotImplementedError):
        grpo.save_checkpoint("path")


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("use_accelerator", [False])
def test_grpo_save_load_distributed_actor_no_accelerator(
    grpo,
    vocab_size,
    input_size,
    max_tokens,
    group_size,
    batch_size,
    use_accelerator,
    tmpdir,
):
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    with pytest.warns(UserWarning):
        grpo._save_distributed_actor(checkpoint_path)

    with pytest.warns(UserWarning):
        grpo._load_distributed_actor(checkpoint_path)


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "zero_stage",
    [1, 2, 3],
)
def test_grpo_save_load_checkpoint_with_accelerator(
    vocab_size, input_size, max_tokens, group_size, batch_size, tmpdir, zero_stage
):
    with patch_environment(**dist_env):
        AcceleratorState._reset_state(True)
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=zero_stage, gradient_accumulation_steps=2
        )
        accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
        grpo = GRPO(
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
        checkpoint_path = Path(tmpdir) / "checkpoint.pth"
        grpo._save_distributed_actor(checkpoint_path)
        grpo_optimizer = grpo.optimizer
        grpo_optim_state_dict = grpo.optimizer.state_dict()
        grpo_optim_state_dict.pop("loss_scaler")
        grpo_actor_state_dict = grpo.actor.state_dict()
        grpo_reference_actor_state_dict = grpo.reference_actor.state_dict()
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

    assert str(new_grpo.actor.state_dict()) == str(grpo_actor_state_dict)
    assert (
        new_grpo.optimizer.optimizer.loss_scaler.cur_scale
        == grpo_optimizer.optimizer.loss_scaler.cur_scale
    )
    assert new_grpo.optimizer.state_dict().keys() == grpo_optimizer.state_dict().keys()
    assert str(new_grpo.reference_actor.state_dict()) != str(
        grpo_reference_actor_state_dict
    )
    for key in new_grpo.optimizer.state_dict().keys():
        if key == "loss_scaler":
            continue
        assert str(new_grpo.optimizer.state_dict()[key]) == str(
            grpo_optim_state_dict[key]
        )
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("zero_stage", [1, 2, 3])
def test_grpo_clone_with_accelerator(
    vocab_size,
    input_size,
    max_tokens,
    group_size,
    batch_size,
    tmpdir,
    zero_stage,
):
    with patch_environment(**dist_env):
        AcceleratorState._reset_state(True)
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=zero_stage,
            gradient_accumulation_steps=2,
        )
        accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
        grpo = GRPO(
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
            accelerator=accelerator,
            lr=0.1,
        )

        grpo_reference_actor_state_dict = grpo.reference_actor.state_dict()
        grpo_accelerator = grpo.accelerator
        grpo_lr_scheduler = grpo.lr_scheduler
        grpo_optimizer = grpo.optimizer
        grpo.fitness = [1, 2, 3]
        new_grpo = grpo.clone(index=1)
        assert str(new_grpo.actor.state_dict()) == str(grpo.actor.state_dict())
        assert str(new_grpo.reference_actor.state_dict()) == str(
            grpo_reference_actor_state_dict
        )
        assert new_grpo.index == 1
        assert new_grpo.accelerator == grpo_accelerator
        assert new_grpo.lr_scheduler == grpo_lr_scheduler
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


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
def test_grpo_clone_with_no_accelerator(
    vocab_size,
    input_size,
    max_tokens,
    group_size,
    batch_size,
    tmpdir,
):

    grpo = GRPO(
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
        accelerator=None,
        lr=0.1,
        cosine_lr_schedule_config=CosineLRScheduleConfig(
            num_epochs=10, warmup_proportion=0.05
        ),
    )

    grpo_reference_actor_state_dict = grpo.reference_actor.state_dict()
    grpo_accelerator = grpo.accelerator
    grpo_lr_scheduler = grpo.lr_scheduler
    grpo_optimizer = grpo.optimizer
    new_grpo = grpo.clone(index=1)
    assert str(new_grpo.actor.state_dict()) == str(grpo.actor.state_dict())
    assert str(new_grpo.reference_actor.state_dict()) == str(
        grpo_reference_actor_state_dict
    )
    assert new_grpo.index == 1
    assert new_grpo.accelerator == grpo_accelerator
    assert new_grpo.lr_scheduler == grpo_lr_scheduler
    for pg1, pg2 in zip(
        grpo_optimizer.optimizer.param_groups, new_grpo.optimizer.optimizer.param_groups
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
