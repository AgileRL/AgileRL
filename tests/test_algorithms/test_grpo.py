import gymnasium as gym
import pytest
import torch
import torch.nn as nn
from transformers import GenerationConfig

from agilerl.algorithms import GRPO
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.modules.base import EvolvableModule
from agilerl.modules.dummy import to_evolvable


class DummyForwardOutput:
    def __init__(self, tensor):
        self.output = nn.Softmax(tensor)
        self.logits = tensor


class DummyLLM(nn.Module):
    def __init__(self, input_size, max_tokens, vocab_size):
        super().__init__()
        self.input_size = input_size
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.net = nn.Linear(
            input_size + max_tokens, (input_size + max_tokens) * vocab_size
        )
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


def create_module(input_size, max_tokens, vocab_size):
    return DummyLLM(input_size, max_tokens, vocab_size)


@pytest.fixture
def grpo(vocab_size, input_size, max_tokens, group_size):
    observation_space = gym.spaces.Box(low=0, high=vocab_size - 1)
    action_space = gym.spaces.Box(
        low=0,
        high=vocab_size - 1,
        shape=(20,),
    )
    return GRPO(
        observation_space,
        action_space,
        actor_network=to_evolvable(
            module_fn=create_module,
            module_kwargs={
                "input_size": input_size,
                "max_tokens": max_tokens,
                "vocab_size": vocab_size,
            },
            device="cuda",
        ),
        pad_token_id=vocab_size - 1,
        device="cuda",
        group_size=group_size,
    )


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
def test_init_grpo(grpo, vocab_size, input_size, max_tokens, group_size):
    assert isinstance(grpo.observation_space, gym.spaces.Box)
    assert isinstance(grpo.action_space, gym.spaces.Box)
    assert grpo.batch_size == 8
    assert grpo.beta == 0.04
    assert grpo.lr == 5e-6
    assert grpo.clip_coef == 0.2
    assert grpo.max_grad_norm == 0.1
    assert grpo.update_epochs == 1
    assert grpo.group_size == group_size
    assert grpo.temperature == 0.9
    assert grpo.calc_position_embeddings
    assert grpo.device == "cuda"
    assert grpo.index == 0
    assert grpo.scores == []
    assert grpo.fitness == []
    assert grpo.steps == [0]
    assert isinstance(grpo.generation_config, GenerationConfig)
    assert isinstance(grpo.actor, EvolvableModule)
    assert isinstance(grpo.reference_actor, EvolvableModule)
    for ref_param, param in zip(
        grpo.reference_actor.parameters(), grpo.actor.parameters()
    ):
        assert torch.equal(ref_param, param)
    assert grpo.actor.module.gradient_checkpointing_enabled
    assert not grpo.reference_actor.module.gradient_checkpointing_enabled
    assert isinstance(grpo.optimizer, OptimizerWrapper)


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("data_batch_size", [8])
@pytest.mark.parametrize("group_size, training", [(5, True), (1, False)])
def test_get_action_grpo(
    grpo, vocab_size, input_size, max_tokens, group_size, data_batch_size, training
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
    assert not grpo.actor.training


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
def test_calculate_advantage(
    grpo, vocab_size, input_size, max_tokens, group_size, batch_size
):
    rewards = torch.tensor([[2, 4, 6, 8, 20], [3, 6, 9, 12, 15]], dtype=torch.float32)
    mean_rewards = torch.mean(rewards, dim=1).unsqueeze(1)
    std_rewards = torch.std(rewards, dim=1).unsqueeze(1)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
    advantages = advantages.flatten().unsqueeze(1)
    assert torch.equal(advantages, grpo._calculate_advantage(rewards))


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
def test_calculate_kl_divergence(grpo, vocab_size, input_size, max_tokens, group_size):
    normal_dist = torch.distributions.normal.Normal(0.0, 1.0)
    reference_log_probs = normal_dist.log_prob(torch.randn(5))
    log_probs = normal_dist.log_prob(torch.randn(5))

    kl = grpo._calculate_kl_divergence(reference_log_probs, log_probs)
    assert torch.all(kl >= 0.0)
    assert isinstance(kl, torch.Tensor)
    assert kl.shape == log_probs.shape
    assert kl.shape == reference_log_probs.shape


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
def test_grpo_loss(grpo, vocab_size, input_size, max_tokens, group_size):
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
    assert not isinstance(loss, torch.Tensor)
    assert not isinstance(loss, torch.Tensor)
    assert loss != 0
    assert kl != 0


@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("batch_size", [8])
def test_get_logprobs(grpo, vocab_size, input_size, max_tokens, group_size, batch_size):
    ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
        grpo.device
    )
    log_probs = grpo._get_logprobs(ids=ids)
    grpo.reduce_memory_peak = True
    log_probs_reduced_mem = grpo._get_logprobs(ids=ids)
    assert log_probs.shape == (ids.shape[0], ids.shape[1] - 1)
    assert log_probs_reduced_mem.shape == (ids.shape[0], ids.shape[1] - 1)
    assert torch.allclose(log_probs, log_probs_reduced_mem)
