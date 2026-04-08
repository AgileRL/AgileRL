import gc
from types import SimpleNamespace
import copy
import pytest
import torch
from accelerate.state import AcceleratorState
from peft import LoraConfig
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig
from tests.utils import spawn_new_process_for_each_test

pytestmark = pytest.mark.llm

deepspeed_base_config = {
    "bf16": {
        "enabled": True,
    },
    "auto_cast": True,
    "gradient_clipping": 0.5,
    "gradient_accumulation_steps": 1,
}

deepspeed_config_stage_2 = deepspeed_base_config | {
    "zero_optimization": {
        "stage": 2,
    },
}


class DummyConfig(PretrainedConfig):
    def __init__(
        self,
        input_size=16,
        max_tokens=8,
        vocab_size=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size


class DummyMLPPreTrainedModelWithValueHead(PreTrainedModel):
    """Dummy model whose forward returns (logits, loss, values) to match
    TRL's AutoModelForCausalLMWithValueHead output expected by PPO."""

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
        seq_len = self.input_size + self.max_tokens
        self.linear_1 = nn.Linear(
            seq_len,
            32,
            device=device,
            dtype=self.datatype,
        )
        # Head for logits: (batch, seq_len, vocab_size)
        self.linear_2 = nn.Linear(
            32,
            seq_len * self.vocab_size,
            device=device,
            dtype=self.datatype,
        )
        # Value head: (batch, seq_len)
        self.v_head = nn.Linear(
            32,
            seq_len,
            device=device,
            dtype=self.datatype,
        )

    @property
    def model(self):
        return self

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        x = input_ids.to(self.datatype)
        hidden = self.linear_1(x)
        seq_len = self.input_size + self.max_tokens
        logits = self.linear_2(hidden).reshape(
            x.shape[0],
            seq_len,
            self.vocab_size,
        )
        values = self.v_head(hidden)  # (batch, seq_len)
        return logits, None, values

    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            msg = "`input_ids` must be provided for generation."
            raise ValueError(msg)
        input_shape = input_ids.shape
        batch_size = input_shape[0]
        prompt_size = input_shape[1]
        return torch.randint(
            0,
            self.vocab_size,
            (batch_size, prompt_size + self.config.max_tokens),
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.gradient_checkpointing_enabled = True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return


class DummyVLLM:
    def __init__(self, *args, **kwargs):
        from unittest.mock import MagicMock

        self.llm_engine = MagicMock()
        self.llm_engine.model_executor = MagicMock()

    def generate(self, prompts, *args, **kwargs):
        import random

        all_outputs = []
        for _ in range(len(prompts)):
            token_length = random.randint(5, 20)
            token_ids = [random.randint(0, 1000) for _ in range(token_length)]
            dummy_output = SimpleNamespace(token_ids=token_ids)
            request_output = SimpleNamespace(outputs=[dummy_output])
            all_outputs.append(request_output)
        return all_outputs

    def reset_prefix_cache(self):
        pass

    def sleep(self, *args, **kwargs):
        pass

    def wake_up(self, *args, **kwargs):
        pass


def create_module(input_size, max_tokens, vocab_size, device):
    return DummyMLPPreTrainedModelWithValueHead(
        config=DummyConfig(
            input_size=input_size,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
        ),
        device=device,
    )


def generate_ppo(
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    vocab_size,
    input_size,
    max_tokens,
    use_vllm,
    pretrained_model_name_or_path,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
    lr=1e-5,
    sleep_mode=False,
    from_name=False,
):
    gc.collect()
    torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)

    accelerator = accelerator_factory(use_deepspeed_optimizer, config)
    if not use_deepspeed_optimizer and accelerator is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config.pop("optimizer", None)

    if use_vllm:
        lora_config = None
        vllm_config = VLLMConfig(
            gpu_memory_utilization=0.2, max_num_seqs=1, sleep_mode=sleep_mode
        )
        actor = model_factory(pretrained_model_name_or_path, add_value_head=True)
    else:
        if pretrained_model_name_or_path is not None:
            actor = model_factory(pretrained_model_name_or_path, add_value_head=True)
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
            modules_to_save=["v_head"],
        )
        vllm_config = None

    ppo = LLMPPO(
        actor_network=actor if not from_name else None,
        model_name=pretrained_model_name_or_path if from_name else None,
        lr_actor=1e-5,
        pad_token_id=vocab_size - 1,
        pad_token="<pad>",
        device="cuda" if torch.cuda.is_available() else "cpu",
        lora_config=lora_config,
        cosine_lr_schedule_config=(
            None
            if accelerator is not None
            else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
        ),
        accelerator=accelerator,
        use_vllm=use_vllm,
        vllm_config=vllm_config,
        max_output_tokens=max_tokens,
        max_model_len=max_tokens + 5,
        reduce_memory_peak=reduce_memory_peak,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
    )
    return ppo


@pytest.fixture(scope="function")
def ppo_factory():
    return generate_ppo


@spawn_new_process_for_each_test
@pytest.mark.parametrize("config", [deepspeed_config_stage_2])
@pytest.mark.parametrize("use_deepspeed_optimizer", [False])
@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("use_vllm", [True, False])
@pytest.mark.parametrize("pretrained_model_name_or_path", ["facebook/opt-125m"])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
@pytest.mark.parametrize("batch_size", [8])
def test_ppo_learns(
    deepspeed_env,
    ppo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_vllm,
    pretrained_model_name_or_path,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
    vocab_size,
    input_size,
    max_tokens,
    batch_size,
):
    ppo = ppo_factory(
        accelerator_factory=accelerator_factory,
        model_factory=model_factory,
        config=config,
        use_deepspeed_optimizer=use_deepspeed_optimizer,
        vocab_size=vocab_size,
        input_size=input_size,
        max_tokens=max_tokens,
        use_vllm=use_vllm,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        reduce_memory_peak=reduce_memory_peak,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        lr_actor=0.01,
    )
    completions = [
        torch.randint(
            0,
            vocab_size,
            (1, input_size + max_tokens),
            device=ppo.device,
        )
        for _ in range(batch_size)
    ]
    action_masks = [
        torch.ones((1, input_size + max_tokens - 1), device=ppo.device)
        for _ in range(batch_size)
    ]
    rewards = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], dtype=torch.float32
    ).unsqueeze(1)

    # DeepSpeed uses first step in cycle to allocate the fp32 master weights and synchronize them with your bf16 params.
    ppo.learn((completions, action_masks, rewards))

    pre_learn_actor_state_dict = {
        name: param.clone().detach() for name, param in ppo.actor.named_parameters()
    }

    for name, param in ppo.actor.named_parameters():
        if param.requires_grad:
            print(name)

    mean_loss, mean_kl, *_ = ppo.learn((completions, action_masks, rewards))
    assert isinstance(mean_loss, float)
    assert isinstance(mean_kl, float)

    # Reference adapter must be frozen.
    for param_name, param in ppo.actor.named_parameters():
        if "reference" in param_name:
            assert torch.equal(param, pre_learn_actor_state_dict[param_name]), (
                f"{param_name} should not change but did"
            )

    vhead_changed = any(
        not torch.allclose(param, pre_learn_actor_state_dict[pname])
        for pname, param in ppo.actor.named_parameters()
        if "v_head" in pname and "reference" not in pname
    )
    assert vhead_changed, "No v_head parameters were updated after learn()"

    actor_lora_changed = any(
        not torch.allclose(param, pre_learn_actor_state_dict[pname])
        for pname, param in ppo.actor.named_parameters()
        if "actor" in pname and "lora" in pname
    )
    assert actor_lora_changed, "No actor LoRA parameters were updated after learn()"

    ppo.clean_up()


# Lightweight stub — only the attributes that _compute_* methods need.


class _PPOStub:
    def __init__(self, gamma: float = 1.0, gae_lambda: float = 1.0):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    _compute_token_rewards = LLMPPO._compute_token_rewards
    _compute_gae_returns = LLMPPO._compute_gae_returns


class TestComputeTokenRewards:
    def test_reward_placed_at_last_valid_position(self):
        """Sequence reward goes to the last True position in the action mask."""
        stub = _PPOStub()
        # valid positions: 1, 2, 3  → last = 3
        action_mask = torch.tensor([[False, True, True, True, False]])
        sequence_rewards = torch.tensor([2.5])

        result = stub._compute_token_rewards(action_mask, sequence_rewards)

        expected = torch.zeros(1, 5)
        expected[0, 3] = 2.5
        assert torch.allclose(result, expected)

    def test_multiple_rows_each_assigned_correctly(self):
        """Each row's reward lands on its own last valid position."""
        stub = _PPOStub()
        action_mask = torch.tensor(
            [
                [True, True, False, False],  # last valid = 1
                [False, True, True, True],  # last valid = 3
            ]
        )
        sequence_rewards = torch.tensor([1.0, 3.0])

        result = stub._compute_token_rewards(action_mask, sequence_rewards)

        expected = torch.zeros(2, 4)
        expected[0, 1] = 1.0
        expected[1, 3] = 3.0
        assert torch.allclose(result, expected)

    def test_all_invalid_row_stays_zero(self):
        """A row with no valid positions receives no reward."""
        stub = _PPOStub()
        action_mask = torch.tensor(
            [
                [True, True, False],
                [False, False, False],
            ]
        )
        sequence_rewards = torch.tensor([1.0, 2.0])

        result = stub._compute_token_rewards(action_mask, sequence_rewards)

        expected = torch.zeros(2, 3)
        expected[0, 1] = 1.0  # only valid row gets assigned
        assert torch.allclose(result, expected)

    def test_single_valid_position(self):
        """When only one position is valid the reward lands there."""
        stub = _PPOStub()
        action_mask = torch.tensor([[False, True, False, False]])
        sequence_rewards = torch.tensor([7.0])

        result = stub._compute_token_rewards(action_mask, sequence_rewards)

        expected = torch.zeros(1, 4)
        expected[0, 1] = 7.0
        assert torch.allclose(result, expected)

    def test_all_valid_positions(self):
        """Reward goes to the very last position when the whole sequence is valid."""
        stub = _PPOStub()
        action_mask = torch.ones(1, 5, dtype=torch.bool)
        sequence_rewards = torch.tensor([4.0])

        result = stub._compute_token_rewards(action_mask, sequence_rewards)

        expected = torch.zeros(1, 5)
        expected[0, 4] = 4.0
        assert torch.allclose(result, expected)


class TestComputeGAEReturns:
    def test_terminal_reward_zero_values_propagates_uniformly(self):
        """With gamma=1, lambda=1 and V=0 a terminal reward propagates back
        to every valid token position with equal magnitude."""
        stub = _PPOStub(gamma=1.0, gae_lambda=1.0)
        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        values = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        # t=2: delta=1, gae=1
        # t=1: delta=0+0−0=0, gae=0+1·1·1=1
        # t=0: delta=0+0−0=0, gae=0+1·1·1=1
        returns, advantages = stub._compute_gae_returns(rewards, values, mask)

        assert torch.allclose(advantages, torch.tensor([[1.0, 1.0, 1.0]]))
        assert torch.allclose(returns, torch.tensor([[1.0, 1.0, 1.0]]))

    def test_value_bootstrap_reduces_advantage(self):
        """Non-zero values are subtracted via the TD-delta, so advantages are
        smaller but returns (adv + V) recover the full expected discounted sum."""
        stub = _PPOStub(gamma=1.0, gae_lambda=1.0)
        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        # t=2: delta=1+0−0.5=0.5,       gae=0.5
        # t=1: delta=0+0.5−0.5=0,        gae=0+0.5=0.5
        # t=0: delta=0+0.5−0.5=0,        gae=0+0.5=0.5
        returns, advantages = stub._compute_gae_returns(rewards, values, mask)

        assert torch.allclose(advantages, torch.tensor([[0.5, 0.5, 0.5]]))
        assert torch.allclose(returns, torch.tensor([[1.0, 1.0, 1.0]]))

    def test_padding_zeroed_does_not_affect_valid_positions(self):
        """Values and rewards at padding positions are zeroed before the loop,
        so GAE for valid tokens is unaffected by whatever was in padding slots."""
        stub = _PPOStub(gamma=1.0, gae_lambda=1.0)
        rewards = torch.tensor([[0.0, 1.0, 0.0]])
        values = torch.tensor(
            [[0.0, 0.0, 9.9]]
        )  # large padding value, should be masked
        mask = torch.tensor([[True, True, False]])

        # After masking: rewards=[0,1,0], values=[0,0,0]
        # t=2: delta=0, gae=0
        # t=1: delta=1+0−0=1, gae=1
        # t=0: delta=0+0−0=0, gae=0+1=1
        returns, advantages = stub._compute_gae_returns(rewards, values, mask)

        assert torch.allclose(advantages, torch.tensor([[1.0, 1.0, 0.0]]))
        assert torch.allclose(returns, torch.tensor([[1.0, 1.0, 0.0]]))

    def test_gamma_discounting(self):
        """gamma < 1 exponentially discounts future rewards."""
        gamma = 0.9
        stub = _PPOStub(gamma=gamma, gae_lambda=1.0)
        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        values = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        # t=2: delta=1, gae=1
        # t=1: delta=0, gae=0+0.9·1·1=0.9
        # t=0: delta=0, gae=0+0.9·1·0.9=0.81
        returns, advantages = stub._compute_gae_returns(rewards, values, mask)

        expected_adv = torch.tensor([[0.81, 0.9, 1.0]])
        assert torch.allclose(advantages, expected_adv, atol=1e-5)
        assert torch.allclose(returns, expected_adv, atol=1e-5)

    def test_gae_lambda_smoothing(self):
        """lambda < 1 blends multi-step returns, biasing towards the 1-step TD
        estimate (high bias / low variance)."""
        stub = _PPOStub(gamma=1.0, gae_lambda=0.0)
        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        # lambda=0 → gae_t = delta_t (pure 1-step TD error, no carry)
        # t=2: delta=1+0−0.5=0.5,  gae=0.5
        # t=1: delta=0+0.5−0.5=0,  gae=0 (carry zeroed by lambda=0)
        # t=0: delta=0+0.5−0.5=0,  gae=0
        returns, advantages = stub._compute_gae_returns(rewards, values, mask)

        assert torch.allclose(advantages, torch.tensor([[0.0, 0.0, 0.5]]))
        assert torch.allclose(returns, torch.tensor([[0.5, 0.5, 1.0]]))
