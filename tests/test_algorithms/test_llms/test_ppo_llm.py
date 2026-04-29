import gc
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from accelerate.state import AcceleratorState
from peft import LoraConfig
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig
from agilerl.utils.llm_utils import ReasoningGym, masked_whiten
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from tests.utils import (
    assert_vllm_get_action_contract,
    make_mock_vllm_instance,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

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
        hidden_size=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class DummyCausalInner(PreTrainedModel):
    """Tiny causal LM used as ``pretrained_model`` inside
    :class:`~agilerl.utils.ppo_value_head.AutoModelForCausalLMWithValueHead`."""

    config_class = DummyConfig
    base_model_prefix = "dummy_inner"

    def __init__(self, config: DummyConfig, device="cpu"):
        super().__init__(config)
        self.name_or_path = "dummy-causal-llm"
        self.gradient_checkpointing_enabled = False
        hs = config.hidden_size
        vs = config.vocab_size
        self.embed = nn.Embedding(vs, hs, device=device)
        self.lin = nn.Linear(hs, hs, device=device)
        self.lm_head = nn.Linear(hs, vs, device=device)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        del attention_mask
        x = self.embed(input_ids)
        h = torch.relu(self.lin(x))
        logits = self.lm_head(h)
        if not return_dict:
            return (logits,)
        hidden = (h,) if output_hidden_states else None
        return CausalLMOutputWithPast(logits=logits, hidden_states=hidden)

    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            msg = "`input_ids` must be provided for generation."
            raise ValueError(msg)
        batch_size, prompt_size = input_ids.shape
        return torch.randint(
            0,
            self.config.vocab_size,
            (batch_size, prompt_size + self.config.max_tokens),
            device=input_ids.device,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self, *args, **kwargs):
        self.gradient_checkpointing_enabled = False

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}


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
    cfg = DummyConfig(
        input_size=input_size,
        max_tokens=max_tokens,
        vocab_size=vocab_size,
    )
    inner = DummyCausalInner(cfg, device=device)
    return AutoModelForCausalLMWithValueHead(inner)


def _cpu_llmppo(**kwargs):
    """Small CPU LLMPPO for fast unit tests (dummy actor + LoRA, no accelerator)."""
    device = "cpu"
    vocab_size = 100
    input_size = 10
    max_tokens = 8
    actor = create_module(input_size, max_tokens, vocab_size, device)
    defaults = {
        "actor_network": actor,
        "pad_token_id": vocab_size - 1,
        "pad_token": "<pad>",
        "lora_config": LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            modules_to_save=["summary"],
        ),
        "batch_size": 4,
        "micro_batch_size_per_gpu": 2,
        "max_output_tokens": max_tokens,
        "max_model_len": input_size + max_tokens + 4,
        "accelerator": None,
        "wrap": False,
        "gradient_checkpointing": False,
        "use_vllm": False,
        "lr_actor": 1e-3,
        "lr_critic": 1e-3,
        "update_epochs": 1,
        "vf_coef": 0.5,
        "beta": 0.01,
        "seed": 0,
        "device": device,
    }
    defaults.update(kwargs)
    return LLMPPO(**defaults)


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
    micro_batch_size_per_gpu,
    lr_actor=1e-5,
    lr_critic=1e-4,
    sleep_mode=False,
    from_name=False,
    use_memory_efficient_params=True,
    use_scheduler=False,
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
            target_modules = ["lin"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            # modules_to_save=["summary"], # I'm not sure if this needs to be here - I think value head is left out of the lora
        )
        vllm_config = None

    ppo = LLMPPO(
        actor_network=actor if not from_name else None,
        model_name=pretrained_model_name_or_path if from_name else None,
        lr_actor=lr_actor,
        lr_critic=lr_critic if lr_critic is not None else 10 * lr_actor,
        pad_token_id=vocab_size - 1,
        pad_token="<pad>",
        device="cuda" if torch.cuda.is_available() else "cpu",
        lora_config=lora_config,
        cosine_lr_schedule_config=(
            None
            if accelerator is not None
            else (
                CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
                if use_scheduler
                else None
            )
        ),
        accelerator=accelerator,
        use_vllm=use_vllm,
        vllm_config=vllm_config,
        max_output_tokens=max_tokens,
        max_model_len=max_tokens + 5,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        use_memory_efficient_params=use_memory_efficient_params,
    )
    return ppo


@pytest.fixture(scope="function")
def ppo_factory():
    return generate_ppo


@patch("agilerl.algorithms.core.base.LLM")
def test_init_llmppo_vllm_sleep_mode_calls_sleep(MockLLM):
    mock_instance = make_mock_vllm_instance()
    MockLLM.return_value = mock_instance

    actor = create_module(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
        modules_to_save=["summary"],
    )
    ppo = LLMPPO(
        actor_network=actor,
        pad_token_id=99,
        pad_token="<pad>",
        lora_config=lora,
        use_vllm=True,
        vllm_config=VLLMConfig(
            gpu_memory_utilization=0.2,
            max_num_seqs=1,
            sleep_mode=True,
        ),
        max_output_tokens=8,
        max_model_len=32,
        wrap=False,
        gradient_checkpointing=False,
        device="cpu",
    )
    assert ppo.use_vllm
    mock_instance.sleep.assert_called()
    ppo.clean_up()


@patch("agilerl.algorithms.core.base.LLM")
def test_init_llmppo_warns_when_hf_generate_chunk_size_set_with_vllm(MockLLM):
    mock_instance = make_mock_vllm_instance()
    MockLLM.return_value = mock_instance
    actor = create_module(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
        modules_to_save=["summary"],
    )
    with pytest.warns(
        UserWarning, match="hf_generate_chunk_size.*ignored.*use_vllm=True"
    ):
        ppo = LLMPPO(
            actor_network=actor,
            pad_token_id=99,
            pad_token="<pad>",
            lora_config=lora,
            use_vllm=True,
            vllm_config=VLLMConfig(
                gpu_memory_utilization=0.2,
                max_num_seqs=1,
                sleep_mode=True,
            ),
            hf_generate_chunk_size=2,
            max_output_tokens=8,
            max_model_len=32,
            wrap=False,
            gradient_checkpointing=False,
            device="cpu",
        )
    ppo.clean_up()


def test_llmppo_get_action_vllm_routes_through_vllm_calls():
    ppo = _cpu_llmppo(use_vllm=False)
    ppo.use_vllm = True
    ppo.vllm_config = VLLMConfig(
        gpu_memory_utilization=0.2,
        max_num_seqs=1,
        sleep_mode=True,
    )
    ppo._vllm_awake = False
    ppo.llm = MagicMock()
    ppo.llm.wake_up = MagicMock()
    ppo.llm.sleep = MagicMock()
    prompts = [
        {
            "input_ids": torch.randint(0, 100, (1, 10), device=ppo.device),
            "attention_mask": torch.ones(1, 10, device=ppo.device),
            "text": "Write me a short story about a cat.",
        }
        for _ in range(2)
    ]
    mocked_ids = [
        torch.ones(1, 12, dtype=torch.long, device=ppo.device),
        torch.ones(1, 12, dtype=torch.long, device=ppo.device),
    ]
    mocked_masks = [
        torch.ones(1, 11, dtype=torch.bool, device=ppo.device),
        torch.ones(1, 11, dtype=torch.bool, device=ppo.device),
    ]
    with (
        patch.object(
            ppo,
            "_prepare_vllm_for_generation",
            wraps=ppo._prepare_vllm_for_generation,
        ) as mock_prepare,
        patch.object(ppo, "_move_model_to_vllm", return_value=None) as mock_move,
        patch.object(
            ppo,
            "_generate_with_vllm_colocate",
            return_value=(mocked_ids, mocked_masks),
        ) as mock_generate,
    ):
        completion_ids, action_masks = ppo.get_action(prompts, training=False)

    mock_prepare.assert_called_once()
    mock_move.assert_called_once()
    mock_generate.assert_called_once_with(prompts, 1, temperature=0.01)
    ppo.llm.wake_up.assert_called_once()
    ppo._prepare_vllm_for_training()
    ppo.llm.sleep.assert_called_once()
    assert completion_ids == mocked_ids
    assert action_masks == mocked_masks
    ppo.clean_up()


class _PPOStub:
    def __init__(
        self,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        turn_value_reduction: str = "mean",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.turn_value_reduction = turn_value_reduction

    _compute_token_rewards = LLMPPO._compute_token_rewards
    _compute_gae_returns = LLMPPO._compute_gae_returns


def test_compute_token_rewards_per_turn_reward_broadcasts_to_that_turns_tokens():
    stub = _PPOStub()
    action_mask = torch.ones(1, 4, dtype=torch.bool)
    turn_ids = torch.tensor([[0, 0, 1, 1]])
    rewards = torch.tensor([[1.0, 2.0]])
    out = stub._compute_token_rewards(action_mask, rewards, turn_ids)
    expected = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    assert torch.allclose(out, expected)


def test_compute_token_rewards_minus_one_positions_ignore_turn_columns():
    stub = _PPOStub()
    action_mask = torch.tensor([[True, True, False, False]])
    turn_ids = torch.tensor([[0, 0, -1, -1]])
    rewards = torch.tensor([[3.0]])
    out = stub._compute_token_rewards(action_mask, rewards, turn_ids)
    expected = torch.tensor([[3.0, 3.0, 0.0, 0.0]])
    assert torch.allclose(out, expected)


def test_compute_gae_returns_two_turns_manual_then_whiten_matches_reference():
    stub = _PPOStub(gamma=1.0, gae_lambda=1.0)
    action_mask = torch.ones(1, 2, dtype=torch.bool)
    turn_ids = torch.tensor([[0, 1]])
    values = torch.tensor([[0.0, 0.0]])
    rewards = torch.tensor([[1.0, 2.0]])
    returns, advantages = stub._compute_gae_returns(
        rewards, values, action_mask, turn_ids
    )
    raw_adv = torch.tensor([[3.0, 2.0]])
    exp_adv = masked_whiten(raw_adv, action_mask) * action_mask
    assert torch.allclose(advantages, exp_adv)
    assert returns.shape == values.shape
    assert torch.allclose(
        returns,
        raw_adv + torch.tensor([[0.0, 0.0]]),
    )


def test_compute_gae_returns_padding_positions_zero_advantage():
    stub = _PPOStub()
    action_mask = torch.tensor([[True, True, False]])
    turn_ids = torch.tensor([[0, 1, -1]])
    values = torch.zeros(1, 3)
    rewards = torch.tensor([[1.0, 2.0, 0.0]])
    _returns, advantages = stub._compute_gae_returns(
        rewards, values, action_mask, turn_ids
    )
    assert advantages[0, 2].item() == 0.0


def test_compute_gae_returns_turn_value_reduction_final_value_uses_last_token_value():
    stub = _PPOStub(gamma=1.0, gae_lambda=1.0, turn_value_reduction="final_value")
    action_mask = torch.ones(1, 4, dtype=torch.bool)
    turn_ids = torch.tensor([[0, 0, 1, 1]])
    values = torch.tensor([[1.0, 4.0, 2.0, 8.0]])
    rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

    returns, advantages = stub._compute_gae_returns(
        rewards,
        values,
        action_mask,
        turn_ids,
    )

    expected_advantages = torch.tensor([[-4.0, -4.0, -8.0, -8.0]])
    expected_returns = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


def test_init_requires_max_output_or_max_model_len():
    actor = create_module(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
        modules_to_save=["summary"],
    )
    with pytest.raises(ValueError, match="Either max_output_tokens or max_model_len"):
        LLMPPO(
            actor_network=actor,
            pad_token_id=99,
            pad_token="<pad>",
            lora_config=lora,
            max_output_tokens=None,
            max_model_len=None,
            wrap=False,
            gradient_checkpointing=False,
        )


def test_init_clip_coef_non_negative():
    actor = create_module(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
        modules_to_save=["summary"],
    )
    with pytest.raises(AssertionError):
        LLMPPO(
            actor_network=actor,
            pad_token_id=99,
            pad_token="<pad>",
            lora_config=lora,
            clip_coef=-0.1,
            wrap=False,
            gradient_checkpointing=False,
        )


def test_init_update_epochs_at_least_one():
    actor = create_module(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
        modules_to_save=["summary"],
    )
    with pytest.raises(AssertionError):
        LLMPPO(
            actor_network=actor,
            pad_token_id=99,
            pad_token="<pad>",
            lora_config=lora,
            update_epochs=0,
            wrap=False,
            gradient_checkpointing=False,
        )


def test_init_clone_requires_pretrained_like_actor():
    with pytest.raises(AssertionError, match="PeftModelProtocol"):
        LLMPPO(
            model_name="facebook/opt-125m",
            actor_network=object(),
            pad_token_id=99,
            pad_token="<pad>",
            clone=True,
            wrap=False,
            gradient_checkpointing=False,
        )


def test_llmppo_get_action_hf_path_contract():
    ppo = _cpu_llmppo(
        use_vllm=False,
        hf_generate_chunk_size=2,
        max_model_len=128,
        max_output_tokens=8,
    )
    batch_size = 4
    prompt_len = 10
    prompts = [
        {
            "input_ids": torch.randint(0, 100, (1, prompt_len), device=ppo.device),
            "attention_mask": torch.ones(1, prompt_len, device=ppo.device),
        }
        for _ in range(batch_size)
    ]
    for training in (True, False):
        completion_ids, action_masks = ppo.get_action(prompts, training=training)
        assert_vllm_get_action_contract(
            completion_ids=completion_ids,
            action_masks=action_masks,
            batch_size=batch_size,
            prompt_len=prompt_len,
            pad_token_id=ppo.pad_token_id,
        )
    ppo.clean_up()


def test_llmppo_get_action_hf_path_handles_actor_without_parameters():
    ppo = _cpu_llmppo(
        use_vllm=False,
        hf_generate_chunk_size=2,
        max_model_len=128,
        max_output_tokens=8,
    )

    class _NoParamModule:
        def parameters(self):
            return iter(())

    prompts = [
        {
            "input_ids": torch.randint(0, 100, (1, 10), device=ppo.device),
            "attention_mask": torch.ones(1, 10, device=ppo.device),
        }
    ]

    with patch.object(ppo, "_get_unwrapped_actor", return_value=_NoParamModule()):
        completion_ids, action_masks = ppo.get_action(prompts, training=True)

    assert_vllm_get_action_contract(
        completion_ids=completion_ids,
        action_masks=action_masks,
        batch_size=1,
        prompt_len=10,
        pad_token_id=ppo.pad_token_id,
    )
    ppo.clean_up()


def test_learn_multi_turn_explicit_turn_ids():
    ppo = _cpu_llmppo(lr_actor=0.05, update_epochs=2)
    vocab = 100
    inp, mtok = 10, 8
    seq_len = inp + mtok
    b = 1
    completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(b)]
    action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(b)]
    turn_ids = torch.tensor(
        [[-1] * (inp - 1) + [0] * (mtok // 2) + [1] * (mtok - mtok // 2)],
        dtype=torch.long,
    )
    turn_ids = turn_ids[:, : seq_len - 1]
    rewards = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    ppo.learn((completions, action_masks, rewards), turn_ids=turn_ids)


@pytest.mark.parametrize("use_vllm", [False, True])
def test_llmppo_learns_multiturn(use_vllm):
    """Multi-turn learn path updates actor/critic adapters without vLLM/DeepSpeed."""
    torch.manual_seed(0)
    ppo = _cpu_llmppo(
        lr_actor=0.05,
        lr_critic=0.05,
        update_epochs=2,
        use_vllm=False,
    )
    if use_vllm:
        ppo.use_vllm = True
        ppo.vllm_config = VLLMConfig(
            gpu_memory_utilization=0.2,
            max_num_seqs=1,
            sleep_mode=True,
        )
        ppo._vllm_awake = True
        ppo.llm = MagicMock()
        ppo.llm.sleep = MagicMock()

    vocab_size = 100
    input_tokens = 10
    generated_tokens = 8
    seq_len = input_tokens + generated_tokens
    batch_size = 2
    completions = [
        torch.randint(0, vocab_size, (1, seq_len), device=ppo.device)
        for _ in range(batch_size)
    ]
    action_masks = [
        torch.ones(1, seq_len - 1, dtype=torch.bool, device=ppo.device)
        for _ in range(batch_size)
    ]

    one_turn_ids = torch.tensor(
        [
            [-1] * (input_tokens - 1)
            + [0] * (generated_tokens // 2)
            + [1] * (generated_tokens - generated_tokens // 2)
        ],
        dtype=torch.long,
        device=ppo.device,
    )[:, : seq_len - 1]
    turn_ids = one_turn_ids.repeat(batch_size, 1)
    rewards = torch.tensor(
        [[1.0, -0.5], [-0.25, 0.75]],
        dtype=torch.float32,
        device=ppo.device,
    )

    with patch.object(
        ppo,
        "_prepare_vllm_for_training",
        wraps=ppo._prepare_vllm_for_training,
    ) as mock_prepare_vllm_for_training:
        ppo.learn((completions, action_masks, rewards), turn_ids=turn_ids)
    assert mock_prepare_vllm_for_training.call_count == 1
    if use_vllm:
        ppo.llm.sleep.assert_called_once()
    pre_learn_actor_state_dict = {
        name: param.clone().detach() for name, param in ppo.actor.named_parameters()
    }

    with patch.object(
        ppo,
        "_prepare_vllm_for_training",
        wraps=ppo._prepare_vllm_for_training,
    ) as mock_prepare_vllm_for_training:
        metrics = ppo.learn((completions, action_masks, rewards), turn_ids=turn_ids)
    assert mock_prepare_vllm_for_training.call_count == 1
    for key in (
        "mean_loss",
        "mean_kl",
        "mean_pg_loss",
        "mean_vf_loss",
        "mean_entropy",
        "mean_clipfrac",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert torch.isfinite(torch.tensor(metrics[key]))

    actor_lora_changed = False
    critic_lora_changed = False
    for param_name, param in ppo.actor.named_parameters():
        before = pre_learn_actor_state_dict[param_name]
        if "reference" in param_name:
            assert torch.equal(param, before), f"{param_name} should not change"
            continue
        if (
            "actor" in param_name
            and "lora" in param_name
            and not torch.equal(param, before)
        ):
            actor_lora_changed = True
        if (
            "critic" in param_name
            and "lora" in param_name
            and not torch.equal(param, before)
        ):
            critic_lora_changed = True

    assert actor_lora_changed, "Expected at least one actor LoRA parameter to update"
    assert critic_lora_changed, "Expected at least one critic LoRA parameter to update"


def test_learn_turn_level_clip_false():
    ppo = _cpu_llmppo(turn_level_clip=False, lr_actor=0.05)
    vocab = 100
    inp, mtok = 10, 8
    seq_len = inp + mtok
    completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
    action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(2)]
    rewards = torch.tensor([1.0, -1.0], dtype=torch.float32).unsqueeze(-1)
    ppo.learn((completions, action_masks, rewards))


def _minimal_reasoning_gym(device: str, vocab_size: int, input_size: int, bs: int):
    env = ReasoningGym.__new__(ReasoningGym)

    @contextmanager
    def eval_mode():
        yield

    env.eval_mode = eval_mode

    def reset(reset_dataloaders=False):
        return {
            "input_ids": torch.randint(0, vocab_size, (bs, input_size), device=device),
            "attention_mask": torch.ones(bs, input_size, device=device),
            "question": [f"q_{i}" for i in range(bs)],
            "answer": [f"a_{i}" for i in range(bs)],
        }

    def step(completion_ids):
        r = torch.ones(bs, device=device)
        return reset(), r

    env.reset = reset
    env.step = step
    return env


def test_test_method_reasoning_gym_branch():
    ppo = _cpu_llmppo()
    env = _minimal_reasoning_gym("cpu", 100, 10, 2)
    out = ppo.test(env, loop=2)
    assert out.shape == ()
    assert out.item() == pytest.approx(1.0)


def test_test_method_multiturn_episode_env_branch():
    class DummyMultiTurnEpisodeEnv:
        max_turns = 2

        def __init__(self):
            self._step_count = 0

        def reset(self, seed=None):
            del seed
            self._step_count = 0
            prompt = {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }
            return prompt, {}

        def step(self, full_completion_ids):
            del full_completion_ids
            self._step_count += 1
            prompt = {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }
            terminated = self._step_count >= 2
            return prompt, 1.0, terminated, False, {}

        def get_episode_data(self):
            return (
                torch.ones(1, 4, dtype=torch.long),
                torch.ones(1, 3, dtype=torch.bool),
                torch.zeros(1, 3, dtype=torch.long),
                torch.tensor([1.0, 1.0], dtype=torch.float32),
            )

        def close(self):
            return None

    ppo = _cpu_llmppo()
    env = DummyMultiTurnEpisodeEnv()
    completion = torch.ones(1, 6, dtype=torch.long)
    action_mask = torch.ones(1, 5, dtype=torch.bool)
    with patch.object(
        ppo, "get_action", return_value=([completion], [action_mask])
    ) as get_action:
        out = ppo.test(env, loop=2)

    assert out.shape == ()
    assert out.item() == pytest.approx(1.0)
    assert get_action.call_count == 4
    assert ppo.fitness[-1] == pytest.approx(1.0)


def test_test_method_unknown_env_typeerror():
    ppo = _cpu_llmppo()
    with pytest.raises(TypeError, match="env must be a ReasoningGym"):
        ppo.test(object(), loop=1)


def test_learn_with_turn_ids_and_1d_reward_vector():
    """When ``turn_ids`` is set and rewards stack to a 1-D tensor, unsqueeze to [B, 1]."""
    ppo = _cpu_llmppo(lr_actor=0.05)
    vocab, inp, mtok = 100, 10, 8
    seq_len = inp + mtok
    slm = seq_len - 1
    completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
    masks = [torch.ones(1, slm, dtype=torch.bool) for _ in range(2)]
    turn_ids = torch.zeros(2, slm, dtype=torch.long)
    rewards = torch.tensor([0.25, -0.25], dtype=torch.float32)
    ppo.learn((completions, masks, rewards), turn_ids=turn_ids)


def test_llmppo_wrap_true_runs_learn():
    """``wrap=True`` with no accelerator still calls :meth:`wrap_models`."""
    actor = create_module(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        modules_to_save=["summary"],
    )
    ppo = LLMPPO(
        actor_network=actor,
        pad_token_id=99,
        pad_token="<pad>",
        lora_config=lora,
        batch_size=2,
        micro_batch_size_per_gpu=2,
        max_output_tokens=8,
        max_model_len=32,
        accelerator=None,
        wrap=True,
        gradient_checkpointing=False,
        use_vllm=False,
        lr_actor=0.05,
        lr_critic=0.05,
        update_epochs=1,
        device="cpu",
        seed=0,
    )
    vocab, inp, mtok = 100, 10, 8
    seq_len = inp + mtok
    completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
    masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(2)]
    rewards = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    ppo.learn((completions, masks, rewards))


def test_llmppo_test_method_token_observation_wrapper_branch():
    from transformers import AutoTokenizer

    from agilerl.utils.probe_envs_llm import ConstantTargetEnv
    from agilerl.llm_envs import TokenObservationWrapper

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    env = TokenObservationWrapper(
        ConstantTargetEnv(target_digit="1", prompt="1"),
        tok,
        max_turns=1,
        pad_id=tok.pad_token_id,
        apply_chat_template=False,
        max_model_len=128,
        max_output_tokens=8,
    )
    ppo = _cpu_llmppo(max_model_len=128, max_output_tokens=8)
    out = ppo.test(env, loop=1)
    assert out.shape == ()
    assert ppo.fitness[-1] == pytest.approx(float(out))
