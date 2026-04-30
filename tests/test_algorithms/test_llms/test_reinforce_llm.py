import gc
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("deepspeed", reason="LLM tests require deepspeed.")
pytest.importorskip("vllm", reason="LLM tests require vllm.")

from accelerate.state import AcceleratorState
from peft import LoraConfig
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms.reinforce_llm import REINFORCE
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig
from agilerl.utils.llm_utils import ReasoningGym
from tests import TINY_LLM_FIXTURE_PATH
from tests.utils import (
    assert_vllm_get_action_contract,
    make_mock_vllm_instance,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

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
    """Tiny causal LM wrapped with PEFT for REINFORCE (no value head)."""

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


def create_dummy_actor(input_size, max_tokens, vocab_size, device):
    """Return a bare causal LM; :class:`REINFORCE` applies PEFT via ``lora_config``."""
    cfg = DummyConfig(
        input_size=input_size,
        max_tokens=max_tokens,
        vocab_size=vocab_size,
    )
    return DummyCausalInner(cfg, device=device)


def _cpu_llmreinforce(**kwargs):
    """Small CPU REINFORCE for fast unit tests (PEFT dummy actor, no accelerator)."""
    device = "cpu"
    vocab_size = 100
    input_size = 10
    max_tokens = 8
    actor = create_dummy_actor(input_size, max_tokens, vocab_size, device)
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
        ),
        "batch_size": 4,
        "micro_batch_size_per_gpu": 2,
        "max_output_tokens": max_tokens,
        "max_model_len": input_size + max_tokens + 4,
        "accelerator": None,
        "wrap": False,
        "gradient_checkpointing": False,
        "use_vllm": False,
        "lr": 1e-3,
        "update_epochs": 1,
        "beta": 0.01,
        "seed": 0,
        "device": device,
    }
    defaults.update(kwargs)
    return REINFORCE(**defaults)


def generate_reinforce(
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
    lr=1e-5,
    lr_eff=None,
    sleep_mode=False,
    from_name=False,
    use_memory_efficient_params=False,
):
    lr_use = lr_eff if lr_eff is not None else lr
    gc.collect()
    torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)

    accelerator = accelerator_factory(use_deepspeed_optimizer, config)
    if not use_deepspeed_optimizer and accelerator is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config.pop("optimizer", None)

    if use_vllm:
        lora_config = None
        # kv_cache_memory_bytes pins KV cache size and skips vLLM's memory-
        # profiling assertion, which fails when peer processes on the shared
        # CI GPU release memory mid-init. See test_grpo.generate_grpo for context.
        vllm_config = VLLMConfig(
            gpu_memory_utilization=0.2,
            kv_cache_memory_bytes=32 * 1024 * 1024,
            max_num_seqs=1,
            sleep_mode=sleep_mode,
        )
        actor = model_factory(pretrained_model_name_or_path, add_value_head=False)
    else:
        if pretrained_model_name_or_path is not None:
            actor = model_factory(pretrained_model_name_or_path, add_value_head=False)
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
            actor = DummyCausalInner(
                DummyConfig(
                    input_size=input_size,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                ),
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            target_modules = ["lin"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        vllm_config = None

    reinforce = REINFORCE(
        actor_network=actor if not from_name else None,
        model_name=pretrained_model_name_or_path if from_name else None,
        lr=lr_use,
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
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        use_memory_efficient_params=use_memory_efficient_params,
    )
    return reinforce


@pytest.fixture(scope="function")
def reinforce_factory():
    return generate_reinforce


@patch("agilerl.algorithms.core.base.LLM")
def test_init_reinforce_vllm_sleep_mode(MockLLM):
    mock_instance = make_mock_vllm_instance()
    MockLLM.return_value = mock_instance

    actor = create_dummy_actor(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
    )
    rf = REINFORCE(
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
    # FIXME need to assert the instances are all set up correctly similar to test_grpo.py
    assert rf.use_vllm
    mock_instance.sleep.assert_called()
    rf.clean_up()


@patch("agilerl.algorithms.core.base.LLM")
def test_init_reinforce_warns_when_hf_generate_chunk_size_set_with_vllm(MockLLM):
    mock_instance = make_mock_vllm_instance()
    MockLLM.return_value = mock_instance
    actor = create_dummy_actor(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
    )
    with pytest.warns(
        UserWarning, match="hf_generate_chunk_size.*ignored.*use_vllm=True"
    ):
        rf = REINFORCE(
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
    rf.clean_up()


def test_llmreinforce_get_action_vllm_routes_through_vllm_calls():
    rf = _cpu_llmreinforce(use_vllm=False)
    rf.use_vllm = True
    rf.vllm_config = VLLMConfig(
        gpu_memory_utilization=0.2,
        max_num_seqs=1,
        sleep_mode=True,
    )
    rf._vllm_awake = False
    rf.llm = MagicMock()
    rf.llm.wake_up = MagicMock()
    rf.llm.sleep = MagicMock()
    prompts = [
        {
            "input_ids": torch.randint(0, 100, (1, 10), device=rf.device),
            "attention_mask": torch.ones(1, 10, device=rf.device),
            "text": "Write me a short story about a cat.",
        }
        for _ in range(2)
    ]
    mocked_ids = [
        torch.ones(1, 12, dtype=torch.long, device=rf.device),
        torch.ones(1, 12, dtype=torch.long, device=rf.device),
    ]
    mocked_masks = [
        torch.ones(1, 11, dtype=torch.bool, device=rf.device),
        torch.ones(1, 11, dtype=torch.bool, device=rf.device),
    ]
    with (
        patch.object(
            rf,
            "_prepare_vllm_for_generation",
            wraps=rf._prepare_vllm_for_generation,
        ) as mock_prepare,
        patch.object(rf, "_move_model_to_vllm", return_value=None) as mock_move,
        patch.object(
            rf,
            "_generate_with_vllm_colocate",
            return_value=(mocked_ids, mocked_masks),
        ) as mock_generate,
    ):
        completion_ids, action_masks = rf.get_action(prompts, training=False)

    mock_prepare.assert_called_once()
    mock_move.assert_called_once()
    mock_generate.assert_called_once_with(prompts, 1, temperature=0.01)
    rf.llm.wake_up.assert_called_once()
    rf._prepare_vllm_for_training()
    rf.llm.sleep.assert_called_once()
    assert completion_ids == mocked_ids
    assert action_masks == mocked_masks
    rf.clean_up()


def test_llmreinforce_get_action_hf_path_contract():
    rf = _cpu_llmreinforce(use_vllm=False, max_model_len=128, max_output_tokens=8)
    prompt_len = 10
    prompts = [
        {
            "input_ids": torch.randint(0, 100, (1, prompt_len), device=rf.device),
            "attention_mask": torch.ones(1, prompt_len, device=rf.device),
        }
        for _ in range(3)
    ]
    for training in (True, False):
        completion_ids, action_masks = rf.get_action(prompts, training=training)
        assert_vllm_get_action_contract(
            completion_ids=completion_ids,
            action_masks=action_masks,
            batch_size=len(prompts),
            prompt_len=prompt_len,
            pad_token_id=rf.pad_token_id,
        )
    rf.clean_up()


def test_llmreinforce_get_action_hf_path_handles_actor_without_parameters():
    rf = _cpu_llmreinforce(use_vllm=False, max_model_len=128, max_output_tokens=8)

    class _NoParamModule:
        def parameters(self):
            return iter(())

    prompts = [
        {
            "input_ids": torch.randint(0, 100, (1, 10), device=rf.device),
            "attention_mask": torch.ones(1, 10, device=rf.device),
        }
    ]

    with patch.object(rf, "_get_unwrapped_actor", return_value=_NoParamModule()):
        completion_ids, action_masks = rf.get_action(prompts, training=True)

    assert_vllm_get_action_contract(
        completion_ids=completion_ids,
        action_masks=action_masks,
        batch_size=1,
        prompt_len=10,
        pad_token_id=rf.pad_token_id,
    )
    rf.clean_up()


class _ReinforceStub:
    _compute_token_rewards = REINFORCE._compute_token_rewards


def test_compute_token_rewards_per_turn_reward_broadcasts_to_that_turns_tokens():
    stub = _ReinforceStub()
    action_mask = torch.ones(1, 4, dtype=torch.bool)
    turn_ids = torch.tensor([[0, 0, 1, 1]])
    rewards = torch.tensor([[1.0, 2.0]])
    out = stub._compute_token_rewards(action_mask, rewards, turn_ids)
    expected = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    assert torch.allclose(out, expected)


def test_compute_token_rewards_minus_one_positions_ignore_turn_columns():
    stub = _ReinforceStub()
    action_mask = torch.tensor([[True, True, False, False]])
    turn_ids = torch.tensor([[0, -1, -1, -1]])
    rewards = torch.tensor([[3.0]])
    out = stub._compute_token_rewards(action_mask, rewards, turn_ids)
    expected = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(out, expected)


class _RebnStub:
    def __init__(self, gamma: float = 1.0, action_granularity: str = "auto"):
        self.gamma = gamma
        self.action_granularity = action_granularity

    _compute_rebn_advantages = REINFORCE._compute_rebn_advantages
    _compute_rebn_advantages_token = REINFORCE._compute_rebn_advantages_token
    _resolve_action_granularity = REINFORCE._resolve_action_granularity


def test_compute_rebn_advantages_single_turn_batch_zscore_broadcasts_to_tokens():
    stub = _RebnStub(gamma=1.0)
    rewards = torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    action_mask = torch.ones(2, 3, dtype=torch.bool)
    turn_ids = torch.zeros(2, 3, dtype=torch.long)
    adv = stub._compute_rebn_advantages(rewards, action_mask, turn_ids)
    assert torch.allclose(adv[0], torch.full((3,), adv[0, 0]))
    assert torch.allclose(adv[1], torch.full((3,), adv[1, 0]))
    assert adv[0, 0] < 0 < adv[1, 0]
    assert torch.allclose(adv[0, 0].abs(), adv[1, 0].abs())


def test_compute_rebn_advantages_gamma_changes_advantages():
    rewards = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    action_mask = torch.ones(1, 4, dtype=torch.bool)
    turn_ids = torch.tensor([[0, 0, 1, 1]])
    a = _RebnStub(gamma=1.0)._compute_rebn_advantages(rewards, action_mask, turn_ids)
    b = _RebnStub(gamma=0.5)._compute_rebn_advantages(rewards, action_mask, turn_ids)
    assert not torch.allclose(a, b)


def test_compute_rebn_advantages_padding_positions_zero_advantage():
    stub = _RebnStub()
    action_mask = torch.tensor([[True, True, False]])
    turn_ids = torch.tensor([[0, 1, -1]])
    rewards = torch.tensor([[1.0, 2.0, 0.0]])
    advantages = stub._compute_rebn_advantages(rewards, action_mask, turn_ids)
    assert advantages[0, 2].item() == 0.0


def test_compute_rebn_advantages_skips_zscore_when_at_most_one_valid_turn_return():
    """Covers the ``valid_returns.numel() <= 1`` branch (no batch z-score)."""
    stub = _RebnStub(gamma=1.0)
    rewards = torch.tensor([[2.0, 2.0, 2.0]])
    action_mask = torch.ones(1, 3, dtype=torch.bool)
    turn_ids = torch.zeros(1, 3, dtype=torch.long)
    advantages = stub._compute_rebn_advantages(rewards, action_mask, turn_ids)
    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_compute_rebn_advantages_token_padding_positions_zero_advantage():
    stub = _RebnStub(gamma=0.99)
    rewards = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    action_mask = torch.tensor([[True, True, False, False]])
    advantages = stub._compute_rebn_advantages_token(rewards, action_mask)
    assert advantages.shape == rewards.shape
    assert torch.allclose(
        advantages[~action_mask], torch.zeros_like(advantages[~action_mask])
    )
    assert not torch.isnan(advantages).any()


def test_compute_rebn_advantages_token_skips_zscore_when_at_most_one_valid_return():
    stub = _RebnStub(gamma=0.99)
    rewards = torch.tensor([[2.0, 0.0, 0.0]])
    action_mask = torch.tensor([[True, False, False]])

    advantages = stub._compute_rebn_advantages_token(rewards, action_mask)

    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_resolve_action_granularity_auto_single_turn_batch_is_token():
    stub = _RebnStub(action_granularity="auto")
    turn_ids = torch.tensor([[0, 0, -1], [0, -1, -1]])
    assert stub._resolve_action_granularity(turn_ids) == "token"


def test_resolve_action_granularity_auto_multi_turn_batch_is_turn():
    stub = _RebnStub(action_granularity="auto")
    turn_ids = torch.tensor([[0, 1, -1], [0, 0, 1]])
    assert stub._resolve_action_granularity(turn_ids) == "turn"


def test_resolve_action_granularity_override_token():
    stub = _RebnStub(action_granularity="token")
    turn_ids = torch.tensor([[0, 1, -1]])
    assert stub._resolve_action_granularity(turn_ids) == "token"


def test_init_requires_max_output_or_max_model_len():
    actor = create_dummy_actor(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
    )
    with pytest.raises(ValueError, match="Either max_output_tokens or max_model_len"):
        REINFORCE(
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
    actor = create_dummy_actor(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
    )
    with pytest.raises(AssertionError):
        REINFORCE(
            actor_network=actor,
            pad_token_id=99,
            pad_token="<pad>",
            lora_config=lora,
            clip_coef=-0.1,
            wrap=False,
            gradient_checkpointing=False,
        )


def test_init_update_epochs_at_least_one():
    actor = create_dummy_actor(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
    )
    with pytest.raises(AssertionError):
        REINFORCE(
            actor_network=actor,
            pad_token_id=99,
            pad_token="<pad>",
            lora_config=lora,
            update_epochs=0,
            wrap=False,
            gradient_checkpointing=False,
        )


def test_init_action_granularity_must_be_valid():
    actor = create_dummy_actor(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
    )
    with pytest.raises(ValueError, match="action_granularity"):
        REINFORCE(
            actor_network=actor,
            pad_token_id=99,
            pad_token="<pad>",
            lora_config=lora,
            action_granularity="bad",
            wrap=False,
            gradient_checkpointing=False,
        )


def test_init_clone_requires_pretrained_like_actor():
    with pytest.raises(AssertionError, match="PeftModelProtocol"):
        REINFORCE(
            model_name=TINY_LLM_FIXTURE_PATH,
            actor_network=object(),
            pad_token_id=99,
            pad_token="<pad>",
            clone=True,
            wrap=False,
            gradient_checkpointing=False,
        )


def test_learn_multi_turn_explicit_turn_ids():
    rf = _cpu_llmreinforce(lr=0.05, update_epochs=2)
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
    rf.learn((completions, action_masks, rewards), turn_ids=turn_ids)


@pytest.mark.parametrize("use_vllm", [False, True])
def test_llmreinforce_learns_multiturn(use_vllm):
    """Multi-turn learn path updates actor adapters without vLLM/DeepSpeed."""
    torch.manual_seed(0)
    rf = _cpu_llmreinforce(
        lr=0.05,
        update_epochs=2,
        use_vllm=False,
    )
    if use_vllm:
        rf.use_vllm = True
        rf.vllm_config = VLLMConfig(
            gpu_memory_utilization=0.2,
            max_num_seqs=1,
            sleep_mode=True,
        )
        rf._vllm_awake = True
        rf.llm = MagicMock()
        rf.llm.sleep = MagicMock()

    vocab_size = 100
    input_tokens = 10
    generated_tokens = 8
    seq_len = input_tokens + generated_tokens
    batch_size = 2
    completions = [
        torch.randint(0, vocab_size, (1, seq_len), device=rf.device)
        for _ in range(batch_size)
    ]
    action_masks = [
        torch.ones(1, seq_len - 1, dtype=torch.bool, device=rf.device)
        for _ in range(batch_size)
    ]

    one_turn_ids = torch.tensor(
        [
            [-1] * (input_tokens - 1)
            + [0] * (generated_tokens // 2)
            + [1] * (generated_tokens - generated_tokens // 2)
        ],
        dtype=torch.long,
        device=rf.device,
    )[:, : seq_len - 1]
    turn_ids = one_turn_ids.repeat(batch_size, 1)
    rewards = torch.tensor(
        [[1.0, -0.5], [-0.25, 0.75]],
        dtype=torch.float32,
        device=rf.device,
    )

    with (
        patch.object(
            rf,
            "_prepare_vllm_for_training",
            wraps=rf._prepare_vllm_for_training,  # FIXME we dont want to be using actual vllm calls here - needs to be mocked
        ) as mock_prepare_vllm_for_training
    ):
        rf.learn((completions, action_masks, rewards), turn_ids=turn_ids)
    assert mock_prepare_vllm_for_training.call_count == 1
    if use_vllm:
        rf.llm.sleep.assert_called_once()
    pre_learn_actor_state_dict = {
        name: param.clone().detach() for name, param in rf.actor.named_parameters()
    }

    with patch.object(
        rf,
        "_prepare_vllm_for_training",
        wraps=rf._prepare_vllm_for_training,
    ) as mock_prepare_vllm_for_training:
        metrics = rf.learn((completions, action_masks, rewards), turn_ids=turn_ids)
    assert mock_prepare_vllm_for_training.call_count == 1
    for key in ("mean_loss", "mean_kl", "mean_pg_loss", "mean_entropy"):
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert torch.isfinite(torch.tensor(metrics[key]))

    actor_lora_changed = False
    for param_name, param in rf.actor.named_parameters():
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

    assert actor_lora_changed, "Expected at least one actor LoRA parameter to update"


def test_learn_token_granularity():
    rf = _cpu_llmreinforce(action_granularity="token", lr=0.05, update_epochs=1)
    vocab = 100
    inp, mtok = 10, 8
    seq_len = inp + mtok
    completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
    action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(2)]
    rewards = torch.tensor([1.0, -1.0], dtype=torch.float32).unsqueeze(-1)
    rf.learn((completions, action_masks, rewards))


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
    rf = _cpu_llmreinforce()
    env = _minimal_reasoning_gym("cpu", 100, 10, 2)
    out = rf.test(env, loop=2)
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

    rf = _cpu_llmreinforce()
    env = DummyMultiTurnEpisodeEnv()
    completion = torch.ones(1, 6, dtype=torch.long)
    action_mask = torch.ones(1, 5, dtype=torch.bool)
    with patch.object(
        rf, "get_action", return_value=([completion], [action_mask])
    ) as get_action:
        out = rf.test(env, loop=2)

    assert out.shape == ()
    assert out.item() == pytest.approx(1.0)
    assert get_action.call_count == 4
    assert rf.fitness[-1] == pytest.approx(1.0)


def test_test_method_unknown_env_typeerror():
    rf = _cpu_llmreinforce()
    with pytest.raises(TypeError, match="env must be a ReasoningGym"):
        rf.test(object(), loop=1)


def test_learn_with_turn_ids_and_1d_reward_vector():
    """When ``turn_ids`` is set and rewards stack to a 1-D tensor, unsqueeze to [B, 1]."""
    rf = _cpu_llmreinforce(lr=0.05)
    vocab, inp, mtok = 100, 10, 8
    seq_len = inp + mtok
    slm = seq_len - 1
    completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
    masks = [torch.ones(1, slm, dtype=torch.bool) for _ in range(2)]
    turn_ids = torch.zeros(2, slm, dtype=torch.long)
    rewards = torch.tensor([0.25, -0.25], dtype=torch.float32)
    rf.learn((completions, masks, rewards), turn_ids=turn_ids)


def test_llmreinforce_wrap_true_runs_learn():
    """``wrap=True`` with no accelerator still calls :meth:`wrap_models`."""
    actor = create_dummy_actor(10, 8, 100, "cpu")
    lora = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["lin"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    rf = REINFORCE(
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
        lr=0.05,
        update_epochs=1,
        device="cpu",
        seed=0,
    )
    vocab, inp, mtok = 100, 10, 8
    seq_len = inp + mtok
    completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
    masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(2)]
    rewards = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    rf.learn((completions, masks, rewards))


def test_test_method_token_observation_wrapper_branch():
    from transformers import AutoTokenizer

    from agilerl.utils.probe_envs_llm import ConstantTargetEnv
    from agilerl.llm_envs import TokenObservationWrapper

    tok = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
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
    rf = _cpu_llmreinforce(max_model_len=128, max_output_tokens=8)
    out = rf.test(env, loop=1)
    assert out.shape == ()
    assert rf.fitness[-1] == pytest.approx(float(out))
