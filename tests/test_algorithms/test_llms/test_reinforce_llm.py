# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import gc
import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch as unittest_patch

import pytest
import torch

pytest.importorskip("deepspeed", reason="LLM tests require deepspeed.")
pytest.importorskip("vllm", reason="LLM tests require vllm.")

from accelerate.state import AcceleratorState
from peft import LoraConfig
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.reinforce_llm import REINFORCE
from agilerl.llm_envs import RolloutHarness
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig
from tests import TINY_LLM_FIXTURE_PATH
from tests.helpers.patch_algo_core import make_core_patch, setattr_core
from tests.helpers.rollout_doubles import FakeEnvClient, RolloutHarnessDouble
from tests.utils import (
    assert_vllm_get_action_contract,
    make_mock_vllm_instance,
)

patch = make_core_patch(unittest_patch)

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
        # Real ``PreTrainedModel``s expose a ``generation_config`` that the HF
        # ``generate`` path (now reached through the PEFT wrappers) reads. This
        # dummy doesn't inherit ``GenerationMixin`` so transformers skips the
        # auto-init; set it explicitly to mirror a generation-capable model.
        self.generation_config = GenerationConfig.from_model_config(config)
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
        # Pin so the unfused learn() path is exercised by default
        # regardless of liger-kernel availability.
        "use_liger_loss": False,
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
    quantization_config=None,
    vllm_config_overrides=None,
    temperature=1.0,
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
        # See ``tests/test_algorithms/test_llms/test_grpo.py:generate_grpo``
        # for the full rationale. tl;dr both settings are required for
        # parallel vLLM testing: ``kv_cache_memory_bytes`` short-circuits
        # vLLM's profile-snapshot assertion, and ``gpu_memory_utilization``
        # has to stay small (here 0.2 → ~2.9 GiB on the 14.58 GiB CI GPU) so
        # the upfront ``free >= total * utilization`` check in
        # ``vllm/v1/worker/gpu_worker.py:init_device`` passes when peer
        # workers have already claimed their share.
        vllm_config = VLLMConfig(
            gpu_memory_utilization=0.2,
            kv_cache_memory_bytes=32 * 1024 * 1024,
            max_num_seqs=1,
            sleep_mode=sleep_mode,
            **(vllm_config_overrides or {}),
        )
        # ``from_name`` loads the trainer base from the model name (real-engine
        # tests), so no stand-in HF actor is built; otherwise (mocked engine)
        # the dummy actor is the trainer base.
        actor = (
            None
            if from_name
            else model_factory(pretrained_model_name_or_path, add_value_head=False)
        )
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

    # Colocated vLLM and the trainer each hold their own base. The mocked-engine
    # tests pass the dummy actor as the trainer base; ``_initialize_actors`` uses
    # it directly when ``base_model`` is given. ``from_name`` loads the base from
    # the model name instead (real-engine tests).
    share_from_name = from_name
    reinforce_kwargs = {
        "actor_network": actor if not share_from_name else None,
        "model_name": pretrained_model_name_or_path if share_from_name else None,
        "lr": lr_use,
        "pad_token_id": vocab_size - 1,
        "pad_token": "<pad>",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lora_config": lora_config,
        "cosine_lr_schedule_config": (
            None
            if accelerator is not None
            else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
        ),
        "accelerator": accelerator,
        "use_vllm": use_vllm,
        "vllm_config": vllm_config,
        "max_output_tokens": max_tokens,
        "max_model_len": max_tokens + 5,
        "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "use_memory_efficient_params": use_memory_efficient_params,
        "quantization_config": quantization_config,
        "temperature": temperature,
        # Pin so the unfused learn() path is exercised by default
        # regardless of liger-kernel availability.
        "use_liger_loss": False,
    }
    return REINFORCE(**reinforce_kwargs)


@pytest.fixture
def reinforce_factory():
    return generate_reinforce


class _ReinforceStub:
    _compute_token_rewards = REINFORCE._compute_token_rewards


class _RebnStub:
    def __init__(self, gamma: float = 1.0, advantage_granularity: str = "auto"):
        self.gamma = gamma
        self.advantage_granularity = advantage_granularity

    _compute_rebn_advantages = REINFORCE._compute_rebn_advantages
    _compute_rebn_advantages_token = REINFORCE._compute_rebn_advantages_token
    _resolve_advantage_granularity = REINFORCE._resolve_advantage_granularity


def _minimal_reasoning_rollout_env(device: str, vocab_size: int, input_size: int):
    """Single-turn reasoning ``RolloutHarness`` stub (the folded reasoning case)."""

    class _SingleTurnReasoning(RolloutHarnessDouble):
        max_turns = 1

        def __init__(self):
            super().__init__()
            self._env_client = None
            self.done = False

        def _prompt(self):
            return {
                "input_ids": torch.randint(
                    0, vocab_size, (1, input_size), device=device
                ),
                "attention_mask": torch.ones(1, input_size, device=device),
                "text": "q",
            }

        def reset(self, seed=None):
            del seed
            self.done = False
            return self._prompt(), {}

        def step(self, token_ids):
            del token_ids
            self.done = True
            return self._prompt(), 1.0, True, False, {}

        def get_episode_data(self):
            return (
                torch.ones(1, input_size, dtype=torch.long, device=device),
                torch.ones(1, input_size - 1, dtype=torch.bool, device=device),
                torch.zeros(1, input_size - 1, dtype=torch.long, device=device),
                torch.tensor([1.0], dtype=torch.float32, device=device),
            )

        @contextmanager
        def eval_mode(self):
            yield

        def close(self):
            return None

    return _SingleTurnReasoning()


class TestREINFORCEInit:
    @patch("agilerl.algorithms.core.base.LLM")
    def test_init_reinforce_vllm_sleep_mode(self, MockLLM):
        mock_instance = make_mock_vllm_instance()
        MockLLM.return_value = mock_instance

        actor = create_dummy_actor(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
        )
        # Colocated vLLM and the trainer each hold their own base. The vLLM
        # engine is mocked here; the dummy actor is passed as the trainer base
        # (``_initialize_actors`` uses it directly when ``base_model`` is given).
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
        assert rf.use_vllm
        mock_instance.sleep.assert_called()
        rf.clean_up()

    @patch("agilerl.algorithms.core.base.LLM")
    def test_init_reinforce_warns_when_hf_generate_chunk_size_set_with_vllm(
        self, MockLLM
    ):
        mock_instance = make_mock_vllm_instance()
        MockLLM.return_value = mock_instance
        actor = create_dummy_actor(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
        )
        # The vLLM engine is mocked; the dummy actor is the trainer base.
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

    def test_init_rejects_output_tokens_not_less_than_model_len(self):
        actor = create_dummy_actor(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
        )
        with pytest.raises(ValueError, match="must be less than"):
            REINFORCE(
                actor_network=actor,
                pad_token_id=99,
                pad_token="<pad>",
                lora_config=lora,
                max_output_tokens=32,
                max_model_len=16,
                wrap=False,
                gradient_checkpointing=False,
            )

    def test_init_clip_coef_non_negative(self):
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

    def test_init_update_epochs_at_least_one(self):
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

    def test_init_advantage_granularity_must_be_valid(self):
        actor = create_dummy_actor(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
        )
        with pytest.raises(ValueError, match="advantage_granularity"):
            REINFORCE(
                actor_network=actor,
                pad_token_id=99,
                pad_token="<pad>",
                lora_config=lora,
                advantage_granularity="bad",
                wrap=False,
                gradient_checkpointing=False,
            )

    def test_init_chunk_rows_must_be_positive(self):
        actor = create_dummy_actor(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
        )
        with pytest.raises(ValueError, match="chunk_rows must be a positive int"):
            REINFORCE(
                actor_network=actor,
                pad_token_id=99,
                pad_token="<pad>",
                lora_config=lora,
                chunk_rows=0,
                wrap=False,
                gradient_checkpointing=False,
            )

    def test_init_stores_chunk_rows(self):
        rf = _cpu_llmreinforce(chunk_rows=256)
        assert rf.chunk_rows == 256

    def test_init_turn_ratio_pooling_must_be_valid(self):
        actor = create_dummy_actor(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
        )
        with pytest.raises(ValueError, match="turn_ratio_pooling"):
            REINFORCE(
                actor_network=actor,
                pad_token_id=99,
                pad_token="<pad>",
                lora_config=lora,
                turn_ratio_pooling="median",
                wrap=False,
                gradient_checkpointing=False,
            )

    def test_init_default_turn_ratio_pooling_is_sum(self):
        rf = _cpu_llmreinforce()
        assert rf.turn_ratio_pooling == "sum"

    def test_init_action_granularity_deprecated_warns_and_overrides(self):
        """The legacy ``action_granularity`` kwarg warns and is carried over
        into ``advantage_granularity``.
        """
        with pytest.warns(DeprecationWarning, match="action_granularity is deprecated"):
            rf = _cpu_llmreinforce(action_granularity="turn")
        assert rf.advantage_granularity == "turn"

    @pytest.mark.parametrize("is_level", ["turn", "trajectory"])
    def test_init_liger_non_token_is_level_warns_memory_unbounded(
        self, monkeypatch, is_level
    ):
        """Liger + non-token IS is permitted but not memory-bounded; the
        constructor emits the canonical warning once via the base helper.
        """
        setattr_core(monkeypatch, "HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True)
        with pytest.warns(UserWarning, match="NOT memory-bounded"):
            rf = _cpu_llmreinforce(
                use_liger_loss=True, importance_sampling_level=is_level
            )
        assert rf._reinforce_liger_mem_warned is True

    def test_init_clone_requires_pretrained_like_actor(self):
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


class TestREINFORCEGetAction:
    def test_llmreinforce_get_action_vllm_routes_through_vllm_calls(self):
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
            patch.object(rf, "_sync_actor_to_vllm", return_value=None) as mock_move,
            patch.object(
                rf,
                "_generate_with_vllm_colocate",
                return_value=(mocked_ids, mocked_masks, None),
            ) as mock_generate,
        ):
            token_ids, action_masks, _ = rf.get_action(prompts, training=False)

        mock_prepare.assert_called_once()
        mock_move.assert_called_once()
        mock_generate.assert_called_once_with(
            prompts, 1, temperature=0.01, capture_sampling_logps=False
        )
        rf.llm.wake_up.assert_called_once()
        rf._prepare_vllm_for_training()
        rf.llm.sleep.assert_called_once()
        assert token_ids == mocked_ids
        assert action_masks == mocked_masks
        rf.clean_up()

    def test_llmreinforce_get_action_hf_path_contract(self):
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
            token_ids, action_masks, _ = rf.get_action(prompts, training=training)
            assert_vllm_get_action_contract(
                token_ids=token_ids,
                action_masks=action_masks,
                batch_size=len(prompts),
                prompt_len=prompt_len,
                pad_token_id=rf.pad_token_id,
            )
        rf.clean_up()

    def test_llmreinforce_get_action_hf_path_handles_actor_without_parameters(self):
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
            token_ids, action_masks, _ = rf.get_action(prompts, training=True)

        assert_vllm_get_action_contract(
            token_ids=token_ids,
            action_masks=action_masks,
            batch_size=1,
            prompt_len=10,
            pad_token_id=rf.pad_token_id,
        )
        rf.clean_up()


class TestREINFORCEComputeTokenRewards:
    def test_compute_token_rewards_per_turn_reward_broadcasts_to_that_turns_tokens(
        self,
    ):
        stub = _ReinforceStub()
        action_mask = torch.ones(1, 4, dtype=torch.bool)
        turn_ids = torch.tensor([[0, 0, 1, 1]])
        rewards = torch.tensor([[1.0, 2.0]])
        out = stub._compute_token_rewards(action_mask, rewards, turn_ids)
        expected = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        assert torch.allclose(out, expected)

    def test_compute_token_rewards_minus_one_positions_ignore_turn_columns(self):
        stub = _ReinforceStub()
        action_mask = torch.tensor([[True, True, False, False]])
        turn_ids = torch.tensor([[0, -1, -1, -1]])
        rewards = torch.tensor([[3.0]])
        out = stub._compute_token_rewards(action_mask, rewards, turn_ids)
        expected = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(out, expected)


class TestREINFORCEComputeRebnAdvantages:
    def test_compute_rebn_advantages_single_turn_batch_zscore_broadcasts_to_tokens(
        self,
    ):
        stub = _RebnStub(gamma=1.0)
        rewards = torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        action_mask = torch.ones(2, 3, dtype=torch.bool)
        turn_ids = torch.zeros(2, 3, dtype=torch.long)
        adv = stub._compute_rebn_advantages(rewards, action_mask, turn_ids)
        assert torch.allclose(adv[0], torch.full((3,), adv[0, 0]))
        assert torch.allclose(adv[1], torch.full((3,), adv[1, 0]))
        assert adv[0, 0] < 0 < adv[1, 0]
        assert torch.allclose(adv[0, 0].abs(), adv[1, 0].abs())

    def test_compute_rebn_advantages_gamma_changes_advantages(self):
        rewards = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        action_mask = torch.ones(1, 4, dtype=torch.bool)
        turn_ids = torch.tensor([[0, 0, 1, 1]])
        a = _RebnStub(gamma=1.0)._compute_rebn_advantages(
            rewards, action_mask, turn_ids
        )
        b = _RebnStub(gamma=0.5)._compute_rebn_advantages(
            rewards, action_mask, turn_ids
        )
        assert not torch.allclose(a, b)

    def test_compute_rebn_advantages_padding_positions_zero_advantage(self):
        stub = _RebnStub()
        action_mask = torch.tensor([[True, True, False]])
        turn_ids = torch.tensor([[0, 1, -1]])
        rewards = torch.tensor([[1.0, 2.0, 0.0]])
        advantages = stub._compute_rebn_advantages(rewards, action_mask, turn_ids)
        assert advantages[0, 2].item() == 0.0

    def test_compute_rebn_advantages_skips_zscore_when_at_most_one_valid_turn_return(
        self,
    ):
        """Covers the ``valid_returns.numel() <= 1`` branch (no batch z-score)."""
        stub = _RebnStub(gamma=1.0)
        rewards = torch.tensor([[2.0, 2.0, 2.0]])
        action_mask = torch.ones(1, 3, dtype=torch.bool)
        turn_ids = torch.zeros(1, 3, dtype=torch.long)
        advantages = stub._compute_rebn_advantages(rewards, action_mask, turn_ids)
        assert torch.allclose(advantages, torch.zeros_like(advantages))


class TestREINFORCEComputeRebnAdvantagesToken:
    def test_compute_rebn_advantages_token_padding_positions_zero_advantage(self):
        stub = _RebnStub(gamma=0.99)
        rewards = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
        action_mask = torch.tensor([[True, True, False, False]])
        advantages = stub._compute_rebn_advantages_token(rewards, action_mask)
        assert advantages.shape == rewards.shape
        assert torch.allclose(
            advantages[~action_mask], torch.zeros_like(advantages[~action_mask])
        )
        assert not torch.isnan(advantages).any()

    def test_compute_rebn_advantages_token_skips_zscore_when_at_most_one_valid_return(
        self,
    ):
        stub = _RebnStub(gamma=0.99)
        rewards = torch.tensor([[2.0, 0.0, 0.0]])
        action_mask = torch.tensor([[True, False, False]])

        advantages = stub._compute_rebn_advantages_token(rewards, action_mask)

        assert torch.allclose(advantages, torch.zeros_like(advantages))


class TestREINFORCEResolveAdvantageGranularity:
    def test_resolve_advantage_granularity_auto_single_turn_batch_is_token(self):
        stub = _RebnStub(advantage_granularity="auto")
        turn_ids = torch.tensor([[0, 0, -1], [0, -1, -1]])
        assert stub._resolve_advantage_granularity(turn_ids) == "token"

    def test_resolve_advantage_granularity_auto_multi_turn_batch_is_turn(self):
        stub = _RebnStub(advantage_granularity="auto")
        turn_ids = torch.tensor([[0, 1, -1], [0, 0, 1]])
        assert stub._resolve_advantage_granularity(turn_ids) == "turn"

    def test_resolve_advantage_granularity_override_token(self):
        stub = _RebnStub(advantage_granularity="token")
        turn_ids = torch.tensor([[0, 1, -1]])
        assert stub._resolve_advantage_granularity(turn_ids) == "token"


class TestREINFORCELearn:
    def test_learn_multi_turn_explicit_turn_ids(self):
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
    def test_llmreinforce_learns_rollout(self, use_vllm):
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
        for key in ("loss", "kl", "entropy"):
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

        assert actor_lora_changed, (
            "Expected at least one actor LoRA parameter to update"
        )

    def test_learn_token_granularity(self):
        rf = _cpu_llmreinforce(advantage_granularity="token", lr=0.05, update_epochs=1)
        vocab = 100
        inp, mtok = 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(2)]
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32).unsqueeze(-1)
        rf.learn((completions, action_masks, rewards))

    def test_learn_with_turn_ids_and_1d_reward_vector(self):
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

    def test_llmreinforce_wrap_true_runs_learn(self):
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


class TestREINFORCETest:
    def test_test_method_reasoning_rollout_branch(self):
        rf = _cpu_llmreinforce()
        env = _minimal_reasoning_rollout_env("cpu", 100, 10)
        completion = torch.ones(1, 12, dtype=torch.long)
        action_mask = torch.ones(1, 11, dtype=torch.bool)
        with patch.object(
            rf, "get_action", return_value=ActionResult([completion], [action_mask])
        ):
            out = rf.test(env, loop=2)
        assert out.shape == ()
        assert out.item() == pytest.approx(1.0)

    def test_test_method_rollout_episode_env_branch(self):
        class DummyRolloutEpisodeEnv(RolloutHarnessDouble):
            max_turns = 2

            def __init__(self):
                super().__init__()
                self._env_client = FakeEnvClient()
                self.done = False
                self.valid_prompt = {
                    "input_ids": torch.ones(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }

            def reset(self, seed=None):
                del seed
                self.done = False
                return self.valid_prompt, {}

            def step(self, token_ids):
                del token_ids
                self.done = True
                return {}, 1.0, True, False, {}

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
        env = DummyRolloutEpisodeEnv()
        completion = torch.ones(1, 6, dtype=torch.long)
        action_mask = torch.ones(1, 5, dtype=torch.bool)
        with patch.object(
            rf, "get_action", return_value=ActionResult([completion], [action_mask])
        ) as get_action:
            out = rf.test(env, loop=2)

        assert out.shape == ()
        assert out.item() == pytest.approx(1.0)
        # One real turn per episode (early terminate); no dummy padding turns.
        assert get_action.call_count == 2
        for call in get_action.call_args_list:
            assert call.args[0][0] is env.valid_prompt
        assert rf.fitness[-1] == pytest.approx(1.0)

    def test_test_method_waits_for_everyone(self):
        class DummyRolloutEpisodeEnv(RolloutHarnessDouble):
            max_turns = 1

            def __init__(self):
                super().__init__()
                self._env_client = FakeEnvClient()
                self.done = False

            def reset(self, seed=None):
                del seed
                self.done = False
                return {
                    "input_ids": torch.ones(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }, {}

            def step(self, token_ids):
                del token_ids
                self.done = True
                return {}, 1.0, True, False, {}

            def get_episode_data(self):
                return (
                    torch.ones(1, 4, dtype=torch.long),
                    torch.ones(1, 3, dtype=torch.bool),
                    torch.zeros(1, 3, dtype=torch.long),
                    torch.tensor([1.0], dtype=torch.float32),
                )

            def close(self):
                return None

        rf = _cpu_llmreinforce()
        acc = MagicMock()
        rf.accelerator = acc
        completion = torch.ones(1, 6, dtype=torch.long)
        with patch.object(
            rf, "get_action", return_value=ActionResult([completion], None)
        ):
            rf.test(DummyRolloutEpisodeEnv(), loop=1)
        acc.wait_for_everyone.assert_called()

    def test_test_method_rollout_continues_when_not_done(self):
        """Cover prompt update when the episode spans turns."""

        class DummyRolloutContinueEnv(RolloutHarnessDouble):
            max_turns = 2

            def __init__(self):
                super().__init__()
                self._step_count = 0
                self._env_client = FakeEnvClient()
                self.done = False
                self.prompt_a = {
                    "input_ids": torch.ones(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }
                self.prompt_b = {
                    "input_ids": torch.ones(1, 5, dtype=torch.long),
                    "attention_mask": torch.ones(1, 5, dtype=torch.long),
                }

            def reset(self, seed=None):
                del seed
                self._step_count = 0
                self.done = False
                return self.prompt_a, {}

            def step(self, token_ids):
                del token_ids
                self._step_count += 1
                if self._step_count == 1:
                    return self.prompt_b, 0.5, False, False, {}
                self.done = True
                return {}, 1.0, True, False, {}

            def get_episode_data(self):
                return (
                    torch.ones(1, 6, dtype=torch.long),
                    torch.ones(1, 5, dtype=torch.bool),
                    torch.zeros(1, 5, dtype=torch.long),
                    torch.tensor([0.5, 1.0], dtype=torch.float32),
                )

            def close(self):
                return None

        rf = _cpu_llmreinforce()
        completion = torch.ones(1, 6, dtype=torch.long)
        with patch.object(
            rf, "get_action", return_value=ActionResult([completion], None)
        ) as get_action:
            out = rf.test(DummyRolloutContinueEnv(), loop=1)

        assert out.shape == ()
        assert get_action.call_count == 2
        assert get_action.call_args_list[0].args[0][0]["input_ids"].shape[-1] == 4
        assert get_action.call_args_list[1].args[0][0]["input_ids"].shape[-1] == 5

    def test_test_method_unknown_env_typeerror(self):
        rf = _cpu_llmreinforce()
        with pytest.raises(TypeError, match="env must be a RolloutHarness"):
            rf.test(object(), loop=1)

    def test_test_method_token_observation_wrapper_branch(self):
        from transformers import AutoTokenizer

        from agilerl.llm_envs.openenv_server import OpenEnvServer
        from agilerl.utils.probe_envs_llm import ConstantTargetEnv

        tok = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        server = OpenEnvServer(ConstantTargetEnv(target_digit="1", prompt="1")).start()
        try:
            env = RolloutHarness(
                server.base_url,
                tok,
                max_turns=1,
                apply_chat_template=False,
                max_model_len=128,
                max_output_tokens=8,
            )
            rf = _cpu_llmreinforce(max_model_len=128, max_output_tokens=8)
            out = rf.test(env, loop=1)
            assert out.shape == ()
            assert rf.fitness[-1] == pytest.approx(float(out))
        finally:
            server.stop()


class TestReinforceLossLiger:
    """Cover ``_reinforce_loss_liger``. The autograd Function it wraps
    requires ``liger-kernel`` but the wrapper (lm_head pre-hook capture →
    Liger Function call → metric unpack) is testable on CPU via mocks.
    """

    def test_raises_when_liger_unavailable(self) -> None:
        rf = _cpu_llmreinforce()
        ids = torch.randint(0, 50, (2, 5), dtype=torch.long)
        mask = torch.ones(2, 4, dtype=torch.float32)
        old_lp = torch.zeros(2, 4)
        ref_lp = torch.zeros(2, 4)
        adv = torch.zeros(2, 4)

        with patch("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", False):
            with pytest.raises(
                ImportError,
                match=r"Liger REINFORCE loss was requested.*Set use_liger_loss=False",
            ):
                rf._reinforce_loss_liger(ids, mask, old_lp, ref_lp, adv)

    def test_drives_actor_forward_and_unpacks_metrics(self) -> None:
        """End-to-end with mocked Liger Function: actor pre-hook captures
        hidden state, Function returns scalar loss + 4-tuple metrics.
        """
        rf = _cpu_llmreinforce(beta=0.01, clip_coef=0.2)
        B, T = 2, 5
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.ones(B, T - 1, dtype=torch.float32)
        old_lp = torch.zeros(B, T - 1)
        ref_lp = torch.zeros(B, T - 1)
        adv = torch.randn(B, T - 1) * 0.1

        fake_loss = torch.tensor(0.42, requires_grad=True)
        fake_aux = (
            torch.tensor(0.05),  # kl
            torch.tensor(0.15),  # clipfrac
            torch.tensor(0.25),  # pg_loss
            torch.tensor(0.35),  # entropy
        )

        with (
            patch("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.reinforce_llm.apply_fused_policy_loss"
            ) as mock_fn,
        ):
            mock_fn.return_value = (fake_loss, fake_aux)
            loss, metrics = rf._reinforce_loss_liger(
                ids,
                mask,
                old_lp,
                ref_lp,
                adv,
            )

        mock_fn.assert_called_once()
        # REINFORCE always passes beta=0 to Liger (KL is folded into the
        # advantage upstream); kl is reported as a metric only.
        # Find beta in the apply args — it's the 9th positional arg
        # (input, weight, bias, ids, mask, advs, ref_lp, old_lp, beta, ...).
        call_args = mock_fn.call_args.args
        assert call_args[8] == 0.0  # beta

        assert metrics["kl"] == pytest.approx(0.05)
        assert metrics["clipfrac"] == pytest.approx(0.15)
        assert metrics["pg_loss"] == pytest.approx(0.25)
        assert metrics["entropy"] == pytest.approx(0.35)
        assert loss is fake_loss

    def test_token_mode_fuses_vllm_is_ratio(self) -> None:
        """token-level IS with captured vLLM logprobs fuses the clamped
        trainer/vLLM ratio into the kernel via the ``vllm_is_ratio`` kwarg.
        """
        rf = _cpu_llmreinforce(beta=0.0)
        B, T = 2, 6
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.ones(B, T - 1, dtype=torch.float32)
        old_lp = torch.zeros(B, T - 1)
        ref_lp = torch.zeros(B, T - 1)
        adv = torch.randn(B, T - 1) * 0.1
        sampling = old_lp - 0.5  # non-trivial trainer/vLLM mismatch
        fake_aux = tuple(torch.tensor(0.0) for _ in range(4))
        with (
            patch("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.reinforce_llm.apply_fused_policy_loss"
            ) as mock_fn,
        ):
            mock_fn.return_value = (torch.tensor(0.5, requires_grad=True), fake_aux)
            rf._reinforce_loss_liger(
                ids,
                mask,
                old_lp,
                ref_lp,
                adv,
                turn_ids=None,
                sampling_log_probs=sampling,
            )
        ratio = mock_fn.call_args.kwargs["vllm_is_ratio"]
        assert ratio is not None
        assert torch.all(ratio <= rf.vllm_importance_sampling_cap)

    def test_forwards_configured_chunk_rows(self) -> None:
        rf = _cpu_llmreinforce(chunk_rows=123)
        B, T = 2, 5
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.ones(B, T - 1, dtype=torch.float32)
        old_lp = torch.zeros(B, T - 1)
        ref_lp = torch.zeros(B, T - 1)
        adv = torch.randn(B, T - 1) * 0.1
        fake_aux = tuple(torch.tensor(0.0) for _ in range(4))

        with (
            patch("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.reinforce_llm.apply_fused_policy_loss"
            ) as mock_apply,
        ):
            mock_apply.return_value = (torch.tensor(0.4, requires_grad=True), fake_aux)
            rf._reinforce_loss_liger(ids, mask, old_lp, ref_lp, adv)

        assert mock_apply.call_args.kwargs["token_chunk_size"] == 123

    def test_turn_level_requires_turn_ids(self) -> None:
        rf = _cpu_llmreinforce(importance_sampling_level="turn")
        B, T = 2, 5
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.ones(B, T - 1, dtype=torch.float32)
        zeros = torch.zeros(B, T - 1)

        with (
            patch("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True),
            # The non-token-IS memory notice fires before the turn_ids check.
            pytest.warns(UserWarning, match="NOT memory-bounded"),
            pytest.raises(
                ValueError,
                match=r"importance_sampling_level='turn' requires turn_ids",
            ),
        ):
            rf._reinforce_loss_liger(ids, mask, zeros, zeros, zeros, turn_ids=None)

    def test_turn_level_pools_advantages_and_passes_turn_args(self) -> None:
        """Turn-level IS pools the per-token advantages per turn (mean) and
        hands ``turn_ids`` / ``full_turn_mask`` / ``max_turns`` to the fused
        Function.
        """
        rf = _cpu_llmreinforce(importance_sampling_level="turn")
        B, T = 2, 5
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.ones(B, T - 1, dtype=torch.float32)
        zeros = torch.zeros(B, T - 1)
        adv = torch.tensor([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]])
        turn_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.long)
        fake_aux = tuple(torch.tensor(0.0) for _ in range(4))

        with (
            patch("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.reinforce_llm.apply_fused_policy_loss"
            ) as mock_apply,
        ):
            mock_apply.return_value = (torch.tensor(0.4, requires_grad=True), fake_aux)
            with pytest.warns(UserWarning, match="NOT memory-bounded"):
                rf._reinforce_loss_liger(
                    ids, mask, zeros, zeros, adv, turn_ids=turn_ids
                )

        call = mock_apply.call_args
        # Per-turn means: row 0 -> [mean(1, 3), mean(5, 7)]; row 1 ->
        # [2, mean(4, 6, 8)].
        assert torch.allclose(
            call.args[5], torch.tensor([[2.0, 6.0], [2.0, 6.0]]), atol=1e-6
        )
        assert call.args[12] == "turn"
        assert torch.equal(call.kwargs["turn_ids"], turn_ids)
        assert torch.allclose(call.kwargs["full_turn_mask"], torch.ones(2, 2))
        assert call.kwargs["max_turns"] == 2
        assert call.kwargs["turn_log_ratio_reduction"] == "sum"

    def test_trajectory_level_pools_advantages_to_per_sample_scalar(self) -> None:
        """Trajectory-level IS pools the per-token advantages to a masked
        per-completion mean ``(B, 1)``.
        """
        rf = _cpu_llmreinforce(importance_sampling_level="trajectory")
        B, T = 2, 5
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32
        )
        zeros = torch.zeros(B, T - 1)
        adv = torch.tensor([[1.0, 3.0, 5.0, 100.0], [2.0, 4.0, 6.0, 8.0]])
        fake_aux = tuple(torch.tensor(0.0) for _ in range(4))

        with (
            patch("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.reinforce_llm.apply_fused_policy_loss"
            ) as mock_apply,
        ):
            mock_apply.return_value = (torch.tensor(0.4, requires_grad=True), fake_aux)
            with pytest.warns(UserWarning, match="NOT memory-bounded"):
                rf._reinforce_loss_liger(ids, mask, zeros, zeros, adv)

        call = mock_apply.call_args
        # Masked means: row 0 -> (1 + 3 + 5) / 3 = 3; row 1 -> 20 / 4 = 5.
        assert torch.allclose(call.args[5], torch.tensor([[3.0], [5.0]]), atol=1e-6)
        assert call.args[12] == "trajectory"
        assert call.kwargs["turn_ids"] is None


class TestREINFORCELearnWithLiger:
    """Cover the ``if self.use_liger_loss:`` branch inside REINFORCE
    ``learn()``. Stubs ``_reinforce_loss_liger`` to a fake (loss,
    metrics) tuple so the test stays CPU-only.
    """

    def test_learn_use_liger_loss_drives_reinforce_loss_liger(self, monkeypatch):
        # Force the construct-time HAS_LIGER_KERNEL guard to allow
        # ``use_liger_loss=True`` even on environments without
        # ``liger-kernel`` installed.
        setattr_core(monkeypatch, "HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True)
        rf = _cpu_llmreinforce(lr=0.05, update_epochs=1, use_liger_loss=True)
        assert rf.use_liger_loss is True

        fake_loss = torch.tensor(0.3, requires_grad=True)
        fake_metrics = {"kl": 0.05, "entropy": 0.15, "pg_loss": 0.25}
        # Stub the inner loss fn — isolates the use_liger_loss=True
        # branch in learn() from the actor Liger forward.
        rf._reinforce_loss_liger = MagicMock(return_value=(fake_loss, fake_metrics))

        vocab = 100
        inp, mtok = 10, 8
        seq_len = inp + mtok
        b = 1
        completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(b)]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(b)]
        turn_ids = torch.tensor(
            [[-1] * (inp - 1) + [0] * (mtok // 2) + [1] * (mtok - mtok // 2)],
            dtype=torch.long,
        )[:, : seq_len - 1]
        rewards = torch.tensor([[0.5, -0.5]], dtype=torch.float32)

        learn_out = rf.learn((completions, action_masks, rewards), turn_ids=turn_ids)

        assert rf._reinforce_loss_liger.call_count >= 1
        assert learn_out["loss"] == pytest.approx(0.3, rel=1e-6)
        assert learn_out["kl"] == pytest.approx(0.05, rel=1e-6)
        assert learn_out["pg_loss"] == pytest.approx(0.25, rel=1e-6)

    def test_learn_liger_token_with_sampling_logps_uses_fused_kernel(self, monkeypatch):
        """token-level use_liger_loss=True + captured vLLM logprobs: the
        correction is fused into the kernel (``vllm_is_ratio``), so learn()
        keeps the fused path and threads ``sampling_log_probs`` through.
        """
        setattr_core(monkeypatch, "HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True)
        rf = _cpu_llmreinforce(lr=0.05, update_epochs=1, use_liger_loss=True)
        assert rf.importance_sampling_level == "token"
        rf._reinforce_loss_liger = MagicMock(
            return_value=(
                torch.tensor(0.5, requires_grad=True),
                {"kl": 0.1, "entropy": 0.2, "pg_loss": 0.3},
            )
        )
        rf._backward_pass = MagicMock(return_value=None)

        vocab, inp, mtok = 100, 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab, (1, seq_len))]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool)]
        turn_ids = torch.tensor(
            [[-1] * (inp - 1) + [0] * (mtok // 2) + [1] * (mtok - mtok // 2)],
            dtype=torch.long,
        )[:, : seq_len - 1]
        rewards = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        n_act = int(action_masks[0].sum())
        sampling_logps = [torch.full((n_act,), -3.0, dtype=torch.float32)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rf.learn(
                (completions, action_masks, rewards),
                turn_ids=turn_ids,
                sampling_logps=sampling_logps,
            )
        rf._reinforce_loss_liger.assert_called()
        # sampling_log_probs threaded in as the final positional arg.
        assert rf._reinforce_loss_liger.call_args.args[6] is not None
        assert not any(
            "token-level importance sampling" in str(w.message) for w in caught
        )
        assert rf._is_correction_liger_warned is False

    def test_learn_liger_nontoken_with_sampling_logps_warns_and_uses_standard_path(
        self, monkeypatch
    ):
        """turn-level use_liger_loss=True + captured vLLM logprobs: the per-token
        reweight can't be pooled into the turn ratio, so learn() warns once and
        routes the minibatch through the standard PyTorch path.
        """
        setattr_core(monkeypatch, "HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.reinforce_llm.HAS_LIGER_KERNEL", True)
        rf = _cpu_llmreinforce(
            lr=0.05,
            update_epochs=1,
            use_liger_loss=True,
            importance_sampling_level="turn",
        )
        rf._reinforce_loss_liger = MagicMock(
            side_effect=AssertionError("fused path should not run")
        )

        vocab, inp, mtok = 100, 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab, (1, seq_len))]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool)]
        turn_ids = torch.tensor(
            [[-1] * (inp - 1) + [0] * (mtok // 2) + [1] * (mtok - mtok // 2)],
            dtype=torch.long,
        )[:, : seq_len - 1]
        rewards = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        n_act = int(action_masks[0].sum())
        sampling_logps = [torch.full((n_act,), -3.0, dtype=torch.float32)]

        with pytest.warns(
            UserWarning,
            match="only at token-level importance sampling",
        ):
            metrics = rf.learn(
                (completions, action_masks, rewards),
                turn_ids=turn_ids,
                sampling_logps=sampling_logps,
            )
        rf._reinforce_loss_liger.assert_not_called()
        assert rf._is_correction_liger_warned is True
        assert "vllm_is_delta_mean" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))


class TestREINFORCEVllmISCorrection:
    """vLLM sampling-mismatch (truncated-IS) correction wiring across IS levels."""

    @pytest.mark.parametrize("is_level", ["token", "turn", "trajectory"])
    def test_learn_emits_vllm_is_metrics_and_reweights(self, is_level):
        rf = _cpu_llmreinforce(
            importance_sampling_level=is_level, lr=0.05, update_epochs=1
        )
        # use_vllm=False auto-disables the correction in __init__; force it on to
        # exercise the capture/align/metrics/reweight path that the base class now
        # shares with GRPO.
        rf.vllm_importance_sampling_correction = True
        rf.vllm_importance_sampling_cap = 2.0
        vocab, inp, mtok = 100, 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab, (1, seq_len))]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool)]
        turn_ids = torch.tensor(
            [[-1] * (inp - 1) + [0] * (mtok // 2) + [1] * (mtok - mtok // 2)],
            dtype=torch.long,
        )[:, : seq_len - 1]
        rewards = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        n_act = int(action_masks[0].sum())
        # vLLM logprobs deliberately offset from the trainer recompute so the
        # importance ratio != 1 and the reweight is non-trivial.
        sampling_logps = [torch.full((n_act,), -3.0, dtype=torch.float32)]
        metrics = rf.learn(
            (completions, action_masks, rewards),
            turn_ids=turn_ids,
            sampling_logps=sampling_logps,
        )
        for key in ("vllm_is_delta_mean", "vllm_is_ratio_mean"):
            assert key in metrics
            assert isinstance(metrics[key], float)
        assert metrics["vllm_is_ratio_mean"] > 0
        assert torch.isfinite(torch.tensor(metrics["loss"]))


class TestREINFORCESequencePacking:
    """use_sequence_packing flag plumbing (inert on CPU / non-FA2 backends)."""

    def test_flag_stored_and_learn_runs_with_padded_fallback(self):
        rf = _cpu_llmreinforce(use_sequence_packing=True, lr=0.05, update_epochs=1)
        # Stored on the base class; REINFORCE routes its gradient forward through
        # _get_logprobs, which packs when a FlashAttention-2 / FlexAttention
        # backend is present and otherwise (CPU eager) falls back to padded.
        assert rf.use_sequence_packing is True
        vocab, inp, mtok = 100, 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab, (1, seq_len))]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool)]
        turn_ids = torch.tensor(
            [[-1] * (inp - 1) + [0] * (mtok // 2) + [1] * (mtok - mtok // 2)],
            dtype=torch.long,
        )[:, : seq_len - 1]
        rewards = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        metrics = rf.learn((completions, action_masks, rewards), turn_ids=turn_ids)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
