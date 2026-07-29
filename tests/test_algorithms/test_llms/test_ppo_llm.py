# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import gc
import warnings
from contextlib import contextmanager, nullcontext
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
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig
from agilerl.utils.llm_utils import ReasoningGym, masked_whiten
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from tests import TINY_LLM_FIXTURE_PATH
from tests.utils import (
    assert_vllm_get_action_contract,
    make_mock_vllm_instance,
    spawn_new_process_for_each_test,
)

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
    :class:`~agilerl.utils.ppo_value_head.AutoModelForCausalLMWithValueHead`.
    """

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
        # Pin to False so the unfused learn() path is exercised by default
        # regardless of whether liger-kernel is installed. Liger-specific
        # tests override this to True.
        "use_liger_loss": False,
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

    return LLMPPO(
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
        # Pin so the unfused learn() path is exercised by default
        # regardless of liger-kernel availability.
        use_liger_loss=False,
    )


@pytest.fixture
def ppo_factory():
    return generate_ppo


class _PPOStub:
    def __init__(
        self,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        turn_value_reduction: str = "mean",
        advantage_granularity: str = "auto",
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        whiten_advantages: bool = True,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.turn_value_reduction = turn_value_reduction
        self.advantage_granularity = advantage_granularity
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.whiten_advantages = whiten_advantages

    _compute_token_rewards = LLMPPO._compute_token_rewards
    _compute_gae_returns = LLMPPO._compute_gae_returns
    _compute_gae_returns_token = LLMPPO._compute_gae_returns_token
    _resolve_advantage_granularity = LLMPPO._resolve_advantage_granularity


class TestPPOInit:
    def test_init_auto_detects_device_when_none_given(self):
        """Regression: no ``device`` must auto-detect, not silently fall back to CPU."""
        with patch(
            "agilerl.algorithms.ppo_llm.resolve_llm_device", return_value="cpu"
        ) as mock_resolve:
            ppo = _cpu_llmppo(device=None)

        mock_resolve.assert_called_once_with(None, None)
        assert ppo.device == "cpu"

    def test_init_honours_an_explicitly_requested_device(self):
        """An explicit ``device`` is used as-is when no accelerator is present."""
        with patch("torch.cuda.is_available", return_value=True):
            ppo = _cpu_llmppo(device="cpu")

        assert ppo.device == "cpu"
        assert all(param.device.type == "cpu" for param in ppo.actor.parameters())

    @patch("agilerl.algorithms.core.base.LLM")
    def test_init_llmppo_vllm_sleep_mode_calls_sleep(self, MockLLM):
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
        # Colocated vLLM and the trainer each hold their own base. The vLLM
        # engine is mocked here; the dummy actor is passed as the trainer base
        # (``_initialize_actors`` uses it directly when ``base_model`` is given).
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
    def test_init_llmppo_warns_when_hf_generate_chunk_size_set_with_vllm(self, MockLLM):
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
        # The vLLM engine is mocked; the dummy actor is the trainer base.
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

    def test_init_rejects_output_tokens_not_less_than_model_len(self):
        actor = create_module(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
            modules_to_save=["summary"],
        )
        with pytest.raises(ValueError, match="must be less than"):
            LLMPPO(
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

    def test_init_update_epochs_at_least_one(self):
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

    def test_init_advantage_granularity_must_be_valid(self):
        actor = create_module(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
            modules_to_save=["summary"],
        )
        with pytest.raises(ValueError, match="advantage_granularity"):
            LLMPPO(
                actor_network=actor,
                pad_token_id=99,
                pad_token="<pad>",
                lora_config=lora,
                advantage_granularity="bad",
                wrap=False,
                gradient_checkpointing=False,
            )

    def test_init_chunk_rows_must_be_positive(self):
        actor = create_module(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
            modules_to_save=["summary"],
        )
        with pytest.raises(ValueError, match="chunk_rows must be a positive int"):
            LLMPPO(
                actor_network=actor,
                pad_token_id=99,
                pad_token="<pad>",
                lora_config=lora,
                chunk_rows=0,
                wrap=False,
                gradient_checkpointing=False,
            )

    def test_init_stores_chunk_rows(self):
        ppo = _cpu_llmppo(chunk_rows=256)
        assert ppo.chunk_rows == 256

    def test_init_action_granularity_deprecated_warns_and_overrides(self):
        """The legacy ``action_granularity`` kwarg warns and is carried over
        into ``advantage_granularity``.
        """
        with pytest.warns(DeprecationWarning, match="action_granularity is deprecated"):
            ppo = _cpu_llmppo(action_granularity="turn")
        assert ppo.advantage_granularity == "turn"

    @pytest.mark.parametrize("is_level", ["turn", "trajectory"])
    def test_init_liger_non_token_is_level_warns_memory_unbounded(
        self, monkeypatch, is_level
    ):
        """Liger + an explicit non-token IS level is permitted but not
        memory-bounded; the constructor emits the canonical warning once via
        the base helper.
        """
        monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", True)
        with pytest.warns(UserWarning, match="NOT memory-bounded"):
            ppo = _cpu_llmppo(use_liger_loss=True, importance_sampling_level=is_level)
        assert ppo._ppo_liger_mem_warned is True

    def test_init_turn_value_reduction_must_be_valid(self):
        actor = create_module(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
            modules_to_save=["summary"],
        )
        with pytest.raises(ValueError, match="turn_value_reduction"):
            LLMPPO(
                actor_network=actor,
                pad_token_id=99,
                pad_token="<pad>",
                lora_config=lora,
                turn_value_reduction="median",
                wrap=False,
                gradient_checkpointing=False,
            )

    def test_init_turn_ratio_pooling_must_be_valid(self):
        actor = create_module(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
            modules_to_save=["summary"],
        )
        with pytest.raises(ValueError, match="turn_ratio_pooling"):
            LLMPPO(
                actor_network=actor,
                pad_token_id=99,
                pad_token="<pad>",
                lora_config=lora,
                turn_ratio_pooling="median",
                wrap=False,
                gradient_checkpointing=False,
            )

    def test_init_default_turn_ratio_pooling_is_sum(self):
        ppo = _cpu_llmppo()
        assert ppo.turn_ratio_pooling == "sum"

    def test_init_whiten_advantages_must_be_boolean(self):
        actor = create_module(10, 8, 100, "cpu")
        lora = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["lin"],
            task_type="CAUSAL_LM",
            modules_to_save=["summary"],
        )
        with pytest.raises(TypeError, match="whiten_advantages must be a boolean"):
            LLMPPO(
                actor_network=actor,
                pad_token_id=99,
                pad_token="<pad>",
                lora_config=lora,
                whiten_advantages="yes",  # type: ignore[arg-type]
                wrap=False,
                gradient_checkpointing=False,
            )

    def test_init_clone_requires_pretrained_like_actor(self):
        with pytest.raises(AssertionError, match="PeftModelProtocol"):
            LLMPPO(
                model_name=TINY_LLM_FIXTURE_PATH,
                actor_network=object(),
                pad_token_id=99,
                pad_token="<pad>",
                clone=True,
                wrap=False,
                gradient_checkpointing=False,
            )


class TestPPOGetAction:
    def test_llmppo_get_action_vllm_routes_through_vllm_calls(self):
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
            patch.object(ppo, "_sync_actor_to_vllm", return_value=None) as mock_move,
            patch.object(
                ppo,
                "_generate_with_vllm_colocate",
                return_value=(mocked_ids, mocked_masks, None),
            ) as mock_generate,
        ):
            completion_ids, action_masks, _ = ppo.get_action(prompts, training=False)

        mock_prepare.assert_called_once()
        mock_move.assert_called_once()
        mock_generate.assert_called_once_with(
            prompts, 1, temperature=0.01, capture_sampling_logps=False
        )
        ppo.llm.wake_up.assert_called_once()
        ppo._prepare_vllm_for_training()
        ppo.llm.sleep.assert_called_once()
        assert completion_ids == mocked_ids
        assert action_masks == mocked_masks
        ppo.clean_up()

    def test_llmppo_get_action_hf_path_contract(self):
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
            completion_ids, action_masks, _ = ppo.get_action(prompts, training=training)
            assert_vllm_get_action_contract(
                completion_ids=completion_ids,
                action_masks=action_masks,
                batch_size=batch_size,
                prompt_len=prompt_len,
                pad_token_id=ppo.pad_token_id,
            )
        ppo.clean_up()

    def test_llmppo_get_action_hf_path_handles_actor_without_parameters(self):
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
            completion_ids, action_masks, _ = ppo.get_action(prompts, training=True)

        assert_vllm_get_action_contract(
            completion_ids=completion_ids,
            action_masks=action_masks,
            batch_size=1,
            prompt_len=10,
            pad_token_id=ppo.pad_token_id,
        )
        ppo.clean_up()


class TestPPOComputeTokenRewards:
    def test_compute_token_rewards_per_turn_reward_broadcasts_to_that_turns_tokens(
        self,
    ):
        stub = _PPOStub()
        action_mask = torch.ones(1, 4, dtype=torch.bool)
        turn_ids = torch.tensor([[0, 0, 1, 1]])
        rewards = torch.tensor([[1.0, 2.0]])
        out = stub._compute_token_rewards(action_mask, rewards, turn_ids)
        expected = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        assert torch.allclose(out, expected)

    def test_compute_token_rewards_minus_one_positions_ignore_turn_columns(self):
        stub = _PPOStub()
        action_mask = torch.tensor([[True, True, False, False]])
        turn_ids = torch.tensor([[0, 0, -1, -1]])
        rewards = torch.tensor([[3.0]])
        out = stub._compute_token_rewards(action_mask, rewards, turn_ids)
        expected = torch.tensor([[3.0, 3.0, 0.0, 0.0]])
        assert torch.allclose(out, expected)


class TestPPOComputeGaeReturns:
    def test_compute_gae_returns_two_turns_manual_matches_reference(self):
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

    def test_compute_gae_returns_padding_positions_zero_advantage(self):
        stub = _PPOStub()
        action_mask = torch.tensor([[True, True, False]])
        turn_ids = torch.tensor([[0, 1, -1]])
        values = torch.zeros(1, 3)
        rewards = torch.tensor([[1.0, 2.0, 0.0]])
        _returns, advantages = stub._compute_gae_returns(
            rewards, values, action_mask, turn_ids
        )
        assert advantages[0, 2].item() == 0.0

    def test_compute_gae_returns_turn_value_reduction_final_value_uses_last_token_value(
        self,
    ):
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

        expected_advantages = torch.tensor(
            [[0.70710677, 0.70710677, -0.70710677, -0.70710677]]
        )
        expected_returns = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(advantages, expected_advantages, atol=1e-6)
        assert torch.allclose(returns, expected_returns)

    def test_compute_gae_returns_without_whitening_uses_raw_turn_advantages(self):
        stub = _PPOStub(gamma=1.0, gae_lambda=1.0, whiten_advantages=False)
        action_mask = torch.ones(1, 2, dtype=torch.bool)
        turn_ids = torch.tensor([[0, 1]])
        values = torch.tensor([[0.0, 0.0]])
        rewards = torch.tensor([[1.0, 2.0]])

        returns, advantages = stub._compute_gae_returns(
            rewards, values, action_mask, turn_ids
        )

        expected_advantages = torch.tensor([[3.0, 2.0]])
        expected_returns = torch.tensor([[3.0, 2.0]])
        assert torch.allclose(advantages, expected_advantages)
        assert torch.allclose(returns, expected_returns)


class TestPPOComputeGaeReturnsToken:
    def test_compute_gae_returns_token_padding_positions_zero_advantage(self):
        stub = _PPOStub(gamma=0.99, gae_lambda=0.95)
        action_mask = torch.tensor([[True, True, False, False]])
        values = torch.tensor([[0.1, 0.2, 0.0, 0.0]])
        rewards = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
        returns, advantages = stub._compute_gae_returns_token(
            rewards, values, action_mask
        )
        assert returns.shape == values.shape
        assert advantages.shape == values.shape
        assert torch.allclose(
            returns[~action_mask], torch.zeros_like(returns[~action_mask])
        )
        assert torch.allclose(
            advantages[~action_mask], torch.zeros_like(advantages[~action_mask])
        )
        assert not torch.isnan(advantages).any()


class TestPPOResolveAdvantageGranularity:
    def test_resolve_advantage_granularity_auto_single_turn_batch_is_token(self):
        stub = _PPOStub(advantage_granularity="auto")
        turn_ids = torch.tensor([[0, 0, -1], [0, -1, -1]])
        assert stub._resolve_advantage_granularity(turn_ids) == "token"

    def test_resolve_advantage_granularity_auto_multi_turn_batch_is_turn(self):
        stub = _PPOStub(advantage_granularity="auto")
        turn_ids = torch.tensor([[0, 1, -1], [0, 0, 1]])
        assert stub._resolve_advantage_granularity(turn_ids) == "turn"

    def test_resolve_advantage_granularity_override_token(self):
        stub = _PPOStub(advantage_granularity="token")
        turn_ids = torch.tensor([[0, 1, -1]])
        assert stub._resolve_advantage_granularity(turn_ids) == "token"


class TestPPOLearn:
    def test_learn_multi_turn_explicit_turn_ids(self):
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
    def test_llmppo_learns_multiturn(self, use_vllm):
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
            "loss",
            "kl",
            "pg_loss",
            "vf_loss",
            "entropy",
            "clipfrac",
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

        assert actor_lora_changed, (
            "Expected at least one actor LoRA parameter to update"
        )
        assert critic_lora_changed, (
            "Expected at least one critic LoRA parameter to update"
        )

    def test_learn_turn_level_clip_false(self):
        ppo = _cpu_llmppo(turn_level_clip=False, lr_actor=0.05)
        vocab = 100
        inp, mtok = 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(2)]
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32).unsqueeze(-1)
        ppo.learn((completions, action_masks, rewards))

    def test_learn_turn_granularity_turn_level_clip_false(self):
        ppo = _cpu_llmppo(
            advantage_granularity="turn", turn_level_clip=False, lr_actor=0.05
        )
        vocab = 100
        inp, mtok = 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(2)]
        one_turn_ids = torch.tensor(
            [[-1] * (inp - 1) + [0] * (mtok // 2) + [1] * (mtok - mtok // 2)],
            dtype=torch.long,
        )[:, : seq_len - 1]
        turn_ids = one_turn_ids.repeat(2, 1)
        rewards = torch.tensor([[1.0, -1.0], [0.5, -0.25]], dtype=torch.float32)

        metrics = ppo.learn((completions, action_masks, rewards), turn_ids=turn_ids)

        assert "loss" in metrics

    def test_learn_token_granularity(self):
        ppo = _cpu_llmppo(advantage_granularity="token", lr_actor=0.05)
        vocab = 100
        inp, mtok = 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(2)]
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32).unsqueeze(-1)
        ppo.learn((completions, action_masks, rewards))

    def test_learn_with_turn_ids_and_1d_reward_vector(self):
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

    def test_llmppo_wrap_true_runs_learn(self):
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


class TestPPOFusedNoGradBaseRoutedReference:
    """With ``use_separate_reference_adapter=False``, ``_fused_forward_no_grad``
    routes the reference rows to PEFT's reserved ``"__base__"`` adapter inside
    the same fused pass (no separate disable-adapter forward). The reference
    log-probs must match the disable-adapter pass the previous implementation
    ran separately.
    """

    def _perturbed_agent(self):
        torch.manual_seed(0)
        ppo = _cpu_llmppo(use_separate_reference_adapter=False)
        # lora_B starts at zero, so adapter outputs equal the base output and
        # any routing mistake would be invisible. Perturb the trainable LoRA
        # weights so actor/critic rows genuinely differ from base rows.
        with torch.no_grad():
            for name, param in ppo.actor.named_parameters():
                if "lora" in name:
                    param.add_(torch.randn_like(param) * 0.5)
        return ppo

    def test_base_routed_reference_matches_disable_adapter_pass(self):
        ppo = self._perturbed_agent()
        vocab_size, seq_len, b = 100, 12, 4
        ids = torch.randint(1, vocab_size - 1, (b, seq_len))

        ref_lp, actor_lp, values = ppo._fused_forward_no_grad(ids, batch_size=2)

        with torch.no_grad():
            expected_ref = ppo._get_logprobs(
                ids, batch_size=2, use_reference=True, eval_mode=True
            )

        assert ref_lp.shape == (b, seq_len - 1)
        assert actor_lp.shape == (b, seq_len - 1)
        assert values.shape == (b, seq_len - 1)
        assert torch.allclose(ref_lp, expected_ref, rtol=1e-5, atol=1e-6)
        # Sanity: per-row routing is live — actor rows carry the perturbed
        # adapter, so they must differ from the base-routed reference rows.
        assert not torch.allclose(ref_lp, actor_lp, rtol=1e-3, atol=1e-3)

    def test_learn_runs_with_base_routed_reference(self):
        ppo = self._perturbed_agent()
        vocab_size = 100
        inp, mtok = 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab_size, (1, seq_len)) for _ in range(2)]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(2)]
        rewards = torch.tensor([[0.5], [-0.5]], dtype=torch.float32)

        metrics = ppo.learn((completions, action_masks, rewards))

        for key in ("loss", "kl", "pg_loss", "vf_loss"):
            assert torch.isfinite(torch.tensor(metrics[key]))


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


class TestPPOTest:
    def test_test_method_reasoning_gym_branch(self):
        ppo = _cpu_llmppo()
        env = _minimal_reasoning_gym("cpu", 100, 10, 2)
        out = ppo.test(env, loop=2)
        assert out.shape == ()
        assert out.item() == pytest.approx(1.0)

    def test_test_method_multiturn_episode_env_branch(self):
        class DummyMultiTurnEpisodeEnv:
            max_turns = 2

            def __init__(self):
                self._step_count = 0
                self.valid_prompt = {
                    "input_ids": torch.ones(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }

            def reset(self, seed=None):
                del seed
                self._step_count = 0
                return self.valid_prompt, {}

            def step(self, full_completion_ids):
                del full_completion_ids
                self._step_count += 1
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

        ppo = _cpu_llmppo()
        env = DummyMultiTurnEpisodeEnv()
        completion = torch.ones(1, 6, dtype=torch.long)
        action_mask = torch.ones(1, 5, dtype=torch.bool)
        with patch.object(
            ppo, "get_action", return_value=ActionResult([completion], [action_mask])
        ) as get_action:
            out = ppo.test(env, loop=2)

        assert out.shape == ()
        assert out.item() == pytest.approx(1.0)
        # One real turn per episode (early terminate); no dummy padding turns.
        assert get_action.call_count == 2
        for call in get_action.call_args_list:
            assert call.args[0][0] is env.valid_prompt
        assert ppo.fitness[-1] == pytest.approx(1.0)

    def test_test_method_waits_for_everyone(self):
        class DummyMultiTurnEpisodeEnv:
            max_turns = 1

            def reset(self, seed=None):
                del seed
                return {
                    "input_ids": torch.ones(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }, {}

            def step(self, full_completion_ids):
                del full_completion_ids
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

        ppo = _cpu_llmppo()
        acc = MagicMock()
        ppo.accelerator = acc
        completion = torch.ones(1, 6, dtype=torch.long)
        with patch.object(
            ppo, "get_action", return_value=ActionResult([completion], None)
        ):
            ppo.test(DummyMultiTurnEpisodeEnv(), loop=1)
        acc.wait_for_everyone.assert_called()

    def test_test_method_multiturn_continues_when_not_done(self):
        """Cover prompt update when the episode spans turns."""

        class DummyMultiTurnContinueEnv:
            max_turns = 2

            def __init__(self):
                self._step_count = 0
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
                return self.prompt_a, {}

            def step(self, full_completion_ids):
                del full_completion_ids
                self._step_count += 1
                if self._step_count == 1:
                    return self.prompt_b, 0.5, False, False, {}
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

        ppo = _cpu_llmppo()
        completion = torch.ones(1, 6, dtype=torch.long)
        with patch.object(
            ppo, "get_action", return_value=ActionResult([completion], None)
        ) as get_action:
            out = ppo.test(DummyMultiTurnContinueEnv(), loop=1)

        assert out.shape == ()
        assert get_action.call_count == 2
        assert get_action.call_args_list[0].args[0][0]["input_ids"].shape[-1] == 4
        assert get_action.call_args_list[1].args[0][0]["input_ids"].shape[-1] == 5

    def test_test_method_unknown_env_typeerror(self):
        ppo = _cpu_llmppo()
        with pytest.raises(TypeError, match="env must be a ReasoningGym"):
            ppo.test(object(), loop=1)

    def test_llmppo_test_method_token_observation_wrapper_branch(self):
        from transformers import AutoTokenizer

        from agilerl.llm_envs import TokenObservationWrapper
        from agilerl.utils.probe_envs_llm import ConstantTargetEnv

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
        ppo = _cpu_llmppo(max_model_len=128, max_output_tokens=8)
        out = ppo.test(env, loop=1)
        assert out.shape == ()
        assert ppo.fitness[-1] == pytest.approx(float(out))


class TestPPOLossLiger:
    """Cover the fused-linear PPO loss method ``_ppo_loss_liger``. The
    autograd Function it wraps requires ``liger-kernel``, but the wrapper
    itself (build args → forward → unpack metrics → critic value loss) is
    testable via a mocked Liger Function on CPU.
    """

    def test_raises_when_liger_unavailable(self) -> None:
        """``_ppo_loss_liger`` raises ImportError when HAS_LIGER_KERNEL is
        False, with a message instructing the user to disable the flag.
        """
        ppo = _cpu_llmppo()
        ids = torch.randint(0, 50, (2, 5), dtype=torch.long)
        mask = torch.ones(2, 4, dtype=torch.float32)
        old_lp = torch.zeros(2, 4)
        ref_lp = torch.zeros(2, 4)
        returns = torch.zeros(2, 4)
        adv = torch.zeros(2, 4)
        old_values = torch.zeros(2, 4)
        turn_ids = torch.zeros(2, 4, dtype=torch.long)

        with patch("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", False):
            with pytest.raises(
                ImportError,
                match=r"Liger PPO loss was requested.*Set use_liger_loss=False",
            ):
                ppo._ppo_loss_liger(
                    ids,
                    mask,
                    old_lp,
                    ref_lp,
                    returns,
                    adv,
                    old_values,
                    turn_ids,
                    "token",
                )

    def test_token_mode_drives_actor_and_critic_forwards(self) -> None:
        """End-to-end: with the Liger Function mocked, ``_ppo_loss_liger``
        runs both the actor pre-hook capture and the critic forward, and
        returns the right metric dict shape.
        """
        ppo = _cpu_llmppo(beta=0.01, clip_coef=0.2, vf_coef=0.5)
        B, T = 2, 5
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.ones(B, T - 1, dtype=torch.float32)
        old_lp = torch.zeros(B, T - 1)
        ref_lp = torch.zeros(B, T - 1)
        returns = torch.zeros(B, T - 1)
        adv = torch.randn(B, T - 1) * 0.1
        old_values = torch.zeros(B, T - 1)
        turn_ids = torch.zeros(B, T - 1, dtype=torch.long)

        # Mock the fused-loss entry point so we don't need liger-kernel
        # installed. ``_ppo_loss_liger`` calls ``apply_fused_policy_loss`` (which
        # wraps ``LigerFusedLinearPolicyLossFunction.apply``), so patch the
        # wrapper. Returns a scalar loss and the four metric scalars the wrapper
        # unpacks.
        fake_loss = torch.tensor(0.5, requires_grad=True)
        fake_aux = (
            torch.tensor(0.1),  # kl
            torch.tensor(0.2),  # clipfrac
            torch.tensor(0.3),  # pg_loss
            torch.tensor(0.4),  # entropy
        )

        with (
            patch("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", True),
            patch("agilerl.algorithms.ppo_llm.apply_fused_policy_loss") as mock_fn,
        ):
            mock_fn.return_value = (fake_loss, fake_aux)
            total_loss, metrics = ppo._ppo_loss_liger(
                ids,
                mask,
                old_lp,
                ref_lp,
                returns,
                adv,
                old_values,
                turn_ids,
                "token",
            )

        # Fused-loss entry point called exactly once for the actor pass.
        mock_fn.assert_called_once()
        # Metric keys/values come from the (mocked) auxiliary tuple +
        # the (real) value-head loss computed outside the fusion.
        assert metrics["kl"] == pytest.approx(0.1)
        assert metrics["clipfrac"] == pytest.approx(0.2)
        assert metrics["pg_loss"] == pytest.approx(0.3)
        assert metrics["entropy"] == pytest.approx(0.4)
        assert "vf_loss" in metrics
        # total_loss = fake_loss (0.5) + vf_loss (real, computed from values)
        assert isinstance(total_loss, torch.Tensor)

    def test_token_mode_forwards_configured_chunk_rows(self) -> None:
        ppo = _cpu_llmppo(chunk_rows=123)
        B, T = 2, 5
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.ones(B, T - 1, dtype=torch.float32)
        old_lp = torch.zeros(B, T - 1)
        ref_lp = torch.zeros(B, T - 1)
        returns = torch.zeros(B, T - 1)
        adv = torch.randn(B, T - 1) * 0.1
        old_values = torch.zeros(B, T - 1)
        turn_ids = torch.zeros(B, T - 1, dtype=torch.long)
        fake_aux = tuple(torch.tensor(0.0) for _ in range(4))

        with (
            patch("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", True),
            patch("agilerl.algorithms.ppo_llm.apply_fused_policy_loss") as mock_apply,
        ):
            mock_apply.return_value = (torch.tensor(0.5, requires_grad=True), fake_aux)
            ppo._ppo_loss_liger(
                ids,
                mask,
                old_lp,
                ref_lp,
                returns,
                adv,
                old_values,
                turn_ids,
                "token",
            )

        assert mock_apply.call_args.kwargs["token_chunk_size"] == 123

    def test_turn_mode_passes_turn_args_to_liger(self) -> None:
        """Turn-granularity passes ``turn_ids`` and ``max_turns`` into the
        Liger Function and uses pooled per-turn advantages.
        """
        ppo = _cpu_llmppo(beta=0.0, advantage_granularity="turn")
        B, T = 2, 6
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.ones(B, T - 1, dtype=torch.float32)
        old_lp = torch.zeros(B, T - 1)
        ref_lp = torch.zeros(B, T - 1)
        returns = torch.zeros(B, T - 1)
        adv = torch.randn(B, T - 1) * 0.1
        old_values = torch.zeros(B, T - 1)
        # Two turns per sample: first half = turn 0, second half = turn 1.
        turn_ids = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.long)

        fake_loss = torch.tensor(0.5, requires_grad=True)
        fake_aux = tuple(torch.tensor(0.0) for _ in range(4))

        with (
            patch("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", True),
            patch("agilerl.algorithms.ppo_llm.apply_fused_policy_loss") as mock_fn,
        ):
            mock_fn.return_value = (fake_loss, fake_aux)
            ppo._ppo_loss_liger(
                ids,
                mask,
                old_lp,
                ref_lp,
                returns,
                adv,
                old_values,
                turn_ids,
                "turn",
            )

        # ``turn_ids``, ``full_turn_mask`` and ``max_turns`` are now passed as
        # keyword args to ``apply_fused_policy_loss`` and must be non-None /
        # correct in turn mode.
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["turn_ids"] is not None
        assert call_kwargs["full_turn_mask"] is not None
        assert call_kwargs["max_turns"] == 2
        assert call_kwargs["turn_log_ratio_reduction"] == "sum"

    def test_token_mode_fuses_vllm_is_ratio(self) -> None:
        """token-level IS with captured vLLM logprobs fuses the clamped
        trainer/vLLM ratio into the kernel via the ``vllm_is_ratio`` kwarg.
        """
        ppo = _cpu_llmppo(beta=0.0)
        B, T = 2, 6
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.ones(B, T - 1, dtype=torch.float32)
        old_lp = torch.zeros(B, T - 1)
        ref_lp = torch.zeros(B, T - 1)
        returns = torch.zeros(B, T - 1)
        adv = torch.randn(B, T - 1) * 0.1
        old_values = torch.zeros(B, T - 1)
        turn_ids = torch.zeros(B, T - 1, dtype=torch.long)
        sampling = old_lp - 0.5  # non-trivial trainer/vLLM mismatch
        fake_aux = tuple(torch.tensor(0.0) for _ in range(4))
        with (
            patch("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", True),
            patch("agilerl.algorithms.ppo_llm.apply_fused_policy_loss") as mock_fn,
        ):
            mock_fn.return_value = (torch.tensor(0.5, requires_grad=True), fake_aux)
            ppo._ppo_loss_liger(
                ids,
                mask,
                old_lp,
                ref_lp,
                returns,
                adv,
                old_values,
                turn_ids,
                "token",
                sampling_log_probs=sampling,
            )
        ratio = mock_fn.call_args.kwargs["vllm_is_ratio"]
        assert ratio is not None
        assert torch.all(ratio <= ppo.vllm_importance_sampling_cap)

    def test_trajectory_is_level_pools_advantages_to_per_sample_scalar(self) -> None:
        """An explicit trajectory IS level pools the per-token advantages to a
        masked per-completion mean ``(B, 1)`` for the Liger Function (and emits
        the canonical not-memory-bounded warning at loss time).
        """
        ppo = _cpu_llmppo(beta=0.0, importance_sampling_level="trajectory")
        B, T = 2, 5
        ids = torch.randint(1, 50, (B, T), dtype=torch.long)
        mask = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32
        )
        zeros = torch.zeros(B, T - 1)
        adv = torch.tensor([[1.0, 3.0, 5.0, 100.0], [2.0, 4.0, 6.0, 8.0]])
        turn_ids = torch.zeros(B, T - 1, dtype=torch.long)
        fake_aux = tuple(torch.tensor(0.0) for _ in range(4))

        with (
            patch("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", True),
            patch("agilerl.algorithms.ppo_llm.apply_fused_policy_loss") as mock_fn,
        ):
            mock_fn.return_value = (torch.tensor(0.5, requires_grad=True), fake_aux)
            with pytest.warns(UserWarning, match="NOT memory-bounded"):
                ppo._ppo_loss_liger(
                    ids,
                    mask,
                    zeros,
                    zeros,
                    zeros,
                    adv,
                    zeros,
                    turn_ids,
                    "token",
                )

        call = mock_fn.call_args
        # Masked means: row 0 -> (1 + 3 + 5) / 3 = 3; row 1 -> 20 / 4 = 5.
        assert torch.allclose(call.args[5], torch.tensor([[3.0], [5.0]]), atol=1e-6)
        assert call.args[12] == "trajectory"
        # Trajectory pooling needs no per-turn scatter.
        assert call.kwargs["turn_ids"] is None


class TestPPOLearnWithLiger:
    """Cover the ``if self.use_liger_loss:`` branch inside ``learn()``.
    The branch calls ``_ppo_loss_liger`` once per minibatch, runs
    backward, and accumulates the four ``aux`` metrics + ``vf_loss``.
    We stub ``_ppo_loss_liger`` to a fake (loss, metrics) tuple so the
    test stays CPU-only and doesn't require ``liger-kernel``.
    """

    def test_learn_use_liger_loss_drives_ppo_loss_liger(self, monkeypatch):
        # ``use_liger_loss=True`` would normally trip the construct-time
        # ``HAS_LIGER_KERNEL`` guard and fall back to ``False``. Patch the
        # flag in both modules so PPO accepts the kwarg as-is.
        monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", True)
        ppo = _cpu_llmppo(lr_actor=0.05, update_epochs=1, use_liger_loss=True)
        assert ppo.use_liger_loss is True

        fake_loss = torch.tensor(0.42, requires_grad=True)
        fake_metrics = {
            "kl": 0.1,
            "entropy": 0.2,
            "clipfrac": 0.3,
            "pg_loss": 0.4,
            "vf_loss": 0.5,
        }
        # Stub the inner loss fn — keeps the test CPU-only and isolates
        # the use_liger_loss=True branch in learn() from the
        # actor/critic Liger forwards.
        ppo._ppo_loss_liger = MagicMock(return_value=(fake_loss, fake_metrics))
        # ``unset_fused_adapter_routing`` walks the actor — stub it.
        monkeypatch.setattr(
            "agilerl.algorithms.ppo_llm.unset_fused_adapter_routing",
            lambda actor: None,
        )

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

        learn_out = ppo.learn((completions, action_masks, rewards), turn_ids=turn_ids)

        # The Liger branch was actually exercised (not the fallback path).
        assert ppo._ppo_loss_liger.call_count >= 1
        # And its returned scalars made it into the aggregated metrics.
        assert learn_out["loss"] == pytest.approx(0.42, rel=1e-6)
        assert learn_out["kl"] == pytest.approx(0.1, rel=1e-6)
        assert learn_out["vf_loss"] == pytest.approx(0.5, rel=1e-6)

    def test_learn_liger_token_with_sampling_logps_uses_fused_kernel(self, monkeypatch):
        """token-level use_liger_loss=True + captured vLLM logprobs: the
        correction is fused into the kernel (``vllm_is_ratio``), so learn()
        keeps the fused path and threads ``sampling_log_probs`` through.
        """
        monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", True)
        # Force token-level IS so the fused correction path is exercised
        # (default turn_level_clip=True resolves multi-turn batches to "turn").
        ppo = _cpu_llmppo(
            lr_actor=0.05,
            update_epochs=1,
            use_liger_loss=True,
            importance_sampling_level="token",
        )
        ppo._ppo_loss_liger = MagicMock(
            return_value=(
                torch.tensor(0.5, requires_grad=True),
                {
                    "kl": 0.1,
                    "entropy": 0.2,
                    "clipfrac": 0.0,
                    "pg_loss": 0.3,
                    "vf_loss": 0.4,
                },
            )
        )
        ppo._backward_pass = MagicMock(return_value=None)

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
            ppo.learn(
                (completions, action_masks, rewards),
                turn_ids=turn_ids,
                sampling_logps=sampling_logps,
            )
        ppo._ppo_loss_liger.assert_called()
        # sampling_log_probs threaded in as the final positional arg.
        assert ppo._ppo_loss_liger.call_args.args[9] is not None
        assert not any(
            "token-level importance sampling" in str(w.message) for w in caught
        )
        assert ppo._is_correction_liger_warned is False

    def test_learn_liger_nontoken_with_sampling_logps_warns_and_uses_standard_path(
        self, monkeypatch
    ):
        """trajectory-level use_liger_loss=True + captured vLLM logprobs: the
        per-token reweight can't be pooled into the sequence ratio, so learn()
        warns once and routes the minibatch through the standard PyTorch path.
        """
        monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.ppo_llm.HAS_LIGER_KERNEL", True)
        ppo = _cpu_llmppo(
            lr_actor=0.05,
            update_epochs=1,
            use_liger_loss=True,
            importance_sampling_level="trajectory",
        )
        ppo._ppo_loss_liger = MagicMock(
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
            metrics = ppo.learn(
                (completions, action_masks, rewards),
                turn_ids=turn_ids,
                sampling_logps=sampling_logps,
            )
        ppo._ppo_loss_liger.assert_not_called()
        assert ppo._is_correction_liger_warned is True
        assert "vllm_is_delta_mean" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))


class TestPPOVllmISCorrection:
    """vLLM sampling-mismatch (truncated-IS) correction wiring, token level."""

    @pytest.mark.parametrize("is_level", ["token", "turn", "trajectory"])
    def test_learn_emits_vllm_is_metrics_and_reweights(self, is_level):
        ppo = _cpu_llmppo(
            importance_sampling_level=is_level, lr_actor=0.05, update_epochs=1
        )
        # use_vllm=False auto-disables the correction in __init__; force it on to
        # exercise the capture/align/metrics/reweight path (applied to the policy
        # surrogate via clipped_is_surrogate's loss_weight hook).
        ppo.vllm_importance_sampling_correction = True
        ppo.vllm_importance_sampling_cap = 2.0
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
        metrics = ppo.learn(
            (completions, action_masks, rewards),
            turn_ids=turn_ids,
            sampling_logps=sampling_logps,
        )
        for key in ("vllm_is_delta_mean", "vllm_is_ratio_mean"):
            assert key in metrics
            assert isinstance(metrics[key], float)
        assert metrics["vllm_is_ratio_mean"] > 0
        assert torch.isfinite(torch.tensor(metrics["loss"]))


class _CtxFreeValueActor(nn.Module):
    """Context-free actor + value head for packing-equivalence tests.

    ``hidden = embed(input_ids)`` and ``value = value_head(hidden)`` — with no
    attention there is no cross-sequence contamination, so a packed forward
    reproduces the padded forward's per-token hidden states and values exactly
    at every real token. Returns the ``(hidden, _, value)`` tuple the fused
    path expects (``output[0]`` is the hidden state once the lm_head is
    identity-patched; ``output[2]`` is the value). Records the last
    ``input_ids`` shape so the test can confirm packing engaged.
    """

    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.value_head = nn.Linear(hidden, 1)
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(self, input_ids=None, **kwargs):
        self.last_input_shape = tuple(input_ids.shape)
        h = self.embed(input_ids)  # (rows, S, H)
        value = self.value_head(h).squeeze(-1)  # (rows, S)
        return (h, None, value)


class TestPPOSequencePacking:
    """Sequence packing for the PPO actor-critic forward.

    The flag is plumbed through the base class and inert on a dense backend
    (CPU eager) where it falls back to the padded doubled forward. On a
    varlen/blockmask backend the packed ``_fused_forward`` must reproduce the
    padded actor log-probs *and* critic values at every action position.
    """

    def test_flag_stored_and_learn_runs_with_padded_fallback(self):
        ppo = _cpu_llmppo(use_sequence_packing=True, lr_actor=0.05, update_epochs=1)
        assert ppo.use_sequence_packing is True
        vocab, inp, mtok = 100, 10, 8
        seq_len = inp + mtok
        completions = [torch.randint(0, vocab, (1, seq_len))]
        action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool)]
        turn_ids = torch.tensor(
            [[-1] * (inp - 1) + [0] * (mtok // 2) + [1] * (mtok - mtok // 2)],
            dtype=torch.long,
        )[:, : seq_len - 1]
        rewards = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        metrics = ppo.learn((completions, action_masks, rewards), turn_ids=turn_ids)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_packed_fused_forward_matches_padded(self):
        ppo = _cpu_llmppo(use_vllm=False)
        ppo.pad_token_id = 0
        assert ppo.use_value_head is True
        vocab, hidden = 16, 8
        actor = _CtxFreeValueActor(vocab, hidden).to(ppo.device)
        ppo.actor = actor

        # Right-padded batch with varied real lengths -> packing has work to do.
        lengths = [5, 3, 4]
        b_size, t = len(lengths), max(lengths)
        torch.manual_seed(0)
        ids = torch.zeros(b_size, t, dtype=torch.long)
        for b, length in enumerate(lengths):
            ids[b, :length] = torch.randint(1, vocab, (length,))
        attention_mask = ids != 0
        action_mask = torch.zeros(b_size, t - 1, dtype=torch.bool)
        for b, length in enumerate(lengths):
            action_mask[b, : length - 1] = True

        def fake_fused_fn(
            h, weight, bias, targets, *, temperature, cast_to_fp32, chunk_rows
        ):
            # Context-free per-token logprob over the (already next-token-
            # shifted) hidden features. Identical closed form padded or packed,
            # so equivalence is entirely down to pack/unpack.
            return h.sum(-1)

        def run():
            with (
                patch.object(ppo, "_patch_lm_head_to_identity", nullcontext),
                patch.object(ppo, "_amp_ctx", nullcontext),
                patch.object(ppo, "_activation_offload_ctx", nullcontext),
                patch.object(ppo, "_get_unwrapped_actor", return_value=actor),
                patch.object(
                    ppo,
                    "_fused_logprob_fn_and_head",
                    return_value=nullcontext((fake_fused_fn, None, None)),
                ),
            ):
                return ppo._fused_forward(
                    ids, batch_size=b_size, attention_mask=attention_mask
                )

        # Padded baseline: actor+critic doubled into one (2B, T) forward.
        ppo.use_sequence_packing = False
        lp_padded, v_padded = run()
        assert actor.last_input_shape == (2 * b_size, t)

        # Packed: actor+critic as two packed rows of length N (one model.forward).
        ppo.use_sequence_packing = True
        ppo.model_config = {"attn_implementation": "flash_attention_2"}
        lp_packed, v_packed = run()
        assert actor.last_input_shape == (2, sum(lengths))

        # Identical at every action position (pad columns differ but are masked).
        am = action_mask.to(lp_padded.dtype)
        assert torch.allclose(lp_packed * am, lp_padded * am, atol=1e-5)
        assert torch.allclose(v_packed * am, v_padded * am, atol=1e-5)
        assert (lp_padded * am).abs().sum() > 0
        assert (v_padded * am).abs().sum() > 0


class TestPPOSaveLoadValueHead:
    """PPO differs from the non-critic LLM algos: it carries a value head
    (``v_head`` Linear) that must survive a checkpoint round-trip.
    """

    @staticmethod
    def _build(model_factory):
        actor = model_factory(TINY_LLM_FIXTURE_PATH, add_value_head=True)
        return LLMPPO(
            actor_network=actor,
            lr_actor=1e-5,
            lr_critic=1e-4,
            pad_token_id=151664,
            pad_token="<pad>",
            device="cpu",
            lora_config=LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
                lora_dropout=0.0,
            ),
            accelerator=None,
            use_vllm=False,
            wrap=False,
            gradient_checkpointing=False,
            max_output_tokens=8,
            max_model_len=64,
        )

    def test_save_load_round_trips_value_head_and_actor_adapter(
        self, tmp_path, model_factory
    ):
        """save_checkpoint -> load_checkpoint must restore the value head and the
        actor LoRA adapter (and not crash on optimizer metadata).
        """
        ppo = self._build(model_factory)
        unwrapped = ppo._get_unwrapped_actor()
        # Make the value head + actor LoRA clearly non-default before saving.
        for p in unwrapped.v_head.parameters():
            p.data.normal_(0.0, 1.0)
        actor_lora = [
            (n, p)
            for n, p in unwrapped.named_parameters()
            if "lora" in n.lower() and "actor" in n.lower()
        ]
        assert actor_lora, "expected actor LoRA params"
        for _, p in actor_lora:
            p.data.normal_(0.0, 0.5)
        saved_vhead = {
            k: v.detach().clone() for k, v in unwrapped.v_head.state_dict().items()
        }
        alora_name = actor_lora[0][0]
        alora_w = actor_lora[0][1].detach().clone()

        ppo.save_checkpoint(str(tmp_path))

        # A fresh agent (freshly-initialised value head) must come back identical
        # after load — exercising the default save_optimizer=True path too.
        new_ppo = self._build(model_factory)
        new_ppo.load_checkpoint(str(tmp_path))
        new_unwrapped = new_ppo._get_unwrapped_actor()

        for k, v in saved_vhead.items():
            assert torch.equal(
                new_unwrapped.v_head.state_dict()[k].float(), v.float()
            ), f"value-head weight {k} not restored"
        assert torch.equal(
            dict(new_unwrapped.named_parameters())[alora_name].detach().float(),
            alora_w.float(),
        ), "actor LoRA adapter not restored"

        ppo.clean_up()
        new_ppo.clean_up()
        AcceleratorState._reset_state(True)


class TestPPOColocatedVllm:
    @spawn_new_process_for_each_test
    @pytest.mark.vllm
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    @pytest.mark.parametrize("pretrained_model_name_or_path", [TINY_LLM_FIXTURE_PATH])
    def test_colocated_learn_with_memory_efficient_params(
        self, pretrained_model_name_or_path
    ):
        """Regression: colocated PPO with framework defaults trains on the GPU.

        Constructed the way the docs and profiling harness do — no ``device``
        argument, ``use_memory_efficient_params`` left at its default — the
        trainer used to stay on CPU for the whole run, so ``learn`` fed CPU
        tensors to the Liger Triton kernels patched into the base model.
        """
        input_size, max_tokens, batch_size = 8, 8, 2
        ppo = LLMPPO(
            model_name=pretrained_model_name_or_path,
            pad_token_id=0,
            pad_token="<pad>",
            lora_config=LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
            use_vllm=True,
            # Both knobs are load-bearing under parallel vLLM testing; see the
            # rationale on ``generate_grpo`` in test_grpo.py.
            vllm_config=VLLMConfig(
                gpu_memory_utilization=0.22,
                kv_cache_memory_bytes=32 * 1024 * 1024,
                max_num_seqs=batch_size,
                sleep_mode=True,
            ),
            max_output_tokens=max_tokens,
            max_model_len=input_size + max_tokens + 4,
            batch_size=batch_size,
            micro_batch_size_per_gpu=1,
            update_epochs=1,
        )
        assert ppo.use_memory_efficient_params
        assert torch.device(ppo.device).type == "cuda"

        vocab_size = ppo._get_unwrapped_actor().config.vocab_size
        prompts = [
            {
                "input_ids": torch.randint(
                    0, vocab_size, (1, input_size), device=ppo.device
                ),
                "attention_mask": torch.ones(1, input_size, device=ppo.device),
                "text": "Write me a short story about a cat.",
            }
            for _ in range(batch_size)
        ]

        completion_ids, action_masks, sampling_logps = ppo.get_action(
            prompts, training=True
        )
        # The rollout parks the trainer on CPU; ``learn`` must bring the whole
        # tree — base weights and value head alike — back onto the GPU.
        unwrapped = ppo._get_unwrapped_actor()
        assert all(p.device.type == "cpu" for p in unwrapped.parameters())

        rewards = torch.randn(len(completion_ids), device=ppo.device)
        metrics = ppo.learn(
            (completion_ids, action_masks, rewards),
            sampling_logps=sampling_logps,
        )
        for key in ("loss", "pg_loss", "vf_loss"):
            assert torch.isfinite(torch.tensor(metrics[key])), f"{key} not finite"

        ppo.clean_up()
