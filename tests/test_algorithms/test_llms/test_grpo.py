import copy
import gc
import inspect
import os
import re
import tempfile
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch

pytest.importorskip("deepspeed", reason="LLM tests require deepspeed.")
pytest.importorskip("vllm", reason="LLM tests require vllm.")
import vllm
from accelerate import Accelerator
from accelerate.scheduler import AcceleratedScheduler
from accelerate.state import AcceleratorState
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from peft import LoraConfig, LoraModel, PeftModel, get_peft_model
from torch import nn
from torch.optim.lr_scheduler import SequentialLR
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from vllm import LLM

from agilerl.algorithms import CISPO, GRPO, GSPO
from agilerl.algorithms.core.base import (
    EvolvableAlgorithm,
    LLMAlgorithm,
    OptimizerWrapper,
)
from agilerl.algorithms.grpo import HAS_LIGER_KERNEL
from agilerl.modules.dummy import DummyEvolvable
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig, clone_llm
from tests import TINY_LLM_FIXTURE_PATH
from tests.utils import (
    assert_vllm_get_action_contract,
    make_mock_vllm_instance,
    spawn_new_process_for_each_test,
)
from agilerl.utils.llm_utils import ReasoningGym

deepspeed_base_config = {
    "bf16": {
        "enabled": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
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
        },
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
            self.input_size + self.max_tokens,
            32,
            device=device,
            dtype=self.datatype,
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
        self,
        input_ids: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> DummyForwardOutput:
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
            msg = "`input_ids` must be provided for generation."
            raise ValueError(msg)
        input_shape = input_ids.shape
        group_size = input_shape[0]
        prompt_size = input_shape[1]
        # Simple generation: just return random tokens based on vocab size and desired length
        return torch.randint(
            0,
            self.vocab_size,
            (group_size, prompt_size + self.config.max_tokens),
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.gradient_checkpointing_enabled = True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return


class DummyReasoningEnv(ReasoningGym):
    def __init__(self, vocab_size, input_size, data_batch_size, device):
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.data_batch_size = data_batch_size
        self.device = device

    def reset(self, reset_dataloaders=False):
        return [
            {
                "input_ids": torch.randint(
                    0,
                    self.vocab_size,
                    (1, self.input_size),
                    device=self.device,
                ),
                "attention_mask": torch.ones(*(1, self.input_size), device=self.device),
                "text": "Write me a short story about a cat.",
            }
            for _ in range(self.data_batch_size)
        ]

    def step(self, completion_ids):
        states = [
            {
                "input_ids": torch.randint(
                    0,
                    self.vocab_size,
                    (1, self.input_size),
                    device=self.device,
                ),
                "attention_mask": torch.ones(*(1, self.input_size), device=self.device),
                "text": "Write me a short story about a cat.",
            }
            for _ in range(self.data_batch_size)
        ]
        return (
            states,
            torch.cat(
                [
                    torch.tensor([1.0], device=self.device)
                    for _ in range(self.data_batch_size)
                ],
            ),
        )

    @contextmanager
    def eval_mode(self):
        try:
            yield
        finally:
            pass

    def close(self):
        pass


class DummyVLLM:
    def __init__(self, *args, **kwargs):
        self.llm_engine = MagicMock()
        self.llm_engine.model_executor = MagicMock()

    def generate(self, prompts, *args, **kwargs):
        """This is the behaviour I need to mock:
        all_outputs = self.llm.generate(
            all_prompts_text,
            sampling_params=sampling_params,
            use_tqdm=True,
        )  # Change this to False

        completion_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        """
        num_prompts = len(prompts)

        # Create dummy outputs that match VLLM's expected format
        all_outputs = []

        for _ in range(num_prompts):
            # Generate random token IDs for testing
            # Using a reasonable range for token IDs (0-1000 for testing)
            import random

            token_length = random.randint(5, 20)  # Random length between 5-20 tokens
            token_ids = [random.randint(0, 1000) for _ in range(token_length)]

            # Create the output structure that matches VLLM's format
            dummy_output = SimpleNamespace(token_ids=token_ids)
            request_output = SimpleNamespace(outputs=[dummy_output])
            all_outputs.append(request_output)

        return all_outputs

    def reset_prefix_cache(self):
        """Reset the prefix cache - dummy implementation"""

    def sleep(self, *args, **kwargs):
        pass

    def wake_up(self, *args, **kwargs):
        pass


def create_module(input_size, max_tokens, vocab_size, device):
    return DummyMLPPreTrainedModel(
        config=DummyConfig(
            input_size=input_size,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
        ),
        device=device,
    )


def _patch_mps_learn_hooks(monkeypatch: pytest.MonkeyPatch, module: str) -> MagicMock:
    """Make ``learn`` think MPS is available and record ``torch.mps.empty_cache`` calls."""
    mock_empty = MagicMock()
    monkeypatch.setattr(f"{module}.torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr(f"{module}.torch.mps.empty_cache", mock_empty)
    return mock_empty


def generate_grpo(
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    vocab_size,
    input_size,
    max_tokens,
    group_size,
    use_separate_reference_adapter,
    use_vllm,
    pretrained_model_name_or_path,
    micro_batch_size_per_gpu,
    sleep_mode=False,
    from_name=False,
    use_liger_loss=False,
):
    if config is not None and not torch.cuda.is_available():
        pytest.skip("DeepSpeed-configured LLM tests require CUDA support.")

    gc.collect()
    torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)

    accelerator = accelerator_factory(use_deepspeed_optimizer, config)
    if not use_deepspeed_optimizer and accelerator is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config.pop("optimizer", None)
    if use_vllm:
        lora_config = None
        # ``kv_cache_memory_bytes`` is **required** for parallel vLLM testing:
        # it bypasses vLLM's startup memory-profiling assertion that fires
        # when peer xdist workers (or sibling CI containers) free GPU memory
        # mid-init. See ``VLLMConfig.kv_cache_memory_bytes`` docstring and
        # ``tests/conftest.py:pytest_collection_modifyitems`` for the full
        # rationale.
        #
        # ``gpu_memory_utilization`` below is **dead config** while
        # ``kv_cache_memory_bytes`` is set — vLLM ignores it
        # (vllm/config/cache.py: "kv_cache_memory_bytes (when not-None)
        # ignores gpu_memory_utilization"). Pinned to vLLM's documented
        # default of 0.9 so future readers don't try to read meaning into a
        # specific number. **Footgun**: if you ever remove
        # ``kv_cache_memory_bytes`` from this config, you MUST also drop
        # this back to a small fraction (~0.05–0.2) to leave room for peer
        # xdist workers on the same GPU, otherwise this single instance will
        # try to grab 90% of GPU memory and OOM the other workers.
        vllm_config = VLLMConfig(
            gpu_memory_utilization=0.9,
            kv_cache_memory_bytes=32 * 1024 * 1024,
            max_num_seqs=1,
            sleep_mode=sleep_mode,
            swap_space=0,
            enforce_eager=False,
        )

        actor = model_factory(pretrained_model_name_or_path)
    else:
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
        vllm_config = None
    grpo = GRPO(
        actor_network=actor if not from_name else None,
        model_name=pretrained_model_name_or_path if from_name else None,
        lr=1e-5,
        pad_token_id=vocab_size - 1,
        pad_token="<pad>",
        device="cuda" if torch.cuda.is_available() else "cpu",
        group_size=group_size,
        lora_config=lora_config,
        cosine_lr_schedule_config=(
            None
            if accelerator is not None
            else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
        ),
        accelerator=accelerator,
        use_separate_reference_adapter=use_separate_reference_adapter,
        use_vllm=use_vllm,
        vllm_config=vllm_config,
        max_output_tokens=max_tokens,
        max_model_len=max_tokens + 5,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        use_liger_loss=use_liger_loss,
    )
    return grpo


@pytest.fixture(scope="function")
def grpo_factory():
    return generate_grpo


def _make_cpu_grpo_for_branch_tests(**kwargs):
    defaults = {
        "actor_network": create_module(
            input_size=6,
            max_tokens=4,
            vocab_size=64,
            device="cpu",
        ),
        "pad_token_id": 63,
        "pad_token": "<pad>",
        "batch_size": 4,
        "group_size": 2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "wrap": False,
        "gradient_checkpointing": False,
        "accelerator": None,
        "device": "cpu",
    }
    defaults.update(kwargs)
    return GRPO(**defaults)


def _build_grpo_for_colocate_tests(
    grpo_factory,
    accelerator_factory,
    model_factory,
    tensor_parallel_size: int = 1,
):
    grpo = grpo_factory(
        accelerator_factory,
        model_factory,
        None,
        False,
        100,
        10,
        8,
        2,
        False,
        False,
        None,
        None,
    )
    grpo.vllm_config = VLLMConfig(
        gpu_memory_utilization=0.2,
        max_num_seqs=1,
        tensor_parallel_size=tensor_parallel_size,
    )
    grpo.llm = MagicMock()
    grpo.tp_group = "tp-group"
    grpo.device = "cpu"
    return grpo


class _GrpoMathStub:
    """Minimal stub exposing GRPO math helpers without model initialization."""

    def __init__(self, group_size: int, adv_norm: str = "mean_std") -> None:
        self.group_size = group_size
        self.adv_norm = adv_norm

    _calculate_advantage = GRPO._calculate_advantage
    _calculate_kl_divergence = GRPO._calculate_kl_divergence


class _GrpoLossStub:
    def __init__(
        self,
        clip_coef_min: float,
        clip_coef_max: float,
        beta: float,
        use_kl_advantage_shaping: bool,
    ) -> None:
        self.clip_coef_min = clip_coef_min
        self.clip_coef_max = clip_coef_max
        self.beta = beta
        self.use_kl_advantage_shaping = use_kl_advantage_shaping

    _calculate_kl_divergence = GRPO._calculate_kl_divergence
    _apply_kl_advantage_shaping = GRPO._apply_kl_advantage_shaping
    _reduce_masked_loss = GRPO._reduce_masked_loss
    _grpo_loss_standard = GRPO._grpo_loss_standard
    _gspo_loss = GRPO._gspo_loss
    _cispo_loss = GRPO._cispo_loss


class _GrpoLossStub:
    def __init__(
        self,
        clip_coef_min: float,
        clip_coef_max: float,
        beta: float,
        use_kl_advantage_shaping: bool,
    ) -> None:
        self.clip_coef_min = clip_coef_min
        self.clip_coef_max = clip_coef_max
        self.beta = beta
        self.use_kl_advantage_shaping = use_kl_advantage_shaping

    _calculate_kl_divergence = GRPO._calculate_kl_divergence
    _apply_kl_advantage_shaping = GRPO._apply_kl_advantage_shaping
    _reduce_masked_loss = GRPO._reduce_masked_loss
    _grpo_loss_standard = GRPO._grpo_loss_standard
    _gspo_loss = GRPO._gspo_loss
    _cispo_loss = GRPO._cispo_loss


def _build_branch_experiences(
    batch_size: int,
    seq_len: int = 10,
    vocab_size: int = 64,
):
    completion_ids = [
        torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
        for _ in range(batch_size)
    ]
    action_masks = [
        torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(batch_size)
    ]
    return completion_ids, action_masks


def _build_branch_experiences(
    batch_size: int,
    seq_len: int = 10,
    vocab_size: int = 64,
):
    completion_ids = [
        torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
        for _ in range(batch_size)
    ]
    action_masks = [
        torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(batch_size)
    ]
    return completion_ids, action_masks


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


class TestGRPOInit:
    def test_init_cispo_sets_fixed_loss_type_and_hides_loss_type_arg(self):
        actor = create_module(input_size=6, max_tokens=4, vocab_size=64, device="cpu")
        cispo = CISPO(
            actor_network=actor,
            pad_token_id=63,
            pad_token="<pad>",
            batch_size=4,
            group_size=2,
            max_output_tokens=4,
            max_model_len=12,
            wrap=False,
            gradient_checkpointing=False,
            accelerator=None,
            device="cpu",
        )
        assert cispo.loss_type == "cispo"
        class_sig = str(inspect.signature(CISPO))
        init_sig = str(inspect.signature(CISPO.__init__))
        assert "loss_type" not in class_sig
        assert "loss_type" not in init_sig
        assert "self" not in class_sig
        assert isinstance(cispo, GRPO)

    def test_init_gspo_sets_fixed_loss_type_and_hides_loss_type_arg(self):
        actor = create_module(input_size=6, max_tokens=4, vocab_size=64, device="cpu")
        gspo = GSPO(
            actor_network=actor,
            pad_token_id=63,
            pad_token="<pad>",
            batch_size=4,
            group_size=2,
            max_output_tokens=4,
            max_model_len=12,
            wrap=False,
            gradient_checkpointing=False,
            accelerator=None,
            device="cpu",
        )
        assert gspo.loss_type == "gspo"
        class_sig = str(inspect.signature(GSPO))
        init_sig = str(inspect.signature(GSPO.__init__))
        assert "loss_type" not in class_sig
        assert "loss_type" not in init_sig
        assert "self" not in class_sig
        assert isinstance(gspo, GRPO)

    @patch("agilerl.algorithms.core.base.LLM")
    def test_init_grpo_warns_when_hf_generate_chunk_size_set_with_vllm(
        self, MockLLM, model_factory
    ):
        mock_instance = make_mock_vllm_instance(vllm.LLM)
        MockLLM.return_value = mock_instance
        with pytest.warns(
            UserWarning, match="hf_generate_chunk_size.*ignored.*use_vllm=True"
        ):
            grpo = GRPO(
                actor_network=model_factory(TINY_LLM_FIXTURE_PATH),
                pad_token_id=999,
                pad_token="<pad>",
                group_size=2,
                use_vllm=True,
                vllm_config=VLLMConfig(
                    gpu_memory_utilization=0.05,
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
        grpo.clean_up()

    @pytest.mark.parametrize(
        "config, use_deepspeed_optimizer",
        [
            (deepspeed_config_stage_1, False),
            (deepspeed_config_stage_1, True),
            (deepspeed_config_stage_1_with_scheduler, False),
            (deepspeed_config_stage_1_with_scheduler, True),
            (deepspeed_config_stage_2, False),
            (deepspeed_config_stage_2, True),
        ],
    )
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_separate_reference_adapter",
        [False, True],
    )
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [
            (False, TINY_LLM_FIXTURE_PATH),
            (True, TINY_LLM_FIXTURE_PATH),
        ],
    )
    @pytest.mark.parametrize(
        "micro_batch_size_per_gpu",
        [None, 2],
    )
    @pytest.mark.parametrize(
        "from_name",
        [True, False],
    )
    @pytest.mark.vllm
    def test_init_grpo_with_accelerator(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_separate_reference_adapter,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
        from_name,
    ):
        mock_llm_instance = make_mock_vllm_instance(vllm.LLM)
        llm_patch_ctx = (
            patch("agilerl.algorithms.core.base.LLM", return_value=mock_llm_instance)
            if use_vllm
            else nullcontext()
        )
        with llm_patch_ctx:
            grpo = grpo_factory(
                accelerator_factory,
                model_factory,
                config,
                use_deepspeed_optimizer,
                vocab_size,
                input_size,
                max_tokens,
                group_size,
                use_separate_reference_adapter,
                use_vllm,
                pretrained_model_name_or_path,
                micro_batch_size_per_gpu,
                from_name=from_name,
            )

        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        assert grpo.batch_size_per_process == 16
        assert grpo.beta == 0.001
        assert grpo.lr == 1e-5
        assert grpo.clip_coef == 0.2
        assert grpo.max_grad_norm == 0.1
        assert grpo.update_epochs == 1
        assert grpo.group_size == group_size
        assert grpo.temperature == 0.9
        assert grpo.calc_position_embeddings
        assert grpo.device == accelerator.device
        assert grpo.index == 0
        assert grpo.scores == []
        assert grpo.fitness == []
        assert grpo.steps == [0]
        assert 3 > grpo.zero_stage >= 1
        assert isinstance(grpo.generation_config, GenerationConfig)
        assert isinstance(grpo.actor, DeepSpeedEngine)
        assert grpo.pad_token_id == 999
        assert grpo.pad_token == "<pad>"
        if not use_deepspeed_optimizer:
            if accelerator is None:
                assert isinstance(
                    grpo.lr_scheduler,
                    AcceleratedScheduler,
                ), grpo.lr_scheduler
                assert isinstance(
                    grpo.cosine_lr_schedule_config,
                    CosineLRScheduleConfig,
                ), type(grpo.cosine_lr_schedule_config)
            assert isinstance(grpo.optimizer, OptimizerWrapper)
            assert isinstance(grpo.optimizer.optimizer, DeepSpeedOptimizerWrapper)
        else:
            assert isinstance(grpo.optimizer, OptimizerWrapper)
            assert isinstance(grpo.optimizer.optimizer, DeepSpeedZeroOptimizer)
            assert isinstance(grpo.actor.optimizer, DeepSpeedZeroOptimizer)
            assert grpo.lr_scheduler is None
            assert grpo.cosine_lr_schedule_config is None

        if use_vllm:
            assert grpo.use_vllm
            assert isinstance(grpo.vllm_config, VLLMConfig)
            assert grpo.llm is mock_llm_instance
        grpo.clean_up()

    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_vllm", [True])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [TINY_LLM_FIXTURE_PATH],
    )
    @pytest.mark.gpu
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize("use_separate_reference_adapter", [True])
    def test_init_grpo_vllm_with_tp_gt_one(
        self,
        deepspeed_env,
        accelerator_factory,
        model_factory,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_separate_reference_adapter,
        use_vllm,
        pretrained_model_name_or_path,
        use_deepspeed_optimizer,
        config,
    ):
        mock_instance = make_mock_vllm_instance(vllm.LLM)
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with (
            patch.object(
                torch.distributed,
                "new_subgroups_by_enumeration",
                return_value=("tp_group_calculated", None),
            ),
            patch(
                "accelerate.Accelerator.num_processes",
                new_callable=PropertyMock,
                return_value=2,
            ),
            patch.object(vllm.LLM, "__init__", return_value=None),
            patch.object(vllm.LLM, "__new__", return_value=mock_instance),
        ):
            grpo = GRPO(
                actor_network=model_factory(pretrained_model_name_or_path),
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=(
                    None
                    if accelerator is not None
                    else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
                ),
                use_vllm=use_vllm,
                vllm_config=VLLMConfig(
                    gpu_memory_utilization=0.05,
                    tensor_parallel_size=2,
                    max_num_seqs=1,
                ),
                accelerator=accelerator,
                use_separate_reference_adapter=use_separate_reference_adapter,
                max_output_tokens=max_tokens,
            )
            assert grpo.tp_group == "tp_group_calculated"
        grpo.clean_up()

    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_vllm", [True])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [TINY_LLM_FIXTURE_PATH],
    )
    @pytest.mark.gpu
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize("use_separate_reference_adapter", [True])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_init_grpo_vllm_tp_value_error(
        self,
        deepspeed_env,
        accelerator_factory,
        model_factory,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_separate_reference_adapter,
        use_vllm,
        pretrained_model_name_or_path,
        use_deepspeed_optimizer,
        config,
        micro_batch_size_per_gpu,
    ):
        mock_instance = make_mock_vllm_instance(vllm.LLM)
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with (
            patch.object(
                torch.distributed,
                "new_subgroups_by_enumeration",
                return_value=("tp_group_calculated", None),
            ),
            patch.object(vllm.LLM, "__init__", return_value=None),
            patch.object(vllm.LLM, "__new__", return_value=mock_instance),
            pytest.raises(
                ValueError,
                match="Tensor parallel size 2 must be a multiple of the number of processes 1.",
            ),
        ):
            GRPO(
                actor_network=model_factory(pretrained_model_name_or_path),
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=(
                    None
                    if accelerator is not None
                    else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
                ),
                use_vllm=use_vllm,
                vllm_config=VLLMConfig(
                    gpu_memory_utilization=0.05,
                    tensor_parallel_size=2,
                    max_num_seqs=1,
                ),
                accelerator=accelerator,
                use_separate_reference_adapter=use_separate_reference_adapter,
                max_output_tokens=max_tokens,
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    def test_init_grpo_vllm_invalid_attention_backend_value_error(
        self,
        deepspeed_env,
        accelerator_factory,
        model_factory,
        use_deepspeed_optimizer,
        config,
    ):
        pretrained_model_name_or_path = TINY_LLM_FIXTURE_PATH
        vocab_size = 1000
        max_tokens = 20
        group_size = 5
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with (
            patch.dict(
                os.environ, {"VLLM_ATTENTION_BACKEND": "TORCH_SDPA"}, clear=False
            ),
            patch(
                "agilerl.algorithms.core.base.LLM",
                side_effect=ValueError(
                    "Backend TORCH_SDPA must be registered before use."
                ),
            ),
            pytest.raises(
                ValueError,
                match=r"unsupported VLLM_ATTENTION_BACKEND='TORCH_SDPA'",
            ),
        ):
            GRPO(
                actor_network=model_factory(pretrained_model_name_or_path),
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=(
                    None
                    if accelerator is not None
                    else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
                ),
                use_vllm=True,
                vllm_config=VLLMConfig(
                    gpu_memory_utilization=0.05,
                    max_num_seqs=1,
                ),
                accelerator=accelerator,
                use_separate_reference_adapter=True,
                max_output_tokens=max_tokens,
            )

    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    def test_init_grpo_scheduler_warning_no_accelerator(
        self,
        deepspeed_env,
        model_factory,
        vocab_size,
        group_size,
        use_separate_reference_adapter,
        use_vllm,
        pretrained_model_name_or_path,
    ):
        with pytest.warns(UserWarning):
            GRPO(
                actor_network=model_factory(pretrained_model_name_or_path),
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=CosineLRScheduleConfig(
                    num_epochs=10,
                    warmup_proportion=0.05,
                ),
                use_vllm=use_vllm,
                accelerator=None,
                use_separate_reference_adapter=use_separate_reference_adapter,
                max_output_tokens=20,
            )

    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_init_grpo_batch_size_value_error(
        self,
        deepspeed_env,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_separate_reference_adapter,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with (
            pytest.raises(ValueError),
            patch(
                "accelerate.Accelerator.num_processes",
                new_callable=PropertyMock,
                return_value=2,
            ),
        ):
            GRPO(
                actor_network=model_factory(pretrained_model_name_or_path),
                pad_token_id=vocab_size - 1,
                batch_size=17,
                pad_token="<pad>",
                accelerator=accelerator,
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=CosineLRScheduleConfig(
                    num_epochs=10,
                    warmup_proportion=0.05,
                ),
                use_vllm=use_vllm,
                use_separate_reference_adapter=use_separate_reference_adapter,
            )

    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_init_grpo_max_model_len_and_max_output_tokens_none_error(
        self,
        deepspeed_env,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_separate_reference_adapter,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with pytest.raises(
            ValueError,
            match="Either max_output_tokens or max_model_len must be specified",
        ):
            GRPO(
                actor_network=model_factory(pretrained_model_name_or_path),
                pad_token_id=vocab_size - 1,
                batch_size=17,
                pad_token="<pad>",
                accelerator=accelerator,
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=CosineLRScheduleConfig(
                    num_epochs=10,
                    warmup_proportion=0.05,
                ),
                use_vllm=use_vllm,
                use_separate_reference_adapter=use_separate_reference_adapter,
                max_output_tokens=None,
                max_model_len=None,
            )

    @pytest.mark.parametrize(
        ("extra_kwargs", "expected_msg"),
        [
            (
                {"adv_norm": "bad_norm"},
                "Invalid adv_norm 'bad_norm'. Expected one of ['mean_std', 'mean_only'].",
            ),
            (
                {"loss_type": "bad_level"},
                "Invalid loss_type 'bad_level'. Expected one of ['grpo', 'gspo', 'cispo'].",
            ),
            (
                {"adv_clip_range": 0.0},
                "adv_clip_range must be > 0 when provided.",
            ),
            (
                {"adv_filter_eps": -1e-6},
                "adv_filter_eps must be >= 0.",
            ),
        ],
    )
    def test_init_grpo_new_validation_errors(self, extra_kwargs, expected_msg):
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            _make_cpu_grpo_for_branch_tests(**extra_kwargs)

    @pytest.mark.skipif(
        not HAS_LIGER_KERNEL,
        reason="KL-shaping liger warning path requires liger-kernel availability.",
    )
    def test_init_grpo_liger_warns_and_disables_unsupported_kl_shaping(self):
        with pytest.warns(
            UserWarning,
            match="use_kl_advantage_shaping is not supported with use_liger_loss=True",
        ):
            grpo = _make_cpu_grpo_for_branch_tests(
                use_liger_loss=True,
                use_kl_advantage_shaping=True,
            )
        assert grpo.use_kl_advantage_shaping is False
        grpo.clean_up()

    @pytest.mark.skipif(
        not HAS_LIGER_KERNEL,
        reason="Non-grpo liger warning path requires liger-kernel availability.",
    )
    def test_init_grpo_liger_warns_and_falls_back_to_standard_path_for_non_grpo_loss(
        self,
    ):
        with pytest.warns(
            UserWarning,
            match="use_liger_loss=True is only supported for loss_type='grpo'",
        ):
            grpo = _make_cpu_grpo_for_branch_tests(
                use_liger_loss=True,
                loss_type="gspo",
            )
        assert grpo.use_liger_loss is False
        grpo.clean_up()

    def test_init_grpo_cispo_warns_when_beta_nonzero(self):
        with pytest.warns(UserWarning, match="CISPO is typically used with beta=0"):
            grpo = _make_cpu_grpo_for_branch_tests(loss_type="cispo", beta=0.1)
        grpo.clean_up()

    @pytest.mark.gpu
    @pytest.mark.parametrize("loss_type", ["grpo", "gspo"])
    def test_init_grpo_non_cispo_nonzero_beta_no_warning(self, loss_type):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grpo = _make_cpu_grpo_for_branch_tests(loss_type=loss_type, beta=0.1)
        assert not any(
            "CISPO is typically used with beta=0" in str(w.message) for w in caught
        )
        grpo.clean_up()

    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_init_grpo_batch_size_grad_accum_error(
        self,
        deepspeed_env,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_separate_reference_adapter,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with (
            pytest.raises(ValueError),
            patch(
                "accelerate.Accelerator.num_processes",
                new_callable=PropertyMock,
                return_value=2,
            ),
        ):
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"
            ] = 7
            GRPO(
                actor_network=model_factory(pretrained_model_name_or_path),
                pad_token_id=vocab_size - 1,
                batch_size=16,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=CosineLRScheduleConfig(
                    num_epochs=10,
                    warmup_proportion=0.05,
                ),
                use_vllm=use_vllm,
                accelerator=accelerator,
                use_separate_reference_adapter=use_separate_reference_adapter,
                max_output_tokens=max_tokens,
            )

    @pytest.mark.parametrize(
        "config, use_deepspeed_optimizer",
        [
            (None, False),
        ],
    )
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_init_grpo_with_no_accelerator(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_separate_reference_adapter,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        assert grpo.batch_size_per_process == 16
        assert grpo.beta == 0.001
        assert grpo.lr == 1e-5
        assert grpo.clip_coef == 0.2
        assert grpo.max_grad_norm == 0.1
        assert grpo.update_epochs == 1
        assert grpo.group_size == 5
        assert grpo.temperature == 0.9
        assert grpo.calc_position_embeddings
        assert isinstance(grpo.cosine_lr_schedule_config, CosineLRScheduleConfig), type(
            grpo.cosine_lr_schedule_config,
        )
        assert grpo.device == (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        assert grpo.index == 0
        assert grpo.scores == []
        assert grpo.fitness == []
        assert grpo.steps == [0]
        assert grpo.pad_token_id == 999
        assert grpo.pad_token == "<pad>"
        assert isinstance(grpo.generation_config, GenerationConfig)
        assert isinstance(grpo.actor, DummyEvolvable)
        assert isinstance(grpo.optimizer, OptimizerWrapper)
        assert isinstance(grpo.lr_scheduler, SequentialLR), grpo.lr_scheduler
        grpo.clean_up()

    @pytest.mark.gpu
    @pytest.mark.parametrize("config", [deepspeed_config_stage_3])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    def test_init_grpo_zero3_warning(
        self,
        deepspeed_env,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with pytest.warns(UserWarning):
            gc.collect()
            vocab_size = 1000
            input_size = 10
            max_tokens = 20
            group_size = 5
            grpo = GRPO(
                actor_network=create_module(
                    input_size=input_size,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                lr=0.1,
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
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
                use_separate_reference_adapter=use_separate_reference_adapter,
                max_output_tokens=max_tokens,
            )
        grpo.clean_up()

    @pytest.mark.gpu
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    def test_init_grpo_lr_warning(
        self,
        deepspeed_env,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with pytest.warns(UserWarning):
            gc.collect()
            vocab_size = 1000
            input_size = 10
            max_tokens = 20
            group_size = 5
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            )
            grpo = GRPO(
                actor_network=create_module(
                    input_size=input_size,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                lr=0.1,
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
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
                use_separate_reference_adapter=use_separate_reference_adapter,
                max_output_tokens=max_tokens,
            )
        assert grpo.lr == 1e-4 if use_deepspeed_optimizer else 0.1
        grpo.clean_up()

    @pytest.mark.gpu
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    def test_init_grpo_max_grad_norm_warning(
        self,
        deepspeed_env,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with pytest.warns(UserWarning):
            gc.collect()
            vocab_size = 1000
            input_size = 10
            max_tokens = 20
            group_size = 5
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            )
            GRPO(
                actor_network=create_module(
                    input_size=input_size,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                lr=0.1,
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
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
                use_separate_reference_adapter=use_separate_reference_adapter,
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("config", [deepspeed_config_stage_1_with_scheduler])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    def test_init_grpo_scheduler_warning(
        self,
        deepspeed_env,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with pytest.warns(UserWarning):
            gc.collect()
            vocab_size = 1000
            input_size = 10
            max_tokens = 20
            group_size = 5
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            )
            GRPO(
                actor_network=create_module(
                    input_size=input_size,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                lr=0.1,
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                lora_config=lora_config,
                cosine_lr_schedule_config=CosineLRScheduleConfig(
                    num_epochs=10,
                    warmup_proportion=0.05,
                ),
                max_grad_norm=0.1,
                accelerator=accelerator,
                use_separate_reference_adapter=use_separate_reference_adapter,
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [7])
    @pytest.mark.parametrize("batch_size", [16])
    def test_init_grpo_micro_batch_size_per_gpu_division_error(
        self,
        deepspeed_env,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        micro_batch_size_per_gpu,
        batch_size,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with pytest.raises(ValueError) as e:
            gc.collect()
            vocab_size = 1000
            input_size = 10
            max_tokens = 20
            group_size = 5
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            )
            GRPO(
                actor_network=create_module(
                    input_size=input_size,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                lr=0.1,
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                lora_config=lora_config,
                cosine_lr_schedule_config=CosineLRScheduleConfig(
                    num_epochs=10,
                    warmup_proportion=0.05,
                ),
                batch_size=batch_size,
                max_grad_norm=0.1,
                accelerator=accelerator,
                use_separate_reference_adapter=use_separate_reference_adapter,
                micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            )
        assert (
            f"When specifying micro_batch_size_per_gpu, batch_size ({batch_size}) must be divisible by the product of the number of processes ({accelerator.num_processes}) and micro_batch_size_per_gpu ({micro_batch_size_per_gpu})."
            in str(e.value)
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("config", [None])
    def test_init_grpo_lora_config_warning(
        self,
        deepspeed_env,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with pytest.warns(
            UserWarning,
            match=r"No LoRA config provided\.\s+AgileRL can only be used to finetune adapters at present\.\s+Using default LoRA configuration for RL finetuning:",
        ):
            gc.collect()
            vocab_size = 1000
            input_size = 10
            max_tokens = 20
            group_size = 5
            GRPO(
                actor_network=create_module(
                    input_size=input_size,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                lr=0.1,
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                cosine_lr_schedule_config=(
                    None
                    if accelerator is not None
                    else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
                ),
                accelerator=accelerator,
            )

    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("config", [None])
    def test_init_grpo_separate_reference_adapter_deprecation_warning(
        self,
        deepspeed_env,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
    ):
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        LLMAlgorithm._separate_reference_adapter_deprecation_emitted = False
        with pytest.warns(
            DeprecationWarning,
            match=r"use_separate_reference_adapter=True.*deprecated",
        ):
            vocab_size = 1000
            input_size = 10
            max_tokens = 20
            group_size = 5
            GRPO(
                actor_network=create_module(
                    input_size=input_size,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                ),
                lr=0.1,
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                use_separate_reference_adapter=True,
                cosine_lr_schedule_config=(
                    None
                    if accelerator is not None
                    else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
                ),
                accelerator=accelerator,
            )
        LLMAlgorithm._separate_reference_adapter_deprecation_emitted = False

    def test_grpo_no_llm_dependencies(
        self, grpo_factory, model_factory, accelerator_factory
    ):
        with (
            mock.patch("agilerl.algorithms.core.base.HAS_LLM_DEPENDENCIES", False),
            pytest.raises(
                ImportError,
                match=r"LLM dependencies are not installed. Please install them using \`pip install agilerl\[llm\]\`.",
            ),
        ):
            grpo_factory(
                accelerator_factory=accelerator_factory,
                model_factory=model_factory,
                config=None,
                use_deepspeed_optimizer=False,
                vocab_size=30,
                input_size=5,
                max_tokens=10,
                use_separate_reference_adapter=False,
                pretrained_model_name_or_path=None,
                micro_batch_size_per_gpu=None,
                from_name=False,
                group_size=2,
                use_vllm=False,
            ).clean_up()
        AcceleratorState._reset_state(True)

    @pytest.mark.parametrize("assertion_mode", ["warns_and_fallback", "private_guard"])
    def test_grpo_liger_unavailable_behaviour(
        self,
        monkeypatch,
        grpo_factory,
        model_factory,
        accelerator_factory,
        assertion_mode,
    ):
        monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False)
        monkeypatch.setattr("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", False)
        if assertion_mode == "warns_and_fallback":
            with pytest.warns(
                UserWarning,
                match=r"use_liger_loss=True requested.*Falling back to standard loss\.",
            ):
                grpo = grpo_factory(
                    accelerator_factory=accelerator_factory,
                    model_factory=model_factory,
                    config=None,
                    use_deepspeed_optimizer=False,
                    vocab_size=30,
                    input_size=5,
                    max_tokens=10,
                    use_separate_reference_adapter=False,
                    pretrained_model_name_or_path=None,
                    micro_batch_size_per_gpu=None,
                    from_name=False,
                    group_size=2,
                    use_vllm=False,
                    use_liger_loss=True,
                )
            assert grpo.use_liger_loss is False
        else:
            grpo = grpo_factory(
                accelerator_factory=accelerator_factory,
                model_factory=model_factory,
                config=None,
                use_deepspeed_optimizer=False,
                vocab_size=30,
                input_size=5,
                max_tokens=10,
                use_separate_reference_adapter=False,
                pretrained_model_name_or_path=None,
                micro_batch_size_per_gpu=None,
                from_name=False,
                group_size=2,
                use_vllm=False,
                use_liger_loss=False,
            )
            with pytest.raises(
                ImportError,
                match=r"Liger GRPO loss was requested but `liger-kernel` is not available\. Set use_liger_loss=False\.",
            ):
                grpo._grpo_loss_liger(
                    batch_ids=torch.ones((1, 2), dtype=torch.long),
                    action_mask=torch.ones((1, 1), dtype=torch.bool),
                    advantages=torch.ones((1,), dtype=torch.float32),
                    old_log_probs=torch.zeros((1, 1), dtype=torch.float32),
                    reference_log_probs=torch.zeros((1, 1), dtype=torch.float32),
                )

        grpo.clean_up()
        AcceleratorState._reset_state(True)


class TestGRPOGetAction:
    def test_get_action_grpo_hf_path_contract(
        self,
        grpo_factory,
        accelerator_factory,
        model_factory,
    ):
        input_size = 10
        max_tokens = 8
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            None,
            False,
            100,
            input_size,
            max_tokens,
            2,
            False,
            False,
            None,
            None,
        )
        prompts = [
            {
                "input_ids": torch.randint(0, 100, (1, input_size), device=grpo.device),
                "attention_mask": torch.ones(1, input_size, device=grpo.device),
            }
            for _ in range(3)
        ]
        for training in (True, False):
            completion_ids, action_masks = grpo.get_action(prompts, training=training)
            expected_group_size = grpo.group_size if training else 1
            assert all(ids.shape[0] == expected_group_size for ids in completion_ids)
            assert_vllm_get_action_contract(
                completion_ids=completion_ids,
                action_masks=action_masks,
                batch_size=len(prompts),
                prompt_len=input_size,
                pad_token_id=grpo.pad_token_id,
            )

        grpo.clean_up()

    def test_get_action_grpo_hf_stop_iteration_device_fallback(self):
        grpo = _make_cpu_grpo_for_branch_tests()
        prompts = [
            {
                "input_ids": torch.randint(0, 64, (1, 6), device=grpo.device),
                "attention_mask": torch.ones(1, 6, device=grpo.device),
            },
        ]
        no_param_actor = SimpleNamespace(parameters=lambda: iter(()))
        with patch.object(grpo, "_get_unwrapped_actor", return_value=no_param_actor):
            completion_ids, action_masks = grpo.get_action(prompts, training=False)
        assert len(completion_ids) == 1
        assert len(action_masks) == 1
        grpo.clean_up()

    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [2])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [
            (True, TINY_LLM_FIXTURE_PATH),
        ],
    )
    @pytest.mark.gpu
    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("data_batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    @pytest.mark.parametrize("sleep_mode", [True])
    @patch("agilerl.algorithms.core.base.LLM")
    def test_get_action_grpo_vllm_sleep_mode(
        self,
        MockLLM,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        training,
        data_batch_size,
        micro_batch_size_per_gpu,
        sleep_mode,
    ):
        mock_instance = make_mock_vllm_instance(vllm.LLM)

        # Make LLM() constructor return mock instance
        MockLLM.return_value = mock_instance

        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
            sleep_mode,
        )
        assert grpo.use_vllm is True
        with (
            patch.object(
                grpo,
                "_prepare_vllm_for_generation",
                wraps=grpo._prepare_vllm_for_generation,
            ) as mock_prepare_vllm_for_generation,
            patch.object(
                grpo,
                "_generate_with_vllm_colocate",
                return_value=(
                    [torch.ones(1, 10, dtype=torch.long)],
                    [torch.ones(1, 9, dtype=torch.bool)],
                ),
            ) as mock_generate_with_vllm_colocate,
        ):
            prompt_dict = {
                "input_ids": torch.randint(0, vocab_size, (1, 10)),
                "attention_mask": torch.randint(0, 2, (1, 10)),
                "text": "Write me a short story about a cat.",
            }
            grpo.get_action([prompt_dict] * data_batch_size, training)
            mock_prepare_vllm_for_generation.assert_called()
            mock_generate_with_vllm_colocate.assert_called()
        mock_instance.sleep.assert_called()
        mock_instance.wake_up.assert_called()
        grpo.clean_up()

    @spawn_new_process_for_each_test
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(True, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.gpu
    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("data_batch_size", [8])
    @pytest.mark.parametrize("tensor_parallel_size", [1, 2])
    def test_get_action_grpo_vllm_multiple_gpus(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        training,
        data_batch_size,
        tensor_parallel_size: int,
    ):
        mock_instance = make_mock_vllm_instance(vllm.LLM)
        with (
            patch.object(vllm.LLM, "__init__", return_value=None),
            patch.object(vllm.LLM, "__new__", return_value=mock_instance),
        ):
            grpo = grpo_factory(
                accelerator_factory,
                model_factory,
                config,
                use_deepspeed_optimizer,
                vocab_size,
                input_size,
                max_tokens,
                group_size,
                use_separate_reference_adapter,
                use_vllm,
                pretrained_model_name_or_path,
                None,
            )
        grpo.vllm_config = VLLMConfig(
            gpu_memory_utilization=0.2,
            max_num_seqs=1,
            tensor_parallel_size=tensor_parallel_size,
        )
        grpo.llm = MagicMock()
        grpo.tp_group = "tp-group"
        grpo.device = "cpu"
        assert grpo.vllm_config.tensor_parallel_size == tensor_parallel_size
        assert isinstance(training, bool)
        assert data_batch_size > 0
        grpo.clean_up()


class TestGRPOMoveModelToVllm:
    @spawn_new_process_for_each_test
    @pytest.mark.parametrize(
        "config",
        [
            deepspeed_config_stage_2,
        ],
    )
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [True])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(True, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_move_model_to_vllm(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        # Change lora B so that the parameters are different
        for name, param in grpo.actor.named_parameters():
            if "lora_B" in name and param is not None:
                param.data.normal_()
        model_ref = grpo.accelerator.unwrap_model(grpo.actor)
        model_ref.set_adapter("actor")
        model_ref.merge_adapter()
        merged_model_ref = copy.deepcopy(model_ref)
        model_ref.unmerge_adapter()
        grpo._move_model_to_vllm()

        # Tiny LLM fixture is Qwen2, whose params live under "model." (not
        # "model.decoder." like OPT). The PEFT-merged copy adds the standard
        # "base_model.model." LoRA prefix on top of the underlying HF param names.
        llm_prefix = "model."
        merged_prefix = "base_model.model.model."

        for (
            name,
            param,
        ) in grpo.llm.llm_engine.model_executor.driver_worker.model_runner.model.named_parameters():
            name = merged_prefix + name.removeprefix(llm_prefix)
            if name in merged_model_ref.state_dict():
                if param.shape != merged_model_ref.state_dict()[name].shape:
                    continue
                assert torch.allclose(
                    param.to(torch.bfloat16),
                    merged_model_ref.state_dict()[name],
                )

        # Test with original_module — exercises the skip path in
        # _move_model_to_vllm. The shape is irrelevant (entry is filtered out
        # before being loaded into vLLM) but we match the tiny Qwen2 hidden_size
        # for consistency.
        fake_named_params = [
            (
                "base_model.model.model.layers.0.input_layernorm.weight.original_module",
                torch.randn(32),
            ),
        ]
        model_ref = grpo.accelerator.unwrap_model(grpo.actor)
        with patch.object(
            model_ref, "named_parameters", return_value=fake_named_params
        ):
            grpo._move_model_to_vllm()

        grpo.clean_up()


class TestGRPOGenerateWithVllmColocate:
    def test_generate_with_vllm_colocate_basic_contract(
        self,
        grpo_factory,
        accelerator_factory,
        model_factory,
    ):
        grpo = _build_grpo_for_colocate_tests(
            grpo_factory, accelerator_factory, model_factory
        )
        prompts = [
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
        ]
        grpo.llm.generate.return_value = [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7, 8])]),
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[9])]),
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[10])]),
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[11, 12])]),
        ]
        completion_ids, action_masks = grpo._generate_with_vllm_colocate(
            prompts=prompts,
            group_size=2,
            temperature=0.7,
        )
        assert_vllm_get_action_contract(
            completion_ids=completion_ids,
            action_masks=action_masks,
            batch_size=2,
            prompt_len=3,
            pad_token_id=grpo.pad_token_id,
        )
        grpo.clean_up()

    def test_generate_with_vllm_colocate_respects_training_kwargs(
        self,
        grpo_factory,
        accelerator_factory,
        model_factory,
    ):
        grpo = _build_grpo_for_colocate_tests(
            grpo_factory, accelerator_factory, model_factory, tensor_parallel_size=2
        )
        prompts = [
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }
        ]
        grpo.llm.generate.return_value = [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7])]),
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[8])]),
        ]

        def fake_all_gather(dest, obj, group):
            del group
            for idx in range(len(dest)):
                dest[idx] = obj

        with (
            patch.object(
                torch.distributed, "all_gather_object", side_effect=fake_all_gather
            ),
            patch.object(torch.distributed, "get_rank", return_value=1),
        ):
            completion_ids, action_masks = grpo._generate_with_vllm_colocate(
                prompts=prompts,
                group_size=1,
                temperature=0.7,
            )
        assert completion_ids[0].shape[0] == 1
        assert completion_ids[0][0, -1].item() == 8
        assert action_masks[0].shape[1] == completion_ids[0].shape[1] - 1
        grpo.clean_up()

    def test_generate_with_vllm_colocate_stitch_path(
        self,
        grpo_factory,
        accelerator_factory,
        model_factory,
    ):
        grpo = _build_grpo_for_colocate_tests(
            grpo_factory, accelerator_factory, model_factory
        )
        prompts = [
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "stitch_prefix_ids": torch.tensor([[9]], dtype=torch.long),
                "initial_prompt_len": 2,
            }
        ]
        grpo.llm.generate.return_value = [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[4, 5])]),
        ]
        with patch(
            "agilerl.algorithms.core.base.stitch_completion_after_windowed_vllm_generate",
            side_effect=lambda completion_ids, *_args, **_kwargs: completion_ids,
        ) as mock_stitch:
            grpo._generate_with_vllm_colocate(
                prompts=prompts,
                group_size=1,
                temperature=0.7,
            )
        mock_stitch.assert_called_once()
        args, _kwargs = mock_stitch.call_args
        (
            completion_ids_arg,
            stitch_prefixes_arg,
            group_prompts_arg,
            group_size_arg,
            prompts_arg,
        ) = args
        assert len(completion_ids_arg) == len(prompts)
        assert len(stitch_prefixes_arg) == len(group_prompts_arg)
        assert group_size_arg == 1
        assert len(prompts_arg) == len(prompts)
        grpo.clean_up()


class TestGRPOCalculateAdvantage:
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "rewards",
        [
            torch.tensor([[2, 4, 6, 8, 20], [3, 6, 9, 12, 15]], dtype=torch.float32),
        ],
    )
    def test_calculate_advantage(
        self,
        group_size,
        rewards,
    ):
        stub = _GrpoMathStub(group_size=group_size)
        calculated_advantage = stub._calculate_advantage(rewards)
        mean_rewards = torch.mean(rewards, dim=1).unsqueeze(1)
        std_rewards = torch.std(rewards, dim=1).unsqueeze(1)
        advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
        advantages = advantages.flatten().unsqueeze(1)
        assert torch.equal(advantages, calculated_advantage)

    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "rewards",
        [
            torch.tensor([[2, 4, 6]], dtype=torch.float32),
        ],
    )
    def test_calculate_advantage_raises_when_rewards_not_divisible_by_group_size(
        self,
        group_size,
        rewards,
    ):
        stub = _GrpoMathStub(group_size=group_size)
        with pytest.raises(ValueError) as e:
            stub._calculate_advantage(rewards)
        assert (
            f"Rewards must have a total element count divisible by group_size ({group_size}); got {rewards.numel()} elements."
            in str(e.value)
        )

    def test_calculate_advantage_mean_only_branch(self):
        stub = _GrpoMathStub(group_size=2, adv_norm="mean_only")
        rewards = torch.tensor([[1.0, 3.0], [4.0, 10.0]], dtype=torch.float32)
        calculated_advantage = stub._calculate_advantage(rewards)
        expected = (rewards - rewards.mean(dim=1, keepdim=True)).flatten().unsqueeze(1)
        assert torch.equal(calculated_advantage, expected)

    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "rewards",
        [
            torch.tensor([[2, 4, 6]], dtype=torch.float32),
        ],
    )
    def test_calculate_advantage_raises_when_rewards_not_divisible_by_group_size(
        self,
        group_size,
        rewards,
    ):
        stub = _GrpoMathStub(group_size=group_size)
        with pytest.raises(ValueError) as e:
            stub._calculate_advantage(rewards)
        assert (
            f"Rewards must have a total element count divisible by group_size ({group_size}); got {rewards.numel()} elements."
            in str(e.value)
        )

    def test_calculate_advantage_mean_only_branch(self):
        stub = _GrpoMathStub(group_size=2, adv_norm="mean_only")
        rewards = torch.tensor([[1.0, 3.0], [4.0, 10.0]], dtype=torch.float32)
        calculated_advantage = stub._calculate_advantage(rewards)
        expected = (rewards - rewards.mean(dim=1, keepdim=True)).flatten().unsqueeze(1)
        assert torch.equal(calculated_advantage, expected)

    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "rewards",
        [
            torch.tensor([[2, 4, 6]], dtype=torch.float32),
        ],
    )
    def test_calculate_advantage_raises_when_rewards_not_divisible_by_group_size(
        self,
        group_size,
        rewards,
    ):
        stub = _GrpoMathStub(group_size=group_size)
        with pytest.raises(ValueError) as e:
            stub._calculate_advantage(rewards)
        assert (
            f"Rewards must have a total element count divisible by group_size ({group_size}); got {rewards.numel()} elements."
            in str(e.value)
        )

    def test_calculate_advantage_mean_only_branch(self):
        stub = _GrpoMathStub(group_size=2, adv_norm="mean_only")
        rewards = torch.tensor([[1.0, 3.0], [4.0, 10.0]], dtype=torch.float32)
        calculated_advantage = stub._calculate_advantage(rewards)
        expected = (rewards - rewards.mean(dim=1, keepdim=True)).flatten().unsqueeze(1)
        assert torch.equal(calculated_advantage, expected)


class TestGRPOCalculateKlDivergence:
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize("batch_size", [1])
    def test_calculate_kl_divergence(
        self,
        group_size,
        batch_size,
    ):
        stub = _GrpoMathStub(group_size=group_size)
        stub = _GrpoMathStub(group_size=group_size)
        normal_dist = torch.distributions.normal.Normal(0.0, 1.0)
        reference_log_probs = normal_dist.log_prob(torch.randn(batch_size))
        log_probs = normal_dist.log_prob(torch.randn(batch_size))
        kl = stub._calculate_kl_divergence(log_probs, reference_log_probs)
        kl = stub._calculate_kl_divergence(log_probs, reference_log_probs)
        assert torch.all(kl >= 0.0)
        assert isinstance(kl, torch.Tensor)
        assert kl.shape == log_probs.shape
        assert kl.shape == reference_log_probs.shape


class TestGRPOGrpoLossStandard:
    def test_grpo_loss_standard_kl_advantage_shaping_path(self):
        stub = _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.05,
            use_kl_advantage_shaping=True,
        )
        mask = torch.tensor([[True, True, False], [True, True, True]])
        log_probs = torch.tensor(
            [[0.2, 0.3, 0.0], [0.4, 0.1, -0.2]], dtype=torch.float32
        )
        old_log_probs = log_probs - 0.15
        reference_log_probs = log_probs + 0.05
        advantages = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)
        loss, kl = stub._grpo_loss_standard(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(kl)

    def test_grpo_loss_standard_kl_advantage_shaping_path(self):
        stub = _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.05,
            use_kl_advantage_shaping=True,
        )
        mask = torch.tensor([[True, True, False], [True, True, True]])
        log_probs = torch.tensor(
            [[0.2, 0.3, 0.0], [0.4, 0.1, -0.2]], dtype=torch.float32
        )
        old_log_probs = log_probs - 0.15
        reference_log_probs = log_probs + 0.05
        advantages = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)
        loss, kl = stub._grpo_loss_standard(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(kl)


class TestGRPOGspoLoss:
    def test_gspo_loss_path(self):
        stub = _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.05,
            use_kl_advantage_shaping=False,
        )
        mask = torch.tensor([[True, True, True], [True, False, True]])
        log_probs = torch.tensor(
            [[0.1, 0.2, 0.0], [0.3, 0.0, -0.1]], dtype=torch.float32
        )
        old_log_probs = log_probs - 0.2
        reference_log_probs = log_probs + 0.03
        advantages = torch.tensor([[0.75], [0.25]], dtype=torch.float32)
        loss, kl = stub._gspo_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(kl)

    def test_gspo_loss_path(self):
        stub = _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.05,
            use_kl_advantage_shaping=False,
        )
        mask = torch.tensor([[True, True, True], [True, False, True]])
        log_probs = torch.tensor(
            [[0.1, 0.2, 0.0], [0.3, 0.0, -0.1]], dtype=torch.float32
        )
        old_log_probs = log_probs - 0.2
        reference_log_probs = log_probs + 0.03
        advantages = torch.tensor([[0.75], [0.25]], dtype=torch.float32)
        loss, kl = stub._gspo_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(kl)


class TestGRPOCispoLoss:
    def test_cispo_loss_path(self):
        stub = _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.05,
            use_kl_advantage_shaping=False,
        )
        mask = torch.tensor([[True, True, True], [True, False, True]])
        log_probs = torch.tensor(
            [[0.1, 0.2, 0.0], [0.3, 0.0, -0.1]], dtype=torch.float32
        )
        old_log_probs = log_probs - 0.2
        reference_log_probs = log_probs + 0.03
        advantages = torch.tensor([[0.75], [0.25]], dtype=torch.float32)
        loss, kl = stub._cispo_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(kl)

    def test_cispo_loss_clamps_importance_ratio_on_both_sides(self):
        stub = _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.0,
            use_kl_advantage_shaping=False,
        )
        mask = torch.tensor([[True, True]])
        log_probs = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
        old_log_probs = torch.zeros_like(log_probs)
        reference_log_probs = log_probs.clone()
        advantages = torch.tensor([[1.0]], dtype=torch.float32)

        loss, kl = stub._cispo_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
        )

        # exp([-1, 1]) -> [0.367..., 2.718...] then clamp to [0.8, 1.2].
        expected_loss = torch.tensor(-0.2, dtype=torch.float32)
        assert torch.allclose(loss, expected_loss, atol=1e-6)
        assert torch.allclose(kl, torch.tensor(0.0, dtype=torch.float32), atol=1e-6)

    def test_cispo_loss_path(self):
        stub = _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.05,
            use_kl_advantage_shaping=False,
        )
        mask = torch.tensor([[True, True, True], [True, False, True]])
        log_probs = torch.tensor(
            [[0.1, 0.2, 0.0], [0.3, 0.0, -0.1]], dtype=torch.float32
        )
        old_log_probs = log_probs - 0.2
        reference_log_probs = log_probs + 0.03
        advantages = torch.tensor([[0.75], [0.25]], dtype=torch.float32)
        loss, kl = stub._cispo_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(kl)

    def test_cispo_loss_clamps_importance_ratio_on_both_sides(self):
        stub = _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.0,
            use_kl_advantage_shaping=False,
        )
        mask = torch.tensor([[True, True]])
        log_probs = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
        old_log_probs = torch.zeros_like(log_probs)
        reference_log_probs = log_probs.clone()
        advantages = torch.tensor([[1.0]], dtype=torch.float32)

        loss, kl = stub._cispo_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
        )

        # exp([-1, 1]) -> [0.367..., 2.718...] then clamp to [0.8, 1.2].
        expected_loss = torch.tensor(-0.2, dtype=torch.float32)
        assert torch.allclose(loss, expected_loss, atol=1e-6)
        assert torch.allclose(kl, torch.tensor(0.0, dtype=torch.float32), atol=1e-6)


class TestGRPOLoss:
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_loss(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        advantages = torch.arange(0, 10, device=grpo.device).unsqueeze(1)
        normal_dist = torch.distributions.normal.Normal(0.0, 1.0)
        reference_log_probs = normal_dist.log_prob(
            torch.randn(200, device=grpo.device)
        ).reshape(10, -1)
        old_log_probs = normal_dist.log_prob(
            torch.randn(200, device=grpo.device)
        ).reshape(10, -1)
        log_probs = normal_dist.log_prob(torch.randn(200, device=grpo.device)).reshape(
            10, -1
        )
        mask = torch.ones_like(log_probs)
        mask[:, -3:] = 0
        mask = mask.to(torch.bool)
        loss, kl = grpo._loss(
            batch_size=10,
            minibatch_idxs=torch.arange(10, device=grpo.device),
            completion_ids=torch.randint(
                0, vocab_size, (10, max_tokens + 1), device=grpo.device
            ),
            action_mask=mask,
            advantages=advantages,
            old_log_probs=old_log_probs,
            reference_log_probs=reference_log_probs,
        )
        assert loss != 0
        assert kl != 0
        grpo.clean_up()


class TestGRPOLearn:
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [6])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [6])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    @pytest.mark.parametrize("use_liger_loss", [False, True])
    def test_grpo_learn(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        batch_size,
        micro_batch_size_per_gpu,
        use_liger_loss,
    ):
        if use_vllm and use_liger_loss:
            pytest.skip("Skip vLLM learn path with liger in this mocked-call test.")
        mock_llm_instance = make_mock_vllm_instance(vllm.LLM)
        llm_patch_ctx = (
            patch("agilerl.algorithms.core.base.LLM", return_value=mock_llm_instance)
            if use_vllm
            else nullcontext()
        )
        with llm_patch_ctx:
            grpo = grpo_factory(
                accelerator_factory,
                model_factory,
                config,
                use_deepspeed_optimizer,
                vocab_size,
                input_size,
                max_tokens,
                group_size,
                use_separate_reference_adapter,
                use_vllm,
                pretrained_model_name_or_path,
                micro_batch_size_per_gpu,
                sleep_mode=True,
                use_liger_loss=use_liger_loss,
            )
        if use_vllm and use_liger_loss:
            pytest.skip("Skip vLLM learn path with liger in this mocked-call test.")
        mock_llm_instance = make_mock_vllm_instance(vllm.LLM)
        llm_patch_ctx = (
            patch("agilerl.algorithms.core.base.LLM", return_value=mock_llm_instance)
            if use_vllm
            else nullcontext()
        )
        with llm_patch_ctx:
            grpo = grpo_factory(
                accelerator_factory,
                model_factory,
                config,
                use_deepspeed_optimizer,
                vocab_size,
                input_size,
                max_tokens,
                group_size,
                use_separate_reference_adapter,
                use_vllm,
                pretrained_model_name_or_path,
                micro_batch_size_per_gpu,
                sleep_mode=True,
                use_liger_loss=use_liger_loss,
            )
        completions = [
            torch.randint(
                0,
                vocab_size,
                (group_size, input_size + max_tokens),
                device=grpo.device,
            )
            for _ in range(batch_size)
        ]
        action_masks = [
            torch.ones((group_size, input_size + max_tokens - 1), device=grpo.device)
            for _ in range(batch_size)
        ]
        rewards = torch.stack(
            [
                torch.rand(group_size, dtype=torch.float32)
                # Use larger, more differentiated rewards to produce meaningful advantages
                for _ in range(batch_size)
            ],
            dim=0,
        )

        for name, param in grpo.actor.named_parameters():
            if ("lora_A" in name or "lora_B" in name) and param is not None:
                param.data.normal_()

        pre_learn_actor_state_dict = copy.deepcopy(grpo.actor.state_dict())
        if use_vllm:
            grpo._vllm_awake = True
        with patch.object(
            grpo,
            "_prepare_vllm_for_training",
            wraps=grpo._prepare_vllm_for_training,
        ) as mock_prepare_vllm_for_training:
            learn_result = grpo.learn((completions, action_masks, rewards))
        assert mock_prepare_vllm_for_training.call_count == 1
        if use_vllm:
            mock_llm_instance.sleep.assert_called_once()
        mean_loss = learn_result["mean_loss"]
        mean_kl = learn_result["mean_kl"]
        if use_vllm:
            grpo._vllm_awake = True
        with patch.object(
            grpo,
            "_prepare_vllm_for_training",
            wraps=grpo._prepare_vllm_for_training,
        ) as mock_prepare_vllm_for_training:
            learn_result = grpo.learn((completions, action_masks, rewards))
        assert mock_prepare_vllm_for_training.call_count == 1
        if use_vllm:
            mock_llm_instance.sleep.assert_called_once()
        mean_loss = learn_result["mean_loss"]
        mean_kl = learn_result["mean_kl"]
        assert isinstance(mean_loss, float)
        assert isinstance(mean_kl, float)

        # Check that the actor network is updated
        for (param_name, param), (_, pre_learn_param) in zip(
            grpo.actor.state_dict().items(),
            pre_learn_actor_state_dict.items(),
            strict=False,
        ):
            if "actor" in param_name:
                assert not torch.equal(param, pre_learn_param)

            elif "reference" in param_name:
                assert torch.equal(param, pre_learn_param)

            else:
                assert torch.equal(param, pre_learn_param)
        grpo.clean_up()

    def test_learn_raises_when_rewards_count_mismatch(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2)
        completion_ids, action_masks = _build_branch_experiences(batch_size=3)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)
        with pytest.raises(
            ValueError, match="Rewards must provide one scalar per trajectory"
        ):
            grpo.learn((completion_ids, action_masks, rewards))
        grpo.clean_up()

    def test_learn_raises_when_batch_not_divisible_by_group_size(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2)
        completion_ids, action_masks = _build_branch_experiences(batch_size=3)
        rewards = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="must be divisible by group_size"):
            grpo.learn((completion_ids, action_masks, rewards))
        grpo.clean_up()

    def test_learn_filter_whiten_clip_branch_path_with_active_subset(self):
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2,
            filter_zero_adv=True,
            whiten_advantages=True,
            adv_clip_range=0.1,
            adv_filter_eps=0.05,
        )
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        rewards = torch.tensor([1.0, 0.0, -1.0, 2.0], dtype=torch.float32)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        fake_advantages = torch.tensor(
            [[0.0], [2.0], [-2.0], [0.0]], dtype=torch.float32
        )
        with (
            patch.object(grpo, "_calculate_advantage", return_value=fake_advantages),
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(
                grpo,
                "_loss",
                return_value=(
                    torch.tensor(1.0, dtype=torch.float32),
                    torch.tensor(0.1, dtype=torch.float32),
                ),
            ) as mock_grpo_loss,
            patch.object(grpo, "_backward_pass", return_value=None),
        ):
            metrics = grpo.learn((completion_ids, action_masks, rewards))
        processed_advantages = mock_grpo_loss.call_args.args[5]
        assert processed_advantages.abs().max().item() <= 0.100001
        assert metrics["mean_loss"] == pytest.approx(1.0)
        assert metrics["mean_kl"] == pytest.approx(0.1)
        grpo.clean_up()

    def test_learn_warns_and_returns_zeros_when_all_filtered(self):
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2,
            filter_zero_adv=True,
            adv_filter_eps=0.5,
            whiten_advantages=True,
        )
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        rewards = torch.tensor([1.0, 0.0, -1.0, 2.0], dtype=torch.float32)
        with (
            pytest.warns(
                UserWarning,
                match="All samples were filtered by advantage threshold; skipping GRPO update.",
            ),
            patch.object(
                grpo,
                "_calculate_advantage",
                return_value=torch.zeros(4, 1, dtype=torch.float32),
            ),
        ):
            metrics = grpo.learn((completion_ids, action_masks, rewards))
        assert metrics == {"mean_loss": 0.0, "mean_kl": 0.0}
        grpo.clean_up()

    def test_learn_warns_and_returns_zeros_when_no_active_samples_after_filtering(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2)
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        rewards = torch.tensor([1.0, 0.0, -1.0, 2.0], dtype=torch.float32)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        with (
            patch(
                "agilerl.algorithms.grpo.np.arange",
                return_value=np.array([], dtype=int),
            ),
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            pytest.warns(
                UserWarning,
                match="No active samples after filtering; skipping GRPO update.",
            ),
        ):
            metrics = grpo.learn((completion_ids, action_masks, rewards))
        assert metrics == {"mean_loss": 0.0, "mean_kl": 0.0}
        grpo.clean_up()

    def test_learn_empty_minibatch_branch_continues_without_grpo_step(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2, update_epochs=1)
        grpo.rng = SimpleNamespace(shuffle=lambda _x: None)
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)

        class EmptySlicingBatchIndices:
            def __len__(self):
                return 1

            def __getitem__(self, item):
                del item
                return np.array([], dtype=int)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        with (
            patch(
                "agilerl.algorithms.grpo.np.arange",
                return_value=EmptySlicingBatchIndices(),
            ),
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(
                grpo, "_loss", side_effect=AssertionError("should not be called")
            ),
        ):
            metrics = grpo.learn((completion_ids, action_masks, rewards))
        assert metrics == {"mean_loss": 0.0, "mean_kl": 0.0}
        grpo.clean_up()

    def test_learn_raises_when_rewards_count_mismatch(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2)
        completion_ids, action_masks = _build_branch_experiences(batch_size=3)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)
        with pytest.raises(
            ValueError, match="Rewards must provide one scalar per trajectory"
        ):
            grpo.learn((completion_ids, action_masks, rewards))
        grpo.clean_up()

    def test_learn_raises_when_batch_not_divisible_by_group_size(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2)
        completion_ids, action_masks = _build_branch_experiences(batch_size=3)
        rewards = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="must be divisible by group_size"):
            grpo.learn((completion_ids, action_masks, rewards))
        grpo.clean_up()

    def test_learn_filter_whiten_clip_branch_path_with_active_subset(self):
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2,
            filter_zero_adv=True,
            whiten_advantages=True,
            adv_clip_range=0.1,
            adv_filter_eps=0.05,
        )
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        rewards = torch.tensor([1.0, 0.0, -1.0, 2.0], dtype=torch.float32)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        fake_advantages = torch.tensor(
            [[0.0], [2.0], [-2.0], [0.0]], dtype=torch.float32
        )
        with (
            patch.object(grpo, "_calculate_advantage", return_value=fake_advantages),
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(
                grpo,
                "_loss",
                return_value=(
                    torch.tensor(1.0, dtype=torch.float32),
                    torch.tensor(0.1, dtype=torch.float32),
                ),
            ) as mock_grpo_loss,
            patch.object(grpo, "_backward_pass", return_value=None),
        ):
            metrics = grpo.learn((completion_ids, action_masks, rewards))
        processed_advantages = mock_grpo_loss.call_args.args[5]
        assert processed_advantages.abs().max().item() <= 0.100001
        assert metrics["mean_loss"] == pytest.approx(1.0)
        assert metrics["mean_kl"] == pytest.approx(0.1)
        grpo.clean_up()

    def test_learn_warns_and_returns_zeros_when_all_filtered(self):
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2,
            filter_zero_adv=True,
            adv_filter_eps=0.5,
            whiten_advantages=True,
        )
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        rewards = torch.tensor([1.0, 0.0, -1.0, 2.0], dtype=torch.float32)
        with (
            pytest.warns(
                UserWarning,
                match="All samples were filtered by advantage threshold; skipping GRPO update.",
            ),
            patch.object(
                grpo,
                "_calculate_advantage",
                return_value=torch.zeros(4, 1, dtype=torch.float32),
            ),
        ):
            metrics = grpo.learn((completion_ids, action_masks, rewards))
        assert metrics == {"mean_loss": 0.0, "mean_kl": 0.0}
        grpo.clean_up()

    def test_learn_warns_and_returns_zeros_when_no_active_samples_after_filtering(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2)
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        rewards = torch.tensor([1.0, 0.0, -1.0, 2.0], dtype=torch.float32)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        with (
            patch(
                "agilerl.algorithms.grpo.np.arange",
                return_value=np.array([], dtype=int),
            ),
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            pytest.warns(
                UserWarning,
                match="No active samples after filtering; skipping GRPO update.",
            ),
        ):
            metrics = grpo.learn((completion_ids, action_masks, rewards))
        assert metrics == {"mean_loss": 0.0, "mean_kl": 0.0}
        grpo.clean_up()

    def test_learn_empty_minibatch_branch_continues_without_grpo_step(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2, update_epochs=1)
        grpo.rng = SimpleNamespace(shuffle=lambda _x: None)
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)

        class EmptySlicingBatchIndices:
            def __len__(self):
                return 1

            def __getitem__(self, item):
                del item
                return np.array([], dtype=int)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        with (
            patch(
                "agilerl.algorithms.grpo.np.arange",
                return_value=EmptySlicingBatchIndices(),
            ),
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(
                grpo, "_loss", side_effect=AssertionError("should not be called")
            ),
        ):
            metrics = grpo.learn((completion_ids, action_masks, rewards))
        assert metrics == {"mean_loss": 0.0, "mean_kl": 0.0}
        grpo.clean_up()

    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_value_error_with_nan_loss(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        batch_size,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        completions = [
            torch.randint(
                0,
                vocab_size,
                (group_size, input_size + max_tokens),
                device=grpo.device,
            )
            for _ in range(batch_size)
        ]
        action_masks = [
            torch.randint(
                0,
                2,
                (group_size, input_size + max_tokens - 1),
                device=grpo.device,
            ).bool()
            for _ in range(batch_size)
        ]
        rewards = torch.stack(
            [torch.ones(group_size) for _ in range(batch_size)], dim=0
        )

        def mock_grpo_loss(*args, **kwargs):
            return torch.tensor(float("nan")), torch.tensor(1.0)

        with (
            patch.object(grpo, "_loss", side_effect=mock_grpo_loss),
            pytest.raises(ValueError) as value_error,
        ):
            grpo.learn((completions, action_masks, rewards))
        assert "Loss is not finite" in str(value_error.value)
        grpo.clean_up()

    def test_grpo_learn_raises_when_loss_not_finite(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config=None,
            use_deepspeed_optimizer=False,
            vocab_size=30,
            input_size=5,
            max_tokens=10,
            group_size=2,
            use_separate_reference_adapter=False,
            use_vllm=False,
            pretrained_model_name_or_path=None,
            micro_batch_size_per_gpu=None,
            from_name=False,
        )

        completions = [
            torch.randint(0, 30, (2, 15), device=grpo.device),
        ]
        action_masks = [torch.ones((2, 14), device=grpo.device, dtype=torch.bool)]
        rewards = torch.stack([torch.rand(2, dtype=torch.float32)], dim=0)

        with (
            patch.object(
                grpo,
                "_loss",
                return_value=(
                    torch.tensor(float("nan"), device=grpo.device),
                    torch.tensor(0.0, device=grpo.device),
                ),
            ),
            pytest.raises(ValueError, match="Loss is not finite"),
        ):
            grpo.learn((completions, action_masks, rewards))
        grpo.clean_up()

    def test_grpo_learn_runs_without_gradient_checkpointing_hooks(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config=None,
            use_deepspeed_optimizer=False,
            vocab_size=30,
            input_size=5,
            max_tokens=10,
            group_size=2,
            use_separate_reference_adapter=False,
            use_vllm=False,
            pretrained_model_name_or_path=None,
            micro_batch_size_per_gpu=None,
            from_name=False,
        )
        for name, param in grpo.actor.named_parameters():
            if ("lora_A" in name or "lora_B" in name) and param is not None:
                param.data.normal_(mean=0, std=0.01)

        completions = [
            torch.randint(0, 30, (2, 15), device=grpo.device),
        ]
        action_masks = [torch.ones((2, 14), device=grpo.device, dtype=torch.bool)]
        rewards = torch.stack([torch.rand(2, dtype=torch.float32)], dim=0)

        metrics = grpo.learn((completions, action_masks, rewards))
        assert set(metrics.keys()) == {"mean_loss", "mean_kl"}
        grpo.clean_up()

    def test_grpo_learn_calls_mps_empty_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
        accelerator_factory,
        model_factory,
    ) -> None:
        """Patch MPS on CI so ``torch.mps.empty_cache()`` in ``learn()`` is exercised."""
        empty = _patch_mps_learn_hooks(monkeypatch, "agilerl.algorithms.grpo")
        grpo = generate_grpo(
            accelerator_factory,
            model_factory,
            config=None,
            use_deepspeed_optimizer=False,
            vocab_size=30,
            input_size=5,
            max_tokens=10,
            group_size=2,
            use_separate_reference_adapter=False,
            use_vllm=False,
            pretrained_model_name_or_path=None,
            micro_batch_size_per_gpu=None,
            from_name=False,
        )
        for name, param in grpo.actor.named_parameters():
            if ("lora_A" in name or "lora_B" in name) and param is not None:
                param.data.normal_(mean=0, std=0.01)

        completions = [
            torch.randint(
                0,
                30,
                (2, 5 + 10),
                device=grpo.device,
            ),
        ]
        action_masks = [
            torch.ones((2, 5 + 10 - 1), device=grpo.device, dtype=torch.bool),
        ]
        rewards = torch.stack(
            [torch.rand(2, dtype=torch.float32) for _ in range(1)], dim=0
        )

        grpo.learn((completions, action_masks, rewards))
        empty.assert_called()
        grpo.clean_up()
        AcceleratorState._reset_state(True)


class TestGRPOGetLogprobs:
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [
            (False, TINY_LLM_FIXTURE_PATH),
            (False, None),
        ],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_get_logprobs(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        batch_size,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
            grpo.device,
        )

        log_probs = grpo._get_logprobs(ids=ids, batch_size=1)
        assert log_probs.shape == (ids.shape[0], ids.shape[1] - 1)
        grpo.clean_up()


class TestGRPOBackwardPass:
    @pytest.mark.parametrize("config", [None])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, None)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_get_backward_pass_with_scheduler(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        batch_size,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
            grpo.device,
        )
        loss = grpo.actor.forward(ids).logits.mean()
        grpo._backward_pass(loss)
        grpo.clean_up()


class TestGRPOLoad:
    def test_grpo_load(self):
        with pytest.raises(NotImplementedError):
            GRPO.load("path")


class TestGRPOSaveLoadCheckpoint:
    @pytest.mark.parametrize(
        "config, use_deepspeed_optimizer",
        [
            (deepspeed_config_stage_2, True),
            (None, False),
            (deepspeed_config_stage_1, True),
        ],
    )
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    @pytest.mark.parametrize("lora_only", [False, True])
    def test_grpo_save_load_checkpoint(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        tmpdir,
        micro_batch_size_per_gpu,
        lora_only,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo.save_checkpoint(tmpdir, lora_only=lora_only)
            new_grpo = GRPO(
                actor_network=model_factory(pretrained_model_name_or_path),
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                group_size=group_size,
                lora_config=copy.deepcopy(grpo.lora_config),
                cosine_lr_schedule_config=(
                    None
                    if accelerator is not None
                    else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
                ),
                use_vllm=use_vllm,
                accelerator=accelerator,
                use_separate_reference_adapter=use_separate_reference_adapter,
            )
            new_grpo.load_checkpoint(tmpdir, merge_lora_configs=False)

            for attr in EvolvableAlgorithm.inspect_attributes(grpo):
                if not attr.startswith("_") and not attr.startswith("__"):
                    if attr == "rng":
                        assert hasattr(new_grpo, attr)
                    elif attr == "actor":
                        for (name, param), (new_name, new_param) in zip(
                            grpo.actor.named_parameters(),
                            new_grpo.actor.named_parameters(),
                            strict=False,
                        ):
                            assert torch.allclose(
                                param,
                                new_param,
                            ), f"Parameter {name} is not equal (new_name: {new_name})"
                    elif attr == "optimizer":
                        for param, new_param in zip(
                            grpo.optimizer.parameters(),
                            new_grpo.optimizer.parameters(),
                            strict=False,
                        ):
                            assert torch.equal(param, new_param)
                    elif attr == "accelerator" or attr == "lr_scheduler":
                        assert (
                            getattr(new_grpo, attr).__class__.__name__
                            == getattr(grpo, attr).__class__.__name__
                        )
                    elif attr == "lora_config":
                        assert getattr(new_grpo, attr) is not None
                        assert getattr(grpo, attr) is not None
                        old_targets = set(getattr(grpo, attr).target_modules)
                        new_targets = set(getattr(new_grpo, attr).target_modules)
                        assert old_targets == new_targets
                        assert getattr(new_grpo, attr).r == getattr(grpo, attr).r
                        assert (
                            getattr(new_grpo, attr).lora_alpha
                            == getattr(grpo, attr).lora_alpha
                        )
                        assert (
                            getattr(new_grpo, attr).lora_dropout
                            == getattr(grpo, attr).lora_dropout
                        )
                    elif not isinstance(getattr(grpo, attr), torch.Tensor):
                        assert getattr(new_grpo, attr) == getattr(
                            grpo,
                            attr,
                        ), f"Attribute {attr} is not equal"
                    else:
                        if attr == "lora_config":
                            print(getattr(new_grpo, attr))
                            print(getattr(grpo, attr))
                        assert torch.equal(getattr(new_grpo, attr), getattr(grpo, attr))
        grpo.clean_up()
        new_grpo.clean_up()


class TestGRPOSaveLoadDistributedActor:
    @pytest.mark.parametrize("config", [None])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_save_load_distributed_actor_no_accelerator(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        batch_size,
        tmpdir,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        checkpoint_path = Path(tmpdir) / "checkpoint.pth"
        with pytest.warns(UserWarning):
            grpo._save_distributed_actor(checkpoint_path)

        with pytest.warns(UserWarning):
            grpo._load_distributed_actor(checkpoint_path)
        grpo.clean_up()

    @pytest.mark.parametrize(
        "config", [deepspeed_config_stage_2, deepspeed_config_stage_1]
    )
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_save_load_distributed_actor(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        tmpdir,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        checkpoint_path = Path(tmpdir) / "checkpoint.pth"
        grpo._save_distributed_actor(checkpoint_path)
        grpo_optim_state_dict = (
            grpo.actor.optimizer.state_dict()
            if use_deepspeed_optimizer
            else grpo.optimizer.state_dict()
        )
        grpo_optim_state_dict.pop("loss_scaler", None)
        new_grpo = GRPO(
            actor_network=model_factory(pretrained_model_name_or_path),
            pad_token_id=vocab_size - 1,
            pad_token="<pad>",
            device="cuda" if torch.cuda.is_available() else "cpu",
            group_size=group_size,
            lora_config=LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
            cosine_lr_schedule_config=(
                None
                if accelerator is not None
                else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
            ),
            accelerator=accelerator,
            use_separate_reference_adapter=use_separate_reference_adapter,
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
            new_grpo.actor.parameters(),
            grpo.actor.parameters(),
            strict=False,
        ):
            assert torch.equal(param, pre_learn_param)

        for key in new_opt.state_dict():
            if key == "loss_scaler":
                continue
            assert str(new_opt.state_dict()[key]) == str(grpo_optim_state_dict[key])
        grpo.clean_up()
        new_grpo.clean_up()

    @pytest.mark.skip(
        reason="This line adds no additional coverage, methods not dependent on vllm.",
    )
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(True, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_save_load_distributed_actor_vllm(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        tmpdir,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        checkpoint_path = Path(tmpdir) / "checkpoint.pth"
        grpo._save_distributed_actor(checkpoint_path)
        grpo_optim_state_dict = (
            grpo.actor.optimizer.state_dict()
            if use_deepspeed_optimizer
            else grpo.optimizer.state_dict()
        )
        grpo_optim_state_dict.pop("loss_scaler", None)
        new_grpo = GRPO(
            actor_network=model_factory(pretrained_model_name_or_path),
            pad_token_id=vocab_size - 1,
            pad_token="<pad>",
            device="cuda" if torch.cuda.is_available() else "cpu",
            group_size=group_size,
            lora_config=LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
            cosine_lr_schedule_config=(
                None
                if accelerator is not None
                else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
            ),
            accelerator=accelerator,
            use_separate_reference_adapter=use_separate_reference_adapter,
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
            new_grpo.actor.parameters(),
            grpo.actor.parameters(),
            strict=False,
        ):
            assert torch.equal(param, pre_learn_param)

        for key in new_opt.state_dict():
            if key == "loss_scaler":
                continue
            assert str(new_opt.state_dict()[key]) == str(grpo_optim_state_dict[key])
        grpo.clean_up()
        new_grpo.clean_up()


class TestGRPOClone:
    @pytest.mark.parametrize(
        "config", [deepspeed_config_stage_2, deepspeed_config_stage_1]
    )
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_clone_with_accelerator(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        tmpdir,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        grpo_accelerator = grpo.accelerator
        grpo_lr_scheduler = grpo.lr_scheduler
        grpo.fitness = [1, 2, 3]
        original_actor_state_dict = (
            grpo.actor.state_dict()
            if grpo.accelerator is None
            else grpo.accelerator.unwrap_model(grpo.actor).state_dict()
        )
        new_grpo = grpo.clone(index=1)

        # Check that the actor network is updated and the reference actor is not
        for (_name, cloned_param), param in zip(
            new_grpo.actor.state_dict().items(),
            original_actor_state_dict.values(),
            strict=False,
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
            strict=False,
        ):
            assert pg1["lr"] == pg2["lr"]
            assert pg1["weight_decay"] == pg2["weight_decay"]
            assert pg1["betas"] == pg2["betas"]
            assert pg1["eps"] == pg2["eps"]

        assert new_grpo.lr == grpo.lr
        assert new_grpo.batch_size_per_process == grpo.batch_size_per_process
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
        grpo.clean_up()
        new_grpo.clean_up()

    @spawn_new_process_for_each_test
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(True, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    @patch("agilerl.algorithms.core.base.LLM", DummyVLLM)
    def test_grpo_clone_with_accelerator_vllm(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        tmpdir,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        grpo_accelerator = grpo.accelerator
        grpo_lr_scheduler = grpo.lr_scheduler
        grpo.fitness = [1, 2, 3]
        original_actor_state_dict = (
            grpo.actor.state_dict()
            if grpo.accelerator is None
            else grpo.accelerator.unwrap_model(grpo.actor).state_dict()
        )
        new_grpo = grpo.clone(index=1)

        # Check that the actor network is updated and the reference actor is not
        for (_name, cloned_param), param in zip(
            new_grpo.actor.state_dict().items(),
            original_actor_state_dict.values(),
            strict=False,
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
            strict=False,
        ):
            assert pg1["lr"] == pg2["lr"]
            assert pg1["weight_decay"] == pg2["weight_decay"]
            assert pg1["betas"] == pg2["betas"]
            assert pg1["eps"] == pg2["eps"]

        assert new_grpo.lr == grpo.lr
        assert new_grpo.batch_size_per_process == grpo.batch_size_per_process
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
        assert isinstance(new_grpo.llm, DummyVLLM)
        grpo.clean_up()
        new_grpo.clean_up()


class TestGRPOTest:
    @pytest.mark.parametrize(
        "config",
        [None, deepspeed_config_stage_2, deepspeed_config_stage_1],
    )
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_test(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        batch_size,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        env = DummyReasoningEnv(vocab_size, input_size, batch_size, device=grpo.device)
        fitnesses = grpo.test(env)
        assert isinstance(fitnesses, np.ndarray)
        grpo.clean_up()

    def test_grpo_test_method_multiturn_episode_env_branch(
        self,
        grpo_factory,
        accelerator_factory,
        model_factory,
    ):
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

        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            None,
            False,
            100,
            10,
            8,
            2,
            False,
            False,
            None,
            None,
        )
        env = DummyMultiTurnEpisodeEnv()
        completion = torch.ones(1, 6, dtype=torch.long)
        action_mask = torch.ones(1, 5, dtype=torch.bool)
        with patch.object(
            grpo, "get_action", return_value=([completion], [action_mask])
        ) as get_action:
            out = grpo.test(env, loop=2)

        assert out.shape == ()
        assert get_action.call_count == 4
        assert grpo.fitness[-1] == pytest.approx(1.0)
        grpo.clean_up()

    def test_grpo_test_method_invalid_env_type_raises(self):
        grpo = _make_cpu_grpo_for_branch_tests()
        with pytest.raises(
            TypeError,
            match=re.escape("env must be a ReasoningGym (or subclass) or MultiTurnEnv"),
        ):
            grpo.test(object(), loop=1)
        grpo.clean_up()


class TestCloneLlm:
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    def test_clone_llm_peft(self, vocab_size, input_size, max_tokens):
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

        # Create PEFT model (AgileRL requires the primary adapter to be named "actor")
        peft_model = get_peft_model(base_model, peft_config, adapter_name="actor")

        # Clone the PEFT model
        cloned_model = clone_llm(peft_model, 0, peft_model.state_dict())

        # Verify the cloned model is a PEFT model
        assert isinstance(cloned_model, type(peft_model))

        # Verify the configurations match
        assert cloned_model.config == peft_model.config
        assert cloned_model.peft_config == peft_model.peft_config

        # Verify the parameters match
        for (name1, param1), (name2, param2) in zip(
            cloned_model.named_parameters(),
            peft_model.named_parameters(),
            strict=False,
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)

        # Verify the model structure
        assert isinstance(cloned_model.model, type(base_model))

        # Verify the PEFT adapter is properly cloned
        assert cloned_model.active_adapter == peft_model.active_adapter
        assert cloned_model.peft_config[cloned_model.active_adapter] == peft_config

    def test_clone_llm_peft_raises_error(self):
        with pytest.raises(ValueError) as e:
            clone_llm(1, 1)
        assert "Invalid 'original_model' type: <class 'int'>" in str(e.value)


class TestGRPOCleanUp:
    @pytest.mark.parametrize(
        "config",
        [None, deepspeed_config_stage_2, deepspeed_config_stage_1],
    )
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_clean_up(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        batch_size,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        grpo.clean_up()
        assert grpo.actor is None
        assert grpo.optimizer is None
        assert grpo.lr_scheduler is None


class TestGRPOPreprocessObservation:
    @pytest.mark.parametrize("config", [None])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_preprocess_observation(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        batch_size,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        obs = grpo.preprocess_observation(
            orig_obs := torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )
        assert torch.equal(obs, orig_obs)
        grpo.clean_up()


class TestGRPOLoadDistributedActor:
    @pytest.mark.parametrize("config", [None])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_load_distributed_actor_value_error(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        batch_size,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        accelerator = MagicMock(spec=Accelerator)
        accelerator.state = MagicMock(spec=AcceleratorState)
        accelerator.free_memory.side_effect = lambda *args: [None] * len(args)
        grpo.accelerator = accelerator
        with pytest.raises(
            TypeError,
            match=r"(argument should be a str or an os\.PathLike object|expected str, bytes or os\.PathLike object).*not\s+'?NoneType'?",
        ):
            grpo._load_distributed_actor(None)
        grpo.clean_up()

    @pytest.mark.parametrize("config", [None])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_load_distributed_actor_warning(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        batch_size,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        with pytest.warns(
            UserWarning,
            match="Distributed actor load not supported for non-distributed training.",
        ):
            grpo._load_distributed_actor(None)
        grpo.clean_up()


class TestGRPOUpdateLr:
    @pytest.mark.parametrize(
        "config",
        [
            deepspeed_config_stage_2,
            deepspeed_config_stage_1,
            deepspeed_config_stage_1_with_scheduler,
        ],
    )
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_update_lr(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        opt = (
            grpo.optimizer.optimizer
            if not use_deepspeed_optimizer
            else grpo.actor.optimizer
        )
        grpo.accelerator, grpo.lr_scheduler = LLMAlgorithm.update_lr(
            opt,
            0.5,
            grpo.accelerator,
            grpo.cosine_lr_schedule_config,
        )
        for param_group in opt.param_groups:
            assert param_group["lr"] == 0.5

        if use_deepspeed_optimizer:
            grpo.accelerator.deepspeed_plugin.deepspeed_config["optimizer"]["params"][
                "lr"
            ] = 0.5

            if (
                grpo.accelerator.deepspeed_plugin.deepspeed_config.get(
                    "scheduler", None
                )
                is not None
            ):
                grpo.accelerator.deepspeed_plugin.deepspeed_config["scheduler"][
                    "params"
                ]["warmup_max_lr"] = 0.5
                grpo.accelerator.deepspeed_plugin.deepspeed_config["scheduler"][
                    "params"
                ]["num_epochs"] = 10
                grpo.accelerator.deepspeed_plugin.deepspeed_config["scheduler"][
                    "params"
                ]["warmup_proportion"] = 0.05
        grpo.clean_up()


class TestGRPOSetReferencePolicy:
    @pytest.mark.parametrize(
        "config",
        [None, deepspeed_config_stage_2, deepspeed_config_stage_1],
    )
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_set_reference_policy(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        reference_update_tracker = 0
        grpo.set_reference_policy(reference_update_tracker)
        input_ids = torch.tensor([[i + 1 for i in range(input_size + max_tokens)]]).to(
            grpo.device,
        )
        action_masks = torch.tensor([[1 for _ in range(input_size + max_tokens)]]).to(
            grpo.device,
        )
        output_before = grpo.actor(
            input_ids=input_ids,
            attention_mask=action_masks,
        ).logits
        assert grpo.reference_update_tracker == reference_update_tracker
        reference_update_tracker += 1
        grpo.set_reference_policy(reference_update_tracker)

        output_after = grpo.actor(
            input_ids=input_ids,
            attention_mask=action_masks,
        ).logits
        assert torch.allclose(output_before, output_after)
        assert grpo.reference_update_tracker == reference_update_tracker
        grpo.clean_up()

    @pytest.mark.parametrize(
        "config",
        [None, deepspeed_config_stage_2, deepspeed_config_stage_1],
    )
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [(False, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_ref_actor_is_same_as_actor_after_learning_reference_adapater(
        self,
        deepspeed_env,
        grpo_factory,
        model_factory,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        gc.collect()
        grpo = GRPO(
            actor_network=create_module(
                input_size=input_size,
                max_tokens=max_tokens,
                vocab_size=vocab_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            lr=0.1,
            pad_token_id=vocab_size - 1,
            pad_token="<pad>",
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
        grpo.clean_up()


class TestGRPORecompile:
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [2])
    @pytest.mark.parametrize(
        "use_vllm, pretrained_model_name_or_path",
        [
            (False, TINY_LLM_FIXTURE_PATH),
        ],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("data_batch_size", [4])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_exception_on_recompile(
        self,
        deepspeed_env,
        grpo_factory,
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        vocab_size,
        input_size,
        max_tokens,
        group_size,
        use_vllm,
        training,
        data_batch_size,
        micro_batch_size_per_gpu,
    ):
        grpo = grpo_factory(
            accelerator_factory,
            model_factory,
            config,
            use_deepspeed_optimizer,
            vocab_size,
            input_size,
            max_tokens,
            group_size,
            use_separate_reference_adapter,
            use_vllm,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        grpo.recompile()
        grpo.clean_up()


class TestGRPOSyncDeepspeedGradientClipping:
    def test_sync_deepspeed_returns_early_when_accelerator_is_none(self):
        """Test that the method returns early when accelerator is None."""
        mock_algorithm = MagicMock(spec=LLMAlgorithm)
        mock_algorithm.accelerator = None
        mock_algorithm.max_grad_norm = 1.0
        mock_algorithm.actor = MagicMock(spec=[])

        # Should return early without error (return None)
        result = LLMAlgorithm._sync_deepspeed_gradient_clipping(mock_algorithm)

        assert result is None

    def test_sync_deepspeed_returns_early_when_gradient_clipping_not_in_config(self):
        """Test that the method returns early when gradient_clipping is not in deepspeed config."""
        mock_ds_plugin = MagicMock()
        mock_ds_plugin.deepspeed_config = {"zero_optimization": {"stage": 2}}

        mock_state = MagicMock()
        mock_state.deepspeed_plugin = mock_ds_plugin

        mock_accelerator = MagicMock()
        mock_accelerator.state = mock_state

        mock_algorithm = MagicMock(spec=LLMAlgorithm)
        mock_algorithm.accelerator = mock_accelerator
        mock_algorithm.max_grad_norm = 1.0
        mock_algorithm.actor = MagicMock()

        LLMAlgorithm._sync_deepspeed_gradient_clipping(mock_algorithm)

        # Verify that the config was not modified (gradient_clipping should not be added)
        assert "gradient_clipping" not in mock_ds_plugin.deepspeed_config

    def test_sync_deepspeed_updates_gradient_clipping_when_different(self):
        """Test that gradient_clipping is updated when it differs from max_grad_norm."""
        mock_ds_plugin = MagicMock()
        mock_ds_plugin.deepspeed_config = {
            "zero_optimization": {"stage": 2},
            "gradient_clipping": 0.5,
        }

        mock_state = MagicMock()
        mock_state.deepspeed_plugin = mock_ds_plugin

        mock_accelerator = MagicMock()
        mock_accelerator.state = mock_state

        mock_algorithm = MagicMock(spec=LLMAlgorithm)
        mock_algorithm.accelerator = mock_accelerator
        mock_algorithm.max_grad_norm = 1.0
        mock_algorithm.actor = MagicMock(spec=[])  # No optimizer attribute

        LLMAlgorithm._sync_deepspeed_gradient_clipping(mock_algorithm)

        # Verify that gradient_clipping was updated to match max_grad_norm
        assert mock_ds_plugin.deepspeed_config["gradient_clipping"] == 1.0

    def test_sync_deepspeed_does_not_update_when_same(self):
        """Test that gradient_clipping is not modified when it matches max_grad_norm."""
        mock_ds_plugin = MagicMock()
        mock_ds_plugin.deepspeed_config = {
            "zero_optimization": {"stage": 2},
            "gradient_clipping": 1.0,
        }

        mock_state = MagicMock()
        mock_state.deepspeed_plugin = mock_ds_plugin

        mock_accelerator = MagicMock()
        mock_accelerator.state = mock_state

        mock_algorithm = MagicMock(spec=LLMAlgorithm)
        mock_algorithm.accelerator = mock_accelerator
        mock_algorithm.max_grad_norm = 1.0
        mock_algorithm.actor = MagicMock(spec=[])  # No optimizer attribute

        LLMAlgorithm._sync_deepspeed_gradient_clipping(mock_algorithm)

        # Verify gradient_clipping is still the same
        assert mock_ds_plugin.deepspeed_config["gradient_clipping"] == 1.0

    def test_sync_deepspeed_updates_actor_optimizer_grad_clip(self):
        """Test that actor.optimizer.grad_clip is updated when it exists."""
        mock_ds_plugin = MagicMock()
        mock_ds_plugin.deepspeed_config = {
            "zero_optimization": {"stage": 2},
            "gradient_clipping": 0.5,
        }

        mock_state = MagicMock()
        mock_state.deepspeed_plugin = mock_ds_plugin

        mock_accelerator = MagicMock()
        mock_accelerator.state = mock_state

        mock_optimizer = MagicMock()
        mock_optimizer.grad_clip = 0.5

        mock_actor = MagicMock()
        mock_actor.optimizer = mock_optimizer

        mock_algorithm = MagicMock(spec=LLMAlgorithm)
        mock_algorithm.accelerator = mock_accelerator
        mock_algorithm.max_grad_norm = 1.0
        mock_algorithm.actor = mock_actor

        LLMAlgorithm._sync_deepspeed_gradient_clipping(mock_algorithm)

        # Verify that optimizer grad_clip was updated
        assert mock_optimizer.grad_clip == 1.0

    def test_sync_deepspeed_updates_actor_optimizer_clip_grad(self):
        """Test that actor.optimizer.clip_grad is updated when it exists."""
        mock_ds_plugin = MagicMock()
        mock_ds_plugin.deepspeed_config = {
            "zero_optimization": {"stage": 2},
            "gradient_clipping": 0.5,
        }

        mock_state = MagicMock()
        mock_state.deepspeed_plugin = mock_ds_plugin

        mock_accelerator = MagicMock()
        mock_accelerator.state = mock_state

        mock_optimizer = MagicMock()
        mock_optimizer.clip_grad = 0.5

        mock_actor = MagicMock()
        mock_actor.optimizer = mock_optimizer

        mock_algorithm = MagicMock(spec=LLMAlgorithm)
        mock_algorithm.accelerator = mock_accelerator
        mock_algorithm.max_grad_norm = 1.0
        mock_algorithm.actor = mock_actor

        LLMAlgorithm._sync_deepspeed_gradient_clipping(mock_algorithm)

        # Verify that optimizer clip_grad was updated
        assert mock_optimizer.clip_grad == 1.0
