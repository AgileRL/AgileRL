# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

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
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim.lr_scheduler import SequentialLR
from transformers.generation.configuration_utils import GenerationConfig

from agilerl.algorithms import CISPO, GRPO, GSPO
from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.core.base import (
    EvolvableAlgorithm,
    LLMAlgorithm,
    OptimizerWrapper,
)
from agilerl.algorithms.grpo import (
    HAS_LIGER_KERNEL,
    LIGER_CLIP_FRACTION_METRIC,
    REFERENCE_KL_METRIC,
)
from agilerl.modules.dummy import DummyEvolvable
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig, clone_llm
from agilerl.utils.llm_utils import ReasoningGym
from tests import TINY_LLM_FIXTURE_PATH
from tests.test_algorithms.test_llms.llm_helpers import (
    DummyConfig,
    DummyMLPPreTrainedModel,
    _patch_mps_learn_hooks,
    create_module,
    deepspeed_config_stage_1,
    deepspeed_config_stage_1_with_scheduler,
    deepspeed_config_stage_2,
    deepspeed_config_stage_3,
)
from tests.utils import (
    assert_vllm_get_action_contract,
    make_mock_vllm_instance,
    spawn_new_process_for_each_test,
)


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
        )  # Change this to False.

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
        """Reset the prefix cache - dummy implementation."""

    def sleep(self, *args, **kwargs):
        pass

    def wake_up(self, *args, **kwargs):
        pass


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
        # Two knobs, both load-bearing for parallel vLLM testing:
        #
        # ``kv_cache_memory_bytes`` pins the KV cache to a tiny fixed size and
        # short-circuits vLLM's ``determine_available_memory`` profile-snapshot
        # assertion (which fires when peer xdist workers free GPU memory
        # mid-init). See ``VLLMConfig.kv_cache_memory_bytes`` docstring and
        # ``tests/conftest.py:pytest_collection_modifyitems`` for the full
        # rationale.
        #
        # ``gpu_memory_utilization`` is **also** load-bearing — it is *not*
        # ignored when ``kv_cache_memory_bytes`` is set. The startup check in
        # ``vllm/v1/worker/gpu_worker.py:init_device`` asserts ``free_memory
        # >= total_memory * gpu_memory_utilization`` *before* the KV-cache
        # path runs. With the vLLM default of 0.9 each worker would demand
        # ~13.1 GiB of the 14.58 GiB CI GPU, so the second concurrent worker
        # would always fail with ``Free memory on device (X/14.58 GiB) on
        # startup is less than desired GPU memory utilization (0.9, 13.12
        # GiB)``. 0.22 → ~3.2 GiB per worker, so 4 concurrent workers
        # (matching the ``-n 4`` cap on the vLLM CI step) fit in 12.8 GiB
        # with headroom.
        vllm_config = VLLMConfig(
            gpu_memory_utilization=0.22,
            kv_cache_memory_bytes=32 * 1024 * 1024,
            max_num_seqs=1,
            sleep_mode=sleep_mode,
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
    # ``from_name`` builds the trainer base from the model name rather than an
    # in-memory actor. Colocated vLLM and the trainer each hold their own base;
    # these tests mock the vLLM engine (no real engine to load), so the dummy
    # actor is passed as the trainer base — when ``base_model`` is non-None
    # ``_initialize_actors`` uses it directly (the trainer-first ordering and
    # any real base load are skipped). The vLLM engine is mocked by the caller.
    share_from_name = from_name
    grpo_kwargs = {
        "actor_network": actor if not share_from_name else None,
        "model_name": pretrained_model_name_or_path if share_from_name else None,
        "lr": 1e-5,
        "pad_token_id": vocab_size - 1,
        "pad_token": "<pad>",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "group_size": group_size,
        "lora_config": lora_config,
        "cosine_lr_schedule_config": (
            None
            if accelerator is not None
            else CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05)
        ),
        "accelerator": accelerator,
        "use_separate_reference_adapter": use_separate_reference_adapter,
        "use_vllm": use_vllm,
        "vllm_config": vllm_config,
        "max_output_tokens": max_tokens,
        "max_model_len": max_tokens + 5,
        "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "use_liger_loss": use_liger_loss,
    }
    return GRPO(**grpo_kwargs)


@pytest.fixture
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
        # Pin so the unfused learn() path is exercised by default
        # regardless of liger-kernel availability. Liger-specific tests
        # override this.
        "use_liger_loss": False,
    }
    defaults.update(kwargs)
    return GRPO(**defaults)


def _patch_surviving_sample_idxs(grpo, batch_idxs):
    """Patch the surviving-sample indices ``learn`` reads off ``_calculate_advantages``."""
    original = grpo._calculate_advantages

    def _with_batch_idxs(*args, **kwargs):
        advantages, _ = original(*args, **kwargs)
        return advantages, batch_idxs

    return patch.object(grpo, "_calculate_advantages", side_effect=_with_batch_idxs)


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


class _GrpoLossStub:
    def __init__(
        self,
        clip_coef_min: float,
        clip_coef_max: float,
        beta: float,
        use_kl_advantage_shaping: bool,
        importance_sampling_level: str = "token",
        group_size: int = 2,
        adv_norm: str = "mean_std",
        advantage_granularity: str = "auto",
        vllm_importance_sampling_cap: float = 2.0,
        turn_advantage_trajectory_fallback: bool = True,
        device: str = "cpu",
        loss_norm: str = "micro_batch",
    ) -> None:
        self.loss_norm = loss_norm
        self.clip_coef_min = clip_coef_min
        self.clip_coef_max = clip_coef_max
        self.beta = beta
        self.use_kl_advantage_shaping = use_kl_advantage_shaping
        self.importance_sampling_level = importance_sampling_level
        self.group_size = group_size
        self.adv_norm = adv_norm
        self.advantage_granularity = advantage_granularity
        self.vllm_importance_sampling_cap = vllm_importance_sampling_cap
        self.turn_advantage_trajectory_fallback = turn_advantage_trajectory_fallback
        self.device = device
        self.accelerator = None
        self._uses_deepspeed = False
        self._window_action_tokens = None

    _apply_kl_advantage_shaping = GRPO._apply_kl_advantage_shaping
    _reduce_masked_loss = GRPO._reduce_masked_loss
    _resolve_loss_window = GRPO._resolve_loss_window
    _accumulation_steps = GRPO._accumulation_steps
    _accumulation_steps_without_deepspeed = GRPO._accumulation_steps_without_deepspeed
    _log_importance_weights = GRPO._log_importance_weights
    _compute_policy_loss = GRPO._compute_policy_loss
    _grpo_loss_standard = GRPO._grpo_loss_standard
    _gspo_loss = GRPO._gspo_loss
    _cispo_loss = GRPO._cispo_loss
    _calculate_turn_advantage = GRPO._calculate_turn_advantage
    _calculate_advantage = GRPO._calculate_advantage
    _resolve_advantage_granularity = GRPO._resolve_advantage_granularity
    _align_sampling_logprobs = GRPO._align_sampling_logprobs
    _sampling_mismatch_metrics = GRPO._sampling_mismatch_metrics
    _aligned_sampling_logprobs_and_metrics = GRPO._aligned_sampling_logprobs_and_metrics
    _assert_batch_divisible_by_group = GRPO._assert_batch_divisible_by_group
    _turn_broadcast_advantages = GRPO._turn_broadcast_advantages
    _trajectory_advantages = GRPO._trajectory_advantages
    _whiten_advantages = GRPO._whiten_advantages


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

    def test_cispo_gspo_signatures_match_grpo_minus_loss_type(self):
        # @inherit_init_signature must expose exactly GRPO's params minus
        # loss_type (no model construction needed — pure signature introspection).
        grpo_params = set(inspect.signature(GRPO.__init__).parameters) - {"loss_type"}
        for variant in (CISPO, GSPO):
            assert set(inspect.signature(variant.__init__).parameters) == grpo_params
            assert "loss_type" not in inspect.signature(variant).parameters
            assert "self" not in inspect.signature(variant).parameters

    @patch("agilerl.algorithms.core.base.LLM")
    def test_init_grpo_warns_when_hf_generate_chunk_size_set_with_vllm(
        self, MockLLM, model_factory
    ):
        mock_instance = make_mock_vllm_instance(vllm.LLM)
        MockLLM.return_value = mock_instance
        actor = model_factory(TINY_LLM_FIXTURE_PATH)
        # Colocated vLLM and the trainer each hold their own base. The vLLM
        # engine is mocked here; the tiny actor is passed as the trainer base
        # (``_initialize_actors`` uses it directly when ``base_model`` is given).
        with pytest.warns(
            UserWarning, match="hf_generate_chunk_size.*ignored.*use_vllm=True"
        ):
            grpo = GRPO(
                actor_network=actor,
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
        ("config", "use_deepspeed_optimizer"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        assert grpo.steps == 0
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
        # Colocated vLLM no longer supports tensor parallelism (the in-process
        # shared base assumes a single worker): TP>1 raises at construction.
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
            pytest.raises(ValueError, match="tensor_parallel_size==1"),
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
            pytest.raises(ValueError, match="tensor_parallel_size==1"),
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
                model_name=pretrained_model_name_or_path,
                actor_network=None,
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        with pytest.warns(
            UserWarning,
            match=r"No LoRA config provided\.\s+AgileRL can only be used to finetune adapters at present\.",
        ):
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
            pytest.raises(
                ValueError,
                match=r"Batch size \(17\) must be divisible by the number of processes \(2\)\.",
            ),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
            (
                {"group_size": 1},
                "group_size must be >= 2 for GRPO-style group-relative",
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

    def test_init_grpo_cispo_warns_when_beta_nonzero(self):
        with pytest.warns(UserWarning, match="CISPO is typically used with beta=0"):
            grpo = _make_cpu_grpo_for_branch_tests(loss_type="cispo", beta=0.1)
        grpo.clean_up()

    def test_init_gspo_overrides_non_trajectory_is_level_with_warning(self):
        """loss_type='gspo' is trajectory-level by definition: an explicit
        non-trajectory importance_sampling_level is overridden with a warning.
        """
        with pytest.warns(
            UserWarning, match="loss_type='gspo' implies trajectory-level"
        ):
            grpo = _make_cpu_grpo_for_branch_tests(
                loss_type="gspo", importance_sampling_level="token"
            )
        assert grpo.importance_sampling_level == "trajectory"
        grpo.clean_up()

    @pytest.mark.parametrize("is_level", [None, "trajectory"])
    def test_init_gspo_no_override_warning_when_level_unset_or_trajectory(
        self, is_level
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grpo = _make_cpu_grpo_for_branch_tests(
                loss_type="gspo", importance_sampling_level=is_level
            )
        assert not any("implies trajectory-level" in str(w.message) for w in caught)
        assert grpo.importance_sampling_level == "trajectory"
        grpo.clean_up()

    @pytest.mark.parametrize(
        ("loss_type", "is_level", "expected_supported"),
        [
            ("grpo", None, True),  # token level -> fused kernel
            ("grpo", "turn", False),  # no fused turn mode
            ("gspo", None, True),  # trajectory-level grpo objective
            ("cispo", None, True),  # cispo @ token
            ("cispo", "trajectory", False),  # cispo only fused at token level
        ],
    )
    def test_init_liger_level_supported_flag(
        self, loss_type, is_level, expected_supported
    ):
        """``_liger_level_supported`` routes ``_loss``: only token/trajectory
        GRPO and token-level CISPO have a fused Liger kernel.
        """
        grpo = _make_cpu_grpo_for_branch_tests(
            loss_type=loss_type,
            importance_sampling_level=is_level,
            beta=0.0,
        )
        assert grpo._liger_level_supported is expected_supported
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ] = 7
        with (
            pytest.raises(
                ValueError,
                match=r"Batch size \(16\) must be divisible by the product of the number of processes \(2\) and gradient accumulation steps \(7\)\.",
            ),
            patch(
                "accelerate.Accelerator.num_processes",
                new_callable=PropertyMock,
                return_value=2,
            ),
        ):
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
        ("config", "use_deepspeed_optimizer"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        assert grpo.steps == 0
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
        gc.collect()
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        with pytest.warns(
            UserWarning,
            match=r"DeepSpeed ZeRO Stage 3 is nascent and may not work as expected",
        ):
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
        with pytest.warns(
            UserWarning,
            match=r"Argument 'max_grad_norm' will overwrite the equivalent value set for 'gradient_clipping'",
        ):
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
        with pytest.warns(
            UserWarning,
            match=r"Argument 'max_grad_norm' will overwrite the equivalent value set for 'gradient_clipping'",
        ):
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
        with pytest.warns(
            UserWarning,
            match=r"Cannot specify the optimizer in the DeepSpeed config and use AgileRL's LR scheduler",
        ):
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
        with pytest.raises(
            ValueError,
            match=r"When specifying micro_batch_size_per_gpu, batch_size \(16\) must be divisible by the product of the number of processes",
        ):
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
        gc.collect()
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        with pytest.warns(
            UserWarning,
            match=r"No LoRA config provided\.\s+AgileRL can only be used to finetune adapters at present\.\s+Using default LoRA configuration for RL finetuning:",
        ):
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
    def test_init_grpo_separate_reference_adapter_no_deprecation_warning(
        self,
        deepspeed_env,
        accelerator_factory,
        config,
        use_deepspeed_optimizer,
    ):
        """use_separate_reference_adapter=True is the supported way to keep an
        updating reference policy and must not emit a DeprecationWarning.
        """
        accelerator = accelerator_factory(use_deepspeed_optimizer, config)
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
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
        assert not [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "use_separate_reference_adapter" in str(w.message)
        ]

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
                match=r"Liger loss was requested but `liger-kernel` is not available\. Set use_liger_loss=False\.",
            ):
                grpo._liger_loss(
                    batch_ids=torch.ones((1, 2), dtype=torch.long),
                    action_mask=torch.ones((1, 1), dtype=torch.bool),
                    advantages=torch.ones((1,), dtype=torch.float32),
                    old_log_probs=torch.zeros((1, 1), dtype=torch.float32),
                    reference_log_probs=torch.zeros((1, 1), dtype=torch.float32),
                )

        grpo.clean_up()
        AcceleratorState._reset_state(True)


class TestGRPOClipCoefTuple:
    """Cover the ``clip_coef`` tuple/list unpacking branch in
    :meth:`GRPO.__init__` (``clip_coef_min = float(clip_coef[0])`` etc).
    Default tests pass ``clip_coef`` as a single float, so this branch
    was otherwise uncovered.
    """

    def test_tuple_clip_coef_unpacks_into_min_and_max(self) -> None:
        grpo = _make_cpu_grpo_for_branch_tests(clip_coef=(0.15, 0.25))
        assert grpo.clip_coef_min == pytest.approx(0.15)
        assert grpo.clip_coef_max == pytest.approx(0.25)

    def test_list_clip_coef_unpacks_into_min_and_max(self) -> None:
        grpo = _make_cpu_grpo_for_branch_tests(clip_coef=[0.1, 0.3])
        assert grpo.clip_coef_min == pytest.approx(0.1)
        assert grpo.clip_coef_max == pytest.approx(0.3)

    def test_wrong_length_tuple_raises(self) -> None:
        with pytest.raises(
            ValueError, match="clip_coef tuple must contain exactly two values"
        ):
            _make_cpu_grpo_for_branch_tests(clip_coef=(0.1,))

    def test_negative_float_clip_coef_raises(self) -> None:
        with pytest.raises(
            ValueError, match="clip_coef must be greater than or equal to zero"
        ):
            _make_cpu_grpo_for_branch_tests(clip_coef=-0.1)

    @pytest.mark.parametrize("bad_value", ["not_a_number", {"min": 0.1}, None])
    def test_non_numeric_clip_coef_raises_typeerror(self, bad_value) -> None:
        with pytest.raises(
            TypeError,
            match="clip_coef must be a float or a tuple or list of two floats",
        ):
            _make_cpu_grpo_for_branch_tests(clip_coef=bad_value)


class TestGRPOLearnRewardsShape:
    """Cover the ``rewards.dim() > 1`` collapse branch in :meth:`GRPO.learn`.

    Default learn-path tests pass 1-D rewards; multi-turn rollouts produce
    [batch, max_turns] rewards that the algo collapses to per-trajectory
    scalars via ``rewards.sum(dim=1)``.
    """

    def test_multi_turn_rewards_are_summed_along_last_dim(self) -> None:
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2)
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        # Shape [batch, max_turns]; sums to [3, 7, 11, 15] post-collapse.
        rewards = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float32,
        )

        captured = {}

        def _capture_calculate_advantage(self, _rewards):
            captured["rewards_passed_to_adv"] = _rewards
            # Return per-sample advantages so the rest of learn() can proceed.
            return torch.zeros(_rewards.shape[0], 1, dtype=torch.float32)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        with (
            patch.object(
                GRPO,
                "_calculate_advantage",
                _capture_calculate_advantage,
            ),
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(
                grpo,
                "_loss",
                return_value=(
                    torch.tensor(0.5, dtype=torch.float32),
                    torch.tensor(0.0, dtype=torch.float32),
                ),
            ),
            patch.object(grpo, "_backward_pass", return_value=None),
        ):
            grpo.learn((completion_ids, action_masks, rewards))

        collapsed = captured["rewards_passed_to_adv"]
        # After collapse, rewards should be 1-D with summed per-trajectory values.
        assert collapsed.dim() == 1
        assert collapsed.tolist() == [3.0, 7.0, 11.0, 15.0]
        grpo.clean_up()


class TestGRPOLigerLossDispatch:
    """Cover the ``loss_type`` -> Liger-API dispatch table in
    :meth:`GRPO._liger_loss`. Each branch sets up ``liger_loss_type`` /
    ``importance_sampling_level`` / ``epsilon_*`` differently. We don't
    care about the actual Liger call result — patch ``_get_lm_head`` and
    ``LigerFusedLinearGRPOFunction.apply`` so the test stays CPU-only and
    runs whether or not ``liger-kernel`` is installed.
    """

    @pytest.mark.parametrize(
        (
            "loss_type",
            "expected_liger_loss_type",
            "expected_is_level",
            "expected_eps_high",
        ),
        [
            ("cispo", "cispo", "token", "clip_coef_max"),
            ("gspo", "grpo", "trajectory", "clip_coef_max - 1.0"),
        ],
    )
    def test_liger_loss_dispatches_per_loss_type(
        self,
        loss_type: str,
        expected_liger_loss_type: str,
        expected_is_level: str,
        expected_eps_high: str,
    ) -> None:
        grpo = _make_cpu_grpo_for_branch_tests(loss_type=loss_type, beta=0.0)
        fake_lm_head = nn.Linear(8, 16, bias=True)
        fake_loss = torch.tensor(0.5, requires_grad=True)
        fake_aux = (torch.tensor(0.1), torch.tensor(0.0))

        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch.object(grpo, "_get_lm_head", return_value=fake_lm_head),
            patch.object(grpo, "_patch_lm_head_to_identity", nullcontext),
            patch.object(
                grpo, "actor", new=MagicMock(wraps=grpo.actor)
            ),  # avoid touching the actor forward
            patch("agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction") as mock_fn,
            patch.object(
                LLMAlgorithm,
                "select_adapter",
                lambda self, name: nullcontext(),
            ),
        ):
            mock_fn.apply.return_value = (fake_loss, fake_aux)
            # With ``lm_head`` identity-patched, ``actor_output.logits`` *is*
            # the hidden-state tensor — return a stub whose ``.logits``
            # attribute is the fake hidden tensor.
            hidden = torch.randn(1, 2, 8, requires_grad=True)
            fake_output = MagicMock()
            fake_output.logits = hidden
            grpo.actor.side_effect = lambda **kwargs: fake_output

            grpo._liger_loss(
                batch_ids=torch.ones((1, 2), dtype=torch.long),
                action_mask=torch.ones((1, 1), dtype=torch.bool),
                advantages=torch.ones((1,), dtype=torch.float32),
                old_log_probs=torch.zeros((1, 1), dtype=torch.float32),
                reference_log_probs=torch.zeros((1, 1), dtype=torch.float32),
            )

        # Inspect the kwargs passed to ``LigerFusedLinearGRPOFunction.apply``.
        # ``apply`` takes positional args in a fixed order — pluck the ones we
        # care about by index based on ``_liger_loss``'s call signature:
        # (hidden, weight, target_ids, mask, adv, bias, ref, old, None,
        #  None, None, beta, epsilon_low, epsilon_high, liger_loss_type,
        #  max_output_tokens, importance_sampling_level, ...).
        call_args = mock_fn.apply.call_args
        positional = call_args.args
        assert positional[14] == expected_liger_loss_type
        assert positional[16] == expected_is_level
        if expected_eps_high == "clip_coef_max":
            assert positional[13] == grpo.clip_coef_max
        else:  # "clip_coef_max - 1.0"
            assert positional[13] == pytest.approx(grpo.clip_coef_max - 1.0)

    def test_token_level_sampling_logps_fuse_vllm_is_ratio(self) -> None:
        """token-level Liger + captured vLLM logprobs fuses the clamped
        trainer/vLLM ratio into the kernel (``vllm_is_ratio`` arg, pos 24).
        """
        grpo = _make_cpu_grpo_for_branch_tests(loss_type="grpo", beta=0.0)
        fake_lm_head = nn.Linear(8, 16, bias=True)
        fake_aux = (torch.tensor(0.1), torch.tensor(0.0))
        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch.object(grpo, "_get_lm_head", return_value=fake_lm_head),
            patch.object(grpo, "_patch_lm_head_to_identity", nullcontext),
            patch.object(grpo, "actor", new=MagicMock(wraps=grpo.actor)),
            patch("agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction") as mock_fn,
            patch.object(
                LLMAlgorithm, "select_adapter", lambda self, name: nullcontext()
            ),
        ):
            mock_fn.apply.return_value = (
                torch.tensor(0.5, requires_grad=True),
                fake_aux,
            )
            fake_output = MagicMock()
            fake_output.logits = torch.randn(1, 2, 8, requires_grad=True)
            grpo.actor.side_effect = lambda **kwargs: fake_output
            grpo._liger_loss(
                batch_ids=torch.ones((1, 2), dtype=torch.long),
                action_mask=torch.ones((1, 1), dtype=torch.bool),
                advantages=torch.ones((1,), dtype=torch.float32),
                old_log_probs=torch.zeros((1, 1), dtype=torch.float32),
                reference_log_probs=torch.zeros((1, 1), dtype=torch.float32),
                sampling_log_probs=torch.full((1, 1), -0.5, dtype=torch.float32),
            )
        # vllm_is_ratio is the 24th positional arg (index 23): present, clamped.
        ratio = mock_fn.apply.call_args.args[23]
        assert ratio is not None
        assert torch.all(ratio <= grpo.vllm_importance_sampling_cap)

    @pytest.mark.parametrize("loss_type", ["grpo", "cispo"])
    @pytest.mark.parametrize("adv_shape", [(1,), (1, 1)])
    def test_token_level_handles_single_sample_minibatch(
        self, loss_type: str, adv_shape: tuple
    ) -> None:
        """A 1-sample minibatch must not crash the token-level Liger path.

        Regression: advantages for ``batch=1`` arrive as ``(1,)`` (or ``(1, 1)``);
        a naive ``squeeze(-1)`` collapses ``(1,)`` to a 0-dim scalar, which the
        token-level advantage-shape detection then rejects with a ``ValueError``.
        """
        grpo = _make_cpu_grpo_for_branch_tests(loss_type=loss_type, beta=0.0)
        assert grpo.importance_sampling_level == "token"
        fake_lm_head = nn.Linear(8, 16, bias=True)
        fake_loss = torch.tensor(0.5, requires_grad=True)
        fake_aux = (torch.tensor(0.1), torch.tensor(0.0))

        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch.object(grpo, "_get_lm_head", return_value=fake_lm_head),
            patch.object(grpo, "_patch_lm_head_to_identity", nullcontext),
            patch.object(grpo, "actor", new=MagicMock(wraps=grpo.actor)),
            patch("agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction") as mock_fn,
            patch.object(
                LLMAlgorithm, "select_adapter", lambda self, name: nullcontext()
            ),
        ):
            mock_fn.apply.return_value = (fake_loss, fake_aux)
            fake_output = MagicMock()
            fake_output.logits = torch.randn(1, 2, 8, requires_grad=True)
            grpo.actor.side_effect = lambda **kwargs: fake_output

            loss, _ = grpo._liger_loss(
                batch_ids=torch.ones((1, 2), dtype=torch.long),
                action_mask=torch.ones((1, 1), dtype=torch.bool),
                advantages=torch.ones(adv_shape, dtype=torch.float32),
                old_log_probs=torch.zeros((1, 1), dtype=torch.float32),
                reference_log_probs=torch.zeros((1, 1), dtype=torch.float32),
            )
        # The per-token advantage broadcast must have received a (1,) vector,
        # not a 0-dim scalar — confirm via the kernel call and a finite loss.
        adv_per_token = mock_fn.apply.call_args.args[4]
        assert adv_per_token.shape == (1,)  # batch(1) * n_act(1)
        assert torch.isfinite(loss)

    def test_token_level_per_token_advantages_flattened(self) -> None:
        """``advantage_granularity='turn'`` broadcasts per-turn advantages to a
        per-token ``(batch, n_act)`` tensor upstream; the token-flatten Liger
        path must reshape it to ``(batch * n_act,)`` alongside the hidden
        states (and not broadcast it like the per-trajectory shapes).
        """
        grpo = _make_cpu_grpo_for_branch_tests(loss_type="grpo", beta=0.0)
        fake_lm_head = nn.Linear(8, 16, bias=True)
        fake_loss = torch.tensor(0.5, requires_grad=True)
        fake_aux = (torch.tensor(0.1), torch.tensor(0.0))

        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch.object(grpo, "_get_lm_head", return_value=fake_lm_head),
            patch.object(grpo, "_patch_lm_head_to_identity", nullcontext),
            patch.object(grpo, "actor", new=MagicMock(wraps=grpo.actor)),
            patch("agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction") as mock_fn,
            patch.object(
                LLMAlgorithm, "select_adapter", lambda self, name: nullcontext()
            ),
        ):
            mock_fn.apply.return_value = (fake_loss, fake_aux)
            fake_output = MagicMock()
            fake_output.logits = torch.randn(2, 3, 8, requires_grad=True)
            grpo.actor.side_effect = lambda **kwargs: fake_output

            per_token_adv = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
            grpo._liger_loss(
                batch_ids=torch.ones((2, 3), dtype=torch.long),
                action_mask=torch.ones((2, 2), dtype=torch.bool),
                advantages=per_token_adv,
                old_log_probs=torch.zeros((2, 2), dtype=torch.float32),
                reference_log_probs=torch.zeros((2, 2), dtype=torch.float32),
            )

        call_args = mock_fn.apply.call_args.args
        # Token-flattened layout: hidden (B*T, 1, H), targets/mask (B*T, 1).
        assert call_args[0].shape == (4, 1, 8)
        assert call_args[2].shape == (4, 1)
        assert call_args[3].shape == (4, 1)
        # Advantages flattened row-major, exactly the per-token values.
        # ``.cpu()``: ``_liger_loss`` moves tensors to the agent's device,
        # which is CUDA on a GPU host.
        assert torch.allclose(
            call_args[4].cpu(), torch.tensor([0.1, 0.2, 0.3, 0.4]), atol=1e-7
        )

    def test_token_level_unexpected_advantage_shape_raises(self) -> None:
        """A per-token advantage whose token dim disagrees with ``n_act`` is
        unmappable to the flattened layout and must be rejected.
        """
        grpo = _make_cpu_grpo_for_branch_tests(loss_type="grpo", beta=0.0)
        fake_lm_head = nn.Linear(8, 16, bias=True)

        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch.object(grpo, "_get_lm_head", return_value=fake_lm_head),
            patch.object(grpo, "_patch_lm_head_to_identity", nullcontext),
            patch.object(grpo, "actor", new=MagicMock(wraps=grpo.actor)),
            patch("agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction") as mock_fn,
            patch.object(
                LLMAlgorithm, "select_adapter", lambda self, name: nullcontext()
            ),
        ):
            fake_output = MagicMock()
            fake_output.logits = torch.randn(2, 3, 8, requires_grad=True)
            grpo.actor.side_effect = lambda **kwargs: fake_output

            with pytest.raises(ValueError, match="Unexpected advantage shape"):
                grpo._liger_loss(
                    batch_ids=torch.ones((2, 3), dtype=torch.long),
                    action_mask=torch.ones((2, 2), dtype=torch.bool),
                    advantages=torch.zeros((2, 3), dtype=torch.float32),
                    old_log_probs=torch.zeros((2, 2), dtype=torch.float32),
                    reference_log_probs=torch.zeros((2, 2), dtype=torch.float32),
                )
            mock_fn.apply.assert_not_called()


class _CtxFreeActor(nn.Module):
    """A context-independent "model": ``hidden = embed(input_ids)``.

    With no attention there is no cross-sequence contamination, so a packed
    forward must reproduce the padded forward's hidden states exactly at every
    real token — which is what lets us assert byte-level equivalence of the
    packed vs padded Liger path. Records the last ``input_ids`` shape so the
    test can confirm packing actually engaged (one ``(1, N)`` row vs ``(B, T)``).
    """

    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(self, input_ids=None, **kwargs):
        self.last_input_shape = tuple(input_ids.shape)
        return SimpleNamespace(logits=self.embed(input_ids))


class _DummyLigerFn:
    """Deterministic stand-in for ``LigerFusedLinearGRPOFunction``.

    Mirrors only the property that matters for packing equivalence: the loss is
    summed over masked-in (action) tokens and ignores everything else. The real
    kernel consumes ``hidden[:, :n_act]`` for the next-token shift, so we slice
    the hidden seq dim down to the mask's action length before reducing.
    """

    @staticmethod
    def apply(*args):
        policy_hidden = args[0]
        mask = args[3]
        h = policy_hidden
        if h.dim() == 3 and h.shape[1] != mask.shape[1]:
            h = h[:, : mask.shape[1], :]
        per_token = h.reshape(*mask.shape, -1).sum(-1)
        loss = (per_token * mask.to(per_token.dtype)).sum()
        return loss, [torch.zeros((), dtype=per_token.dtype)]


class TestGRPOLigerSequencePacking:
    """Sequence packing co-exists with the Liger fused-loss path.

    The transformer forward runs on a packed padding-free row, then the hidden
    states are scattered back onto the padded frame so the kernel call is
    unchanged. For a context-independent actor (no cross-sequence attention) the
    packed and padded paths must yield the same masked loss, and the packed run
    must actually feed the actor a single ``(1, N)`` row.
    """

    @pytest.mark.parametrize(
        ("loss_type", "expected_level"),
        [("grpo", "token"), ("cispo", "token"), ("gspo", "trajectory")],
    )
    def test_packed_liger_matches_padded(self, loss_type, expected_level):
        grpo = _make_cpu_grpo_for_branch_tests(loss_type=loss_type, beta=0.0)
        grpo.pad_token_id = 0
        assert grpo.importance_sampling_level == expected_level

        # Actor + lm_head live on grpo.device (cuda on a GPU box): _liger_loss
        # moves its inputs there, so the collaborators must match.
        vocab, hidden = 16, 8
        actor = _CtxFreeActor(vocab, hidden).to(grpo.device)
        grpo.actor = actor
        lm_head = nn.Linear(hidden, vocab).to(grpo.device)

        # Right-padded batch with varied real lengths -> packing has work to do.
        lengths = [5, 3, 4]
        b_size, t = len(lengths), max(lengths)
        torch.manual_seed(0)
        batch_ids = torch.zeros(b_size, t, dtype=torch.long)
        for b, length in enumerate(lengths):
            batch_ids[b, :length] = torch.randint(1, vocab, (length,))
        # Action mask over the (B, T-1) next-token frame: mark within-sequence
        # predictions (all masked-in positions are real, non-pad tokens).
        action_mask = torch.zeros(b_size, t - 1, dtype=torch.bool)
        for b, length in enumerate(lengths):
            action_mask[b, : length - 1] = True
        advantages = torch.randn(b_size)
        old_log_probs = torch.randn(b_size, t - 1)
        reference_log_probs = torch.zeros(b_size, t - 1)

        def run():
            with (
                patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
                patch(
                    "agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction",
                    _DummyLigerFn,
                ),
                patch.object(grpo, "_get_lm_head", return_value=lm_head),
                patch.object(grpo, "_patch_lm_head_to_identity", nullcontext),
                patch.object(grpo, "_get_unwrapped_actor", return_value=actor),
                patch.object(
                    LLMAlgorithm, "select_adapter", lambda self, name: nullcontext()
                ),
            ):
                loss, _ = grpo._liger_loss(
                    batch_ids=batch_ids,
                    action_mask=action_mask,
                    advantages=advantages,
                    old_log_probs=old_log_probs,
                    reference_log_probs=reference_log_probs,
                )
            return loss, actor.last_input_shape

        # Padded baseline.
        grpo.use_sequence_packing = False
        loss_padded, padded_shape = run()
        assert padded_shape == (b_size, t)

        # Packed: same numbers, but the forward sees one (1, N) row.
        grpo.use_sequence_packing = True
        grpo.model_config = {"attn_implementation": "flash_attention_2"}
        # Pre-set the consolidated warn-once flag to silence the canonical
        # non-token-IS memory notice (now owned by the base helper).
        grpo._liger_non_token_warned = True
        assert grpo._packing_mode() == "varlen"
        loss_packed, packed_shape = run()
        assert packed_shape == (1, sum(lengths))

        # The masked loss must be unchanged by packing, and non-trivial.
        assert torch.isfinite(loss_padded)
        assert loss_padded.abs() > 0
        assert torch.allclose(loss_packed, loss_padded, atol=1e-5)

    def test_packing_disabled_on_dense_backend_falls_back(self):
        """An unsupported (dense) backend disables packing: the forward stays
        padded even with ``use_sequence_packing=True``.
        """
        grpo = _make_cpu_grpo_for_branch_tests(loss_type="grpo", beta=0.0)
        grpo.pad_token_id = 0
        vocab, hidden = 16, 8
        actor = _CtxFreeActor(vocab, hidden).to(grpo.device)
        grpo.actor = actor
        lm_head = nn.Linear(hidden, vocab).to(grpo.device)
        grpo.use_sequence_packing = True
        grpo.model_config = {"attn_implementation": "sdpa"}

        batch_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
        action_mask = torch.tensor(
            [[True, True, False], [True, False, False]], dtype=torch.bool
        )
        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction", _DummyLigerFn
            ),
            patch.object(grpo, "_get_lm_head", return_value=lm_head),
            patch.object(grpo, "_patch_lm_head_to_identity", nullcontext),
            patch.object(grpo, "_get_unwrapped_actor", return_value=actor),
            patch.object(
                LLMAlgorithm, "select_adapter", lambda self, name: nullcontext()
            ),
            pytest.warns(UserWarning, match="varlen/block-sparse"),
        ):
            grpo._liger_loss(
                batch_ids=batch_ids,
                action_mask=action_mask,
                advantages=torch.randn(2),
                old_log_probs=torch.zeros(2, 3),
                reference_log_probs=torch.zeros(2, 3),
            )
        # Dense backend -> no packing -> padded (B, T) forward.
        assert actor.last_input_shape == (2, 4)


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
            completion_ids, action_masks, _ = grpo.get_action(
                prompts, training=training
            )
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
            completion_ids, action_masks, _ = grpo.get_action(prompts, training=False)
        assert len(completion_ids) == 1
        assert len(action_masks) == 1
        grpo.clean_up()

    def test_get_action_grpo_hf_repeats_single_row_stitch_ids_when_grouping(self):
        """When training with ``group_size > 1`` and a prompt carries a single-row
        ``stitch_prefix_ids`` tensor, the HF generate path must repeat the stitch
        prefix to match the grouped batch dimension. Otherwise downstream
        ``stitch_completion_after_windowed_hf_generate`` would receive a [1, N]
        prefix against a [group_size, T] completion.
        """
        grpo = _make_cpu_grpo_for_branch_tests(group_size=3)
        seq_len = 4
        prompts = [
            {
                "input_ids": torch.randint(0, 60, (1, seq_len), device=grpo.device),
                "attention_mask": torch.ones(1, seq_len, device=grpo.device),
                "stitch_prefix_ids": torch.tensor(
                    [[7, 8]], dtype=torch.long, device=grpo.device
                ),
                "initial_prompt_len": 2,
            },
        ]
        observed_stitch = {}

        def fake_actor_generate(input_ids, attention_mask, generation_config=None):
            # After repeat, the grouped input_ids should already have the group
            # dim baked in.
            return torch.cat(
                [input_ids, torch.full_like(input_ids[:, :2], 1)],
                dim=1,
            )

        def fake_stitch(completion_id, stitch, initial_len):
            observed_stitch["stitch_shape"] = (
                None if stitch is None else tuple(stitch.shape)
            )
            return completion_id, initial_len

        with (
            patch.object(grpo.actor, "generate", side_effect=fake_actor_generate),
            patch(
                "agilerl.algorithms.grpo.stitch_completion_after_windowed_hf_generate",
                side_effect=fake_stitch,
            ),
        ):
            completion_ids, _, _ = grpo.get_action(prompts, training=True)

        # The single-row stitch prefix should have been broadcast to group_size.
        assert observed_stitch["stitch_shape"] == (3, 2)
        assert completion_ids[0].shape[0] == 3
        grpo.clean_up()

    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [2])
    @pytest.mark.parametrize(
        ("use_vllm", "pretrained_model_name_or_path"),
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
                    None,
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
        [(True, TINY_LLM_FIXTURE_PATH)],
    )
    @pytest.mark.vllm
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_grpo_sync_actor_to_vllm(
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
        grpo._sync_actor_to_vllm()
        assert grpo._vllm_lora_loaded
        adapter_dir = grpo._vllm_lora_staging_dir / "actor"
        assert adapter_dir.is_dir()
        assert (adapter_dir / "adapter_config.json").is_file()
        loaded_loras = grpo.llm.llm_engine.list_loras()
        assert 1 in loaded_loras  # the fixed rollout LoRARequest id
        assert grpo._vllm_rollout_lora_request is not None

        grpo._vllm_moved = False
        grpo._sync_actor_to_vllm()

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
        completion_ids, action_masks, _ = grpo._generate_with_vllm_colocate(
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
            completion_ids, action_masks, _ = grpo._generate_with_vllm_colocate(
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
        with pytest.raises(
            ValueError,
            match=r"Rewards must have a total element count divisible by group_size",
        ):
            stub._calculate_advantage(rewards)

    def test_calculate_advantage_mean_only_branch(self):
        stub = _GrpoMathStub(group_size=2, adv_norm="mean_only")
        rewards = torch.tensor([[1.0, 3.0], [4.0, 10.0]], dtype=torch.float32)
        calculated_advantage = stub._calculate_advantage(rewards)
        expected = (rewards - rewards.mean(dim=1, keepdim=True)).flatten().unsqueeze(1)
        assert torch.equal(calculated_advantage, expected)


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

    def test_gspo_memory_efficient_pipeline_matches_naive(self):
        """End-to-end GSPO memory-efficient path: Stage-1 fused log-probs
        (vocab-chunked, gradient-checkpointed lm_head) feeding the
        trajectory-level Stage-2 loss must match computing log-probs from
        materialized ``(B, T, V)`` logits — in both loss value and the
        gradient back to the policy hidden states. This is the path GSPO
        uses with ``use_liger_loss=False`` (the fused-linear-logprob path is
        always on).
        """
        stub = _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.04,
            use_kl_advantage_shaping=False,
        )
        torch.manual_seed(0)
        B, T, H, V = 4, 6, 16, 1024  # T = action tokens (already shifted)
        weight = torch.randn(V, H) * 0.02
        bias = torch.randn(V) * 0.02
        targets = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[1, T - 2 :] = False  # ragged sequence
        advantages = torch.randn(B, 1)
        temperature = 0.7

        def naive_logps(hidden):
            logits = (hidden @ weight.t() + bias) / temperature
            return (
                torch.log_softmax(logits.float(), dim=-1)
                .gather(dim=-1, index=targets.unsqueeze(-1))
                .squeeze(-1)
            )

        # Frozen old/ref logprobs (no grad) from an arbitrary reference hidden.
        ref_hidden = torch.randn(B, T, H)
        with torch.no_grad():
            old_log_probs = naive_logps(ref_hidden) - 0.1
            reference_log_probs = naive_logps(ref_hidden) + 0.05

        base_hidden = torch.randn(B, T, H)

        # Naive path: materialize logits, log_softmax, then GSPO loss.
        hid_naive = base_hidden.clone().requires_grad_(True)
        loss_naive, kl_naive = stub._gspo_loss(
            mask,
            naive_logps(hid_naive),
            old_log_probs,
            reference_log_probs,
            advantages,
        )
        loss_naive.backward()

        # Fused path: vocab-chunked gradient-checkpointed log-probs, same loss.
        hid_fused = base_hidden.clone().requires_grad_(True)
        fused_logps = LLMAlgorithm._logprobs_from_hidden_fused_grad(
            hid_fused,
            weight,
            bias,
            targets,
            temperature=temperature,
            cast_to_fp32=True,
            chunk_rows=5,
        )
        loss_fused, kl_fused = stub._gspo_loss(
            mask,
            fused_logps,
            old_log_probs,
            reference_log_probs,
            advantages,
        )
        loss_fused.backward()

        assert torch.allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-5)
        assert torch.allclose(kl_fused, kl_naive, rtol=1e-4, atol=1e-5)
        assert torch.allclose(hid_fused.grad, hid_naive.grad, rtol=1e-4, atol=1e-5)


class TestGRPOTurnLevel:
    """Turn-level (pool-per-turn) importance sampling + advantages for the
    GRPO family. Turn-level sits between token and trajectory (GSPO) on the IS
    axis: pool the per-token log-ratio over each turn's tokens.
    """

    def _stub(self, **kwargs):
        defaults = {
            "clip_coef_min": 0.8,
            "clip_coef_max": 1.2,
            "beta": 0.04,
            "use_kl_advantage_shaping": False,
            "importance_sampling_level": "turn",
            "group_size": 2,
            "adv_norm": "mean_std",
        }
        defaults.update(kwargs)
        return _GrpoLossStub(**defaults)

    def test_log_importance_weights_turn_collapses_to_token(self):
        """With each token its own turn, turn pooling == token-level."""
        stub = self._stub()
        torch.manual_seed(0)
        B, T = 3, 6
        token_log_ratio = torch.randn(B, T)
        mask = torch.ones(B, T)
        mask[0, 5:] = 0
        turn_each = torch.arange(T).unsqueeze(0).repeat(B, 1)
        out = stub._log_importance_weights(token_log_ratio, mask, turn_each, "turn")
        assert torch.allclose(out * mask, token_log_ratio * mask, atol=1e-6)

    def test_log_importance_weights_turn_collapses_to_sequence(self):
        """A single turn covering the whole completion == trajectory-level."""
        stub = self._stub()
        torch.manual_seed(1)
        B, T = 4, 7
        token_log_ratio = torch.randn(B, T)
        mask = torch.ones(B, T)
        mask[1, 4:] = 0
        turn_one = torch.zeros(B, T, dtype=torch.long)
        seq = stub._log_importance_weights(token_log_ratio, mask, None, "trajectory")
        turn = stub._log_importance_weights(token_log_ratio, mask, turn_one, "turn")
        assert torch.allclose(turn * mask, seq * mask, atol=1e-6)

    def test_log_importance_weights_turn_manual(self):
        """Per-turn length-normalized mean, scattered back to tokens."""
        stub = self._stub()
        token_log_ratio = torch.tensor([[0.2, 0.4, 1.0, -0.2]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        turn_ids = torch.tensor([[0, 0, 1, -1]])
        out = stub._log_importance_weights(token_log_ratio, mask, turn_ids, "turn")
        turn0 = (0.2 + 0.4) / 2
        turn1 = 1.0
        assert out[0, 0].item() == pytest.approx(turn0)
        assert out[0, 1].item() == pytest.approx(turn0)
        assert out[0, 2].item() == pytest.approx(turn1)

    def test_calculate_turn_advantage_group_relative_per_turn(self):
        """Each turn is normalized within its group, independently."""
        stub = self._stub(group_size=2)
        rewards = torch.tensor(
            [[1.0, 2.0], [3.0, 0.0], [-1.0, 5.0], [1.0, 1.0]]
        )  # 2 groups of 2, 2 turns
        adv = stub._calculate_turn_advantage(rewards)
        assert adv.shape == (4, 2)
        # group 0 (rows 0,1), turn 0: [1,3] -> centered/std
        g = torch.tensor([1.0, 3.0])
        ref = (g - g.mean()) / (g.std() + 1e-8)
        assert torch.allclose(adv[:2, 0], ref, atol=1e-5)

    def test_turn_advantage_single_turn_matches_trajectory(self):
        """One turn reduces to the per-trajectory group-relative advantage."""
        stub = self._stub(group_size=2)
        rewards = torch.tensor([[1.0], [3.0], [-1.0], [2.0]])
        turn_adv = stub._calculate_turn_advantage(rewards).squeeze(-1)
        traj_adv = GRPO._calculate_advantage(stub, rewards.squeeze(-1)).squeeze(-1)
        assert torch.allclose(turn_adv, traj_adv, atol=1e-6)

    @pytest.mark.parametrize("objective", ["grpo", "cispo"])
    def test_turn_level_pipeline_matches_naive(self, objective):
        """End-to-end memory-efficient turn-level path: fused-grad Stage-1
        log-probs feeding the turn-pooled Stage-2 loss matches computing
        log-probs from materialized logits, in loss value and policy
        gradient.
        """
        stub = self._stub(importance_sampling_level="turn")
        loss_fn = stub._cispo_loss if objective == "cispo" else stub._grpo_loss_standard

        torch.manual_seed(0)
        B, T, H, V = 4, 6, 16, 1024
        weight = torch.randn(V, H) * 0.02
        bias = torch.randn(V) * 0.02
        targets = torch.randint(0, V, (B, T))
        temperature = 0.7
        mask = torch.ones(B, T)
        mask[1, 5:] = 0
        turn_ids = torch.zeros(B, T, dtype=torch.long)
        turn_ids[:, 3:] = 1
        turn_ids = torch.where(mask.bool(), turn_ids, torch.full_like(turn_ids, -1))
        turn_adv = stub._calculate_turn_advantage(torch.randn(B, 2))
        advantages = turn_adv.gather(1, turn_ids.clamp(min=0)) * mask  # (B, T)

        ref_hidden = torch.randn(B, T, H)

        def naive_logps(hidden):
            logits = (hidden @ weight.t() + bias) / temperature
            return (
                torch.log_softmax(logits.float(), dim=-1)
                .gather(dim=-1, index=targets.unsqueeze(-1))
                .squeeze(-1)
            )

        with torch.no_grad():
            old = naive_logps(ref_hidden) - 0.1
            ref = naive_logps(ref_hidden) + 0.05
        base = torch.randn(B, T, H)

        hid_naive = base.clone().requires_grad_(True)
        loss_naive, _ = loss_fn(
            mask, naive_logps(hid_naive), old, ref, advantages, turn_ids
        )
        loss_naive.backward()

        hid_fused = base.clone().requires_grad_(True)
        fused = LLMAlgorithm._logprobs_from_hidden_fused_grad(
            hid_fused,
            weight,
            bias,
            targets,
            temperature=temperature,
            cast_to_fp32=True,
            chunk_rows=5,
        )
        loss_fused, _ = loss_fn(mask, fused, old, ref, advantages, turn_ids)
        loss_fused.backward()

        assert torch.allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-5)
        assert torch.allclose(hid_fused.grad, hid_naive.grad, rtol=1e-4, atol=1e-4)


class TestGRPOAdvantageGranularityDecoupling:
    """advantage granularity (advantage_granularity) is independent of the IS /
    ratio-pooling level (importance_sampling_level) for the GRPO family.
    """

    @staticmethod
    def _stub(advantage_granularity, importance_sampling_level):
        return _GrpoLossStub(
            clip_coef_min=0.8,
            clip_coef_max=1.2,
            beta=0.0,
            use_kl_advantage_shaping=False,
            importance_sampling_level=importance_sampling_level,
            advantage_granularity=advantage_granularity,
        )

    @pytest.mark.parametrize(
        ("advantage_granularity", "is_level", "expected"),
        [
            ("auto", "token", "trajectory"),
            ("auto", "turn", "turn"),
            ("auto", "trajectory", "trajectory"),
            ("trajectory", "turn", "trajectory"),  # decoupled
            ("turn", "token", "turn"),  # decoupled
            ("turn", "trajectory", "turn"),
            ("trajectory", "token", "trajectory"),
        ],
    )
    def test_resolve_advantage_granularity(
        self, advantage_granularity, is_level, expected
    ):
        stub = self._stub(advantage_granularity, is_level)
        assert stub._resolve_advantage_granularity() == expected

    @pytest.mark.parametrize(
        ("is_level", "adv_shape"),
        [("token", "turn"), ("trajectory", "turn"), ("turn", "trajectory")],
    )
    def test_decoupled_advantage_is_combos_run(self, is_level, adv_shape):
        """Any (advantage granularity, IS level) pairing produces a finite
        loss — the surrogate broadcasts a per-turn (B,T) or per-trajectory
        (B,1) advantage against a token/turn/trajectory-pooled ratio.
        """
        stub = self._stub("auto", is_level)
        torch.manual_seed(0)
        B, T = 4, 6
        log_probs = torch.randn(B, T)
        old = log_probs - 0.2
        ref = log_probs + 0.05
        mask = torch.ones(B, T)
        mask[1, 5:] = 0
        turn_ids = torch.where(
            mask.bool(),
            (torch.arange(T) >= 3).long().unsqueeze(0).expand(B, T),
            torch.full((B, T), -1, dtype=torch.long),
        )
        if adv_shape == "turn":
            turn_adv = stub._calculate_turn_advantage(torch.randn(B, 2))
            advantages = turn_adv.gather(1, turn_ids.clamp(min=0)) * mask  # (B, T)
        else:
            advantages = stub._calculate_advantage(torch.randn(B))  # (B, 1)
        loss_turn_ids = turn_ids if is_level == "turn" else None
        loss, kl = stub._grpo_loss_standard(
            mask, log_probs, old, ref, advantages, loss_turn_ids
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(kl)

    def test_invalid_token_advantage_rejected(self):
        """GRPO has no token-level advantage; the constructor must reject it."""
        with pytest.raises(ValueError, match="advantage_granularity"):
            _make_cpu_grpo_for_branch_tests(advantage_granularity="token")


def _adv_stub(
    group_size: int = 2,
    adv_norm: str = "mean_only",
    turn_advantage_trajectory_fallback: bool = True,
):
    """Loss stub configured for the advantage-branch helpers extracted from
    ``learn`` (``_turn_broadcast_advantages`` / ``_trajectory_advantages``).
    ``mean_only`` keeps the hand-traced expectations integer-clean.
    """
    return _GrpoLossStub(
        clip_coef_min=0.8,
        clip_coef_max=1.2,
        beta=0.0,
        use_kl_advantage_shaping=False,
        group_size=group_size,
        adv_norm=adv_norm,
        turn_advantage_trajectory_fallback=turn_advantage_trajectory_fallback,
    )


class TestGRPOTurnBroadcastAdvantages:
    """``_turn_broadcast_advantages``: per-turn group-relative advantages
    gathered onto token positions via ``turn_ids`` and masked to actions.
    """

    def test_broadcasts_per_turn_advantages_to_tokens(self):
        stub = _adv_stub()
        # One group of 2 trajectories, 2 turns. mean_only group-centering:
        # turn 0 rewards [1, 3] -> [-1, 1]; turn 1 rewards [2, 0] -> [1, -1].
        rewards = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
        turn_ids = torch.tensor([[0, 0, 1, -1], [0, 1, 1, -1]])
        action_masks = torch.tensor(
            [[True, True, True, False], [True, True, True, False]]
        )
        out = stub._turn_broadcast_advantages(rewards, turn_ids, action_masks, 2)
        expected = torch.tensor([[-1.0, -1.0, 1.0, 0.0], [1.0, -1.0, -1.0, 0.0]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_flat_rewards_are_reshaped_per_sample(self):
        """Flat rewards (one row per (sample, turn)) reshape to (B, turns)."""
        stub = _adv_stub()
        rewards = torch.tensor([1.0, 2.0, 3.0, 0.0])  # -> [[1, 2], [3, 0]]
        turn_ids = torch.tensor([[0, 0, 1, -1], [0, 1, 1, -1]])
        action_masks = torch.tensor(
            [[True, True, True, False], [True, True, True, False]]
        )
        out = stub._turn_broadcast_advantages(rewards, turn_ids, action_masks, 2)
        expected = torch.tensor([[-1.0, -1.0, 1.0, 0.0], [1.0, -1.0, -1.0, 0.0]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_raises_when_turn_ids_exceed_reward_turns(self):
        stub = _adv_stub()
        rewards = torch.tensor([[1.0, 2.0], [3.0, 0.0]])  # 2 reward turns
        turn_ids = torch.tensor([[0, 1, 2, -1], [0, 1, 1, -1]])  # turn 2!
        action_masks = torch.ones(2, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="turn_ids reference a turn index beyond"):
            stub._turn_broadcast_advantages(rewards, turn_ids, action_masks, 2)

    def test_raises_when_batch_not_divisible_by_group_size(self):
        stub = _adv_stub(group_size=2)
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        turn_ids = torch.zeros(3, 2, dtype=torch.long)
        action_masks = torch.ones(3, 2, dtype=torch.bool)
        with pytest.raises(
            ValueError, match=r"Batch size \(3\) must be divisible by group_size \(2\)"
        ):
            stub._turn_broadcast_advantages(rewards, turn_ids, action_masks, 3)


class TestGRPOTrajectoryAdvantages:
    """``_trajectory_advantages``: collapse per-turn rewards to episode
    returns, then per-trajectory group-relative advantage ``(B, 1)``.
    """

    def test_collapses_per_turn_rewards_to_episode_returns(self):
        stub = _adv_stub()
        completion_ids = torch.zeros(2, 5, dtype=torch.long)
        # Per-turn rewards sum to [3, 6]; mean_only centering -> [-1.5, 1.5].
        rewards = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
        out = stub._trajectory_advantages(rewards, 2, completion_ids)
        assert torch.allclose(out, torch.tensor([[-1.5], [1.5]]), atol=1e-6)

    def test_accepts_flat_per_trajectory_rewards(self):
        stub = _adv_stub()
        completion_ids = torch.zeros(2, 5, dtype=torch.long)
        out = stub._trajectory_advantages(torch.tensor([1.0, 5.0]), 2, completion_ids)
        assert torch.allclose(out, torch.tensor([[-2.0], [2.0]]), atol=1e-6)

    def test_raises_when_rewards_do_not_collapse_to_one_per_trajectory(self):
        stub = _adv_stub()
        completion_ids = torch.zeros(2, 5, dtype=torch.long)
        with pytest.raises(
            ValueError, match="Rewards must provide one scalar per trajectory"
        ):
            stub._trajectory_advantages(
                torch.tensor([1.0, 2.0, 3.0]), 2, completion_ids
            )

    def test_raises_when_batch_not_divisible_by_group_size(self):
        stub = _adv_stub(group_size=2)
        completion_ids = torch.zeros(3, 5, dtype=torch.long)
        with pytest.raises(
            ValueError, match=r"Batch size \(3\) must be divisible by group_size \(2\)"
        ):
            stub._trajectory_advantages(
                torch.tensor([1.0, 2.0, 3.0]), 3, completion_ids
            )


class TestGRPOWhitenAdvantages:
    """``_whiten_advantages``: shape-aware whitening over (B, 1) per-trajectory
    and (B, T-1) per-token advantages, with active-sample masking and a
    <=1-active guard (variance undefined).
    """

    @staticmethod
    def _stub():
        return _adv_stub()

    def test_per_trajectory_whitens_across_samples(self):
        stub = self._stub()
        adv = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        action_masks = torch.ones(4, 3, dtype=torch.bool)
        out = stub._whiten_advantages(adv, action_masks, None)
        # masked_whiten: (v - mean) * rsqrt(unbiased_var + 1e-8);
        # mean = 2.5, var([1, 2, 3, 4]) = 5/3.
        flat = adv.reshape(-1)
        expected = (flat - flat.mean()) * torch.rsqrt(flat.var() + 1e-8)
        assert out.shape == (4, 1)
        assert torch.allclose(out, expected.reshape(4, 1), atol=1e-5)
        assert torch.allclose(
            out.reshape(-1),
            torch.tensor([-1.161895, -0.387298, 0.387298, 1.161895]),
            atol=1e-4,
        )

    def test_per_trajectory_active_mask_keeps_inactive_rows_untouched(self):
        stub = self._stub()
        adv = torch.tensor([[1.0], [2.0], [3.0], [100.0]])
        action_masks = torch.ones(4, 3, dtype=torch.bool)
        active = torch.tensor([True, True, True, False])
        out = stub._whiten_advantages(adv, action_masks, active)
        # Stats from the active rows only: mean([1, 2, 3]) = 2, var = 1 ->
        # active rows become [-1, 0, 1]; the filtered row keeps its raw value.
        assert torch.allclose(
            out, torch.tensor([[-1.0], [0.0], [1.0], [100.0]]), atol=1e-4
        )

    def test_per_trajectory_all_false_active_mask_falls_back_to_all_samples(self):
        stub = self._stub()
        adv = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        action_masks = torch.ones(4, 3, dtype=torch.bool)
        none_active = torch.zeros(4, dtype=torch.bool)
        out = stub._whiten_advantages(adv, action_masks, none_active)
        baseline = stub._whiten_advantages(adv, action_masks, None)
        assert torch.allclose(out, baseline, atol=1e-7)

    def test_per_token_whitens_over_action_positions_only(self):
        stub = self._stub()
        adv = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, -7.0]])
        action_masks = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        out = stub._whiten_advantages(adv, action_masks, None)
        # Stats over the 5 action tokens [1..5]: mean = 3, unbiased var = 2.5.
        scale = torch.rsqrt(torch.tensor(2.5) + 1e-8)
        expected = (adv - 3.0) * scale
        expected[1, 2] = -7.0  # non-action position keeps its raw value
        assert out.shape == adv.shape
        assert torch.allclose(out, expected, atol=1e-5)

    def test_per_token_active_mask_restricts_rows(self):
        stub = self._stub()
        adv = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        action_masks = torch.ones(2, 3)
        active = torch.tensor([True, False])
        out = stub._whiten_advantages(adv, action_masks, active)
        # Stats from row 0 only ([1, 2, 3]: mean 2, var 1); row 1 untouched.
        assert torch.allclose(out[0], torch.tensor([-1.0, 0.0, 1.0]), atol=1e-4)
        assert torch.allclose(out[1], adv[1], atol=1e-7)

    def test_single_active_value_guard_returns_unchanged(self):
        stub = self._stub()
        action_masks = torch.ones(2, 3, dtype=torch.bool)
        adv = torch.tensor([[5.0], [7.0]])
        active = torch.tensor([True, False])
        out = stub._whiten_advantages(adv, action_masks, active)
        assert torch.equal(out, adv)
        # Same guard on the per-token shape: one action token in total.
        adv_tok = torch.tensor([[5.0, 0.0]])
        single_mask = torch.tensor([[1.0, 0.0]])
        out_tok = stub._whiten_advantages(adv_tok, single_mask, None)
        assert torch.equal(out_tok, adv_tok)


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


class TestGRPOLoss:
    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        mean_loss = learn_result["loss"]
        mean_kl = learn_result["kl"]
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
        mean_loss = learn_result["loss"]
        mean_kl = learn_result["kl"]
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
        assert metrics["loss"] == pytest.approx(1.0)
        assert metrics["kl"] == pytest.approx(0.1)
        grpo.clean_up()

    def test_filter_zero_adv_masks_instead_of_dropping_under_dp(self):
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2,
            filter_zero_adv=True,
            adv_filter_eps=0.05,
        )
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        rewards = torch.tensor([1.0, 1.0, -1.0, 2.0], dtype=torch.float32)
        fake_advantages = torch.tensor(
            [[0.0], [2.0], [-2.0], [0.0]], dtype=torch.float32
        )
        grpo.accelerator = MagicMock(num_processes=2)
        with patch.object(
            grpo, "_trajectory_advantages", return_value=fake_advantages
        ):
            advantages, batch_idxs = grpo._calculate_advantages(
                rewards, completion_ids, action_masks, None
            )
        assert list(batch_idxs) == [0, 1, 2, 3]
        assert advantages[0].item() == 0.0
        assert advantages[3].item() == 0.0
        assert advantages[1].item() == pytest.approx(2.0)
        assert advantages[2].item() == pytest.approx(-2.0)
        grpo.accelerator = None
        grpo.clean_up()

    def test_filter_zero_adv_drops_samples_single_process(self):
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2,
            filter_zero_adv=True,
            adv_filter_eps=0.05,
        )
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        rewards = torch.tensor([1.0, 1.0, -1.0, 2.0], dtype=torch.float32)
        fake_advantages = torch.tensor(
            [[0.0], [2.0], [-2.0], [0.0]], dtype=torch.float32
        )
        with patch.object(
            grpo, "_trajectory_advantages", return_value=fake_advantages
        ):
            advantages, batch_idxs = grpo._calculate_advantages(
                rewards, completion_ids, action_masks, None
            )
        assert list(batch_idxs) == [1, 2]
        grpo.clean_up()

    def test_learn_records_the_window_action_tokens_of_active_samples(self):
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2,
            filter_zero_adv=True,
            adv_filter_eps=0.05,
            loss_norm="accumulation_window",
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
            ),
            patch.object(grpo, "_backward_pass", return_value=None),
        ):
            grpo.learn((completion_ids, action_masks, rewards))
        # Two of the four samples survive the filter, each with 9 action tokens.
        assert grpo._window_action_tokens == 18
        grpo.clean_up()

    def test_learn_records_window_tokens_per_optimizer_step(self):
        grpo = _make_cpu_grpo_for_branch_tests(loss_norm="accumulation_window")
        completion_ids, action_masks = _build_branch_experiences(batch_size=4)
        rewards = torch.tensor([1.0, 0.0, -1.0, 2.0], dtype=torch.float32)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        grpo._uses_deepspeed = True
        grpo.micro_batch_size_per_gpu = 1
        grpo.actor.gradient_accumulation_steps = lambda: 2
        original_record = GRPO._record_window_action_tokens
        window_sizes: list[int] = []

        def record_spy(masks, idxs):
            window_sizes.append(len(idxs))
            original_record(grpo, masks, idxs)

        with (
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
            ) as mock_loss,
            patch.object(grpo, "_backward_pass", return_value=None),
            patch.object(grpo, "_record_window_action_tokens", side_effect=record_spy),
        ):
            grpo.learn((completion_ids, action_masks, rewards))
        # Four micro-batches of one sample fold into two optimizer steps, so
        # the window normalizer is recorded once per step over two samples.
        assert window_sizes == [2, 2]
        assert grpo._window_action_tokens == 18
        assert mock_loss.call_count == 4
        grpo.clean_up()

    def test_learn_warns_when_micro_batches_straddle_optimizer_steps(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=3)
        completion_ids, action_masks = _build_branch_experiences(batch_size=3)
        rewards = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        grpo._uses_deepspeed = True
        grpo.micro_batch_size_per_gpu = 1
        grpo.actor.gradient_accumulation_steps = lambda: 2
        with (
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
            ),
            patch.object(grpo, "_backward_pass", return_value=None),
            pytest.warns(UserWarning, match="whole optimizer steps"),
        ):
            grpo.learn((completion_ids, action_masks, rewards))
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
        assert metrics == {"loss": 0.0, "kl": 0.0}
        grpo.clean_up()

    def test_learn_early_return_waits_for_everyone_when_all_filtered(self):
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2,
            filter_zero_adv=True,
            adv_filter_eps=0.5,
            whiten_advantages=True,
        )
        acc = MagicMock()
        acc.num_processes = 2
        acc.free_memory.side_effect = lambda *objs: objs
        grpo.accelerator = acc
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
        assert metrics == {"loss": 0.0, "kl": 0.0}
        acc.wait_for_everyone.assert_called_once()
        grpo.clean_up()

    def test_learn_raises_on_cross_rank_seq_len_mismatch(self):
        grpo = _make_cpu_grpo_for_branch_tests()
        grpo.zero_stage = 3
        acc = MagicMock()
        acc.num_processes = 2
        acc.device = torch.device("cpu")
        acc.gather.side_effect = lambda t: torch.tensor([3, 5], dtype=t.dtype)
        acc.free_memory.side_effect = lambda *objs: objs
        grpo.accelerator = acc
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)
        with pytest.raises(
            RuntimeError, match="Cross-rank completion sequence length mismatch"
        ):
            grpo.learn((completion_ids, action_masks, rewards))
        grpo.clean_up()

    def test_learn_empty_minibatch_branch_continues_without_grpo_step(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2, update_epochs=1)
        grpo.rng = SimpleNamespace(shuffle=lambda _x: None)
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)

        class EmptySlicingBatchIndices(np.ndarray):
            """One surviving sample whose minibatch slices all come back empty."""

            def __new__(cls):
                return np.zeros(1, dtype=int).view(cls)

            def __getitem__(self, item):
                if isinstance(item, slice):
                    return np.array([], dtype=int)
                return super().__getitem__(item)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        with (
            _patch_surviving_sample_idxs(grpo, EmptySlicingBatchIndices()),
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(
                grpo, "_loss", side_effect=AssertionError("should not be called")
            ),
        ):
            metrics = grpo.learn((completion_ids, action_masks, rewards))
        assert metrics["loss"] == 0.0
        assert metrics["kl"] == 0.0
        grpo.clean_up()

    @pytest.mark.parametrize("config", [deepspeed_config_stage_2])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        ("use_vllm", "pretrained_model_name_or_path"),
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
            pytest.raises(ValueError, match=r"Loss is not finite"),
        ):
            grpo.learn((completions, action_masks, rewards))
        grpo.clean_up()

    def test_grpo_learn_raises_when_peer_rank_has_nonfinite_loss(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2, update_epochs=1)
        mock_acc = MagicMock()
        mock_acc.num_processes = 2
        mock_acc.device = torch.device("cpu")
        mock_acc.free_memory.side_effect = lambda *objs: objs
        grpo.accelerator = mock_acc
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        with (
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(
                grpo,
                "_loss",
                return_value=(
                    torch.tensor(0.5, dtype=torch.float32),
                    torch.tensor(0.0, dtype=torch.float32),
                ),
            ),
            patch(
                "agilerl.algorithms.core.base.allreduce_minmax_int",
                return_value=(0, 1),
            ) as mock_reduce,
            patch.object(grpo, "_backward_pass") as mock_backward,
            pytest.raises(ValueError, match="Loss is not finite"),
        ):
            grpo.learn((completion_ids, action_masks, rewards))

        mock_reduce.assert_called_once_with(0, mock_acc)
        mock_backward.assert_not_called()
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
        assert set(metrics.keys()) == {"loss", "kl", "completion_length"}
        grpo.clean_up()

    @pytest.mark.gpu
    def test_grpo_learn_with_natively_loaded_fp16_checkpoint(self):
        """A checkpoint declaring fp16 trains without an explicit recast.

        transformers honors the checkpoint's ``dtype`` (the tiny_llm fixture
        stores float16), and under the bf16 autocast in ``_amp_ctx`` the final
        norm promotes hidden states to fp32 while the lm_head weight stays
        fp16 — the fused logprob matmul must reconcile the operand dtypes.
        """
        if not torch.cuda.is_available():
            pytest.skip("The autocast/checkpoint dtype mismatch requires CUDA.")
        from transformers import AutoModelForCausalLM

        actor = AutoModelForCausalLM.from_pretrained(
            TINY_LLM_FIXTURE_PATH, attn_implementation="sdpa"
        )
        # The fixture must keep declaring fp16 for this regression test to
        # exercise the mixed-dtype path.
        assert next(actor.parameters()).dtype == torch.float16

        vocab_size, input_size, max_tokens, group_size = 1000, 5, 10, 2
        grpo = GRPO(
            actor_network=actor,
            lr=1e-5,
            pad_token_id=vocab_size - 1,
            pad_token="<pad>",
            device="cuda",
            group_size=group_size,
            lora_config=LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
            max_output_tokens=max_tokens,
            max_model_len=input_size + max_tokens,
        )
        completions = [
            torch.randint(
                0, vocab_size, (group_size, input_size + max_tokens), device="cuda"
            )
            for _ in range(2)
        ]
        action_masks = [
            torch.ones((group_size, input_size + max_tokens - 1), device="cuda")
            for _ in range(2)
        ]
        rewards = torch.stack([torch.rand(group_size) for _ in range(2)], dim=0)

        metrics = grpo.learn((completions, action_masks, rewards))
        assert np.isfinite(metrics["loss"])
        assert np.isfinite(metrics["kl"])
        grpo.clean_up()

    def test_grpo_learn_calls_mps_empty_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
        accelerator_factory,
        model_factory,
    ) -> None:
        """Patch MPS on CI so ``torch.mps.empty_cache()`` in ``learn()`` is exercised."""
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
        # Patch MPS only *after* the agent is built: patching is_available()
        # before construction makes the device resolve to "mps", and the dummy
        # actor's ``.to("mps")`` then crashes on a non-MPS (Linux/CI) torch build.
        empty = _patch_mps_learn_hooks(monkeypatch, "agilerl.algorithms.grpo")
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("config", "use_deepspeed_optimizer"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
                # Match the saved agent's setting so the constructor doesn't
                # mutate ``lora_config`` differently (``use_liger_loss=True``
                # adds ``exclude_modules=["lm_head"]``).
                use_liger_loss=grpo.use_liger_loss,
            )
            new_grpo.load_checkpoint(tmpdir)

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
                    elif attr in {"accelerator", "lr_scheduler"}:
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


class TestLLMRestoreCheckpointAttributes:
    def test_live_device_and_lora_config_survive_restore(self):
        """A checkpoint written on another device must not relocate the loader.

        Population members share a checkpoint but occupy different GPUs, so the
        device recorded in the payload belongs to whichever member saved it.
        """
        agent = SimpleNamespace(
            device=torch.device("cuda", 2),
            lora_config="live",
            selected_adapters=("actor",),
        )
        checkpoint = {
            "device": torch.device("cuda", 0),
            "lora_config": "stale",
            "selected_adapters": ("actor", "reference"),
            "steps": 6,
        }

        LLMAlgorithm._restore_checkpoint_attributes(agent, checkpoint)

        assert agent.device == torch.device("cuda", 2)
        assert agent.lora_config == "live"
        assert agent.selected_adapters == ("actor",)
        assert agent.steps == 6


class TestGRPOSaveLoadDistributedActor:
    @pytest.mark.parametrize("config", [None])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        ("use_vllm", "pretrained_model_name_or_path"),
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
        with pytest.warns(
            UserWarning,
            match=r"Distributed actor save not supported for non-distributed training\.",
        ):
            grpo._save_distributed_actor(checkpoint_path)

        with pytest.warns(
            UserWarning,
            match=r"Distributed actor load not supported for non-distributed training\.",
        ):
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
            grpo, "get_action", return_value=ActionResult([completion], [action_mask])
        ) as get_action:
            out = grpo.test(env, loop=2)

        assert out.shape == ()
        # One real turn per episode (early terminate); no dummy padding turns.
        assert get_action.call_count == 2
        for call in get_action.call_args_list:
            assert call.args[0][0] is env.valid_prompt
        assert grpo.fitness[-1] == pytest.approx(1.0)
        grpo.clean_up()

    def test_grpo_test_method_waits_for_everyone(self):
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

        grpo = _make_cpu_grpo_for_branch_tests()
        acc = MagicMock()
        grpo.accelerator = acc
        completion = torch.ones(1, 6, dtype=torch.long)
        with patch.object(
            grpo, "get_action", return_value=ActionResult([completion], None)
        ):
            grpo.test(DummyMultiTurnEpisodeEnv(), loop=1)
        acc.wait_for_everyone.assert_called()

    def test_grpo_test_method_multiturn_continues_when_not_done(self):
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

        grpo = _make_cpu_grpo_for_branch_tests()
        completion = torch.ones(1, 6, dtype=torch.long)
        with patch.object(
            grpo, "get_action", return_value=ActionResult([completion], None)
        ) as get_action:
            out = grpo.test(DummyMultiTurnContinueEnv(), loop=1)

        assert out.shape == ()
        assert get_action.call_count == 2
        assert get_action.call_args_list[0].args[0][0]["input_ids"].shape[-1] == 4
        assert get_action.call_args_list[1].args[0][0]["input_ids"].shape[-1] == 5
        grpo.clean_up()

    def test_grpo_test_method_invalid_env_type_raises(self):
        grpo = _make_cpu_grpo_for_branch_tests()
        with pytest.raises(
            TypeError,
            match=re.escape(
                "env must be a ReasoningGym (or subclass) or TokenizedMultiTurnEnv"
            ),
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
        with pytest.raises(
            ValueError, match=r"Invalid 'original_model' type: <class 'int'>"
        ):
            clone_llm(1, 1)


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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        vocab_size = 1000
        input_size = 10
        max_tokens = 20
        group_size = 5
        gc.collect()
        with pytest.warns(UserWarning, match=r"DeepSpeed ZeRO Stage 3"):
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


class TestGRPOLoadDistributedActor:
    @pytest.mark.parametrize("config", [None])
    @pytest.mark.parametrize("use_deepspeed_optimizer", [False, True])
    @pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
    @pytest.mark.parametrize("vocab_size", [1000])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize("group_size", [5])
    @pytest.mark.parametrize(
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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
        ("use_vllm", "pretrained_model_name_or_path"),
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


class TestGRPOVLLMSamplingCorrection:
    """vLLM-vs-trainer sampling-mismatch correction (truncated importance
    sampling). The rollout is drawn from vLLM but the loss treats the trainer's
    recomputed ``old_log_probs`` as the behaviour policy; the correction
    reweights each token by ``clamp(exp(old - sampling), max=cap)``.
    """

    def _stub(self, **kwargs):
        defaults = {
            "clip_coef_min": 0.8,
            "clip_coef_max": 1.2,
            "beta": 0.0,
            "use_kl_advantage_shaping": False,
            "importance_sampling_level": "token",
            "group_size": 2,
            "adv_norm": "mean_std",
        }
        defaults.update(kwargs)
        return _GrpoLossStub(**defaults)

    @staticmethod
    def _lp(value):
        return SimpleNamespace(logprob=value)

    def test_sampled_token_logprobs_extraction(self):
        from agilerl.algorithms.core.base import _vllm_sampled_token_logprobs

        out = SimpleNamespace(
            token_ids=[5, 9, 2],
            logprobs=[
                {5: self._lp(-0.1), 1: self._lp(-3.0)},
                {9: self._lp(-0.5)},
                {2: self._lp(-1.2)},
            ],
        )
        assert _vllm_sampled_token_logprobs(out) == pytest.approx([-0.1, -0.5, -1.2])

    def test_sampled_token_logprobs_handles_missing_nan_and_none(self):
        from agilerl.algorithms.core.base import _vllm_sampled_token_logprobs

        # token 5 missing from its dict; token 9 has NaN -> both fall back to 0.0
        out = SimpleNamespace(
            token_ids=[5, 9],
            logprobs=[{1: self._lp(-3.0)}, {9: self._lp(float("nan"))}],
        )
        assert _vllm_sampled_token_logprobs(out) == [0.0, 0.0]
        # logprobs absent entirely -> zeros of the right length
        out2 = SimpleNamespace(token_ids=[1, 2, 3], logprobs=None)
        assert _vllm_sampled_token_logprobs(out2) == [0.0, 0.0, 0.0]

    def test_unit_ratio_when_sampling_equals_old(self):
        """Sampling == old -> ratio exp(0)=1 -> loss identical to baseline."""
        stub = self._stub()
        torch.manual_seed(0)
        mask = torch.ones(2, 3)
        log_probs = torch.randn(2, 3)
        old = log_probs - 0.1
        ref = log_probs.clone()
        adv = torch.tensor([[1.0], [-1.0]])
        base, _ = stub._compute_policy_loss(
            mask, log_probs, old, ref, adv, None, level="token", objective="grpo"
        )
        corr, _ = stub._compute_policy_loss(
            mask,
            log_probs,
            old,
            ref,
            adv,
            None,
            level="token",
            objective="grpo",
            sampling_log_probs=old.clone(),
        )
        assert torch.allclose(base, corr, atol=1e-7)

    def test_correction_matches_reference_token_grpo(self):
        """Exact match against a hand-rolled token-level GRPO reference."""
        cap = 2.0
        stub = self._stub(vllm_importance_sampling_cap=cap, beta=0.0)
        torch.manual_seed(2)
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        log_probs = torch.randn(2, 3)
        old = log_probs - torch.randn(2, 3) * 0.1
        ref = log_probs.clone()
        adv = torch.tensor([[0.7], [-0.4]])
        # Build a sampling tensor whose ratio spans below and above the clamp.
        sampling = old - torch.tensor([[0.3, 0.0, 1.5], [-0.2, 0.9, 0.0]])

        corr, _ = stub._compute_policy_loss(
            mask,
            log_probs,
            old,
            ref,
            adv,
            None,
            level="token",
            objective="grpo",
            sampling_log_probs=sampling,
        )

        ratio = torch.exp(log_probs - old)
        clipped = ratio.clamp(0.8, 1.2)
        loss_i = -torch.min(ratio * adv, clipped * adv)
        r = torch.exp((old - sampling) * mask).clamp(max=cap)
        loss_i = loss_i * r
        denom = mask.sum(-1).clamp(min=1.0)
        expected = ((loss_i * mask).sum(-1) / denom).mean()
        assert torch.allclose(corr, expected, atol=1e-6)

    def test_sampling_mismatch_metrics(self):
        stub = self._stub(vllm_importance_sampling_cap=2.0)
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        old = torch.tensor([[-0.5, -1.0, -9.0], [-0.2, -0.3, -2.0]])
        # log-diff over masked tokens: [0.1, 0.9, (masked)], [0.0, 1.0, 5.0]
        sampling = old - torch.tensor([[0.1, 0.9, 3.0], [0.0, 1.0, 5.0]])
        metrics = stub._sampling_mismatch_metrics(old, sampling, mask.bool())

        mask_b = mask.bool()
        log_diff = (old - sampling) * mask
        denom = mask.sum().clamp(min=1.0)
        exp_delta = (log_diff.abs().sum() / denom).item()
        ratios = torch.exp(log_diff).clamp(max=2.0)[mask_b]
        assert metrics["vllm_is_delta_mean"] == pytest.approx(exp_delta, rel=1e-6)
        assert metrics["vllm_is_ratio_mean"] == pytest.approx(
            ratios.mean().item(), rel=1e-6
        )
        # ratios that exceed cap (log-diff 0.9 -> 2.46 and 1.0 -> 2.72, 5.0 huge)
        # are clamped; fraction clamped is 3 of the 5 masked tokens.
        assert metrics["vllm_is_frac_clamped"] == pytest.approx(3 / 5, rel=1e-6)

    def test_align_scatters_flat_logprobs_onto_action_positions(self):
        stub = self._stub()
        # Two rows; action mask marks the generated tokens (varying counts,
        # non-contiguous to mimic prompt/observation gaps).
        action_masks = torch.tensor(
            [[False, True, True, False], [True, False, True, True]]
        )
        old = torch.full((2, 4), -9.0)
        flat = [torch.tensor([-0.1, -0.2]), torch.tensor([-0.3, -0.4, -0.5])]
        aligned, n_skipped = stub._align_sampling_logprobs(flat, action_masks, old)
        assert n_skipped == 0
        # Logprobs land exactly on the True positions, in order; elsewhere = old.
        expected = torch.tensor([[-9.0, -0.1, -0.2, -9.0], [-0.3, -9.0, -0.4, -0.5]])
        assert torch.allclose(aligned, expected, atol=1e-7)

    def test_align_count_mismatch_falls_back_to_old_for_that_row(self):
        stub = self._stub()
        action_masks = torch.tensor([[True, True, False], [True, True, True]])
        old = torch.tensor([[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]])
        # Row 0: 2 action tokens but only 1 logprob -> skip (keep old -> ratio 1).
        # Row 1: matches -> scattered.
        flat = [torch.tensor([-0.5]), torch.tensor([-0.3, -0.4, -0.6])]
        aligned, n_skipped = stub._align_sampling_logprobs(flat, action_masks, old)
        assert n_skipped == 1
        assert torch.allclose(aligned[0], old[0])  # untouched
        assert torch.allclose(aligned[1], torch.tensor([-0.3, -0.4, -0.6]), atol=1e-7)

    def test_align_none_and_empty_return_none(self):
        stub = self._stub()
        masks = torch.ones(2, 3, dtype=torch.bool)
        old = torch.zeros(2, 3)
        assert stub._align_sampling_logprobs(None, masks, old) == (None, 0)
        assert stub._align_sampling_logprobs([], masks, old) == (None, 0)

    def test_align_handles_per_row_none(self):
        stub = self._stub()
        action_masks = torch.tensor([[True, True], [True, True]])
        old = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
        flat = [None, torch.tensor([-0.1, -0.2])]
        aligned, n_skipped = stub._align_sampling_logprobs(flat, action_masks, old)
        assert n_skipped == 1
        assert torch.allclose(aligned[0], old[0])
        assert torch.allclose(aligned[1], torch.tensor([-0.1, -0.2]), atol=1e-7)

    def test_aligned_and_metrics_none_path(self):
        """No captured logprobs -> (None, {}) and no metrics computed."""
        stub = self._stub()
        masks = torch.ones(2, 3, dtype=torch.bool)
        old = torch.zeros(2, 3)
        assert stub._aligned_sampling_logprobs_and_metrics(None, masks, old) == (
            None,
            {},
        )
        assert stub._aligned_sampling_logprobs_and_metrics([], masks, old) == (
            None,
            {},
        )

    def test_aligned_and_metrics_full_match_no_skip_metric(self):
        """All rows align: metrics computed, no rows-skipped key, no warning."""
        stub = self._stub(vllm_importance_sampling_cap=2.0)
        masks = torch.ones(1, 3, dtype=torch.bool)
        old = torch.full((1, 3), -1.0)
        flat = [torch.full((3,), -1.5)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            aligned, metrics = stub._aligned_sampling_logprobs_and_metrics(
                flat, masks, old
            )
        assert torch.allclose(aligned, torch.full((1, 3), -1.5), atol=1e-7)
        # log-diff is 0.5 on every token: delta = 0.5, ratio = e^0.5 (< cap).
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.5, rel=1e-6)
        assert metrics["vllm_is_ratio_mean"] == pytest.approx(
            torch.exp(torch.tensor(0.5)).item(), rel=1e-6
        )
        assert metrics["vllm_is_frac_clamped"] == pytest.approx(0.0)
        assert "vllm_is_rows_skipped" not in metrics
        assert not any("token-count mismatch" in str(w.message) for w in caught)

    def test_aligned_and_metrics_skipped_rows_warns_and_sets_metric(self):
        """A row whose captured token count disagrees with the action mask is
        skipped (ratio 1 fallback), counted in the metrics, and warned once.
        """
        stub = self._stub(vllm_importance_sampling_cap=2.0)
        masks = torch.ones(2, 2, dtype=torch.bool)
        old = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
        # Row 0 aligns; row 1 has 1 logprob for 2 action tokens -> skipped.
        flat = [torch.full((2,), -1.5), torch.tensor([-0.5])]
        with pytest.warns(UserWarning, match="1/2 rows had a token-count mismatch"):
            aligned, metrics = stub._aligned_sampling_logprobs_and_metrics(
                flat, masks, old
            )
        assert torch.allclose(aligned[0], torch.full((2,), -1.5), atol=1e-7)
        assert torch.allclose(aligned[1], old[1], atol=1e-7)  # ratio-1 fallback
        assert metrics["vllm_is_rows_skipped"] == pytest.approx(1.0)
        # Row 0 contributes |log-diff| 0.5 per token, row 1 contributes 0.
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.25, rel=1e-6)

    def test_learn_liger_token_with_sampling_logps_uses_fused_kernel(self, monkeypatch):
        """token-level use_liger_loss=True + captured vLLM logprobs: the
        correction is fused into the kernel (``vllm_is_ratio``), so ``_loss``
        keeps the Liger path and threads ``sampling_log_probs`` through —
        no fallback warning.
        """
        monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True)
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2, update_epochs=1, use_liger_loss=True
        )
        assert grpo.importance_sampling_level == "token"
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)
        n_act = int(action_masks[0].sum())
        sampling_logps = [
            torch.full((n_act,), -3.0, dtype=torch.float32) for _ in range(2)
        ]

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        with (
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(grpo, "_backward_pass", return_value=None),
            patch.object(
                grpo,
                "_liger_loss",
                return_value=(
                    torch.tensor(0.5, requires_grad=True),
                    torch.tensor(0.1),
                ),
            ) as mock_liger_loss,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            grpo.learn(
                (completion_ids, action_masks, rewards),
                sampling_logps=sampling_logps,
            )
        # Liger path taken, with the correction threaded into it.
        mock_liger_loss.assert_called()
        assert mock_liger_loss.call_args.kwargs["sampling_log_probs"] is not None
        assert not any(
            "token-level importance sampling" in str(w.message) for w in caught
        )
        assert grpo._is_correction_liger_warned is False
        grpo.clean_up()

    def test_learn_liger_nontoken_with_sampling_logps_warns_and_uses_standard_path(
        self, monkeypatch
    ):
        """trajectory-level (GSPO) use_liger_loss=True + captured vLLM logprobs:
        the per-token reweight can't be pooled into the sequence ratio, so the
        correction warns once and runs the standard path (``_liger_loss`` not
        called).
        """
        monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True)
        monkeypatch.setattr("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True)
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2,
            update_epochs=1,
            use_liger_loss=True,
            importance_sampling_level="trajectory",
        )
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)
        n_act = int(action_masks[0].sum())
        sampling_logps = [
            torch.full((n_act,), -3.0, dtype=torch.float32) for _ in range(2)
        ]

        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        def fake_get_logprobs(ids, batch_size, use_reference=False, eval_mode=False):
            return torch.zeros(
                ids.shape[0],
                ids.shape[1] - 1,
                device=ids.device,
                requires_grad=True,
            )

        with (
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(grpo, "_get_logprobs", side_effect=fake_get_logprobs),
            patch.object(grpo, "_backward_pass", return_value=None),
            patch.object(grpo, "_liger_loss") as mock_liger_loss,
            pytest.warns(
                UserWarning,
                match="only at token-level importance sampling",
            ),
        ):
            metrics = grpo.learn(
                (completion_ids, action_masks, rewards),
                sampling_logps=sampling_logps,
            )
        # The fused kernel was bypassed in favour of the standard path.
        mock_liger_loss.assert_not_called()
        assert grpo._is_correction_liger_warned is True
        assert "vllm_is_delta_mean" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        grpo.clean_up()


class TestGRPONonFinitePaddingIsIsolated:
    """A non-finite logprob at a non-action position must not reach the update.

    Long fused forwards can emit NaN/Inf at padding slots; every masked
    reduction weights by the mask, and ``nan * 0`` is ``nan``, so without an
    explicit fill one pad slot NaNs the whole minibatch loss and the IS metrics.
    """

    def _stub(self, **kwargs):
        defaults = {
            "clip_coef_min": 0.8,
            "clip_coef_max": 1.2,
            "beta": 0.04,
            "use_kl_advantage_shaping": False,
            "importance_sampling_level": "token",
            "group_size": 2,
            "adv_norm": "mean_std",
        }
        defaults.update(kwargs)
        return _GrpoLossStub(**defaults)

    @staticmethod
    def _inputs(pad_value: float):
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        pad = ~mask.bool()
        log_probs = torch.full((2, 3), -0.5).masked_fill(pad, pad_value)
        old = torch.full((2, 3), -0.7).masked_fill(pad, pad_value)
        ref = torch.full((2, 3), -0.6).masked_fill(pad, pad_value)
        sampling = torch.full((2, 3), -0.9).masked_fill(pad, pad_value)
        adv = torch.tensor([[0.7], [-0.4]])
        return mask, log_probs, old, ref, sampling, adv

    @pytest.mark.parametrize("objective", ["grpo", "cispo"])
    @pytest.mark.parametrize("pad_value", [float("nan"), float("inf")])
    def test_standard_path_loss_and_kl_stay_finite(self, objective, pad_value):
        stub = self._stub()
        mask, log_probs, old, ref, sampling, adv = self._inputs(pad_value)

        loss, kl = stub._compute_policy_loss(
            mask,
            log_probs,
            old,
            ref,
            adv,
            None,
            level="token",
            objective=objective,
            sampling_log_probs=sampling,
        )

        assert torch.isfinite(loss)
        assert torch.isfinite(kl)

    def test_padding_value_does_not_change_the_loss(self):
        stub = self._stub()
        nan_args = self._inputs(float("nan"))
        finite_args = self._inputs(-123.0)

        def run(args):
            mask, log_probs, old, ref, sampling, adv = args
            return stub._compute_policy_loss(
                mask,
                log_probs,
                old,
                ref,
                adv,
                None,
                level="token",
                objective="cispo",
                sampling_log_probs=sampling,
            )

        nan_loss, nan_kl = run(nan_args)
        finite_loss, finite_kl = run(finite_args)

        assert torch.allclose(nan_loss, finite_loss, atol=1e-7)
        assert torch.allclose(nan_kl, finite_kl, atol=1e-7)

    def test_reduce_masked_loss_ignores_non_finite_padding(self):
        stub = self._stub()
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        loss = torch.tensor([[2.0, 4.0, float("nan")]])

        reduced = stub._reduce_masked_loss(loss, mask)

        assert reduced.tolist() == pytest.approx([3.0])

    def test_sampling_mismatch_metrics_stay_finite(self):
        stub = self._stub(vllm_importance_sampling_cap=2.0)
        mask, _, old, _, sampling, _ = self._inputs(float("nan"))

        metrics = stub._sampling_mismatch_metrics(old, sampling, mask.bool())

        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.2, rel=1e-5)

    def test_liger_kernel_receives_only_finite_inputs(self):
        grpo = _make_cpu_grpo_for_branch_tests(loss_type="cispo", beta=0.04)
        fake_lm_head = nn.Linear(8, 16, bias=True)
        nan = float("nan")
        # Action mask marks token 0 only; token 1 is padding and carries NaN in
        # every per-token tensor, including the hidden state feeding the kernel.
        action_mask = torch.tensor([[True, False]])
        hidden = torch.randn(1, 3, 8, requires_grad=True)
        hidden = hidden.masked_fill(
            torch.tensor([[[False], [True], [True]]]),
            nan,
        )

        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch.object(grpo, "_get_lm_head", return_value=fake_lm_head),
            patch.object(grpo, "_patch_lm_head_to_identity", nullcontext),
            patch.object(grpo, "actor", new=MagicMock(wraps=grpo.actor)),
            patch("agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction") as mock_fn,
            patch.object(
                LLMAlgorithm, "select_adapter", lambda self, name: nullcontext()
            ),
        ):
            mock_fn.apply.return_value = (
                torch.tensor(0.5, requires_grad=True),
                (torch.tensor(0.1), torch.tensor(0.0)),
            )
            fake_output = MagicMock()
            fake_output.logits = hidden
            grpo.actor.side_effect = lambda **kwargs: fake_output
            grpo._liger_loss(
                batch_ids=torch.ones((1, 3), dtype=torch.long),
                action_mask=action_mask,
                advantages=torch.ones((1,), dtype=torch.float32),
                old_log_probs=torch.tensor([[-0.7, nan]]),
                reference_log_probs=torch.tensor([[-0.6, nan]]),
                sampling_log_probs=torch.tensor([[-0.9, nan]]),
            )

        positional = mock_fn.apply.call_args.args
        policy_arg, ref_arg, old_arg, ratio_arg = (
            positional[0],
            positional[6],
            positional[7],
            positional[23],
        )
        for name, tensor in (
            ("policy_hidden", policy_arg),
            ("reference_log_probs", ref_arg),
            ("old_log_probs", old_arg),
            ("vllm_is_ratio", ratio_arg),
        ):
            assert torch.isfinite(tensor).all(), name
        # The in-mask token keeps its own values; only padding was filled.
        assert old_arg[0].item() == pytest.approx(-0.7)
        assert ratio_arg[1].item() == pytest.approx(1.0)
        grpo.clean_up()


class TestGRPORealLigerVllmCorrection:
    """Validate the *real* fused GRPO kernel's ``vllm_is_ratio`` semantics on
    GPU, exercising the exact positional, token-flattened ``(n_tokens, 1)`` call
    that :meth:`GRPO._liger_loss` makes (``vllm_is_ratio`` at positional index
    24). CPU CI cannot import the Triton kernel; this guards the position-24
    alignment against an unnoticed liger-kernel signature change.
    """

    _CONST_RATIO = 1.7

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not (HAS_LIGER_KERNEL and torch.cuda.is_available()),
        reason="The real Liger Triton/CUDA kernel requires liger-kernel + a GPU.",
    )
    @pytest.mark.parametrize("loss_type", ["cispo", "grpo"])
    def test_real_liger_kernel_vllm_is_ratio_gpu(self, loss_type):
        """ratio==1 is a no-op and a constant ratio ``c`` scales the loss by
        ``c`` at ``beta=0`` — the truncated-IS reweight the standard path also
        applies, here proven through the installed fused kernel.
        """
        from agilerl.algorithms.grpo import LigerFusedLinearGRPOFunction

        torch.manual_seed(0)
        device = "cuda"
        n_tokens, hidden_dim, vocab = 12, 16, 64
        epsilon_low = 0.2
        epsilon_high = 1.2 if loss_type == "cispo" else 0.2
        hidden = torch.randn(n_tokens, 1, hidden_dim, device=device)
        weight = torch.randn(vocab, hidden_dim, device=device) * 0.02
        bias = torch.randn(vocab, device=device) * 0.02
        target_ids = torch.randint(0, vocab, (n_tokens, 1), device=device)
        mask = torch.ones(n_tokens, 1, dtype=torch.bool, device=device)
        adv = torch.randn(n_tokens, device=device)
        old = torch.randn(n_tokens, 1, device=device) * 0.1

        def run(vllm_is_ratio):
            # Positional layout mirrors GRPO._liger_loss exactly (pos 24 ratio).
            loss, _ = LigerFusedLinearGRPOFunction.apply(
                hidden,
                weight,
                target_ids,
                mask,
                adv,
                bias,
                None,  # ref_per_token_logps
                old,
                None,  # ref_input
                None,  # ref_weight
                None,  # ref_bias
                0.0,  # beta
                epsilon_low,
                epsilon_high,
                loss_type,
                None,  # max_completion_length
                "token",
                None,  # sapo_temperature_pos
                None,  # sapo_temperature_neg
                1.0,  # temperature
                None,  # compiled
                False,  # use_ref_model
                1,  # chunk_size
                vllm_is_ratio,
            )
            return loss.detach()

        base = run(None)
        ones = run(torch.ones(n_tokens, 1, device=device))
        c = self._CONST_RATIO
        scaled = run(torch.full((n_tokens, 1), float(c), device=device))

        assert base.abs() > 1e-6
        assert torch.allclose(ones, base, atol=1e-5, rtol=1e-4)
        assert torch.allclose(scaled, c * base, atol=1e-5, rtol=1e-4)


class TestGRPOInitWarnings:
    def test_init_action_granularity_deprecated_warns_and_overrides(self):
        with pytest.warns(DeprecationWarning, match="action_granularity is deprecated"):
            grpo = _make_cpu_grpo_for_branch_tests(action_granularity="turn")
        assert grpo.advantage_granularity == "turn"
        grpo.clean_up()

    @pytest.mark.parametrize(
        ("level", "algo_name"), [("turn", "GRPO"), ("trajectory", "GSPO")]
    )
    def test_init_liger_non_token_level_warns_memory_unbounded(self, level, algo_name):
        with (
            patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True),
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            pytest.warns(UserWarning, match="NOT memory-bounded"),
        ):
            grpo = _make_cpu_grpo_for_branch_tests(
                use_liger_loss=True, importance_sampling_level=level
            )
        assert grpo._liger_non_token_warned
        grpo.clean_up()


class TestGRPOTurnAdvantageLearnPath:
    def test_calculate_turn_advantage_indivisible_batch_raises(self):
        stub = _GrpoMathStub(group_size=2, adv_norm="mean_std")
        stub._calculate_turn_advantage = GRPO._calculate_turn_advantage.__get__(stub)
        rewards = torch.ones(3, 2)  # 3 % 2 != 0
        with pytest.raises(ValueError, match="must be divisible by"):
            stub._calculate_turn_advantage(rewards)

    def _stubbed_forwards(self, grpo):
        def fake_fused_forward(ids, batch_size):
            shape = (ids.shape[0], ids.shape[1] - 1)
            zeros = torch.zeros(shape, dtype=torch.float32, device=ids.device)
            return zeros, zeros, None

        def fake_get_logprobs(ids, batch_size, use_reference=False, eval_mode=False):
            return torch.zeros(
                ids.shape[0], ids.shape[1] - 1, device=ids.device, requires_grad=True
            )

        return (
            patch.object(
                grpo, "_fused_forward_no_grad", side_effect=fake_fused_forward
            ),
            patch.object(grpo, "_get_logprobs", side_effect=fake_get_logprobs),
            patch.object(grpo, "_backward_pass", return_value=None),
        )

    def test_learn_turn_ids_batch_mismatch_raises(self):
        grpo = _make_cpu_grpo_for_branch_tests(group_size=2, update_epochs=1)
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        rewards = torch.tensor([1.0, -1.0])
        bad_turn_ids = torch.zeros(3, 9, dtype=torch.long)  # batch 3 != 2
        p1, p2, p3 = self._stubbed_forwards(grpo)
        with p1, p2, p3, pytest.raises(ValueError, match="must match"):
            grpo.learn((completion_ids, action_masks, rewards), turn_ids=bad_turn_ids)
        grpo.clean_up()

    def test_learn_turn_advantage_path_end_to_end(self):
        """Per-turn rewards + turn_ids route learn() through the turn-broadcast
        advantage branch (and stack turn_ids into the minibatches).
        """
        grpo = _make_cpu_grpo_for_branch_tests(
            group_size=2, update_epochs=1, advantage_granularity="turn"
        )
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        turn_rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (B, max_turns)
        turn_ids = torch.zeros(2, 9, dtype=torch.long)
        turn_ids[:, 5:] = 1
        p1, p2, p3 = self._stubbed_forwards(grpo)
        with p1, p2, p3:
            metrics = grpo.learn(
                (completion_ids, action_masks, turn_rewards), turn_ids=turn_ids
            )
        assert np.isfinite(metrics["loss"])
        grpo.clean_up()

    def test_learn_liger_turn_level_falls_back_to_standard_path(self):
        """Liger + turn-level IS has no fused kernel: learn() must warn and
        run the standard path (turn_ids stacked into the minibatches).
        """
        with (
            patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True),
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            pytest.warns(UserWarning, match="NOT memory-bounded"),
        ):
            grpo = _make_cpu_grpo_for_branch_tests(
                group_size=2,
                update_epochs=1,
                use_liger_loss=True,
                importance_sampling_level="turn",
                advantage_granularity="turn",
            )
        completion_ids, action_masks = _build_branch_experiences(batch_size=2)
        turn_rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        turn_ids = torch.zeros(2, 9, dtype=torch.long)
        turn_ids[:, 5:] = 1
        p1, p2, p3 = self._stubbed_forwards(grpo)
        with p1, p2, p3, patch.object(grpo, "_liger_loss") as mock_liger:
            metrics = grpo.learn(
                (completion_ids, action_masks, turn_rewards), turn_ids=turn_ids
            )
        mock_liger.assert_not_called()
        assert np.isfinite(metrics["loss"])
        grpo.clean_up()


@contextmanager
def _liger_available():
    """Make both liger gates report the kernel as installed."""
    with (
        patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True),
        patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
    ):
        yield


class TestGRPOAuxMetricNaming:
    """Live-tracker registration of the auxiliary scalar beside ``loss``.

    Naming and ``learn`` key routing are pinned in
    ``test_grpo_metric_naming.py``; this class only checks what a real
    ``GRPO`` metrics tracker registers at init.
    """

    def test_the_fused_path_registers_only_the_clip_fraction(self):
        with _liger_available():
            grpo = _make_cpu_grpo_for_branch_tests(use_liger_loss=True, beta=0.0)
        assert np.isnan(grpo.metrics.get_mean(LIGER_CLIP_FRACTION_METRIC))
        with pytest.raises(KeyError):
            grpo.metrics.get_mean(REFERENCE_KL_METRIC)
        grpo.clean_up()

    def test_the_standard_path_registers_only_kl(self):
        grpo = _make_cpu_grpo_for_branch_tests(beta=0.0)
        with pytest.raises(KeyError):
            grpo.metrics.get_mean(LIGER_CLIP_FRACTION_METRIC)
        grpo.clean_up()
