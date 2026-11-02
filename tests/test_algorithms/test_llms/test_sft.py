# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers", reason="LLM tests require transformers.")
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer

from agilerl.algorithms.core.base import EvolvableAlgorithm, OptimizerWrapper
from agilerl.algorithms.sft import SFT
from agilerl.arena.models.fsdp import FSDP_MEMORY_FIELDS
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import FSDPConfig, resolve_device
from agilerl.llm_envs import DatasetEnv
from agilerl.utils.llm_utils import LEARN_PHASE_METRIC_NAMES
from tests import TINY_LLM_FIXTURE_PATH
from tests.test_algorithms.test_llms.llm_helpers import (
    create_module,
    create_tiny_qwen3,
    optimizer_state,
    record_outputs,
    right_padded_batch,
    scale_losses,
    trainable_weights,
)
from tests.test_algorithms.test_llms.test_ppo_llm_passes import (
    patch_cuda_device,
    save_checkpoint_config,
)


def make_sft_gym(
    num_samples: int,
    tokenizer: AutoTokenizer,
    data_batch_size_per_gpu: int = 8,
    response_column: str = "response",
):
    train_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(num_samples)],
            response_column: [f"Response {i}" for i in range(num_samples)],
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(num_samples)],
            response_column: [f"Response {i}" for i in range(num_samples)],
        }
    )
    return DatasetEnv(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        objective="sft",
        data_batch_size_per_gpu=data_batch_size_per_gpu,
        response_column=response_column,
    )


@pytest.fixture
def sft_dataset_factory():
    return make_sft_gym


def generate_sft(
    dist_mode_factory,
    model_factory,
    dist_mode,
    vocab_size,
    input_size,
    max_tokens,
    pretrained_model_name_or_path,
    micro_batch_size_per_gpu,
    from_name=False,
    update_epochs=1,
):
    gc.collect()
    torch.cuda.empty_cache()

    dist_mode_factory(dist_mode)
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
    return SFT(
        actor_network=actor if not from_name else None,
        model_name=pretrained_model_name_or_path if from_name else None,
        pad_token_id=vocab_size - 1,
        pad_token="<pad>",
        lora_config=lora_config,
        fsdp_config=(
            FSDPConfig(routed_expert_chunk_mib=64, optim_cpu_offload=True)
            if dist_mode == "fsdp2"
            else None
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        update_epochs=update_epochs,
    )


@pytest.fixture
def sft_factory():
    return generate_sft


class TestSFTInit:
    @pytest.mark.parametrize("dist_mode", [None, "dist"])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [
            TINY_LLM_FIXTURE_PATH,
        ],
    )
    @pytest.mark.gpu
    @pytest.mark.parametrize("data_batch_size", [4])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    @pytest.mark.parametrize("from_name", [True, False])
    def test_init_sft(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        pretrained_model_name_or_path,
        vocab_size,
        input_size,
        max_tokens,
        data_batch_size,
        micro_batch_size_per_gpu,
        from_name,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
            from_name=from_name,
        )
        assert sft.batch_size_per_process == 16
        assert sft.lr == 5e-5
        assert sft.max_grad_norm == 0.1
        assert sft.update_epochs == 1
        assert sft.temperature == 0
        assert sft.calc_position_embeddings
        assert sft.distributed == (dist_mode is not None)
        assert sft.device == resolve_device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        assert sft.index == 0
        assert sft.scores == []
        assert sft.fitness == []
        assert sft.steps == 0
        assert isinstance(sft.actor, torch.nn.Module)
        assert isinstance(sft.optimizer, OptimizerWrapper)
        assert sft.optimizer.optimizer_cls is torch.optim.AdamW
        assert isinstance(sft.optimizer.optimizer, torch.optim.AdamW)
        sft.clean_up()

    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_init_sft_model_name_none_actor_network_none(
        self,
        vocab_size,
        micro_batch_size_per_gpu,
    ):
        with pytest.raises(
            ValueError,
            match=r"At least one of model_name or actor_network must be provided\.",
        ):
            SFT(
                actor_network=None,
                model_name=None,
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            )


class TestSFTGetAction:
    @pytest.mark.parametrize("dist_mode", [None, "dist"])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [
            TINY_LLM_FIXTURE_PATH,
        ],
    )
    @pytest.mark.gpu
    @pytest.mark.parametrize("data_batch_size", [4])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_sft_get_action(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        pretrained_model_name_or_path,
        vocab_size,
        input_size,
        max_tokens,
        data_batch_size,
        micro_batch_size_per_gpu,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        with pytest.raises(NotImplementedError):
            sft.get_action(obs=None)
        sft.clean_up()


class TestSFTLearn:
    @pytest.mark.parametrize("dist_mode", ["dist"])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [
            TINY_LLM_FIXTURE_PATH,
        ],
    )
    @pytest.mark.gpu
    @pytest.mark.parametrize("data_batch_size", [32])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_sft_learn(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        pretrained_model_name_or_path,
        vocab_size,
        input_size,
        max_tokens,
        data_batch_size,
        micro_batch_size_per_gpu,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        env = make_sft_gym(
            num_samples=100,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
        )
        for name, param in sft.actor.named_parameters():
            if ("lora_A" in name or "lora_B" in name) and param is not None:
                param.data.normal_(mean=0, std=1.0)
        prompts = env.reset()
        pre_learn_actor_state_dict = copy.deepcopy(sft.actor.state_dict())
        metrics = sft.learn(prompts)
        loss = metrics["loss"]
        perplexity = metrics["perplexity"]
        assert isinstance(loss, float)
        assert isinstance(perplexity, float)
        assert perplexity >= 1.0  # perplexity is exp(loss), always >= 1

        # Check that the actor parameters are updated after learning
        any_updated = False
        for (param_name, param), (_, pre_learn_param) in zip(
            sft.actor.state_dict().items(),
            pre_learn_actor_state_dict.items(),
            strict=False,
        ):
            if "lora" in param_name and not torch.equal(param, pre_learn_param):
                any_updated = True
                break
        assert any_updated, (
            "Expected at least one LoRA parameter to be updated after learn()"
        )

        sft.clean_up()

    @pytest.mark.parametrize("dist_mode", ["dist"])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [
            TINY_LLM_FIXTURE_PATH,
        ],
    )
    @pytest.mark.gpu
    def test_sft_learn_loss_falls_on_fixed_batch(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
    ):
        """Repeated steps on one tiny batch must lower the loss (learning)."""
        torch.manual_seed(0)
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            None,
        )
        for group in sft.optimizer.optimizer.param_groups:
            group["lr"] = 1e-3
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        env = make_sft_gym(
            num_samples=4,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=4,
        )
        for name, param in sft.actor.named_parameters():
            if ("lora_A" in name or "lora_B" in name) and param is not None:
                param.data.normal_(mean=0, std=1.0)
        prompts = env.reset()
        losses = [sft.learn(prompts)["loss"] for _ in range(30)]
        print(f"\nSFT-TREND first={losses[0]} last={losses[-1]}")
        assert losses[-1] < losses[0], f"loss did not fall: {losses[0]} -> {losses[-1]}"
        sft.clean_up()


class TestSFTTest:
    @pytest.mark.parametrize("dist_mode", ["dist"])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [
            TINY_LLM_FIXTURE_PATH,
        ],
    )
    @pytest.mark.gpu
    @pytest.mark.parametrize("data_batch_size", [2])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    @pytest.mark.parametrize("loop", [1, 2])
    def test_sft_test(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        pretrained_model_name_or_path,
        vocab_size,
        input_size,
        max_tokens,
        data_batch_size,
        micro_batch_size_per_gpu,
        loop,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        env = make_sft_gym(
            num_samples=100,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
        )
        fitness = sft.test(env, loop=loop)
        assert isinstance(fitness, np.ndarray)
        assert fitness <= 0.0  # fitness is negative mean loss
        assert len(sft.fitness) == 1
        sft.clean_up()

    def test_sft_test_rejects_a_batch_from_the_wrong_objective(self):
        """An objective='preference' env fails at the boundary, not inside the loss."""
        import contextlib

        class DummyPreferenceEnv:
            def eval_mode(self):
                return contextlib.nullcontext()

            def reset(self):
                return {
                    "prompt": ["p"],
                    "prompt_lengths": [1],
                    "chosen": ["c"],
                    "rejected": ["r"],
                    "chosen_input_ids": torch.ones(1, 3, dtype=torch.long),
                    "chosen_attention_mask": torch.ones(1, 3, dtype=torch.long),
                    "rejected_input_ids": torch.ones(1, 3, dtype=torch.long),
                    "rejected_attention_mask": torch.ones(1, 3, dtype=torch.long),
                }

        sft = SFT(
            actor_network=create_module(
                input_size=10, max_tokens=20, vocab_size=100, device="cpu"
            ),
            pad_token_id=99,
            pad_token="<pad>",
            lora_config=LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
            device="cpu",
            micro_batch_size_per_gpu=1,
        )
        with pytest.raises(TypeError, match="needs an objective='sft'"):
            sft.test(DummyPreferenceEnv(), loop=1)


class TestSFTLoad:
    def test_sft_load(self):
        with pytest.raises(NotImplementedError):
            SFT.load("path")


class TestSFTCleanUp:
    @pytest.mark.parametrize("dist_mode", [None])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [TINY_LLM_FIXTURE_PATH],
    )
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_sft_clean_up(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        pretrained_model_name_or_path,
        vocab_size,
        input_size,
        max_tokens,
        micro_batch_size_per_gpu,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        sft.clean_up()
        assert sft.actor is None
        assert sft.optimizer is None
        assert sft.lr_scheduler is None


class TestSFTSaveLoadCheckpoint:
    @pytest.mark.parametrize("dist_mode", [None])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [TINY_LLM_FIXTURE_PATH],
    )
    @pytest.mark.gpu
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_sft_save_load_checkpoint(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sft.save_checkpoint(tmpdir)
            new_sft = SFT(
                actor_network=model_factory(pretrained_model_name_or_path),
                pad_token_id=vocab_size - 1,
                pad_token="<pad>",
                device="cuda" if torch.cuda.is_available() else "cpu",
                lora_config=copy.deepcopy(sft.lora_config),
            )
            own_profiler = new_sft.learn_profiler
            new_sft.load_checkpoint(tmpdir)

            for attr in EvolvableAlgorithm.inspect_attributes(sft):
                if attr.startswith("_"):
                    continue
                if attr == "rng":
                    assert hasattr(new_sft, attr)
                elif attr == "actor":
                    for (name, param), (new_name, new_param) in zip(
                        sft.actor.named_parameters(),
                        new_sft.actor.named_parameters(),
                        strict=False,
                    ):
                        assert torch.allclose(param, new_param), (
                            f"Parameter {name} is not equal (new_name: {new_name})"
                        )
                elif attr == "optimizer":
                    for param, new_param in zip(
                        sft.optimizer.parameters(),
                        new_sft.optimizer.parameters(),
                        strict=False,
                    ):
                        assert torch.equal(param, new_param)
                elif attr == "lr_scheduler":
                    assert (
                        getattr(new_sft, attr).__class__.__name__
                        == getattr(sft, attr).__class__.__name__
                    )
                elif attr == "shard_runtime":
                    assert type(getattr(new_sft, attr)) is type(getattr(sft, attr))
                    assert (
                        getattr(new_sft, attr).is_sharded
                        == getattr(sft, attr).is_sharded
                    )
                elif attr == "learn_profiler":
                    assert new_sft.learn_profiler is own_profiler
                elif attr == "lora_config":
                    assert getattr(new_sft, attr) is not None
                    assert getattr(sft, attr) is not None
                    old_targets = set(getattr(sft, attr).target_modules)
                    new_targets = set(getattr(new_sft, attr).target_modules)
                    assert old_targets == new_targets
                    assert getattr(new_sft, attr).r == getattr(sft, attr).r
                    assert (
                        getattr(new_sft, attr).lora_alpha
                        == getattr(sft, attr).lora_alpha
                    )
                    assert (
                        getattr(new_sft, attr).lora_dropout
                        == getattr(sft, attr).lora_dropout
                    )
                elif not isinstance(getattr(sft, attr), torch.Tensor):
                    assert getattr(new_sft, attr) == getattr(sft, attr), (
                        f"Attribute {attr} is not equal"
                    )
                else:
                    assert torch.equal(getattr(new_sft, attr), getattr(sft, attr))
        sft.clean_up()
        new_sft.clean_up()


class TestSFTRecompile:
    @pytest.mark.parametrize("dist_mode", [None])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [TINY_LLM_FIXTURE_PATH],
    )
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_sft_exception_on_recompile(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        pretrained_model_name_or_path,
        vocab_size,
        input_size,
        max_tokens,
        micro_batch_size_per_gpu,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        # LLM algorithms run with torch_compiler=None, so recompile is a no-op.
        sft.recompile()
        sft.clean_up()


class TestSFTNoLLMDependencies:
    def test_sft_no_llm_dependencies(
        self, sft_factory, model_factory, dist_mode_factory
    ):
        with (
            mock.patch("agilerl.algorithms.core.base.HAS_LLM_DEPENDENCIES", False),
            pytest.raises(
                ImportError,
                match=r"LLM dependencies are not installed. Please install them using \`pip install agilerl\[llm\]\`.",
            ),
        ):
            sft_factory(
                dist_mode_factory=dist_mode_factory,
                model_factory=model_factory,
                dist_mode=None,
                vocab_size=30,
                input_size=5,
                max_tokens=10,
                pretrained_model_name_or_path=None,
                micro_batch_size_per_gpu=None,
                from_name=False,
            )


class TestSFTGetLogprobs:
    @pytest.mark.parametrize("dist_mode", [None])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [TINY_LLM_FIXTURE_PATH],
    )
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    @pytest.mark.gpu  # real Qwen2 forward is Liger/Triton-fused → needs CUDA
    def test_sft_get_logprobs(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        batch_size,
        micro_batch_size_per_gpu,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
            sft.device,
        )
        log_probs = sft._get_logprobs(ids=ids, batch_size=1)
        assert log_probs.shape == (ids.shape[0], ids.shape[1] - 1)
        sft.clean_up()


class TestSFTBackwardPass:
    @pytest.mark.parametrize("dist_mode", [None])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [TINY_LLM_FIXTURE_PATH],
    )
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    @pytest.mark.gpu  # real Qwen2 forward is Liger/Triton-fused → needs CUDA
    def test_sft_backward_pass(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        batch_size,
        micro_batch_size_per_gpu,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
            sft.device,
        )
        loss = sft.actor.forward(ids).logits.mean()
        sft._backward_pass(loss)
        sft.clean_up()


class TestSFTPreprocessObservation:
    @pytest.mark.parametrize("dist_mode", [None])
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("input_size", [10])
    @pytest.mark.parametrize("max_tokens", [20])
    @pytest.mark.parametrize(
        "pretrained_model_name_or_path",
        [TINY_LLM_FIXTURE_PATH],
    )
    @pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
    def test_sft_preprocess_observation(
        self,
        sft_factory,
        dist_mode_factory,
        model_factory,
        dist_mode,
        pretrained_model_name_or_path,
        vocab_size,
        input_size,
        max_tokens,
        micro_batch_size_per_gpu,
    ):
        sft = sft_factory(
            dist_mode_factory,
            model_factory,
            dist_mode,
            vocab_size,
            input_size,
            max_tokens,
            pretrained_model_name_or_path,
            micro_batch_size_per_gpu,
        )
        obs = sft.preprocess_observation(
            orig_obs := torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )
        assert torch.equal(obs, orig_obs)
        sft.clean_up()


class TestSFTLearnPhaseTimings:
    def test_reports_and_logs_every_learn_phase(self) -> None:
        # Arrange
        sft = SFT(
            actor_network=create_module(
                input_size=10, max_tokens=20, vocab_size=100, device="cpu"
            ),
            pad_token_id=99,
            pad_token="<pad>",
            lora_config=LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.0,
            ),
            batch_size=4,
            micro_batch_size_per_gpu=2,
            wrap=False,
            device="cpu",
        )
        experiences = {
            "input_ids": torch.randint(
                0, 99, (4, 6), generator=torch.Generator().manual_seed(0)
            ),
            "attention_mask": torch.ones(4, 6, dtype=torch.long),
            "prompt_lengths": [2, 2, 2, 2],
        }

        # Act
        metrics = sft.learn(experiences)

        # Assert
        assert set(LEARN_PHASE_METRIC_NAMES) <= set(metrics)
        assert metrics["learn_phase_forward_s"] > 0.0
        assert sft.metrics.get_mean("learn_phase_forward_s") == pytest.approx(
            metrics["learn_phase_forward_s"]
        )
        assert sft.shard_runtime.phase_timer.marks is None
        sft.clean_up()


def make_cpu_sft() -> SFT:
    """Tiny fp32 SFT on CPU with one optimizer step of two micro-batches per learn."""
    return SFT(
        actor_network=create_module(
            input_size=10, max_tokens=20, vocab_size=100, device="cpu"
        ),
        pad_token_id=99,
        pad_token="<pad>",
        lora_config=LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
        batch_size=4,
        micro_batch_size_per_gpu=2,
        wrap=False,
        device="cpu",
    )


def random_sft_batch() -> dict[str, torch.Tensor | list[int]]:
    """Four six-token examples with two-token prompts."""
    return {
        "input_ids": torch.randint(
            0, 99, (4, 6), generator=torch.Generator().manual_seed(0)
        ),
        "attention_mask": torch.ones(4, 6, dtype=torch.long),
        "prompt_lengths": [2, 2, 2, 2],
    }


class TestSFTLearnMetrics:
    def test_metrics_are_means_of_the_micro_batch_losses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        sft = make_cpu_sft()
        sft_loss, outputs = record_outputs(sft._sft_loss)
        monkeypatch.setattr(sft, "_sft_loss", sft_loss)

        # Act
        metrics = sft.learn(random_sft_batch())

        # Assert
        losses = [loss.item() for loss in outputs]
        assert len(losses) == 2
        assert metrics["loss"] == pytest.approx(sum(losses) / 2, rel=1e-6)
        assert metrics["perplexity"] == pytest.approx(
            sum(np.exp(loss) for loss in losses) / 2, rel=1e-6
        )
        sft.clean_up()


class TestSFTLearnNonFiniteLoss:
    def test_finite_losses_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        sft = make_cpu_sft()
        monkeypatch.setattr(
            sft, "_sft_loss", scale_losses(sft._sft_loss, iter([1.0, 1.0]))
        )
        before = trainable_weights(sft.actor)

        # Act
        sft.learn(random_sft_batch())

        # Assert
        after = trainable_weights(sft.actor)
        assert any(
            not torch.equal(new, old) for new, old in zip(after, before, strict=True)
        )
        sft.clean_up()

    @pytest.mark.parametrize(
        "window_scales",
        [[float("nan"), 1.0], [1.0, float("nan")]],
        ids=["first_micro_batch", "last_micro_batch"],
    )
    def test_raises_before_the_step_when_a_micro_batch_loss_is_not_finite(
        self, monkeypatch: pytest.MonkeyPatch, window_scales: list[float]
    ) -> None:
        # Arrange: one optimizer step of two micro-batches per learn; a finite
        # learn first gives the optimizer state to keep.
        sft = make_cpu_sft()
        scales = iter([1.0, 1.0, *window_scales])
        monkeypatch.setattr(sft, "_sft_loss", scale_losses(sft._sft_loss, scales))
        sft.learn(random_sft_batch())
        before = trainable_weights(sft.actor)
        state_before = optimizer_state(sft.optimizer)

        # Act
        with pytest.raises(ValueError, match="Loss is not finite"):
            sft.learn(random_sft_batch())

        # Assert: it raises at the window's last micro-batch, before the step.
        assert list(scales) == []
        for new, old in zip(trainable_weights(sft.actor), before, strict=True):
            assert torch.equal(new, old)
        state_after = optimizer_state(sft.optimizer)
        assert state_before
        assert len(state_after) == len(state_before)
        for new, old in zip(state_after, state_before, strict=True):
            assert torch.equal(new, old)
        sft.clean_up()

    def test_raises_when_the_micro_batch_left_pending_a_step_is_not_finite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: three micro-batches per step over two micro-batches leave
        # the learn call's gradients pending the next one.
        sft = make_cpu_sft()
        sft.gradient_accumulation_steps = 3
        scales = iter([1.0, float("nan")])
        monkeypatch.setattr(sft, "_sft_loss", scale_losses(sft._sft_loss, scales))

        # Act / Assert
        with pytest.raises(ValueError, match="Loss is not finite"):
            sft.learn(random_sft_batch())
        assert list(scales) == []
        sft.clean_up()


class TestSFTUnsetFSDPMemory:
    def test_set_fields_skip_the_estimate(self) -> None:
        # Arrange
        agent = make_cpu_sft()
        agent.device = "cuda"
        config = FSDPConfig(
            checkpoint_every_n_blocks=1,
            reshard_after_forward=True,
            routed_expert_chunk_mib=64,
            optim_cpu_offload=True,
            activation_offload=False,
            cpu_offload=False,
        )

        # Act
        resolved = agent._resolve_fsdp_config(config)

        # Assert
        assert resolved == config
        agent.clean_up()

    def test_unset_fields_are_searched_from_max_model_len(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        agent = make_cpu_sft()
        save_checkpoint_config(
            agent, tmp_path, SUPPORTED_MODEL_INFO["Qwen/Qwen2.5-0.5B-Instruct"].config
        )
        patch_cuda_device(agent, monkeypatch, 80 * 2**30)

        # Act: a set memory field keeps the run FSDP.
        resolved = agent._resolve_fsdp_config(FSDPConfig(cpu_offload=False))

        # Assert
        assert all(getattr(resolved, name) is not None for name in FSDP_MEMORY_FIELDS)
        assert isinstance(agent.gradient_checkpointing, bool)
        agent.clean_up()

    def test_off_cuda_unset_fields_raise(self) -> None:
        # Arrange
        agent = make_cpu_sft()

        # Act / Assert
        with pytest.raises(ValueError, match=r"fsdp\.reshard_after_forward.*CUDA"):
            agent._resolve_fsdp_config(FSDPConfig())
        agent.clean_up()


# Packed and padded attention differ only in fp32 summation order.
PACKING_ATOL = 1e-5


def make_qwen3_sft() -> SFT:
    """Tiny fp32 Qwen3 SFT on CPU with non-zero LoRA deltas."""
    # Liger's instance patches are Triton kernels; this model runs on CPU.
    with mock.patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False):
        sft = SFT(
            actor_network=create_tiny_qwen3(),
            pad_token_id=63,
            pad_token="<pad>",
            lora_config=LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
            ),
            batch_size=4,
            wrap=False,
            gradient_checkpointing=False,
            device="cpu",
        )
    with torch.no_grad():
        for name, param in sft.actor.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.5)
    return sft


def packing_sft_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-padded rows of 12, 9, 6 and 11 tokens with their response labels."""
    ids, mask = right_padded_batch([12, 9, 6, 11])
    prompt_masks = SFT._create_prompt_masks([3, 2, 4, 2], max_length=12)
    labels = torch.where((prompt_masks.bool() & mask)[:, 1:], ids[:, 1:], -100)
    return ids, mask.long(), labels


def lora_grads(sft: SFT) -> list[torch.Tensor]:
    """Clone and clear the gradient of every trainable parameter."""
    grads = []
    for param in sft.actor.parameters():
        if param.requires_grad:
            grads.append(param.grad.clone())
            param.grad = None
    return grads


class TestSFTSFTLoss:
    def test_padded_loss_is_the_token_mean_cross_entropy_over_response_tokens(
        self,
    ) -> None:
        # Arrange
        sft = make_qwen3_sft()
        ids, mask, labels = packing_sft_batch()
        with torch.no_grad():
            logits = sft.actor(input_ids=ids, attention_mask=mask).logits
        expected = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # Act
        with torch.no_grad():
            loss = sft._sft_loss(ids, mask, labels, training=False)

        # Assert
        torch.testing.assert_close(loss, expected, atol=PACKING_ATOL, rtol=0)
        sft.clean_up()

    def test_packed_loss_and_grads_match_padded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        sft = make_qwen3_sft()
        ids, mask, labels = packing_sft_batch()
        forward_shapes = []
        sft.actor.register_forward_pre_hook(
            lambda _m, _a, kwargs: forward_shapes.append(kwargs["input_ids"].shape),
            with_kwargs=True,
        )
        padded_loss = sft._sft_loss(ids, mask, labels, training=True)
        padded_loss.backward()
        padded_grads = lora_grads(sft)
        monkeypatch.setattr(sft, "_packing_mode", lambda: "varlen")

        # Act
        packed_loss = sft._sft_loss(ids, mask, labels, training=True)
        packed_loss.backward()
        packed_grads = lora_grads(sft)

        # Assert: the packed forward is one row of the 38 real tokens.
        assert forward_shapes == [torch.Size([4, 12]), torch.Size([1, 38])]
        torch.testing.assert_close(packed_loss, padded_loss, atol=PACKING_ATOL, rtol=0)
        assert padded_grads
        for packed, padded in zip(packed_grads, padded_grads, strict=True):
            torch.testing.assert_close(packed, padded, atol=PACKING_ATOL, rtol=0)
        sft.clean_up()
