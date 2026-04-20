import copy
import gc
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch

pytest.importorskip("deepspeed", reason="LLM tests require deepspeed.")
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
from datasets import Dataset
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from peft import LoraConfig
from transformers import AutoTokenizer

from agilerl.algorithms.core.base import EvolvableAlgorithm, OptimizerWrapper
from agilerl.algorithms.sft import SFT
from agilerl.wrappers.llm_envs import SFTGym
from tests.test_algorithms.test_llms.test_grpo import (
    _patch_mps_learn_hooks,
    create_module,
    deepspeed_config_stage_1,
    deepspeed_config_stage_2,
)


def make_sft_gym(
    num_samples: int,
    accelerator: Accelerator | None,
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
    return SFTGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=data_batch_size_per_gpu,
        response_column=response_column,
        accelerator=accelerator,
    )


pytestmark = pytest.mark.llm


@pytest.fixture
def sft_dataset_factory():
    return make_sft_gym


def generate_sft(
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    vocab_size,
    input_size,
    max_tokens,
    pretrained_model_name_or_path,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
    from_name=False,
    use_liger_loss=False,
    update_epochs=1,
):
    if config is not None and not torch.cuda.is_available():
        pytest.skip("DeepSpeed-configured LLM tests require CUDA support.")

    gc.collect()
    torch.cuda.empty_cache()
    AcceleratorState._reset_state(True)

    accelerator = accelerator_factory(use_deepspeed_optimizer, config)
    if not use_deepspeed_optimizer and accelerator is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config.pop("optimizer", None)
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
    sft = SFT(
        actor_network=actor if not from_name else None,
        model_name=pretrained_model_name_or_path if from_name else None,
        pad_token_id=vocab_size - 1,
        pad_token="<pad>",
        lora_config=lora_config,
        accelerator=accelerator,
        device="cuda" if torch.cuda.is_available() else "cpu",
        reduce_memory_peak=reduce_memory_peak,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        use_liger_loss=use_liger_loss,
        update_epochs=update_epochs,
    )
    return sft


@pytest.fixture(scope="function")
def sft_factory():
    return generate_sft


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (None, False),
        (deepspeed_config_stage_1, True),
        (deepspeed_config_stage_1, False),
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    ],
)
@pytest.mark.parametrize("data_batch_size", [4])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
@pytest.mark.parametrize("from_name", [True, False])
def test_init_sft(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
    from_name,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
        from_name=from_name,
    )
    assert sft.batch_size_per_process == 16 if not reduce_memory_peak else 1
    assert sft.lr == 5e-5
    assert sft.max_grad_norm == 0.1
    assert sft.update_epochs == 1
    assert sft.temperature == 0
    assert sft.calc_position_embeddings
    assert sft.device == (
        sft.accelerator.device
        if torch.cuda.is_available() and sft.accelerator is not None
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert sft.index == 0
    assert sft.scores == []
    assert sft.fitness == []
    assert sft.steps == [0]
    if config is not None:
        assert isinstance(sft.actor, DeepSpeedEngine)
        if not use_deepspeed_optimizer:
            assert isinstance(sft.optimizer, OptimizerWrapper)
            assert isinstance(sft.optimizer.optimizer, DeepSpeedOptimizerWrapper)
        else:
            assert isinstance(sft.optimizer, OptimizerWrapper)
            assert isinstance(sft.optimizer.optimizer, DeepSpeedZeroOptimizer)
            assert isinstance(sft.actor.optimizer, DeepSpeedZeroOptimizer)
    else:
        assert isinstance(sft.actor, torch.nn.Module)
    sft.clean_up()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_init_sft_model_name_none_actor_network_none(
    vocab_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    with pytest.raises(
        ValueError,
        match="At least one of model_name or actor_network must be provided.",
    ):
        SFT(
            actor_network=None,
            model_name=None,
            pad_token_id=vocab_size - 1,
            pad_token="<pad>",
            accelerator=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
            reduce_memory_peak=reduce_memory_peak,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        )

    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (None, False),
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    ],
)
@pytest.mark.parametrize("data_batch_size", [4])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_sft_get_action(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    with pytest.raises(NotImplementedError):
        sft.get_action(obs=None)
    sft.clean_up()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    ],
)
@pytest.mark.parametrize("data_batch_size", [32])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
@pytest.mark.parametrize("use_liger_loss", [False, True])
def test_sft_learn(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
    use_liger_loss,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
        use_liger_loss=use_liger_loss,
    )

    train_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(100)],
            "response": [f"This is a good response for prompt {i}" for i in range(100)],
        },
    )
    test_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(100)],
            "response": [f"This is a good response for prompt {i}" for i in range(100)],
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    env = SFTGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=data_batch_size,
        accelerator=sft.accelerator,
    )
    for name, param in sft.actor.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and param is not None:
            param.data.normal_(mean=0, std=1.0)
    prompts = env.reset()
    pre_learn_actor_state_dict = copy.deepcopy(sft.actor.state_dict())
    metrics = sft.learn(prompts)
    loss = metrics["mean_loss"]
    perplexity = metrics["mean_perplexity"]
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
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    ],
)
@pytest.mark.parametrize("data_batch_size", [2])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
@pytest.mark.parametrize("loop", [1, 2])
def test_sft_test(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
    loop,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    train_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(100)],
            "response": [f"This is a good response for prompt {i}" for i in range(100)],
        },
    )
    test_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(100)],
            "response": [f"This is a good response for prompt {i}" for i in range(100)],
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    env = SFTGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=data_batch_size,
        accelerator=sft.accelerator,
    )
    fitness = sft.test(env, loop=loop)
    assert isinstance(fitness, np.ndarray)
    assert fitness <= 0.0  # fitness is negative mean loss
    assert len(sft.fitness) == 1
    sft.clean_up()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize("assertion_mode", ["warns_and_fallback", "private_guard"])
def test_sft_liger_unavailable_behaviour(
    monkeypatch,
    sft_factory,
    accelerator_factory,
    model_factory,
    assertion_mode,
):
    monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False)
    monkeypatch.setattr("agilerl.algorithms.sft.HAS_LIGER_KERNEL", False)
    if assertion_mode == "warns_and_fallback":
        with pytest.warns(
            UserWarning,
            match=r"use_liger_loss=True requested.*Falling back to standard loss\.",
        ):
            sft = sft_factory(
                accelerator_factory=accelerator_factory,
                model_factory=model_factory,
                config=None,
                use_deepspeed_optimizer=False,
                vocab_size=30,
                input_size=5,
                max_tokens=10,
                pretrained_model_name_or_path=None,
                reduce_memory_peak=False,
                micro_batch_size_per_gpu=None,
                from_name=False,
                use_liger_loss=True,
            )
        assert sft.use_liger_loss is False
    else:
        # When liger is unavailable and use_liger_loss=False, training should
        # proceed normally using the standard PyTorch cross-entropy loss path.
        sft = sft_factory(
            accelerator_factory=accelerator_factory,
            model_factory=model_factory,
            config=None,
            use_deepspeed_optimizer=False,
            vocab_size=30,
            input_size=5,
            max_tokens=10,
            pretrained_model_name_or_path=None,
            reduce_memory_peak=False,
            micro_batch_size_per_gpu=None,
            from_name=False,
            use_liger_loss=False,
        )
        assert sft.use_liger_loss is False

    sft.clean_up()
    AcceleratorState._reset_state(True)


def test_sft_load():
    with pytest.raises(NotImplementedError):
        SFT.load("path")


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [(None, False)],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    ["trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"],
)
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_sft_clean_up(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    sft.clean_up()
    assert sft.actor is None
    assert sft.optimizer is None
    assert sft.lr_scheduler is None


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [(None, False)],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    ["trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"],
)
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_sft_save_load_checkpoint(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    vocab_size,
    input_size,
    max_tokens,
    pretrained_model_name_or_path,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    accelerator = accelerator_factory(use_deepspeed_optimizer, config)
    with tempfile.TemporaryDirectory() as tmpdir:
        sft.save_checkpoint(tmpdir)
        new_sft = SFT(
            actor_network=model_factory(pretrained_model_name_or_path),
            pad_token_id=vocab_size - 1,
            pad_token="<pad>",
            device="cuda" if torch.cuda.is_available() else "cpu",
            accelerator=accelerator,
        )
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
            elif attr in ("accelerator", "lr_scheduler"):
                assert (
                    getattr(new_sft, attr).__class__.__name__
                    == getattr(sft, attr).__class__.__name__
                )
            elif not isinstance(getattr(sft, attr), torch.Tensor):
                assert getattr(new_sft, attr) == getattr(sft, attr), (
                    f"Attribute {attr} is not equal"
                )
            else:
                assert torch.equal(getattr(new_sft, attr), getattr(sft, attr))
    sft.clean_up()
    new_sft.clean_up()


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [(None, False)],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    ["trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"],
)
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_sft_exception_on_recompile(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    # LLMAlgorithm.recompile() is a guarded no-op unless torch_compiler is enabled.
    sft.recompile()
    sft.clean_up()


def test_sft_no_llm_dependencies(sft_factory, model_factory, accelerator_factory):
    with (
        mock.patch("agilerl.algorithms.core.base.HAS_LLM_DEPENDENCIES", False),
        pytest.raises(
            ImportError,
            match=r"LLM dependencies are not installed. Please install them using \`pip install agilerl\[llm\]\`.",
        ),
    ):
        sft_factory(
            accelerator_factory=accelerator_factory,
            model_factory=model_factory,
            config=None,
            use_deepspeed_optimizer=False,
            vocab_size=30,
            input_size=5,
            max_tokens=10,
            pretrained_model_name_or_path=None,
            reduce_memory_peak=False,
            micro_batch_size_per_gpu=None,
            from_name=False,
        )
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [(None, False)],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    ["trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"],
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_sft_get_logprobs(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    vocab_size,
    input_size,
    max_tokens,
    pretrained_model_name_or_path,
    batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
        sft.device,
    )
    log_probs = sft._get_logprobs(ids=ids, batch_size=1)
    assert log_probs.shape == (ids.shape[0], ids.shape[1] - 1)
    sft.clean_up()


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [(None, False)],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    ["trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"],
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_sft_backward_pass(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    vocab_size,
    input_size,
    max_tokens,
    pretrained_model_name_or_path,
    batch_size,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
        sft.device,
    )
    loss = sft.actor.forward(ids).logits.mean()
    sft._backward_pass(loss)
    sft.clean_up()


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [(None, False)],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    ["trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"],
)
@pytest.mark.parametrize("reduce_memory_peak", [True])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_sft_preprocess_observation(
    deepspeed_env,
    sft_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    reduce_memory_peak,
    micro_batch_size_per_gpu,
):
    sft = sft_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        pretrained_model_name_or_path,
        reduce_memory_peak,
        micro_batch_size_per_gpu,
    )
    obs = sft.preprocess_observation(
        orig_obs := torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )
    assert torch.equal(obs, orig_obs)
    sft.clean_up()


def test_sft_learn_calls_mps_empty_cache(
    monkeypatch: pytest.MonkeyPatch,
    accelerator_factory,
    model_factory,
) -> None:
    """Patch MPS on CI so ``torch.mps.empty_cache()`` in ``learn()`` is exercised."""
    empty = _patch_mps_learn_hooks(monkeypatch, "agilerl.algorithms.sft")
    sft = generate_sft(
        accelerator_factory,
        model_factory,
        config=None,
        use_deepspeed_optimizer=False,
        vocab_size=30,
        input_size=5,
        max_tokens=10,
        pretrained_model_name_or_path=None,
        reduce_memory_peak=False,
        micro_batch_size_per_gpu=None,
        from_name=False,
    )
    seq_len = 5 + 10
    prompt_len = 4
    experiences = {
        "input_ids": torch.randint(0, 30, (2, seq_len)),
        "attention_mask": torch.ones(2, seq_len, dtype=torch.long),
        "prompt_lengths": [prompt_len, prompt_len],
    }
    sft.learn(experiences, training=True)
    empty.assert_called()
    sft.clean_up()
    AcceleratorState._reset_state(True)
