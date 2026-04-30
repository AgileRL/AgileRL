import copy
import gc
import tempfile
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

pytest.importorskip("deepspeed", reason="LLM tests require deepspeed.")
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
from datasets import Dataset
from deepspeed.runtime.engine import DeepSpeedEngine
from peft import LoraConfig
from transformers import AutoTokenizer

from agilerl.algorithms.core.base import (
    EvolvableAlgorithm,
    LLMAlgorithm,
    OptimizerWrapper,
)
from agilerl.algorithms.dpo import DPO
from agilerl.llm_envs import PreferenceGym
from tests import TINY_LLM_FIXTURE_PATH
from tests.test_algorithms.test_llms.test_grpo import (
    create_module,
    deepspeed_config_stage_1,
    deepspeed_config_stage_2,
)


def make_preference_gym(
    num_samples: int,
    accelerator: Accelerator | None,
    tokenizer: AutoTokenizer,
    data_batch_size_per_gpu: int = 8,
):
    train_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(num_samples)],
            "chosen": [f"Chosen {i}" for i in range(num_samples)],
            "rejected": [f"Rejected {i}" for i in range(num_samples)],
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(num_samples)],
            "chosen": [f"Chosen {i}" for i in range(num_samples)],
            "rejected": [f"Rejected {i}" for i in range(num_samples)],
        }
    )
    return PreferenceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=data_batch_size_per_gpu,
        accelerator=accelerator,
    )


pytestmark = pytest.mark.llm


@pytest.fixture
def preference_dataset_factory():
    return make_preference_gym


def generate_dpo(
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    vocab_size,
    input_size,
    max_tokens,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    micro_batch_size_per_gpu,
    from_name=False,
    use_liger_loss=False,
):
    if config is not None and not torch.cuda.is_available():
        pytest.skip("DeepSpeed-configured LLM tests require CUDA support.")

    config = copy.deepcopy(config)
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
    dpo = DPO(
        actor_network=actor if not from_name else None,
        model_name=pretrained_model_name_or_path if from_name else None,
        pad_token_id=vocab_size - 1,
        pad_token="<pad>",
        lora_config=lora_config,
        accelerator=accelerator,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_separate_reference_adapter=use_separate_reference_adapter,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        use_liger_loss=use_liger_loss,
    )
    return dpo


@pytest.fixture(scope="function")
def dpo_factory():
    return generate_dpo


def _make_cpu_dpo_for_branch_tests(**kwargs):
    vocab_size = kwargs.pop("vocab_size", 100)
    input_size = kwargs.pop("input_size", 10)
    max_tokens = kwargs.pop("max_tokens", 20)
    defaults = {
        "actor_network": create_module(
            input_size=input_size,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
            device="cpu",
        ),
        "pad_token_id": vocab_size - 1,
        "pad_token": "<pad>",
        "lora_config": LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        ),
        "batch_size": 4,
        "micro_batch_size_per_gpu": 2,
        "accelerator": None,
        "wrap": False,
        "gradient_checkpointing": False,
        "device": "cpu",
    }
    defaults.update(kwargs)
    return DPO(**defaults)


@pytest.mark.parametrize(
    (
        "config",
        "use_deepspeed_optimizer",
        "pretrained_model_name_or_path",
        "from_name",
        "use_separate_reference_adapter",
    ),
    [
        pytest.param(None, False, None, False, False, id="actor-network"),
        pytest.param(None, False, None, False, True, id="actor-network-reference"),
        pytest.param(
            None,
            False,
            TINY_LLM_FIXTURE_PATH,
            True,
            False,
            id="model-name",
        ),
        pytest.param(
            deepspeed_config_stage_1,
            False,
            None,
            False,
            False,
            id="zero1-torch-optimizer",
        ),
        pytest.param(
            deepspeed_config_stage_2,
            False,
            None,
            False,
            False,
            id="zero2-torch-optimizer",
        ),
    ],
)
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_init_dpo(
    deepspeed_env,
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    micro_batch_size_per_gpu,
    from_name,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
        from_name=from_name,
    )
    assert dpo.batch_size_per_process == 16
    assert dpo.beta == 0.1
    assert dpo.lr == 0.000005
    assert dpo.max_grad_norm == 0.1
    assert dpo.update_epochs == 1
    assert dpo.temperature == 1
    assert dpo.calc_position_embeddings
    assert dpo.device == (
        dpo.accelerator.device
        if torch.cuda.is_available() and dpo.accelerator is not None
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert dpo.index == 0
    assert dpo.scores == []
    assert dpo.fitness == []
    assert dpo.steps == [0]
    if config is not None:
        assert isinstance(dpo.actor, DeepSpeedEngine)
        assert isinstance(dpo.optimizer, OptimizerWrapper)
        assert isinstance(dpo.optimizer.optimizer, DeepSpeedOptimizerWrapper)
    else:
        assert isinstance(dpo.actor, torch.nn.Module)
    dpo.clean_up()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_init_dpo_model_name_none_actor_network_none(
    use_separate_reference_adapter,
    vocab_size,
    micro_batch_size_per_gpu,
):
    with pytest.raises(
        ValueError,
        match="At least one of model_name or actor_network must be provided.",
    ):
        DPO(
            actor_network=None,
            model_name=None,
            pad_token_id=vocab_size - 1,
            pad_token="<pad>",
            accelerator=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_separate_reference_adapter=use_separate_reference_adapter,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        )

    AcceleratorState._reset_state(True)


def test_dpo_get_action():
    dpo = _make_cpu_dpo_for_branch_tests()
    with pytest.raises(NotImplementedError):
        dpo.get_action(obs=None)
    dpo.clean_up()


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        TINY_LLM_FIXTURE_PATH,
    ],
)
@pytest.mark.parametrize("data_batch_size", [2])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
@pytest.mark.parametrize("use_liger_loss", [False, True])
def test_dpo_learn(
    deepspeed_env,
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    micro_batch_size_per_gpu,
    use_liger_loss,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
        use_liger_loss=use_liger_loss,
    )

    num_samples = 4
    train_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(num_samples)],
            "chosen": [
                f"This prompt is better than the rejected prompt {i}"
                for i in range(num_samples)
            ],
            "rejected": [f"REALLY BAD RESPONSE {i}" for i in range(num_samples)],
        },
    )
    test_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(num_samples)],
            "chosen": [
                f"This prompt is better than the rejected prompt {i}"
                for i in range(num_samples)
            ],
            "rejected": [f"REALLY BAD RESPONSE {i}" for i in range(num_samples)],
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    env = PreferenceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=data_batch_size,
        accelerator=dpo.accelerator,
    )
    for name, param in dpo.actor.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and param is not None:
            param.data.normal_(mean=0, std=1.0)

    prompts = env.reset()
    pre_learn_actor_state_dict = copy.deepcopy(dpo.actor.state_dict())
    learn_result = dpo.learn(prompts)
    loss = learn_result["mean_loss"]
    chosen_reward = learn_result["mean_chosen_reward"]
    rejected_reward = learn_result["mean_rejected_reward"]

    assert isinstance(loss, float)
    assert isinstance(chosen_reward, float)
    assert isinstance(rejected_reward, float)

    # Check that the actor network is updated
    for (param_name, param), (_, pre_learn_param) in zip(
        dpo.actor.state_dict().items(),
        pre_learn_actor_state_dict.items(),
        strict=False,
    ):
        if "actor" in param_name:
            assert not torch.equal(param, pre_learn_param)

        elif "reference" in param_name:
            assert torch.equal(param, pre_learn_param)

        else:
            assert torch.equal(param, pre_learn_param)
    dpo.clean_up()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [
        (deepspeed_config_stage_2, True),
        (deepspeed_config_stage_2, False),
    ],
)
@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        TINY_LLM_FIXTURE_PATH,
    ],
)
@pytest.mark.parametrize("data_batch_size", [2])
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_dpo_test(
    deepspeed_env,
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    data_batch_size,
    micro_batch_size_per_gpu,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    )
    num_samples = 4
    train_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(num_samples)],
            "chosen": [
                f"This prompt is better than the rejected prompt {i}"
                for i in range(num_samples)
            ],
            "rejected": [f"Bad response {i}" for i in range(num_samples)],
        },
    )
    test_dataset = Dataset.from_dict(
        {
            "prompt": [f"Prompt {i}" for i in range(num_samples)],
            "chosen": [
                f"This prompt is better than the rejected prompt {i}"
                for i in range(num_samples)
            ],
            "rejected": [f"Bad response {i}" for i in range(num_samples)],
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    env = PreferenceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=data_batch_size,
        accelerator=dpo.accelerator,
    )
    fitness = dpo.test(env)
    assert isinstance(fitness, np.ndarray)
    dpo.clean_up()
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize("assertion_mode", ["warns_and_fallback", "private_guard"])
def test_dpo_liger_unavailable_behaviour(
    monkeypatch,
    dpo_factory,
    accelerator_factory,
    model_factory,
    assertion_mode,
):
    # LLMAlgorithm reads HAS_LIGER_KERNEL from core.base; DPO's module copy is
    # separate. Patch both so fallback warning runs on machines where liger is
    # installed and _dpo_loss_liger still raises when called directly.
    monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False)
    monkeypatch.setattr("agilerl.algorithms.dpo.HAS_LIGER_KERNEL", False)
    if assertion_mode == "warns_and_fallback":
        with pytest.warns(
            UserWarning,
            match=r"use_liger_loss=True requested.*Falling back to standard loss\.",
        ):
            dpo = dpo_factory(
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
                use_liger_loss=True,
            )
        assert dpo.use_liger_loss is False
    else:
        dpo = dpo_factory(
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
            use_liger_loss=False,
        )
        with pytest.raises(
            ImportError,
            match=r"Liger DPO loss was requested but `liger-kernel` is not available\. Set use_liger_loss=False\.",
        ):
            dpo._dpo_loss_liger(
                chosen_ids=torch.ones((1, 2), dtype=torch.long),
                rejected_ids=torch.ones((1, 2), dtype=torch.long),
                chosen_attn=torch.ones((1, 2), dtype=torch.long),
                rejected_attn=torch.ones((1, 2), dtype=torch.long),
                chosen_mask=torch.ones((1, 1), dtype=torch.long),
                rejected_mask=torch.ones((1, 1), dtype=torch.long),
            )

    dpo.clean_up()
    AcceleratorState._reset_state(True)


def test_dpo_load():
    with pytest.raises(NotImplementedError):
        DPO.load("path")


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [(None, False)],
)
@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [TINY_LLM_FIXTURE_PATH],
)
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_dpo_clean_up(
    deepspeed_env,
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    micro_batch_size_per_gpu,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    )
    dpo.clean_up()
    assert dpo.actor is None
    assert dpo.optimizer is None
    assert dpo.lr_scheduler is None


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [(None, False)],
)
@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [TINY_LLM_FIXTURE_PATH],
)
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
@pytest.mark.parametrize("lora_only", [False, True])
def test_dpo_save_load_checkpoint(
    deepspeed_env,
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    vocab_size,
    input_size,
    max_tokens,
    pretrained_model_name_or_path,
    micro_batch_size_per_gpu,
    lora_only,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    )
    accelerator = accelerator_factory(use_deepspeed_optimizer, config)
    with tempfile.TemporaryDirectory() as tmpdir:
        dpo.save_checkpoint(tmpdir, lora_only=lora_only)
        new_dpo = DPO(
            actor_network=model_factory(pretrained_model_name_or_path),
            pad_token_id=vocab_size - 1,
            pad_token="<pad>",
            device="cuda" if torch.cuda.is_available() else "cpu",
            accelerator=accelerator,
            use_separate_reference_adapter=use_separate_reference_adapter,
            lora_config=copy.deepcopy(dpo.lora_config),
        )
        new_dpo.load_checkpoint(tmpdir)

        for attr in EvolvableAlgorithm.inspect_attributes(dpo):
            if attr.startswith("_"):
                continue
            if attr == "rng":
                assert hasattr(new_dpo, attr)
            elif attr == "actor":
                for (name, param), (new_name, new_param) in zip(
                    dpo.actor.named_parameters(),
                    new_dpo.actor.named_parameters(),
                    strict=False,
                ):
                    assert torch.allclose(param, new_param), (
                        f"Parameter {name} is not equal (new_name: {new_name})"
                    )
            elif attr == "optimizer":
                for param, new_param in zip(
                    dpo.optimizer.parameters(),
                    new_dpo.optimizer.parameters(),
                    strict=False,
                ):
                    assert torch.equal(param, new_param)
            elif attr == "lora_config":
                assert getattr(new_dpo, attr) is not None
                assert getattr(dpo, attr) is not None
                old_targets = set(getattr(dpo, attr).target_modules)
                new_targets = set(getattr(new_dpo, attr).target_modules)
                assert old_targets == new_targets
                assert getattr(new_dpo, attr).r == getattr(dpo, attr).r
                assert (
                    getattr(new_dpo, attr).lora_alpha == getattr(dpo, attr).lora_alpha
                )
                assert (
                    getattr(new_dpo, attr).lora_dropout
                    == getattr(dpo, attr).lora_dropout
                )
            elif attr in ("accelerator", "lr_scheduler"):
                assert (
                    getattr(new_dpo, attr).__class__.__name__
                    == getattr(dpo, attr).__class__.__name__
                )
            elif not isinstance(getattr(dpo, attr), torch.Tensor):
                assert getattr(new_dpo, attr) == getattr(dpo, attr), (
                    f"Attribute {attr} is not equal"
                )
            else:
                assert torch.equal(getattr(new_dpo, attr), getattr(dpo, attr))
    dpo.clean_up()
    new_dpo.clean_up()


@pytest.mark.parametrize(
    "config, use_deepspeed_optimizer",
    [(None, False)],
)
@pytest.mark.parametrize("use_separate_reference_adapter", [False])
@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("input_size", [10])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [TINY_LLM_FIXTURE_PATH],
)
@pytest.mark.parametrize("micro_batch_size_per_gpu", [None])
def test_dpo_exception_on_recompile(
    deepspeed_env,
    dpo_factory,
    accelerator_factory,
    model_factory,
    config,
    use_deepspeed_optimizer,
    use_separate_reference_adapter,
    pretrained_model_name_or_path,
    vocab_size,
    input_size,
    max_tokens,
    micro_batch_size_per_gpu,
):
    dpo = dpo_factory(
        accelerator_factory,
        model_factory,
        config,
        use_deepspeed_optimizer,
        vocab_size,
        input_size,
        max_tokens,
        use_separate_reference_adapter,
        pretrained_model_name_or_path,
        micro_batch_size_per_gpu,
    )
    dpo.recompile()
    dpo.clean_up()


def test_dpo_no_llm_dependencies(dpo_factory, model_factory, accelerator_factory):
    with (
        mock.patch("agilerl.algorithms.core.base.HAS_LLM_DEPENDENCIES", False),
        pytest.raises(
            ImportError,
            match=r"LLM dependencies are not installed. Please install them using \`pip install agilerl\[llm\]\`.",
        ),
    ):
        dpo_factory(
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
        )
    AcceleratorState._reset_state(True)


@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
@pytest.mark.parametrize("batch_size", [1])
def test_dpo_get_logprobs(
    use_separate_reference_adapter,
    batch_size,
):
    vocab_size = 100
    input_size = 10
    max_tokens = 20
    dpo = _make_cpu_dpo_for_branch_tests(
        use_separate_reference_adapter=use_separate_reference_adapter,
    )
    ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
        dpo.device,
    )
    log_probs = dpo._get_logprobs(ids=ids, batch_size=1)
    assert log_probs.shape == (ids.shape[0], ids.shape[1] - 1)
    dpo.clean_up()


@pytest.mark.parametrize("batch_size", [1])
def test_dpo_backward_pass(batch_size):
    vocab_size = 100
    input_size = 10
    max_tokens = 20
    dpo = _make_cpu_dpo_for_branch_tests()
    ids = torch.randint(0, vocab_size, (batch_size, input_size + max_tokens)).to(
        dpo.device,
    )
    loss = dpo.actor.forward(ids).logits.mean()
    dpo._backward_pass(loss)
    dpo.clean_up()


def test_dpo_preprocess_observation():
    dpo = _make_cpu_dpo_for_branch_tests()
    obs = dpo.preprocess_observation(
        orig_obs := torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )
    assert torch.equal(obs, orig_obs)
    dpo.clean_up()


@pytest.mark.parametrize("use_separate_reference_adapter", [False, True])
def test_dpo_set_reference_policy(
    use_separate_reference_adapter,
):
    input_size = 10
    max_tokens = 20
    dpo = _make_cpu_dpo_for_branch_tests(
        use_separate_reference_adapter=use_separate_reference_adapter,
    )
    reference_update_tracker = 0
    dpo.set_reference_policy(reference_update_tracker)
    input_ids = torch.tensor([[i + 1 for i in range(input_size + max_tokens)]]).to(
        dpo.device,
    )
    attention_mask = torch.ones_like(input_ids).to(dpo.device)
    output_before = dpo.actor(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits
    assert dpo.reference_update_tracker == reference_update_tracker
    reference_update_tracker += 1
    dpo.set_reference_policy(reference_update_tracker)
    output_after = dpo.actor(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits
    assert torch.allclose(output_before, output_after)
    assert dpo.reference_update_tracker == reference_update_tracker
    dpo.clean_up()
