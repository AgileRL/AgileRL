# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The builder layer: how a contract spec becomes a live algorithm."""

import pytest

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import (
    LLMAlgorithm,
    MultiAgentAlgorithm,
    SingleAgentAlgorithm,
)
from agilerl.arena.models.algorithms import (
    DPOSpec,
    DQNSpec,
    GRPOSpec,
    LLMAlgorithmSpec,
    MADDPGSpec,
    MultiAgentAlgorithmSpec,
    PPOSpec,
    SFTSpec,
    SingleAgentAlgorithmSpec,
)
from agilerl.arena.models.networks import LoraConfigDict
from agilerl.arena.models.registry import MANIFEST_REGISTRY
from agilerl.builders import (
    LLMBuilder,
    MultiAgentBuilder,
    SingleAgentBuilder,
    select_builder,
)
from agilerl.builders import llm as llm_builder

requires_llm = pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed"
)


class TestBuilderFor:
    def test_dispatches_on_the_contract_paradigm(self):
        assert select_builder(DQNSpec()) is SingleAgentBuilder
        assert select_builder(MADDPGSpec()) is MultiAgentBuilder
        assert select_builder(DPOSpec.model_construct()) is LLMBuilder

    def test_rejects_a_non_spec(self):
        with pytest.raises(TypeError, match="not an algorithm spec"):
            select_builder(object())  # type: ignore[arg-type]

    def test_a_user_subclass_dispatches_like_its_parent(self):
        class MyPPOSpec(PPOSpec):
            pass

        assert select_builder(MyPPOSpec()) is SingleAgentBuilder


class TestAlgoClass:
    @pytest.mark.parametrize(
        ("spec_cls", "builder", "base"),
        [
            (DQNSpec, SingleAgentBuilder, SingleAgentAlgorithm),
            (MADDPGSpec, MultiAgentBuilder, MultiAgentAlgorithm),
            pytest.param(DPOSpec, LLMBuilder, LLMAlgorithm, marks=requires_llm),
            pytest.param(SFTSpec, LLMBuilder, LLMAlgorithm, marks=requires_llm),
        ],
    )
    def test_resolves_by_name_convention(self, spec_cls, builder, base):
        import agilerl.algorithms as algorithms

        resolved = builder.algo_class(spec_cls.model_construct())
        assert issubclass(resolved, base)
        assert resolved is getattr(algorithms, spec_cls.__name__.removesuffix("Spec"))

    def test_a_user_subclass_resolves_to_its_parent(self):
        from agilerl.algorithms import DQN

        class MyDQNSpec(DQNSpec):
            pass

        assert SingleAgentBuilder.algo_class(MyDQNSpec()) is DQN

    @requires_llm
    def test_paradigm_mismatch_is_caught(self):
        # An LLM spec handed to the RL builder is a programming error, not a
        # silently wrong build.
        with pytest.raises(TypeError):
            SingleAgentBuilder.algo_class(DPOSpec.model_construct())

    def test_every_registered_spec_resolves(self):
        for spec_cls in dict(MANIFEST_REGISTRY.items()).values():
            spec = spec_cls.model_construct()
            builder = select_builder(spec)
            if not HAS_LLM_DEPENDENCIES and isinstance(spec, LLMAlgorithmSpec):
                # The builder resolves, but algo_class import-gates on the
                # llm extra.
                continue
            assert builder.algo_class(spec) is not None


class TestBuildRequiresRuntimeArgs:
    def test_rl_requires_spaces_and_index(self):
        with pytest.raises(ValueError, match="observation_space"):
            SingleAgentBuilder.build(SingleAgentAlgorithmSpec(learn_step=1))

    def test_multi_agent_requires_spaces_and_index(self):
        with pytest.raises(ValueError, match="observation_spaces"):
            MultiAgentBuilder.build(MultiAgentAlgorithmSpec())

    def test_llm_requires_tokenizer(self):
        with pytest.raises(ValueError, match="requires a tokenizer"):
            LLMBuilder.build(LLMAlgorithmSpec.model_construct())


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
class TestPeftLoraConfig:
    def test_manifest_lora_becomes_the_peft_object(self):
        from peft import LoraConfig

        lora = llm_builder.peft_lora_config(
            LoraConfigDict(lora_r=8, lora_alpha=16, target_modules=["q_proj"])
        )
        assert isinstance(lora, LoraConfig)
        assert lora.r == 8
        assert lora.lora_alpha == 16
        assert set(lora.target_modules) == {"q_proj"}


def test_peft_lora_config_needs_the_llm_extras(monkeypatch):
    monkeypatch.setattr(llm_builder, "HAS_LLM_DEPENDENCIES", False)
    with pytest.raises(ImportError, match="LLM dependencies are required"):
        llm_builder.peft_lora_config(LoraConfigDict())


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
def test_arena_only_fields_are_not_forwarded_to_the_constructor():
    from agilerl.algorithms import GRPO
    from agilerl.builders.base import constructor_kwargs, spec_kwargs

    spec = GRPOSpec(
        group_size=4,
        zero_stage=2,
        deepspeed={"train_batch_size": 1},
        vllm_engine_args={"trust_remote_code": True},
        pretrained_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    )
    kwargs = spec_kwargs(spec, hp_config=None)
    filtered = constructor_kwargs(GRPO, kwargs)
    assert "zero_stage" not in filtered
    assert "deepspeed" not in filtered
    assert "vllm_engine_args" not in filtered
    assert "group_size" in filtered


class TestAlgoClassParadigmGuards:
    def test_a_paradigm_base_spec_resolves_no_algorithm(self):
        # The registry walks the MRO but stops at the paradigm bases, which
        # name no concrete algorithm.
        with pytest.raises(AttributeError, match="No algorithm class"):
            SingleAgentBuilder.algo_class(SingleAgentAlgorithmSpec())

    def test_unresolved_llm_spec_points_at_the_llm_extra(self, monkeypatch):
        from agilerl.builders import base as builders_base

        monkeypatch.setattr(builders_base, "HAS_LLM_DEPENDENCIES", False)

        class NoSuchLLMSpec(LLMAlgorithmSpec):
            pass

        with pytest.raises(AttributeError, match=r"pip install agilerl\[llm\]"):
            LLMBuilder.algo_class(NoSuchLLMSpec.model_construct())

    def test_llm_builder_rejects_an_rl_resolution(self):
        with pytest.raises(TypeError, match="not a subclass of LLMAlgorithm"):
            LLMBuilder.algo_class(DQNSpec())

    def test_multi_agent_builder_rejects_an_rl_resolution(self):
        with pytest.raises(TypeError, match="not a subclass of MultiAgentAlgorithm"):
            MultiAgentBuilder.algo_class(DQNSpec())


class TestLLMBuildGuards:
    def test_rejects_a_non_llm_spec(self):
        from unittest.mock import MagicMock

        with pytest.raises(TypeError, match="not an LLMAlgorithmSpec"):
            LLMBuilder.build(DQNSpec(), tokenizer=MagicMock())

    def test_a_vllm_config_model_is_dumped_for_the_constructor(self, monkeypatch):
        """A manifest ``vllm_config`` section reaches the constructor as VLLMConfig."""
        from unittest.mock import MagicMock

        from agilerl.arena.models.networks import VLLMConfig as VLLMConfigSpec
        from agilerl.utils.algo_utils import VLLMConfig

        built = {}

        class DummyAlgo:
            def __init__(self, **kwargs):
                built.update(kwargs)

        monkeypatch.setattr(
            LLMBuilder, "algo_class", classmethod(lambda cls, spec: DummyAlgo)
        )
        monkeypatch.setattr(
            llm_builder,
            "load_pad_token_configs",
            lambda name: (MagicMock(), MagicMock()),
        )
        monkeypatch.setattr(
            llm_builder, "resolve_pad_token_id", lambda tok, **kw: (0, "eos")
        )
        monkeypatch.setattr(llm_builder, "apply_pad_token_id", lambda tok, pad: None)

        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
            use_vllm=True,
            vllm_config=VLLMConfigSpec(tensor_parallel_size=2),
        )
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        LLMBuilder.build(spec, tokenizer=tokenizer)
        assert isinstance(built["vllm_config"], VLLMConfig)
        assert built["vllm_config"].tensor_parallel_size == 2
