# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The builder layer: how a contract spec becomes a live algorithm."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import (
    LLMAlgorithm,
    MultiAgentAlgorithm,
    SingleAgentAlgorithm,
)
from agilerl.arena.memory.manifest import training_settings_from_spec
from agilerl.arena.memory.resolve.batch import resolve_micro_batch_size_per_gpu
from agilerl.arena.memory.specs import DeviceSpec
from agilerl.arena.models.algorithms import (
    DPOSpec,
    DQNSpec,
    GRPOSpec,
    IPPOSpec,
    LLMAlgorithmSpec,
    MADDPGSpec,
    MultiAgentAlgorithmSpec,
    PPOSpec,
    SFTSpec,
    SingleAgentAlgorithmSpec,
)
from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.arena.models.networks import (
    CosineLRScheduleConfig as CosineLRScheduleSpec,
)
from agilerl.arena.models.networks import (
    LoraConfigDict,
    MlpSpec,
    QNetworkSpec,
    StochasticActorSpec,
)
from agilerl.arena.models.networks import VLLMConfig as VLLMConfigSpec
from agilerl.arena.models.registry import MANIFEST_REGISTRY
from agilerl.builders import (
    AlgorithmBuildRuntime,
    LLMBuilder,
    MultiAgentBuilder,
    SingleAgentBuilder,
    select_builder,
)
from agilerl.builders import llm as llm_builder
from agilerl.utils.algo_utils import CosineLRScheduleConfig

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


class TestAlgorithmBuildRuntime:
    def test_rl_forwards_slot_and_hp_config(self, monkeypatch):
        from agilerl.algorithms.core.registry import HyperparameterConfig

        built = {}

        class DummyAlgo:
            def __init__(self, **kwargs):
                built.update(kwargs)

        monkeypatch.setattr(
            SingleAgentBuilder,
            "algo_class",
            classmethod(lambda cls, spec: DummyAlgo),
        )
        hp_config = HyperparameterConfig()
        obs = MagicMock()
        action = MagicMock()

        SingleAgentBuilder.build(
            DQNSpec(learn_step=1),
            obs,
            action,
            runtime=AlgorithmBuildRuntime(
                index=3,
                device="cuda:0",
                hp_config=hp_config,
            ),
        )

        assert built["index"] == 3
        assert built["device"] == "cuda:0"
        assert built["hp_config"] is hp_config
        assert built["observation_space"] is obs
        assert built["action_space"] is action


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

    def test_tower_targets_stay_out_of_the_peft_object(self):
        lora = llm_builder.peft_lora_config(
            LoraConfigDict(target_modules=["q_proj"], tower_target_modules=["query"])
        )

        assert set(lora.target_modules) == {"q_proj"}
        assert not hasattr(lora, "tower_target_modules")

    def test_already_built_peft_config_is_returned(self):
        from peft import LoraConfig

        peft_lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"])

        assert llm_builder.peft_lora_config(peft_lora) is peft_lora


class TestCosineLrScheduleConfig:
    def test_manifest_schedule_becomes_the_framework_config(self):
        spec = CosineLRScheduleSpec(
            num_steps=200, warmup_proportion=0.05, min_lr_ratio=0.2
        )

        config = llm_builder.cosine_lr_schedule_config(spec)

        assert config == CosineLRScheduleConfig(
            num_steps=200, warmup_proportion=0.05, min_lr_ratio=0.2
        )


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
        quantization="nf4",
        vllm_engine_args={"trust_remote_code": True},
        pretrained_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    )

    kwargs = spec_kwargs(spec, hp_config=None)
    filtered = constructor_kwargs(GRPO, kwargs)

    assert kwargs["quantization"] == "nf4"
    assert kwargs["vllm_engine_args"] == {"trust_remote_code": True}
    assert "quantization" not in filtered
    assert "vllm_engine_args" not in filtered
    assert filtered["group_size"] == 4


def test_hp_config_is_forwarded_when_given():
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.builders.base import spec_kwargs

    hp_config = HyperparameterConfig(lr=RLParameter(min=1e-5, max=1e-3))
    kwargs = spec_kwargs(DQNSpec(learn_step=1), hp_config=hp_config)
    assert kwargs["hp_config"] is hp_config


class TestSpecKwargsNetConfig:
    def test_dumps_nested_net_config_to_plain_dicts(self):
        from agilerl.builders.base import spec_kwargs

        spec = DQNSpec(
            learn_step=1,
            net_config=QNetworkSpec(
                encoder_config=MlpSpec(hidden_size=[32]),
                head_config=MlpSpec(hidden_size=[32]),
            ),
        )

        kwargs = spec_kwargs(spec, hp_config=None)

        net_config = kwargs["net_config"]
        assert isinstance(net_config, dict)
        encoder = net_config["encoder_config"]
        assert isinstance(encoder, dict)
        assert encoder["hidden_size"] == [32]
        head = net_config["head_config"]
        assert isinstance(head, dict)
        assert head["hidden_size"] == [32]

    def test_dumps_per_agent_net_config_to_plain_dicts(self):
        from agilerl.builders.base import spec_kwargs

        spec = IPPOSpec(
            net_config={
                "agent_0": StochasticActorSpec(
                    encoder_config=MlpSpec(hidden_size=[16]),
                    head_config=MlpSpec(hidden_size=[16]),
                ),
            },
        )

        kwargs = spec_kwargs(spec, hp_config=None)

        net_config = kwargs["net_config"]
        assert isinstance(net_config, dict)
        agent = net_config["agent_0"]
        assert isinstance(agent, dict)
        assert agent["encoder_config"]["hidden_size"] == [16]


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


CATALOG_HUB_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CUDA_CARD_BYTES = 4 * 1024**3


def _patch_cuda_card(monkeypatch, total_bytes: int = CUDA_CARD_BYTES) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            total_memory=total_bytes,
            name="NVIDIA L4",
            major=8,
            minor=9,
        ),
    )


def _cuda_device_spec(
    total_bytes: int = CUDA_CARD_BYTES,
    available_bytes: int | None = None,
) -> DeviceSpec:
    return DeviceSpec(
        total_bytes=total_bytes,
        name="NVIDIA L4",
        compute_capability=(8, 9),
        available_bytes=available_bytes,
    )


def _build_llm_with_dummy_algo(
    spec,
    monkeypatch,
    *,
    rollout_mode: str = "",
    device: str = "cpu",
    environment=None,
):
    built = {}

    class DummyAlgo:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(
        LLMBuilder, "algo_class", classmethod(lambda cls, spec: DummyAlgo)
    )
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *args, **kwargs: MagicMock(model_type="llama"),
    )
    monkeypatch.setattr("importlib.util.find_spec", lambda name, package=None: None)
    monkeypatch.setattr(
        llm_builder,
        "load_pad_token_configs",
        lambda name: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        llm_builder, "resolve_pad_token_id", lambda tok, **kw: (0, "eos")
    )
    monkeypatch.setattr(llm_builder, "apply_pad_token_id", lambda tok, pad: None)

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    LLMBuilder.build(
        spec,
        tokenizer=tokenizer,
        runtime=AlgorithmBuildRuntime(device=device),
        rollout_mode=rollout_mode,
        environment=environment,
    )
    return built


class TestLLMBuildGuards:
    def test_rejects_a_non_llm_spec(self):
        with pytest.raises(TypeError, match="not an LLMAlgorithmSpec"):
            LLMBuilder.build(DQNSpec(), tokenizer=MagicMock())

    def test_a_vllm_config_model_is_dumped_for_the_constructor(self, monkeypatch):
        """A manifest ``vllm_config`` section reaches the constructor as VLLMConfig."""
        from agilerl.arena.models.networks import VLLMConfig as VLLMConfigSpec
        from agilerl.utils.algo_utils import VLLMConfig

        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
            vllm_config=VLLMConfigSpec(tensor_parallel_size=2),
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch, rollout_mode="colocated")
        assert isinstance(built["vllm_config"], VLLMConfig)
        assert built["vllm_config"].tensor_parallel_size == 2

    def test_colocated_rollout_fills_a_default_vllm_config(self, monkeypatch):
        from agilerl.utils.algo_utils import VLLMConfig

        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch, rollout_mode="colocated")

        assert isinstance(built["vllm_config"], VLLMConfig)

    @pytest.mark.parametrize(
        "vllm_config", [None, VLLMConfigSpec(gpu_memory_utilization=0.6)]
    )
    def test_colocated_vllm_sleeps(self, monkeypatch, vllm_config):
        # Arrange
        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
            vllm_config=vllm_config,
        )

        # Act
        built = _build_llm_with_dummy_algo(spec, monkeypatch, rollout_mode="colocated")

        # Assert
        assert built["vllm_config"].sleep_mode is True

    def test_colocated_leaves_sampling_logprob_options_to_the_constructor(
        self, monkeypatch
    ):
        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch, rollout_mode="colocated")

        assert "old_logprobs_source" not in built
        assert "vllm_importance_sampling_correction" not in built

    def test_vllm_config_is_dropped_when_the_trainer_does_not_host_vllm(
        self, monkeypatch
    ):
        from agilerl.arena.models.networks import VLLMConfig as VLLMConfigSpec

        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
            vllm_config=VLLMConfigSpec(tensor_parallel_size=2),
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch)
        assert "vllm_config" not in built

    def test_sft_does_not_forward_vllm(self, monkeypatch):
        spec = SFTSpec(pretrained_model_name_or_path="stub/model")
        built = _build_llm_with_dummy_algo(spec, monkeypatch)

        assert "vllm_config" not in built

    def test_tower_lora_targets_reach_the_constructor(self, monkeypatch):
        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
            lora_config=LoraConfigDict(
                target_modules=["q_proj"], tower_target_modules=["query"]
            ),
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch)

        assert built["lora_tower_target_modules"] == ["query"]
        assert set(built["lora_config"].target_modules) == {"q_proj"}

    def test_defaults_index_to_zero(self, monkeypatch):
        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch)

        assert built["index"] == 0
        assert built["device"] == "cpu"

    def test_fsdp_spec_reaches_the_constructor_as_fsdp_config(self, monkeypatch):
        from agilerl.distributed import FSDPConfig

        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
            fsdp=True,
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch)

        assert built["fsdp_config"] == FSDPConfig()

    def test_distributed_omitted_micro_batch_is_rank_trajectories(self, monkeypatch):
        monkeypatch.setattr(llm_builder, "is_distributed", lambda: True)
        monkeypatch.setattr(llm_builder, "get_world_size", lambda: 2)
        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
            batch_size=8,
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch)

        assert built["micro_batch_size_per_gpu"] == resolve_micro_batch_size_per_gpu(
            spec, 2
        )
        assert built["micro_batch_size_per_gpu"] == 8

    def test_cpu_omitted_micro_batch_stays_unset(self, monkeypatch):
        spec = GRPOSpec(
            group_size=2,
            pretrained_model_name_or_path="stub/model",
            batch_size=8,
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch)

        assert "micro_batch_size_per_gpu" not in built

    def test_cuda_omitted_micro_batch_is_the_fit(self, monkeypatch):
        _patch_cuda_card(monkeypatch)
        spec = GRPOSpec(
            group_size=4,
            batch_size=4,
            pretrained_model_name_or_path=CATALOG_HUB_ID,
            max_model_len=1024,
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch, device="cuda:0")

        device = _cuda_device_spec()
        settings = training_settings_from_spec(spec, 1, micro_batch_size=1)
        expected = resolve_micro_batch_size_per_gpu(spec, 1, device, settings=settings)
        omitted = resolve_micro_batch_size_per_gpu(spec, 1)

        assert built["micro_batch_size_per_gpu"] == expected
        assert expected < omitted

    def test_cuda_omitted_micro_batch_uses_launcher_world_size(self, monkeypatch):
        card_bytes = 24 * 1024**3
        _patch_cuda_card(monkeypatch, total_bytes=card_bytes)
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        spec = GRPOSpec(
            group_size=2,
            batch_size=8,
            pretrained_model_name_or_path=CATALOG_HUB_ID,
            max_model_len=128,
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch, device="cuda:0")

        device = _cuda_device_spec(total_bytes=card_bytes)
        settings = training_settings_from_spec(spec, 4, micro_batch_size=1)
        expected = resolve_micro_batch_size_per_gpu(spec, 4, device, settings=settings)
        world_one = resolve_micro_batch_size_per_gpu(
            spec,
            1,
            device,
            settings=training_settings_from_spec(spec, 1, micro_batch_size=1),
        )

        assert built["micro_batch_size_per_gpu"] == expected
        assert expected == 4
        assert world_one == 16

    def test_cuda_explicit_micro_batch_is_kept(self, monkeypatch):
        _patch_cuda_card(monkeypatch, total_bytes=1024**3)
        spec = GRPOSpec(
            group_size=4,
            batch_size=4,
            pretrained_model_name_or_path=CATALOG_HUB_ID,
            max_model_len=128,
            micro_batch_size_per_gpu=8,
        )
        built = _build_llm_with_dummy_algo(spec, monkeypatch, device="cuda:0")

        assert built["micro_batch_size_per_gpu"] == 8


@requires_llm
class TestLLMBuilderBuildDerivedEngineSettings:
    @pytest.mark.parametrize(
        ("modality", "expected"), [("text", True), ("vision", False), (None, False)]
    )
    def test_text_only_env_strips_a_vl_model_towers(
        self, monkeypatch, modality, expected
    ):
        spec = GRPOSpec(
            group_size=4,
            batch_size=4,
            pretrained_model_name_or_path="Qwen/Qwen3.5-2B",
            max_model_len=128,
            vllm_config=VLLMConfigSpec(),
        )
        environment = LLMEnvSpec(
            env_type="rollout", env_url="http://env", max_turns=1, modality=modality
        )

        built = _build_llm_with_dummy_algo(spec, monkeypatch, environment=environment)

        assert built["rollout_strip_multimodal_towers"] is expected


_TINY_TEXT_CONFIG = {
    "model_type": "llama",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
}


class _NoRowLength:
    def __init__(self) -> None:
        pass


class _TakesMaxModelLen:
    def __init__(self, max_model_len: int = 1024) -> None:
        self.max_model_len = max_model_len


@pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES,
    reason="LLM specs are registered when the LLM extra is installed.",
)
class TestFSDPConfigForAlgorithm:
    def test_cuda_fills_unset_fields_from_the_spec_context_length(
        self, monkeypatch
    ) -> None:
        # Arrange
        _patch_cuda_card(monkeypatch)
        spec = DPOSpec(
            pretrained_model_name_or_path="tiny",
            max_model_len=16384,
            fsdp=FSDPConfig(),
        )
        seen: list[int] = []
        real_resolve = llm_builder.resolve_fsdp_memory

        def spy(model, device, settings, breakdown, orchestrated, micro_batch_floor):
            seen.append(settings.max_model_len)
            return real_resolve(
                model,
                device,
                settings,
                breakdown,
                orchestrated=orchestrated,
                micro_batch_floor=micro_batch_floor,
            )

        monkeypatch.setattr(llm_builder, "resolve_fsdp_memory", spy)
        monkeypatch.setattr(
            "transformers.PretrainedConfig.get_config_dict",
            lambda _name, **_kwargs: (_TINY_TEXT_CONFIG, {}),
        )

        # Act
        resolved = llm_builder.fsdp_config_for_algorithm(
            spec, _NoRowLength, "cuda", 1, 1
        )

        # Assert
        assert seen == [16384]
        assert resolved.routed_expert_chunk_mib is not None
        assert resolved.optim_cpu_offload is not None

    def test_sft_uses_the_spec_context_length(self, monkeypatch) -> None:
        # Arrange
        _patch_cuda_card(monkeypatch)
        spec = SFTSpec(
            pretrained_model_name_or_path="tiny",
            max_model_len=8192,
            fsdp=FSDPConfig(),
        )
        seen: list[int] = []
        real_resolve = llm_builder.resolve_fsdp_memory

        def spy(model, device, settings, breakdown, orchestrated, micro_batch_floor):
            seen.append(settings.max_model_len)
            return real_resolve(
                model,
                device,
                settings,
                breakdown,
                orchestrated=orchestrated,
                micro_batch_floor=micro_batch_floor,
            )

        monkeypatch.setattr(llm_builder, "resolve_fsdp_memory", spy)
        monkeypatch.setattr(
            "transformers.PretrainedConfig.get_config_dict",
            lambda _name, **_kwargs: (_TINY_TEXT_CONFIG, {}),
        )

        # Act
        llm_builder.fsdp_config_for_algorithm(spec, _NoRowLength, "cuda", 2, 1)

        # Assert
        assert seen == [8192]

    def test_keeps_unset_fields_when_the_constructor_takes_max_model_len(self) -> None:
        fsdp = FSDPConfig()
        spec = DPOSpec(
            pretrained_model_name_or_path="tiny",
            max_model_len=2048,
            fsdp=fsdp,
        )

        resolved = llm_builder.fsdp_config_for_algorithm(
            spec, _TakesMaxModelLen, "cuda", 1, 1
        )

        assert resolved is fsdp
        assert resolved.routed_expert_chunk_mib is None

    def test_keeps_unset_fields_off_cuda(self) -> None:
        fsdp = FSDPConfig()
        spec = SFTSpec(pretrained_model_name_or_path="tiny", fsdp=fsdp)

        resolved = llm_builder.fsdp_config_for_algorithm(
            spec, _NoRowLength, "cpu", 1, 1
        )

        assert resolved is fsdp

    def test_keeps_fields_that_are_already_set(self) -> None:
        fsdp = FSDPConfig(
            reshard_after_forward=True,
            routed_expert_chunk_mib=64,
            optim_cpu_offload=True,
            activation_offload=False,
            cpu_offload=False,
        )
        spec = DPOSpec(pretrained_model_name_or_path="tiny", fsdp=fsdp)

        resolved = llm_builder.fsdp_config_for_algorithm(
            spec, _NoRowLength, "cuda", 1, 1
        )

        assert resolved is fsdp

    def test_returns_none_without_an_fsdp_section(self) -> None:
        spec = DPOSpec(pretrained_model_name_or_path="tiny")

        assert (
            llm_builder.fsdp_config_for_algorithm(spec, _NoRowLength, "cuda", 1, 1)
            is None
        )

    def test_cuda_without_model_id_raises(self) -> None:
        spec = DPOSpec(fsdp=FSDPConfig())

        with pytest.raises(
            ValueError, match="FSDP memory settings need pretrained_model_name_or_path"
        ):
            llm_builder.fsdp_config_for_algorithm(spec, _NoRowLength, "cuda", 1, 1)
