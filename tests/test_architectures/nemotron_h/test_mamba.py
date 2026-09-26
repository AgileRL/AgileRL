# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Nemotron-H Mamba2 mixer class-level patches.

The mixer class and CUDA streams are replaced with doubles, so these run
without transformers or a GPU.
"""

import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
import torch

import agilerl.architectures.nemotron_h.mamba as mamba
from agilerl.architectures.nemotron_h.mamba import (
    patch_nemotron_mamba_fused_path,
    patch_nemotron_mamba_stream_ordering,
)
from agilerl.architectures.runtime import MambaPatchConfig, PatchRuntimeConfig

MIXER = "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer"


def scaled_dot_product_attention(value: torch.Tensor) -> torch.Tensor:
    """Stand-in whose name matches the SDPA call the patch detects."""
    return value * float("nan")


class FakeConfig:
    """Config double carrying the fused-path preference the mixer reads."""

    def __init__(self, use_mem_eff_path=True):
        self.use_mem_eff_path = use_mem_eff_path


def _make_fused_path_mixer_class(events):
    class Mixer:
        def __init__(self, config, layer_idx=None):
            events.append("init")
            self.config = config
            self.layer_idx = layer_idx
            self.use_mem_eff_path = config.use_mem_eff_path

    return Mixer


def _make_mixer_class_without_mem_eff_attr():
    class Mixer:
        def __init__(self, config):
            self.config = config

    return Mixer


class FakeModel:
    """Model double exposing only the module walk the instance sweep needs."""

    def __init__(self, submodules):
        self.submodules = submodules

    def modules(self):
        return iter(self.submodules)


class TestNemotronMambaFusedPath:
    def test_apply_is_idempotent(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        patch_nemotron_mamba_fused_path(mixer=MIXER)
        patched_init = mixer_cls.__init__
        patch_nemotron_mamba_fused_path(mixer=MIXER)
        patch_nemotron_mamba_fused_path(mixer=MIXER)

        assert mixer_cls.__init__ is patched_init

        mixer = mixer_cls(FakeConfig())

        assert events == ["init"]
        assert mixer.use_mem_eff_path is True

    def test_disabled_leaves_init_unpatched(self, monkeypatch, caplog):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        original_init = mixer_cls.__init__
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        with caplog.at_level(logging.INFO):
            patch_nemotron_mamba_fused_path(mixer=MIXER, enabled=False)

        assert mixer_cls.__init__ is original_init
        assert any("disabled by caller" in record.message for record in caplog.records)
        assert mixer_cls(FakeConfig()).use_mem_eff_path is True

    def test_missing_mixer_class_logs_warning_and_does_not_raise(
        self,
        monkeypatch,
        caplog,
    ):
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: None,
        )

        with caplog.at_level(logging.WARNING):
            patch_nemotron_mamba_fused_path(mixer=MIXER)

        assert any(
            "NemotronHMamba2Mixer unavailable" in record.message
            for record in caplog.records
        )

    def test_missing_attribute_leaves_init_alone(self, monkeypatch):
        mixer_cls = _make_mixer_class_without_mem_eff_attr()
        original_init = mixer_cls.__init__
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        patch_nemotron_mamba_fused_path(mixer=MIXER)

        assert mixer_cls.__init__ is original_init
        assert not hasattr(mixer_cls(FakeConfig()), "use_mem_eff_path")

    def test_original_init_runs_once_and_its_side_effects_survive(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        patch_nemotron_mamba_fused_path(mixer=MIXER)
        config = FakeConfig()

        mixer = mixer_cls(config, layer_idx=7)

        assert events == ["init"]
        assert mixer.config is config
        assert mixer.layer_idx == 7

    def test_instance_keeps_the_fused_path_the_config_asks_for(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        patch_nemotron_mamba_fused_path(mixer=MIXER)

        assert mixer_cls(FakeConfig(use_mem_eff_path=True)).use_mem_eff_path is True

    def test_model_sweep_keeps_mixers_built_before_the_patch(
        self,
        monkeypatch,
        caplog,
    ):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        early = mixer_cls(FakeConfig())
        other = FakeConfig()

        with caplog.at_level(logging.INFO):
            patch_nemotron_mamba_fused_path(
                mixer=MIXER, model=FakeModel([early, other])
            )

        assert early.use_mem_eff_path is True
        assert mixer_cls(FakeConfig()).use_mem_eff_path is True
        assert any(
            "fused scan kept on 1 mixers" in record.message for record in caplog.records
        )

    def test_model_sweep_runs_when_the_class_is_already_patched(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        early = mixer_cls(FakeConfig())
        patch_nemotron_mamba_fused_path(mixer=MIXER)
        assert early.use_mem_eff_path is True

        patch_nemotron_mamba_fused_path(mixer=MIXER, model=FakeModel([early]))

        assert early.use_mem_eff_path is True

    def test_model_without_mixers_leaves_the_model_alone(self, monkeypatch, caplog):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        with caplog.at_level(logging.INFO):
            patch_nemotron_mamba_fused_path(
                mixer=MIXER, model=FakeModel([FakeConfig()])
            )

        assert not any("existing mixers" in record.message for record in caplog.records)

    def test_model_sweep_ignores_a_same_name_class_without_the_attribute(
        self, monkeypatch
    ):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        imposter_cls = type("Mixer", (), {})
        imposter = imposter_cls()

        patch_nemotron_mamba_fused_path(mixer=MIXER, model=FakeModel([imposter]))

        assert not hasattr(imposter, "use_mem_eff_path")

    def test_model_sweep_keeps_a_fused_scan_mixer_on_the_fused_kernel(
        self, monkeypatch
    ):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        def mamba_split_conv1d_scan_combined(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        class Plain(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value

        class Mixer(torch.nn.Module):
            def cuda_kernels_forward(self, value: torch.Tensor) -> torch.Tensor:
                if self.training:
                    return mamba_split_conv1d_scan_combined(value)
                return value

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.cuda_kernels_forward(value)

        Plain.__name__ = "Mixer"
        first = Mixer()
        second = Mixer()
        plain = Plain()
        first.train()
        value = torch.ones(2)

        patch_nemotron_mamba_fused_path(
            mixer=MIXER, model=FakeModel([plain, first, second])
        )

        assert torch.equal(plain(value), value)
        assert torch.equal(first(value), value + 1)
        assert first.training
        assert torch.equal(second(value), value + 1)

    def test_open_time_step_limit_is_clamped_for_the_fused_scan(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        def mamba_split_conv1d_scan_combined(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        class Mixer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.time_step_limit = (0.0, float("inf"))
                self.time_step_min = 0.001
                self.time_step_max = 0.1
                self.seen = None

            def cuda_kernels_forward(self, value: torch.Tensor) -> torch.Tensor:
                self.seen = self.time_step_limit
                if self.training:
                    return mamba_split_conv1d_scan_combined(value)
                return value

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.cuda_kernels_forward(value)

        class Tight(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.time_step_limit = (0.001, 0.1)
                self.time_step_min = 0.001
                self.time_step_max = 0.1
                self.seen = None

            def cuda_kernels_forward(self, value: torch.Tensor) -> torch.Tensor:
                self.seen = self.time_step_limit
                if self.training:
                    return mamba_split_conv1d_scan_combined(value)
                return value

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.cuda_kernels_forward(value)

        Mixer.__name__ = "Mixer"
        Tight.__name__ = "Mixer"
        mixer = Mixer()
        tight = Tight()
        mixer.train()
        value = torch.ones(2)

        patch_nemotron_mamba_fused_path(mixer=MIXER, model=FakeModel([mixer, tight]))

        assert torch.equal(mixer(value), value + 1)
        assert mixer.seen == (0.001, 0.1)
        assert mixer.time_step_limit == (0.0, float("inf"))
        assert mixer.training
        assert torch.equal(tight(value), value + 1)
        assert tight.seen == (0.001, 0.1)
        assert tight.training

    def test_fused_scan_mixer_gathers_dtensor_kernel_arguments(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        seen: list[object] = []

        def mamba_chunk_scan_combined(weight: object, bias: object = None) -> object:
            seen.append(weight)
            seen.append(bias)
            return weight

        module = ModuleType("agilerl_tests_remote_mamba")
        module.mamba_chunk_scan_combined = mamba_chunk_scan_combined
        sys.modules[module.__name__] = module

        def mamba_split_conv1d_scan_combined(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        class Mixer(torch.nn.Module):
            def cuda_kernels_forward(self, value: torch.Tensor) -> torch.Tensor:
                if self.training:
                    return mamba_split_conv1d_scan_combined(value)
                return value + 1

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.cuda_kernels_forward(value)

        class MissingModule(torch.nn.Module):
            def cuda_kernels_forward(self, value: torch.Tensor) -> torch.Tensor:
                return mamba_split_conv1d_scan_combined(value)

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value

        Mixer.__name__ = "Mixer"
        Mixer.__module__ = module.__name__
        MissingModule.__name__ = "Mixer"
        MissingModule.__module__ = "agilerl_tests_remote_mamba_missing"
        mixer = Mixer()
        missing = MissingModule()
        mixer.train()

        class DTensor:
            def full_tensor(self) -> torch.Tensor:
                return torch.full((2,), 4.0)

        stuck = type("DTensor", (), {"full_tensor": None})()

        patch_nemotron_mamba_fused_path(mixer=MIXER, model=FakeModel([mixer, missing]))
        patch_nemotron_mamba_fused_path(mixer=MIXER, model=FakeModel([mixer]))

        gathered = module.mamba_chunk_scan_combined(DTensor(), bias=DTensor())
        bare = module.mamba_chunk_scan_combined(stuck)
        plain = torch.ones(2)
        passed = module.mamba_chunk_scan_combined(plain)
        value = torch.ones(2)

        assert torch.equal(gathered, torch.full((2,), 4.0))
        assert bare is stuck
        assert torch.equal(passed, plain)
        assert torch.equal(mixer(value), value + 1)
        assert mixer.training
        assert seen[0].__class__.__name__ == "Tensor"
        assert torch.equal(seen[1], torch.full((2,), 4.0))

    def test_ada_ssm_kernel_runs_floating_inputs_in_fp32(self, monkeypatch):
        class Device:
            type = "cuda"

        class Weight:
            device = Device()
            dtype = torch.bfloat16

            def is_floating_point(self) -> bool:
                return True

        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 9))
        assert mamba._ssm_scan_in_fp32(Weight())
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))
        assert not mamba._ssm_scan_in_fp32(Weight())
        Weight.dtype = torch.float32
        assert not mamba._ssm_scan_in_fp32(Weight())

        class Integer:
            device = Device()
            dtype = torch.int64
            is_floating_point = None

        class Quiet:
            device = Device()
            dtype = torch.bfloat16

            def is_floating_point(self) -> bool:
                return False

        assert not mamba._ssm_scan_in_fp32(Integer())
        assert not mamba._ssm_scan_in_fp32(Quiet())
        assert not mamba._ssm_scan_in_fp32(torch.ones(2, dtype=torch.bfloat16))
        kept = torch.ones(2)
        assert mamba._cast_floating_to(kept, torch.float32) is kept

        monkeypatch.setattr(mamba, "_ssm_scan_in_fp32", lambda _value: True)
        seen: list[torch.dtype] = []

        def kernel(weight: torch.Tensor) -> tuple[torch.Tensor, str]:
            seen.append(weight.dtype)
            return weight, "state"

        wrapped = mamba._call_kernel_with_full_tensors(kernel)
        out, state = wrapped(torch.ones(2, dtype=torch.bfloat16))
        again, state_again = wrapped(torch.ones(2, dtype=torch.bfloat16))

        assert seen == [torch.float32, torch.float32]
        assert out.dtype == torch.bfloat16
        assert again.dtype == torch.bfloat16
        assert state == "state"
        assert state_again == "state"

    def test_model_sweep_patches_a_same_name_mixer_that_sets_the_attribute(
        self, monkeypatch
    ):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        remote_cls = _make_fused_path_mixer_class(events)
        remote_cls.__module__ = "transformers_modules.nano.modeling_nemotron_h"
        early = remote_cls(FakeConfig(use_mem_eff_path=True))

        patch_nemotron_mamba_fused_path(mixer=MIXER, model=FakeModel([early]))

        assert early.use_mem_eff_path is True
        later = remote_cls(FakeConfig(use_mem_eff_path=True))
        assert later.use_mem_eff_path is True

    def test_model_sweep_keeps_a_mixer_that_sets_the_attribute_indirectly(
        self, monkeypatch
    ):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        def assign(module: object) -> None:
            module.use_mem_eff_path = True

        class Mixer:
            def __init__(self) -> None:
                assign(self)

        early = Mixer()

        patch_nemotron_mamba_fused_path(mixer=MIXER, model=FakeModel([early]))

        assert early.use_mem_eff_path is True
        assert Mixer().use_mem_eff_path is True

    def test_disabled_skips_the_model_sweep(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        early = mixer_cls(FakeConfig())

        patch_nemotron_mamba_fused_path(
            mixer=MIXER, enabled=False, model=FakeModel([early])
        )

        assert early.use_mem_eff_path is True

    def test_missing_mixer_class_skips_the_model_sweep(self, monkeypatch, caplog):
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: None,
        )
        events = []
        early = _make_fused_path_mixer_class(events)(FakeConfig())

        with caplog.at_level(logging.WARNING):
            patch_nemotron_mamba_fused_path(mixer=MIXER, model=FakeModel([early]))

        assert early.use_mem_eff_path is True


class FakeStream:
    """Stream double recording the streams it is told to wait on."""

    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait_stream(self, other):
        self.events.append(f"{self.name}.wait_stream({other.name})")


class FakeTensor:
    """Tensor double exposing the attributes the patch inspects."""

    def __init__(self, device="cuda:0", is_cuda=True):
        self.device = device
        self.is_cuda = is_cuda
        self.recorded = []

    def record_stream(self, stream):
        self.recorded.append(stream.name)


class Streams:
    """Current and default stream doubles sharing the caller's event log."""

    def __init__(self, events, same=False):
        self.events = events
        self.default = FakeStream("default", events)
        self.current = self.default if same else FakeStream("current", events)


def _make_stream_mixer_class(events, output):
    class Mixer:
        calls: ClassVar[list] = []

        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
            events.append("forward")
            Mixer.calls.append((hidden_states, cache_params, attention_mask, kw))
            return output

    return Mixer


@pytest.fixture
def cuda_env(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _install_streams(monkeypatch, streams):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: streams.current)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device: streams.default)


class TestNemotronMambaStreamOrdering:
    def test_apply_is_idempotent(self, cuda_env, monkeypatch):
        events = []
        mixer_cls = _make_stream_mixer_class(events, FakeTensor())
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        patch_nemotron_mamba_stream_ordering(mixer=MIXER)
        patched_forward = mixer_cls.forward
        patch_nemotron_mamba_stream_ordering(mixer=MIXER)
        patch_nemotron_mamba_stream_ordering(mixer=MIXER)

        assert mixer_cls.forward is patched_forward

        streams = Streams(events)
        _install_streams(monkeypatch, streams)
        mixer_cls().forward(FakeTensor())

        assert events.count("forward") == 1

    def test_disabled_leaves_forward_unpatched(self, cuda_env, monkeypatch, caplog):
        events = []
        mixer_cls = _make_stream_mixer_class(events, FakeTensor())
        original_forward = mixer_cls.forward
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        with caplog.at_level(logging.INFO):
            patch_nemotron_mamba_stream_ordering(mixer=MIXER, enabled=False)

        assert mixer_cls.forward is original_forward
        assert any("disabled by caller" in record.message for record in caplog.records)

        streams = Streams(events)
        _install_streams(monkeypatch, streams)
        mixer_cls().forward(FakeTensor())

        assert events == ["forward"]

    def test_missing_mixer_class_logs_warning_and_does_not_raise(
        self,
        monkeypatch,
        caplog,
    ):
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: None,
        )

        with caplog.at_level(logging.WARNING):
            patch_nemotron_mamba_stream_ordering(mixer=MIXER)

        assert any(
            "NemotronHMamba2Mixer unavailable" in record.message
            for record in caplog.records
        )

    def test_missing_forward_raises(self, monkeypatch):
        class Mixer:
            pass

        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: Mixer,
        )

        with pytest.raises(RuntimeError, match="lacks forward"):
            patch_nemotron_mamba_stream_ordering(mixer=MIXER)

    def test_model_sweep_wraps_a_same_name_mixer(self, monkeypatch):
        events = []
        mixer_cls = _make_stream_mixer_class(events, FakeTensor())
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        remote_cls = _make_stream_mixer_class(events, FakeTensor())
        original_remote_forward = remote_cls.forward

        class Mixer:
            pass

        patch_nemotron_mamba_stream_ordering(
            mixer=MIXER,
            model=FakeModel([remote_cls(), remote_cls(), Mixer()]),
        )

        assert remote_cls.forward is not original_remote_forward

    def test_identical_streams_issue_no_waits(self, cuda_env, monkeypatch):
        events = []
        output = FakeTensor()
        mixer_cls = _make_stream_mixer_class(events, output)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering(mixer=MIXER)
        streams = Streams(events, same=True)
        _install_streams(monkeypatch, streams)
        hidden_states = FakeTensor()

        assert mixer_cls().forward(hidden_states) is output
        assert events == ["forward"]
        assert hidden_states.recorded == []
        assert output.recorded == []

    def test_distinct_streams_wait_in_order_around_the_original(
        self,
        cuda_env,
        monkeypatch,
    ):
        events = []
        output = FakeTensor()
        mixer_cls = _make_stream_mixer_class(events, output)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering(mixer=MIXER)
        streams = Streams(events)
        _install_streams(monkeypatch, streams)
        hidden_states = FakeTensor()

        assert mixer_cls().forward(hidden_states, attention_mask="mask") is output
        assert events == [
            "default.wait_stream(current)",
            "forward",
            "current.wait_stream(default)",
        ]
        assert mixer_cls.calls[0][2] == "mask"

    def test_record_stream_called_on_input_and_output(self, cuda_env, monkeypatch):
        events = []
        output = FakeTensor()
        mixer_cls = _make_stream_mixer_class(events, output)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering(mixer=MIXER)
        streams = Streams(events)
        _install_streams(monkeypatch, streams)
        hidden_states = FakeTensor()

        mixer_cls().forward(hidden_states)

        assert hidden_states.recorded == ["default"]
        assert output.recorded == ["default", "current"]

    def test_record_stream_reaches_tensors_inside_a_tuple_output(
        self,
        cuda_env,
        monkeypatch,
    ):
        events = []
        first, second = FakeTensor(), FakeTensor()
        mixer_cls = _make_stream_mixer_class(events, (first, None, second))
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering(mixer=MIXER)
        streams = Streams(events)
        _install_streams(monkeypatch, streams)

        mixer_cls().forward(FakeTensor())

        assert first.recorded == ["default", "current"]
        assert second.recorded == ["default", "current"]

    def test_non_cuda_input_falls_through_untouched(self, cuda_env, monkeypatch):
        events = []
        output = FakeTensor()
        mixer_cls = _make_stream_mixer_class(events, output)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering(mixer=MIXER)
        streams = Streams(events)
        _install_streams(monkeypatch, streams)
        hidden_states = FakeTensor(device="cpu", is_cuda=False)

        assert mixer_cls().forward(hidden_states) is output
        assert events == ["forward"]
        assert output.recorded == []

    def test_model_is_accepted_for_dispatch_parity_and_ignored(
        self,
        cuda_env,
        monkeypatch,
    ):
        events = []
        output = FakeTensor()
        mixer_cls = _make_stream_mixer_class(events, output)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )

        patch_nemotron_mamba_stream_ordering(mixer=MIXER, model=FakeModel([]))

        streams = Streams(events)
        _install_streams(monkeypatch, streams)
        assert mixer_cls().forward(FakeTensor()) is output
        assert events == [
            "default.wait_stream(current)",
            "forward",
            "current.wait_stream(default)",
        ]

    def test_falls_through_when_cuda_is_unavailable(self, cuda_env, monkeypatch):
        events = []
        output = FakeTensor()
        mixer_cls = _make_stream_mixer_class(events, output)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering(mixer=MIXER)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        streams = Streams(events)
        _install_streams(monkeypatch, streams)

        assert mixer_cls().forward(FakeTensor()) is output
        assert events == ["forward"]


class TestBothMixerPatchesCoexist:
    def test_stream_and_fused_path_patches_apply_to_one_class(
        self,
        cuda_env,
        monkeypatch,
    ):
        events = []

        class Mixer:
            def __init__(self, config):
                events.append("init")
                self.use_mem_eff_path = config.use_mem_eff_path

            def forward(self, hidden_states, **kwargs):
                events.append("forward")
                return hidden_states

        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda _mixer: Mixer,
        )
        patch_nemotron_mamba_fused_path(mixer=MIXER)
        patch_nemotron_mamba_stream_ordering(mixer=MIXER)
        _install_streams(monkeypatch, Streams(events))

        mixer = Mixer(FakeConfig())
        hidden_states = FakeTensor()
        mixer.forward(hidden_states)

        assert mixer.use_mem_eff_path is True
        assert events == [
            "init",
            "default.wait_stream(current)",
            "forward",
            "current.wait_stream(default)",
        ]


class TestResolveMixerClass:
    def test_raises_when_attribute_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(mamba, "try_import", lambda _path: type("M", (), {})())

        with pytest.raises(RuntimeError, match="NemotronHMamba2Mixer"):
            mamba._resolve_mixer_class(MIXER)

    def test_returns_none_when_module_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(mamba, "try_import", lambda _path: None)

        assert mamba._resolve_mixer_class(MIXER) is None

    def test_returns_the_mixer(self, monkeypatch) -> None:
        mixer = type("NemotronHMamba2Mixer", (), {})
        seen: list[str] = []

        def fake_import(path):
            seen.append(path)
            return SimpleNamespace(NemotronHMamba2Mixer=mixer)

        monkeypatch.setattr(mamba, "try_import", fake_import)

        assert mamba._resolve_mixer_class(MIXER) is mixer
        assert seen == ["transformers.models.nemotron_h.modeling_nemotron_h"]

    @pytest.mark.parametrize("mixer", ["NemotronHMamba2Mixer", "module.", ".Mixer", ""])
    def test_invalid_path_raises(self, mixer: str) -> None:
        with pytest.raises(RuntimeError, match="invalid mixer path"):
            mamba._resolve_mixer_class(mixer)


class TestInstallMambaPatches:
    def test_runs_fused_path_and_stream_ordering(self, monkeypatch):
        seen: list[tuple[str, object]] = []
        actor = torch.nn.Module()
        monkeypatch.setattr(
            mamba,
            "patch_nemotron_mamba_fused_path",
            lambda *, mixer, model=None: seen.append(("fused", mixer, model)),
        )
        monkeypatch.setattr(
            mamba,
            "patch_nemotron_mamba_stream_ordering",
            lambda *, mixer, model=None: seen.append(("stream", mixer, model)),
        )

        mamba.install_mamba_patches(
            PatchRuntimeConfig(mamba=MambaPatchConfig(mixer=MIXER)), model=actor
        )

        assert seen == [("fused", MIXER, actor), ("stream", MIXER, actor)]

    def test_fused_path_false_skips_fused_path(self, monkeypatch):
        seen: list[tuple[str, object]] = []
        actor = torch.nn.Module()
        monkeypatch.setattr(
            mamba,
            "patch_nemotron_mamba_fused_path",
            lambda *, mixer, model=None: seen.append(("fused", mixer, model)),
        )
        monkeypatch.setattr(
            mamba,
            "patch_nemotron_mamba_stream_ordering",
            lambda *, mixer, model=None: seen.append(("stream", mixer, model)),
        )

        mamba.install_mamba_patches(
            PatchRuntimeConfig(mamba=MambaPatchConfig(mixer=MIXER, fused_path=False)),
            model=actor,
        )

        assert seen == [("stream", MIXER, actor)]

    def test_stream_ordering_false_skips_stream_ordering(self, monkeypatch):
        seen: list[tuple[str, object]] = []
        actor = torch.nn.Module()
        monkeypatch.setattr(
            mamba,
            "patch_nemotron_mamba_fused_path",
            lambda *, mixer, model=None: seen.append(("fused", mixer, model)),
        )
        monkeypatch.setattr(
            mamba,
            "patch_nemotron_mamba_stream_ordering",
            lambda *, mixer, model=None: seen.append(("stream", mixer, model)),
        )

        mamba.install_mamba_patches(
            PatchRuntimeConfig(
                mamba=MambaPatchConfig(mixer=MIXER, stream_ordering=False)
            ),
            model=actor,
        )

        assert seen == [("fused", MIXER, actor)]

    def test_missing_mamba_is_a_no_op(self, monkeypatch):
        seen: list[tuple[str, object]] = []
        monkeypatch.setattr(
            mamba,
            "patch_nemotron_mamba_fused_path",
            lambda *, mixer, model=None: seen.append(("fused", mixer, model)),
        )
        monkeypatch.setattr(
            mamba,
            "patch_nemotron_mamba_stream_ordering",
            lambda *, mixer, model=None: seen.append(("stream", mixer, model)),
        )

        mamba.install_mamba_patches(PatchRuntimeConfig(), model=object())

        assert seen == []


class TestSdpaFullyMaskedRows:
    def test_fully_masked_sdpa_rows_become_zero(self):
        class Attention(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, None]:
                return scaled_dot_product_attention(value), None

        class TensorAttention(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return scaled_dot_product_attention(value)

        class Plain(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value

        class NoCode:
            forward = None

        class Odd(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> str:
                scaled_dot_product_attention(value)
                return "ok"

        class Empty(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> tuple[()]:
                scaled_dot_product_attention(value)
                return ()

        class Label(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> tuple[str]:
                scaled_dot_product_attention(value)
                return ("x",)

        first = Attention()
        second = Attention()
        tensor_attention = TensorAttention()
        plain = Plain()
        odd = Odd()
        empty = Empty()
        label = Label()
        value = torch.ones(2)

        mamba.patch_sdpa_fully_masked_rows(
            FakeModel(
                [first, second, tensor_attention, plain, NoCode(), odd, empty, label]
            )
        )
        mamba.patch_sdpa_fully_masked_rows(FakeModel([first]))

        finite, extra = first(value)
        second_finite, second_extra = second(value)

        assert torch.equal(finite, torch.zeros(2))
        assert extra is None
        assert torch.equal(second_finite, torch.zeros(2))
        assert second_extra is None
        assert torch.equal(tensor_attention(value), torch.zeros(2))
        assert torch.equal(plain(value), value)
        assert odd(value) == "ok"
        assert empty(value) == ()
        assert label(value) == ("x",)


class TestBlockTypeMaskMapping:
    def test_builds_full_and_linear_keys(self, monkeypatch):
        masking = SimpleNamespace(
            create_causal_mask=lambda **_kw: "causal",
            create_recurrent_attention_mask=lambda **_kw: "linear",
        )
        monkeypatch.setattr(
            mamba,
            "try_import",
            lambda path: masking if path == "transformers.masking_utils" else None,
        )
        embeds = torch.zeros(1, 3, 2)

        mapping, position_ids = mamba.block_type_mask_mapping(
            SimpleNamespace(config=object()),
            embeds=embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=None,
        )

        assert mapping == {
            "full_attention": "causal",
            "linear_attention": "linear",
        }
        assert position_ids.shape == (1, 3)

    def test_missing_masking_utils_returns_empty_mapping(self, monkeypatch):
        monkeypatch.setattr(mamba, "try_import", lambda _path: None)

        mapping, position_ids = mamba.block_type_mask_mapping(
            SimpleNamespace(config=object()),
            embeds=torch.zeros(1, 2, 2),
            attention_mask=None,
            past_key_values=None,
            position_ids=None,
        )

        assert mapping == {}
        assert position_ids.shape == (1, 2)
