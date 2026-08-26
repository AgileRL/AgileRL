# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Nemotron-H Mamba2 mixer class-level patches.

The mixer class and CUDA streams are replaced with doubles, so these run
without transformers or a GPU.
"""

import logging
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

import agilerl.architectures.nemotron_h.mamba as mamba
from agilerl.architectures.nemotron_h.mamba import (
    patch_nemotron_mamba_fused_path,
    patch_nemotron_mamba_stream_ordering,
)


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
            lambda: mixer_cls,
        )

        patch_nemotron_mamba_fused_path()
        patched_init = mixer_cls.__init__
        patch_nemotron_mamba_fused_path()
        patch_nemotron_mamba_fused_path()

        assert mixer_cls.__init__ is patched_init

        mixer = mixer_cls(FakeConfig())

        assert events == ["init"]
        assert mixer.use_mem_eff_path is False

    def test_disabled_leaves_init_unpatched(self, monkeypatch, caplog):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        original_init = mixer_cls.__init__
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )

        with caplog.at_level(logging.INFO):
            patch_nemotron_mamba_fused_path(enabled=False)

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
            lambda: None,
        )

        with caplog.at_level(logging.WARNING):
            patch_nemotron_mamba_fused_path()

        assert any(
            "NemotronHMamba2Mixer unavailable" in record.message
            for record in caplog.records
        )

    def test_missing_attribute_raises(self, monkeypatch):
        mixer_cls = _make_mixer_class_without_mem_eff_attr()
        original_init = mixer_cls.__init__
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )

        with pytest.raises(RuntimeError, match="does not set use_mem_eff_path"):
            patch_nemotron_mamba_fused_path()

        assert mixer_cls.__init__ is original_init
        assert not hasattr(mixer_cls(FakeConfig()), "use_mem_eff_path")

    def test_original_init_runs_once_and_its_side_effects_survive(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )
        patch_nemotron_mamba_fused_path()
        config = FakeConfig()

        mixer = mixer_cls(config, layer_idx=7)

        assert events == ["init"]
        assert mixer.config is config
        assert mixer.layer_idx == 7

    def test_instance_drops_the_fused_path_the_config_asks_for(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )
        patch_nemotron_mamba_fused_path()

        assert mixer_cls(FakeConfig(use_mem_eff_path=True)).use_mem_eff_path is False

    def test_model_sweep_clears_mixers_built_before_the_patch(
        self,
        monkeypatch,
        caplog,
    ):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )
        early = mixer_cls(FakeConfig())
        other = FakeConfig()

        with caplog.at_level(logging.INFO):
            patch_nemotron_mamba_fused_path(model=FakeModel([early, other]))

        assert early.use_mem_eff_path is False
        assert mixer_cls(FakeConfig()).use_mem_eff_path is False
        assert any(
            "cleared on 1 existing mixers" in record.message
            for record in caplog.records
        )

    def test_model_sweep_runs_when_the_class_is_already_patched(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )
        early = mixer_cls(FakeConfig())
        patch_nemotron_mamba_fused_path()
        assert early.use_mem_eff_path is True

        patch_nemotron_mamba_fused_path(model=FakeModel([early]))

        assert early.use_mem_eff_path is False

    def test_model_without_mixers_leaves_the_model_alone(self, monkeypatch, caplog):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )

        with caplog.at_level(logging.INFO):
            patch_nemotron_mamba_fused_path(model=FakeModel([FakeConfig()]))

        assert not any("existing mixers" in record.message for record in caplog.records)

    def test_model_sweep_raises_on_mixer_class_identity_skew(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )
        imposter_cls = type("Mixer", (), {})

        with pytest.raises(RuntimeError, match="different mixer class"):
            patch_nemotron_mamba_fused_path(model=FakeModel([imposter_cls()]))

    def test_disabled_skips_the_model_sweep(self, monkeypatch):
        events = []
        mixer_cls = _make_fused_path_mixer_class(events)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )
        early = mixer_cls(FakeConfig())

        patch_nemotron_mamba_fused_path(enabled=False, model=FakeModel([early]))

        assert early.use_mem_eff_path is True

    def test_missing_mixer_class_skips_the_model_sweep(self, monkeypatch, caplog):
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: None,
        )
        events = []
        early = _make_fused_path_mixer_class(events)(FakeConfig())

        with caplog.at_level(logging.WARNING):
            patch_nemotron_mamba_fused_path(model=FakeModel([early]))

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
            lambda: mixer_cls,
        )

        patch_nemotron_mamba_stream_ordering()
        patched_forward = mixer_cls.forward
        patch_nemotron_mamba_stream_ordering()
        patch_nemotron_mamba_stream_ordering()

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
            lambda: mixer_cls,
        )

        with caplog.at_level(logging.INFO):
            patch_nemotron_mamba_stream_ordering(enabled=False)

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
            lambda: None,
        )

        with caplog.at_level(logging.WARNING):
            patch_nemotron_mamba_stream_ordering()

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
            lambda: Mixer,
        )

        with pytest.raises(RuntimeError, match="lacks forward"):
            patch_nemotron_mamba_stream_ordering()

    def test_identical_streams_issue_no_waits(self, cuda_env, monkeypatch):
        events = []
        output = FakeTensor()
        mixer_cls = _make_stream_mixer_class(events, output)
        monkeypatch.setattr(
            mamba,
            "_resolve_mixer_class",
            lambda: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering()
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
            lambda: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering()
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
            lambda: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering()
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
            lambda: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering()
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
            lambda: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering()
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
            lambda: mixer_cls,
        )

        patch_nemotron_mamba_stream_ordering(model=FakeModel([]))

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
            lambda: mixer_cls,
        )
        patch_nemotron_mamba_stream_ordering()
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
            lambda: Mixer,
        )
        patch_nemotron_mamba_fused_path()
        patch_nemotron_mamba_stream_ordering()
        _install_streams(monkeypatch, Streams(events))

        mixer = Mixer(FakeConfig())
        hidden_states = FakeTensor()
        mixer.forward(hidden_states)

        assert mixer.use_mem_eff_path is False
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
            mamba._resolve_mixer_class()

    def test_returns_none_when_module_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(mamba, "try_import", lambda _path: None)

        assert mamba._resolve_mixer_class() is None

    def test_returns_the_mixer(self, monkeypatch) -> None:
        mixer = type("NemotronHMamba2Mixer", (), {})
        monkeypatch.setattr(
            mamba,
            "try_import",
            lambda _path: SimpleNamespace(NemotronHMamba2Mixer=mixer),
        )

        assert mamba._resolve_mixer_class() is mixer
