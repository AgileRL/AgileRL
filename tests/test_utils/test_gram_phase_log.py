# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from agilerl.utils.gram_phase_log import (
    GRAM_PHASE_LOG_ENV,
    log_gram_phase,
    snapshot_gram,
)


class TestGramPhaseLog:
    def test_log_gram_phase_emits_peaks_when_enabled(self, monkeypatch, caplog) -> None:
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.current_device.return_value = 0
        cuda.memory_allocated.side_effect = [10, 20]
        cuda.memory_reserved.side_effect = [30, 40]
        cuda.max_memory_allocated.return_value = 50
        cuda.max_memory_reserved.return_value = 60

        monkeypatch.setenv(GRAM_PHASE_LOG_ENV, "1")
        monkeypatch.setattr("agilerl.utils.gram_phase_log.torch.cuda", cuda)

        caplog.set_level("INFO", logger="agilerl.gram_phase")
        with log_gram_phase("forward"):
            pass

        cuda.reset_peak_memory_stats.assert_called_once_with(0)
        assert "GRAM phase=forward" in caplog.text
        assert "peak_alloc_gb=" in caplog.text

    def test_log_gram_phase_is_silent_when_disabled(self, monkeypatch, caplog) -> None:
        cuda = MagicMock()
        cuda.is_available.return_value = True
        monkeypatch.delenv(GRAM_PHASE_LOG_ENV, raising=False)
        monkeypatch.setattr("agilerl.utils.gram_phase_log.torch.cuda", cuda)

        caplog.set_level("INFO", logger="agilerl.gram_phase")
        with log_gram_phase("forward"):
            pass

        cuda.reset_peak_memory_stats.assert_not_called()
        assert "GRAM phase=" not in caplog.text

    def test_snapshot_gram_is_silent_when_disabled(self, monkeypatch, caplog) -> None:
        cuda = MagicMock()
        cuda.is_available.return_value = True
        monkeypatch.delenv(GRAM_PHASE_LOG_ENV, raising=False)
        monkeypatch.setattr("agilerl.utils.gram_phase_log.torch.cuda", cuda)

        caplog.set_level("INFO", logger="agilerl.gram_phase")
        snapshot_gram("after_setup")

        cuda.memory_allocated.assert_not_called()
        assert "GRAM snapshot" not in caplog.text
