from unittest.mock import patch

import numpy as np
import pytest
import torch

from agilerl.utils.log_utils import DistributeCombineLogs, label_logs


class TestDistributeCombineLogsInit:
    def test_init_dcl_invalid_use_wandb_type(self):
        with pytest.raises(TypeError, match="use_wandb must be a boolean"):
            DistributeCombineLogs(use_wandb="nope")

    def test_init_DCL(self):
        use_wandb = False

        DCL = DistributeCombineLogs(use_wandb=use_wandb)

        assert DCL.totals == {}
        assert isinstance(DCL.device, torch.device)
        assert DCL.use_wandb == use_wandb

    def test_init_dcl_explicit_device(self):
        DCL = DistributeCombineLogs(device="cpu", use_wandb=False)
        assert DCL.device == torch.device("cpu")


class TestDistributeCombineLogsConvertKey:
    def test_convert_key(self):
        DCL = DistributeCombineLogs(use_wandb=False)

        k = (1,)

        conv = DCL.convert_key(k)

        assert conv == ("__count__", 1)
        assert DCL.key_is_count(conv)


class TestDistributeCombineLogsLog:
    def test_log(self):
        with patch("agilerl.utils.log_utils.wandb.log") as mock_wandb_log:

            def dummy_func(a):
                return a

            DCL = DistributeCombineLogs(use_wandb=True)
            _ = DCL.log(dummy_func)

            mock_wandb_log.assert_called()

    def test_log_use_wandb_false(self):
        """log() must not call wandb.log when use_wandb=False."""
        DCL = DistributeCombineLogs(use_wandb=False)

        with patch("agilerl.utils.log_utils.wandb.log") as mock_wandb_log:
            DCL.log()
            mock_wandb_log.assert_not_called()

    def test_log_non_main_process_skips_wandb(self):
        """log() must not call wandb.log on non-main ranks even when use_wandb=True."""
        DCL = DistributeCombineLogs(use_wandb=True)

        with (
            patch("agilerl.utils.log_utils.wandb.log") as mock_wandb_log,
            patch("agilerl.utils.log_utils.is_main_process", return_value=False),
        ):
            DCL.log()
            mock_wandb_log.assert_not_called()


class TestDistributeCombineLogsAccumLogs:
    def test_accum_logs(self):
        DCL = DistributeCombineLogs(use_wandb=False)

        DCL.totals = {("__count__", "a"): [0, 1], ("a",): [1, 2]}
        logs = {"a": [1, 2], "b": [2, 3]}
        DCL.accum_logs(logs)

        assert DCL.totals != {("__count__", "a"): [0, 1], ("a",): [1, 2]}

    def test_accum_logs_growth_path_nested_keys(self):
        """accum_logs growth path: key already in totals, with nested keys."""
        DCL = DistributeCombineLogs(use_wandb=False)

        # First call: initialize totals
        DCL.accum_logs({"nested": {"key": [10.0, 2]}})
        assert ("nested", "key") in DCL.totals
        assert ("__count__", "nested", "key") in DCL.totals

        # Second call: growth path (k in self.totals)
        DCL.accum_logs({"nested": {"key": [20.0, 3]}})
        # 10*2 + 20*3 = 20 + 60 = 80, count 2+3=5
        assert DCL.totals[("nested", "key")].item() == pytest.approx(80.0)
        assert DCL.totals[("__count__", "nested", "key")].item() == pytest.approx(5.0)


class TestDistributeCombineLogsGatherLogs:
    def test_gather_logs(self):
        DCL = DistributeCombineLogs(use_wandb=False)

        DCL.totals = {
            ("__count__", "a"): torch.tensor([0]),
            ("a",): torch.tensor([1, 2]),
        }

        def dummy_func(a):
            return a

        logs = DCL.gather_logs(dummy_func)

        assert logs == {"a": np.inf}

    def test_gather_logs_not_count(self):
        DCL = DistributeCombineLogs(use_wandb=False)

        DCL.totals = {
            ("__count__", "a"): torch.tensor([0, 1]),
            ("a",): torch.tensor([1, 2]),
        }

        def dummy_func(a):
            return a

        logs = DCL.gather_logs(dummy_func)

        assert logs == {"a": 3.0}

    def test_gather_logs_postproc_returns_none(self):
        """When postproc returns None, final_logs is left unchanged."""
        DCL = DistributeCombineLogs(use_wandb=False)
        DCL.totals = {
            ("__count__", "a"): torch.tensor([2]),
            ("a",): torch.tensor([6.0]),
        }

        def return_none(_logs):
            return None

        logs = DCL.gather_logs(return_none)
        assert logs == {"a": 3.0}

    def test_gather_logs_postproc_returns_non_none(self):
        """When postproc returns a dict, it replaces final_logs."""
        DCL = DistributeCombineLogs(use_wandb=False)
        DCL.totals = {
            ("__count__", "a"): torch.tensor([2]),
            ("a",): torch.tensor([6.0]),
        }

        def add_prefix(logs):
            return {"prefixed_a": logs["a"]}

        logs = DCL.gather_logs(add_prefix)
        assert logs == {"prefixed_a": 3.0}

    def test_gather_logs_additional_items_merge(self):
        """additional_items are merged into the returned logs."""
        DCL = DistributeCombineLogs(use_wandb=False)
        DCL.totals = {
            ("__count__", "a"): torch.tensor([1]),
            ("a",): torch.tensor([5.0]),
        }

        logs = DCL.gather_logs(extra_key=42, other="value")
        assert logs["a"] == 5.0
        assert logs["extra_key"] == 42
        assert logs["other"] == "value"

    def test_gather_logs_distributed_all_reduces_totals(self):
        """When distributed, totals are summed across ranks via all_reduce."""
        DCL = DistributeCombineLogs(use_wandb=False)
        DCL.totals = {
            ("__count__", "a"): torch.tensor([1.0]),
            ("a",): torch.tensor([5.0]),
        }

        def double_in_place(tensor, op=None):
            tensor.mul_(2)

        with (
            patch("agilerl.utils.log_utils.is_distributed", return_value=True),
            patch(
                "agilerl.utils.log_utils.dist.all_reduce",
                side_effect=double_in_place,
            ) as mock_all_reduce,
        ):
            logs = DCL.gather_logs()

        assert mock_all_reduce.call_count == 2
        # Both the value-sum and the count double, so the mean is unchanged.
        assert logs == {"a": 5.0}


class TestDistributeCombineLogsKeyIsCount:
    def test_key_is_count_false(self):
        """key_is_count returns False for non-count keys."""
        DCL = DistributeCombineLogs(use_wandb=False)
        assert DCL.key_is_count(("a",)) is False
        assert DCL.key_is_count(("nested", "key")) is False


class TestDistributeCombineLogsResetLogs:
    def test_reset_totals(self):
        DCL = DistributeCombineLogs(use_wandb=False)
        DCL.totals = {"asdfghjkl"}
        DCL.reset_logs()

        assert DCL.totals == {}


def test_label_logs():
    logs = "log"
    label = "label"

    labelled = label_logs(logs, label)

    assert labelled == {label: logs}
