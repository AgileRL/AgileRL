from unittest.mock import patch

import numpy as np
import pytest
import torch
from accelerate import Accelerator

from agilerl.utils.log_utils import DistributeCombineLogs, label_logs


def test_init_dcl_invalid_accelerator_type():
    with pytest.raises(
        TypeError, match="accelerator must be an instance of Accelerator"
    ):
        DistributeCombineLogs(accelerator=object(), use_wandb=False)


def test_init_dcl_invalid_use_wandb_type():
    accelerator = Accelerator()
    with pytest.raises(TypeError, match="use_wandb must be a boolean"):
        DistributeCombineLogs(accelerator=accelerator, use_wandb="nope")


def test_init_DCL():
    accelerator = Accelerator()
    use_wandb = False

    DCL = DistributeCombineLogs(accelerator, use_wandb)

    assert DCL.totals == {}
    assert DCL.accelerator == accelerator
    assert DCL.use_wandb == use_wandb


def test_convert_key():
    accelerator = Accelerator()
    use_wandb = False

    DCL = DistributeCombineLogs(accelerator, use_wandb)

    k = (1,)

    conv = DCL.convert_key(k)

    assert conv == ("__count__", 1)
    assert DCL.key_is_count(conv)


def test_log():
    accelerator = Accelerator()
    use_wandb = True

    with patch("agilerl.utils.log_utils.wandb.log") as mock_wandb_log:

        def dummy_func(a):
            return a

        DCL = DistributeCombineLogs(accelerator, use_wandb)
        _ = DCL.log(dummy_func)

        mock_wandb_log.assert_called()


def test_log_use_wandb_false():
    """log() must not call wandb.log when use_wandb=False."""
    accelerator = Accelerator()
    DCL = DistributeCombineLogs(accelerator, use_wandb=False)

    with patch("agilerl.utils.log_utils.wandb.log") as mock_wandb_log:
        DCL.log()
        mock_wandb_log.assert_not_called()


def test_accum_logs():
    accelerator = Accelerator()
    use_wandb = False

    DCL = DistributeCombineLogs(accelerator, use_wandb)

    DCL.totals = {("__count__", "a"): [0, 1], ("a",): [1, 2]}
    logs = {"a": [1, 2], "b": [2, 3]}
    DCL.accum_logs(logs)

    assert DCL.totals != {("__count__", "a"): [0, 1], ("a",): [1, 2]}


def test_gather_logs():
    accelerator = Accelerator()
    use_wandb = False

    DCL = DistributeCombineLogs(accelerator, use_wandb)

    DCL.totals = {("__count__", "a"): torch.tensor([0]), ("a",): torch.tensor([1, 2])}

    def dummy_func(a):
        return a

    logs = DCL.gather_logs(dummy_func)

    assert logs == {"a": np.inf}


def test_gather_logs_not_count():
    accelerator = Accelerator()
    use_wandb = False

    DCL = DistributeCombineLogs(accelerator, use_wandb)

    DCL.totals = {
        ("__count__", "a"): torch.tensor([0, 1]),
        ("a",): torch.tensor([1, 2]),
    }

    def dummy_func(a):
        return a

    logs = DCL.gather_logs(dummy_func)

    assert logs == {"a": 3.0}


def test_gather_logs_postproc_returns_none():
    """When postproc returns None, final_logs is left unchanged."""
    accelerator = Accelerator()
    DCL = DistributeCombineLogs(accelerator, use_wandb=False)
    DCL.totals = {
        ("__count__", "a"): torch.tensor([2]),
        ("a",): torch.tensor([6.0]),
    }

    def return_none(_logs):
        return None

    logs = DCL.gather_logs(return_none)
    assert logs == {"a": 3.0}


def test_gather_logs_postproc_returns_non_none():
    """When postproc returns a dict, it replaces final_logs."""
    accelerator = Accelerator()
    DCL = DistributeCombineLogs(accelerator, use_wandb=False)
    DCL.totals = {
        ("__count__", "a"): torch.tensor([2]),
        ("a",): torch.tensor([6.0]),
    }

    def add_prefix(logs):
        return {"prefixed_a": logs["a"]}

    logs = DCL.gather_logs(add_prefix)
    assert logs == {"prefixed_a": 3.0}


def test_gather_logs_additional_items_merge():
    """additional_items are merged into the returned logs."""
    accelerator = Accelerator()
    DCL = DistributeCombineLogs(accelerator, use_wandb=False)
    DCL.totals = {
        ("__count__", "a"): torch.tensor([1]),
        ("a",): torch.tensor([5.0]),
    }

    logs = DCL.gather_logs(extra_key=42, other="value")
    assert logs["a"] == 5.0
    assert logs["extra_key"] == 42
    assert logs["other"] == "value"


def test_accum_logs_growth_path_nested_keys():
    """accum_logs growth path: key already in totals, with nested keys."""
    accelerator = Accelerator()
    DCL = DistributeCombineLogs(accelerator, use_wandb=False)

    # First call: initialize totals
    DCL.accum_logs({"nested": {"key": [10.0, 2]}})
    assert ("nested", "key") in DCL.totals
    assert ("__count__", "nested", "key") in DCL.totals

    # Second call: growth path (k in self.totals)
    DCL.accum_logs({"nested": {"key": [20.0, 3]}})
    # 10*2 + 20*3 = 20 + 60 = 80, count 2+3=5
    assert DCL.totals[("nested", "key")].item() == pytest.approx(80.0)
    assert DCL.totals[("__count__", "nested", "key")].item() == pytest.approx(5.0)


def test_key_is_count_false():
    """key_is_count returns False for non-count keys."""
    accelerator = Accelerator()
    DCL = DistributeCombineLogs(accelerator, use_wandb=False)
    assert DCL.key_is_count(("a",)) is False
    assert DCL.key_is_count(("nested", "key")) is False


def test_reset_totals():
    accelerator = Accelerator()
    use_wandb = False

    DCL = DistributeCombineLogs(accelerator, use_wandb)
    DCL.totals = {"asdfghjkl"}
    DCL.reset_logs()

    assert DCL.totals == {}


def test_label_logs():
    logs = "log"
    label = "label"

    labelled = label_logs(logs, label)

    assert labelled == {label: logs}
