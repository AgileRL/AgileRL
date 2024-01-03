from unittest.mock import patch

import numpy as np
import torch
from accelerate import Accelerator

from agilerl.utils.log_utils import DistributeCombineLogs, label_logs


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

    with patch("agilerl.training.train_off_policy.wandb.log") as mock_wandb_log:

        def dummy_func(a):
            return a

        DCL = DistributeCombineLogs(accelerator, use_wandb)
        _ = DCL.log(dummy_func)

        mock_wandb_log.assert_called()


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
