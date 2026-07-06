from typing import Any

import torch
import torch.distributed as dist
import wandb
from flatten_dict import flatten, unflatten

from agilerl.utils.distributed import (
    barrier,
    is_distributed,
    is_main_process,
    resolve_device,
)


class DistributeCombineLogs:
    count_tag = "__count__"

    def __init__(
        self,
        device: str | torch.device | None = None,
        use_wandb: bool = False,
    ) -> None:
        """Initialize the DistributeCombineLogs object.

        :param device: Device used to accumulate log tensors; resolved
            automatically when ``None``.
        :type device: str | torch.device | None
        :param use_wandb: Whether to use wandb.
        :type use_wandb: bool
        """
        if not isinstance(use_wandb, bool):
            msg = "use_wandb must be a boolean"
            raise TypeError(msg)
        self.totals: dict[tuple[str, ...], torch.Tensor] = {}
        self.device = torch.device(resolve_device(device))
        self.use_wandb = use_wandb

    def convert_key(self, k: tuple[str, ...]) -> tuple[str, ...]:
        """Convert a key to a tuple.

        :param k: Key to convert.
        :type k: tuple

        :return: Converted key.
        :rtype: tuple
        """
        return (self.count_tag, *k)

    def key_is_count(self, k: tuple[str, ...]) -> bool:
        """Check if a key is a count key.

        :param k: Key to check.
        :type k: tuple

        :return: True if the key is a count key, False otherwise.
        :rtype: bool
        """
        return k[0] == self.count_tag

    def log(self, *postproc_funcs: Any, **additional_items: Any) -> dict:
        """Log the results.

        :param postproc_funcs: Post-processing functions.
        :type postproc_funcs: list
        :param additional_items: Additional items to log.
        :type additional_items: dict

        :return: Total logs.
        :rtype: dict
        """
        barrier()
        total_logs = self.gather_logs(*postproc_funcs, **additional_items)
        if is_main_process() and self.use_wandb:
            wandb.log(total_logs)
        barrier()
        return total_logs

    def accum_logs(self, logs: dict) -> None:
        """Accumulates the logs.

        :param logs: Logs to accumulate.
        :type logs: dict
        """
        logs = flatten(logs)
        for k, (item, n) in logs.items():
            new_item = torch.tensor([item]).float().to(self.device)
            count_item = torch.tensor([n]).float().to(self.device)
            if k in self.totals:
                self.totals[k] += new_item * count_item
                self.totals[self.convert_key(k)] += count_item
            else:
                self.totals[k] = new_item * count_item
                self.totals[self.convert_key(k)] = count_item

    def gather_logs(self, *postproc_funcs: Any, **additional_items: Any) -> dict:
        """Gathers the logs.

        :param postproc_funcs: Post-processing functions.
        :type postproc_funcs: list
        :param additional_items: Additional items to log.
        :type additional_items: dict

        :return: Total logs.
        :rtype: dict
        """
        combined_totals = {}
        for k, v in self.totals.items():
            total = v.clone()
            if is_distributed():
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            combined_totals[k] = total.sum().item()
        final_logs = {}
        for k, v in combined_totals.items():
            if not self.key_is_count(k):
                if combined_totals[self.convert_key(k)] == 0:
                    final_logs[k] = v * float("inf")
                else:
                    final_logs[k] = v / combined_totals[self.convert_key(k)]
        final_logs = unflatten(final_logs)
        for f in postproc_funcs:
            result = f(final_logs)
            if result is not None:
                final_logs = result
        return {**final_logs, **additional_items}

    def reset_logs(self) -> None:
        """Reset the logs.

        :return: Total logs.
        :rtype: dict
        """
        self.totals = {}


def label_logs(logs: dict, label: str) -> dict:
    """Labels the logs.

    :param logs: Logs to label.
    :type logs: dict
    :param label: Label to add.
    :type label: str

    :return: Labeled logs.
    :rtype: dict
    """
    return {label: logs}
