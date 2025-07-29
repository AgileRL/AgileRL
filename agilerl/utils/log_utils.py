import json

import torch
import wandb
from accelerate import Accelerator
from flatten_dict import flatten, unflatten


class DistributeCombineLogs:
    count_tag = "__count__"

    def __init__(self, accelerator: Accelerator, use_wandb: bool = False) -> None:
        """Initializes the DistributeCombineLogs object.

        :param accelerator: Accelerator object.
        :type accelerator: Accelerator
        :param use_wandb: Whether to use wandb.
        :type use_wandb: bool
        """
        if not isinstance(accelerator, Accelerator):
            raise ValueError("accelerator must be an instance of Accelerator")
        if not isinstance(use_wandb, bool):
            raise ValueError("use_wandb must be a boolean")
        self.totals: dict[tuple[str, ...], torch.Tensor] = {}
        self.accelerator = accelerator
        self.use_wandb = use_wandb

    def convert_key(self, k: tuple[str, ...]) -> tuple[str, ...]:
        """Converts a key to a tuple.

        :param k: Key to convert.
        :type k: tuple

        :return: Converted key.
        :rtype: tuple
        """
        return (self.count_tag,) + k

    def key_is_count(self, k: tuple[str, ...]) -> bool:
        """Checks if a key is a count key.

        :param k: Key to check.
        :type k: tuple

        :return: True if the key is a count key, False otherwise.
        :rtype: bool
        """
        return k[0] == self.count_tag

    def log(self, *postproc_funcs, **additional_items) -> dict:
        """Logs the results.

        :param postproc_funcs: Post-processing functions.
        :type postproc_funcs: list
        :param additional_items: Additional items to log.
        :type additional_items: dict

        :return: Total logs.
        :rtype: dict
        """
        self.accelerator.wait_for_everyone()
        total_logs = self.gather_logs(*postproc_funcs, **additional_items)
        if self.accelerator.is_main_process:
            if self.use_wandb:
                wandb.log(total_logs)
            print(total_logs)
        self.accelerator.wait_for_everyone()
        return total_logs

    def accum_logs(self, logs: dict) -> None:
        """Accumulates the logs.

        :param logs: Logs to accumulate.
        :type logs: dict
        """
        logs = flatten(logs)
        for k, (item, n) in logs.items():
            new_item = torch.tensor([item]).float().to(self.accelerator.device)
            count_item = torch.tensor([n]).float().to(self.accelerator.device)
            if k in self.totals:
                self.totals[k] += new_item * count_item
                self.totals[self.convert_key(k)] += count_item
            else:
                self.totals[k] = new_item * count_item
                self.totals[self.convert_key(k)] = count_item

    def gather_logs(self, *postproc_funcs, **additional_items) -> dict:
        """Gathers the logs.

        :param postproc_funcs: Post-processing functions.
        :type postproc_funcs: list
        :param additional_items: Additional items to log.
        :type additional_items: dict

        :return: Total logs.
        :rtype: dict
        """
        str_totals = {json.dumps(list(k)): v for k, v in self.totals.items()}
        combined_totals = self.accelerator.gather(str_totals)
        combined_totals = {
            tuple(json.loads(k)): v.sum().item() for k, v in combined_totals.items()
        }
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
        final_logs = {**final_logs, **additional_items}
        return final_logs

    def reset_logs(self) -> None:
        """Resets the logs.

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
