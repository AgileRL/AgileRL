import json

import torch
import wandb
from flatten_dict import flatten, unflatten


class DistributeCombineLogs:
    count_tag = "__count__"

    def __init__(self, accelerator, use_wandb=False):
        self.totals = {}
        self.accelerator = accelerator
        self.use_wandb = use_wandb

    def convert_key(self, k):
        return (self.count_tag,) + k

    def key_is_count(self, k):
        return k[0] == self.count_tag

    def log(self, *postproc_funcs, **additional_items):
        self.accelerator.wait_for_everyone()
        total_logs = self.gather_logs(*postproc_funcs, **additional_items)
        if self.accelerator.is_main_process:
            if self.use_wandb:
                wandb.log(total_logs)
            print(total_logs)
        self.accelerator.wait_for_everyone()
        return total_logs

    def accum_logs(self, logs):
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

    def gather_logs(self, *postproc_funcs, **additional_items):
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

    def reset_logs(self):
        self.totals = {}


def label_logs(logs, label):
    return {label: logs}
