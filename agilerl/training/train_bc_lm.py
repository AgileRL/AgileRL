import json
import os
from collections import deque
from functools import partial

import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from tqdm.auto import tqdm

from agilerl.data.rl_data import Iterable_RL_Dataset
from agilerl.data.torch_datasets import GeneralDataset, GeneralIterDataset
from agilerl.utils.ilql_utils import add_system_configs, convert_path
from agilerl.utils.load_objects import load_item
from agilerl.utils.log_utils import DistributeCombineLogs, label_logs
from agilerl.utils.torch_utils import to


def train(cfg):
    print("using config:", cfg)
    train_cfg = cfg["train"]
    train_cfg["save_checkpoint_dir"] = convert_path(train_cfg["save_checkpoint_dir"])
    train_cfg["optim_state_path"] = convert_path(train_cfg["optim_state_path"])
    wandb_cfg = cfg["wandb"]
    accelerator = Accelerator()
    system_cfg = add_system_configs(cfg, accelerator)
    print("using device:", system_cfg["device"])
    print("num processes:", system_cfg["num_processes"])
    print("using fp16:", system_cfg["use_fp16"])
    if not os.path.exists(train_cfg["save_checkpoint_dir"]):
        os.makedirs(train_cfg["save_checkpoint_dir"])
    with open(os.path.join(train_cfg["save_checkpoint_dir"], "config.json"), "w") as f:
        json.dump(cfg, f)

    if wandb_cfg["use_wandb"]:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            wandb.init(project=wandb_cfg["wandb_project"], config=cfg)
        accelerator.wait_for_everyone()

    raw_dataset_train = load_item(cfg["train_dataset"], system_cfg["device"])
    raw_dataset_eval = load_item(cfg["eval_dataset"], system_cfg["device"])
    if isinstance(raw_dataset_train, Iterable_RL_Dataset):
        dataset_train = GeneralIterDataset(raw_dataset_train, "cpu")
    else:
        dataset_train = GeneralDataset(raw_dataset_train, "cpu")
    if isinstance(raw_dataset_eval, Iterable_RL_Dataset):
        dataset_eval = GeneralIterDataset(raw_dataset_eval, "cpu")
    else:
        dataset_eval = GeneralDataset(raw_dataset_eval, "cpu")
    train_data_loader_kwargs = {
        "num_workers": train_cfg["dataloader_workers"],
        "batch_size": train_cfg["bsize"],
        "collate_fn": dataset_train.collate,
    }
    eval_data_loader_kwargs = {
        "num_workers": train_cfg["dataloader_workers"],
        "batch_size": train_cfg["eval_bsize"],
        "collate_fn": dataset_eval.collate,
    }
    if not isinstance(dataset_train, IterableDataset):
        train_data_loader_kwargs["shuffle"] = True
    if not isinstance(dataset_eval, IterableDataset):
        eval_data_loader_kwargs["shuffle"] = True
    data_loader = DataLoader(dataset_train, **train_data_loader_kwargs)
    eval_data_loader = DataLoader(dataset_eval, **eval_data_loader_kwargs)

    evaluator = None
    if cfg["evaluator"] is not None:
        evaluator = load_item(cfg["evaluator"], system_cfg["device"])

    model = load_item(cfg["model"], system_cfg["device"])
    model.train()

    if hasattr(model, "param_groups"):
        params = [
            {
                "params": frozenset().union(*list(map(lambda x: x.parameters(), p))),
                **f(train_cfg),
            }
            for p, f in model.param_groups
        ]
    else:
        params = model.parameters()
    optim = torch.optim.AdamW(
        params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    if train_cfg["optim_state_path"] is not None and os.path.exists(
        train_cfg["optim_state_path"]
    ):
        print(f'loading optimizer state from: {train_cfg["optim_state_path"]}')
        optim.load_state_dict(
            torch.load(train_cfg["optim_state_path"], map_location=system_cfg["device"])
        )
        print("loaded.")
    if isinstance(dataset_train, IterableDataset) and isinstance(
        dataset_eval, IterableDataset
    ):
        model, optim = accelerator.prepare(model, optim)
    elif isinstance(dataset_train, IterableDataset):
        model, optim, eval_data_loader = accelerator.prepare(
            model, optim, eval_data_loader
        )
    elif isinstance(dataset_eval, IterableDataset):
        model, optim, data_loader = accelerator.prepare(model, optim, data_loader)
    else:
        model, optim, data_loader, eval_data_loader = accelerator.prepare(
            model, optim, data_loader, eval_data_loader
        )

    train_logs = DistributeCombineLogs(accelerator, use_wandb=wandb_cfg["use_wandb"])
    eval_logs = DistributeCombineLogs(accelerator, use_wandb=wandb_cfg["use_wandb"])
    step = 0
    best_loss = float("inf")
    saved_checkpoints = deque([])
    for epoch in tqdm(
        range(train_cfg["epochs"]), disable=not accelerator.is_local_main_process
    ):
        for items in tqdm(data_loader, disable=not accelerator.is_local_main_process):
            items = to(items, system_cfg["device"])
            loss, logs, postproc_fs = accelerator.unwrap_model(model).get_loss(
                items, **train_cfg["loss"]
            )
            accelerator.backward(loss / train_cfg["grad_accum_steps"])
            train_logs.accum_logs(logs)
            if (step + 1) % train_cfg["grad_accum_steps"] == 0:
                optim.step()
                optim.zero_grad()
            if (step + 1) % train_cfg["log_every"] == 0:
                train_logs.log(
                    *postproc_fs,
                    partial(label_logs, label="train"),
                    iteration=step,
                    epoch=epoch,
                )
            if (step + 1) % train_cfg["grad_accum_steps"] == 0:
                train_logs.reset_logs()
            if (step + 1) % train_cfg["eval_every"] == 0:
                model.eval()
                eval_logs.reset_logs()
                with torch.no_grad():
                    for i, eval_items in enumerate(eval_data_loader):
                        eval_items = to(eval_items, system_cfg["device"])
                        if i >= train_cfg["eval_batches"]:
                            break
                        _, logs, postproc_fs = accelerator.unwrap_model(model).get_loss(
                            eval_items, **train_cfg["loss"]
                        )
                        if evaluator is not None:
                            evaluator_logs = evaluator.evaluate(
                                accelerator.unwrap_model(model), eval_items
                            )
                            if evaluator_logs is not None:
                                logs["evaluation"] = evaluator_logs
                        eval_logs.accum_logs(logs)
                eval_label = "eval"
                eval_total_logs = eval_logs.log(
                    *postproc_fs,
                    partial(label_logs, label=eval_label),
                    iteration=step,
                    epoch=epoch,
                )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if eval_total_logs[eval_label]["loss"] < best_loss:
                        print("new best eval loss! Saving ...")
                        if not os.path.exists(train_cfg["save_checkpoint_dir"]):
                            os.makedirs(train_cfg["save_checkpoint_dir"])
                        torch.save(
                            accelerator.unwrap_model(model).state_dict(),
                            os.path.join(train_cfg["save_checkpoint_dir"], "model.pkl"),
                        )
                        torch.save(
                            optim.state_dict(),
                            os.path.join(train_cfg["save_checkpoint_dir"], "optim.pkl"),
                        )
                        print("saved.")
                        best_loss = eval_total_logs[eval_label]["loss"]
                accelerator.wait_for_everyone()
                model.train()
            if (
                train_cfg["save_every"] is not None
                and (step + 1) % train_cfg["save_every"] == 0
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print("saving checkpoint...")
                    if not os.path.exists(train_cfg["save_checkpoint_dir"]):
                        os.makedirs(train_cfg["save_checkpoint_dir"])
                    if (train_cfg["max_checkpoints"] is not None) and (
                        len(saved_checkpoints) >= train_cfg["max_checkpoints"]
                    ):
                        os.system("rm -rf %s" % (saved_checkpoints.popleft()))
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        os.path.join(
                            train_cfg["save_checkpoint_dir"], "model_%d.pkl" % (step)
                        ),
                    )
                    saved_checkpoints.append(
                        os.path.join(
                            train_cfg["save_checkpoint_dir"], "model_%d.pkl" % (step)
                        )
                    )
                    print("saved.")
                accelerator.wait_for_everyone()
            step += 1
            if train_cfg["max_steps"] is not None and step >= train_cfg["max_steps"]:
                return
