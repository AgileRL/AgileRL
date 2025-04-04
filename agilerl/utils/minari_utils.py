import os
from typing import Optional

import h5py
import minari
import torch
from accelerate import Accelerator
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.hosting import download_dataset
from minari.storage.local import load_dataset

from agilerl.components.data import Transition
from agilerl.components.replay_buffer import ReplayBuffer


def load_minari_dataset(
    dataset_id: str, accelerator: Optional[Accelerator] = None, remote: bool = False
) -> minari.MinariDataset:
    """Load a Minari dataset either from local storage or remote repository.

    :param dataset_id: The ID of the Minari dataset to load
    :param accelerator: Optional accelerator for distributed training
    :param remote: Whether to load from remote repository. Defaults to False.
    :return: The loaded Minari dataset
    :raises KeyError: If remote=True and dataset_id is not a valid remote dataset
    :raises FileNotFoundError: If remote=False and dataset not found locally
    """
    if remote:
        if dataset_id not in list(minari.list_remote_datasets().keys()):
            raise KeyError(
                "Enter a valid remote Minari Dataset ID. check https://minari.farama.org/ for more details."
            )

    file_path = get_dataset_path(dataset_id)

    if not os.path.exists(file_path):
        if remote:
            if accelerator is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print("download dataset: ", dataset_id)
                    download_dataset(dataset_id)
                accelerator.wait_for_everyone()
            else:
                print("download dataset: ", dataset_id)
                download_dataset(dataset_id)
        else:
            raise FileNotFoundError(
                f"No local Dataset found for dataset id {dataset_id}. check https://minari.farama.org/ for "
                "more details on remote dataset. For loading a remote dataset assign remote=True"
            )

    minari_dataset = load_dataset(dataset_id)

    return minari_dataset


def minari_to_agile_buffer(
    dataset_id: str,
    memory: ReplayBuffer,
    accelerator: Optional[Accelerator] = None,
    remote: bool = False,
):
    """Convert a Minari dataset to an agile buffer.

    :param dataset_id: The ID of the Minari dataset to load
    :param memory: The memory to save the dataset to
    :param accelerator: Optional accelerator for distributed training
    :param remote: Whether to load from remote repository. Defaults to False.
    :return: The loaded Minari dataset
    """
    minari_dataset = load_minari_dataset(dataset_id, accelerator, remote)
    for episode in minari_dataset.iterate_episodes():
        for num_steps in range(0, len(episode.rewards)):
            observation = episode.observations[num_steps]
            next_observation = episode.observations[num_steps + 1]
            action = episode.actions[num_steps]
            reward = episode.rewards[num_steps]
            terminal = episode.terminations[num_steps]

            transition = Transition(
                obs=torch.tensor(observation),
                action=torch.tensor(action),
                reward=torch.tensor(reward),
                next_obs=torch.tensor(next_observation),
                done=torch.tensor(terminal),
            ).to_tensordict()
            transition = transition.unsqueeze(0)
            transition.batch_size = [1]
            memory.add(transition)

    return memory


def minari_to_agile_dataset(dataset_id: str, remote: bool = False) -> h5py.File:
    """Convert a Minari dataset to an agile dataset.

    :param dataset_id: The ID of the Minari dataset to load
    :param remote: Whether to load from remote repository. Defaults to False.
    :return: The loaded Minari dataset
    """
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []

    dataset = load_minari_dataset(dataset_id, remote)

    for episode in dataset.iterate_episodes():
        observations.extend(episode.observations[:-1])
        next_observations.extend(episode.observations[1:])
        actions.extend(episode.actions[:])
        rewards.extend(episode.rewards[:])
        terminals.extend(episode.terminations[:])

    agile_dataset_id = dataset_id.split("-")
    agile_dataset_id[0] = agile_dataset_id[0] + "_agile"
    agile_dataset_id = "-".join(agile_dataset_id)

    agile_file_path = get_dataset_path(agile_dataset_id)

    agile_dataset_path = os.path.join(agile_file_path, "data")
    os.makedirs(agile_dataset_path, exist_ok=True)
    data_path = os.path.join(agile_dataset_path, "main_data.hdf5")

    # with h5py.File(os.path.join(agile_file_path, "data", "main_data.hdf5"), "w") as f:
    f = h5py.File(data_path, "w")

    f.create_dataset("observations", data=observations)
    f.create_dataset("next_observations", data=next_observations)
    f.create_dataset("actions", data=actions)
    f.create_dataset("rewards", data=rewards)
    f.create_dataset("terminals", data=terminals)

    return f
