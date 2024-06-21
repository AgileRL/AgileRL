import os

import h5py
import minari
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.hosting import download_dataset
from minari.storage.local import load_dataset


def load_minari_dataset(dataset_id, accelerator=None, remote=False):
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
                f"No local Dataset found for dataset id {dataset_id}. check https://minari.farama.org/ for more details on remote dataset. For loading a remote dataset assign remote=True"
            )

    minari_dataset = load_dataset(dataset_id)

    return minari_dataset


def minari_to_agile_buffer(dataset_id, memory, accelerator=None, remote=False):
    minari_dataset = load_minari_dataset(dataset_id, accelerator, remote)

    for episode in minari_dataset.iterate_episodes():
        for num_steps in range(0, len(episode.rewards)):
            observation = episode.observations[num_steps]
            next_observation = episode.observations[num_steps + 1]
            action = episode.actions[num_steps]
            reward = episode.rewards[num_steps]
            terminal = episode.terminations[num_steps]
            memory.save_to_memory(
                observation, action, reward, next_observation, terminal
            )

    return memory


def minari_to_agile_dataset(dataset_id, remote=False):
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []

    minari_dataset = load_minari_dataset(dataset_id, remote)

    for episode in minari_dataset.iterate_episodes():
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
