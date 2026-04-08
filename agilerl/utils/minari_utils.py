from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import minari
import torch
from accelerate import Accelerator
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.hosting import download_dataset
from minari.storage.local import load_dataset

from agilerl.components.data import Transition
from agilerl.components.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from tensordict import TensorDictBase


class MinariDatasetNotFoundError(KeyError):
    """Raised when a Minari dataset is not found."""

    pass


def load_minari_dataset(
    dataset_id: str,
    accelerator: Accelerator | None = None,
    remote: bool = False,
) -> minari.MinariDataset:
    """Load a Minari dataset either from local storage or remote repository.

    :param dataset_id: The ID of the Minari dataset to load
    :type dataset_id: str
    :param accelerator: Optional accelerator for distributed training
    :type accelerator: Accelerator | None
    :param remote: Whether to load from remote repository. Defaults to False.
    :return: The loaded Minari dataset
    :raises KeyError: If remote=True and dataset_id is not a valid remote dataset
    :raises FileNotFoundError: If remote=False and dataset not found locally
    """
    remote_dataset_error = (
        "Enter a valid remote Minari Dataset ID. "
        "Check https://minari.farama.org/ for more details."
    )
    if remote:
        try:
            remote_datasets = minari.list_remote_datasets()
        except Exception:
            # Some Minari/HF combinations fail while listing all remote datasets
            # (e.g. platform-specific path issues). In that case, fall back to
            # direct download validation below.
            remote_datasets = None

        if remote_datasets is not None and dataset_id not in list(
            remote_datasets.keys()
        ):
            raise MinariDatasetNotFoundError(remote_dataset_error)

    file_path = get_dataset_path(dataset_id)

    if not Path(file_path).exists():
        if remote:
            try:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        download_dataset(dataset_id)
                    accelerator.wait_for_everyone()
                else:
                    download_dataset(dataset_id)
            except Exception as err:
                raise KeyError(remote_dataset_error) from err
        else:
            msg = (
                f"No local Dataset found for dataset id {dataset_id}. "
                "Check https://minari.farama.org/ for more details on remote datasets. "
                "For loading a remote dataset assign remote=True"
            )
            raise FileNotFoundError(
                msg,
            )

    return load_dataset(dataset_id)


def minari_to_agile_buffer(
    dataset_id: str,
    memory: ReplayBuffer,
    accelerator: Accelerator | None = None,
    remote: bool = False,
) -> ReplayBuffer:
    """Convert a Minari dataset to an agile buffer.

    :param dataset_id: The ID of the Minari dataset to load
    :type dataset_id: str
    :param memory: The memory to save the dataset to
    :type memory: ReplayBuffer
    :param accelerator: Optional accelerator for distributed training
    :type accelerator: Accelerator | None
    :param remote: Whether to load from remote repository. Defaults to False.
    :type remote: bool
    :return: The loaded Minari dataset
    :rtype: ReplayBuffer
    :raises MinariDatasetNotFoundError: If the dataset is not found
    """
    minari_dataset = load_minari_dataset(dataset_id, accelerator, remote)
    for episode in minari_dataset.iterate_episodes():
        for num_steps in range(len(episode.rewards)):
            # Get the observation, next observation, action, reward, and terminal at the current step
            observation = episode.observations[num_steps]
            next_observation = episode.observations[num_steps + 1]
            action = episode.actions[num_steps]
            reward = episode.rewards[num_steps]
            terminal = episode.terminations[num_steps]

            # Create a TensorDict for the transition
            transition: TensorDictBase = Transition(
                obs=torch.tensor(observation),
                action=torch.tensor(action),
                reward=torch.tensor(reward),
                next_obs=torch.tensor(next_observation),
                done=torch.tensor(terminal),
            ).to_tensordict()

            # Add the transition to the memory
            transition = transition.unsqueeze(0)
            transition.batch_size = [1]
            memory.add(transition)

    return memory


def minari_to_agile_dataset(dataset_id: str, remote: bool = False) -> h5py.File:
    """Convert a Minari dataset to an agile dataset.

    :param dataset_id: The ID of the Minari dataset to load
    :type dataset_id: str
    :param remote: Whether to load from remote repository. Defaults to False.
    :type remote: bool
    :return: The loaded Minari dataset
    :rtype: h5py.File
    :raises MinariDatasetNotFoundError: If the dataset is not found
    """
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []

    # Load the Minari dataset
    dataset = load_minari_dataset(dataset_id, remote)

    # Iterate through the episodes in the dataset
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

    agile_dataset_path = Path(agile_file_path) / "data"
    agile_dataset_path.mkdir(parents=True, exist_ok=True)
    data_path = agile_dataset_path / "main_data.hdf5"

    # with h5py.File(os.path.join(agile_file_path, "data", "main_data.hdf5"), "w") as f:
    f = h5py.File(data_path, "w")

    f.create_dataset("observations", data=observations)
    f.create_dataset("next_observations", data=next_observations)
    f.create_dataset("actions", data=actions)
    f.create_dataset("rewards", data=rewards)
    f.create_dataset("terminals", data=terminals)

    return f
