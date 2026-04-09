from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import minari
import pytest
import torch
from accelerate import Accelerator
from minari import MinariDataset
from minari.data_collector import EpisodeBuffer
from minari.storage.datasets_root_dir import get_dataset_path
from requests import HTTPError
from requests.exceptions import ReadTimeout

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils import minari_utils


def check_delete_dataset(dataset_id: str) -> None:
    """Test deletion of local Minari datasets.

    :param dataset_id: name of Minari dataset to test
    :type dataset_id: str
    """
    # Check dataset path exists locally.
    dataset_path = Path(get_dataset_path(dataset_id))
    assert dataset_path.exists()

    # delete dataset and check that it's no longer present in local database
    minari.delete_dataset(dataset_id)
    assert not dataset_path.exists()


def create_dataset_return_timesteps(dataset_id: str, env_id: str) -> int:
    buffer = []

    # Delete the test dataset if it already exists.
    if Path(get_dataset_path(dataset_id)).exists():
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    num_episodes = 10
    total_timesteps = 0

    observation, info = env.reset()
    episode_buffer = EpisodeBuffer(observations=observation, infos=info)
    for _episode in range(num_episodes):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            total_timesteps += 1
            observation, reward, terminated, truncated, info = env.step(action)
            episode_buffer = episode_buffer.add_step_data(
                {
                    "observation": observation,
                    "action": action,
                    "reward": reward,
                    "termination": terminated,
                    "truncation": truncated,
                    "info": info,
                },
            )

        buffer.append(episode_buffer)

        observation, _ = env.reset()
        episode_buffer = EpisodeBuffer(observations=observation)

    # Create Minari dataset and store locally
    minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=env,
        buffer=buffer,
        algorithm_name="random_policy",
        author="agile-rl",
        code_permalink="https://github.com/AgileRL/AgileRL",
        description="Random policy data collection for tests",
    )

    return total_timesteps


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [("cartpole/test-v0", "CartPole-v1")],
)
def test_minari_to_agile_dataset(dataset_id: str, env_id: str) -> None:
    """Test create agile dataset from minari dataset."""
    total_timesteps = create_dataset_return_timesteps(dataset_id, env_id)
    dataset = minari_utils.minari_to_agile_dataset(dataset_id)
    assert len(dataset["rewards"][:]) == total_timesteps
    check_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [("cartpole/test-v0", "CartPole-v1")],
)
def test_minari_to_agile_buffer(dataset_id: str, env_id: str) -> None:
    """Test create agile buffer from minari dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory = ReplayBuffer(10000, device=device)

    total_timesteps = create_dataset_return_timesteps(dataset_id, env_id)

    minari_utils.minari_to_agile_buffer(dataset_id, memory)

    assert len(memory) == total_timesteps

    check_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id",
    ["cartpole/test-v0"],
)
def test_load_minari_dataset_errors(dataset_id: str) -> None:
    # test load a dataset absent in local
    with pytest.raises(
        FileNotFoundError,
        match=f"No local Dataset found for dataset id {dataset_id}. "
        "Check https://minari.farama.org/ "
        "for more details on remote datasets. "
        "For loading a remote dataset assign remote=True",
    ):
        minari_utils.load_minari_dataset(dataset_id)

    # test load a dataset absent in remote
    try:
        with pytest.raises(
            KeyError,
            match="Enter a valid remote Minari Dataset ID. "
            "Check https://minari.farama.org/ for more details.",
        ):
            minari_utils.load_minari_dataset(dataset_id, remote=True)
    except ReadTimeout as e:
        pytest.skip(f"Skipping test due to remote dataset not being available: {e}")


@pytest.mark.parametrize(
    "dataset_id",
    ["D4RL/door/human-v2"],
)
def test_load_remote_minari_dataset(dataset_id: str) -> None:
    dataset: MinariDataset | None = None
    try:
        dataset = minari_utils.load_minari_dataset(dataset_id, remote=True)
    except (KeyError, HTTPError, ValueError) as e:
        pytest.skip(f"Skipping test due to remote dataset not being available: {e}")

    assert dataset is not None
    assert isinstance(dataset, MinariDataset)

    check_delete_dataset(dataset_id)


def test_load_minari_dataset_remote_list_raises_fallback():
    """When list_remote_datasets raises, fall back to direct download."""
    dataset_id = "D4RL/door/human-v2"
    mock_ds = type("MinariDataset", (), {"iterate_episodes": lambda self: iter([])})()

    with (
        patch("agilerl.utils.minari_utils.minari.list_remote_datasets") as mock_list,
        patch("agilerl.utils.minari_utils.get_dataset_path") as mock_path,
        patch("agilerl.utils.minari_utils.Path") as mock_path_cls,
        patch("agilerl.utils.minari_utils.download_dataset"),
        patch("agilerl.utils.minari_utils.load_dataset", return_value=mock_ds),
    ):
        mock_list.side_effect = OSError("platform-specific path issue")
        mock_path.return_value = "/nonexistent/path"
        mock_path_cls.return_value.exists.return_value = False

        result = minari_utils.load_minari_dataset(dataset_id, remote=True)
        assert result is mock_ds


def test_load_minari_dataset_remote_no_accelerator_download():
    """Remote download when accelerator is None (else branch)."""
    dataset_id = "D4RL/door/human-v2"
    mock_ds = type("MinariDataset", (), {"iterate_episodes": lambda self: iter([])})()

    with (
        patch("agilerl.utils.minari_utils.minari.list_remote_datasets") as mock_list,
        patch("agilerl.utils.minari_utils.get_dataset_path") as mock_path,
        patch("agilerl.utils.minari_utils.Path") as mock_path_cls,
        patch("agilerl.utils.minari_utils.download_dataset") as mock_dl,
        patch("agilerl.utils.minari_utils.load_dataset", return_value=mock_ds),
    ):
        mock_list.return_value = {dataset_id: None}
        mock_path.return_value = "/nonexistent/path"
        mock_path_cls.return_value.exists.return_value = False

        result = minari_utils.load_minari_dataset(
            dataset_id, accelerator=None, remote=True
        )
        mock_dl.assert_called_once_with(dataset_id)
        assert result is mock_ds


def test_load_minari_dataset_remote_with_accelerator_main_process():
    dataset_id = "D4RL/door/human-v2"
    mock_ds = type("MinariDataset", (), {"iterate_episodes": lambda self: iter([])})()

    with (
        patch("agilerl.utils.minari_utils.minari.list_remote_datasets") as mock_list,
        patch("agilerl.utils.minari_utils.get_dataset_path") as mock_path,
        patch("agilerl.utils.minari_utils.Path") as mock_path_cls,
        patch("agilerl.utils.minari_utils.download_dataset") as mock_dl,
        patch("agilerl.utils.minari_utils.load_dataset", return_value=mock_ds),
    ):
        mock_list.return_value = {dataset_id: None}
        mock_path.return_value = "/nonexistent/path"
        mock_path_cls.return_value.exists.return_value = False

        acc = MagicMock(spec=Accelerator)
        acc.is_main_process = True
        acc.wait_for_everyone = MagicMock()

        result = minari_utils.load_minari_dataset(
            dataset_id, accelerator=acc, remote=True
        )
        mock_dl.assert_called_once_with(dataset_id)
        acc.wait_for_everyone.assert_called()
        assert result is mock_ds


def test_load_minari_dataset_remote_worker_process():
    """Worker process does not call download_dataset; waits for main process."""
    dataset_id = "D4RL/door/human-v2"
    mock_ds = type("MinariDataset", (), {"iterate_episodes": lambda self: iter([])})()

    with (
        patch("agilerl.utils.minari_utils.minari.list_remote_datasets") as mock_list,
        patch("agilerl.utils.minari_utils.get_dataset_path") as mock_path,
        patch("agilerl.utils.minari_utils.Path") as mock_path_cls,
        patch("agilerl.utils.minari_utils.download_dataset") as mock_dl,
        patch("agilerl.utils.minari_utils.load_dataset", return_value=mock_ds),
    ):
        mock_list.return_value = {dataset_id: None}
        mock_path.return_value = "/nonexistent/path"
        mock_path_cls.return_value.exists.return_value = False

        acc = MagicMock(spec=Accelerator)
        acc.is_main_process = False
        acc.wait_for_everyone = MagicMock()

        result = minari_utils.load_minari_dataset(
            dataset_id, accelerator=acc, remote=True
        )
        mock_dl.assert_not_called()
        acc.wait_for_everyone.assert_called()
        assert result is mock_ds


@pytest.mark.parametrize(
    "dataset_id",
    ["D4RL/door/human-v2"],
)
def test_load_remote_minari_dataset_accelerator(dataset_id: str) -> None:
    accelerator = Accelerator()
    dataset: MinariDataset | None = None

    try:
        dataset = minari_utils.load_minari_dataset(
            dataset_id,
            accelerator=accelerator,
            remote=True,
        )
    except (HTTPError, KeyError, ValueError) as e:
        pytest.skip(f"Skipping test due to remote dataset not being available: {e}")

    assert dataset is not None
    assert isinstance(dataset, MinariDataset)

    check_delete_dataset(dataset_id)
