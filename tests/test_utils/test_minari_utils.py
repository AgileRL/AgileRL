import gymnasium as gym
import minari
import pytest
import torch
from accelerate import Accelerator
from minari import MinariDataset
from minari.data_collector import EpisodeBuffer
from requests import HTTPError
from requests.exceptions import ReadTimeout

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils import minari_utils


def check_delete_dataset(dataset_id: str) -> None:
    """Test deletion of local Minari datasets.

    :param dataset_id: name of Minari dataset to test
    :type dataset_id: str
    """
    # check dataset name is present in local database
    local_datasets = minari.list_local_datasets()
    assert dataset_id in local_datasets

    # delete dataset and check that it's no longer present in local database
    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets


def create_dataset_return_timesteps(dataset_id: str, env_id: str) -> int:
    buffer = []

    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    num_episodes = 10
    total_timesteps = 0

    observation, info = env.reset()
    episode_buffer = EpisodeBuffer(observations=observation, infos=info)
    for episode in range(num_episodes):
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
                }
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
        match=f"No local Dataset found for dataset id {dataset_id}. check https://minari.farama.org/ "
        "for more details on remote dataset. For loading a remote dataset assign remote=True",
    ):
        minari_utils.load_minari_dataset(dataset_id)

    # test load a dataset absent in remote
    try:
        with pytest.raises(
            KeyError,
            match="Enter a valid remote Minari Dataset ID. check https://minari.farama.org/ "
            "for more details.",
        ):
            minari_utils.load_minari_dataset(dataset_id, remote=True)
    except ReadTimeout as e:
        pytest.skip(f"Skipping test due to remote dataset not being available: {e}")


@pytest.mark.parametrize(
    "dataset_id",
    ["D4RL/door/human-v2"],
)
def test_load_remote_minari_dataset(dataset_id: str) -> None:
    try:
        dataset = minari_utils.load_minari_dataset(dataset_id, remote=True)
    except HTTPError as e:
        pytest.skip(f"Skipping test due to remote dataset not being available: {e}")

    assert isinstance(dataset, MinariDataset)

    check_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id",
    ["D4RL/door/human-v2"],
)
def test_load_remote_minari_dataset_accelerator(dataset_id: str) -> None:
    accelerator = Accelerator()

    try:
        dataset = minari_utils.load_minari_dataset(
            dataset_id, accelerator=accelerator, remote=True
        )
    except HTTPError as e:
        pytest.skip(f"Skipping test due to remote dataset not being available: {e}")

    assert isinstance(dataset, MinariDataset)

    check_delete_dataset(dataset_id)
