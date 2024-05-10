import copy

import gymnasium as gym
import minari
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from minari import MinariDataset

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils import minari_utils


def check_delete_dataset(dataset_id: str):
    """Test deletion of local Minari datasets.

    Args:
        dataset_id (str): name of Minari dataset to test
    """
    # check dataset name is present in local database
    local_datasets = minari.list_local_datasets()
    assert dataset_id in local_datasets

    # delete dataset and check that it's no longer present in local database
    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets


def create_dataset_return_timesteps(dataset_id, env_id):
    buffer = []

    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []

    num_episodes = 10

    observation, info = env.reset(seed=42)
    total_timesteps = 0
    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    observation, _ = env.reset()
    observations.append(observation)
    for episode in range(num_episodes):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            total_timesteps += 1
            observation, reward, terminated, truncated, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)

        episode_buffer = {
            "observations": copy.deepcopy(observations),
            "actions": copy.deepcopy(actions),
            "rewards": np.asarray(rewards),
            "terminations": np.asarray(terminations),
            "truncations": np.asarray(truncations),
        }
        buffer.append(episode_buffer)

        observations.clear()
        actions.clear()
        rewards.clear()
        terminations.clear()
        truncations.clear()

        observation, _ = env.reset()
        observations.append(observation)

    # Create Minari dataset and store locally
    minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=env,
        buffer=buffer,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    return total_timesteps


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [("cartpole-test-v0", "CartPole-v1")],
)
def test_minari_to_agile_dataset(dataset_id, env_id):
    """Test create agile dataset from minari dataset."""

    total_timesteps = create_dataset_return_timesteps(dataset_id, env_id)

    dataset = minari_utils.MinariToAgileDataset(dataset_id)

    assert len(dataset["rewards"][:]) == total_timesteps

    check_delete_dataset(dataset_id)
    agile_dataset_id = dataset_id.split("-")
    agile_dataset_id[0] = agile_dataset_id[0] + "_agile"
    agile_dataset_id = "-".join(agile_dataset_id)
    check_delete_dataset(agile_dataset_id)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [("cartpole-test-v0", "CartPole-v1")],
)
def test_minari_to_agile_buffer(dataset_id, env_id):
    """Test create agile buffer from minari dataset."""

    action_dim = 2
    field_names = ["state", "action", "reward", "next_state", "done"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory = ReplayBuffer(action_dim, 10000, field_names=field_names, device=device)

    total_timesteps = create_dataset_return_timesteps(dataset_id, env_id)

    minari_utils.MinariToAgileBuffer(dataset_id, memory)

    assert len(memory) == total_timesteps

    check_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id",
    [("cartpole-test-v0")],
)
def test_load_minari_dataset_errors(dataset_id):
    # test load a dataset absent in local
    with pytest.raises(
        FileNotFoundError,
        match=f"No local Dataset found for dataset id {dataset_id}. check https://minari.farama.org/ for more details on remote dataset. For loading a remote dataset assign remote=True",
    ):
        minari_utils.load_minari_dataset(dataset_id)

    # test load a dataset absent in remote
    with pytest.raises(
        KeyError,
        match="Enter a valid remote Minari Dataset ID. check https://minari.farama.org/ for more details.",
    ):
        minari_utils.load_minari_dataset(dataset_id, remote=True)


@pytest.mark.parametrize(
    "dataset_id",
    [("door-human-v2")],
)
def test_load_remote_minari_dataset(dataset_id):
    dataset = minari_utils.load_minari_dataset(dataset_id, remote=True)

    assert isinstance(dataset, MinariDataset)

    check_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id",
    [("door-human-v2")],
)
def test_load_remote_minari_dataset_accelerator(dataset_id):
    accelerator = Accelerator()

    dataset = minari_utils.load_minari_dataset(
        dataset_id, accelerator=accelerator, remote=True
    )

    assert isinstance(dataset, MinariDataset)

    check_delete_dataset(dataset_id)
