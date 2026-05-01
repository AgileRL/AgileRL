import numpy as np
import pytest
import torch

from agilerl.components.data import (
    ReplayDataset,
    Transition,
    to_tensordict,
    to_torch_tensor,
)
from agilerl.components.replay_buffer import ReplayBuffer


class TestReplayDatasetInit:
    # The dataset can be initialized with a buffer and batch size
    def test_initialization_with_buffer_and_batch_size(self):
        buffer = ReplayBuffer(
            max_size=1000,
        )
        batch_size = 2
        dataset = ReplayDataset(buffer, batch_size=batch_size)
        assert dataset.buffer == buffer
        assert dataset.batch_size == batch_size

    def test_replay_dataset_batch_size_zero_raises(self):
        buffer = ReplayBuffer(max_size=100)
        with pytest.raises(
            AssertionError, match="Batch size must be greater than zero"
        ):
            ReplayDataset(buffer, batch_size=0)

    def test_replay_dataset_non_replay_buffer_warns(self):
        with pytest.warns(UserWarning, match="Buffer is not an agilerl ReplayBuffer"):
            ReplayDataset(object(), batch_size=8)


class TestReplayDatasetIter:
    # Sampling a batch of experiences from the buffer works correctly
    def test_sampling_batch_from_buffer(self):
        buffer = ReplayBuffer(
            max_size=1000,
        )

        state1 = np.array([1, 2, 3])
        action1 = np.array([0])
        reward1 = np.array([1])
        next_state1 = np.array([4, 5, 6])
        done1 = np.array([False])

        transition1 = Transition(
            obs=state1,
            action=action1,
            reward=reward1,
            next_obs=next_state1,
            done=done1,
        ).to_tensordict()

        transition1 = transition1.unsqueeze(0)
        transition1.batch_size = [1]
        buffer.add(transition1)
        buffer.add(transition1)

        batch_size = 2
        dataset = ReplayDataset(buffer, batch_size=batch_size)
        iterator = iter(dataset)
        batch = next(iterator)

        assert len(batch) == batch_size
        assert len(batch["obs"]) == batch_size
        assert torch.equal(batch["obs"][0], torch.from_numpy(state1).float())
        assert torch.equal(batch["obs"][1], torch.from_numpy(state1).float())
        assert torch.equal(batch["action"][0], torch.from_numpy(action1).float())
        assert torch.equal(batch["action"][1], torch.from_numpy(action1).float())
        assert torch.equal(batch["reward"][0], torch.from_numpy(reward1).float())
        assert torch.equal(batch["reward"][1], torch.from_numpy(reward1).float())
        assert torch.equal(batch["next_obs"][0], torch.from_numpy(next_state1).float())
        assert torch.equal(batch["next_obs"][1], torch.from_numpy(next_state1).float())
        assert torch.equal(batch["done"][0], torch.from_numpy(done1).float())
        assert torch.equal(batch["done"][1], torch.from_numpy(done1).float())


class TestTransitionPostInit:
    def test_transition_tuple_observation_converts_to_tensordict(self):
        transition = Transition(
            obs=(np.array([1.0, 2.0]), 3.0),
            action=1,
            reward=1.0,
            next_obs=(np.array([4.0, 5.0]), 6.0),
            done=False,
        )
        obs_td = transition.obs
        assert "tuple_obs_0" in obs_td.keys()
        assert "tuple_obs_1" in obs_td.keys()

    def test_transition_tuple_observation_invalid_element_raises(self):
        with pytest.raises(AssertionError, match="Expected all elements of the tuple"):
            Transition(
                obs=(np.array([1.0]), object()),
                action=0,
                reward=0.0,
                next_obs=np.array([2.0]),
                done=False,
            )

    def test_transition_dict_observation(self):
        transition = Transition(
            obs={"vec": np.array([1.0, 2.0])},
            action=0,
            reward=1.0,
            next_obs={"vec": np.array([3.0, 4.0])},
            done=False,
        )
        assert "vec" in transition.obs.keys()
        assert "vec" in transition.next_obs.keys()

    def test_transition_scalar_reward_done_unsqueezed(self):
        transition = Transition(
            obs=np.array([1.0]),
            action=0,
            reward=1.0,
            next_obs=np.array([2.0]),
            done=False,
        )
        assert transition.reward.ndim >= 1
        assert transition.done.ndim >= 1


class TestToTensordict:
    def test_to_tensordict_tuple(self):
        td = to_tensordict((np.array([1.0, 2.0]), 3.0))
        assert "tuple_obs_0" in td.keys()
        assert "tuple_obs_1" in td.keys()
        assert td["tuple_obs_1"].dtype == torch.float32

    def test_to_tensordict_dict(self):
        td = to_tensordict({"a": np.array([1.0]), "b": torch.tensor([2.0])})
        assert "a" in td.keys()
        assert "b" in td.keys()
        assert td.dtype == torch.float32

    def test_to_tensordict_dict_invalid_value_raises(self):
        with pytest.raises(AssertionError, match="Expected all values of the dict"):
            to_tensordict({"a": np.array([1.0]), "b": "invalid"})


class TestToTorchTensor:
    def test_to_torch_tensor_ndarray(self):
        t = to_torch_tensor(np.array([1.0, 2.0]))
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32

    def test_to_torch_tensor_number(self):
        t = to_torch_tensor(42)
        assert isinstance(t, torch.Tensor)
        assert t.item() == 42

    def test_to_torch_tensor_bool(self):
        t = to_torch_tensor(True)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32

    def test_to_torch_tensor_tensor(self):
        x = torch.tensor([1.0], dtype=torch.float64)
        t = to_torch_tensor(x)
        assert t.dtype == torch.float32

    def test_to_torch_tensor_other_converts(self):
        t = to_torch_tensor([1, 2, 3])
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)
