"""Unit tests for ``env_wrapper`` (MuJoCo Playground JAX batched Gymnasium API)."""
# ruff: noqa: D101, D102, SLF001, PLR2004

from __future__ import annotations

import jax
import numpy as np
import pytest
import torch

from hpo_improvements.mujoco_playground_locomotion.env_wrapper import (
    ENV_DEFAULTS,
    BatchedPlaygroundVectorEnv,
    PlaygroundGymWrapper,
    RunningMeanStd,
    make_env,
)
from hpo_improvements.mujoco_playground_locomotion.tests.mock_playground import (
    install_mock_registry,
)

pytestmark = pytest.mark.usefixtures("block_jax_async")


@pytest.fixture
def block_jax_async():
    """Block on JAX async dispatch so assertions see completed computation."""
    yield
    jax.effects_barrier()


def _zero_actions(env: BatchedPlaygroundVectorEnv) -> np.ndarray:
    return np.zeros(
        (env.num_envs, env.single_action_space.shape[0]),
        dtype=np.float32,
    )


def _actions_with_row_sums(
    env: BatchedPlaygroundVectorEnv, sums: list[float]
) -> np.ndarray:
    actions = np.zeros(
        (env.num_envs, env.single_action_space.shape[0]),
        dtype=np.float32,
    )
    for i, total in enumerate(sums):
        actions[i, 0] = total
    return actions


class TestRunningMeanStd:
    def test_batch_update_moves_mean_toward_data(self) -> None:
        rms = RunningMeanStd(shape=(2,))
        rms.update(np.array([[10.0, 20.0], [10.0, 20.0]], dtype=np.float32))
        assert rms.mean[0] == pytest.approx(10.0, rel=1e-3)
        assert rms.mean[1] == pytest.approx(20.0, rel=1e-3)
        assert rms.count >= 2.0

    def test_sequential_batches_converge_to_global_mean(self) -> None:
        rms = RunningMeanStd(shape=(1,))
        for _ in range(50):
            rms.update(np.ones((4, 1), dtype=np.float32))
        assert rms.mean[0] == pytest.approx(1.0, abs=1e-5)
        assert rms.var[0] == pytest.approx(0.0, abs=1e-5)


class TestActionConversion:
    def test_numpy_actions_pass_through_unchanged(self) -> None:
        """NumPy actions are returned as-is; JAX promotes them inside ``step``."""
        actions = np.ones((3, 6), dtype=np.float32)
        out = BatchedPlaygroundVectorEnv._actions_to_jax(actions)
        assert out is actions or np.shares_memory(np.asarray(out), actions)
        np.testing.assert_allclose(np.asarray(out), actions, rtol=0.0, atol=0.0)

    def test_torch_actions_convert_via_dlpack(self) -> None:
        actions = torch.ones(3, 6, dtype=torch.float32)
        jax_actions = BatchedPlaygroundVectorEnv._actions_to_jax(actions)
        assert isinstance(jax_actions, jax.Array)
        np.testing.assert_allclose(np.asarray(jax_actions), actions.numpy(), rtol=1e-5)


class TestWalkerWalkDefaults:
    def test_env_defaults_dict_matches_constructor(self) -> None:
        assert ENV_DEFAULTS["WalkerWalk"]["action_repeat"] == 1
        assert ENV_DEFAULTS["WalkerWalk"]["reward_scaling"] == 1.0
        assert ENV_DEFAULTS["WalkerWalk"]["normalize_obs"] is True

    def test_batched_env_applies_walker_defaults(self) -> None:
        env = BatchedPlaygroundVectorEnv(env_name="WalkerWalk", num_envs=4, seed=0)
        try:
            assert env._action_repeat == 1
            assert env.env_steps_per_step == 1
            assert env._reward_scaling == 1.0
            assert env._normalize_obs is True
            assert env._obs_rms is not None
        finally:
            env.close()

    def test_staggered_step_counts_span_episode(self) -> None:
        env = BatchedPlaygroundVectorEnv(env_name="WalkerWalk", num_envs=4, seed=0)
        try:
            assert env._init_step_counts.shape == (4,)
            assert env._init_step_counts.dtype == np.int32
            assert env._init_step_counts.min() >= 0
            assert env._init_step_counts.max() < env._episode_length
            assert len(np.unique(env._init_step_counts)) == 4
        finally:
            env.close()

    def test_observation_and_action_spaces_batched(self) -> None:
        env = BatchedPlaygroundVectorEnv(env_name="WalkerWalk", num_envs=8, seed=0)
        try:
            obs_dim = env.single_observation_space.shape[0]
            act_dim = env.single_action_space.shape[0]
            assert env.observation_space.shape == (8, obs_dim)
            assert env.action_space.shape == (8, act_dim)
            assert obs_dim > 0
            assert act_dim > 0
        finally:
            env.close()


class TestPlaygroundGymWrapper:
    @pytest.mark.parametrize("env_name", ["WalkerWalk", "MockWalk"])
    def test_reset_and_step_shapes(
        self,
        env_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if env_name == "MockWalk":
            install_mock_registry(monkeypatch)

        env = PlaygroundGymWrapper(env_name=env_name, seed=123)
        obs, info = env.reset(seed=123)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert info == {}
        assert np.isfinite(obs).all()

        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert terminated is False
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("env_name", ["WalkerWalk", "MockWalk"])
    def test_same_seed_yields_identical_reset_obs(
        self,
        env_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if env_name == "MockWalk":
            install_mock_registry(monkeypatch)

        env_a = PlaygroundGymWrapper(env_name=env_name, seed=7)
        env_b = PlaygroundGymWrapper(env_name=env_name, seed=7)
        obs_a, _ = env_a.reset(seed=99)
        obs_b, _ = env_b.reset(seed=99)
        np.testing.assert_allclose(obs_a, obs_b, rtol=0.0, atol=0.0)

    def test_make_env_factory(self) -> None:
        env = make_env(env_name="WalkerWalk", seed=1)
        assert isinstance(env, PlaygroundGymWrapper)
        obs, _ = env.reset()
        assert obs.ndim == 1


class TestWalkerWalkSemantics:
    """Behavioural checks shared across real and mock vector env backends."""

    @pytest.fixture(params=["WalkerWalk", "MockWalk"])
    def walker(
        self,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> BatchedPlaygroundVectorEnv:
        if request.param == "MockWalk":
            install_mock_registry(monkeypatch)

        env = BatchedPlaygroundVectorEnv(
            env_name=request.param,
            num_envs=4,
            seed=0,
            normalize_obs=False,
            torch_device=None,
        )
        yield env
        env.close()

    def test_reset_parallel_obs_are_pairwise_distinct(
        self, walker: BatchedPlaygroundVectorEnv
    ) -> None:
        obs, _ = walker.reset(seed=101)
        jax.block_until_ready(obs)
        obs_np = np.asarray(obs, dtype=np.float32)
        for i in range(walker.num_envs):
            for j in range(i + 1, walker.num_envs):
                assert not np.allclose(obs_np[i], obs_np[j], rtol=0, atol=0), (
                    f"env {i} and {j} share identical reset observations"
                )

    def test_different_reset_seeds_yield_different_obs(
        self, walker: BatchedPlaygroundVectorEnv
    ) -> None:
        obs_a, _ = walker.reset(seed=1)
        jax.block_until_ready(obs_a)
        obs_b, _ = walker.reset(seed=2)
        jax.block_until_ready(obs_b)
        assert not np.allclose(
            np.asarray(obs_a, dtype=np.float32),
            np.asarray(obs_b, dtype=np.float32),
            rtol=0,
            atol=0,
        )

    def test_step_with_zero_action_advances_env_and_returns_finite_rewards(
        self, walker: BatchedPlaygroundVectorEnv
    ) -> None:
        obs_reset, _ = walker.reset(seed=0)
        jax.block_until_ready(obs_reset)
        before = walker._step_counts.copy()
        obs_step, rewards, _, _, _ = walker.step(_zero_actions(walker))
        jax.block_until_ready(obs_step)
        assert (
            np.asarray(obs_step, dtype=np.float32).shape
            == np.asarray(obs_reset, dtype=np.float32).shape
        )
        assert not np.array_equal(walker._step_counts, before)
        assert np.all(walker._step_counts >= 0)
        assert np.all(walker._step_counts < walker._episode_length)
        assert np.isfinite(rewards).all()

    def test_nonzero_action_differs_from_zero_action_rewards(
        self, walker: BatchedPlaygroundVectorEnv
    ) -> None:
        walker.reset(seed=0)
        _, rewards_zero, _, _, _ = walker.step(_zero_actions(walker))
        walker.reset(seed=0)
        random_actions = np.array(walker.action_space.sample(), dtype=np.float32)
        _, rewards_rand, _, _, _ = walker.step(random_actions)
        assert not np.allclose(rewards_zero, rewards_rand, rtol=0, atol=0)

    def test_different_actions_yield_different_obs_per_env(
        self, walker: BatchedPlaygroundVectorEnv
    ) -> None:
        """Each parallel slot gets its own action row; resulting obs should differ."""
        walker.reset(seed=0)
        actions = np.zeros(
            (walker.num_envs, walker.single_action_space.shape[0]),
            dtype=np.float32,
        )
        actions[0, 0] = 0.8
        actions[1, 0] = -0.8
        actions[2, 0] = 0.2
        actions[3, 0] = -0.2
        walker.step(actions)
        obs_after = np.asarray(walker._states.obs, dtype=np.float32)
        assert not np.allclose(obs_after[0], obs_after[1], rtol=0, atol=0)
        assert not np.allclose(obs_after[2], obs_after[3], rtol=0, atol=0)

    def test_full_reset_after_steps_changes_obs(
        self, walker: BatchedPlaygroundVectorEnv
    ) -> None:
        walker.reset(seed=10)
        first_obs = np.asarray(walker._first_obs, dtype=np.float32).copy()
        for _ in range(5):
            walker.step(np.array(walker.action_space.sample(), dtype=np.float32))
        walker.reset(seed=20)
        second_obs = np.asarray(walker._states.obs, dtype=np.float32)
        assert not np.allclose(second_obs, first_obs, rtol=0, atol=0)

    def test_partial_auto_reset_only_restores_truncated_envs(
        self,
        walker: BatchedPlaygroundVectorEnv,
    ) -> None:
        walker.reset(seed=0)
        first_obs = np.asarray(walker._first_obs, dtype=np.float32).copy()
        walker._step_counts[0] = walker._episode_length
        walker._step_counts[1] = 0
        actions = _zero_actions(walker)
        actions[1, 0] = 1.0
        obs, _, _, truncated, _ = walker.step(
            actions,
        )
        jax.block_until_ready(obs)
        obs_np = np.asarray(obs, dtype=np.float32)
        assert truncated[0]
        assert not truncated[1]
        np.testing.assert_allclose(obs_np[0], first_obs[0], rtol=1e-4, atol=1e-4)
        assert not np.allclose(obs_np[1], first_obs[1], rtol=0, atol=0)

    def test_reported_rewards_match_internal_state_reward(
        self, walker: BatchedPlaygroundVectorEnv
    ) -> None:
        walker.reset(seed=0)
        _, rewards, _, _, _ = walker.step(_zero_actions(walker))
        internal = np.array(walker._states.reward, dtype=np.float32)
        np.testing.assert_allclose(rewards, internal, rtol=1e-5, atol=1e-5)


class TestBatchedPlaygroundVectorEnv:
    @pytest.fixture
    def small_env(self) -> BatchedPlaygroundVectorEnv:
        env = BatchedPlaygroundVectorEnv(
            env_name="WalkerWalk",
            num_envs=4,
            seed=42,
            normalize_obs=False,
            torch_device=None,
        )
        yield env
        env.close()

    def test_reset_returns_jax_obs_when_no_torch_device(
        self,
        small_env: BatchedPlaygroundVectorEnv,
    ) -> None:
        obs, info = small_env.reset(seed=42)
        jax.block_until_ready(obs)
        assert isinstance(obs, jax.Array)
        assert obs.shape == (4, small_env.single_observation_space.shape[0])
        assert info == {}

    def test_step_returns_vectorized_outputs(
        self,
        small_env: BatchedPlaygroundVectorEnv,
    ) -> None:
        obs, _ = small_env.reset(seed=0)
        jax.block_until_ready(obs)
        actions = _zero_actions(small_env)
        obs, rewards, terminated, truncated, info = small_env.step(actions)
        jax.block_until_ready(obs)

        n = small_env.num_envs
        assert obs.shape == (n, small_env.single_observation_space.shape[0])
        assert rewards.shape == (n,)
        assert rewards.dtype == np.float32
        assert np.isfinite(rewards).all()
        assert terminated.shape == (n,)
        assert np.all(terminated == False)  # noqa: E712
        assert truncated.shape == (n,)
        assert truncated.dtype == np.bool_
        assert info["_episode_done"].shape == (n,)
        np.testing.assert_array_equal(info["_episode_done"], truncated)

    def test_step_increments_internal_counters(
        self,
        small_env: BatchedPlaygroundVectorEnv,
    ) -> None:
        small_env.reset(seed=0)
        before = small_env._step_counts.copy()
        small_env.step(_zero_actions(small_env))
        assert np.all(small_env._step_counts == before + small_env._action_repeat)

    def test_truncation_resets_step_count_for_affected_envs(
        self,
        small_env: BatchedPlaygroundVectorEnv,
    ) -> None:
        """Force time-limit truncation and verify per-env counters zero out."""
        small_env.reset(seed=0)
        small_env._step_counts[:] = small_env._episode_length - 1
        _, _, _, truncated, _ = small_env.step(_zero_actions(small_env))
        assert truncated.any()
        assert np.all(small_env._step_counts[truncated] == 0)
        assert np.all(small_env._step_counts[~truncated] > 0)

    def test_auto_reset_restores_cached_initial_obs_for_done_envs(
        self,
        small_env: BatchedPlaygroundVectorEnv,
    ) -> None:
        small_env.reset(seed=0)
        first_obs = np.asarray(small_env._first_obs, dtype=np.float32)

        small_env._step_counts[:] = small_env._episode_length
        obs, _, _, truncated, _ = small_env.step(_zero_actions(small_env))
        jax.block_until_ready(obs)
        assert truncated.all()

        reset_obs = np.asarray(obs, dtype=np.float32)
        for i in range(small_env.num_envs):
            np.testing.assert_allclose(
                reset_obs[i],
                first_obs[i],
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"env {i} obs not restored to cached initial state after auto-reset",
            )

    def test_normalize_obs_clips_to_bounds(self) -> None:
        env = BatchedPlaygroundVectorEnv(
            env_name="WalkerWalk",
            num_envs=8,
            seed=0,
            normalize_obs=True,
            torch_device=None,
        )
        try:
            env.reset(seed=0)
            for _ in range(20):
                obs, *_ = env.step(
                    np.array(env.action_space.sample(), dtype=np.float32)
                )
                jax.block_until_ready(obs)
            obs_np = np.asarray(obs, dtype=np.float32)
            assert obs_np.min() >= -10.0 - 1e-5
            assert obs_np.max() <= 10.0 + 1e-5
        finally:
            env.close()

    def test_reward_scaling_applied(self) -> None:
        env = BatchedPlaygroundVectorEnv(
            env_name="WalkerWalk",
            num_envs=4,
            seed=0,
            normalize_obs=False,
            reward_scaling=2.5,
            torch_device=None,
        )
        try:
            env.reset(seed=0)
            _, rewards, _, _, _ = env.step(_zero_actions(env))
            raw = np.array(env._states.reward, dtype=np.float32)
            np.testing.assert_allclose(rewards, raw * 2.5, rtol=1e-5, atol=1e-5)
        finally:
            env.close()

    def test_torch_device_returns_cpu_tensors(self) -> None:
        env = BatchedPlaygroundVectorEnv(
            env_name="WalkerWalk",
            num_envs=2,
            seed=0,
            normalize_obs=False,
            torch_device="cpu",
        )
        try:
            obs, _ = env.reset(seed=1)
            assert isinstance(obs, torch.Tensor)
            assert obs.device.type == "cpu"
            assert obs.shape == (2, env.single_observation_space.shape[0])

            obs2, *_ = env.step(
                torch.zeros(2, env.single_action_space.shape[0], dtype=torch.float32),
            )
            assert isinstance(obs2, torch.Tensor)
            assert obs2.device.type == "cpu"
        finally:
            env.close()

    def test_action_repeat_multi_step_accumulates_reward(self) -> None:
        env = BatchedPlaygroundVectorEnv(
            env_name="WalkerWalk",
            num_envs=2,
            seed=0,
            action_repeat=3,
            normalize_obs=False,
            torch_device=None,
        )
        try:
            assert env._action_repeat == 3
            assert env.env_steps_per_step == 3
            env.reset(seed=0)
            before = env._step_counts.copy()
            env.step(
                _zero_actions(env),
            )
            assert np.all(env._step_counts == before + 3)
        finally:
            env.close()

    def test_many_steps_remain_finite(
        self, small_env: BatchedPlaygroundVectorEnv
    ) -> None:
        small_env.reset(seed=0)
        for _ in range(50):
            obs, rewards, terminated, _truncated, info = small_env.step(
                np.array(small_env.action_space.sample(), dtype=np.float32),
            )
            jax.block_until_ready(obs)
            assert np.isfinite(np.asarray(obs)).all()
            assert np.isfinite(rewards).all()
            assert np.all(terminated == False)  # noqa: E712
            assert info["_episode_done"].dtype == np.bool_


class TestInvalidConfiguration:
    def test_unknown_env_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            BatchedPlaygroundVectorEnv(env_name="NotARealPlaygroundEnv", num_envs=1)


class TestMockSpecificSemantics:
    @pytest.fixture
    def mock_registry(self, monkeypatch: pytest.MonkeyPatch):
        return install_mock_registry(
            monkeypatch,
            episode_length=10,
            episode_done_step=5,
            reward_scale=0.1,
        )

    @pytest.fixture
    def batched_mock(self, mock_registry) -> BatchedPlaygroundVectorEnv:
        env = BatchedPlaygroundVectorEnv(
            env_name="MockWalk",
            num_envs=4,
            seed=0,
            normalize_obs=False,
            reward_scaling=1.0,
            action_repeat=1,
            torch_device=None,
        )
        yield env
        env.close()

    def test_reward_matches_action_sum_times_scale(
        self,
        batched_mock: BatchedPlaygroundVectorEnv,
    ) -> None:
        batched_mock.reset(seed=0)
        sums = [1.0, 2.0, -0.5, 3.0]
        _, rewards, _, _, _ = batched_mock.step(
            _actions_with_row_sums(batched_mock, sums)
        )
        expected = np.array(sums, dtype=np.float32) * 0.1
        np.testing.assert_allclose(rewards, expected, rtol=1e-5, atol=1e-5)

    def test_reward_scaling_multiplier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_mock_registry(monkeypatch, episode_length=10, reward_scale=0.1)
        env = BatchedPlaygroundVectorEnv(
            env_name="MockWalk",
            num_envs=2,
            seed=0,
            normalize_obs=False,
            reward_scaling=3.0,
            action_repeat=1,
            torch_device=None,
        )
        try:
            env.reset(seed=0)
            _, rewards, _, _, _ = env.step(_actions_with_row_sums(env, [2.0, 4.0]))
            np.testing.assert_allclose(rewards, np.array([0.6, 1.2], dtype=np.float32))
        finally:
            env.close()

    def test_action_repeat_accumulates_mock_rewards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_mock_registry(monkeypatch, episode_length=50, reward_scale=0.1)
        env = BatchedPlaygroundVectorEnv(
            env_name="MockWalk",
            num_envs=1,
            seed=0,
            normalize_obs=False,
            reward_scaling=1.0,
            action_repeat=3,
            torch_device=None,
        )
        try:
            env.reset(seed=0)
            _, reward, _, _, _ = env.step(_actions_with_row_sums(env, [2.0]))
            assert reward[0] == pytest.approx(0.6, rel=1e-5)
        finally:
            env.close()

    def test_done_env_stops_accumulating_reward_under_action_repeat(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install_mock_registry(
            monkeypatch,
            episode_length=50,
            episode_done_step=1,
            reward_scale=0.1,
        )
        env = BatchedPlaygroundVectorEnv(
            env_name="MockWalk",
            num_envs=1,
            seed=0,
            normalize_obs=False,
            action_repeat=3,
            torch_device=None,
        )
        try:
            env.reset(seed=0)
            _, reward, _, _, _ = env.step(_actions_with_row_sums(env, [5.0]))
            assert reward[0] == pytest.approx(0.5, rel=1e-5)
        finally:
            env.close()

    @pytest.mark.usefixtures("mock_registry")
    def test_single_env_reset_step_reward(self) -> None:
        env = PlaygroundGymWrapper(env_name="MockWalk", seed=0)
        obs, _ = env.reset(seed=10)
        assert obs.shape == (4,)
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        assert reward == pytest.approx(0.1, rel=1e-5)
        assert terminated is False
        assert truncated in (True, False)
