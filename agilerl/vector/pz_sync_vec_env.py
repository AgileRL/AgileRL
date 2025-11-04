from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from agilerl.typing import NumpyObsType
from agilerl.vector.pz_async_vec_env import process_transition
from agilerl.vector.pz_vec_env import PettingZooVecEnv


class SyncPettingZooVecEnv(PettingZooVecEnv):
    """Sync vectorized PettingZoo environment that runs envs serially.

    Matches the return format of AsyncPettingZooVecEnv: batched dict observations
    and dicts of arrays for rewards/terminated/truncated plus aggregated infos.

    :param env_fns: List of callables creating individual ParallelEnv instances
    :type env_fns: list[Callable[[], ParallelEnv]]
    """

    def __init__(self, env_fns: list[Callable[[], ParallelEnv]]):
        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.envs = [env_fn() for env_fn in env_fns]

        # Infer static attributes from a dummy env
        dummy_env = self.envs[0]
        self.metadata = (
            dummy_env.metadata
            if hasattr(dummy_env, "metadata")
            else dummy_env.unwrapped.metadata
        )
        self.render_mode = (
            dummy_env.render_mode
            if hasattr(dummy_env, "render_mode")
            else dummy_env.unwrapped.render_mode
        )
        self.possible_agents = dummy_env.possible_agents
        self.agents = dummy_env.possible_agents[:]
        action_spaces = {agent: dummy_env.action_space(agent) for agent in self.agents}
        observation_spaces = {
            agent: dummy_env.observation_space(agent) for agent in self.agents
        }

        super().__init__(
            num_envs=self.num_envs,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            possible_agents=self.agents,
        )

        self._pending_actions: list[dict[str, Any]] | None = None

    def close(self) -> None:
        if getattr(self, "closed", False):
            return
        for env in self.envs:
            env.close()
        self.closed = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NumpyObsType], dict[str, Any]]:
        # Distribute seeds like the async version
        if seed is None:
            seeds = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            raise AssertionError("seed must be int or None")

        per_env_obs: list[dict[str, NumpyObsType]] = []
        infos_accum: dict[str, Any] = {}

        for i, (env, env_seed) in enumerate(zip(self.envs, seeds)):
            obs_i, info_i = env.reset(seed=env_seed, options=options)
            # Ensure all agents present, fill placeholders like async does
            obs_i_proc, info_i_proc = process_transition(
                (obs_i, info_i),
                self._single_observation_spaces,
                ["observation", "info"],
                self.agents,
            )
            per_env_obs.append(obs_i_proc)
            infos_accum = self._add_info(infos_accum, info_i_proc, i)

        batched_obs = self._batch_observations(
            per_env_obs, self._single_observation_spaces
        )
        return batched_obs, infos_accum

    def step_async(self, actions: list[dict[str, Any]]) -> None:
        self._pending_actions = actions

    def step_wait(
        self,
    ) -> tuple[
        dict[str, NumpyObsType],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, Any],
    ]:
        assert self._pending_actions is not None, "step_wait called before step_async"

        per_env_obs: list[dict[str, NumpyObsType]] = []
        rewards_accum: dict[str, list[Any]] = {agent: [] for agent in self.agents}
        terms_accum: dict[str, list[bool]] = {agent: [] for agent in self.agents}
        truncs_accum: dict[str, list[bool]] = {agent: [] for agent in self.agents}
        infos_accum: dict[str, Any] = {}

        for env_idx, (env, action_dict) in enumerate(
            zip(self.envs, self._pending_actions)
        ):
            # Normalize action values similarly to async worker
            normalized_actions = {
                agent: (np.array(a).squeeze() if not isinstance(a, int) else a)
                for agent, a in action_dict.items()
            }

            obs, reward, terminated, truncated, info = env.step(normalized_actions)

            # Auto-reset finished envs to keep vector alive (matching async worker)
            if all(
                [(bool(terminated[a]) | bool(truncated[a])) for a in terminated.keys()]
            ):
                obs, info = env.reset()

            # Fill placeholders to ensure all agents present
            obs_p, rew_p, term_p, trunc_p, info_p = process_transition(
                (obs, reward, terminated, truncated, info),
                self._single_observation_spaces,
                ["observation", "reward", "terminated", "truncated", "info"],
                self.agents,
            )

            per_env_obs.append(obs_p)
            for agent in self.agents:
                rewards_accum[agent].append(rew_p[agent])
                terms_accum[agent].append(bool(term_p[agent]))
                truncs_accum[agent].append(bool(trunc_p[agent]))
            infos_accum = self._add_info(infos_accum, info_p, env_idx)

        batched_obs = self._batch_observations(
            per_env_obs, self._single_observation_spaces
        )
        rewards = {agent: np.asarray(vals) for agent, vals in rewards_accum.items()}
        terminateds = {
            agent: np.asarray(vals, dtype=np.bool_)
            for agent, vals in terms_accum.items()
        }
        truncations = {
            agent: np.asarray(vals, dtype=np.bool_)
            for agent, vals in truncs_accum.items()
        }

        self._pending_actions = None
        return batched_obs, rewards, terminateds, truncations, infos_accum

    # Helpers
    def _batch_observations(
        self,
        per_env_obs: list[dict[str, NumpyObsType]],
        obs_spaces: dict[str, spaces.Space],
    ) -> dict[str, NumpyObsType]:
        result: dict[str, NumpyObsType] = {}
        for agent, space in obs_spaces.items():
            if isinstance(space, spaces.Dict):
                batched = {}
                for key, subspace in space.spaces.items():
                    values = [obs[agent][key] for obs in per_env_obs]
                    batched[key] = self._stack_to_space(values, subspace)
                result[agent] = batched
            elif isinstance(space, spaces.Tuple):
                sub_values = []
                for idx, subspace in enumerate(space.spaces):
                    values = [obs[agent][idx] for obs in per_env_obs]
                    sub_values.append(self._stack_to_space(values, subspace))
                result[agent] = tuple(sub_values)
            else:
                values = [obs[agent] for obs in per_env_obs]
                result[agent] = self._stack_to_space(values, space)
        return result

    @staticmethod
    def _stack_to_space(values: list[np.ndarray], space: spaces.Space) -> np.ndarray:
        # Reshape scalar spaces to (num_envs, 1) like async reshape_observation
        arr = np.asarray(values)
        if space.shape == ():
            arr = arr.reshape((len(values), 1)).astype(space.dtype)
        else:
            arr = np.stack([np.asarray(v, dtype=space.dtype) for v in values], axis=0)
        return arr

    def _add_info(
        self, vector_infos: dict[str, Any], env_info: dict[str, Any], env_num: int
    ) -> dict[str, Any]:
        # Copied behavior from async env aggregator
        for key, value in env_info.items():
            if isinstance(value, dict):
                array = self._add_info(vector_infos.get(key, {}), value, env_num)
            else:
                if key not in vector_infos:
                    if type(value) in [int, float, bool] or issubclass(
                        type(value), np.number
                    ):
                        array = np.zeros(self.num_envs, dtype=type(value))
                    elif isinstance(value, np.ndarray):
                        array = np.zeros(
                            (self.num_envs, *value.shape), dtype=value.dtype
                        )
                    elif value is None:
                        array = np.full(
                            self.num_envs, fill_value=np.nan, dtype=np.float32
                        )
                    else:
                        array = np.full(self.num_envs, fill_value=None, dtype=object)
                else:
                    array = vector_infos[key]

                array[env_num] = value

            array_mask = vector_infos.get(
                f"_{key}", np.zeros(self.num_envs, dtype=np.bool_)
            )
            array_mask[env_num] = True

            vector_infos[key], vector_infos[f"_{key}"] = array, array_mask

        return vector_infos
