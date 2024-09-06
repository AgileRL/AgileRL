"""
Author: Burak M Gonultas
https://github.com/gonultasbu
---
Original Reference:
https://github.com/Farama-Foundation/SuperSuit/issues/43#issuecomment-751792111
"""

import traceback
from multiprocessing import Pipe, Process

import numpy as np
from gymnasium.vector.utils import CloudpickleWrapper
from pettingzoo.utils.env import ParallelEnv


def worker(remote, parent_remote, env_fn_wrapper, enable_autoreset=True, env_args={}):
    """Worker class

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    parent_remote.close()
    env = env_fn_wrapper()

    autoreset = False
    if not isinstance(env, ParallelEnv):
        env = env.parallel_env(**env_args)

    while True:
        cmd, data = remote.recv()
        try:
            if cmd == "step":
                if autoreset:
                    obs, infos = env.reset()
                    reward = {agent_id: 0 for agent_id in env.agents}
                    dones = {agent_id: False for agent_id in env.agents}
                    truncs = {agent_id: False for agent_id in env.agents}
                else:
                    data = {
                        possible_agent: np.array(data[idx]).squeeze()
                        for idx, possible_agent in enumerate(env.possible_agents)
                    }
                    obs, reward, dones, truncs, infos = env.step(data)

                if enable_autoreset:
                    autoreset = all(
                        [
                            term | trunc
                            for term, trunc in zip(dones.values(), truncs.values())
                        ]
                    )

                # Deal with pettingzoo action masking
                # observation returned in the following format when action masking is used
                # obs = {"agent_0" : {"observation": 1, "action_mask": 0},...}
                try:
                    ob = {agent: obs[agent]["observation"] for agent in obs.keys()}
                    action_mask = {
                        agent: obs[agent]["action_mask"] for agent in obs.keys()
                    }
                    ob = list(ob.values())
                    action_mask = list(action_mask.values())
                except Exception:
                    ob = list(obs.values())
                    action_mask = None

                # Info returned in the following format when agent masking is used
                # info = {"agent_0": {"env_defined_actions":np.array([1,0,1])}, ...}
                try:
                    env_defined_actions = {
                        agent: infos[agent].pop("env_defined_actions")
                        for agent in infos.keys()
                    }
                    env_defined_actions = list(env_defined_actions.values())
                    info = list(infos.values())
                except Exception:
                    env_defined_actions = None
                    info = list(infos.values())

                reward = list(reward.values())
                dones = list(dones.values())
                truncs = list(truncs.values())
                remote.send(
                    (ob, reward, dones, truncs, info, action_mask, env_defined_actions)
                )
            elif cmd == "reset":
                ob, infos = env.reset(seed=data, options=None)
                ob = list(ob.values())
                infos = list(infos.values())
                remote.send((ob, infos))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "seed":
                env.seed(data)
            elif cmd == "render":
                env.render()
            else:
                raise NotImplementedError
        except Exception as e:
            if isinstance(e, NotImplementedError):
                raise NotImplementedError
            tb = traceback.format_exc()
            remote.send(("error", e, tb))


class VecEnv:
    """An abstract asynchronous, vectorized environment

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """

    def __init__(self, num_envs, possible_agents):
        self.num_envs = num_envs
        self.agents = possible_agents
        self.num_agents = len(self.agents)

    def reset(self, seed=None, options=None):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        passed_actions_list = [[] for _ in list(actions.values())[0]]
        for env_idx, _ in enumerate(list(actions.values())[0]):
            for possible_agent in self.agents:
                passed_actions_list[env_idx].append(actions[possible_agent][env_idx])
        self.step_async(passed_actions_list)
        step_wait = self.step_wait()
        return step_wait


class SubprocVecEnv(VecEnv):
    """Vectorized environment class that collects samples in parallel using subprocesses

    Args:
        env_fns (list): list of gym environments to run in subprocesses
        enable_autoreset: Boolean flag to enable autoreset when environment terminates or truncates
        env_args: Dictionary of environments arguments

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """

    def __init__(self, env_fns, enable_autoreset=True, env_args={}):
        env = env_fns[0]()
        if isinstance(env, ParallelEnv):
            self.env = env
        else:
            self.env = env.parallel_env(**env_args)
        self.waiting = False
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [
            Process(
                target=worker,
                args=(
                    work_remote,
                    remote,
                    CloudpickleWrapper(env_fn),
                    enable_autoreset,
                    env_args,
                ),
            )
            for idx, (work_remote, remote, env_fn) in enumerate(
                zip(self.work_remotes, self.remotes, env_fns)
            )
        ]
        for p in self.ps:
            p.daemon = (
                True  # If the main process crashes, we should not cause things to hang
            )
            p.start()
        VecEnv.__init__(
            self,
            len(env_fns),
            self.env.possible_agents,
        )

    # Note: this is not part of the PettingZoo API
    def seed(self, value):
        for i_remote, remote in enumerate(self.remotes):
            remote.send(("seed", value + i_remote))

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, truncs, infos, action_mask, env_defined_actions = (
            self.process_results(results, waiting_flag=False)
        )

        ret_obs_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }

        ret_action_mask_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }

        ret_rews_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_dones_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_truncs_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }

        ret_infos_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_env_def_act_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }

        for env_idx, _ in enumerate(obs):
            for agent_idx, possible_agent in enumerate(self.env.possible_agents):
                if action_mask[0] is not None:
                    ret_action_mask_dict[possible_agent].append(
                        action_mask[env_idx][agent_idx]
                    )
                ret_obs_dict[possible_agent].append(obs[env_idx][agent_idx])
                ret_rews_dict[possible_agent].append(rews[env_idx][agent_idx])
                ret_dones_dict[possible_agent].append(dones[env_idx][agent_idx])
                ret_truncs_dict[possible_agent].append(truncs[env_idx][agent_idx])
                if env_defined_actions[0] is not None:
                    if env_defined_actions[env_idx][agent_idx] is None:
                        if hasattr(self.env.action_space(possible_agent), "n"):
                            action_dim = self.action_space(possible_agent).n
                        else:
                            action_dim = self.action_space(possible_agent).shape[0]
                        env_defined_actions[env_idx][agent_idx] = np.zeros(
                            action_dim
                        ).fill(np.nan)
                    ret_env_def_act_dict[possible_agent].append(
                        env_defined_actions[env_idx][agent_idx]
                    )
                ret_infos_dict[possible_agent].append(infos[env_idx][agent_idx])

        for agent_idx, possible_agent in enumerate(self.env.possible_agents):
            for op_dict in [
                ret_obs_dict,
                ret_rews_dict,
                ret_dones_dict,
                ret_truncs_dict,
                ##
                ret_action_mask_dict,
                ret_env_def_act_dict,
            ]:

                # base case no masking
                if op_dict[possible_agent]:
                    op_dict[possible_agent] = np.stack(op_dict[possible_agent])

        # Deal with the infos dict
        vect_ret_infos_dict = {}
        for agent_idx, possible_agent in enumerate(self.env.possible_agents):
            merged_dict = {key: [] for key in ret_infos_dict[possible_agent][0].keys()}
            # Collect values for each key from all dictionaries
            for d in ret_infos_dict[possible_agent]:
                for key in d:
                    merged_dict[key].append(d[key])
            # Convert the lists of values into stacked NumPy arrays
            for key in merged_dict:
                merged_dict[key] = np.array(merged_dict[key])
            vect_ret_infos_dict[possible_agent] = merged_dict
        ret_infos_dict = vect_ret_infos_dict

        if action_mask[0] is not None:
            new_obs_dict = {}
            for possible_agent in self.env.possible_agents:
                new_obs_dict[possible_agent] = {}
                new_obs_dict[possible_agent]["observation"] = ret_obs_dict[
                    possible_agent
                ]
                new_obs_dict[possible_agent]["action_mask"] = ret_action_mask_dict[
                    possible_agent
                ]
            ret_obs_dict = new_obs_dict

        if env_defined_actions[0] is not None:
            new_info_dict = {}
            for possible_agent in self.env.possible_agents:
                new_info_dict[possible_agent] = ret_infos_dict[possible_agent]
                new_info_dict[possible_agent]["env_defined_actions"] = (
                    ret_env_def_act_dict[possible_agent]
                )
            ret_infos_dict = new_info_dict

        return (
            ret_obs_dict,
            ret_rews_dict,
            ret_dones_dict,
            ret_truncs_dict,
            ret_infos_dict,
        )

    def reset(self, seed=None, options=None):
        for remote in self.remotes:
            remote.send(("reset", seed))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = self.process_results(results, waiting_flag=self.waiting)
        ret_obs_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_infos_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        for env_idx, _ in enumerate(obs):
            for agent_idx, possible_agent in enumerate(self.env.possible_agents):
                ret_obs_dict[possible_agent].append(obs[env_idx][agent_idx])
                ret_infos_dict[possible_agent].append(infos[env_idx][agent_idx])
        for agent_idx, possible_agent in enumerate(self.env.possible_agents):
            for op_dict in [
                ret_obs_dict,
                ret_infos_dict,
            ]:
                op_dict[possible_agent] = np.stack(op_dict[possible_agent])
        return (ret_obs_dict, ret_infos_dict)

    def render(self):
        self.remotes[0].send(("render", None))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            results = [remote.recv() for remote in self.remotes]
            self.process_results(results, self.waiting)
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
            self.closed = True

    def sample_personas(self, is_train, is_val=True, path="./"):
        return self.env.sample_personas(is_train=is_train, is_val=is_val, path=path)

    def step(self, actions):
        return_val = super().step(actions)
        return return_val

    def process_results(self, results, waiting_flag):
        if len(results) == 1 and results[0] is None:
            return
        self.waiting = waiting_flag
        zipped_results = list(zip(*results))
        if "error" in zipped_results[0]:
            e = zipped_results[1][zipped_results[0].index("error")]
            tb = zipped_results[2][zipped_results[0].index("error")]
            raise Exception(f"{e} \nTraceback:\n{tb}")
        return zipped_results
