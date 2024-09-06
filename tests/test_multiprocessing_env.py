from multiprocessing import Pipe

import numpy as np
import pytest
from gymnasium.vector.utils import CloudpickleWrapper
from pettingzoo.mpe import simple_adversary_v3

from agilerl.utils.multiprocessing_env import VecEnv, worker
from tests.test_vectorization import parallel_env_disc


class DummyRecv:
    def __init__(self, cmd, data):
        self.call_count = 0
        self.cmd = cmd
        self.data = data

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count > 1:
            return "close", None
        else:
            return self.cmd, self.data


@pytest.mark.parametrize("enable_autoreset", [True, False])
def test_worker_custom(enable_autoreset):
    cmds = ["step", "reset", "seed", "render"]
    data_s = [
        np.array([0, 0]),
        None,
        None,
        None,
    ]

    for cmd, data in zip(cmds, data_s):
        env = parallel_env_disc()
        env.reset()

        env_fns = [lambda: env for _ in range(2)]

        remote, parent_remote = Pipe()
        env_fn_wrapper = CloudpickleWrapper(env_fns[0])

        def dummy_send(*args):
            remote.close()

        remote.send = dummy_send
        remote.recv = DummyRecv(cmd, data)

        worker(remote, parent_remote, env_fn_wrapper, enable_autoreset)


@pytest.mark.parametrize("enable_autoreset", [True, False])
def test_worker_pettingzoo(enable_autoreset):
    cmds = ["reset"]
    data_s = [
        None,
    ]

    for cmd, data in zip(cmds, data_s):
        env = simple_adversary_v3

        env_fns = [lambda: env for _ in range(2)]

        remote, parent_remote = Pipe()
        env_fn_wrapper = CloudpickleWrapper(env_fns[0])

        def dummy_send(*args):
            remote.close()

        remote.send = dummy_send
        remote.recv = DummyRecv(cmd, data)
        worker(remote, parent_remote, env_fn_wrapper, enable_autoreset)


def test_worker_cmd_not_implemented():
    cmd = ["Command that doesn't exist"]
    data = [
        None,
    ]

    env = parallel_env_disc()
    env.reset()

    env_fns = [lambda: env for _ in range(2)]

    remote, parent_remote = Pipe()
    env_fn_wrapper = CloudpickleWrapper(env_fns[0])

    def dummy_send(*args):
        remote.close()

    remote.send = dummy_send
    remote.recv = DummyRecv(cmd, data)

    with pytest.raises(NotImplementedError) as error:
        worker(remote, parent_remote, env_fn_wrapper)

        assert error is NotImplementedError


def test_vecenv():
    env = VecEnv(1, ["agent_0"])

    env.reset()
    env.step_async({"agent_0": [0]})
    env.step_wait()
    env.close()
    result = env.step({"agent_0": [0]})

    assert result is None
