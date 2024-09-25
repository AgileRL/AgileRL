import multiprocessing as mp
import time

import numpy as np


def create_shared_memory(n_envs, ctx):
    x = ctx.Array("l", size_or_initializer=n_envs)
    return x


def _async_worker(index, pipe, parent_pipe, shared_memory, t):

    parent_pipe.close()
    cmd = pipe.recv()

    while True:

        for i in range(cmd):
            write_to_shared_memory(index, shared_memory)
            # print(f"worker-{index}", f"Step: {i}", "---->", time.time_ns()- t)
        break
    pipe.send("done")


def write_to_shared_memory(index, shared_memory):
    destination = np.frombuffer(shared_memory.get_obj(), dtype=int)
    destination[index] = index


def main():
    independent = True
    num_envs = 16
    ctx = mp.get_context()
    buffer = create_shared_memory(num_envs, ctx)
    parent_pipes, processes = [], []
    target = _async_worker
    t = time.time_ns()
    for idx in range(num_envs):
        parent_pipe, child_pipe = ctx.Pipe()
        if independent:
            buffer = create_shared_memory(num_envs, ctx)
        process = ctx.Process(
            target=target,
            args=(idx, child_pipe, parent_pipe, buffer, t),
        )

        parent_pipes.append(parent_pipe)
        processes.append(process)

        process.daemon = True
        process.start()
        child_pipe.close()

    t = time.time_ns()
    for p in parent_pipes:
        p.send(100)

    for p in parent_pipes:
        p.recv()

    print("PARETN TIMING", time.time_ns() - t)
    time.sleep(5)


if __name__ == "__main__":
    main()
