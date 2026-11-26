# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The asyncio event loop every verifiers env in a process runs on."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Coroutine

__all__ = ["SHARED_LOOP", "EventLoopThread"]

T = TypeVar("T")


class EventLoopThread:
    """An asyncio event loop on a daemon thread, started on first use."""

    def __init__(self) -> None:
        """Create the loop; the thread starts on first use."""
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()
        self._started = False
        self._failure: str | None = None
        self._waiting: set[concurrent.futures.Future[Any]] = set()

    def call(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the loop and wait for its result."""
        return self._submit(coro).result()

    async def acall(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the loop and await its result without holding a thread."""
        return await asyncio.wrap_future(self._submit(coro))

    def _submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        """Schedule ``coro`` on the loop; its future fails if the loop dies.

        :raises RuntimeError: If the loop has already died.
        """
        with self._lock:
            if self._failure is not None:
                coro.close()
                raise RuntimeError(self._failure)
            if not self._started:
                threading.Thread(
                    target=self._run, name="verifiers-env", daemon=True
                ).start()
                self._started = True
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            self._waiting.add(future)
        future.add_done_callback(self._forget)
        return future

    def _forget(self, future: concurrent.futures.Future[Any]) -> None:
        """Stop tracking a finished call."""
        with self._lock:
            self._waiting.discard(future)

    def _retrieve_task_exceptions(self) -> None:
        """Read stored task exceptions so asyncio does not log them at collection."""
        # SystemExit aborts the loop before done callbacks run, so the task
        # stays tracked and the exception stays stored.
        module = asyncio.tasks
        done: list[asyncio.Task[Any]] = []
        for name in ("_scheduled_tasks", "_all_tasks", "_eager_tasks"):
            group = getattr(module, name, None)
            if group is None:
                continue
            done.extend(
                task
                for task in list(group)
                if task.get_loop() is self._loop
                and task.done()
                and not task.cancelled()
            )
        for task in done:
            task.exception()

    def _run(self) -> None:
        """Run the loop, failing every waiting call if it dies."""
        try:
            self._loop.run_forever()
        except BaseException as err:
            # verifiers raises SystemExit from its own loop callbacks (missing
            # Prime auth), which stops the loop under every waiter.
            self._retrieve_task_exceptions()
            failure = f"verifiers stopped its event loop: {err}"
            with self._lock:
                self._failure = failure
                waiting = list(self._waiting)
            for future in waiting:
                if future.set_running_or_notify_cancel():
                    future.set_exception(RuntimeError(failure))
            # SystemExit stops this thread. Waiters already hold the failure.
            if not isinstance(err, SystemExit):
                raise


# verifiers keeps loop-bound asyncio state process-wide, so every env in a process
# shares one loop. It lives in this module, imported by name, because the env host
# loads verifiers_env.py by file path once per session.
SHARED_LOOP = EventLoopThread()
