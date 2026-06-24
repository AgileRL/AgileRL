"""Shared helpers for algorithm coverage tests."""

from typing import Any

import agilerl.utils.algo_utils as algo_utils


class TransposeImageObservationSpy:
    """Records calls to ``transpose_image_observation`` while delegating."""

    def __init__(self, original: Any) -> None:
        self._original = original
        self.call_count = 0
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        self.calls.append((args, kwargs))
        return self._original(*args, **kwargs)


def patch_transpose_image_observation(monkeypatch: Any) -> TransposeImageObservationSpy:
    """Patch ``transpose_image_observation`` in algo_utils and return a call spy."""
    spy = TransposeImageObservationSpy(algo_utils.transpose_image_observation)
    monkeypatch.setattr(algo_utils, "transpose_image_observation", spy)
    return spy


def assert_transpose_image_observation_called(
    spy: TransposeImageObservationSpy, min_calls: int = 1
) -> None:
    """Assert ``transpose_image_observation`` was invoked during preprocessing."""
    assert spy.call_count >= min_calls, (
        f"Expected transpose_image_observation to be called at least {min_calls} "
        f"time(s), but it was called {spy.call_count} time(s)"
    )
