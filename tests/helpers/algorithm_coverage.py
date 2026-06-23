"""Shared helpers for algorithm coverage tests."""

from typing import Any


class ObsChannelsSpy:
    """Records calls to ``obs_channels_to_first`` while passing observations through."""

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        self.calls.append((args, kwargs))
        return args[0]


def patch_obs_channels_to_first(monkeypatch: Any, module_path: str) -> ObsChannelsSpy:
    """Patch ``obs_channels_to_first`` in *module_path* and return a call spy."""
    spy = ObsChannelsSpy()
    monkeypatch.setattr(f"{module_path}.obs_channels_to_first", spy)
    return spy


def assert_swap_channels_called(spy: ObsChannelsSpy, min_calls: int = 1) -> None:
    """Assert ``obs_channels_to_first`` was invoked when ``swap_channels=True``."""
    assert spy.call_count >= min_calls, (
        f"Expected obs_channels_to_first to be called at least {min_calls} time(s), "
        f"but it was called {spy.call_count} time(s)"
    )
