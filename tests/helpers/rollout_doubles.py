"""Shared mixin for ``RolloutEnv`` test doubles driven by ``BatchRolloutEnv``.

``BatchRolloutEnv`` drives its envs through the phased interface
(``_reset_fetch`` / ``_reset_apply`` and ``_step_prepare`` / ``_step_env`` /
``_step_apply``) so it can overlap the backend round-trips across envs. A double
only needs to implement plain ``reset()`` / ``step()``; this mixin maps the
phased interface onto them so the double drives the real (concurrent) collector
path without reimplementing the split.
"""

from __future__ import annotations

import inspect
from typing import Any


class RolloutEnvDoubleMixin:
    """Map ``BatchRolloutEnv``'s phased reset/step onto a double's reset()/step()."""

    def _reset_fetch(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> Any:
        params = inspect.signature(self.reset).parameters
        kwargs = {"row_index": row_index} if "row_index" in params else {}
        self._pending_reset = self.reset(seed=seed, **kwargs)
        return self._pending_reset

    def _reset_apply(
        self, obs_text: Any, info: Any, *, row_index: int | None = None
    ) -> Any:
        del obs_text, info, row_index
        return self._pending_reset

    def _step_prepare(self, full_completion: Any, sampling_logps: Any = None) -> str:
        self._pending_step = (full_completion, sampling_logps)
        return ""

    def _step_env(self, gen_text: str) -> Any:
        del gen_text
        return None

    def _step_apply(self, env_result: Any) -> Any:
        del env_result
        full_completion, sampling_logps = self._pending_step
        return self.step(full_completion, sampling_logps=sampling_logps)
