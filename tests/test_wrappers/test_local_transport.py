"""In-process backend + dataset cursor: ``LocalEnvClient`` / ``RolloutEnv.local`` /
``RolloutEnv.from_spec`` (no HTTP) and the reusable ``BatchPointer``.
"""

from __future__ import annotations

from typing import Any

import torch

from agilerl.llm_envs import (
    BatchPointer,
    LocalEnvClient,
    OpenEnvClient,
    RolloutEnv,
)


class _DatasetEnv:
    """A tiny dataset-backed local env (the text contract), used in-process and by spec."""

    def __init__(self, n: int = 3, prefix: str = "row") -> None:
        self._n = n
        self._prefix = prefix
        self._q = ""
        self.tools: list[Any] = []

    @property
    def dataset_size(self) -> int:
        return self._n

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
        evaluation: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del seed
        idx = (row_index or 0) % self._n
        self._q = f"{'eval' if evaluation else self._prefix}{idx}"
        return self._q, {"suffix": "answer:"}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        return "", (1.0 if "x" in str(action) else 0.0), True, False, {}

    def close(self) -> None:
        pass


class _MiniTok:
    """Minimal tokenizer for the non-chat-template token path."""

    pad_token_id = 0

    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        del texts
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: Any, **_: Any) -> str:
        del ids
        return "x"

    def encode(self, text: str, **_: Any) -> list[int]:
        del text
        return [7, 7]


# --- LocalEnvClient: drive a local env directly, no HTTP -------------------
def test_local_env_client_drives_env_directly() -> None:
    """``LocalEnvClient`` calls the env in-process and folds the suffix like the server."""
    client = LocalEnvClient(_DatasetEnv())
    assert client.dataset_size == 3
    prompt, info = client.reset(row_index=1)
    assert prompt == "row1\nanswer:"  # suffix folded in-process (no OpenEnvWrapper)
    assert info == {}
    assert client.step("x marks it") == ("", 1.0, True, False, {})


def test_local_env_client_eval_mode_routes_split() -> None:
    """``eval_mode`` flags reset for the held-out split, then restores."""
    client = LocalEnvClient(_DatasetEnv())
    with client.eval_mode():
        assert client.reset(row_index=0)[0] == "eval0\nanswer:"
    assert client.reset(row_index=0)[0] == "row0\nanswer:"


def test_local_env_client_close_is_forwarded() -> None:
    """``close`` reaches the wrapped env when it has one."""
    closed = {"n": 0}

    class _Env:
        def reset(self, seed: int | None = None) -> tuple[str, dict]:
            del seed
            return "p", {}

        def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
            del action
            return "", 0.0, True, False, {}

        def close(self) -> None:
            closed["n"] += 1

    LocalEnvClient(_Env()).close()
    assert closed["n"] == 1


# --- RolloutEnv.local / from_spec ------------------------------------------
def test_rollout_env_local_uses_in_process_backend() -> None:
    """``RolloutEnv.local`` drives a local env with a ``LocalEnvClient`` (no server)."""
    env = RolloutEnv.local(
        _DatasetEnv(), _MiniTok(), max_turns=1, apply_chat_template=False
    )
    assert isinstance(env._backend, LocalEnvClient)
    assert env._owned_server is None
    obs, _ = env.reset(row_index=2)
    assert "input_ids" in obs
    assert env.dataset_size == 3
    env.close()


def test_from_spec_url_builds_http_client() -> None:
    """A URL spec routes to the HTTP ``OpenEnvClient`` backend."""
    env = RolloutEnv.from_spec("http://localhost:0", None, _MiniTok())
    assert isinstance(env._backend, OpenEnvClient)
    env.close()


def test_from_spec_entrypoint_runs_in_process() -> None:
    """A ``path.py:Class`` entrypoint loads + drives the env in-process (no HTTP)."""
    spec = f"{__file__}:_DatasetEnv"
    env = RolloutEnv.from_spec(
        spec, {"prefix": "Q"}, _MiniTok(), max_turns=1, apply_chat_template=False
    )
    assert isinstance(env._backend, LocalEnvClient)
    assert env._owned_server is None
    obs, _ = env.reset(row_index=0)
    assert "input_ids" in obs
    env.close()


# --- BatchPointer ----------------------------------------------------------
def test_batch_pointer_reshuffles_each_epoch() -> None:
    """Each epoch is a fresh full permutation of the rows."""
    pointer = BatchPointer(4, seed=0)
    rows = [pointer.next_row() for _ in range(8)]
    assert sorted(rows[:4]) == [0, 1, 2, 3]
    assert sorted(rows[4:]) == [0, 1, 2, 3]


def test_batch_pointer_assign_is_group_contiguous() -> None:
    """``assign`` gives one ``(seed, row)`` per slot, shared within each group."""
    assigned = BatchPointer(4, seed=0).assign(2, 3, base_seed=100)
    assert len(assigned) == 6
    assert assigned[0] == assigned[1] == assigned[2]  # group 0 shares prompt + seed
    assert assigned[3] == assigned[4] == assigned[5]  # group 1 shares
    assert assigned[0][0] == 100  # seed = base_seed + batch item 0
    assert assigned[3][0] == 101  # seed = base_seed + batch item 1


def test_batch_pointer_without_dataset_yields_no_rows() -> None:
    """A non-dataset env (size 0) gets ``row_index=None`` for every slot."""
    assert BatchPointer(0).assign(2, 2) == [(None, None)] * 4
