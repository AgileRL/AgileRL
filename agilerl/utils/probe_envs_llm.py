"""Lightweight string-observation probe environments for multi-turn LLM debugging.

These are ordered by increasing complexity for debugging LLM PPO training
pipelines: fixed target, input-dependent target, multi-digit arithmetic
target, then multi-turn spatial navigation.
"""

from __future__ import annotations

from random import Random

__all__ = [
    "ConditionalTargetEnv",
    "ConstantTargetEnv",
    "GridNavigationEnv",
    "MultiInputConditionalEnv",
]


class ConstantTargetEnv:
    """Single-turn probe with a fixed prompt and a fixed correct digit."""

    def __init__(
        self,
        target_digit: str = "3",
        prompt: str = "11",
        seed: int = 42,
    ) -> None:
        self.target_digit = target_digit
        self.prompt = prompt
        self.seed = seed
        self.max_turns = 1

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        if seed is not None:
            self.seed = seed
        return self.prompt, {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        reward = 1.0 if self.target_digit in str(action) else -1.0
        return "", reward, True, False, {}


class ConditionalTargetEnv:
    """Single-turn probe: observation is one digit; target is ``(digit % 3) + 1``."""

    def __init__(
        self,
        digits: tuple[int, ...] = (1, 2, 3),
        seed: int = 42,
    ) -> None:
        self.digits = digits
        self.rng = Random(seed)
        self.target: str = ""
        self.max_turns = 1

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        if seed is not None:
            self.rng = Random(seed)
        digit = self.rng.choice(self.digits)
        target_val = (int(digit) % 3) + 1
        self.target = str(target_val)
        return str(digit), {"target": self.target}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        reward = 1.0 if self.target in str(action) else -1.0
        return "", reward, True, False, {}


class MultiInputConditionalEnv:
    """Single-turn probe: two-digit prompt; target is ``(sum(digits) % 3) + 1``."""

    def __init__(
        self,
        digits: tuple[int, ...] = (1, 2, 3),
        seed: int = 42,
    ) -> None:
        self.digits = digits
        self.rng = Random(seed)
        self.target: str = ""
        self.max_turns = 1

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        if seed is not None:
            self.rng = Random(seed)
        d1 = self.rng.choice(self.digits)
        d2 = self.rng.choice(self.digits)
        total = int(d1) + int(d2)
        target_val = (total % 3) + 1
        self.target = str(target_val)
        return f"{d1}{d2}", {"target": self.target}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        reward = 1.0 if self.target in str(action) else -1.0
        return "", reward, True, False, {}


class GridNavigationEnv:
    """1D grid navigation: move left/right to reach a target position.

    The agent sees its starting position and target once (initial obs),
    then receives only its current position as feedback after each move.
    It must remember the target from context and navigate toward it.

    Actions map to: "1" = left, "2" = stay, "3" = right.
    Positions are clamped to [0, grid_size-1].
    """

    def __init__(
        self,
        grid_size: int = 4,
        max_turns: int = 5,
        step_cost: float = -0.1,
        seed: int = 42,
    ):
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.step_cost = step_cost
        self.rng = Random(seed)
        self.position = 0
        self.target = 0
        self.turn = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = Random(seed)
        self.position = self.rng.randint(0, self.grid_size - 1)
        others = [p for p in range(self.grid_size) if p != self.position]
        self.target = self.rng.choice(others)
        self.turn = 0
        obs = f"{self.position}{self.target}"
        return obs, {"position": self.position, "target": self.target}

    def step(self, action):
        self.turn += 1

        move = None
        for ch in str(action):
            if ch in "123":
                move = int(ch)
                break

        if move == 1:
            self.position = max(0, self.position - 1)
        elif move == 3:
            self.position = min(self.grid_size - 1, self.position + 1)

        obs = str(self.position)

        if self.position == self.target:
            return obs, 1.0, True, False, {"success": True}
        if self.turn >= self.max_turns:
            return obs, -1.0, True, False, {"success": False}
        return obs, self.step_cost, False, False, {}
