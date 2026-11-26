# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Hashable key for :class:`agilerl.arena.memory.specs.TrainingSettings`."""

from __future__ import annotations

from dataclasses import dataclass, field

from agilerl.arena.memory.specs import TrainingSettings


@dataclass(frozen=True)
class TrainingSettingsKey:
    """Hashable stand-in for :class:`TrainingSettings`, keyed on its field values.

    ``fsdp`` is a mutable dataclass, so it enters the key as a snapshot of its
    field values, ``parallel_pins`` included.
    """

    values: tuple[object, ...]
    settings: TrainingSettings = field(compare=False)

    @classmethod
    def of(cls, settings: TrainingSettings) -> TrainingSettingsKey:
        """Key ``settings`` by every field value as it is now.

        :param settings: Training settings.
        :return: The key, carrying ``settings``.
        """
        fsdp = settings.fsdp
        values = tuple(
            tuple(vars(fsdp).values()) if name == "fsdp" and fsdp is not None else value
            for name, value in settings.__dict__.items()
        )
        return cls(values, settings)
