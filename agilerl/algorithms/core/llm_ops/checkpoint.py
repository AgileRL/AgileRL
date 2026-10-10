# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Host copies of LLM checkpoints that write while training continues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import torch

from agilerl.utils.llm_utils import (
    CheckpointFileOpener,
    LoraAdapterSnapshot,
    directory_opener,
    write_lora_adapters,
)


@dataclass
class LLMCheckpointSnapshot:
    """Host copy of an LLM checkpoint, taken by :meth:`LLMAlgorithm.snapshot_checkpoint`.

    Shares no storage with live weights or optimizer state, and :meth:`write`
    runs no collectives, so it can be written while training continues.
    Ranks other than the main process hold an empty snapshot.

    :param attributes: ``attributes.pt`` payload; ``None`` off the main process.
    :type attributes: dict[str, Any] | None
    :param adapters: Adapter weights for a ``lora_only`` checkpoint.
    :type adapters: LoraAdapterSnapshot | None
    """

    attributes: dict[str, Any] | None
    adapters: LoraAdapterSnapshot | None = None

    def write(self, path: str | Path) -> None:
        """Write the checkpoint directory that :meth:`LLMAlgorithm.load_checkpoint` reads.

        :param path: Directory to write the checkpoint into.
        :type path: str | Path
        """
        self.write_files(directory_opener(path))

    def write_files(self, open_file: CheckpointFileOpener) -> None:
        """Write each checkpoint file through ``open_file``.

        ``attributes.pt`` goes last, so a reader that finds it finds every
        other file of the checkpoint.

        :param open_file: Opens a file by its path relative to the checkpoint directory.
        :type open_file: CheckpointFileOpener
        """
        if self.attributes is None:
            return
        if self.adapters is not None:
            write_lora_adapters(self.adapters, open_file)
        with open_file("attributes.pt") as file:
            torch.save(self.attributes, file, pickle_module=dill)
