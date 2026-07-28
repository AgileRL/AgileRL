# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Attribute a torch allocator snapshot to the estimator's components.

The sweep validates predicted *totals*: it compares one measured peak per
configuration against one predicted peak. That leaves the per-component
split unverified — the bars could each be wrong and still sum correctly.

A snapshot closes that gap, because every allocation carries the Python
stack that made it. This module rolls those stacks up into the same
component keys the estimator emits, so predicted and observed splits can be
compared line by line.

Two limits are structural, not incidental:

- **vLLM is invisible.** Its weights and KV pool come from CuMem, outside
  the torch allocator, so a snapshot covers the trainer only. The generation
  bar still needs NVML.
- **It is expensive.** Recording every allocation with stacks is far too
  heavy for a 21-point sweep, so this is for single configurations.

Usage::

    python -m agilerl.memory.profiling.harness --model <id> --snapshot t.pickle ...
    python -m agilerl.memory.profiling.snapshot t.pickle
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

GIB = 1024**3

#: Frame markers mapped to estimator component keys, most specific first.
#: A frame matches if the marker appears in its filename or function name.
ATTRIBUTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # vLLM first: its pool is registered with the torch allocator but is not
    # trainer memory, and after sleep it is not resident at all. Leaving it
    # in would swamp every other component.
    ("vllm_engine", ("vllm/", "cumem", "gpu_worker", "model_runner")),
    ("logits_workspace", ("fused_logprobs", "fused_loss", "logsumexp", "lm_head")),
    (
        "grads_optimizer",
        ("optimizer", "adamw", "accumulate_grad", "_engine.run_backward"),
    ),
    # Before the adapter rule: a forward through a LoRA-wrapped layer has
    # peft frames in its stack but is allocating activations, not adapter
    # weights. Ordering these first stops activations being booked as
    # adapters — measured at 1.24 GiB mis-attributed on one corner.
    ("activations", ("checkpoint", "decoder_layer", "attention", "mlp", "forward")),
    (
        "base_weights",
        ("from_pretrained", "load_state_dict", "modeling_", "_apply", "move_params"),
    ),
    ("adapters", ("lora", "peft")),
)


def classify(frames: list[dict]) -> str:
    """Assign one allocation to a component by its deepest matching frame."""
    for frame in frames:
        haystack = f"{frame.get('filename', '')}::{frame.get('name', '')}".lower()
        for component, markers in ATTRIBUTION_RULES:
            if any(marker in haystack for marker in markers):
                return component
    return "unattributed"


def summarise_final(path: Path) -> dict[str, int]:
    """Bytes per component still live when the snapshot was taken.

    Rarely what you want: by the time ``learn`` returns, activations are
    freed and the trainer's parameters are back on the host, so the
    interesting components read as zero.
    """
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)

    totals: dict[str, int] = defaultdict(int)
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            frames = block.get("frames") or []
            totals[classify(frames)] += int(block.get("size", 0))
    return dict(totals)


def summarise(path: Path) -> dict[str, int]:
    """Bytes per component at the moment of peak allocation.

    Replays the allocation event trace rather than reading the final block
    list, because the peak is what the estimator predicts and it has long
    passed by the time the snapshot is written.
    """
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)

    traces = snapshot.get("device_traces") or []
    events = max(traces, key=len) if traces else []
    if not events:
        return summarise_final(path)

    allocs = {"alloc", "segment_alloc"}
    frees = {"free_completed", "segment_free"}

    live: dict[int, int] = {}
    total = 0
    peak = 0
    peak_index = 0
    for index, event in enumerate(events):
        action = event.get("action")
        size = int(event.get("size", 0))
        if action in allocs:
            live[event.get("addr")] = size
            total += size
            if total > peak:
                peak, peak_index = total, index
        elif action in frees and event.get("addr") in live:
            total -= live.pop(event.get("addr"))

    totals: dict[str, int] = defaultdict(int)
    live_frames: dict[int, tuple[int, list]] = {}
    for event in events[: peak_index + 1]:
        action = event.get("action")
        if action in allocs:
            live_frames[event.get("addr")] = (
                int(event.get("size", 0)),
                event.get("frames") or [],
            )
        elif action in frees:
            live_frames.pop(event.get("addr"), None)
    for size, frames in live_frames.values():
        totals[classify(frames)] += size
    return dict(totals)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agilerl.memory.profiling.snapshot", description=__doc__
    )
    parser.add_argument("snapshot", help="Pickle written by --snapshot")
    args = parser.parse_args(argv)

    totals = summarise(Path(args.snapshot))
    if not totals:
        print("No live allocations found in snapshot.", file=sys.stderr)
        return 1
    grand = sum(totals.values())
    print(f"{'component':<22}{'GiB':>8}{'share':>9}")
    for component, size in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{component:<22}{size / GIB:>8.2f}{size / grand:>8.1%}")
    print(f"{'TOTAL (torch only)':<22}{grand / GIB:>8.2f}")
    print("\nvLLM's CuMem allocations are not represented here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
