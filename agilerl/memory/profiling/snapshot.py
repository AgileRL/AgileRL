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
    ("logits_workspace", ("fused_logprobs", "fused_loss", "logsumexp", "lm_head")),
    ("adapters", ("lora", "peft")),
    ("grads_optimizer", ("optimizer", "adamw", "backward", "accumulate_grad")),
    ("activations", ("checkpoint", "decoder_layer", "attention", "mlp", "forward")),
    ("base_weights", ("from_pretrained", "load_state_dict", "_load", "to(")),
)


def classify(frames: list[dict]) -> str:
    """Assign one allocation to a component by its deepest matching frame."""
    for frame in frames:
        haystack = f"{frame.get('filename', '')}::{frame.get('name', '')}".lower()
        for component, markers in ATTRIBUTION_RULES:
            if any(marker in haystack for marker in markers):
                return component
    return "unattributed"


def summarise(path: Path) -> dict[str, int]:
    """Bytes of live allocation per component at the snapshot's peak."""
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
